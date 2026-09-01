#!/usr/bin/env python3
"""Run the frozen v16 HSC control-architecture scGPT value-ablation test."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("PN_ROOT", REPO_ROOT / "data/external_runtime"))
MODEL_DIR = ROOT / "00_raw/models_v15/scgpt_whole_human"
V16 = Path(os.environ.get("V16_ROOT", REPO_ROOT / "work/v16_runtime"))
INPUT_DIR = Path(os.environ.get("SCGPT_INPUT_DIR", V16 / "05_perturbation_inputs"))
OUTPUT = Path(os.environ.get("SCGPT_OUTPUT_DIR", V16 / "02_results/S7_hsc_control_perturbation/scgpt_primary"))
SEED = 20260710
CELL_CAP_PER_DISEASE_DONOR = 200
MIN_TARGET_CELLS = int(os.environ.get("MIN_TARGET_CELLS", "50"))
MIN_DONORS = int(os.environ.get("MIN_DONORS", "3"))
CONTROL_CODETECTION_FLOOR = float(os.environ.get("CONTROL_CODETECTION_FLOOR", "0"))


def build_model(vocab: GeneVocab, args: dict[str, object]) -> TransformerModel:
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=int(args["embsize"]),
        nhead=int(args["nheads"]),
        d_hid=int(args["d_hid"]),
        nlayers=int(args["nlayers"]),
        nlayers_cls=int(args.get("n_layers_cls", 3)),
        n_cls=1,
        vocab=vocab,
        dropout=float(args.get("dropout", 0.2)),
        pad_token=str(args.get("pad_token", "<pad>")),
        pad_value=int(args.get("pad_value", -2)),
        do_mvc=bool(args.get("MVC", True)),
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=str(args.get("input_emb_style", "continuous")),
        n_input_bins=int(args.get("n_bins", 51)),
        cell_emb_style="cls",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )
    raw = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    rules = {
        r"self_attn\._impl\.Wqkv\.": "self_attn.in_proj_",
        r"self_attn\.Wqkv\.": "self_attn.in_proj_",
        r"self_attn\._impl\.out_proj\.": "self_attn.out_proj.",
    }
    renamed = {}
    for key, value in raw.items():
        for pattern, replacement in rules.items():
            key = re.sub(pattern, replacement, key)
        renamed[key] = value
    state = model.state_dict()
    compatible = {
        key: value
        for key, value in renamed.items()
        if key in state and value.shape == state[key].shape
    }
    checkpoint_numel = sum(value.numel() for value in renamed.values())
    coverage = sum(value.numel() for value in compatible.values()) / checkpoint_numel
    if coverage < 0.99:
        raise RuntimeError(f"Checkpoint parameter coverage is {coverage:.4f}, expected >=0.99")
    state.update(compatible)
    model.load_state_dict(state)
    return model


def select_cells(metadata: pd.DataFrame, state_index: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + state_index)
    selected: list[int] = []
    for _, positions in metadata.groupby(["disease", "donor"], observed=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        if len(positions) > CELL_CAP_PER_DISEASE_DONOR:
            positions = rng.choice(positions, CELL_CAP_PER_DISEASE_DONOR, replace=False)
        selected.extend(positions.tolist())
    return np.asarray(sorted(selected), dtype=int)


def preprocess_counts(counts: sparse.csr_matrix, n_bins: int) -> np.ndarray:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    normalized = counts.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    dense = normalized.toarray().astype(np.float32)
    np.random.seed(SEED)
    return np.vstack([binning(row, n_bins=n_bins) for row in dense]).astype(np.float32)


def choose_controls(
    counts: sparse.csr_matrix,
    genes: pd.Index,
    metadata: pd.DataFrame,
    targets: list[str],
    excluded: set[str],
) -> pd.DataFrame:
    cirrhotic = metadata["disease"].eq("cirrhotic").to_numpy()
    matrix = counts[cirrhotic]
    means = np.asarray(matrix.mean(axis=0)).ravel()
    detection = np.asarray((matrix > 0).mean(axis=0)).ravel()
    index = {gene: i for i, gene in enumerate(genes)}
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for target in targets:
        target_index = index[target]
        target_cells = np.asarray(matrix[:, target_index].toarray()).ravel() > 0
        pool = [
            gene
            for gene in genes
            if gene not in excluded
            and gene not in targets
            and gene not in selected
            and means[index[gene]] > 0
            and not gene.startswith(("MT-", "RPS", "RPL"))
        ]
        distances: dict[str, float] = {}
        codetection: dict[str, float] = {}
        for gene in pool:
            gene_index = index[gene]
            co = (
                float((np.asarray(matrix[target_cells, gene_index].toarray()).ravel() > 0).mean())
                if target_cells.sum() > 0
                else 0.0
            )
            codetection[gene] = co
            distances[gene] = (
                abs(np.log1p(means[gene_index]) - np.log1p(means[target_index]))
                + abs(detection[gene_index] - detection[target_index])
                + 0.5 * (1 - co)
            )
        eligible = [gene for gene in pool if codetection[gene] >= CONTROL_CODETECTION_FLOOR]
        if not eligible:
            eligible = pool
        control = min(eligible, key=distances.get)
        selected.append(control)
        rows.append(
            {
                "target": target,
                "control": control,
                "matching_distance": distances[control],
                "cirrhotic_codetection": codetection[control],
                "target_mean_count": means[target_index],
                "control_mean_count": means[index[control]],
                "target_detection": detection[target_index],
                "control_detection": detection[index[control]],
                "control_codetection_floor": CONTROL_CODETECTION_FLOOR,
            }
        )
    return pd.DataFrame(rows)


def infer_batches(
    model: TransformerModel,
    genes: torch.Tensor,
    values: torch.Tensor,
    pad_id: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for start in range(0, len(genes), batch_size):
        end = min(start + batch_size, len(genes))
        batch_genes = genes[start:end].to(device, non_blocking=True)
        batch_values = values[start:end].to(device, non_blocking=True)
        padding = batch_genes.eq(pad_id)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            output = model(batch_genes, batch_values, padding, MVC=False)
        embeddings.append(output["cell_emb"].float().cpu().numpy())
        predictions.append(output["mlm_output"].float().cpu().numpy())
    return np.vstack(embeddings), np.vstack(predictions)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.mean(rng.choice(values, (5000, len(values)), replace=True), axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def run_state(
    state_slug: str,
    state_index: int,
    model: TransformerModel,
    vocab: GeneVocab,
    args: dict[str, object],
    device: torch.device,
    fra_weights: pd.Series,
    excluded: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_dir = INPUT_DIR / state_slug
    counts = sparse.load_npz(state_dir / "scgpt_raw_counts_cells_by_genes.npz").tocsr()
    genes = pd.Index(pd.read_csv(state_dir / "scgpt_genes.tsv", header=None)[0].astype(str))
    metadata = pd.read_csv(state_dir / "scgpt_cells.tsv", sep="\t")
    selected = select_cells(metadata, state_index)
    counts = counts[selected]
    metadata = metadata.iloc[selected].reset_index(drop=True)

    vocabulary = set(vocab.get_stoi())
    keep = genes.isin(vocabulary)
    counts = counts[:, keep]
    genes = genes[keep]
    targets_table = pd.read_csv(INPUT_DIR / "perturbation_targets.tsv", sep="\t")
    targets = targets_table.loc[targets_table["state_slug"].eq(state_slug), "target"].unique().tolist()
    if not set(targets).issubset(genes):
        raise RuntimeError(f"Targets absent after vocab filtering in {state_slug}: {set(targets)-set(genes)}")
    controls = choose_controls(counts, genes, metadata, targets, excluded)

    binned = preprocess_counts(counts, int(args["n_bins"]))
    gene_ids = np.asarray([vocab[gene] for gene in genes], dtype=np.int64)
    np.random.seed(SEED + state_index)
    tokenized = tokenize_and_pad_batch(
        binned,
        gene_ids,
        max_len=int(args["max_seq_len"]),
        vocab=vocab,
        pad_token=str(args["pad_token"]),
        pad_value=int(args["pad_value"]),
        append_cls=True,
        include_zero_gene=False,
    )
    token_genes = tokenized["genes"]
    token_values = tokenized["values"]
    batch_size = int(os.environ.get("SCGPT_BATCH_SIZE", "8"))
    baseline_embedding, baseline_prediction = infer_batches(
        model, token_genes, token_values, vocab[args["pad_token"]], device, batch_size
    )
    healthy = metadata["disease"].eq("healthy").to_numpy()
    cirrhotic = metadata["disease"].eq("cirrhotic").to_numpy()
    healthy_centroid = baseline_embedding[healthy].mean(axis=0)
    before_distance = np.linalg.norm(baseline_embedding - healthy_centroid, axis=1)

    fra_lookup = np.zeros(len(vocab), dtype=np.float32)
    for gene, weight in fra_weights.items():
        if gene in vocab:
            fra_lookup[vocab[gene]] = weight
    fra_position_weights = fra_lookup[token_genes.numpy()]
    cell_rows: list[pd.DataFrame] = []

    for pair_index, pair in controls.iterrows():
        target, control = pair["target"], pair["control"]
        target_present = token_genes.eq(vocab[target]) & token_values.gt(0)
        control_present = token_genes.eq(vocab[control]) & token_values.gt(0)
        usable = cirrhotic & target_present.any(dim=1).numpy() & control_present.any(dim=1).numpy()
        usable_index = np.flatnonzero(usable)
        usable_donors = metadata.loc[usable_index, "donor"].nunique()
        if len(usable_index) < MIN_TARGET_CELLS or usable_donors < MIN_DONORS:
            cell_rows.append(
                pd.DataFrame(
                    {
                        "state_slug": [state_slug],
                        "target": [target],
                        "matched_control": [control],
                        "status": ["insufficient_joint_token_coverage"],
                        "n_cells": [len(usable_index)],
                        "n_donors": [usable_donors],
                    }
                )
            )
            continue

        perturb_metrics = {}
        for role, gene in (("target", target), ("matched_control", control)):
            values = token_values[usable_index].clone()
            gene_positions = token_genes[usable_index].eq(vocab[gene])
            values[gene_positions] = 0.0
            embedding, prediction = infer_batches(
                model,
                token_genes[usable_index],
                values,
                vocab[args["pad_token"]],
                device,
                batch_size,
            )
            after_distance = np.linalg.norm(embedding - healthy_centroid, axis=1)
            embedding_rescue = (before_distance[usable_index] - after_distance) / np.maximum(
                before_distance[usable_index], 1e-8
            )
            prediction_delta = prediction - baseline_prediction[usable_index]
            weights = fra_position_weights[usable_index]
            fra_rescue = -np.divide(
                (prediction_delta * weights).sum(axis=1),
                weights.sum(axis=1),
                out=np.full(len(usable_index), np.nan),
                where=weights.sum(axis=1) > 0,
            )
            perturb_metrics[role] = (embedding_rescue, fra_rescue)

        frame = metadata.loc[usable_index, ["cell", "donor"]].reset_index(drop=True)
        frame.insert(0, "state_slug", state_slug)
        frame["target"] = target
        frame["matched_control"] = control
        frame["status"] = "completed"
        frame["n_cells"] = len(usable_index)
        frame["n_donors"] = usable_donors
        frame["target_embedding_rescue"] = perturb_metrics["target"][0]
        frame["control_embedding_rescue"] = perturb_metrics["matched_control"][0]
        frame["adjusted_embedding_rescue"] = (
            frame["target_embedding_rescue"] - frame["control_embedding_rescue"]
        )
        frame["target_fra_decoder_rescue"] = perturb_metrics["target"][1]
        frame["control_fra_decoder_rescue"] = perturb_metrics["matched_control"][1]
        frame["adjusted_fra_decoder_rescue"] = (
            frame["target_fra_decoder_rescue"] - frame["control_fra_decoder_rescue"]
        )
        cell_rows.append(frame)

    cells = pd.concat(cell_rows, ignore_index=True, sort=False)
    complete = cells[cells["status"].eq("completed")].copy()
    donor = (
        complete.groupby(["state_slug", "target", "matched_control", "donor"], observed=True)[
            ["adjusted_embedding_rescue", "adjusted_fra_decoder_rescue"]
        ]
        .mean()
        .reset_index()
    )
    summary_rows = []
    for key, frame in donor.groupby(["state_slug", "target", "matched_control"], observed=True):
        row = dict(zip(["state_slug", "target", "matched_control"], key))
        row["n_donors"] = len(frame)
        for metric in ["adjusted_embedding_rescue", "adjusted_fra_decoder_rescue"]:
            values = frame[metric].dropna().to_numpy()
            row[f"mean_{metric}"] = values.mean() if len(values) else np.nan
            if len(values) >= 3:
                low, high = bootstrap_ci(values, SEED + state_index)
            else:
                low, high = np.nan, np.nan
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row["embedding_rescue_supported"] = row["adjusted_embedding_rescue_ci_low"] > 0
        row["fra_decoder_rescue_supported"] = row["adjusted_fra_decoder_rescue_ci_low"] > 0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.merge(
            controls,
            left_on=["target", "matched_control"],
            right_on=["target", "control"],
            how="left",
        )
    return cells, donor, summary


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu_name = torch.cuda.get_device_name(0)
    if "5090" not in gpu_name and os.environ.get("ALLOW_NON_5090") != "1":
        raise RuntimeError(f"Refusing full run on {gpu_name}; switch to RTX 5090 or set ALLOW_NON_5090=1")
    device = torch.device("cuda")
    args = json.loads((MODEL_DIR / "args.json").read_text())
    vocab = GeneVocab.from_file(MODEL_DIR / "vocab.json")
    vocab.set_default_index(vocab[args["pad_token"]])
    model = build_model(vocab, args).eval().to(device)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fra = pd.read_csv(ROOT / "05_GRN/v15/fibrosis_reversibility_axis_v15.tsv", sep="\t")
    fra_weights = (
        fra.loc[fra["fra_axis_weight"] > 0, ["gene_symbol", "fra_axis_weight"]]
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")["fra_axis_weight"]
    )
    axes = pd.read_csv(ROOT / "04_CCC_LR/v15/mechanism_axis_consensus.tsv", sep="\t")
    excluded = set(fra["gene_symbol"].dropna().astype(str))
    excluded.update(axes["ligand"].dropna().astype(str))
    excluded.update(axes["receptor"].dropna().astype(str))

    cell_frames, donor_frames, summary_frames = [], [], []
    targets_table = pd.read_csv(INPUT_DIR / "perturbation_targets.tsv", sep="\t")
    state_slugs = targets_table["state_slug"].drop_duplicates().tolist()
    for state_index, state_slug in enumerate(state_slugs, start=1):
        cells, donors, summary = run_state(
            state_slug, state_index, model, vocab, args, device, fra_weights, excluded
        )
        cell_frames.append(cells)
        donor_frames.append(donors)
        summary_frames.append(summary)
    pd.concat(cell_frames, ignore_index=True).to_csv(
        OUTPUT / "scgpt_cell_level_metrics.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.concat(donor_frames, ignore_index=True).to_csv(
        OUTPUT / "scgpt_donor_level_metrics.tsv", sep="\t", index=False
    )
    summary = pd.concat(summary_frames, ignore_index=True)
    summary = summary.merge(
        targets_table[["state_slug", "target", "expected_ko_direction"]].drop_duplicates(),
        on=["state_slug", "target"], how="left",
    )
    summary["directional_prediction_pass"] = np.where(
        summary["expected_ko_direction"].eq("rescue_expected"),
        (summary["mean_adjusted_embedding_rescue"] > 0) & (summary["mean_adjusted_fra_decoder_rescue"] > 0),
        (summary["mean_adjusted_embedding_rescue"] < 0) & (summary["mean_adjusted_fra_decoder_rescue"] < 0),
    )
    summary["gpu_name"] = gpu_name
    summary["torch_version"] = torch.__version__
    summary["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    summary.to_csv(OUTPUT / "scgpt_v16_hsc_control_perturb.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
