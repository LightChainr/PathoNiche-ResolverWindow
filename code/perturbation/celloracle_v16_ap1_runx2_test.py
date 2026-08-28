#!/usr/bin/env python3
"""Orthogonally compare RUNX2, JUN and FOS in the cirrhotic HSC GRN."""

from __future__ import annotations

import contextlib
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
from scipy import sparse, stats

from celloracle.trajectory.oracle_GRN import _do_simulation


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("PN_ROOT", REPO_ROOT / "data/external_runtime"))
INPUT = ROOT / "01_scRNA/v15/perturbation_inputs"
V16 = Path(os.environ.get("V16_ROOT", REPO_ROOT / "work/v16_runtime"))
OUTPUT = V16 / "02_results/S7_hsc_control_perturbation/celloracle_ap1_runx2"
BASE_GRN = (
    ROOT
    / "00_raw/models_v15/celloracle/"
    / "human_promoter_base_GRN_hg19_gimmemotifsv5_fpr2.parquet"
)
SEED = 20260710
STATE_SLUGS = ["hsc_myofibroblast"]
V16_TARGETS = ["RUNX2", "JUN", "FOS"]


def build_oracle(adata: ad.AnnData, disease: str, seed: int) -> co.Oracle:
    np.random.seed(seed)
    subset = adata[adata.obs["disease"].astype(str).eq(disease)].copy()
    subset.obs["donor"] = subset.obs["donor"].astype(str).astype("category")
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        subset,
        cluster_column_name="donor",
        embedding_name="X_umap",
        transform="natural_log",
    )
    oracle.import_TF_data(TF_info_matrix_path=str(BASE_GRN))
    n_components = min(50, subset.n_obs - 1, subset.n_vars - 1)
    oracle.perform_PCA(n_components=n_components)
    k = max(10, min(50, int(subset.n_obs * 0.025)))
    oracle.knn_imputation(
        k=k,
        n_pca_dims=min(40, n_components),
        balanced=True,
        b_sight=min(k * 8, subset.n_obs - 1),
        b_maxl=min(k * 4, subset.n_obs - 1),
        n_jobs=3,
    )
    oracle.fit_GRN_for_simulation(
        GRN_unit="whole", alpha=10, use_cluster_specific_TFdict=False, verbose_level=0
    )
    return oracle


def donor_means(oracle: co.Oracle) -> pd.DataFrame:
    expression = oracle.adata.to_df(layer="imputed_count")
    expression["__donor"] = oracle.adata.obs["donor"].astype(str).to_numpy()
    return expression.groupby("__donor", observed=True).mean()


def choose_controls(
    coefficient: pd.DataFrame,
    expression: pd.DataFrame,
    targets: list[str],
    excluded: set[str],
) -> pd.DataFrame:
    out_strength = coefficient.abs().sum(axis=1)
    mean_expression = expression.mean(axis=0)
    pool = [
        gene
        for gene in coefficient.index
        if gene not in excluded
        and gene not in targets
        and out_strength.get(gene, 0) > 0
        and mean_expression.get(gene, 0) > 0.02
        and not gene.startswith(("MT-", "RPS", "RPL"))
    ]
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for target in targets:
        candidates = [gene for gene in pool if gene not in selected]
        if not candidates:
            raise RuntimeError(f"No matched-control pool remains for {target}")
        target_degree = np.log1p(out_strength[target])
        target_expression = np.log1p(mean_expression[target])
        distances = {
            gene: abs(np.log1p(out_strength[gene]) - target_degree)
            + abs(np.log1p(mean_expression[gene]) - target_expression)
            for gene in candidates
        }
        control = min(distances, key=distances.get)
        selected.append(control)
        rows.append(
            {
                "target": target,
                "control": control,
                "target_out_strength": out_strength[target],
                "control_out_strength": out_strength[control],
                "target_mean_expression": mean_expression[target],
                "control_mean_expression": mean_expression[control],
                "matching_distance": distances[control],
            }
        )
    return pd.DataFrame(rows)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return np.nan
    return float(np.dot(left, right) / denominator)


def simulate(
    expression: pd.DataFrame,
    coefficient: pd.DataFrame,
    gene: str,
) -> pd.DataFrame:
    simulation_input = expression.copy()
    simulation_input.loc[:, gene] = 0.0
    simulated = _do_simulation(
        coef_matrix=coefficient,
        simulation_input=simulation_input,
        gem=expression,
        n_propagation=3,
    )
    return simulated - expression


def summarize_delta(
    delta: pd.DataFrame,
    disease_vector: pd.Series,
    fra_weights: pd.Series,
    state_slug: str,
    perturb_gene: str,
    role: str,
    matched_to: str | None,
    network_type: str,
) -> pd.DataFrame:
    fra_genes = fra_weights.index.intersection(delta.columns)
    weights = fra_weights.loc[fra_genes]
    weights = weights / weights.sum()
    rows: list[dict[str, object]] = []
    for donor, values in delta.iterrows():
        fra_delta = values.loc[fra_genes]
        disease = disease_vector.loc[fra_genes]
        rows.append(
            {
                "state_slug": state_slug,
                "donor": donor,
                "perturb_gene": perturb_gene,
                "role": role,
                "matched_to": matched_to,
                "network_type": network_type,
                "fra_rescue_score": -float(np.average(fra_delta, weights=weights)),
                "fra_down_fraction": float((fra_delta < 0).mean()),
                "fra_disease_reversal_cosine": cosine(
                    fra_delta.to_numpy(), -disease.to_numpy()
                ),
                "global_disease_reversal_cosine": cosine(
                    values.to_numpy(), -disease_vector.loc[values.index].to_numpy()
                ),
                "rms_shift": float(np.sqrt(np.mean(values.to_numpy() ** 2))),
                "target_delta": float(values.get(perturb_gene, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarize_donors(donor_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["state_slug", "perturb_gene", "role", "matched_to", "network_type"]
    metrics = [
        "fra_rescue_score",
        "fra_down_fraction",
        "fra_disease_reversal_cosine",
        "global_disease_reversal_cosine",
        "rms_shift",
        "target_delta",
    ]
    rows: list[dict[str, object]] = []
    for key, frame in donor_metrics.groupby(keys, dropna=False, observed=True):
        row = dict(zip(keys, key))
        row["n_donors"] = len(frame)
        for metric in metrics:
            values = frame[metric].dropna().to_numpy(float)
            row[f"mean_{metric}"] = np.mean(values) if len(values) else np.nan
            row[f"sd_{metric}"] = np.std(values, ddof=1) if len(values) > 1 else np.nan
        rescue = frame["fra_rescue_score"].dropna().to_numpy(float)
        row["n_donors_positive_fra_rescue"] = int((rescue > 0).sum())
        row["fra_rescue_ttest_p"] = (
            stats.ttest_1samp(rescue, popmean=0).pvalue if len(rescue) >= 3 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.mean(
        rng.choice(values, size=(5000, len(values)), replace=True), axis=1
    )
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def run_state(state_slug: str, state_index: int) -> dict[str, str]:
    state_output = OUTPUT / state_slug
    state_output.mkdir(parents=True, exist_ok=True)
    log_path = state_output / "run.log"
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            print(f"state={state_slug} seed={SEED + state_index}", flush=True)
            adata = ad.read_h5ad(INPUT / state_slug / "celloracle_input.h5ad")
            targets = V16_TARGETS
            fra = pd.read_csv(
                ROOT / "05_GRN/v15/fibrosis_reversibility_axis_v15.tsv", sep="\t"
            )
            fra_weights = (
                fra.loc[fra["fra_axis_weight"] > 0, ["gene_symbol", "fra_axis_weight"]]
                .drop_duplicates("gene_symbol")
                .set_index("gene_symbol")["fra_axis_weight"]
            )
            axes = pd.read_csv(
                ROOT / "04_CCC_LR/v15/mechanism_axis_consensus.tsv", sep="\t"
            )
            excluded = set(fra["gene_symbol"].dropna().astype(str))
            excluded.update(axes["ligand"].dropna().astype(str))
            excluded.update(axes["receptor"].dropna().astype(str))

            healthy = build_oracle(adata, "healthy", SEED + state_index * 10 + 1)
            cirrhotic = build_oracle(adata, "cirrhotic", SEED + state_index * 10 + 2)
            healthy_expression = donor_means(healthy)
            cirrhotic_expression = donor_means(cirrhotic)
            disease_vector = cirrhotic_expression.mean() - healthy_expression.mean()
            coefficient = cirrhotic.coef_matrix.copy()
            healthy_coefficient = healthy.coef_matrix.copy()

            genes = coefficient.index.astype(str)
            pd.Series(genes).to_csv(state_output / "coefficient_genes.tsv", index=False, header=False)
            sparse.save_npz(
                state_output / "cirrhotic_coefficient_matrix.npz",
                sparse.csr_matrix(coefficient.to_numpy()),
                compressed=True,
            )
            sparse.save_npz(
                state_output / "healthy_coefficient_matrix.npz",
                sparse.csr_matrix(healthy_coefficient.loc[genes, genes].to_numpy()),
                compressed=True,
            )

            active_targets = [
                gene
                for gene in targets
                if gene in coefficient.index and coefficient.loc[gene].abs().sum() > 0
            ]
            unavailable = sorted(set(targets) - set(active_targets))
            pd.DataFrame(
                {
                    "target": targets,
                    "status": [
                        "active_cirrhotic_GRN" if gene in active_targets else "no_outgoing_cirrhotic_edge"
                        for gene in targets
                    ],
                }
            ).to_csv(state_output / "target_availability.tsv", sep="\t", index=False)
            if not active_targets:
                raise RuntimeError(f"No active CellOracle targets in {state_slug}: {unavailable}")

            controls = choose_controls(
                coefficient, cirrhotic_expression, active_targets, excluded
            )
            controls.to_csv(
                state_output / "degree_expression_matched_controls.tsv", sep="\t", index=False
            )
            cirrhotic.calculate_randomized_coef_table()
            randomized = cirrhotic.coef_matrix_randomized.copy()

            donor_frames: list[pd.DataFrame] = []
            gene_frames: list[pd.DataFrame] = []
            perturbations = [
                (gene, "prespecified_candidate", None) for gene in active_targets
            ] + [
                (row.control, "degree_expression_matched_null", row.target)
                for row in controls.itertuples(index=False)
            ]
            for gene, role, matched_to in perturbations:
                delta = simulate(cirrhotic_expression, coefficient, gene)
                donor_frames.append(
                    summarize_delta(
                        delta,
                        disease_vector,
                        fra_weights,
                        state_slug,
                        gene,
                        role,
                        matched_to,
                        "observed_cirrhotic_GRN",
                    )
                )
                long = delta.stack().rename("delta_expression").reset_index()
                long.columns = ["donor", "gene", "delta_expression"]
                long.insert(0, "state_slug", state_slug)
                long["perturb_gene"] = gene
                long["role"] = role
                long["matched_to"] = matched_to
                long["network_type"] = "observed_cirrhotic_GRN"
                gene_frames.append(long)

            for gene in active_targets:
                delta = simulate(cirrhotic_expression, randomized, gene)
                donor_frames.append(
                    summarize_delta(
                        delta,
                        disease_vector,
                        fra_weights,
                        state_slug,
                        gene,
                        "prespecified_candidate",
                        None,
                        "target_column_shuffled_GRN",
                    )
                )

            donor_metrics = pd.concat(donor_frames, ignore_index=True)
            donor_metrics.to_csv(
                state_output / "donor_level_metrics.tsv", sep="\t", index=False
            )
            summarize_donors(donor_metrics).to_csv(
                state_output / "perturbation_summary.tsv", sep="\t", index=False
            )
            pd.concat(gene_frames, ignore_index=True).to_csv(
                state_output / "gene_level_signed_shifts.tsv.gz",
                sep="\t",
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )

            adjusted_rows: list[dict[str, object]] = []
            for row_index, row in controls.iterrows():
                target = row["target"]
                control = row["control"]
                candidate_values = donor_metrics.loc[
                    (donor_metrics["perturb_gene"] == target)
                    & (donor_metrics["network_type"] == "observed_cirrhotic_GRN")
                ].set_index("donor")["fra_rescue_score"]
                control_values = donor_metrics.loc[
                    (donor_metrics["perturb_gene"] == control)
                    & (donor_metrics["network_type"] == "observed_cirrhotic_GRN")
                ].set_index("donor")["fra_rescue_score"]
                randomized_values = donor_metrics.loc[
                    (donor_metrics["perturb_gene"] == target)
                    & (donor_metrics["network_type"] == "target_column_shuffled_GRN")
                ].set_index("donor")["fra_rescue_score"]
                donors = candidate_values.index.intersection(control_values.index).intersection(
                    randomized_values.index
                )
                adjusted = candidate_values.loc[donors] - 0.5 * (
                    control_values.loc[donors] + randomized_values.loc[donors]
                )
                ci_low, ci_high = bootstrap_mean(
                    adjusted.to_numpy(), SEED + state_index * 100 + row_index
                )
                adjusted_rows.append(
                    {
                        "state_slug": state_slug,
                        "target": target,
                        "matched_control": control,
                        "n_donors": len(adjusted),
                        "mean_fra_rescue_observed": candidate_values.loc[donors].mean(),
                        "mean_fra_rescue_matched_control": control_values.loc[donors].mean(),
                        "mean_fra_rescue_randomized_grn": randomized_values.loc[donors].mean(),
                        "mean_null_adjusted_fra_rescue": adjusted.mean(),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "n_donors_positive_adjusted_rescue": int((adjusted > 0).sum()),
                        "directional_rescue_supported": bool(ci_low > 0),
                        "native_method": "CellOracle_0.18.0_donor_mean_exact_propagation",
                    }
                )
            pd.DataFrame(adjusted_rows).to_csv(
                state_output / "candidate_null_adjusted_summary.tsv", sep="\t", index=False
            )
            print(f"completed active={len(active_targets)} unavailable={unavailable}", flush=True)
            return {
                "state_slug": state_slug,
                "candidate_path": str(
                    state_output / "candidate_null_adjusted_summary.tsv"
                ),
            }
        except Exception:
            traceback.print_exc()
            raise


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_state, state, index): state
            for index, state in enumerate(STATE_SLUGS, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())

    candidates = pd.concat(
        [pd.read_csv(result["candidate_path"], sep="\t") for result in results],
        ignore_index=True,
    ).sort_values(
        ["directional_rescue_supported", "mean_null_adjusted_fra_rescue"],
        ascending=[False, False],
    )
    candidates.insert(0, "celloracle_rank_v16", np.arange(1, len(candidates) + 1))
    candidates.to_csv(
        OUTPUT / "celloracle_v16_ap1_runx2_ko.tsv", sep="\t", index=False
    )
    report = [
        "# CellOracle v16 AP-1/RUNX2 orthogonal test",
        "",
        "- CellOracle 0.18.0 with the human promoter hg19 base GRN.",
        "- Four states are modeled independently; healthy and cirrhotic GRNs are fitted separately to prevent cross-disease KNN smoothing.",
        "- Each disease-state input uses 1,500 genes, within-disease balanced KNN imputation, and Ridge alpha=10.",
        "- KO propagation uses the package's exact three-step signal-propagation function on each cirrhotic donor mean (five donor units where available).",
        "- A higher FRA rescue score means a predicted decrease in the frozen regression-anchored FRA.",
        "- Each candidate is calibrated against an expression/out-strength-matched TF KO and a target-column-shuffled GRN.",
        "- `directional_rescue_supported` requires the donor bootstrap 95% interval of the null-adjusted FRA rescue to remain above zero.",
        "- CellOracle can perturb only TFs with active outgoing edges in the fitted base-GRN-constrained model; non-TF candidates are assigned to scTenifold/scGPT.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
