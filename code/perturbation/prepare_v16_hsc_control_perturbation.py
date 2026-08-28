#!/usr/bin/env python3
"""Prepare an HSC-only perturbation input with the v16 control architecture frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "data/raw_external/GSE136103/GSE136103_human_liver_cell_state_labels_v1.h5ad"
BASE = REPO_ROOT / "data/perturbation_inputs/hsc_myofibroblast"
OUT = REPO_ROOT / "work/perturbation_inputs"
STATE = "hsc_myofibroblast"
SEED = 20260711
TARGETS = [
    "RUNX1", "RUNX2", "YAP1", "WWTR1", "TEAD1", "TEAD2", "TEAD3", "TEAD4",
    "JUN", "FOS", "FOSL1", "FOSL2", "ETS1", "ETS2", "ITGB5", "GAS7",
    "SPON1", "SVEP1", "PDGFRA", "LAMA2", "EPHA3", "DPYSL3", "LTBP2",
    "EFEMP1", "SERPINE1",
]
EXPECTED_KO_DIRECTION = {
    **{g: "rescue_expected" for g in ["RUNX1", "RUNX2", "YAP1", "WWTR1", "TEAD1", "TEAD2", "TEAD3", "TEAD4", "JUN", "FOS", "FOSL1", "FOSL2", "ITGB5", "GAS7", "SPON1", "SVEP1", "PDGFRA", "LAMA2", "EPHA3", "DPYSL3", "LTBP2", "EFEMP1", "SERPINE1"]},
    "ETS1": "worsening_expected",
    "ETS2": "worsening_expected",
}


def balanced_positions(obs: pd.DataFrame, cap: int = 500) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    selected = []
    for _, positions in obs.groupby(["disease", "donor"], observed=True).indices.items():
        positions = np.asarray(positions, int)
        if len(positions) > cap:
            positions = rng.choice(positions, cap, replace=False)
        selected.extend(positions.tolist())
    return np.asarray(sorted(selected), int)


def collapse(matrix: sparse.csr_matrix, symbols: np.ndarray, genes: list[str]) -> sparse.csr_matrix:
    old = np.flatnonzero(np.isin(symbols, genes))
    new_index = {g: i for i, g in enumerate(genes)}
    columns = np.asarray([new_index[g] for g in symbols[old]], int)
    mapper = sparse.csr_matrix(
        (np.ones(len(old), np.float32), (np.arange(len(old)), columns)),
        shape=(len(old), len(genes)),
    )
    return (matrix[:, old].tocsr() @ mapper).tocsr().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, default=SOURCE)
    parser.add_argument("--base-input-dir", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global SOURCE, BASE, OUT
    args = parse_args()
    SOURCE, BASE, OUT = args.source_h5ad, args.base_input_dir, args.output_dir
    out = OUT / STATE
    out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(SOURCE)
    mask = a.obs["cell_state_v1"].astype(str).eq("HSC_Myofibroblast").to_numpy()
    state_positions = np.flatnonzero(mask)
    local = balanced_positions(a.obs.iloc[state_positions])
    positions = state_positions[local]
    obs = a.obs.iloc[positions].copy()
    counts = a.layers["counts"][positions].tocsr()
    symbols = a.var["gene_symbol"].fillna(a.var_names.to_series()).astype(str).str.upper().to_numpy()
    base_genes = pd.read_csv(BASE / "scgpt_genes.tsv", header=None)[0].astype(str).str.upper().tolist()
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    expressed_symbols = set(symbols[detected >= 10])
    targets = [g for g in TARGETS if g in expressed_symbols]
    genes = list(dict.fromkeys(base_genes + targets))
    matrix = collapse(counts, symbols, genes)
    sparse.save_npz(out / "scgpt_raw_counts_cells_by_genes.npz", matrix, compressed=True)
    pd.Series(genes).to_csv(out / "scgpt_genes.tsv", sep="\t", index=False, header=False)
    cells = pd.DataFrame({
        "cell": a.obs_names[positions].astype(str),
        "donor": obs["donor"].astype(str).to_numpy(),
        "disease": obs["disease"].astype(str).to_numpy(),
        "cell_state_v1": obs["cell_state_v1"].astype(str).to_numpy(),
    })
    cells.to_csv(out / "scgpt_cells.tsv", sep="\t", index=False)
    cirrhotic = cells["disease"].eq("cirrhotic").to_numpy()
    rows = []
    for gene in targets:
        i = genes.index(gene)
        values = matrix[:, i]
        cvalues = matrix[cirrhotic, i]
        detected_all = int(values.getnnz())
        detected_cirrhotic = int(cvalues.getnnz())
        donors = cells.loc[cirrhotic & (np.asarray(values.toarray()).ravel() > 0), "donor"].nunique()
        rows.append({
            "state_slug": STATE, "target": gene,
            "expected_ko_direction": EXPECTED_KO_DIRECTION[gene],
            "detected_cells_all": detected_all, "detected_cells_cirrhotic": detected_cirrhotic,
            "detected_cirrhotic_donors": int(donors),
        })
    target_table = pd.DataFrame(rows)
    target_table.to_csv(OUT / "perturbation_targets.tsv", sep="\t", index=False)
    sampling = cells.groupby(["disease", "donor"]).size().rename("n_cells").reset_index()
    sampling.to_csv(OUT / "sampling_manifest.tsv", sep="\t", index=False)
    manifest = {
        "source": str(SOURCE), "state": STATE, "seed": SEED,
        "n_cells": int(len(cells)), "n_genes": int(len(genes)),
        "targets_requested": TARGETS, "targets_in_input": targets,
        "missing_or_detected_lt10": sorted(set(TARGETS) - set(targets)),
    }
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(target_table.to_string(index=False))


if __name__ == "__main__":
    main()
