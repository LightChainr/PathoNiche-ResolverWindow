#!/usr/bin/env python3
"""Sequential pooled-library state trajectories from GSE305331 snRNA-seq."""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "data/raw_external/GSE305331/GSE305331_RAW.tar"
OUT = REPO_ROOT / "results/GSE305331_state_trajectories"

SAMPLES = {
    "GSM9168817": ("CCl4", "peak", 0), "GSM9168818": ("CCl4", "2wk", 2),
    "GSM9168819": ("CCl4", "6wk", 6), "GSM9168820": ("FAT_MASH", "peak", 0),
    "GSM9168821": ("FAT_MASH", "2wk", 2), "GSM9168822": ("FAT_MASH", "6wk", 6),
    "GSM9168823": ("Untreated", "baseline", -1),
}
MARKERS = {
    "Hepatocyte": ["Alb", "Apoa1", "Apoa2", "Ttr", "Cyp2e1", "Cps1", "Ass1"],
    "HSC_Mesenchymal": ["Col1a1", "Col1a2", "Col3a1", "Dcn", "Cxcl12", "Hand2", "Pdgfra", "Des"],
    "Endothelial": ["Aqp1", "Egfl7", "Dnase1l3", "Ptprb", "Clec4g", "Kdr", "Esam", "Stab2"],
    "Myeloid": ["Lyz2", "Adgre1", "Csf1r", "Tyrobp", "C1qa", "C1qb", "Trem2"],
    "T_NK": ["Cd3d", "Cd3e", "Trbc1", "Nkg7", "Klrk1", "Ccl5"],
    "B_cell": ["Cd79a", "Ms4a1", "Cd37", "Cd74", "H2-Aa", "Cd19"],
    "Cholangiocyte": ["Krt19", "Krt8", "Krt18", "Krt7", "Epcam", "Spp1"],
}
MODULES = {
    "HSC_pathogenic_retention": ["Itgb5", "Gas7", "Spon1", "Pdgfra", "Lama2", "Epha3", "Dpysl3", "Svep1", "Fbln5", "Srpx2", "Nedd9"],
    "HSC_SFRP_brake": ["Sfrp1", "Sfrp2", "Dkk3", "Timp3"],
    "Endo4_WNT_resolver": ["Wnt9b", "Vwf", "Ace", "Wnt7b", "Kdr", "Emcn", "Esm1"],
    "LAM_resolver": ["Trem2", "Gpnmb", "Spp1", "Cd9", "Lpl", "Mmp14", "Ctsb", "Ctsd"],
}
EXPECTED_TYPE = {
    "HSC_pathogenic_retention": "HSC_Mesenchymal", "HSC_SFRP_brake": "HSC_Mesenchymal",
    "Endo4_WNT_resolver": "Endothelial", "LAM_resolver": "Myeloid",
}


def member_name(archive: tarfile.TarFile, sample: str, suffix: str) -> str:
    matches = [name for name in archive.getnames() if name.startswith(sample) and name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {sample} *{suffix}, found {matches}")
    return matches[0]


def read_sample(archive: tarfile.TarFile, sample: str) -> ad.AnnData:
    matrix_member = archive.extractfile(member_name(archive, sample, "_matrix.mtx.gz"))
    feature_member = archive.extractfile(member_name(archive, sample, "_features.tsv.gz"))
    barcode_member = archive.extractfile(member_name(archive, sample, "_barcodes.tsv.gz"))
    if matrix_member is None or feature_member is None or barcode_member is None:
        raise OSError(f"Cannot extract {sample}")
    with gzip.GzipFile(fileobj=matrix_member) as handle:
        matrix = mmread(handle).tocsr().T
    with gzip.GzipFile(fileobj=feature_member) as handle:
        features = pd.read_csv(handle, sep="\t", header=None)
    with gzip.GzipFile(fileobj=barcode_member) as handle:
        barcodes = pd.read_csv(handle, sep="\t", header=None)[0].astype(str)
    genes = pd.Index(features[1].astype(str).to_numpy(), name=None)
    result = ad.AnnData(matrix)
    result.var_names = genes
    result.var_names_make_unique()
    result.obs_names = pd.Index((sample + "_" + barcodes).to_numpy(), name=None)
    result.obs["sample"] = sample
    return result


def process_sample(archive: tarfile.TarFile, sample: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = read_sample(archive, sample)
    input_cells, input_genes, input_nnz = data.n_obs, data.n_vars, data.X.nnz
    sc.pp.calculate_qc_metrics(data, inplace=True, percent_top=None)
    upper_counts = np.quantile(data.obs.total_counts, 0.995)
    upper_genes = np.quantile(data.obs.n_genes_by_counts, 0.995)
    data = data[(data.obs.total_counts >= 200) & (data.obs.n_genes_by_counts >= 100)
                & (data.obs.total_counts <= upper_counts) & (data.obs.n_genes_by_counts <= upper_genes)].copy()
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=min(2500, data.n_vars), flavor="seurat")
    representation = data[:, data.var.highly_variable].copy()
    sc.pp.scale(representation, max_value=10)
    sc.tl.pca(representation, n_comps=35)
    sc.pp.neighbors(representation, n_neighbors=20, n_pcs=30)
    sc.tl.leiden(representation, resolution=0.7, key_added="leiden", random_state=20260711)
    data.obs["leiden"] = representation.obs["leiden"].astype(str).to_numpy()
    del representation

    score_columns = []
    for cell_type, markers in MARKERS.items():
        genes = [gene for gene in markers if gene in data.var_names]
        column = f"score_{cell_type}"
        sc.tl.score_genes(data, genes, score_name=column)
        score_columns.append(column)
    cluster_scores = data.obs.groupby("leiden", observed=True)[score_columns].mean()
    standardized = (cluster_scores - cluster_scores.mean()) / cluster_scores.std(ddof=0).replace(0, np.nan)
    labels = standardized.idxmax(axis=1).str.removeprefix("score_")
    labels[standardized.max(axis=1) < 1.0] = "Unresolved"
    data.obs["cell_type"] = data.obs.leiden.map(labels).astype(str)
    annotation = cluster_scores.copy()
    annotation[[f"z_{column}" for column in score_columns]] = standardized
    annotation["cell_type"] = labels
    annotation["n_cells"] = data.obs.leiden.value_counts().reindex(annotation.index)
    annotation["sample"] = sample
    annotation = annotation.reset_index()

    etiology, timepoint, week = SAMPLES[sample]
    composition = data.obs.cell_type.value_counts().rename_axis("cell_type").reset_index(name="n_cells")
    composition["sample"] = sample
    composition["etiology"] = etiology
    composition["timepoint"] = timepoint
    composition["week"] = week

    trajectory_rows = []
    for module, markers in MODULES.items():
        genes = [gene for gene in markers if gene in data.var_names]
        sc.tl.score_genes(data, genes, score_name=module)
        cell_type = EXPECTED_TYPE[module]
        values = data.obs.loc[data.obs.cell_type == cell_type, module]
        trajectory_rows.append({
            "sample": sample, "etiology": etiology, "timepoint": timepoint, "week": week,
            "cell_type": cell_type, "module": module, "mean": values.mean(),
            "median": values.median(), "count": len(values), "n_genes": len(genes),
        })
    trajectory = pd.DataFrame(trajectory_rows)
    summary = pd.DataFrame([{
        "sample": sample, "etiology": etiology, "timepoint": timepoint, "week": week,
        "input_cells": input_cells, "cells_after_qc": data.n_obs, "genes": input_genes,
        "input_nnz": input_nnz, "n_clusters": data.obs.leiden.nunique(),
    }])
    return annotation, composition, trajectory, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global ARCHIVE, OUT
    args = parse_args()
    ARCHIVE, OUT = args.archive, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    annotations, compositions, trajectories, summaries = [], [], [], []
    with tarfile.open(ARCHIVE) as archive:
        for sample in SAMPLES:
            annotation, composition, trajectory, summary = process_sample(archive, sample)
            annotations.append(annotation); compositions.append(composition)
            trajectories.append(trajectory); summaries.append(summary)
            print(summary.to_string(index=False), flush=True)
    pd.concat(annotations, ignore_index=True).to_csv(OUT / "cluster_annotation.tsv", sep="\t", index=False)
    pd.concat(compositions, ignore_index=True).to_csv(OUT / "sample_celltype_counts.tsv", sep="\t", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(OUT / "state_module_trajectories.tsv", sep="\t", index=False)
    summary_table = pd.concat(summaries, ignore_index=True)
    summary_table.to_csv(OUT / "sample_processing_summary.tsv", sep="\t", index=False)
    (OUT / "inference_limit.json").write_text(json.dumps({
        "limit": "one pooled library per condition; descriptive trajectory only; cells are not biological replicates",
        "annotation": "independent per-library clustering with broad marker z-score confidence gate",
    }, indent=2))


if __name__ == "__main__":
    main()
