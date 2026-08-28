#!/usr/bin/env python3
"""QC, broad cell typing, and donor-level pseudobulk for GSE295293."""

from __future__ import annotations

import argparse
import gzip
import re
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "data/raw_external/GSE295293/GSE295293_RAW.tar"
WORK = REPO_ROOT / "work/GSE295293"
OUT = REPO_ROOT / "results/GSE295293_dynamic_regression"

MARKERS = {
    "Hepatocyte": ["Alb", "Apoa1", "Apoa2", "Ttr", "Cyp2e1", "Cps1", "Ass1"],
    "HSC_Mesenchymal": ["Col1a1", "Col1a2", "Col3a1", "Dcn", "Col6a1", "Pdgfra", "Des", "Rgs5"],
    "Endothelial": ["Pecam1", "Kdr", "Klf2", "Emcn", "Vwf", "Esam", "Klf4"],
    "Myeloid": ["Lyz2", "Adgre1", "Csf1r", "Tyrobp", "C1qa", "C1qb", "Trem2"],
    "T_NK": ["Cd3d", "Cd3e", "Trbc1", "Nkg7", "Klrk1", "Ccl5"],
    "B_cell": ["Cd79a", "Ms4a1", "Cd37", "Cd74", "H2-Aa", "Cd19"],
    "Cholangiocyte": ["Krt19", "Krt8", "Krt18", "Krt7", "Epcam", "Krt23"],
}

# Frozen after review of canonical top markers from the first unsupervised run.
MANUAL_CLUSTER_LABELS = {
    "0": "Endothelial", "1": "Endothelial", "2": "HSC_Mesenchymal",
    "3": "Myeloid", "4": "Hepatocyte", "5": "Myeloid",
    "6": "Cholangiocyte", "7": "Hepatocyte", "8": "B_cell",
    "9": "Hepatocyte", "10": "T_NK", "11": "Myeloid",
    "12": "Hepatocyte", "13": "Hepatocyte", "14": "Hepatocyte",
    "15": "Hepatocyte", "16": "Hepatocyte", "17": "Hepatocyte",
    "18": "Endothelial", "19": "Myeloid", "20": "Hepatocyte",
    "21": "HSC_Mesenchymal", "22": "Hepatocyte", "23": "Cholangiocyte",
    "24": "Endothelial", "25": "Hepatocyte", "26": "HSC_Mesenchymal",
    "27": "Myeloid", "28": "Endothelial", "29": "Myeloid",
    "30": "HSC_Mesenchymal", "31": "Endothelial", "32": "Myeloid",
}


def condition_from_name(name: str) -> str:
    lowered = name.lower()
    if "regression" in lowered:
        return "regression"
    if "firosis" in lowered or "fibrosis" in lowered:
        return "fibrosis"
    if "control" in lowered:
        return "control"
    raise ValueError(f"Cannot infer condition from {name}")


def extract_archive() -> None:
    sentinel = WORK / ".extracted"
    if sentinel.exists():
        return
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(WORK)
    sentinel.touch()


def read_sample(matrix_path: Path) -> ad.AnnData:
    prefix = matrix_path.name.removesuffix("_matrix.mtx.gz")
    feature_path = matrix_path.with_name(prefix + "_features.tsv.gz")
    barcode_path = matrix_path.with_name(prefix + "_barcodes.tsv.gz")
    with gzip.open(matrix_path, "rb") as handle:
        matrix = mmread(handle).tocsr().T
    features = pd.read_csv(feature_path, sep="\t", header=None)
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None)[0].astype(str)
    genes = pd.Index((features[1] if features.shape[1] > 1 else features[0]).astype(str), name=None)
    genes.name = None
    sample = re.match(r"(GSM\d+)", matrix_path.name).group(1)
    result = ad.AnnData(matrix)
    result.var_names = genes
    result.var_names_make_unique()
    result.obs_names = pd.Index((sample + "_" + barcodes).to_numpy(), name=None)
    result.obs["sample"] = sample
    result.obs["condition"] = condition_from_name(matrix_path.name)
    result.obs["source_file"] = matrix_path.name
    return result


def annotate_clusters(adata: ad.AnnData) -> pd.DataFrame:
    available = set(adata.var_names)
    score_columns = []
    for cell_type, markers in MARKERS.items():
        genes = [gene for gene in markers if gene in available]
        if len(genes) < 2:
            continue
        column = f"score_{cell_type}"
        sc.tl.score_genes(adata, genes, score_name=column, use_raw=False)
        score_columns.append(column)
    cluster_scores = adata.obs.groupby("leiden", observed=True)[score_columns].mean()
    score_label = cluster_scores.idxmax(axis=1).str.removeprefix("score_")
    cluster_label = pd.Series(
        [MANUAL_CLUSTER_LABELS.get(str(cluster), score_label.loc[cluster]) for cluster in cluster_scores.index],
        index=cluster_scores.index,
        name="cell_type",
    )
    adata.obs["cell_type"] = adata.obs.leiden.map(cluster_label).astype(str)
    output = cluster_scores.copy()
    output["cell_type"] = cluster_label
    output["n_cells"] = adata.obs.leiden.value_counts().reindex(output.index)
    return output.reset_index()


def pseudobulk(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, metadata = [], []
    for (sample, cell_type), index in adata.obs.groupby(["sample", "cell_type"], observed=True).indices.items():
        if len(index) < 30:
            continue
        counts = np.asarray(adata.layers["counts"][index].sum(axis=0)).ravel()
        label = f"{sample}__{cell_type}"
        rows.append(pd.Series(counts, index=adata.var_names, name=label))
        condition = adata.obs.iloc[index[0]].condition
        metadata.append({"pseudobulk": label, "sample": sample, "cell_type": cell_type,
                         "condition": condition, "n_cells": len(index)})
    matrix = pd.DataFrame(rows).T
    return matrix, pd.DataFrame(metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global ARCHIVE, WORK, OUT
    args = parse_args()
    ARCHIVE, WORK, OUT = args.archive, args.work_dir, args.output_dir
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sc.settings.seed = 20260711
    extract_archive()
    matrices = sorted(WORK.glob("GSM*_matrix.mtx.gz"))
    datasets = [read_sample(path) for path in matrices]
    adata = ad.concat(datasets, join="inner", merge="same", index_unique=None)
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)
    upper_counts = float(np.quantile(adata.obs.total_counts, 0.995))
    upper_genes = float(np.quantile(adata.obs.n_genes_by_counts, 0.995))
    adata = adata[(adata.obs.total_counts >= 150) & (adata.obs.n_genes_by_counts >= 80)
                  & (adata.obs.total_counts <= upper_counts) & (adata.obs.n_genes_by_counts <= upper_genes)].copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2500, adata.n_vars), flavor="seurat", subset=False)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=40, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=30)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden")
    sc.tl.umap(adata, min_dist=0.35)
    adata.X = adata.layers["lognorm"].copy()
    cluster_annotation = annotate_clusters(adata)
    cluster_annotation.to_csv(OUT / "cluster_annotation.tsv", sep="\t", index=False)

    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", pts=True)
    markers = sc.get.rank_genes_groups_df(adata, group=None)
    markers.groupby("group", observed=True).head(30).to_csv(OUT / "cluster_top_markers.tsv", sep="\t", index=False)

    coordinates = adata.obs[["sample", "condition", "leiden", "cell_type", "total_counts", "n_genes_by_counts"]].copy()
    coordinates[["UMAP1", "UMAP2"]] = adata.obsm["X_umap"]
    coordinates.to_csv(OUT / "cell_metadata_umap.tsv.gz", sep="\t", compression="gzip")
    composition = adata.obs.groupby(["sample", "condition", "cell_type"], observed=True).size().rename("n_cells").reset_index()
    composition.to_csv(OUT / "sample_celltype_counts.tsv", sep="\t", index=False)

    matrix, metadata = pseudobulk(adata)
    matrix.to_csv(WORK / "pseudobulk_counts.tsv.gz", sep="\t", compression="gzip")
    metadata.to_csv(WORK / "pseudobulk_metadata.tsv", sep="\t", index=False)
    summary = {
        "n_cells_after_qc": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_samples": int(adata.obs["sample"].nunique()),
        "condition_cells": adata.obs.condition.value_counts().to_dict(),
        "cell_type_cells": adata.obs.cell_type.value_counts().to_dict(),
        "n_pseudobulks": int(metadata.shape[0]),
    }
    (OUT / "preparation_summary.json").write_text(pd.Series(summary).to_json(indent=2))
    print(summary)


if __name__ == "__main__":
    main()
