#!/usr/bin/env python3
"""Prepare matrix-column annotations for streaming GSE244832 HSC pseudobulk."""

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "data/raw_external/GSE244832"
OUT = REPO_ROOT / "work/GSE244832_hsc_pseudobulk"

DONOR_ORDER = [
    "JB336", "JB288", "JB303", "JB317", "JB320", "JB289", "JB304", "JB318", "JB321",
    "JB337", "JB305", "JB290", "JB319", "JB322", "JB338", "JB339", "JB340", "JB341",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global SOURCE, OUT
    args = parse_args()
    SOURCE, OUT = args.source_dir, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(SOURCE / "hLIVER_cells.csv", header=None, names=["cell"])
    metadata = pd.read_csv(SOURCE / "hLIVER_metadata.csv").rename(columns={"Unnamed: 0": "cell"})
    if not cells.cell.equals(metadata.cell):
        raise ValueError("Matrix columns and metadata rows are not in identical cell order")

    observed = set(metadata["orig.ident"])
    if observed != set(DONOR_ORDER):
        raise ValueError(f"Unexpected donor IDs: {sorted(observed ^ set(DONOR_ORDER))}")
    donor_index = {donor: index for index, donor in enumerate(DONOR_ORDER)}
    mapping = pd.DataFrame({
        "donor_index": metadata["orig.ident"].map(donor_index).astype(int),
        "cluster": metadata["seurat_clusters"].astype(int),
    })
    mapping.to_csv(OUT / "cell_map.tsv", sep="\t", header=False, index=False)

    donor_meta = (
        metadata[["orig.ident", "condition"]]
        .drop_duplicates()
        .set_index("orig.ident")
        .loc[DONOR_ORDER]
        .reset_index()
        .rename(columns={"orig.ident": "donor"})
    )
    donor_meta.insert(0, "donor_index", range(len(donor_meta)))
    hsc_counts = metadata.loc[metadata.seurat_clusters == 3, "orig.ident"].value_counts()
    donor_meta["hsc_cells"] = donor_meta.donor.map(hsc_counts).fillna(0).astype(int)
    donor_meta.to_csv(OUT / "donor_metadata.tsv", sep="\t", index=False)

    genes = pd.read_csv(SOURCE / "hLIVER_genes.csv", header=None, names=["gene"])
    genes.to_csv(OUT / "genes.tsv", sep="\t", header=False, index=False)
    print(donor_meta.to_string(index=False))


if __name__ == "__main__":
    main()
