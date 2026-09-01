#!/usr/bin/env python3
"""Recover biopsy-core geometry from the public UQ MASLD Visium arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


PUBLISHED_ARRAYS = [
    "VLP115_A",
    "VLP116_D",
    "VLP119_A",
    "VLP119_D",
    "VLP120_A",
    "VLP120_D",
    "VLP121_A",
    "VLP121_D",
]


def label_hex_components(positions: pd.DataFrame) -> pd.DataFrame:
    tissue = positions.loc[positions["in_tissue"].eq(1)].copy().reset_index(drop=True)
    coords = list(zip(tissue["array_row"], tissue["array_col"], strict=True))
    index = {coord: i for i, coord in enumerate(coords)}
    edges_i: list[int] = []
    edges_j: list[int] = []
    offsets = ((0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for i, (row, col) in enumerate(coords):
        for dr, dc in offsets:
            j = index.get((row + dr, col + dc))
            if j is not None:
                edges_i.append(i)
                edges_j.append(j)
    graph = coo_matrix(
        (np.ones(len(edges_i), dtype=np.uint8), (edges_i, edges_j)),
        shape=(len(tissue), len(tissue)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    tissue["raw_component"] = labels

    sizes = tissue.groupby("raw_component", observed=True).size()
    keep = sizes.index[sizes.ge(20)]
    tissue = tissue.loc[tissue["raw_component"].isin(keep)].copy()

    centroids = (
        tissue.groupby("raw_component", observed=True)
        .agg(
            n_spots=("barcode", "size"),
            centroid_col=("array_col", "mean"),
            centroid_row=("array_row", "mean"),
        )
        .sort_values(["centroid_col", "centroid_row"])
        .reset_index()
    )
    centroids["component"] = [f"C{i + 1}" for i in range(len(centroids))]
    tissue = tissue.merge(centroids, on="raw_component", how="left", validate="many_to_one")
    return tissue


def plot_arrays(all_spots: pd.DataFrame, output: Path) -> None:
    arrays = list(dict.fromkeys(PUBLISHED_ARRAYS + sorted(set(all_spots["array"]) - set(PUBLISHED_ARRAYS))))
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.2), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for ax, array in zip(axes.flat, arrays, strict=True):
        frame = all_spots.loc[all_spots["array"].eq(array)]
        for i, (component, core) in enumerate(frame.groupby("component", sort=False, observed=True)):
            ax.scatter(
                core["array_col"],
                -core["array_row"],
                s=7,
                color=palette(i % 10),
                linewidths=0,
                rasterized=True,
            )
            ax.text(
                core["centroid_col"].iat[0],
                -core["centroid_row"].iat[0],
                f"{component}\n(n={len(core)})",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
            )
        suffix = "published" if array in PUBLISHED_ARRAYS else "additional"
        ax.set_title(f"{array} ({suffix})", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.suptitle("UQ MASLD Visium biopsy-core geometry", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visium-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    arrays = sorted(path.name for path in args.visium_root.iterdir() if path.is_dir())
    component_frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for array in arrays:
        positions_path = args.visium_root / array / "spatial" / "tissue_positions.csv"
        positions = pd.read_csv(positions_path)
        labelled = label_hex_components(positions)
        labelled.insert(0, "array", array)
        component_frames.append(labelled)
        summaries.append(
            labelled[
                ["array", "component", "n_spots", "centroid_col", "centroid_row"]
            ].drop_duplicates()
        )

    all_spots = pd.concat(component_frames, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary["published_primary"] = summary["array"].isin(PUBLISHED_ARRAYS)
    summary = summary.sort_values(["published_primary", "array", "centroid_col"], ascending=[False, True, True])

    all_spots.to_parquet(args.results_dir / "UQ_Visium_spot_component_map.parquet", index=False)
    summary.to_csv(args.results_dir / "UQ_Visium_component_summary.tsv", sep="\t", index=False)
    plot_arrays(all_spots, args.figures_dir / "UQ_Visium_component_geometry.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
