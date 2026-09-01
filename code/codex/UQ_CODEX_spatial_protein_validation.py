#!/usr/bin/env python3
"""Quantify CODEX protein organization around Collagen-IV-rich MASLD scar fields."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from scipy.spatial import cKDTree
from skimage import measure, morphology
from skimage.filters import gaussian


SAMPLES = {
    "17P29186": {"fibrosis_stage": "F1", "stage_numeric": 1},
    "16P15496": {"fibrosis_stage": "F2", "stage_numeric": 2},
    "16P33251": {"fibrosis_stage": "F4", "stage_numeric": 4},
}
MARKERS = ["DAPI", "CD44", "Vimentin", "CD14", "CD31", "PANCK", "CD163", "CollagenIV"]
OUTCOMES = ["Vimentin", "CD31", "CD163", "CD14", "CD44", "mesenchymal_endothelial_balance"]
COLORS = {
    "Vimentin": "#B24C63",
    "CD31": "#2789A7",
    "CD163": "#6B8E23",
    "CD14": "#B07B2A",
    "CD44": "#76528B",
    "mesenchymal_endothelial_balance": "#7A1F3D",
}


def tag(description: str, name: str) -> str:
    match = re.search(fr"<{name}>(.*?)</{name}>", description, flags=re.S)
    return match.group(1).strip() if match else ""


def channel_metadata(path: Path) -> pd.DataFrame:
    rows = []
    with tifffile.TiffFile(path) as handle:
        for index, page in enumerate(handle.series[0].pages):
            description = page.description or ""
            rows.append(
                {
                    "channel_index": index,
                    "name": tag(description, "Name"),
                    "biomarker": tag(description, "Biomarker"),
                    "exposure_time": tag(description, "ExposureTime"),
                    "signal_units": tag(description, "SignalUnits"),
                }
            )
    return pd.DataFrame(rows)


def largest_tissue_mask(dapi: np.ndarray, vimentin: np.ndarray, panck: np.ndarray) -> np.ndarray:
    base = np.maximum.reduce([dapi.astype(float), 0.25 * vimentin, 0.25 * panck])
    mask = gaussian(base, sigma=2, preserve_range=True) > 1.0
    mask = morphology.closing(mask, morphology.disk(3))
    labels = measure.label(mask)
    if labels.max() == 0:
        raise RuntimeError("No tissue component detected")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def block_sum(values: np.ndarray, size: int) -> np.ndarray:
    rows = values.shape[0] // size
    columns = values.shape[1] // size
    cropped = values[: rows * size, : columns * size]
    return cropped.reshape(rows, size, columns, size).sum(axis=(1, 3))


def winsorized_z(values: np.ndarray) -> np.ndarray:
    low, high = np.nanpercentile(values, [1, 99])
    clipped = np.clip(values, low, high)
    return (clipped - np.nanmean(clipped)) / max(np.nanstd(clipped, ddof=1), 1e-8)


def patch_table(images: dict[str, np.ndarray], mask: np.ndarray, patch_size: int) -> tuple[pd.DataFrame, tuple[int, int]]:
    denominator = block_sum(mask.astype(float), patch_size)
    coverage = denominator / (patch_size**2)
    valid = coverage >= 0.15
    row_index, column_index = np.indices(coverage.shape)
    table = pd.DataFrame(
        {
            "patch_row": row_index[valid],
            "patch_col": column_index[valid],
            "tissue_coverage": coverage[valid],
        }
    )
    for marker, image in images.items():
        mean = block_sum(image.astype(float) * mask, patch_size) / np.maximum(denominator, 1)
        table[marker] = mean[valid]
        table[f"z_{marker}"] = winsorized_z(mean[valid])
    table["mesenchymal_endothelial_balance"] = table["z_Vimentin"] - table["z_CD31"]
    table["scar_patch"] = table["z_CollagenIV"].ge(table["z_CollagenIV"].quantile(0.80))
    return table, coverage.shape


def nearest_fill(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()
    missing = np.argwhere(~np.isfinite(result))
    valid = np.argwhere(np.isfinite(result))
    if len(missing):
        tree = cKDTree(valid)
        _, nearest = tree.query(missing, k=1)
        result[missing[:, 0], missing[:, 1]] = result[valid[nearest, 0], valid[nearest, 1]]
    return result


def rectangular_transforms(rows: int, columns: int) -> list[np.ndarray]:
    grid = np.arange(rows * columns).reshape(rows, columns)
    forms = [grid, np.flipud(grid), np.fliplr(grid), np.flipud(np.fliplr(grid))]
    identity = tuple(grid.ravel())
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for form in forms:
        for dy in range(rows):
            for dx in range(columns):
                index = np.roll(np.roll(form, dy, axis=0), dx, axis=1).ravel()
                key = tuple(map(int, index))
                if key != identity:
                    unique[key] = index
    return list(unique.values())


def toroidal_moran(grid: np.ndarray) -> float:
    centered = grid - grid.mean()
    lag = (
        np.roll(centered, 1, 0)
        + np.roll(centered, -1, 0)
        + np.roll(centered, 1, 1)
        + np.roll(centered, -1, 1)
    )
    denominator = np.sum(centered**2)
    return float(np.sum(centered * lag) / (4 * denominator)) if denominator else np.nan


def fit_spatial_null(table: pd.DataFrame, outcome: str, shape: tuple[int, int]) -> dict[str, float]:
    y = table[outcome].to_numpy(float) if outcome == "mesenchymal_endothelial_balance" else table[f"z_{outcome}"].to_numpy(float)
    collagen = table["z_CollagenIV"].to_numpy(float)
    nuisance = np.column_stack(
        [np.ones(len(table)), table["z_DAPI"].to_numpy(float), table["tissue_coverage"].to_numpy(float)]
    )
    full = np.column_stack([nuisance, collagen])
    coefficients = np.linalg.lstsq(full, y, rcond=None)[0]
    beta = float(coefficients[-1])
    residual = y - full @ coefficients
    dof = max(len(y) - full.shape[1], 1)
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(full.T @ full)
    se = float(np.sqrt(max(covariance[-1, -1], 0)))

    nuisance_coefficients = np.linalg.lstsq(nuisance, y, rcond=None)[0]
    fitted = nuisance @ nuisance_coefficients
    field = y - fitted
    rows, columns = shape
    position = table["patch_row"].to_numpy(int) * columns + table["patch_col"].to_numpy(int)
    grid = np.full(shape, np.nan)
    grid.ravel()[position] = field
    grid = nearest_fill(grid)
    transforms = rectangular_transforms(*shape)
    null = np.empty(len(transforms))
    moran = np.empty(len(transforms))
    for i, index in enumerate(transforms):
        transformed = grid.ravel()[index][position]
        null[i] = np.linalg.lstsq(full, fitted + transformed, rcond=None)[0][-1]
        moran[i] = toroidal_moran(grid.ravel()[index].reshape(shape))
    p = float((1 + np.sum(np.abs(null) >= abs(beta))) / (len(null) + 1))
    return {
        "outcome": outcome,
        "collagen_coupling_beta": beta,
        "beta_se": se,
        "beta_ci_low": beta - 1.96 * se,
        "beta_ci_high": beta + 1.96 * se,
        "spatial_field_p": p,
        "n_spatial_transforms": len(transforms),
        "residual_moran": toroidal_moran(grid),
        "permuted_moran_min": float(np.min(moran)),
        "permuted_moran_max": float(np.max(moran)),
    }


def normalized_channel(image: np.ndarray) -> np.ndarray:
    positive = image[image > 0]
    high = np.percentile(positive, 99.7) if len(positive) else 1
    return np.clip(image.astype(float) / max(high, 1), 0, 1)


def preview_composite(images: dict[str, np.ndarray]) -> np.ndarray:
    dapi = normalized_channel(images["DAPI"])
    vimentin = normalized_channel(images["Vimentin"])
    cd31 = normalized_channel(images["CD31"])
    cd163 = normalized_channel(images["CD163"])
    collagen = normalized_channel(images["CollagenIV"])
    return np.stack(
        [np.clip(0.75 * vimentin + 0.35 * cd163 + 0.18 * dapi, 0, 1),
         np.clip(0.75 * cd31 + 0.15 * dapi, 0, 1),
         np.clip(0.90 * collagen + 0.30 * dapi, 0, 1)],
        axis=-1,
    )


def crop_to_mask(image: np.ndarray, mask: np.ndarray, margin: int = 8) -> np.ndarray:
    rows, columns = np.where(mask)
    y0, y1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, image.shape[0])
    x0, x1 = max(columns.min() - margin, 0), min(columns.max() + margin + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def plot_results(
    previews: dict[str, np.ndarray],
    patch_scores: pd.DataFrame,
    coupling: pd.DataFrame,
    enrichment: pd.DataFrame,
    output: Path,
) -> None:
    sns.set_theme(style="ticks", context="paper")
    ordered = sorted(SAMPLES, key=lambda sample: SAMPLES[sample]["stage_numeric"])
    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.85, 0.75, 1.0])

    for column, sample in enumerate(ordered):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(previews[sample])
        ax.set_axis_off()
        ax.set_title(f"{SAMPLES[sample]['fibrosis_stage']} | {sample}", fontweight="bold")
        if column == 0:
            ax.text(-0.08, 1.05, "A", transform=ax.transAxes, fontsize=15, fontweight="bold")

    for column, sample in enumerate(ordered):
        ax = fig.add_subplot(grid[1, column])
        local = patch_scores.loc[patch_scores["sample"].eq(sample)]
        shape = (int(local["grid_rows"].iat[0]), int(local["grid_columns"].iat[0]))
        image = np.full(shape, np.nan)
        image[local["patch_row"].to_numpy(int), local["patch_col"].to_numpy(int)] = local["mesenchymal_endothelial_balance"]
        artist = ax.imshow(image, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
        ax.set_axis_off()
        ax.set_title("Vimentin - CD31 balance", fontsize=9)
        if column == 0:
            ax.text(-0.08, 1.05, "B", transform=ax.transAxes, fontsize=15, fontweight="bold")
    colorbar = fig.colorbar(artist, ax=fig.axes[3:6], shrink=0.78, pad=0.01)
    colorbar.set_label("Winsorized z-score difference")

    ax = fig.add_subplot(grid[2, 0])
    show = ["Vimentin", "CD31", "CD163", "mesenchymal_endothelial_balance"]
    for outcome in ["Vimentin", "CD31", "CD163"]:
        local = coupling.loc[coupling["outcome"].eq(outcome)].sort_values("stage_numeric")
        ax.plot(local["stage_numeric"], local["collagen_coupling_beta"], marker="o", lw=2, color=COLORS[outcome], label=outcome.replace("_", " "))
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks([1, 2, 4], ["F1", "F2", "F4"])
    ax.set_xlabel("Fibrosis stage")
    ax.set_ylabel("Collagen-IV coupling beta")
    ax.set_title("C  Protein coupling at the scar field", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(grid[2, 1])
    local = enrichment.loc[enrichment["outcome"].isin(show)].copy()
    sns.barplot(data=local, x="fibrosis_stage", y="scar_enrichment", hue="outcome", hue_order=show, palette=COLORS, ax=ax)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Fibrosis stage")
    ax.set_ylabel("Top-scar minus non-scar z-score")
    ax.set_title("D  Enrichment in top 20% scar patches", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(grid[2, 2])
    heat = enrichment.pivot(index="outcome", columns="fibrosis_stage", values="scar_enrichment")
    heat = heat.reindex(show)[["F1", "F2", "F4"]]
    sns.heatmap(heat, cmap="vlag", center=0, annot=True, fmt=".2f", cbar_kws={"label": "Scar-patch enrichment"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("E  Scar-patch compositional shift", loc="left", fontweight="bold")

    for axis in fig.axes:
        if hasattr(axis, "spines"):
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CODEX supports progressive mesenchymal occupation of Collagen-IV-rich scar fields", fontsize=15, fontweight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=128)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    patch_frames = []
    coupling_rows = []
    enrichment_rows = []
    sensitivity_coupling_rows = []
    sensitivity_enrichment_rows = []
    metadata_frames = []
    previews: dict[str, np.ndarray] = {}

    for sample, sample_metadata in SAMPLES.items():
        path = next((args.codex_root / sample).glob("*.qptiff"))
        channels = channel_metadata(path)
        channels.insert(0, "sample", sample)
        metadata_frames.append(channels)
        lookup = dict(zip(channels["biomarker"], channels["channel_index"], strict=True))
        indices = [lookup[marker] for marker in MARKERS]
        with tifffile.TiffFile(path) as handle:
            array = handle.series[0].levels[args.level].asarray(key=indices)
            preview_array = handle.series[0].levels[min(args.level + 1, len(handle.series[0].levels) - 1)].asarray(key=indices)
        images = {marker: image for marker, image in zip(MARKERS, array, strict=True)}
        preview_images = {marker: image for marker, image in zip(MARKERS, preview_array, strict=True)}
        mask = largest_tissue_mask(images["DAPI"], images["Vimentin"], images["PANCK"])
        preview_mask = largest_tissue_mask(preview_images["DAPI"], preview_images["Vimentin"], preview_images["PANCK"])
        previews[sample] = crop_to_mask(preview_composite(preview_images), preview_mask)

        sensitivity_sizes = sorted(set([96, args.patch_size, 160]))
        for patch_size in sensitivity_sizes:
            patches, shape = patch_table(images, mask, patch_size)
            patches.insert(0, "sample", sample)
            patches["fibrosis_stage"] = sample_metadata["fibrosis_stage"]
            patches["stage_numeric"] = sample_metadata["stage_numeric"]
            patches["grid_rows"] = shape[0]
            patches["grid_columns"] = shape[1]
            patches["patch_size"] = patch_size
            if patch_size == args.patch_size:
                patch_frames.append(patches)

            for outcome in OUTCOMES:
                result = fit_spatial_null(patches, outcome, shape)
                coupling_record = {
                    "sample": sample,
                    **sample_metadata,
                    "patch_size": patch_size,
                    "n_patches": len(patches),
                    **result,
                }
                sensitivity_coupling_rows.append(coupling_record)
                if patch_size == args.patch_size:
                    coupling_rows.append(coupling_record)
                values = patches[outcome] if outcome == "mesenchymal_endothelial_balance" else patches[f"z_{outcome}"]
                enrichment_record = {
                    "sample": sample,
                    **sample_metadata,
                    "patch_size": patch_size,
                    "outcome": outcome,
                    "scar_enrichment": float(values[patches["scar_patch"]].mean() - values[~patches["scar_patch"]].mean()),
                }
                sensitivity_enrichment_rows.append(enrichment_record)
                if patch_size == args.patch_size:
                    enrichment_rows.append(enrichment_record)

    patch_scores = pd.concat(patch_frames, ignore_index=True)
    coupling = pd.DataFrame(coupling_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    channel_table = pd.concat(metadata_frames, ignore_index=True)
    patch_scores.to_parquet(args.results_dir / "UQ_CODEX_patch_scores.parquet", index=False)
    coupling.to_csv(args.results_dir / "UQ_CODEX_spatial_coupling.tsv", sep="\t", index=False)
    enrichment.to_csv(args.results_dir / "UQ_CODEX_scar_enrichment.tsv", sep="\t", index=False)
    pd.DataFrame(sensitivity_coupling_rows).to_csv(
        args.results_dir / "UQ_CODEX_patch_size_sensitivity_coupling.tsv", sep="\t", index=False
    )
    pd.DataFrame(sensitivity_enrichment_rows).to_csv(
        args.results_dir / "UQ_CODEX_patch_size_sensitivity_enrichment.tsv", sep="\t", index=False
    )
    channel_table.to_csv(args.results_dir / "UQ_CODEX_channel_metadata.tsv", sep="\t", index=False)
    plot_results(previews, patch_scores, coupling, enrichment, args.figures_dir / "Figure_UQ_CODEX_spatial_protein_validation")
    print("\nSPATIAL COUPLING\n", coupling.to_string(index=False))
    print("\nSCAR ENRICHMENT\n", enrichment.to_string(index=False))


if __name__ == "__main__":
    main()
