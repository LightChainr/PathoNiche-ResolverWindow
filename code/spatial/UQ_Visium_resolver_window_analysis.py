#!/usr/bin/env python3
"""Test scar-proximal retention/resolution balance across human MASLD fibrosis stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree


MODULES = {
    "ECM_scar": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "DCN", "LUM", "SPARC", "THBS2", "TIMP1", "LOXL2", "ACTA2", "TAGLN"],
    "HSC_identity": ["RGS5", "PDGFRB", "CSPG4", "DES", "COL6A1", "NOTCH3"],
    "Endo_identity": ["PECAM1", "CDH5", "RAMP2", "EGFL7", "ENG", "ERG", "TEK", "ROBO4"],
    "Myeloid_identity": ["LST1", "TYROBP", "FCER1G", "AIF1", "CTSS", "SPI1", "CD68"],
    "HSC_retention": ["ITGB5", "GAS7", "SPON1", "SVEP1", "PDGFRA", "LAMA2", "EPHA3", "DPYSL3", "LTBP2", "EFEMP1", "SERPINE1"],
    "Endo_resolver": ["WNT9B", "WNT7B", "VWF", "ACE", "KDR", "ESAM", "EMCN", "PLVAP", "KLF2", "NR2F2"],
    "LAM_resolver": ["TREM2", "GPNMB", "LPL", "MMP14", "CTSB", "CTSD", "AXL", "MERTK"],
    "AP1_control": ["FOS", "JUN", "JUNB", "JUND", "FOSL1", "FOSL2"],
    "TGFb_integrator": ["TGFB1", "TGFBR1", "TGFBR2", "SMAD3", "SMAD7", "CTGF", "CCN2", "THBS1"],
}

OUTCOMES = {
    "HSC_retention": ["HSC_identity", "log_library"],
    "Endo_resolver": ["HSC_identity", "Endo_identity", "log_library"],
    "LAM_resolver": ["HSC_identity", "Myeloid_identity", "log_library"],
    "retention_dominance": ["HSC_identity", "Endo_identity", "Myeloid_identity", "log_library"],
    "AP1_control": ["HSC_identity", "log_library"],
    "TGFb_integrator": ["HSC_identity", "Myeloid_identity", "log_library"],
}

STAGE_ORDER = ["F0", "F1", "F2", "F3a", "F3b", "F4"]
GROUP_ORDER = ["early", "intermediate", "late"]
COLORS = {
    "HSC_retention": "#C7472E",
    "Endo_resolver": "#247BA0",
    "LAM_resolver": "#4C956C",
    "retention_dominance": "#7A1F3D",
    "AP1_control": "#D88C1D",
    "TGFb_integrator": "#6C5B7B",
}


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.clip(q, 0, 1)
    return result


def read_module_matrix(path: Path, genes: list[str]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    with h5py.File(path) as handle:
        matrix = handle["matrix"]
        names = [name.upper() for name in decode(matrix["features/name"][:])]
        barcodes = decode(matrix["barcodes"][:])
        shape = tuple(map(int, matrix["shape"][:]))
        sparse = csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=shape,
        )
    lookup: dict[str, int] = {}
    for index, name in enumerate(names):
        lookup.setdefault(name, index)
    present = [gene for gene in genes if gene in lookup]
    selected = sparse[[lookup[gene] for gene in present], :].T.tocsr()
    total = np.asarray(sparse.sum(axis=0)).ravel().astype(float)
    normalized = selected.multiply((1e4 / np.maximum(total, 1))[:, None]).toarray()
    normalized = np.log1p(normalized)
    return barcodes, present, normalized, total


def explode_biopsy_map(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in mapping.to_dict("records"):
        for component in str(record["component_members"]).split(","):
            rows.append({**record, "component": component})
    return pd.DataFrame(rows).drop(columns="component_members")


def score_array(
    array: str,
    visium_root: Path,
    component_spots: pd.DataFrame,
    expanded_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    positions = pd.read_csv(visium_root / array / "spatial/tissue_positions.csv")
    positions = positions.loc[positions["in_tissue"].eq(1)].copy()
    component = component_spots.loc[component_spots["array"].eq(array), ["barcode", "component"]]
    metadata = positions.merge(component, on="barcode", how="left")
    metadata["array"] = array
    metadata = metadata.merge(
        expanded_map.loc[expanded_map["array"].eq(array)],
        on=["array", "component"],
        how="left",
        validate="many_to_one",
    )

    labelled = metadata.dropna(subset=["biopsy_id"])
    missing = metadata["biopsy_id"].isna()
    rescued = 0
    if missing.any() and len(labelled):
        tree = cKDTree(labelled[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float))
        distance, nearest = tree.query(metadata.loc[missing, ["pxl_col_in_fullres", "pxl_row_in_fullres"]], k=1)
        max_distance = np.median(tree.query(labelled[["pxl_col_in_fullres", "pxl_row_in_fullres"]], k=2)[0][:, 1]) * 5
        rescue_index = metadata.index[missing][distance <= max_distance]
        source_index = labelled.index.to_numpy()[nearest[distance <= max_distance]]
        transfer = ["component", "biopsy_id", "fibrosis_stage", "stage_numeric", "stage_group", "patient_cluster", "mapping_source"]
        metadata.loc[rescue_index, transfer] = labelled.loc[source_index, transfer].to_numpy()
        rescued = len(rescue_index)

    union = list(dict.fromkeys(gene for module in MODULES.values() for gene in module))
    barcodes, present, values, total = read_module_matrix(
        visium_root / array / "filtered_feature_bc_matrix.h5", union
    )
    expression = pd.DataFrame(values, index=barcodes, columns=present)
    metadata = metadata.set_index("barcode").reindex(barcodes)
    keep = metadata["biopsy_id"].notna().to_numpy()
    metadata = metadata.loc[keep].copy()
    expression = expression.loc[keep].copy()
    total = total[keep]
    metadata["log_library"] = np.log1p(total)

    mapped: dict[str, list[str]] = {}
    for module, requested in MODULES.items():
        genes = [gene for gene in requested if gene in expression.columns]
        genes = list(dict.fromkeys(genes))
        mapped[module] = genes
        local = expression[genes].to_numpy(float)
        mean = local.mean(axis=0)
        sd = local.std(axis=0, ddof=1)
        usable = sd > 1e-8
        z = (local[:, usable] - mean[usable]) / sd[usable]
        metadata[module] = np.clip(z, -5, 5).mean(axis=1)
    metadata["retention_dominance"] = metadata["HSC_retention"] - 0.5 * (
        metadata["Endo_resolver"] + metadata["LAM_resolver"]
    )
    metadata["array"] = array
    audit = {
        "matrix_barcodes": len(barcodes),
        "mapped_spots": len(metadata),
        "unmapped_spots": len(barcodes) - len(metadata),
        "rescued_small_fragment_spots": rescued,
    }
    return metadata.reset_index(names="barcode"), mapped, audit


def add_scar_proximity(data: pd.DataFrame, scar_quantile: float = 0.85) -> pd.DataFrame:
    frames = []
    for _, local in data.groupby("biopsy_id", sort=False, observed=True):
        local = local.copy()
        threshold = local["ECM_scar"].quantile(scar_quantile)
        local["scar_core"] = local["ECM_scar"].ge(threshold)
        coordinates = local[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
        scar = coordinates[local["scar_core"].to_numpy()]
        distance = cKDTree(scar).query(coordinates, k=1)[0]
        proximity = -distance
        local["scar_proximity"] = (proximity - proximity.mean()) / max(proximity.std(ddof=1), 1e-8)
        local["scar_quantile"] = scar_quantile
        frames.append(local)
    return pd.concat(frames, ignore_index=True)


def pca_coordinates(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coordinates = data[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    centered = coordinates - coordinates.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    return rotated[:, 0], rotated[:, 1]


def make_blocks(data: pd.DataFrame, n_long: int, n_cross: int) -> pd.DataFrame:
    local = data.copy()
    long_axis, cross_axis = pca_coordinates(local)
    local["long_bin"] = pd.qcut(
        pd.Series(long_axis).rank(method="first"), n_long, labels=False
    ).to_numpy()
    local["cross_bin"] = pd.qcut(
        pd.Series(cross_axis).rank(method="first"), n_cross, labels=False
    ).to_numpy()
    columns = ["scar_proximity", *OUTCOMES.keys(), "HSC_identity", "Endo_identity", "Myeloid_identity", "log_library"]
    return local.groupby(["long_bin", "cross_bin"], observed=True)[columns].mean().reset_index()


def nearest_fill(grid: np.ndarray) -> np.ndarray:
    filled = grid.copy()
    missing = np.argwhere(~np.isfinite(filled))
    valid = np.argwhere(np.isfinite(filled))
    if len(missing):
        tree = cKDTree(valid)
        _, nearest = tree.query(missing, k=1)
        filled[missing[:, 0], missing[:, 1]] = filled[valid[nearest, 0], valid[nearest, 1]]
    return filled


def rectangular_transforms(n_long: int, n_cross: int) -> list[np.ndarray]:
    grid = np.arange(n_long * n_cross).reshape(n_long, n_cross)
    forms = [grid, np.flipud(grid), np.fliplr(grid), np.flipud(np.fliplr(grid))]
    identity = tuple(grid.ravel())
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for form in forms:
        for dx in range(n_long):
            for dy in range(n_cross):
                index = np.roll(np.roll(form, dx, axis=0), dy, axis=1).ravel()
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
    return float(np.sum(centered * lag) / (4 * denominator)) if denominator > 0 else np.nan


def fit_spatial_field_null(
    blocks: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    n_long: int,
    n_cross: int,
) -> dict[str, float]:
    y = blocks[outcome].to_numpy(float)
    nuisance = np.column_stack([np.ones(len(blocks))] + [blocks[name].to_numpy(float) for name in covariates])
    proximity = blocks["scar_proximity"].to_numpy(float)
    full = np.column_stack([nuisance, proximity])
    coefficients, _, _, _ = np.linalg.lstsq(full, y, rcond=None)
    residual = y - full @ coefficients
    dof = max(len(y) - full.shape[1], 1)
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(full.T @ full)
    beta = float(coefficients[-1])
    se = float(np.sqrt(max(covariance[-1, -1], 0)))

    nuisance_coefficients = np.linalg.lstsq(nuisance, y, rcond=None)[0]
    fitted = nuisance @ nuisance_coefficients
    field = y - fitted
    flat_position = blocks["long_bin"].to_numpy(int) * n_cross + blocks["cross_bin"].to_numpy(int)
    grid = np.full((n_long, n_cross), np.nan)
    grid.ravel()[flat_position] = field
    grid = nearest_fill(grid)
    transforms = rectangular_transforms(n_long, n_cross)
    null = np.empty(len(transforms))
    moran = np.empty(len(transforms))
    for i, index in enumerate(transforms):
        transformed = grid.ravel()[index][flat_position]
        null[i] = np.linalg.lstsq(full, fitted + transformed, rcond=None)[0][-1]
        moran[i] = toroidal_moran(grid.ravel()[index].reshape(n_long, n_cross))
    p = float((1 + np.sum(np.abs(null) >= abs(beta))) / (len(null) + 1))
    return {
        "beta": beta,
        "beta_se": se,
        "beta_ci_low": beta - 1.96 * se,
        "beta_ci_high": beta + 1.96 * se,
        "spatial_field_p": p,
        "n_transforms": len(transforms),
        "n_blocks": len(blocks),
        "grid_occupancy": len(blocks) / (n_long * n_cross),
        "residual_moran": toroidal_moran(grid),
        "permuted_moran_min": float(np.min(moran)),
        "permuted_moran_max": float(np.max(moran)),
    }


def fit_biopsies(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for biopsy, local in data.groupby("biopsy_id", sort=True, observed=True):
        candidates = [(8, 4), (6, 3), (5, 3)]
        blocks = None
        chosen = None
        for n_long, n_cross in candidates:
            candidate = make_blocks(local, n_long, n_cross)
            if len(candidate) >= max(12, int(0.55 * n_long * n_cross)):
                blocks = candidate
                chosen = (n_long, n_cross)
                break
        if blocks is None or chosen is None:
            continue
        metadata = local.iloc[0]
        for outcome, covariates in OUTCOMES.items():
            result = fit_spatial_field_null(blocks, outcome, covariates, *chosen)
            rows.append(
                {
                    "biopsy_id": biopsy,
                    "patient_cluster": metadata["patient_cluster"],
                    "array": metadata["array"],
                    "fibrosis_stage": metadata["fibrosis_stage"],
                    "stage_numeric": metadata["stage_numeric"],
                    "stage_group": metadata["stage_group"],
                    "n_spots": len(local),
                    "outcome": outcome,
                    "covariates": ";".join(covariates),
                    "grid_long": chosen[0],
                    "grid_cross": chosen[1],
                    **result,
                }
            )
    results = pd.DataFrame(rows)
    results["spatial_fdr_within_outcome"] = results.groupby("outcome", observed=True)["spatial_field_p"].transform(bh)
    return results


def array_adjusted_trend(frame: pd.DataFrame, value: str, permutations: int = 9999) -> dict[str, float]:
    patient = (
        frame.groupby(["patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group"], observed=True)[value]
        .mean()
        .reset_index()
    )
    array_dummies = pd.get_dummies(patient["array"], drop_first=True, dtype=float)
    x_reduced = np.column_stack([np.ones(len(patient)), array_dummies.to_numpy(float)])
    stage = patient["stage_numeric"].to_numpy(float)
    x_full = np.column_stack([x_reduced, stage])
    y = patient[value].to_numpy(float)
    fit = sm.OLS(y, x_full).fit(cov_type="HC3")
    observed = float(fit.params[-1])

    reduced_coef = np.linalg.lstsq(x_reduced, y, rcond=None)[0]
    fitted = x_reduced @ reduced_coef
    residual = y - fitted
    rng = np.random.default_rng(20260711)
    null = np.empty(permutations)
    groups = [indices.to_numpy() for _, indices in patient.groupby("array", observed=True).groups.items()]
    for i in range(permutations):
        permuted = residual.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        null[i] = np.linalg.lstsq(x_full, fitted + permuted, rcond=None)[0][-1]
    permutation_p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1))
    rho, spearman_p = stats.spearmanr(patient["stage_numeric"], patient[value])
    return {
        "n_patients": len(patient),
        "stage_beta_array_adjusted": observed,
        "stage_beta_hc3_se": float(fit.bse[-1]),
        "stage_beta_hc3_p": float(fit.pvalues[-1]),
        "stage_beta_within_array_permutation_p": permutation_p,
        "spearman_rho": float(rho),
        "spearman_p": float(spearman_p),
    }


def trend_tables(coupling: pd.DataFrame, spot_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coupling_rows = []
    for outcome, local in coupling.groupby("outcome", observed=True):
        coupling_rows.append({"outcome": outcome, **array_adjusted_trend(local, "beta")})
    coupling_trends = pd.DataFrame(coupling_rows)

    means = spot_scores.groupby(
        ["biopsy_id", "patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group"], observed=True
    )[[*OUTCOMES.keys(), "ECM_scar"]].mean().reset_index()
    mean_rows = []
    for outcome in [*OUTCOMES.keys(), "ECM_scar"]:
        mean_rows.append({"outcome": outcome, **array_adjusted_trend(means, outcome)})
    return coupling_trends, pd.DataFrame(mean_rows), means


def choose_representatives(coupling: pd.DataFrame) -> list[str]:
    dominance = coupling.loc[coupling["outcome"].eq("retention_dominance")].copy()
    selected = []
    for group in GROUP_ORDER:
        local = dominance.loc[dominance["stage_group"].eq(group)].copy()
        target = local["beta"].median()
        selected.append(local.loc[(local["beta"] - target).abs().idxmin(), "biopsy_id"])
    return selected


def plot_main(data: pd.DataFrame, coupling: pd.DataFrame, coupling_trends: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.05, 1.0])
    representatives = choose_representatives(coupling)
    stage_palette = {"early": "#4E79A7", "intermediate": "#E0A12B", "late": "#A23B3B"}

    for column, biopsy in enumerate(representatives):
        ax = fig.add_subplot(grid[0, column])
        local = data.loc[data["biopsy_id"].eq(biopsy)]
        scatter = ax.scatter(
            local["pxl_col_in_fullres"],
            -local["pxl_row_in_fullres"],
            c=local["retention_dominance"],
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            s=8,
            linewidths=0,
            rasterized=True,
        )
        row = local.iloc[0]
        ax.set_title(f"{row['stage_group'].title()} | {row['fibrosis_stage']} | {biopsy}", fontweight="bold")
        ax.set_aspect("equal")
        ax.set_axis_off()
        if column == 0:
            ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=15, fontweight="bold")
    colorbar = fig.colorbar(scatter, ax=[fig.axes[0], fig.axes[1], fig.axes[2]], shrink=0.72, pad=0.01)
    colorbar.set_label("Retention dominance score")

    ax = fig.add_subplot(grid[1, :2])
    selected_outcomes = ["HSC_retention", "Endo_resolver", "LAM_resolver", "retention_dominance"]
    plot = coupling.loc[coupling["outcome"].isin(selected_outcomes)].copy()
    plot["stage_group"] = pd.Categorical(plot["stage_group"], GROUP_ORDER, ordered=True)
    sns.boxplot(
        data=plot,
        x="stage_group",
        y="beta",
        hue="outcome",
        hue_order=selected_outcomes,
        palette=COLORS,
        showfliers=False,
        width=0.72,
        ax=ax,
    )
    sns.stripplot(
        data=plot,
        x="stage_group",
        y="beta",
        hue="outcome",
        hue_order=selected_outcomes,
        dodge=True,
        palette=COLORS,
        size=2.7,
        alpha=0.65,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Scar-proximity coupling beta")
    ax.set_title("B  Fibrosis stage shifts the scar-coupled control balance", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = fig.add_subplot(grid[1, 2])
    dominance = coupling.loc[coupling["outcome"].eq("retention_dominance")].copy()
    model = sm.OLS(dominance["beta"], sm.add_constant(dominance["stage_numeric"].astype(float))).fit()
    line_x = np.linspace(0, 4, 100)
    prediction = model.get_prediction(sm.add_constant(line_x)).summary_frame()
    ax.plot(line_x, prediction["mean"], color=COLORS["retention_dominance"], lw=2)
    ax.fill_between(
        line_x,
        prediction["mean_ci_lower"].to_numpy(float),
        prediction["mean_ci_upper"].to_numpy(float),
        color=COLORS["retention_dominance"],
        alpha=0.15,
        linewidth=0,
    )
    offsets = {"F3a": -0.11, "F3b": 0.11}
    rng = np.random.default_rng(20260711)
    point_x = dominance["stage_numeric"].astype(float).to_numpy(copy=True)
    point_x += dominance["fibrosis_stage"].astype(str).map(offsets).fillna(0).to_numpy(float)
    point_x += rng.uniform(-0.035, 0.035, len(dominance))
    ax.scatter(
        point_x,
        dominance["beta"],
        c=dominance["stage_group"].astype(str).map(stage_palette),
        s=27,
        linewidths=0,
        alpha=0.9,
    )
    trend = coupling_trends.set_index("outcome").loc["retention_dominance"]
    ax.text(
        0.03,
        0.97,
        f"array-adjusted beta={trend['stage_beta_array_adjusted']:.3f}\npermutation P={trend['stage_beta_within_array_permutation_p']:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks([0, 1, 2, 3, 4], ["F0", "F1", "F2", "F3a / F3b", "F4"])
    ax.set_xlabel("Fibrosis stage")
    ax.set_ylabel("Retention-dominance coupling")
    ax.set_title("C  Closing resolver window", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[2, :2])
    heat = coupling.pivot(index="biopsy_id", columns="outcome", values="beta")
    annotation = coupling.drop_duplicates("biopsy_id").set_index("biopsy_id")
    order = annotation.sort_values(["stage_numeric", "array"]).index
    columns = ["HSC_retention", "Endo_resolver", "LAM_resolver", "retention_dominance", "AP1_control", "TGFb_integrator"]
    heat = heat.loc[order, columns]
    sns.heatmap(heat.T, cmap="vlag", center=0, robust=True, xticklabels=False, cbar_kws={"label": "Coupling beta"}, ax=ax)
    boundaries = np.where(np.diff(annotation.loc[order, "stage_numeric"].to_numpy()) != 0)[0] + 1
    for boundary in boundaries:
        ax.axvline(boundary, color="white", lw=1.5)
    ax.set_xlabel("33 biopsies ordered by fibrosis stage")
    ax.set_ylabel("")
    ax.set_title("D  Coordinated spatial control architecture", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[2, 2])
    pplot = coupling.loc[coupling["outcome"].isin(selected_outcomes)].copy()
    summary = pplot.groupby(["stage_group", "outcome"], observed=True)["spatial_field_p"].apply(lambda x: (x < 0.05).mean()).reset_index(name="fraction")
    summary["stage_group"] = pd.Categorical(summary["stage_group"], GROUP_ORDER, ordered=True)
    sns.barplot(data=summary, x="stage_group", y="fraction", hue="outcome", hue_order=selected_outcomes, palette=COLORS, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction spatial-null P < 0.05")
    ax.set_title("E  Autocorrelation-preserving support", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    for axis in fig.axes:
        if hasattr(axis, "spines"):
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Human MASLD progression shifts the scar niche from resolution toward stellate-cell retention",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visium-root", type=Path, required=True)
    parser.add_argument("--component-map", type=Path, required=True)
    parser.add_argument("--biopsy-map", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(args.biopsy_map, sep="\t")
    mapping["fibrosis_stage"] = pd.Categorical(mapping["fibrosis_stage"], STAGE_ORDER, ordered=True)
    mapping["stage_group"] = pd.Categorical(mapping["stage_group"], GROUP_ORDER, ordered=True)
    expanded = explode_biopsy_map(mapping)
    component_spots = pd.read_parquet(args.component_map)
    frames = []
    audits = []
    module_mapping: dict[str, set[str]] = {name: set() for name in MODULES}
    for array in mapping["array"].drop_duplicates():
        scores, mapped, audit = score_array(array, args.visium_root, component_spots, expanded)
        frames.append(scores)
        audits.append({"array": array, **audit})
        for module, genes in mapped.items():
            module_mapping[module].update(genes)
    scored = pd.concat(frames, ignore_index=True)
    data = add_scar_proximity(scored, 0.85)
    coupling = fit_biopsies(data)
    coupling_trends, mean_trends, biopsy_means = trend_tables(coupling, data)

    sensitivity_frames = []
    sensitivity_trends = []
    for scar_quantile in [0.80, 0.85, 0.90]:
        local_data = data if scar_quantile == 0.85 else add_scar_proximity(scored, scar_quantile)
        local_coupling = coupling if scar_quantile == 0.85 else fit_biopsies(local_data)
        local_coupling = local_coupling.copy()
        local_coupling["scar_quantile"] = scar_quantile
        local_trends, _, _ = trend_tables(local_coupling, local_data)
        local_trends["scar_quantile"] = scar_quantile
        sensitivity_frames.append(local_coupling)
        sensitivity_trends.append(local_trends)

    data.to_parquet(args.results_dir / "UQ_Visium_spot_module_scores.parquet", index=False)
    coupling.to_csv(args.results_dir / "UQ_Visium_biopsy_spatial_coupling.tsv", sep="\t", index=False)
    coupling_trends.to_csv(args.results_dir / "UQ_Visium_coupling_stage_trends.tsv", sep="\t", index=False)
    pd.concat(sensitivity_frames, ignore_index=True).to_csv(
        args.results_dir / "UQ_Visium_scar_threshold_sensitivity_coupling.tsv", sep="\t", index=False
    )
    pd.concat(sensitivity_trends, ignore_index=True).to_csv(
        args.results_dir / "UQ_Visium_scar_threshold_sensitivity_trends.tsv", sep="\t", index=False
    )
    mean_trends.to_csv(args.results_dir / "UQ_Visium_module_mean_stage_trends.tsv", sep="\t", index=False)
    biopsy_means.to_csv(args.results_dir / "UQ_Visium_biopsy_module_means.tsv", sep="\t", index=False)
    pd.DataFrame(audits).to_csv(args.results_dir / "UQ_Visium_mapping_audit.tsv", sep="\t", index=False)
    (args.results_dir / "UQ_Visium_module_gene_mapping.json").write_text(
        json.dumps({name: sorted(genes) for name, genes in module_mapping.items()}, indent=2) + "\n"
    )
    plot_main(data, coupling, coupling_trends, args.figures_dir / "Figure_UQ_MASLD_resolver_window_validation")
    print("\nCOUPLING STAGE TRENDS\n", coupling_trends.to_string(index=False))
    print("\nMODULE-MEAN STAGE TRENDS\n", mean_trends.to_string(index=False))
    print("\nMAPPING AUDIT\n", pd.DataFrame(audits).to_string(index=False))


if __name__ == "__main__":
    main()
