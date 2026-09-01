#!/usr/bin/env python3
"""Test a SERPINE1-high/MMP-low proteolytic lock in the UQ MASLD Visium cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RESULTS = REPO_ROOT / "data/UQ_resolver_source_inputs"
VISIUM = REPO_ROOT / "data/raw_external/UQ_Visium"
HELPER = REPO_ROOT / "code/spatial/UQ_Visium_resolver_window_analysis.py"
OUTPUT = REPO_ROOT / "results/S10_UQ_spatial_proteolytic_lock"

GENES = [
    "SERPINE1", "SPON1", "ITGAV", "ITGB5", "MMP2", "MMP9", "MMP14",
    "PLAU", "PLAT", "TIMP1", "TIMP2",
]
STRUCTURAL_SCAR_GENES = [
    "COL1A1", "COL1A2", "COL3A1", "COL6A1", "DCN", "LUM", "SPARC",
    "THBS2", "LOXL2", "ACTA2", "TAGLN",
]
ALL_GENES = GENES + STRUCTURAL_SCAR_GENES
SCORES = {
    "mechanical_barrier": (["SERPINE1"], ["MMP2", "MMP9"]),
    "feedback_core": (["SERPINE1", "SPON1", "ITGAV", "ITGB5"], []),
    "proteolysis": (["MMP2", "MMP9", "MMP14"], []),
    "feedback_lock": (["SERPINE1", "SPON1", "ITGAV", "ITGB5"], ["MMP2", "MMP9", "MMP14"]),
    # Post-hoc refinement after the prespecified MMP-low scores failed in human tissue.
    "protease_activation_potential": (["PLAU", "PLAT", "MMP2", "MMP9", "MMP14"], []),
    "antiproteolytic_brake": (["SERPINE1", "TIMP1", "TIMP2"], []),
    "clearance_inefficiency": (["SERPINE1", "TIMP1", "TIMP2"], ["PLAU", "PLAT", "MMP2", "MMP9", "MMP14"]),
}
OUTCOMES = {
    "mechanical_barrier": ["HSC_identity", "Myeloid_identity", "log_library"],
    "feedback_core": ["HSC_identity", "log_library"],
    "proteolysis": ["HSC_identity", "Myeloid_identity", "log_library"],
    "feedback_lock": ["HSC_identity", "Myeloid_identity", "log_library"],
    "protease_activation_potential": ["HSC_identity", "Myeloid_identity", "log_library"],
    "antiproteolytic_brake": ["HSC_identity", "Myeloid_identity", "log_library"],
    "clearance_inefficiency": ["HSC_identity", "Myeloid_identity", "log_library"],
}
MODULE_TARGETS = ["HSC_retention", "TGFb_integrator", "Endo_resolver", "LAM_resolver"]
PALETTE = {
    "mechanical_barrier": "#B43E55",
    "feedback_core": "#C67A2E",
    "proteolysis": "#2F6F9F",
    "feedback_lock": "#7D3047",
    "protease_activation_potential": "#4D87A8",
    "antiproteolytic_brake": "#A45A52",
    "clearance_inefficiency": "#7F3C4D",
}


def load_helper():
    spec = importlib.util.spec_from_file_location("uq_resolver", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OUTCOMES = OUTCOMES
    return module


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def read_expression(path: Path) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    with h5py.File(path) as handle:
        matrix = handle["matrix"]
        names = [name.upper() for name in decode(matrix["features/name"][:])]
        barcodes = decode(matrix["barcodes"][:])
        shape = tuple(map(int, matrix["shape"][:]))
        sparse = csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=shape,
        )
    lookup = {}
    for index, name in enumerate(names):
        lookup.setdefault(name, index)
    missing = [gene for gene in ALL_GENES if gene not in lookup]
    if missing:
        raise ValueError(f"Missing genes in {path}: {missing}")
    selected = sparse[[lookup[gene] for gene in ALL_GENES], :].T.tocsr()
    total = np.asarray(sparse.sum(axis=0)).ravel().astype(float)
    normalized = selected.multiply((1e4 / np.maximum(total, 1))[:, None]).toarray()
    normalized = np.log1p(normalized)
    return barcodes, pd.DataFrame(normalized, index=barcodes, columns=ALL_GENES), total


def add_gene_scores(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    detection = []
    for array, metadata in base.groupby("array", sort=True, observed=True):
        barcodes, expression, _ = read_expression(VISIUM / array / "filtered_feature_bc_matrix.h5")
        local = metadata.copy().set_index("barcode")
        expression = expression.reindex(local.index)
        if expression.isna().any().any():
            raise ValueError(f"Barcode mismatch in {array}")
        for gene in ALL_GENES:
            values = expression[gene].to_numpy(float)
            sd = values.std(ddof=1)
            local[f"expr_{gene}"] = values
            local[f"z_{gene}"] = np.clip((values - values.mean()) / max(sd, 1e-8), -5, 5)
            if gene in GENES:
                detection.append(
                    {
                        "scope": "array",
                        "unit": array,
                        "array": array,
                        "gene": gene,
                        "n_spots": len(values),
                        "n_detected": int(np.sum(values > 0)),
                        "detection_fraction": float(np.mean(values > 0)),
                        "mean_log1p_cp10k": float(np.mean(values)),
                    }
                )
        for score, (positive, negative) in SCORES.items():
            value = local[[f"z_{gene}" for gene in positive]].mean(axis=1)
            if negative:
                value = value - local[[f"z_{gene}" for gene in negative]].mean(axis=1)
            local[score] = value
        local["structural_ECM_scar"] = local[[f"z_{gene}" for gene in STRUCTURAL_SCAR_GENES]].mean(axis=1)
        frames.append(local.reset_index())

    data = pd.concat(frames, ignore_index=True)
    data = data.rename(
        columns={
            "ECM_scar": "ECM_scar_original_with_TIMP1",
            "scar_core": "scar_core_original",
            "scar_proximity": "scar_proximity_original",
            "scar_quantile": "scar_quantile_original",
        }
    )
    rebuilt = []
    for _, local in data.groupby("biopsy_id", sort=False, observed=True):
        local = local.copy()
        threshold = local["structural_ECM_scar"].quantile(0.85)
        local["scar_core"] = local["structural_ECM_scar"].ge(threshold)
        coordinates = local[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
        scar = coordinates[local["scar_core"].to_numpy()]
        distance = cKDTree(scar).query(coordinates, k=1)[0]
        proximity = -distance
        local["scar_proximity"] = (proximity - proximity.mean()) / max(proximity.std(ddof=1), 1e-8)
        local["scar_quantile"] = 0.85
        local["ECM_scar"] = local["structural_ECM_scar"]
        rebuilt.append(local)
    data = pd.concat(rebuilt, ignore_index=True)
    for (biopsy, array), local in data.groupby(["biopsy_id", "array"], observed=True):
        for gene in GENES:
            values = local[f"expr_{gene}"].to_numpy(float)
            detection.append(
                {
                    "scope": "biopsy",
                    "unit": biopsy,
                    "array": array,
                    "gene": gene,
                    "n_spots": len(values),
                    "n_detected": int(np.sum(values > 0)),
                    "detection_fraction": float(np.mean(values > 0)),
                    "mean_log1p_cp10k": float(np.mean(values)),
                }
            )
    return data, pd.DataFrame(detection)


def residualize(local: pd.DataFrame, outcome: str, covariates: list[str]) -> np.ndarray:
    y = local[outcome].to_numpy(float)
    x = np.column_stack([np.ones(len(local))] + [local[name].to_numpy(float) for name in covariates])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def cross_module_correlations(data: pd.DataFrame, helper) -> pd.DataFrame:
    rows = []
    covariates = ["HSC_identity", "Endo_identity", "Myeloid_identity", "log_library"]
    for biopsy, local in data.groupby("biopsy_id", sort=True, observed=True):
        blocks = None
        for n_long, n_cross in [(8, 4), (6, 3), (5, 3)]:
            candidate = helper.make_blocks_extended(
                local,
                n_long,
                n_cross,
                ["mechanical_barrier", *MODULE_TARGETS, *covariates],
            ) if hasattr(helper, "make_blocks_extended") else None
            if candidate is not None and len(candidate) >= max(12, int(0.55 * n_long * n_cross)):
                blocks = candidate
                break
        if blocks is None:
            long_axis, cross_axis = helper.pca_coordinates(local)
            local = local.copy()
            local["long_bin"] = pd.qcut(pd.Series(long_axis).rank(method="first"), 6, labels=False).to_numpy()
            local["cross_bin"] = pd.qcut(pd.Series(cross_axis).rank(method="first"), 3, labels=False).to_numpy()
            blocks = local.groupby(["long_bin", "cross_bin"], observed=True)[
                ["mechanical_barrier", *MODULE_TARGETS, *covariates]
            ].mean().reset_index()
        x_resid = residualize(blocks, "mechanical_barrier", covariates)
        meta = local.iloc[0]
        for target in MODULE_TARGETS:
            y_resid = residualize(blocks, target, covariates)
            r = float(np.corrcoef(x_resid, y_resid)[0, 1])
            rows.append(
                {
                    "biopsy_id": biopsy,
                    "patient_cluster": meta["patient_cluster"],
                    "array": meta["array"],
                    "fibrosis_stage": meta["fibrosis_stage"],
                    "stage_numeric": meta["stage_numeric"],
                    "stage_group": meta["stage_group"],
                    "target": target,
                    "partial_r": r,
                    "fisher_z": float(np.arctanh(np.clip(r, -0.999999, 0.999999))),
                    "n_blocks": len(blocks),
                }
            )
    result = pd.DataFrame(rows)
    result["nominal_p"] = result.apply(
        lambda row: float(2 * stats.t.sf(abs(row.partial_r) * np.sqrt((row.n_blocks - 6) / max(1 - row.partial_r**2, 1e-8)), row.n_blocks - 6)),
        axis=1,
    )
    return result


def patient_summary(frame: pd.DataFrame, value: str, group: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260712)
    for name, local in frame.groupby(group, observed=True):
        patient = local.groupby("patient_cluster", observed=True)[value].mean().to_numpy(float)
        boot = np.empty(10000)
        for i in range(len(boot)):
            boot[i] = np.median(rng.choice(patient, len(patient), replace=True))
        nonzero = patient[np.abs(patient) > 1e-12]
        wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else np.nan
        rows.append(
            {
                group: name,
                "n_patients": len(patient),
                "median": float(np.median(patient)),
                "mean": float(np.mean(patient)),
                "positive_fraction": float(np.mean(patient > 0)),
                "bootstrap_median_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_median_ci_high": float(np.quantile(boot, 0.975)),
                "wilcoxon_signed_rank_p": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)


def stage_trends(coupling: pd.DataFrame, cross: pd.DataFrame, data: pd.DataFrame, helper) -> pd.DataFrame:
    rows = []
    for outcome, local in coupling.groupby("outcome", observed=True):
        rows.append({"analysis": "scar_coupling", "feature": outcome, **helper.array_adjusted_trend(local, "beta")})
    for target, local in cross.groupby("target", observed=True):
        rows.append({"analysis": "barrier_partial_correlation", "feature": target, **helper.array_adjusted_trend(local, "fisher_z")})
    means = data.groupby(
        ["biopsy_id", "patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group"],
        observed=True,
    )[[*OUTCOMES, "ECM_scar"]].mean().reset_index()
    for outcome in [*OUTCOMES, "ECM_scar"]:
        rows.append({"analysis": "biopsy_mean", "feature": outcome, **helper.array_adjusted_trend(means, outcome)})
    return pd.DataFrame(rows)


def plot_results(data: pd.DataFrame, coupling: pd.DataFrame, cross: pd.DataFrame, summary: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(14.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.08])

    ax = fig.add_subplot(grid[0, 0])
    primary = coupling.loc[coupling.outcome.eq("mechanical_barrier")].copy()
    sns.boxplot(data=primary, x="stage_numeric", y="beta", color="#F1C7CE", width=0.58, fliersize=0, ax=ax)
    sns.stripplot(data=primary, x="stage_numeric", y="beta", hue="spatial_field_p", palette="viridis_r", size=5, ax=ax)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set(xlabel="Fibrosis stage", ylabel="Scar-proximity beta")
    ax.set_title("A  Human scar coupling of the mechanical barrier", loc="left", fontweight="bold")
    if ax.legend_:
        ax.legend_.remove()

    ax = fig.add_subplot(grid[0, 1])
    order = [
        "mechanical_barrier", "feedback_lock", "clearance_inefficiency",
        "antiproteolytic_brake", "protease_activation_potential",
        "feedback_core", "proteolysis",
    ]
    s = summary.set_index("outcome").loc[order].reset_index()
    y = np.arange(len(s))
    for yi, row in s.iterrows():
        ax.errorbar(
            row["median"],
            yi,
            xerr=[[row["median"] - row["bootstrap_median_ci_low"]], [row["bootstrap_median_ci_high"] - row["median"]]],
            fmt="o",
            color=PALETTE[row["outcome"]],
            ecolor=PALETTE[row["outcome"]],
            capsize=3,
        )
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_yticks(y, [name.replace("_", " ") for name in s.outcome])
    ax.invert_yaxis()
    ax.set_xlabel("Patient-level median scar-proximity beta (95% bootstrap CI)")
    ax.set_title("B  Scar coupling indicates turnover with an inhibitory brake", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[1, 0])
    target_order = ["HSC_retention", "TGFb_integrator", "Endo_resolver", "LAM_resolver"]
    sns.boxplot(data=cross, y="target", x="partial_r", order=target_order, color="#D7E1E7", fliersize=0, ax=ax)
    sns.stripplot(data=cross, y="target", x="partial_r", order=target_order, color="#334E5C", size=3.2, alpha=0.65, ax=ax)
    ax.axvline(0, color="#777", lw=0.8)
    ax.set(xlabel="Biopsy-level partial correlation", ylabel="")
    ax.set_title("C  Barrier coupling after lineage and library adjustment", loc="left", fontweight="bold")

    holder = fig.add_subplot(grid[1, 1])
    holder.set_axis_off()
    holder.set_title("D  Exploratory clearance-inefficiency fields", loc="left", fontweight="bold")
    exploratory = coupling.loc[coupling.outcome.eq("clearance_inefficiency")].copy()
    representatives = []
    for group in ["early", "intermediate", "late"]:
        local = exploratory.loc[exploratory.stage_group.eq(group)]
        target = local.beta.median()
        representatives.append(local.loc[(local.beta - target).abs().idxmin(), "biopsy_id"])
    positions = [(0.02, 0.08, 0.30, 0.79), (0.35, 0.08, 0.30, 0.79), (0.68, 0.08, 0.30, 0.79)]
    for biopsy, position in zip(representatives, positions):
        inset = holder.inset_axes(position)
        local = data.loc[data.biopsy_id.eq(biopsy)]
        inset.scatter(
            local.pxl_col_in_fullres,
            -local.pxl_row_in_fullres,
            c=local.clearance_inefficiency,
            cmap="RdBu_r",
            vmin=-2.5,
            vmax=2.5,
            s=5,
            linewidths=0,
            rasterized=True,
        )
        row = local.iloc[0]
        inset.set_title(f"{row.stage_group}\n{row.fibrosis_stage} | {biopsy}", fontsize=8)
        inset.set_aspect("equal")
        inset.set_axis_off()
    holder.text(0.5, 0.0, "Blue: activation potential   Red: SERPINE1/TIMP-dominant brake", ha="center", fontsize=8)

    for ax in fig.axes[:3]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Human spatial data refine proteolytic shutdown into clearance inefficiency", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT / "UQ_Visium_spatial_proteolytic_lock.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT / "UQ_Visium_spatial_proteolytic_lock.pdf", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results",type=Path,default=SOURCE_RESULTS)
    parser.add_argument("--visium-dir",type=Path,default=VISIUM)
    parser.add_argument("--helper-script",type=Path,default=HELPER)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    global SOURCE_RESULTS,VISIUM,HELPER,OUTPUT
    args=parse_args()
    SOURCE_RESULTS,VISIUM,HELPER,OUTPUT=args.source_results,args.visium_dir,args.helper_script,args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    helper = load_helper()
    base = pd.read_parquet(SOURCE_RESULTS / "UQ_Visium_spot_module_scores.parquet")
    data, detection = add_gene_scores(base)
    coupling = helper.fit_biopsies(data)
    cross = cross_module_correlations(data, helper)
    coupling_summary = patient_summary(coupling, "beta", "outcome")
    cross_summary = patient_summary(cross, "partial_r", "target")
    trends = stage_trends(coupling, cross, data, helper)
    coupling_summary["wilcoxon_fdr"] = helper.bh(coupling_summary["wilcoxon_signed_rank_p"])
    cross_summary["wilcoxon_fdr"] = helper.bh(cross_summary["wilcoxon_signed_rank_p"])
    trends["permutation_fdr_within_analysis"] = trends.groupby("analysis", observed=True)[
        "stage_beta_within_array_permutation_p"
    ].transform(helper.bh)

    data.to_parquet(OUTPUT / "UQ_proteolytic_lock_spot_scores.parquet", index=False)
    detection.to_csv(OUTPUT / "UQ_proteolytic_lock_gene_detection.tsv", sep="\t", index=False)
    coupling.to_csv(OUTPUT / "UQ_proteolytic_lock_biopsy_coupling.tsv", sep="\t", index=False)
    coupling_summary.to_csv(OUTPUT / "UQ_proteolytic_lock_patient_summary.tsv", sep="\t", index=False)
    cross.to_csv(OUTPUT / "UQ_proteolytic_lock_cross_module_correlations.tsv", sep="\t", index=False)
    cross_summary.to_csv(OUTPUT / "UQ_proteolytic_lock_cross_module_summary.tsv", sep="\t", index=False)
    trends.to_csv(OUTPUT / "UQ_proteolytic_lock_stage_trends.tsv", sep="\t", index=False)
    plot_results(data, coupling, cross, coupling_summary)

    primary = coupling_summary.loc[coupling_summary.outcome.eq("mechanical_barrier")].iloc[0]
    exploratory = coupling_summary.loc[coupling_summary.outcome.eq("clearance_inefficiency")].iloc[0]
    exploratory_trend = trends.loc[
        trends.analysis.eq("scar_coupling") & trends.feature.eq("clearance_inefficiency")
    ].iloc[0]
    summary = {
        "n_spots": int(len(data)),
        "n_biopsies": int(data.biopsy_id.nunique()),
        "n_patients": int(data.patient_cluster.nunique()),
        "n_arrays": int(data.array.nunique()),
        "primary_score": "z(SERPINE1)-mean[z(MMP2),z(MMP9)]",
        "primary_scar_coupling": primary.to_dict(),
        "exploratory_clearance_inefficiency": {
            "post_hoc": True,
            "score": "mean[z(SERPINE1),z(TIMP1),z(TIMP2)]-mean[z(PLAU),z(PLAT),z(MMP2),z(MMP9),z(MMP14)]",
            "scar_coupling": exploratory.to_dict(),
            "stage_trend": exploratory_trend.to_dict(),
        },
        "claim_boundary": "Spatial association after lineage and library adjustment; not cell-specific causality.",
    }
    (OUTPUT / "UQ_proteolytic_lock_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
