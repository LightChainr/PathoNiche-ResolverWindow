#!/usr/bin/env python3
"""Rebuild reviewer-facing replacement panels from frozen derived outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "figures/reviewer_revision"

FIGURE_DATA = ROOT / "data/figure_reconstruction"
UQ_DERIVED = ROOT / "data/UQ_spatial_derived"
HSC_TABLE = FIGURE_DATA / "integrated_retention_candidates.tsv"
MAC_TABLE = FIGURE_DATA / "LAM_priority_gene_fates.tsv"
UQ_CODEX = FIGURE_DATA
PHASE_DIR = UQ_DERIVED / "S11_UQ_phase_basin_control"
MOUSE_DIR = UQ_DERIVED / "S13_mouse_resolver_lock_trajectory"
DRUG_DIR = UQ_DERIVED / "S12_sequential_basin_switch_drugs"

RETENTION = "#B23A48"
REPAIR = "#1B7F79"
TRANSITION = "#D9A441"
INK = "#24313A"
MUTED = "#68757D"
GRID = "#DCE2E5"
LIGHT = "#F3F5F6"
STAGE_COLORS = {0: "#4E9F8E", 1: "#7BB8A5", 2: "#D6AA52", 3: "#D77A57", 4: RETENTION}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#A9B2B7",
            "axes.linewidth": 0.7,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 360},
    }.items():
        fig.savefig(OUT / f"{stem}{suffix}", **kwargs)
    plt.close(fig)


def status_human(row: pd.Series) -> int:
    if bool(row["human_fdr_any_010"]):
        return 2
    if bool(row["human_nominal_both"]):
        return 1
    return 0


def build_fig3d_categorical_evidence() -> None:
    hsc = pd.read_csv(HSC_TABLE, sep="\t").set_index("gene_upper")
    mac = pd.read_csv(MAC_TABLE, sep="\t").set_index("gene")
    hsc_genes = ["ITGB5", "GAS7", "SPON1", "SVEP1"]
    mac_genes = ["Trem2", "Gpnmb", "Lpl", "Mmp14", "Spp1"]
    columns = [
        "Human HSC\ndisease increase",
        "Published HSC\nATAC + RNA",
        "Mouse injury\nincrease",
        "Residual after\nwithdrawal",
        "Direct functional\nsupport",
    ]
    rows: list[dict[str, object]] = []
    for gene in hsc_genes:
        row = hsc.loc[gene]
        rows.append(
            {
                "gene": gene,
                "class": "HSC retention",
                columns[0]: status_human(row),
                columns[1]: 2 if bool(row["author_four_way"]) else 0,
                columns[2]: 2 if row["mouse_progression_FDR"] < 0.10 and row["mouse_progression_logFC"] > 0 else 0,
                columns[3]: 2 if bool(row["mouse_persistent"]) else 0,
                columns[4]: 2 if isinstance(row["functional_support"], str) and row["functional_support"].strip() else 0,
            }
        )
    for gene in mac_genes:
        row = mac.loc[gene]
        rows.append(
            {
                "gene": gene.upper(),
                "class": "Macrophage repair",
                columns[0]: -1,
                columns[1]: -1,
                columns[2]: 2 if row["progression_FDR"] < 0.10 and row["progression_logFC"] > 0 else (1 if row["progression_logFC"] > 0 else 0),
                columns[3]: 2 if row["fate"] == "persistent" and row["residual_logFC"] > 0 and row["residual_FDR"] < 0.10 else (1 if row["residual_logFC"] > 0 else 0),
                columns[4]: 2 if gene == "Trem2" else -1,
            }
        )
    evidence = pd.DataFrame(rows)
    values = evidence[columns].to_numpy(int)

    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    ax.set_facecolor("white")
    for i, row in evidence.iterrows():
        color = RETENTION if row["class"] == "HSC retention" else REPAIR
        for j, value in enumerate(values[i]):
            face = "#E8ECEE" if value == -1 else ("white" if value == 0 else ("#C7CFD3" if value == 1 else color))
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.40), 0.92, 0.80, facecolor=face, edgecolor="white", linewidth=1.1))
            if value == -1:
                ax.text(j, i, "NA", ha="center", va="center", color="#8C969C", fontsize=6.5)
            elif value == 1:
                ax.text(j, i, "+", ha="center", va="center", color=INK, fontweight="bold")
            elif value == 2:
                ax.text(j, i, "+", ha="center", va="center", color="white", fontweight="bold")
    ax.axhline(3.5, color=INK, linewidth=1.0)
    ax.text(-1.38, 1.5, "HSC\nretention", ha="center", va="center", rotation=90, color=RETENTION, fontweight="bold")
    ax.text(-1.38, 6.0, "Macrophage\nrepair", ha="center", va="center", rotation=90, color=REPAIR, fontweight="bold")
    ax.set_xlim(-0.55, len(columns) - 0.45)
    ax.set_ylim(len(evidence) - 0.5, -0.5)
    ax.set_xticks(range(len(columns)), columns)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=8, length=0)
    ax.set_yticks(range(len(evidence)), evidence["gene"])
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=INK, markeredgecolor="none", markersize=8, label="criterion met"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#C7CFD3", markeredgecolor="none", markersize=8, label="directional only"),
        Line2D([0], [0], marker="s", color="#B8C0C4", markerfacecolor="white", markeredgewidth=0.8, markersize=8, label="criterion not met"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#E8ECEE", markeredgecolor="none", markersize=8, label="not applicable"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.31), frameon=False, ncol=4)
    ax.set_title("Categorical evidence prevents comparison of incompatible numeric scores", loc="left", pad=28)
    fig.text(0.01, 0.01, "Cells report evidence type and threshold status; no cross-row magnitude scale is implied.", color=MUTED, fontsize=7)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.75, bottom=0.22)
    evidence.to_csv(OUT / "Fig3D_Categorical_Evidence_source.tsv", sep="\t", index=False)
    save_figure(fig, "Fig3D_Categorical_Evidence")


def build_fig4f_codex_sections() -> None:
    data = pd.read_csv(UQ_CODEX / "UQ_CODEX_scar_enrichment.tsv", sep="\t")
    data = data.loc[data["outcome"].eq("mesenchymal_endothelial_balance")].sort_values("stage_numeric")
    fig, ax = plt.subplots(figsize=(4.25, 3.35))
    colors = [STAGE_COLORS[int(x)] for x in data["stage_numeric"]]
    ax.bar(data["fibrosis_stage"], data["scar_enrichment"], color=colors, width=0.58, edgecolor="white")
    ax.axhline(0, color=INK, linewidth=0.9)
    for x, (_, row) in enumerate(data.iterrows()):
        value = row["scar_enrichment"]
        ax.text(x, value + (0.045 if value >= 0 else -0.055), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontweight="bold", color=INK)
        ax.text(x, -0.54, "1 section", ha="center", va="top", color=MUTED, fontsize=7)
    ax.set_ylim(-0.62, 0.62)
    ax.set_xlabel("Fibrosis stage represented by the section")
    ax.set_ylabel("Vimentin - CD31 enrichment\nin collagen-rich patches")
    ax.set_title("Section-specific CODEX concordance", loc="left")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.01, "Descriptive sections: one F1, one F2, and one F4 biopsy. Bars are not connected and do not estimate a stage trend.", color=MUTED, fontsize=6.9)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.25)
    save_figure(fig, "Fig4F_CODEX_SectionConcordance")


def fit_two_component_gmm(blocks: pd.DataFrame) -> tuple[GaussianMixture, int, int]:
    x = blocks[["resolver_axis", "lock_axis"]].to_numpy(float)
    model = GaussianMixture(n_components=2, covariance_type="full", n_init=50, random_state=20260712).fit(x)
    order = np.argsort(model.means_[:, 1] - model.means_[:, 0])
    return model, int(order[0]), int(order[1])


def build_fig6_static_distribution() -> None:
    blocks = pd.read_parquet(PHASE_DIR / "phase_spatial_blocks.parquet")
    bic = pd.read_csv(PHASE_DIR / "phase_gmm_model_selection.tsv", sep="\t")
    fractions = pd.read_csv(PHASE_DIR / "phase_biopsy_state_fractions.tsv", sep="\t")
    trends = pd.read_csv(PHASE_DIR / "phase_state_fraction_trends.tsv", sep="\t")
    sensitivity = pd.read_csv(PHASE_DIR / "phase_axis_definition_sensitivity.tsv", sep="\t")
    loo = pd.read_csv(PHASE_DIR / "phase_leave_one_patient_out.tsv", sep="\t")
    mouse = pd.read_csv(MOUSE_DIR / "GSE295293_mouse_phase_scores.tsv", sep="\t")
    summary = json.loads((MOUSE_DIR / "mouse_phase_trajectory_summary.json").read_text())
    model, repair_component, retention_component = fit_two_component_gmm(blocks)

    fig = plt.figure(figsize=(10.8, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    x = blocks[["resolver_axis", "lock_axis"]].to_numpy(float)
    xpad = 0.10 * (x.max(axis=0) - x.min(axis=0))
    gx, gy = np.meshgrid(
        np.linspace(x[:, 0].min() - xpad[0], x[:, 0].max() + xpad[0], 220),
        np.linspace(x[:, 1].min() - xpad[1], x[:, 1].max() + xpad[1], 220),
    )
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    p = model.predict_proba(grid)[:, retention_component].reshape(gx.shape)
    ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 11), cmap=mpl.colors.LinearSegmentedColormap.from_list("repair_retention", [REPAIR, "#F7F7F5", RETENTION]), alpha=0.42)
    ax.contour(gx, gy, p, levels=[0.5], colors=[INK], linewidths=1.4)
    for stage, local in blocks.groupby("stage_numeric"):
        ax.scatter(local["resolver_axis"], local["lock_axis"], s=11, color=STAGE_COLORS[int(stage)], alpha=0.48, edgecolor="none", label=f"F{int(stage)}")
    ax.scatter(*model.means_[repair_component], s=95, color=REPAIR, marker="D", edgecolor="white", linewidth=1.0, zorder=5)
    ax.scatter(*model.means_[retention_component], s=95, color=RETENTION, marker="D", edgecolor="white", linewidth=1.0, zorder=5)
    ax.text(*model.means_[repair_component], "  repair-dominant", color=REPAIR, fontweight="bold", va="center")
    ax.text(*model.means_[retention_component], "  retention-dominant", color=RETENTION, fontweight="bold", va="center")
    ax.set(xlabel="Repair program axis", ylabel="HSC-retention axis")
    ax.set_title("A  Static two-component fit to 589 spatial blocks", loc="left")
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.31), handletextpad=0.2, columnspacing=0.6)

    ax = fig.add_subplot(gs[0, 1])
    delta = bic["bic"] - bic["bic"].min()
    ax.plot(bic["n_components"], delta, marker="o", color=INK, linewidth=1.8)
    ax.scatter([2], [float(delta.loc[bic["n_components"].eq(2)].iloc[0])], s=80, color=RETENTION, edgecolor="white", zorder=4)
    ax.set(xticks=bic["n_components"], xlabel="Number of Gaussian components", ylabel="BIC above minimum")
    ax.set_title("B  Model selection and refit stability", loc="left")
    ax.text(
        0.97,
        0.95,
        f"BIC(1) - BIC(2) = {bic.loc[bic.n_components.eq(1), 'bic'].iloc[0] - bic.loc[bic.n_components.eq(2), 'bic'].iloc[0]:.1f}\n"
        f"LOPO median agreement = {loo.basin_agreement.median():.3f}\n"
        f"minimum agreement = {loo.basin_agreement.min():.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#AAB3B8"},
    )

    ax = fig.add_subplot(gs[1, 0])
    patient = fractions.groupby(["patient_cluster", "stage_numeric", "basin_state"], observed=True)["fraction"].mean().reset_index()
    label_map = {"resolved": "repair-dominant", "transition": "transitional", "locked": "retention-dominant"}
    palette = {"resolved": REPAIR, "transition": TRANSITION, "locked": RETENTION}
    for state in ["resolved", "transition", "locked"]:
        local = patient.loc[patient["basin_state"].eq(state)]
        jitter = ((np.arange(len(local)) % 7) - 3) * 0.025
        ax.scatter(local["stage_numeric"] + jitter, local["fraction"], s=18, alpha=0.55, color=palette[state], label=label_map[state])
        means = local.groupby("stage_numeric")["fraction"].mean()
        ax.plot(means.index, means.values, color=palette[state], marker="o", linewidth=2.0)
    retained = trends.loc[trends["basin_state"].eq("locked")].iloc[0]
    ax.text(0.03, 0.95, f"Retention-dominant fraction: beta={retained.stage_beta:+.3f}/stage\npermutation FDR={retained.permutation_fdr:.4f}", transform=ax.transAxes, va="top", color=RETENTION, fontweight="bold")
    ax.set(xticks=range(5), xticklabels=[f"F{x}" for x in range(5)], xlabel="Fibrosis stage", ylabel="Patient-level tissue fraction", ylim=(-0.02, 1.02))
    ax.set_title("C  Stage changes component occupancy", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    condition_colors = {"control": REPAIR, "fibrosis": RETENTION, "regression": TRANSITION}
    for condition, local in mouse.groupby("condition", sort=False):
        ax.scatter(local["resolver_axis"], local["lock_axis"], s=42, color=condition_colors[condition], edgecolor="white", linewidth=0.7, label=condition.capitalize(), zorder=3)
    centroids = mouse.groupby("condition", sort=False)[["resolver_axis", "lock_axis"]].mean()
    for start, end, color in [("control", "fibrosis", RETENTION), ("fibrosis", "regression", TRANSITION)]:
        ax.annotate("", xy=centroids.loc[end], xytext=centroids.loc[start], arrowprops={"arrowstyle": "->", "color": color, "lw": 2.2})
    ax.scatter(centroids["resolver_axis"], centroids["lock_axis"], s=105, facecolors="none", edgecolors=INK, linewidth=1.1, zorder=4)
    ratio = summary["geometry"]["residual_distance_ratio"]
    ax.text(0.03, 0.96, f"9 mice (3/group)\nresidual distance = {100 * ratio:.1f}% of progression", transform=ax.transAxes, va="top", color=INK)
    ax.set(xlabel="Repair axis (mouse z-score)", ylabel="HSC-retention axis (mouse z-score)")
    ax.set_title("D  Replicated withdrawal remains displaced", loc="left")
    ax.legend(frameon=False, loc="lower right")

    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A reproducible two-component spatial distribution accompanies incomplete recovery", fontsize=13.2, fontweight="bold", x=0.02, ha="left")
    fig.text(0.01, 0.005, "The Gaussian mixture is a static distributional model. Contours are posterior probabilities, and arrows show observed group-centroid contrasts.", color=MUTED, fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.09)
    save_figure(fig, "Fig6_Static_TwoComponent_Revision")


def build_fig7d_order_score_audit() -> None:
    pairs = pd.read_csv(DRUG_DIR / "LINCS_ordered_basin_switch_pairs.tsv", sep="\t")
    sensitivity = pd.read_csv(DRUG_DIR / "ezetimibe_verteporfin_weight_sensitivity.tsv", sep="\t")
    target = pairs.loc[pairs["early_compound"].eq("ezetimibe") & pairs["late_compound"].eq("verteporfin")].iloc[0]

    fig = plt.figure(figsize=(9.2, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.0], wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar([0, 1], [target["forward_score"], target["reverse_score"]], color=[REPAIR, RETENTION], width=0.62)
    for bar, value in zip(bars, [target["forward_score"], target["reverse_score"]], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontweight="bold")
    ax.set_xticks([0, 1], ["Ezetimibe\nas early role", "Verteporfin\nas early role"])
    ax.set_ylabel("Role-weighted score")
    ax.set_ylim(0, 1.08)
    ax.set_title("A  Same profiles, exchanged roles", loc="left")
    ax.text(0.5, 0.68, f"formula difference\n{target.order_advantage:+.3f}", ha="center", color=INK, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    formula = (
        "E(e->l) = E(e) + 0.5 E(l)\n"
        "M(e->l) = M(e) + 0.5 M(l)\n"
        "R(e->l) = R(l) + 0.5 R(e)\n"
        "P = 1.5 max[0, -min(E, M)]\n\n"
        "S(e->l) = R + 0.5 E + 0.5 M - P\n"
        "A = S(e->l) - S(l->e)"
    )
    ax.text(0.03, 0.93, formula, va="top", family="monospace", fontsize=8.4, linespacing=1.35, bbox={"boxstyle": "round,pad=0.55", "facecolor": LIGHT, "edgecolor": "#AAB3B8"})
    ax.text(
        0.03,
        0.09,
        "The asymmetric early/late coefficients generate the forward-reverse\n"
        "contrast. This score nominates an equal-exposure factorial experiment\n"
        "and provides no observed treatment-order estimate.",
        va="bottom",
        color=MUTED,
        fontsize=6.8,
        linespacing=1.25,
    )
    ax.set_title("B  The complete scoring rule", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(20260712)
    x = sensitivity["rank"].to_numpy(float)
    y = rng.uniform(-0.10, 0.10, len(x))
    ax.scatter(x, y, s=34, color=TRANSITION, edgecolor="white", linewidth=0.5)
    ax.axvline(24, color=INK, linestyle="--", linewidth=1.0)
    ax.set(xlabel="Rank among 299 ordered pairs", yticks=[], ylim=(-0.28, 0.28), xlim=(15, 51))
    ax.set_title("C  Weight-scheme sensitivity", loc="left")
    ax.text(0.97, 0.90, f"27 schemes\nmedian rank {sensitivity['rank'].median():.0f}\nrange {sensitivity['rank'].min():.0f}-{sensitivity['rank'].max():.0f}", transform=ax.transAxes, ha="right", va="top", color=INK)

    for ax in [fig.axes[0], fig.axes[2]]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Order-aware LINCS scoring nominates a factorial experiment", fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.24)
    save_figure(fig, "Fig7D_OrderAwareScore_Audit")


def write_manifest() -> None:
    rows = [
        ["Fig3D_Categorical_Evidence", HSC_TABLE.relative_to(ROOT).as_posix(), "categorical evidence; no shared numeric scale"],
        ["Fig4F_CODEX_SectionConcordance", (UQ_CODEX / "UQ_CODEX_scar_enrichment.tsv").relative_to(ROOT).as_posix(), "one section per displayed stage; descriptive"],
        ["Fig6_Static_TwoComponent_Revision", (PHASE_DIR / "phase_spatial_blocks.parquet").relative_to(ROOT).as_posix(), "static full-covariance two-component GMM"],
        ["Fig7D_OrderAwareScore_Audit", (DRUG_DIR / "LINCS_ordered_basin_switch_pairs.tsv").relative_to(ROOT).as_posix(), "asymmetric role-weighted computational score"],
    ]
    pd.DataFrame(rows, columns=["output_stem", "primary_input", "interpretation"]).to_csv(OUT / "replacement_panel_manifest.tsv", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global OUT
    OUT = parse_args().output_dir
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    build_fig3d_categorical_evidence()
    build_fig4f_codex_sections()
    build_fig6_static_distribution()
    build_fig7d_order_score_audit()
    write_manifest()
    print(OUT)


if __name__ == "__main__":
    main()
