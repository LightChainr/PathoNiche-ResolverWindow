#!/usr/bin/env python3
"""Test whether locked spatial domains coalesce as the resolver window closes."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
import statsmodels.formula.api as smf


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "data/UQ_spatial_derived/S11_UQ_phase_basin_control/phase_spot_states.parquet"
OUTPUT = REPO_ROOT / "results/S14_spatial_retention_domains"
SEED = 20260712

COLORS = {
    "Endothelial resolver": "#2D8FB3",
    "Macrophage resolver": "#58A47A",
    "Protease activation": "#7D71B3",
    "Antiproteolytic brake": "#D18F24",
    "HSC retention": "#C43D55",
    "TGF-beta integrator": "#8D5A4A",
    "Structural ECM": "#3F4548",
}

FEATURES = {
    "Endothelial resolver": "adj_Endo_resolver",
    "Macrophage resolver": "adj_LAM_resolver",
    "Protease activation": "adj_protease_activation_potential",
    "Antiproteolytic brake": "adj_antiproteolytic_brake",
    "HSC retention": "adj_HSC_retention",
    "TGF-beta integrator": "adj_TGFb_integrator",
    "Structural ECM": "structural_ECM_scar",
}


def bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.clip(ranked, 0, 1)
    return result


def graph_for_biopsy(local: pd.DataFrame) -> tuple[list[list[int]], list[tuple[int, int]]]:
    xy = local[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    tree = cKDTree(xy)
    nearest = tree.query(xy, k=2)[0][:, 1]
    radius = float(np.median(nearest) * 1.25)
    pairs = sorted(tree.query_pairs(radius))
    adjacency = [[] for _ in range(len(local))]
    for i, j in pairs:
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency, pairs


def signed_domain_distance(state: np.ndarray, adjacency: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    distance = np.full(len(state), 10_000, dtype=int)
    boundary = np.zeros(len(state), dtype=bool)
    queue: deque[int] = deque()
    for i, neighbors in enumerate(adjacency):
        if any(state[j] != state[i] for j in neighbors):
            distance[i] = 1
            boundary[i] = True
            queue.append(i)
    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if state[j] == state[i] and distance[j] > distance[i] + 1:
                distance[j] = distance[i] + 1
                queue.append(j)
    finite = distance < 10_000
    if not finite.all():
        distance[~finite] = 7
    signed = np.minimum(distance, 7) * np.where(state, 1, -1)
    return signed, boundary


def largest_component(state: np.ndarray, adjacency: list[list[int]]) -> int:
    seen: set[int] = set()
    sizes: list[int] = []
    for start in np.flatnonzero(state):
        if int(start) in seen:
            continue
        stack = [int(start)]
        seen.add(int(start))
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if state[neighbor] and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return max(sizes, default=0)


def build_front_data(spots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated = []
    topology = []
    for biopsy_id, source in spots.groupby("biopsy_id", sort=False):
        local = source.reset_index(drop=True).copy()
        adjacency, pairs = graph_for_biopsy(local)
        state = local.p_locked.to_numpy(float) >= 0.5
        signed, boundary = signed_domain_distance(state, adjacency)
        local["locked_binary"] = state
        local["signed_domain_distance"] = signed
        local["domain_boundary"] = boundary
        annotated.append(local)

        locked_n = int(state.sum())
        largest = largest_component(state, adjacency)
        cross_edges = int(sum(state[i] != state[j] for i, j in pairs))
        topology.append(
            {
                "biopsy_id": biopsy_id,
                "array": local["array"].iloc[0],
                "patient_cluster": local.patient_cluster.iloc[0],
                "stage_numeric": float(local.stage_numeric.iloc[0]),
                "stage_group": local.stage_group.iloc[0],
                "n_spots": int(len(local)),
                "locked_fraction": locked_n / len(local),
                "largest_locked_domain_fraction": largest / len(local),
                "locked_domain_coherence": largest / max(locked_n, 1),
                "interface_edge_fraction": cross_edges / max(len(pairs), 1),
                "n_graph_edges": int(len(pairs)),
            }
        )
    return pd.concat(annotated, ignore_index=True), pd.DataFrame(topology)


def biopsy_profiles(front: pd.DataFrame) -> pd.DataFrame:
    distances = [-4, -3, -2, -1, 1, 2, 3, 4]
    rows = []
    for (biopsy_id, distance), local in front.loc[
        front.signed_domain_distance.isin(distances)
    ].groupby(["biopsy_id", "signed_domain_distance"], observed=True):
        row = {
            "biopsy_id": biopsy_id,
            "patient_cluster": local.patient_cluster.iloc[0],
            "stage_numeric": float(local.stage_numeric.iloc[0]),
            "distance": int(distance),
            "n_spots": int(len(local)),
        }
        for label, feature in FEATURES.items():
            row[label] = float(local[feature].mean())
        resolver = local[[
            FEATURES["Endothelial resolver"],
            FEATURES["Macrophage resolver"],
            FEATURES["Protease activation"],
        ]].mean(axis=1)
        lock = local[[
            FEATURES["Antiproteolytic brake"],
            FEATURES["HSC retention"],
            FEATURES["TGF-beta integrator"],
        ]].mean(axis=1)
        row["resolver_advantage"] = float((resolver - lock).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_bootstrap(values: pd.DataFrame, column: str, n_boot: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    cluster_stats = values.groupby("patient_cluster")[column].agg(["sum", "count"])
    sums = cluster_stats["sum"].to_numpy(float)
    counts = cluster_stats["count"].to_numpy(float)
    sampled = rng.integers(0, len(cluster_stats), size=(n_boot, len(cluster_stats)))
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def boundary_effects(front: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for biopsy_id, local in front.groupby("biopsy_id", sort=False):
        resolved = local.signed_domain_distance.isin([-2, -1])
        locked = local.signed_domain_distance.isin([1, 2])
        if not resolved.any() or not locked.any():
            continue
        base = {
            "biopsy_id": biopsy_id,
            "patient_cluster": local.patient_cluster.iloc[0],
            "stage_numeric": float(local.stage_numeric.iloc[0]),
        }
        for label, feature in FEATURES.items():
            rows.append(
                {
                    **base,
                    "module": label,
                    "locked_minus_resolved": float(local.loc[locked, feature].mean() - local.loc[resolved, feature].mean()),
                }
            )
    effects = pd.DataFrame(rows)
    summaries = []
    for module, local in effects.groupby("module", sort=False):
        statistic, p_value = wilcoxon(local.locked_minus_resolved, alternative="two-sided")
        low, high = clustered_bootstrap(local, "locked_minus_resolved")
        summaries.append(
            {
                "module": module,
                "n_biopsies": int(local.biopsy_id.nunique()),
                "mean_shift": float(local.locked_minus_resolved.mean()),
                "median_shift": float(local.locked_minus_resolved.median()),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["fdr"] = bh(summary.p_value)
    return effects.merge(summary[["module", "fdr"]], on="module", how="left"), summary


def topology_trends(topology: pd.DataFrame, n_perm: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    metrics = [
        "locked_fraction",
        "largest_locked_domain_fraction",
        "locked_domain_coherence",
        "interface_edge_fraction",
    ]
    rows = []
    groups = [np.asarray(list(index), dtype=int) for index in topology.groupby("array").groups.values()]
    stage = topology.stage_numeric.to_numpy(float)
    stage_centered = stage.copy()
    for index in groups:
        stage_centered[index] -= stage_centered[index].mean()
    for metric in metrics:
        fit = smf.ols(f"{metric} ~ stage_numeric + C(array)", data=topology).fit(cov_type="HC3")
        observed = float(fit.params["stage_numeric"])
        outcome_centered = topology[metric].to_numpy(float).copy()
        for index in groups:
            outcome_centered[index] -= outcome_centered[index].mean()
        null = np.empty(n_perm)
        for index in range(n_perm):
            permuted = stage_centered.copy()
            for positions in groups:
                permuted[positions] = rng.permutation(permuted[positions])
            null[index] = np.dot(permuted, outcome_centered) / np.dot(permuted, permuted)
        permutation_p = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
        rho, spearman_p = spearmanr(topology.stage_numeric, topology[metric])
        rows.append(
            {
                "metric": metric,
                "stage_beta_array_adjusted": observed,
                "hc3_p_value": float(fit.pvalues["stage_numeric"]),
                "within_array_permutation_p": float(permutation_p),
                "spearman_rho": float(rho),
                "spearman_p_value": float(spearman_p),
            }
        )
    trends = pd.DataFrame(rows)
    trends["permutation_fdr"] = bh(trends.within_array_permutation_p)
    return trends


def threshold_sensitivity(spots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.40, 0.50, 0.60, 0.67]:
        metrics = []
        for biopsy_id, source in spots.groupby("biopsy_id", sort=False):
            local = source.reset_index(drop=True)
            adjacency, pairs = graph_for_biopsy(local)
            state = local.p_locked.to_numpy(float) >= threshold
            largest = largest_component(state, adjacency)
            metrics.append(
                {
                    "biopsy_id": biopsy_id,
                    "array": local["array"].iloc[0],
                    "stage_numeric": float(local.stage_numeric.iloc[0]),
                    "largest_locked_domain_fraction": largest / len(local),
                    "interface_edge_fraction": sum(state[i] != state[j] for i, j in pairs) / max(len(pairs), 1),
                }
            )
        local_metrics = pd.DataFrame(metrics)
        for metric in ["largest_locked_domain_fraction", "interface_edge_fraction"]:
            fit = smf.ols(f"{metric} ~ stage_numeric + C(array)", data=local_metrics).fit(cov_type="HC3")
            rows.append(
                {
                    "p_locked_threshold": threshold,
                    "metric": metric,
                    "stage_beta_array_adjusted": float(fit.params["stage_numeric"]),
                    "hc3_p_value": float(fit.pvalues["stage_numeric"]),
                }
            )
    result = pd.DataFrame(rows)
    result["hc3_fdr"] = bh(result.hc3_p_value)
    return result


def profile_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [*FEATURES.keys(), "resolver_advantage"]
    for distance, local in profiles.groupby("distance"):
        for column in columns:
            low, high = clustered_bootstrap(local, column, n_boot=2500)
            rows.append(
                {
                    "distance": int(distance),
                    "module": column,
                    "mean": float(local[column].mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_biopsies": int(local.biopsy_id.nunique()),
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    front: pd.DataFrame,
    topology: pd.DataFrame,
    profile_stats: pd.DataFrame,
    effect_summary: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(17.5, 11.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=.09, right=.985, bottom=.065, top=.91, wspace=.34, hspace=.37)

    mixed = front.groupby("biopsy_id").agg(n=("barcode", "size"), locked=("locked_binary", "mean"))
    representative = (mixed.n * 4 * mixed.locked * (1 - mixed.locked)).idxmax()
    local = front.loc[front.biopsy_id.eq(representative)].copy()
    ax = fig.add_subplot(grid[0, 0])
    cmap = LinearSegmentedColormap.from_list("basin", ["#2A9D8F", "#E5B84B", "#C43D55"])
    ax.scatter(
        local.pxl_col_in_fullres,
        -local.pxl_row_in_fullres,
        c=local.p_locked,
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=16,
        linewidth=0,
    )
    wall = local.loc[local.domain_boundary]
    ax.scatter(
        wall.pxl_col_in_fullres,
        -wall.pxl_row_in_fullres,
        facecolors="none",
        edgecolors="#222222",
        linewidth=.45,
        s=27,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"A  A resolved-to-locked domain wall ({representative})", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    modules = ["Endothelial resolver", "Macrophage resolver", "Antiproteolytic brake", "HSC retention", "Structural ECM"]
    for module in modules:
        plot = profile_stats.loc[profile_stats.module.eq(module) & profile_stats.n_biopsies.ge(8)].sort_values("distance")
        ax.plot(plot.distance, plot["mean"], marker="o", ms=3.5, lw=1.8, color=COLORS[module], label=module)
        ax.fill_between(plot.distance, plot.ci_low, plot.ci_high, color=COLORS[module], alpha=.12, linewidth=0)
    ax.axvline(0, color="#59636A", ls="--", lw=1)
    ax.set(xlabel="Signed graph distance from separatrix", ylabel="Array-adjusted module activity")
    ax.set_title("B  State programs align across the separatrix", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    ax = fig.add_subplot(grid[0, 2])
    advantage = profile_stats.loc[
        profile_stats.module.eq("resolver_advantage") & profile_stats.n_biopsies.ge(8)
    ].sort_values("distance")
    ax.plot(advantage.distance, advantage["mean"], color="#3A6670", marker="o", lw=2.3)
    ax.fill_between(advantage.distance, advantage.ci_low, advantage.ci_high, color="#3A6670", alpha=.18, linewidth=0)
    ax.axhline(0, color="#444", lw=.8)
    ax.axvline(0, color="#59636A", ls="--", lw=1)
    ax.fill_between([-4, 0], -1.2, 1.2, color="#2A9D8F", alpha=.06)
    ax.fill_between([0, 4], -1.2, 1.2, color="#C43D55", alpha=.06)
    ax.text(-3.8, .95, "resolver window open", color="#287D72", fontsize=8)
    ax.text(.2, .95, "lock exceeds resolver", color="#A52E46", fontsize=8)
    ax.set(xlabel="Signed graph distance from separatrix", ylabel="Resolver advantage")
    ax.set_ylim(-1.15, 1.1)
    ax.set_title("C  The resolver window closes inside the locked domain", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[1, 0])
    order = effect_summary.sort_values("mean_shift").reset_index(drop=True)
    y = np.arange(len(order))
    for index, row in order.iterrows():
        ax.plot([row.bootstrap_ci_low, row.bootstrap_ci_high], [index, index], color=COLORS[row.module], lw=2)
        ax.scatter(row.mean_shift, index, color=COLORS[row.module], s=42, edgecolor="white", linewidth=.5, zorder=3)
        ax.text(row.bootstrap_ci_high + .012, index, f"q={row.fdr:.2g}", va="center", fontsize=7, color="#4D5559")
    labels = [row.module for row in order.itertuples()]
    ax.set_yticks(y, labels)
    ax.axvline(0, color="#555", lw=.8)
    ax.set_xlim(-.03, .58)
    ax.set_xlabel("Locked minus resolved boundary activity")
    ax.set_title("D  State alignment across 33 biopsies", loc="left", fontweight="bold")

    stage_palette = {0: "#87B7A7", 1: "#68A7A0", 2: "#D8AD55", 3: "#C36C64", 4: "#A93651"}
    ax = fig.add_subplot(grid[1, 1])
    rng = np.random.default_rng(SEED)
    for stage, local_stage in topology.groupby("stage_numeric"):
        x = stage + rng.normal(0, .045, len(local_stage))
        ax.scatter(x, local_stage.largest_locked_domain_fraction, color=stage_palette[int(stage)], s=38, edgecolor="white", linewidth=.5)
    sns.regplot(data=topology, x="stage_numeric", y="largest_locked_domain_fraction", scatter=False, color="#5A3740", ci=95, ax=ax)
    trend = trends.set_index("metric").loc["largest_locked_domain_fraction"]
    ax.text(.03, .96, f"array-adjusted beta={trend.stage_beta_array_adjusted:.3f}\npermutation q={trend.permutation_fdr:.3g}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set(xlabel="Fibrosis stage", ylabel="Largest locked domain / biopsy")
    ax.set_title("E  Locked islands coalesce into a dominant domain", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[1, 2])
    for stage, local_stage in topology.groupby("stage_numeric"):
        x = stage + rng.normal(0, .045, len(local_stage))
        ax.scatter(x, local_stage.interface_edge_fraction, color=stage_palette[int(stage)], s=38, edgecolor="white", linewidth=.5)
    sns.regplot(data=topology, x="stage_numeric", y="interface_edge_fraction", scatter=False, color="#3A6670", ci=95, ax=ax)
    trend = trends.set_index("metric").loc["interface_edge_fraction"]
    ax.text(.03, .96, f"array-adjusted beta={trend.stage_beta_array_adjusted:.3f}\npermutation q={trend.permutation_fdr:.3g}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set(xlabel="Fibrosis stage", ylabel="Resolved-locked interface / graph edges")
    ax.set_title("F  Interface compression marks spatial lock-in", loc="left", fontweight="bold")

    for axis in fig.axes:
        if axis.axison:
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Locked domains coalesce as the spatial resolver window closes", fontsize=17, fontweight="bold")
    fig.savefig(OUTPUT / "Spatial_domain_wall_resolver_window.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT / "Spatial_domain_wall_resolver_window.pdf", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    global INPUT, OUTPUT
    args = parse_args()
    INPUT, OUTPUT = args.input, args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spots = pd.read_parquet(INPUT)
    front, topology = build_front_data(spots)
    profiles = biopsy_profiles(front)
    profile_stats = profile_summary(profiles)
    effects, effect_summary = boundary_effects(front)
    trends = topology_trends(topology)
    sensitivity = threshold_sensitivity(spots)
    plot_results(front, topology, profile_stats, effect_summary, trends)

    front[[
        "barcode", "array", "biopsy_id", "patient_cluster", "stage_numeric", "p_locked",
        "locked_binary", "signed_domain_distance", "domain_boundary",
    ]].to_parquet(OUTPUT / "domain_wall_spot_annotations.parquet", index=False)
    topology.to_csv(OUTPUT / "spatial_domain_topology.tsv", sep="\t", index=False)
    profiles.to_csv(OUTPUT / "domain_wall_biopsy_profiles.tsv", sep="\t", index=False)
    profile_stats.to_csv(OUTPUT / "domain_wall_profile_summary.tsv", sep="\t", index=False)
    effects.to_csv(OUTPUT / "domain_wall_boundary_effects_by_biopsy.tsv", sep="\t", index=False)
    effect_summary.to_csv(OUTPUT / "domain_wall_boundary_effect_summary.tsv", sep="\t", index=False)
    trends.to_csv(OUTPUT / "spatial_domain_stage_trends.tsv", sep="\t", index=False)
    sensitivity.to_csv(OUTPUT / "spatial_domain_threshold_sensitivity.tsv", sep="\t", index=False)

    trend_index = trends.set_index("metric")
    summary = {
        "n_spots": int(len(front)),
        "n_biopsies": int(front.biopsy_id.nunique()),
        "n_patients": int(front.patient_cluster.nunique()),
        "boundary_definition": "p_locked >= 0.5; signed shortest-path distance to nearest opposite-basin spot",
        "largest_domain_stage_beta": float(trend_index.loc["largest_locked_domain_fraction", "stage_beta_array_adjusted"]),
        "largest_domain_permutation_fdr": float(trend_index.loc["largest_locked_domain_fraction", "permutation_fdr"]),
        "interface_stage_beta": float(trend_index.loc["interface_edge_fraction", "stage_beta_array_adjusted"]),
        "interface_permutation_fdr": float(trend_index.loc["interface_edge_fraction", "permutation_fdr"]),
        "threshold_sensitivity": {
            "thresholds": sensitivity.p_locked_threshold.unique().tolist(),
            "largest_domain_beta_range": [
                float(sensitivity.loc[sensitivity.metric.eq("largest_locked_domain_fraction"), "stage_beta_array_adjusted"].min()),
                float(sensitivity.loc[sensitivity.metric.eq("largest_locked_domain_fraction"), "stage_beta_array_adjusted"].max()),
            ],
            "interface_beta_range": [
                float(sensitivity.loc[sensitivity.metric.eq("interface_edge_fraction"), "stage_beta_array_adjusted"].min()),
                float(sensitivity.loc[sensitivity.metric.eq("interface_edge_fraction"), "stage_beta_array_adjusted"].max()),
            ],
        },
        "interpretation": "Fibrosis progression consolidates initially fragmented locked islands into a dominant spatial domain while compressing the resolved-locked interface; across that interface the resolver advantage changes sign as HSC retention and antiproteolytic control overtake resolver activity.",
    }
    (OUTPUT / "spatial_domain_wall_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
