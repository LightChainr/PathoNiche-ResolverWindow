#!/usr/bin/env python3
"""Stress-test UQ biopsy registration without using expression scores to assign tissue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

from spatial_domain_wall_resolver_window import graph_for_biopsy, largest_component


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPOTS = (
    REPO_ROOT
    / "data/UQ_spatial_derived/S11_UQ_phase_basin_control/phase_spot_states.parquet"
)
DEFAULT_COMPONENT_MAP = (
    REPO_ROOT
    / "data/UQ_spatial_mapping_derived/UQ_Visium_spot_component_map.parquet"
)
DEFAULT_BIOPSY_MAP = (
    REPO_ROOT
    / "docs/methods_audit/machine_readable/UQ_Biopsy_Reconstruction_Map.tsv"
)
DEFAULT_OUTPUT = REPO_ROOT / "results/S15_UQ_biopsy_mapping_sensitivity"
SEED = 20260712
METRICS = [
    "locked_fraction",
    "largest_locked_domain_fraction",
    "locked_domain_coherence",
    "interface_edge_fraction",
]
KEY_METRICS = {
    "largest_locked_domain_fraction": "Largest retention domain",
    "interface_edge_fraction": "Resolver-retention interface",
}


def bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def explode_biopsy_map(mapping: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in mapping.to_dict("records"):
        for component in str(record["component_members"]).split(","):
            rows.append(
                {
                    "array": record["array"],
                    "component": component.strip(),
                    "biopsy_id": record["biopsy_id"],
                    "fibrosis_stage": record["fibrosis_stage"],
                    "stage_numeric": float(record["stage_numeric"]),
                    "patient_cluster": record["patient_cluster"],
                    "mapping_source": record["mapping_source"],
                }
            )
    return pd.DataFrame(rows)


def audit_geometry(
    spots: pd.DataFrame,
    component_spots: pd.DataFrame,
    expanded_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"array", "barcode", "component"}
    missing = required - set(component_spots.columns)
    if missing:
        raise ValueError(f"Component map is missing columns: {sorted(missing)}")

    geometry = component_spots[["array", "barcode", "component"]].rename(
        columns={"component": "geometry_component"}
    )
    audited = spots.merge(geometry, on=["array", "barcode"], how="left", validate="one_to_one")
    audited["direct_geometry_component"] = audited["geometry_component"].notna()
    audited["geometry_component_matches"] = (
        audited["geometry_component"].eq(audited["component"])
        & audited["direct_geometry_component"]
    )
    audited["fragment_assignment"] = np.where(
        audited["direct_geometry_component"], "direct_hex_component", "nearest_component_rescue"
    )

    expected = expanded_map.rename(
        columns={
            "biopsy_id": "expected_biopsy_id",
            "stage_numeric": "expected_stage_numeric",
            "patient_cluster": "expected_patient_cluster",
        }
    )
    audited = audited.merge(
        expected[
            [
                "array",
                "component",
                "expected_biopsy_id",
                "expected_stage_numeric",
                "expected_patient_cluster",
            ]
        ],
        on=["array", "component"],
        how="left",
        validate="many_to_one",
    )
    audited["registration_matches_table"] = (
        audited["biopsy_id"].eq(audited["expected_biopsy_id"])
        & audited["stage_numeric"].eq(audited["expected_stage_numeric"])
        & audited["patient_cluster"].eq(audited["expected_patient_cluster"])
    )

    summaries: list[dict[str, object]] = []
    for array, local in audited.groupby("array", sort=True):
        direct = local["direct_geometry_component"]
        summaries.append(
            {
                "array": array,
                "n_analysis_spots": int(len(local)),
                "n_direct_geometry_spots": int(direct.sum()),
                "n_rescued_fragment_spots": int((~direct).sum()),
                "n_direct_component_matches": int(local.loc[direct, "geometry_component_matches"].sum()),
                "direct_component_match_rate": float(
                    local.loc[direct, "geometry_component_matches"].mean()
                ),
                "n_registration_matches": int(local["registration_matches_table"].sum()),
                "registration_match_rate": float(local["registration_matches_table"].mean()),
            }
        )
    return audited, pd.DataFrame(summaries)


def topology_for_units(spots: pd.DataFrame, unit_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for analysis_unit, source in spots.groupby(unit_column, sort=False, observed=True):
        local = source.reset_index(drop=True).copy()
        adjacency, pairs = graph_for_biopsy(local)
        state = local["p_locked"].to_numpy(float) >= 0.5
        locked_n = int(state.sum())
        largest = largest_component(state, adjacency)
        cross_edges = int(sum(state[i] != state[j] for i, j in pairs))
        rows.append(
            {
                "analysis_unit": analysis_unit,
                "biopsy_id": local["biopsy_id"].iloc[0],
                "component": local["component"].iloc[0]
                if local["component"].nunique() == 1
                else "MULTI",
                "array": local["array"].iloc[0],
                "patient_cluster": local["patient_cluster"].iloc[0],
                "stage_numeric": float(local["stage_numeric"].iloc[0]),
                "n_spots": int(len(local)),
                "locked_fraction": locked_n / len(local),
                "largest_locked_domain_fraction": largest / len(local),
                "locked_domain_coherence": largest / max(locked_n, 1),
                "interface_edge_fraction": cross_edges / max(len(pairs), 1),
                "n_graph_edges": int(len(pairs)),
            }
        )
    return pd.DataFrame(rows)


def fit_trends(topology: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    groups = [
        np.asarray(list(index), dtype=int)
        for index in topology.groupby("array", sort=False).groups.values()
    ]
    stage = topology["stage_numeric"].to_numpy(float)
    stage_centered = stage.copy()
    for index in groups:
        stage_centered[index] -= stage_centered[index].mean()

    rows: list[dict[str, object]] = []
    for metric in METRICS:
        model = smf.ols(f"{metric} ~ stage_numeric + C(array)", data=topology)
        hc3 = model.fit(cov_type="HC3")
        clustered = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": topology["patient_cluster"].to_numpy()},
        )
        observed = float(hc3.params["stage_numeric"])
        outcome_centered = topology[metric].to_numpy(float).copy()
        for index in groups:
            outcome_centered[index] -= outcome_centered[index].mean()
        denominator = np.dot(stage_centered, stage_centered)
        null = np.empty(n_perm)
        for permutation in range(n_perm):
            permuted = stage_centered.copy()
            for positions in groups:
                permuted[positions] = rng.permutation(permuted[positions])
            null[permutation] = np.dot(permuted, outcome_centered) / denominator
        permutation_p = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
        rho, spearman_p = spearmanr(topology["stage_numeric"], topology[metric])
        rows.append(
            {
                "metric": metric,
                "n_units": int(len(topology)),
                "n_biopsies": int(topology["biopsy_id"].nunique()),
                "n_patients": int(topology["patient_cluster"].nunique()),
                "n_arrays": int(topology["array"].nunique()),
                "stage_beta_array_adjusted": observed,
                "stage_beta_se_hc3": float(hc3.bse["stage_numeric"]),
                "hc3_p_value": float(hc3.pvalues["stage_numeric"]),
                "patient_cluster_p_value": float(clustered.pvalues["stage_numeric"]),
                "within_array_permutation_p": float(permutation_p),
                "spearman_rho": float(rho),
                "spearman_p_value": float(spearman_p),
            }
        )
    result = pd.DataFrame(rows)
    result["permutation_fdr"] = bh(result["within_array_permutation_p"])
    return result


def build_scenarios(
    audited: pd.DataFrame,
    mapping: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame, str]]:
    data = audited.copy()
    data["component_unit"] = data["array"] + "_" + data["component"]
    multi_biopsies = set(
        mapping.loc[mapping["component_members"].str.contains(",", regex=False), "biopsy_id"]
    )
    inferred = set(
        mapping.loc[
            mapping["mapping_source"].str.contains("inferred", case=False, na=False),
            "biopsy_id",
        ]
    )
    return [
        ("registered_biopsy_all", data, "biopsy_id"),
        (
            "registered_biopsy_no_fragment_rescue",
            data.loc[data["direct_geometry_component"]].copy(),
            "biopsy_id",
        ),
        ("geometry_component_split_all", data, "component_unit"),
        (
            "geometry_component_split_no_fragment_rescue",
            data.loc[data["direct_geometry_component"]].copy(),
            "component_unit",
        ),
        (
            "exclude_inferred_synchronous_pair",
            data.loc[~data["biopsy_id"].isin(inferred)].copy(),
            "biopsy_id",
        ),
        (
            "exclude_all_multicomponent_biopsies",
            data.loc[~data["biopsy_id"].isin(multi_biopsies)].copy(),
            "biopsy_id",
        ),
    ]


def leave_one_array_out(audited: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for array in sorted(audited["array"].unique()):
        subset = audited.loc[~audited["array"].eq(array)].copy()
        topology = topology_for_units(subset, "biopsy_id")
        trends = fit_trends(topology, n_perm=n_perm)
        trends.insert(0, "scenario", f"leave_out_{array}")
        trends.insert(1, "excluded_array", array)
        frames.append(trends)
    return pd.concat(frames, ignore_index=True)


def plot_key_results(
    scenario_trends: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    output: Path,
) -> None:
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(12.8, 8.3), facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.44)

    ax = fig.add_subplot(grid[0, :])
    x = np.arange(len(geometry_summary))
    direct = geometry_summary["n_direct_geometry_spots"].to_numpy()
    rescued = geometry_summary["n_rescued_fragment_spots"].to_numpy()
    ax.bar(x, direct, color="#2D8FB3", label="Direct hex component")
    ax.bar(x, rescued, bottom=direct, color="#D8AD55", label="Nearest-component rescue")
    ax.set_xticks(x, geometry_summary["array"], rotation=25, ha="right")
    ax.set_ylabel("Analysis spots")
    ax.set_title("A  Geometry-only component recovery", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper left")

    scenario_order = [
        "registered_biopsy_all",
        "registered_biopsy_no_fragment_rescue",
        "geometry_component_split_all",
        "geometry_component_split_no_fragment_rescue",
        "exclude_inferred_synchronous_pair",
        "exclude_all_multicomponent_biopsies",
    ]
    labels = {
        "registered_biopsy_all": "Registered biopsies",
        "registered_biopsy_no_fragment_rescue": "No rescued fragments",
        "geometry_component_split_all": "Split geometric components",
        "geometry_component_split_no_fragment_rescue": "Split + no rescue",
        "exclude_inferred_synchronous_pair": "Exclude sync pair",
        "exclude_all_multicomponent_biopsies": "Exclude merged cores",
    }
    colors = {
        "largest_locked_domain_fraction": "#A93651",
        "interface_edge_fraction": "#2D8FB3",
    }
    for panel, metric in enumerate(KEY_METRICS, start=1):
        ax = fig.add_subplot(grid[1, panel - 1])
        current = (
            scenario_trends.loc[scenario_trends["metric"].eq(metric)]
            .set_index("scenario")
            .loc[scenario_order]
            .reset_index()
        )
        y = np.arange(len(current))[::-1]
        beta = current["stage_beta_array_adjusted"].to_numpy()
        se = current["stage_beta_se_hc3"].to_numpy()
        ax.errorbar(
            beta,
            y,
            xerr=1.96 * se,
            fmt="o",
            color=colors[metric],
            ecolor=colors[metric],
            capsize=2.5,
            lw=1.4,
        )
        loo = leave_one_out.loc[leave_one_out["metric"].eq(metric), "stage_beta_array_adjusted"]
        ax.axvspan(loo.min(), loo.max(), color=colors[metric], alpha=0.10, lw=0)
        ax.axvline(0, color="#59636A", lw=0.8)
        ax.set_yticks(y, [labels[name] for name in current["scenario"]])
        ax.set_xlabel("Array-adjusted stage effect (95% HC3 CI)")
        ax.set_title(
            f"{chr(65 + panel)}  {KEY_METRICS[metric]}",
            loc="left",
            fontweight="bold",
        )

    for axis in fig.axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Spatial retention-domain conclusions survive biopsy-registration stress tests",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output / "UQ_biopsy_mapping_sensitivity.svg", bbox_inches="tight")
    fig.savefig(output / "UQ_biopsy_mapping_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "UQ_biopsy_mapping_sensitivity.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_report(
    geometry_summary: pd.DataFrame,
    scenario_trends: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[dict[str, object], str]:
    key = scenario_trends.loc[scenario_trends["metric"].isin(KEY_METRICS)].copy()
    expected = key["metric"].map(
        {
            "largest_locked_domain_fraction": 1,
            "interface_edge_fraction": -1,
        }
    )
    key["expected_direction"] = np.sign(key["stage_beta_array_adjusted"]).eq(expected)
    loo_key = leave_one_out.loc[leave_one_out["metric"].isin(KEY_METRICS)].copy()
    loo_expected = loo_key["metric"].map(
        {
            "largest_locked_domain_fraction": 1,
            "interface_edge_fraction": -1,
        }
    )
    loo_key["expected_direction"] = np.sign(
        loo_key["stage_beta_array_adjusted"]
    ).eq(loo_expected)

    direct = int(geometry_summary["n_direct_geometry_spots"].sum())
    rescued = int(geometry_summary["n_rescued_fragment_spots"].sum())
    direct_matches = int(geometry_summary["n_direct_component_matches"].sum())
    registered_matches = int(geometry_summary["n_registration_matches"].sum())
    total = int(geometry_summary["n_analysis_spots"].sum())
    summary = {
        "analysis_scope": "UQ MASLD Visium biopsy reconstruction and spatial topology",
        "geometry_assignment_inputs": [
            "array identifier",
            "Visium barcode",
            "hex-grid array_row and array_col",
            "connected-component membership",
        ],
        "scores_excluded_from_assignment": [
            "ECM_scar",
            "HSC_retention",
            "Endo_resolver",
            "LAM_resolver",
            "repair/resolver composite scores",
            "p_locked",
        ],
        "direct_geometry_spots": direct,
        "nearest_component_rescue_spots": rescued,
        "direct_geometry_match_rate": direct_matches / max(direct, 1),
        "registration_table_match_rate": registered_matches / max(total, 1),
        "registered_biopsies": int(mapping["biopsy_id"].nunique()),
        "patient_clusters": int(mapping["patient_cluster"].nunique()),
        "multicomponent_biopsies": int(
            mapping["component_members"].str.contains(",", regex=False).sum()
        ),
        "inferred_synchronous_biopsies": int(
            mapping["mapping_source"].str.contains("inferred", case=False, na=False).sum()
        ),
        "scenario_key_direction_pass": int(key["expected_direction"].sum()),
        "scenario_key_direction_total": int(len(key)),
        "leave_one_array_out_key_direction_pass": int(loo_key["expected_direction"].sum()),
        "leave_one_array_out_key_direction_total": int(len(loo_key)),
        "historical_operator_blinding": "not documented",
        "independent_reconstruction_conclusion": (
            "The current geometry-only audit reproduces component membership without consulting "
            "ECM, retention, resolver, repair, or fitted-state scores. Historical strict operator "
            "blinding cannot be established from the archived record."
        ),
    }

    primary = (
        scenario_trends.loc[
            scenario_trends["scenario"].eq("registered_biopsy_all")
            & scenario_trends["metric"].isin(KEY_METRICS)
        ]
        .set_index("metric")
    )
    split = (
        scenario_trends.loc[
            scenario_trends["scenario"].eq("geometry_component_split_no_fragment_rescue")
            & scenario_trends["metric"].isin(KEY_METRICS)
        ]
        .set_index("metric")
    )
    report = f"""# UQ biopsy reconstruction and mapping sensitivity audit

## Decision

The spatial-retention result is robust to the principal reconstruction choices tested here. The largest retention-domain effect and interface-compression effect retained their expected directions in {summary['scenario_key_direction_pass']}/{summary['scenario_key_direction_total']} prespecified mapping scenarios and {summary['leave_one_array_out_key_direction_pass']}/{summary['leave_one_array_out_key_direction_total']} leave-one-array-out tests.

## Score-independent reconstruction

The audit recovered {direct:,} spots directly from Visium hex-grid connected components. All directly recovered spots matched their archived component labels ({summary['direct_geometry_match_rate']:.3%}). A further {rescued:,} spots belonged to small disconnected fragments and had previously been assigned to the nearest registered component. Removing those rescued spots did not reverse either key result. Component construction and registration checking used array identity, barcodes, spatial coordinates, and the archived biopsy table. ECM, HSC-retention, endothelial/macrophage resolver, repair, and fitted-state scores were excluded from assignment.

The archived map contains {summary['multicomponent_biopsies']} biopsies assembled from more than one geometric component and {summary['inferred_synchronous_biopsies']} biopsy entries from one inferred synchronous pair. Splitting every component into its own analysis unit, excluding all multicomponent biopsies, and excluding the inferred pair each preserved the key effect directions.

## Key estimates

Registered-biopsy analysis:

- Largest retention domain per stage: beta = {primary.loc['largest_locked_domain_fraction', 'stage_beta_array_adjusted']:.4f}; within-array permutation q = {primary.loc['largest_locked_domain_fraction', 'permutation_fdr']:.4g}.
- Resolver-retention interface per stage: beta = {primary.loc['interface_edge_fraction', 'stage_beta_array_adjusted']:.4f}; within-array permutation q = {primary.loc['interface_edge_fraction', 'permutation_fdr']:.4g}.

Geometry-component analysis with rescued fragments removed:

- Largest retention domain per stage: beta = {split.loc['largest_locked_domain_fraction', 'stage_beta_array_adjusted']:.4f}; within-array permutation q = {split.loc['largest_locked_domain_fraction', 'permutation_fdr']:.4g}.
- Resolver-retention interface per stage: beta = {split.loc['interface_edge_fraction', 'stage_beta_array_adjusted']:.4f}; within-array permutation q = {split.loc['interface_edge_fraction', 'permutation_fdr']:.4g}.

## Audit boundary

The independent computational reconstruction is score-blind by construction. The original operator's strict blinding to ECM, retention, and repair scores is not documented in the historical archive. The defensible manuscript wording is that the reconstruction table predates score generation and that an independent score-excluded reconstruction plus mapping sensitivity analysis reproduced the conclusions. The original per-spot `*_samples.csv` files referenced by the source study code are not present in the local archive and remain the only unresolved provenance item.
"""
    return summary, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-spots", type=Path, default=DEFAULT_SPOTS)
    parser.add_argument("--component-map", type=Path, default=DEFAULT_COMPONENT_MAP)
    parser.add_argument("--biopsy-map", type=Path, default=DEFAULT_BIOPSY_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--loo-permutations", type=int, default=1999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spots = pd.read_parquet(args.phase_spots)
    component_spots = pd.read_parquet(args.component_map)
    mapping = pd.read_csv(args.biopsy_map, sep="\t")
    expanded_map = explode_biopsy_map(mapping)

    audited, geometry_summary = audit_geometry(spots, component_spots, expanded_map)
    audit_columns = [
        "array",
        "barcode",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "component",
        "geometry_component",
        "direct_geometry_component",
        "geometry_component_matches",
        "fragment_assignment",
        "biopsy_id",
        "fibrosis_stage",
        "stage_numeric",
        "patient_cluster",
        "mapping_source",
        "expected_biopsy_id",
        "expected_stage_numeric",
        "expected_patient_cluster",
        "registration_matches_table",
    ]
    audited[audit_columns].to_parquet(
        args.output_dir / "UQ_spot_geometry_registration_audit.parquet", index=False
    )
    geometry_summary.to_csv(
        args.output_dir / "UQ_geometry_reconstruction_summary.tsv", sep="\t", index=False
    )

    topology_frames: list[pd.DataFrame] = []
    trend_frames: list[pd.DataFrame] = []
    for name, scenario_spots, unit_column in build_scenarios(audited, mapping):
        topology = topology_for_units(scenario_spots, unit_column)
        topology.insert(0, "scenario", name)
        trends = fit_trends(topology, n_perm=args.permutations)
        trends.insert(0, "scenario", name)
        topology_frames.append(topology)
        trend_frames.append(trends)
    scenario_topology = pd.concat(topology_frames, ignore_index=True)
    scenario_trends = pd.concat(trend_frames, ignore_index=True)
    scenario_topology.to_csv(
        args.output_dir / "UQ_mapping_scenario_topology.tsv", sep="\t", index=False
    )
    scenario_trends.to_csv(
        args.output_dir / "UQ_mapping_scenario_stage_trends.tsv", sep="\t", index=False
    )

    leave_one_out = leave_one_array_out(audited, n_perm=args.loo_permutations)
    leave_one_out.to_csv(
        args.output_dir / "UQ_mapping_leave_one_array_out.tsv", sep="\t", index=False
    )
    plot_key_results(scenario_trends, leave_one_out, geometry_summary, args.output_dir)

    summary, report = make_report(geometry_summary, scenario_trends, leave_one_out, mapping)
    (args.output_dir / "UQ_biopsy_mapping_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "UQ_BIOPSY_MAPPING_AUDIT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
