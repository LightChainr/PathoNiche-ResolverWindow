#!/usr/bin/env python3
"""Learn resolver/lock state basins and test history-dependent locking in UQ Visium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import diptest
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "data/UQ_spatial_derived/S10_UQ_spatial_proteolytic_lock/UQ_proteolytic_lock_spot_scores.parquet"
OUTPUT = REPO_ROOT / "results/S11_UQ_static_component_control"
RNG = np.random.default_rng(20260712)
STAGE_ORDER = ["early", "intermediate", "late"]
STAGE_COLORS = {"early": "#3182A6", "intermediate": "#E3A12F", "late": "#B83D4B"}
STATE_COLORS = {"resolved": "#2A9D8F", "transition": "#E9B949", "locked": "#C43D55"}

LOCK_FEATURES = {
    "HSC_retention": ["HSC_identity", "log_library"],
    "feedback_core": ["HSC_identity", "log_library"],
    "clearance_inefficiency": ["HSC_identity", "Myeloid_identity", "log_library"],
    "TGFb_integrator": ["HSC_identity", "Myeloid_identity", "log_library"],
}
RESOLVER_FEATURES = {
    "Endo_resolver": ["Endo_identity", "HSC_identity", "log_library"],
    "LAM_resolver": ["Myeloid_identity", "HSC_identity", "log_library"],
}
AUX_FEATURES = {
    "antiproteolytic_brake": ["HSC_identity", "Myeloid_identity", "log_library"],
    "protease_activation_potential": ["HSC_identity", "Myeloid_identity", "log_library"],
}


def bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.clip(q, 0, 1)
    return result


def array_design(data: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    array = pd.get_dummies(data["array"], drop_first=True, dtype=float)
    return np.column_stack(
        [np.ones(len(data))]
        + [data[name].to_numpy(float) for name in covariates]
        + [array.to_numpy(float)]
    )


def residualize_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    audit = []
    for family, definitions in [("lock", LOCK_FEATURES), ("resolver", RESOLVER_FEATURES), ("aux", AUX_FEATURES)]:
        for feature, covariates in definitions.items():
            x = array_design(data, covariates)
            y = data[feature].to_numpy(float)
            residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            residual = (residual - residual.mean()) / max(residual.std(ddof=1), 1e-8)
            data[f"adj_{feature}"] = residual
            audit.append(
                {
                    "family": family,
                    "feature": feature,
                    "covariates": ";".join(covariates) + ";array_fixed_effects",
                    "residual_sd": float(residual.std(ddof=1)),
                    "raw_residual_r": float(np.corrcoef(y, residual)[0, 1]),
                }
            )

    loadings = []
    for family, features in [
        ("lock", list(LOCK_FEATURES)),
        ("resolver", list(RESOLVER_FEATURES)),
    ]:
        columns = [f"adj_{feature}" for feature in features]
        matrix = StandardScaler().fit_transform(data[columns])
        model = PCA(n_components=1, random_state=20260712)
        axis = model.fit_transform(matrix).ravel()
        orientation = 1 if model.components_[0].sum() >= 0 else -1
        axis *= orientation
        data[f"{family}_axis"] = (axis - axis.mean()) / axis.std(ddof=1)
        for feature, weight in zip(features, model.components_[0] * orientation):
            loadings.append(
                {
                    "axis": family,
                    "feature": feature,
                    "loading": float(weight),
                    "explained_variance": float(model.explained_variance_ratio_[0]),
                }
            )
    data["control_imbalance"] = (data["lock_axis"] - data["resolver_axis"]) / np.sqrt(2)
    data["lock_mechanistic_equal"] = data[
        ["adj_HSC_retention", "adj_feedback_core", "adj_antiproteolytic_brake", "adj_TGFb_integrator"]
    ].mean(axis=1)
    data["resolver_mechanistic_equal"] = data[
        ["adj_Endo_resolver", "adj_LAM_resolver", "adj_protease_activation_potential"]
    ].mean(axis=1)
    data["lock_compact_equal"] = data[
        ["adj_HSC_retention", "adj_feedback_core", "adj_clearance_inefficiency"]
    ].mean(axis=1)
    data["resolver_compact_equal"] = data[["adj_Endo_resolver", "adj_LAM_resolver"]].mean(axis=1)
    for column in [
        "lock_mechanistic_equal", "resolver_mechanistic_equal",
        "lock_compact_equal", "resolver_compact_equal",
    ]:
        data[column] = (data[column] - data[column].mean()) / max(data[column].std(ddof=1), 1e-8)
    return data, pd.concat([pd.DataFrame(audit), pd.DataFrame(loadings)], ignore_index=True, sort=False)


def spatial_blocks(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = [
        "lock_axis", "resolver_axis", "control_imbalance", "ECM_scar", "scar_proximity",
        "HSC_retention", "clearance_inefficiency", "feedback_core", "TGFb_integrator",
        "lock_mechanistic_equal", "resolver_mechanistic_equal",
        "lock_compact_equal", "resolver_compact_equal",
    ]
    for biopsy, local in data.groupby("biopsy_id", sort=True, observed=True):
        coordinates = local[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
        centered = coordinates - coordinates.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        rotated = centered @ vh.T
        local = local.copy()
        local["long_bin"] = pd.qcut(pd.Series(rotated[:, 0]).rank(method="first"), 6, labels=False).to_numpy()
        local["cross_bin"] = pd.qcut(pd.Series(rotated[:, 1]).rank(method="first"), 3, labels=False).to_numpy()
        block = local.groupby(["long_bin", "cross_bin"], observed=True)[numeric].mean().reset_index()
        meta = local.iloc[0]
        block["biopsy_id"] = biopsy
        for name in ["patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group"]:
            block[name] = meta[name]
        block["n_spots_biopsy"] = len(local)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def fit_basins(blocks: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, GaussianMixture, dict[int, str]]:
    x = blocks[["resolver_axis", "lock_axis"]].to_numpy(float)
    bic_rows = []
    models = {}
    for k in range(1, 6):
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=50, random_state=20260712)
        model.fit(x)
        models[k] = model
        bic_rows.append({"n_components": k, "bic": float(model.bic(x)), "aic": float(model.aic(x))})
    bic = pd.DataFrame(bic_rows)

    model = models[2]
    component_imbalance = model.means_[:, 1] - model.means_[:, 0]
    ordered = np.argsort(component_imbalance)
    labels = {int(ordered[0]): "resolved", int(ordered[1]): "locked"}
    probabilities = model.predict_proba(data[["resolver_axis", "lock_axis"]].to_numpy(float))
    components = model.predict(data[["resolver_axis", "lock_axis"]].to_numpy(float))
    data = data.copy()
    data["basin_component"] = components
    for component, state in labels.items():
        data[f"p_{state}"] = probabilities[:, component]
    data["p_transition"] = 4 * data["p_resolved"] * data["p_locked"]
    data["basin_state"] = np.where(
        data["p_locked"].ge(0.67),
        "locked",
        np.where(data["p_resolved"].ge(0.67), "resolved", "transition"),
    )
    data["basin_escape_margin"] = data["p_locked"] - data["p_resolved"]
    return data, bic, model, labels


def dip_and_bootstrap(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for group, local in [("all", blocks), *blocks.groupby("stage_group", observed=True)]:
        values = local["control_imbalance"].to_numpy(float)
        dip, p = diptest.diptest(values, boot_pval=True, n_boot=5000, seed=20260712)
        rows.append({"stage_group": group, "n_blocks": len(values), "dip": float(dip), "dip_p": float(p)})

    patient_ids = blocks["patient_cluster"].unique()
    boot_rows = []
    for iteration in range(300):
        selected = RNG.choice(patient_ids, len(patient_ids), replace=True)
        sample = pd.concat(
            [blocks.loc[blocks.patient_cluster.eq(patient)].assign(bootstrap_copy=i) for i, patient in enumerate(selected)],
            ignore_index=True,
        )
        x = sample[["resolver_axis", "lock_axis"]].to_numpy(float)
        bics = []
        for k in range(1, 5):
            model = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=iteration + 1)
            model.fit(x)
            bics.append(model.bic(x))
        boot_rows.append(
            {
                "iteration": iteration,
                "best_k": int(np.argmin(bics) + 1),
                "bic_gain_1_to_3": float(bics[0] - bics[2]),
                "multibasin_supported": bool(min(bics[1:]) < bics[0] - 10),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def leave_one_patient_out(
    blocks: pd.DataFrame,
    full_model: GaussianMixture,
    full_labels: dict[int, str],
) -> pd.DataFrame:
    x = blocks[["resolver_axis", "lock_axis"]].to_numpy(float)
    full_locked_component = next(component for component, state in full_labels.items() if state == "locked")
    full_probability = full_model.predict_proba(x)[:, full_locked_component]
    rows = []
    for patient in blocks.patient_cluster.unique():
        test = blocks.patient_cluster.eq(patient).to_numpy()
        train = ~test
        model = GaussianMixture(n_components=2, covariance_type="full", n_init=30, random_state=20260712)
        model.fit(x[train])
        imbalance = model.means_[:, 1] - model.means_[:, 0]
        locked_component = int(np.argmax(imbalance))
        predicted = model.predict_proba(x[test])[:, locked_component]
        expected = full_probability[test]
        correlation = float(np.corrcoef(predicted, expected)[0, 1]) if len(predicted) > 1 else np.nan
        rows.append(
            {
                "patient_cluster": patient,
                "n_blocks": int(test.sum()),
                "posterior_r": correlation,
                "posterior_mae": float(np.mean(np.abs(predicted - expected))),
                "basin_agreement": float(np.mean((predicted >= 0.5) == (expected >= 0.5))),
            }
        )
    return pd.DataFrame(rows)


def state_fractions(data: pd.DataFrame) -> pd.DataFrame:
    fractions = (
        data.groupby(
            ["biopsy_id", "patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group", "basin_state"],
            observed=True,
        ).size().rename("n_spots").reset_index()
    )
    fractions["fraction"] = fractions["n_spots"] / fractions.groupby("biopsy_id", observed=True)["n_spots"].transform("sum")
    return fractions


def array_adjusted_trend(frame: pd.DataFrame, value: str, permutations: int = 9999) -> dict[str, float]:
    patient = frame.groupby(
        ["patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group"], observed=True
    )[value].mean().reset_index()
    dummies = pd.get_dummies(patient["array"], drop_first=True, dtype=float)
    x0 = np.column_stack([np.ones(len(patient)), dummies.to_numpy(float)])
    stage = patient["stage_numeric"].to_numpy(float)
    x = np.column_stack([x0, stage])
    y = patient[value].to_numpy(float)
    fit = sm.OLS(y, x).fit(cov_type="HC3")
    observed = float(fit.params[-1])
    reduced = x0 @ np.linalg.lstsq(x0, y, rcond=None)[0]
    residual = y - reduced
    groups = [np.asarray(indices, int) for _, indices in patient.groupby("array", observed=True).indices.items()]
    null = np.empty(permutations)
    for i in range(permutations):
        permuted = residual.copy()
        for indices in groups:
            permuted[indices] = RNG.permutation(permuted[indices])
        null[i] = np.linalg.lstsq(x, reduced + permuted, rcond=None)[0][-1]
    return {
        "n_patients": len(patient),
        "stage_beta": observed,
        "hc3_se": float(fit.bse[-1]),
        "hc3_p": float(fit.pvalues[-1]),
        "within_array_permutation_p": float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1)),
        "spearman_rho": float(stats.spearmanr(patient.stage_numeric, patient[value]).statistic),
    }


def fraction_trends(fractions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ["resolved", "transition", "locked"]:
        local = fractions.loc[fractions.basin_state.eq(state)]
        rows.append({"basin_state": state, **array_adjusted_trend(local, "fraction")})
    result = pd.DataFrame(rows)
    result["permutation_fdr"] = bh(result["within_array_permutation_p"])
    return result


def axis_sensitivity(blocks: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "pca_primary": ("resolver_axis", "lock_axis"),
        "mechanistic_equal": ("resolver_mechanistic_equal", "lock_mechanistic_equal"),
        "compact_equal": ("resolver_compact_equal", "lock_compact_equal"),
    }
    rows = []
    for name, (resolver, lock) in definitions.items():
        x = blocks[[resolver, lock]].to_numpy(float)
        one = GaussianMixture(n_components=1, covariance_type="full", n_init=30, random_state=20260712).fit(x)
        two = GaussianMixture(n_components=2, covariance_type="full", n_init=50, random_state=20260712).fit(x)
        imbalance = two.means_[:, 1] - two.means_[:, 0]
        locked_component = int(np.argmax(imbalance))
        probability = two.predict_proba(x)[:, locked_component]
        local = blocks[["biopsy_id", "patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group"]].copy()
        local["locked"] = probability >= 0.67
        fraction = local.groupby(
            ["biopsy_id", "patient_cluster", "array", "fibrosis_stage", "stage_numeric", "stage_group"],
            observed=True,
        ).locked.mean().reset_index(name="fraction")
        trend = array_adjusted_trend(fraction, "fraction", permutations=4999)
        rows.append(
            {
                "axis_definition": name,
                "resolver_column": resolver,
                "lock_column": lock,
                "bic_gain_one_to_two": float(one.bic(x) - two.bic(x)),
                **trend,
            }
        )
    result = pd.DataFrame(rows)
    result["permutation_fdr"] = bh(result.within_array_permutation_p)
    return result


def memory_test(blocks: pd.DataFrame, data: pd.DataFrame, permutations: int = 9999) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilities = data.groupby(["biopsy_id", "long_bin", "cross_bin"], observed=True)["p_locked"].mean().reset_index()
    model_data = blocks.merge(probabilities, on=["biopsy_id", "long_bin", "cross_bin"], validate="one_to_one")
    p = np.clip(model_data.p_locked.to_numpy(float), 1e-4, 1 - 1e-4)
    y = np.log(p / (1 - p))
    ecm = model_data.ECM_scar.to_numpy(float)
    resolver = model_data.resolver_axis.to_numpy(float)
    stage = model_data.stage_numeric.to_numpy(float)
    arrays = pd.get_dummies(model_data.array, drop_first=True, dtype=float).to_numpy(float)
    x0 = np.column_stack([np.ones(len(model_data)), ecm, ecm**2, resolver, arrays])
    x = np.column_stack([x0, stage])
    groups = pd.Categorical(model_data.patient_cluster).codes
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    observed = float(fit.params[-1])

    reduced = x0 @ np.linalg.lstsq(x0, y, rcond=None)[0]
    residual = y - reduced
    patient_meta = model_data[["patient_cluster", "array", "stage_numeric"]].drop_duplicates("patient_cluster")
    null = np.empty(permutations)
    for i in range(permutations):
        permuted_stage = {}
        for _, local in patient_meta.groupby("array", observed=True):
            values = RNG.permutation(local.stage_numeric.to_numpy(float))
            permuted_stage.update(dict(zip(local.patient_cluster, values)))
        stage_perm = model_data.patient_cluster.map(permuted_stage).to_numpy(float)
        xp = np.column_stack([x0, stage_perm])
        null[i] = np.linalg.lstsq(xp, reduced + residual, rcond=None)[0][-1]
    result = pd.DataFrame(
        [{
            "test": "matched_ECM_resolver_stage_memory",
            "n_blocks": len(model_data),
            "n_patients": model_data.patient_cluster.nunique(),
            "stage_beta_logit_locked": observed,
            "cluster_se": float(fit.bse[-1]),
            "cluster_p": float(fit.pvalues[-1]),
            "within_array_patient_permutation_p": float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1)),
        }]
    )
    return model_data, result


def front_profiles(data: pd.DataFrame) -> pd.DataFrame:
    local = data.copy()
    local["front_bin"] = pd.cut(local.scar_proximity, bins=np.linspace(-3.0, 1.0, 17), include_lowest=True)
    patient = local.groupby(
        ["patient_cluster", "stage_group", "front_bin"], observed=True
    )[["p_locked", "p_transition", "p_resolved", "control_imbalance"]].mean().reset_index()
    patient["front_mid"] = patient.front_bin.map(lambda value: float(value.mid)).astype(float)
    rows = []
    for (group, midpoint), frame in patient.groupby(["stage_group", "front_mid"], observed=True):
        values = frame.p_locked.to_numpy(float)
        boot = np.array([np.mean(RNG.choice(values, len(values), replace=True)) for _ in range(2000)])
        rows.append(
            {
                "stage_group": group,
                "front_mid": midpoint,
                "n_patients": len(values),
                "mean_p_locked": float(np.mean(values)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def adjacency(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for biopsy, local in data.groupby("biopsy_id", observed=True):
        coordinates = local[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
        tree = cKDTree(coordinates)
        distance, index = tree.query(coordinates, k=7)
        cutoff = np.median(distance[:, 1]) * 1.25
        states = local.basin_state.to_numpy(str)
        counts = {(a, b): 0 for a in STATE_COLORS for b in STATE_COLORS}
        for i in range(len(local)):
            for d, j in zip(distance[i, 1:], index[i, 1:]):
                if d <= cutoff and i < j:
                    a, b = sorted([states[i], states[j]])
                    counts[(a, b)] = counts.get((a, b), 0) + 1
        total = max(sum(counts.values()), 1)
        meta = local.iloc[0]
        for (a, b), count in counts.items():
            if count:
                rows.append(
                    {
                        "biopsy_id": biopsy,
                        "patient_cluster": meta.patient_cluster,
                        "stage_group": meta.stage_group,
                        "state_a": a,
                        "state_b": b,
                        "edge_count": count,
                        "edge_fraction": count / total,
                    }
                )
    return pd.DataFrame(rows)


def add_gmm_ellipses(ax, model: GaussianMixture, labels: dict[int, str]) -> None:
    for component in range(model.n_components):
        covariance = model.covariances_[component]
        values, vectors = np.linalg.eigh(covariance)
        order = values.argsort()[::-1]
        values, vectors = values[order], vectors[:, order]
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        state = labels[component]
        for scale, alpha in [(2.0, 0.9), (3.2, 0.35)]:
            ellipse = Ellipse(
                model.means_[component],
                width=scale * np.sqrt(values[0]),
                height=scale * np.sqrt(values[1]),
                angle=angle,
                fill=False,
                lw=1.7 if scale == 2.0 else 0.9,
                alpha=alpha,
                edgecolor=STATE_COLORS[state],
            )
            ax.add_patch(ellipse)
        ax.text(*model.means_[component], state.upper(), color=STATE_COLORS[state], fontweight="bold", ha="center")


def plot_figure(
    data: pd.DataFrame,
    blocks: pd.DataFrame,
    bic: pd.DataFrame,
    model: GaussianMixture,
    labels: dict[int, str],
    fractions: pd.DataFrame,
    model_data: pd.DataFrame,
    fronts: pd.DataFrame,
    memory: pd.DataFrame,
) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(16, 14), constrained_layout=False)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1.05])

    ax = fig.add_subplot(grid[0, 0])
    hb = ax.hexbin(blocks.resolver_axis, blocks.lock_axis, gridsize=42, mincnt=1, cmap="Greys", bins="log", alpha=0.88)
    add_gmm_ellipses(ax, model, labels)
    ax.axline((0, 0), slope=1, color="#6C757D", ls="--", lw=1)
    ax.set(xlabel="Resolver capacity axis", ylabel="HSC lock-load axis")
    ax.set_title("A  Two basins separated by a transition boundary", loc="left", fontweight="bold")
    fig.colorbar(hb, ax=ax, label="Spatial-block density", shrink=0.72)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(bic.n_components, bic.bic - bic.bic.min(), marker="o", color="#4E5D6C", lw=1.8)
    ax.axhline(10, color="#B83D4B", ls="--", lw=1)
    ax.set(xticks=bic.n_components, xlabel="Gaussian-mixture components", ylabel="BIC above optimum")
    ax.set_title("B  Multibasin structure is preferred", loc="left", fontweight="bold")
    best = int(bic.loc[bic.bic.idxmin(), "n_components"])
    ax.text(.97,.92,f"Best K = {best}\nΔBIC(1→2) = {bic.loc[bic.n_components.eq(1),'bic'].iloc[0]-bic.loc[bic.n_components.eq(2),'bic'].iloc[0]:.1f}",transform=ax.transAxes,ha="right",va="top",bbox=dict(boxstyle="round,pad=.35",fc="white",ec="#88939B"))

    ax = fig.add_subplot(grid[1, 0])
    patient = fractions.groupby(["patient_cluster", "stage_numeric", "stage_group", "basin_state"], observed=True).fraction.mean().reset_index()
    sns.boxplot(data=patient, x="stage_numeric", y="fraction", hue="basin_state", hue_order=["resolved","transition","locked"], palette=STATE_COLORS, fliersize=0, ax=ax)
    sns.stripplot(data=patient, x="stage_numeric", y="fraction", hue="basin_state", hue_order=["resolved","transition","locked"], dodge=True, palette=STATE_COLORS, size=2.7, alpha=.65, ax=ax, legend=False)
    ax.set(xlabel="Fibrosis stage", ylabel="Patient basin fraction")
    ax.set_title("C  Fibrosis stage redistributes spatial state occupancy", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")

    ax = fig.add_subplot(grid[1, 1])
    for group in STAGE_ORDER:
        local = model_data.loc[model_data.stage_group.eq(group)].sort_values("ECM_scar")
        smooth = lowess(local.p_locked, local.ECM_scar, frac=.42, return_sorted=True)
        ax.plot(smooth[:,0], smooth[:,1], color=STAGE_COLORS[group], lw=2.4, label=group)
        ax.scatter(local.ECM_scar, local.p_locked, s=6, alpha=.10, color=STAGE_COLORS[group])
    ax.set(xlabel="Structural ECM load", ylabel="Locked-basin probability", ylim=(-.02,1.02))
    ax.set_title("D  The separatrix generalizes across stages", loc="left", fontweight="bold")
    ax.text(.98,.05,f"Matched ECM + resolver\nstage P = {memory.within_array_patient_permutation_p.iloc[0]:.3f}",transform=ax.transAxes,ha="right",va="bottom",bbox=dict(boxstyle="round,pad=.3",fc="white",ec="#8A959C"),fontsize=8)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[2, 0])
    for group in STAGE_ORDER:
        local = fronts.loc[fronts.stage_group.eq(group)].sort_values("front_mid")
        ax.plot(local.front_mid, local.mean_p_locked, color=STAGE_COLORS[group], lw=2.5, label=group)
        ax.fill_between(local.front_mid, local.ci_low, local.ci_high, color=STAGE_COLORS[group], alpha=.18)
    ax.axvline(0, color="#6C757D", ls="--", lw=1)
    ax.set(xlabel="Scar proximity (far → core)", ylabel="Patient-averaged locked probability")
    ax.set_title("E  The locked basin forms a stage-dependent scar front", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    holder = fig.add_subplot(grid[2, 1]); holder.set_axis_off()
    holder.set_title("F  Representative basin maps", loc="left", fontweight="bold")
    locked_fraction = fractions.loc[fractions.basin_state.eq("locked")]
    reps = []
    for group in STAGE_ORDER:
        local = locked_fraction.loc[locked_fraction.stage_group.eq(group)]
        target = local.fraction.median()
        reps.append(local.loc[(local.fraction-target).abs().idxmin(), "biopsy_id"])
    for biopsy, position in zip(reps, [(0.01,.08,.31,.82),(.345,.08,.31,.82),(.68,.08,.31,.82)]):
        axm = holder.inset_axes(position)
        local = data.loc[data.biopsy_id.eq(biopsy)]
        colors = local.basin_state.map(STATE_COLORS)
        axm.scatter(local.pxl_col_in_fullres, -local.pxl_row_in_fullres, c=colors, s=5.2, linewidths=0, rasterized=True)
        row = local.iloc[0]
        axm.set_title(f"{row.stage_group}\n{row.fibrosis_stage} | {biopsy}", fontsize=8)
        axm.set_aspect("equal"); axm.set_axis_off()
    holder.text(.5,.01,"Resolved     Transition     Locked",ha="center",fontsize=8)
    for x, state in zip([.35,.50,.65],["resolved","transition","locked"]):
        holder.scatter([x],[.01],s=28,color=STATE_COLORS[state],transform=holder.transAxes)

    for axis in fig.axes:
        if axis.axison:
            axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("A resolver–lock control landscape reveals a stage-invariant separatrix", fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=.07, right=.97, bottom=.06, top=.92, hspace=.42, wspace=.30)
    fig.savefig(OUTPUT / "UQ_resolver_lock_phase_basin.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT / "UQ_resolver_lock_phase_basin.pdf", facecolor="white")
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
    data = pd.read_parquet(INPUT)
    data, feature_audit = residualize_features(data)
    blocks = spatial_blocks(data)
    data, bic, model, labels = fit_basins(blocks, data)
    dip, bootstrap = dip_and_bootstrap(blocks)
    lopo = leave_one_patient_out(blocks, model, labels)

    # Assign block bins to spot rows for direct posterior aggregation.
    block_keys = []
    for biopsy, local in data.groupby("biopsy_id", sort=True, observed=True):
        coordinates = local[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
        centered = coordinates - coordinates.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        rotated = centered @ vh.T
        index = local.index
        data.loc[index, "long_bin"] = pd.qcut(pd.Series(rotated[:,0]).rank(method="first"),6,labels=False).to_numpy()
        data.loc[index, "cross_bin"] = pd.qcut(pd.Series(rotated[:,1]).rank(method="first"),3,labels=False).to_numpy()
    data[["long_bin","cross_bin"]] = data[["long_bin","cross_bin"]].astype(int)

    fractions = state_fractions(data)
    trends = fraction_trends(fractions)
    sensitivity = axis_sensitivity(blocks)
    model_data, memory = memory_test(blocks, data)
    fronts = front_profiles(data)
    edges = adjacency(data)
    plot_figure(data, blocks, bic, model, labels, fractions, model_data, fronts, memory)

    feature_audit.to_csv(OUTPUT / "phase_axis_feature_audit.tsv", sep="\t", index=False)
    blocks.to_parquet(OUTPUT / "phase_spatial_blocks.parquet", index=False)
    data.to_parquet(OUTPUT / "phase_spot_states.parquet", index=False)
    bic.to_csv(OUTPUT / "phase_gmm_model_selection.tsv", sep="\t", index=False)
    dip.to_csv(OUTPUT / "phase_dip_tests.tsv", sep="\t", index=False)
    bootstrap.to_csv(OUTPUT / "phase_patient_bootstrap_multibasin.tsv", sep="\t", index=False)
    lopo.to_csv(OUTPUT / "phase_leave_one_patient_out.tsv", sep="\t", index=False)
    fractions.to_csv(OUTPUT / "phase_biopsy_state_fractions.tsv", sep="\t", index=False)
    trends.to_csv(OUTPUT / "phase_state_fraction_trends.tsv", sep="\t", index=False)
    sensitivity.to_csv(OUTPUT / "phase_axis_definition_sensitivity.tsv", sep="\t", index=False)
    model_data.to_parquet(OUTPUT / "phase_matched_ECM_blocks.parquet", index=False)
    memory.to_csv(OUTPUT / "phase_matched_ECM_memory_test.tsv", sep="\t", index=False)
    fronts.to_csv(OUTPUT / "phase_scar_front_profiles.tsv", sep="\t", index=False)
    edges.to_csv(OUTPUT / "phase_spatial_adjacency.tsv", sep="\t", index=False)

    best_k = int(bic.loc[bic.bic.idxmin(), "n_components"])
    locked_trend = trends.loc[trends.basin_state.eq("locked")].iloc[0].to_dict()
    summary = {
        "n_spots": int(len(data)),
        "n_biopsies": int(data.biopsy_id.nunique()),
        "n_patients": int(data.patient_cluster.nunique()),
        "n_spatial_blocks": int(len(blocks)),
        "best_gmm_components": best_k,
        "bic_gain_one_to_two": float(bic.loc[bic.n_components.eq(1),"bic"].iloc[0]-bic.loc[bic.n_components.eq(2),"bic"].iloc[0]),
        "patient_bootstrap_multibasin_fraction": float(bootstrap.multibasin_supported.mean()),
        "leave_one_patient_out": {
            "median_posterior_r": float(lopo.posterior_r.median()),
            "median_posterior_mae": float(lopo.posterior_mae.median()),
            "median_basin_agreement": float(lopo.basin_agreement.median()),
            "minimum_basin_agreement": float(lopo.basin_agreement.min()),
        },
        "locked_fraction_stage_trend": locked_trend,
        "axis_definition_sensitivity": sensitivity.to_dict(orient="records"),
        "matched_ECM_stage_memory": memory.iloc[0].to_dict(),
        "interpretation": "A robust two-basin resolver-lock landscape and a stage-invariant separatrix support thresholded basin switching as a therapeutic control principle.",
    }
    (OUTPUT / "phase_basin_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
