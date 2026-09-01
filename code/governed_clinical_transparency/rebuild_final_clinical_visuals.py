#!/usr/bin/env python3
"""Rebuild the proprietary clinical evidence layer and publication figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
CLIN = REPO_ROOT / "data/governed/clinical"
DYN = REPO_ROOT / "data/governed/dynamics"
OUTPUT_ROOT = REPO_ROOT / "results/clinical"
TABLES = OUTPUT_ROOT / "source_tables"
MAIN = OUTPUT_ROOT / "main_figures"
ED = OUTPUT_ROOT / "extended_data_figures"
LOGS = OUTPUT_ROOT / "logs"


COLORS = {
    "blue": "#2678B2",
    "cyan": "#47B5C4",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#D1495B",
    "magenta": "#B05A8F",
    "purple": "#6C5B7B",
    "gray": "#8A8F98",
    "lightgray": "#E8EAED",
    "dark": "#20242A",
}
STATE_COLORS = {
    "LL": "#4C9F70",
    "HL": "#E9C46A",
    "LH": "#8F6BAE",
    "HH": "#D1495B",
}
STATE_LABELS = {
    "LL": "Low CAP / low LSM",
    "HL": "High CAP / low LSM",
    "LH": "Low CAP / high LSM",
    "HH": "High CAP / high LSM",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "figure.dpi": 160,
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "text.color": COLORS["dark"],
        "axes.labelcolor": COLORS["dark"],
        "xtick.color": COLORS["dark"],
        "ytick.color": COLORS["dark"],
    }
)
sns.set_style("white")


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=7)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor="white")
    plt.close(fig)


def median_iqr(s: pd.Series, digits: int = 1) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def n_pct(mask: pd.Series) -> str:
    x = mask.dropna().astype(bool)
    return f"{int(x.sum()):,} ({100 * x.mean():.1f}%)"


def smd_cont(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else np.nan


def smd_binary(a: pd.Series, b: pd.Series) -> float:
    pa, pb = a.astype(float).mean(), b.astype(float).mean()
    pooled = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pb - pa) / pooled) if pooled > 0 else np.nan


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n_boot: int = 1200, seed: int = 20260712) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    estimate = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        ind = rng.integers(0, n, n)
        if np.unique(y[ind]).size == 2:
            vals.append(roc_auc_score(y[ind], p[ind]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(estimate), float(lo), float(hi)


def odds_ratio_glm(data: pd.DataFrame, outcome: str, predictor: str) -> tuple[float, float, float, float]:
    d = data[[outcome, predictor]].dropna().copy()
    d[outcome] = d[outcome].astype(int)
    d[predictor] = (d[predictor] - d[predictor].mean()) / d[predictor].std(ddof=1)
    model = sm.GLM(d[outcome], sm.add_constant(d[predictor]), family=sm.families.Binomial()).fit()
    beta = model.params[predictor]
    lo, hi = model.conf_int().loc[predictor]
    return float(np.exp(beta)), float(np.exp(lo)), float(np.exp(hi)), float(model.pvalues[predictor])


def state_code(cap: pd.Series, lsm: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                (cap < 280) & (lsm < 9.5),
                (cap >= 280) & (lsm < 9.5),
                (cap < 280) & (lsm >= 9.5),
                (cap >= 280) & (lsm >= 9.5),
            ],
            ["LL", "HL", "LH", "HH"],
            default="NA",
        ),
        index=cap.index,
    )


def build_longitudinal(elasto: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pid, group in elasto.groupby("medical_record_hash", sort=False):
        group = group.drop_duplicates("scan_date").sort_values("scan_timestamp")
        if len(group) < 2:
            continue
        first, last = group.iloc[0], group.iloc[-1]
        days = int((last.scan_date - first.scan_date).days)
        if days < 90:
            continue
        rows.append(
            {
                "patient": pid,
                "n_scans": len(group),
                "baseline_date": first.scan_date,
                "followup_days": days,
                "age": first.age_at_scan,
                "male": int(first.sex == "M"),
                "BMI": first.BMI,
                "CAP0": first.CAP,
                "CAP1": last.CAP,
                "LSM0": first.LSM,
                "LSM1": last.LSM,
            }
        )
    d = pd.DataFrame(rows)
    d["state0"] = state_code(d.CAP0, d.LSM0)
    d["state1"] = state_code(d.CAP1, d.LSM1)
    d["delta_CAP"] = d.CAP1 - d.CAP0
    d["relative_LSM"] = d.LSM1 / d.LSM0 - 1
    d["advanced0"] = d.LSM0 >= 9.5
    d["advanced1"] = d.LSM1 >= 9.5
    d["outcome"] = np.select(
        [
            d.advanced0 & ~d.advanced1,
            ~d.advanced0 & d.advanced1,
            d.advanced0 & d.advanced1,
            ~d.advanced0 & ~d.advanced1,
        ],
        ["Threshold regression", "Threshold progression", "Persistent advanced", "Persistent nonadvanced"],
        default="Unknown",
    )
    d["lsm_change"] = np.select(
        [d.relative_LSM <= -0.20, d.relative_LSM >= 0.20],
        ["Decrease >=20%", "Increase >=20%"],
        default="Stable within 20%",
    )
    d["regression_resistant"] = d.advanced0 & d.advanced1 & (d.relative_LSM > -0.20)
    d["baseline_era"] = np.where(d.baseline_date.dt.year <= 2024, "2023-2024", "2025-2026")
    return d


def baseline_table(index: pd.DataFrame) -> pd.DataFrame:
    low = index.loc[~index.advanced_fibrosis_lsm_ge9_5]
    high = index.loc[index.advanced_fibrosis_lsm_ge9_5]
    specs = [
        ("Age, years", "continuous", "age_at_scan", 1),
        ("Male sex", "binary", "male", 1),
        ("BMI, kg/m2", "continuous", "BMI", 1),
        ("CAP, dB/m", "continuous", "CAP", 1),
        ("LSM, kPa", "continuous", "LSM", 1),
        ("CAP >=248 dB/m", "binary", "cap248", 1),
        ("CAP >=280 dB/m", "binary", "cap280", 1),
        ("LSM 9.5-<12.5 kPa", "binary", "f3", 1),
        ("LSM >=12.5 kPa", "binary", "f4", 1),
        ("Gastro/hepatobiliary", "binary", "gi", 1),
    ]
    rows = []
    for label, kind, col, digits in specs:
        if kind == "continuous":
            rows.append(
                {
                    "Variable": label,
                    "All (n=8,142)": median_iqr(index[col], digits),
                    "LSM <9.5 (n=5,857)": median_iqr(low[col], digits),
                    "LSM >=9.5 (n=2,285)": median_iqr(high[col], digits),
                    "SMD": smd_cont(low[col], high[col]),
                }
            )
        else:
            rows.append(
                {
                    "Variable": label,
                    "All (n=8,142)": n_pct(index[col]),
                    "LSM <9.5 (n=5,857)": n_pct(low[col]),
                    "LSM >=9.5 (n=2,285)": n_pct(high[col]),
                    "SMD": smd_binary(low[col], high[col]),
                }
            )
    return pd.DataFrame(rows)


def validation_analysis(index: pd.DataFrame) -> pd.DataFrame:
    features = ["age_at_scan", "male", "BMI", "CAP"]
    train = index.loc[index.scan_date.dt.year <= 2024].copy()
    temporal = index.loc[index.scan_date.dt.year >= 2025].copy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20260712))
    cv = StratifiedKFold(5, shuffle=True, random_state=20260712)
    cv_pred = cross_val_predict(model, train[features], train.advanced.astype(int), cv=cv, method="predict_proba")[:, 1]
    model.fit(train[features], train.advanced.astype(int))
    temporal_pred = model.predict_proba(temporal[features])[:, 1]

    gi = index.loc[index.gi].copy()
    other = index.loc[~index.gi].copy()
    dept_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20260712))
    dept_model.fit(gi[features], gi.advanced.astype(int))
    other_pred = dept_model.predict_proba(other[features])[:, 1]

    rows = []
    for label, data, pred in [
        ("Internal 5-fold CV (2023-2024)", train, cv_pred),
        ("Temporal holdout (2025-2026)", temporal, temporal_pred),
        ("Independent department subcohort", other, other_pred),
    ]:
        est, lo, hi = bootstrap_auc(data.advanced.to_numpy(), pred)
        rows.append(
            {
                "validation": label,
                "n": len(data),
                "advanced_n": int(data.advanced.sum()),
                "advanced_fraction": float(data.advanced.mean()),
                "AUROC": est,
                "ci_low": lo,
                "ci_high": hi,
                "Brier": brier_score_loss(data.advanced.astype(int), pred),
            }
        )
    return pd.DataFrame(rows)


def draw_workflow(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel(ax, "A", "Proprietary clinical cohort, fixed eligibility, and de-identification workflow")
    items = [
        ("Raw elastography", "9,703 scans\n2023-03-01 to 2026-07-09", COLORS["blue"]),
        ("Technical QC", "9,644 scans\n>=10 valid; LSM IQR/M <=0.30", COLORS["cyan"]),
        ("Adult analysis", "9,562 scans\nage >=18 years", COLORS["teal"]),
        ("Index cohort", "8,142 patients\none scan per hashed MRN", COLORS["gold"]),
        ("Clinical linkage", "Metabolic 1,490 | LIS 746\nthree-source 243 scans", COLORS["red"]),
    ]
    x0, gap, width, height, y = 0.015, 0.018, 0.175, 0.60, 0.18
    for i, (head, body, color) in enumerate(items):
        x = x0 + i * (width + gap)
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            facecolor="white", edgecolor=color, linewidth=1.6,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.add_patch(Rectangle((x, y + height - 0.11), width, 0.11, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.95))
        ax.text(x + 0.012, y + height - 0.055, head, transform=ax.transAxes, color="white", fontweight="bold", va="center", fontsize=8.2)
        ax.text(x + width / 2, y + 0.24, body, transform=ax.transAxes, ha="center", va="center", fontsize=7.2, linespacing=1.25)
        if i < len(items) - 1:
            ax.annotate("", xy=(x + width + gap - 0.004, y + height / 2), xytext=(x + width + 0.004, y + height / 2), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=1.2))
    ax.text(0.015, 0.02, "Single center: Fudan University Affiliated Pudong Hospital", transform=ax.transAxes, fontsize=7.4, fontweight="bold")
    ax.text(0.42, 0.02, "Linkage: laboratory-verified unique patient key; nearest result within +/-90 days", transform=ax.transAxes, fontsize=7.2)
    ax.text(0.985, 0.02, "Ethics details: AUTHORS TO COMPLETE BEFORE SUBMISSION", transform=ax.transAxes, ha="right", fontsize=7.2, color=COLORS["red"], fontweight="bold")


def figure_main_1(index: pd.DataFrame, baseline: pd.DataFrame, etiology: pd.DataFrame, validation: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16.2, 10.4))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.25, 3.1, 2.8], hspace=0.55, wspace=1.00)
    draw_workflow(fig.add_subplot(gs[0, :]))

    ax = fig.add_subplot(gs[1:, :4])
    panel(ax, "B", "Baseline profile of 8,142 adults")
    ax.axis("off")
    show = baseline.copy().rename(
        columns={
            "All (n=8,142)": "All\n(n=8,142)",
            "LSM <9.5 (n=5,857)": "LSM <9.5\n(n=5,857)",
            "LSM >=9.5 (n=2,285)": "LSM >=9.5\n(n=2,285)",
        }
    )
    show["SMD"] = show.SMD.map(lambda x: f"{x:.2f}")
    table = ax.table(cellText=show.values, colLabels=show.columns, cellLoc="left", colLoc="left", bbox=[0, 0.01, 1, 0.96], colWidths=[0.34, 0.22, 0.23, 0.23, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(COLORS["dark"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F5F6")
        if col == 0 and row > 0:
            cell.get_text().set_fontweight("bold")
    ax.text(0, -0.03, "Values are median [IQR] or n (%). SMD compares LSM strata.", transform=ax.transAxes, fontsize=6.8, color=COLORS["gray"])

    ax = fig.add_subplot(gs[1, 4:8])
    panel(ax, "C", "LSM-defined imaging strata across calendar years")
    stage_order = ["<8.0", "8.0-<9.5", "9.5-<12.5", ">=12.5"]
    stage_colors = [COLORS["teal"], COLORS["gold"], COLORS["orange"], COLORS["red"]]
    imaging_stratum = pd.cut(index.LSM, [-np.inf, 8.0, 9.5, 12.5, np.inf], right=False, labels=stage_order)
    tab = pd.crosstab(index.scan_date.dt.year, imaging_stratum).reindex(columns=stage_order, fill_value=0)
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(pct))
    for stage, color in zip(stage_order, stage_colors):
        ax.bar(pct.index.astype(str), pct[stage], bottom=bottom, color=color, width=0.68, label=stage, edgecolor="white", linewidth=0.5)
        bottom += pct[stage].to_numpy()
    for i, year in enumerate(pct.index):
        ax.text(i, 102, f"n={tab.loc[year].sum():,}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(0, 109)
    ax.set_ylabel("Patients (%)")
    ax.set_xlabel("Index year")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28), title="LSM (kPa)")

    ax = fig.add_subplot(gs[1, 8:12])
    panel(ax, "D", "CAP-LSM landscape separates steatotic and stiffness axes")
    hb = ax.hexbin(index.CAP, index.LSM, gridsize=48, mincnt=1, bins="log", cmap="viridis", linewidths=0, extent=[150, 400, 2.5, 33])
    ax.axvline(280, color=COLORS["red"], ls="--", lw=1.1)
    ax.axhline(9.5, color=COLORS["red"], ls="--", lw=1.1)
    counts = index.state.value_counts()
    positions = {"LL": (175, 4.5), "HL": (316, 4.5), "LH": (175, 27.5), "HH": (316, 27.5)}
    for st, (x, y) in positions.items():
        ax.text(x, y, f"{st}\nn={counts.get(st, 0):,}", ha="center", va="center", fontsize=7.2, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=STATE_COLORS[st], lw=1.1, alpha=0.9))
    ax.set_xlim(150, 400)
    ax.set_ylim(2.5, 33)
    ax.set_xlabel("CAP (dB/m)")
    ax.set_ylabel("LSM (kPa)")
    cb = fig.colorbar(hb, ax=ax, fraction=0.047, pad=0.02)
    cb.set_label("log10 density", fontsize=7)
    ax.text(0.02, 0.02, "Spearman rho=0.385", transform=ax.transAxes, fontsize=6.8, bbox=dict(fc="white", ec="none", alpha=0.8))
    top = inset_axes(ax, width="82%", height="15%", loc="upper left", borderpad=0.6)
    top.hist(index.CAP, bins=44, color=COLORS["blue"], alpha=0.55)
    top.patch.set_alpha(0.82)
    top.axis("off")
    right = inset_axes(ax, width="15%", height="72%", loc="upper right", borderpad=0.6)
    right.hist(index.LSM, bins=38, orientation="horizontal", color=COLORS["purple"], alpha=0.55)
    right.patch.set_alpha(0.82)
    right.axis("off")

    ax = fig.add_subplot(gs[2, 4:8])
    panel(ax, "E", "Observed etiology markers and partial phenotypes")
    order = [
        "metabolic_risk_observed",
        "metabolic_steatotic_phenotype",
        "diagnosis_steatotic_liver_text",
        "HBsAg_marker_positive",
        "AMA_M2_marker_positive",
        "anti_HCV_marker_positive",
    ]
    labels = {
        "metabolic_risk_observed": "Metabolic risk",
        "metabolic_steatotic_phenotype": "CAP >=248 + metabolic",
        "diagnosis_steatotic_liver_text": "Steatotic-liver text",
        "HBsAg_marker_positive": "HBsAg positive",
        "AMA_M2_marker_positive": "AMA-M2 positive",
        "anti_HCV_marker_positive": "anti-HCV positive",
    }
    d = etiology.set_index("marker_or_phenotype").loc[order].reset_index()
    y = np.arange(len(d))[::-1]
    ax.barh(y, 100 * d.proportion, color=[COLORS["gold"], COLORS["orange"], COLORS["gray"], COLORS["blue"], COLORS["magenta"], COLORS["purple"]], height=0.65)
    ax.set_yticks(y, [labels[x] for x in d.marker_or_phenotype])
    ax.tick_params(axis="y", labelsize=6.4, pad=2)
    ax.set_xlabel("Positive among observed (%)")
    ax.set_xlim(0, 100)
    for yi, row in zip(y, d.itertuples()):
        ax.text(min(98, 100 * row.proportion + 2), yi, f"{row.n_positive}/{row.n_observed}", va="center", fontsize=6.8)
    ax.text(0.0, -0.25, "Markers and text-derived phenotypes are not adjudicated etiologies.", transform=ax.transAxes, fontsize=6.8, color=COLORS["gray"])

    ax = fig.add_subplot(gs[2, 8:12])
    panel(ax, "F", "Internal, temporal, and independent-subcohort validation")
    y = np.arange(len(validation))[::-1]
    short_validation = {
        "Internal 5-fold CV (2023-2024)": "Internal 5-fold CV\n2023-2024",
        "Temporal holdout (2025-2026)": "Temporal holdout\n2025-2026",
        "Independent department subcohort": "Other-department\nsubcohort",
    }
    for yi, row in zip(y, validation.itertuples()):
        ax.errorbar(row.AUROC, yi, xerr=[[row.AUROC - row.ci_low], [row.ci_high - row.AUROC]], fmt="o", color=COLORS["blue"], capsize=3, lw=1.3, ms=6)
        ax.text(row.ci_high + 0.012, yi, f"{row.AUROC:.3f} ({row.ci_low:.3f}-{row.ci_high:.3f})", va="center", fontsize=6.8)
    ax.axvline(0.5, color=COLORS["gray"], ls="--", lw=1)
    ax.set_yticks(y, [f"{short_validation[r.validation]}\nn={r.n:,}" for r in validation.itertuples()])
    ax.tick_params(axis="y", labelsize=6.4, pad=2)
    ax.set_xlim(0.45, 0.86)
    ax.set_xlabel("AUROC for LSM >=9.5 kPa")
    ax.text(0.0, -0.25, "Frozen predictors: age, sex, BMI, and CAP. Other-department validation excludes gastro/hepatobiliary sources.", transform=ax.transAxes, fontsize=6.7, color=COLORS["gray"])

    fig.suptitle("Figure 1 | Proprietary clinical cohort architecture and baseline landscape", x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold")
    save_figure(fig, MAIN / "Figure_1")


def draw_alluvial(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    order = ["LL", "HL", "LH", "HH"]
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    total = matrix.to_numpy().sum()
    left_totals = matrix.sum(axis=1).to_numpy()
    right_totals = matrix.sum(axis=0).to_numpy()
    gap = 0.025
    usable = 0.90 - gap * (len(order) - 1)

    def spans(totals: np.ndarray) -> dict[str, tuple[float, float]]:
        cursor = 0.05
        out = {}
        for st, n in zip(order, totals):
            h = usable * n / total
            out[st] = (cursor, cursor + h)
            cursor += h + gap
        return out

    left = spans(left_totals)
    right = spans(right_totals)
    left_cursor = {k: v[0] for k, v in left.items()}
    right_cursor = {k: v[0] for k, v in right.items()}
    x0, x1, node_w = 0.17, 0.82, 0.035

    for a in order:
        for b in order:
            n = matrix.loc[a, b]
            if n <= 0:
                continue
            h = usable * n / total
            y0a, y0b = left_cursor[a], left_cursor[a] + h
            y1a, y1b = right_cursor[b], right_cursor[b] + h
            left_cursor[a] += h
            right_cursor[b] += h
            verts = [
                (x0 + node_w, y0a),
                (0.43, y0a),
                (0.57, y1a),
                (x1, y1a),
                (x1, y1b),
                (0.57, y1b),
                (0.43, y0b),
                (x0 + node_w, y0b),
                (x0 + node_w, y0a),
            ]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=STATE_COLORS[a], edgecolor="none", alpha=0.40))

    for st in order:
        for x, mapping, totals in [(x0, left, left_totals), (x1, right, right_totals)]:
            y0, y1 = mapping[st]
            ax.add_patch(Rectangle((x, y0), node_w, y1 - y0, facecolor=STATE_COLORS[st], edgecolor="white", lw=0.6))
        y0, y1 = left[st]
        ax.text(x0 - 0.015, (y0 + y1) / 2, f"{st}  {matrix.loc[st].sum():,}", ha="right", va="center", fontsize=7)
        y0, y1 = right[st]
        ax.text(x1 + node_w + 0.015, (y0 + y1) / 2, f"{matrix[st].sum():,}  {st}", ha="left", va="center", fontsize=7)
    ax.text(x0, 0.985, "Baseline", ha="left", va="top", fontweight="bold", fontsize=7.5)
    ax.text(x1 + node_w, 0.985, "Final", ha="right", va="top", fontweight="bold", fontsize=7.5)
    ax.text(0.50, 0.015, f"n={total:,}; median follow-up 292 days", ha="center", fontsize=6.8, color=COLORS["gray"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def serum_associations(score: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {
        "SDS_v0": "Composite SDS_v0",
        "axis_ECM": "ECM retention axis",
        "axis_cholestatic": "Cholestatic-support axis",
        "axis_portal_synthetic": "Portal/synthetic deficit axis",
    }
    for col in labels:
        or_, lo, hi, p = odds_ratio_glm(score, "advanced_fibrosis_by_LSM", col)
        rows.append({"term": labels[col], "OR": or_, "ci_low": lo, "ci_high": hi, "p": p, "n": len(score)})
    return pd.DataFrame(rows)


def figure_main_2(longitudinal: pd.DataFrame, score: pd.DataFrame, serum_or: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16.2, 11.0))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.35, 3.3, 3.1], hspace=0.62, wspace=1.35)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    panel(ax, "A", "Operational bridge from tissue mechanism to measurable clinical states")
    cards = [
        ("Imaging retention", "LSM >=9.5 kPa while CAP <280 dB/m\n885/8,142 baseline patients", COLORS["purple"]),
        ("Serum imbalance", "ECM 50% + cholestatic support 25%\n+ inverse portal/synthetic reserve 25%", COLORS["red"]),
        ("Imaging persistence", "Advanced LSM persists and falls <20%\nafter >=90 days", COLORS["gold"]),
        ("Outcome boundary", "Imaging-defined progression/regression only\nno adjudicated readmission or treatment exposure", COLORS["gray"]),
    ]
    for i, (head, body, color) in enumerate(cards):
        x, w = 0.025 + i * 0.245, 0.22
        ax.add_patch(FancyBboxPatch((x, 0.18), w, 0.62, boxstyle="round,pad=0.01,rounding_size=0.008", transform=ax.transAxes, fc="white", ec=color, lw=1.6))
        ax.text(x + 0.015, 0.69, head, transform=ax.transAxes, color=color, fontsize=8.5, fontweight="bold")
        ax.text(x + 0.015, 0.43, body, transform=ax.transAxes, fontsize=7.1, va="center", linespacing=1.35)
        if i < len(cards) - 1:
            ax.annotate("", xy=(x + w + 0.021, 0.49), xytext=(x + w + 0.004, 0.49), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=1.1))

    ax = fig.add_subplot(gs[1:, :5])
    panel(ax, "B", "Four-state longitudinal flow in 772 repeat patients")
    flow = pd.crosstab(longitudinal.state0, longitudinal.state1)
    draw_alluvial(ax, flow)

    ax = fig.add_subplot(gs[1, 5:8])
    panel(ax, "C", "Imaging-defined longitudinal outcomes")
    order = ["Persistent nonadvanced", "Threshold regression", "Persistent advanced", "Threshold progression"]
    colors = [COLORS["teal"], COLORS["blue"], COLORS["red"], COLORS["orange"]]
    counts = longitudinal.outcome.value_counts().reindex(order)
    bars = ax.barh(np.arange(4)[::-1], counts.values, color=colors, height=0.65)
    ax.set_yticks(np.arange(4)[::-1], ["Nonadvanced", "Regressed", "Advanced persistent", "Progressed"])
    ax.tick_params(axis="y", labelsize=6.3, pad=2)
    ax.set_xlabel("Patients")
    for bar, n in zip(bars, counts.values):
        ax.text(n + 7, bar.get_y() + bar.get_height() / 2, f"{n:,} ({100*n/len(longitudinal):.1f}%)", va="center", fontsize=6.8)
    ax.set_xlim(0, max(counts) * 1.27)

    ax = fig.add_subplot(gs[1, 8:12])
    panel(ax, "D", "CAP and LSM can change nonreciprocally")
    cap_change = pd.cut(longitudinal.delta_CAP, [-np.inf, -20, 20, np.inf], labels=["CAP down >=20", "CAP stable", "CAP up >=20"])
    lsm_change = pd.cut(longitudinal.relative_LSM, [-np.inf, -0.20, 0.20, np.inf], labels=["LSM down >=20%", "LSM stable", "LSM up >=20%"])
    mat = pd.crosstab(cap_change, lsm_change).reindex(index=["CAP down >=20", "CAP stable", "CAP up >=20"], columns=["LSM down >=20%", "LSM stable", "LSM up >=20%"], fill_value=0)
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Patients"}, ax=ax, linewidths=1, linecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    subset = longitudinal[(longitudinal.CAP0 >= 0) & (longitudinal.LSM0 >= 9.5) & (longitudinal.delta_CAP <= -20)]
    persistent = int(subset.advanced1.sum())
    ax.text(0.0, -0.30, f"Among baseline advanced cases with CAP falling >=20 dB/m, {persistent}/{len(subset)} ({100*persistent/len(subset):.1f}%) remained advanced.", transform=ax.transAxes, fontsize=6.8, color=COLORS["gray"])

    ax = fig.add_subplot(gs[2, 5:8])
    panel(ax, "E", "Imaging persistence by baseline state")
    high = longitudinal.loc[longitudinal.advanced0 & longitudinal.state0.isin(["LH", "HH"])].copy()
    summary = high.groupby("state0").agg(n=("patient", "size"), persistent=("advanced1", "mean"), resistant=("regression_resistant", "mean")).reindex(["LH", "HH"])
    x = np.arange(2)
    ax.bar(x - 0.17, 100 * summary.persistent, width=0.34, color=COLORS["purple"], label="Persistent advanced")
    ax.bar(x + 0.17, 100 * summary.resistant, width=0.34, color=COLORS["gold"], label="Persistent + <20% LSM fall")
    ax.set_xticks(x, [f"{STATE_LABELS[s]}\nn={summary.loc[s, 'n']:,}" for s in summary.index])
    ax.set_ylabel("Patients (%)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, loc="upper right")
    for xi, st in enumerate(summary.index):
        ax.text(xi - 0.17, 100 * summary.loc[st, "persistent"] + 2, f"{100*summary.loc[st, 'persistent']:.1f}", ha="center", fontsize=6.6)
        ax.text(xi + 0.17, 100 * summary.loc[st, "resistant"] + 2, f"{100*summary.loc[st, 'resistant']:.1f}", ha="center", fontsize=6.6)
    ax.text(0, -0.31, "Persistence is imaging-defined, not adjudicated biological regression or treatment failure.", transform=ax.transAxes, fontsize=6.7, color=COLORS["gray"])

    ax = fig.add_subplot(gs[2, 8:12])
    panel(ax, "F", "Serum imbalance and advanced LSM")
    short_terms = {
        "Composite SDS_v0": "SDS_v0",
        "ECM retention axis": "ECM",
        "Cholestatic-support axis": "Cholestatic",
        "Portal/synthetic deficit axis": "Portal/synthetic",
    }
    y = np.arange(len(serum_or))[::-1]
    for yi, row in zip(y, serum_or.itertuples()):
        ax.errorbar(row.OR, yi, xerr=[[row.OR - row.ci_low], [row.ci_high - row.OR]], fmt="o", color=COLORS["red"] if row.term == "Composite SDS_v0" else COLORS["blue"], capsize=3, lw=1.2, ms=5.5)
        ax.text(row.ci_high + 0.15, yi, f"OR {row.OR:.2f} ({row.ci_low:.2f}-{row.ci_high:.2f}); P={row.p:.3f}", va="center", fontsize=6.5)
    ax.axvline(1, color=COLORS["gray"], ls="--", lw=1)
    ax.set_yticks(y, [short_terms[x] for x in serum_or.term])
    ax.tick_params(axis="y", labelsize=6.3, pad=2)
    ax.set_xscale("log")
    ax.set_xlim(0.22, 10.5)
    ax.set_xlabel("Odds ratio per 1-SD increase for LSM >=9.5 kPa")
    ax.text(0.0, -0.30, "n=46 complete cases; fixed outcome-independent weights; AUROC 0.688 (95% CI 0.497-0.862).", transform=ax.transAxes, fontsize=6.7, color=COLORS["gray"])

    fig.suptitle("Figure 2 | Clinical persistence phenotype and longitudinal outcomes", x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold")
    save_figure(fig, MAIN / "Figure_2")


def figure_ed1(index: pd.DataFrame, matched_index: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), gridspec_kw={"hspace": 0.52, "wspace": 0.38})
    ax = axes[0, 0]
    panel(ax, "A", "Cohort attrition and denominator changes")
    labels = ["Raw scans", "Technical QC", "Adult QC scans", "Adult index patients"]
    values = [9703, 9644, 9562, 8142]
    colors = [COLORS["blue"], COLORS["cyan"], COLORS["teal"], COLORS["gold"]]
    y = np.arange(4)[::-1]
    ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Records")
    ax.set_xlim(0, 10500)
    for yi, v in zip(y, values):
        ax.text(v + 120, yi, f"{v:,}", va="center", fontsize=7)

    ax = axes[0, 1]
    panel(ax, "B", "Variable-specific 90-day linkage completeness")
    variables = ["TG", "HDL", "UA", "FPG", "AST", "ALT", "GGT", "ALB", "PLT", "HA", "CIV", "LN", "PIIINP"]
    completeness = matched_index[variables].notna().mean().sort_values()
    ax.barh(np.arange(len(completeness)), 100 * completeness.values, color=[COLORS["teal"] if v > 0.1 else COLORS["gray"] for v in completeness.values], height=0.68)
    ax.set_yticks(np.arange(len(completeness)), completeness.index)
    ax.set_xlabel("Index patients with matched value (%)")
    ax.set_xlim(0, max(20, 100 * completeness.max() * 1.15))
    for yi, v in enumerate(completeness.values):
        ax.text(100 * v + 0.3, yi, f"{100*v:.1f}", va="center", fontsize=6.5)

    ax = axes[1, 0]
    panel(ax, "C", "Absolute linkage offsets by analyte")
    vars_off = ["TG", "HDL", "AST", "GGT", "PLT", "HA"]
    data = []
    for var in vars_off:
        vals = matched_index[f"{var}_off_days"].dropna().abs()
        data.extend([(var, x) for x in vals])
    off = pd.DataFrame(data, columns=["analyte", "days"])
    sns.boxplot(data=off, x="analyte", y="days", color=COLORS["cyan"], width=0.58, fliersize=1.2, ax=ax)
    ax.axhline(90, color=COLORS["red"], ls="--", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("Absolute offset (days)")
    ax.set_ylim(-2, 95)

    ax = axes[1, 1]
    panel(ax, "D", "Calendar drift in baseline stiffness prevalence")
    yearly = index.groupby(index.scan_date.dt.year).agg(n=("advanced", "size"), advanced=("advanced", "mean"), high_cap=("cap280", "mean"), retention=("retention", "mean")).reset_index(names="year")
    ax.plot(yearly.year, 100 * yearly.advanced, marker="o", color=COLORS["red"], label="LSM >=9.5")
    ax.plot(yearly.year, 100 * yearly.high_cap, marker="s", color=COLORS["gold"], label="CAP >=280")
    ax.plot(yearly.year, 100 * yearly.retention, marker="^", color=COLORS["purple"], label="Low-CAP/high-LSM")
    for row in yearly.itertuples():
        ax.text(row.year, 100 * row.high_cap + 2.2, f"n={row.n:,}", ha="center", fontsize=6.5)
    ax.set_xticks(yearly.year)
    ax.set_ylabel("Index patients (%)")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 58)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    fig.suptitle("Extended Data Figure 1 | Clinical data quality, linkage, and calendar structure", x=0.02, y=0.995, ha="left", fontsize=12.5, fontweight="bold")
    save_figure(fig, ED / "Extended_Data_Figure_1")


def forest_effect_panel(ax: plt.Axes, data: pd.DataFrame, title: str, endpoint: str, color: str) -> None:
    panel(ax, title[0], title[2:])
    d = data.loc[data.endpoint == endpoint].sort_values("beta_per_SD")
    y = np.arange(len(d))
    ax.errorbar(d.beta_per_SD, y, xerr=[d.beta_per_SD - d.ci_low, d.ci_high - d.beta_per_SD], fmt="o", color=color, ecolor=color, capsize=2.5, ms=4.5)
    ax.axvline(0, color=COLORS["gray"], ls="--", lw=0.9)
    n_col = "n" if "n" in d.columns else "n_keys"
    labels = [f"{a}  n={int(n)}" for a, n in zip(d.analyte, d[n_col])]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Adjusted effect per 1-SD analyte")
    for yi, row in enumerate(d.itertuples()):
        if row.q_value_BH < 0.05:
            ax.text(row.ci_high, yi + 0.18, f"q={row.q_value_BH:.2g}", fontsize=6, color=color, ha="right")


def figure_ed2() -> None:
    metabolic = pd.read_csv(CLIN / "metabolic_models.tsv", sep="\t")
    lis = pd.read_csv(CLIN / "lis_models.tsv", sep="\t")
    keep_met = metabolic.loc[(metabolic.q_value_BH < 0.10) | metabolic.analyte.isin(["TG", "HDL", "UA", "FPG"])].copy()
    keep_lis = lis.loc[(lis.q_value_BH < 0.10) | lis.analyte.isin(["AST", "ALT", "GGT", "HA", "LN", "CIV", "PIIINP", "ALB", "PLT"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.3), gridspec_kw={"hspace": 0.48, "wspace": 0.42})
    forest_effect_panel(axes[0, 0], keep_met, "A Metabolic markers -> CAP", "CAP_dB_per_m", COLORS["gold"])
    forest_effect_panel(axes[0, 1], keep_met, "B Metabolic markers -> log-LSM", "log_LSM", COLORS["orange"])
    forest_effect_panel(axes[1, 0], keep_lis, "C LIS markers -> CAP", "CAP_dB_per_m", COLORS["blue"])
    forest_effect_panel(axes[1, 1], keep_lis, "D LIS markers -> log-LSM", "log_LSM", COLORS["red"])
    fig.suptitle("Extended Data Figure 2 | Adjusted metabolic and laboratory associations with CAP and LSM", x=0.02, y=0.995, ha="left", fontsize=12.5, fontweight="bold")
    save_figure(fig, ED / "Extended_Data_Figure_2")


def figure_ed3(index_matched: pd.DataFrame) -> None:
    fast = pd.read_csv(CLIN / "fast_score.tsv", sep="\t")
    perf = pd.read_csv(CLIN / "fib4_ecm_performance.tsv", sep="\t")
    cal = pd.read_csv(CLIN / "fib4_ecm_calibration.tsv", sep="\t")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), gridspec_kw={"hspace": 0.52, "wspace": 0.42})

    ax = axes[0, 0]
    panel(ax, "A", "FAST distribution in CAP >=248 screening context")
    d = fast.loc[fast.CAP_steatotic_phenotype]
    ax.hist(d.FAST, bins=np.linspace(0, 1, 31), color=COLORS["teal"], edgecolor="white")
    ax.axvline(0.35, color=COLORS["gold"], ls="--", lw=1.2)
    ax.axvline(0.67, color=COLORS["red"], ls="--", lw=1.2)
    ax.set_xlabel("FAST score")
    ax.set_ylabel("Patients")
    ax.text(0.03, 0.94, "n=286", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[0, 1]
    panel(ax, "B", "FAST varies across CAP-LSM states (descriptive)")
    d = fast.copy()
    d["state"] = state_code(d.CAP, d.LSM)
    sns.boxplot(data=d, x="state", y="FAST", order=["LL", "HL", "LH", "HH"], palette=STATE_COLORS, width=0.62, fliersize=1.5, ax=ax)
    ax.set_xlabel("CAP-LSM state")
    ax.set_ylabel("FAST")

    ax = axes[1, 0]
    panel(ax, "C", "Observed advanced-LSM fraction across score quartiles")
    for score, color, marker in [("FIB4", COLORS["blue"], "o"), ("ECM4", COLORS["red"], "s")]:
        q = cal.loc[cal.score == score].reset_index(drop=True)
        x = np.arange(1, 5)
        observed = q["observed_fraction_LSM_ge9.5"]
        ax.errorbar(x, 100 * observed, yerr=[100 * (observed - q.ci_low), 100 * (q.ci_high - observed)], marker=marker, color=color, capsize=3, label=score)
    ax.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Score quartile")
    ax.set_ylabel("LSM >=9.5 kPa (%)")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    panel(ax, "D", "Imaging-reference discrimination remains imprecise")
    auc = perf.loc[perf.metric == "AUROC_vs_LSM_ge9.5"].copy().iloc[::-1]
    y = np.arange(len(auc))
    ax.errorbar(auc.estimate, y, xerr=[auc.estimate - auc.ci_low, auc.ci_high - auc.estimate], fmt="o", color=COLORS["blue"], capsize=3)
    ax.axvline(0.5, color=COLORS["gray"], ls="--", lw=1)
    ax.set_yticks(y, [f"{a}  n={int(n)}" for a, n in zip(auc.analysis, auc.n)])
    ax.set_xlim(0.42, 1.0)
    ax.set_xlabel("AUROC vs LSM >=9.5")
    ax.text(0, -0.24, "FAST is not independently validated here because LSM and CAP are components of FAST.", transform=ax.transAxes, fontsize=6.8, color=COLORS["gray"])

    fig.suptitle("Extended Data Figure 3 | FAST, FIB-4, and ECM4 clinical calibration", x=0.02, y=0.995, ha="left", fontsize=12.5, fontweight="bold")
    save_figure(fig, ED / "Extended_Data_Figure_3")


def figure_ed4() -> None:
    sens = pd.read_csv(DYN / "S5_clinical_history_gain/threshold_sensitivity.tsv", sep="\t")
    trans = pd.read_csv(DYN / "S3_clinical_multistate/transition_matrix.tsv", sep="\t")
    curves = pd.read_csv(DYN / "S3_clinical_multistate/branch_prediction_curves.tsv", sep="\t")
    branch = pd.read_csv(DYN / "S3_clinical_multistate/branch_model_summary.tsv", sep="\t")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), gridspec_kw={"hspace": 0.52, "wspace": 0.40})

    ax = axes[0, 0]
    panel(ax, "A", "History-gain threshold landscape")
    piv = sens.pivot(index="lsm_cut", columns="cap_cut", values="log_loss_gain").sort_index(ascending=False)
    sns.heatmap(piv, cmap="RdBu_r", center=0, annot=True, fmt=".3f", linewidths=1, linecolor="white", cbar_kws={"label": "Held-out log-loss gain"}, ax=ax)
    ax.set_xlabel("CAP threshold (dB/m)")
    ax.set_ylabel("LSM threshold (kPa)")
    cap_cols = list(piv.columns)
    lsm_rows = list(piv.index)
    if 280 in cap_cols and 9.5 in lsm_rows:
        ax.add_patch(Rectangle((cap_cols.index(280), lsm_rows.index(9.5)), 1, 1, fill=False, ec=COLORS["dark"], lw=2.2))

    ax = axes[0, 1]
    panel(ax, "B", "Prespecified and exploratory uncertainty")
    d = sens.loc[((sens.cap_cut == 280) & (sens.lsm_cut == 9.5)) | ((sens.lsm_cut == 12.0) & sens.log_loss_gain_ci_low.notna())].copy()
    d["label"] = np.where(d.lsm_cut == 9.5, "Prespecified CAP280 / LSM9.5", "Exploratory CAP" + d.cap_cut.astype(int).astype(str) + " / LSM12")
    d = d.sort_values(["lsm_cut", "cap_cut"])
    y = np.arange(len(d))[::-1]
    for yi, row in zip(y, d.itertuples()):
        color = COLORS["red"] if row.lsm_cut == 9.5 else COLORS["blue"]
        ax.errorbar(row.log_loss_gain, yi, xerr=[[row.log_loss_gain - row.log_loss_gain_ci_low], [row.log_loss_gain_ci_high - row.log_loss_gain]], fmt="o", color=color, capsize=3)
        ax.text(row.log_loss_gain_ci_high + 0.004, yi, f"Pperm={row.history_permutation_p:.3f}", va="center", fontsize=6.5)
    ax.axvline(0, color=COLORS["gray"], ls="--", lw=1)
    ax.set_yticks(y, d.label)
    ax.set_xlabel("History-model log-loss gain")
    ax.text(0, -0.23, "LSM12 results follow a 15-threshold search and remain hypothesis-generating.", transform=ax.transAxes, fontsize=6.8, color=COLORS["gray"])

    ax = axes[1, 0]
    panel(ax, "C", "Four-state transition probabilities at >=90 days")
    t = trans.loc[trans.min_interval_days == 90].pivot(index="from_state", columns="to_state", values="row_probability")
    order = ["LL", "HL", "LH", "HH"]
    t = t.reindex(index=order, columns=order)
    sns.heatmap(100 * t, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=1, linecolor="white", cbar_kws={"label": "Row probability (%)"}, ax=ax)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")

    ax = axes[1, 1]
    panel(ax, "D", "Primary and lag-adjusted CAP-LSM branches")
    for model, ls in [("branch_primary", "-"), ("branch_lag_adjusted", "--")]:
        for direction, color in [("up", COLORS["red"]), ("down", COLORS["blue"] )]:
            d = curves.loc[(curves.model == model) & (curves.direction == direction)].sort_values("CAP")
            ax.plot(d.CAP, d.pred_LSM, color=color, ls=ls, lw=1.5, label=f"{model.replace('branch_', '')}: CAP {direction}")
            ax.fill_between(d.CAP, d.ci_low, d.ci_high, color=color, alpha=0.08)
    ax.set_xlabel("CAP (dB/m)")
    ax.set_ylabel("Predicted LSM (kPa)")
    ax.legend(frameon=False, fontsize=6.4, ncol=2)
    txt = " | ".join([f"{r.model.replace('branch_', '')}: joint P={r.joint_direction_p:.3g}" for r in branch.itertuples()])
    ax.text(0, -0.23, txt, transform=ax.transAxes, fontsize=6.6, color=COLORS["gray"])

    fig.suptitle("Extended Data Figure 4 | Multistate dynamics, threshold sensitivity, and branch asymmetry", x=0.02, y=0.995, ha="left", fontsize=12.5, fontweight="bold")
    save_figure(fig, ED / "Extended_Data_Figure_4")


def figure_ed5(index: pd.DataFrame, longitudinal: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), gridspec_kw={"hspace": 0.52, "wspace": 0.40})
    ax = axes[0, 0]
    panel(ax, "A", "Follow-up interval distribution")
    ax.hist(longitudinal.followup_days, bins=35, color=COLORS["blue"], edgecolor="white")
    ax.axvline(longitudinal.followup_days.median(), color=COLORS["red"], ls="--", lw=1.2)
    ax.set_xlabel("First-to-last interval (days)")
    ax.set_ylabel("Patients")
    ax.text(0.98, 0.94, f"median {longitudinal.followup_days.median():.0f} days\nn={len(longitudinal):,}", transform=ax.transAxes, ha="right", va="top", fontsize=7)

    ax = axes[0, 1]
    panel(ax, "B", "Baseline versus final LSM")
    ax.scatter(longitudinal.LSM0, longitudinal.LSM1, s=8, alpha=0.30, color=COLORS["blue"], edgecolor="none")
    lim = [2.5, 33]
    ax.plot(lim, lim, color=COLORS["gray"], ls="--", lw=1)
    ax.axvline(9.5, color=COLORS["red"], ls=":", lw=1)
    ax.axhline(9.5, color=COLORS["red"], ls=":", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Baseline LSM (kPa)")
    ax.set_ylabel("Final LSM (kPa)")

    ax = axes[1, 0]
    panel(ax, "C", "Relative LSM change by baseline state")
    sns.violinplot(data=longitudinal, x="state0", y="relative_LSM", order=["LL", "HL", "LH", "HH"], palette=STATE_COLORS, inner="quartile", cut=0, linewidth=0.8, ax=ax)
    ax.axhline(0, color=COLORS["gray"], lw=0.8)
    ax.axhline(-0.2, color=COLORS["blue"], ls="--", lw=0.9)
    ax.axhline(0.2, color=COLORS["red"], ls="--", lw=0.9)
    ax.set_ylim(-1.0, 2.5)
    ax.set_xlabel("Baseline CAP-LSM state")
    ax.set_ylabel("Relative LSM change")

    ax = axes[1, 1]
    panel(ax, "D", "Longitudinal outcomes by baseline era")
    tab = pd.crosstab(longitudinal.baseline_era, longitudinal.outcome).reindex(columns=["Persistent nonadvanced", "Threshold regression", "Persistent advanced", "Threshold progression"], fill_value=0)
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(pct))
    colors = [COLORS["teal"], COLORS["blue"], COLORS["red"], COLORS["orange"]]
    for col, color in zip(pct.columns, colors):
        ax.bar(pct.index, pct[col], bottom=bottom, color=color, label=col, width=0.58, edgecolor="white")
        bottom += pct[col].to_numpy()
    for i, era in enumerate(pct.index):
        ax.text(i, 102, f"n={tab.loc[era].sum():,}", ha="center", fontsize=6.8)
    ax.set_ylim(0, 109)
    ax.set_ylabel("Patients (%)")
    ax.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)

    fig.suptitle("Extended Data Figure 5 | Longitudinal distributions and temporal robustness", x=0.02, y=0.995, ha="left", fontsize=12.5, fontweight="bold")
    save_figure(fig, ED / "Extended_Data_Figure_5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-input-dir", type=Path, default=CLIN)
    parser.add_argument("--dynamics-input-dir", type=Path, default=DYN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    global CLIN, DYN, OUTPUT_ROOT, TABLES, MAIN, ED, LOGS
    args = parse_args()
    CLIN, DYN, OUTPUT_ROOT = args.clinical_input_dir, args.dynamics_input_dir, args.output_dir
    TABLES = OUTPUT_ROOT / "source_tables"
    MAIN = OUTPUT_ROOT / "main_figures"
    ED = OUTPUT_ROOT / "extended_data_figures"
    LOGS = OUTPUT_ROOT / "logs"
    for folder in [TABLES, MAIN, ED, LOGS]:
        folder.mkdir(parents=True, exist_ok=True)
    elasto = pd.read_parquet(CLIN / "elasto_clean_full.parquet")
    elasto = elasto.loc[elasto.adult_at_scan].copy().sort_values(["medical_record_hash", "scan_timestamp"])
    index = elasto.groupby("medical_record_hash", as_index=False).first()
    index["male"] = index.sex.eq("M")
    index["advanced"] = index.advanced_fibrosis_lsm_ge9_5.astype(bool)
    index["cap248"] = index.CAP.ge(248)
    index["cap280"] = index.CAP.ge(280)
    index["f3"] = index.fibrosis_stage.eq("F3")
    index["f4"] = index.fibrosis_stage.eq("F4")
    index["gi"] = index.department.fillna("").str.contains("消化|肝胆")
    index["state"] = state_code(index.CAP, index.LSM)
    index["retention"] = index.state.eq("LH")

    matched = pd.read_parquet(CLIN / "matched_90d.parquet")
    matched_index = index[["scan_hash"]].merge(matched, on="scan_hash", how="left", suffixes=("", "_matched"))
    etiology = pd.read_csv(CLIN / "etiology_summary.tsv", sep="\t")
    score = pd.read_csv(CLIN / "serum_decodable_state_score.tsv", sep="\t")

    baseline = baseline_table(index)
    validation = validation_analysis(index)
    longitudinal = build_longitudinal(elasto)
    serum_or = serum_associations(score)

    baseline.to_csv(TABLES / "Table1_baseline_8142.tsv", sep="\t", index=False)
    validation.to_csv(TABLES / "clinical_validation_metrics.tsv", sep="\t", index=False)
    pd.crosstab(longitudinal.state0, longitudinal.state1).to_csv(TABLES / "longitudinal_state_flow.tsv", sep="\t")
    longitudinal.outcome.value_counts().rename_axis("outcome").reset_index(name="n").to_csv(TABLES / "longitudinal_outcomes.tsv", sep="\t", index=False)
    serum_or.to_csv(TABLES / "serum_imbalance_associations.tsv", sep="\t", index=False)

    summary = {
        "cohort": {
            "raw_scans": 9703,
            "technical_qc_scans": 9644,
            "adult_qc_scans": int(len(elasto)),
            "adult_index_patients": int(len(index)),
            "advanced_lsm_n": int(index.advanced.sum()),
            "advanced_lsm_fraction": float(index.advanced.mean()),
            "retention_phenotype_n": int(index.retention.sum()),
        },
        "longitudinal": {
            "n": int(len(longitudinal)),
            "median_followup_days": float(longitudinal.followup_days.median()),
            "outcomes": {k: int(v) for k, v in longitudinal.outcome.value_counts().items()},
            "regression_resistant_n": int(longitudinal.regression_resistant.sum()),
        },
        "ethics": "approval number, approval date, committee, and consent-waiver status pending author input",
        "clinical_event_boundary": "readmission, decompensation, mortality, transplant, treatment exposure, and prescription outcomes not present in current source files",
    }
    (TABLES / "clinical_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_main_1(index, baseline, etiology, validation)
    figure_main_2(longitudinal, score, serum_or)
    figure_ed1(index, matched_index)
    figure_ed2()
    figure_ed3(matched_index)
    figure_ed4()
    figure_ed5(index, longitudinal)

    log = {
        "script": str(SCRIPT),
        "inputs": {"clinical": str(CLIN), "dynamics": str(DYN)},
        "main_figures": sorted(str(p) for p in MAIN.glob("*.png")),
        "extended_figures": sorted(str(p) for p in ED.glob("*.png")),
        "tables": sorted(str(p) for p in TABLES.glob("*")),
    }
    (LOGS / "rebuild_clinical_visuals_manifest.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
