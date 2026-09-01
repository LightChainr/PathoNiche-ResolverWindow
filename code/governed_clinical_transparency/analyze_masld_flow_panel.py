#!/usr/bin/env python3
"""Deidentify and analyze real-world lymphocyte panels from fatty-liver-coded encounters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data/governed/flow_panel"
RESULTS = REPO_ROOT / "results/flow_panel"
FIGURES = RESULTS / "figures"
CLINICAL = REPO_ROOT / "data/governed/clinical/matched_90d.parquet"
SALT_PATH = REPO_ROOT / "data/governed/clinical_hash_salt.txt"

MARKERS = ["CD16+CD56+", "CD3+", "CD3+CD4+", "CD3+CD8+", "CD3-CD19+", "CD4/CD8"]
DISPLAY = {
    "CD16+CD56+": "NK",
    "CD3+": "T",
    "CD3+CD4+": "CD4 T",
    "CD3+CD8+": "CD8 T",
    "CD3-CD19+": "B",
    "CD4/CD8": "CD4/CD8",
}
CONTEXT_ORDER = ["metabolic-dominant", "acute-inflammatory", "oncology/hematology"]
CONTEXT_COLORS = {
    "metabolic-dominant": "#2A9D8F",
    "acute-inflammatory": "#E2A33A",
    "oncology/hematology": "#B4495B",
}


def digest(salt: str, namespace: str, value: object) -> str:
    payload = f"{salt}|{namespace}|{str(value).strip()}".encode()
    return hashlib.sha256(payload).hexdigest()


def bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.clip(q, 0, 1)
    return result


def load_panel() -> pd.DataFrame:
    salt = SALT_PATH.read_text().strip()
    frames = []
    for path in sorted(RAW.glob("*.xlsx")):
        local = pd.read_excel(path)
        local["source_file"] = path.name
        frames.append(local)
    long = pd.concat(frames, ignore_index=True)
    long["sex"] = long["性别"].map({"男": "M", "女": "F"})
    long["birth_date"] = pd.to_datetime(long["出生日期"], errors="coerce")
    long["flow_timestamp"] = pd.to_datetime(long["审核时间"], errors="coerce")
    key_plain = long.birth_date.dt.strftime("%Y-%m-%d") + "|" + long.sex
    long["dob_sex_key_hash"] = key_plain.map(lambda value: digest(salt, "dob_sex", value))
    long["flow_subject_hash"] = long["身份证号"].astype(str).map(lambda value: digest(salt, "flow_subject", value))
    long["flow_event_hash"] = [
        digest(salt, "flow_event", f"{subject}|{timestamp}|{barcode}")
        for subject, timestamp, barcode in zip(long.flow_subject_hash, long.flow_timestamp, long["条码号"])
    ]
    index = [
        "flow_event_hash", "flow_subject_hash", "dob_sex_key_hash", "flow_timestamp",
        "birth_date", "sex", "年龄", "诊断", "source_file",
    ]
    wide = long.pivot_table(index=index, columns="项目代码", values="测定结果", aggfunc="first").reset_index()
    wide.columns.name = None
    missing = [marker for marker in MARKERS if marker not in wide]
    if missing:
        raise ValueError(f"Missing flow markers: {missing}")
    wide["age"] = pd.to_numeric(wide["年龄"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    diagnosis = wide["诊断"].fillna("")
    oncology = diagnosis.str.contains(r"淋巴瘤|白血病|骨髓瘤|骨髓增生|恶性肿瘤|化学治疗|靶向治疗", regex=True)
    infection = diagnosis.str.contains(r"感染|肺炎|发热|败血|脓毒|急性下呼吸道感染", regex=True)
    wide["context"] = np.where(
        oncology,
        "oncology/hematology",
        np.where(infection, "acute-inflammatory", "metabolic-dominant"),
    )
    wide["diabetes"] = diagnosis.str.contains("糖尿病")
    wide["hypertension"] = diagnosis.str.contains("高血压")
    wide["viral_hepatitis"] = diagnosis.str.contains(r"乙型病毒性肝炎|丙型病毒性肝炎|病毒性肝炎", regex=True)
    wide["liver_dysfunction"] = diagnosis.str.contains(r"肝功能不全|肝硬化|肝衰竭", regex=True)
    total = wide["CD16+CD56+"] + wide["CD3+"] + wide["CD3-CD19+"]
    wide["NK_fraction"] = wide["CD16+CD56+"] / total
    wide["T_fraction"] = wide["CD3+"] / total
    wide["B_fraction"] = wide["CD3-CD19+"] / total
    wide["CD4_T_fraction"] = wide["CD3+CD4+"] / wide["CD3+"]
    wide["CD8_T_fraction"] = wide["CD3+CD8+"] / wide["CD3+"]
    wide["computed_CD4_CD8"] = wide["CD3+CD4+"] / wide["CD3+CD8+"]
    wide["ratio_relative_error"] = (wide["computed_CD4_CD8"] - wide["CD4/CD8"]).abs() / wide["CD4/CD8"].replace(0, np.nan)
    return wide


def match_clinical(panel: pd.DataFrame) -> pd.DataFrame:
    clinical = pd.read_parquet(CLINICAL)
    rows = []
    for record in panel.itertuples(index=False):
        candidates = clinical.loc[clinical.dob_sex_key_hash.eq(record.dob_sex_key_hash)].copy()
        if candidates.empty:
            continue
        candidates["flow_scan_days"] = (
            candidates.scan_timestamp - record.flow_timestamp
        ).abs().dt.total_seconds() / 86400
        best = candidates.sort_values(["flow_scan_days", "scan_timestamp"]).iloc[0]
        if best.flow_scan_days <= 90:
            rows.append(
                {
                    "flow_event_hash": record.flow_event_hash,
                    "flow_subject_hash": record.flow_subject_hash,
                    "scan_hash": best.scan_hash,
                    "flow_scan_days": float(best.flow_scan_days),
                    "LSM": best.LSM,
                    "CAP": best.CAP,
                    "fibrosis_stage": best.fibrosis_stage,
                    "qc_pass": best.qc_pass,
                    "BMI": best.BMI,
                }
            )
    return pd.DataFrame(rows)


def context_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    context = pd.Categorical(panel.context, CONTEXT_ORDER)
    dummies = pd.get_dummies(context, drop_first=True, dtype=float)
    sex = panel.sex.eq("M").astype(float).to_numpy()
    age = (panel.age - panel.age.mean()) / panel.age.std(ddof=1)
    x = np.column_stack([np.ones(len(panel)), dummies.to_numpy(float), age.to_numpy(float), sex])
    terms = ["intercept", "acute-inflammatory", "oncology/hematology", "age_z", "male"]
    groups = pd.Categorical(panel.flow_subject_hash).codes
    for marker in MARKERS:
        y = np.log1p(panel[marker].to_numpy(float))
        y = (y - y.mean()) / y.std(ddof=1)
        fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
        for index, term in enumerate(terms[1:3], start=1):
            rows.append(
                {
                    "marker": marker,
                    "comparison": f"{term} vs metabolic-dominant",
                    "beta_sd": float(fit.params[index]),
                    "se": float(fit.bse[index]),
                    "ci_low": float(fit.params[index] - 1.96 * fit.bse[index]),
                    "ci_high": float(fit.params[index] + 1.96 * fit.bse[index]),
                    "p_value": float(fit.pvalues[index]),
                }
            )
    result = pd.DataFrame(rows)
    result["fdr"] = bh(result.p_value)
    return result


def immune_pca(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = MARKERS
    matrix = np.log1p(panel[features].to_numpy(float))
    matrix = StandardScaler().fit_transform(matrix)
    model = PCA(n_components=2, random_state=20260712)
    coordinates = model.fit_transform(matrix)
    scores = panel[["flow_event_hash", "flow_subject_hash", "flow_timestamp", "context", "age", "sex"]].copy()
    scores["PC1"] = coordinates[:, 0]
    scores["PC2"] = coordinates[:, 1]
    loadings = pd.DataFrame(
        {
            "marker": features,
            "PC1_loading": model.components_[0],
            "PC2_loading": model.components_[1],
            "PC1_variance": model.explained_variance_ratio_[0],
            "PC2_variance": model.explained_variance_ratio_[1],
        }
    )
    return scores, loadings


def plot_figure(panel: pd.DataFrame, models: pd.DataFrame, pca_scores: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(15.5, 11.5))
    grid = fig.add_gridspec(2, 3, left=.06, right=.98, bottom=.07, top=.91, wspace=.35, hspace=.36)

    ax = fig.add_subplot(grid[0, 0])
    order = panel.groupby("flow_subject_hash").flow_timestamp.min().sort_values().index
    rank = {subject: i for i, subject in enumerate(order)}
    for subject, local in panel.groupby("flow_subject_hash", observed=True):
        local = local.sort_values("flow_timestamp")
        y = rank[subject]
        ax.plot(local.flow_timestamp, np.repeat(y, len(local)), color="#C8CED2", lw=.7)
        ax.scatter(local.flow_timestamp, np.repeat(y, len(local)), c=local.context.map(CONTEXT_COLORS), s=17, edgecolor="white", linewidth=.3)
    ax.set(xlabel="Flow-cytometry date", ylabel="Deidentified patient", yticks=[])
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title("A  Real-world sampling architecture", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    x = panel.T_fraction + .5 * panel.B_fraction
    y = np.sqrt(3) / 2 * panel.B_fraction
    triangle = np.array([[0,0],[1,0],[.5,np.sqrt(3)/2],[0,0]])
    ax.plot(triangle[:,0],triangle[:,1],color="#4D5559",lw=1)
    ax.scatter(x,y,c=panel.context.map(CONTEXT_COLORS),s=32,alpha=.8,edgecolor="white",linewidth=.4)
    ax.text(-.03,-.04,"NK",ha="right"); ax.text(1.03,-.04,"T",ha="left"); ax.text(.5,.91,"B",ha="center")
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("B  Lymphocyte composition simplex", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[0, 2])
    for context in CONTEXT_ORDER:
        local = pca_scores.loc[pca_scores.context.eq(context)]
        ax.scatter(local.PC1,local.PC2,s=34,color=CONTEXT_COLORS[context],label=context,alpha=.82,edgecolor="white",linewidth=.4)
    ax.axhline(0,color="#D5D9DB",lw=.7); ax.axvline(0,color="#D5D9DB",lw=.7)
    ax.set(xlabel="Immune PC1",ylabel="Immune PC2")
    ax.set_title("C  Peripheral immune-state diversity", loc="left", fontweight="bold")
    ax.legend(frameon=False,fontsize=7,loc="best")

    ax = fig.add_subplot(grid[1, 0])
    plot = models.loc[models.comparison.str.startswith("acute")].copy()
    plot["label"] = plot.marker.map(DISPLAY)
    plot = plot.sort_values("beta_sd")
    y = np.arange(len(plot))
    ax.errorbar(plot.beta_sd,y,xerr=[plot.beta_sd-plot.ci_low,plot.ci_high-plot.beta_sd],fmt="o",color=CONTEXT_COLORS["acute-inflammatory"],ecolor=CONTEXT_COLORS["acute-inflammatory"],capsize=3)
    ax.axvline(0,color="#666",lw=.8)
    ax.set_yticks(y,plot.label); ax.set_xlabel("Adjusted difference (SD)")
    ax.set_title("D  Acute-inflammatory immune displacement", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[1, 1])
    correlation = panel[MARKERS].rename(columns=DISPLAY).corr(method="spearman")
    correlation.index.name = None
    correlation.columns.name = None
    sns.heatmap(correlation,cmap="RdBu_r",center=0,vmin=-1,vmax=1,square=True,annot=True,fmt=".2f",annot_kws={"fontsize":7},cbar_kws={"shrink":.65},ax=ax)
    ax.set_title("E  Coordinated lymphocyte architecture", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[1, 2])
    repeated = panel.groupby("flow_subject_hash").filter(lambda x: len(x)>1).copy()
    repeated["days"] = repeated.groupby("flow_subject_hash").flow_timestamp.transform(lambda x:(x-x.min()).dt.total_seconds()/86400)
    for subject, local in repeated.groupby("flow_subject_hash",observed=True):
        local=local.sort_values("days")
        ax.plot(local.days,local["CD4/CD8"],color="#87949A",lw=1,alpha=.75)
        ax.scatter(local.days,local["CD4/CD8"],c=local.context.map(CONTEXT_COLORS),s=24,edgecolor="white",linewidth=.3)
    ax.axhline(1,color="#C43D55",ls="--",lw=1)
    ax.set(xlabel="Days from first panel",ylabel="CD4/CD8 ratio")
    ax.set_title("F  Within-patient immune-state mobility", loc="left", fontweight="bold")

    for ax in fig.axes:
        if ax.axison:
            ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Peripheral lymphocyte-state diversity in fatty-liver-coded clinical encounters",fontsize=16,fontweight="bold")
    fig.savefig(FIGURES / "MASLD_peripheral_lymphocyte_landscape.png",dpi=320,facecolor="white")
    fig.savefig(FIGURES / "MASLD_peripheral_lymphocyte_landscape.pdf",facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-input-dir",type=Path,default=RAW)
    parser.add_argument("--clinical-linkage",type=Path,default=CLINICAL)
    parser.add_argument("--hash-salt",type=Path,default=SALT_PATH)
    parser.add_argument("--output-dir",type=Path,default=RESULTS)
    return parser.parse_args()


def main() -> None:
    global RAW,CLINICAL,SALT_PATH,RESULTS,FIGURES
    args=parse_args()
    RAW,CLINICAL,SALT_PATH,RESULTS=args.flow_input_dir,args.clinical_linkage,args.hash_salt,args.output_dir
    FIGURES=RESULTS/"figures"
    RESULTS.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    panel=load_panel()
    clinical=match_clinical(panel)
    models=context_models(panel)
    pca_scores,pca_loadings=immune_pca(panel)
    plot_figure(panel,models,pca_scores)

    safe_columns=[
        "flow_event_hash","flow_subject_hash","dob_sex_key_hash","flow_timestamp","sex","age","context",
        "diabetes","hypertension","viral_hepatitis","liver_dysfunction",*MARKERS,
        "NK_fraction","T_fraction","B_fraction","CD4_T_fraction","CD8_T_fraction","ratio_relative_error",
    ]
    panel[safe_columns].to_parquet(RESULTS/"flow_panel_deidentified.parquet",index=False)
    panel[safe_columns].to_csv(RESULTS/"flow_panel_deidentified.tsv",sep="\t",index=False)
    models.to_csv(RESULTS/"flow_context_models.tsv",sep="\t",index=False)
    pca_scores.to_csv(RESULTS/"flow_immune_pca_scores.tsv",sep="\t",index=False)
    pca_loadings.to_csv(RESULTS/"flow_immune_pca_loadings.tsv",sep="\t",index=False)
    clinical.to_csv(RESULTS/"flow_liver_stiffness_90d_linkage.tsv",sep="\t",index=False)

    context_counts=panel.groupby("context",observed=True).agg(panels=("flow_event_hash","size"),patients=("flow_subject_hash","nunique")).reset_index()
    context_counts.to_csv(RESULTS/"flow_context_counts.tsv",sep="\t",index=False)
    summary={
        "n_rows_original":int(len(panel)*len(MARKERS)),
        "n_complete_panels":int(len(panel)),
        "n_patients":int(panel.flow_subject_hash.nunique()),
        "n_repeated_patients":int((panel.groupby("flow_subject_hash").size()>1).sum()),
        "max_panels_per_patient":int(panel.groupby("flow_subject_hash").size().max()),
        "context_counts":context_counts.to_dict(orient="records"),
        "median_ratio_relative_error":float(panel.ratio_relative_error.median()),
        "n_liver_stiffness_links_90d":int(len(clinical)),
        "role_in_manuscript":"Independent real-world peripheral immune-state atlas and longitudinal supplement; not a fibrosis-stage association cohort.",
    }
    (RESULTS/"flow_analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
