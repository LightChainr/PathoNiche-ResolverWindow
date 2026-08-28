#!/usr/bin/env python3
"""Collapse LINCS entities by compound name and apply robustness/safety gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HARD_EXCLUDE = {
    "alvocidib", "aminopurvalanol-a", "cyclosporin-a", "dacinostat", "daunorubicin",
    "digoxin", "doxorubicin", "geldanamycin", "menadione", "mitoxantrone",
    "nsc-632839", "ouabain", "panobinostat", "prochlorperazine", "proscillaridin", "terfenadine",
    "thiostrepton", "triclosan",
}
MECHANISTIC_PROBE = {
    "verteporfin": "YAP/TEAD-mechanotransduction probe; independent HSC fibrosis evidence",
    "WZ-4002": "EGFR-family kinase probe; tests growth-factor input into the retention attractor",
    "tyrphostin-1": "broad receptor-tyrosine-kinase probe",
    "U-0126": "MEK-ERK/AP-1 pathway probe",
    "GDC-0941": "PI3K pathway probe",
    "NVP-TAE684": "oncology kinase probe; experimental use only",
    "NSC-632839": "deubiquitinase-pathway probe; toxicity gate required",
    "PRE-084": "sigma-1 receptor probe; liver mechanism unresolved",
    "forskolin": "cAMP/PKA probe; conflicts with reported PKA-RUNX2 activation",
}
TRANSLATION_ANCHOR = {
    "pioglitazone": "PPAR-gamma re-quiescence/metabolic anchor",
    "oxfendazole": "named repurposing candidate; mechanism and exposure review required",
    "dehydrocholate": "bile-acid-context candidate; formulation and mechanism review required",
    "ezetimibe": "metabolic translation anchor; dual-axis consistency review required",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else x.std()
    return (x - med) / (scale if scale and np.isfinite(scale) else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v16", type=Path, default=REPO_ROOT / "work/v16_runtime")
    ap.add_argument("--legacy", type=Path, default=REPO_ROOT / "data/legacy_runtime")
    args = ap.parse_args()
    out = args.v16 / "02_results/S5_regulatory_drug_control"
    sig = pd.read_csv(out / "LINCS_dual_axis_signature_scores.tsv.gz", sep="\t")
    sig["pert_iname_norm"] = sig["pert_iname"].astype(str).str.strip().str.lower()

    per_id = sig.groupby(["pert_iname_norm", "pert_id"]).agg(
        id_retention=("retention_suppression", "median"),
        id_resolver_floor=("resolver_floor", "median"),
    ).reset_index()
    conflicts = per_id.groupby("pert_iname_norm").agg(
        n_pert_ids=("pert_id", "nunique"), min_id_retention=("id_retention", "min"),
        max_id_retention=("id_retention", "max"), min_id_resolver_floor=("id_resolver_floor", "min"),
        max_id_resolver_floor=("id_resolver_floor", "max"),
    ).reset_index()
    conflicts["retention_direction_conflict_across_ids"] = (conflicts["min_id_retention"] < 0) & (conflicts["max_id_retention"] > 0)
    conflicts["resolver_direction_conflict_across_ids"] = (conflicts["min_id_resolver_floor"] < 0) & (conflicts["max_id_resolver_floor"] > 0)

    d = sig.groupby("pert_iname_norm").agg(
        pert_iname=("pert_iname", "first"), pert_ids=("pert_id", lambda x: ";".join(sorted(set(map(str, x))))),
        n_signatures=("sig_id", "nunique"), n_cell_lines=("cell_id", "nunique"),
        cell_lines=("cell_id", lambda x: ";".join(sorted(set(map(str, x))))),
        median_retention_suppression=("retention_suppression", "median"),
        q25_retention_suppression=("retention_suppression", lambda x: x.quantile(.25)),
        median_endo_support=("endothelial_resolution_support", "median"),
        median_macro_support=("macrophage_resolution_support", "median"),
        median_resolver_floor=("resolver_floor", "median"),
        median_dual_control=("dual_control_score", "median"),
        q25_dual_control=("dual_control_score", lambda x: x.quantile(.25)),
        fraction_dual_positive=("dual_positive", "mean"),
    ).reset_index().merge(conflicts, on="pert_iname_norm", how="left")
    d = d[(d["n_signatures"] >= 3) & (d["n_cell_lines"] >= 2)].copy()
    d["robust_dual"] = d["median_dual_control"] + .5 * d["q25_dual_control"]
    d["concurrent_score"] = robust_z(d["robust_dual"]) + robust_z(d["fraction_dual_positive"])
    d["early_score"] = robust_z(d[["median_endo_support", "median_macro_support"]].mean(axis=1)) + .5 * robust_z(d["median_resolver_floor"])
    d["late_score"] = robust_z(d["median_retention_suppression"]) + .5 * robust_z(d["q25_retention_suppression"])
    d["concurrent_rank"] = d["concurrent_score"].rank(ascending=False, method="min").astype(int)

    sensitivity = []
    for wr, we, wm in [(1, .5, .5), (1.5, .25, .25), (.75, .75, .75), (1, 1, 0), (1, 0, 1)]:
        score = wr * d["median_retention_suppression"] + we * d["median_endo_support"] + wm * d["median_macro_support"]
        rank = score.rank(ascending=False, method="min")
        for idx in d.index:
            sensitivity.append((idx, int(rank.loc[idx])))
    sens = pd.DataFrame(sensitivity, columns=["idx", "rank"]).groupby("idx").agg(
        median_sensitivity_rank=("rank", "median"), worst_sensitivity_rank=("rank", "max"),
        n_top50_schemes=("rank", lambda x: int((x <= 50).sum())),
    )
    d = d.join(sens, how="left")

    pert = pd.read_csv(args.legacy / "00_raw/cmap_lincs/GSE92742/GSE92742_pert_info.txt.gz", sep="\t", dtype=str)
    pert["pert_iname_norm"] = pert["pert_iname"].astype(str).str.strip().str.lower()
    identity = pert[pert["pert_type"].eq("trt_cp")].groupby("pert_iname_norm").agg(
        is_touchstone=("is_touchstone", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).max())),
        has_structure=("canonical_smiles", lambda x: bool((x.astype(str) != "-666").any())),
        pubchem_cids=("pubchem_cid", lambda x: ";".join(sorted(set(v for v in map(str, x) if v != "-666")))),
    ).reset_index()
    d = d.merge(identity, on="pert_iname_norm", how="left")
    d["named_identity"] = ~d["pert_iname"].astype(str).str.upper().str.startswith("BRD-")
    d["identity_resolved"] = d["named_identity"] & (d["has_structure"].fillna(False) | d["is_touchstone"].eq(1))
    d["duplicate_id_conflict"] = d["retention_direction_conflict_across_ids"] | d["resolver_direction_conflict_across_ids"]
    d["safety_gate"] = "manual_review_required"
    d.loc[d["pert_iname_norm"].isin(HARD_EXCLUDE), "safety_gate"] = "exclude_narrow_margin_or_cytotoxic"
    d.loc[d["pert_iname"].isin(MECHANISTIC_PROBE), "safety_gate"] = "mechanistic_probe_only"
    d.loc[d["pert_iname_norm"].isin(TRANSLATION_ANCHOR), "safety_gate"] = "translation_review"
    d["mechanism_context"] = d["pert_iname"].map(MECHANISTIC_PROBE).fillna(d["pert_iname_norm"].map(TRANSLATION_ANCHOR)).fillna("unresolved_or_not_curated")

    d["computational_hard_gate"] = (
        d["identity_resolved"] & ~d["duplicate_id_conflict"]
        & (d["median_retention_suppression"] > 0) & (d["median_resolver_floor"] > 0)
        & (d["fraction_dual_positive"] >= .5) & (d["n_top50_schemes"] >= 2)
    )
    d["translation_tier"] = "Tier_4_not_prioritized"
    d.loc[d["computational_hard_gate"], "translation_tier"] = "Tier_3_computational_hit"
    d.loc[d["computational_hard_gate"] & d["safety_gate"].eq("mechanistic_probe_only"), "translation_tier"] = "Tier_1_mechanistic_probe"
    d.loc[(d["concurrent_rank"] <= 40) & d["pert_iname_norm"].isin(TRANSLATION_ANCHOR) & ~d["duplicate_id_conflict"], "translation_tier"] = "Tier_2_translation_anchor"
    d.loc[d["safety_gate"].eq("exclude_narrow_margin_or_cytotoxic"), "translation_tier"] = "Excluded_safety"
    d = d.sort_values(["translation_tier", "concurrent_rank"])
    d.to_csv(out / "LINCS_name_collapsed_hard_gated.tsv", sep="\t", index=False)

    eligible = d[
        d["computational_hard_gate"]
        & ~d["safety_gate"].eq("exclude_narrow_margin_or_cytotoxic")
        & ~d["pert_iname_norm"].isin({"estradiol"})
    ].copy()
    early = eligible.nlargest(25, "early_score")
    late = eligible.nlargest(25, "late_score")
    pairs = []
    for _, e in early.iterrows():
        for _, l in late.iterrows():
            if e["pert_iname_norm"] == l["pert_iname_norm"]:
                continue
            pr = l["median_retention_suppression"] + .5 * e["median_retention_suppression"]
            pe = e["median_endo_support"] + .5 * l["median_endo_support"]
            pm = e["median_macro_support"] + .5 * l["median_macro_support"]
            score = pr + .5 * pe + .5 * pm - 1.5 * max(0, -min(pe, pm))
            mechanism_resolved = e["pert_iname"] in MECHANISTIC_PROBE and l["pert_iname"] in MECHANISTIC_PROBE
            pairs.append({"early_resolver_compound": e["pert_iname"], "late_retention_compound": l["pert_iname"], "phase_selective_pair_score": score, "early_resolver_floor": e["median_resolver_floor"], "late_retention_suppression": l["median_retention_suppression"], "both_mechanisms_curated": mechanism_resolved, "interpretation": "ordered additive hypothesis; not synergy"})
    pairs = pd.DataFrame(pairs).sort_values(["both_mechanisms_curated", "phase_selective_pair_score"], ascending=[False, False])
    rng = np.random.default_rng(20260711)
    null = []
    vals = eligible.reset_index(drop=True)
    for _ in range(20000):
        i, j = rng.choice(len(vals), 2, replace=False)
        e, l = vals.iloc[i], vals.iloc[j]
        pr = l["median_retention_suppression"] + .5 * e["median_retention_suppression"]
        pe = e["median_endo_support"] + .5 * l["median_endo_support"]
        pm = e["median_macro_support"] + .5 * l["median_macro_support"]
        null.append(pr + .5 * pe + .5 * pm - 1.5 * max(0, -min(pe, pm)))
    null = np.asarray(null)
    pairs["random_eligible_pair_percentile"] = [(null <= x).mean() for x in pairs["phase_selective_pair_score"]]
    pairs.to_csv(out / "LINCS_phase_pairs_hard_gated.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    plot = d.copy()
    resolver = plot[["median_endo_support", "median_macro_support"]].mean(axis=1)
    colors = plot["translation_tier"].map({"Tier_1_mechanistic_probe": "#D1495B", "Tier_2_translation_anchor": "#00798C", "Tier_3_computational_hit": "#EDAE49", "Excluded_safety": "#777777"}).fillna("#D5D5D5")
    axes[0].scatter(plot["median_retention_suppression"], resolver, c=colors, s=18, alpha=.65)
    axes[0].axhline(0, color="#777", lw=.8); axes[0].axvline(0, color="#777", lw=.8)
    label = plot[plot["translation_tier"].isin(["Tier_1_mechanistic_probe", "Tier_2_translation_anchor"])].nsmallest(12, "concurrent_rank")
    for _, r in label.iterrows():
        axes[0].text(r["median_retention_suppression"], np.mean([r["median_endo_support"], r["median_macro_support"]]), r["pert_iname"], fontsize=8)
    axes[0].set(xlabel="HSC retention suppression", ylabel="Mean resolver support", title="A  Name-collapsed, hard-gated compounds")
    top = pairs.head(12).iloc[::-1]
    axes[1].barh(np.arange(len(top)), top["phase_selective_pair_score"], color="#4D908E")
    axes[1].set_yticks(np.arange(len(top)), [f"{a} -> {b}" for a, b in zip(top["early_resolver_compound"], top["late_retention_compound"])], fontsize=7)
    axes[1].set(xlabel="Ordered additive control score", title="B  Phase-selective hypotheses after safety gate")
    fig.savefig(args.v16 / "03_figures/Figure_v16_drug_hard_gate.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.v16 / "03_figures/Figure_v16_drug_hard_gate.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "name_collapsed_compounds": int(len(d)), "duplicate_id_conflicts": int(d["duplicate_id_conflict"].sum()),
        "computational_hard_gate": int(d["computational_hard_gate"].sum()),
        "tier1_mechanistic": d[d["translation_tier"].eq("Tier_1_mechanistic_probe")]["pert_iname"].tolist(),
        "tier2_translation": d[d["translation_tier"].eq("Tier_2_translation_anchor")]["pert_iname"].tolist(),
        "top_pairs": pairs.head(20).to_dict("records"),
    }
    (out / "stage4_hard_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
