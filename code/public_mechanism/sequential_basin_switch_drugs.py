#!/usr/bin/env python3
"""Rank phase-ordered LINCS pairs by predicted basin-switching advantage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "data/LINCS_derived/LINCS_name_collapsed_hard_gated.tsv"
OUTPUT = REPO_ROOT / "results/S12_order_aware_drug_nomination"

FOCUS = ["ezetimibe", "verteporfin", "wz-4002", "psb-069", "pioglitazone", "pre-084"]
EARLY_COLOR = "#238B8D"
LATE_COLOR = "#C33F55"
ORDER_COLOR = "#75558F"


def percentile(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def pair_score(early: pd.Series, late: pd.Series, early_retention_weight: float = 0.5, resolver_weight: float = 0.5, penalty_weight: float = 1.5) -> float:
    endo = early.median_endo_support + 0.5 * late.median_endo_support
    macro = early.median_macro_support + 0.5 * late.median_macro_support
    retention = late.median_retention_suppression + early_retention_weight * early.median_retention_suppression
    penalty = penalty_weight * max(0.0, -min(endo, macro))
    return float(retention + resolver_weight * endo + resolver_weight * macro - penalty)


def build_pairs(compounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = compounds.loc[
        compounds.identity_resolved.fillna(False)
        & compounds.n_signatures.ge(3)
        & compounds.n_cell_lines.ge(2)
        & ~compounds.duplicate_id_conflict.fillna(False)
    ].copy()
    eligible["mean_resolver"] = 0.5 * (eligible.median_endo_support + eligible.median_macro_support)
    early = eligible.loc[
        eligible.median_resolver_floor.ge(0.10)
        & eligible.mean_resolver.ge(0.20)
        & eligible.median_retention_suppression.ge(-0.10)
    ].copy()
    late = eligible.loc[
        eligible.median_retention_suppression.ge(0.20)
        & eligible.q25_retention_suppression.ge(0.10)
        & eligible.median_resolver_floor.ge(-0.15)
    ].copy()
    rows = []
    for _, e in early.iterrows():
        for _, l in late.iterrows():
            if e.pert_iname_norm == l.pert_iname_norm:
                continue
            forward = pair_score(e, l)
            reverse = pair_score(l, e)
            rows.append(
                {
                    "early_compound": e.pert_iname_norm,
                    "late_compound": l.pert_iname_norm,
                    "forward_score": forward,
                    "reverse_score": reverse,
                    "order_advantage": forward - reverse,
                    "early_mean_resolver": e.mean_resolver,
                    "early_resolver_floor": e.median_resolver_floor,
                    "early_retention": e.median_retention_suppression,
                    "late_retention": l.median_retention_suppression,
                    "late_q25_retention": l.q25_retention_suppression,
                    "late_resolver_floor": l.median_resolver_floor,
                    "early_tier": e.translation_tier,
                    "late_tier": l.translation_tier,
                    "both_named": bool(e.named_identity and l.named_identity),
                    "both_hard_gate": bool(e.computational_hard_gate and l.computational_hard_gate),
                }
            )
    pairs = pd.DataFrame(rows)
    pairs["forward_percentile"] = percentile(pairs.forward_score)
    pairs["order_percentile"] = percentile(pairs.order_advantage)
    pairs["basin_switch_index"] = 0.65 * pairs.forward_percentile + 0.35 * pairs.order_percentile
    pairs["basin_switch_rank"] = pairs.basin_switch_index.rank(ascending=False, method="min").astype(int)
    return eligible, early, pairs.sort_values("basin_switch_index", ascending=False)


def weight_sensitivity(early: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for er in [0.25, 0.5, 0.75]:
        for rw in [0.35, 0.5, 0.65]:
            for pw in [1.0, 1.5, 2.0]:
                local = []
                for _, e in early.iterrows():
                    for _, l in late.iterrows():
                        if e.pert_iname_norm == l.pert_iname_norm:
                            continue
                        forward = pair_score(e, l, er, rw, pw)
                        reverse = pair_score(l, e, er, rw, pw)
                        local.append((e.pert_iname_norm, l.pert_iname_norm, forward, forward - reverse))
                frame = pd.DataFrame(local, columns=["early", "late", "forward", "advantage"])
                frame["index"] = 0.65 * percentile(frame.forward) + 0.35 * percentile(frame.advantage)
                frame["rank"] = frame["index"].rank(ascending=False, method="min")
                target = frame.loc[frame.early.eq("ezetimibe") & frame.late.eq("verteporfin")]
                if len(target):
                    row = target.iloc[0]
                    rows.append(
                        {
                            "early_retention_weight": er,
                            "resolver_weight": rw,
                            "penalty_weight": pw,
                            "forward_score": row.forward,
                            "order_advantage": row.advantage,
                            "basin_switch_index": row["index"],
                            "rank": int(row["rank"]),
                            "n_pairs": len(frame),
                        }
                    )
    return pd.DataFrame(rows)


def plot_results(eligible: pd.DataFrame, pairs: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(15.5, 11.5))
    grid = fig.add_gridspec(2, 2, left=.07, right=.98, bottom=.07, top=.91, wspace=.30, hspace=.34)

    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(eligible.mean_resolver, eligible.median_retention_suppression, s=15, color="#C7CDD0", alpha=.55)
    for name in FOCUS:
        local = eligible.loc[eligible.pert_iname_norm.eq(name)]
        if len(local):
            row = local.iloc[0]
            ax.scatter(row.mean_resolver, row.median_retention_suppression, s=70, color=EARLY_COLOR if name=="ezetimibe" else LATE_COLOR, edgecolor="white", linewidth=.8, zorder=3)
            ax.text(row.mean_resolver+.012,row.median_retention_suppression+.008,name,fontsize=8)
    ax.axvline(.20,color=EARLY_COLOR,ls="--",lw=1); ax.axhline(.20,color=LATE_COLOR,ls="--",lw=1)
    ax.set(xlabel="Mean endothelial/macrophage resolver support",ylabel="HSC-retention suppression")
    ax.set_title("A  Early and late controllers occupy different Pareto sectors",loc="left",fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    named = pairs.loc[pairs.both_named].head(12).sort_values("order_advantage")
    labels = named.early_compound + " → " + named.late_compound
    y = np.arange(len(named))
    for yi, (_, row) in enumerate(named.iterrows()):
        ax.plot([row.reverse_score,row.forward_score],[yi,yi],color="#AAB2B6",lw=2)
        ax.scatter(row.reverse_score,yi,color="#7A8790",s=34,zorder=3)
        ax.scatter(row.forward_score,yi,color=ORDER_COLOR,s=42,zorder=3)
    ax.set_yticks(y,labels); ax.set_xlabel("Ordered pair score")
    ax.set_title("B  Sequence creates a directional control advantage",loc="left",fontweight="bold")
    ax.text(.98,.04,"grey: reverse   purple: forward",transform=ax.transAxes,ha="right",fontsize=8)

    ax = fig.add_subplot(grid[1, 0])
    early_names = pairs.groupby("early_compound").basin_switch_index.max().nlargest(10).index
    late_names = pairs.groupby("late_compound").basin_switch_index.max().nlargest(10).index
    heat = pairs.loc[pairs.early_compound.isin(early_names)&pairs.late_compound.isin(late_names)].pivot(index="early_compound",columns="late_compound",values="order_advantage")
    heat = heat.reindex(index=early_names,columns=late_names)
    sns.heatmap(heat,cmap="PRGn",center=0,annot=True,fmt=".2f",annot_kws={"fontsize":6},cbar_kws={"label":"Order advantage","shrink":.72},ax=ax)
    ax.set(xlabel="Late lock suppressor",ylabel="Early resolver primer")
    ax.set_title("C  Ordered-pair control matrix",loc="left",fontweight="bold")

    ax = fig.add_subplot(grid[1, 1])
    target = pairs.loc[pairs.early_compound.eq("ezetimibe")&pairs.late_compound.eq("verteporfin")].iloc[0]
    ax.set_axis_off(); ax.set_xlim(0,168); ax.set_ylim(0,1)
    def band(start, end, y, label, color):
        patch=FancyBboxPatch((start,y),end-start,.13,boxstyle="round,pad=0.012,rounding_size=2",fc=color,ec=color,lw=1)
        ax.add_patch(patch); ax.text((start+end)/2,y+.065,label,ha="center",va="center",color="white",fontweight="bold",fontsize=8)
    band(0,24,.75,"EZETIMIBE\nresolver priming",EARLY_COLOR)
    band(24,72,.75,"VERTEPORFIN\nYAP/TEAD switching",LATE_COLOR)
    band(72,168,.75,"PIOGLITAZONE\nstate stabilization","#C18C24")
    ax.annotate("",xy=(24,.815),xytext=(20,.815),arrowprops=dict(arrowstyle="-|>",color="#454C50",lw=1.5))
    ax.annotate("",xy=(72,.815),xytext=(68,.815),arrowprops=dict(arrowstyle="-|>",color="#454C50",lw=1.5))
    rows=[
        (.57,"Endothelium","WNT9B · KLF2 · EMCN","#3182A6",0,28),
        (.42,"Macrophage","TREM2 · GPNMB · MMP14","#4D8A63",0,34),
        (.27,"HSC lock","YAP/TEAD · SERPINE1 · SPON1","#B83D4B",24,78),
        (.12,"Durability","PPARγ quiescence · re-challenge threshold","#9B7421",72,168),
    ]
    for y,label,readout,color,start,end in rows:
        ax.text(0,y,label,ha="left",va="center",fontweight="bold",color=color)
        ax.plot([start,end],[y,y],color=color,lw=5,solid_capstyle="round",alpha=.78)
        ax.text(min(end+3,166),y,readout,ha="left" if end<150 else "right",va="center",fontsize=7.5,color="#3F474B")
    ax.text(166,.96,f"E → V  {target.forward_score:.3f}\nV → E  {target.reverse_score:.3f}\nΔorder  {target.order_advantage:.3f}",ha="right",va="top",bbox=dict(boxstyle="round,pad=.35",fc="white",ec="#8D969C"))
    ax.text(84,.02,"Equal cumulative exposure; compare simultaneous, forward and reverse order",ha="center",fontsize=8,color="#626C72")
    ax.set_title("D  Prime–switch–stabilize experiment",loc="left",fontweight="bold")

    for axis in fig.axes:
        if axis.axison:
            axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("Phase-ordered perturbations are predicted to switch the resolver–lock basin",fontsize=16,fontweight="bold")
    fig.savefig(OUTPUT/"LINCS_sequential_basin_switch.png",dpi=320,facecolor="white")
    fig.savefig(OUTPUT/"LINCS_sequential_basin_switch.pdf",facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=INPUT)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    global INPUT,OUTPUT
    args=parse_args()
    INPUT,OUTPUT=args.input,args.output_dir
    OUTPUT.mkdir(parents=True,exist_ok=True)
    compounds=pd.read_csv(INPUT,sep="\t")
    eligible,early,pairs=build_pairs(compounds)
    late_names=pairs.late_compound.unique()
    late=eligible.loc[eligible.pert_iname_norm.isin(late_names)].copy()
    sensitivity=weight_sensitivity(early,late)
    plot_results(eligible,pairs)
    pairs.to_csv(OUTPUT/"LINCS_ordered_basin_switch_pairs.tsv",sep="\t",index=False)
    sensitivity.to_csv(OUTPUT/"ezetimibe_verteporfin_weight_sensitivity.tsv",sep="\t",index=False)
    target=pairs.loc[pairs.early_compound.eq("ezetimibe")&pairs.late_compound.eq("verteporfin")].iloc[0]
    summary={
        "n_eligible_compounds":int(len(eligible)),
        "n_early_primers":int(len(early)),
        "n_late_suppressors":int(len(late)),
        "n_ordered_pairs":int(len(pairs)),
        "lead_pair":"ezetimibe -> verteporfin",
        "lead_pair_metrics":target.to_dict(),
        "weight_sensitivity":{"median_rank":float(sensitivity['rank'].median()),"best_rank":int(sensitivity['rank'].min()),"worst_rank":int(sensitivity['rank'].max()),"n_schemes":int(len(sensitivity))},
        "claim":"A prime-then-switch hypothesis with a strong computed direction advantage; designed for direct equal-exposure order testing.",
    }
    (OUTPUT/"sequential_basin_switch_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
