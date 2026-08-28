#!/usr/bin/env python3
"""Test AP-1 factor direction with LINCS consensus shRNA, hairpins and overexpression."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
META = REPO_ROOT / "data/raw_external/LINCS/GSE92742"
GCTX = REPO_ROOT / "data/raw_external/LINCS/GSE92742_Level5_COMPZ.MODZ.gctx"
FRA = REPO_ROOT / "data/perturbation_inputs/fibrosis_reversibility_axis_v15.tsv"
OUT = REPO_ROOT / "results/S8_LINCS_AP1_genetic_perturbation"
FIG = OUT / "figures"
TARGETS = ["JUN", "FOS", "JUNB", "FOSL2"]

RETENTION = [
    "ITGB5", "GAS7", "SPON1", "SVEP1", "PDGFRA", "LAMA2", "EPHA3",
    "DPYSL3", "LTBP2", "EFEMP1", "SERPINE1", "COL1A1", "COL3A1",
    "LOXL2", "ACTA2", "TIMP1", "THBS2", "FBLN5",
]
ENDO = ["WNT9B", "WNT7B", "VWF", "ACE", "KDR", "ESAM", "EMCN", "PLVAP", "KLF2", "NR2F2"]
MACRO = ["TREM2", "GPNMB", "LPL", "MMP14", "CTSB", "CTSD", "AXL", "MERTK"]


def decode(values: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=META)
    parser.add_argument("--gctx", type=Path, default=GCTX)
    parser.add_argument("--fra-axis", type=Path, default=FRA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--figure-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    global META, GCTX, FRA, OUT, FIG
    args = parse_args()
    META, GCTX, FRA, OUT = args.metadata_dir, args.gctx, args.fra_axis, args.output_dir
    FIG = args.figure_dir or OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(META / "GSE92742_sig_info.txt.gz", sep="\t", low_memory=False)
    gene = pd.read_csv(META / "GSE92742_gene_info.txt.gz", sep="\t", dtype=str)
    gene["symbol"] = gene["pr_gene_symbol"].str.upper()
    symbol_to_id = gene.drop_duplicates("symbol").set_index("symbol")["pr_gene_id"].to_dict()
    fra = pd.read_csv(FRA, sep="\t")
    fra = fra.loc[fra["fra_axis_weight"] > 0, ["gene_symbol", "fra_axis_weight"]].drop_duplicates("gene_symbol")
    fra["gene_symbol"] = fra["gene_symbol"].str.upper()

    target_mask = sig["pert_iname"].astype(str).str.upper().isin(TARGETS)
    target_mask &= sig["pert_type"].isin(["trt_sh.cgs", "trt_sh", "trt_oe"])
    null_mask = sig["pert_type"].eq("trt_sh.cgs")
    selected = sig[target_mask | null_mask].copy()

    with h5py.File(GCTX, "r") as handle:
        matrix = handle["0/DATA/0/matrix"]
        sig_ids = decode(handle["0/META/COL/id"][:])
        gene_ids = decode(handle["0/META/ROW/id"][:])
        sig_index = {value: i for i, value in enumerate(sig_ids)}
        gene_index = {value: i for i, value in enumerate(gene_ids)}
        selected = selected[selected["sig_id"].astype(str).isin(sig_index)].copy()

        mapped = {
            "retention": [(g, symbol_to_id.get(g)) for g in RETENTION],
            "endo": [(g, symbol_to_id.get(g)) for g in ENDO],
            "macro": [(g, symbol_to_id.get(g)) for g in MACRO],
        }
        mapped = {name: [(g, gene_index[i]) for g, i in values if i in gene_index] for name, values in mapped.items()}
        fra["gctx_index"] = fra["gene_symbol"].map(symbol_to_id).map(gene_index)
        fra = fra.dropna(subset=["gctx_index"]).copy()
        fra["gctx_index"] = fra["gctx_index"].astype(int)
        all_cols = sorted(set(i for values in mapped.values() for _, i in values) | set(fra["gctx_index"]))
        col_pos = {value: i for i, value in enumerate(all_cols)}
        positions = {name: [col_pos[i] for _, i in values] for name, values in mapped.items()}
        fra_pos = [col_pos[i] for i in fra["gctx_index"]]
        fra_weights = fra["fra_axis_weight"].to_numpy(float)
        fra_weights = fra_weights / fra_weights.sum()

        pairs = sorted((sig_index[sid], sid) for sid in selected["sig_id"].astype(str))
        output = []
        for start in range(0, len(pairs), 1000):
            batch = pairs[start:start + 1000]
            values = matrix[[i for i, _ in batch], :][:, all_cols].astype(np.float64)
            retention = -np.nanmean(values[:, positions["retention"]], axis=1)
            endo = np.nanmean(values[:, positions["endo"]], axis=1)
            macro = np.nanmean(values[:, positions["macro"]], axis=1)
            fra_rescue = -(values[:, fra_pos] @ fra_weights)
            floor = np.minimum(endo, macro)
            dual = retention + 0.5 * endo + 0.5 * macro - 1.5 * np.maximum(0, -floor)
            for j, (_, sid) in enumerate(batch):
                output.append({
                    "sig_id": sid,
                    "retention_suppression": retention[j],
                    "endo_resolver_support": endo[j],
                    "macro_resolver_support": macro[j],
                    "fra_rescue": fra_rescue[j],
                    "dual_control_score": dual[j],
                })

    scores = pd.DataFrame(output).merge(selected, on="sig_id", how="left")
    scores.to_csv(OUT / "LINCS_AP1_genetic_signature_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    cgs = scores[scores["pert_type"].eq("trt_sh.cgs")].copy()
    target_cgs = cgs[cgs["pert_iname"].astype(str).str.upper().isin(TARGETS)].copy()
    null_rows = []
    for row in target_cgs.itertuples(index=False):
        null = cgs[(cgs["cell_id"].eq(row.cell_id)) & (cgs["pert_time"].eq(row.pert_time))]
        record = {"sig_id": row.sig_id, "target": str(row.pert_iname).upper(), "cell_id": row.cell_id,
                  "pert_time": row.pert_time, "n_same_context_cgs": len(null)}
        for metric in ["retention_suppression", "fra_rescue", "dual_control_score"]:
            value = getattr(row, metric)
            record[metric] = value
            record[f"{metric}_percentile"] = float((null[metric] <= value).mean())
            p_low = float((1 + (null[metric] <= value).sum()) / (len(null) + 1))
            p_high = float((1 + (null[metric] >= value).sum()) / (len(null) + 1))
            record[f"{metric}_empirical_p_low"] = p_low
            record[f"{metric}_empirical_p_high"] = p_high
            record[f"{metric}_empirical_p_two_sided"] = min(1.0, 2 * min(p_low, p_high))
        null_rows.append(record)
    target_null = pd.DataFrame(null_rows)
    target_null.to_csv(OUT / "AP1_consensus_shRNA_context_null.tsv", sep="\t", index=False)

    target_scores = scores[scores["pert_iname"].astype(str).str.upper().isin(TARGETS)].copy()
    target_scores["target"] = target_scores["pert_iname"].astype(str).str.upper()
    summary = target_scores.groupby(["target", "pert_type", "cell_id"], observed=True).agg(
        n_signatures=("sig_id", "nunique"), n_pert_ids=("pert_id", "nunique"),
        median_retention_suppression=("retention_suppression", "median"),
        fraction_retention_positive=("retention_suppression", lambda x: float((x > 0).mean())),
        median_fra_rescue=("fra_rescue", "median"),
        fraction_fra_positive=("fra_rescue", lambda x: float((x > 0).mean())),
        median_dual_control=("dual_control_score", "median"),
    ).reset_index()
    summary.to_csv(OUT / "AP1_genetic_perturbation_summary.tsv", sep="\t", index=False)

    target_cgs["target"] = target_cgs["pert_iname"].astype(str).str.upper()
    pan = target_cgs.groupby("target", observed=True).agg(
        n_contexts=("sig_id", "size"),
        cell_lines=("cell_id", lambda x: ";".join(sorted(set(map(str, x))))),
        median_retention_suppression=("retention_suppression", "median"),
        fraction_retention_positive=("retention_suppression", lambda x: float((x > 0).mean())),
        median_fra_rescue=("fra_rescue", "median"),
        fraction_fra_positive=("fra_rescue", lambda x: float((x > 0).mean())),
        median_dual_control=("dual_control_score", "median"),
        fraction_dual_positive=("dual_control_score", lambda x: float((x > 0).mean())),
    ).reset_index()
    pan.to_csv(OUT / "AP1_consensus_shRNA_pan_context.tsv", sep="\t", index=False)

    wide = summary[summary["cell_id"].eq("HEPG2")].pivot(index="target", columns="pert_type", values="median_fra_rescue")
    for column in ["trt_sh.cgs", "trt_sh", "trt_oe"]:
        if column not in wide:
            wide[column] = np.nan
    wide["cgs_hairpin_same_sign"] = np.sign(wide["trt_sh.cgs"]) == np.sign(wide["trt_sh"])
    wide["cgs_oe_opposite_sign"] = np.sign(wide["trt_sh.cgs"]) == -np.sign(wide["trt_oe"])
    wide.reset_index().to_csv(OUT / "AP1_HEPG2_directional_concordance.tsv", sep="\t", index=False)

    arbitration = target_null[target_null["cell_id"].eq("HEPG2")].merge(
        pan[["target", "n_contexts", "median_fra_rescue", "fraction_fra_positive",
             "median_dual_control", "fraction_dual_positive"]], on="target", how="left"
    ).merge(wide.reset_index(), on="target", how="left", suffixes=("", "_hepg2_median"))
    arbitration.to_csv(OUT / "AP1_HEPG2_arbitration.tsv", sep="\t", index=False)

    sns.set_theme(style="ticks", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    hep = target_cgs[target_cgs["cell_id"].eq("HEPG2")].copy()
    hep["target"] = hep["pert_iname"].astype(str).str.upper()
    hep = hep.set_index("target").reindex(TARGETS).reset_index()
    x = np.arange(len(hep))
    ax = axes[0, 0]
    metrics = [("retention_suppression", "Retention", "#C7472E"), ("fra_rescue", "FRA", "#2A6F97"),
               ("dual_control_score", "Dual control", "#6A994E")]
    for j, (column, label, color) in enumerate(metrics):
        ax.bar(x + (j - 1) * .24, hep[column], width=.22, color=color, label=label)
    ax.axhline(0, color="#777777", lw=.8)
    ax.set_xticks(x, hep["target"])
    ax.set_ylabel("LINCS Level 5 score")
    ax.set_title("A  HepG2 consensus shRNA direction", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    null_hep = target_null[target_null["cell_id"].eq("HEPG2")].set_index("target").reindex(TARGETS).reset_index()
    ax.bar(x, null_hep["fra_rescue_percentile"], color="#2A6F97")
    ax.axhline(.95, color="#C7472E", ls="--", lw=1)
    ax.set_xticks(x, null_hep["target"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Percentile among same-context CGS")
    ax.set_title("B  Empirical same-cell, same-time null", loc="left", fontweight="bold")

    ax = axes[1, 0]
    pan_heatmap = target_cgs.pivot_table(index="pert_iname", columns="cell_id", values="fra_rescue", aggfunc="median")
    pan_heatmap.index = pan_heatmap.index.astype(str).str.upper()
    pan_heatmap = pan_heatmap.reindex(TARGETS)
    sns.heatmap(pan_heatmap, cmap="vlag", center=0, linewidths=.3, cbar_kws={"label": "FRA rescue"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title("C  Consensus knockdown across cell contexts", loc="left", fontweight="bold")

    ax = axes[1, 1]
    hep_summary = summary[summary["cell_id"].eq("HEPG2")].copy()
    plot = hep_summary.pivot(index="target", columns="pert_type", values="median_fra_rescue").reindex(TARGETS)
    for j, (column, label, color) in enumerate([
        ("trt_sh.cgs", "Consensus shRNA", "#2A6F97"),
        ("trt_sh", "Individual shRNA", "#EDAE49"),
        ("trt_oe", "Overexpression", "#C7472E"),
    ]):
        values = plot[column] if column in plot else np.full(len(plot), np.nan)
        ax.bar(x + (j - 1) * .24, values, width=.22, label=label, color=color)
    ax.axhline(0, color="#777777", lw=.8)
    ax.set_xticks(x, plot.index)
    ax.set_ylabel("Median FRA rescue")
    ax.set_title("D  Hairpin and overexpression polarity", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Genetic LINCS signatures test AP-1 factor direction outside the network models",
                 fontsize=14, fontweight="bold")
    fig.savefig(FIG / "Figure_v16_LINCS_AP1_genetic_perturbation.png", dpi=350, bbox_inches="tight")
    fig.savefig(FIG / "Figure_v16_LINCS_AP1_genetic_perturbation.pdf", bbox_inches="tight")
    plt.close(fig)

    print(target_null[target_null["cell_id"].eq("HEPG2")].to_string(index=False))
    print(wide.to_string())


if __name__ == "__main__":
    main()
