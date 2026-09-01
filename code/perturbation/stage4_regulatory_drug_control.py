#!/usr/bin/env python3
"""Build the v16 regulatory-switch and phase-selective drug-control layer."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


RETENTION_GENES = [
    "ITGB5", "GAS7", "SPON1", "SVEP1", "PDGFRA", "LAMA2", "EPHA3",
    "DPYSL3", "LTBP2", "EFEMP1", "SERPINE1", "COL1A1", "COL3A1",
    "LOXL2", "ACTA2", "TIMP1", "THBS2", "FBLN5",
]
ENDO_RESOLVER_GENES = [
    "WNT9B", "WNT7B", "VWF", "ACE", "KDR", "ESAM", "EMCN", "PLVAP",
    "KLF2", "NR2F2",
]
MACRO_RESOLVER_GENES = [
    "TREM2", "GPNMB", "LPL", "MMP14", "CTSB", "CTSD", "AXL", "MERTK",
]
REGULATORY_CANDIDATES = [
    "ITGB5", "GAS7", "SPON1", "SVEP1", "PDGFRA", "LAMA2", "EPHA3",
    "DPYSL3", "LTBP2", "EFEMP1", "SERPINE1",
]

# Consensus sequences are taken directly from the authors' HOMER table.
MOTIF_CONSENSUS = {
    "AP1": ["VTGACTCATC", "GATGASTCATCN", "NDATGASTCAYN"],
    "RUNX": ["SAAACCACAG", "AAACCACARM", "NWAACCACADNN"],
    "ETS": ["ACAGGAAGTG", "AACCGGAAGT", "NRYTTCCGGH"],
}
IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "R": "[AG]", "Y": "[CT]",
    "S": "[GC]", "W": "[AT]", "K": "[GT]", "M": "[AC]", "B": "[CGT]",
    "D": "[AGT]", "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}
COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v16", type=Path, default=REPO_ROOT / "work/v16_runtime")
    p.add_argument("--legacy", type=Path, default=REPO_ROOT / "data/legacy_runtime")
    p.add_argument("--gctx", type=Path, default=REPO_ROOT / "data/raw_external/LINCS/GSE92742_Level5_COMPZ.MODZ.gctx")
    p.add_argument("--skip-drugs", action="store_true")
    return p.parse_args()


def read_excel_table(path: Path, sheet: str, skiprows: int) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name=sheet, skiprows=skiprows)
    if "PeakID" not in d.columns:
        unnamed = [c for c in d.columns if str(c).startswith("Unnamed:")]
        described = [c for c in d.columns if str(c).startswith("PeakID (")]
        if unnamed:
            d = d.rename(columns={unnamed[0]: "PeakID"})
        elif described:
            d = d.rename(columns={described[0]: "PeakID"})
        if described and described[0] in d.columns and described[0] != "PeakID":
            d = d.rename(columns={described[0]: "PeakID_annotation_command"})
    return d.dropna(how="all").reset_index(drop=True)


def annotation_class(value: object) -> str:
    s = str(value).lower()
    if "promoter" in s or "tts" in s or "utr" in s or "exon" in s:
        return "promoter_or_genic_boundary"
    if "intron" in s:
        return "intron"
    return "intergenic"


def select_matched_controls(peaks: pd.DataFrame, candidates: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    pool = peaks[~peaks["Gene Name"].astype(str).str.upper().isin(REGULATORY_CANDIDATES)].copy()
    for d in (pool, candidates):
        d["length"] = pd.to_numeric(d["End"], errors="coerce") - pd.to_numeric(d["Start"], errors="coerce")
        d["abs_tss"] = pd.to_numeric(d["Distance to TSS"], errors="coerce").abs().fillna(1e6)
        d["lfc"] = pd.to_numeric(d["avg_log2FC"], errors="coerce").fillna(0)
        d["ann_class"] = d["Annotation"].map(annotation_class)
    used: set[str] = set()
    selected = []
    for _, c in candidates.sort_values(["Gene Name", "p_val_adj", "avg_log2FC"]).iterrows():
        q = pool[~pool["PeakID"].astype(str).isin(used)].copy()
        q["distance"] = (
            np.abs(np.log1p(q["length"]) - math.log1p(c["length"]))
            + np.abs(np.log1p(q["abs_tss"]) - math.log1p(c["abs_tss"]))
            + np.abs(q["lfc"] - c["lfc"])
            + 1.5 * (q["ann_class"] != c["ann_class"]).astype(float)
            + 0.5 * (q["Chr"] != c["Chr"]).astype(float)
        )
        take = q.nsmallest(k, "distance").copy()
        take["matched_to_peak"] = c["PeakID"]
        take["matched_to_gene"] = c["Gene Name"]
        used.update(take["PeakID"].astype(str))
        selected.append(take)
    return pd.concat(selected, ignore_index=True) if selected else pool.iloc[0:0]


def fetch_ucsc_sequence(row: dict, retries: int = 4) -> tuple[str, str]:
    peak = str(row["PeakID"])
    params = urllib.parse.urlencode({
        "genome": "hg38", "chrom": row["Chr"], "start": int(row["Start"]), "end": int(row["End"]),
    })
    url = "https://api.genome.ucsc.edu/getData/sequence?" + params
    for attempt in range(retries):
        try:
            ucsc_ip = os.environ.get("UCSC_API_IP")
            if ucsc_ip:
                completed = subprocess.run(
                    ["curl", "--resolve", f"api.genome.ucsc.edu:443:{ucsc_ip}", "-fsSL", "--max-time", "30", url],
                    check=True, capture_output=True, text=True,
                )
                payload = json.loads(completed.stdout)
            else:
                with urllib.request.urlopen(url, timeout=30) as r:
                    payload = json.load(r)
            return peak, str(payload["dna"]).upper()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def get_sequences(rows: pd.DataFrame, cache_path: Path) -> dict[str, str]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    unique = rows.drop_duplicates("PeakID")
    todo = [r._asdict() for r in unique.itertuples(index=False) if str(r.PeakID) not in cache]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_ucsc_sequence, r) for r in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                peak, sequence = fut.result()
                cache[peak] = sequence
                if i % 100 == 0:
                    cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))
    return cache


def reverse_complement_iupac(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def consensus_regex(consensus: str) -> re.Pattern[str]:
    return re.compile("".join(IUPAC[x] for x in consensus.upper()))


def motif_hit(sequence: str, consensuses: list[str]) -> bool:
    for motif in consensuses:
        for query in (motif, reverse_complement_iupac(motif)):
            if consensus_regex(query).search(sequence):
                return True
    return False


def family_from_motif(name: str) -> str | None:
    u = name.upper()
    if "RUNX" in u:
        return "RUNX"
    if any(x in u for x in ("AP-1", "FOS", "FRA", "JUN")):
        return "AP1"
    if any(x in u for x in ("ETS1", "ERG(", "FLI1", "ETV", "GABPA", "ETS(ETS)")):
        return "ETS"
    return None


def regulatory_analysis(v16: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xdir = v16 / "00_raw/GSE244832/literature/xlsx"
    s9 = read_excel_table(xdir / "1-s2.0-S0168827824026679-mmc11.xlsx", "Table S9", 7)
    s9["Gene Name"] = s9["Gene Name"].astype(str).str.upper()
    activated = s9[s9["cluster"].eq("A")].copy()
    candidates = activated[activated["Gene Name"].isin(REGULATORY_CANDIDATES)].copy()
    candidates = candidates.sort_values(["Gene Name", "p_val_adj", "avg_log2FC"], ascending=[True, True, False])
    controls = select_matched_controls(activated, candidates)
    candidates["set"] = "retention_locus"
    controls["set"] = "matched_activated_peak"
    all_peaks = pd.concat([candidates, controls], ignore_index=True)
    sequences = get_sequences(all_peaks, out / "hg38_peak_sequence_cache.json")
    for fam, consensuses in MOTIF_CONSENSUS.items():
        all_peaks[f"{fam}_hit"] = all_peaks["PeakID"].astype(str).map(
            lambda x: motif_hit(sequences.get(x, ""), consensuses)
        )
    all_peaks.to_csv(out / "retention_locus_matched_motif_scan.tsv", sep="\t", index=False)

    family_rows = []
    motif_book = xdir / "1-s2.0-S0168827824026679-mmc13.xlsx"
    for cluster in ("A", "Q", "INF"):
        motifs = pd.read_excel(motif_book, sheet_name=cluster)
        motifs["family"] = motifs["Motif Name"].astype(str).map(family_from_motif)
        motifs = motifs[motifs["family"].notna()].copy()
        motifs["target_pct"] = pd.to_numeric(motifs["% of Target Sequences with Motif"].astype(str).str.rstrip("%"), errors="coerce")
        motifs["background_pct"] = pd.to_numeric(motifs["% of Background Sequences with Motif"].astype(str).str.rstrip("%"), errors="coerce")
        motifs["enrichment_or"] = (motifs["target_pct"] / (100 - motifs["target_pct"])) / (motifs["background_pct"] / (100 - motifs["background_pct"]))
        for fam, d in motifs.groupby("family"):
            r = d.sort_values("P-value").iloc[0]
            family_rows.append({
                "cluster": cluster, "family": fam, "representative_motif": r["Motif Name"],
                "consensus": r["Consensus"], "p_value": r["P-value"],
                "target_pct": r["target_pct"], "background_pct": r["background_pct"],
                "global_enrichment_or": r["enrichment_or"],
            })
    family = pd.DataFrame(family_rows)
    family.to_csv(out / "HSC_state_motif_family_enrichment.tsv", sep="\t", index=False)

    matched_rows = []
    cand = all_peaks[all_peaks["set"].eq("retention_locus")]
    ctrl = all_peaks[all_peaks["set"].eq("matched_activated_peak")]
    for fam in MOTIF_CONSENSUS:
        a = int(cand[f"{fam}_hit"].sum())
        b = int(len(cand) - a)
        c = int(ctrl[f"{fam}_hit"].sum())
        d = int(len(ctrl) - c)
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        matched_rows.append({
            "family": fam, "candidate_hits": a, "candidate_total": len(cand),
            "control_hits": c, "control_total": len(ctrl), "matched_odds_ratio": odds,
            "fisher_one_sided_p": p,
        })
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(out / "retention_locus_motif_matched_null.tsv", sep="\t", index=False)

    a_global = family[family["cluster"].eq("A")].set_index("family")["global_enrichment_or"].to_dict()
    edges = []
    for gene, d in cand.groupby("Gene Name"):
        for fam in MOTIF_CONSENSUS:
            hits = d[d[f"{fam}_hit"]]
            if hits.empty:
                continue
            strongest = hits.sort_values(["p_val_adj", "avg_log2FC"], ascending=[True, False]).iloc[0]
            score = float(strongest["avg_log2FC"]) + math.log2(max(a_global.get(fam, 1), 1e-6)) + 0.5 * math.log2(1 + len(hits))
            edges.append({
                "tf_family": fam, "gene": gene, "n_candidate_peaks": len(d),
                "n_motif_positive_peaks": len(hits), "strongest_peak": strongest["PeakID"],
                "peak_log2fc_A_vs_rest": strongest["avg_log2FC"], "peak_fdr_A_vs_rest": strongest["p_val_adj"],
                "nearest_gene_distance_to_tss": strongest["Distance to TSS"],
                "global_A_motif_enrichment_or": a_global.get(fam), "locus_edge_score": score,
                "edge_semantics": "motif-supported nearest-gene locus edge",
            })
    edges = pd.DataFrame(edges).sort_values("locus_edge_score", ascending=False)
    edges.to_csv(out / "HSC_retention_regulatory_locus_edges.tsv", sep="\t", index=False)
    return family, edges


def decode(values: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values]


def robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else series.std()
    return (series - med) / (scale if scale and np.isfinite(scale) else 1)


def drug_analysis(v16: Path, legacy: Path, gctx: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cmap = legacy / "02_results/CMap"
    gene_info = pd.read_csv(legacy / "00_raw/cmap_lincs/GSE92742/GSE92742_gene_info.txt.gz", sep="\t", dtype=str)
    gene_info["symbol"] = gene_info["pr_gene_symbol"].str.upper()
    symbol_to_id = gene_info.drop_duplicates("symbol").set_index("symbol")["pr_gene_id"].to_dict()
    candidates = pd.read_csv(cmap / "LINCS_candidate_liver_or_hepatic_signatures.tsv", sep="\t", low_memory=False)
    candidates = candidates[candidates["pert_type"].eq("trt_cp")].drop_duplicates("sig_id")

    sets = {"retention": RETENTION_GENES, "endo": ENDO_RESOLVER_GENES, "macro": MACRO_RESOLVER_GENES}
    rows = []
    with h5py.File(gctx, "r") as h:
        matrix = h["0/DATA/0/matrix"]
        sig_ids = decode(h["0/META/COL/id"][:])
        gene_ids = decode(h["0/META/ROW/id"][:])
        sig_index = {s: i for i, s in enumerate(sig_ids)}
        gene_index = {g: i for i, g in enumerate(gene_ids)}
        candidates = candidates[candidates["sig_id"].astype(str).isin(sig_index)].copy()
        set_cols = {}
        for name, genes in sets.items():
            mapped = [(g, symbol_to_id.get(g)) for g in genes]
            set_cols[name] = [(g, gene_index[e]) for g, e in mapped if e in gene_index]
        (out / "dual_axis_gene_mapping.json").write_text(json.dumps({k: [g for g, _ in v] for k, v in set_cols.items()}, indent=2) + "\n")
        all_cols = sorted({i for values in set_cols.values() for _, i in values})
        positions = {name: [all_cols.index(i) for _, i in values] for name, values in set_cols.items()}
        indexed = sorted((sig_index[s], s) for s in candidates["sig_id"].astype(str))
        for start in range(0, len(indexed), 1000):
            batch = indexed[start:start + 1000]
            data = matrix[[i for i, _ in batch], :][:, all_cols].astype(np.float64)
            means = {name: np.nanmean(data[:, pos], axis=1) for name, pos in positions.items()}
            for j, (_, sid) in enumerate(batch):
                retention = -float(means["retention"][j])
                endo = float(means["endo"][j])
                macro = float(means["macro"][j])
                resolver_floor = min(endo, macro)
                dual = retention + 0.5 * endo + 0.5 * macro - 1.5 * max(0, -resolver_floor)
                rows.append({
                    "sig_id": sid, "retention_suppression": retention,
                    "endothelial_resolution_support": endo, "macrophage_resolution_support": macro,
                    "resolver_floor": resolver_floor, "dual_control_score": dual,
                })
    signature = pd.DataFrame(rows).merge(candidates, on="sig_id", how="left")
    signature["dual_positive"] = (signature["retention_suppression"] > 0) & (signature["resolver_floor"] >= 0)
    signature.to_csv(out / "LINCS_dual_axis_signature_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    agg = signature.groupby(["pert_id", "pert_iname"], dropna=False).agg(
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
    ).reset_index()
    agg = agg[(agg["n_signatures"] >= 3) & (agg["n_cell_lines"] >= 2)].copy()
    agg["robust_dual_control"] = agg["median_dual_control"] + 0.5 * agg["q25_dual_control"]
    agg["concurrent_score"] = robust_z(agg["robust_dual_control"]) + robust_z(agg["fraction_dual_positive"])
    agg["early_resolver_score"] = robust_z(agg[["median_endo_support", "median_macro_support"]].mean(axis=1)) + 0.5 * robust_z(agg["median_resolver_floor"])
    agg["late_retention_score"] = robust_z(agg["median_retention_suppression"]) + 0.5 * robust_z(agg["q25_retention_suppression"])

    annotation_path = legacy / "06_CMap_LINCS/drug_reverse_score_trt_cp_annotated_top500.tsv"
    if annotation_path.exists():
        ann = pd.read_csv(annotation_path, sep="\t", low_memory=False)
        keys = [c for c in ["pert_id", "pert_iname"] if c in ann.columns]
        useful = keys + [c for c in ann.columns if any(x in c.lower() for x in ("moa", "target", "safety", "identity"))]
        ann = ann[useful].drop_duplicates(keys)
        agg = agg.merge(ann, on=keys, how="left", suffixes=("", "_annotation"))
    agg = agg.sort_values("concurrent_score", ascending=False)
    agg["concurrent_rank"] = np.arange(1, len(agg) + 1)
    agg["early_resolver_rank"] = agg["early_resolver_score"].rank(ascending=False, method="min").astype(int)
    agg["late_retention_rank"] = agg["late_retention_score"].rank(ascending=False, method="min").astype(int)
    agg.to_csv(out / "LINCS_dual_axis_compound_ranking.tsv", sep="\t", index=False)

    early = agg[(agg["median_resolver_floor"] > 0) & (agg["median_retention_suppression"] > -0.25)].nsmallest(30, "early_resolver_rank")
    late = agg[(agg["median_retention_suppression"] > 0) & (agg["median_resolver_floor"] > -0.25)].nsmallest(30, "late_retention_rank")
    pairs = []
    for _, e in early.iterrows():
        for _, l in late.iterrows():
            if e["pert_id"] == l["pert_id"]:
                continue
            pair_ret = l["median_retention_suppression"] + 0.5 * e["median_retention_suppression"]
            pair_endo = e["median_endo_support"] + 0.5 * l["median_endo_support"]
            pair_macro = e["median_macro_support"] + 0.5 * l["median_macro_support"]
            score = pair_ret + 0.5 * pair_endo + 0.5 * pair_macro - 1.5 * max(0, -min(pair_endo, pair_macro))
            pairs.append({
                "early_resolver_compound": e["pert_iname"], "early_pert_id": e["pert_id"],
                "late_retention_compound": l["pert_iname"], "late_pert_id": l["pert_id"],
                "predicted_pair_retention_suppression": pair_ret,
                "predicted_pair_endo_support": pair_endo, "predicted_pair_macro_support": pair_macro,
                "phase_selective_pair_score": score,
                "interpretation": "ordered additive hypothesis; not a synergy estimate",
            })
    pairs = pd.DataFrame(pairs).sort_values("phase_selective_pair_score", ascending=False)
    null_scores = []
    rng = np.random.default_rng(20260711)
    eligible = agg.index.to_numpy()
    for _ in range(20000):
        i, j = rng.choice(eligible, 2, replace=False)
        e, l = agg.loc[i], agg.loc[j]
        pr = l["median_retention_suppression"] + .5 * e["median_retention_suppression"]
        pe = e["median_endo_support"] + .5 * l["median_endo_support"]
        pm = e["median_macro_support"] + .5 * l["median_macro_support"]
        null_scores.append(pr + .5 * pe + .5 * pm - 1.5 * max(0, -min(pe, pm)))
    null = np.asarray(null_scores)
    pairs["random_pair_percentile"] = [float((null <= x).mean()) for x in pairs["phase_selective_pair_score"]]
    pairs.to_csv(out / "LINCS_phase_selective_ordered_pairs.tsv", sep="\t", index=False)

    sens = []
    for w_ret, w_endo, w_macro in [(1, .5, .5), (1.5, .25, .25), (.75, .75, .75), (1, 1, 0), (1, 0, 1)]:
        score = w_ret * agg["median_retention_suppression"] + w_endo * agg["median_endo_support"] + w_macro * agg["median_macro_support"]
        rank = score.rank(ascending=False, method="min")
        for idx in agg.nsmallest(100, "concurrent_rank").index:
            sens.append({"pert_id": agg.loc[idx, "pert_id"], "pert_iname": agg.loc[idx, "pert_iname"], "weight_scheme": f"{w_ret}:{w_endo}:{w_macro}", "rank": int(rank.loc[idx])})
    sensitivity = pd.DataFrame(sens)
    stability = sensitivity.groupby(["pert_id", "pert_iname"]).agg(median_sensitivity_rank=("rank", "median"), worst_sensitivity_rank=("rank", "max"), n_top50_schemes=("rank", lambda x: int((x <= 50).sum()))).reset_index()
    stability.to_csv(out / "LINCS_dual_axis_weight_sensitivity.tsv", sep="\t", index=False)
    return agg, pairs, signature


def make_figure(family: pd.DataFrame, edges: pd.DataFrame, compounds: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    colors = {"A": "#C7472E", "Q": "#2A6F97", "INF": "#6A994E"}
    ax = axes[0, 0]
    pivot = family.pivot(index="family", columns="cluster", values="global_enrichment_or").reindex(["AP1", "RUNX", "ETS"])
    x = np.arange(len(pivot))
    for i, cl in enumerate(["A", "Q", "INF"]):
        ax.bar(x + (i - 1) * .23, pivot[cl], width=.22, color=colors[cl], label=cl)
    ax.axhline(1, color="black", lw=.8)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Target/background motif odds ratio")
    ax.set_title("A  HSC state regulatory architecture", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    if not edges.empty:
        matrix = edges.pivot_table(index="gene", columns="tf_family", values="locus_edge_score", aggfunc="max").reindex(columns=["AP1", "RUNX", "ETS"])
        matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
        im = ax.imshow(matrix.fillna(0), cmap="YlOrRd", aspect="auto")
        ax.set_yticks(np.arange(len(matrix)), matrix.index, fontsize=8)
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
        fig.colorbar(im, ax=ax, fraction=.04, label="Motif-supported locus edge score")
    ax.set_title("B  Candidate retention loci", loc="left", fontweight="bold")

    ax = axes[1, 0]
    plot = compounds.copy()
    resolver = plot[["median_endo_support", "median_macro_support"]].mean(axis=1)
    ax.scatter(plot["median_retention_suppression"], resolver, c=plot["fraction_dual_positive"], cmap="viridis", s=18, alpha=.55)
    ax.axhline(0, color="#777777", lw=.8)
    ax.axvline(0, color="#777777", lw=.8)
    for _, r in plot.nsmallest(10, "concurrent_rank").iterrows():
        ax.text(r["median_retention_suppression"], np.mean([r["median_endo_support"], r["median_macro_support"]]), str(r["pert_iname"]), fontsize=7)
    ax.set_xlabel("HSC retention suppression")
    ax.set_ylabel("Mean resolver support")
    ax.set_title("C  LINCS dual-control landscape", loc="left", fontweight="bold")

    ax = axes[1, 1]
    top = pairs.head(12).iloc[::-1]
    labels = [f"{a} -> {b}" for a, b in zip(top["early_resolver_compound"], top["late_retention_compound"])]
    ax.barh(np.arange(len(top)), top["phase_selective_pair_score"], color="#4D908E")
    ax.set_yticks(np.arange(len(top)), labels, fontsize=7)
    ax.set_xlabel("Ordered additive control score")
    ax.set_title("D  Phase-selective intervention hypotheses", loc="left", fontweight="bold")
    fig.suptitle("Regulatory and pharmacologic control of the scar-interface reaction front", fontsize=16, fontweight="bold")
    fig.savefig(out / "Figure_v16_regulatory_drug_control.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "Figure_v16_regulatory_drug_control.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.v16 / "02_results/S5_regulatory_drug_control"
    figdir = args.v16 / "03_figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    family, edges = regulatory_analysis(args.v16, out)
    if args.skip_drugs:
        return
    compounds, pairs, signature = drug_analysis(args.v16, args.legacy, args.gctx, out)
    make_figure(family, edges, compounds, pairs, figdir)
    summary = {
        "regulatory_edges": int(len(edges)), "compounds_robustly_scored": int(len(compounds)),
        "signatures_scored": int(len(signature)), "ordered_pairs_scored": int(len(pairs)),
        "top_concurrent": compounds.head(20)[["pert_iname", "concurrent_score"]].to_dict("records"),
        "top_ordered_pairs": pairs.head(20)[["early_resolver_compound", "late_retention_compound", "phase_selective_pair_score", "random_pair_percentile"]].to_dict("records"),
    }
    (out / "stage4_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
