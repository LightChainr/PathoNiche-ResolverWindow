#!/usr/bin/env python3
"""Build an evidence-coded candidate circuit for HSC retention and scar persistence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EDGES = REPO_ROOT / "data/mechanism_summary/mechanism_edge_evidence.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "figures/source_panels/Fig7A_Evidence_Coded_Mechanism"

NODE_STYLE = {
    "resolver": {"face": "#E8F5F5", "edge": "#258B92", "text": "#165F65"},
    "proteolysis": {"face": "#EEF7F3", "edge": "#4A9A79", "text": "#286A51"},
    "retention": {"face": "#FBEDEF", "edge": "#BF4960", "text": "#873146"},
    "mechanical": {"face": "#F2EEF7", "edge": "#76648E", "text": "#534366"},
    "matrix": {"face": "#F4F2F1", "edge": "#6C6A68", "text": "#42413F"},
}

EDGE_STYLE = {
    "literature-established": {"color": "#4D565C", "linestyle": "-", "linewidth": 1.45},
    "project-supported association": {"color": "#B43E58", "linestyle": "--", "linewidth": 1.55},
    "testable candidate edge": {"color": "#C58B22", "linestyle": ":", "linewidth": 1.8},
}

NODES = {
    "WNT9B+ endothelium": (0.08, 0.80, "resolver"),
    "HSC deactivation": (0.28, 0.80, "resolver"),
    "TREM2+ macrophage": (0.08, 0.53, "proteolysis"),
    "PLAU / PLAT": (0.28, 0.58, "proteolysis"),
    "Plasmin": (0.45, 0.58, "proteolysis"),
    "MMP2 / 9 / 14": (0.28, 0.34, "proteolysis"),
    "ECM cleavage": (0.45, 0.34, "proteolysis"),
    "THBS-dependent TGF-beta activation": (0.55, 0.79, "matrix"),
    "Active TGF-beta": (0.68, 0.64, "retention"),
    "HSC retention": (0.81, 0.64, "retention"),
    "SPON1-rich matrix": (0.78, 0.87, "retention"),
    "Integrin alpha-v": (0.93, 0.87, "retention"),
    "SERPINE1 / PAI-1": (0.59, 0.42, "retention"),
    "TIMP1 / TIMP2": (0.73, 0.42, "retention"),
    "LOX / crosslinking": (0.84, 0.28, "mechanical"),
    "Stiff ECM": (0.93, 0.28, "mechanical"),
    "YAP / TEAD": (0.93, 0.09, "mechanical"),
    "AP-1 / RUNX": (0.80, 0.09, "mechanical"),
    "Retained matrix": (0.57, 0.12, "matrix"),
}

NODE_WIDTH = {
    "THBS-dependent TGF-beta activation": 0.16,
    "WNT9B+ endothelium": 0.14,
    "TREM2+ macrophage": 0.14,
    "SERPINE1 / PAI-1": 0.13,
    "LOX / crosslinking": 0.13,
    "SPON1-rich matrix": 0.13,
}

CURVATURE = {
    "E03": -0.08,
    "E05": 0.10,
    "E07": -0.08,
    "E08": 0.10,
    "E09": 0.08,
    "E10": -0.22,
    "E11": 0.10,
    "E12": -0.08,
    "E13": 0.18,
    "E14": -0.17,
    "E17": 0.07,
    "E20": -0.19,
    "E21": 0.08,
    "E22": 0.16,
    "E24": -0.10,
}


def draw_node(ax: plt.Axes, name: str, x: float, y: float, category: str) -> None:
    width = NODE_WIDTH.get(name, 0.115)
    height = 0.072
    style = NODE_STYLE[category]
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.007",
        facecolor=style["face"],
        edgecolor=style["edge"],
        linewidth=1.25,
        zorder=3,
    )
    ax.add_patch(box)
    label = name.replace("THBS-dependent TGF-beta activation", "THBS-dependent\nTGF-beta activation")
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        color=style["text"],
        fontsize=7.0 if "\n" in label else 7.4,
        fontweight="semibold",
        zorder=4,
    )


def draw_edge(ax: plt.Axes, row: pd.Series) -> None:
    source = NODES[row.source]
    target = NODES[row.target]
    style = EDGE_STYLE[row.evidence_class]
    arrowstyle = "-[" if row.relation == "inhibits" else "-|>"
    patch = FancyArrowPatch(
        (source[0], source[1]),
        (target[0], target[1]),
        arrowstyle=arrowstyle,
        mutation_scale=10.5,
        shrinkA=23,
        shrinkB=24,
        connectionstyle=f"arc3,rad={CURVATURE.get(row.edge_id, 0.0)}",
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        alpha=0.92,
        zorder=1,
    )
    ax.add_patch(patch)


def build(edges: pd.DataFrame, output: Path) -> None:
    unknown_nodes = (set(edges.source) | set(edges.target)) - set(NODES)
    if unknown_nodes:
        raise ValueError(f"Missing node coordinates: {sorted(unknown_nodes)}")
    unknown_classes = set(edges.evidence_class) - set(EDGE_STYLE)
    if unknown_classes:
        raise ValueError(f"Unknown evidence classes: {sorted(unknown_classes)}")

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(14.5, 8.3), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.axvspan(0.005, 0.50, ymin=0.055, ymax=0.94, color="#F3F9F9", zorder=0)
    ax.axvspan(0.50, 0.995, ymin=0.055, ymax=0.94, color="#FBF6F7", zorder=0)
    ax.text(0.025, 0.955, "RESOLVER ACCESS", color="#267B83", fontsize=11, fontweight="bold")
    ax.text(0.525, 0.955, "RETENTION REINFORCEMENT", color="#98384D", fontsize=11, fontweight="bold")

    for _, row in edges.iterrows():
        draw_edge(ax, row)
    for name, (x, y, category) in NODES.items():
        draw_node(ax, name, x, y, category)

    loop_specs = [
        ("1", 0.64, 0.90, "TGF-beta - SPON1 - alpha-v"),
        ("2", 0.79, 0.20, "LOX - stiffness - YAP/AP-1"),
        ("3", 0.52, 0.25, "protease restraint - matrix retention"),
    ]
    for number, x, y, label in loop_specs:
        ax.scatter([x], [y], s=165, color="#963A50", edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(x, y, number, ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=6)
        ax.text(x + 0.018, y, label, ha="left", va="center", color="#6E3947", fontsize=7.2, zorder=5)

    handles = [
        plt.Line2D([0], [0], color=EDGE_STYLE[name]["color"], linestyle=EDGE_STYLE[name]["linestyle"], linewidth=2, label=name)
        for name in ["literature-established", "project-supported association", "testable candidate edge"]
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.002),
        frameon=False,
        ncol=3,
        fontsize=8,
        handlelength=3.2,
        columnspacing=2.0,
    )
    ax.text(
        0.995,
        0.018,
        "arrow = activation   T-bar = inhibition",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#70777B",
    )
    ax.set_title(
        "Evidence-coded candidate circuit for HSC retention and matrix persistence",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges = pd.read_csv(args.edges, sep="\t")
    build(edges, args.output)
    print(f"Wrote {args.output}.svg/.pdf/.png from {len(edges)} evidence-coded edges")


if __name__ == "__main__":
    main()
