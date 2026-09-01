#!/usr/bin/env python3
"""Apply terminology-safe SVG edits and compose the evidence-coded Figure 7 plate."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES = REPO_ROOT / "figures/main"
DEFAULT_PANELS = REPO_ROOT / "figures/source_panels"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


TEXT_REPLACEMENTS = {
    "Figure_1_Clinical_Cohort_and_Persistence.svg": [
        ("one scan per hashed MRN", "one index scan per linked patient"),
        ("Other-department", "Department-based"),
        ("subcohort", "transport set"),
        (
            "Internal, temporal, and independent-subcohort validation",
            "Internal, temporal, and department-based transport assessment",
        ),
        (
            "Internal, temporal, and independent-transport set validation",
            "Internal, temporal, and department-based transport assessment",
        ),
    ],
    "Figure_2_Withdrawal_and_Resolver_Timing.svg": [
        ("Lock Proxy", "HSC-retention proxy"),
        ("Resolver Proxy", "Resolver proxy"),
    ],
    "Figure_5_Progressive_Retention_Domain_Occupation.svg": [
        ("Largest locked domain / biopsy area", "Largest retention-dominant domain / biopsy area"),
        ("F One locked domain becomes dominant", "F Retention domains coalesce"),
        ("F One retention-dominant domain becomes dominant", "F Retention domains coalesce"),
        ("Resolved–locked interface fraction", "Resolver–retention interface fraction"),
        ("locked domains", "retention domains"),
        ("Locked-posterior threshold", "Retention-component posterior threshold"),
        ("J Mean basin composition", "J Mean static-component composition"),
        ("K Patient-level occupancy distributions", "K Biopsy-level occupancy distributions"),
        ("K Biopsy-level occupancy distributions", "K Patient-cluster occupancy distributions"),
        ("Biopsy state fraction", "Patient-cluster state fraction"),
        ("locked=", "retention-dominant="),
        ("Resolved", "Resolver-accessible"),
        ("Locked", "Retention-dominant"),
        ("Transition", "Intermediate"),
        ("locked", "retention-dominant"),
        ("resolved", "resolver-accessible"),
        ("transition", "intermediate"),
    ],
}

SOURCE_PANEL_REPLACEMENTS = {
    "Fig7C_Dual_Axis_Gate.svg": [
        ("Phase-weighted basin-switch score", "Phase-weighted control score"),
    ],
}


def replace_fragment(node: ET.Element, old: str, new: str) -> None:
    if node.text:
        node.text = node.text.replace(old, new)
    if node.tail:
        node.tail = node.tail.replace(old, new)
    for child in node:
        replace_fragment(child, old, new)


def rewrite_svg(path: Path, replacements: list[tuple[str, str]]) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    for old, new in replacements:
        replace_fragment(root, old, new)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None


def retune_figure_5(path: Path) -> None:
    """Remove duplicate legends and space the retained three-state legend."""
    tree = ET.parse(path)
    root = tree.getroot()
    by_id = {node.get("id"): node for node in root.iter() if node.get("id")}

    for node in root.iter():
        if node.tag.endswith("text") and node.text and "becomes dominant" in node.text:
            node.text = "F  Retention domains coalesce"

    for legend_id in ("panel2_gGroup_06_2_legend_1", "panel2_gGroup_06_2_legend_2"):
        legend = by_id.get(legend_id)
        if legend is not None:
            parent = find_parent(root, legend)
            if parent is not None:
                parent.remove(legend)

    placements = {
        "panel1_use4442": 487.0,
        "panel1_text4443": 501.0,
        "panel1_use4443": 584.0,
        "panel1_text4444": 598.0,
        "panel1_use4444": 684.0,
        "panel1_text4445": 698.0,
    }
    for node_id, x_value in placements.items():
        node = by_id.get(node_id)
        if node is None:
            continue
        node.set("x", f"{x_value:g}")
        if node.tag.endswith("text"):
            y_value = float(node.get("y", "0"))
            node.set("transform", f"rotate(0 {x_value:g} {y_value:g})")
            style = node.get("style", "")
            node.set("style", re.sub(r"font-size:\s*7\.8px", "font-size:7.2px", style))

    tree.write(path, encoding="utf-8", xml_declaration=True)


def svg_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    raw = root.get("viewBox")
    if raw:
        values = [float(value) for value in re.split(r"[ ,]+", raw.strip())]
        if len(values) == 4:
            return tuple(values)  # type: ignore[return-value]
    width = float(re.sub(r"[^0-9.+-]", "", root.get("width", "1000")))
    height = float(re.sub(r"[^0-9.+-]", "", root.get("height", "1000")))
    return 0.0, 0.0, width, height


def prefix_ids(root: ET.Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for node in root.iter():
        old = node.get("id")
        if old:
            new = f"{prefix}_{old}"
            id_map[old] = new
            node.set("id", new)
    if not id_map:
        return
    pattern = re.compile(r"#([A-Za-z_][A-Za-z0-9_.:-]*)")
    for node in root.iter():
        for attribute, value in list(node.attrib.items()):
            if "#" in value:
                node.set(attribute, pattern.sub(lambda match: "#" + id_map.get(match.group(1), match.group(1)), value))
        if node.tag.endswith("style") and node.text and "#" in node.text:
            node.text = pattern.sub(lambda match: "#" + id_map.get(match.group(1), match.group(1)), node.text)


def add_nested_panel(
    canvas: ET.Element,
    path: Path,
    label: str,
    cell: tuple[float, float, float, float],
    prefix: str,
) -> None:
    x, y, width, height = cell
    source = ET.parse(path).getroot()
    prefix_ids(source, prefix)
    viewbox = svg_viewbox(source)
    nested = ET.SubElement(
        canvas,
        f"{{{SVG_NS}}}svg",
        {
            "x": f"{x:g}",
            "y": f"{y:g}",
            "width": f"{width:g}",
            "height": f"{height:g}",
            "viewBox": " ".join(f"{value:g}" for value in viewbox),
            "preserveAspectRatio": "xMidYMid meet",
        },
    )
    for child in list(source):
        nested.append(copy.deepcopy(child))
    label_node = ET.SubElement(
        canvas,
        f"{{{SVG_NS}}}text",
        {
            "x": f"{max(8, x - 45):g}",
            "y": f"{y + 45:g}",
            "style": "font-family:Arial,Helvetica,sans-serif;font-size:42px;font-weight:700;fill:#111111;letter-spacing:0",
        },
    )
    label_node.text = label


def compose_figure_7(panels: Path, output: Path) -> None:
    width, height = 3600, 2100
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "version": "1.1",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {"x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "#ffffff"},
    )
    specs = [
        ("Fig7A_Evidence_Coded_Mechanism.svg", "A", (60, 30, 1720, 1050)),
        ("Fig7B_Perturbation_Triangulation.svg", "B", (1820, 30, 1720, 1050)),
        ("Fig7C_Dual_Axis_Gate.svg", "C", (60, 1120, 1720, 940)),
        ("Fig7D_OrderAware_Score_Audit.svg", "D", (1820, 1220, 1720, 640)),
    ]
    for index, (filename, label, cell) in enumerate(specs, start=1):
        add_nested_panel(root, panels / filename, label, cell, f"fig7_{index}")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def render(svg: Path) -> None:
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(svg.with_suffix(".pdf")), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-w", "3200", "-o", str(svg.with_suffix(".png")), str(svg)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--source-panels", type=Path, default=DEFAULT_PANELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for filename, replacements in SOURCE_PANEL_REPLACEMENTS.items():
        path = args.source_panels / filename
        if not path.exists():
            raise FileNotFoundError(path)
        rewrite_svg(path, replacements)
    for filename, replacements in TEXT_REPLACEMENTS.items():
        path = args.figures_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        rewrite_svg(path, replacements)
        if filename == "Figure_5_Progressive_Retention_Domain_Occupation.svg":
            retune_figure_5(path)
        render(path)
    figure_7 = args.figures_dir / "Figure_7_Mechanism_and_OrderAware_Nomination.svg"
    compose_figure_7(args.source_panels, figure_7)
    render(figure_7)
    print("Rebuilt terminology-safe Figures 1, 2, 5 and evidence-coded Figure 7")


if __name__ == "__main__":
    main()
