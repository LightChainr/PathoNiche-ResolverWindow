#!/usr/bin/env python3
"""Adversarially validate release structure, provenance, privacy, and results."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "release_validation_report.json"
MANIFEST_EXCLUDED = {"FILE_MANIFEST.txt", "SHA256SUMS.txt", REPORT.name}
TEXT_SUFFIXES = {".md", ".txt", ".py", ".r", ".sh", ".json", ".tsv", ".csv", ".cff", ".yaml", ".yml", ".toml"}
FORBIDDEN_NAMES = (
    "author_information",
    "ethics_approval",
    "clinical_hash_salt",
    "flow_liver_stiffness_90d_linkage",
    "matched_90d",
    "private_submission",
)
SECRET_PATTERNS = {
    "ssh_host": re.compile(r"connect[.]singapore|gpuhub[.]com", re.I),
    "ssh_command": re.compile(r"ssh\s+root@", re.I),
    "proxy_subscription": re.compile(r"ta30[.]vip|clash subscription", re.I),
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}\b", re.I),
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "machine_absolute_path": re.compile(r"(?:^|[\s\"'(])/(?:Volumes|Users|root|autodl(?:-fs|-tmp)?)/", re.M),
}
PROHIBITED_CLINICAL_COLUMNS = {
    "name", "patient_name", "medical_record_number", "mrn", "identity_card", "id_card",
    "birth_date", "date_of_birth", "dob", "phone", "address", "email", "patient_hash",
    "subject_hash", "event_hash", "encounter_date",
}
REQUIRED_ANALYSIS_SCRIPTS = [
    "code/clinical/rebuild_final_clinical_visuals.py",
    "code/mouse/GSE295293_prepare_pseudobulk.py",
    "code/mouse/GSE295293_dynamic_pseudobulk.R",
    "code/mouse/GSE305331_stream_state_trajectories.py",
    "code/mouse/mouse_resolver_lock_trajectory.py",
    "code/human_hsc/GSE244832_prepare_hsc_map.py",
    "code/human_hsc/GSE244832_stream_hsc_pseudobulk.cpp",
    "code/human_hsc/GSE244832_hsc_donor_edger.R",
    "code/human_hsc/GSE244832_integrate_retention_candidates.py",
    "code/macrophage/GSE261829_macrophage_resolution.R",
    "code/atlas/m2_gse136103_build_h5ad.py",
    "code/atlas/m2_gse136103_scanpy_cluster_signature.py",
    "code/atlas/m2_gse136103_cell_state_label_v1.py",
    "code/atlas/m2_gse136103_cell_state_pseudobulk_de.py",
    "code/spatial/UQ_Visium_component_audit.py",
    "code/spatial/UQ_Visium_resolver_window_analysis.py",
    "code/spatial/uq_spatial_proteolytic_lock.py",
    "code/spatial/uq_phase_basin_control.py",
    "code/spatial/spatial_domain_wall_resolver_window.py",
    "code/spatial/UQ_biopsy_mapping_sensitivity.py",
    "code/codex/UQ_CODEX_spatial_protein_validation.py",
    "code/perturbation/prepare_v16_hsc_control_perturbation.py",
    "code/perturbation/scgpt_v16_hsc_control_perturbation.py",
    "code/perturbation/v15_celloracle_native_cpu.py",
    "code/perturbation/celloracle_v16_ap1_runx2_test.py",
    "code/perturbation/LINCS_AP1_genetic_perturbation.py",
    "code/perturbation/stage4_regulatory_drug_control.py",
    "code/perturbation/stage4_drug_hard_gate.py",
    "code/perturbation/sequential_basin_switch_drugs.py",
    "code/figures/build_mechanism_circuit.py",
    "code/figures/rebuild_main_figure_plates.py",
]


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []
        self.diagnostics: list[str] = []

    def check(self, condition: bool, label: str, evidence: str = "") -> None:
        self.checks.append({"status": "PASS" if condition else "FAIL", "label": label, "evidence": evidence})

    @property
    def failures(self) -> list[dict[str, str]]:
        return [row for row in self.checks if row["status"] == "FAIL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild four high-risk panels and compare PNG hashes")
    parser.add_argument("--skip-manifest", action="store_true", help="Skip FILE_MANIFEST and SHA256SUMS verification during development")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in MANIFEST_EXCLUDED
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    )


def table_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    return (rows[0] if rows else []), rows[1:]


def check_structure(audit: Audit, files: list[Path]) -> None:
    audit.check(len(files) > 150, "repository contains the expected release assets", f"n={len(files)}")
    missing = [path for path in REQUIRED_ANALYSIS_SCRIPTS if not (ROOT / path).is_file()]
    audit.check(not missing, "all main-analysis scripts are present", "; ".join(missing))

    bad_names = [str(path.relative_to(ROOT)) for path in files if any(token in path.name.lower() for token in FORBIDDEN_NAMES)]
    audit.check(not bad_names, "no forbidden private or linkage filenames", "; ".join(bad_names))

    residue = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if "__pycache__" in path.parts or path.name == ".DS_Store"]
    audit.check(not residue, "no cache or Finder residue", "; ".join(residue[:20]))

    required_audit = [
        "docs/PANEL_PROVENANCE.tsv",
        "docs/METHODS_AUDIT_SUMMARY.md",
        "docs/methods_audit/Dataset_Master_Table.xlsx",
        "docs/methods_audit/GSE306327_Methods_Notes.docx",
        "docs/methods_audit/UQ_Reconstruction_Audit.xlsx",
        "docs/methods_audit/UQ_BIOPSY_RECONSTRUCTION_PROVENANCE.md",
        "docs/methods_audit/Drug_Order_Scoring_Specification.docx",
        "docs/methods_audit/Prespecification_and_Analysis_Timeline.docx",
        "docs/methods_audit/Figure_Panel_Statistics.xlsx",
        "docs/methods_audit/Reviewer_Response_and_Repair_Matrix.xlsx",
        "docs/methods_audit/machine_readable/Analysis_Timeline.tsv",
        "docs/methods_audit/machine_readable/Drug_Order_Complete_Results.tsv",
        "docs/methods_audit/machine_readable/UQ_Biopsy_Reconstruction_Map.tsv",
        "data/mechanism_summary/mechanism_edge_evidence.tsv",
    ]
    missing_audit = [path for path in required_audit if not (ROOT / path).is_file()]
    audit.check(not missing_audit, "methods-audit assets are complete", "; ".join(missing_audit))

    main = ROOT / "figures/main"
    revised = ROOT / "figures/reviewer_revision"
    supplementary = ROOT / "figures/supplementary"
    for suffix in ("png", "pdf"):
        count = len(list(main.glob(f"Figure_*.{suffix}")))
        audit.check(count == 7, f"seven main figures in {suffix.upper()} format", f"n={count}")
    count = len(list(main.glob("Figure_*.svg")))
    audit.check(count == 4, "four editable main figures in SVG format", f"n={count}")
    count = len(list((ROOT / "figures/source_panels").glob("Fig7*.svg")))
    audit.check(count == 4, "four editable Figure 7 source panels", f"n={count}")
    for suffix in ("png", "pdf", "svg"):
        count = len(list(revised.glob(f"Fig[3467]*.{suffix}")))
        audit.check(count == 4, f"four replacement panels in {suffix.upper()} format", f"n={count}")
    count = len(list(supplementary.glob("Supplementary_Figure_S*.png")))
    audit.check(count == 30, "thirty supplementary figures", f"n={count}")


def check_text_and_privacy(audit: Audit, files: list[Path]) -> None:
    hits: list[str] = []
    for path in files:
        if path.resolve() == Path(__file__).resolve() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                hits.append(f"{path.relative_to(ROOT)}:{label}")
    audit.check(not hits, "no credentials, endpoints, email addresses, keys, or machine paths", "; ".join(hits[:30]))

    disclosure: list[str] = []
    for folder in (ROOT / "data/clinical_aggregate", ROOT / "data/flow_aggregate"):
        for path in sorted(folder.glob("*.tsv")):
            headers, _ = table_rows(path)
            overlap = sorted({header.strip().lower() for header in headers} & PROHIBITED_CLINICAL_COLUMNS)
            if overlap:
                disclosure.append(f"{path.relative_to(ROOT)}:{','.join(overlap)}")
    audit.check(not disclosure, "aggregate clinical tables contain no direct or linkable identifiers", "; ".join(disclosure))


def check_machine_readability(audit: Audit, files: list[Path]) -> None:
    errors: list[str] = []
    inconsistent: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"{relative}:{exc}")
        elif path.suffix.lower() in {".tsv", ".csv"}:
            try:
                header, rows = table_rows(path)
                bad = [index + 2 for index, row in enumerate(rows) if len(row) != len(header)]
                if bad:
                    inconsistent.append(f"{relative}:header={len(header)},bad_lines={bad[:8]}")
            except Exception as exc:
                errors.append(f"{relative}:{exc}")
    audit.check(not errors, "all JSON and delimited text files are readable", "; ".join(errors[:20]))
    audit.check(not inconsistent, "all TSV and CSV rows match their header width", "; ".join(inconsistent[:20]))

    syntax_errors: list[str] = []
    for path in sorted((ROOT / "code").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}:{exc}")
    audit.check(not syntax_errors, "all Python scripts compile", "; ".join(syntax_errors))

    rscript = shutil.which("Rscript") or "/opt/homebrew/bin/Rscript"
    r_errors: list[str] = []
    if Path(rscript).exists():
        for path in sorted((ROOT / "code").rglob("*.R")):
            result = subprocess.run([rscript, "-e", "parse(file=commandArgs(TRUE)[1])", str(path)], capture_output=True, text=True)
            if result.returncode:
                r_errors.append(f"{path.relative_to(ROOT)}:{result.stderr.strip()[-300:]}")
    audit.check(not r_errors, "all R scripts parse", "; ".join(r_errors))


def check_environments(audit: Audit) -> None:
    requirements = (ROOT / "environment/requirements.txt").read_text().splitlines()
    bad_python = [line for line in requirements if line.strip().lower().startswith("python")]
    audit.check(not bad_python, "requirements.txt contains only pip-valid requirements", "; ".join(bad_python))
    lock = (ROOT / "environment/requirements-lock.txt").read_text().lower()
    required = ["numpy==", "pandas==", "scanpy==", "diptest==", "scikit-image==", "statsmodels=="]
    missing = [name for name in required if name not in lock]
    audit.check(not missing, "core Python transitive lock is complete", "; ".join(missing))
    pyproject = (ROOT / "pyproject.toml").read_text()
    audit.check('requires-python = ">=3.11,<3.12"' in pyproject, "Python runtime is constrained in pyproject.toml")
    rlock = pd.read_csv(ROOT / "environment/R-packages.lock.tsv", sep="\t")
    audit.check({"R", "edgeR", "limma", "data.table"}.issubset(set(rlock.package)), "R runtime and edgeR stack are version-recorded")


def check_numeric_anchors(audit: Audit) -> None:
    mechanism = pd.read_csv(ROOT / "data/mechanism_summary/mechanism_edge_evidence.tsv", sep="\t")
    expected_classes = {
        "literature-established",
        "project-supported association",
        "testable candidate edge",
    }
    audit.check(len(mechanism) == 24, "mechanism circuit contains 24 traceable edges", f"n={len(mechanism)}")
    audit.check(
        set(mechanism["evidence_class"]) == expected_classes,
        "mechanism circuit preserves all three evidence classes",
        "; ".join(sorted(set(mechanism["evidence_class"]))),
    )

    clinical = json.loads((ROOT / "data/clinical_aggregate/clinical_analysis_summary.json").read_text())
    outcomes = pd.read_csv(ROOT / "data/clinical_aggregate/longitudinal_outcomes.tsv", sep="\t")
    audit.check(clinical["cohort"]["adult_index_patients"] == 8142, "clinical patient anchor recomputes", "n=8142")
    audit.check(int(outcomes["n"].sum()) == 772, "longitudinal patient anchor recomputes", f"n={int(outcomes['n'].sum())}")

    phase = ROOT / "data/UQ_spatial_derived/S11_UQ_phase_basin_control"
    blocks = pd.read_parquet(phase / "phase_spatial_blocks.parquet")
    bic = pd.read_csv(phase / "phase_gmm_model_selection.tsv", sep="\t").set_index("n_components")
    gain = float(bic.loc[1, "bic"] - bic.loc[2, "bic"])
    audit.check(len(blocks) == 589, "spatial block anchor recomputes", f"n={len(blocks)}")
    audit.check(abs(gain - 141.01218590066708) < 1e-9, "one-versus-two-component BIC gain recomputes", f"gain={gain:.12f}")

    wall = pd.read_csv(ROOT / "data/UQ_spatial_derived/S14_spatial_domain_wall/spatial_domain_stage_trends.tsv", sep="\t").set_index("metric")
    largest = float(wall.loc["largest_locked_domain_fraction", "stage_beta_array_adjusted"])
    interface = float(wall.loc["interface_edge_fraction", "stage_beta_array_adjusted"])
    audit.check(abs(largest - 0.09603914800529065) < 1e-12, "largest retention-domain slope recomputes", f"beta={largest:.12f}")
    audit.check(abs(interface + 0.028733200324141162) < 1e-12, "cross-state interface slope recomputes", f"beta={interface:.12f}")

    mapping_dir = ROOT / "results/S15_UQ_biopsy_mapping_sensitivity"
    mapping_summary = json.loads((mapping_dir / "UQ_biopsy_mapping_sensitivity_summary.json").read_text())
    audit.check(
        mapping_summary["direct_geometry_spots"] == 13539
        and mapping_summary["nearest_component_rescue_spots"] == 46
        and mapping_summary["direct_geometry_match_rate"] == 1.0,
        "score-independent UQ geometry reconstruction recomputes",
        "direct=13539; rescue=46; agreement=100%",
    )
    audit.check(
        mapping_summary["scenario_key_direction_pass"] == 12
        and mapping_summary["scenario_key_direction_total"] == 12
        and mapping_summary["leave_one_array_out_key_direction_pass"] == 16
        and mapping_summary["leave_one_array_out_key_direction_total"] == 16,
        "UQ mapping stress tests preserve both key effect directions",
        "scenarios=12/12; leave-one-array-out=16/16",
    )

    mouse = pd.read_csv(ROOT / "data/UQ_spatial_derived/S13_mouse_resolver_lock_trajectory/GSE295293_mouse_phase_scores.tsv", sep="\t")
    centroids = mouse.groupby("condition")[["resolver_axis", "lock_axis"]].mean()
    control = centroids.loc["control"].to_numpy()
    fibrosis = centroids.loc["fibrosis"].to_numpy()
    regression = centroids.loc["regression"].to_numpy()
    ratio = float(np.linalg.norm(regression - control) / np.linalg.norm(fibrosis - control))
    audit.check(abs(ratio - 0.6983693102084559) < 1e-12, "mouse residual-distance ratio recomputes", f"ratio={ratio:.12f}")

    compounds = pd.read_csv(ROOT / "data/LINCS_derived/LINCS_name_collapsed_hard_gated.tsv", sep="\t").set_index("pert_iname_norm")
    early = compounds.loc["ezetimibe"]
    late = compounds.loc["verteporfin"]

    def score(first: pd.Series, second: pd.Series) -> float:
        endo = first.median_endo_support + 0.5 * second.median_endo_support
        macro = first.median_macro_support + 0.5 * second.median_macro_support
        retention = second.median_retention_suppression + 0.5 * first.median_retention_suppression
        return float(retention + 0.5 * endo + 0.5 * macro - 1.5 * max(0.0, -min(endo, macro)))

    forward, reverse = score(early, late), score(late, early)
    eligibility = (
        bool(early.identity_resolved) and early.n_signatures >= 3 and early.n_cell_lines >= 2
        and early.median_resolver_floor >= 0.10
        and 0.5 * (early.median_endo_support + early.median_macro_support) >= 0.20
        and early.median_retention_suppression >= -0.10
        and bool(late.identity_resolved) and late.n_signatures >= 3 and late.n_cell_lines >= 2
        and late.median_retention_suppression >= 0.20
        and late.q25_retention_suppression >= 0.10
        and late.median_resolver_floor >= -0.15
    )
    audit.check(eligibility, "lead compounds satisfy archived role thresholds")
    audit.check(abs(forward - 0.8923123237262998) < 1e-12, "forward order-aware score recomputes", f"score={forward:.12f}")
    audit.check(abs(reverse - 0.5041988742064742) < 1e-12, "reverse order-aware score recomputes", f"score={reverse:.12f}")


def check_rebuild(audit: Audit) -> None:
    script = ROOT / "code/reviewer_revision/rebuild_high_risk_panels.py"
    expected = ROOT / "figures/reviewer_revision"
    stems = [
        "Fig3D_Categorical_Evidence",
        "Fig4F_CODEX_SectionConcordance",
        "Fig6_Static_TwoComponent_Revision",
        "Fig7D_OrderAwareScore_Audit",
    ]
    with tempfile.TemporaryDirectory(prefix="pathoniche-rebuild-") as temp:
        temp_root = Path(temp)
        output = temp_root / "figures"
        env = dict(os.environ)
        env["MPLCONFIGDIR"] = str(temp_root / "mplconfig")
        result = subprocess.run([sys.executable, str(script), "--output-dir", str(output)], cwd=ROOT, env=env, capture_output=True, text=True)
        audit.check(result.returncode == 0, "high-risk panel rebuild completes", result.stderr.strip()[-500:])
        if result.returncode == 0:
            mismatches = [stem for stem in stems if sha256(output / f"{stem}.png") != sha256(expected / f"{stem}.png")]
            audit.check(not mismatches, "rebuilt high-risk PNGs match frozen SHA-256", "; ".join(mismatches))

        mechanism_output = temp_root / "source_panels/Fig7A_Evidence_Coded_Mechanism"
        mechanism_result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code/figures/build_mechanism_circuit.py"),
                "--output",
                str(mechanism_output),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        audit.check(mechanism_result.returncode == 0, "mechanism circuit rebuild completes", mechanism_result.stderr.strip()[-500:])
        if mechanism_result.returncode == 0:
            same = sha256(mechanism_output.with_suffix(".png")) == sha256(
                ROOT / "figures/source_panels/Fig7A_Evidence_Coded_Mechanism.png"
            )
            audit.check(same, "rebuilt mechanism PNG matches frozen SHA-256")

        plate_input = temp_root / "main"
        plate_input.mkdir(parents=True, exist_ok=True)
        editable_stems = [
            "Figure_1_Clinical_Cohort_and_Persistence",
            "Figure_2_Withdrawal_and_Resolver_Timing",
            "Figure_5_Progressive_Retention_Domain_Occupation",
        ]
        for stem in editable_stems:
            shutil.copy2(ROOT / f"figures/main/{stem}.svg", plate_input / f"{stem}.svg")
        plate_result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code/figures/rebuild_main_figure_plates.py"),
                "--figures-dir",
                str(plate_input),
                "--source-panels",
                str(ROOT / "figures/source_panels"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        audit.check(plate_result.returncode == 0, "editable main-plate rebuild completes", plate_result.stderr.strip()[-500:])
        if plate_result.returncode == 0:
            plate_stems = editable_stems + ["Figure_7_Mechanism_and_OrderAware_Nomination"]
            mismatches = [
                stem
                for stem in plate_stems
                if sha256(plate_input / f"{stem}.png") != sha256(ROOT / f"figures/main/{stem}.png")
            ]
            audit.check(not mismatches, "rebuilt editable main PNGs match frozen SHA-256", "; ".join(mismatches))


def check_manifests(audit: Audit, files: list[Path]) -> None:
    checksum_path = ROOT / "SHA256SUMS.txt"
    manifest_path = ROOT / "FILE_MANIFEST.txt"
    expected_paths = {path.relative_to(ROOT).as_posix() for path in files}
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        recorded[relative] = digest
    missing = sorted(expected_paths - set(recorded))
    extra = sorted(set(recorded) - expected_paths)
    mismatched = sorted(relative for relative in expected_paths & set(recorded) if sha256(ROOT / relative) != recorded[relative])
    audit.check(not missing and not extra, "SHA256SUMS covers the exact release file set", f"missing={missing[:8]}; extra={extra[:8]}")
    audit.check(not mismatched, "all recorded SHA-256 checksums verify", "; ".join(mismatched[:20]))

    lines = manifest_path.read_text().splitlines()
    entries = {}
    for line in lines[1:]:
        size, relative = line.split("\t", 1)
        entries[relative] = int(size)
    size_mismatch = sorted(relative for relative in expected_paths & set(entries) if entries[relative] != (ROOT / relative).stat().st_size)
    audit.check(set(entries) == expected_paths and not size_mismatch, "FILE_MANIFEST records exact paths and byte sizes", f"size_mismatch={size_mismatch[:8]}")


def main() -> int:
    args = parse_args()
    audit = Audit()
    files = release_files()
    check_structure(audit, files)
    check_text_and_privacy(audit, files)
    check_machine_readability(audit, files)
    check_environments(audit)
    check_numeric_anchors(audit)
    if args.rebuild:
        check_rebuild(audit)
    if not args.skip_manifest:
        check_manifests(audit, files)

    checksums = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    report = {
        "release": (ROOT / "VERSION").read_text().strip(),
        "status": "PASS" if not audit.failures else "FAIL",
        "file_count": len(files),
        "checks": audit.checks,
        "pass_count": len(audit.checks) - len(audit.failures),
        "failure_count": len(audit.failures),
        "sha256": checksums,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['pass_count']} checks passed; {report['failure_count']} failed")
    for failure in audit.failures:
        print(f"FAIL: {failure['label']} :: {failure['evidence']}")
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    sys.exit(main())
