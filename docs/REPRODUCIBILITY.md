# Reproducibility notes

## Release-level audit

Create a clean Python environment and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python scripts/validate_release.py
```

This validates the public package and its principal numerical anchors. It does not reconstruct raw public data or governed patient-level analyses.

## Reviewer-facing panel reconstruction

The four panels revised after peer review rebuild directly from included derived inputs:

```bash
python code/reviewer_revision/rebuild_high_risk_panels.py
```

Expected outputs are written to `figures/reviewer_revision/`. They replace an incompatible mixed evidence scale in Figure 3, a pseudo-replicated CODEX stage line in Figure 4, unestimated dynamics in Figure 6, and an overinterpreted order-score display in Figure 7.

The evidence-coded mechanism panel and editable main SVG plates rebuild with:

```bash
python code/figures/build_mechanism_circuit.py
python code/figures/rebuild_main_figure_plates.py
```

## Public mechanism workflow

The final analysis order was:

1. `UQ_Visium_resolver_window_analysis.py`
2. `uq_spatial_proteolytic_lock.py`
3. `uq_phase_basin_control.py`
4. `spatial_domain_wall_resolver_window.py`
5. `mouse_resolver_lock_trajectory.py`
6. `sequential_basin_switch_drugs.py`

The public-mechanism scripts are frozen provenance snapshots. Module definitions, thresholds, seeds, biological-unit handling, and statistical tests are retained. Some original workflow scripts require raw public archives that are intentionally excluded. Map those logical inputs to the following resources:

| Logical input | Public reconstruction source | Included shortcut |
|---|---|---|
| UQ Visium filtered matrices and spatial metadata | DOI `10.48610/e95155f` | Derived module/spot tables under `data/UQ_resolver_source_inputs/` |
| UQ S10 proteolytic-lock results | Output of step 2 | `data/UQ_spatial_derived/S10_UQ_spatial_proteolytic_lock/` |
| UQ S11 phase/basin results | Output of step 3 | `data/UQ_spatial_derived/S11_UQ_phase_basin_control/` |
| GSE295293 pseudobulk | GEO reconstruction | `data/mouse_pseudobulk_derived/` |
| GSE305331 pooled trajectories | GEO reconstruction | `data/mouse_pseudobulk_derived/state_module_trajectories.tsv` |
| LINCS hard-gated compound table | GSE92742/LINCS reconstruction | `data/LINCS_derived/LINCS_name_collapsed_hard_gated.tsv` |

Any change to module membership, thresholds, seeds, or analysis units should be declared as a sensitivity analysis rather than a reproduction.

## Chronology and exploratory status

The four-part framework was finalized after core analyses and is not represented as prospective preregistration. UQ component geometry and biopsy registration preceded score generation in the archived run, but strict analyst blinding is not documented. The order-aware pair script uses an asymmetric role-weighted formula and did not enforce the archived safety or computational hard gates during pair generation. These facts and their source timestamps are recorded under `docs/methods_audit/`.

## What can be verified immediately

The included JSON/TSV/Parquet files permit direct inspection of:

- cohort and longitudinal totals;
- clinical aggregate associations and validation metrics;
- spatial coupling, stage trends, threshold sensitivity, and biopsy summaries;
- two-component model selection and leave-one-patient-out stability;
- retention-domain topology and interface contraction;
- geometry-only biopsy reconstruction and mapping sensitivity;
- edge-level mechanism evidence classes and direct-test flags;
- replicated mouse withdrawal geometry;
- LINCS eligibility, ordered-pair scores, and weight sensitivity.

## Governed clinical workflow

`code/governed_clinical_transparency/` documents the institutional workflow and expected transformations. It references controlled inputs such as the verified clinical linkage table and the institutional hashing salt. Those inputs are intentionally absent. The scripts cannot be executed from this public repository and must not be modified to imply public availability of participant-level data.

## Public raw-data reconstruction

Large sequencing, imaging, LINCS, and model files are excluded. Obtain them from the original repositories listed in `RAW_DATA_ACCESSIONS.md`, retain their original identifiers, and cite the source studies. Source terms continue to apply.
