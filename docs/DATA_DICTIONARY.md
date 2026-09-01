# Data dictionary

## Clinical aggregate tables

- `Table1_baseline_8142.tsv`: baseline variables, overall summaries, stiffness-stratum summaries, and standardized mean differences.
- `clinical_validation_metrics.tsv`: derivation, temporal-validation, and department-based transport AUROC and calibration metrics.
- `longitudinal_state_flow.tsv`: counts and percentages for low/high LSM state transitions among 772 longitudinal patients.
- `longitudinal_outcomes.tsv`: aggregate persistence and regression categories.
- `serum_imbalance_associations.tsv`: odds ratios, confidence intervals, P values, and AUROC estimates for the frozen serum proxy and component axes.
- `clinical_analysis_summary.json`: frozen cohort totals, thresholds, and summary estimates used in the manuscript.
- `legacy_compatible/`: aggregate model and donor-burden source tables retained for supplementary figures.

## Flow-cytometry aggregate tables

- `flow_analysis_summary.json`: cohort-level panel and patient counts plus the declared manuscript role.
- `flow_context_counts.tsv`: panels and patients by broad clinical context; minimum released group size is 10 patients.
- `flow_context_models.tsv`: aggregate effect estimates for immune features by context.
- `flow_immune_pca_loadings.tsv`: principal-component loadings for the six-marker immune panel.

The individual flow-elastography linkage table is intentionally absent.

## UQ spatial data

Files in `UQ_resolver_source_inputs` contain public-resource spot, biopsy, module, and sensitivity summaries generated before the final S10-S14 analyses.

Files in `UQ_spatial_derived` contain:

- S10: proteolytic-lock module scores, detection audits, biopsy coupling, and cross-module summaries.
- S11: repair-retention axes, static Gaussian-mixture model selection, patient/bootstrap sensitivity, spatial blocks, and component assignments.
- S12: order-aware LINCS pair scores and weighting sensitivity.
- S13: mouse module mapping, phase scores, exact tests, and trajectory geometry.
- S14: retention-domain topology, equal-posterior-boundary profiles, boundary effects, threshold sensitivity, and spot annotations.

## Figure-reconstruction inputs

- `integrated_retention_candidates.tsv`: categorical HSC evidence fields used in revised Figure 3.
- `LAM_priority_gene_fates.tsv`: macrophage repair-gene fate fields used in revised Figure 3.
- `UQ_CODEX_scar_enrichment.tsv`: three section-specific CODEX enrichment estimates used in revised Figure 4.
- The static GMM, mouse, and order-aware replacement panels read the corresponding S11-S13 files above.

## Mechanism evidence inputs

- `data/mechanism_summary/mechanism_edge_evidence.tsv`: one row per displayed circuit edge, with relationship direction, evidence class, project source, literature reference or DOI, and a direct-test flag.
- `figures/source_panels/Fig7*.svg`: editable Figure 7 source panels used by the main-plate compositor.

## Methods-audit files

`docs/methods_audit/` contains the dataset master table, GSE306327 sample notes, UQ reconstruction audit, drug-score specification, dated analysis timeline, panel-statistics workbook, and machine-readable supporting tables. Audit labels distinguish documented facts, chronology-based inference, and information not documented in the archived workspace.

## Common statistical fields

- `estimate`, `effect`, `slope`, `coef`: estimated effect in the scale documented by the file or generating script.
- `se`: standard error.
- `ci_low`, `ci_high`: confidence-interval bounds.
- `p`, `p_value`, `exact_*_p`: nominal or exact P value.
- `q`, `fdr`: multiplicity-adjusted P value where calculated.
- `n`, `n_patients`, `n_biopsies`, `n_observations`: biological or observational unit count.
- `stage`: fibrosis stage as encoded by the public source resource.
- `patient`, `biopsy`, `array`, `barcode`: public-resource identifiers, not proprietary clinical identifiers.

Exact formulas, module membership, seeds, and file-level assumptions are preserved in the corresponding scripts.

## Figure files

- `figures/main/Figure_*.png` and `Figure_*.pdf`: seven revised composite main figures.
- `figures/main/Figure_{1,2,5,7}_*.svg`: editable main-figure plates.
- `figures/reviewer_revision/Fig*.{png,pdf,svg}`: four corrected analytical panels and source manifest.
- `figures/supplementary/Supplementary_Figure_S1.png` through `Supplementary_Figure_S30_UQ_Mapping_Sensitivity.png`: complete supplementary figure atlas.
- `FIGURE_SOURCE_MAP.md`: links each main display to the released source-data families and states its reproduction boundary.
