# Release QA report

## Scope

Release `v2.1.0` contains privacy-screened aggregate institutional outputs, public-data-derived results, exact analytical audits, environment locks, executable workflows, and publication figures. Manuscript drafts, graphical-abstract concepts, private author or ethics material, participant-level clinical data, linkage files, credentials, and large public raw datasets are excluded.

## Structural validation

- 259 release files are covered by deterministic path, size, and SHA-256 manifests.
- All machine-readable TSV, CSV, JSON, and Parquet assets pass parsing and table-width checks.
- All Python scripts pass compilation and all R scripts pass parsing checks.
- Seven main figures are present in PNG and PDF; Figures 1, 2, 5, and 7 also include editable SVG plates.
- Supplementary Figures S1-S30 are present, including the UQ biopsy-mapping sensitivity plate.
- No cache or Finder residue is present.
- No release file exceeds the GitHub 100 MB limit.

## Privacy and security validation

- No participant-level clinical or flow table is present.
- No direct or transformed clinical identifier column was detected in proprietary aggregate tables.
- No email address, SSH endpoint, password, proxy subscription, private key, or credential string was detected in released content.
- No ethics PDF, author-private file, linkage table, hashing salt, manuscript DOCX, raw sequencing archive, LINCS Level 5 matrix, or model weight is present.
- Governed clinical scripts contain workflow definitions only; controlled inputs and secret values are absent.

## Numerical reconciliation

- Adult index cohort: 8,142.
- Longitudinal cohort: 772.
- UQ spatial analysis: 13,585 spots, 33 biopsy fragments, 32 patient clusters, and 589 spatial blocks.
- Coordinate-only reconstruction: 43 components, 13,539 direct assignments, and 46 nearest-component rescues.
- Largest retention-domain stage slope: +0.0960391.
- Resolver-retention interface stage slope: -0.0287332.
- Scenario sensitivity: both directions retained in 12/12 scenario-by-metric tests and 16/16 leave-one-array-out tests.
- One- versus two-component BIC gain: 141.0121859.
- Mouse withdrawal residual distance: 0.6983693.
- Order-aware forward and reverse scores: 0.8923123 and 0.5041989.
- Mechanism circuit: 14 literature-established, eight project-supported, and two candidate edges.

The working release and a separate clean copy each completed with `PASS: 46 checks passed; 0 failed` after reconstruction was enabled.

## Remaining publication holds

- Complete all-author review and governance confirmation.
- Replace repository and DOI placeholders in `CITATION.cff` and manuscript availability statements.
- Remove the `rc1` suffix when the release is frozen.
- Regenerate manifests and the archive after any final metadata change.
