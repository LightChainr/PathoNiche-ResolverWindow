# Public release checklist

## Required before publication

- [ ] Every author approves authorship order, CRediT roles, licenses, repository contents, and public release.
- [ ] Continuing ethics coverage and the exact consent/waiver wording are confirmed for the full clinical-data period.
- [x] The repository owner/name replaces the GitHub placeholder in `CITATION.cff`.
- [x] The reserved Zenodo DOI replaces the DOI placeholder in `CITATION.cff`.
- [x] The same DOI is inserted into the manuscript's Data availability and Code availability statements.
- [ ] The final manuscript title and repository title match.
- [x] `python scripts/validate_release.py` exits successfully.
- [x] `SHA256SUMS.txt` and `FILE_MANIFEST.txt` are regenerated after the last change.
- [x] The release archive checksum is recorded outside the archive.
- [ ] A second author manually inspects `data/clinical_aggregate/` and `data/flow_aggregate/`.
- [x] The `release/v2.1.0-rc1` branch remains unmerged until the Zenodo record and manuscript metadata are synchronized.

## Never add

- Participant-level clinical or flow records.
- Exact dates linked to participants, dates of birth, names, medical-record numbers, identity-card numbers, addresses, telephone numbers, or hashes derived from identifiers.
- The verified clinical linkage key or flow-elastography linkage table.
- Ethics approval PDFs, consent forms, or author-private files.
- SSH credentials, server hostnames, proxy subscriptions, API keys, or local infrastructure notes.
- Public raw sequencing/imaging archives, LINCS Level 5 matrices, or model weights that should be downloaded from their original hosts.
- AI-generated graphical-abstract or mechanism concept images.
