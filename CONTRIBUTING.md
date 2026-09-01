# Contributing

This repository is an archival research companion. Please use an issue for reproducibility questions, suspected data-dictionary errors, or discrepancies between a released table and the manuscript.

Do not submit participant-level clinical data, dates linked to individuals, identifiers, ethics documents, credentials, unpublished institutional files, or third-party raw data through issues or pull requests.

Code contributions should preserve biological-unit handling, frozen module definitions, thresholds, and random seeds unless the change is explicitly labeled as a sensitivity analysis. Run `python scripts/validate_release.py` before proposing a change.
