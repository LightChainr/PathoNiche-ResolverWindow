# Repository and DOI release guide

## Recommended sequence

1. Push the contents of this directory to `release/v2.1.0-rc1` in `LightChainr/PathoNiche-ResolverWindow`.
2. Open a pull request against `main` and leave it unmerged while the new Zenodo version is prepared.
3. Run `python scripts/validate_release.py --rebuild` in a clean checkout of the release branch.
4. Complete `PUBLIC_RELEASE_CHECKLIST.md` with a second author.
5. Use the Zenodo new-version draft with reserved DOI `10.5281/zenodo.22138229`.
6. Verify the GitHub URL and DOI in `CITATION.cff` and the manuscript availability statements.
7. Regenerate `FILE_MANIFEST.txt`, `SHA256SUMS.txt`, and the ZIP archive.
8. Merge the reviewed release branch and tag the resulting commit `v2.1.0`.
9. Publish the Zenodo record after the GitHub and manuscript metadata agree.

## Suggested repository settings

- Default branch: `main`.
- Visibility before release: private.
- Issues: enabled for reproducibility questions.
- Wiki and Discussions: optional.
- Branch protection: require one approval for `main` after public release.
- Large-file storage: unnecessary for the current package; do not add raw public datasets.

## Final metadata to add

- Reserved Zenodo DOI.
- Final repository URL.
- Confirmed ORCID identifiers, when supplied by each author.
- Related preprint DOI and journal DOI when available.
- Final release date and version without the `rc` suffix.

## Release description

Use the opening paragraph of `README.md`, followed by the five-point scientific scope and the explicit exclusion statement. State that participant-level clinical data are available only through an institutionally governed access process, if such a process is formally established.
