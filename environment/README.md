# Reproducible environments

## Core Python environment

Python is constrained by `pyproject.toml` to 3.11. The installable top-level specification is `requirements-core.in`; `requirements-lock.txt` is generated with `uv pip compile` and contains the transitive lock used for blank-environment testing.

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python environment/requirements-lock.txt
```

`requirements.txt` remains as a compatibility entry point and contains only a valid requirement-file include. Python itself is intentionally declared in `pyproject.toml`, not as a pip requirement.

## R environment

The three pseudobulk workflows require R 4.6.0, `edgeR`, `limma`, `data.table`, `Matrix`, and `jsonlite`. Exact versions used on the local reconstruction host are recorded in `R-packages.lock.tsv`. A clean R-library test is required before release.

## Specialized perturbation environments

scGPT and CellOracle are separated from the core environment because they require model weights, motif resources, and platform-specific compiled dependencies. The archived RTX 5090 environment freeze is retained as `scgpt5090-pip-freeze.txt`. CellOracle is treated as Linux-only until its dedicated clean-environment test succeeds; absence from the core lock is deliberate.
