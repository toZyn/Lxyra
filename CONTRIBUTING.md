# Contributing

## Development environment

Use Python 3.11 or later. Install the package and development dependencies with `pip install -e ".[dev]"`.

Before opening a pull request, run:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python scripts/verify_models.py
python -m build
```

## Scope and review

Keep inference, model training, datasets and evaluation responsibilities separate. Introduce dependencies only when a documented need exists. Add tests for regression fixes and provide reproducible evidence for model quality changes. Do not claim that all interfaces are recognized accurately based on prepared screenshots.

Pull requests should describe the behavior changed, the validation performed, and any known limitations. Breaking schema changes require an explicit version increment and migration notes.

## Data and privacy

Do not commit downloaded datasets, user screenshots, credentials, local model checkpoints, private app records or system logs. Model binaries included in releases must have verified provenance and matching checksums.
