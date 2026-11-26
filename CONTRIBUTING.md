# Contributing

Issues and pull requests are welcome.

```bash
git clone https://github.com/albertlieyingadrian/behavior-diff && cd behavior-diff
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy src && pytest
```

- The tests run on CPU in a few seconds, with a fake judge and saved data. No test may call the network.
- The package itself uses the standard library only; backends import their SDKs lazily (`tinker`, `modal`,
  `transformers`), behind the extras in `pyproject.toml`.
- Docs live in `docs/` (Mintlify). Preview with `npm i -g mint && cd docs && mint dev`; check with `mint validate`
  and `mint broken-links`.
- Releases: bump `version` in `pyproject.toml`, add a CHANGELOG entry, then publish a GitHub release tagged
  `v<version>`. The release workflow uploads to PyPI with trusted publishing.
