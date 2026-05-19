# Contributing to Semantic Compressor

Thanks for considering a contribution! This is a proof-of-concept project,
so the bar for new features is informal but quality matters: every change
should keep the test suite green and either improve the compression ratio,
improve fidelity, or expand the kinds of datasets the POC can handle.

## Development setup

Clone the repo and create a virtual environment.

PowerShell (Windows):

```powershell
git clone <repo-url> semantic-compressor
cd semantic-compressor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e ".[dev]"
```

Bash (Linux / macOS / Git Bash):

```bash
git clone <repo-url> semantic-compressor
cd semantic-compressor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Python 3.11 or later is required (the codebase uses PEP 604 union syntax
and `list[...]` / `dict[...]` generics directly).

## Running tests

Run the full suite:

```bash
pytest
```

Run a single module's tests:

```bash
pytest tests/test_pattern_detector.py
```

Run a single test by name pattern:

```bash
pytest -k "random_format"
```

The suite currently has 80 tests and takes ~70 s. All tests must pass
before opening a PR.

End-to-end smoke test on the bundled synthetic dataset:

```bash
python -m src.orchestrator run-poc --input data/original/users.csv
```

Expected: ratio around 6.24:1 (or 15.89:1 with `--aggressive-uuid`),
fidelity 100/100.

## Code style

- Python 3.11+ syntax (`str | None`, `list[str]`, not `Optional`, `List`).
- Type hints on all public function signatures.
- Docstrings on all public functions (1-3 lines, Google or simple style).
- Logging via the `logging` module with module-level loggers
  (`logger = logging.getLogger(__name__)`).
- French comments are OK where natural, but code identifiers stay in
  English.
- No emojis in code. Emojis are fine in markdown and user-facing CLI
  output (Rich panels) only when explicitly part of the design.
- Format with `ruff format` or a compatible tool. No enforcement yet, so
  match the surrounding style of the file you're editing.

## Adding a new pattern type

If you want to add a pattern type (for example `OFFSET_FROM_COLUMN` for
temporal correlations, or `LOG_NORMAL` for price distributions):

1. Add the variant to `PatternType` enum in `src/models.py`.
2. Update `Pattern._check_shape` validator if the pattern needs specific
   fields (e.g., `format_spec` for `RANDOM_FORMAT`, `source_column` and
   `offset_distribution` for `OFFSET_FROM_COLUMN`).
3. Add detection logic in `src/pattern_detector.py`. Plug it into
   `build_patterns()` at the right place in the pipeline.
4. Add reconstruction logic in `src/reconstructor.py`. Make sure it uses
   the per-row sha256 seed for deterministic output.
5. Add validation logic in `src/validator.py` if the new pattern needs a
   different test than exact match (e.g., regex + uniqueness for
   `RANDOM_FORMAT`).
6. Add tests. At minimum: detection round-trip (profile -> detect ->
   write recipe -> parse -> reconstruct -> validate).
7. Update `docs/IMPLEMENTATION_NOTES.md` under "Lessons learned" or
   "Future work" so the rationale is documented.
8. Add an entry to `CHANGELOG.md` under `[Unreleased]`.

## Testing on a new dataset

To validate the POC on your own data:

1. Place the CSV in `data/original/<name>.csv`.
2. Run:

   ```bash
   python -m src.orchestrator run-poc --input data/original/<name>.csv
   ```

3. Check the fidelity score and compression ratio in the Rich summary.
4. If fidelity is below 95, look at the failing tests in the Rich output.
   Common causes: undetected datetime columns (check profiler typing),
   over-tight KS threshold on skewed distributions, missing pattern type
   for the dataset's specific structure.
5. If patterns aren't detected as expected, open
   `output/recipes/<name>.md` and inspect the detected patterns directly.
   The Markdown is designed to be read by humans.
6. If the ratio is poor and the data has high-entropy columns (hashes,
   UUIDs, random tokens), try `--aggressive-uuid` or check whether a
   `RANDOM_FORMAT` entry should be added to `KNOWN_RANDOM_FORMATS` in
   `src/pattern_detector.py`.

## Reporting issues

Open an issue with:

- The dataset shape (rows, columns, types) and a minimal reproducer if
  possible.
- The command you ran.
- Expected vs actual behavior.
- The recipe file (`output/recipes/<name>.md`) when the issue is about
  pattern detection or reconstruction.

If the dataset is sensitive, a synthetic dataset that reproduces the
same shape and statistical profile is just as useful.

## Pull requests

- Keep commits focused. One feature, fix, or refactor per commit.
- Run `pytest` before pushing. PRs that break the test suite will not be
  reviewed.
- Update `CHANGELOG.md` under `[Unreleased]` with an `Added`, `Changed`,
  `Fixed`, `Removed`, or `Performance` entry.
- If the change affects measured numbers (compression ratio, fidelity,
  wall time), include the new numbers in the commit message body and
  in the changelog entry. Honest numbers, not best-of-N.
- If the change touches the recipe format on disk, bump the format
  contract section in `docs/IMPLEMENTATION_NOTES.md` and add a parser
  compatibility note.
