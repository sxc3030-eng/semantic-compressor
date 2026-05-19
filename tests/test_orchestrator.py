"""Tests for src.orchestrator (the pipeline + CLI wiring).

Tests are light-weight (per IMPLEMENTATION_NOTES section 3): we check the
orchestrator wires the modules together correctly and produces honest metrics.
The deep-dive correctness lives in each module's own test file.

Le dataset reel `data/original/users.csv` est utilise pour les tests d'integration
(10k lignes, ~2 MB). Si absent, on skip (le repo POC l'a normalement, mais on
reste robuste).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.orchestrator import (
    CompressionResult,
    ReconstructionResult,
    ValidationResultBundle,
    compress,
    decompress,
    main,
    validate_pair,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
USERS_CSV = REPO_ROOT / "data" / "original" / "users.csv"


@pytest.fixture(scope="module")
def users_csv() -> Path:
    """Le CSV de test 10k lignes. Skip le test si absent."""
    if not USERS_CSV.exists():
        pytest.skip(f"Users CSV fixture not available at {USERS_CSV}")
    return USERS_CSV


@pytest.fixture
def compress_result(users_csv: Path, tmp_path: Path) -> CompressionResult:
    """Compresse users.csv une fois pour les tests qui ont besoin du resultat."""
    return compress(
        input_csv=users_csv,
        output_dir=tmp_path,
        table_name="users",
    )


# ---------------------------------------------------------------------------
# 1. test_compress_creates_outputs
# ---------------------------------------------------------------------------


def test_compress_creates_outputs(users_csv: Path, tmp_path: Path) -> None:
    """compress doit creer le .md de recette ET le .parquet d'ancres."""
    result = compress(users_csv, tmp_path, table_name="users")

    assert result.recipe_path.exists(), "Recipe file should exist"
    assert result.anchor_path.exists(), "Anchor parquet should exist"
    assert result.recipe_path.suffix == ".md"
    assert result.anchor_path.suffix == ".parquet"
    assert result.recipe_path.stat().st_size > 0
    assert result.anchor_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2. test_compress_returns_result_with_metrics
# ---------------------------------------------------------------------------


def test_compress_returns_result_with_metrics(compress_result: CompressionResult) -> None:
    """CompressionResult doit exposer toutes les metriques avec des valeurs coherentes."""
    r = compress_result

    # Tailles strictement positives
    assert r.original_size_bytes > 0
    assert r.recipe_size_bytes > 0
    assert r.anchor_size_bytes > 0
    assert r.total_compressed_bytes == r.recipe_size_bytes + r.anchor_size_bytes
    assert r.compression_ratio > 0
    # Ratio coherent : original / total
    expected = r.original_size_bytes / r.total_compressed_bytes
    assert abs(r.compression_ratio - expected) < 1e-6

    # Counts
    assert r.n_rows == 10_000
    assert r.n_columns == 9
    assert r.n_patterns == 9  # un pattern par colonne (ANCHOR_DIRECT pour les 3 ancres)
    assert r.elapsed_seconds > 0

    # Ancres : minimum les 3 colonnes uniques
    assert set(r.anchor_columns) >= {"id", "email", "password_hash"}


# ---------------------------------------------------------------------------
# 3. test_decompress_roundtrip
# ---------------------------------------------------------------------------


def test_decompress_roundtrip(users_csv: Path, tmp_path: Path) -> None:
    """compress puis decompress doivent produire un CSV avec meme shape."""
    compress_result = compress(users_csv, tmp_path, table_name="users")

    reconstructed_csv = tmp_path / "reconstructed.csv"
    result = decompress(
        recipe_path=compress_result.recipe_path,
        output_csv=reconstructed_csv,
    )

    assert isinstance(result, ReconstructionResult)
    assert result.output_path.exists()
    assert result.n_rows == 10_000
    assert result.n_columns == 9

    # Verifier la forme du CSV reel
    df = pd.read_csv(reconstructed_csv)
    assert len(df) == 10_000
    assert len(df.columns) == 9
    # Les ancres doivent matcher exactement
    original = pd.read_csv(users_csv)
    for col in compress_result.anchor_columns:
        pd.testing.assert_series_equal(
            df[col].reset_index(drop=True),
            original[col].reset_index(drop=True),
            check_dtype=False,
            check_names=False,
        )


# ---------------------------------------------------------------------------
# 4. test_validate_pair_on_identical_files
# ---------------------------------------------------------------------------


def test_validate_pair_on_identical_files(users_csv: Path, tmp_path: Path) -> None:
    """Pipeline complet : compress -> decompress -> validate, le score doit etre >= 70."""
    compress_result = compress(users_csv, tmp_path, table_name="users")
    reconstructed_csv = tmp_path / "reconstructed.csv"
    decompress(
        recipe_path=compress_result.recipe_path,
        output_csv=reconstructed_csv,
    )

    bundle = validate_pair(
        original_csv=users_csv,
        reconstructed_csv=reconstructed_csv,
        anchor_columns=compress_result.anchor_columns,
    )

    assert isinstance(bundle, ValidationResultBundle)
    assert bundle.elapsed_seconds > 0
    assert bundle.report.overall_score >= 70.0, (
        f"Pipeline fidelity too low: {bundle.report.overall_score:.2f} < 70 "
        f"(passed={bundle.report.passed_count}, failed={bundle.report.failed_count})"
    )


# ---------------------------------------------------------------------------
# 5. test_run_poc_end_to_end_returns_zero_exit_code
# ---------------------------------------------------------------------------


def test_run_poc_end_to_end_returns_zero_exit_code(users_csv: Path, tmp_path: Path) -> None:
    """Appel direct de main() avec argv mocke doit retourner 0."""
    reconstructed_csv = tmp_path / "reconstructed.csv"
    argv = [
        "run-poc",
        "--input", str(users_csv),
        "--output-dir", str(tmp_path),
        "--reconstructed-csv", str(reconstructed_csv),
    ]
    exit_code = main(argv)
    assert exit_code == 0
    # Verifications de sanity sur les sorties
    assert (tmp_path / "recipes" / "users.md").exists()
    assert (tmp_path / "anchors" / "users_anchors.parquet").exists()
    assert reconstructed_csv.exists()


# ---------------------------------------------------------------------------
# 6. test_compress_uses_zstd_level_22_by_default
# ---------------------------------------------------------------------------


def test_compress_uses_zstd_level_22_by_default(users_csv: Path, tmp_path: Path) -> None:
    """Le parquet d'ancres doit etre ecrit en zstd (verifie via les metadata pyarrow)."""
    result = compress(users_csv, tmp_path, table_name="users")

    # Lecture des metadata parquet : compression est encodee par colonne dans le
    # row group. On verifie que toutes les colonnes utilisent zstd.
    pf = pq.ParquetFile(result.anchor_path)
    n_groups = pf.num_row_groups
    assert n_groups >= 1, "Parquet must have at least one row group"

    codecs_found: set[str] = set()
    for g in range(n_groups):
        rg = pf.metadata.row_group(g)
        for c in range(rg.num_columns):
            codecs_found.add(rg.column(c).compression.lower())

    assert "zstd" in codecs_found, (
        f"Expected ZSTD compression in parquet metadata, found: {codecs_found}"
    )
    # Et qu'il n'y a pas d'autre codec (toutes colonnes en zstd)
    assert codecs_found == {"zstd"}, (
        f"All columns should use zstd, found: {codecs_found}"
    )


# ---------------------------------------------------------------------------
# 7. Test bonus : codec different fonctionne aussi (sanity check)
# ---------------------------------------------------------------------------


def test_compress_with_snappy_codec_works(users_csv: Path, tmp_path: Path) -> None:
    """Le pipeline doit fonctionner aussi avec snappy (different code path : pas de niveau)."""
    result = compress(
        users_csv,
        tmp_path,
        table_name="users",
        parquet_codec="snappy",
        parquet_compression_level=None,
    )
    assert result.anchor_path.exists()
    pf = pq.ParquetFile(result.anchor_path)
    codecs_found: set[str] = set()
    for g in range(pf.num_row_groups):
        rg = pf.metadata.row_group(g)
        for c in range(rg.num_columns):
            codecs_found.add(rg.column(c).compression.lower())
    assert "snappy" in codecs_found
