"""Tests pour `src/reconstructor.py`.

Couvre les 10 cas obligatoires :
    1. test_derive_row_seed_deterministic
    2. test_derive_row_seed_int_in_32bits
    3. test_parse_recipe_extracts_all_sections
    4. test_reconstruct_anchor_only_recipe
    5. test_reconstruct_normal_distribution
    6. test_reconstruct_categorical_distribution
    7. test_reconstruct_functional_dependency
    8. test_reconstruction_reproducibility
    9. test_end_to_end_users_csv
   10. test_full_pipeline_validation_score_high
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import (
    ColumnProfile,
    ColumnType,
    DistributionType,
    Pattern,
    PatternType,
    Recipe,
    RecipeMetadata,
)
from src.reconstructor import (
    derive_row_seed,
    parse_recipe,
    reconstruct,
    reconstruct_from_files,
)


# Les conversions string->datetime via pd.to_datetime ne devinent pas toujours le format
# sur les ISO timestamps avec offset numerique (+0000) ; on prefere les masquer en test.
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


USERS_CSV = Path(__file__).resolve().parent.parent / "data" / "original" / "users.csv"
TMP_DIR = Path(__file__).resolve().parent.parent / "output" / "tests_reconstructor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(
    name: str,
    column_type: ColumnType,
    dtype: str = "object",
    n_total: int = 5,
    n_unique: int | None = None,
    n_null: int = 0,
    is_anchor_candidate: bool = False,
) -> ColumnProfile:
    """Construit un ColumnProfile minimal et valide pour les tests."""
    return ColumnProfile(
        name=name,
        column_type=column_type,
        dtype=dtype,
        n_unique=(n_unique if n_unique is not None else n_total),
        n_null=n_null,
        n_total=n_total,
        is_anchor_candidate=is_anchor_candidate,
    )


def _minimal_metadata(table_name: str = "test", n_rows: int = 5) -> RecipeMetadata:
    """Metadata bidon mais valide pour les tests."""
    return RecipeMetadata(
        table_name=table_name,
        n_rows=n_rows,
        original_size_bytes=100,
        anchor_size_bytes=50,
        compression_ratio=2.0,
    )


def _render_recipe_md(recipe: Recipe) -> str:
    """Render minimal d'une Recipe au format `.md` selon le contrat partage.

    Reproduit le format attendu par `parse_recipe` : sections delimitees par
    `## NAME`, blocs JSON precedes de `<!-- DATA -->`.
    """
    sections: list[str] = []
    sections.append(f"# RECIPE: {recipe.metadata.table_name}")
    sections.append("")

    def _emit_section(name: str, data: object) -> str:
        return (
            f"## {name}\n\n"
            f"<!-- DATA -->\n"
            f"```json\n"
            f"{json.dumps(data, indent=2, default=str)}\n"
            f"```\n"
        )

    sections.append(_emit_section("METADATA", recipe.metadata.model_dump(mode="json")))
    sections.append(
        _emit_section(
            "SCHEMA",
            # On exclut les champs computed (cardinality, null_ratio) qui sinon
            # sont rejetes par Pydantic au reload (extra="forbid").
            [
                c.model_dump(mode="json", exclude={"cardinality", "null_ratio"})
                for c in recipe.schema_columns
            ],
        )
    )
    sections.append(
        _emit_section(
            "ANCHORS",
            {
                "anchor_file": recipe.anchor_file,
                "anchor_columns": list(recipe.anchor_columns),
            },
        )
    )
    sections.append(
        _emit_section(
            "PATTERNS",
            [p.model_dump(mode="json") for p in recipe.patterns],
        )
    )
    sections.append(
        _emit_section(
            "CORRELATIONS",
            [c.model_dump(mode="json") for c in recipe.correlations],
        )
    )
    sections.append(
        _emit_section(
            "FUNCTIONAL_DEPENDENCIES",
            [f.model_dump(mode="json") for f in recipe.functional_dependencies],
        )
    )
    sections.append("## RECONSTRUCTION_ALGORITHM\n\nText-only section, ignored.\n")
    sections.append(_emit_section("VALIDATION_TESTS", recipe.validation_tests))

    return "\n".join(sections)


def _has_recipe_writer() -> bool:
    """Detecte si src.recipe_writer.write_recipe est disponible.

    Les tests end-to-end qui dependent du writer sont skip si absent (autre agent
    travaille en parallele sur ce module).
    """
    spec = importlib.util.find_spec("src.recipe_writer")
    if spec is None:
        return False
    try:
        from src import recipe_writer  # noqa: F401
        return hasattr(recipe_writer, "write_recipe")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. derive_row_seed : deterministe
# ---------------------------------------------------------------------------


def test_derive_row_seed_deterministic() -> None:
    """Meme input -> meme seed ; inputs differents -> seeds differents."""
    assert derive_row_seed("abc") == derive_row_seed("abc")
    assert derive_row_seed("abc") != derive_row_seed("def")
    # Stabilite avec salt explicite.
    assert derive_row_seed("abc", salt="x") == derive_row_seed("abc", salt="x")
    assert derive_row_seed("abc", salt="x") != derive_row_seed("abc", salt="y")
    # Stabilite type-insensitive : int vs str donnent la meme cle si str(int)==str.
    assert derive_row_seed(42) == derive_row_seed("42")


# ---------------------------------------------------------------------------
# 2. derive_row_seed : 32 bits
# ---------------------------------------------------------------------------


def test_derive_row_seed_int_in_32bits() -> None:
    """Seed doit etre < 2**32 (compatible np.random.default_rng)."""
    for sample in ["abc", "xyz123", "8826d916-cdfb-41c6-81ff-91a761565a70", 0, 999999]:
        seed = derive_row_seed(sample)
        assert 0 <= seed < 2**32, f"seed out of 32-bit range: {seed} for {sample!r}"
    # Et le seed est utilisable par numpy sans lever.
    rng = np.random.default_rng(derive_row_seed("test"))
    _ = rng.random()


# ---------------------------------------------------------------------------
# 3. parse_recipe : round-trip
# ---------------------------------------------------------------------------


def test_parse_recipe_extracts_all_sections(tmp_path: Path) -> None:
    """Construit une Recipe minimale, la render en .md, la reparse, verifie l'egalite."""
    schema_columns = [
        _make_profile("id", ColumnType.STRING, dtype="object", is_anchor_candidate=True),
        _make_profile("age", ColumnType.NUMERIC, dtype="int64"),
    ]
    patterns = [
        Pattern(
            column="id",
            pattern_type=PatternType.ANCHOR_DIRECT,
            fidelity_estimate=1.0,
        ),
        Pattern(
            column="age",
            pattern_type=PatternType.DISTRIBUTION,
            distribution=DistributionType.NORMAL,
            distribution_params={"loc": 34.0, "scale": 12.0},
            fidelity_estimate=0.9,
        ),
    ]
    original = Recipe(
        metadata=_minimal_metadata(),
        schema_columns=schema_columns,
        anchor_columns=["id"],
        anchor_file="anchors_test.parquet",
        patterns=patterns,
    )

    md_text = _render_recipe_md(original)
    recipe_path = tmp_path / "recipe.md"
    recipe_path.write_text(md_text, encoding="utf-8")

    parsed = parse_recipe(recipe_path)
    # Equivalence semantique : meme model_dump.
    assert parsed.metadata.table_name == original.metadata.table_name
    assert parsed.metadata.n_rows == original.metadata.n_rows
    assert [c.name for c in parsed.schema_columns] == [c.name for c in original.schema_columns]
    assert parsed.anchor_columns == original.anchor_columns
    assert parsed.anchor_file == original.anchor_file
    assert len(parsed.patterns) == 2
    assert parsed.patterns[0].pattern_type == PatternType.ANCHOR_DIRECT
    assert parsed.patterns[1].pattern_type == PatternType.DISTRIBUTION
    assert parsed.patterns[1].distribution == DistributionType.NORMAL


# ---------------------------------------------------------------------------
# 4. reconstruct anchor-only
# ---------------------------------------------------------------------------


def test_reconstruct_anchor_only_recipe() -> None:
    """Une seule colonne ANCHOR_DIRECT : reconstruction = copy."""
    profile = _make_profile(
        "id", ColumnType.STRING, dtype="object", n_total=5, n_unique=5, is_anchor_candidate=True
    )
    pattern = Pattern(
        column="id", pattern_type=PatternType.ANCHOR_DIRECT, fidelity_estimate=1.0
    )
    recipe = Recipe(
        metadata=_minimal_metadata(n_rows=5),
        schema_columns=[profile],
        anchor_columns=["id"],
        anchor_file="x.parquet",
        patterns=[pattern],
    )
    anchors = pd.DataFrame({"id": ["a", "b", "c", "d", "e"]})

    df = reconstruct(recipe, anchors)
    assert df.shape == (5, 1)
    assert list(df.columns) == ["id"]
    assert df["id"].tolist() == ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------------------
# 5. reconstruct normal distribution
# ---------------------------------------------------------------------------


def test_reconstruct_normal_distribution() -> None:
    """Pattern DISTRIBUTION normale : mean/std de la reconstruction proches des params."""
    n = 1000
    anchors = pd.DataFrame({"id": [f"u{i:04d}" for i in range(n)]})
    recipe = Recipe(
        metadata=_minimal_metadata(n_rows=n),
        schema_columns=[
            _make_profile(
                "id", ColumnType.STRING, dtype="object", n_total=n, n_unique=n,
                is_anchor_candidate=True,
            ),
            _make_profile("age", ColumnType.NUMERIC, dtype="int64", n_total=n, n_unique=n),
        ],
        anchor_columns=["id"],
        anchor_file="x.parquet",
        patterns=[
            Pattern(column="id", pattern_type=PatternType.ANCHOR_DIRECT, fidelity_estimate=1.0),
            Pattern(
                column="age",
                pattern_type=PatternType.DISTRIBUTION,
                distribution=DistributionType.NORMAL,
                distribution_params={"loc": 34.0, "scale": 12.0, "clip": [18, 99]},
                fidelity_estimate=0.9,
            ),
        ],
    )

    df = reconstruct(recipe, anchors)
    assert len(df) == n
    assert df["age"].min() >= 18
    assert df["age"].max() <= 99
    # Tolerance large : avec 1000 echantillons + clip a [18,99], on a un biais.
    assert abs(df["age"].mean() - 34) < 3.0, f"mean too far: {df['age'].mean()}"
    # std clipee perd un peu, mais reste de l'ordre de 10-12.
    assert 7.0 < df["age"].std() < 15.0, f"std out of range: {df['age'].std()}"


# ---------------------------------------------------------------------------
# 6. reconstruct categorical distribution
# ---------------------------------------------------------------------------


def test_reconstruct_categorical_distribution() -> None:
    """Pattern CATEGORICAL_FREQ {"A": 0.8, "B": 0.2}, 1000 IDs -> ~80% A, ~20% B."""
    n = 1000
    anchors = pd.DataFrame({"id": [f"u{i:04d}" for i in range(n)]})
    recipe = Recipe(
        metadata=_minimal_metadata(n_rows=n),
        schema_columns=[
            _make_profile(
                "id", ColumnType.STRING, dtype="object", n_total=n, n_unique=n,
                is_anchor_candidate=True,
            ),
            _make_profile(
                "category", ColumnType.CATEGORICAL, dtype="object", n_total=n, n_unique=2
            ),
        ],
        anchor_columns=["id"],
        anchor_file="x.parquet",
        patterns=[
            Pattern(column="id", pattern_type=PatternType.ANCHOR_DIRECT, fidelity_estimate=1.0),
            Pattern(
                column="category",
                pattern_type=PatternType.DISTRIBUTION,
                distribution=DistributionType.CATEGORICAL_FREQ,
                distribution_params={"frequencies": {"A": 0.8, "B": 0.2}},
                fidelity_estimate=1.0,
            ),
        ],
    )

    df = reconstruct(recipe, anchors)
    freqs = df["category"].value_counts(normalize=True)
    assert set(freqs.index) <= {"A", "B"}
    # Tolerance ~3pp sur 1000 echantillons.
    assert abs(freqs.get("A", 0.0) - 0.8) < 0.05, f"A freq off: {freqs.get('A')}"
    assert abs(freqs.get("B", 0.0) - 0.2) < 0.05, f"B freq off: {freqs.get('B')}"


# ---------------------------------------------------------------------------
# 7. reconstruct functional dependency
# ---------------------------------------------------------------------------


def test_reconstruct_functional_dependency() -> None:
    """FUNCTIONAL_DEP lookup table {1:'a', 2:'b'}, ancres ont colonne 'code'."""
    anchors = pd.DataFrame({"id": ["u1", "u2", "u3", "u4"], "code": [1, 2, 1, 2]})
    recipe = Recipe(
        metadata=_minimal_metadata(n_rows=4),
        schema_columns=[
            _make_profile(
                "id", ColumnType.STRING, dtype="object", n_total=4, n_unique=4,
                is_anchor_candidate=True,
            ),
            _make_profile(
                "code", ColumnType.NUMERIC, dtype="int64", n_total=4, n_unique=2,
                is_anchor_candidate=True,
            ),
            _make_profile(
                "name", ColumnType.STRING, dtype="object", n_total=4, n_unique=2
            ),
        ],
        anchor_columns=["id", "code"],
        anchor_file="x.parquet",
        patterns=[
            Pattern(column="id", pattern_type=PatternType.ANCHOR_DIRECT, fidelity_estimate=1.0),
            Pattern(column="code", pattern_type=PatternType.ANCHOR_DIRECT, fidelity_estimate=1.0),
            Pattern(
                column="name",
                pattern_type=PatternType.FUNCTIONAL_DEP,
                source_column="code",
                lookup_table={1: "a", 2: "b"},
                fidelity_estimate=1.0,
                dependencies=["code"],
            ),
        ],
    )

    df = reconstruct(recipe, anchors)
    assert df["name"].tolist() == ["a", "b", "a", "b"]
    assert df["code"].tolist() == [1, 2, 1, 2]


# ---------------------------------------------------------------------------
# 8. reproductibilite bit-a-bit
# ---------------------------------------------------------------------------


def test_reconstruction_reproducibility() -> None:
    """Deux reconstructions consecutives doivent etre rigoureusement identiques.

    C'est le critere d'acceptation reproductibilite de la spec.
    """
    n = 200
    anchors = pd.DataFrame({"id": [f"u{i:04d}" for i in range(n)]})
    recipe = Recipe(
        metadata=_minimal_metadata(n_rows=n),
        schema_columns=[
            _make_profile(
                "id", ColumnType.STRING, dtype="object", n_total=n, n_unique=n,
                is_anchor_candidate=True,
            ),
            _make_profile("age", ColumnType.NUMERIC, dtype="int64", n_total=n, n_unique=n),
            _make_profile(
                "country", ColumnType.CATEGORICAL, dtype="object", n_total=n, n_unique=3
            ),
        ],
        anchor_columns=["id"],
        anchor_file="x.parquet",
        patterns=[
            Pattern(column="id", pattern_type=PatternType.ANCHOR_DIRECT, fidelity_estimate=1.0),
            Pattern(
                column="age",
                pattern_type=PatternType.DISTRIBUTION,
                distribution=DistributionType.NORMAL,
                distribution_params={"loc": 34.0, "scale": 12.0, "clip": [18, 99]},
                fidelity_estimate=0.9,
            ),
            Pattern(
                column="country",
                pattern_type=PatternType.DISTRIBUTION,
                distribution=DistributionType.CATEGORICAL_FREQ,
                distribution_params={"frequencies": {"US": 0.5, "FR": 0.3, "DE": 0.2}},
                fidelity_estimate=1.0,
            ),
        ],
    )

    df1 = reconstruct(recipe, anchors)
    df2 = reconstruct(recipe, anchors)
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)


# ---------------------------------------------------------------------------
# Tests end-to-end : dependent du recipe_writer (autre agent)
# ---------------------------------------------------------------------------


def _build_recipe_from_users_csv(df: pd.DataFrame, anchor_path: Path) -> Recipe:
    """Helper local : factorise le pipeline profile->patterns->anchors->Recipe pour
    les tests 9 et 10.
    """
    from src.anchor_extractor import extract_anchors, write_anchors_parquet
    from src.pattern_detector import build_patterns
    from src.profiler import profile_dataframe

    profiles = profile_dataframe(df, table_name="users")
    anchor_cols_pre = [p.name for p in profiles if p.is_anchor_candidate]
    patterns = build_patterns(df, profiles, anchor_columns=anchor_cols_pre)
    df_anchors, anchor_cols_final = extract_anchors(df, profiles, patterns=patterns)

    write_anchors_parquet(df_anchors, anchor_path, compression="zstd")

    csv_size = USERS_CSV.stat().st_size
    anchor_size = anchor_path.stat().st_size
    return Recipe(
        metadata=RecipeMetadata(
            table_name="users",
            n_rows=len(df),
            original_size_bytes=csv_size,
            anchor_size_bytes=anchor_size,
            compression_ratio=csv_size / max(anchor_size, 1),
        ),
        schema_columns=profiles,
        anchor_columns=anchor_cols_final,
        anchor_file=anchor_path.name,
        patterns=patterns,
    )


@pytest.mark.skipif(
    not USERS_CSV.exists(), reason="users.csv not generated yet"
)
@pytest.mark.skipif(
    not _has_recipe_writer(),
    reason="src.recipe_writer.write_recipe not available (parallel agent)",
)
def test_end_to_end_users_csv(tmp_path: Path) -> None:
    """Profile + build_patterns + extract_anchors + render recipe -> reconstruct -> verifie."""
    from src.recipe_writer import write_recipe  # type: ignore

    df = pd.read_csv(USERS_CSV)
    anchor_path = tmp_path / "anchors.parquet"
    recipe_obj = _build_recipe_from_users_csv(df, anchor_path)
    recipe_path = tmp_path / "recipe.md"
    write_recipe(recipe_obj, recipe_path)

    # Charge + reconstruit.
    reconstructed = reconstruct_from_files(recipe_path, anchor_path)

    assert len(reconstructed) == 10_000, f"expected 10000 rows, got {len(reconstructed)}"
    assert set(reconstructed.columns) == set(df.columns)
    # Anchor columns identiques (id, email, password_hash).
    for col in recipe_obj.anchor_columns:
        assert (reconstructed[col].reset_index(drop=True) == df[col].reset_index(drop=True)).all(), (
            f"anchor column {col} differs from original"
        )
    # Age borne [18, 99].
    age_min, age_max = reconstructed["age"].min(), reconstructed["age"].max()
    assert 18 <= age_min and age_max <= 99, f"age out of range: [{age_min}, {age_max}]"
    # Country dans la liste autorisee.
    countries_orig = set(df["country"].unique())
    countries_reco = set(reconstructed["country"].unique())
    assert countries_reco.issubset(countries_orig), (
        f"unexpected countries in reconstruction: {countries_reco - countries_orig}"
    )


@pytest.mark.skipif(
    not USERS_CSV.exists(), reason="users.csv not generated yet"
)
@pytest.mark.skipif(
    not _has_recipe_writer(),
    reason="src.recipe_writer.write_recipe not available (parallel agent)",
)
def test_full_pipeline_validation_score_high(tmp_path: Path) -> None:
    """End-to-end + validation : on attend un score >= 70 a ce stade du POC."""
    from src.recipe_writer import write_recipe  # type: ignore
    from src.validator import validate

    df = pd.read_csv(USERS_CSV)
    anchor_path = tmp_path / "anchors.parquet"
    recipe_obj = _build_recipe_from_users_csv(df, anchor_path)
    recipe_path = tmp_path / "recipe.md"
    write_recipe(recipe_obj, recipe_path)

    reconstructed = reconstruct_from_files(recipe_path, anchor_path)

    report = validate(df, reconstructed, anchor_columns=recipe_obj.anchor_columns)
    assert report.overall_score >= 70.0, (
        f"validation score too low: {report.overall_score} "
        f"(passed={report.passed_count}, failed={report.failed_count})"
    )
