"""Tests pour src/pattern_detector.py.

Couvre les 9 cas demandes dans la spec :
1-2. Dependances fonctionnelles (positif + non-trivial)
3-4. Distributions (normal, exponential)
5-6. Cramer's V (perfect / no correlation)
7. Correlation num-num (Pearson)
8. Correlation cat-cat sur users.csv reel
9. Pipeline build_patterns sur users.csv reel
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import ColumnProfile, ColumnType, DistributionType, PatternType
from src.pattern_detector import (
    build_patterns,
    cramers_v,
    detect_correlations,
    detect_distributions,
    detect_functional_dependencies,
)

# Les conversions string->datetime via pd.to_datetime ne devinent pas toujours le format
# sur les ISO timestamps avec offset numerique (+0000) ; on prefere les masquer en test.
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

USERS_CSV = Path(__file__).resolve().parent.parent / "data" / "original" / "users.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(
    name: str,
    column_type: ColumnType,
    series: pd.Series,
    is_anchor_candidate: bool = False,
) -> ColumnProfile:
    """Construit un ColumnProfile minimal a partir d'une serie pandas."""
    return ColumnProfile(
        name=name,
        column_type=column_type,
        dtype=str(series.dtype),
        n_unique=int(series.nunique(dropna=True)),
        n_null=int(series.isna().sum()),
        n_total=len(series),
        is_anchor_candidate=is_anchor_candidate,
    )


def _users_profiles(df: pd.DataFrame) -> list[ColumnProfile]:
    """Profils pour le DataFrame de users.csv."""
    type_map = {
        "id": ColumnType.STRING,
        "email": ColumnType.STRING,
        "password_hash": ColumnType.STRING,
        "created_at": ColumnType.DATETIME,
        "country": ColumnType.CATEGORICAL,
        "age": ColumnType.NUMERIC,
        "premium": ColumnType.BOOLEAN,
        "last_login": ColumnType.DATETIME,
        "signup_source": ColumnType.CATEGORICAL,
    }
    profiles = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        is_anchor = n_unique / len(df) >= 1.0
        profiles.append(
            _make_profile(col, type_map[col], df[col], is_anchor_candidate=is_anchor)
        )
    return profiles


# ---------------------------------------------------------------------------
# 1. Dependance fonctionnelle bijective
# ---------------------------------------------------------------------------


def test_functional_dependency_country_code():
    """country_code <-> country_name doivent etre detectes comme une bijection."""
    df = pd.DataFrame(
        {
            "country_code": ["FR", "US", "FR", "DE", "US", "DE", "FR"] * 50,
            "country_name": [
                "France",
                "United States",
                "France",
                "Germany",
                "United States",
                "Germany",
                "France",
            ]
            * 50,
        }
    )
    deps = detect_functional_dependencies(df)
    assert len(deps) >= 1, "Should detect at least one functional dependency"
    # On verifie qu'au moins une dep est bijective (peut etre A->B ou B->A).
    assert any(
        {fd.determinant, fd.dependent} == {"country_code", "country_name"}
        and fd.is_bijection
        for fd in deps
    ), f"Expected bijection between country_code and country_name. Got: {deps}"


# ---------------------------------------------------------------------------
# 2. Pas de dependance triviale sur les colonnes uniques
# ---------------------------------------------------------------------------


def test_no_functional_dependency_on_uniques():
    """Une colonne ID unique ne doit pas etre marquee comme determinant trivial."""
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(100)],
            "country": ["FR", "US"] * 50,
        }
    )
    # `id` a cardinalite 1.0 -> doit etre filtre comme determinant.
    deps = detect_functional_dependencies(df)
    determinant_names = {fd.determinant for fd in deps}
    assert "id" not in determinant_names, (
        f"id should not be a determinant (trivially true). Got determinants: {determinant_names}"
    )


# ---------------------------------------------------------------------------
# 3. Detection de distribution normale
# ---------------------------------------------------------------------------


def test_detect_normal_distribution():
    """Une serie normale(10, 2) doit etre detectee comme NORMAL avec params corrects."""
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=10.0, scale=2.0, size=5000)
    df = pd.DataFrame({"x": arr})
    profile = _make_profile("x", ColumnType.NUMERIC, df["x"])
    dists = detect_distributions(df, [profile])
    assert "x" in dists
    result = dists["x"]
    assert result["distribution"] == DistributionType.NORMAL, (
        f"Expected NORMAL, got {result['distribution']}"
    )
    params = result["params"]
    assert abs(params["mean"] - 10.0) < 0.2, f"mean off: {params['mean']}"
    assert abs(params["std"] - 2.0) < 0.2, f"std off: {params['std']}"


# ---------------------------------------------------------------------------
# 4. Detection de distribution exponentielle
# ---------------------------------------------------------------------------


def test_detect_exponential_distribution():
    """Une serie exponential(5) doit etre detectee comme EXPONENTIAL."""
    rng = np.random.default_rng(0)
    arr = rng.exponential(scale=5.0, size=5000)
    df = pd.DataFrame({"x": arr})
    profile = _make_profile("x", ColumnType.NUMERIC, df["x"])
    dists = detect_distributions(df, [profile])
    assert "x" in dists
    result = dists["x"]
    assert result["distribution"] == DistributionType.EXPONENTIAL, (
        f"Expected EXPONENTIAL, got {result['distribution']}"
    )


# ---------------------------------------------------------------------------
# 5. Cramer's V : correlation parfaite
# ---------------------------------------------------------------------------


def test_cramers_v_perfect_correlation():
    """Deux colonnes identiques doivent donner Cramer's V proche de 1.0."""
    x = pd.Series(["a", "b", "c", "a", "b", "c"] * 100)
    y = x.copy()
    v = cramers_v(x, y)
    assert v > 0.99, f"Expected ~1.0, got {v}"


# ---------------------------------------------------------------------------
# 6. Cramer's V : pas de correlation
# ---------------------------------------------------------------------------


def test_cramers_v_no_correlation():
    """Deux colonnes independantes doivent donner Cramer's V faible (< 0.1)."""
    rng = np.random.default_rng(42)
    n = 5000
    x = pd.Series(rng.choice(["a", "b", "c"], size=n))
    y = pd.Series(rng.choice(["x", "y", "z"], size=n))
    v = cramers_v(x, y)
    assert v < 0.1, f"Expected < 0.1 for independent series, got {v}"


# ---------------------------------------------------------------------------
# 7. Correlation num-num (Pearson)
# ---------------------------------------------------------------------------


def test_detect_correlation_num_num():
    """y = 2x + bruit -> Pearson doit etre detecte au-dessus de 0.3."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 1000)
    y = 2 * x + rng.normal(0, 0.5, 1000)
    df = pd.DataFrame({"x": x, "y": y})
    profiles = [
        _make_profile("x", ColumnType.NUMERIC, df["x"]),
        _make_profile("y", ColumnType.NUMERIC, df["y"]),
    ]
    corrs = detect_correlations(df, profiles, threshold=0.3)
    pearsons = [c for c in corrs if c.correlation_type == "pearson"]
    assert len(pearsons) >= 1, f"Expected at least one Pearson correlation. Got: {corrs}"
    found = pearsons[0]
    assert {found.col_a, found.col_b} == {"x", "y"}
    assert found.strength > 0.9, f"Expected very high correlation, got {found.strength}"


# ---------------------------------------------------------------------------
# 8. Correlation cat-cat sur users.csv reel
# ---------------------------------------------------------------------------


def test_detect_correlation_cat_cat_on_real_users():
    """Sur users.csv, (country, signup_source) doit apparaitre dans les correlations."""
    df = pd.read_csv(USERS_CSV)
    profiles = _users_profiles(df)
    # On baisse le seuil a 0.05 car la correlation reelle est ~0.16 (forte mais pas >0.3).
    corrs = detect_correlations(df, profiles, threshold=0.05)
    pairs = {frozenset({c.col_a, c.col_b}) for c in corrs if c.correlation_type == "cramers_v"}
    assert frozenset({"country", "signup_source"}) in pairs, (
        f"Expected (country, signup_source) in correlations. Got pairs: {pairs}"
    )


# ---------------------------------------------------------------------------
# 9. Pipeline build_patterns sur users.csv reel
# ---------------------------------------------------------------------------


def test_build_patterns_on_real_users():
    """build_patterns sur users.csv : 9 patterns, ancres correctes, distributions correctes."""
    df = pd.read_csv(USERS_CSV)
    profiles = _users_profiles(df)
    anchors = ["id", "email", "password_hash"]

    patterns = build_patterns(df, profiles, anchors)
    by_col = {p.column: p for p in patterns}

    # Une Pattern par colonne (9 total)
    assert len(patterns) == 9, f"Expected 9 patterns, got {len(patterns)}"
    assert set(by_col.keys()) == set(df.columns)

    # Ancres
    for anchor in anchors:
        assert by_col[anchor].pattern_type == PatternType.ANCHOR_DIRECT, (
            f"{anchor} should be ANCHOR_DIRECT, got {by_col[anchor].pattern_type}"
        )

    # country et signup_source : l'un des deux doit etre CONDITIONAL_DISTRIBUTION
    # (l'ordre depend de la cardinalite mais ils sont lies). Le cas typique :
    # country -> CONDITIONAL_DISTRIBUTION(source=signup_source) ou inverse.
    cat_patterns = {by_col["country"].pattern_type, by_col["signup_source"].pattern_type}
    assert PatternType.CONDITIONAL_DISTRIBUTION in cat_patterns, (
        f"Expected at least one of (country, signup_source) to be CONDITIONAL_DISTRIBUTION. "
        f"Got: country={by_col['country'].pattern_type}, "
        f"signup_source={by_col['signup_source'].pattern_type}"
    )

    # age : distribution univariee (normal)
    age_pat = by_col["age"]
    assert age_pat.pattern_type == PatternType.DISTRIBUTION, (
        f"age should be DISTRIBUTION, got {age_pat.pattern_type}"
    )
    assert age_pat.distribution == DistributionType.NORMAL, (
        f"age should be NORMAL, got {age_pat.distribution}"
    )

    # premium : CONDITIONAL_DISTRIBUTION (correlation avec country meme faible)
    # OU DISTRIBUTION (categorical_freq) si la correlation reste sous le seuil.
    # On accepte les deux pour ne pas etre trop strict, mais on verifie le type valide.
    premium_pat = by_col["premium"]
    assert premium_pat.pattern_type in (
        PatternType.CONDITIONAL_DISTRIBUTION,
        PatternType.DISTRIBUTION,
    ), f"premium has unexpected pattern type: {premium_pat.pattern_type}"

    # created_at / last_login : datetimes detectes comme distribution (exponential ou autre)
    for col in ("created_at", "last_login"):
        pat = by_col[col]
        assert pat.pattern_type == PatternType.DISTRIBUTION, (
            f"{col} should be DISTRIBUTION, got {pat.pattern_type}"
        )
        # Verifie que l'unite datetime est stockee pour reconstruction.
        assert pat.distribution_params.get("unit") == "seconds_since_epoch", (
            f"{col} should have 'unit'='seconds_since_epoch' in params. "
            f"Got: {pat.distribution_params}"
        )

    # Verifie aussi qu'aucun pattern n'a de NaN/Inf dans ses params (JSON-safety).
    import json

    for pat in patterns:
        try:
            json.dumps(pat.distribution_params)
            if pat.conditional_buckets:
                json.dumps(pat.conditional_buckets)
            if pat.lookup_table:
                json.dumps(pat.lookup_table)
        except (TypeError, ValueError) as exc:
            pytest.fail(
                f"Pattern {pat.column} has non-JSON-serializable content: {exc}"
            )


# ---------------------------------------------------------------------------
# Bonus : tests defensifs sur le bucketing
# ---------------------------------------------------------------------------


def test_conditional_distribution_categorical_pivot():
    """Quand le pivot est categoriel, les buckets sont indexes par valeur unique."""
    from src.pattern_detector import detect_conditional_distributions

    rng = np.random.default_rng(0)
    pivot = pd.Series(rng.choice(["A", "B", "C"], size=300))
    # target qui depend de pivot
    target = pd.Series(
        [
            "x" if p == "A" else ("y" if p == "B" else "z")
            for p in pivot
        ]
    )
    df = pd.DataFrame({"pivot": pivot, "target": target})
    buckets = detect_conditional_distributions(df, "pivot", "target")
    assert len(buckets) == 3, f"Expected 3 buckets for cat pivot, got {len(buckets)}"
    for bucket in buckets:
        assert "bucket_value" in bucket
        assert "frequencies" in bucket
