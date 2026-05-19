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

USERS_CSV = Path(__file__).parent.parent / "data" / "original" / "users.csv"


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


@pytest.mark.skipif(
    not USERS_CSV.exists(),
    reason=f"Test fixture missing: {USERS_CSV} (data/ is gitignored, regenerate via examples/)",
)
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


@pytest.mark.skipif(
    not USERS_CSV.exists(),
    reason=f"Test fixture missing: {USERS_CSV} (data/ is gitignored, regenerate via examples/)",
)
def test_build_patterns_on_real_users():
    """build_patterns sur users.csv : 9 patterns, ancres correctes, distributions correctes.

    Note : `password_hash` est auto-detecte comme RANDOM_FORMAT (bcrypt). `email`
    est auto-detecte comme EMAIL_SPLIT (3 domaines distincts dans users.csv).
    `id` reste ANCHOR_DIRECT par defaut (UUID v4 + nom id-like = preservation).
    """
    df = pd.read_csv(USERS_CSV)
    profiles = _users_profiles(df)
    anchors = ["id", "email", "password_hash"]

    patterns = build_patterns(df, profiles, anchors)
    by_col = {p.column: p for p in patterns}

    # Une Pattern par colonne (9 total)
    assert len(patterns) == 9, f"Expected 9 patterns, got {len(patterns)}"
    assert set(by_col.keys()) == set(df.columns)

    # id reste ANCHOR_DIRECT (UUID v4 + nom 'id' = guard actif sans aggressive_uuid).
    assert by_col["id"].pattern_type == PatternType.ANCHOR_DIRECT, (
        f"id should be ANCHOR_DIRECT, got {by_col['id'].pattern_type}"
    )
    # email est EMAIL_SPLIT (3 domaines distincts dans users.csv).
    assert by_col["email"].pattern_type == PatternType.EMAIL_SPLIT, (
        f"email should be EMAIL_SPLIT, got {by_col['email'].pattern_type}"
    )
    assert by_col["email"].format_spec is not None
    assert by_col["email"].format_spec["separator"] == "@"
    assert len(by_col["email"].format_spec["domain_dict"]) >= 1
    # password_hash est RANDOM_FORMAT (bcrypt-like).
    assert by_col["password_hash"].pattern_type == PatternType.RANDOM_FORMAT, (
        f"password_hash should be RANDOM_FORMAT (bcrypt auto-detected), "
        f"got {by_col['password_hash'].pattern_type}"
    )
    assert by_col["password_hash"].format_spec is not None
    assert by_col["password_hash"].format_spec["body_type"] == "hex"
    assert by_col["password_hash"].format_spec["body_length"] == 64

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


# ---------------------------------------------------------------------------
# RANDOM_FORMAT detection tests
# ---------------------------------------------------------------------------


def test_detect_bcrypt_format():
    """Une serie de 100 valeurs bcrypt-like doit etre detectee comme RANDOM_FORMAT."""
    from src.pattern_detector import detect_random_format

    rng = np.random.default_rng(0)
    # Genere 100 valeurs bcrypt-formed avec hex bodies de longueur 64.
    values = []
    for _ in range(100):
        body = "".join(rng.choice(list("0123456789abcdef"), size=64).tolist())
        values.append(f"$bcrypt$2b$12${body}$")
    series = pd.Series(values, name="password_hash")

    spec = detect_random_format(series)
    assert spec is not None, "Expected bcrypt format to be detected"
    assert spec["body_type"] == "hex"
    assert spec["body_length"] == 64
    assert spec["prefix"] == "$bcrypt$2b$12$"
    assert spec["suffix"] == "$"
    # Le regex doit matcher au moins une des valeurs.
    import re

    pat = re.compile(spec["regex"])
    assert pat.match(values[0]), f"Regex {spec['regex']!r} should match {values[0]!r}"


def test_random_format_uuid_detected_as_format():
    """Une serie d'UUIDs v4 est detectee par `detect_random_format` au niveau de
    la *primitive* (KNOWN_RANDOM_FORMATS contient uuid_v4_anchor depuis l'ajout
    de l'option --aggressive-uuid).

    Le garde-fou "id-like column name -> reste ANCHOR_DIRECT" est applique au
    niveau de `build_patterns`, pas dans `detect_random_format` qui est une
    primitive bas-niveau. Cf. `test_uuid_with_id_name_stays_anchor` pour le test
    de garde-fou.
    """
    from src.pattern_detector import detect_random_format
    import uuid

    values = [str(uuid.uuid4()) for _ in range(100)]
    series = pd.Series(values, name="id")

    spec = detect_random_format(series)
    assert spec is not None, (
        f"UUID v4 should be detected by detect_random_format. Got spec={spec}"
    )
    assert spec["body_type"] == "uuid_v4"
    assert spec["body_length"] == 36
    assert spec.get("detected_as") == "uuid_v4_anchor"


def test_uuid_with_id_name_stays_anchor():
    """build_patterns ne marque PAS une colonne `id` UUID comme RANDOM_FORMAT
    par defaut (aggressive_uuid=False), pour preserver la valeur exacte de
    l'id (FK potentielles).
    """
    import uuid
    from src.pattern_detector import build_patterns

    values = [str(uuid.uuid4()) for _ in range(200)]
    df = pd.DataFrame({"id": values, "name": ["alice", "bob"] * 100})
    profiles = [
        _make_profile("id", ColumnType.STRING, df["id"], is_anchor_candidate=True),
        _make_profile("name", ColumnType.CATEGORICAL, df["name"]),
    ]
    patterns = build_patterns(df, profiles, anchor_columns=["id"])
    by_col = {p.column: p for p in patterns}

    assert by_col["id"].pattern_type == PatternType.ANCHOR_DIRECT, (
        f"id should stay ANCHOR_DIRECT (default mode), got {by_col['id'].pattern_type}"
    )


def test_aggressive_uuid_forces_random_format():
    """Avec aggressive_uuid=True, une colonne `id` UUID DOIT etre marquee
    RANDOM_FORMAT meme si son nom ressemble a un id.
    """
    import uuid
    from src.pattern_detector import build_patterns

    values = [str(uuid.uuid4()) for _ in range(200)]
    df = pd.DataFrame({"id": values, "name": ["alice", "bob"] * 100})
    profiles = [
        _make_profile("id", ColumnType.STRING, df["id"], is_anchor_candidate=True),
        _make_profile("name", ColumnType.CATEGORICAL, df["name"]),
    ]
    patterns = build_patterns(
        df, profiles, anchor_columns=["id"], aggressive_uuid=True
    )
    by_col = {p.column: p for p in patterns}

    assert by_col["id"].pattern_type == PatternType.RANDOM_FORMAT, (
        f"id with aggressive_uuid=True should be RANDOM_FORMAT, got {by_col['id'].pattern_type}"
    )
    assert by_col["id"].format_spec is not None
    assert by_col["id"].format_spec["body_type"] == "uuid_v4"


def test_uuid_in_non_id_column_auto_random_format():
    """Une colonne UUID v4 dont le nom ne ressemble PAS a un id (ex: `token`,
    `session_key`, etc.) est auto-marquee RANDOM_FORMAT meme sans aggressive_uuid.
    """
    import uuid
    from src.pattern_detector import build_patterns

    values = [str(uuid.uuid4()) for _ in range(200)]
    df = pd.DataFrame({"token": values, "name": ["alice", "bob"] * 100})
    profiles = [
        _make_profile("token", ColumnType.STRING, df["token"], is_anchor_candidate=True),
        _make_profile("name", ColumnType.CATEGORICAL, df["name"]),
    ]
    patterns = build_patterns(df, profiles, anchor_columns=["token"])
    by_col = {p.column: p for p in patterns}

    assert by_col["token"].pattern_type == PatternType.RANDOM_FORMAT, (
        f"token should be RANDOM_FORMAT (no id-like guard), got {by_col['token'].pattern_type}"
    )


# ---------------------------------------------------------------------------
# EMAIL_SPLIT detection tests
# ---------------------------------------------------------------------------


def test_email_split_detection():
    """Une serie d'emails avec peu de domaines distincts doit etre detectee
    comme EMAIL_SPLIT par `detect_email_split`.
    """
    from src.pattern_detector import detect_email_split

    rng = np.random.default_rng(0)
    domains = ["example.com", "example.net", "example.org"]
    emails = [
        f"user{i}@{rng.choice(domains)}" for i in range(500)
    ]
    series = pd.Series(emails, name="email")

    spec = detect_email_split(series)
    assert spec is not None, "Expected email split detection to succeed"
    assert spec["separator"] == "@"
    assert set(spec["domain_dict"]) == set(domains), (
        f"domain_dict should contain {domains}, got {spec['domain_dict']}"
    )
    # Verifie l'ordre alphabetique deterministe.
    assert spec["domain_dict"] == sorted(spec["domain_dict"])


def test_email_split_skips_too_many_domains():
    """Si le nombre de domaines distincts depasse max_dict_size, on retourne None.
    """
    from src.pattern_detector import detect_email_split

    emails = [f"user{i}@domain{i}.com" for i in range(500)]
    series = pd.Series(emails, name="email")
    spec = detect_email_split(series, max_dict_size=10)
    assert spec is None, "Expected None when too many distinct domains"


def test_email_split_skips_non_emails():
    """Si <95% des valeurs sont des emails, on retourne None.
    """
    from src.pattern_detector import detect_email_split

    values = [f"user{i}@example.com" for i in range(50)] + [
        f"not-an-email-{i}" for i in range(50)
    ]
    series = pd.Series(values, name="email")
    spec = detect_email_split(series)
    assert spec is None, "Expected None when match_ratio < 0.95"


def test_build_patterns_emits_email_split():
    """build_patterns doit emettre un Pattern EMAIL_SPLIT pour une colonne email
    avec peu de domaines distincts.
    """
    from src.pattern_detector import build_patterns

    emails = [f"user{i}@example.com" for i in range(500)]
    df = pd.DataFrame({"email": emails, "name": ["alice", "bob"] * 250})
    profiles = [
        _make_profile("email", ColumnType.STRING, df["email"], is_anchor_candidate=True),
        _make_profile("name", ColumnType.CATEGORICAL, df["name"]),
    ]
    patterns = build_patterns(df, profiles, anchor_columns=["email"])
    by_col = {p.column: p for p in patterns}

    assert by_col["email"].pattern_type == PatternType.EMAIL_SPLIT, (
        f"email should be EMAIL_SPLIT, got {by_col['email'].pattern_type}"
    )
    assert by_col["email"].format_spec is not None
    assert by_col["email"].format_spec["separator"] == "@"
    assert by_col["email"].format_spec["domain_dict"] == ["example.com"]
