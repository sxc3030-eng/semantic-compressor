"""Tests du module `src.profiler`.

Six cas couvrent les principaux scenarios :
- detection des trois types non-triviaux (NUMERIC, CATEGORICAL, DATETIME)
- detection de pattern regex (EMAIL, UUID)
- profilage de la vraie BD de test (users.csv) avec verifications de bout en bout
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

from src.models import ColumnType, RegexPattern
from src.profiler import detect_regex_pattern, profile_dataframe


REPO_ROOT = Path(__file__).resolve().parents[1]
USERS_CSV = REPO_ROOT / "data" / "original" / "users.csv"


# ---------------------------------------------------------------------------
# 1. Numeric
# ---------------------------------------------------------------------------


def test_profile_numeric_column() -> None:
    df = pd.DataFrame({"age": [18, 25, 33, 42, 51, 60, 27, 34, 45, 19]})
    profiles = profile_dataframe(df, table_name="numeric_test")

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.name == "age"
    assert profile.column_type == ColumnType.NUMERIC
    assert profile.numeric_stats is not None
    assert profile.categorical_stats is None
    assert profile.datetime_stats is None
    stats = profile.numeric_stats
    assert stats.min == 18.0
    assert stats.max == 60.0
    # Verifie que les quartiles sont coherents.
    assert stats.q25 <= stats.q50 <= stats.q75
    # 5 exemples max, ici on a 10 lignes donc on prend bien 5.
    assert len(profile.examples) == 5


# ---------------------------------------------------------------------------
# 2. Categorical
# ---------------------------------------------------------------------------


def test_profile_categorical_column() -> None:
    # 200 lignes, 4 categories distinctes -> ratio 0.02 < 0.05 -> CATEGORICAL.
    values = (["US"] * 100) + (["FR"] * 50) + (["DE"] * 30) + (["JP"] * 20)
    df = pd.DataFrame({"country": values})
    profiles = profile_dataframe(df)

    profile = profiles[0]
    assert profile.column_type == ColumnType.CATEGORICAL
    assert profile.categorical_stats is not None
    assert profile.numeric_stats is None
    stats = profile.categorical_stats
    assert stats.n_categories == 4
    assert stats.top_values["US"] == 100
    assert stats.top_values["FR"] == 50
    assert stats.top_values["DE"] == 30
    assert stats.top_values["JP"] == 20


# ---------------------------------------------------------------------------
# 3. Datetime stocke en string ISO
# ---------------------------------------------------------------------------


def test_profile_datetime_column() -> None:
    dates = [
        "2025-01-15T08:30:00+0000",
        "2025-02-20T11:45:10+0000",
        "2025-06-01T22:10:55+0000",
        "2025-09-12T03:00:00+0000",
        "2025-12-31T23:59:59+0000",
    ] * 20  # 100 lignes, toutes parseables.
    df = pd.DataFrame({"created_at": dates})
    profiles = profile_dataframe(df)

    profile = profiles[0]
    # Meme si pandas voit object, on doit detecter DATETIME.
    assert profile.column_type == ColumnType.DATETIME
    assert profile.datetime_stats is not None
    assert profile.numeric_stats is None
    assert profile.categorical_stats is None
    stats = profile.datetime_stats
    assert stats.min.year == 2025
    assert stats.max.year == 2025
    assert stats.range_days > 0


# ---------------------------------------------------------------------------
# 4. Pattern EMAIL
# ---------------------------------------------------------------------------


def test_detect_email_pattern() -> None:
    emails = pd.Series([f"user{i}@example.com" for i in range(100)])
    pattern = detect_regex_pattern(emails)
    assert pattern == RegexPattern.EMAIL


# ---------------------------------------------------------------------------
# 5. Pattern UUID
# ---------------------------------------------------------------------------


def test_detect_uuid_pattern() -> None:
    uuids = pd.Series([str(uuid.uuid4()) for _ in range(100)])
    pattern = detect_regex_pattern(uuids)
    assert pattern == RegexPattern.UUID


# ---------------------------------------------------------------------------
# 6. Profil sur la vraie BD users.csv
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not USERS_CSV.exists(), reason="users.csv non genere")
def test_profile_real_users_csv() -> None:
    df = pd.read_csv(USERS_CSV)
    profiles = profile_dataframe(df, table_name="users")

    # 9 colonnes attendues.
    assert len(profiles) == 9

    by_name = {p.name: p for p in profiles}

    # id : UUID, ancre.
    id_profile = by_name["id"]
    assert id_profile.is_anchor_candidate is True
    assert id_profile.regex_pattern == RegexPattern.UUID

    # email : ancre, pattern EMAIL.
    email_profile = by_name["email"]
    assert email_profile.is_anchor_candidate is True
    assert email_profile.regex_pattern == RegexPattern.EMAIL

    # age : numerique, mean entre 30 et 38.
    age_profile = by_name["age"]
    assert age_profile.column_type == ColumnType.NUMERIC
    assert age_profile.numeric_stats is not None
    assert 30 <= age_profile.numeric_stats.mean <= 38

    # country : categorial avec ~10 valeurs uniques.
    country_profile = by_name["country"]
    assert country_profile.column_type == ColumnType.CATEGORICAL
    assert country_profile.categorical_stats is not None
    assert 8 <= country_profile.categorical_stats.n_categories <= 12

    # created_at : datetime detecte depuis string ISO.
    created_at_profile = by_name["created_at"]
    assert created_at_profile.column_type == ColumnType.DATETIME
    assert created_at_profile.datetime_stats is not None
