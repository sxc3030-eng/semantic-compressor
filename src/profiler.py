"""Profileur statistique d'un DataFrame pandas.

Pour chaque colonne, on extrait un `ColumnProfile` (cf. `models.py`) qui contient :
- le type semantique (NUMERIC, CATEGORICAL, DATETIME, STRING, BOOLEAN)
- les comptes (n_total, n_unique, n_null)
- des statistiques typees (NumericStats / CategoricalStats / DatetimeStats)
- jusqu'a 5 exemples non-null serialises en types JSON-safe
- la detection d'un pattern regex (email / phone / url / uuid) pour les colonnes string
- l'eligibilite "ancre" (cardinalite 1.0 et zero null)

Aucune detection de distribution ou de correlation ici : c'est le job de `pattern_detector.py`.

Ce module expose aussi `generate_html_report` : un wrapper minimal sur ydata-profiling
qui sert au debug humain.
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .models import (
    CategoricalStats,
    ColumnProfile,
    ColumnType,
    DatetimeStats,
    NumericStats,
    RegexPattern,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes module-level : regex compilees une seule fois.
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: dict[RegexPattern, re.Pattern[str]] = {
    RegexPattern.EMAIL: re.compile(r"^[\w\.\-+]+@[\w\.\-]+\.\w{2,}$"),
    RegexPattern.URL: re.compile(r"^https?://[^\s]+$"),
    RegexPattern.UUID: re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),
    RegexPattern.PHONE: re.compile(r"^[\+\d][\d\s\-\(\)\.]{6,}$"),
}

# Ordre de priorite a la detection : on cherche d'abord les patterns les plus
# specifiques. EMAIL et UUID sont tres specifiques, URL aussi. PHONE est large
# (n'importe quelle chaine de chiffres avec des separateurs) donc en dernier.
_REGEX_PRIORITY: tuple[RegexPattern, ...] = (
    RegexPattern.EMAIL,
    RegexPattern.URL,
    RegexPattern.UUID,
    RegexPattern.PHONE,
)

# Seuils pour la classification de colonne et la detection de pattern.
_CATEGORICAL_RATIO_THRESHOLD = 0.05
_REGEX_MATCH_THRESHOLD = 0.95
_REGEX_SAMPLE_SIZE = 100
_EXAMPLES_SIZE = 5
_TOP_CATEGORIES = 20

# Seuil pour decider si une serie object est en realite du datetime ISO :
# on accepte la conversion si >= 95% des valeurs non-null parsent.
_DATETIME_PARSE_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Detection de type
# ---------------------------------------------------------------------------


def detect_column_type(series: pd.Series) -> ColumnType:
    """Detecte le type semantique d'une colonne, decouple du dtype pandas brut.

    Regles de priorite :
    1. dtype bool -> BOOLEAN
    2. dtype numerique -> NUMERIC
    3. dtype datetime ou colonne object dont >= 95% des non-null parsent en datetime -> DATETIME
    4. dtype object/string avec ratio cardinalite < 0.05 -> CATEGORICAL
    5. sinon -> STRING
    """
    # 1) Booleens (avant le check numerique : pandas considere bool comme integer-like).
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN

    # 2) Numerique (int et float).
    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERIC

    # 3) Datetime natif.
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    # 3 bis) Datetime stocke en string : on tente la conversion sur les non-null.
    non_null = series.dropna()
    if len(non_null) > 0 and _looks_like_datetime(non_null):
        return ColumnType.DATETIME

    # 4) String avec faible cardinalite -> categoriel.
    n_total = len(series)
    n_unique = int(series.nunique(dropna=True))
    if n_total > 0:
        ratio = n_unique / n_total
        if ratio < _CATEGORICAL_RATIO_THRESHOLD:
            return ColumnType.CATEGORICAL

    # 5) Default : string libre.
    return ColumnType.STRING


def _looks_like_datetime(non_null_series: pd.Series) -> bool:
    """True si une serie non-null parse en datetime a >= 95%.

    On echantillonne pour eviter de payer un parse complet sur des colonnes
    string libres qui n'ont rien d'un datetime.
    """
    if not pd.api.types.is_object_dtype(non_null_series) and not pd.api.types.is_string_dtype(
        non_null_series
    ):
        return False

    sample = non_null_series
    if len(sample) > 200:
        sample = sample.sample(n=200, random_state=0)

    # Premier filtre rapide : un datetime ressemble a une string, pas a une autre structure.
    if not sample.map(lambda v: isinstance(v, str)).all():
        return False

    try:
        with warnings.catch_warnings():
            # Pandas verbeux quand le format ISO n'est pas explicite : OK pour nous.
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    except (ValueError, TypeError):
        return False

    n_parsed = int(parsed.notna().sum())
    return n_parsed / len(sample) >= _DATETIME_PARSE_THRESHOLD


# ---------------------------------------------------------------------------
# Detection de pattern regex
# ---------------------------------------------------------------------------


def detect_regex_pattern(series: pd.Series) -> RegexPattern:
    """Detecte si une colonne string matche un pattern connu.

    On echantillonne max 100 valeurs non-null. Un pattern matche si >= 95%
    des valeurs echantillonnees passent. On retourne le premier qui matche
    dans l'ordre EMAIL > URL > UUID > PHONE.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return RegexPattern.NONE

    sample = non_null
    if len(sample) > _REGEX_SAMPLE_SIZE:
        sample = sample.sample(n=_REGEX_SAMPLE_SIZE, random_state=0)

    # On force en str pour eviter qu'une valeur non-string fasse exploser le match.
    str_sample = sample.astype(str)
    n_sample = len(str_sample)

    for pattern_name in _REGEX_PRIORITY:
        compiled = _REGEX_PATTERNS[pattern_name]
        match_count = int(str_sample.map(lambda v, _r=compiled: bool(_r.match(v))).sum())
        if match_count / n_sample >= _REGEX_MATCH_THRESHOLD:
            return pattern_name

    return RegexPattern.NONE


# ---------------------------------------------------------------------------
# Sous-routines de stats par type
# ---------------------------------------------------------------------------


def _compute_numeric_stats(series: pd.Series) -> NumericStats:
    """Calcule mean/std/min/max/quartiles/skewness/kurtosis sur une serie numerique."""
    clean = series.dropna()
    if len(clean) == 0:
        # Tout est null : on remplit avec des zeros pour respecter le schema.
        return NumericStats(
            mean=0.0, std=0.0, min=0.0, max=0.0, q25=0.0, q50=0.0, q75=0.0
        )

    desc = clean.describe()

    # scipy renvoie nan si n < 2 ou si tous les elements sont identiques.
    skew_val: float | None
    kurt_val: float | None
    if len(clean) >= 3 and clean.nunique() > 1:
        skew_val = float(scipy_stats.skew(clean.values, bias=False))
        kurt_val = float(scipy_stats.kurtosis(clean.values, bias=False))
    else:
        skew_val = None
        kurt_val = None

    return NumericStats(
        mean=float(desc["mean"]),
        std=float(desc["std"]) if not pd.isna(desc["std"]) else 0.0,
        min=float(desc["min"]),
        max=float(desc["max"]),
        q25=float(desc["25%"]),
        q50=float(desc["50%"]),
        q75=float(desc["75%"]),
        skewness=skew_val,
        kurtosis=kurt_val,
    )


def _compute_categorical_stats(series: pd.Series) -> CategoricalStats:
    """Top-20 frequences. Les cles sont serialisees en str pour rester JSON-safe."""
    counts = series.value_counts(dropna=True).head(_TOP_CATEGORIES)
    top_values = {str(k): int(v) for k, v in counts.items()}
    n_categories = int(series.nunique(dropna=True))
    return CategoricalStats(top_values=top_values, n_categories=n_categories)


def _compute_datetime_stats(series: pd.Series) -> DatetimeStats:
    """Min/max/range_days. Force la conversion en datetime si necessaire."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        parsed = series

    clean = parsed.dropna()
    if len(clean) == 0:
        # Serie entierement null : on retombe sur un range degnere autour de l'epoch
        # plutot que de lever (le profil reste valide et le schema impose min <= max).
        epoch = datetime(1970, 1, 1)
        return DatetimeStats(min=epoch, max=epoch, range_days=0.0)

    min_ts = clean.min()
    max_ts = clean.max()

    # Pydantic accepte pd.Timestamp si on le convertit en datetime "vanille" python.
    # On vire la timezone (Pydantic v2 accepte les datetime timezone-aware mais le
    # passage par tz_localize(None) garantit un comportement homogene quel que soit
    # le fuseau d'entree).
    min_dt = _to_naive_datetime(min_ts)
    max_dt = _to_naive_datetime(max_ts)
    range_days = float((max_ts - min_ts).total_seconds() / 86400.0)

    return DatetimeStats(min=min_dt, max=max_dt, range_days=range_days)


def _to_naive_datetime(ts: pd.Timestamp) -> datetime:
    """Convertit un Timestamp pandas en datetime python naive."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


# ---------------------------------------------------------------------------
# Helpers d'exemples JSON-safe
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Convertit une valeur en type primitif JSON-serialisable.

    pandas / numpy renvoient des scalars (np.int64, pd.Timestamp, etc.) qui ne sont
    pas sereniement serialisables. On les ramene a des types Python natifs.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        # ISO string : portable, sans dependance au tz local.
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        # np.int64 / np.float64 / np.bool_ -> int / float / bool
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    # Fallback prudent : on convertit en str plutot que de risquer une erreur de
    # serialisation en aval (Pydantic acceptera dans tous les cas une str ici).
    return str(value)


def _sample_examples(series: pd.Series, n: int = _EXAMPLES_SIZE) -> list[Any]:
    """Echantillonne jusqu'a n exemples non-null, serialises JSON-safe."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    take = min(n, len(non_null))
    # random_state fige pour rendre les profils reproductibles run-to-run.
    sampled = non_null.sample(n=take, random_state=0).tolist()
    return [_json_safe(v) for v in sampled]


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------


def profile_dataframe(df: pd.DataFrame, table_name: str = "table") -> list[ColumnProfile]:
    """Profil statistique complet, une entree par colonne du DataFrame.

    Args:
        df: DataFrame a profiler.
        table_name: nom logique de la table, utilise pour le logging uniquement.

    Returns:
        Liste de `ColumnProfile`, dans l'ordre des colonnes du DataFrame.
    """
    logger.info("profile_dataframe start: table=%s rows=%d cols=%d", table_name, len(df), len(df.columns))

    profiles: list[ColumnProfile] = []
    for col_name in df.columns:
        profile = _profile_column(col_name, df[col_name])
        logger.debug(
            "column %s -> type=%s n_unique=%d n_null=%d anchor=%s pattern=%s",
            profile.name,
            profile.column_type.value,
            profile.n_unique,
            profile.n_null,
            profile.is_anchor_candidate,
            profile.regex_pattern.value,
        )
        profiles.append(profile)

    logger.info("profile_dataframe done: table=%s profiles=%d", table_name, len(profiles))
    return profiles


def _profile_column(name: str, series: pd.Series) -> ColumnProfile:
    """Construit le ColumnProfile d'une colonne unique."""
    n_total = int(len(series))
    n_null = int(series.isnull().sum())
    n_unique = int(series.nunique(dropna=True))
    column_type = detect_column_type(series)
    examples = _sample_examples(series)

    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None
    regex_pattern = RegexPattern.NONE

    if column_type == ColumnType.NUMERIC:
        numeric_stats = _compute_numeric_stats(series)
    elif column_type == ColumnType.CATEGORICAL:
        categorical_stats = _compute_categorical_stats(series)
    elif column_type == ColumnType.DATETIME:
        datetime_stats = _compute_datetime_stats(series)
    elif column_type == ColumnType.STRING:
        regex_pattern = detect_regex_pattern(series)
    elif column_type == ColumnType.BOOLEAN:
        # Pour les booleens on garde simplement les comptages True/False : utile
        # au reconstructor pour reproduire le ratio. On reutilise CategoricalStats.
        categorical_stats = _compute_categorical_stats(series)

    is_anchor_candidate = (n_unique == n_total) and (n_null == 0) and (n_total > 0)

    return ColumnProfile(
        name=str(name),
        column_type=column_type,
        dtype=str(series.dtype),
        n_unique=n_unique,
        n_null=n_null,
        n_total=n_total,
        examples=examples,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        datetime_stats=datetime_stats,
        regex_pattern=regex_pattern,
        is_anchor_candidate=is_anchor_candidate,
    )


# ---------------------------------------------------------------------------
# Rapport HTML (ydata-profiling)
# ---------------------------------------------------------------------------


def generate_html_report(
    df: pd.DataFrame, output_path: Path, minimal: bool = True
) -> Path:
    """Genere un rapport HTML d'exploration via ydata-profiling.

    Args:
        df: DataFrame source.
        output_path: chemin du fichier HTML a ecrire.
        minimal: mode minimal de ydata (plus rapide, moins de stats).

    Returns:
        Le `output_path` apres ecriture.

    Raises:
        RuntimeError: si ydata-profiling echoue (compatibilite Python recente, etc.).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Import differe : ydata-profiling pese et n'est pas necessaire au profilage
        # statistique strict, seulement pour le rapport HTML humain.
        from ydata_profiling import ProfileReport
    except ImportError as exc:
        raise RuntimeError(
            "ydata-profiling n'est pas disponible : installer avec `pip install ydata-profiling`"
        ) from exc

    logger.info("generate_html_report: %s (minimal=%s)", output_path, minimal)
    try:
        report = ProfileReport(df, minimal=minimal, progress_bar=False)
        report.to_file(output_path)
    except Exception as exc:
        # On choisit raise plutot que silencieux : le POC veut un signal clair en cas
        # d'echec du profiler exterieur.
        logger.exception("ydata-profiling a echoue : %s", exc)
        raise RuntimeError(f"ydata-profiling failed: {exc}") from exc

    return output_path


__all__ = [
    "profile_dataframe",
    "detect_column_type",
    "detect_regex_pattern",
    "generate_html_report",
]
