"""Detection des patterns : dependances fonctionnelles, distributions, correlations.

Ce module est le coeur intellectuel du semantic-compressor. A partir d'un DataFrame
et de ses profils par colonne, il produit une liste de `Pattern` qui decrivent
comment regenerer chaque colonne :

- Colonnes ancres (`is_anchor_candidate`) => ANCHOR_DIRECT (lecture directe depuis le parquet)
- Colonnes deductibles d'une autre via mapping bijectif => FUNCTIONAL_DEP
- Colonnes correlees a une autre (num-num, cat-cat, num-cat) => CONDITIONAL_DISTRIBUTION
- Colonnes independantes => DISTRIBUTION (normal / exponential / uniform / power_law)

Le module est volontairement deterministe : pas de seed interne, pas de cache.
La reproductibilite vient de l'amont (anchors) et de l'aval (reconstructor).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from .models import (
    ColumnProfile,
    ColumnType,
    Correlation,
    DistributionType,
    FunctionalDependency,
    Pattern,
    PatternType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes / petits helpers numeriques
# ---------------------------------------------------------------------------

_SHAPIRO_MAX = 5000  # scipy.stats.shapiro est plafonne a ~5000 observations.
_MIN_SAMPLE_FOR_FIT = 8  # En-dessous on ne tente meme pas un fit.
_DATETIME_UNIT_KEY = "seconds_since_epoch"


def _jsonable(value: Any) -> Any:
    """Convertit numpy scalars / arrays en types JSON-safe (float, int, list, dict)."""
    if isinstance(value, (np.floating, np.integer)):
        v = value.item()
        # Remplace NaN / Inf par None pour passer le serialiseur JSON strict.
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convertit en float fini, fallback sur default si NaN/Inf/None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


# ---------------------------------------------------------------------------
# 1. Dependances fonctionnelles
# ---------------------------------------------------------------------------


def detect_functional_dependencies(
    df: pd.DataFrame,
    max_cardinality_ratio: float = 0.5,
    min_support: int = 2,
) -> list[FunctionalDependency]:
    """Detecte les dependances A -> B (chaque valeur de A mappe vers UNE seule valeur de B).

    Pour chaque paire ordonnee (A, B) avec A != B :
    - on filtre A si sa cardinalite depasse `max_cardinality_ratio` (sinon la dependance
      est trivialement vraie sur n'importe quelle colonne unique)
    - on filtre A si A a moins de `min_support` valeurs distinctes
    - on groupe par A et on verifie que chaque groupe a exactement 1 valeur de B
    - si une dependance A -> B est detectee, on essaie aussi B -> A pour marquer `is_bijection`

    Complexite : O(n_cols^2 * n_rows) au pire, mitige par le filtre de cardinalite.
    """
    logger.info("detect_functional_dependencies: %d cols, %d rows", df.shape[1], df.shape[0])
    n_rows = len(df)
    if n_rows == 0:
        return []

    # Pre-compute cardinality and dropna-clean series for each candidate column.
    candidates: dict[str, pd.Series] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        n_unique = series.nunique()
        if n_unique < min_support:
            continue
        if n_unique / n_rows > max_cardinality_ratio:
            continue
        candidates[col] = series

    logger.info(
        "  %d/%d cols pass cardinality filter for FD determinant role",
        len(candidates),
        df.shape[1],
    )

    deps: list[FunctionalDependency] = []
    seen_pairs: set[tuple[str, str]] = set()

    for a, series_a in candidates.items():
        # On utilise tout df pour B (les colonnes non-candidates peuvent etre dependees).
        for b in df.columns:
            if a == b:
                continue
            if (a, b) in seen_pairs:
                continue
            # On ne garde que les lignes ou A et B sont non-null.
            paired = df[[a, b]].dropna()
            if len(paired) < min_support:
                continue
            grouped = paired.groupby(a, observed=True)[b].nunique()
            if (grouped <= 1).all():
                # A -> B est une dependance fonctionnelle.
                mapping_size = int(grouped.size)
                # Verifie aussi le sens inverse pour marquer une bijection.
                inverse = paired.groupby(b, observed=True)[a].nunique()
                is_bijection = bool((inverse <= 1).all())
                deps.append(
                    FunctionalDependency(
                        determinant=a,
                        dependent=b,
                        is_bijection=is_bijection,
                        mapping_size=mapping_size,
                    )
                )
                seen_pairs.add((a, b))
                if is_bijection:
                    # Evite de re-detecter B -> A juste apres.
                    seen_pairs.add((b, a))

    logger.info("  %d functional dependencies detected", len(deps))
    return deps


# ---------------------------------------------------------------------------
# 2. Detection de distributions (univariees)
# ---------------------------------------------------------------------------


def _to_seconds_since_epoch(series: pd.Series) -> tuple[np.ndarray, bool]:
    """Convertit une serie datetime (ou string convertible) en float (Unix seconds).

    Retourne (array, was_datetime_string). was_datetime_string indique si la conversion
    a necessite pd.to_datetime (utile pour signaler au reconstructor le format de retour).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        ts = series.astype("int64") / 1e9
        return ts.dropna().to_numpy(), False
    # Tente la conversion depuis string ISO.
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    if converted.notna().sum() == 0:
        raise ValueError("series is not convertible to datetime")
    # Strip timezone for int64 conversion (UTC naive).
    converted = converted.dt.tz_convert(None) if converted.dt.tz is not None else converted
    ts = converted.astype("int64") / 1e9
    return ts.dropna().to_numpy(), True


def _fit_normal(arr: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Fit normal distribution, retourne (fit_quality, params). fit_quality dans [0, 1]."""
    mu, sigma = stats.norm.fit(arr)
    if sigma == 0:
        # Degenere : tous identiques. Match parfait formel.
        return 1.0, {"mean": float(mu), "std": 0.0}
    # KS-statistic (D) : plus petit = meilleur. On le convertit en score [0,1].
    ks = stats.kstest(arr, "norm", args=(mu, sigma))
    quality = 1.0 - float(ks.statistic)
    # On peut aussi tenter Shapiro si la taille le permet, et garder le meilleur des deux.
    if len(arr) <= _SHAPIRO_MAX:
        try:
            shap_stat, _ = stats.shapiro(arr)
            # Shapiro stat est dans [0, 1], proche de 1 = normalite. On en prend le max.
            quality = max(quality, float(shap_stat))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Shapiro failed: %s", exc)
    return _safe_float(quality), {"mean": float(mu), "std": float(sigma)}


def _fit_exponential(arr: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Fit exponential. scipy.stats.expon.fit estime (loc, scale), avec loc = min de la PDF.

    On laisse scipy decider du `loc` (qui sera typiquement arr.min()), puis on teste
    aussi la version "miroir" (arr.max() - arr) car certaines exponentielles sont
    decroissantes (ex: last_login skew vers les dates recentes). On garde la meilleure.
    """
    best_q = 0.0
    best_params: dict[str, Any] = {"loc": float(arr.min()), "scale": 1.0, "reversed": False}

    # Sens normal (croissant).
    loc, scale = stats.expon.fit(arr)
    if scale > 0 and math.isfinite(scale):
        ks = stats.kstest(arr, "expon", args=(loc, scale))
        q = 1.0 - float(ks.statistic)
        if q > best_q:
            best_q = q
            best_params = {"loc": float(loc), "scale": float(scale), "reversed": False}

    # Sens inverse (decroissant) : on miroir.
    arr_mirrored = arr.max() - arr
    loc_m, scale_m = stats.expon.fit(arr_mirrored)
    if scale_m > 0 and math.isfinite(scale_m):
        ks = stats.kstest(arr_mirrored, "expon", args=(loc_m, scale_m))
        q = 1.0 - float(ks.statistic)
        if q > best_q:
            best_q = q
            best_params = {
                "loc": float(loc_m),
                "scale": float(scale_m),
                "reversed": True,
                "mirror_max": float(arr.max()),
            }
    return _safe_float(best_q), best_params


def _fit_uniform(arr: np.ndarray) -> tuple[float, dict[str, Any]]:
    loc, scale = stats.uniform.fit(arr)
    if scale <= 0:
        return 0.0, {"loc": float(loc), "scale": 0.0}
    ks = stats.kstest(arr, "uniform", args=(loc, scale))
    quality = 1.0 - float(ks.statistic)
    return _safe_float(quality), {"loc": float(loc), "scale": float(scale)}


def _fit_powerlaw(arr: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Fit power law (scipy.stats.powerlaw). Demande des valeurs dans (0, 1] apres scaling.

    On penalise les fits ou `a` est trop extreme (typiquement >100 ou <0.01) car ils
    indiquent en realite que la donnee est presque uniforme/deterministe -- le fit
    est techniquement "bon" mais sans contenu informationnel utile.
    """
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return 0.0, {"a": 1.0, "loc": float(arr_min), "scale": 1.0}
    try:
        a, loc, scale = stats.powerlaw.fit(arr)
        if scale <= 0 or a <= 0 or not math.isfinite(a) or not math.isfinite(scale):
            return 0.0, {"a": 1.0, "loc": float(arr_min), "scale": float(arr_max - arr_min)}
        ks = stats.kstest(arr, "powerlaw", args=(a, loc, scale))
        quality = 1.0 - float(ks.statistic)
        # Penalisation : un `a` trop extreme indique un fit "degenere" qui colle a la donnee
        # mais ne represente pas une vraie loi de puissance.
        if a > 50 or a < 0.02:
            quality *= 0.5
        return _safe_float(quality), {"a": float(a), "loc": float(loc), "scale": float(scale)}
    except Exception as exc:
        logger.debug("powerlaw fit failed: %s", exc)
        return 0.0, {"a": 1.0, "loc": float(arr_min), "scale": float(arr_max - arr_min)}


def _best_continuous_distribution(arr: np.ndarray) -> dict[str, Any]:
    """Teste plusieurs distributions et garde la meilleure (fit_quality max)."""
    candidates: list[tuple[DistributionType, float, dict[str, Any]]] = []

    q, p = _fit_normal(arr)
    candidates.append((DistributionType.NORMAL, q, p))

    # Exponential demande arr >= 0 (le fit shift via loc, mais autant filtrer si trop negatif).
    q, p = _fit_exponential(arr)
    candidates.append((DistributionType.EXPONENTIAL, q, p))

    q, p = _fit_uniform(arr)
    candidates.append((DistributionType.UNIFORM, q, p))

    q, p = _fit_powerlaw(arr)
    candidates.append((DistributionType.POWER_LAW, q, p))

    # Tri stable : meilleure qualite en tete.
    candidates.sort(key=lambda t: t[1], reverse=True)
    best_type, best_q, best_params = candidates[0]
    return {
        "distribution": best_type,
        "params": _jsonable(best_params),
        "fit_quality": float(best_q),
    }


def detect_distributions(
    df: pd.DataFrame,
    profiles: list[ColumnProfile] | None = None,
) -> dict[str, dict]:
    """Pour chaque colonne numerique ou datetime convertible, teste les distributions.

    Retourne un dict { column_name: { 'distribution': DistributionType,
                                       'params': {...}, 'fit_quality': float } }.

    Strategie:
    - colonnes profilees comme NUMERIC ou DATETIME (ou auto-detect si pas de profil) : fit
    - sample a `_SHAPIRO_MAX` pour Shapiro, mais utilise toute la serie pour KS
    - datetimes converties en secondes Unix ; on injecte `{'unit': 'seconds_since_epoch'}`
      dans les params pour que le reconstructor sache reconvertir
    """
    logger.info("detect_distributions: %d cols, %d rows", df.shape[1], df.shape[0])
    profile_index = {p.name: p for p in (profiles or [])}
    out: dict[str, dict] = {}

    for col in df.columns:
        profile = profile_index.get(col)
        series = df[col].dropna()
        if len(series) < _MIN_SAMPLE_FOR_FIT:
            continue

        col_type: ColumnType | None = profile.column_type if profile else None
        is_datetime_like = False
        try_fit = False
        arr: np.ndarray | None = None
        is_datetime_string = False

        if col_type == ColumnType.NUMERIC or (
            col_type is None
            and pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            try_fit = True
            arr = series.astype(float).to_numpy()
        elif col_type == ColumnType.DATETIME or pd.api.types.is_datetime64_any_dtype(series):
            try_fit = True
            is_datetime_like = True
            arr, _ = _to_seconds_since_epoch(series)
        elif col_type in (None, ColumnType.STRING):
            # Tente la conversion datetime opportuniste (les strings ISO sont frequents).
            try:
                arr, is_datetime_string = _to_seconds_since_epoch(series)
                if is_datetime_string:
                    try_fit = True
                    is_datetime_like = True
            except Exception:
                try_fit = False

        if not try_fit or arr is None or len(arr) < _MIN_SAMPLE_FOR_FIT:
            continue

        # NaN-safe et fini-safe.
        arr = arr[np.isfinite(arr)]
        if len(arr) < _MIN_SAMPLE_FOR_FIT:
            continue

        # Pour Shapiro on echantillonne, mais on garde tout pour les autres tests.
        if len(arr) > _SHAPIRO_MAX:
            rng = np.random.default_rng(0)
            arr_shapiro_sample = rng.choice(arr, _SHAPIRO_MAX, replace=False)
        else:
            arr_shapiro_sample = arr

        # Note : on fit sur l'array complet (KS), Shapiro uniquement sur le sample.
        # Pour simplifier le _fit_normal on ne lui passe que le sample et il decide.
        # Ici, on prefere fit sur arr complet pour KS, et juste rajouter Shapiro sur sample.
        result = _best_continuous_distribution(arr)

        # Override Shapiro avec le sample taille acceptable
        if result["distribution"] == DistributionType.NORMAL and len(arr_shapiro_sample) <= _SHAPIRO_MAX:
            try:
                shap_stat, _ = stats.shapiro(arr_shapiro_sample)
                # Combine : on garde le max (Shapiro proche de 1 = normal).
                result["fit_quality"] = max(result["fit_quality"], float(shap_stat))
            except Exception:
                pass

        if is_datetime_like:
            params = dict(result["params"])
            params["unit"] = _DATETIME_UNIT_KEY
            params["from_string"] = bool(is_datetime_string)
            result["params"] = params

        out[col] = result
        logger.debug(
            "  %s: %s fit_quality=%.3f", col, result["distribution"].value, result["fit_quality"]
        )

    logger.info("  %d distributions detected", len(out))
    return out


# ---------------------------------------------------------------------------
# 3. Cramer's V et correlations
# ---------------------------------------------------------------------------


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V entre deux series categorielles. Retourne un float dans [0, 1].

    On utilise la formule standard sans correction de biais (suffisant pour le POC).
    """
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        return 0.0
    contingency = pd.crosstab(aligned.iloc[:, 0], aligned.iloc[:, 1])
    if contingency.size == 0:
        return 0.0
    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency)
    except ValueError:
        return 0.0
    n = contingency.values.sum()
    r, k = contingency.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    v = float(np.sqrt(chi2 / denom))
    # Clamp [0, 1] : la formule peut donner > 1 sur des contingences degenrees.
    return max(0.0, min(1.0, v))


def _eta_squared(num: pd.Series, cat: pd.Series) -> float:
    """Effect size eta^2 d'une ANOVA num ~ cat. Retourne float dans [0, 1]."""
    paired = pd.concat([num, cat], axis=1).dropna()
    if paired.empty:
        return 0.0
    n_col = paired.columns[0]
    c_col = paired.columns[1]
    groups = [g[n_col].to_numpy() for _, g in paired.groupby(c_col, observed=True)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0
    grand_mean = paired[n_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = float(((paired[n_col] - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


def _is_numeric_profile(profile: ColumnProfile | None, series: pd.Series) -> bool:
    if profile is not None:
        return profile.column_type == ColumnType.NUMERIC
    # bool est techniquement numerique en pandas, mais on le traite comme categoriel.
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _is_categorical_profile(profile: ColumnProfile | None, series: pd.Series) -> bool:
    if profile is not None:
        return profile.column_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    if pd.api.types.is_bool_dtype(series):
        return True
    return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series)


def detect_correlations(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    threshold: float = 0.3,
) -> list[Correlation]:
    """Detecte les correlations significatives entre paires de colonnes.

    - num vs num : Pearson, |r| >= threshold
    - cat vs cat : Cramer's V >= threshold
    - num vs cat : ANOVA + eta^2, eta^2 >= threshold

    Skip les colonnes anchor_candidate (cardinalite 1.0 => correlations triviales / inutiles).
    Skip paires (A, A).
    """
    logger.info("detect_correlations: %d cols, threshold=%.2f", df.shape[1], threshold)
    profile_index = {p.name: p for p in profiles}

    # Colonnes eligibles : ni ancres, ni manquantes.
    eligible: list[str] = []
    for col in df.columns:
        profile = profile_index.get(col)
        if profile is not None and profile.is_anchor_candidate:
            continue
        if df[col].dropna().empty:
            continue
        eligible.append(col)

    logger.info("  %d/%d cols eligible for correlation analysis", len(eligible), df.shape[1])

    correlations: list[Correlation] = []
    for i, a in enumerate(eligible):
        for b in eligible[i + 1 :]:
            profile_a = profile_index.get(a)
            profile_b = profile_index.get(b)
            series_a = df[a]
            series_b = df[b]
            a_is_num = _is_numeric_profile(profile_a, series_a)
            b_is_num = _is_numeric_profile(profile_b, series_b)
            a_is_cat = _is_categorical_profile(profile_a, series_a)
            b_is_cat = _is_categorical_profile(profile_b, series_b)

            # On ignore les datetimes ici (le pattern_detector les traite via detect_distributions).
            if not (a_is_num or a_is_cat) or not (b_is_num or b_is_cat):
                continue

            if a_is_num and b_is_num:
                paired = pd.concat([series_a, series_b], axis=1).dropna()
                if len(paired) < 3:
                    continue
                try:
                    r, p = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
                except Exception:
                    continue
                strength = abs(float(r))
                if strength >= threshold:
                    correlations.append(
                        Correlation(
                            col_a=a,
                            col_b=b,
                            correlation_type="pearson",
                            strength=min(strength, 1.0),
                            p_value=_safe_float(p, default=1.0),
                        )
                    )
            elif a_is_cat and b_is_cat:
                v = cramers_v(series_a, series_b)
                if v >= threshold:
                    correlations.append(
                        Correlation(
                            col_a=a,
                            col_b=b,
                            correlation_type="cramers_v",
                            strength=v,
                            p_value=None,
                        )
                    )
            else:
                # num vs cat
                if a_is_num and b_is_cat:
                    num_col, cat_col = a, b
                else:
                    num_col, cat_col = b, a
                paired = pd.concat([df[num_col], df[cat_col]], axis=1).dropna()
                if len(paired) < 3:
                    continue
                groups = [g[num_col].to_numpy() for _, g in paired.groupby(cat_col, observed=True)]
                groups = [g for g in groups if len(g) > 1]
                if len(groups) < 2:
                    continue
                try:
                    f, p = stats.f_oneway(*groups)
                except Exception:
                    continue
                eta2 = _eta_squared(df[num_col], df[cat_col])
                if eta2 >= threshold:
                    correlations.append(
                        Correlation(
                            col_a=num_col,
                            col_b=cat_col,
                            correlation_type="anova",
                            strength=min(eta2, 1.0),
                            p_value=_safe_float(p, default=1.0),
                        )
                    )

    logger.info("  %d correlations detected", len(correlations))
    return correlations


# ---------------------------------------------------------------------------
# 4. Distributions conditionnelles (bucketing)
# ---------------------------------------------------------------------------


def _summarize_numeric_bucket(arr: np.ndarray) -> dict[str, Any]:
    """Resume statistique d'un bucket numerique : on essaie normal puis uniform, garde le meilleur."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < _MIN_SAMPLE_FOR_FIT:
        # Trop peu : on retourne juste l'empirical (min, max, mean) en empirical fallback.
        return {
            "distribution": DistributionType.EMPIRICAL.value,
            "params": _jsonable({
                "values": arr.tolist(),
            }),
        }
    candidates: list[tuple[str, float, dict[str, Any]]] = []
    q, p = _fit_normal(arr)
    candidates.append((DistributionType.NORMAL.value, q, p))
    q, p = _fit_uniform(arr)
    candidates.append((DistributionType.UNIFORM.value, q, p))
    candidates.sort(key=lambda t: t[1], reverse=True)
    name, _, params = candidates[0]
    return {"distribution": name, "params": _jsonable(params)}


def _summarize_categorical_bucket(series: pd.Series) -> dict[str, Any]:
    """Resume d'un bucket categoriel : table de frequences normalisees."""
    cleaned = series.dropna()
    if cleaned.empty:
        return {
            "distribution": DistributionType.CATEGORICAL_FREQ.value,
            "frequencies": {},
        }
    freqs = cleaned.value_counts(normalize=True)
    return {
        "distribution": DistributionType.CATEGORICAL_FREQ.value,
        "frequencies": _jsonable({str(k): float(v) for k, v in freqs.items()}),
    }


def detect_conditional_distributions(
    df: pd.DataFrame,
    pivot_column: str,
    target_column: str,
    n_buckets: int = 10,
) -> list[dict]:
    """Pour un pivot (num OU cat) et une cible (num ou cat), retourne une liste de buckets.

    Chaque bucket est un dict :
    - pivot numerique : { 'bucket_range': (lo, hi), 'distribution': '...', 'params': {...} }
    - pivot categoriel : { 'bucket_value': value, 'distribution': '...', 'params': {...} ou 'frequencies': {...} }

    Utilise par recipe/reconstructor pour les patterns CONDITIONAL_DISTRIBUTION.
    """
    if pivot_column == target_column:
        raise ValueError("pivot and target must differ")
    sub = df[[pivot_column, target_column]].dropna()
    if sub.empty:
        return []

    pivot_series = sub[pivot_column]
    target_series = sub[target_column]
    # bool est techniquement is_numeric_dtype True mais on le traite comme categoriel.
    target_is_numeric = pd.api.types.is_numeric_dtype(target_series) and not pd.api.types.is_bool_dtype(
        target_series
    )
    pivot_is_numeric = pd.api.types.is_numeric_dtype(pivot_series) and not pd.api.types.is_bool_dtype(
        pivot_series
    )

    buckets: list[dict] = []

    if pivot_is_numeric:
        try:
            qbins = pd.qcut(pivot_series, n_buckets, duplicates="drop")
        except ValueError:
            # Trop peu de valeurs distinctes : on retombe sur un bucket par valeur unique.
            qbins = pivot_series.astype(str)
            pivot_is_numeric = False
        else:
            for interval, group in sub.groupby(qbins, observed=True):
                lo = float(interval.left)
                hi = float(interval.right)
                if target_is_numeric:
                    summary = _summarize_numeric_bucket(group[target_column].to_numpy())
                else:
                    summary = _summarize_categorical_bucket(group[target_column])
                bucket = {"bucket_range": [lo, hi], **summary}
                buckets.append(bucket)
            return buckets

    # Pivot categoriel (ou fallback) : un bucket par valeur unique.
    for value, group in sub.groupby(pivot_column, observed=True):
        if target_is_numeric:
            summary = _summarize_numeric_bucket(group[target_column].to_numpy())
        else:
            summary = _summarize_categorical_bucket(group[target_column])
        bucket = {"bucket_value": _jsonable(value), **summary}
        buckets.append(bucket)
    return buckets


# ---------------------------------------------------------------------------
# 4b. Detection des "random formats" (valeurs uniques mais valeur exacte non
# informative, ex: bcrypt hashes). Pour ces colonnes on ne stocke PAS d'ancre :
# on garde uniquement le format dans la recette et on regenere des valeurs
# conformes au format a la reconstruction. Cf. PatternType.RANDOM_FORMAT.
# ---------------------------------------------------------------------------


def _parse_bcrypt(sample_value: str) -> dict[str, Any]:
    """Parse une chaine bcrypt-like et extrait son format spec.

    Format reconnu : `$bcrypt$<version>$<cost>$<hex_body>$`
    Exemple : `$bcrypt$2b$12$abc...def$` -> prefix=`$bcrypt$2b$12$`, suffix=`$`,
    body_type=hex, body_length=len(hex_body).
    """
    match = re.match(r"^(\$bcrypt\$\d+[a-z]?\$\d+\$)([0-9a-f]+)(\$)$", sample_value)
    if match is None:
        raise ValueError(f"Not a bcrypt-formatted value: {sample_value!r}")
    prefix, body, suffix = match.group(1), match.group(2), match.group(3)
    body_length = len(body)
    # Le prefix contient des caracteres regex speciaux ($, on les escape).
    escaped_prefix = re.escape(prefix)
    escaped_suffix = re.escape(suffix)
    regex = rf"^{escaped_prefix}[0-9a-f]{{{body_length}}}{escaped_suffix}$"
    return {
        "prefix": prefix,
        "suffix": suffix,
        "body_type": "hex",
        "body_length": body_length,
        "regex": regex,
    }


def _parse_hex_hash(sample_value: str) -> dict[str, Any]:
    """Parse une chaine hex (sans prefix/suffix) et extrait son format spec.

    Exemple : `a3b9c2...` (32 a 128 chars hex purs) -> body_type=hex, body_length=len.
    """
    if not re.match(r"^[0-9a-f]+$", sample_value):
        raise ValueError(f"Not a hex-only value: {sample_value!r}")
    body_length = len(sample_value)
    regex = rf"^[0-9a-f]{{{body_length}}}$"
    return {
        "prefix": "",
        "suffix": "",
        "body_type": "hex",
        "body_length": body_length,
        "regex": regex,
    }


#: Regex strict de detection d'un UUID v4 RFC 4122 :
#: - 8 hex / 4 hex / 4 hex (commence par "4" = version 4) / 4 hex (commence par
#:   [89ab] = variant 10xx) / 12 hex.
_UUID_V4_REGEX_STR = (
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

#: Regex compilee pour identifier les colonnes dont le NOM ressemble a un id
#: user-facing (preserve typiquement pour les FK). Ces colonnes restent
#: ANCHOR_DIRECT par defaut meme si leur contenu matche un format aleatoire :
#: la valeur exacte de l'id porte de l'information (correspondance FK).
#: L'utilisateur peut forcer l'auto-detection RANDOM_FORMAT via le flag
#: `aggressive_uuid` (CLI : --aggressive-uuid).
_ID_LIKE_COLUMN_NAME_REGEX = re.compile(
    r"^(id|uuid|uid|pk|.*_id|.*_uuid|.*_uid|.*_pk)$",
    re.IGNORECASE,
)


def _parse_uuid_v4(sample_value: str) -> dict[str, Any]:
    """Parse une chaine UUID v4 (RFC 4122) et extrait son format spec.

    Format attendu : `xxxxxxxx-xxxx-4xxx-[89ab]xxx-xxxxxxxxxxxx` (36 chars,
    minuscules). Le `body_type=uuid_v4` est consomme specialement par
    `generate_random_format_value` (cf. reconstructor) qui regenere des UUIDs
    valides avec les bits de version/variant corrects.
    """
    if not re.match(_UUID_V4_REGEX_STR, sample_value):
        raise ValueError(f"Not a UUID v4 value: {sample_value!r}")
    return {
        "prefix": "",
        "suffix": "",
        "body_type": "uuid_v4",
        "body_length": 36,
        "regex": _UUID_V4_REGEX_STR,
    }


#: Catalogue des formats aleatoires reconnus automatiquement par
#: `detect_random_format`. Une entree par format ; chaque entree contient :
#: - `regex` : pattern de matching utilise pour valider qu'une serie complete
#:   colle a ce format (verifie au moins 95% des valeurs)
#: - `extract_format` : fonction (sample_value) -> format_spec dict
#:
#: UUID v4 est dans ce catalogue, MAIS avec un garde-fou cote `build_patterns`:
#: les colonnes dont le NOM correspond a `_ID_LIKE_COLUMN_NAME_REGEX` (id, uuid,
#: *_id, *_uuid, etc.) restent ANCHOR_DIRECT par defaut (preservation des FK).
#: Pour passer outre, utiliser le flag `aggressive_uuid=True` qui force
#: l'auto-detection meme sur ces colonnes (l'orchestrator l'expose via
#: --aggressive-uuid : on accepte alors que la valeur exacte de l'id change
#: entre runs).
#:
#: Email n'est PAS dans ce catalogue : on le traite via le mecanisme dedie
#: EMAIL_SPLIT (lossless : preserve local_part + domain_dict, voir
#: `detect_email_split`).
KNOWN_RANDOM_FORMATS: dict[str, dict[str, Any]] = {
    "bcrypt_hash": {
        "regex": re.compile(r"^\$bcrypt\$\d+[a-z]?\$\d+\$[0-9a-f]+\$$"),
        "extract_format": _parse_bcrypt,
    },
    "uuid_v4_anchor": {
        # Format strict RFC 4122 v4. La regenaration se fait via uuid_v4 body
        # type qui force les bits version/variant corrects.
        "regex": re.compile(_UUID_V4_REGEX_STR),
        "extract_format": _parse_uuid_v4,
    },
    "hex_hash": {
        # On exige une longueur >= 32 pour eviter de matcher des UUIDs courts
        # ou des prefixes hex. Pas de borne sup pour les SHA-512 etc.
        # Ordre : place APRES uuid_v4 pour que les UUIDs (qui contiennent des
        # tirets) soient testes d'abord et matches par uuid_v4.
        "regex": re.compile(r"^[0-9a-f]{32,128}$"),
        "extract_format": _parse_hex_hash,
    },
}


def detect_random_format(
    series: pd.Series,
    *,
    sample_size: int = 100,
    min_match_ratio: float = 0.95,
) -> dict[str, Any] | None:
    """Detecte si une colonne string matche un format aleatoire connu.

    Echantillonne jusqu'a `sample_size` valeurs non-nulles, teste chaque format
    de `KNOWN_RANDOM_FORMATS`, et retourne le `format_spec` du format qui matche
    au moins `min_match_ratio` (95% par defaut). Retourne None si aucun format
    ne convient.
    """
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    # On garantit un sample stable : si la serie est plus petite que `sample_size`,
    # on prend tout. Sinon on prend les `sample_size` premieres pour reproductibilite.
    sample = cleaned.head(sample_size).astype(str).tolist()
    if not sample:
        return None

    for format_name, spec in KNOWN_RANDOM_FORMATS.items():
        regex: re.Pattern[str] = spec["regex"]
        matches = sum(1 for v in sample if regex.match(v))
        if matches / len(sample) >= min_match_ratio:
            # Format detecte : extrait le format_spec depuis la 1ere valeur qui matche.
            for v in sample:
                if regex.match(v):
                    try:
                        format_spec = spec["extract_format"](v)
                        format_spec["detected_as"] = format_name
                        logger.info(
                            "detect_random_format: column %r matches %r (%d/%d sample matches)",
                            series.name, format_name, matches, len(sample),
                        )
                        return format_spec
                    except ValueError as exc:
                        logger.debug(
                            "  format %r regex matched but extract_format failed: %s",
                            format_name, exc,
                        )
                        break  # passe au format suivant
    return None


# ---------------------------------------------------------------------------
# 4c. Detection des colonnes EMAIL splittables en local_part + domain_dict
# ---------------------------------------------------------------------------


#: Regex permissive pour valider qu'une chaine ressemble a un email :
#: `local_part@domain.tld` avec un seul "@". On ne tente pas d'etre RFC-compliant,
#: juste suffisamment strict pour eviter les faux positifs (URLs, etc.).
_EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def detect_email_split(
    series: pd.Series,
    *,
    min_match_ratio: float = 0.95,
    max_dict_size: int = 256,
) -> dict[str, Any] | None:
    """Detecte si une colonne string peut etre splittee en local_part + domain dict.

    Conditions :
    - >= `min_match_ratio` des valeurs non-nulles matchent `_EMAIL_REGEX`
    - le dictionnaire des domaines distincts a une taille <= `max_dict_size`
      (typique : 5-10 entrees pour un dataset de 10k emails sur 2-3 providers)

    Retourne un dict :
        {
            "separator": "@",
            "domain_dict": ["example.com", "example.net", ...],
        }
    Ou None si les conditions ne sont pas remplies.

    Le `domain_dict` est trie alphabetiquement (deterministe inter-process).
    L'index dans le dict sert ensuite a stocker la colonne `<col>__domain_idx`
    en uint8/uint16 dans le parquet d'ancres.
    """
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    str_series = cleaned.astype(str)
    matches_mask = str_series.str.match(_EMAIL_REGEX, na=False)
    match_ratio = float(matches_mask.mean())
    if match_ratio < min_match_ratio:
        logger.debug(
            "detect_email_split: column %r match_ratio=%.3f < %.3f, skipping",
            series.name, match_ratio, min_match_ratio,
        )
        return None

    # Extrait les domaines via split sur "@". On ne garde que les emails valides.
    matching_emails = str_series[matches_mask]
    # str.rsplit("@", 1) plus robuste si jamais un local_part contenait "@" (rare).
    domains = matching_emails.str.rsplit("@", n=1).str[-1]
    distinct_domains = sorted(domains.unique().tolist())

    if len(distinct_domains) > max_dict_size:
        logger.debug(
            "detect_email_split: column %r has %d distinct domains > max_dict_size=%d, skipping",
            series.name, len(distinct_domains), max_dict_size,
        )
        return None

    logger.info(
        "detect_email_split: column %r matches email pattern "
        "(match_ratio=%.3f, %d distinct domains)",
        series.name, match_ratio, len(distinct_domains),
    )
    return {
        "separator": "@",
        "domain_dict": distinct_domains,
    }


# ---------------------------------------------------------------------------
# 5. Build patterns (pipeline complet)
# ---------------------------------------------------------------------------


def _choose_strongest_correlation(
    target: str,
    correlations: list[Correlation],
    anchor_set: set[str],
    profile_index: dict[str, ColumnProfile] | None = None,
    excluded_sources: set[str] | None = None,
) -> Correlation | None:
    """Choisit la correlation la plus forte impliquant `target` avec un partenaire non-ancre.

    Resolution de la direction (qui depend de qui) : la colonne avec la PLUS HAUTE
    cardinalite est consideree comme source (plus informative), et l'autre comme
    dependante. Cette regle casse les cycles de correlations mutuelles
    (ex: country<->signup_source) et garantit que les colonnes a haute
    cardinalite (typiquement les plus biaisees) conservent leur distribution
    marginale, au lieu d'etre diluees par un conditioning sur une colonne a
    faible cardinalite.

    En cas d'egalite de cardinalite, on departage par ordre lexicographique
    sur le nom (target est dependant si target_name > partner_name).

    Si `profile_index` est None, on retombe sur l'ancien comportement (sans
    resolution de direction) pour compatibilite ascendante.
    """
    excluded = excluded_sources or set()
    target_key: tuple[int, str] | None = None
    if profile_index is not None and target in profile_index:
        target_key = (profile_index[target].n_unique, target)

    best: Correlation | None = None
    for corr in correlations:
        if target == corr.col_a:
            partner = corr.col_b
        elif target == corr.col_b:
            partner = corr.col_a
        else:
            continue
        if partner in anchor_set or partner in excluded:
            continue
        # Resolution de direction par cardinalite : target ne peut etre dependant
        # que si sa cardinalite est <= celle du partenaire (sinon target est la
        # source naturelle, pas le dependant).
        if target_key is not None and partner in profile_index:
            partner_key = (profile_index[partner].n_unique, partner)
            # Strict : target_key doit etre < partner_key.
            if target_key >= partner_key:
                continue
        if best is None or corr.strength > best.strength:
            best = corr
    return best


def build_patterns(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    anchor_columns: list[str],
    manual_random_format_columns: list[str] | None = None,
    aggressive_uuid: bool = False,
) -> list[Pattern]:
    """Pipeline de detection complet : combine FD + distributions + correlations en list[Pattern].

    Resolution des conflits (par ordre de priorite pour chaque colonne) :
    0a. Colonne dans `manual_random_format_columns` OU dans `anchor_columns` ET
        detectee automatiquement comme format aleatoire connu -> RANDOM_FORMAT
        (la valeur n'est PAS stockee comme ancre, on regenere au runtime).
        EXCEPTION : si le format detecte est `uuid_v4_anchor` et que le NOM de
        la colonne matche `_ID_LIKE_COLUMN_NAME_REGEX` (id/uuid/*_id/etc.), on
        IGNORE la detection et on garde la colonne en ANCHOR_DIRECT pour
        preserver la valeur exacte de l'id (FK potentielles). Override via
        `aggressive_uuid=True`.
    0b. Colonne email (auto-detectee) -> EMAIL_SPLIT (encodage lossless, split
        local_part + domain index ; les anchors contiennent 2 colonnes splittees
        au lieu de la colonne email).
    1. Colonne dans `anchor_columns` -> ANCHOR_DIRECT
    2. Dependance fonctionnelle A -> B (B == colonne courante, A != colonne courante) -> FUNCTIONAL_DEP
    3. Correlation forte avec une autre colonne non-ancre -> CONDITIONAL_DISTRIBUTION
    4. Fallback -> DISTRIBUTION univariee (ou CATEGORICAL_FREQ pour les colonnes categorielles)

    Args:
        df: DataFrame source.
        profiles: profils par colonne (un par colonne attendue).
        anchor_columns: liste des colonnes ancres candidates (cardinalite 1.0 ou forcees).
        manual_random_format_columns: liste de colonnes a forcer en RANDOM_FORMAT
            (la detection automatique ne marque PAS ces colonnes par defaut sauf
            si elles matchent un format de `KNOWN_RANDOM_FORMATS`).
        aggressive_uuid: si True, les colonnes dont le nom ressemble a un id
            (id, uuid, *_id, *_uuid, etc.) sont aussi auto-marquees RANDOM_FORMAT
            si leur contenu est un UUID v4. Par defaut False : on preserve la
            valeur exacte de ces colonnes (mode "fidelity-first" pour FK).
    """
    logger.info(
        "build_patterns: %d cols (%d ancres), %d rows, manual_random_format=%s, aggressive_uuid=%s",
        df.shape[1],
        len(anchor_columns),
        df.shape[0],
        manual_random_format_columns or [],
        aggressive_uuid,
    )
    anchor_set = set(anchor_columns)
    manual_random_set = set(manual_random_format_columns or [])
    profile_index = {p.name: p for p in profiles}

    # Pre-detection des colonnes "random format" : on auto-detecte sur les
    # colonnes a haute cardinalite (>= 95% unique), qui sont typiquement des
    # candidates ancres mais pourraient avoir ete exclues du `anchor_columns`
    # entrant (ex: pass 3 de l'orchestrator qui passe `anchor_columns` sans les
    # RANDOM_FORMAT pour eviter le conflit).
    # Les colonnes dans `manual_random_format_columns` sont prises avec detection
    # automatique : si le format n'est pas reconnu on log un warning et on
    # retombe sur le comportement par defaut.
    random_format_specs: dict[str, dict[str, Any]] = {}
    email_split_specs: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        profile = profile_index.get(col)
        # Eligibilite a l'auto-detection : la colonne doit etre unique ou
        # quasi-unique (sinon ce n'est pas un "random format"). Le critere est
        # n_unique / n_total >= 0.95 ; le profil expose `cardinality` pour ca.
        is_high_cardinality = (
            profile is not None
            and profile.n_total > 0
            and (profile.n_unique / profile.n_total) >= 0.95
        )
        name_looks_like_id = bool(_ID_LIKE_COLUMN_NAME_REGEX.match(col))

        # Detection auto sur les colonnes a haute cardinalite (peut etre ancre
        # ou non, peu importe : si ca matche un format connu on le marque).
        if is_high_cardinality or col in anchor_set:
            spec = detect_random_format(df[col])
            if spec is not None:
                # Garde-fou UUID-as-id : si le format detecte est uuid_v4 et que
                # le nom de la colonne ressemble a un id (id, uuid, *_id, ...),
                # on prefere preserver la valeur exacte (FK potentielles).
                # Override via aggressive_uuid=True.
                if (
                    spec.get("detected_as") == "uuid_v4_anchor"
                    and name_looks_like_id
                    and not aggressive_uuid
                ):
                    logger.info(
                        "  column %r matches uuid_v4 format but name looks like id, "
                        "keeping ANCHOR_DIRECT (use aggressive_uuid=True to force RANDOM_FORMAT)",
                        col,
                    )
                else:
                    random_format_specs[col] = spec
                    logger.info("  column %r auto-detected as RANDOM_FORMAT (%s)", col, spec.get("detected_as"))
                    continue
        # Forcage manuel : on tente quand meme de detecter le format pour extraire
        # le spec. Si la detection echoue (format inconnu), on log un warning et
        # on retombe sur le comportement par defaut pour cette colonne.
        if col in manual_random_set:
            spec = detect_random_format(df[col])
            if spec is not None:
                random_format_specs[col] = spec
                logger.info("  column %r forced as RANDOM_FORMAT (%s)", col, spec.get("detected_as"))
                continue
            else:
                logger.warning(
                    "  column %r is in manual_random_format_columns but no known format matches; "
                    "falling back to default behavior",
                    col,
                )

        # Detection EMAIL_SPLIT : on tente sur les colonnes a haute cardinalite
        # qui ne sont PAS deja marquees random_format. La detection verifie
        # que >= 95% des valeurs matchent un format email et que le nombre de
        # domaines distincts est borne (<=256, suffisant pour la plupart des
        # datasets B2C / B2B).
        if is_high_cardinality or col in anchor_set:
            email_spec = detect_email_split(df[col])
            if email_spec is not None:
                email_split_specs[col] = email_spec
                logger.info(
                    "  column %r auto-detected as EMAIL_SPLIT (%d domains: %s)",
                    col, len(email_spec["domain_dict"]),
                    email_spec["domain_dict"][:5],
                )

    # 1. Pre-compute dependances fonctionnelles, distributions, correlations.
    fdeps = detect_functional_dependencies(df)
    # Index : pour chaque dependant, liste de determinants candidats.
    fdep_index: dict[str, list[FunctionalDependency]] = {}
    for fd in fdeps:
        fdep_index.setdefault(fd.dependent, []).append(fd)

    # On utilise un threshold plus bas (0.05) pour les correlations utilisees en pattern building :
    # le seuil de 0.3 de la spec correspond a un *reporting* user-facing, mais pour la generation
    # on veut capturer toute correlation modeste pour ameliorer la fidelite (premium ~ country
    # par exemple a une Cramer's V ~0.06 mais reste informatif).
    correlations = detect_correlations(df, profiles, threshold=0.05)
    distributions = detect_distributions(df, profiles)

    patterns: list[Pattern] = []

    for col in df.columns:
        profile = profile_index.get(col)

        # 0a. Random format : la colonne a une valeur unique mais sa valeur exacte
        # n'est pas informative (ex: bcrypt hash). On stocke uniquement le format
        # dans la recette ; pas d'ancre, pas de cardinalite preservee.
        if col in random_format_specs:
            patterns.append(
                Pattern(
                    column=col,
                    pattern_type=PatternType.RANDOM_FORMAT,
                    format_spec=random_format_specs[col],
                    fidelity_estimate=1.0,
                    dependencies=[],
                )
            )
            continue

        # 0b. EMAIL_SPLIT : encodage lossless. La colonne email est splittee en
        # local_part (ancre texte) + domain_index (ancre uint8 via dictionnaire).
        # La reconstruction recompose `local_part + "@" + domain_dict[idx]`. Le
        # `format_spec.domain_dict` est embarque dans la recette.
        if col in email_split_specs:
            patterns.append(
                Pattern(
                    column=col,
                    pattern_type=PatternType.EMAIL_SPLIT,
                    format_spec=email_split_specs[col],
                    fidelity_estimate=1.0,
                    dependencies=[],
                )
            )
            continue

        # 1. Ancre
        if col in anchor_set:
            patterns.append(
                Pattern(
                    column=col,
                    pattern_type=PatternType.ANCHOR_DIRECT,
                    fidelity_estimate=1.0,
                    dependencies=[],
                )
            )
            continue

        # 2. Dependance fonctionnelle (col est dependee d'une autre).
        fdeps_for_col = fdep_index.get(col, [])
        # On exclut les determinants qui sont col elle-meme.
        valid_fdeps = [fd for fd in fdeps_for_col if fd.determinant != col]
        if valid_fdeps:
            # Choix : determinant prefere = celui avec mapping_size le plus petit (plus compact).
            # En cas d'egalite, on prend l'ordre lexicographique stable.
            valid_fdeps.sort(key=lambda fd: (fd.mapping_size, fd.determinant))
            chosen = valid_fdeps[0]
            # Construit le lookup table.
            paired = df[[chosen.determinant, chosen.dependent]].dropna()
            # On groupe et prend la 1ere valeur (la dep garantit l'unicite).
            mapping = (
                paired.groupby(chosen.determinant, observed=True)[chosen.dependent]
                .first()
                .to_dict()
            )
            mapping_json = {_jsonable(k): _jsonable(v) for k, v in mapping.items()}
            patterns.append(
                Pattern(
                    column=col,
                    pattern_type=PatternType.FUNCTIONAL_DEP,
                    source_column=chosen.determinant,
                    lookup_table=mapping_json,
                    fidelity_estimate=1.0,
                    dependencies=[chosen.determinant],
                )
            )
            continue

        # 3. Correlation conditionnelle (avec resolution de direction par cardinalite)
        best_corr = _choose_strongest_correlation(
            col, correlations, anchor_set, profile_index=profile_index
        )
        if best_corr is not None:
            pivot = best_corr.col_b if best_corr.col_a == col else best_corr.col_a
            try:
                buckets = detect_conditional_distributions(
                    df, pivot_column=pivot, target_column=col
                )
            except ValueError:
                buckets = []
            if buckets:
                patterns.append(
                    Pattern(
                        column=col,
                        pattern_type=PatternType.CONDITIONAL_DISTRIBUTION,
                        source_column=pivot,
                        conditional_buckets=buckets,
                        fidelity_estimate=_safe_float(best_corr.strength, default=0.5),
                        dependencies=[pivot],
                    )
                )
                continue

        # 4. Fallback : distribution univariee
        dist_info = distributions.get(col)
        if dist_info is not None:
            patterns.append(
                Pattern(
                    column=col,
                    pattern_type=PatternType.DISTRIBUTION,
                    distribution=dist_info["distribution"],
                    distribution_params=_jsonable(dist_info["params"]),
                    fidelity_estimate=_safe_float(dist_info["fit_quality"], default=0.5),
                    dependencies=[],
                )
            )
            continue

        # 4b. Categoriel / boolean : table de frequences
        is_cat = profile is not None and profile.column_type in (
            ColumnType.CATEGORICAL,
            ColumnType.BOOLEAN,
        )
        is_bool_series = pd.api.types.is_bool_dtype(df[col])
        if is_cat or is_bool_series or (
            profile is None and not pd.api.types.is_numeric_dtype(df[col])
        ):
            freqs = (
                df[col]
                .dropna()
                .astype(object)
                .value_counts(normalize=True)
            )
            patterns.append(
                Pattern(
                    column=col,
                    pattern_type=PatternType.DISTRIBUTION,
                    distribution=DistributionType.CATEGORICAL_FREQ,
                    distribution_params={
                        "frequencies": _jsonable({str(k): float(v) for k, v in freqs.items()})
                    },
                    fidelity_estimate=1.0,
                    dependencies=[],
                )
            )
            continue

        # 4c. Dernier recours : EMPIRICAL avec sample des valeurs uniques.
        sample_values = df[col].dropna().unique().tolist()[:50]
        patterns.append(
            Pattern(
                column=col,
                pattern_type=PatternType.DISTRIBUTION,
                distribution=DistributionType.EMPIRICAL,
                distribution_params={"values": _jsonable(sample_values)},
                fidelity_estimate=0.5,
                dependencies=[],
            )
        )

    logger.info("  %d patterns built", len(patterns))
    return patterns


__all__ = [
    "detect_functional_dependencies",
    "detect_distributions",
    "cramers_v",
    "detect_correlations",
    "detect_conditional_distributions",
    "detect_random_format",
    "detect_email_split",
    "KNOWN_RANDOM_FORMATS",
    "build_patterns",
]
