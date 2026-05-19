"""Validator du semantic-compressor.

Compare un DataFrame original avec sa reconstruction et produit un `ValidationReport`
agrege. La validation se decompose en deux familles de tests :

- **Structurels (exact)** : compte de lignes, schema, ancres. Une erreur ici est
  critique : on considere ces tests comme des erreurs bloquantes (poids x2 dans
  le score global, voir `_compute_score`).
- **Statistiques (tolerance)** : KS-test sur les distributions, moyennes,
  correlations, frequences categorielles.

API publique :
- `ValidationThresholds`            : configuration des seuils
- `validate(...)`                   : pipeline complet
- `test_row_count(...)`             : test structurel
- `test_schema_match(...)`          : test structurel
- `test_anchors_exact(...)`         : tests structurels (un par ancre)
- `test_distribution_ks(...)`       : KS-test avec relaxation pour skew eleve
- `test_mean_within_tolerance(...)` : moyennes (numerique uniquement)
- `test_correlation_preserved(...)` : pearson / cramer's V / ANOVA selon types
- `test_categorical_frequencies(...)`: frequences par valeur
- `print_report(...)`               : sortie console Rich

Aucune dependance sur les autres modules du POC : ce module est strictement
"compare two DataFrames", il n'a pas connaissance des recettes / ancres
extraites en amont.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats

from .models import ValidationReport, ValidationTestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass
class ValidationThresholds:
    """Seuils de tolerance utilises par la pipeline de validation.

    Les defauts proviennent de `pyproject.toml [tool.semantic_compressor]`.
    """

    ks_pvalue_min: float = 0.05
    mean_tolerance_pct: float = 0.02
    correlation_tolerance: float = 0.05
    categorical_freq_tolerance: float = 0.03
    fidelity_target: float = 0.95
    skew_relax_threshold: float = 2.0  # skewness > 2 -> p > 0.01 plutot que 0.05
    ks_pvalue_min_relaxed: float = 0.01
    # En dessous de ce seuil, on considere la correlation comme statistiquement
    # negligeable (cohen 1988 : |r| < 0.1 = trivial, 0.1-0.3 = faible). La spec
    # SPEC.md utilise 0.3 comme seuil de "significativite" pour la detection.
    # On choisit 0.15 cote validation : assez bas pour catcher les correlations
    # notables, assez haut pour eviter de tester du bruit statistique non
    # encodable proprement (ex: dependances faibles entre colonnes datetime).
    min_correlation_to_check: float = 0.15
    max_correlation_pairs: int = 20
    # A grand n, la p-value KS tend vers 0 meme pour des distributions tres
    # proches. On bascule alors sur le D-statistic (Kolmogorov distance, dans
    # [0,1]), plus interpretable. Au-dessus de ks_large_n_threshold lignes :
    # passe si D < ks_distance_max. Pour les distributions tres skewed
    # (|skew| > skew_relax_threshold), on relache a ks_distance_max_relaxed.
    ks_large_n_threshold: int = 1000
    ks_distance_max: float = 0.05
    ks_distance_max_relaxed: float = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NUMERIC_KINDS = ("i", "u", "f")  # int, unsigned, float


def _is_numeric(series: pd.Series) -> bool:
    return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)


def _is_datetime(series: pd.Series) -> bool:
    """True pour dtype datetime natif OU pour strings ISO datetime parseables.

    Le CSV est lu sans parse_dates pour preserver les types ; les datetimes
    apparaissent en dtype `object`. On detecte ce cas en parsant un echantillon.
    """
    if pdt.is_datetime64_any_dtype(series):
        return True
    if pdt.is_object_dtype(series) or pdt.is_string_dtype(series):
        sample = series.dropna().head(50)
        if sample.empty:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce", utc=True)
        except (ValueError, TypeError):
            return False
        # Seuil 90% : evite les faux positifs sur des strings libres contenant
        # quelques timestamps.
        return float(parsed.notna().mean()) >= 0.9
    return False


def _is_categorical_like(series: pd.Series) -> bool:
    """True pour les colonnes categorielles/booleennes/strings (non datetime)."""
    if _is_datetime(series):
        return False
    if pdt.is_bool_dtype(series):
        return True
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pdt.is_object_dtype(series) or pdt.is_string_dtype(series):
        return True
    return False


def _to_unix_seconds(series: pd.Series) -> pd.Series:
    """Convertit un series datetime en timestamps Unix (secondes), drop NaT."""
    s = pd.to_datetime(series, errors="coerce").dropna()
    # `astype("int64")` sur datetime64[ns] donne des nanosecondes ; on divise.
    return s.astype("int64").astype("float64") / 1e9


def _safe_skew(series: pd.Series) -> float:
    """Calcule la skewness en tolerant les nulls et series degenerees."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3 or s.nunique() < 2:
        return 0.0
    return float(stats.skew(s, bias=False, nan_policy="omit"))


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V — meme implementation que IMPLEMENTATION_NOTES section 1.5."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return 0.0
    contingency = pd.crosstab(df["x"], df["y"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    r, k = contingency.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def _anova_strength(numeric: pd.Series, categorical: pd.Series) -> float:
    """ANOVA -> eta squared (force d'association numerique/categoriel, 0-1)."""
    df = pd.DataFrame({"num": pd.to_numeric(numeric, errors="coerce"), "cat": categorical}).dropna()
    if df.empty:
        return 0.0
    groups = [g["num"].values for _, g in df.groupby("cat") if len(g) >= 2]
    if len(groups) < 2:
        return 0.0
    try:
        f_stat, _ = stats.f_oneway(*groups)
    except Exception:  # pragma: no cover - defensive
        return 0.0
    if not np.isfinite(f_stat) or f_stat <= 0:
        return 0.0
    # Conversion F -> eta^2 approchee : eta^2 = F * dfb / (F * dfb + dfw)
    n = len(df)
    k = len(groups)
    dfb = k - 1
    dfw = n - k
    if dfw <= 0:
        return 0.0
    return float((f_stat * dfb) / (f_stat * dfb + dfw))


def _pair_correlation(
    df: pd.DataFrame, col_a: str, col_b: str
) -> tuple[float, str]:
    """Calcule la force d'association d'une paire (col_a, col_b) selon les types.

    Retourne (strength in [0,1], method_name).
    """
    sa, sb = df[col_a], df[col_b]
    a_num = _is_numeric(sa) or _is_datetime(sa)
    b_num = _is_numeric(sb) or _is_datetime(sb)

    if a_num and b_num:
        x = _to_unix_seconds(sa) if _is_datetime(sa) else pd.to_numeric(sa, errors="coerce")
        y = _to_unix_seconds(sb) if _is_datetime(sb) else pd.to_numeric(sb, errors="coerce")
        paired = pd.concat([x, y], axis=1).dropna()
        if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
            return 0.0, "pearson"
        try:
            r, _ = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        except Exception:  # pragma: no cover - defensive
            return 0.0, "pearson"
        return float(abs(r)) if np.isfinite(r) else 0.0, "pearson"

    if not a_num and not b_num:
        return _cramers_v(sa, sb), "cramers_v"

    # Mixte : numerique-categoriel
    if a_num:
        return _anova_strength(sa, sb), "anova"
    return _anova_strength(sb, sa), "anova"


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_row_count(
    original: pd.DataFrame, reconstructed: pd.DataFrame
) -> ValidationTestResult:
    """Verifie que le nombre de lignes est identique."""
    n_orig = len(original)
    n_recon = len(reconstructed)
    passed = n_orig == n_recon
    logger.debug("row_count: original=%s reconstructed=%s passed=%s", n_orig, n_recon, passed)
    return ValidationTestResult(
        test_name="row_count",
        metric="n_rows",
        expected=n_orig,
        actual=n_recon,
        threshold=None,
        passed=passed,
        details=(
            f"Original has {n_orig} rows, reconstructed has {n_recon}"
            if passed
            else f"Row count mismatch: {n_orig} vs {n_recon} (delta={n_recon - n_orig})"
        ),
    )


def test_schema_match(
    original: pd.DataFrame, reconstructed: pd.DataFrame
) -> ValidationTestResult:
    """Verifie que le set de colonnes est identique (ordre indifferent)."""
    cols_orig = list(original.columns)
    cols_recon = list(reconstructed.columns)
    set_orig = set(cols_orig)
    set_recon = set(cols_recon)
    missing = sorted(set_orig - set_recon)
    extra = sorted(set_recon - set_orig)
    passed = not missing and not extra

    if passed:
        details = f"Schemas match: {len(cols_orig)} columns"
    else:
        parts = []
        if missing:
            parts.append(f"missing in reconstructed: {missing}")
        if extra:
            parts.append(f"extra in reconstructed: {extra}")
        details = "; ".join(parts)

    logger.debug("schema_match: passed=%s details=%s", passed, details)
    return ValidationTestResult(
        test_name="schema_match",
        metric="columns_set",
        expected=sorted(cols_orig),
        actual=sorted(cols_recon),
        threshold=None,
        passed=passed,
        details=details,
    )


def test_anchors_exact(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    anchor_columns: list[str],
) -> list[ValidationTestResult]:
    """Teste que chaque colonne ancre est identique entre original et reconstruction.

    Retourne UN `ValidationTestResult` par colonne ancre. `check_dtype=False` pour
    tolerer int32 vs int64 apres aller-retour parquet.
    """
    results: list[ValidationTestResult] = []
    for col in anchor_columns:
        if col not in original.columns or col not in reconstructed.columns:
            results.append(
                ValidationTestResult(
                    test_name=f"anchor_exact[{col}]",
                    metric="series_equal",
                    expected="present in both",
                    actual=f"orig={col in original.columns}, recon={col in reconstructed.columns}",
                    threshold=None,
                    passed=False,
                    details=f"Anchor column '{col}' is missing in one of the DataFrames",
                )
            )
            continue

        # Si les tailles different on ne peut meme pas comparer ligne a ligne.
        if len(original) != len(reconstructed):
            results.append(
                ValidationTestResult(
                    test_name=f"anchor_exact[{col}]",
                    metric="series_equal",
                    expected=f"len={len(original)}",
                    actual=f"len={len(reconstructed)}",
                    threshold=None,
                    passed=False,
                    details="Cannot compare anchors: row counts differ",
                )
            )
            continue

        try:
            pd.testing.assert_series_equal(
                original[col].reset_index(drop=True),
                reconstructed[col].reset_index(drop=True),
                check_dtype=False,
                check_names=False,
            )
            passed = True
            details = f"Anchor '{col}' matches exactly ({len(original)} rows)"
        except AssertionError as exc:
            passed = False
            # On capture juste la 1ere ligne du message d'erreur pour la lisibilite.
            msg = str(exc).splitlines()[0] if str(exc) else "values differ"
            details = f"Anchor '{col}' mismatch: {msg}"

        logger.debug("anchor_exact[%s]: passed=%s", col, passed)
        results.append(
            ValidationTestResult(
                test_name=f"anchor_exact[{col}]",
                metric="series_equal",
                expected="identical",
                actual="identical" if passed else "different",
                threshold=None,
                passed=passed,
                details=details,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def test_distribution_ks(
    original: pd.Series,
    reconstructed: pd.Series,
    thresholds: ValidationThresholds,
) -> ValidationTestResult:
    """KS-test sur deux echantillons.

    - Skip si la colonne n'est ni numerique ni datetime (renvoyer passed=True avec
      mention "skip").
    - Datetime : conversion en secondes Unix avant ks_2samp.
    - Skewness > seuil : relaxation a p > 0.01.
    """
    col = original.name or "<unnamed>"

    if _is_categorical_like(original) and not _is_datetime(original):
        return ValidationTestResult(
            test_name=f"ks_test[{col}]",
            metric="ks_pvalue",
            expected=">= threshold",
            actual="skipped",
            threshold=None,
            passed=True,
            details="Skipped: column is categorical (use frequency test instead)",
        )

    if _is_datetime(original):
        a = _to_unix_seconds(original)
        b = _to_unix_seconds(reconstructed)
    elif _is_numeric(original):
        a = pd.to_numeric(original, errors="coerce").dropna()
        b = pd.to_numeric(reconstructed, errors="coerce").dropna()
    else:
        return ValidationTestResult(
            test_name=f"ks_test[{col}]",
            metric="ks_pvalue",
            expected=">= threshold",
            actual="skipped",
            threshold=None,
            passed=True,
            details=f"Skipped: dtype {original.dtype} not supported for KS-test",
        )

    if len(a) < 2 or len(b) < 2:
        return ValidationTestResult(
            test_name=f"ks_test[{col}]",
            metric="ks_pvalue",
            expected=">= threshold",
            actual="skipped",
            threshold=None,
            passed=True,
            details=f"Skipped: too few non-null values (orig={len(a)}, recon={len(b)})",
        )

    skew = _safe_skew(a)
    use_relaxed = abs(skew) > thresholds.skew_relax_threshold
    p_threshold = (
        thresholds.ks_pvalue_min_relaxed if use_relaxed else thresholds.ks_pvalue_min
    )

    ks_stat, p_value = stats.ks_2samp(a, b)

    # A grand n, la p-value KS converge vers 0 meme pour des distributions tres
    # proches (le test gagne en puissance avec n). On bascule sur le D-stat,
    # qui mesure la distance reelle entre distributions independamment de n.
    n_min = min(len(a), len(b))
    use_distance_threshold = n_min >= thresholds.ks_large_n_threshold

    if use_distance_threshold:
        d_threshold = (
            thresholds.ks_distance_max_relaxed if use_relaxed
            else thresholds.ks_distance_max
        )
        d_label = "relaxed-skew" if use_relaxed else "standard"
        passed = bool(ks_stat <= d_threshold)
        details = (
            f"KS stat={ks_stat:.4f}, p={p_value:.4f}, D_threshold={d_threshold} "
            f"({d_label} large-n mode, n={n_min}, skew={skew:.2f})"
        )
        actual_val = float(ks_stat)
        threshold_val = float(d_threshold)
        metric_label = "ks_distance"
        expected_label = f"<= {d_threshold}"
    else:
        passed = bool(p_value > p_threshold)
        threshold_label = "relaxed" if use_relaxed else "standard"
        details = (
            f"KS stat={ks_stat:.4f}, p={p_value:.4f}, p_threshold={p_threshold} "
            f"({threshold_label}, n={n_min}, skew={skew:.2f})"
        )
        actual_val = float(p_value)
        threshold_val = float(p_threshold)
        metric_label = "ks_pvalue"
        expected_label = f"> {p_threshold}"

    logger.debug("ks_test[%s]: %s passed=%s", col, details, passed)

    return ValidationTestResult(
        test_name=f"ks_test[{col}]",
        metric=metric_label,
        expected=expected_label,
        actual=actual_val,
        threshold=threshold_val,
        passed=passed,
        details=details,
    )


def test_mean_within_tolerance(
    original: pd.Series,
    reconstructed: pd.Series,
    tolerance_pct: float,
) -> ValidationTestResult:
    """Compare les moyennes en pourcentage relatif.

    - Skip si la colonne n'est pas numerique (renvoie passed=True + mention).
    - Si mean_orig == 0 : on compare |mean_recon| <= tolerance_pct directement.
    """
    col = original.name or "<unnamed>"

    if not _is_numeric(original):
        return ValidationTestResult(
            test_name=f"mean[{col}]",
            metric="mean_relative_diff",
            expected=f"<= {tolerance_pct}",
            actual="skipped",
            threshold=tolerance_pct,
            passed=True,
            details=f"Skipped: column dtype {original.dtype} is not numeric",
        )

    a = pd.to_numeric(original, errors="coerce").dropna()
    b = pd.to_numeric(reconstructed, errors="coerce").dropna()
    if a.empty or b.empty:
        return ValidationTestResult(
            test_name=f"mean[{col}]",
            metric="mean_relative_diff",
            expected=f"<= {tolerance_pct}",
            actual="skipped",
            threshold=tolerance_pct,
            passed=True,
            details="Skipped: empty after NaN drop",
        )

    mean_orig = float(a.mean())
    mean_recon = float(b.mean())

    if mean_orig == 0:
        diff = abs(mean_recon)
        details = (
            f"mean_orig=0; |mean_recon|={diff:.4f} (tolerance={tolerance_pct})"
        )
    else:
        diff = abs(mean_orig - mean_recon) / abs(mean_orig)
        details = (
            f"mean_orig={mean_orig:.4f}, mean_recon={mean_recon:.4f}, "
            f"relative_diff={diff:.4f} (tolerance={tolerance_pct})"
        )

    passed = bool(diff <= tolerance_pct)
    logger.debug("mean[%s]: %s passed=%s", col, details, passed)

    return ValidationTestResult(
        test_name=f"mean[{col}]",
        metric="mean_relative_diff",
        expected=f"<= {tolerance_pct}",
        actual=float(diff),
        threshold=tolerance_pct,
        passed=passed,
        details=details,
    )


def test_correlation_preserved(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    col_a: str,
    col_b: str,
    tolerance: float,
) -> ValidationTestResult:
    """Verifie qu'une correlation (col_a, col_b) est preservee.

    - Numerique-numerique : Pearson
    - Categoriel-categoriel : Cramer's V
    - Mixte : ANOVA -> eta^2

    Si |corr_orig| < min_correlation_to_check (par defaut 0.1), retourne un test
    "skipped" (passed=True) : il n'y a rien a preserver.
    """
    test_name = f"corr[{col_a}~{col_b}]"

    if col_a not in original.columns or col_b not in original.columns:
        return ValidationTestResult(
            test_name=test_name,
            metric="corr_diff",
            expected=f"<= {tolerance}",
            actual="error",
            threshold=tolerance,
            passed=False,
            details=f"Column missing in original: {col_a} or {col_b}",
        )
    if col_a not in reconstructed.columns or col_b not in reconstructed.columns:
        return ValidationTestResult(
            test_name=test_name,
            metric="corr_diff",
            expected=f"<= {tolerance}",
            actual="error",
            threshold=tolerance,
            passed=False,
            details=f"Column missing in reconstructed: {col_a} or {col_b}",
        )

    corr_orig, method = _pair_correlation(original, col_a, col_b)
    corr_recon, _ = _pair_correlation(reconstructed, col_a, col_b)

    if corr_orig < 0.1:
        return ValidationTestResult(
            test_name=test_name,
            metric="corr_diff",
            expected=f"<= {tolerance}",
            actual="skipped",
            threshold=tolerance,
            passed=True,
            details=(
                f"Skipped: original correlation too weak ({method}={corr_orig:.3f} < 0.1), "
                "nothing meaningful to preserve"
            ),
        )

    diff = abs(corr_orig - corr_recon)
    passed = bool(diff <= tolerance)
    details = (
        f"{method}: orig={corr_orig:.4f}, recon={corr_recon:.4f}, "
        f"|diff|={diff:.4f} (tolerance={tolerance})"
    )
    logger.debug("%s: %s passed=%s", test_name, details, passed)

    return ValidationTestResult(
        test_name=test_name,
        metric="corr_diff",
        expected=f"<= {tolerance}",
        actual=float(diff),
        threshold=tolerance,
        passed=passed,
        details=details,
    )


def test_categorical_frequencies(
    original: pd.Series,
    reconstructed: pd.Series,
    tolerance: float,
) -> ValidationTestResult:
    """Verifie que les frequences (proportions) categorielles sont preservees.

    Si le set de valeurs uniques differe (valeurs en plus ou en moins) : test failed.
    Sinon : pour chaque valeur, on verifie |freq_orig - freq_recon| <= tolerance.
    """
    col = original.name or "<unnamed>"

    a = original.dropna()
    b = reconstructed.dropna()

    if a.empty:
        return ValidationTestResult(
            test_name=f"cat_freq[{col}]",
            metric="max_freq_diff",
            expected=f"<= {tolerance}",
            actual="skipped",
            threshold=tolerance,
            passed=True,
            details="Skipped: original is empty",
        )

    freq_a = a.value_counts(normalize=True)
    freq_b = b.value_counts(normalize=True)
    set_a, set_b = set(freq_a.index), set(freq_b.index)

    extra = sorted(set_b - set_a, key=str)
    missing = sorted(set_a - set_b, key=str)

    if extra or missing:
        parts = []
        if missing:
            parts.append(f"missing values: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            parts.append(f"extra values: {extra[:5]}{'...' if len(extra) > 5 else ''}")
        details = "; ".join(parts)
        return ValidationTestResult(
            test_name=f"cat_freq[{col}]",
            metric="max_freq_diff",
            expected=f"<= {tolerance}",
            actual="value_set_mismatch",
            threshold=tolerance,
            passed=False,
            details=details,
        )

    # Aligner et calculer la difference max sur l'union (sans valeur manquante ici)
    aligned = freq_a.subtract(freq_b, fill_value=0.0).abs()
    max_diff = float(aligned.max())
    passed = bool(max_diff <= tolerance)
    worst = aligned.idxmax() if not aligned.empty else None
    details = (
        f"{len(set_a)} categories, max |freq_diff|={max_diff:.4f} "
        f"on '{worst}' (tolerance={tolerance})"
    )
    logger.debug("cat_freq[%s]: %s passed=%s", col, details, passed)

    return ValidationTestResult(
        test_name=f"cat_freq[{col}]",
        metric="max_freq_diff",
        expected=f"<= {tolerance}",
        actual=max_diff,
        threshold=tolerance,
        passed=passed,
        details=details,
    )


# ---------------------------------------------------------------------------
# Random format compliance tests (cf. PatternType.RANDOM_FORMAT)
# ---------------------------------------------------------------------------


def test_random_format_compliance(
    original: pd.Series,
    reconstructed: pd.Series,
    format_regex: str,
) -> list[ValidationTestResult]:
    """Tests softs pour les colonnes RANDOM_FORMAT.

    Les colonnes RANDOM_FORMAT regenerent des valeurs uniques au lieu de
    preserver les valeurs originales. On NE compare donc PAS les valeurs exactes.
    On verifie a la place :

    1. **format_match** : toutes les valeurs reconstruites matchent le regex.
    2. **uniqueness**   : `n_unique(reconstructed)` >= 99% de `n_unique(original)`.
    3. **length**       : moyenne de longueurs +/- 1 caractere (idem distribution
       triviale puisque le format_spec impose une longueur exacte).

    Retourne une liste de `ValidationTestResult` (3 entrees typiques).

    Args:
        original: serie d'origine (utile pour cardinality + longueur attendues).
        reconstructed: serie reconstruite a verifier.
        format_regex: regex du format spec ; toutes les valeurs doivent matcher.
    """
    col = original.name or "<unnamed>"
    results: list[ValidationTestResult] = []

    a = original.dropna().astype(str)
    b = reconstructed.dropna().astype(str)

    if a.empty or b.empty:
        results.append(
            ValidationTestResult(
                test_name=f"random_format_compliance[{col}]",
                metric="non_empty",
                expected="non-empty original and reconstructed",
                actual=f"orig={len(a)}, recon={len(b)}",
                threshold=None,
                passed=False,
                details="Cannot validate random format on empty series",
            )
        )
        return results

    # ---- 1. format_match ----
    try:
        compiled = re.compile(format_regex)
    except re.error as exc:
        results.append(
            ValidationTestResult(
                test_name=f"random_format[{col}]/format_match",
                metric="all_match_regex",
                expected="valid regex",
                actual=f"invalid: {exc}",
                threshold=None,
                passed=False,
                details=f"format_regex did not compile: {exc}",
            )
        )
        return results

    # On verifie TOUTES les valeurs reconstruites (cheap : str regex sur ~10k).
    n_match = int(b.map(lambda v: bool(compiled.match(v))).sum())
    match_passed = n_match == len(b)
    results.append(
        ValidationTestResult(
            test_name=f"random_format[{col}]/format_match",
            metric="all_match_regex",
            expected=f"100% match ({len(b)}/{len(b)})",
            actual=f"{n_match}/{len(b)}",
            threshold=None,
            passed=match_passed,
            details=(
                f"All {len(b)} reconstructed values match regex {format_regex!r}"
                if match_passed
                else f"Only {n_match}/{len(b)} values match regex {format_regex!r}"
            ),
        )
    )

    # ---- 2. uniqueness ----
    n_unique_orig = int(a.nunique())
    n_unique_recon = int(b.nunique())
    # Tolerance : on accepte 99% du compte original (collisions stochastiques
    # possibles sur de tres petits body_length, mais sur ~64 hex chars la
    # probabilite de collision sur 10k tirages est negligeable -- approx 2^-200).
    uniqueness_threshold = 0.99
    if n_unique_orig == 0:
        uniqueness_passed = n_unique_recon == 0
        uniqueness_ratio = 1.0 if uniqueness_passed else 0.0
    else:
        uniqueness_ratio = n_unique_recon / n_unique_orig
        uniqueness_passed = uniqueness_ratio >= uniqueness_threshold
    results.append(
        ValidationTestResult(
            test_name=f"random_format[{col}]/uniqueness",
            metric="unique_ratio",
            expected=f">= {uniqueness_threshold}",
            actual=float(uniqueness_ratio),
            threshold=uniqueness_threshold,
            passed=uniqueness_passed,
            details=(
                f"original n_unique={n_unique_orig}, reconstructed n_unique={n_unique_recon} "
                f"(ratio={uniqueness_ratio:.4f})"
            ),
        )
    )

    # ---- 3. length distribution ----
    mean_len_orig = float(a.map(len).mean())
    mean_len_recon = float(b.map(len).mean())
    length_diff = abs(mean_len_orig - mean_len_recon)
    length_passed = length_diff <= 1.0
    results.append(
        ValidationTestResult(
            test_name=f"random_format[{col}]/length",
            metric="mean_length_diff",
            expected="<= 1",
            actual=float(length_diff),
            threshold=1.0,
            passed=length_passed,
            details=(
                f"mean_length orig={mean_len_orig:.2f}, recon={mean_len_recon:.2f}, "
                f"diff={length_diff:.2f}"
            ),
        )
    )

    logger.debug(
        "random_format_compliance[%s]: match=%s unique=%s length=%s",
        col, match_passed, uniqueness_passed, length_passed,
    )
    return results


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def _select_top_correlation_pairs(
    original: pd.DataFrame,
    candidate_cols: list[str],
    max_pairs: int,
    min_strength: float,
) -> list[tuple[str, str, float]]:
    """Selectionne les paires les plus correlees parmi `candidate_cols`.

    Retourne une liste triee par force decroissante (col_a, col_b, strength).
    """
    pairs: list[tuple[str, str, float]] = []
    for col_a, col_b in itertools.combinations(candidate_cols, 2):
        try:
            strength, _ = _pair_correlation(original, col_a, col_b)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("pair correlation failed for (%s, %s): %s", col_a, col_b, exc)
            continue
        if not np.isfinite(strength):
            continue
        if strength >= min_strength:
            pairs.append((col_a, col_b, float(strength)))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:max_pairs]


def _compute_score(
    structural: list[ValidationTestResult],
    statistical: list[ValidationTestResult],
) -> float:
    """Score global 0-100.

    Schema de ponderation : les tests structurels comptent x2 (les erreurs y sont
    bloquantes : row_count, schema, ancres). Les tests statistiques x1.

    Formula : (weighted_passed / weighted_total) * 100.
    """
    structural_weight = 2.0
    statistical_weight = 1.0

    weighted_total = (
        len(structural) * structural_weight + len(statistical) * statistical_weight
    )
    if weighted_total == 0:
        return 0.0

    weighted_passed = (
        sum(structural_weight for t in structural if t.passed)
        + sum(statistical_weight for t in statistical if t.passed)
    )
    return round(100.0 * weighted_passed / weighted_total, 2)


def validate(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    anchor_columns: list[str] | None = None,
    thresholds: ValidationThresholds | None = None,
    random_format_columns: dict[str, str] | None = None,
) -> ValidationReport:
    """Pipeline complet de validation original vs reconstruit.

    Etapes :
    1. Tests structurels (row count, schema, ancres). Si row_count differe,
       court-circuit : on retourne immediatement un rapport partiel ou les tests
       statistiques colonne-par-colonne ne sont pas executes (impossible).
    2. Pour chaque colonne, lance les tests statistiques adaptes :
       - numerique : KS + mean
       - datetime : KS
       - categoriel : frequences
       - random_format : compliance (regex + uniqueness + length)
    3. Selectionne les paires les plus correlees et teste la preservation.
    4. Construit le ValidationReport.

    Schema de ponderation du score : tests structurels x2, statistiques x1
    (voir `_compute_score`).

    Args:
        original: DataFrame d'origine.
        reconstructed: DataFrame reconstruit.
        anchor_columns: colonnes a comparer en strict (egalite). Si non fournies,
            on n'execute pas les tests d'ancres.
        thresholds: seuils de tolerance ; defaut = ValidationThresholds().
        random_format_columns: mapping `{col_name: format_regex}` pour les colonnes
            qui ont ete generees via RANDOM_FORMAT. Pour ces colonnes on saute
            les tests d'egalite + cat_freq, on lance a la place le test de
            compliance (regex + uniqueness + length).
    """
    thresholds = thresholds or ValidationThresholds()
    anchor_columns = anchor_columns or []
    random_format_columns = random_format_columns or {}
    random_format_set = set(random_format_columns.keys())

    logger.info(
        "Starting validation: original=%s rows x %s cols, reconstructed=%s rows x %s cols, "
        "anchors=%s, random_format_columns=%s",
        len(original), len(original.columns), len(reconstructed), len(reconstructed.columns),
        anchor_columns, sorted(random_format_set),
    )

    structural: list[ValidationTestResult] = []
    statistical: list[ValidationTestResult] = []

    # ----- Tests structurels -----
    logger.info("Running structural tests")
    row_result = test_row_count(original, reconstructed)
    structural.append(row_result)

    schema_result = test_schema_match(original, reconstructed)
    structural.append(schema_result)

    if anchor_columns:
        # On ne teste les ancres en strict QUE pour les colonnes qui ne sont pas
        # RANDOM_FORMAT (ces dernieres regenerent des valeurs differentes par
        # design ; voir test_random_format_compliance pour le test softs).
        anchor_cols_to_check = [c for c in anchor_columns if c not in random_format_set]
        if anchor_cols_to_check:
            anchor_results = test_anchors_exact(original, reconstructed, anchor_cols_to_check)
            structural.extend(anchor_results)

    # Court-circuit : si les comptes different, les tests statistiques colonne-par-colonne
    # auraient un comportement non defini (ex: ks_2samp sur tailles tres differentes
    # est techniquement valide mais les autres tests reposent sur la comparaison
    # element par element).
    if not row_result.passed:
        logger.warning(
            "Row count mismatch detected, skipping per-column statistical tests"
        )
        score = _compute_score(structural, statistical)
        passed_count = sum(1 for t in structural + statistical if t.passed)
        failed_count = len(structural) + len(statistical) - passed_count
        return ValidationReport(
            overall_score=score,
            passed_count=passed_count,
            failed_count=failed_count,
            structural_tests=structural,
            statistical_tests=statistical,
        )

    # ----- Tests statistiques par colonne -----
    logger.info("Running per-column statistical tests")
    common_cols = [c for c in original.columns if c in reconstructed.columns]
    # Pour les tests stats non-ancres : on exclut a la fois les anchor_columns
    # (deja testees en strict) et les random_format_columns (testees a part).
    non_anchor_cols = [
        c for c in common_cols
        if c not in set(anchor_columns) and c not in random_format_set
    ]

    for col in non_anchor_cols:
        s_orig, s_recon = original[col], reconstructed[col]

        if _is_numeric(s_orig):
            statistical.append(test_distribution_ks(s_orig, s_recon, thresholds))
            statistical.append(
                test_mean_within_tolerance(s_orig, s_recon, thresholds.mean_tolerance_pct)
            )
        elif _is_datetime(s_orig):
            statistical.append(test_distribution_ks(s_orig, s_recon, thresholds))
        elif _is_categorical_like(s_orig):
            statistical.append(
                test_categorical_frequencies(s_orig, s_recon, thresholds.categorical_freq_tolerance)
            )
        else:
            logger.debug("Column %s skipped: unsupported dtype %s", col, s_orig.dtype)

    # ----- Random format compliance -----
    if random_format_columns:
        logger.info("Running random format compliance tests on %d cols", len(random_format_columns))
        for col, format_regex in random_format_columns.items():
            if col not in original.columns or col not in reconstructed.columns:
                logger.debug("random_format column %r missing from one DataFrame, skipping", col)
                continue
            rf_results = test_random_format_compliance(
                original[col], reconstructed[col], format_regex
            )
            statistical.extend(rf_results)

    # ----- Correlations -----
    logger.info("Running correlation preservation tests")
    if len(non_anchor_cols) >= 2:
        pairs = _select_top_correlation_pairs(
            original,
            non_anchor_cols,
            max_pairs=thresholds.max_correlation_pairs,
            min_strength=thresholds.min_correlation_to_check,
        )
        logger.info("Selected %d correlation pairs to validate", len(pairs))
        for col_a, col_b, _ in pairs:
            statistical.append(
                test_correlation_preserved(
                    original, reconstructed, col_a, col_b, thresholds.correlation_tolerance
                )
            )

    # ----- Aggregation -----
    score = _compute_score(structural, statistical)
    passed_count = sum(1 for t in structural + statistical if t.passed)
    failed_count = len(structural) + len(statistical) - passed_count

    logger.info(
        "Validation complete: score=%.2f passed=%d failed=%d (structural=%d, statistical=%d)",
        score, passed_count, failed_count, len(structural), len(statistical),
    )

    return ValidationReport(
        overall_score=score,
        passed_count=passed_count,
        failed_count=failed_count,
        structural_tests=structural,
        statistical_tests=statistical,
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_report(report: ValidationReport) -> None:
    """Affiche un rapport joliment formate avec Rich (tableau + score global)."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    # ----- Header / score global -----
    score = report.overall_score
    if score >= 95:
        color = "bright_green"
    elif score >= 75:
        color = "yellow"
    else:
        color = "red"

    target_status = (
        "[bright_green]TARGET MET[/]" if report.is_fidelity_target_met
        else "[red]TARGET MISSED[/]"
    )

    header = Text.assemble(
        ("Overall score: ", "bold"),
        (f"{score:.2f}/100", f"bold {color}"),
        ("   "),
        ("Passed: ", "bold"),
        (f"{report.passed_count}", "bright_green"),
        ("   Failed: ", "bold"),
        (f"{report.failed_count}", "red" if report.failed_count else "bright_green"),
        ("   "),
    )
    console.print(Panel(header, title="Validation Report", subtitle=target_status))

    def _add_test_rows(table: Table, tests: list[ValidationTestResult]) -> None:
        for t in tests:
            status = "[bright_green]PASS[/]" if t.passed else "[red]FAIL[/]"
            table.add_row(
                status,
                t.test_name,
                t.metric,
                str(t.expected),
                str(t.actual),
                t.details or "",
            )

    # ----- Tests structurels -----
    if report.structural_tests:
        struct_table = Table(title="Structural tests", show_lines=False)
        struct_table.add_column("Status", justify="center", width=6)
        struct_table.add_column("Name", overflow="fold")
        struct_table.add_column("Metric", overflow="fold")
        struct_table.add_column("Expected", overflow="fold")
        struct_table.add_column("Actual", overflow="fold")
        struct_table.add_column("Details", overflow="fold")
        _add_test_rows(struct_table, report.structural_tests)
        console.print(struct_table)

    # ----- Tests statistiques (top 20) -----
    if report.statistical_tests:
        # On affiche d'abord les FAILED, puis les passed, dans la limite de 20 entrees.
        sorted_stats = sorted(report.statistical_tests, key=lambda t: (t.passed, t.test_name))
        shown = sorted_stats[:20]
        title = "Statistical tests"
        if len(report.statistical_tests) > 20:
            title += f" (showing 20 of {len(report.statistical_tests)})"
        stat_table = Table(title=title, show_lines=False)
        stat_table.add_column("Status", justify="center", width=6)
        stat_table.add_column("Name", overflow="fold")
        stat_table.add_column("Metric", overflow="fold")
        stat_table.add_column("Expected", overflow="fold")
        stat_table.add_column("Actual", overflow="fold")
        stat_table.add_column("Details", overflow="fold")
        _add_test_rows(stat_table, shown)
        console.print(stat_table)


__all__ = [
    "ValidationThresholds",
    "validate",
    "test_row_count",
    "test_schema_match",
    "test_anchors_exact",
    "test_distribution_ks",
    "test_mean_within_tolerance",
    "test_correlation_preserved",
    "test_categorical_frequencies",
    "test_random_format_compliance",
    "print_report",
]


# Pytest, par defaut, collecte toute fonction nommee `test_*` rencontree dans
# l'espace de nom de modules importes. Nos fonctions atomiques `test_*` ne sont
# PAS des tests pytest mais des operations metier (test = "verification de
# validation"). On les marque explicitement pour qu'elles soient ignorees par
# le collecteur quand `src/validator.py` est re-importe depuis tests/.
for _fn in (
    test_row_count,
    test_schema_match,
    test_anchors_exact,
    test_distribution_ks,
    test_mean_within_tolerance,
    test_correlation_preserved,
    test_categorical_frequencies,
    test_random_format_compliance,
):
    _fn.__test__ = False  # type: ignore[attr-defined]
del _fn
