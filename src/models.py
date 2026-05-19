"""Schemas Pydantic v2 pour le semantic-compressor.

Ce module definit la structure de donnees utilisee a travers tout le pipeline :
profiling, detection de patterns, ecriture de recette, reconstruction et validation.

Aucune logique metier ici : uniquement schemas + validation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def _utcnow() -> datetime:
    """Timezone-aware UTC now (remplace datetime.utcnow() deprecated en 3.12+)."""
    return datetime.now(timezone.utc)
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ColumnType(str, Enum):
    """Type semantique d'une colonne, decouple du dtype pandas brut."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    STRING = "string"
    BOOLEAN = "boolean"


class DistributionType(str, Enum):
    """Familles de distributions reconnues par le pattern detector."""

    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    POWER_LAW = "power_law"
    CATEGORICAL_FREQ = "categorical_freq"
    EMPIRICAL = "empirical"


class RegexPattern(str, Enum):
    """Patterns regex reconnus sur les colonnes string."""

    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    UUID = "uuid"
    NONE = "none"


class PatternType(str, Enum):
    """Type de regle de generation pour une colonne donnee."""

    DISTRIBUTION = "distribution"
    LOOKUP = "lookup"
    FUNCTIONAL_DEP = "functional_dep"
    CONDITIONAL_DISTRIBUTION = "conditional_distribution"
    ANCHOR_DIRECT = "anchor_direct"


# ---------------------------------------------------------------------------
# Base config commune
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base pour tous les modeles : interdit les champs inconnus et valide a l'assignation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Sous-modeles statistiques
# ---------------------------------------------------------------------------


class NumericStats(_StrictModel):
    """Statistiques descriptives d'une colonne numerique."""

    mean: float
    std: float
    min: float
    max: float
    q25: float
    q50: float
    q75: float
    skewness: float | None = None
    kurtosis: float | None = None

    @model_validator(mode="after")
    def _check_order(self) -> NumericStats:
        # min <= q25 <= q50 <= q75 <= max ; tolerance pour les flottants en NaN-friendly.
        if self.min > self.max:
            raise ValueError("min must be <= max")
        ordered = [self.q25, self.q50, self.q75]
        if any(a > b for a, b in zip(ordered, ordered[1:])):
            raise ValueError("quartiles must satisfy q25 <= q50 <= q75")
        return self


class CategoricalStats(_StrictModel):
    """Statistiques d'une colonne categorielle : top valeurs et frequences."""

    top_values: dict[str, int] = Field(default_factory=dict)
    n_categories: int = 0

    @field_validator("n_categories")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("n_categories must be >= 0")
        return v


class DatetimeStats(_StrictModel):
    """Statistiques d'une colonne datetime."""

    min: datetime
    max: datetime
    range_days: float

    @model_validator(mode="after")
    def _check_range(self) -> DatetimeStats:
        if self.min > self.max:
            raise ValueError("datetime min must be <= max")
        if self.range_days < 0:
            raise ValueError("range_days must be >= 0")
        return self


# ---------------------------------------------------------------------------
# ColumnProfile
# ---------------------------------------------------------------------------


class ColumnProfile(_StrictModel):
    """Profil complet d'une colonne : type, stats, exemples et eligibilite d'ancre."""

    name: str
    column_type: ColumnType
    dtype: str
    n_unique: int = Field(ge=0)
    n_null: int = Field(ge=0)
    n_total: int = Field(ge=0)
    examples: list[Any] = Field(default_factory=list, max_length=5)
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None
    regex_pattern: RegexPattern = RegexPattern.NONE
    is_anchor_candidate: bool = False

    @model_validator(mode="after")
    def _check_counts(self) -> ColumnProfile:
        if self.n_unique > self.n_total:
            raise ValueError("n_unique cannot exceed n_total")
        if self.n_null > self.n_total:
            raise ValueError("n_null cannot exceed n_total")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cardinality(self) -> float:
        """Ratio n_unique / n_total (0 si n_total == 0)."""
        if self.n_total == 0:
            return 0.0
        return self.n_unique / self.n_total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def null_ratio(self) -> float:
        """Ratio n_null / n_total (0 si n_total == 0)."""
        if self.n_total == 0:
            return 0.0
        return self.n_null / self.n_total


# ---------------------------------------------------------------------------
# Correlations et dependances fonctionnelles
# ---------------------------------------------------------------------------


class Correlation(_StrictModel):
    """Correlation detectee entre deux colonnes."""

    col_a: str
    col_b: str
    correlation_type: str  # "pearson" | "cramers_v" | "anova"
    strength: float = Field(ge=0.0, le=1.0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("correlation_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        allowed = {"pearson", "cramers_v", "anova"}
        if v not in allowed:
            raise ValueError(f"correlation_type must be one of {sorted(allowed)}")
        return v


class FunctionalDependency(_StrictModel):
    """Dependance fonctionnelle A -> B (ou A <-> B si bijection)."""

    determinant: str
    dependent: str
    is_bijection: bool = False
    mapping_size: int = Field(ge=0)

    @model_validator(mode="after")
    def _no_self_dep(self) -> FunctionalDependency:
        if self.determinant == self.dependent:
            raise ValueError("determinant and dependent must differ")
        return self


# ---------------------------------------------------------------------------
# Patterns de generation
# ---------------------------------------------------------------------------


class Pattern(_StrictModel):
    """Regle de generation pour UNE colonne, utilisee par le reconstructor."""

    column: str
    pattern_type: PatternType
    distribution: DistributionType | None = None
    distribution_params: dict[str, Any] = Field(default_factory=dict)
    lookup_table: dict[Any, Any] | None = None
    source_column: str | None = None
    conditional_buckets: list[dict[str, Any]] | None = None
    fidelity_estimate: float = Field(default=1.0, ge=0.0, le=1.0)
    dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_shape(self) -> Pattern:
        # Coherence minimale entre pattern_type et les champs renseignes.
        pt = self.pattern_type
        if pt == PatternType.DISTRIBUTION and self.distribution is None:
            raise ValueError("DISTRIBUTION pattern requires `distribution`")
        if pt in (PatternType.LOOKUP, PatternType.FUNCTIONAL_DEP) and self.source_column is None:
            raise ValueError(f"{pt.value} pattern requires `source_column`")
        if pt == PatternType.FUNCTIONAL_DEP and self.lookup_table is None:
            raise ValueError("FUNCTIONAL_DEP pattern requires `lookup_table`")
        if pt == PatternType.CONDITIONAL_DISTRIBUTION:
            if self.source_column is None:
                raise ValueError("CONDITIONAL_DISTRIBUTION pattern requires `source_column`")
            if not self.conditional_buckets:
                raise ValueError("CONDITIONAL_DISTRIBUTION pattern requires `conditional_buckets`")
        return self


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


class RecipeMetadata(_StrictModel):
    """Metadata de la recette : tailles, fidelite cible, stratification de seed."""

    table_name: str
    version: str = "1.0"
    n_rows: int = Field(ge=0)
    original_size_bytes: int = Field(ge=0)
    anchor_size_bytes: int = Field(ge=0)
    recipe_size_bytes: int = Field(default=0, ge=0)
    compression_ratio: float = Field(ge=0.0)
    fidelity_target: float = Field(default=0.95, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=_utcnow)
    seed_strategy: str = "sha256_anchor_id_int32"
    python_version: str | None = Field(default_factory=lambda: sys.version.split()[0])


class Recipe(_StrictModel):
    """Modele racine : tout ce qui est necessaire pour reconstruire la BD."""

    metadata: RecipeMetadata
    # Note : `schema` est un attribut reserve par Pydantic v1, on prefere un nom explicite.
    schema_columns: list[ColumnProfile]
    anchor_columns: list[str] = Field(default_factory=list)
    anchor_file: str
    patterns: list[Pattern] = Field(default_factory=list)
    correlations: list[Correlation] = Field(default_factory=list)
    functional_dependencies: list[FunctionalDependency] = Field(default_factory=list)
    validation_tests: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_anchor_columns_exist(self) -> Recipe:
        known = {c.name for c in self.schema_columns}
        unknown = [a for a in self.anchor_columns if a not in known]
        if unknown:
            raise ValueError(f"anchor_columns reference unknown columns: {unknown}")
        return self

    def to_json(self, indent: int = 2) -> str:
        """Serialise la recette en JSON formate (utile pour debug et inspection humaine)."""
        return json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        """Construit une Recipe a partir d'un dict (typiquement charge depuis JSON)."""
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


class ValidationTestResult(_StrictModel):
    """Resultat d'un test de validation unitaire (structurel ou statistique)."""

    test_name: str
    metric: str
    expected: Any
    actual: Any
    threshold: float | None = None
    passed: bool
    details: str | None = None


class ValidationReport(_StrictModel):
    """Rapport agrege de la validation original vs reconstruit."""

    overall_score: float = Field(ge=0.0, le=100.0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    structural_tests: list[ValidationTestResult] = Field(default_factory=list)
    statistical_tests: list[ValidationTestResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_fidelity_target_met(self) -> bool:
        """True si overall_score / 100 >= 0.95."""
        return (self.overall_score / 100.0) >= 0.95


__all__ = [
    "ColumnType",
    "DistributionType",
    "RegexPattern",
    "PatternType",
    "NumericStats",
    "CategoricalStats",
    "DatetimeStats",
    "ColumnProfile",
    "Correlation",
    "FunctionalDependency",
    "Pattern",
    "RecipeMetadata",
    "Recipe",
    "ValidationTestResult",
    "ValidationReport",
]
