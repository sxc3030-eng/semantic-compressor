"""Anchor extractor : separe l'irreductible (a stocker) du regenerable (deductible).

Une colonne "ancre" est une colonne dont le contenu doit etre stocke tel quel parce
qu'il ne peut pas etre regenere par une regle (distribution, lookup, etc.).

Regles d'identification :
    1. Cardinalite stricte == 1.0 (toutes valeurs uniques, ex: UUID, email).
    2. Forcage manuel via `manual_anchor_columns`.
    3. Absence de pattern detectable (si une liste de patterns est fournie).

Les ancres sont serialisees en parquet (snappy par defaut). L'index du DataFrame
d'ancres est preserve : il sert de cle de seeding pour la reconstruction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .models import ColumnProfile, Pattern

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_anchors(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    patterns: list[Pattern] | None = None,
    manual_anchor_columns: list[str] | None = None,
    random_format_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Identifie les colonnes ancres et retourne (df_anchors, anchor_column_names).

    Une colonne est ancre si :
        1. `profile.is_anchor_candidate` est True (cardinalite stricte == 1.0,
           sans nulls — calcule en amont par le profiler).
        2. Elle est dans `manual_anchor_columns`.
        3. `patterns` est fourni mais aucun pattern ne mappe cette colonne.

    EXCEPTIONS (colonne PAS marquee ancre meme si la regle 1 s'applique) :
        - Colonne dont le pattern est `RANDOM_FORMAT` (regenerable depuis spec) :
          si `patterns` est fourni, exclure les colonnes RANDOM_FORMAT.
        - Colonne dans `random_format_columns` : exclusion directe (utilise quand
          `patterns` n'est pas encore disponible, typ. pass 1 de l'orchestrator).

    L'ordre des colonnes en sortie suit l'ordre d'origine dans `df`. L'index est
    preserve (sera utilise comme cle de seeding par le reconstructor).

    Args:
        df: DataFrame source.
        profiles: profils par colonne (un par colonne attendue dans `df`).
        patterns: regles de generation deja detectees ; si None, on n'utilise
            pas la regle "absence de pattern".
        manual_anchor_columns: noms forces comme ancres ; doivent exister dans `df`.
        random_format_columns: noms forces comme RANDOM_FORMAT a exclure des ancres,
            utile quand `patterns` n'est pas encore construit. Si `patterns` est
            fourni, ce parametre est redondant (les RANDOM_FORMAT sont identifies
            via patterns) mais reste accepte par coherence.

    Returns:
        (df_anchors, anchor_column_names) ou df_anchors est restreint aux ancres.

    Raises:
        ValueError: si `manual_anchor_columns` reference une colonne absente.
    """
    manual = list(manual_anchor_columns or [])
    if manual:
        missing = [c for c in manual if c not in df.columns]
        if missing:
            raise ValueError(
                f"manual_anchor_columns reference unknown columns: {missing}"
            )

    # Index par nom pour acces O(1) ; tolere des profils superflus mais on log.
    profiles_by_name = {p.name: p for p in profiles}
    unknown_profiles = [n for n in profiles_by_name if n not in df.columns]
    if unknown_profiles:
        logger.debug(
            "Profiles for columns absent from DataFrame are ignored: %s",
            unknown_profiles,
        )

    # Colonnes ayant un pattern de generation (autre qu'ANCHOR_DIRECT) =
    # regenerables. ANCHOR_DIRECT ne compte pas comme regenerable.
    # RANDOM_FORMAT est aussi regenerable -> a exclure des ancres.
    columns_with_pattern: set[str] = set()
    random_format_from_patterns: set[str] = set()
    if patterns is not None:
        for pat in patterns:
            if pat.pattern_type.value != "anchor_direct":
                columns_with_pattern.add(pat.column)
            if pat.pattern_type.value == "random_format":
                random_format_from_patterns.add(pat.column)

    # Union des colonnes a exclure (via patterns ou via param explicite).
    excluded_random_format = set(random_format_columns or []) | random_format_from_patterns

    manual_set = set(manual)
    anchor_columns: list[str] = []

    for col in df.columns:  # preservation de l'ordre d'origine
        # Exclusion prioritaire : les colonnes random_format ne deviennent JAMAIS
        # des ancres (meme si manuellement forcees comme ancres -- ce serait une
        # contradiction). On laisse l'utilisateur arbitrer en amont.
        if col in excluded_random_format:
            logger.debug("Column %r excluded from anchors (reason=random_format)", col)
            continue

        is_anchor = False
        reason = ""

        if col in manual_set:
            is_anchor = True
            reason = "manual"
        else:
            profile = profiles_by_name.get(col)
            if profile is not None and profile.is_anchor_candidate:
                is_anchor = True
                reason = "is_anchor_candidate"
            elif patterns is not None and col not in columns_with_pattern:
                # Regle 3 : aucune regle de generation -> on stocke tel quel.
                is_anchor = True
                reason = "no_pattern"

        if is_anchor:
            anchor_columns.append(col)
            logger.debug("Column %r marked as anchor (reason=%s)", col, reason)

    df_anchors = df.loc[:, anchor_columns].copy()
    # Index strictement identique (par construction de .loc, mais on est explicite).
    df_anchors.index = df.index

    logger.info(
        "Extracted %d anchor column(s) out of %d: %s",
        len(anchor_columns),
        len(df.columns),
        anchor_columns,
    )

    return df_anchors, anchor_columns


# ---------------------------------------------------------------------------
# IO parquet
# ---------------------------------------------------------------------------


_VALID_COMPRESSIONS = {"snappy", "gzip", "zstd", "brotli", "lz4", "none"}


def write_anchors_parquet(
    df_anchors: pd.DataFrame,
    output_path: Path,
    compression: str = "snappy",
) -> Path:
    """Ecrit le DataFrame d'ancres en parquet compresse. Retourne le chemin ecrit.

    Args:
        df_anchors: DataFrame des ancres (retourne par `extract_anchors`).
        output_path: chemin du fichier parquet de sortie (cree les parents si besoin).
        compression: codec parquet (`snappy`, `gzip`, `zstd`, `brotli`).

    Returns:
        Path absolu du fichier ecrit.
    """
    if compression not in _VALID_COMPRESSIONS:
        raise ValueError(
            f"Unknown parquet compression {compression!r}; "
            f"expected one of {sorted(_VALID_COMPRESSIONS)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Taille avant compression : approximation via memory_usage (deep pour les objets).
    uncompressed_bytes = int(df_anchors.memory_usage(deep=True).sum())

    df_anchors.to_parquet(
        output_path,
        compression=compression,
        engine="pyarrow",
        index=True,  # On preserve l'index : c'est la cle de seeding.
    )

    written_bytes = output_path.stat().st_size
    ratio = (uncompressed_bytes / written_bytes) if written_bytes else 0.0
    logger.info(
        "Wrote anchors parquet to %s (compression=%s, "
        "in-memory=%d B, on-disk=%d B, ratio=%.2fx)",
        output_path,
        compression,
        uncompressed_bytes,
        written_bytes,
        ratio,
    )

    return output_path.resolve()


def load_anchors_parquet(input_path: Path) -> pd.DataFrame:
    """Charge un parquet d'ancres ecrit par `write_anchors_parquet`.

    Le DataFrame retourne preserve l'index utilise comme cle de seeding.
    """
    input_path = Path(input_path)
    df = pd.read_parquet(input_path, engine="pyarrow")
    logger.debug("Loaded anchors parquet from %s (%d rows)", input_path, len(df))
    return df


# ---------------------------------------------------------------------------
# Estimation d'economies
# ---------------------------------------------------------------------------


def estimate_anchor_savings(
    df: pd.DataFrame,
    anchor_columns: list[str],
    original_csv_path: Path | None = None,
) -> dict[str, int]:
    """Estime le gain de taille apporte par la compression des ancres seules.

    Args:
        df: DataFrame complet (origine).
        anchor_columns: noms des colonnes ancres.
        original_csv_path: si fourni, on lit la taille reelle du CSV original ;
            sinon on retombe sur la taille en memoire de `df`.

    Returns:
        Dict avec :
            - `original_bytes`               : taille de reference.
            - `anchor_bytes_uncompressed`    : ancres serialisees en CSV (estime).
            - `anchor_bytes_compressed_snappy`: parquet snappy reel sur disque.
            - `compression_ratio`            : original_bytes / parquet snappy.
    """
    if original_csv_path is not None:
        original_path = Path(original_csv_path)
        if not original_path.exists():
            raise FileNotFoundError(f"Original CSV not found: {original_path}")
        original_bytes = original_path.stat().st_size
    else:
        original_bytes = int(df.memory_usage(deep=True).sum())

    missing = [c for c in anchor_columns if c not in df.columns]
    if missing:
        raise ValueError(f"anchor_columns not in DataFrame: {missing}")

    df_anchors = df.loc[:, anchor_columns]

    # CSV non compresse en memoire (sans index, comme un dump CSV "nu").
    anchor_csv = df_anchors.to_csv(index=False)
    anchor_bytes_uncompressed = len(anchor_csv.encode("utf-8"))

    # Snappy reel : on ecrit dans un fichier temporaire pour mesurer la taille
    # sur disque (la seule mesure honnete pour le ratio de compression).
    import tempfile

    with tempfile.NamedTemporaryFile(
        suffix=".parquet", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df_anchors.to_parquet(
            tmp_path, compression="snappy", engine="pyarrow", index=True
        )
        anchor_bytes_compressed_snappy = tmp_path.stat().st_size
    finally:
        tmp_path.unlink(missing_ok=True)

    if anchor_bytes_compressed_snappy == 0:
        compression_ratio = 0.0
    else:
        compression_ratio = original_bytes / anchor_bytes_compressed_snappy

    return {
        "original_bytes": int(original_bytes),
        "anchor_bytes_uncompressed": int(anchor_bytes_uncompressed),
        "anchor_bytes_compressed_snappy": int(anchor_bytes_compressed_snappy),
        "compression_ratio": float(compression_ratio),
    }


__all__ = [
    "extract_anchors",
    "write_anchors_parquet",
    "load_anchors_parquet",
    "estimate_anchor_savings",
]
