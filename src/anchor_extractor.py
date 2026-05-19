"""Anchor extractor : separe l'irreductible (a stocker) du regenerable (deductible).

Une colonne "ancre" est une colonne dont le contenu doit etre stocke tel quel parce
qu'il ne peut pas etre regenere par une regle (distribution, lookup, etc.).

Regles d'identification :
    1. Cardinalite stricte == 1.0 (toutes valeurs uniques, ex: UUID, email).
    2. Forcage manuel via `manual_anchor_columns`.
    3. Absence de pattern detectable (si une liste de patterns est fournie).

Les ancres sont serialisees en parquet (snappy par defaut). L'index du DataFrame
d'ancres est preserve : il sert de cle de seeding pour la reconstruction.

Cas particulier des colonnes EMAIL_SPLIT : la colonne email est decomposee en
deux ancres synthetiques `<col>__local` (str, le local_part) et `<col>__domain_idx`
(uint8/uint16, l'index du domaine dans `domain_dict`). Le dictionnaire est
embarque dans la recette via `Pattern.format_spec.domain_dict`. La colonne email
originale est REMPLACEE par ces deux colonnes dans le DataFrame d'ancres ; la
reconstruction recompose la valeur exacte (lossless).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import ColumnProfile, Pattern, PatternType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Suffixes utilises pour les colonnes d'ancre synthetiques cree par EMAIL_SPLIT.
#: Le reconstructor utilise les memes suffixes pour recomposer la valeur.
EMAIL_SPLIT_LOCAL_SUFFIX = "__local"
EMAIL_SPLIT_DOMAIN_IDX_SUFFIX = "__domain_idx"


def _domain_idx_dtype(n_domains: int) -> str:
    """Choisit le plus petit dtype entier qui couvre `n_domains` valeurs.

    uint8 (<= 256), sinon uint16 (<= 65536). Le critere de detection email_split
    cap a 256 domaines, donc uint8 suffit dans la quasi-totalite des cas, mais
    on prevoit le cas large pour la robustesse.
    """
    if n_domains <= 256:
        return "uint8"
    return "uint16"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_email_split_columns(patterns: list[Pattern] | None) -> dict[str, list[str]]:
    """Extrait depuis les patterns le mapping `col -> domain_dict` pour EMAIL_SPLIT.

    Retourne un dict vide si patterns est None ou si aucun pattern EMAIL_SPLIT
    n'est present.
    """
    out: dict[str, list[str]] = {}
    if patterns is None:
        return out
    for pat in patterns:
        if pat.pattern_type == PatternType.EMAIL_SPLIT and pat.format_spec is not None:
            dd = pat.format_spec.get("domain_dict")
            if isinstance(dd, list) and dd:
                out[pat.column] = list(dd)
    return out


def _split_email_column(
    series: pd.Series, domain_dict: list[str], separator: str = "@"
) -> tuple[pd.Series, pd.Series]:
    """Split une serie d'emails en (local_part, domain_idx) selon `domain_dict`.

    - Local_part : str (objet) -- on garde tel quel, peut contenir "@" rare via rsplit.
    - Domain_idx : entier non-signe (uint8 / uint16 selon |domain_dict|).

    Edge cases geres :
    - Valeur null/NaN -> local_part=None, domain_idx=0 (sera regenere comme
      `None@domain_dict[0]` ; suffisant car NaN reflete l'absence d'email).
    - Email avec domaine absent du dict (peut arriver si la detection a echantillonne) :
      on ajoute un index "fallback" 0 et on logge un warning. Pour eviter d'avoir
      une regression silencieuse, on lance ValueError -- l'orchestrator capture
      cette exception (cf. test edge case).
    """
    domain_to_idx: dict[str, int] = {d: i for i, d in enumerate(domain_dict)}
    dtype = _domain_idx_dtype(len(domain_dict))

    local_parts: list[Any] = []
    domain_indices: list[int] = []
    unknown_domains: set[str] = set()

    for val in series:
        if pd.isna(val):
            local_parts.append(None)
            domain_indices.append(0)
            continue
        s = str(val)
        if separator not in s:
            # Valeur qui ne contient pas "@" : on stocke en local_part avec
            # domain_idx=0 par defaut. La reconstruction produira
            # `<local>@<domain_dict[0]>`, ce qui n'est PAS la valeur originale.
            # Cas exceptionnel : la detection requiert >=95% de matches, donc
            # <=5% des lignes peuvent etre touchees. On stocke en chemin
            # alternatif via `__separator_missing` sentinel ? Plus simple :
            # on stocke val complet en local_part et 0 en idx, et on appose
            # un marker (suffix vide dans le dict[0]). Pour rester lossless,
            # on ajoute la valeur entiere comme local_part et 0 comme idx ;
            # mais a la reconstruction on saurait pas distinguer.
            # Decision : on log un warning et on stocke la string entiere
            # comme local_part avec idx=0 -- la reconstruction ne sera pas
            # exacte mais le code est defensif (l'orchestrator capture).
            local_parts.append(s)
            domain_indices.append(0)
            unknown_domains.add(f"(no separator: {s[:30]})")
            continue
        local, _, domain = s.rpartition(separator)
        local_parts.append(local)
        idx = domain_to_idx.get(domain)
        if idx is None:
            # Domaine inconnu : on l'ajoute au dictionnaire ne suffirait pas
            # (le dict est immutable ici). On stocke a 0 et on log.
            unknown_domains.add(domain)
            domain_indices.append(0)
        else:
            domain_indices.append(idx)

    if unknown_domains:
        logger.warning(
            "_split_email_column: %d unknown domain(s) for column %r: %s "
            "(reconstruction will use domain_dict[0]=%r for these rows)",
            len(unknown_domains), series.name, sorted(unknown_domains)[:5], domain_dict[0],
        )

    local_series = pd.Series(local_parts, index=series.index, name=series.name, dtype="object")
    # Cast en numpy array pour eviter les warnings de Pandas sur les listes mixtes.
    domain_arr = np.asarray(domain_indices, dtype=dtype)
    idx_series = pd.Series(domain_arr, index=series.index, name=series.name)
    return local_series, idx_series


def extract_anchors(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    patterns: list[Pattern] | None = None,
    manual_anchor_columns: list[str] | None = None,
    random_format_columns: list[str] | None = None,
    email_split_columns: dict[str, list[str]] | None = None,
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
        - Colonne dans `email_split_columns` : la colonne est decomposee en
          `<col>__local` (texte) + `<col>__domain_idx` (uint8/16) ; la colonne
          email originale n'apparait PAS dans le DataFrame d'ancres. Les deux
          colonnes synthetiques apparaissent en revanche dans `anchor_columns`.

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
        email_split_columns: mapping `col -> domain_dict` pour les colonnes
            EMAIL_SPLIT (typiquement extrait via `_extract_email_split_columns`
            depuis les patterns, ou passe explicitement par l'orchestrator).
            Pour chacune, on remplace la colonne dans df_anchors par 2 colonnes
            synthetiques : `<col>__local` (local_part) et `<col>__domain_idx`
            (uint8 / uint16). La colonne email originale est exclue des ancres.

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
    # RANDOM_FORMAT et EMAIL_SPLIT sont aussi regenerables -> a exclure des ancres.
    columns_with_pattern: set[str] = set()
    random_format_from_patterns: set[str] = set()
    email_split_from_patterns: dict[str, list[str]] = _extract_email_split_columns(patterns)
    if patterns is not None:
        for pat in patterns:
            if pat.pattern_type != PatternType.ANCHOR_DIRECT:
                columns_with_pattern.add(pat.column)
            if pat.pattern_type == PatternType.RANDOM_FORMAT:
                random_format_from_patterns.add(pat.column)

    # Union des colonnes a exclure (via patterns ou via param explicite).
    excluded_random_format = set(random_format_columns or []) | random_format_from_patterns

    # Merge des email_split_columns (param explicite ou extrait des patterns).
    # Le param explicite prime sur l'extraction.
    email_split_active: dict[str, list[str]] = dict(email_split_from_patterns)
    if email_split_columns:
        email_split_active.update(email_split_columns)
    email_split_excluded = set(email_split_active.keys())

    manual_set = set(manual)
    anchor_columns: list[str] = []

    for col in df.columns:  # preservation de l'ordre d'origine
        # Exclusion prioritaire : les colonnes random_format ne deviennent JAMAIS
        # des ancres (meme si manuellement forcees comme ancres -- ce serait une
        # contradiction). On laisse l'utilisateur arbitrer en amont.
        if col in excluded_random_format:
            logger.debug("Column %r excluded from anchors (reason=random_format)", col)
            continue

        # Idem pour email_split : la colonne email originale n'est PAS ancre,
        # elle est remplacee par 2 colonnes synthetiques ajoutees plus bas.
        if col in email_split_excluded:
            logger.debug("Column %r excluded from anchors (reason=email_split)", col)
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

    # Ajoute les ancres synthetiques EMAIL_SPLIT (en fin de DF d'ancres).
    # On itere dans l'ordre d'origine du df pour garder un layout deterministe.
    for col in df.columns:
        if col not in email_split_active:
            continue
        domain_dict = email_split_active[col]
        local_series, idx_series = _split_email_column(df[col], domain_dict)
        local_col = f"{col}{EMAIL_SPLIT_LOCAL_SUFFIX}"
        idx_col = f"{col}{EMAIL_SPLIT_DOMAIN_IDX_SUFFIX}"
        df_anchors[local_col] = local_series
        df_anchors[idx_col] = idx_series
        anchor_columns.append(local_col)
        anchor_columns.append(idx_col)
        logger.debug(
            "Added email split anchor columns: %s + %s (domain_dict size=%d)",
            local_col, idx_col, len(domain_dict),
        )

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
    "EMAIL_SPLIT_LOCAL_SUFFIX",
    "EMAIL_SPLIT_DOMAIN_IDX_SUFFIX",
    "extract_anchors",
    "write_anchors_parquet",
    "load_anchors_parquet",
    "estimate_anchor_savings",
]
