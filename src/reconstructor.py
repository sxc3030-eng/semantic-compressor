"""Reconstructor : regenere un DataFrame complet a partir d'une recette `.md` et
d'un parquet d'ancres.

Le module est strictement deterministe : meme recette + memes ancres -> meme
DataFrame, bit a bit. La graine est derivee de l'`id` d'ancre via SHA-256 (32
premiers bits du digest hex), garantissant la reproductibilite inter-process.

Pipeline a haut niveau :
    1. `parse_recipe`        : lit le `.md` -> objet `Recipe` (Pydantic)
    2. `load_anchors_parquet`: lit le parquet d'ancres (via anchor_extractor)
    3. `reconstruct`         : pour chaque ligne d'ancres, applique les patterns
       de generation dans l'ordre topologique.

Sampling :
    - Chaque ligne ouvre un `np.random.default_rng(derive_row_seed(anchor_id))`.
    - Les patterns DISTRIBUTION / CONDITIONAL_DISTRIBUTION puisent dans ce rng.
    - Les patterns FUNCTIONAL_DEP / ANCHOR_DIRECT n'utilisent pas le rng.

Notes :
    - L'ordre des colonnes de sortie suit `recipe.schema_columns` (ordre original),
      pas l'ordre topologique des patterns.
    - Les datetimes sont restituees au format ISO 8601 avec offset `+0000`
      (format CSV de la BD de test) si `params['from_string']` est True.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anchor_extractor import load_anchors_parquet
from .models import (
    ColumnProfile,
    ColumnType,
    DistributionType,
    Pattern,
    PatternType,
    Recipe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Format datetime de sortie pour les colonnes qui etaient stockees en string ISO
# dans le CSV original (ex: "2025-07-02T21:27:47+0000"). Doit matcher le format
# ecrit par examples/generate_fake_db.py (`%Y-%m-%dT%H:%M:%S%z`).
_DATETIME_OUTPUT_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Regex utilisee par `parse_recipe` pour extraire les blocs ```json ... ```
# precedes par un marqueur HTML <!-- DATA -->.
_SECTION_DATA_RE = re.compile(
    r"<!--\s*DATA\s*-->\s*\n```json\s*\n(.*?)\n```",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# 1. Parsing de la recette
# ---------------------------------------------------------------------------


def _split_sections(md_text: str) -> dict[str, str]:
    """Decoupe le .md en sections nommees par leur titre `## NAME`.

    La premiere section (le header `# RECIPE:`) est stockee sous la cle `_HEADER`.
    Le titre est conserve en majuscules.
    """
    sections: dict[str, str] = {}
    # On force un saut de ligne en amont pour que la 1ere section (`# RECIPE:`)
    # soit traitee uniformement par le split.
    text = "\n" + md_text.lstrip("\n")
    # split sur les titres de niveau 2 : conserve le contenu de chaque bloc.
    chunks = re.split(r"\n##\s+", text)
    # Le 1er chunk contient le `# RECIPE: ...` et tout ce qui precede la 1ere section.
    sections["_HEADER"] = chunks[0].lstrip("\n").rstrip()
    for chunk in chunks[1:]:
        # `chunk` ressemble a : "METADATA\n\n<!-- DATA -->\n```json\n{...}\n```\n..."
        if not chunk.strip():
            continue
        first_line, _, body = chunk.partition("\n")
        name = first_line.strip().upper()
        if not name:
            continue
        sections[name] = body
    return sections


def _extract_json_block(section_body: str) -> str | None:
    """Cherche le pattern `<!-- DATA -->\\n```json\\n...\\n``` ` dans le corps d'une section.

    Retourne le contenu JSON (str) ou None si absent.
    """
    match = _SECTION_DATA_RE.search(section_body)
    if match is None:
        return None
    return match.group(1)


def parse_recipe(recipe_path: Path) -> Recipe:
    """Parse un fichier `.md` de recette et retourne un objet `Recipe` Pydantic.

    Le contrat de format (cf. brief) :
    - Sections separees par `## <NAME>`.
    - Chaque section porteuse de donnees structuree contient un bloc
      ```json ... ``` precede d'une ligne `<!-- DATA -->`.
    - Sections attendues : METADATA, SCHEMA, ANCHORS, PATTERNS, CORRELATIONS,
      FUNCTIONAL_DEPENDENCIES, RECONSTRUCTION_ALGORITHM (texte libre, ignore),
      VALIDATION_TESTS.

    Raises:
        FileNotFoundError: le fichier n'existe pas.
        ValueError: section obligatoire manquante ou JSON invalide.
    """
    import json

    recipe_path = Path(recipe_path)
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    t0 = time.perf_counter()
    logger.info("parse_recipe: %s", recipe_path)

    md_text = recipe_path.read_text(encoding="utf-8")
    sections = _split_sections(md_text)

    def _get_json(name: str, required: bool = True) -> Any:
        body = sections.get(name)
        if body is None:
            if required:
                raise ValueError(f"Recipe missing required section: ## {name}")
            return None
        block = _extract_json_block(body)
        if block is None:
            if required:
                raise ValueError(
                    f"Recipe section ## {name} is missing the `<!-- DATA -->` JSON block"
                )
            return None
        try:
            return json.loads(block)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Recipe section ## {name} contains invalid JSON: {exc}"
            ) from exc

    metadata_data = _get_json("METADATA", required=True)
    schema_data = _get_json("SCHEMA", required=True)
    anchors_data = _get_json("ANCHORS", required=True)
    patterns_data = _get_json("PATTERNS", required=True)
    correlations_data = _get_json("CORRELATIONS", required=False) or []
    fdeps_data = _get_json("FUNCTIONAL_DEPENDENCIES", required=False) or []
    validation_data = _get_json("VALIDATION_TESTS", required=False) or []

    # Le schema peut etre fourni soit comme une liste de ColumnProfile,
    # soit comme dict avec une cle `columns` -- on accepte les deux pour
    # tolerer la variabilite du writer.
    schema_columns = (
        schema_data["columns"] if isinstance(schema_data, dict) and "columns" in schema_data
        else schema_data
    )
    # Idem pour les ancres : le writer canonique emet `{anchor_file, anchor_columns}`
    # mais on accepte aussi `{file, columns}` ou la liste nue pour souplesse.
    if isinstance(anchors_data, dict):
        anchor_columns = (
            anchors_data.get("anchor_columns")
            or anchors_data.get("columns")
            or []
        )
        anchor_file = anchors_data.get("anchor_file") or anchors_data.get("file") or ""
    else:
        anchor_columns = anchors_data
        anchor_file = ""

    recipe_dict = {
        "metadata": metadata_data,
        "schema_columns": schema_columns,
        "anchor_columns": anchor_columns,
        "anchor_file": anchor_file,
        "patterns": patterns_data,
        "correlations": correlations_data,
        "functional_dependencies": fdeps_data,
        "validation_tests": validation_data,
    }

    recipe = Recipe.model_validate(recipe_dict)

    dt = time.perf_counter() - t0
    logger.info(
        "parse_recipe done: %d schema cols, %d patterns, %d anchors (%.3fs)",
        len(recipe.schema_columns),
        len(recipe.patterns),
        len(recipe.anchor_columns),
        dt,
    )
    return recipe


# ---------------------------------------------------------------------------
# 2. Seeding deterministe
# ---------------------------------------------------------------------------


def derive_row_seed(anchor_id: Any, salt: str = "") -> int:
    """Derive une graine 32 bits deterministe a partir d'un id d'ancre.

    sha256(salt + str(anchor_id)).hexdigest()[:8] -> int (base 16).
    Cf. IMPLEMENTATION_NOTES section 1.2 : `hash()` natif Python est randomise
    par defaut, on doit utiliser sha256 pour la reproductibilite inter-process.
    """
    digest = hashlib.sha256((salt + str(anchor_id)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)  # 32 bits, valide pour np.random.default_rng


# ---------------------------------------------------------------------------
# 3. Tri topologique des patterns
# ---------------------------------------------------------------------------


def _topological_order(patterns: list[Pattern], schema_columns: list[str]) -> list[str]:
    """Retourne l'ordre topologique d'application des patterns.

    Nodes = noms de colonnes (patterns). Aretes = `dependencies` (predecesseurs).
    Les colonnes du schema sans pattern sont ignorees ici (gerees a part).

    Si le graphe contient un cycle (cas typique : 2 colonnes correlees mutuellement
    via CONDITIONAL_DISTRIBUTION reciproques), on casse iterativement la 1ere
    arete fermante et on log un warning. Memes semantiques que recipe_writer.

    Raises:
        ValueError: si une dependance pointe vers une colonne absente du
            schema ET des patterns (incoherent).
    """
    pattern_index = {p.column: p for p in patterns}
    schema_set = set(schema_columns)

    # Edges : col -> set(deps). On filtre les deps qui pointent vers des colonnes
    # absentes du graphe (les ancres pures n'ont pas de pattern, on les considere
    # comme deja-disponibles).
    edges: dict[str, set[str]] = {}
    for pat in patterns:
        deps: set[str] = set()
        for dep in pat.dependencies:
            if dep in pattern_index:
                deps.add(dep)
            elif dep not in schema_set:
                # Incoherence reelle : dep n'est ni un pattern, ni une colonne du schema.
                raise ValueError(
                    f"Pattern for column {pat.column!r} depends on unknown column {dep!r} "
                    f"(not in schema nor in patterns)"
                )
            # else: dep est une colonne du schema sans pattern propre (ex: ancre),
            # consideree comme deja-disponible -- on ne l'ajoute pas au sorter.
        edges[pat.column] = deps

    def _try_sort() -> list[str]:
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for col, deps in edges.items():
            sorter.add(col, *deps)
        return list(sorter.static_order())

    max_break_iterations = len(patterns) * 2  # garde-fou pour eviter une boucle infinie
    for _ in range(max_break_iterations):
        try:
            return _try_sort()
        except CycleError as exc:
            cycle_nodes = exc.args[1] if len(exc.args) > 1 else []
            if len(cycle_nodes) < 2:
                # Cycle degenere : on abandonne la topo et on retourne l'ordre d'insertion.
                logger.warning(
                    "Cannot parse cycle info, falling back to pattern insertion order"
                )
                return list(edges.keys())
            # cycle = [a, b, ..., a] : on retire l'edge a -> b (premiere edge fermante).
            a, b = cycle_nodes[0], cycle_nodes[1]
            if b in edges.get(a, set()):
                logger.warning(
                    "Breaking dependency cycle in reconstructor: dropping edge %s -> %s "
                    "(from cycle %s). Reconstruction will use a partial dependency order.",
                    a, b, cycle_nodes,
                )
                edges[a].discard(b)
                continue
            # Sinon, cherche n'importe quelle edge du cycle a retirer.
            broke = False
            for i in range(len(cycle_nodes) - 1):
                src, dst = cycle_nodes[i], cycle_nodes[i + 1]
                if dst in edges.get(src, set()):
                    logger.warning(
                        "Breaking dependency cycle in reconstructor: dropping edge %s -> %s "
                        "(from cycle %s)",
                        src, dst, cycle_nodes,
                    )
                    edges[src].discard(dst)
                    broke = True
                    break
            if not broke:
                logger.error(
                    "Could not break cycle %s, falling back to insertion order", cycle_nodes
                )
                return list(edges.keys())

    logger.error(
        "Cycle-breaking did not converge after %d iterations, falling back to insertion order",
        max_break_iterations,
    )
    return list(edges.keys())


# ---------------------------------------------------------------------------
# 4. Sampling depuis une distribution
# ---------------------------------------------------------------------------


def _sample_distribution(
    rng: np.random.Generator,
    distribution: str | DistributionType,
    params: dict[str, Any],
) -> Any:
    """Sample 1 valeur depuis une distribution + ses parametres.

    Supporte : NORMAL, EXPONENTIAL, UNIFORM, POWER_LAW, CATEGORICAL_FREQ, EMPIRICAL.
    Retourne un scalar Python (float / str / int selon la distribution).
    """
    if isinstance(distribution, DistributionType):
        distribution = distribution.value

    if distribution == DistributionType.NORMAL.value:
        # Notes : pattern_detector store mean/std as either {"mean", "std"} or
        # {"loc", "scale"} selon les buckets (cf. _summarize_numeric_bucket).
        loc = params.get("loc", params.get("mean", 0.0))
        scale = params.get("scale", params.get("std", 1.0))
        value = float(rng.normal(loc=loc, scale=scale))
        clip = params.get("clip")
        if clip is not None and len(clip) == 2:
            value = float(np.clip(value, clip[0], clip[1]))
        return value

    if distribution == DistributionType.EXPONENTIAL.value:
        scale = float(params.get("scale", 1.0))
        if params.get("reversed") and "mirror_max" in params:
            # Distribution decroissante (ex: last_login, dense pres de la fin).
            return float(params["mirror_max"]) - float(rng.exponential(scale=scale))
        loc = float(params.get("loc", 0.0))
        return float(rng.exponential(scale=scale)) + loc

    if distribution == DistributionType.UNIFORM.value:
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        if scale <= 0:
            return loc
        return float(rng.uniform(low=loc, high=loc + scale))

    if distribution == DistributionType.POWER_LAW.value:
        a = float(params.get("a", 1.0))
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        if a <= 0:
            a = 1.0
        # rng.power tire dans [0, 1]. On rescale loc + x*scale pour matcher le fit scipy.
        x = float(rng.power(a))
        return loc + x * scale

    if distribution == DistributionType.CATEGORICAL_FREQ.value:
        freqs = params.get("frequencies") or {}
        if not freqs:
            return None
        keys = list(freqs.keys())
        probs = np.array(list(freqs.values()), dtype=np.float64)
        total = probs.sum()
        if total <= 0:
            # Egalite parfaite ou input vide : on choisit uniformement.
            return rng.choice(keys)
        probs = probs / total
        idx = int(rng.choice(len(keys), p=probs))
        return keys[idx]

    if distribution == DistributionType.EMPIRICAL.value:
        values = params.get("values") or []
        if not values:
            return None
        idx = int(rng.choice(len(values)))
        return values[idx]

    raise ValueError(f"Unknown distribution type: {distribution!r}")


# ---------------------------------------------------------------------------
# 5. Conversion de valeurs (datetime / typage final)
# ---------------------------------------------------------------------------


def _format_datetime_value(seconds_since_epoch: float, from_string: bool) -> Any:
    """Convertit une valeur secondes Unix en datetime / string ISO.

    - Si `from_string` est True : retourne une str ISO 8601 au format `+0000`
      (matche le format CSV de generate_fake_db, qui utilise `%Y-%m-%dT%H:%M:%S%z`).
    - Sinon : retourne un objet `datetime` timezone-aware (UTC).
    """
    # `seconds_since_epoch` peut etre out-of-range (ex: distribution exponentielle
    # qui tire negatif ou tres lointain). On clamp pour eviter une exception.
    try:
        dt = datetime.fromtimestamp(float(seconds_since_epoch), tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        # Bornes raisonnables : 1970..3000.
        if seconds_since_epoch < 0:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            dt = datetime(3000, 1, 1, tzinfo=timezone.utc)

    if from_string:
        # Strftime en format `%Y-%m-%dT%H:%M:%S%z` produit "+0000" (pas "+00:00").
        return dt.strftime(_DATETIME_OUTPUT_FORMAT)
    return dt


def _cast_to_column_type(value: Any, profile: ColumnProfile) -> Any:
    """Cast une valeur generee vers le type attendu par le profil de colonne.

    - NUMERIC + dtype int -> int(round)
    - NUMERIC + dtype float -> float
    - BOOLEAN -> bool (gere les strings "True"/"False")
    - DATETIME -> laisse tel quel (la conversion ISO est faite en amont si requise)
    - STRING / CATEGORICAL -> str
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None

    col_type = profile.column_type
    dtype = profile.dtype

    if col_type == ColumnType.NUMERIC:
        if dtype and (
            dtype.startswith("int") or dtype.startswith("uint") or dtype == "int64" or dtype == "Int64"
        ):
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if col_type == ColumnType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # Tolere "True"/"False" (strings) souvent issues de CATEGORICAL_FREQ sur bool.
            stripped = value.strip().lower()
            if stripped in ("true", "1", "yes"):
                return True
            if stripped in ("false", "0", "no"):
                return False
        try:
            return bool(value)
        except Exception:  # pragma: no cover - defensive
            return None

    if col_type == ColumnType.DATETIME:
        # Si on a un float (rare ici si l'amont a deja converti), on le rend en datetime.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _format_datetime_value(float(value), from_string=False)
        return value

    # STRING / CATEGORICAL : on serialise tel quel (str).
    if isinstance(value, str):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# 6. CONDITIONAL_DISTRIBUTION : lookup de bucket
# ---------------------------------------------------------------------------


def _find_bucket(buckets: list[dict[str, Any]], source_value: Any) -> dict[str, Any]:
    """Trouve le bucket correspondant a `source_value`.

    - Si les buckets ont `bucket_range` (pivot numerique) : range-lookup.
    - Si les buckets ont `bucket_value` (pivot categoriel) : value-lookup.
    - Si aucun bucket ne matche : retourne le bucket le plus proche (numeric)
      ou le 1er bucket (categorical) + log warning.
    """
    if not buckets:
        raise ValueError("Cannot find bucket in empty bucket list")

    first = buckets[0]
    is_numeric_pivot = "bucket_range" in first

    if is_numeric_pivot:
        # source_value attendu numerique. On tolere les strings convertibles.
        try:
            val = float(source_value)
        except (TypeError, ValueError):
            logger.warning(
                "Conditional bucket: source value %r not numeric, falling back to bucket 0",
                source_value,
            )
            return buckets[0]

        for i, bucket in enumerate(buckets):
            lo, hi = bucket["bucket_range"]
            # Inclusivite a droite uniquement sur le dernier bucket (qcut).
            if i == len(buckets) - 1:
                if lo <= val <= hi:
                    return bucket
            else:
                if lo <= val < hi:
                    return bucket
        # Hors plage : bucket le plus proche.
        if val < buckets[0]["bucket_range"][0]:
            return buckets[0]
        return buckets[-1]

    # Pivot categoriel.
    for bucket in buckets:
        if str(bucket.get("bucket_value")) == str(source_value):
            return bucket
    # Nouvelle categorie : fallback bucket 0.
    logger.warning(
        "Conditional bucket: category %r not found, falling back to bucket 0",
        source_value,
    )
    return buckets[0]


def _sample_from_bucket(
    rng: np.random.Generator, bucket: dict[str, Any]
) -> Any:
    """Sample une valeur depuis un bucket conditionnel (numerique ou categoriel)."""
    dist = bucket.get("distribution")
    # Bucket categoriel : `frequencies` est directement dans le bucket.
    if "frequencies" in bucket:
        return _sample_distribution(
            rng,
            DistributionType.CATEGORICAL_FREQ,
            {"frequencies": bucket["frequencies"]},
        )
    params = bucket.get("params") or {}
    return _sample_distribution(rng, dist, params)


# ---------------------------------------------------------------------------
# 7. Application d'un pattern a une ligne
# ---------------------------------------------------------------------------


def _apply_pattern(
    rng: np.random.Generator,
    pattern: Pattern,
    row: dict[str, Any],
    anchor_row: dict[str, Any],
    profile: ColumnProfile,
) -> Any:
    """Applique un pattern unique et retourne la valeur generee.

    - ANCHOR_DIRECT       : copie depuis anchor_row[col]
    - DISTRIBUTION        : sample depuis distribution + params
    - FUNCTIONAL_DEP      : lookup_table.get(row[source_column])
    - CONDITIONAL_DISTRIBUTION : trouve le bucket de row[source_column] puis sample
    """
    col = pattern.column
    pt = pattern.pattern_type

    if pt == PatternType.ANCHOR_DIRECT:
        return anchor_row.get(col)

    if pt == PatternType.DISTRIBUTION:
        if pattern.distribution is None:  # pragma: no cover - garantie par le model_validator
            raise ValueError(f"Pattern for {col} missing `distribution`")
        params = pattern.distribution_params or {}
        value = _sample_distribution(rng, pattern.distribution, params)
        # Cas datetime : reconvertir secondes -> ISO/datetime. On preferera la
        # forme str si le profil dit que le dtype d'origine etait object
        # (CSV charge sans parse_dates) -- meme si `from_string=False` dans les
        # params (le profiler classe en DATETIME quand un object string parse,
        # mais ne propage pas toujours le flag).
        if params.get("unit") == "seconds_since_epoch":
            wants_string = bool(params.get("from_string")) or (
                profile.column_type == ColumnType.DATETIME and profile.dtype == "object"
            )
            value = _format_datetime_value(float(value), from_string=wants_string)
        else:
            # Pour les colonnes numeriques : clip aux bornes observees dans le
            # profil. Le pattern_detector ne propage pas les bornes (NORMAL fit
            # n'a pas de clip), mais le profiler les a en `numeric_stats.min/max`.
            # Sans clip, NORMAL(34, 12) peut produire des ages negatifs.
            if (
                profile.column_type == ColumnType.NUMERIC
                and profile.numeric_stats is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                value = float(
                    np.clip(value, profile.numeric_stats.min, profile.numeric_stats.max)
                )
        return _cast_to_column_type(value, profile)

    if pt == PatternType.FUNCTIONAL_DEP:
        if pattern.lookup_table is None or pattern.source_column is None:
            raise ValueError(f"FUNCTIONAL_DEP pattern for {col} missing lookup/source")
        src_value = row.get(pattern.source_column)
        # Recherche directe puis (fallback) sur la version str de la cle (JSON serialise).
        if src_value in pattern.lookup_table:
            result = pattern.lookup_table[src_value]
        else:
            result = pattern.lookup_table.get(str(src_value))
        return _cast_to_column_type(result, profile)

    if pt == PatternType.CONDITIONAL_DISTRIBUTION:
        if pattern.source_column is None or not pattern.conditional_buckets:
            raise ValueError(
                f"CONDITIONAL_DISTRIBUTION pattern for {col} missing source/buckets"
            )
        src_value = row.get(pattern.source_column)
        bucket = _find_bucket(pattern.conditional_buckets, src_value)
        value = _sample_from_bucket(rng, bucket)
        # Cas datetime conditional : meme logique que DISTRIBUTION, on consulte
        # le profil pour choisir str vs datetime.
        bucket_params = bucket.get("params") or {}
        if bucket_params.get("unit") == "seconds_since_epoch":
            wants_string = bool(bucket_params.get("from_string")) or (
                profile.column_type == ColumnType.DATETIME and profile.dtype == "object"
            )
            value = _format_datetime_value(float(value), from_string=wants_string)
        elif (
            profile.column_type == ColumnType.NUMERIC
            and profile.numeric_stats is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            # Clip aux bornes observees, idem cas DISTRIBUTION.
            value = float(
                np.clip(value, profile.numeric_stats.min, profile.numeric_stats.max)
            )
        return _cast_to_column_type(value, profile)

    if pt == PatternType.LOOKUP:
        # LOOKUP non utilise actuellement par pattern_detector mais on le couvre
        # pour completude (meme semantique que FUNCTIONAL_DEP sans guarantee de FD).
        if pattern.lookup_table is None or pattern.source_column is None:
            raise ValueError(f"LOOKUP pattern for {col} missing lookup/source")
        src_value = row.get(pattern.source_column)
        if src_value in pattern.lookup_table:
            return _cast_to_column_type(pattern.lookup_table[src_value], profile)
        return _cast_to_column_type(
            pattern.lookup_table.get(str(src_value)), profile
        )

    raise ValueError(f"Unsupported pattern_type: {pt!r}")


# ---------------------------------------------------------------------------
# 8. Reconstruction
# ---------------------------------------------------------------------------


def reconstruct(recipe: Recipe, anchors: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit le DataFrame complet a partir de la recette et des ancres.

    Algorithme :
        1. Tri topologique des patterns selon `Pattern.dependencies`.
        2. Pour chaque ligne d'ancres (preserve l'index, ordre de ligne stable):
           a. Initialise `rng = np.random.default_rng(derive_row_seed(anchor_id))`
              ou anchor_id = anchors.iloc[i, 0] (1ere colonne d'ancre).
           b. Applique chaque pattern en ordre topologique.
        3. Reconstruit le DataFrame avec ordre colonnes = recipe.schema_columns.

    Retourne pd.DataFrame avec meme schema, meme index, et reproductible.
    """
    t0 = time.perf_counter()
    n_rows = len(anchors)
    n_cols = len(recipe.schema_columns)
    logger.info(
        "reconstruct start: %d rows, %d columns, %d patterns",
        n_rows,
        n_cols,
        len(recipe.patterns),
    )

    if n_rows == 0:
        # Cas degenere : retourne un DataFrame vide avec le bon schema.
        empty = {p.name: pd.Series(dtype=object) for p in recipe.schema_columns}
        return pd.DataFrame(empty)

    # 1. Ordre topologique.
    schema_col_names = [p.name for p in recipe.schema_columns]
    ordered_columns = _topological_order(recipe.patterns, schema_col_names)
    pattern_index = {p.column: p for p in recipe.patterns}
    profile_index = {p.name: p for p in recipe.schema_columns}

    # 2. Pre-cache : convertir les ancres en list-of-dicts une fois (O(N)) pour
    # eviter la penalite iloc[i] x N. On preserve l'ordre d'index.
    anchor_columns_in_df = list(anchors.columns)
    if not anchor_columns_in_df:
        raise ValueError("Anchors DataFrame is empty (no columns); cannot derive seeds")
    seed_col = anchor_columns_in_df[0]
    anchor_records = anchors.to_dict(orient="records")
    anchor_index = list(anchors.index)

    # 3. Generation ligne par ligne. Une cle pour la reproductibilite :
    # on itere sur les ancres dans l'ordre fourni (anchors.index), pas dans
    # un ordre aleatoire. Le seed depend de l'id d'ancre, pas de l'index physique.
    generated_rows: list[dict[str, Any]] = []
    log_every = max(1, n_rows // 10)  # logs progressifs sur les gros datasets

    for i, anchor_row in enumerate(anchor_records):
        seed = derive_row_seed(anchor_row[seed_col])
        rng = np.random.default_rng(seed)

        # Etat de la ligne en cours de construction (ordre topologique).
        row: dict[str, Any] = {}

        for col in ordered_columns:
            pattern = pattern_index.get(col)
            profile = profile_index.get(col)
            if pattern is None:
                # Colonne du schema sans pattern : on copie depuis l'ancre si possible.
                if col in anchor_row:
                    row[col] = anchor_row[col]
                else:
                    # Aucun moyen de la regenerer : on laisse None et on log.
                    logger.debug("No pattern nor anchor for column %r, leaving None", col)
                    row[col] = None
                continue
            if profile is None:
                raise ValueError(
                    f"Pattern targets column {col!r} not present in schema_columns"
                )
            row[col] = _apply_pattern(rng, pattern, row, anchor_row, profile)

        generated_rows.append(row)
        if (i + 1) % log_every == 0:
            logger.debug("reconstruct: %d/%d rows processed", i + 1, n_rows)

    # 4. Assemblage du DataFrame final.
    # On respecte l'ordre original des colonnes (schema_columns), pas l'ordre topo.
    df = pd.DataFrame(generated_rows, columns=schema_col_names, index=anchor_index)

    # 5. Cast final par colonne pour homogeneiser les dtypes.
    for profile in recipe.schema_columns:
        col = profile.name
        if col not in df.columns:
            continue
        if profile.column_type == ColumnType.NUMERIC and profile.dtype and (
            profile.dtype.startswith("int")
            or profile.dtype.startswith("uint")
            or profile.dtype in ("Int64", "Int32")
        ):
            # Cast int : on tolere les NaN en passant par Int64 (pandas nullable).
            try:
                df[col] = df[col].astype("Int64")
                # Si pas de null, on peut downcast en int64 standard pour matcher l'original.
                if df[col].isna().sum() == 0:
                    df[col] = df[col].astype("int64")
            except (TypeError, ValueError) as exc:
                logger.debug("Cast int failed for %s: %s", col, exc)
        elif profile.column_type == ColumnType.BOOLEAN:
            try:
                df[col] = df[col].astype(bool)
            except (TypeError, ValueError):
                pass

    dt = time.perf_counter() - t0
    logger.info(
        "reconstruct done: shape=%s (%.3fs, %.0f rows/s)",
        df.shape,
        dt,
        n_rows / dt if dt > 0 else 0.0,
    )
    return df


# ---------------------------------------------------------------------------
# 9. Convenience : end-to-end depuis fichiers
# ---------------------------------------------------------------------------


def reconstruct_from_files(recipe_path: Path, anchor_path: Path) -> pd.DataFrame:
    """Parse + load + reconstruct : utilitaire pour la CLI et les tests end-to-end."""
    recipe = parse_recipe(Path(recipe_path))
    anchors = load_anchors_parquet(Path(anchor_path))
    return reconstruct(recipe, anchors)


__all__ = [
    "parse_recipe",
    "derive_row_seed",
    "reconstruct",
    "reconstruct_from_files",
]
