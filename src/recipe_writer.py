"""Serialise une Recipe en fichier Markdown lisible ET parsable.

Le format produit suit un contrat strict, partage avec `reconstructor.py` :
- 8 sections markdown niveau 2, dans un ordre fige :
  METADATA, SCHEMA, ANCHORS, PATTERNS, CORRELATIONS,
  FUNCTIONAL_DEPENDENCIES, RECONSTRUCTION_ALGORITHM, VALIDATION_TESTS
- Chaque section "data" est precedee d'un marker HTML `<!-- DATA -->`
  immediatement suivi d'un bloc ```json ... ```
- Le payload JSON provient toujours de `model_dump(mode="json")` (zero
  surprise de serialisation, compatible json.loads)
- Encodage UTF-8 strict, newline LF (`\n`) meme sur Windows

Les patterns sont emis en ordre topologique (dependencies first) via
`graphlib.TopologicalSorter`, pour que `reconstructor` puisse les appliquer
sequentiellement sans avoir a re-trier.
"""

from __future__ import annotations

import json
import logging
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

from .models import Pattern, Recipe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes de format (PARTAGEES avec reconstructor)
# ---------------------------------------------------------------------------

#: Marker place IMMEDIATEMENT avant chaque bloc ```json ... ``` parseable.
#: Le reconstructor utilise ce marker pour reperer les blocs de donnees vs
#: les blocs de code purement illustratifs.
DATA_MARKER = "<!-- DATA -->"

#: Ordre exact (et uppercase) des sections obligatoires.
SECTION_ORDER: tuple[str, ...] = (
    "METADATA",
    "SCHEMA",
    "ANCHORS",
    "PATTERNS",
    "CORRELATIONS",
    "FUNCTIONAL_DEPENDENCIES",
    "RECONSTRUCTION_ALGORITHM",
    "VALIDATION_TESTS",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dump_json(value: Any) -> str:
    """Serialise en JSON formate (2 espaces, UTF-8 preserve) pour les blocs ```json."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _data_block(payload: Any) -> str:
    """Emet le bloc canonique : marker HTML + fence ```json + payload formate + fence."""
    return f"{DATA_MARKER}\n```json\n{_dump_json(payload)}\n```"


def _topo_sort_patterns(
    patterns: list[Pattern], *, break_cycles: bool = False
) -> list[Pattern]:
    """Tri topologique des patterns selon `Pattern.dependencies`.

    Les dependances apparaissent AVANT les patterns qui en dependent.
    En cas d'egalite a un niveau topologique donne, l'ordre d'insertion
    original est preserve (graphlib.TopologicalSorter le garantit).

    Args:
        patterns: liste a trier.
        break_cycles: si True, retire iterativement les edges qui creent
            des cycles (fidelite degradee mais ordonnancement garanti).
            Si False, raise ValueError au premier cycle detecte.

    Raises:
        ValueError: en cas de cycle dans le DAG des dependances et
            `break_cycles=False`.
    """
    if not patterns:
        return []

    # Index : column_name -> Pattern (un seul pattern par colonne par construction).
    by_col: dict[str, Pattern] = {}
    for p in patterns:
        if p.column in by_col:
            # Defensive : on accepte mais on log -- ne devrait pas arriver.
            logger.warning("Duplicate pattern for column %r ; keeping the last one", p.column)
        by_col[p.column] = p

    # Edges : col -> set(deps existantes parmi les patterns connus).
    # Les deps absentes sont ignorees silencieusement (ex : ancres listees
    # comme source d'une FUNCTIONAL_DEP -- resolues en amont par anchor_direct).
    edges: dict[str, set[str]] = {}
    for p in patterns:
        edges[p.column] = {d for d in p.dependencies if d in by_col}

    def _try_sort() -> list[str]:
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for col, deps in edges.items():
            sorter.add(col, *deps)
        return list(sorter.static_order())

    while True:
        try:
            order = _try_sort()
            break
        except CycleError as exc:
            if not break_cycles:
                raise ValueError(f"Pattern dependency cycle detected: {exc.args}") from exc
            # Casse une edge dans le cycle detecte. exc.args == ("nodes are in a cycle", [...]).
            # On retire le dernier edge du cycle : cycle = [a, b, ..., a], on supprime
            # edges[cycle[0]].discard(cycle[1]) qui est l'edge "fermante".
            cycle_nodes = exc.args[1] if len(exc.args) > 1 else []
            if len(cycle_nodes) < 2:
                # Cycle degenere : on abandonne, on retourne l'ordre d'insertion.
                logger.warning("Unable to parse cycle info, falling back to input order")
                return list(patterns)
            # cycle = [a, b, c, a] => edge a -> b est l'edge entrante qui ferme.
            # On supprime b de edges[a].
            a, b = cycle_nodes[0], cycle_nodes[1]
            if b in edges.get(a, set()):
                logger.warning(
                    "Breaking dependency cycle: dropping edge %s -> %s (from cycle %s)",
                    a, b, cycle_nodes,
                )
                edges[a].discard(b)
            else:
                # Cherche n'importe quelle edge dans le cycle a retirer.
                for i in range(len(cycle_nodes) - 1):
                    src, dst = cycle_nodes[i], cycle_nodes[i + 1]
                    if dst in edges.get(src, set()):
                        logger.warning(
                            "Breaking dependency cycle: dropping edge %s -> %s (from cycle %s)",
                            src, dst, cycle_nodes,
                        )
                        edges[src].discard(dst)
                        break
                else:
                    logger.error("Could not break cycle %s, returning input order", cycle_nodes)
                    return list(patterns)

    # Retourne les Pattern dans l'ordre topologique. On filtre les noeuds qui
    # apparaissent comme deps mais n'ont pas de Pattern propre (peut arriver
    # si on relachait la regle de filtre ci-dessus dans le futur).
    return [by_col[col] for col in order if col in by_col]


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def render_summary(recipe: Recipe) -> str:
    """Court resume tete de fichier : `Compression of X : N rows, ratio Y:1, fidelity target Z%`."""
    m = recipe.metadata
    target_pct = int(round(m.fidelity_target * 100))
    return (
        f"Compression of {m.table_name} : "
        f"{m.n_rows} rows, "
        f"ratio {m.compression_ratio:.1f}:1, "
        f"fidelity target {target_pct}%"
    )


def render_reconstruction_algorithm(recipe: Recipe) -> str:
    """Texte libre decrivant l'algo de reconstruction.

    Le contrat ne fixe pas le contenu exact de cette section (parsee comme
    "ignored" par reconstructor). On emet un texte canonique qui decrit les
    4 types de patterns supportes ; le reconstructor les implemente.
    """
    n_anchors = len(recipe.anchor_columns)
    n_patterns = len(recipe.patterns)
    anchor_list = ", ".join(f"`{c}`" for c in recipe.anchor_columns) or "(none)"
    lines = [
        f"For each of the {recipe.metadata.n_rows} rows in `{recipe.anchor_file}`:",
        "",
        "1. Derive a per-row seed from the first anchor column:",
        "   `seed = int(sha256(str(anchor_id).encode()).hexdigest()[:8], 16)`",
        "2. Initialize `rng = numpy.random.default_rng(seed)` for this row.",
        f"3. For each of the {n_patterns} patterns in the topological order listed above,",
        "   apply the rule matching its `pattern_type`:",
        "   - `ANCHOR_DIRECT`: copy the value from the anchors parquet.",
        "   - `DISTRIBUTION`: sample `rng` from the named distribution with `distribution_params`.",
        "   - `FUNCTIONAL_DEP`: lookup `lookup_table[row[source_column]]` (deterministic, no rng draw).",
        "   - `CONDITIONAL_DISTRIBUTION`: locate the bucket containing `row[source_column]`",
        "     (by `bucket_range` for numeric pivots or `bucket_value` for categorical pivots),",
        "     then sample from that bucket's distribution / frequencies.",
        "",
        f"Anchor columns ({n_anchors}): {anchor_list}.",
        "",
        "Reproducibility guarantee: two runs with the same anchors parquet and the same",
        "recipe MUST produce byte-identical reconstructions, because every random draw",
        "is keyed off the deterministic per-row seed derived from the anchor id.",
    ]
    return "\n".join(lines)


def render_recipe_markdown(recipe: Recipe, *, break_cycles: bool = True) -> str:
    """Pure function : Recipe -> str markdown selon le contrat de format.

    Le contrat impose :
        - 8 sections dans l'ordre exact de SECTION_ORDER (## NAME en uppercase).
        - Chaque section data : marker `<!-- DATA -->` + bloc ```json contenant
          `model.model_dump(mode="json")`.
        - Listes vides => bloc ```json contenant `[]` (jamais d'omission de section).
        - Patterns en ordre topologique.

    Args:
        recipe: Recipe a serialiser.
        break_cycles: si True (defaut), casse les cycles de dependances en
            dropping iterativement la 1ere edge fermante de chaque cycle.
            Si False, raise ValueError au premier cycle detecte. La pratique
            (CONDITIONAL_DISTRIBUTION reciproques entre 2 colonnes correlees)
            cree regulierement des cycles 2-edges ; le defaut True les absorbe
            sans casser la pipeline.

    Note sur la sortie des computed fields : `ColumnProfile` expose `cardinality`
    et `null_ratio` comme @computed_field. `model_dump(mode="json")` les inclut,
    ce qui pose probleme au reload (le modele est `extra="forbid"`). On les
    exclut manuellement du payload SCHEMA pour garantir le round-trip.
    """
    # Sanity checks defensifs : on ne touche pas au recipe en entree, on utilise
    # uniquement model_dump qui est cycle-safe.
    metadata_payload = recipe.metadata.model_dump(mode="json")
    schema_payload = [
        c.model_dump(mode="json", exclude={"cardinality", "null_ratio"})
        for c in recipe.schema_columns
    ]
    sorted_patterns = _topo_sort_patterns(list(recipe.patterns), break_cycles=break_cycles)
    patterns_payload = [p.model_dump(mode="json") for p in sorted_patterns]
    correlations_payload = [c.model_dump(mode="json") for c in recipe.correlations]
    fdeps_payload = [fd.model_dump(mode="json") for fd in recipe.functional_dependencies]
    validation_payload = list(recipe.validation_tests)  # deja list[dict]

    anchors_payload = {
        "anchor_file": recipe.anchor_file,
        "anchor_columns": list(recipe.anchor_columns),
    }

    parts: list[str] = []

    # En-tete
    parts.append(f"# RECIPE: {recipe.metadata.table_name}")
    parts.append("")
    parts.append(f"> {render_summary(recipe)}")
    parts.append("")

    # 1. METADATA
    parts.append("## METADATA")
    parts.append("")
    parts.append(_data_block(metadata_payload))
    parts.append("")

    # 2. SCHEMA
    parts.append("## SCHEMA")
    parts.append("")
    parts.append(f"> {len(schema_payload)} columns described below.")
    parts.append("")
    parts.append(_data_block(schema_payload))
    parts.append("")

    # 3. ANCHORS
    parts.append("## ANCHORS")
    parts.append("")
    parts.append("> Stored separately as a parquet file.")
    parts.append("")
    parts.append(f"- **File**: `{recipe.anchor_file}`")
    if recipe.anchor_columns:
        cols_md = ", ".join(f"`{c}`" for c in recipe.anchor_columns)
    else:
        cols_md = "(none)"
    parts.append(f"- **Columns**: {cols_md}")
    parts.append(f"- **Size**: {recipe.metadata.anchor_size_bytes} bytes")
    parts.append("")
    parts.append(_data_block(anchors_payload))
    parts.append("")

    # 4. PATTERNS
    parts.append("## PATTERNS")
    parts.append("")
    parts.append("> One Pattern per column, in topological order (dependencies first).")
    parts.append("")
    parts.append(_data_block(patterns_payload))
    parts.append("")

    # 5. CORRELATIONS
    parts.append("## CORRELATIONS")
    parts.append("")
    parts.append(
        "> Detected significant correlations (informational; not used directly by "
        "reconstructor since conditional distributions encode the relationships)."
    )
    parts.append("")
    parts.append(_data_block(correlations_payload))
    parts.append("")

    # 6. FUNCTIONAL_DEPENDENCIES
    parts.append("## FUNCTIONAL_DEPENDENCIES")
    parts.append("")
    parts.append(_data_block(fdeps_payload))
    parts.append("")

    # 7. RECONSTRUCTION_ALGORITHM (texte libre, pas de bloc data)
    parts.append("## RECONSTRUCTION_ALGORITHM")
    parts.append("")
    parts.append("> Plain text description of how to reconstruct from anchors + patterns.")
    parts.append("")
    parts.append(render_reconstruction_algorithm(recipe))
    parts.append("")

    # 8. VALIDATION_TESTS
    parts.append("## VALIDATION_TESTS")
    parts.append("")
    parts.append(_data_block(validation_payload))
    parts.append("")

    # Newline trailing pour respect POSIX text-file convention.
    return "\n".join(parts)


def write_recipe(recipe: Recipe, output_path: Path) -> Path:
    """Ecrit la recette en .md UTF-8 / LF. Met a jour `recipe.metadata.recipe_size_bytes`.

    Strategie pour `recipe_size_bytes` :
        - On rend une 1ere fois avec la valeur actuelle de `recipe_size_bytes`.
        - On ecrit le fichier, on mesure sa taille reelle.
        - On met a jour `recipe.metadata.recipe_size_bytes` en memoire.
        - Le caller decide s'il veut persister la mise a jour (typiquement non,
          puisque la taille reelle apres re-ecriture pourrait differer de 1-2 bytes
          a cause du chiffre lui-meme, ce qui creerait une boucle si on ecrit
          a chaque fois).

    Args:
        recipe: Recipe a serialiser.
        output_path: chemin du .md a ecrire (parents crees si besoin).

    Returns:
        Le chemin absolu du fichier ecrit.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = render_recipe_markdown(recipe)
    # newline="" + write(content) : on a deja insere des '\n' manuellement,
    # on veut absolument PAS que Windows convertisse en \r\n.
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    actual_size = output_path.stat().st_size
    # Mise a jour in-memory uniquement (cf. docstring).
    # validate_assignment est True dans _StrictModel : Pydantic re-validera.
    recipe.metadata.recipe_size_bytes = actual_size

    logger.info(
        "write_recipe wrote %s (%d bytes, table=%s, %d patterns, %d anchors)",
        output_path,
        actual_size,
        recipe.metadata.table_name,
        len(recipe.patterns),
        len(recipe.anchor_columns),
    )

    return output_path.resolve()


__all__ = [
    "DATA_MARKER",
    "SECTION_ORDER",
    "render_recipe_markdown",
    "render_summary",
    "render_reconstruction_algorithm",
    "write_recipe",
]
