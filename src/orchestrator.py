"""Orchestrator : pipeline end-to-end et CLI du semantic-compressor.

Cable les modules unitaires (profiler, pattern_detector, anchor_extractor,
recipe_writer, reconstructor, validator) en trois operations principales :

- `compress(input_csv, output_dir)`         : CSV -> recipe.md + anchors.parquet
- `decompress(recipe_path, output_csv)`     : recipe + ancres -> CSV reconstruit
- `validate_pair(original, reconstructed)`  : compare + score de fidelite

Et expose une CLI a quatre sous-commandes (`compress`, `decompress`, `validate`,
`run-poc`). La sous-commande `run-poc` enchaine les trois et affiche un rapport
Rich complet.

Notes d'implementation :
- La dependance cyclique anchors <-> patterns (extract_anchors a besoin de patterns
  pour detecter les colonnes "sans pattern" ; build_patterns a besoin de la liste
  d'ancres pour les marquer ANCHOR_DIRECT) est resolue par double passe :
  pass 1 = ancres heuristiques (cardinalite + manuelles) ; pass 2 = patterns ;
  pass 3 = ancres finales (au cas ou pass 1 aurait manque des colonnes "no pattern").
- Pour le parquet zstd avec niveau >= 4 (snappy n'a pas de niveau) on ne passe pas
  par `df.to_parquet` mais directement par `pyarrow.parquet.write_table`, qui
  expose `compression_level`. Wrapper interne : `_write_parquet_with_level`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .anchor_extractor import extract_anchors, load_anchors_parquet
from .models import (
    ColumnProfile,
    ColumnType,
    Pattern,
    PatternType,
    Recipe,
    RecipeMetadata,
    ValidationReport,
)
from .pattern_detector import build_patterns
from .profiler import generate_html_report, profile_dataframe
from .recipe_writer import write_recipe
from .reconstructor import parse_recipe, reconstruct
from .validator import ValidationThresholds, print_report, validate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de retour
# ---------------------------------------------------------------------------


@dataclass
class CompressionResult:
    """Resultat structure d'une operation de compression."""

    recipe_path: Path
    anchor_path: Path
    original_size_bytes: int
    recipe_size_bytes: int
    anchor_size_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    elapsed_seconds: float
    n_rows: int
    n_columns: int
    anchor_columns: list[str]
    n_patterns: int
    html_profile_path: Path | None = None


@dataclass
class ReconstructionResult:
    """Resultat structure d'une operation de decompression."""

    output_path: Path
    n_rows: int
    n_columns: int
    elapsed_seconds: float


@dataclass
class ValidationResultBundle:
    """Bundle de retour pour validate_pair : rapport + temps mesure."""

    report: ValidationReport
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet_with_level(
    df: pd.DataFrame,
    path: Path,
    codec: str,
    level: int | None,
) -> int:
    """Ecrit un DataFrame en parquet avec controle du niveau de compression.

    `df.to_parquet` n'expose pas `compression_level` ; on passe par
    `pyarrow.parquet.write_table` qui le fait. Si `level` est None ou si le
    codec ne le supporte pas, on retombe sur les defauts.

    Args:
        df: DataFrame d'ancres a serialiser.
        path: chemin du parquet de sortie.
        codec: snappy / zstd / gzip / brotli / lz4.
        level: niveau de compression (entier, sens depend du codec).

    Returns:
        Taille reelle du fichier ecrit, en bytes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=True)

    # snappy ne supporte PAS compression_level (pyarrow leve InvalidArgument).
    # Les autres codecs l'acceptent.
    write_kwargs: dict[str, Any] = {"compression": codec}
    if level is not None and codec != "snappy":
        write_kwargs["compression_level"] = int(level)

    try:
        pq.write_table(table, str(path), **write_kwargs)
    except Exception as exc:
        # Fallback : si pyarrow refuse le niveau pour ce codec, on retente sans.
        logger.warning(
            "pyarrow.write_table failed with codec=%s level=%s (%s); retrying without level",
            codec, level, exc,
        )
        pq.write_table(table, str(path), compression=codec)

    size = path.stat().st_size
    logger.info(
        "Wrote anchors parquet to %s (codec=%s, level=%s, %d bytes)",
        path, codec, level, size,
    )
    return size


def _detect_unique_anchor_columns(df: pd.DataFrame) -> list[str]:
    """Detecte les colonnes uniques (cardinalite stricte == 1.0) pour fallback validate."""
    n = len(df)
    if n == 0:
        return []
    out = []
    for col in df.columns:
        try:
            if df[col].nunique(dropna=True) == n and df[col].isnull().sum() == 0:
                out.append(col)
        except (TypeError, ValueError):
            continue
    return out


def _extract_random_format_columns_from_recipe(recipe: Recipe) -> dict[str, str]:
    """Extrait le mapping `{col: format_regex}` pour les colonnes RANDOM_FORMAT d'une recette.

    Utile pour appeler `validate` avec le bon parametre `random_format_columns`
    apres une decompression.
    """
    out: dict[str, str] = {}
    for pat in recipe.patterns:
        if pat.pattern_type == PatternType.RANDOM_FORMAT and pat.format_spec is not None:
            regex = pat.format_spec.get("regex")
            if regex:
                out[pat.column] = regex
    return out


def _normalize_anchor_columns_for_validation(
    anchor_columns: list[str],
) -> list[str]:
    """Remplace les ancres synthetiques EMAIL_SPLIT (`*__local`, `*__domain_idx`)
    par leur colonne racine pour la validation.

    Les ancres EMAIL_SPLIT n'existent que dans le parquet d'ancres. Pour le test
    `anchor_exact` du validator, qui compare original (CSV brut) vs reconstruit
    (CSV brut), on remappe vers la colonne email originale (qui doit etre
    losslessly identique apres reconstruction).
    """
    out: list[str] = []
    seen: set[str] = set()
    for col in anchor_columns:
        root: str
        if col.endswith("__local"):
            root = col[: -len("__local")]
        elif col.endswith("__domain_idx"):
            root = col[: -len("__domain_idx")]
        else:
            root = col
        if root not in seen:
            out.append(root)
            seen.add(root)
    return out


def _build_validation_tests(
    profiles: list[ColumnProfile], n_rows: int
) -> list[dict[str, Any]]:
    """Genere une liste minimale de tests recommandes a inclure dans la recette.

    On emet :
    - row_count        : le compte de lignes attendu
    - schema_match     : pour memoire (verifie par le validator de toute facon)
    - ks_test_<col>    : pour chaque colonne numerique ou datetime
    """
    tests: list[dict[str, Any]] = [
        {"test_name": "row_count", "expected": int(n_rows)},
        {"test_name": "schema_match", "expected": [p.name for p in profiles]},
    ]
    for profile in profiles:
        if profile.column_type in (ColumnType.NUMERIC, ColumnType.DATETIME):
            tests.append(
                {
                    "test_name": f"ks_test_{profile.name}",
                    "threshold": 0.05,
                    "metric": "ks_pvalue",
                }
            )
    return tests


# ---------------------------------------------------------------------------
# Pipeline : compress
# ---------------------------------------------------------------------------


def compress(
    input_csv: Path,
    output_dir: Path,
    table_name: str | None = None,
    manual_anchor_columns: list[str] | None = None,
    manual_random_format_columns: list[str] | None = None,
    parquet_codec: str = "zstd",
    parquet_compression_level: int | None = 22,
    generate_html_profile: bool = False,
    aggressive_uuid: bool = False,
) -> CompressionResult:
    """Pipeline complet de compression CSV -> (recipe.md, anchors.parquet).

    Args:
        input_csv: chemin du CSV source a compresser.
        output_dir: repertoire de sortie ; on y cree `recipes/` et `anchors/`.
        table_name: nom logique de la table (defaut : nom du fichier sans extension).
        manual_anchor_columns: colonnes a forcer en ancre (en plus de la detection).
        manual_random_format_columns: colonnes a forcer comme RANDOM_FORMAT (valeur
            non preservee, regeneree depuis un format_spec). Utile pour les
            colonnes du genre `password_hash` ou la valeur exacte ne porte pas
            d'information mais le format si.
        parquet_codec: codec de compression parquet (zstd / snappy / gzip / brotli).
        parquet_compression_level: niveau de compression (22 = max zstd ; ignore par snappy).
        generate_html_profile: si True, genere aussi le rapport ydata-profiling.
        aggressive_uuid: si True, force les colonnes UUID v4 dont le nom ressemble
            a un id (id, uuid, *_id, *_uuid, etc.) a etre marquees RANDOM_FORMAT
            (gain de compression ~3-5x sur ces colonnes au prix de la
            preservation de la valeur exacte des ids). Defaut False : on
            preserve les ids comme ANCHOR_DIRECT (mode fidelity-first).

    Returns:
        CompressionResult avec metriques et chemins de sortie.
    """
    t0 = time.perf_counter()

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    if table_name is None:
        table_name = input_csv.stem

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    original_size_bytes = input_csv.stat().st_size
    logger.info(
        "compress start: input=%s (%d bytes) output_dir=%s table=%s",
        input_csv, original_size_bytes, output_dir, table_name,
    )

    # 1. Lecture CSV. PAS de parse_dates : on preserve les strings ISO telles que
    # le CSV les contient (le profileur les detecte comme DATETIME via le sniff).
    df = pd.read_csv(input_csv)
    n_rows = len(df)
    n_columns = len(df.columns)
    logger.info("  loaded %d rows x %d columns", n_rows, n_columns)

    # 2. Profiling
    profiles = profile_dataframe(df, table_name=table_name)

    # 2b. Optionnel : rapport HTML d'exploration.
    # Si la generation echoue (timeout, version Python incompatible, etc.), on
    # log un warning mais on continue le pipeline : la compression elle-meme ne
    # doit jamais etre bloquee par un echec du profiler exterieur.
    html_profile_path: Path | None = None
    if generate_html_profile:
        html_target = output_dir / "profiling_reports" / f"{table_name}.html"
        try:
            generate_html_report(df, html_target, minimal=True)
            html_profile_path = html_target
            logger.info("  HTML profile written to %s", html_profile_path)
        except Exception as exc:
            logger.warning(
                "HTML profile generation failed (%s); skipping (pipeline continues)",
                exc,
            )
            html_profile_path = None

    # 3. Resolution de la dependance cyclique anchors <-> patterns en triple passe.
    # Pass 1 : ancres heuristiques (cardinalite stricte + manuelles).
    # On passe random_format_columns pour exclure ces colonnes des ancres des le pass 1.
    _prelim_anchors_df, prelim_anchor_cols = extract_anchors(
        df, profiles,
        patterns=None,
        manual_anchor_columns=manual_anchor_columns,
        random_format_columns=manual_random_format_columns,
    )
    logger.info("  pass 1 (prelim anchors): %s", prelim_anchor_cols)

    # Pass 2 : on construit les patterns avec ces ancres comme reference. Le
    # forcage manuel de RANDOM_FORMAT est passe a build_patterns ; les colonnes
    # ancres candidates restantes peuvent aussi etre auto-detectees comme
    # RANDOM_FORMAT (ex: si on n'a pas force et qu'une colonne unique matche
    # KNOWN_RANDOM_FORMATS, build_patterns la marque RANDOM_FORMAT).
    patterns: list[Pattern] = build_patterns(
        df, profiles,
        anchor_columns=prelim_anchor_cols,
        manual_random_format_columns=manual_random_format_columns,
        aggressive_uuid=aggressive_uuid,
    )
    logger.info("  pass 2 (patterns): %d patterns built", len(patterns))

    # Pass 3 : ancres finales. Les patterns RANDOM_FORMAT et EMAIL_SPLIT
    # excluent les colonnes concernees des ancres (via extract_anchors qui
    # regarde patterns). Pour EMAIL_SPLIT, des colonnes synthetiques
    # `<col>__local` et `<col>__domain_idx` sont ajoutees a la place.
    final_anchors_df, final_anchor_cols = extract_anchors(
        df, profiles,
        patterns=patterns,
        manual_anchor_columns=manual_anchor_columns,
        random_format_columns=manual_random_format_columns,
    )
    logger.info("  pass 3 (final anchors): %s", final_anchor_cols)

    # Si pass 3 a ajoute des ancres par rapport a pass 1, on doit reconstruire les
    # patterns avec cette nouvelle liste pour eviter qu'une colonne soit a la fois
    # ancre ET avec un pattern non-ancre (qui aurait priorite sur l'ANCHOR_DIRECT).
    # Note : on filtre les colonnes synthetiques EMAIL_SPLIT (`*__local`,
    # `*__domain_idx`) du test d'egalite : elles sont AJOUTEES par extract_anchors
    # mais ne sont PAS des candidates ancres dans le df d'origine, leur presence
    # ne doit pas declencher une rebuild des patterns.
    final_real_anchors = [
        c for c in final_anchor_cols
        if not (c.endswith("__local") or c.endswith("__domain_idx"))
    ]
    if set(final_real_anchors) != set(prelim_anchor_cols):
        logger.info(
            "  anchor set changed between pass 1 and pass 3 (%s -> %s); rebuilding patterns",
            prelim_anchor_cols, final_real_anchors,
        )
        patterns = build_patterns(
            df, profiles,
            anchor_columns=final_real_anchors,
            manual_random_format_columns=manual_random_format_columns,
            aggressive_uuid=aggressive_uuid,
        )

    anchor_path = output_dir / "anchors" / f"{table_name}_anchors.parquet"
    recipe_path = output_dir / "recipes" / f"{table_name}.md"

    # 4. Ecriture des ancres avec niveau de compression controle.
    anchor_size_bytes = _write_parquet_with_level(
        final_anchors_df,
        anchor_path,
        codec=parquet_codec,
        level=parquet_compression_level,
    )

    # 5. Construction de la Recipe.
    # Le chemin d'ancre est stocke relativement au repertoire du recipe : si le
    # POC est deplace, les chemins restent valides tant que la structure est
    # preservee. recipes/{table}.md + anchors/{table}_anchors.parquet -> "../anchors/...".
    # On utilise os.path.relpath qui gere proprement le cas ".." (Path.relative_to
    # echoue si la cible n'est pas un sous-chemin du depart).
    import os as _os

    try:
        rel = _os.path.relpath(anchor_path.resolve(), start=recipe_path.parent.resolve())
        anchor_file_rel = Path(rel).as_posix()
    except (ValueError, OSError):
        # Cas degenere (volumes differents sous Windows par exemple) : on tombe
        # sur le chemin absolu, ce qui reste resolvable.
        anchor_file_rel = anchor_path.resolve().as_posix()

    # On calcule le ratio sur la taille totale (ancres + recette estimee).
    # La taille de la recette n'est pas encore connue : on emet une 1ere version
    # avec recipe_size=0 pour write_recipe qui mettra a jour in-memory.
    estimated_total = max(anchor_size_bytes, 1)
    metadata = RecipeMetadata(
        table_name=table_name,
        n_rows=n_rows,
        original_size_bytes=original_size_bytes,
        anchor_size_bytes=anchor_size_bytes,
        recipe_size_bytes=0,  # sera ecrase par write_recipe
        compression_ratio=original_size_bytes / estimated_total,
    )

    recipe = Recipe(
        metadata=metadata,
        schema_columns=profiles,
        anchor_columns=final_anchor_cols,
        anchor_file=anchor_file_rel,
        patterns=patterns,
        correlations=[],  # detecte dans build_patterns mais pas remonte ; OK pour POC.
        functional_dependencies=[],
        validation_tests=_build_validation_tests(profiles, n_rows),
    )

    # 6. Ecriture de la recette (met a jour metadata.recipe_size_bytes in-memory).
    write_recipe(recipe, recipe_path)
    recipe_size_bytes = recipe.metadata.recipe_size_bytes

    # 7. Recalcul du ratio avec la taille reelle de la recette.
    total_compressed_bytes = recipe_size_bytes + anchor_size_bytes
    compression_ratio = (
        original_size_bytes / total_compressed_bytes
        if total_compressed_bytes > 0
        else 0.0
    )
    # On met a jour la metadata en memoire (le fichier .md n'est pas reecrit :
    # on accepte l'ecart de 1-2 octets sur le ratio inscrit dans le .md, qui
    # reste honnete a 1% pres).
    recipe.metadata.compression_ratio = compression_ratio

    elapsed = time.perf_counter() - t0
    logger.info(
        "compress done in %.3fs: ratio=%.2f:1 (recipe=%d B + anchors=%d B vs original=%d B)",
        elapsed, compression_ratio, recipe_size_bytes, anchor_size_bytes, original_size_bytes,
    )

    return CompressionResult(
        recipe_path=recipe_path,
        anchor_path=anchor_path,
        original_size_bytes=original_size_bytes,
        recipe_size_bytes=recipe_size_bytes,
        anchor_size_bytes=anchor_size_bytes,
        total_compressed_bytes=total_compressed_bytes,
        compression_ratio=compression_ratio,
        elapsed_seconds=elapsed,
        n_rows=n_rows,
        n_columns=n_columns,
        anchor_columns=final_anchor_cols,
        n_patterns=len(patterns),
        html_profile_path=html_profile_path,
    )


# ---------------------------------------------------------------------------
# Pipeline : decompress
# ---------------------------------------------------------------------------


def decompress(
    recipe_path: Path,
    output_csv: Path,
    anchor_path: Path | None = None,
) -> ReconstructionResult:
    """Reconstruit un CSV depuis une recette + son parquet d'ancres.

    Args:
        recipe_path: chemin du `.md` de recette.
        output_csv: chemin du CSV a generer.
        anchor_path: chemin du parquet d'ancres ; si None, derive depuis
            `recipe.anchor_file` (interprete relatif au dossier du recipe).

    Returns:
        ReconstructionResult avec metriques.
    """
    t0 = time.perf_counter()
    recipe_path = Path(recipe_path)
    output_csv = Path(output_csv)

    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    logger.info("decompress start: recipe=%s -> output=%s", recipe_path, output_csv)

    recipe = parse_recipe(recipe_path)

    if anchor_path is None:
        # `recipe.anchor_file` est stocke relatif au dossier du recipe (cf. compress).
        anchor_path = (recipe_path.parent / recipe.anchor_file).resolve()
    else:
        anchor_path = Path(anchor_path)
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"Anchor parquet not found: {anchor_path} (derived from recipe.anchor_file={recipe.anchor_file!r})"
        )

    anchors_df = load_anchors_parquet(anchor_path)
    reconstructed = reconstruct(recipe, anchors_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reconstructed.to_csv(output_csv, index=False)

    elapsed = time.perf_counter() - t0
    n_rows = len(reconstructed)
    n_columns = len(reconstructed.columns)
    logger.info(
        "decompress done in %.3fs: wrote %d rows x %d cols to %s",
        elapsed, n_rows, n_columns, output_csv,
    )

    return ReconstructionResult(
        output_path=output_csv,
        n_rows=n_rows,
        n_columns=n_columns,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Pipeline : validate_pair
# ---------------------------------------------------------------------------


def validate_pair(
    original_csv: Path,
    reconstructed_csv: Path,
    anchor_columns: list[str] | None = None,
    random_format_columns: dict[str, str] | None = None,
) -> ValidationResultBundle:
    """Lit deux CSVs et compare leur fidelite statistique.

    Args:
        original_csv: chemin du CSV original.
        reconstructed_csv: chemin du CSV reconstruit.
        anchor_columns: colonnes a valider en strict (egalite exacte). Si None,
            on detecte automatiquement les colonnes a cardinalite 1.0 dans
            l'original.
        random_format_columns: mapping `{col: format_regex}` pour les colonnes
            RANDOM_FORMAT (test softs au lieu de comparaison stricte). Voir
            `validator.test_random_format_compliance`.

    Returns:
        ValidationResultBundle (report + temps ecoule).
    """
    t0 = time.perf_counter()
    original_csv = Path(original_csv)
    reconstructed_csv = Path(reconstructed_csv)

    if not original_csv.exists():
        raise FileNotFoundError(f"Original CSV not found: {original_csv}")
    if not reconstructed_csv.exists():
        raise FileNotFoundError(f"Reconstructed CSV not found: {reconstructed_csv}")

    logger.info(
        "validate_pair start: original=%s vs reconstructed=%s",
        original_csv, reconstructed_csv,
    )

    # Pas de parse_dates : on compare les strings telles quelles (cf. compress).
    original = pd.read_csv(original_csv)
    reconstructed = pd.read_csv(reconstructed_csv)

    if anchor_columns is None:
        anchor_columns = _detect_unique_anchor_columns(original)
        logger.info("  auto-detected anchor columns: %s", anchor_columns)

    # Normalisation EMAIL_SPLIT : les ancres synthetiques `*__local` et
    # `*__domain_idx` sont remappees vers leur colonne racine (email originale)
    # pour le test `anchor_exact` qui compare des colonnes CSV.
    normalized_anchors = _normalize_anchor_columns_for_validation(anchor_columns)
    if normalized_anchors != anchor_columns:
        logger.info(
            "  normalized anchor columns for validation: %s -> %s",
            anchor_columns, normalized_anchors,
        )

    report = validate(
        original, reconstructed,
        anchor_columns=normalized_anchors,
        random_format_columns=random_format_columns,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "validate_pair done in %.3fs: score=%.2f passed=%d failed=%d",
        elapsed, report.overall_score, report.passed_count, report.failed_count,
    )

    return ValidationResultBundle(report=report, elapsed_seconds=elapsed)


# ---------------------------------------------------------------------------
# Affichage : run-poc report
# ---------------------------------------------------------------------------


def _format_bytes(n: int) -> str:
    """Format en `12345 B (12 KB)` ou `1.2 MB` selon taille."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n:,} B ({n / 1024:.1f} KB)"
    return f"{n:,} B ({n / (1024 * 1024):.2f} MB)"


def _color_for_ratio(ratio: float) -> str:
    """Vert si >= 5, jaune si >= 3, rouge sinon (cf. brief)."""
    if ratio >= 5.0:
        return "bright_green"
    if ratio >= 3.0:
        return "yellow"
    return "red"


def _color_for_score(score: float) -> str:
    """Vert si >= 95, jaune si >= 80, rouge sinon (cf. brief)."""
    if score >= 95.0:
        return "bright_green"
    if score >= 80.0:
        return "yellow"
    return "red"


def _print_poc_report(
    compress_result: CompressionResult,
    decompress_result: ReconstructionResult,
    validate_result: ValidationResultBundle,
    total_elapsed: float,
) -> None:
    """Affiche le MEGA-tableau Rich qui resume tout le run."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    ratio = compress_result.compression_ratio
    score = validate_result.report.overall_score
    ratio_color = _color_for_ratio(ratio)
    score_color = _color_for_score(score)

    table = Table(
        title="SEMANTIC COMPRESSOR - POC REPORT",
        title_style="bold cyan",
        box=box.HEAVY,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="left")

    # ----- Tailles -----
    table.add_row("Original size", _format_bytes(compress_result.original_size_bytes))
    table.add_row("Recipe size", _format_bytes(compress_result.recipe_size_bytes))
    table.add_row(
        "Anchors size (zstd)",
        _format_bytes(compress_result.anchor_size_bytes),
    )
    table.add_row(
        "Total compressed",
        _format_bytes(compress_result.total_compressed_bytes),
    )
    table.add_section()

    # ----- Ratio + Fidelite -----
    table.add_row(
        "Compression ratio",
        f"[{ratio_color}]{ratio:.2f}:1[/]",
    )
    target_str = (
        f"[bright_green]TARGET MET[/]"
        if score >= 95.0
        else f"[{score_color}]target >= 95[/]"
    )
    table.add_row(
        "Fidelity score",
        f"[{score_color}]{score:.2f} / 100[/]  ({target_str})",
    )
    table.add_section()

    # ----- Temps -----
    table.add_row("Compress time", f"{compress_result.elapsed_seconds:.2f} s")
    table.add_row("Decompress time", f"{decompress_result.elapsed_seconds:.2f} s")
    table.add_row("Validate time", f"{validate_result.elapsed_seconds:.2f} s")
    table.add_row("Total wall time", f"[bold]{total_elapsed:.2f} s[/]")
    table.add_section()

    # ----- Meta -----
    anchor_summary = ", ".join(compress_result.anchor_columns) or "(none)"
    table.add_row(
        "Anchor columns",
        f"{len(compress_result.anchor_columns)} ({anchor_summary})",
    )
    table.add_row("Patterns built", str(compress_result.n_patterns))
    table.add_row(
        "Rows x Columns",
        f"{compress_result.n_rows} x {compress_result.n_columns}",
    )

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI a sous-commandes."""
    parser = argparse.ArgumentParser(
        prog="python -m src.orchestrator",
        description="Semantic compressor : compress / decompress / validate / run-poc.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- compress ---
    p_compress = subparsers.add_parser("compress", help="Compress a CSV into recipe + anchors.")
    p_compress.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    p_compress.add_argument("--output", type=Path, required=True, help="Output directory.")
    p_compress.add_argument("--table-name", type=str, default=None, help="Logical table name.")
    p_compress.add_argument(
        "--manual-anchors",
        type=str,
        default=None,
        help="Comma-separated list of columns to force as anchors.",
    )
    p_compress.add_argument(
        "--random-format",
        type=str,
        default=None,
        help=(
            "Comma-separated list of columns to mark as RANDOM_FORMAT: "
            "their exact value is not preserved across reconstruction, only "
            "the format spec (regex/length) is stored in the recipe and unique "
            "values are regenerated at runtime. Typical use: password_hash. "
            "These columns are excluded from anchors."
        ),
    )
    p_compress.add_argument(
        "--codec",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4"],
        help="Parquet compression codec (default: zstd).",
    )
    p_compress.add_argument(
        "--level",
        type=int,
        default=22,
        help="Parquet compression level (default: 22 for zstd).",
    )
    p_compress.add_argument(
        "--generate-html-profile",
        action="store_true",
        help="Also generate an HTML profile with ydata-profiling.",
    )
    p_compress.add_argument(
        "--aggressive-uuid",
        action="store_true",
        help=(
            "Force UUID v4 columns with id-like names (id, uuid, *_id, *_uuid, ...) "
            "to be RANDOM_FORMAT (not preserved). Default: keep them as ANCHOR_DIRECT "
            "to preserve FK references. Use this to gain ~3-5x compression on id columns "
            "when exact id preservation is not required (synthetic data, audit logs, etc.)."
        ),
    )

    # --- decompress ---
    p_decompress = subparsers.add_parser(
        "decompress", help="Reconstruct a CSV from a recipe + anchors."
    )
    p_decompress.add_argument("--recipe", type=Path, required=True, help="Recipe `.md` path.")
    p_decompress.add_argument(
        "--output", type=Path, required=True, help="Output CSV path for the reconstruction."
    )
    p_decompress.add_argument(
        "--anchors",
        type=Path,
        default=None,
        help="Anchor parquet path (default: derived from the recipe).",
    )

    # --- validate ---
    p_validate = subparsers.add_parser(
        "validate", help="Compare an original CSV with its reconstruction."
    )
    p_validate.add_argument("--original", type=Path, required=True, help="Original CSV path.")
    p_validate.add_argument(
        "--reconstructed", type=Path, required=True, help="Reconstructed CSV path."
    )
    p_validate.add_argument(
        "--anchor-columns",
        type=str,
        default=None,
        help="Comma-separated list of anchor columns (default: auto-detect).",
    )

    # --- run-poc ---
    p_run = subparsers.add_parser(
        "run-poc", help="Run compress -> decompress -> validate end-to-end."
    )
    p_run.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    p_run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for recipe and anchors (default: output/).",
    )
    p_run.add_argument(
        "--reconstructed-csv",
        type=Path,
        default=None,
        help=(
            "Where to write the reconstructed CSV "
            "(default: data/reconstructed/<table>.csv)."
        ),
    )
    p_run.add_argument(
        "--codec",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4"],
        help="Parquet compression codec (default: zstd).",
    )
    p_run.add_argument(
        "--level",
        type=int,
        default=22,
        help="Parquet compression level (default: 22 for zstd).",
    )
    p_run.add_argument(
        "--manual-anchors",
        type=str,
        default=None,
        help="Comma-separated list of columns to force as anchors.",
    )
    p_run.add_argument(
        "--random-format",
        type=str,
        default=None,
        help=(
            "Comma-separated list of columns to mark as RANDOM_FORMAT "
            "(value not preserved, regenerated from format spec at runtime). "
            "Typical use: password_hash."
        ),
    )
    p_run.add_argument(
        "--aggressive-uuid",
        action="store_true",
        help=(
            "Force UUID v4 columns with id-like names (id, uuid, *_id, *_uuid, ...) "
            "to be RANDOM_FORMAT. See compress --help."
        ),
    )

    return parser


def _parse_csv_list(value: str | None) -> list[str] | None:
    """Parse "a,b,c" en ["a", "b", "c"] ; None / "" -> None."""
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _cmd_compress(args: argparse.Namespace) -> int:
    """Implementation de la sous-commande compress."""
    from rich.console import Console

    manual = _parse_csv_list(args.manual_anchors)
    random_format = _parse_csv_list(args.random_format)
    result = compress(
        input_csv=args.input,
        output_dir=args.output,
        table_name=args.table_name,
        manual_anchor_columns=manual,
        manual_random_format_columns=random_format,
        parquet_codec=args.codec,
        parquet_compression_level=args.level,
        generate_html_profile=args.generate_html_profile,
        aggressive_uuid=getattr(args, "aggressive_uuid", False),
    )

    console = Console()
    console.print(
        f"[bright_green]Compressed[/] {result.original_size_bytes:,} B -> "
        f"{result.total_compressed_bytes:,} B "
        f"(ratio [bold]{result.compression_ratio:.2f}:1[/], "
        f"{result.elapsed_seconds:.2f} s)"
    )
    console.print(f"  Recipe : {result.recipe_path}")
    console.print(f"  Anchors: {result.anchor_path}")
    return 0


def _cmd_decompress(args: argparse.Namespace) -> int:
    """Implementation de la sous-commande decompress."""
    from rich.console import Console

    result = decompress(
        recipe_path=args.recipe,
        output_csv=args.output,
        anchor_path=args.anchors,
    )
    console = Console()
    console.print(
        f"[bright_green]Reconstructed[/] {result.n_rows} rows x {result.n_columns} cols "
        f"in {result.elapsed_seconds:.2f} s -> {result.output_path}"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Implementation de la sous-commande validate."""
    from rich.console import Console

    anchor_cols = _parse_csv_list(args.anchor_columns)
    bundle = validate_pair(
        original_csv=args.original,
        reconstructed_csv=args.reconstructed,
        anchor_columns=anchor_cols,
    )
    print_report(bundle.report)

    console = Console()
    color = _color_for_score(bundle.report.overall_score)
    console.print(
        f"[{color}]Validation done[/] in {bundle.elapsed_seconds:.2f} s "
        f"(score [bold]{bundle.report.overall_score:.2f}/100[/])"
    )
    return 0 if bundle.report.overall_score >= 50.0 else 1


def _cmd_run_poc(args: argparse.Namespace) -> int:
    """Implementation de la sous-commande run-poc : compress -> decompress -> validate."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    t_total = time.perf_counter()

    # Header
    console.print()
    header = Text.assemble(
        ("SEMANTIC COMPRESSOR\n", "bold bright_cyan"),
        ("End-to-end POC run (compress -> decompress -> validate)", "italic"),
    )
    console.print(Panel(header, expand=False, border_style="bright_cyan"))

    # 1. Compress
    console.print("\n[bold]Step 1 / 3 :[/] compress")
    manual = _parse_csv_list(args.manual_anchors)
    random_format = _parse_csv_list(args.random_format)
    compress_result = compress(
        input_csv=args.input,
        output_dir=args.output_dir,
        manual_anchor_columns=manual,
        manual_random_format_columns=random_format,
        parquet_codec=args.codec,
        parquet_compression_level=args.level,
        aggressive_uuid=getattr(args, "aggressive_uuid", False),
    )
    console.print(
        f"  -> recipe {compress_result.recipe_path}, anchors {compress_result.anchor_path}"
    )

    # 2. Decompress
    console.print("\n[bold]Step 2 / 3 :[/] decompress")
    table_name = compress_result.recipe_path.stem
    if args.reconstructed_csv is not None:
        reconstructed_csv = Path(args.reconstructed_csv)
    else:
        reconstructed_csv = Path("data") / "reconstructed" / f"{table_name}.csv"
    decompress_result = decompress(
        recipe_path=compress_result.recipe_path,
        output_csv=reconstructed_csv,
    )
    console.print(f"  -> reconstructed {decompress_result.output_path}")

    # 3. Validate
    console.print("\n[bold]Step 3 / 3 :[/] validate")
    # Pour les colonnes RANDOM_FORMAT, on relit la recette pour extraire
    # leur format_regex et passer ce mapping au validator.
    recipe = parse_recipe(compress_result.recipe_path)
    random_format_map = _extract_random_format_columns_from_recipe(recipe)
    validate_result = validate_pair(
        original_csv=args.input,
        reconstructed_csv=reconstructed_csv,
        anchor_columns=compress_result.anchor_columns,
        random_format_columns=random_format_map or None,
    )
    console.print(
        f"  -> {validate_result.report.passed_count} passed, "
        f"{validate_result.report.failed_count} failed"
    )

    # MEGA-tableau final
    total_elapsed = time.perf_counter() - t_total
    _print_poc_report(compress_result, decompress_result, validate_result, total_elapsed)

    # Exit code : 0 si pipeline complet, meme avec score bas (le rapport parle de lui-meme).
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


_COMMAND_DISPATCH = {
    "compress": _cmd_compress,
    "decompress": _cmd_decompress,
    "validate": _cmd_validate,
    "run-poc": _cmd_run_poc,
}


def _configure_logging() -> None:
    """Configure le logging selon le format de IMPLEMENTATION_NOTES section 1.9.

    Idempotent : si root est deja configure (ex: import depuis tests), on n'ajoute pas
    de handler en double.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point CLI. Retourne l'exit code."""
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = _COMMAND_DISPATCH.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2  # pragma: no cover - argparse leve SystemExit avant

    try:
        return handler(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - on veut tout capturer pour exit code propre.
        logger.error("Pipeline failed: %s", exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CompressionResult",
    "ReconstructionResult",
    "ValidationResultBundle",
    "compress",
    "decompress",
    "validate_pair",
    "main",
]
