"""Demo end-to-end du semantic-compressor.

Lance le pipeline complet (generate fake DB optionnel + compress + decompress + validate)
et affiche un rapport final Rich.

Utilisation :
    # Pipeline complet (assume que data/original/users.csv existe)
    python examples/run_poc.py

    # Avec regeneration prealable de la fausse BD
    python examples/run_poc.py --regenerate-db --n-rows 10000 --seed 42

    # Personnaliser les chemins
    python examples/run_poc.py --input data/original/users.csv --output-dir output/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ajoute le projet au path pour permettre les imports `src.*` quand le script
# est lance depuis n'importe quel CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator import (  # noqa: E402  (sys.path injection avant import)
    _extract_random_format_columns_from_recipe,
    compress,
    decompress,
    validate_pair,
)
from src.reconstructor import parse_recipe  # noqa: E402
from src.validator import print_report  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def regenerate_fake_db(n_rows: int, seed: int, output_csv: Path) -> None:
    """Lance examples/generate_fake_db.py via subprocess pour regenerer la BD de test."""
    script = PROJECT_ROOT / "examples" / "generate_fake_db.py"
    cmd = [
        sys.executable,
        str(script),
        "--n-rows", str(n_rows),
        "--seed", str(seed),
        "--output", str(output_csv),
    ]
    logger.info("Regenerating fake DB: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]generate_fake_db.py a echoue (exit {result.returncode}):[/red]")
        console.print(result.stderr)
        raise RuntimeError("Regeneration de la fausse BD echouee")
    logger.info("Fake DB regenerated -> %s", output_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo end-to-end du semantic-compressor POC.")
    p.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "original" / "users.csv",
        help="Chemin du CSV original (default: data/original/users.csv).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output",
        help="Dossier de sortie pour la recette et les ancres (default: output/).",
    )
    p.add_argument(
        "--reconstructed-csv", type=Path,
        default=PROJECT_ROOT / "data" / "reconstructed" / "users.csv",
        help="Chemin du CSV reconstruit (default: data/reconstructed/users.csv).",
    )
    p.add_argument(
        "--regenerate-db", action="store_true",
        help="Regenere la fausse BD via examples/generate_fake_db.py avant compression.",
    )
    p.add_argument(
        "--n-rows", type=int, default=10000,
        help="Nb de lignes pour la regeneration de BD (default: 10000).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed pour la regeneration de BD (default: 42).",
    )
    p.add_argument(
        "--codec", default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli"],
        help="Codec parquet pour les ancres (default: zstd).",
    )
    p.add_argument(
        "--level", type=int, default=22,
        help="Niveau de compression (zstd: 1-22, default: 22).",
    )
    p.add_argument(
        "--random-format", type=str, default=None,
        help=(
            "Liste de colonnes a forcer en RANDOM_FORMAT (valeur non preservee, "
            "regeneree depuis un format spec). Typique : password_hash. "
            "Format : virgules (ex: --random-format password_hash,other_col)."
        ),
    )
    p.add_argument(
        "--profile-report", action="store_true",
        help=(
            "Genere un rapport HTML d'exploration ydata-profiling "
            "dans output/profiling_reports/<table>.html en plus de la compression."
        ),
    )
    p.add_argument(
        "--aggressive-uuid", action="store_true",
        help=(
            "Force les colonnes UUID v4 dont le nom ressemble a un id (id, uuid, "
            "*_id, *_uuid, ...) a etre RANDOM_FORMAT. Gain typique : ~3-5x sur "
            "ces colonnes au prix de la preservation de la valeur exacte. "
            "Defaut : preserver les ids comme ANCHOR_DIRECT."
        ),
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Logging niveau DEBUG.",
    )
    return p.parse_args()


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    console.print(Panel.fit(
        "[bold cyan]SEMANTIC COMPRESSOR - END-TO-END POC[/bold cyan]\n"
        "compress -> decompress -> validate",
        border_style="cyan",
    ))

    # 0. Regeneration optionnelle de la BD
    if args.regenerate_db or not args.input.exists():
        if not args.input.exists():
            console.print(f"[yellow]Input {args.input} introuvable, regeneration auto.[/yellow]")
        regenerate_fake_db(args.n_rows, args.seed, args.input)

    if not args.input.exists():
        console.print(f"[red]Input introuvable apres regeneration: {args.input}[/red]")
        return 2

    # 1. COMPRESS
    console.print("\n[bold]Step 1 / 3 : compress[/bold]")
    random_format = _parse_csv_list(args.random_format)
    cresult = compress(
        input_csv=args.input,
        output_dir=args.output_dir,
        manual_anchor_columns=None,
        manual_random_format_columns=random_format,
        parquet_codec=args.codec,
        parquet_compression_level=args.level,
        generate_html_profile=args.profile_report,
        aggressive_uuid=args.aggressive_uuid,
    )
    console.print(
        f"  -> recipe={cresult.recipe_path.name} ({cresult.recipe_size_bytes:,} B)"
        f", anchors={cresult.anchor_path.name} ({cresult.anchor_size_bytes:,} B)"
    )
    if args.profile_report:
        if cresult.html_profile_path is not None:
            console.print(
                f"  -> HTML profile: [cyan]{cresult.html_profile_path}[/cyan]"
            )
        else:
            console.print(
                "  -> [yellow]HTML profile generation failed[/yellow] "
                "(voir logs ; le pipeline a continue)"
            )

    # 2. DECOMPRESS
    console.print("\n[bold]Step 2 / 3 : decompress[/bold]")
    rresult = decompress(
        recipe_path=cresult.recipe_path,
        output_csv=args.reconstructed_csv,
    )
    console.print(f"  -> reconstructed {rresult.output_path}")

    # 3. VALIDATE
    console.print("\n[bold]Step 3 / 3 : validate[/bold]")
    # Pour les colonnes RANDOM_FORMAT, on extrait leur regex depuis la recette
    # pour le validator.
    recipe = parse_recipe(cresult.recipe_path)
    random_format_map = _extract_random_format_columns_from_recipe(recipe)
    vresult = validate_pair(
        original_csv=args.input,
        reconstructed_csv=args.reconstructed_csv,
        anchor_columns=cresult.anchor_columns,
        random_format_columns=random_format_map or None,
    )
    console.print(
        f"  -> {vresult.report.passed_count} passed, {vresult.report.failed_count} failed"
    )

    # Rapport final detaille (les tests qui ont passe + ceux qui ont echoue)
    print_report(vresult.report)

    # Mega-resume
    total_compressed = cresult.recipe_size_bytes + cresult.anchor_size_bytes
    ratio = cresult.original_size_bytes / max(total_compressed, 1)
    total_time = cresult.elapsed_seconds + rresult.elapsed_seconds + vresult.elapsed_seconds

    ratio_color = "green" if ratio >= 5 else "yellow" if ratio >= 3 else "red"
    score_color = (
        "green" if vresult.report.overall_score >= 95
        else "yellow" if vresult.report.overall_score >= 80
        else "red"
    )

    summary = (
        f"[bold]Compression ratio:[/bold] [{ratio_color}]{ratio:.2f}:1[/{ratio_color}]   "
        f"[bold]Fidelity:[/bold] [{score_color}]{vresult.report.overall_score:.1f}/100[/{score_color}]   "
        f"[bold]Wall time:[/bold] {total_time:.1f}s"
    )
    console.print(Panel.fit(summary, border_style="cyan", title="POC RESULT"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
