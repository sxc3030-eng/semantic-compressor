"""Stress test the semantic-compressor at 1k / 10k / 100k rows and plot scaling.

Produit :
- output/benchmarks/scaling.json  : donnees brutes (timing + tailles + ratio + fidelity)
- output/benchmarks/scaling.png   : plot log-log time + linear ratio

Utilisation :
    python examples/run_benchmarks.py

Note: ce script ne modifie PAS data/original/users.csv (le canonical 10k).
Il genere ses propres datasets users_1k.csv / users_10k.csv / users_100k.csv.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator import (  # noqa: E402
    _extract_random_format_columns_from_recipe,
    compress,
    decompress,
    validate_pair,
)
from src.reconstructor import parse_recipe  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,  # WARNING pour reduire le bruit ; les mesures sont via print
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

# Reduire le bruit des modules internes (sinon le timing est pollue par les logs).
for noisy in ("src", "src.orchestrator", "src.profiler", "src.pattern_detector",
              "src.anchor_extractor", "src.reconstructor", "src.validator",
              "src.recipe_writer"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


SIZES = [1_000, 10_000, 100_000]
SEED = 42


@dataclass
class BenchmarkRun:
    """Une mesure unique : 1 taille x 1 mode."""

    n_rows: int
    mode: str  # "default" ou "aggressive"
    original_bytes: int
    recipe_bytes: int
    anchor_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    fidelity: float
    compress_seconds: float
    decompress_seconds: float
    validate_seconds: float
    peak_memory_mb: float


def _label_for(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def _generate_dataset(n_rows: int, output_csv: Path) -> None:
    """Genere un CSV via examples/generate_fake_db.py."""
    script = PROJECT_ROOT / "examples" / "generate_fake_db.py"
    cmd = [
        sys.executable,
        str(script),
        "--n-rows", str(n_rows),
        "--seed", str(SEED),
        "--output", str(output_csv),
    ]
    print(f"  generating {n_rows} rows -> {output_csv.name}...", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    [ERROR] generate_fake_db.py exit {res.returncode}", flush=True)
        print(res.stderr, flush=True)
        raise RuntimeError("dataset generation failed")
    elapsed = time.perf_counter() - t0
    print(f"    done in {elapsed:.1f}s ({output_csv.stat().st_size:,} B)", flush=True)


def _run_pipeline(
    input_csv: Path,
    output_dir: Path,
    reconstructed_csv: Path,
    aggressive_uuid: bool,
    n_rows: int,
) -> BenchmarkRun:
    """Execute compress + decompress + validate pour une combinaison et mesure tout."""
    mode = "aggressive" if aggressive_uuid else "default"
    print(f"  [{mode}] running pipeline...", flush=True)

    tracemalloc.start()

    # 1. COMPRESS
    t0 = time.perf_counter()
    cresult = compress(
        input_csv=input_csv,
        output_dir=output_dir,
        manual_anchor_columns=None,
        manual_random_format_columns=["password_hash"],  # toujours active (cf README)
        parquet_codec="zstd",
        parquet_compression_level=22,
        generate_html_profile=False,
        aggressive_uuid=aggressive_uuid,
    )
    compress_seconds = time.perf_counter() - t0

    # 2. DECOMPRESS
    t0 = time.perf_counter()
    rresult = decompress(
        recipe_path=cresult.recipe_path,
        output_csv=reconstructed_csv,
    )
    decompress_seconds = time.perf_counter() - t0

    # 3. VALIDATE
    recipe = parse_recipe(cresult.recipe_path)
    random_format_map = _extract_random_format_columns_from_recipe(recipe)

    t0 = time.perf_counter()
    vresult = validate_pair(
        original_csv=input_csv,
        reconstructed_csv=reconstructed_csv,
        anchor_columns=cresult.anchor_columns,
        random_format_columns=random_format_map or None,
    )
    validate_seconds = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    total = cresult.recipe_size_bytes + cresult.anchor_size_bytes
    ratio = cresult.original_size_bytes / max(total, 1)

    run = BenchmarkRun(
        n_rows=n_rows,
        mode=mode,
        original_bytes=cresult.original_size_bytes,
        recipe_bytes=cresult.recipe_size_bytes,
        anchor_bytes=cresult.anchor_size_bytes,
        total_compressed_bytes=total,
        compression_ratio=ratio,
        fidelity=vresult.report.overall_score,
        compress_seconds=compress_seconds,
        decompress_seconds=decompress_seconds,
        validate_seconds=validate_seconds,
        peak_memory_mb=peak_mb,
    )
    print(
        f"    -> ratio={ratio:.2f}:1, fidelity={vresult.report.overall_score:.1f}/100, "
        f"compress={compress_seconds:.2f}s, decompress={decompress_seconds:.2f}s, "
        f"validate={validate_seconds:.2f}s, peak_mem={peak_mb:.1f}MB",
        flush=True,
    )
    return run


def _generate_plot(runs: list[BenchmarkRun], output_png: Path) -> None:
    """Genere le plot scaling : 2 subplots (time, ratio)."""
    fig, (ax_time, ax_ratio) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1 : time vs n_rows (log-log) ---
    # Pour le timing on prend le mode aggressive comme reference (pipeline le plus
    # representatif documente dans le README).
    aggro = [r for r in runs if r.mode == "aggressive"]
    aggro.sort(key=lambda r: r.n_rows)
    xs = [r.n_rows for r in aggro]
    compress_ys = [r.compress_seconds for r in aggro]
    decompress_ys = [r.decompress_seconds for r in aggro]
    validate_ys = [r.validate_seconds for r in aggro]

    ax_time.loglog(xs, compress_ys, marker="o", label="compress", color="#1f77b4", linewidth=2)
    ax_time.loglog(xs, decompress_ys, marker="s", label="decompress", color="#2ca02c", linewidth=2)
    ax_time.loglog(xs, validate_ys, marker="^", label="validate", color="#d62728", linewidth=2)

    # Ligne de reference O(n) pour comparer.
    if xs:
        ref_y0 = min(compress_ys + decompress_ys + validate_ys) * 0.5
        ref_x0 = xs[0]
        ref_xs = xs
        ref_ys = [ref_y0 * (x / ref_x0) for x in ref_xs]
        ax_time.loglog(ref_xs, ref_ys, linestyle="--", color="gray", alpha=0.5, label="O(n) reference")

    ax_time.set_xlabel("n_rows (log)")
    ax_time.set_ylabel("seconds (log)")
    ax_time.set_title("Pipeline timing (--aggressive-uuid mode)")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend(loc="upper left")

    # --- Subplot 2 : compression ratio vs n_rows ---
    default = [r for r in runs if r.mode == "default"]
    default.sort(key=lambda r: r.n_rows)
    default_xs = [r.n_rows for r in default]
    default_ys = [r.compression_ratio for r in default]

    aggro_xs = [r.n_rows for r in aggro]
    aggro_ys = [r.compression_ratio for r in aggro]

    ax_ratio.plot(default_xs, default_ys, marker="o", label="default mode", color="#1f77b4", linewidth=2)
    ax_ratio.plot(aggro_xs, aggro_ys, marker="s", label="--aggressive-uuid mode", color="#ff7f0e", linewidth=2)

    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("n_rows (log)")
    ax_ratio.set_ylabel("compression ratio (X:1)")
    ax_ratio.set_title("Compression ratio")
    ax_ratio.grid(True, which="both", alpha=0.3)
    ax_ratio.legend(loc="lower right")

    # Caption global.
    fig.suptitle("Semantic Compressor - scaling behavior", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"Measured at n_rows in {[_label_for(n) for n in SIZES]} ; "
        f"codec=zstd ; level=22 ; password_hash forced RANDOM_FORMAT.",
        ha="center", fontsize=9, style="italic", color="#555",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  plot saved to {output_png} ({output_png.stat().st_size:,} B)", flush=True)


def main() -> int:
    bench_dir = PROJECT_ROOT / "output" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "original"
    data_dir.mkdir(parents=True, exist_ok=True)

    recon_dir = PROJECT_ROOT / "data" / "reconstructed"
    recon_dir.mkdir(parents=True, exist_ok=True)

    out_dir = PROJECT_ROOT / "output"

    runs: list[BenchmarkRun] = []

    for n_rows in SIZES:
        label = _label_for(n_rows)
        print(f"\n=== n_rows = {n_rows:,} ({label}) ===", flush=True)

        input_csv = data_dir / f"users_{label}.csv"
        if not input_csv.exists():
            _generate_dataset(n_rows, input_csv)
        else:
            print(f"  dataset already present: {input_csv} ({input_csv.stat().st_size:,} B)", flush=True)

        # Pour les benchmarks, on isole les sorties dans des sous-dossiers par taille
        # pour ne pas ecraser le canonical (output/recipes/users.md = 10k aggressive).
        bench_out = out_dir / "_benchmark_runs" / label
        bench_out.mkdir(parents=True, exist_ok=True)

        # On renomme symboliquement : copie le CSV avec un nom unique pour qu'il devienne
        # le `table_name` cohérent. Plus simple : on passe input_csv tel quel, le table_name
        # devient `users_1k` (= stem du fichier).
        reconstructed = recon_dir / f"users_{label}.csv"

        # Default mode
        run_def = _run_pipeline(
            input_csv=input_csv,
            output_dir=bench_out,
            reconstructed_csv=reconstructed,
            aggressive_uuid=False,
            n_rows=n_rows,
        )
        runs.append(run_def)

        # Aggressive mode
        run_aggro = _run_pipeline(
            input_csv=input_csv,
            output_dir=bench_out,
            reconstructed_csv=reconstructed,
            aggressive_uuid=True,
            n_rows=n_rows,
        )
        runs.append(run_aggro)

    # Persiste les donnees brutes (utile pour reproduire la table BENCHMARKS.md sans
    # relancer toute la batterie).
    json_path = bench_dir / "scaling.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in runs], f, indent=2)
    print(f"\nRaw data saved to {json_path}", flush=True)

    # Plot.
    png_path = bench_dir / "scaling.png"
    _generate_plot(runs, png_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
