"""Genere une empreinte visuelle de la BD compressee.

Idee : chaque pixel = 3 bytes (R, G, B) de la recette + des ancres concatenees.
La meme BD compressee = la meme empreinte. Une BD differente = empreinte
visiblement differente. C'est l'equivalent d'un "ADN visuel" du dataset.

On produit 3 images :
- fingerprint_dna.png       : "ADN" base-4 (A/C/G/T) en couleurs distinctes
- fingerprint_rgb.png       : densite max (1 pixel = 3 bytes, RGB)
- fingerprint_comparison.png: avant/apres (CSV original vs recipe+ancres)

Utilisation :
    python examples/render_fingerprint.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image  # via matplotlib deps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "fingerprints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _square_dims(n_pixels: int) -> tuple[int, int]:
    """Trouve (w, h) approximativement carre tel que w*h >= n_pixels."""
    side = int(math.ceil(math.sqrt(n_pixels)))
    return side, int(math.ceil(n_pixels / side))


def render_rgb_fingerprint(data: bytes, output_path: Path) -> tuple[int, int]:
    """Affiche les bytes 3-par-3 comme pixels RGB. 1 pixel = 3 bytes = 24 bits."""
    n_pixels = math.ceil(len(data) / 3)
    w, h = _square_dims(n_pixels)
    padded = data + b"\x00" * (w * h * 3 - len(data))
    arr = np.frombuffer(padded, dtype=np.uint8).reshape((h, w, 3))
    img = Image.fromarray(arr, mode="RGB")
    img.save(output_path)
    return w, h


def render_dna_fingerprint(data: bytes, output_path: Path) -> tuple[int, int]:
    """Encode les bytes 2 bits par 2 bits sur 4 couleurs (palette ADN A/C/G/T)."""
    # 4 nucleotides = 4 couleurs evocatrices (vert/bleu/rouge/jaune)
    palette = np.array(
        [
            [76, 175, 80],     # A (Adenine)   - vert
            [33, 150, 243],    # C (Cytosine)  - bleu
            [244, 67, 54],     # G (Guanine)   - rouge
            [255, 193, 7],     # T (Thymine)   - jaune ambre
        ],
        dtype=np.uint8,
    )
    # Chaque byte -> 4 nucleotides (4 * 2 bits)
    arr_bytes = np.frombuffer(data, dtype=np.uint8)
    # Extrait 4 paires de bits par byte (MSB d'abord)
    nucs = np.stack(
        [
            (arr_bytes >> 6) & 0b11,
            (arr_bytes >> 4) & 0b11,
            (arr_bytes >> 2) & 0b11,
            arr_bytes & 0b11,
        ],
        axis=1,
    ).flatten()
    n_nucs = nucs.size
    w, h = _square_dims(n_nucs)
    padded = np.concatenate([nucs, np.zeros(w * h - n_nucs, dtype=np.uint8)])
    arr = palette[padded.reshape(h, w)]
    # Agrandir pour bien voir les "nucleotides" : upscale x2 (sans interpolation)
    arr_big = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    img = Image.fromarray(arr_big, mode="RGB")
    img.save(output_path)
    return w, h


def render_side_by_side(
    original_data: bytes,
    compressed_data: bytes,
    output_path: Path,
    target_height: int = 500,
) -> None:
    """Cote-a-cote: original CSV vs compressed (recipe+anchors), a la meme echelle."""
    def to_rgb_image(data: bytes) -> Image.Image:
        n_pixels = math.ceil(len(data) / 3)
        w, h = _square_dims(n_pixels)
        padded = data + b"\x00" * (w * h * 3 - len(data))
        arr = np.frombuffer(padded, dtype=np.uint8).reshape((h, w, 3))
        return Image.fromarray(arr, mode="RGB")

    img_o = to_rgb_image(original_data)
    img_c = to_rgb_image(compressed_data)

    # Echelle : on garde la PROPORTION des tailles (ratio entre les images).
    # On scale a target_height en preservant aspect ratio individuel.
    scale_o = target_height / img_o.height
    scale_c = (target_height / img_o.height) * (img_c.height / img_c.height)  # noqa
    # En fait : on garde la PROPORTION reelle entre les 2 images, pas un meme height.
    # L'image compressed doit etre visiblement plus petite si le ratio est important.
    sqrt_ratio = math.sqrt(len(compressed_data) / len(original_data))
    new_o_w = int(target_height * img_o.width / img_o.height)
    new_o_h = target_height
    new_c_w = max(1, int(new_o_w * sqrt_ratio))
    new_c_h = max(1, int(new_o_h * sqrt_ratio))

    img_o_resized = img_o.resize((new_o_w, new_o_h), Image.NEAREST)
    img_c_resized = img_c.resize((new_c_w, new_c_h), Image.NEAREST)

    # Compose horizontalement avec un padding noir
    gap = 40
    total_w = new_o_w + gap + new_c_w + 80  # 80 = marges
    total_h = target_height + 100  # 100 = marge bas pour labels (texte ajoute apres)
    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 28))
    canvas.paste(img_o_resized, (40, 40))
    canvas.paste(img_c_resized, (40 + new_o_w + gap, 40 + (new_o_h - new_c_h) // 2))

    # Annotations (texte basique via PIL)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text(
        (40, 40 + new_o_h + 10),
        f"ORIGINAL  {len(original_data):,} B",
        fill=(220, 220, 220),
        font=font,
    )
    draw.text(
        (40 + new_o_w + gap, 40 + new_o_h + 10),
        f"COMPRESSED  {len(compressed_data):,} B",
        fill=(220, 220, 220),
        font=font,
    )
    ratio = len(original_data) / len(compressed_data)
    draw.text(
        (total_w - 220, 40),
        f"ratio {ratio:.2f}:1",
        fill=(76, 175, 80),
        font=font,
    )

    canvas.save(output_path)


def main() -> int:
    original_csv = PROJECT_ROOT / "data" / "original" / "users.csv"
    recipe_path = PROJECT_ROOT / "output" / "recipes" / "users.md"
    anchor_path = PROJECT_ROOT / "output" / "anchors" / "users_anchors.parquet"

    if not (original_csv.exists() and recipe_path.exists() and anchor_path.exists()):
        print(
            "Inputs manquants. Lance d'abord :",
            file=sys.stderr,
        )
        print("  python examples/run_poc.py", file=sys.stderr)
        return 1

    original_bytes = original_csv.read_bytes()
    recipe_bytes = recipe_path.read_bytes()
    anchor_bytes = anchor_path.read_bytes()
    compressed_bytes = recipe_bytes + anchor_bytes  # union "compressed payload"

    # 1. RGB fingerprint (densite max)
    rgb_path = OUTPUT_DIR / "fingerprint_rgb.png"
    rgb_w, rgb_h = render_rgb_fingerprint(compressed_bytes, rgb_path)
    print(f"RGB fingerprint   : {rgb_path} ({rgb_w}x{rgb_h})")

    # 2. DNA fingerprint (base-4 colore)
    dna_path = OUTPUT_DIR / "fingerprint_dna.png"
    dna_w, dna_h = render_dna_fingerprint(compressed_bytes, dna_path)
    print(f"DNA fingerprint   : {dna_path} ({dna_w}x{dna_h} pre-upscale)")

    # 3. Comparison side-by-side
    cmp_path = OUTPUT_DIR / "fingerprint_comparison.png"
    render_side_by_side(original_bytes, compressed_bytes, cmp_path)
    print(f"Side-by-side      : {cmp_path}")

    # Stats
    print()
    print(f"Original         : {len(original_bytes):>10,} B")
    print(f"Recipe           : {len(recipe_bytes):>10,} B")
    print(f"Anchors (parquet): {len(anchor_bytes):>10,} B")
    print(f"Compressed total : {len(compressed_bytes):>10,} B")
    print(f"Ratio            : {len(original_bytes)/len(compressed_bytes):.2f}:1")
    print(f"PNG (rgb) size   : {rgb_path.stat().st_size:>10,} B  ({(rgb_path.stat().st_size/len(compressed_bytes))*100:.1f}% du compressed)")
    print(f"PNG (dna) size   : {dna_path.stat().st_size:>10,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
