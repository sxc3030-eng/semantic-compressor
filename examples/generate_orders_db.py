"""Generate a synthetic e-commerce orders database for the semantic-compressor POC.

Produit un CSV de N commandes synthetiques avec :
- Des ancres irreductibles (order_id UUID, customer_id UUID) → uniques par ligne.
- Des colonnes regenerables avec patterns detectables :
  * categorical biaise (product_sku Pareto, currency, status)
  * dependance fonctionnelle (category derive de product_sku via lookup)
  * distributions (quantity log-normal, unit_price_cents normal par category)
  * datetime exponentiel (ordered_at) + dependance temporelle (shipped_at)
  * conditional distribution (shipped_at conditionne par status)

Objectif : second banc d'essai pour valider la generalite du POC. Les patterns
sont volontairement differents de `users.csv` (Pareto + dep fonctionnelle +
conditional sur datetime + nulls structures) pour stresser le pattern detector.

Reproductibilite : meme `--seed` + meme `--n-rows` → meme CSV bit a bit.
"""

from __future__ import annotations

import argparse
import logging
import string
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from faker import Faker
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

DEFAULT_SEED: Final[int] = 42
DEFAULT_N_ROWS: Final[int] = 10_000
DEFAULT_OUTPUT: Final[Path] = (
    Path(__file__).resolve().parent.parent / "data" / "original" / "orders.csv"
)

# Bornes temporelles fixees pour la reproductibilite.
DATA_NOW: Final[datetime] = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
DATA_EPOCH: Final[datetime] = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# Scale exponentiel (en jours) pour `ordered_at` : plus c'est petit, plus la
# distribution est concentree pres de DATA_NOW (commerce → activite recente plus dense).
ORDERED_AT_SCALE_DAYS: Final[float] = 180.0

# Catalogue produits : ~500 SKUs uniques, distribution Pareto (top 20 = 60%).
N_PRODUCT_SKUS: Final[int] = 500
TOP_SKUS_COUNT: Final[int] = 20
TOP_SKUS_MASS: Final[float] = 0.60  # 60% des commandes sur le top 20.

# 10 categories produit, mappees de facon deterministe a chaque SKU.
CATEGORIES: Final[list[str]] = [
    "electronics", "clothing", "home", "books", "toys",
    "sports", "beauty", "grocery", "automotive", "garden",
]

# Distribution des devises (somme = 1.0). Refl. une boutique majoritairement US.
CURRENCIES: Final[list[str]] = ["USD", "EUR", "GBP", "JPY", "CAD"]
CURRENCY_WEIGHTS: Final[list[float]] = [0.60, 0.20, 0.10, 0.05, 0.05]

# Workflow status. Distribution dominante "delivered" (commerce stable).
STATUSES: Final[list[str]] = [
    "delivered", "shipped", "processing", "returned", "cancelled", "pending",
]
STATUS_WEIGHTS: Final[list[float]] = [0.80, 0.10, 0.05, 0.03, 0.01, 0.01]

# Sous-ensemble de statuts pour lesquels `shipped_at` est NON-NULL.
STATUSES_WITH_SHIPPED_AT: Final[set[str]] = {"shipped", "delivered", "returned"}
# Delai d'expedition : exponentiel en jours.
SHIPPED_DELAY_SCALE_DAYS: Final[float] = 3.0

# Quantity : log-normal clipe.
QTY_LOG_MEAN: Final[float] = 1.5  # log-mean (parametre de la log-normale "underlying normal")
# Lecture stricte de la spec : "log-normal (mean=1.5, sigma=0.8)".
# On interprete (mean, sigma) comme parametres de la N(mean, sigma) sous-jacente
# (convention numpy.lognormal). E[X] = exp(mean + sigma^2/2) ~ 6.16.
QTY_LOG_SIGMA: Final[float] = 0.8
QTY_MIN: Final[int] = 1
QTY_MAX: Final[int] = 50

# Unit price (cents) : normal centre, parametres specifiques par category.
PRICE_GLOBAL_MEAN_CENTS: Final[float] = 2000.0
PRICE_GLOBAL_STD_CENTS: Final[float] = 1500.0
PRICE_MIN_CENTS: Final[int] = 99
PRICE_MAX_CENTS: Final[int] = 99_999
# Multiplicateurs par category pour creer un signal CONDITIONAL detectable.
# Garde mean global ~2000 cents mais induit une dispersion croisee.
PRICE_CATEGORY_MULTIPLIER: Final[dict[str, float]] = {
    "electronics": 3.0,   # cher
    "automotive": 2.5,
    "home": 1.6,
    "sports": 1.4,
    "beauty": 1.0,
    "clothing": 1.0,
    "garden": 0.9,
    "toys": 0.8,
    "books": 0.6,         # bon marche
    "grocery": 0.4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Initialise le logging au niveau INFO (pas de duplication si rappel)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )


def _generate_uuids(rng: np.random.Generator, n: int) -> list[str]:
    """Genere n UUIDs v4 deterministes via le RNG fourni.

    On contourne `uuid.uuid4()` (qui utilise os.urandom, non-seedable) en
    construisant l'UUID a partir de 16 octets tires de `rng`.
    """
    raw = rng.bytes(n * 16)
    uuids: list[str] = []
    for i in range(n):
        chunk = bytearray(raw[i * 16 : (i + 1) * 16])
        # Force version 4 et variant RFC 4122 (cf. RFC 4122 §4.4).
        chunk[6] = (chunk[6] & 0x0F) | 0x40
        chunk[8] = (chunk[8] & 0x3F) | 0x80
        uuids.append(str(uuid.UUID(bytes=bytes(chunk))))
    return uuids


def _build_sku_catalog(rng: np.random.Generator) -> tuple[list[str], dict[str, str]]:
    """Construit le catalogue de N_PRODUCT_SKUS SKUs et la lookup table SKU -> category.

    Format SKU : `PRD-{6-digits}-{2-letters}`. Les ID 6-digits sont tires sans
    remise dans [0, 999999] pour garantir l'unicite (en theorie). Les 2 lettres
    en suffixe sont aleatoires uniformes.
    """
    # Numeros uniques : tirage sans remise dans [0, 999999].
    digits = rng.choice(1_000_000, size=N_PRODUCT_SKUS, replace=False)
    # 2 lettres uppercase tirees independamment (peuvent coller, peu importe :
    # l'unicite est garantie par les digits).
    letters_idx = rng.integers(0, 26, size=N_PRODUCT_SKUS * 2)
    letters = np.array(list(string.ascii_uppercase))[letters_idx].reshape(-1, 2)

    skus: list[str] = []
    sku_to_category: dict[str, str] = {}
    # Assignation de category : round-robin pour avoir une distribution
    # equilibree dans le catalogue (la distribution observee dans les ordres
    # sera elle, biaisee, parce que la frequence des SKUs est Pareto).
    n_cat = len(CATEGORIES)
    for i in range(N_PRODUCT_SKUS):
        sku = f"PRD-{int(digits[i]):06d}-{letters[i, 0]}{letters[i, 1]}"
        skus.append(sku)
        sku_to_category[sku] = CATEGORIES[i % n_cat]
    return skus, sku_to_category


def _pareto_weights(rng: np.random.Generator) -> np.ndarray:
    """Construit un vecteur de poids pour les SKUs : top 20 = TOP_SKUS_MASS, reste uniforme.

    On veut une distribution Pareto-like : 20 SKUs "stars" se partagent 60% des
    commandes, les 480 autres se partagent 40% uniformement.

    Le top 20 est aussi pondere : decroissance lineaire dans la tete pour
    accentuer la queue lourde sur les TOUT premiers.
    """
    n = N_PRODUCT_SKUS
    weights = np.empty(n, dtype=np.float64)
    # Top : poids proportionnels a (TOP_SKUS_COUNT - i), normalises a TOP_SKUS_MASS.
    head_raw = np.arange(TOP_SKUS_COUNT, 0, -1, dtype=np.float64)
    head = head_raw / head_raw.sum() * TOP_SKUS_MASS
    weights[:TOP_SKUS_COUNT] = head
    # Tail uniforme sur le reste de la masse (1 - TOP_SKUS_MASS).
    tail_mass = 1.0 - TOP_SKUS_MASS
    weights[TOP_SKUS_COUNT:] = tail_mass / (n - TOP_SKUS_COUNT)
    # Defense en profondeur contre les arrondis float.
    weights = weights / weights.sum()
    # Optionnel : on melange l'ordre des SKUs pour que les "stars" ne soient pas
    # forcement les premiers du catalogue (plus realiste). On utilise le rng pour
    # rester deterministe.
    perm = rng.permutation(n)
    permuted = np.empty_like(weights)
    permuted[perm] = weights
    return permuted


def _generate_product_skus(
    rng: np.random.Generator, skus: list[str], weights: np.ndarray, n: int
) -> np.ndarray:
    """Tirage multinomial pondere des SKUs."""
    return rng.choice(skus, size=n, p=weights)


def _generate_categories(
    product_skus: np.ndarray, sku_to_category: dict[str, str]
) -> np.ndarray:
    """Lookup deterministe SKU -> category. Pas de tirage : c'est une vraie dep fonctionnelle."""
    return np.array([sku_to_category[s] for s in product_skus], dtype=object)


def _generate_quantities(rng: np.random.Generator, n: int) -> np.ndarray:
    """Quantity log-normal(mean=1.5, sigma=0.8), clipe [QTY_MIN, QTY_MAX], renvoye en int."""
    raw = rng.lognormal(mean=QTY_LOG_MEAN, sigma=QTY_LOG_SIGMA, size=n)
    clipped = np.clip(raw, QTY_MIN, QTY_MAX)
    return np.round(clipped).astype(np.int64)


def _generate_unit_prices(
    rng: np.random.Generator, categories: np.ndarray
) -> np.ndarray:
    """Unit price (cents) normal par category, clipe [PRICE_MIN_CENTS, PRICE_MAX_CENTS].

    Pour chaque ligne :
        mean_cat = PRICE_GLOBAL_MEAN_CENTS * multiplier(cat)
        std_cat  = PRICE_GLOBAL_STD_CENTS  * multiplier(cat) * 0.5
        price = clip(Normal(mean_cat, std_cat), 99, 99999)

    Le std est multiplie aussi pour preserver un coefficient de variation
    raisonnable par category (sinon les categories cheres deviendraient trop
    uniformes en valeur absolue).
    """
    n = len(categories)
    mults = np.array(
        [PRICE_CATEGORY_MULTIPLIER[c] for c in categories], dtype=np.float64
    )
    means = PRICE_GLOBAL_MEAN_CENTS * mults
    stds = PRICE_GLOBAL_STD_CENTS * mults * 0.5  # CV constant ~37.5%
    raw = rng.normal(loc=means, scale=stds, size=n)
    clipped = np.clip(raw, PRICE_MIN_CENTS, PRICE_MAX_CENTS)
    return np.round(clipped).astype(np.int64)


def _generate_currencies(rng: np.random.Generator, n: int) -> np.ndarray:
    """Tirage multinomial des devises selon CURRENCY_WEIGHTS."""
    return rng.choice(CURRENCIES, size=n, p=CURRENCY_WEIGHTS)


def _generate_statuses(rng: np.random.Generator, n: int) -> np.ndarray:
    """Tirage multinomial des statuts selon STATUS_WEIGHTS."""
    return rng.choice(STATUSES, size=n, p=STATUS_WEIGHTS)


def _generate_ordered_at(rng: np.random.Generator, n: int) -> list[datetime]:
    """Genere n timestamps avec distribution exponentielle (densite recente).

    On tire des "jours dans le passe" en exponentielle, puis on les retranche
    de DATA_NOW. Clipe a DATA_EPOCH au plus tot.
    """
    days_back = rng.exponential(scale=ORDERED_AT_SCALE_DAYS, size=n)
    max_days = (DATA_NOW - DATA_EPOCH).total_seconds() / 86400.0
    days_back = np.clip(days_back, 0.0, max_days)
    return [DATA_NOW - timedelta(days=float(d)) for d in days_back]


def _generate_shipped_at(
    rng: np.random.Generator,
    statuses: np.ndarray,
    ordered_at: list[datetime],
) -> list[datetime | None]:
    """`shipped_at` = ordered_at + Exponential(SHIPPED_DELAY_SCALE_DAYS) si status ∈
    {shipped, delivered, returned}. Sinon None (clean NULL dans le CSV).

    Contrainte : shipped_at <= DATA_NOW (sinon on clipe).
    """
    n = len(statuses)
    delays = rng.exponential(scale=SHIPPED_DELAY_SCALE_DAYS, size=n)
    result: list[datetime | None] = [None] * n
    for i in range(n):
        if statuses[i] not in STATUSES_WITH_SHIPPED_AT:
            continue
        candidate = ordered_at[i] + timedelta(days=float(delays[i]))
        if candidate > DATA_NOW:
            candidate = DATA_NOW
        result[i] = candidate
    return result


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def generate_fake_orders(n_rows: int, seed: int) -> pd.DataFrame:
    """Genere un DataFrame de n_rows commandes synthetiques reproductible.

    Args:
        n_rows: Nombre de commandes a generer.
        seed: Graine pour le PRNG (controle l'ensemble de la generation).

    Returns:
        DataFrame avec les colonnes :
        order_id, customer_id, product_sku, category, quantity, unit_price_cents,
        currency, status, ordered_at, shipped_at.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows doit etre > 0, recu {n_rows}")

    logger.info("Generating %d synthetic orders (seed=%d)", n_rows, seed)
    rng = np.random.default_rng(seed)
    # Faker n'est pas utilise pour des champs realistes ici (UUID + SKU + enum
    # tout deterministe), mais on l'instancie pour rester coherent avec
    # generate_fake_db.py et garder une porte ouverte a des extensions futures.
    faker = Faker("en_US")
    Faker.seed(seed)
    _ = faker  # pragma: explicit no-op marker

    # Ancres irreductibles -----------------------------------------------------
    logger.info("  → generating anchors (order_id, customer_id)")
    order_ids = _generate_uuids(rng, n_rows)
    customer_ids = _generate_uuids(rng, n_rows)

    # Catalogue + Pareto weights ----------------------------------------------
    logger.info(
        "  → building product catalog (%d SKUs, top %d = %.0f%% mass)",
        N_PRODUCT_SKUS, TOP_SKUS_COUNT, TOP_SKUS_MASS * 100,
    )
    skus, sku_to_category = _build_sku_catalog(rng)
    weights = _pareto_weights(rng)

    # Colonnes regenerables ----------------------------------------------------
    logger.info("  → generating product_sku (Pareto) + category (functional dep)")
    product_skus = _generate_product_skus(rng, skus, weights, n_rows)
    categories = _generate_categories(product_skus, sku_to_category)

    logger.info("  → generating quantity (log-normal) + unit_price_cents (conditional)")
    quantities = _generate_quantities(rng, n_rows)
    unit_prices = _generate_unit_prices(rng, categories)

    logger.info("  → generating currency + status (categorical_freq)")
    currencies = _generate_currencies(rng, n_rows)
    statuses = _generate_statuses(rng, n_rows)

    logger.info("  → generating ordered_at (exponential) + shipped_at (conditional)")
    ordered_at = _generate_ordered_at(rng, n_rows)
    shipped_at = _generate_shipped_at(rng, statuses, ordered_at)

    # Assemblage ---------------------------------------------------------------
    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "product_sku": product_skus,
            "category": categories,
            "quantity": quantities,
            "unit_price_cents": unit_prices,
            "currency": currencies,
            "status": statuses,
            "ordered_at": ordered_at,
            "shipped_at": shipped_at,
        }
    )

    logger.info("Generated DataFrame shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic e-commerce orders CSV for the semantic-compressor POC.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"PRNG seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=DEFAULT_N_ROWS,
        help=f"Number of orders to generate (default: {DEFAULT_N_ROWS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def _print_summary(df: pd.DataFrame, output: Path, console: Console) -> None:
    """Affiche un resume du dataset genere via Rich."""
    size_bytes = output.stat().st_size
    size_kb = size_bytes / 1024.0

    n_unique_skus = df["product_sku"].nunique()
    n_unique_customers = df["customer_id"].nunique()
    sku_freq = df["product_sku"].value_counts(normalize=True)
    top20_mass = float(sku_freq.head(TOP_SKUS_COUNT).sum())
    shipped_pct = float(df["shipped_at"].notna().mean())
    cat_counts = df["category"].value_counts(normalize=True).sort_values(ascending=False)
    cur_counts = df["currency"].value_counts(normalize=True).sort_values(ascending=False)
    status_counts = df["status"].value_counts(normalize=True).sort_values(ascending=False)

    summary = Table(title="Generated orders dataset summary", show_header=True, header_style="bold")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Rows", f"{len(df):,}")
    summary.add_row("Output", str(output))
    summary.add_row("Size", f"{size_kb:,.1f} KB ({size_bytes:,} bytes)")
    summary.add_row("Distinct SKUs", f"{n_unique_skus:,}")
    summary.add_row("Distinct customers", f"{n_unique_customers:,}")
    summary.add_row(f"Top {TOP_SKUS_COUNT} SKUs mass", f"{top20_mass * 100:.2f}%")
    summary.add_row("Rows with shipped_at", f"{shipped_pct * 100:.2f}%")
    summary.add_row("Mean quantity", f"{df['quantity'].mean():.2f}")
    summary.add_row("Mean unit_price_cents", f"{df['unit_price_cents'].mean():.0f}")
    console.print(summary)

    cat_table = Table(title="Category distribution", show_header=True, header_style="bold")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Share", justify="right", style="green")
    cat_table.add_column("Count", justify="right")
    for cat, share in cat_counts.items():
        count = int((df["category"] == cat).sum())
        cat_table.add_row(str(cat), f"{share * 100:.2f}%", f"{count:,}")
    console.print(cat_table)

    cur_table = Table(title="Currency / status distribution", show_header=True, header_style="bold")
    cur_table.add_column("Currency", style="cyan")
    cur_table.add_column("Share", justify="right", style="green")
    for cur, share in cur_counts.items():
        cur_table.add_row(str(cur), f"{share * 100:.2f}%")
    console.print(cur_table)

    status_table = Table(title="Status distribution", show_header=True, header_style="bold")
    status_table.add_column("Status", style="cyan")
    status_table.add_column("Share", justify="right", style="green")
    for st, share in status_counts.items():
        status_table.add_row(str(st), f"{share * 100:.2f}%")
    console.print(status_table)


def main(argv: list[str] | None = None) -> int:
    """Entry point CLI. Retourne exit code."""
    _setup_logging()
    args = _parse_args(argv)

    console = Console()
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = generate_fake_orders(n_rows=args.n_rows, seed=args.seed)
    except Exception:
        logger.exception("Generation failed")
        return 1

    logger.info("Writing CSV to %s", output)
    # `date_format` ISO 8601 pour une serialisation stable et reparsable.
    # Les valeurs None de shipped_at sont serialisees en cellule vide par pandas.
    df.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")

    _print_summary(df, output, console)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
