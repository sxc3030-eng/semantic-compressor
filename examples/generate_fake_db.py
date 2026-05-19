"""Generate a synthetic users database for the semantic-compressor POC.

Produit un CSV de N utilisateurs synthetiques avec :
- Des ancres irreductibles (id UUID, email, password_hash) → uniques par ligne.
- Des colonnes regenerables avec patterns detectables (distributions et
  correlations conditionnelles).

L'objectif est d'avoir un banc d'essai realiste : assez de patterns pour que la
compression semantique ait du sens, et assez de "bruit" irreductible pour que
les ancres soient indispensables.

Reproductibilite : meme `--seed` + meme `--n-rows` → meme CSV bit a bit.
"""

from __future__ import annotations

import argparse
import logging
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
DEFAULT_OUTPUT: Final[Path] = Path(r"D:\semantic-compressor\data\original\users.csv")

# Bornes temporelles fixees pour la reproductibilite (la "vue actuelle" du dataset
# est figee a la date de la spec, sinon "now" rendrait la sortie non-deterministe).
DATA_NOW: Final[datetime] = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
DATA_EPOCH: Final[datetime] = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# Scale exponentiel (en jours) pour `created_at` : plus c'est petit, plus la
# distribution est concentree pres de DATA_NOW.
CREATED_AT_SCALE_DAYS: Final[float] = 365.0

# Distribution biaisee par pays (somme = 1.0).
COUNTRIES: Final[list[str]] = ["US", "FR", "DE", "UK", "CA", "ES", "IT", "JP", "BR", "AU"]
COUNTRY_WEIGHTS: Final[list[float]] = [0.50, 0.20, 0.10, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]

# Source d'inscription : enum correle au pays.
# Pondrations (web, mobile, referral) par groupe de pays.
SIGNUP_SOURCES: Final[list[str]] = ["web", "mobile", "referral"]
SIGNUP_WEIGHTS_BY_GROUP: Final[dict[str, list[float]]] = {
    "us_uk": [0.60, 0.30, 0.10],          # US, UK
    "eu_latin": [0.40, 0.50, 0.10],       # FR, DE, ES, IT
    "mobile_first": [0.30, 0.60, 0.10],   # BR, JP, AU
    "ca": [0.50, 0.40, 0.10],             # CA
}
SIGNUP_GROUP_BY_COUNTRY: Final[dict[str, str]] = {
    "US": "us_uk",
    "UK": "us_uk",
    "FR": "eu_latin",
    "DE": "eu_latin",
    "ES": "eu_latin",
    "IT": "eu_latin",
    "BR": "mobile_first",
    "JP": "mobile_first",
    "AU": "mobile_first",
    "CA": "ca",
}

# Probabilites pour `premium` : base + boosts. La spec impose 25% max par bucket
# (age_bucket, country), donc on capote la probabilite a 0.25.
PREMIUM_BASE_PROB: Final[float] = 0.07
PREMIUM_BOOST_AGE_GE_30: Final[float] = 0.05  # +5pp
PREMIUM_BOOST_COUNTRY_US: Final[float] = 0.03  # +3pp
PREMIUM_MAX_PROB: Final[float] = 0.25

# Distributions pour `last_login` (en jours depuis DATA_NOW).
LAST_LOGIN_SCALE_PREMIUM: Final[float] = 7.0
LAST_LOGIN_SCALE_NON_PREMIUM: Final[float] = 60.0

# Age : normal centre, clipe.
AGE_MEAN: Final[float] = 34.0
AGE_STD: Final[float] = 12.0
AGE_MIN: Final[int] = 18
AGE_MAX: Final[int] = 99


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


def premium_probability(age: int, country: str) -> float:
    """Renvoie la probabilite que l'utilisateur soit premium.

    Combine la base 7% avec :
      - +5pp si age >= 30
      - +3pp si country == "US"
    Plafonnee a 25% (cap par bucket (age_bucket, country)).
    """
    prob = PREMIUM_BASE_PROB
    if age >= 30:
        prob += PREMIUM_BOOST_AGE_GE_30
    if country == "US":
        prob += PREMIUM_BOOST_COUNTRY_US
    return min(prob, PREMIUM_MAX_PROB)


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


def _generate_unique_emails(faker: Faker, n: int) -> list[str]:
    """Genere n emails uniques au format `{name}{n}@{domain}.{tld}`.

    Faker.unique garantit l'unicite ; on suffixe l'index pour eviter d'epuiser
    le pool pour des grands n.
    """
    emails: list[str] = []
    seen: set[str] = set()
    for i in range(n):
        # Boucle de garde : tres rare collision, on retire au besoin.
        while True:
            base = faker.email()
            local, _, domain = base.partition("@")
            candidate = f"{local}{i}@{domain}"
            if candidate not in seen:
                seen.add(candidate)
                emails.append(candidate)
                break
    return emails


def _generate_password_hashes(rng: np.random.Generator, n: int) -> list[str]:
    """Genere n password hashes uniques au format `$bcrypt$...$<64 hex>$`.

    Le format mime un bcrypt pour realisme, mais les caracteristiques exactes
    n'ont pas d'importance : c'est une ancre opaque, pure entropie.
    """
    # 32 octets → 64 caracteres hex. Tire en bloc puis decoupe pour la vitesse.
    raw = rng.bytes(n * 32)
    hashes: list[str] = []
    for i in range(n):
        hex_part = raw[i * 32 : (i + 1) * 32].hex()
        hashes.append(f"$bcrypt$2b$12${hex_part}$")
    return hashes


def _generate_created_at(rng: np.random.Generator, n: int) -> list[datetime]:
    """Genere n timestamps avec distribution exponentielle (densite recente).

    On tire des "jours dans le passe" en exponentielle, puis on les retranche
    de DATA_NOW. Clipe a DATA_EPOCH au plus tot.
    """
    days_back = rng.exponential(scale=CREATED_AT_SCALE_DAYS, size=n)
    max_days = (DATA_NOW - DATA_EPOCH).total_seconds() / 86400.0
    days_back = np.clip(days_back, 0.0, max_days)
    return [DATA_NOW - timedelta(days=float(d)) for d in days_back]


def _generate_countries(rng: np.random.Generator, n: int) -> np.ndarray:
    """Tirage multinomial pour les pays selon COUNTRY_WEIGHTS."""
    return rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)


def _generate_ages(rng: np.random.Generator, n: int) -> np.ndarray:
    """Normal(34, 12) clipe a [18, 99], renvoye en int."""
    raw = rng.normal(loc=AGE_MEAN, scale=AGE_STD, size=n)
    clipped = np.clip(raw, AGE_MIN, AGE_MAX)
    return np.round(clipped).astype(np.int64)


def _generate_premium(
    rng: np.random.Generator, ages: np.ndarray, countries: np.ndarray
) -> np.ndarray:
    """Bernoulli par ligne selon `premium_probability(age, country)`."""
    probs = np.array(
        [premium_probability(int(a), str(c)) for a, c in zip(ages, countries)],
        dtype=np.float64,
    )
    draws = rng.uniform(size=len(probs))
    return draws < probs


def _generate_last_login(
    rng: np.random.Generator,
    premium: np.ndarray,
    created_at: list[datetime],
) -> list[datetime]:
    """`last_login` exponentiel : ~7j si premium, ~60j sinon.

    Contraintes : `created_at <= last_login <= DATA_NOW`.
    """
    n = len(premium)
    scales = np.where(premium, LAST_LOGIN_SCALE_PREMIUM, LAST_LOGIN_SCALE_NON_PREMIUM)
    # `rng.exponential` accepte un scale scalaire ; on tire un U(0,1) et on
    # applique l'inverse CDF par-ligne pour passer un scale vectoriel.
    u = rng.uniform(size=n)
    # Clip pour eviter log(0) = -inf en floating-point.
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    days_back = -np.log(1.0 - u) * scales

    result: list[datetime] = []
    for i in range(n):
        candidate = DATA_NOW - timedelta(days=float(days_back[i]))
        # Garantir l'ordre temporel : last_login >= created_at, <= DATA_NOW.
        if candidate < created_at[i]:
            candidate = created_at[i]
        if candidate > DATA_NOW:
            candidate = DATA_NOW
        result.append(candidate)
    return result


def _generate_signup_sources(rng: np.random.Generator, countries: np.ndarray) -> np.ndarray:
    """Tire signup_source par pays selon SIGNUP_WEIGHTS_BY_GROUP.

    Pour la reproductibilite et la vectorisation, on groupe par groupe-pays.
    """
    n = len(countries)
    sources = np.empty(n, dtype=object)
    # Tirage par groupe pour vectoriser et garder un ordre deterministe.
    for group, weights in SIGNUP_WEIGHTS_BY_GROUP.items():
        mask = np.array(
            [SIGNUP_GROUP_BY_COUNTRY[str(c)] == group for c in countries],
            dtype=bool,
        )
        count = int(mask.sum())
        if count == 0:
            continue
        sources[mask] = rng.choice(SIGNUP_SOURCES, size=count, p=weights)
    return sources


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def generate_fake_users(n_rows: int, seed: int) -> pd.DataFrame:
    """Genere un DataFrame de n_rows utilisateurs synthetiques reproductible.

    Args:
        n_rows: Nombre d'utilisateurs a generer.
        seed: Graine pour le PRNG (controle l'ensemble de la generation).

    Returns:
        DataFrame avec les colonnes :
        id, email, password_hash, created_at, country, age, premium,
        last_login, signup_source.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows doit etre > 0, recu {n_rows}")

    logger.info("Generating %d synthetic users (seed=%d)", n_rows, seed)
    rng = np.random.default_rng(seed)
    faker = Faker("en_US")
    Faker.seed(seed)

    # Ancres irreductibles -----------------------------------------------------
    logger.info("  → generating anchors (id, email, password_hash)")
    ids = _generate_uuids(rng, n_rows)
    emails = _generate_unique_emails(faker, n_rows)
    password_hashes = _generate_password_hashes(rng, n_rows)

    # Colonnes regenerables ----------------------------------------------------
    # Ordre important : countries et ages sont generes AVANT premium et
    # signup_source car ces derniers en dependent.
    logger.info("  → generating distributions (created_at, country, age)")
    created_at = _generate_created_at(rng, n_rows)
    countries = _generate_countries(rng, n_rows)
    ages = _generate_ages(rng, n_rows)

    logger.info("  → generating correlations (premium, last_login, signup_source)")
    premium = _generate_premium(rng, ages, countries)
    last_login = _generate_last_login(rng, premium, created_at)
    signup_sources = _generate_signup_sources(rng, countries)

    # Assemblage ---------------------------------------------------------------
    df = pd.DataFrame(
        {
            "id": ids,
            "email": emails,
            "password_hash": password_hashes,
            "created_at": created_at,
            "country": countries,
            "age": ages,
            "premium": premium,
            "last_login": last_login,
            "signup_source": signup_sources,
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
        description="Generate a synthetic users CSV for the semantic-compressor POC.",
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
        help=f"Number of users to generate (default: {DEFAULT_N_ROWS})",
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

    premium_rate = float(df["premium"].mean())
    country_counts = df["country"].value_counts(normalize=True).sort_values(ascending=False)

    summary = Table(title="Generated dataset summary", show_header=True, header_style="bold")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Rows", f"{len(df):,}")
    summary.add_row("Output", str(output))
    summary.add_row("Size", f"{size_kb:,.1f} KB ({size_bytes:,} bytes)")
    summary.add_row("Premium rate", f"{premium_rate * 100:.2f}%")
    summary.add_row("Mean age", f"{df['age'].mean():.1f}")
    console.print(summary)

    country_table = Table(title="Country distribution", show_header=True, header_style="bold")
    country_table.add_column("Country", style="cyan")
    country_table.add_column("Share", justify="right", style="green")
    country_table.add_column("Count", justify="right")
    for country, share in country_counts.items():
        count = int((df["country"] == country).sum())
        country_table.add_row(str(country), f"{share * 100:.2f}%", f"{count:,}")
    console.print(country_table)


def main(argv: list[str] | None = None) -> int:
    """Entry point CLI. Retourne exit code."""
    _setup_logging()
    args = _parse_args(argv)

    console = Console()
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = generate_fake_users(n_rows=args.n_rows, seed=args.seed)
    except Exception:
        logger.exception("Generation failed")
        return 1

    logger.info("Writing CSV to %s", output)
    # `date_format` ISO 8601 pour une serialisation stable et reparsable.
    df.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")

    _print_summary(df, output, console)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
