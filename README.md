# Semantic Compressor — POC

> Stocker la **recette** d'une base de données au lieu des données brutes.
> Compresser une BD en : un `.md` lisible (la recette) + un `.parquet` minimal (les ancres irréductibles).
> Régénérer la BD avec une fidélité statistique ≥ 95%.

## Idée

Au lieu de stocker des millions de lignes brutes, on extrait :

1. **Une recette** (`output/recipes/<table>.md`) — décrit comment régénérer la BD : schéma, distributions, corrélations, dépendances fonctionnelles, règles de génération.
2. **Des ancres** (`output/anchors/<table>.parquet`) — les valeurs irréductibles uniquement (IDs, emails, hash de password — tout ce qui ne peut pas être inféré statistiquement).
3. **Un générateur** — reconstruit la BD originale à partir de la recette + des ancres, avec seeding déterministe pour garantir la reproductibilité bit à bit.

**Analogie** : c'est comme stocker la recette d'un gâteau (1 page) plutôt qu'une photo HD de chaque tranche. Avec la recette + un peu de farine spécifique (les ancres), on régénère le gâteau quasi-identique.

## Objectifs du POC

- Ratio de compression ≥ 5:1 (objectif 10:1) sur 10 000 lignes synthétiques
- Fidélité statistique ≥ 95% (KS-test, corrélations, moyennes, fréquences catégorielles)
- Reconstruction **strictement reproductible** : deux runs = même output bit à bit
- CLI complète : `compress` / `decompress` / `validate`
- Pas de LLM, pas de magie. Tout est déterministe et explicable.

## Stack

Python 3.11+, pandas, numpy, DuckDB, Pydantic, ydata-profiling, scipy, Faker, Rich, pytest.

## Quick start

```powershell
# Setup
cd D:\semantic-compressor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Generate the test database (10k synthetic users)
python examples\generate_fake_db.py

# Run the full POC end-to-end (compress + decompress + validate + report)
python examples\run_poc.py
```

## CLI

```powershell
# Compress
python -m src.orchestrator compress --input data\original\users.csv --output output\

# Decompress (reconstruct)
python -m src.orchestrator decompress --recipe output\recipes\users.md --output data\reconstructed\users.csv

# Validate (compare original vs reconstructed)
python -m src.orchestrator validate --original data\original\users.csv --reconstructed data\reconstructed\users.csv
```

## Architecture

```
src/
  models.py             Pydantic schemas (ColumnProfile, Recipe, Pattern, etc.)
  profiler.py           Profil statistique par colonne
  pattern_detector.py   Distributions, corrélations, dépendances fonctionnelles
  anchor_extractor.py   Sépare l'irréductible du régénérable
  recipe_writer.py      Sérialise la recette en Markdown
  reconstructor.py      Régénère la BD depuis recette + ancres
  validator.py          Compare original vs reconstruit (structurel + statistique)
  orchestrator.py       Pipeline + CLI (compress / decompress / validate)
```

## Tests

```powershell
pytest
```

## Spec complète

Voir [docs/SPEC.md](docs/SPEC.md) et [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).

## V2 (non implémenté dans ce POC)

- Intégration LLM (Claude/GPT) pour détecter des patterns sémantiques dans le texte libre
- Support de tables relationnelles (foreign keys)
- Compression incrémentale (ajouter des lignes sans recompresser l'intégralité)
- Format binaire de recette (gain de taille additionnel)
- Plugins PostgreSQL / MySQL pour ingestion directe

## Licence

MIT.
