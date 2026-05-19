# SEMANTIC COMPRESSOR — Proof of Concept

## Objectif du projet

Construire un proof of concept (POC) d'un système de **compression sémantique** de base de données.

Au lieu de stocker les données brutes, on extrait :
1. Une **recette** (`.md` lisible) qui décrit comment régénérer les données
2. Des **ancres** (`.parquet`) qui contiennent uniquement l'irréductible (IDs uniques, valeurs non régénérables)
3. Un **générateur** qui reconstruit la BD à partir de la recette + ancres

Métrique de succès : compresser une BD de N lignes à un ratio ≥ 10:1 avec une fidélité statistique ≥ 95%.

## Stack technique

- **Python 3.11+**
- **Pandas** + **NumPy** pour la manipulation
- **DuckDB** pour les requêtes rapides
- **Pydantic** pour les schémas
- **ydata-profiling** pour le profilage initial
- **Faker** pour générer la fausse BD de test
- **Rich** pour l'affichage console
- **pytest** pour les tests

Pas de LLM externe dans le POC initial — on prouve d'abord que la logique algorithmique fonctionne. Le LLM viendra en V2 pour détecter les patterns sémantiques avancés.

## Architecture cible

```
semantic-compressor/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── profiler.py          # Analyse la BD, sort un profil statistique
│   ├── pattern_detector.py  # Trouve les patterns et corrélations
│   ├── anchor_extractor.py  # Identifie l'irréductible (IDs, valeurs uniques)
│   ├── recipe_writer.py     # Écrit le fichier .md de recette
│   ├── reconstructor.py     # Régénère la BD depuis recette + ancres
│   ├── validator.py         # Compare original vs reconstruit (fidélité)
│   ├── orchestrator.py      # Pipeline complet (compress + decompress)
│   └── models.py            # Schémas Pydantic
├── examples/
│   ├── generate_fake_db.py  # Crée une BD de test (10k users)
│   └── run_poc.py           # Démo end-to-end
├── tests/
│   ├── test_profiler.py
│   ├── test_reconstructor.py
│   └── test_validator.py
├── output/
│   ├── recipes/             # .md générés
│   └── anchors/             # .parquet d'ancres
└── data/
    └── original/            # BD de test
```

## Spécifications détaillées par module

### 1. `models.py` — Schémas Pydantic

Définir :
- `ColumnProfile` : nom, type, distribution, % nulls, cardinalité, exemples
- `Correlation` : col_a, col_b, type, force (0-1)
- `Pattern` : type (template/distribution/lookup), règle, fidélité
- `Recipe` : metadata + schema + anchors_ref + rules + validation_tests
- `ValidationReport` : metric, expected, actual, passed

### 2. `profiler.py`

Fonction principale : `profile_dataframe(df: pd.DataFrame) -> dict`

Pour chaque colonne, détecter :
- Type (numeric, categorical, datetime, string, boolean)
- Distribution statistique (mean, std, min, max, quartiles pour num ; freq pour cat)
- Cardinalité (uniques / total)
- Taux de nullité
- Détection de patterns regex pour les strings (email, phone, URL, UUID)

Output : dict structuré sérialisable en JSON.

### 3. `pattern_detector.py`

Trois détecteurs principaux :

**a) `detect_functional_dependencies(df)`**
Trouve les colonnes déductibles d'autres. Exemple : `country_name` ↔ `country_code`.
Méthode : pour chaque paire (A, B), vérifier si `A → B` est une fonction (chaque valeur de A mappe vers une seule valeur de B).

**b) `detect_distributions(df)`**
Pour chaque colonne numérique, tester :
- Normale (Shapiro-Wilk)
- Exponentielle
- Uniforme
- Power law

Garder la meilleure avec ses paramètres.

**c) `detect_correlations(df)`**
- Pearson pour numérique-numérique
- Cramér's V pour catégoriel-catégoriel
- ANOVA pour numérique-catégoriel

Seuil de signification : |corr| ≥ 0.3

### 4. `anchor_extractor.py`

Identifier ce qui DOIT être stocké :
- Toute colonne avec cardinalité = 1.0 (unique par ligne, ex: IDs, emails)
- Toute colonne marquée comme `irreducible` manuellement
- Tout ce qui n'a pas de pattern détectable

Output : DataFrame réduit + liste des colonnes ancres.

### 5. `recipe_writer.py`

Génère le fichier `.md` selon le template suivant :

```markdown
# RECIPE: {table_name}
Version: 1.0
Source rows: {n_rows}
Compressed size: {size_kb} KB
Original size: {orig_size_kb} KB
Compression ratio: {ratio}:1
Fidelity target: 95%
Generated: {timestamp}

## SCHEMA
{column definitions}

## ANCHORS
File: anchors_{table}.parquet
Columns: {anchor_columns}
Size: {anchor_size}

## GENERATION RULES
{rules per column with seed strategy}

## RECONSTRUCTION ALGORITHM
{step-by-step}

## VALIDATION TESTS
{list of automated tests with thresholds}
```

Le `.md` doit être **lisible par un humain** ET **parsable** (sections délimitées clairement).

### 6. `reconstructor.py`

Fonction : `reconstruct(recipe_path, anchor_path) -> pd.DataFrame`

Étapes :
1. Parser le `.md` de recette
2. Charger les ancres
3. Pour chaque ligne d'ancre, générer les colonnes manquantes selon les règles
4. Utiliser `seed = hash(id)` pour la reproductibilité
5. Appliquer les corrélations dans l'ordre topologique

### 7. `validator.py`

Compare l'original et le reconstruit selon :

**Tests structurels (exact)**
- Nombre de lignes identique
- Schéma identique
- Ancres identiques (test exact sur les colonnes d'ancres)

**Tests statistiques (tolérance)**
- Distribution de chaque colonne (KS-test, p > 0.05)
- Moyennes ± 2%
- Corrélations préservées ± 5%
- Fréquences catégorielles ± 3%

**Output** : `ValidationReport` avec score global (0-100) + détails.

### 8. `orchestrator.py`

Pipeline complet exposé via CLI :

```bash
# Compression
python -m src.orchestrator compress --input data/users.csv --output output/

# Reconstruction
python -m src.orchestrator decompress --recipe output/recipes/users.md --output reconstructed.csv

# Validation
python -m src.orchestrator validate --original data/users.csv --reconstructed reconstructed.csv
```

Affichage avec Rich (table de résultats, barres de progression).

## Fausse BD de test (`examples/generate_fake_db.py`)

Générer 10 000 utilisateurs avec :
- `id` : UUID unique → ANCRE
- `email` : unique → ANCRE
- `password_hash` : unique → ANCRE
- `created_at` : timestamp avec distribution exponentielle (plus dense récemment)
- `country` : enum (10 pays) avec distribution biaisée (50% US, 20% FR, etc.)
- `age` : normal(34, 12), clipé [18, 99]
- `premium` : 7% true globalement, mais corrélé à age et country
- `last_login` : exponentielle, plus récent si premium
- `signup_source` : enum (web, mobile, referral) corrélé à country

Le but : avoir une BD avec **des patterns détectables** ET des **valeurs uniques irréductibles**, pour tester les deux mécanismes.

## Démo end-to-end (`examples/run_poc.py`)

Script qui :
1. Génère la fausse BD (10k lignes, ~2 MB)
2. La profile et affiche les patterns détectés
3. Extrait les ancres
4. Écrit la recette `.md`
5. Reconstruit la BD depuis recette + ancres
6. Valide la reconstruction
7. Affiche un rapport final :
   - Taille originale vs taille compressée
   - Ratio de compression
   - Score de fidélité
   - Temps de compression et de reconstruction

## Critères d'acceptation du POC

- Compression ratio >= 5:1 (objectif 10:1)
- Fidelite statistique >= 95% sur toutes les metriques
- Reconstruction reproductible (deux runs = meme output exact)
- Recette `.md` lisible par un humain
- CLI fonctionnelle (compress / decompress / validate)
- Tests unitaires sur les 3 modules cles (profiler, reconstructor, validator)
- README clair avec exemples

## Notes d'implémentation

- **Reproductibilité** : tous les RNG doivent être seedés à partir de `hash(anchor_id)` pour garantir que reconstruire deux fois donne le même résultat.
- **Performance** : viser <30 secondes pour compresser 10 000 lignes en local.
- **Modularité** : chaque module doit être utilisable indépendamment (testable, importable).
- **Pas de magie** : pas de LLM externe dans cette V1. Tout doit être déterministe et explicable.
- **Logging** : utiliser le module `logging` standard, avec niveau INFO par défaut.

## V2 future (NE PAS implémenter maintenant, juste documenter dans README)

- Intégration LLM (Claude/GPT) pour détecter des patterns sémantiques dans le texte libre
- Support de tables relationnelles (foreign keys)
- Compression incrémentale (ajouter des lignes sans recompresser tout)
- Format binaire de recette pour gain de taille
- Plugin pour PostgreSQL / MySQL direct

## Ordre d'implémentation suggéré

1. `models.py` (les schémas Pydantic — fondation)
2. `examples/generate_fake_db.py` (avoir des données pour tester)
3. `profiler.py` (analyse de base)
4. `anchor_extractor.py` (séparer le réductible de l'irréductible)
5. `pattern_detector.py` (le cœur intellectuel)
6. `recipe_writer.py` (output lisible)
7. `reconstructor.py` (le décompresseur)
8. `validator.py` (mesurer la fidélité)
9. `orchestrator.py` (assembler en CLI)
10. `examples/run_poc.py` (démo finale)
11. Tests pytest

À chaque étape : commit Git avec message clair. Tester avant de passer à l'étape suivante.
