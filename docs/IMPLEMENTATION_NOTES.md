# IMPLEMENTATION NOTES

Ce document complète [SPEC.md](SPEC.md) en capturant les décisions techniques que la spec ne tranche pas, ainsi que les risques identifiés.

> **Lecture** : SPEC.md = le **quoi**. Ce document = le **comment** et le **pourquoi**.

---

## 1. Décisions techniques

### 1.1 Environnement Python : `venv` + `pip`

- Pas de Poetry, Hatch ou uv. `python -m venv .venv` + `pip install -r requirements.txt`.
- **Pourquoi** : cohérent avec le principe "pas de magie, tout doit être explicable". Marche sur tout environnement Python 3.11+ sans outil tiers.

### 1.2 Seeding reproductible : `hashlib.sha256`

```python
import hashlib

def derive_seed(anchor_id: str) -> int:
    """Convertit un ID d'ancre en seed 32 bits déterministe et reproductible."""
    digest = hashlib.sha256(str(anchor_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)  # 32 bits, valide pour np.random.default_rng
```

- **Pourquoi** : `hash()` natif Python est randomisé par défaut (cf. variable d'environnement `PYTHONHASHSEED`). Il casserait la reproductibilité inter-process.
- **Alternative écartée** : `zlib.crc32` (plus rapide mais 32 bits seulement, plus de collisions sur 10k+ lignes).

### 1.3 Profiling : `ydata-profiling` intégré dans `profiler.py`

- Utiliser `ProfileReport(df).description_set` pour extraire les statistiques par colonne, puis mapper vers nos `ColumnProfile` Pydantic.
- Générer un rapport HTML d'exploration dans `output/profiling_reports/` (utile pour debug humain).
- **Risque** : compatibilité Python 3.13. Si `pip install ydata-profiling` échoue, fallback : profileur maison avec pandas + scipy uniquement. Documenter le fallback dans le code.

### 1.4 Stats : `scipy.stats`

- KS-test : `scipy.stats.ks_2samp`
- Shapiro-Wilk : `scipy.stats.shapiro` (attention : limité à n <= 5000, on échantillonne au besoin)
- Pearson : `scipy.stats.pearsonr`
- ANOVA : `scipy.stats.f_oneway`
- Distribution fitting : `scipy.stats.expon.fit`, `scipy.stats.norm.fit`, `scipy.stats.uniform.fit`, `scipy.stats.powerlaw.fit`

### 1.5 Cramér's V : implémentation maison

`scipy` n'a pas de fonction directe. ~5 lignes :

```python
from scipy.stats import chi2_contingency
import numpy as np

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.values.sum()
    r, k = contingency.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))
```

### 1.6 Parquet : `pyarrow` engine

- Engine : `pandas` détecte `pyarrow` automatiquement.
- Compression : `snappy` par défaut (équilibre vitesse/taille).
- Sur les ancres `(id, email, password_hash)` — du texte hautement entropique — la compression snappy gagne ~3-5x.

### 1.7 Corrélations conditionnelles : bucketing simple

Pour les paires (numérique pivot, autre colonne) :
1. Bucketer le pivot en `qcut` (10 quantiles par défaut).
2. Calculer les paramètres de distribution (ou fréquences) par bucket.
3. À la reconstruction : pour chaque ligne, identifier son bucket via la valeur pivot, tirer dans la distribution conditionnelle.

- **Pourquoi pas de copule gaussienne** : plus propre statistiquement mais lourd à implémenter et expliquer pour un POC. Bucketing donne ~95% de la fidélité pour 10% de la complexité.

### 1.8 Ordre topologique des règles

Construction d'un DAG :
- Nœuds = colonnes
- Arêtes = dépendances (`A -> B` si B est dérivée de A : dépendance fonctionnelle ou corrélation conditionnelle)
- Tri topologique : `graphlib.TopologicalSorter` (stdlib Python 3.9+)

À la reconstruction, on génère dans l'ordre topologique : d'abord les colonnes racines (ancres + colonnes purement distributionnelles), puis les colonnes dérivées.

### 1.9 Logging

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
```

Chaque module crée son logger via `logger = logging.getLogger(__name__)`.

---

## 2. Risques identifiés et mitigations

### Risque 1 : Ratio 10:1 difficile à atteindre

**Constat** :
- UUID v4 = 36 caractères / ligne
- Email synthétique = ~25 caractères / ligne
- Password hash (bcrypt-like) = ~64 caractères / ligne
- → Minimum incompressible : ~125 octets × 10 000 lignes = ~1.25 MB d'ancres
- BD originale CSV : ~2 MB → ratio max théorique sans compression d'ancres = 1.6:1

**Mitigation** :
- Compression snappy des ancres en parquet : gain ~3-5x → ~250-400 KB
- Ratio atteignable : 2 MB / ~300 KB = **6:1 à 7:1**
- Pour atteindre 10:1, il faudrait :
  - Soit accepter de hasher les colonnes d'ancres et ne stocker que les hash courts (mais on perd la régénération exacte des emails)
  - Soit dictionnary-coder les patterns d'emails (préfixe + suffixe domaine)
- **Décision** : on vise 5:1 (critère d'acceptation), on note 10:1 comme stretch goal. La spec elle-même autorise ≥5:1.

### Risque 2 : ydata-profiling sur Python 3.13

**Constat** : ydata-profiling a parfois du retard sur les versions Python récentes. Python 3.13 est sorti en octobre 2024.

**Mitigation** :
- Tester `pip install ydata-profiling` en premier ; si échec :
  - Plan B : downgrade venv à Python 3.12 (`py -3.12 -m venv .venv`)
  - Plan C : profileur maison (pandas + scipy uniquement), garder l'API `ColumnProfile` identique

### Risque 3 : KS-test trop strict pour distributions skewed

**Constat** : pour `last_login` (exponentielle), même une reconstruction de bonne qualité peut être rejetée par KS-test à p=0.05 sur 10k échantillons.

**Mitigation** :
- Pour les colonnes avec skewness > 2 ou kurtosis > 5, relâcher à p > 0.01.
- Documenter le seuil utilisé dans le `ValidationReport`.
- Préférer "distance KS < seuil" plutôt que "p-value > seuil" : moins sensible à la taille d'échantillon.

### Risque 4 : Reproductibilité bit-à-bit avec pandas/numpy

**Constat** : certaines opérations pandas (`sort_values`, `groupby` sans clé explicite) peuvent ne pas être stables entre versions.

**Mitigation** :
- Toujours trier les ancres par leur ID (la colonne pivot) avant de générer.
- Toujours utiliser `kind="mergesort"` ou `kind="stable"` pour les tris.
- Test dédié `test_reconstruction_reproducibility` : exécute la reconstruction deux fois et vérifie `pd.testing.assert_frame_equal`.

---

## 3. Stratégie de tests (légère)

- **Pas de TDD strict** (le user a dit "tests basiques")
- Tests pytest par module :
  - `test_profiler.py` : 3-4 cas sur DataFrames synthétiques (numérique, catégoriel, mixte, avec nulls)
  - `test_pattern_detector.py` : dépendance fonctionnelle évidente, détection de distribution normale
  - `test_anchor_extractor.py` : colonne unique marquée comme ancre, colonne avec pattern non marquée
  - `test_reconstructor.py` : reproductibilité (2 runs → même output)
  - `test_validator.py` : KS-test sur deux échantillons de la même distribution → score haut
- Pas d'objectif de couverture chiffré. Les tests sont des filets de sécurité, pas une preuve d'exhaustivité.

---

## 4. Conventions de code

- **Type hints partout** (Python 3.11+ syntax : `list[str]` pas `List[str]`)
- **Docstrings** courtes (1-3 lignes) au format Google ou simple
- **Commentaires** : uniquement quand le pourquoi n'est pas évident, en français quand c'est plus naturel
- **Noms** : anglais pour le code (variables, fonctions, classes) ; français pour les docstrings et commentaires d'intention si plus clair
- **Imports** : groupés (stdlib, tiers, local) et triés
- **Pas de code mort, pas de TODO sans owner**
- **Pas de print()** : tout passe par `logging`. Sauf dans la CLI où Rich gère l'affichage utilisateur.

---

## 5. Ordre d'exécution (vagues parallélisées)

| Vague | Modules | Parallélisable ? |
|---|---|---|
| 0 | `models.py`, `examples/generate_fake_db.py` | 2 agents en parallèle (pas de dépendances) |
| 1 | `profiler.py`, `anchor_extractor.py`, `pattern_detector.py`, `validator.py` | 4 agents en parallèle (tous dépendent uniquement de models) |
| 2 | `recipe_writer.py` | Séquentiel (dépend de vague 1) |
| 3 | `reconstructor.py` | Séquentiel (dépend de recipe_writer) |
| 4 | `orchestrator.py` + tests | 2 agents en parallèle |
| 5 | `examples/run_poc.py` + intégration end-to-end | Séquentiel + supervision |

Le superviseur (Claude) vérifie chaque module après réception, lance les tests, et commit avant de passer à la vague suivante.
