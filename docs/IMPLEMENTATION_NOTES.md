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

---

## 6. Évolutions post-spec

Cette section documente les décisions techniques prises pendant l'implémentation, après rédaction de la spec initiale. Toutes ces évolutions sont déjà intégrées dans le code des modules `src/*.py`.

### 6.1 Direction resolution par cardinalité (`pattern_detector.py`)

La première version du pattern_detector promouvait en `CONDITIONAL_DISTRIBUTION` n'importe quelle paire de colonnes au-dessus du seuil de corrélation. Conséquence sur `users.csv` : `country` et `signup_source` étaient mutuellement corrélés (Cramér's V élevé), et la résolution arbitraire (premier croisement gagnant) créait un cycle. La colonne à haute cardinalité (`country`, 10 valeurs avec distribution biaisée 50/20/10/...) finissait conditionnée sur la colonne à faible cardinalité (`signup_source`, 3 valeurs), ce qui diluait sa distribution marginale lors de la reconstruction.

**Fix** : `_choose_strongest_correlation` (cf. `src/pattern_detector.py:930`) ne retourne un partenaire QUE si :

```
(target.n_unique, target_name) < (partner.n_unique, partner_name)
```

La colonne à haute cardinalité reste donc DISTRIBUTION marginale ; la colonne à faible cardinalité devient CONDITIONAL_DISTRIBUTION conditionnée sur la colonne à haute cardinalité. Tiebreaker lexicographique stable sur le nom de colonne en cas d'égalité de cardinalité.

**Effet mesuré** : fidélité `users.csv` 84/100 → 100/100 (cf. commit `78af4ef fix(fidelity): reach 100/100 on users.csv`).

### 6.2 KS D-statistic à grand n (`validator.py`)

Le KS-test renvoie une p-value qui tend vers 0 quand n grandit, même pour des distributions visuellement identiques. À n ≥ 1000 la p-value devient trompeuse : un écart imperceptible (D ~ 0.03) suffit à faire chuter p sous 0.05 et donc rejeter une reconstruction parfaitement acceptable.

**Fix** : au-dessus du seuil `ks_large_n_threshold = 1000` lignes, la validation bascule sur le D-statistic (distance de Kolmogorov, valeur dans [0, 1], indépendante de n). Seuils :

- `ks_distance_max = 0.05` (standard)
- `ks_distance_max_relaxed = 0.10` pour les distributions très skewed (`|skew| > skew_relax_threshold = 2.0`)

Implémentation dans `ValidationThresholds` (cf. `src/validator.py:53`). En dessous de n=1000, on conserve l'ancien critère p-value pour la cohérence statistique sur petits échantillons.

### 6.3 Détection des datetimes en colonnes `object` (`validator.py`)

Le CSV est lu sans `parse_dates` (volontaire : on veut préserver les types tels qu'ils ont été générés et éviter les heuristiques pandas implicites). Conséquence : les datetimes ISO 8601 arrivent en dtype `object`, indissociables au premier coup d'œil d'une colonne de strings catégorielles.

**Fix** : `_is_datetime` (cf. `src/validator.py:96`) parse un échantillon de 50 valeurs via `pd.to_datetime(..., errors="coerce", utc=True)`. Si au moins 90% du sample parse avec succès, la colonne est classée DATETIME. Sinon, fallback string/catégoriel.

**Pourquoi ce seuil compte** : sans cette détection, une colonne datetime serait routée vers `test_categorical_frequencies` (puisqu'elle est `object`), qui ferait toujours échouer `value_set_mismatch` car chaque timestamp est unique.

### 6.4 `PatternType.RANDOM_FORMAT`

Pour les colonnes dont la valeur exacte n'a pas d'information sémantique (filler aléatoire : hashes bcrypt, UUID, hashes hex), on stocke uniquement le format spec dans la recette — pas les 10 000 valeurs. À la reconstruction, le `reconstructor` génère des valeurs uniques conformes au format via une seed `sha256` per-row.

**Recette** (par colonne) :
```json
{
  "type": "RANDOM_FORMAT",
  "format_spec": {
    "regex": "^\\$bcrypt\\$2b\\$12\\$[0-9a-f]{64}\\$$",
    "prefix": "$bcrypt$2b$12$",
    "suffix": "$",
    "body_type": "hex",
    "body_length": 64,
    "detected_as": "bcrypt_hash"
  }
}
```

**Validation** : `test_random_format_compliance` vérifie regex match + uniqueness + length — pas d'exact-match. Ces tests sont "soft" : on accepte que les valeurs reconstruites diffèrent de l'original ligne-à-ligne, tant qu'elles respectent le format et que l'ensemble reste unique.

**KNOWN_RANDOM_FORMATS** auto-détectés (cf. `src/pattern_detector.py:785`) :

| Nom | Regex | Cible |
|---|---|---|
| `bcrypt_hash` | `^\$bcrypt\$\d+[a-z]?\$\d+\$[0-9a-f]+\$$` | Hashes bcrypt synthétiques |
| `uuid_v4_anchor` | RFC 4122 v4 strict | UUIDs (avec garde-fou, cf. 6.6) |
| `hex_hash` | `^[0-9a-f]{32,128}$` | SHA-256, SHA-512, MD5... |

Ordre des regex important : `uuid_v4_anchor` est testé avant `hex_hash` (les UUIDs contiennent des tirets et matchent `uuid_v4_anchor` d'abord ; sans tirets ils tomberaient sur `hex_hash`).

### 6.5 `PatternType.EMAIL_SPLIT`

Pour les colonnes EMAIL avec un petit dictionnaire de domaines (typiquement < 256 distincts), on décompose `user@domain` en deux ancres synthétiques :

- `<col>__local` : texte (le `local_part`, irréductible en général)
- `<col>__domain_idx` : entier non signé (uint8 si dict ≤ 256, sinon uint16)

Le dictionnaire `{idx: domain_str}` (≤ ~100 octets pour 10 domaines) est stocké dans la recette. **Lossless** : la reconstruction recolle `local + "@" + dict[idx]` et obtient l'email original octet pour octet.

Le gain de compression vient du fait que stocker N lois sur 10 domaines en uint8 (N octets) est beaucoup plus dense que stocker N strings de domaine (~12 octets/ligne après compression snappy). Sur `users.csv` (10 000 emails, 4 domaines distincts), gain ≈ 80% de la portion "domaine" des ancres email.

Implémentation : `detect_email_split` (`src/pattern_detector.py:864`) + post-traitement par `anchor_extractor._extract_email_split_columns`.

### 6.6 Auto-détection vs CLI override

Décisions par défaut au moment de classifier une colonne ancre candidate :

| Colonne typique | Format détecté | Décision par défaut | Override |
|---|---|---|---|
| `password_hash` | `bcrypt_hash` | RANDOM_FORMAT (regen) | `--no-random-format` (manuel) |
| `id`, `*_id`, `uuid` | `uuid_v4_anchor` | ANCHOR_DIRECT (préservation exacte) | `--aggressive-uuid` |
| `email` (≤ 256 domaines) | EMAIL_REGEX + small dict | EMAIL_SPLIT (lossless) | implicite (pas de flag dédié) |
| UUID dans une colonne au nom NON id-like (ex: `session_token`) | `uuid_v4_anchor` | RANDOM_FORMAT (regen) | aucun (comportement attendu) |

**Garde-fou UUID** : seules les colonnes dont le nom matche `_ID_LIKE_COLUMN_NAME_REGEX` (cf. `src/pattern_detector.py:743`) :
```
^(id|uuid|uid|pk|.*_id|.*_uuid|.*_uid|.*_pk)$
```
restent en ANCHOR_DIRECT même quand leur contenu matche `uuid_v4_anchor`. Le flag CLI `--aggressive-uuid` désactive ce garde-fou et force RANDOM_FORMAT, ce qui est utile sur les datasets où on sait que les UUIDs ne sont pas des PK user-facing (et donc qu'aucune cohérence cross-table n'a besoin d'être préservée).

**Rationale du défaut** :

- `password_hash` (et tout filler aléatoire) : la valeur exacte n'a aucune valeur informationnelle. On stocke 200 octets de spec au lieu de ~640 KB de hashes. Pas de risque sémantique.
- `id` (et autres `*_id`) : c'est typiquement une PK user-facing, exposée dans des URLs, partagée avec d'autres systèmes ou tables. Préserver la valeur exacte est la valeur par défaut prudente.
- `email` : EMAIL_SPLIT est lossless, donc aucun risque ; et le gain est mesurable (cf. 6.5).

---

## 7. Lessons learned

### Tests parallèles d'agents

- **Wave 1 (4 agents parallèles)** : profiler, anchor_extractor, pattern_detector, validator développés en parallèle. Aucune dépendance croisée (tous consomment uniquement `models.py`). Aucun conflit, premier essai concluant.
- **Wave 2 (2 agents parallèles + contrat de format strict)** : `recipe_writer` et `reconstructor` ont un contrat partagé (le format `.md` que l'un écrit et l'autre parse). Avant de lancer les agents, on a figé le contrat dans `models.py` (`Recipe`, `Pattern`, `format_spec`). Pas de divergence.
- **Wave 5 (3 agents parallèles, ratio + HTML profiling + 2nd dataset)** : pendant que l'agent 5.2 mesurait les ratios sur `users.csv`, l'agent 5.1 n'avait pas encore commité son fix RANDOM_FORMAT → l'agent 5.2 a mesuré un état intermédiaire et reporté un faux échec. Résolu en re-mesurant après commit de 5.1.

**Conclusion** : la parallelisation marche tant que les agents n'écrivent pas dans le même fichier ET que les agents de mesure tournent *après* les agents qui modifient le pipeline. Dans le doute : séquencer manuellement.

### Direction resolution était critique

Sans le fix de la section 6.1, le ratio sur `users.csv` plafonnait à 84/100 de fidélité — passable mais sous le critère d'acceptation 95%. Avec le fix, 100/100. Le bug est subtil : aucune assertion ne plantait, les distributions étaient juste "un peu fausses" sur deux colonnes. Sans test de fidélité automatisé, on serait passé à côté. **Investir dans la métrique de qualité paye plus que d'investir dans plus de patterns.**

### Le seuil de signification statistique dépend de n

À n=10 000, la p-value KS est trompeuse (cf. 6.2). Le critère SPEC.md "p > 0.05" était littéralement infaisable sur les distributions skewed à 10k lignes. **Décision** : on a remplacé "p > seuil" par "D-statistic < seuil" pour les gros n. Cette substitution est documentée dans le `ValidationReport` (chaque test reporte quel critère il a utilisé).

**Leçon générale** : les seuils statistiques de la spec sont des défauts raisonnables sur petits échantillons. Sur grand n, il faut basculer sur des métriques size-invariant (distances normalisées plutôt que tests d'hypothèse).

### Tous les anchors ne sont pas égaux

La spec initiale traitait toute "colonne unique par ligne" comme une ancre à préserver. La réalité est plus nuancée :

- **PK user-facing** (`id`, `email`) : préservation exacte requise (FK potentielles, URLs, partages cross-system).
- **Filler aléatoire** (`password_hash`, `session_token`, `*_secret`) : seul le format compte. Sa préservation exacte n'a aucune valeur métier.
- **Identifiant lossless splittable** (`email` avec petit dict de domaines) : préservation exacte tout en réduisant l'entropie stockée.

Distinguer ces trois cas (RANDOM_FORMAT vs ANCHOR_DIRECT vs EMAIL_SPLIT) débloque des compressions massives. Sur `users.csv` : ratio 3.07:1 → 6.15:1 (commit `3e93d42`) → 15.89:1 avec flags explicites (commit `ced3ab0`).

### Tester sur un 2nd dataset révèle les biais

Le pattern_detector tuned sur `users.csv` (distributions propres, un pivot catégoriel `country`) a donné 3.52:1 et 66.7% de fidélité sur `orders.csv` (cf. `SECOND_DATASET.md`). Ce n'est **pas un bug** : c'est une couverture de patterns insuffisante (pas de support log-normal, pas d'OFFSET_FROM_COLUMN, pas de CONDITIONAL_NULL).

**Leçon** : le risque dans ce genre de projet n'est pas que le code soit faux sur le dataset de référence — c'est qu'il soit *trop tuned* pour lui. Tester sur un dataset au profil différent dès qu'on en a un est indispensable pour valider la généralité. La V2 doit ajouter ces patterns avant de prétendre couvrir le cas général.

---

## 8. Future work

Liste priorisée des extensions identifiées pendant le POC, à instruire en V2. Ordre approximatif de coût croissant / gain décroissant.

### 8.1 Nouveaux pattern types (priorité 1)

Ces patterns débloqueraient `orders.csv` au-delà de 90% de fidélité (cf. `SECOND_DATASET.md` §6) :

- **`LOG_NORMAL`** : distribution log-normale (scipy.stats.lognorm). Cas d'usage : `quantity` dans `orders.csv` (skew ~3, mal approximé par exponentielle). Coût faible (un fitter scipy de plus dans le catalogue de distributions).
- **`OFFSET_FROM_COLUMN`** : col B = col A + Distribution(delta). Cas d'usage : `shipped_at = ordered_at + Exp(3j)` dans `orders.csv`. Coût moyen (nouveau type de relation à serialiser dans la recette + nouveau path de génération dans le reconstructor).
- **`CONDITIONAL_NULL`** : col B est NULL si col A ∈ ensemble. Cas d'usage : `shipped_at` NULL si `status ∈ {pending, processing, cancelled}`. Coût faible (un masque NULL appliqué après génération).
- **`CATEGORICAL_HIERARCHY`** : extension de FUNCTIONAL_DEP avec ouvertures partielles (mapping SKU → category sans devoir matérialiser les 500 SKUs). Coût moyen.

### 8.2 Intégration LLM (priorité 2)

Ce qui n'est pas faisable proprement en algorithmique pure :

- Détection sémantique de patterns dans le texte libre (commentaires, descriptions, free-text).
- Génération de prompts pour reconstruire du texte naturel à partir d'une distribution de tons / sujets.
- Sentiment / topic patterns (la recette devient un prompt + un seed, le générateur appelle Claude/GPT à la reconstruction).

Coût élevé (changement d'architecture : la reconstruction n'est plus déterministe sans LLM).

### 8.3 Multi-table relational (priorité 2)

Étend la recette à plusieurs tables :

- Foreign keys explicites dans le schéma.
- Reconstruction de plusieurs tables avec contraintes référentielles (typiquement : générer la table parent d'abord, échantillonner les FK depuis ses PKs).
- Joins déclarés pour la validation.

Débloque le cas relationnel typique (users + orders + items). Coût moyen-élevé (refonte du format de recette et de l'orchestrator).

### 8.4 Compression incrémentale (priorité 3)

Ajouter N nouvelles lignes à un dataset déjà compressé sans tout re-profiler :

- Conserve la recette telle quelle (suppose que les patterns restent valides).
- Étend uniquement le parquet d'ancres avec les nouvelles lignes.
- Re-vérifie la fidélité sur l'union (full original + delta).

Utile pour les datasets append-only (logs, événements). Coût moyen.

### 8.5 Format binaire de recette (priorité 4)

Le `.md` reste l'output de référence (lisible humain). En option, un format binaire (msgpack, CBOR) plus dense pour les pipelines automatisés où la lisibilité ne compte pas. Gain marginal (la recette pèse déjà <10% du compressé). Coût faible.

### 8.6 Pluggable storage backends (priorité 4)

Ingestion directe depuis PostgreSQL / MySQL au lieu de passer par un CSV intermédiaire. Sortie vers les mêmes backends (reconstruction → INSERT). Coût moyen (adapters par BD). Surtout utile pour les datasets qui ne tiennent pas en mémoire.
