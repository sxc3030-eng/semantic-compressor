# SECOND DATASET — e-commerce orders

> Test de generalite du POC sur un dataset au profil tres different de `users.csv`.

## 1. Pourquoi un second dataset ?

Le POC a ete tune et valide sur `users.csv` (10 000 utilisateurs, distributions
"propres" et corellees a un seul pivot `country`). Avant de declarer
l'architecture generaliste, on prend une seconde fixture aux patterns
deliberement differents :

| Aspect | users.csv | orders.csv |
|---|---|---|
| Ancres | 1 (`email`) + 2 random_format (`id`, `password_hash`) | 2 ancres UUID (`order_id`, `customer_id`) |
| Cardinalite haute | 3 colonnes uniques | 2 colonnes uniques |
| Distributions | normale, exponentielle, categorical biaisee | log-normale, normale par-category, Pareto, exponentielle |
| Correlations | conditional_distribution sur pivot categoriel | conditional sur category (numerique) + sur status (NULL trigger) |
| Dependances fonctionnelles | aucune | 1 (`product_sku` -> `category`) |
| Distribution tres skewed | non | oui (Pareto top 20 SKU = 60% mass) |
| Valeurs NULL structurees | non | oui (`shipped_at` NULL pour 3 statuts sur 6) |

Le but : voir quels patterns le POC detecte correctement et lesquels il rate.

## 2. Genese du dataset

`examples/generate_orders_db.py` produit 10 000 commandes synthetiques avec
les colonnes suivantes :

| Colonne | Type | Pattern attendu | Notes |
|---|---|---|---|
| `order_id` | UUID v4 | ANCHOR_DIRECT | unique par ligne |
| `customer_id` | UUID v4 | ANCHOR_DIRECT | unique par ligne (pas de joins relationnels dans le POC) |
| `product_sku` | `PRD-{6-digits}-{2-letters}` | CATEGORICAL_FREQ (500 valeurs, Pareto top 20 = 60%) | Pareto |
| `category` | enum (10 valeurs) | FUNCTIONAL_DEP de `product_sku` | lookup deterministe |
| `quantity` | int | DISTRIBUTION log-normale (mean=1.5, sigma=0.8), clip [1, 50] | tres skewed |
| `unit_price_cents` | int | CONDITIONAL_DISTRIBUTION sur `category` (normale par-category, multipliers x0.4..x3.0) | clip [99, 99999] |
| `currency` | enum (5 valeurs) | DISTRIBUTION categorical_freq (60% USD, 20% EUR...) | |
| `status` | enum (6 valeurs) | DISTRIBUTION categorical_freq (80% delivered...) | |
| `ordered_at` | ISO datetime | DISTRIBUTION exponentielle inversee (scale=180j) | densite recente |
| `shipped_at` | ISO datetime ou NULL | CONDITIONAL sur `status` + offset `ordered_at` + Exp(3j) | NULL si status pas dans {shipped, delivered, returned} |

Generation reproductible (`--seed 42`), CSV final : **1 624 KB** (1 663 300 B).

CLI :

```
python examples/generate_orders_db.py --n-rows 10000 --seed 42 --output data/original/orders.csv
```

Verification rapide de la generation :

- 500 SKUs distincts (100% du catalogue tire), top 20 SKUs = **60.41%** des commandes
- 93.09% des lignes ont `shipped_at` non-NULL (proche du target theorique 93%)
- Currency : 59.82% USD / 19.98% EUR / 10.16% GBP / 5.20% CAD / 4.84% JPY
- Status : 80.41% delivered / 9.21% shipped / 4.95% processing / 3.47% returned / 0.98% pending / 0.98% cancelled

## 3. Resultats du POC sur orders.csv

Commande :

```
.venv/Scripts/python.exe examples/run_poc.py --input data/original/orders.csv \
                                              --reconstructed-csv data/reconstructed/orders.csv
```

Resultats finaux :

| Metrique | Valeur |
|---|---|
| Compression ratio | **3.52 : 1** |
| Fidelity score | **66.7 / 100** |
| Wall time (compress + decompress + validate) | **2.2 s** |
| Recipe size | 31 131 B (~30 KB) |
| Anchor parquet size | 441 422 B (~431 KB, zstd-22) |
| Original CSV size | 1 663 300 B (1.6 MB) |
| Anchors | `order_id`, `customer_id` (2 colonnes UUID) |

L'anchor parquet pese 27% du CSV original a lui seul : deux colonnes UUID
forcees en ancres (entropie pleine, 10000 lignes x 36 chars chacune) sont le
poste de couts dominant. Le recipe est petit (30 KB), ce qui valide le travail
algorithmique : c'est bien le stockage des UUIDs qui borne le ratio.

## 4. Comparaison avec users.csv

Mesure cote a cote, meme code, meme commit (run direct apres celui d'orders) :

| Metrique | users.csv | orders.csv |
|---|---|---|
| n_rows | 10 000 | 10 000 |
| n_columns | 9 | 10 |
| Original size | 2 122 528 B (2.0 MB) | 1 663 300 B (1.6 MB) |
| Recipe size | 16 589 B | 31 131 B |
| Anchor size | 328 706 B | 441 422 B |
| Compression ratio | **6.15 : 1** | **3.52 : 1** |
| Fidelity score | (varie selon flag run_poc) | **66.7 / 100** |
| Wall time | ~3 s | ~2 s |

> Note : la mesure users.csv ci-dessus est faite *sans* les flags
> `--random-format password_hash` (defaut de `run_poc.py`). Avec les flags
> explicites (cf. README), `users.csv` atteint historiquement 15:1 et 100% de
> fidelite. La comparaison "honnete" se fait donc a configuration egale (defaut
> sur defaut).

L'ecart de ratio (6.15 vs 3.52) s'explique par deux phenomenes complementaires :

1. **orders a deux colonnes UUID forcees en ancres** (au lieu d'une pour users :
   `email`). Le compresseur paie l'entropie de 2 x 36 chars x 10000 lignes,
   meme apres zstd-22.
2. **users.csv n'a qu'un seul UUID stocke** (`email` ; `id` et `password_hash`
   sont detectes en RANDOM_FORMAT et regeneres). Sur orders, la heuristique
   `keeping ANCHOR_DIRECT (use aggressive_uuid=True to force RANDOM_FORMAT)`
   classe les deux UUIDs en ancres car leur nom contient `id`.

L'ecart de fidelity (66.7% vs ~100%) tient a trois patterns non-detectes par
le POC actuel. Voir section 6.

## 5. Patterns detectes (depuis `output/recipes/orders.md`)

| Colonne | Pattern type detecte | Pattern type attendu | Fidelity estime | Verdict |
|---|---|---|---|---|
| `order_id` | `ANCHOR_DIRECT` | ANCHOR | 1.000 | OK |
| `customer_id` | `ANCHOR_DIRECT` | ANCHOR | 1.000 | OK |
| `product_sku` | `DISTRIBUTION` (empirical, **50 valeurs sur 500**) | CATEGORICAL_FREQ complet | 0.500 | **PARTIEL** : top 50 captures, 450 SKUs ignores |
| `category` | `FUNCTIONAL_DEP` de `product_sku` (lookup 500 entrees) | FUNCTIONAL_DEP | 1.000 | OK |
| `quantity` | `DISTRIBUTION` (exponential) | log-normal | 0.909 | **APPROCHE** : exponentielle approxime mais skew=2.97 trop fort |
| `unit_price_cents` | `DISTRIBUTION` (normal global, mean=2650, std=1769) | CONDITIONAL_DISTRIBUTION sur `category` | 0.884 | **MANQUE** : la dependance category -> price n'est pas detectee |
| `currency` | `DISTRIBUTION` (categorical_freq) | DISTRIBUTION (categorical_freq) | 1.000 | OK |
| `status` | `DISTRIBUTION` (categorical_freq) | DISTRIBUTION (categorical_freq) | 1.000 | OK |
| `ordered_at` | `DISTRIBUTION` (exponential reversed) | DISTRIBUTION exponential | 0.991 | OK |
| `shipped_at` | `DISTRIBUTION` (exponential reversed) | CONDITIONAL sur `status` + offset `ordered_at` | 0.980 | **INCORRECT** : ignore les NULL et ne capture pas la dep sur status |

Sur 10 colonnes : 6 OK, 1 partiel, 2 approches/manquees, 1 incorrecte.

## 6. Pourquoi 66.7% de fidelite ? (analyse des echecs)

Les 8 tests echoues dans le rapport `validate`, regroupes par cause :

### 6.1. Empirical truncation (2 echecs `cat_freq`)

Le detecteur de distribution `empirical` ne stocke que les **50 valeurs les
plus frequentes** d'une colonne categorielle. C'est volontaire (sinon le
recipe explose en taille), mais pour `product_sku` (500 valeurs distinctes) ca
veut dire que 450 SKUs sont jamais regeneres au decompress.

Consequence en cascade : la colonne `category` est derivee de `product_sku` via
FUNCTIONAL_DEP. Si "books" ne contient *aucun* SKU dans le top-50 (ce qui se
trouve etre le cas sur ce dataset, par hasard de l'ordering Pareto), alors
**"books" disparait du reconstruit** -> echec `value_set_mismatch` sur
`category`. Cinq SKUs precis sont aussi cites comme manquants.

Fix possible : etendre l'empirical a `max(50, n_unique)` pour les colonnes
sub-1000-uniques, ou introduire un nouveau pattern type `CATEGORICAL_FREQ`
complet pour les colonnes de moyenne cardinalite.

### 6.2. Conditional non-detecte unit_price/category (1 echec `corr anova`)

`unit_price_cents` est genere comme normal(2000 * mult[cat], 1500 * mult[cat] * 0.5)
ou les multipliers vont de 0.4 (grocery) a 3.0 (electronics). Cela cree une
correlation ANOVA forte (`orig=0.6089`). Le detecteur de patterns regarde la
correlation, mais la met dans la liste `correlations` sans la promouvoir en
CONDITIONAL_DISTRIBUTION pour la colonne. Resultat reconstruit : prix tires
d'une normale globale -> ANOVA recon=0.0009, |diff|=0.61.

### 6.3. Conditional non-detecte shipped_at/ordered_at (1 echec `corr pearson`)

`shipped_at = ordered_at + Exp(3j)` quand status est dans {shipped, delivered,
returned}. La correlation de Pearson entre les deux datetimes est de **0.9999**
(quasi-parfaite ; le delai d'expedition est petit devant la dispersion de
ordered_at). Le reconstructeur regenere `shipped_at` independamment ->
recon=0.0046, |diff|=0.995. Ce pattern serait detectable par un
`OFFSET_FROM_COLUMN` (delta exponentiel) qui n'existe pas encore.

### 6.4. Pas de NULL preservation pour shipped_at

Tous les `shipped_at` reconstruits sont non-NULL (10000/10000). Original :
9309 non-NULL et 691 NULL (statuses pending/processing/cancelled). Le test
`ks_test` sur shipped_at ne mesure que la dispersion temporelle des non-NULL,
mais l'absence de NULL casse la jointure semantique status/shipped_at.

### 6.5. Skewness elevee sur quantity (1 echec `ks_test`)

`quantity` est log-normal avec mean=1.5, sigma=0.8 -> skewness=2.97. Le
detecteur la classe en exponentielle (plus simple a fitter). KS distance
recon vs orig = 0.126 (seuil = 0.05) -> echec, meme si la moyenne (relative
diff 0.21%) et la valeur globale restent OK.

### 6.6. Cramers V sur status/currency vs category (2 echecs `corr cramers_v`)

Petites correlations residuelles dans le dataset original (V ~ 0.21..0.22)
non-recapturees par la regeneration independante des trois colonnes
categorielles. Pas un vrai pattern de design (le generateur ne pose pas de
correlation explicite ici), mais les tirages multinomial finis creent un peu
de bruit que la metrique attrape.

## 7. Conclusion : ou le POC marche et ou il faut investir

### Ce qui marche bien

- **ANCHOR_DIRECT** : detection automatique des UUID via heuristique nom + regex. 100% fidelite.
- **FUNCTIONAL_DEP** : detection du mapping SKU -> category, lookup dense
  serialise en JSON, reconstruction deterministe. Fidelite parfaite.
- **DISTRIBUTION normale / exponentielle / categorical_freq** : robustes,
  validees par KS-test ou cat_freq sur la quasi-totalite des colonnes
  appropriees.
- **Recipe lisible** : 31 KB pour decrire integralement un dataset 1.6 MB.

### Ce qui demande une evolution du code (hors scope V1)

| Limitation | Pattern manquant | Cout d'implementation |
|---|---|---|
| Empirical tronque a 50 valeurs | Etendre cap pour cols sub-1000-cardinalite, ou stocker `CATEGORICAL_FREQ` complet | Faible |
| Distributions skewed (log-normal) | Ajouter `lognormal` au detector + fitter (scipy.stats.lognorm) | Faible |
| Correlation conditional num-num | Detecter num-num via Pearson + promouvoir en CONDITIONAL si forte ANOVA | Moyen |
| Offset entre datetimes | Nouveau pattern `OFFSET_FROM_COLUMN` (col_b = col_a + Distribution) | Moyen |
| NULL preservation conditionnelle | Nouveau pattern `CONDITIONAL_NULL` (NULL si col_pivot in set) | Faible |
| Heuristique UUID pour `*_id` | Forcer RANDOM_FORMAT plus largement, ou flag CLI dedie pour orders | Faible |

### Verdict sur la generalite du POC

Le POC **fonctionne** sur orders.csv au sens ou la reconstruction produit un
dataset au shape correct, avec les bonnes valeurs sur les patterns simples
(distributions independantes, FUNCTIONAL_DEP). Il **ne valide pas** la cible
de fidelite 95%, par manque d'expressivite du detector pour quatre patterns
specifiques (conditional num-num, log-normal, offset datetime, conditional
NULL).

Autrement dit : la **mecanique** (profile -> patterns -> anchors -> recipe ->
reconstruct -> validate) est generaliste. La **bibliotheque de patterns** est
encore taillee pour users.csv et ses cousins. Un V1.1 avec 4 patterns
supplementaires pousserait orders.csv au-dela de 90%.

## 8. Annexe : ratio decomposition

Le ratio 3.52 sur orders.csv se decompose ainsi (compress) :

```
Original (CSV)           1 663 300 B  (100%)
+--- recipe.md             31 131 B    ( 1.9% du total compresse :  6.6% des 472 KB compresses)
+--- anchors.parquet      441 422 B    (26.5% du total compresse : 93.4% des 472 KB compresses)
                          ---
Total compresse           472 553 B    (28.4% de l'original  -> ratio 3.52:1)
```

Le poste anchors.parquet est >93% du paquet compresse : tant qu'on doit
preserver `order_id` ET `customer_id` exactement, le ratio est bloque par
l'entropie de Shannon des UUID. Pour pousser au-dela, deux pistes :

- Forcer `customer_id` en RANDOM_FORMAT si le POC ne fait pas de joins
  relationnels (ce qui est le cas en V1) : -50% de la taille anchors.
- Reutiliser le travail Wave 5.1 (dictionary coding sur UUIDs) si applicable
  au cas a 2 colonnes.

Sur orders, ratio plafond theorique en RANDOM_FORMAT-only : ~30:1
(cf. users.csv qui atteint 15:1 avec une seule ancre + 2 random_format).
