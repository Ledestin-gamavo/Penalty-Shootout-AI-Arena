# Protocole expérimental et de test — Benchmark (Membre 4)

Ce document décrit comment le benchmark compare les agents, comment lire les
indicateurs et comment valider la chaîne de bout en bout. Les définitions
mathématiques de base sont dans `docs/game_theory.md` ; ici figurent celles
ajoutées par `theory/benchmark_metrics.py`.

## 1. Conditions de comparaison

| Paramètre | Valeur |
|-----------|--------|
| Paires jouées | produit attaquants x gardiens (jusqu'à 4 x 4), les deux rôles pour chaque agent |
| Manches par match | `DEFAULT_NUM_ROUNDS_SIMULATION` (1 000), identique pour toutes les paires |
| Graines | `base_seed + k`, `k = 0 .. n-1` (`DEFAULT_SEED = 42`, `DEFAULT_BENCHMARK_NUM_SEEDS = 10`) |
| Matrice | matrice de référence (gains ±1) ; aucune matrice personnalisée |
| Moteur | `game.simulation.run_match`, une expérience par (paire, graine) |

Toutes les paires reçoivent **les mêmes graines**, le même nombre de manches et
la même matrice : seules les paires d'agents diffèrent.

## 2. Indicateurs

Chaque ligne d'échantillon correspond à un agent dans une expérience.

| Colonne | Définition |
|---------|------------|
| `goal_rate` | buts / manches de l'agent (pour un gardien : buts **concédés**) |
| `save_rate` | arrêts / manches de l'agent (complément de `goal_rate`) |
| `mean_reward` | gain moyen par manche du point de vue de l'agent |
| `mean_decision_time` | temps moyen de `choose_action`, en secondes (affiché en ms) |
| `equilibrium_gap` | variation totale entre stratégie empirique et équilibre : `0.5 * sum(abs(p_i - q_i))` |
| `exploitability` | attaquant : `v - min_j (p^T A)_j` ; gardien : `max_i (A q)_i - v` (≥ 0, nulle à l'équilibre) |
| `predictability` | précision d'un prédicteur d'ordre 1 appris **en ligne** sur la suite d'actions de l'agent |

### Prévisibilité séquentielle (`predictability`)

L'exploitabilité porte sur les fréquences globales : une séquence G, C, D, G,
C, D... a une exploitabilité nulle alors qu'elle est entièrement prévisible.
La prévisibilité comble ce trou. Un prédicteur compte, pour chaque action, ce
qui l'a suivie, et prédit la suite la plus fréquente (égalité : indice le plus
faible). La métrique est la part des actions, à partir de la deuxième, qu'il a
devinées avant de les voir.

- 1 : séquence parfaitement prévisible (par exemple « toujours gauche ») ;
- environ 1/3 : suite uniforme et indépendante (Minimax à l'équilibre) ;
- elle est calculée sur une seule graine ; elle est NaN pour un match d'une
  seule manche.

## 3. Statistiques

- Chaque métrique est résumée par graine puis agrégée : moyenne `_mean`,
  écart-type corrigé `_std` (`ddof = 1`) et demi-largeur `_ci95` de
  l'intervalle de confiance à 95 % (loi de Student, `n - 1` degrés de liberté).
- Avec une seule graine, `_std` et `_ci95` valent 0 : l'incertitude n'est pas
  estimable.
- Les barres d'erreur des graphiques sont les `_ci95`.
- L'agrégation « tous adversaires » (`by_opponent=False`) met en commun les
  échantillons de toutes les paires ; elle suppose le même nombre de graines par
  adversaire, garanti par `run_benchmark`. Les intervalles correspondants
  supposent des mesures indépendantes, ce qui est approximatif.

## 4. Lecture et limites

- Le gain moyen seul ne mesure pas la qualité stratégique : un agent déterministe
  peut gagner contre un adversaire faible. Lire `mean_reward` avec
  `exploitability` et `predictability`, et sur plusieurs adversaires.
- Un agent exploitable peut obtenir un gain élevé contre un adversaire donné ;
  seule sa mesure d'exploitabilité indique ce qu'un adversaire optimal pourrait
  lui prendre.
- `goal_rate` d'un gardien est un taux de buts concédés : ne pas le comparer
  directement à celui d'un attaquant.
- La matrice est un modèle pédagogique (valeur du jeu 1/3), pas un taux réel
  de réussite des penalties.

## 5. Export CSV (F09, EF11)

`analysis.export.export_csv` écrit dans `data/exports/` (UTF-8 avec BOM pour
Excel). Un fichier d'échantillons contient, par ligne : `agent`, `role`,
`opponent`, les paramètres `seed` et `num_rounds`, puis les sept métriques ci-dessus.

## 6. Protocole de test

| Critère (cahier des charges §10.2 / contrat §11) | Test |
|--------------------------------------------------|------|
| Un match IA contre IA termine exactement N manches | `test_integration.py::test_every_match_plays_exactly_the_requested_rounds` |
| Gain attaquant + gain gardien = 0 à chaque manche | `test_integration.py::test_every_round_is_zero_sum` |
| Reproductibilité par graine | `test_integration.py::test_same_seed_gives_same_rounds`, `test_benchmark.py::test_run_benchmark_is_reproducible` |
| Le benchmark compare dans des conditions identiques | `test_integration.py::test_benchmark_conditions_are_identical_for_all_pairs`, `test_benchmark.py::test_run_benchmark_uses_identical_conditions_for_all_pairs` |
| Distributions positives de somme 1 | `test_integration.py::test_empirical_strategies_are_valid_distributions`, `test_visualization.py` (barres, évolution des fréquences) |
| Métriques bornées | `test_integration.py::test_metrics_are_within_their_bounds` |
| Un agent exploitable est bien détecté | `test_integration.py::test_deterministic_goalkeeper_is_exploited_by_fictitious_play` |
| Minimax garantit la valeur du jeu | `test_integration.py::test_minimax_attacker_secures_the_game_value` |
| Exploitabilité aveugle à la séquence | `test_benchmark_metrics.py::test_predictability_detects_cycle_that_exploitability_misses` |
| Export CSV : paramètres et métriques, accents, nom sûr | `test_export.py` |
| Pages : formulaire appliqué (agents, manches, graine), résultats, export, erreurs | `test_pages.py` (Streamlit `AppTest`) |

Les tests de `test_integration.py` sont ignorés tant que `agents/` et
`game.simulation` ne sont pas disponibles. Les autres s'exécutent avec le faux
moteur `tests/fakes.py`.

```bash
python -m pytest
python -m mypy --strict analysis visualization theory/benchmark_metrics.py tests
```

Configuration mypy : `ignore_missing_imports` pour `scipy.*` et `plotly.*`
(`pandas-stubs` est recommandé en développement).
