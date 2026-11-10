# ⚽ Penalty Shootout AI Arena

Un jeu de tirs au but interactif fondé sur la **théorie des jeux à deux
joueurs et à somme nulle** : jouer contre une IA, faire s'affronter des agents
et comprendre leurs stratégies grâce à un laboratoire d'analyse.

Projet de classe réalisé par une équipe de quatre membres. Le cahier des
charges complet est dans [`docs/`](docs/).

---

## Sommaire

1. [Présentation](#1-présentation)
2. [Le modèle mathématique](#2-le-modèle-mathématique)
3. [Les quatre agents](#3-les-quatre-agents)
4. [Installation](#4-installation)
5. [Lancer l'application](#5-lancer-lapplication)
6. [Guide d'utilisation](#6-guide-dutilisation)
7. [Benchmark et protocole expérimental](#7-benchmark-et-protocole-expérimental)
8. [Architecture du dépôt](#8-architecture-du-dépôt)
9. [Tests et qualité](#9-tests-et-qualité)
10. [Équipe et organisation](#10-équipe-et-organisation)
11. [Limites et extensions](#11-limites-et-extensions)
12. [Documentation](#12-documentation)

---

## 1. Présentation

Chaque joueur choisit **en même temps** une zone parmi *gauche*, *centre* et
*droite*. Si le gardien plonge dans la zone du tir, le ballon est **arrêté** ;
sinon c'est un **but**.

L'application propose trois usages, tous bâtis sur le même moteur :

| Usage | Description |
|-------|-------------|
| **Humain contre IA** | Vous jouez l'attaquant ou le gardien contre l'un des quatre agents, avec quatre niveaux de difficulté. |
| **IA contre IA** | Deux agents jouent automatiquement un nombre de manches choisi, avec une graine reproductible. |
| **Laboratoire** | Lancer des expériences, visualiser les stratégies, comparer les agents sur un benchmark et exporter les résultats en CSV. |

| ID | Fonctionnalité |
|----|----------------|
| F01 | Humain contre IA |
| F02 | IA contre IA |
| F03 | Quatre agents : déterministe, fréquences, Minimax, Fictitious Play |
| F04 | Quatre niveaux de difficulté |
| F05 | Laboratoire d'expériences |
| F06 | Visualisation : probabilités, fréquences, matrice, historique |
| F07 | Benchmark : comparaison des métriques et graphiques |
| F08 | Historique des manches de la session |
| F09 | Export CSV |

**Hors périmètre** : humain contre humain, agent purement aléatoire comme
adversaire, Q-learning et apprentissage profond, animation 3D, multijoueur
réseau.

---

## 2. Le modèle mathématique

Le duel est un jeu matriciel. La matrice `A` donne le gain de l'**attaquant**
(`A[i][j]`, `i` = tir, `j` = plongeon) ; celui du gardien est `-A`.

|                 | Gardien G | Gardien C | Gardien D |
|-----------------|:---------:|:---------:|:---------:|
| **Attaquant G** |    -1     |    +1     |    +1     |
| **Attaquant C** |    +1     |    -1     |    +1     |
| **Attaquant D** |    +1     |    +1     |    -1     |

- **Somme nulle par construction** : une seule matrice est stockée.
- **Équilibre** : la stratégie mixte optimale est uniforme `(1/3, 1/3, 1/3)`
  pour les deux joueurs ; la **valeur du jeu** pour l'attaquant est `v = 1/3`.
- **Calcul** : programmation linéaire (`scipy.optimize.linprog`, méthode
  HiGHS). Le même programme sert aux deux rôles (le gardien utilise `-Aᵀ`).
- **Meilleure réponse** : l'action qui maximise le gain espéré contre une
  stratégie adverse ; en cas d'égalité, l'action d'indice le plus faible est
  retenue (comportement déterministe et reproductible).

> La matrice est un modèle pédagogique, pas un taux réel de réussite des
> penalties.

### Indicateurs

| Indicateur | Définition |
|------------|------------|
| Taux de buts / d'arrêts | buts / manches, arrêts / manches (pour un gardien : buts *concédés*) |
| Gain moyen | somme des gains du rôle / manches |
| Écart à l'équilibre | variation totale `0.5 · Σ abs(pᵢ − qᵢ)`, entre 0 et 1 |
| Exploitabilité | attaquant : `v − min_j (pᵀA)_j` ; gardien : `max_i (Aq)_i − v`. Nulle pour une stratégie optimale |
| Prévisibilité (ordre 1) | précision d'un prédicteur de Markov d'ordre 1 appris en ligne sur la suite d'actions ; ≈ 1/3 pour un jeu imprévisible, 1 pour une séquence répétée |
| Temps de décision | temps moyen de `choose_action`, en millisecondes |

> **Attention** : l'exploitabilité porte sur les *fréquences globales*. Un agent
> qui joue G, C, D, G, C, D… a une exploitabilité nulle alors qu'il est
> entièrement prévisible. C'est pourquoi la prévisibilité séquentielle existe :
> lisez toujours ces indicateurs ensemble, et sur plusieurs adversaires.

Les définitions détaillées sont dans [`docs/game_theory.md`](docs/game_theory.md).

---

## 3. Les quatre agents

| Niveau | Agent | Comportement |
|--------|-------|--------------|
| 1 — Découverte | **Déterministe** | Rejoue une séquence fixe (par défaut : toujours à gauche). Prévisible, donc exploitable. |
| 2 — Adaptatif | **Fréquences** | Estime l'adversaire sur ses 10 dernières actions (fenêtre glissante, lissage de Laplace), puis joue la meilleure réponse. Réagit vite aux changements de style. |
| 3 — Stratégique | **Fictitious Play** | Compte toutes les actions adverses depuis le début (croyance initiale uniforme) et joue la meilleure réponse à ces fréquences cumulées. Ses fréquences convergent vers l'équilibre. |
| 4 — Optimal | **Minimax** | Joue la stratégie mixte optimale calculée par programmation linéaire. Inexploitable : son gain garanti est la valeur du jeu. |

> Fréquences et Fictitious Play ne diffèrent que par leur mémoire : **fenêtre**
> contre **cumul**. Contre une croyance uniforme, ils jouent *gauche* à la
> première manche (départage déterministe).

---

## 4. Installation

**Prérequis** : Python **3.11 ou plus**.

```bash
# 1. Récupérer le dépôt
git clone <https://github.com/Ledestin-gamavo/Penalty-Shootout-AI-Arena>
cd penalty_shootout_ai

# 2. (Recommandé) créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

Contenu attendu de `requirements.txt` :

```text
streamlit>=1.50
numpy
scipy
pandas
plotly
pytest
```

| Bibliothèque | Utilisation |
|--------------|-------------|
| Streamlit (≥ 1.50) | Interface : navigation, formulaires, jeu, laboratoire |
| NumPy | Matrices, distributions, générateurs aléatoires |
| SciPy | `linprog` pour la stratégie minimax, loi de Student pour les intervalles de confiance |
| pandas | Tableaux de résultats et export CSV |
| Plotly | Graphiques (barres, courbes, cartes thermiques) |

**Outils de développement** (non listés dans `requirements.txt`) :

```bash
pip install pytest mypy pandas-stubs
```

---

## 5. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur. La barre latérale permet de naviguer
entre les quatre pages et rappelle les règles du jeu.

---

## 6. Guide d'utilisation

### Page « Jouer »

- **Humain contre IA** : choisissez votre rôle (attaquant ou gardien), le niveau
  de l'adversaire (1 à 4), le nombre de manches et une graine. À chaque manche,
  cliquez sur *Gauche*, *Centre* ou *Droite* ; l'action de l'IA n'est révélée
  qu'après votre choix. Un bilan final montre votre gain et vos tendances
  face à l'équilibre `1/3`.
- **IA contre IA** : choisissez deux agents, un nombre de manches (jusqu'à
  100 000), une graine et un mode d'affichage (*accélérée* ou *détaillée*).

### Page « Laboratoire »

Configurez une expérience (agents, manches, graine) puis lancez-la. Quatre
onglets présentent le résultat :

- **Métriques** : tableau des deux agents et aide à la lecture des indicateurs ;
- **Stratégies** : fréquences jouées contre la stratégie d'équilibre ;
- **Matrice** : matrice des gains, avec les meilleures réponses entourées ;
- **Historique** : les dernières manches.

Deux boutons exportent les métriques et toutes les manches en CSV.

### Page « Analyse »

Choisissez une expérience de la session et étudiez un rôle : **évolution des
fréquences** cumulées (avec l'équilibre en pointillés) et **gain moyen
cumulé**. Un tableau compare toutes les expériences de la session (20 au
maximum).

### Page « Benchmark »

Sélectionnez les agents attaquants et gardiens, le nombre de manches et de
graines. Le benchmark joue toutes les paires, puis affiche : performance,
qualité stratégique, temps de décision, carte des confrontations, tableaux
« moyenne ± IC à 95 % » et exports CSV.

---

## 7. Benchmark et protocole expérimental

Toutes les paires attaquant/gardien sont jouées dans des **conditions
identiques** :

| Paramètre | Valeur par défaut |
|-----------|-------------------|
| Manches par match | 1 000 |
| Graines | `42 + k`, pour `k = 0 … 9` (10 graines) |
| Matrice | Matrice de référence (gains ±1) |
| Paires | Produit attaquants × gardiens (jusqu'à 4 × 4), les deux rôles |

- Chaque métrique est agrégée par moyenne, écart-type corrigé et **intervalle
  de confiance à 95 %** (loi de Student) ; les barres d'erreur des graphiques
  sont ces intervalles.
- Avec une seule graine, l'incertitude n'est pas estimable (barre nulle).
- Le benchmark par défaut (160 matchs de 1 000 manches) s'exécute en quelques
  secondes.

Le protocole complet, la définition des métriques, les limites de lecture et la
correspondance avec les tests sont dans
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

### Reproductibilité

- **Même graine + mêmes paramètres → mêmes manches.** Deux graines
  indépendantes (une par agent) sont dérivées de la graine de l'expérience avec
  `numpy.random.SeedSequence`.
- La page « Jouer » (IA contre IA) et la page « Laboratoire » appellent le même
  moteur : une même graine y donne exactement les mêmes manches.
- `created_at` est la seule donnée non reproductible d'une expérience.

### Export CSV

Les exports sont téléchargeables depuis l'interface (UTF-8 avec BOM, pour un
affichage correct des accents dans Excel). Un fichier d'échantillons contient,
par ligne : `agent`, `role`, `opponent`, `seed`, `num_rounds`, puis les
métriques (`goal_rate`, `save_rate`, `mean_reward`, `mean_decision_time`,
`equilibrium_gap`, `exploitability`, `predictability`).

---

## 8. Architecture du dépôt

```text
penalty_shootout_ai/
├── app.py                      # point d'entrée Streamlit
├── requirements.txt
├── README.md
├── docs/                       # contrat d'API, architecture, théorie, protocole
├── game/                       # moteur du jeu
│   ├── actions.py              #   enum Action
│   ├── constants.py            #   constantes, tolérances, valeurs par défaut
│   ├── models.py               #   dataclasses partagées
│   ├── payoff.py               #   matrice et lecture des gains
│   ├── rules.py                #   validation, résolution d'une manche, score
│   └── simulation.py           #   orchestration des matchs
├── theory/                     # mathématiques
│   ├── validation.py           #   validité des stratégies et matrices
│   ├── equilibrium.py          #   minimax (linprog), meilleure réponse
│   ├── metrics.py              #   indicateurs et AgentStats
│   └── benchmark_metrics.py    #   prévisibilité, résumés statistiques, fréquences cumulées
├── agents/                     # intelligences artificielles
│   ├── base_agent.py           #   classe abstraite BaseAgent
│   ├── deterministic_agent.py
│   ├── frequency_agent.py
│   ├── minimax_agent.py
│   ├── fictitious_play_agent.py
│   └── __init__.py             #   fabrique create_agent
├── analysis/                   # benchmark, tableaux et exports
│   ├── benchmark.py            #   run_benchmark, échantillons, résumés
│   └── export.py               #   historique, tableaux d'affichage, CSV
├── visualization/              # figures Plotly
│   ├── strategy_plots.py       #   stratégie, matrice, fréquences, gain
│   ├── benchmark_plots.py      #   comparaisons et confrontations
│   └── theme.py                #   couleurs et constantes visuelles
├── ui/                         # pages Streamlit
│   ├── play_page.py            #   Jouer
│   ├── laboratory_page.py      #   Laboratoire
│   ├── analysis_page.py        #   Analyse
│   ├── benchmark_page.py       #   Benchmark
│   ├── match.py                #   logique d'une partie humain contre IA
│   ├── compat.py               #   pont entre l'interface et le moteur
│   ├── lab_state.py            #   expériences conservées pour la session
│   └── state.py                #   clés de st.session_state
├── data/                       # experiments/, exports/, samples/
├── tests/                      # tests pytest
└── notebooks/                  # expérimentations
```

### Principes

1. **Un seul endroit par responsabilité.** La meilleure réponse, la validation,
   la matrice effective et les métriques existent une seule fois : on les
   appelle, on ne les recopie jamais.
2. **Les dépendances vont dans un seul sens.** Un module n'importe que des
   modules des couches situées au-dessus de lui :
   `actions → constants → models → validation → payoff → rules → equilibrium →
   metrics → agents → simulation → visualization / analysis → ui → app`.
   Le moteur ne connaît ni l'interface ni les agents ; l'interface et les
   figures ne calculent rien.
3. **Un agent ne peut pas tricher.** Il reçoit une `Observation` (`role`,
   `round_number`) qui n'a aucun champ pour l'action de l'adversaire, et ne
   découvre les actions jouées qu'après la résolution de la manche, via
   `update`.

### Flux d'une manche

1. Chaque agent choisit son action à partir de l'`Observation` seule.
2. Les actions sont validées.
3. `resolve_round` lit la matrice et fabrique le `RoundResult` (seul endroit où
   il est créé).
4. Seulement ensuite, les deux agents reçoivent le résultat via `update`.

> **Ajouter un agent** : un fichier dans `agents/`, un membre de `AgentType`,
> une branche dans `create_agent`.

---

## 9. Tests et qualité

```bash
python -m pytest                 # tous les tests
python -m mypy --strict .        # typage strict
```

La configuration pytest est à la racine :

```ini
[pytest]
pythonpath = .
testpaths = tests
```

Les tests couvrent :

- **Moteur et théorie** : règles, matrice, équilibre, métriques, validation ;
- **Agents** : conformité au contrat (signature de `choose_action`, même graine
  et même historique → mêmes actions, `reset`, ordre `choose` puis `update`,
  imports interdits, absence d'état partagé) ;
- **Benchmark** : conditions identiques, reproductibilité, résumés statistiques,
  export CSV ;
- **Visualisation** : cohérence des figures avec les données ;
- **Interface** : pages Streamlit et application complète avec
  `streamlit.testing.v1.AppTest` (partie humaine complète contre chaque niveau,
  simulation, laboratoire) ;
- **Intégration** : critères d'acceptation avec les vrais agents (N manches
  exactes, somme nulle, reproductibilité, métriques bornées, Minimax garantit la
  valeur du jeu, Fictitious Play exploite un gardien déterministe).

**Conventions de code** : Python 3.11+, annotations obligatoires,
syntaxe moderne (`X | None`, `list[int]`), docstrings en français au format
Google, identifiants en anglais. Pour mypy, ajouter `ignore_missing_imports`
pour `scipy.*` et `plotly.*`.

---

## 10. Équipe et organisation

| Membre | Responsabilité | Fichiers |
|--------|----------------|----------|
| **M1** — Moteur et théorie | Règles, matrice, équilibre, métriques, simulation, intégration | `game/*`, `theory/*` |
| **M2** — Agents IA | Quatre agents, `BaseAgent`, `create_agent` | `agents/*` |
| **M3** — Interface et jeu | Navigation, humain contre IA, IA contre IA, score, historique | `app.py`, `ui/play_page.py`, `ui/match.py`, `ui/compat.py` |
| **M4** — Visualisation et qualité | Graphiques, laboratoire, analyse, benchmark, export CSV | `visualization/*`, `analysis/*`, `theory/benchmark_metrics.py`, pages du laboratoire, de l'analyse et du benchmark |

**Règle d'or** : on ne modifie pas le module d'un autre membre sans l'accord de
son propriétaire. Chacun teste son propre module.

### Workflow Git

1. Une branche par fonctionnalité : `feature/<membre>-<sujet>`
   (ex. `feature/m2-minimax-agent`).
2. Commits au format `type: description` (`feat`, `fix`, `test`, `docs`,
   `refactor`).
3. Pull Request vers `main`, avec au moins une relecture d'un autre membre.
4. **« Terminé »** = fonctionne dans l'application intégrée, au moins un test,
   respecte les règles du jeu, passe `mypy --strict`.
5. Changer le contrat (`models.py`, `constants.py`, `actions.py` ou une
   signature) : PR avec le label `contract`, mise à jour de
   `docs/api_contract.md` dans la même PR, approbation de M1 et des membres
   impactés.

Avant chaque PR :

```bash
python -m pytest
python -m mypy --strict .
```

---

## 11. Limites et extensions

### Limites

- **Matrice ±1 uniquement.** Un but est déduit du gain de l'attaquant : ne pas
  fournir de matrice dont les gains diffèrent de ±1.
- **Matrice pédagogique** : elle ne reproduit pas les statistiques réelles des
  penalties.
- **Exploitabilité sur les fréquences globales** : elle ne voit pas l'ordre des
  coups (voir la prévisibilité séquentielle).
- **Parties humaines non persistées** : l'historique d'une partie humain contre
  IA est une suite de manches de la session, sans `Experiment`. Les expériences
  du laboratoire vivent dans la session (20 au maximum) et ne sont pas écrites
  sur disque.

### Extensions possibles

| Extension | Remarque |
|-----------|----------|
| Matrice réaliste `A[i][j] = 2p(i,j) − 1` | Impose de stocker l'issue (but ou arrêt) dans `RoundResult` au lieu de la déduire du gain |
| Variante séquentielle (feinte, puissance puis direction) | Nouveau module ; un arbre de décision devient utile |
| Export SQLite | Écriture dans `data/experiments/` depuis une couche dédiée |
| Nouveaux agents | Voir « Ajouter un agent » |

---

## 12. Documentation

| Document | Contenu |
|----------|---------|
| [`docs/architecture.md`](docs/architecture.md) | Construction du projet, dépendances, flux, sécurité contre les fuites d'information, Git |
| [`docs/api_contract.md`](docs/api_contract.md) | Signatures exactes (le contrat fait foi en cas de désaccord) |
| [`docs/game_theory.md`](docs/game_theory.md) | Modèle, équilibre, meilleure réponse, indicateurs |
| [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) | Protocole expérimental et de test du benchmark |

### Scénario de démonstration

1. Présenter les règles et la matrice de gains.
2. Afficher l'équilibre uniforme et expliquer la valeur `1/3`.
3. Jouer contre l'agent déterministe et exploiter sa prévisibilité.
4. Rejouer contre l'agent Fréquences pour illustrer l'adaptation.
5. Montrer Fictitious Play et l'évolution de ses fréquences empiriques
   (page « Analyse »).
6. Afficher Minimax et expliquer la stratégie mixte optimale.
7. Lancer un duel IA contre IA dans le laboratoire.
8. Comparer les quatre agents avec le benchmark.
9. Conclure sur les limites du modèle et les extensions possibles.
