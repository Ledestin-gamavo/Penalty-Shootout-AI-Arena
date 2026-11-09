# Contrat d'API — Penalty Shootout AI Arena

| Version | Date       | Auteur   | Changement                                        |
|---------|------------|----------|---------------------------------------------------|
| 1.0.0   | 2026-11-09 | Ledestin | Version initiale (fondations)                     |
| 1.1.0   | 2026-11-09 | Ledestin | Graphe de dépendances corrigé, `validate_config`, `resolve_payoff_matrix`, `Equilibrium.strategy_of`, `equilibrium.py` et `metrics.py` |

Ce document est la **source de vérité** pour les interfaces entre les quatre
membres. En cas de désaccord entre le code et ce document, on corrige l'un des
deux dans la même Pull Request.

---

## 1. Objet et périmètre

Le contrat définit :

- les responsabilités de chaque module ;
- les structures de données partagées (`game/models.py`) ;
- les conventions de nommage et de typage ;
- les signatures principales prévues ;
- les règles de communication entre membres ;
- la politique anti-fuite d'information entre agents.

Il ne modifie pas l'architecture du cahier des charges. Il la précise.

---

## 2. Principes d'architecture

1. **Dépendances à sens unique** : un module n'importe que des modules des
   couches situées au-dessus de lui.

```text
1. game/actions.py
2. game/constants.py        (importe : actions)
3. game/models.py           (importe : actions, constants)
4. theory/validation.py     (importe : constants, models)
5. game/payoff.py           (importe : actions, constants, models)
6. game/rules.py            (importe : actions, models, payoff, theory.validation)
7. theory/equilibrium.py    (importe : actions, constants, models, theory.validation)
8. theory/metrics.py        (importe : actions, models, game.rules,
                                       theory.equilibrium, theory.validation)
9. agents/*                 (importent : actions, constants, models,
                                         theory.equilibrium, theory.validation)
10. game/simulation.py      (importe : agents, game.*, theory.metrics)
11. visualization/*         (importent : modèles et résultats)
12. ui/*                    (importent : simulation, agents, visualization)
13. app.py
```

2. **Responsabilité unique** : un module = une raison de changer.
3. **Aucune logique dupliquée** : meilleure réponse, validation, résolution de
   la matrice effective et métriques n'existent qu'à un seul endroit (voir §3).
4. **Données immuables** : toutes les dataclasses partagées sont
   `frozen=True, slots=True`.
5. **Aucun état global** : pas de variable de module mutable, pas de
   générateur aléatoire global, pas de cache global.

---

## 3. Responsabilités des modules

| Module | Responsable | Responsabilité | Ne doit PAS |
|--------|-------------|----------------|-------------|
| `game/actions.py` | M1 | Enum `Action` | Contenir de la logique de jeu |
| `game/constants.py` | M1 | Constantes métier, tolérances, valeurs par défaut | Importer autre chose que `Action` |
| `game/models.py` | M1 | Structures de données partagées | Contenir de la logique métier (hors invariants et propriétés dérivées) |
| `theory/validation.py` | M1 | Vérifie stratégies et matrices (taille, valeurs finies, positivité, somme = 1) | Importer `game.rules`, `game.payoff`, un agent, `game.simulation`, `ui` ou `visualization` ; corriger silencieusement les données |
| `game/payoff.py` | M1 | Matrice de référence, résolution de la matrice effective, lecture d'un gain | Valider les actions ou la matrice |
| `game/rules.py` | M1 | Validation des actions et des configurations, résolution d'une manche, score. Peut importer `theory.validation`. | Connaître les agents |
| `theory/equilibrium.py` | M1 | Stratégie minimax (programmation linéaire), valeur du jeu, **meilleure réponse** | Dépendre des agents ou de `game.rules` |
| `theory/metrics.py` | M1 (formules) ; M4 peut ajouter des métriques de benchmark | Stratégie empirique, taux de buts/arrêts, gain moyen, distance à l'équilibre, exploitabilité, `AgentStats` | Faire des appels à l'UI |
| `agents/base_agent.py` | M2 | Classe abstraite `BaseAgent` | Connaître l'adversaire |
| `agents/*_agent.py` | M2 | Un agent par fichier | Importer `game.simulation`, `ui`, `visualization` |
| `agents/__init__.py` | M2 | Fabrique `create_agent` (seul point de création) | Contenir de la logique d'agent |
| `game/simulation.py` | M1 | Orchestration d'un match IA vs IA et d'une manche humain vs IA | Calculer des métriques ou des graphiques |
| `ui/*` | M3 | Pages Streamlit, saisie, affichage | Calculer des métriques ou des équilibres |
| `visualization/*` | M4 | Figures Plotly à partir des structures partagées | Recalculer des résultats |
| `tests/*` | Propriétaire du module testé | Tests unitaires et d'intégration | — |

**Règle de réutilisation** : toute nouvelle logique commune est placée dans le
module de la table ci-dessus qui en a la responsabilité, jamais copiée.
Exemples : la meilleure réponse est dans `theory/equilibrium.py`, la
validation dans `theory/validation.py`, la matrice effective dans
`game.payoff.resolve_payoff_matrix`, l'affichage d'une action est
`Action.label`, la stratégie d'un rôle est `Equilibrium.strategy_of`.

**Règles d'import de `theory/validation.py`** : il ne doit jamais importer
`game.rules`, jamais importer `game.payoff`, jamais importer un agent. Il ne
dépend que de `game.constants` et `game.models`.

---

## 4. Structures de données partagées

Toutes sont définies dans `game/models.py` (sauf `Action`, dans
`game/actions.py`).

| Nom | Type | Rôle |
|-----|------|------|
| `Action` | `IntEnum` | `LEFT=0`, `CENTER=1`, `RIGHT=2` ; sert aussi d'indice |
| `Role` | `StrEnum` | `ATTACKER`, `GOALKEEPER` ; propriétés `opponent`, `label` |
| `AgentType` | `StrEnum` | `DETERMINISTIC`, `FREQUENCY`, `MINIMAX`, `FICTITIOUS_PLAY` |
| `Strategy` | alias `tuple[float, ...]` | Distribution sur les 3 actions, indexée par `Action` |
| `PayoffMatrix` | alias `tuple[tuple[float, ...], ...]` | Gains de l'attaquant ; `m[attaquant][gardien]` |
| `Observation` | dataclass | Information **autorisée** donnée à un agent : `role`, `round_number` |
| `RoundResult` | dataclass | Résultat d'une manche révélée ; `action_of(role)`, `reward_for(role)` |
| `MatchConfig` | dataclass | Paramètres d'un match ou d'une expérience |
| `AgentStats` | dataclass | Statistiques agrégées d'un agent sur un match |
| `Experiment` | dataclass | Match IA vs IA complet, reproductible |
| `Equilibrium` | dataclass | Stratégies optimales et valeur du jeu ; `strategy_of(role)` |

### Conventions sur les données

- **Point de vue des gains** : la matrice et `attacker_reward` sont exprimés
  du point de vue de l'attaquant. `goalkeeper_reward = -attacker_reward`.
- **Orientation de la matrice** : lignes = action de l'attaquant,
  colonnes = action du gardien.
- **Numéro de manche** : commence à **1**.
- **Temps** : en **secondes** (`float`).
- **Date** : `datetime` en UTC, avec fuseau horaire.
- **Probabilités** : positives, somme à 1 à `PROBABILITY_TOLERANCE` près.
- **`AgentStats.goals` / `saves`** : buts marqués / arrêts réalisés lors des
  manches jouées par l'agent, quel que soit son rôle (pour un gardien,
  `goals` = buts concédés).
- **Humain vs IA** : produit une liste de `RoundResult` (historique de session,
  F08) mais **pas** d'`Experiment`. Seuls les matchs IA vs IA en produisent.
- **`MatchConfig.payoff_matrix = None`** signifie « matrice de référence ».
  Les consommateurs obtiennent la matrice effective avec
  `game.payoff.resolve_payoff_matrix`, sans réécrire cette règle.
- **Agent à `None` dans `MatchConfig`** : désigne le joueur humain. Les deux
  à `None` sont interdits (pas de mode humain contre humain).
- **Reproductibilité des `Experiment`** : `created_at` est la seule donnée non
  reproductible. Pour comparer deux expériences de même graine, comparer
  `rounds` (et non l'objet entier).

---

## 5. Conventions de nommage

| Élément | Convention | Exemple |
|---------|------------|---------|
| Fichiers, modules, fonctions, variables | `snake_case` | `fictitious_play_agent.py` |
| Classes, Enums, alias de types | `PascalCase` | `RoundResult`, `Strategy` |
| Membres d'Enum et constantes | `UPPER_SNAKE_CASE` | `Action.LEFT`, `REWARD_GOAL` |
| Éléments privés | préfixe `_` | `_ACTION_LABELS` |
| Booléens | préfixe `is_`, `has_` | `is_goal` |
| Fonctions | verbe d'action ou nom de métrique | `resolve_round`, `solve_equilibrium`, `goal_rate` |
| Branches Git | `feature/<membre>-<sujet>` | `feature/m2-minimax-agent` |
| Commits | `type: description` (`feat`, `fix`, `test`, `docs`, `refactor`) | `feat: add frequency agent` |

- **Identifiants de code en anglais.**
- **Docstrings, commentaires et textes d'interface en français.**
- Les mots-clés de sections Google Style restent en anglais
  (`Args`, `Returns`, `Raises`, `Attributes`).
- Pas de nombre magique : toute valeur métier est dans `game/constants.py`.

---

## 6. Conventions de typage

- Python **3.11+**, vérification par `mypy --strict` (outil de développement
  local, non listé dans `requirements.txt`). SciPy n'étant pas typé de façon
  complète, ajouter `ignore_missing_imports` pour le module `scipy.*` dans la
  configuration de mypy.
- Annotations obligatoires sur tous les paramètres, retours et attributs
  de classe.
- Syntaxe moderne : `X | None`, `list[int]`, `tuple[float, ...]`.
  Pas de `Optional`, pas de `List` importé de `typing`.
- `from __future__ import annotations` uniquement si une référence avant
  définition est nécessaire.
- Pas de `Any` sans commentaire justificatif.
- Tableaux NumPy : `NDArray[np.float64]` (`numpy.typing`). Ils restent
  **internes** aux modules ; les interfaces publiques échangent `Strategy`
  et `PayoffMatrix`.
- Paramètres de lecture seule : `collections.abc.Sequence`.
- Constantes : `Final`.
- Dataclasses : `frozen=True, slots=True`.
- Style : **PEP 8**, lignes de 88 caractères maximum, imports triés
  (stdlib, tiers, projet), imports **absolus** depuis la racine
  (`from game.models import Role`).
- Docstrings **Google Style** complètes pour tout module public, classe,
  énumération, dataclass et fonction.
- Erreurs : `ValueError` pour toute valeur invalide (actions, stratégies,
  matrices, configurations). `RuntimeError` uniquement pour l'échec d'un
  solveur numérique. Pas d'`assert` pour valider des entrées, pas de `print`,
  pas d'erreur avalée silencieusement.

---

## 7. Signatures principales

Les signatures sont **gelées à l'issue du jour 2** (voir §9). Les noms de
paramètres font partie du contrat.

### 7.1 `game/payoff.py` (M1)

```python
def build_reference_matrix() -> PayoffMatrix:
    """Construit la matrice de référence 3x3 à partir des récompenses."""

def resolve_payoff_matrix(payoff_matrix: PayoffMatrix | None) -> PayoffMatrix:
    """Retourne la matrice effective : la matrice fournie, ou la matrice de
    référence si l'argument est None."""

def attacker_reward(
    attacker_action: Action,
    goalkeeper_action: Action,
    payoff_matrix: PayoffMatrix,
) -> float:
    """Retourne le gain de l'attaquant pour un couple d'actions."""
```

### 7.2 `game/rules.py` (M1)

```python
def validate_action(value: object) -> Action:
    """Convertit une valeur en Action valide ; lève ValueError sinon."""

def validate_config(config: MatchConfig) -> None:
    """Vérifie les règles métier d'une configuration (matrice personnalisée) ;
    lève ValueError sinon. Peut s'appuyer sur theory.validation."""

def resolve_round(
    round_number: int,
    attacker_action: Action,
    goalkeeper_action: Action,
    payoff_matrix: PayoffMatrix,
) -> RoundResult:
    """Résout une manche à partir des deux actions révélées."""

def compute_score(results: Sequence[RoundResult]) -> tuple[int, int]:
    """Retourne (buts, arrêts) sur une liste de manches."""
```

### 7.3 `game/simulation.py` (M1)

```python
def run_match(config: MatchConfig) -> Experiment:
    """Exécute un match IA vs IA complet et retourne l'expérience."""

def play_human_round(
    agent: BaseAgent,
    human_role: Role,
    human_action: Action,
    round_number: int,
    payoff_matrix: PayoffMatrix,
) -> RoundResult:
    """Joue une manche humain vs IA : l'agent décide, la manche est
    résolue, puis l'agent est mis à jour."""
```

L'interface valide la matrice (`validate_config` ou `validate_payoff_matrix`)
à la création d'une partie : `play_human_round` ne la revalide pas à chaque
manche.

### 7.4 `theory/equilibrium.py` (M1)

```python
def solve_equilibrium(payoff_matrix: PayoffMatrix) -> Equilibrium:
    """Calcule les stratégies minimax et la valeur du jeu (SciPy linprog).

    Les stratégies retournées sont nettoyées (valeurs comprises entre
    -NUMERICAL_NOISE_TOLERANCE et 0 ramenées à 0), renormalisées, puis
    validées par validate_strategy. Lève ValueError si la matrice est
    invalide, RuntimeError si le solveur échoue."""

def best_response(
    payoff_matrix: PayoffMatrix,
    role: Role,
    opponent_strategy: Strategy,
) -> Action:
    """Meilleure réponse pure à une stratégie adverse.

    Égalité (à NUMERICAL_NOISE_TOLERANCE près) : l'action d'indice le plus
    faible est retenue (comportement déterministe et reproductible)."""
```

### 7.5 `theory/metrics.py` (M1)

```python
def empirical_strategy(actions: Sequence[Action]) -> Strategy: ...

def goal_rate(rounds: Sequence[RoundResult]) -> float:
    """Buts / manches."""

def save_rate(rounds: Sequence[RoundResult]) -> float:
    """Arrêts / manches."""

def average_reward(rounds: Sequence[RoundResult], role: Role) -> float:
    """Gain moyen par manche du point de vue du rôle donné."""

def distance_to_equilibrium(strategy: Strategy, reference: Strategy) -> float:
    """Distance en variation totale : 0.5 * somme(|p_i - q_i|)."""

def exploitability(
    payoff_matrix: PayoffMatrix, role: Role, strategy: Strategy
) -> float:
    """Convention (toujours >= 0, nulle à l'équilibre) :
    attaquant -> v - min_j (p^T A)_j ;
    gardien   -> max_i (A q)_i - v."""

def compute_agent_stats(
    agent_type: AgentType,
    role: Role,
    rounds: Sequence[RoundResult],
    mean_decision_time: float,
    payoff_matrix: PayoffMatrix,
) -> AgentStats: ...
```

`AgentStats.equilibrium_gap` est la valeur de `distance_to_equilibrium`
entre la stratégie empirique et `Equilibrium.strategy_of(role)`.
`AgentStats.goal_rate` / `save_rate` (propriétés) lisent les compteurs d'un
`AgentStats` déjà agrégé ; `metrics.goal_rate` / `save_rate` travaillent sur
une liste brute de `RoundResult`.

### 7.6 `theory/validation.py` (M1)

```python
def validate_strategy(strategy: Strategy) -> None:
    """Lève ValueError si la stratégie n'est pas une distribution valide."""

def validate_payoff_matrix(payoff_matrix: PayoffMatrix) -> None:
    """Lève ValueError si la matrice n'est pas N x N finie."""
```

### 7.7 `agents/` (M2)

```python
class BaseAgent(ABC):
    def __init__(self, payoff_matrix: PayoffMatrix, seed: int) -> None: ...

    @abstractmethod
    def choose_action(self, observation: Observation) -> Action:
        """Choisit une action. Reçoit UNIQUEMENT l'observation autorisée."""

    def update(
        self,
        observation: Observation,
        own_action: Action,
        opponent_action: Action,
        reward: float,
    ) -> None:
        """Mise à jour après révélation (par défaut : ne fait rien)."""

    def reset(self) -> None:
        """Réinitialise l'état interne ET le générateur aléatoire
        (graine d'origine)."""
```

```python
# agents/__init__.py
def create_agent(
    agent_type: AgentType, payoff_matrix: PayoffMatrix, seed: int
) -> BaseAgent:
    """Fabrique unique d'agents."""
```

- Un agent est créé **pour un rôle précis**, fourni via `Observation.role`.
- L'aléa passe exclusivement par `numpy.random.default_rng(seed)`, conservé
  dans l'instance. Minimax tire son action selon sa stratégie mixte. Ce
  n'est pas un agent aléatoire (cf. cahier des charges, §2).
- Les agents Fréquences et Fictitious Play utilisent **obligatoirement**
  `theory.equilibrium.best_response`. Minimax utilise
  `solve_equilibrium(...).strategy_of(role)`.

### 7.8 `ui/` et `visualization/` (M3 / M4)

```python
# visualization : une fonction = une figure, aucune logique métier
def plot_strategy_bars(...) -> go.Figure: ...
def plot_payoff_heatmap(payoff_matrix: PayoffMatrix) -> go.Figure: ...
def plot_frequency_evolution(rounds: Sequence[RoundResult], role: Role) -> go.Figure: ...

# ui : chaque page expose une seule fonction de rendu
def render() -> None: ...
```

---

## 8. Politique anti-fuite d'information

**Principe : un agent ne peut jamais connaître l'action courante de
l'adversaire avant de décider.** La protection repose sur la structure des
données, pas sur la discipline des développeurs.

1. **Entrée restreinte.** `choose_action` reçoit uniquement une `Observation`
   (`role`, `round_number`). Cette structure ne possède aucun champ pour
   l'action courante de l'adversaire.
2. **L'historique transite par `update()`.** L'agent apprend les actions
   adverses **après** révélation des deux actions, et seulement via
   `update()`. Il construit lui-même ses compteurs.
3. **Ordre imposé dans `game/simulation.py`** : `choose_action` des deux
   agents, validation, `resolve_round`, puis `update` des deux agents.
   Aucun `update` n'a lieu avant que les deux actions soient fixées.
4. **Isolation des agents.** Un agent ne reçoit aucune référence vers
   l'adversaire, la simulation, la configuration du match, l'interface ou
   l'`Experiment`. Il ne connaît ni le type de l'adversaire ni la graine de
   l'adversaire.
5. **Pas d'état partagé.** Pas de variable de module mutable, pas de singleton,
   pas de cache global. Chaque agent possède son propre générateur
   `numpy.random.Generator`.
6. **Immuabilité.** `Observation` et `RoundResult` sont figés : un agent ne
   peut pas modifier ce qu'il reçoit.
7. **Information publique autorisée** : la matrice des gains (fournie au
   constructeur), son propre rôle, le numéro de manche et le résultat des
   manches terminées.
8. **Interface humain vs IA.** `play_human_round` valide l'action humaine,
   puis appelle `choose_action` avec une `Observation` seule : l'action
   humaine n'est jamais transmise à l'agent avant résolution. L'interface
   n'affiche l'action de l'IA qu'**après** validation du choix humain.
9. **Imports interdits** dans `agents/` : `game.simulation`, `ui`,
   `visualization`, `streamlit`.
10. **Tests obligatoires** (`tests/test_agents.py`, `tests/test_simulation.py`) :
    - même graine et même historique → même action, quelle que soit l'action
      que l'adversaire jouera ensuite ;
    - `update` n'est appelé qu'après résolution de la manche ;
    - `reset` restaure l'état initial et la séquence aléatoire ;
    - la signature de `choose_action` n'accepte que `Observation`.

---

## 9. Règles de communication entre membres

### 9.1 Interfaces

| De | Vers | Fournit |
|----|------|---------|
| M1 | M2 | `Action`, `Observation`, matrice, `best_response`, `solve_equilibrium`, `Equilibrium.strategy_of` |
| M2 | M3 | `BaseAgent` et `create_agent` stables |
| M1, M2 | M4 | `RoundResult`, `AgentStats`, `Experiment`, fonctions de `theory/metrics.py` |
| M3 ↔ M4 | — | Composants communs du laboratoire et du benchmark |

### 9.2 Évolution du contrat

- Toute modification de `game/models.py`, `game/constants.py`,
  `game/actions.py` ou des signatures ci-dessus passe par une PR **avec le
  label `contract`**, qui met aussi à jour ce document.
- Une PR `contract` doit être approuvée par **M1 (intégration)** et par les
  membres dont les modules sont impactés.
- **Changement non rétrocompatible** (renommage, suppression, nouveau champ
  obligatoire) : annonce préalable au groupe, incrément de version
  majeure/mineure dans le tableau en tête de ce document, correction des
  consommateurs dans la même PR.
- Ajouter un champ avec valeur par défaut est rétrocompatible.
- **Gel des signatures après le jour 2.** Ensuite, seules les corrections de
  bogue et les changements validés par M1.
- Aucun membre ne modifie le module d'un autre sans l'accord de son
  propriétaire.

### 9.3 Git et revue

- Une branche par fonctionnalité, fusion vers `main` par Pull Request.
- Au moins **une relecture** par un autre membre avant fusion.
- Intégration quotidienne ou à chaque jalon. M1 est responsable de
  l'intégration et des conflits.
- Une fonctionnalité est terminée si : elle fonctionne dans l'application
  intégrée, a au moins un test, respecte les règles et passe `mypy --strict`.
- Les décisions techniques sont consignées dans `docs/meeting_notes.md`.

---

## 10. Reproductibilité

- Toute expérience est identifiée par `MatchConfig.seed` (entier ≥ 0,
  défaut `DEFAULT_SEED`).
- `simulation.run_match` dérive **deux graines indépendantes** (attaquant,
  gardien) de `config.seed` avec `numpy.random.SeedSequence`.
- Benchmark : plusieurs graines `DEFAULT_SEED + k`, avec le même nombre de
  manches et les mêmes adversaires pour tous les agents.
- Les paramètres et graines sont enregistrés dans `Experiment.config`.
- `Experiment.created_at` n'est pas reproductible : les tests comparent
  `Experiment.rounds`.

---

## 11. Tests de conformité au contrat

Chaque propriétaire teste son module. Tests transversaux :

- `Action` contient exactement 3 membres, indices 0, 1, 2.
- Les neuf cases de la matrice de référence donnent le résultat attendu.
- `attacker_reward + goalkeeper_reward == 0` pour toute manche.
- Toute `Strategy` retournée est positive et de somme 1.
- `solve_equilibrium` retrouve la valeur `REFERENCE_GAME_VALUE` et la
  stratégie `REFERENCE_EQUILIBRIUM_STRATEGY` à `EQUILIBRIUM_TOLERANCE` près,
  pour l'attaquant comme pour le gardien.
- `best_response` contre une stratégie uniforme retourne `Action.LEFT`
  (règle de départage) ; contre « toujours gauche », l'attaquant répond
  `Action.CENTER` et le gardien `Action.LEFT`.
- `exploitability` vaut 0 à `EQUILIBRIUM_TOLERANCE` près pour la stratégie
  d'équilibre et est strictement positive pour une stratégie pure.
- `distance_to_equilibrium` vaut 0 pour deux stratégies identiques et 1 pour
  deux stratégies pures différentes.
- Un match IA vs IA termine exactement `num_rounds` manches.
- Les tests d'anti-fuite du §8.

---

## 12. État d'avancement et dépendances à terme

`game/simulation.py` dépend de deux modules :

| Dépendance | Responsable | État (v1.1.0) |
|------------|-------------|---------------|
| `theory.metrics` (`compute_agent_stats`) | M1 | Disponible |
| `agents` (`create_agent`, `BaseAgent`) | M2 | **Pas encore disponible** |

Tant que `agents/` n'existe pas, `game/simulation.py` ne peut pas être importé
et ses tests doivent attendre la livraison du Membre 2 (ou utiliser un agent
de test local, hors dépôt). Les autres modules `game/` et `theory/` n'ont
aucune dépendance vers `agents/` et sont testables dès maintenant.