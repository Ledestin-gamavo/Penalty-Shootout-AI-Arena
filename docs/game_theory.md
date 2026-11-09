# Théorie des jeux — Penalty Shootout AI Arena

Ce document décrit les notions mathématiques utilisées par le projet. Les
conventions de code correspondantes sont dans `docs/api_contract.md`.

---

## 1. Modèle

Un duel de tirs au but est un jeu matriciel à deux joueurs, à somme nulle et à
décisions simultanées. Chaque joueur choisit une zone parmi gauche, centre et
droite. La matrice `A` donne le gain de l'attaquant ; celui du gardien est
`-A`.

|                  | Gardien G | Gardien C | Gardien D |
|------------------|-----------|-----------|-----------|
| **Attaquant G**  | -1        | +1        | +1        |
| **Attaquant C**  | +1        | -1        | +1        |
| **Attaquant D**  | +1        | +1        | -1        |

- Convention : `A[i][j]`, `i` = action de l'attaquant, `j` = action du gardien.
- La somme nulle est garantie par construction : seule `A` est stockée.
- Cette matrice est un modèle pédagogique, pas un taux réel de réussite des
  penalties.

## 2. Stratégies et valeur du jeu

Une stratégie pure choisit une action. Une stratégie mixte est une distribution
de probabilités. Pour la matrice de référence, l'équilibre est uniforme
(1/3, 1/3, 1/3) pour les deux joueurs et la valeur du jeu pour l'attaquant est
`v = 1/3`. Cette valeur ne vaut que pour la matrice de référence.

## 3. Calcul de l'équilibre (minimax par programmation linéaire)

Pour un rôle qui maximise sa matrice `M` (lignes = ses actions, colonnes = les
actions adverses) :

```text
maximiser    v
sous         sum_i p_i * M[i][k] >= v    pour toute colonne k
             sum_i p_i = 1,  p_i >= 0,  v libre
```

- Attaquant : `M = A`.
- Gardien : `M = -A^T` (il maximise l'opposé du gain de l'attaquant). Le même
  programme est donc utilisé pour les deux rôles.
- Par dualité, la valeur obtenue par le gardien vaut `-v`.
- Le solveur est `scipy.optimize.linprog` (méthode HiGHS).
- Après résolution, toute valeur comprise entre `-NUMERICAL_NOISE_TOLERANCE` et
  0 est ramenée à 0, puis la stratégie est renormalisée et validée.

## 4. Meilleure réponse

Pour une stratégie adverse donnée, la meilleure réponse est l'action pure qui
maximise le gain espéré du rôle. En cas d'égalité (à
`NUMERICAL_NOISE_TOLERANCE` près), l'action d'indice le plus faible est
retenue : le comportement est déterministe et reproductible.

Conséquence : contre une stratégie uniforme, toutes les actions sont à égalité
et la réponse est toujours `LEFT`. Un agent Fictitious Play est donc un peu
prévisible lors des premières manches, tant que sa croyance reste proche de
l'uniforme.

## 5. Indicateurs

| Indicateur | Définition |
|------------|------------|
| Taux de buts | buts / manches |
| Taux d'arrêts | arrêts / manches |
| Gain moyen | somme des gains du rôle / manches |
| Écart à l'équilibre | variation totale : `0.5 * sum_i abs(p_i - q_i)`, entre 0 et 1 |
| Exploitabilité | attaquant : `v - min_j (p^T A)_j` ; gardien : `max_i (A q)_i - v` |

L'exploitabilité est toujours positive ou nulle et vaut 0 pour une stratégie
optimale. Elle est calculée sur la **stratégie empirique** de l'agent, c'est-à-dire
ses fréquences sur l'ensemble du match.

Limite : un agent dont les fréquences globales sont uniformes mais dont la
séquence est prévisible (par exemple G, C, D, G, C, D…) obtient une
exploitabilité nulle. Cet indicateur ne mesure pas la prévisibilité
séquentielle. Il doit toujours être lu avec le gain moyen contre plusieurs
adversaires.

## 6. Fréquences et Fictitious Play

- **Agent fréquences** : fenêtre glissante sur les dernières manches, avec
  lissage, puis meilleure réponse.
- **Fictitious Play** : comptage cumulatif depuis le début, croyance initiale
  uniforme, puis meilleure réponse.

Les deux utilisent `best_response`. Leur différence (fenêtre contre cumul) doit
rester codée et testée.

## 7. ATTENTION — hypothèse ±1 du moteur actuel

La propriété `RoundResult.is_goal` repose sur les gains ±1 : elle compare le
gain de l'attaquant à `REWARD_GOAL`.

Si une matrice réaliste est utilisée plus tard, avec
`A[i][j] = 2 * p(i, j) - 1` (où `p(i, j)` est la probabilité de but), les gains
ne valent plus ±1 et un but n'est plus déterminé par le gain. Il faudra alors
repenser :

- `RoundResult.is_goal` ;
- `AgentStats.goal_rate` et `theory.metrics.goal_rate` / `save_rate` ;
- `game.rules.compute_score`.

Une piste est de stocker explicitement dans `RoundResult` l'issue de la manche
(but ou arrêt), tirée selon `p(i, j)`, au lieu de la déduire du gain. Tant que
cette extension n'est pas implémentée, **ne pas fournir de matrice
personnalisée dont les gains diffèrent de ±1**.