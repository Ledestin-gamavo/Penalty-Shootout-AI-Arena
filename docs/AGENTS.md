# Module agents : guide pour l'équipe

Ce dossier contient les 4 agents du cahier des charges et le contrat commun `BaseAgent`.
Lancer les tests depuis la racine du projet : `python -m pytest tests -v` (33 tests).
Lancer l'exemple : `python demo_agents.py`.

## 1. Décisions de contrat (à valider avec les membres 1, 3 et 4)

Le cahier des charges laisse trois points ouverts. Voici le choix retenu, et pourquoi.

| Point | Choix | Raison |
|---|---|---|
| Actions | entiers 0 = gauche, 1 = centre, 2 = droite, pour les deux rôles | un seul codage partout (matrice, historique, graphiques) |
| Matrice `payoff` | gains de l'attaquant, `payoff[tir, plongeon]` ; le gardien reçoit l'opposé | convention du cahier (section 3.2) |
| Rôle | fixé à la création : `Agent("attacker")` ou `Agent("goalkeeper")` | évite les erreurs de signe ; une instance par rôle |
| Observation | `Observation(payoff, round_index)`, rien d'autre | aucun champ ne peut contenir l'action courante adverse (exigence EF04) |
| Retour de `choose_action` | toujours un entier ; la politique est exposée par `policy(obs)` | le cahier dit « action ou politique » : on sépare les deux, la politique sert à la visualisation (EF07) et à l'exploitabilité |
| Égalités | si plusieurs actions sont optimales, on tire au hasard parmi elles | le jeu de base est symétrique : face à un adversaire uniforme, tout est à égalité |
| Graine | passée au constructeur, remise à zéro par `reset()` | reproductibilité (ENF04) |

**Cycle d'une manche** (à respecter dans `simulation.py` et dans l'interface) :
1. `a = att.choose_action(obs)` et `g = keeper.choose_action(obs)` : les deux choisissent sans rien savoir de l'autre ;
2. le moteur révèle les actions et calcule `r = payoff[a, g]` ;
3. `att.update(obs, a, g, r)` et `keeper.update(obs, g, a, -r)` : seulement maintenant, les agents apprennent.

## 2. Astuce commune : une seule matrice « vue par l'agent »

`own_utility(payoff, role)` renvoie une matrice U dont les lignes sont les actions de l'agent et que l'agent **maximise** toujours :
attaquant : U = payoff ; gardien : U = −payoffᵀ. Ainsi la meilleure réponse et la programmation linéaire s'écrivent une seule fois pour les deux rôles.

## 3. Les quatre agents

**DeterministicAgent** (niveau 1). Rejoue une séquence en boucle (par défaut toujours à gauche). Ne regarde jamais l'adversaire. Sert de référence prévisible.

**FrequencyAgent** (niveau 2). Mémoire courte :
1. garder les `window` (10) dernières actions adverses ;
2. estimer leurs fréquences avec un lissage de 0,5 par action (évite les fréquences nulles) ;
3. jouer la meilleure réponse à cette estimation.

**FictitiousPlayAgent** (niveau 3). Mémoire complète :
1. compter toutes les actions adverses depuis le début (en partant de 1 par action) ;
2. stratégie empirique = comptes / total ;
3. jouer la meilleure réponse.
Différence avec l'agent fréquences, testée : après 100 tirs à gauche puis 20 à droite, l'agent à fenêtre plonge à droite, Fictitious Play continue de plonger à gauche. En auto-jeu, ce sont les **fréquences** qui convergent vers 1/3, pas chaque coup.

**MinimaxAgent** (niveau 4). Résout par programmation linéaire max v sous v ≤ (xᵀU)ⱼ pour chaque colonne j, Σx = 1, x ≥ 0 (programme de J1 du cours). Sa politique ne dépend pas de l'historique. Sur la matrice de base elle est (1/3, 1/3, 1/3) et la valeur est 1/3. Le tirage dans `choose_action` exécute cette stratégie mixte ; ce n'est pas un agent aléatoire.
Le membre 1 fournit aussi `theory/equilibrium.py` : à terme, `solve_maximin` peut y être déplacé pour éviter le doublon.

## 4. Ce que les tests vérifient

Politiques valides (≥ 0, somme 1) pour les 4 agents et les 2 rôles ; aucune fuite d'information (observation sans action adverse, politique inchangée tant qu'il n'y a pas de `update`) ; meilleure réponse correcte pour chaque rôle ; différence fenêtre / cumul ; Minimax uniforme sur la base, bon sur le jeu 2×2 du cours (x* = (1/2, 1/2), v = 5/2) et certificat d'optimalité sur une matrice asymétrique (garantie de l'attaquant = plafond du gardien) ; convergence des fréquences de Fictitious Play en auto-jeu ; `reset` et reproductibilité par graine. Une erreur volontaire de signe a bien fait échouer 2 tests.

## 5. Résultats de l'exemple (gain moyen de l'attaquant, 3000 manches, graines fixées)

| Duel | Gain moyen |
|---|---|
| Déterministe contre Fictitious Play | −0,999 (presque tout est arrêté) |
| Minimax contre Minimax | 0,331 (valeur du jeu : 1/3) |
| Minimax contre Fréquences | 0,340 (Minimax n'est pas exploitable) |
| Fréquences contre Fictitious Play | 0,459 (au-dessus de 1/3 : à analyser avec le benchmark) |
