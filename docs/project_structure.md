# Structure du projet

Ce document décrit le rôle de chaque dossier et de chaque fichier du projet.


# Racine du projet

## app.py

Point d'entrée principal de l'application Streamlit.
Initialise l'interface, la navigation entre les pages et coordonne les différents modules.


## requirements.txt

Liste des dépendances Python nécessaires à l'exécution du projet.


## README.md

Présentation générale du projet, installation, lancement et utilisation.


# docs/

Documentation technique et organisationnelle du projet.

## architecture.md

Description de l'architecture logicielle globale et des interactions entre les modules.

## api_contract.md

Définition des structures de données, signatures de fonctions et contrats d'interface entre les membres.

## game_theory.md

Documentation mathématique : théorie des jeux, équilibre de Nash, minimax et autres notions utilisées.

## meeting_notes.md

Compte-rendu des réunions, décisions techniques et suivi du projet.

## project_structure.md

Description détaillée de l'arborescence du projet.

---

# game/

Contient le moteur du jeu.

## actions.py

Définition des actions disponibles dans le jeu (LEFT, CENTER, RIGHT).

## constants.py

Constantes globales du moteur : récompenses, paramètres par défaut, configurations communes.

## rules.py

Implémente les règles officielles du jeu et vérifie les actions valides.

## payoff.py

Contient la matrice des gains et les fonctions permettant de calculer les récompenses.

## simulation.py

Gère l'exécution d'une manche ou d'un match complet.

## models.py

Définition des classes de données partagées par tout le projet (MatchConfig, RoundResult, etc.).


# agents/

Contient les différents agents IA.

## base_agent.py

Classe abstraite qui définit l'interface commune de tous les agents.

## deterministic_agent.py

Implémente un agent utilisant une stratégie fixe et prévisible.

## frequency_agent.py

Implémente un agent qui s'adapte à partir des fréquences observées.

## minimax_agent.py

Implémente un agent utilisant la stratégie minimax optimale.

## fictitious_play_agent.py

Implémente un agent utilisant l'algorithme de Fictitious Play.

---

# theory/

Fonctions mathématiques utilisées par les agents et les analyses.

## equilibrium.py

Calcul des équilibres de Nash et des stratégies mixtes optimales.

## metrics.py

Calcul des indicateurs de performance utilisés dans le benchmark.

## validation.py

Fonctions permettant de vérifier les propriétés mathématiques du modèle.


# ui/

Composants de l'interface utilisateur Streamlit.

## play_page.py

Page permettant à un humain de jouer contre une IA.

## laboratory_page.py

Page d'expérimentation pour lancer des simulations et comparer des stratégies.

## benchmark_page.py

Page dédiée à la comparaison quantitative des agents.

## analysis_page.py

Page présentant les analyses détaillées des stratégies et résultats.

---

# visualization/

Fonctions responsables des graphiques et visualisations.

## strategy_plots.py

Création des courbes et graphiques liés aux stratégies.

## benchmark_plots.py

Visualisations des résultats comparatifs entre agents.

## heatmaps.py

Génération des cartes thermiques des matrices ou fréquences.

## dashboards.py

Assemblage des différentes visualisations dans des tableaux de bord.

---

# data/

Stockage des données produites par l'application.

## experiments/

Résultats bruts des expériences et simulations.

## exports/

Exports générés pour téléchargement ou archivage (CSV).

## samples/

Jeux de données d'exemple utilisés pour les démonstrations.

---

# tests/

Tests automatisés du projet.

## test_rules.py

Validation du comportement des règles du jeu.

## test_payoff.py

Vérification de la matrice des gains.

## test_simulation.py

Tests du moteur de simulation.

## test_equilibrium.py

Validation du calcul de l'équilibre de Nash et du minimax.

## test_agents.py

Tests des comportements attendus des agents.

---

# notebooks/

Carnets de recherche et d'expérimentation.

## minimax_validation.ipynb

Validation expérimentale des calculs minimax.

## fictitious_play_analysis.ipynb

Étude du comportement et de la convergence du Fictitious Play.

## benchmark_experiments.ipynb

Comparaison expérimentale des performances des agents.