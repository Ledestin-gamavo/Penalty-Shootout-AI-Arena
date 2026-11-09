# Guide de prise en main — Penalty Shootout AI Arena

## 1. Installation et lancement

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre dans le navigateur (par défaut http://localhost:8501).

## 2. Les règles en 20 secondes

- L'attaquant et le gardien choisissent **en même temps** une zone : gauche, centre ou droite.
- Même zone : **arrêt** (le gardien gagne). Zones différentes : **but** (l'attaquant gagne).
- Le jeu est à somme nulle : ce que gagne l'un, l'autre le perd.
- À l'équilibre, jouer chaque zone avec une probabilité de 1/3 est optimal.

## 3. Jouer contre l'IA (page « Jouer » → Humain contre IA)

1. Choisis ton rôle : **attaquant** ou **gardien**.
2. Choisis le niveau de l'adversaire :
   - **1 — Découverte** : joue toujours pareil, à toi de le repérer ;
   - **2 — Adaptatif** : contre tes habitudes récentes ;
   - **3 — Stratégique** : répond à tes fréquences cumulées (Fictitious Play) ;
   - **4 — Optimal** : stratégie mixte optimale (Minimax), impossible à exploiter.
3. Règle le nombre de manches et la graine (même graine = même partie côté IA).
4. Clique sur **Lancer la partie**, puis sur **Gauche / Centre / Droite** à chaque manche.
5. À la fin : bilan, tendances (tes fréquences contre celles de l'IA) et historique.

Astuce : contre le niveau 1, change de zone dès que tu as repéré son réflexe. Contre le niveau 4, aucune stratégie ne fait mieux que le hasard équilibré.

## 4. IA contre IA (page « Jouer » → IA contre IA)

1. Choisis l'agent attaquant et l'agent gardien.
2. Choisis le nombre de manches (jusqu'à 100 000) et la graine.
3. **Accélérée** : seul le bilan et les 20 dernières manches sont affichés. **Détaillée** : tout l'historique.
4. Lis le taux de buts, le taux d'arrêts, le gain moyen de l'attaquant et les fréquences de chaque agent.

## 5. Dépannage

| Problème | Solution |
|---|---|
| Message « Agents factices en usage » | Normal tant que `agents/` n'est pas branché (voir ci-dessous). |
| La partie « repart de zéro » | Ne pas recharger la page (F5) : l'état est conservé dans la session du navigateur. |
| `ModuleNotFoundError: ui` | Lancer `streamlit run app.py` depuis la racine du projet. |

## 6. Pour l'équipe : brancher les vrais modules

Tout passe par **`ui/compat.py`** (seul fichier à modifier) :

- `_REAL_AGENTS` : noms de modules et de classes des agents du Membre 2 (constructeur attendu : `Classe(seed=...)`).
- `resolve_round(...)` : à remplacer par les fonctions de `game/` (Membre 1).
- Les pages Laboratoire / Analyse / Benchmark sont des placeholders pour le Membre 4.

Lancer les tests : `python -m pytest -q`
