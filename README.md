# Projet 2 — Détecter des attaques réseau avec le Machine Learning

## Lancer le projet
```bash
pip install -r requirements.txt
python src/train.py      # compare les modèles et enregistre models/model.joblib
python src/predict.py    # petit test de la fonction predict
```

## Structure
```
├── README.md
├── requirements.txt
├── data/flux_train.csv
├── src/train.py
├── src/predict.py       (fonction predict(df) -> list de 0/1)
└── models/model.joblib  (Pipeline : prétraitement + modèle)
```

## Méthode
- **Exploration** : 3000 flux, 90 valeurs manquantes (colonne `duration`), 27,8 % d'attaques → classes déséquilibrées, donc évaluation avec le **F1 de la classe « attaque »**.
- **Prétraitement (dans un `Pipeline`)** : imputation par la médiane, `StandardScaler` sur les variables numériques, `OneHotEncoder` sur `protocol`.
- **Validation** : 75 % entraînement / 25 % validation (stratifié) + validation croisée à 5 plis.
- Modèles à classes pondérées (`class_weight="balanced"`).

## Résultats

**Nom / Prénom :** Cheikh Baye

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 validation croisée (5 plis) |
|---|---|---|---|---|
| Régression logistique | 0,919 | 0,899 | 0,940 | 0,896 |
| Arbre de décision (profondeur 6) | 0,959 | 0,957 | 0,961 | 0,935 |
| **Forêt aléatoire (300 arbres)** | **0,973** | **0,971** | **0,976** | **0,960** |

**Modèle retenu et pourquoi :** la **forêt aléatoire**. Elle obtient le meilleur F1 sur la validation (0,973) et sur la validation croisée (0,960), avec les meilleurs rappel et précision. Elle capture les relations non linéaires entre les variables (ex. beaucoup de connexions vers la même machine + SYN élevé = DoS/scan), ce que la régression logistique modélise mal. Elle a été ré-entraînée sur les 3000 flux avant d'être enregistrée dans `models/model.joblib`. Variables les plus importantes : `dst_bytes` (0,37), `nb_conn_same_host` (0,23), `syn_rate` (0,12), `nb_failed_logins` (0,10).

**Analyse des erreurs (faux négatifs / faux positifs) :** matrice de confusion de la forêt aléatoire sur la validation (750 flux) :

| | Prédit normal | Prédit attaque |
|---|---|---|
| **Réel normal** | 537 (VN) | 5 (FP) |
| **Réel attaque** | 6 (FN) | 202 (VP) |

- **6 faux négatifs** : attaques non détectées (sur 208, soit 2,9 %). C'est le **pire cas en sécurité** : l'attaque passe sans alerte et l'intrusion, le DoS ou la force brute peut réussir sans que personne ne le sache, alors qu'une fausse alerte ne coûte que du temps d'analyse.
- **5 faux positifs** : flux normaux signalés à tort (0,9 %), ce qui reste gérable mais peut provoquer une « fatigue d'alertes » s'ils se multiplient.
- Pistes d'amélioration : abaisser le seuil de décision pour augmenter le rappel, ajouter des variables (rythme des connexions, taille moyenne des paquets) ou utiliser plus de données.
