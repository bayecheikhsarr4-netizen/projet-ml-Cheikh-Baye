"""Projet 2 - Entrainement du detecteur d'attaques reseau (IDS).

Lance depuis la racine du depot : python src/train.py
- explore les donnees
- compare 3 modeles (regression logistique, arbre de decision, foret aleatoire)
- evalue sur un jeu de validation (F1, rappel, precision de la classe attaque)
- re-entraine le meilleur sur toutes les donnees et l'enregistre dans models/model.joblib
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "flux_train.csv"
MODEL = BASE / "models" / "model.joblib"
SEED = 42

NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
       "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
CAT = ["protocol"]


def make_pipeline(clf) -> Pipeline:
    """Pretraitement (imputation + echelle + encodage) + modele, dans un seul Pipeline."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def explorer(df: pd.DataFrame) -> None:
    print("=== Exploration ===")
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Repartition de 'attaque' :\n", df["attaque"].value_counts(normalize=True).round(3))


def main() -> None:
    df = pd.read_csv(DATA)
    explorer(df)
    X, y = df.drop(columns="attaque"), df["attaque"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    modeles = {
        "Regression logistique": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Arbre de decision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Foret aleatoire": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=SEED, n_jobs=-1),
    }

    scores = {}
    print("\n=== Comparaison des modeles (validation 25 %) ===")
    for nom, clf in modeles.items():
        pipe = make_pipeline(clf).fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        cv = cross_val_score(make_pipeline(clf), X, y, scoring="f1",
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        scores[nom] = {
            "f1": f1_score(y_val, pred), "rappel": recall_score(y_val, pred),
            "precision": precision_score(y_val, pred), "f1_cv": cv.mean(),
        }
        print(f"\n--- {nom} ---")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        print(f"Matrice de confusion : VN={tn}  FP={fp}  FN={fn}  VP={tp}")
        print(f"F1 validation={scores[nom]['f1']:.3f} | F1 validation croisee (5 plis)={cv.mean():.3f}")

    meilleur = max(scores, key=lambda n: scores[n]["f1_cv"])
    print(f"\n>>> Modele retenu : {meilleur}")

    final = make_pipeline(modeles[meilleur]).fit(X, y)   # re-entrainement sur toutes les donnees
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL)
    print(f"Modele enregistre dans {MODEL.relative_to(BASE)}")


if __name__ == "__main__":
    main()
