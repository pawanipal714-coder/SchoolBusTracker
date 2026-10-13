"""
train_model.py  -  Bus delay prediction for SchoolBus Safety Tracker

Task   : predict whether a school bus trip will be DELAYED (> 10 min late)
Models : Logistic Regression (baseline), SVM, Random Forest (proposed)
Outputs: model.joblib, results/*.csv, results/*.png

NOTE ON DATA: no public school-bus dataset is bundled, so generate_data()
simulates trips using realistic relationships. This is stated openly in
the report. If you collect/obtain real trip logs, save them as
data/trips.csv with the same columns and the script will use them instead.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

SEED = 42
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

NUMERIC = ["distance_km", "num_stops", "students_onboard", "hour",
           "traffic_index", "rain_mm", "prev_trip_delay_min"]
CATEGORICAL = ["day_of_week", "zone"]
TARGET = "delayed"


def generate_data(n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "distance_km": rng.uniform(3, 25, n).round(1),
        "num_stops": rng.integers(4, 25, n),
        "students_onboard": rng.integers(10, 55, n),
        "hour": rng.choice([7, 8, 14, 15, 16], n),
        "traffic_index": rng.uniform(0, 10, n).round(2),
        "rain_mm": np.where(rng.random(n) < 0.25, rng.exponential(6, n), 0).round(1),
        "prev_trip_delay_min": np.clip(rng.normal(3, 4, n), 0, None).round(1),
        "day_of_week": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], n),
        "zone": rng.choice(["Urban", "Suburban", "Rural"], n, p=[.4, .4, .2]),
    })
    z = (-6.6 + 0.07 * df.distance_km + 0.08 * df.num_stops
         + 0.012 * df.students_onboard + 0.35 * df.traffic_index
         + 0.09 * df.rain_mm + 0.12 * df.prev_trip_delay_min
         + 0.5 * df.hour.isin([8, 16])
         + 0.4 * (df.zone == "Rural")
         + 0.3 * (df.day_of_week == "Mon"))
    # interaction: rain hurts much more in rural zones (unpaved roads)
    z += 0.15 * df.rain_mm * (df.zone == "Rural")
    p = 1 / (1 + np.exp(-z))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


def load_data():
    path = "data/trips.csv"
    if os.path.exists(path):
        print("Using real data:", path)
        return pd.read_csv(path)
    df = generate_data()
    df.to_csv(path, index=False)
    print("Generated simulated dataset -> data/trips.csv")
    return df


def metrics(y, pred):
    return {"Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred),
            "F1-score": f1_score(y, pred)}


def main():
    df = load_data()
    print("\nDataset:", df.shape, "| delayed rate: %.3f" % df[TARGET].mean())
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])

    # --- preprocessing: impute, clip outliers (1st-99th pct), scale, one-hot
    for c in NUMERIC:
        df[c] = df[c].fillna(df[c].median())
        lo, hi = df[c].quantile([.01, .99])
        df[c] = df[c].clip(lo, hi)
    for c in CATEGORICAL:
        df[c] = df[c].fillna(df[c].mode()[0])

    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])

    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=1000),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "SVM (RBF)": (SVC(probability=True, random_state=SEED),
                      {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED),
                          {"clf__n_estimators": [100, 200, 300],
                           "clf__max_depth": [5, 10, None],
                           "clf__min_samples_leaf": [1, 3, 5]}),
    }

    rows, tuning_logs, best_models = [], [], {}
    for name, (clf, grid) in candidates.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best_models[name] = gs.best_estimator_
        pred = gs.best_estimator_.predict(X_te)
        rows.append({"Method": name, **metrics(y_te, pred),
                     "Best params": str(gs.best_params_)})
        log = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score"]]
        log.insert(0, "model", name)
        tuning_logs.append(log)

    res = pd.DataFrame(rows)
    res.to_csv("results/comparison_table.csv", index=False)
    pd.concat(tuning_logs).to_csv("results/hyperparameter_tuning.csv", index=False)
    print("\n=== Comparison (test set) ===")
    print(res.drop(columns="Best params").round(3).to_string(index=False))

    best_name = res.sort_values("F1-score", ascending=False).iloc[0]["Method"]
    best = best_models[best_name]
    print("\nBest model:", best_name)
    joblib.dump(best, "model.joblib")

    # --- confusion matrix figure
    cm = confusion_matrix(y_te, best.predict(X_te))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["On time", "Delayed"]); ax.set_yticklabels(["On time", "Delayed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(best_name)
    fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=200)

    # --- model comparison bar chart
    res.set_index("Method")[["Accuracy", "Precision", "Recall", "F1-score"]].plot.bar(
        figsize=(7, 4), rot=0, ylim=(0, 1))
    plt.title("Model comparison (test set)"); plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=200)

    # --- fairness / ethics: performance by zone (AI & Society angle)
    pred = best.predict(X_te)
    fair = []
    for z in sorted(X_te.zone.unique()):
        m = (X_te.zone == z).values
        fair.append({"Zone": z, "n": int(m.sum()),
                     "Actual delay rate": y_te[m].mean(),
                     "Predicted delay rate": pred[m].mean(),
                     "Accuracy": accuracy_score(y_te[m], pred[m]),
                     "Recall": recall_score(y_te[m], pred[m])})
    fair = pd.DataFrame(fair)
    fair.to_csv("results/fairness_by_zone.csv", index=False)
    print("\n=== Performance by zone ===")
    print(fair.round(3).to_string(index=False))

    # --- feature importance (Random Forest) for interpretation
    rf = best_models["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    imp.sort_values().tail(10).plot.barh(figsize=(6, 4))
    plt.title("Top 10 feature importances (Random Forest)"); plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=200)
    print("\nSaved model.joblib and results/ folder.")


if __name__ == "__main__":
    main()
