"""Text-based supervised baseline: TF-IDF bag-of-words + logistic regression on discharge summaries.

Same design as the structured supervised models, so that input representation can be compared:
  training    silver-labelled admissions (ICD proxy) from the MIPA training indexes; gold patients
              were already excluded from these cohorts
  tuning      C chosen by 5-fold stratified cross-validation on the silver training set (F1)
  evaluation  the phenotype's gold test half (same rows as the structured supervised models);
              decision threshold 0.5, as for the structured models
The gold validation half is not used (this model has no second learning stage).

Inputs are MIMIC-IV derived and credentialed; pass paths, never commit the data.
Usage: python3 text_supervised.py --discharge discharge.csv.gz --gold-notes golden_notes.csv
           --train-index-dir DIR --test-index-dir DIR --out DIR [--phenotypes ...]
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

PHENOTYPES = ["alcohol_abuse", "c_diff_complication", "c_diff_past", "dementia", "depression",
              "diabetes_type_1", "diabetes_type_2", "hfpef", "hfref", "hypertension",
              "metastatic_cancer", "obesity", "rheumatoid_arthritis", "sle", "vte_complication", "vte_past"]
SEED = 42
MAX_TRAIN = 20000  # stratified cap on silver training admissions, for tractability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discharge", required=True); ap.add_argument("--gold-notes", required=True)
    ap.add_argument("--train-index-dir", required=True); ap.add_argument("--test-index-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--phenotypes", nargs="+", default=PHENOTYPES)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "predictions"), exist_ok=True)

    needed = set()
    for ph in a.phenotypes:
        needed |= set(pd.read_csv(os.path.join(a.train_index_dir, f"{ph}_indexes.csv"))["hadm_id"].astype(int))
    notes = []
    for chunk in pd.read_csv(a.discharge, usecols=["hadm_id", "text"], chunksize=50000):
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        notes.append(chunk[chunk["hadm_id"].isin(needed)])
    silver_text = pd.concat(notes).groupby("hadm_id")["text"].apply(lambda s: "\n".join(s.astype(str)))
    gold = pd.read_csv(a.gold_notes, usecols=["hadm_id", "text"])
    gold_text = gold.assign(hadm_id=gold["hadm_id"].astype(int)).set_index("hadm_id")["text"].fillna("")
    print(f"silver notes found for {len(silver_text)} of {len(needed)} training admissions", flush=True)

    rows = []
    for ph in a.phenotypes:
        tr = pd.read_csv(os.path.join(a.train_index_dir, f"{ph}_indexes.csv"))
        tr["hadm_id"] = tr["hadm_id"].astype(int)
        n_index = len(tr)
        tr = tr[tr["hadm_id"].isin(silver_text.index)].drop_duplicates("hadm_id")
        if len(tr) > MAX_TRAIN:
            frac = MAX_TRAIN / len(tr)
            tr = pd.concat([g.sample(frac=frac, random_state=SEED) for _, g in tr.groupby("y")])
        te = pd.read_csv(os.path.join(a.test_index_dir, f"{ph}_indexes.csv"))
        te["hadm_id"] = te["hadm_id"].astype(int)
        te = te[te["set"] == "test"]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                      min_df=5, max_df=0.9, max_features=50000, sublinear_tf=True)),
            ("lr", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")),
        ])
        grid = GridSearchCV(pipe, {"lr__C": [1, 10, 100]}, scoring="f1", n_jobs=-1,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        grid.fit(silver_text.loc[tr["hadm_id"]].to_numpy(), tr["y"].to_numpy())
        prob = grid.predict_proba(gold_text.loc[te["hadm_id"]].to_numpy())[:, 1]
        pred = (prob >= 0.5).astype(int)
        pd.DataFrame({"hadm_id": te["hadm_id"].to_numpy(), "y": te["y"].to_numpy(),
                      "prediction": pred, "probability": prob}).to_csv(
            os.path.join(a.out, "predictions", f"{ph}.csv"), index=False)
        y = te["y"].to_numpy()
        tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        rows.append(dict(phenotype=ph, train_index_n=n_index, train_n_used=len(tr),
                         train_pos_pct=round(100 * tr["y"].mean(), 1), best_C=grid.best_params_["lr__C"],
                         cv_f1_silver=round(grid.best_score_, 3), test_n=len(te), test_pos=int(y.sum()),
                         TP=tp, FP=fp, FN=fn, TN=tn, f1=round(f1, 3)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "text_supervised_summary.csv"), index=False)


if __name__ == "__main__":
    main()
