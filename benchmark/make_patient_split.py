"""Patient-grouped validation/test split of the MIPA gold set.

One 50/50 split of all 1,388 gold summaries, shared by all phenotypes, in which every summary of a
patient (subject_id) falls in the same half. Provided for users who need patient-level separation;
the per-phenotype admission-level splits used in the paper are kept unchanged.

Among candidate group splits (seeds 0..999), the one with the smallest maximum deviation from a
50% share of positives across the 16 phenotypes is kept, so that both halves contain positives for
every phenotype.

Usage: python benchmark/make_patient_split.py LABELS_CSV OUT_CSV
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PHENOTYPES = ["alcohol_abuse", "c_diff_complication", "c_diff_past", "dementia", "depression",
              "diabetes_type_1", "diabetes_type_2", "hfpef", "hfref", "hypertension",
              "metastatic_cancer", "obesity", "rheumatoid_arthritis", "sle", "vte_complication", "vte_past"]


def main(labels_csv, out_csv):
    df = pd.read_csv(labels_csv)
    assert len(df) == 1388 and df["hadm_id"].is_unique
    best = None
    for seed in range(1000):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_idx, test_idx = next(gss.split(df, groups=df["subject_id"]))
        share = df.iloc[test_idx][PHENOTYPES].sum() / df[PHENOTYPES].sum()
        score = max(abs(share - 0.5).max(), abs(len(test_idx) / len(df) - 0.5))
        if best is None or score < best[0]:
            best = (score, seed, val_idx, test_idx)
    score, seed, val_idx, test_idx = best
    out = df[["hadm_id", "subject_id"]].copy()
    out["set"] = "validation"
    out.iloc[test_idx, out.columns.get_loc("set")] = "test"
    assert out.groupby("subject_id")["set"].nunique().max() == 1
    out.to_csv(out_csv, index=False)
    summary = pd.DataFrame({s: df.loc[out["set"] == s, PHENOTYPES].sum() for s in ["validation", "test"]})
    print(f"seed {seed}; max deviation from 50% positives {score:.3f}")
    print(out["set"].value_counts().to_string())
    print(f"patients: {out.subject_id.nunique()}")
    print(summary.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
