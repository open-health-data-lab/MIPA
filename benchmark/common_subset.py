"""Compare every method on one common held-out subset per phenotype, with paired bootstrap CIs.

Common subset = the phenotype's gold test half, restricted to summaries scored by every method.
Zero-shot predictions come from zero_shot_benchmark.py (setting 'test'); supervised predictions
from rerun_supervised.py and text_supervised.py.

Outputs
  common_subset_metrics.csv      n, positives, confusion matrix, PPV, sensitivity, specificity, NPV,
                                 F1 (95% CI), accuracy, per method and phenotype
  common_subset_differences.csv  paired F1 differences vs. a reference method, 95% CI
  calibration.csv                Brier score, calibration intercept and slope (probabilistic models)
CIs: percentile intervals from a patient-level (subject_id) cluster bootstrap, 2,000 resamples,
paired across methods.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 20261123


def confusion(y, p, w):
    return (np.sum(w * (y == 1) * (p == 1)), np.sum(w * (y == 0) * (p == 1)),
            np.sum(w * (y == 1) * (p == 0)), np.sum(w * (y == 0) * (p == 0)))


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def div(a, b):
    return a / b if b else float("nan")


def logistic_calibration(y, prob):
    """Calibration intercept (slope fixed at 1) and slope, by Newton-Raphson on logit(prob)."""
    eps = 1e-6
    z = np.log(np.clip(prob, eps, 1 - eps) / (1 - np.clip(prob, eps, 1 - eps)))
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(1000):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        W = mu * (1 - mu) + 1e-12
        step = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - mu))
        step *= min(1.0, 1.0 / max(np.max(np.abs(step)), 1e-12))  # damped Newton: convex, so this converges
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    a = 0.0
    for _ in range(1000):  # intercept with offset z (calibration-in-the-large)
        mu = 1 / (1 + np.exp(-(a + z)))
        step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12)
        step = float(np.clip(step, -1.0, 1.0))  # damped Newton step
        a += step
        if abs(step) < 1e-8:
            break
    return a, beta[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--zero-shot", required=True, help="zero_shot_predictions.csv")
    ap.add_argument("--supervised-dir", required=True, help="rerun output dir (contains predictions/<model>/)")
    ap.add_argument("--text-dir", required=True, help="text model output dir (contains predictions/)")
    ap.add_argument("--reference", default="GPT-4o")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    labels = pd.read_csv(a.labels)
    subj = dict(zip(labels["hadm_id"].astype(int), labels["subject_id"]))

    preds = {}  # (phenotype, method) -> DataFrame hadm_id, y?, prediction, probability?
    z = pd.read_csv(a.zero_shot)
    for (ph, m), g in z[z["setting"] == "test"].groupby(["phenotype", "configuration"]):
        preds[(ph, m)] = g[["hadm_id", "prediction"]].astype({"hadm_id": int})
    names = {"lr": "Logistic regression", "naive_bayes": "Naive Bayes",
             "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}
    for f in glob.glob(os.path.join(a.supervised_dir, "predictions", "*", "*.csv")):
        m = os.path.basename(os.path.dirname(f)); ph = os.path.basename(f)[:-4]
        preds[(ph, names.get(m, m))] = pd.read_csv(f).astype({"hadm_id": int})
    for f in glob.glob(os.path.join(a.text_dir, "predictions", "*.csv")):
        preds[(os.path.basename(f)[:-4], "Text TF-IDF + LR")] = pd.read_csv(f).astype({"hadm_id": int})

    met_rows, diff_rows, cal_rows = [], [], []
    for ph in sorted({k[0] for k in preds}):
        methods = sorted(m for (p, m) in preds if p == ph)
        common = set.intersection(*(set(preds[(ph, m)]["hadm_id"]) for m in methods))
        ids = np.array(sorted(common))
        y = labels.set_index(labels["hadm_id"].astype(int)).loc[ids, ph].to_numpy().astype(int)
        s = np.array([subj[h] for h in ids])
        P = {m: preds[(ph, m)].set_index("hadm_id").loc[ids, "prediction"].to_numpy().astype(int) for m in methods}
        uniq, inv = np.unique(s, return_inverse=True)
        boots = {m: np.empty(N_BOOT) for m in methods}
        for b in range(N_BOOT):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
            for m in methods:
                tp, fp, fn, _ = confusion(y, P[m], w)
                boots[m][b] = f1(tp, fp, fn)
        ones = np.ones(len(y))
        for m in methods:
            tp, fp, fn, tn = (int(v) for v in confusion(y, P[m], ones))
            met_rows.append(dict(phenotype=ph, method=m, n=len(y), positives=int(y.sum()),
                                 patients=len(uniq), TP=tp, FP=fp, FN=fn, TN=tn,
                                 ppv=div(tp, tp + fp), sensitivity=div(tp, tp + fn),
                                 specificity=div(tn, tn + fp), npv=div(tn, tn + fn),
                                 accuracy=div(tp + tn, len(y)), f1=f1(tp, fp, fn),
                                 f1_ci_low=np.percentile(boots[m], 2.5), f1_ci_high=np.percentile(boots[m], 97.5)))
            if m != a.reference and a.reference in P:
                d = boots[a.reference] - boots[m]
                lo, hi = np.percentile(d, [2.5, 97.5])
                diff_rows.append(dict(phenotype=ph, comparison=f"{a.reference} minus {m}",
                                      f1_difference=f1(*confusion(y, P[a.reference], ones)[:3]) - f1(tp, fp, fn),
                                      ci_low=lo, ci_high=hi, interval_excludes_zero=bool(lo > 0 or hi < 0)))
            df = preds[(ph, m)]
            if "probability" in df.columns and df["probability"].notna().all():
                prob = df.set_index("hadm_id").loc[ids, "probability"].to_numpy()
                ci_int, slope = logistic_calibration(y, prob)
                cal_rows.append(dict(phenotype=ph, method=m, n=len(y), observed_prevalence=y.mean(),
                                     mean_predicted=prob.mean(), brier=np.mean((prob - y) ** 2),
                                     calibration_intercept=ci_int, calibration_slope=slope))
        print(f"{ph:22s} common n={len(y)} positives={y.sum()} methods={len(methods)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(met_rows).to_csv(os.path.join(a.out, "common_subset_metrics.csv"), index=False)
    pd.DataFrame(diff_rows).to_csv(os.path.join(a.out, "common_subset_differences.csv"), index=False)
    pd.DataFrame(cal_rows).to_csv(os.path.join(a.out, "calibration.csv"), index=False)


if __name__ == "__main__":
    main()
