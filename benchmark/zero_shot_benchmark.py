"""Score the zero-shot MIPA baselines (ICD code rule, keyword TF-IDF, GPT-4o) against the gold labels.

Two evaluation settings
  full   all gold-labelled summaries (n = 1,388). No method learns from labels, so the whole
         gold set can be used; the best configuration per phenotype is reported as observed.
  test   each phenotype's supervised-learning test half (same rows as the supervised models).
         ICD threshold and TF-IDF percentile are chosen on that phenotype's validation half.

TF-IDF scoring (corrected): every keyword is tokenised with the same word-boundary tokenizer as
the notes and matched as a whole 1-6 word term. A note's score is the sum over keyword terms of
(term count / note length in tokens) x smoothed IDF, IDF computed on all gold notes (label-free).
A note is positive when its score is strictly above the chosen percentile of scores.

Confidence intervals: 95% percentile intervals from a patient-level (subject_id) cluster bootstrap,
paired across methods (every method is scored on the same resample).

Inputs are MIMIC-IV derived and credentialed; pass paths, never commit the data.
Usage: python3 zero_shot_benchmark.py --labels L --notes N --icd I --gpt-predictions P --index-dir X --out O
       (--gpt-dir G instead of --gpt-predictions reads the raw aggregated GPT-4o outputs)
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHENOTYPES = ["alcohol_abuse", "c_diff_complication", "c_diff_past", "dementia", "depression",
              "diabetes_type_1", "diabetes_type_2", "hfpef", "hfref", "hypertension",
              "metastatic_cancer", "obesity", "rheumatoid_arthritis", "sle", "vte_complication", "vte_past"]
ICD_THRESHOLDS = [1, 2, 3]
PERCENTILES = [25, 50, 75, 90]
TOKEN = re.compile(r"\b\w+\b")
N_BOOT = 2000
SEED = 20261123


def tokens(text):
    return TOKEN.findall(str(text).lower())


def confusion(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    tp = float(np.sum(w * (y == 1) * (p == 1))); fp = float(np.sum(w * (y == 0) * (p == 1)))
    fn = float(np.sum(w * (y == 1) * (p == 0))); tn = float(np.sum(w * (y == 0) * (p == 0)))
    return tp, fp, fn, tn


def f1_from(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def metrics(y, p):
    tp, fp, fn, tn = confusion(y, p)
    div = lambda a, b: a / b if b else float("nan")
    return dict(n=len(y), positives=int(y.sum()), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                precision_ppv=div(tp, tp + fp), recall_sensitivity=div(tp, tp + fn),
                specificity=div(tn, tn + fp), npv=div(tn, tn + fn),
                accuracy=div(tp + tn, len(y)), f1=f1_from(tp, fp, fn))


def bootstrap_f1(subjects, y, preds, rng):
    """preds: dict name -> 0/1 array. Returns dict name -> array of bootstrap F1 (paired)."""
    uniq, inv = np.unique(subjects, return_inverse=True)
    out = {k: np.empty(N_BOOT) for k in preds}
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        for k, p in preds.items():
            tp, fp, fn, _ = confusion(y, p, w)
            out[k][b] = f1_from(tp, fp, fn)
    return out


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def tfidf_scores(notes, keywords):
    """notes: Series hadm_id -> text. Returns DataFrame hadm_id x phenotype of keyword TF-IDF scores."""
    toks = {h: tokens(t) for h, t in notes.items()}
    terms = {ph: sorted({" ".join(tokens(k)) for k in kws if tokens(k)}) for ph, kws in keywords.items()}
    vocab = sorted({t for ts in terms.values() for t in ts})
    max_n = max(len(t.split()) for t in vocab)
    vocab_set = set(vocab)
    counts = {}
    for h, tk in toks.items():
        c = Counter()
        for n in range(1, max_n + 1):
            for i in range(len(tk) - n + 1):
                g = " ".join(tk[i:i + n])
                if g in vocab_set:
                    c[g] += 1
        counts[h] = c
    N = len(toks)
    df = Counter(t for c in counts.values() for t in c)
    idf = {t: math.log((1 + N) / (1 + df[t])) + 1 for t in vocab}
    rows = {}
    for h, c in counts.items():
        length = max(len(toks[h]), 1)
        rows[h] = {ph: sum(c[t] / length * idf[t] for t in ts) for ph, ts in terms.items()}
    return pd.DataFrame.from_dict(rows, orient="index"), terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--notes", required=True)
    ap.add_argument("--icd", required=True)
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--gpt-predictions", help="CSV: hadm_id + one 0/1 column per phenotype (MIPA-datacard)")
    g_in.add_argument("--gpt-dir", help="directory of per-phenotype aggregated GPT-4o outputs")
    ap.add_argument("--index-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    labels = pd.read_csv(a.labels)
    labels["hadm_id"] = labels["hadm_id"].astype(int)
    labels = labels.set_index("hadm_id")
    notes = pd.read_csv(a.notes, usecols=["hadm_id", "text"])
    notes = notes.assign(hadm_id=notes["hadm_id"].astype(int)).set_index("hadm_id")["text"].reindex(labels.index).fillna("")
    icd = pd.read_csv(a.icd, dtype={"icd_code": str})
    icd_sets = icd.groupby(icd["hadm_id"].astype(int))["icd_code"].apply(set).to_dict()
    code_lists = json.load(open(os.path.join(HERE, "icd_code_lists.json")))
    keywords = json.load(open(os.path.join(HERE, "tfidf_keywords.json")))

    scores, terms = tfidf_scores(notes, keywords)
    json.dump(terms, open(os.path.join(a.out, "tfidf_keyword_terms.json"), "w"), indent=1)

    full_rows, test_rows, diff_rows, pred_rows = [], [], [], []
    for ph in PHENOTYPES:
        y_all = labels[ph]
        ids = y_all.dropna().index
        y = y_all.loc[ids].astype(int).to_numpy()
        subj = labels.loc[ids, "subject_id"].to_numpy()
        n_codes = np.array([len(set(code_lists[ph]) & icd_sets.get(h, set())) for h in ids])
        s = scores.loc[ids, ph].to_numpy()
        if a.gpt_predictions:
            gpt = pd.read_csv(a.gpt_predictions)
            gpt = gpt.assign(hadm_id=gpt["hadm_id"].astype(int)).set_index("hadm_id")[ph]
        else:
            gpt = pd.read_csv(os.path.join(a.gpt_dir, ph, "aggregated_none_any_COT_gpt_4o_post.csv"))
            gpt = gpt.assign(hadm_id=gpt["hadm_id"].astype(int)).set_index("hadm_id")["phenotype_status"]
        g = gpt.reindex(ids)
        assert g.notna().all(), f"missing GPT-4o predictions for {ph}"

        def configs(mask):
            """All configurations; TF-IDF percentile thresholds computed on the rows in `mask`."""
            c = {f"ICD >={t}": (n_codes >= t).astype(int) for t in ICD_THRESHOLDS}
            for p in PERCENTILES:
                thr = np.percentile(s[mask], p)
                c[f"TF-IDF >p{p}"] = (s > thr).astype(int)
            c["GPT-4o"] = g.to_numpy().astype(int)
            return c

        # ---- full gold set ----
        allmask = np.ones(len(ids), bool)
        cf = configs(allmask)
        for name, p in cf.items():
            pred_rows += [(ph, "full", name, h, int(v)) for h, v in zip(ids, p)]
        boot = bootstrap_f1(subj, y, cf, rng)
        for name, p in cf.items():
            m = metrics(y, p); lo, hi = ci(boot[name])
            full_rows.append(dict(phenotype=ph, setting="full", method=name.split(" ")[0],
                                  configuration=name, **m, f1_ci_low=lo, f1_ci_high=hi))

        # ---- supervised test half, configuration chosen on validation half ----
        idx = pd.read_csv(os.path.join(a.index_dir, f"{ph}_indexes.csv"))
        idx["hadm_id"] = idx["hadm_id"].astype(int)
        part = idx.set_index("hadm_id")["set"].reindex(ids)
        val, test = (part == "validation").to_numpy(), (part == "test").to_numpy()
        cv = configs(val)
        chosen = {}
        for fam in ["ICD", "TF-IDF"]:
            cands = [k for k in cv if k.startswith(fam)]
            chosen[fam] = max(cands, key=lambda k: metrics(y[val], cv[k][val])["f1"])
        ct = configs(test)  # TF-IDF thresholds at the chosen percentile, computed label-free on test notes
        sel = {"ICD": ct[chosen["ICD"]][test], "TF-IDF": ct[chosen["TF-IDF"]][test], "GPT-4o": ct["GPT-4o"][test]}
        yt, st_ = y[test], subj[test]
        boot = bootstrap_f1(st_, yt, sel, rng)
        for fam, p in sel.items():
            m = metrics(yt, p); lo, hi = ci(boot[fam])
            test_rows.append(dict(phenotype=ph, setting="test", method=fam,
                                  configuration=chosen.get(fam, "GPT-4o"), **m, f1_ci_low=lo, f1_ci_high=hi))
            pred_rows += [(ph, "test", fam, h, int(v)) for h, v in zip(ids[test], p)]
        for fam in ["ICD", "TF-IDF"]:
            d = boot["GPT-4o"] - boot[fam]
            lo, hi = ci(d)
            diff_rows.append(dict(phenotype=ph, comparison=f"GPT-4o minus {fam}",
                                  f1_difference=metrics(yt, sel["GPT-4o"])["f1"] - metrics(yt, sel[fam])["f1"],
                                  ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))
        print(f"{ph:22s} full n={len(y)} test n={test.sum()} val n={val.sum()} chosen={chosen}")

    pd.DataFrame(full_rows).to_csv(os.path.join(a.out, "zero_shot_full_gold.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(a.out, "zero_shot_test_half.csv"), index=False)
    pd.DataFrame(diff_rows).to_csv(os.path.join(a.out, "zero_shot_test_half_differences.csv"), index=False)
    pd.DataFrame(pred_rows, columns=["phenotype", "setting", "configuration", "hadm_id", "prediction"]).to_csv(
        os.path.join(a.out, "zero_shot_predictions.csv"), index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
