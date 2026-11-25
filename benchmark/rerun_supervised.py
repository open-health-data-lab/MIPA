"""Re-run the published supervised pipeline unchanged and keep note-level test predictions.

Training code, features, seeds and the gold validation/test split are those of the original run
(scripts/model/modeling.py, features read from the MIPA bucket). Two things differ:
  * results are written under --out instead of gs://mipa-bucket/models/ (published files untouched);
  * for every model and phenotype, the test-set predictions and probabilities are saved.

Usage (from the repository root, so that `scripts` is importable):
  python benchmark/rerun_supervised.py --out DIR [--models lr naive_bayes random_forest gradient_boosting]
                                       [--phenotypes hypertension sle ...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", default=["lr", "naive_bayes", "random_forest", "gradient_boosting"])
ap.add_argument("--phenotypes", nargs="+", default=None)
ap.add_argument("--local-processed", default=None,
                help="local mirror of gs://mipa-bucket/processed/ (same layout); avoids re-downloading features")
ap.add_argument("--lr-imputation", default="knn",
                help="imputation method for logistic regression (knn, n=7, as described in the paper)")
args = ap.parse_args()
OUT = os.path.abspath(args.out).rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)

# Redirect every output path BEFORE the pipeline modules copy the constant into their namespaces.
import scripts.constants as k_const  # noqa: E402
k_const.MODEL_PATH = OUT
if args.local_processed:
    GCS = "gs://mipa-bucket/processed/"
    LOCAL = os.path.abspath(args.local_processed).rstrip("/") + "/"
    for _name, _val in list(vars(k_const).items()):
        if isinstance(_val, str) and _val.startswith(GCS):
            setattr(k_const, _name, LOCAL + _val[len(GCS):])
    print(f"reading processed data from {LOCAL}", flush=True)

import pandas as pd  # noqa: E402
import scripts.model.evaluate as ev  # noqa: E402
import scripts.model.modeling as md  # noqa: E402

assert ev.MODEL_PATH == OUT and md.MODEL_PATH == OUT, "MODEL_PATH redirection failed"

_original_eval = ev.evaluate_classification_model
CURRENT = {"model": None}


def evaluate_and_keep_predictions(model, X_test, y_test, phenotype):
    metrics_df = _original_eval(model, X_test, y_test, phenotype)
    y = y_test["y"] if isinstance(y_test, pd.DataFrame) and "y" in y_test.columns else y_test.squeeze()
    pred = pd.DataFrame({"hadm_id": X_test.index, "y": y.loc[X_test.index].to_numpy(),
                         "prediction": model.predict(X_test)})
    if hasattr(model, "predict_proba"):
        pred["probability"] = model.predict_proba(X_test)[:, 1]
    d = os.path.join(OUT, "predictions", CURRENT["model"])
    os.makedirs(d, exist_ok=True)
    pred.to_csv(os.path.join(d, f"{phenotype}.csv"), index=False)
    return metrics_df


ev.evaluate_classification_model = evaluate_and_keep_predictions

TRAINERS = {
    "lr": (md.train_all_lr, {"imputation_method": args.lr_imputation}),
    "naive_bayes": (md.train_all_naive_bayes, {}),
    "random_forest": (md.train_all_random_forest, {"imputation_method": "simple"}),
    "gradient_boosting": (md.train_all_gradient_boosting, {"imputation_method": "simple"}),
}
# scripts.constants.PHENOTYPE_LIST is a debugging subset; default to all sixteen explicitly.
ALL_PHENOTYPES = ["alcohol_abuse", "c_diff_complication", "c_diff_past", "dementia", "depression",
                  "diabetes_type_1", "diabetes_type_2", "hfpef", "hfref", "hypertension",
                  "metastatic_cancer", "obesity", "rheumatoid_arthritis", "sle", "vte_complication", "vte_past"]
phenotypes = args.phenotypes or ALL_PHENOTYPES

for name in args.models:
    fn, kw = TRAINERS[name]
    CURRENT["model"] = name
    print(f"\n=== {name} on {phenotypes} ===", flush=True)
    models = fn(feature_reduction=True, phenotypes=phenotypes, **kw)
    ev.evaluate_all(models, model_type=name)
print("RERUN COMPLETED", flush=True)
