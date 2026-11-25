# MIPA benchmark

Scripts that produce the benchmarking case study of the MIPA paper: zero-shot baselines, the
supervised models, a text-based supervised model, and the statistics (confidence intervals,
common held-out subset, calibration).

No MIMIC-IV data are stored here. Inputs marked *credentialed* require PhysioNet access to
MIMIC-IV v2.2 (and MIMIC-IV-Note); the others are public in
[MIPA-datacard](https://github.com/open-health-data-lab/MIPA-datacard).

## Inputs

| Input | Source | Access |
|---|---|---|
| Gold labels (1,388 summaries) | `MIPA-datacard/data/annotations/labels.csv` | public |
| Validation/test split per phenotype | `MIPA-datacard/data/cohort/test_val/indexes/` | public |
| Silver training indexes | `MIPA-datacard/data/cohort/train/indexes/` | public |
| GPT-4o predictions (0/1 per summary and phenotype) | `MIPA-datacard/data/benchmark/gpt4o_predictions.csv` | public |
| Gold discharge summaries (`hadm_id`, `text`) | `MIPA-datacard/data/notes/notes.sql` | credentialed |
| ICD codes of the gold admissions | `MIPA-datacard/data/benchmark/icd_codes.sql` | credentialed |
| All discharge summaries (text model training) | MIMIC-IV-Note `discharge.csv.gz` | credentialed |
| Structured feature tables (supervised models) | built by this repository's pipeline (`scripts/`) | credentialed |

ICD code lists and TF-IDF keywords are in `icd_code_lists.json` and `tfidf_keywords.json`
(Supplemental Files 1 and 2 of the paper).

## Scripts

| Script | Produces |
|---|---|
| `zero_shot_benchmark.py` | ICD rule, keyword TF-IDF and GPT-4o on the full gold set and on each phenotype's test half, with 95% CIs |
| `rerun_supervised.py` | Logistic regression, Naive Bayes, random forest and gradient boosting (anchor-and-learn), with saved test-set predictions |
| `text_supervised.py` | TF-IDF bag-of-words + logistic regression trained on silver-labelled discharge summaries |
| `common_subset.py` | All methods on the same test rows: metrics, paired F1 differences, calibration |
| `make_patient_split.py` | Patient-grouped 50/50 split of the gold set (`patient_grouped_split.csv` in MIPA-datacard) |

Each script documents its arguments in its header (`python <script> -h`).

```bash
python benchmark/zero_shot_benchmark.py --labels labels.csv --notes golden_notes.csv \
    --icd icd_codes.csv --gpt-predictions gpt4o_predictions.csv \
    --index-dir MIPA-datacard/data/cohort/test_val/indexes --out out/benchmark
python benchmark/text_supervised.py --discharge discharge.csv.gz --gold-notes golden_notes.csv \
    --train-index-dir MIPA-datacard/data/cohort/train/indexes \
    --test-index-dir MIPA-datacard/data/cohort/test_val/indexes --out out/text_supervised
python benchmark/rerun_supervised.py --out out/supervised      # from the repository root
python benchmark/common_subset.py --labels labels.csv --zero-shot out/benchmark/zero_shot_predictions.csv \
    --supervised-dir out/supervised --text-dir out/text_supervised --out out/common_subset
```

## Reproducibility

- **Zero-shot baselines and statistics:** reproduced exactly from the inputs above.
- **Text model:** deterministic given the inputs (fixed seeds).
- **Supervised models:** `rerun_supervised.py` runs the training code in `scripts/model/` on the
  structured feature tables, which are produced by this repository's processing pipeline and read
  from the paths in `scripts/constants.py` (use `--local-processed DIR` for a local copy). Logistic
  regression uses KNN imputation (7 neighbours); the other models use zero-filling. Seeds are fixed.
  The feature tables are not distributed, so this step requires rebuilding them from MIMIC-IV.
- **GPT-4o:** the predictions were obtained once (model `openai/gpt-4o-2024-11-20`, March 2025;
  prompts in Supplemental File 3). New API calls would not return identical outputs, so the saved
  predictions are provided.
