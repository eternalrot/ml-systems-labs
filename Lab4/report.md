# Lab 4: Data pipeline and quality

## 1. Goal
Build a reproducible, leakage-free data pipeline for the UCI Adult (Census Income) dataset: ingest the raw data, validate it against a schema, clean and transform it using statistics from the training split only, engineer two derived features, check the quality of the labels, and produce a hash of the processed data, a configuration file and a data card.

## 2. Problem definition
**Task.** Predict whether a person earns more than 50K USD per year from the 14 census attributes of the Adult dataset (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country). The output is the binary label `income` (`>50K` or `<=50K`). The positive class (`>50K`) is the minority class, about 24% of the rows.

**Success metric.** F1 score of the positive class (`>50K`) together with accuracy. F1 is the main metric because the classes are imbalanced: a model that always answers `<=50K` reaches about 76% accuracy but has an F1 of 0. Both are measured on a held-out test set that is never used for fitting any step of the pipeline.

**Constraints.** (1) No leakage: the split is made first, and every statistic (medians, modes, clipping percentiles, scaling, category lists) is learned on the training split only and then applied to the test split. (2) Reproducibility: fixed seed 42, a configuration file, pinned library versions, a hash of the processed data and one command that runs everything. (3) No silent changes: validation reports violations and does not drop rows; the only rows removed are exact duplicates in the training split, and this is counted and logged.

## 3. Method
The whole pipeline is one command, run from the `Lab4` folder: `python src/run_pipeline.py`. Settings are in `config.yaml`, code is in `src/` (`schema.py`, `preprocessing.py`, `run_pipeline.py`).

1. **Ingestion.** `fetch_openml(name='adult', version=2, as_frame=True)` reads the raw data without modifying it. The row and column counts are logged (`results/ingestion_log.json`), together with a content hash of the raw data that is compared again at the end of the run to prove that nothing changed it. The only change is a rename of the target column from `class` to `income`, done on a copy.
2. **Validation.** Types, ranges, allowed categories (the `?` value is allowed), missing values, duplicates, and two consistency rules (education against education-num, relationship against sex) are checked. All findings go to `results/validation_report.csv`; no row is dropped. `?` and NaN are both treated as missing.
3. **Split and cleaning.** 70/30 stratified split (seed 42) before any cleaning. A scikit-learn pipeline fitted on the train split only: missing numeric values -> median, missing categorical values (`?`) -> most frequent value, numeric features clipped to the 1st and 99th percentile of the train data. Exact duplicates are dropped from the train split only. Then standard scaling of numeric features and one-hot encoding of categorical features (unknown categories ignored). The fitted pipeline is saved with `joblib.dump` to `results/preprocessor.joblib`.
4. **Derived features.** `capital_net = capital-gain - capital-loss` and `hours_per_age = hours-per-week / (age - 16)`. Age is at least 17, so the denominator is at least 1. They are computed after cleaning, inside the same pipeline.
5. **Baseline.** `LogisticRegression(max_iter=1000)` with and without the two derived features; accuracy and F1 on the test set.
6. **Label quality.** 50 random rows (seed 42) saved for review, plus a duplicate-conflict audit over the whole dataset.
7. **Hash and data card.** SHA256 of the processed train and test files, stored with the configuration and library versions in `results/run_manifest.yaml`; the data card is in `data_card.md`.

## 4. Results

### 4.1 Ingestion
48 842 rows and 15 columns, exactly as expected.

### 4.2 Validation report (`results/validation_report.csv`)
32 checks were run. 30 passed: all types, all ranges (age 17-90, hours 1-99, capital-gain 0-99999, capital-loss 0-4356), all categories and the education/education-num mapping. Two checks found problems:

| Check | Violations | Meaning |
|---|---|---|
| Relationship vs sex | 4 rows | `Husband` recorded with sex `Female` or `Wife` with sex `Male` |
| Exact duplicate rows | 52 rows | identical copies of an earlier row |

Missing values (`?`): workclass 2 799 (5.73%), occupation 2 809 (5.75%), native-country 857 (1.75%). They are allowed by the null policy and are imputed in the cleaning step. None of these rows was dropped by the validation.

### 4.3 Split and cleaning
Train 34 189 rows -> 34 165 after removing 24 exact duplicates from the train split; the test set has 14 653 rows and was not touched. Both sets have 107 columns after encoding. Leakage checks (`results/leakage_checks.txt`) all pass: train and test do not share rows; the fitted medians and the clipping bounds are equal to values recomputed from the train split alone; the test set kept all its rows; the raw data had the same content hash before and after the run.

### 4.4 Derived features (LogisticRegression, test set)
| Features | Columns after encoding | Accuracy | F1 (>50K) |
|---|---|---|---|
| Original 14 | 105 | 0.8540 | 0.6601 |
| + capital_net, hours_per_age | 107 | 0.8578 | 0.6696 |

The two derived features give a small improvement: +0.38 points of accuracy and +0.95 points of F1. The test set has 14 653 rows, so the standard error of the accuracy is about 0.3 points; the accuracy gain is therefore only about one standard error, and the F1 gain is also small. A single split was used.

### 4.5 Label quality (`results/label_quality.json`)
- Duplicate-conflict audit: 54 feature vectors occur more than once; in 5 of these groups the identical 14 features come with different income labels (10 rows in total). This is an irreducible label noise: no model can predict both labels for the same input.
- 50 random rows (`results/label_audit_sample_50.csv`) were checked against the data dictionary with automatic rules (category and range, education mapping, relationship against sex, capital-gain and capital-loss both positive): 0 of the 50 rows were flagged. The same rows were then reviewed by hand and all 50 were marked OK in the column `manual_check` (see the notes below).
- On the whole dataset the same rules found 4 relationship/sex conflicts and no other impossible value.

### 4.6 Hash and reproducibility
SHA256 of the processed data (files `adult_train_processed.csv` and `adult_test_processed.csv` in `data/processed/`, hash and configuration stored together in `results/run_manifest.yaml`, hash alone in `results/processed_dataset.sha256`):

- train: `641eff829315bdf42ff27cd20c095ada94fa028181f34d3aae12fd63f5527828`
- test: `808a06c68e5bfcb4b727812f66859ff39619c66f65d13bfda5637e9338e413b0`
- combined: `f093328c7e5da425bf10fab0bb9867e79180d90df94c7dd207f6d7f73d539412`

These values come from my own run on Windows (Python 3.11.8, scikit-learn 1.5.0, data from OpenML).

## 5. Conclusions
1. What keeps this pipeline free of leakage is the order of the steps. I split the data first, and only then learn anything from it: medians, most frequent values, clipping bounds, scaling and the list of categories all come from the training rows. The test rows only get these numbers applied. I did not just trust this; `leakage_checks.txt` recomputes the medians and bounds from the train split alone and they match, and no test row was ever dropped.
2. A validation step should tell me what is wrong, not quietly fix it. Only 2 of the 32 checks failed (4 rows where relationship and sex contradict each other, and 52 duplicate rows), and because they are reported, I know exactly how many rows are affected. The 5-6% of missing values are allowed by my schema and handled later in cleaning, so they are not an error, but they are still counted in `missing_values.csv`.
3. Better data matters more than cleverer features. The two features I added raised F1 by under one point (0.660 to 0.670) and accuracy by about 0.4 points, which is close to the noise of a single split. Meanwhile the data has built-in limits: 5 groups of identical inputs with different labels, duplicates, and census data from 1994 with a skewed sex and race balance. That is why the data card lists these biases next to the numbers.

## Notes and limitations
- The data was loaded from OpenML (`fetch_openml`); the ingestion log records the source. If OpenML is not reachable the script falls back to a public copy of the same UCI file, and then the hashes can differ.
- I reviewed the 50 sampled rows by hand for plausibility and consistency (category values, education against education-num, relationship against sex, hours and capital values) and marked all of them OK in `manual_check`. I cannot check a label against the real person, so this only shows that no row is impossible or self-contradictory. Rows worth a remark: row 3375 has only 4 working hours per week, and a few rows are labelled `Own-child` although the person is 25 and married or 40 and divorced; they are unusual but possible.
- One train/test split and one seed were used, so small differences in the metrics are within noise.
