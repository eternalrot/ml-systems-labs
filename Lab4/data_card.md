# Data card: UCI Adult (Census Income)

## Overview
| | |
|---|---|
| Name | Adult (also called Census Income), version 2 on OpenML (dataset 1590) |
| Task | Binary classification: does a person earn more than 50K USD per year? |
| Source | UCI Machine Learning Repository, Becker and Kohavi (1996), extracted by Barry Becker from the 1994 US Census Bureau database. Loaded with `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)` |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI repository page. Check the current licence text on the source page before redistribution. |
| Size | 48 842 rows, 15 columns (14 features and the target `income`); about 5 MB as CSV |
| Class balance | `<=50K`: 76.1%, `>50K`: 23.9% |
| Processed version | train 34 165 rows (after removing 24 exact duplicates), test 14 653 rows, 107 columns after scaling and one-hot encoding; SHA256 in `results/run_manifest.yaml` |

## Features
Numeric: age, fnlwgt (census sampling weight), education-num, capital-gain, capital-loss, hours-per-week. Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country. Missing values are marked `?` (workclass 5.7%, occupation 5.8%, native-country 1.8%). Derived features added by the pipeline: `capital_net`, `hours_per_age`.

## Known biases and limitations
- **Representation.** Men are 66.9% of the rows and women 33.1%; 85.5% of the rows are White and 0.8% are "Other"; 89.7% of the people have `United-States` as native country. Results for small groups are less reliable.
- **Outcome differences between groups.** The share of `>50K` is 30.4% for men and 10.9% for women, and 25.4% for White against 12.1% for Black people. A model learns these historical differences and can reproduce them in its predictions.
- **Age of the data.** The data comes from 1994. The 50K threshold, the jobs and the education levels do not describe today's world, and the threshold is not adjusted for inflation.
- **Features that behave unusually.** capital-gain is top-coded at 99 999 (244 rows have exactly this value) and is zero for 91.7% of people; capital-loss is zero for 95.3%. `fnlwgt` is a sampling weight, not a personal attribute.
- **Missing data.** `?` is not random: it concerns workclass and occupation together and may be related to unemployment or to unwillingness to answer. Imputing the most frequent value hides this.
- **Label noise and duplicates.** 52 exact duplicate rows; 5 groups of identical feature vectors have different income labels. 4 rows have a relationship that conflicts with sex.
- **Not a good benchmark for fairness claims** without further analysis, because several attributes are correlated with the protected attributes (sex, race, native country).

## Privacy concerns
The data is a public, anonymised census sample without names or addresses. However, it contains quasi-identifiers (age, sex, race, native country, marital status, occupation, hours worked). Combinations of them can be rare enough to identify a person if the data is joined with other sources, so it should not be linked with other personal data sets, and predictions about individuals should not be used for decisions about real people.

## Intended and unintended use
Intended: teaching and benchmarking of tabular classification and data-pipeline techniques. Not intended: real decisions about hiring, credit, housing or any other matter that affects people.

## Pipeline and reproducibility
Run from the `Lab4` folder: `python src/run_pipeline.py`. Configuration: `config.yaml`; seed 42; library versions are written to `results/run_manifest.yaml` together with the SHA256 of the processed train and test files.
