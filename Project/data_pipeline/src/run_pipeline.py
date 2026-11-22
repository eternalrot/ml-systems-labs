"""Lab 4: reproducible, leakage-free Adult pipeline.   One command, from Lab4/:

    python src/run_pipeline.py
"""
import hashlib
import json
import os
import platform
import random
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import Cleaner, build_preprocessor  # noqa: E402
from schema import (CAT_COLS, CATEGORIES, COLUMNS, EDU_NUM, FEATURES, INT_COLS, MISSING,  # noqa: E402
                    RANGES, TARGET)

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
CFG = yaml.safe_load(open(os.path.join(LAB, "config.yaml"), encoding="utf-8"))
RES = os.path.join(LAB, CFG["paths"]["results_dir"])
PROC = os.path.abspath(os.path.join(LAB, CFG["paths"]["processed_dir"]))
os.makedirs(RES, exist_ok=True)
os.makedirs(PROC, exist_ok=True)
SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
MIRROR = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes()).hexdigest()


# ---------------------------------------------------------------- 2. ingestion
def ingest():
    """Read the raw data without modifying it."""
    d = CFG["dataset"]
    try:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml(name=d["openml_name"], version=d["openml_version"], as_frame=True).frame
        source = "OpenML adult v2 (fetch_openml)"
    except Exception as e:  # offline / OpenML unreachable -> public mirror of the same UCI file
        print("fetch_openml failed (%s); using the GitHub mirror of UCI Adult" % type(e).__name__)
        raw = pd.read_csv(MIRROR, header=None, names=COLUMNS, keep_default_na=False)
        source = "GitHub mirror of the UCI Adult file (fetch_openml unavailable)"
    return raw, source


raw, source = ingest()
raw_hash_before = frame_hash(raw)
rows, cols = raw.shape
ing = {"source": source, "rows": rows, "columns": cols,
       "expected_rows": CFG["dataset"]["expected_rows"],
       "expected_columns": CFG["dataset"]["expected_columns"],
       "raw_content_hash": raw_hash_before}
ing["ok"] = bool(rows == ing["expected_rows"] and cols == ing["expected_columns"])
json.dump(ing, open(os.path.join(RES, "ingestion_log.json"), "w"), indent=2)
print(f"[ingest] {source}: {rows} rows x {cols} columns (expected {ing['expected_rows']} x {ing['expected_columns']}) ok={ing['ok']}")

# the only renaming: OpenML calls the target 'class'; the manual calls it 'income'
df = raw.rename(columns={"class": TARGET}).copy()
df[TARGET] = df[TARGET].astype(str).str.strip().str.rstrip(".")


# ---------------------------------------------------------------- 3. validation
def is_missing(s):
    return s.isna() | (s.astype(str).str.strip() == MISSING)


report = []


def add(check, column, rule, n_bad, n_checked, example=""):
    report.append(dict(check=check, column=column, rule=rule, violations=int(n_bad),
                       rows_checked=int(n_checked), status="PASS" if n_bad == 0 else "VIOLATION",
                       example=str(example)[:120]))


add("schema", "(all)", "15 expected columns present", len(set(COLUMNS) - set(df.columns)), len(COLUMNS),
    sorted(set(COLUMNS) - set(df.columns)))
for c in INT_COLS:
    v = pd.to_numeric(df[c], errors="coerce")
    bad = v.isna() | (v != v.round())
    add("type", c, "integer, not null", bad.sum(), len(df), df.loc[bad, c].head(3).tolist())
for c in CAT_COLS + [TARGET]:
    add("type", c, "categorical (string/category)", 0 if (df[c].dtype == object or str(df[c].dtype) == "category"
                                                         or str(df[c].dtype) == "string") else len(df), len(df))
for c, (lo, hi) in RANGES.items():
    v = pd.to_numeric(df[c], errors="coerce")
    bad = (v < lo) | (v > hi)
    add("range", c, f"{lo} <= x <= {hi}", bad.sum(), len(df), df.loc[bad, c].head(3).tolist())
for c in CAT_COLS + [TARGET]:
    vals = df[c].astype(str).str.strip()
    bad = ~vals.isin(CATEGORIES[c]) & ~df[c].isna()
    add("categories", c, "value in allowed set (incl. '?')", bad.sum(), len(df), vals[bad].unique()[:3].tolist())
edu_bad = df["education"].astype(str).str.strip().map(EDU_NUM) != pd.to_numeric(df["education-num"])
add("consistency", "education vs education-num", "mapping from data dictionary", edu_bad.sum(), len(df))
rel_bad = ((df["relationship"].astype(str).str.strip() == "Husband") & (df["sex"].astype(str).str.strip() == "Female")) | \
          ((df["relationship"].astype(str).str.strip() == "Wife") & (df["sex"].astype(str).str.strip() == "Male"))
add("consistency", "relationship vs sex", "Husband=>Male, Wife=>Female", rel_bad.sum(), len(df))
n_dup = int(df.duplicated().sum())
add("duplicates", "(all)", "no exact duplicate rows (reported, NOT dropped here)", n_dup, len(df))
rep = pd.DataFrame(report)
rep.to_csv(os.path.join(RES, "validation_report.csv"), index=False)
miss = pd.DataFrame({"column": FEATURES, "missing": [int(is_missing(df[c]).sum()) for c in FEATURES]})
miss["missing_pct"] = (100 * miss["missing"] / len(df)).round(2)
miss.to_csv(os.path.join(RES, "missing_values.csv"), index=False)
print(f"[validate] {len(rep)} checks, {int((rep.violations > 0).sum())} with violations, "
      f"{int(rep.violations.sum())} violating cells in total; rows were NOT dropped")
print(rep[rep.violations > 0][["check", "column", "violations", "example"]].to_string(index=False))
print("[validate] missing ('?'): " + ", ".join(f"{r.column}={r.missing}" for r in miss.itertuples() if r.missing))

# ---------------------------------------------------------------- 4. split + cleaning
y_all = (df[TARGET] == ">50K").astype(int)
train, test = train_test_split(df, test_size=CFG["split"]["test_size"], stratify=y_all, random_state=SEED)
n_train_before = len(train)
train = train.drop_duplicates()                    # train only; the test set is untouched
n_dup_dropped = n_train_before - len(train)
print(f"[split] train {n_train_before} -> {len(train)} after dropping {n_dup_dropped} exact duplicates; test {len(test)} (untouched)")
y_tr, y_te = (train[TARGET] == ">50K").astype(int), (test[TARGET] == ">50K").astype(int)

# ---------------------------------------------------------------- 5. derived features + baseline
q = tuple(CFG["cleaning"]["clip_percentiles"])
metrics = []
fitted = {}
for derived in (False, True):
    pre = build_preprocessor(derived=derived, q=q)
    Xtr = pre.fit_transform(train[FEATURES])
    Xte = pre.transform(test[FEATURES])
    clf = LogisticRegression(max_iter=CFG["model"]["max_iter"], random_state=SEED).fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    metrics.append(dict(derived_features=derived, n_features_after_encoding=Xtr.shape[1],
                        accuracy=round(accuracy_score(y_te, pred), 4), f1_income_gt_50K=round(f1_score(y_te, pred), 4)))
    fitted[derived] = (pre, Xtr, Xte)
    print(f"[baseline] derived={derived}: accuracy {metrics[-1]['accuracy']}, F1 {metrics[-1]['f1_income_gt_50K']}")
pd.DataFrame(metrics).to_csv(os.path.join(RES, "baseline_metrics.csv"), index=False)

pre, Xtr, Xte = fitted[True]
joblib.dump(pre, os.path.join(RES, "preprocessor.joblib"))

# ---------------------------------------------------------------- leakage checks
cleaner = pre.named_steps["clean"]
chk = []
chk.append(("train and test share no rows (index)", len(set(train.index) & set(test.index)) == 0))
ref_med = {c: float(pd.to_numeric(train[c]).median()) for c in INT_COLS}
chk.append(("fitted medians == medians recomputed from the TRAIN split only", ref_med == cleaner.medians_))
ref_hi = {c: float(pd.to_numeric(train[c]).fillna(ref_med[c]).quantile(q[1])) for c in INT_COLS}
ref_lo = {c: float(pd.to_numeric(train[c]).fillna(ref_med[c]).quantile(q[0])) for c in INT_COLS}
chk.append(("clip bounds (1st/99th percentile) == bounds recomputed from the TRAIN split only",
            ref_hi == cleaner.high_ and ref_lo == cleaner.low_))
chk.append(("test rows were never dropped (no duplicate removal on test)", Xte.shape[0] == len(test)))
full_hi = {c: float(pd.to_numeric(df[c]).quantile(q[1])) for c in INT_COLS}
info = "info: 99th percentile train vs full data: " + ", ".join(
    f"{c} {cleaner.high_[c]:.0f}/{full_hi[c]:.0f}" for c in ("fnlwgt", "capital-gain"))
raw_hash_after = frame_hash(raw)
chk.append(("raw data unchanged by the whole pipeline (content hash identical)", raw_hash_after == raw_hash_before))
with open(os.path.join(RES, "leakage_checks.txt"), "w", encoding="utf-8") as f:
    for name, ok in chk:
        f.write(f"{'PASS' if ok else 'FAIL'}  {name}\n")
    f.write(info + "\n")
print("[leakage]", info)
for name, ok in chk:
    print(f"[leakage] {'PASS' if ok else 'FAIL'}: {name}")
assert all(ok for _, ok in chk), "leakage check failed"

# ---------------------------------------------------------------- 6. label quality
sample = df.sample(n=CFG["label_audit"]["sample_size"], random_state=SEED).copy()


def row_flags(r):
    f = []
    for c in CAT_COLS + [TARGET]:
        if str(r[c]).strip() not in CATEGORIES[c] and not pd.isna(r[c]):
            f.append(f"bad category in {c}")
    for c, (lo, hi) in RANGES.items():
        if not lo <= float(r[c]) <= hi:
            f.append(f"{c} out of range")
    if EDU_NUM.get(str(r["education"]).strip()) != int(float(r["education-num"])):
        f.append("education != education-num")
    if (str(r["relationship"]).strip(), str(r["sex"]).strip()) in {("Husband", "Female"), ("Wife", "Male")}:
        f.append("relationship conflicts with sex")
    if float(r["capital-gain"]) > 0 and float(r["capital-loss"]) > 0:
        f.append("both capital-gain and capital-loss > 0")
    return "; ".join(f)


sample["auto_flags"] = sample.apply(row_flags, axis=1)
sample["manual_check(OK/ISSUE)"] = ""      # to be filled in by hand
sample.to_csv(os.path.join(RES, "label_audit_sample_50.csv"), index=True)
n_flag = int((sample["auto_flags"] != "").sum())

feat = df[FEATURES].astype(str)
grp = df.assign(_k=feat.apply(lambda r: "|".join(r), axis=1)).groupby("_k")[TARGET].agg(["nunique", "size"])
conflict_groups = int((grp["nunique"] > 1).sum())
conflict_rows = int(grp.loc[grp["nunique"] > 1, "size"].sum())
dup_groups = int((grp["size"] > 1).sum())
dup_rows_same_label = int(df.duplicated(keep=False).sum())
lq = {"sampled_rows": len(sample), "sample_rows_with_auto_flags": n_flag,
      "feature_vectors_that_occur_more_than_once": dup_groups,
      "conflicting_groups(same 14 features, different income)": conflict_groups,
      "rows_in_conflicting_groups": conflict_rows,
      "exact_duplicate_rows(all 15 columns, incl. all copies)": dup_rows_same_label,
      "full_dataset_relationship_sex_conflicts": int(rel_bad.sum()),
      "full_dataset_education_mismatches": int(edu_bad.sum())}
json.dump(lq, open(os.path.join(RES, "label_quality.json"), "w"), indent=2)
print("[labels]", json.dumps(lq))

# ---------------------------------------------------------------- 7. processed data, hash, manifest
names = pre.named_steps["encode"].get_feature_names_out()
paths = {}
for part, X, y in (("train", Xtr, y_tr), ("test", Xte, y_te)):
    out = pd.DataFrame(X, columns=names)
    out[TARGET] = y.values
    p = os.path.join(PROC, f"adult_{part}_processed.csv")
    out.to_csv(p, index=False, float_format="%.10g")
    paths[part] = p
hashes = {k: sha256_file(p) for k, p in paths.items()}
combined = hashlib.sha256((hashes["train"] + hashes["test"]).encode()).hexdigest()
manifest = {"config": CFG, "processed_files": {k: os.path.basename(p) for k, p in paths.items()},
            "sha256": {**hashes, "combined": combined},
            "shapes": {"train": list(Xtr.shape), "test": list(Xte.shape)},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
            "ingestion": ing}
yaml.safe_dump(manifest, open(os.path.join(RES, "run_manifest.yaml"), "w", encoding="utf-8"), sort_keys=False)
with open(os.path.join(RES, "processed_dataset.sha256"), "w") as f:
    f.write(f"{hashes['train']}  adult_train_processed.csv\n{hashes['test']}  adult_test_processed.csv\n"
            f"{combined}  combined\n")
print("[hash] sha256 train", hashes["train"])
print("[hash] sha256 test ", hashes["test"])
print("[hash] combined    ", combined)
print("done")
