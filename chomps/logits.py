"""Token-logit features of the answers (DESIGN.md, "Logits").

Every SHROOM-CAP answer carries `output_logits`, one float per generated token
(the generator's logit for the emitted token; higher = more confident). This
module summarises that sequence into eight features, measures what they alone
say about each label, and exposes `features(example)` for `meta.py --logits`.

Features of x_1..x_n (n >= 1), in FEATURES order:
    mean, std (population, ddof=0: n=1 gives 0), min,
    p10, p25 (numpy percentiles, linear interpolation),
    frac_below_15 (share of tokens with x_i < 15), n_tokens (n),
    mean_first_10 (mean of x_1..x_min(n,10)).

Input checks (no fallbacks): the logits must be a non-empty list of finite
numbers, as long as `output_tokens` when that field is present. Twenty Spanish
train rows (es-train-0..19) store both fields as the string serialisation of
the list; those strings are decoded strictly (JSON array for the logits,
Python literal for the tokens) and then checked like every other row.

Analysis, per language and track, on train+validation pooled (test untouched):
    cv_auc          ROC AUC of StandardScaler + meta.logreg(class_weight=None) on
                    the eight features, meta.cv_folds (stratified 5-fold,
                    shuffled, random_state=42; scaler fitted inside each training
                    fold). Rows fold1..fold5, mean, std (ddof=1).
    point_biserial  scipy.stats.pointbiserialr(label, feature): r and two-sided p.
Writes ../results/tables/logits_{lang}.csv (n, n_pos = the pooled set).
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data, meta

FEATURES = ("mean", "std", "min", "p10", "p25", "frac_below_15", "n_tokens", "mean_first_10")
LOW_LOGIT = 15.0
FIRST_K = 10
POOLED_SPLITS = ("train", "validation")
COLUMNS = ("lang", "track", "kind", "term", "estimate", "p_value", "n", "n_pos")


def token_logits(example: dict) -> np.ndarray:
    """The example's per-token logits as a validated float array."""
    where = example.get("index", "<no index>")
    raw = example["output_logits"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}: output_logits must be a non-empty list, got {type(raw).__name__} {raw!r:.60}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
        raise TypeError(f"{where}: output_logits holds non-numeric values")
    x = np.asarray(raw, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{where}: output_logits holds non-finite values")
    if "output_tokens" in example:
        tokens = example["output_tokens"]
        if isinstance(tokens, str):
            tokens = ast.literal_eval(tokens)
        if not isinstance(tokens, list):
            raise TypeError(f"{where}: output_tokens must be a list, got {type(tokens).__name__}")
        if len(tokens) != x.size:
            raise ValueError(f"{where}: {x.size} logits but {len(tokens)} output_tokens")
    return x


def features(example: dict) -> list:
    """The eight FEATURES of one example, as floats."""
    x = token_logits(example)
    p10, p25 = np.percentile(x, [10, 25])
    return [float(v) for v in (x.mean(), x.std(), x.min(), p10, p25, np.mean(x < LOW_LOGIT),
                               x.size, x[:FIRST_K].mean())]


def feature_matrix(examples: list) -> np.ndarray:
    """(n_examples, len(FEATURES)) matrix in example order."""
    return np.array([features(ex) for ex in examples], dtype=float).reshape(len(examples), len(FEATURES))


def cv_auc(X: np.ndarray, y: np.ndarray) -> list:
    """Held-out ROC AUC of each of the 5 stratified folds."""
    aucs = []
    for tr, te in meta.cv_folds(y):
        model = meta.fit(make_pipeline(StandardScaler(), meta.logreg(None)), X[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1])))
    return aucs


def point_biserial(X: np.ndarray, y: np.ndarray) -> list:
    """[(feature, r, p)]; a constant feature has no defined correlation and raises."""
    out = []
    for j, name in enumerate(FEATURES):
        col = X[:, j]
        if np.ptp(col) == 0:
            raise ValueError(f"feature {name} is constant over {col.size} examples; correlation undefined")
        res = pointbiserialr(y, col)
        out.append((name, float(res.statistic), float(res.pvalue)))
    return out


def analyze(lang: str) -> list:
    """CSV rows (COLUMNS) for both tracks of `lang`."""
    examples = [ex for s in POOLED_SPLITS for ex in data.load_examples(lang, s)]
    X = feature_matrix(examples)
    rows = []
    for track in config.TRACKS:
        y = meta.labels(examples, track)
        base = {"lang": lang, "track": track, "n": int(y.size), "n_pos": int(y.sum())}
        aucs = cv_auc(X, y)
        stats = [(f"fold{k}", a) for k, a in enumerate(aucs, 1)]
        stats += [("mean", np.mean(aucs)), ("std", np.std(aucs, ddof=meta.STD_DDOF))]
        rows += [{**base, "kind": "cv_auc", "term": t, "estimate": f"{v:.6f}", "p_value": ""} for t, v in stats]
        rows += [{**base, "kind": "point_biserial", "term": name, "estimate": f"{r:.6f}", "p_value": f"{p:.4g}"}
                 for name, r, p in point_biserial(X, y)]
    return rows


def out_path(lang: str) -> Path:
    return config.RESULTS / "tables" / f"logits_{lang}.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chomps.logits",
                                 description="Token-logit features: 5-fold CV AUC and point-biserial correlations.")
    ap.add_argument("--lang", nargs="+", choices=config.LANGS, default=list(config.LANGS))
    args = ap.parse_args(argv)
    for lang in args.lang:
        rows = analyze(lang)
        path = out_path(lang)
        meta.write_csv(path, rows, COLUMNS)
        for track in config.TRACKS:
            tr = [r for r in rows if r["track"] == track]
            auc = {r["term"]: r["estimate"] for r in tr if r["kind"] == "cv_auc"}
            corr = sorted((r for r in tr if r["kind"] == "point_biserial"), key=lambda r: -abs(float(r["estimate"])))
            top = ", ".join(f"{r['term']} {float(r['estimate']):+.3f} (p={r['p_value']})" for r in corr[:3])
            print(f"[logits] {lang}/{track} n={tr[0]['n']} (y={tr[0]['n_pos']}): "
                  f"CV AUC {float(auc['mean']):.4f} ± {float(auc['std']):.4f}; top |r|: {top}")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
