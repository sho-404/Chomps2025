"""Meta-learner over the council's votes (DESIGN.md, "Meta-learner").

The competition system (approach 3) fed the three council judges' votes
(y=1, n=0) to sklearn's default LogisticRegression, trained it on the train
split and predicted validation and test. This module reproduces that protocol
(`original`), adds a cross-validated one (`cv`), optional class balancing and
optional token-logit features, reads the votes through `judge.load_votes` and
writes its predictions back into the votes store as one more system.

Protocol decisions
* Features: the votes of `--judges` (in that order) at repeat r. Every example
  of every split needs a vote from every judge; a missing or unknown index
  raises. `--logits` appends the eight `logits.FEATURES`, standardised inside
  every fit (scaler fitted on the fitting rows only); votes stay 0/1, so each
  judge's beta keeps the "log-odds per y vote" reading of the original model.
* Repeats: `--repeats R` fits one meta-learner per judge repeat r = 0..R-1
  (repeat r of every judge, the pairing metrics.council_by_repeat uses).
  R=1 is DESIGN's default (repeat 0). The votes store only holds repeats
  0..R-1, so a single non-zero repeat cannot be stored and is not offered.
* Model: LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs",
  class_weight=None|"balanced", random_state=42). l1_ratio=0.0 is the L2
  penalty: sklearn >= 1.8 deprecates `penalty`, so this is DESIGN's
  penalty="l2" model (and the original LogisticRegression(random_state=42))
  without the FutureWarning. A ConvergenceWarning is raised as an error.
* original: fit on train; predict validation and test.
  cv: stratified 5-fold (shuffle, random_state=42) on train+validation
  pooled -> macro-F1 of every held-out fold, mean and sample std (ddof=1, as
  metrics.summarize_system); then fit on the pooled set and predict test.
* Coefficients: beta_0 (intercept) and one beta per feature from the fit on the
  whole fitting set; 95 % CI = 2.5/97.5 percentiles over `n_boot` refits on
  stratified bootstrap resamples (each class resampled with replacement at its
  own size, so every refit sees both classes and the class prior behind beta_0
  is held fixed).
* Predictions: y iff P(y) > 0.5 (`predict`). Stored as judge
  `meta-{protocol}-{cw}[-logits]` under the same fmt/temp as the input votes
  (DESIGN's fmt=judge, temp=default for the canonical council), one row per
  (example, repeat) in the judge-row schema, like judge.py's AUTO rows (no
  call: tokens and latency 0, cached=False, finish_reason=None) with
  `raw` = P(y) and `model` = provenance (council, inputs, repeat). Files are
  rewritten in full; re-running rewrites the same bytes.
* Summary: ../results/tables/meta_{lang}_{track}_{condition}_{protocol}_{cw}[_logits].csv,
  tidy: one row per coefficient (kind=coef), CV fold (cv_macro_f1) and
  predicted split (macro_f1), per repeat.
"""
import argparse
import re
import csv
import importlib
import io
import json
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data, metrics

PROTOCOLS = ("original", "cv")
CLASS_WEIGHTS = {"none": None, "balanced": "balanced"}
FMTS = ("judge", "ft")
TEMPS = ("default", "0", "1.0")
N_SPLITS = 5
RANDOM_STATE = 42
STD_DDOF = 1
SUMMARY_COLUMNS = ("lang", "track", "condition", "protocol", "class_weight", "features", "judges",
                   "fmt", "temp", "repeat", "kind", "term", "estimate", "ci_lo", "ci_hi", "n", "n_pos",
                   "n_boot", "seed")


# --- model -------------------------------------------------------------------------

def logreg(class_weight=None) -> LogisticRegression:
    """DESIGN's meta-learner (L2 via l1_ratio=0.0, C=1.0, lbfgs, random_state=42)."""
    return LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", class_weight=class_weight,
                              random_state=RANDOM_STATE)


def fit(model, X, y):
    """`model.fit` with ConvergenceWarning promoted to an error (no unconverged betas)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return model.fit(X, y)


def make_model(class_weight: str, n_votes: int, n_logit_features: int = 0):
    """Votes only: the bare logistic regression. With logits: the votes pass through
    unchanged (first, keeping column order) and the logit features are standardised."""
    lr = logreg(CLASS_WEIGHTS[class_weight])
    if n_logit_features == 0:
        return lr
    prep = ColumnTransformer([
        ("votes", "passthrough", list(range(n_votes))),
        ("logits", StandardScaler(), list(range(n_votes, n_votes + n_logit_features))),
    ])
    return Pipeline([("prep", prep), ("lr", lr)])


def coefficients(model) -> np.ndarray:
    """[beta_0, beta_1, ...] in feature order."""
    lr = model[-1] if isinstance(model, Pipeline) else model
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def bootstrap_coefficients(make, X, y, n_boot: int, seed: int) -> tuple:
    """Percentile 95 % CI of the coefficients over `n_boot` refits on stratified
    bootstrap resamples. Returns (lo, hi) arrays aligned with `coefficients`."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == c) for c in (0, 1)]
    draws = np.empty((n_boot, X.shape[1] + 1))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in by_class])
        draws[b] = coefficients(fit(make(), X[idx], y[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return lo, hi


def labels(examples: list, track: str) -> np.ndarray:
    """Gold 0/1 labels of `track` (1 = hallucination present) in example order."""
    if track not in config.TRACKS:
        raise ValueError(f"unknown track {track!r}")
    y = np.array([ex["labels"][track] for ex in examples])
    if y.size == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{track} labels must be a non-empty list of 0/1, got {sorted(set(y.tolist()))}")
    return y.astype(int)


def cv_folds(y: np.ndarray) -> list:
    """Stratified 5-fold splits (shuffled, random_state=42). Raises when a class
    has fewer members than folds (some fold would miss that class)."""
    counts = np.bincount(y, minlength=2)
    if counts.min() < N_SPLITS:
        raise ValueError(f"stratified {N_SPLITS}-fold CV needs >= {N_SPLITS} examples per class, "
                         f"got n={counts[0]}, y={counts[1]}")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(np.zeros((y.size, 1)), y))


def macro_f1(y_true, y_pred) -> float:
    """The official task metric, with metrics.scores' definitions."""
    return metrics.scores([int(v) for v in y_true], ["y" if v == 1 else "n" for v in y_pred])["macro_f1"]


# --- votes store ---------------------------------------------------------------------

def _judge_store():
    """chomps.judge owns the votes-store layout. Imported lazily: it imports llm,
    which opens the response cache, and logits.py imports this module."""
    return importlib.import_module(f"{__package__}.judge")


def load_vote_map(lang, track, split, condition, judge, fmt, temp, repeat) -> dict:
    """{index: 1 (y) | 0 (n)} of one judge at one repeat, via judge.load_votes."""
    store = _judge_store()
    path = store.votes_path(lang, track, split, condition, judge, fmt, temp)
    if not path.exists():
        raise FileNotFoundError(f"no votes for {judge} at {path} (run chomps.judge on this split first)")
    by_repeat = store.load_votes(path)
    if repeat not in by_repeat:
        raise ValueError(f"{path}: has repeats 0..{len(by_repeat) - 1}, repeat {repeat} is needed")
    out = {}
    for idx, vote in by_repeat[repeat].items():
        if vote not in ("y", "n"):
            raise ValueError(f"{path}: vote {vote!r} for {idx} is not 'y'/'n'")
        out[idx] = 1 if vote == "y" else 0
    return out


def vote_matrix(examples: list, votes: dict, judges: list, where: str) -> np.ndarray:
    """(n_examples, n_judges) 0/1 matrix in example order; every example needs a vote
    from every judge and no judge may vote on an index outside the split."""
    order = [ex["index"] for ex in examples]
    known = set(order)
    for j in judges:
        missing = [i for i in order if i not in votes[j]]
        unknown = sorted(set(votes[j]) - known)
        if missing or unknown:
            raise ValueError(f"{where} [{j}]: {len(missing)} examples without a vote {missing[:3]}, "
                             f"{len(unknown)} votes for unknown indices {unknown[:3]}")
    return np.array([[votes[j][i] for j in judges] for i in order], dtype=float).reshape(len(order), len(judges))


# --- outputs -------------------------------------------------------------------------

def system_name(protocol: str, class_weight: str, use_logits: bool, tag: str = "") -> str:
    """meta-{protocol}-{cw}[-logits][-{tag}]; the tag names a council other than the canonical one."""
    if tag and not re.fullmatch(r"[a-z0-9]+", tag):
        raise ValueError(f"tag must be lowercase letters/digits, got {tag!r}")
    return f"meta-{protocol}-{class_weight}" + ("-logits" if use_logits else "") + (f"-{tag}" if tag else "")


def summary_path(lang, track, condition, protocol, class_weight, use_logits, tag: str = "") -> Path:
    suffix = ("_logits" if use_logits else "") + (f"_{tag}" if tag else "")
    return config.RESULTS / "tables" / f"meta_{lang}_{track}_{condition}_{protocol}_{class_weight}{suffix}.csv"


def write_text(path: Path, text: str):
    """Replace `path` atomically (temp file + rename): no crash leaves a partial file.
    The temp name carries the pid; a plain open() keeps the umask's permissions
    (mkstemp would make the result owner-only, 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_csv(path: Path, rows: list, columns: tuple):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buf.getvalue())


# --- one configuration ---------------------------------------------------------------

def run(lang: str, track: str, condition: str, judges: list, *, fmt: str = "judge", temp: str = "default",
        protocol: str = "original", class_weight: str = "none", repeats: int = 1, use_logits: bool = False,
        n_boot: int = 2000, seed: int = 0, tag: str = "") -> list:
    """Fit and evaluate one configuration at repeats 0..repeats-1, write its votes-store
    files and summary CSV, and return the summary rows."""
    if lang not in config.LANGS or track not in config.TRACKS:
        raise ValueError(f"unknown lang/track {lang!r}/{track!r}")
    if protocol not in PROTOCOLS or class_weight not in CLASS_WEIGHTS or fmt not in FMTS or temp not in TEMPS:
        raise ValueError(f"bad protocol/class_weight/fmt/temp: {protocol!r} {class_weight!r} {fmt!r} {temp!r}")
    judges = list(judges)
    if not judges or len(set(judges)) != len(judges):
        raise ValueError(f"judges must be non-empty and distinct, got {judges}")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"repeats must be a positive int, got {repeats!r}")

    examples = {s: data.load_examples(lang, s) for s in config.SPLITS}
    ys = {s: labels(examples[s], track) for s in config.SPLITS}
    extra, terms = {}, ["intercept", *judges]
    if use_logits:
        from . import logits
        extra = {s: logits.feature_matrix(examples[s]) for s in config.SPLITS}
        terms += [f"logit:{f}" for f in logits.FEATURES]
    n_extra = len(terms) - 1 - len(judges)

    def make():
        return make_model(class_weight, len(judges), n_extra)

    targets = ("validation", "test") if protocol == "original" else ("test",)
    name = system_name(protocol, class_weight, use_logits, tag)
    features = "votes+logits" if use_logits else "votes"
    base = {"lang": lang, "track": track, "condition": condition, "protocol": protocol,
            "class_weight": class_weight, "features": features, "judges": ";".join(judges),
            "fmt": fmt, "temp": temp, "n_boot": n_boot, "seed": seed}

    def row(r, kind, term, value, n, n_pos, lo=None, hi=None):
        num = lambda v: "" if v is None else f"{float(v):.6f}"  # noqa: E731
        return {**base, "repeat": r, "kind": kind, "term": term, "estimate": num(value),
                "ci_lo": num(lo), "ci_hi": num(hi), "n": int(n), "n_pos": int(n_pos)}

    store_rows = {s: [] for s in targets}
    summary = []
    for r in range(repeats):
        X = {}
        for s in config.SPLITS:
            votes = {j: load_vote_map(lang, track, s, condition, j, fmt, temp, r) for j in judges}
            V = vote_matrix(examples[s], votes, judges, f"{lang}/{track}/{s}/{condition} repeat {r}")
            X[s] = np.hstack([V, extra[s]]) if use_logits else V
        if protocol == "original":
            X_fit, y_fit, fit_on = X["train"], ys["train"], "train"
        else:
            X_fit = np.vstack([X["train"], X["validation"]])
            y_fit = np.concatenate([ys["train"], ys["validation"]])
            fit_on = "train+validation"

        model = fit(make(), X_fit, y_fit)
        beta = coefficients(model)
        lo, hi = bootstrap_coefficients(make, X_fit, y_fit, n_boot, seed)
        summary += [row(r, "coef", t, b, y_fit.size, y_fit.sum(), l, h) for t, b, l, h in zip(terms, beta, lo, hi)]
        fold_scores = []
        if protocol == "cv":
            for k, (tr, te) in enumerate(cv_folds(y_fit), 1):
                fold_model = fit(make(), X_fit[tr], y_fit[tr])
                fold_scores.append(macro_f1(y_fit[te], fold_model.predict(X_fit[te])))
                summary.append(row(r, "cv_macro_f1", f"fold{k}", fold_scores[-1], te.size, y_fit[te].sum()))
            summary.append(row(r, "cv_macro_f1", "mean", np.mean(fold_scores), y_fit.size, y_fit.sum()))
            summary.append(row(r, "cv_macro_f1", "std", np.std(fold_scores, ddof=STD_DDOF), y_fit.size, y_fit.sum()))
        provenance = (f"sklearn LogisticRegression(l2,C=1.0,lbfgs,class_weight={class_weight}); "
                      f"protocol={protocol}; fit={fit_on}; judges={','.join(judges)}; fmt={fmt}; "
                      f"temp={temp}; repeat={r}; features={features}")
        split_scores = {}
        for s in targets:
            p_y = model.predict_proba(X[s])[:, list(model.classes_).index(1)]
            pred = model.predict(X[s])
            split_scores[s] = macro_f1(ys[s], pred)
            summary.append(row(r, "macro_f1", s, split_scores[s], ys[s].size, ys[s].sum()))
            store_rows[s] += [{"index": ex["index"], "repeat": r, "vote": "y" if v == 1 else "n", "parse_ok": True,
                               "raw": f"p_y={p:.6f}", "prompt_tokens": 0, "completion_tokens": 0,
                               "reasoning_tokens": 0, "latency_s": 0.0, "cached": False, "model": provenance,
                               "finish_reason": None}
                              for ex, v, p in zip(examples[s], pred, p_y)]
        _print_run(lang, track, condition, name, r, fit_on, y_fit, terms, beta, lo, hi, fold_scores, split_scores)

    judge_store = _judge_store()
    for s in targets:
        pos = {ex["index"]: k for k, ex in enumerate(examples[s])}
        rows = sorted(store_rows[s], key=lambda q: (pos[q["index"]], q["repeat"]))
        path = judge_store.votes_path(lang, track, s, condition, name, fmt, temp)
        write_text(path, "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows))
        print(f"  -> {path}")
    path = summary_path(lang, track, condition, protocol, class_weight, use_logits, tag)
    write_csv(path, summary, SUMMARY_COLUMNS)
    print(f"  -> {path}")
    return summary


def _print_run(lang, track, condition, name, r, fit_on, y_fit, terms, beta, lo, hi, fold_scores, split_scores):
    print(f"[meta] {lang}/{track}/{condition} {name} repeat {r}: fit on {fit_on}, "
          f"n={y_fit.size} (y={int(y_fit.sum())})")
    width = max(len(t) for t in terms)
    for t, b, l, h in zip(terms, beta, lo, hi):
        print(f"  beta {t:<{width}} {b:+.4f}  95% CI [{l:+.4f}, {h:+.4f}]")
    if fold_scores:
        folds = " ".join(f"{f:.4f}" for f in fold_scores)
        print(f"  cv macro-F1 {np.mean(fold_scores):.4f} ± {np.std(fold_scores, ddof=STD_DDOF):.4f} (folds {folds})")
    for s, f in split_scores.items():
        print(f"  {s} macro-F1 {f:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chomps.meta",
                                 description="Logistic-regression meta-learner over council votes.")
    ap.add_argument("--lang", required=True, choices=config.LANGS)
    ap.add_argument("--track", required=True, choices=config.TRACKS)
    ap.add_argument("--condition", nargs="+", required=True, help="retrieval condition(s); fluency: none")
    ap.add_argument("--judges", nargs="+", required=True, help="council members, in coefficient order")
    ap.add_argument("--fmt", default="judge", choices=FMTS, help="prompt format of the input votes")
    ap.add_argument("--temp", default="default", choices=TEMPS, help="temperature of the input votes")
    ap.add_argument("--class-weight", nargs="+", default=list(CLASS_WEIGHTS), choices=list(CLASS_WEIGHTS))
    ap.add_argument("--protocol", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    ap.add_argument("--repeats", type=int, default=1,
                    help="fit one meta-learner on each judge repeat 0..R-1 (default 1: repeat 0)")
    ap.add_argument("--logits", action="store_true", help="append the eight logits.FEATURES to the votes")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap refits for the coefficient CIs")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--tag", default="", help="council tag appended to the system name (e.g. fluft for the dedicated fluency council)")
    args = ap.parse_args(argv)
    for condition in args.condition:
        for protocol in args.protocol:
            for cw in args.class_weight:
                run(args.lang, args.track, condition, args.judges, fmt=args.fmt, temp=args.temp,
                    protocol=protocol, class_weight=cw, repeats=args.repeats, use_logits=args.logits,
                    n_boot=args.n_boot, seed=args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
