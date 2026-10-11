"""Classification metrics, resampling statistics and vote aggregation (pure: no IO).

Conventions, fixed here so that every table uses the same definitions:

* Gold labels are ints (1 = hallucination present); predictions are votes
  "y"/"n"; the positive class is y/1. Inputs are validated strictly: unequal
  lengths, empty input, a gold value that is not the int 0 or 1 (Python or
  numpy int; bools and floats such as 1.0 are rejected) or a prediction
  outside {"y", "n"} raises.
* A ratio whose denominator is zero is 0.0. This is sklearn's
  `zero_division=0` and the rule of the original evaluate_approaches.py (a
  classifier that never predicts y has precision 0.0000). Both classes always
  count, i.e. per-class and macro scores equal sklearn's with labels=[0, 1].
* macro-F1, the official task metric, is the unweighted mean of the F1 of
  class y and the F1 of class n. `f1_y` alone is the positive-class F1 that
  the original evaluation script printed as "F1 Score".
* Bootstrap: resample the examples with replacement `n_boot` times from
  `numpy.random.default_rng(seed)`, and take the 2.5th and 97.5th
  percentiles of the metric. The paired version uses the same resamples for
  both systems; the CI of a mean over repeats applies each resample to every
  repeat and averages, so it is the interval of the number the tables print.

Votes stores are loaded with `judge.load_votes(path) -> {repeat: {index: vote}}`.
That function is the single reader of the store format, so this module does
not read files itself. Everything here works on those mappings or on aligned
lists.
"""
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar

METRICS = ("accuracy", "balanced_accuracy", "precision_y", "recall_y", "f1_y",
           "precision_n", "recall_n", "f1_n", "macro_f1", "mcc")
VOTES = ("y", "n")


# --- point metrics -------------------------------------------------------------

def _is_label(g) -> bool:
    """The int 0 or 1 (numpy ints too). bool is an int subclass, so it is excluded explicitly."""
    return isinstance(g, (int, np.integer)) and not isinstance(g, bool) and g in (0, 1)


def _as_arrays(gold, pred) -> tuple:
    """Validate one aligned (gold, pred) pair; boolean arrays, True = y/1."""
    gold, pred = list(gold), list(pred)
    if len(gold) != len(pred):
        raise ValueError(f"gold and pred differ in length: {len(gold)} vs {len(pred)}")
    if not gold:
        raise ValueError("empty gold/pred")
    bad = {repr(g) for g in gold if not _is_label(g)}
    if bad:
        raise ValueError(f"gold labels must be the ints 0/1 (not bools or floats), got {sorted(bad)[:6]}")
    if not set(pred) <= set(VOTES):
        raise ValueError(f"predictions must be votes 'y'/'n', got {sorted(map(repr, set(pred)))[:6]}")
    return np.asarray(gold, dtype=int) == 1, np.asarray(pred, dtype=str) == "y"


def _counts(g: np.ndarray, p: np.ndarray) -> dict:
    """Confusion counts along the last axis (1-D: one sample; 2-D: one per bootstrap row)."""
    return {"tp": (g & p).sum(-1), "fp": (~g & p).sum(-1),
            "tn": (~g & ~p).sum(-1), "fn": (g & ~p).sum(-1)}


def _ratio(num, den) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den != 0)


def _from_counts(tp, fp, tn, fn) -> dict:
    """Every metric from confusion counts, elementwise (scalars or bootstrap arrays)."""
    tp, fp, tn, fn = (np.asarray(x, dtype=float) for x in (tp, fp, tn, fn))
    m = {"accuracy": _ratio(tp + tn, tp + fp + tn + fn),
         "precision_y": _ratio(tp, tp + fp), "recall_y": _ratio(tp, tp + fn),
         "f1_y": _ratio(2 * tp, 2 * tp + fp + fn),
         "precision_n": _ratio(tn, tn + fn), "recall_n": _ratio(tn, tn + fp),
         "f1_n": _ratio(2 * tn, 2 * tn + fn + fp)}
    m["balanced_accuracy"] = (m["recall_y"] + m["recall_n"]) / 2
    m["macro_f1"] = (m["f1_y"] + m["f1_n"]) / 2
    m["mcc"] = _ratio(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {k: m[k] for k in METRICS}


def _check_metric(metric: str):
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")


def confusion(gold, pred) -> dict:
    """{"tp", "fp", "tn", "fn"} with y/1 as the positive class."""
    return {k: int(v) for k, v in _counts(*_as_arrays(gold, pred)).items()}


def scores(gold, pred) -> dict:
    """accuracy, balanced_accuracy, precision/recall/F1 of y and of n, macro_f1, mcc, n."""
    c = confusion(gold, pred)
    out = {k: float(v) for k, v in _from_counts(**c).items()}
    out["n"] = sum(c.values())
    return out


# --- resampling ------------------------------------------------------------------

def _resample(n: int, n_boot: int, seed: int) -> np.ndarray:
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _boot(g: np.ndarray, p: np.ndarray, idx: np.ndarray, metric: str) -> np.ndarray:
    return _from_counts(**_counts(g[idx], p[idx]))[metric]


def bootstrap_ci(gold, pred, metric: str = "macro_f1", n_boot: int = 2000, seed: int = 0) -> tuple:
    """(lo, hi): 95 % percentile interval of `metric` over example resamples."""
    _check_metric(metric)
    g, p = _as_arrays(gold, pred)
    lo, hi = np.percentile(_boot(g, p, _resample(len(g), n_boot, seed), metric), [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_mean(gold, preds, metric: str = "macro_f1", n_boot: int = 2000, seed: int = 0) -> tuple:
    """(lo, hi): 95 % percentile interval of the mean of `metric` over several
    prediction lists for the same examples (a system's repeats). Each resample
    of examples is applied to every list and contributes the mean of their
    scores, so this is the interval of the mean over repeats, not of one run.
    With a single list it equals bootstrap_ci (same seed, same resamples)."""
    _check_metric(metric)
    preds = list(preds)
    if not preds:
        raise ValueError("bootstrap_ci_mean needs at least one prediction list")
    arrays = [_as_arrays(gold, p) for p in preds]
    g = arrays[0][0]
    idx = _resample(len(g), n_boot, seed)
    stat = np.mean([_boot(g, p, idx, metric) for _, p in arrays], axis=0)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_diff(gold, pred_a, pred_b, metric: str = "macro_f1", n_boot: int = 2000,
                          seed: int = 0) -> tuple:
    """(mean_diff, lo, hi) of metric(a) - metric(b), with each resample applied to
    both systems. mean_diff is the mean over the resamples, as specified; the
    observed difference is scores(a)[metric] - scores(b)[metric]."""
    _check_metric(metric)
    g, a = _as_arrays(gold, pred_a)
    _, b = _as_arrays(gold, pred_b)
    idx = _resample(len(g), n_boot, seed)
    d = _boot(g, a, idx, metric) - _boot(g, b, idx, metric)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def mcnemar(gold, pred_a, pred_b) -> tuple:
    """(b, c, p): b = examples A gets right and B gets wrong, c = the reverse,
    and p is the exact (binomial) McNemar p-value from statsmodels."""
    g, a = _as_arrays(gold, pred_a)
    _, bb = _as_arrays(gold, pred_b)
    ok_a, ok_b = a == g, bb == g
    table = [[int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
             [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())]]
    return table[0][1], table[1][0], float(_sm_mcnemar(table, exact=True).pvalue)


# --- aligning and aggregating votes -------------------------------------------------

def _check_vote_map(votes: dict, what: str):
    bad = {v for v in votes.values() if v not in VOTES}
    if bad:
        raise ValueError(f"{what}: votes must be 'y'/'n', got {sorted(map(repr, bad))[:6]}")


def align(gold_map: dict, pred_map: dict) -> tuple:
    """(gold list, pred list) in the gold map's order. The predictions must cover
    exactly the gold examples: a store with too few or extra examples raises
    an error instead of being scored on a subset."""
    missing = [i for i in gold_map if i not in pred_map]
    extra = [i for i in pred_map if i not in gold_map]
    if missing or extra:
        raise ValueError(f"predictions do not match the gold examples: {len(missing)} of {len(gold_map)} "
                         f"missing (e.g. {missing[:3]}), {len(extra)} unknown (e.g. {extra[:3]})")
    return [gold_map[i] for i in gold_map], [pred_map[i] for i in gold_map]


def council_majority(judge_votes: list) -> dict:
    """{index: vote} majority of judges' {index: vote} maps taken at the same repeat.
    The original council took Counter(votes).most_common(1) over three votes,
    which is a strict majority. An even-sized council would need a tie rule
    that the original never had, so it raises."""
    k = len(judge_votes)
    if k == 0 or k % 2 == 0:
        raise ValueError(f"a council needs an odd number of judges, got {k}")
    keys = set(judge_votes[0])
    for j, votes in enumerate(judge_votes):
        _check_vote_map(votes, f"judge {j}")
        if set(votes) != keys:
            raise ValueError(f"judge {j} covers {len(votes)} examples, judge 0 covers {len(keys)}; "
                             f"symmetric difference e.g. {sorted(set(votes) ^ keys)[:3]}")
    return {i: "y" if 2 * sum(v[i] == "y" for v in judge_votes) > k else "n" for i in judge_votes[0]}


def council_by_repeat(judges: list) -> dict:
    """{repeat: {index: vote}}: the council at repeat r combines every judge's
    repeat r. Repeats are independent runs, so pairing them by index is
    arbitrary but consistent. The judges must have the same repeat indices."""
    reps = [sorted(j) for j in judges]
    if not reps or any(r != reps[0] for r in reps):
        raise ValueError(f"council members have different repeat indices: {reps}")
    return {r: council_majority([j[r] for j in judges]) for r in reps[0]}


def consensus(preds_by_repeat: dict) -> dict:
    """{index: vote}: majority over a system's repeats. A tie (even number of
    repeats) counts as "y", the same direction as the original parser's
    fallback (an unclear reply counts as a mistake)."""
    if not preds_by_repeat:
        raise ValueError("no repeats to take a consensus over")
    maps = [preds_by_repeat[r] for r in sorted(preds_by_repeat)]
    keys = set(maps[0])
    for r, m in zip(sorted(preds_by_repeat), maps):
        _check_vote_map(m, f"repeat {r}")
        if set(m) != keys:
            raise ValueError(f"repeat {r} covers {len(m)} examples, the first repeat covers {len(keys)}")
    return {i: "y" if 2 * sum(m[i] == "y" for m in maps) >= len(maps) else "n" for i in maps[0]}


def summarize_system(gold_map: dict, preds_by_repeat: dict, metric: str = "macro_f1",
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Scores of one system over its repeats.

    Returns n, n_repeats, per_repeat {r: scores}, mean and std {metric: value}
    across repeats, ci_mean, repeat0 scores and ci_repeat0, consensus scores
    and ci_consensus (the CIs are 95 % bootstrap intervals of `metric`). std
    is the sample std (ddof=1). With a single repeat (for example the
    meta-learner) it is None, because run-to-run spread was not measured.
    ci_mean is the interval of the mean over repeats (bootstrap_ci_mean), the
    one to print next to the mean; ci_repeat0 is that of repeat 0, a single
    concrete run as the paper reports it; ci_consensus that of the majority
    over repeats. With one repeat the three coincide.
    """
    _check_metric(metric)
    if 0 not in preds_by_repeat:
        raise ValueError(f"no repeat 0 among repeats {sorted(preds_by_repeat)}")
    reps = sorted(preds_by_repeat)
    aligned = {r: align(gold_map, preds_by_repeat[r]) for r in reps}
    per = {r: scores(*aligned[r]) for r in reps}
    mean = {m: float(np.mean([per[r][m] for r in reps])) for m in METRICS}
    std = {m: float(np.std([per[r][m] for r in reps], ddof=1)) if len(reps) > 1 else None for m in METRICS}
    cons = align(gold_map, consensus(preds_by_repeat))
    return {"n": per[0]["n"], "n_repeats": len(reps), "metric": metric, "per_repeat": per,
            "mean": mean, "std": std,
            "ci_mean": bootstrap_ci_mean(aligned[0][0], [aligned[r][1] for r in reps], metric, n_boot, seed),
            "repeat0": per[0], "ci_repeat0": bootstrap_ci(*aligned[0], metric, n_boot, seed),
            "consensus": scores(*cons), "ci_consensus": bootstrap_ci(*cons, metric, n_boot, seed)}
