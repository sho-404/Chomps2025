"""Offline tests of chomps.metrics: synthetic data only, no API and no ../datasets."""
import itertools
from collections import Counter

import numpy as np
import pytest
from scipy.stats import binomtest
from sklearn import metrics as skm

from chomps import metrics as M


def _case(seed: int, n: int = 240, p_pos: float = 0.8, flip: float = 0.2) -> tuple:
    """Gold ints and votes that flip the gold label with probability `flip`."""
    rng = np.random.default_rng(seed)
    gold = (rng.random(n) < p_pos).astype(int).tolist()
    pred = ["y" if (g == 1) != (rng.random() < flip) else "n" for g in gold]
    return gold, pred


def _ints(pred: list) -> list:
    return [1 if v == "y" else 0 for v in pred]


@pytest.mark.parametrize("seed", range(6))
def test_scores_match_sklearn(seed):
    gold, pred = _case(seed)
    s, p = M.scores(gold, pred), _ints(pred)
    assert s["macro_f1"] == pytest.approx(skm.f1_score(gold, p, average="macro"), abs=1e-12)
    assert s["macro_f1"] == (s["f1_y"] + s["f1_n"]) / 2
    assert s["accuracy"] == pytest.approx(skm.accuracy_score(gold, p), abs=1e-12)
    assert s["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(gold, p), abs=1e-12)
    assert s["mcc"] == pytest.approx(skm.matthews_corrcoef(gold, p), abs=1e-12)
    for label, cls in ((1, "y"), (0, "n")):
        assert s[f"precision_{cls}"] == pytest.approx(skm.precision_score(gold, p, pos_label=label), abs=1e-12)
        assert s[f"recall_{cls}"] == pytest.approx(skm.recall_score(gold, p, pos_label=label), abs=1e-12)
        assert s[f"f1_{cls}"] == pytest.approx(skm.f1_score(gold, p, pos_label=label), abs=1e-12)
    assert s["n"] == len(gold)
    assert list(s) == [*M.METRICS, "n"]


def test_zero_division_is_zero_and_both_classes_count():
    # A system that never votes y (the collapsed meta-learner of the paper) gets P = R = F1(y) = 0.
    gold, pred = [1, 1, 0, 1], ["n"] * 4
    s = M.scores(gold, pred)
    assert s["precision_y"] == s["recall_y"] == s["f1_y"] == s["mcc"] == 0.0
    assert s["macro_f1"] == skm.f1_score(gold, [0] * 4, average="macro", labels=[0, 1], zero_division=0)
    # A split with one gold class: the absent class contributes F1 = 0, like sklearn with labels=[0, 1].
    s = M.scores([1, 1], ["y", "y"])
    assert (s["f1_y"], s["f1_n"], s["macro_f1"]) == (1.0, 0.0, 0.5)
    assert s["macro_f1"] == skm.f1_score([1, 1], [1, 1], average="macro", labels=[0, 1], zero_division=0)


def test_paper_english_test_council_numbers():
    """The original English test council row reports Accuracy 0.8958, Precision 0.9695 and
    Recall 0.9095 on 240 examples with 210 positives. Only one confusion matrix rounds to
    those three numbers, and scores() reproduces them from it."""
    n, pos = 240, 210
    published = (0.8958, 0.9695, 0.9095)
    solutions = [(tp, fp, n - pos - fp, pos - tp)
                 for tp, fp in itertools.product(range(1, pos + 1), range(n - pos + 1))
                 if (round((tp + n - pos - fp) / n, 4), round(tp / (tp + fp), 4), round(tp / pos, 4)) == published]
    assert solutions == [(191, 6, 24, 19)]
    tp, fp, tn, fn = solutions[0]
    gold = [1] * (tp + fn) + [0] * (fp + tn)
    pred = ["y"] * tp + ["n"] * fn + ["y"] * fp + ["n"] * tn
    assert M.confusion(gold, pred) == {"tp": 191, "fp": 6, "tn": 24, "fn": 19}
    s = M.scores(gold, pred)
    assert (round(s["accuracy"], 4), round(s["precision_y"], 4), round(s["recall_y"], 4)) == published
    # The paper's "Macro-F1" cell for this row, 0.9386, is the F1 of class y. The macro-F1 is 0.7981.
    assert round(s["f1_y"], 4) == 0.9386
    assert round(s["macro_f1"], 4) == 0.7981


@pytest.mark.parametrize("gold,pred", [
    ([1, 0], ["y"]),             # length mismatch
    ([], []),                    # empty
    ([2, 0], ["y", "n"]),        # label outside 0/1
    (["y", "n"], ["y", "n"]),    # gold must be ints
    ([1.0, 0.0], ["y", "n"]),    # ... not floats
    ([True, False], ["y", "n"]),  # ... not bools
    ([np.bool_(True), 0], ["y", "n"]),
    ([1, 0], ["y", "N"]),        # vote outside y/n
    ([1, 0], [1, 0]),            # predictions must be votes
])
def test_invalid_input_raises(gold, pred):
    with pytest.raises(ValueError):
        M.scores(gold, pred)


def test_numpy_int_gold_is_accepted():
    gold = np.array([1, 0, 1, 1])
    assert M.scores(list(gold), ["y", "n", "n", "y"]) == M.scores([1, 0, 1, 1], ["y", "n", "n", "y"])


def test_unknown_metric_and_bad_n_boot_raise():
    gold, pred = _case(0, n=20)
    with pytest.raises(ValueError, match="unknown metric"):
        M.bootstrap_ci(gold, pred, metric="f1")
    with pytest.raises(ValueError, match="n_boot"):
        M.bootstrap_ci(gold, pred, n_boot=0)


def test_bootstrap_ci_is_the_percentile_interval_of_resampled_scores():
    gold, pred = _case(7, n=60)
    lo, hi = M.bootstrap_ci(gold, pred, "macro_f1", n_boot=300, seed=3)
    idx = np.random.default_rng(3).integers(0, 60, size=(300, 60))
    vals = [M.scores([gold[i] for i in row], [pred[i] for i in row])["macro_f1"] for row in idx]
    assert (lo, hi) == pytest.approx(tuple(np.percentile(vals, [2.5, 97.5])), abs=1e-12)
    assert M.bootstrap_ci(gold, pred, "macro_f1", n_boot=300, seed=3) == (lo, hi)
    assert lo <= M.scores(gold, pred)["macro_f1"] <= hi
    # A perfect predictor has a degenerate interval. 20/20 classes make a one-class resample
    # (macro-F1 0.5 under the both-classes convention) practically impossible.
    assert M.bootstrap_ci([1, 0] * 20, ["y", "n"] * 20, n_boot=200) == (1.0, 1.0)


def test_bootstrap_ci_mean_is_the_interval_of_the_mean_over_repeats():
    gold, a = _case(4, n=60)
    b, c = _case(4, n=60, flip=0.05)[1], _case(4, n=60, flip=0.4)[1]
    assert M.bootstrap_ci_mean(gold, [a], "macro_f1", 300, 2) == M.bootstrap_ci(gold, a, "macro_f1", 300, 2)
    lo, hi = M.bootstrap_ci_mean(gold, [a, b, c], "macro_f1", 300, 2)
    idx = np.random.default_rng(2).integers(0, 60, size=(300, 60))
    vals = [np.mean([M.scores([gold[i] for i in row], [p[i] for i in row])["macro_f1"] for p in (a, b, c)])
            for row in idx]
    assert (lo, hi) == pytest.approx(tuple(np.percentile(vals, [2.5, 97.5])), abs=1e-12)
    assert lo <= np.mean([M.scores(gold, p)["macro_f1"] for p in (a, b, c)]) <= hi
    with pytest.raises(ValueError):
        M.bootstrap_ci_mean(gold, [])
    with pytest.raises(ValueError):
        M.bootstrap_ci_mean(gold, [a, b[:-1]])


def test_paired_bootstrap_diff():
    gold, a = _case(1, n=80)
    _, b = _case(1, n=80, flip=0.35)  # same gold (same seed), different noise level
    assert M.paired_bootstrap_diff(gold, a, a, "macro_f1") == (0.0, 0.0, 0.0)
    mean_ab, lo_ab, hi_ab = M.paired_bootstrap_diff(gold, a, b, "macro_f1", n_boot=400, seed=5)
    mean_ba, lo_ba, hi_ba = M.paired_bootstrap_diff(gold, b, a, "macro_f1", n_boot=400, seed=5)
    assert (mean_ab, lo_ab, hi_ab) == pytest.approx((-mean_ba, -hi_ba, -lo_ba), abs=1e-12)
    idx = np.random.default_rng(5).integers(0, 80, size=(400, 80))
    diffs = [M.scores([gold[i] for i in r], [a[i] for i in r])["macro_f1"]
             - M.scores([gold[i] for i in r], [b[i] for i in r])["macro_f1"] for r in idx]
    assert (mean_ab, lo_ab, hi_ab) == pytest.approx((np.mean(diffs), *np.percentile(diffs, [2.5, 97.5])), abs=1e-12)


def test_mcnemar_counts_and_exact_p():
    gold = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    a = ["y", "y", "y", "y", "y", "n", "n", "n", "y", "n", "y", "n"]  # wrong on items 8 and 9
    b = ["n", "n", "n", "y", "y", "y", "n", "n", "y", "y", "y", "n"]  # wrong on items 0, 1, 2, 5 and 8
    n_b, n_c, p = M.mcnemar(gold, a, b)
    assert (n_b, n_c) == (4, 1)  # A right / B wrong: 0, 1, 2, 5; A wrong / B right: 9
    assert p == pytest.approx(binomtest(1, 5, 0.5).pvalue, abs=1e-12)
    assert M.mcnemar(gold, a, a) == (0, 0, 1.0)


def test_council_majority_matches_the_original_counter_rule():
    for votes in itertools.product("yn", repeat=3):
        expected = Counter(votes).most_common(1)[0][0]  # approach_2_multi_agent.py
        assert M.council_majority([{"x": v} for v in votes]) == {"x": expected}
    j1, j2 = {"a": "y", "b": "n"}, {"a": "n", "b": "n"}
    for bad in ([], [j1, j2], [j1, j2, {"a": "y"}], [j1, j2, {"a": "y", "b": "?"}]):
        with pytest.raises(ValueError):
            M.council_majority(bad)


def test_council_by_repeat_pairs_repeat_indices():
    a = {0: {"i": "y"}, 1: {"i": "n"}}
    b = {0: {"i": "y"}, 1: {"i": "n"}}
    c = {0: {"i": "n"}, 1: {"i": "y"}}
    assert M.council_by_repeat([a, b, c]) == {0: {"i": "y"}, 1: {"i": "n"}}
    with pytest.raises(ValueError, match="different repeat indices"):
        M.council_by_repeat([a, b, {0: {"i": "y"}}])


def test_consensus_majority_with_ties_to_y():
    preds = {0: {"a": "y", "b": "n", "c": "n"}, 1: {"a": "y", "b": "y", "c": "n"}, 2: {"a": "n", "b": "n", "c": "n"}}
    assert M.consensus(preds) == {"a": "y", "b": "n", "c": "n"}
    assert M.consensus({0: {"a": "y"}, 1: {"a": "n"}}) == {"a": "y"}
    with pytest.raises(ValueError):
        M.consensus({})
    with pytest.raises(ValueError):
        M.consensus({0: {"a": "y"}, 1: {"b": "y"}})


def test_align_orders_by_gold_and_rejects_partial_stores():
    gold = {"a": 1, "b": 0, "c": 1}
    assert M.align(gold, {"c": "y", "a": "n", "b": "n"}) == ([1, 0, 1], ["n", "n", "y"])
    with pytest.raises(ValueError, match="1 of 3 missing"):
        M.align(gold, {"a": "y", "b": "n"})
    with pytest.raises(ValueError, match="1 unknown"):
        M.align(gold, {"a": "y", "b": "n", "c": "y", "d": "n"})


def test_summarize_system():
    gold_list, p0 = _case(11, n=50)
    p1 = _case(11, n=50, flip=0.3)[1]
    p2 = _case(11, n=50, flip=0.1)[1]
    gold = {f"i{k}": g for k, g in enumerate(gold_list)}
    preds = {r: {f"i{k}": v for k, v in enumerate(p)} for r, p in enumerate((p0, p1, p2))}
    s = M.summarize_system(gold, preds, n_boot=300, seed=1)
    per = [M.scores(gold_list, p) for p in (p0, p1, p2)]
    assert (s["n"], s["n_repeats"], s["metric"]) == (50, 3, "macro_f1")
    assert s["per_repeat"] == dict(enumerate(per))
    assert s["mean"]["macro_f1"] == pytest.approx(np.mean([x["macro_f1"] for x in per]), abs=1e-12)
    assert s["std"]["macro_f1"] == pytest.approx(np.std([x["macro_f1"] for x in per], ddof=1), abs=1e-12)
    assert s["repeat0"] == per[0]
    assert s["ci_repeat0"] == M.bootstrap_ci(gold_list, p0, "macro_f1", 300, 1)
    assert s["ci_mean"] == M.bootstrap_ci_mean(gold_list, [p0, p1, p2], "macro_f1", 300, 1)
    g, c = M.align(gold, M.consensus(preds))
    assert s["consensus"] == M.scores(g, c)
    assert s["ci_consensus"] == M.bootstrap_ci(g, c, "macro_f1", 300, 1)
    single = M.summarize_system(gold, {0: preds[0]}, n_boot=10)
    assert single["n_repeats"] == 1 and single["std"]["macro_f1"] is None
    assert single["ci_mean"] == single["ci_repeat0"] == single["ci_consensus"]
    with pytest.raises(ValueError, match="repeat 0"):
        M.summarize_system(gold, {1: preds[1]})
