"""Offline tests for chomps.logits: synthetic examples only (no dataset, no API)."""
import csv
import json
import math

import numpy as np
import pytest
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chomps import config, data, logits, meta


def ex(index, values, tokens=None, fact=0, flu=0):
    return {"index": index, "output_logits": values,
            "output_tokens": [f"t{k}" for k in range(len(values))] if tokens is None else tokens,
            "labels": {"factuality": fact, "fluency": flu}}


def named(example):
    return dict(zip(logits.FEATURES, logits.features(example)))


def test_features_known_values():
    f = named(ex("a", [10.0, 20.0, 30.0, 40.0]))
    assert f == pytest.approx({"mean": 25.0, "std": math.sqrt(125.0),  # population std
                               "min": 10.0, "p10": 13.0, "p25": 17.5,  # linear interpolation
                               "frac_below_15": 0.25, "n_tokens": 4.0, "mean_first_10": 25.0})


def test_mean_first_10_and_single_token():
    f = named(ex("a", [20.0] * 10 + [40.0] * 5))
    assert (f["mean_first_10"], f["n_tokens"], f["mean"]) == pytest.approx((20.0, 15.0, 80 / 3))
    one = named(ex("b", [12.5]))
    assert (one["std"], one["frac_below_15"], one["p10"], one["mean_first_10"]) == (0.0, 1.0, 12.5, 12.5)


def test_string_serialised_rows_decode_to_the_same_features():
    """The 20 Spanish train rows store both lists as strings (JSON logits, Python-repr tokens)."""
    values, tokens = [22.38, 24.27, 9.5, 31.0], ["En", "Ġel", "Ġart", "'s"]
    assert logits.features(ex("es-train-0", json.dumps(values), tokens=repr(tokens))) == \
        logits.features(ex("es-train-0", values, tokens=tokens))


@pytest.mark.parametrize("values, tokens, error", [
    ([], [], ValueError),                          # empty
    (None, [], ValueError),                        # not a list
    ("not json", [], ValueError),                  # undecodable string
    ('{"a": 1}', [], ValueError),                  # decodes to a non-list
    ("[20.0, NaN]", ["a", "b"], ValueError),       # non-finite after decoding
    ([20.0, float("inf")], ["a", "b"], ValueError),
    ([20.0, "21"], ["a", "b"], TypeError),         # non-numeric element
    ([True, 21.0], ["a", "b"], TypeError),         # bools are not logits
    ([20.0, 21.0], ["a"], ValueError),             # length differs from the tokens
    ([20.0, 21.0], 5, TypeError),                  # tokens not a list
])
def test_bad_logits_raise(values, tokens, error):
    with pytest.raises(error):
        logits.features(ex("x", values, tokens=tokens))


def test_missing_logits_field_raises():
    with pytest.raises(KeyError):
        logits.features({"index": "x", "output_tokens": []})


def test_cv_auc_matches_a_manual_computation():
    rng = np.random.default_rng(0)
    y = (rng.random(120) < 0.7).astype(int)
    X = rng.normal(size=(120, len(logits.FEATURES))) + y[:, None] * np.linspace(0, 1, len(logits.FEATURES))
    manual = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(X[tr], y[tr])
        manual.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    assert logits.cv_auc(X, y) == pytest.approx(manual)


def test_point_biserial_matches_scipy_and_rejects_constant_features():
    rng = np.random.default_rng(1)
    y = (rng.random(50) < 0.5).astype(int)
    X = rng.normal(size=(50, len(logits.FEATURES))) + y[:, None]
    got = logits.point_biserial(X, y)
    assert [g[0] for g in got] == list(logits.FEATURES)
    for j, (_, r, p) in enumerate(got):
        ref = pointbiserialr(y, X[:, j])
        assert (r, p) == pytest.approx((ref.statistic, ref.pvalue))
    X[:, 3] = 7.0
    with pytest.raises(ValueError, match="p10 is constant"):
        logits.point_biserial(X, y)


def test_cv_needs_five_examples_per_class():
    y = np.array([1] * 20 + [0] * 4)
    with pytest.raises(ValueError, match="per class"):
        logits.cv_auc(np.random.default_rng(2).normal(size=(24, len(logits.FEATURES))), y)


def synthetic_split(split, n, seed):
    """Factuality: hallucinated answers have lower logits; fluency: random labels."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fact, flu = int(rng.random() < 0.7), int(rng.random() < 0.3)
        k = int(rng.integers(3, 60))
        out.append(ex(f"en-{split}-{i}", rng.normal(28 - 8 * fact, 4, size=k).tolist(), fact=fact, flu=flu))
    return out


@pytest.fixture
def env(monkeypatch, tmp_path):
    splits = {"train": synthetic_split("train", 60, 1), "validation": synthetic_split("validation", 90, 2),
              "test": synthetic_split("test", 50, 3)}
    loaded = []

    def load(lang, split):
        loaded.append((lang, split))
        return [dict(e) for e in splits[split]]

    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(data, "load_examples", load)
    return splits, loaded


def test_cli_writes_both_tracks_from_train_and_validation_only(env, tmp_path):
    splits, loaded = env
    logits.main(["--lang", "english"])
    assert sorted(loaded) == [("english", "train"), ("english", "validation")]
    path = tmp_path / "results" / "tables" / "logits_english.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert tuple(rows[0]) == logits.COLUMNS
    pooled = splits["train"] + splits["validation"]
    X = logits.feature_matrix(pooled)
    for track in config.TRACKS:
        y = meta.labels(pooled, track)
        tr = [r for r in rows if r["track"] == track]
        auc = [r for r in tr if r["kind"] == "cv_auc"]
        assert [r["term"] for r in auc] == ["fold1", "fold2", "fold3", "fold4", "fold5", "mean", "std"]
        folds = logits.cv_auc(X, y)
        assert [float(r["estimate"]) for r in auc] == pytest.approx(
            folds + [np.mean(folds), np.std(folds, ddof=1)], abs=1e-6)
        assert [r["term"] for r in tr if r["kind"] == "point_biserial"] == list(logits.FEATURES)
        assert {(int(r["n"]), int(r["n_pos"])) for r in tr} == {(150, int(y.sum()))}
    mean_auc = {r["track"]: float(r["estimate"]) for r in rows if r["kind"] == "cv_auc" and r["term"] == "mean"}
    assert mean_auc["factuality"] > 0.85
    assert 0.3 < mean_auc["fluency"] < 0.7


def test_rerun_rewrites_identical_bytes(env, tmp_path):
    logits.main(["--lang", "english"])
    path = tmp_path / "results" / "tables" / "logits_english.csv"
    first = path.read_bytes()
    logits.main(["--lang", "english"])
    assert path.read_bytes() == first
