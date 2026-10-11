"""Offline tests for chomps.meta: synthetic examples and votes stores under tmp_path,
written and read through the real chomps.judge store API (config.RESULTS patched)."""
import csv
import json
import types
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from chomps import config, data, judge, logits, meta, metrics

JUDGES = ["good", "noisy", "yes-man"]
FLIP = {"good": 0.1, "noisy": 0.4}
SIZES = {"train": 80, "validation": 120, "test": 100}
ROW_KEYS = {"index", "repeat", "vote", "parse_ok", "raw", "prompt_tokens", "completion_tokens",
            "reasoning_tokens", "latency_s", "cached", "model", "finish_reason"}
BASE = ["--lang", "english", "--track", "factuality", "--condition", "pseudo_rag", "--judges", *JUDGES]


def make_examples(split, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fact, flu, k = int(rng.random() < 0.75), int(rng.random() < 0.25), int(rng.integers(3, 40))
        out.append({"index": f"en-{split}-{i}", "output_logits": rng.normal(28 - 6 * fact, 4, size=k).tolist(),
                    "output_tokens": ["t"] * k, "labels": {"factuality": fact, "fluency": flu}})
    return out


def synth_votes(examples, track, j, r, split):
    """'good' follows the label with 10 % flips, 'noisy' 40 %, 'yes-man' says y 95 % of the time."""
    rng = np.random.default_rng([JUDGES.index(j), r, config.SPLITS.index(split)])
    out = []
    for e in examples:
        y = e["labels"][track]
        out.append(int(rng.random() < 0.95) if j == "yes-man" else (y if rng.random() >= FLIP[j] else 1 - y))
    return out


def write_store(track, split, condition, j, fmt, temp, examples, repeats):
    votes = [synth_votes(examples, track, j, r, split) for r in range(repeats)]
    path = judge.votes_path("english", track, split, condition, j, fmt, temp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for k, e in enumerate(examples):
            for r in range(repeats):
                v = "y" if votes[r][k] else "n"
                fh.write(json.dumps({"index": e["index"], "repeat": r, "vote": v, "parse_ok": True, "raw": v,
                                     "prompt_tokens": 100, "completion_tokens": 1, "reasoning_tokens": 0,
                                     "latency_s": 0.5, "cached": False, "model": j, "finish_reason": "stop"}) + "\n")


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    examples = {s: make_examples(s, n, seed) for seed, (s, n) in enumerate(SIZES.items())}
    monkeypatch.setattr(data, "load_examples", lambda lang, split: [dict(e) for e in examples[split]])

    def store(track="factuality", condition="pseudo_rag", fmt="judge", temp="default", repeats=1, judges=JUDGES):
        for s in config.SPLITS:
            for j in judges:
                write_store(track, s, condition, j, fmt, temp, examples[s], repeats)

    return types.SimpleNamespace(examples=examples, store=store, results=tmp_path / "results")


def inputs(env, split, r=0, track="factuality"):
    """Vote matrix and labels read back independently of meta.py."""
    by_judge = [judge.load_votes(judge.votes_path("english", track, split, "pseudo_rag", j, "judge", "default"))[r]
                for j in JUDGES]
    exs = env.examples[split]
    X = np.array([[1.0 if v[e["index"]] == "y" else 0.0 for v in by_judge] for e in exs])
    return X, np.array([e["labels"][track] for e in exs])


def summary(protocol, cw, use_logits=False, track="factuality", condition="pseudo_rag"):
    path = meta.summary_path("english", track, condition, protocol, cw, use_logits)
    return list(csv.DictReader(open(path, encoding="utf-8")))


def meta_path(split, name, track="factuality", condition="pseudo_rag"):
    return judge.votes_path("english", track, split, condition, name, "judge", "default")


def test_original_protocol_reproduces_the_competition_meta_learner(env):
    env.store()
    meta.main(BASE + ["--protocol", "original", "--class-weight", "none", "--n-boot", "50"])
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (inputs(env, s) for s in config.SPLITS)
    ref = LogisticRegression(random_state=42).fit(Xtr, ytr)  # the original approach_3 model
    assert not meta_path("train", "meta-original-none").exists()
    for split, X, y in (("validation", Xva, yva), ("test", Xte, yte)):
        path = meta_path(split, "meta-original-none")
        assert path.is_relative_to(env.results)
        rows = judge.load_vote_rows(path)
        assert all(set(q) == ROW_KEYS and q["repeat"] == 0 and q["parse_ok"] for q in rows)
        assert [q["index"] for q in rows] == [e["index"] for e in env.examples[split]]
        assert [q["vote"] for q in rows] == ["y" if v else "n" for v in ref.predict(X)]
        assert [float(q["raw"][4:]) for q in rows] == pytest.approx(ref.predict_proba(X)[:, 1], abs=1e-6)
    rows = summary("original", "none")
    coef = {r["term"]: r for r in rows if r["kind"] == "coef"}
    assert list(coef) == ["intercept", *JUDGES]
    assert [float(r["estimate"]) for r in coef.values()] == pytest.approx(
        np.r_[ref.intercept_, ref.coef_.ravel()], abs=1e-6)
    assert all(float(r["ci_lo"]) <= float(r["ci_hi"]) and (int(r["n"]), int(r["n_pos"])) == (80, ytr.sum())
               for r in coef.values())
    assert float(coef["good"]["ci_lo"]) > max(float(coef["noisy"]["estimate"]), 0)
    f1 = {r["term"]: float(r["estimate"]) for r in rows if r["kind"] == "macro_f1"}
    assert f1 == pytest.approx({
        "validation": metrics.scores(list(yva), ["y" if v else "n" for v in ref.predict(Xva)])["macro_f1"],
        "test": metrics.scores(list(yte), ["y" if v else "n" for v in ref.predict(Xte)])["macro_f1"]}, abs=1e-6)
    assert not [r for r in rows if r["kind"] == "cv_macro_f1"]


def test_cv_protocol_folds_pooled_fit_and_test_predictions(env):
    env.store()
    meta.main(BASE + ["--protocol", "cv", "--class-weight", "balanced", "--n-boot", "50"])
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (inputs(env, s) for s in config.SPLITS)
    X, y = np.vstack([Xtr, Xva]), np.r_[ytr, yva]
    manual = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        pred = LogisticRegression(class_weight="balanced", random_state=42).fit(X[tr], y[tr]).predict(X[te])
        manual.append(metrics.scores(list(y[te]), ["y" if v else "n" for v in pred])["macro_f1"])
    rows = summary("cv", "balanced")
    cv = {r["term"]: float(r["estimate"]) for r in rows if r["kind"] == "cv_macro_f1"}
    assert list(cv) == ["fold1", "fold2", "fold3", "fold4", "fold5", "mean", "std"]
    assert list(cv.values()) == pytest.approx(manual + [np.mean(manual), np.std(manual, ddof=1)], abs=1e-6)
    ref = LogisticRegression(class_weight="balanced", random_state=42).fit(X, y)
    coef = [float(r["estimate"]) for r in rows if r["kind"] == "coef"]
    assert coef == pytest.approx(np.r_[ref.intercept_, ref.coef_.ravel()], abs=1e-6)
    assert {int(r["n"]) for r in rows if r["kind"] == "coef"} == {200}
    assert [r["term"] for r in rows if r["kind"] == "macro_f1"] == ["test"]
    assert not meta_path("validation", "meta-cv-balanced").exists()
    votes = judge.load_votes(meta_path("test", "meta-cv-balanced"))
    assert list(votes) == [0]
    assert [votes[0][e["index"]] for e in env.examples["test"]] == ["y" if v else "n" for v in ref.predict(Xte)]


def test_default_cli_runs_every_protocol_and_class_weight(env):
    env.store()
    meta.main(BASE + ["--n-boot", "20"])
    for protocol in meta.PROTOCOLS:
        for cw in meta.CLASS_WEIGHTS:
            assert meta_path("test", f"meta-{protocol}-{cw}").exists()
            assert summary(protocol, cw)
    intercept = {cw: float(next(r["estimate"] for r in summary("original", cw) if r["term"] == "intercept"))
                 for cw in meta.CLASS_WEIGHTS}
    assert intercept["balanced"] < intercept["none"]  # 75 % y: balancing removes the prior toward y


def test_logits_features_are_appended_and_standardised(env):
    env.store()
    meta.main(BASE + ["--protocol", "original", "--class-weight", "none", "--logits", "--n-boot", "20"])
    rows = summary("original", "none", use_logits=True)
    coef = [r for r in rows if r["kind"] == "coef"]
    assert [r["term"] for r in coef] == ["intercept", *JUDGES, *(f"logit:{f}" for f in logits.FEATURES)]
    assert {r["features"] for r in rows} == {"votes+logits"}
    (Xtr, ytr), (Xte, _) = inputs(env, "train"), inputs(env, "test")
    Ltr, Lte = (logits.feature_matrix(env.examples[s]) for s in ("train", "test"))
    mu, sd = Ltr.mean(0), Ltr.std(0)
    ref = LogisticRegression(random_state=42).fit(np.hstack([Xtr, (Ltr - mu) / sd]), ytr)
    assert [float(r["estimate"]) for r in coef] == pytest.approx(np.r_[ref.intercept_, ref.coef_.ravel()], abs=1e-5)
    votes = judge.load_votes(meta_path("test", "meta-original-none-logits"))[0]
    pred = ref.predict(np.hstack([Xte, (Lte - mu) / sd]))
    assert [votes[e["index"]] for e in env.examples["test"]] == ["y" if v else "n" for v in pred]
    assert not meta_path("test", "meta-original-none").exists()


def test_one_meta_learner_per_repeat(env):
    env.store(repeats=3)
    meta.main(BASE + ["--protocol", "original", "--class-weight", "none", "--repeats", "3", "--n-boot", "20"])
    votes = judge.load_votes(meta_path("test", "meta-original-none"))
    assert sorted(votes) == [0, 1, 2]
    rows = judge.load_vote_rows(meta_path("test", "meta-original-none"))
    assert [(q["index"], q["repeat"]) for q in rows] == [(e["index"], r) for e in env.examples["test"] for r in range(3)]
    for r in range(3):
        (Xtr, ytr), (Xte, _) = inputs(env, "train", r), inputs(env, "test", r)
        pred = LogisticRegression(random_state=42).fit(Xtr, ytr).predict(Xte)
        assert [votes[r][e["index"]] for e in env.examples["test"]] == ["y" if v else "n" for v in pred]
    assert sorted({int(r["repeat"]) for r in summary("original", "none")}) == [0, 1, 2]


def test_rerun_rewrites_identical_bytes(env):
    env.store()
    argv = BASE + ["--protocol", "cv", "--class-weight", "none", "--n-boot", "30"]
    meta.main(argv)
    files = sorted(p for p in env.results.rglob("*") if p.is_file() and "meta" in p.name)
    first = {p: p.read_bytes() for p in files}
    meta.main(argv)
    assert {p: p.read_bytes() for p in files} == first
    assert sorted(p for p in env.results.rglob("*") if p.is_file() and "meta" in p.name) == files


def edit_store(path, keep=lambda i, row: True, extra=()):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = [r for i, r in enumerate(rows) if keep(i, r)] + list(extra)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def nothing_written(env):
    return not [p for p in env.results.rglob("*") if p.is_file() and "meta" in p.name]


def test_missing_vote_raises_and_writes_nothing(env):
    env.store()
    edit_store(judge.votes_path("english", "factuality", "test", "pseudo_rag", "noisy", "judge", "default"),
               keep=lambda i, row: i != 0)
    with pytest.raises(ValueError, match=r"\[noisy\]: 1 examples without a vote \['en-test-0'\]"):
        meta.run("english", "factuality", "pseudo_rag", JUDGES, n_boot=10)
    assert nothing_written(env)


def test_vote_for_an_unknown_index_raises(env):
    env.store()
    edit_store(judge.votes_path("english", "factuality", "validation", "pseudo_rag", "good", "judge", "default"),
               extra=[{"index": "en-validation-999", "repeat": 0, "vote": "y"}])
    with pytest.raises(ValueError, match="1 votes for unknown indices"):
        meta.run("english", "factuality", "pseudo_rag", JUDGES, n_boot=10)


def test_missing_store_or_repeat_raises(env):
    env.store(judges=JUDGES[:2])
    with pytest.raises(FileNotFoundError, match="no votes for yes-man"):
        meta.run("english", "factuality", "pseudo_rag", JUDGES, n_boot=10)
    env.store()
    with pytest.raises(ValueError, match="repeat 1 is needed"):
        meta.run("english", "factuality", "pseudo_rag", JUDGES, repeats=2, n_boot=10)
    assert nothing_written(env)


def test_fluency_track_uses_condition_none(env):
    env.store(track="fluency", condition="none")
    rows = meta.run("english", "fluency", "none", JUDGES, protocol="cv", n_boot=10)
    assert {r["condition"] for r in rows} == {"none"}
    assert meta_path("test", "meta-cv-none", track="fluency", condition="none").exists()
    with pytest.raises(ValueError, match="only condition 'none'"):
        meta.run("english", "fluency", "pseudo_rag", JUDGES, n_boot=10)


@pytest.mark.parametrize("kwargs", [dict(repeats=0), dict(repeats=True), dict(judges=["good", "good"]),
                                    dict(judges=[]), dict(protocol="loo"), dict(class_weight="auto"),
                                    dict(temp="0.7"), dict(n_boot=0)])
def test_bad_arguments_raise(env, kwargs):
    env.store()
    args = dict(lang="english", track="factuality", condition="pseudo_rag", judges=JUDGES, n_boot=10) | kwargs
    with pytest.raises(ValueError):
        meta.run(**args)


def test_bootstrap_is_stratified_and_seeded():
    rng = np.random.default_rng(5)
    y = np.array([1] * 27 + [0] * 3)  # a plain bootstrap would often drop every negative
    X = rng.integers(0, 2, size=(30, 3)).astype(float)
    make = lambda: meta.make_model("none", 3)  # noqa: E731
    lo, hi = meta.bootstrap_coefficients(make, X, y, n_boot=200, seed=0)
    assert lo.shape == hi.shape == (4,) and (lo <= hi).all()
    again = meta.bootstrap_coefficients(make, X, y, n_boot=200, seed=0)
    assert np.array_equal(lo, again[0]) and np.array_equal(hi, again[1])
    assert not np.array_equal(lo, meta.bootstrap_coefficients(make, X, y, n_boot=200, seed=1)[0])


def test_unconverged_fit_raises():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 5)) * 100
    y = (X[:, 0] + rng.normal(size=200) > 0).astype(int)
    with pytest.raises(ConvergenceWarning):
        meta.fit(LogisticRegression(max_iter=1), X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        meta.fit(meta.logreg(None), X[:, :1] / 100, y)  # the real model converges without warnings


def test_cv_folds_need_five_examples_per_class():
    with pytest.raises(ValueError, match="per class"):
        meta.cv_folds(np.array([1] * 30 + [0] * 4))
