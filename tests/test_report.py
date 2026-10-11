"""Offline tests of chomps.report.

Gold labels (data.load_labels), judge.py's store helpers (report._store_api) and meta.py's
summary path (report._meta_summary_path) are replaced. Votes stores, dataset files and tables
live in tmp_path. No API calls and no ../datasets.
"""
import csv
import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from chomps import config, data, metrics, report

N = 40
IDS = [f"en-tst-{k}" for k in range(N)]
TABLES_DIR = lambda: config.RESULTS / "tables"  # noqa: E731  (config.RESULTS is patched per test)


def _bits(seed: int, p_pos: float, n: int = N) -> list:
    return [int(x) for x in np.random.default_rng(seed).random(n) < p_pos]


def _noisy(gold: dict, seed: int, flip: float = 0.25) -> dict:
    """Votes that flip the gold label with probability `flip`."""
    rng = np.random.default_rng(seed)
    return {i: ("y" if g else "n") if rng.random() >= flip else ("n" if g else "y") for i, g in gold.items()}


def _votes_path(lang, track, split, condition, judge, fmt, temp):
    """DESIGN.md's store layout (what judge.votes_path builds)."""
    return config.RESULTS / "votes" / lang / track / split / f"{condition}__{judge}__{fmt}__t{temp}.jsonl"


def _load_votes(path) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.setdefault(r["repeat"], {})[r["index"]] = r["vote"]
    return out


def _write(track, condition, judge, fmt, temp, preds_by_repeat, lang="english", split="test"):
    path = _votes_path(lang, track, split, condition, judge, fmt, temp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r, votes in preds_by_repeat.items():
            for i, v in votes.items():
                f.write(json.dumps({"index": i, "repeat": r, "vote": v}) + "\n")
    return path


def _summary_path(lang, track, cond, protocol, cw, logits=False, tag=""):
    return TABLES_DIR() / f"meta_{lang}_{track}_{cond}_{protocol}_{cw}{'_logits' if logits else ''}.csv"


def _meta_summary(lang, track, cond, variant, judges, fmts=(("judge", "default"),), cv=(0.712345, 0.031)):
    """meta.py's summary CSV, with the columns report reads: one coef row per judge and, for
    the cv protocol, fold/mean/std rows, for each (fmt, temp) in `fmts`."""
    _, protocol, cw, *logits = variant.split("-")
    path = _summary_path(lang, track, cond, protocol, cw, bool(logits))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fmt, temp in fmts:
        base = {"judges": ";".join(judges), "fmt": fmt, "temp": temp, "repeat": 0}
        rows += [{**base, "kind": "coef", "term": t, "estimate": "0.5"} for t in ("intercept", *judges)]
        if protocol == "cv":
            rows += [{**base, "kind": "cv_macro_f1", "term": "fold1", "estimate": "0.7"},
                     {**base, "kind": "cv_macro_f1", "term": "mean", "estimate": f"{cv[0]:.6f}"},
                     {**base, "kind": "cv_macro_f1", "term": "std", "estimate": f"{cv[1]:.6f}"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["judges", "fmt", "temp", "repeat", "kind", "term", "estimate"])
        w.writeheader()
        w.writerows(rows)
    return path


def _read_csv(name: str) -> list:
    with open(TABLES_DIR() / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ctx(langs=("english",), splits=("test",), **kw) -> report.Context:
    return report.Context(list(langs), list(splits), n_boot=kw.pop("n_boot", 50), **kw)


def _macro(gold: dict, pred: dict) -> float:
    return f1_score(list(gold.values()), [1 if pred[i] == "y" else 0 for i in gold], average="macro")


def _majority(maps: list) -> dict:
    return {i: "y" if 2 * sum(m[i] == "y" for m in maps) > len(maps) else "n" for i in maps[0]}


@pytest.fixture
def labels(tmp_path, monkeypatch) -> dict:
    """{(lang, split): {index: {"factuality", "fluency"}}}, mutable by the test."""
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "DATASETS", tmp_path / "datasets")
    table = {("english", "test"): {i: {"factuality": f, "fluency": u}
                                   for i, f, u in zip(IDS, _bits(0, 0.7), _bits(1, 0.3))}}
    monkeypatch.setattr(data, "load_labels", lambda lang, split: table[(lang, split)])
    monkeypatch.setattr(report, "_store_api", lambda: (_votes_path, _load_votes))
    monkeypatch.setattr(report, "_meta_summary_path", _summary_path)
    return table


def _add_split(labels: dict, lang: str, split: str, seed: int):
    ids = [f"{lang[:2]}-{split[:3]}-{k}" for k in range(N)]
    labels[(lang, split)] = {i: {"factuality": f, "fluency": u}
                             for i, f, u in zip(ids, _bits(seed, 0.7), _bits(seed + 1, 0.3))}


def _gold(labels: dict, track: str, lang="english", split="test") -> dict:
    return {i: lab[track] for i, lab in labels[(lang, split)].items()}


COUNCIL = {"english": ("gpt-5-mini", "ft-fact-en-mini", "ft-fact-en-nano"),
           "spanish": ("gpt-5-mini", "ft-fact-es-mini", "ft-fact-es-nano")}


def _council_stores(labels: dict, repeats: int = 3, lang="english", split="test", skip=()) -> dict:
    gold = _gold(labels, "factuality", lang, split)
    stores = {j: {r: _noisy(gold, seed + r) for r in range(repeats)}
              for j, seed in zip(COUNCIL[lang], (10, 20, 30)) if j not in skip}
    for j, preds in stores.items():
        _write("factuality", "pseudo_rag", j, "judge", "default", preds, lang, split)
    return stores


# --- systems -------------------------------------------------------------------------

def test_council_means_the_papers_council_on_both_tracks():
    """The original fluency scripts (approach_2/3) used gpt-5-mini plus the factuality
    fine-tunes, so `council` has those members on both tracks; the fluency fine-tunes
    form council_fluft*."""
    fact = {s.name: s for s in report.main_systems("english", "factuality", "pseudo_rag")}
    flu = {s.name: s for s in report.main_systems("english", "fluency", "none")}
    assert [m.judge for m in flu["council"].members] == list(COUNCIL["english"])
    assert [m.judge for m in fact["council"].members] == list(COUNCIL["english"])
    assert [(m.judge, m.fmt) for m in flu["council_fluft"].members] == [
        ("gpt-5-mini", "judge"), ("ft-flu-en-mini", "judge"), ("ft-flu-en-nano", "judge")]
    assert [(m.judge, m.fmt) for m in flu["council_fluft_ftfmt"].members] == [
        ("gpt-5-mini", "judge"), ("ft-flu-en-mini", "ft"), ("ft-flu-en-nano", "ft")]
    assert "council_ftfmt" not in flu and "council_fluft" not in fact
    pairs = {(f, a.name, b.name) for f, a, b in report.comparisons("english", "fluency", "none")}
    assert {("main", "single", "council"), ("main", "council", "council_fluft"),
            ("main", "council", "council_fluft_ftfmt"), ("main", "council", "meta-original-none")} <= pairs
    # The fluency temperature family includes the paper council's members.
    assert ("temp", "ft-fact-en-mini@tdefault", "ft-fact-en-mini@t0") in pairs
    # Same composition, language-independent identity across languages.
    es = {s.name: s for s in report.main_systems("spanish", "fluency", "none")}
    assert es["council"].key == flu["council"].key != flu["council_fluft"].key


def test_stores_that_cannot_exist_are_never_asked_for():
    names = [s.name for s in report.main_systems("english", "factuality", "none")]
    assert "council_ftfmt" not in names  # the fine-tuning format needs a context
    assert not [p for p in report.comparisons("english", "factuality", "none") if p[0] == "fmt"]
    meta = {s.name: s.splits for s in report.main_systems("english", "factuality", "pseudo_rag")
            if s.name.startswith("meta-")}
    assert set(meta.values()) == {("validation", "test"), ("test",)}
    assert meta["meta-original-none"] == ("validation", "test") and meta["meta-cv-balanced-logits"] == ("test",)
    assert all(m.temp == "default" for f, a, b in report.comparisons("english", "factuality", "pseudo_rag")
               for m in (*a.members, *b.members) if m.judge == "gpt-5-mini")


# --- tables --------------------------------------------------------------------------

def test_tables_are_skipped_until_their_inputs_exist(labels, capsys):
    written = report.build(report.TABLES, _ctx())
    out = capsys.readouterr().out
    assert list(written) == ["class_distribution"]  # needs only the gold labels
    for name in report.TABLES:
        if name != "class_distribution":
            assert f"[report] {name}: SKIPPED (no inputs yet" in out
            assert not (TABLES_DIR() / f"{name}.csv").exists()
    assert "factuality_main: SKIPPED (no inputs yet; none of its 11 row groups has a store)" in out
    assert "per_judge: SKIPPED (no inputs yet)" in out  # discovered from disk: nothing to list
    assert "<-" not in out  # nothing is in play, so nothing counts as missing


def test_main_table_scores_single_council_and_meta(labels):
    gold = _gold(labels, "factuality")
    stores = _council_stores(labels)
    meta = {0: _noisy(gold, 99, flip=0.1)}
    _write("factuality", "pseudo_rag", "meta-original-none", "judge", "default", meta)
    _meta_summary("english", "factuality", "pseudo_rag", "meta-original-none", COUNCIL["english"])

    rows = report.build(["factuality_main"], _ctx())["factuality_main"]
    assert [r["system"] for r in rows] == ["single", "council", "meta-original-none"]  # council_ftfmt: not run
    by = {r["system"]: r for r in rows}
    single = [_macro(gold, stores["gpt-5-mini"][r]) for r in range(3)]
    councils = [_majority([stores[j][r] for j in stores]) for r in range(3)]
    assert by["single"]["macro_f1"] == pytest.approx(np.mean(single))
    assert by["single"]["macro_f1_std"] == pytest.approx(np.std(single, ddof=1))
    g = list(gold.values())
    assert (by["single"]["ci_mean_lo"], by["single"]["ci_mean_hi"]) == metrics.bootstrap_ci_mean(
        g, [[stores["gpt-5-mini"][r][i] for i in gold] for r in range(3)], "macro_f1", 50, 0)
    assert by["council"]["macro_f1"] == pytest.approx(np.mean([_macro(gold, c) for c in councils]))
    assert by["council"]["macro_f1_consensus"] == pytest.approx(_macro(gold, _majority(councils)))
    assert (by["council"]["n"], by["council"]["n_repeats"]) == (N, 3)
    assert by["meta-original-none"]["macro_f1"] == pytest.approx(_macro(gold, meta[0]))
    assert by["meta-original-none"]["macro_f1_std"] is None
    assert by["council"]["members"].count("[pseudo_rag,judge,tdefault]") == 3
    assert by["meta-original-none"]["members"] == "meta-original-none(" + " + ".join(
        f"{j}[pseudo_rag,judge,tdefault]" for j in COUNCIL["english"]) + ")"

    csv_rows = _read_csv("factuality_main")
    assert csv_rows[0]["macro_f1"] == f"{np.mean(single):.4f}"
    assert csv_rows[2]["macro_f1_std"] == ""
    tex = (TABLES_DIR() / "factuality_main.tex").read_text()
    assert tex.startswith("% generated by chomps.report") and "\\begin{tabular}{lllccccccc}" in tex
    assert tex.count("\\\\") == 1 + 3  # header + one line per row
    assert "meta-original-none" in tex and "$\\pm$" in tex and "INCOMPLETE" not in tex
    assert f"%   english council: {by['council']['members']}" in tex  # composition legend
    assert not (TABLES_DIR() / "factuality_main.missing.txt").exists()


def test_ci_column_is_the_ci_of_the_printed_mean(labels):
    """A weak repeat 0: the .tex CI now brackets the mean it is printed next to."""
    gold = _gold(labels, "factuality")
    preds = {0: _noisy(gold, 1, flip=0.45), 1: _noisy(gold, 2, flip=0.05), 2: _noisy(gold, 3, flip=0.05)}
    _write("factuality", "pseudo_rag", "gpt-5-mini", "judge", "default", preds)
    row = report.build(["factuality_main"], _ctx(n_boot=500))["factuality_main"][0]
    assert row["ci_r0_hi"] < row["macro_f1"]  # the old pairing: repeat 0's CI excludes the mean
    assert row["ci_mean_lo"] <= row["macro_f1"] <= row["ci_mean_hi"]
    tex = (TABLES_DIR() / "factuality_main.tex").read_text()
    assert "\\textbf{95\\% CI}" in tex
    assert f"{row['macro_f1']:.4f} $\\pm$ {row['macro_f1_std']:.4f} & [{row['ci_mean_lo']:.4f}, " \
           f"{row['ci_mean_hi']:.4f}] \\\\" in tex


def test_partial_council_is_a_gap_not_a_skipped_row(labels, capsys):
    """The reviewer's case: every store present except the test store of ft-fact-en-nano."""
    _add_split(labels, "english", "validation", 5)
    _council_stores(labels, lang="english", split="validation")
    _council_stores(labels, lang="english", split="test", skip=("ft-fact-en-nano",))
    ctx = lambda: _ctx(splits=("validation", "test"))  # noqa: E731  (a Context is a snapshot of the stores)
    with pytest.raises(report.IncompleteInputs, match="factuality_main"):
        report.build(["factuality_main", "class_distribution"], ctx())
    out = capsys.readouterr().out
    missing = str(_votes_path("english", "factuality", "test", "pseudo_rag", "ft-fact-en-nano", "judge", "default"))
    assert f"{missing}  <- english/factuality/test council" in out
    assert "factuality_main: NOT WRITTEN" in out
    assert not (TABLES_DIR() / "factuality_main.csv").exists()
    assert (TABLES_DIR() / "class_distribution.csv").exists()  # complete tables are still written

    rows = report.build(["factuality_main"], ctx(), allow_missing=True)["factuality_main"]
    assert [(r["split"], r["system"]) for r in rows] == [
        ("validation", "single"), ("validation", "council"), ("test", "single")]
    tex = (TABLES_DIR() / "factuality_main.tex").read_text()
    assert "% INCOMPLETE (built with --allow-missing): 1 expected row(s) absent" in tex and missing in tex
    assert missing in (TABLES_DIR() / "factuality_main.missing.txt").read_text()

    _council_stores(labels, lang="english", split="test")  # the run finishes
    report.build(["factuality_main"], ctx())
    assert "INCOMPLETE" not in (TABLES_DIR() / "factuality_main.tex").read_text()
    assert not (TABLES_DIR() / "factuality_main.missing.txt").exists()


def test_allow_missing_with_no_complete_row_writes_nothing(labels, capsys):
    _council_stores(labels, repeats=1, skip=("gpt-5-mini", "ft-fact-en-nano"))  # one member of the council only
    assert report.build(["factuality_main"], _ctx(), allow_missing=True) == {}
    out = capsys.readouterr().out
    assert "factuality_main: SKIPPED (no complete row)" in out and "gpt-5-mini__judge__tdefault.jsonl  <-" in out
    assert not (TABLES_DIR() / "factuality_main.csv").exists()


def test_a_run_missing_in_one_language_is_a_gap(labels, capsys):
    _add_split(labels, "spanish", "test", 7)
    _write("factuality", "pseudo_rag", "gpt-5-mini", "judge", "default", {0: _noisy(_gold(labels, "factuality"), 1)})
    with pytest.raises(report.IncompleteInputs):
        report.build(["factuality_main"], _ctx(langs=("english", "spanish")))
    out = capsys.readouterr().out
    assert str(_votes_path("spanish", "factuality", "test", "pseudo_rag", "gpt-5-mini", "judge", "default")) in out
    assert report.build(["factuality_main"], _ctx())["factuality_main"]  # English alone is complete


def test_judge_runs_on_some_splits_only_are_not_gaps(labels, capsys):
    """An ablation run only on test is fine: judge runs are expected per split."""
    _add_split(labels, "english", "validation", 5)
    _write("factuality", "bm25_k3", "gpt-5-mini", "judge", "default", {0: _noisy(_gold(labels, "factuality"), 2)})
    rows = report.build(["retrieval_ablation"], _ctx(splits=("validation", "test")))["retrieval_ablation"]
    assert [(r["split"], r["condition"]) for r in rows] == [("test", "bm25_k3")]
    assert "only on the splits that have inputs: factuality single@bm25_k3 (test)" in capsys.readouterr().out


def test_meta_learner_is_expected_on_all_its_targets(labels, capsys):
    """meta.py writes validation and test in one run: one without the other is a gap.
    It never writes train, so train is not asked for."""
    for split in ("train", "validation"):
        _add_split(labels, "english", split, 11 if split == "train" else 5)
    val_gold = _gold(labels, "factuality", "english", "validation")
    _write("factuality", "pseudo_rag", "meta-original-none", "judge", "default", {0: _noisy(val_gold, 3)},
           split="validation")
    _meta_summary("english", "factuality", "pseudo_rag", "meta-original-none", COUNCIL["english"])
    ctx = lambda: _ctx(splits=("train", "validation", "test"))  # noqa: E731
    with pytest.raises(report.IncompleteInputs):
        report.build(["factuality_main"], ctx())
    out = capsys.readouterr().out
    assert "english/factuality/test meta-original-none" in out and "/train/" not in out
    _write("factuality", "pseudo_rag", "meta-original-none", "judge", "default",
           {0: _noisy(_gold(labels, "factuality"), 4)})
    rows = report.build(["factuality_main"], ctx())["factuality_main"]
    assert [(r["split"], r["system"]) for r in rows] == [("validation", "meta-original-none"),
                                                        ("test", "meta-original-none")]


def test_meta_row_needs_its_summary(labels, capsys):
    _write("factuality", "pseudo_rag", "meta-original-none", "judge", "default",
           {0: _noisy(_gold(labels, "factuality"), 3)})
    with pytest.raises(report.IncompleteInputs):
        report.build(["factuality_main"], _ctx())
    assert str(_summary_path("english", "factuality", "pseudo_rag", "original", "none")) in capsys.readouterr().out
    # A summary that only holds another fmt/temp (meta.py names it without fmt/temp) does not count.
    _meta_summary("english", "factuality", "pseudo_rag", "meta-original-none", COUNCIL["english"],
                  fmts=(("ft", "default"),))
    with pytest.raises(report.IncompleteInputs):
        report.build(["factuality_main"], _ctx())
    assert "[no rows for fmt=judge temp=default]" in capsys.readouterr().out


def test_fluency_council_rows(labels, capsys):
    flu = _gold(labels, "fluency")
    for j, seed in zip(COUNCIL["english"], (1, 2, 3)):
        _write("fluency", "none", j, "judge", "default", {0: _noisy(flu, seed)})
    rows = report.build(["fluency_main"], _ctx())["fluency_main"]
    assert [r["system"] for r in rows] == ["single", "council"]
    assert "ft-fact-en-nano[none,judge,tdefault]" in rows[1]["members"]
    assert "fluency council_fluft" in capsys.readouterr().out  # listed as not run

    _write("fluency", "none", "ft-flu-en-mini", "judge", "default", {0: _noisy(flu, 4)})  # half a council
    with pytest.raises(report.IncompleteInputs):
        report.build(["fluency_main"], _ctx())
    assert "ft-flu-en-nano__judge__tdefault.jsonl  <- english/fluency/test council_fluft" in capsys.readouterr().out


def test_store_missing_examples_raises(labels):
    gold = _gold(labels, "factuality")
    _write("factuality", "pseudo_rag", "gpt-5-mini", "judge", "default",
           {0: _noisy(dict(list(gold.items())[:-1]), 1)})
    with pytest.raises(ValueError, match="1 of 40 missing"):
        report.build(["factuality_main"], _ctx())


def test_council_with_mismatched_repeats_raises(labels):
    gold = _gold(labels, "factuality")
    for j, reps in (("gpt-5-mini", 2), ("ft-fact-en-mini", 1), ("ft-fact-en-nano", 2)):
        _write("factuality", "pseudo_rag", j, "judge", "default", {r: _noisy(gold, r) for r in range(reps)})
    with pytest.raises(ValueError, match="different repeat indices"):
        report.build(["factuality_main"], _ctx())


def test_confusion_matrices_and_significance(labels):
    gold = _gold(labels, "factuality")
    stores = _council_stores(labels, repeats=1)
    single, council = stores["gpt-5-mini"][0], _majority([s[0] for s in stores.values()])

    written = report.build(["confusion_matrices", "significance"], _ctx())
    conf = {(r["system"], r["pred"]): r for r in written["confusion_matrices"]}
    assert set(conf) == {(s, p) for s in ("single", "council") for p in ("repeat0", "consensus")}
    expect = {"tp": sum(gold[i] == 1 and single[i] == "y" for i in IDS),
              "fp": sum(gold[i] == 0 and single[i] == "y" for i in IDS),
              "tn": sum(gold[i] == 0 and single[i] == "n" for i in IDS),
              "fn": sum(gold[i] == 1 and single[i] == "n" for i in IDS)}
    assert {k: conf[("single", "repeat0")][k] for k in expect} == expect

    sig = [r for r in written["significance"] if (r["system_a"], r["system_b"]) == ("single", "council")]
    assert [r["pred"] for r in sig] == ["repeat0", "consensus"]
    s = sig[0]
    right_a = {i: single[i] == ("y" if gold[i] else "n") for i in IDS}
    right_b = {i: council[i] == ("y" if gold[i] else "n") for i in IDS}
    assert s["mcnemar_b"] == sum(right_a[i] and not right_b[i] for i in IDS)
    assert s["mcnemar_c"] == sum(right_b[i] and not right_a[i] for i in IDS)
    assert s["diff"] == pytest.approx(_macro(gold, single) - _macro(gold, council))
    assert s["ci_lo"] <= s["diff_boot_mean"] <= s["ci_hi"]
    assert 0.0 <= s["p_mcnemar"] <= 1.0
    assert {r["family"] for r in written["significance"]} == {"main"}


def test_retrieval_ablation_per_judge_and_their_comparisons(labels):
    fact, flu = _gold(labels, "factuality"), _gold(labels, "fluency")
    _write("factuality", "pseudo_rag", "gpt-5-mini", "judge", "default", {0: _noisy(fact, 1)})
    _write("factuality", "bm25_k3", "gpt-5-mini", "judge", "default", {0: _noisy(fact, 2)})
    _write("factuality", "pseudo_rag", "gpt-4.1-nano", "judge", "default", {0: _noisy(fact, 3)})
    _write("factuality", "pseudo_rag", "gpt-4.1-nano", "judge", "0", {0: _noisy(fact, 4)})
    _write("fluency", "none", "ft-fact-en-mini", "judge", "default", {0: _noisy(flu, 5)})  # paper council member

    written = report.build(["retrieval_ablation", "per_judge", "significance"], _ctx())
    assert [r["condition"] for r in written["retrieval_ablation"]] == ["pseudo_rag", "bm25_k3"]
    assert [(r["track"], r["judge"], r["fmt"], r["temp"]) for r in written["per_judge"]] == [
        ("factuality", "gpt-5-mini", "judge", "default"), ("factuality", "gpt-4.1-nano", "judge", "default"),
        ("factuality", "gpt-4.1-nano", "judge", "0"), ("fluency", "ft-fact-en-mini", "judge", "default")]
    pairs = {(r["family"], r["system_a"], r["system_b"]) for r in written["significance"]}
    assert pairs == {("retrieval", "single@pseudo_rag", "single@bm25_k3"),
                     ("temp", "gpt-4.1-nano@tdefault", "gpt-4.1-nano@t0")}


def test_per_judge_matches_fine_tunes_across_languages(labels, capsys):
    _add_split(labels, "spanish", "test", 7)
    _write("factuality", "pseudo_rag", "ft-fact-en-mini", "ft", "0", {0: _noisy(_gold(labels, "factuality"), 1)})
    with pytest.raises(report.IncompleteInputs):
        report.build(["per_judge"], _ctx(langs=("english", "spanish")))
    assert "pseudo_rag__ft-fact-es-mini__ft__t0.jsonl  <- spanish/factuality/test ft-fact-es-mini" \
        in capsys.readouterr().out
    _write("factuality", "pseudo_rag", "ft-fact-es-mini", "ft", "0",
           {0: _noisy(_gold(labels, "factuality", "spanish"), 2)}, lang="spanish")
    rows = report.build(["per_judge"], _ctx(langs=("english", "spanish")))["per_judge"]
    assert [(r["lang"], r["judge"], r["fmt"], r["temp"]) for r in rows] == [
        ("english", "ft-fact-en-mini", "ft", "0"), ("spanish", "ft-fact-es-mini", "ft", "0")]


def test_unrecognised_store_file_raises(labels):
    path = _votes_path("english", "factuality", "test", "pseudo_rag", "gpt-5-mini", "judge", "default")
    path.parent.mkdir(parents=True)
    (path.parent / "pseudo_rag__gpt-5-mini__judge__twarm.jsonl").write_text("")
    with pytest.raises(ValueError, match="not a votes store name"):
        report.build(["per_judge"], _ctx())


def test_meta_cv_discovers_variants_and_adds_cv_fold_scores(labels, capsys):
    gold = _gold(labels, "factuality")
    for name, fmt in (("meta-cv-balanced", "judge"), ("meta-cv-balanced", "ft"), ("meta-cv-none", "judge")):
        _write("factuality", "pseudo_rag", name, fmt, "default", {0: _noisy(gold, 7)})
    _meta_summary("english", "factuality", "pseudo_rag", "meta-cv-balanced", COUNCIL["english"],
                  fmts=(("judge", "default"), ("ft", "default")))

    with pytest.raises(report.IncompleteInputs):  # meta-cv-none has no summary
        report.build(["meta_cv"], _ctx())
    assert str(_summary_path("english", "factuality", "pseudo_rag", "cv", "none")) in capsys.readouterr().out

    _meta_summary("english", "factuality", "pseudo_rag", "meta-cv-none", COUNCIL["english"], cv=(0.6, 0.05))
    out_rows = report.build(["meta_cv"], _ctx())["meta_cv"]
    by = {(r["variant"], r["fmt"]): r for r in out_rows}
    assert list(by) == [("meta-cv-none", "judge"), ("meta-cv-balanced", "judge"), ("meta-cv-balanced", "ft")]
    balanced, none = by[("meta-cv-balanced", "judge")], by[("meta-cv-none", "judge")]
    assert (balanced["cv_macro_f1_mean"], balanced["cv_macro_f1_std"]) == (0.712345, 0.031)
    assert none["cv_macro_f1_mean"] == 0.6 and none["judges"] == " + ".join(COUNCIL["english"])
    assert (none["protocol"], none["class_weight"], none["features"]) == ("cv", "none", "votes")
    assert _read_csv("meta_cv")[0]["cv_macro_f1_mean"] == "0.6000"

    # meta.py never predicts validation with the cv protocol: such a file is an error, not a row.
    _add_split(labels, "english", "validation", 5)
    _write("factuality", "pseudo_rag", "meta-cv-none", "judge", "default", {0: _noisy(gold, 7)}, split="validation")
    with pytest.raises(ValueError, match="meta.py writes no such store"):
        report.build(["meta_cv"], _ctx(splits=("validation", "test")))


def test_class_distribution(labels):
    labels[("english", "validation")] = {f"en-val-{k}": {"factuality": int(k < 3), "fluency": 0} for k in range(4)}
    rows = report.build(["class_distribution"], _ctx(splits=("validation", "test")))["class_distribution"]
    val = {r["track"]: r for r in rows if r["split"] == "validation"}
    assert (val["factuality"]["n"], val["factuality"]["n_y"], val["factuality"]["n_n"]) == (4, 3, 1)
    assert (val["fluency"]["n_y"], val["fluency"]["pct_y"]) == (0, 0.0)
    test_fact = next(r for r in rows if (r["split"], r["track"]) == ("test", "factuality"))
    assert test_fact["n_y"] == sum(_bits(0, 0.7))
    assert _read_csv("class_distribution")[0]["pct_y"] == "75.0"


def _approach_file(lang, split, k, track, ids, votes):
    path = config.DATASETS / lang / split / f"approach_{k}_answers_{track}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    other = "has_fluency_mistakes" if track == "factuality" else "has_factual_mistakes"
    with open(path, "w", encoding="utf-8") as f:  # the original writer's schema, other key a placeholder
        for i, v in zip(ids, votes):
            f.write(json.dumps({"index": i, other: "n", config.LABEL_KEY[track]: v}) + "\n")


def test_original_rescored_reproduces_the_paper_council_row(labels, capsys):
    """English test, approach 2: tp/fp/tn/fn = 191/6/24/19 on 240 examples."""
    ids = [f"en-tst-{k}" for k in range(240)]
    gold = [1] * 210 + [0] * 30
    votes = ["y"] * 191 + ["n"] * 19 + ["y"] * 6 + ["n"] * 24
    labels[("english", "test")] = {i: {"factuality": g, "fluency": 0} for i, g in zip(ids, gold)}
    _approach_file("english", "test", 2, "factuality", ids, votes)

    rows = report.build(["original_rescored"], _ctx())["original_rescored"]
    assert [(r["track"], r["approach"], r["system"]) for r in rows] == [("factuality", "approach_2", "council")]
    row = _read_csv("original_rescored")[0]
    assert (row["accuracy"], row["precision_y"], row["recall_y"]) == ("0.8958", "0.9695", "0.9095")
    assert (row["f1_y"], row["macro_f1"]) == ("0.9386", "0.7981")
    assert "no input at all (5): factuality approach_1, factuality approach_3" in capsys.readouterr().out


def test_original_approach_3_is_not_expected_on_train(labels):
    """Approach 3 was fitted on train and predicted only validation and test."""
    _add_split(labels, "english", "train", 11)
    for split in ("train", "test"):
        ids = list(labels[("english", split)])
        for k in ((1, 3) if split == "test" else (1,)):
            _approach_file("english", split, k, "factuality", ids, ["y"] * N)
    rows = report.build(["original_rescored"], _ctx(splits=("train", "test")))["original_rescored"]
    assert [(r["split"], r["approach"]) for r in rows] == [("train", "approach_1"), ("test", "approach_1"),
                                                           ("test", "approach_3")]


def test_cli(labels, capsys):
    report.main(["--tables", "class_distribution", "--lang", "english", "--split", "test", "--n-boot", "10"])
    assert (TABLES_DIR() / "class_distribution.csv").exists()
    assert (TABLES_DIR() / "class_distribution.tex").exists()
    with pytest.raises(SystemExit):
        report.main(["--tables", "no_such_table"])
    _council_stores(labels, repeats=1, skip=("ft-fact-en-nano",))
    argv = ["--tables", "factuality_main", "--lang", "english", "--split", "test", "--n-boot", "10"]
    with pytest.raises(SystemExit) as e:
        report.main(argv)
    assert "not written, expected inputs missing: factuality_main" in str(e.value.code)
    report.main(argv + ["--allow-missing"])
    assert (TABLES_DIR() / "factuality_main.missing.txt").exists()


def test_with_the_real_judge_and_meta_path_helpers(tmp_path, monkeypatch):
    """Interface check against judge.votes_path/load_votes and meta.summary_path (offline)."""
    judge = pytest.importorskip("chomps.judge")
    meta = pytest.importorskip("chomps.meta")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    gold = dict(zip(IDS, _bits(0, 0.7)))
    monkeypatch.setattr(data, "load_labels", lambda lang, split: {i: {"factuality": g, "fluency": 0}
                                                                   for i, g in gold.items()})
    path = judge.votes_path("english", "factuality", "test", "pseudo_rag", "gpt-5-mini", "judge", "default")
    path.parent.mkdir(parents=True)
    preds = {r: _noisy(gold, r) for r in range(2)}
    with open(path, "w", encoding="utf-8") as f:
        for i in IDS:
            for r in range(2):
                f.write(json.dumps({"index": i, "repeat": r, "vote": preds[r][i]}) + "\n")
    written = report.build(["factuality_main", "per_judge"], _ctx())
    assert [r["system"] for r in written["factuality_main"]] == ["single"]
    assert written["factuality_main"][0]["macro_f1"] == pytest.approx(
        np.mean([_macro(gold, preds[r]) for r in range(2)]))
    assert [(r["judge"], r["fmt"], r["temp"]) for r in written["per_judge"]] == [("gpt-5-mini", "judge", "default")]
    assert report._meta_summary_path("english", "factuality", "pseudo_rag", "cv", "none", False) == \
        meta.summary_path("english", "factuality", "pseudo_rag", "cv", "none", False)
    assert {c for c in ("judges", "fmt", "temp", "repeat", "kind", "term", "estimate")} <= set(meta.SUMMARY_COLUMNS)
