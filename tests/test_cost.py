"""Offline tests for chomps.cost: synthetic votes and contexts stores under a
tmp_path standing in for ../results (config.RESULTS is monkeypatched)."""
import json
import math

import pandas as pd
import pytest

from chomps import config, cost, data, judge

USD = "usd_per_1000_examples_per_repeat"


@pytest.fixture
def results(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    return tmp_path


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_votes(results, rows, lang="english", track="factuality", split="test", condition="pseudo_rag",
                judge="gpt-5-mini", fmt="judge", temp="default"):
    write_jsonl(results / "votes" / lang / track / split / f"{condition}__{judge}__{fmt}__t{temp}.jsonl", rows)


def api(index, repeat, pt, ct, rt=0, lat=1.0, cached=False, model="gpt-5-mini-2025-08-07"):
    return {"index": index, "repeat": repeat, "vote": "n", "parse_ok": True, "raw": "n", "prompt_tokens": pt,
            "completion_tokens": ct, "reasoning_tokens": rt, "latency_s": lat, "cached": cached, "model": model,
            "finish_reason": "stop"}


def auto(index, repeat, marker="AUTO_EMPTY_CONTEXT"):
    """Same schema as judge._auto_row."""
    return {"index": index, "repeat": repeat, "vote": "y", "parse_ok": True, "raw": marker, "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0, "cached": False, "model": "",
            "finish_reason": None}


def test_fixture_rows_have_the_judge_store_schema():
    reply = {"content": "n", "prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": 0, "latency_s": 1.0,
             "cached": False, "model": "m", "finish_reason": "stop"}
    assert api("e0", 0, 1, 1).keys() == judge._row("e0", 0, reply).keys()
    assert auto("e0", 0) == judge._auto_row("e0", 0, "AUTO_EMPTY_CONTEXT")


def write_contexts(results, lang, split, condition, rows):
    write_jsonl(results / "contexts" / lang / split / f"{condition}.jsonl", rows)


def only(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, sel
    return sel.iloc[0]


def test_no_stores_writes_header_only(results, capsys):
    df = cost.build()
    out = capsys.readouterr().out
    assert df.empty and "no judge rows" in out and "no retrieval rows" in out
    assert list(pd.read_csv(results / "tables" / "cost.csv").columns) == cost.COLUMNS


def test_judge_rows_pool_splits_and_handle_auto_and_cached_rows(results):
    write_votes(results, [api("e0", 0, 100, 10, rt=4, lat=1.0), api("e0", 1, 100, 20, rt=8, lat=3.0, cached=True),
                          auto("e1", 0), auto("e1", 1)], split="test")
    write_votes(results, [api("v0", 0, 200, 30, lat=2.0)], split="validation")
    r = only(cost.build(), stage="judge")
    assert (r["lang"], r["track"], r["condition"], r["judge"], r["fmt"], r["temp"]) == (
        "english", "factuality", "pseudo_rag", "gpt-5-mini", "judge", "default")
    assert r["splits"] == "validation+test" and r["models"] == "gpt-5-mini-2025-08-07"
    assert (r["n_rows"], r["n_examples"], r["n_api_rows"], r["n_auto"], r["n_fresh_rows"]) == (5, 3, 3, 2, 2)
    assert r["repeats"] == "1|2"  # validation was judged once, test twice
    assert (r["prompt_tokens_mean"], r["prompt_tokens_median"]) == (133.33, 100)  # AUTO rows are not calls
    assert (r["completion_tokens_mean"], r["reasoning_tokens_mean"]) == (20, 4)
    assert (r["latency_s_mean"], r["latency_s_median"]) == (2.0, 2.0)  # the cached row's latency counts
    # gpt-5-mini $0.25 in / $2.00 out per 1M; per row: 45e-6, 65e-6, 0, 0, 110e-6 -> mean 44e-6 per judgment
    assert r[USD] == pytest.approx(0.044)
    assert math.isnan(r["hosting_usd_per_hour"]) and r["tokens_source"] == "api_usage"
    on_disk = pd.read_csv(results / "tables" / "cost.csv")
    assert on_disk.loc[0, "n_api_rows"] == 3 and str(on_disk["n_api_rows"].dtype) == "int64"


def test_rows_sharing_one_fresh_reply_all_count_as_fresh(results):
    """judge.py sends identical inputs once per repeat and copies the reply to every such row."""
    write_votes(results, [api("e0", 0, 100, 10), api("e1", 0, 100, 10)])
    r = only(cost.build(), stage="judge")
    assert (r["n_api_rows"], r["n_fresh_rows"]) == (2, 2)
    assert r[USD] == pytest.approx(1000 * (100 * 0.25 + 10 * 2.0) / 1e6)  # each row pays the full call


def test_repeats_must_cover_the_same_examples(results):
    write_votes(results, [api("e0", 0, 100, 1), api("e0", 1, 100, 1), api("e1", 0, 100, 1)])
    with pytest.raises(ValueError, match="covers different examples"):
        cost.build()


def test_groups_are_separate_per_judge_fmt_temp_and_track(results):
    write_votes(results, [api("e0", 0, 100, 1)], judge="gpt-4.1-mini", temp="0")
    write_votes(results, [api("e0", 0, 100, 1)], judge="gpt-4.1-mini", temp="1.0")
    write_votes(results, [api("e0", 0, 100, 1)], judge="ft-fact-en-mini", fmt="ft", temp="0")
    write_votes(results, [api("e0", 0, 50, 1), auto("e1", 0, "AUTO_EMPTY_ANSWER")], track="fluency",
                condition="none", judge="gpt-4.1-nano")
    df = cost.build()
    assert len(df) == 4 and set(df["temp"]) == {"0", "1.0", "default"}
    assert only(df, track="fluency")["n_auto"] == 1


def test_fine_tuned_prices_and_hosting(results):
    write_votes(results, [api("e0", 0, 1000, 1, model="gpt-4.1-nano-2025-04-14.ft-x")], judge="ft-fact-en-nano",
                fmt="ft", temp="0")
    r = only(cost.build(), judge="ft-fact-en-nano")
    assert r[USD] == pytest.approx((1000 * 0.20 + 1 * 0.80) / 1e6 * 1000)
    assert r["hosting_usd_per_hour"] == 1.70


def test_meta_learner_stores_are_listed_not_costed(results, capsys):
    write_votes(results, [api("e0", 0, 100, 1)])
    meta_row = {**auto("e0", 0), "raw": "p_y=0.700000", "model": "sklearn LogisticRegression(...)"}
    write_votes(results, [meta_row], judge="meta-cv-balanced")
    df = cost.build()
    assert list(df["judge"]) == ["gpt-5-mini"]
    assert "pseudo_rag__meta-cv-balanced__judge__tdefault.jsonl" in capsys.readouterr().out


def test_unpriced_judge_raises(results):
    write_votes(results, [api("e0", 0, 100, 1)], judge="gpt-4o")
    with pytest.raises(KeyError, match="no price"):
        cost.build()


@pytest.mark.parametrize("bad", [
    lambda r: write_votes(r, [{k: v for k, v in api("e0", 0, 100, 1).items() if k != "latency_s"}]),
    lambda r: write_votes(r, [api("e0", 0, 0, 1)]),  # a reply with no prompt tokens is not an API call
    lambda r: write_votes(r, [{**api("e0", 0, 100, 1), "raw": None}]),
    lambda r: write_jsonl(r / "votes" / "english" / "factuality" / "test" / "pseudo_rag__gpt-5-mini__judge.jsonl",
                          [api("e0", 0, 100, 1)]),
    lambda r: write_votes(r, [api("e0", 0, 100, 1)], temp="0.5"),
    lambda r: write_jsonl(r / "votes" / "english" / "factuality" / "pseudo_rag__gpt-5-mini__judge__t0.jsonl",
                          [api("e0", 0, 100, 1)]),
    lambda r: write_votes(r, []),  # empty store
])
def test_malformed_votes_raise(results, bad):
    bad(results)
    with pytest.raises((KeyError, ValueError, TypeError)):
        cost.build()


# --- retrieval stage ------------------------------------------------------------------------

def rerun_meta(pt, ct, rt=0, lat=10.0, cached=False):
    return {"model": "gpt-5-mini-2025-08-07", "prompt_tokens": pt, "completion_tokens": ct,
            "reasoning_tokens": rt, "latency_s": lat, "finish_reason": "stop", "cached": cached}


def dense_meta(query, paper_chunks):
    return {"chunk_ids": [0], "scores": [0.5], "n_chunks": 4,
            "embed_tokens": {"query": query, "paper_chunks": paper_chunks}}


@pytest.fixture
def titles(monkeypatch):
    """test: t0, t1 -> paper A; validation: v0 -> paper A, v1 -> paper B."""
    papers = {"test": {"t0": "A", "t1": "A"}, "validation": {"v0": "A", "v1": "B"}}
    monkeypatch.setattr(data, "load_examples", lambda lang, split: [
        {"index": i, "title": t} for i, t in papers[split].items()])


def test_rerun_retrieval_row(results):
    write_contexts(results, "english", "test", "pseudo_rag_rerun", [
        {"index": "t0", "context": "span", "meta": rerun_meta(30000, 500, rt=400, lat=10.0)},
        {"index": "t1", "context": "span", "meta": rerun_meta(20000, 300, rt=200, lat=20.0, cached=True)}])
    r = only(cost.build(), stage="retrieval")
    assert (r["condition"], r["judge"], r["track"], r["fmt"], r["splits"]) == (
        "pseudo_rag_rerun", "gpt-5-mini", "factuality", "", "test")
    assert (r["n_examples"], r["repeats"], r["n_api_rows"], r["n_fresh_rows"]) == (2, "1", 2, 1)
    assert (r["prompt_tokens_mean"], r["reasoning_tokens_mean"], r["latency_s_mean"]) == (25000, 300, 15.0)
    per_example = [(30000 * 0.25 + 500 * 2.0) / 1e6, (20000 * 0.25 + 300 * 2.0) / 1e6]
    assert r[USD] == pytest.approx(1000 * sum(per_example) / 2)


def test_rerun_outputs_of_one_question_each_pay_the_whole_call(results):
    """llm.chat's cache gives every output of a question the same extraction reply; the standalone
    convention charges each example that whole call, as dense charges each its whole paper."""
    shared = rerun_meta(30000, 500)
    write_contexts(results, "english", "test", "pseudo_rag_rerun", [
        {"index": f"t{i}", "context": "span", "meta": {**shared, "cached": i > 0}} for i in range(3)])
    r = only(cost.build(), stage="retrieval")
    assert (r["n_api_rows"], r["n_fresh_rows"]) == (3, 1)
    assert r[USD] == pytest.approx(1000 * (30000 * 0.25 + 500 * 2.0) / 1e6)


def test_rerun_meta_without_tokens_raises(results):
    meta = {k: v for k, v in rerun_meta(1, 1).items() if k != "prompt_tokens"}
    write_contexts(results, "english", "test", "pseudo_rag_rerun", [{"index": "t0", "context": "s", "meta": meta}])
    with pytest.raises(KeyError, match="prompt_tokens"):
        cost.build()


def test_dense_charges_each_example_its_query_and_whole_paper_across_splits(results, titles):
    write_contexts(results, "english", "test", "dense_k3", [
        {"index": "t0", "context": "c", "meta": dense_meta(10, 900)},
        {"index": "t1", "context": "c", "meta": dense_meta(20, 900)}])
    write_contexts(results, "english", "validation", "dense_k3", [
        {"index": "v0", "context": "c", "meta": dense_meta(30, 900)},
        {"index": "v1", "context": "c", "meta": dense_meta(40, 600)}])
    r = only(cost.build(), stage="retrieval")
    tokens = [10 + 900, 20 + 900, 30 + 900, 40 + 600]  # paper A (900) is not split among its 3 examples
    assert (r["judge"], r["splits"], r["n_examples"], r["repeats"], r["tokens_source"]) == (
        "text-embedding-3-large", "validation+test", 4, "1", "cl100k_base")
    assert r["prompt_tokens_mean"] == pytest.approx(sum(tokens) / 4) and r["prompt_tokens_median"] == 915
    assert r[USD] == pytest.approx(1000 * sum(tokens) / 4 * 0.13 / 1e6)
    assert pd.isna(r["n_api_rows"]) and pd.isna(r["n_fresh_rows"]) and pd.isna(r["latency_s_mean"])


@pytest.mark.parametrize("rows", [
    [{"index": "t0", "context": "c", "meta": dense_meta(10, 900)},
     {"index": "t1", "context": "c", "meta": dense_meta(20, 901)}],  # same paper, different index size
    [{"index": "t0", "context": "c", "meta": {"chunk_ids": [0], "scores": [1.0], "n_chunks": 2}}],
    [{"index": "zz", "context": "c", "meta": dense_meta(10, 900)}],  # not in the dataset
])
def test_dense_inconsistent_or_missing_tokens_raise(results, titles, rows):
    write_contexts(results, "english", "test", "dense_k1", rows)
    with pytest.raises((KeyError, ValueError)):
        cost.build()


def test_free_conditions_get_no_retrieval_row_and_unknown_files_raise(results):
    for cond in ("pseudo_rag", "bm25_k3", "full", "none"):
        write_contexts(results, "english", "test", cond, [{"index": "t0", "context": "c", "meta": {}}])
    assert cost.build().empty
    write_contexts(results, "english", "test", "mystery", [{"index": "t0", "context": "c", "meta": {}}])
    with pytest.raises(ValueError):
        cost.build()


def test_cli(results, capsys):
    write_votes(results, [api("e0", 0, 100, 10)])
    cost.main([])
    out = capsys.readouterr().out
    assert USD in out and (results / "tables" / "cost.csv").exists()
