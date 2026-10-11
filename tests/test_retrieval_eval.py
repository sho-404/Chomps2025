"""Offline tests for chomps.retrieval_eval: synthetic paper, examples and context
stores; chomps.retrieval / chomps.data are monkeypatched, outputs go to tmp_path."""
import json
import math

import pandas as pd
import pytest
from rapidfuzz import fuzz

from chomps import data, prompts, retrieval_eval as re_

REAL_COUNT_TOKENS = re_.count_tokens
WORDS = [f"w{i}" for i in range(400)]
PAPER = ("Title of the paper\nWe propose a method for communi-\ncation analysis.\n"
         "The model reaches 91.2 accuracy on the benchmark. ﬁne-tuning improves results by 3 points.\n"
         "Such effects were not found to be significant.\n"
         + " ".join(WORDS))
CHUNK_A, CHUNK_B = " ".join(WORDS[0:100]), " ".join(WORDS[200:300])
NOT_FOUND = "Context not found in paper."


@pytest.fixture(autouse=True)
def word_tokens(monkeypatch):
    """Deterministic token count for every test except the one checking tiktoken."""
    monkeypatch.setattr(re_, "count_tokens", lambda s: len(s.split()))


def stats(context):
    return re_.context_stats(context, re_.norm(PAPER), re_.loose(PAPER))


# --- per-example stats -------------------------------------------------------------

def test_norm_and_loose():
    assert re_.norm("  a \n b\t\tc ") == "a b c"
    assert re_.loose("communi-\ncation ﬁne  x") == "communicationfinex"
    assert re_.loose("fine-tuning") == "fine-tuning"  # a hyphen without a line break stays


def test_verbatim_modulo_whitespace_is_exact():
    s = stats("The model reaches  91.2\naccuracy on the benchmark.")
    assert (s["verbatim"], s["exact_substring"], s["exact_substring_loose"], s["n_segments"]) == (100.0, 1, 1, 1)
    assert (s["empty"], s["not_found"]) == (0, 0)


@pytest.mark.parametrize("context", ["We propose a method for communication analysis.",  # de-hyphenated
                                     "fine-tuning improves results by 3 points."])          # ligature repaired
def test_pdf_artifact_repairs_are_loose_exact_only(context):
    s = stats(context)
    assert s["exact_substring"] == 0 and s["exact_substring_loose"] == 1
    assert 85 < s["verbatim"] < 100


def test_segment_cut_at_a_line_break_hyphen_is_loose_exact():
    s = stats("We propose a method for communi-")  # a chunk boundary inside "communi-\ncation"
    assert s["exact_substring"] == 1 and s["exact_substring_loose"] == 1


def test_altered_number_is_not_exact():
    s = stats("The model reaches 99.9 accuracy on the benchmark.")
    assert s["exact_substring"] == 0 and s["exact_substring_loose"] == 0 and s["verbatim"] < 100


def test_chunk_contexts_are_scored_per_segment():
    ctx = CHUNK_A + re_.CHUNK_SEP + CHUNK_B
    # Scoring the joined string would measure the separator and the gap between chunks
    # (~65 on real papers; this synthetic filler is self-similar, hence only < 100).
    assert fuzz.partial_ratio(re_.norm(ctx), re_.norm(PAPER)) < 100
    s = stats(ctx)
    assert (s["verbatim"], s["exact_substring"], s["exact_substring_loose"], s["n_segments"]) == (100.0, 1, 1, 2)


def test_segment_verbatim_is_length_weighted():
    bad = "zz " * 50
    s = stats(CHUNK_A + re_.CHUNK_SEP + bad)
    la, lb = len(CHUNK_A), len(re_.norm(bad))
    expected = (100 * la + fuzz.partial_ratio(re_.norm(bad), re_.norm(PAPER)) * lb) / (la + lb)
    assert s["verbatim"] == pytest.approx(expected) and s["exact_substring"] == 0


@pytest.mark.parametrize("context", [None, "", " \n\t"])
def test_empty_contexts_have_no_verbatim(context):
    s = stats(context)
    assert s["empty"] == 1 and s["n_segments"] == 0
    assert all(math.isnan(s[k]) for k in ("verbatim", "exact_substring", "exact_substring_loose"))
    if context is None:
        assert (s["chars"], s["tokens"]) == (0, 0)


@pytest.mark.parametrize("context", [NOT_FOUND, '"CONTEXT NOT FOUND IN PAPER."', "Context not found in paper",
                                     "Context  not found\nin paper. The paper does not report it."])
def test_refusal_variants_are_flagged_and_still_scored(context):
    s = stats(context)
    assert s["not_found"] == 1 and s["empty"] == 0 and not math.isnan(s["verbatim"])


@pytest.mark.parametrize("context", ["Such effects were not found to be significant.",  # verbatim paper text
                                     CHUNK_A + re_.CHUNK_SEP + "Such effects were not found to be significant."])
def test_not_found_in_paper_text_is_not_a_refusal(context):
    s = stats(context)
    assert s["not_found"] == 0 and s["exact_substring"] == 1


def test_refusal_is_the_retrieval_prompts_own_string():
    # prompts.retrieval is English for both languages, so the refusal is too.
    assert f'return "{re_.REFUSAL}"' in prompts.retrieval("q", "paper text")[1]["content"]
    assert not re_.is_refusal(None) and not re_.is_refusal("")


@pytest.mark.parametrize("context", [re_.CHUNK_SEP, CHUNK_A + re_.CHUNK_SEP, re_.CHUNK_SEP + CHUNK_A,
                                     CHUNK_A + re_.CHUNK_SEP + re_.CHUNK_SEP + CHUNK_B,
                                     CHUNK_A + re_.CHUNK_SEP + " \n " + re_.CHUNK_SEP + CHUNK_B])
def test_blank_segment_raises(context):
    with pytest.raises(ValueError, match="blank segment"):
        stats(context)


def test_segment_longer_than_paper_raises():
    paper, ctx = "abc def ghi", "abc def ghi jkl mno pqr stu"
    assert fuzz.partial_ratio(ctx, paper) == 100  # why: the paper aligns inside the segment
    with pytest.raises(ValueError, match="longer than its paper"):
        re_.context_stats(ctx, paper, re_.loose(paper))
    assert re_.context_stats(paper, paper, re_.loose(paper))["verbatim"] == 100.0  # equal length is fine


def test_context_sha1_identifies_the_text():
    a, b, c = stats("The model reaches 91.2"), stats("The model reaches 91.2"), stats("The model reaches 91.3")
    assert len(a["context_sha1"]) == 12 and a["context_sha1"] == b["context_sha1"] != c["context_sha1"]


def test_non_string_context_raises():
    with pytest.raises(TypeError):
        stats(["a list"])


def test_real_o200k_token_count():
    assert REAL_COUNT_TOKENS("hello world") == 2
    assert REAL_COUNT_TOKENS("") == 0
    assert REAL_COUNT_TOKENS("a <|endoftext|> b") > 3  # special-token text counts as plain text


def test_question_ids_cluster_outputs_of_one_question():
    exs = [{"index": "x0", "title": "A", "question": "q"}, {"index": "x1", "title": "B", "question": "q"},
           {"index": "x2", "title": "A", "question": "q"}, {"index": "x3", "title": "A", "question": "r"}]
    assert re_.question_ids(exs) == {"x0": "x0", "x1": "x1", "x2": "x0", "x3": "x3"}


# --- agreement ------------------------------------------------------------------------

def test_agreement_identical_and_containment():
    assert re_.agreement("Same span.", "Same  span.") == {"jaccard": 1.0, "partial_ratio": 100.0}
    a = re_.agreement("the model reaches 91.2 accuracy",
                      "Results: the model reaches 91.2 accuracy on the benchmark")
    assert a["partial_ratio"] == 100.0  # one span contains the other
    assert a["jaccard"] == pytest.approx(6 / 9)  # {the,model,reaches,91,2,accuracy} vs 9 words


@pytest.mark.parametrize("a,b", [(None, "x"), ("x", ""), ("", None), (None, None), (" ", " "),
                                 (NOT_FOUND, NOT_FOUND), ('"Context not found in paper."', NOT_FOUND),
                                 ("a real span", NOT_FOUND), (NOT_FOUND, "")])
def test_agreement_undefined_unless_both_are_spans(a, b):
    assert all(math.isnan(v) for v in re_.agreement(a, b).values())


# --- store loading -----------------------------------------------------------------------

EXAMPLES = [{"index": f"en-tst-{i}"} for i in range(3)]


def test_load_condition_validates_store(monkeypatch):
    good = {e["index"]: "ctx" for e in EXAMPLES}
    cases = [(good, None), ({**good, "en-tst-9": "x"}, ValueError), (dict(list(good.items())[:2]), ValueError),
             ({**good, "en-tst-0": 3}, TypeError), ([("en-tst-0", "ctx")], TypeError)]
    for store, err in cases:
        monkeypatch.setattr(re_, "_load_contexts", lambda *a, s=store: s)
        if err is None:
            assert re_.load_condition("english", "test", "bm25_k3", EXAMPLES) == good
        else:
            with pytest.raises(err):
                re_.load_condition("english", "test", "bm25_k3", EXAMPLES)


def test_separator_contract_with_retrieval(monkeypatch):
    from chomps import retrieval
    assert re_.CHUNK_SEP == retrieval.SEPARATOR
    monkeypatch.setattr(retrieval, "load_contexts", lambda *a: {})
    assert re_._load_contexts("english", "test", "bm25_k3") == {}
    monkeypatch.setattr(retrieval, "SEPARATOR", "\n---\n")
    with pytest.raises(RuntimeError):
        re_._load_contexts("english", "test", "bm25_k3")


# --- end to end ------------------------------------------------------------------------------

def make_examples(split):
    """Four model outputs of two questions (0, 1 -> q0; 2, 3 -> q1), as in SHROOM-CAP."""
    tag = {"test": "tst", "validation": "val"}[split]
    return [{"index": f"en-{tag}-{i}", "title": "Paper A", "question": f"q{i // 2}", "prompt": f"p{i}",
             "output_text": f"a{i}", "labels": {"factuality": i % 2, "fluency": 0}} for i in range(4)]


def make_stores(split):
    ix = [e["index"] for e in make_examples(split)]
    verbatim = "The model reaches 91.2 accuracy on the benchmark."
    return {  # original spans vary within a question; the rerun gives one span per question
        "pseudo_rag": dict(zip(ix, [verbatim, "We propose a method for communication analysis.", "", NOT_FOUND])),
        "pseudo_rag_rerun": dict(zip(ix, [verbatim, verbatim, NOT_FOUND, NOT_FOUND])),
        "bm25_k3": dict(zip(ix, [CHUNK_A + re_.CHUNK_SEP + CHUNK_B] * 4)),
        "dense_k3": dict(zip(ix, [CHUNK_B] * 4)),
    }


@pytest.fixture
def fake_world(monkeypatch):
    stores = {(s, c): ctx for s in ("test", "validation") for c, ctx in make_stores(s).items()}

    def load_contexts(lang, split, condition):
        if (split, condition) not in stores:
            raise FileNotFoundError(f"no context store for {lang}/{split}/{condition}")
        return stores[split, condition]

    def paper_text(lang, title):
        assert title == "Paper A"
        return PAPER

    monkeypatch.setattr(re_, "_load_contexts", load_contexts)
    monkeypatch.setattr(data, "load_examples", lambda lang, split: make_examples(split))
    monkeypatch.setattr(data, "paper_text", paper_text)
    return stores


def test_run_writes_stats_agreement_grading_and_summary(fake_world, tmp_path):
    summary = re_.run(["english"], ["validation", "test"], list(re_.DEFAULT_CONDITIONS), tmp_path, workers=2)
    for split in ("validation", "test"):
        for c in re_.DEFAULT_CONDITIONS:
            df = pd.read_csv(tmp_path / f"stats_english_{split}_{c}.csv")
            assert list(df.columns) == re_.STATS_COLUMNS and len(df) == 4
        assert (tmp_path / f"agreement_english_{split}.csv").exists()
    assert not (tmp_path / "grading_sheet_english_validation.jsonl").exists()
    st = pd.read_csv(tmp_path / "stats_english_test_pseudo_rag.csv")
    assert list(st["question_id"]) == ["en-tst-0", "en-tst-0", "en-tst-2", "en-tst-2"]

    rows = [json.loads(line) for line in open(tmp_path / "grading_sheet_english_test.jsonl", encoding="utf-8")]
    assert [r["index"] for r in rows] == [e["index"] for e in make_examples("test")]
    assert set(rows[0]) == {"index", "question", "prompt", "output_text", "gold_factuality", "contexts"}
    assert [r["gold_factuality"] for r in rows] == [0, 1, 0, 1]
    assert rows[2]["contexts"] == {"pseudo_rag": "", "bm25_k3": CHUNK_A + re_.CHUNK_SEP + CHUNK_B,
                                   "dense_k3": CHUNK_B}

    assert len(summary) == 8 and list(summary.columns) == re_.SUMMARY_COLUMNS
    assert list(summary["split"][:4]) == ["validation"] * 4  # config order, not alphabetical
    test = summary[summary.split == "test"].set_index("condition")
    pr = test.loc["pseudo_rag"]
    assert (pr["n"], pr["n_questions"], pr["n_distinct_contexts"]) == (4, 2, 4)
    assert (pr["empty_n"], pr["not_found_n"], pr["scored_n"], pr["scored_questions"]) == (1, 1, 2, 1)
    assert pr["exact_pct"] == 50.0 and pr["exact_loose_pct"] == 100.0  # refusal and empty excluded
    assert math.isnan(pr["agree_n"])  # agreement is reported on the rerun row
    rr = test.loc["pseudo_rag_rerun"]
    assert (rr["n_questions"], rr["n_distinct_contexts"], rr["not_found_n"]) == (2, 2, 2)
    # Pairs: (span, span) x2 scored; ("", refusal) one refusal; (refusal, refusal) both.
    assert (rr["agree_n"], rr["agree_questions"], rr["agree_both_not_found_n"], rr["agree_one_not_found_n"]) \
        == (2, 1, 1, 1)
    agree = pd.read_csv(tmp_path / "agreement_english_test.csv")
    assert list(agree.columns) == re_.AGREEMENT_COLUMNS
    assert (agree.loc[0, "jaccard"], agree.loc[0, "partial_ratio"]) == (1.0, 100.0)
    assert agree.loc[1, "jaccard"] < 1 and agree[["jaccard", "partial_ratio"]].iloc[2:].isna().all().all()
    assert list(agree["pseudo_rag_not_found"]) == [0, 0, 0, 1]
    assert list(agree["pseudo_rag_rerun_not_found"]) == [0, 0, 1, 1]
    assert rr["agree_jaccard_mean"] == pytest.approx(agree["jaccard"].iloc[:2].mean(), abs=1e-4)
    bm = test.loc["bm25_k3"]
    assert (bm["verbatim_mean"], bm["exact_pct"], bm["n_questions"], bm["n_distinct_contexts"]) == (
        100.0, 100.0, 2, 1)


def test_missing_store_fails_before_writing(fake_world, tmp_path):
    del fake_world["test", "dense_k3"]
    del fake_world["validation", "pseudo_rag_rerun"]
    with pytest.raises(FileNotFoundError) as e:
        re_.run(["english"], ["validation", "test"], list(re_.DEFAULT_CONDITIONS), tmp_path / "out")
    assert "english/test/dense_k3" in str(e.value) and "english/validation/pseudo_rag_rerun" in str(e.value)
    assert not (tmp_path / "out").exists()


def test_subset_run_skips_grading_and_agreement_and_summary_accumulates(fake_world, tmp_path, capsys):
    re_.run(["english"], ["test"], ["pseudo_rag", "bm25_k3"], tmp_path)
    out = capsys.readouterr().out
    assert "grading sheet english/test not written" in out and "agreement english/test not computed" in out
    assert not list(tmp_path.glob("grading_sheet_*")) and not list(tmp_path.glob("agreement_*"))
    summary = re_.run(["english"], ["validation"], ["dense_k3"], tmp_path)
    assert sorted(zip(summary.split, summary.condition)) == [
        ("test", "bm25_k3"), ("test", "pseudo_rag"), ("validation", "dense_k3")]


def test_summary_rejects_unknown_stats_file(tmp_path):
    pd.DataFrame(columns=re_.STATS_COLUMNS).to_csv(tmp_path / "stats_english_test_mystery.csv", index=False)
    with pytest.raises(ValueError):
        re_.summarize(tmp_path)


def test_summary_rejects_a_stats_file_with_another_layout(tmp_path):
    old = [c for c in re_.STATS_COLUMNS if c not in ("question_id", "context_sha1")]
    pd.DataFrame(columns=old).to_csv(tmp_path / "stats_english_test_bm25_k3.csv", index=False)
    with pytest.raises(ValueError, match="expected"):
        re_.summarize(tmp_path)


def test_cli(fake_world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(re_, "OUT_DIR", tmp_path)
    re_.main(["--lang", "english", "--split", "test", "--conditions", "pseudo_rag", "bm25_k3", "dense_k3",
              "--workers", "1"])
    out = capsys.readouterr().out
    assert "grading_sheet_english_test.jsonl" in out and "summary" in out
    assert (tmp_path / "summary.csv").exists()
    with pytest.raises(SystemExit):
        re_.main(["--conditions", "full"])  # no evidence span to score
    with pytest.raises(SystemExit):
        re_.main(["--workers", "0"])
