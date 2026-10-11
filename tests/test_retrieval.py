"""Offline tests for chomps.retrieval: llm.chat / llm.embed / data.* are faked,
results/ is redirected to a tmp dir, and any real API access fails the test."""
import json
import re

import pytest

from chomps import config, data, llm, prompts
from chomps import retrieval as R


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real API call")
    for name in ("client", "chat", "embed"):
        monkeypatch.setattr(llm, name, forbidden)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")


def words(n, prefix="f"):
    return [f"{prefix}{i}" for i in range(n)]


def zebra_paper():
    w = words(300)          # chunks [0:100] [50:150] [100:200] [150:250] [200:300]
    w[120] = "zebra"        # inside chunks 1 and 2 only
    return " ".join(w)


def example(index, title, question="Where is the zebra?", context_original="span"):
    return {"index": index, "title": title, "question": question, "prompt": f"About {title}: {question}",
            "output_text": "answer", "context_original": context_original,
            "labels": {"factuality": 1, "fluency": 0}}


def use_papers(monkeypatch, papers):
    calls = []

    def paper_text(lang, title):
        calls.append(title)
        if title not in papers:
            raise FileNotFoundError(f"no PDF for {lang} paper: {title!r}")
        return papers[title]
    monkeypatch.setattr(data, "paper_text", paper_text)
    return calls


def reply(content="span", finish_reason="stop"):
    return {"content": content, "finish_reason": finish_reason, "prompt_tokens": 120, "completion_tokens": 40,
            "reasoning_tokens": 30, "latency_s": 1.5, "model": "gpt-5-mini-2025-08-07", "cached": False}


# The competition's extraction request, transcribed by hand from get_context_from_llm in
# {english,spanish}/scripts/inject_context_single_block.py, trailing spaces included.
ORIGINAL_SYSTEM = "You are a helpful assistant that extracts verbatim text."
ORIGINAL_USER = "\n".join([
    "",
    "    You are a precise researcher. ",
    "    ",
    "    I will provide you with a Question and the full text of a Research Paper.",
    "    Your task is to find the specific segment of text (context) within the paper that contains the answer "
    "to the question.",
    "    ",
    "    Rules:",
    "    1. The context must be VERBATIM from the text.",
    "    2. Do not paraphrase.",
    '    3. If the answer is not explicitly in the text, return "Context not found in paper."',
    "    4. Return ONLY the context text, nothing else.",
    "",
    "    Question: {question}",
    "",
    "    Paper Text:",
    "    {paper} ",
    "    ",
])


def original_messages(question, paper):
    return [{"role": "system", "content": ORIGINAL_SYSTEM},
            {"role": "user", "content": ORIGINAL_USER.format(question=question, paper=paper[:120_000])}]


@pytest.fixture
def original_prompt(monkeypatch):
    """prompts.retrieval as the original script builds it (the real one drifts: see the xfail test)."""
    monkeypatch.setattr(prompts, "retrieval", original_messages)


# --- chunking and ranking ------------------------------------------------------------

@pytest.mark.parametrize("n, expected", [(1, 1), (100, 1), (101, 2), (150, 2), (151, 3), (200, 3), (230, 4), (260, 5)])
def test_chunk_windows_cover_the_tail_without_nested_chunks(n, expected):
    w = words(n)
    chunks = R.chunk_paper("\n".join(w[:5]) + "\t  " + " ".join(w[5:]))
    assert len(chunks) == expected
    assert chunks == [" ".join(w[50 * i:50 * i + 100]) for i in range(expected)]
    assert chunks[-1].split()[-1] == w[-1]
    assert all(chunks[i] not in chunks[i - 1] for i in range(1, len(chunks)))


def test_chunking_reads_only_the_truncated_paper(monkeypatch):
    monkeypatch.setattr(config, "MAX_PAPER_CHARS", 17)
    assert R.chunk_paper("aaaa bbbb\tcccc\ndddd eeee") == ["aaaa bbbb cccc dd"]
    with pytest.raises(ValueError):
        R.chunk_paper(" \n\t ")


def test_bm25_tokenizer():
    assert R.bm25_tokens("Café-Olé, GPT-4 3.5!") == ["café", "olé", "gpt", "4", "3", "5"]


def test_top_k_breaks_ties_by_lower_id_and_returns_document_order():
    s = [0.1, 0.9, 0.5, 0.9, 0.0]
    assert R.top_k(s, 1) == [1]
    assert R.top_k(s, 2) == [1, 3]
    assert R.top_k(s, 3) == [1, 2, 3]
    assert R.top_k([0, 0, 0, 0], 2) == [0, 1]
    assert R.top_k(s, 10) == [0, 1, 2, 3, 4]
    for bad, k in (([], 1), ([0.1, float("nan")], 1), (s, 0)):
        with pytest.raises(ValueError):
            R.top_k(bad, k)


# --- conditions --------------------------------------------------------------------

def test_bm25_selects_matching_chunks_and_ranks_once_for_all_k(monkeypatch):
    calls = use_papers(monkeypatch, {"P": zebra_paper()})
    out = dict(R.build("english", [example("e0", "P")], ["bm25_k1", "bm25_k3", "bm25_k5"]))
    chunks = R.chunk_paper(zebra_paper())
    (r1,), (r3,), (r5,) = out["bm25_k1"], out["bm25_k3"], out["bm25_k5"]
    assert r1["meta"]["chunk_ids"] == [1] and r1["context"] == chunks[1]      # 1 vs 2 tie -> lower id
    assert r3["meta"]["chunk_ids"] == [0, 1, 2]                                # zero-score tie -> chunk 0
    assert r3["context"] == R.SEPARATOR.join(chunks[:3])
    zero, a, b = r3["meta"]["scores"]
    assert zero == 0 and a == b > 0 and r3["meta"]["n_chunks"] == 5
    assert r5["meta"]["chunk_ids"] == [0, 1, 2, 3, 4]
    assert calls == ["P"]


def test_bm25_zero_overlap_is_reported_not_hidden(monkeypatch):
    use_papers(monkeypatch, {"P": zebra_paper()})
    rows = dict(R.build("spanish", [example("e0", "P", question="¿Dónde está la cebra?")], ["bm25_k1"]))["bm25_k1"]
    assert rows[0]["meta"] == {"chunk_ids": [0], "scores": [0.0], "n_chunks": 5}
    assert "zero_overlap=1" in R.summary("bm25_k1", rows)


def test_bm25_rejects_a_question_without_words(monkeypatch):
    use_papers(monkeypatch, {"P": zebra_paper()})
    with pytest.raises(ValueError, match="no word tokens"):
        dict(R.build("english", [example("e0", "P", question="¿?")], ["bm25_k1"]))


def test_dense_ranks_by_cosine_and_embeds_each_paper_once(monkeypatch):
    w = words(200)                      # chunks [0:100] [50:150] [100:200]
    w[10], w[190] = "big", "aligned"
    use_papers(monkeypatch, {"P": " ".join(w)})
    calls = []

    def vec(t):
        if t.startswith("Q"):
            return [1.0, 0.0]
        if "big" in t.split():
            return [10.0, 10.0]         # largest dot product, but 45 degrees off
        if "aligned" in t.split():
            return [1.0, 0.0]           # small dot product, same direction
        return [0.0, 1.0]

    def embed(texts, deployment):
        calls.append((list(texts), deployment))
        return [vec(t) for t in texts]
    monkeypatch.setattr(llm, "embed", embed)
    exs = [example("e0", "P", question="Q one"), example("e1", "P", question="Q two")]
    out = dict(R.build("english", exs, ["dense_k1", "dense_k3"], workers=2))
    assert [r["meta"]["chunk_ids"] for r in out["dense_k1"]] == [[2], [2]]
    meta = out["dense_k3"][0]["meta"]
    assert meta["chunk_ids"] == [0, 1, 2] and meta["scores"] == [0.707107, 0.0, 1.0]
    assert meta["n_chunks"] == 3 and meta["embed_tokens"]["query"] > 0 and meta["embed_tokens"]["paper_chunks"] > 0
    assert calls == [(["Q one", "Q two"], "text-embedding-3-large"),
                     (R.chunk_paper(" ".join(w)), "text-embedding-3-large")]


def test_dense_maps_scores_back_to_example_order_across_interleaved_papers(monkeypatch):
    """Papers B,A,B,A,B interleaved, scored per paper in parallel: each question points at a
    different chunk id of its own paper, so any mix-up in the regrouping changes an id."""
    papers = {"A": " ".join(words(300, "a")), "B": " ".join(words(250, "b"))}     # 5 and 4 chunks
    use_papers(monkeypatch, papers)
    starts = [f"a{50 * i}" for i in range(5)] + [f"b{50 * i}" for i in range(4)]  # first word of every chunk

    def vec(text):              # one-hot on the chunk a text is (or a question names), no zero vector
        key = text.split()[1] if text.startswith("Q ") else text.split()[0]
        return [1.0 if s == key else 0.1 for s in starts]
    monkeypatch.setattr(llm, "embed", lambda texts, deployment: [vec(t) for t in texts])
    targets = [("B", "b150", 3), ("A", "a50", 1), ("B", "b0", 0), ("A", "a200", 4), ("B", "b100", 2)]
    exs = [example(f"e{i}", title, question=f"Q {word}") for i, (title, word, _) in enumerate(targets)]
    rows = dict(R.build("english", exs, ["dense_k1"], workers=3))["dense_k1"]
    assert [r["index"] for r in rows] == ["e0", "e1", "e2", "e3", "e4"]
    assert [r["meta"]["chunk_ids"] for r in rows] == [[3], [1], [0], [4], [2]]
    assert [r["meta"]["n_chunks"] for r in rows] == [4, 5, 4, 5, 4]
    assert [r["context"] for r in rows] == [R.chunk_paper(papers[t])[c] for t, _, c in targets]


def test_pseudo_rag_none_and_full(monkeypatch):
    monkeypatch.setattr(config, "MAX_PAPER_CHARS", 10)
    calls = use_papers(monkeypatch, {"short": "tiny text", "long": "0123456789abcdef"})
    exs = [example("e0", "short", context_original=""), example("e1", "long")]
    out = dict(R.build("english", exs, ["pseudo_rag", "none"]))
    assert calls == []                  # neither condition needs the papers
    assert [(r["context"], r["meta"]) for r in out["pseudo_rag"]] == [("", {"source": "original"}),
                                                                       ("span", {"source": "original"})]
    assert [(r["context"], r["meta"]) for r in out["none"]] == [(None, {}), (None, {})]
    full = dict(R.build("english", exs, ["full"]))["full"]
    assert [r["context"] for r in full] == ["tiny text", "0123456789"]
    assert [r["meta"] for r in full] == [{"chars": 9, "truncated": False, "paper_chars": 9},
                                         {"chars": 10, "truncated": True, "paper_chars": 16}]


def test_rerun_repeats_the_original_extraction_call(monkeypatch, original_prompt):
    paper = "paper body " * 12_000                    # 132k chars: the original 120k cut applies
    use_papers(monkeypatch, {"P": paper})
    calls = []

    def chat(deployment, messages, **kwargs):
        calls.append((deployment, messages, kwargs))
        return reply("\n  Verbatim span.  \n")
    monkeypatch.setattr(llm, "chat", chat)
    (row,) = dict(R.build("english", [example("e0", "P", question="What?")], ["pseudo_rag_rerun"]))["pseudo_rag_rerun"]
    assert calls == [("gpt-5-mini", original_messages("What?", paper), {"max_completion_tokens": 4096, "cache_tag": "e0"})]  # no temperature; cache_tag is local
    assert row["context"] == "Verbatim span."
    assert row["meta"] == {k: v for k, v in reply().items() if k != "content"}


@pytest.mark.parametrize("lang", config.LANGS)
def test_original_extraction_call_is_read_from_the_competition_script(lang, original_prompt):
    paper = "0123456789" * 12_500                      # 125k chars: the original 120k cut applies
    assert R.original_extraction_call(lang, "What is X?", paper) == {
        "model": "gpt-5-mini", "max_completion_tokens": 4096, "messages": original_messages("What is X?", paper)}
    R.check_rerun_call(lang)                           # the transcription passes the check


@pytest.mark.parametrize("lang", config.LANGS)
def test_prompts_retrieval_is_the_original_request(lang):
    R.check_rerun_call(lang)


def _user_edited(question, paper, edit):
    system, user = original_messages(question, paper)
    return [system, {**user, "content": edit(user["content"])}]


DEVIATIONS = {  # one part of the request changed at a time
    "user prompt whitespace": (prompts, "retrieval", lambda q, p: _user_edited(q, p, lambda u: u.replace(" \n", "\n"))),
    "system message": (prompts, "retrieval", lambda q, p: [{"role": "system", "content": "You are helpful."},
                                                           original_messages(q, p)[1]]),
    "paper cut": (prompts, "retrieval", lambda q, p: original_messages(q, p[:100_000])),
    "max_completion_tokens": (config, "RETRIEVAL_MAX_COMPLETION_TOKENS", 16_384),
    "model": (R, "RERUN_MODEL", "gpt-4.1-mini"),
}


@pytest.mark.parametrize("deviation", DEVIATIONS)
def test_rerun_refuses_a_request_that_is_not_the_original(monkeypatch, original_prompt, deviation):
    """Checked before any condition runs: no call (llm.chat is the forbidden fake), no store at all."""
    monkeypatch.setattr(*DEVIATIONS[deviation])
    monkeypatch.setattr(data, "load_examples", lambda lang, split: [example("e0", "P")])
    use_papers(monkeypatch, {"P": zebra_paper()})
    with pytest.raises(RuntimeError, match="would not repeat the competition's extraction call"):
        R.main(["--lang", "spanish", "--split", "test", "--conditions", "none", "pseudo_rag_rerun"])
    assert not (config.RESULTS / "contexts").exists()


def test_rerun_empty_extraction_fails_loudly_and_writes_no_store(monkeypatch, original_prompt):
    monkeypatch.setattr(data, "load_examples",
                        lambda lang, split: [example("e0", "P", question="A?"), example("e1", "P", question="B?")])
    use_papers(monkeypatch, {"P": zebra_paper()})

    def chat(deployment, messages, **kwargs):
        empty = "Question: B?\n" in messages[1]["content"]
        return reply("" if empty else "span", "length" if empty else "stop")
    monkeypatch.setattr(llm, "chat", chat)
    with pytest.raises(RuntimeError, match=r"1/2 extractions are empty .*e1 \(finish_reason=length"):
        R.main(["--lang", "english", "--split", "test", "--conditions", "pseudo_rag_rerun", "--workers", "2"])
    assert not R.store_path("english", "test", "pseudo_rag_rerun").exists()


def test_missing_paper_is_an_error(monkeypatch):
    use_papers(monkeypatch, {})
    with pytest.raises(FileNotFoundError, match="no PDF"):
        dict(R.build("english", [example("e0", "absent")], ["full"]))


def test_build_rejects_duplicate_indices_and_unknown_conditions():
    with pytest.raises(ValueError, match="unique"):
        next(R.build("english", [example("e0", "P"), example("e0", "P")], ["none"]))
    with pytest.raises(ValueError, match="unknown condition"):
        next(R.build("english", [example("e0", "P")], ["bm25_k2"]))


# --- store and CLI -------------------------------------------------------------------

def test_cli_writes_ordered_stores_that_load_contexts_reads(monkeypatch, capsys):
    exs = [example("e2", "P"), example("e0", "Q"), example("e1", "P")]

    def load_examples(lang, split):
        assert (lang, split) == ("spanish", "validation")
        return [dict(e) for e in exs]
    monkeypatch.setattr(data, "load_examples", load_examples)
    use_papers(monkeypatch, {"P": zebra_paper(), "Q": " ".join(words(120, "q"))})
    R.main(["--lang", "spanish", "--split", "validation", "--conditions", "none", "pseudo_rag", "full", "bm25_k3"])
    folder = config.RESULTS / "contexts" / "spanish" / "validation"
    assert sorted(p.name for p in folder.iterdir()) == ["bm25_k3.jsonl", "full.jsonl", "none.jsonl", "pseudo_rag.jsonl"]
    rows = [json.loads(line) for line in (folder / "bm25_k3.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["index"] for r in rows] == ["e2", "e0", "e1"]
    assert rows == R.load_rows("spanish", "validation", "bm25_k3")
    assert R.load_contexts("spanish", "validation", "none") == {"e2": None, "e0": None, "e1": None}
    assert '"context": null' in (folder / "none.jsonl").read_text(encoding="utf-8")
    assert R.load_contexts("spanish", "validation", "pseudo_rag") == {"e2": "span", "e0": "span", "e1": "span"}
    out = capsys.readouterr().out
    assert "[spanish/validation] bm25_k3: n=3 null=0 empty=0" in out and "[spanish/validation] none: n=3 null=3" in out


def test_load_contexts_errors():
    with pytest.raises(FileNotFoundError, match="--lang english --split test --conditions dense_k3"):
        R.load_contexts("english", "test", "dense_k3")
    with pytest.raises(ValueError):
        R.load_contexts("english", "test", "bm25_k2")
    with pytest.raises(ValueError):
        R.load_contexts("french", "test", "none")
    path = R.store_path("english", "test", "none")
    path.parent.mkdir(parents=True)
    for text, message in (('{"index": "a", "context": null, "meta": {}}\n' * 2, "duplicate"),
                          ('{"index": "a", "context": null}\n', "malformed"), ("", "empty")):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            R.load_contexts("english", "test", "none")


@pytest.mark.parametrize("condition, bad_line, message", [
    ("none", '["index", "context", "meta"]', "malformed row"),               # a list has the right "keys"
    ("none", '{"index": "b", "context": null, "meta": []}', "meta must be an object"),
    ("none", '{"index": "b", "context": "text", "meta": {}}', "context must be null"),
    ("bm25_k3", '{"index": "b", "context": null, "meta": {}}', "context must be a string"),
    ("full", '{"index": ["b"], "context": "x", "meta": {}}', "index must be a string or an integer"),
    ("pseudo_rag", "", "not valid JSON"),                                      # a blank line
])
def test_load_rows_rejects_a_bad_row_naming_its_line(condition, bad_line, message):
    good = {"index": "a", "context": None if condition == "none" else "x", "meta": {}}
    path = R.store_path("english", "test", condition)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(good) + "\n" + bad_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}:2: {message}")):
        R.load_rows("english", "test", condition)


def test_write_store_is_atomic():
    R.write_store("english", "test", "none", [{"index": "a", "context": None, "meta": {}}])
    with pytest.raises(TypeError):
        R.write_store("english", "test", "none", [{"index": "b", "context": None, "meta": {}},
                                                  {"index": "c", "context": object(), "meta": {}}])
    assert R.load_contexts("english", "test", "none") == {"a": None}
    assert [p.name for p in R.store_path("english", "test", "none").parent.iterdir()] == ["none.jsonl"]


def test_cli_defaults_and_choices():
    args = R.parser().parse_args([])
    assert args.lang == list(config.LANGS) and args.split == list(config.SPLITS) and args.workers == 12
    assert sorted(args.conditions) == sorted(["pseudo_rag", "pseudo_rag_rerun", "bm25_k1", "bm25_k3", "bm25_k5",
                                              "dense_k1", "dense_k3", "dense_k5", "full", "none"])
    with pytest.raises(SystemExit):
        R.parser().parse_args(["--conditions", "bm25_k2"])
