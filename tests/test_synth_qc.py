"""Offline tests for chomps.synth_qc: llm.chat is faked, HTTP is faked, every
path (inputs, outputs, PDF dir, text cache) points into tmp_path, and the
task datasets are replaced by a synthetic stand-in."""
import csv
import json
import threading

import pytest
import requests

from chomps import llm
from chomps import synth_qc as q

CTX = "WetSAM reaches an average F1-score of 85.58% across the eight study regions."


def row(question, veredict="n", answer="85.58%", context=CTX):
    return {"question": question, "context": context, "answer": answer, "veredict": veredict}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def make_pdf(lines):
    """A minimal one-page PDF (Helvetica text) that pypdf can extract."""
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "\n".join(["BT", "/F1 10 Tf", "12 TL", "40 800 Td"]
                       + [f"({esc(t)}) Tj T*" for t in lines] + ["ET"]).encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for k, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % k + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


PAPER_LINES = [CTX] + [f"Filler sentence number {i} about wetland mapping with SAM." for i in range(30)]
BLEU = "Bleu: a Method for Automatic Evaluation of Machine Translation"
CORPUS = [{"id": "2601.00001v1", "title": "WetSAM:  Wetland Mapping\n with SAM", "primary": "cs.CV",
           "cats": ["cs.CV"], "published": "2026-01-16T18:59:00Z"},
          {"id": "2601.00005v2", "title": "BLEU - a method for automatic evaluation of machine translation",
           "primary": "cs.CL", "cats": ["cs.CL"], "published": "2025-12-30T10:00:00Z"},
          {"id": "2601.00007v1", "title": "Something else entirely", "primary": "cs.LG", "cats": ["cs.LG"],
           "published": "2026-01-02T00:00:00Z"}]


def write_corpus(d, entries=CORPUS):
    d.mkdir(parents=True, exist_ok=True)
    return write_jsonl(d / q.CORPUS_META_NAME, entries)
CTX_B = "The final VELI5 dataset comprises 17,522 instances split into train and test."
PAPER_B_LINES = [f"Another line {i} on explanation datasets for children." for i in range(30)] + [CTX_B]


class FakeChat:
    """Stands in for llm.chat. Support check: 'contradicted' when the answer
    contains WRONG, else 'supported'. Type check: wrong_number. `bad[(kind, repeat)]`
    overrides the reply content for that call."""

    def __init__(self, bad=None):
        self.calls, self.bad, self.lock = [], bad or {}, threading.Lock()

    def __call__(self, deployment, messages, **kw):
        user = messages[1]["content"]
        kind = "support" if user.startswith("Decide whether") else "type"
        with self.lock:
            self.calls.append({"deployment": deployment, "messages": messages, "kind": kind, **kw})
        if (kind, kw["repeat"]) in self.bad:
            content = self.bad[(kind, kw["repeat"])]
        elif kind == "support":
            answer = user.rsplit("ANSWER:\n", 1)[1]
            content = json.dumps({"verdict": "contradicted" if "WRONG" in answer else "supported",
                                  "rationale": "checked"})
        else:
            content = json.dumps({"error_found": True, "type": "Wrong Number", "rationale": "number differs"})
        return {"content": content, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 40,
                "reasoning_tokens": 30, "latency_s": 0.5, "model": "gpt-5-mini-2025-08-07", "cached": False}


@pytest.fixture
def fake_chat(monkeypatch):
    fake = FakeChat()
    monkeypatch.setattr(llm, "chat", fake)
    return fake


# --- input and ids ------------------------------------------------------------

def test_load_raw_keeps_file_order(tmp_path):
    rows = [row("In the paper a?"), row("In the paper b?", "y", "WRONG")]
    assert q.load_raw(write_jsonl(tmp_path / "raw.jsonl", rows)) == rows


@pytest.mark.parametrize("line,msg", [
    ("{not json", "invalid JSON"),
    ("[1, 2]", "not a JSON object"),
    ('{"question": "q", "context": "c", "answer": "a"}', "'veredict' missing"),
    ('{"question": "q", "context": "c", "answer": "  ", "veredict": "n"}', "'answer' missing"),
    ('{"question": "q", "context": "c", "answer": "a", "veredict": "maybe"}', "veredict must be"),
    ("", "invalid JSON"),
    ('{"question": "In the paper 2601.00001v1 and 2601.00002v1?", "context": "c", "answer": "a", "veredict": "n"}',
     "several arXiv papers"),
])
def test_load_raw_rejects_malformed_rows(tmp_path, line, msg):
    good = json.dumps(row("In the paper a?"))
    path = tmp_path / "raw.jsonl"
    path.write_text(good + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=msg) as e:
        q.load_raw(path)
    assert ":2:" in str(e.value)


def test_load_raw_rejects_empty_file(tmp_path):
    (tmp_path / "raw.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        q.load_raw(tmp_path / "raw.jsonl")


def test_question_arxiv_id():
    assert q.question_arxiv_id("In the paper 2601.11190v1, how many triples?") == "2601.11190v1"
    assert q.question_arxiv_id("In the paper 2601.1119 and 2601.1119 again?") == "2601.1119"
    assert q.question_arxiv_id("In the paper 2601.11190 (2601.11190v1), how many?") == "2601.11190v1"  # one paper
    assert q.question_arxiv_id("In the paper what fraction was annotated?") is None
    assert q.question_arxiv_id("Values 12601.11190 and 2601.111901 are not ids") is None
    with pytest.raises(ValueError, match="several arXiv papers"):
        q.question_arxiv_id("Compare 2601.11190v1 with 2601.11400v1")
    with pytest.raises(ValueError, match="several versions"):
        q.question_arxiv_id("In the paper 2601.11190v1 (2601.11190v2), how many?")
    rows = [row("In the paper 2601.11190 and 2601.11190v1?"), row("In the paper 2601.11190v1?"), row("In the paper?")]
    assert q.referenced_ids(rows) == ["2601.11190v1"]


def test_raw_path_rejects_unknown_language():
    with pytest.raises(ValueError):
        q.raw_path("french")


# --- descriptive statistics -----------------------------------------------------

def test_duplicate_stats():
    rows = [row("In the paper 2601.00001v1, what F1 does WetSAM reach?"),           # 0
            row("In the paper 2601.00001v1, what F1 does WetSAM reach?", "y"),      # 1 exact dup of 0
            row("In the paper what F1 does WetSAM reach?"),                         # 2 same core as 0
            row("En el documento, ¿cuántas regiones de estudio se usaron?", "y"),   # 3 unrelated
            row("In the paper what F1 does WetSAM reach?")]                         # 4 exact dup of 2, same example
    s = q.duplicate_stats(rows)
    assert s["questions_unique"] == 3
    assert s["examples_unique"] == 4
    assert s["q_near90_pairs"] == 6            # all pairs among rows 0, 1, 2, 4
    assert s["q_near90_pairs_opposite_veredict"] == 3   # row 1 against 0, 2, 4
    assert s["q_near90_pairs_same_veredict"] == 3
    assert (s["q_near90_rows"], s["q_near90_clusters"]) == (4, 1)


def test_length_stats():
    rows = [row("q1", "n", "one two three", "abcd"), row("q2", "y", "one", "ab cd ef")]
    s = q.length_stats(rows)
    assert (s["answer_chars_min"], s["answer_chars_max"], s["answer_words_mean"]) == (3, 13, 2.0)
    assert (s["answer_words_mean_n"], s["answer_words_mean_y"]) == (3.0, 1.0)
    assert (s["context_chars_mean"], s["context_chars_median"], s["context_words_mean"]) == (6.0, 6.0, 2.0)


def test_arxiv_stats_and_overlap():
    rows = [row("In the paper 2601.00001v1, x?"), row("In the paper 2601.00001v1, y?", "y"),
            row("In the paper 2512.00002v2, z?"), row("In the paper what?")]
    task = {"titles": [BLEU, "Other paper"], "arxiv_ids": {"2512.00002"}, "papers_with_arxiv_id": 0}
    s = q.arxiv_stats(rows, task, metadata={})
    assert (s["arxiv_rows"], s["arxiv_ids"], s["arxiv_id_months"]) == (3, 2, "2512:1 2601:1")
    assert (s["task_titles"], s["task_papers_with_arxiv_id"], s["task_arxiv_like_ids"]) == (2, 0, 1)
    assert s["overlap_ids_with_task"] == 1
    assert (s["ft_titles_known"], s["overlap_titles_with_task"], s["ft_published_range"]) == (0, None, None)
    meta = {"2601.00001v1": {"title": "BLEU - a method for automatic evaluation of machine translation",
                             "published": "2026-01-16T18:00:00Z"},
            "2512.00002v2": {"title": "Something unrelated entirely", "published": "2025-12-01T00:00:00Z"}}
    s = q.arxiv_stats(rows, task, metadata=meta)
    assert (s["ft_titles_known"], s["overlap_titles_with_task"]) == (2, 1)
    assert s["ft_published_range"] == "2025-12-01..2026-01-16"


def test_task_papers_separates_own_ids_from_citations(monkeypatch):
    acl = "https://aclanthology.org/2023.acl-long.1.pdf"
    examples = {"train": [{"title": "Paper A", "url": acl, "doi": None, "output_logits": [1.0],
                           "output_text": "As in arXiv:2106.14662v2 and 2005.14165, the model ..."},
                          {"title": "Paper B", "url": "https://arxiv.org/abs/2104.00001",
                           "doi": "10.48550/arXiv.2104.00001", "output_text": "no ids here"}],
                "validation": [{"title": "Paper A", "url": acl, "doi": None, "output_text": "12106.146620 is no id"}],
                "test": []}
    monkeypatch.setattr(q.data, "load_examples", lambda lang, split: examples[split])
    assert q.task_papers("english") == {"titles": ["Paper A", "Paper B"], "papers_with_arxiv_id": 1,
                                        "arxiv_ids": {"2106.14662", "2005.14165", "2104.00001"}}


def test_load_corpus(tmp_path):
    write_corpus(tmp_path)
    c = q.load_corpus(tmp_path)
    assert list(c) == ["2601.00001", "2601.00005", "2601.00007"]
    assert c["2601.00001"] == {"id": "2601.00001v1", "title": "WetSAM: Wetland Mapping with SAM",
                               "published": "2026-01-16T18:59:00Z"}


@pytest.mark.parametrize("line,msg", [
    ("{not json", "invalid JSON"),
    ('{"id": "2601.00009", "title": " ", "published": "2026-01-01T00:00:00Z"}', "needs an arXiv 'id'"),
    ('{"id": "paper-9", "title": "t", "published": "2026-01-01T00:00:00Z"}', "needs an arXiv 'id'"),
    ('{"id": "2601.00009", "title": "t", "published": "January"}', "needs an arXiv 'id'"),
    ('{"id": "2601.00001v2", "title": "t", "published": "2026-01-01T00:00:00Z"}', "2601.00001 repeated"),
])
def test_load_corpus_rejects_malformed_rows(tmp_path, line, msg):
    (tmp_path / q.CORPUS_META_NAME).write_text(json.dumps(CORPUS[0]) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=msg) as e:
        q.load_corpus(tmp_path)
    assert ":2:" in str(e.value)


def test_load_corpus_requires_a_non_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_arxiv_range.py"):
        q.load_corpus(tmp_path)
    (tmp_path / q.CORPUS_META_NAME).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        q.load_corpus(tmp_path)


def test_corpus_stats_and_title_scores(tmp_path):
    corpus = q.load_corpus(write_corpus(tmp_path).parent)
    task = {"titles": [BLEU, "Other paper"], "arxiv_ids": set(), "papers_with_arxiv_id": 0}
    rows = [row("In the paper 2601.00001v1, F1?"), row("In the paper 2601.00005v1, size?"),  # other version: in corpus
            row("In the paper 2601.99999v1, F1?"), row("In the paper what?")]
    assert q.corpus_stats(rows, task, corpus) == {
        "corpus_papers": 3, "corpus_published_range": "2025-12-30..2026-01-16", "arxiv_ids_in_corpus": 2,
        "overlap_titles_corpus_with_task": 1, "corpus_title_best_ratio": 99.2}
    scores = q.title_scores({b: m["title"] for b, m in corpus.items()}, task["titles"])
    assert [s[0] for s in scores] == ["2601.00005", "2601.00007", "2601.00001"] and scores[0][2] == BLEU
    assert q.title_matches({b: m["title"] for b, m in corpus.items()}, task["titles"]) == \
        [("2601.00005", CORPUS[1]["title"], BLEU, 99.2)]
    with pytest.raises(ValueError, match="no task titles"):
        q.title_scores({"2601.00001": "t"}, [])


# --- LLM audit ------------------------------------------------------------------

def test_support_prompt_never_sees_the_label():
    n, y = row("In the paper q?", "n", "ans"), row("In the paper q?", "y", "ans")
    assert q.support_messages(n) == q.support_messages(y)
    assert "wrong" not in q.support_messages(n)[1]["content"].lower()


def test_validators():
    assert q.validate_support({"verdict": " Not Addressed", "rationale": "r"})["verdict"] == "not_addressed"
    assert q.validate_type({"error_found": False, "type": None, "rationale": "r"})["type"] is None
    assert q.validate_type({"error_found": True, "type": "wrong-entity", "rationale": "r"})["type"] == "wrong_entity"
    for bad in ([], {"verdict": "partially", "rationale": "r"}, {"verdict": "supported"}):
        with pytest.raises(ValueError):
            q.validate_support(bad)
    for bad in ({"error_found": "yes", "type": "other", "rationale": "r"},
                {"error_found": False, "type": "other", "rationale": "r"},
                {"error_found": True, "type": "typo", "rationale": "r"}):
        with pytest.raises(ValueError):
            q.validate_type(bad)


def test_audit_calls_and_rows(fake_chat):
    rows = [row("In the paper 2601.00001v1, F1?"), row("In the paper F1?", "y", "WRONG 90%"), row("In the paper F1?")]
    out = q.audit("english", rows, workers=2)
    assert [c["kind"] for c in fake_chat.calls].count("support") == 3
    assert [c["kind"] for c in fake_chat.calls].count("type") == 1
    for c in fake_chat.calls:
        assert c["deployment"] == "gpt-5-mini" and "temperature" not in c and c["repeat"] == 0
        assert c["response_format"] == {"type": "json_object"}
        assert c["max_completion_tokens"] == q.AUDIT_MAX_COMPLETION_TOKENS
    assert [a["row"] for a in out] == [0, 1, 2]
    assert out[0]["arxiv_id"] == "2601.00001v1" and out[1]["arxiv_id"] is None
    assert [a["support"] for a in out] == ["supported", "contradicted", "supported"]
    assert out[0]["hallucination_type"] is None and out[0]["calls"]["type"] is None
    assert (out[1]["error_found"], out[1]["hallucination_type"]) == (True, "wrong_number")
    s = q.audit_stats(rows, out)
    assert s["audit_agreement_strict_pct"] == s["audit_agreement_lenient_pct"] == 100.0
    assert (s["audit_n_supported"], s["audit_y_contradicted"], s["type_wrong_number"]) == (2, 1, 1)
    assert (s["type_wrong_number_pct"], s["type_none_found"]) == (100.0, 0)
    assert (s["audit_calls"], s["audit_reasked_calls"], s["audit_prompt_tokens"]) == (4, 0, 400)
    assert s["audit_usd"] == round((400 * 0.25 + 160 * 2.00) / 1e6, 4)


def test_audit_stats_agreement_definitions():
    """strict: n = supported, y = contradicted; lenient also counts y = not_addressed."""
    cells = [(v, s) for v in ("n", "y") for s in q.SUPPORT_LABELS]
    call = {"attempts": 1, "prompt_tokens": 10, "completion_tokens": 5}
    audit_rows = [{"row": i, "veredict": v, "support": s, "error_found": True if v == "y" else None,
                   "hallucination_type": "other" if v == "y" else None,
                   "calls": {"support": call, "type": call if v == "y" else None}} for i, (v, s) in enumerate(cells)]
    s = q.audit_stats([row(f"In the paper {i}?", v) for i, (v, _) in enumerate(cells)], audit_rows)
    assert all(s[f"audit_{v}_{label}"] == 1 for v, label in cells)
    assert (s["audit_agreement_strict_pct"], s["audit_agreement_lenient_pct"]) == (33.33, 50.0)


def test_audit_resolves_the_question_id_before_any_call(fake_chat):
    with pytest.raises(ValueError, match="several arXiv papers"):
        q.audit("english", [row("In the paper 2601.00001v1 and 2601.00002v1?")], workers=1)
    assert fake_chat.calls == []


def test_audit_stats_rejects_misaligned_rows(fake_chat):
    rows = [row("In the paper a?"), row("In the paper b?", "y", "WRONG")]
    out = q.audit("english", rows, workers=1)
    with pytest.raises(ValueError, match="not aligned"):
        q.audit_stats(rows[::-1], out)


def test_ask_json_reasks_with_a_new_repeat(monkeypatch, capsys):
    fake = FakeChat(bad={("support", 0): "not json at all"})
    monkeypatch.setattr(llm, "chat", fake)
    parsed, call = q.ask_json(q.support_messages(row("In the paper q?")), q.validate_support, "t")
    assert parsed["verdict"] == "supported"
    assert [c["repeat"] for c in fake.calls] == [0, 1]
    assert call["attempts"] == 2 and call["prompt_tokens"] == 200 and call["latency_s"] == 1.0
    assert "invalid reply, re-asking" in capsys.readouterr().out


def test_ask_json_raises_after_max_attempts(monkeypatch):
    bad = {("support", r): json.dumps({"verdict": "maybe", "rationale": "r"}) for r in range(q.MAX_ATTEMPTS)}
    fake = FakeChat(bad=bad)
    monkeypatch.setattr(llm, "chat", fake)
    with pytest.raises(ValueError, match=f"no valid reply in {q.MAX_ATTEMPTS} attempts"):
        q.ask_json(q.support_messages(row("In the paper q?")), q.validate_support, "t")
    assert [c["repeat"] for c in fake.calls] == list(range(q.MAX_ATTEMPTS))


def test_audit_propagates_api_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("API down")
    monkeypatch.setattr(llm, "chat", boom)
    with pytest.raises(RuntimeError, match="API down"):
        q.audit("english", [row("In the paper q?")], workers=1)


# --- verbatim check ---------------------------------------------------------------

@pytest.fixture
def papers(tmp_path, monkeypatch):
    d = tmp_path / "finetuning_papers"
    d.mkdir()
    (d / "2601.00001v1.pdf").write_bytes(make_pdf(PAPER_LINES))
    (d / "2601.00005v2.pdf").write_bytes(make_pdf(PAPER_B_LINES))
    monkeypatch.setattr(q, "TEXT_CACHE", tmp_path / "text_cache")
    return d


def test_verbatim_own_paper_and_pool(papers, tmp_path):
    altered = CTX.replace("85.58", "12.34").replace("eight", "seven")
    rows = [row("In the paper 2601.00001v1, F1?"),                       # 0 verbatim in its own paper
            row("In the paper 2601.00001v1, F1?", "y", context=altered),  # 1 paraphrased
            row("In the paper what dataset size?", context=CTX_B),        # 2 no id: pool finds paper B
            row("In the paper 2601.99999v1, F1?"),                        # 3 own PDF absent
            row("In the paper 2601.00005v1, size?", context=CTX_B),       # 4 only v2 on disk: not substituted
            row("In the paper 2601.00001v1, size?", context=CTX_B)]       # 5 context is from another paper
    out = q.verbatim(rows, papers)
    assert [r["status"] for r in out] == ["scored", "scored", "no_arxiv_id", "pdf_missing", "pdf_missing", "scored"]
    assert (out[0]["partial_ratio"], out[0]["exact_substring"]) == (100.0, True)
    assert 80 < out[1]["partial_ratio"] < 95 and out[1]["exact_substring"] is False
    assert out[5]["partial_ratio"] < 95 and out[5]["exact_substring"] is False
    assert all(r["partial_ratio"] is None for r in out[2:5])
    assert [r["pool_best_id"] for r in out] == ["2601.00001v1", "2601.00001v1", "2601.00005v2",
                                                "2601.00001v1", "2601.00005v2", "2601.00005v2"]
    assert [r["pool_exact_substring"] for r in out] == [True, False, True, True, True, True]
    assert {r["pool_papers"] for r in out} == {2}
    assert sorted(p.name for p in (tmp_path / "text_cache").iterdir()) == ["2601.00001v1.txt", "2601.00005v2.txt"]
    s = q.verbatim_stats(out)
    assert (s["verbatim_scored"], s["verbatim_no_arxiv_id"], s["verbatim_pdf_missing"]) == (3, 1, 2)
    assert (s["verbatim_ge95_pct"], s["verbatim_exact_pct"]) == (33.33, 33.33)
    assert (s["pool_papers"], s["pool_ge95_pct"], s["pool_exact_pct"]) == (2, 83.33, 83.33)
    assert (s["pool_ge95_pct_no_arxiv_id"], s["pool_attribution_agree_pct"]) == (100.0, 66.67)


def test_verbatim_requires_downloaded_pdfs(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download-arxiv"):
        q.verbatim([row("In the paper 2601.00001v1, F1?")], tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no PDFs"):
        q.verbatim([row("In the paper 2601.00001v1, F1?")], tmp_path / "empty")


def test_ft_paper_text_rejects_textless_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "TEXT_CACHE", tmp_path / "text_cache")
    (tmp_path / "2601.00002v1.pdf").write_bytes(make_pdf(["tiny"]))
    with pytest.raises(ValueError, match="too short"):
        q.ft_paper_text("2601.00002v1", tmp_path)


def test_ft_paper_text_never_leaves_a_partial_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "TEXT_CACHE", tmp_path / "text_cache")
    (tmp_path / "2601.00001v1.pdf").write_bytes(make_pdf(PAPER_LINES))
    real_replace = q.os.replace

    def crash(src, dst):
        raise OSError("killed while writing")
    monkeypatch.setattr(q.os, "replace", crash)
    with pytest.raises(OSError, match="killed"):
        q.ft_paper_text("2601.00001v1", tmp_path)
    assert not (tmp_path / "text_cache" / "2601.00001v1.txt").exists()  # nothing a later read would trust
    monkeypatch.setattr(q.os, "replace", real_replace)
    text = q.ft_paper_text("2601.00001v1", tmp_path)
    assert CTX in text and [p.name for p in (tmp_path / "text_cache").iterdir()] == ["2601.00001v1.txt"]
    assert (tmp_path / "text_cache" / "2601.00001v1.txt").read_text(encoding="utf-8") == text


def test_medians_are_rounded_like_means():
    base = {"status": "scored", "arxiv_id": "2601.00001v1", "exact_substring": False, "pool_papers": 1,
            "pool_best_id": "2601.00001v1", "pool_exact_substring": False}
    rows = [{**base, "partial_ratio": p, "pool_best_ratio": b} for p, b in ((88.11, 99.51), (88.12, 99.59))]
    s = q.verbatim_stats(rows)  # raw medians: 88.11500000000001 and 99.55000000000001
    assert (s["verbatim_partial_ratio_median"], s["pool_best_ratio_median"]) == (88.12, 99.55)
    assert q.length_stats([row("q", answer="a" * 3), row("q", answer="a" * 4)])["answer_chars_median"] == 3.5


def test_verbatim_stats_without_own_paper_rows():
    s = q.verbatim_stats([{"status": "no_arxiv_id", "arxiv_id": None, "partial_ratio": None,
                           "exact_substring": None, "pool_papers": 3, "pool_best_id": "2601.00001v1",
                           "pool_best_ratio": 97.0, "pool_exact_substring": False}])
    assert s["verbatim_partial_ratio_mean"] is None and s["verbatim_exact_pct"] is None
    assert s["pool_attribution_agree_pct"] is None and s["pool_ge95_pct_no_arxiv_id"] == 100.0


# --- arXiv downloads ----------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t, self.sleeps = 100.0, []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_spacer_keeps_three_seconds_between_request_starts():
    fc = FakeClock()
    sp = q.Spacer(3.0, clock=fc.clock, sleep=fc.sleep)
    starts = []
    for work in (1.0, 0.0, 5.0, 0.0):  # waits 2 s, then 3 s, then none (5 s already passed)
        sp.wait()
        starts.append(fc.t)
        fc.t += work
    assert fc.sleeps == [2.0, 3.0]
    assert all(b - a >= 3.0 for a, b in zip(starts, starts[1:]))


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry><id>http://arxiv.org/abs/2601.00001v1</id><published>2026-01-16T18:59:00Z</published>
  <title>WetSAM: Wetland
   Mapping with SAM</title><arxiv:primary_category term="cs.CV"/></entry>
 <entry><id>http://arxiv.org/abs/2601.00003v2</id><published>2026-01-15T10:00:00Z</published>
  <title>Third</title><arxiv:primary_category term="cs.CL"/></entry>
 <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_bad</id><title>Error</title></entry>
</feed>""".encode()


class Resp:
    def __init__(self, status, content, ctype="application/pdf"):
        self.status_code, self.content, self.headers = status, content, {"content-type": ctype}


def test_parse_arxiv_atom():
    got = q.parse_arxiv_atom(ATOM)
    assert set(got) == {"2601.00001", "2601.00003"}
    assert got["2601.00001"]["title"] == "WetSAM: Wetland Mapping with SAM"
    assert got["2601.00003"] == {"id": "2601.00003v2", "title": "Third", "published": "2026-01-15T10:00:00Z",
                                 "primary_category": "cs.CL"}


def test_download_arxiv(tmp_path, capsys):
    (tmp_path / "2601.00003v2.pdf").write_bytes(b"%PDF-1.4 already here")
    fc, requested = FakeClock(), []

    def get(url, params=None, headers=None, timeout=None):
        requested.append((url, params, fc.t))
        assert headers["User-Agent"] == q.USER_AGENT and timeout == q.HTTP_TIMEOUT_S
        if url == q.ARXIV_API_URL:
            return Resp(200, ATOM, "application/atom+xml")
        if url.endswith("2601.00001v1"):
            return Resp(200, b"%PDF-1.5 body")
        return Resp(200, b"<html>captcha</html>", "text/html")

    ids = ["2601.00001v1", "2601.00002v1", "2601.00003v2"]
    with pytest.raises(RuntimeError) as e:
        q.download_arxiv(ids, tmp_path, get=get, spacer=q.Spacer(3.0, clock=fc.clock, sleep=fc.sleep))
    msg = str(e.value)
    assert "1 arXiv id(s) failed" in msg and "2601.00002v1: PDF: HTTP 200" in msg and "absent from" in msg
    assert [u for u, _, _ in requested] == ["https://arxiv.org/pdf/2601.00001v1", "https://arxiv.org/pdf/2601.00002v1",
                                            q.ARXIV_API_URL]
    assert requested[2][1] == {"id_list": ",".join(ids), "max_results": 3}
    assert all(b[2] - a[2] >= 3.0 for a, b in zip(requested, requested[1:]))
    assert (tmp_path / "2601.00001v1.pdf").read_bytes() == b"%PDF-1.5 body"
    assert not (tmp_path / "2601.00002v1.pdf").exists() and not list(tmp_path.glob("*.part"))
    meta = q.load_metadata(tmp_path)
    assert set(meta) == {"2601.00001v1", "2601.00003v2"} and meta["2601.00003v2"]["title"] == "Third"


def test_download_arxiv_resumes_and_records_network_errors(tmp_path):
    fc = FakeClock()
    spacer = q.Spacer(3.0, clock=fc.clock, sleep=fc.sleep)
    (tmp_path / "2601.00001v1.pdf").write_bytes(b"%PDF-1.4 x")
    (tmp_path / q.METADATA_NAME).write_text(json.dumps({"2601.00001v1": {"title": "t", "published": "p"}}))
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        raise requests.ConnectionError("offline")

    assert q.download_arxiv(["2601.00001v1"], tmp_path, get=get, spacer=spacer) == \
        {"fetched": 0, "on_disk_before": 1, "metadata": 1}
    assert calls == []
    with pytest.raises(RuntimeError, match="PDF: ConnectionError: offline; metadata: ConnectionError"):
        q.download_arxiv(["2601.00001v1", "2601.00004v1"], tmp_path, get=get, spacer=spacer)
    assert calls == ["https://arxiv.org/pdf/2601.00004v1", q.ARXIV_API_URL]


# --- CLI end to end -----------------------------------------------------------------

EN_ROWS = [row("In the paper 2601.00001v1, what F1 does WetSAM reach?"),
           row("In the paper 2601.00001v1, what F1 does WetSAM reach?", "y", "WRONG: 91.2%"),
           row("In the paper how many regions?", answer="eight regions"),
           row("In the paper how many regions were used?", "y", "WRONG: six")]
ES_ROWS = [row("En el documento ¿qué F1 alcanza WetSAM?", answer="85,58 %"),
           row("En el documento ¿qué F1 alcanza WetSAM?", "y", "WRONG: 70 %")]


@pytest.fixture
def cli_env(tmp_path, monkeypatch, fake_chat):
    raws = {"english": write_jsonl(tmp_path / "en.jsonl", EN_ROWS), "spanish": write_jsonl(tmp_path / "es.jsonl", ES_ROWS)}
    monkeypatch.setattr(q, "raw_path", lambda lang: raws[lang])
    monkeypatch.setattr(q, "task_papers", lambda lang: {"titles": ["A task paper"], "arxiv_ids": set(),
                                                        "papers_with_arxiv_id": 0})
    monkeypatch.setattr(q, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(q, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(q, "TEXT_CACHE", tmp_path / "text_cache")
    return tmp_path


def read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], {r[0]: r[1:] for r in rows[1:]}


def test_write_summary_keeps_other_languages(tmp_path):
    path = tmp_path / "summary.csv"
    assert q.write_summary({"spanish": {"n": 2, "pct_y": 50.0, "only_old": None}}, path) == []
    assert q.write_summary({"english": {"n": 4, "pct_y": 25.0, "only_new": 1}}, path) == ["spanish"]
    header, s = read_summary(path)
    assert header == ["metric", "english", "spanish"]
    assert list(s) == ["n", "pct_y", "only_new", "only_old"]
    assert s == {"n": ["4", "2"], "pct_y": ["25.0", "50.0"], "only_new": ["1", ""], "only_old": ["", ""]}
    assert q.write_summary({"english": {"n": 5}, "spanish": {"n": 3}}, path) == []
    assert read_summary(path) == (["metric", "english", "spanish"], {"n": ["5", "3"]})


def test_write_summary_refuses_foreign_files_and_languages(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a summary table"):
        q.write_summary({"english": {"n": 1}}, path)
    assert path.read_text(encoding="utf-8") == "a,b\n1,2\n"
    with pytest.raises(ValueError, match="unknown language"):
        q.write_summary({"french": {"n": 1}}, tmp_path / "other.csv")


def test_main_full_run(cli_env, fake_chat):
    write_corpus(cli_env / "papers")
    q.main(["--workers", "2"])
    out = cli_env / "out"
    en = [json.loads(l) for l in (out / "english_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [a["row"] for a in en] == [0, 1, 2, 3] and en[1]["hallucination_type"] == "wrong_number"
    assert len((out / "spanish_audit.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert not (out / "english_verbatim.jsonl").exists()
    header, s = read_summary(out / "summary.csv")
    assert header == ["metric", "english", "spanish"]
    assert s["n"] == ["4", "2"] and s["pct_y"] == ["50.0", "50.0"] and s["arxiv_rows"] == ["2", "0"]
    assert s["overlap_titles_with_task"] == ["", ""] and s["ft_titles_known"] == ["0", "0"]
    assert (s["task_titles"], s["task_arxiv_like_ids"], s["overlap_ids_with_task"]) == (["1", "1"], ["0", "0"], ["0", "0"])
    assert (s["corpus_papers"], s["arxiv_ids_in_corpus"]) == (["3", "3"], ["1", "0"])
    assert s["overlap_titles_corpus_with_task"] == ["0", "0"]
    assert s["audit_agreement_strict_pct"] == s["audit_agreement_lenient_pct"] == ["100.0", "100.0"]
    assert "verbatim_scored" not in s


def test_main_partial_runs_merge_the_summary(cli_env, capsys):
    write_corpus(cli_env / "papers")
    q.main(["--lang", "spanish", "--workers", "1"])
    q.main(["--lang", "english", "--workers", "1"])
    assert "kept the spanish column(s) of an earlier run" in capsys.readouterr().out
    header, s = read_summary(cli_env / "out" / "summary.csv")
    assert header == ["metric", "english", "spanish"] and s["n"] == ["4", "2"]


def test_main_fails_before_any_llm_call_without_the_corpus(cli_env, fake_chat):
    with pytest.raises(FileNotFoundError, match=q.CORPUS_META_NAME):
        q.main([])
    assert fake_chat.calls == [] and not (cli_env / "out").exists()


def test_main_rejects_an_ambiguous_question_before_any_llm_call(cli_env, fake_chat):
    write_corpus(cli_env / "papers")
    write_jsonl(cli_env / "en.jsonl", EN_ROWS + [row("In the paper 2601.00001v1 and 2601.00002v1, F1?")])
    with pytest.raises(ValueError, match=r"en\.jsonl:5: question names several arXiv papers"):
        q.main([])
    assert fake_chat.calls == [] and not (cli_env / "out").exists()


def test_main_with_verbatim(cli_env, monkeypatch, capsys):
    write_corpus(cli_env / "papers")
    (cli_env / "papers" / "2601.00001v1.pdf").write_bytes(make_pdf(PAPER_LINES))
    monkeypatch.setattr(q, "task_papers", lambda lang: {"titles": [BLEU], "arxiv_ids": set(), "papers_with_arxiv_id": 0})
    q.main(["--lang", "english", "--verbatim", "--workers", "1"])
    assert f"english: corpus paper 2601.00005 {CORPUS[1]['title']!r} matches task paper {BLEU!r}" in capsys.readouterr().out
    v = [json.loads(l) for l in (cli_env / "out" / "english_verbatim.jsonl").read_text().splitlines()]
    assert [r["status"] for r in v] == ["scored", "scored", "no_arxiv_id", "no_arxiv_id"]
    header, s = read_summary(cli_env / "out" / "summary.csv")
    assert header == ["metric", "english"]
    assert s["verbatim_scored"] == ["2"] and s["verbatim_exact_pct"] == ["100.0"]
    assert s["pool_papers"] == ["1"] and s["pool_attribution_agree_pct"] == ["100.0"]
    assert s["overlap_titles_corpus_with_task"] == ["1"] and s["corpus_title_best_ratio"] == ["99.2"]


def test_main_verbatim_fails_before_any_llm_call(cli_env, fake_chat):
    with pytest.raises(FileNotFoundError, match="--download-arxiv"):
        q.main(["--verbatim"])
    assert fake_chat.calls == []


def test_main_smoke_prints_and_writes_nothing(cli_env, fake_chat, capsys):
    q.main(["--smoke", "1", "--workers", "1"])
    printed = capsys.readouterr().out
    assert '"lang": "english"' in printed and '"lang": "spanish"' in printed and "nothing written" in printed
    assert len(fake_chat.calls) == 2 and not (cli_env / "out").exists()


def test_main_download_mode_uses_question_ids(cli_env, fake_chat, monkeypatch):
    seen = []
    monkeypatch.setattr(q, "download_arxiv", lambda ids: seen.append(ids) or {"fetched": 0})
    q.main(["--download-arxiv"])
    assert seen == [["2601.00001v1"]] and fake_chat.calls == []


@pytest.mark.parametrize("argv", [["--verbatim", "--smoke", "1"], ["--smoke", "0"],
                                  ["--download-arxiv", "--smoke", "1"], ["--lang", "french"]])
def test_main_rejects_bad_arguments(cli_env, argv):
    with pytest.raises(SystemExit):
        q.main(argv)
