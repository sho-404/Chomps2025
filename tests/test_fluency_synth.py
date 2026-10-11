"""Offline tests of chomps.fluency_synth: llm.chat is monkeypatched and every input is a
synthetic file written under tmp_path (FT_ROOT / CHECK_DIR are redirected there)."""
import hashlib
import json
import random
import re
from collections import Counter

import pytest

from chomps import config, llm, prompts
from chomps import fluency_synth as fs

ANSWERS = [
    "400 triples",
    "WetSAM achieves an average F1-score of 85.58%.",
    "They annotated 0.001% of the DocRED DS dataset, corresponding to 400 triples.",
    "VELI5 contains 17,522 instances. The splits are: training = 14,017, validation = 1,752 and test = 1,753.",
    "1) Introduce a mechanism; 2) Use graph networks to extract features; 3) Run experiments.",
    "El documento indica que se utiliza Proximal Policy Optimization (PPO) como algoritmo.",
    "2.45 seconds",
    "0.683",
]


class FakeChat:
    """Stands in for llm.chat. Generation calls (JSON response_format) get a corrupted
    answer derived from the prompt; judge calls get `vote(messages)`."""

    def __init__(self, gen=None, vote=None, finish_reason="stop"):
        self.calls = []
        self.gen = gen or (lambda m: json.dumps(
            {"answer": "BROKEN " + hashlib.md5(m[1]["content"].encode()).hexdigest()[:8]}))
        self.vote = vote or (lambda m: "y")
        self.finish_reason = finish_reason

    def __call__(self, deployment, messages, **kw):
        self.calls.append({"deployment": deployment, "messages": messages, **kw})
        content = self.gen(messages) if kw.get("response_format") else self.vote(messages)
        return {"content": content, "finish_reason": self.finish_reason, "prompt_tokens": 1,
                "completion_tokens": 1, "reasoning_tokens": 0, "latency_s": 0.0,
                "model": deployment, "cached": False}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "FT_ROOT", tmp_path)
    monkeypatch.setattr(fs, "CHECK_DIR", tmp_path / "results" / "fluency_synth")
    monkeypatch.setattr(fs, "RESAMPLE", {})
    return tmp_path


@pytest.fixture
def fake(monkeypatch):
    f = FakeChat()
    monkeypatch.setattr(llm, "chat", f)
    return f


def seed_rows(n_clean: int) -> list:
    """Factuality-FT-style rows: clean seeds interleaved with 'y' rows (which must be ignored)."""
    rows = []
    for i in range(n_clean):
        q = f"In the paper 2601.0000{i}v1, what is reported for setting {i}?"
        rows.append({"question": q, "context": f"Excerpt {i}: setting {i} reports a value.",
                     "answer": ANSWERS[i % len(ANSWERS)], "veredict": "n"})
        rows.append({"question": q, "context": "x", "answer": "a wrong value", "veredict": "y"})
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- type assignment -------------------------------------------------------------------

def test_error_types_cycle_by_seed_index():
    assert fs.ERROR_TYPES == ("repetition", "truncation", "grammar", "non_sequitur",
                              "code_switching", "gibberish")  # missing_separators dropped
    assert [fs.assign_type(k, "two words") for k in range(12)] == list(fs.ERROR_TYPES) * 2


def test_one_token_answers_always_get_repetition():
    for k in range(12):
        assert fs.assign_type(k, "0.683") == fs.assign_type(k, "Three.") == "repetition"
    assert fs.assign_type(1, "2.45 seconds") == "truncation"
    assert fs.assign_type(3, "2.45 seconds") == "non_sequitur"


# --- programmatic corruptions --------------------------------------------------------------

@pytest.mark.parametrize("name", ["repetition", "truncation"])
def test_programmatic_corruptions_are_deterministic_and_change_the_answer(name):
    assert set(fs.PROGRAMMATIC) == {"repetition", "truncation"}
    fn = fs.PROGRAMMATIC[name]
    for answer in ANSWERS:
        if name != "repetition" and len(answer.split()) < 2:
            continue  # assign_type never sends one-token answers to truncation
        outs = set()
        for k in range(40):
            out = fn(answer, random.Random(k))
            assert out == fn(answer, random.Random(k))
            assert out.strip() and not fs._same(out, answer) and "\n" not in out
            assert not fs.GLUED_BOUNDARY.search(out)  # the cue missing_separators was dropped for
            outs.add(out)
        assert len(outs) > 1


def _repetition_mode(text, out):
    unit = re.split(r"(?<=[.!?])\s+", text)[-1]
    if any(out == text + (" " + unit) * c for c in (2, 3, 4)):
        return "sentence"
    end = re.search(r"[.!?]*$", text).group()
    body = text[:len(text) - len(end)]
    if not (out.startswith(body) and out.endswith(end)):
        return None
    loop = out[len(body):len(out) - len(end)]
    for c in (2, 3, 4):
        piece = loop[:len(loop) // c]
        if piece * c == loop and piece.startswith(" ") and body.endswith(piece[1:]) \
                and 2 <= len(piece.split()) <= 5:
            return "phrase"
    return None


def test_repetition_appends_the_last_sentence_or_phrase_two_to_four_times():
    modes = set()
    for answer in ANSWERS:
        for k in range(30):
            mode = _repetition_mode(answer.strip(), fs.repetition(answer, random.Random(k)))
            assert mode is not None, (answer, k)
            modes.add(mode)
    assert modes == {"sentence", "phrase"}
    # sentence copies are separated by one space; only the last sentence repeats
    out = fs.repetition("First part. Second part.", random.Random(0))
    assert out.startswith("First part. Second part. Second part.") and out.count("First") == 1
    assert fs.repetition("0.683", random.Random(0)) in {" ".join(["0.683"] * c) for c in (3, 4, 5)}


def test_truncation_cuts_or_drops_leading_words():
    modes = set()
    for answer in ANSWERS:
        if len(answer.split()) < 2:
            continue
        text = answer.strip()
        for k in range(40):
            out = fs.truncation(answer, random.Random(k))
            if text.startswith(out):
                modes.add("cut")
                assert 0.4 * len(text) - 1 <= len(out) <= len(text) - 1
                assert not out[-1].isspace() and out[-1] not in ".!?"
                assert any(c.isalpha() for c in out.split()[-1])
            else:
                modes.add("drop_start")
                assert text.endswith(out)
                dropped = text[:len(text) - len(out)].split()
                assert 1 <= len(dropped) <= 3 and len(out.split()) >= 3 and out[0].islower()
    assert modes == {"cut", "drop_start"}
    # a remainder that would start with a capital (a fluent-looking sentence) is never kept
    for k in range(40):
        out = fs.truncation("The ShapeR Dataset Contains Objects", random.Random(k))
        assert "The ShapeR Dataset Contains Objects".startswith(out)


@pytest.mark.parametrize("answer", [
    "The CLASSLA-web 2.0 collection contains 17 billion words in seven languages.",
    "Se entrenaron y evaluaron 1.000 modelos en total.",
    "La precisión reportada para CIFAR-10 con ResNet18 es 1.00 (100%).",
    "WetSAM achieves an average F1-score of 85.58% on the test set",
])
def test_truncation_cut_never_ends_on_a_number(answer):
    """A head ending in a number reads as a complete answer with another value."""
    text, cuts = answer.strip(), 0
    for k in range(200):
        out = fs.truncation(answer, random.Random(k))
        if text.startswith(out):
            cuts += 1
            assert any(c.isalpha() for c in out.split()[-1]), (k, out)
            assert any(c.isalnum() for c in text[len(out):]), (k, out)
    assert cuts > 50


def test_truncation_moves_back_when_the_rest_is_numbers_and_raises_without_a_cut():
    for k in range(40):  # past 40 % there are only numbers: the cut moves back to "Scores:"
        assert fs.truncation("Scores: 85.58 86.12 87.01 88.40", random.Random(k)) == "Scores:"
    with pytest.raises(ValueError, match="no visible mid-sentence cut"):
        fs.truncation("0.683 0.701", random.Random(0))


# --- LLM corruptions -------------------------------------------------------------------

def test_generation_messages_per_type():
    m = fs.generation_messages("english", "grammar", "Q?", "The answer.", "CTX")
    assert m[0]["role"] == "system" and "JSON" in m[0]["content"]
    user = m[1]["content"]
    assert "Question: Q?" in user and "Original answer: The answer." in user
    assert "write in English" in user and "CTX" not in user
    es = fs.generation_messages("spanish", "code_switching", "Q?", "A.", "CTX")[1]["content"]
    assert "write in Spanish" in es and "anglicisms" in es
    assert "word-salad" in fs.generation_messages("english", "code_switching", "Q?", "A.", "C")[1]["content"]
    for variant in fs.NON_SEQUITUR_VARIANTS:
        ns = fs.generation_messages("english", "non_sequitur", "Q?", "A.", "CTX", variant)[1]["content"]
        assert ns.endswith("Excerpt from the same paper: CTX") and fs.PDF_SPACING_RULE in ns
    for t in ("grammar", "code_switching", "gibberish"):
        assert fs.PDF_SPACING_RULE not in fs.generation_messages("spanish", t, "Q?", "A.", "C")[1]["content"]
    with pytest.raises(ValueError):
        fs.generation_messages("english", "non_sequitur", "Q?", "A.", "CTX")
    with pytest.raises(ValueError):
        fs.generation_messages("english", "grammar", "Q?", "A.", "CTX", "new_questions")
    with pytest.raises(ValueError):
        fs.generation_messages("english", "repetition", "Q?", "A.", "CTX")
    with pytest.raises(ValueError):
        fs.generation_messages("french", "grammar", "Q?", "A.", "CTX")


def test_llm_corruption_call_parameters(tree, fake, monkeypatch):
    write_jsonl(fs.seed_path("english"), seed_rows(7))
    seeds = fs.load_seeds("english")
    row = fs.corrupt("english", seeds[2])  # k=2 -> grammar
    assert row["error_type"] == "grammar" and row["veredict"] == "y"
    assert row["answer"].startswith("BROKEN ") and row["question"] == seeds[2]["question"]
    call = fake.calls[-1]
    assert call["deployment"] == "gpt-5-mini"
    assert call["response_format"] == {"type": "json_object"}
    assert call.get("temperature") is None and call["repeat"] == 0
    assert call["max_completion_tokens"] == config.JUDGE_MAX_COMPLETION_TOKENS
    monkeypatch.setattr(fs, "RESAMPLE", {("english", 2): 1})
    fs.corrupt("english", seeds[2])
    assert fake.calls[-1]["repeat"] == 1


SEED = "WetSAM achieves an average F1-score of 85.58%."


@pytest.mark.parametrize("content,finish_reason", [
    ("not json", "stop"),
    ('{"text": "x"}', "stop"),
    ('["answer"]', "stop"),
    ('{"answer": ""}', "stop"),
    ('{"answer": "   "}', "stop"),
    ('{"answer": 5}', "stop"),
    (json.dumps({"answer": SEED}), "stop"),
    (json.dumps({"answer": " WetSAM achieves  an average F1-score of 85.58%. "}), "stop"),
    (json.dumps({"answer": "WetSAM achieve a average F1-score"}), "length"),
])
def test_parse_generated_raises_on_invalid_replies(content, finish_reason):
    with pytest.raises(fs.GenerationError):
        fs.parse_generated({"content": content, "finish_reason": finish_reason}, SEED, "test")


def test_parse_generated_collapses_line_breaks():
    reply = {"content": json.dumps({"answer": " WetSAM achieve\n\n a average F1-score. "}),
             "finish_reason": "stop"}
    assert fs.parse_generated(reply, SEED, "test") == "WetSAM achieve a average F1-score."


def test_invalid_llm_reply_aborts_the_run_without_writing(tree, monkeypatch):
    write_jsonl(fs.seed_path("english"), seed_rows(8))
    echo = FakeChat(gen=lambda m: json.dumps(
        {"answer": m[1]["content"].split("Original answer: ")[1].split("\n")[0]}))
    monkeypatch.setattr(llm, "chat", echo)
    with pytest.raises(fs.GenerationError, match="identical to the seed"):
        fs.generate("english", workers=2)
    assert not fs.raw_path("english").exists() and not fs.ready_path("english").exists()


# --- generation end to end ---------------------------------------------------------------

def test_generate_writes_balanced_shuffled_raw_and_ready_files(tree, fake, capsys):
    rows_in = seed_rows(16)
    write_jsonl(fs.seed_path("spanish"), rows_in)
    rows = fs.generate("spanish", workers=4)
    raw, ready = read_jsonl(fs.raw_path("spanish")), read_jsonl(fs.ready_path("spanish"))
    assert raw == rows and len(raw) == 32 and len(ready) == 32
    assert all(set(r) == {"question", "answer", "error_type", "veredict"} for r in raw)
    clean = [r for r in raw if r["veredict"] == "n"]
    bad = [r for r in raw if r["veredict"] == "y"]
    assert len(clean) == len(bad) == 16
    assert all(r["error_type"] == "none" for r in clean)
    seeds = [r for r in rows_in if r["veredict"] == "n"]
    assert sorted((r["question"], r["answer"]) for r in clean) == \
        sorted((s["question"], s["answer"]) for s in seeds)
    expected_types = [fs.assign_type(k, s["answer"]) for k, s in enumerate(seeds)]
    assert sorted(r["error_type"] for r in bad) == sorted(expected_types)
    assert expected_types[15] == "repetition"  # "0.683" at k=15 (cycle: non_sequitur)
    # the shuffle is random.Random(42) over the (clean, corrupted) pairs in seed order
    unshuffled = []
    for k, s in enumerate(seeds):
        unshuffled.append({"question": s["question"], "answer": s["answer"],
                           "error_type": "none", "veredict": "n"})
        unshuffled.append(fs.corrupt("spanish", {**s, "k": k, "line": 2 * k + 1}))
    random.Random(42).shuffle(unshuffled)
    assert raw == unshuffled
    # ready file: same order, Azure chat format built with prompts.ft_fluency
    for r, rec in zip(raw, ready):
        assert rec == {"messages": prompts.ft_fluency("spanish", r["question"], r["answer"])
                       + [{"role": "assistant", "content": r["veredict"]}]}
    out = capsys.readouterr().out
    assert "32 rows from 16 seeds" in out and "n 16 (50.0%)" in out and "per type" in out
    assert "{'truncation->repetition': 1, 'non_sequitur->repetition': 1}" in out
    assert "answers with a glued boundary: n 0, y 0" in out
    # idempotent: a rerun rewrites byte-identical files
    before = fs.raw_path("spanish").read_bytes(), fs.ready_path("spanish").read_bytes()
    fs.generate("spanish", workers=3)
    assert (fs.raw_path("spanish").read_bytes(), fs.ready_path("spanish").read_bytes()) == before


def test_generate_counts_glued_boundaries_per_class(tree, monkeypatch, capsys):
    rows = seed_rows(6)  # k = 0..5 -> one seed per type
    rows[0]["answer"] = "The splits are:1. train2. test"  # a glued clean seed (k=0, repetition)
    write_jsonl(fs.seed_path("english"), rows)
    monkeypatch.setattr(llm, "chat", FakeChat(gen=lambda m: json.dumps({"answer": "It works.It loops"})))
    fs.generate("english", workers=2)
    # n: the glued seed; y: its repetition + the four LLM rows (truncation of ANSWERS[1] is clean)
    assert "answers with a glued boundary: n 1, y 5" in capsys.readouterr().out


@pytest.mark.parametrize("bad_line", [
    json.dumps({"question": "q", "answer": "a", "veredict": "n"}),               # no context
    json.dumps({"question": "q", "context": "c", "answer": "a", "veredict": "maybe"}),
    json.dumps({"question": "q", "context": "c", "answer": "  ", "veredict": "n"}),
    json.dumps({"question": "q", "context": "c", "answer": None, "veredict": "n"}),
    "{not json",
    "",                                                                           # blank line
])
def test_load_seeds_rejects_malformed_rows(tree, bad_line):
    good = json.dumps({"question": "q", "context": "c", "answer": "a b", "veredict": "n"})
    path = fs.seed_path("english")
    path.parent.mkdir(parents=True)
    path.write_text(good + "\n" + bad_line + "\n" + good + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fs.load_seeds("english")


def test_load_seeds_needs_clean_rows(tree):
    write_jsonl(fs.seed_path("english"), [r for r in seed_rows(3) if r["veredict"] == "y"])
    with pytest.raises(ValueError, match="no veredict='n'"):
        fs.load_seeds("english")


# --- sanity check ----------------------------------------------------------------------

def _fluency_rows():
    rows = [{"question": f"Q{i}?", "answer": f"Fine answer {i}.", "error_type": "none", "veredict": "n"}
            for i in range(5)]
    rows += [{"question": f"Q{i}?", "answer": f"BROKEN answer {i}", "error_type": t, "veredict": "y"}
             for i, t in enumerate(fs.ERROR_TYPES[:5])]
    return rows


def _vote(messages):
    user = messages[1]["content"]
    if "Fine answer 0." in user:
        return "y"        # one false positive
    if "BROKEN answer 1" in user:
        return "N."       # a missed error, parsed as "n" with parse_ok=False
    return "y" if "BROKEN" in user else "n"


def test_check_scores_with_the_original_fluency_judge(tree, monkeypatch, capsys):
    write_jsonl(fs.raw_path("english"), _fluency_rows())
    judge = FakeChat(vote=_vote)
    monkeypatch.setattr(llm, "chat", judge)
    result = fs.run_check("english", n=10, workers=3)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["confusion"] == {("y", "y"): 4, ("y", "n"): 1, ("n", "y"): 1, ("n", "n"): 4}
    assert result["by_type"]["none"] == (1, 5) and result["by_type"]["truncation"] == (0, 1)
    for call in judge.calls:
        assert call["deployment"] == "gpt-5-mini" and call.get("temperature") is None
        assert call.get("response_format") is None
        assert call["max_completion_tokens"] == config.JUDGE_MAX_COMPLETION_TOKENS
    q_a = {(c["messages"][1]["content"]) for c in judge.calls}
    assert q_a == {prompts.fluency_judge("english", r["question"], r["answer"])[1]["content"]
                   for r in _fluency_rows()}
    written = read_jsonl(fs.check_path("english"))
    assert len(written) == 10 and sum(not w["parse_ok"] for w in written) == 1
    assert all(_fluency_rows()[w["line"] - 1]["answer"] == w["answer"] for w in written)
    out = capsys.readouterr().out
    assert "accuracy=0.8000 (8/10)" in out and "gold y" in out and "gold n" in out


def test_check_sample_size_and_missing_file(tree, fake):
    with pytest.raises(FileNotFoundError):
        fs.run_check("english", n=4)
    write_jsonl(fs.raw_path("english"), _fluency_rows())
    with pytest.raises(ValueError):
        fs.run_check("english", n=11)
    with pytest.raises(ValueError):
        fs.run_check("english", n=0)
    assert len(fs.run_check("english", n=4)["items"]) == 4


def _raw_rows(n_none, n_per_type):
    rows = [{"question": f"Q{i}?", "answer": f"Fine {i}.", "error_type": "none", "veredict": "n"}
            for i in range(n_none)]
    rows += [{"question": f"Q{t}{i}?", "answer": f"BROKEN {t} {i}", "error_type": t, "veredict": "y"}
             for t in fs.ERROR_TYPES for i in range(n_per_type)]
    random.Random(0).shuffle(rows)
    return rows


def test_check_sample_is_stratified_and_deterministic():
    rows = _raw_rows(100, 8)
    lines = fs.check_sample(rows, 60)
    assert lines == sorted(set(lines)) == fs.check_sample(rows, 60)
    assert Counter(rows[i]["error_type"] for i in lines) == {"none": 30, **{t: 5 for t in fs.ERROR_TYPES}}
    # 15 -> 7 fluent + 8 corrupted; the first two types take the remainder
    assert Counter(rows[i]["error_type"] for i in fs.check_sample(rows, 15)) == {
        "none": 7, "repetition": 2, "truncation": 2, "grammar": 1, "non_sequitur": 1,
        "code_switching": 1, "gibberish": 1}
    with pytest.raises(ValueError, match="needs 9 rows of type 'repetition', the file has 8"):
        fs.check_sample(rows, 108)
    with pytest.raises(ValueError, match="no corrupted rows"):
        fs.check_sample([r for r in rows if r["error_type"] == "none"], 4)


def test_check_types_scores_every_row_of_those_types(tree, fake, capsys):
    rows = _raw_rows(10, 3)
    write_jsonl(fs.raw_path("english"), rows)
    result = fs.run_check("english", types=("truncation", "repetition"), workers=2)
    assert sorted(it["error_type"] for it in result["items"]) == ["repetition"] * 3 + ["truncation"] * 3
    path = fs.check_path("english", ("repetition", "truncation"))
    assert path.name == "check_english_repetition+truncation.jsonl" and not fs.check_path("english").exists()
    assert sorted(w["line"] for w in read_jsonl(path)) == \
        [i + 1 for i, r in enumerate(rows) if r["error_type"] in ("repetition", "truncation")]
    assert "(all rows of repetition+truncation): n=6" in capsys.readouterr().out
    with pytest.raises(ValueError, match="mutually exclusive"):
        fs.run_check("english", n=4, types=("truncation",))
    with pytest.raises(ValueError, match="unknown types"):
        fs.run_check("english", types=("missing_separators",))


@pytest.mark.parametrize("bad", [
    {"error_type": "missing_separators", "veredict": "y"},  # a file from the seven-type version
    {"error_type": "none", "veredict": "y"},
    {"error_type": "truncation", "veredict": "n"},
])
def test_check_rejects_rows_this_module_does_not_produce(tree, fake, bad):
    write_jsonl(fs.raw_path("english"), _raw_rows(4, 1) + [{"question": "Q?", "answer": "A", **bad}])
    with pytest.raises(ValueError, match="regenerate the file"):
        fs.run_check("english", n=2)
    assert not fake.calls


# --- CLI ---------------------------------------------------------------------------------

def test_cli_generate_then_check(tree, fake, capsys):
    write_jsonl(fs.seed_path("english"), seed_rows(9))
    write_jsonl(fs.seed_path("spanish"), seed_rows(7))
    fs.main(["--workers", "2"])  # both languages by default
    assert len(read_jsonl(fs.raw_path("english"))) == 18
    assert len(read_jsonl(fs.raw_path("spanish"))) == 14
    fs.main(["--check", "--lang", "spanish", "--n", "4", "--workers", "2"])
    assert len(read_jsonl(fs.check_path("spanish"))) == 4
    assert not fs.check_path("english").exists()
    fs.main(["--check", "--lang", "spanish", "--types", "repetition", "--workers", "2"])
    raw = read_jsonl(fs.raw_path("spanish"))
    assert len(read_jsonl(fs.check_path("spanish", ("repetition",)))) == \
        sum(r["error_type"] == "repetition" for r in raw) == 2  # k = 0 and 6


@pytest.mark.parametrize("argv", [
    ["--n", "4"], ["--workers", "0"], ["--lang", "klingon"], ["--types", "truncation"],
    ["--check", "--n", "4", "--types", "truncation"], ["--check", "--types", "missing_separators"]])
def test_cli_rejects_bad_arguments(tree, fake, argv):
    with pytest.raises(SystemExit):
        fs.main(argv)
