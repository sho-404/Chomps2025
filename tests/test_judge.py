"""Offline tests for chomps.judge: llm.chat, data.load_examples and retrieval.load_contexts
are replaced by fakes, and the votes store lives in a temporary directory."""
import json
import os
import stat
import sys
import threading
import types

import pytest

import chomps
from chomps import config, data, judge, llm, prompts

EXAMPLES = {
    "test": [
        {"index": "en-tst-0", "prompt": "In paper A, what is X?", "output_text": "answer zero"},
        {"index": "en-tst-1", "prompt": "In paper B, what is Y?", "output_text": "answer one"},
        # same prompt and answer as en-tst-0: one cache entry per repeat, hence one call
        {"index": "en-tst-2", "prompt": "In paper A, what is X?", "output_text": "answer zero"},
        {"index": "en-tst-3", "prompt": "In paper C, what is Z?", "output_text": "  \n "},
    ],
    "validation": [
        {"index": "en-val-0", "prompt": "In paper D, what is W?", "output_text": "answer one"},
        {"index": "en-val-1", "prompt": "In paper E, what is V?", "output_text": "answer five"},
    ],
}
CONTEXTS = {  # condition -> what retrieval.load_contexts returns: {index: context}
    "pseudo_rag": {"en-tst-0": "ctx zero", "en-tst-1": "", "en-tst-2": "ctx zero",
                   "en-tst-3": "ctx three", "en-val-0": "ctx v0", "en-val-1": "ctx v1"},
    "full": {"en-tst-0": "F" * 70_000, "en-tst-1": "ctx one", "en-tst-2": "ctx two",
             "en-tst-3": "ctx three"},
}
ROW_KEYS = {"index", "repeat", "vote", "parse_ok", "raw", "prompt_tokens", "completion_tokens",
            "reasoning_tokens", "latency_s", "cached", "model", "finish_reason"}


class FakeChat:
    """Stands in for llm.chat (same signature); records calls, answers by a rule on the user text."""

    def __init__(self, reply=None, fail=None):
        self.calls, self.lock = [], threading.Lock()
        self.reply = reply or (lambda user: "y" if "answer one" in user else "n")
        self.fail = fail

    def __call__(self, deployment, messages, *, temperature=None, max_completion_tokens=None,
                 seed=None, repeat=0, response_format=None, reasoning_effort=None, cache_tag=None):
        with self.lock:
            self.calls.append({"deployment": deployment, "messages": messages, "temperature": temperature,
                               "max_completion_tokens": max_completion_tokens, "repeat": repeat,
                               "cache_tag": cache_tag,
                               "seed": seed, "response_format": response_format,
                               "reasoning_effort": reasoning_effort})
        if self.fail and self.fail(deployment, messages):
            raise RuntimeError("simulated API failure")
        return {"content": self.reply(messages[-1]["content"]), "finish_reason": "stop",
                "prompt_tokens": 120, "completion_tokens": 1, "reasoning_tokens": 0,
                "latency_s": 0.25, "model": f"{deployment}-2025-04-14", "cached": False}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Fake examples, contexts store and chat; results under tmp_path. Returns the fakes."""
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(data, "load_examples", lambda lang, split: [dict(e) for e in EXAMPLES[split]])
    ctx_calls = []

    def load_contexts(lang, split, condition):
        ctx_calls.append((lang, split, condition))
        return {e["index"]: CONTEXTS[condition][e["index"]] for e in EXAMPLES[split]}

    fake_retrieval = types.ModuleType("chomps.retrieval")
    fake_retrieval.load_contexts = load_contexts
    monkeypatch.setitem(sys.modules, "chomps.retrieval", fake_retrieval)
    monkeypatch.setattr(chomps, "retrieval", fake_retrieval, raising=False)
    chat = FakeChat()
    monkeypatch.setattr(llm, "chat", chat)
    return types.SimpleNamespace(chat=chat, ctx_calls=ctx_calls, retrieval=fake_retrieval,
                                 results=tmp_path / "results")


def cli(*args):
    judge.main(list(args))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- store paths and loaders -------------------------------------------------------

def test_votes_path_layout(env):
    p = judge.votes_path("english", "factuality", "test", "none", "gpt-4.1-nano", "judge", "0")
    assert p == env.results / "votes/english/factuality/test/none__gpt-4.1-nano__judge__t0.jsonl"
    p = judge.votes_path("spanish", "fluency", "train", "none", "ft-flu-es-mini", "ft", "1.0")
    assert p.name == "none__ft-flu-es-mini__ft__t1.0.jsonl"
    # meta.py stores its systems under any judge name
    assert judge.votes_path("english", "factuality", "test", "pseudo_rag", "meta-original-none",
                            "judge", "default").name == "pseudo_rag__meta-original-none__judge__tdefault.jsonl"


@pytest.mark.parametrize("args", [
    ("french", "factuality", "test", "none", "gpt-5-mini", "judge", "default"),
    ("english", "fluency", "test", "pseudo_rag", "gpt-5-mini", "judge", "default"),
    ("english", "factuality", "dev", "none", "gpt-5-mini", "judge", "default"),
    ("english", "factuality", "test", "bm25_k2", "gpt-5-mini", "judge", "default"),
    ("english", "factuality", "test", "none", "a__b", "judge", "default"),
    ("english", "factuality", "test", "none", "../x", "judge", "default"),
    ("english", "factuality", "test", "none", "gpt-5-mini", "prompt", "default"),
    ("english", "factuality", "test", "none", "gpt-4.1-nano", "judge", 0),
    ("english", "factuality", "test", "none", "gpt-4.1-nano", "judge", "0.0"),
])
def test_votes_path_rejects_bad_arguments(env, args):
    with pytest.raises(ValueError):
        judge.votes_path(*args)


def write_lines(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_load_votes_roundtrip(tmp_path):
    p = tmp_path / "v.jsonl"
    write_lines(p, [json.dumps({"index": i, "repeat": r, "vote": v, "extra": 1})
                    for i, v in (("a", "y"), ("b", "n")) for r in (0, 1)])
    assert judge.load_votes(p) == {0: {"a": "y", "b": "n"}, 1: {"a": "y", "b": "n"}}
    assert [r["extra"] for r in judge.load_vote_rows(p)] == [1, 1, 1, 1]


@pytest.mark.parametrize("lines", [
    [],                                                                    # empty file
    ['{"index": "a", "repeat": 0, "vote": "maybe"}'],                      # bad vote
    ['{"index": "a", "repeat": 0}'],                                       # no vote
    ['{"index": "a", "repeat": -1, "vote": "y"}'],                         # bad repeat
    ['{"index": "a", "repeat": 0, "vote": "y"}', ""],                      # blank line
    ['{"index": "a", "repeat": 0, "vote": "y"}'] * 2,                      # duplicate row
    ['{"index": "a", "repeat": 0, "vote": "y"}', '{"index": "b", "repeat": 0, "vote": "y"}',
     '{"index": "a", "repeat": 1, "vote": "y"}'],                          # repeat 1 lacks b
])
def test_load_votes_rejects_malformed_files(tmp_path, lines):
    p = tmp_path / "v.jsonl"
    write_lines(p, lines)
    with pytest.raises(ValueError):
        judge.load_votes(p)


def test_load_votes_keeps_the_stored_repeats(tmp_path):
    p = tmp_path / "v.jsonl"
    write_lines(p, ['{"index": "a", "repeat": 2, "vote": "y"}', '{"index": "a", "repeat": 0, "vote": "n"}'])
    assert judge.load_votes(p) == {0: {"a": "n"}, 2: {"a": "y"}}


def test_load_votes_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        judge.load_votes(tmp_path / "nope.jsonl")


# --- the judge protocol --------------------------------------------------------------

def test_factuality_none_rows_and_call_parameters(env):
    cli("--lang", "english", "--track", "factuality", "--split", "test", "--condition", "none",
        "--judge", "gpt-4.1-nano", "--fmt", "judge", "--temp", "0", "--repeats", "2", "--workers", "3")
    path = judge.votes_path("english", "factuality", "test", "none", "gpt-4.1-nano", "judge", "0")
    rows = read_rows(path)
    assert [(r["index"], r["repeat"]) for r in rows] == [
        (e["index"], r) for e in EXAMPLES["test"] for r in (0, 1)]
    assert all(set(r) == ROW_KEYS for r in rows)
    assert [r["vote"] for r in rows] == ["n", "n", "y", "y", "n", "n", "n", "n"]
    assert all(r["parse_ok"] and r["model"] == "gpt-4.1-nano-2025-04-14" and r["cached"] is False
               for r in rows)
    # en-tst-2 duplicates en-tst-0: 3 distinct inputs x 2 repeats; the blank answer is judged
    # (the empty-answer rule belongs to the fluency track only)
    assert len(env.chat.calls) == 8 and env.ctx_calls == []   # 4 examples x 2 repeats, no dedup
    for call in env.chat.calls:
        assert call["deployment"] == "gpt-4.1-nano" and call["temperature"] == 0.0
        assert call["max_completion_tokens"] == config.JUDGE_MAX_COMPLETION_TOKENS
        assert call["seed"] is None and call["reasoning_effort"] is None
        assert prompts.NO_CONTEXT_NOTE["english"] in call["messages"][1]["content"]
    sent = {(json.dumps(c["messages"]), c["repeat"]) for c in env.chat.calls}
    want = {(json.dumps(prompts.factuality_judge("english", None, e["prompt"], e["output_text"])), r)
            for e in EXAMPLES["test"] for r in (0, 1)}
    assert sent == want
    assert judge.load_votes(path) == {r: {"en-tst-0": "n", "en-tst-1": "y", "en-tst-2": "n",
                                          "en-tst-3": "n"} for r in (0, 1)}


def test_temperature_mapping(env):
    judge.run("english", "fluency", ["validation"], ["none"], ["gpt-5-mini"], "judge", "default", 1)
    judge.run("english", "fluency", ["validation"], ["none"], ["gpt-4.1-mini"], "judge", "1.0", 1)
    temps = {(c["deployment"], c["temperature"]) for c in env.chat.calls}
    assert temps == {("gpt-5-mini", None), ("gpt-4.1-mini", 1.0)}


def test_reasoning_judge_with_temperature_fails_before_any_call(env):
    with pytest.raises(ValueError, match="temperature"):
        cli("--lang", "english", "--track", "factuality", "--split", "test", "--condition", "none",
            "--judge", "gpt-4.1-nano", "gpt-5-mini", "--fmt", "judge", "--temp", "0", "--repeats", "1")
    assert env.chat.calls == [] and not env.results.joinpath("votes").exists()


def test_empty_context_is_an_automatic_y(env):
    judge.run("english", "factuality", ["test"], ["pseudo_rag"], ["gpt-4.1-mini"], "judge", "0", 2)
    rows = read_rows(judge.votes_path("english", "factuality", "test", "pseudo_rag",
                                      "gpt-4.1-mini", "judge", "0"))
    auto = [r for r in rows if r["index"] == "en-tst-1"]
    assert auto == [{"index": "en-tst-1", "repeat": r, "vote": "y", "parse_ok": True,
                     "raw": "AUTO_EMPTY_CONTEXT", "prompt_tokens": 0, "completion_tokens": 0,
                     "reasoning_tokens": 0, "latency_s": 0.0, "cached": False, "model": "",
                     "finish_reason": None} for r in (0, 1)]
    assert env.ctx_calls == [("english", "test", "pseudo_rag")]
    sent = [c["messages"][1]["content"] for c in env.chat.calls]
    assert len(sent) == 6 and not any("answer one" in s for s in sent)  # 3 non-empty contexts x 2 repeats
    assert all("ctx zero" in s or "ctx three" in s for s in sent)


def test_context_cap_is_60k_except_full(env):
    long_ctx = "F" * 70_000
    assert config.MAX_CONTEXT_CHARS < len(long_ctx) < config.MAX_PAPER_CHARS
    env.retrieval.load_contexts = lambda lang, split, cond: {e["index"]: long_ctx for e in EXAMPLES[split]}
    for cond, cap in (("full", config.MAX_PAPER_CHARS), ("bm25_k3", config.MAX_CONTEXT_CHARS)):
        env.chat.calls.clear()
        judge.run("english", "factuality", ["test"], [cond], ["gpt-4.1-nano"], "judge", "0", 1)
        want = {json.dumps(prompts.factuality_judge("english", long_ctx, e["prompt"], e["output_text"],
                                                    max_chars=cap)) for e in EXAMPLES["test"]}
        assert {json.dumps(c["messages"]) for c in env.chat.calls} == want


def test_ft_format_uses_prompt_as_question(env):
    judge.run("english", "factuality", ["test"], ["pseudo_rag"], ["ft-fact-en-mini"], "ft", "1.0", 1)
    judge.run("english", "fluency", ["test"], ["none"], ["ft-flu-en-nano"], "ft", "1.0", 1)
    sent = {json.dumps(c["messages"]) for c in env.chat.calls}
    ex = EXAMPLES["test"]
    want = {json.dumps(prompts.ft_factuality("english", CONTEXTS["pseudo_rag"][e["index"]],
                                             e["prompt"], e["output_text"]))
            for e in ex if CONTEXTS["pseudo_rag"][e["index"]]}
    want |= {json.dumps(prompts.ft_fluency("english", e["prompt"], e["output_text"]))
             for e in ex if e["output_text"].strip()}
    assert sent == want


def test_ft_format_without_context_is_rejected(env):
    with pytest.raises(ValueError, match="context-free"):
        judge.run("english", "factuality", ["test"], ["none"], ["ft-fact-en-mini"], "ft", "0", 1)
    assert env.chat.calls == []


def test_fluency_track(env):
    env.retrieval.load_contexts = lambda *a: pytest.fail("fluency must not load contexts")
    judge.run("english", "fluency", ["test"], ["none"], ["gpt-4.1-nano"], "judge", "0", 1)
    rows = read_rows(judge.votes_path("english", "fluency", "test", "none", "gpt-4.1-nano", "judge", "0"))
    assert [(r["vote"], r["raw"]) for r in rows] == [("n", "n"), ("y", "y"), ("n", "n"),
                                                     ("y", "AUTO_EMPTY_ANSWER")]
    want = {json.dumps(prompts.fluency_judge("english", e["prompt"], e["output_text"]))
            for e in EXAMPLES["test"][:3]}
    assert {json.dumps(c["messages"]) for c in env.chat.calls} == want
    with pytest.raises(ValueError, match="fluency"):
        judge.run("english", "fluency", ["test"], ["pseudo_rag"], ["gpt-4.1-nano"], "judge", "0", 1)


def test_parse_vote_is_recorded(env):
    replies = {"answer zero": "Yes.", "answer one": " N \n"}
    env.chat.reply = lambda user: next((v for k, v in replies.items() if k in user), "")
    judge.run("english", "factuality", ["test"], ["none"], ["gpt-4.1-nano"], "judge", "0", 1)
    rows = read_rows(judge.votes_path("english", "factuality", "test", "none", "gpt-4.1-nano", "judge", "0"))
    assert [(r["vote"], r["parse_ok"], r["raw"]) for r in rows] == [
        ("y", False, "Yes."), ("n", True, " N \n"), ("y", False, "Yes."), ("y", False, "")]


@pytest.mark.parametrize("contexts, error", [
    (lambda ctx: dict(list(ctx.items())[:-1]), ValueError),                  # an example missing
    (lambda ctx: {**ctx, "en-tst-9": "c"}, ValueError),                      # an unknown index
    (lambda ctx: list(ctx.values()), ValueError),                            # not {index: context}
    (lambda ctx: {**ctx, "en-tst-2": None}, TypeError),                      # null context
])
def test_bad_contexts_store_fails_before_any_call(env, contexts, error):
    ctx = {e["index"]: "c" for e in EXAMPLES["test"]}
    env.retrieval.load_contexts = lambda lang, split, cond: contexts(ctx)
    with pytest.raises(error):
        judge.run("english", "factuality", ["test"], ["none", "pseudo_rag"], ["gpt-4.1-nano"],
                  "judge", "0", 1)
    assert env.chat.calls == []


# --- several combinations in one call --------------------------------------------------

MULTI = ("--lang", "english", "--track", "factuality", "--split", "test", "validation",
         "--condition", "none", "pseudo_rag", "--judge", "gpt-4.1-nano", "gpt-4.1-mini",
         "--fmt", "judge", "--temp", "0", "--repeats", "2", "--workers", "4")


def test_several_combinations_each_get_a_complete_store(env):
    cli(*MULTI)
    for split in ("test", "validation"):
        for cond in ("none", "pseudo_rag"):
            for j in ("gpt-4.1-nano", "gpt-4.1-mini"):
                votes = judge.load_votes(judge.votes_path("english", "factuality", split, cond, j, "judge", "0"))
                assert sorted(votes) == [0, 1]
                assert list(votes[0]) == [e["index"] for e in EXAMPLES[split]]
    assert len(list(env.results.rglob("*.jsonl"))) == 8 and not list(env.results.rglob("*.tmp"))
    assert env.ctx_calls == [("english", "test", "pseudo_rag"), ("english", "validation", "pseudo_rag")]
    # rerunning rewrites the same stores
    before = {p: p.read_text() for p in env.results.joinpath("votes").rglob("*.jsonl")}
    cli(*MULTI)
    assert {p: p.read_text() for p in env.results.joinpath("votes").rglob("*.jsonl")} == before


def test_failed_combination_keeps_finished_stores_and_writes_nothing_partial(env):
    env.chat.fail = lambda deployment, messages: (deployment == "gpt-4.1-mini"
                                                  and "paper D" in messages[1]["content"])
    with pytest.raises(RuntimeError, match="simulated"):
        cli(*MULTI)
    written = {p.name + "@" + p.parent.name for p in env.results.joinpath("votes").rglob("*.jsonl")}
    # order: split, then condition, then judge; validation/none/gpt-4.1-mini is the first to fail
    assert written == {"none__gpt-4.1-nano__judge__t0.jsonl@test", "none__gpt-4.1-mini__judge__t0.jsonl@test",
                       "pseudo_rag__gpt-4.1-nano__judge__t0.jsonl@test",
                       "pseudo_rag__gpt-4.1-mini__judge__t0.jsonl@test",
                       "none__gpt-4.1-nano__judge__t0.jsonl@validation"}
    assert not [p for p in env.results.rglob("*.tmp")]


@pytest.mark.parametrize("argv", [
    ["--judge", "gpt-4.1-nano", "gpt-4.1-nano"],     # repeated value
    ["--judge", "gpt-4o"],                            # unknown judge
    ["--temp", "0.5"],                                # temp outside default/0/1.0
    ["--repeats", "0"],
])
def test_cli_rejects_bad_arguments(env, argv):
    base = {"--lang": ["english"], "--track": ["factuality"], "--split": ["test"],
            "--condition": ["none"], "--judge": ["gpt-4.1-nano"], "--fmt": ["judge"],
            "--temp": ["0"], "--repeats": ["1"]}
    base[argv[0]] = argv[1:]
    with pytest.raises((SystemExit, ValueError)):
        cli(*[x for k, v in base.items() for x in (k, *v)])
    assert env.chat.calls == []


# --- writing and replacing stores -------------------------------------------------------

STORE_ARGS = ("english", "factuality", ["test"], ["none"], ["gpt-4.1-nano"], "judge", "0")
FLAGS = ("--lang", "english", "--track", "factuality", "--split", "test", "--condition", "none",
         "--judge", "gpt-4.1-nano", "--fmt", "judge", "--temp", "0")


def store_path():
    return judge.votes_path("english", "factuality", "test", "none", "gpt-4.1-nano", "judge", "0")


@pytest.mark.parametrize("umask, mode", [(0o022, 0o644), (0o002, 0o664)])
def test_store_permissions_follow_the_umask(env, umask, mode):
    old = os.umask(umask)
    try:
        judge.run(*STORE_ARGS, 1)
    finally:
        os.umask(old)
    assert stat.S_IMODE(store_path().stat().st_mode) == mode


def test_failed_write_keeps_the_previous_store(tmp_path):
    path = tmp_path / "votes" / "s.jsonl"
    path.parent.mkdir()
    path.write_text("previous\n", encoding="utf-8")
    with pytest.raises(TypeError):  # the second row is not JSON-serialisable
        judge._write_rows(path, [{"index": "a"}, {"index": object()}])
    assert path.read_text(encoding="utf-8") == "previous\n"
    assert [p.name for p in path.parent.iterdir()] == ["s.jsonl"]


def test_dropping_stored_repeats_needs_overwrite(env, capsys):
    path = store_path()
    judge.run(*STORE_ARGS, 3)
    stored, n_calls = path.read_text(), len(env.chat.calls)
    with pytest.raises(ValueError, match=r"overwrite[^$]*repeats \[0, 1, 2\]"):
        judge.run(*STORE_ARGS, 1)
    with pytest.raises(ValueError, match="--overwrite"):
        cli(*FLAGS, "--repeats", "2")
    # refused before any call, store untouched
    assert path.read_text() == stored and len(env.chat.calls) == n_calls
    judge.run(*STORE_ARGS, 3)  # the same repeats: no flag needed
    judge.run(*STORE_ARGS, 4)  # more repeats extend the store
    assert sorted(judge.load_votes(path)) == [0, 1, 2, 3]
    capsys.readouterr()
    cli(*FLAGS, "--repeats", "1", "--overwrite")
    assert sorted(judge.load_votes(path)) == [0]
    assert f"OVERWRITE {path}: stored repeats [0, 1, 2, 3] -> [0]" in capsys.readouterr().out


def test_unreadable_existing_store_fails_before_any_call(env):
    path = store_path()
    path.parent.mkdir(parents=True)
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not JSON"):
        judge.run(*STORE_ARGS, 1, overwrite=True)
    assert env.chat.calls == [] and path.read_text(encoding="utf-8") == "not json\n"


def test_run_prints_how_many_examples_share_an_input(env, capsys):
    judge.run(*STORE_ARGS, 2)  # en-tst-2 repeats en-tst-0's prompt and answer
    out = capsys.readouterr().out
    assert "4 examples (1 share an earlier example's input, sampled independently) x 2 repeats, 8 calls" in out
    judge.run("english", "fluency", ["validation"], ["none"], ["gpt-4.1-nano"], "judge", "0", 1)
    assert "2 examples (0 share an earlier example's input, sampled independently) x 1 repeats" in capsys.readouterr().out


# --- integration with the real retrieval.load_contexts ------------------------------------

def test_reads_a_design_contexts_store_through_retrieval(monkeypatch, tmp_path):
    """The real chomps.retrieval reads a store written as DESIGN.md specifies it."""
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(data, "load_examples", lambda lang, split: [dict(e) for e in EXAMPLES[split]])
    chat = FakeChat()
    monkeypatch.setattr(llm, "chat", chat)
    args = ("english", "factuality", ["test"], ["pseudo_rag"], ["gpt-4.1-nano"], "judge", "0", 1)
    with pytest.raises(FileNotFoundError):
        judge.run(*args)
    assert chat.calls == []
    store = tmp_path / "results/contexts/english/test/pseudo_rag.jsonl"
    store.parent.mkdir(parents=True)
    write_lines(store, [json.dumps({"index": e["index"], "context": CONTEXTS["pseudo_rag"][e["index"]],
                                    "meta": {"source": "original"}}) for e in EXAMPLES["test"]])
    judge.run(*args)
    rows = read_rows(judge.votes_path("english", "factuality", "test", "pseudo_rag", "gpt-4.1-nano", "judge", "0"))
    assert [r["raw"] for r in rows] == ["n", "AUTO_EMPTY_CONTEXT", "n", "n"]


def test_conditions_match_retrieval():
    from chomps import retrieval
    assert set(judge.CONDITIONS) == set(retrieval.CONDITIONS)
