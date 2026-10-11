"""Synthetic fluency fine-tuning data (DESIGN.md, "Fluency fine-tuning data").

Seeds are the clean rows (`veredict == "n"`) of each language's factuality fine-tuning
file `code/{lang}/finetuning/finetune_dataset_raw.jsonl`, in file order and kept
verbatim (duplicates included). Every seed yields two rows: itself as a fluent negative
(`error_type="none"`, `veredict="n"`) and one corrupted copy of its answer
(`veredict="y"`), so the classes are 1:1 by construction.

Protocol decisions
* Six error types, not DESIGN's seven: `missing_separators` ("1. A2. B") is left out.
  The SHROOM-CAP outputs lost their line breaks, so a glued sentence or list boundary
  (`GLUED_BOUNDARY`, "are:1. BERT2. GPT-2") occurs in 38 % (English) and 62 % (Spanish)
  of the fluent test outputs, but in none of the seeds. Gluing seeds would put that cue
  in the positive class only and teach the judges that glued text is broken, against
  the gold labels. For the same reason the repetition copies are separated by a space.
  Every run prints how many answers of each class contain a glued boundary.
* Seed index `k` = 0-based position among the seeds. Its error type is
  `ERROR_TYPES[k % 6]`, which balances the six types to within one item. Exception: a
  one-token answer ("0.683", "Three.") always gets `repetition`, the only type that
  visibly breaks a bare value without changing it. A cut of "0.01" gives "0.0", and the
  LLM types are asked for about one word, where a different answer ("72.4%") is a
  factual error, not a fluency error. The run summary counts these moves.
* Every random choice for seed k comes from `random.Random(k)`, so a programmatic
  corruption is a pure function of (answer, k). An LLM corruption is a pure function of
  the cached gpt-5-mini reply: `llm.chat` caches by arguments, and `repeat` is 0.
* LLM corruptions ask gpt-5-mini for `{"answer": ...}` (JSON-object output, no
  temperature). `GenerationError` is raised if the reply is cut off, is not JSON, lacks
  "answer", or gives an answer that is empty or identical to the seed (whitespace
  ignored). The bad reply stays in the cache, so a rerun hits the same failure.
  Re-drawing a seed must be a deliberate act, recorded in `RESAMPLE`.
* Line breaks inside an LLM answer are collapsed to a single space. Neither the seeds
  nor the SHROOM-CAP outputs contain line breaks, so leaving them would give the
  positive class a spurious cue.
* Rows are shuffled with `random.Random(42)`. The raw and ready files are written in
  the same order (line i of one matches line i of the other) and rewritten in full on
  every run, from the cache.

Sanity check (`--check`): gpt-5-mini scores raw-file rows with the original
`prompts.fluency_judge`, with the question as the user prompt. By default it scores 60
rows drawn with `random.Random(42)` and stratified by type: half fluent rows, half spread
evenly over the error types, so every type is scored. `--types` scores every row of the
listed types instead, which is what a per-type detection rate needs. Prints accuracy,
the confusion matrix and the per-type detection rate, and writes the scored items to
`../results/fluency_synth/check_{lang}.jsonl` (`check_{lang}_{type+type}.jsonl` with --types).

    python -m chomps.fluency_synth [--lang english spanish] [--workers 12]
    python -m chomps.fluency_synth --check [--lang ...] [--n 60 | --types T [T ...]] [--workers 12]
"""
import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from . import config, llm, prompts

ERROR_TYPES = ("repetition", "truncation", "grammar", "non_sequitur", "code_switching", "gibberish")
LLM_TYPES = ("grammar", "non_sequitur", "code_switching", "gibberish")
# A sentence or list boundary that lost the space after it ("are:1. BERT2. GPT-2").
GLUED_BOUNDARY = re.compile(r"[a-z)\]%][.:;!?][A-Z0-9¿(]")
DEPLOYMENT = "gpt-5-mini"  # generator of the LLM corruptions and judge of the sanity check
SHUFFLE_SEED = 42
CHECK_SEED = 42
CHECK_N = 60

FT_ROOT = config.CODE_ROOT  # files live in code/{lang}/finetuning/
CHECK_DIR = config.RESULTS / "fluency_synth"

# (lang, k) -> llm.chat `repeat` index, for a deliberate re-draw of a seed whose cached
# gpt-5-mini reply failed validation. Empty unless a generation run raised.
RESAMPLE: dict = {}


class GenerationError(ValueError):
    """An LLM corruption failed validation (see the module docstring)."""


def _check_lang(lang: str) -> None:
    if lang not in config.LANGS:
        raise ValueError(f"lang must be one of {config.LANGS}, got {lang!r}")


def _ft_dir(lang: str) -> Path:
    _check_lang(lang)
    return FT_ROOT / lang / "finetuning"


def seed_path(lang: str) -> Path:
    return _ft_dir(lang) / "finetune_dataset_raw.jsonl"


def raw_path(lang: str) -> Path:
    return _ft_dir(lang) / "finetune_fluency_raw.jsonl"


def ready_path(lang: str) -> Path:
    return _ft_dir(lang) / "finetune_fluency_ready.jsonl"


def check_path(lang: str, types: tuple = None) -> Path:
    """check_{lang}.jsonl for the sample, check_{lang}_{type+type}.jsonl for --types."""
    _check_lang(lang)
    return CHECK_DIR / (f"check_{lang}.jsonl" if types is None else f"check_{lang}_{'+'.join(types)}.jsonl")


def _read_jsonl(path: Path, keys: tuple) -> list:
    """Rows of a JSONL file. A blank line, invalid JSON, or a missing or non-string
    field raises, naming the line."""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: invalid JSON line") from e
        if not isinstance(r, dict) or any(not isinstance(r.get(key), str) for key in keys):
            raise ValueError(f"{path}:{i}: every row needs string fields {keys}")
        rows.append(r)
    return rows


def load_seeds(lang: str) -> list:
    """The clean (veredict "n") rows of the factuality FT file, in file order:
    {k, line, question, answer, context}. Raises on malformed rows or an empty answer."""
    path = seed_path(lang)
    seeds = []
    for line, r in enumerate(_read_jsonl(path, ("question", "context", "answer", "veredict")), start=1):
        if r["veredict"] not in ("y", "n"):
            raise ValueError(f"{path}:{line}: veredict must be 'y' or 'n', got {r['veredict']!r}")
        if r["veredict"] == "y":
            continue
        if not r["question"].strip() or not r["answer"].strip():
            raise ValueError(f"{path}:{line}: clean seed with an empty question or answer")
        seeds.append({"k": len(seeds), "line": line, "question": r["question"],
                      "answer": r["answer"], "context": r["context"]})
    if not seeds:
        raise ValueError(f"{path}: no veredict='n' rows to use as seeds")
    return seeds


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


# --- programmatic corruptions (pure functions of the answer and random.Random(k)) ----

_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+")


def repetition(answer: str, rng: random.Random) -> str:
    """Degenerate looping: the last sentence is repeated 2-4 more times, each copy after
    one space ("...85.58%. WetSAM achieves ... 85.58%. WetSAM achieves ..."). With p=0.5,
    when that sentence has more than two words, only its last 2-5 words loop, and the
    answer's final punctuation stays once at the end ("... of 85.58% of 85.58% of 85.58%.").
    DESIGN says "without separators"; the space stays because a glued boundary would be
    a class cue (module docstring)."""
    text = answer.strip()
    unit = _SENTENCE_GAP.split(text)[-1]
    copies, phrase_mode, n_words = rng.randint(2, 4), rng.random() < 0.5, rng.randint(2, 5)
    words = unit.split()
    if phrase_mode and len(words) > 2:
        end = re.search(r"[.!?]*$", text).group()
        phrase = " ".join(words[-min(n_words, len(words) - 1):])
        body, phrase = text[:len(text) - len(end)], phrase[:len(phrase) - len(end)]
        return body + (" " + phrase) * copies + end
    return text + (" " + unit) * copies


def _visible_cut(text: str, k: int) -> bool:
    """Whether text[:k] reads as cut off mid-sentence. It must drop a word or number of
    the answer, must not end on whitespace or a sentence end, and its last token must
    contain a letter. A head that ends in a number reads as a complete answer with
    another value: "...contains 1" for "...contains 17 billion words", "...of 85.58" for
    "...of 85.58% on the test set"."""
    head = text[:k].split()
    return (bool(head) and not text[k - 1].isspace() and text[k - 1] not in ".!?"
            and any(c.isalpha() for c in head[-1]) and any(c.isalnum() for c in text[k:]))


def truncation(answer: str, rng: random.Random) -> str:
    """Output cut off, or started, mid-sentence; the mode is drawn with p=0.5.
    `cut` aims at 40-70 % of the characters and may split a word, as a token limit does.
    From that point it moves forward to the first `_visible_cut`, or back to the nearest
    one if the rest of the answer has none; an answer without one raises.
    `drop_start` removes the first 1-3 words and keeps the original casing. It is used
    only when at least three words remain and the remainder starts with a lowercase
    letter; otherwise `cut` is used. Dropping "The" from "The ShapeR dataset contains 178
    objects." would leave a fluent sentence labelled as broken."""
    text = answer.strip()
    drop_start, n_drop, frac = rng.random() < 0.5, rng.randint(1, 3), rng.uniform(0.40, 0.70)
    words = list(re.finditer(r"\S+", text))
    if drop_start and len(words) - n_drop >= 3 and text[words[n_drop].start()].islower():
        return text[words[n_drop].start():]
    start = min(max(round(len(text) * frac), 1), len(text) - 1)
    for k in (*range(start, len(text)), *range(start - 1, 0, -1)):
        if _visible_cut(text, k):
            return text[:k]
    raise ValueError(f"truncation: no visible mid-sentence cut in {answer!r}")


PROGRAMMATIC = {"repetition": repetition, "truncation": truncation}


def assign_type(k: int, answer: str) -> str:
    """ERROR_TYPES[k % 6], or `repetition` for a one-token answer (module docstring)."""
    return "repetition" if len(answer.split()) < 2 else ERROR_TYPES[k % len(ERROR_TYPES)]


# --- LLM corruptions -----------------------------------------------------------------

GEN_SYSTEM = (
    "You create training data for a classifier that detects fluency errors in answers to "
    "questions about scientific papers. You receive a question and a correct, fluent answer, "
    "and you rewrite the answer so that it contains one specific kind of fluency error. "
    'Reply with a JSON object of the form {"answer": "<rewritten answer>"} and nothing else.')

NON_SEQUITUR_VARIANTS = ("other_question", "new_questions")

LLM_INSTRUCTIONS = {
    ("grammar", "english"):
        "Keep the meaning and every fact (numbers, names, technical terms), but inject two to "
        "four grammatical mistakes: subject-verb agreement errors, wrong verb tenses, wrong or "
        "missing prepositions or articles, and one or two typos. A very short answer still "
        "needs at least one such mistake.",
    ("grammar", "spanish"):
        "Keep the meaning and every fact (numbers, names, technical terms), but inject two to "
        "four grammatical mistakes: gender or number agreement errors between articles, nouns "
        "and adjectives, wrong verb tense or person, wrong prepositions, and one or two typos or "
        "missing accents. A very short answer still needs at least one such mistake.",
    ("non_sequitur", "other_question"):
        "Do not answer the question that was asked. Instead, write a fluent, grammatical answer "
        "to a different question about the same paper, for example about another detail in the "
        "excerpt below (a dataset, a method, a setting or a result). It must not contain the "
        "information that answers the question asked, and it must not say that it answers a "
        "different question.",
    ("non_sequitur", "new_questions"):
        "Do not answer the question. Start as if you were about to answer it (you may restate "
        "its topic), but instead of giving the answer, ask one to three new questions about the "
        "paper. The text must be grammatical; its only flaw is that it never answers the question.",
    ("code_switching", "english"):
        "Make two to four word-salad substitutions: replace words or short phrases with words "
        "that do not fit (unrelated nouns or verbs, wrong parts of speech, malapropisms), so "
        "that the sentence frame stays recognisable but no longer reads correctly. Leave the "
        "other words, including numbers and technical terms, unchanged.",
    ("code_switching", "spanish"):
        "Introduce code-switching: render several ordinary words or phrases (verbs, connectors, "
        "common nouns; not technical terms, which Spanish papers often keep in English anyway) "
        "as untranslated English fragments or needless anglicisms, e.g. 'performa mejor', "
        "'los results muestran', 'usando el training set', switching language mid-sentence in a "
        "way a careful Spanish writer would not. Keep the facts and numbers. Use at least three "
        "switched fragments (one or two if the answer is very short).",
    ("gibberish", None):
        "Turn the answer into a word salad, like the output of a malfunctioning language model: "
        "reuse some topical words of the question and the answer (technical terms, names, numbers) "
        "and mix them with ordinary words in an order without grammatical structure, so that no "
        "clear statement can be read from it.",
}
# Only non_sequitur sees the excerpt: raw PDF text, sometimes with glued words ("merely0.001%of").
PDF_SPACING_RULE = ("- The excerpt is raw text extracted from a PDF, where words and sentences "
                    "sometimes run together without spaces; write yours with normal spacing.")


def generation_messages(lang: str, error_type: str, question: str, answer: str, context: str,
                        variant: str = None) -> list:
    """Chat messages asking gpt-5-mini for one corruption. `variant` is the
    non_sequitur sub-mode (NON_SEQUITUR_VARIANTS) and must be None for every other type.
    Only non_sequitur sees the context excerpt: it needs material from the same paper.
    The prompt gives the seed's word count as the target length, and it allows no
    remarks about the text. Both rules came from the smoke test. Without the first,
    non_sequitur answers came out ~6x longer than their seeds. Without the second,
    gibberish described itself ("frase sin unión coherente"). Words such as
    "incoherent" are also kept out of the instructions, because the model echoed them.
    non_sequitur also gets PDF_SPACING_RULE, so that glued words of its excerpt are not
    copied into the answer. Any of these would give the positive class a cue that has
    nothing to do with fluency."""
    _check_lang(lang)
    if error_type not in LLM_TYPES:
        raise ValueError(f"not an LLM error type: {error_type!r}")
    if error_type == "non_sequitur":
        if variant not in NON_SEQUITUR_VARIANTS:
            raise ValueError(f"non_sequitur needs a variant from {NON_SEQUITUR_VARIANTS}, got {variant!r}")
        key = (error_type, variant)
    elif variant is not None:
        raise ValueError(f"only non_sequitur takes a variant, got {variant!r} for {error_type!r}")
    else:
        key = (error_type, None if error_type == "gibberish" else lang)
    n_words = len(answer.split())
    rules = [f"- Apart from what these instructions ask for, write in {lang.capitalize()}.",
             "- Keep it to one paragraph without line breaks and about as long as the original "
             f"answer (around {n_words} word{'s' if n_words != 1 else ''}).",
             "- The rewritten answer must clearly differ from the original answer.",
             "- Output only the rewritten answer text: no remarks about its quality, its errors or "
             "how it was written, and no quotes around it."]
    if error_type == "non_sequitur":
        rules.append(PDF_SPACING_RULE)
    lines = [f"Rewrite the original answer below as follows. {LLM_INSTRUCTIONS[key]}", "",
             "Rules:", *rules, "", f"Question: {question}", f"Original answer: {answer}"]
    if error_type == "non_sequitur":
        lines.append(f"Excerpt from the same paper: {context}")
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse_generated(reply: dict, seed_answer: str, where: str) -> str:
    """The validated corrupted answer of an llm.chat reply (see the module docstring)."""
    content = reply["content"]

    def fail(reason):
        raise GenerationError(f"{where}: {reason}; reply={content[:300]!r}. The reply is cached; "
                              "to re-draw this seed deliberately set RESAMPLE[(lang, k)] = 1.")

    if reply["finish_reason"] != "stop":
        fail(f"finish_reason={reply['finish_reason']!r}")
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        fail("reply is not JSON")
    if not isinstance(obj, dict) or "answer" not in obj:
        fail('JSON has no "answer"')
    answer = obj["answer"]
    if not isinstance(answer, str) or not answer.strip():
        fail("empty answer")
    answer = re.sub(r"\s*\n\s*", " ", answer.strip())
    if _same(answer, seed_answer):
        fail("answer identical to the seed")
    return answer


# --- generation ----------------------------------------------------------------------

def corrupt(lang: str, seed: dict) -> dict:
    """The corrupted row of one seed (a dict from load_seeds)."""
    k = seed["k"]
    rng = random.Random(k)
    error_type = assign_type(k, seed["answer"])
    where = f"{lang} seed k={k} ({seed_path(lang).name}:{seed['line']}, {error_type})"
    if error_type in PROGRAMMATIC:
        answer = PROGRAMMATIC[error_type](seed["answer"], rng)
        if not answer.strip() or _same(answer, seed["answer"]):
            raise RuntimeError(f"{where}: programmatic corruption left the answer unchanged")
    else:
        variant = rng.choice(NON_SEQUITUR_VARIANTS) if error_type == "non_sequitur" else None
        messages = generation_messages(lang, error_type, seed["question"], seed["answer"],
                                       seed["context"], variant)
        reply = llm.chat(DEPLOYMENT, messages, max_completion_tokens=config.JUDGE_MAX_COMPLETION_TOKENS,
                         response_format={"type": "json_object"}, repeat=RESAMPLE.get((lang, k), 0))
        answer = parse_generated(reply, seed["answer"], where)
    return {"question": seed["question"], "answer": answer, "error_type": error_type, "veredict": "y"}


def build_rows(lang: str, workers: int = 12) -> tuple:
    """(rows shuffled with seed 42, seeds). The parallel pass fills the cache. The rows are
    then rebuilt from the cache sequentially, so this run and any rerun write the same
    files even if two identical requests raced in the first pass."""
    seeds = load_seeds(lang)
    llm.run_parallel(lambda s: corrupt(lang, s), seeds, workers=workers, desc=f"fluency_synth {lang}")
    rows = []
    for seed in seeds:
        rows.append({"question": seed["question"], "answer": seed["answer"],
                     "error_type": "none", "veredict": "n"})
        rows.append(corrupt(lang, seed))
    random.Random(SHUFFLE_SEED).shuffle(rows)
    return rows, seeds


def ready_record(lang: str, row: dict) -> dict:
    """Azure chat fine-tuning record: prompts.ft_fluency + the assistant verdict."""
    return {"messages": prompts.ft_fluency(lang, row["question"], row["answer"])
            + [{"role": "assistant", "content": row["veredict"]}]}


def _write_jsonl(path: Path, records) -> None:
    """Write the whole file to a temp file, then rename it, so a crash never leaves a
    half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def generate(lang: str, workers: int = 12) -> list:
    """Build, write and summarise one language's fluency FT data. Returns the rows."""
    rows, seeds = build_rows(lang, workers)
    _write_jsonl(raw_path(lang), rows)
    _write_jsonl(ready_path(lang), (ready_record(lang, r) for r in rows))
    verdicts = Counter(r["veredict"] for r in rows)
    types = Counter(r["error_type"] for r in rows)
    glued = Counter(r["veredict"] for r in rows if GLUED_BOUNDARY.search(r["answer"]))
    moved = Counter((ERROR_TYPES[s["k"] % len(ERROR_TYPES)], assign_type(s["k"], s["answer"]))
                    for s in seeds)
    moved = {f"{a}->{b}": n for (a, b), n in moved.items() if a != b}
    print(f"[fluency_synth] {lang}: {len(rows)} rows from {len(seeds)} seeds | "
          + " | ".join(f"{v} {verdicts[v]} ({verdicts[v] / len(rows):.1%})" for v in ("n", "y")))
    print(f"[fluency_synth] {lang} per type: "
          + ", ".join(f"{t} {types[t]}" for t in ("none",) + ERROR_TYPES))
    print(f"[fluency_synth] {lang}: one-token answers moved to repetition: {moved or 'none'}")
    print(f"[fluency_synth] {lang}: answers with a glued boundary: n {glued['n']}, y {glued['y']}")
    print(f"[fluency_synth] wrote {raw_path(lang)} and {ready_path(lang).name}")
    return rows


# --- sanity check ----------------------------------------------------------------------

def check(lang: str, rows: list, workers: int = 12) -> dict:
    """Score `rows` (raw-file rows) with gpt-5-mini and the original fluency judge
    prompt, as judge.py does for fmt=judge, temp=default, repeat 0. The scores are
    rebuilt from the cache after the parallel pass, so a rerun reproduces them.
    Returns {"items", "accuracy", "confusion", "by_type"}. The confusion matrix is
    keyed (gold, pred); `by_type` maps each type to (voted y, n)."""
    if not rows:
        raise ValueError("check needs at least one row")

    def score(row):
        if not row["answer"].strip():
            raise ValueError(f"empty answer in the fluency data: {row!r}")
        reply = llm.chat(DEPLOYMENT, prompts.fluency_judge(lang, row["question"], row["answer"]),
                         max_completion_tokens=config.JUDGE_MAX_COMPLETION_TOKENS)
        vote, parse_ok = prompts.parse_vote(reply["content"])
        return {**row, "vote": vote, "parse_ok": parse_ok, "raw": reply["content"]}

    llm.run_parallel(score, rows, workers=workers, desc=f"fluency_synth check {lang}")
    items = [score(r) for r in rows]
    confusion = Counter((it["veredict"], it["vote"]) for it in items)
    by_type = {}
    for it in items:
        y, n = by_type.get(it["error_type"], (0, 0))
        by_type[it["error_type"]] = (y + (it["vote"] == "y"), n + 1)
    accuracy = sum(it["vote"] == it["veredict"] for it in items) / len(items)
    return {"items": items, "accuracy": accuracy, "confusion": confusion, "by_type": by_type}


def load_raw(lang: str) -> list:
    """The generated raw file of `lang`. Raises if it is missing or has a row this module
    does not produce (an unknown error_type, or a veredict that does not match it), e.g.
    a file generated before missing_separators was dropped."""
    path = raw_path(lang)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run the generation first")
    rows = _read_jsonl(path, ("question", "answer", "error_type", "veredict"))
    for i, r in enumerate(rows, start=1):
        if r["error_type"] not in ("none",) + ERROR_TYPES \
                or r["veredict"] != ("n" if r["error_type"] == "none" else "y"):
            raise ValueError(f"{path}:{i}: error_type {r['error_type']!r} with veredict "
                             f"{r['veredict']!r} is not produced by this module; regenerate the file")
    return rows


def check_sample(rows: list, n: int) -> list:
    """0-based indices of the n rows the default check scores, in file order. Stratified
    by type so that every type is scored: n // 2 fluent rows, and the rest spread evenly
    over the error types present (ERROR_TYPES order; the first types take the remainder).
    The strata are sampled in that order with one random.Random(CHECK_SEED). A stratum
    smaller than its share raises."""
    if n < 1:
        raise ValueError(f"--n must be >= 1, got {n}")
    present = [t for t in ERROR_TYPES if any(r["error_type"] == t for r in rows)]
    n_bad = n - n // 2
    if not present:
        raise ValueError("the file has no corrupted rows to sample")
    shares = [("none", n // 2)] + [(t, n_bad // len(present) + (i < n_bad % len(present)))
                                    for i, t in enumerate(present)]
    rng, picked = random.Random(CHECK_SEED), []
    for t, m in shares:
        pool = [i for i, r in enumerate(rows) if r["error_type"] == t]
        if m > len(pool):
            raise ValueError(f"--n {n} needs {m} rows of type {t!r}, the file has {len(pool)}")
        picked += rng.sample(pool, m)
    return sorted(picked)


def run_check(lang: str, n: int = None, workers: int = 12, types: tuple = None) -> dict:
    """Score the stratified sample of n rows (default CHECK_N) of the generated raw file,
    or every row of `types`, print the results and write them to check_path."""
    if n is not None and types is not None:
        raise ValueError("n and types are mutually exclusive")
    rows = load_raw(lang)
    if types is None:
        lines = check_sample(rows, CHECK_N if n is None else n)
    else:
        unknown = set(types) - set(("none",) + ERROR_TYPES)
        if unknown:
            raise ValueError(f"unknown types {sorted(unknown)}; choose from {('none',) + ERROR_TYPES}")
        types = tuple(t for t in ("none",) + ERROR_TYPES if t in types)
        lines = [i for i, r in enumerate(rows) if r["error_type"] in types]
        if not lines:
            raise ValueError(f"{raw_path(lang).name} has no rows of types {types}")
    result = check(lang, [rows[i] for i in lines], workers)
    out = check_path(lang, types)
    _write_jsonl(out, ({"line": i + 1, **it} for i, it in zip(lines, result["items"])))
    c, items, n = result["confusion"], result["items"], len(lines)
    what = "stratified sample" if types is None else "all rows of " + "+".join(types)
    print(f"[fluency_synth check] {lang} ({what}): n={n} accuracy={result['accuracy']:.4f} "
          f"({sum(it['vote'] == it['veredict'] for it in items)}/{n}), "
          f"parse_ok {sum(it['parse_ok'] for it in items)}/{n}")
    print("          pred y  pred n")
    for gold in ("y", "n"):
        print(f"  gold {gold}  {c[(gold, 'y')]:6d}  {c[(gold, 'n')]:6d}")
    print("  voted y by type: " + ", ".join(f"{t} {y}/{m}" for t, (y, m) in sorted(
        result["by_type"].items(), key=lambda kv: (("none",) + ERROR_TYPES).index(kv[0]))))
    print(f"[fluency_synth check] wrote {out}")
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m chomps.fluency_synth",
                                description="Build the synthetic fluency fine-tuning data, or "
                                            "(--check) score some of it with gpt-5-mini.")
    p.add_argument("--lang", nargs="+", choices=config.LANGS, default=list(config.LANGS))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--check", action="store_true",
                   help="sanity check of the already generated files instead of generating")
    p.add_argument("--n", type=int, default=None,
                   help=f"rows in the stratified --check sample (default {CHECK_N})")
    p.add_argument("--types", nargs="+", choices=("none",) + ERROR_TYPES, default=None,
                   help="with --check: score every row of these types instead of a sample")
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if (args.n is not None or args.types is not None) and not args.check:
        p.error("--n and --types only apply to --check")
    if args.n is not None and args.types is not None:
        p.error("--n and --types are mutually exclusive")
    for lang in args.lang:
        if args.check:
            run_check(lang, args.n, args.workers, None if args.types is None else tuple(args.types))
        else:
            generate(lang, args.workers)


if __name__ == "__main__":
    main()
