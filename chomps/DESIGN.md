# Revision experiments — design contract

Package `chomps/` (run from `code/` with `../.venv/bin/python -m chomps.<module>`).
Repo root is `code/`; sibling dirs `../datasets`, `../results`, `../cache`, `../.env`.
Existing foundation (do not modify without saying so): `config.py`, `llm.py`, `data.py`, `prompts.py`.

Rules for every module: no silent fallbacks (raise on bad input), no sleeps for
timing, every run is idempotent/resumable (the LLM cache in `llm.py` makes
re-execution free; store files are rewritten in full from the cache), outputs are
JSONL/CSV under `../results/`, CLI via `argparse` with `--lang`, `--split`, etc.
Progress printing via `llm.run_parallel`. Tests under `code/tests/` (pytest), runnable
offline with monkeypatched `llm.chat`/`llm.embed`; real API use is allowed only on ≤3
examples and only in a manual smoke test, never in pytest.

## Data model

* `data.load_examples(lang, split)` → list of dicts: `index, title, question, prompt,
  output_text, output_tokens, output_logits, context_original, labels{factuality,
  fluency}` (1 = hallucination present), plus other raw fields.
* `data.paper_text(lang, title)` → full extracted text (cached). Raises if missing.
* Vote values are strings `"y"` (hallucination) / `"n"`; labels are ints 1/0. The
  positive class for precision/recall/F1 is `y`/1. The official task metric is
  **macro-F1** (unweighted mean of the F1 of class y and class n).

## Retrieval conditions (factuality track only) — `retrieval.py`

Store: `../results/contexts/{lang}/{split}/{condition}.jsonl`, one row per example:
`{"index", "context", "meta": {...}}`. Rows ordered as `load_examples`.

| condition | context | meta |
|---|---|---|
| `pseudo_rag` | `context_original` (the competition span) | `{"source":"original"}` |
| `pseudo_rag_rerun` | new extraction: `llm.chat("gpt-5-mini", prompts.retrieval(question, paper_text), max_completion_tokens=config.RETRIEVAL_MAX_COMPLETION_TOKENS)` | tokens, latency, model |
| `bm25_k{K}` (K∈{1,3,5}) | top-K chunks by BM25Okapi (rank_bm25), concatenated in document order with `"\n[...]\n"` between | `{"chunk_ids":[...], "scores":[...], "n_chunks": N}` |
| `dense_k{K}` | top-K chunks by cosine similarity of `text-embedding-3-large` (via `llm.embed`) | same as bm25 |
| `full` | `paper_text[:config.MAX_PAPER_CHARS]` | `{"chars": n, "truncated": bool}` |
| `none` | `null` (JSON null) | `{}` |

Chunking (shared by bm25/dense): over `paper_text[:MAX_PAPER_CHARS]`, whitespace
tokens, windows of 100 words with stride 50; chunk text = the words joined by single
spaces. Query = `question` (not `prompt`). BM25 tokenizer: `re.findall(r"\w+", s.lower())`.
Embeddings: embed every chunk of every paper once (cache keyed by text); query embedding
likewise. `pseudo_rag_rerun`: the exact original prompt (`prompts.retrieval`). The original
protocol truncated empty-context items? No: empty context only occurred when a PDF was
missing; every paper is now available, so a rerun must produce a non-empty string for all.

CLI: `python -m chomps.retrieval --lang english --split test --conditions pseudo_rag bm25_k3 dense_k3 full none pseudo_rag_rerun` (defaults: all langs, all splits, all conditions). Writes the store; prints a summary (empty/not-found counts, mean chars).

## Judge runs — `judge.py`

Store: `../results/votes/{lang}/{track}/{split}/{condition}__{judge}__{fmt}__t{temp}.jsonl`,
one row per (example, repeat): `{"index", "repeat", "vote": "y"|"n", "parse_ok": bool,
"raw": str, "prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s",
"cached": bool, "model": str}`. `temp` ∈ `default` (no temperature sent), `0`, `1.0`.
`fmt` ∈ `judge` (the original evaluator prompt: `prompts.factuality_judge` /
`prompts.fluency_judge`) or `ft` (the fine-tuning format: `prompts.ft_factuality` /
`prompts.ft_fluency`, with `question=prompt`).

Judges: `gpt-5-mini` (base; must use temp `default`, never send a temperature —
`llm.chat` raises otherwise), `gpt-4.1-mini`, `gpt-4.1-nano` (base), and the fine-tuned
deployments from `config.FT_DEPLOYMENTS` (names like `ft-fact-en-mini`). A judge name is
the deployment name.

Protocol details (reproduce the original scripts):
* factuality, any condition except `none`: if the context string is empty → no API call,
  `vote="y"`, `raw="AUTO_EMPTY_CONTEXT"`, `parse_ok=True`, tokens 0, latency 0 (the
  original `if not context: return "y"` rule).
* condition `full`: call `prompts.factuality_judge(..., max_chars=config.MAX_PAPER_CHARS)`.
* condition `none`: `context=None`.
* fluency track: condition is always `none`; prompt = `prompts.fluency_judge(lang, prompt, output_text)` for `fmt=judge`, `prompts.ft_fluency(lang, prompt, output_text)` for `fmt=ft`. The original fluency script also had `if not model_answer.strip(): return "y"` — reproduce it with `raw="AUTO_EMPTY_ANSWER"`.
* `max_completion_tokens=config.JUDGE_MAX_COMPLETION_TOKENS`; `llm.chat(..., repeat=r)` for r in range(repeats); parse with `prompts.parse_vote`.
* Rows are written in full after the run (sorted by example order then repeat). Rerunning with the same arguments is free (cache).

CLI: `python -m chomps.judge --lang english --track factuality --split test --condition pseudo_rag --judge gpt-5-mini --fmt judge --temp default --repeats 5 --workers 12`. Multiple values allowed for `--split`, `--condition`, `--judge`; `--temp` is one of default/0/1.0.

## Metrics — `metrics.py`

Pure functions over aligned lists of gold ints and predicted votes:
* `confusion(gold, pred)` → dict tp/fp/tn/fn (positive = y/1).
* `scores(gold, pred)` → dict: `accuracy, balanced_accuracy, precision_y, recall_y, f1_y, precision_n, recall_n, f1_n, macro_f1, mcc, n`.
* `bootstrap_ci(gold, pred, metric="macro_f1", n_boot=2000, seed=0)` → (lo, hi) percentile 95 % CI over example resampling.
* `paired_bootstrap_diff(gold, pred_a, pred_b, metric, n_boot=2000, seed=0)` → (mean_diff, lo, hi).
* `mcnemar(gold, pred_a, pred_b)` → (b, c, exact p-value) using `statsmodels.stats.contingency_tables.mcnemar(exact=True)`.
* Helpers to load a votes file into `{repeat: {index: vote}}`, to compute a **council majority** from three judges' votes at the same repeat index, and to compute a **consensus over repeats** (majority of a judge's repeats, ties → "y").
* `summarize_system(gold_map, preds_by_repeat)` → per-repeat scores + mean/std across repeats + bootstrap CI on repeat 0 and on the consensus prediction.

## Meta-learner — `meta.py`

Features per example = the three council judges' votes (ints) at a given repeat (default 0),
optionally `+ logits features` (see `logits.py`). Model: `LogisticRegression(penalty="l2",
C=1.0, class_weight=None|"balanced", solver="lbfgs", random_state=42)`.
Protocols:
* `original`: fit on train, predict validation and test (as in the paper).
* `cv`: stratified 5-fold on train+validation pooled, report fold macro-F1 mean±std, then
  fit on the pooled set and predict test.
Report coefficients β₀..β₃ (with bootstrap 95 % CI over refits), and write predictions
to the votes store as judge name `meta-{protocol}-{cw}` (e.g. `meta-original-none`), `fmt=judge`,
`temp=default`, repeat 0, so `metrics.py` treats them like any system.
CLI: `python -m chomps.meta --lang english --track factuality --condition pseudo_rag --judges gpt-5-mini ft-fact-en-mini ft-fact-en-nano --fmt judge --temp default --class-weight balanced --protocol cv`.

## Logits — `logits.py`

From `output_logits` (list of floats, one per generated token; higher = more confident):
features `mean, std, min, p10, p25, frac_below_15, n_tokens, mean_first_10`. For each
track and language: 5-fold CV AUC (train+validation pooled) of a logistic regression on
these features alone, and point-biserial correlation of each feature with the label.
Write `../results/tables/logits_{lang}.csv`. Expose `features(example) -> list[float]` for `meta.py`.

## Retrieval quality — `retrieval_eval.py`

For each context condition (pseudo_rag, pseudo_rag_rerun, bm25_k3, dense_k3) and split:
* `verbatim`: `rapidfuzz.fuzz.partial_ratio(norm(context), norm(paper_text))` (0–100) with whitespace-normalised strings; `exact_substring` bool.
* `not_found` (context contains "not found" case-insensitive) and `empty` flags.
* `chars`, `tokens` (tiktoken `o200k_base`).
* Agreement pseudo_rag vs pseudo_rag_rerun: token-set Jaccard and partial_ratio.
Write `../results/retrieval_eval/stats_{lang}_{split}_{condition}.csv` and a summary table.
Also export grading sheets for human/LLM relevance grading:
`../results/retrieval_eval/grading_sheet_{lang}_{split}.jsonl` rows
`{"index", "question", "prompt", "output_text", "gold_factuality", "contexts": {condition: context}}`
for the test split and conditions pseudo_rag, bm25_k3, dense_k3.

## Synthetic fine-tuning data QC — `synth_qc.py`

Input `../code/{english,spanish}/finetuning/finetune_dataset_raw.jsonl` (`question, context,
answer, veredict`). Report per language: n, class balance, duplicate/near-duplicate
questions (exact and rapidfuzz ratio ≥ 90), answer/context length stats, the set of arXiv
ids referenced in questions (regex `\d{4}\.\d{4,5}(v\d+)?`), overlap of those ids/titles
with the task papers (expect none), and an LLM audit: for every example ask `gpt-5-mini`
(a) to classify the hallucination type of `veredict=y` answers into
`{wrong_number, wrong_entity, negation_or_contradiction, unsupported_claim, wrong_method_or_result, other}`
and (b) to verify independently whether the answer is supported by the context
(`supported`/`contradicted`/`not_addressed`) using JSON output (`response_format={"type":"json_object"}`).
Write `../results/synth_qc/{lang}_audit.jsonl` and `../results/synth_qc/summary.csv`.

## Fluency fine-tuning data — `fluency_synth.py`

Seeds: the `veredict=="n"` rows of each language's `finetune_dataset_raw.jsonl` (clean
question/answer pairs). For each seed produce ONE corrupted variant; cycle error types
deterministically by row index so the set is balanced across types:
1. `repetition` (programmatic): repeat the last sentence/phrase 2–4 times without separators.
2. `truncation` (programmatic): cut the answer mid-sentence (keep 40–70 % of characters) or drop the first 1–3 words so it starts mid-sentence.
3. `missing_separators` (programmatic): remove newlines/spaces after list markers and sentence ends ("1. A2. B").
4. `grammar` (LLM): inject agreement/tense/preposition errors and typos while keeping meaning.
5. `non_sequitur` (LLM): answer a different question from the same paper, or append new questions instead of answering.
6. `code_switching` (LLM; Spanish: anglicisms and untranslated English fragments; English: word-salad substitutions).
7. `gibberish` (LLM): partially incoherent word salad while keeping some topical words.
LLM corruptions use `gpt-5-mini` with a JSON-object response giving `{"answer": ...}`.
Output `../code/{lang}/finetuning/finetune_fluency_raw.jsonl` rows
`{"question", "answer", "error_type", "veredict"}` (clean seeds with `error_type="none"`,
`veredict="n"`; corrupted with `veredict="y"`), and `finetune_fluency_ready.jsonl` in Azure
chat format built with `prompts.ft_fluency(lang, question, answer)` + assistant `veredict`.
Shuffle with seed 42. Print class balance and per-type counts. Then a sanity check: run
`gpt-5-mini` with `prompts.fluency_judge` on 60 random items and print accuracy.

## Cost and latency — `cost.py`

Aggregate the votes and contexts stores: per (lang, track, condition, judge, fmt, temp):
n calls, mean/median prompt tokens, completion tokens, reasoning tokens, mean/median
latency (only `cached==False` rows carry real latency; use all rows for tokens), USD per
1000 examples from `config.PRICES_USD_PER_M` via `config.price_key`. Retrieval-stage cost
per example for `pseudo_rag_rerun` (tokens from meta) and `dense` (embedding tokens ≈
`len(text)/4` estimate if not recorded; record real `prompt_tokens` from `llm.embed` when
available). Write `../results/tables/cost.csv`.

## Reporting — `report.py`

Build the paper tables from the stores (CSV + LaTeX fragments in `../results/tables/`):
`factuality_main` (lang × split × system: accuracy, balanced acc, P/R/F1 of y, macro-F1
mean ± std over repeats, 95 % CI), `fluency_main`, `retrieval_ablation` (single judge
gpt-5-mini × condition), `per_judge`, `meta_cv`, `class_distribution` (per lang/split/track
counts and %), `confusion_matrices`, `significance` (McNemar + paired bootstrap between
chosen pairs). Numbers formatted to 4 decimals as in the paper.
