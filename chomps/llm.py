"""Azure OpenAI access with an on-disk cache, retries and usage/latency capture.

Every call is keyed by its full argument set plus a `repeat` index, so
repeated runs (for variance estimates) are distinct cache entries while
re-executing a script after a crash costs nothing. Nothing is swallowed: a
non-retryable API error propagates to the caller.
"""
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

import openai
from openai import AzureOpenAI

from . import config

_client: Optional[AzureOpenAI] = None
_client_lock = threading.Lock()
_db_lock = threading.Lock()
_DB_PATH = config.CACHE / "llm_cache.sqlite"
_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
              openai.InternalServerError)


def client() -> AzureOpenAI:
    global _client
    with _client_lock:
        if _client is None:
            if not config.AZURE_ENDPOINT or not config.AZURE_API_KEY:
                raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY missing from .env")
            _client = AzureOpenAI(api_version=config.API_VERSION,
                                  azure_endpoint=config.AZURE_ENDPOINT,
                                  api_key=config.AZURE_API_KEY,
                                  max_retries=0, timeout=180.0)
        return _client


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")  # several runner processes share the cache
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, kind TEXT, value TEXT, created REAL)")
    return conn


_conn = _db()


def _cache_get(key: str):
    with _db_lock:
        row = _conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _cache_put(key: str, kind: str, value: dict):
    with _db_lock:
        _conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                      (key, kind, json.dumps(value, ensure_ascii=False), time.time()))
        _conn.commit()


def _key(kind: str, payload: dict) -> str:
    return hashlib.sha256((kind + json.dumps(payload, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def _with_retries(fn: Callable, what: str):
    delay = 2.0
    for attempt in range(16):   # up to ~14 min of sustained throttling before giving up
        try:
            return fn()
        except _RETRYABLE as e:
            if attempt == 15:
                raise
            print(f"[llm] {what}: {type(e).__name__}, retry {attempt + 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def chat(deployment: str, messages: list, *, temperature: Optional[float] = None,
         max_completion_tokens: int = config.JUDGE_MAX_COMPLETION_TOKENS,
         seed: Optional[int] = None, repeat: int = 0, response_format: Optional[dict] = None,
         reasoning_effort: Optional[str] = None, cache_tag: Optional[str] = None) -> dict:
    """Return {content, prompt_tokens, completion_tokens, reasoning_tokens,
    latency_s, model, cached, finish_reason}. `temperature=None` sends no
    temperature (the model default). Reasoning deployments refuse an explicit
    temperature, so passing one for them is an error here rather than a 400.
    `cache_tag` is never sent to the API; it is part of the cache key so that
    two dataset items with identical inputs get independent samples (as they
    did in the original per-example scripts) instead of one shared reply."""
    if temperature is not None and deployment in config.REASONING_DEPLOYMENTS:
        raise ValueError(f"{deployment} does not accept a temperature parameter")
    payload = {"deployment": deployment, "messages": messages, "temperature": temperature,
               "max_completion_tokens": max_completion_tokens, "seed": seed, "repeat": repeat,
               "response_format": response_format, "reasoning_effort": reasoning_effort,
               "cache_tag": cache_tag}
    key = _key("chat", payload)
    hit = _cache_get(key)
    if hit is not None:
        hit["cached"] = True
        return hit
    kwargs = dict(model=deployment, messages=messages, max_completion_tokens=max_completion_tokens)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    if response_format is not None:
        kwargs["response_format"] = response_format
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    t0 = time.perf_counter()
    resp = _with_retries(lambda: client().chat.completions.create(**kwargs), deployment)
    latency = time.perf_counter() - t0
    choice = resp.choices[0]
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    out = {
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        "latency_s": round(latency, 3),
        "model": resp.model,
        "cached": False,
    }
    _cache_put(key, "chat", out)
    return out


def embed(texts: list, deployment: str = config.DEPLOYMENTS["embedding"]) -> list:
    """Embed a list of strings (cached per string). Returns list of float lists."""
    keys = [_key("embed", {"deployment": deployment, "text": t}) for t in texts]
    out = [None] * len(texts)
    todo = []
    for i, k in enumerate(keys):
        hit = _cache_get(k)
        if hit is not None:
            out[i] = hit["embedding"]
        else:
            todo.append(i)
    for start in range(0, len(todo), 64):
        batch = todo[start:start + 64]
        resp = _with_retries(lambda: client().embeddings.create(model=deployment, input=[texts[i] for i in batch]),
                             deployment)
        for j, item in zip(batch, resp.data):
            out[j] = item.embedding
            _cache_put(keys[j], "embed", {"embedding": item.embedding, "prompt_tokens": resp.usage.prompt_tokens / len(batch)})
    return out


def run_parallel(fn: Callable, items: Iterable, workers: int = 12, desc: str = "") -> list:
    """Apply fn to every item concurrently, preserving order; prints progress.
    An exception in any item propagates (the whole run aborts)."""
    items = list(items)
    results = [None] * len(items)
    done = 0
    lock = threading.Lock()

    def wrap(i):
        nonlocal done
        results[i] = fn(items[i])
        with lock:
            done += 1
            if done % 25 == 0 or done == len(items):
                print(f"[{desc}] {done}/{len(items)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(wrap, range(len(items))))
    return results
