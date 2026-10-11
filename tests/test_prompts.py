"""prompts.py must reproduce the competition scripts' requests byte for byte."""
import ast

import pytest

from chomps import config, prompts, extract_prompts
from chomps import _prompt_templates as T


def test_generated_templates_are_current():
    """_prompt_templates.py matches a fresh extraction from the original scripts."""
    fresh = extract_prompts.extract()
    for k, v in fresh.items():
        assert getattr(T, k) == v, k


def _render_original(path, funcname, var, env):
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == funcname][0]
    node = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == var][0]
    expr = ast.Expression(body=node)
    ast.fix_missing_locations(expr)
    return eval(compile(expr, "<orig>", "eval"), {"__builtins__": {}}, env)


ENV = {"context": "C" * 70_000, "user_prompt": "In the article X, what?", "model_answer": "An answer.",
       "question": "What?", "paper_text": "P" * 130_000}


@pytest.mark.parametrize("lang,script,func", [
    ("english", "english/scripts/factuality/approach_1_single_agent.py", "evaluate_factuality"),
    ("spanish", "spanish/scripts/factuality/approach_1_single_agent.py", "evaluate_factuality"),
])
def test_factuality_judge_equals_original(lang, script, func):
    """Includes the original context[:60000] cut."""
    orig = _render_original(config.CODE_ROOT / script, func, "eval_prompt", ENV)
    ours = prompts.factuality_judge(lang, ENV["context"], ENV["user_prompt"], ENV["model_answer"])[1]["content"]
    assert ours == orig


@pytest.mark.parametrize("lang,script,func", [
    ("english", "english/scripts/fluency/approach_1_single_agent.py", "evaluate_fluency"),
    ("spanish", "spanish/scripts/fluency/approach_1_single_agent.py", "evaluate_fluency_es"),
])
def test_fluency_judge_equals_original(lang, script, func):
    orig = _render_original(config.CODE_ROOT / script, func, "eval_prompt", ENV)
    ours = prompts.fluency_judge(lang, ENV["user_prompt"], ENV["model_answer"])[1]["content"]
    assert ours == orig


@pytest.mark.parametrize("lang", ["english", "spanish"])
def test_retrieval_prompt_equals_original(lang):
    """Includes the paper_text[:120000] cut; both languages' scripts are identical."""
    orig = _render_original(config.CODE_ROOT / lang / "scripts/inject_context_single_block.py",
                            "get_context_from_llm", "prompt", ENV)
    ours = prompts.retrieval(ENV["question"], ENV["paper_text"])[1]["content"]
    assert ours == orig


def test_full_condition_raises_context_cap():
    long_ctx = "\u2603" * (config.MAX_PAPER_CHARS + 5)
    capped = prompts.factuality_judge("english", long_ctx, "p", "a")[1]["content"]
    full = prompts.factuality_judge("english", long_ctx, "p", "a", max_chars=config.MAX_PAPER_CHARS)[1]["content"]
    assert capped.count("\u2603") == config.MAX_CONTEXT_CHARS
    assert full.count("\u2603") == config.MAX_PAPER_CHARS


def test_no_context_variant_and_parse():
    msgs = prompts.factuality_judge("spanish", None, "p", "a")
    assert prompts.NO_CONTEXT_NOTE["spanish"] in msgs[1]["content"]
    assert prompts.parse_vote(" Y ") == ("y", True)
    assert prompts.parse_vote("n.") == ("n", False)
    assert prompts.parse_vote("") == ("y", False)
