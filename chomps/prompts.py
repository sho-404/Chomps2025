"""The exact prompts of the competition pipeline, plus the two variants the
revision needs (no-context judging and the fine-tuned judge format).

The judge and retrieval texts are NOT transcribed here: they are extracted
from the original scripts by chomps/extract_prompts.py into
chomps/_prompt_templates.py (trailing spaces and indented blank lines
included) and filled in with str.replace, so every request the revision sends
is byte-identical to what the submission sent. tests/test_prompts.py verifies this.
"""
from . import config
from . import _prompt_templates as T

JUDGE_SYSTEM = T.JUDGE_SYSTEM            # same text for factuality and fluency in both languages
RETRIEVAL_SYSTEM = T.RETRIEVAL_SYSTEM
assert T.FLUENCY_SYSTEM == T.JUDGE_SYSTEM

FT_SYSTEM = {
    "factuality": {
        "english": "You are an evaluator reviewing an answer given by an LLM regarding a scientific paper question. Respond with 'y' if the answer has mistakes, or 'n' if there are no mistakes.",
        "spanish": "Eres un evaluador revisando una respuesta dada por un LLM sobre una pregunta de un artículo científico. Responde con 'y' si la respuesta tiene errores, o 'n' si no hay errores.",
    },
    "fluency": {
        "english": "You are an evaluator reviewing the fluency of an answer given by an LLM regarding a scientific paper question. Respond with 'y' if the answer has fluency mistakes (broken grammar, incoherence, repetition, truncation, or an answer that does not respond to the question), or 'n' if it is fluent.",
        "spanish": "Eres un evaluador revisando la fluidez de una respuesta dada por un LLM sobre una pregunta de un artículo científico. Responde con 'y' si la respuesta tiene errores de fluidez (gramática rota, incoherencia, repetición, texto cortado o una respuesta que no responde a la pregunta), o 'n' si es fluida.",
    },
}

NO_CONTEXT_NOTE = {
    "english": "(No context is provided for this item. Judge the Model Answer against your own knowledge of the paper named in the User Prompt.)",
    "spanish": "(No se proporciona contexto para este ejemplo. Juzga la Respuesta del Modelo con tu propio conocimiento del artículo nombrado en el Prompt del Usuario.)",
}


def _fill(template: str, **values) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def retrieval(question: str, paper_text: str) -> list:
    """Original extraction prompt; the paper text is cut at MAX_PAPER_CHARS as in the script."""
    user = _fill(T.RETRIEVAL_USER, question=question, paper_text=paper_text[:config.MAX_PAPER_CHARS])
    return [{"role": "system", "content": RETRIEVAL_SYSTEM}, {"role": "user", "content": user}]


def factuality_judge(lang: str, context, user_prompt: str, model_answer: str,
                     max_chars: int = config.MAX_CONTEXT_CHARS) -> list:
    """Original single-judge / council factuality prompt. `context=None`
    produces the no-retrieval variant (same wording, context block replaced
    by NO_CONTEXT_NOTE). `max_chars` is the original 60k-character cap on the
    context block; the full-document condition raises it to MAX_PAPER_CHARS."""
    if lang not in T.FACTUALITY_USER:
        raise ValueError(lang)
    ctx = NO_CONTEXT_NOTE[lang] if context is None else context[:max_chars]
    user = _fill(T.FACTUALITY_USER[lang], ctx=ctx, user_prompt=user_prompt, model_answer=model_answer)
    return [{"role": "system", "content": JUDGE_SYSTEM[lang]}, {"role": "user", "content": user}]


def fluency_judge(lang: str, user_prompt: str, model_answer: str) -> list:
    if lang not in T.FLUENCY_USER:
        raise ValueError(lang)
    user = _fill(T.FLUENCY_USER[lang], user_prompt=user_prompt, model_answer=model_answer)
    return [{"role": "system", "content": JUDGE_SYSTEM[lang]}, {"role": "user", "content": user}]


def ft_factuality(lang: str, context: str, question: str, answer: str) -> list:
    """The format the factuality judges were fine-tuned on (generate_ready_ft_dataset.py)."""
    user = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    return [{"role": "system", "content": FT_SYSTEM["factuality"][lang]}, {"role": "user", "content": user}]


def ft_fluency(lang: str, question: str, answer: str) -> list:
    """Format used both to build the fluency fine-tuning data and to query the fluency judges."""
    user = f"Question: {question}\n\nAnswer: {answer}"
    return [{"role": "system", "content": FT_SYSTEM["fluency"][lang]}, {"role": "user", "content": user}]


def parse_vote(content: str) -> tuple:
    """Original parsing rule: first look for 'y', then 'n', else fall back to
    'y' (treated as a mistake). Returns (vote, parse_ok) where parse_ok is
    True only when the reply is exactly one of the two characters."""
    c = (content or "").strip().lower()
    if "y" in c:
        return "y", c == "y"
    if "n" in c:
        return "n", c == "n"
    return "y", False
