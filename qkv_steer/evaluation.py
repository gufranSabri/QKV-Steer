from __future__ import annotations

import os
import re

from qkv_steer._vendor.bleurt import score_bleurt_batch

JUDGE_PROMPT = (
    "You are evaluating whether a model's answer to a question is factually correct, "
    "given one or more reference answers.\n\n"
    "Question: {prompt}\n"
    "Reference answer(s): {gold}\n"
    "Model's answer: {answer}\n\n"
    "Is the model's answer consistent with the reference answer(s)? "
    "Reply with exactly one word: Yes or No."
)


def bleurt(candidates: list[str], golds: list, checkpoint: str, threshold: float) -> list[tuple[float, int]]:
    return score_bleurt_batch(candidates, golds, checkpoint=checkpoint, threshold=threshold)


def llm_judge(
    prompts: list[str], candidates: list[str], golds: list, model: str = "claude-sonnet-5"
) -> list[int]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    labels = []
    for prompt, answer, gold in zip(prompts, candidates, golds):
        text = JUDGE_PROMPT.format(prompt=prompt, gold=gold, answer=answer)
        reply = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": text}],
        )
        verdict = reply.content[0].text.strip().lower()
        hallucinated = 0 if re.match(r"^yes", verdict) else 1
        labels.append(hallucinated)
    return labels
