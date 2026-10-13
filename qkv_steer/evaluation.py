from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from qkv_steer._vendor.bleurt import score_bleurt_batch

LEV_CHECKPOINT = "interfaze-ai/lev"

NOVITA_BASE_URL = "https://api.novita.ai/openai"
CLOUD_JUDGE_MODEL = "qwen/qwen3.8-27b"
CLOUD_JUDGE_SYSTEM_PROMPT = (
    "You are a strict hallucination judge. You will be given a model's "
    "generated response and a list of gold (reference) answers. Decide "
    "whether the generated response hallucinates, i.e. it does NOT match "
    "/ is NOT supported by / contradicts or omits the information in the "
    "gold answers. If the generated response agrees with (or is a "
    "reasonable paraphrase / subset / superset that is consistent with) "
    "any one of the gold answers, it is NOT a hallucination.\n\n"
    "Respond with exactly one character and nothing else:\n"
    "1 = hallucination (response does not match the gold answers)\n"
    "0 = not a hallucination (response matches/is supported by a gold answer)\n"
    "Do not output any explanation, punctuation, or whitespace other than "
    "the single digit."
)
CLOUD_JUDGE_MAX_RETRIES = 3
CLOUD_JUDGE_RETRY_DELAY_SECONDS = 2
CLOUD_JUDGE_REQUEST_TIMEOUT = 30

_lev_model = None
_novita_client = None


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def bleurt(candidates: list[str], golds: list, checkpoint: str, threshold: float) -> list[tuple[float, int]]:
    return score_bleurt_batch(candidates, golds, checkpoint=checkpoint, threshold=threshold)


def _get_lev_model():
    global _lev_model
    if _lev_model is None:
        import lev

        _lev_model = lev.load(LEV_CHECKPOINT)
    return _lev_model


def lev_judge(prompts: list[str], candidates: list[str], golds: list) -> list[tuple[float, int]]:
    model = _get_lev_model()
    results = []
    for prompt, answer, gold in zip(prompts, candidates, golds):
        gold_block = "\n".join(f"- {g}" for g in (gold if isinstance(gold, list) else [gold]))
        state = (
            f"Question: {prompt}\n\n"
            f"Reference answer(s):\n{gold_block}\n\n"
            f"Model's answer: {answer}"
        )
        result = model.system_one(
            state,
            {
                "hallucinated": {
                    "type": "noul",
                    "instructions": (
                        "Does the model's answer hallucinate, i.e. fail to match / contradict / "
                        "omit the information in the reference answer(s)? True if it hallucinates, "
                        "false if it is consistent with (or a reasonable paraphrase of) a reference answer."
                    ),
                }
            },
        )
        p_hallucinated = result.answers["hallucinated"].noul
        results.append((p_hallucinated, int(p_hallucinated >= 0.5)))
    return results


def _get_novita_client():
    global _novita_client
    if _novita_client is None:
        from openai import OpenAI
        from qkv_steer.config import REPO_ROOT

        _load_dotenv(REPO_ROOT / ".env")
        api_key = os.environ.get("NOVITA_API_KEY")
        if not api_key:
            sys.exit("NOVITA_API_KEY not set. Add it to .env (repo root) or export it in your shell.")
        _novita_client = OpenAI(api_key=api_key, base_url=NOVITA_BASE_URL)
    return _novita_client


def _cloud_judge_user_prompt(answer: str, gold) -> str:
    gold_answers = gold if isinstance(gold, list) else [gold]
    gold_block = "\n".join(f"- {g}" for g in gold_answers)
    return (
        f"Gold answers:\n{gold_block}\n\n"
        f"Generated response:\n{answer}\n\n"
        "Does the generated response hallucinate relative to the gold "
        "answers? Answer with 1 or 0 only."
    )


def _extract_binary_label(text: str) -> str | None:
    for ch in text:
        if ch in ("0", "1"):
            return ch
    return None


def _do_cloud_request(client, messages, result_box) -> None:
    try:
        result_box["response"] = client.chat.completions.create(
            model=CLOUD_JUDGE_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=100,
            extra_body={"enable_thinking": False},
        )
    except Exception as e:  # noqa: BLE001 -- surfaced to the caller below
        result_box["error"] = e


def _query_cloud_judge(client, answer: str, gold) -> str | None:
    messages = [
        {"role": "system", "content": CLOUD_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _cloud_judge_user_prompt(answer, gold)},
    ]

    last_err = None
    for attempt in range(1, CLOUD_JUDGE_MAX_RETRIES + 1):
        result_box: dict = {}
        worker = threading.Thread(target=_do_cloud_request, args=(client, messages, result_box), daemon=True)
        worker.start()
        worker.join(timeout=CLOUD_JUDGE_REQUEST_TIMEOUT)

        if worker.is_alive():
            last_err = f"request hard-hung past {CLOUD_JUDGE_REQUEST_TIMEOUT}s"
        elif "error" in result_box:
            last_err = str(result_box["error"])
        else:
            try:
                content = (result_box["response"].choices[0].message.content or "").strip()
                label = _extract_binary_label(content)
                if label is not None:
                    return label
                last_err = f"unparseable response: content={content!r}"
            except (KeyError, IndexError, AttributeError) as e:
                last_err = str(e)

        if attempt < CLOUD_JUDGE_MAX_RETRIES:
            time.sleep(CLOUD_JUDGE_RETRY_DELAY_SECONDS)

    print(f"[llm_judge_cloud] giving up after {CLOUD_JUDGE_MAX_RETRIES} attempts ({last_err})", file=sys.stderr)
    return None


def llm_judge_cloud(prompts: list[str], candidates: list[str], golds: list) -> list[tuple[float, int]]:
    client = _get_novita_client()
    results = []
    for answer, gold in zip(candidates, golds):
        label = _query_cloud_judge(client, answer, gold)
        if label is None:
            results.append((float("nan"), 0))
        else:
            results.append((float(label), int(label)))
    return results
