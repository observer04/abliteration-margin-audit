"""JSONL row IO for the PLAN.md generation/judgment interfaces.

Generations JSONL (``results/gens/<run_id>.jsonl``), one row per completion:
``run_id, model_id, defense, attack, attack_rank, layer, strength, quant,
prompt_id, split, dataset, lang, category, prompt, response, n_new_tokens,
capped, gen_cfg_hash, seed`` (plus optional per-row extras such as ``target``
for capability rows or ``first_token_kl`` computed at generation time —
unknown keys are preserved everywhere).

Judgments JSONL (``results/judged/<run_id>.<judge>.jsonl``): the same keys
plus ``judge, judge_harmful, judge_refusal, judge_raw, marker_refusal``.

Resume keys are ``(run_id, prompt_id, gen_cfg_hash)`` — a regenerated row
(new gen_cfg_hash) is never skipped as already-judged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

GENS_FIELDS = (
    "run_id", "model_id", "defense", "attack", "attack_rank", "layer", "strength",
    "quant", "prompt_id", "split", "dataset", "lang", "category", "prompt",
    "response", "n_new_tokens", "capped", "gen_cfg_hash", "seed",
)
JUDGE_FIELDS = ("judge", "judge_harmful", "judge_refusal", "judge_raw", "marker_refusal")
JUDGED_FIELDS = GENS_FIELDS + JUDGE_FIELDS
CONDITION_KEYS = ("model_id", "defense", "attack", "attack_rank", "layer", "strength", "quant")

# Datasets by role (PLAN.md "Evaluation protocol"). ``dataset_role`` returns
# "unknown" for unknown datasets so they are never silently pooled into a rate.
# T09 fix: the actual split/gens ``dataset`` strings are org-prefixed Hub ids
# (LibrAI/do-not-answer, mlabonne/*, bench-llm/or-bench, Alpaca) — without
# these entries every E0 rate came out NaN on n_unknown_role.
HARMFUL_DATASETS = frozenset({"harmbench", "jailbreakbench", "strongreject",
                              "librai/do-not-answer", "mlabonne/harmful_behaviors"})
BENIGN_DATASETS = frozenset({"xstest", "xstest_safe", "or-bench", "orbench", "or-bench-hard",
                             "bench-llm/or-bench", "alpaca", "mlabonne/harmless_alpaca"})
CAPABILITY_DATASETS = frozenset({"mmlu", "gsm8k"})
MULTILINGUAL_DATASETS = frozenset({"multijail", "multi_jail"})

ROLE_PREFIXES = {
    "harmful": ("harmbench", "jailbreakbench", "strongreject", "advbench",
                "librai/do-not-answer", "mlabonne/harmful"),
    "benign": ("xstest", "or-bench", "orbench", "bench-llm/or-bench",
               "alpaca", "mlabonne/harmless"),
    "capability": ("mmlu", "gsm8k"),
    "multilingual": ("multijail", "multi_jail"),
}
ROLE_EXACT = {
    "harmful": HARMFUL_DATASETS,
    "benign": BENIGN_DATASETS,
    "capability": CAPABILITY_DATASETS,
    "multilingual": MULTILINGUAL_DATASETS,
}


def dataset_role(dataset: str) -> str:
    """harmful / benign / capability / multilingual / unknown.

    Tolerates prefixed ids like ``harmbench_test`` or ``xstest-safe``;
    unknown datasets return "unknown" so they are excluded from every rate
    and surfaced through ``n_unknown_role`` in the condition table instead.
    """
    d = (dataset or "").strip().lower()
    if not d:
        return "unknown"
    for role, prefixes in ROLE_PREFIXES.items():
        if d in ROLE_EXACT[role] or d.startswith(prefixes):
            return role
    return "unknown"


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Blank lines are skipped;
    invalid JSON raises ValueError naming the line."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path}: line {n}: invalid JSON ({e})") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {n}: expected object, got {type(row).__name__}")
            yield row


def write_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]] | Iterator[dict[str, Any]],
                append: bool = False) -> int:
    """Write rows as JSONL (flushed per call so a crashed writer loses nothing).

    Returns the number of rows written. Append mode keeps existing content.
    """
    mode = "a" if append else "w"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        fh.flush()
    return count


def validate_gens_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate a generations row; returns it unchanged. Raises ValueError
    naming the missing required keys. ``attack_rank``/``layer``/``strength``
    may be None (base condition has no attack); ``n_new_tokens``/``seed`` are
    expected but tolerated missing for pre-judged legacy rows."""
    hard = [k for k in GENS_FIELDS
            if row.get(k) is None and k not in ("attack_rank", "layer", "strength",
                                                "n_new_tokens", "seed")]
    if hard:
        raise ValueError(f"gens row missing required keys {hard}: {sorted(row)[:12]}...")
    return row


# short alias used by the judge path
gens_row = validate_gens_row


def validate_judged_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate a judgments row: a valid gens row plus a judge name and at
    least one verdict field (``judge_harmful`` and/or ``judge_refusal``).
    ``judge_raw``/``marker_refusal`` may be absent (diagnostic extras)."""
    validate_gens_row(row)
    if row.get("judge") in (None, ""):
        raise ValueError(f"judged row missing judge: {sorted(row)[:12]}...")
    if "judge_harmful" not in row and "judge_refusal" not in row:
        raise ValueError(f"judged row has no verdict field "
                         f"(judge_harmful/judge_refusal): {sorted(row)[:12]}...")
    return row


# short aliases used by the judge path and tests
gens_row = validate_gens_row
judged_row = validate_judged_row


def _resume_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("run_id")), str(row.get("prompt_id")), str(row.get("gen_cfg_hash")))


def judged_keys(path: str | os.PathLike[str], judge: str | None = None) -> set[tuple[str, str, str]]:
    """Resume keys ``(run_id, prompt_id, gen_cfg_hash)`` already in a judged
    file, optionally filtered to one judge name. A regenerated row (new
    gen_cfg_hash) is therefore never skipped."""
    out: set[tuple[str, str, str]] = set()
    if not os.path.exists(path):
        return out
    for row in iter_jsonl(path):
        if judge is not None and str(row.get("judge") or "") != judge:
            continue
        out.add(_resume_key(row))
    return out
