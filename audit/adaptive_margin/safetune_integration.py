"""Fitting the audit into SafeTune's benchmark registry (proposal glue).

SafeTune declares a ``tamper`` benchmark category
(``safetune.instrumentation.evaluate.suite.benchmarks.CATEGORIES``) with
*zero* benchmarks registered under it, and its only white-box attack class
(``AbliterationAttack``) is rank-1 + raw-direction.  This module is the
drop-in sketch for registering the adaptive-margin audit under that category:

* :func:`benchmark_spec` , the :class:`BenchmarkSpec`-shaped entry (plain
  dataclass; no safetune import needed to read it).
* :func:`register` , append it to a live ``benchmarks`` registry module (only
  when safetune is importable; used by the upstream PR, not by us).
* :func:`load_adaptive_margin_contrast_sets` , the loader the spec points at:
  reads leakage-controlled extract/dev splits (CuratorKIT-produced JSONL with
  ``prompt_id, prompt, dataset, category, lang, family``) and returns the
  prompt bundles + screens the audit needs.

Loader path in the spec: ``adaptive_margin.safetune_integration.
load_adaptive_margin_contrast_sets`` (upstream: move this file under
``safetune/data/loaders/`` and the module under
``safetune/instrumentation/evaluate/redteam/``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .screens import build_screens
from .types import PromptItem, PromptSets


@dataclass
class TamperBenchmarkSpec:
    """BenchmarkSpec-shaped entry for SafeTune's registry (category: tamper)."""

    name: str = "adaptive_margin"
    display_name: str = "Adaptive Margin (weight-space tamper audit)"
    loader: str = "adaptive_margin.safetune_integration.load_adaptive_margin_contrast_sets"
    category: str = "tamper"
    description: str = (
        "White-box stress test for refusal-hardened checkpoints: iterated "
        "re-extraction k* (2 selection seeds), single-layer additive dose c* "
        "with norm-matched random controls, base-direction repair and the "
        "margin after repair, capability screen. Judge-injected."
    )
    tags: List[str] = field(default_factory=lambda: ["tamper", "abliteration",
                                                     "whitebox", "attack-cost"])


def benchmark_spec() -> TamperBenchmarkSpec:
    return TamperBenchmarkSpec()


def register(benchmarks_module: Any) -> str:
    """Append the spec to a live ``safetune...suite.benchmarks`` module.

    Returns the registration note; a no-op with a message when the module
    does not look like the registry (so the sketch can never corrupt it).
    """
    specs = getattr(benchmarks_module, "_BENCHMARKS", None)
    if specs is None or not isinstance(specs, list):
        return "registry not found (expected _BENCHMARKS list); nothing registered"
    if any(getattr(s, "name", None) == "adaptive_margin" for s in specs):
        return "already registered"
    specs.append(benchmark_spec())
    return f"registered adaptive_margin under category 'tamper' ({len(specs)} benchmarks)"


# --------------------------------------------------------------------------
# the loader the spec points at
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_adaptive_margin_contrast_sets(splits_dir: str, *, n_extract: int = 128,
                                       n_b: int = 150, n_o: int = 350,
                                       seed1_size: int = 32
                                       ) -> Dict[str, Any]:
    """Load the audit's prompt bundles from a CuratorKIT splits directory.

    Expected files (the leakage-controlled layout used for the frozen runs):
    ``extract_harmful.jsonl``, ``extract_harmless.jsonl``, ``dev_harmful.jsonl``,
    ``dev_overrefusal.jsonl``, ``dev_harmless.jsonl``,
    ``test_capability_mmlu.jsonl``, ``test_capability_gsm8k.jsonl``.

    Returns ``{"sets": PromptSets, "screens": {seed: ScreenInfo}}`` , the two
    objects :class:`~adaptive_margin.benchmark.AdaptiveMarginBenchmark` takes.
    """
    d = Path(splits_dir)
    eh = _read_jsonl(d / "extract_harmful.jsonl")
    eb = _read_jsonl(d / "extract_harmless.jsonl")
    dh = _read_jsonl(d / "dev_harmful.jsonl")
    do = _read_jsonl(d / "dev_overrefusal.jsonl")
    dl = _read_jsonl(d / "dev_harmless.jsonl")
    mm = _read_jsonl(d / "test_capability_mmlu.jsonl")
    gs = _read_jsonl(d / "test_capability_gsm8k.jsonl")

    # the eligible pool for selection screens: the extract split's remaining
    # harmful rows of the behaviour dataset (leakage-controlled: screens are
    # drawn from extract-side pools, never from the B/O eval sets)
    eligible = [r for r in eh[n_extract:] if r["dataset"] == "mlabonne/harmful_behaviors"]
    fixed = eligible[:seed1_size]

    sets = PromptSets(
        extract_h=[r["prompt"] for r in eh[:n_extract]],
        extract_b=[r["prompt"] for r in eb[:n_extract]],
        hl=[r["prompt"] for r in dl],
        B=[PromptItem(id=r["prompt_id"], prompt=r["prompt"]) for r in dh[:n_b]],
        O=[PromptItem(id=r["prompt_id"], prompt=r["prompt"]) for r in do[:n_o]],
        mmlu=[{"prompt": r["prompt"], "gold": r["gold"]} for r in mm],
        gsm8k=[{"prompt": r["prompt"], "gold": r["gold"]} for r in gs],
    )
    screens = build_screens(
        pool_ids=[r["prompt_id"] for r in eligible],
        pool_prompts=[r["prompt"] for r in eligible],
        fixed_ids=[r["prompt_id"] for r in fixed],
        fixed_prompts=[r["prompt"] for r in fixed],
        benign_screen=[r["prompt"] for r in eb[n_extract:n_extract + 32]],
        seed1_size=seed1_size,
    )
    return {"sets": sets, "screens": screens}
