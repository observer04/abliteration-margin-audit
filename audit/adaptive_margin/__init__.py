"""adaptive_margin , SafeTune-style white-box tamper audit for refusal-hardened
checkpoints: the adaptive margin k*, the additive dose c*, repair, and the
margin after repair, with norm-matched random controls and a capability screen.

Pure core (importable everywhere, no torch):

* :class:`~adaptive_margin.benchmark.AdaptiveMarginBenchmark` , the
  ``TamperBenchmark``-like orchestrator (drives any backend).
* :mod:`~adaptive_margin.selection` / :mod:`~adaptive_margin.accounting` ,
  the decisions and the verdicts (unit-tested without a model).
* :class:`~adaptive_margin.fake_backend.SyntheticBackend` , deterministic
  synthetic model for tests and demos.

Remote-only pieces (import torch; never on the local machine):

* :mod:`~adaptive_margin.torch_backend` , the real backend.
* :class:`~adaptive_margin.judges.WildGuardJudge` , the semantic judge.

Reference results: ``audit/reproduce_frozen.py`` recomputes the frozen
E3 panel verdicts (4 checkpoints) from counts alone.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .accounting import (c_star, first_break, k_star, randoms_hold, repair_account,
                         summarize, load_counts_csv, csv_to_records)
from .benchmark import AdaptiveMarginBenchmark
from .config import (BelowHalfBreak, BreakCriterion, D5AttackSuccessBreak,
                     E3_REFERENCE, MarginSpec, RelativeBreak, ReferenceRow)
from .judges import Judge, MarkerJudge, count_labels
from .safetune_integration import benchmark_spec, register
from .selection import pick_direction
from .types import (CapabilityCounts, ConditionRecord, DosePoint, LadderStep,
                    MarginReport, NormRatio, PromptItem, PromptSets, ScreenInfo,
                    SelectionRow, SplitCounts)

__all__ = [
    "AdaptiveMarginBenchmark",
    "MarginSpec", "BreakCriterion", "BelowHalfBreak", "RelativeBreak",
    "D5AttackSuccessBreak", "ReferenceRow", "E3_REFERENCE",
    "Judge", "MarkerJudge", "count_labels",
    "benchmark_spec", "register",
    "pick_direction",
    "first_break", "k_star", "c_star", "randoms_hold", "repair_account",
    "summarize", "load_counts_csv", "csv_to_records",
    "MarginReport", "ConditionRecord", "SplitCounts", "LadderStep", "DosePoint",
    "SelectionRow", "PromptSets", "PromptItem", "ScreenInfo", "NormRatio",
    "CapabilityCounts",
    "__version__",
]
