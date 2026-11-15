"""Candidate-direction selection (pure).

Mirrors the harness's ``select()`` decision: among candidate (layer, pos)
directions, keep those whose benign-screen KL is within budget, then take the
one with the lowest screen marker-refusal rate (KL as tie-break).  If nothing
is within budget, fall back to all candidates (budget as preference, not
gate) , exactly ``ok = [r for r in rows if r.kl <= 0.5] or rows`` in the
harness.
"""
from __future__ import annotations

from typing import List, Optional

from .types import SelectionRow


def pick_direction(rows: List[SelectionRow], kl_budget: float = 0.5) -> SelectionRow:
    """Select the best candidate direction under the KL budget.

    Deterministic: min over (marker_rate, kl); ties resolved by list order
    (layers ascending, then positions in spec order).
    """
    if not rows:
        raise ValueError("no candidate rows to select from")
    ok = [r for r in rows if r.kl <= kl_budget] or rows
    return min(ok, key=lambda r: (r.marker_rate, r.kl))


def within_budget(rows: List[SelectionRow], kl_budget: float = 0.5) -> List[SelectionRow]:
    """The budget-passing subset (empty list when nothing passes)."""
    return [r for r in rows if r.kl <= kl_budget]
