"""Selection-screen construction (pure).

The k* ladder needs a small harmful "selection screen" to score candidate
directions (cheap marker counts, not the full judge).  Screen stability is
part of the audit: seed 0 is the fixed screen (first 32 of the eligible pool,
in file order, as in the validated harness) and seed 1 is a *different*
screen drawn from the same pool with a fixed RNG , two screens, one k*
verdict.  The benign screen is held fixed across seeds so the KL tie-break
stays comparable.
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Sequence, Tuple

from .types import ScreenInfo


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_screens(pool_ids: Sequence[str], pool_prompts: Sequence[str],
                  fixed_ids: Sequence[str], fixed_prompts: Sequence[str],
                  benign_screen: Sequence[str], seed1_size: int = 32,
                  seed1_rng: int = 1) -> Dict[int, ScreenInfo]:
    """Build the seed-0 (fixed) and seed-1 (reshuffled) selection screens.

    pool_ids/pool_prompts: the full eligible harmful pool (screen candidates).
    fixed_ids/fixed_prompts: the harness's fixed screen (a subset of the pool,
        first-k in file order).
    benign_screen: the fixed benign screen (KL reference for selection).
    """
    if len(pool_ids) != len(pool_prompts):
        raise ValueError("pool ids/prompts length mismatch")
    if len(fixed_ids) != len(fixed_prompts):
        raise ValueError("fixed screen ids/prompts length mismatch")
    pool = list(zip(pool_ids, pool_prompts))
    fixed_pairs = list(zip(fixed_ids, fixed_prompts))
    for fid in fixed_ids:
        if fid not in pool_ids:
            raise ValueError(f"fixed screen id {fid!r} not in the eligible pool")

    rng = random.Random(seed1_rng)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    take = shuffled[:seed1_size]
    fixed_hashes = {i: sha_text(i)[:16] for i in fixed_ids}
    seed1_hashes = {i: sha_text(i)[:16] for i, _ in take}

    screens = {
        0: ScreenInfo(
            seed=0, ids=list(fixed_ids), harmful=list(fixed_prompts),
            benign=list(benign_screen),
            ids_sha256=sha_text("\n".join(fixed_ids)),
            pool_sha256=sha_text("\n".join(pool_ids)),
            pool_n=len(pool_ids),
            note="fixed screen: first rows of the eligible pool in file order "
                 "(the validated harness's seed-0 screen); benign screen fixed",
            row_id_hashes=fixed_hashes,
        ),
        1: ScreenInfo(
            seed=1, ids=[i for i, _ in take], harmful=[p for _, p in take],
            benign=list(benign_screen),
            ids_sha256=sha_text("\n".join(i for i, _ in take)),
            pool_sha256=sha_text("\n".join(pool_ids)),
            pool_n=len(pool_ids),
            note=f"random.Random({seed1_rng}).shuffle over the {len(pool_ids)}-row eligible "
                 f"pool, first {seed1_size}; benign screen fixed",
            row_id_hashes=seed1_hashes,
            eligible_index=[pool_ids.index(i) for i, _ in take],
        ),
    }
    return screens


def screens_disjoint(a: ScreenInfo, b: ScreenInfo) -> List[str]:
    """Ids appearing in both screens (normally empty for harmful screens)."""
    return sorted(set(a.ids) & set(b.ids))
