"""Paired statistics for A/B condition comparisons (T02).

PLAN.md requires paired bootstrap 95% CIs and exact McNemar. AuditKIT ships
only a two-sided paired bootstrap *p-value* (``auditkit._bootstrap``), so:

- ``paired_bootstrap_p`` delegates to the vendored AuditKIT implementation
  when importable (identical by construction) and otherwise runs a verbatim
  reimplementation of it (stdlib ``random.Random``, centered differences,
  ties-as-extreme, ``(count+1)/(n+1)``) so remote jobs without ``vendor/``
  produce the same numbers. ``paired_bootstrap_ci`` wraps it with the
  percentile CI AuditKIT lacks.
- ``exact_mcnemar`` / ``binary_discordants`` / ``mcnemar_from_pairs`` are ours
  (exact binomial, no scipy needed).
- ``wilson_ci`` returns (NaN, NaN) for n <= 0; k outside [0, n] raises.
- ``cohen_kappa``: judge-agreement statistic (degenerate → NaN).

All functions are pure (stdlib), CPU, deterministic given ``seed``.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n → (lo, hi).

    ``n <= 0`` → (NaN, NaN) (the caller checks n first); k outside [0, n]
    raises ValueError. Wilson over Wald because ASR/over-refusal rates
    approach 0 and 1, where Wald collapses.
    """
    if n <= 0:
        return float("nan"), float("nan")
    if not 0 <= k <= n:
        raise ValueError(f"k must be within [0, n], got k={k}, n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def exact_mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts b, c.

    b = #pairs where A succeeds and B fails; c = the reverse. Under the null
    b ~ Binom(b+c, 0.5); the two-sided exact p is the probability of a split
    at least as imbalanced (tail masses summed, doubly-counted middle
    removed when the tails overlap, i.e. when 2*min(b,c) >= b+c).
    ``b + c == 0`` → 1.0 (no evidence). Uses math.comb, no scipy.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_low = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    p_high = sum(math.comb(n, i) for i in range(n - k, n + 1)) / 2**n
    p = p_low + p_high
    if 2 * k < n:  # disjoint tails
        return min(1.0, p)
    # overlapping tails: subtract the doubly counted middle mass
    return min(1.0, p - sum(math.comb(n, i) for i in range(k + 1, n - k)) / 2**n)


def binary_discordants(pairs: Sequence[tuple[Any, Any]]) -> tuple[int, int]:
    """(b, c) discordant counts from paired boolean outcomes:
    b = left-True/right-False, c = left-False/right-True."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if (not x) and y)
    return b, c


def mcnemar_from_pairs(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """Exact McNemar over paired booleans (baseline, candidate) → detail dict."""
    b, c = binary_discordants(pairs)
    return {"b": b, "c": c, "n_discordant": b + c, "n_pairs": len(pairs),
            "p_value": exact_mcnemar(b, c)}


def _ak_paired_bootstrap(pairs: Sequence[tuple[float, float]], n_resamples: int,
                         seed: int) -> dict[str, Any]:
    """Verbatim reimplementation of ``auditkit._bootstrap.paired_bootstrap``.

    Same stdlib ``random.Random(seed)`` stream, same centered-resampling
    p-value with ties-as-extreme, same rounding — so results match the
    vendored library bit-for-bit when it is not importable (remote jobs).
    """
    n = len(pairs)
    if n < 2:
        return {"error": "need at least 2 aligned sample pairs", "n": n}
    diffs = [c - b for b, c in pairs]
    mean_baseline = math.fsum(b for b, _ in pairs) / n
    mean_candidate = math.fsum(c for _, c in pairs) / n
    observed = math.fsum(diffs) / n
    rng = random.Random(seed)
    centred = [d - observed for d in diffs]
    threshold = abs(observed) - 1e-9 * max(1.0, abs(observed))
    count = 0
    for _ in range(n_resamples):
        resample = [rng.choice(centred) for _ in range(n)]
        if abs(math.fsum(resample) / n) >= threshold:
            count += 1
    p = (count + 1) / (n_resamples + 1)
    return {
        "n": n,
        "mean_baseline": round(mean_baseline, 6),
        "mean_candidate": round(mean_candidate, 6),
        "delta": round(observed, 6),
        "p_value": round(p, 6),
        "significant": p < 0.05,
    }


def paired_bootstrap_p(pairs: Sequence[tuple[float, float]],
                       n_resamples: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Two-sided paired bootstrap p-value, AuditKIT-identical.

    Delegates to ``auditkit._bootstrap.paired_bootstrap`` when the vendor
    clone is importable; otherwise runs the verbatim local copy.
    """
    from ._auditkit import import_auditkit
    ak = import_auditkit()
    if ak is not None:
        try:
            from auditkit._bootstrap import paired_bootstrap as _ak_pb
            return _ak_pb(list(pairs), n_resamples=n_resamples, seed=seed)
        except Exception:
            pass  # fall through to the local copy
    return _ak_paired_bootstrap(pairs, n_resamples, seed)


def bootstrap_diff_ci(pairs: Sequence[tuple[float, float]],
                      n_resamples: int = 10_000, seed: int = 42,
                      alpha: float = 0.05) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean difference (candidate - baseline).

    Deterministic (stdlib ``random.Random(seed)``). Returns n, means, delta
    and ``ci_low``/``ci_high``; fewer than 2 pairs → NaN bounds.
    """
    n = len(pairs)
    if n < 2:
        return {"n": n, "mean_baseline": float("nan"), "mean_candidate": float("nan"),
                "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    base = [float(b) for b, _ in pairs]
    cand = [float(c) for _, c in pairs]
    diffs = [c - b for b, c in zip(base, cand)]
    mean_baseline = math.fsum(base) / n
    mean_candidate = math.fsum(cand) / n
    delta = math.fsum(diffs) / n
    rng = random.Random(seed)
    means = sorted(
        math.fsum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo_idx = max(0, min(int(math.floor(alpha / 2 * n_resamples)), n_resamples - 1))
    hi_idx = max(0, min(int(math.ceil((1 - alpha / 2) * n_resamples)) - 1, n_resamples - 1))
    return {"n": n, "mean_baseline": round(mean_baseline, 6),
            "mean_candidate": round(mean_candidate, 6), "delta": round(delta, 6),
            "ci_low": means[lo_idx], "ci_high": means[hi_idx]}


def paired_bootstrap_ci(pairs: Sequence[tuple[float, float]],
                        n_resamples: int = 10_000, seed: int = 42) -> dict[str, Any]:
    """Combined paired-bootstrap result: delta + 95% percentile CI + p-value.

    The CI resamples the observed differences (effect estimate); the p-value
    comes from ``paired_bootstrap_p`` (centered-difference null, AuditKIT's
    algorithm). Fewer than 2 pairs → NaN everywhere.
    """
    ci = bootstrap_diff_ci(pairs, n_resamples=n_resamples, seed=seed)
    if ci["n"] < 2:
        return {"n": ci["n"], "delta": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_value": float("nan")}
    pv = paired_bootstrap_p(pairs, n_resamples=n_resamples, seed=seed)
    return {"n": ci["n"], "delta": ci["delta"], "lo": ci["ci_low"],
            "hi": ci["ci_high"], "p_value": pv.get("p_value", float("nan"))}


def compare_proportions(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """Paired comparison of two success rates (baseline, candidate).

    ``rate_baseline``/``rate_candidate`` are the marginal rates; b/c are the
    McNemar discordants; ``p_value`` is the exact McNemar p. Pairs where
    either side is None are dropped.
    """
    clean = [(bool(x), bool(y)) for x, y in pairs if x is not None and y is not None]
    n = len(clean)
    b, c = binary_discordants(clean)
    rate_b = sum(1 for x, _ in clean if x) / n if n else float("nan")
    rate_c = sum(1 for _, y in clean if y) / n if n else float("nan")
    return {
        "b": b, "c": c, "n_discordant": b + c, "n_pairs": n,
        "p_value": exact_mcnemar(b, c),
        "rate_baseline": rate_b, "rate_candidate": rate_c,
        "delta": rate_c - rate_b if n else float("nan"),
    }


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa for two same-length label sequences (judge agreement).

    NaN when empty or when one judge labels everything the same way
    (po = pe = 1 → 0/0; degenerate agreement is NaN, not 1).
    """
    if len(a) != len(b) or not a:
        return float("nan")
    labels = sorted(set(a) | set(b), key=str)
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    table = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        table[idx[x]][idx[y]] += 1
    po = sum(table[i][i] for i in range(len(labels))) / n
    pe = sum(sum(table[i]) * sum(table[j][i] for j in range(len(labels)))
             for i in range(len(labels))) / n**2
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def align_by_prompt_id(
    rows_a: Iterable[dict[str, Any]], rows_b: Iterable[dict[str, Any]],
    key: str | Callable[[dict[str, Any]], Any] = "judge_harmful",
) -> list[tuple[Any, Any]]:
    """(value_a, value_b) pairs for rows sharing a prompt_id (first wins per
    id; None values are kept so McNemar callers can drop them themselves).

    ``key`` is a column name or a ``row -> value`` callable.
    """
    def value(row: dict[str, Any]) -> Any:
        return key(row) if callable(key) else row.get(key)

    def index(rows):
        out: dict[str, Any] = {}
        for r in rows:
            pid = str(r.get("prompt_id"))
            if pid not in out:
                out[pid] = value(r)
        return out

    ia, ib = index(rows_a), index(rows_b)
    shared = sorted(ia.keys() & ib.keys())
    return [(ia[pid], ib[pid]) for pid in shared]
