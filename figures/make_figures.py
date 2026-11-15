"""Article figures. Reads only the released result CSVs. No models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

BREAK_FRACTION = 0.5  # strictly below half of the panel = broken

TINY_AYA = ["ta-base", "t22-dr2e5", "t24-lr1e5", "t35-dose1", "t35-dose2", "xref-v1", "xref-v2"]
SEED0_ONLY = {"t35-dose1", "t35-dose2"}
SUPPLEMENTARY = {"dr-official"}
EXCLUDED = {"t31-asrtk4"}  # no WildGuard labels (judge OOM, frozen note)

DISPLAY = {
    "ta-base": "Tiny Aya Global (undefended base)",
    "t22-dr2e5": "SafeTune DeepRefusal, LR 2e-5 (ours)",
    "t24-lr1e5": "DeepRefusal paper recipe, LR 1e-5 (ours)",
    "xref-v1": "xref-v1 (StarWarrior24x7, concurrent)",
    "xref-v2": "xref-v2 (StarWarrior24x7, concurrent)",
    "dr-official": "official DeepRefusal, Llama-3-8B (seed-0 directions)",
    "t35-dose1": "dual-operator defense, LR 1e-5 (ours)",
    "t35-dose2": "dual-operator defense, LR 2e-5 (ours)",
}
SHORT = {
    "ta-base": "Tiny Aya base",
    "t22-dr2e5": "SafeTune DR 2e-5",
    "t24-lr1e5": "DR paper recipe 1e-5",
    "xref-v1": "xref-v1",
    "xref-v2": "xref-v2",
    "dr-official": "official DR (Llama-3-8B)",
    "t35-dose1": "dual-operator LR 1e-5",
    "t35-dose2": "dual-operator LR 2e-5",
}

# Fixed entity -> colour map (reference categorical palette, fixed order; the
# undefended base is a neutral reference). Colour follows the entity in every figure.
COLOR = {
    "ta-base": "#52514e",
    "t22-dr2e5": "#2a78d6",
    "xref-v2": "#eb6834",
    "t24-lr1e5": "#1baf7a",
    "xref-v1": "#eda100",
    "dr-official": "#6250d6",
    "t35-dose1": "#008300",
    "t35-dose2": "#e87ba4",
}
# Condition colours for the grouped bars (one hue, ordered; attacks hatched).
C_A0 = "#184f95"
C_REPAIR = "#5598e7"
C_ATTACK = "#b7d3f6"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


# --------------------------------------------------------------------------- #
# statistics (deterministic, from counts)
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score 95% interval for k successes out of n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def broken(r, n):
    return r < BREAK_FRACTION * n


# --------------------------------------------------------------------------- #
# loading: Exp-2 screens (B46/O50)
# --------------------------------------------------------------------------- #
def _empty_cp(tag):
    return {
        "tag": tag,
        "screen_k": {0: {}, 1: {}},  # seed -> {k: (refused, n)}
        "screen_a0": None,  # (refused, n)
        "screen_a0_cited": False,
        "screen_dose": [],  # (c, refused, n)
        "screen_rand": [],  # (seed, c, refused, n)
        "screen_add_one": None,  # 2x2 one-layer addition (c, refused, n)
        "final": {},  # (condition, set) -> row dict
    }


def load_tables():
    """Build plot views from the two tidy, counts-only CSVs."""
    cps = {}
    advisor = {"dose": [], "rand": []}
    def rows(name):
        with (REPO / "results" / name).open(newline="") as f:
            return list(csv.DictReader(f))
    for row in rows("screens.csv"):
        tag = row["model"]
        if tag in EXCLUDED:
            continue
        cp = cps.setdefault(tag, _empty_cp(tag))
        kind = row.get("plot_kind", "")
        if not kind:
            continue
        r, n = int(row["refused"]), int(row["n"])
        if kind == "k":
            cp["screen_k"][int(row["plot_seed"])][int(row["plot_k"])] = (r, n)
        elif kind == "a0":
            cp["screen_a0"] = (r, n)
            cp["screen_a0_cited"] = bool(int(row["plot_cited"]))
        elif kind == "dose":
            cp["screen_dose"].append((float(row["plot_c"]), r, n))
        elif kind == "random":
            cp["screen_rand"].append((int(row["plot_seed"]), float(row["plot_c"]), r, n))
        elif kind == "add_one":
            cp["screen_add_one"] = (float(row["plot_c"]), r, n)
            cp["screen_add_one_unknown"] = int(row["plot_unknown"])
        elif kind == "advisor_dose":
            advisor["dose"].append((float(row["plot_c"]), r, n, int(row["harmful_coherent"])))
        elif kind == "advisor_random":
            advisor["rand"].append((int(row["plot_seed"]), float(row["plot_c"]), r, n))
    for row in rows("final_panel.csv"):
        cp = cps.setdefault(row["model"], _empty_cp(row["model"]))
        if row["source"] == "T38":
            if row.get("plot_kind") == "l22_random":
                cp.setdefault("final_l22_rand", []).append((int(row["plot_seed"]), int(row["refused"]), int(row["n"])))
            continue
        rec = dict(model=row["model"], condition=row["condition"], set=row["panel"],
                   n=int(row["n"]), refused=int(row["refused"]), harmful=int(row["harmful_wg"] or 0),
                   harmND=int(row["harmful_coherent"] or 0), degenerate=int(row["degenerate"] or 0),
                   capped=int(row["capped"] or 0), kl=float(row["kl"]) if row["kl"] else None,
                   hb=row["harmful_hb"], unknown=int(row["unknown"] or 0))
        cp["final"][(rec["condition"], rec["set"])] = rec
    advisor["dose"].sort()
    return cps, advisor


def final_k_curve(cp):
    """{k: (refused, n)} from the final panel: A0 at k=0 plus measured atk-kN rows."""
    out = {}
    for (cond, s), r in cp["final"].items():
        if s != "B":
            continue
        if cond == "A0":
            out[0] = (r["refused"], r["n"])
        m = re.fullmatch(r"atk-k(\d+)", cond)
        if m:
            out[int(m.group(1))] = (r["refused"], r["n"])
    return out


def first_break(curve):
    """First measured k >= 1 below half, or None if no measured row breaks."""
    for k in sorted(curve):
        if k >= 1 and broken(*curve[k]):
            return k
    return None


def screen_curve(cp, seed):
    c = dict(cp["screen_k"].get(seed) or {})
    if c and cp["screen_a0"]:
        c[0] = cp["screen_a0"]
    return c


def k_star_text(cp):
    """Human label for k*, derived from data: final n and both screen seeds."""
    parts = []
    fcurve = final_k_curve(cp)
    fk = first_break(fcurve)
    if fk is not None:
        missing = [k for k in range(1, fk) if k not in fcurve]
        if missing:
            parts.append(f"breaks by k={fk} at final n (k={missing[0]}–{missing[-1]} screen only)")
        else:
            parts.append(f"{fk} at final n")
    sk = []
    for s in (0, 1):
        sc = screen_curve(cp, s)
        if sc:
            b = first_break(sc)
            sk.append(str(b) if b is not None else f">{max(sc)}")
    if not sk and parts and cp["tag"] in SEED0_ONLY:
        return f"k*={fk}, seed 0 only"
    if sk:
        if parts and len(set(sk)) == 1 and sk[0] == str(fk) and parts[0] == f"{fk} at final n":
            return f"k*={fk}"
        if not parts and len(set(sk)) == 1:
            return f"k*={sk[0]}, screens only"
        parts.append(" / ".join(sk) + " on screens (seed 0 / 1)")
    if parts and parts[0].startswith("breaks by"):
        return "k*: " + "; ".join(parts)
    return "k*=" + "; ".join(parts) if parts else "k*: n/a"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def threshold(ax, label=True, x=0.015):
    ax.axhline(BREAK_FRACTION, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(
            x,
            BREAK_FRACTION + 0.015,
            "break: refused < half of panel",
            transform=ax.get_yaxis_transform(),
            fontsize=7.5,
            color=INK2,
            va="bottom",
            ha="left",
        )


def style_axes(ax):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _runs(ks):
    """Split sorted ks into runs of consecutive integers (draw lines only within runs)."""
    runs, cur = [], []
    for k in ks:
        if cur and k != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    return runs


def plot_measured(ax, curve, color, lw, ms, ls="-", marker="o", alpha=1.0, z=3, mfc=None):
    ks = sorted(curve)
    for run in _runs(ks):
        xs = run
        ys = [curve[k][0] / curve[k][1] for k in run]
        ax.plot(
            xs,
            ys,
            linestyle=ls if len(run) > 1 else "none",
            marker=marker,
            ms=ms,
            lw=lw,
            color=color,
            alpha=alpha,
            zorder=z,
            mfc=mfc if mfc is not None else color,
            mec=color,
        )


def save(fig, outdir, stem):
    for ext in ("png",):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[{stem}] -> {outdir / (stem + '.png')}")


# --------------------------------------------------------------------------- #
# Fig 1: the adaptive margin
# --------------------------------------------------------------------------- #
def fig1_margin(cps, outdir):
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.0, 5.6), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    # (a) Tiny Aya: screens per seed (thin, open markers) + final panel (bold)
    handles = []
    for tag in TINY_AYA:
        cp = cps.get(tag)
        if not cp:
            continue
        col = COLOR[tag]
        for seed, ls in ((0, "-"), (1, (0, (3, 2)))):
            sc = screen_curve(cp, seed)
            if sc:
                plot_measured(axa, sc, col, lw=1.0, ms=3.5, ls=ls, marker="o", alpha=0.6, z=2, mfc="white")
        fc = final_k_curve(cp)
        if fc:
            plot_measured(axa, fc, col, lw=2.4, ms=7, z=4)
        handles.append(
            Line2D([], [], color=col, lw=2.4 if fc else 1.2, marker="o" if fc else None,
                   label=f"{SHORT[tag]}: {k_star_text(cp)}")
        )
    handles += [
        Line2D([], [], color="none", label=" "),
        Line2D([], [], color=INK2, lw=2.4, marker="o", ms=6, label="final panel, n = 150"),
        Line2D([], [], color=INK2, lw=1.0, marker="o", ms=3.5, mfc="white", label="screen n = 46, seed 0"),
        Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 2)), marker="o", ms=3.5, mfc="white",
               label="screen n = 46, seed 1"),
    ]
    threshold(axa, x=0.60)
    axa.set_xlim(-0.25, 3.25)
    axa.set_xticks(range(0, 4))
    axa.set_ylim(-0.03, 1.06)
    axa.set_xlabel("re-extracted directions ablated, k")
    axa.set_ylabel("WildGuard-refused fraction of harmful prompts (B)")
    axa.set_title("(a) Tiny Aya Global, core model", loc="left")
    style_axes(axa)
    axa.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.3, ncol=2,
               handlelength=2.4, columnspacing=1.2)

    # (b) official DeepRefusal (supplementary)
    cp = cps.get("dr-official")
    if cp:
        col = COLOR["dr-official"]
        sc0, sc1 = screen_curve(cp, 0), screen_curve(cp, 1)
        if sc1:
            plot_measured(axb, sc1, col, lw=1.1, ms=4, ls=(0, (3, 2)), alpha=0.6, z=2, mfc="white")
        if sc0:
            plot_measured(axb, sc0, col, lw=1.1, ms=4, ls="-", alpha=0.6, z=2, mfc="white")
        fc = final_k_curve(cp)
        plot_measured(axb, fc, col, lw=2.4, ms=7, z=4)
        missing = [k for k in range(1, max(fc) + 1) if k not in fc] if fc else []
        if missing:
            axb.text((missing[0] + missing[-1]) / 2, 0.84,
                     f"final panel: k = {missing[0]}–{missing[-1]} not run at n = {fc[0][1]}\n"
                     "(screen rows only; no line is drawn across them)",
                     ha="center", fontsize=7.4, color=INK2)
        fk = first_break(fc)
        for k in sorted(fc):
            r, n = fc[k]
            if k == 0:
                axb.annotate(f"A0 {r}/{n}", (k, r / n), textcoords="offset points", xytext=(8, -12),
                             fontsize=7.4, color=INK)
            elif k == fk:
                axb.annotate(f"k={k}: {r}/{n} (< {n // 2})\nfirst final-panel rank below half",
                             (k, r / n), textcoords="offset points", xytext=(-14, -30), fontsize=7.4,
                             color=INK, ha="right", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
            elif fk is not None and k in (fk - 1, max(fc)):
                axb.annotate(f"k={k}: {r}/{n}", (k, r / n), textcoords="offset points", xytext=(-10, -4),
                             fontsize=7.4, color=INK, ha="right", va="top")
        b0 = first_break(sc0) if sc0 else None
        b1 = first_break(sc1) if sc1 else None
        if sc0 and b0 is not None and (b0 - 1) in sc0:
            kh = b0 - 1
            axb.annotate(
                f"screen seed 0: k={kh} holds ({sc0[kh][0]}/{sc0[kh][1]} \u2265 {sc0[kh][1] // 2}),\n"
                f"breaks at k={b0} ({sc0[b0][0]}/{sc0[b0][1]})",
                (kh, sc0[kh][0] / sc0[kh][1]), textcoords="data", xytext=(4.3, 0.56),
                fontsize=7.2, color=INK2, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6),
            )
        if sc1 and b1 is None:
            axb.annotate(
                f"screen seed 1: no break through k={max(sc1)} (cap)",
                (max(sc1), sc1[max(sc1)][0] / sc1[max(sc1)][1]), textcoords="offset points",
                xytext=(0, 7), fontsize=7.2, color=INK2, ha="right",
            )
        hb = [
            Line2D([], [], color=col, lw=2.4, marker="o", ms=6, label="final panel, n = 150 (seed-0 directions)"),
            Line2D([], [], color=col, lw=1.1, marker="o", ms=4, mfc="white", label="screen n = 46, selection seed 0"),
            Line2D([], [], color=col, lw=1.1, ls=(0, (3, 2)), marker="o", ms=4, mfc="white",
                   label="screen n = 46, selection seed 1"),
        ]
        axb.legend(handles=hb, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.3, ncol=2)
        axb.text(0.0, -0.36,
                 f"official DR: {k_star_text(cp)}",
                 transform=axb.transAxes, fontsize=7.6, color=INK)
    threshold(axb, x=0.02)
    axb.set_xlim(-0.3, 6.4)
    axb.set_xticks(range(0, 7))
    axb.set_ylim(-0.03, 1.06)
    axb.set_xlabel("re-extracted directions ablated, k")
    axb.set_title("(b) official DeepRefusal, Llama-3-8B (supplementary)", loc="left")
    style_axes(axb)

    fig.suptitle(
        "Fig 1. The adaptive margin k*: harmful-prompt refusal after k iteratively re-extracted directions"
        " (only measured points are drawn)",
        x=0.01, ha="left", fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir, "fig1_margin")
    return True


# --------------------------------------------------------------------------- #
# Fig 2: the steering bypass
# --------------------------------------------------------------------------- #
def fig2_dose(cps, adv, outdir):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 5.8))

    # (a) Tiny Aya screens + xref-v2 final
    handles = []
    jitter = {t: (i - 2) * 0.022 for i, t in enumerate(TINY_AYA)}
    for tag in TINY_AYA:
        cp = cps.get(tag)
        if not cp or not cp["screen_dose"]:
            continue
        col = COLOR[tag]
        xs = [d[0] for d in cp["screen_dose"]]
        ys = [d[1] / d[2] for d in cp["screen_dose"]]
        axa.plot(xs, ys, "-", marker="o", ms=4, lw=1.1, color=col, alpha=0.8, mfc="white", zorder=2)
        for sd, c, r, n in cp["screen_rand"]:
            axa.plot([c + jitter[tag]], [r / n], marker="x", ms=7, mew=1.6, color=col, ls="none", zorder=3)
            if not broken(r, n) and r / n < 0.8:
                axa.annotate(f"{SHORT[tag]} random s{sd}: {r}/{n}\n(erodes, does not break)", (c + jitter[tag], r / n),
                             textcoords="offset points", xytext=(-10, 0), fontsize=7.0, color=INK2,
                             ha="right", va="center")
        handles.append(Line2D([], [], color=col, lw=1.1, marker="o", ms=4, mfc="white", label=SHORT[tag]))
    cp = cps.get("xref-v2")
    if cp:
        col = COLOR["xref-v2"]
        c_star = None
        for (cond, s), r in cp["final"].items():
            m = re.fullmatch(r"add-c(\d+(?:\.\d+)?)", cond)
            if s == "B" and m:
                c_star = float(m.group(1))
                axa.plot([c_star], [r["refused"] / r["n"]], "o", ms=9, color=col, zorder=5, mec="white", mew=1.0)
                axa.annotate(
                    f"xref-v2 final, c* = {c_star:g}: {r['refused']}/{r['n']} refused,\n"
                    f"{r['harmND']} judge-flagged harmful (upper bound),\n{r['degenerate']} degenerate, KL {r['kl']:.2f}",
                    (c_star, r["refused"] / r["n"]), textcoords="data", xytext=(1.30, 0.36), fontsize=7.2,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6),
                )
        finals_rand = [r for (cond, s), r in cp["final"].items() if cond.startswith("rand-") and s == "B"]
        for r in finals_rand:
            axa.plot([c_star], [r["refused"] / r["n"]], marker="X", ms=10, color=col, mec="white", mew=0.8,
                     ls="none", zorder=5)
        if finals_rand and c_star is not None:
            lo = min(r["refused"] for r in finals_rand)
            hi = max(r["refused"] for r in finals_rand)
            axa.annotate(
                f"xref-v2 final norm-matched randoms at c*:\n{lo}/150 and {hi}/150 refused (hold)",
                (c_star, hi / 150), textcoords="offset points", xytext=(12, -4), fontsize=7.2, color=INK,
                va="center",
            )
    threshold(axa, x=0.60)
    axa.set_xlim(0.9, 2.12)
    axa.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])
    axa.set_ylim(-0.03, 1.06)
    axa.set_xlabel("dose c (subtract c × own DIM vector at one layer)")
    axa.set_ylabel("WildGuard-refused fraction of harmful prompts (B)")
    axa.set_title("(a) Tiny Aya Global, core model", loc="left")
    style_axes(axa)
    key = [
        Line2D([], [], color="none", label=" "),
        Line2D([], [], color=INK2, lw=1.1, marker="o", ms=4, mfc="white", label="DIM dose curve, screen n = 46"),
        Line2D([], [], color=INK2, marker="x", ms=7, mew=1.6, ls="none", label="random, screen n = 46 (at c = 2.0)"),
        Line2D([], [], color=INK2, marker="o", ms=8, ls="none", label="DIM, final n = 150"),
        Line2D([], [], color=INK2, marker="X", ms=9, ls="none", label="random, final n = 150 (at c*)"),
    ]
    axa.legend(handles=handles + key, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.2, ncol=2,
               columnspacing=1.2)

    # (b) official DeepRefusal
    col = COLOR["dr-official"]
    if adv:
        xs = [d[0] for d in adv["dose"]]
        ys = [d[1] / d[2] for d in adv["dose"]]
        axb.plot(xs, ys, "-", marker="o", ms=4.5, lw=1.2, color=col, alpha=0.8, mfc="white", zorder=2)
        for sd, c, r, n in adv["rand"]:
            axb.plot([c - 0.10 + 0.03 * (sd - 1)], [r / n], marker="x", ms=7, mew=1.6, color=col, ls="none", zorder=3)
    cp = cps.get("dr-official")
    if cp:
        if cp["screen_add_one"]:
            c, r, n = cp["screen_add_one"]
            axb.plot([c], [r / n], marker="s", ms=6, color=col, mfc="white", mew=1.4, ls="none", zorder=3)
            unk = cp.get("screen_add_one_unknown") or 0
            axb.annotate(f"L28 screen (2×2 control): {r}/{n}" + (f",\n{unk}/{n} WildGuard-unknown" if unk else ""),
                         (c, r / n), textcoords="offset points", xytext=(8, -24), fontsize=7.0, color=INK2)
        for (cond, s), r in sorted(cp["final"].items()):
            if s != "B" or not cond.startswith("add-c"):
                continue
            c = float(re.match(r"add-c(\d+(?:\.\d+)?)", cond).group(1))
            lay = re.search(r"-L(\d+)", cond)
            lay = f"L{lay.group(1)}" if lay else ""
            adv_dir = "advisor" in cond
            axb.plot([c], [r["refused"] / r["n"]], "o" if adv_dir else "s", ms=9, color=col, mec="white", mew=1.0,
                     zorder=5)
            unk = r.get("unknown") or 0
            if unk:
                txt = (f"final {lay} (our k1 direction):\n{r['refused']}/{r['n']} refused, {r['degenerate']} degenerate,\n"
                       f"{unk} WildGuard-unknown: {r['refused']}/{r['n'] - unk} of known\n"
                       f"(range {r['refused']}–{r['refused'] + unk}/{r['n']}); no break established")
            else:
                txt = (f"final {lay}{' (sweep direction)' if adv_dir else ''}:\n{r['refused']}/{r['n']} refused, "
                       f"{r['harmND']} judge-flagged\nharmful (upper bound), {r['degenerate']} degenerate")
            axb.annotate(txt, (c, r["refused"] / r["n"]), textcoords="data",
                         xytext=(1.13, 0.06) if adv_dir else (1.13, 0.27), fontsize=7.2, color=INK,
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        rands = [r for (cond, s), r in cp["final"].items() if cond.startswith("rand-") and s == "B"]
        for i, r in enumerate(rands):
            axb.plot([2.07 + 0.03 * i], [r["refused"] / r["n"]], marker="P", ms=9, color=col, mec="white",
                     mew=0.8, ls="none", zorder=5)
        l22r = cp.get("final_l22_rand") or []
        for i, (sd, r, n) in enumerate(l22r):
            axb.plot([1.98 + 0.03 * i], [r / n], marker="X", ms=10, color=col, mec="white", mew=0.8, ls="none",
                     zorder=6)
        if rands or l22r:
            lines = ["final norm-matched randoms (all hold):"]
            if l22r:
                lines.append("matched to L22: " + ", ".join(f"{r}/{n}" for _, r, n in l22r) + " refused")
            if rands:
                lines.append("matched to L28: " + ", ".join(f"{r['refused']}/{r['n']}" for r in rands) + " refused")
            axb.annotate("\n".join(lines), (2.03, 0.965), textcoords="data", xytext=(1.72, 0.76),
                         fontsize=7.2, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    threshold(axb, x=0.60)
    axb.set_xlim(1.1, 2.62)
    axb.set_xticks([1.25, 1.5, 1.75, 2.0, 2.5])
    axb.set_ylim(-0.03, 1.06)
    axb.set_xlabel("dose c (subtract c × own DIM vector at one layer)")
    axb.set_title("(b) official DeepRefusal, Llama-3-8B (supplementary)", loc="left")
    style_axes(axb)
    kb = [
        Line2D([], [], color=col, lw=1.2, marker="o", ms=4.5, mfc="white",
               label="L22 dose curve, screen n = 46 (preliminary sweep)"),
        Line2D([], [], color=col, marker="x", ms=7, mew=1.6, ls="none", label="L22 random, screen n = 46"),
        Line2D([], [], color=col, marker="s", ms=6, mfc="white", mew=1.4, ls="none", label="L28, screen n = 46"),
        Line2D([], [], color=col, marker="o", ms=8, ls="none", label="L22, final n = 150"),
        Line2D([], [], color=col, marker="s", ms=7, ls="none", label="L28, final n = 150"),
        Line2D([], [], color=col, marker="X", ms=9, ls="none", label="random matched to L22, final n = 150"),
        Line2D([], [], color=col, marker="P", ms=9, ls="none", label="random matched to L28, final n = 150"),
    ]
    axb.legend(handles=kb, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.2, ncol=2)

    fig.suptitle(
        "Fig 2. The steering bypass c*: a single-layer additive shift of the model's own direction"
        " vs norm-matched random vectors",
        x=0.01, ha="left", fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir, "fig2_dose")
    return True


# --------------------------------------------------------------------------- #
# Fig 3: repair as margin spend
# --------------------------------------------------------------------------- #
def _bar(ax, x, w, row, color, hatch=None, label_txt=None):
    r, n = row
    ax.bar(x, r / n, w, color=color, hatch=hatch, edgecolor=INK2 if hatch else color, lw=0.6, zorder=3)
    ax.text(x, r / n + 0.012, label_txt or f"{r}", ha="center", va="bottom", fontsize=7.2, color=INK)


def _nr(ax, x, txt="not\nrun"):
    ax.text(x, 0.012, txt, ha="center", va="bottom", fontsize=6.8, color=INK2, style="italic")


def fig3_repair(cps, outdir):
    tags = [t for t in ["ta-base", "t22-dr2e5", "xref-v2", "dr-official"] if cps.get(t) and cps[t]["final"]]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    w = 0.26
    ticks = []
    for i, tag in enumerate(tags):
        fin = cps[tag]["final"]
        a0 = fin.get(("A0", "B"))
        rep = fin.get(("repair", "B"))
        rp1 = fin.get(("repair+1", "B"))
        if a0:
            _bar(ax, i - w, w, (a0["refused"], a0["n"]), C_A0)
        if rep:
            _bar(ax, i, w, (rep["refused"], rep["n"]), C_REPAIR)
        elif tag == "ta-base" and fin.get(("atk-k1", "B")):
            k1 = fin[("atk-k1", "B")]
            _bar(ax, i, w, (k1["refused"], k1["n"]), C_ATTACK, hatch="////", label_txt=f"{k1['refused']}\n(k=1 attack)")
        else:
            _nr(ax, i)
        if rp1:
            _bar(ax, i + w, w, (rp1["refused"], rp1["n"]), C_ATTACK, hatch="////")
        else:
            _nr(ax, i + w)
        # margin after repair, derived
        if rp1:
            mar = "1 (repair+1 breaks)" if broken(rp1["refused"], rp1["n"]) else "≥ 2 (repair+1 holds)"
        elif rep and broken(rep["refused"], rep["n"]):
            mar = "0 (repair alone breaks)"
        else:
            mar = "n/a (undefended)"
        kst = k_star_text(cps[tag]).replace("; ", ";\n")
        ticks.append(f"{SHORT[tag]}{' [suppl.]' if tag in SUPPLEMENTARY else ''}\n{kst}\nmargin after repair:\n{mar}")
    threshold(ax, label=False)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(ticks, fontsize=7.6)
    ax.set_ylim(0, 1.17)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("WildGuard-refused fraction of B (final, n = 150)")
    style_axes(ax)
    leg = [
        Patch(color=C_A0, label="A0 (no attack)"),
        Patch(color=C_REPAIR, label="after repair: ablate the fixed base direction"),
        Patch(facecolor=C_ATTACK, hatch="////", edgecolor=INK2, label="attacked: repair + 1 re-extracted direction\n(base: the k = 1 attack itself)"),
        Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)), label="break: refused < 75 of 150"),
    ]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7.4, ncol=4)
    ax.set_title(
        "Fig 3. Harmful refusal after the standard base-direction repair, and after one more\n"
        "re-extracted direction (bar labels = refused of 150)",
        loc="left", fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, outdir, "fig3_repair")
    return True


# --------------------------------------------------------------------------- #
# Fig 4: the over-refusal bill
# --------------------------------------------------------------------------- #
def fig4_overrefusal(cps, outdir):
    tags = [t for t in ["ta-base", "t22-dr2e5", "t35-dose1", "t35-dose2", "xref-v2", "dr-official"]
            if cps.get(t) and cps[t]["final"].get(("A0", "O"))]
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    w = 0.26
    ticks = []
    for i, tag in enumerate(tags):
        fin = cps[tag]["final"]
        a0 = fin[("A0", "O")]
        _bar(ax, i - w, w, (a0["refused"], a0["n"]), C_A0,
             label_txt=f"{a0['refused']}\n{100 * a0['refused'] / a0['n']:.1f}%")
        rep = fin.get(("repair", "O"))
        if rep:
            _bar(ax, i, w, (rep["refused"], rep["n"]), C_REPAIR,
                 label_txt=f"{rep['refused']}\n{100 * rep['refused'] / rep['n']:.1f}%")
        else:
            _nr(ax, i, "not\nrun" if tag != "ta-base" else "n/a")
        steer = [r for (cond, s), r in fin.items() if s == "O" and cond.startswith("add-c")]
        if steer:
            r = steer[0]
            lay = re.search(r"-L(\d+)", r["condition"])
            _bar(ax, i + w, w, (r["refused"], r["n"]), C_ATTACK, hatch="////",
                 label_txt=f"{r['refused']}\n{100 * r['refused'] / r['n']:.1f}%\n({r['degenerate']} degen.)")
        else:
            _nr(ax, i + w)
        ticks.append(f"{SHORT[tag]}{' [suppl.]' if tag in SUPPLEMENTARY else ''}")
    base = cps.get("ta-base", {}).get("final", {}).get(("A0", "O"))
    base_lbl = None
    if base:
        tiny = [i for i, t in enumerate(tags) if t in TINY_AYA]
        y = base["refused"] / base["n"]
        ax.hlines(y, min(tiny) - 0.45, max(tiny) + 0.45, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=4)
        base_lbl = f"Tiny Aya base A0 level ({100 * y:.1f}%)"
    if "dr-official" in tags:
        i = tags.index("dr-official")
        ax.text(i, 0.93, "own base (Llama-3-8B-Instruct)\nnot measured on O350", ha="center", fontsize=7.0,
                color=INK2, style="italic")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("WildGuard-refused fraction of O (final, n = 350)")
    style_axes(ax)
    leg = [
        Patch(color=C_A0, label="A0: the over-refusal bill"),
        Patch(color=C_REPAIR, label="after base-direction repair"),
        Patch(facecolor=C_ATTACK, hatch="////", edgecolor=INK2, label="after the c* steering attack"),
    ]
    if base_lbl:
        leg.append(Line2D([], [], color=INK2, lw=1.0, ls=(0, (1, 2)), label=base_lbl))
    ax.legend(handles=leg, loc="upper left", fontsize=7.4, ncol=4)
    ax.set_title(
        "Fig 4. The over-refusal bill on benign-but-edgy prompts (O, n = 350) and how it is paid down\n"
        "(DR steering row uses the L28 direction; degenerate outputs counted separately)",
        loc="left", fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, outdir, "fig4_overrefusal")
    return True


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Regenerate the four article PNGs from results/*.csv.")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cps, advisor = load_tables()
    ok = all([fig1_margin(cps, args.outdir), fig2_dose(cps, advisor, args.outdir),
              fig3_repair(cps, args.outdir), fig4_overrefusal(cps, args.outdir)])
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
