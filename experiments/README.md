# Archived GPU experiments

These files preserve the as-run GPU compute code. They are not a turnkey pipeline. They need the original payloads, GPU hosts, path layouts and harness wiring. The dataset loader rebuilds prompt text, but does not rebuild the full experiment payloads or capability gold labels. Generated text and per-prompt judgments are withheld.

- `ladder.py` holds the adaptive rank ladder compute bodies. Launch plumbing was removed.
- `e3a_core.py` and `e3b_core.py` hold the Tiny Aya and official DeepRefusal final-panel compute code.
- `bh3_core.py` provides shared generation, editing and judging helpers.
- `official_l22_random_controls.py` runs the matched official L22 random controls and the L28 reproduction.
- `official_ranks_and_capability.py` completes the official rank ladder and measures steering capability.
- `eval/` preserves the original judge, metric and statistics helpers. Its original local harness imports are not all included.

On a remote GPU host, install the dependencies from the repository root:

```bash
python -m pip install -r requirements-gpu.txt
```

Installing these dependencies does not restore the original payloads or harness wiring. For figures and count checks, use the light quickstart in the root README.
