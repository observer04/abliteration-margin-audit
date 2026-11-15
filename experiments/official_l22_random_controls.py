"""T38 remote-only E3B reproduction and matched L22 controls.

Imports the archived E3B engine and judges unchanged. All generation text stays
in memory; exported records contain IDs, labels, hashes, counts and norms only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import torch
import e3b_core as core

START = time.time()
OUT = Path(os.environ.get("T38_OUT", "/kaggle/working/t38"))
OUT.mkdir(parents=True, exist_ok=True)


def log(message):
    print(f"[T38 {time.time()-START:.1f}s] {message}", flush=True)


def save(name, obj, upload=True):
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    (OUT / name).write_text(text)
    if upload:
        try:
            core.bh3.upload_text("t38/" + name, text)
        except Exception as exc:
            log(f"upload {name} failed: {type(exc).__name__}; Kaggle copy retained")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def counts(rows):
    n = len(rows)
    return {
        "n": n,
        "wg_refused": sum(x[0] == "yes" for x in rows.values()),
        "wg_harmful": sum(x[1] == "yes" for x in rows.values()),
        "wg_harmful_nondegenerate": sum(x[1] == "yes" and not x[2] for x in rows.values()),
        "wg_refusal_unknown": sum(x[0] is None for x in rows.values()),
        "wg_harmful_unknown": sum(x[1] is None for x in rows.values()),
        "wg_either_unknown": sum(x[0] is None or x[1] is None for x in rows.values()),
        "degenerate": sum(x[2] for x in rows.values()),
        "capped": sum(x[3] for x in rows.values()),
        "hb_harmful": sum(x[4] is True for x in rows.values()),
        "hb_unknown": sum(x[4] is None for x in rows.values()),
    }


def main():
    P = json.loads(Path("payload.json").read_text())
    lane = P["lane"]
    C = P["common"]
    originals = P["originals"]
    cfg = lane["gen"]
    assert torch.cuda.device_count() == 2, "T38 requires frozen T4x2 split"
    hardware = [torch.cuda.get_device_name(i) for i in range(2)]
    assert all("T4" in name for name in hardware), hardware
    from importlib.metadata import version
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "cuda": torch.version.cuda,
               **{name: version(name) for name in ("transformers", "accelerate", "huggingface-hub")}}
    engine = core.Engine(dtype_name="fp16", gen_bs=cfg["gen_bs"],
                         lp_bs=cfg["lp_bs"], dim_bs=cfg["dim_bs"], multi_gpu=True)
    engine.load(lane["model"]["repo"], lane["model"]["revision"])
    # Same extraction prompts, fp16 forwards, dim batch, positions and hooks as
    # E3B. No layer/position/dose selection and no test-split access.
    directions = engine.dims(C["extract_h"], C["extract_b"],
                             list(range(6, 30, 2)), [-1, -3, -5])
    targets = {
        22: originals["L22"]["gates"]["advisor_L22_p-3"]["recomputed_v_norm"],
        28: originals["L28"]["gates"]["k1_L28_p-3"]["recomputed_v_norm"],
    }
    vectors = {}
    gates = {}
    for layer in (22, 28):
        vector = directions[layer][1].float().cpu()
        norm = float(vector.norm())
        target = targets[layer]
        rel = abs(norm / target - 1)
        assert rel <= 0.03, f"L{layer} frozen E3B norm gate failed"
        # Preserve the *original applied* norm exactly, rather than silently
        # matching the 0.16%-different advisor-screen norm or the L28 norm.
        vectors[layer] = vector if norm == target else vector * (target / norm)
        gates[str(layer)] = {"recomputed_dim_norm": norm,
                             "original_e3b_dim_norm": target,
                             "relative_difference": rel, "gate_fraction": 0.03,
                             "pass": True,
                             "rescaled_to_original_norm": norm != target,
                             "applied_shift_norm": float((-2 * vectors[layer]).norm()),
                             "vector_sha256": hashlib.sha256(vectors[layer].numpy().tobytes()).hexdigest()}
    specs = [
        {"id": "L22-DIM-c2.0-reproduction", "layer": 22, "kind": "dim"},
        {"id": "L22-rand-s101-c2.0", "layer": 22, "kind": "rand", "seed": 101},
        {"id": "L22-rand-s202-c2.0", "layer": 22, "kind": "rand", "seed": 202},
        {"id": "L28-DIM-c2.0-reproduction", "layer": 28, "kind": "dim"},
    ]
    records, items = {}, []
    for spec in specs:
        assert time.time() - START < 5000, "T38 generation wall guard"
        cid = spec["id"]
        vector = vectors[spec["layer"]]
        if spec["kind"] == "rand":
            generator = torch.Generator().manual_seed(spec["seed"])
            random_vector = torch.randn(vector.shape[0], generator=generator)
            vector = random_vector / random_vector.norm() * vector.norm()
        applied = -2.0 * vector
        rec = {"condition": cid, "spec": spec, "status": "generating",
               "shift_norm": float(applied.norm()), "gates": gates,
               "shift_sha256": hashlib.sha256(applied.numpy().tobytes()).hexdigest(),
               "per_prompt": {}, "response_sha256": {}, "judge_passes": {}}
        before = time.time()
        with engine.edit(add=(spec["layer"], applied)):
            gens = engine.gen([r["prompt"] for r in C["B"]], cfg["B_max_new"])
        for row, (response, tokens, capped) in zip(C["B"], gens, strict=True):
            pid = row["id"]
            deg = core.bh3.degenerate(response)
            rec["per_prompt"][pid] = [None, None, deg, bool(capped), None]
            rec["response_sha256"][pid] = sha(response)
            items.append({"cell": cid, "id": pid, "prompt": row["prompt"],
                          "resp": response, "deg": deg, "cap": bool(capped)})
        rec["generation_s"] = time.time() - before
        rec["generation_batch_final"] = engine.gen_bs
        rec["status"] = "generated"
        records[cid] = rec
        save(cid + ".json", rec)
        log(f"generated {cid}: n={len(gens)} shift_norm={rec['shift_norm']:.9f}")
        del gens
    engine.free()
    assert len(items) == 600
    save("generation_manifest.json", {
        "n": len(items), "hardware": hardware, "runtime": runtime, "gates": gates,
        "model": lane["model"], "provenance": P["provenance"],
        "original_text_retained": False, "original_text_identity_verifiable": False,
        "note": "Frozen original generations/hashes were never retained. DIM rows are faithful reproductions, never silent replacements for frozen E3B labels.",
    })

    def wg_pass(name, selected):
        saved = {}

        def on_slice(start, labels):
            for item, label in zip(selected[start:start+len(labels)], labels, strict=True):
                saved.setdefault(item["cell"], {})[item["id"]] = [label["refusal"], label["harmful"]]

        _, where = core.judge_wg(selected, batch=lane["judge"]["wg_batch"],
                                 on_slice=on_slice, log=log)
        for cid, pp in saved.items():
            records[cid]["judge_passes"][name] = pp
        save("wg-" + name + ".json", {"n": len(selected), "where": where, "labels": saved,
                                      "max_new_tokens": 32, "do_sample": False,
                                      "batch": lane["judge"]["wg_batch"]})
        log(f"WG {name} complete: n={len(selected)}")
        return saved

    first = wg_pass("pass1", items)
    for cid, pp in first.items():
        for pid, values in pp.items():
            records[cid]["per_prompt"][pid][0:2] = values
    for cid, rec in records.items():
        rec["counts"] = counts(rec["per_prompt"])
        save(cid + ".json", rec)

    # Exactly the SAME generated strings are rejudged; no regeneration between
    # passes. Audit every L22 item, all 55 original L28-unknown prompt IDs, plus
    # any new L28 unknowns. Keep pass1 primary and pass2 as a diagnostic.
    old_unknown = {pid for pid, v in originals["L28"]["B"]["per_prompt"].items()
                   if v[0] is None or v[1] is None}
    assert len(old_unknown) == 55
    second_items = [it for it in items if it["cell"].startswith("L22-")
                    or it["id"] in old_unknown
                    or any(v is None for v in first[it["cell"]][it["id"]])]
    second = wg_pass("pass2", second_items)
    # Repeat the L28-unknown item order separately at the same settings to
    # measure whether identical deterministic retries resolve parser failures.
    l28_unknown_items = [it for it in items if it["cell"].startswith("L28-")
                         and it["id"] in old_unknown]
    third = wg_pass("l28-original55-repeat", l28_unknown_items)
    hb_items = [it for it in items if it["cell"].startswith("L22-")]

    def hb_slice(start, labels):
        for it, label in zip(hb_items[start:start+len(labels)], labels, strict=True):
            records[it["cell"]]["per_prompt"][it["id"]][4] = label["harmful"]

    _, hb_where = core.judge_hb(hb_items, batch=lane["judge"]["hb_batch"],
                               on_slice=hb_slice, log=log)
    log(f"HarmBench complete: n={len(hb_items)}")
    for cid, rec in records.items():
        rec["counts"] = counts(rec["per_prompt"])
        rec["status"] = "complete"
        rec["hb_status"] = "complete" if cid.startswith("L22-") else "not_requested"
        rec["hb_where"] = hb_where if cid.startswith("L22-") else None
        save(cid + ".json", rec)
    unknown_audit = {}
    for cid in records:
        pp1 = first[cid]
        pp2 = second.get(cid, {})
        common = sorted(set(pp1) & set(pp2))
        unknown_audit[cid] = {
            "pass1_unknown_refusal": sum(v[0] is None for v in pp1.values()),
            "pass1_unknown_either": sum(any(x is None for x in v) for v in pp1.values()),
            "pass2_n": len(pp2),
            "pass2_unknown_refusal": sum(v[0] is None for v in pp2.values()),
            "pass2_unknown_either": sum(any(x is None for x in v) for v in pp2.values()),
            "pass1_vs_pass2_label_changes": sum(pp1[pid] != pp2[pid] for pid in common),
            "pass1_unknown_resolved_on_pass2": sum(any(x is None for x in pp1[pid]) and all(x is not None for x in pp2[pid]) for pid in common),
        }
    cid = "L28-DIM-c2.0-reproduction"
    old = originals["L28"]["B"]["per_prompt"]
    subset = {pid: second[cid][pid] for pid in old_unknown}
    repeated = third[cid]
    unknown_audit["L28_original55"] = {
        "n": 55, "prompt_ids": sorted(old_unknown),
        "first_unknown_either": sum(any(x is None for x in first[cid][pid]) for pid in old_unknown),
        "first_to_rejudge_newly_resolved": sum(any(x is None for x in first[cid][pid]) and all(x is not None for x in subset[pid]) for pid in old_unknown),
        "first_to_rejudge_new_unknown": sum(all(x is not None for x in first[cid][pid]) and any(x is None for x in subset[pid]) for pid in old_unknown),
        "rejudge_unknown_either": sum(any(x is None for x in v) for v in subset.values()),
        "repeat_unknown_either": sum(any(x is None for x in v) for v in repeated.values()),
        "rejudge_resolved": sum(all(x is not None for x in v) for v in subset.values()),
        "repeat_resolved": sum(all(x is not None for x in v) for v in repeated.values()),
        "rejudge_vs_repeat_label_changes": sum(subset[pid] != repeated[pid] for pid in old_unknown),
        "original_text_identity_verifiable": False,
    }
    for tag, condition in [("L22", "L22-DIM-c2.0-reproduction"), ("L28", cid)]:
        original = originals[tag]["B"]["per_prompt"]
        current = records[condition]["per_prompt"]
        unknown_audit[tag + "_frozen_vs_reproduction"] = {
            "n": len(original),
            "wg_refusal_label_changes": sum(original[p][0] != current[p][0] for p in original),
            "wg_harmful_label_changes": sum(original[p][1] != current[p][1] for p in original),
            "degenerate_flag_changes": sum(original[p][2] != current[p][2] for p in original),
            "cap_flag_changes": sum(original[p][3] != current[p][3] for p in original),
        }
    save("unknown_audit.json", unknown_audit)
    elapsed = time.time() - START
    summary = {"status": "complete", "hardware": hardware, "runtime": runtime, "wall_s": elapsed,
               "gpu_h": 2 * elapsed / 3600, "cash_usd": 0,
               "wg_model": core.WG, "hb_model": core.HB_MODEL,
               "conditions": {cid: rec["counts"] for cid, rec in records.items()},
               "gates": gates, "unknown_audit": unknown_audit,
               "provenance": P["provenance"], "engine_notes": engine.notes,
               "generation_calls": 4, "generated_rows": 600,
               "judge_rows": {"wg": len(items) + len(second_items) + 55, "hb": len(hb_items)}}
    save("summary.json", summary)
    log(f"COMPLETE wall_s={elapsed:.1f} gpu_h={summary['gpu_h']:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Counts only even when a model/library error embeds input text.
        save("failure.json", {"error_type": type(exc).__name__,
                              "wall_s": time.time()-START,
                              "traceback_locations": [{"file": f.filename, "line": f.lineno, "function": f.name}
                                                      for f in traceback.extract_tb(exc.__traceback__)]})
        log(f"FAILED {type(exc).__name__}; counts-only failure receipt saved")
        raise SystemExit(1)
