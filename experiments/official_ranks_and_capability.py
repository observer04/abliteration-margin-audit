"""Remote-only T39 orchestration. Frozen engines are imported byte-for-byte.

Only this wrapper persists responses (private HF) and observes capability
predictions for paired statistics; it does not modify extraction or inference.
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
import e3a_core as tiny

START = time.time()
OUT = Path('/kaggle/working/t39')
OUT.mkdir(parents=True, exist_ok=True)
HB_SHA = '51182c7cdaf9b2e5f05b745b81b047b67e7384a1'


def log(message):
    print(f'[T39 {time.time()-START:.1f}s] {message}', flush=True)


def save(name, obj):
    data = json.dumps(obj, indent=2, allow_nan=False) + '\n'
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(data)
    # Mandatory durability, not best effort: no silent loss of response bytes.
    core.bh3.upload_text('t39/' + name, data)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def retain(cid, rows, gens):
    records = [dict(prompt_id=r['id'], response=t, n_new_tokens=n,
                    capped=bool(c), response_sha256=sha(t))
               for r, (t, n, c) in zip(rows, gens, strict=True)]
    raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    path = 'responses/' + cid + '.jsonl'
    (OUT / path).parent.mkdir(parents=True, exist_ok=True)
    (OUT / path).write_text(raw)
    core.bh3.upload_text('t39/' + path, raw)
    return dict(path='t39/' + path, sha256=sha(raw), n=len(records),
                response_sha256={r['prompt_id']: r['response_sha256'] for r in records})


def guard(reserve=0):
    # Conservative interpretation: 4 physical T4-hours = 2 session wall-hours.
    assert time.time()-START+reserve < 6800, 'T39 quota/wall guard; no new phase'


def judge_pin_gate():
    from huggingface_hub import HfApi
    current = HfApi(token=os.environ['HF_TOKEN']).model_info(core.HB_MODEL[0]).sha
    assert current == HB_SHA, 'Frozen HB repository changed; STOP, no alternate judge'
    return dict(engine_revision_argument=core.HB_MODEL[1], verified_main_sha=current,
                expected_frozen_team_sha=HB_SHA,
                note='Unchanged E3B uses revision=None for HB. Main is commit-gated before and after judging; historical E3B resolution was not recorded.')


def ranks(P):
    lane, C, F = P['lane'], P['common'], P['frozen']
    cfg = lane['gen']
    eng = core.Engine(dtype_name='fp16', gen_bs=cfg['gen_bs'],
                      lp_bs=cfg['lp_bs'], dim_bs=cfg['dim_bs'], multi_gpu=True)
    eng.load(lane['model']['repo'], lane['model']['revision'])
    # Exact E3B prepare_vectors procedure; the frozen sequence selects no sites.
    D = eng.dims(C['extract_h'], C['extract_b'], list(range(*lane['grid']['layers'])), lane['grid']['pos'])
    vectors, gates, items, records = [], [], [], {}
    for row in F['iterate'][:4]:
        guard(1500)
        if vectors:
            D = eng.dims(C['extract_h'], C['extract_b'], list(range(*lane['grid']['layers'])),
                         lane['grid']['pos'], basis=eng.orth(*vectors))
        v = D[row['layer']][lane['grid']['pos'].index(row['pos'])]
        norm = float(v.norm())
        gate = dict(k=row['k'], layer=row['layer'], pos=row['pos'],
                    recomputed_v_norm=norm, frozen_v_norm=row['v_norm'],
                    relative_difference=abs(norm / row['v_norm']-1), gate_fraction=0.03,
                    vector_sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest())
        assert gate['relative_difference'] <= 0.03, 'Frozen direction gate failed; STOP'
        vectors.append(v)
        gates.append(gate)
    for k in range(1, 5):
        guard(1200)
        cid = f'atk-k{k}'
        spec = dict(id=cid, kind='basis', kmax=k, sets=['B'])
        rec = core.cell_rec(spec)
        rec['gates'] = gates[:k]
        with eng.edit(basis=eng.orth(*vectors[:k])):
            gens = eng.gen([r['prompt'] for r in C['B']], cfg['B_max_new'])
        rec['private_responses'] = retain('dr-'+cid, C['B'], gens)
        core.gen_counts(rec, 'B', gens, cfg['B_max_new'])
        for r, (t, n, capped) in zip(C['B'], gens, strict=True):
            deg = bool(core.bh3.degenerate(t))
            rec['B']['per_prompt'][r['id']] = [None, None, deg, bool(capped), None]
            items.append(dict(cell=cid, id=r['id'], prompt=r['prompt'], resp=t))
        rec['generation_batch_final'] = eng.gen_bs
        rec['status'] = 'generated'
        records[cid] = rec
        save('ranks/'+cid+'.json', rec)
        log(f'generated {cid}: n={len(gens)}')
    eng.free()
    pins_before = judge_pin_gate()

    def apply(kind):
        def callback(start, labels):
            for it, label in zip(items[start:start+len(labels)], labels, strict=True):
                pp = records[it['cell']]['B']['per_prompt'][it['id']]
                if kind == 'wg':
                    pp[:2] = [label['refusal'], label['harmful']]
                else:
                    pp[4] = label['harmful']
        return callback

    _, wg_where = core.judge_wg(items, batch=lane['judge']['wg_batch'], on_slice=apply('wg'), log=log)
    for rec in records.values():
        rec['B']['wg_status'] = 'complete'
        rec['judges']['wg'] = 'complete'
        core.finalize_counts(rec)
        save('ranks/'+rec['condition']+'.json', rec)
    log('WildGuard ranks complete')
    _, hb_where = core.judge_hb(items, batch=lane['judge']['hb_batch'], on_slice=apply('hb'), log=log)
    pins_after = judge_pin_gate()
    for rec in records.values():
        rec['B']['hb_status'] = 'complete'
        rec['judges']['hb'] = 'complete'
        rec['status'] = 'complete'
        core.finalize_counts(rec)
        save('ranks/'+rec['condition']+'.json', rec)
    summary = dict(status='complete', gates=gates, n=600, elapsed_s=time.time()-START,
                   judge_pins_before=pins_before, judge_pins_after=pins_after,
                   judge_placement=dict(wg=wg_where, hb=hb_where),
                   counts={cid: {k:v for k,v in r['B'].items() if k!='per_prompt'} for cid,r in records.items()})
    save('rank_summary.json', summary)
    log('ITEM1_COMPLETE: rank_summary.json durably retained privately')


def capability_observed(eng, cid, mm, gs, add, mmlu_bs, cap_bs):
    """Call frozen E3A capability verbatim; hooks only observe its outputs."""
    eng.torch = torch
    letters = [eng.tok.encode(x, add_special_tokens=False)[0] for x in 'ABCD']
    if cid.startswith('dr-'):
        assert letters == [eng.tok.convert_tokens_to_ids(x) for x in 'ABCD']
    predictions, generated = [], []
    original_gen = eng.gen

    def observe_model(module, args, output):
        if len(predictions) < len(mm):
            predictions.extend(output.logits[:, -1, :].float()[:, letters].argmax(-1).tolist())
        # A forward hook returning None leaves all original values unchanged.

    def observe_gen(*args, **kwargs):
        result = original_gen(*args, **kwargs)
        generated.extend(result)
        return result

    handle = eng.model.register_forward_hook(observe_model)
    eng.gen = observe_gen
    try:
        before = time.time()
        counts = tiny.Runner.capability(eng, mm, gs, add=add, mmlu_bs=mmlu_bs, cap_bs=cap_bs)
    finally:
        eng.gen = original_gen
        handle.remove()
    assert len(predictions) == len(mm) and len(generated) == len(gs)
    mm_ok = {r['id']: 'ABCD'[p] == r['gold'] for r,p in zip(mm,predictions,strict=True)}
    gs_ok = {r['id']: tiny.gsm_ok(t,r['gold']) for r,(t,_,_) in zip(gs,generated,strict=True)}
    assert counts['mmlu_correct']==sum(mm_ok.values())
    assert counts['gsm8k_correct']==sum(gs_ok.values())
    rec = dict(condition=cid, counts=counts, mmlu=mm_ok, gsm8k=gs_ok,
               mmlu_predictions={r['id']:'ABCD'[p] for r,p in zip(mm,predictions,strict=True)},
               private_responses=retain(cid, gs, generated),
               protocol=dict(mmlu_bs=mmlu_bs, cap_bs=cap_bs, gsm_max_new=320,
                             do_sample=False, dtype='fp16', mmlu='four-letter final-token argmax'),
               elapsed_s=time.time()-before)
    save('capability/'+cid+'.json',rec)
    log(f'capability {cid}: {counts}')
    return rec


def official_cap(P):
    guard(2200)
    C, lane = P['common'], P['lane']
    eng = core.Engine(dtype_name='fp16', gen_bs=16, lp_bs=12, dim_bs=8, multi_gpu=True)
    eng.load(lane['model']['repo'], lane['model']['revision'])
    D=eng.dims(C['extract_h'], C['extract_b'], list(range(6,30,2)), [-1,-3,-5])
    v=D[22][1].float().cpu()
    expected=P['t38_L22_norm']
    assert float(v.norm()) == expected, 'T38 exact vector norm gate failed; STOP'
    assert hashlib.sha256(v.numpy().tobytes()).hexdigest()==P['t38_L22_sha'], 'T38 vector hash differs; STOP'
    g=torch.Generator().manual_seed(101)
    rv=torch.randn(v.shape[0],generator=g)
    rv=rv/rv.norm()*v.norm()
    records={}
    for cid,add in [('dr-A0',None),('dr-L22-DIM-c2.0',(22,-2*v)),('dr-L22-rand-s101-c2.0',(22,-2*rv))]:
        guard(650)
        records[cid]=capability_observed(eng,cid,C['mmlu'][:200],C['gsm8k'][:100],add,16,50)
    eng.free()
    save('official_capability_summary.json',dict(status='complete',counts={k:v['counts'] for k,v in records.items()},
                                               note='Advisor subsets and scoring; E3B fp16/T4 engine and adaptive generation OOM fallback. Per-item predictions observed without altering outputs.'))


def xref_cap(P):
    guard(1500)
    C=P['common']
    runner=tiny.Runner(P['xref']['repo'],P['xref']['revision'],log=log)
    v=runner.dim_at(24,-1,bs=12,ex_h=C['extract_h'],ex_b=C['extract_b']).float().cpu()
    target=P['xref']['v_norm']
    assert abs(float(v.norm())/target-1)<=0.03, 'xref frozen direction gate failed; STOP'
    output={}
    for cid,add in [('xref-A0',None),('xref-add-c1.25',(24,-1.25*v))]:
        for rank in range(2):
            guard(450)
            key=cid+f'-partition{rank}'
            output[key]=capability_observed(runner,key,C['mmlu'][rank::2],C['gsm8k'][rank::2],add,8,10)
    runner.close()
    save('xref_capability_summary.json',dict(status='complete',counts={k:v['counts'] for k,v in output.items()},
                                           direction_norm=float(v.norm()), frozen_norm=target,
                                           note='Frozen E3A Runner, parity partitions, bs8/10, MMLU500/GSM250. T4 fp16 versus original Modal bf16 is disclosed.'))


def main():
    P=json.loads(Path('payload.json').read_text())
    hardware=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    assert len(hardware)==2 and all('T4' in x for x in hardware), 'Frozen T4x2 required'
    from importlib.metadata import version
    save('runtime.json',dict(hardware=hardware,python=platform.python_version(),torch=torch.__version__,
                            cuda=torch.version.cuda,packages={n:version(n) for n in ('transformers','accelerate','huggingface-hub')},
                            source_hashes=P['source_hashes'],provenance=P['provenance']))
    ranks(P)
    official_cap(P)
    if time.time()-START < 4900:
        xref_cap(P)
    else:
        save('xref_capability_summary.json',dict(status='dropped_quota_guard',elapsed_s=time.time()-START))
    save('final.json',dict(status='complete',elapsed_s=time.time()-START,physical_gpu_h=2*(time.time()-START)/3600,
                           cash_usd=0,modal_usd=0))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # Exception messages contain no input/response text; full trace is remote.
        save('failure.json',dict(status='stopped',type=type(exc).__name__,message=str(exc),elapsed_s=time.time()-START))
        traceback.print_exc()
        raise
