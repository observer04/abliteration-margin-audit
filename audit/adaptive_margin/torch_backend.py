"""The real backend: torch + transformers.  **REMOTE ONLY , never import on
the local (no-GPU) machine.**

A faithful port of the validated research harness (``lab/code/ladder.py``,
phases 1/2 payload bodies) onto the
:class:`~adaptive_margin.backend.MarginBackend` protocol:

* ``extract_dims`` , forward hooks capturing the residual stream at the
  prompt's ``positions`` (default (-1, -2, -4)) on every layer in the grid,
  mean(harmful) - mean(harmless) per (layer, position), extracted *with* the
  accumulated basis projected out (re-extraction on the edited model).
* ``edit`` , the harness's ``Edit`` context: basis projection at
  embed/every attn+mlp writer (scope "all") or one layer's writers
  (scope "layer"); single-layer and all-layer additive shifts.
* ``generate`` , greedy, harness EOS handling, capped counts.
* ``first_logprobs`` / ``kl`` , first-token log-softmax, mean KL vs the
  unedited reference on harmless prompts (the collateral diagnostic).
* ``capability`` , MMLU A/B/C/D letter argmax + GSM8K final-number match.

Mapping to the harness (for review): Edit -> Edit; dims -> dims; dim_at ->
dim_at; first_lp/kl -> first_lp/kl; gen -> gen; capability -> capability.

Everything here needs the model on a GPU (bf16, device 0).  The tests in
``tests/test_torch_backend_remote.py`` are marked ``remote`` and excluded
from the default run; run them on Kaggle / HF Jobs (see README).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch  # remote only , this import is why the module is excluded locally

from .backend import MarginBackend
from .textutils import gsm_ok
from .types import CapabilityCounts, Generation, Vector


class TorchBackend(MarginBackend):
    """Model access for :class:`~adaptive_margin.benchmark.AdaptiveMarginBenchmark`.

    Args:
        model: an HF causal LM (bf16, on device 0, ``.eval()``).
        tokenizer: its tokenizer (``padding_side="left"``).
        spec: the :class:`~adaptive_margin.config.MarginSpec` grid.
        extract_h / extract_b: contrastive prompt lists for DIM extraction.
    """

    def __init__(self, model, tokenizer, spec, extract_h: Sequence[str],
                 extract_b: Sequence[str]) -> None:
        self.model = model
        self.tok = tokenizer
        self.spec = spec
        self.extract_h = list(extract_h)
        self.extract_b = list(extract_b)
        eos = model.generation_config.eos_token_id
        self.eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        for extra in (tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids("<|eot_id|>")
                      if "<|eot_id|>" in tokenizer.get_vocab() else None,
                      tokenizer.convert_tokens_to_ids("<|im_end|>")
                      if "<|im_end|>" in tokenizer.get_vocab() else None):
            if extra is not None and extra not in self.eos:
                self.eos.append(extra)

    # ------------------------------------------------------------ vector math
    def norm(self, v: Vector) -> float:
        return float(v.float().norm())

    def dot(self, a: Vector, b: Vector) -> float:
        return float(a.float() @ b.float())

    def scale(self, v: Vector, s: float) -> Vector:
        return v.float() * s

    def unit(self, v: Vector) -> Vector:
        return v.float() / v.float().norm()

    def orthonormalize(self, vectors: Sequence[Vector]) -> List[Vector]:
        Q = []
        for v in vectors:
            w = v.float().clone()
            for q in Q:
                w = w - (w @ q) * q
            n = w.norm()
            if n > 1e-12:
                Q.append(w / n)
        return Q

    def random_direction_norm_matched(self, v: Vector, seed: int) -> Vector:
        g = torch.Generator().manual_seed(seed)
        rv = torch.randn(v.shape[0], generator=g).to(v.device, v.dtype)
        return rv / rv.norm() * v.norm()

    @property
    def positions(self) -> List[int]:
        return list(self.spec.positions)

    # --------------------------------------------------------------- plumbing
    def _enc(self, prompts: Sequence[str]):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                                              tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        return self.tok(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to("cuda")

    @contextmanager
    def edit(self, basis: Optional[Sequence[Vector]] = None,
             add: Optional[Tuple[int, Vector]] = None,
             add_all: Optional[Vector] = None,
             basis_scope: str = "all", basis_layer: Optional[int] = None):
        handles = []
        B = None if basis is None else torch.stack(list(basis)).to("cuda", torch.float32)

        def proj(h):
            hf = h.float()
            return (hf - (hf @ B.T) @ B).to(h.dtype)

        def tup(mod, a, o):
            return (proj(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else proj(o)

        m = self.model.model
        try:
            if B is not None:
                if basis_scope == "all":
                    handles.append(m.embed_tokens.register_forward_hook(
                        lambda mod, a, o: proj(o)))
                    for layer in m.layers:
                        handles.append(layer.self_attn.register_forward_hook(tup))
                        handles.append(layer.mlp.register_forward_hook(tup))
                elif basis_scope == "layer":
                    layer = m.layers[basis_layer]
                    handles.append(layer.self_attn.register_forward_hook(tup))
                    handles.append(layer.mlp.register_forward_hook(tup))
                else:
                    raise ValueError(basis_scope)
            if add is not None:
                li, vec = add
                v = vec.to("cuda", torch.float32)

                def addh(mod, a, o, v=v):
                    if isinstance(o, tuple):
                        return ((o[0].float() + v).to(o[0].dtype),) + tuple(o[1:])
                    return (o.float() + v).to(o.dtype)

                handles.append(m.layers[li].register_forward_hook(addh))
            if add_all is not None:
                v = add_all.to("cuda", torch.float32)

                def addh_all(mod, a, o, v=v):
                    if isinstance(o, tuple):
                        return ((o[0].float() + v).to(o[0].dtype),) + tuple(o[1:])
                    return (o.float() + v).to(o.dtype)

                for layer in m.layers:
                    handles.append(layer.register_forward_hook(addh_all))
            yield
        finally:
            for h in handles:
                h.remove()
            handles.clear()

    # -------------------------------------------------------- direction access
    @torch.no_grad()
    def extract_dims(self, basis: Optional[Sequence[Vector]] = None,
                     bs: int = 16) -> Dict[int, List[Vector]]:
        layers = self.spec.layers
        pos = list(self.spec.positions)

        def cap(prompts):
            store = {i: [] for i in layers}
            hs = []

            def hook(mod, a, o, i=i):
                h = o[0] if isinstance(o, tuple) else o
                store[i].append(h[:, pos, :].float().cpu())

            for i in layers:
                hs.append(self.model.model.layers[i].register_forward_hook(hook))
            try:
                with self.edit(basis=basis):
                    for s in range(0, len(prompts), bs):
                        self.model(**self._enc(prompts[s:s + bs]))
            finally:
                for h in hs:
                    h.remove()
            return {i: torch.cat(v) for i, v in store.items()}

        a = cap(self.extract_h)
        b = cap(self.extract_b)
        return {i: a[i].mean(0) - b[i].mean(0) for i in a}

    @torch.no_grad()
    def dim_at(self, layer: int, pos: int, basis: Optional[Sequence[Vector]] = None,
               bs: int = 16) -> Vector:
        store: List[torch.Tensor] = []

        def hook(mod, a, o):
            h = o[0] if isinstance(o, tuple) else o
            store.append(h[:, pos, :].float().cpu())

        hs = [self.model.model.layers[layer].register_forward_hook(hook)]
        try:
            with self.edit(basis=basis):
                for prompts in (self.extract_h, self.extract_b):
                    for s in range(0, len(prompts), bs):
                        self.model(**self._enc(prompts[s:s + bs]))
        finally:
            for h in hs:
                h.remove()
        ah = torch.cat(store[: len(self.extract_h)]).mean(0)
        bh = torch.cat(store[len(self.extract_h):]).mean(0)
        return ah - bh

    # ------------------------------------------------------------ observation
    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int,
                 bs: int = 24) -> List[Generation]:
        out: List[Generation] = []
        for s in range(0, len(prompts), bs):
            e = self._enc(prompts[s:s + bs])
            g = self.model.generate(**e, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.eos)
            for row in g[:, e["input_ids"].shape[1]:]:
                ids = row.tolist()
                n = len(ids)
                for j, t in enumerate(ids):
                    if t in self.eos or t == self.tok.pad_token_id:
                        n = j
                        break
                out.append(Generation(text=self.tok.decode(ids[:n], skip_special_tokens=True),
                                      n_tokens=n, capped=n >= max_new_tokens))
        return out

    @torch.no_grad()
    def first_logprobs(self, prompts: Sequence[str], bs: int = 16):
        parts = []
        for s in range(0, len(prompts), bs):
            parts.append(torch.log_softmax(
                self.model(**self._enc(prompts[s:s + bs]), logits_to_keep=1)
                .logits[:, -1, :].float(), -1).cpu())
        return torch.cat(parts)

    def kl(self, ref, cur) -> float:
        return float((ref.exp() * (ref - cur)).sum(-1).mean())

    @torch.no_grad()
    def capability(self, mmlu, gsm8k, basis=None, add=None, add_all=None,
                   bs: int = 8) -> CapabilityCounts:
        letters = [self.tok.encode(x, add_special_tokens=False)[0] for x in ["A", "B", "C", "D"]]
        correct = 0
        with self.edit(basis=basis, add=add, add_all=add_all):
            for s in range(0, len(mmlu), bs):
                lg = self.model(**self._enc([r["prompt"] for r in mmlu[s:s + bs]]),
                                logits_to_keep=1).logits[:, -1, :].float()
                pred = lg[:, letters].argmax(-1).tolist()
                correct += sum("ABCD"[p] == r["gold"] for p, r in zip(pred, mmlu[s:s + bs]))
            torch.cuda.empty_cache()
            gens = self.generate([r["prompt"] for r in gsm8k],
                                 self.spec.gsm8k_max_new_tokens, bs=25)
        return CapabilityCounts(
            mmlu_correct=correct, mmlu_n=len(mmlu),
            gsm8k_correct=sum(gsm_ok(g.text, r["gold"]) for g, r in zip(gens, gsm8k)),
            gsm8k_n=len(gsm8k), gsm8k_capped=sum(g.capped for g in gens))
