"""The backend protocol: everything model-touching the audit needs.

The orchestrator (:mod:`adaptive_margin.benchmark`) drives the audit entirely
through this interface, which is what keeps it pure (and unit-testable with
the synthetic backend).  Vectors are opaque backend-owned objects; only the
backend does vector math.

Two implementations ship:

* :class:`adaptive_margin.fake_backend.SyntheticBackend` , a deterministic,
  dependency-free two-cluster refusal geometry (tests + API demo).
* :class:`adaptive_margin.torch_backend.TorchBackend` , the real one
  (forward hooks, greedy generation, logit capability screen), a faithful
  port of the validated research harness.  **REMOTE ONLY.**
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Dict, List, Optional, Sequence, Tuple

from .types import CapabilityCounts, Generation, Vector


class MarginBackend:
    """Interface contract for adaptive-margin backends."""

    # ------------------------------------------------------------ vector math
    def orthonormalize(self, vectors: Sequence[Vector]) -> List[Vector]:
        """Gram-Schmidt the vectors into an orthonormal list (order kept)."""
        raise NotImplementedError

    def norm(self, v: Vector) -> float:
        raise NotImplementedError

    def unit(self, v: Vector) -> Vector:
        raise NotImplementedError

    def scale(self, v: Vector, s: float) -> Vector:
        """Scalar multiple s * v (callers pass negative s to subtract)."""
        raise NotImplementedError

    def dot(self, a: Vector, b: Vector) -> float:
        raise NotImplementedError

    def random_direction_norm_matched(self, v: Vector, seed: int) -> Vector:
        """A random direction with the same norm as ``v`` (fixed seed)."""
        raise NotImplementedError

    # -------------------------------------------------------- direction access
    def extract_dims(self, basis: Optional[Sequence[Vector]] = None
                     ) -> Dict[int, List[Vector]]:
        """mean(harmful) - mean(harmless) per (layer, position), extracted on
        the *currently edited* model (i.e. with ``basis`` projected out ,
        re-extraction on the edited model is the adaptive part of the attack).
        Returns {layer: [vector per spec.positions index]}."""
        raise NotImplementedError

    def dim_at(self, layer: int, pos: int, basis: Optional[Sequence[Vector]] = None) -> Vector:
        """The DIM at a single (layer, pos) (cheaper than a full extraction)."""
        idx = self.positions.index(pos)
        return self.extract_dims(basis)[layer][idx]

    # ------------------------------------------------------------------ edits
    def edit(self, basis: Optional[Sequence[Vector]] = None,
             add: Optional[Tuple[int, Vector]] = None,
             add_all: Optional[Vector] = None,
             basis_scope: str = "all", basis_layer: Optional[int] = None) -> ContextManager:
        """Install runtime edits (context manager):

        * ``basis`` , project the basis out at every writer (scope "all") or
          only at one layer's writers (scope "layer", ``basis_layer``);
        * ``add`` , (layer, vector): add ``vector`` to one layer's output
          (the single-layer additive dose attack);
        * ``add_all`` , add ``vector`` at every layer (the all-layer control).
        """
        raise NotImplementedError

    # ------------------------------------------------------------ observation
    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> List[Generation]:
        """Greedy generation under the currently active edit."""
        raise NotImplementedError

    def first_logprobs(self, prompts: Sequence[str]) -> object:
        """Next-token log-probs of the prompts' last position (opaque table)."""
        raise NotImplementedError

    def kl(self, ref: object, cur: object) -> float:
        """Mean first-token KL(cur || ref) over the prompt batch."""
        raise NotImplementedError

    def capability(self, mmlu: Sequence[dict], gsm8k: Sequence[dict],
                   basis: Optional[Sequence[Vector]] = None,
                   add: Optional[Tuple[int, Vector]] = None,
                   add_all: Optional[Vector] = None) -> CapabilityCounts:
        """Capability screen under an edit: MMLU letter argmax + GSM8K
        final-number match (greedy generation)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ attrs
    @property
    def positions(self) -> List[int]:
        """The position grid (spec order), used by :meth:`dim_at`."""
        raise NotImplementedError
