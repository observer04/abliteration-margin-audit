"""Data types for the adaptive-margin audit.

Everything in this module is plain data: no torch, no model, no I/O.  The
orchestrator (:mod:`adaptive_margin.benchmark`) and the accounting
(:mod:`adaptive_margin.accounting`) speak only in these types, which is what
makes the selection / accounting logic unit-testable without a GPU.

Vector types are intentionally opaque (``Vector = Any``): the synthetic
backend uses tuples of floats, the torch backend uses tensors.  Only the
backend is allowed to do math on vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

#: Opaque vector type owned by the backend (tuple for the fake, tensor for torch).
Vector = Any


@dataclass
class Generation:
    """One greedy completion, as produced by a backend."""

    text: str
    n_tokens: int
    capped: bool


@dataclass
class Labels:
    """One judge verdict over a (prompt, response) pair."""

    response_refusal: Optional[bool]
    harmful_response: Optional[bool]
    source: str = "unknown"


@dataclass
class SplitCounts:
    """Counted outcomes of one condition on one prompt set (split).

    ``marker_refusals`` is the cheap lexical diagnostic (regex, always
    available); ``judge_*`` fields are the semantic counts and are ``None``
    until a judge has labelled the generations (or when loading count-only
    artifacts such as the frozen CSV).  ``hb_harmful`` is the HarmBench-cls
    harm count (the D5 endpoint's harm judge); it is populated from the
    frozen CSV and stays ``None`` unless a second harm judge is run.
    """

    n: int = 0
    marker_refusals: int = 0
    judge_refusals: Optional[int] = None
    judge_harmful: Optional[int] = None
    judge_harmful_nondegenerate: Optional[int] = None
    hb_harmful: Optional[int] = None
    capped: int = 0
    degenerate: int = 0
    mean_tokens: float = 0.0
    unknown_verdicts: int = 0


@dataclass
class CapabilityCounts:
    """Capability screen results (MMLU letter-argmax + GSM8K final-number)."""

    mmlu_correct: int = 0
    mmlu_n: int = 0
    gsm8k_correct: int = 0
    gsm8k_n: int = 0
    gsm8k_capped: int = 0


@dataclass
class ConditionRecord:
    """Everything measured for one named condition (A0, s0-k1, dose-c1.5, ...)."""

    name: str
    counts: Dict[str, SplitCounts] = field(default_factory=dict)
    kl_harmless: float = 0.0
    capability: Optional[CapabilityCounts] = None


@dataclass
class SelectionRow:
    """One candidate (layer, pos) direction as scored by the selection screen."""

    layer: int
    pos: int
    marker_n: int
    marker_rate: float
    kl: float
    v_norm: float


@dataclass
class LadderStep:
    """One step k of an iterated re-extraction ladder."""

    k: int
    layer: int
    pos: int
    sel_kl: float
    sel_marker_rate: float
    v_norm: float
    condition: str
    marker_n: int
    marker_rate: float
    kl_harmless: float


@dataclass
class DosePoint:
    """One point of the single-layer additive dose curve (or a random control)."""

    c: float
    condition: str
    kind: str = "dose"  # "dose" | "random_norm_matched"
    seed: Optional[int] = None
    marker_n: int = 0
    marker_rate: float = 0.0
    kl_harmless: float = 0.0
    capped: int = 0
    degenerate: int = 0


@dataclass
class PromptItem:
    """An eval prompt with a stable id (B and O sets)."""

    id: str
    prompt: str


@dataclass
class PromptSets:
    """The prompt bundles the audit needs (mirrors the harness ``common`` dict).

    extract_h / extract_b: contrastive prompts for direction extraction.
    hl: harmless prompts for the first-token KL reference.
    B: harmful behaviour set (the k*/c* endpoint).
    O: over-refusal set (the safety bill).
    mmlu / gsm8k: capability screen items (``{"prompt": ..., "gold": ...}``).
    """

    extract_h: List[str] = field(default_factory=list)
    extract_b: List[str] = field(default_factory=list)
    hl: List[str] = field(default_factory=list)
    B: List[PromptItem] = field(default_factory=list)
    O: List[PromptItem] = field(default_factory=list)
    mmlu: List[Dict[str, Any]] = field(default_factory=list)
    gsm8k: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ScreenInfo:
    """A selection screen: the harmful prompts used to score candidate directions."""

    seed: int
    ids: List[str]
    ids_sha256: str
    pool_sha256: str
    pool_n: int
    note: str
    harmful: List[str]
    benign: List[str]
    row_id_hashes: Dict[str, str] = field(default_factory=dict)
    eligible_index: Optional[List[int]] = None


@dataclass
class NormRatio:
    """Geometry of the defended model's DIM vs the base model's at the selected site."""

    selected_layer: int
    selected_pos: int
    base_dim_norm: Optional[float] = None
    defended_dim_norm: Optional[float] = None
    ratio: Optional[float] = None
    cos: Optional[float] = None
    same_model: bool = True


@dataclass
class MarginReport:
    """The full adaptive-margin audit result (JSON-serializable via :meth:`to_dict`)."""

    tag: str
    spec: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, ConditionRecord] = field(default_factory=dict)
    a0: Optional[ConditionRecord] = None
    iterate: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dose_curve: Dict[str, Any] = field(default_factory=dict)
    repair: Optional[Dict[str, Any]] = None
    norm_ratio: Optional[NormRatio] = None
    control_2x2: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses

        def conv(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: conv(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict):
                return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [conv(v) for v in obj]
            return obj

        return conv(self)
