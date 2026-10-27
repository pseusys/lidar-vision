"""Three-horizon streaming person detector (TAKHeLiPeD): per-beam input features, the single-frame calibration network, and the
object memory.

Design and reasoning in `docs/PROPOSAL.md` (Â§5.3 input, Â§5.4 calibration network); build status in `TODO.md` A43. Stage 2 read one
frame and its predecessors through a coarse temporal convolution until 2026-10-08, when A60 measured it end to end and dropped it;
that code is at commit e76aa94.
Conventions: beam angle 0 is forward, counterclockwise positive, increasing with beam index;
poses are world odometry `(x, y, theta)` with x forward at theta = 0, y left, theta counterclockwise.
"""
from dataclasses import dataclass, fields, replace
from math import log, radians, sqrt
from typing import List, Optional, Tuple

from torch import (
    Tensor, arange, arctan, argsort, atan2, cat, cos, cumprod, exp, expm1, floor, full, full_like, isfinite, linspace, log1p, long,
    minimum, nn, ones_like, randn, as_tensor,
    sigmoid,
    no_grad, ones, sin, stack, tanh, where, zeros, zeros_like,
)
from torch import log as tlog
from torch.nn.functional import interpolate, pad, softmax, softplus
# Item 43's association only. The default path is pure torch, so this import costs nothing it does not use, and scipy
# is already a dependency of `takheliped.utils.drow_utils`.
from scipy.optimize import linear_sum_assignment

MAX_RANGE_M = 10.0
MEDIAN_WINDOW_DEG = 11.0
NEIGHBOUR_OFFSETS = (-2, -1, 1, 2)
N_BEAM_FEATURES = 2 + 2 * len(NEIGHBOUR_OFFSETS)
LFE_NEAR_M = 0.2                     # LFE clips readings below this (`lfe_detector._SCAN_NEAR`)
INPUT_CHANNELS = {"beam": N_BEAM_FEATURES, "lfe": 1}
LEGACY_VALIDITY_CHANNEL = 2          # index of the validity input that checkpoints before 2026-09-17 carried
LAST_TEMPORAL_COMMIT = "e76aa94"     # the last commit with stage 2's coarse temporal convolution (TODO.md A60)
SLOT_STATISTICS = 8
CANDIDATE_FEATURES = 5
PROBABILITY_EPS = 1e-6
SILENT_COAST_LOGIT = -10.0
# Item 16's arms. `selective` is the adopted one: a diagonal state whose decay rate is a function of the input, applied in
# **real seconds** because a robot's frames are not uniformly spaced. `gru` and `mingru` take a *step* and have no notion of
# elapsed time, which is not an oversight in the arm -- it is what the comparison measures.
RECURRENCES = ("selective", "gru", "mingru")
ASSOCIATIONS = ("mutual", "hungarian")
VOTE_FRAMES = ("beam", "sensor")
VISIBILITY_SCALE_M = 1.0        # a metre of free space past a slot saturates the evidence against it
HIT_RATE_TIME_S = 1.0
FEATURE_SCALE = 10.0


def _wrap(angle: Tensor) -> Tensor:
    return atan2(sin(angle), cos(angle))


def sanitize_ranges(ranges: Tensor, max_range_m: float) -> Tensor:
    """Every reading that says "nothing within `max_range_m`" becomes `max_range_m` itself: non-finite, non-positive, or at or
    beyond the limit. One rule for every dataset's encoding of a missing return -- FROG's +inf and 61.0 m, DROW's 29.96 m
    (`memory/noise-structure.md`) -- with nothing to configure per dataset but the model's own range."""
    return where(isfinite(ranges) & (ranges > 0) & (ranges < max_range_m), ranges, full_like(ranges, max_range_m))


def beam_features(ranges: Tensor, angles: Tensor, max_range_m: float = MAX_RANGE_M, median_window_deg: float = MEDIAN_WINDOW_DEG) -> Tensor:
    """Per-beam input channels `[..., N] -> [..., 10, N]`: range, range minus local median, and each neighbour's endpoint
    (along, across) in the beam's own frame relative to the beam's endpoint. Ranges pass through `sanitize_ranges` first."""
    n = ranges.shape[-1]
    step = float(angles[1] - angles[0])
    r = sanitize_ranges(ranges, max_range_m)

    flat = r.reshape(-1, 1, n)
    window = max(1, round(radians(median_window_deg) / step)) | 1
    median = pad(flat, (window // 2, window // 2), mode="replicate").unfold(-1, window, 1).median(dim=-1).values.reshape(r.shape)

    reach = max(abs(o) for o in NEIGHBOUR_OFFSETS)
    r_pad = pad(flat, (reach, reach), mode="replicate").reshape(*r.shape[:-1], n + 2 * reach)
    a_pad = pad(angles.view(1, 1, n), (reach, reach), mode="replicate").view(n + 2 * reach)
    channels = [r, r - median]
    for offset in NEIGHBOUR_OFFSETS:
        neighbour = r_pad[..., reach + offset:reach + offset + n]
        dphi = a_pad[reach + offset:reach + offset + n] - angles
        channels += [neighbour * cos(dphi) - r, neighbour * sin(dphi)]
    return stack(channels, dim=-2)


def lfe_features(ranges: Tensor, max_range_m: float = MAX_RANGE_M) -> Tensor:
    """LFE's whole input (`lfe_detector._normalize_scan`), `[..., N] -> [..., 1, N]`: `1 - r / max_range_m`, readings clipped below
    at `LFE_NEAR_M`, so a near obstacle reads 1 and empty space 0. Missing returns go through `sanitize_ranges` and read as empty
    space, whatever their encoding. Only the attribution arm F1 uses it (`TODO.md` A59)."""
    return (1 - sanitize_ranges(ranges, max_range_m).clamp(min=LFE_NEAR_M) / max_range_m).unsqueeze(-2)


def sensor_to_world(xy: Tensor, pose: Tensor) -> Tensor:
    """Detection-frame points `[B, P, 2]` (x right, y forward) seen from `pose` `[B, 3]` -> world odometry coordinates."""
    x0, y0, th = pose[:, 0:1], pose[:, 1:2], pose[:, 2:3]
    fwd, left = xy[..., 1], -xy[..., 0]
    return stack([x0 + cos(th) * fwd - sin(th) * left, y0 + sin(th) * fwd + cos(th) * left], dim=-1)


def world_to_sensor(xy: Tensor, pose: Tensor) -> Tensor:
    """World points `[B, P, 2]` -> the detection frame of `pose` `[B, 3]`; the inverse of `sensor_to_world`."""
    x0, y0, th = pose[:, 0:1], pose[:, 1:2], pose[:, 2:3]
    dx, dy = xy[..., 0] - x0, xy[..., 1] - y0
    return stack([sin(th) * dx - cos(th) * dy, cos(th) * dx + sin(th) * dy], dim=-1)


@dataclass(frozen=True)
class SlotRules:
    """Fixed rules for creating, matching, replacing and retiring object slots (`docs/PROPOSAL.md` Â§5.5 and Â§6.3).

    `latest_value` is the ablation of the design's value rule (`TODO.md` A51 D2): a matched slot takes the candidate's score
    instead of accumulating towards certainty, so a person hidden for a moment no longer keeps their slot.

    `association` chooses how slots are paired to candidates (`TODO.md` item 43). **mutual**, the adopted rule: a slot's
    nearest candidate and that candidate's nearest slot must agree. **hungarian**: the globally optimal one-to-one
    assignment under the same gate, which is the standard primitive and differs from `mutual` only under contention.
    """
    capacity: int = 256
    floor: float = 0.01
    margin: float = 0.1
    fade_time_s: float = 10.0
    gain: float = 0.1
    gate_m: float = 0.6
    velocity_smoothing: float = 0.1
    latest_value: bool = False
    association: str = "mutual"
    #: `TODO.md` item 22, the owner's design. Above zero, a slot's position **snaps to the centre of a room cell of this
    #: many metres and never moves again**: it addresses a patch of the room rather than an object. Everything else is
    #: unchanged -- same capacity, same room frame, same recurrence, recall, head and eviction -- so the arm varies the
    #: one thing in question, whether *spatial* binding substitutes for *object* binding. Item 36 established that
    #: addressability matters; it never tested which kind. Sparse by construction: only occupied cells take a slot, which
    #: is what makes a 150 m venue affordable where a dense person-scale grid would need ~46,000 cells.
    cell_m: float = 0.0


@dataclass
class Slots:
    """Batched object memory `[B, K]`: slot in use, value, world position and velocity, where it started, age, time since last
    matched, recent match rate, and learned state `[B, K, D]`."""
    alive: Tensor
    value: Tensor
    position: Tensor
    velocity: Tensor
    origin: Tensor
    age_s: Tensor
    since_match_s: Tensor
    hit_rate: Tensor
    state: Tensor


@dataclass
class SlotEvents:
    """What one bookkeeping step did to each slot `[B, K]`: matched to candidate `match_index`, or (re)filled from candidate
    `source`; -1 where neither."""
    matched: Tensor
    match_index: Tensor
    source: Tensor


def initial_slots(batch: int, rules: SlotRules, state_dim: int, device=None) -> Slots:
    """An empty memory for `batch` streams."""
    k = rules.capacity
    return Slots(zeros(batch, k, device=device).bool(), zeros(batch, k, device=device), zeros(batch, k, 2, device=device),
                 zeros(batch, k, 2, device=device), zeros(batch, k, 2, device=device), zeros(batch, k, device=device),
                 zeros(batch, k, device=device), zeros(batch, k, device=device), zeros(batch, k, state_dim, device=device))


def _where_slots(mask: Tensor, a: Slots, b: Slots) -> Slots:
    """Per stream `[B]`: the memory `a` where `mask`, else `b`."""
    picked = {}
    for f in fields(Slots):
        x, y = getattr(a, f.name), getattr(b, f.name)
        picked[f.name] = where(mask.view(-1, *([1] * (x.dim() - 1))), x, y)
    return Slots(**picked)


def _assign(dist: Tensor, gate_m: float) -> Tuple[Tensor, Tensor]:
    """The globally optimal one-to-one slot-candidate assignment under `gate_m` (`TODO.md` item 43).

    Solved **over the gated sub-problem only**, one stream at a time. Two reasons, and both matter: `dist` is `inf`
    outside `alive & usable` and `linear_sum_assignment` cannot take that; and the gated block is far smaller than the
    full matrix -- roughly 24 live slots by 22 usable candidates against 256 by 64 -- which is what keeps a per-frame
    scipy call affordable inside the phase that already dominates the CPU budget.

    A slot the assignment leaves unmatched keeps index 0, which is safe because every use of the index downstream is
    gated by `matched`.
    """
    b, k, _ = dist.shape
    index = zeros(b, k, dtype=long, device=dist.device)
    matched = zeros(b, k, dtype=bool, device=dist.device)
    within = dist <= gate_m                       # already inf-masked, so this is alive & usable & inside the gate
    for i in range(b):
        rows = within[i].any(dim=1).nonzero(as_tuple=True)[0]
        cols = within[i].any(dim=0).nonzero(as_tuple=True)[0]
        if not len(rows) or not len(cols):
            continue
        block = dist[i][rows][:, cols]
        # A finite stand-in for the out-of-gate pairs, large enough that the solver takes one only when forced -- and
        # the gate check below then drops it, so a forced pair cannot become a match.
        block = block.masked_fill(~isfinite(block) | (block > gate_m), 10.0 * gate_m * (len(rows) + len(cols) + 1))
        picked_rows, picked_cols = linear_sum_assignment(block.cpu().numpy())
        picked_rows = as_tensor(picked_rows, device=dist.device)
        picked_cols = as_tensor(picked_cols, device=dist.device)
        keep = block[picked_rows, picked_cols] <= gate_m
        index[i, rows[picked_rows[keep]]] = cols[picked_cols[keep]]
        matched[i, rows[picked_rows[keep]]] = True
    return index, matched


@no_grad()
def manage_slots(slots: Slots, cand_xy: Tensor, cand_score: Tensor, cand_valid: Tensor, dt: Tensor, rules: SlotRules) -> Tuple[Slots, SlotEvents]:
    """One bookkeeping step in world coordinates, candidates `[B, P, 2]` with scores and validity, `dt` `[B]` seconds.

    Slots move by their velocity; mutual nearest slot-candidate pairs within the gate match; matched values rise by
    `gain * (1 - v) * score`, the rest fade to the floor over `fade_time_s`, and slots under the floor retire. Unmatched
    candidates at or above the floor fill free slots strongest first; the strongest remaining ones then replace the weakest
    slots not matched or filled this step, while `score > value + margin`. Learned state passes through: the caller sets it
    for refilled slots.
    """
    b, k = slots.alive.shape
    p = cand_xy.shape[1]
    device = cand_xy.device
    step = dt[:, None]
    alive = slots.alive
    if rules.cell_m:
        # Item 22. Quantise every candidate to its cell centre before anything reads a position, so a slot's position
        # *is* a cell: spawning puts it at a centre, matching compares centres, and nothing below can move it. A cell
        # does not drift either, so there is no constant-velocity prediction to make.
        cand_xy = (floor(cand_xy / rules.cell_m) + 0.5) * rules.cell_m
        predicted = slots.position
    else:
        predicted = slots.position + slots.velocity * step[..., None] * alive[..., None]
    usable = cand_valid & (cand_score >= rules.floor)

    if rules.association not in ASSOCIATIONS:
        raise ValueError(f"association must be one of {list(ASSOCIATIONS)}, not {rules.association!r}")

    dist = (predicted[:, :, None, :] - cand_xy[:, None, :, :]).norm(dim=-1)
    dist = dist.masked_fill(~(alive[:, :, None] & usable[:, None, :]), float("inf"))
    # With quantised candidates a same-cell pair is exactly 0 m apart and the nearest other cell is `cell_m` away, so
    # half a cell separates "same cell" from everything else however the gate was tuned for tracking slots.
    gate = rules.gate_m if not rules.cell_m else min(rules.gate_m, 0.5 * rules.cell_m)
    if rules.association == "hungarian":
        nearest_candidate, matched = _assign(dist, gate)
    else:
        nearest_candidate = dist.argmin(dim=2)
        mutual = dist.argmin(dim=1).gather(1, nearest_candidate) == arange(k, device=device)[None]
        matched = alive & mutual & (dist.gather(2, nearest_candidate[..., None])[..., 0] <= gate)

    match_xy = cand_xy.gather(1, nearest_candidate[..., None].expand(-1, -1, 2))
    fade = exp(-step * log(1.0 / rules.floor) / rules.fade_time_s)
    observed = cand_score.gather(1, nearest_candidate)
    raised = observed if rules.latest_value else slots.value + rules.gain * (1.0 - slots.value) * observed
    value = where(matched, raised, slots.value * fade)
    velocity = slots.velocity if rules.cell_m else where(matched[..., None], slots.velocity + rules.velocity_smoothing * ((match_xy - slots.position) / step[..., None] - slots.velocity), slots.velocity)
    # A cell keeps its centre whatever matched it; only a tracking slot follows its candidate.
    position = predicted if rules.cell_m else where(matched[..., None], match_xy, predicted)
    hit_rate = slots.hit_rate + (1.0 - exp(-step / HIT_RATE_TIME_S)) * (matched.float() - slots.hit_rate)
    since = where(matched, zeros_like(slots.since_match_s), slots.since_match_s + step)
    age = slots.age_s + step
    alive = alive & (value >= rules.floor)

    claimed = zeros(b, p, device=device).scatter_add(1, nearest_candidate, matched.float()) > 0
    eligible = usable & ~claimed
    if rules.cell_m:
        # A cell is unique, so two things the tracking path allows have to be refused here: spawning into a cell a live
        # slot already holds (it exists, it simply did not match this frame), and two candidates in one cell spawning
        # two slots. Without the second, a crowded cell would quietly consume the whole budget.
        held = (dist == 0).any(dim=1)
        same = (cand_xy[:, :, None, :] == cand_xy[:, None, :, :]).all(dim=-1)
        index = arange(p, device=device)
        stronger = (cand_score[:, :, None] < cand_score[:, None, :]) |                    ((cand_score[:, :, None] == cand_score[:, None, :]) & (index[None, :, None] > index[None, None, :]))
        eligible = eligible & ~held & ~(same & stronger & usable[:, None, :]).any(dim=2)
    n_eligible = eligible.sum(dim=1)
    by_score = argsort(where(eligible, -cand_score, full_like(cand_score, float("inf"))), dim=1, stable=True)
    by_index = argsort(alive.float(), dim=1, stable=True)
    ranks = arange(min(p, k), device=device)[None]
    n_spawn = minimum(n_eligible, (~alive).sum(dim=1))
    source = full((b, k), -1, device=device)
    source = source.scatter(1, by_index[:, :ranks.shape[1]], where(ranks < n_spawn[:, None], by_score[:, :ranks.shape[1]], full_like(ranks, -1).expand(b, -1)))

    evictable = alive & ~matched & (source < 0)
    weakest = argsort(where(evictable, value, full_like(value, float("inf"))), dim=1, stable=True)[:, :ranks.shape[1]]
    leftover = by_score.gather(1, (n_spawn[:, None] + ranks).clamp(max=p - 1))
    replace = (n_spawn[:, None] + ranks < n_eligible[:, None]) & (ranks < evictable.sum(dim=1)[:, None])
    replace = cumprod((replace & (cand_score.gather(1, leftover) > value.gather(1, weakest) + rules.margin)).long(), dim=1).bool()
    source = source.scatter(1, weakest, where(replace, leftover, source.gather(1, weakest)))

    refill = source >= 0
    src = source.clamp(min=0)
    src_xy = cand_xy.gather(1, src[..., None].expand(-1, -1, 2))
    new = Slots(
        alive=alive | refill,
        value=where(refill, cand_score.gather(1, src), value),
        position=where(refill[..., None], src_xy, position),
        velocity=where(refill[..., None], zeros_like(velocity), velocity),
        origin=where(refill[..., None], src_xy, slots.origin),
        age_s=where(refill, zeros_like(age), age),
        since_match_s=where(refill, zeros_like(since), since),
        hit_rate=where(refill, full_like(hit_rate, 1.0), hit_rate),
        state=slots.state,
    )
    return new, SlotEvents(matched, where(matched, nearest_candidate, full_like(nearest_candidate, -1)), source)


class _BiasedAttention(nn.Module):
    """Multi-head attention with an additive per-pair bias, a validity mask over the attended items, and a learned null item
    that is always attendable, so a query with nothing to attend to still has a defined answer."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value, self.out = (nn.Linear(dim, dim) for _ in range(4))
        self.null_key = nn.Parameter(zeros(1, 1, dim))
        self.null_value = nn.Parameter(zeros(1, 1, dim))

    def _split(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, query: Tensor, items: Tensor, bias: Tensor, valid: Tensor) -> Tensor:
        b, q, d = query.shape
        keys = cat([self.null_key.expand(b, -1, -1), items], dim=1)
        values = cat([self.null_value.expand(b, -1, -1), items], dim=1)
        bias = cat([zeros(b, q, 1, device=query.device), bias], dim=2)
        valid = cat([ones(b, 1, device=query.device).bool(), valid], dim=1)
        logits = self._split(self.query(query)) @ self._split(self.key(keys)).transpose(-1, -2) / sqrt(d // self.heads) + bias[:, None]
        mixed = softmax(logits.masked_fill(~valid[:, None, None, :], float("-inf")), dim=-1) @ self._split(self.value(values))
        return self.out(mixed.transpose(1, 2).reshape(b, q, d))


class _SlotAttention(nn.Module):
    """Slot Attention's attention and its post-update MLP, for one iteration (Locatello et al., NeurIPS 2020) -- item 42.

    The defining detail is the softmax **axis**: over slots, so slots compete to explain each candidate and the weights
    are a fixed budget per candidate. A transformer's cross-attention normalises over candidates instead, which lets
    every slot read the same candidate at full weight -- that is `_BiasedAttention`, and it is what this replaces.
    Single-head, as published, and the paper's second normalisation over candidates turns the competition into a
    weighted *mean* rather than a sum.

    **A null slot is always attendable** and holds the weight of a candidate no slot explains. It is not in the paper,
    because the paper does not have this problem: Slot Attention re-initialises its slots from a learned Gaussian on
    every input, so at least one always fits, while here slots persist across frames and a stream's first frame has
    none alive -- and a softmax over an entirely masked axis is NaN. Its weight is dropped from the update, since it
    has no state to carry, and `read` gives it a learned value so the recall has a defined answer too.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm_slots, self.norm_items = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        # A query in projected space rather than a state put through `to_q`: the null slot has no state, and at zero it
        # competes at logit 0, which is a neutral starting point the loss can move either way.
        self.null_query = nn.Parameter(zeros(1, 1, dim))
        self.null_value = nn.Parameter(zeros(1, 1, dim))
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, state: Tensor, items: Tensor, bias: Tensor, valid: Tensor, alive: Tensor) -> Tuple[Tensor, Tensor]:
        """`state [B, K, D]`, candidates `items [B, P, D]`, their pairwise `bias [B, P, K]`, `valid [B, P]`, `alive [B, K]`.

        Returns the per-slot update `[B, K, D]` and the candidate-to-slot attention `[B, P, K + 1]`, the null slot last.
        """
        batch, dim = state.shape[0], state.shape[-1]
        items = self.norm_items(items)
        queries = cat([self.to_q(self.norm_slots(state)), self.null_query.expand(batch, -1, -1)], dim=1)
        logits = self.to_k(items) @ queries.transpose(-1, -2) / sqrt(dim)
        logits = logits + cat([bias, zeros(batch, bias.shape[1], 1, device=bias.device)], dim=-1)
        attendable = cat([alive, ones_like(alive[:, :1])], dim=1)
        attention = softmax(logits.masked_fill(~attendable[:, None, :], float("-inf")), dim=-1)
        weights = attention[..., :-1] * valid[..., None]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=PROBABILITY_EPS)
        return weights.transpose(-1, -2) @ self.to_v(items), attention

    def read(self, attention: Tensor, state: Tensor) -> Tensor:
        """The recall, through the same matrix: each candidate reads the slots that competed for it, `[B, P, D]`.

        This is why the arm removes `self.recall` rather than keeping it beside the update. A separate attention would
        be a second, differently normalised association over the same pairs, and then "the published update rule" would
        no longer be what is being measured.
        """
        return attention @ cat([state, self.null_value.expand(state.shape[0], -1, -1)], dim=1)


@dataclass
class MemoryOutput:
    """One object-memory step: rescored candidate logits `[B, P]`; person logits of slots `[B, K]` and their positions in the
    current detection frame `[B, K, 2]`; which slots may report a person no candidate covers (`coasting`); the new memory; and
    what the bookkeeping did -- which slot matched or spawned from which candidate.

    `events` carries no gradient and changes nothing about the step; it is here so a diagnostic can read **which slot backs a
    rescored candidate**, which is what a creation or confirmation rule would act on (`TODO.md` Â§A item 14)."""
    candidate_logit: Tensor
    #: Where each candidate is reported, `[B, P, 2]`. Equal to the input `cand_xy` unless the memory has a position
    #: refinement head (`TODO.md` item 46), so everything downstream reads one field rather than branching on a flag.
    candidate_xy: Tensor
    slot_logit: Tensor
    slot_xy: Tensor
    coasting: Tensor
    slots: Slots
    events: SlotEvents


def _logit(p: Tensor) -> Tensor:
    p = p.clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return tlog(p) - log1p(-p)


def _proximity(a: Tensor, b: Tensor, sigma_m: float) -> Tensor:
    """Attention bias `-|a_i - b_j|^2 / (2 sigma^2)` between point sets `[B, I, 2]` and `[B, J, 2]`."""
    return -((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(dim=-1) / (2.0 * sigma_m ** 2)


class ObjectMemory(nn.Module):
    """Stage 3, the short and long horizons (`docs/PROPOSAL.md` Â§5.5): object slots in world coordinates, learned matching of
    candidates to slots, exchange between slots, selective recurrence in real time, rule-based slot bookkeeping, candidate
    rescoring that starts silent. **Coasting -- a slot reporting a person no candidate covers -- was removed from the
    architecture on 2026-10-16** after item 24 measured it costing nothing and firing essentially never; `coast=True`
    restores the head, which every number published before that date was produced with.

    The ablations of A51 phases 3b and 5, each off by default: `room_frame` false keeps slots in the sensor frame, where a static
    object drifts with the robot; `SlotRules.latest_value` replaces accumulated value with the latest score; `fixed_decay_s` gives
    every slot one time constant instead of a learned one; `learn_value` turns the value rule's two constants into parameters;
    `exchange` false removes the slot-to-slot self-attention, the only O(K^2) term; `learned_match` false removes the
    cross-attention, leaving each slot the candidate its hand-coded association matched. Both false together is the per-slot
    recurrent memory a reviewer would ask for as a baseline; `recurrent` false removes the slot state itself, leaving the head a
    candidate and the statistics of the slot matched to it -- the test of whether this is a learned memory or a tracker with a
    learned scorer (A51 E10).

    `rescore` false removes the correction entirely, so a candidate keeps its own score and the memory can reach the loss only
    through the prior it draws into stage 2 -- the test of whether that feedback path works at all (A51 E12a).

    Rescoring starts silent either way (`TODO.md` A51): through a scalar gate initialised at zero, or -- with `rescore_gate` false --
    through a zero-initialised last layer, which leaves the head's own gradient unscaled. `rescore_limit`, when set, bounds the
    correction to +-`rescore_limit` logits through a tanh, against the signed correction that shifted the whole score distribution
    down.
    """

    def __init__(self, rules: SlotRules = SlotRules(), dim: int = 128, heads: int = 4, sigma_m: float = 0.6, decay_init_s: Tuple[float, float] = (1.0, 60.0),
                 feature_dim: int = 0, rescore_gate: bool = True, rescore_limit: float = 0.0, room_frame: bool = True,
                 fixed_decay_s: float = 0.0, learn_value: bool = False, exchange: bool = True, learned_match: bool = True,
                 recurrent: bool = True, rescore: bool = True, visibility: bool = False, coast: bool = False,
                 recurrence: str = "selective", global_state: bool = False, refine: bool = False,
                 iterations: int = 0):
        super().__init__()
        # `TODO.md` item 42: Slot Attention's own update in place of the association's role in the state update. Each
        # refusal is a feature the published mechanism either duplicates or cannot be given.
        if iterations:
            if learned_match:
                raise ValueError("iterations already associates candidates to slots, so learned_match must be False "
                                 "(TODO.md item 42)")
            if exchange:
                raise ValueError("iterations: published Slot Attention has no slot-to-slot exchange, so exchange must "
                                 "be False (TODO.md item 42)")
            if not recurrent:
                raise ValueError("iterations updates a slot state, so recurrent must be True (TODO.md item 42)")
            if global_state:
                raise ValueError("iterations needs addressable slots to compete, so global_state must be False "
                                 "(TODO.md item 42)")
        # `TODO.md` item 46 (A51's O3). Item 35 found the memory net negative on *coverage* -- 203 recovered against
        # 2,990 lost at its own threshold -- so all of its gain is ranking. This is the only head that could move a
        # detection rather than rescore it. It needs `rescore`, because without it `recalled` has nowhere else to go
        # and the arm would silently become A51 E12a's experiment instead of this one.
        if refine and not rescore:
            raise ValueError("refine needs rescore: without it the recalled state reaches nothing else, which is "
                             "A51 E12a's arm rather than item 46's")
        # `TODO.md` item 36: one scene-level state in place of `rules.capacity` addressable per-object slots. The three
        # refusals are not conservatism -- each of those features reads a slot's *position*, which this arm does not have.
        if global_state:
            if coast:
                raise ValueError("global_state carries no slot positions, so coast must be False (TODO.md item 36)")
            if visibility:
                raise ValueError("global_state carries no slot positions, so visibility must be False (TODO.md item 36)")
            if not recurrent:
                raise ValueError("global_state with recurrent=False would leave no memory at all, which is not an arm")
            rules = replace(rules, capacity=1)      # a capacity counts addressable slots, and this arm has none
        self.global_state = global_state
        addressable = not global_state
        self.rules, self.dim, self.sigma_m, self.feature_dim = rules, dim, sigma_m, feature_dim
        self.visibility = visibility
        statistics = SLOT_STATISTICS + int(visibility)
        self.rescore_limit, self.room_frame, self.fixed_decay_s = rescore_limit, room_frame, fixed_decay_s
        self.recurrent = recurrent
        self.encode = nn.Sequential(nn.Linear(CANDIDATE_FEATURES + feature_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.spawn = nn.Linear(dim, dim) if recurrent and addressable else None
        self.statistics = nn.Linear(statistics, dim)
        self.match = _BiasedAttention(dim, heads) if recurrent and learned_match and addressable else None
        self.exchange = _BiasedAttention(dim, heads) if recurrent and exchange and addressable else None
        self.exchange_norm = nn.LayerNorm(dim) if recurrent and exchange and addressable else None
        # Attention pooling over the frame's candidates, with a learned query: the owner's call of 2026-10-22 is to probe
        # the ceiling of a swap with the strongest global variant, because if this cannot reach the frozen model's 86%
        # then a masked mean cannot either and the slots have earned their place. `_BiasedAttention` carries an always
        # attendable null item, so a frame with no valid candidate still has a defined answer.
        self.pool = _BiasedAttention(dim, heads) if global_state else None
        if global_state:
            self.query = nn.Parameter(randn(1, 1, dim) * 0.02)
        else:
            self.register_parameter("query", None)
        if recurrence not in RECURRENCES:
            raise ValueError(f"recurrence must be one of {list(RECURRENCES)}, not {recurrence!r}")
        self.recurrence = recurrence
        self.update = nn.Linear(dim + statistics, dim) if recurrent else None
        # A GRU needs an update and a reset gate, both reading the state; minGRU (Feng et al., *Were RNNs All We Needed?*,
        # 2024) drops the reset gate and the state dependence, leaving one gate over the input alone.
        self.gate = nn.Linear(dim + statistics, dim) if recurrent and recurrence in ("gru", "mingru") else None
        self.reset = nn.Linear(dim + statistics, dim) if recurrent and recurrence == "gru" else None
        self.gru_state = nn.Linear(dim, dim, bias=False) if recurrent and recurrence == "gru" else None
        if fixed_decay_s or not recurrent or recurrence != "selective":
            self.decay, self.decay_bias = None, None        # one time constant for every slot, or no slot state at all
        else:
            self.decay = nn.Linear(dim + statistics, dim, bias=False)
            times = exp(linspace(log(decay_init_s[0]), log(decay_init_s[1]), dim))
            self.decay_bias = nn.Parameter(tlog(expm1(1.0 / times)))
        if learn_value:                                      # the value rule's two constants, as parameters the loss can reach
            self.value_gain = nn.Parameter(tlog(full((), rules.gain / (1.0 - rules.gain))))
            self.value_fade = nn.Parameter(tlog(expm1(full((), rules.fade_time_s))))
        self.iterations = iterations
        self.attend = _SlotAttention(dim) if iterations else None
        # Item 42 reads the recall off its own attention matrix, so the separate one is not built at all.
        self.recall = _BiasedAttention(dim, heads) if recurrent and addressable and not iterations else None
        self.rescore = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1)) if rescore else None
        # Zero-initialised last layer, so an untrained arm reports exactly the candidate's own position and its first
        # epochs are the control's. Without that, "refinement" and "a randomly displaced detection" are the same thing.
        self.refine = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 2)) if refine else None
        if self.refine is not None:
            nn.init.zeros_(self.refine[-1].weight), nn.init.zeros_(self.refine[-1].bias)
        if rescore and rescore_gate:
            self.rescore_gate = nn.Parameter(zeros(()))
        else:
            self.register_parameter("rescore_gate", None)       # the head starts silent through its own last layer instead
            if rescore:
                nn.init.zeros_(self.rescore[-1].weight), nn.init.zeros_(self.rescore[-1].bias)
        # **Off by default since 2026-10-16** (`TODO.md` item 24, the owner's call): ablating the head cost nothing
        # measurable -- AP 86.08% against the matched seed's 86.35% and a three-seed control range of 85.49-86.35 --
        # and it fired 0.0012-0.0020 times per populated frame and 2/0/0 times across 72,533 person-free frames. The
        # argument is not its 137 parameters but that a write-up should not describe a mechanism that never acts.
        # `coast=True` restores it, because every number published before that date was produced with it.
        self.coast = nn.Linear(dim + statistics, 1) if coast else None
        if self.coast is not None:
            nn.init.zeros_(self.coast.weight)
            nn.init.constant_(self.coast.bias, SILENT_COAST_LOGIT)

    def _recur(self, state: Tensor, u: Tensor, dt: Tensor) -> Tensor:
        """One slot-state update, in whichever recurrence this memory was built with (`TODO.md` item 16).

        **`selective`**, the adopted one: a diagonal state whose decay rate is read off the input, applied over the real
        elapsed time -- `keep = exp(-dt * rate)` in seconds. A robot's frames are not uniformly spaced, and D3 measured the
        learned rate as load-bearing (fixing it costs 1.04 pp).

        **`gru`** is the textbook gated recurrent unit (Cho et al., 2014): an update gate, a reset gate and a candidate that
        reads the gated state. **`mingru`** is Feng et al.'s minimal variant (*Were RNNs All We Needed?*, 2024), which drops
        the reset gate and the hidden-state dependence inside the gates.

        **Neither of them sees `dt`**, and that is deliberate: they take a step, which is exactly the assumption the adopted
        arm abandons. An arm with `dt` bolted on would measure something else.
        """
        if self.recurrence == "selective":
            rate = 1.0 / self.fixed_decay_s if self.decay is None else softplus(self.decay(u) + self.decay_bias)
            keep = exp(-dt[:, None, None] * rate)
            return keep * state + (1.0 - keep) * tanh(self.update(u))
        z = sigmoid(self.gate(u))
        if self.recurrence == "mingru":
            return (1.0 - z) * state + z * tanh(self.update(u))
        candidate = tanh(self.update(u) + self.gru_state(sigmoid(self.reset(u)) * state))
        return (1.0 - z) * state + z * candidate

    def _slot_attention(self, state: Tensor, encoded: Tensor, world: Tensor, cand_valid: Tensor, slots: Slots,
                        stats: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
        """`TODO.md` item 42: iterated competitive attention in place of the association's role in the state update.

        Per iteration: slots compete for the frame's candidates, each slot's update is the attention-weighted mean of
        what it won, the recurrence folds that into its state, and the MLP refines it. The slot statistics ride along in
        the recurrence's input, exactly as they do in the hand-coded path, so `_recur` keeps its width and item 16's
        recurrence family stays orthogonal to this ablation.

        **`dt` is spent on the first iteration only.** Elapsed time passes once per frame, not once per refinement step,
        so a `selective` arm at three iterations would otherwise decay three frames' worth of state in one frame.

        The returned attention is the last iteration's, which read the state before that iteration's update. That is
        the paper's arrangement -- the matrix is a by-product of the update, and the decoder reads the final slots.
        """
        bias = _proximity(world, slots.position, self.sigma_m)
        attention = None
        for iteration in range(self.iterations):
            update, attention = self.attend(state, encoded, bias, cand_valid, slots.alive)
            state = self._recur(state, cat([update, stats], dim=-1), dt if iteration == 0 else zeros_like(dt))
            state = state + self.attend.mlp(state)
        return state, attention

    def _coasting(self, state: Tensor, stats: Tensor, slots: Slots, events: SlotEvents, refill: Tensor):
        """Which slots may report a person no candidate covers, and how loudly -- or neither, when the head is ablated.

        Without the head there is no logit to learn and nothing to report, so the logit is pinned silent and `coasting` is
        empty: a slot can then only reach the output through a candidate it matched (`TODO.md` item 24).
        """
        if self.coast is None:
            return full_like(slots.value, SILENT_COAST_LOGIT), zeros_like(slots.alive)
        return self.coast(cat([state, stats], dim=-1))[..., 0], slots.alive & ~events.matched & ~refill

    def _scene_statistics(self, cand_xy: Tensor, cand_score: Tensor, cand_valid: Tensor, dt: Tensor, slots: Slots) -> Tensor:
        """The scene-level analogues of the eight per-slot statistics, `[B, 1, SLOT_STATISTICS]` (`TODO.md` item 36).

        Deliberately the same width as the slot statistics they replace, so `_recur` and every gate keep the input width
        they have in the addressable model and the recurrence family stays orthogonal to this ablation. With no
        bookkeeping, **this is `dt`'s only route into the model**, which matters because item 37 showed that route
        carries real signal.
        """
        v = cand_valid.float()
        n = v.sum(-1).clamp(min=1.0)
        return stack([(cand_score * v).sum(-1) / n,
                      (cand_score * v).amax(-1),
                      v.sum(-1) / FEATURE_SCALE,
                      v.mean(-1),
                      (cand_xy.norm(dim=-1) * v).sum(-1) / n / FEATURE_SCALE,
                      (_logit(cand_score) * v).sum(-1) / n / FEATURE_SCALE,
                      dt,
                      slots.age_s[:, 0] / FEATURE_SCALE], dim=-1)[:, None]

    def _global_step(self, cand_xy: Tensor, cand_score: Tensor, cand_valid: Tensor, pose: Tensor, dt: Tensor,
                     slots: Slots, cand_features: Optional[Tensor] = None) -> MemoryOutput:
        """One frame with a single scene-level state and no addressable slots at all (`TODO.md` item 36).

        `manage_slots` **is** the slot mechanism, so this path never calls it: no association, no spawn, no eviction, no
        per-slot decay, no proximity bias anywhere. The frame's candidates are pooled into one vector, that vector plus
        the scene statistics updates the one state through the same `_recur`, and **every candidate is rescored against
        the same state** -- which is exactly the question, since the adopted model rescores each candidate against the
        state of the slot its association gave it.

        It is a fair like-for-like because the frozen memory only rescores: the coasting head went in item 24, so
        neither arm places a detection and the only difference is whether the memory is addressable.
        """
        features = stack([cand_score, _logit(cand_score) / FEATURE_SCALE, cand_xy[..., 0] / FEATURE_SCALE,
                          cand_xy[..., 1] / FEATURE_SCALE, cand_xy.norm(dim=-1) / FEATURE_SCALE], dim=-1)
        if self.feature_dim:
            features = cat([features, cand_features], dim=-1)
        encoded = self.encode(features)
        b, p = cand_score.shape
        pooled = self.pool(self.query.expand(b, -1, -1), encoded, zeros(b, 1, p, device=cand_score.device), cand_valid)
        stats = self._scene_statistics(cand_xy, cand_score, cand_valid, dt, slots)
        state = self._recur(slots.state, cat([pooled, stats], dim=-1), dt)
        slots = Slots(ones_like(slots.alive), slots.value, slots.position, slots.velocity, slots.origin,
                      slots.age_s + dt[:, None], slots.since_match_s, slots.hit_rate, state)
        recalled = state.expand(-1, p, -1)
        candidate_logit = self._rescored(encoded, recalled, cand_score)
        empty = full_like(slots.alive, -1, dtype=long)
        return MemoryOutput(candidate_logit, self._refined(encoded, recalled, cand_xy),
                            full_like(slots.value, SILENT_COAST_LOGIT), slots.position,
                            zeros_like(slots.alive), slots, SlotEvents(zeros_like(slots.alive), empty, empty))

    def _refined(self, encoded: Tensor, recalled: Tensor, cand_xy: Tensor) -> Tensor:
        """Where each candidate is reported: its own position, plus the memory's correction when it has one.

        The head predicts an **offset**, not an absolute position (`TODO.md` item 46): a displaced detection has to stay
        near its candidate, and a head predicting absolute coordinates would start at the origin and have to learn the
        whole room's geometry before it beat the control.
        """
        if self.refine is None:
            return cand_xy
        return cand_xy + self.refine(cat([encoded, recalled], dim=-1))

    def _rescored(self, encoded: Tensor, recalled: Tensor, cand_score: Tensor) -> Tensor:
        """A candidate's logit after the memory's correction, or its own logit untouched when `rescore` is false (A51 E12a),
        which leaves the prior drawn into stage 2 as the memory's only remaining route to the loss."""
        if self.rescore is None:
            return _logit(cand_score)
        correction = self.rescore(cat([encoded, recalled], dim=-1))[..., 0]
        if self.rescore_limit:
            correction = self.rescore_limit * tanh(correction)
        return _logit(cand_score) + (correction if self.rescore_gate is None else self.rescore_gate * correction)

    def initial_state(self, batch: int, device=None) -> Slots:
        return initial_slots(batch, self.rules, self.dim, device)

    def slot_rules(self) -> SlotRules:
        """The bookkeeping rules for this frame. With `learn_value` the value rule's two constants come from parameters, so the
        rule itself moves as they train, while the discrete decisions inside `manage_slots` stay under `no_grad`."""
        if not hasattr(self, "value_gain"):
            return self.rules
        return replace(self.rules, gain=self.value_gain.detach().sigmoid().item(), fade_time_s=softplus(self.value_fade.detach()).item())

    def _value_with_gradient(self, previous: Slots, slots: Slots, events: SlotEvents, cand_score: Tensor, dt: Tensor) -> Tensor:
        """The same value the rule just computed, recomputed so the loss reaches `value_gain` and `value_fade` (`TODO.md` A51 D4).
        The bookkeeping keeps its own detached value; this one is what the network reads."""
        gain, fade_time = self.value_gain.sigmoid(), softplus(self.value_fade)
        observed = cand_score.gather(1, events.match_index.clamp(min=0))
        raised = observed if self.rules.latest_value else previous.value + gain * (1.0 - previous.value) * observed
        faded = previous.value * exp(-dt[:, None] * log(1.0 / self.rules.floor) / fade_time)
        return where(events.source >= 0, slots.value, where(events.matched, raised, faded))

    def step(self, cand_xy: Tensor, cand_score: Tensor, cand_valid: Tensor, pose: Tensor, dt: Tensor, slots: Slots, reset: Optional[Tensor] = None,
             cand_features: Optional[Tensor] = None, free_range: Optional[Tensor] = None, angles: Optional[Tensor] = None) -> MemoryOutput:
        """One frame: candidates `[B, P, 2]` in the detection frame with scores and validity, robot pose `[B, 3]`, seconds since
        the previous frame `[B]`; `reset` `[B]` empties those streams' memory first; `cand_features` `[B, P, feature_dim]` are the
        detector's own features of each candidate, when the memory was built with `feature_dim`."""
        if reset is not None:
            slots = _where_slots(reset, self.initial_state(len(reset), cand_xy.device), slots)
        if self.global_state:
            return self._global_step(cand_xy, cand_score, cand_valid, pose, dt, slots, cand_features)
        world = sensor_to_world(cand_xy, pose) if self.room_frame else cand_xy
        previous = slots
        slots, events = manage_slots(slots, world, cand_score, cand_valid, dt, self.slot_rules())
        if hasattr(self, "value_gain"):
            slots = replace(slots, value=self._value_with_gradient(previous, slots, events, cand_score, dt))

        features = stack([cand_score, _logit(cand_score) / FEATURE_SCALE, cand_xy[..., 0] / FEATURE_SCALE, cand_xy[..., 1] / FEATURE_SCALE, cand_xy.norm(dim=-1) / FEATURE_SCALE], dim=-1)
        if self.feature_dim:
            features = cat([features, cand_features], dim=-1)
        encoded = self.encode(features)
        refill = events.source >= 0
        if self.recurrent:
            spawned = tanh(self.spawn(encoded.gather(1, events.source.clamp(min=0)[..., None].expand(-1, -1, self.dim))))
            state = where(refill[..., None], spawned, slots.state)
        else:
            state = slots.state

        observed_score = where(refill, cand_score.gather(1, events.source.clamp(min=0)), cand_score.gather(1, events.match_index.clamp(min=0)) * events.matched)
        stats = stack([slots.value, slots.age_s / FEATURE_SCALE, slots.since_match_s / FEATURE_SCALE, (slots.position - slots.origin).norm(dim=-1),
                       slots.velocity.norm(dim=-1), slots.hit_rate, (events.matched | refill).float(), observed_score], dim=-1)
        if self.visibility:
            if free_range is None or angles is None:
                raise ValueError("this memory was built with visibility=True, so step() needs free_range and angles; "
                                 "free_range_profile() builds the first from the raw scan")
            here = world_to_sensor(slots.position, pose) if self.room_frame else slots.position
            stats = cat([stats, visibility_feature(here, free_range, angles, free_range.shape[-1])[..., None]], dim=-1)

        if not self.recurrent:
            # no slot state: the head reads each candidate beside the statistics of the slot its association gave it
            # one row per candidate plus a discard row, so the slots that matched nothing cannot overwrite a real entry
            candidates = cand_score.shape[1]
            held = zeros(stats.shape[0], candidates + 1, stats.shape[-1], device=cand_score.device)
            taken = events.matched | refill
            target = where(taken, where(refill, events.source, events.match_index), full_like(events.source, candidates))
            held = held.scatter(1, target.clamp(min=0)[..., None].expand(-1, -1, stats.shape[-1]), stats)
            recalled = self.statistics(held[:, :candidates])
            slots = Slots(slots.alive, slots.value, slots.position, slots.velocity, slots.origin, slots.age_s, slots.since_match_s,
                          slots.hit_rate, state)
            candidate_logit = self._rescored(encoded, recalled, cand_score)
            slot_logit, coasting = self._coasting(state, stats, slots, events, refill)
            slot_xy = world_to_sensor(slots.position, pose) if self.room_frame else slots.position
            return MemoryOutput(candidate_logit, self._refined(encoded, recalled, cand_xy), slot_logit, slot_xy, coasting, slots, events)

        if self.attend is None:
            queried = state + self.statistics(stats)
            if self.match is None:                           # the slot reads only what the hand-coded association gave it
                matched_encoding = encoded.gather(1, events.match_index.clamp(min=0)[..., None].expand(-1, -1, self.dim))
                found = queried + matched_encoding * (events.matched | refill)[..., None]
            else:
                found = self.match(queried, encoded, _proximity(slots.position, world, self.sigma_m), cand_valid)
            mixed = found if self.exchange is None else self.exchange_norm(found + self.exchange(found, found, _proximity(slots.position, slots.position, self.sigma_m), slots.alive))
            u = cat([mixed, stats], dim=-1)
            state = self._recur(state, u, dt) * slots.alive[..., None]
            attention = None
        else:
            state, attention = self._slot_attention(state, encoded, world, cand_valid, slots, stats, dt)
            state = state * slots.alive[..., None]
        slots = Slots(slots.alive, slots.value, slots.position, slots.velocity, slots.origin, slots.age_s, slots.since_match_s, slots.hit_rate, state)

        recalled = self.attend.read(attention, state) if self.attend is not None \
            else self.recall(encoded, state, _proximity(world, slots.position, self.sigma_m), slots.alive)
        candidate_logit = self._rescored(encoded, recalled, cand_score)
        slot_logit, coasting = self._coasting(state, stats, slots, events, refill)
        slot_xy = world_to_sensor(slots.position, pose) if self.room_frame else slots.position
        return MemoryOutput(candidate_logit, self._refined(encoded, recalled, cand_xy), slot_logit, slot_xy, coasting, slots, events)


class ConvNeXtBlock(nn.Module):
    """Depthwise large-kernel convolution, normalisation, pointwise widen-GELU-narrow, residual (ConvNeXt, Liu et al., CVPR 2022)."""

    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.widen = nn.Conv1d(channels, expansion * channels, 1)
        self.activation = nn.GELU()
        self.narrow = nn.Conv1d(expansion * channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.narrow(self.activation(self.widen(self.norm(self.depthwise(x)))))


class ResidualBlock(nn.Module):
    """Two full convolutions with normalisation and a GELU between them, residual (He et al., CVPR 2016) -- `TODO.md` item 19.

    The ablation's point is what it does *not* do: no depthwise separation, so the large kernel is spent across every
    channel rather than per channel, and no inverted bottleneck, so `expansion` has nothing to widen and is accepted only
    to keep `_stage`'s signature. A59's W arm priced shrinking the widths at 1.4 pp without buying speed, so capacity is
    not where the score is; whether the *block* is remains untested, and this is the arm that tests it.
    """

    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.second = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.out_norm = nn.GroupNorm(1, channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.out_norm(self.second(self.activation(self.norm(self.first(x)))))


BLOCKS = {"convnext": ConvNeXtBlock, "residual": ResidualBlock}


def metric_offsets(ranges: Tensor, step_rad: float, width_m: float, samples: int) -> Tensor:
    """Where each beam should read, in beams, to span a fixed **metric** width `[..., N] -> [..., N, samples]`.

    A person 0.8 m wide subtends 42 beams at 2 m and 9 at 9 m, so a fixed-width kernel is correct at exactly one range and
    wrong everywhere else. Half the width subtends `atan(width/2 / r)` radians at range `r`, which is that many beams at this
    laser's increment, so the offsets run from `-span` to `+span` and shrink as the range grows (`TODO.md` item 15).
    """
    span = arctan(0.5 * width_m / ranges.clamp(min=1e-3)) / step_rad
    steps = linspace(-1.0, 1.0, samples, device=ranges.device, dtype=ranges.dtype) if samples > 1 \
        else zeros(1, device=ranges.device, dtype=ranges.dtype)
    return span[..., None] * steps


def gather_beams(x: Tensor, offsets: Tensor) -> Tensor:
    """Read `x` `[B, C, N]` at each beam's fractional offsets `[B, N, K]`, linearly interpolated `-> [B, C, N, K]`.

    Past either end of the scan the edge beam repeats: a scan is not periodic, so wrapping would hand the first beam the
    far side of the room as its neighbour.
    """
    b, c, n = x.shape
    index = arange(n, device=x.device, dtype=offsets.dtype)[None, :, None] + offsets
    index = index.clamp(0.0, n - 1.0)
    low = index.floor()
    frac = (index - low).unsqueeze(1)
    low = low.long().clamp(0, n - 1)
    high = (low + 1).clamp(max=n - 1)
    k = offsets.shape[-1]
    left = x.gather(2, low.reshape(b, 1, -1).expand(b, c, n * k)).reshape(b, c, n, k)
    right = x.gather(2, high.reshape(b, 1, -1).expand(b, c, n * k)).reshape(b, c, n, k)
    return left + (right - left) * frac


class MetricGather(nn.Module):
    """A depthwise read over a window of fixed metric width, in place of one of fixed beam count (`TODO.md` item 15).

    Tier 0's 1-NN probe priced the range-adaptive angular window at **+2.5 pp**, and the advantage was **purely
    long-range** -- +4.9 pp at 8-10 m, zero at 4 m, where the fixed window happens to be calibrated. DROW gets this free
    because its cutout is resampled per beam; a full-scan network has to be given it. `width_m` is DROW's own 1.66 m of
    arc by default.

    A59's F1 arm moved the prior against input-*feature* changes -- swapping the ten per-beam channels for LFE's measured
    nothing -- but this changes *which beams the layer reads*, not what each one says.
    """

    def __init__(self, channels: int, width_m: float, samples: int, step_rad: float):
        super().__init__()
        self.width_m, self.samples, self.step_rad = width_m, samples, step_rad
        self.mix = nn.Conv2d(channels, channels, (1, samples), groups=channels)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: Tensor, ranges: Tensor) -> Tensor:
        offsets = metric_offsets(ranges, self.step_rad, self.width_m, self.samples)
        return x + self.norm(self.mix(gather_beams(x, offsets))[..., 0])


def _stage(channels: int, blocks: int, kernel_size: int, expansion: int, block: str = "convnext") -> nn.Sequential:
    return nn.Sequential(*(BLOCKS[block](channels, kernel_size, expansion) for _ in range(blocks)))


class CalibrationNetwork(nn.Module):
    """Stages 1-2 (`docs/PROPOSAL.md` Â§5.3-Â§5.4): one scan in, no state, no pose.

    Per-beam input features read up to `max_range_m` (`sanitize_ranges`); a 1D U-Net of ConvNeXt blocks (widths `channels` at N, N/2
    and N/4 beams, bottleneck at N/8 sectors) with a global max aggregator and a 3-tap convolution across bottleneck sectors
    (`context`); a DROW-style head giving every beam a person logit and a vote to the person's centre. The memory's prior, when the
    network has prior channels, enters at the bottleneck. `dropout` regularises the head's input only, so the decoder features
    stage 3 pools from stay deterministic; it holds no parameters. Neither the per-beam fine nor the bottleneck coarse temporal
    convolution survived its ablation (`TODO.md` A50, A60; `memory/rejected-ideas.md`). `input_features` "lfe" replaces the ten
    beam features with LFE's one input channel (`lfe_features`), for the attribution arm F1 (`TODO.md` A59).
    `gather_width_m` above zero inserts a `MetricGather` after the stem, so the first spatial layer reads a window of that many
    metres of arc instead of a fixed count of beams (`TODO.md` item 15); zero, the default, leaves the network exactly as it was.
    `vote_frame` records which frame the head's two outputs are an offset in -- DROW's beam-relative `(across, along)` by
    default, or the sensor's shared Cartesian frame (`TODO.md` item 18) -- and `block` which block carries the depth,
    ConvNeXt by default or a plain residual one (item 19).
    """

    def __init__(self, angles: Tensor, channels: Tuple[int, int, int] = (32, 64, 128), blocks: Tuple[int, int, int] = (2, 2, 2),
                 bottleneck_blocks: int = 2, kernel_size: int = 7, expansion: int = 4,
                 prior_channels: int = 0, dropout: float = 0.0, max_range_m: float = MAX_RANGE_M, input_features: str = "beam",
                 gather_width_m: float = 0.0, gather_samples: int = 9, vote_frame: str = "beam", block: str = "convnext",
                 aggregator: bool = True):
        super().__init__()
        if input_features not in INPUT_CHANNELS:
            raise ValueError(f"input_features must be one of {sorted(INPUT_CHANNELS)}, not {input_features!r}")
        if vote_frame not in VOTE_FRAMES:
            raise ValueError(f"vote_frame must be one of {list(VOTE_FRAMES)}, not {vote_frame!r}")
        if block not in BLOCKS:
            raise ValueError(f"block must be one of {sorted(BLOCKS)}, not {block!r}")
        c0, c1, c2 = channels
        self.prior_channels = prior_channels
        self.max_range_m = max_range_m
        self.input_features = input_features
        # What the head's two outputs mean, and which block carries the depth. Neither is read inside `forward` -- the
        # frame is a property of the targets the weights were fitted to, and the block has already chosen the modules --
        # but both are recorded so `network_shape` restores a checkpoint into the network it was trained as.
        self.vote_frame, self.block = vote_frame, block
        self.register_buffer("angles", angles.clone(), persistent=False)
        self.stem = nn.Conv1d(INPUT_CHANNELS[input_features], c0, 1)
        # Off unless asked for, and bit-identical when off, so every trained checkpoint keeps reading the same
        # (`TODO.md` item 15). The step comes from the laser's own angles rather than a constant: a different
        # sensor has a different increment, and the window is specified in metres.
        self.gather = MetricGather(c0, gather_width_m, gather_samples, float(angles[1] - angles[0])) \
            if gather_width_m > 0 else None
        self.encoder = nn.ModuleList([_stage(c, n, kernel_size, expansion, block) for c, n in zip(channels, blocks)])
        self.down = nn.ModuleList([nn.Conv1d(c0, c1, 2, stride=2), nn.Conv1d(c1, c2, 2, stride=2), nn.Conv1d(c2, c2, 2, stride=2)])
        # The whole-scan max, and `TODO.md` item 40's ablation of it. It is the one path by which a return at one end of
        # the field of view can reach the other end's logits: the U-Net's receptive field is wide but finite. Named as a
        # defining feature of this architecture in `docs/RESEARCH.md` Â§5.2 and never tested until item 40, which matters
        # because item 19 has since shown the widths flat and the block interchangeable, leaving this as the leading
        # unexplained part of stage 2's lead over LFE-Peaks.
        self.aggregator = aggregator
        pooled = 2 if aggregator else 1
        self.aggregate = nn.Conv1d(pooled * c2 + prior_channels, c2, 1)
        with no_grad():
            self.aggregate.weight[:, pooled * c2:].zero_()   # the memory prior starts silent
        self.context = nn.Conv1d(c2, c2, 3, padding=1)
        self.bottleneck = _stage(c2, bottleneck_blocks, kernel_size, expansion, block)
        self.merge = nn.ModuleList([nn.Conv1d(2 * c2, c2, 1), nn.Conv1d(c2 + c1, c1, 1), nn.Conv1d(c1 + c0, c0, 1)])
        self.decoder = nn.ModuleList([_stage(c, 1, kernel_size, expansion, block) for c in (c2, c1, c0)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv1d(c0, 3, 1)

    def _encode(self, x: Tensor, prior: Optional[Tensor]) -> Tuple[List[Tensor], Tensor]:
        skips = []
        for block, down in zip(self.encoder, self.down):
            x = block(x)
            skips.append(x)
            x = down(x)
        bottom = [x] + ([x.amax(dim=-1, keepdim=True).expand_as(x)] if self.aggregator else []) \
            + ([prior] if self.prior_channels else [])
        return skips, self.aggregate(cat(bottom, dim=1))

    def _decode(self, bottom: Tensor, skips: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.bottleneck(bottom)
        for merge, block, skip in zip(self.merge, self.decoder, reversed(skips)):
            x = block(merge(cat([interpolate(x, scale_factor=2), skip], dim=1)))
        out = self.head(self.dropout(x))
        return out[:, 0], out[:, 1:].transpose(1, 2), x

    def forward(self, ranges: Tensor, prior: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Person logits `[B, N]`, votes `[B, N, 2]` and decoder features `[B, C, N]` for scans `[B, N]`, with the memory prior
        `[B, prior_channels, N/8]` when the network has prior channels."""
        x = lfe_features(ranges, self.max_range_m) if self.input_features == "lfe" else beam_features(ranges, self.angles, self.max_range_m)
        lifted = self.stem(x)
        if self.gather is not None:
            lifted = self.gather(lifted, sanitize_ranges(ranges, self.max_range_m))
        skips, bottom = self._encode(lifted, prior)
        return self._decode(self.context(bottom), skips)


def fold_legacy_calibration(state: dict) -> dict:
    """A calibration checkpoint from an earlier architecture in the current one; a current one is returned as it is.

    Checkpoints from before 2026-09-17 carry an 11-channel stem, whose input `LEGACY_VALIDITY_CHANNEL` flagged valid returns, and a
    fine temporal convolution after it. With one fine tap (`--fine-lags 0`, the no-fine and static arms) the fine convolution is a
    1x1 convolution straight after the 1x1 stem, so the two compose into one exactly. The validity input was 1 on every reading
    those networks saw, because the FROG loader had already clamped every missing return, so its weights fold into the bias.
    Checkpoints from before 2026-10-08 carry the bottleneck's convolution as `coarse.conv`; with one lag (`--coarse-lags 0`, every
    no-temporal stage 2 that stage 3 was trained on) it is exactly today's `context`. A fine or coarse convolution that read past
    frames has no equivalent any more, and runs at commit `LAST_TEMPORAL_COMMIT`.
    """
    folded = state
    if "fine.conv.weight" in state:
        stem_w, stem_b = state["stem.weight"][:, :, 0], state["stem.bias"]
        fine_w, fine_b = state["fine.conv.weight"][:, :, 0], state["fine.conv.bias"]
        if fine_w.shape[1] != stem_w.shape[0]:
            raise ValueError(f"this checkpoint's fine convolution reads {fine_w.shape[1] // stem_w.shape[0]} frames; the fine convolution was removed "
                             f"2026-09-17 (TODO.md A50) and only one-tap checkpoints (--fine-lags 0) fold into the current network")
        weight, bias = fine_w @ stem_w, fine_w @ stem_b + fine_b
        if weight.shape[1] == N_BEAM_FEATURES + 1:
            bias = bias + weight[:, LEGACY_VALIDITY_CHANNEL]
            weight = cat([weight[:, :LEGACY_VALIDITY_CHANNEL], weight[:, LEGACY_VALIDITY_CHANNEL + 1:]], dim=1)
        folded = {k: v for k, v in folded.items() if not k.startswith("fine.")}
        folded["stem.weight"], folded["stem.bias"] = weight[:, :, None], bias
    if "coarse.conv.weight" in folded:
        weight = folded["coarse.conv.weight"]
        if weight.shape[1] != weight.shape[0]:
            raise ValueError(f"this checkpoint's coarse convolution reads past frames ({weight.shape[1]} inputs for {weight.shape[0]} outputs); "
                             f"it was removed 2026-10-08 (TODO.md A60) and runs at commit {LAST_TEMPORAL_COMMIT}")
        folded = {("context." + k[len("coarse.conv."):] if k.startswith("coarse.conv.") else k): v for k, v in folded.items()}
    return folded


def load_calibration_weights(net: CalibrationNetwork, state: dict) -> None:
    """Load a calibration network's weights into `net`, which may have prior channels the checkpoint has not: their weights start at
    zero, so `net` computes exactly what the checkpoint did until they are trained. Checkpoints from earlier architectures are folded
    first (`fold_legacy_calibration`)."""
    state = fold_legacy_calibration(state)
    own = net.state_dict()["aggregate.weight"]
    if state["aggregate.weight"].shape != own.shape:
        padded = zeros_like(own)
        padded[:, :state["aggregate.weight"].shape[1]] = state["aggregate.weight"]
        state = {**state, "aggregate.weight": padded}
    net.load_state_dict(state)


def free_range_profile(ranges: Tensor, sectors: int, max_range_m: float = MAX_RANGE_M) -> Tensor:
    """The nearest thing the laser actually saw per angular sector, `[B, S]`, from raw ranges `[B, N]`.

    Readings that mean "nothing within range" go through `sanitize_ranges` first, so a missing return reads as empty space
    rather than as the closest obstacle in its sector -- the opposite of what a raw minimum would say.
    """
    clean = sanitize_ranges(ranges, max_range_m)
    return clean.reshape(clean.shape[0], sectors, clean.shape[-1] // sectors).amin(dim=-1)


def visibility_feature(slot_xy: Tensor, free_range: Tensor, angles: Tensor, sectors: int, scale: float = VISIBILITY_SCALE_M) -> Tensor:
    """Per slot `[B, K]`: how much further the laser reached than the slot claims to be, scaled and clamped to `[-1, 1]`.

    Negative means something is in front of the slot, so a person there could still be hidden; positive means the beam passed
    through the slot's own position and found nothing, which is direct evidence against it. Slots outside the field of view read
    0 -- nothing known rather than either. This is the evidence the coasting head was missing (`TODO.md` A51 S2): it must tell a
    genuinely occluded person from a slot that is simply wrong, and until now it saw nothing of the current scan at all.
    """
    width = (angles[1] - angles[0]) * len(angles) / sectors
    index = ((atan2(-slot_xy[..., 0], slot_xy[..., 1]) - angles[0]) / width).floor().long()
    inside = (index >= 0) & (index < sectors)
    seen = free_range.gather(1, where(inside, index, zeros_like(index)))
    margin = (seen - slot_xy.norm(dim=-1)) / scale
    return where(inside, margin.clamp(-1.0, 1.0), zeros_like(margin))


def render_prior(slot_xy: Tensor, value: Tensor, person: Tensor, alive: Tensor, angles: Tensor, sectors: int) -> Tensor:
    """The memory drawn onto stage 2's bottleneck sectors, `[B, 2, S]`: per sector, the largest value and the largest person
    probability among live slots whose detection-frame position `[B, K, 2]` falls in it; slots outside the field of view draw nothing."""
    width = (angles[1] - angles[0]) * len(angles) / sectors
    index = ((atan2(-slot_xy[..., 0], slot_xy[..., 1]) - angles[0]) / width).floor().long()
    inside = alive & (index >= 0) & (index < sectors)
    target = where(inside, index, zeros_like(index))
    empty = zeros(slot_xy.shape[0], sectors, device=slot_xy.device, dtype=slot_xy.dtype)
    return stack([empty.scatter_reduce(1, target, where(inside, v, zeros_like(v)), reduce="amax") for v in (value, person)], dim=1)
