"""Three-horizon streaming person detector: input features, sector alignment and causal temporal convolution.

Design and reasoning in `docs/PROPOSAL.md` (§5.3 input, §5.4 calibration horizon); build status in `TODO.md` A43.
Conventions: beam angle 0 is forward, counterclockwise positive, increasing with beam index;
poses are world odometry `(x, y, theta)` with x forward at theta = 0, y left, theta counterclockwise.
"""
from dataclasses import dataclass, fields
from math import log, radians, sqrt
from typing import List, Optional, Sequence, Tuple

from torch import (
    Tensor, arange, argsort, atan2, cat, cos, cumprod, exp, expm1, full, full_like, isfinite, linspace, log1p, minimum, nn,
    no_grad, ones, sin, stack, tanh, where, zeros, zeros_like,
)
from torch import log as tlog
from torch.nn.functional import interpolate, pad, softmax, softplus

MAX_RANGE_M = 10.0
MEDIAN_WINDOW_DEG = 11.0
NEIGHBOUR_OFFSETS = (-2, -1, 1, 2)
N_BEAM_FEATURES = 2 + 2 * len(NEIGHBOUR_OFFSETS)
LEGACY_VALIDITY_CHANNEL = 2          # index of the validity input that checkpoints before 2026-09-17 carried
POSE_CHANNELS = 3
SLOT_STATISTICS = 8
CANDIDATE_FEATURES = 5
PROBABILITY_EPS = 1e-6
SILENT_COAST_LOGIT = -10.0
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


def align_sectors(features: Tensor, pose_from: Tensor, pose_to: Tensor, angles: Tensor) -> Tensor:
    """Shift angular-sector features `[B, C, S]` of `pose_from` to undo the rotation to `pose_to`, and append the past pose
    expressed in the current frame `(forward, left, heading change)` as 3 constant channels, `-> [B, C + 3, S]`.
    """
    b, c, s = features.shape
    width = (angles[1] - angles[0]) * len(angles) / s
    xf, yf, tf = pose_from.unbind(-1)
    xt, yt, tt = pose_to.unbind(-1)
    turn = _wrap(tf - tt)
    source = arange(s, device=features.device)[None] - (turn / width).round().long()[:, None]
    inside = (source >= 0) & (source < s)
    shifted = features.gather(2, source.clamp(0, s - 1)[:, None, :].expand(-1, c, -1)) * inside[:, None, :]
    dx, dy = xf - xt, yf - yt
    pose = stack([dx * cos(tt) + dy * sin(tt), -dx * sin(tt) + dy * cos(tt), turn], dim=-1)
    return cat([shifted, pose[:, :, None].expand(-1, -1, s)], dim=1)


@dataclass
class TemporalCache:
    """The frames a causal temporal convolution still reads, oldest first: features and poses."""
    features: List[Tensor]
    poses: List[Tensor]


class CausalTemporalConv(nn.Module):
    """A convolution over the current frame and frames `lags` steps back, each re-aligned to the current pose.

    `forward` runs a clip `[B, T, C, S]` of angular-sector features for training; `step` runs one frame with a `TemporalCache`
    and returns exactly the clip's output for that frame. Frames before the first read the first. Past frames are rotated onto
    the current sectors and carry their pose in the current frame as 3 extra channels (`align_sectors`).
    """

    def __init__(self, in_channels: int, out_channels: int, lags: Sequence[int], angles: Tensor, kernel_size: int = 1):
        super().__init__()
        self.lags = list(lags)
        self.register_buffer("angles", angles.clone(), persistent=False)
        pose_channels = POSE_CHANNELS * sum(1 for lag in self.lags if lag)
        self.conv = nn.Conv1d(in_channels * len(self.lags) + pose_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def _convolve(self, taps: Sequence[Tuple[int, Tensor, Tensor]], pose_now: Tensor) -> Tensor:
        parts = [x if lag == 0 else align_sectors(x, pose, pose_now, self.angles) for lag, x, pose in taps]
        return self.conv(cat(parts, dim=1))

    def forward(self, x: Tensor, poses: Tensor) -> Tensor:
        b, t = x.shape[:2]
        now = arange(t, device=x.device)
        taps = []
        for lag in self.lags:
            src = (now - lag).clamp(min=0)
            taps.append((lag, x[:, src].flatten(0, 1), poses[:, src].flatten(0, 1)))
        y = self._convolve(taps, poses.flatten(0, 1))
        return y.view(b, t, *y.shape[1:])

    def step(self, x: Tensor, pose: Tensor, cache: Optional[TemporalCache]) -> Tuple[Tensor, TemporalCache]:
        keep = max(self.lags) + 1
        previous = cache or TemporalCache([], [])
        cache = TemporalCache((previous.features + [x])[-keep:], (previous.poses + [pose])[-keep:])
        last = len(cache.features) - 1
        taps = [(lag, cache.features[max(last - lag, 0)], cache.poses[max(last - lag, 0)]) for lag in self.lags]
        return self._convolve(taps, pose), cache


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
    """Fixed rules for creating, matching, replacing and retiring object slots (`docs/PROPOSAL.md` §5.5 and §6.3)."""
    capacity: int = 256
    floor: float = 0.01
    margin: float = 0.1
    fade_time_s: float = 10.0
    gain: float = 0.1
    gate_m: float = 0.6
    velocity_smoothing: float = 0.1


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
    predicted = slots.position + slots.velocity * step[..., None] * alive[..., None]
    usable = cand_valid & (cand_score >= rules.floor)

    dist = (predicted[:, :, None, :] - cand_xy[:, None, :, :]).norm(dim=-1)
    dist = dist.masked_fill(~(alive[:, :, None] & usable[:, None, :]), float("inf"))
    nearest_candidate = dist.argmin(dim=2)
    mutual = dist.argmin(dim=1).gather(1, nearest_candidate) == arange(k, device=device)[None]
    matched = alive & mutual & (dist.gather(2, nearest_candidate[..., None])[..., 0] <= rules.gate_m)

    match_xy = cand_xy.gather(1, nearest_candidate[..., None].expand(-1, -1, 2))
    fade = exp(-step * log(1.0 / rules.floor) / rules.fade_time_s)
    value = where(matched, slots.value + rules.gain * (1.0 - slots.value) * cand_score.gather(1, nearest_candidate), slots.value * fade)
    velocity = where(matched[..., None], slots.velocity + rules.velocity_smoothing * ((match_xy - slots.position) / step[..., None] - slots.velocity), slots.velocity)
    position = where(matched[..., None], match_xy, predicted)
    hit_rate = slots.hit_rate + (1.0 - exp(-step / HIT_RATE_TIME_S)) * (matched.float() - slots.hit_rate)
    since = where(matched, zeros_like(slots.since_match_s), slots.since_match_s + step)
    age = slots.age_s + step
    alive = alive & (value >= rules.floor)

    claimed = zeros(b, p, device=device).scatter_add(1, nearest_candidate, matched.float()) > 0
    eligible = usable & ~claimed
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


@dataclass
class MemoryOutput:
    """One object-memory step: rescored candidate logits `[B, P]`; person logits of slots `[B, K]` and their positions in the
    current detection frame `[B, K, 2]`; which slots may report a person no candidate covers (`coasting`); the new memory."""
    candidate_logit: Tensor
    slot_logit: Tensor
    slot_xy: Tensor
    coasting: Tensor
    slots: Slots


def _logit(p: Tensor) -> Tensor:
    p = p.clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return tlog(p) - log1p(-p)


def _proximity(a: Tensor, b: Tensor, sigma_m: float) -> Tensor:
    """Attention bias `-|a_i - b_j|^2 / (2 sigma^2)` between point sets `[B, I, 2]` and `[B, J, 2]`."""
    return -((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(dim=-1) / (2.0 * sigma_m ** 2)


class ObjectMemory(nn.Module):
    """Stage 3, the short and long horizons (`docs/PROPOSAL.md` §5.5): object slots in world coordinates, learned matching of
    candidates to slots, exchange between slots, selective recurrence in real time, rule-based slot bookkeeping, candidate
    rescoring through a gate that starts closed, and coasting slots that report people no candidate covers.
    """

    def __init__(self, rules: SlotRules = SlotRules(), dim: int = 128, heads: int = 4, sigma_m: float = 0.6, decay_init_s: Tuple[float, float] = (1.0, 60.0),
                 feature_dim: int = 0):
        super().__init__()
        self.rules, self.dim, self.sigma_m, self.feature_dim = rules, dim, sigma_m, feature_dim
        self.encode = nn.Sequential(nn.Linear(CANDIDATE_FEATURES + feature_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.spawn = nn.Linear(dim, dim)
        self.statistics = nn.Linear(SLOT_STATISTICS, dim)
        self.match = _BiasedAttention(dim, heads)
        self.exchange = _BiasedAttention(dim, heads)
        self.exchange_norm = nn.LayerNorm(dim)
        self.update = nn.Linear(dim + SLOT_STATISTICS, dim)
        self.decay = nn.Linear(dim + SLOT_STATISTICS, dim, bias=False)
        times = exp(linspace(log(decay_init_s[0]), log(decay_init_s[1]), dim))
        self.decay_bias = nn.Parameter(tlog(expm1(1.0 / times)))
        self.recall = _BiasedAttention(dim, heads)
        self.rescore = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.rescore_gate = nn.Parameter(zeros(()))
        self.coast = nn.Linear(dim + SLOT_STATISTICS, 1)
        nn.init.zeros_(self.coast.weight)
        nn.init.constant_(self.coast.bias, SILENT_COAST_LOGIT)

    def initial_state(self, batch: int, device=None) -> Slots:
        return initial_slots(batch, self.rules, self.dim, device)

    def step(self, cand_xy: Tensor, cand_score: Tensor, cand_valid: Tensor, pose: Tensor, dt: Tensor, slots: Slots, reset: Optional[Tensor] = None,
             cand_features: Optional[Tensor] = None) -> MemoryOutput:
        """One frame: candidates `[B, P, 2]` in the detection frame with scores and validity, robot pose `[B, 3]`, seconds since
        the previous frame `[B]`; `reset` `[B]` empties those streams' memory first; `cand_features` `[B, P, feature_dim]` are the
        detector's own features of each candidate, when the memory was built with `feature_dim`."""
        if reset is not None:
            slots = _where_slots(reset, self.initial_state(len(reset), cand_xy.device), slots)
        world = sensor_to_world(cand_xy, pose)
        slots, events = manage_slots(slots, world, cand_score, cand_valid, dt, self.rules)

        features = stack([cand_score, _logit(cand_score) / FEATURE_SCALE, cand_xy[..., 0] / FEATURE_SCALE, cand_xy[..., 1] / FEATURE_SCALE, cand_xy.norm(dim=-1) / FEATURE_SCALE], dim=-1)
        if self.feature_dim:
            features = cat([features, cand_features], dim=-1)
        encoded = self.encode(features)
        refill = events.source >= 0
        spawned = tanh(self.spawn(encoded.gather(1, events.source.clamp(min=0)[..., None].expand(-1, -1, self.dim))))
        state = where(refill[..., None], spawned, slots.state)

        observed_score = where(refill, cand_score.gather(1, events.source.clamp(min=0)), cand_score.gather(1, events.match_index.clamp(min=0)) * events.matched)
        stats = stack([slots.value, slots.age_s / FEATURE_SCALE, slots.since_match_s / FEATURE_SCALE, (slots.position - slots.origin).norm(dim=-1),
                       slots.velocity.norm(dim=-1), slots.hit_rate, (events.matched | refill).float(), observed_score], dim=-1)

        found = self.match(state + self.statistics(stats), encoded, _proximity(slots.position, world, self.sigma_m), cand_valid)
        mixed = self.exchange_norm(found + self.exchange(found, found, _proximity(slots.position, slots.position, self.sigma_m), slots.alive))
        u = cat([mixed, stats], dim=-1)
        keep = exp(-dt[:, None, None] * softplus(self.decay(u) + self.decay_bias))
        state = (keep * state + (1.0 - keep) * tanh(self.update(u))) * slots.alive[..., None]
        slots = Slots(slots.alive, slots.value, slots.position, slots.velocity, slots.origin, slots.age_s, slots.since_match_s, slots.hit_rate, state)

        recalled = self.recall(encoded, state, _proximity(world, slots.position, self.sigma_m), slots.alive)
        candidate_logit = _logit(cand_score) + self.rescore_gate * self.rescore(cat([encoded, recalled], dim=-1))[..., 0]
        slot_logit = self.coast(cat([state, stats], dim=-1))[..., 0]
        coasting = slots.alive & ~events.matched & ~refill
        return MemoryOutput(candidate_logit, slot_logit, world_to_sensor(slots.position, pose), coasting, slots)


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


def _stage(channels: int, blocks: int, kernel_size: int, expansion: int) -> nn.Sequential:
    return nn.Sequential(*(ConvNeXtBlock(channels, kernel_size, expansion) for _ in range(blocks)))


@dataclass
class CalibrationState:
    """Streaming cache of the calibration network: the bottleneck features the coarse convolution reads."""
    coarse: TemporalCache


class CalibrationNetwork(nn.Module):
    """Stages 1-2, the calibration horizon (`docs/PROPOSAL.md` §5.3-§5.4).

    Per-beam input features read up to `max_range_m` (`sanitize_ranges`), with no per-beam temporal convolution: that was tested
    and found unneeded, since the sensors' jitter is far below the match radius (`TODO.md` A50, `memory/noise-structure.md`);
    a 1D U-Net of ConvNeXt blocks (widths `channels` at
    N, N/2, N/4 beams, bottleneck at N/8 sectors) with a global max aggregator and a coarse causal space-time convolution at the
    bottleneck; a DROW-style head giving every beam a person logit and a vote to the person's centre. `forward` scores the last
    frame of a clip `[B, T, N]` and encodes only the frames the temporal taps read; `step` scores one frame from its caches, the
    same computation. `dropout` regularises the head's input only, so the decoder features stage 3 pools from stay
    deterministic; it holds no parameters, so a checkpoint trained at any rate loads into a network built with any other.
    """

    def __init__(self, angles: Tensor, channels: Tuple[int, int, int] = (32, 64, 128), blocks: Tuple[int, int, int] = (2, 2, 2),
                 bottleneck_blocks: int = 2, kernel_size: int = 7, expansion: int = 4,
                 coarse_lags: Sequence[int] = (0, 8, 16, 24, 32), prior_channels: int = 0, dropout: float = 0.0, max_range_m: float = MAX_RANGE_M):
        super().__init__()
        c0, c1, c2 = channels
        self.prior_channels = prior_channels
        self.max_range_m = max_range_m
        self.register_buffer("angles", angles.clone(), persistent=False)
        self.stem = nn.Conv1d(N_BEAM_FEATURES, c0, 1)
        self.encoder = nn.ModuleList([_stage(c, n, kernel_size, expansion) for c, n in zip(channels, blocks)])
        self.down = nn.ModuleList([nn.Conv1d(c0, c1, 2, stride=2), nn.Conv1d(c1, c2, 2, stride=2), nn.Conv1d(c2, c2, 2, stride=2)])
        self.aggregate = nn.Conv1d(2 * c2 + prior_channels, c2, 1)
        with no_grad():
            self.aggregate.weight[:, 2 * c2:].zero_()        # the memory prior starts silent
        self.coarse = CausalTemporalConv(c2, c2, coarse_lags, angles, kernel_size=3)
        self.bottleneck = _stage(c2, bottleneck_blocks, kernel_size, expansion)
        self.merge = nn.ModuleList([nn.Conv1d(2 * c2, c2, 1), nn.Conv1d(c2 + c1, c1, 1), nn.Conv1d(c1 + c0, c0, 1)])
        self.decoder = nn.ModuleList([_stage(c, 1, kernel_size, expansion) for c in (c2, c1, c0)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv1d(c0, 3, 1)

    def _encode(self, x: Tensor, prior: Optional[Tensor]) -> Tuple[List[Tensor], Tensor]:
        skips = []
        for block, down in zip(self.encoder, self.down):
            x = block(x)
            skips.append(x)
            x = down(x)
        bottom = [x, x.amax(dim=-1, keepdim=True).expand_as(x)] + ([prior] if self.prior_channels else [])
        return skips, self.aggregate(cat(bottom, dim=1))

    def _decode(self, bottom: Tensor, skips: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.bottleneck(bottom)
        for merge, block, skip in zip(self.merge, self.decoder, reversed(skips)):
            x = block(merge(cat([interpolate(x, scale_factor=2), skip], dim=1)))
        out = self.head(self.dropout(x))
        return out[:, 0], out[:, 1:].transpose(1, 2), x

    def forward(self, ranges: Tensor, poses: Tensor, prior: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Person logits `[B, N]`, votes `[B, N, 2]` and decoder features `[B, C, N]` for the last frame of `ranges` `[B, T, N]` at
        `poses` `[B, T, 3]`, with the memory prior `[B, T, prior_channels, N/8]` of every frame when the network has prior channels."""
        b = ranges.shape[0]
        last = ranges.shape[1] - 1
        tapped = [max(last - lag, 0) for lag in self.coarse.lags]
        stems = cat([self.stem(beam_features(ranges[:, g], self.angles, self.max_range_m)) for g in tapped], dim=0)
        skips, bottom = self._encode(stems, cat([prior[:, g] for g in tapped], dim=0) if self.prior_channels else None)
        bottom = bottom.view(len(tapped), b, *bottom.shape[1:])
        coarse = self.coarse._convolve([(lag, bottom[j], poses[:, g]) for j, (lag, g) in enumerate(zip(self.coarse.lags, tapped))], poses[:, last])
        now = self.coarse.lags.index(0)
        return self._decode(coarse, [s[now * b:(now + 1) * b] for s in skips])

    def step(self, ranges: Tensor, pose: Tensor, state: Optional[CalibrationState], prior: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, CalibrationState]:
        """One frame `[B, N]` at `pose` `[B, 3]`, with its memory prior `[B, prior_channels, N/8]`: person logits, votes, decoder features
        and the updated caches."""
        skips, bottom = self._encode(self.stem(beam_features(ranges, self.angles, self.max_range_m)), prior)
        coarse, coarse_cache = self.coarse.step(bottom, pose, None if state is None else state.coarse)
        logit, votes, features = self._decode(coarse, skips)
        return logit, votes, features, CalibrationState(coarse_cache)


def fold_legacy_calibration(state: dict) -> dict:
    """A calibration checkpoint from before 2026-09-17 in the current architecture; a current one is returned as it is.

    Those checkpoints carry an 11-channel stem, whose input `LEGACY_VALIDITY_CHANNEL` flagged valid returns, and a fine temporal
    convolution after it. With one fine tap (`--fine-lags 0`, the no-fine and static arms) the fine convolution is a 1x1
    convolution straight after the 1x1 stem, so the two compose into one exactly. The validity input was 1 on every reading
    those networks saw, because the FROG loader had already clamped every missing return, so its weights fold into the bias.
    A multi-tap fine convolution read past frames and has no equivalent any more.
    """
    if "fine.conv.weight" not in state:
        return state
    stem_w, stem_b = state["stem.weight"][:, :, 0], state["stem.bias"]
    fine_w, fine_b = state["fine.conv.weight"][:, :, 0], state["fine.conv.bias"]
    if fine_w.shape[1] != stem_w.shape[0]:
        raise ValueError(f"this checkpoint's fine convolution reads {fine_w.shape[1] // stem_w.shape[0]} frames; the fine convolution was removed "
                         "2026-09-17 (TODO.md A50) and only one-tap checkpoints (--fine-lags 0) fold into the current network")
    weight, bias = fine_w @ stem_w, fine_w @ stem_b + fine_b
    if weight.shape[1] == N_BEAM_FEATURES + 1:
        bias = bias + weight[:, LEGACY_VALIDITY_CHANNEL]
        weight = cat([weight[:, :LEGACY_VALIDITY_CHANNEL], weight[:, LEGACY_VALIDITY_CHANNEL + 1:]], dim=1)
    folded = {k: v for k, v in state.items() if not k.startswith("fine.")}
    folded["stem.weight"], folded["stem.bias"] = weight[:, :, None], bias
    return folded


def load_calibration_weights(net: CalibrationNetwork, state: dict) -> None:
    """Load a calibration network's weights into `net`, which may have prior channels the checkpoint has not: their weights start at
    zero, so `net` computes exactly what the checkpoint did until they are trained. Checkpoints from before the fine convolution's
    removal are folded first (`fold_legacy_calibration`)."""
    state = fold_legacy_calibration(state)
    own = net.state_dict()["aggregate.weight"]
    if state["aggregate.weight"].shape != own.shape:
        padded = zeros_like(own)
        padded[:, :state["aggregate.weight"].shape[1]] = state["aggregate.weight"]
        state = {**state, "aggregate.weight": padded}
    net.load_state_dict(state)


def render_prior(slot_xy: Tensor, value: Tensor, person: Tensor, alive: Tensor, angles: Tensor, sectors: int) -> Tensor:
    """The memory drawn onto stage 2's bottleneck sectors, `[B, 2, S]`: per sector, the largest value and the largest person
    probability among live slots whose detection-frame position `[B, K, 2]` falls in it; slots outside the field of view draw nothing."""
    width = (angles[1] - angles[0]) * len(angles) / sectors
    index = ((atan2(-slot_xy[..., 0], slot_xy[..., 1]) - angles[0]) / width).floor().long()
    inside = alive & (index >= 0) & (index < sectors)
    target = where(inside, index, zeros_like(index))
    empty = zeros(slot_xy.shape[0], sectors, device=slot_xy.device, dtype=slot_xy.dtype)
    return stack([empty.scatter_reduce(1, target, where(inside, v, zeros_like(v)), reduce="amax") for v in (value, person)], dim=1)
