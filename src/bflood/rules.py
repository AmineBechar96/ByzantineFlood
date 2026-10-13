"""Fusion rules: ByzantineFlood v2, the v1 rule as published, and baselines.

Every rule implements
    reset(graph)                          -- clear per-node state
    payload_from_belief(p)               -- what a sender transmits
    decide(v, own, received, hop) -> (level, payload_out)
    alarm_score(v, own, received, hop) -> float in [0, 1]
`own` and the values of `received` are payloads; for belief rules a payload
is a probability vector, for W-MSR it is a scalar level.

alarm_score is the rule's own evidence that v is not dry, computed from the
same inputs as decide (threshold sweep, exp10). It never changes the rule
state, so the simulator calls it before decide and both see the same state.
Per rule:
    local            1 - p_0 of the own belief
    unfiltered_mean  mean over the inputs (own + received) of 1 - p_0,
                     unweighted
    majority         fraction of the input levels (own + received) >= 1
    coord_median     1 - p_0 of the coordinatewise median belief,
                     renormalised (the rule's own aggregate)
    wmsr             trimmed average level / 4 (the rule's own value x_new)
    bf_v1            1 - p_0 of its filtered temporal mean (the aggregate it
                     would store, computed with the state before this round)
    bf_v2            1 - p_0 of the weighted mixture of the inputs that
                     survive trimming (own included; weights 1 + gamma, own
                     times alpha); the own belief with fewer than
                     min_inputs inputs
    oracle           1 - p_0 of the same weighted mixture over the own
                     belief and the honest inputs only (no trimming)

Flags read by the worst-case adversary AW (sim.py):
    stateful       decide changes state that later decisions read (bf_v1;
                   bf_v2 with use_trust); AW searches on deep copies
    uses_messages  False for rules that ignore neighbours (local)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .levels import (N_LEVELS, confidence, hellinger, normalise, ordinal_median,
                     sym_kl, weighted_median_level)


class Rule:
    name = "rule"
    scalar = False
    stateful = False
    uses_messages = True

    def reset(self, g):
        pass

    def payload_from_belief(self, p):
        return normalise(p)

    def decide(self, v, own, received, hop):
        raise NotImplementedError

    def alarm_score(self, v, own, received, hop) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------- v2 ----
class ByzantineFloodV2(Rule):
    """Own-centred trimmed, confidence-weighted median of ordinal levels.

    Paper rule (decided 5 Oct 2026 from the exp5 ablation): trim F = 2,
    confidence weights, own weight alpha = 1.5, NO distance filter and NO
    trust (both added nothing measurable; the guarantees do not need them).
    With fewer than `min_inputs` inputs the node outputs its own level.

    Optional, for the ablation only (`use_filter=True`, `use_trust=True`):
    Hellinger filter with threshold median + max(mad_k * MAD, tau_min) (own
    belief never filtered); trust multiplied by `decay` when filtered out,
    +`recover` when kept (hop 1 only).
    """

    name = "bf_v2"

    def __init__(self, alpha=1.5, mad_k=3.0, tau_min=0.15, decay=0.5, recover=0.25,
                 min_inputs=3, use_filter=False, use_trust=False, use_conf=True,
                 distance="hellinger", trim=2, output="median"):
        self.alpha, self.mad_k, self.tau_min = alpha, mad_k, tau_min
        self.decay, self.recover, self.min_inputs = decay, recover, min_inputs
        self.use_filter, self.use_trust, self.use_conf = use_filter, use_trust, use_conf
        self.dist = {"hellinger": hellinger, "sym_kl": sym_kl}[distance]
        # Own-centred trimming (W-MSR style) before the median: drop up to
        # `trim` neighbour levels strictly above the own level (the highest)
        # and up to `trim` strictly below it (the lowest). With at most
        # `trim` Byzantine neighbours every kept level lies within the honest
        # observations, whatever the weights (see tests/test_rules.py).
        # trim="auto": per node, F_v = floor((d_v - 1) / 2) with d_v the
        # number of neighbour reports received (the neighbour-majority bound).
        self.trim = trim if trim == "auto" else int(trim)
        # output="median": weighted median of the kept ordinal levels (as in
        # the paper so far). output="mixture": ordinal median of the weighted
        # mixture of the kept beliefs. Both stay in the range of the kept
        # levels: if every kept belief has its ordinal median in [a, b], its
        # CDF is < 1/2 below a and >= 1/2 at b, and so is any mixture.
        if output not in ("median", "mixture"):
            raise ValueError("output must be 'median' or 'mixture'")
        self.output = output

    @property
    def stateful(self):
        # trust is the only state later decisions read (last_kept is a
        # diagnostic record)
        return bool(self.use_trust)

    def reset(self, g):
        self.trust = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.last_kept = {}

    def _filter(self, beliefs):
        m = len(beliefs)
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                D[i, j] = D[j, i] = self.dist(beliefs[i], beliefs[j])
        dbar = np.array([np.median(np.delete(D[j], j)) for j in range(m)])
        med = np.median(dbar)
        mad = 1.4826 * np.median(np.abs(dbar - med))
        theta = med + max(self.mad_k * mad, self.tau_min)
        keep = dbar <= theta
        keep[0] = True
        return keep

    def _fuse(self, v, own, received):
        """The fusion without state update: (level, mixture, ids, keep,
        fused). keep is None with fewer than min_inputs inputs; fused is
        False when the own level is returned without fusion."""
        ids = list(received)
        beliefs = [own] + [received[u] for u in ids]
        m = len(beliefs)
        if m < self.min_inputs:
            return ordinal_median(own), own, ids, None, False
        keep = self._filter(beliefs) if self.use_filter else np.ones(m, dtype=bool)
        lv, w, kept_b = [], [], []
        for j in range(m):
            if not keep[j]:
                continue
            b = beliefs[j]
            wj = (1.0 + confidence(b)) if self.use_conf else 1.0
            wj *= self.alpha if j == 0 else self.trust[v][ids[j - 1]]
            lv.append(ordinal_median(b))
            w.append(wj)
            kept_b.append(b)
        F = (max(0, (m - 2) // 2) if self.trim == "auto" else self.trim)
        if F > 0:
            o = lv[0]  # own belief is always kept and comes first
            nb = list(range(1, len(lv)))
            above = sorted((j for j in nb if lv[j] > o), key=lambda j: -lv[j])[:F]
            below = sorted((j for j in nb if lv[j] < o), key=lambda j: lv[j])[:F]
            drop = set(above) | set(below)
            keep_idx = [j for j in range(len(lv)) if j not in drop]
            lv = [lv[j] for j in keep_idx]
            w = [w[j] for j in keep_idx]
            kept_b = [kept_b[j] for j in keep_idx]
        w = np.asarray(w)
        if w.sum() <= 0:  # all neighbours fully distrusted; own weight > 0 always
            return ordinal_median(own), own, ids, keep, False
        agg = normalise(np.sum(np.array(kept_b) * w[:, None], axis=0))
        level = ordinal_median(agg) if self.output == "mixture" else weighted_median_level(lv, w)
        return level, agg, ids, keep, True

    def decide(self, v, own, received, hop):
        level, agg, ids, keep, fused = self._fuse(v, own, received)
        if keep is None:
            self.last_kept[v] = {}
            return level, agg
        if not fused:
            return level, agg
        if hop == 1:
            self.last_kept[v] = {u: bool(keep[j + 1]) for j, u in enumerate(ids)}
            if self.use_trust and self.use_filter:
                for j, u in enumerate(ids):
                    t = self.trust[v][u]
                    self.trust[v][u] = min(1.0, t + self.recover) if keep[j + 1] else t * self.decay
        return level, agg

    def alarm_score(self, v, own, received, hop):
        return float(1.0 - self._fuse(v, own, received)[1][0])


# ---------------------------------------------------------------- v1 ----
class ByzantineFloodV1(Rule):
    """The v1 rule as published (Algorithm 1): symmetric KL, median + 2 std,
    unfiltered fallback, temporal factor, argmax output. For comparison only."""

    name = "bf_v1"
    stateful = True  # the temporal factor reads the previous aggregate

    def __init__(self, alpha=1.5):
        self.alpha = alpha

    def reset(self, g):
        self.prev = {}

    def _fuse(self, v, own, received):
        """The aggregate, without storing it."""
        beliefs = [own] + list(received.values())
        m = len(beliefs)
        keep = np.ones(m, dtype=bool)
        if m >= 3:
            D = np.array([[sym_kl(a, b) for b in beliefs] for a in beliefs])
            dbar = np.array([np.median(np.delete(D[j], j)) for j in range(m)])
            sd = np.std(dbar)
            theta = np.median(dbar) + (2 * sd if sd > 0 else 0.5)
            keep = dbar <= theta
            if keep.sum() < 2:
                keep[:] = True
        w = []
        for j in range(m):
            wj = 1.0 + confidence(beliefs[j])
            if j == 0:
                wj *= self.alpha
            if v in self.prev:
                wj *= max(0.5, 1.0 - sym_kl(beliefs[j], self.prev[v]))
            w.append(max(0.01, wj) if keep[j] else 0.0)
        w = np.asarray(w)
        return normalise(np.sum(np.array(beliefs) * w[:, None], axis=0))

    def decide(self, v, own, received, hop):
        agg = self._fuse(v, own, received)
        if hop == 1:
            self.prev[v] = agg
        return int(np.argmax(agg)), agg

    def alarm_score(self, v, own, received, hop):
        return float(1.0 - self._fuse(v, own, received)[0])


# ----------------------------------------------------------- baselines --
class LocalOnly(Rule):
    """B0: ignore neighbours."""
    name = "local"
    uses_messages = False

    def decide(self, v, own, received, hop):
        return ordinal_median(own), own

    def alarm_score(self, v, own, received, hop):
        return float(1.0 - normalise(own)[0])


class UnfilteredMean(Rule):
    """B1: confidence-weighted mean of all beliefs, no filter."""
    name = "unfiltered_mean"

    def decide(self, v, own, received, hop):
        bs = [own] + list(received.values())
        w = np.array([1.0 + confidence(b) for b in bs])
        agg = normalise(np.sum(np.array(bs) * w[:, None], axis=0))
        return ordinal_median(agg), agg

    def alarm_score(self, v, own, received, hop):
        bs = [own] + list(received.values())
        return float(np.mean([1.0 - normalise(b)[0] for b in bs]))


class MajorityVote(Rule):
    """B2: most frequent level among own and received; ties go to the level
    closest to the own level."""
    name = "majority"

    def decide(self, v, own, received, hop):
        o = ordinal_median(own)
        lv = [o] + [ordinal_median(b) for b in received.values()]
        counts = np.bincount(lv, minlength=N_LEVELS)
        best = np.flatnonzero(counts == counts.max())
        level = int(best[np.argmin(np.abs(best - o))])
        p = np.zeros(N_LEVELS)
        p[level] = 1.0
        return level, p

    def alarm_score(self, v, own, received, hop):
        lv = [ordinal_median(own)] + [ordinal_median(b) for b in received.values()]
        return float(np.mean(np.asarray(lv) >= 1))


class CoordMedian(Rule):
    """B3: coordinate-wise median of beliefs."""
    name = "coord_median"

    def decide(self, v, own, received, hop):
        agg = normalise(np.median(np.array([own] + list(received.values())), axis=0))
        return ordinal_median(agg), agg

    def alarm_score(self, v, own, received, hop):
        return float(1.0 - self.decide(v, own, received, hop)[1][0])


class WMSR(Rule):
    """B4: W-MSR (LeBlanc et al., 2013) on scalar levels. Removes up to F
    values strictly above and F strictly below the own value, then averages
    the rest with equal weights. One iteration per hop."""
    name = "wmsr"
    scalar = True

    def __init__(self, F=1):
        self.F = int(F)

    def payload_from_belief(self, p):
        return float(ordinal_median(p))

    def decide(self, v, own, received, hop):
        x = float(own)
        vals = np.array(list(received.values()), dtype=float)
        hi = np.sort(vals[vals > x])[::-1]
        lo = np.sort(vals[vals < x])
        eq = vals[vals == x]
        kept = np.concatenate([hi[self.F:], lo[self.F:], eq, [x]])
        x_new = float(kept.mean())
        return int(np.clip(np.rint(x_new), 0, N_LEVELS - 1)), x_new

    def alarm_score(self, v, own, received, hop):
        return float(self.decide(v, own, received, hop)[1] / (N_LEVELS - 1))


class OracleMedian(Rule):
    """B5 (upper reference, not deployable): knows the Byzantine set and
    fuses only the own belief and honest neighbours with the same
    confidence-weighted median as v2 (own weight alpha, no filter, no trim).
    `byz` is set by the simulator before each run."""
    name = "oracle"
    needs_oracle = True

    def __init__(self, alpha=1.5, output="median"):
        self.alpha = alpha
        self.output = output
        self.byz = set()

    def decide(self, v, own, received, hop):
        bs = [own] + [b for u, b in received.items() if u not in self.byz]
        lv = [ordinal_median(b) for b in bs]
        w = np.array([(1.0 + confidence(b)) * (self.alpha if j == 0 else 1.0)
                      for j, b in enumerate(bs)])
        agg = normalise(np.sum(np.array(bs) * w[:, None], axis=0))
        if self.output == "mixture":
            return ordinal_median(agg), agg
        return weighted_median_level(lv, w), agg

    def alarm_score(self, v, own, received, hop):
        return float(1.0 - self.decide(v, own, received, hop)[1][0])


RULES = {
    "bf_v2": ByzantineFloodV2,
    "bf_v1": ByzantineFloodV1,
    "local": LocalOnly,
    "unfiltered_mean": UnfilteredMean,
    "majority": MajorityVote,
    "coord_median": CoordMedian,
    "wmsr": WMSR,
    "oracle": OracleMedian,
}


def build_rule(spec: dict) -> Rule:
    spec = dict(spec)
    kind = spec.pop("rule")
    spec.pop("hops", None)
    return RULES[kind](**spec)
