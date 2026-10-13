"""Byzantine behaviours A0-A8 (A6 = all-Byzantine setting, A7 = + loss).

An adversary maps (sender, recipient, round, hop, context) to the belief
the Byzantine sender transmits to that recipient. Randomness is derived from
(seed, round, hop, sender, recipient) so every method in a run faces the
same adversarial draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .levels import N_LEVELS, confidence, one_hot, ordinal_median, peaked_belief

TOP = N_LEVELS - 1


@dataclass
class AdvContext:
    t: int
    hop: int
    truth: dict          # node -> true level
    beliefs: dict        # node -> honest detector belief this round
    epicenter: int
    detector: object
    seed: int
    extra: dict = field(default_factory=dict)

    def rng(self, *key) -> np.random.Generator:
        return np.random.default_rng([self.seed, 4, self.t, self.hop, *key])


def _far_target(level: int) -> int:
    return 0 if level >= 2 else TOP


def _mimic(recipient: int, ctx: AdvContext) -> np.ndarray:
    base = ctx.beliefs.get(recipient)
    if base is None:
        base = peaked_belief(ctx.truth[recipient], 0.8)
    obs = ordinal_median(base)
    lvl = obs + 1 if obs < TOP else obs - 1
    return peaked_belief(lvl, max(confidence(base), 0.55))


class Adversary:
    def __init__(self, name: str, switch_round: int = 8):
        if name not in ADVERSARY_NAMES:
            raise ValueError(f"unknown adversary {name!r}")
        self.name = name
        self.switch_round = switch_round

    def message(self, sender: int, recipient: int, ctx: AdvContext) -> np.ndarray:
        n = self.name
        if n == "A0":
            return ctx.rng(sender, recipient).dirichlet(np.ones(N_LEVELS))
        if n == "A1":
            return one_hot(_far_target(ctx.truth[recipient]))
        if n == "A2":
            return _mimic(recipient, ctx)
        if n == "A3":
            return one_hot(_far_target(ctx.truth[ctx.epicenter]))
        if n == "A4":
            offset = min(TOP, ctx.t)
            d = -1 if ctx.truth[recipient] >= 2 else 1
            lvl = int(np.clip(ctx.truth[recipient] + d * offset, 0, TOP))
            return peaked_belief(lvl, 0.9)
        if n == "A5":
            own = ctx.beliefs.get(recipient)
            o = ordinal_median(own) if own is not None else ctx.truth[recipient]
            return one_hot(_far_target(o))
        if n == "A8":
            if ctx.t < self.switch_round:
                # same message to every neighbour while acting honestly
                return ctx.detector.sample(ctx.truth[sender], ctx.rng(sender, 1_000_000))
            return _mimic(recipient, ctx)
        raise AssertionError(n)


ADVERSARY_NAMES = ("A0", "A1", "A2", "A3", "A4", "A5", "A8")
