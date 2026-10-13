import networkx as nx
import numpy as np

from bflood.levels import one_hot, ordinal_median, peaked_belief
from bflood.rules import WMSR, ByzantineFloodV2, MajorityVote


def _star(d):
    g = nx.star_graph(d)  # node 0 with neighbours 1..d
    return g


def test_v2_range_under_neighbour_majority_bound():
    """Definition 1(a) for f <= floor((d-1)/2), full trust, any Byzantine
    beliefs: the output stays within the honest observations."""
    rng = np.random.default_rng(2)
    for d in (3, 4, 5, 6, 8):
        f_max = (d - 1) // 2
        for _ in range(400):
            f = int(rng.integers(0, f_max + 1))
            # untrimmed variant with the filter: needs the weight condition
            rule = ByzantineFloodV2(trim=0, use_filter=True, use_trust=True)
            rule.reset(_star(d))
            own = peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
            rec = {}
            for u in range(1, d + 1 - f):
                rec[u] = peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
            honest_levels = [ordinal_median(own)] + [ordinal_median(b) for b in rec.values()]
            for u in range(d + 1 - f, d + 1):
                rec[u] = one_hot(int(rng.integers(0, 5)), 1.0) if rng.random() < .5 \
                    else rng.dirichlet(np.ones(5))
            y, _ = rule.decide(0, own, rec, hop=1)
            assert min(honest_levels) <= y <= max(honest_levels), (d, f)


def test_v2_correction_when_neighbours_agree():
    """Definition 1(b), untrimmed rule: unanimous confident honest neighbours
    override a wrong own observation, with f = 1 confident liar, k = 4."""
    rule = ByzantineFloodV2(trim=0, use_filter=True, use_trust=True)
    rule.reset(_star(4))
    own = peaked_belief(1, 0.6)  # own detector wrong (truth is 3)
    rec = {1: peaked_belief(3, 0.9), 2: peaked_belief(3, 0.85), 3: peaked_belief(3, 0.9),
           4: one_hot(0, 1.0)}
    y, _ = rule.decide(0, own, rec, hop=1)
    assert y == 3


def test_v2_fewer_than_three_inputs_returns_own():
    rule = ByzantineFloodV2()
    rule.reset(_star(1))
    y, _ = rule.decide(0, peaked_belief(2, 0.8), {1: one_hot(4, 1.0)}, hop=1)
    assert y == 2


def test_trust_decays_for_persistent_outlier():
    rule = ByzantineFloodV2(trim=0, use_filter=True, use_trust=True)
    rule.reset(_star(4))
    for _ in range(5):
        rule.decide(0, peaked_belief(0, 0.9),
                    {1: peaked_belief(0, 0.9), 2: peaked_belief(0, 0.85),
                     3: peaked_belief(0, 0.9), 4: one_hot(4, 1.0)}, hop=1)
    assert rule.trust[0][4] < 0.05
    assert rule.trust[0][1] == 1.0


def test_wmsr_trims_extremes():
    r = WMSR(F=1)
    y, x = r.decide(0, 2.0, {1: 2.0, 2: 3.0, 3: 2.0, 4: 4.0}, hop=1)
    assert x == (2.0 + 3.0 + 2.0 + 2.0) / 4 and y == 2


def test_majority_ties_go_to_own():
    r = MajorityVote()
    y, _ = r.decide(0, peaked_belief(1, 0.9),
                    {1: peaked_belief(3, 0.9), 2: peaked_belief(1, 0.9), 3: peaked_belief(3, .9)},
                    hop=1)
    assert y == 1


def test_trimmed_v2_range_holds_for_any_weights():
    """With trim = F and at most F Byzantine neighbours, the output stays in
    the honest range even when honest neighbours are fully distrusted and
    Byzantine ones claim full confidence (no weight condition needed)."""
    rng = np.random.default_rng(3)
    for d in (3, 4, 5, 6, 7, 8):
        for F in (1, 2, 3):
            if F > d:
                continue
            for _ in range(300):
                f = int(rng.integers(0, F + 1))
                rule = ByzantineFloodV2(trim=F, use_filter=bool(rng.random() < 0.5))
                rule.reset(_star(d))
                own = peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
                rec, honest = {}, [ordinal_median(own)]
                for u in range(1, d + 1 - f):
                    rec[u] = peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
                    honest.append(ordinal_median(rec[u]))
                    rule.trust[0][u] = rng.uniform(0.0, 1.0)  # adversarially low trust
                for u in range(d + 1 - f, d + 1):
                    rec[u] = one_hot(int(rng.integers(0, 5)), 1.0)
                y, _ = rule.decide(0, own, rec, hop=1)
                assert min(honest) <= y <= max(honest), (d, F, f)


def test_theorem_correction_quorum_trimmed():
    """Theorem 3: if (h - F) * w_min > alpha(1+gamma_v) + 2 f, unanimous honest
    neighbours (level c) fix a wrong own level, for any Byzantine beliefs."""
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(6000):
        d = int(rng.integers(3, 11)); F = int(rng.integers(0, 4)); f = int(rng.integers(0, F + 1))
        h = d - f
        if h < 1:
            continue
        alpha = float(rng.choice([1.0, 1.5]))
        c = int(rng.integers(0, 5)); o = int(rng.choice([x for x in range(5) if x != c]))
        g_v = rng.uniform(0.51, 0.99); g_h = rng.uniform(0.51, 0.99, size=h)
        w_min = float((1 + g_h).min())
        if not (h - F) * w_min > alpha * (1 + g_v) + 2 * f:
            continue
        rule = ByzantineFloodV2(trim=F, alpha=alpha, use_filter=False)
        rule.reset(_star(d))
        rec = {u + 1: peaked_belief(c, g_h[u]) for u in range(h)}
        for u in range(h + 1, d + 1):
            rec[u] = peaked_belief(o, 0.99) if rng.random() < .5 else one_hot(int(rng.integers(0, 5)), 1.0)
        y, _ = rule.decide(0, peaked_belief(o, g_v), rec, hop=1)
        assert y == c, (d, F, f, alpha)
        checked += 1
    assert checked > 300


def test_theorem_range_is_stateless():
    """Self-stabilisation of safety: Theorem 1 holds for arbitrary (corrupted)
    trust state, e.g. trust 0 for honest and 1 for Byzantine neighbours."""
    rng = np.random.default_rng(5)
    for _ in range(2000):
        d = int(rng.integers(3, 9)); f = int(rng.integers(0, 3))
        if f > d:
            continue
        rule = ByzantineFloodV2(trim=2)
        rule.reset(_star(d))
        own = peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
        honest, rec = [ordinal_median(own)], {}
        for u in range(1, d + 1):
            if u <= d - f:
                rec[u] = peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
                honest.append(ordinal_median(rec[u])); rule.trust[0][u] = 0.0
            else:
                rec[u] = one_hot(int(rng.integers(0, 5)), 1.0); rule.trust[0][u] = 1.0
        y, _ = rule.decide(0, own, rec, hop=1)
        assert min(honest) <= y <= max(honest)


def test_paper_rule_defaults():
    """The paper rule: trim 2, confidence weights, no filter, no trust."""
    r = ByzantineFloodV2()
    assert r.trim == 2 and r.use_conf and not r.use_filter and not r.use_trust and r.alpha == 1.5
