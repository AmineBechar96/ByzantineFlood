import numpy as np

from bflood.levels import (hellinger, one_hot, ordinal_median, peaked_belief,
                           weighted_median_level)


def test_ordinal_median_basic():
    assert ordinal_median([0.5, 0.5, 0, 0, 0]) == 0
    assert ordinal_median([0.1, 0.1, 0.1, 0.6, 0.1]) == 3
    # argmax would say 4; ordinal median says 2
    assert ordinal_median([0.3, 0.0, 0.25, 0.0, 0.45]) == 2


def test_peaked_belief_has_requested_median():
    for lvl in range(5):
        for conf in (0.51, 0.7, 0.99):
            assert ordinal_median(peaked_belief(lvl, conf)) == lvl


def test_hellinger_is_a_metric_on_random_beliefs():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        p, q, r = rng.dirichlet(np.ones(5), size=3)
        assert 0 <= hellinger(p, q) <= 1
        assert abs(hellinger(p, q) - hellinger(q, p)) < 1e-12
        assert hellinger(p, r) <= hellinger(p, q) + hellinger(q, r) + 1e-12
    assert hellinger(one_hot(0, 1.0), one_hot(4, 1.0)) > 0.99


def test_weighted_median_range_when_honest_weight_dominates():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        h = rng.integers(0, 5, size=rng.integers(1, 6))
        wh = rng.uniform(0.1, 3, size=len(h))
        b = rng.integers(0, 5, size=rng.integers(0, 4))
        wb = rng.uniform(0.1, 3, size=len(b))
        if wb.sum() >= wh.sum():
            continue
        m = weighted_median_level(np.r_[h, b], np.r_[wh, wb])
        assert h.min() <= m <= h.max()
