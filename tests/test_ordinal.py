import numpy as np

from bflood.levels import ordinal_median
from detector import ordinal as O
from detector import report


def test_interval_targets_mask_only_determined_thresholds():
    t, m = O.interval_targets([2, 0, 1], [2, 0, 4])
    assert t.tolist()[0] == [1, 1, 0, 0] and m[0].all()        # exact level 2
    assert t.tolist()[1] == [0, 0, 0, 0] and m[1].all()        # dry
    assert m[2].tolist() == [True, False, False, False]          # flooded, depth unknown
    assert t[2, 0] == 1


def test_beliefs_valid_and_median_matches_threshold_count():
    rng = np.random.default_rng(0)
    q = np.sort(rng.uniform(0, 1, (500, 4)), axis=1)[:, ::-1]
    p = O.beliefs_from_cum(q)
    assert np.allclose(p.sum(1), 1) and (p >= 0).all()
    lv = O.ordinal_level(p)
    assert (lv == (q > 0.5).sum(1)).all()
    assert all(ordinal_median(p[i]) == lv[i] for i in range(50))


def test_non_monotone_cum_is_repaired():
    p = O.beliefs_from_cum(np.array([[0.3, 0.6, 0.2, 0.1]]))
    assert (p >= 0).all() and np.isclose(p.sum(), 1)


def test_auc_and_bootstrap():
    assert O.binary_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    est, lo, hi = O.bootstrap_ci(np.mean, np.r_[np.zeros(50), np.ones(50)], n=300)
    assert est == 0.5 and lo < 0.5 < hi


def test_report_on_synthetic_predictions(tmp_path=None):
    rng = np.random.default_rng(1)

    def fake(n, levels):
        y = rng.choice(levels, n)
        p = np.full((n, 5), 0.05)
        p[np.arange(n), y] = 0.8
        return y, p / p.sum(1, keepdims=True)

    y, p = fake(200, [0, 1, 2, 3, 4])
    s3, ps = fake(100, [0, 1, 2])
    yd, pd = fake(60, [0])
    pred = {"deepflood_test": {"probs": p, "lo": y, "hi": y, "level3": -np.ones(200), "ambiguous": np.zeros(200)},
            "sturm_test": {"probs": ps, "lo": -np.ones(100), "hi": -np.ones(100), "level3": s3, "ambiguous": np.zeros(100)},
            "fe8k_test": {"probs": pd, "lo": yd, "hi": yd, "level3": -np.ones(60), "ambiguous": np.zeros(60)}}
    met = report.evaluate(pred)
    assert met["deepflood_acc"][0] == 1.0 and met["fe8k_false_alarm"][0] == 0.0
    labels, probs, _ = report.belief_bank(pred)
    assert len(labels) == 260 and probs.shape == (260, 5)
