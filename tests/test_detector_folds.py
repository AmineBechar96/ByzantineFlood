from detector.train import make_folds


def _rows():
    rows = []
    for c in range(1, 9):  # real fisheye cameras, train/val pool
        rows += [{"source": "fisheye8k", "role": "train", "synthetic": "0", "group": f"cam{c}", "id": f"fe{c}_{i}"}
                 for i in range(3)]
    for c in range(1, 7):  # synthetic edits of the same cameras
        rows += [{"source": "grok", "role": "train", "synthetic": "1", "group": f"cam{c}", "id": f"gk{c}_{i}"}
                 for i in range(2)]
    rows += [{"source": "deepflood", "role": "train", "synthetic": "0", "group": f"dup{g}", "id": f"df{g}"}
             for g in range(10)]
    rows += [{"source": "fisheye8k", "role": "test", "synthetic": "0", "group": "cam20", "id": "fe_test"}]
    return rows


def _ids(rs):
    return sorted(r["id"] for r in rs)


def test_make_folds_synthetic_training_only_by_default():
    rows, src = _rows(), {"fisheye8k", "grok", "deepflood"}
    folds = make_folds(rows, src, 4, 0)
    assert [(_ids(a), _ids(b)) for a, b in folds] == \
           [(_ids(a), _ids(b)) for a, b in make_folds(rows, src, 4, 0, val_synthetic=False)]
    real = [r for r in rows if r["synthetic"] == "0" and r["role"] != "test"]
    assert sorted(i for _, va in folds for i in _ids(va)) == _ids(real)  # each real row validated once
    for tr, va in folds:
        held = {r["group"] for r in va}
        assert all(r["synthetic"] == "0" for r in va)
        assert {r["group"] for r in tr if r["synthetic"] == "0"}.isdisjoint(held)
        assert _ids(r for r in tr if r["synthetic"] == "1") == \
               _ids(r for r in rows if r["synthetic"] == "1" and r["group"] not in held)
        assert "fe_test" not in _ids(tr) + _ids(va)


def test_make_folds_val_synthetic_adds_held_out_synthetic_to_validation():
    rows, src = _rows(), {"fisheye8k", "grok", "deepflood"}
    base = make_folds(rows, src, 4, 0)
    folds = make_folds(rows, src, 4, 0, val_synthetic=True)
    synth = [r for r in rows if r["synthetic"] == "1"]
    assert sorted(r["id"] for _, va in folds for r in va if r["synthetic"] == "1") == _ids(synth)
    for (tr0, va0), (tr, va) in zip(base, folds):
        assert _ids(tr) == _ids(tr0)  # training parts unchanged
        held = {r["group"] for r in va0}
        assert _ids(r for r in va if r["synthetic"] == "0") == _ids(va0)
        assert _ids(r for r in va if r["synthetic"] == "1") == _ids(r for r in synth if r["group"] in held)
        assert set(_ids(tr)).isdisjoint(_ids(va))
