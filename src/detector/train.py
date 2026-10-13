"""Train, calibrate and evaluate the ordinal flood detector (PyTorch).

    run(config_path, data_dir, out_root)

For each of `folds` folds over the training pool (train+val rows of the chosen
sources, split by duplicate group): train EfficientNet-B0 with a rank-consistent
ordinal head, keep the best epoch on the fold's validation part, fit a
temperature on it, and predict every test set. The fold predictions are
averaged into an ensemble, and report.write() produces metrics.json,
summary.md and belief_bank.npz under out_root/<config name>/.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import ordinal as O
from . import report
from .io import ZipImageReader
from .transforms import augment, evaluate_transform, fisheye_warp, letterbox

cv2.setNumThreads(1)  # one OpenCV thread per data-loader worker

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TEST_SETS = ("deepflood_test", "sturm_test", "alley_test", "fe8k_test")


def _int(x, default=-1):
    return int(x) if str(x).strip() != "" else default


class ManifestDataset(Dataset):
    def __init__(self, rows, data_dir, mode, cfg, seed=0, fisheye_eval=False):
        self.rows, self.data_dir, self.mode, self.cfg = rows, str(data_dir), mode, cfg
        self.seed, self.epoch, self.fisheye_eval, self._reader = seed, 0, fisheye_eval, None
        lo = np.array([_int(r["level_lo"]) for r in rows])
        hi = np.array([_int(r["level_hi"]) for r in rows])
        t, m = O.interval_targets(np.where(lo < 0, 0, lo), np.where(hi < 0, 4, hi))
        m[lo < 0] = False  # no 5-level label (STURM)
        self.t, self.m = t, m

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._reader is None:
            self._reader = ZipImageReader(self.data_dir)
        r = self.rows[i]
        img = self._reader.read(r["zip"], r["member"])
        size = self.cfg["size"]
        fisheye = r["is_fisheye"] == "1"
        if self.mode == "train":
            rng = np.random.default_rng([self.seed, self.epoch, i])
            a = self.cfg.get("augment", {})
            x, _ = augment(img, rng, fisheye, size, p_fisheye=a.get("p_fisheye", 0.7),
                           nuisance=a.get("nuisance", True))
        elif self.fisheye_eval and not fisheye:
            x = letterbox(fisheye_warp(img, np.random.default_rng([12345, i])), size)
        else:
            x = evaluate_transform(img, size)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(self.t[i]),
                torch.from_numpy(self.m[i].astype(np.float32)), i)


class OrdinalNet(nn.Module):
    """EfficientNet-B0 features -> one score s -> logits s + b_k with b_k
    strictly decreasing, so P(level > k) is non-increasing in k by construction."""

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        n = net.classifier[1].in_features
        net.classifier = nn.Identity()
        self.backbone, self.drop = net, nn.Dropout(dropout)
        self.fc = nn.Linear(n, 1, bias=False)
        self.b0 = nn.Parameter(torch.tensor([1.5]))
        self.delta = nn.Parameter(torch.full((O.N_THRESH - 1,), math.log(math.e - 1)))

    def biases(self):
        steps = torch.cat([torch.zeros(1, device=self.b0.device), torch.cumsum(F.softplus(self.delta), 0)])
        return self.b0 - steps

    def forward(self, x):
        return self.fc(self.drop(self.backbone(x))) + self.biases()


def masked_bce(logits, t, m):
    loss = F.binary_cross_entropy_with_logits(logits, t, reduction="none") * m
    return loss.sum() / m.sum().clamp(min=1.0)


def make_folds(rows, sources, k, seed, val_synthetic=False):
    """Folds over REAL train+val rows, split by group (duplicate group, or
    camera for fisheye images). Synthetic rows join a fold's training part
    unless their camera is in that fold's validation; there they are dropped,
    or, with val_synthetic (sim2real only: no real flooded image to validate
    on), added to the validation part. Training parts do not depend on it."""
    pool = [r for r in rows if r["role"] in ("train", "val") and r["source"] in sources]
    real = [r for r in pool if r["synthetic"] != "1"]
    synth = [r for r in pool if r["synthetic"] == "1"]
    rng = np.random.default_rng(seed)
    fold_of = {}
    for kind in (True, False):  # camera groups dealt out separately, so each fold gets 2-3 cameras
        gs = sorted({r["group"] for r in real if r["group"].startswith("cam") == kind})
        rng.shuffle(gs)
        fold_of.update({g: i % k for i, g in enumerate(gs)})
    folds = []
    for f in range(k):
        va = [r for r in real if fold_of[r["group"]] == f]
        held = {r["group"] for r in va}
        tr = [r for r in real if fold_of[r["group"]] != f] + [r for r in synth if r["group"] not in held]
        if val_synthetic:
            va = va + [r for r in synth if r["group"] in held]
        folds.append((tr, va))
    return folds


def stratified_sample(rows, n, seed, key=lambda r: r["source"]):
    """Seeded sample of n rows with every stratum represented (round-robin),
    in original order. Used only for capped smoke-test runs."""
    if n >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault(key(r), []).append(i)
    pools = [list(rng.permutation(v)) for _, v in sorted(strata.items())]
    picked = []
    while len(picked) < n:
        for p in pools:
            if p and len(picked) < n:
                picked.append(p.pop())
    return [rows[i] for i in sorted(picked)]


def sampler_for(rows):
    key = [(r["source"], r["level_lo"] if r["level_lo"] == r["level_hi"] else "interval") for r in rows]
    c = Counter(key)
    w = torch.tensor([1.0 / c[k] for k in key], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(rows), replacement=True)


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out = np.zeros((len(loader.dataset), O.N_THRESH), np.float32)
    for x, _, _, idx in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True))
        out[idx.numpy()] = z.float().cpu().numpy()
    return out


def fit_temperature(logits, t, m):
    z, tt, mm = (torch.tensor(a, dtype=torch.float32) for a in (logits, t, m))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = masked_bce(z / log_t.exp(), tt, mm)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.2, 5.0))


def train_fold(cfg, tr_rows, va_rows, data_dir, device, seed, log):
    ds_tr = ManifestDataset(tr_rows, data_dir, "train", cfg, seed)
    ds_va = ManifestDataset(va_rows, data_dir, "eval", cfg)
    nw = cfg.get("num_workers", 8)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], num_workers=nw, pin_memory=True)
    model = OrdinalNet(pretrained=cfg.get("pretrained", True)).to(device)
    head = [model.fc.weight, model.b0, model.delta]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
                             {"params": head, "lr": cfg["lr_head"]}], weight_decay=cfg["weight_decay"])
    E = cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / 2 if e < 1 else 0.5 * (1 + math.cos(math.pi * (e - 1) / max(1, E - 1))))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(E):
        ds_tr.set_epoch(ep)
        dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], sampler=sampler_for(tr_rows),
                           num_workers=nw, pin_memory=True, drop_last=len(tr_rows) > cfg["batch_size"])
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, t, m, _ in dl_tr:
            x, t, m = x.to(device, non_blocking=True), t.to(device), m.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z = model(x)
            loss = masked_bce(z.float(), t, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        sched.step()
        zv = predict_logits(model, dl_va, device)
        vloss = float(masked_bce(torch.tensor(zv), torch.tensor(ds_va.t), torch.tensor(ds_va.m.astype(np.float32))))
        log(f"  epoch {ep + 1}/{E} train {tot / max(n, 1):.4f} val {vloss:.4f} ({time.time() - t0:.0f}s)")
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log("  early stop")
                break
    if best_state is None:  # validation loss never finite: keep the last weights
        log("  warning: no finite validation loss; using last epoch")
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    zv = predict_logits(model, dl_va, device)
    T = fit_temperature(zv, ds_va.t, ds_va.m.astype(np.float32))
    return model, T, best


def predict_sets(model, T, rows, data_dir, cfg, device):
    out = {}
    sets = {s: [r for r in rows if r["test_set"] == s] for s in TEST_SETS}
    if cfg.get("fisheye_test", True):
        sets["deepflood_test_fisheye"] = sets["deepflood_test"]
    for name, rs in sets.items():
        if not rs:
            continue
        mx = cfg.get("max_test")
        rs = stratified_sample(rs, mx, 0, key=lambda r: (r["level_lo"], r["level_hi"], r["level3"])) if mx else rs
        ds = ManifestDataset(rs, data_dir, "eval", cfg, fisheye_eval=name.endswith("_fisheye"))
        dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 8))
        q = 1 / (1 + np.exp(-predict_logits(model, dl, device) / T))
        out[name] = {"probs": O.beliefs_from_cum(q),
                     "lo": np.array([_int(r["level_lo"]) for r in rs]),
                     "hi": np.array([_int(r["level_hi"]) for r in rs]),
                     "level3": np.array([_int(r["level3"]) for r in rs]),
                     "ambiguous": np.array([_int(r["ambiguous"], 0) for r in rs]),
                     "member": np.array([r["member"] for r in rs])}
    return out


def run(config_path, data_dir, out_root):
    cfg = yaml.safe_load(Path(config_path).read_text())
    name = cfg["name"]
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))
    rows = list(csv.DictReader(open(Path(data_dir) / "processed" / "manifest.csv", encoding="utf-8")))
    folds = make_folds(rows, set(cfg["sources"]), cfg["folds"], cfg.get("seed", 0),
                       val_synthetic=cfg.get("val_synthetic", False))
    log(f"{name}: device {device}, sources {cfg['sources']}, {len(folds)} folds")
    preds = []
    for f, (tr, va) in enumerate(folds[: cfg.get("run_folds", cfg["folds"])]):
        mx = cfg.get("max_train")
        if mx:  # smoke tests: seeded sample spread over sources, not the first rows
            tr = stratified_sample(tr, mx, cfg.get("seed", 0) + f)
            va = stratified_sample(va, max(8, mx // 4), cfg.get("seed", 0) + f)
        log(f"fold {f}: train {len(tr)}, val {len(va)}")
        model, T, vloss = train_fold(cfg, tr, va, data_dir, device, cfg.get("seed", 0) + f, log)
        log(f"fold {f}: best val {vloss:.4f}, temperature {T:.3f}")
        torch.save({"state_dict": model.state_dict(), "temperature": T, "config": cfg}, out / f"fold{f}.pt")
        p = predict_sets(model, T, rows, data_dir, cfg, device)
        np.savez(out / f"fold{f}_preds.npz", **{f"{k}__{kk}": vv for k, v in p.items() for kk, vv in v.items()})
        report.write(p, out / f"fold{f}", f"{name} fold {f}")
        preds.append(p)
    ens = {k: dict(preds[0][k], probs=np.mean([p[k]["probs"] for p in preds], axis=0)) for k in preds[0]}
    met = report.write(ens, out / "ensemble", f"{name} ensemble of {len(preds)} folds")
    log(json.dumps({k: v for k, v in met.items() if not isinstance(v, dict)}, indent=1))
    logf.close()
    return met
