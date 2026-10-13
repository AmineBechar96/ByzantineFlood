"""One simulation run: a graph, a flood, a fault set, an adversary, and a
set of fusion rules that all see identical beliefs and adversarial draws.

Adversary names: A0-A8 (adversaries.py); "AW" the worst case, searched here
per recipient, round and method (below); "none" for runs without attack
(n_byz must be 0)."""
from __future__ import annotations

import copy
import time
from itertools import product

import numpy as np

from .adversaries import AdvContext, Adversary
from .beliefs import build_detector
from .graphs import build_graph, central_node
from .levels import N_LEVELS, one_hot, ordinal_median, peaked_belief
from .metrics import METRICS, RUN_METRICS, round_metrics
from .rules import build_rule
from .errors import apply_errors
from .scenario import FloodScenario, choose_pond, max_local_byz, place_faults

TOP = N_LEVELS - 1

# ------------------------------------------------- worst-case adversary AW --
# For every honest recipient v with b >= 1 Byzantine neighbours, every round,
# hop and method separately, the Byzantine neighbours send the messages from
# the candidate set C that maximise the error of the output of v, knowing the
# rule, the truth and the honest messages. Equivocation is allowed, so the
# search is independent per recipient. The search is exhaustive over C^b
# when |C|^b <= AW_EXACT_LIMIT, otherwise coordinate ascent from A1.
AW_EXACT_LIMIT = 15000
AW_PASSES = 3


def aw_candidates() -> list:
    """The candidate set C (11 messages): a belief of full confidence at
    each level (one_hot, mass 0.992, as A1 sends), a belief of low confidence
    at each level (peaked_belief(l, 0.2): ordinal median 1, 1, 2, 3, 3 for
    l = 0..4, confidence 0.29-0.52) and silence (None, no message)."""
    return ([one_hot(lv) for lv in range(N_LEVELS)]
            + [peaked_belief(lv, 0.2) for lv in range(N_LEVELS)] + [None])


def _same(p, q) -> bool:
    if p is None or q is None:
        return p is None and q is None
    return np.array_equal(np.asarray(p), np.asarray(q))


def aw_payloads(rule) -> list:
    """The distinct payloads of C for `rule` (W-MSR maps C to its five
    levels and silence); None stands for silence."""
    P = []
    for c in aw_candidates():
        p = None if c is None else rule.payload_from_belief(c)
        if not any(_same(p, q) for q in P):
            P.append(p)
    return P


def aw_start(rule, payloads, truth_v: int) -> int:
    """Index in `payloads` of the A1 message to a recipient with truth_v."""
    a1 = rule.payload_from_belief(one_hot(0 if truth_v >= 2 else TOP))
    return next(i for i, p in enumerate(payloads) if _same(p, a1))


def aw_objective(y: int, truth_v: int) -> tuple:
    """Lexicographic: (i) the error |y - truth|; (ii) for a dry camera a
    false alarm (y >= 1), for a wet camera a miss (y == 0)."""
    return (abs(int(y) - int(truth_v)), bool(y >= 1) if truth_v == 0 else bool(y == 0))


def aw_search(rule, v, own, nbrs, honest_payload, byz_nbrs, truth_v, hop, payloads,
              start, mode="auto"):
    """Messages of the Byzantine neighbours `byz_nbrs` of honest v that
    maximise aw_objective of the output of v.

    nbrs: the neighbours whose messages reach v, in the order the rule
    receives them; honest_payload: u -> payload of the honest ones;
    payloads: the candidate payloads (None = silence); start: index of the
    A1 payload. mode "auto": exhaustive when len(payloads)**b <=
    AW_EXACT_LIMIT, else coordinate ascent (every Byzantine neighbour starts
    at A1; one neighbour at a time takes its best candidate with the others
    fixed; repeat until no change or AW_PASSES passes). "exhaustive" and
    "ascent" force one of the two. The A1 tuple is evaluated first and kept
    unless a tuple is strictly better.
    Returns (u -> payload or None, exact).

    Stateless rules are evaluated directly and the search over C^b is
    exact. Stateful rules (bf_v1, rules with use_trust/decay) are evaluated
    on copy.deepcopy(rule), and only the error of this round is maximised:
    for them the damage found is a lower bound (a lie that hurts later rounds
    through the state is not searched for).
    """
    b = len(byz_nbrs)
    stateful = bool(getattr(rule, "stateful", False))

    def value(idx):
        pick = dict(zip(byz_nbrs, idx))
        rec = {}
        for u in nbrs:
            if u in pick:
                if payloads[pick[u]] is not None:
                    rec[u] = payloads[pick[u]]
            else:
                rec[u] = honest_payload[u]
        r = copy.deepcopy(rule) if stateful else rule
        return aw_objective(r.decide(v, own, rec, hop)[0], truth_v)

    ub = max(aw_objective(y, truth_v) for y in range(N_LEVELS))
    best_idx = (start,) * b
    best = value(best_idx)
    exact = mode == "exhaustive" or (mode == "auto" and len(payloads) ** b <= AW_EXACT_LIMIT)
    if exact:
        if best < ub:
            for idx in product(range(len(payloads)), repeat=b):
                s = value(idx)
                if s > best:
                    best, best_idx = s, idx
                    if best == ub:
                        break
    else:
        cur = list(best_idx)
        for _ in range(AW_PASSES):
            changed = False
            for i in range(b):
                for c in range(len(payloads)):
                    if best == ub:
                        break
                    if c == cur[i]:
                        continue
                    trial = cur.copy()
                    trial[i] = c
                    s = value(trial)
                    if s > best:
                        best, cur, changed = s, trial, True
            if not changed or best == ub:
                break
        best_idx = tuple(cur)
    return {u: payloads[i] for u, i in zip(byz_nbrs, best_idx)}, exact


def thresholded(y: int, score: float, tau: float) -> int:
    """Alarm threshold (exp10, exp12): 0 if the alarm score of the rule is
    below tau, else max(1, y) with y the normal output."""
    return 0 if score < tau else max(1, int(y))


def resolve_n_byz(n_byz, n: int) -> int:
    """An int is a count; a float in (0, 1) is a fraction of the n nodes."""
    if isinstance(n_byz, float) and 0.0 < n_byz < 1.0:
        return int(round(n_byz * n))
    return int(n_byz)


def run_once(cfg: dict, n_byz, adversary: str, seed: int) -> list[dict]:
    """Run every method in cfg['methods'] once; return one row per method
    with metrics averaged over the rounds in which the flood is active.
    `n_byz` is a count, or a fraction of the nodes if a float in (0, 1).

    With cfg['sweep_threshold'] = [tau, ...] every method also gets one row
    per tau, in which the output is 0 if rule.alarm_score < tau, else
    max(1, y) with y the normal output; column `tau` is "none" for the
    normal output. The adversary (AW included) targets the normal output;
    the threshold is applied afterwards."""
    g = build_graph(cfg["graph"], seed=seed)
    n_byz_spec = n_byz
    n_byz = resolve_n_byz(n_byz, g.number_of_nodes())
    sc_cfg = cfg.get("scenario", {})
    # The flood spreads over the physical layout, not over the communication
    # links. With scenario.reference_k set, flood arrival and the BFS fault
    # placement use a fixed k-NN graph on the same positions (so changing
    # graph.k changes only who talks to whom). Without it, the communication
    # graph is used (all experiments before exp8; identical when k matches).
    ref_k = sc_cfg.get("reference_k")
    g_ref = build_graph({**cfg["graph"], "k": int(ref_k)}, seed=seed) if ref_k else g
    epi = sc_cfg.get("epicenter", "center")
    epi = central_node(g_ref) if epi == "center" else int(epi)
    pond_cfg = sc_cfg.get("pond")
    pond = choose_pond(g, epi, pond_cfg.get("node", "random"), seed) if pond_cfg else None
    scen = FloodScenario(g_ref, epi, t0=int(sc_cfg.get("t0", 3)),
                         speed=float(sc_cfg.get("speed", 1.0)),
                         growth=float(sc_cfg.get("growth", 0.8)),
                         pond_node=pond,
                         pond_level=int(pond_cfg.get("level", 3)) if pond_cfg else 3,
                         pond_mode=pond_cfg.get("mode", "isolated") if pond_cfg else "isolated")
    flt = cfg.get("faults", {})
    byz, crash = place_faults(g_ref, epi, n_byz, int(flt.get("n_crash", 0)),
                              flt.get("placement", "bfs"),
                              np.random.default_rng([seed, 1]),
                              exclude=() if pond is None else (pond,))
    honest = set(g.nodes) - byz - crash
    if not honest:
        return []
    detector = build_detector(cfg.get("detector", {}))
    if adversary == "none":
        if n_byz:
            raise ValueError("adversary 'none' needs n_byz = 0")
        adv = None
    else:
        # AW: A1 is the start of the search and what the rules that skip the
        # search (local, oracle) receive; both ignore it.
        adv = Adversary("A1" if adversary == "AW" else adversary,
                        switch_round=int(cfg.get("sleeper_switch", 8)))
    worst = adversary == "AW"
    R = int(cfg.get("rounds", 20))
    loss = float(cfg.get("loss", 0.0))
    n = g.number_of_nodes()
    taus = [float(x) for x in cfg.get("sweep_threshold", [])]
    variants = [None] + taus  # None: the normal output
    pond_nodes = {pond} if pond is not None else set()

    # identical inputs for every method
    rng_b = np.random.default_rng([seed, 2])
    truths = [scen.levels(t) for t in range(R)]
    beliefs = [{v: detector.sample(truths[t][v], rng_b) for v in g.nodes} for t in range(R)]
    if cfg.get("errors"):  # correlated detector errors (exp13), honest cameras only
        apply_errors(cfg["errors"], beliefs, truths, g_ref, honest, detector,
                     np.random.default_rng([seed, 6]), scen.t0)
    max_hops = max(int(m.get("hops", 1)) for m in cfg["methods"].values())
    drop = np.random.default_rng([seed, 3]).random((R, max_hops, n, n)) < loss

    f_local = max_local_byz(g, byz, honest)
    timing = bool(cfg.get("timing", False))
    # first round each honest node is reached by the flood (None if never)
    arrival = {v: next((t for t in range(R) if truths[t][v] >= 1), None) for v in honest}
    rows = []
    for mname, mspec in cfg["methods"].items():
        rule = build_rule(mspec)
        rule.reset(g)
        oracle = getattr(rule, "needs_oracle", False)
        if oracle:
            rule.byz = set(byz)
        T = int(mspec.get("hops", 1))
        # AW skips rules that ignore messages (local) and the oracle (it
        # drops the Byzantine inputs by construction)
        search = worst and rule.uses_messages and not oracle
        payloads = aw_payloads(rule) if search else None
        acc = {var: {k: [] for k in METRICS} for var in variants}
        first_alarm = {var: {v: None for v in honest} for var in variants}
        t_dec, n_dec = 0.0, 0
        n_search = n_exact = 0
        for t in range(R):
            truth = truths[t]
            obs = {v: ordinal_median(beliefs[t][v]) for v in honest}
            cur = {v: rule.payload_from_belief(beliefs[t][v]) for v in honest}
            score = {}
            for hop in range(1, T + 1):
                ctx = AdvContext(t=t, hop=hop, truth=truth, beliefs=beliefs[t],
                                 epicenter=epi, detector=detector, seed=seed)
                dec = {}
                for v in honest:
                    nbrs = [u for u in g.neighbors(v)
                            if not (u in crash or drop[t, hop - 1, u, v])]
                    bnb = [u for u in nbrs if u in byz]
                    if search and bnb:
                        pick, exact = aw_search(rule, v, cur[v], nbrs, cur, bnb, truth[v], hop,
                                                payloads, aw_start(rule, payloads, truth[v]))
                        n_search += 1
                        n_exact += int(exact)
                    else:
                        pick = {u: rule.payload_from_belief(adv.message(u, v, ctx)) for u in bnb}
                    rec = {}
                    for u in nbrs:
                        if u in byz:
                            if pick[u] is not None:  # None: AW chose silence
                                rec[u] = pick[u]
                        else:
                            rec[u] = cur[u]
                    if taus and hop == T:  # before decide: same state as decide
                        score[v] = rule.alarm_score(v, cur[v], rec, hop)
                    if timing:
                        t1 = time.perf_counter()
                        dec[v] = rule.decide(v, cur[v], rec, hop)
                        t_dec += time.perf_counter() - t1
                        n_dec += 1
                    else:
                        dec[v] = rule.decide(v, cur[v], rec, hop)
                cur = {v: dec[v][1] for v in honest}
            out = {v: dec[v][0] for v in honest}
            for var in variants:
                o = out if var is None else {v: thresholded(out[v], score[v], var)
                                             for v in honest}
                fa = first_alarm[var]
                for v in honest:
                    a = arrival[v]
                    if a is not None and t >= a and fa[v] is None and o[v] >= 1:
                        fa[v] = t
                if t >= scen.t0:
                    m = round_metrics(g, honest, truth, obs, o, scen.hops, pond=pond_nodes)
                    for k in METRICS:
                        acc[var][k].append(m[k])
        for var in variants:
            row = {"method": mname, "adversary": adversary, "n_byz": n_byz_spec,
                   "byz_count": n_byz, "f_local": f_local, "seed": seed, "n": n,
                   "messages_per_round": T * sum(g.degree(u) for u in g.nodes if u not in crash)}
            if taus:
                row["tau"] = "none" if var is None else var
            if pond is not None:
                row["pond_node"], row["pond_degree"] = pond, g.degree(pond)
            for k in METRICS:
                vals = np.array(acc[var][k], dtype=float)
                row[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            reached = [v for v in honest if arrival[v] is not None]
            if reached:
                fa = first_alarm[var]
                d = [(fa[v] if fa[v] is not None else R) - arrival[v] for v in reached]
                row["delay"] = float(np.mean(d))
                row["miss"] = float(np.mean([fa[v] is None for v in reached]))
            else:
                row["delay"] = row["miss"] = float("nan")
            row["decide_us"] = 1e6 * t_dec / n_dec if n_dec else float("nan")
            row["search_exact"] = n_exact / n_search if n_search else float("nan")
            rows.append(row)
    return rows
