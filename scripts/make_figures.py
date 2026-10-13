"""Paper figures and tables from the simulator results (step 27).

Usage: python scripts/make_figures.py [--results results] [--out paper]
Reads results/exp*/summary.csv; writes paper/figures/*.pdf (+ .png previews)
and paper/tables/*.tex. Uses only the final runs (detector_base bank, the
simplified paper rule). Never point it at results/_old_filter.

Colours: categorical slots validated for colour-blind separation (all pairs,
light surface); identity is also carried by marker and line style, and every
figure has a matching table.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ------------------------------------------------------------------ style --
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd6"
STYLE = {  # method -> (label, colour, marker, linestyle, zorder)
    "bf_v2":   ("ByzantineFlood (ours)",    "#2a78d6", "o", "-",  5),
    "bf_v1":   ("Filtered temporal mean",   "#eb6834", "s", "-",  4),
    "wmsr_F2": ("W-MSR ($F{=}2$)",          "#1baf7a", "^", "-",  3),
    "oracle":  ("Oracle (knows attackers)", "#4a3aa7", "D", "--", 2),
    "local":   ("Single camera",            INK2,      "x", ":",  1),
}
FIG_ORDER = ["bf_v2", "bf_v1", "wmsr_F2", "oracle", "local"]
LABEL = {
    "local": "Single camera", "unfiltered": "Unfiltered mean", "majority": "Majority vote",
    "coord_median": "Coordinate median", "wmsr_F1": "W-MSR ($F{=}1$)",
    "wmsr_F2": "W-MSR ($F{=}2$)", "bf_v1": "Filtered temporal mean",
    "bf_v2": "\\textbf{ByzantineFlood (ours)}", "bf_v2_mix": "Ours, mixture output",
    "oracle": "Oracle (knows attackers)", "oracle_mix": "Oracle, mixture output",
}
ADV = ["A0", "A1", "A2", "A3", "A4", "A5", "A8"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL, DCOL = 3.5, 7.16  # IEEE column / double-column width, inches


def line(ax, x, y, ci, m):
    lab, c, mk, ls, z = STYLE[m]
    ax.plot(x, y, color=c, marker=mk, ls=ls, label=lab, zorder=z,
            markeredgecolor="white" if mk not in "x" else c, markeredgewidth=0.6)
    if ci is not None:
        ax.fill_between(x, y - ci, y + ci, color=c, alpha=0.12, lw=0, zorder=z - 0.5)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    (out / "preview").mkdir(exist_ok=True)
    fig.savefig(out / "preview" / f"{name}.png", dpi=200)
    plt.close(fig)


def legend_below(fig, axes, ncol=5):
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.06))


def sel(s, **kw):
    for k, v in kw.items():
        s = s[s[k] == v]
    return s


# ----------------------------------------------------------------- figures --
# The main adversary is the worst case AW (exp9, Oct 2026). Single camera and
# oracle ignore the Byzantine messages, so AW skips them; their values come
# from exp1_main with adversary A1 at the same n_byz.
NO_AW = ("local", "oracle")


def _aw_rows(R):
    s9 = pd.read_csv(R / "exp9_worstcase/summary.csv")
    s1 = sel(pd.read_csv(R / "exp1_main/summary.csv"), adversary="A1")
    return pd.concat([s9[~s9.method.isin(NO_AW)], s1[s1.method.isin(NO_AW)]], ignore_index=True)


def fig_main(R, out):
    s = _aw_rows(R)
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.2))
    panels = [("acc", "Accuracy (exact level)", 1), ("false_alarm", "False alarm rate", 1),
              ("range_viol", "Range violations (%)", 100)]
    for ax, letter, (k, title, sc) in zip(axes, "abc", panels):
        for m in FIG_ORDER:
            d = sel(s, method=m).sort_values("n_byz")
            line(ax, d.n_byz.values, sc * d[k].values, sc * d[k + "_ci"].values, m)
        ax.set_xlabel("Number of Byzantine cameras (of 18)")
        ax.set_ylabel(title)
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
        ax.set_xticks(range(7))
    # shade the Byzantine counts whose largest local count f is within F = 2
    # (BFS placement: f = 2 up to five attackers, f = 3 at six)
    f_of = sel(s, method="bf_v2").set_index("n_byz").f_local
    n_in = int(f_of[f_of <= 2].index.max())
    axes[2].axvspan(-0.3, n_in + 0.5, color="#f0efec", zorder=0, lw=0)
    axes[2].text(n_in / 2, axes[2].get_ylim()[1] * 0.92, "within bound ($f\\leq 2$)",
                 ha="center", va="top", fontsize=6.5, color=INK2)
    axes[2].set_xlim(-0.3, 6.3)
    legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_main")


def fig_detector_quality(R, out):
    s = sel(pd.read_csv(R / "exp2_rho/summary.csv"), adversary="A1", n_byz=2)
    loc = sel(s, method="local").set_index("detector.swap")["acc"]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 1.45, 1.9))
    for m in FIG_ORDER:
        d = sel(s, method=m).sort_values("detector.swap")
        x = loc.loc[d["detector.swap"]].values
        line(axes[0], x, d.acc.values, d.acc_ci.values, m)
        line(axes[1], x, d.false_alarm.values, d.false_alarm_ci.values, m)
    axes[0].set_ylabel("Accuracy (exact level)")
    axes[1].set_ylabel("False-alarm rate")
    for ax, letter in zip(axes, "ab"):
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
        ax.set_xlabel("Single-camera accuracy\n(right = weaker detector)")
        ax.invert_xaxis()
    legend_below(fig, axes, ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out, "fig_detector_quality")


def fig_scale(R, out):
    s = sel(pd.read_csv(R / "exp3_scale/summary.csv"), **{"graph.k": 4})
    fig, axes = plt.subplots(1, 2, figsize=(COL * 1.45, 1.9))
    for m in FIG_ORDER:
        d = sel(s, method=m, adversary="A1", n_byz=0.10).sort_values("graph.n")
        line(axes[0], d["graph.n"].values, d.acc.values, d.acc_ci.values, m)
        d = sel(s, method=m, adversary="A5", **{"graph.n": 400}).sort_values("n_byz")
        line(axes[1], 100 * d.n_byz.values, 100 * d.range_viol.values, 100 * d.range_viol_ci.values, m)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([50, 100, 200, 400], ["50", "100", "200", "400"])
    axes[0].set_xlabel("Number of cameras (10 % Byzantine)")
    axes[0].set_ylabel("Accuracy (exact level)")
    axes[0].set_title("(a)", loc="left", fontweight="bold")
    axes[1].set_xticks([0, 5, 10, 15])
    axes[1].set_xlabel("Byzantine cameras (% of 400)")
    axes[1].set_ylabel("Range violations (%)")
    axes[1].set_title("(b)", loc="left", fontweight="bold")
    legend_below(fig, axes, ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out, "fig_scale")


def fig_faults(R, out):
    s = sel(pd.read_csv(R / "exp4_faults/summary.csv"), n_byz=2, **{"faults.n_crash": 4})
    fig, axes = plt.subplots(1, 2, figsize=(COL * 1.45, 1.9))
    for m in FIG_ORDER:
        d = sel(s, method=m).sort_values("loss")
        line(axes[0], 100 * d.loss.values, d.acc.values, d.acc_ci.values, m)
        line(axes[1], 100 * d.loss.values, d.false_alarm.values, d.false_alarm_ci.values, m)
    axes[0].set_ylabel("Accuracy (exact level)")
    axes[1].set_ylabel("False-alarm rate")
    for ax, letter in zip(axes, "ab"):
        ax.set_title(f"({letter}) 4 crashed, 2 Byzantine cameras", loc="left", fontweight="bold")
        ax.set_xlabel("Message loss (%)")
        ax.set_xticks([0, 10, 20, 30])
    legend_below(fig, axes, ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out, "fig_faults")


# ------------------------------------------------------------------ tables --
def f2(x):
    return "--" if not np.isfinite(x) else f"{x:.2f}"


def pct(x):
    return "--" if not np.isfinite(x) else ("0" if x == 0 else f"{100 * x:.1f}")


def tab_main(R, out):
    """Main table: the worst case AW (exp9); single camera and oracle from
    exp1 A1. f <= 2 column: max over n_byz 1-5; f = 3: n_byz 6."""
    s = _aw_rows(R)
    s9 = pd.read_csv(R / "exp9_worstcase/summary.csv")
    cols = [1, 2, 4, 6]
    f = {nb: int(sel(s9, n_byz=nb).f_local.median()) for nb in cols}
    lab = {**LABEL, "coord_median": "Coordinatewise median"}
    methods = ["local", "unfiltered", "majority", "coord_median", "wmsr_F1", "wmsr_F2",
               "bf_v1", "bf_v2", "oracle"]
    rows = []
    for m in methods:
        d = sel(s, method=m)
        r0 = sel(d, n_byz=0).iloc[0]
        cells = [lab[m], f2(r0.acc), f2(r0.false_alarm)]
        for nb in cols:
            r = sel(d, n_byz=nb).iloc[0]
            cells.append(f"{f2(r.acc)}/{f2(r.false_alarm)}")
        cells.append(pct(d[(d.n_byz >= 1) & (d.n_byz <= 5)].range_viol.max()))
        cells.append(pct(sel(d, n_byz=6).range_viol.iloc[0]))
        rows.append(" & ".join(cells) + r" \\")
    head = " & ".join(f"{nb} ($f{{=}}{f[nb]}$)" for nb in cols)
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
% sources: results/exp9_worstcase (AW); single camera and oracle ignore Byzantine
% messages, values from results/exp1_main (A1, same n_byz)
\begin{{table*}}[t]
\caption{{Worst case adversary on the 18-camera layout}}
\label{{tab:main}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{No attack}} & \multicolumn{{4}}{{c}}{{AW, Byzantine cameras: accuracy / false alarms}} & \multicolumn{{2}}{{c}}{{Range viol. (\%)}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-7}}\cmidrule(l){{8-9}}
Method & Acc. & FA & {head} & $f\le2$ & $f=3$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / "tab_main.tex").write_text(tex)


def tab_adversaries(R, out):
    """The seven fixed adversaries A0-A8 with 4 Byzantine cameras (exp1);
    the main table before the worst case AW, now Table S3."""
    s = pd.read_csv(R / "exp1_main/summary.csv")
    methods = ["local", "unfiltered", "majority", "coord_median", "wmsr_F1", "wmsr_F2",
               "bf_v1", "bf_v2", "oracle"]
    rows = []
    for m in methods:
        d = sel(s, method=m)
        cells = [LABEL[m], f2(sel(d, n_byz=0, adversary="A0").acc.iloc[0]),
                 f2(sel(d, n_byz=0, adversary="A0").false_alarm.iloc[0])]
        for a in ADV:
            r = sel(d, n_byz=4, adversary=a).iloc[0]
            cells.append(f"{f2(r.acc)}/{f2(r.false_alarm)}")
        cells.append(pct(d[d.n_byz <= 4].range_viol.max()))
        cells.append(pct(d[d.n_byz == 6].range_viol.max()))
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
\begin{{table*}}[t]
\caption{{Seven fixed adversaries, four Byzantine cameras}}
\label{{tab:adversaries}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lcc{'c' * len(ADV)}cc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{No attack}} & \multicolumn{{{len(ADV)}}}{{c}}{{4 Byzantine cameras: accuracy / false alarms}} & \multicolumn{{2}}{{c}}{{Range viol. (\%)}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-{3 + len(ADV)}}}\cmidrule(l){{{4 + len(ADV)}-{5 + len(ADV)}}}
Method & Acc. & FA & {' & '.join(ADV)} & $f\le2$ & $f=3$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / "tab_adversaries.tex").write_text(tex)


def tab_ablation(R, out):
    s = pd.read_csv(R / "exp5_ablation/summary.csv")
    variants = [
        ("bf_v2", "\\textbf{Ours (paper rule)}"),
        ("trim0", "no trimming"), ("trim1", "trim $F{=}1$"),
        ("trim_auto", "trim $F_v{=}\\lfloor (d_v{-}1)/2\\rfloor$"),
        ("no_conf", "no confidence weights"),
        ("alpha1", "own weight $\\alpha{=}1$"), ("alpha3", "own weight $\\alpha{=}3$"),
        ("with_filter", "+ Hellinger filter"), ("with_filter_trust", "+ Hellinger filter + trust"),
        ("symkl_filter_trust", "+ sym.-KL filter + trust"),
        ("mixture", "mixture output instead of median"),
    ]
    rows = []
    for m, lab in variants:
        d = sel(s, method=m)
        r0 = sel(d, n_byz=0, adversary="A1").iloc[0]
        r4 = sel(d, n_byz=4, adversary="A1").iloc[0]
        rows.append(" & ".join([lab, f2(r0.acc), f2(r0.false_alarm), f2(r4.acc), f2(r4.false_alarm),
                                pct(d[d.n_byz <= 4].range_viol.max()),
                                pct(d[d.n_byz == 6].range_viol.max())]) + r" \\")
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
\begin{{table}}[t]
\caption{{Ablation of the rule}}
\label{{tab:ablation}}
\centering\scriptsize\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{No attack}} & \multicolumn{{2}}{{c}}{{4 Byz.}} & \multicolumn{{2}}{{c}}{{Viol. (\%)}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(l){{6-7}}
Variant & Acc. & FA & Acc. & FA & $f\le2$ & $f=3$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (out / "tab_ablation.tex").write_text(tex)


def tab_scale(R, out):
    s = pd.read_csv(R / "exp3_scale/summary.csv")
    rows = []
    for k in (4, 6):
        for n in (50, 100, 200, 400):
            d = sel(s, **{"graph.k": k, "graph.n": n})
            v2 = sel(d, method="bf_v2")
            v1 = sel(d, method="bf_v1")
            gain = (sel(v2, adversary="A1", n_byz=0.10).acc.iloc[0]
                    - sel(d, method="local", adversary="A1", n_byz=0.10).acc.iloc[0])
            rows.append(" & ".join([str(k), str(n), f"{sel(v2, n_byz=0.0).messages_per_round.mean():.0f}",
                                    f"{v2.decide_us.mean() / 1000:.2f}", f"{v1.decide_us.mean() / 1000:.2f}",
                                    f"+{100 * gain:.0f}", pct(v2.range_viol.max()),
                                    pct(v1.range_viol.max())]) + r" \\")
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
\begin{{table}}[t]
\caption{{Scale and cost on random layouts}}
\label{{tab:scale}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}cc r cc c cc@{{}}}}
\toprule
 & & Msgs/ & \multicolumn{{2}}{{c}}{{ms/decision}} & Gain & \multicolumn{{2}}{{c}}{{Viol. (\%)}} \\
\cmidrule(lr){{4-5}}\cmidrule(l){{7-8}}
$k$ & $n$ & round & ours & FTM & (pts) & ours & FTM \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (out / "tab_scale.tex").write_text(tex)


def tab_hops(R, out):
    s = sel(pd.read_csv(R / "exp7_hops/summary.csv"), adversary="A1")
    rows = []
    for m, lab in [("local", "Single camera"), ("bf_v2_T1", "Ours, $T{=}1$"), ("bf_v2_T2", "Ours, $T{=}2$"),
                   ("bf_v2_T3", "Ours, $T{=}3$"), ("wmsr_F2_T1", "W-MSR, $T{=}1$"),
                   ("wmsr_F2_T2", "W-MSR, $T{=}2$"), ("wmsr_F2_T3", "W-MSR, $T{=}3$")]:
        d = sel(s, method=m)
        r0, r4 = sel(d, n_byz=0).iloc[0], sel(d, n_byz=4).iloc[0]
        msgs = 0 if m == "local" else r0.messages_per_round
        rows.append(" & ".join([lab, f"{msgs:.0f}", f2(r0.acc), f2(r0.false_alarm),
                                f2(r4.acc), f2(r4.false_alarm)]) + r" \\")
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
\begin{{table}}[t]
\caption{{$T$-hop exchange, 18-camera layout, 30 seeds, A1.}}
\label{{tab:hops}}
\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrcccc@{{}}}}
\toprule
 & Msgs/ & \multicolumn{{2}}{{c}}{{No attack}} & \multicolumn{{2}}{{c}}{{4 Byzantine}} \\
\cmidrule(lr){{3-4}}\cmidrule(l){{5-6}}
Method & round & Acc. & FA & Acc. & FA \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (out / "tab_hops.tex").write_text(tex)


def fig_degree(R, out):
    """exp8: neighbours per camera (17 = complete graph). Flood and attacker
    placement fixed on the physical 4-NN layout; only the links change."""
    p = R / "exp8_degree/summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    ks = sorted(s["graph.k"].unique())
    xpos = {k: i for i, k in enumerate(ks)}
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.2))
    panels = [("false_alarm", 0, "False-alarm rate", "no attack"),
              ("acc", 0, "Accuracy (exact level)", "no attack"),
              ("acc", 4, "Accuracy (exact level)", "4 Byzantine cameras")]
    for ax, letter, (k, nb, ylab, title) in zip(axes, "abc", panels):
        for m in ["bf_v2", "wmsr_F2", "oracle", "local"]:
            d = sel(s, method=m, adversary="A1", n_byz=nb).sort_values("graph.k")
            line(ax, np.array([xpos[v] for v in d["graph.k"]]), d[k].values, d[k + "_ci"].values, m)
        ax.axvspan(xpos[4] - 0.3, xpos[4] + 0.3, color="#f0efec", zorder=0, lw=0)
        ax.set_xticks(range(len(ks)), [f"{v}" if v < 17 else "17\n(all)" for v in ks])
        ax.set_xlabel("Neighbours per camera, $k$")
        ax.set_ylabel(ylab)
        ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")
    legend_below(fig, axes, ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_degree")


# ------------------------------------------- system diagram (R1.6, v1 review) --
def fig_system(out):
    """Labelled system model: sensing, communication/threat, fusion rule."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
    BLUE, ORANGE, GREY, PANEL, EDGE = "#2a78d6", "#eb6834", "#8a8984", "#f6f5f2", "#c9c8c2"
    fig = plt.figure(figsize=(DCOL, 2.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 39); ax.axis("off")

    def panel(x, w, title, sec):
        ax.add_patch(FancyBboxPatch((x, 1.2), w, 36.3, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=PANEL, ec=EDGE, lw=0.8))
        ax.text(x + 1.2, 35.6, title, fontsize=8.5, fontweight="bold", va="center", color=INK)
        ax.text(x + w - 1.2, 35.6, sec, fontsize=7, va="center", ha="right", color=INK2)

    def box(x, y, w, h, txt, fc="white", ec=INK2, fs=7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)

    def arrow(x1, y1, x2, y2, c=INK2, ls="-", lw=0.9, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     color=c, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))

    # ---------------- (a) sensing model
    panel(1, 27, "(a) Sensing", "Sec. III-A")
    ax.add_patch(Rectangle((3, 24.5), 23, 8.5, fc="#cfe0f3", ec="none"))
    ax.text(14.5, 31.6, "street scene (flood hazard)", ha="center", fontsize=6.5, color=INK2)
    ax.add_patch(Circle((14.5, 27.6), 3.0, fc="white", ec=BLUE, lw=1.0))
    ax.text(14.5, 27.6, "fisheye\ncamera $v$", ha="center", va="center", fontsize=6.3)
    arrow(14.5, 24.4, 14.5, 22.0)
    box(4, 16.2, 21, 5.6, "Flood-level detector\n(CNN, ordinal head, calibrated)")
    arrow(14.5, 16.0, 14.5, 13.6)
    box(4, 4.0, 21, 9.4, "", fc="white")
    ax.text(14.5, 11.9, "belief $p_v$ over 5 levels", ha="center", fontsize=7)
    for i, hgt in enumerate([0.4, 1.1, 3.6, 1.6, 0.5]):
        ax.add_patch(Rectangle((8.6 + 2.5 * i, 5.4), 1.8, hgt, fc=BLUE, ec="none"))
        ax.text(9.5 + 2.5 * i, 4.6, str(i), ha="center", fontsize=5.8, color=INK2)
    ax.text(14.5, 9.95, "level $\\ell(p_v)$ = ordinal median", ha="center", fontsize=6.2, color=INK2)

    # ---------------- (b) communication + threat model
    panel(30, 33, "(b) Network and threats", "Sec. III-B, III-C")
    cx, cy = 46.5, 18.5
    nbrs = [(37.0, 26.5, "honest", BLUE, "white"), (56.0, 26.5, "honest", BLUE, "white"),
            (37.5, 10.0, "Byzantine", ORANGE, ORANGE), (56.5, 10.0, "crashed", GREY, "#e7e6e2")]
    for i, (x, y, lab, ec, fc) in enumerate(nbrs):
        dashed = lab == "crashed"
        ax.plot([cx, x], [cy, y], color=EDGE if dashed else INK2, lw=0.9, ls=":" if dashed else "-", zorder=1)
        ax.add_patch(Circle((x, y), 2.6, fc=fc, ec=ec, lw=1.1, zorder=2))
        ax.text(x, y, "×" if dashed else f"$u_{i + 1}$", ha="center", va="center",
                fontsize=11 if dashed else 7, color=GREY if dashed else ("white" if lab == "Byzantine" else INK),
                zorder=3)
        ax.text(x, y - 4.2, lab, ha="center", fontsize=6.5, color=INK2)
    ax.add_patch(Circle((cx, cy), 3.0, fc="white", ec=BLUE, lw=1.4, zorder=2))
    ax.text(cx, cy, "$v$", ha="center", va="center", fontsize=8, zorder=3)
    ax.text(40.8, 8.6, "may lie and send\ndifferent beliefs\nto each neighbour", fontsize=5.8,
            color=ORANGE, ha="left", va="center")
    ax.text(56.5, 4.3, "(silent)", ha="center", fontsize=5.8, color=INK2)
    ax.text(cx, 32.4, "links to the $k=4$ nearest cameras,\nauthenticated (HMAC)",
            ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.15)
    ax.text(cx, 2.6, "$\\leq f_v$ Byzantine neighbours; crashes and losses unbounded",
            ha="center", fontsize=6.0, color=INK2)

    # ---------------- (c) fusion rule
    panel(65, 34, "(c) Fusion rule at camera $v$", "Sec. IV")
    steps = [
        ("1  Collect the own belief and those of the neighbours;\n    fewer than 3 inputs: output the own level", 27.2),
        ("2  Map each belief to its level $\\ell(\\cdot)$", 21.5),
        ("3  Trimming: drop up to $F{=}2$ levels\n    above and $F$ below the own level", 14.8),
        ("4  Weighted median, weights = confidence\n    (own weight $\\alpha{=}1.5$) $\\rightarrow$ output $y_v$", 8.1),
    ]
    for txt, y in steps:
        ax.add_patch(FancyBboxPatch((67, y), 30, 5.0, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="white", ec=INK2, lw=0.8))
        ax.text(68.2, y + 2.5, txt, ha="left", va="center", fontsize=6.6, color=INK, linespacing=1.2)
    for y in (27.2, 21.5, 14.8):
        nxt = {27.2: 26.5, 21.5: 19.8, 14.8: 13.1}[y]
        arrow(82, y - 0.1, 82, nxt + 0.05) if False else None
    for y1, y2 in ((27.2, 26.5), (21.5, 19.8), (14.8, 13.1)):
        arrow(82, y1, 82, y2)
    ax.add_patch(FancyBboxPatch((67, 2.2), 30, 4.6, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="#e3eefb", ec=BLUE, lw=0.9))
    ax.text(82, 4.5, "Guarantee (Thm. 2): if $f_v \\leq F$, $y_v$ lies within\nthe levels seen by the honest cameras around $v$",
            ha="center", va="center", fontsize=6.4, color=INK, linespacing=1.2)

    # cross-panel arrows
    arrow(25.2, 8.7, 43.3, 17.0, c=BLUE, rad=-0.15)   # own belief to v
    ax.text(28.6, 5.6, "own belief", fontsize=6.2, color=BLUE)
    ax.plot([49.6, 64.5, 64.5], [18.5, 18.5, 29.7], color=INK2, lw=0.9)   # beliefs at v to rule
    arrow(64.4, 29.7, 66.8, 29.7)
    ax.text(57.0, 19.2, "all beliefs\nat $v$", fontsize=6.2, color=INK2, ha="center", va="bottom",
            linespacing=1.1)
    save(fig, out, "fig_system")


# ------------------------------------------------- layout map (step 27b) --
def fig_layout(R, out, k=4, n_byz=4):
    """The 18-camera layout: k-NN links, the flood epicentre and the cameras
    made Byzantine in the main experiment (BFS from the epicentre)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bflood.graphs import build_graph, central_node
    from bflood.scenario import place_faults
    g = build_graph({"type": "testbed", "k": k})
    pos = {v: g.nodes[v]["pos"] for v in g}
    epi = central_node(g)
    byz, _ = place_faults(g, epi, n_byz, 0, "bfs", np.random.default_rng(0))
    xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
    x0, y0 = xs.min(), ys.min()
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    for u, v in g.edges:
        ax.plot([pos[u][0] - x0, pos[v][0] - x0], [pos[u][1] - y0, pos[v][1] - y0],
                color="#b8b7b1", lw=0.8, zorder=1)
    for v in g:
        x, y = pos[v][0] - x0, pos[v][1] - y0
        if v in byz:
            fc, ec = "#eb6834", "#eb6834"
        else:
            fc, ec = "white", "#2a78d6"
        ax.scatter([x], [y], s=150, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
        ax.text(x, y, str(v + 1), ha="center", va="center", fontsize=6.5, zorder=4,
                color="white" if v in byz else INK)
        if v == epi:
            ax.scatter([x], [y], s=330, facecolor="none", edgecolor=INK, linewidth=0.9,
                       linestyle="--", zorder=2)
    h = [plt.Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor="white",
                    markeredgecolor="#2a78d6", label="honest camera"),
         plt.Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor="#eb6834",
                    markeredgecolor="#eb6834", label=f"Byzantine ({n_byz}-attacker runs)"),
         plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor="none",
                    markeredgecolor=INK, linestyle="", markeredgewidth=0.9, label="flood epicentre"),
         plt.Line2D([], [], color="#b8b7b1", lw=0.8, label=f"link ({k}-NN, symmetric)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_xlim(-30, xs.max() - x0 + 30)
    ax.set_ylim(-30, ys.max() - y0 + 30)
    fig.tight_layout()
    save(fig, out, "fig_layout")
    deg = [d for _, d in g.degree()]
    return g.number_of_edges(), min(deg), max(deg)


# ------------------------------------------------- detector table (27b) --
def tab_detector(R, out):
    m = pd.read_csv(R / "detector_final/metrics.csv")
    s2r = pd.read_csv(R / "detector_final_sim2real/metrics.csv")
    m = pd.concat([m[m.run != "detector_sim2real"], s2r[s2r.run == "detector_sim2real"]])
    runs = [("detector_base", "5 folds", "\\textbf{Detector (all data)}"),
            ("detector_base", "folds 0-2", "\\quad same, folds 0--2"),
            ("detector_no_fisheye", "3 folds", "\\quad no fisheye warp"),
            ("detector_no_nuisance", "3 folds", "\\quad no nuisance aug."),
            ("detector_sim2real", "5 folds", "Synthetic + dry fisheye only")]
    cols = [("deepflood_test", "acc"), ("deepflood_test", "within1"), ("deepflood_test", "mae"),
            ("deepflood_test", "ece"), ("deepflood_test_fisheye", "acc"),
            ("sturm_test", "flooded_auc"), ("sturm_test", "kendall_tau_flooded"),
            ("alley_test", "flooded_auc"), ("alley_test", "false_alarm"), ("fe8k_test", "false_alarm")]

    def get(run, ens, ts, k):
        r = m[(m.run == run) & (m.ensemble == ens) & (m.test_set == ts) & (m.metric == k)]
        return r.iloc[0] if len(r) else None

    rows = []
    for run, ens, lab in runs:
        cells = [lab]
        for ts, k in cols:
            r = get(run, ens, ts, k)
            cells.append("--" if r is None else f"{r.value:.2f}")
        rows.append(" & ".join(cells) + r" \\")
        # 95 % bootstrap intervals of the main row are quoted in the text
        # (Sec. VI-C), not in the table (author decision, 8 Oct 2026).
    n = {ts: int(get("detector_base", "5 folds", ts, k).n) for ts, k in
         [("deepflood_test", "acc"), ("sturm_test", "flooded_auc"), ("alley_test", "flooded_auc"),
          ("fe8k_test", "false_alarm")]}
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
\begin{{table*}}[t]
\caption{{Detector on the four test sets (FA: false alarm rate)}}
\label{{tab:detector}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lcccccccccc@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{DeepFlood}} & Fisheye & \multicolumn{{2}}{{c}}{{STURM}} & \multicolumn{{2}}{{c}}{{AlleyFloodNet}} & FishEye8K \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-10}}\cmidrule(l){{11-11}}
Training & Acc. & $\pm1$ & MAE & ECE & Acc. & AUC & $\tau$ & AUC & FA & FA \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / "tab_detector.tex").write_text(tex)


# ------------------------------------- matched false alarms (exp10, Oct 2026) --
# The three simple fusion rules share one muted ink and differ by marker and
# dash only: seven hues cannot all be told apart pairwise on a scatter-like
# plot (validator, all pairs), so colour stays on the four rules the text
# discusses and TRADEOFF.md carries every number.
MUTED = "#8a8984"
TRADEOFF_STYLE = {  # method -> (label, colour, marker, linestyle, zorder)
    "bf_v2":        ("ByzantineFlood",           "#2a78d6", "o", "-",            6),
    "bf_v1":        ("Filtered temporal mean",   "#eb6834", "s", "-",            5),
    "wmsr_F2":      ("W-MSR ($F{=}2$)",          "#1baf7a", "^", "-",            4),
    "oracle":       ("Oracle (knows attackers)", "#4a3aa7", "D", "--",           3),
    "local":        ("Single camera",            INK2,      "X", ":",            2),
    "unfiltered":   ("Unfiltered mean",          MUTED,     "v", (0, (4, 1.5)),  1),
    "majority":     ("Majority vote",            MUTED,     "P", (0, (1, 1)),    1),
    "coord_median": ("Coordinate median",        MUTED,     "<", (0, (4, 1, 1, 1)), 1),
}
TRADEOFF_ORDER = ["bf_v2", "bf_v1", "wmsr_F2", "oracle", "local", "unfiltered", "majority",
                  "coord_median"]
TRADEOFF_PANELS = [("none", 0, "No attack"), ("A1", 2, "A1, 2 Byzantine cameras"),
                   ("AW", 2, "Worst case AW, 2 Byzantine cameras")]
# exp10 also runs AW with 4 Byzantine cameras: TRADEOFF.md only, not the paper
REPORT_PANELS = TRADEOFF_PANELS + [("AW", 4, "Worst case AW, 4 Byzantine cameras")]


def _tradeoff_curves(R):
    s = pd.read_csv(R / "exp10_tradeoff/summary.csv")
    s["tau"] = s["tau"].astype(str)
    return s


def _curve(d):
    """Threshold curve of one method, in increasing tau (false alarms fall)."""
    c = d[d.tau != "none"].copy()
    c["t"] = c.tau.astype(float)
    return c.sort_values("t")


def _at_fa(c, fa, col):
    """`col` at false alarm rate `fa`, linear along the threshold curve
    between consecutive taus; the best value if several segments bracket fa;
    None if fa lies outside the curve."""
    x, y = c.false_alarm.values, c[col].values
    vals = []
    for i in range(len(x) - 1):
        lo, hi = min(x[i], x[i + 1]), max(x[i], x[i + 1])
        if lo - 1e-12 <= fa <= hi + 1e-12:
            if hi - lo < 1e-12:
                vals.append(max(y[i], y[i + 1]) if col == "acc" else min(y[i], y[i + 1]))
            else:
                vals.append(y[i] + (fa - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
    if not vals:
        return None
    return max(vals) if col == "acc" else min(vals)


def fig_tradeoff(R, out):
    p = R / "exp10_tradeoff/summary.csv"
    if not p.exists():
        return
    s = _tradeoff_curves(R)
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.45), sharey=True)
    titles = {"none": "No attack", "A1": "A1, 2 Byzantine", "AW": "AW (worst case), 2 Byzantine"}
    for ax, letter, (adv, nb, _) in zip(axes, "abc", TRADEOFF_PANELS):
        for m in TRADEOFF_ORDER:
            lab, c, mk, ls, z = TRADEOFF_STYLE[m]
            d = sel(s, adversary=adv, n_byz=nb, method=m)
            cur = _curve(d)
            ax.plot(cur.false_alarm.values, cur.acc.values, color=c, ls=ls,
                    lw=1.7 if m == "bf_v2" else 1.0, zorder=z)
            n0 = d[d.tau == "none"].iloc[0]
            ax.plot([n0.false_alarm], [n0.acc], ls="", marker=mk, markersize=6.5 if m == "bf_v2" else 5,
                    markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.8, zorder=z + 10)
        ax.set_xlabel("False alarm rate")
        ax.set_title(f"({letter}) {titles[adv]}", loc="left", fontweight="bold")
        ax.set_xlim(-0.015, 0.62)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
    axes[0].set_ylabel("Accuracy (exact level)")
    h = [plt.Line2D([], [], color=TRADEOFF_STYLE[m][1], ls=TRADEOFF_STYLE[m][3],
                    lw=1.7 if m == "bf_v2" else 1.0, marker=TRADEOFF_STYLE[m][2], markersize=5,
                    markerfacecolor=TRADEOFF_STYLE[m][1], markeredgecolor="white")
         for m in TRADEOFF_ORDER]
    lab = [TRADEOFF_STYLE[m][0] for m in TRADEOFF_ORDER]
    h += [plt.Line2D([], [], color=INK2, lw=1.0),
          plt.Line2D([], [], ls="", marker="o", markersize=5, markerfacecolor=INK2,
                     markeredgecolor="white")]
    lab += ["line: threshold $\\tau$ = 0.05-0.95", "filled marker: normal output"]
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, "fig_tradeoff")


def tradeoff_report(R):
    """results/exp10_tradeoff/TRADEOFF.md: accuracy of every method at the
    false alarm rate of the normal ByzantineFlood output, and every
    (method, tau) that beats that output on both axes."""
    p = R / "exp10_tradeoff/summary.csv"
    if not p.exists():
        return
    s = _tradeoff_curves(R)
    runs = int(s.runs.max())
    L = ["# Matched false alarms (exp10_tradeoff)", "",
         f"18-camera layout, k = 4, detector_base bank, {runs} seeds, 20 rounds. Every method",
         "also runs with an alarm threshold tau in 0.05..0.95 on its own alarm score",
         "(src/bflood/rules.py): output 0 if score < tau, else max(1, y). Matched values are",
         "linear along the threshold curve of each method between consecutive taus (the best",
         "value where several segments bracket the target). Means over seeds; +- is the 95 %",
         "CI half-width. Generated by scripts/make_figures.py.", ""]
    for adv, nb, name in REPORT_PANELS:
        d = sel(s, adversary=adv, n_byz=nb)
        if d.empty:
            continue
        b0 = sel(d, method="bf_v2", tau="none").iloc[0]
        fa0, acc0 = b0.false_alarm, b0.acc
        L += [f"## {name}", "",
              f"ByzantineFlood, normal output: accuracy {acc0:.3f} +- {b0.acc_ci:.3f}, "
              f"false alarm rate {fa0:.3f} +- {b0.false_alarm_ci:.3f}, delay {b0.delay:.2f} rounds.", "",
              f"| Method | normal output: acc / FA / delay | acc at FA = {fa0:.3f} | delay at FA = {fa0:.3f} | acc minus ByzantineFlood |",
              "|---|---|---|---|---|"]
        for m in TRADEOFF_ORDER:
            dm = sel(d, method=m)
            n0 = dm[dm.tau == "none"].iloc[0]
            c = _curve(dm)
            a = _at_fa(c, fa0, "acc")
            dl = _at_fa(c, fa0, "delay")
            if a is None:
                rng = f"not reached (curve FA {c.false_alarm.min():.3f}-{c.false_alarm.max():.3f})"
                cells = [rng, "-", "-"]
            else:
                cells = [f"{a:.3f}", f"{dl:.2f}", f"{a - acc0:+.3f}"]
            L.append(f"| {TRADEOFF_STYLE[m][0]} | {n0.acc:.3f} / {n0.false_alarm:.3f} / {n0.delay:.2f} | "
                     + " | ".join(cells) + " |")
        L += ["", "Every (method, tau) with higher accuracy AND a lower false alarm rate than the "
              "normal ByzantineFlood output (delay and miss shown, ByzantineFlood: "
              f"{b0.delay:.2f} rounds, miss {b0.miss:.3f}):", "",
              "| Method | tau | accuracy | false alarm rate | delay | miss |", "|---|---|---|---|---|---|"]
        n_beat, n_beat_delay = 0, 0
        for m in TRADEOFF_ORDER:
            if m == "bf_v2":
                continue
            for _, r in sel(d, method=m).iterrows():
                if r.acc > acc0 and r.false_alarm < fa0:
                    n_beat += 1
                    n_beat_delay += int(r.delay <= b0.delay and r.miss <= b0.miss)
                    L.append(f"| {TRADEOFF_STYLE[m][0]} | {r.tau} | {r.acc:.3f} +- {r.acc_ci:.3f} "
                             f"({r.acc - acc0:+.3f}) | {r.false_alarm:.3f} +- {r.false_alarm_ci:.3f} "
                             f"({r.false_alarm - fa0:+.3f}) | {r.delay:.2f} | {r.miss:.3f} |")
        if not n_beat:
            L.append("| none | | | | | |")
        L += ["", f"Answer: {'**yes**' if n_beat else '**no**'}, {n_beat} (method, tau) points beat the "
              "normal ByzantineFlood output on both accuracy and false alarm rate; "
              f"{n_beat_delay} of them also have no larger delay and miss rate."]
        # the same question against the whole ByzantineFlood threshold curve
        cb = _curve(sel(d, method="bf_v2"))
        above = []
        for m in TRADEOFF_ORDER:
            if m == "bf_v2":
                continue
            for _, r in _curve(sel(d, method=m)).iterrows():
                ab = _at_fa(cb, r.false_alarm, "acc")
                if ab is not None and r.acc > ab:
                    above.append((m, r.tau, r.false_alarm, r.acc - ab))
        L.append("")
        if above:
            worst = max(above, key=lambda x: x[3])
            L.append(f"Points of other methods above the ByzantineFlood threshold curve at the same "
                     f"false alarm rate: {len(above)}; largest margin {worst[3]:.3f} "
                     f"({TRADEOFF_STYLE[worst[0]][0]}, tau = {worst[1]}, FA {worst[2]:.3f}).")
        else:
            L.append("Points of other methods above the ByzantineFlood threshold curve at the same "
                     "false alarm rate (where the curves overlap): none.")
        L.append("")
    L += threshold_report_lines(R)
    (R / "exp10_tradeoff/TRADEOFF.md").write_text("\n".join(L) + "\n")


THRESHOLD_ROWS = [("local", "Single camera"), ("unfiltered", "Unfiltered mean"),
                  ("majority", "Majority vote"), ("coord_median", "Coord.\\ median"),
                  ("wmsr_F2", "W-MSR ($F{=}2$)"), ("bf_v1", "FTM"),
                  ("bf_v2", "\\textbf{ByzantineFlood}"), ("oracle", "Oracle")]


def threshold_rows(R, nb=2):
    """Matched false alarm protocol: per method, tau maximises accuracy without
    attack subject to FA <= FA of the normal ByzantineFlood output without
    attack; the same tau is kept under AW with `nb` Byzantine cameras.
    ByzantineFlood and single camera keep their normal output (tau "--").
    Returns [(method, label, tau, no-attack row, AW row)]."""
    s = _tradeoff_curves(R)
    s0, sa = sel(s, adversary="none"), sel(s, adversary="AW", n_byz=nb)
    fa_max = sel(s0, method="bf_v2", tau="none").false_alarm.iloc[0]
    out = []
    for m, lab in THRESHOLD_ROWS:
        if m in ("bf_v2", "local"):
            tau = "none"
        else:
            c = _curve(sel(s0, method=m))
            ok = c[c.false_alarm <= fa_max]
            tau = ok.loc[ok.acc.idxmax()].tau if len(ok) else None
        if tau is None:
            out.append((m, lab, None, None, None))
            continue
        out.append((m, lab, tau, sel(s0, method=m, tau=tau).iloc[0],
                    sel(sa, method=m, tau=tau).iloc[0]))
    return out


def tab_threshold(R, out):
    rows = []
    for m, lab, tau, r0, ra in threshold_rows(R, 2):
        t = "--" if tau in (None, "none") else f"{float(tau):.2f}"
        if r0 is None:
            rows.append(" & ".join([lab, t] + ["--"] * 6) + r" \\")
            continue
        rows.append(" & ".join([lab, t, f2(r0.acc), f2(r0.false_alarm), f2(ra.acc),
                                f2(ra.false_alarm), f2(ra.delay), pct(ra.range_viol)]) + r" \\")
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
% source: results/exp10_tradeoff (tau tuned without attack, kept under AW)
\begin{{table}}[t]
\caption{{Alarm threshold tuned without attack}}
\label{{tab:threshold}}
\centering\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lccccccc@{{}}}}
\toprule
 & & \multicolumn{{2}}{{c}}{{No attack}} & \multicolumn{{4}}{{c}}{{AW, 2 Byzantine}} \\
\cmidrule(lr){{3-4}}\cmidrule(l){{5-8}}
Method & $\tau$ & Acc. & FA & Acc. & FA & Delay & Viol.\,\% \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (out / "tab_threshold.tex").write_text(tex)


def threshold_report_lines(R):
    """tab_threshold rows for every AW case in exp10 (markdown, TRADEOFF.md)."""
    s = _tradeoff_curves(R)
    L = ["## Threshold tuned without attack (tab_threshold protocol)", "",
         "tau maximises accuracy without attack subject to FA <= FA of the normal ByzantineFlood",
         "output without attack; the same tau is kept under AW. ByzantineFlood and single camera:",
         "normal output.", ""]
    for nb in sorted(sel(s, adversary="AW").n_byz.unique()):
        L += [f"### AW, {nb} Byzantine cameras", "",
              "| Method | tau | no attack acc | no attack FA | AW acc | AW FA | AW delay | AW viol. % |",
              "|---|---|---|---|---|---|---|---|"]
        for m, lab, tau, r0, ra in threshold_rows(R, nb):
            name = TRADEOFF_STYLE[m][0]
            if r0 is None:
                L.append(f"| {name} | -- | | | | | | |")
                continue
            t = "--" if tau == "none" else f"{float(tau):.2f}"
            L.append(f"| {name} | {t} | {r0.acc:.3f} | {r0.false_alarm:.3f} | {ra.acc:.3f} | "
                     f"{ra.false_alarm:.3f} | {ra.delay:.2f} | {100 * ra.range_viol:.1f} |")
        L.append("")
    return L


# ------------------------------------------------------ pond (exp11, Oct 2026) --
def tab_pond(R, out):
    p = R / "exp11_pond/summary.csv"
    if not p.exists():
        return
    s = sel(pd.read_csv(p), **{"scenario.pond.level": 3})
    methods = [("local", "Single camera"), ("unfiltered", "Unfiltered mean"),
               ("coord_median", "Coordinate median"), ("wmsr_F2", "W-MSR ($F{=}2$)"),
               ("bf_v1", "Filtered temporal mean"), ("bf_v2", "\\textbf{ByzantineFlood}"),
               ("bf_v2_a3", "ByzantineFlood, $\\alpha{=}3$"), ("oracle", "Oracle (knows attackers)")]
    cols = [("isolated", "none"), ("isolated", "AW"), ("inside", "none"), ("inside", "AW")]
    rows = []
    for m, lab in methods:
        cells = [lab]
        for mode, adv in cols:
            r = sel(s, method=m, adversary=adv, **{"scenario.pond.mode": mode}).iloc[0]
            cells += [f2(r.acc_pond), f2(r.alarm_pond), f2(r.false_alarm), pct(r.range_viol)]
        rows.append(" & ".join(cells) + r" \\")
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
\begin{{table*}}[t]
\caption{{Pond camera, level 3}}
\label{{tab:pond}}
\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}l cccc cccc cccc cccc@{{}}}}
\toprule
 & \multicolumn{{8}}{{c}}{{Isolated pond, level 3}} & \multicolumn{{8}}{{c}}{{Pond inside the flood, 2 levels above}} \\
\cmidrule(lr){{2-9}}\cmidrule(l){{10-17}}
 & \multicolumn{{4}}{{c}}{{No attack}} & \multicolumn{{4}}{{c}}{{AW, 2 Byz.}} & \multicolumn{{4}}{{c}}{{No attack}} & \multicolumn{{4}}{{c}}{{AW, 2 Byz.}} \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}\cmidrule(lr){{10-13}}\cmidrule(l){{14-17}}
Method & Acc. & Alarm & FA & Viol.\% & Acc. & Alarm & FA & Viol.\% & Acc. & Alarm & FA & Viol.\% & Acc. & Alarm & FA & Viol.\% \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / "tab_pond.tex").write_text(tex)


# ---------------------------------------- threshold transfer (exp12, Oct 2026) --
# tau tuned in exp10 without attack (paper Table IV, decision 11); ByzantineFlood
# and the single camera keep their normal output.
FIXED_TAU = {"unfiltered": 0.75, "majority": 0.85, "coord_median": 0.85, "wmsr_F2": 0.35,
             "bf_v1": 0.90, "oracle": 0.65}
TRANSFER_SETTINGS = [("base", "Base"), ("swap0.2", "Swap 0.2"), ("swap0.4", "Swap 0.4"),
                     ("k6", "$k{=}6$"), ("speed0.5", "Speed 0.5"), ("speed2", "Speed 2"),
                     ("rgg100", "100 cameras")]
TRANSFER_MD = {"base": "base (18 cameras, k = 4, speed 1)", "swap0.2": "detector degraded, swap 0.2",
               "swap0.4": "detector degraded, swap 0.4",
               "k6": "k = 6 (flood and attackers on the 4-NN layout)",
               "speed0.5": "flood speed 0.5 hops/round", "speed2": "flood speed 2 hops/round",
               "rgg100": "random layout, 100 cameras, k = 4 (20 seeds)"}
TRANSFER_KEYS = ("acc", "false_alarm", "delay", "shallow_recall", "range_viol")


def _tau_str(t):
    return "none" if t in (None, "none") else f"{float(t):.2f}"


def _transfer_row(d, adv, m, tau):
    r = sel(d, adversary=adv, method=m, tau=_tau_str(tau))
    return r.iloc[0] if len(r) else None


def transfer_table(R):
    """{setting: (FA cap, {method: fixed/tuned tau and rows, no attack and AW})}"""
    s = pd.read_csv(R / "exp12_transfer/summary.csv")
    s["tau"] = s["tau"].astype(str).map(lambda x: x if x == "none" else f"{float(x):.2f}")
    out = {}
    for st, _ in TRANSFER_SETTINGS:
        d = sel(s, setting=st)
        if d.empty:
            continue
        fa_max = _transfer_row(d, "none", "bf_v2", "none").false_alarm
        res = {}
        for m, _ in THRESHOLD_ROWS:
            ft = FIXED_TAU.get(m, "none")
            if m in FIXED_TAU:
                c = _curve(sel(d, adversary="none", method=m))
                ok = c[c.false_alarm <= fa_max]
                tt = ok.loc[ok.acc.idxmax()].tau if len(ok) else None
            else:
                tt = "none"
            res[m] = {"fixed_tau": ft, "tuned_tau": tt,
                      "fixed_none": _transfer_row(d, "none", m, ft),
                      "fixed_aw": _transfer_row(d, "AW", m, ft),
                      "tuned_none": _transfer_row(d, "none", m, tt) if tt is not None else None,
                      "tuned_aw": _transfer_row(d, "AW", m, tt) if tt is not None else None}
        out[st] = (fa_max, res)
    return out


def _md_cells(r):
    if r is None:
        return "-"
    return " / ".join(f"{r[k]:.2f}" if k == "delay" else f"{r[k]:.3f}" for k in TRANSFER_KEYS)


def _md_label(lab):
    return lab.replace("$", "").replace("{=}", " = ")


def transfer_report(R):
    if not (R / "exp12_transfer/summary.csv").exists():
        return
    T = transfer_table(R)
    base = T["base"][1]
    L = ["# Threshold transfer (exp12_transfer)", "",
         "Each rule keeps the alarm threshold tuned in exp10 without attack (paper Table IV):",
         ", ".join(f"{TRADEOFF_STYLE[m][0]} {t:.2f}" for m, t in FIXED_TAU.items())
         + "; ByzantineFlood and the single camera use their normal output. Seven settings,",
         "without attack and under AW with 2 Byzantine cameras; 50 seeds (20 for the random layout).",
         "Re-tuned tau: the exp10 rule in that setting (max accuracy without attack subject to",
         "FA <= FA of the normal ByzantineFlood output in that setting). Cells: acc / FA / delay",
         "(rounds) / shallow recall (alarm at true level 1) / range violations (fraction).",
         "Generated by scripts/make_figures.py.", "",
         "## Summary", "",
         "| Setting | W-MSR fixed tau, AW: acc / FA | ByzantineFlood, AW: acc / FA | W-MSR re-tuned tau | FA cap (ByzantineFlood FA, no attack) |",
         "|---|---|---|---|---|"]
    for st, _ in TRANSFER_SETTINGS:
        if st not in T:
            continue
        fa_max, res = T[st]
        w, b = res["wmsr_F2"], res["bf_v2"]
        L.append(f"| {TRANSFER_MD[st]} | {w['fixed_aw'].acc:.3f} / {w['fixed_aw'].false_alarm:.3f} | "
                 f"{b['fixed_aw'].acc:.3f} / {b['fixed_aw'].false_alarm:.3f} | "
                 f"{_tau_str(w['tuned_tau'])} | {fa_max:.3f} |")
    labs = [_md_label(lab) for _, lab in TRANSFER_SETTINGS]
    L += ["", "Re-tuned tau per setting:", "",
          "| Method | fixed (exp10) | " + " | ".join(labs) + " |",
          "|---|---|" + "---|" * len(labs)]
    for m in FIXED_TAU:
        L.append(f"| {TRADEOFF_STYLE[m][0]} | {FIXED_TAU[m]:.2f} | "
                 + " | ".join(_tau_str(T[st][1][m]["tuned_tau"]) if st in T else "-"
                              for st, _ in TRANSFER_SETTINGS) + " |")
    for adv, title in (("none", "no attack"), ("aw", "AW, 2 Byzantine")):
        L += ["", f"Fixed tau, {title}: change from base in accuracy / FA (points):", "",
              "| Method | " + " | ".join(labs[1:]) + " |", "|---|" + "---|" * (len(labs) - 1)]
        for m, _ in THRESHOLD_ROWS:
            b0 = base[m][f"fixed_{adv}"]
            cells = []
            for st, _ in TRANSFER_SETTINGS[1:]:
                r = T[st][1][m][f"fixed_{adv}"] if st in T else None
                cells.append("-" if r is None else
                             f"{100 * (r.acc - b0.acc):+.1f} / {100 * (r.false_alarm - b0.false_alarm):+.1f}")
            L.append(f"| {TRADEOFF_STYLE[m][0]} | " + " | ".join(cells) + " |")
    for st, _ in TRANSFER_SETTINGS:
        if st not in T:
            continue
        fa_max, res = T[st]
        L += ["", f"## {TRANSFER_MD[st]}", "", f"FA cap for re-tuning: {fa_max:.3f}.", "",
              "| Method | tau | fixed, no attack | fixed, AW | re-tuned tau | re-tuned, no attack | re-tuned, AW |",
              "|---|---|---|---|---|---|---|"]
        for m, _ in THRESHOLD_ROWS:
            x = res[m]
            L.append(f"| {TRADEOFF_STYLE[m][0]} | {_tau_str(x['fixed_tau'])} | {_md_cells(x['fixed_none'])} | "
                     f"{_md_cells(x['fixed_aw'])} | {_tau_str(x['tuned_tau'])} | "
                     f"{_md_cells(x['tuned_none'])} | {_md_cells(x['tuned_aw'])} |")
    (R / "exp12_transfer/TRANSFER.md").write_text("\n".join(L) + "\n")


def tab_transfer(R, out):
    if not (R / "exp12_transfer/summary.csv").exists():
        return
    T = transfer_table(R)
    rows = []
    for m, lab in THRESHOLD_ROWS:
        cells = [lab]
        for st, _ in TRANSFER_SETTINGS:
            r = T[st][1][m]["fixed_aw"] if st in T else None
            cells.append("--" if r is None else f"{f2(r.acc)}/{f2(r.false_alarm)}")
        rows.append(" & ".join(cells) + r" \\")
    head = " & ".join(lab for _, lab in TRANSFER_SETTINGS)
    n = len(TRANSFER_SETTINGS)
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
% source: results/exp12_transfer (tau fixed at the exp10 values; AW, 2 Byzantine)
\begin{{table*}}[t]
\caption{{Thresholds tuned once, tested elsewhere}}
\label{{tab:transfer}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}l{'c' * n}@{{}}}}
\toprule
 & \multicolumn{{{n}}}{{c}}{{AW, 2 Byzantine cameras: accuracy / false alarms}} \\
\cmidrule(l){{2-{n + 1}}}
Method & {head} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / "tab_transfer.tex").write_text(tex)


# ------------------------------------- correlated detector errors (exp13, Oct 2026) --
CORR_CASES = [("independent", "none"), ("S_p0.1_h1", "none"), ("S_p0.1_h2", "none"),
              ("S_p0.3_h1", "none"), ("S_p0.3_h2", "none"), ("T_q0.05", "none"), ("T_q0.10", "none"),
              ("C_a0.25", "none"), ("C_a0.5", "none"), ("S_p0.3_h2", "AW")]
CORR_MD = {"independent": "independent errors (base)", "S_p0.1_h1": "spatial p = 0.1, h = 1",
           "S_p0.1_h2": "spatial p = 0.1, h = 2", "S_p0.3_h1": "spatial p = 0.3, h = 1",
           "S_p0.3_h2": "spatial p = 0.3, h = 2", "T_q0.05": "temporal (stuck) q = 0.05",
           "T_q0.10": "temporal (stuck) q = 0.10", "C_a0.25": "adverse condition a = 0.25",
           "C_a0.5": "adverse condition a = 0.5"}
CORR_ORDER = ["local", "unfiltered", "majority", "coord_median", "wmsr_F2", "bf_v1", "bf_v2", "oracle"]


def correlated_report(R):
    p = R / "exp13_correlated/summary.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    L = ["# Correlated detector errors (exp13_correlated)", "",
         "18-camera layout, k = 4, `detector_base` bank, 50 seeds, normal outputs. Error models",
         "(src/bflood/errors.py) on honest cameras, on top of the independent bank draws: spatial",
         "patches (shared error of +-1 level within h hops, probability p per round), temporal",
         "stuck faults (probability q per round, mean 5 rounds, one wrong belief repeated), adverse",
         "condition (a contiguous block of a fraction a of the rounds drawn from the adverse bank).",
         "Gain = accuracy minus that of the single camera in the same case. Generated by",
         "scripts/make_figures.py.", ""]
    adv_md = R / "detector_base/ensemble/ADVERSE_BANK.md"
    if adv_md.exists():
        L += ["Adverse bank (results/detector_base/ensemble/ADVERSE_BANK.md):", ""]
        L += [x for x in adv_md.read_text().splitlines() if x.startswith("|")] + [""]
    L += ["## Summary (no attack; last row AW with 2 Byzantine)", "",
          "| Case | single camera acc / FA | ByzantineFlood acc / FA | W-MSR (F=2) acc / FA | unfiltered mean acc / FA | ByzantineFlood gain |",
          "|---|---|---|---|---|---|"]
    for st, adv in CORR_CASES:
        d = sel(s, setting=st, adversary=adv)
        if d.empty:
            continue
        g = {m: sel(d, method=m).iloc[0] for m in CORR_ORDER}
        name = CORR_MD[st] + (", AW 2 Byzantine" if adv == "AW" else "")
        L.append(f"| {name} | " + " | ".join(f"{g[m].acc:.3f} / {g[m].false_alarm:.3f}"
                                              for m in ("local", "bf_v2", "wmsr_F2", "unfiltered"))
                 + f" | {g['bf_v2'].acc - g['local'].acc:+.3f} |")
    bugs = []
    for st in CORR_MD:
        for adv in ("none", "AW"):
            d = sel(s, setting=st, adversary=adv)
            if d.empty:
                continue
            g = {m: sel(d, method=m).iloc[0] for m in CORR_ORDER}
            r = g["bf_v2"]
            if r.f_local <= 2 and r.range_viol > 0:
                bugs.append(f"{st}, {adv}: ByzantineFlood range_viol {r.range_viol:.4f} at f = {r.f_local}")
            L += ["", f"## {CORR_MD[st]}, {'no attack' if adv == 'none' else 'AW, 2 Byzantine'}", "",
                  "| Method | acc | FA | delay | shallow recall | range viol. % | gain over single camera |",
                  "|---|---|---|---|---|---|---|"]
            for m in CORR_ORDER:
                x = g[m]
                L.append(f"| {TRADEOFF_STYLE[m][0]} | {x.acc:.3f} | {x.false_alarm:.3f} | {x.delay:.2f} | "
                         f"{x.shallow_recall:.3f} | {100 * x.range_viol:.2f} | {x.acc - g['local'].acc:+.3f} |")
    L += ["", "## Theorem 2 check", "",
          ("ByzantineFlood range violations are 0 at f <= 2 in every case (runs with f <= 2)." if not bugs
           else "**BUG: nonzero ByzantineFlood range violations at f <= 2:** " + "; ".join(bugs))]
    (R / "exp13_correlated/CORRELATED.md").write_text("\n".join(L) + "\n")
    return bugs


def tab_correlated(R, out):
    p = R / "exp13_correlated/summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    rows, gains = [], ["Gain of ByzantineFlood"]
    for m, lab in THRESHOLD_ROWS:
        cells = [lab]
        for st, adv in CORR_CASES:
            r = sel(s, setting=st, adversary=adv, method=m).iloc[0]
            cells.append(f"{f2(r.acc)}/{f2(r.false_alarm)}")
        rows.append(" & ".join(cells) + r" \\")
    for st, adv in CORR_CASES:
        a = sel(s, setting=st, adversary=adv, method="bf_v2").acc.iloc[0]
        b = sel(s, setting=st, adversary=adv, method="local").acc.iloc[0]
        gains.append(f"{a - b:+.2f}")
    tex = rf"""% generated by scripts/make_figures.py -- do not edit by hand
% source: results/exp13_correlated (normal outputs; no attack except the last column)
\begin{{table*}}[t]
\caption{{Correlated detector errors}}
\label{{tab:correlated}}
\centering\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lcccccccccc@{{}}}}
\toprule
 & & \multicolumn{{4}}{{c}}{{Spatial ($p$, $h$)}} & \multicolumn{{2}}{{c}}{{Stuck ($q$)}} & \multicolumn{{2}}{{c}}{{Adverse ($a$)}} & AW, 2 Byz. \\
\cmidrule(lr){{3-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-10}}\cmidrule(l){{11-11}}
Method & Indep. & 0.1, 1 & 0.1, 2 & 0.3, 1 & 0.3, 2 & 0.05 & 0.10 & 0.25 & 0.5 & S 0.3, 2 \\
\midrule
{chr(10).join(rows)}
\midrule
{" & ".join(gains)} \\
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / "tab_correlated.tex").write_text(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--only", default=None,
                    help="comma list of outputs to make (e.g. tradeoff,pond); default all")
    a = ap.parse_args()
    R, P = Path(a.results), Path(a.out)
    figs, tabs = P / "figures", P / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    if a.only:
        jobs = {"main": lambda: (tab_main(R, tabs), fig_main(R, figs)),
                "adversaries": lambda: tab_adversaries(R, tabs),
                "threshold": lambda: tab_threshold(R, tabs),
                "tradeoff": lambda: (fig_tradeoff(R, figs), tradeoff_report(R)),
                "pond": lambda: tab_pond(R, tabs),
                "transfer": lambda: (transfer_report(R), tab_transfer(R, tabs)),
                "correlated": lambda: (correlated_report(R), tab_correlated(R, tabs))}
        for k in a.only.split(","):
            jobs[k.strip()]()
        print("made:", a.only)
        return
    fig_main(R, figs)
    fig_detector_quality(R, figs)
    fig_scale(R, figs)
    fig_faults(R, figs)
    fig_degree(R, figs)
    fig_system(figs)
    tab_main(R, tabs)
    tab_adversaries(R, tabs)
    tab_threshold(R, tabs)
    tab_ablation(R, tabs)
    tab_scale(R, tabs)
    tab_hops(R, tabs)
    tab_detector(R, tabs)
    fig_tradeoff(R, figs)
    tradeoff_report(R)
    tab_pond(R, tabs)
    transfer_report(R)
    tab_transfer(R, tabs)
    correlated_report(R)
    tab_correlated(R, tabs)
    print("layout: %d links, degree %d-%d" % fig_layout(R, figs))
    print("figures ->", figs, "| tables ->", tabs)


if __name__ == "__main__":
    main()
