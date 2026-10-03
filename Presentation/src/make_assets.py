"""Charts and tables for both decks, drawn from the thesis tables.

Beamer: vector charts in the thesis figure style and the backup tables.
PowerPoint: the same charts as 300 dpi images in the deck's own style.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' styles used by the thesis figures)
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from thesis_data import (EXPECTED_SHARE, IMPUTERS, LEVEL_LABELS, METHOD_COLORS,
                         PATTERN_COLORS, PATTERN_NAMES, PATTERNS, rows, series, zero_vs_best)

OUT = Path(__file__).resolve().parents[1] / "beamer" / "figures"
PPT_OUT = Path(__file__).resolve().parents[1] / "powerpoint" / "assets"
INK, ACCENT, MUTED = "#14213D", "#E4572E", "#6B7280"

THESIS_STYLE = ["science", "grid", "no-latex",
                {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                 "xtick.labelsize": 7, "ytick.labelsize": 7, "grid.alpha": 0.6,
                 "legend.frameon": True, "legend.framealpha": 0.95,
                 "legend.edgecolor": "#BBBBBB", "pdf.fonttype": 42}]
DECK_STYLE = {"font.family": "Arial", "font.size": 13, "axes.labelsize": 13, "axes.edgecolor": "#C9CED6",
              "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
              "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "axes.spines.top": False,
              "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E7EB",
              "grid.linewidth": 0.9, "axes.axisbelow": True, "legend.frameon": False,
              "legend.fontsize": 12, "xtick.major.size": 0, "ytick.major.size": 0,
              "xtick.minor.size": 0, "ytick.minor.size": 0, "savefig.dpi": 300,
              "axes.linewidth": 1.0}

SHARE_TICKS = FixedLocator(EXPECTED_SHARE)
SHARE_FMT = FuncFormatter(lambda x, _: f"{x * 100:g}%")


def share_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(SHARE_TICKS)
    ax.xaxis.set_major_formatter(SHARE_FMT)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.0007, 0.42)
    ax.set_xlabel("expected share of missing cells")


def style_line(model):
    if model == "EM":
        return dict(color=METHOD_COLORS[model], lw=2.0, marker="o", ms=3, zorder=5)
    if model in ("Zero", "Deletion"):
        return dict(color=METHOD_COLORS[model], lw=1.1, ls="--", marker="s", ms=2.5)
    return dict(color=METHOD_COLORS[model], lw=1.0, marker="o", ms=2.2)


def mae_floor():
    fig, ax = plt.subplots(figsize=(3.15, 2.2))
    for m in ["Zero"] + IMPUTERS:
        ax.plot(EXPECTED_SHARE, series("metrics", "sparse", m, "MAE"), label=m, **style_line(m))
    share_axis(ax)
    ax.set_ylim(0.0060, 0.0138)
    ax.set_ylabel("MAE on imputed cells")
    ax.legend(ncol=3, loc="upper right", columnspacing=0.9, handlelength=1.6)
    fig.savefig(OUT / "mae_floor.pdf")
    plt.close(fig)


def coverr_growth():
    fig, ax = plt.subplots(figsize=(2.85, 1.75))
    for m in ["Deletion", "Zero"] + IMPUTERS:
        ax.plot(EXPECTED_SHARE, series("metrics", "sparse", m, "CovErr"), label=m, **style_line(m))
    share_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.0008, 6.0)
    ax.set_ylabel("CovErr (log scale)")
    em = series("metrics", "sparse", "EM", "CovErr")
    ax.text(0.012, 0.0016, f"EM: {em[0]:.4f} to {em[4]:.4f},\nalmost 40 times larger",
            color="#B5541C", fontsize=6.5)
    ax.legend(ncol=4, loc="upper left", handlelength=1.3, fontsize=5.6, columnspacing=0.7,
              handletextpad=0.4, borderpad=0.35)
    fig.savefig(OUT / "coverr_growth.pdf")
    plt.close(fig)


def zero_ratio():
    ratios = zero_vs_best("sparse", 4)
    order = ["MAE", "RMSE", "VaRErr", "WassErr", "CovErr", "AbsACFErr", "LevErr"]
    family = {"MAE": "pointwise", "RMSE": "pointwise", "VaRErr": "marginal",
              "WassErr": "marginal", "CovErr": "covariance", "AbsACFErr": "temporal",
              "LevErr": "temporal"}
    colors = {"pointwise": "#9AA5B1", "marginal": "#7FA7D9", "covariance": ACCENT,
              "temporal": INK}
    fig, ax = plt.subplots(figsize=(2.85, 1.75))
    y = list(range(len(order)))[::-1]
    vals = [ratios[m] for m in order]
    ax.barh(y, vals, color=[colors[family[m]] for m in order], height=0.62, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.15, yi, f"{v:.1f}" + r"$\times$", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="both", length=0)
    ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("Zero error / best imputer error, sparse 0.3")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in colors.items()], loc="upper right",
              fontsize=6, handlelength=1.0)
    fig.savefig(OUT / "zero_ratio.pdf")
    plt.close(fig)


def mechanism():
    fig, ax = plt.subplots(figsize=(3.15, 2.2))
    markers = {"sparse": "o", "gaps": "s", "stressed": "D"}
    for p in PATTERNS:
        vals = series("metrics", p, "EM", "CovErr")
        ax.plot(EXPECTED_SHARE, vals, color=PATTERN_COLORS[p], marker=markers[p], ms=3,
                lw=2.0 if p == "stressed" else 1.3, label=PATTERN_NAMES[p] + (" (MNAR)" if p == "stressed" else " (MCAR)"))
    share_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.0007, 1.5)
    ax.set_ylabel("CovErr of EM (log scale)")
    s, g, st = (series("metrics", p, "EM", "CovErr")[4] for p in PATTERNS)
    ax.text(0.2, st, f"{st:.4f}", color=ACCENT, fontsize=7, ha="right", va="center")
    ax.text(0.33, 0.017, f"sparse {s:.4f}\ngaps {g:.4f}", color=INK, fontsize=6.5,
            ha="right", va="top", linespacing=1.15)
    ax.legend(loc="upper left", handlelength=1.8)
    fig.savefig(OUT / "mechanism.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- PowerPoint charts
def _deck_save(fig, name):
    fig.savefig(PPT_OUT / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def _deck_line(model):
    if model == "EM":
        return dict(color=METHOD_COLORS[model], lw=3.6, marker="o", ms=7.5, zorder=5)
    if model in ("Zero", "Deletion"):
        return dict(color=METHOD_COLORS[model], lw=2.0, ls=(0, (4, 2.5)), marker="s", ms=5.5)
    return dict(color=METHOD_COLORS[model], lw=2.0, marker="o", ms=5.5)


def _top_legend(ax, ncol, **kw):
    ax.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0, 1.01), handlelength=1.8,
              columnspacing=1.1, borderaxespad=0, **kw)


def deck_mae_floor():
    fig, ax = plt.subplots(figsize=(7.3, 4.95))
    for m in ["Zero"] + IMPUTERS:
        ax.plot(EXPECTED_SHARE, series("metrics", "sparse", m, "MAE"), label=m, **_deck_line(m))
    share_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0.006, 0.0125)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.set_ylabel("MAE on imputed cells")
    _top_legend(ax, 6)
    fig.tight_layout()
    _deck_save(fig, "chart_mae_floor")


def deck_coverr_growth():
    fig, ax = plt.subplots(figsize=(6.25, 3.75))
    for m in ["Deletion", "Zero"] + IMPUTERS:
        ax.plot(EXPECTED_SHARE, series("metrics", "sparse", m, "CovErr"), label=m, **_deck_line(m))
    share_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_yscale("log")
    ax.set_ylim(0.0008, 1.2)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("CovErr (log scale)")
    em = series("metrics", "sparse", "EM", "CovErr")
    ax.text(0.012, 0.00092, f"EM: {em[0]:.4f} to {em[4]:.4f},\nalmost 40 times larger",
            color="#C2410C", fontsize=12, va="bottom")
    _top_legend(ax, 4, fontsize=11)
    fig.tight_layout()
    _deck_save(fig, "chart_coverr_growth")


def deck_zero_ratio():
    ratios = zero_vs_best("sparse", 4)
    order = ["MAE", "RMSE", "VaRErr", "WassErr", "CovErr", "AbsACFErr", "LevErr"]
    family = {"MAE": "pointwise", "RMSE": "pointwise", "VaRErr": "marginal",
              "WassErr": "marginal", "CovErr": "covariance", "AbsACFErr": "temporal",
              "LevErr": "temporal"}
    colors = {"pointwise": "#9AA5B1", "marginal": "#7FA7D9", "covariance": ACCENT, "temporal": INK}
    fig, ax = plt.subplots(figsize=(5.85, 3.75))
    y = list(range(len(order)))[::-1]
    vals = [ratios[m] for m in order]
    ax.barh(y, vals, color=[colors[family[m]] for m in order], height=0.66, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.15, yi, f"{v:.1f}\u00d7", va="center", fontsize=12, color=INK,
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axvline(1, color=MUTED, lw=1.0, ls=":")
    ax.set_xlim(0, 10.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("error of Zero relative to the best imputer, sparse 0.3")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in colors.items()], loc="upper right",
              fontsize=11, handlelength=1.0)
    fig.tight_layout()
    _deck_save(fig, "chart_zero_ratio")


def deck_mechanism():
    fig, ax = plt.subplots(figsize=(6.9, 4.95))
    markers = {"sparse": "o", "gaps": "s", "stressed": "D"}
    for p in PATTERNS:
        vals = series("metrics", p, "EM", "CovErr")
        label = {"sparse": "Sparse (MCAR)", "gaps": "Gaps (MCAR)", "stressed": "Stressed (MNAR)"}[p]
        ax.plot(EXPECTED_SHARE, vals, color=PATTERN_COLORS[p], marker=markers[p], ms=7.5,
                lw=3.8 if p == "stressed" else 2.6, label=label)
    share_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_yscale("log")
    ax.set_ylim(0.0008, 1.4)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("CovErr of EM (log scale)")
    s, g, st = (series("metrics", p, "EM", "CovErr")[4] for p in PATTERNS)
    ax.text(0.2, st, f"{st:.4f}", color=ACCENT, fontsize=14, fontweight="bold", ha="right",
            va="center")
    ax.text(0.33, 0.018, f"sparse {s:.4f}\ngaps {g:.4f}", color=INK, fontsize=12, ha="right",
            va="top", linespacing=1.2)
    _top_legend(ax, 3)
    fig.tight_layout()
    _deck_save(fig, "chart_mechanism")


# ---------------------------------------------------------------- backup tables
BACKUP = Path(__file__).resolve().parents[1] / "beamer" / "backup"
TEX_LEVELS = {p: labels for p, labels in LEVEL_LABELS.items()}


def _cell(v, bold, fmt):
    if v is None:
        return "--"
    s = format(v, fmt)
    return f"\\textbf{{{s}}}" if bold else s


def metrics_table(pattern):
    lines = ["{\\fontsize{4.9}{5.5}\\selectfont\\setlength{\\tabcolsep}{9pt}",
             "\\setlength{\\aboverulesep}{0.25ex}\\setlength{\\belowrulesep}{0.4ex}",
             "\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
             "Level & Model & MAE & RMSE & CovErr & VaRErr & WassErr & AbsACFErr & LevErr \\\\",
             "\\midrule"]
    for r in rows("metrics", pattern):
        if r["level_index"] > 0 and r["model"] == "Deletion":
            lines.append("\\midrule")
        level = TEX_LEVELS[pattern][r["level_index"]] if r["model"] == "Deletion" else ""
        cells = [_cell(v, b, ".4f") for v, b in zip(r["values"], r["bold"])]
        lines.append(f"{level} & {r['model']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    (BACKUP / f"metrics_{pattern}.tex").write_text("\n".join(lines) + "\n")


def matrix_table(kind, blocks, name):
    """Methods in rows, the fifteen levels in columns, one block per statistic."""
    head = " & ".join(f"\\multicolumn{{5}}{{c}}{{{PATTERN_NAMES[p]}}}" for p in PATTERNS)
    shares = " & ".join(["0.1\\% & 1\\% & 5\\% & 10\\% & 30\\%"] * 3)
    lines = ["{\\fontsize{5.6}{6.5}\\selectfont\\setlength{\\tabcolsep}{3.2pt}",
             "\\begin{tabular}{@{}l" + "|rrrrr" * 3 + "@{}}", "\\toprule",
             f" & {head} \\\\", f"Expected share & {shares} \\\\"]
    reference = "True (oracle)" if kind == "portfolio" else "True (reference)"
    for column, title, fmt in blocks:
        ref = series(kind, "sparse", reference, column)[0]
        ref_txt = "" if ref is None else f" {{\\color{{muted}}(true {format(ref, fmt)})}}"
        lines += ["\\midrule", f"\\multicolumn{{16}}{{@{{}}l}}{{\\textbf{{{title}}}{ref_txt}}} \\\\"]
        for m in IMPUTERS:
            cells = []
            for p in PATTERNS:
                j = {"portfolio": ["Mean", "Variance", "Sharpe", "dw1"],
                     "factors": ["PC1", "Top5", "D1", "D5"]}[kind].index(column)
                for r in rows(kind, p):
                    if r["model"] == m:
                        cells.append(_cell(r["values"][j], r["bold"][j], fmt))
            lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    (BACKUP / f"{name}.tex").write_text("\n".join(lines) + "\n")


def backup_tables():
    BACKUP.mkdir(parents=True, exist_ok=True)
    for p in PATTERNS:
        metrics_table(p)
    matrix_table("portfolio", [("Sharpe", "Sharpe ratio", ".3f"),
                               ("dw1", "$L^1$ weight distance to the oracle", ".2f"),
                               ("Mean", "Annualised mean", ".3f"),
                               ("Variance", "Annualised variance", ".4f")], "portfolio")
    matrix_table("factors", [("PC1", "Variance share of the first component (\\%)", ".2f"),
                             ("Top5", "Variance share of the first five (\\%)", ".2f"),
                             ("D1", "Subspace distance $D_1$", ".4f"),
                             ("D5", "Subspace distance $D_5$", ".4f")], "factors")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    PPT_OUT.mkdir(parents=True, exist_ok=True)
    with plt.style.context(THESIS_STYLE):
        for make in (mae_floor, coverr_growth, zero_ratio, mechanism):
            make()
            print("beamer chart:", make.__name__)
    with plt.style.context(DECK_STYLE):
        for make in (deck_mae_floor, deck_coverr_growth, deck_zero_ratio, deck_mechanism):
            make()
            print("pptx chart:", make.__name__)
    backup_tables()
    print("backup tables")
