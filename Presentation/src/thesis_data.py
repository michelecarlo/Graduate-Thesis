"""Numbers for both slide decks, read directly from the thesis tables in Results.tex."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_TEX = ROOT / "Latex" / "Chapters" / "Results.tex"
THESIS_FIGURES = ROOT / "Latex" / "Figures"

PATTERNS = ["sparse", "gaps", "stressed"]
PATTERN_NAMES = {"sparse": "Sparse", "gaps": "Contiguous gaps", "stressed": "Stressed"}
EXPECTED_SHARE = [0.001, 0.01, 0.05, 0.1, 0.3]
LEVEL_LABELS = {
    "sparse": ["0.001", "0.01", "0.05", "0.1", "0.3"],
    "gaps": ["(20, 10)", "(200, 10)", "(1000, 10)", "(1000, 20)", "(2000, 30)"],
    "stressed": ["(0, 0.1, 0.99)", "(0.005, 0.5, 0.99)", "(0.04, 0.2, 0.95)",
                 "(0.05, 0.5, 0.9)", "(0.2, 1, 0.9)"],
}
METHODS = ["Deletion", "Zero", "KNN", "LowRank", "EM", "BRITS", "CSDI"]
IMPUTERS = ["KNN", "LowRank", "EM", "BRITS", "CSDI"]
METRICS = ["MAE", "RMSE", "CovErr", "VaRErr", "WassErr", "AbsACFErr", "LevErr"]

# Colours of the thesis figures (SciencePlots cycle), so new charts read the same way.
METHOD_COLORS = {
    "True": "#000000", "KNN": "#0C5DA5", "LowRank": "#00B945", "EM": "#FF9500",
    "BRITS": "#FF2C00", "CSDI": "#845B97", "Zero": "#474747", "Deletion": "#9E9E9E",
}
PATTERN_COLORS = {"sparse": "#7FA7D9", "gaps": "#14213D", "stressed": "#E4572E"}

_TABLE_LABELS = {
    "metrics": {"sparse": "tab:results_sparse", "gaps": "tab:results_gaps",
                "stressed": "tab:results_stressed"},
    "portfolio": {"sparse": "tab:portfolio_sparses", "gaps": "tab:portfolio_gaps",
                  "stressed": "tab:portfolio_stressed"},
    "factors": {"sparse": "tab:factors_sparses", "gaps": "tab:factors_gaps",
                "stressed": "tab:factors_stressed"},
}
_COLUMNS = {
    "metrics": METRICS,
    "portfolio": ["Mean", "Variance", "Sharpe", "dw1"],
    "factors": ["PC1", "Top5", "D1", "D5"],
}


def _parse_tables(text):
    tables = {}
    pattern = (r"\\begin\{table\}.*?\\label\{(.*?)\}.*?"
               r"\\begin\{tabular\}\{.*?\}(.*?)\\end\{tabular\}")
    for match in re.finditer(pattern, text, re.S):
        label, body = match.group(1), match.group(2)
        rows, header, level = [], None, None
        for line in (l.strip() for l in body.split("\n")):
            if not line.endswith("\\\\"):
                continue
            cells = [c.strip() for c in line[:-2].split("&")]
            if header is None:
                header = cells
                continue
            level = cells[0] or level
            values, bold = [], []
            for cell in cells[2:]:
                bold.append("\\textbf" in cell)
                cell = re.sub(r"\\textbf\{(.*?)\}", r"\1", cell)
                values.append(None if cell in ("--", "-") else float(cell))
            rows.append({"level": level, "model": cells[1], "values": values, "bold": bold})
        tables[label] = rows
    return tables


_TABLES = _parse_tables(RESULTS_TEX.read_text())


def rows(kind, pattern):
    """Rows of one thesis table, each with level index, model, values and bold flags."""
    out, levels = [], []
    for row in _TABLES[_TABLE_LABELS[kind][pattern]]:
        if row["level"] not in levels:
            levels.append(row["level"])
        out.append({**row, "level_index": levels.index(row["level"])})
    assert len(levels) == 5, (kind, pattern, levels)
    return out


def series(kind, pattern, model, column):
    """Values of one column for one model across the five levels of a pattern."""
    j = _COLUMNS[kind].index(column)
    vals = [r["values"][j] for r in rows(kind, pattern) if r["model"] == model]
    assert len(vals) == 5, (kind, pattern, model, column)
    return vals


def value(kind, pattern, level_index, model, column):
    return series(kind, pattern, model, column)[level_index]


def zero_vs_best(pattern="sparse", level_index=4):
    """Ratio of zero imputation's error to the best imputer's, per metric, from rounded table values."""
    level_rows = [r for r in rows("metrics", pattern) if r["level_index"] == level_index]
    zero = next(r for r in level_rows if r["model"] == "Zero")["values"]
    out = {}
    for j, metric in enumerate(METRICS):
        best = min(r["values"][j] for r in level_rows
                   if r["model"] in IMPUTERS and r["values"][j] is not None)
        out[metric] = zero[j] / best
    return out


if __name__ == "__main__":
    for kind in _TABLE_LABELS:
        for p in PATTERNS:
            print(kind, p, len(rows(kind, p)), "rows")
    print("EM CovErr:", {p: series("metrics", p, "EM", "CovErr") for p in PATTERNS})
    print("Zero/best at sparse 0.3:", {k: round(v, 2) for k, v in zero_vs_best().items()})
