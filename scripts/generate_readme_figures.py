"""Regenerate README figures from metric tables saved in Project.ipynb.

This script uses only the Python standard library. It parses executed notebook
cells 55, 58, 64, and 76 and writes SVG assets that GitHub renders directly.
"""

import html
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
NOTEBOOK = ROOT / "Project.ipynb"

HEADER_RE = re.compile(r'<th id="[^"]+_level0_col\d+"[^>]*>(.*?)</th>', re.DOTALL)
CELL_RE = re.compile(r'<td id="[^"]+"[^>]*>(.*?)</td>', re.DOTALL)


def clean_html(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def saved_table(notebook, cell_index):
    fragments = []
    for output in notebook["cells"][cell_index].get("outputs", []):
        fragments.extend(output.get("data", {}).get("text/html", []))
    source = "".join(fragments)
    headers = [clean_html(value) for value in HEADER_RE.findall(source)]
    cells = [clean_html(value) for value in CELL_RE.findall(source)]
    if not headers or len(cells) % len(headers):
        raise ValueError(f"Cell {cell_index} does not contain the expected saved table")
    return [dict(zip(headers, cells[i:i + len(headers)])) for i in range(0, len(cells), len(headers))]


def load_metrics():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    final_rows = saved_table(notebook, 76)
    display_names = {
        "KNN (PCA Only)": "KNN (PCA only)",
        "Naive Bayes (PCA + Clustered)": "Naive Bayes (PCA + cluster)",
        "Decision Tree (PCA Only)": "Decision tree (PCA only)",
        "Soft Voting Ensemble": "Soft voting",
        "Weighted Voting Ensemble": "Weighted voting",
    }
    final_models = [
        (display_names[row["Model"]], float(row["ROC-AUC"]), float(row["F1 Score"]))
        for row in final_rows
    ]

    variant_results = {}
    for label, cell_index in (("KNN", 55), ("Naive Bayes", 58), ("Decision tree", 64)):
        rows = saved_table(notebook, cell_index)
        by_model = {row["Model"]: row for row in rows}
        pca, cluster = by_model["PCA Only"], by_model["PCA + Clustered"]
        variant_results[label] = {
            "pca": tuple(float(pca[name]) for name in ("ROC-AUC", "Accuracy", "F1 Score")),
            "cluster": tuple(float(cluster[name]) for name in ("ROC-AUC", "Accuracy", "F1 Score")),
        }
    return final_models, variant_results


def text(x, y, value, size=18, weight="400", anchor="start", fill="#172033"):
    return (
        f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{escape(str(value))}</text>'
    )


def write_svg(name, width, height, body):
    ASSETS.mkdir(exist_ok=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        + "".join(body)
        + "</svg>\n"
    )
    (ASSETS / name).write_text(svg, encoding="utf-8")


def performance_figure(final_models):
    width, height = 1280, 610
    left, right, top = 360, 80, 120
    plot_width = width - left - right
    body = [
        text(50, 48, "Recorded validation performance", 28, "700"),
        text(50, 78, "ROC–AUC and F1 score from notebook cell 76", 17, fill="#5c667a"),
    ]

    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + tick * plot_width
        body.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="535" stroke="#e4e8ef"/>')
        body.append(text(x, 565, f"{tick:.1f}", 15, anchor="middle", fill="#5c667a"))

    row_gap = 80
    for index, (label, auc, f1) in enumerate(final_models):
        y = top + 42 + index * row_gap
        body.append(text(left - 24, y + 5, label, 17, "600", anchor="end"))
        for value, color, offset in ((auc, "#276fbf", -11), (f1, "#d49a00", 11)):
            x = left + value * plot_width
            body.append(f'<line x1="{left}" y1="{y + offset}" x2="{x}" y2="{y + offset}" stroke="{color}" stroke-width="5" opacity="0.28"/>')
            body.append(f'<circle cx="{x}" cy="{y + offset}" r="7" fill="{color}"/>')
            body.append(text(x + 12, y + offset + 5, f"{value:.3f}", 14, "600", fill=color))

    body.extend([
        '<circle cx="505" cy="595" r="6" fill="#276fbf"/>',
        text(520, 600, "ROC–AUC", 15, fill="#39445a"),
        '<circle cx="625" cy="595" r="6" fill="#d49a00"/>',
        text(640, 600, "F1 score", 15, fill="#39445a"),
    ])
    write_svg("model-performance.svg", width, height, body)


def cluster_delta_figure(variant_results):
    width, height = 1280, 620
    left, right, top = 210, 80, 130
    plot_width = width - left - right
    minimum, maximum = -2.0, 14.0

    def scale(value):
        return left + (value - minimum) / (maximum - minimum) * plot_width

    body = [
        text(50, 48, "Effect of appending the K-means cluster label", 28, "700"),
        text(50, 78, "Change from PCA only to PCA + cluster, in percentage points", 17, fill="#5c667a"),
    ]
    for tick in (-2, 0, 2, 4, 6, 8, 10, 12, 14):
        x = scale(tick)
        color = "#7f899c" if tick == 0 else "#e4e8ef"
        width_line = 2 if tick == 0 else 1
        body.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="535" stroke="{color}" stroke-width="{width_line}"/>')
        body.append(text(x, 565, f"{tick:+d}", 15, anchor="middle", fill="#5c667a"))

    metrics = ((0, "ROC–AUC", "#276fbf"), (1, "Accuracy", "#d49a00"), (2, "F1", "#d96c30"))
    row_gap = 125
    bar_height = 24
    for row, (model, values) in enumerate(variant_results.items()):
        base_y = top + 40 + row * row_gap
        body.append(text(left - 24, base_y + 28, model, 17, "600", anchor="end"))
        for metric_index, metric_label, color in metrics:
            delta = (values["cluster"][metric_index] - values["pca"][metric_index]) * 100
            y = base_y + metric_index * 30
            x0, x1 = scale(0), scale(delta)
            bar_x, bar_width = min(x0, x1), max(abs(x1 - x0), 1)
            body.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="3" fill="{color}"/>')
            label_x = x1 + (8 if delta >= 0 else -8)
            anchor = "start" if delta >= 0 else "end"
            body.append(text(label_x, y + 18, f"{delta:+.2f}", 14, "600", anchor=anchor, fill=color))

    legend_x = 390
    for index, (_, metric_label, color) in enumerate(metrics):
        x = legend_x + index * 180
        body.append(f'<rect x="{x}" y="590" width="16" height="16" rx="2" fill="{color}"/>')
        body.append(text(x + 26, 604, metric_label, 15, fill="#39445a"))
    write_svg("cluster-augmentation-effect.svg", width, height, body)


if __name__ == "__main__":
    final_models, variant_results = load_metrics()
    performance_figure(final_models)
    cluster_delta_figure(variant_results)
