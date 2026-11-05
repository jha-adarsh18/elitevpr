"""Item 5: every table, built from the result JSONs and nothing else.

A table may not read a matrix, a frame or a checkpoint. If a number is not in a
result JSON it does not appear in a table, which is what makes a published
table reproducible from the artefacts alone.

Each result JSON is one (method, config, seed, pair). The builders below group
them into the main table (the two headline columns per dataset), the per-pair
table, and an arbitrary sweep over any recorded config field.
"""

import json
from collections import defaultdict
from pathlib import Path

from .protocol import COLUMNS, PAIRS, pair_name

KS = ("1", "5", "10")


def load_results(directory):
    """Every `*.json` written by the scorer, grouped by (run, dataset)."""
    runs = defaultdict(dict)
    for path in sorted(Path(directory).glob("*.json")):
        blob = json.loads(path.read_text())
        if "pair" not in blob or "dataset" not in blob:
            continue                      # a manifest, not a result
        key = (blob["run"], blob["dataset"])
        runs[key][blob["pair"]] = blob
    return runs


def column(run_pairs, dataset, which, k):
    """One headline cell: AR@k averaged over the pairs that form the column."""
    wanted = [pair_name(*p) for p in COLUMNS[dataset][which]]
    values = []
    for name in wanted:
        blob = run_pairs.get(name)
        if blob is None or "ar_at_k" not in blob.get("scores", {}):
            return None
        values.append(blob["scores"]["ar_at_k"][k])
    return sum(values) / len(values)


def main_table(runs, dataset, order=None):
    """run x {day_day, day_night} x K, as AR@K over L = 10/20/30."""
    rows = []
    names = order or sorted({run for run, ds in runs if ds == dataset})
    for run in names:
        pairs = runs.get((run, dataset))
        if not pairs:
            continue
        row = {"run": run}
        for which in ("day_day", "day_night"):
            for k in KS:
                row[f"{which}@{k}"] = column(pairs, dataset, which, k)
        rows.append(row)
    return rows


def per_pair_table(runs, dataset, order=None):
    """run x pair x K, so every headline cell can be traced to its pairs."""
    rows = []
    names = order or sorted({run for run, ds in runs if ds == dataset})
    for run in names:
        pairs = runs.get((run, dataset))
        if not pairs:
            continue
        row = {"run": run}
        for reference, query in PAIRS[dataset]:
            name = pair_name(reference, query)
            blob = pairs.get(name)
            for k in KS:
                row[f"{name}@{k}"] = (
                    blob["scores"]["ar_at_k"][k]
                    if blob and "ar_at_k" in blob.get("scores", {}) else None)
        rows.append(row)
    return rows


def sweep_table(runs, dataset, field, which="day_night", k="1"):
    """One recorded config field against a headline cell."""
    rows = []
    for (run, ds), pairs in runs.items():
        if ds != dataset:
            continue
        any_blob = next(iter(pairs.values()))
        value = any_blob.get("config", {}).get(field)
        if value is None:
            continue
        rows.append({"run": run, field: value,
                     f"{which}@{k}": column(pairs, dataset, which, k)})
    return sorted(rows, key=lambda r: (str(r[field]), r["run"]))


def render(rows, title=None, width=24, places=2):
    """Fixed-width text, so a table can be pasted into a log or an email."""
    if not rows:
        return f"{title or ''}\n  (no rows)\n"
    keys = [k for k in rows[0] if k != "run"]
    out = []
    if title:
        out.append(title)
        out.append("-" * (width + 11 * len(keys)))
    out.append(f"{'run':<{width}}" + "".join(f"{k:>11}" for k in keys))
    for row in rows:
        cells = ""
        for k in keys:
            v = row[k]
            cells += (f"{v:11.{places}f}" if isinstance(v, (int, float))
                      else f"{'-':>11}")
        out.append(f"{str(row['run']):<{width}}" + cells)
    return "\n".join(out) + "\n"


def write_tables(directory, out_path, order=None):
    runs = load_results(directory)
    datasets = sorted({ds for _run, ds in runs})
    text = []
    for dataset in datasets:
        text.append(render(main_table(runs, dataset, order),
                           f"MAIN -- {dataset}: AR@K (%), mean over L = 10/20/30"))
        text.append(render(per_pair_table(runs, dataset, order),
                           f"PER PAIR -- {dataset}: AR@K (%)"))
    rendered = "\n".join(text)
    Path(out_path).write_text(rendered)
    return rendered
