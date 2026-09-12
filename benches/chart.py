#!/usr/bin/env python3
"""Draw the selectivity sweep as an SVG.

Reads the CSV `benches/sql/selectivity.sql` writes and emits a two-panel chart:
recall@k on top, median latency below, both against predicate selectivity, for
the *correlated* label at the default ef_search. That is the realistic shape --
a filter whose matching rows form a region of the vector space -- and the one
the design exists for.

Standard library only, on purpose. This repository has no plotting dependency
and the benchmark's contract is that one documented command regenerates
everything; adding matplotlib for a single figure would make that false.

Usage:  python3 benches/chart.py <results.csv> <out.svg> "<subtitle>"
"""

import csv
import sys

# Selectivity is plotted on a log axis: the interesting behaviour is all at the
# tight end, and a linear axis spends three quarters of its width on the gap
# between 50% and 10% where every engine is fine.
from math import log10

ENGINES = [
    ("brindle", "#2b6cb0", "Brindle (predicate in traversal)"),
    ("pgv_iter", "#b7791f", "pgvector iterative scan"),
    ("pgv_post", "#c53030", "pgvector post-filter"),
    ("exact", "#4a5568", "exact scan (the ceiling)"),
]

W, H = 760, 592  # the wrapped legend row needs space for descenders
PAD_L, PAD_R, PAD_T = 64, 18, 20
PANEL_H, PANEL_GAP = 200, 76


def read(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["sel"] = int(r["sel"])
        r["ef"] = int(r["ef"])
        r["recall"] = float(r["recall"])
        r["p50_ms"] = float(r["p50_ms"])
    return rows


def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def panel(out, rows, sels, top, title, value_of, fmt, ymax=None, ylog=False):
    """One panel: axes, gridlines, and a polyline per engine."""
    x_of = lambda s: PAD_L + (W - PAD_L - PAD_R) * (
        (log10(sels[0]) - log10(s)) / (log10(sels[0]) - log10(sels[-1]))
    )
    hi = ymax if ymax else max(value_of(r) for r in rows) * 1.15
    lo = 0.0
    if ylog:
        lo = min(value_of(r) for r in rows) * 0.8
        y_of = lambda v: top + PANEL_H - PANEL_H * (
            (log10(max(v, lo)) - log10(lo)) / (log10(hi) - log10(lo))
        )
    else:
        y_of = lambda v: top + PANEL_H - PANEL_H * (v - lo) / (hi - lo)

    out.append(f'<text x="{PAD_L}" y="{top - 8}" class="ttl">{esc(title)}</text>')

    for i in range(5):
        v = lo + (hi - lo) * i / 4 if not ylog else lo * (hi / lo) ** (i / 4)
        y = y_of(v)
        out.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" class="grid"/>'
        )
        out.append(f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" class="ax ar">{fmt(v)}</text>')

    for s in sels:
        x = x_of(s)
        out.append(
            f'<text x="{x:.1f}" y="{top + PANEL_H + 18}" class="ax mid">{s}%</text>'
        )

    for key, colour, _ in ENGINES:
        pts = [(x_of(r["sel"]), y_of(value_of(r))) for r in rows if r["engine"] == key]
        if not pts:
            continue
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="2.4"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{colour}"/>')


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    subtitle = sys.argv[3] if len(sys.argv) > 3 else ""

    rows = read(src)
    efs = sorted({r["ef"] for r in rows})
    ef = efs[0]  # the default ef_search: what a user gets untuned
    data = [r for r in rows if r["shape"] == "local" and r["ef"] == ef]
    if not data:
        print("chart: no correlated-label rows in the results", file=sys.stderr)
        return 1
    sels = sorted({r["sel"] for r in data}, reverse=True)
    if len(sels) < 2:
        print("chart: need at least two selectivity points", file=sys.stderr)
        return 1

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        "<style>"
        ".ax{font-size:11px;fill:#4a5568}.ar{text-anchor:end}.mid{text-anchor:middle}"
        ".ttl{font-size:13px;fill:#1a202c;font-weight:600}"
        ".sub{font-size:11px;fill:#718096}.lg{font-size:12px;fill:#2d3748}"
        ".grid{stroke:#e2e8f0;stroke-width:1}"
        "</style>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="26" class="ttl" style="font-size:15px">'
        f"Filtered vector search as the predicate tightens</text>",
        f'<text x="{PAD_L}" y="44" class="sub">'
        f"{esc(subtitle)} &#183; ef_search {ef} &#183; label correlated with vector position"
        f" &#183; higher is better above, lower is better below</text>",
    ]

    top1 = PAD_T + 58
    panel(out, data, sels, top1, "recall@10 (1.0 = the exact answer)",
          lambda r: r["recall"], lambda v: f"{v:.2f}", ymax=1.05)

    top2 = top1 + PANEL_H + PANEL_GAP
    panel(out, data, sels, top2, "median latency, ms (log scale)",
          lambda r: r["p50_ms"], lambda v: f"{v:.2f}", ylog=True)

    out.append(
        f'<text x="{PAD_L + (W - PAD_L - PAD_R) / 2:.0f}" y="{top2 + PANEL_H + 40}"'
        f' class="ax mid">share of rows matching the predicate</text>'
    )

    # two legend rows: start high enough that the wrapped one clears the edge
    lx, ly = PAD_L, H - 36
    # four entries do not fit on one row at this width
    for key, colour, label in ENGINES:
        out.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 22}" y2="{ly - 4}" stroke="{colour}" stroke-width="2.4"/>')
        out.append(f'<circle cx="{lx + 11}" cy="{ly - 4}" r="3.4" fill="{colour}"/>')
        out.append(f'<text x="{lx + 28}" y="{ly}" class="lg">{esc(label)}</text>')
        lx += 28 + int(len(label) * 6.4) + 18
        if lx > W - 200:
            lx, ly = PAD_L, ly + 16

    # A machine-readable record of what was plotted. The chart has twice been
    # committed a run behind the tables it illustrates, and that was caught both
    # times by hand-inverting the polyline coordinates. This lets
    # benches/verify_writeup.py do it instead.
    out.append("<!-- plotted "
               + ";".join(f"{r['engine']}:{r['sel']}:{r['recall']:.4f}:{r['p50_ms']:.3f}"
                          for r in sorted(data, key=lambda r: (r["engine"], -r["sel"])))
               + " -->")
    out.append("</svg>")
    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
