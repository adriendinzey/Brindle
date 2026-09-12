#!/usr/bin/env python3
"""Check that every figure in the filtered-search write-up came from a real run.

    python3 benches/verify_writeup.py <run log>   # non-zero exit on a mismatch

A figure matches if some number the run printed rounds to it *at the precision it
is quoted to*: a sentence may say "3.2 ms" for a measured 3.196. `0.17` against a
measured `0.193` is not rounding, and is caught.

That rule is the whole point, and the first version of this file got it wrong in
a way worth recording. It tested *membership* — "does this string appear anywhere
in the log" — with a fuzzy fallback that reduced to `num.startswith("")`, which is
true of every string. It could not fail. It passed a log whose entire content was
`total 0 rows`, and it passed the very commit it was written to guard, which
carried four stale figures.

Membership was the wrong question regardless. A document quoting dozens of
numbers will have most of its *stale* values appear somewhere in a long log; two
of those four would have survived a repaired membership test. What matters is
whether a figure corresponds to a measurement at its own precision.

Self-test: `python3 benches/verify_writeup.py --self-test` exercises the matcher
against the cases this file has actually failed on.
"""

import re
import sys
from decimal import Decimal, ROUND_HALF_UP

# Only the sections that report this benchmark. The rest of BENCHMARKS.md quotes
# the unfiltered baseline, which this log does not contain.
SECTIONS = [
    ("docs/BENCHMARKS.md", "## Filtered search across selectivity",
     "## What this baseline does not show"),
    ("README.md", "## Does it actually work?", "## Where it stands"),
]

# Derived figures — ratios and multipliers — are arithmetic on measured values
# rather than measurements, so they cannot be traced to a printed number.
RATIO = re.compile(r"\d+(?:\.\d+)?\s*(?:x|×)")

# Figures the write-up deliberately cites from somewhere other than this run.
# Each needs a reason: the point of this file is that an unexplained number is a
# defect, so an explained one has to be explained here rather than waved through.
# Not measurements of an engine: a definitional ceiling, a build parameter, a
# correlation coefficient. Each still has to be right, but no table cell
# produces them.
# Values that no table cell produces: a build parameter, and the exact arm's
# definitional 1.000. Deliberately short. An earlier version also listed 0.98,
# 0.997 and 0.993 -- 0.98 was exempting pgvector's *measured* wide-beam recall,
# which is a measurement wearing a definition's clothes, and the other two
# matched no prose at all and would have silently waved through a future figure.
STRUCTURAL = {"1.0"}

CROSS_RUN = {
    "0.057": "pgvector rebuild range, low end, observed on an earlier run",
    "0.163": "pgvector rebuild range, high end, observed on an earlier run",
    "0.053": "strict_order on an earlier build, quoted as a range endpoint",
    "0.183": "strict_order on this build, from the scan-budget table",
    "0.773": "relaxed_order at an open budget, from the scan-budget table",
    "0.53":  "the ef_search ceiling before the fragmentation fix, quoted from it",
    "0.004": "ctid/label correlation, measured once against the live fixture",
    "1.000": "the other half of that correlation pair, and the exact arm's ceiling",
    "0.803": "the in-place re-run that produced a different graph, by design",
}


def measured(log):
    out = set()
    for tok in re.findall(r"\d+\.\d+|\b\d+\b", log):
        try:
            out.add(float(tok))
        except ValueError:
            pass
    return out


def matches(tok, nums):
    """True if some measured number rounds to `tok` at `tok`'s own precision.

    Half-up, not Python's banker's rounding: a measured 0.715 written as "0.72"
    is a correct quotation, and `round(0.715, 2)` answers 0.71.
    """
    places = len(tok.split(".")[1]) if "." in tok else 0
    q = Decimal(1).scaleb(-places)
    target = Decimal(tok)
    return any(Decimal(repr(n)).quantize(q, rounding=ROUND_HALF_UP) == target
               for n in nums)


PROSE_CELLS = ({}, {})


def check(nums):
    bad = []
    for path, start, end in SECTIONS:
        text = open(path).read()
        i = text.index(start)
        body = text[i:text.index(end, i)]
        for line in body.split("\n"):
            stripped = line.strip()
            kind = "table" if stripped.startswith("|") else "prose"
            for m in re.finditer(r"\d+\.\d+", line):
                tok = m.group()
                # skip ratios like "8.1x" / "44×"
                if RATIO.match(line[m.start():m.start() + len(tok) + 3]):
                    continue
                if kind == "prose" and tok in CROSS_RUN:
                    continue
                if kind == "prose" and tok in STRUCTURAL:
                    continue
                if kind == "table":
                    if not matches(tok, nums):
                        bad.append(f"{path} (table): {tok} is in no measurement — "
                                   f'"{stripped[:64]}"')
                    continue
                # Prose. Membership in the log is not enough: with a few hundred
                # measured values a stale figure lands on an unrelated cell often
                # enough to be the norm rather than the exception -- 0.963 quoted
                # for an uncorrelated recall matched a *correlated* one, and
                # passed. A prose figure must name the cell it is quoting.
                tail = line[m.end():m.end() + 80]
                cite = re.match(r"\s*<!--\s*([a-z]+),(\d+),(\d+|-),"
                                r"(brindle|pgv_iter|pgv_post|exact),"
                                r"(recall|ms)\s*-->", tail)
                if not cite:
                    bad.append(f"{path} (prose): {tok} names no cell — add "
                               f"<!--shape,sel,ef,engine,recall|ms--> after it, or "
                               f'restate it as a reference to the table: "{stripped[:52]}"')
                    continue
                shape, sel, ef, engine, what = cite.groups()
                key = (shape, int(sel), None if ef == "-" else int(ef), engine)
                got = (PROSE_CELLS[0] if what == "recall" else PROSE_CELLS[1]).get(key)
                if got is None:
                    bad.append(f"{path} (prose): {tok} cites {shape},{sel},{ef},"
                               f"{engine},{what}, which the run has no cell for")
                elif got.quantize(Decimal(1).scaleb(-len(tok.split(".")[1])),
                                  rounding=ROUND_HALF_UP) != Decimal(tok):
                    bad.append(f"{path} (prose): {tok} cites {shape} {sel}%/ef{ef} "
                               f"{engine} {what}, which measured {got}")
    return bad


def log_tables(log):
    """The harness's own tables, keyed by what each cell describes.

    Parsing these is the difference between checking a claim and checking a
    coincidence. An earlier version asked only "does this number appear in the
    log", which a 3-decimal recall can satisfy by colliding with an unrelated
    latency -- an injected 0.899 in place of a measured 0.940 passed, because
    0.899 was some other cell's millisecond figure.
    """
    recall, lat = {}, {}
    for shape, marker in (("local", "CORRELATED label"),
                          ("spread", "uncorrelated label")):
        if marker not in log:
            continue
        blk = log.split(marker, 1)[1].split("rows)", 1)[0]
        for line in blk.split("\n"):
            m = re.match(r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
                         r"\s*([\d.]+)\s*\|\s*([\d.]+)", line)
            if m:
                sel, ef, b, pp, pi = m.groups()
                for engine, v in (("brindle", b), ("pgv_post", pp), ("pgv_iter", pi)):
                    recall[(shape, int(sel), int(ef), engine)] = Decimal(v)
    if "=== median latency" in log:
        for line in log.split("=== median latency", 1)[1].split("\n"):
            m = re.match(r"\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(\d*)\s*\|\s*(\w+)"
                         r"\s*\|\s*([\d.]+)", line)
            if m:
                shape, sel, ef, engine, ms = m.groups()
                lat[(shape, int(sel), int(ef) if ef.strip() else None, engine)] = Decimal(ms)
    if "the exact arm" in log:
        blk = log.split("the exact arm", 1)[1].split("rows)", 1)[0]
        for line in blk.split("\n"):
            m = re.match(r"\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", line)
            if m:
                shape, sel, rc, ms = m.groups()
                recall[(shape, int(sel), None, "exact")] = Decimal(rc)
                lat[(shape, int(sel), None, "exact")] = Decimal(ms)
    return recall, lat


def tables_agree(log):
    """BENCHMARKS.md's two predicate tables, cell by cell, against the harness.

    Scope, stated because overstating it is the mistake this file keeps making:
    this covers the Correlated and Uncorrelated tables in `docs/BENCHMARKS.md`
    only. The README's headline table, the scan-budget table and the ef_search
    ladder are *not* checked cell-by-cell -- they fall back to the weaker
    "some measurement rounds to this" test in `check()`. Extending it means
    giving those tables the same shape of key.
    """
    recall, lat = log_tables(log)
    if not recall:
        return ["the log carries no result tables; is it a completed run?"]
    bench = open("docs/BENCHMARKS.md").read()
    bad = []
    for heading, shape in (("### Correlated predicate", "local"),
                           ("### Uncorrelated predicate", "spread")):
        i = bench.index(heading)
        seg = bench[i:bench.index("###", i + 5)]
        for sel, ef, rest in re.findall(r"^\| (\d+)% \| (\d+) \|(.+)$", seg, re.M):
            cells = [c.strip().replace("**", "") for c in rest.split("|")]
            for engine, cell in zip(("brindle", "pgv_iter", "pgv_post", "exact"), cells):
                if not cell or cell == '"':
                    continue
                key = (shape, int(sel), None if engine == "exact" else int(ef), engine)
                parts = [x.strip() for x in cell.split("/")]
                for want_txt, table in ((parts[0], recall),
                                        (parts[1].replace("ms", "").strip()
                                         if len(parts) > 1 else None, lat)):
                    if want_txt is None:
                        continue
                    got = table.get(key)
                    if got is None:
                        bad.append(f"BENCHMARKS {shape} {sel}%/ef{ef} {engine}: "
                                   "no such cell in the run")
                        continue
                    q = Decimal(1).scaleb(-len(want_txt.split(".")[1]))
                    if got.quantize(q, rounding=ROUND_HALF_UP) != Decimal(want_txt):
                        bad.append(f"BENCHMARKS {shape} {sel}%/ef{ef} {engine}: "
                                   f"published {want_txt}, run measured {got}")
    return bad


def chart_agrees(path="docs/assets/selectivity.svg"):
    """Check the committed chart against the committed correlated table.

    The chart has twice been committed a run behind the tables it illustrates --
    once plotting pgvector at 0.077 while the table three lines below said 0.117,
    in Brindle's favour. Both times a reviewer caught it by inverting the
    polyline coordinates. `chart.py` now records what it plotted, so this is a
    lookup rather than a reconstruction.
    """
    svg = open(path).read()
    m = re.search(r"<!-- plotted ([^>]+?) -->", svg)
    if not m:
        return [f"{path}: carries no record of what it plotted; regenerate it"]
    plotted = {}
    for item in m.group(1).split(";"):
        engine, sel, recall, p50 = item.split(":")
        plotted[(engine, int(sel))] = (Decimal(recall), Decimal(p50))

    bench = open("docs/BENCHMARKS.md").read()
    i = bench.index("### Correlated predicate")
    rows = re.findall(r"^\| (\d+)% \| (\d+) \|(.+)$",
                      bench[i:bench.index("###", i + 5)], re.M)
    bad = []
    for sel, ef, rest in rows:
        if int(ef) != 64:
            continue
        cells = [c.strip().replace("**", "") for c in rest.split("|")]
        for engine, cell in zip(("brindle", "pgv_iter", "pgv_post", "exact"), cells):
            got = plotted.get((engine, int(sel)))
            if got is None:
                bad.append(f"{path}: no {engine} point at {sel}% — the chart is "
                           "missing a series the table has")
                continue
            parts = [x.strip().replace("ms", "").strip() for x in cell.split("/")]
            for idx, txt in enumerate(parts[:2]):
                if not txt:
                    continue
                q = Decimal(1).scaleb(-len(txt.split(".")[1])) if "." in txt else Decimal(1)
                if got[idx].quantize(q, rounding=ROUND_HALF_UP) != Decimal(txt):
                    what = "recall" if idx == 0 else "latency"
                    bad.append(f"{path}: plots {engine} {what} at {sel}% as "
                               f"{got[idx]}, table says {txt} — the chart is from "
                               "a different run")
    return bad


def self_test():
    """The matcher, against cases this file has actually got wrong."""
    nums = {0.193, 3.196, 0.953, 4.057, 0.117}
    cases = [
        ("0.193", True,  "exact value"),
        ("3.2",   True,  "legitimate rounding of 3.196"),
        ("0.17",  False, "stale: 0.193 does not round to 0.17"),
        ("0.963", False, "stale: measured 0.953"),
        ("4.05",  False, "stale: measured 4.057 -> 4.06"),
        ("0.083", False, "stale: measured 0.117"),
        ("0.72",  True,  "half-up: measured 0.715 quoted to 2 places"),
    ]
    nums |= {0.715}
    ok = True
    for tok, want, why in cases:
        got = matches(tok, nums)
        flag = "ok  " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {flag} {tok:>6} -> {got!s:5} (expected {want!s:5}) {why}")
    # The citation resolver and the table/chart comparisons are exercised by
    # mutation rather than here -- see the commit history for the substitutions
    # each was checked against. What this covers is `matches()`, which is where
    # the two silent failures lived.
    #
    # the bug that made the first version vacuous: a near-empty log must not pass
    if matches("0.963", measured("total 0 rows")):
        print("  FAIL a near-empty log still satisfies a figure")
        ok = False
    else:
        print("  ok   a near-empty log satisfies nothing")
    return 0 if ok else 1


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        return self_test()
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    nums = measured(open(sys.argv[1]).read())
    if len(nums) < 50:
        print(f"verify: {sys.argv[1]} yielded only {len(nums)} numbers — "
              "that is not a run log", file=sys.stderr)
        return 2

    log = open(sys.argv[1]).read()
    PROSE_CELLS[0].update(log_tables(log)[0])
    PROSE_CELLS[1].update(log_tables(log)[1])
    bad = check(nums) + tables_agree(log) + chart_agrees()
    if bad:
        print("\n".join(bad))
        print(f"\n{len(bad)} figure(s) do not come from this run.")
        return 1
    print(f"every figure in the write-up traces to this run "
          f"({len(nums)} measured values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
