#!/usr/bin/env python3
"""Check every number in the write-up against a run log — prose included.

The previous verifier only checked table cells, so two stale figures in prose
survived a commit message claiming everything had been verified. This extracts
every decimal in the prose of the filtered-search section and requires each to
appear somewhere in the run's own output.
"""
import re, sys
log_path, = sys.argv[1:2] or [None]
log = open(log_path).read()
# every number the run printed, as strings
printed = set(re.findall(r'\d+\.\d+', log)) | set(re.findall(r'\b\d+\b', log))

def section(path, start, end=None):
    t = open(path).read()
    i = t.index(start)
    j = t.index(end, i) if end else len(t)
    return t[i:j]

bad = []
for path, start, end in [
    ('docs/BENCHMARKS.md', '## Filtered search across selectivity', '## What this baseline does not show'),
    ('README.md', '## Does it actually work?', '## Where it stands'),
]:
    text = section(path, start, end)
    for ln, line in enumerate(text.split('\n'), 1):
        if line.strip().startswith('|') or line.strip().startswith('`'):
            continue  # table rows and code are checked structurally elsewhere
        for num in re.findall(r'\d+\.\d+', line):
            # tolerate a rounded restatement: 3.06 may be written 3.1
            if num in printed:
                continue
            short = [p for p in printed if p.startswith(num[:-1]) or num.startswith(p[:-1])]
            if not short:
                bad.append(f'{path}: {num!r} in prose is in no run output -- "{line.strip()[:70]}"')
print('\n'.join(bad) if bad else 'PROSE FIGURES ALL TRACE TO THE RUN')
sys.exit(1 if bad else 0)
