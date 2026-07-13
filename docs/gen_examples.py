#!/usr/bin/env python
"""Regenerate the rendered SVG examples in docs/examples/.

The docs site is built from static, committed SVGs so the GitHub Pages build
does not need polars2svg installed (the package source lives in a separate
private repo until it is published to PyPI). Run this locally against a
checkout that has polars2svg installed, then commit the changed SVGs:

    /Users/user/repos/polars2svg/.venv/bin/python docs/gen_examples.py

All data is generated from fixed seeds, so re-running reproduces the same
charts (font metrics permitting) and the docs build stays reproducible.
"""
import os
import random
from datetime import datetime, timedelta

import polars as pl
from polars2svg import Polars2SVG

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

p2s = Polars2SVG()


def save(component, name):
    path = os.path.join(EXAMPLES_DIR, name)
    with open(path, "w") as f:
        f.write(component.svg)
    print(f"saved {path}")


def make_xyp():
    random.seed(7)
    n = 120
    df = pl.DataFrame({
        "x":     [random.gauss(0, 1) for _ in range(n)],
        "y":     [random.gauss(0, 1) for _ in range(n)],
        "group": [random.choice(["alpha", "beta", "gamma"]) for _ in range(n)],
    })
    save(p2s.xyp(df, "x", "y", color="group", dot_size=5, wxh=(400, 300), legend=True),
         "xyp_scatter.svg")


def make_histop():
    random.seed(11)
    endpoints = ["/api/search", "/api/login", "/api/items", "/api/cart",
                 "/api/checkout", "/api/profile", "/static/js", "/static/css"]
    weights = [30, 14, 22, 9, 4, 7, 45, 38]
    rows = []
    for ep, w in zip(endpoints, weights):
        for _ in range(w):
            rows.append((ep, max(200, int(random.gauss(6000, 3000)))))
    df = pl.DataFrame({"endpoint": [r[0] for r in rows],
                       "bytes":    [r[1] for r in rows]})
    # bar length = byte volume, bar color = raw row count: the two encodings
    # are orthogonal (see the count=/color= guide)
    save(p2s.histop(df, "endpoint", count="bytes", color=p2s.CROW_MAGNITUDEp,
                    wxh=(420, 220), legend=True),
         "histop_bars.svg")


def make_timep():
    random.seed(13)
    base = datetime(2026, 3, 2)  # a Monday
    dow_weight = [6, 5, 5, 6, 8, 3, 2]  # busier weekdays, quiet weekend
    ts = []
    for day in range(56):  # eight weeks
        t0 = base + timedelta(days=day)
        for _ in range(random.randint(0, dow_weight[t0.weekday()] * 3)):
            ts.append(t0 + timedelta(minutes=random.randint(0, 24 * 60 - 1)))
    df = pl.DataFrame({"timestamp": ts})
    save(p2s.timep(df, "timestamp", wxh=(420, 200)), "timep_linear.svg")
    save(p2s.timep(df, p2s.tField("timestamp", p2s.PT_DoWp), wxh=(420, 200)),
         "timep_periodic.svg")


def make_piep():
    random.seed(17)
    share = {"solar": 34, "wind": 27, "hydro": 18, "nuclear": 13, "gas": 8}
    rows = [k for k, v in share.items() for _ in range(v)]
    df = pl.DataFrame({"source": rows})
    save(p2s.piep(df, "source", style=p2s.DONUTp, draw_labels=True,
                  color="source", wxh=(320, 320)),
         "piep_donut.svg")


def make_linkp():
    random.seed(19)
    edges = [
        ("api", "auth"), ("api", "db"), ("api", "cache"), ("auth", "db"),
        ("web", "api"), ("web", "cdn"), ("worker", "db"), ("worker", "queue"),
        ("queue", "worker"), ("cache", "db"), ("cdn", "web"), ("mobile", "api"),
        ("mobile", "cdn"), ("report", "db"), ("report", "cache"),
    ]
    tier = {
        "web": "edge", "mobile": "edge", "cdn": "edge", "api": "service",
        "auth": "service", "worker": "service", "report": "service",
        "db": "data", "cache": "data", "queue": "data",
    }
    df = pl.DataFrame({
        "src":  [a for a, _ in edges],
        "dst":  [b for _, b in edges],
        "tier": [tier[a] for a, _ in edges],
    })
    save(p2s.linkp(df, [("src", "dst")], node_color="tier", color="tier",
                   node_size="medium", draw_labels=True, wxh=(400, 360),
                   legend=True),
         "linkp_network.svg")


def make_chordp():
    flows = [
        ("North", "South", 8.0), ("North", "East", 5.0), ("North", "West", 3.0),
        ("South", "East", 6.0), ("South", "West", 4.0), ("East", "West", 7.0),
        ("West", "North", 5.0), ("East", "North", 2.0), ("South", "North", 4.0),
    ]
    df = pl.DataFrame({
        "fm":     [a for a, _, _ in flows],
        "to":     [b for _, b, _ in flows],
        "weight": [w for _, _, w in flows],
    })
    save(p2s.chordp(df, [("fm", "to")], count="weight", node_size="vary",
                    link_size="vary", color="fm",
                    node_color=p2s.COLOR_BY_NODE_NAME, wxh=(360, 360)),
         "chordp_flows.svg")


def make_spreadlinesp():
    random.seed(23)
    others = ["bob", "carol", "dave", "erin", "frank", "grace", "heidi"]
    base = datetime(2026, 4, 6)
    rows = []
    for day in range(10):
        t = base + timedelta(days=day)
        for _ in range(random.randint(2, 6)):
            other = random.choice(others)
            if random.random() < 0.5:
                rows.append((other, "alice", t))   # sender → ego
            else:
                rows.append(("alice", other, t))   # ego → receiver
    df = pl.DataFrame({
        "src": [r[0] for r in rows],
        "dst": [r[1] for r in rows],
        "ts":  [r[2] for r in rows],
    })
    save(p2s.spreadlinesp(df, [("src", "dst")], ego="alice", time="ts",
                          wxh=(520, 260)),
         "spreadlinesp_ego.svg")


def make_smallp():
    random.seed(29)
    n = 400
    centers = {"north": (0, 0), "south": (2, 1), "east": (-1, 2), "west": (1, -2)}
    regions = [random.choice(list(centers)) for _ in range(n)]
    df = pl.DataFrame({
        "region": regions,
        "x": [random.gauss(centers[r][0], 0.8) for r in regions],
        "y": [random.gauss(centers[r][1], 0.8) for r in regions],
    })
    tmpl = p2s.xyp(x="x", y="y", dot_size=2, wxh=(190, 140),
                   sm_shared={p2s.SM_X, p2s.SM_Y})
    save(p2s.smallp(df, tmpl, "region", wxh=(420, 340)), "smallp_facets.svg")


MAKERS = [make_xyp, make_histop, make_timep, make_piep, make_linkp,
          make_chordp, make_spreadlinesp, make_smallp]

if __name__ == "__main__":
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    failures = []
    for maker in MAKERS:
        try:
            maker()
        except Exception as e:
            failures.append((maker.__name__, e))
            print(f"FAILED {maker.__name__}: {e!r}")
    if failures:
        raise SystemExit(f"{len(failures)} maker(s) failed: "
                         f"{[name for name, _ in failures]}")
