#!/usr/bin/env python
"""Regenerate the README gallery images in docs/images/.

polars2svg renders to SVG. PyPI does not render SVG in READMEs (and strips it),
so the landing-page gallery needs PNGs. svglib (the dev-only rasterizer used by
the golden-image tests) silently drops the plot marks, so these PNGs are produced
by a headless Chrome screenshot of the SVG at 2x device scale for crisp output.

Usage:
    .venv/bin/python docs/generate_images.py

Requires Google Chrome installed at the macOS default path (edit CHROME below for
other platforms). Deterministic: seeds are fixed, so re-running reproduces the
same images byte-for-similar (Chrome rasterization is stable across runs on a
given machine).
"""
import os
import re
import random
import tempfile
import subprocess

import networkx as nx
import polars as pl
from polars2svg import Polars2SVG

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
SCALE = 2

p2s = Polars2SVG()


def _svg_dims(svg):
    w = re.search(r'width="(\d+(?:\.\d+)?)"', svg)
    h = re.search(r'height="(\d+(?:\.\d+)?)"', svg)
    return int(float(w.group(1))), int(float(h.group(1)))


def render_png(svg, name):
    w, h = _svg_dims(svg)
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<style>*{margin:0;padding:0}html,body{background:#fff}</style>'
        f"</head><body>{svg}</body></html>"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=IMAGES_DIR) as f:
        f.write(html)
        html_path = f.name
    out = os.path.join(IMAGES_DIR, name)
    try:
        subprocess.run(
            [
                CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                f"--force-device-scale-factor={SCALE}",
                f"--window-size={w},{h}",
                "--default-background-color=FFFFFFFF",
                f"--screenshot={out}",
                f"file://{html_path}",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        os.unlink(html_path)
    print(f"saved {out} ({w * SCALE}x{h * SCALE})")


def make_xyp():
    random.seed(7)
    n = 120
    df = pl.DataFrame({
        "x":     [random.gauss(0, 1) for _ in range(n)],
        "y":     [random.gauss(0, 1) for _ in range(n)],
        "group": [random.choice(["alpha", "beta", "gamma"]) for _ in range(n)],
    })
    render_png(p2s.xyp(df, "x", "y", color="group", dot_size=5, wxh=(360, 300)).svg,
               "xyp_scatter.png")


def make_linkp():
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
        "src": [a for a, _ in edges],
        "dst": [b for _, b in edges],
        "tier": [tier[a] for a, _ in edges],
    })
    # Explicit positions: without pos= linkp places nodes with random.random() in an
    # order that varies between processes, so random.seed() alone does not pin it.
    pos = nx.spring_layout(nx.Graph(edges), seed=3)
    render_png(
        p2s.linkp(df, [("src", "dst")], pos={n: (float(x), float(y)) for n, (x, y) in pos.items()},
                  node_color="tier", color="tier",
                  node_size="medium", draw_node_labels=True, wxh=(360, 340)).svg,
        "linkp_network.png",
    )


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
    render_png(
        p2s.chordp(df, [("fm", "to")], count="weight", node_size="vary",
                   link_size="vary", color="fm",
                   node_color=p2s.COLOR_BY_NODE_NAME, wxh=(340, 340)).svg,
        "chordp_flows.png",
    )


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    make_xyp()
    make_linkp()
    make_chordp()
