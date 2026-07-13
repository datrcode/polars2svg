# polars2svg

**Polars-native DataFrame → SVG visualizations for Jupyter.**

📖 **Documentation & gallery:** https://datrcode.github.io/polars2svg_prod/

This repository hosts the public documentation site (and, eventually, the
released package source) for polars2svg.

## Building the docs locally

```bash
pip install mkdocs-material
mkdocs serve          # live preview at http://127.0.0.1:8000
```

The rendered SVG examples in `docs/examples/` are generated from seeded data
by `docs/gen_examples.py` and committed, so the docs build needs no
dependencies beyond mkdocs-material.
