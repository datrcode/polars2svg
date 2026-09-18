# Output contract

Most people never need this page. If you are rendering in a notebook, or serving a
Panel view you opened yourself, the contract is not aimed at you — skip to
[Interactivity](interactivity.md).

It matters when you render **on a server, from data you do not control, and embed
the result in a page someone else loads**. `SECURITY.md` calls that *Profile A —
Appliance*, and it is supported for the render path.

## Why there is a contract at all

What makes that profile supportable is not a promise that every label is escaped.
It is that the **output is checkable**: polars2svg generates every element and
attribute in a render itself, from numeric geometry and a fixed vocabulary. So the
finished document can be held against an allow-list, and anything outside it is a
violation by construction.

`polars2svg/svg_contract.py` defines that allow-list. A conforming render:

- contains only the elements and attributes the components actually emit;
- resolves every reference (`href`, `url(#…)`) inside the same document — in
  `<style>` text as well as in attributes, since CSS fetches too;
- carries no event-handler attribute;
- carries no `javascript:`, `vbscript:` or `data:` scheme in any value;
- carries no DOCTYPE.

## Using it

```python
svg = p2s.xyp(df, "x", "y").svg

p2s.assertOutputContract(svg)   # raises OutputContractError on the first violation
                                # -> this is the line to put in front of a response

for v in p2s.checkOutputContract(svg):   # or inspect them yourself
    print(v)
```

!!! warning "It reports, it does not rewrite"
    Neither function is a sanitizer. A violation means the render **is not fit to
    serve** — not that it has been made fit. Gate on the exception; do not try to
    continue with the value you passed in.

## What proves it

`tests/test_profile_conformance.py` runs the contract over the entire golden corpus,
and over every payload in `tests/injection_corpus.py` driven through every untrusted-text
surface in every component. The corpus is itself checked for vacuity — a render that
silently dropped the label would otherwise pass trivially — the checker is checked
against hand-written violating documents, so the suite cannot go green with a broken
checker, and a coverage ratchet fails the suite if a new component ships without
injection cases.

## What you are still responsible for

**`tile()` is outside the profile.** It embeds foreign SVG verbatim — that is the
component, not a defect — so it makes no promise about markup it did not produce.
Do not build an `svg_list` entry from untrusted input.

The contract is not *blind* to it, and the difference matters: `checkOutputContract()`
reads the composed document, so hostile markup embedded through `svg_list` is reported
like any other violation. An appliance that gates on `assertOutputContract()` is covered
even when it tiles markup it did not write. But being caught at the gate is not the same
as being safe to compose, which is why the advice above stands.

**Path parameters are inputs here.** `save()`, `savePNG()`, `savePositions()` and
`loadPositions()` take a path from the caller. Never forward a request field into one.

**Serve it correctly.** Send a `Content-Security-Policy`, and do not serve a render from
its own URL as `image/svg+xml` on a domain that matters — a standalone SVG document is a
same-origin script context. Embed it, or serve it from a separate origin.

**The contract is about markup polars2svg generates.** It is a strong, tested property
of this library's output. It is not a proof that no XSS is possible anywhere in your page.

---

Signatures in the [API reference](../api.md#output-contract). The full threat model,
including the deployment profiles and what is escaped where, is in
[SECURITY.md](https://github.com/datrcode/polars2svg/blob/main/SECURITY.md).
