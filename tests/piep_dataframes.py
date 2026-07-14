import polars as pl
import random

__name__ = 'piep_dataframes'


def makePieDf(n=200, seed=42):
    '''Create a DataFrame suitable for piep testing.

    Columns:
      cat     (pl.Utf8)    – random choice of 'A', 'B', 'C', 'D'
      group   (pl.Utf8)    – random choice of 'x', 'y', 'z'  (small-multiple facet)
      value   (pl.Int32)   – random int 1-100
      score   (pl.Float64) – random float 0-10
    '''
    rng = random.Random(seed)
    rows = {'cat': [], 'group': [], 'value': [], 'score': []}
    for _ in range(n):
        rows['cat'].append(rng.choice(['A', 'B', 'C', 'D']))
        rows['group'].append(rng.choice(['x', 'y', 'z']))
        rows['value'].append(rng.randint(1, 100))
        rows['score'].append(round(rng.uniform(0.0, 10.0), 3))
    return pl.DataFrame(rows)


def makeKnownPieDf():
    '''DataFrame with known per-slice row counts: A=5, B=3, C=2  (total 10).'''
    return pl.DataFrame({
        'cat':   ['A'] * 5 + ['B'] * 3 + ['C'] * 2,
        'group': ['x'] * 5 + ['y'] * 3 + ['z'] * 2,
        'value': [10, 20, 30, 40, 50, 5, 15, 25, 7, 3],
    })
