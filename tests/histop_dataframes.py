import polars as pl
import random

__name__ = 'histop_dataframes'


def makeHistoDf(n=100, seed=42):
    '''Create a DataFrame suitable for histop testing.

    Returns a pl.DataFrame with columns:
      cat     (pl.Utf8)    – random choice of 'A', 'B', 'C'
      group   (pl.Utf8)    – random choice of 'x', 'y'
      value   (pl.Int32)   – random int 1-100
      score   (pl.Float64) – random float 0-10
    '''
    rng = random.Random(seed)
    rows = {'cat': [], 'group': [], 'value': [], 'score': []}
    for _ in range(n):
        rows['cat'].append(rng.choice(['A', 'B', 'C']))
        rows['group'].append(rng.choice(['x', 'y']))
        rows['value'].append(rng.randint(1, 100))
        rows['score'].append(round(rng.uniform(0.0, 10.0), 3))
    return pl.DataFrame(rows)


def makeOrderedHistoDf():
    '''DataFrame with known per-bin row counts: A=5, B=3, C=1.

    Useful for asserting _sorted_bins_ ordering.
    '''
    return pl.DataFrame({
        'cat':   ['A'] * 5 + ['B'] * 3 + ['C'] * 1,
        'group': ['x'] * 5 + ['y'] * 3 + ['x'] * 1,
        'value': [10, 20, 30, 40, 50, 5, 15, 25, 7],
        'score': [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 0.7],
    })
