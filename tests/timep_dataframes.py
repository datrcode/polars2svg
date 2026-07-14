import polars as pl
import random
import calendar

__name__ = 'timep_dataframes'


def makeTimeDf(n=50, year=(2020, 2025), month=(1, 12), day=(1, 28),
               hour=0, minute=0, second=0):
    '''Create a datetime DataFrame for timep testing.

    Returns a pl.DataFrame with columns:
      ts        (pl.Datetime) – constructed from the given components
      value     (pl.Int32)    – random int 0-100
      category  (pl.Utf8)     – random choice of 'A', 'B', 'C'
      numeric   (pl.Float64)  – random float 0-10

    Each component may be:
      int        – fixed value for all rows
      (lo, hi)   – independently uniform-random for each row
    '''
    rows = {'ts': [], 'value': [], 'category': [], 'numeric': []}
    for _ in range(n):
        y  = random.randint(*year)   if isinstance(year,   tuple) else year
        m  = random.randint(*month)  if isinstance(month,  tuple) else month
        d  = random.randint(*day)    if isinstance(day,    tuple) else day
        h  = random.randint(*hour)   if isinstance(hour,   tuple) else hour
        mi = random.randint(*minute) if isinstance(minute, tuple) else minute
        s  = random.randint(*second) if isinstance(second, tuple) else second
        d  = min(d, calendar.monthrange(y, m)[1])   # clamp to valid month-end
        rows['ts'].append(f'{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02}')
        rows['value'].append(random.randint(0, 100))
        rows['category'].append(random.choice(['A', 'B', 'C']))
        rows['numeric'].append(round(random.uniform(0.0, 10.0), 3))
    return pl.DataFrame(rows).with_columns(pl.col('ts').str.to_datetime())


def makeDateDf(n=50, year=(2020, 2025), month=(1, 12), day=(1, 28)):
    '''Create a date-only DataFrame for timep testing.

    Returns a pl.DataFrame with columns:
      dt        (pl.Date)
      value     (pl.Int32)
      category  (pl.Utf8)
      numeric   (pl.Float64)
    '''
    rows = {'dt': [], 'value': [], 'category': [], 'numeric': []}
    for _ in range(n):
        y = random.randint(*year)  if isinstance(year,  tuple) else year
        m = random.randint(*month) if isinstance(month, tuple) else month
        d = random.randint(*day)   if isinstance(day,   tuple) else day
        d = min(d, calendar.monthrange(y, m)[1])
        rows['dt'].append(f'{y:04}-{m:02}-{d:02}')
        rows['value'].append(random.randint(0, 100))
        rows['category'].append(random.choice(['A', 'B', 'C']))
        rows['numeric'].append(round(random.uniform(0.0, 10.0), 3))
    return pl.DataFrame(rows).with_columns(pl.col('dt').cast(pl.Date))
