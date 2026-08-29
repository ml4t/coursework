# The course price panel

`etf_close.parquet`: split- and dividend-adjusted daily closing prices for 100 ETFs, 5,031
sessions from 2006-01-03 to 2025-12-31, a wide frame indexed by date with one column per symbol.
More than one ML4T course reads it, which is why the machinery that fetches and checks it lives
here rather than inside either of them.

**The file is not in any repository, and you fetch it once.** The price history is licensed and we
are not permitted to redistribute it. Everything else - the symbol list, the window, the code that
fetches it, and a fingerprint you can check your copy against - is in this package.

In a notebook:

```python
from ml4t_coursework import data

data.download()
```

It writes into your project folder and needs no path. From a shell, in a checkout with no Colab
around it:

```bash
uv run --with "ml4t-coursework[data]" python -m ml4t_coursework.data --out data/etf_close.parquet
```

Either way it takes a minute or two, then compares your copy against the packaged fingerprint and
tells you whether it is sound. Do this once. No unit notebook downloads anything, and none of them
will run until the file is there.

## Will your numbers match ours?

Close enough that it does not matter, and here is the measurement rather than the reassurance.

Adjusted prices are revised. Every dividend and every split restates the entire history behind it,
so a series downloaded next year is not the same series of numbers as one downloaded today. That
sounds worse than it is: the revision is a *constant factor* applied to the whole series, and a
constant factor cancels out of every return. The courses compute on returns and on cross-sectional
ranks, so they do not see the factor at all.

Checked on 2026-08-17, a fresh download of all 100 symbols matched the copy the slides were
computed on, session for session. Fifty-eight symbols were identical; the rest differed by a single
scale factor, the largest of them 1.7%. The largest disagreement in any daily return anywhere in
the panel was 0.000003.

So expect your results to match the slides to three or four decimal places, and expect the fourth
to move as time passes and revisions accumulate. If a unit reports an information coefficient of
0.0164 and you get 0.0161, nothing has gone wrong and there is nothing to debug. What the course is
teaching you to do is run the process and read the result correctly; a difference in the fourth
decimal is not a difference in the answer.

What would be worth investigating is a symbol missing entirely, a session count off by more than a
few, or a volatility more than 1% away from the fingerprint. Those are delistings, ticker reuse and
bad bars rather than revisions, and `data.check()` flags them by name.

## Why there is no "exact reconstruction" file

We considered shipping the adjustment factors so you could transform a fresh download back into the
exact series behind the slides. Once the measurement above was in hand it stopped being worth
doing: the factor is one number per symbol, it cancels in returns, and applying it would change no
result you compute. A mechanism that reproduces the fourth decimal place of a number whose third
decimal is already immaterial is complexity without a payoff.

The fingerprint is deliberately one-way. It carries per-symbol session counts, dates, annualized
volatility and average daily move - enough to tell you your download is sound, not enough to
rebuild prices we are not allowed to hand you.
