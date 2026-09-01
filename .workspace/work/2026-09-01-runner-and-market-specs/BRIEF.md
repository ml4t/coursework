# The runner, the long panel, and the tier A market specs

Decided 2026-09-01. Authority: `~/ml4t/courses/strategy/learning_architecture.md` § 4h.

Both courses are blocked on this. R2P's own syllabus lists the supplied pipeline as not existing
and on the critical path for week 1, the entry review, Module 0 and the capstone baseline.

## What it is

**The pipeline is the twenty components wired by a runner.** One function, stage order, short
enough to read in one sitting. No facade: anything between the student and the components fails
the removable-abstraction rule (§ 4g) the first time a student wants a stage it does not expose,
and it hides the exact code R2P's exercises ask them to change.

**A market is a spec**, not a pipeline: the configuration plus a data adapter that produces the
panel. Nine markets is one implementation and nine specs. The spec's field vocabulary mirrors
`~/ml4t/public/case_studies/<market>/config/setup.yaml`, so a student moving to a case-study
notebook meets fields they have already set. The implementations stay independent - the book repo
must be able to change without the course changing (§ 4d).

## Build order

1. **`data_panel` goes from wide to long.** MultiIndex `(date, asset)`, a price column, optional
   supplied feature columns. Wide is one `pivot` away, so nothing downstream loses anything.
   Seven components take a panel and change with it: `availability_lag`, `quality_gates`,
   `universe`, `baseline_strategy`, `labeler`, `features`, `exit_rule`.

   **This is not needed for tier A** - all five are price panels a wide frame handles. It is
   needed for tier B, and it is being done first because Foundations has eight of thirty-seven
   unit folders authored and only two of them (`02.5-the-baseline`, `08.4-position-controls-and-
   exits`) consume a panel. Deferring it means redoing roughly ten units instead of two.

   Coordinate with `foundations-dev`: `components.md` states the interface, and the two authored
   units carry it in their notebooks.

2. **The runner.** `run(spec) -> results row`, composing the components in stage order and
   appending through `append_result`. It is what Foundations' five assembly units (2.6, 6.2, 6.4,
   7.4, 8.5) and R2P's Module 0 notebook both execute.

3. **Tier A market specs**, in order of distance from `etfs`: `fx_pairs` (daily, 20, nearly a
   config alone), `crypto_perps_funding` (8h index, 19, no exchange calendar), `cme_futures`
   (daily, 30, plus a roll rule and a billed fetch), `us_equities_panel` (daily, ~3,200, same
   shape and roughly thirty times the compute).

   `etfs` is on the critical path; the other four are not and must not block it.

**Every spec carries its download size and its expected end-to-end runtime.** Those are what
actually gate a student, and a config file does not otherwise show that `us_equities_panel` is
thirty times `etfs`.

## Not in scope

Tier B (`us_firm_characteristics`, `sp500_equity_option_analytics`, `nasdaq100_microstructure`)
is a second wave behind step 1. `sp500_options` is not a student target at all: its tradeable is
an option position with an expiry and a daily delta hedge, so the signal-to-position-to-cost
chain differs rather than the data. It stays instructor-shown.
