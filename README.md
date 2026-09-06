# ml4t-coursework

**Support package for the ML for Trading courses.** It is the plumbing their notebooks run on: the
student's project folder, the conformance checks that run when they save a piece of the pipeline
they have written, the results log the closing unit reads, and the report they submit. It is named
for what it holds rather than for a course, because more than one course installs it.

It is published here so that the first cell of a notebook on a free Colab runtime can be

```python
!pip install -q "ml4t-coursework[data]"
```

with no account, no credential and no repository to clone. That is the whole reason it is on PyPI.

It also owns the course price panel: the symbol list, the window, the one-time fetch and a one-way
fingerprint to check a download against. [The dataset note](DATASET.md) explains what that panel is
and why a fresh download differs from the slides in the fourth decimal place. More than one course
reads it, which is why the machinery is here rather than inside either of them.

**This package is not the course.** The video, the reading, the exercises, the checks for
understanding and the market data all live elsewhere. What is here is useful mainly to someone
enrolled; it is installable by anyone, and it will not teach you anything on its own.

## What it does

A notebook opens by saying which course it belongs to. That is what decides the project folder on
the student's Drive and which stage names their results rows may carry, so two courses on one Drive
never write over each other:

```python
import ml4t_coursework as mlc
mlc.use("foundations")

from ml4t_coursework import save_component, load_component, append_result, report
```

- **`save_component(name, obj)`** runs the component's contract test, prints what passed and what
  did not, stamps the verdict, and writes the student's own source to their project folder.
- **`load_component(name)`** returns their implementation when it is conformant and the shipped
  reference when it is not, and always says which. A wrong Part 4 does not block Part 7.
- **`append_result(row)`** appends one row per pipeline run. The final unit reads three of them
  against each other and they are written weeks apart, so what persists is the numbers.
- **`report(answers)`** writes the submission report: every component's verdict, the results rows,
  and the written answers.

`status()` shows where every component stands; `contract(name).describe()` shows what any one of
them has to satisfy; `catalog()` lists them all.

A course adds its own components and its own stage names without a change here:

```python
from ml4t_coursework import Course, register_course, add_source

register_course(Course(key="...", title="...", folder="ml4t-...", env_var="ML4T_..._HOME",
                       stages=("...",), terminal_stages=("...",)))
add_source("your_package.components")   # every module in it registers one contract
```

## How the checks work, and what they deliberately do not do

Each component carries a contract asserting **properties, never equality with a reference output**.
Two correct fold splitters legitimately differ, and an equality check would fail correct work and
teach the student to copy. Every contract runs the same four gating checks - interface, a leakage
probe, determinism, and the component's own domain invariants - plus a reference delta that is
reported and never enforced.

A component is validated on **its source re-executed in an empty namespace**, not on the object in
memory, because what a later notebook loads on a cold session is the file. Something that only
works because of another cell in the same session fails at save time, with a message naming the
missing symbol.

Reference implementations ship with the package. That is by design rather than an oversight:
the fallback is what stops one wrong component from blocking the rest of the course, so there is
nothing here that could be leaked.

## Data

The course's price panel is not distributed with this package and is not ours to redistribute. What
ships is everything else - the symbol list, the window, the fetch, and a one-way fingerprint to
check a download against:

```python
from ml4t_coursework import data
data.download()     # once, into the project folder
data.load()         # what every notebook reads
```

## Licence

MIT, in line with the other `ml4t-*` libraries. See `LICENSE`.
