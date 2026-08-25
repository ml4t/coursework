# ml4t-foundations

The plumbing every notebook in **ML4T: Foundations** runs on: your project folder, the
conformance checks that run when you save a component, the results log the final unit reads, and
the report you submit.

```python
!pip install -q ml4t-foundations
from ml4t_foundations import save_component, load_component, append_result, report
```

You do not need a repository, a git checkout or a login. The setup notebook creates the project
folder once; every later notebook mounts it.

## What each call does

- `save_component(name, obj)` runs the component's contract test, tells you what passed and what
  did not, stamps the verdict, and writes it to your project folder.
- `load_component(name)` returns your implementation when it is conformant and the shipped
  reference when it is not, and always says which. A wrong Part 4 does not block your Part 7.
- `append_result(row)` appends one row per pipeline run. The final unit reads three of them
  against each other, and they are written weeks apart, so what persists is their numbers.
- `report()` writes the submission report: every component's verdict, the results rows, and your
  two written answers.

`components.contracts()` lists every component the course asks for, with what its checks assert.
