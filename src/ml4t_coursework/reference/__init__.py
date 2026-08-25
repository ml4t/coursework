"""One reference implementation per component, each declaring its own contract.

The reference is two things at once: the implementation `load_component` falls back to when a
student's version is absent or non-conformant, so a wrong Part 4 does not block Part 7, and the
object each contract test was authored against, so the test is guaranteed passable.
"""
