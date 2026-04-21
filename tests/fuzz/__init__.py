"""Atheris-based fuzz harnesses for the I3 codebase.

These are not regular pytest tests.  They are standalone scripts that
atheris runs in a fuzzing loop.  Each harness asserts that a specific
function never raises on arbitrary byte input.  See ``README.md`` in this
directory for run instructions.
"""
