"""Hypothesis-based property tests for the I3 codebase.

Each test in this package declares an invariant that must hold for every
input drawn from a strategy.  Tests use ``@given`` / ``@settings`` /
``@example`` decorators from the Hypothesis library and rely on the
reproducibility hooks configured in :mod:`tests.property.conftest`.
"""
