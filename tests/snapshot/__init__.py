"""Syrupy snapshot tests for I3.

Each test serialises a deterministic output (AdaptationVector, routing
decision, state dict) and compares it against a committed snapshot in
``__snapshots__/``.  A diff against the snapshot fails the test,
prompting a deliberate review + ``pytest --snapshot-update`` run.
"""
