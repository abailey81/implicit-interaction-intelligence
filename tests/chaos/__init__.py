"""Chaos-engineering tests for I3 pipeline resilience.

Each test simulates a specific failure mode (cloud LLM timeout, empty
sanitiser output, encoder exception) and asserts that the pipeline
degrades gracefully — never None, never an uncaught exception, always
returns a serialisable ``PipelineOutput`` or raises a well-typed error.
"""
