"""Load / DoS-resistance tests for the I3 server.

These tests exercise the WebSocket and REST endpoints under
high-concurrency / high-volume scenarios to verify the configured rate
limits, message-size caps, and connection limits all engage as expected.
They are marked ``slow`` and ``load`` so they can be excluded from the
hot-path CI job.
"""
