"""Contract tests for the I3 REST and WebSocket protocols.

``test_rest_schema.py`` validates the FastAPI ``/api/openapi.json`` schema
via :mod:`schemathesis` and runs a stateful fuzz over every documented
endpoint.  ``test_websocket_protocol.py`` pins the exact JSON envelope
shape sent from the server to connected dashboards, guarding against
accidental breaking changes to the demo SPA in ``web/js``.
"""
