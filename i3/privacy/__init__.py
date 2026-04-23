"""Privacy Layer for Implicit Interaction Intelligence (I3).

Core architectural guarantee: raw user text is NEVER persisted to disk.
Only abstract representations (embeddings, scalar metrics, keywords) are stored.
This module provides PII sanitization and encryption at rest.

Exports:
    PrivacySanitizer  - PII detection and removal before any processing
    ModelEncryptor    - Fernet symmetric encryption for user models at rest
    PrivacyAuditor   - Runtime auditing of privacy guarantees
"""

from i3.privacy.encryption import ModelEncryptor
from i3.privacy.sanitizer import PrivacyAuditor, PrivacySanitizer, SanitizationResult

__all__ = ["ModelEncryptor", "PrivacyAuditor", "PrivacySanitizer", "SanitizationResult"]
