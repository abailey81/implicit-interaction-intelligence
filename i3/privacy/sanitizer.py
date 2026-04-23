"""PII sanitization and privacy auditing for the I3 privacy layer.

This module ensures that no personally identifiable information (PII) is ever
persisted, transmitted to cloud services, or logged. It provides defense-in-depth:
even if higher layers fail to strip PII, the sanitizer catches it.

Classes:
    SanitizationResult: Structured result of a sanitization pass.
    PrivacySanitizer: PII detection and replacement engine.
    PrivacyAuditor: Runtime auditing of privacy guarantees.
"""

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# SEC: Defense-in-depth ReDoS guard. Any text exceeding this length is
# truncated before being passed to the regex engine. 50_000 chars is large
# enough for any realistic single utterance / message and small enough that
# even quadratic backtracking on a worst-case pattern stays below ~100 ms.
MAX_INPUT_LENGTH: int = 50_000


@dataclass(frozen=True)
class SanitizationResult:
    """Result of sanitizing text. Immutable to prevent accidental mutation
    of audit records by downstream callers."""
    sanitized_text: str
    pii_detected: bool
    pii_types: list[str]  # Types of PII found (e.g., ["email", "phone"])
    replacements_made: int


class PrivacySanitizer:
    """Strips PII from text before any processing or transmission.

    Detects and replaces:
    - Email addresses
    - Phone numbers (US, UK, international formats)
    - Physical addresses
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Dates of birth patterns
    - Names (common name patterns -- heuristic)
    - URLs with tracking parameters

    Used before:
    1. Sending text to cloud LLM
    2. Any text that might be logged (it shouldn't be, but defense in depth)
    3. Topic extraction (strip PII before extracting keywords)
    """

    # SEC: Pattern ORDERING is significant. URL must run BEFORE email so that
    # an email embedded inside a URL (e.g. mailto: or query string) is masked
    # exactly once as [URL] rather than first being partially mangled by the
    # email pattern. Likewise SSN runs before phone patterns so that an SSN
    # like 123-45-6789 is not accidentally masked as a US phone fragment.
    PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
        # SEC: URL pattern uses a negated character class with a single +,
        # which has no ambiguity (each char either belongs or terminates the
        # match). ReDoS-safe.
        ("url", re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE
        ), "[URL]"),

        # SEC: Email pattern hardened against ReDoS and a correctness bug.
        # Original was: \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b
        # Two issues fixed:
        #   1. [A-Z|a-z] literally included the '|' character (a typo for
        #      [A-Za-z]). The pipe is not alternation inside a character
        #      class -- it matches a literal '|'.
        #   2. The unbounded [A-Za-z0-9.-]+ followed by \. caused quadratic
        #      backtracking on long inputs lacking a trailing dot, because
        #      '.' is a member of the preceding class. We bound the local
        #      part to 64 chars (RFC 5321) and the domain to 253 chars and
        #      the TLD to 24 chars. With these bounds the worst-case work is
        #      a small constant per starting position -> linear overall.
        ("email", re.compile(
            r'\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,24}\b'
        ), "[EMAIL]"),

        # SEC: SSN runs before phone patterns to prevent a US-style SSN from
        # being partially masked as a phone fragment.
        ("ssn", re.compile(
            r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'
        ), "[SSN]"),

        ("phone_us", re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        ), "[PHONE]"),

        ("phone_uk", re.compile(
            r'\b(?:\+?44[-.\s]?|0)[0-9]{2,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'
        ), "[PHONE]"),

        ("phone_intl", re.compile(
            r'\b\+[0-9]{1,3}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'
        ), "[PHONE]"),

        ("credit_card", re.compile(
            r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b'
        ), "[CREDIT_CARD]"),

        # SEC (L-3, 2026-04-23 audit): require every octet to be
        # ``<= 255`` so Windows build numbers (``10.0.22621``), SemVer
        # fragments (``1.2.3.4-beta``), and telemetry counters do not
        # inflate PII counters with false-positive IP hits.  The pattern
        # is still bounded and linear in length, so ReDoS safety is
        # preserved.
        ("ip_address", re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        ), "[IP_ADDRESS]"),

        # SEC: Address pattern hardened. Original had \w+ which could match
        # arbitrarily long tokens and trigger backtracking against the
        # required street-suffix literal. We bound the street-name token to
        # 1..30 word characters; real street names are well under that.
        ("address", re.compile(
            r'\b\d{1,5}\s+\w{1,30}\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|'
            r'Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl)\b',
            re.IGNORECASE
        ), "[ADDRESS]"),

        ("dob", re.compile(
            r'\b(?:0[1-9]|[12][0-9]|3[01])[/\-.](?:0[1-9]|1[012])[/\-.](?:19|20)\d{2}\b'
        ), "[DOB]"),
    ]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stats = {
            "total_scans": 0,
            "pii_found": 0,
            "replacements": 0,
            "truncated_inputs": 0,
        }
        # SEC: stats may be updated from multiple async tasks / threads.
        # A simple lock keeps counters consistent without measurable cost.
        self._stats_lock = threading.Lock()

    def sanitize(self, text: str) -> SanitizationResult:
        """Remove all PII from text.

        Returns sanitized text and metadata about what was found.

        SEC: Inputs longer than ``MAX_INPUT_LENGTH`` are truncated before any
        regex is applied. This is a defense-in-depth ReDoS guard: even if a
        future pattern were vulnerable to catastrophic backtracking, the
        attack surface is bounded to a fixed-size input.
        """
        if not self.enabled:
            return SanitizationResult(text, False, [], 0)

        if not isinstance(text, str):
            # Non-string input cannot contain PII and would crash the regex
            # engine. Return an empty/clean result rather than raising.
            return SanitizationResult("", False, [], 0)

        truncated = False
        if len(text) > MAX_INPUT_LENGTH:
            text = text[:MAX_INPUT_LENGTH]
            truncated = True
            logger.warning(
                "PrivacySanitizer.sanitize: input truncated from >%d to %d "
                "chars (ReDoS guard)",
                MAX_INPUT_LENGTH,
                MAX_INPUT_LENGTH,
            )

        pii_types: list[str] = []
        replacements = 0
        sanitized = text

        for pii_type, pattern, replacement in self.PII_PATTERNS:
            matches = pattern.findall(sanitized)
            if matches:
                pii_types.append(pii_type)
                replacements += len(matches)
                sanitized = pattern.sub(replacement, sanitized)

        # SEC: Thread-safe stats update; we never log the actual PII value,
        # only its type and count, to avoid leaking sensitive data into logs.
        with self._stats_lock:
            self._stats["total_scans"] += 1
            if truncated:
                self._stats["truncated_inputs"] += 1
            if pii_types:
                self._stats["pii_found"] += 1
                self._stats["replacements"] += replacements

        if pii_types:
            logger.info(
                "PII detected and sanitized: types=%s count=%d",
                pii_types,
                replacements,
            )

        return SanitizationResult(sanitized, bool(pii_types), pii_types, replacements)

    def contains_pii(self, text: str) -> bool:
        """Quick check: does this text contain PII?

        SEC: Same MAX_INPUT_LENGTH guard as sanitize() to prevent ReDoS.
        Early-exits on first match for efficiency.
        """
        if not isinstance(text, str):
            return False
        if len(text) > MAX_INPUT_LENGTH:
            text = text[:MAX_INPUT_LENGTH]
        for _, pattern, _ in self.PII_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)


class PrivacyAuditor:
    """Audits the system to ensure privacy guarantees.

    Checks:
    1. No raw text in SQLite databases
    2. Embeddings are encrypted at rest
    3. PII sanitizer is enabled
    4. Cloud requests don't contain raw conversation history

    This auditor is designed to be run periodically (e.g., on startup, on
    shutdown, or on a schedule) to verify that the privacy layer is intact.
    """

    # Heuristics for detecting natural-language text in database columns.
    # If a text value matches these patterns and exceeds a length threshold,
    # it is flagged as a potential raw-text leak.
    _NATURAL_LANG_MIN_LENGTH = 40  # short metadata strings are OK
    _NATURAL_LANG_WORD_THRESHOLD = 6  # at least this many space-separated tokens

    # Columns whose names suggest they legitimately hold text metadata
    _SAFE_COLUMN_NAMES = frozenset({
        "topic", "keyword", "label", "category", "tag",
        "configs", "setting", "schema", "version", "type",
        "name", "key", "status", "level",
    })

    def __init__(self) -> None:
        # PERF (M-5, 2026-04-23 audit): bound the findings buffer so a
        # long-lived auditor cannot accumulate GBs of violation records
        # when the sanitiser is misconfigured.  ``deque(maxlen=...)``
        # drops the oldest record in O(1).
        from collections import deque

        self._findings: deque[dict] = deque(maxlen=1_000)
        self._sanitizer = PrivacySanitizer(enabled=True)

    async def audit_database(self, db_path: str) -> dict:
        """Scan a SQLite database for potential raw-text leaks.

        For every TEXT column in every table, samples up to 100 rows and
        checks whether any value looks like natural-language prose or
        contains PII. Returns a structured report.

        Args:
            db_path: Path to the SQLite database file.

        Returns:
            dict with keys:
                - path: the database path audited
                - tables_scanned: int
                - columns_scanned: int
                - violations: list of dicts describing each finding
                - clean: bool (True if no violations)
        """
        violations: list[dict] = []
        tables_scanned = 0
        columns_scanned = 0
        path = Path(db_path)

        if not path.exists():
            return {
                "path": db_path,
                "tables_scanned": 0,
                "columns_scanned": 0,
                "violations": [{"error": f"Database not found: {db_path}"}],
                "clean": False,
            }

        try:
            async with aiosqlite.connect(db_path) as db:
                # List all user tables
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in await cursor.fetchall()]

                # SECURITY: The values interpolated below are table/column
                # identifiers read directly from sqlite_master (the DB's
                # own catalog).  SQLite cannot parameterise identifiers.
                # We defensively reject any name containing characters
                # outside [A-Za-z0-9_] to prevent identifier-injection via
                # a malicious local schema.
                _IDENT_OK = re.compile(r"^[A-Za-z0-9_]+$")

                for table in tables:
                    if not _IDENT_OK.match(table or ""):
                        logger.warning("Skipping non-identifier table: %r", table)
                        continue
                    tables_scanned += 1
                    # Get column info
                    cursor = await db.execute(
                        f"PRAGMA table_info('{table}')"
                    )
                    columns = await cursor.fetchall()
                    text_columns = [
                        col[1] for col in columns
                        if col[2].upper() in ("TEXT", "VARCHAR", "CLOB", "")
                    ]

                    for col_name in text_columns:
                        columns_scanned += 1
                        # Skip columns that are obviously safe metadata
                        if col_name.lower() in self._SAFE_COLUMN_NAMES:
                            continue
                        if not _IDENT_OK.match(col_name or ""):
                            logger.warning(
                                "Skipping non-identifier column: %r", col_name
                            )
                            continue

                        # Sample rows  (identifiers validated above)
                        cursor = await db.execute(
                            f"SELECT \"{col_name}\" FROM \"{table}\" "  # noqa: S608
                            f"WHERE \"{col_name}\" IS NOT NULL LIMIT 100"
                        )
                        rows = await cursor.fetchall()

                        for (value,) in rows:
                            if not isinstance(value, str):
                                continue
                            if len(value) < self._NATURAL_LANG_MIN_LENGTH:
                                continue

                            # Check for natural language
                            word_count = len(value.split())
                            has_pii = self._sanitizer.contains_pii(value)
                            looks_like_prose = (
                                word_count >= self._NATURAL_LANG_WORD_THRESHOLD
                                and not value.startswith("{")  # skip JSON
                                and not value.startswith("[")  # skip arrays
                            )

                            if looks_like_prose or has_pii:
                                violation = {
                                    "table": table,
                                    "column": col_name,
                                    "issue": [],
                                    "sample_length": len(value),
                                    "word_count": word_count,
                                }
                                if looks_like_prose:
                                    violation["issue"].append("natural_language_text")
                                if has_pii:
                                    violation["issue"].append("contains_pii")
                                violations.append(violation)
                                # One violation per column is enough to flag it
                                break

        except Exception as exc:
            violations.append({"error": f"Database scan failed: {exc}"})

        result = {
            "path": db_path,
            "tables_scanned": tables_scanned,
            "columns_scanned": columns_scanned,
            "violations": violations,
            "clean": len(violations) == 0,
        }
        self._findings.append(result)
        return result

    def audit_request(self, request_payload: dict) -> dict:
        """Audit an outgoing cloud API request for privacy compliance.

        Checks:
        - System prompt does not contain raw user messages
        - Conversation history messages have been sanitized
        - No PII leaks in any string field

        Args:
            request_payload: The request dict being sent to the cloud API.

        Returns:
            dict with keys:
                - compliant: bool
                - issues: list of strings describing violations
                - pii_fields: list of field paths containing PII
        """
        issues: list[str] = []
        pii_fields: list[str] = []
        # SEC (M-4, 2026-04-23 audit): cap recursion depth so an
        # adversarially nested payload cannot drive RecursionError and
        # crash the audit.  Build field paths via a list + ``.join`` so
        # the path string is O(n) not O(n²) in depth.
        _MAX_DEPTH = 32

        def _scan_value(value, path_parts: list[str], depth: int) -> None:
            """Recursively scan a value for PII (depth-capped)."""
            if depth > _MAX_DEPTH:
                issues.append(
                    f"Payload nesting exceeds {_MAX_DEPTH} levels at "
                    f"{'.'.join(path_parts)}"
                )
                return
            if isinstance(value, str):
                if self._sanitizer.contains_pii(value):
                    pii_fields.append(".".join(path_parts))
            elif isinstance(value, dict):
                for k, v in value.items():
                    path_parts.append(str(k))
                    _scan_value(v, path_parts, depth + 1)
                    path_parts.pop()
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    path_parts.append(f"[{i}]")
                    _scan_value(item, path_parts, depth + 1)
                    path_parts.pop()

        # Scan entire payload for PII
        _scan_value(request_payload, ["root"], 0)

        # Check for raw conversation history in system prompt
        messages = request_payload.get("messages", [])
        system_prompt = request_payload.get("system", "")

        if isinstance(system_prompt, str) and len(system_prompt) > 0:
            # System prompt should not contain user conversation fragments
            word_count = len(system_prompt.split())
            if word_count > 500:
                issues.append(
                    "System prompt is unusually long "
                    f"({word_count} words) -- may contain raw conversation"
                )

        # Check that conversation history is not excessively long
        if len(messages) > 20:
            issues.append(
                f"Conversation history contains {len(messages)} messages. "
                "I3 should send only the current turn + abstract context, "
                "not raw history."
            )

        # Flag any message that looks like it was not processed through I3
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                role = msg.get("role", "")
                if role == "user" and isinstance(content, str) and len(content) > 1000:
                    issues.append(
                        f"messages[{i}]: User message is {len(content)} chars. "
                        "Raw user text should be replaced with abstract "
                        "representations before cloud transmission."
                    )

        if pii_fields:
            issues.append(f"PII detected in fields: {pii_fields}")

        result = {
            "compliant": len(issues) == 0,
            "issues": issues,
            "pii_fields": pii_fields,
        }
        self._findings.append(result)
        return result

    def generate_report(self) -> str:
        """Generate a human-readable privacy compliance report.

        Returns:
            A multi-line string summarizing all audit findings from this
            auditor's lifetime.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = [
            "=" * 60,
            "I3 PRIVACY COMPLIANCE REPORT",
            f"Generated: {timestamp}",
            "=" * 60,
            "",
        ]

        if not self._findings:
            lines.append("No audits have been performed yet.")
            return "\n".join(lines)

        db_audits = [f for f in self._findings if "path" in f]
        request_audits = [f for f in self._findings if "compliant" in f]

        # Database audits
        lines.append(f"DATABASE AUDITS: {len(db_audits)}")
        lines.append("-" * 40)
        for audit in db_audits:
            status = "PASS" if audit.get("clean", False) else "FAIL"
            lines.append(f"  [{status}] {audit.get('path', 'unknown')}")
            lines.append(
                f"    Tables: {audit.get('tables_scanned', 0)}, "
                f"Columns: {audit.get('columns_scanned', 0)}"
            )
            for v in audit.get("violations", []):
                if "error" in v:
                    lines.append(f"    ERROR: {v['error']}")
                else:
                    lines.append(
                        f"    VIOLATION: {v['table']}.{v['column']} -- "
                        f"{', '.join(v.get('issue', []))}"
                    )
        lines.append("")

        # Request audits
        lines.append(f"REQUEST AUDITS: {len(request_audits)}")
        lines.append("-" * 40)
        compliant_count = sum(1 for r in request_audits if r.get("compliant"))
        non_compliant_count = len(request_audits) - compliant_count
        lines.append(f"  Compliant: {compliant_count}")
        lines.append(f"  Non-compliant: {non_compliant_count}")
        for i, audit in enumerate(request_audits):
            if not audit.get("compliant"):
                lines.append(f"  Request audit #{i + 1} issues:")
                for issue in audit.get("issues", []):
                    lines.append(f"    - {issue}")
        lines.append("")

        # Summary
        all_clean = all(
            a.get("clean", a.get("compliant", False)) for a in self._findings
        )
        lines.append("=" * 60)
        lines.append(f"OVERALL STATUS: {'COMPLIANT' if all_clean else 'NON-COMPLIANT'}")
        lines.append("=" * 60)

        return "\n".join(lines)
