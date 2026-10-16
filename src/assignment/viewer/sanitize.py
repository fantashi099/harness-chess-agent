"""Conservative public-export redaction, including JSON embedded in transcripts."""

import json
import re

REDACTED = "[redacted]"
PRIVATE_KEY = re.compile(
    r"(?:api[_-]?key|secret|password|passwd|authorization|credential|access[_-]?token|"
    r"refresh[_-]?token|private[_-]?key|cookie|email|username|hostname|base_url|"
    r"endpoint|system_fingerprint|organization|project_id|sandbox_id|account_id|"
    r"user_id|ip_address|device_id|request_headers)", re.I
)
PATTERNS = [
    re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.S),
    re.compile(r"https?://[^\s<>\"']+", re.I),
    re.compile(r"(?:Bearer\s+)[A-Za-z0-9._~+/=-]+", re.I),
    re.compile(r"\b(?:sk-|gh[pousr]_|github_pat_|AKIA)[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?<!\w)(?:/(?:home|Users|root|tmp|workspace|workspaces|mnt|private|var|opt|srv)/[^\s\"'<>;,)}\]]+|[A-Za-z]:\\[^\s\"'<>]+)"),
    re.compile(r"\b(?:api[_-]?key|password|secret|access[_-]?token|authorization)\s*[:=]\s*[^\s,;\"'}]+", re.I),
    re.compile(r"\b(?:api[_-]?key|password|secret|access[_-]?token|authorization)[\"']?\s*[:=]\s*([\"'])(?:(?!\1).)*?\1", re.I),
]


def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(str(k)): REDACTED if PRIVATE_KEY.search(str(k)) else sanitize(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if not isinstance(value, str):
        return value
    # Prompt contents frequently contain serialized context, not just plain text.
    try:
        embedded = json.loads(value)
    except (ValueError, TypeError):
        embedded = None
    if isinstance(embedded, (dict, list)):
        return json.dumps(sanitize(embedded), ensure_ascii=False)
    for pattern in PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value
