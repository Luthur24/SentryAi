"""
Fluid Intelligence backend — updated version.
Run:  gunicorn main:app   (or `python main.py` locally)

Changes in this update:
- MAX_INPUT_TOKENS raised to 12000
- Real SSE streaming from providers (not single-chunk shim)
- Reasoning effort param support (Groq)
- Response format validation with retry
- Structured request logging
- Per-key cooldowns (not global provider cooldowns)
- Cloudinary URL support for images/videos
- File text extraction (PDF via PyMuPDF/pdfminer/pdftotext, DOCX, TXT/MD/CSV)
"""

import os
import json
import secrets
import re
import logging
import datetime as dt
from datetime import timezone
from contextlib import contextmanager

import bcrypt
import jwt
import requests
from flask import Flask, request, jsonify, Response, g, stream_with_context
from flask_cors import CORS
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# =========================================================================
# LOGGING
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================================
# CONFIG
# =========================================================================

APP_VERSION = os.environ.get("APP_VERSION", "update-2026-09-15")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://avnadmin:AVNS_q_zR9FvhvJHJGbiL0zp@pg-235735e2-ub7499710-7253.j.aivencloud.com:11602/defaultdb?sslmode=require",
)

JWT_SECRET = "21c8524d2320d22706fb366fc6ca9037050cdfda26c104ac0ecfa0215bebda7c"
SESSION_HOURS = 24 * 7
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

DAILY_REQUEST_CAP = int(os.environ.get("DAILY_REQUEST_CAP", "3000"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "12000"))  # Raised from 8000
MAX_TOOL_ITERATIONS = 5

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

# Cloudinary config (for image/video uploads)
# NOTE: these are server-side only. Never expose CLOUDINARY_API_SECRET to the
# frontend/browser — anyone who saw it could fully control this Cloudinary account.
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "ddusfl7pi")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", "599965682593626")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "pUcb90_1jtv-rDlHXRRsfDcBK5k")
CLOUDINARY_UPLOAD_PRESET = os.environ.get("CLOUDINARY_UPLOAD_PRESET", "")  # not needed for signed uploads

# =========================================================================
# DATABASE
# =========================================================================

_pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL, sslmode="require")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    key_prefix  TEXT NOT NULL,
    key_hash    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

CREATE TABLE IF NOT EXISTS usage_counters (
    api_key_id    INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    usage_date    DATE NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (api_key_id, usage_date)
);

CREATE TABLE IF NOT EXISTS backend_status (
    backend_name       TEXT PRIMARY KEY,
    unavailable_until  TIMESTAMPTZ,
    last_error         TEXT
);

-- Per-key cooldown tracking
CREATE TABLE IF NOT EXISTS key_status (
    provider     TEXT NOT NULL,
    key_hash     TEXT NOT NULL,
    unavailable_until TIMESTAMPTZ,
    last_error   TEXT,
    PRIMARY KEY (provider, key_hash)
);
"""

MIGRATION_SQL = """
ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_in INTEGER NOT NULL DEFAULT 0;
ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_out INTEGER NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS key_status (
    provider     TEXT NOT NULL,
    key_hash     TEXT NOT NULL,
    unavailable_until TIMESTAMPTZ,
    last_error   TEXT,
    PRIMARY KEY (provider, key_hash)
);
"""


@contextmanager
def get_cursor(commit=False):
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def init_schema():
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(MIGRATION_SQL)
        conn.commit()
    finally:
        _pool.putconn(conn)


def utcnow():
    return dt.datetime.now(timezone.utc)


def seconds_until_midnight_utc():
    now = utcnow()
    midnight = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))


# =========================================================================
# AUTH — passwords, session tokens, API keys
# =========================================================================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def create_session_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": utcnow() + dt.timedelta(hours=SESSION_HOURS),
        "iat": utcnow(),
        "v": 1,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token.decode() if isinstance(token, bytes) else token


def verify_session_token(token: str):
    """Returns (user_id, error). error is None, 'expired', or 'invalid'."""
    if not token:
        return None, "invalid"
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return int(payload["sub"]), None
    except jwt.ExpiredSignatureError:
        return None, "expired"
    except jwt.InvalidTokenError:
        return None, "invalid"
    except Exception:
        return None, "invalid"


def bearer_token():
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header.startswith("Bearer ") else ""


def require_session():
    """Returns (user_id, http_error_tuple_or_None)."""
    token = bearer_token()
    if not token:
        return None, (jsonify(error="unauthorized"), 401)
    user_id, err = verify_session_token(token)
    if err == "expired":
        return None, (jsonify(error="session_expired"), 401)
    if err:
        return None, (jsonify(error="invalid_token"), 401)
    return user_id, None


def generate_api_key():
    raw = secrets.token_hex(24)
    full_key = f"sk-sentry-{raw}"
    prefix = f"sk-sentry-{raw[:8]}..."
    key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode()
    return full_key, prefix, key_hash


def create_api_key_for_user(user_id: int, name: str):
    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM api_keys WHERE user_id = %s AND revoked_at IS NULL", (user_id,)
        )
        if cur.fetchone():
            raise ValueError("Account already has an active API key. Revoke it first.")
    full_key, prefix, key_hash = generate_api_key()
    with get_cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO api_keys (user_id, name, key_prefix, key_hash)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (user_id, name, prefix, key_hash),
        )
        key_id = cur.fetchone()["id"]
    return {"id": key_id, "key": full_key, "prefix": prefix, "name": name}


def authenticate_api_key(token: str):
    if not token or not token.startswith("sk-sentry-"):
        return None
    prefix_guess = token[:18] + "..."
    with get_cursor() as cur:
        cur.execute(
            """SELECT id, user_id, key_hash FROM api_keys
               WHERE key_prefix = %s AND revoked_at IS NULL""",
            (prefix_guess,),
        )
        row = cur.fetchone()
    if not row:
        return None
    if not bcrypt.checkpw(token.encode(), row["key_hash"].encode()):
        return None
    return {"api_key_id": row["id"], "user_id": row["user_id"]}


def get_users_active_key_id(user_id: int):
    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM api_keys WHERE user_id = %s AND revoked_at IS NULL", (user_id,)
        )
        row = cur.fetchone()
    return row["id"] if row else None


# =========================================================================
# USAGE / RATE LIMITING
# =========================================================================

def check_and_increment_rate_limit(api_key_id: int, tokens_in: int = 0) -> int:
    today = dt.date.today()
    with get_cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO usage_counters (api_key_id, usage_date, request_count, tokens_in)
               VALUES (%s, %s, 1, %s)
               ON CONFLICT (api_key_id, usage_date)
               DO UPDATE SET request_count = usage_counters.request_count + 1,
                             tokens_in = usage_counters.tokens_in + EXCLUDED.tokens_in
               RETURNING request_count""",
            (api_key_id, today, tokens_in),
        )
        return cur.fetchone()["request_count"]


def record_tokens_out(api_key_id: int, tokens_out: int):
    if tokens_out <= 0:
        return
    with get_cursor(commit=True) as cur:
        cur.execute(
            """UPDATE usage_counters SET tokens_out = usage_counters.tokens_out + %s
               WHERE api_key_id = %s AND usage_date = %s""",
            (tokens_out, api_key_id, dt.date.today()),
        )


def set_rate_limit_headers(count: int):
    g.rl_limit = DAILY_REQUEST_CAP
    g.rl_remaining = max(0, DAILY_REQUEST_CAP - count)
    g.rl_reset = seconds_until_midnight_utc()


# =========================================================================
# BACKEND COOLDOWNS — PER KEY
# =========================================================================

def _hash_key(key: str) -> str:
    """Create a short hash for tracking without exposing the key."""
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def mark_key_down(provider: str, key: str, error, cooldown_seconds: int = 30):
    """Mark a specific key as cooling down, not the whole provider."""
    key_hash = _hash_key(key)
    until = utcnow() + dt.timedelta(seconds=cooldown_seconds)
    with get_cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO key_status (provider, key_hash, unavailable_until, last_error)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (provider, key_hash) DO UPDATE
               SET unavailable_until = EXCLUDED.unavailable_until,
                   last_error = EXCLUDED.last_error""",
            (provider, key_hash, until, str(error)[:500]),
        )


def is_key_available(provider: str, key: str) -> bool:
    """Check if a specific key is available."""
    key_hash = _hash_key(key)
    with get_cursor() as cur:
        cur.execute(
            "SELECT unavailable_until FROM key_status WHERE provider = %s AND key_hash = %s",
            (provider, key_hash)
        )
        row = cur.fetchone()
    if not row or not row["unavailable_until"]:
        return True
    return utcnow() > row["unavailable_until"]


# Keep old functions for backward compatibility but make them no-ops
def mark_backend_down(name: str, error, cooldown_seconds: int = 30):
    """Deprecated: use mark_key_down instead."""
    pass


def is_backend_available(name: str) -> bool:
    """Deprecated: use is_key_available instead."""
    return True


# =========================================================================
# MODEL PROVIDERS
# =========================================================================

# ---------------------------------------------------------------------------
# ROUTING TABLE
# Each provider declares what it can actually do on the free tier it runs on.
# The router picks from this table instead of hardcoded order lists, so a
# request always lands on a provider that can serve it — and when a provider
# is dead/cooling down, the router moves to the next *capable* one.
# ---------------------------------------------------------------------------
PROVIDERS = {
    "groq": {
        "text": True, "tools": True, "media": True, "streaming": True,
        "reasoning": True,
    },
    "mistral": {
        "text": True, "tools": True, "media": False, "streaming": True,
        "reasoning": False,
    },
    "zai": {
        "text": True, "tools": True, "media": False, "streaming": True,
        "reasoning": False,
    },
    "gemini": {
        "text": True, "tools": False, "media": True, "streaming": True,
        "reasoning": False,
    },
}

DEFAULT_ORDER = ["groq", "mistral", "zai", "gemini"]

OPENAI_STYLE = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_envs": ["GROQ_API_KEY_1", "GROQ_API_KEY_2"],
        "api_key_defaults": [
            "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",
            "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",
        ],
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        "extra_body": {},
        "supports_reasoning": True,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "api_key_envs": ["MISTRAL_API_KEY_1", "MISTRAL_API_KEY_2"],
        "api_key_defaults": [
            "uZIGi6VCzkcJ0i5X5mYYc0Nr1XWX21YR",
            "emd7ZA8yvckaHrGvZ9jsxKMA6y3wNSbn",
        ],
        "model": os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
        "extra_body": {},
        "supports_reasoning": False,
    },
    "zai": {
        "base_url": "https://api.z.ai/api/paas/v4/chat/completions",
        "api_key_envs": ["ZAI_API_KEY_1", "ZAI_API_KEY_2"],
        "api_key_defaults": [
            "ccd29e152e6941098d7e9e1fd91f24a1.thzU8g4LLNVm6qea",
            "f6bc4aac8cfe425cadde84bba181710e.wX0Wm0I8l3YDihGy",
        ],
        "model": os.environ.get("ZAI_MODEL", "glm-4.7-flash"),
        "extra_body": {"thinking": {"type": "disabled"}},
        "supports_reasoning": False,
    },
}

GEMINI_API_KEY_ENVS = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"]
GEMINI_API_KEY_DEFAULTS = [
    "AQ.Ab8RN6Kq43fqm482ux_6BD71H67SjJlBjE4qbqY94AGUW1JyLw",
    "AIzaSyBRB2XWuUT-E_X0D8F7B1YTdjrkMIRMBRY",
]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

PASSTHROUGH_PARAMS = ("temperature", "top_p", "top_k", "max_tokens", "stop", "seed",
                      "presence_penalty", "frequency_penalty", "response_format")


class ProviderError(Exception):
    pass


def has_media(messages: list) -> bool:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("image_url", "input_audio", "video_url"):
                    return True
    return False


def estimate_tokens(messages) -> int:
    return max(1, len(json.dumps(messages)) // 4)


def _forward_sampling_params(body: dict, client_body: dict):
    for p in PASSTHROUGH_PARAMS:
        if p in client_body and client_body[p] is not None:
            body[p] = client_body[p]


def _get_reasoning_params(provider: str, client_body: dict) -> dict:
    """Extract reasoning params for providers that support them."""
    reasoning = client_body.get("reasoning") or {}
    if not reasoning.get("enabled"):
        return {}

    effort = reasoning.get("effort", "medium")

    if provider == "groq":
        # Groq supports reasoning_effort
        return {"reasoning_effort": effort}

    return {}


def _error_detail(resp) -> str:
    """Pull the real error message out of a provider's response body instead
    of the generic 'NNN Client Error' text that raise_for_status() gives."""
    try:
        data = resp.json()
        msg = (
            (data.get("error") or {}).get("message")
            if isinstance(data.get("error"), dict)
            else data.get("error") or data.get("message")
        )
        if msg:
            return str(msg)[:300]
        return json.dumps(data)[:300]
    except Exception:
        return (resp.text or "")[:300]


def _call_openai_style(provider: str, messages: list, client_body: dict, tools, timeout: int, stream: bool = False):
    cfg = OPENAI_STYLE[provider]
    keys = [os.environ.get(e, d) for e, d in zip(cfg["api_key_envs"], cfg["api_key_defaults"])]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError(f"{provider}: no API keys configured")

    body = {"model": cfg["model"], "messages": messages}
    _forward_sampling_params(body, client_body or {})

    # Add reasoning params if supported
    if cfg.get("supports_reasoning"):
        body.update(_get_reasoning_params(provider, client_body or {}))

    if tools:
        body["tools"] = tools
    if cfg.get("extra_body"):
        body.update(cfg["extra_body"])

    if stream:
        body["stream"] = True

    last_error = None
    for key in keys:
        # Check per-key cooldown
        if not is_key_available(provider, key):
            logger.info(f"{provider}: key {_hash_key(key)} cooling down, skipping")
            continue

        try:
            start_time = dt.datetime.now()
            resp = requests.post(
                cfg["base_url"],
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
                stream=stream,
            )
            latency = (dt.datetime.now() - start_time).total_seconds() * 1000

            if resp.status_code == 429:
                last_error = f"{provider}: rate limited on key {_hash_key(key)} — {_error_detail(resp)}"
                mark_key_down(provider, key, last_error, cooldown_seconds=60)
                logger.warning(f"{provider}: 429 on key {_hash_key(key)}, cooling down 60s")
                continue
            if resp.status_code in (401, 403):
                # Key is invalid/revoked/restricted -- not a transient error.
                # Long cooldown so we fail fast and clearly instead of
                # hammering a dead key on every request.
                last_error = f"{provider}: key {_hash_key(key)} rejected with {resp.status_code} — {_error_detail(resp)}"
                mark_key_down(provider, key, last_error, cooldown_seconds=6 * 3600)
                logger.error(f"{provider}: {resp.status_code} on key {_hash_key(key)} -- {_error_detail(resp)}")
                continue
            if resp.status_code >= 500:
                last_error = f"{provider}: server error {resp.status_code} — {_error_detail(resp)}"
                mark_key_down(provider, key, last_error, cooldown_seconds=30)
                logger.warning(f"{provider}: {resp.status_code} on key {_hash_key(key)}")
                continue
            if resp.status_code >= 400:
                # Any other 4xx (400, 404, 422...) is almost always a bad
                # request (e.g. this model doesn't support image input) --
                # not a dead key. Surface the real reason, and use a short
                # cooldown since retrying the *same* key with a *different*
                # request could well work.
                last_error = f"{provider}: {resp.status_code} — {_error_detail(resp)}"
                mark_key_down(provider, key, last_error, cooldown_seconds=5)
                logger.error(f"{provider}: {resp.status_code} on key {_hash_key(key)} -- {_error_detail(resp)}")
                continue

            if stream:
                # Return the streaming response object
                logger.info(f"{provider}: streaming started, key {_hash_key(key)}, latency {latency:.0f}ms")
                return {
                    "stream": resp.iter_lines(),
                    "provider": provider,
                    "key_hash": _hash_key(key),
                }

            data = resp.json()
            choice = data["choices"][0]["message"]

            usage = data.get("usage", {})
            logger.info(f"{provider}: success, key {_hash_key(key)}, latency {latency:.0f}ms, tokens {usage.get('prompt_tokens', '?')}/{usage.get('completion_tokens', '?')}")

            return {
                "content": choice.get("content"),
                "tool_calls": choice.get("tool_calls"),
                "provider": provider,
                "usage": usage,
            }
        except requests.RequestException as e:
            last_error = f"{provider}: {e}"
            mark_key_down(provider, key, last_error, cooldown_seconds=30)
            logger.error(f"{provider}: error on key {_hash_key(key)}: {e}")
            continue

    raise ProviderError(last_error or f"{provider}: all keys failed")


def _fetch_media_as_base64(url: str, max_size_mb: int = 5) -> tuple:
    """Fetch a media URL and return (base64_data, mime_type)."""
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        # Check content length if available
        content_length = resp.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            raise ProviderError(f"Media too large: {int(content_length) / 1024 / 1024:.1f}MB > {max_size_mb}MB")

        # Download with size limit
        chunks = []
        total_size = 0
        for chunk in resp.iter_content(chunk_size=8192):
            chunks.append(chunk)
            total_size += len(chunk)
            if total_size > max_size_mb * 1024 * 1024:
                raise ProviderError(f"Media too large: >{max_size_mb}MB")

        data = b''.join(chunks)
        import base64
        b64 = base64.b64encode(data).decode('utf-8')

        # Determine mime type from URL or content-type header
        content_type = resp.headers.get('content-type', '').split(';')[0]
        if not content_type:
            # Guess from URL extension
            ext = url.split('.')[-1].lower().split('?')[0]
            mime_map = {
                'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
                'gif': 'image/gif', 'webp': 'image/webp', 'mp4': 'video/mp4',
                'webm': 'video/webm', 'mov': 'video/quicktime'
            }
            content_type = mime_map.get(ext, 'application/octet-stream')

        return b64, content_type
    except requests.RequestException as e:
        raise ProviderError(f"Failed to fetch media: {e}")


def _media_messages_to_data_urls(messages: list) -> list:
    """Convert remote image URLs in messages to data: URLs so a media request
    can still be served by OpenAI-style vision providers when Gemini is down
    or its keys are invalid."""
    converted = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            converted.append(m)
            continue
        blocks = []
        changed = False
        for block in content:
            b = dict(block)
            if b.get("type") == "image_url":
                url = (b.get("image_url") or {}).get("url", "")
                if url.startswith("https://"):
                    try:
                        b64, mime = _fetch_media_as_base64(url, max_size_mb=5)
                        b["image_url"] = {"url": f"data:{mime};base64,{b64}"}
                        changed = True
                    except ProviderError as e:
                        logger.warning(f"media fallback: could not fetch image ({url[:60]}...): {e}")
            blocks.append(b)
        converted.append({**m, "content": blocks} if changed else m)
    return converted


def _messages_to_gemini_contents(messages: list):
    system_texts = [
        m["content"] for m in messages
        if m["role"] == "system" and isinstance(m["content"], str)
    ]
    prefix = ("\n\n".join(system_texts) + "\n\n") if system_texts else ""

    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        content = m["content"]
        parts = []

        if isinstance(content, str):
            text = prefix + content if prefix and role == "user" else content
            parts.append({"text": text})
            prefix = ""
        elif isinstance(content, list):
            first_text_done = False
            for block in content:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if prefix and role == "user" and not first_text_done:
                        text = prefix + text
                        prefix = ""
                    first_text_done = True
                    parts.append({"text": text})
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        mime, _, b64data = url.partition(";base64,")
                        parts.append({"inline_data": {"mime_type": mime.replace("data:", ""), "data": b64data}})
                    elif url.startswith("https://"):
                        # Fetch hosted image (e.g., Cloudinary)
                        try:
                            b64, mime = _fetch_media_as_base64(url, max_size_mb=5)
                            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                        except ProviderError as e:
                            logger.warning(f"Failed to fetch image: {e}")
                            parts.append({"text": f"[Image unavailable: {str(e)}]"})
                elif block.get("type") == "video_url":
                    url = block.get("video_url", {}).get("url", "")
                    if url.startswith("https://"):
                        try:
                            b64, mime = _fetch_media_as_base64(url, max_size_mb=50)
                            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                        except ProviderError as e:
                            logger.warning(f"Failed to fetch video: {e}")
                            parts.append({"text": f"[Video unavailable: {str(e)}]"})
                elif block.get("type") == "file_url":
                    # Handle file attachments - extract text
                    url = block.get("file_url", {}).get("url", "")
                    if url.startswith("https://"):
                        try:
                            text_content = _extract_text_from_url(url)
                            if prefix and role == "user" and not first_text_done:
                                text_content = prefix + text_content
                                prefix = ""
                            first_text_done = True
                            parts.append({"text": text_content})
                        except Exception as e:
                            logger.warning(f"Failed to extract file text: {e}")
                            parts.append({"text": f"[File content unavailable: {str(e)}]"})

        contents.append({"role": role, "parts": parts})

    return contents


def _extract_text_from_url(url: str) -> str:
    """Extract text from a file URL (PDF, DOCX, TXT, etc.)."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '').lower()

        # Determine file type from URL or content-type
        ext = url.split('.')[-1].lower().split('?')[0]

        if 'pdf' in content_type or ext == 'pdf':
            return _extract_pdf_text(resp.content)
        elif 'word' in content_type or ext in ('docx', 'doc'):
            return _extract_docx_text(resp.content)
        elif 'text' in content_type or ext in ('txt', 'md', 'csv'):
            return resp.text[:100000]  # Limit to 100KB of text
        else:
            # Try to decode as text
            try:
                return resp.text[:100000]
            except:
                return "[Unsupported file type]"
    except requests.RequestException as e:
        raise ProviderError(f"Failed to fetch file: {e}")


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF content using available libraries."""
    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > 100000:  # Limit output
                text += "\n\n[content truncated]"
                break
        doc.close()
        return text
    except ImportError:
        pass

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text
        import io
        text = extract_text(io.BytesIO(content))
        return text[:100000]
    except ImportError:
        pass

    # Fall back to pdftotext command
    try:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(content)
            tmp_path = f.name

        result = subprocess.run(['pdftotext', tmp_path, '-'], capture_output=True, text=True, timeout=30)
        os.unlink(tmp_path)

        if result.returncode == 0:
            return result.stdout[:100000]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "[PDF text extraction not available. Install PyMuPDF or pdfminer.six, or ensure pdftotext is installed]"


def _extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX content."""
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text[:100000]
    except ImportError:
        return "[DOCX support not available. Install python-docx or convert to PDF/TXT]"


def _call_gemini(messages: list, timeout: int, stream: bool = False):
    keys = [os.environ.get(e, d) for e, d in zip(GEMINI_API_KEY_ENVS, GEMINI_API_KEY_DEFAULTS)]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError("gemini: no API keys configured")

    body = {"contents": _messages_to_gemini_contents(messages)}

    # Add streaming config if needed
    if stream:
        body["generationConfig"] = {"maxOutputTokens": 8192}

    last_error = None
    for key in keys:
        # Check per-key cooldown
        if not is_key_available("gemini", key):
            logger.info(f"gemini: key {_hash_key(key)} cooling down, skipping")
            continue

        try:
            start_time = dt.datetime.now()
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:{'streamGenerateContent' if stream else 'generateContent'}?key={key}"
            )

            if stream:
                url += "&alt=sse"

            resp = requests.post(url, json=body, timeout=timeout, stream=stream)
            latency = (dt.datetime.now() - start_time).total_seconds() * 1000

            if resp.status_code == 429:
                last_error = f"gemini: rate limited on key {_hash_key(key)} — {_error_detail(resp)}"
                mark_key_down("gemini", key, last_error, cooldown_seconds=60)
                logger.warning(f"gemini: 429 on key {_hash_key(key)}")
                continue
            if resp.status_code in (401, 403):
                last_error = f"gemini: key {_hash_key(key)} rejected with {resp.status_code} — {_error_detail(resp)}"
                mark_key_down("gemini", key, last_error, cooldown_seconds=6 * 3600)
                logger.error(f"gemini: {resp.status_code} on key {_hash_key(key)} -- {_error_detail(resp)}")
                continue
            if resp.status_code >= 500:
                last_error = f"gemini: server error {resp.status_code} — {_error_detail(resp)}"
                mark_key_down("gemini", key, last_error, cooldown_seconds=30)
                logger.warning(f"gemini: {resp.status_code} on key {_hash_key(key)}")
                continue
            if resp.status_code >= 400:
                last_error = f"gemini: {resp.status_code} — {_error_detail(resp)}"
                mark_key_down("gemini", key, last_error, cooldown_seconds=5)
                logger.error(f"gemini: {resp.status_code} on key {_hash_key(key)} -- {_error_detail(resp)}")
                continue

            if stream:
                logger.info(f"gemini: streaming started, key {_hash_key(key)}, latency {latency:.0f}ms")
                return {
                    "stream": resp.iter_lines(),
                    "provider": "gemini",
                    "key_hash": _hash_key(key),
                }

            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]

            usage = data.get("usageMetadata", {})
            logger.info(f"gemini: success, key {_hash_key(key)}, latency {latency:.0f}ms, tokens {usage.get('promptTokenCount', '?')}/{usage.get('candidatesTokenCount', '?')}")

            return {"content": text, "tool_calls": None, "provider": "gemini", "usage": usage}
        except requests.RequestException as e:
            last_error = f"gemini: {e}"
            mark_key_down("gemini", key, last_error, cooldown_seconds=30)
            logger.error(f"gemini: error on key {_hash_key(key)}: {e}")
            continue

    raise ProviderError(last_error or "gemini: all keys failed")


def _rank_providers(messages: list, tools, stream: bool, client_body: dict) -> list:
    """Rank all capable providers for this request.

    Scoring (lower = better):
      +0  primary path — provider natively supports everything requested
      +2  media fallback — media request served via data-URL conversion
      +3  tool fallback — tools stripped, Tavily pre-search injected instead
    """
    needs_media = has_media(messages)
    needs_tools = bool(tools)
    wants_reasoning = bool(((client_body or {}).get("reasoning") or {}).get("enabled"))

    candidates = []  # (score, position_in_DEFAULT_ORDER, name, converted_msgs, use_tools)
    for pos, name in enumerate(DEFAULT_ORDER):
        caps = PROVIDERS[name]
        if not caps["text"]:
            continue
        score = pos  # base preference from DEFAULT_ORDER

        use_tools = needs_tools
        msgs = messages

        if needs_media:
            if caps["media"]:
                # Native media provider — boost above text-only providers
                # so gemini/groq-vision are tried before mistral/zai.
                score -= 5
            elif caps["text"]:
                # Can serve media if we convert remote URLs to data URLs.
                # +10 ensures every native-media provider is tried first.
                score += 10
            else:
                continue

        if needs_tools:
            if caps["tools"]:
                pass  # native tool support
            else:
                # Can't do tools — will strip them and pre-inject Tavily results.
                # +10 ensures every tool-capable provider is tried first.
                score += 10
                use_tools = False

        if wants_reasoning and caps.get("reasoning"):
            score -= 1  # slight boost — provider handles reasoning natively

        candidates.append((score, pos, name, msgs, use_tools))

    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates


def call_model(messages: list, client_body: dict = None, tools=None, timeout: int = 30, order=None, stream: bool = False):
    """Route a request to the best provider that can actually serve it.

    Strategy per request shape:
      text only        -> groq / mistral / zai (skip gemini unless others dead)
      text + tools     -> groq / mistral / zai (gemini can't do tools)
      text + media     -> gemini first; if dead, groq with data-URL conversion
      text + tools + media -> gemini can't do tools; groq does both natively
      any + streaming  -> same routing, streaming passed through
    """
    errors = []

    # Explicit order override (env or caller) still respected
    if order is not None:
        names = order
    else:
        env_order = os.environ.get("PROVIDER_ORDER")
        if env_order:
            names = [p.strip() for p in env_order.split(",")]
        else:
            names = None  # use capability ranking

    if names:
        # Legacy explicit-order path: filter to capable providers only
        needs_media = has_media(messages)
        needs_tools = bool(tools)
        ranked = []
        for name in names:
            caps = PROVIDERS.get(name)
            if not caps or not caps["text"]:
                continue
            if needs_media and not caps["media"]:
                continue
            if needs_tools and not caps["tools"]:
                continue
            ranked.append((0, 0, name, messages, bool(tools)))
    else:
        ranked = _rank_providers(messages, tools, stream, client_body)

    # --- Pass 1: try every capable provider, native capabilities only ---
    for score, pos, name, msgs, use_tools in ranked:
        # Media fallback entries (score penalty) are tried in pass 2
        if score >= 2:
            continue
        try:
            if name == "gemini":
                return _call_gemini(msgs, timeout, stream=stream)
            return _call_openai_style(name, msgs, client_body, use_tools, timeout, stream=stream)
        except (ProviderError, requests.RequestException) as e:
            errors.append(f"{name}: {e}")
            logger.error(f"Provider {name} failed: {e}")
            continue

    # --- Pass 2: media fallback — convert remote URLs to data URLs ---
    if has_media(messages) and not stream:
        for score, pos, name, msgs, use_tools in ranked:
            if score < 2 or PROVIDERS[name]["media"]:
                continue
            try:
                converted = _media_messages_to_data_urls(messages)
                logger.warning(f"native media provider(s) unavailable; trying {name} with data-URL conversion")
                return _call_openai_style(name, converted, client_body, use_tools, timeout, stream=stream)
            except (ProviderError, requests.RequestException) as e:
                errors.append(f"{name} (media fallback): {e}")
                logger.error(f"Provider {name} media fallback failed: {e}")
                continue
            except ProviderError as e:
                errors.append(f"media conversion: {e}")

    # --- Pass 3: tool fallback — strip tools, answer without them ---
    # (only reached when every tool-capable provider is dead)
    if tools:
        for score, pos, name, msgs, use_tools in ranked:
            if use_tools:
                continue  # already tried in pass 1
            try:
                logger.warning(f"all tool-capable providers unavailable; trying {name} without tools")
                if name == "gemini":
                    return _call_gemini(messages, timeout, stream=stream)
                return _call_openai_style(name, messages, client_body, None, timeout, stream=stream)
            except (ProviderError, requests.RequestException) as e:
                errors.append(f"{name} (no-tools fallback): {e}")
                logger.error(f"Provider {name} no-tools fallback failed: {e}")
                continue

    raise ProviderError("All providers/keys failed -> " + " | ".join(errors))


# =========================================================================
# RESPONSE FORMAT VALIDATION
# =========================================================================

def validate_and_fix_json(content: str, response_format: dict, messages: list, client_body: dict, api_key_id: int) -> str:
    """Validate JSON response, retry once if invalid."""
    if not response_format:
        return content

    try:
        json.loads(content)
        return content  # Valid JSON
    except json.JSONDecodeError:
        pass

    # Retry with stricter prompt
    logger.info("JSON validation failed, retrying with temperature=0")

    retry_body = dict(client_body or {})
    retry_body["temperature"] = 0

    # Add system message instructing valid JSON
    retry_messages = list(messages)
    retry_messages.insert(0, {
        "role": "system",
        "content": "You must respond with valid JSON only. No markdown, no explanation, just pure JSON."
    })

    try:
        result = call_model(retry_messages, client_body=retry_body)
        retry_content = result["content"]
        json.loads(retry_content)  # Validate
        return retry_content
    except (json.JSONDecodeError, ProviderError) as e:
        logger.error(f"JSON retry failed: {e}")
        raise ProviderError("Could not produce valid JSON after retry")


# =========================================================================
# TOOLS — Tavily live web search
# =========================================================================

TAVILY_KEYS = [
    os.environ.get("TAVILY_API_KEY_1", "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"),
    os.environ.get("TAVILY_API_KEY_2", "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"),
]

WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the live web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}


def web_search(query: str, max_results: int = 5):
    last_error = None
    for key in TAVILY_KEYS:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": max_results},
                timeout=15,
            )
            if resp.status_code == 429:
                last_error = "rate limited"
                continue
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
                for r in data.get("results", [])
            ]
        except requests.RequestException as e:
            last_error = str(e)
            continue
    raise RuntimeError(f"Tavily search failed on all keys: {last_error}")


TOOL_IMPLEMENTATIONS = {"web_search": web_search}


def _inject_tavily_results(messages: list, query: str, max_results: int = 5) -> list:
    """Run a Tavily search and inject the results into the conversation as
    context, so the model can answer with live data even when no provider
    is available to do native tool-calling."""
    try:
        results = web_search(query, max_results=max_results)
        if not results:
            return messages
        lines = ["Live web search results (use these to answer):"]
        for r in results:
            snippet = (r.get("content") or "")[:400]
            lines.append(f'- {r.get("title")}\n  {r.get("url")}\n  {snippet}')
        context = "\n\n".join(lines)
        injected = list(messages)
        injected.insert(-1, {"role": "system", "content": context})
        return injected
    except Exception as e:
        logger.warning(f"Tavily pre-search failed: {e}")
        return messages


def run_agentic_completion(messages: list, client_body: dict = None, web_search_enabled: bool = True, stream: bool = False):
    """Model -> tool calls -> results -> model, until no more tool calls or
    MAX_TOOL_ITERATIONS is hit.

    If no provider can serve the tool-enabled request, falls back to:
      1. Tavily pre-search — search the user's question first, inject results
         as context, then call the model without tools.
      2. Plain no-tools completion if Tavily is also unavailable.
    """
    tools = [WEB_SEARCH_TOOL_SCHEMA] if web_search_enabled else None
    working = list(messages)
    degraded = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        is_final = (iteration == MAX_TOOL_ITERATIONS - 1)
        should_stream = stream and is_final and not degraded

        try:
            result = call_model(working, client_body=client_body, tools=tools, stream=should_stream)
        except ProviderError as e:
            if not tools:
                raise
            # No provider could serve the tool-enabled request.
            # Fall back to Tavily pre-search instead of returning a 502.
            logger.warning(f"Tool-enabled call failed ({e}); falling back to Tavily pre-search")
            degraded = True

            # Extract the user's last message as the search query
            last_user = ""
            for m in reversed(working):
                if m.get("role") == "user":
                    c = m.get("content")
                    last_user = c if isinstance(c, str) else ""
                    break

            if last_user:
                working = _inject_tavily_results(working, last_user)

            result = call_model(working, client_body=client_body, tools=None, stream=False)

        if should_stream and "stream" in result and not degraded:
            # Return streaming response for final iteration
            return {
                "stream": result["stream"],
                "provider": result["provider"],
                "streaming": True,
            }

        if not result.get("tool_calls"):
            out = {"content": result["content"], "provider": result["provider"], "streaming": False}
            if degraded:
                out["web_search_degraded"] = True
            return out

        working.append({"role": "assistant", "content": result.get("content") or ""})
        for call in result["tool_calls"]:
            fn_name = call["function"]["name"]
            try:
                fn_args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            impl = TOOL_IMPLEMENTATIONS.get(fn_name)
            try:
                tool_output = impl(**fn_args) if impl else {"error": f"unknown tool {fn_name}"}
            except Exception as e:
                tool_output = {"error": str(e)}
            working.append({"role": "tool", "name": fn_name, "content": json.dumps(tool_output)})

    out = {"content": "I wasn't able to finish that within the tool-call limit.", "provider": None, "streaming": False}
    if degraded:
        out["web_search_degraded"] = True
    return out


# =========================================================================
# STREAMING HELPERS
# =========================================================================

def sse_stream_generator(stream_data):
    """Generate SSE format from provider stream."""
    try:
        for line in stream_data:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if not line:
                continue
            if line.startswith('data: '):
                # Already SSE format from Gemini
                yield line + '\n\n'
            elif line.startswith('{'):
                # OpenAI-style chunk
                try:
                    chunk = json.loads(line)
                    # Normalize to OpenAI chunk format
                    if 'choices' in chunk:
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # Gemini format - convert
                        yield f"data: {json.dumps(chunk)}\n\n"
                except json.JSONDecodeError:
                    continue
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


# =========================================================================
# FLASK APP
# =========================================================================

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

    with app.app_context():
        init_schema()

    @app.after_request
    def attach_rate_limit_headers(resp):
        if hasattr(g, "rl_limit"):
            resp.headers["X-RateLimit-Limit"] = str(g.rl_limit)
            resp.headers["X-RateLimit-Remaining"] = str(g.rl_remaining)
            resp.headers["X-RateLimit-Reset"] = str(g.rl_reset)
        return resp

    # ---------- meta ----------

    @app.get("/healthz")
    def healthz():
        return jsonify(ok=True, version=APP_VERSION)

    @app.get("/version")
    def version():
        return jsonify(version=APP_VERSION, python=os.sys.version.split()[0])

    @app.get("/stats/public")
    def public_stats():
        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM users")
            users_count = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM api_keys WHERE revoked_at IS NULL")
            keys_count = cur.fetchone()["c"]
        return jsonify(users=users_count, keys=keys_count)

    @app.get("/v1/models")
    def list_models():
        resp = jsonify(data=[{"id": "sentry-1", "object": "model"}])
        return resp

    # ---------- auth ----------

    @app.post("/auth/signup")
    def signup():
        data = request.get_json(force=True, silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if not EMAIL_RE.match(email):
            return jsonify(error="enter a valid email address"), 400
        if len(password) < 8:
            return jsonify(error="password must be at least 8 characters"), 400

        try:
            with get_cursor(commit=True) as cur:
                cur.execute(
                    "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                    (email, hash_password(password)),
                )
                user_id = cur.fetchone()["id"]
        except Exception:
            return jsonify(error="email already registered"), 409

        return jsonify(token=create_session_token(user_id)), 201

    @app.post("/auth/signin")
    def signin():
        data = request.get_json(force=True, silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        with get_cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
        if not row or not check_password(password, row["password_hash"]):
            return jsonify(error="invalid credentials"), 401
        return jsonify(token=create_session_token(row["id"]))

    @app.get("/auth/session")
    def auth_session():
        user_id, err = require_session()
        if err:
            return err
        with get_cursor() as cur:
            cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
        if not row:
            return jsonify(error="invalid_token"), 401
        return jsonify(email=row["email"])

    @app.post("/console/password")
    def update_password():
        user_id, err = require_session()
        if err:
            return err
        data = request.get_json(force=True, silent=True) or {}
        password = data.get("password") or ""
        if len(password) < 8:
            return jsonify(error="password must be at least 8 characters"), 400
        with get_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (hash_password(password), user_id),
            )
        return jsonify(ok=True)

    # ---------- console: API keys ----------

    @app.post("/console/keys")
    def create_key():
        user_id, err = require_session()
        if err:
            return err
        data = request.get_json(force=True, silent=True) or {}
        name = (data.get("name") or "default").strip()[:80] or "default"
        try:
            return jsonify(create_api_key_for_user(user_id, name)), 201
        except ValueError as e:
            return jsonify(error=str(e)), 409

    @app.get("/console/keys")
    def list_keys():
        user_id, err = require_session()
        if err:
            return err
        with get_cursor() as cur:
            cur.execute(
                """SELECT id, name, key_prefix AS prefix, created_at
                   FROM api_keys WHERE user_id = %s AND revoked_at IS NULL
                   ORDER BY created_at DESC""",
                (user_id,),
            )
            rows = cur.fetchall()
        return jsonify(keys=[dict(r, created_at=r["created_at"].isoformat()) for r in rows])

    @app.delete("/console/keys/<int:key_id>")
    def revoke_key(key_id):
        user_id, err = require_session()
        if err:
            return err
        with get_cursor(commit=True) as cur:
            cur.execute(
                """UPDATE api_keys SET revoked_at = now()
                   WHERE id = %s AND user_id = %s AND revoked_at IS NULL""",
                (key_id, user_id),
            )
        return "", 204

    @app.get("/console/usage")
    def console_usage():
        user_id, err = require_session()
        if err:
            return err

        key_id = get_users_active_key_id(user_id)
        if not key_id:
            return jsonify(requests_today=0, daily_cap=DAILY_REQUEST_CAP, history=[0] * 7)

        today = dt.date.today()
        history = []
        with get_cursor() as cur:
            for i in range(6, -1, -1):
                day = today - dt.timedelta(days=i)
                cur.execute(
                    "SELECT request_count FROM usage_counters WHERE api_key_id = %s AND usage_date = %s",
                    (key_id, day),
                )
                row = cur.fetchone()
                history.append(row["request_count"] if row else 0)

        return jsonify(requests_today=history[-1], daily_cap=DAILY_REQUEST_CAP, history=history)

    # ---------- Cloudinary upload endpoint ----------

    @app.post("/console/upload")
    def upload_to_cloudinary():
        """Upload a file to Cloudinary and return the URL."""
        user_id, err = require_session()
        if err:
            return err

        if 'file' not in request.files:
            return jsonify(error="no file provided"), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify(error="no file selected"), 400

        # Check file size (10MB max)
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > 10 * 1024 * 1024:
            return jsonify(error="file too large (max 10MB)"), 413

        if not CLOUDINARY_CLOUD_NAME or not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET:
            return jsonify(error="Cloudinary not configured"), 500

        try:
            import cloudinary
            import cloudinary.uploader

            cloudinary.config(
                cloud_name=CLOUDINARY_CLOUD_NAME,
                api_key=CLOUDINARY_API_KEY,
                api_secret=CLOUDINARY_API_SECRET,
                secure=True,
            )

            result = cloudinary.uploader.upload(
                file,
                resource_type="auto",
                folder="sentry_uploads"
            )

            return jsonify({
                "url": result["secure_url"],
                "public_id": result["public_id"],
                "resource_type": result["resource_type"],
                "format": result.get("format"),
                "bytes": result["bytes"]
            })
        except Exception as e:
            logger.error(f"Cloudinary upload failed: {e}")
            return jsonify(error=f"upload failed: {str(e)}"), 500

    # ---------- completions (shared core, two auth front-ends) ----------

    def completion_core(body: dict, api_key_id: int):
        """Returns (response_dict, status, count_after_increment)."""
        messages = body.get("messages") or []
        if not messages:
            return {"error": {"message": "messages required"}}, 400, None

        tokens_in = estimate_tokens(messages)
        if tokens_in > MAX_INPUT_TOKENS:
            return {"error": {"message": f"request exceeds the {MAX_INPUT_TOKENS}-token limit"}}, 413, None

        count = check_and_increment_rate_limit(api_key_id, tokens_in=tokens_in)
        set_rate_limit_headers(count)
        if count > DAILY_REQUEST_CAP:
            resp = jsonify(error={"message": "daily request cap exceeded"})
            resp.status_code = 429
            resp.headers["Retry-After"] = str(seconds_until_midnight_utc())
            return None, 429, count

        web_search_enabled = bool((body.get("web_search") or {}).get("enabled"))
        stream = bool(body.get("stream"))

        try:
            result = run_agentic_completion(
                messages, 
                client_body=body, 
                web_search_enabled=web_search_enabled,
                stream=stream
            )
        except ProviderError as e:
            logger.error(f"Provider error: {e}")
            return {"error": {"message": str(e)}}, 502, count

        # Handle streaming response
        if result.get("streaming") and "stream" in result:
            return {
                "streaming": True,
                "stream": result["stream"],
                "provider": result["provider"],
            }, 200, count

        # Validate JSON if response_format requested
        response_format = body.get("response_format")
        content = result["content"]
        if response_format:
            try:
                content = validate_and_fix_json(content, response_format, messages, body, api_key_id)
            except ProviderError as e:
                return {"error": {"message": str(e)}}, 502, count

        tokens_out = estimate_tokens([{"content": content}])
        record_tokens_out(api_key_id, tokens_out)

        payload = {
            "id": "chatcmpl-sentry1",
            "object": "chat.completion",
            "model": "sentry-1",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}}
            ],
            "_provider_used": result["provider"],
            "usage": {"prompt_tokens_est": tokens_in, "completion_tokens_est": tokens_out},
        }
        if result.get("web_search_degraded"):
            payload["_web_search_degraded"] = True
        return payload, 200, count

    def sse_response(stream_data, provider):
        """Create a proper SSE response."""
        return Response(
            stream_with_context(sse_stream_generator(stream_data)),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Provider-Used": provider,
            }
        )

    @app.post("/v1/chat/completions")
    def chat_completions():
        identity = authenticate_api_key(bearer_token())
        if not identity:
            return jsonify(error={"message": "invalid API key"}), 401

        body = request.get_json(force=True, silent=True) or {}
        payload, status, _ = completion_core(body, identity["api_key_id"])
        if status == 429:
            return payload
        if status != 200:
            return jsonify(payload), status
        if payload.get("streaming"):
            return sse_response(payload["stream"], payload["provider"])
        return jsonify(payload)

    @app.post("/console/playground/chat")
    def playground_chat():
        user_id, err = require_session()
        if err:
            body, code = err
            return jsonify(error=body.get_json().get("error", "unauthorized")), code

        key_id = get_users_active_key_id(user_id)
        if not key_id:
            return jsonify(error={"message": "create an API key first"}), 400

        body = request.get_json(force=True, silent=True) or {}
        payload, status, _ = completion_core(body, key_id)
        if status == 429:
            return payload
        if status != 200:
            return jsonify(payload), status
        if payload.get("streaming"):
            return sse_response(payload["stream"], payload["provider"])
        return jsonify(payload)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
