"""
Fluid Intelligence backend — Gemini-only edition.

What changed in this edit (everything the user asked for):
1) Gemini is now the ONLY active provider. Groq/Mistral/Zai code stays in the
   file but is disconnected via GEMINI_ONLY=1 (default). Flip the env var to
   re-enable the old router.
2) Gemini key slots: GEMINI_API_KEY_1 / GEMINI_API_KEY_2 env vars, or a single
   GEMINI_API_KEY env var with comma-separated keys.
3) JSON mode is now driven the same way as web search: the system prompt tells
   the model JSON mode is on and exactly what schema to produce; the backend
   validates and retries once on failure (validate_and_fix_json kept).
4) MAX_INPUT_TOKENS raised to the Gemini context maximum (1M), env-overridable,
   and identical across every Gemini model we route to.
5) Reasoning effort now maps to different Gemini models (max input stays 1M):
      low    -> gemini-3.1-flash-lite
      medium -> gemini-3-flash-preview
      high   -> gemini-3.1-pro-preview
   (model IDs are env-overridable via GEMINI_MODEL_LOW/MEDIUM/HIGH.)
6) Tavily: the model controls it via tags, like web search already did:
      [SEARCH: query | n=10 | depth=advanced | images=yes]
      [FETCH: https://example.com/article]   <- full-page crawl of any URL
   Search results can include images; those URLs are injected into context and
   the model is explicitly allowed to reference them as markdown images
   (![desc](url)) which the playground renders inline. Non-streaming responses
   also carry a _search_images field.
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

APP_VERSION = os.environ.get("APP_VERSION", "gemini-only-2026-09-16")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://avnadmin:AVNS_q_zR9FvhvJHJGbiL0zp@pg-235735e2-ub7499710-7253.j.aivencloud.com:11602/defaultdb?sslmode=require",
)

JWT_SECRET = "21c8524d2320d22706fb366fc6ca9037050cdfda26c104ac0ecfa0215bebda7c"
SESSION_HOURS = 24 * 7
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

DAILY_REQUEST_CAP = int(os.environ.get("DAILY_REQUEST_CAP", "3000"))
# Raised to the Gemini context maximum. Same value for every Gemini model.
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "1000000"))
MAX_TOOL_ITERATIONS = 6
MAX_SEARCH_RESULTS = 20        # hard cap on n= the model may request per search
MAX_FETCH_CHARS = 8000         # chars of a crawled page injected per [FETCH:]

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

# --- Provider selection -------------------------------------------------
# Gemini-only is the default now. Set GEMINI_ONLY=0 to re-enable the old
# multi-provider router (groq/mistral/zai code paths are still in the file).
GEMINI_ONLY = os.environ.get("GEMINI_ONLY", "1") == "1"

# Cloudinary config (for image/video uploads)
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "ddusfl7pi")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", "599965682593626")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "pUcb90_1jtv-rDlHXRRsfDcBK5k")
CLOUDINARY_UPLOAD_PRESET = os.environ.get("CLOUDINARY_UPLOAD_PRESET", "")

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
# AUTH
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
# BACKEND COOLDOWNS (kept for Gemini key rotation)
# =========================================================================

def _hash_key(key: str) -> str:
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def mark_key_down(provider: str, key: str, error, cooldown_seconds: int = 30):
    """Cooldowns are disabled — no-op. Every request tries every configured
    key fresh."""
    pass


def get_key_cooldown_info(provider: str, key: str):
    key_hash = _hash_key(key)
    with get_cursor() as cur:
        cur.execute(
            "SELECT unavailable_until, last_error FROM key_status WHERE provider = %s AND key_hash = %s",
            (provider, key_hash)
        )
        row = cur.fetchone()
    if not row:
        return None, None
    return row["unavailable_until"], row["last_error"]


def is_key_available(provider: str, key: str) -> bool:
    return True


def mark_backend_down(name: str, error, cooldown_seconds: int = 30):
    pass


def is_backend_available(name: str) -> bool:
    return True


# =========================================================================
# MODEL PROVIDERS — Gemini active, others disconnected (not deleted)
# =========================================================================

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
        "reasoning": True,
    },
}

DEFAULT_ORDER = ["groq", "mistral", "zai", "gemini"]

# --- Gemini: model per reasoning effort (max input identical for all: 1M) ---
# NOTE (fixed 2026-09-16): "gemini-3.1-flash" and "gemini-3.1-pro" are not
# real model IDs — that was the cause of the 404 "is not found for API
# version v1beta" error. The actual current IDs are gemini-3.1-flash-lite
# (stable), gemini-3-flash-preview (mid-tier, has a free tier), and
# gemini-3.1-pro-preview (top tier — check your quota, Pro-class models may
# need billing enabled even on an otherwise-free project).
GEMINI_MODEL_DEFAULT = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_MODEL_BY_EFFORT = {
    "low": os.environ.get("GEMINI_MODEL_LOW", "gemini-3.1-flash-lite"),
    "medium": os.environ.get("GEMINI_MODEL_MEDIUM", "gemini-3-flash-preview"),
    "high": os.environ.get("GEMINI_MODEL_HIGH", "gemini-3.1-pro-preview"),
}


def _gemini_model_for_request(client_body: dict) -> str:
    """Pick the Gemini model from the requested reasoning effort.
    MAX_INPUT_TOKENS is the same (1M) for every model in this map."""
    reasoning = (client_body or {}).get("reasoning") or {}
    if reasoning.get("enabled"):
        effort = (reasoning.get("effort") or "medium").lower()
        return GEMINI_MODEL_BY_EFFORT.get(effort, GEMINI_MODEL_DEFAULT)
    return GEMINI_MODEL_DEFAULT


# Gemini keys — two slots, or one comma-separated GEMINI_API_KEY env var.
GEMINI_API_KEY_ENVS = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"]
GEMINI_API_KEY_DEFAULTS = ["", ""]
GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEY", "").split(",") if k.strip()]
if not GEMINI_KEYS:
    GEMINI_KEYS = [
        os.environ.get(e, d)
        for e, d in zip(GEMINI_API_KEY_ENVS, GEMINI_API_KEY_DEFAULTS)
    ]
    GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

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

PASSTHROUGH_PARAMS = ("temperature", "top_p", "top_k", "max_tokens", "stop", "seed",
                      "presence_penalty", "frequency_penalty", "response_format")


class ProviderError(Exception):
    pass


def has_media(messages: list) -> bool:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("image_url", "input_audio", "audio_url", "video_url"):
                    return True
    return False


def _transcribe_audio_bytes(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Voice notes are transcribed via Groq Whisper. NOTE: with GEMINI_ONLY=1
    this still runs (it's transcription, not chat) — if no Groq key is valid
    it raises and the caller falls back gracefully."""
    keys = [os.environ.get(e, d) for e, d in zip(OPENAI_STYLE["groq"]["api_key_envs"], OPENAI_STYLE["groq"]["api_key_defaults"])]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError("audio transcription: no Groq API key configured")

    last_error = None
    for key in keys:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (filename, audio_bytes)},
                data={"model": "whisper-large-v3-turbo"},
                timeout=60,
            )
            if resp.status_code >= 400:
                last_error = f"groq whisper: {resp.status_code} — {_error_detail(resp)}"
                logger.error(f"audio transcription failed: {last_error}")
                continue
            text = resp.json().get("text", "").strip()
            logger.info(f"audio transcription succeeded, {len(text)} chars")
            return text
        except requests.RequestException as e:
            last_error = f"groq whisper: {e}"
            logger.error(f"audio transcription request failed: {e}")
            continue
    raise ProviderError(last_error or "audio transcription failed on all keys")


def _transcribe_audio_blocks(messages: list) -> list:
    out = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue

        new_content = []
        changed = False
        for block in content:
            btype = block.get("type") if isinstance(block, dict) else None
            if btype not in ("audio_url", "input_audio"):
                new_content.append(block)
                continue

            changed = True
            try:
                if btype == "audio_url":
                    url = (block.get("audio_url") or {}).get("url")
                    audio_resp = requests.get(url, timeout=30)
                    audio_resp.raise_for_status()
                    audio_bytes = audio_resp.content
                else:
                    import base64
                    b64_data = (block.get("input_audio") or {}).get("data", "")
                    audio_bytes = base64.b64decode(b64_data)

                transcript = _transcribe_audio_bytes(audio_bytes)
                new_content.append({
                    "type": "text",
                    "text": f"[Voice message transcript]: {transcript}" if transcript else "[Voice message was empty or inaudible]"
                })
            except (requests.RequestException, ProviderError, ValueError) as e:
                logger.error(f"failed to transcribe audio block: {e}")
                new_content.append({"type": "text", "text": f"[Voice message could not be transcribed: {e}]"})

        out.append({**m, "content": new_content if changed else content})
    return out


def estimate_tokens(messages) -> int:
    return max(1, len(json.dumps(messages)) // 4)


def _forward_sampling_params(body: dict, client_body: dict):
    for p in PASSTHROUGH_PARAMS:
        if p in client_body and client_body[p] is not None:
            body[p] = client_body[p]


def _get_reasoning_params(provider: str, client_body: dict) -> dict:
    """Reasoning params for providers that support them (disconnected providers kept for completeness)."""
    reasoning = client_body.get("reasoning") or {}
    if not reasoning.get("enabled"):
        return {}
    effort = reasoning.get("effort", "medium")
    if provider == "groq":
        return {"reasoning_effort": effort}
    return {}


def _error_detail(resp) -> str:
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
    """DISCONNECTED under GEMINI_ONLY=1 — kept in the file, not deleted."""
    cfg = OPENAI_STYLE[provider]
    keys = [os.environ.get(e, d) for e, d in zip(cfg["api_key_envs"], cfg["api_key_defaults"])]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError(f"{provider}: no API keys configured")

    body = {"model": cfg["model"], "messages": messages}
    _forward_sampling_params(body, client_body or {})

    if cfg.get("supports_reasoning") and not tools:
        body.update(_get_reasoning_params(provider, client_body or {}))

    if tools:
        body["tools"] = tools
    if cfg.get("extra_body"):
        body.update(cfg["extra_body"])
    if stream:
        body["stream"] = True

    last_error = None
    for key in keys:
        if not is_key_available(provider, key):
            until, prev_err = get_key_cooldown_info(provider, key)
            last_error = f"{provider}: key {_hash_key(key)} still cooling down until {until} — last error: {prev_err}"
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
                continue
            if resp.status_code in (401, 403):
                last_error = f"{provider}: key {_hash_key(key)} rejected with {resp.status_code} — {_error_detail(resp)}"
                mark_key_down(provider, key, last_error, cooldown_seconds=6 * 3600)
                continue
            if resp.status_code >= 500:
                last_error = f"{provider}: server error {resp.status_code} — {_error_detail(resp)}"
                mark_key_down(provider, key, last_error, cooldown_seconds=30)
                continue
            if resp.status_code >= 400:
                last_error = f"{provider}: {resp.status_code} — {_error_detail(resp)}"
                mark_key_down(provider, key, last_error, cooldown_seconds=5)
                continue

            if stream:
                return {"stream": resp.iter_lines(), "provider": provider, "key_hash": _hash_key(key)}

            data = resp.json()
            choice = data["choices"][0]["message"]
            usage = data.get("usage", {})
            return {
                "content": choice.get("content"),
                "tool_calls": choice.get("tool_calls"),
                "provider": provider,
                "usage": usage,
            }
        except requests.RequestException as e:
            last_error = f"{provider}: {e}"
            mark_key_down(provider, key, last_error, cooldown_seconds=30)
            continue

    raise ProviderError(last_error or f"{provider}: all keys failed")


def _fetch_media_as_base64(url: str, max_size_mb: int = 5) -> tuple:
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        content_length = resp.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            raise ProviderError(f"Media too large: {int(content_length) / 1024 / 1024:.1f}MB > {max_size_mb}MB")

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

        content_type = resp.headers.get('content-type', '').split(';')[0]
        if not content_type:
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
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '').lower()
        ext = url.split('.')[-1].lower().split('?')[0]

        if 'pdf' in content_type or ext == 'pdf':
            return _extract_pdf_text(resp.content)
        elif 'word' in content_type or ext in ('docx', 'doc'):
            return _extract_docx_text(resp.content)
        elif 'text' in content_type or ext in ('txt', 'md', 'csv'):
            return resp.text[:100000]
        else:
            try:
                return resp.text[:100000]
            except Exception:
                return "[Unsupported file type]"
    except requests.RequestException as e:
        raise ProviderError(f"Failed to fetch file: {e}")


def _extract_pdf_text(content: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > 100000:
                text += "\n\n[content truncated]"
                break
        doc.close()
        return text
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text
        import io
        text = extract_text(io.BytesIO(content))
        return text[:100000]
    except ImportError:
        pass

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
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text[:100000]
    except ImportError:
        return "[DOCX support not available. Install python-docx or convert to PDF/TXT]"


def _call_gemini(messages: list, timeout: int, stream: bool = False, model: str = None):
    """The one active provider. Model is chosen per reasoning effort."""
    model = model or GEMINI_MODEL_DEFAULT
    keys = GEMINI_KEYS
    if not keys:
        raise ProviderError("gemini: no API keys configured (set GEMINI_API_KEY or GEMINI_API_KEY_1/2)")

    body = {"contents": _messages_to_gemini_contents(messages)}
    if stream:
        body["generationConfig"] = {"maxOutputTokens": 8192}

    last_error = None
    for key in keys:
        if not is_key_available("gemini", key):
            until, prev_err = get_key_cooldown_info("gemini", key)
            last_error = f"gemini: key {_hash_key(key)} still cooling down until {until} — last error: {prev_err}"
            logger.info(f"gemini: key {_hash_key(key)} cooling down, skipping")
            continue

        try:
            start_time = dt.datetime.now()
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:{'streamGenerateContent' if stream else 'generateContent'}?key={key}"
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
                logger.info(f"gemini({model}): streaming started, key {_hash_key(key)}, latency {latency:.0f}ms")
                return {
                    "stream": resp.iter_lines(),
                    "provider": "gemini",
                    "model": model,
                    "key_hash": _hash_key(key),
                }

            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]

            usage = data.get("usageMetadata", {})
            logger.info(f"gemini({model}): success, key {_hash_key(key)}, latency {latency:.0f}ms, tokens {usage.get('promptTokenCount', '?')}/{usage.get('candidatesTokenCount', '?')}")

            return {"content": text, "tool_calls": None, "provider": "gemini", "model": model, "usage": usage}
        except requests.RequestException as e:
            last_error = f"gemini: {e}"
            mark_key_down("gemini", key, last_error, cooldown_seconds=30)
            logger.error(f"gemini: error on key {_hash_key(key)}: {e}")
            continue

    raise ProviderError(last_error or "gemini: all keys failed")


def _rank_providers(messages: list, tools, stream: bool, client_body: dict) -> list:
    """Capability ranking — only used when GEMINI_ONLY=0."""
    needs_media = has_media(messages)
    needs_tools = bool(tools)
    wants_reasoning = bool(((client_body or {}).get("reasoning") or {}).get("enabled"))

    candidates = []
    for pos, name in enumerate(DEFAULT_ORDER):
        caps = PROVIDERS[name]
        if not caps["text"]:
            continue
        score = pos

        use_tools = needs_tools
        msgs = messages

        if needs_media:
            if caps["media"]:
                score -= 5
            elif caps["text"]:
                score += 10
            else:
                continue

        if needs_tools:
            if caps["tools"]:
                pass
            else:
                score += 10
                use_tools = False

        if wants_reasoning and caps.get("reasoning"):
            score -= 1

        candidates.append((score, pos, name, msgs, use_tools))

    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates


def call_model(messages: list, client_body: dict = None, tools=None, timeout: int = 30, order=None, stream: bool = False):
    """Route to a provider. With GEMINI_ONLY=1 (default) everything goes to
    Gemini with the effort-based model pick; other providers stay in the file
    but are disconnected."""
    errors = []

    if GEMINI_ONLY:
        names = ["gemini"]
    elif order is not None:
        names = order
    else:
        env_order = os.environ.get("PROVIDER_ORDER")
        if env_order:
            names = [p.strip() for p in env_order.split(",")]
        else:
            names = None

    if names:
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

    model = _gemini_model_for_request(client_body)

    for score, pos, name, msgs, use_tools in ranked:
        if score >= 2:
            continue
        try:
            if name == "gemini":
                return _call_gemini(msgs, timeout, stream=stream, model=model)
            return _call_openai_style(name, msgs, client_body, use_tools, timeout, stream=stream)
        except (ProviderError, requests.RequestException) as e:
            errors.append(f"{name}: {e}")
            logger.error(f"Provider {name} failed: {e}")
            continue

    if not GEMINI_ONLY:
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

        if tools:
            for score, pos, name, msgs, use_tools in ranked:
                if use_tools:
                    continue
                try:
                    logger.warning(f"all tool-capable providers unavailable; trying {name} without tools")
                    if name == "gemini":
                        return _call_gemini(messages, timeout, stream=stream, model=model)
                    return _call_openai_style(name, messages, client_body, None, timeout, stream=stream)
                except (ProviderError, requests.RequestException) as e:
                    errors.append(f"{name} (no-tools fallback): {e}")
                    logger.error(f"Provider {name} no-tools fallback failed: {e}")
                    continue

    raise ProviderError("All providers/keys failed -> " + " | ".join(errors))


# =========================================================================
# RESPONSE FORMAT VALIDATION (JSON mode)
# =========================================================================

def validate_and_fix_json(content: str, response_format: dict, messages: list, client_body: dict, api_key_id: int) -> str:
    """Validate JSON, retry once at temperature=0 with a stricter instruction."""
    if not response_format:
        return content

    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass

    logger.info("JSON validation failed, retrying with temperature=0")

    retry_body = dict(client_body or {})
    retry_body["temperature"] = 0
    retry_body["_json_retry"] = True  # signal to build the stricter system prompt

    retry_messages = list(messages)
    retry_messages.insert(0, {
        "role": "system",
        "content": "JSON MODE RETRY: your previous response was not valid JSON. Respond with valid JSON only — no markdown fences, no commentary, no trailing text. Match the requested schema exactly."
    })

    try:
        result = call_model(retry_messages, client_body=retry_body)
        retry_content = result["content"]
        json.loads(retry_content)
        return retry_content
    except (json.JSONDecodeError, ProviderError) as e:
        logger.error(f"JSON retry failed: {e}")
        raise ProviderError("Could not produce valid JSON after retry")


# =========================================================================
# TOOLS — Tavily live web search + page crawl (model-controlled via tags)
# =========================================================================

TAVILY_KEYS = [
    os.environ.get("TAVILY_API_KEY_1", "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"),
    os.environ.get("TAVILY_API_KEY_2", "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"),
]


def _pick_tavily_key():
    for key in TAVILY_KEYS:
        if key:
            return key
    return None


def web_search(query: str, max_results: int = 5, search_depth: str = "basic",
               include_images: bool = True):
    """Tavily search with images. Returns {"results": [...], "images": [...]}.
    search_depth: "basic" | "advanced" (advanced crawls more, costs more)."""
    max_results = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    last_error = None
    for key in TAVILY_KEYS:
        if not key:
            continue
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                    "include_images": include_images,
                    "include_image_descriptions": include_images,
                },
                timeout=20,
            )
            if resp.status_code == 429:
                last_error = "rate limited"
                continue
            resp.raise_for_status()
            data = resp.json()
            results = [
                {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
                for r in data.get("results", [])
            ]
            images = [
                {"url": img.get("url"), "description": img.get("description") or ""}
                for img in (data.get("images") or [])
                if img.get("url")
            ][:10]
            return {"results": results, "images": images}
        except requests.RequestException as e:
            last_error = str(e)
            continue
    raise RuntimeError(f"Tavily search failed on all keys: {last_error}")


def fetch_page_text(url: str, max_chars: int = MAX_FETCH_CHARS) -> tuple:
    """Crawl any URL and return (title, visible text). Used for [FETCH: url]."""
    import html as html_mod

    resp = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SentryBot/1.0; +https://api.sentryai.ai)"},
    )
    resp.raise_for_status()
    raw = resp.text

    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else url

    raw = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|li|h1|h2|h3|h4|h5|h6|tr|td|th|blockquote)>", "\n", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html_mod.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

    return title, text[:max_chars]


TOOL_IMPLEMENTATIONS = {"web_search": web_search, "fetch_page": fetch_page_text}

SEARCH_TAG_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE | re.DOTALL)
FETCH_TAG_RE = re.compile(r"\[FETCH:\s*(https?://[^\s\]]+)\]", re.IGNORECASE)


def _parse_search_tag(raw: str):
    """[SEARCH: query | n=10 | depth=advanced | images=no] -> (query, opts)."""
    parts = [p.strip() for p in raw.split("|")]
    query = parts[0]
    opts = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip().lower()] = v.strip()
    return query, opts


def _extract_search_tags(text: str):
    if not text:
        return []
    return [q.strip() for q in SEARCH_TAG_RE.findall(text) if q.strip()]


def _extract_fetch_tags(text: str):
    if not text:
        return []
    return [u.strip() for u in FETCH_TAG_RE.findall(text) if u.strip()]


def _format_search_context(query, search_data) -> str:
    lines = [f'Search results for "{query}":']
    for i, r in enumerate(search_data.get("results", []), 1):
        snippet = (r.get("content") or "")[:400]
        lines.append(f'{i}. {r.get("title")}\n   {r.get("url")}\n   {snippet}')
    images = search_data.get("images") or []
    if images:
        lines.append("")
        lines.append("Images found (you may embed any of these in your answer as markdown: ![description](url)):")
        for img in images:
            desc = img.get("description") or "image"
            lines.append(f'- {img.get("url")} — {desc}')
    return "\n".join(lines)


def _inject_tavily_results(messages: list, query: str, max_results: int = 5) -> tuple:
    """Provider-degraded fallback: search the user's question, inject context.
    Returns (messages, images)."""
    try:
        search_data = web_search(query, max_results=max_results)
        if not search_data.get("results"):
            return messages, []
        context = _format_search_context(query, search_data)
        injected = list(messages)
        injected.insert(-1, {"role": "system", "content": context})
        return injected, search_data.get("images", [])
    except Exception as e:
        logger.warning(f"Tavily pre-search failed: {e}")
        return messages, []


def run_agentic_completion(messages: list, client_body: dict = None, web_search_enabled: bool = True, stream: bool = False):
    """Model -> [SEARCH: ...] / [FETCH: ...] tags -> real results injected ->
    model again, until no tags appear or MAX_TOOL_ITERATIONS is hit.

    The Gemini model controls Tavily directly through tags:
      [SEARCH: your query]                          basic search, ~5 results + images
      [SEARCH: your query | n=10]                   up to 20 results
      [SEARCH: your query | depth=advanced]         deeper crawl search
      [SEARCH: your query | images=no]              text-only results
      [FETCH: https://example.com/page]             crawl that exact page
    """
    working = list(messages)
    degraded = False
    collected_images = []
    default_max_results = int(((client_body or {}).get("web_search") or {}).get("max_results") or 5)

    for iteration in range(MAX_TOOL_ITERATIONS):
        is_final = (iteration == MAX_TOOL_ITERATIONS - 1)
        should_stream = stream and is_final and not degraded

        try:
            result = call_model(working, client_body=client_body, tools=None, stream=should_stream)
        except ProviderError as e:
            if not web_search_enabled or degraded:
                raise
            logger.warning(f"Model call failed ({e}); falling back to Tavily pre-search")
            degraded = True

            last_user = ""
            for m in reversed(working):
                if m.get("role") == "user":
                    c = m.get("content")
                    last_user = c if isinstance(c, str) else ""
                    break

            if last_user:
                working, imgs = _inject_tavily_results(working, last_user)
                collected_images.extend(imgs)

            result = call_model(working, client_body=client_body, tools=None, stream=False)

        if should_stream and "stream" in result and not degraded:
            return {
                "stream": result["stream"],
                "provider": result["provider"],
                "model": result.get("model"),
                "streaming": True,
                "images": collected_images,
            }

        content = result.get("content") or ""
        search_tags = _extract_search_tags(content) if web_search_enabled else []
        fetch_tags = _extract_fetch_tags(content) if web_search_enabled else []

        if not search_tags and not fetch_tags:
            out = {
                "content": content,
                "provider": result["provider"],
                "model": result.get("model"),
                "streaming": False,
                "images": collected_images,
            }
            if degraded:
                out["web_search_degraded"] = True
            return out

        # Keep the model's tag message for context, inject real results, loop.
        working.append({"role": "assistant", "content": content})

        for raw_tag in search_tags:
            query, opts = _parse_search_tag(raw_tag)
            if not query:
                continue
            n = int(opts.get("n", default_max_results) or default_max_results)
            depth = opts.get("depth", "basic")
            include_images = opts.get("images", "yes").lower() not in ("no", "false", "0")
            try:
                search_data = web_search(query, max_results=n, search_depth=depth, include_images=include_images)
                collected_images.extend(search_data.get("images", []))
                tool_output = _format_search_context(query, search_data)
            except Exception as e:
                tool_output = f'Search for "{query}" failed: {e}'
            working.append({
                "role": "system",
                "content": (
                    f"{tool_output}\n\n"
                    "Answer using these results. Only output another [SEARCH: ...] "
                    "tag if you genuinely need a different query, or [FETCH: url] to "
                    "read a specific page in full."
                ),
            })

        for url in fetch_tags:
            try:
                title, text = fetch_page_text(url)
                tool_output = f'Fetched page: {title}\nURL: {url}\n\n{text}'
            except Exception as e:
                tool_output = f'Failed to fetch {url}: {e}'
            working.append({
                "role": "system",
                "content": (
                    f"{tool_output}\n\n"
                    "Use this page content to answer. You may [FETCH: another-url] if needed."
                ),
            })

    out = {
        "content": "I wasn't able to finish that within the search-iteration limit.",
        "provider": None,
        "streaming": False,
        "images": collected_images,
    }
    if degraded:
        out["web_search_degraded"] = True
    return out


# =========================================================================
# SYSTEM PROMPT — web search + JSON mode, same pattern
# =========================================================================

def build_system_prompt(body: dict, is_json_retry: bool = False) -> str:
    web_cfg = body.get("web_search") or {}
    web_enabled = bool(web_cfg.get("enabled"))
    default_max_results = int(web_cfg.get("max_results") or 5)
    response_format = body.get("response_format")
    json_enabled = bool(response_format)

    lines = ["You are Sentry 1, an AI assistant accessed via the Fluid Intelligence API."]

    # ---- live web search (Tavily), model-controlled via tags ----
    if web_enabled:
        lines.append("")
        lines.append("LIVE WEB SEARCH (Tavily) — currently ENABLED. You control it directly by outputting tags in your reply:")
        lines.append('- [SEARCH: your query]  — search the live web (default ~%d results, includes images)' % default_max_results)
        lines.append('- [SEARCH: your query | n=10]  — control how many results (max 20)')
        lines.append('- [SEARCH: your query | depth=advanced]  — deeper, more thorough crawl search')
        lines.append('- [SEARCH: your query | images=no]  — text-only results')
        lines.append('- [FETCH: https://example.com/page]  — crawl one specific URL and read its full text')
        lines.append("")
        lines.append("SEARCH RULES:")
        lines.append("- Only use these tags for current events, news, prices, weather, or data that may have changed recently.")
        lines.append("- Do NOT use them for general knowledge, math, coding, or historical facts.")
        lines.append("- After results are injected, answer using them. Don't re-search the same thing.")
        lines.append("- If Tavily returns images, you MAY embed them in your answer as markdown images, e.g. ![description](image-url), when a visual reference genuinely helps.")
        lines.append("- You may also cite sources as normal markdown links.")
    else:
        lines.append("")
        lines.append("LIVE WEB SEARCH (Tavily) — currently DISABLED by the user.")
        lines.append("- If asked about current events, news, prices, weather, or recent data, say search is available but turned off and suggest enabling it.")
        lines.append("- Do not pretend to search. Answer general knowledge directly.")

    # ---- JSON mode, same pattern: enabled via system prompt + enforced server-side ----
    if json_enabled and not is_json_retry:
        lines.append("")
        lines.append("JSON MODE — currently ENABLED. Your entire reply MUST be a single valid JSON value:")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            if schema:
                lines.append("You MUST conform to this JSON schema exactly:")
                lines.append(json.dumps(schema, indent=2))
        else:
            lines.append("Respond with a JSON object that best fits the request.")
        lines.append("")
        lines.append("JSON RULES:")
        lines.append("- Output raw JSON only: no markdown code fences, no explanation before or after, no trailing commas, no comments.")
        lines.append("- The server validates your output; invalid JSON triggers a retry, so get it right the first time.")

    if is_json_retry:
        lines.append("")
        lines.append("JSON MODE RETRY: your previous reply was not valid JSON. Output corrected raw JSON only, conforming to the schema above. No markdown fences, no commentary.")

    return "\n".join(lines)


# =========================================================================
# STREAMING HELPERS
# =========================================================================

def sse_stream_generator(stream_data):
    """SSE format from provider stream. Gemini lines pass through as-is."""
    try:
        for line in stream_data:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if not line:
                continue
            if line.startswith('data: '):
                yield line + '\n\n'
            elif line.startswith('{'):
                try:
                    chunk = json.loads(line)
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
        return jsonify(ok=True, version=APP_VERSION, gemini_only=GEMINI_ONLY,
                       gemini_keys_configured=len(GEMINI_KEYS))

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
        return jsonify(data=[
            {"id": "sentry-1", "object": "model"},
            {"id": "gemini-3.1-flash-lite", "object": "model"},
            {"id": "gemini-3-flash-preview", "object": "model"},
            {"id": "gemini-3.1-pro-preview", "object": "model"},
        ])

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
        user_id, err = require_session()
        if err:
            return err

        if 'file' not in request.files:
            return jsonify(error="no file provided"), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify(error="no file selected"), 400

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
        """Returns (response_dict_or_flask_response_parts, status, count)."""
        messages = body.get("messages") or []
        if not messages:
            return {"error": {"message": "messages required"}}, 400, None

        is_json_retry = bool(body.get("_json_retry"))
        system_content = build_system_prompt(body, is_json_retry=is_json_retry)
        messages.insert(0, {"role": "system", "content": system_content})

        try:
            messages = _transcribe_audio_blocks(messages)
        except Exception as e:
            logger.error(f"audio preprocessing failed: {e}")
            return {"error": {"message": f"Failed to process audio: {e}"}}, 502, None

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
                "model": result.get("model"),
                "images": result.get("images", []),
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
            "_gemini_model": result.get("model"),
            "usage": {"prompt_tokens_est": tokens_in, "completion_tokens_est": tokens_out},
        }
        if result.get("images"):
            payload["_search_images"] = result["images"]
        if result.get("web_search_degraded"):
            payload["_web_search_degraded"] = True
        return payload, 200, count

    def sse_response(stream_data, provider, model=None, images=None):
        """Create a proper SSE response. Search images ride along as an SSE
        comment-style event the playground can pick up if it wants."""
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Provider-Used": provider,
        }
        if model:
            headers["X-Gemini-Model"] = model

        def generate():
            if images:
                yield f"event: search-images\ndata: {json.dumps({'images': images})}\n\n"
            for chunk in sse_stream_generator(stream_data):
                yield chunk

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers=headers,
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
            return sse_response(payload["stream"], payload["provider"],
                                model=payload.get("model"), images=payload.get("images"))
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
            return sse_response(payload["stream"], payload["provider"],
                                model=payload.get("model"), images=payload.get("images"))
        return jsonify(payload)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
