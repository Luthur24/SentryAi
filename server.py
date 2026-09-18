"""
Fluid Intelligence backend — text-only, multi-model fallback, branding-safe.

Architecture as of this rebuild (2026-09-17):

1) TEXT ONLY. No multimodal input anywhere — no Cloudinary, no audio
   transcription, no image/video/file blocks. Every provider is plain text.

2) NO COOLDOWNS. No DB-backed "key is down" state. Every request tries
   every model/key fresh; a failure reroutes immediately to the next one.
   Nothing persisted between requests.

3) NO STREAMING. Every response is one plain JSON payload.

4) NO REASONING TIERS. No low/medium/high, no reasoning_effort param.

5) MULTI-MODEL FALLBACK PER PROVIDER (new, replaces the single fixed model
   each provider had before). This is the fix for repeated "model does not
   exist / no access" 404s: if a model 404s, skip straight to the next
   MODEL for that provider (no point retrying a dead model ID on another
   key) — only fall through to the next PROVIDER once every model for the
   current one is exhausted.
       gemini  -> gemini-3.7-flash, gemini-3.5-flash, gemini-3.5-flash-lite
       groq    -> llama-3.3-70b-versatile, openai/gpt-oss-120b,
                  llama-3.1-8b-instant
                  (dropped qwen/qwen3.6-27b and llama-4-scout — both
                  confirmed 404 "no access" on this account. These three
                  are Groq's established production text models, not
                  preview/vision tier, so far less likely to be gated)
       mistral -> ministral-8b-2512, ministral-3b-2512
                  (mistral-large-2512 still isn't on this account's tier)
       zai     -> glm-4.7-flash, glm-4.5-flash (both genuinely $0 on Z.ai's
                  own API)
   Env-overridable as comma-separated lists: GEMINI_MODELS / GROQ_MODELS /
   MISTRAL_MODELS / ZAI_MODELS.

6) MAX_INPUT_TOKENS = 120,000 (was 1,000,000, sized only for Gemini). The
   real context window of every model above: Gemini ~1M+, Mistral 262K,
   Zai 200K, Groq 131K — Groq's 131,072 is the tightest ceiling across all
   four providers now in play. 120,000 sits safely under that with room
   for output, while still clearing the requested 100K+ floor.

7) MAX_OUTPUT_TOKENS = 100,000 request-side ceiling, clamped per provider
   to what its models can actually produce (PROVIDER_MAX_OUTPUT_TOKENS) so
   an over-large max_tokens gets capped instead of erroring out.

8) BRANDING / PRIVACY (new, 2026-09-17): Sentry 1 is presented as its own
   system. The specific backend providers/models it routes across
   (Gemini, Groq, Mistral, Zai) are an internal implementation detail and
   must never reach anything a user or API caller can see:
     - The JSON response never includes which provider/model answered.
     - A total-failure error returns one generic message to the caller;
       the real per-provider detail goes to logger.error only (check
       server logs, not the API response, to see which backend failed).
     - The debug trace (_debug_trace / the playground's Debug toggle)
       shows stage/timing info and generic route labels ("route 1",
       "route 2", ...) — never real provider or model identifiers.
     - Real names live in three places only: this file's config, the
       DEFAULT_ORDER/OPENAI_STYLE/GEMINI_MODELS structures, and server
       logs. Nowhere else.

9) Tavily web search unchanged — model-controlled via tags:
      [SEARCH: query | n=10 | depth=advanced | images=yes]
      [FETCH: https://example.com/article]
"""

import os
import json
import secrets
import re
import logging
import time as _time
import datetime as dt
from datetime import timezone
from contextlib import contextmanager

import bcrypt
import jwt
import requests
from flask import Flask, request, jsonify, g
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

APP_VERSION = os.environ.get("APP_VERSION", "multimodel-branded-2026-09-17")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://avnadmin:AVNS_q_zR9FvhvJHJGbiL0zp@pg-235735e2-ub7499710-7253.j.aivencloud.com:11602/defaultdb?sslmode=require",
)

JWT_SECRET = "21c8524d2320d22706fb366fc6ca9037050cdfda26c104ac0ecfa0215bebda7c"
SESSION_HOURS = 24 * 7
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

DAILY_REQUEST_CAP = int(os.environ.get("DAILY_REQUEST_CAP", "3000"))
# Tightest real context window among the models actually in use (Groq's
# 131,072) minus headroom for output. See docstring item 6.
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "120000"))
# Ceiling a client may request via max_tokens. Actual delivered output is
# still bounded by PROVIDER_MAX_OUTPUT_TOKENS below.
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "100000"))
MAX_TOOL_ITERATIONS = 6
MAX_SEARCH_RESULTS = 20        # hard cap on n= the model may request per search
MAX_FETCH_CHARS = 8000         # chars of a crawled page injected per [FETCH:]

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

GEMINI_ONLY = os.environ.get("GEMINI_ONLY", "0") == "1"

# Real per-provider output ceilings, so a MAX_OUTPUT_TOKENS=100000 request
# gets clamped to what that provider can actually produce instead of
# erroring out. Groq's is set conservatively (8192) since it now covers
# three different models with different real ceilings and this is the
# safe shared floor across all of them.
PROVIDER_MAX_OUTPUT_TOKENS = {
    "gemini": int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "65536")),
    "groq": int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "8192")),
    "mistral": int(os.environ["MISTRAL_MAX_OUTPUT_TOKENS"]) if os.environ.get("MISTRAL_MAX_OUTPUT_TOKENS") else None,
    "zai": int(os.environ.get("ZAI_MAX_OUTPUT_TOKENS", "32000")),
}


def _clamp_max_tokens(provider: str, requested):
    """Cap a requested max_tokens to what `provider`'s models can actually
    produce. Falls back to MAX_OUTPUT_TOKENS when nothing was requested."""
    value = requested if requested else MAX_OUTPUT_TOKENS
    ceiling = PROVIDER_MAX_OUTPUT_TOKENS.get(provider)
    if ceiling:
        value = min(value, ceiling)
    return value


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
"""

MIGRATION_SQL = """
ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_in INTEGER NOT NULL DEFAULT 0;
ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_out INTEGER NOT NULL DEFAULT 0;
DROP TABLE IF EXISTS key_status;
DROP TABLE IF EXISTS backend_status;
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


def _hash_key(key: str) -> str:
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# =========================================================================
# MODEL PROVIDERS — internal only. Real names never leave this file except
# to server logs. See docstring item 8.
# =========================================================================

DEFAULT_ORDER = [p.strip() for p in os.environ.get("PROVIDER_ORDER", "gemini,groq,mistral,zai").split(",") if p.strip()]

GEMINI_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS", "gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite"
).split(",") if m.strip()]

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
        "models": [m.strip() for m in os.environ.get(
            "GROQ_MODELS", "llama-3.3-70b-versatile,openai/gpt-oss-120b,llama-3.1-8b-instant"
        ).split(",") if m.strip()],
        "extra_body": {},
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "api_key_envs": ["MISTRAL_API_KEY_1", "MISTRAL_API_KEY_2"],
        "api_key_defaults": [
            "uZIGi6VCzkcJ0i5X5mYYc0Nr1XWX21YR",
            "emd7ZA8yvckaHrGvZ9jsxKMA6y3wNSbn",
        ],
        "models": [m.strip() for m in os.environ.get(
            "MISTRAL_MODELS", "ministral-8b-2512,ministral-3b-2512"
        ).split(",") if m.strip()],
        "extra_body": {},
    },
    "zai": {
        "base_url": "https://api.z.ai/api/paas/v4/chat/completions",
        "api_key_envs": ["ZAI_API_KEY_1", "ZAI_API_KEY_2"],
        "api_key_defaults": [
            "ccd29e152e6941098d7e9e1fd91f24a1.thzU8g4LLNVm6qea",
            "f6bc4aac8cfe425cadde84bba181710e.wX0Wm0I8l3YDihGy",
        ],
        "models": [m.strip() for m in os.environ.get(
            "ZAI_MODELS", "glm-4.7-flash,glm-4.5-flash"
        ).split(",") if m.strip()],
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}

PASSTHROUGH_PARAMS = ("temperature", "top_p", "top_k", "stop", "seed",
                      "presence_penalty", "frequency_penalty", "response_format")


class Tracer:
    """Step-by-step trace for one request — timing and generic route labels
    only. Real provider/model identifiers are NEVER put in here; this
    object's .events is what the API response and the playground's Debug
    toggle both see, so anything added here is customer-visible by design.
    Use logger.info/error for anything that should name real backends."""

    def __init__(self):
        self._t0 = _time.time()
        self.events = []

    def add(self, stage: str, message: str, **data):
        entry = {
            "t_ms": int((_time.time() - self._t0) * 1000),
            "stage": stage,
            "message": message,
        }
        if data:
            safe = {}
            for k, v in data.items():
                try:
                    json.dumps(v)
                    safe[k] = v
                except TypeError:
                    safe[k] = str(v)
            entry["data"] = safe
        self.events.append(entry)
        return entry


class ProviderError(Exception):
    pass


def estimate_tokens(messages) -> int:
    return max(1, len(json.dumps(messages)) // 4)


def _forward_sampling_params(body: dict, client_body: dict):
    for p in PASSTHROUGH_PARAMS:
        if p in client_body and client_body[p] is not None:
            body[p] = client_body[p]


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


def _call_openai_style(provider: str, messages: list, client_body: dict, tools, timeout: int):
    """Groq / Mistral / Zai leg. Tries every model in cfg['models'] in
    order; for each model, every configured key. A 404 skips straight to
    the next MODEL (retrying a dead model ID with a different key wastes a
    call); everything else tries the next key first, then the next model.
    All logging here uses the real provider/model name — server-side only,
    never returned to the caller (see ProviderError raised at the bottom,
    and how call_model() below converts it before it reaches a response)."""
    cfg = OPENAI_STYLE[provider]
    keys = [os.environ.get(e, d) for e, d in zip(cfg["api_key_envs"], cfg["api_key_defaults"])]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError(f"{provider}: no API keys configured")

    last_error = None
    for model_id in cfg["models"]:
        body = {"model": model_id, "messages": messages}
        _forward_sampling_params(body, client_body or {})
        body["max_tokens"] = _clamp_max_tokens(provider, (client_body or {}).get("max_tokens"))
        if tools:
            body["tools"] = tools
        if cfg.get("extra_body"):
            body.update(cfg["extra_body"])

        model_dead = False
        for key in keys:
            try:
                resp = requests.post(
                    cfg["base_url"],
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=timeout,
                )

                if resp.status_code == 404:
                    last_error = f"{provider}/{model_id}: 404 — {_error_detail(resp)}"
                    logger.warning(f"{provider}: model {model_id} unavailable, moving to next model")
                    model_dead = True
                    break
                if resp.status_code == 429:
                    last_error = f"{provider}/{model_id}: rate limited on key {_hash_key(key)} — {_error_detail(resp)}"
                    logger.info(last_error)
                    continue
                if resp.status_code in (401, 403):
                    last_error = f"{provider}/{model_id}: key {_hash_key(key)} rejected with {resp.status_code} — {_error_detail(resp)}"
                    logger.error(last_error)
                    continue
                if resp.status_code >= 500:
                    last_error = f"{provider}/{model_id}: server error {resp.status_code} — {_error_detail(resp)}"
                    logger.warning(last_error)
                    continue
                if resp.status_code >= 400:
                    last_error = f"{provider}/{model_id}: {resp.status_code} — {_error_detail(resp)}"
                    logger.error(last_error)
                    continue

                data = resp.json()
                choice = data["choices"][0]["message"]
                usage = data.get("usage", {})
                logger.info(f"{provider}/{model_id}: success, key {_hash_key(key)}")
                return {
                    "content": choice.get("content"),
                    "tool_calls": choice.get("tool_calls"),
                    "provider": provider,
                    "model": model_id,
                    "usage": usage,
                }
            except requests.RequestException as e:
                last_error = f"{provider}/{model_id}: {e}"
                logger.error(last_error)
                continue

        if model_dead:
            continue

    raise ProviderError(last_error or f"{provider}: all models/keys failed")


def _messages_to_gemini_contents(messages: list):
    """Text-only. A legacy content-list shape is accepted defensively (text
    blocks concatenated) so an older client doesn't hard-fail."""
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

        if isinstance(content, list):
            text = "\n".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = content or ""

        if prefix and role == "user":
            text = prefix + text
            prefix = ""

        contents.append({"role": role, "parts": [{"text": text}]})

    return contents


def _call_gemini(messages: list, timeout: int, client_body: dict = None):
    """Gemini leg. Tries every model in GEMINI_MODELS in order, each across
    every configured key — same dead-model-skips-remaining-keys logic as
    _call_openai_style. All logging uses the real model name, server-side
    only."""
    keys = GEMINI_KEYS
    if not keys:
        raise ProviderError("gemini: no API keys configured (set GEMINI_API_KEY or GEMINI_API_KEY_1/2)")

    contents = _messages_to_gemini_contents(messages)

    gen_cfg = {"maxOutputTokens": _clamp_max_tokens("gemini", (client_body or {}).get("max_tokens"))}
    if client_body:
        for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK")):
            v = client_body.get(src)
            if v is not None:
                gen_cfg[dst] = v

    body = {"contents": contents, "generationConfig": gen_cfg}

    last_error = None
    for model in GEMINI_MODELS:
        model_dead = False
        for key in keys:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                resp = requests.post(url, json=body, timeout=timeout)

                if resp.status_code == 404:
                    last_error = f"gemini/{model}: 404 — {_error_detail(resp)}"
                    logger.warning(f"gemini: model {model} unavailable, moving to next model")
                    model_dead = True
                    break
                if resp.status_code == 429:
                    last_error = f"gemini/{model}: rate limited on key {_hash_key(key)} — {_error_detail(resp)}"
                    logger.warning(last_error)
                    continue
                if resp.status_code in (401, 403):
                    last_error = f"gemini/{model}: key {_hash_key(key)} rejected with {resp.status_code} — {_error_detail(resp)}"
                    logger.error(last_error)
                    continue
                if resp.status_code >= 500:
                    last_error = f"gemini/{model}: server error {resp.status_code} — {_error_detail(resp)}"
                    logger.warning(last_error)
                    continue
                if resp.status_code >= 400:
                    last_error = f"gemini/{model}: {resp.status_code} — {_error_detail(resp)}"
                    logger.error(last_error)
                    continue

                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    block_reason = (data.get("promptFeedback") or {}).get("blockReason")
                    last_error = f"gemini/{model}: no candidates returned (blockReason={block_reason})"
                    logger.error(last_error)
                    continue

                candidate = candidates[0]
                parts = ((candidate.get("content") or {}).get("parts")) or []
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))

                if not text:
                    finish_reason = candidate.get("finishReason")
                    last_error = f"gemini/{model}: empty response (finishReason={finish_reason})"
                    logger.error(last_error)
                    continue

                usage = data.get("usageMetadata", {})
                logger.info(f"gemini/{model}: success, key {_hash_key(key)}")
                return {"content": text, "tool_calls": None, "provider": "gemini", "model": model, "usage": usage}
            except requests.RequestException as e:
                last_error = f"gemini/{model}: {e}"
                logger.error(last_error)
                continue

        if model_dead:
            continue

    raise ProviderError(last_error or "gemini: all models/keys failed")


def call_model(messages: list, client_body: dict = None, tools=None, timeout: int = 30, order=None, tracer: "Tracer" = None):
    """Route to a provider. Tries each in order (DEFAULT_ORDER, or `order`)
    until one succeeds — each provider tries every model it's configured
    with, across every key, before giving up. No cooldowns: a dead
    key/model/provider is skipped this request and tried fresh next time.

    Adds a GENERIC (no real names) route event to `tracer` per attempt, so
    the customer-visible debug trace shows that fallback happened without
    revealing what it fell back to. The real detail goes to logger only."""
    errors = []
    names = ["gemini"] if GEMINI_ONLY else (order if order is not None else DEFAULT_ORDER)

    for i, name in enumerate(names):
        if tracer:
            tracer.add("route_attempt", f"trying route {i + 1} of {len(names)}")
        try:
            if name == "gemini":
                result = _call_gemini(messages, timeout, client_body=client_body)
            else:
                result = _call_openai_style(name, messages, client_body, tools, timeout)
            if tracer:
                tracer.add("route_ok", f"route {i + 1} answered")
            return result
        except (ProviderError, requests.RequestException) as e:
            errors.append(f"{name}: {e}")
            logger.error(f"Provider {name} failed, rerouting: {e}")
            if tracer:
                tracer.add("route_failed", f"route {i + 1} failed, rerouting")
            continue

    # Full detail (real provider/model names) goes to the server log only.
    logger.error("All backends failed -> " + " | ".join(errors))
    raise ProviderError("Sentry-1 is temporarily unable to process this request. Please try again in a moment.")


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
    retry_body["_json_retry"] = True

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


def web_search(query: str, max_results: int = 5, search_depth: str = "basic",
               include_images: bool = True):
    """Tavily search with images. Returns {"results": [...], "images": [...]}."""
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


SEARCH_TAG_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE | re.DOTALL)
FETCH_TAG_RE = re.compile(r"\[FETCH:\s*(https?://[^\s\]]+)\]", re.IGNORECASE)


def _parse_search_tag(raw: str):
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


def run_agentic_completion(messages: list, client_body: dict = None, web_search_enabled: bool = True, tracer: "Tracer" = None):
    """Model -> [SEARCH: ...] / [FETCH: ...] tags -> real results injected ->
    model again, until no tags appear or MAX_TOOL_ITERATIONS is hit."""
    working = list(messages)
    degraded = False
    collected_images = []
    default_max_results = int(((client_body or {}).get("web_search") or {}).get("max_results") or 5)
    tr = tracer or Tracer()

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            result = call_model(working, client_body=client_body, tools=None, tracer=tr)
        except ProviderError:
            if not web_search_enabled or degraded:
                raise
            logger.warning("Model call failed; falling back to Tavily pre-search")
            degraded = True
            tr.add("fallback", "answering from a live search instead")

            last_user = ""
            for m in reversed(working):
                if m.get("role") == "user":
                    c = m.get("content")
                    last_user = c if isinstance(c, str) else ""
                    break

            if last_user:
                working, imgs = _inject_tavily_results(working, last_user)
                collected_images.extend(imgs)

            result = call_model(working, client_body=client_body, tools=None, tracer=tr)

        content = result.get("content") or ""
        search_tags = _extract_search_tags(content) if web_search_enabled else []
        fetch_tags = _extract_fetch_tags(content) if web_search_enabled else []

        if not search_tags and not fetch_tags:
            out = {
                "content": content,
                "images": collected_images,
                "trace": tr.events,
            }
            if degraded:
                out["web_search_degraded"] = True
            return out

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
                tr.add("search", f"searched: {query}")
            except Exception as e:
                tool_output = f'Search for "{query}" failed: {e}'
                tr.add("search_failed", f"search failed: {query}")
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
                tr.add("fetch", f"fetched: {url}")
            except Exception as e:
                tool_output = f'Failed to fetch {url}: {e}'
                tr.add("fetch_failed", f"fetch failed: {url}")
            working.append({
                "role": "system",
                "content": (
                    f"{tool_output}\n\n"
                    "Use this page content to answer. You may [FETCH: another-url] if needed."
                ),
            })

    out = {
        "content": "I wasn't able to finish that within the search-iteration limit.",
        "images": collected_images,
        "trace": tr.events,
    }
    if degraded:
        out["web_search_degraded"] = True
    return out


# =========================================================================
# SYSTEM PROMPT — web search + JSON mode
# =========================================================================

def build_system_prompt(body: dict, is_json_retry: bool = False) -> str:
    web_cfg = body.get("web_search") or {}
    web_enabled = bool(web_cfg.get("enabled"))
    default_max_results = int(web_cfg.get("max_results") or 5)
    response_format = body.get("response_format")
    json_enabled = bool(response_format)

    lines = ["You are Sentry 1, an AI assistant accessed via the Fluid Intelligence API."]
    lines.append(
        "Current date and time (UTC): %s. This is live, authoritative data from the "
        "server clock — trust it completely over anything you recall from training, "
        "and use it whenever asked about today's date, the current time, or how long "
        "ago/until something is." % utcnow().strftime("%A, %B %d, %Y, %H:%M UTC")
    )
    lines.append(
        "If asked what model, provider, or company is behind you, answer simply that "
        "you are Sentry 1. Don't speculate about or name any underlying model, vendor, "
        "or infrastructure — you don't have that information to share."
    )

    lines.append("")
    lines.append("EXECUTABLE JAVASCRIPT — the playground can run JavaScript you write:")
    lines.append('- Write JavaScript inside a ```js fenced code block and the playground renders a "run" button next to it.')
    lines.append("- The code executes in a sandboxed iframe and console.log() output is shown to the user under the code block.")
    lines.append("- Use this for calculations, algorithms, data transformations, simulations, or small demos — anything where running code is clearer than describing it.")
    lines.append("- Code must be self-contained. You may load external libraries only via a CDN <script> tag you write yourself inside the code.")
    lines.append("- Do NOT use executable code blocks for trivial one-liners or for code the user only wants to read/copy.")

    lines.append("")
    lines.append("IMAGE EMBEDDING — this API is text-only input, but you may still embed images IN YOUR REPLY using markdown ![description](url), when a web search turned up a relevant image URL:")
    lines.append("- The playground automatically scales every image to fit the chat width, so never worry about pixel dimensions.")
    lines.append("- Only embed an image when it genuinely helps the answer, and always write a meaningful description in the alt text.")
    lines.append("- If a search returned images, prefer the most relevant one rather than embedding many.")

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
        lines.append("- Never output anything shaped like [SEARCH: ...], [FETCH: ...], or a field describing a search you're about to run.")

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
        # Internal-only: this endpoint is for your own uptime checks, not
        # public docs. Still no real provider/model names, on principle.
        return jsonify(
            ok=True, version=APP_VERSION,
            max_input_tokens=MAX_INPUT_TOKENS,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            routes_configured=len(DEFAULT_ORDER),
        )

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

    # ---------- completions (shared core, two auth front-ends) ----------

    def completion_core(body: dict, api_key_id: int):
        """Returns (response_dict, status, count). Response never includes
        real provider/model names — see Tracer and call_model() docstrings."""
        tr = Tracer()
        messages = body.get("messages") or []
        if not messages:
            return {"error": {"message": "messages required"}, "_debug_trace": tr.events}, 400, None

        is_json_retry = bool(body.get("_json_retry"))
        system_content = build_system_prompt(body, is_json_retry=is_json_retry)
        messages.insert(0, {"role": "system", "content": system_content})
        tr.add("routing", "request received")

        tokens_in = estimate_tokens(messages)
        if tokens_in > MAX_INPUT_TOKENS:
            tr.add("error", "input exceeds token limit")
            return {"error": {"message": f"request exceeds the {MAX_INPUT_TOKENS}-token limit"}, "_debug_trace": tr.events}, 413, None

        count = check_and_increment_rate_limit(api_key_id, tokens_in=tokens_in)
        set_rate_limit_headers(count)
        if count > DAILY_REQUEST_CAP:
            resp = jsonify(error={"message": "daily request cap exceeded"}, _debug_trace=tr.events)
            resp.status_code = 429
            resp.headers["Retry-After"] = str(seconds_until_midnight_utc())
            return None, 429, count

        web_search_enabled = bool((body.get("web_search") or {}).get("enabled"))

        try:
            result = run_agentic_completion(messages, client_body=body, web_search_enabled=web_search_enabled, tracer=tr)
        except ProviderError as e:
            # e's message is already the generic customer-facing string —
            # call_model() logged the real per-provider detail already.
            tr.add("error", "request failed")
            return {"error": {"message": str(e)}, "_debug_trace": tr.events}, 502, count

        content = result["content"]
        response_format = body.get("response_format")
        if response_format:
            try:
                content = validate_and_fix_json(content, response_format, messages, body, api_key_id)
            except ProviderError as e:
                tr.add("error", "JSON validation failed")
                return {"error": {"message": str(e)}, "_debug_trace": tr.events}, 502, count

        tokens_out = estimate_tokens([{"content": content}])
        record_tokens_out(api_key_id, tokens_out)
        tr.add("routing", "request complete")

        payload = {
            "id": "chatcmpl-sentry1",
            "object": "chat.completion",
            "model": "sentry-1",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}}
            ],
            "usage": {"prompt_tokens_est": tokens_in, "completion_tokens_est": tokens_out},
            "_debug_trace": result.get("trace", tr.events),
        }
        if result.get("images"):
            payload["_search_images"] = result["images"]
        if result.get("web_search_degraded"):
            payload["_web_search_degraded"] = True
        return payload, 200, count

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
        return jsonify(payload)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
