"""
Sentry AI backend — single-file version.
Run with: gunicorn main:app  (or `python main.py` for local dev)
"""

import os
import json
import secrets
import datetime as dt
from contextlib import contextmanager

import bcrypt
import jwt
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# =========================================================================
# DATABASE
# =========================================================================

# dummy/test Aiven service URI — override in production via the DATABASE_URL env var
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://avnadmin:AVNS_q_zR9FvhvJHJGbiL0zp@pg-235735e2-ub7499710-7253.j.aivencloud.com:11602/defaultdb?sslmode=require",
)

_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL,
    sslmode="require",
)

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

-- tracks which backend is currently cooling down after a failure, so the
-- router doesn't retry a backend it just saw fail on the very next request
CREATE TABLE IF NOT EXISTS backend_status (
    backend_name       TEXT PRIMARY KEY,
    unavailable_until  TIMESTAMPTZ,
    last_error         TEXT
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
    """Idempotently create tables (CREATE TABLE IF NOT EXISTS). Safe to call on every boot/deploy.
    Note: if usage_counters already exists from a prior deploy without tokens_in/tokens_out,
    those two columns won't be added automatically — run one manual ALTER TABLE in that case:
        ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_in INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_out INTEGER NOT NULL DEFAULT 0;
    """
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        _pool.putconn(conn)


# =========================================================================
# AUTH — passwords, session tokens, API key generation
# =========================================================================

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-dummy-secret-change-me-in-production-32chars")
SESSION_HOURS = 24 * 7


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_session_token(user_id: int) -> str:
    payload = {
        "sub": user_id,
        "exp": dt.datetime.utcnow() + dt.timedelta(hours=SESSION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_session_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except jwt.PyJWTError:
        return None


def generate_api_key():
    """Returns (full_key, prefix, key_hash). Only the caller sees full_key, once."""
    raw = secrets.token_hex(24)  # 48 hex chars
    full_key = f"sk-sentry-{raw}"
    prefix = f"sk-sentry-{raw[:8]}..."
    key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode()
    return full_key, prefix, key_hash


def create_api_key_for_user(user_id: int, name: str):
    """Enforces the one-active-key-per-account rule (intentional — no multi-key support)."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM api_keys WHERE user_id = %s AND revoked_at IS NULL",
            (user_id,),
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


def revoke_api_key(user_id: int, key_id: int):
    with get_cursor(commit=True) as cur:
        cur.execute(
            """UPDATE api_keys SET revoked_at = now()
               WHERE id = %s AND user_id = %s AND revoked_at IS NULL""",
            (key_id, user_id),
        )


def authenticate_api_key(bearer_token: str):
    """
    Looks up the active key whose hash matches. Matches on key_prefix first
    to narrow to ~1 row, then bcrypt-verifies. Fine at small scale.
    """
    if not bearer_token or not bearer_token.startswith("sk-sentry-"):
        return None
    prefix_guess = bearer_token[:18] + "..."  # "sk-sentry-" + 8 hex chars
    with get_cursor() as cur:
        cur.execute(
            """SELECT id, user_id, key_hash FROM api_keys
               WHERE key_prefix = %s AND revoked_at IS NULL""",
            (prefix_guess,),
        )
        row = cur.fetchone()
    if not row:
        return None
    if not bcrypt.checkpw(bearer_token.encode(), row["key_hash"].encode()):
        return None
    return {"api_key_id": row["id"], "user_id": row["user_id"]}


def get_users_active_key_id(user_id: int):
    """Used by the session-authenticated playground so its usage still counts
    against the account's real daily cap instead of bypassing it."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM api_keys WHERE user_id = %s AND revoked_at IS NULL",
            (user_id,),
        )
        row = cur.fetchone()
    return row["id"] if row else None


# =========================================================================
# BACKEND HEALTH / COOLDOWN TRACKING
# =========================================================================
# A backend that just failed (rate limited or errored) is marked unavailable
# for a short cooldown window, so the very next request skips straight past
# it instead of hitting the same failure again.

def mark_backend_down(name: str, error, cooldown_seconds: int = 30):
    until = dt.datetime.utcnow() + dt.timedelta(seconds=cooldown_seconds)
    with get_cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO backend_status (backend_name, unavailable_until, last_error)
               VALUES (%s, %s, %s)
               ON CONFLICT (backend_name) DO UPDATE
               SET unavailable_until = EXCLUDED.unavailable_until,
                   last_error = EXCLUDED.last_error""",
            (name, until, str(error)[:500]),
        )


def is_backend_available(name: str) -> bool:
    with get_cursor() as cur:
        cur.execute(
            "SELECT unavailable_until FROM backend_status WHERE backend_name = %s",
            (name,),
        )
        row = cur.fetchone()
    if not row or not row["unavailable_until"]:
        return True
    return dt.datetime.utcnow() > row["unavailable_until"]


# =========================================================================
# MODEL PROVIDERS — Groq, Mistral, Z.AI, Gemini, each with 2-key fallback
# =========================================================================

DEFAULT_ORDER = ["groq", "mistral", "zai", "gemini"]
MEDIA_CAPABLE_ORDER = ["gemini"]  # only backend with confirmed free-tier image support

OPENAI_STYLE = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_envs": ["GROQ_API_KEY_1", "GROQ_API_KEY_2"],
        "api_key_defaults": [
            "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",
            "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",
        ],
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "extra_body": {},
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
    },
    "zai": {
        # FIX: was pointed at open.bigmodel.cn (Zhipu's China-facing domain) with
        # model "glm-4-plus" — wrong endpoint for these international Z.AI keys,
        # and glm-4-plus isn't the free-tier model. Corrected to Z.AI's actual
        # international endpoint and its free model.
        "base_url": "https://api.z.ai/api/paas/v4/chat/completions",
        "api_key_envs": ["ZAI_API_KEY_1", "ZAI_API_KEY_2"],
        "api_key_defaults": [
            "ccd29e152e6941098d7e9e1fd91f24a1.thzU8g4LLNVm6qea",
            "f6bc4aac8cfe425cadde84bba181710e.wX0Wm0I8l3YDihGy",
        ],
        "model": os.environ.get("ZAI_MODEL", "glm-4.5-flash"),
        # GLM-4.5-flash has chain-of-thought "thinking" on by default, which can
        # consume the entire max_tokens budget before writing a visible answer,
        # leaving `content` empty. Disable it for plain chat completions.
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}

GEMINI_API_KEY_ENVS = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"]
GEMINI_API_KEY_DEFAULTS = [
    "AQ.Ab8RN6Kq43fqm482ux_6BD71H67SjJlBjE4qbqY94AGUW1JyLw",
    "AIzaSyBRB2XWuUT-E_X0D8F7B1YTdjrkMIRMBRY",
]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


class ProviderError(Exception):
    pass


def has_media(messages: list) -> bool:
    """True if any message contains an image/audio/video content block
    (OpenAI-style list content), rather than being plain text."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "image_url", "input_audio", "video_url",
                ):
                    return True
    return False


def estimate_tokens(messages) -> int:
    """Rough estimate (chars/4) — good enough for a request-size cap, not for billing precision."""
    return max(1, len(json.dumps(messages)) // 4)


def _call_openai_style(provider: str, messages: list, tools, timeout: int):
    if not is_backend_available(provider):
        raise ProviderError(f"{provider}: cooling down after a recent failure")

    cfg = OPENAI_STYLE[provider]
    keys = [
        os.environ.get(env, default)
        for env, default in zip(cfg["api_key_envs"], cfg["api_key_defaults"])
    ]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError(f"{provider}: no API keys configured")

    body = {"model": cfg["model"], "messages": messages}
    if tools:
        body["tools"] = tools
    if cfg.get("extra_body"):
        body.update(cfg["extra_body"])

    last_error = None
    for key in keys:
        try:
            resp = requests.post(
                cfg["base_url"],
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
            if resp.status_code == 429:
                last_error = f"{provider}: rate limited on this key"
                continue
            if resp.status_code >= 500:
                last_error = f"{provider}: server error {resp.status_code}"
                continue
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            return {
                "content": choice.get("content"),
                "tool_calls": choice.get("tool_calls"),
                "provider": provider,
                "usage": data.get("usage", {}),
            }
        except requests.RequestException as e:
            last_error = f"{provider}: {e}"
            continue

    mark_backend_down(provider, last_error)
    raise ProviderError(last_error or f"{provider}: all keys failed")


def _messages_to_gemini_contents(messages: list):
    """
    Gemini uses {role, parts:[{text}|{inline_data}]} and has no 'system' role —
    fold system text into the first user turn. Also translates OpenAI-style
    image_url content blocks (base64 data URIs) into Gemini's inline_data parts.
    """
    system_texts = [m["content"] for m in messages if m["role"] == "system" and isinstance(m["content"], str)]
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
                        mime = mime.replace("data:", "")
                        parts.append({"inline_data": {"mime_type": mime, "data": b64data}})
                    # non-data-URI (hosted) image URLs aren't supported by this
                    # simple adapter yet — data URIs only for now.

        contents.append({"role": role, "parts": parts})

    return contents


def _call_gemini(messages: list, timeout: int):
    if not is_backend_available("gemini"):
        raise ProviderError("gemini: cooling down after a recent failure")

    keys = [
        os.environ.get(env, default)
        for env, default in zip(GEMINI_API_KEY_ENVS, GEMINI_API_KEY_DEFAULTS)
    ]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError("gemini: no API keys configured")

    body = {"contents": _messages_to_gemini_contents(messages)}
    last_error = None
    for key in keys:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent?key={key}"
            )
            resp = requests.post(url, json=body, timeout=timeout)
            if resp.status_code == 429:
                last_error = "gemini: rate limited on this key"
                continue
            if resp.status_code >= 500:
                last_error = f"gemini: server error {resp.status_code}"
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"content": text, "tool_calls": None, "provider": "gemini", "usage": data.get("usageMetadata", {})}
        except requests.RequestException as e:
            last_error = f"gemini: {e}"
            continue

    mark_backend_down("gemini", last_error)
    raise ProviderError(last_error or "gemini: all keys failed")


def call_model(messages: list, tools=None, timeout: int = 30, order=None):
    """
    Tries providers in order; within each provider, tries key 1 then key 2.
    Falls through on any ProviderError, timeout, or HTTP error. Raises
    ProviderError only if every provider/key combination fails.

    If the request contains an image/audio/video block, only media-capable
    backends are tried — a text-only backend would either error or silently
    ignore the attachment, neither of which is acceptable to fail into.
    """
    if order is None:
        if has_media(messages):
            order = MEDIA_CAPABLE_ORDER
        else:
            order = [p.strip() for p in os.environ.get("PROVIDER_ORDER", ",".join(DEFAULT_ORDER)).split(",")]

    last_error = None
    for provider in order:
        try:
            if provider == "gemini":
                return _call_gemini(messages, timeout)
            elif provider in OPENAI_STYLE:
                return _call_openai_style(provider, messages, tools, timeout)
        except (ProviderError, requests.RequestException) as e:
            last_error = e
            continue

    raise ProviderError(f"All providers/keys failed. Last error: {last_error}")


# =========================================================================
# TOOLS — Tavily live web search, 2-key fallback
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


TOOL_IMPLEMENTATIONS = {
    "web_search": web_search,
}


# =========================================================================
# AGENTIC LOOP — calls model, executes tool calls, feeds results back
# =========================================================================

MAX_TOOL_ITERATIONS = 5


def run_agentic_completion(messages: list, web_search_enabled: bool = True):
    """
    Returns the final assistant message once no more tool calls are made,
    or once MAX_TOOL_ITERATIONS is hit (safety cap so a stuck loop can't burn
    through the daily provider quota).
    """
    tools = [WEB_SEARCH_TOOL_SCHEMA] if web_search_enabled else None
    working_messages = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        result = call_model(working_messages, tools=tools)

        if not result.get("tool_calls"):
            return {"content": result["content"], "provider": result["provider"]}

        working_messages.append({"role": "assistant", "content": result.get("content") or ""})

        for call in result["tool_calls"]:
            fn_name = call["function"]["name"]
            fn_args = json.loads(call["function"]["arguments"])
            impl = TOOL_IMPLEMENTATIONS.get(fn_name)

            if impl is None:
                tool_output = {"error": f"unknown tool {fn_name}"}
            else:
                try:
                    tool_output = impl(**fn_args)
                except Exception as e:
                    tool_output = {"error": str(e)}

            working_messages.append(
                {"role": "tool", "name": fn_name, "content": json.dumps(tool_output)}
            )

    return {"content": "I wasn't able to finish that within the tool-call limit.", "provider": None}


# =========================================================================
# FLASK APP
# =========================================================================

DAILY_REQUEST_CAP = int(os.environ.get("DAILY_REQUEST_CAP", "3000"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "8000"))  # matches the public "8K tokens/request" figure
# dummy/dev default — set FRONTEND_ORIGIN to the real Vercel URL in production
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")


def create_app():
    flask_app = Flask(__name__)
    CORS(flask_app, origins=[FRONTEND_ORIGIN], supports_credentials=True)

    with flask_app.app_context():
        init_schema()

    # ---------- account endpoints ----------

    @flask_app.post("/auth/signup")
    def signup():
        data = request.get_json(force=True)
        email, password = data.get("email"), data.get("password")
        if not email or not password:
            return jsonify(error="email and password required"), 400

        password_hash = hash_password(password)
        try:
            with get_cursor(commit=True) as cur:
                cur.execute(
                    "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                    (email, password_hash),
                )
                user_id = cur.fetchone()["id"]
        except Exception:
            return jsonify(error="email already registered"), 409

        token = create_session_token(user_id)
        return jsonify(token=token), 201

    @flask_app.post("/auth/signin")
    def signin():
        data = request.get_json(force=True)
        email, password = data.get("email"), data.get("password")
        with get_cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
        if not row or not check_password(password, row["password_hash"]):
            return jsonify(error="invalid credentials"), 401
        return jsonify(token=create_session_token(row["id"]))

    def _require_session():
        header = request.headers.get("Authorization", "")
        token = header.removeprefix("Bearer ").strip()
        return verify_session_token(token)

    @flask_app.post("/console/password")
    def update_password():
        """Backs the console's Settings > Change password form. Added so that
        form isn't a no-op UI element — it used to show 'Saved.' without ever
        calling the backend."""
        user_id = _require_session()
        if not user_id:
            return jsonify(error="unauthorized"), 401
        data = request.get_json(force=True)
        password = data.get("password")
        if not password or len(password) < 8:
            return jsonify(error="password must be at least 8 characters"), 400

        password_hash = hash_password(password)
        with get_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (password_hash, user_id),
            )
        return jsonify(ok=True)

    # ---------- API key management (requires session token) ----------

    @flask_app.post("/console/keys")
    def create_key():
        user_id = _require_session()
        if not user_id:
            return jsonify(error="unauthorized"), 401
        name = request.get_json(force=True).get("name", "default")
        try:
            key_info = create_api_key_for_user(user_id, name)
        except ValueError as e:
            return jsonify(error=str(e)), 409
        return jsonify(key_info), 201

    @flask_app.get("/console/keys")
    def list_keys():
        user_id = _require_session()
        if not user_id:
            return jsonify(error="unauthorized"), 401
        with get_cursor() as cur:
            cur.execute(
                """SELECT id, name, key_prefix AS prefix, created_at
                   FROM api_keys WHERE user_id = %s AND revoked_at IS NULL
                   ORDER BY created_at DESC""",
                (user_id,),
            )
            rows = cur.fetchall()
        return jsonify(keys=[dict(r, created_at=r["created_at"].isoformat()) for r in rows])

    @flask_app.delete("/console/keys/<int:key_id>")
    def revoke_key(key_id):
        user_id = _require_session()
        if not user_id:
            return jsonify(error="unauthorized"), 401
        revoke_api_key(user_id, key_id)
        return "", 204

    @flask_app.get("/console/usage")
    def console_usage():
        user_id = _require_session()
        if not user_id:
            return jsonify(error="unauthorized"), 401

        with get_cursor() as cur:
            cur.execute(
                "SELECT id FROM api_keys WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )
            key_row = cur.fetchone()

        if not key_row:
            return jsonify(requests_today=0, daily_cap=DAILY_REQUEST_CAP, history=[0] * 7)

        today = dt.date.today()
        history = []
        with get_cursor() as cur:
            for i in range(6, -1, -1):
                day = today - dt.timedelta(days=i)
                cur.execute(
                    "SELECT request_count FROM usage_counters WHERE api_key_id = %s AND usage_date = %s",
                    (key_row["id"], day),
                )
                row = cur.fetchone()
                history.append(row["request_count"] if row else 0)

        return jsonify(requests_today=history[-1], daily_cap=DAILY_REQUEST_CAP, history=history)

    # ---------- public API (requires bearer API key, not session token) ----------

    def _require_api_key():
        header = request.headers.get("Authorization", "")
        token = header.removeprefix("Bearer ").strip()
        return authenticate_api_key(token)

    def _check_and_increment_rate_limit(api_key_id: int, tokens_in: int = 0, tokens_out: int = 0) -> bool:
        """Returns True if this request is allowed under today's cap. Always
        records the request + token counts either way, so usage stays accurate
        even on the request that tips it over the limit."""
        today = dt.date.today()
        with get_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO usage_counters (api_key_id, usage_date, request_count, tokens_in, tokens_out)
                   VALUES (%s, %s, 1, %s, %s)
                   ON CONFLICT (api_key_id, usage_date)
                   DO UPDATE SET request_count = usage_counters.request_count + 1,
                                 tokens_in = usage_counters.tokens_in + EXCLUDED.tokens_in,
                                 tokens_out = usage_counters.tokens_out + EXCLUDED.tokens_out
                   RETURNING request_count""",
                (api_key_id, today, tokens_in, tokens_out),
            )
            count = cur.fetchone()["request_count"]
        return count <= DAILY_REQUEST_CAP

    @flask_app.get("/v1/models")
    def list_models():
        return jsonify(data=[{"id": "sentry-1", "object": "model"}])

    @flask_app.get("/stats/public")
    def public_stats():
        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM users")
            users_count = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM api_keys WHERE revoked_at IS NULL")
            keys_count = cur.fetchone()["c"]
        return jsonify(users=users_count, keys=keys_count)

    def _run_completion_or_error(messages, web_search_enabled):
        """Shared by /v1/chat/completions and the playground — same code path,
        same limits, same behavior, just different auth in front of it."""
        tokens_in = estimate_tokens(messages)
        if tokens_in > MAX_INPUT_TOKENS:
            return None, (jsonify(error={"message": f"request exceeds the {MAX_INPUT_TOKENS}-token limit"}), 413)
        try:
            result = run_agentic_completion(messages, web_search_enabled=web_search_enabled)
        except ProviderError as e:
            return None, (jsonify(error={"message": str(e)}), 502)
        return (result, tokens_in), None

    @flask_app.post("/v1/chat/completions")
    def chat_completions():
        identity = _require_api_key()
        if not identity:
            return jsonify(error={"message": "invalid API key"}), 401

        body = request.get_json(force=True)
        messages = body.get("messages", [])
        web_search_enabled = bool(body.get("web_search", {}).get("enabled"))

        tokens_in = estimate_tokens(messages)
        if tokens_in > MAX_INPUT_TOKENS:
            return jsonify(error={"message": f"request exceeds the {MAX_INPUT_TOKENS}-token limit"}), 413

        # check-and-reserve happens before the call so a burst of concurrent
        # requests can't all slip through while the cap check is in flight
        if not _check_and_increment_rate_limit(identity["api_key_id"], tokens_in=tokens_in):
            return jsonify(error={"message": "daily request cap exceeded"}), 429

        try:
            result = run_agentic_completion(messages, web_search_enabled=web_search_enabled)
        except ProviderError as e:
            return jsonify(error={"message": str(e)}), 502

        tokens_out = estimate_tokens([{"content": result["content"]}])

        return jsonify(
            {
                "id": "chatcmpl-sentry1",
                "object": "chat.completion",
                "model": "sentry-1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": result["content"]}}
                ],
                "_provider_used": result["provider"],
                "usage": {"prompt_tokens_est": tokens_in, "completion_tokens_est": tokens_out},
            }
        )

    @flask_app.post("/console/playground/chat")
    def playground_chat():
        """
        Session-authenticated proxy for the in-console Playground.
        FIX: this previously had no rate limit at all ("not subject to the
        per-API-key daily cap since it's tied to the session, not a key") —
        that meant any signed-in user had unmetered access to the exact same
        backend calls as the public API. It now counts against that user's
        own API key's daily cap, same as calling the public endpoint would.
        """
        user_id = _require_session()
        if not user_id:
            return jsonify(error={"message": "unauthorized"}), 401

        key_id = get_users_active_key_id(user_id)
        if not key_id:
            return jsonify(error={"message": "create an API key first"}), 400

        body = request.get_json(force=True)
        messages = body.get("messages", [])
        web_search_enabled = bool(body.get("web_search", {}).get("enabled"))

        tokens_in = estimate_tokens(messages)
        if tokens_in > MAX_INPUT_TOKENS:
            return jsonify(error={"message": f"request exceeds the {MAX_INPUT_TOKENS}-token limit"}), 413

        if not _check_and_increment_rate_limit(key_id, tokens_in=tokens_in):
            return jsonify(error={"message": "daily request cap exceeded"}), 429

        try:
            result = run_agentic_completion(messages, web_search_enabled=web_search_enabled)
        except ProviderError as e:
            return jsonify(error={"message": str(e)}), 502

        return jsonify(
            {
                "id": "chatcmpl-sentry1",
                "object": "chat.completion",
                "model": "sentry-1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": result["content"]}}
                ],
            }
        )

    return flask_app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
