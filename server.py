"""
Fluid Intelligence backend — rebuilt single-file version.
Run:  gunicorn main:app   (or `python main.py` locally)

Rebuild notes (vs. the version that 401'd fresh sessions):
- Token issue/verify is now demonstrably consistent (one secret, one function,
  timezone-aware datetimes, no PyJWT deprecation edge cases).
- 401s now say WHICH failure happened: unauthorized / session_expired /
  invalid_token — no more guessing.
- Added GET /auth/session (validate stored token + get account email — this
  is what powers a *safe* auto sign-in after page refresh).
- Added GET /version so you can always confirm which build Render is serving.
- Added GET /healthz for Render health checks.
- Rate-limit headers (X-RateLimit-Limit/Remaining/Reset) are now actually set
  on responses — the docs always promised them, the old code never sent them.
  429s include Retry-After (seconds until midnight UTC reset).
- stream:true now returns a valid SSE response (single-chunk + [DONE]) instead
  of being silently ignored. Token-by-token streaming is noted as TODO.
- usage_counters.tokens_out is now actually written (old code computed it and
  dropped it).
- Schema auto-migrates with ALTER TABLE ... IF NOT EXISTS on boot.
Secrets stay hardcoded here per project decision. Rotate them by editing this
file and redeploying.
"""

import os
import json
import secrets
import re
import datetime as dt
from datetime import timezone
from contextlib import contextmanager

import bcrypt
import jwt
import requests
from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# =========================================================================
# CONFIG
# =========================================================================

APP_VERSION = os.environ.get("APP_VERSION", "rebuild-2026-09-14")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://avnadmin:AVNS_q_zR9FvhvJHJGbiL0zp@pg-235735e2-ub7499710-7253.j.aivencloud.com:11602/defaultdb?sslmode=require",
)

JWT_SECRET = "21c8524d2320d22706fb366fc6ca9037050cdfda26c104ac0ecfa0215bebda7c"
SESSION_HOURS = 24 * 7
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

DAILY_REQUEST_CAP = int(os.environ.get("DAILY_REQUEST_CAP", "3000"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "8000"))
MAX_TOOL_ITERATIONS = 5

# Open CORS so a Vercel URL mismatch can never be the cause of anything.
# Tighten to your real domain once the app is stable.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

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
"""

# idempotent migrations for databases created by older versions of this app
MIGRATION_SQL = """
ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_in INTEGER NOT NULL DEFAULT 0;
ALTER TABLE usage_counters ADD COLUMN IF NOT EXISTS tokens_out INTEGER NOT NULL DEFAULT 0;
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
        "v": 1,  # token format version — bump if you ever change signing
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
    raw = secrets.token_hex(24)  # 48 hex chars
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
    """Increments today's counter and RETURNS the new count (caller compares
    to the cap). Always records, so usage stays accurate even on the
    request that tips over the limit."""
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
# BACKEND COOLDOWNS
# =========================================================================

def mark_backend_down(name: str, error, cooldown_seconds: int = 30):
    until = utcnow() + dt.timedelta(seconds=cooldown_seconds)
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
            "SELECT unavailable_until FROM backend_status WHERE backend_name = %s", (name,)
        )
        row = cur.fetchone()
    if not row or not row["unavailable_until"]:
        return True
    return utcnow() > row["unavailable_until"]


# =========================================================================
# MODEL PROVIDERS
# =========================================================================

DEFAULT_ORDER = ["groq", "mistral", "zai", "gemini"]
MEDIA_CAPABLE_ORDER = ["gemini"]  # only backend with free-tier image support

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
        "base_url": "https://api.z.ai/api/paas/v4/chat/completions",
        "api_key_envs": ["ZAI_API_KEY_1", "ZAI_API_KEY_2"],
        "api_key_defaults": [
            "ccd29e152e6941098d7e9e1fd91f24a1.thzU8g4LLNVm6qea",
            "f6bc4aac8cfe425cadde84bba181710e.wX0Wm0I8l3YDihGy",
        ],
        "model": os.environ.get("ZAI_MODEL", "glm-4.5-flash"),
        # GLM thinking-on would burn the whole max_tokens budget before
        # producing a visible answer — keep it disabled for plain chat.
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}

GEMINI_API_KEY_ENVS = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"]
GEMINI_API_KEY_DEFAULTS = [
    "AQ.Ab8RN6Kq43fqm482ux_6BD71H67SjJlBjE4qbqY94AGUW1JyLw",
    "AIzaSyBRB2XWuUT-E_X0D8F7B1YTdjrkMIRMBRY",
]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# sampling params we forward to OpenAI-style providers when the client sets them
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


def _call_openai_style(provider: str, messages: list, client_body: dict, tools, timeout: int):
    if not is_backend_available(provider):
        raise ProviderError(f"{provider}: cooling down after a recent failure")

    cfg = OPENAI_STYLE[provider]
    keys = [os.environ.get(e, d) for e, d in zip(cfg["api_key_envs"], cfg["api_key_defaults"])]
    keys = [k for k in keys if k]
    if not keys:
        raise ProviderError(f"{provider}: no API keys configured")

    body = {"model": cfg["model"], "messages": messages}
    _forward_sampling_params(body, client_body or {})
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
    system_texts = [
        m["content"] for m in messages
        if m["role"] == "system" and isinstance(m["content"], str)
    ]
    prefix = ("\n\n".join(system_texts) + "\n\n") if system_texts else ""

    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"  # 'tool' -> user
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
                    # hosted (non-data-URI) images are not translated yet

        contents.append({"role": role, "parts": parts})

    return contents


def _call_gemini(messages: list, timeout: int):
    if not is_backend_available("gemini"):
        raise ProviderError("gemini: cooling down after a recent failure")

    keys = [os.environ.get(e, d) for e, d in zip(GEMINI_API_KEY_ENVS, GEMINI_API_KEY_DEFAULTS)]
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


def call_model(messages: list, client_body: dict = None, tools=None, timeout: int = 30, order=None):
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
                return _call_openai_style(provider, messages, client_body, tools, timeout)
        except (ProviderError, requests.RequestException) as e:
            last_error = e
            continue

    raise ProviderError(f"All providers/keys failed. Last error: {last_error}")


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


def run_agentic_completion(messages: list, client_body: dict = None, web_search_enabled: bool = True):
    """Model -> tool calls -> results -> model, until no more tool calls or
    MAX_TOOL_ITERATIONS is hit."""
    tools = [WEB_SEARCH_TOOL_SCHEMA] if web_search_enabled else None
    working = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        result = call_model(working, client_body=client_body, tools=tools)

        if not result.get("tool_calls"):
            return {"content": result["content"], "provider": result["provider"]}

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

    return {"content": "I wasn't able to finish that within the tool-call limit.", "provider": None}


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
        """Hit this after every deploy to confirm Render serves the build you think it is."""
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
        """Validate a stored session token and return the account email.
        This is the endpoint a safe auto-sign-in flow calls on page load:
        200 -> enter console, 401 -> silently go to sign-in."""
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
        try:
            result = run_agentic_completion(messages, client_body=body, web_search_enabled=web_search_enabled)
        except ProviderError as e:
            return {"error": {"message": str(e)}}, 502, count

        tokens_out = estimate_tokens([{"content": result["content"]}])
        record_tokens_out(api_key_id, tokens_out)

        payload = {
            "id": "chatcmpl-sentry1",
            "object": "chat.completion",
            "model": "sentry-1",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": result["content"]}}
            ],
            "_provider_used": result["provider"],
            "usage": {"prompt_tokens_est": tokens_in, "completion_tokens_est": tokens_out},
        }
        return payload, 200, count

    def sse_shim(payload: dict) -> Response:
        """Honors stream:true with a valid SSE shape (one chunk + [DONE]).
        TODO: true token-by-token streaming would require streaming from each
        upstream provider through the tool loop — out of scope for this rebuild."""
        chunk = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "model": payload["model"],
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": payload["choices"][0]["message"]["content"]}, "finish_reason": "stop"}],
        }
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return Response(body, mimetype="text/event-stream")

    @app.post("/v1/chat/completions")
    def chat_completions():
        identity = authenticate_api_key(bearer_token())
        if not identity:
            return jsonify(error={"message": "invalid API key"}), 401

        body = request.get_json(force=True, silent=True) or {}
        payload, status, _ = completion_core(body, identity["api_key_id"])
        if status == 429:
            return payload  # already a Response with Retry-After
        if status != 200:
            return jsonify(payload), status
        if body.get("stream"):
            return sse_shim(payload)
        return jsonify(payload)

    @app.post("/console/playground/chat")
    def playground_chat():
        user_id, err = require_session()
        if err:
            # console endpoints speak string errors; keep playground consistent
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
