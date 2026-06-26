#!/usr/bin/env python3
"""
AI Gateway Shield 🛡️
Protects against AI provider downtime with automatic failover.
Tries free → paid → fallback, no user intervention needed.
"""
import http.server
import json
import urllib.request
import urllib.error
import time
import os
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
HOST = "127.0.0.1"
PORT = 8015

# Providers ordered by priority (first = cheapest/free)
PROVIDERS = [
    {
        "name": "zen-free",
        "base_url": "https://opencode.ai/zen/v1/chat/completions",
        "api_key_env": "ZEN_KEY",
        "models": ["deepseek-v4-flash-free"],
        "cost": "free",
        "timeout": 30,
    },
    {
        "name": "zen-paid",
        "base_url": "https://opencode.ai/zen/v1/chat/completions",
        "api_key_env": "ZEN_KEY",
        "models": ["deepseek-v4-flash"],
        "cost": "paid",
        "timeout": 30,
    },
    {
        "name": "pollinations",
        "base_url": "https://text.pollinations.ai",
        "api_key_env": None,
        "models": [],
        "cost": "free",
        "timeout": 30,
    },
]

# Circuit breaker settings
MAX_FAILURES = 3          # Failures before cooldown
COOLDOWN_SECONDS = 120    # Wait 2 min before retrying failed provider
RATE_LIMIT_BACKOFF = 60   # Backoff on 429

# ============================================================
# SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GATEWAY] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gateway")


class CircuitBreaker:
    """Tracks provider health with automatic cooldown."""

    def __init__(self):
        self.failures = defaultdict(int)
        self.cooldown_until = {}
        self.stats = defaultdict(lambda: {"tries": 0, "success": 0, "fail": 0, "last_used": None})

    def record_success(self, name):
        self.failures[name] = 0
        self.cooldown_until.pop(name, None)
        self.stats[name]["success"] += 1
        self.stats[name]["last_used"] = datetime.now()

    def record_failure(self, name, status_code=None):
        self.failures[name] += 1
        self.stats[name]["fail"] += 1
        self.stats[name]["last_used"] = datetime.now()

        if self.failures[name] >= MAX_FAILURES:
            until = datetime.now() + timedelta(seconds=COOLDOWN_SECONDS)
            self.cooldown_until[name] = until
            log.warning(f"🛑 {name}: Circuit opened (cooldown {COOLDOWN_SECONDS}s)")
        elif status_code == 429:
            until = datetime.now() + timedelta(seconds=RATE_LIMIT_BACKOFF)
            self.cooldown_until[name] = until
            log.warning(f"⏳ {name}: Rate limited (backoff {RATE_LIMIT_BACKOFF}s)")

    def is_available(self, name):
        if name in self.cooldown_until:
            if datetime.now() < self.cooldown_until[name]:
                remaining = (self.cooldown_until[name] - datetime.now()).seconds
                log.info(f"  ⏸️  {name} in cooldown ({remaining}s left)")
                return False
            else:
                self.cooldown_until.pop(name, None)
                self.failures[name] = 0
                log.info(f"  🔄 {name} cooldown expired, retrying")
        return True

    def get_report(self):
        lines = []
        for name, s in sorted(self.stats.items()):
            status = "🟢" if s["success"] > max(s["fail"], 1) else "🔴"
            lines.append(f"  {status} {name}: {s['tries']} tries, {s['success']} ok, {s['fail']} fail")
        return "\n".join(lines)


circuit = CircuitBreaker()


def get_api_key(env_var):
    """Get API key from environment."""
    if not env_var:
        return None
    return os.environ.get(env_var, "")


def call_provider(provider, model, messages, max_tokens=500):
    """Call a single provider. Returns (content, model_used, cost, raw)."""
    api_key = get_api_key(provider["api_key_env"])
    circuit.stats[provider["name"]]["tries"] += 1

    if provider["name"] == "pollinations":
        return call_pollinations(provider, messages, max_tokens)

    payload = json.dumps({
        "model": model or provider["models"][0],
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        provider["base_url"],
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "gateway/1.0",
        },
        method="POST",
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        resp = urllib.request.urlopen(req, timeout=provider["timeout"])
        data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"].get("content", "")
        cost = data.get("cost", "?")
        log.info(f"  ✅ {provider['name']} → {len(content)} chars, cost={cost}")
        return content, model or provider["models"][0], cost, data
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        circuit.record_failure(provider["name"], e.code)
        log.warning(f"  ❌ {provider['name']} HTTP {e.code}: {body[:80]}")
        raise
    except Exception as e:
        circuit.record_failure(provider["name"])
        log.warning(f"  ❌ {provider['name']} error: {e}")
        raise


def call_pollinations(provider, messages, max_tokens):
    """Call Pollinations free text API as last-resort fallback."""
    last_msg = messages[-1]["content"][:500] if messages else "hello"
    url = f"https://text.pollinations.ai/{urllib.parse.quote(last_msg)}"
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=provider["timeout"])
        content = resp.read().decode()[:max_tokens]
        log.info(f"  ✅ pollinations → {len(content)} chars (free text)")
        circuit.record_success(provider["name"])
        return content, "pollinations-text", "free", None
    except Exception as e:
        circuit.record_failure(provider["name"])
        log.warning(f"  ❌ pollinations: {e}")
        raise


def smart_chat(messages, model=None, max_tokens=500):
    """
    Try providers in priority order. Free first, fallback to paid.
    Circuit breaker skips unhealthy providers.
    """
    last_error = None

    for provider in PROVIDERS:
        name = provider["name"]
        if not circuit.is_available(name):
            continue

        # Skip zen-paid if no key set
        if provider["api_key_env"] and not get_api_key(provider["api_key_env"]):
            if "paid" in name or "zen" in name:
                log.info(f"  ⏭️  {name}: No API key set")
                continue

        try:
            content, used_model, cost, raw = call_provider(provider, model, messages, max_tokens)
            circuit.record_success(name)
            return {
                "content": content,
                "model": used_model,
                "provider": name,
                "cost": cost,
                "fallback_used": name != PROVIDERS[0]["name"],
            }
        except Exception as e:
            last_error = str(e)
            continue

    # All providers failed
    return {
        "content": f"⚠️ All AI providers unavailable. Last error: {last_error}",
        "model": "none",
        "provider": "none",
        "cost": "N/A",
        "error": last_error,
    }


# ============================================================
# HTTP SERVER (OpenAI-compatible endpoint)
# ============================================================
class GatewayHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        elif self.path == "/v1/chat/turbo":
            self._handle_chat()
        else:
            self._send_json({"error": "not_found"}, 404)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "uptime": str(datetime.now() - start_time).split(".")[0],
                "providers": [
                    {
                        "name": p["name"],
                        "cost": p["cost"],
                        "available": circuit.is_available(p["name"]),
                    }
                    for p in PROVIDERS
                ],
                "stats": circuit.get_report(),
            })
        elif self.path == "/v1/models":
            models = []
            for p in PROVIDERS:
                for m in p.get("models", []):
                    models.append({"id": m, "provider": p["name"], "cost": p["cost"]})
            models.append({"id": "pollinations-text", "provider": "pollinations", "cost": "free"})
            self._send_json({"data": models})
        else:
            self._send_json({"error": "not_found"}, 404)

    def _handle_chat(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            data = json.loads(body)
        except Exception:
            self._send_json({"error": "invalid_json"}, 400)
            return

        messages = data.get("messages", [])
        model = data.get("model", None)
        max_tokens = data.get("max_tokens", 500)

        if not messages:
            self._send_json({"error": "no_messages"}, 400)
            return

        log.info(f"→ Request: {len(messages)} msgs, model={model}, max_tokens={max_tokens}")
        result = smart_chat(messages, model, max_tokens)

        # Format as OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result["model"],
            "provider": result["provider"],
            "cost": result["cost"],
            "fallback": result.get("fallback_used", False),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": len(result["content"].split())},
        }

        if result.get("error"):
            response["error"] = result["error"]

        log.info(f"← Response: {result['provider']} ({result['cost']}), content={len(result['content'])} chars")
        self._send_json(response)


def run_server():
    server = http.server.HTTPServer((HOST, PORT), GatewayHandler)
    log.info(f"🛡️  AI Gateway Shield running on http://{HOST}:{PORT}")
    log.info(f"   Providers: {[p['name'] for p in PROVIDERS]}")
    log.info(f"   Circuit breaker: {MAX_FAILURES} failures → {COOLDOWN_SECONDS}s cooldown")
    log.info(f"   📌 Set ZEN_KEY env var for DeepSeek access")
    log.info(f"   📌 Health: http://{HOST}:{PORT}/health")
    log.info(f"   📌 API:    http://{HOST}:{PORT}/v1/chat/completions")
    server.serve_forever()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start_time = datetime.now()
    run_server()
