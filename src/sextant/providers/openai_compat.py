"""OpenAI-compatible chat providers behind one implementation: DeepSeek, Gemini, OpenAI.

The model id is never hardcoded: `setup` reads the provider's model list and stores the
chosen id in config.toml. Structured output is requested through response_format with
the request's JSON Schema; providers that reject json_schema fall back to json_object and
the runtime's own validation does the rest.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from sextant.providers.base import Answer, Request

PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY", "family": "deepseek"},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "family": "google",
    },
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "family": "openai"},
}


class OpenAICompatProvider:
    isolation: Literal["same-session", "isolated", "n/a"] = "isolated"

    def __init__(self, preset: str, model: str | None = None, timeout_s: int = 180):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset}; known: {', '.join(PRESETS)}")
        self.name = preset
        self.preset = PRESETS[preset]
        self.family = self.preset["family"]
        self.model = model or ""
        self.timeout_s = timeout_s

    @property
    def key(self) -> str | None:
        return os.environ.get(self.preset["key_env"])

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.preset["base_url"].rstrip("/") + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(
            self.preset["base_url"].rstrip("/") + path,
            headers={"Authorization": f"Bearer {self.key}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data

    def list_models(self) -> list[str]:
        data = self._get("/models")
        items = data.get("data", []) if isinstance(data, dict) else []
        return sorted(str(m.get("id")) for m in items if isinstance(m, dict) and m.get("id"))

    def fulfil(self, request: Request, decision_dir: Path) -> Answer | None:
        if not self.key:
            raise RuntimeError(f"{self.name}: {self.preset['key_env']} is not set")
        if not self.model:
            raise RuntimeError(f"{self.name}: no model id configured; run setup")
        system = "Answer with a single JSON object that conforms to the given JSON Schema. No prose."
        user = request.prompt + "\n\nJSON Schema for your answer:\n" + json.dumps(request.answer_schema)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": request.answer_schema, "strict": True},
            },
        }
        try:
            data = self._post("/chat/completions", body)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                body["response_format"] = {"type": "json_object"}
                data = self._post("/chat/completions", body)
            else:
                raise
        content = data["choices"][0]["message"]["content"]
        try:
            answer_obj = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{self.name} returned non-JSON content: {content[:300]}") from e
        return Answer(
            request_id=request.id,
            answer=answer_obj,
            provider=self.name,
            family=self.family,
            isolation=self.isolation,
            model=str(data.get("model") or self.model),
        )

    def probe(self) -> dict[str, Any]:
        if not self.key:
            return {"available": False, "reason": f"{self.preset['key_env']} not set"}
        t0 = time.perf_counter()
        try:
            models = self.list_models()
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"{type(e).__name__}: {str(e)[:120]}"}
        return {
            "available": True,
            "models_count": len(models),
            "models_sample": models[:8],
            "configured_model": self.model or None,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }
