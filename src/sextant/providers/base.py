"""The request/answer file handshake: how the runtime asks an LLM for a step, provider-agnostic.

The runtime writes requests/<id>.request.json with the prompt, the input pack and a JSON
Schema for the answer. Whoever fulfils it (an interactive Claude Code session, an isolated subagent,
headless Codex, an OpenAI-compatible API) writes requests/<id>.answer.json. The runtime
validates the answer against the schema, records provider, family and isolation in the
llm-calls ledger, and only then continues.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field

from sextant.ledger import now_iso

Role = Literal["generator", "challenger", "grader", "paraphraser", "translator", "renderer", "mapper", "demo"]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^rq_[A-Za-z0-9_-]{1,40}$")
    role: Role
    purpose: str
    node_ids: list[str] = Field(default_factory=list)
    prompt: str
    answer_schema: dict[str, Any]
    constraints: dict[str, Any] = Field(default_factory=dict)
    provider: str
    created: str
    attempt: int = 1
    previous_errors: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(pattern=r"^rq_[A-Za-z0-9_-]{1,40}$")
    answer: Any
    provider: str
    family: str
    isolation: Literal["same-session", "isolated", "n/a"]
    created: str = Field(default_factory=now_iso)
    notes: str | None = None
    model: str | None = None


class Provider(Protocol):
    name: str
    family: str
    isolation: Literal["same-session", "isolated", "n/a"]

    def fulfil(self, request: Request, decision_dir: Path) -> Answer | None: ...

    def probe(self) -> dict[str, Any]: ...


REQUEST_ID_RE = re.compile(r"^rq_[A-Za-z0-9_-]{1,40}$")


def new_request_id() -> str:
    return "rq_" + uuid.uuid4().hex[:10]


def check_request_id(request_id: str) -> str:
    """Request ids name files; anything outside the safe pattern is refused (no path traversal)."""
    if not REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError(f"invalid request id {request_id!r}")
    return request_id


def rejected_answer_path(decision_dir: Path, request_id: str, attempt: int) -> Path:
    return requests_dir(decision_dir) / f"{check_request_id(request_id)}.answer.rejected{attempt}.json"


def check_answer_schema(schema: dict[str, Any]) -> list[str]:
    """Errors in the request's own JSON Schema (a malformed schema is a runtime bug, not the executor's)."""
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        return [f"answer_schema invalid: {e.message}"]
    return []


def requests_dir(decision_dir: Path, create: bool = False) -> Path:
    """The requests folder. Path lookups never create it; only writers pass create=True."""
    d = decision_dir / "requests"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def request_path(decision_dir: Path, request_id: str) -> Path:
    return requests_dir(decision_dir) / f"{check_request_id(request_id)}.request.json"


def answer_path(decision_dir: Path, request_id: str) -> Path:
    return requests_dir(decision_dir) / f"{check_request_id(request_id)}.answer.json"


def write_request(decision_dir: Path, request: Request) -> Path:
    requests_dir(decision_dir, create=True)
    p = request_path(decision_dir, request.id)
    p.write_text(request.model_dump_json(indent=2))
    return p


def read_request(decision_dir: Path, request_id: str) -> Request:
    return Request.model_validate_json(request_path(decision_dir, request_id).read_text())


def read_answer(decision_dir: Path, request_id: str) -> Answer:
    p = answer_path(decision_dir, request_id)
    if not p.exists():
        raise FileNotFoundError(f"no answer file yet: {p}")
    return Answer.model_validate_json(p.read_text())


def write_answer(decision_dir: Path, answer: Answer) -> Path:
    requests_dir(decision_dir, create=True)
    p = answer_path(decision_dir, answer.request_id)
    p.write_text(answer.model_dump_json(indent=2))
    return p


def validate_answer(request: Request, answer: Answer) -> list[str]:
    """Errors from the request's JSON Schema; empty means the answer is admissible."""
    if answer.request_id != request.id:
        return [f"answer is for {answer.request_id}, request is {request.id}"]
    schema_errors = check_answer_schema(request.answer_schema)
    if schema_errors:
        return schema_errors
    validator = Draft202012Validator(request.answer_schema)
    errors = sorted(validator.iter_errors(answer.answer), key=lambda e: list(e.path))
    return [f"{'/'.join(str(p) for p in e.path) or '$'}: {e.message}" for e in errors]


def pending_requests(decision_dir: Path) -> list[str]:
    """File view of pending: request files with neither an accepted nor a rejected answer file.

    Read-only: never creates the requests directory.
    """
    d = decision_dir / "requests"
    if not d.is_dir():
        return []
    ids = sorted(p.name[: -len(".request.json")] for p in d.glob("*.request.json"))
    out: list[str] = []
    for i in ids:
        if answer_path(decision_dir, i).exists():
            continue
        if any(d.glob(f"{i}.answer.rejected*.json")):
            continue
        out.append(i)
    return out


def schema_for(answer_schema: dict[str, Any]) -> dict[str, Any]:
    """Return the schema with the 2020-12 dialect declared (some providers require it)."""
    s = dict(answer_schema)
    s.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    return s


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
