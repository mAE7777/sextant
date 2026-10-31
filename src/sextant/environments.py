"""Explicit local source, Python sandbox, and inherited-pipe environment ports."""

from __future__ import annotations

import hashlib
import json
import os
import select
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from sextant.domain import DomainError, Receipt, canonical, digest, utcnow
from sextant.isolation import fresh_python, python_reads, run_isolated, seatbelt_profile
from sextant.runtime import AttemptContext, Invocation, NotSent


class LocalExecutor:
    name = "local"

    def cancel(self, context: AttemptContext) -> dict[str, Any]:
        return {"status": "unsupported", "reason": "no recoverable process handle in this host"}

    def reconcile(self, context: AttemptContext) -> Invocation | None:
        return None


def read_authorized_file(path: Path, allowed: list[Path], *, limit: int = 8_000_000) -> bytes:
    """Walk using no-follow directory descriptors; avoid check-then-open symlink races."""
    path = path.absolute()
    if ".." in path.parts:
        raise DomainError("unsafe_path", "parent traversal is forbidden")
    if not any(path == root.absolute() or (root.is_dir() and path.is_relative_to(root.absolute())) for root in allowed):
        raise DomainError("unauthorized_read", f"file is outside the authorized read scope: {path}")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(file_fd, "rb") as f:
            info = os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise DomainError("unsafe_source", "source must be a regular file")
            data = f.read(limit + 1)
        if len(data) > limit:
            raise DomainError("source_limit", "file exceeds the operational read limit; select a smaller source")
        return data
    except OSError as exc:
        raise DomainError("unsafe_source", "source unavailable or contains a symlink") from exc
    finally:
        os.close(descriptor)


class FileSource(LocalExecutor):
    name = "source.file"

    def __init__(self, allowed_paths: list[str]):
        self.allowed_paths = [Path(p) for p in allowed_paths]

    def capabilities(self) -> dict[str, Any]:
        return {"available": True, "read_only": True, "idempotent": True, "proves": "source contents only"}

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        started = utcnow()
        # A local read has no external effect: a refused or failed read is a known
        # local outcome with its real cause, never an unknown one.
        try:
            data = read_authorized_file(Path(request["path"]), self.allowed_paths)
            start, end = int(request.get("start", 0)), int(request.get("end", len(data)))
        except DomainError as exc:
            raise NotSent(exc.code, str(exc), kind="refused", receipt_fields={"auth_route": "mission-local-read"}) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise NotSent("invalid_input", f"source.file needs path and optional integer start/end: {exc}",
                          kind="refused", receipt_fields={"auth_route": "mission-local-read"}) from exc
        if not 0 <= start <= end <= len(data):
            raise NotSent("source_range", "requested byte range is outside source", kind="refused",
                          receipt_fields={"auth_route": "mission-local-read"})
        output = {"content": data[start:end].decode("utf-8"), "source": str(request["path"]),
                  "source_sha256": hashlib.sha256(data).hexdigest(), "read_range": [start, end],
                  "total_bytes": len(data), "truncated": start != 0 or end != len(data),
                  "provenance": "local-file", "measurement_protocol": "local-byte-read/1",
                  "supports_scope": "the source contains these bytes; source claims are not independently verified"}
        return Invocation(Receipt(executor=self.name, started_at=started, ended_at=utcnow(),
                                  request_hash=digest(request), output_hash=digest(output), status="completed",
                                  auth_route="mission-local-read", tool_state="read completed",
                                  residual_processes="none"), output)


SYSCTL_READ_RULE = "(allow sysctl-read)"
MACH_LOOKUP_RULE = "(allow mach-lookup)"
PROCESS_INFO_DENY = "(deny process-info*)"
DROPPED_FOR_GENERATED_CODE = (SYSCTL_READ_RULE, MACH_LOOKUP_RULE)


def experiment_profile(profile: str) -> str:
    """Restrict host information and services for model-generated programs.

    The shared builder grants sysctl-read and mach-lookup for the trusted native
    client. A generated program must not read other processes' names, arguments
    or environment (including the host process that holds provider credentials)
    or reach login session services. On current macOS a deny-default profile does
    not by itself deny process information, so the experiment profile drops both
    grants and adds an explicit deny. This is not process invisibility: basic
    POSIX calls (for example getpgid/getsid/getpriority/getppid) still reveal
    which process IDs exist and their group, session and priority. Pure
    standard-library computation still runs; host queries such as os.uname(),
    platform/sysconfig platform detection and ctypes are unavailable, and
    os.cpu_count() returns None.
    """
    lines = profile.splitlines()
    if (any(lines.count(rule) != 1 for rule in DROPPED_FOR_GENERATED_CODE) or "(deny default)" not in lines
            or any("process-info" in line for line in lines)):
        raise DomainError("sandbox_profile", "unexpected shared profile; refusing to run generated code")
    return "\n".join([*(line for line in lines if line not in DROPPED_FOR_GENERATED_CODE), PROCESS_INFO_DENY]) + "\n"


def host_information_policy(profile: str) -> str:
    """Describe the restriction actually present in the profile that was used."""
    lines = profile.splitlines()
    if PROCESS_INFO_DENY in lines and not any(rule in lines for rule in DROPPED_FOR_GENERATED_CODE):
        return "sysctl-read, mach-lookup and process-info denied"
    return "not the restricted experiment profile"


class PythonSandbox(LocalExecutor):
    name = "sandbox.python"

    def __init__(self, *, timeout: float = 30, cpu_seconds: int = 20):
        self.timeout, self.cpu_seconds = timeout, cpu_seconds

    def capabilities(self) -> dict[str, Any]:
        return {"available": sys.platform == "darwin", "network": "denied", "tools": ["python-stdlib"],
                "evidence": "requires executable OS probe", "timeout_seconds": self.timeout,
                "limits_basis": "host feasibility, adjustable", "idempotent": False}

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        started = utcnow()
        if not self.capabilities()["available"]:
            raise NotSent("runtime_unavailable", "the Python experiment sandbox needs macOS /usr/bin/sandbox-exec, "
                          f"which this host ({sys.platform}) does not provide; sandbox execution is disabled")
        program = request.get("program")
        if not isinstance(program, str) or not program.strip():
            raise NotSent("program", "Python experiment requires a program string", kind="refused")
        executable = Path(sys.executable).resolve()
        with tempfile.TemporaryDirectory(prefix="sextant-experiment-") as directory:
            work = Path(directory).resolve()
            (work / "program.py").write_text(program)
            (work / "input.json").write_text(canonical(request.get("inputs", {})))
            output_dir = work / "output"
            output_dir.mkdir()
            # Generated code cannot fork a child into a new session and outlive
            # the worker's process-group cancellation boundary.
            profile = experiment_profile(seatbelt_profile(reads=[work, *python_reads()], writes=[output_dir],
                                                          executables=[executable], allow_fork=False))
            with fresh_python(work) as child:
                result = run_isolated([*child, str(work / "program.py")], cwd=work,
                                      profile=profile, stdin=canonical(request.get("inputs", {})),
                                      timeout=self.timeout, cpu_seconds=self.cpu_seconds)
            artifacts = {}
            for p in sorted(output_dir.rglob("*")):
                if p.is_symlink():
                    raise DomainError("unsafe_output", "experiment created a symlink output")
                if p.is_file():
                    data = read_authorized_file(p, [output_dir])
                    artifacts[str(p.relative_to(output_dir))] = {
                        "sha256": hashlib.sha256(data).hexdigest(), "content": data.decode(errors="replace")}
        errors = []
        try:
            value = json.loads(result.stdout)
            canonical(value)
        except (ValueError, TypeError):
            value = None
            errors.append("experiment stdout must be one finite JSON value")
        if result.returncode:
            errors.append(f"experiment exited {result.returncode}: {result.stderr[-800:]}")
        if result.timed_out:
            errors.append("experiment timed out; local process group terminated")
        output = {"value": value, "files": artifacts, "stdout": result.stdout, "stderr": result.stderr,
                  "provenance": "constructed", "measurement_protocol": "sandbox-python/1",
                  "supports_scope": "this program on the declared inputs in the recorded sandbox"}
        return Invocation(Receipt(executor=self.name, started_at=started, ended_at=utcnow(),
                                  request_hash=digest(request), output_hash=digest(output),
                                  status="failed" if errors else "completed", exit_code=result.returncode,
                                  timed_out=result.timed_out, auth_route="mission-sandbox",
                                  isolation={"mechanism": "Seatbelt deny-default", "profile_hash": digest(profile),
                                             "network": "denied", "tools": ["python-stdlib"],
                                             "process_fork": "denied",
                                             "kernel_process_info": host_information_policy(profile),
                                             "python_cache": "fresh private per-child lookup prefix; bytecode writes disabled",
                                             "writes": "isolated output root only"},
                                  residual_processes="process group killed and leader reaped", tool_state="finished",
                                  errors=errors), output)


class PipeEnvironment(LocalExecutor):
    """No endpoint URL or scorer path reaches the candidate: just explicit inherited FDs."""

    name = "environment"

    def __init__(self, read_fd: int, write_fd: int, *, timeout: float = 120):
        if min(read_fd, write_fd) < 3 or read_fd == write_fd:
            raise DomainError("broker_fd", "broker requires distinct non-stdio inherited descriptors")
        self.read_fd, self.write_fd, self.timeout = read_fd, write_fd, timeout
        # These descriptors belong exclusively to this bridge. select() alone
        # does not bound a subsequent blocking write larger than pipe capacity.
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)
        self._lock = threading.Lock()
        self._broken = False

    def capabilities(self) -> dict[str, Any]:
        return {"available": True, "protocol": "sextant-eval-env/1", "idempotent": "receipt-specific",
                "network": "none; inherited pipe", "truth_access": "not supplied"}

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        started = utcnow()
        envelope = {**request, "schema_version": 1, "protocol": "sextant-eval-env/1",
                    "request_id": context.attempt_id}
        deadline = time.monotonic() + self.timeout
        if not self._lock.acquire(timeout=max(0, self.timeout)):
            raise TimeoutError("environment bridge is busy")
        try:
            if self._broken:
                raise ConnectionError("environment bridge requires a new connection after interrupted framing")
            message = memoryview((canonical(envelope) + "\n").encode())
            while len(message):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.write_fd], [], remaining)[1]:
                    raise TimeoutError("environment write timed out")
                try:
                    written = os.write(self.write_fd, message)
                except BlockingIOError:
                    continue
                if not written:
                    raise ConnectionError("environment write made no progress")
                message = message[written:]
            data = bytearray()
            while not data.endswith(b"\n"):
                if len(data) >= 8_000_000:
                    raise DomainError("broker_output_limit", "environment response too large")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.read_fd], [], [], remaining)[0]:
                    raise TimeoutError("environment response timed out")
                try:
                    byte = os.read(self.read_fd, 1)
                except BlockingIOError:
                    continue
                if not byte:
                    raise ConnectionError("environment pipe closed")
                data.extend(byte)
        except Exception:
            # A partial outgoing frame or late response cannot be safely reused
            # as the next request/response. Runtime records the effect as unknown.
            self._broken = True
            raise
        finally:
            self._lock.release()
        response = json.loads(data)
        if response.get("request_id") != context.attempt_id or response.get("episode_id") != request.get("episode_id"):
            raise DomainError("broker_mismatch", "environment response is for a different request or episode")
        if response.get("schema_version") != 1 or response.get("protocol") != "sextant-eval-env/1":
            raise DomainError("broker_schema", "unsupported environment response version")
        status = response.get("status")
        if status not in ("completed", "failed", "reconciliation_required"):
            raise DomainError("broker_status", "environment did not supply a known status")
        receipt = response.get("receipt", {})
        if receipt.get("input_hash") != digest(envelope):
            raise DomainError("broker_receipt", "environment receipt request hash mismatch")
        if receipt.get("output_hash") != digest({k: v for k, v in response.items() if k != "receipt"}):
            raise DomainError("broker_receipt", "environment receipt output hash mismatch")
        return Invocation(Receipt(executor=self.name, started_at=started, ended_at=utcnow(),
                                  request_hash=digest(request), output_hash=digest(response),
                                  status="unknown" if status == "reconciliation_required" else status,
                                  auth_route="mission-inherited-broker", external_id=receipt.get("effect_id"),
                                  isolation={"broker": "inherited FDs", "environment_revision": response.get("environment_revision")},
                                  tool_state=status, residual_processes="owned by external environment",
                                  errors=[str(e) for e in response.get("errors", [])]), response)


class BrokerExecutor(LocalExecutor):
    """Host-installed bridge for a runtime already inside an OS sandbox."""

    def __init__(self, name: str, pipe: PipeEnvironment, episode_id: str, *, phase: int = 0):
        if name not in ("claude-code", "codex", "typesafe", "sandbox.python"):
            raise DomainError("broker_service", "unsupported broker executor")
        self.name, self.pipe, self.episode_id, self.phase = name, pipe, episode_id, phase

    def capabilities(self) -> dict[str, Any]:
        return {"available": True, "executor": self.name, "transport": "inherited-broker",
                "evidence": "actual inner execution receipt required", "reconcile": "unsupported"}

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        delivered = self.pipe.invoke({"episode_id": self.episode_id, "phase": self.phase, "kind": "experiment",
                                      "action": "runtime.invoke", "inputs": {"executor": self.name, "request": request}}, context)
        if delivered.receipt.status != "completed":
            receipt = delivered.receipt.model_copy(update={"executor": self.name, "request_hash": digest(request),
                                                           "output_hash": digest(None)})
            return Invocation(receipt, None)
        raw = delivered.output.get("service_result")
        if not isinstance(raw, dict):
            raise DomainError("broker_service", "broker omitted the actual execution result")
        receipt = Receipt.model_validate(raw["receipt"])
        if receipt.executor != self.name or receipt.request_hash != digest(request) or receipt.output_hash != digest(raw["output"]):
            raise DomainError("broker_service_receipt", "inner receipt does not bind executor/request/output")
        receipt.isolation = {**receipt.isolation, "bridge_receipt": delivered.receipt.model_dump()}
        return Invocation(receipt, raw["output"])
