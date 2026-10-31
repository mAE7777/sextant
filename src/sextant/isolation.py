"""Enforced local process isolation. An unavailable OS boundary is a blocker."""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sextant.domain import DomainError


def _literal(path: Path) -> str:
    return f"(literal {json.dumps(str(path))})"


def _subpath(path: Path) -> str:
    return f"(subpath {json.dumps(str(path))})"


def seatbelt_profile(*, reads: list[Path], writes: list[Path], network: bool = False,
                     native_auth: bool = False, executables: list[Path] | None = None,
                     allow_fork: bool = True) -> str:
    """Default-deny, explicit content roots; ancestor DIRECTORY metadata is not content access."""
    ancestors = {p for path in reads + writes for p in path.absolute().parents}
    system_reads = [Path(p) for p in ("/System", "/usr", "/bin", "/sbin", "/Library")]
    execution = "(allow process-exec)" if executables is None else (
        "(allow process-exec " + " ".join(_literal(p) for p in executables) + ")")
    lines = ["(version 1)", "(deny default)", execution,
             "(allow sysctl-read)", "(allow mach-lookup)"]
    if allow_fork:
        lines.append("(allow process-fork)")
    lines.append("(allow file-read* " + " ".join(_subpath(p) for p in system_reads + reads) + ")")
    lines.append("(allow file-read* " + " ".join(_literal(p) for p in sorted(ancestors)) + ")")
    lines.append('(allow file-read* (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))')
    lines.append('(allow file-write* (literal "/dev/null"))')
    if writes:
        lines.append("(allow file-write* " + " ".join(_subpath(p) for p in writes) + ")")
    if network:
        # Only the trusted native client gets network. Model-selected tools are disabled separately.
        lines.extend(["(allow network-outbound)", "(allow network-bind)",
                      "(allow system-socket)", '(allow mach-register (local-name-prefix ""))',
                      '(allow file-read-metadata (literal "/etc") (literal "/var") (literal "/tmp") (literal "/private"))',
                      '(allow file-read* (subpath "/private/etc") (subpath "/private/var/db"))'])
    if native_auth:
        lines.extend(["(allow ipc-posix-shm)", "(allow ipc-posix-sem)",
                      "(allow user-preference-read)"])
    return "\n".join(lines) + "\n"


def python_reads() -> list[Path]:
    # Resolve executable and stdlib; do not grant the checkout or developer home.
    return sorted({Path(sys.executable).resolve().parent.parent,
                   Path(sys.base_prefix).resolve(), Path("/opt/homebrew/lib"),
                   Path("/opt/homebrew/opt"), Path("/opt/homebrew/Cellar")})


@contextmanager
def fresh_python(cwd: Path) -> Iterator[list[str]]:
    """Use source imports with a new private cache lookup root for each child.

    -B alone prevents writes but still reads old bytecode. An explicit -X
    prefix also changes lookup; -I ignores the equivalent environment setting.
    CPython does not populate it because bytecode writes are disabled. It is
    removed after the owned child has been reaped.
    """
    with tempfile.TemporaryDirectory(prefix="python-cache-", dir=cwd) as directory:
        cache = Path(directory).resolve()
        yield [str(Path(sys.executable).resolve()), "-I", "-S", "-B", "-X",
               f"pycache_prefix={cache}"]


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    process_group_reaped: bool


def run_isolated(command: list[str], *, cwd: Path, profile: str, stdin: str = "",
                 environment: dict[str, str] | None = None, timeout: float = 120,
                 output_limit: int = 16_000_000, cpu_seconds: int | None = None,
                 pass_fds: tuple[int, ...] = ()) -> ProcessResult:
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists():
        raise DomainError("isolation_unavailable", "this build requires macOS sandbox-exec for local execution")
    profile_path = cwd / "process.sb"
    profile_path.write_text(profile)
    env = environment or {"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8", "TMPDIR": str(cwd)}

    # A fixed launcher sets limits then execs Seatbelt. Avoid Python preexec_fn,
    # which can deadlock a multithreaded embedding application's forked child.
    limiter = (
        "import os,resource,sys; "
        f"resource.setrlimit(resource.RLIMIT_FSIZE,({output_limit},{output_limit})); "
        "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
        + (f"resource.setrlimit(resource.RLIMIT_CPU,({cpu_seconds},{cpu_seconds + 1})); " if cpu_seconds else "")
        + "os.execv(sys.argv[1],sys.argv[1:])"
    )

    # Output files are bounded by RLIMIT_FSIZE; pipes would permit unbounded RAM allocation.
    # A regular-file stdin avoids a circular wait in clients which inspect EOF
    # before draining large pipe prompts. It also bounds input to exact staged
    # bytes; no inherited terminal or host filesystem descriptor is exposed.
    with fresh_python(cwd) as launcher, tempfile.TemporaryFile(dir=cwd) as input_stream, tempfile.TemporaryFile(dir=cwd) as stdout, tempfile.TemporaryFile(dir=cwd) as stderr:
        input_stream.write(stdin.encode())
        input_stream.seek(0)
        proc = subprocess.Popen([*launcher, "-c", limiter,
                                 "/usr/bin/sandbox-exec", "-f", str(profile_path), *command],
                                cwd=cwd, env=env, stdin=input_stream, stdout=stdout, stderr=stderr,
                                start_new_session=True, pass_fds=pass_fds)
        timed_out = False
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        stdout.seek(0)
        stderr.seek(0)
        out = stdout.read(output_limit + 1)
        err = stderr.read(output_limit + 1)
        if len(out) > output_limit or len(err) > output_limit:
            raise DomainError("output_limit", "process output exceeded the operational byte limit")
    return ProcessResult(proc.returncode, out.decode(errors="replace"), err.decode(errors="replace"),
                         timed_out, True)


def verify_local_boundary() -> dict[str, object]:
    """Executable canary; no claim based merely on the binary's existence."""
    with tempfile.TemporaryDirectory(prefix="sextant-boundary-") as base:
        root = Path(base).resolve()
        private = root / "denied.txt"
        private.write_text("constructed boundary canary")
        work = root / "work"
        work.mkdir()
        program = (
            "import pathlib,socket,json\n"
            "r={}\n"
            f"p=pathlib.Path({str(private)!r})\n"
            "try: p.read_text(); r['private_read_denied']=False\n"
            "except PermissionError: r['private_read_denied']=True\n"
            "try: socket.socket().connect(('127.0.0.1',1)); r['network_denied']=False\n"
            "except PermissionError: r['network_denied']=True\n"
            "except OSError: r['network_denied']=False\n"
            "pathlib.Path('output.txt').write_text('allowed')\n"
            "print(json.dumps(r))\n"
        )
        profile = seatbelt_profile(reads=[work, *python_reads()], writes=[work])
        with fresh_python(work) as child:
            result = run_isolated([*child, "-c", program],
                                  cwd=work, profile=profile, timeout=15)
        if result.returncode != 0:
            return {"available": False, "reason": result.stderr[-600:], "exit_code": result.returncode}
        checks = json.loads(result.stdout)
        return {"available": all(checks.values()) and (work / "output.txt").read_text() == "allowed",
                "checks": checks, "mechanism": "macOS Seatbelt deny-default", "network": "denied"}
