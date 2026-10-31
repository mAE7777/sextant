from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from sextant import environments, isolation
from sextant.domain import digest
from sextant.runtime import AttemptContext


def test_stale_valid_bytecode_is_ignored_even_when_b_alone_reads_it(tmp_path):
    module = tmp_path / "cache_canary.py"
    poisoned = "value = 'cached'\n"
    current = "value = 'source'\n"
    assert len(poisoned) == len(current)
    module.write_text(poisoned)
    timestamp = module.stat().st_mtime_ns
    # The legacy child has no -X prefix, even when pytest itself does. Place
    # this deliberately stale fixture in that child's ordinary lookup path.
    old_cache = tmp_path / "__pycache__" / f"{module.stem}.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(str(module), cfile=str(old_cache), doraise=True)
    module.write_text(current)
    os.utime(module, ns=(timestamp, timestamp))
    before = old_cache.read_bytes()
    command = f"import sys;sys.path.insert(0,{str(tmp_path)!r});import cache_canary;print(cache_canary.value)"
    legacy = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", command],
                            capture_output=True, text=True, timeout=10, check=True)
    assert legacy.stdout.strip() == "cached"
    with isolation.fresh_python(tmp_path) as child:
        cache = Path(child[child.index("-X") + 1].split("=", 1)[1])
        assert cache.is_dir() and list(cache.iterdir()) == []
        fresh = subprocess.run([*child, "-c", command], capture_output=True,
                               text=True, timeout=10, check=True)
        assert fresh.stdout.strip() == "source"
        assert list(cache.iterdir()) == []
    assert not cache.exists()
    assert old_cache.read_bytes() == before


def test_fresh_cache_is_distinct_private_and_removed_on_error(tmp_path):
    with isolation.fresh_python(tmp_path) as outer:
        first = Path(outer[-1].split("=", 1)[1])
        with pytest.raises(RuntimeError), isolation.fresh_python(tmp_path) as inner:
            second = Path(inner[-1].split("=", 1)[1])
            assert first != second and first.is_dir() and second.is_dir()
            assert second.stat().st_mode & 0o077 == 0
            raise RuntimeError("constructed interruption")
        assert not second.exists() and first.is_dir()
    assert not first.exists()


def test_limiter_receives_explicit_cache_flags_without_changing_target(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")

    class Process:
        pid = 912345678
        returncode = 0

        def __init__(self, args, **kwargs):
            captured["args"] = args
            cache = Path(args[args.index("-X") + 1].split("=", 1)[1])
            captured["cache"] = cache
            assert list(cache.iterdir()) == []
            kwargs["stdout"].write(b"bounded")

        def communicate(self, *, timeout):
            pass

        def wait(self):
            return self.returncode

    monkeypatch.setattr(isolation.subprocess, "Popen", Process)
    monkeypatch.setattr(isolation.os, "killpg", lambda *args: None)
    target = ["/usr/bin/true", "--constructed-argument"]
    result = isolation.run_isolated(target, cwd=tmp_path, profile="(version 1)\n(deny default)")
    assert result.stdout == "bounded" and result.process_group_reaped
    args = captured["args"]
    assert args[1:5] == ["-I", "-S", "-B", "-X"]
    assert args[-len(target):] == target
    assert not captured["cache"].exists()


def test_sandbox_target_receives_separate_empty_cache_and_keeps_output_only_writes(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(environments.sys, "platform", "darwin")

    def run(command, **kwargs):
        cache = Path(command[command.index("-X") + 1].split("=", 1)[1])
        captured["cache"] = cache
        assert cache.is_dir() and list(cache.iterdir()) == []
        assert command[1:5] == ["-I", "-S", "-B", "-X"]
        assert command[-1].endswith("program.py")
        assert '(allow file-write* (subpath ' in kwargs["profile"]
        assert str(cache) not in kwargs["profile"]
        return isolation.ProcessResult(0, '{"ok":true}', "", False, True)

    monkeypatch.setattr(environments, "run_isolated", run)
    request = {"program": "print('unused fixture')", "inputs": {}}
    result = environments.PythonSandbox().invoke(request, AttemptContext("op", "attempt", "effect", 1, "cache"))
    assert result.output["value"] == {"ok": True}
    assert result.receipt.request_hash == digest(request)
    assert "fresh private per-child" in result.receipt.isolation["python_cache"]
    assert not captured["cache"].exists()


def test_real_python_child_observes_fresh_prefix_with_environment_override_ignored(tmp_path):
    script = "import json,sys;print(json.dumps({'cache':sys.pycache_prefix,'no_write':sys.dont_write_bytecode,'isolated':sys.flags.isolated,'no_site':sys.flags.no_site}))"
    with isolation.fresh_python(tmp_path) as child:
        expected = child[-1].split("=", 1)[1]
        result = subprocess.run([*child, "-c", script], capture_output=True, text=True,
                                timeout=10, check=True, env={"PYTHONPYCACHEPREFIX": "/ignored-unrelated-cache"})
        assert json.loads(result.stdout) == {"cache": expected, "no_write": True, "isolated": 1, "no_site": 1}
