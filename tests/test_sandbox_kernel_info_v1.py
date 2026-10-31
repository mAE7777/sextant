from __future__ import annotations

import sys

import pytest

from sextant import environments, isolation
from sextant.domain import DomainError
from sextant.runtime import AttemptContext


def shared_profile() -> str:
    return isolation.seatbelt_profile(reads=[], writes=[], executables=[], allow_fork=False)


def test_experiment_profile_drops_host_grants_and_denies_process_info():
    shared = shared_profile()
    for rule in environments.DROPPED_FOR_GENERATED_CODE:
        assert rule in shared.splitlines()
    narrowed = environments.experiment_profile(shared).splitlines()
    assert "sysctl" not in "\n".join(narrowed) and "mach-lookup" not in "\n".join(narrowed)
    assert narrowed[-1] == environments.PROCESS_INFO_DENY
    assert narrowed[:-1] == [line for line in shared.splitlines() if line not in environments.DROPPED_FOR_GENERATED_CODE]
    assert "(deny default)" in narrowed


@pytest.mark.parametrize("variant", ["without-sysctl", "duplicate-sysctl", "without-mach-lookup",
                                     "allow-process-info", "no-deny-default"])
def test_experiment_profile_refuses_unexpected_shared_profiles(variant):
    lines = shared_profile().splitlines()
    if variant == "without-sysctl":
        lines = [line for line in lines if line != environments.SYSCTL_READ_RULE]
    elif variant == "duplicate-sysctl":
        lines.append(environments.SYSCTL_READ_RULE)
    elif variant == "without-mach-lookup":
        lines = [line for line in lines if line != environments.MACH_LOOKUP_RULE]
    elif variant == "allow-process-info":
        lines.append("(allow process-info*)")
    else:
        lines = [line for line in lines if line != "(deny default)"]
    with pytest.raises(DomainError):
        environments.experiment_profile("\n".join(lines) + "\n")


def test_receipt_label_is_derived_from_the_profile_used():
    assert environments.host_information_policy(environments.experiment_profile(shared_profile())) == \
        "sysctl-read, mach-lookup and process-info denied"
    assert environments.host_information_policy(shared_profile()) == "not the restricted experiment profile"
    partial = "\n".join(line for line in shared_profile().splitlines()
                        if line not in environments.DROPPED_FOR_GENERATED_CODE) + "\n"
    assert environments.host_information_policy(partial) == "not the restricted experiment profile"


def test_python_sandbox_uses_the_restricted_profile(monkeypatch):
    captured = {}
    monkeypatch.setattr(environments.sys, "platform", "darwin")

    def run(command, **kwargs):
        captured["profile"] = kwargs["profile"]
        return isolation.ProcessResult(0, '{"ok":true}', "", False, True)

    monkeypatch.setattr(environments, "run_isolated", run)
    result = environments.PythonSandbox().invoke({"program": "print(1)", "inputs": {}},
                                                 AttemptContext("op", "attempt", "mission", 1, "effect"))
    assert environments.PROCESS_INFO_DENY in captured["profile"].splitlines()
    assert result.receipt.isolation["kernel_process_info"] == "sysctl-read, mach-lookup and process-info denied"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_real_sandboxed_program_runs_pure_stdlib_work_and_cannot_query_the_host():
    program = ("import decimal, hashlib, json, math, os, random, re, statistics, sys\n"
               "data = json.load(sys.stdin)\n"
               "def blocked(fn):\n"
               "    try:\n"
               "        fn()\n"
               "        return False\n"
               "    except PermissionError:\n"
               "        return True\n"
               "print(json.dumps({'sum': sum(data['values']), 'mean': statistics.mean(data['values']),\n"
               "                  'digest': hashlib.sha256(b'x').hexdigest()[:8], 'root': math.isqrt(49),\n"
               "                  'match': bool(re.fullmatch(r'a+', 'aaa')), 'dec': str(decimal.Decimal('0.1') * 3),\n"
               "                  'rand': 0 <= random.Random(1).random() < 1, 'cpus': os.cpu_count(),\n"
               "                  'uname_blocked': blocked(os.uname)}))\n")
    result = environments.PythonSandbox(timeout=30).invoke(
        {"program": program, "inputs": {"values": [1, 2, 3]}}, AttemptContext("op", "attempt", "mission", 1, "effect"))
    assert result.receipt.status == "completed", result.output["stderr"]
    value = result.output["value"]
    assert (value["sum"], value["mean"], value["root"], value["match"], value["dec"], value["rand"]) == (6, 2, 7, True, "0.3", True)
    assert value["cpus"] is None
    assert value["uname_blocked"] is True
