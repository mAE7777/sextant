"""Installed V1 CLI and single-request sextant-agent/1 protocol."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shlex
import shutil
import sys
from contextlib import redirect_stderr
from importlib.resources import files
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sextant.application import Application
from sextant.domain import (
    AuthorityContext,
    Command,
    CommandResult,
    DomainError,
    canonical,
    identity,
)
from sextant.environments import BrokerExecutor, PipeEnvironment
from sextant.render import chinese
from sextant.runtime import Executor
from sextant.storage import Store

COMMANDS = ("start", "inspect", "run", "pause", "resume", "cancel", "observe", "withdraw", "resolve", "revise", "export", "import", "authorize", "replay", "reconcile")
GUIDE_TOPICS = ("README", "data-and-recovery", "runtime")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sextant", description="Sextant V1: 可追溯的任务、实验、条件策略与修订。")
    p.add_argument("--root", default="runtime-v1", help="local runtime directory (default: ./runtime-v1)")
    p.add_argument("--delegated", action="store_true", help="host withholds authority to create/update goals or grants")
    p.add_argument("--broker-read-fd", type=int, help="host-supplied environment response descriptor")
    p.add_argument("--broker-write-fd", type=int, help="host-supplied environment request descriptor")
    p.add_argument("--broker-services", action="store_true", help="host broker owns approved providers/compute; required for an already-sandboxed runtime")
    p.add_argument("--episode-id", help="public environment episode ID for broker services")
    p.add_argument("--phase", type=int, default=0, help="public environment phase")
    sub = p.add_subparsers(dest="kind", required=True)
    for kind in COMMANDS:
        q = sub.add_parser(kind, help={"start": "创建任务与明确授权", "run": "推进编译/实验/修订至当前停止点", "inspect": "查询当前对象、有效性和原始回执", "revise": "修改目标或建立新依赖视图", "export": "导出完整可追溯任务；不执行效果", "import": "导入，默认撤销执行授权"}.get(kind, kind))
        if kind not in ("start", "import"):
            q.add_argument("mission_id", nargs="?" if kind == "inspect" else None)
        q.add_argument("--expected-revision", help="required for mutations; obtain from inspect")
        q.add_argument("--command-id", help="idempotency ID; reuse only for the identical request")
        q.add_argument("--payload", default="{}", help="JSON object, @file, or - for stdin")
        q.add_argument("--json", action="store_true", help="print machine result instead of Chinese report")
        if kind == "start":
            q.add_argument("--goal", help="convenience goal; authorization remains in payload")
        if kind == "export":
            q.add_argument("--output", help="write export package to a new file")
    sub.add_parser("agent", help="one JSON command on stdin, one JSON result on stdout; diagnostics on stderr")
    q = sub.add_parser("protocol-schema", help="print the versioned command/result/contract schemas")
    q.add_argument("--output")
    q = sub.add_parser("backup", help="create a new complete SQLite + immutable artifact backup directory")
    q.add_argument("destination")
    q.add_argument("--json", action="store_true", help="print the machine backup receipt")
    q = sub.add_parser("examples", help="copy the packaged V1 examples into a new directory; no runtime/model calls")
    q.add_argument("destination")
    q.add_argument("--json", action="store_true", help="print the machine copy receipt")
    q = sub.add_parser("guide", help="read the installed user guide; no runtime/model calls")
    q.add_argument("topic", nargs="?", default="README", choices=GUIDE_TOPICS)
    q = sub.add_parser("legacy", help="historical diagnostics; cannot establish V1 authority")
    q.add_argument("arguments", nargs=argparse.REMAINDER)
    return p


def schemas() -> dict[str, Any]:
    from sextant.compiler import CompileOptions
    from sextant.domain import Mission, Observation, Policy, Prediction, Proposal, Receipt, Revision
    return {"schema_version": 1, "protocol": "sextant-agent/1",
            "schemas": {c.__name__: c.model_json_schema() for c in (Command, CommandResult, Mission, Observation, Policy, Prediction, Proposal, Receipt, Revision, CompileOptions)}}


def _payload(text: str) -> dict[str, Any]:
    value = json.loads(sys.stdin.read() if text == "-" else Path(text[1:]).read_text() if text.startswith("@") else text)
    if not isinstance(value, dict):
        raise DomainError("payload", "payload must be a JSON object")
    return value


def _legacy(arguments: list[str]) -> int:
    """Audit the front door before dispatching the unchanged historical library."""
    from sextant.__main__ import build_parser, legacy_main

    message = ("历史入口仅允许只读 schema（不写文件）、status、pending、ledger 和帮助。"
               "旧写命令、运行检查和模型调用已停用。保留原文件；用 sextant guide data-and-recovery "
               "查看迁移，再通过 V1 import/start 建立任务及明确授权。")
    has_help = any(arg in ("-h", "--help") for arg in arguments[:arguments.index("--") if "--" in arguments else len(arguments)])
    if arguments and arguments[0] not in ("schema", "status", "pending", "ledger") and not has_help:
        raise DomainError("legacy_read_only", message)
    diagnostic = StringIO()
    try:
        with redirect_stderr(diagnostic):
            parsed = build_parser().parse_args(arguments or ["--help"])
    except SystemExit as exc:
        if exc.code == 0:
            raise  # argparse help remains compatible and never calls a handler
        raise DomainError("legacy_read_only", message + " 参数不受支持；用 legacy --help 查询历史语法。") from exc
    if parsed.cmd not in ("schema", "status", "pending", "ledger") or (parsed.cmd == "schema" and parsed.export is not None):
        raise DomainError("legacy_read_only", message)
    return legacy_main(arguments)


def _recovery(result: CommandResult, args: argparse.Namespace) -> str:
    if not any(error['code'] in ('conflict', 'expected_revision_required') for error in result.errors):
        return ''
    mission = result.mission_id or getattr(args, 'mission_id', None)
    inspect = f"sextant --root {shlex.quote(args.root)} inspect {shlex.quote(mission or 'MISSION_ID')}"
    return (f"恢复：先运行 {inspect}。确认原修改仍适用，再携带最新 --expected-revision CURRENT_REVISION "
            "和新的 --command-id 重试；不要直接重发未知的外部效果。\n")


def _human_view(app: Application, command: Command, result: CommandResult) -> CommandResult:
    """Query the committed view; never mix a later writer's revision into this result."""
    if not result.mission_id or not result.current_revision or command.kind in ('inspect', 'export', 'replay'):
        return result
    snapshot = app.execute(Command(kind='inspect', command_id=identity('view'), mission_id=result.mission_id))
    if snapshot.errors or snapshot.current_revision != result.current_revision:
        return result.model_copy(update={'next_actions': [*result.next_actions, 'inspect 最新版本；报告未合并后续变化']})
    return result.model_copy(update={'outputs': {**snapshot.outputs, **result.outputs}})


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and not arguments[0].startswith('-') and arguments[0] not in (
            *COMMANDS, 'agent', 'protocol-schema', 'backup', 'examples', 'guide', 'legacy'):
        arguments = ['legacy', *arguments]
    args = parser().parse_args(arguments)
    try:
        if args.kind == "legacy":
            return _legacy(args.arguments)
        if args.kind in ("examples", "guide"):
            resource = Path(str(files("sextant") / "resources" / ("examples/v1" if args.kind == "examples" else "user")))
            if not resource.is_dir():
                # Source checkout/editable install: the same explicit public
                # files bundled by the wheel's force-include mapping.
                resource = Path(__file__).resolve().parents[2] / ("examples/v1" if args.kind == "examples" else "docs/user")
            if args.kind == "examples":
                shutil.copytree(resource, Path(args.destination), dirs_exist_ok=False)
                if args.json:
                    print(canonical({"status": "copied", "destination": args.destination, "effects_executed": 0}))
                else:
                    print(f"示例已复制到：{args.destination}；未运行示例或调用模型。")
            else:
                print((resource / (args.topic + ".md")).read_text(), end="")
            return 0
        if args.kind == "protocol-schema":
            value = json.dumps(schemas(), ensure_ascii=False, indent=2) + "\n"
            if args.output:
                with Path(args.output).open("x") as f:
                    f.write(value)
            else:
                print(value, end="")
            return 0
        command: Command | None = None
        if args.kind == "agent":
            raw = sys.stdin.read(16_000_001)
            if len(raw) > 16_000_000:
                raise DomainError("command_limit", "command exceeds the local input limit")
            command = Command.model_validate_json(raw)
        effective_kind = command.kind if command is not None else args.kind
        store = Store(Path(args.root), create=effective_kind not in ("inspect", "export", "replay", "backup"))
        if args.kind == "backup":
            destination = Path(args.destination).absolute()
            report = store.backup_store(destination)
            if args.json:
                print(canonical({"status": "backed_up", "destination": str(destination), "artifacts": len(report["artifacts"]), "effects_executed": 0}))
            else:
                print(f"备份已保存：{destination}；{len(report['artifacts'])} 个 artifacts；清单：backup-manifest.json。")
            return 0
        services: dict[str, Executor] = {}
        if args.broker_read_fd is not None or args.broker_write_fd is not None:
            if args.broker_read_fd is None or args.broker_write_fd is None:
                raise DomainError("broker", "both inherited descriptors are required")
            pipe = PipeEnvironment(args.broker_read_fd, args.broker_write_fd, timeout=900)
            services["environment"] = pipe
            if args.broker_services:
                if not args.episode_id:
                    raise DomainError("broker", "broker services require a public episode ID")
                services.update({name: BrokerExecutor(name, pipe, args.episode_id, phase=args.phase)
                                 for name in ("codex", "claude-code", "typesafe", "sandbox.python")})
        elif args.broker_services:
            raise DomainError("broker", "broker services require inherited descriptors")
        authority = AuthorityContext(actor=f"local:{getpass.getuser()}:{os.getuid()}", source="local CLI host", can_authorize=not args.delegated)
        app = Application(store, authority, executors=services)
        if args.kind == "agent":
            assert command is not None
        else:
            if (not args.json and args.kind not in ('start', 'import', 'inspect', 'export', 'replay')
                    and not args.expected_revision):
                raise DomainError('expected_revision_required', '修改现有任务需要 --expected-revision；先 inspect 获取当前版本。')
            payload = _payload(args.payload)
            if args.kind == "start" and args.goal:
                payload["mission"] = {**payload.get("mission", {}), "goal": args.goal}
            command = Command(kind=args.kind, command_id=args.command_id or identity("cmd"),
                              mission_id=getattr(args, "mission_id", None), expected_revision=args.expected_revision, payload=payload)
        result = app.execute(command)
        export_receipt = ''
        if args.kind == "export" and args.output and not result.errors:
            data = canonical(result.outputs) + "\n"
            with Path(args.output).open("x") as f:
                f.write(data)
            export_receipt = f"已写入：{Path(args.output).absolute()}；{len(data.encode('utf-8'))} 字节；SHA-256：{hashlib.sha256(data.encode('utf-8')).hexdigest()}\n"
        if args.kind == "agent" or args.json:
            print(canonical(result))
        else:
            print(chinese(_human_view(app, command, result)), end="")
            print(_recovery(result, args), end="")
            if args.kind == 'export' and not result.errors:
                print(export_receipt or '尚未写入文件；用 --output 新文件保存完整导出包，或用 --json 输出机器结果。\n', end='')
        return 1 if result.errors else 0
    except (DomainError, ValidationError, ValueError, OSError) as exc:
        code = exc.code if isinstance(exc, DomainError) else "invalid_input"
        result = CommandResult(command_id="unparsed", status=code, errors=[{"code": code, "message": str(exc)}])
        if args.kind == "legacy":
            result = result.model_copy(update={'next_actions': ['sextant guide data-and-recovery', '保留原件；V1 import/start 建立明确授权的任务']})
            print(canonical(result))
        elif args.kind == "agent":
            print(canonical(result))
        elif not getattr(args, 'json', False):
            print(chinese(result) + _recovery(result, args), file=sys.stderr, end='')
        else:
            print(f"{code}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
