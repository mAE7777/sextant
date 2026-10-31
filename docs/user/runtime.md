# Live runs

The offline features need only Python. A live run also sends the task's permitted material to a model client for proposals and to TypeSafe's Jev model for judgments, and it runs experiments in a local sandbox. Each of those needs something on the host machine.

## Check what a live run needs

- **A model client for proposals.** New tasks use `"generator": "codex"` with model `gpt-6.1-sol` and effort `xhigh`, unless the start payload's `options` say otherwise. Sextant looks for `codex` and `codex-code-mode-host` on your `PATH`. It sends a request only on macOS, only for that model and effort, and only when both binaries match the SHA-256 values in the package file `resources/native-codex-qualification.json`. Any other client version stops before sending, with `native_boundary_unqualified`. Set `"generator": "claude-code"` in `options` to use the `claude` command-line client instead.
- **A TypeSafe API key for judgments.** Set `TYPESAFE_API_KEY` in the environment. Without it, the judgments are marked unavailable, nothing is sent to TypeSafe, and the policy lists each missing judgment as unresolved.
- **macOS for experiments.** The Python sandbox runs under `/usr/bin/sandbox-exec` with no network access. Besides the inputs named for the experiment, it can read only the system and Python locations it needs to run. On another system the experiment is not run and the attempt fails with `runtime_unavailable`.

## Grant permissions in the start payload

A task can use only what its `authorization` grants. The live examples grant the model client, Jev, and the sandbox:

```json
"authorization": {
  "tools": ["codex", "typesafe", "sandbox.python"],
  "send_to": {"codex": ["*"], "typesafe": ["*"], "sandbox.python": ["*"]}
}
```

- `tools` lists the services the task may call.
- `send_to` lists, for each service, which task objects may be sent to it. `"*"` allows all of them.
- `read_paths` lists the local files the task may read. The live examples grant none.

Change these grants to fit your own material before you run a real task. `sextant authorize` updates them later. A proposal that asks for a tool outside the grant is refused with `unauthorized_tool`, and the grant does not grow.

To run a packaged live example with these grants:

```sh
sextant examples ./my-examples
python my-examples/workflow.py my-examples/finite-table-payload.json --output ./finite-table-example --live
```

## When a model client is unavailable

When a required client cannot run, nothing is sent and the task becomes `blocked`. `inspect` shows the cause as the task's blocker:

- `native_boundary_unqualified`: the Codex client or host is missing from your `PATH`, or its bytes do not match the pinned values.
- `runtime_unavailable`: another prerequisite of the client is missing, such as its login or the `claude` binary.

Fix the cause, then run again. Sextant never falls back to another provider.

If a request was sent and the reply was lost, the outcome is recorded as unknown. Run `reconcile` before you continue. Sextant does not resend an unknown request on its own.

## Run inside a host that already sandboxes Sextant

macOS does not allow one Seatbelt sandbox inside another. If a host program already isolates Sextant, the host can run the model clients, Jev, and the sandbox itself and pass them in through two inherited file descriptors:

```sh
sextant --broker-read-fd N --broker-write-fd M --broker-services --episode-id PUBLIC_ID --phase 0 run MISSION_ID --expected-revision CURRENT_REVISION
```

The host is then responsible for the actual isolation and for each call's receipt.
