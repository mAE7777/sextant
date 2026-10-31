# Sextant user guide

Sextant keeps each task in a local runtime directory: its goal, permissions, evidence, proposals, policy, and a receipt for every external call. This page shows the main commands and walks through one task that runs offline. Two more topics cover the rest:

- `sextant guide data-and-recovery`: inspect, replay, export, import, back up, and recover a task.
- `sextant guide runtime`: what a live run needs, and what to do when a model client is unavailable.

The commands below use `sextant`. In a source checkout after `uv sync`, run `.venv/bin/sextant` instead.

## Run a task offline

A goal written as `calculate <expression>` is solved exactly without a model, so this task makes no external calls.

```sh
sextant --root ./my-runtime start --goal "calculate 7 * (3 + 2)" --json
```

The JSON result holds `mission_id` and `current_revision`. Save both. Every command that changes a task needs the task's current revision, so a stale command cannot overwrite a newer change.

```sh
sextant --root ./my-runtime run MISSION_ID --expected-revision CURRENT_REVISION --json
```

The result has `"status": "complete"`, a new `current_revision`, and a compile step whose `direct_result` is 35. Show the whole task:

```sh
sextant --root ./my-runtime inspect MISSION_ID
```

Run it again with the new revision. With no new input and no untried experiment, `run` returns `"status": "waiting"` with the reason `no new input or untried experiment`, and it calls nothing.

Without `--json`, commands print a report in Chinese. The JSON result carries the same content with English field names. Leave out the mission ID to list every task in the runtime:

```sh
sextant --root ./my-runtime inspect
```

## Know the main commands

| Command | What it does |
|---|---|
| `start` | Creates a task from a goal, optional observations, permissions, and options. |
| `run MISSION_ID` | Advances the task until it needs something. One call takes up to 3 steps by default. Set `--payload '{"steps": 7}'` for another number from 1 to 100. |
| `inspect [MISSION_ID]` | Lists tasks, or shows one task with saved receipts next to a fresh check of what is still valid. |
| `observe MISSION_ID` | Adds an observation. Objects that depended on the old view become invalid. |
| `withdraw MISSION_ID` | Withdraws an observation. Objects that cited it become invalid. |
| `revise MISSION_ID` | Changes the goal (`mission_input`) or the options of a task. |
| `resolve MISSION_ID` | Settles a frozen prediction against an observation. |
| `authorize MISSION_ID` | Updates or revokes the task's permissions. |
| `pause`, `resume`, `cancel` | Stop or continue advancement. Effects already sent stay recorded. |
| `replay`, `export`, `import`, `backup`, `reconcile` | Rebuild, move, copy, and repair saved tasks. See `sextant guide data-and-recovery`. |
| `agent` | Reads one JSON command from standard input and writes one JSON result to standard output. |
| `protocol-schema` | Prints the JSON schemas for commands, results, and task objects. |
| `examples DEST` | Copies the packaged example files into a new directory. |
| `legacy` | Read-only access to the older decision-graph tools. |

Each command takes `--payload` as a JSON object, `@file`, or `-` for standard input. Pass `--command-id ID` to make a retry safe: repeating the same ID with the same request returns the saved result instead of running again, and reusing an ID for a different request fails with `command_conflict`.

## Add and withdraw evidence

An observation records its source, the scope it supports, and its content. `sextant examples ./my-examples` copies a sample observation file:

```sh
sextant examples ./my-examples
sextant --root ./my-runtime observe MISSION_ID --expected-revision CURRENT_REVISION --payload @my-examples/service-cache-feedback.json --json
```

The result lists the `observation_ref` and the objects that became invalid. To withdraw that observation, pass its reference and a reason:

```sh
sextant --root ./my-runtime withdraw MISSION_ID --expected-revision CURRENT_REVISION --json --payload '{"observation_ref": OBSERVATION_REF, "reason": "added to the wrong task"}'
```

The withdrawn observation stays in the history with its reason. The next `run` compiles again without it.

## Run the packaged examples

The example script runs `start`, `run`, `inspect`, `replay`, and `export` through the `agent` command. It saves each request and result as numbered JSON files in a new output directory, with the task's runtime in its `runtime` folder. Run it with the Python that has Sextant installed:

```sh
python my-examples/workflow.py my-examples/arithmetic-payload.json --output ./arithmetic-example
```

The other payloads, `finite-table-payload.json` and `service-cache-payload.json`, authorize a model client, Jev, and the Python sandbox. The script refuses them unless you add `--live`. See `sextant guide runtime` before you do.

## Script it with the agent command

```sh
sextant --root ./my-runtime agent < my-examples/arithmetic-start.json
```

Each process reads one command and writes one result. Diagnostics go to standard error. A command names its `kind`, `command_id`, `mission_id`, `expected_revision`, and `payload`. `sextant protocol-schema --output schema.json` writes the full schema to a new file.

## Know where output goes

Everything a task produces stays in the runtime directory given by `--root`. Without `--root`, the directory is `./runtime-v1` under the current directory.

- `state.sqlite3` is the authoritative record of every task, revision, event, and receipt.
- `artifacts/` holds immutable content files, each named by its SHA-256.
- `native-evidence/` holds the audit files of native model client calls.

Commands write outside the runtime only where you name a path, and they never overwrite one: `export --output FILE`, `backup DEST`, `examples DEST`, and `protocol-schema --output FILE` all require a path that does not exist yet.
