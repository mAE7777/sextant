# Sextant

Sextant helps you rethink a problem before you decide what to do. You give it a goal, your materials, and the permissions it may use. It builds several competing explanations of what is going on, a plan whose next step depends on what you observe, and the experiment most worth running next. When a result or a piece of counterevidence arrives, Sextant revises the plan and records what changed and why.

Every model call, observation, and revision is saved with its receipt. You can replay how a recommendation was reached and see which evidence it still depends on.

## How a task runs

1. `sextant start` creates a task from a goal, materials, and explicit permissions: which paths it may read, which services may receive which data, and which tools it may run.
2. `sextant run` moves the task forward until it needs something. A generator model proposes competing mechanisms. A challenger that sees the goal and the evidence, but not the main proposal, tries to refute them. Sextant turns what survives into a conditional policy and a next experiment.
3. Experiments run through a controlled interface, and their observations are settled against the policy's predictions.
4. When an observation is withdrawn or contradicted, Sextant invalidates everything that depended on it and revises the policy. `sextant inspect` shows the saved receipts next to a fresh check of what is still valid.

If there is no new information and no untried experiment, `run` returns `waiting` instead of calling a model again. A simple task such as `calculate 2 + 3 * 4` is solved offline, without a model.

Models only propose. Goals, preferences, and authorization come from you. Yes-or-no and multiple-choice judgments come from TypeSafe's Jev model, and Sextant records them as advice, not as proof.

[docs/architecture.md](docs/architecture.md) explains the design and the trade-offs behind it.

## Repository layout

- `src/sextant/`: the engine and the command line. Tasks live in a SQLite store with immutable artifacts that supports revisions, replay, export and import, migration, and backup.
- `src/sextant/providers/`: the adapters for the local model clients that generate proposals.
- `examples/v1/`: offline example tasks and a script that runs one from start to finish.
- `docs/user/`: the user guide, packaged with the wheel and printed by `sextant guide`.
- `tests/`: the offline test suite.
- `schema/` and `config.toml`: the decision-graph schema and default settings used by the older decision-graph and calibration tools, which `sextant legacy` exposes read-only.

## Try it offline

Sextant needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/). The commands below make no model calls.

```sh
uv sync --frozen
.venv/bin/python -m sextant.cli --help
.venv/bin/python examples/v1/workflow.py examples/v1/arithmetic-payload.json --output "$(mktemp -d)/arithmetic-example"
```

`.venv/bin/sextant guide` prints the user guide: the main commands, how a task runs, and how to inspect, replay, export, and back it up.

To run the offline test suite:

```sh
.venv/bin/python -m pytest -q -m "not live" -p no:cacheprovider
```

## Status

Sextant is a working prototype (`0.1.0rc11`). The task engine, the store, the command line, and the offline examples work, and the offline test suite covers them. The full version is still being developed and validated. Live runs need a local model client for proposals, a TypeSafe API key for judgments, and macOS for the execution sandbox. Those live routes and a comparison against baseline approaches are not yet qualified, so no release has been cut.

Command output is in Chinese. Use `--json` or the `agent` command for machine-readable output.

## License

No license is granted. All rights reserved.
