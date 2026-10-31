# Data, export, and recovery

A runtime directory keeps every task in `state.sqlite3` and its content files in `artifacts/`. Content files are written, synced, and checked before the database transaction that refers to them commits. Do not edit either by hand: inspection checks every hash, and a rewritten hash does not repair evidence.

## Inspect and replay a task

`inspect` shows the saved objects and receipts next to a fresh check of what is still valid. An object that depends on withdrawn or changed input is shown with the reason it is no longer current.

```sh
sextant --root ./my-runtime inspect MISSION_ID
```

`replay` rebuilds the task's state from its saved events and compares the result with the stored state. In the JSON result, `matches_state`, `matches_heads`, and `matches_current` report the comparison.

```sh
sextant --root ./my-runtime replay MISSION_ID --json
```

Neither command sends anything. Both report `effects_executed` as 0.

## Export and import a task

```sh
sextant --root ./my-runtime export MISSION_ID --output mission-export.json
```

The output file must not exist yet. The report shows its path, size, and SHA-256. Without `--output`, nothing is written; `--json` prints the whole package instead. The package (format `sextant-export/1`) holds every revision, event, operation, receipt, and registered content file of the task. It does not hold the files in `native-evidence/`. Save those with the companion script described under "Back up a runtime".

To import, wrap the exported object as `{"package": <export>}` and pass it as the payload:

```sh
sextant --root ./restored-runtime import --payload @import-payload.json
```

An import keeps the original IDs and history and grants no authority:

- The task arrives with status `needs_authority` and its permissions revoked.
- Imported conclusions are kept as history and shown as not verified by the receiving host.
- Imported observations become declared, and old predictions stay historical.
- Importing the same package again returns `already_imported`.

To let the imported task run again, grant permissions explicitly:

```sh
sextant --root ./restored-runtime authorize MISSION_ID --expected-revision CURRENT_REVISION --payload '{"action": "update", "authorization": {"status": "active"}}'
```

Use `{"action": "revoke"}` to take permissions away. Add `tools`, `send_to`, or `read_paths` to the `authorization` object to grant more. See `sextant guide runtime`.

## Move older records into a task

Records from the older decision-graph tools are read-only under `sextant legacy`. Wrap them in the legacy import format to make a task:

```json
{"package": {"format": "sextant-legacy-import/1", "goal": "Review an old decision", "graph": {}, "ledger_lines": ["{\"historical\": true}"]}}
```

```sh
sextant --root ./my-runtime import --payload @legacy-import.json --json
```

The graph and the ledger lines are stored whole as declared observations. Old thresholds and gate stamps are kept as historical diagnostics only. The new task starts with its permissions revoked, as any import does.

## Back up a runtime

```sh
sextant --root ./my-runtime backup ./backup-new-directory
```

The destination must be a new directory. The backup takes a consistent SQLite snapshot, copies and checks every content file, and writes `backup-manifest.json` when it is complete. An interrupted backup leaves a `backup.incomplete` marker, and Sextant refuses to open that directory as a runtime. Make a new backup from the original instead.

The backup does not include `native-evidence/`. To keep the native audit files, stop all writers, copy the packaged examples, and save a companion bundle from the original runtime directory:

```sh
sextant examples ./recovery-tools
python ./recovery-tools/preserve_native_evidence.py preserve --runtime ./my-runtime --bundle ./native-audit-backup
python ./recovery-tools/preserve_native_evidence.py verify --bundle ./native-audit-backup
```

The script only reads the runtime. It never calls a model or changes a task. `"audit_availability": "complete"` means every audit file that a saved receipt names was found and copied. Keep the bundle with the backup made at the same time.

## Recover from errors

- `expected_revision_required`, `revision_required`, or `conflict`: the task changed or you gave no revision. Run `inspect`, check that your change still applies, and retry with the current revision and a new `--command-id`.
- `in_progress`: a command with the same ID is still running. Check that the earlier process stopped before you do anything else.
- `reconciliation_required`: an external call was sent and its outcome is unknown. `reconcile` asks the adapter for the actual result. Nothing is resent automatically.
- `damaged_evidence`: an input observation's content no longer verifies, and the task is blocked. Restore the original data and run again, or withdraw the observation and continue without it.
- `missing_artifact` or `corrupt_artifact`: a content file is missing or does not match its hash. Restore it from a backup.

## Upgrade with care

A task records a fingerprint of the installed package code. After an upgrade, its derived objects show as not current, and the next `run` compiles again, which can call models. Observations, receipts, history, and exports are not affected. `run` with `--payload '{"mode": "recompute"}'` runs local checks on the saved proposal without a model call, and it does not promote a new policy. `{"mode": "reexecute"}` asks for new model calls. Back up or export a task before upgrading if you need its conclusions unchanged.
