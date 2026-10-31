"""Portable original native bytes, explicit loss, and interrupted-copy boundaries."""
import importlib.util
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('native_companion', Path(__file__).resolve().parents[1] / 'examples/v1/preserve_native_evidence.py')
assert SPEC and SPEC.loader
COMPANION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPANION)


def source(tmp_path, *, state='completed', missing=False, mismatch=False, outside=False):
    runtime = tmp_path / 'original'
    evidence = runtime / 'native-evidence/invocation-example'
    evidence.mkdir(parents=True)
    raw = evidence / 'rollout-0.jsonl'
    raw.write_bytes(b'{"type":"session_meta","payload":{"id":"fixture"}}\n')
    context = evidence / 'turn-context.json'
    context.write_bytes(b'{"type":"turn_context","payload":{"model":"fixture"}}\n')
    receipt = {'executor': 'codex', 'isolation': {'raw_artifacts': {
        'rollout-0.jsonl': {'path': str(raw), 'sha256': '0' * 64 if mismatch else COMPANION.sha(raw.read_bytes())}},
        'client_configuration_observation': {'path': str(context), 'sha256': COMPANION.sha(context.read_bytes())},
        'matching_fresh_rollouts': [{'path': str(raw), 'sha256': COMPANION.sha(raw.read_bytes())}]}}
    if outside:
        receipt['isolation']['raw_artifacts']['outside'] = {'path': str(tmp_path / 'do-not-read'), 'sha256': '1' * 64}
    (evidence / 'receipt.json').write_text(json.dumps(receipt))
    if missing:
        raw.unlink()
    with sqlite3.connect(runtime / 'state.sqlite3') as db:
        db.execute('CREATE TABLE attempts(attempt_id TEXT,executor TEXT,state TEXT,receipt TEXT)')
        db.execute('CREATE TABLE missions(mission_id TEXT,revision TEXT)')
        db.execute('INSERT INTO attempts VALUES(?,?,?,?)', ('attempt-fixture', 'codex', state, json.dumps(receipt)))
        db.execute('INSERT INTO missions VALUES(?,?)', ('mission-fixture', 'original-revision'))
    return runtime, evidence


def test_relocated_raw_trace_verifies_without_original_paths(tmp_path):
    runtime, evidence = source(tmp_path)
    before = (runtime / 'state.sqlite3').read_bytes()
    bundle = tmp_path / 'bundle'
    manifest = COMPANION.preserve(runtime, bundle)
    assert manifest['audit_availability'] == 'complete'
    assert {f['role'] for f in manifest['files']} == {'rollout-0.jsonl', 'turn-context.json', 'receipt.json'}
    assert len(manifest['references']) == 3
    assert (runtime / 'state.sqlite3').read_bytes() == before
    relocated = tmp_path / 'relocated'
    shutil.move(bundle, relocated)
    shutil.rmtree(runtime)
    assert not evidence.exists()
    assert COMPANION.verify(relocated)['audit_availability'] == 'complete'


@pytest.mark.parametrize('defect,expected', [('missing', 'missing'), ('mismatch', 'hash_mismatch'), ('outside', 'outside_explicit_evidence_root')])
def test_historical_loss_stays_explicit_without_reconstruction(tmp_path, defect, expected):
    runtime, _ = source(tmp_path, **{defect: True})
    bundle = tmp_path / 'bundle'
    manifest = COMPANION.preserve(runtime, bundle)
    assert manifest['audit_availability'] == 'incomplete'
    checked = COMPANION.verify(bundle)
    assert expected in {ref['status'] for ref in checked['missing_or_invalid_references']}
    assert checked['effects_executed'] == 0


def test_corrupt_copied_rollout_is_not_a_verified_trace(tmp_path):
    runtime, _ = source(tmp_path)
    bundle = tmp_path / 'bundle'
    COMPANION.preserve(runtime, bundle)
    (bundle / 'files/invocation-example/rollout-0.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        COMPANION.verify(bundle)


def test_inflight_is_not_a_quiescent_backup(tmp_path):
    runtime, _ = source(tmp_path, state='in_flight')
    bundle = tmp_path / 'bundle'
    with pytest.raises(ValueError, match='in-flight'):
        COMPANION.preserve(runtime, bundle)
    assert not bundle.exists()


def test_symlink_is_rejected_without_reading_target(tmp_path):
    runtime, evidence = source(tmp_path)
    (evidence / 'unexpected').symlink_to(tmp_path / 'never-open')
    bundle = tmp_path / 'bundle'
    with pytest.raises(ValueError, match='symlinks'):
        COMPANION.preserve(runtime, bundle)
    assert (bundle / 'native-evidence.incomplete').exists()
    with pytest.raises(ValueError, match='interrupted'):
        COMPANION.verify(bundle)


def test_false_availability_label_does_not_hide_missing_original(tmp_path):
    runtime, _ = source(tmp_path, missing=True)
    bundle = tmp_path / 'bundle'
    COMPANION.preserve(runtime, bundle)
    manifest_path = bundle / 'native-evidence-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['audit_availability'] = 'complete'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='availability differs'):
        COMPANION.verify(bundle)


def test_unreferenced_directory_scan_failure_keeps_partial_marker(tmp_path, monkeypatch):
    runtime, evidence = source(tmp_path)
    hidden = evidence / 'ordinary-sidecar'
    hidden.mkdir()
    (hidden / 'receipt.json').write_text('{"unreferenced":"preserve original"}')
    original_walk = COMPANION.os.walk

    def interrupted_walk(path, *, followlinks, onerror=None):
        for index, row in enumerate(original_walk(path, followlinks=followlinks)):
            yield row
            if index == 0:
                error = PermissionError('inert scan failure; no real permission change')
                if onerror:
                    onerror(error)
                return

    monkeypatch.setattr(COMPANION.os, 'walk', interrupted_walk)
    bundle = tmp_path / 'bundle'
    with pytest.raises(PermissionError, match='inert scan failure'):
        COMPANION.preserve(runtime, bundle)
    assert (bundle / 'native-evidence.incomplete').exists()
    assert not (bundle / 'native-evidence-manifest.json').exists()


def uncaptured(runtime, state, receipt):
    with sqlite3.connect(runtime / 'state.sqlite3') as db:
        db.execute('UPDATE attempts SET state=?,receipt=?', (state, json.dumps(receipt)))


@pytest.mark.parametrize('state', ['unknown', 'completed', 'failed'])
def test_bare_not_sent_text_does_not_hide_unestablished_capture(tmp_path, state):
    runtime, _ = source(tmp_path)
    receipt = {'executor': 'codex', 'status': state, 'tool_state': 'not sent: arbitrary text', 'isolation': {}}
    uncaptured(runtime, state, receipt)
    bundle = tmp_path / 'bundle'
    manifest = COMPANION.preserve(runtime, bundle)
    assert manifest['audit_availability'] == 'incomplete'
    assert manifest['unavailable_captures'][0]['state'] == state
    assert json.loads((bundle / 'original-receipts.json').read_bytes())[0]['receipt'] == receipt
    assert COMPANION.verify(bundle)['audit_availability'] == 'incomplete'


@pytest.mark.parametrize('evidence', [
    {'isolation': {'request_state': 'not_sent', 'actual_native_sends': 0}},
    {'errors': ['qualified host refusal (no external request was sent)']},
])
def test_coherent_failed_no_send_declaration_preserved_without_expected_raw_capture(tmp_path, evidence):
    runtime, _ = source(tmp_path)
    receipt = {'executor': 'codex', 'status': 'failed', 'tool_state': 'not sent: host refusal', **evidence}
    uncaptured(runtime, 'failed', receipt)
    bundle = tmp_path / 'bundle'
    manifest = COMPANION.preserve(runtime, bundle)
    assert manifest['audit_availability'] == 'complete'
    assert json.loads((bundle / 'original-receipts.json').read_bytes())[0]['receipt'] == receipt
    assert COMPANION.verify(bundle)['audit_availability'] == 'complete'


def test_boolean_zero_send_alias_is_not_a_coherent_count(tmp_path):
    runtime, _ = source(tmp_path)
    receipt = {'executor': 'codex', 'status': 'failed', 'tool_state': 'not sent: text',
               'isolation': {'request_state': 'not_sent', 'actual_native_sends': False}}
    uncaptured(runtime, 'failed', receipt)
    manifest = COMPANION.preserve(runtime, tmp_path / 'bundle')
    assert manifest['audit_availability'] == 'incomplete'


@pytest.mark.parametrize('metadata', [
    {'request_state': 'attempted', 'actual_native_sends': 0},
    {'request_state': 'not_sent', 'actual_native_sends': 1},
    {'request_state': 'not_sent', 'actual_native_sends': False},
    {'request_state': None, 'actual_native_sends': 0},
])
def test_no_send_error_phrase_cannot_override_contradictory_or_unknown_send_metadata(tmp_path, metadata):
    runtime, _ = source(tmp_path)
    receipt = {'executor': 'codex', 'status': 'failed', 'tool_state': 'not sent: runtime refusal',
               'isolation': metadata, 'errors': ['runtime declaration (no external request was sent)']}
    uncaptured(runtime, 'failed', receipt)
    bundle = tmp_path / 'bundle'
    manifest = COMPANION.preserve(runtime, bundle)
    assert manifest['audit_availability'] == 'incomplete'
    assert json.loads((bundle / 'original-receipts.json').read_bytes())[0]['receipt'] == receipt
    assert COMPANION.verify(bundle)['audit_availability'] == 'incomplete'


def test_verify_recomputes_uncaptured_native_availability_from_saved_receipts(tmp_path):
    runtime, _ = source(tmp_path)
    receipt = {'executor': 'codex', 'status': 'unknown', 'tool_state': 'unestablished capture'}
    uncaptured(runtime, 'unknown', receipt)
    bundle = tmp_path / 'bundle'
    COMPANION.preserve(runtime, bundle)
    path = bundle / 'native-evidence-manifest.json'
    manifest = json.loads(path.read_bytes())
    manifest.update(unavailable_captures=[], audit_availability='complete')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='availability differs'):
        COMPANION.verify(bundle)


@pytest.mark.parametrize('defect,expected', [
    ('missing', 'missing'), ('foreign', 'receipt_mismatch'), ('integer_boolean', 'receipt_mismatch'),
    ('duplicate_key', 'invalid_receipt_json'), ('nonfinite', 'invalid_receipt_json'),
])
def test_standalone_receipt_loss_or_foreign_identity_stays_explicit_after_relocation(tmp_path, defect, expected):
    runtime, evidence = source(tmp_path)
    sidecar = evidence / 'receipt.json'
    original = json.loads(sidecar.read_bytes())
    if defect == 'missing':
        sidecar.unlink()
    elif defect == 'foreign':
        sidecar.write_text('{"executor":"codex","session":"unrelated invocation"}')
    elif defect == 'integer_boolean':
        original['capture_count'] = 1
        with sqlite3.connect(runtime / 'state.sqlite3') as db:
            db.execute('UPDATE attempts SET receipt=?', (json.dumps(original),))
        sidecar.write_text(json.dumps({**original, 'capture_count': True}))
    elif defect == 'duplicate_key':
        encoded = json.dumps(original)
        sidecar.write_text('{"executor":"foreign",' + encoded[1:])
    else:
        sidecar.write_text('{"capture_count":NaN}')
    bundle = tmp_path / 'bundle'
    manifest = COMPANION.preserve(runtime, bundle)
    assert manifest['audit_availability'] == 'incomplete'
    assert manifest['standalone_receipts'][0]['status'] == expected
    assert json.loads((bundle / 'original-receipts.json').read_bytes())[0]['receipt'] == original
    relocated = tmp_path / 'relocated'
    shutil.move(bundle, relocated)
    shutil.rmtree(runtime)
    verified = COMPANION.verify(relocated)
    assert verified['audit_availability'] == 'incomplete'
    assert verified['missing_or_invalid_standalone_receipts'][0]['status'] == expected
    assert verified['effects_executed'] == 0


def test_standalone_receipt_formatting_is_not_a_different_invocation(tmp_path):
    runtime, evidence = source(tmp_path)
    sidecar = evidence / 'receipt.json'
    original = json.loads(sidecar.read_bytes())
    sidecar.write_text(json.dumps(original, sort_keys=True, indent=4) + '\n')
    manifest = COMPANION.preserve(runtime, tmp_path / 'bundle')
    assert manifest['standalone_receipts'][0]['status'] == 'present'
    assert manifest['audit_availability'] == 'complete'


def test_false_standalone_receipt_status_is_recomputed_from_preserved_bytes(tmp_path):
    runtime, evidence = source(tmp_path)
    (evidence / 'receipt.json').unlink()
    bundle = tmp_path / 'bundle'
    COMPANION.preserve(runtime, bundle)
    path = bundle / 'native-evidence-manifest.json'
    manifest = json.loads(path.read_bytes())
    manifest['standalone_receipts'] = []
    manifest['audit_availability'] = 'complete'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='availability differs'):
        COMPANION.verify(bundle)


def test_older_companion_is_preserved_without_stronger_format_certification(tmp_path):
    runtime, _ = source(tmp_path)
    bundle = tmp_path / 'bundle'
    COMPANION.preserve(runtime, bundle)
    path = bundle / 'native-evidence-manifest.json'
    manifest = json.loads(path.read_bytes())
    manifest['protocol'] = 'sextant-native-evidence-companion/1'
    encoded = json.dumps(manifest).encode()
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match='keep the old bundle'):
        COMPANION.verify(bundle)
    assert path.read_bytes() == encoded
