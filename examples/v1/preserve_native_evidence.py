"""Preserve product-owned native audit files beside a Sextant state backup.

Stop runtime writers first. This companion does not call a provider, restore lost
bytes, copy native homes, or alter historical receipts. Verify works after moving
the bundle and never opens original receipt paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())


def plain_path(path: Path) -> Path:
    if '..' in path.parts:
        raise ValueError('parent traversal is not supported')
    result = path.absolute()
    if any(part.is_symlink() for part in (result, *result.parents)):
        raise ValueError('symlinks are not supported')
    return result


def references(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    isolation = receipt.get('isolation') or {}
    raw = isolation.get('raw_artifacts') or {}
    values = [(f'raw_artifacts/{key}', value) for key, value in raw.items()]
    values += [(f'matching_fresh_rollouts/{index}', value)
               for index, value in enumerate(isolation.get('matching_fresh_rollouts') or [])]
    observation = isolation.get('client_configuration_observation')
    if observation:
        values.append(('client_configuration_observation', observation))
    result = []
    for role, value in values:
        if not isinstance(value, dict) or not isinstance(value.get('path'), str):
            result.append({'role': role, 'status': 'unavailable_locator', 'record': value})
        else:
            result.append({'role': role, 'original_locator': value['path'],
                           'expected_sha256': value.get('sha256')})
    return result


def assess(ref: dict[str, Any], indexed: dict[str, Any], evidence: Path) -> dict[str, Any]:
    item = indexed.get(ref.get('original_locator', ''))
    if 'status' not in ref:
        locator = Path(ref['original_locator'])
        if not locator.is_absolute() or '..' in locator.parts or not locator.is_relative_to(evidence):
            ref['status'] = 'outside_explicit_evidence_root'
        elif item is None:
            ref['status'] = 'missing'
        elif ref['expected_sha256'] != item['sha256']:
            ref['status'] = 'hash_mismatch'
        else:
            ref.update(status='present', bundle_relative_path=item['bundle_relative_path'], bytes=item['bytes'])
    return ref


def declared_no_send(state: str, receipt: dict[str, Any] | None) -> bool:
    """Recognized coherent declaration only; never infer no-send from free text alone."""
    if not receipt or state != 'failed' or receipt.get('status') != 'failed':
        return False
    tool_state = receipt.get('tool_state')
    if not isinstance(tool_state, str) or not tool_state.startswith('not sent:'):
        return False
    isolation = receipt.get('isolation') or {}
    if ('request_state' in isolation and isolation['request_state'] != 'not_sent') or (
            'actual_native_sends' in isolation and
            (type(isolation['actual_native_sends']) is not int or isolation['actual_native_sends'] != 0)):
        return False  # A no-send phrase cannot override contradictory or unknown send metadata.
    declared_zero = (isolation.get('request_state') == 'not_sent'
                     and type(isolation.get('actual_native_sends')) is int
                     and isolation['actual_native_sends'] == 0)
    errors = receipt.get('errors') or []
    runtime_declaration = (type(errors) is list and any(isinstance(error, str)
                           and '(no external request was sent)' in error for error in errors))
    return declared_zero or runtime_declaration


def scan_error(error: OSError) -> None:
    raise error  # Keep the partial marker; unreferenced sidecars matter too.


def strict_json(data: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def nonfinite(value: str) -> Any:
        raise ValueError(f'nonfinite JSON value: {value}')

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def canonical_receipt(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def standalone_receipts(saved: dict[str, Any], indexed: dict[str, Any], evidence: Path,
                        bundle: Path) -> list[dict[str, Any]]:
    """Require the known Codex invocation's sidecar without opening original paths."""
    if saved['executor'] != 'codex' or not saved['receipt']:
        return []
    captured = references(saved['receipt'])
    if not captured:
        return []  # No-send declarations and unestablished captures are assessed separately.
    directories = set()
    for ref in captured:
        if not ref['role'].startswith('raw_artifacts/') or 'original_locator' not in ref:
            continue
        locator = Path(ref['original_locator'])
        if locator.is_absolute() and '..' not in locator.parts and locator.is_relative_to(evidence):
            directories.add(locator.parent)
    base = {'attempt_id': saved['attempt_id'], 'role': 'standalone_receipt',
            'expected_receipt_sha256': sha(canonical_receipt(saved['receipt']))}
    if not directories:
        return [{**base, 'status': 'unestablished_invocation_directory'}]
    result = []
    for directory in sorted(directories):
        ref = {**base, 'original_locator': str(directory / 'receipt.json')}
        item = indexed.get(ref['original_locator'])
        if item is None:
            ref['status'] = 'missing'
        else:
            ref.update(bundle_relative_path=item['bundle_relative_path'], bytes=item['bytes'], sha256=item['sha256'])
            data = plain_path(bundle / item['bundle_relative_path']).read_bytes()
            try:
                actual = canonical_receipt(strict_json(data))
            except (ValueError, UnicodeError):
                ref['status'] = 'invalid_receipt_json'
            else:
                ref['status'] = 'present' if actual == canonical_receipt(saved['receipt']) else 'receipt_mismatch'
        result.append(ref)
    return result


def preserve(runtime: Path, bundle: Path, evidence_root: Path | None = None) -> dict[str, Any]:
    runtime, bundle = plain_path(runtime), plain_path(bundle)
    evidence = plain_path(evidence_root or runtime / 'native-evidence')
    if bundle.is_relative_to(evidence) or evidence.is_relative_to(bundle):
        raise ValueError('bundle and original evidence must have separate roots')
    database = plain_path(runtime / 'state.sqlite3')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        active = db.execute("SELECT attempt_id FROM attempts WHERE state='in_flight'").fetchall()
        if active:
            raise ValueError('in-flight attempts remain; stop/reconcile writers before preservation')
        rows = db.execute('SELECT attempt_id,executor,state,receipt FROM attempts ORDER BY attempt_id').fetchall()
        heads = db.execute('SELECT mission_id,revision FROM missions ORDER BY mission_id').fetchall()
        db.rollback()
    bundle.mkdir(parents=True, exist_ok=False)
    marker = bundle / 'native-evidence.incomplete'
    marker.write_text('Preservation has not completed. Keep the original runtime.\n')
    files = []
    if evidence.exists():
        if not evidence.is_dir():
            raise ValueError('evidence root must be a product-owned directory')
        for parent, directories, names in os.walk(evidence, followlinks=False, onerror=scan_error):
            for name in directories:
                plain_path(Path(parent) / name)
            for name in sorted(names):
                source = plain_path(Path(parent) / name)
                if not source.is_file():
                    raise ValueError('native evidence contains a non-regular file')
                relative = source.relative_to(evidence)
                destination = bundle / 'files' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = source.read_bytes()
                with destination.open('xb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if source.read_bytes() != data or destination.read_bytes() != data:
                    raise ValueError('native evidence changed during preservation')
                files.append({'original_locator': str(source), 'bundle_relative_path': str(destination.relative_to(bundle)),
                              'sha256': sha(data), 'bytes': len(data), 'role': name})
    indexed = {item['original_locator']: item for item in files}
    saved_receipts, trace_refs, unavailable = [], [], []
    for attempt, executor, state, encoded in rows:
        receipt = json.loads(encoded) if encoded else None
        captured = references(receipt) if receipt else []
        if executor == 'codex' or captured:
            saved_receipts.append({'attempt_id': attempt, 'executor': executor, 'state': state, 'receipt': receipt})
        if not captured:
            if executor == 'codex' and not declared_no_send(state, receipt):
                unavailable.append({'attempt_id': attempt, 'state': state, 'reason': 'native raw capture unestablished'})
            continue
        for ref in captured:
            ref['attempt_id'] = attempt
            trace_refs.append(assess(ref, indexed, evidence))
    receipts = bundle / 'original-receipts.json'
    write(receipts, saved_receipts)
    sidecars = [ref for saved in saved_receipts for ref in standalone_receipts(saved, indexed, evidence, bundle)]
    manifest = {'protocol': 'sextant-native-evidence-companion/2', 'runtime_original_locator': str(runtime),
                'evidence_original_locator': str(evidence), 'state_heads': heads,
                'scope': 'product-owned native evidence only; state backup and native homes are excluded',
                'historical_receipts_unchanged': True, 'files': files, 'references': trace_refs,
                'standalone_receipts': sidecars,
                'unavailable_captures': unavailable, 'receipt_file_sha256': sha(receipts.read_bytes()),
                'audit_availability': 'complete' if not unavailable and all(r['status'] == 'present' for r in trace_refs + sidecars) else 'incomplete'}
    write(bundle / 'native-evidence-manifest.json', manifest)
    verify(bundle, allow_marker=True)
    marker.unlink()
    return manifest


def verify(bundle: Path, *, allow_marker: bool = False) -> dict[str, Any]:
    bundle = plain_path(bundle)
    if (bundle / 'native-evidence.incomplete').exists() and not allow_marker:
        raise ValueError('companion preservation was interrupted; keep the original and create a new bundle')
    manifest = json.loads(plain_path(bundle / 'native-evidence-manifest.json').read_bytes())
    if manifest['protocol'] != 'sextant-native-evidence-companion/2':
        raise ValueError('unsupported companion protocol; keep the old bundle and create a new /2 bundle from surviving originals')
    for item in manifest['files']:
        relative = Path(item['bundle_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid bundle-relative path')
        data = plain_path(bundle / relative).read_bytes()
        if sha(data) != item['sha256'] or len(data) != item['bytes']:
            raise ValueError('companion file is missing or differs from its inventory')
    receipt_data = plain_path(bundle / 'original-receipts.json').read_bytes()
    if sha(receipt_data) != manifest['receipt_file_sha256']:
        raise ValueError('preserved original receipt inventory differs')
    indexed = {item['original_locator']: item for item in manifest['files']}
    expected_refs, expected_unavailable, expected_sidecars = [], [], []
    for saved in strict_json(receipt_data):
        captured = references(saved['receipt']) if saved['receipt'] else []
        if not captured and saved['executor'] == 'codex' and not declared_no_send(saved['state'], saved['receipt']):
            expected_unavailable.append({'attempt_id': saved['attempt_id'], 'state': saved['state'],
                                         'reason': 'native raw capture unestablished'})
        for ref in captured:
            ref['attempt_id'] = saved['attempt_id']
            expected_refs.append(assess(ref, indexed, Path(manifest['evidence_original_locator'])))
        expected_sidecars.extend(standalone_receipts(saved, indexed, Path(manifest['evidence_original_locator']), bundle))
    available = 'complete' if not expected_unavailable and all(r['status'] == 'present' for r in expected_refs + expected_sidecars) else 'incomplete'
    if (expected_refs != manifest['references'] or expected_unavailable != manifest['unavailable_captures']
            or expected_sidecars != manifest['standalone_receipts']
            or available != manifest['audit_availability']):
        raise ValueError('companion availability differs from its original receipt references')
    return {'copied_files_verified': len(manifest['files']), 'audit_availability': manifest['audit_availability'],
            'missing_or_invalid_references': [ref for ref in manifest['references'] if ref['status'] != 'present'],
            'missing_or_invalid_standalone_receipts': [ref for ref in expected_sidecars if ref['status'] != 'present'],
            'unavailable_captures': manifest['unavailable_captures'], 'effects_executed': 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    copy = commands.add_parser('preserve', help='create a new companion after stopping runtime writers')
    copy.add_argument('--runtime', required=True, type=Path)
    copy.add_argument('--bundle', required=True, type=Path)
    copy.add_argument('--evidence-root', type=Path, help='explicit product-owned root for a custom native port only')
    check = commands.add_parser('verify', help='verify a relocated companion without original paths')
    check.add_argument('--bundle', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'preserve':
        preserve(args.runtime, args.bundle, args.evidence_root)
    result = verify(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['audit_availability'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
