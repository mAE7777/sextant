"""Deterministic question-scoped Jev views; raw mission evidence is never pruned.

This is a conservative structural selection, not a semantic relevance oracle.
Unscoped inputs, source peers and explicit joint/dependency links cannot be
dropped from a selected component. Cross-component completeness stays unknown.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sextant.domain import ObjectRef, Revision, canonical, digest

SELECTION_PROTOCOL = "jev-bounded-context/1"


@dataclass
class ContextSelection:
    pack: dict[str, Any]
    manifest: dict[str, Any]
    refs: list[ObjectRef]
    failure: str | None = None


def _scope(row: Revision) -> str | None:
    stated = row.payload.get("supports_scope")
    if not isinstance(stated, str) or stated.strip().lower() in ("", "unknown", "whole", "*"):
        return None
    return canonical({"supports_scope": stated, "subject": row.scope.subject,
                      "domain": row.scope.domain, "environment": row.scope.environment})


def _references(value: Any, aliases: dict[str, set[str]]) -> tuple[set[str], set[str]]:
    found: set[str] = set()
    missing: set[str] = set()
    if isinstance(value, dict):
        if {"object_id", "revision_id", "object_kind"}.issubset(value):
            try:
                key = canonical(ObjectRef.model_validate(value))
            except ValueError:
                return set(), {"malformed-observation-reference"}
            if key in aliases:
                if len(aliases[key]) == 1:
                    found.update(aliases[key])
                else:
                    missing.add("ambiguous:" + key)
            elif value["object_kind"] == "observation":
                missing.add(key)
        else:
            for item in value.values():
                yes, no = _references(item, aliases)
                found.update(yes)
                missing.update(no)
    elif isinstance(value, list):
        for item in value:
            yes, no = _references(item, aliases)
            found.update(yes)
            missing.update(no)
    elif isinstance(value, str):
        if value in aliases:
            if len(aliases[value]) == 1:
                found.update(aliases[value])
            else:
                missing.add("ambiguous:" + value)
        elif value.startswith(("object:", "external:")):
            missing.add(value)
    return found, missing


def select_context(card_id: str, pack: dict[str, Any], records: list[Revision], *,
                   bounded: bool, validity: dict[str, Any] | None = None) -> ContextSelection:
    """Select one explicit evidence closure, or one declared-scope component.

    No recency, model score or favorable-answer ordering is used. An arbitrary
    component is only a local diagnostic focus; it cannot certify the mission.
    The caller must capacity-check the final state AND actual rendered questions.
    """
    by_id = {row.ref.object_id: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("context must contain unique current observation heads")
    aliases: dict[str, set[str]] = {}
    for row in records:
        names = [row.ref.object_id, "object:" + row.ref.object_id, canonical(row.ref)]
        if row.payload.get("external_observation_id"):
            external = str(row.payload["external_observation_id"])
            names.extend([external, "external:" + external])
        # Only a withdrawn head can answer an old citation as a withdrawal
        # notice. An active correction is not the exact historical evidence.
        if row.payload.get("status") == "withdrawn":
            if row.predecessor is not None:
                names.append(canonical(row.predecessor))
            for old in row.payload.get("supersedes", []):
                if isinstance(old, dict):
                    names.append(canonical(ObjectRef.model_validate(old)))
        for name in names:
            aliases.setdefault(name, set()).add(row.ref.object_id)

    proposal = pack["proposal"]
    # R2's all-input context is a view description, not evidence that every raw
    # observation is necessary for each local question. Rebuild only that view.
    context_view = card_id == "R2" and isinstance(proposal, dict) and set(proposal) == {"context"}
    roots, missing = _references({} if context_view else proposal, aliases)
    required, absent = _references({key: pack[key] for key in ("task", "evaluation", "access")}, aliases)
    roots.update(required)
    missing.update(absent)

    def links_for(row: Revision) -> tuple[set[str], set[str]]:
        return _references([*[r.model_dump() for r in row.dependencies],
                            row.payload.get("supersedes", []), row.payload.get("content")], aliases)

    mode = "all-supplied-material"
    selected = set(by_id)
    if bounded:
        if roots:
            mode = "explicit-proposal-evidence-closure"
            selected = set(roots)
        else:
            mode = "declared-scope-component-only"
            scope_keys = sorted({key for row in records if (key := _scope(row)) is not None})
            selected = {row.ref.object_id for row in records if _scope(row) in ({scope_keys[0]} if scope_keys else {None})}
        selected.update(row.ref.object_id for row in records if _scope(row) is None)
        while True:
            before = set(selected)
            selected_scopes = {_scope(by_id[key]) for key in selected}
            sources = {str(by_id[key].payload.get("source_group")) for key in selected
                       if by_id[key].payload.get("source_group") not in (None, "", "unknown")}
            source_revisions = {str(by_id[key].payload["source_revision"]) for key in selected
                                if by_id[key].payload.get("source_revision")}
            for key, row in by_id.items():
                if (_scope(row) in selected_scopes
                        or str(row.payload.get("source_group")) in sources
                        or (row.payload.get("source_revision") and str(row.payload["source_revision"]) in source_revisions)):
                    selected.add(key)
            # Dependencies are undirected here: a counter-observation referring
            # back to selected evidence is as important as forward support.
            for key, row in by_id.items():
                links, absent = links_for(row)
                if key in selected or links.intersection(selected):
                    selected.add(key)
                    selected.update(links)
                    missing.update(absent)
            if before == selected:
                break
    # Capacity is not completeness: every selected record's explicit references
    # must resolve on the full-pack path as well as the bounded path.
    for key in selected:
        _, absent = links_for(by_id[key])
        missing.update(absent)
        if (validity or {}).get(key, {}).get("valid") is False:
            missing.add("invalid-current-observation:" + key)

    chosen = sorted((by_id[key] for key in selected), key=lambda row: (row.ref.object_id, row.ref.revision_id))
    omitted = sorted((row for key, row in by_id.items() if key not in selected),
                     key=lambda row: (row.ref.object_id, row.ref.revision_id))

    def item(row: Revision, reason: str) -> dict[str, Any]:
        fields = sorted(row.payload) if row.payload.get("status") == "active" else [
            key for key in ("source", "source_group", "source_revision", "status", "supports_scope",
                            "supersedes", "correction_reason", "observed_at", "occurred_at") if key in row.payload]
        return {"ref": row.ref.model_dump(), "payload_sha256": digest(row.payload),
                "scope": row.scope.model_dump(), "supports_scope": row.payload.get("supports_scope"),
                "status": row.payload.get("status"), "source": row.payload.get("source"),
                "source_group": row.payload.get("source_group"), "source_revision": row.payload.get("source_revision"),
                "dependencies": [r.model_dump() for r in row.dependencies],
                "predecessor": row.predecessor.model_dump() if row.predecessor else None,
                "reference_aliases": sorted(name for name, targets in aliases.items() if row.ref.object_id in targets),
                "historical_alias_scope": "withdrawal attribution only; never active historical content" if row.payload.get("status") == "withdrawn" else "current exact revision only",
                "artifacts": [a.model_dump() for a in row.artifacts], "reason": reason,
                "included_payload_fields": fields if row.ref.object_id in selected else [],
                "omitted_payload_fields": sorted(set(row.payload) - set(fields)) if row.ref.object_id in selected else sorted(row.payload),
                "current_validity": (validity or {}).get(row.ref.object_id, {"valid": None, "basis": "not supplied to this pure selector"})}

    loss = (["Only this declared diagnostic component is judged; omitted material and cross-component interactions remain unassessed.",
             "No whole-task sufficiency, absence-of-counterevidence, correctness or execution authority follows from this view."]
            if omitted else ["All supplied authorized current records are included; world completeness is not established."])
    scope = {"method": mode, "supports_scopes": sorted({_scope(row) or "unknown" for row in chosen}),
             "subject": "the supplied proposal and selected evidence only",
             "whole_mission_completeness": False, "omitted_material_assessed": False}
    manifest = {"protocol": SELECTION_PROTOCOL, "card_id": card_id, "bounded_selection": bounded,
                "original_pack_sha256": digest(pack), "original_proposal_sha256": digest(proposal),
                "judgment_scope": scope, "selected": [item(row, "required selected component closure") for row in chosen],
                "omitted": [item(row, "outside this explicit diagnostic component; no judgment of irrelevance") for row in omitted],
                "unresolved_references": sorted(missing), "loss": loss,
                "selection_rule": "explicit citations else canonical first declared scope; unscoped inputs, same-scope/source and bidirectional dependency closure"}
    result = deepcopy(pack)
    result["evidence"] = []
    for row in chosen:
        payload = row.payload
        if payload.get("status") != "active":
            # Preserve withdrawal/correction attribution without presenting old
            # content as active evidence. Full raw bytes remain in the Store.
            payload = {key: payload[key] for key in ("source", "source_group", "source_revision", "status", "supports_scope",
                        "supersedes", "correction_reason", "observed_at", "occurred_at") if key in payload}
        result["evidence"].append({"ref": row.ref.model_dump(), "scope": row.scope.model_dump(),
                                   "observation": deepcopy(payload), "dependencies": [r.model_dump() for r in row.dependencies],
                                   "current_validity": (validity or {}).get(row.ref.object_id, {"valid": None})})
    if context_view:
        selected_refs = [row.ref.model_dump() for row in chosen if row.payload.get("status") == "active"]
        result["proposal"] = {"context": {"selected": selected_refs,
            "joint_dependencies": [selected_refs] if len(selected_refs) > 1 else [],
            "counterevidence": selected_refs, "loss": loss,
            "scope": "selected component only; this is not the original all-history context"}}
    result["evaluation"] = {**result["evaluation"], "context_selection": {
        "protocol": SELECTION_PROTOCOL, "manifest_sha256": digest(manifest),
        "selected_count": len(chosen), "omitted_count": len(omitted), "judgment_scope": scope, "loss": loss,
        "unresolved_reference_count": len(missing), "manifest_read_by_evaluator": False}}
    failure = "Required observation references are missing or ambiguous; bounded diagnostic is unavailable" if missing else None
    return ContextSelection(result, manifest, [row.ref for row in chosen], failure)
