"""The Jev client: one run per `ask`, every question of a role card in each request, a ledger
row for every call, and judgments derived from the run's recorded calls.

Discipline (from the plan, the spike and the wire protocol):
  - every request is composed by one versioned wire renderer, so the card's purpose and shared
    instructions, the question, its criteria and the state all reach the transport
  - every call of one `ask` carries the same `run_id`, its role in the run, and the re-test
    settings in force; a judgment is re-derived from all of the run's calls, never from a list
    the client chose, so an audit can re-derive it the same way
  - every raw answer is checked against the answer space its question declared before anything
    is derived from it; an invalid answer stays in the ledger and stops the run loudly
  - every yes/no question is asked `1 + reruns` times; every choice question over
    `permutations` option orders; the spread is recorded, never hidden
  - negation twins are asked in the same request; the gap is recorded, never enforced
  - the model id is pinned; the response's model id is recorded on every row
  - a question runs only with its sealed hash; a drifted or unhashed question is refused
  - the state pack is the card's named fields and nothing else; over-budget requests are refused
  - calibration authority comes only from a certificate that re-verifies against its evidence
  - a degraded card (rung != jev) is refused unless the caller explicitly allows it
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sextant import wire
from sextant.certificates import certificate_state
from sextant.jev_transport import JevResult, call_jev
from sextant.ledger import Ledger, new_id, now_iso, sha_of
from sextant.rolecards import QuestionSpec, RoleCard, compute_hash, require_sealed
from sextant.schema import (
    CalibrationStatus,
    ContributingCall,
    Estimate,
    Primitive,
    Rung,
    SourceTier,
    Stability,
    ThresholdRef,
)

Caller = Callable[[Any, dict[str, dict[str, Any]], str], JevResult]

OUTCOME_CARDS = frozenset({"R7"})
MAX_PAIRWISE_ITEMS = 8


class JevClientError(RuntimeError):
    pass


class RungError(JevClientError):
    pass


class JevAnswerError(JevClientError):
    """A transport answer that does not fit its declared answer space. The raw row is kept.

    `run_id` and `call_ids` name the run and every call it made before stopping, so a caller
    that records the failed attempt can point at the evidence.
    """

    def __init__(self, message: str, *, run_id: str = "", call_ids: list[str] | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.call_ids = list(call_ids or [])


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def estimate_tokens(obj: Any) -> int:
    return wire.estimate_tokens(obj)


def calibration_root_of(config: dict[str, Any]) -> Path | None:
    raw = (config.get("paths") or {}).get("calibration_dir")
    return Path(str(raw)) if raw else None


@dataclass
class RenderedQuestion:
    key: str
    base_key: str
    primitive: Primitive
    text: str
    sdk: dict[str, Any]
    template_hash: str
    rendered_hash: str
    role: str
    fmt: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionEstimate:
    key: str
    base_key: str
    estimate: Estimate
    raw: dict[str, Any]
    contested: bool = False
    runs: int = 1


@dataclass
class CardResult:
    card_id: str
    pack_sha: str
    model_id: str
    transport: str
    run_id: str = ""
    call_ids: list[str] = field(default_factory=list)
    estimates: dict[str, QuestionEstimate] = field(default_factory=dict)
    usage_input_tokens: int = 0
    requests: int = 0

    def value(self, key: str) -> float | None:
        qe = self.estimates.get(key)
        if qe is None:
            return None
        e = qe.estimate
        if e.primitive == Primitive.noul:
            return e.p
        if e.primitive == Primitive.score:
            return e.score
        if e.primitive == Primitive.choice and e.probabilities and e.choice:
            return e.probabilities.get(e.choice)
        return None

    def passes(self, key: str) -> bool | None:
        """The calibrated yes/no answer, or None when there is no authority to decide.

        Only an estimate whose certificate re-verified against its evidence decides, and only
        when its runs agreed; anything else abstains.
        """
        qe = self.estimates.get(key)
        if qe is None or qe.estimate.primitive != Primitive.noul:
            return None
        e = qe.estimate
        p = e.p
        ref = e.threshold_ref
        if p is None or e.calibration_status.state != "calibrated" or ref is None:
            return None
        return wire.decision(p, ref.value, contested=bool(e.contested or qe.contested))

    def keys_for(self, base_key: str) -> list[str]:
        return [k for k, qe in self.estimates.items() if qe.base_key == base_key]


class JevClient:
    def __init__(
        self,
        config: dict[str, Any],
        cards: dict[str, RoleCard],
        ledger: Ledger | None = None,
        caller: Caller | None = None,
        transport: str | None = None,
        seed: int | None = None,
        calibration_root: Path | None = None,
    ):
        """`transport` names where answers come from, and it must be true.

        With no caller the real transport runs and its label comes from config. An injected
        caller is a controlled transport and must be labelled `controlled:<name>`; letting a
        scripted caller claim a real label would let test answers certify live use.
        """
        self.config = config
        self.cards = cards
        self.ledger = ledger
        if caller is None:
            self.caller: Caller = call_jev
            self.transport = transport or str(config["jev"].get("transport", "typesafe"))
            if self.transport.startswith("controlled:"):
                raise JevClientError("a controlled transport label needs the controlled caller that produces it")
        else:
            self.caller = caller
            self.transport = transport or "controlled:injected"
            if not self.transport.startswith("controlled:"):
                raise JevClientError(
                    f"an injected caller cannot claim the real transport {self.transport!r}; "
                    "label it controlled:<name>"
                )
        self.model_id = str(config["jev"]["model_id"])
        self.max_state_tokens = int(config["jev"]["max_state_tokens"])
        self.max_request_tokens = int(config["jev"]["max_request_tokens"])
        self.language = str(config["jev"].get("canonical_language", "en"))
        self.rpm = int(config["jev"].get("requests_per_minute", 1000))
        self.calibration_root = calibration_root if calibration_root is not None else calibration_root_of(config)
        self._last_call = 0.0
        self.rng = random.Random(seed)

    # ------------------------------------------------------------ rendering

    def _list_field(self, card: RoleCard, pack: dict[str, Any]) -> tuple[str, list[Any]] | None:
        for f in card.state_pack:
            v = pack.get(f)
            if isinstance(v, list):
                return f, v
        return None

    def _sdk_for(
        self,
        card: RoleCard,
        q: QuestionSpec,
        wire_key: str,
        text: str,
        fmt: dict[str, Any],
        leaf_space: dict[str, Any] | None,
    ) -> tuple[Primitive, dict[str, Any]]:
        """The exact payload for one question. `instructions` is the composed wire prompt."""
        sent = wire.compose_instructions(card, wire_key, text, language=self.language)
        primitive = q.primitive
        if q.role == "main" and leaf_space and q.text.strip() == "{question}":
            primitive = Primitive(leaf_space["primitive"])
            if primitive == Primitive.choice:
                return primitive, {"type": "choice", "instructions": sent, "criteria": dict(leaf_space["options"])}
            if primitive == Primitive.score:
                return primitive, {"type": "score", "instructions": sent, "criteria": list(leaf_space["levels"])}
        if primitive == Primitive.noul:
            crit = None
            if q.criteria_true and q.criteria_false:
                crit = {"true": q.criteria_true.format_map(fmt), "false": q.criteria_false.format_map(fmt)}
            d: dict[str, Any] = {"type": "noul", "instructions": sent}
            if crit:
                d["criteria"] = crit
            return primitive, d
        if primitive == Primitive.choice:
            return primitive, {"type": "choice", "instructions": sent, "criteria": dict(q.options or {})}
        if primitive == Primitive.score:
            return primitive, {"type": "score", "instructions": sent, "criteria": list(q.levels or [])}
        raise JevClientError(f"unsupported primitive {primitive}")

    def _rendered(
        self,
        card: RoleCard,
        q: QuestionSpec,
        wire_key: str,
        base_key: str,
        text: str,
        fmt: dict[str, Any],
        template_hash: str,
        leaf_space: dict[str, Any] | None = None,
    ) -> RenderedQuestion:
        prim, sdk = self._sdk_for(card, q, wire_key, text, fmt, leaf_space)
        sdk = wire.ordered_payload(sdk)
        return RenderedQuestion(
            wire_key,
            base_key,
            prim,
            text,
            sdk,
            template_hash,
            wire.rendered_hash(card.id, wire_key, prim.value, sdk),
            q.role,
            dict(fmt),
        )

    def render(self, card: RoleCard, pack: dict[str, Any], keys: list[str] | None = None, leaf_space: dict[str, Any] | None = None) -> list[RenderedQuestion]:
        out: list[RenderedQuestion] = []
        listing = self._list_field(card, pack)
        base_fmt = _SafeDict({k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in pack.items()})
        for key, q in card.questions.items():
            if keys is not None and key not in keys:
                continue
            template_hash = require_sealed(card, key)
            if q.role == "item":
                if listing is None:
                    raise JevClientError(f"{card.id}.{key} needs a list field in the pack")
                for n, _ in enumerate(listing[1], start=1):
                    fmt = _SafeDict({**base_fmt, "n": n})
                    out.append(self._rendered(card, q, f"{key}[{n}]", key, q.text.format_map(fmt), fmt, template_hash))
            elif q.role == "pairwise":
                if listing is None:
                    raise JevClientError(f"{card.id}.{key} needs a list field in the pack")
                n_items = len(listing[1])
                pairs = list(itertools.combinations(range(1, n_items + 1), 2))
                if n_items > MAX_PAIRWISE_ITEMS:
                    pairs = [(i, j) for i, j in pairs if j - i <= 3]
                for i, j in pairs:
                    fmt = _SafeDict({**base_fmt, "i": i, "j": j})
                    out.append(self._rendered(card, q, f"{key}[{i},{j}]", key, q.text.format_map(fmt), fmt, template_hash))
            else:
                text = q.text.format_map(base_fmt)
                out.append(self._rendered(card, q, key, key, text, base_fmt, template_hash, leaf_space))
        return out

    # ------------------------------------------------------------ calling

    def _throttle(self) -> None:
        min_interval = 60.0 / max(1, self.rpm)
        wait = self._last_call + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call(
        self,
        card: RoleCard,
        pack: dict[str, Any],
        rqs: list[RenderedQuestion],
        *,
        variant: str,
        role: wire.CallRole,
        run_id: str,
        input_language: str | None,
    ) -> tuple[JevResult, dict[str, Any]]:
        questions = {r.key: r.sdk for r in rqs}
        n_tokens = wire.request_tokens(pack, questions)
        if n_tokens > self.max_request_tokens:
            raise JevClientError(
                f"request for {card.id} is about {n_tokens} tokens over {len(rqs)} questions, "
                f"past the request budget {self.max_request_tokens}"
            )
        self._throttle()
        result = self.caller(pack, questions, self.model_id)
        if result.model_id != self.model_id:
            raise JevClientError(f"transport answered with model {result.model_id}, pinned {self.model_id}")
        bindings = {
            r.key: {"template_hash": r.template_hash, "rendered_hash": r.rendered_hash} for r in rqs
        }
        digest = wire.request_digest(
            model_id=self.model_id, transport=self.transport, state=pack, questions=questions,
            input_language=input_language,
        )
        fields: dict[str, Any] = {
            "card_id": card.id,
            "protocol_version": wire.WIRE_PROTOCOL_VERSION,
            "run_id": run_id,
            "call_role": role,
            "run_protocol": wire.run_protocol(card),
            "question_keys": [r.key for r in rqs],
            "question_bindings": bindings,
            "questions_digest": wire.questions_digest(bindings),
            "model_id": result.model_id,
            "transport": self.transport,
            "state_sha": sha_of(pack),
            "input_language": input_language,
            "variant": variant,
            "request": {"state": pack, "questions": questions, "request_digest": digest},
            "response": {"answers": result.answers, "usage": result.usage, "request_id": result.request_id},
        }
        if self.ledger is not None:
            row = self.ledger.record_jev_call(
                card_id=fields["card_id"],
                protocol_version=fields["protocol_version"],
                question_keys=fields["question_keys"],
                question_bindings=fields["question_bindings"],
                questions_digest=fields["questions_digest"],
                model_id=fields["model_id"],
                transport=fields["transport"],
                state_sha=fields["state_sha"],
                request=fields["request"],
                response=fields["response"],
                latency_ms=result.latency_ms,
                usage=result.usage,
                run_id=run_id,
                call_role=role,
                run_protocol=fields["run_protocol"],
                variant=variant,
                input_language=input_language,
            )
        else:
            row = {"id": f"jc_unrecorded_{uuid.uuid4().hex[:12]}", **fields}
        problems = [
            f"{r.key}: {p}"
            for r in rqs
            for p in wire.validate_answer(r.sdk, result.answers.get(r.key))
        ]
        if problems:
            raise JevAnswerError(
                f"call {row['id']} ({variant}) returned answers outside their declared space: "
                + "; ".join(problems[:5]),
                run_id=run_id,
                call_ids=[str(row["id"])],
            )
        return result, row

    def _calibration(
        self, card: RoleCard, rq: RenderedQuestion, asked: list[str], input_language: str | None
    ) -> tuple[CalibrationStatus, ThresholdRef | None]:
        """`input_language` is the effective one: the declaration, if the state is written in its script."""
        kind: Literal["agreement", "outcome"] = "outcome" if card.id in OUTCOME_CARDS and rq.role == "main" else "agreement"
        verdict = certificate_state(
            card, rq.base_key, model_id=self.model_id, transport=self.transport,
            language=self.language, root=self.calibration_root, asked_keys=asked,
            input_language=input_language,
        )
        cert = verdict.certificate
        ref = None
        if verdict.state == "calibrated" and cert is not None and cert.value is not None:
            ref = ThresholdRef(
                card_id=card.id, key=rq.base_key, value=cert.value, fitted_on=cert.at,
                n=int(cert.counts.get("eligible", 0)), status="validated",
                experiment_id=cert.experiment_id, dataset_version=cert.dataset_version,
                measurement_conditions=cert.measurement_conditions,
            )
        n = int(cert.counts.get("eligible", 0)) if cert is not None else 0
        return CalibrationStatus(kind=kind, state=verdict.state, n=n, reason=verdict.reason), ref

    def ask(
        self,
        card_id: str,
        pack: dict[str, Any],
        *,
        keys: list[str] | None = None,
        variant: str = "base",
        leaf_space: dict[str, Any] | None = None,
        paraphrases: dict[str, list[str]] | None = None,
        allow_degraded: bool = False,
        input_language: str | None = None,
    ) -> CardResult:
        """Ask one card about one state pack; every call of the run is recorded.

        `input_language` declares the language the pack is written in. It is recorded on every
        call. A certificate covers only inputs whose declared language it was confirmed on and
        whose text is in that language's script; an undeclared input is never calibrated.
        """
        card = self.cards.get(card_id)
        if card is None:
            raise JevClientError(f"unknown card {card_id}")
        if card.rung != Rung.jev and not allow_degraded:
            raise RungError(f"card {card.id} runs at rung {card.rung.value} (degraded on {card.degraded_on}: {card.degrade_reason}); route to that rung")
        missing = [f for f in card.state_pack if f not in pack]
        if missing:
            raise JevClientError(f"pack for {card.id} is missing fields: {', '.join(missing)}")
        extra = [f for f in pack if f not in card.state_pack]
        if extra:
            raise JevClientError(f"pack for {card.id} carries fields outside its state pack: {', '.join(extra)}")
        n_tokens = wire.estimate_tokens(pack)
        if n_tokens > self.max_state_tokens:
            raise JevClientError(f"pack is about {n_tokens} tokens, over the state budget {self.max_state_tokens}")

        rqs = self.render(card, pack, keys=keys, leaf_space=leaf_space)
        if not rqs:
            raise JevClientError("no questions to ask")
        run_id = new_id("run")
        result = CardResult(card_id=card.id, pack_sha=sha_of(pack), model_id=self.model_id, transport=self.transport, run_id=run_id)
        rows: list[dict[str, Any]] = []

        def call(batch: list[RenderedQuestion], variant_label: str, role: wire.CallRole) -> None:
            try:
                res, row = self._call(card, pack, batch, variant=variant_label, role=role, run_id=run_id,
                                      input_language=input_language)
            except JevAnswerError as exc:
                raise JevAnswerError(str(exc), run_id=run_id, call_ids=[*result.call_ids, *exc.call_ids]) from exc
            rows.append(row)
            result.call_ids.append(str(row["id"]))
            result.requests += 1
            result.usage_input_tokens += int(res.usage.get("input_tokens", 0))

        call(rqs, variant, "primary")

        # option-order averaging: every choice question over `permutations` orders
        choice_rqs = [r for r in rqs if r.primitive == Primitive.choice]
        if choice_rqs:
            for k in range(max(0, card.permutations - 1)):
                shuffled: list[RenderedQuestion] = []
                for r in choice_rqs:
                    r2 = copy.deepcopy(r)
                    items = list(r2.sdk["criteria"].items())
                    self.rng.shuffle(items)
                    r2.sdk["criteria"] = dict(items)
                    r2.sdk["criteria_order"] = [name for name, _ in items]
                    # a re-ordered option list is a different payload, so it hashes differently
                    r2.rendered_hash = wire.rendered_hash(card.id, r2.key, r2.primitive.value, r2.sdk)
                    shuffled.append(r2)
                call(shuffled, f"perm{k + 1}", "permutation")

        # re-runs: every yes/no question a fixed number of times, whatever it answered, plus
        # stability re-runs of every choice the option orders left contested
        noul_rqs = [r for r in rqs if r.primitive == Primitive.noul]
        contested_choices = [r for r in choice_rqs if wire.derive_question(rows, r.key).contested]
        rerun_rqs = noul_rqs + contested_choices
        if rerun_rqs:
            for k in range(max(0, card.reruns)):
                call(rerun_rqs, f"rerun{k + 1}", "rerun")

        # paraphrase runs (texts supplied by the paraphraser role), filled like the question
        if paraphrases:
            for r in noul_rqs:
                for i, ptxt in enumerate(paraphrases.get(r.base_key) or []):
                    r2 = copy.deepcopy(r)
                    text = ptxt.format_map(_SafeDict(r.fmt))
                    r2.sdk["instructions"] = wire.compose_instructions(card, r.key, text, language=self.language)
                    r2.text = text
                    r2.rendered_hash = wire.rendered_hash(card.id, r2.key, r2.primitive.value, r2.sdk)
                    call([r2], f"paraphrase{i + 1}", "paraphrase")

        at = now_iso()
        asked = wire.asked_base_keys(rows)
        effective_language = wire.input_language_of(pack, input_language)
        derived = {r.key: wire.derive_question(rows, r.key) for r in rqs}
        for r in rqs:
            d = derived[r.key]
            if d.problems:
                raise JevAnswerError(
                    f"{card.id}.{r.key} cannot be derived from its run: " + "; ".join(d.problems[:3]),
                    run_id=run_id, call_ids=list(result.call_ids),
                )
            status, ref = self._calibration(card, r, asked, effective_language)
            stability = Stability(
                rerun_sd=d.rerun_sd,
                paraphrase_sd=d.paraphrase_sd,
                permutation_sd=d.permutation_sd,
                negation_gap=wire.negation_gap(rows, r.key) if r.primitive == Primitive.noul else None,
                n_reruns=d.n_reruns,
                n_paraphrases=d.n_paraphrases,
                n_permutations=d.n_permutations,
            )
            primary_answer = dict(rows[0]["response"]["answers"][r.key])
            est = Estimate(
                primitive=r.primitive,
                p=d.p,
                probabilities=d.probabilities if r.primitive != Primitive.noul else None,
                choice=d.choice,
                score=d.score,
                card_id=card.id,
                question_key=r.key,
                question_hash=r.template_hash,
                rendered_hash=r.rendered_hash,
                state_sha=result.pack_sha,
                protocol_version=wire.WIRE_PROTOCOL_VERSION,
                run_id=run_id,
                language=self.language,
                input_language=input_language,
                aggregation=d.aggregation,
                contested=d.contested,
                contributions=[ContributingCall(**vars(c)) for c in d.contributions],
                model_id=self.model_id,
                transport=self.transport,
                source_tier=SourceTier.jev_judged,
                stability=stability,
                sufficiency_p=None,
                jev_confidence=float(primary_answer["confidence"]) if primary_answer.get("confidence") is not None else None,
                threshold_ref=ref,
                calibration_status=status,
                at=at,
            )
            result.estimates[r.key] = QuestionEstimate(
                key=r.key, base_key=r.base_key, estimate=est, raw=primary_answer,
                contested=d.contested,
                runs=sum(1 for c in d.contributions if c.contributes_to_value),
            )
        # The sufficiency twin rides along: its own derived value is copied onto the main
        # estimates, named by key, so a derived number is never mistaken for this one's own.
        suff_keys = [r.key for r in rqs if r.role == "sufficiency"]
        if suff_keys:
            sp = derived[suff_keys[0]].p
            for r in rqs:
                if r.role == "main" and sp is not None:
                    result.estimates[r.key].estimate = result.estimates[r.key].estimate.model_copy(
                        update={"sufficiency_p": sp, "sufficiency_question_key": suff_keys[0]}
                    )
        return result


def card_hashes(cards: dict[str, RoleCard]) -> dict[str, dict[str, str]]:
    return {cid: {k: compute_hash(c, k) for k in c.questions} for cid, c in cards.items()}
