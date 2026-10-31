# Sextant architecture

Sextant is built around one idea: a recommendation is a commitment that reality can later contradict, so the system has to remember exactly what it committed to and change when the evidence changes. Everything below follows from that. This document explains the main parts, why they are separated the way they are, and what is still missing.

## The task is the unit of work

A task starts as a `Mission` (`src/sextant/domain.py`): a goal, where the goal came from, hard constraints, a preference order, the material in scope, and the conditions for stopping or waiting. The goal and the preferences always come from the person running the task. A model can suggest a new goal, but the suggestion stays a suggestion until a person authorizes it.

Facts enter as `Observation` records. Each one keeps its source, the part of the source that was read, when it happened, how it was measured, and whether it is complete. An observation that says something is not the same as that thing being true, so the record keeps the claim and its provenance apart. A correction or a withdrawal is a new event that points at the old one. Nothing is overwritten.

From those inputs the engine produces a `Proposal` with competing candidates, a `Policy` (actions that depend on conditions, with reversal triggers and open questions), `Prediction` records frozen before an experiment runs, and a `Receipt` for every external call. Each object gets a new `Revision` when it changes.

## Models propose and the host decides

The split of authority is the core of the design.

- The application service (`application.py`) receives the host's authority as a constructor argument. No request, observation, import, or model output can create authority.
- The task compiler (`compiler.py`) has no filesystem, network, or tool access of its own. Every model call goes through the operation's dispatcher, and its output stays a proposal.
- The runtime (`runtime.py`) is the single place where effects are sent. It checks the authorization for each send, writes a receipt, and fences operations so that two writers cannot run the same step.

When an external call fails after it was sent, the runtime cannot know whether the effect happened. It records the outcome as `unknown` and requires reconciliation before anything is retried. An effect that was refused before sending is recorded as not sent, with its cause.

This costs some ceremony on every call. The alternative, a single long-running agent with memory and broad permissions, is more flexible, but it mixes facts, judgments, and permissions in one place, and then it is hard to tell a real improvement from a leaked hint or a lucky prompt. Sextant keeps a small kernel that owns evidence and effects, and keeps the models replaceable.

## Competing explanations and a blind challenger

`sextant run` asks a generator model for several candidate mechanisms that could explain the situation. A separate challenger sees the goal and the evidence but not the generator's design, and tries to refute the candidates. The compiler joins what survives into a conditional policy and asks for experiments that would tell the candidates apart. If none is proposed, the policy stays provisional.

Deterministic checks can issue narrow certificates. Finite arithmetic is one example: `calculate 7 * (3 + 2)` is solved exactly without a model, and the result carries a certificate that covers only that computation. Everything a model says stays a proposal, even when it is labeled deterministic.

## Numbers keep their meaning

The decision code (`decision.py`) applies hard constraints first, so an option that violates one is infeasible no matter how well it scores elsewhere. Where only an ordering is known, the engine keeps a partial order instead of inventing weights. An infeasible option gets no score.

Judgments from TypeSafe's Jev model are recorded as advisory diagnostics. The spread of repeated answers to the same prompt describes how stable the prompt is, not how likely an event is, so the engine does not report it as a confidence. Agreement between several checks is not multiplied into a reliability figure.

## Context for a judgment is scoped and states its losses

A long task accumulates more material than a single judgment should see. `jev_context.py` builds a deterministic, question-scoped view: the current goal and constraints, the object being judged, the inputs it depends on, supporting and contrary evidence, and observations that share a source. Material linked as jointly relevant cannot be dropped from a selected part. Each view records what it left out, and a judgment made from a partial view cannot be used to claim that nothing is missing. The raw mission evidence is never pruned.

## Revision follows the evidence

Tasks live in a SQLite store (`storage.py`) with immutable, hash-verified artifacts written before each transaction. The store supports export, import, replay, backup, and invalidation. When an observation is withdrawn, the store invalidates the objects that cited it, and only those. A branch that cited nothing specific still depends on the whole context, so it is invalidated too. `sextant inspect` shows the saved receipts next to a fresh validity check, so a stale conclusion is visible instead of silently trusted.

Imported assertions never bring authority with them. By default an imported task arrives with its execution permissions revoked, and the person running Sextant grants them again.

## Execution happens behind a controlled interface

Experiments run through environment ports (`environments.py`): a file source limited to the paths the task may read, a Python sandbox, and a pipe that lets a host program broker providers and tools. Local execution runs under the macOS `sandbox-exec` profile built in `isolation.py`. If that boundary is not available, the run stops with a blocker instead of running unprotected.

The store is a cooperative local runtime. It detects damage and inconsistency, but it does not protect against a host that can write to the database directly.

## What is not built yet

The engine, the store, the command line, and the offline examples work. The live routes, which send proposals to a local model client and judgments to TypeSafe, have not been qualified, and there is no completed comparison against baseline approaches. The older decision-graph tools still rely on a single-writer lock instead of crash-safe multi-file transactions. Learning across tasks, where a method that worked once becomes a reusable method, is a design goal that the current code does not implement.
