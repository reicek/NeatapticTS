---
name: checkpointing-persistence
description: 'Use when: checkpoint save, resume, restore, and persistence boundaries for NeatapticTS evolution or network state.'
argument-hint: 'Describe the save/resume boundary, active plan step, strict versus best-effort restore behavior, and required validation.'
user-invocable: false
disable-model-invocation: false
skills:
  - tracker-handoff
  - reproducibility-contracts
  - research-methodology
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Checkpointing and Persistence Playbook

Use this skill when NeatapticTS work needs to save, resume, restore, or
migrate evolution or network state across sessions, workers, or environments.

This skill owns the durable workflow for checkpoint schema design, strict
versus best-effort restore behavior, RNG/counter persistence, full versus light
checkpoints, and versioned public checkpoint formats. It coordinates closely
with `reproducibility-contracts` for determinism claims and with
`tracker-handoff` for plan/log continuity.

## When to Use

- Starting or scoping save and resume work for evolution state.
- Expanding `Population_Save_Resume_and_Checkpointing.md` Step 0 or Step 1.
- Mapping strict versus best-effort restore behavior.
- Deciding whether persistence belongs to checkpointing, replay, transport,
  worker-pool, or optimizer-vector concerns.
- Designing full versus light checkpoint modes.
- Adding schema versioning or migration logic to checkpoint files.

## When NOT to use

Do NOT use for replay semantics alone — use `reproducibility-contracts` instead.
Do NOT use for worker payload transport — use `worker-inference-transport` instead.
Do NOT use for parameter-vector optimizer handoff — use `hybrid-training-interop` instead.

## Workflow Diagram

```text
Flowchart summary: "Need persistence" → "Full or light checkpoint?"; "Full or light checkpoint?" → "Capture full replay tuple" (Full), "Capture restart state" (Light); "Capture full replay tuple" → "Version schema"; "Capture restart state" → "Version schema"; "Version schema" → "Validate restore"; "Validate restore" → "Done".
```

## Task Packet

Pass a compact packet that includes:

- the checkpoint boundary (network, population, evolution run),
- the active plan step,
- full versus light mode,
- strict versus best-effort restore target,
- which tuple components must be captured (seed, RNG state, counters, adaptive state),
- the validation target.

## Required Workflow

1. Read `plans/completed/Population_Save_Resume_and_Checkpointing.md` and any
   active follow-up amendment in `plans/`.
2. Identify whether the work is strict restore (full replay tuple) or best-effort
   restart (light state).
3. Reuse network serialization for graph identity; own orchestration-level state
   separately.
4. Version every public checkpoint shape explicitly.
5. Add the smallest focused validation that can falsify the restore claim.
6. Document fallback behavior and schema migration rules.

## Guardrails

- Do not treat seed capture alone as proof of exact resume.
- Do not conflate full replay fidelity with light restart state.
- Do not let checkpoint schema drift across versions without a migration rule.
- Do not hide restore failures behind silent best-effort guesses in strict mode.
- Do not duplicate graph serialization that network serialization already owns.

## References

- [Checkpoint reference notes](./references/checkpoint-sources.md)
