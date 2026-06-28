---
name: checkpointing-persistence
description: 'Use when: designing, implementing, or validating network checkpoint save/resume.'
argument-hint: 'Describe the checkpoint mode, current step in Population_Save_Resume_and_Checkpointing.md (including Step 0 kickoff when relevant), target orchestration surface, determinism requirement, and whether the pass is design, implementation, migration, or validation.'
user-invocable: true
disable-model-invocation: false
skills:
  - reproducibility-contracts
  - hybrid-training-interop
  - multithread-evaluation
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Checkpointing And Persistence Playbook

Use this skill when NeatapticTS work touches the Phase 4 save, load, or resume
surface described in `plans/Population_Save_Resume_and_Checkpointing.md`.

This skill owns the durable workflow for checkpoint format design, full versus
light checkpoint semantics, strict restore validation, schema versioning,
counter restoration, and deterministic replay guarantees.

When tracker files need updating, `tracker-handoff` owns plan and log shape.
When sequencing or dependency tension is unclear, use `plan-alignment`.

See [checkpoint sources](./references/checkpoint-sources.md) for reference notes
on PRNG state, serialization compatibility, and restore invariants.

## Scope Boundary

- In scope: versioned checkpoint schemas, full and light checkpoint modes,
  orchestration-level save or load APIs, RNG state capture, innovation and ID
  counters, adaptive mutation state, species membership state, migration guards,
  exact versus best-effort restore behavior, checkpoint metadata and examples.
- Out of scope: low-level network JSON tuple evolution on its own, worker pool
  scheduling (owned by `multithread-evaluation`), payload transport (owned by
  `worker-inference-transport`), optimizer-vector contracts (owned by
  `hybrid-training-interop`), and browser bundle distribution.

## When to Use

- The task is to start or expand the checkpoint plan before code exists.
- A checkpoint schema or restore API is being introduced or revised.
- The repo needs exact resume for long-running deterministic evolution.
- A saved run must move across machines or sessions safely.
- Full versus light checkpoint behavior is unclear.
- ID counters, innovation counters, or RNG state restoration is the active bug.
- NEATchat needs durable identity or memory metadata on top of a shared
  checkpoint seam.

## When NOT to use

Do NOT use for determinism contracts - use `reproducibility-contracts` instead. Do NOT use for parameter-vector training bridges - use `hybrid-training-interop` instead.

## Step 0 kickoff

If the active task is planning or Step 0/Step 1 reconnaissance rather than code
changes:

1. Read [the Step 0 kickoff guide](references/checkpoint-step-0.md).
2. Produce or update the state-owner matrix, exactness categories, strict
   restore matrix, and red-phase test shortlist before proposing code work.
3. Update `plans/Population_Save_Resume_and_Checkpointing.md`,
   `plans/Roadmap.md`, and `plans/README.md` together if checkpointing becomes
   the active lane.
4. Stop and narrow the boundary again if the orchestration save/load entry
   surface is still ambiguous after the first mapping pass.

## Resume Contracts

### Full checkpoint

Full checkpoint is the exact-resume contract. It should capture every stateful
input that can change the future trajectory of the run.

Minimum set:

- generation index,
- evolution config relevant to behavior,
- population genomes and scores,
- species state when speciation is enabled,
- RNG state,
- innovation or gene counters,
- adaptive mutation or controller state,
- any replay-relevant metadata exposed by the orchestration layer.

### Light checkpoint

Light checkpoint is the best-effort continuation contract. It is allowed to keep
useful solution state without claiming exact replay.

Typical contents:

- best genomes or networks,
- enough config to restart evolution sensibly,
- seed or seed description,
- high-level metadata.

Do not promise exact future trajectory from light mode.

### Strict restore rule

If exact resume requires fields that are missing, the API must either:

- throw in strict mode, or
- warn and downgrade to best-effort behavior in non-strict mode.

Never silently pretend the resume is exact.

## Exactness Heuristic

Treat exact replay as this conjunction:

$$
R_{exact} = population \land species \land rng \land counters \land adaptive\ state \land deterministic\ evaluation
$$

If any term is absent, the checkpoint is not exact-resume capable.

## Workflow Diagram

```text
Flowchart summary: "Training iteration" → "Save checkpoint"; "Save checkpoint" → "Checkpoint type?"; "Checkpoint type?" → "Save all state + RNG" (Full), "Save counters only" (Light); "Save all state + RNG" → "Write to disk"; "Save counters only" → "Write to disk"; "Write to disk" → "Continue training"; "Continue training" → "Interrupted?"; "Interrupted?" → "Resume from checkpoint" (Yes), "Complete" (No); "Resume from checkpoint" → "Training iteration"; "Complete".
```

## Task Packet

Pass a compact packet that includes:

- active plan step,
- checkpoint mode in scope,
- orchestration entry point being changed,
- strict or non-strict restore requirement,
- determinism requirement,
- migration or backward-compatibility concern,
- validation target: roundtrip test, replay test, example docs, or all three.

Compact example:

```text
Use checkpointing-persistence for full checkpoint Step 2.
Plan: plans/Population_Save_Resume_and_Checkpointing.md.
Mode: full + strict restore.
Invariant: resumed deterministic run must match the original trajectory once the checkpoint is restored.
Migration concern: network serialization already preserves innovation and gene identity, but orchestration counters and RNG state are still missing.
Validate with: save/load roundtrip tests and deterministic replay test. Only run `npm run test:silent` if the active step packet or user explicitly requires repo-wide confirmation.
```

## Required Workflow

1. Read `plans/README.md`, then
   `plans/Roadmap.md`, then
   `plans/Population_Save_Resume_and_Checkpointing.md`.
2. Read the nearest relevant README and restore boundaries before deep source
   edits.
3. Inventory every stateful input that affects future behavior.
4. Classify each item as one of:
   - required for exact replay,
   - optional but useful metadata,
   - unsafe or explicitly out of scope.
5. Add or update the smallest red-phase test for the active restore contract.
6. Implement the smallest boundary-local save or load change.
7. Immediately rerun focused validation after the first substantive edit.
8. Run `coverage-guard` on every touched `src/` file.
9. Document the checkpoint schema, restore caveats, and mode differences.
10. Update the plan only after the code and validation are green.

## Schema Rules

- Every checkpoint format must be explicitly versioned.
- Public checkpoint fields must be self-describing enough to support migration.
- Separate metadata that affects behavior from metadata that only aids
  inspection.
- Keep v1 JSON-serializable unless there is a proven need for a more complex
  format.

## Restore Rules

### Counters

- Restore all counters that affect future IDs or innovation numbering.
- Do not assume network-level deserialization alone advances the orchestration
  counters far enough.

### RNG

- Persist the actual RNG state, not only the seed, whenever exact replay is the
  claim.
- A seed without current generator state is usually a restart hint, not an exact
  resume guarantee.

### Metadata extension

- Allow downstream metadata in a documented extension surface.
- Keep downstream metadata from redefining the core checkpoint contract.
- NEATchat-specific metadata should fit into the checkpoint surface without
  forking the base schema.

## Validation Cadence

- Save and load roundtrip tests for the touched orchestration surface.
- Deterministic replay tests when exact resume is part of the claim.
- Negative tests for strict restore failures.
- Focused serialization tests when the change interacts with network identity.
- `npm run test:silent` only when the active step packet or user explicitly requires repo-wide confirmation; otherwise, report the focused slice result as the gate evidence.

## Decision Tree

```text
Flowchart summary: "Checkpoint needed" → "Exact resume required?"; "Exact resume required?" → "Full checkpoint + strict restore" (Yes), "Light checkpoint" (No, best-effort), "Strict mode?" (Missing fields on load); "Full checkpoint + strict restore" → "Capture full state tuple"; "Light checkpoint" → "Capture best genomes + seed + config"; "Strict mode?" → "Throw on missing required fields" (Yes), "Warn + downgrade to best-effort" (No); "Capture full state tuple"; "Capture best genomes + seed + config"; "Throw on missing required fields"; "Warn + downgrade to best-effort".
```

## Before / After Examples

**Before:**

```json
{
  "version": 1,
  "generation": 50,
  "population": [...],
  "seed": 42
}
// incomplete: missing RNG state, counters, species state → cannot exact-resume
```

**After:**

```json
{
  "version": 1,
  "generation": 50,
  "population": [...],
  "species": [...],
  "rngState": { ... },
  "innovationCounter": 1024,
  "geneCounter": 2048,
  "adaptiveState": { ... },
  "seed": 42
}
// complete: all stateful inputs captured → strict exact-resume capable
```

## Guardrails

- Do not store callbacks or other non-portable runtime objects in v1 unless the
  public contract explicitly owns them.
- Do not treat `node:v8` serialization as a stable cross-runtime wire format for
  public checkpoints by default.
- Do not collapse full and light checkpoint semantics into one ambiguous mode.
- Do not claim exact replay if evaluation order or RNG state is not restored.
- Do not silently migrate incompatible checkpoint versions.
- Do not bypass schema versioning just because internal code currently lines up.
- Do not reopen lower-level network serialization design during Step 0 unless a
  proven checkpoint gap crosses that boundary.

## Expected Final Output

A strong checkpointing pass should report:

- the checkpoint mode and orchestration surface targeted,
- the state inventory added or restored,
- strict versus best-effort behavior,
- schema versioning or migration impact,
- focused validation results,
- coverage result for touched `src/` files,
- the updated plan step.
