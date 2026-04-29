---
name: architecture-builder
description: 'Design, implement, test, and document preconfigured architecture builders (MLP, LSTM, GRU, NARX, sparse) in NeatapticTS. Use when adding a new builder entrypoint, extending an existing one, hardening builder validation, or documenting an architecture API.'
argument-hint: 'Name the architecture type (mlp, lstm, gru, narx, sparse), describe the current state of the builder, and state whether this is API design, implementation, testing, or documentation.'
user-invocable: true
disable-model-invocation: false
---

# Architecture Builder Playbook

Use this skill when a preconfigured architecture builder needs to be added,
extended, validated, or documented in NeatapticTS.

Preconfigured builders are the Phase 2 user-facing API for constructing
standard neural network architectures. They wrap `Network.construct()` with a
clean, typed, opinionated surface so a caller can produce a working MLP, LSTM,
GRU, or NARX network with sensible defaults and actionable diagnostics —
without having to assemble nodes and connections manually.

This skill owns the durable workflow for builder API design, implementation,
testing, and documentation. When tracker files need updating, `tracker-handoff`
owns the plan/log shape. When roadmap alignment is needed, consult
`plan-alignment` first.

When a builder step is complete, `educational-docs` is the mandatory follow-up
for the JSDoc and generated README surface — the same pattern as `solid-split`.

## Phase 2 Gate Conditions

All preconfigured builders must satisfy the Phase 2 gate before the lane closes:

- **Explicit I/O roles:** builders must produce networks with correct input
  and output node role assignments.
- **Deterministic construction:** same config + same seed → same network shape
  every time.
- **Actionable diagnostics:** validation failures must produce error messages
  that name the invalid field and suggest a correction.
- **Flat, typed API:** builder functions accept a typed config object with
  sensible defaults and return a `Network` or throw a descriptive error.
- **Demo integration gate:** the builder must work correctly in at least one
  verified usage context (Flappy Bird profile list, interactive example, or
  equivalent proof) before the architecture lane closes.

## When to Use

- A new architecture type (MLP, LSTM, GRU, NARX, random-sparse) needs a
  public builder entrypoint.
- An existing builder needs extended config options, validation messages, or
  determinism hardening.
- A builder is being added to the Flappy Bird profile list or an interactive
  example.
- Builder JSDoc needs improvement so the generated README chapter is
  educational.
- A builder's API contract needs test coverage for determinism, diagnostics,
  and roundtrip shape.

## Task Packet

Pass a compact packet that includes:

- architecture type,
- current state (`new` / `extending` / `validating` / `documenting`),
- relevant plan file (`plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`),
- whether browser demo integration is in scope for this pass,
- validation expectations.

Compact example:

```text
Use architecture-builder for the GRU builder in src/architecture/network/.
State: implementing builder + roundtrip test.
Plan: plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md.
Demo: not in scope this pass.
Validate with: npx jest --testPathPattern=gru, then npm run test:silent.
```

## Required Workflow

1. Read `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md` before
   editing.
2. Read the nearest relevant folder README (`src/architecture/network/README.md`
   or similar) and `src/architecture/README.md`.
3. Identify the current plan step and confirm it satisfies the Phase 2 gate
   conditions above.
4. Design the builder API contract before implementation:
   - typed config interface with full defaults,
   - builder function signature (e.g. `buildGRU(config?: Partial<GRUConfig>): Network`),
   - validation rules and expected error shapes,
   - determinism contract (same config + same seed → same output shape).
5. Implement the builder following the repo module naming pattern:
   ```
   network/builders/<arch>/network.builders.<arch>.ts       ← orchestration
   network/builders/<arch>/network.builders.<arch>.types.ts ← config types
   network/builders/<arch>/network.builders.<arch>.utils.ts ← helpers
   network/builders/<arch>/network.builders.<arch>.errors.ts← error classes
   ```
6. Write the minimum owner-local tests that verify:
   - correct network shape (node count, layer structure, connection count),
   - correct I/O role assignments,
   - determinism under a fixed seed (same output shape for same config),
   - validation rejects invalid configs with actionable error messages.
7. Validate with a focused Jest slice for the builder boundary.
8. Raise coverage on the new builder boundary toward 100% before closing the step.
9. Improve JSDoc on all public exported symbols so the generated README reads as
   an educational chapter — explain the architecture, its historical basis, and
   tradeoffs, not just the signature.
10. Run `npm run docs` to verify generated README output.
11. Update `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md` with the
    completed step.
12. Run `npm run test:silent` to confirm repo-wide green.
13. Invoke `educational-docs` on the changed boundary as the mandatory follow-up
    documentation pass.

## API Design Rules

- Builder functions must be named `build<Arch>(config?: Partial<ArchConfig>): Network`.
- Default configs must produce a minimal but runnable network on a zero-argument
  call.
- Error messages must name the invalid field and suggest a fix:
  `"gru.units must be ≥ 1, got 0"`.
- Recurrent builders (LSTM, GRU, NARX) must produce networks with the correct
  recurrent connection policy and explicit self-connection semantics.
- All builders must use `Network.construct()` internally — they must not bypass
  the Phase 2 construction pipeline.
- Each builder exports its config type from the same file so callers can type
  their own config objects.

## Educational Documentation Rules

Builder README chapters should teach the reader:

- what the architecture is (brief conceptual framing with Wikipedia or paper
  citation),
- why it was designed this way (design tradeoffs, when to use vs. alternatives),
- the config fields and their defaults (table is ideal),
- at least one short `ts` code example,
- at least one Mermaid diagram showing the network topology shape.

These rules apply to every architecture type: MLP, LSTM, GRU, NARX, sparse.

## Guardrails

- Do not merge multiple architecture types into one pass unless they share an
  identical builder scaffold and the user explicitly approves.
- Do not bypass the Phase 2 gate conditions to close a step faster.
- Do not hand-edit generated README files; improve JSDoc and run `npm run docs`.
- Do not add config options without defaults; every field must have a sensible
  default so zero-argument calls work.
- Do not describe a builder step as complete when the `educational-docs`
  follow-up is still pending.
- Follow `tracker-handoff` when updating plan/log files.
- Do not prepend calendar dates to plan headings or session logs.

## Expected Final Output

A strong architecture builder pass should report:

- the architecture type added or extended,
- the public builder API surface,
- test coverage for the builder boundary (per category),
- whether the demo integration gate was satisfied or explicitly deferred,
- JSDoc and generated README state,
- the updated plan state,
- whether `educational-docs` follow-up was completed or deferred.
