---
name: onnx-work
description: 'Use when: extending or validating ONNX export/import for networks.'
argument-hint: 'Describe the ONNX target (export operator / import subset / recurrent hardening / external seed path), the current plan phase, and whether this is implementation, import hardening, documentation, or roundtrip validation.'
user-invocable: true
disable-model-invocation: false
skills:
  - reproducibility-contracts
  - hybrid-training-interop
  - neatchat-systems
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# ONNX Work Playbook

Use this skill when ONNX-like export or import needs to be extended, hardened,
or validated in NeatapticTS.

ONNX is Phase 6 in the roadmap and runs as a **parallel lane** (Parallel
Lane B) after Phase 2 provides stable preconfigured architectures to test
against. Clean, stable network shapes from the architecture builders are the
best test input for ONNX export correctness.

This skill owns the durable workflow for operator extension, roundtrip
validation, constrained import hardening, honest supported-subset
documentation, and deterministic export or import claims. When tracker files
need updating, `tracker-handoff` owns the plan/log shape. When roadmap
alignment is needed, use `plan-alignment`.

## Local Reference Pack

Before fetching ONNX docs again, read `ONNX_1_22_0_REFERENCE.md` in this skill
folder.

Use that local pack for:

- ONNX 1.22.0 model and graph concepts,
- opset and domain rules,
- converter parity expectations,
- low-bit type facts (`float8`, `int4`, `float4`, `int2`),
- and the repo-local audit checklist for deciding whether a claim is
  JSON-first roundtrip, binary ONNX compatibility, or true runtime
  interoperability.

Go back to the web only when operator-specific schema details, checker behavior,
or newer-version deltas are needed.

## Scope Boundary

- **In scope:** op-to-ONNX mapping, JSON-first export/import fidelity, output
  determinism, recurrent import hardening for explicitly supported subsets,
  runtime-load utilities, layer-analysis utilities, honest external-seed import
  contracts, supported-subset documentation, ONNX test coverage expansion.
- **Out of scope:** general Network topology refactors (owned by Phase 2/3
  plans), performance optimization (owned by `performance-optimization`),
  browser bundling (owned by `browser-build`), algorithm correctness fixes
  (owned by NEAT plans), checkpoint semantics (owned by
  `checkpointing-persistence`), and parameter-vector training bridges (owned by
  `hybrid-training-interop`).

## When to Use

- A new activation function, layer type, or connection pattern needs an ONNX
  operator mapping.
- The ONNX export roundtrip does not preserve a network's inference behavior.
- An existing operator mapping produces incorrect output shapes or constant
  values.
- The import path needs to accept or reject a constrained recurrent subset
  honestly.
- The supported-subset documentation needs updating after operator additions.
- ONNX import needs to reconstruct a `Network` from a `.onnx` file correctly.
- A new architecture builder (from `architecture-builder` skill) needs ONNX
  export coverage added.
- A downstream consumer such as the planned NEATchat follow-up lane needs an
  honest recurrent seed-import boundary.

## When NOT to use

Do NOT use for internal serialization - use Network native methods instead. Do NOT use for general network construction - use `architecture-builder` instead.

## Workflow Diagram

```text
Flowchart summary: "Network" → "Export to ONNX-like JSON"; "Export to ONNX-like JSON" → "Import back"; "Import back" → "Reconstruct Network"; "Reconstruct Network" → "Compare activation output"; "Compare activation output" → "Within tolerance?"; "Within tolerance?" → "Roundtrip verified" (Yes), "Debug operator mapping" (No); "Roundtrip verified"; "Debug operator mapping" → "Export to ONNX-like JSON".
```

## Task Packet

Pass a compact packet that includes:

- target (layer type / operator / import subset / external seed path),
- current ONNX baseline from `plans/completed/ONNX_EXPORT_PLAN.md` and any narrower active follow-up amendment in `plans/` when one exists,
- whether this is implementation, import hardening, documentation, or roundtrip
  validation,
- required validation (roundtrip test, focused Jest slice, or full suite).

Compact example:

```text
Use onnx-work for recurrent import hardening.
Plan: plans/completed/ONNX_EXPORT_PLAN.md (archived baseline; create or follow a narrower active amendment before widening support).
Target: compact recurrent import subset for external seed compatibility.
Mode: import hardening + supported-subset documentation.
Validate with: focused ONNX Jest slice and import acceptance/rejection tests. Only run `npm run test:silent` if the active step packet or user explicitly requires repo-wide confirmation.
```

## Required Workflow

1. Read `plans/completed/ONNX_EXPORT_PLAN.md` before editing, plus any newer active ONNX follow-up amendment in `plans/` when one exists.
2. Read `src/architecture/network/onnx/README.md` and the nearest parent README.
3. Identify whether the active pass is:

- export operator mapping,
- import hardening,
- recurrent subset support,
- or an explicit non-ONNX bridge decision for a downstream consumer.

4. For a new operator mapping:
   - Consult the ONNX operator spec at `https://onnx.ai/onnx/operators/` for
     the canonical attribute and type constraints.
   - Identify the nearest existing mapping as a reference pattern.
5. For an import-hardening pass:

- Name the exact supported subset and the exact unsupported inputs that must
  reject cleanly.
- If the target is a downstream external seed path, confirm whether ONNX is
  the honest bridge or whether the plan should document a non-ONNX path
  instead of widening support claims.

6. Implement the change in the correct sub-boundary following the repo naming
   pattern.
7. Add focused validation:

- export or roundtrip: known network → export → import → compare activation
  output within float32 tolerance,
- import hardening: acceptance tests for the supported subset plus rejection
  tests for unsupported external graphs.

8. Validate with a focused Jest slice for the ONNX boundary.
9. Run `coverage-guard` on every `src/` file added or changed in this step.
   100% in all four categories (statements, branches, functions, lines) is
   required before proceeding.
10. Update the supported-subset operator table in the nearest JSDoc or README
    surface to reflect the new operator.
11. Run `npm run docs` to verify generated output.
12. Update the active ONNX follow-up amendment with the completed step, or create a narrower active amendment in `plans/` before coding if the work widens support beyond the archived ONNX baseline.
13. Run `npm run test:silent` only if the active step packet or user explicitly requires repo-wide confirmation; otherwise, report the focused slice result as the gate evidence.

## JSON-First Trust Boundary

- The current public boundary is intentionally JSON-first and ONNX-like; it is
  not yet a promise of universal protobuf or ONNX Runtime compatibility.
- Treat imported models as untrusted input.
- Do not widen compatibility claims beyond what the importer validates and what
  focused tests prove.

## Supported Subset Rule

ONNX work must maintain honest "supported subset" documentation:

- After every operator addition, update the operator table in the nearest
  relevant JSDoc or README surface.
- Never claim support for an operator whose roundtrip test does not exist and
  pass.
- When a network graph feature cannot be mapped to ONNX (dynamic recurrent
  connections, custom operators), document the limitation explicitly rather
  than silently omitting it.
- When a downstream consumer needs a stronger recurrent seed than the supported
  subset can host honestly, document the approved non-ONNX bridge instead of
  pretending the ONNX import surface is broader than it is.
- The supported-subset table is the contract with downstream users — it must
  be accurate, not aspirational.

## Roundtrip Determinism Contract

ONNX export must satisfy:

- For the same network, same inputs, and same (fixed) graph topology, the
  exported ONNX model's inference output must match `Network.activate()` within
  float32 tolerance.
- Recurrent networks must maintain correct state carry-over semantics across
  exported steps.
- Import hardening must make unsupported recurrent graphs fail clearly rather
  than reconstructing a misleading best-effort network silently.
- Do not claim ONNX output is deterministic unless a test verifies it under a
  fixed graph and fixed inputs.

## Recurrent Import Honesty Rule

- Treat recurrent import as a constrained subset, not as arbitrary ONNX chat
  model hosting.
- The first supported external-seed target should stay narrow and testable.
- If NEATchat or another downstream system needs a stronger seed path before the
  recurrent subset is honest, choose and document a non-ONNX bridge explicitly.
- Keep the downstream consumer honest about what it can actually host.

## Decision Tree

```text
Flowchart summary: "ONNX task" → "Which surface?"; "Which surface?" → "Export mapping" (Add operator mapping), "Import hardening" (Constrained recurrent import), "Roundtrip validation" (Export/import fidelity), "Supported-subset docs" (Operator table update); "Export mapping"; "Import hardening"; "Roundtrip validation"; "Supported-subset docs".
```

## Before / After Examples

**Before:**

```ts
const shape = [1, 4, 4]; // hardcoded for one network config
```

**After:**

```ts
const shape = deriveShapeFromGraph(graph); // derived from live graph topology
```

## Guardrails

- Do not merge multiple operator targets into one pass unless they share an
  identical mapping scaffold.
- Do not treat an operator as supported until the roundtrip test exists and
  passes.
- Do not claim arbitrary external ONNX import when only a constrained internal
  or same-version-family subset is proven.
- Do not bypass `npm run test:silent` after a focused tranche succeeds.
- Do not hand-edit generated README files.
- Do not add an operator mapping that hard-codes shape constants which will
  break for different network configs; always derive shapes from the graph.
- Follow `tracker-handoff` when updating plan/log files.
- Follow `educational-docs` for JSDoc and diagram quality in generated README
  chapters — the ONNX chapter should explain the operator taxonomy and export
  pipeline, not just list symbols.
- Do not prepend calendar dates to plan headings or session logs.

## Expected Final Output

A strong ONNX work pass should report:

- the operator, import subset, or external-seed bridge targeted,
- the mapping or hardening summary,
- the roundtrip or import-hardening test result,
- the updated supported-subset documentation state,
- whether a downstream consumer gate such as NEATchat moved forward,
- the plan step updated,
- repo-wide suite result.
