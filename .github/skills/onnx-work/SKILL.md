---
name: onnx-work
description: 'Extend, harden, or validate ONNX export and import in NeatapticTS. Use when adding new operator support, hardening export determinism, adding roundtrip tests, or updating the supported-subset documentation.'
argument-hint: 'Describe the ONNX target (layer type, operator, or hardening goal), the current plan phase, and whether this is reconnaissance, implementation, or roundtrip validation.'
user-invocable: true
disable-model-invocation: false
---

# ONNX Work Playbook

Use this skill when ONNX export or import needs to be extended, hardened, or
validated in NeatapticTS.

ONNX is Phase 6 in the roadmap and runs as a **parallel lane** (Parallel
Lane B) after Phase 2 provides stable preconfigured architectures to test
against. Clean, stable network shapes from the architecture builders are the
best test input for ONNX export correctness.

This skill owns the durable workflow for ONNX operator extension, roundtrip
validation, supported-subset documentation, and export determinism hardening.
When tracker files need updating, `tracker-handoff` owns the plan/log shape.
When roadmap alignment is needed, use `plan-alignment`.

## Scope Boundary

- **In scope:** op-to-ONNX mapping, roundtrip export/import fidelity, output
  determinism, layer-analysis utilities, supported-subset documentation,
  ONNX test coverage expansion.
- **Out of scope:** general Network topology refactors (owned by Phase 2/3
  plans), performance optimization (owned by `performance-optimization`),
  browser bundling (owned by `browser-build`), algorithm correctness fixes
  (owned by NEAT plans).

## When to Use

- A new activation function, layer type, or connection pattern needs an ONNX
  operator mapping.
- The ONNX export roundtrip does not preserve a network's inference behavior.
- An existing operator mapping produces incorrect output shapes or constant
  values.
- The supported-subset documentation needs updating after operator additions.
- ONNX import needs to reconstruct a `Network` from a `.onnx` file correctly.
- A new architecture builder (from `architecture-builder` skill) needs ONNX
  export coverage added.

## Task Packet

Pass a compact packet that includes:

- target (layer type / operator / hardening goal),
- current plan phase from `plans/ONNX_EXPORT_PLAN.md`,
- whether this is reconnaissance, implementation, or roundtrip validation,
- required validation (roundtrip test, focused Jest slice, or full suite).

Compact example:

```text
Use onnx-work for Conv2D operator in src/architecture/network/onnx/.
Plan: plans/ONNX_EXPORT_PLAN.md (Phase 3 — convolutional groundwork).
Mode: implementation + roundtrip test.
Validate with: npx jest --testPathPattern=onnx, then npm run test:silent.
```

## Required Workflow

1. Read `plans/ONNX_EXPORT_PLAN.md` before editing.
2. Read `src/architecture/network/onnx/README.md` and the nearest parent README.
3. Identify the current plan phase and the specific operator or goal in scope.
4. For a new operator mapping:
   - Consult the ONNX operator spec at `https://onnx.ai/onnx/operators/` for
     the canonical attribute and type constraints.
   - Identify the nearest existing mapping as a reference pattern.
5. Implement the mapping in the correct sub-boundary following the repo naming
   pattern (`network.onnx.export.<target>.ts`, etc.).
6. Add a roundtrip test:
   - Build a known network → export to ONNX → import back → compare activation
     output for the same input vector within float32 tolerance.
7. Validate with a focused Jest slice for the ONNX boundary.
8. Update the supported-subset operator table in the nearest JSDoc or README
   surface to reflect the new operator.
9. Run `npm run docs` to verify generated output.
10. Update `plans/ONNX_EXPORT_PLAN.md` with the completed step.
11. Run `npm run test:silent` to confirm repo-wide green.

## Supported Subset Rule

ONNX work must maintain honest "supported subset" documentation:

- After every operator addition, update the operator table in the nearest
  relevant JSDoc or README surface.
- Never claim support for an operator whose roundtrip test does not exist and
  pass.
- When a network graph feature cannot be mapped to ONNX (dynamic recurrent
  connections, custom operators), document the limitation explicitly rather
  than silently omitting it.
- The supported-subset table is the contract with downstream users — it must
  be accurate, not aspirational.

## Roundtrip Determinism Contract

ONNX export must satisfy:

- For the same network, same inputs, and same (fixed) graph topology, the
  exported ONNX model's inference output must match `Network.activate()` within
  float32 tolerance.
- Recurrent networks must maintain correct state carry-over semantics across
  exported steps.
- Do not claim ONNX output is deterministic unless a test verifies it under a
  fixed graph and fixed inputs.

## Guardrails

- Do not merge multiple operator targets into one pass unless they share an
  identical mapping scaffold.
- Do not treat an operator as supported until the roundtrip test exists and
  passes.
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

- the operator or layer type targeted,
- the mapping implementation summary,
- the roundtrip test result (pass/fail + tolerance used),
- the updated supported-subset documentation state,
- the plan step updated,
- repo-wide suite result.
