---
name: performance-optimization
description: 'Implement memory efficiency, slab optimization, typed-array usage, or benchmark-driven performance improvements in NeatapticTS. Use when executing a Track 1 phase of Memory_Optimization.md or addressing a profiled hotspot identified via trace-audit-reporting.'
argument-hint: 'Describe the optimization target (slab, typed array, activation path, worker payload), the evidence (trace report or benchmark), and the current plan phase.'
user-invocable: true
disable-model-invocation: false
---

# Performance Optimization Playbook

Use this skill when NeatapticTS source needs memory or runtime performance
improvements tied to `plans/Memory_Optimization.md` or a profiled hotspot.

For **identifying** hotspots, use `trace-audit-reporting` first, then use this
skill for the implementation pass once the evidence is clear. This skill owns
the implementation workflow; the trace skills own the diagnostic workflow.

Performance work is Phase 5 in the roadmap (Memory Optimization, Track 1) and
runs as a **parallel lane** after Phase 1 stabilizes. Track 2 (Hyper work)
remains gated behind Track 1 stability conditions defined in the plan.

When tracker files need updating, `tracker-handoff` owns the plan/log shape.
When roadmap alignment is needed, use `plan-alignment`.

## Scope Boundary

- **In scope:** typed-array pooling, slab allocator improvements, activation
  fast-path optimization, worker payload size reduction, memory reuse, allocator
  lifecycle cleanup, benchmark authoring.
- **Out of scope:** algorithm correctness (owned by NEAT plans), ONNX
  portability (owned by `onnx-work`), browser bundling (owned by
  `browser-build`), demo-level rendering performance (owned by
  `trace-audit-reporting` or `flappy-architecture-polish`).

## When to Use

- A Track 1 phase from `Memory_Optimization.md` is ready to implement.
- A profiled hotspot (from `trace-audit-reporting`) points to a library-level
  allocation or activation-path improvement.
- Activation throughput benchmarks show regression.
- A slab or typed-array change is needed before worker serialization work
  (Phase 4) can proceed efficiently.

## Task Packet

Pass a compact packet that includes:

- optimization target (what is being improved and why),
- evidence (trace report summary, benchmark delta, or plan phase reference),
- current Track 1 phase from `Memory_Optimization.md`,
- correctness invariant that must be preserved,
- validation expectations (benchmark target, focused test, regression suite).

Compact example:

```text
Use performance-optimization for slab allocator pool reuse.
Evidence: trace shows ~40% of activation time in Float64Array allocation.
Plan: Memory_Optimization.md Track 1 Phase 4.
Invariant: activation output must be numerically identical before and after.
Validate with: benchmarks/activation.bench.ts, then npm run test:silent.
```

## Required Workflow

1. Read `plans/Memory_Optimization.md` to confirm the current Track 1 phase and
   that Track 2 gates are not yet open.
2. Read the relevant trace report or benchmark result before editing.
3. Read the source boundary to be changed and its nearest test file.
4. Confirm the correctness invariant explicitly before touching any code.
5. Write a correctness regression test (or verify one exists) that will fail
   if activation output changes for the same inputs:
   - same network + same inputs + same seed → bitwise identical output before
     and after the change.
6. Implement the optimization in the smallest safe boundary.
7. Confirm the correctness invariant still holds with the focused test.
8. Run or author a benchmark to measure the improvement.
9. Run `npm run test:silent` to confirm no regressions.
10. Update `plans/Memory_Optimization.md` with the completed phase.

## Correctness Contract

Every performance change must satisfy:

- Activation output for the same network, same inputs, and same seed must be
  bitwise identical before and after the change.
- Any optimization that changes memory layout must preserve correct recurrent
  state semantics — prior activation values must not be clobbered between
  forward passes.
- Slab or pool changes must not introduce data leakage between independent
  networks in the same process.
- These invariants must be tested, not assumed.

## Track 2 Gate

Do not start Hyper (Track 2) work until the Track 1 stability conditions in
`Memory_Optimization.md` are fully satisfied. If those conditions appear met,
use `plan-alignment` to verify before proceeding.

## Guardrails

- Do not merge a performance change with a correctness change in the same step.
- Do not close a Track 1 phase until the correctness invariant test exists and
  passes.
- Do not ship a slab or pool change without a test that detects cross-network
  data leakage.
- Do not claim a performance improvement without measured benchmark evidence.
- Do not hand-edit generated README files.
- Follow `tracker-handoff` when updating plan/log files.
- Follow `educational-docs` for JSDoc on new helpers or changed public surfaces.
- Do not prepend calendar dates to plan headings or session logs.

## Expected Final Output

A strong performance optimization pass should report:

- the optimization target,
- the correctness invariant test result,
- before/after benchmark numbers (ops/second or ms/op),
- the Track 1 phase updated,
- repo-wide suite result.
