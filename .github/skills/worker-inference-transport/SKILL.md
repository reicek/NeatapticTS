---
name: worker-inference-transport
description: 'Design, implement, harden, or validate worker-friendly network serialization and inference transport in NeatapticTS. Use when working on inference IR extraction, portable or transferable payloads, MessageChannel predictors, SharedArrayBuffer paths, workerUrl overrides, structured clone constraints, or browser and node worker parity.'
argument-hint: 'Describe the transport strategy, current phase in Worker_Friendly_Network_Serialization_Fastpath.md, target environment, and whether this pass is reconnaissance, implementation, benchmark, or validation.'
user-invocable: true
disable-model-invocation: false
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Worker Inference Transport Playbook

Use this skill when NeatapticTS work touches the Phase 4 worker payload layer:
stable inference IR extraction, transport encoding, predictor construction, or
cross-environment worker communication.

This skill is the canonical workflow for
`plans/Worker_Friendly_Network_Serialization_Fastpath.md`. It owns the durable
knowledge for the four-strategy transport ladder, portability contracts,
feature gating, and the validation cadence that keeps worker transport honest.

When tracker files need updating, `tracker-handoff` owns plan or log shape.
When sequencing or dependency tension is unclear, use `plan-alignment`.

See [worker transport sources](./references/worker-transport-sources.md) for
platform notes, cost models, and attribution.

## Scope Boundary

- In scope: `NetworkInferenceIR`, transport encoding, transfer lists,
  `createInferencePredictor(...)`, `workerUrl` delivery rules, portable or
  transferable payloads, `MessageChannel` transport, `SharedArrayBuffer` and
  `Atomics` gating, browser and Node worker parity, inline-blob worker script
  delivery, worker transport benchmarks.
- Out of scope: worker pool scheduling and load balancing (owned by
  `multithread-evaluation`), checkpoint persistence (owned by
  `checkpointing-persistence`), parameter-vector or optimizer handoff (owned by
  `hybrid-training-interop`), ONNX graph import or export (owned by
  `onnx-work`), and demo-only workaround logic.

## When to Use

- A new transport strategy from
  `plans/Worker_Friendly_Network_Serialization_Fastpath.md` is being started.
- A worker payload is too large, too slow, or too environment-specific.
- `postMessage(...)` copy cost, transfer-list handling, or cloned object drift
  is the active problem.
- Recurrent reset semantics for a predictor are ambiguous.
- Browser worker delivery, blob worker fallback, or CSP-sensitive workerUrl
  behavior needs hardening.
- Shared memory is being considered and the task needs explicit fallback rules.

## Transport Ladder

Treat the plan's four strategies as an explicit progression rather than four
interchangeable implementation styles.

1. `PortableInferencePayload`
   - Universal fallback.
   - Best for correctness-first bring-up, strict CSP hosts, and inspection.
2. `TransferableInferencePayload`
   - Same semantic model with lower copy cost.
   - Use when message volume is high and typed-array ownership transfer is safe.
3. `InferenceChannel`
   - Persistent port pair with warm worker-side predictor state.
   - Use when startup cost dominates and repeated requests reuse one worker.
4. `SharedInferenceWorker`
   - Lowest-latency shared-memory path.
   - Use only when cross-origin isolation, synchronization discipline, and
     fallback behavior are all explicit.

Do not skip the ladder without a concrete user request or a validated reason
recorded in the plan.

## Core Contracts

### Determinism contract

- The same `Network`, same extracted inference IR, same input vector, and same
  reset state must produce the same output across all strategies within the
  stated floating-point tolerance.
- IR extraction must be stable enough that repeated extraction from the same
  frozen network is byte-identical or deep-equal identical, depending on the
  representation.
- Recurrent state reset must be explicit. A predictor may keep warm state
  between calls only when its public contract names that behavior.

### Portability contract

- Every optimized strategy must preserve a portable fallback surface.
- Any feature that depends on `SharedArrayBuffer`, CSP exceptions, or custom
  worker delivery must expose a documented capability check and a documented
  fallback.
- The payload contract must remain worker-consumable without demo-local
  message conventions.

### Cost model

Use this simple model when comparing strategies:

- clone cost: $T_{clone} = O(n)$ bytes copied per message
- transfer cost: $T_{transfer} \approx O(1)$ ownership move plus setup
- shared memory cost: $T_{shared} \approx O(1)$ setup plus synchronization and
  cache-coherency overhead

Choose the next strategy only if the additional complexity is justified by a
measured bottleneck, not by intuition alone.

## Task Packet

Pass a compact packet that includes:

- active plan phase and step,
- target strategy,
- environment scope: browser, Node, or both,
- portability constraints: CSP, workerUrl, same-origin, COOP or COEP,
- correctness invariant,
- validation target: focused Jest, benchmark, browser smoke, or all three.

Compact example:

```text
Use worker-inference-transport for Phase 2 transferable payloads.
Plan: plans/Worker_Friendly_Network_Serialization_Fastpath.md Step 2.4.
Environment: browser and Node.
Constraint: transfer only owned ArrayBuffers, keep portable fallback intact.
Invariant: predictor output must match portable mode for the same inputs.
Validate with: focused worker-payload tests, multithreading tests if loader code changes, and a roundtrip transport benchmark.
```

## Required Workflow

1. Read `plans/README.md`, then
   `plans/Worker_Friendly_Network_Serialization_Fastpath.md`.
2. Read the nearest relevant README surfaces before source reads.
   - When touching network payload code, start with the nearest
     `src/architecture/network/**/README.md`.
   - When touching worker loaders or runtime adapters, read
     `src/multithreading/README.md` and the environment boundary first.
3. Identify the exact strategy in scope and the lowest valid fallback.
4. Write down the invariants explicitly before editing:
   - output equivalence,
   - reset semantics,
   - transport ownership,
   - environment feature detection.
5. Add or update the smallest focused red-phase test that would fail if the
   transport contract is wrong.
6. Implement the change in the smallest owner-local boundary.
7. Run the same focused validation immediately after the first substantive
   change.
8. Run `coverage-guard` on every touched `src/` file before calling the step
   complete.
9. If browser worker delivery or environment adapters were touched, validate the
   fallback behavior explicitly.
10. Update docs or JSDoc so the payload surface explains strategy tradeoffs and
    fallback semantics.
11. Update the plan step once the code and validation are both green.

## Environment Rules

### Browser workers

- Prefer capability checks over environment-name checks.
- Blob or data-url worker delivery must document CSP implications.
- `SharedArrayBuffer` use must state that cross-origin isolation is required.
- Dedicated worker startup URLs should be bundler-safe and explicitly overridable
  when CSP or packaging constraints exist.

### Node workers

- Prefer custom `MessageChannel` ports for focused protocols instead of pushing
  every concern through the default parent channel.
- Do not transfer pooled or ambiguous `Buffer` storage.
  Transfer only owned `ArrayBuffer`s from typed arrays you control.
- Handle `'error'`, `'exit'`, and `'messageerror'` as first-class failure paths.

### Shared-memory paths

- Treat shared memory as opt-in, not default.
- Keep shared control blocks versioned and documented.
- Model coordination around `Atomics` and explicit reset or teardown semantics.
- Always provide a lower-tier fallback path.

## Validation Cadence

- Focused owner-local tests for the touched worker payload boundary.
- Existing multithreading tests when loader or adapter code changes, especially:
  - `src/multithreading/multi.utils.test.ts`
  - `src/multithreading/multi.test.ts`
  - `src/multithreading/workers/workers.test.ts`
- A benchmark or timing probe when the reason for the change is transport cost.
- `npm run test:silent` only when the active step packet or user explicitly requires repo-wide confirmation; otherwise, report the focused slice result as the gate evidence.

## Guardrails

- Do not couple a public payload API to Astro Bird-specific message shapes.
- Do not treat `SharedArrayBuffer` as universally available.
- Do not rely on structured clone preserving prototypes, accessors, or class
  identity.
- Do not mutate or reuse a transferred buffer after ownership has moved.
- Do not move pool scheduling concerns into this layer; that belongs to
  `multithread-evaluation`.
- Do not hide predictor reset behavior behind implicit worker lifecycle rules.
- Do not ship an optimized strategy without a documented portable fallback.

## Expected Final Output

A strong worker-transport pass should report:

- the plan phase and strategy targeted,
- the transport contract changed,
- the fallback behavior preserved,
- focused validation results,
- coverage result for touched `src/` files,
- benchmark evidence when performance is part of the claim,
- the updated plan step.
