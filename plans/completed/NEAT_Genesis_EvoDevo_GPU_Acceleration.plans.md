# NEAT Genesis EvoDevo: GPU Acceleration

**Status:** [DONE]

## Scope

Add an optional, transparent WebGPU inference fast path to `Network.activate()`
for acyclic, non-gated, non-recurrent NEAT networks that already qualify for the
existing typed-array slab fast path. The GPU path must:

- reuse the existing SoA/CSR slab layout (`weights`, `from`, `to`, `flags`,
  `outStart`/`outOrder`) so no expensive re-serialization is required;
- support the full built-in activation function registry used by the worker
  serialization contract, with deterministic f32 semantics;
- fall back to the existing CPU slab or legacy object path automatically when
  WebGPU is unavailable, the device is lost, or the network is ineligible;
- preserve the bitwise-identical replay contract required by deterministic
  evaluation packs in the racing curriculum and other NGE benchmarks;
- be exercised first through the racing-curriculum worker controller because
  that is the most inference-heavy NGE demo and its per-car `network.activate()`
  seam is already isolated.

### Non-goals

- This plan does **not** port training, mutation, or evolution to the GPU.
- It does **not** replace the CPU paths; CPU remains the source-of-truth.
- It does **not** guarantee cross-device bitwise-identical output; the
  contract is "same GPU device + same inputs ? same output", matching CPU
  results within a documented float tolerance.
- It does **not** require a Node.js WebGPU backend for unit tests; mocked
  device tests plus browser smoke are sufficient for the first pass.

## Current state

- Phase 1 — Foundation and feasibility [DONE]
- Phase 2 — Slab layout and WGSL kernels [DONE]
- Phase 3 — Integration and fallback [DONE]
- Phase 4 — End-to-end validation [DONE]
- **Phase 5 — Green validation and coverage guard [DONE]**
- **Phase 6 — Documentation and usage contract [DONE]**
- Phase 7 — Tracker closure [DONE]

**Active frontier:** None — workstream closed.

## Coverage backlog

- WebGPU capability probe and device-lost handling.
- Slab-to-GPU buffer upload with minimal memory re-layout.
- WGSL topological-order forward-pass kernel for the slab CSR graph.
- Per-node activation switch matching the worker activation registry.
- Single-network CPU-vs-GPU parity within tolerance.
- Batched multi-agent inference for racing-curriculum scale.
- Transparent fallback when WebGPU is absent or the network is ineligible.
- Deterministic replay guard and opt-in policy for benchmark evaluation packs.
- Browser smoke validation via Chrome DevTools MCP.
- Typedoc/JSDoc usage contract and example.

## Implementation phases

### Phase 1 — GPU acceleration workstream planning [DONE]

[DONE] Phase 1 — GPU acceleration workstream planning. Step 01 authored the full phase/step packet set, registered the plan in `plans/README.md` and `plans/Roadmap.md`, and prepared all downstream phase/step packets. Detailed evidence archived in `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md` § Phase 1.

### Phase 2 — WebGPU feasibility and CPU parity baseline [DONE]

```yaml
phase: 2
title: 'WebGPU feasibility and CPU parity baseline'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_phase: 'Phase 3 — Red tests for GPU inference path'
skills:
  - 'plan-alignment'
  - 'research-methodology'
  - 'nge-benchmark-scout'
  - 'browser-runtime-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
acceptance_criteria:
  - 'Research brief documents WebGPU availability, WGSL limits, and mapping of slab/serialized layout to GPU buffers'
  - 'Activation subset and f32 precision contract are defined'
  - 'CPU baseline harness exists for deterministic parity comparison'
placeholder_steps:
  - 'Step 01 — Survey WebGPU API and WGSL constraints'
  - 'Step 02 — Map slab/serialized network layout to GPU buffers'
  - 'Step 03 — Define activation subset and precision contract'
  - 'Step 04 — Establish deterministic CPU baseline and demo scale target'
  - 'Step 05 — Author red-test plan and GPU-capability contract'
  - 'Step 06 — Document research findings and risk register'
  - 'Step 07 — Logging and tracker handoff'
```

**Phase objective:** Produce a source-grounded research brief that confirms the
GPU path is feasible, identifies the exact data-flow seam, and records the
precision/fallback contracts before any red tests or implementation begin.

**Stop conditions:**

- Done: all Phase 2 steps are `[DONE]` and the research brief is accepted.
- Hold: user must confirm the activation subset that the first WGSL kernel will
  support.
- Blocked: a WebGPU or browser-runtime prerequisite is missing; route to
  `browser-runtime-scout` or `00.cross-tier-helper`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`

[DONE] Phase 2 — WebGPU feasibility and CPU parity baseline. Detailed research
notes, the Phase 3 red-test plan, the GPU-capability predicate, the jsdom mock
strategy, and the risk register are recorded in
`plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md`.

Phase 2 coverage:

- Step 01: WebGPU API/WGSL constraints surveyed; adapter/device probe and limit table recorded.
- Step 02: CPU slab/SoA/CSR layout mapped to GPU storage buffers; binding contract captured in `WebGPU_architecture/webgpu.architecture.md` and `.github/skills/webgpu/SKILL.md`.
- Step 03: Activation subset and f32 precision contract defined; CPU fallback rule, absolute tolerance `5e-1` (mean `=1e-1`), and deterministic replay policy recorded.
- Step 04: CPU baseline and demo scale targets established (racing-browser 76/288, racing-worker tiers, NGE ceiling 8k/32k); speed-up threshold `=2×` and `<4 ms` per frame resolved.
- Step 05: Phase 3 red-test plan authored with eight test files, GPU-capability predicate, and jsdom mock-device strategy.
- Step 06: Research findings and risk register documented; reusable `webgpu` skill created and frontmatter validated.
- Step 07: Phase 2 history compressed and tracker advanced to Phase 3.

### Phase 3 — Red tests for GPU inference path [DONE]

[DONE] Phase 3 — Red tests for GPU inference path. Steps 01–07 complete; the full `src/architecture/network/gpu/` red-test suite reports 8 suites, 54 passed, 15 failed, all failures honest placeholder seams. Detailed step/slice evidence and VALIDATION_EVIDENCE blocks are archived in `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md` under "Phase 3 — Red tests for GPU inference path".

### Phase 4 — WebGPU inference implementation [DONE]

[DONE] Phase 4 — WebGPU inference implementation. Steps 01–07 complete; all focused GPU suites pass (9 suites / 119 tests), the racing worker GPU seam passes (1 suite / 11 tests), coverage guard reports 100% on all non-declaration `src/architecture/network/gpu/**/*.ts` files, and `npm run lint` is green. Detailed step/slice evidence and VALIDATION_EVIDENCE blocks are archived in `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md` under "Phase 4 — WebGPU inference implementation".

### Phase 5 — Green validation and coverage guard [DONE]

[DONE] Phase 5: Green validation and coverage guard completed. Detailed step evidence moved to [plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md](NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md) § Phase 5.

### Phase 6 — Documentation and usage contract [DONE]

```yaml
phase: 6
title: 'Documentation and usage contract'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_phase: 'Phase 7 — Tracker closure'
skills:
  - 'plan-alignment'
  - 'documenting'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
acceptance_criteria:
  - 'Public API, opt-in policy, fallback behavior, and example are documented'
  - 'Typedoc/JSDoc generation passes'
placeholder_steps:
  - 'Step 01 — Document GPU inference public API and opt-in contract'
  - 'Step 02 — Add GPU example and racing demo README notes'
  - 'Step 03 — Update typedoc and JSDoc for GPU module'
  - 'Step 04 — Validate docs build and mermaid diagrams'
  - 'Step 05 — Logging and tracker handoff'
```

**Phase objective:** Make the GPU path discoverable and safe to consume.

**Coverage note:** Detailed evidence for Steps 01–05 archived in `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md` § Phase 6. Summary of completed work:

- Step 01 — Documented the GPU inference public API (`network.gpuDevice`, `network.activate(input, { useGPU: true })`) and the opt-in/fallback/tolerance contract in `docs/webgpu-inference.md`, `README.md`, and source JSDoc; generated GPU README uses `network.gpu.activate.ts` as chapter intro with Mermaid flowchart and Wikipedia citation.
- Step 02 — Added `examples/racing_curriculum/gpu-enabled-racing.example.ts` and a GPU opt-in section to `examples/racing_curriculum/README.md` that explains the opt-in contract, eligibility, and fallback behavior.
- Step 03 — Audited and updated JSDoc across `src/architecture/network/gpu/*.ts` and `src/architecture/network/network.ts`; removed test/fixture jargon, added module-level educational JSDoc to `network.gpu.batched.ts` and `network.gpu.fallback.ts`, and confirmed no stale `activateGpu` references or broken `{@link}` tags.
- Step 04 — `npm run docs` and `npm run docs:quality:gate` pass with no GPU-related Typedoc warnings; Mermaid diagrams in `docs/architecture/network/gpu/README.md` and `WebGPU.md` validated.
- Step 05 — Phase 6 history compressed into the matching `.logs.md` record; tracker handed off to Phase 7.

**Stop conditions:**

- Done: docs build passes and the usage contract is clear.
- Hold: example cannot run headlessly; document the manual verification steps.
- Blocked: docs generation breaks; route to `06-documenting`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`

### Phase 7 — Tracker closure [DONE]

```yaml
phase: 7
title: 'Tracker closure'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_phase: 'Archive to plans/completed/'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'logging'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
acceptance_criteria:
  - 'All prior phases are [DONE] and green-gated'
  - 'Plan/log pair archived to plans/completed/'
placeholder_steps:
  - 'Step 01 — Compress completed phase histories into logs'
  - 'Step 02 — Move plan/log pair to plans/completed'
  - 'Step 03 — Final gate checks and closeout'
```

**Phase objective:** Close the workstream cleanly and archive the tracker pair.

**Stop conditions:**

- Done: archive move complete and stale-wip-plans gate passes.
- Hold: carry-forward blockers remain; document them before closing.
- Blocked: closure gates fail; route to `07-logging`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json`
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json`

#### Step 01 — Compress completed phase histories into logs [DONE]

```yaml
phase: 7
step: 1
title: 'Compress completed phase histories into logs'
status: '[DONE]'
goal: 'logging'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_step: 'Phase 7 Step 02 — Move plan/log pair to plans/completed'
skills:
  - 'logging'
  - 'tracker-handoff'
  - 'plan-alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
acceptance_criteria:
  - 'Phases 1–6 histories are fully compressed into the .logs.md record'
  - 'The active plan file contains only concise [DONE] coverage notes for completed phases'
  - 'phase-compression gate passes'
  - 'log-completion-marker gate passes'
```

**Step objective:** Ensure the durable log contains all completed phase evidence and the active plan is reduced to concise coverage notes before archive.

**User instruction:** Paste this full step packet into a fresh `07-logging` session.

**Execution steps:**

1. Verify that Phases 1–6 detailed evidence exists in `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md`.
2. Confirm the active plan has concise `[DONE]` coverage notes for Phases 1–6 and no orphaned verbose step content.
3. Run `phase-compression.gate` and `log-completion-marker.gate`; fix any reported drift.
4. Update plan `## Current state` to mark Phase 7 Step 01 `[WIP]`.

**Stop conditions:**

- Done: compression gates pass and the plan is ready for archive move.
- Hold: a phase still has verbose content in the plan; compress it into the log first.
- Blocked: a gate reports a structural issue that cannot be resolved by compression alone; route to `07-logging`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json`

## Validation gates

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync`
- `neataptic-gate-mcp:run_gate_check --gate=step-packet`
- `neataptic-gate-mcp:run_gate_check --gate=agent-graph`

## Deferred questions

- Exact f32 tolerance threshold for CPU-vs-GPU parity tests: resolved to **absolute `5e-1` (0.5)** hard gate with **mean absolute error `= 1e-1`** soft diagnostic; relative tolerance is not used because controller outputs near zero make relative error unstable.
- Minimum network size / agent count at which GPU overhead pays off: resolved to **= 2× speed-up and total per-frame inference < 4 ms**; cross-over at current CPU costs is roughly **= 6 NGE-cap agents** or **= 130 racing-browser agents** per batch.
- Whether the first racing-curriculum evaluation pack allows GPU opt-in or stays CPU-only for cross-machine determinism (to be decided with user input before Phase 5).

## Latest validation evidence

- **Phase 7 closure green light** — Phase 7 Step 01 tracker closure complete; all prior phases compressed into `plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md`; plan marked top-level `[DONE]`; plan/log pair archived to `plans/completed/`; `stale-wip-plans.gate.mjs` passes; `plans/README.md` and `plans/Roadmap.md` updated to `[DONE]` with `completed/` path; Racing Curriculum v2 now unblocked.

## Reopen conditions

This workstream is closed. If the GPU Acceleration path needs to resume (e.g., browser smoke validation at scale, new WGSL kernel coverage, or racing-curriculum worker integration), either:
- move this `.plans.md` and its same-boundary `.logs.md` back to `plans/`, or
- open a new active tracker that references `plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.{plans,logs}.md`.

Add a fresh `## Handoff query` on reopen; do not reuse this closure-era section.
