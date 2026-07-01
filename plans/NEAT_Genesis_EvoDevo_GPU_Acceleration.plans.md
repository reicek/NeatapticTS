# NEAT Genesis EvoDevo: GPU Acceleration

**Status:** [WIP]

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
  contract is "same GPU device + same inputs → same output", matching CPU
  results within a documented float tolerance.
- It does **not** require a Node.js WebGPU backend for unit tests; mocked
  device tests plus browser smoke are sufficient for the first pass.

## Current state

Claim: 05-green-testing @ 2026-06-30T20:18:14-04:00 (Phase 3 Step 07 [DONE])

```yaml
PlanUpdate:
  slice_id: 03-07
  changed_files:
    - plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  validation:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/'
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
    - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
    - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
    - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
  artifacts:
    - coverage/lcov.info
  next: 'Phase 4 Step 01'
  notes:
    - 'Full Phase 3 GPU red-test suite: 8 suites, 54 passed, 15 failed. All failures are honest placeholder seams; no syntax/import/setup failures.'
    - 'Per-suite: capability PASS; device FAIL (5/5 not-implemented); buffer PASS; kernel PASS; parity FAIL (5/5 zero-placeholder tolerance); batched FAIL (2/7 dispatch/read-back placeholders); fallback FAIL (1/4 zero-placeholder parity); racing FAIL (1/8 zero-placeholder parity).'
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.test.ts does not exist; left as a Phase 4 integration surface.'
    - 'Removed stale duplicate ## Latest validation evidence section that contained an outdated green-light marker.'
    - 'Plan-sync gate passes; step-packet gate passes with planReadinessWarnings for the active red-testing and implementing packets (no green-light marker yet).'
    - 'Hand-off to 04-implementing requires a fresh 01-planning verification that records green-light: true in ## Latest validation evidence.'
```

## Latest validation evidence

`green-light: true`

Phase 3 is `[DONE]` and compressed to `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md`. Phase 4 Step 01 is `[WIP]` and ready for `04-implementing` dispatch. Independent `01-planning` verification confirms:

- Plan structure is valid: `validate-plan-sync.mjs` → PASS; `validate-plan-phase-packets.mjs` → PASS.
- MCP gates: `plan-sync` → pass; `step-packet` → pass with **zero** `planReadinessWarnings`.
- `plan-readiness.gate.mjs` → `greenLightFound: true`, `sectionFound: true`.
- Phase 3 red-test suite is complete and honest: the aggregate `src/architecture/network/gpu/` Jest run reports **8 test suites, 54 tests passed, 15 tests failed**, all on expected placeholder seams.
  - `network.gpu.capability.test.ts` → PASS.
  - `network.gpu.device.test.ts` → FAIL (5 failed): `requestGPUDevice not implemented` ×4, `isDeviceReady not implemented` ×1.
  - `network.gpu.buffer.test.ts` → PASS.
  - `network.gpu.kernel.test.ts` → PASS.
  - `network.gpu.parity.test.ts` → FAIL (5 failed): zero-placeholder GPU output outside tolerance vs CPU reference.
  - `network.gpu.batched.test.ts` → FAIL (2 failed): `device.recorded.submissions.length` expected `batchSize`, received `0`; `device.recorded.mapAsyncCalls.length` expected `>0`, received `0`.
  - `network.gpu.fallback.test.ts` → FAIL (1 failed): zero-placeholder GPU output vs CPU reference.
  - `network.gpu.racing.test.ts` → FAIL (1 failed): zero-placeholder GPU batch output vs CPU reference.
- `npx tsc --noEmit -p tsconfig.json` → PASS (exit 0).
- Hand-off target: Phase 4 Step 01 slice `04-01-impl` — implement GPU capability probe and device manager.

Verified by: 01-planning @ 2026-07-02.

- 01-planning has confirmed that the repository contains **no existing GPU,
  WebGPU, WebGL, shader, or GPGPU infrastructure**.
- The CPU fast path is defined in
  `src/architecture/network/slab/network.slab.fast-path.helpers.utils.ts` and
  `network.slab.utils.ts`; eligibility already excludes gates, self-connections,
  recurrent structure, dropout, weight noise, and stochastic depth.
- The worker-compatible serialized network format in
  `src/multithreading/multi.utils.ts` encodes activation functions by stable
  index, which can be mirrored in a WGSL `switch` over the same ordered registry.
- The racing-curriculum simulation worker calls `network.activate()` once per car
  per tick in `examples/racing_curriculum/workers/simulation-worker/`.
- The NGE Core Algorithm Workstream and NGE Core Growth Engine Wiring are both
  `[DONE]` and archived, so the 8,000+ neuron / 32,000+ connection growth target
  has been demonstrated on CPU and the racing-curriculum v2 demo is unblocked.
- Phase 2 research and Phase 3 red tests are `[DONE]` and compressed to `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md`; Phase 4 Step 01 is `[WIP]`.
- Phase 3 and Phase 4 step packets have been tightened into thin, one-behavior-per-file
  execution slices; the active tracker is ready for red-test implementation.
- A reusable `webgpu` skill has been created at `.github/skills/webgpu/SKILL.md` and
  passed strict skill-frontmatter validation.

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

```yaml
phase: 1
title: 'GPU acceleration workstream planning'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_phase: 'Phase 2 — WebGPU feasibility and CPU parity baseline'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
acceptance_criteria:
  - 'Plan file is registered in plans/README.md and plans/Roadmap.md with a consistent [WIP] status'
  - 'Phase/step YAML blocks pass plan-sync and step-packet gates'
placeholder_steps:
  - 'Step 01 — Author step packets for GPU acceleration'
```

**Phase objective:** Define the GPU acceleration workstream, position it after
the completed NGE core workstreams and before the racing-curriculum v2 demo, and
author the full phase/step packet set that downstream SDLC agents will
execute.

**Stop conditions:**

- Done: the plan file exists, is indexed, and passes the sync/phase-packet
  validators.
- Hold: user must confirm scope/non-goals or execution order.
- Blocked: roadmap/README registration is malformed; route to
  `plan-sync-validation` or `00.cross-tier-helper`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync`
- `neataptic-gate-mcp:run_gate_check --gate=step-packet`
- `neataptic-gate-mcp:run_gate_check --gate=agent-graph`

#### Step 01 — Author step packets for GPU acceleration [DONE]

```yaml
phase: 1
step: 1
title: 'Author step packets for GPU acceleration'
status: '[DONE]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 01 — Survey WebGPU API and WGSL constraints'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
acceptance_criteria:
  - 'Plan file authored with phase/step YAML blocks'
  - 'README and Roadmap references updated'
  - 'Sync and step-packet gates pass'
```

Step 01 produced this plan, registered it in the index and roadmap, and
prepared all downstream phase/step packets. Phase 1 is compressed to the
observations above; the active frontier is Phase 2 research.

### Phase 2 — WebGPU feasibility and CPU parity baseline [DONE]

```yaml
phase: 2
title: 'WebGPU feasibility and CPU parity baseline'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
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
- Step 03: Activation subset and f32 precision contract defined; CPU fallback rule, absolute tolerance `5e-1` (mean `≤1e-1`), and deterministic replay policy recorded.
- Step 04: CPU baseline and demo scale targets established (racing-browser 76/288, racing-worker tiers, NGE ceiling 8k/32k); speed-up threshold `≥2×` and `<4 ms` per frame resolved.
- Step 05: Phase 3 red-test plan authored with eight test files, GPU-capability predicate, and jsdom mock-device strategy.
- Step 06: Research findings and risk register documented; reusable `webgpu` skill created and frontmatter validated.
- Step 07: Phase 2 history compressed and tracker advanced to Phase 3.

### Phase 3 — Red tests for GPU inference path [DONE]

[DONE] Phase 3 — Red tests for GPU inference path. Steps 01–07 complete; the full `src/architecture/network/gpu/` red-test suite reports 8 suites, 54 passed, 15 failed, all failures honest placeholder seams. Detailed step/slice evidence and VALIDATION_EVIDENCE blocks are archived in `plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md` under "Phase 3 — Red tests for GPU inference path".

### Phase 4 — WebGPU inference implementation [WIP]

```yaml
phase: 4
title: 'WebGPU inference implementation'
status: '[WIP]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_phase: 'Phase 5 — Green validation and coverage guard'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
  - 'browser-runtime-scout'
  - 'webgpu'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
acceptance_criteria:
  - 'GPU capability probe, buffer upload, WGSL kernel, single-network inference, batched dispatch, and transparent fallback are implemented'
  - 'Each implementation slice turns its matching red test green while keeping prior slices green'
  - 'Racing-curriculum worker can invoke the GPU path when the batch is eligible'
placeholder_steps:
  - 'Step 01 — Implement GPU capability probe and device manager'
  - 'Step 02 — Implement GPU buffer allocator and slab-to-GPU upload'
  - 'Step 03 — Implement WGSL activation kernel and pipeline factory'
  - 'Step 04 — Implement single-network GPU inference and parity'
  - 'Step 05 — Implement batched multi-agent GPU inference'
  - 'Step 06 — Implement transparent fallback and racing worker integration'
  - 'Step 07 — Green validation and tracker handoff'
```

**Phase objective:** Build the GPU inference path end-to-end while leaving CPU
paths untouched except for the transparent dispatch seam.

**Stop conditions:**

- Done: all implementation steps/slices pass green validation.
- Hold: a WebGPU API detail is unstable; mock or guard it.
- Blocked: a CPU correctness contract would be violated; escalate.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`

#### Step 01 — Implement GPU capability probe and device manager [WIP]

```yaml
phase: 4
step: 1
title: Implement GPU capability probe and device manager
status: '[WIP]'
goal: implementing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
copy_paste: true
next_step: Phase 4 Step 02 — Implement GPU buffer allocator and slab-to-GPU upload
skills:
  - implementation-standards
  - browser-runtime-scout
  - webgpu
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.capability.test.ts
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.device.test.ts
acceptance_criteria:
  - '@webgpu/types (or equivalent GPU type stubs) are added without breaking the existing build'
  - canUseGPU matches all assertions in network.gpu.capability.test.ts
  - requestGPUDevice and isDeviceReady match all assertions in network.gpu.device.test.ts
  - No CPU path behavior changes
slices:
  - slice_id: 04-01-impl
    title: Add WebGPU types and implement GPU capability probe/device manager
    status: '[WIP]'
    goal: implementing
    estimate_hours: 6
    files_to_change:
      - package.json
      - tsconfig.json
      - src/architecture/network/gpu/network.gpu.types.ts
      - src/architecture/network/gpu/network.gpu.capability.ts
      - src/architecture/network/gpu/network.gpu.device.ts
    acceptance_criteria:
      - Type declarations allow GPU types in source and tests without compilation errors
      - Build and lint pass
      - canUseGPU passes all capability red tests
      - requestGPUDevice and isDeviceReady pass all device red tests
      - Device-lost state is tracked and causes subsequent canUseGPU to return false
    parallelizable: false
    dependencies: []
    next_slice: 04-01-green
  - slice_id: 04-01-green
    title: 'Green check: confirm capability and device red tests pass'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - src/architecture/network/gpu/network.gpu.capability.ts
      - src/architecture/network/gpu/network.gpu.device.ts
    acceptance_criteria:
      - Run the focused Jest suites and confirm tests pass
      - 'Coverage guard reaches 100% on touched src/ files'
    parallelizable: false
    dependencies:
      - 04-01-impl
    next_slice: Phase 4 Step 02
```

**User instruction:** Implement the WebGPU capability predicate and device probe
so that the Phase 3 capability and device red tests turn green. Add the minimal
WebGPU type declarations needed by the project without changing CPU behavior.

**Step objective:** Provide a reliable, testable answer to "can this network run
on the GPU?" for a given WebGPU device.

**Stop conditions:**

- Done: capability and device tests pass, build/lint are green.
- Hold: adapter limits vary too widely; document the conservative default.
- Blocked: type declarations break the build; escalate.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.capability.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.device.test.ts`

#### Step 02 — Implement GPU buffer allocator and slab-to-GPU upload [PLANNED]

```yaml
phase: 4
step: 2
title: Implement GPU buffer allocator and slab-to-GPU upload
status: '[PLANNED]'
goal: implementing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
copy_paste: true
next_step: Phase 4 Step 03 — Implement WGSL activation kernel and pipeline factory
skills:
  - implementation-standards
  - webgpu
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.buffer.test.ts
acceptance_criteria:
  - uploadNetworkToGPU passes all buffer red tests
  - 'Buffer sizes, usage flags, and write offsets match the Phase 2 binding contract'
  - No re-layout of the CPU slab occurs; arrays are uploaded as-is
slices:
  - slice_id: 04-02-allocator
    title: Implement GPU buffer allocator
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - src/architecture/network/gpu/network.gpu.buffer.ts
    acceptance_criteria:
      - 'createGPUBuffer(device, byteLength, label, usage) produces a storage buffer with COPY_DST'
      - Allocator respects device.limits.maxBufferSize and maxStorageBufferBindingSize
    parallelizable: false
    dependencies:
      - 04-01-probe
    next_slice: 04-02-upload
  - slice_id: 04-02-upload
    title: Implement slab-to-GPU upload
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - src/architecture/network/gpu/network.gpu.buffer.ts
      - src/architecture/network/gpu/network.gpu.types.ts
    acceptance_criteria:
      - uploadNetworkToGPU creates the buffers declared in the Phase 2 contract
      - Each slab array is written exactly once via queue.writeBuffer
      - Buffer-to-binding index mapping is stable and exported for the kernel
    parallelizable: false
    dependencies:
      - 04-02-allocator
    next_slice: 04-02-green
  - slice_id: 04-02-green
    title: 'Green check: confirm red tests fail for the right reason'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - src/architecture/network/gpu/network.gpu.buffer.ts
      - src/architecture/network/gpu/network.gpu.types.ts
    acceptance_criteria:
      - Run the focused Jest suite and capture failing assertions
      - 'Confirm failures target missing GPU functions, not syntax/setup errors'
    parallelizable: false
    dependencies:
      - 04-02-upload
    next_slice: Phase 4 Step 03
```

**User instruction:** Implement the WebGPU buffer allocator and the
`uploadNetworkToGPU` seam. Create the exact storage-buffer set and upload order
specified in the research brief, reusing the existing CPU slab arrays without
re-serialization.

**Step objective:** Get network data onto the GPU with the same layout the CPU
slab already uses.

**Stop conditions:**

- Done: buffer red tests pass and the upload contract is stable.
- Hold: alignment padding is needed; document the adjustment.
- Blocked: slab arrays are not externally visible; spike a minimal accessor.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.buffer.test.ts`

#### Step 03 — Implement WGSL activation kernel and pipeline factory [PLANNED]

```yaml
phase: 4
step: 3
title: Implement WGSL activation kernel and pipeline factory
status: '[PLANNED]'
goal: implementing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
copy_paste: true
next_step: Phase 4 Step 04 — Implement single-network GPU inference and parity
skills:
  - implementation-standards
  - webgpu
  - nge-core-scout
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.kernel.test.ts
acceptance_criteria:
  - compileActivationKernel passes all kernel red tests
  - WGSL source contains the activation switch and storage-buffer bindings in the correct order
  - Pipelines are cached by topology hash
slices:
  - slice_id: 04-03-registry
    title: Implement the WGSL activation function registry
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - src/architecture/network/gpu/network.gpu.activation.wgsl.ts
    acceptance_criteria:
      - Each supported activation maps to a WGSL function by the same index as the worker serialization contract
      - Unsupported activations are excluded from the generated switch
    parallelizable: false
    dependencies:
      - 04-02-upload
    next_slice: 04-03-kernel
  - slice_id: 04-03-kernel
    title: Implement the compute shader and pipeline factory
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 5
    files_to_change:
      - src/architecture/network/gpu/network.gpu.kernel.ts
      - src/architecture/network/gpu/network.gpu.types.ts
    acceptance_criteria:
      - compileActivationKernel generates a compute pipeline with correct bind-group layout
      - Shader dispatches one thread per node in topological order
      - Pipeline is cached per topology and reused
    parallelizable: false
    dependencies:
      - 04-03-registry
    next_slice: 04-03-green
  - slice_id: 04-03-green
    title: 'Green check: confirm red tests fail for the right reason'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - src/architecture/network/gpu/network.gpu.kernel.ts
      - src/architecture/network/gpu/network.gpu.types.ts
    acceptance_criteria:
      - Run the focused Jest suite and capture failing assertions
      - 'Confirm failures target missing GPU functions, not syntax/setup errors'
    parallelizable: false
    dependencies:
      - 04-03-kernel
    next_slice: Phase 4 Step 04
```

**User instruction:** Implement the WGSL activation kernel generator and the
compute-pipeline factory. The shader must switch on the same activation indices
used by the worker serialization contract and dispatch over nodes in
topological order.

**Step objective:** Have a compilable, bindable GPU kernel that matches the
uploaded slab buffers.

**Stop conditions:**

- Done: kernel red tests pass; the shader source is inspectable through the mock.
- Hold: an activation has no stable f32 implementation; mark it unsupported.
- Blocked: topology order cannot be expressed in WGSL; escalate.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.kernel.test.ts`

#### Step 04 — Implement single-network GPU inference and parity [PLANNED]

```yaml
phase: 4
step: 4
title: Implement single-network GPU inference and parity
status: '[PLANNED]'
goal: implementing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
copy_paste: true
next_step: Phase 4 Step 05 — Implement batched multi-agent GPU inference
skills:
  - implementation-standards
  - webgpu
  - determinism-scout
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.parity.test.ts
acceptance_criteria:
  - 'activateGPU(device, network, input) exists and is testable through the parity red tests'
  - 'GPU output matches CPU output within the documented tolerance for small, racing-browser, and NGE-cap shapes'
  - CPU path remains the default; useGPU opt-in is honored
slices:
  - slice_id: 04-04-inference
    title: Implement single-network GPU inference
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 5
    files_to_change:
      - src/architecture/network/gpu/network.gpu.inference.ts
      - src/architecture/network/network.ts
    acceptance_criteria:
      - 'activateGPU uploads input, dispatches kernel, reads back output, and returns a Float32Array'
      - 'Network.activate(input, { useGPU: true }) routes to activateGPU when canUseGPU is true'
      - CPU path remains unchanged and is the default
    parallelizable: false
    dependencies:
      - 04-03-kernel
    next_slice: 04-04-parity
  - slice_id: 04-04-parity
    title: Confirm CPU-vs-GPU parity within tolerance
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 3
    files_to_change:
      - src/architecture/network/gpu/network.gpu.parity.test.ts
    acceptance_criteria:
      - 'Small, racing-browser, and NGE-cap parity tests pass within 5e-1 absolute and mean ≤ 1e-1'
      - Coverage guard passes on touched src/ files
    parallelizable: false
    dependencies:
      - 04-04-inference
    next_slice: Phase 4 Step 05
```

**User instruction:** Wire a single-network GPU inference path into
`Network.activate` behind an opt-in `useGPU` flag. Use the capability predicate
and the uploaded slab/kernel to run inference and return a result comparable to
the CPU path.

**Step objective:** Make one network run correctly on the GPU.

**Stop conditions:**

- Done: parity red tests pass within tolerance.
- Hold: parity is outside tolerance; tune f32 math or tighten the supported
  activation subset.
- Blocked: CPU output changes when GPU path is added; escalate.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.parity.test.ts`

#### Step 05 — Implement batched multi-agent GPU inference [PLANNED]

```yaml
phase: 4
step: 5
title: Implement batched multi-agent GPU inference
status: '[PLANNED]'
goal: implementing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
copy_paste: true
next_step: Phase 4 Step 06 — Implement transparent fallback and racing worker integration
skills:
  - implementation-standards
  - webgpu
  - nge-benchmark-scout
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.batched.test.ts
acceptance_criteria:
  - 'batchActivate(device, networks, inputs) passes all batched red tests'
  - One output is returned per network; empty batches submit no GPU work
  - Networks with shared topology reuse one pipeline
slices:
  - slice_id: 04-05-batched
    title: Implement multi-agent batch dispatch
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 5
    files_to_change:
      - src/architecture/network/gpu/network.gpu.batched.ts
    acceptance_criteria:
      - 'batchActivate uploads all inputs, reuses topology pipelines, and reads back one output per network'
      - 'Empty networks array returns [] without creating a command encoder'
      - Each network output matches its CPU counterpart within tolerance
    parallelizable: false
    dependencies:
      - 04-04-parity
    next_slice: 04-05-green
  - slice_id: 04-05-green
    title: 'Green check: confirm red tests fail for the right reason'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - src/architecture/network/gpu/network.gpu.batched.ts
    acceptance_criteria:
      - Run the focused Jest suite and capture failing assertions
      - 'Confirm failures target missing GPU functions, not syntax/setup errors'
    parallelizable: false
    dependencies:
      - 04-05-batched
    next_slice: Phase 4 Step 06
```

**User instruction:** Implement `batchActivate` so that the racing worker can
evaluate multiple agents in one GPU dispatch when they share topology. Keep the
interface simple enough to drop into the existing worker loop.

**Step objective:** Unlock the primary GPU use case: batched population
evaluation.

**Stop conditions:**

- Done: batched red tests pass and pipeline reuse is verified.
- Hold: batch size is limited by buffer binding limits; document the cap.
- Blocked: output readback corrupts results; escalate.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.batched.test.ts`

#### Step 06 — Implement transparent fallback and racing worker integration [PLANNED]

```yaml
phase: 4
step: 6
title: Implement transparent fallback and racing worker integration
status: '[PLANNED]'
goal: implementing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
copy_paste: true
next_step: Phase 4 Step 07 — Green validation and tracker handoff
skills:
  - implementation-standards
  - nge-benchmark-scout
  - browser-runtime-scout
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.fallback.test.ts
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.test.ts
acceptance_criteria:
  - 'All fallback red tests pass: missing navigator.gpu, lost device, unsupported activation, and disabled float32 mode'
  - 'All racing demo seam red tests pass: batch eligibility helper and per-network useGPU hint'
  - CPU-only behavior is unchanged when the GPU path is not triggered
slices:
  - slice_id: 04-06-fallback
    title: Implement transparent fallback rules and device recovery
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - src/architecture/network/gpu/network.gpu.fallback.ts
      - src/architecture/network/network.ts
    acceptance_criteria:
      - 'Missing navigator.gpu, lost device, and unsupported activations route to CPU without throwing'
      - Device-lost listener disables GPU path until recreated
      - CPU output remains unchanged
    parallelizable: false
    dependencies:
      - 04-05-batched
    next_slice: 04-06-racing
  - slice_id: 04-06-racing
    title: Integrate GPU path into the racing-curriculum worker
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.ts
      - examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.test.ts
    acceptance_criteria:
      - 'shouldUseGPUForBatch(agentCount, network) matches the Phase 2 crossover thresholds'
      - Worker passes useGPU hint to network.activate when the batch is eligible
      - Existing worker tests remain green
    parallelizable: false
    dependencies:
      - 04-06-fallback
    next_slice: 04-06-green
  - slice_id: 04-06-green
    title: 'Green check: confirm red tests fail for the right reason'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.ts
      - examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.test.ts
    acceptance_criteria:
      - Run the focused Jest suite and capture failing assertions
      - 'Confirm failures target missing GPU functions, not syntax/setup errors'
    parallelizable: false
    dependencies:
      - 04-06-racing
    next_slice: Phase 4 Step 07
```

**User instruction:** Implement the transparent fallback rules and wire the GPU
path into the racing-curriculum simulation worker. The worker should decide when
a batch is large enough to justify GPU overhead and pass the `useGPU` hint to
`network.activate`.

**Step objective:** Make the GPU path usable by the primary NGE demo while
preserving CPU-only behavior.

**Stop conditions:**

- Done: fallback and racing red tests pass; existing worker tests remain green.
- Hold: worker refactor is larger than expected; split into a separate slice.
- Blocked: worker controller changes break unrelated demos; escalate.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.fallback.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.test.ts`

#### Step 07 — Green validation and tracker handoff [PLANNED]

```yaml
phase: 4
step: 7
title: Green validation and tracker handoff
status: '[PLANNED]'
goal: green-testing
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
copy_paste: true
next_step: Phase 5 Step 01 — Run focused GPU and regression suites
skills:
  - green-testing
  - coverage-guard
  - tracker-handoff
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.test.ts
  - npx jest --config=jest.config.mjs --no-cache --coverage --collectCoverageFrom="src/architecture/network/gpu/**/*.ts" --testPathPatterns=src/architecture/network/gpu/
  - npm run lint
acceptance_criteria:
  - All Phase 3 red tests now pass
  - Focused GPU suites and existing regression suites remain green
  - 100% coverage on all touched src/ files under src/architecture/network/gpu/
  - Lint passes
  - 'Phase 4 marked [DONE] and Phase 5 Step 01 marked [WIP]'
```

**User instruction:** Run the full focused GPU test suite, the regression suites
that touch `Network.activate` and the racing worker, the coverage guard on
`touched src/architecture/network/gpu/` files, and lint. Then compress Phase 4
and advance to Phase 5.

**Step objective:** Prove the implementation is correct, covered, and does not
regress existing behavior.

**Stop conditions:**

- Done: all gates pass, Phase 4 is `[DONE]`, Phase 5 is active.
- Hold: a regression or coverage gap appears; route back to the smallest prior
  slice.
- Blocked: a workflow gate fails; route to `05-green-testing` or
  `07-logging`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --coverage --collectCoverageFrom="src/architecture/network/gpu/**/*.ts" --testPathPatterns=src/architecture/network/gpu/`
- `npm run lint`

### Phase 5 — Green validation and coverage guard [PLANNED]

```yaml
phase: 5
title: 'Green validation and coverage guard'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_phase: 'Phase 6 — Documentation and usage contract'
skills:
  - 'plan-alignment'
  - 'green-testing'
  - 'coverage-guard'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
acceptance_criteria:
  - 'All red tests pass; focused and regression suites remain green'
  - '100% coverage on all touched src/ files'
  - 'Browser smoke confirms racing demo still runs'
placeholder_steps:
  - 'Step 01 — Run red-test suite and confirm parity within tolerance'
  - 'Step 02 — Run full regression suite and lint/build gates'
  - 'Step 03 — Run coverage guard on touched src/ files'
  - 'Step 04 — Run racing demo browser smoke with Chrome DevTools MCP'
  - 'Step 05 — Run benchmark harness comparing CPU vs GPU latency'
  - 'Step 06 — Record evidence and triage carry-forward blockers'
  - 'Step 07 — Logging and tracker handoff'
```

**Phase objective:** Prove the GPU path is correct, covered, and does not
regress existing behavior.

**Stop conditions:**

- Done: all validation steps pass.
- Hold: a performance regression appears; triage before claiming done.
- Blocked: a green gate fails; route back to the smallest relevant prior phase.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`

### Phase 6 — Documentation and usage contract [PLANNED]

```yaml
phase: 6
title: 'Documentation and usage contract'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
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

**Stop conditions:**

- Done: docs build passes and the usage contract is clear.
- Hold: example cannot run headlessly; document the manual verification steps.
- Blocked: docs generation breaks; route to `06-documenting`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`

### Phase 7 — Tracker closure [PLANNED]

```yaml
phase: 7
title: 'Tracker closure'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
copy_paste: true
next_phase: 'Archive to plans/completed/'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md'
acceptance_criteria:
  - 'All phases are [DONE] and green-gated'
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
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json`
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json`

## Validation gates

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync`
- `neataptic-gate-mcp:run_gate_check --gate=step-packet`
- `neataptic-gate-mcp:run_gate_check --gate=agent-graph`

## Deferred questions

- Exact f32 tolerance threshold for CPU-vs-GPU parity tests: resolved to **absolute `5e-1` (0.5)** hard gate with **mean absolute error `≤ 1e-1`** soft diagnostic; relative tolerance is not used because controller outputs near zero make relative error unstable.
- Minimum network size / agent count at which GPU overhead pays off: resolved to **≥ 2× speed-up and total per-frame inference < 4 ms**; cross-over at current CPU costs is roughly **≥ 6 NGE-cap agents** or **≥ 130 racing-browser agents** per batch.
- Whether the first racing-curriculum evaluation pack allows GPU opt-in or stays CPU-only for cross-machine determinism (to be decided with user input before Phase 5).

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Active workstream: plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md
Current boundary: Phase 4 Step 01 — Implement GPU capability probe and device manager [WIP]
  - Slice 04-01-impl (status [PLANNED]): implement `requestGPUDevice`, `isDeviceReady`,
    `canUseGPU`, and `getGPUDevice` in `src/architecture/network/gpu/network.gpu.capability.ts`
    and `network.gpu.device.ts`; add minimal `@webgpu/types` type stubs to
    `package.json`/`tsconfig.json` if needed.
  - Slice 04-01-green (status [PLANNED]): run the focused capability/device suite and
    confirm all Phase 3 red tests for capability and device turn green.
What is already covered:
  - Phase 1 planning and step packets authored.
  - Phase 2 Steps 01–07 are [DONE]; durable log archived at plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md.
  - Phase 3 Steps 01–07 are [DONE]; detailed red-test evidence archived under
    "Phase 3 — Red tests for GPU inference path" in the same .logs.md file.
  - WebGPU/WGSL survey captured in WebGPU_architecture/*.md; compact playbook in .github/skills/webgpu/SKILL.md.
  - Buffer binding contract, activation subset, precision contract, fallback rules,
    determinism policy, and risk register are documented.
  - Phase 4 step packets are tightened into green-only slices (no standalone red tests needed
    because Phase 3 red tests already exist).
  - GPU capability predicate and jsdom mock-device strategy are defined so tests can assert
    on adapter/device requests, buffer creation, WGSL source, and dispatch dimensions without
    a real GPU.
Next narrow task: dispatch 04-implementing for slice 04-01-impl.
  1. Implement `requestGPUDevice`/`isDeviceReady` plus device-manager helpers, honoring the
     mock-device recording contract used by the Phase 3 tests.
  2. Run `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.capability.test.ts`.
  3. Run `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.device.test.ts`.
  4. Confirm TypeScript (`npx tsc --noEmit -p tsconfig.json`) and lint/prettier checks pass for the touched files.
  5. Update the tracker: mark slice 04-01-impl [DONE] and advance to slice 04-01-green.
Required validations:
  - `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md` → PASS.
  - `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md` → PASS.
  - `neataptic-gate-mcp:run_gate_check --gate=plan-sync` → pass.
  - `neataptic-gate-mcp:run_gate_check --gate=step-packet` → pass with zero planReadinessWarnings.
Known cautions: no GPU implementation code exists in the repo yet; jsdom tests have no real
navigator.gpu, so the first implementation must satisfy the mocked-device recording contract
captured in the Phase 3 tests. CPU paths must remain untouched except for the transparent
dispatch seam.
```

## Research brief

The full Phase 2 research brief, risk register, WebGPU/WGSL survey, and the
Phase 3 red-test plan have been moved to the durable log at
`plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md` to keep the active tracker
focused on the red-test frontier. A compact playbook remains in
`WebGPU_architecture/webgpu.architecture.md` and `.github/skills/webgpu/SKILL.md`.
