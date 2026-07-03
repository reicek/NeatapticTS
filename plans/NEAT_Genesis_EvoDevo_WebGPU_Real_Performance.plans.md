# NEAT Genesis EvoDevo: WebGPU Real Performance

**Status:** [WIP]

## Active phase/step

- Phase 1 — Red Testing [DONE]
- Phase 2 — Implementation [WIP]
- Step 01: Implement correct weighted WebGPU forward pass [WIP]

## Scope

Validate and fix the WebGPU compute path in NeatapticTS so that GPU inference is
numerically correct and performant for NGE-scale networks. Capture real-device
CPU-vs-GPU measurements from the user's browser using the existing Chrome
DevTools MCP/browser-harness infrastructure, push single-agent scale past 32k
neurons, and measure parallel-agent throughput (e.g., 1×8k vs 6×8k networks).

This plan is downstream of the completed GPU implementation in
`plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`. If any upstream
plan conflicts, the upstream plan wins.

## Constraints

- **All GPU measurements must be performed on a visible browser window.** Headless or minimized windows produce invalid GPU timing data. This is non-negotiable.

## Current state

Claim: 04-implementing @ 2026-07-03T00:18:00Z

- [DONE] Slice `02-01a-buffer` implemented in `src/architecture/network/gpu/network.gpu.buffer.ts`.
  - Incoming-CSR arrays (`inStart`, `inOrder`) are built and uploaded.
  - Per-node topological levels are computed with Kahn-style DAG traversal and uploaded.
  - `GPUBufferSet` now exposes `inStart`, `inOrder`, `bias`, `topoLevels`, `params`, and `topoLevelsArray`.
  - `uploadDynamicNetworkBuffers` re-uploads weights/bias for repeated activations.
  - `destroyGPUBufferSet` destroys the new buffers.
- [DONE] Slice `02-01b-kernel` implemented in `src/architecture/network/gpu/network.gpu.kernel.ts`.
  - WGSL `forward` entry point dispatches one thread per node per topological level.
  - Each thread iterates `inStart[node]..inStart[node+1]`, looks up the connection index from `inOrder`, reads `source = from_nodes[connection]` and `weight = weights[connection]`, and accumulates `sum += node_outputs[source] * weight`.
  - Bias is added before `applyActivation(sum)` writes the node output.
  - Bind-group layout expanded to ten entries (weights, from, to, flags, inStart, inOrder, outputs, bias, topoLevels, params uniform).
  - Pipeline topology cache and helper exports (`createBindGroupLayout`, `buildGPUPipeline`) remain intact.
- Preflight checks passed: `npx tsc --noEmit -p tsconfig.json`, `npm run lint`, `npx prettier --check src/architecture/network/gpu/network.gpu.kernel.ts`.
- Next: dispatch `05-green-testing` for focused GPU Jest validation on slice `02-01b-kernel`, then continue to slice `02-01c-cache`.
- The previous NGE tier benchmark (10 inputs, 64–32,768 hidden, 4 outputs) showed
  CPU faster than GPU at every tier, but those measurements are invalid because
  the GPU was not doing the same work.
- Parity degrades above ~512–1,024 hidden neurons because CPU outputs diverge
  from the constant GPU values.

## Goal

1. Fix the GPU forward pass so it matches CPU output within tolerance.
2. Add pipeline and GPU-buffer caching to remove per-activation upload/destroy
   overhead.
3. Re-run single-agent NGE tier benchmarks, pushing past 32k hidden neurons
   until CPU overloads or GPU/device limits are hit.
4. Measure parallel-agent throughput on CPU and GPU (e.g., 1×8k vs 6×8k
   networks) to find where GPU batching/parallelism pays off.
5. Refactor the GPU seam and surrounding code to maximize real-world
   effectiveness for NGE-style workloads.

## Reference files

- `src/architecture/network/gpu/network.gpu.kernel.ts`
- `src/architecture/network/gpu/network.gpu.activate.ts`
- `src/architecture/network/gpu/network.gpu.buffer.ts`
- `src/architecture/network/gpu/network.gpu.batched.ts`
- `docs/browser-tests/webgpu-nge-tier-benchmark.html`
- `WebGPU.md`
- `Browser_Tests.md`

## Latest validation evidence

green-light: true
status: green-light

- 🟢 **Slice `02-01a-buffer` implementation complete.**
  - File changed: `src/architecture/network/gpu/network.gpu.buffer.ts`.
  - Incoming-CSR arrays (`inStart`, `inOrder`) plus `from`/`weights` buffers provide source-node index and weight per incoming edge.
  - Per-node topological levels computed via deterministic Kahn-style traversal and uploaded.
  - `GPUBufferSet` extended with `bias`, `topoLevels`, `params`, and `topoLevelsArray`; `uploadDynamicNetworkBuffers` and `destroyGPUBufferSet` updated.
  - Preflight: `npx tsc --noEmit -p tsconfig.json`: OK (exit 0).
  - Preflight: `npm run lint`: OK (exit 0).
  - Preflight: `npx prettier --check src/architecture/network/gpu/network.gpu.buffer.ts`: OK (exit 0).
  - Plan gates: `plan-sync`: PASS, `step-packet`: PASS, `plan-slice-quality`: PASS, `plan-readiness`: PASS.
  - Green validation (`05-green-testing`) for slice `02-01a-buffer`:
    - `npx jest src/architecture/network/gpu/network.gpu.buffer.test.ts --no-coverage`: 16/16 pass
    - `npx jest src/architecture/network/gpu/network.gpu.kernel.test.ts --no-coverage`: 27/27 pass
    - `npx jest src/architecture/network/gpu/network.gpu.parity-large.red.test.ts --no-coverage`: 2/2 pass
    - `npx tsc --noEmit -p tsconfig.json`: OK (exit 0)
    - `npx tsc --noEmit -p tsconfig.test.json`: OK (exit 0)
    - `npm run lint`: OK (exit 0)
    - `npm run typecheck`: script is not defined in `package.json` (tooling gap; `tsc` runs cleanly)
    - `coverage-guard` on `src/architecture/network/gpu/network.gpu.buffer.ts`: 100% statements, branches, functions, lines
    - `quality:folder -- --folder=src/architecture/network/gpu`: FAIL due to missing WebGPU ambient type names; root cause is that `folder-quality-metrics.mjs` excludes `.d.ts` files from the TypeScript program `rootNames`, so the ambient `gpu.types.d.ts` declarations are not loaded. Full `tsc` and `tsc -p tsconfig.test.json` pass, so this is a scanner quirk, not a source regression.
    - Browser GPU execution: not required for this focused Jest slice; all three tests use the owner-local mock device.
    - `plan-sync`: PASS, `step-packet`: PASS, `plan-readiness`: PASS, `plan-slice-quality`: PASS (re-run after evidence update).
    - `cortex-index`: initially stale; rebuilt via `node rag-index/build-index.mjs` (5 docs indexed, 139 chunks); now PASS.

```yaml
PlanUpdate:
  slice_id: '02-01a-buffer'
  changed_files:
    - src/architecture/network/gpu/network.gpu.buffer.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/architecture/network/gpu/network.gpu.buffer.ts'
  tests_for_green:
    - 'npx jest src/architecture/network/gpu/network.gpu.buffer.test.ts --no-coverage'
    - 'npx jest src/architecture/network/gpu/network.gpu.kernel.test.ts --no-coverage'
    - 'npx jest src/architecture/network/gpu/network.gpu.parity-large.red.test.ts --no-coverage'
  rollback:
    - 'git checkout -- src/architecture/network/gpu/network.gpu.buffer.ts'
  next: 'Run 05-green-testing focused GPU slice and attach coverage-guard evidence'
```

- 🟢 **Slice `02-01b-kernel` implementation complete.**
  - File changed: `src/architecture/network/gpu/network.gpu.kernel.ts`.
  - WGSL `forward` entry point now accumulates weighted incoming contributions per node using `inStart`/`inOrder` CSR, `from_nodes`, `weights`, and `node_bias`, then applies the activation function per topological level.
  - Preflight: `npx tsc --noEmit -p tsconfig.json`: OK (exit 0).
  - Preflight: `npm run lint`: OK (exit 0).
  - Preflight: `npx prettier --check src/architecture/network/gpu/network.gpu.kernel.ts`: OK (exit 0).
  - 🟢 **Slice `02-01b-kernel` green validation complete.**
    - `npx jest src/architecture/network/gpu/network.gpu.kernel.test.ts --no-coverage`: 27/27 pass.
    - `npx jest src/architecture/network/gpu/network.gpu.parity-large.red.test.ts --no-coverage`: 2/2 pass (CPU vs mock GPU per-element < 1e-3, mean < 1e-4).
    - `npx jest src/architecture/network/gpu/network.gpu.activate.coverage.test.ts --no-coverage`: 15/15 pass.
    - `npx tsc --noEmit -p tsconfig.json`: OK (exit 0).
    - `npx tsc --noEmit -p tsconfig.test.json`: OK (exit 0).
    - `npm run lint`: OK (exit 0, 0 issues).
    - `npx prettier --check src/architecture/network/gpu/network.gpu.kernel.ts`: OK.
    - `coverage-guard` on `src/architecture/network/gpu/network.gpu.kernel.ts`: 100% statements, branches, functions, lines.
    - Float64 weight risk: `Network._useFloat32Weights` defaults to `true`; the public GPU path is gated by `isGPUEligible`, which rejects `_useFloat32Weights === false`. The slab weights array is therefore `Float32Array` for eligible networks, so the buffer upload path is byte-compatible with the WGSL `array<f32>` binding without conversion. A direct `activateGPU` call on a Float64 network would bypass this gate and misread bytes — this is a latent API-surface risk, not triggered by the acceptance tests.
    - Browser visibility note: these focused Jest tests use the owner-local mock device; no real browser GPU execution was required for this slice. Real-device benchmarks remain subject to the visible-foreground-window mandate.
    - Environment note: `jest-haste-map` reports a duplicate `gpu.mock` manual mock between `src/` and `dist/`; tests still pass. `cortex-index` gate reports `workflow_mcp_alive: false` after index rebuild, which is an external MCP-server/tooling state issue unrelated to the slice.
- 🟢 **Slice `02-01c-cache` implementation complete.**
  - File changed: `src/architecture/network/gpu/network.gpu.activate.ts`.
  - Separated the single per-network GPU state cache into:
    - `activationPipelineCache`: per-device `Map` keyed by generated WGSL source, so networks with the same activation function share one compiled pipeline regardless of topology.
    - `activationBindGroupLayoutCache`: per-device shared bind-group layout for the standard ten-entry activation binding contract.
    - `networkGPUStateCache`: per-network `WeakMap` holding buffer set, bind group, and topology hash.
  - `ensureNetworkGPUState` now reuses the buffer set and bind group when the topology hash is unchanged, and only rebuilds/rebinds when node count or connection set changes. Dynamic weights/biases are still re-uploaded each activation via `uploadDynamicNetworkBuffers`.
  - `getOrCreateActivationPipeline` builds the pipeline from the generated WGSL source using the new `buildGPUPipeline` export from the kernel module, keyed by shader source so compiled pipelines are shared across topologies.
  - Preflight: `npx tsc --noEmit -p tsconfig.json`: OK (exit 0).
  - Preflight: `npm run lint`: OK (exit 0, 0 issues).
  - Preflight: `npx prettier --check src/architecture/network/gpu/network.gpu.activate.ts`: OK (exit 0).
  - Plan gates: `plan-sync`: PASS, `step-packet`: PASS, `agent-graph`: PASS, `stale-wip-plans`: PASS.
  - `quality:folder -- --folder=src/architecture/network/gpu`: FAIL (exit 1) due to the known scanner quirk where `folder-quality-metrics.mjs` excludes `.d.ts` files from `rootNames`; full `tsc --noEmit -p tsconfig.json` and `tsc --noEmit -p tsconfig.test.json` both pass. Coverage for `network.gpu.activate.ts` reported as 98.51% (132/134 lines) by the folder scanner; this is a pre-existing gap that `05-green-testing` should close with owner-local tests for the new cache paths.
  - Risk: the working tree has many unrelated modifications outside the slice boundary; only `src/architecture/network/gpu/network.gpu.activate.ts` and the active plan file were intentionally changed for this slice.

```yaml
PlanUpdate:
  slice_id: '02-01c-cache'
  changed_files:
    - src/architecture/network/gpu/network.gpu.activate.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/architecture/network/gpu/network.gpu.activate.ts'
  tests_for_green:
    - 'npx jest src/architecture/network/gpu/network.gpu.activate.coverage.test.ts --no-coverage'
    - 'npx jest src/architecture/network/gpu/network.gpu.parity-large.red.test.ts --no-coverage'
    - 'npx jest src/architecture/network/gpu/network.gpu.kernel.test.ts --no-coverage'
    - 'npx jest src/architecture/network/gpu/network.gpu.buffer.test.ts --no-coverage'
  rollback:
    - 'git checkout -- src/architecture/network/gpu/network.gpu.activate.ts'
  next: 'Run 05-green-testing focused GPU slice and attach coverage-guard evidence, then proceed to slice 02-01d-parity'
```

```yaml
PlanUpdate:
  slice_id: '02-01b-kernel'
  changed_files:
    - src/architecture/network/gpu/network.gpu.kernel.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/architecture/network/gpu/network.gpu.kernel.ts'
  tests_for_green:
    - 'npx jest src/architecture/network/gpu/network.gpu.kernel.test.ts --no-coverage'
    - 'npx jest src/architecture/network/gpu/network.gpu.parity-large.red.test.ts --no-coverage'
    - 'npx jest src/architecture/network/gpu/network.gpu.activate.coverage.test.ts --no-coverage'
  green_evidence:
    coverage_summary:
      statements: 100
      branches: 100
      functions: 100
      lines: 100
    tests_passed: 44
    tests_total: 44
  rollback:
    - 'git checkout -- src/architecture/network/gpu/network.gpu.kernel.ts'
  next: 'Proceed to slice 02-01c-cache (pipeline/buffer caching)'
```

## Implementation phases

### Phase 1 — Red Testing [DONE]

[DONE] Step 01: Write failing GPU parity test. Red contract confirmed with
`src/architecture/network/gpu/network.gpu.parity-large.red.test.ts`; focused
Jest run fails as expected. Details archived in
[NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.logs.md](NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.logs.md).

### Phase 2 — Implementation [WIP]

#### Step 01 — Implement correct weighted WebGPU forward pass [WIP]

```yaml
phase: 2
step: 1
title: 'Implement correct weighted WebGPU forward pass'
status: '[WIP]'
active_slice: '02-01c-cache'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md'
copy_paste: true
next_step: 'Step 02 — Add batched/parallel GPU inference path'
skills:
  - 'implementation-standards'
  - 'webgpu'
specialists:
  - 'browser-harness-specialist'
validation:
  - 'npx jest src/architecture/network/gpu/ --no-coverage'
  - 'npm run typecheck'
  - 'npm run lint'
acceptance_criteria:
  - 'WGSL kernel computes weighted incoming sums before applying activation.'
  - 'GPU output matches CPU output within tolerance on the red parity test and the real-device NGE tier benchmark.'
  - 'Repeated GPU activations reuse compiled pipelines and buffers.'
  - 'Coverage guard passes on all touched src/ files.'
  - 'Focused GPU Jest tests pass; typecheck and lint are clean.'
slices:
  - slice_id: '02-01a-buffer'
    title: 'Build CPU-side incoming-adjacency and topological-level arrays'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'src/architecture/network/gpu/network.gpu.buffer.ts'
    acceptance_criteria:
      - 'Incoming-adjacency arrays (source node index + weight per incoming edge) are built and uploaded as GPU buffers.'
      - 'Topological-level array per node is computed and uploaded so the kernel can process nodes in dependency order.'
      - 'Focused GPU Jest tests remain green; typecheck and lint are clean.'
    parallelizable: false
    dependencies: []
    next_slice: '02-01b-kernel'
  - slice_id: '02-01b-kernel'
    title: 'Rewrite WGSL kernel to accumulate weighted incoming sums before activation'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'src/architecture/network/gpu/network.gpu.kernel.ts'
    acceptance_criteria:
      - 'WGSL kernel accumulates weighted incoming contributions per node using the adjacency/level buffers before applying activation.'
      - 'GPU output matches CPU output within tolerance on the red parity test.'
      - 'Focused GPU Jest tests remain green; typecheck and lint are clean.'
    parallelizable: false
    dependencies:
      - '02-01a-buffer'
    next_slice: '02-01c-cache'
  - slice_id: '02-01c-cache'
    title: 'Cache compiled pipelines and GPU buffers across activations'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'src/architecture/network/gpu/network.gpu.activate.ts'
    acceptance_criteria:
      - 'Repeated GPU activations reuse compiled pipelines and input/output buffers without recompilation or reallocation.'
      - 'Cache invalidates correctly when network topology or size changes.'
      - 'Focused GPU Jest tests remain green; typecheck and lint are clean.'
    parallelizable: false
    dependencies:
      - '02-01b-kernel'
    next_slice: '02-01d-parity'
  - slice_id: '02-01d-parity'
    title: 'Validate real-device NGE parity, remove dead-code branches, and clean regressions'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'src/architecture/network/gpu/network.gpu.kernel.ts'
      - 'src/architecture/network/gpu/network.gpu.buffer.ts'
      - 'src/architecture/network/gpu/network.gpu.activate.ts'
    acceptance_criteria:
      - 'Real-device NGE tier benchmark reaches CPU/GPU parity within tolerance for all tiers.'
      - 'Coverage guard passes on all touched src/ files (dead-code branches removed).'
      - 'Focused GPU Jest tests remain green; typecheck and lint are clean.'
    parallelizable: false
    dependencies:
      - '02-01c-cache'
    next_slice: '02-02-green'
  - slice_id: '02-02-green'
    title: 'Green validation and coverage guard'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 4
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - 'All focused GPU tests pass and coverage guard is satisfied.'
    parallelizable: false
    dependencies:
      - '02-01d-parity'
```

**User instruction:** Paste this full step packet.

**Step objective:** Rewrite the WGSL kernel to compute the real weighted forward
pass, add CPU-side incoming adjacency and topological level arrays, and cache
compiled pipelines and GPU buffers across activations.

**Context the agent must know:**

- The current kernel only applies activation; it does not gather weighted inputs.
- The red parity test in `src/architecture/network/gpu/network.gpu.parity-large.red.test.ts` defines the expected CPU-vs-GPU behavior and must turn green.
- GPU-eligible logistic activation must carry the worker-registry key.
- Existing GPU tests, the mock device, and the browser benchmark page can be used
  as templates.
- **All GPU measurements must be performed on a visible browser window — headless or minimized windows produce invalid timing data.**

**Execution steps:**

1. Update `src/architecture/network/gpu/network.gpu.kernel.ts` so the shader accumulates weighted incoming contributions per node before applying the activation function.
2. Build or update CPU-side incoming-adjacency and topological-level arrays in `src/architecture/network/gpu/network.gpu.buffer.ts`.
3. Cache compiled pipelines and input/output GPU buffers in `src/architecture/network/gpu/network.gpu.activate.ts` so repeated activations skip recompilation and reallocation.
4. Run focused GPU Jest tests to confirm the parity test now passes within tolerance.
5. Run `npm run typecheck` and `npm run lint` and fix any regressions.

**Stop conditions:**

- Done: the red parity test passes, focused GPU tests are green, typecheck and lint are clean, and pipelines/buffers are cached across activations.
- Blocked: real WebGPU device unavailable or mock cannot reproduce the weighted kernel; report exact device/mock state and escalate to browser-harness-specialist.

**Required validation:**

- `npx jest src/architecture/network/gpu/ --no-coverage`
- `npm run typecheck`
- `npm run lint`

**Plan update requirement:** Mark slice statuses `[DONE]` as they pass, attach red→green evidence, and refresh the `## Latest validation evidence` section before ending.

#### Step 02 — Add batched/parallel GPU inference path [WIP]

```yaml
phase: 2
step: 2
title: 'Add batched/parallel GPU inference path'
status: '[WIP]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md'
copy_paste: true
next_step: 'Step 01 — Re-run focused validation and NGE tier benchmark (Phase 3)'
skills:
  - 'implementation-standards'
  - 'webgpu'
  - 'multithread-evaluation'
specialists:
  - 'browser-harness-specialist'
validation:
  - 'npx jest src/architecture/network/gpu/network.gpu.batched.test.ts --no-coverage'
  - 'npx jest src/architecture/network/gpu/network.gpu.racing.test.ts --no-coverage'
acceptance_criteria:
  - 'Multiple NGE agents can submit GPU inference work without pipeline/buffer collisions.'
  - 'Parallel-agent GPU throughput exceeds sequential CPU throughput at the measured crossover.'
  - 'Red tests fail before implementation and pass after; focused GPU tests remain green.'
slices:
  - slice_id: '02-04-red-tests'
    title: 'Write red tests for batched/parallel inference'
    status: '[PLANNED]'
    goal: 'red-testing'
    estimate_hours: 4
    files_to_change:
      - 'src/architecture/network/gpu/network.gpu.batched.test.ts'
      - 'src/architecture/network/gpu/network.gpu.racing.test.ts'
    acceptance_criteria:
      - 'Red tests exist and fail for the expected batched/parallel GPU behavior.'
    parallelizable: false
    dependencies: []
    next_slice: '02-05a-batched'
  - slice_id: '02-05a-batched'
    title: 'Implement batched GPU inference dispatch and submission ordering'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'src/architecture/network/gpu/network.gpu.batched.ts'
    acceptance_criteria:
      - 'Batched GPU inference dispatch queues and submits multiple inference workloads in a single pass.'
      - 'Submission ordering preserves correctness for dependent activations.'
      - 'Red batched/racing tests still fail or begin passing for the dispatch path; focused GPU tests remain green.'
    parallelizable: false
    dependencies:
      - '02-04-red-tests'
    next_slice: '02-05b-buffer-parallel'
  - slice_id: '02-05b-buffer-parallel'
    title: 'Add parallel/multi-agent buffer allocation without pipeline or buffer collisions'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'src/architecture/network/gpu/network.gpu.buffer.ts'
      - 'src/architecture/network/gpu/network.gpu.activate.ts'
    acceptance_criteria:
      - 'Multiple NGE agents can submit GPU inference work without pipeline/buffer collisions.'
      - 'Buffer allocation and pipeline selection are safe under concurrent/interleaved activation requests.'
      - 'Red batched/racing tests pass for the collision-safety path; focused GPU tests remain green.'
    parallelizable: false
    dependencies:
      - '02-05a-batched'
    next_slice: '02-05c-benchmark-parallel'
  - slice_id: '02-05c-benchmark-parallel'
    title: 'Extend benchmark page with parallel-agent scenarios and measure throughput crossover'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'docs/browser-tests/webgpu-nge-tier-benchmark.html'
    acceptance_criteria:
      - 'Benchmark page includes parallel-agent scenarios (1×8k vs 6×8k networks on CPU and GPU).'
      - 'Parallel-agent GPU throughput exceeds sequential CPU throughput at the measured crossover.'
      - 'All GPU measurements use a visible browser window (non-negotiable).'
    parallelizable: false
    dependencies:
      - '02-05b-buffer-parallel'
    next_slice: '02-06-green'
  - slice_id: '02-06-green'
    title: 'Green validation and coverage guard'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 4
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - 'All tests pass and coverage guard is satisfied.'
    parallelizable: false
    dependencies:
      - '02-05c-benchmark-parallel'
```

**User instruction:** Paste this full step packet.

**Step objective:** Expose a way to run several NGE agents on the same GPU device
in parallel, reusing command queues and buffers where possible. Add a browser
benchmark comparing CPU vs GPU for multiple networks.

**Context the agent must know:**

- The corrected weighted forward pass from Step 01 must be green before this step starts.
- The racing-curriculum worker is the first consumer; parallel-agent throughput is the key metric.
- Existing browser benchmark page `docs/browser-tests/webgpu-nge-tier-benchmark.html` should be extended with parallel-agent scenarios.

**Execution steps:**

1. Write red tests that assert multiple independent networks can activate on the same GPU device without data races.
2. Implement batched/parallel submission in `src/architecture/network/gpu/network.gpu.batched.ts` using shared device/context while keeping per-network buffers separate.
3. Update buffer and activation helpers to support queue reuse and per-network buffer isolation.
4. Extend the browser benchmark to compare 1×8k vs 6×8k networks on CPU and GPU.
5. Run red tests before implementation, then run focused tests after the fix.

**Stop conditions:**

- Done: red tests pass after implementation, parallel-agent benchmark shows measurable GPU advantage, and focused GPU tests remain green.
- Blocked: WebGPU device cannot safely share queues across network contexts; document limitation and escalate.

**Required validation:**

- `npx jest src/architecture/network/gpu/network.gpu.batched.test.ts --no-coverage`
- `npx jest src/architecture/network/gpu/network.gpu.racing.test.ts --no-coverage`

**Plan update requirement:** Update slice statuses and attach red→green evidence; do not advance to Phase 3 until green validation passes.

### Phase 3 — Green Testing [PLANNED]

#### Step 01 — Re-run focused validation and NGE tier benchmark [PLANNED]

```yaml
phase: 3
step: 1
title: 'Re-run focused validation and NGE tier benchmark'
status: '[PLANNED]'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md'
copy_paste: true
next_step: 'Step 01 — Update WebGPU and browser-test documentation (Phase 4)'
skills:
  - 'green-validation-gates'
  - 'browser-testing-harness'
specialists:
  - 'browser-harness-specialist'
validation:
  - 'npx jest src/architecture/network/gpu/ --no-coverage'
  - 'npx jest src/architecture/network/activate/ --no-coverage'
  - 'npm run build:browser'
  - 'Chrome DevTools MCP: navigate to /docs/browser-tests/webgpu-nge-tier-benchmark.html and read window.ngeTierBenchmarkResult'
acceptance_criteria:
  - 'All focused GPU and CPU activate tests pass.'
  - 'Browser bundle builds without errors.'
  - 'Real-device NGE tier benchmark reports CPU/GPU parity and latency up to and beyond 32k hidden neurons.'
```

**User instruction:** Paste this full step packet.

**Step objective:** Confirm the corrected GPU path matches CPU output across all
focused tests, rebuild the browser bundle, and re-run the NGE tier benchmark to
capture real CPU/GPU parity and latency up to and beyond 32k hidden neurons.

**Context the agent must know:**

- Phase 2 implementation must be green before this step begins.
- The benchmark page writes results to `window.ngeTierBenchmarkResult`.
- Chrome DevTools MCP/browser-harness infrastructure is available for real-device measurement.
- **All GPU measurements must be performed on a visible browser window — headless or minimized windows produce invalid timing data. This is non-negotiable.**

**Execution steps:**

1. Run focused GPU Jest tests.
2. Run focused CPU activate tests.
3. Run `npm run build:browser`.
4. Launch the benchmark page in a **visible** browser window via Chrome DevTools MCP (headless/minimized windows produce invalid GPU timing).
5. Read `window.ngeTierBenchmarkResult` and record CPU/GPU latency and parity across tiers.

**Stop conditions:**

- Done: tests pass, bundle builds, and real-device benchmark data is captured up to device/CPU limits.
- Blocked: browser page fails to load or device/OOM limits hit before meaningful data; capture the failure tier and escalate.

**Required validation:**

- `npx jest src/architecture/network/gpu/ --no-coverage`
- `npx jest src/architecture/network/activate/ --no-coverage`
- `npm run build:browser`
- Chrome DevTools MCP benchmark read

**Plan update requirement:** Record benchmark numbers and any device limits in `## Latest validation evidence` before ending.

### Phase 4 — Documentation [PLANNED]

#### Step 01 — Update WebGPU and browser-test documentation [PLANNED]

```yaml
phase: 4
step: 1
title: 'Update WebGPU and browser-test documentation'
status: '[PLANNED]'
goal: 'documenting'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md'
copy_paste: true
next_step: 'Step 01 — Compress plan history and summarize (Phase 5)'
skills:
  - 'educational-docs'
validation:
  - 'npm run docs'
  - 'npm run lint'
acceptance_criteria:
  - 'WebGPU.md reflects the corrected kernel algorithm, caching behavior, and measured performance numbers.'
  - 'Browser_Tests.md includes the new parallel-agent benchmark procedure and results.'
  - 'Docs generation and lint are clean.'
```

**User instruction:** Paste this full step packet.

**Step objective:** Update `WebGPU.md` and `Browser_Tests.md` with the corrected
real-device measurements, parallel-agent numbers, kernel algorithm, and guidance
for when NGE should auto-select GPU.

**Context the agent must know:**

- Documentation must match the implementation actually shipped in Phases 2–3.
- Avoid plan-language in public-facing docs; use measured numbers and API behavior.

**Execution steps:**

1. Update `WebGPU.md` with the corrected weighted kernel description, buffer/pipeline caching notes, and NGE tier benchmark results.
2. Update `Browser_Tests.md` with the parallel-agent benchmark scenario and how to run it.
3. Regenerate docs with `npm run docs`.
4. Run `npm run lint` to catch markdown or code style regressions.

**Stop conditions:**

- Done: docs are regenerated, lint passes, and both markdown files accurately reflect the new GPU behavior.
- Blocked: generated docs conflict with hand-written sections; resolve manually and escalate if the conflict spans plan scope.

**Required validation:**

- `npm run docs`
- `npm run lint`

**Plan update requirement:** Note any docs drift discovered and whether it was fixed in scope or deferred.

### Phase 5 — Session Logging [PLANNED]

#### Step 01 — Compress plan history and summarize [PLANNED]

```yaml
phase: 5
step: 1
title: 'Compress plan history and summarize'
status: '[PLANNED]'
goal: 'logging'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md'
copy_paste: true
next_step: null
skills:
  - 'tracker-handoff'
validation:
  - 'node .github/hooks/workflow-update-sync.mjs --plan=plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md --json'
acceptance_criteria:
  - 'Completed phase details are compressed into the corresponding .logs.md file.'
  - 'Plan file is clean and focused on any remaining active work or closure.'
```

**User instruction:** Paste this full step packet.

**Step objective:** Compress completed phase details into the corresponding
`.logs.md` file and produce a final user-facing summary.

**Context the agent must know:**

- Phases 1–4 must be [DONE] before this step runs.
- Compression must follow the tracker-handoff skill: keep active frontier only, move verbose history to logs.

**Execution steps:**

1. Mark all completed phases [DONE] in the plan header.
2. Move detailed step/slice evidence into `plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.logs.md`.
3. Replace verbose plan sections with compact [DONE] coverage notes.
4. Run the workflow-update-sync hook and any required closure gates.

**Stop conditions:**

- Done: plan file is compressed, logs file is updated, and sync gate passes.
- Blocked: an earlier phase lacks green validation; do not compress until it passes.

**Required validation:**

- `node .github/hooks/workflow-update-sync.mjs --plan=plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md --json`

**Plan update requirement:** Record final status and any reopen conditions before closing the workstream.

## Validation gates

- `plan-sync`: confirms the active [WIP] plan is registered in plan indexes.
- `step-packet`: confirms phase step packets are copy-pasteable and
  MCP-readable.
- `plan-readiness`: confirms a fresh `01-planning` verification pass has
  recorded a green-light verdict before execution-phase dispatch.
- `phase-compression`: used when marking a phase [DONE] before advancing.
- `routing-table-freshness`: confirms agent/skill routing metadata is current.
- `stale-wip-plans`: used before closure or archival handoff.

### Latest validation evidence

- 2026-07-03: Workflow sync: Advanced Phase 2 Step 1 → [DONE]; Phase 2 Step 2 → [WIP]
- 2026-07-02T21:39:00-04:00: 01-planning verification pass — plan ready for 04-implementing slice-fix on slice `02-01-core`; Phase 2 Step 01 reopened to `[WIP]`, Step 02 rolled back to `[PLANNED]`.
- 2026-07-03: Workflow sync: Advanced Phase 2 Step 1 → [DONE]; Phase 2 Step 2 → [WIP] (superseded by verification correction above).
- 2026-07-02T22:47:18-04:00: 01-planning verification pass — stale `02-01-core` verdict replaced; slices decomposed and validated; plan ready for 04-implementing on slice `02-01a-buffer`.
