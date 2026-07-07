# NEAT Genesis EvoDevo: WebGPU Real Performance

**Status:** [DONE]

## Scope

Validate and fix the WebGPU compute path in NeatapticTS so that GPU inference is numerically correct and performant for NGE-scale networks. Capture real-device CPU-vs-GPU measurements from a visible browser window, push single-agent scale past 32k hidden neurons, and measure parallel-agent throughput.

This plan is downstream of the completed GPU implementation in `plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`.

## Constraints

- **Real visible-window GPU validation is a mandatory green gate for any slice touching `src/architecture/network/gpu/*`.** Mock-only Jest validation is insufficient; headless or minimized windows produce invalid GPU timing and parity data.

## Implementation phases

### Phase 1 — Red Testing [DONE]

[DONE] Step 01: Write failing GPU parity test. Red contract confirmed with `src/architecture/network/gpu/network.gpu.parity-large.red.test.ts`. Details archived in [NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.logs.md](NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.logs.md).

### Phase 2 — Implementation [DONE]

[DONE] Step 01 — Implement correct weighted WebGPU forward pass: struct-packed 4-buffer layout (connections struct, nodes struct, output buffer, constants/uniform buffer), weighted kernel that accumulates incoming contributions before activation, no `requiredLimits` request for `maxStorageBuffersPerShaderStage`, compiled pipeline/buffer caching with correct invalidation. 100% coverage on touched `src/` files, real GPU parity confirmed. Details archived in logs.

[DONE] Step 02 — Add batched/parallel GPU inference path: fused-iteration `batchActivate` (N passes in one encoder, single `mapAsync`), `skipUpload` option for repeated evaluations, parallel-agent cohort dispatch with no pipeline/buffer collisions, real visible-window 6×8k crossover speedup 15.28×. 100% statements/branches/functions/lines coverage on `network.gpu.activate.ts` and `network.gpu.batched.ts`. Details archived in logs.

### Phase 3 — Green Testing [DONE]

#### Step 01 — Re-run focused validation and NGE tier benchmark [DONE]

- Focused GPU Jest tests pass (`npx jest src/architecture/network/gpu/ --no-coverage`).
- Focused CPU activate tests pass (`npx jest src/architecture/network/activate/ --no-coverage`).
- Browser bundle builds cleanly (`npm run build:browser`).
- Real visible-window NGE tier benchmark completed on NVIDIA Lovelace:
  - **6×8k parallel crossover:** CPU ≈ 906 act/s, GPU ≈ 13,846 act/s, speedup ratio ≈ 15.28×.
  - **Scaling ladder all passing:** 12×8k, 6×16k, 6×32k, 20×32k, 50×32k, 2×64k, 2×128k, 2×254k.
  - `2×254k` treated as the current implementation limit; benchmark succeeded without crash or OOM.
- Documentation artifacts produced: `docs/webgpu-performance-guide.md` and `docs/research/flappy-webgpu-feasibility.md`.

### Phase 4 — Documentation [DONE]

[DONE] Step 01 — Update WebGPU and browser-test documentation: created/updated `docs/webgpu-performance-guide.md` with async readback patterns, dispatch overhead measurements, six optimization strategies, CPU/GPU crossover analysis, and citations; refreshed source JSDoc in five GPU modules; regenerated `src/architecture/network/gpu/README.md`; docs-quality gate passed. Details archived in logs.

### Phase 5 — Session Logging [DONE]

[DONE] Step 01 — Compress plan history and summarize: completed phase details moved to this boundary's `.logs.md` file; plan/log pair archived to `plans/completed/`.

## Latest validation evidence

- `green-light: true` recorded by a fresh `01-planning` verification pass.
- Phase 3 real-device benchmark: 6×8k crossover 15.28×, scaling ladder through 2×254k passing, all on visible-foreground Chrome / NVIDIA Lovelace.
- Tier-1 gates: `plan-sync` pass, `agent-graph` pass, `learning-event` pass.

## Design decision: struct-packed 4-buffer layout

```yaml
decision_record:
  id: 'DR-20260703-01'
  context: 'Visible-browser GPU probe on this machine (NVIDIA Lovelace adapter) returned maxStorageBuffersPerShaderStage = 10. Requesting requiredLimits maxStorageBuffersPerShaderStage: 16 failed with "Required limit (16) is greater than the supported limit (10)". The WebGPU spec default guaranteed minimum is 8. The previous flat-buffer design used 9-10 storage buffers and would fail on strict-default adapters; it also failed to obtain 16 on this device.'
  options:
    - id: optA
      desc: 'Pack per-connection and per-node data into struct buffers, using exactly 4 storage buffers (connections struct, nodes struct, output buffer, constants/uniform buffer) and no requiredLimits request.'
    - id: optB
      desc: 'Retain 9-10 flat storage buffers and request a higher maxStorageBuffersPerShaderStage limit.'
  chosen: optA
  rationale: '4 buffers fits both the WebGPU spec default (8) and the probed GPU limit (10), leaving 6 slots for future expansion. Struct packing improves cache locality because all fields for one connection or node are fetched from a single contiguous cache line. Buffer count is not a GPU throughput dial; parallelism comes from workgroups and compute units.'
  owner: '01-planning'
  rollback_plan: 'Revert to the previous flat-buffer binding contract and reintroduce the requiredLimits request; re-verify on the target GPU.'
```

## Reopen conditions

- New GPU hardware family or adapter limit requires re-validation.
- Scope expands to double-buffered staging ring, single compute pass for all topological levels, or batched single-dispatch WGSL shader optimizations.
- Regression in real visible-window benchmark parity or OOM below the validated scaling ladder.

## Audit log

- Archived: `plans/completed/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.logs.md`
