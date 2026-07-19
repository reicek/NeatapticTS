# NGE Grow-Stabilize Cycle Improvement — Session Implementation Plan

**Session:** `30c2d818-bb83-4ea1-858d-02ef5f470d6e`  
**Workstream:** NGE grow-stabilize cycle extraction, performance auto-enable, and brain-like stabilization.  
**Repo:** `reicek/NeatapticTS`  
**Primary repo tracker:** `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` (currently marked `[DONE]` at file level, but `plans/Roadmap.md` still lists the Racing Curriculum v2 lane as `[WIP]` with Step 21 in progress; tracker alignment is a pre-execution housekeeping item).  
**Source plan line referenced:** 337, Phase 9.  
**Date:** 2026-07-12  
**Status:** [DONE]

**Workstream status:** [DONE] — All phases (B extraction, C performance defaults, A brain-like stabilization) complete. Plan and logs ready to archive to `plans/completed/`.

---

## Scope

Close the three structural gaps blocking NGE from being a reusable, performant, brain-like runtime neuro-evolution product:

1. **Phase B — Complete the Extraction:** Move the adaptation state machine (commit/rollback, candidate scoring, config defaults, cadence/throttle wiring, morph-delta registry) out of `examples/racing_curriculum/controller/runtime.adaptation.ts` and into `src/neat/nge-juvenile/` behind small, testable, domain-agnostic interfaces.
2. **Phase C — Performance by Default:** Auto-enable WebGPU and worker acceleration when available and beneficial, surface availability/fallback contracts, add buffer recycling, and emit performance telemetry.
3. **Phase A — Brain-Like Stabilization + Baby Phase:** Replace random-only stabilization with activity/bias-aware plasticity, add parallel weight-variant evaluation, introduce explicit baby/juvenile/adult lifecycle stages, and thin the racing demo to a composer around the new public `adapt()` API.

**Execution order:** Phase B and Phase C run in parallel. Phase A starts only after both B and C are green and compressed.

---

## Non-Negotiable Constraints

1. **All gates and values are DEFAULTS, never hardcoded.** Every numeric policy constant must be a config property on `NgeGrowStabilizeConfig` (or a child options object) with a documented default. Core code must read values from config, never from module-level constants, except where those constants are themselves the declared defaults.
2. **No backward-compatibility wrappers, no dual-path code, no deferred cleanup.** Old demo-local logic is removed in the same step that introduces the core replacement. Dead code is deleted immediately.
3. **Each slice ≤ 4 hours (ideally 2–3 hours).** Oversized slices must be split before execution.
4. **RED → IMPLEMENT → GREEN per slice.** Every behavioral slice starts with failing tests, proceeds to implementation, and ends with focused green validation + coverage guard on touched `src/` files.
5. **Follow-up implementation agent validation.** After each implementation slice, a fresh `04-implementing` instance (or the appropriate specialist) reviews the diff/evidence. Only if it reports “no changes required” may the workstream advance to the next slice.
6. **GPU slices require real visible-window browser validation.** Any slice touching `src/architecture/network/gpu/*` must include a browser-based measurement on a real foreground GPU, not mock-only Jest validation.
7. **Targeted tests only.** Never run the full test suite in a single shell invocation. Use `--testPathPattern` or `--testNamePattern`.
8. **Plan passes 3 independent agent reviews before implementation begins:**
   - **NGE compliance:** confirms the plan honors the NGE vision, DNA/structure-vs-weights invariant, lifecycle semantics, and avoids leaking training-only assumptions into core.
   - **Library-friendliness:** confirms the public `adapt()` API, pluggable interfaces, and config surface are reusable for non-racing demos.
   - **Performance/scaling:** confirms GPU/worker auto-enable, buffer pooling, and variant evaluation will scale to the 2×256k-node NGE target without silent regressions.

---

## Decision Record

```yaml
decision_record:
  id: DR-20260712-01
  context: |
    Three independent research passes (NGE compliance, library-friendliness,
    performance/scaling) all recommended the same high-level order: extract
    structural seams first, add performance defaults in parallel, then layer
    brain-like stabilization on top of a clean core.
  options:
    - id: optA
      desc: Implement A (brain-like stabilization) first.
    - id: optB
      desc: Implement B (extraction) first, then C, then A.
    - id: optC
      desc: Run B and C in parallel, then A.
  chosen: optC
  rationale: |
    Phase B creates the clean seams and public API that Phase C and Phase A
    depend on. Phase C can be built in parallel with B because it mostly adds
    orthogonal acceleration contracts and auto-enable logic on top of the same
    boundary. Phase A mutates the stabilization algorithm itself and must not
    be written against the old demo-local seam.
  owner: 01-planning
  rollback_plan: |
    If Phase B and Phase C cannot be safely parallelized (e.g., shared file
    conflicts in neat.nge-juvenile.types.ts), collapse to optB and run B to
    completion before C.
  created_at: 2026-07-12T14:26:58-04:00
```

---

## Repo Tracker Alignment Note

Before Phase B execution begins, refresh the repo tracker so it reflects the new workstream:

- Reopen `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` from `[DONE]` to `[WIP]` or create a new active plan file (e.g., `plans/NGE_Grow_Stabilize_Improvement.plans.md`) if the racing curriculum is considered closed and this is a new initiative.
- Update `plans/README.md` and `plans/Roadmap.md` so the active lane points to the correct plan file.
- Run `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=<active-plan-path>`.

This plan document is the session-level implementation brief; the repo tracker update is Step 01 of Phase 0.

---

## Phase 0 — Pre-Implementation Review Gate [DONE]

**Phase objective:** Author the complete implementation plan, resolve any repo tracker inconsistency, and obtain green-light approval from three independent specialist reviewers before any code changes begin.

**Coverage note:** Phase 0 completed with green-light from three independent reviews and all readiness gates passed. Detailed history, clarifications, and gate outputs are in `plans/NGE_Grow_Stabilize_Cycle.logs.md`.

- [DONE] Step 01: Session plan authored; repo tracker aligned (`validate-plan-sync` pass).
- [DONE] Step 02: NGE compliance review passed (`nge-core-scout`, zero blocking findings).
- [DONE] Step 03: Library-friendliness review passed (`boundary-mapper`, zero blocking findings).
- [DONE] Step 04: Performance/scaling review passed (`performance-trace-specialist` + `worker-payload-scout`, zero blocking findings).
- [DONE] Step 05: Blocking findings resolved; accepted changes recorded in plan `## Clarifications` (now in logs).
- [DONE] Step 06: Fresh `01-planning` verification recorded green light; `plan-readiness`, `plan-slice-quality`, `plan-sync`, `step-packet`, and `agent-graph` gates passed.
- [DONE] Step 07: Phase 0 history compressed and this log entry created.

---

## Phase 1 — Complete the Extraction (B) [DONE]

**Phase objective:** Move the grow-stabilize adaptation state machine and all hard-coded policy constants into `src/neat/nge-juvenile/` behind clean, config-driven, domain-agnostic interfaces. The racing demo becomes a thin composer.

**Coverage note:** Phase 1 completed with all B1-B7 slices green with 100% coverage on touched src/ files. Detailed step/slice/VALIDATION_EVIDENCE blocks are in `plans/NGE_Grow_Stabilize_Cycle.logs.md`.

- [DONE] Step 01: Plan Phase B extraction slices.
- [DONE] Step 02: Research (skipped; research complete).
- [DONE] Step 03: B1-B7 extraction primitives — all green.
  - B1: Expanded NgeGrowStabilizeConfig with all defaults; constants.ts + grow-stabilize.ts 100% coverage; type-only types.ts handled by gate fix.
  - B2: Extracted score-gated commit/rollback loop into adapt(); 100% coverage; dead ?? removed; ResolvedAdaptConfig introduced.
  - B3: Extracted forward-pass candidate scoring primitives (collectForwardPassOutputs, buildCandidateScoreWindow, resolveSampleIndices); 100% coverage; no racing-specific imports.
  - B4: Wired computeGrowthThrottle into runNgeGrowStabilizeCycle; added pluggable NgeMetricsProvider/NgeCadencePolicy/NgeObservationEncoder/NgeLifecycleRunner to adapt(); 100% coverage.
  - B5: Refactored NgeMorphDelta validation to registry/dispatch table; dynamic kind registration without editing core switch; dead has() method removed.
  - B6: Removed duplicate metrics/budget builders and demo re-exports; demo imports from core; 10 stale tests updated; 81/81 tests pass.
  - B7: Implemented NGE_DNA governance translator (translateDnaToGrowStabilizeConfig); 100% coverage; pure function, no input mutation.
- [DONE] Step 04: Integration pass — adapt() API composes with all primitives.
- [DONE] Step 05: Phase B green validation — 100% coverage on all touched src/ files.
- [DONE] Step 06: Phase B documentation — JSDoc updated for public API.
- [DONE] Step 07: Phase B logging/compression.

---

## Phase 2 — Performance by Default (C) [DONE]

**Phase objective:** Build a performance-optimization / worker-inference-transport overlay (`src/performance/nge/*`) that auto-enables WebGPU and worker acceleration when beneficial, surfaces availability contracts and fallback notifications, recycles GPU buffers for variant evaluation, and emits performance telemetry.

**Coverage note:** Phase 2 completed with all C1-C5 slices green with 100% coverage on touched src/ files. Detailed step/slice/VALIDATION_EVIDENCE blocks are in `plans/NGE_Grow_Stabilize_Cycle.logs.md`.

- [DONE] Step 01: Plan Phase C performance slices.
- [DONE] Step 02: Research (skipped; research complete).
- [DONE] Step 03: C1-C5 performance defaults — all green.
  - C1: Acceleration availability contract (detectNgeAcceleration, onFallback, NgeAccelerationHandle); 100% coverage; slice-fix added 5 branch-coverage tests.
  - C2: GPU auto-enable in acceleration factory (shouldAutoEnableGpu, autoEnableGpu); thresholds config-overridable; dead = {} default removed; 100% coverage.
  - C3: Worker auto-enable (autoEnableWorkers, resolveWorkerConfig); workerCount = min(hardwareConcurrency - 1, 4); 100% coverage.
  - C4: GPUBufferSetPool for variant buffer recycling; single-pass eviction refactor; 12/12 tests pass; 100% coverage.
  - C5: evaluateWeightVariants primitive with resolveVariantCount; 16 variants ≤1k nodes, ramps 1k-4k, 2 >4k; 100% coverage; dead DEFAULT_MUTATION_SCALE removed.
- [DONE] Step 04: Integration pass — acceleration overlay composes cleanly with core adapt().
- [DONE] Step 05: Phase C green validation — 100% coverage on all touched src/ files.
- [DONE] Step 06: Phase C documentation — JSDoc updated for acceleration API.
- [DONE] Step 07: Phase C logging/compression.

---

## Phase 3 — Brain-Like Stabilization + Baby Phase (A) [DONE]

**Phase objective:** Replace random-only stabilization with activity/bias-aware plasticity, add parallel weight-variant evaluation, introduce explicit baby/juvenile/adult lifecycle stages, wire memory-tier signals into juvenile focus scoring, and thin the racing demo to the new public dapt() API.

**Coverage note:** Phase 3 completed with A1-A5 slices green with 100% coverage on touched src/ files. A4 memory-tier wiring was deferred per P3S02 research verdict. Detailed step/slice/VALIDATION_EVIDENCE blocks are in plans/NGE_Grow_Stabilize_Cycle.logs.md.

- [DONE] Step 01: Plan Phase A stabilization and baby-phase slices.
- [DONE] Step 02: Research memory-tier wiring feasibility — verdict: NOT FEASIBLE TODAY; A4 slices SKIPPED.
- [DONE] Step 03: A1-A5 slices — all green.
  - A1: Activity/bias-aware plasticity; replaced applyWeightMutations with applyPlasticity; 13/13 tests pass; 100% coverage.
  - A2: evaluateWeightVariants primitive with resolveVariantCount; 32/32 tests pass; 100% coverage.
  - A3: Baby/juvenile/adult lifecycle stages with config-driven thresholds; 22/22 tests pass; 100% coverage.
  - A4: Memory-tier focus wiring — SKIPPED (deferred to future runtime memory-tier telemetry workstream).
  - A5: Racing demo refactored to thin composer around adapt(); 94/94 tests pass; 100% coverage on touched src/ files.
- [DONE] Step 04: Phase A integration — plasticity, variants, lifecycle, and demo composer wired cleanly.
- [DONE] Step 05: Phase A green validation — 219 tests passed across 5 targeted suites, zero failures; 100% coverage on touched src/ files.
- [DONE] Step 06: Phase A documentation — JSDoc updated; docs build and lint pass.
- [DONE] Step 07: Phase A logging/compression.

---

## Latest validation evidence

```yaml
green-light: true
verified_at: 2026-07-12T18:04:46-04:00
verifier: 01-planning (verification mode)
notes:
  - 'Phase 0 was already verified by 3 independent reviews (NGE compliance, library-friendliness, performance/scaling) and all readiness gates passed before compression.'
  - 'Phase 1 (B extraction) is DONE and compressed — all B1-B7 slices green with 100% coverage. Detailed history in plans/NGE_Grow_Stabilize_Cycle.logs.md.'
  - 'Phase 2 (C performance defaults) is DONE and compressed — all C1-C5 slices green with 100% coverage. Detailed history in plans/NGE_Grow_Stabilize_Cycle.logs.md.'
  - 'Phase 3 (A) slices are unchanged from original approval — all 15 slices across A1-A5 are <= 4 hours (max 4h, typically 2-3h).'
  - 'Step 01 (Plan Phase A) is DONE — all step packets 02-07 authored with full slice lists.'
  - 'Original green-light and gate outputs were moved to logs during Phase 0-2 compression; this section restores the readiness marker for the plan-readiness gate.'
```

---

## Evidence Commands (to be run by user and attached to VALIDATION_EVIDENCE)

```bash
# Plan sync after tracker update
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Grow_Stabilize_Cycle.plans.md

# Plan readiness and slice quality
node scripts/agent-customization/gates/plan-readiness.gate.mjs --json --plan=plans/NGE_Grow_Stabilize_Cycle.plans.md
node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/NGE_Grow_Stabilize_Cycle.plans.md

# Tier-1 gates
neataptic-gate-mcp:run_gate_check --gate=plan-sync --json
neataptic-gate-mcp:run_gate_check --gate=step-packet --json
neataptic-gate-mcp:run_gate_check --gate=agent-graph --json

# Phase compression when each phase completes
node scripts/agent-customization/gates/phase-compression.gate.mjs --json --plan=plans/NGE_Grow_Stabilize_Cycle.plans.md

# Closure gates
node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json --plan=plans/NGE_Grow_Stabilize_Cycle.plans.md
node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json
```

---

## Appendix: Target API shapes

### A. NgeGrowStabilizeConfig

```typescript
export interface NgeGrowStabilizeConfig {
  // Structural caps (already exposed)
  readonly maxStructuralEditsPerStep: number;
  readonly maxNodes: number;
  readonly maxConnections: number;
  readonly maxEpisodicSlots: number;
  readonly moduleId: string;

  // Plateau detection
  readonly plateauWindowSize: number; // default 5
  readonly plateauVarianceThreshold: number; // default 0.1
  readonly minStabilizationTicks: number; // default 5
  readonly maxStabilizationTicks: number; // default 25

  // Stabilization weight/bias mutation
  readonly weightMutationRate: number; // default 0.3
  readonly weightMutationMagnitude: number; // default 0.1
  readonly biasMutationRate: number; // default 0
  readonly biasMutationMagnitude: number; // default 0

  // Growth throttle
  readonly largeNetworkNodeThreshold: number; // default 1000
  readonly growthThrottleBaseIntervalTicks: number; // default 3

  // Candidate evaluation
  readonly maxForwardPassSamples: number; // default 5
  readonly improvementThreshold: number; // default 0.02

  // Lifecycle hysteresis / cooldown
  readonly mutationCooldownTicks: number; // default 5
  readonly rollbackCooldownTicks: number; // default 5
  readonly lifecycleCooldownWindowCount: number; // default 5

  // Lifecycle stage bands (new)
  readonly babyNodeThreshold: number; // default 1000
  readonly juvenileNodeThreshold: number; // default 4000
  readonly babyVariantCount: number; // default 16
  readonly juvenileVariantCount: number; // default 8
  readonly adultVariantCount: number; // default 2

  // Acceleration defaults (new)
  readonly disableGPU: boolean; // default false
  readonly disableWorkers: boolean; // default false
}
```

### B. NgeAccelerationOptions / NgeAccelerationHandle

```typescript
export interface NgeAccelerationOptions {
  readonly useGPU?: boolean; // default false in Node/test; auto in browser
  readonly useWorkers?: boolean; // default false in Node/test; auto in browser
  readonly gpuDevice?: GPUDevice; // injected by caller/browser-runtime layer
  readonly workerPool?: ParallelInferencePool;
  readonly onFallback?: (requested: 'gpu' | 'workers', reason: string) => void;
}

export interface NgeAccelerationHandle {
  readonly gpuDevice: GPUDevice | null;
  readonly workerPool: ParallelInferencePool | null;
  readonly gpuReason: string; // why GPU is available / unavailable
  readonly workerReason: string; // why workers are available / unavailable
  notifyFallback(requested: 'gpu' | 'workers', reason: string): void;
}
```

### C. NgeAdaptOptions / NgeAdaptResult

```typescript
export interface NgeAdaptOptions {
  readonly network: Network;
  readonly scoreHistory: readonly number[];
  readonly qualityScoreHistory?: readonly number[];
  readonly hasGrownBefore?: boolean;
  readonly stabilizationTicksSinceGrowth?: number;
  readonly hysteresis?: NgeHysteresisState;
  readonly random?: () => number;
  readonly config: NgeGrowStabilizeConfig;
  readonly evaluator: NgeCandidateEvaluator;
  readonly metrics: NgeMetricsProvider;
  readonly cadence?: NgeCadencePolicy;
  readonly observation?: NgeObservationEncoder;
  readonly lifecycle?: NgeLifecycleRunner;
  readonly acceleration?: NgeAccelerationHandle;
  readonly previousResult?: NgeAdaptResult;
}

export interface NgeAdaptResult {
  readonly network: Network;
  readonly committed: boolean;
  readonly stage: 'embryo' | 'baby' | 'juvenile' | 'adult' | 'equilibrium';
  readonly tick: number;
  readonly gpuUsed: boolean;
  readonly workersUsed: boolean;
  readonly accelerationReason: { gpu?: string; workers?: string };
  readonly timingMs: number;
  readonly nodeCount: number;
  readonly connectionCount: number;
}
```

### D. Pluggable interfaces

```typescript
export interface NgeCandidateEvaluator {
  evaluate(
    network: Network,
    samples: readonly number[][],
  ): Promise<NgeCandidateScore>;
}

export interface NgeMetricsProvider {
  snapshot(network: Network): NgeMetricsSnapshot;
}

export interface NgeCadencePolicy {
  shouldGrow(tick: NgeAdaptationTick): boolean;
  shouldStabilize(tick: NgeAdaptationTick): boolean;
}

export interface NgeAdaptationTick {
  readonly tick: number;
  readonly nodeCount: number;
  readonly connectionCount: number;
  readonly stage: NgeAdaptResult['stage'];
  readonly recentScores: readonly number[];
  readonly qualityScores: readonly number[];
}

export interface NgeObservationEncoder {
  encode(observation: unknown): number[];
}

export interface NgeLifecycleRunner {
  enterStage(stage: NgeAdaptResult['stage'], network: Network): void;
}

export interface NgeCandidateScore {
  readonly score: number;
  readonly quality: number;
}
```

### E. NgeDnaGovernance (DNA translator)

```typescript
export interface NgeDnaGovernance {
  readonly stageSchedule?: Partial<Record<NgeAdaptResult['stage'], number>>;
  readonly growthBudget?: { maxNodes: number; maxConnections: number };
  readonly wiringCostPreference?: number;
  readonly morphPolicyKnobs?: {
    readonly addNodeBias: number;
    readonly addConnectionBias: number;
    readonly pruneBias: number;
  };
}

export function translateDnaToGrowStabilizeConfig(
  dna: NgeDna | null,
  defaults: NgeGrowStabilizeConfig,
): NgeGrowStabilizeConfig;
```

Every numeric field in `NgeGrowStabilizeConfig` must have a declared default in `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts` and be overridable via `Partial<NgeGrowStabilizeConfig>` at call sites. The acceleration overlay lives in `src/performance/nge/` and is injected; core `nge-juvenile` must not import it directly.
