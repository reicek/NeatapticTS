# Library-Friendliness Analysis: NGE Grow-Stabilize Cycle

**Scope:** Reusability of the NGE grow-stabilize algorithm for future demos and external library users.  
**Date:** 2026-07-12  
**Sources verified:**
- `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.types.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.grow.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.ts`
- `src/neat/nge-experimental.ts`
- `src/neat/neat.nge-lifecycle.ts`
- `examples/racing_curriculum/controller/runtime.adaptation.ts`
- `src/architecture/network/network.ts`
- `src/architecture/network/gpu/network.gpu.capability.ts`
- `src/neataptic.ts`

**Constraint:** Analysis only. No source edits or build/test commands were run.

---

## Executive Summary

NGE is positioned as a reusable runtime neuro-evolution product, but the current grow-stabilize surface is still a **low-level building block** rather than a **library-ready adaptation API**. A second demo cannot be built today without copying large pieces of `examples/racing_curriculum/controller/runtime.adaptation.ts`. The most urgent gaps are:

1. **Score-gated commit/rollback** and **forward-pass candidate evaluation** are trapped in the racing demo.
2. Most tunable knobs are hard-coded constants, not config fields.
3. The public surface is still under an `experimental` namespace and mixes domain-specific racing concerns with core algorithmic concerns.

This document answers the seven library-friendliness questions, proposes concrete TypeScript API shapes, draws the core-vs-demo boundary, and recommends a three-phase extraction roadmap.

---

## Prioritized Findings Table

| # | Finding | Severity | Impact | Status |
|---|---------|----------|--------|--------|
| 1 | Forward-pass candidate scoring (`buildCandidateScoreWindow`, `collectForwardPassOutputs`, `resolveBehavioralComplexity`, `resolveSampleIndices`) lives only in the racing demo; core cannot evaluate post-mutation candidates. | **Blocker** | High | Demo-only |
| 2 | Improvement-gated commit/rollback loop is entirely in the demo. Core `NgeGrowStabilizeResult.committed` only says whether a mutation was applied, not whether it improved the score. | **Blocker** | High | Demo-only |
| 3 | Plateau/throttle/weight-mutation/improvement constants are hard-coded, contradicting the documented “sensible defaults and overridable parameters” promise. | **Blocker** | High | Core |
| 4 | `computeGrowthThrottle` is exported from core but never called by core; the demo wires it manually. | Medium | Medium | Inverted leakage |
| 5 | Duplicate metrics/budget builders: core `buildDefaultMetrics/Budget/PruneBudget` vs. demo `buildModuleMetricsSnapshot/buildGrowthBudget/buildPruneBudget`. | Medium | Medium | Both |
| 6 | `NgeGrowStabilizeInput` bundles 10 fields and inlines the `lifecycleRunner` callback type; no named interface exists. | Medium | Medium | Core API |
| 7 | `NgeMorphDelta.kind` is a closed union; `validateMorphDelta` uses an exhaustive switch, so adding a morph kind requires editing core (OCP violation). | Medium | Medium | Core |
| 8 | App layer re-exports core internal `resolveAdaptiveHysteresis` (runtime.adaptation.ts:200), making the demo a publication surface. | Low | Low | Demo |
| 9 | No core primitive to evaluate N weight-variants of a single topology in one dispatch; users must hand-roll parallel candidate evaluation. | Medium | High | Core gap |
| 10 | GPU path is explicit opt-in (`useGPU: true`) with silent CPU fallback and no notification; worker acceleration is also explicit. Library users cannot discover availability. | Medium | Medium | Core |

---

## Answers to the 7 Library-Friendliness Questions

### 1. Can a library user use NGE grow-stabilize without the racing demo? What is the minimum viable API?

**Today, yes — but only as a low-level mutation primitive.**

The only public entry point is:

```ts
// src/neat/nge-experimental.ts exports the nge namespace as experimental.
import { nge } from 'neataptic';

const result = nge.juvenile.runNgeGrowStabilizeCycle({
  network,
  scoreHistory,
  hasGrownBefore,
  stabilizationTicksSinceGrowth,
  qualityScoreHistory,
  hysteresis,
  random,
  config,
  lifecycleRunner,
});
```

The function mutates `network` in place and returns a `NgeGrowStabilizeResult` (`src/neat/nge-juvenile/neat.nge-juvenile.types.ts:323-340`) that tells you which phase ran and which string operations were applied, but **it does not tell the caller whether the mutation actually improved anything**.

A library user still has to write the missing adaptation loop that the racing demo provides:

- Snapshot the network and the global innovation counter before the call.
- Build a pre-mutation baseline score.
- Call `runNgeGrowStabilizeCycle`.
- Build a post-mutation candidate score (the demo uses forward passes over `buildCandidateScoreWindow`).
- Compare scores against an improvement threshold.
- Roll back on failure.
- Manage cadence, cooldowns, and telemetry.

**Proposed minimum viable public API shape:**

```ts
// Core abstraction: something the caller can evaluate.
export interface NgeCandidateEvaluator {
  /**
   * Compute a scalar score for the current network.
   * Called both before and after the candidate mutation.
   */
  evaluate(network: Network, scoreHistory: readonly number[]): number;
}

// Higher-level adaptation entry point that wraps the low-level cycle.
export interface NgeAdaptOptions {
  network: Network;
  scoreHistory: readonly number[];
  qualityScoreHistory?: readonly number[];
  hasGrownBefore?: boolean;
  stabilizationTicksSinceGrowth?: number;
  hysteresis?: NgeHysteresisState;
  random?: () => number;
  config?: Partial<NgeGrowStabilizeConfig>;
  evaluator: NgeCandidateEvaluator;
  /** Optional cadence policy; defaults to every tick. */
  cadence?: NgeCadencePolicy;
}

export interface NgeAdaptResult extends NgeGrowStabilizeResult {
  /** Score before the candidate mutation. */
  baselineScore: number;
  /** Score after the candidate mutation. */
  candidateScore: number;
  /** Whether the mutation was kept based on the evaluator. */
  accepted: boolean;
}

export function adapt(options: NgeAdaptOptions): NgeAdaptResult;
```

With this shape a second demo can reuse NGE by supplying only its own `NgeCandidateEvaluator` instead of re-implementing the snapshot/rollback/commit loop.

---

### 2. What MUST be extracted from the racing demo into core for NGE to be reusable? Prioritize.

#### Blockers — any future demo will have to duplicate these

1. **Score-gated adaptation loop** (`runtime.adaptation.ts:391-606`).
   - Snapshot/rollback (`restoreNetworkSnapshot` with captured `Connection.nextInnovation`).
   - Pre/post candidate scoring.
   - Improvement threshold comparison (`improvement >= improvementThreshold`, first-growth exemption).
   - Result telemetry with `scoreBefore`, `scoreAfter`, `committed`, `reason`.
2. **Forward-pass candidate scoring primitives** (`runtime.adaptation.ts:716-826`).
   - `collectForwardPassOutputs`, `buildCandidateScoreWindow`, `resolveSampleIndices`, `resolveObservationVector`.
   - Racing-specific observation tiling (`RacingQualitySignal` → input vector) must stay in the demo, but the generic sampling/activation loop belongs in core.
3. **Config constants** (see Question 3). Without overridable defaults, every external user forks the constants.

#### High-impact but not blockers

4. **Growth throttle wiring**. `computeGrowthThrottle` exists in core but is not wired into `runNgeGrowStabilizeCycle`. Either add it to core or remove it from the core export and let the caller supply cadence.
5. **Metrics/budget factories**. Replace `buildDefaultMetrics/Budget/PruneBudget` with injectable interfaces so demos stop duplicating them.
6. **Cadence policy** (`isCadenceReady`, lap/sector boundary tracking). The *domain-agnostic* “every N ticks”/“every tick” policy should be in core; lap/sector modes stay in racing.

#### Nice-to-have

7. **Behavioral complexity bonus** (`resolveBehavioralComplexity`). Useful but not required for a second demo.
8. **Telemetry reasons** (`RuntimeAdaptationTelemetry.reason` union). Keep domain reasons in the demo; core can emit generic reasons.

---

### 3. Is the current config surface adequate? What should be overridable but isn’t?

**No.** `NgeGrowStabilizeConfig` currently exposes only five fields (`src/neat/nge-juvenile/neat.nge-juvenile.types.ts:299-310`):

```ts
export interface NgeGrowStabilizeConfig {
  maxStructuralEditsPerStep: number;
  maxNodes: number;
  maxConnections: number;
  maxEpisodicSlots: number;
  moduleId: string;
}
```

The following constants from `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts` are hard-coded in core:

| Constant | Default | Should be config field |
|----------|---------|------------------------|
| `NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE` | `5` | `plateauWindowSize` |
| `NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD` | `0.1` | `plateauVarianceThreshold` |
| `NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS` | `5` | `minStabilizationTicks` |
| `NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS` | `25` | `maxStabilizationTicks` |
| `NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE` | `0.3` | `weightMutationRate` |
| `NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE` | `0.1` | `weightMutationMagnitude` |
| `NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD` | `1_000` | `largeNetworkNodeThreshold` |
| `NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS` | `3` | `growthThrottleBaseIntervalTicks` |
| `NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES` | `5` | `maxForwardPassSamples` |
| `DEFAULT_IMPROVEMENT_THRESHOLD` (demo) | `0.02` | `improvementThreshold` |
| `mutationCooldownTicks` / `rollbackCooldownTicks` (demo) | `5` | `mutationCooldownTicks`, `rollbackCooldownTicks` |
| `cooldownWindowCount` passed to lifecycle runner | `5` | `lifecycleCooldownWindowCount` |

**Proposed expanded interface:**

```ts
export interface NgeGrowStabilizeConfig {
  // Structural caps (already exposed)
  maxStructuralEditsPerStep: number;
  maxNodes: number;
  maxConnections: number;
  maxEpisodicSlots: number;
  moduleId: string;

  // Plateau detection
  plateauWindowSize: number;
  plateauVarianceThreshold: number;
  minStabilizationTicks: number;
  maxStabilizationTicks: number;

  // Stabilization weight mutation
  weightMutationRate: number;
  weightMutationMagnitude: number;

  // Growth throttle
  largeNetworkNodeThreshold: number;
  growthThrottleBaseIntervalTicks: number;

  // Candidate evaluation
  maxForwardPassSamples: number;
  improvementThreshold: number;

  // Lifecycle hysteresis / cooldown
  mutationCooldownTicks: number;
  rollbackCooldownTicks: number;
  lifecycleCooldownWindowCount: number;
}
```

Defaults should be extracted from the existing constants so racing behavior is preserved out of the box.

---

### 4. Should parallel weight-evaluation be a core NGE feature or an opt-in? How should it be exposed?

**It should be a core, opt-in primitive.** Future NGE consumers will repeatedly need to try several weight perturbations and keep the best one; this is the natural stabilization counterpart to structural growth. The primitive also closes the “evaluate N weight-variants of one topology in a single dispatch” gap noted in the research findings.

**Recommended API shape:**

```ts
export interface WeightVariant {
  /** Description for telemetry; e.g. "mutation-0". */
  label: string;
  /** Mutate a clone of the network and return the candidate. */
  mutate(network: Network, random?: () => number): Network;
}

export interface EvaluateWeightVariantsOptions {
  baseNetwork: Network;
  variants: WeightVariant[];
  scorer: NgeCandidateEvaluator;
  scoreHistory: readonly number[];
  /** Use GPU batch evaluation when eligible. */
  useGPU?: boolean;
  /** Use worker pool batch evaluation when eligible. */
  useWorkers?: boolean;
}

export interface EvaluateWeightVariantsResult {
  bestIndex: number;
  bestScore: number;
  baselineScore: number;
  /** The chosen candidate network (may be the original). */
  selectedNetwork: Network;
  perVariant: Array<{ label: string; score: number }>;
}

export function evaluateWeightVariants(
  options: EvaluateWeightVariantsOptions,
): Promise<EvaluateWeightVariantsResult>;
```

Implementation should reuse existing infrastructure:

- `batchActivate` for GPU batch evaluation when `useGPU` is true and `canUseGPU` passes.
- `ParallelInferencePool` / `evaluateInWorkers` for worker dispatch when `useWorkers` is true.
- CPU fallback runs clones sequentially.

This primitive can then be called from the stabilization phase of a higher-level `adapt()` loop instead of the current single-shot `applyWeightMutations`.

---

### 5. How should GPU/worker acceleration be exposed to NGE consumers? Auto-detect? Explicit? Notification?

**Recommendation: explicit opt-in + availability contract + fallback notification.**

The current `Network.activate(input, { useGPU: true })` silently falls back to CPU when `isGPUEligible` returns false (`src/architecture/network/network.ts:1096-1108`). That is safe but makes debugging and performance tuning impossible for library users.

**Proposed acceleration-availability contract:**

```ts
export interface NgeAccelerationCapabilities {
  gpu: {
    available: boolean;
    /** Why the GPU path is unavailable (e.g. "no device", "unsupported activation", "gated network"). */
    reason?: string;
  };
  workers: {
    available: boolean;
    reason?: string;
  };
}

export interface NgeAccelerationOptions {
  /** Request GPU when eligible. */
  useGPU?: boolean;
  /** Request worker pool when eligible. */
  useWorkers?: boolean;
  /** Called when the requested accelerator cannot be used. */
  onFallback?: (requested: 'gpu' | 'workers', reason: string) => void;
}

export function detectNgeAcceleration(): NgeAccelerationCapabilities;
```

`runNgeGrowStabilizeCycle` (and the higher-level `adapt`) should accept `acceleration?: NgeAccelerationOptions` and return which path was actually used:

```ts
export interface NgeGrowStabilizeResult {
  // ...existing fields...
  acceleration: {
    gpuUsed: boolean;
    workersUsed: boolean;
  };
}
```

This solves the silent-fallback problem without forcing auto-detection on users who need deterministic CPU behavior.

---

### 6. Are there API design issues that would block future demos (not just racing)?

Yes. The current surface forces every demo to rebuild the racing adaptation loop:

| Coupling / issue | Why it blocks future demos | Proposed seam |
|------------------|---------------------------|---------------|
| `runNgeGrowStabilizeCycle` mutates in place and returns only `committed` | Caller must implement snapshot/rollback/ scoring. | Provide `adapt()` with built-in rollback and an `NgeCandidateEvaluator`. |
| `scoreHistory` and `qualityScoreHistory` are separate arrays with no abstraction | Callers must manage two windows with different semantics. | Introduce `NgeSignalWindow` with metadata (score vs. quality, timestamp). |
| `buildDefaultMetrics` hard-codes utilization=mean, novelty=0, stabilityAge=0 | Domains with richer signals cannot feed them in. | Define `NgeMetricsProvider` interface; default provider uses the existing formula. |
| `lifecycleRunner` callback type is inlined in `NgeGrowStabilizeInput` (`types.ts:380-398`) | Callers couple to exact `runNgeLifecycle` shape. | Extract named `NgeLifecycleRunner` interface. |
| Forward-pass observation tiling is racing-specific (`RacingQualitySignal` fields) | Any non-racing demo must rewrite `resolveObservationVector`. | Core provides generic sample/activation loop; demo supplies `ObservationEncoder`. |
| `NgeMorphDelta.kind` closed union | New morph kinds require editing core. | Replace `validateMorphDelta` switch with a registry/dispatch table. |
| No weight-variant primitive | Stabilization is limited to a single perturbation. | Add `evaluateWeightVariants` (Question 4). |

**Clean-seam proposal:**

Core should own the adaptation state machine and delegate domain concerns through small interfaces:

- `NgeCandidateEvaluator` — scoring.
- `NgeMetricsProvider` — module metrics.
- `NgeCadencePolicy` — when to run.
- `NgeObservationEncoder` — how history entries become input vectors.
- `NgeAccelerationProvider` — how to evaluate candidates in parallel.

A future demo only implements the first four; the rest are supplied by core defaults.

---

### 7. What is the clean separation between “NGE core algorithm” and “demo-specific adaptation”?

**Core belongs in `src/neat/nge-juvenile/` and `src/neat/neat.nge-lifecycle.ts`.**  
**Demo belongs in `examples/racing_curriculum/controller/runtime.adaptation.ts`.**

#### Core responsibilities

- Plateau detection (`isPlateauReached`).
- Weight mutation during stabilization (`applyWeightMutations`).
- Growth throttle decision (`computeGrowthThrottle` or config-driven cadence).
- Lifecycle orchestration (`runNgeLifecycle`).
- Structural budget enforcement (`NgeGrowthBudget`, `NgePruneBudget`).
- Focus scoring, dry-run morph planning, and morph application.
- Score-gated commit/rollback loop with generic candidate evaluation.
- Config resolution and sensible defaults.
- Acceleration hooks (GPU/worker evaluation primitives).
- A registry-friendly morph delta validator.

#### Demo responsibilities

- Domain-specific score/quality signal definitions (`RacingQualitySignal`, `toDrivingQuality`).
- Domain-specific observation encoder (tiling racing signals into the network input vector).
- Domain-specific candidate evaluator (driving-quality trend + complexity penalty).
- Domain-specific cadence triggers (lap boundary, sector boundary).
- Telemetry labels and UI reasons that mention racing concepts.
- Browser vs. worker default overrides.

#### Boundary contract (TypeScript)

```ts
// Core side of the boundary
export interface NgeCandidateEvaluator {
  evaluate(network: Network, scoreHistory: readonly number[]): number;
}

export interface NgeMetricsProvider {
  buildMetrics(
    network: Network,
    scoreHistory: readonly number[],
    moduleId: string,
  ): NgeModuleMetricsSnapshot;
}

export interface NgeCadencePolicy {
  /** Return true when the adaptation engine is allowed to run this tick. */
  isReady(tick: NgeAdaptationTick): boolean;
}

export interface NgeObservationEncoder {
  /** Convert one history entry into an input vector of the requested size. */
  encode(entry: unknown, inputSize: number): number[];
}

export interface NgeAccelerationProvider {
  capabilities: NgeAccelerationCapabilities;
  evaluateBatch(networks: Network[], inputs: number[][]): Promise<number[][]>;
}

// Demo side implements these four interfaces and passes them into core's adapt() loop.
```

The racing demo keeps its `createRuntimeAdaptationEngine` but the engine becomes a thin composer around the core `adapt()` function:

```ts
const result = nge.juvenile.adapt({
  network,
  scoreHistory,
  evaluator: racingCandidateEvaluator,
  cadence: racingCadencePolicy,
  metricsProvider: racingMetricsProvider,
  observationEncoder: racingObservationEncoder,
  config: racingConfig,
});
```

---

## Phased Extraction Roadmap

### Phase 1 — Unblock reuse (blockers, ~1 sprint)

1. **Expand `NgeGrowStabilizeConfig`** with all hard-coded constants listed in Question 3.
2. **Extract the score-gated adaptation loop** from `runtime.adaptation.ts:391-606` into a new core function `nge.juvenile.adapt()`.
3. **Extract generic forward-pass candidate scoring** (`collectForwardPassOutputs`, `buildCandidateScoreWindow`, `resolveSampleIndices`) into core, parameterized by `NgeObservationEncoder`.
4. **Introduce `NgeCandidateEvaluator`** as the boundary between core and domain scoring.
5. **Wire `computeGrowthThrottle`** into core through config or a cadence policy; stop exporting it as a standalone demo-only helper.

### Phase 2 — Clean seams (~1 sprint)

1. **Introduce `NgeMetricsProvider`, `NgeCadencePolicy`, `NgeObservationEncoder`, and `NgeLifecycleRunner` interfaces** so the demo stops duplicating core builders.
2. **Replace duplicate metrics/budget builders** in the demo with implementations of the new interfaces.
3. **Refactor `NgeMorphDelta` validation** from an exhaustive switch to a registry/dispatch table so new morph kinds can be added without touching core.
4. **Name the lifecycle runner type** and slim `NgeGrowStabilizeInput`.
5. **Remove the demo re-export of `resolveAdaptiveHysteresis`**; expose it from core only.

### Phase 3 — Performance and stabilization primitives (~1 sprint)

1. **Add `evaluateWeightVariants`** using `batchActivate` / `ParallelInferencePool`.
2. **Add acceleration availability contract** (`detectNgeAcceleration`, `onFallback`, result flags).
3. **Promote NGE out of the experimental namespace** or create a stable `nge` barrel with a semver contract.
4. **Add a minimal non-racing example** (e.g. XOR or cart-pole) that uses only the public `adapt()` API to prove the seam is clean.

---

## Conclusion

The NGE grow-stabilize cycle is algorithmically sound but **not yet library-friendly**. The racing demo currently owns the adaptation loop that every other consumer will need. The highest-leverage changes are:

1. Make all hard-coded constants configurable.
2. Move the score-gated commit/rollback and forward-pass candidate evaluation into core behind small interfaces.
3. Provide a higher-level `adapt()` entry point so future demos only supply domain-specific scoring, cadence, and observation encoding.

With those three changes, NGE becomes a reusable product; without them, every new demo will become a fork of `runtime.adaptation.ts`.
