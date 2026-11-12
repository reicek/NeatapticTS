/*
 * Ambient WebGPU declarations anchor for the NeatRacing lane program.
 * This module is the first example in the lane to import `src/` directly,
 * which pulls `src/architecture/network/network.ts` and its GPU modules
 * into `tsconfig.neatRacing.json`'s program. Those GPU modules reference
 * `GPUDevice`/`GPUBuffer`-family globals that only exist in
 * `src/architecture/network/gpu/gpu.types.d.ts` (the sole `declare global`
 * block in `src/`); without them the lane fails with ~180 TS2304 errors,
 * because the lane config pins `types: ["jest", "node"]` and never loads
 * the ambient file on its own.
 *
 * The import below is type-only and binding-free: TypeScript erases it at
 * emit, so it costs nothing at runtime and cannot be flagged as an unused
 * import. It exists solely to pull the ambient declaration file into the
 * lane program. A `/// <reference path="..." />` anchor is the other
 * conventional tool, but it is banned here by
 * `@typescript-eslint/triple-slash-reference` (`path: 'never'` via the
 * recommended preset), which leaves this eslint-legal type-only import as
 * the minimal equivalent.
 */
import type {} from '../../../src/architecture/network/gpu/gpu.types';
/**
 * NeatRacing runtime adaptation engine.
 *
 * Sequences one NGE grow-stabilize cycle per adaptation tick. It converts
 * composite driving-quality signals into scalar scores, decides whether the
 * controller network should grow new structure or stabilize existing weights,
 * and routes the resulting mutations back to the live network.
 *
 * THE E11 OWNERSHIP (spec §9.7): this module implements the EVAL-side half of
 * the two-worker split — growth/variant evaluation, rolling score windows,
 * weight-exhaustion state, oscillation state, and commit/rollback bookkeeping
 * — while the sim worker owns the per-tick control path (`./nge.controller.ts`).
 * The module is deliberately realm-free and decoupled from both workers so the
 * same adaptation policy can run in the eval worker, in the browser host, or
 * in unit tests: the core `runNgeGrowStabilizeCycle` only sees plain numeric
 * score history and a mutable network, and eval results re-enter the sim only
 * at deterministic adaptation-tick boundaries. Round boundaries call
 * `reset()` (E11 item d).
 *
 * ## D1 seeded randomness (normative)
 *
 * `createRuntimeAdaptationEngine` REQUIRES an explicit `random:` stream — a
 * seeded xorshift32 stream derived from the AC-026 reproducibility tuple
 * (`deriveNeatRacingAdaptationChildSeed(tupleSeed, carIndex)` through
 * `createNeatRacingXorShift32Stream`). The recycled ambient fallback
 * (`Math.random`) is banned sim-side and removed here: constructing the engine
 * without an explicit stream throws. `createNeatRacingPerCarAdaptationEngines`
 * derives every per-car stream from the tuple child-seed table so whole-fleet
 * adaptation decisions replay byte-identically (AC-026 support).
 *
 * ## Capacity limits (AC-029)
 *
 * The recycled capacity contract is preserved verbatim:
 * `NEATRACING_ADAPTATION_MAX_NODES` (8,000) and
 * `NEATRACING_ADAPTATION_MAX_CONNECTIONS` (32,000) hard caps, size-based
 * growth throttling before any lifecycle run, and a rollback + safety-reject
 * telemetry path for candidates that exceed the caps.
 *
 * ## Per-tick budget (AC-029 / AC-043)
 *
 * The sim worker's per-tick adaptation work must stay within
 * `NEATRACING_ADAPTATION_TICK_BUDGET_MS` (2 ms of residual work). The budget
 * lives in the sim-worker wiring boundary (the growth/variant evaluation runs
 * in the eval worker and is not budget-bound by this constant): the sim side
 * calls `resolveNeatRacingAdaptationTickBudget` to enforce it, and this engine
 * STAMPS every telemetry with its own measured `tickCostMs` and
 * `withinTickBudget` through an injectable clock (`options.now`,
 * jest-fake-timer friendly). Budget fields are TELEMETRY ONLY — they never
 * gate adaptation decisions, so committed outcomes stay clock-independent and
 * replay-safe (D1).
 *
 * ## Juvenile-only lifecycle (E10)
 *
 * The injected lifecycle runner throws for any stage other than `'juvenile'`
 * (`NEATRACING_ADAPTATION_MAX_EPISODIC_SLOTS` = 15 caps the juvenile growth
 * budget). The grow-stabilize orchestrator invokes the runner with the
 * juvenile stage, so the guard is a hard E10 boundary, not a runtime path.
 *
 * ## The AC-030 oscillation law (one declaration site)
 *
 * The oscillation retune of spec section 9.5 has ONE owner:
 * `./oscillation-penalty.utils`.  This module IMPORTS the shared predicate and
 * RE-EXPORTS the penalty weight, the commit gate and the commit boost under
 * their historical names, so the public surface of this module is unchanged
 * while the declarations themselves live in exactly one place.
 *
 * The engine evaluates the shared predicate EXACTLY ONCE per adaptation tick,
 * against the PRE-mutation network, and applies the resulting engagement to
 * the baseline score, to the pre-mutation baseline, to the stabilization
 * candidate, to the growth candidate and to the cycle's `scoreFn` result.
 * The penalty is therefore SYMMETRIC: the baseline and the variant are damped
 * by the SAME fraction, so the variant-vs-baseline ordering (and the commit
 * decision) survives the penalty.  The engagement's `commitBoost` raises the
 * growth-commit threshold instead of the legacy metric-driven boost.
 *
 * The legacy 200-node oscillation penalty that used to live in
 * {@link scoreRacingVariant} and {@link evaluateRacingTrendScore} is REMOVED:
 * keeping it alive would double-penalize the live path (legacy law plus the
 * shared level-gated ramped law) and would mix score spaces between the
 * engine's `evaluateScore` path and the cycle's `scoreFn` path — a dual path.
 *
 * ## Score-space invariant
 *
 * The grow-stabilize cycle commits a weight variant only when
 * `bestVariantScore > baselineScore + threshold`. That inequality is only
 * meaningful when both scores share the same units and direction. The default
 * variant scorer returns negative mean-squared-error against a target vector,
 * while the racing baseline is a positive driving-quality score. This engine
 * therefore injects a racing-specific `VariantScorer` that collapses the 2-D
 * controller output `[throttle, steer]` to a scalar and returns a positive
 * quality score in the same space as the baseline.
 *
 * ```mermaid
 * flowchart LR
 *   Tick["Adaptation tick"] --> Cadence{"Cadence gate open?"}
 *   Cadence -->|no| Skip["Skip adaptation"]
 *   Cadence -->|yes| Evidence["Build evidence window"]
 *   Evidence --> Baseline["Compute positive driving-quality baseline"]
 *   Baseline --> Cycle["runNgeGrowStabilizeCycle"]
 *   Cycle --> Committed{"Committed?"}
 *   Committed -->|yes| Apply["Apply mutation / keep weights"]
 *   Committed -->|no| Rollback["Rollback network"]
 *   Apply --> Telemetry["Emit telemetry (budget-stamped)"]
 *   Rollback --> Telemetry
 * ```
 *
 * ## Background reading
 *
 * - NEAT and topology-evolving neuroevolution:
 *   K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through
 *   Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2,
 *   pp. 99-127, 2002.
 *   [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)
 * - Growth/stabilization as an explore–exploit tradeoff:
 *   [Wikipedia — Exploration–exploitation dilemma](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma)
 * - Mean squared error:
 *   [Wikipedia — Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
 */

import type {
  AccelerationConfig,
  VariantScorer,
} from '../../../src/acceleration/acceleration.types';
import Connection from '../../../src/architecture/connection/connection';
import Network from '../../../src/architecture/network';
import { Activation } from '../../../src/methods/methods';
import {
  restoreNetworkSnapshot,
  runNgeLifecycle,
} from '../../../src/neat/neat.nge-lifecycle';
import { advanceGrowthHysteresis } from '../../../src/neat/nge-juvenile/neat.nge-juvenile';
import {
  computeGrowthThrottle,
  resolveAdaptiveHysteresis,
  runNgeGrowStabilizeCycle,
} from '../../../src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize';
import { NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE } from '../../../src/neat/nge-juvenile/neat.nge-juvenile.constants';
import type {
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
} from '../../../src/neat/nge-juvenile/neat.nge-juvenile.types';
import {
  createNeatRacingXorShift32Stream,
  deriveNeatRacingAdaptationChildSeed,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import {
  RACING_OSCILLATION_COMMIT_THRESHOLD,
  RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
  RACING_OSCILLATION_PENALTY_WEIGHT,
  applyOscillationPenalty,
  applyOscillationPenaltySymmetrically,
  createOscillationPredicateState,
  evaluateOscillationEngagement,
  type OscillationEngagement,
  type OscillationPredicateState,
} from './oscillation-penalty.utils';

/**
 * THE AC-030 CONSTANT OWNERSHIP (one declaration site).
 *
 * `./oscillation-penalty.utils` is the CANONICAL owner of the oscillation
 * penalty weight, the commit gate and the commit boost (spec section 9.5).
 * This module no longer declares its own copies: it imports the three
 * constants from the shared module and RE-EXPORTS them under the same names
 * and values, so this module's public surface is preserved while the
 * declaration itself lives in exactly ONE place (no dual path).  The shared
 * module deliberately does NOT import back from here — an import in that
 * direction would create a cycle now that the runtime consumes the
 * predicate.
 */
export {
  RACING_OSCILLATION_COMMIT_THRESHOLD,
  RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
  RACING_OSCILLATION_PENALTY_WEIGHT,
};

/** Cadence modes supported by the runtime adaptation engine (recycled). */
export type NeatRacingRuntimeAdaptationCadenceMode =
  'every_tick' | 'every_n_ticks' | 'lap_boundary' | 'sector_boundary';

/** Candidate mutation operations supported by the runtime adaptation engine. */
export type NeatRacingRuntimeAdaptationOperation =
  'param_nudge' | 'add_edge' | 'add_node' | 'prune_edge';

/** Per-tick network size snapshot used by adaptation telemetry. */
export interface NeatRacingAdaptationNetworkSizeSnapshot {
  /** Live node count at a specific adaptation checkpoint. */
  readonly nodes: number;
  /** Live connection count at a specific adaptation checkpoint. */
  readonly connections: number;
}

/** Cadence policy configuration for per-adaptation-tick checks. */
export interface NeatRacingAdaptationCadenceOptions {
  /** Trigger mode used to decide whether the current tick should adapt. */
  readonly mode: NeatRacingRuntimeAdaptationCadenceMode;
  /** Tick interval used by `every_n_ticks`. */
  readonly everyNTicks?: number;
  /** Boundary interval used by `lap_boundary` and `sector_boundary`. */
  readonly boundaryInterval?: number;
}

/** Hard bounds and cooldown controls for one adaptation step. */
export interface NeatRacingAdaptationLimits {
  /** Maximum number of structural operations (`add/prune`) in one step. */
  readonly maxStructuralEditsPerStep: number;
  /** Hard cap for node count after mutation. */
  readonly maxNodes: number;
  /** Hard cap for connection count after mutation. */
  readonly maxConnections: number;
  /** Cooldown (ticks) after any committed mutation attempt. */
  readonly mutationCooldownTicks: number;
  /** Cooldown (ticks) after a rollback outcome. */
  readonly rollbackCooldownTicks: number;
}

/**
 * Composite driving-quality signal used by the racing trend evaluator.
 * Recycled structure verbatim: spline-track progress, forward speed, heading
 * alignment with the track, and an off-track penalty describe how well the
 * car is driving on one tick.
 */
export interface NeatRacingQualitySignal {
  /** Spline track progress (0..1). */
  readonly trackProgress: number;
  /** Forward speed vs track direction. */
  readonly forwardSpeed: number;
  /** Heading alignment with track forward. */
  readonly headingAlignment: number;
  /** Penalty when leaving drivable surface. */
  readonly offTrackPenalty: number;
  /** Physics reward/penalty from the step service. */
  readonly physicsReward?: number;
  /** Optional scalar score shortcut used by custom evaluators. */
  readonly score?: number;
}

/** Adaptation telemetry emitted on every adaptation attempt. */
export interface NeatRacingAdaptationTelemetry {
  /** Fixed-timestep tick at which the adaptation attempt was processed. */
  readonly tick: number;
  /** Ordered operations applied to the candidate (or attempted before rollback). */
  readonly operations: readonly NeatRacingRuntimeAdaptationOperation[];
  /** Baseline score from the rolling evidence window. */
  readonly scoreBefore: number;
  /** Candidate score from the same rolling evidence window. */
  readonly scoreAfter: number;
  /** Whether the candidate was committed to the live network. */
  readonly committed: boolean;
  /** Outcome reason for deterministic replay and diagnostics. */
  readonly reason:
    | 'cadence_not_reached'
    | 'insufficient_evidence'
    | 'mutation_cooldown_active'
    | 'rollback_cooldown_active'
    | 'growth_throttled'
    | 'no_candidate_operations'
    | 'safety_checks_failed'
    | 'improvement_below_threshold'
    | 'plateau_not_reached'
    | 'no_weight_mutations'
    | 'weight_mutation_committed'
    | 'weight_mutation_rolled_back'
    | 'committed';
  /** Network size before candidate mutations were applied. */
  readonly networkSizeBefore: NeatRacingAdaptationNetworkSizeSnapshot;
  /** Network size after commit/rollback resolution. */
  readonly networkSizeAfter: NeatRacingAdaptationNetworkSizeSnapshot;
  /** Current adaptation phase — growth (structural) or stabilization (weight tuning). */
  readonly adaptationPhase: 'growth' | 'stabilization';
  /** Ticks elapsed in the stabilization phase since the last structural growth. */
  readonly stabilizationTicksSinceGrowth: number;
  /**
   * Engine-side per-tick cost in milliseconds measured through the engine's
   * clock (AC-043 observability). TELEMETRY ONLY — never gates decisions.
   */
  readonly tickCostMs: number;
  /** Whether {@link tickCostMs} stayed within `NEATRACING_ADAPTATION_TICK_BUDGET_MS`. */
  readonly withinTickBudget: boolean;
  /**
   * THE AC-030 OSCILLATION VERDICT OF THIS ADAPTATION TICK (spec section 9.5).
   *
   * The shared predicate is evaluated EXACTLY ONCE per adaptation tick — by
   * the engine, from the PRE-mutation network — and the resulting engagement
   * is applied to the baseline AND the variant score symmetrically.  The
   * verdict is published here so the host session can report the live
   * engagement of the tick without re-evaluating the predicate (a second
   * evaluation would advance the hysteresis ramp twice).
   */
  readonly oscillationEngagement: OscillationEngagement;
}

/** Engine options used by the NeatRacing runtime adaptation engine. */
export interface NeatRacingAdaptationEngineOptions {
  /** Cadence policy for adaptation checks. */
  readonly cadence?: NeatRacingAdaptationCadenceOptions;
  /** Hard bounds and cooldown settings. */
  readonly limits?: Partial<NeatRacingAdaptationLimits>;
  /** Minimum score improvement required to commit a candidate. */
  readonly improvementThreshold?: number;
  /** Minimum rolling evidence window length required to evaluate. */
  readonly minimumEvidenceWindow?: number;
  /**
   * THE D1 SEEDED RANDOM STREAM (required). An explicit xorshift32 stream
   * derived from the AC-026 tuple child-seed table —
   * `createNeatRacingXorShift32Stream(deriveNeatRacingAdaptationChildSeed(tupleSeed, carIndex))`.
   * The recycled ambient `Math.random` fallback is banned sim-side and
   * removed; constructing the engine without this stream throws.
   */
  readonly random: () => number;
  /**
   * Optional acceleration configuration forwarded to the grow-stabilize cycle's
   * parallel variant evaluator.
   */
  readonly accelerationConfig?: AccelerationConfig;
  /**
   * Optional human-readable car identifier. When omitted the per-car factory
   * defaults this to the car index.
   */
  readonly carId?: string | number;
  /**
   * Evaluates a network candidate against a short rolling score history.
   *
   * @param network - Candidate network.
   * @param scoreHistory - Rolling score window.
   * @returns Scalar score where larger is better.
   */
  readonly evaluateScore?: (
    network: Network,
    scoreHistory: readonly (number | NeatRacingQualitySignal)[],
  ) => number;
  /**
   * THE SESSION LEVEL GATING THE AC-030 OSCILLATION LAW (spec section 9.5 /
   * TP-07).  The level selects the promotion node threshold through the
   * shared predicate's table (level 1 -> 300 nodes, level 2 -> 600 nodes,
   * level 3 -> 600 nodes); an unknown level falls back to the highest band.
   *
   * Defaults to level 1 (the solo rung).  The level NEVER changes the
   * oscillation METRIC — it only selects the node-count gate.
   */
  readonly level?: number;
  /**
   * Per-tick adaptation budget in milliseconds used for the AC-043 telemetry
   * stamp. Defaults to `NEATRACING_ADAPTATION_TICK_BUDGET_MS`. Telemetry only;
   * never gates decisions.
   */
  readonly tickBudgetMs?: number;
  /**
   * Clock used for `tickCostMs` accounting (jest-fake-timer friendly).
   * Defaults to `Date.now`.
   */
  readonly now?: () => number;
}

/** Per-adaptation-tick input contract for adaptation checks. */
export interface NeatRacingAdaptationTickInput {
  /** Current fixed-timestep tick. */
  readonly tick: number;
  /** Live mutable controller network. */
  readonly network: Network;
  /** Rolling score history assembled by the caller. */
  readonly scoreHistory: readonly (number | NeatRacingQualitySignal)[];
  /** Current completed lap count (for `lap_boundary` cadence). */
  readonly completedLaps?: number;
  /** Current completed sector count (for `sector_boundary` cadence). */
  readonly completedSectors?: number;
}

/** Stateful runtime adaptation engine surface used by browser or worker loops. */
export interface NeatRacingAdaptationEngine {
  /**
   * Runs one adaptation decision against the current tick input.
   *
   * @param tickInput - Per-tick runtime inputs.
   * @returns Promise resolving to budget-stamped deterministic telemetry.
   */
  adaptOnTick(
    tickInput: NeatRacingAdaptationTickInput,
  ): Promise<NeatRacingAdaptationTelemetry>;
  /** Resets cadence boundaries and cooldown state (round boundary, E11 d). */
  reset(): void;
}

/** Accounted per-tick adaptation budget state (the AC-043 sim-side fixture). */
export interface NeatRacingAdaptationTickBudgetState {
  /** The budget the accounting ran against (milliseconds). */
  readonly budgetMs: number;
  /** Measured elapsed time between the tick start and the accounting call. */
  readonly elapsedMs: number;
  /** Remaining within-tick budget, clamped at 0 (never negative). */
  readonly residualBudgetMs: number;
  /** Whether the elapsed time stayed within the named budget (inclusive). */
  readonly withinBudget: boolean;
}

// ── Recycled engine constants (NeatRacing-published) ──────────────────────────

/**
 * Hard node capacity preserved from the recycled engine (AC-029). Mirrors the
 * NGE library's `NGE_MAX_NODE_CAPACITY` so the adaptation limits and the
 * growth budget agree.
 */
export const NEATRACING_ADAPTATION_MAX_NODES = 8_000;

/**
 * Hard connection capacity preserved from the recycled engine (AC-029).
 * Mirrors the NGE library's `NGE_MAX_EDGE_CAPACITY` so the adaptation limits
 * and the growth budget agree.
 */
export const NEATRACING_ADAPTATION_MAX_CONNECTIONS = 32_000;

/**
 * The sim worker's per-tick adaptation budget in milliseconds (AC-029, TP-18,
 * AC-043). The sim-side wiring enforces it with
 * {@link resolveNeatRacingAdaptationTickBudget}; the engine's own telemetry
 * stamps carry `tickCostMs`/`withinTickBudget` against the same constant.
 */
export const NEATRACING_ADAPTATION_TICK_BUDGET_MS = 2;

/**
 * Maximum number of episodic growth slots the NGE lifecycle may allocate
 * during adaptation (E10 juvenile budget). A positive value enables the
 * slot-expansion path; zero would silently disable growth.
 */
export const NEATRACING_ADAPTATION_MAX_EPISODIC_SLOTS = 15;

/** Module identifier passed to the NGE lifecycle focus planner. */
export const NEATRACING_ADAPTATION_MODULE_ID = 'neatracing:runtime';

/** Recycled default cadence: every 4 ticks (`every_n_ticks`, 4). */
const NEATRACING_ADAPTATION_DEFAULT_CADENCE: NeatRacingAdaptationCadenceOptions =
  {
    mode: 'every_n_ticks',
    everyNTicks: 4,
  };

/**
 * Recycled default limits verbatim, published for the worker wiring and the
 * capacity-cap tests (the recycled `DEFAULT_LIMITS` was module-private).
 */
export const NEATRACING_ADAPTATION_DEFAULT_LIMITS: NeatRacingAdaptationLimits =
  {
    maxStructuralEditsPerStep: 5,
    maxNodes: NEATRACING_ADAPTATION_MAX_NODES,
    maxConnections: NEATRACING_ADAPTATION_MAX_CONNECTIONS,
    mutationCooldownTicks: 5,
    rollbackCooldownTicks: 5,
  };

/** Recycled default commit threshold verbatim. */
const DEFAULT_IMPROVEMENT_THRESHOLD = 0.02;
/** Recycled default minimum evidence window length verbatim. */
const DEFAULT_MINIMUM_EVIDENCE_WINDOW = 4;

/**
 * The default session level of the runtime adaptation engine.  The level
 * selects the TP-07 promotion node threshold that gates the shared AC-030
 * oscillation law (level 1 -> 300 nodes, level 2 -> 600 nodes, level 3 ->
 * 600 nodes), so an engine that is not told its level behaves as a level-1
 * engine.
 */
const DEFAULT_OSCILLATION_LEVEL = 1;

/**
 * The forward-pass output index of the STEERING channel of the racing
 * controller (index 0 carries the throttle).  The AC-030 steering detector
 * reads this channel, so the index is named instead of inlined.
 */
const RACING_STEERING_OUTPUT_INDEX = 1;

/**
 * Exclusive upper bound for the deterministic lifecycle morph seed derived
 * from one `[0, 1)` draw of the engine's seeded stream. `2 ** 31` keeps the
 * product inside the double-precision exact-integer range and yields a
 * non-negative 31-bit seed accepted by `Network.setSeed`.
 */
const NEATRACING_ADAPTATION_LIFECYCLE_SEED_BOUND = 0x8000_0000;

/** Re-export the adaptive hysteresis resolver through the engine surface. */
export { resolveAdaptiveHysteresis };

// ── Per-tick budget accounting (AC-029 / AC-043) ──────────────────────────────

/**
 * Accounts one adaptation tick against the named `NEATRACING_ADAPTATION_TICK_BUDGET_MS`
 * budget.
 *
 * This is the sim-worker wiring boundary's enforcement fixture (AC-043): the
 * sim worker records a tick start time, runs its per-tick adaptation work, and
 * calls this helper with the completion time to decide whether the residual
 * work stayed within budget. The engine itself stamps the same accounting into
 * every telemetry (`tickCostMs`, `withinTickBudget`) via its injectable clock.
 * Determinism note: budget fields are observability only and NEVER gate
 * adaptation decisions, so committed outcomes are identical across clocks
 * (D1 replay parity).
 *
 * @param tickStartedAtMs - Clock reading captured when the tick work started.
 * @param nowMs - Clock reading captured when the accounting is called.
 * @param budgetMs - The budget to account against; defaults to
 *   `NEATRACING_ADAPTATION_TICK_BUDGET_MS` (2 ms).
 * @returns The accounted budget state (elapsed, residual, within flag).
 *
 * @example
 * ```ts
 * const startedAtMs = Date.now();
 * runSimSideAdaptationWork();
 * const budget = resolveNeatRacingAdaptationTickBudget(startedAtMs, Date.now());
 * if (!budget.withinBudget) { /* residual work exceeded 2 ms *\/ }
 * ```
 */
export function resolveNeatRacingAdaptationTickBudget(
  tickStartedAtMs: number,
  nowMs: number,
  budgetMs: number = NEATRACING_ADAPTATION_TICK_BUDGET_MS,
): NeatRacingAdaptationTickBudgetState {
  const resolvedBudgetMs = Math.max(0, budgetMs);
  const elapsedMs = Math.max(0, nowMs - tickStartedAtMs);
  return {
    budgetMs: resolvedBudgetMs,
    elapsedMs,
    residualBudgetMs: Math.max(0, resolvedBudgetMs - elapsedMs),
    withinBudget: elapsedMs <= resolvedBudgetMs,
  };
}

/**
 * Reduce a multi-dimensional controller output vector to a scalar driving
 * quality proxy by averaging the absolute activation magnitudes.
 *
 * This matches the reduction used by {@link buildCandidateScoreWindow} so that
 * the racing variant scorer and the rolling candidate scores live in the same
 * units.
 *
 * @param outputVector - Raw network output vector (e.g. `[throttle, steering]`).
 * @returns Scalar proxy in the same units as the racing trend score.
 */
function reduceOutputToScalar(outputVector: readonly number[]): number {
  if (outputVector.length === 0) {
    return 0;
  }

  const sum = outputVector.reduce(
    (accumulated, value) => accumulated + Math.abs(value),
    0,
  );
  return sum / outputVector.length;
}

/**
 * Internal racing variant scorer with an explicit neuron-count gate.
 *
 * The network emits a 2-D controller vector (`[throttle, steering]`), but the
 * grow-stabilize evaluator expects the baseline and variant scores to share the
 * same positive driving-quality score space. This scorer mirrors the
 * {@link evaluateRacingTrendScore} baseline computation: it collapses each
 * output row with {@link reduceOutputToScalar}, then combines the mean trend
 * of the resulting scalar window with a behavioral-complexity bonus (gated on a
 * non-negative trend). Higher scores mean better driving quality, so a variant
 * can win the commit decision when it genuinely outperforms the baseline.
 *
 * The `target` argument mirrors the {@link VariantScorer} contract but is
 * intentionally not used here; the score is derived from the candidate
 * network's own forward-pass outputs so that it lives in the same space as
 * the pre-mutation baseline.
 *
 * The scorer is PURE: it carries NO oscillation penalty of its own.  The
 * AC-030 law lives in the shared predicate
 * (`./oscillation-penalty.utils`), whose engagement the runtime engine
 * evaluates ONCE per adaptation tick and applies symmetrically to the
 * baseline and the variant score.  A second penalty inside the scorer would
 * double-penalize the live path and mix two score spaces.
 *
 * @param outputs - Stack of network output vectors, one per input sample.
 * @param _target - Scalar target value for each sample (unused).
 * @returns Positive racing-trend quality score (higher is better).
 */
export function scoreRacingVariant(
  outputs: readonly number[][],
  _target: readonly number[],
): number {
  if (outputs.length === 0) {
    return 0;
  }

  // The `VariantScorer` contract passes a per-sample target that the racing
  // score space never reads; reference it explicitly (house idiom) so the
  // unused parameter stays deliberate.
  void _target;

  // Step 1: collapse the multi-dimensional controller output to a scalar
  // driving-quality proxy, exactly like {@link buildCandidateScoreWindow}.
  const scalarWindow = outputs.map((outputVector) =>
    reduceOutputToScalar(outputVector),
  );

  // Step 2: compute the same positive trend/mean/complexity score used for
  // the pre-mutation baseline so the stabilization commit inequality compares
  // values in the same units.
  const scoreTrend = (scalarWindow.at(-1) ?? 0) - (scalarWindow[0] ?? 0);
  const scoreMean =
    scalarWindow.reduce((accumulated, value) => accumulated + value, 0) /
    scalarWindow.length;
  const behavioralComplexity = resolveBehavioralComplexity(outputs);
  const qualityImprovedOrSame = scoreTrend >= 0;
  const complexityBonus = qualityImprovedOrSame
    ? behavioralComplexity * RACING_COMPLEXITY_WEIGHT
    : 0;

  // Step 3: fold the trend/mean baseline with the complexity bonus.  The
  // AC-030 oscillation penalty is NOT applied here — the runtime engine
  // applies the shared predicate's engagement (evaluated ONCE per
  // adaptation tick) symmetrically to the baseline and the variant score.
  const baseScore = scoreMean + scoreTrend * 0.5;

  return baseScore + complexityBonus;
}

/**
 * Racing-specific variant scorer for the NGE grow-stabilize cycle.
 *
 * This is a {@link VariantScorer}-compatible wrapper around
 * {@link scoreRacingVariant}.  The variant-scorer contract only passes
 * `outputs` and `target`, so the wrapper adds nothing: the AC-030
 * oscillation penalty is applied by the runtime engine (the shared
 * predicate's engagement, evaluated ONCE per adaptation tick), never inside
 * the scorer.
 *
 * @param outputs - Stack of network output vectors, one per input sample.
 * @param target - Scalar target value for each sample (unused).
 * @returns Positive driving-quality score (higher is better).
 */
export const RACING_VARIANT_SCORER: VariantScorer = (outputs, target) =>
  scoreRacingVariant(outputs, target);

/**
 * Creates a reusable per-tick adaptation engine for the NeatRacing racing
 * runtime loops.
 *
 * The engine sequences one NGE grow-stabilize cycle per adaptation tick. It
 * enforces a configurable cadence policy so adaptation attempts do not fire on
 * every tick, applies a rollback cooldown after rejected candidates, and
 * respects a growth throttle that slows structural mutation as the network
 * grows. The growth-phase commit decision trusts the grow-stabilize cycle's
 * own commit flag, while the engine adds safety-limit checks and an
 * unconditional first-growth path so a brand-new network cannot stall.
 *
 * D1 (normative): the engine REQUIRES an explicit seeded `random` stream —
 * usually `createNeatRacingXorShift32Stream(deriveNeatRacingAdaptationChildSeed(tupleSeed, carIndex))`
 * — and throws when it is omitted. The recycled ambient `Math.random` fallback
 * is banned sim-side and removed.
 *
 * AC-043: every telemetry is stamped with the measured `tickCostMs` and
 * `withinTickBudget` against `NEATRACING_ADAPTATION_TICK_BUDGET_MS` (or the
 * `tickBudgetMs` override) through the injectable `now` clock. Budget fields
 * never gate decisions (replay-safe).
 *
 * E10 (juvenile-only): the injected lifecycle runner throws for any stage
 * other than `'juvenile'`.
 *
 * @param options - Cadence, bounds, evaluation policy, the REQUIRED seeded
 *   random stream, and optional budget/clock overrides.
 * @returns Stateful runtime adaptation engine.
 *
 * @example
 * ```ts
 * const engine = createRuntimeAdaptationEngine({
 *   random: createNeatRacingXorShift32Stream(
 *     deriveNeatRacingAdaptationChildSeed(tuple.seed, carIndex),
 *   ),
 * });
 * const telemetry = await engine.adaptOnTick({ tick, network, scoreHistory });
 * if (!telemetry.withinTickBudget) { /* eval-side cost telemetry *\/ }
 * ```
 */
export function createRuntimeAdaptationEngine(
  options: NeatRacingAdaptationEngineOptions,
): NeatRacingAdaptationEngine {
  // Step 0: Enforce the D1 seeded-stream contract (no ambient Math.random).
  const randomSource = options.random;
  if (typeof randomSource !== 'function') {
    throw new RangeError(
      'neatRacing runtime adaptation failed: options.random must be an explicit seeded stream (D1 bans Math.random sim-side)',
    );
  }

  const clock = options.now ?? Date.now;
  const tickBudgetMs = Math.max(
    0,
    Math.floor(options.tickBudgetMs ?? NEATRACING_ADAPTATION_TICK_BUDGET_MS),
  );
  const cadence = resolveCadenceOptions(options.cadence);
  const limits = resolveLimitOptions(options.limits);
  const evaluateScore = options.evaluateScore ?? evaluateRacingTrendScore;
  const oscillationLevel = options.level ?? DEFAULT_OSCILLATION_LEVEL;
  const minimumEvidenceWindow = Math.max(
    1,
    Math.floor(
      options.minimumEvidenceWindow ?? DEFAULT_MINIMUM_EVIDENCE_WINDOW,
    ),
  );
  const improvementThreshold =
    options.improvementThreshold ?? DEFAULT_IMPROVEMENT_THRESHOLD;

  let hysteresis: NgeHysteresisState = {
    growthPositiveWindowCount: 0,
    pruneUnderuseWindowCount: 0,
    lastMorphKind: 'none',
    cooldownWindowsRemaining: 0,
  };
  let nextRollbackTick = Number.NEGATIVE_INFINITY;
  let lastLapBoundary = Number.NEGATIVE_INFINITY;
  let lastSectorBoundary = Number.NEGATIVE_INFINITY;
  let qualityScoreHistory: number[] = [];
  let hasGrownBefore = false;
  let stabilizationTicksSinceGrowth = 0;
  let currentPhase: 'growth' | 'stabilization' = 'growth';
  let consecutiveWeightExhaustion = 0;
  let consecutiveStabilizationFailures = 0;
  let postGrowthThresholdActive = false;
  let preGrowthBaseline: number | undefined;
  let previousScore: number | undefined;
  // THE AC-030 PREDICATE STATE (spec section 9.5): the engine owns the
  // hysteresis memory of the shared oscillation predicate and threads it
  // from one adaptation tick to the next.  It is reset with the rest of the
  // engine state on `reset()`.
  let oscillationState: OscillationPredicateState =
    createOscillationPredicateState();
  // The engagement verdict of the MOST RECENT adaptation tick.  It is stamped
  // onto the telemetry by the `adaptOnTick` wrapper, which is the only place
  // that sees the tick's own verdict after the tick body has resolved.
  let currentOscillationEngagement: OscillationEngagement =
    createIdleOscillationEngagement(oscillationLevel);

  /**
   * The recycled `adaptOnTick` body, lifted verbatim as the engine's inner
   * run so the AC-043 budget stamping can wrap every returned telemetry.
   *
   * @param tickInput - Per-tick runtime inputs.
   * @returns The recycled deterministic-friendly telemetry (no budget stamp).
   */
  const runAdaptationTick = async (
    tickInput: NeatRacingAdaptationTickInput,
  ): Promise<
    Omit<
      NeatRacingAdaptationTelemetry,
      'tickCostMs' | 'withinTickBudget' | 'oscillationEngagement'
    >
  > => {
    const networkSizeBefore = resolveNetworkSizeSnapshot(tickInput.network);
    const evidenceWindow = resolveEvidenceWindow(tickInput.scoreHistory);

    // Step 0 (AC-030, spec section 9.5): evaluate the SHARED oscillation
    // predicate EXACTLY ONCE per adaptation tick, against the PRE-mutation
    // network.  The engagement it produces is applied SYMMETRICALLY to the
    // baseline and to every variant score of this tick, and its commit boost
    // raises the growth-commit threshold.
    const oscillationEngagement = evaluateTickOscillationEngagement(
      tickInput.network,
      evidenceWindow,
      oscillationState,
      oscillationLevel,
    );
    oscillationState = oscillationEngagement.state;
    currentOscillationEngagement = oscillationEngagement;

    const rawBaselineScore = evaluateScore(tickInput.network, evidenceWindow);
    const baselineScore = applyOscillationPenalty(
      rawBaselineScore,
      oscillationEngagement,
    );
    const inputSize = tickInput.network.input;
    const trainingInputs =
      inputSize > 0
        ? evidenceWindow.map((entry) =>
            resolveObservationVector(entry, inputSize),
          )
        : [];
    const trainingTarget = evidenceWindow.map(toDrivingQuality);

    // Step 1: Verify cadence gating before proposing any mutations.
    if (
      !isCadenceReady(
        cadence,
        tickInput,
        lastLapBoundary,
        lastSectorBoundary,
        (nextLapBoundary, nextSectorBoundary) => {
          lastLapBoundary = nextLapBoundary;
          lastSectorBoundary = nextSectorBoundary;
        },
      )
    ) {
      return createTelemetry(
        {
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'cadence_not_reached',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        },
        currentPhase,
        stabilizationTicksSinceGrowth,
      );
    }

    // Step 2: Guard against insufficient evidence and active cooldown windows.
    if (evidenceWindow.length < minimumEvidenceWindow) {
      return createTelemetry(
        {
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'insufficient_evidence',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        },
        currentPhase,
        stabilizationTicksSinceGrowth,
      );
    }

    // Step 2.5: Record the baseline quality score for plateau detection.
    //           The score is tracked at every adaptation tick with sufficient
    //           evidence, regardless of cooldown or throttle state, so the
    //           rolling window reflects the network's actual performance.
    qualityScoreHistory.push(baselineScore);
    if (qualityScoreHistory.length > NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE) {
      qualityScoreHistory.shift();
    }

    // Step 3: Advance lifecycle hysteresis for the current evaluation window.
    const isPositiveFocusWindow =
      evidenceWindow.length >= 2 &&
      toDrivingQuality(evidenceWindow.at(-1) ?? 0) >
        toDrivingQuality(evidenceWindow[0] ?? 0);
    hysteresis = advanceGrowthHysteresis(hysteresis, isPositiveFocusWindow);

    if (hysteresis.cooldownWindowsRemaining > 0) {
      return createTelemetry(
        {
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'mutation_cooldown_active',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        },
        currentPhase,
        stabilizationTicksSinceGrowth,
      );
    }

    if (tickInput.tick < nextRollbackTick) {
      return createTelemetry(
        {
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'rollback_cooldown_active',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        },
        currentPhase,
        stabilizationTicksSinceGrowth,
      );
    }

    // Step 3.5: Size-based growth throttle — back off lifecycle runs when
    // the network has grown large to preserve real-time performance.
    const growthThrottle = computeGrowthThrottle(
      tickInput.network,
      tickInput.tick,
    );
    if (growthThrottle.shouldThrottle) {
      return createTelemetry(
        {
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'growth_throttled',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        },
        currentPhase,
        stabilizationTicksSinceGrowth,
      );
    }

    // Step 3.6: Run the NGE grow-stabilize cycle (plateau detection +
    //           weight mutation stabilization / structural growth via the
    //           NGE lifecycle). The cycle internally decides whether to
    //           stabilize (weight perturbations) or grow (structural morphs)
    //           based on the plateau detection logic.
    // plateau_not_reached: when the cycle returns stabilization phase, the
    // network is still learning to use its current structure.
    const preMutationScoreWindow = buildCandidateScoreWindow(
      tickInput.network,
      evidenceWindow,
    );
    const rawPreMutationBaselineScore = evaluateScore(
      tickInput.network,
      preMutationScoreWindow,
    );
    const preMutationBaselineScore = applyOscillationPenalty(
      rawPreMutationBaselineScore,
      oscillationEngagement,
    );

    // Snapshot the network and the global connection innovation counter
    // so a rollback can restore both pieces of process state.
    const capturedInnovation = Connection.nextInnovation;
    const rollbackSnapshot = tickInput.network.toJSON();
    // AC-026 replay parity: capture every node's pre-growth activation
    // before the grow-stabilize cycle runs. The NGE lifecycle assigns
    // activation functions to growth-added hidden nodes through the
    // node-level MOD_ACTIVATION mutation, which draws from the unseeded
    // global Math.random (src/architecture/node/node.ts
    // applyActivationMutation) and is therefore not reproducible from the
    // lifecycle seed alone.
    const preGrowthSquashByNode = new Map(
      tickInput.network.nodes.map((node) => [node, node.squash] as const),
    );

    const cycleResult = await runNgeGrowStabilizeCycle({
      network: tickInput.network,
      scoreHistory: evidenceWindow.map(toDrivingQuality),
      hasGrownBefore,
      stabilizationTicksSinceGrowth,
      qualityScoreHistory,
      hysteresis,
      random: randomSource,
      accelerationConfig: options.accelerationConfig,
      previousScore,
      consecutiveWeightExhaustion,
      consecutiveStabilizationFailures,
      postGrowthThresholdActive,
      preGrowthBaseline,
      baselineScore: preMutationBaselineScore,
      inputs: trainingInputs,
      target: trainingTarget,
      scoreFn: (outputs, target) =>
        applyOscillationPenalty(
          scoreRacingVariant(outputs, target),
          oscillationEngagement,
        ),
      lifecycleStage: 'baby',
      config: {
        maxStructuralEditsPerStep: limits.maxStructuralEditsPerStep,
        maxNodes: limits.maxNodes,
        maxConnections: limits.maxConnections,
        maxEpisodicSlots: NEATRACING_ADAPTATION_MAX_EPISODIC_SLOTS,
        moduleId: NEATRACING_ADAPTATION_MODULE_ID,
      },
      lifecycleRunner: (lifecycleInput) => {
        const { stage, ...lifecycleRest } = lifecycleInput;
        // E10 (juvenile-only): the recycled engine hard-gates the lifecycle
        // stage; the orchestrator invokes this runner with 'juvenile'.
        if (stage !== 'juvenile') {
          throw new Error(
            `Runtime adaptation only supports the juvenile NGE lifecycle stage, received: ${stage}`,
          );
        }
        const adaptiveHysteresis = resolveAdaptiveHysteresis(
          tickInput.network.nodes.length,
        );
        return runNgeLifecycle({
          ...lifecycleRest,
          stage,
          // D1/AC-026: `runNgeLifecycle` is NON-DETERMINISTIC when `seed` is
          // omitted (its own contract). Derive the morph seed from one draw
          // of the per-car seeded stream so twin engines (and repeat runs)
          // plan and apply bitwise-identical topology.
          seed: Math.floor(
            randomSource() * NEATRACING_ADAPTATION_LIFECYCLE_SEED_BOUND,
          ),
          metrics: buildModuleMetricsSnapshot(
            tickInput.network,
            evidenceWindow,
          ),
          budget: buildGrowthBudget(tickInput.network, limits),
          pruneBudget: buildPruneBudget(tickInput.network),
          config: {
            ...lifecycleRest.config,
            hysteresisWindowCount: adaptiveHysteresis,
            cooldownWindowCount: limits.mutationCooldownTicks,
            maxStructuralEditsPerStep: limits.maxStructuralEditsPerStep,
          },
        });
      },
    });

    // AC-026 replay parity: adaptation treats activation policy as
    // immutable. Restore every surviving node's pre-growth activation and
    // give growth-added nodes the Node constructor default (logistic)
    // BEFORE any post-mutation scoring (candidate window, oscillation
    // detection, commit/rollback decision), so candidate evaluation is
    // bit-reproducible across twin engines and repeat runs despite the
    // src-level unseeded MOD_ACTIVATION draw consumed by growth.
    for (const node of tickInput.network.nodes) {
      const preGrowthSquash = preGrowthSquashByNode.get(node);
      if (preGrowthSquash) {
        node.squash = preGrowthSquash;
      } else {
        node.squash = Activation.logistic;
      }
    }

    // Carry the grow-stabilize cycle state forward so that weight-exhaustion
    // counting, the post-growth anti-runaway boost, and the pre-growth
    // baseline persist across adaptation ticks.
    consecutiveWeightExhaustion = cycleResult.consecutiveWeightExhaustion ?? 0;
    consecutiveStabilizationFailures =
      cycleResult.consecutiveStabilizationFailures ??
      consecutiveStabilizationFailures;
    postGrowthThresholdActive =
      cycleResult.postGrowthThresholdActive ?? postGrowthThresholdActive;
    preGrowthBaseline = cycleResult.preGrowthBaseline;
    previousScore = preMutationBaselineScore;
    stabilizationTicksSinceGrowth = cycleResult.stabilizationTicksSinceGrowth;

    // Stabilization phase: weight mutations applied by the cycle.
    if (cycleResult.phase === 'stabilization') {
      currentPhase = 'stabilization';

      if (!cycleResult.committed) {
        consecutiveStabilizationFailures += 1;
        return createTelemetry(
          {
            tick: tickInput.tick,
            operations: [],
            scoreBefore: baselineScore,
            scoreAfter: baselineScore,
            committed: false,
            reason: 'no_weight_mutations',
            networkSizeBefore,
            networkSizeAfter: networkSizeBefore,
          },
          'stabilization',
          stabilizationTicksSinceGrowth,
        );
      }

      // Evaluate stabilization candidate score.
      const stabilizationScoreWindow = buildCandidateScoreWindow(
        tickInput.network,
        evidenceWindow,
      );
      const rawStabilizationScore = evaluateScore(
        tickInput.network,
        stabilizationScoreWindow,
      );
      // AC-030: the SAME engagement damps the baseline and the stabilization
      // candidate SYMMETRICALLY, so the variant-vs-baseline ordering (and
      // therefore the commit decision) survives the penalty.
      const stabilizationScores = applyOscillationPenaltySymmetrically(
        rawBaselineScore,
        rawStabilizationScore,
        oscillationEngagement,
      );
      const stabilizationScore = stabilizationScores.variantScore;
      const stabilizationImprovement =
        stabilizationScores.variantScore - stabilizationScores.baselineScore;

      if (stabilizationImprovement >= 0) {
        consecutiveStabilizationFailures = 0;
        return createTelemetry(
          {
            tick: tickInput.tick,
            operations: ['param_nudge'],
            scoreBefore: baselineScore,
            scoreAfter: stabilizationScore,
            committed: true,
            reason: 'weight_mutation_committed',
            networkSizeBefore,
            networkSizeAfter: networkSizeBefore,
          },
          'stabilization',
          stabilizationTicksSinceGrowth,
        );
      }

      // Rollback weight mutations that did not improve the score.
      consecutiveStabilizationFailures += 1;
      restoreNetworkSnapshot(
        tickInput.network,
        rollbackSnapshot,
        capturedInnovation,
      );

      return createTelemetry(
        {
          tick: tickInput.tick,
          operations: ['param_nudge'],
          scoreBefore: baselineScore,
          scoreAfter: stabilizationScore,
          committed: false,
          reason: 'weight_mutation_rolled_back',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        },
        'stabilization',
        stabilizationTicksSinceGrowth,
      );
    }

    // Growth phase: structural mutation via NGE lifecycle.
    currentPhase = 'growth';
    const operations =
      cycleResult.operations as NeatRacingRuntimeAdaptationOperation[];

    if (operations.length === 0) {
      // Restore the global innovation counter that
      // syncInnovationCounterToNetwork inside runNgeLifecycle may have
      // lowered. Without this, the no-ops early return leaves
      // Connection.nextInnovation at the network-pinned value instead of
      // the pre-lifecycle captured value, corrupting subsequent mutation
      // innovation assignments.
      Connection.resetInnovationCounter(capturedInnovation);
      return createTelemetry(
        {
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'no_candidate_operations',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        },
        currentPhase,
        stabilizationTicksSinceGrowth,
      );
    }

    // Evaluate candidate score after lifecycle mutation.
    // The candidateScoreWindow is built from fresh forward-pass outputs
    // of the post-mutation network, NOT from the shared scoreHistory. This
    // ensures the candidate score reflects the actual driving-quality
    // impact of the structural mutation, not historical performance.
    const rawCandidateSize = resolveNetworkSizeSnapshot(tickInput.network);
    const safetyChecksPass = passesSafetyChecks(rawCandidateSize, limits);
    const candidateScoreWindow = buildCandidateScoreWindow(
      tickInput.network,
      evidenceWindow,
    );
    const rawCandidateScore = evaluateScore(
      tickInput.network,
      candidateScoreWindow,
    );
    // AC-030: the growth pair is penalized SYMMETRICALLY as well — the
    // pre-mutation baseline and the post-mutation candidate are damped by the
    // SAME fraction of the tick's single engagement.
    const growthScores = applyOscillationPenaltySymmetrically(
      rawPreMutationBaselineScore,
      rawCandidateScore,
      oscillationEngagement,
    );
    const candidateScore = growthScores.variantScore;
    const improvement = candidateScore - growthScores.baselineScore;
    // For the very first growth, commit unconditionally (if safety checks
    // pass) — the structural mutation adds capacity that the stabilization
    // phase will tune; requiring an immediate score improvement would
    // rollback the first growth and leave the network stuck forever.
    const isFirstGrowth = !hasGrownBefore;
    // Boost the improvement threshold while the SHARED AC-030 predicate is
    // engaged, making it harder to commit structural growth during unstable
    // reward windows.  The boost comes from the tick's single engagement and
    // therefore ramps with the hysteresis instead of stepping.
    const effectiveImprovementThreshold =
      improvementThreshold + oscillationEngagement.commitBoost;
    // Trust the grow-stabilize cycle's commit decision. The cycle already
    // applied the structural mutation in-place; re-evaluating with a fixed
    // improvement threshold here would roll back valid growth. We still
    // enforce safety limits and keep the unconditional first-growth path.
    const shouldCommit =
      safetyChecksPass &&
      (isFirstGrowth ||
        cycleResult.committed ||
        improvement >= effectiveImprovementThreshold);

    // Commit or rollback based on score improvement.
    if (shouldCommit) {
      hysteresis = cycleResult.hysteresis ?? hysteresis;
      hasGrownBefore = true;
      qualityScoreHistory = [];
      stabilizationTicksSinceGrowth = 0;
      currentPhase = 'stabilization';
      return createTelemetry(
        {
          tick: tickInput.tick,
          operations,
          scoreBefore: preMutationBaselineScore,
          scoreAfter: candidateScore,
          committed: true,
          reason: 'committed',
          networkSizeBefore,
          networkSizeAfter: rawCandidateSize,
        },
        'growth',
        0,
      );
    }

    // Rollback the network and the global innovation counter, then
    // set the rollback cooldown.
    restoreNetworkSnapshot(
      tickInput.network,
      rollbackSnapshot,
      capturedInnovation,
    );
    nextRollbackTick = tickInput.tick + limits.rollbackCooldownTicks;
    const restoredSize = resolveNetworkSizeSnapshot(tickInput.network);

    return createTelemetry(
      {
        tick: tickInput.tick,
        operations,
        scoreBefore: preMutationBaselineScore,
        scoreAfter: candidateScore,
        committed: false,
        reason: safetyChecksPass
          ? 'improvement_below_threshold'
          : 'safety_checks_failed',
        networkSizeBefore,
        networkSizeAfter: restoredSize,
      },
      currentPhase,
      stabilizationTicksSinceGrowth,
    );
  };

  return {
    async adaptOnTick(
      tickInput: NeatRacingAdaptationTickInput,
    ): Promise<NeatRacingAdaptationTelemetry> {
      // AC-043: measure the per-tick engine work against the named budget
      // and stamp the outcome onto the telemetry. The measurement is
      // observability only — the decision itself is clock-independent.
      const tickStartedAtMs = clock();
      const telemetry = await runAdaptationTick(tickInput);
      const budgetState = resolveNeatRacingAdaptationTickBudget(
        tickStartedAtMs,
        clock(),
        tickBudgetMs,
      );
      return {
        ...telemetry,
        oscillationEngagement: currentOscillationEngagement,
        tickCostMs: budgetState.elapsedMs,
        withinTickBudget: budgetState.withinBudget,
      };
    },
    reset(): void {
      hysteresis = {
        growthPositiveWindowCount: 0,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      };
      nextRollbackTick = Number.NEGATIVE_INFINITY;
      lastLapBoundary = Number.NEGATIVE_INFINITY;
      lastSectorBoundary = Number.NEGATIVE_INFINITY;
      qualityScoreHistory = [];
      hasGrownBefore = false;
      stabilizationTicksSinceGrowth = 0;
      currentPhase = 'growth';
      consecutiveWeightExhaustion = 0;
      consecutiveStabilizationFailures = 0;
      postGrowthThresholdActive = false;
      preGrowthBaseline = undefined;
      previousScore = undefined;
      // AC-030: the predicate hysteresis memory is engine state — a reset
      // starts the shared predicate from a fully disengaged state.
      oscillationState = createOscillationPredicateState();
      currentOscillationEngagement =
        createIdleOscillationEngagement(oscillationLevel);
    },
  };
}

/**
 * Creates one independent runtime adaptation engine per car index, with every
 * per-car random stream DERIVED FROM THE AC-026 REPRODUCIBILITY TUPLE
 * (D1/D3, AC-026 support): car `i`'s engine draws from the xorshift32 stream
 * seeded by the `adaptation-<i>` surface child seed
 * (`deriveNeatRacingAdaptationChildSeed(tuple.seed, i)` through
 * `createNeatRacingXorShift32Stream`).
 *
 * Each car in the fleet maintains its own adaptation state, cooldowns, and
 * cadence boundaries: the factory returns a `Map<number, NeatRacingAdaptationEngine>`
 * keyed by car index (0 to `carCount - 1`) where every engine has fully
 * independent closure-scoped state — no shared mutable state across cars, and
 * no ambient `Math.random` anywhere (D1).
 *
 * @param tuple - The AC-026 reproducibility tuple whose seed drives the
 *   adaptation child-seed table.
 * @param carCount - Number of cars to create engines for (non-negative integer).
 * @param options - Optional engine options applied identically to every car's
 *   engine (the `random` and `carId` fields are derived per car and cannot be
 *   overridden here).
 * @returns Map keyed by car index of independent seeded adaptation engines.
 * @throws {RangeError} When `carCount` is not a non-negative integer.
 *
 * @example
 * ```ts
 * const engines = createNeatRacingPerCarAdaptationEngines(tuple, 3);
 * const car0Engine = engines.get(0); // adaptation-0 derived stream
 * const car1Engine = engines.get(1); // adaptation-1 derived stream
 * ```
 */
export function createNeatRacingPerCarAdaptationEngines(
  tuple: NeatRacingReproducibilityTuple,
  carCount: number,
  options: Omit<NeatRacingAdaptationEngineOptions, 'random' | 'carId'> = {},
): Map<number, NeatRacingAdaptationEngine> {
  if (!Number.isInteger(carCount) || carCount < 0) {
    throw new RangeError(
      'neatRacing runtime adaptation failed: carCount must be a non-negative integer',
    );
  }

  const engines = new Map<number, NeatRacingAdaptationEngine>();
  for (let carIndex = 0; carIndex < carCount; carIndex++) {
    engines.set(
      carIndex,
      createRuntimeAdaptationEngine({
        ...options,
        // D1/D3: the adaptation-<carIndex> surface child seed drives the
        // engine's seeded xorshift32 stream (AC-026 support).
        random: createNeatRacingXorShift32Stream(
          deriveNeatRacingAdaptationChildSeed(tuple.seed, carIndex),
        ),
        carId: carIndex,
      }),
    );
  }

  return engines;
}

/**
 * Lightweight default evaluator for rolling score history windows.
 * Recycled verbatim.
 *
 * @param network - Candidate network.
 * @param scoreHistory - Rolling score window.
 * @returns Combined trend/complexity score.
 */
export function evaluateRollingScoreWindow(
  network: Network,
  scoreHistory: readonly number[],
): number {
  if (scoreHistory.length === 0) {
    return 0;
  }

  const scoreTrend = (scoreHistory.at(-1) ?? 0) - (scoreHistory[0] ?? 0);
  const scoreMean =
    scoreHistory.reduce(
      (accumulatedScore, scoreValue) => accumulatedScore + scoreValue,
      0,
    ) / scoreHistory.length;
  const sizePenalty =
    (network.nodes.length + network.connections.length) * 0.000_1;

  return scoreMean + scoreTrend * 0.5 - sizePenalty;
}

/**
 * Weight applied to network complexity (nodes + connections) in the racing
 * trend evaluator.  A small positive weight ensures the candidate (post-morph)
 * network scores slightly higher than the baseline (pre-morph) network when
 * structural mutations add capacity, allowing growth mutations to pass the
 * improvement threshold.  The weight is kept small so the driving-quality
 * trend remains the dominant signal.
 */
export const RACING_COMPLEXITY_WEIGHT = 0.000_1;

/**
 * Minimum mean absolute steering magnitude required for steering oscillation
 * to contribute to the oscillation metric.  Gentle corrections below this
 * deadband are treated as smooth steering, so legitimate S-curves through
 * chicanes are not penalized the same as aggressive zig-zags.
 */
export const RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15;

/**
 * Maximum number of sample observations drawn from the score history for the
 * forward-pass evaluation.  Up to 5 evenly-spaced samples are taken to keep the
 * evaluation cheap while capturing enough behavioral variation to detect
 * non-trivial mutations.
 */
const MAX_FORWARD_PASS_SAMPLES = 5;

/**
 * Detects steering oscillation from a window of steering angles.  A series is
 * considered oscillating when the number of sign changes exceeds a small,
 * length-scaled threshold and the mean steering magnitude is above a deadband,
 * which catches aggressive zig-zags without penalizing gentle corrections or
 * legitimate S-curves through chicanes.
 *
 * @param angles - Recent steering outputs in chronological order.
 * @returns Oscillation metric in [0, 1]; 0 means smooth steering and values
 *   near 1 indicate strong back-and-forth swings.
 */
export function detectSteeringOscillation(angles: readonly number[]): number {
  const length = angles.length;
  if (length <= 2) {
    return 0;
  }

  const meanMagnitude =
    angles.reduce((sum, angle) => sum + Math.abs(angle), 0) / length;
  if (meanMagnitude < RACING_OSCILLATION_MIN_MEAN_STEERING) {
    return 0;
  }

  let crossings = 0;
  for (let i = 1; i < length; i++) {
    const previous = angles[i - 1] ?? 0;
    const current = angles[i] ?? 0;
    if (previous === 0 || current === 0) {
      continue;
    }
    if (Math.sign(previous) !== Math.sign(current)) {
      crossings++;
    }
  }

  return Math.min(1, crossings / (length * 0.5));
}

/**
 * Convert one history entry into a scalar driving-quality score.
 *
 * Legacy numeric entries pass through unchanged so existing callers and the
 * default engine can keep using raw score windows.  Composite signals are
 * weighted so that better progress, speed, and alignment increase the score,
 * while a larger off-track penalty decreases it.
 *
 * @param entry - Numeric score or composite driving-quality signal.
 * @returns Scalar quality value for trend/mean scoring.
 */
function toDrivingQuality(entry: number | NeatRacingQualitySignal): number {
  if (typeof entry === 'number') {
    return entry;
  }

  return (
    entry.trackProgress * 0.35 +
    entry.forwardSpeed * 0.25 +
    entry.headingAlignment * 0.3 -
    entry.offTrackPenalty * 0.3 +
    (entry.physicsReward ?? 0) * 0.5
  );
}

/**
 * Detects score-window oscillation from a window of scalar scores or quality
 * signals.  The series is considered oscillating when the number of direction
 * changes (peaks or troughs) exceeds a length-scaled threshold.
 *
 * @param scores - Recent scores in chronological order.
 * @returns Oscillation metric in [0, 1]; 0 means stable scores and values near
 *   1 indicate strong up/down reward swings.
 */
export function detectScoreWindowOscillation(
  scores: readonly (number | NeatRacingQualitySignal)[],
): number {
  const length = scores.length;
  if (length < 4) {
    return 0;
  }

  const values = scores.map((entry) => {
    if (typeof entry === 'number') {
      return entry;
    }

    if (entry.score !== undefined) {
      return entry.score;
    }

    return toDrivingQuality(entry);
  });

  let peaks = 0;
  let troughs = 0;
  for (let i = 1; i < length - 1; i++) {
    const previous = values[i - 1] ?? 0;
    const current = values[i] ?? 0;
    const next = values[i + 1] ?? 0;
    if (previous < current && current > next) {
      peaks++;
    } else if (previous > current && current < next) {
      troughs++;
    }
  }

  return Math.min(1, (peaks + troughs) / (length * 0.5));
}

/**
 * Evaluates the SHARED AC-030 oscillation predicate for ONE adaptation tick.
 *
 * The metric is the maximum of the steering oscillation and the score-window
 * oscillation, computed from the PRE-mutation network's forward-pass outputs
 * over the evidence window.  The steering deadband already lives inside
 * {@link detectSteeringOscillation}, so a gentle correction yields a metric of
 * exactly 0 and never engages the law.
 *
 * @param network - The PRE-mutation network of the tick.
 * @param evidenceWindow - The tick's evidence window.
 * @param state - The previous tick's predicate state (the hysteresis memory).
 * @param level - The session level whose TP-07 node threshold gates the law.
 * @returns The engagement verdict of the tick, carrying the next state.
 */
function evaluateTickOscillationEngagement(
  network: Network,
  evidenceWindow: readonly (number | NeatRacingQualitySignal)[],
  state: OscillationPredicateState,
  level: number,
): OscillationEngagement {
  const forwardPassOutputs = collectForwardPassOutputs(network, evidenceWindow);
  const steeringAngles = forwardPassOutputs.map(
    (outputVector) => outputVector[RACING_STEERING_OUTPUT_INDEX] ?? 0,
  );
  const oscillationMetric = Math.max(
    detectSteeringOscillation(steeringAngles),
    detectScoreWindowOscillation(evidenceWindow),
  );

  return evaluateOscillationEngagement(
    { nodeCount: network.nodes.length, oscillationMetric, level },
    state,
  );
}

/**
 * Builds the idle engagement verdict of a fresh engine.
 *
 * The shared predicate is evaluated against an empty network, so the hard
 * node-count gate keeps the law fully disengaged until the engine's first
 * real tick evaluates it.
 *
 * @param level - The session level whose TP-07 node threshold gates the law.
 * @returns The fully disengaged engagement verdict.
 */
function createIdleOscillationEngagement(level: number): OscillationEngagement {
  return evaluateOscillationEngagement(
    { nodeCount: 0, oscillationMetric: 0, level },
    createOscillationPredicateState(),
  );
}

/**
 * Collects forward-pass outputs from the network by activating it on sample
 * observations drawn from the score history.  For numeric entries the scalar
 * is repeated to fill the input vector; for composite signals the five
 * signal fields are tiled or truncated to the input size.
 *
 * @param network - Candidate network to activate.
 * @param scoreHistory - Rolling score window used to derive observations.
 * @returns Array of output vectors, one per sample observation.
 */
function collectForwardPassOutputs(
  network: Network,
  scoreHistory: readonly (number | NeatRacingQualitySignal)[],
): number[][] {
  const inputSize = network.input;
  if (inputSize <= 0) {
    return [];
  }

  const maxSamples = Math.min(MAX_FORWARD_PASS_SAMPLES, scoreHistory.length);
  const sampleIndices = resolveSampleIndices(scoreHistory.length, maxSamples);

  return sampleIndices.map((index) => {
    const entry = scoreHistory[index] ?? 0;
    const observation = resolveObservationVector(entry, inputSize);
    return [...network.activate(observation)];
  });
}

/**
 * Build a fresh candidate score window from the post-mutation network's
 * forward-pass outputs.
 *
 * Unlike the shared {@link NeatRacingAdaptationTickInput.scoreHistory} (which
 * represents historical driving quality and is identical for both baseline
 * and candidate evaluations), this window is derived by activating the
 * post-mutation network on sample observations from the evidence window and
 * converting each output vector into a scalar quality score.  This ensures
 * the candidate score reflects the actual behavioral impact of the structural
 * mutation, not historical performance.
 *
 * Each output vector is reduced to a scalar by taking the mean of its
 * absolute activation values.  A mutation that disrupts driving behavior
 * produces different activation magnitudes, yielding a different candidate
 * score window and therefore a different candidate score — even when the
 * historical scoreHistory is unchanged.
 *
 * @param network - Post-mutation candidate network to evaluate.
 * @param evidenceWindow - Filtered rolling score history used to derive
 *   sample observations for the forward passes.
 * @returns Array of scalar quality scores, one per sample observation.
 */
function buildCandidateScoreWindow(
  network: Network,
  evidenceWindow: readonly (number | NeatRacingQualitySignal)[],
): number[] {
  const outputs = collectForwardPassOutputs(network, evidenceWindow);
  return outputs.map((outputVector) => reduceOutputToScalar(outputVector));
}

/**
 * Resolve evenly-spaced sample indices from the score history.
 *
 * @param historyLength - Total number of entries in the history.
 * @param maxSamples - Maximum number of samples to select.
 * @returns Array of indices into the score history.
 */
function resolveSampleIndices(
  historyLength: number,
  maxSamples: number,
): number[] {
  if (historyLength <= maxSamples) {
    return Array.from({ length: historyLength }, (_, i) => i);
  }
  const indices: number[] = [];
  for (let i = 0; i < maxSamples; i++) {
    indices.push(Math.floor((i * historyLength) / maxSamples));
  }
  return indices;
}

/**
 * Build an observation vector of the given size from a single score history
 * entry.  Numeric entries are repeated to fill the input; composite signals
 * tile their five fields (or truncate) to match the network input dimension.
 *
 * @param entry - Numeric score or composite driving-quality signal.
 * @param inputSize - Number of input nodes in the candidate network.
 * @returns Input vector suitable for `network.activate`.
 */
function resolveObservationVector(
  entry: number | NeatRacingQualitySignal,
  inputSize: number,
): number[] {
  if (typeof entry === 'number') {
    return Array<number>(inputSize).fill(entry);
  }

  const signalValues = [
    entry.trackProgress,
    entry.forwardSpeed,
    entry.headingAlignment,
    entry.offTrackPenalty,
    entry.physicsReward ?? 0,
  ];

  const observation: number[] = [];
  for (let i = 0; i < inputSize; i++) {
    observation.push(signalValues[i % signalValues.length] ?? 0);
  }
  return observation;
}

/**
 * Compute behavioral complexity as the total variance of forward-pass outputs
 * across sample observations.  If all samples produce identical outputs (e.g.
 * a dead-weight mutation that does not change the forward pass), the variance
 * is zero and the complexity bonus is correctly zero.
 *
 * @param outputs - Array of output vectors, one per sample observation.
 * @returns Total variance across all output dimensions.
 */
function resolveBehavioralComplexity(outputs: readonly number[][]): number {
  if (outputs.length <= 1) {
    return 0;
  }

  const outputSize = outputs[0]?.length ?? 0;
  const mean = new Array<number>(outputSize).fill(0);

  for (const output of outputs) {
    for (let i = 0; i < outputSize; i++) {
      mean[i]! += output[i] ?? 0;
    }
  }

  for (let i = 0; i < outputSize; i++) {
    mean[i]! /= outputs.length;
  }

  let variance = 0;
  for (const output of outputs) {
    for (let i = 0; i < outputSize; i++) {
      const deviation = (output[i] ?? 0) - (mean[i] ?? 0);
      variance += deviation * deviation;
    }
  }

  return variance / outputs.length;
}

/**
 * Racing-specific trend evaluator that consumes a composite driving-quality
 * signal and accounts for network complexity via a forward pass.
 *
 * Each history entry is either a legacy numeric score or a
 * {@link NeatRacingQualitySignal} that carries track progress, forward speed,
 * heading alignment, off-track penalty, and an optional physics reward.
 * The composite quality is folded into the same trend/mean combination used
 * by the default rolling-window evaluator.
 *
 * The complexity bonus is computed from the variance of forward-pass outputs
 * across sample observations drawn from the score history.  A behaviorally-
 * neutral mutation (e.g. a disconnected dead-weight node) produces identical
 * forward-pass outputs and therefore zero variance, yielding no complexity
 * bonus — the evaluator correctly rejects it.  The bonus is additionally
 * gated on a non-negative driving-quality trend so that structural growth is
 * only rewarded when the car is not getting worse.
 *
 * @param network - Candidate network whose forward pass determines behavioral
 *   complexity.
 * @param scoreHistory - Rolling score window of numeric scores or composite
 *   driving-quality signals.
 * @returns Trend/mean score with performance-gated complexity bonus.
 */
export function evaluateRacingTrendScore(
  network: Network,
  scoreHistory: readonly (number | NeatRacingQualitySignal)[],
): number {
  if (scoreHistory.length === 0) {
    return 0;
  }

  const qualities = scoreHistory.map(toDrivingQuality);
  const scoreTrend = (qualities.at(-1) ?? 0) - (qualities[0] ?? 0);
  const scoreMean =
    qualities.reduce(
      (accumulatedQuality, qualityValue) => accumulatedQuality + qualityValue,
      0,
    ) / qualities.length;

  // Forward pass via network.activate() on sample observations to detect
  // behavioral complexity. A dead-weight mutation produces identical outputs
  // → zero variance → zero complexity bonus, so the evaluator rejects
  // behaviorally-neutral mutations.
  const forwardPassOutputs = collectForwardPassOutputs(network, scoreHistory);
  const behavioralComplexity = resolveBehavioralComplexity(forwardPassOutputs);

  // Gate the complexity bonus on driving-quality improvement: only reward
  // structural complexity when the quality trend is non-negative.
  const qualityImprovedOrSame = scoreTrend >= 0;
  const complexityBonus = qualityImprovedOrSame
    ? behavioralComplexity * RACING_COMPLEXITY_WEIGHT
    : 0;

  // The AC-030 oscillation penalty is NOT applied here: the shared predicate
  // in `./oscillation-penalty.utils` owns it, and the adaptation engine
  // applies the tick's single engagement to the baseline and to every variant
  // score.  Applying a second, legacy penalty inside the evaluator would
  // double-penalize the live path and mix score spaces between the engine's
  // `evaluateScore` path and the cycle's `scoreFn` path (a dual path).
  const baseScore = scoreMean + scoreTrend * 0.5;

  return baseScore + complexityBonus;
}

/** Validates and normalizes the cadence options (recycled defaults). */
function resolveCadenceOptions(
  cadence?: NeatRacingAdaptationCadenceOptions,
): NeatRacingAdaptationCadenceOptions {
  if (cadence === undefined) {
    return NEATRACING_ADAPTATION_DEFAULT_CADENCE;
  }

  const everyNTicks = Math.max(1, Math.floor(cadence.everyNTicks ?? 1));
  const boundaryInterval = Math.max(
    1,
    Math.floor(cadence.boundaryInterval ?? 1),
  );

  return {
    ...cadence,
    everyNTicks,
    boundaryInterval,
  };
}

/** Validates and normalizes the capacity/cooldown limits (recycled defaults). */
function resolveLimitOptions(
  limits?: Partial<NeatRacingAdaptationLimits>,
): NeatRacingAdaptationLimits {
  return {
    maxStructuralEditsPerStep: Math.max(
      0,
      Math.floor(
        limits?.maxStructuralEditsPerStep ??
          NEATRACING_ADAPTATION_DEFAULT_LIMITS.maxStructuralEditsPerStep,
      ),
    ),
    maxNodes: Math.max(
      1,
      Math.floor(
        limits?.maxNodes ?? NEATRACING_ADAPTATION_DEFAULT_LIMITS.maxNodes,
      ),
    ),
    maxConnections: Math.max(
      1,
      Math.floor(
        limits?.maxConnections ??
          NEATRACING_ADAPTATION_DEFAULT_LIMITS.maxConnections,
      ),
    ),
    mutationCooldownTicks: Math.max(
      0,
      Math.floor(
        limits?.mutationCooldownTicks ??
          NEATRACING_ADAPTATION_DEFAULT_LIMITS.mutationCooldownTicks,
      ),
    ),
    rollbackCooldownTicks: Math.max(
      0,
      Math.floor(
        limits?.rollbackCooldownTicks ??
          NEATRACING_ADAPTATION_DEFAULT_LIMITS.rollbackCooldownTicks,
      ),
    ),
  };
}

/** Filters non-finite entries out of the rolling score history. */
function resolveEvidenceWindow(
  scoreHistory: readonly (number | NeatRacingQualitySignal)[],
): readonly (number | NeatRacingQualitySignal)[] {
  return scoreHistory.filter((entry) => {
    if (typeof entry === 'number') {
      return Number.isFinite(entry);
    }
    return (
      Number.isFinite(entry.trackProgress) &&
      Number.isFinite(entry.forwardSpeed) &&
      Number.isFinite(entry.headingAlignment) &&
      Number.isFinite(entry.offTrackPenalty) &&
      Number.isFinite(entry.physicsReward ?? 0)
    );
  });
}

/** Decides whether the current tick passes the cadence gate (recycled). */
function isCadenceReady(
  cadence: NeatRacingAdaptationCadenceOptions,
  tickInput: NeatRacingAdaptationTickInput,
  lastLapBoundary: number,
  lastSectorBoundary: number,
  storeBoundaryState: (
    nextLapBoundary: number,
    nextSectorBoundary: number,
  ) => void,
): boolean {
  const lapBoundary = Math.floor(tickInput.completedLaps ?? 0);
  const sectorBoundary = Math.floor(tickInput.completedSectors ?? 0);
  storeBoundaryState(lapBoundary, sectorBoundary);

  if (cadence.mode === 'every_tick') {
    return true;
  }

  if (cadence.mode === 'every_n_ticks') {
    return tickInput.tick % (cadence.everyNTicks ?? 1) === 0;
  }

  if (cadence.mode === 'lap_boundary') {
    if (!Number.isFinite(lastLapBoundary)) {
      return false;
    }
    const lapDelta = lapBoundary - lastLapBoundary;
    return lapDelta >= (cadence.boundaryInterval ?? 1);
  }

  if (!Number.isFinite(lastSectorBoundary)) {
    return false;
  }
  const sectorDelta = sectorBoundary - lastSectorBoundary;
  return sectorDelta >= (cadence.boundaryInterval ?? 1);
}

/**
 * Build a module metrics snapshot from the runtime evidence window.
 *
 * @param network - Live controller network.
 * @param evidenceWindow - Filtered rolling score history.
 * @returns NGE module metrics for the lifecycle focus scorer.
 */
function buildModuleMetricsSnapshot(
  network: Network,
  evidenceWindow: readonly (number | NeatRacingQualitySignal)[],
): NgeModuleMetricsSnapshot {
  const qualities = evidenceWindow.map(toDrivingQuality);
  const scoreTrend =
    qualities.length >= 2 ? (qualities.at(-1) ?? 0) - (qualities[0] ?? 0) : 0;
  const scoreMean =
    qualities.length > 0
      ? qualities.reduce(
          (accumulatedScore, scoreValue) => accumulatedScore + scoreValue,
          0,
        ) / qualities.length
      : 0;

  return {
    moduleId: NEATRACING_ADAPTATION_MODULE_ID,
    utilization: Math.min(scoreMean, 1),
    rewardDelta: scoreTrend,
    novelty: 0,
    stabilityAge: 0,
    wiringCost: network.nodes.length + network.connections.length,
  };
}

/**
 * Build a growth budget from the runtime limits and live network.
 *
 * @param network - Live controller network.
 * @param limits - Runtime adaptation limits.
 * @returns NGE growth budget for the lifecycle apply phase.
 */
function buildGrowthBudget(
  network: Network,
  limits: NeatRacingAdaptationLimits,
): NgeGrowthBudget {
  return {
    maxNodes: limits.maxNodes,
    maxEdges: limits.maxConnections,
    maxEpisodicSlots: NEATRACING_ADAPTATION_MAX_EPISODIC_SLOTS,
    currentNodeCount: network.nodes.length,
    currentEdgeCount: network.connections.length,
    currentEpisodicSlotCount: 0,
  };
}

/**
 * Build a prune budget from the live network.
 *
 * @param network - Live controller network.
 * @returns NGE prune budget for the lifecycle apply phase.
 */
function buildPruneBudget(network: Network): NgePruneBudget {
  return {
    minEdges: 0,
    minNodes: 1,
    costExemptEdgeIds: [],
    currentEdgeCount: network.connections.length,
    currentNodeCount: network.nodes.length,
    currentWiringCost: network.nodes.length + network.connections.length,
  };
}

/** Hard AC-029 capacity gate: candidate size must stay within both caps. */
function passesSafetyChecks(
  networkSize: NeatRacingAdaptationNetworkSizeSnapshot,
  limits: NeatRacingAdaptationLimits,
): boolean {
  return (
    networkSize.nodes <= limits.maxNodes &&
    networkSize.connections <= limits.maxConnections
  );
}

/** Snapshots the live network size for telemetry. */
function resolveNetworkSizeSnapshot(
  network: Network,
): NeatRacingAdaptationNetworkSizeSnapshot {
  return {
    nodes: network.nodes.length,
    connections: network.connections.length,
  };
}

/** Stamps the phase fields onto a telemetry record (recycled helper). */
function createTelemetry(
  telemetry: Omit<
    NeatRacingAdaptationTelemetry,
    | 'adaptationPhase'
    | 'stabilizationTicksSinceGrowth'
    | 'tickCostMs'
    | 'withinTickBudget'
    | 'oscillationEngagement'
  >,
  adaptationPhase: 'growth' | 'stabilization' = 'growth',
  stabilizationTicksSinceGrowth: number = 0,
): Omit<
  NeatRacingAdaptationTelemetry,
  'tickCostMs' | 'withinTickBudget' | 'oscillationEngagement'
> {
  return {
    ...telemetry,
    adaptationPhase,
    stabilizationTicksSinceGrowth,
  };
}
