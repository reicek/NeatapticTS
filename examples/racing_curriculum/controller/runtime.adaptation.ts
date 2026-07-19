/**
 * Racing-curriculum runtime adaptation engine.
 *
 * Sequences one NGE grow-stabilize cycle per controller tick. It converts
 * composite driving-quality signals into scalar scores, decides whether the
 * controller network should grow new structure or stabilize existing weights,
 * and routes the resulting mutations back to the live network.
 *
 * The engine is deliberately decoupled from the simulation worker so the same
 * adaptation policy can run in the browser host, in a worker, or in unit tests.
 * All demo-specific knowledge lives here; the core `runNgeGrowStabilizeCycle`
 * only sees plain numeric score history and a mutable network.
 *
 * ## Score-space invariant
 *
 * The grow-stabilize cycle commits a weight variant only when
 * `bestVariantScore > baselineScore + threshold`. That inequality is only
 * meaningful when both scores share the same units and direction. The default
 * variant scorer returns negative mean-squared-error against a target vector,
 * while the racing baseline is a positive driving-quality score. This engine
 * therefore injects a racing-specific `VariantScorer` that collapses the
 * 2-D controller output `[throttle, steering]` to a scalar and returns a
 * positive quality score in the same space as the baseline.
 *
 * ```mermaid
 * flowchart LR
 *   Tick["Controller tick"] --> Cadence{"Cadence gate open?"}
 *   Cadence -->|no| Skip["Skip adaptation"]
 *   Cadence -->|yes| Evidence["Build evidence window"]
 *   Evidence --> Baseline["Compute positive driving-quality baseline"]
 *   Baseline --> Cycle["runNgeGrowStabilizeCycle"]
 *   Cycle --> Committed{"Committed?"}
 *   Committed -->|yes| Apply["Apply mutation / keep weights"]
 *   Committed -->|no| Rollback["Rollback network"]
 *   Apply --> Telemetry["Emit telemetry"]
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
import { Connection, Network } from '../../../src/browser-entry';
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
/** Cadence modes supported by the runtime adaptation engine. */
export type RuntimeAdaptationCadenceMode =
  'every_tick' | 'every_n_ticks' | 'lap_boundary' | 'sector_boundary';

/** Candidate mutation operations supported by the runtime adaptation engine. */
export type RuntimeAdaptationOperation =
  'param_nudge' | 'add_edge' | 'add_node' | 'prune_edge';

/** Per-step network size snapshot used by adaptation telemetry. */
export interface RuntimeNetworkSizeSnapshot {
  /** Live node count at a specific adaptation checkpoint. */
  readonly nodes: number;
  /** Live connection count at a specific adaptation checkpoint. */
  readonly connections: number;
}

/** Adaptation telemetry emitted on every adaptation attempt. */
export interface RuntimeAdaptationTelemetry {
  /** Fixed-timestep tick at which the adaptation attempt was processed. */
  readonly tick: number;
  /** Ordered operations applied to the candidate (or attempted before rollback). */
  readonly operations: readonly RuntimeAdaptationOperation[];
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
  readonly networkSizeBefore: RuntimeNetworkSizeSnapshot;
  /** Network size after commit/rollback resolution. */
  readonly networkSizeAfter: RuntimeNetworkSizeSnapshot;
  /** Current adaptation phase — growth (structural) or stabilization (weight tuning). */
  readonly adaptationPhase: 'growth' | 'stabilization';
  /** Ticks elapsed in the stabilization phase since the last structural growth. */
  readonly stabilizationTicksSinceGrowth: number;
}

/**
 * Composite driving-quality signal used by the racing trend evaluator.
 *
 * Encapsulates the four per-tick telemetry components that together describe
 * how well the car is driving: spline-track progress, forward speed, heading
 * alignment with the track, and an off-track penalty.  The composite replaces
 * the older heading-alignment-only scalar.
 */
export interface RacingQualitySignal {
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

/** Cadence policy configuration for per-tick adaptation checks. */
export interface RuntimeAdaptationCadenceOptions {
  /** Trigger mode used to decide whether the current tick should adapt. */
  readonly mode: RuntimeAdaptationCadenceMode;
  /** Tick interval used by `every_n_ticks`. */
  readonly everyNTicks?: number;
  /** Boundary interval used by `lap_boundary` and `sector_boundary`. */
  readonly boundaryInterval?: number;
}

/** Hard bounds and cooldown controls for one adaptation step. */
export interface RuntimeAdaptationLimits {
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

/** Engine options used by the racing runtime adaptation POC. */
export interface RuntimeAdaptationEngineOptions {
  /** Cadence policy for adaptation checks. */
  readonly cadence?: RuntimeAdaptationCadenceOptions;
  /** Hard bounds and cooldown settings. */
  readonly limits?: Partial<RuntimeAdaptationLimits>;
  /** Minimum score improvement required to commit a candidate. */
  readonly improvementThreshold?: number;
  /** Minimum rolling evidence window length required to evaluate. */
  readonly minimumEvidenceWindow?: number;
  /** Optional deterministic random source for operation proposal. */
  readonly random?: () => number;
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
    scoreHistory: readonly (number | RacingQualitySignal)[],
  ) => number;
}

/** Per-tick input contract for adaptation checks. */
export interface RuntimeAdaptationTickInput {
  /** Current fixed-timestep tick. */
  readonly tick: number;
  /** Live mutable controller network. */
  readonly network: Network;
  /** Rolling score history assembled by the caller. */
  readonly scoreHistory: readonly (number | RacingQualitySignal)[];
  /** Current completed lap count (for `lap_boundary` cadence). */
  readonly completedLaps?: number;
  /** Current completed sector count (for `sector_boundary` cadence). */
  readonly completedSectors?: number;
}

/** Stateful runtime adaptation engine surface used by browser or worker loops. */
export interface RuntimeAdaptationEngine {
  /**
   * Runs one adaptation decision against the current tick input.
   *
   * @param tickInput - Per-tick runtime inputs.
   * @returns Promise resolving to deterministic-friendly telemetry.
   */
  adaptOnTick(
    tickInput: RuntimeAdaptationTickInput,
  ): Promise<RuntimeAdaptationTelemetry>;
  /** Resets cadence boundaries and cooldown state. */
  reset(): void;
}

/**
 * Score-window separation contract for the racing trend evaluator.
 *
 * The evaluator distinguishes a preMutationBaseline score (captured from the
 * pre-mutation network before lifecycle morphs are applied) from a
 * postMutationCandidate score (computed from a fresh candidateScoreWindow
 * derived via forward-pass evaluation of the post-mutation network).  This
 * prevents the commit/rollback decision from being driven by the shared
 * scoreHistory rolling window, ensuring structural mutations are evaluated
 * on their actual driving-quality impact rather than historical performance.
 */
const DEFAULT_CADENCE: RuntimeAdaptationCadenceOptions = {
  mode: 'every_n_ticks',
  everyNTicks: 4,
};

const DEFAULT_LIMITS: RuntimeAdaptationLimits = {
  maxStructuralEditsPerStep: 5,
  maxNodes: 8_000,
  maxConnections: 32_000,
  mutationCooldownTicks: 5,
  rollbackCooldownTicks: 5,
};

const DEFAULT_IMPROVEMENT_THRESHOLD = 0.02;
const DEFAULT_MINIMUM_EVIDENCE_WINDOW = 4;
const RUNTIME_MODULE_ID = 'racing:runtime';

export { resolveAdaptiveHysteresis };

/**
 * Maximum number of episodic growth slots the NGE lifecycle may allocate
 * during adaptation. A positive value enables the slot-expansion path in
 * `neat.nge-juvenile.grow.ts`; setting it to zero silently disables growth.
 */
const MAX_EPISODIC_SLOTS = 15;

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
 * @param outputs - Stack of network output vectors, one per input sample.
 * @param _target - Scalar target value for each sample (unused).
 * @param neuronCount - Live neuron count used to tier-gate the oscillation
 *   penalty.  Penalty is skipped below
 *   {@link RACING_OSCILLATION_MIN_NEURONS}.
 * @returns Positive racing-trend quality score (higher is better).
 */
export function scoreRacingVariant(
  outputs: readonly number[][],
  _target: readonly number[],
  neuronCount: number = Number.POSITIVE_INFINITY,
): number {
  if (outputs.length === 0) {
    return 0;
  }

  // Step 1: collapse the multi-dimensional controller output to a scalar
  // driving-quality proxy, exactly like {@link buildCandidateScoreWindow}.
  const scalarWindow = outputs.map((outputVector) =>
    reduceOutputToScalar(outputVector),
  );

  // Step 2: compute the same positive trend/mean/complexity score used for
  // the pre-mutation baseline so the stabilization commit inequality compares
  // values in the same units.
  const scoreTrend = scalarWindow.at(-1)! - scalarWindow[0]!;
  const scoreMean =
    scalarWindow.reduce((accumulated, value) => accumulated + value, 0) /
    scalarWindow.length;
  const behavioralComplexity = resolveBehavioralComplexity(outputs);
  const qualityImprovedOrSame = scoreTrend >= 0;
  const complexityBonus = qualityImprovedOrSame
    ? behavioralComplexity * RACING_COMPLEXITY_WEIGHT
    : 0;

  // Step 3: penalize rapid steering oscillation so the adaptation loop
  // rewards sustained progress instead of noisy back-and-forth control.
  // The penalty is proportional to the base score so it damps oscillation
  // without inverting a positive score into a negative one.  The penalty is
  // only applied once a network has reached the child tier size (200+ neurons);
  // newborn and baby networks naturally oscillate while learning to steer and
  // must not be trapped by the penalty.
  const baseScore = scoreMean + scoreTrend * 0.5;
  const steeringAngles = outputs.map((outputVector) => outputVector[1] ?? 0);
  const steeringOscillation = detectSteeringOscillation(steeringAngles);
  const scoreOscillation = detectScoreWindowOscillation(scalarWindow);
  const oscillationMetric = Math.max(steeringOscillation, scoreOscillation);
  const applyOscillationPenalty = neuronCount >= RACING_OSCILLATION_MIN_NEURONS;
  const oscillationPenalty = applyOscillationPenalty
    ? baseScore * oscillationMetric * RACING_OSCILLATION_PENALTY_WEIGHT
    : 0;

  return baseScore + complexityBonus - oscillationPenalty;
}

/**
 * Racing-specific variant scorer for the NGE grow-stabilize cycle.
 *
 * This is a {@link VariantScorer}-compatible wrapper around
 * {@link scoreRacingVariant}.  Because the variant-scorer contract only passes
 * `outputs` and `target`, the wrapper uses a default neuron count of
 * `Number.POSITIVE_INFINITY` so direct callers continue to apply the
 * oscillation penalty.  The runtime engine calls {@link scoreRacingVariant}
 * directly with the live candidate size to tier-gate young networks.
 *
 * @example
 * ```ts
 * const outputs = [
 *   [0.5, -0.1], // throttle, steering
 *   [0.6, 0.0],
 * ];
 * const score = RACING_VARIANT_SCORER(outputs, [0]);
 * // score is a positive driving-quality proxy; higher is better
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
 *
 * @param outputs - Stack of network output vectors, one per input sample.
 * @param target - Scalar target value for each sample (unused).
 * @returns Positive racing-trend quality score (higher is better).
 */
export const RACING_VARIANT_SCORER: VariantScorer = (outputs, target) =>
  scoreRacingVariant(outputs, target, Number.POSITIVE_INFINITY);

/**
 * Creates a reusable per-tick adaptation engine for racing runtime loops.
 *
 * The engine sequences one NGE grow-stabilize cycle per tick. It enforces a
 * configurable cadence policy so adaptation attempts do not fire on every tick,
 * applies a rollback cooldown after rejected candidates, and respects a
 * growth throttle that slows structural mutation as the network grows. The
 * growth-phase commit decision trusts the grow-stabilize cycle's own commit
 * flag, while the engine adds safety-limit checks and an unconditional
 * first-growth path so a brand-new network cannot stall.
 *
 * When an `accelerationConfig` is supplied, variant counts are forwarded to the
 * grow-stabilize cycle's parallel evaluator. The stabilization phase caps the
 * evaluated variant count to `NGE_GROW_STABILIZE_STABILIZATION_VARIANT_COUNT`
 * (32 by default) regardless of the acceleration configuration, while the
 * growth-phase variant count follows the acceleration configuration's stage
 * limits.
 *
 * A racing-specific `VariantScorer` is injected into the grow-stabilize
 * cycle as `scoreFn` so the stabilization baseline and variant scores share the
 * same positive driving-quality score space. Without that alignment, the commit
 * inequality `bestScore > baselineScore + threshold` compares incommensurate
 * values (for example, negative MSE against a positive quality score) and
 * stabilization commits cannot occur.
 *
 * @param options - Optional cadence, bounds, evaluation policy, and
 *   acceleration configuration.
 * @returns Stateful runtime adaptation engine.
 */
export function createRuntimeAdaptationEngine(
  options: RuntimeAdaptationEngineOptions = {},
): RuntimeAdaptationEngine {
  const cadence = resolveCadenceOptions(options.cadence);
  const limits = resolveLimitOptions(options.limits);
  const evaluateScore = options.evaluateScore ?? evaluateRacingTrendScore;
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
  const randomSource = options.random ?? Math.random;

  return {
    async adaptOnTick(
      tickInput: RuntimeAdaptationTickInput,
    ): Promise<RuntimeAdaptationTelemetry> {
      const networkSizeBefore = resolveNetworkSizeSnapshot(tickInput.network);
      const evidenceWindow = resolveEvidenceWindow(tickInput.scoreHistory);
      const baselineScore = evaluateScore(tickInput.network, evidenceWindow);
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
      const preMutationBaselineScore = evaluateScore(
        tickInput.network,
        preMutationScoreWindow,
      );

      // Snapshot the network and the global connection innovation counter
      // so a rollback can restore both pieces of process state.
      const capturedInnovation = Connection.nextInnovation;
      const rollbackSnapshot = tickInput.network.toJSON();

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
          scoreRacingVariant(outputs, target, tickInput.network.nodes.length),
        lifecycleStage: 'baby',
        config: {
          maxStructuralEditsPerStep: limits.maxStructuralEditsPerStep,
          maxNodes: limits.maxNodes,
          maxConnections: limits.maxConnections,
          maxEpisodicSlots: MAX_EPISODIC_SLOTS,
          moduleId: RUNTIME_MODULE_ID,
        },
        lifecycleRunner: (lifecycleInput) => {
          const { stage, ...lifecycleRest } = lifecycleInput;
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

      // Carry the grow-stabilize cycle state forward so that weight-exhaustion
      // counting, the post-growth anti-runaway boost, and the pre-growth
      // baseline persist across adaptation ticks.
      consecutiveWeightExhaustion =
        cycleResult.consecutiveWeightExhaustion ?? 0;
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
        const stabilizationScore = evaluateScore(
          tickInput.network,
          stabilizationScoreWindow,
        );
        const stabilizationImprovement = stabilizationScore - baselineScore;

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
      const operations = cycleResult.operations as RuntimeAdaptationOperation[];

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
      const candidateScore = evaluateScore(
        tickInput.network,
        candidateScoreWindow,
      );
      const improvement = candidateScore - preMutationBaselineScore;
      // For the very first growth, commit unconditionally (if safety checks
      // pass) — the structural mutation adds capacity that the stabilization
      // phase will tune; requiring an immediate score improvement would
      // rollback the first growth and leave the network stuck forever.
      const isFirstGrowth = !hasGrownBefore;
      // Boost the improvement threshold when either the evidence window is
      // oscillating or the candidate network's steering outputs are zig-zagging,
      // making it harder to commit structural growth during unstable reward
      // windows.
      const candidateOutputs = collectForwardPassOutputs(
        tickInput.network,
        evidenceWindow,
      );
      const steeringAngles = candidateOutputs.map(
        (outputVector) => outputVector[1] ?? 0,
      );
      const steeringOscillation = detectSteeringOscillation(steeringAngles);
      const scoreOscillation = detectScoreWindowOscillation(evidenceWindow);
      const oscillationMetric = Math.max(steeringOscillation, scoreOscillation);
      const effectiveImprovementThreshold =
        improvementThreshold +
        resolveOscillationThresholdBoost(
          oscillationMetric,
          tickInput.network.nodes.length,
        );
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
    },
  };
}

/**
 * Creates one independent runtime adaptation engine per car index.
 *
 * Each car in a multi-car racing simulation maintains its own adaptation
 * state, cooldowns, and cadence boundaries.  This factory creates a
 * `Map<number, RuntimeAdaptationEngine>` keyed by car index (0 to
 * `carCount - 1`) where every engine has fully independent closure-scoped
 * state — no shared mutable state across cars.
 *
 * @param carCount - Number of cars to create engines for.
 * @param options - Optional engine options applied identically to every car's engine.
 * @returns Map keyed by car index of independent adaptation engines.
 * @example
 * ```ts
 * const engines = createPerCarAdaptationEngines(3, {
 *   limits: { mutationCooldownTicks: 100 },
 * });
 * const car0Engine = engines.get(0); // independent state
 * const car1Engine = engines.get(1); // independent state
 * ```
 */
export function createPerCarAdaptationEngines(
  carCount: number,
  options: RuntimeAdaptationEngineOptions = {},
): Map<number, RuntimeAdaptationEngine> {
  const engines = new Map<number, RuntimeAdaptationEngine>();
  const safeCarCount = Math.max(0, Math.floor(carCount));

  for (let carIndex = 0; carIndex < safeCarCount; carIndex++) {
    const carOptions: RuntimeAdaptationEngineOptions = {
      ...options,
      carId: options.carId ?? carIndex,
    };
    engines.set(carIndex, createRuntimeAdaptationEngine(carOptions));
  }

  return engines;
}

/**
 * Lightweight default evaluator for rolling score history windows.
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

  const scoreTrend = scoreHistory.at(-1)! - scoreHistory[0]!;
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
 * Weight applied to steering- and score-window oscillation penalties in the
 * racing trend evaluator.  A positive weight penalizes wild steering swings and
 * reward oscillation, steering evolution toward smooth, consistent driving
 * behavior.
 */
export const RACING_OSCILLATION_PENALTY_WEIGHT = 0.25;

/**
 * Oscillation gate for the growth-commit threshold boost.  The small additive
 * {@link RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST} is only applied when
 * the maximum of steering and score-window oscillation exceeds this value;
 * below the gate the threshold is left unchanged.  This is a gate, not a
 * multiplier.
 */
export const RACING_OSCILLATION_COMMIT_THRESHOLD = 0.5;

/**
 * Small additive boost added to the growth commit threshold when the
 * oscillation metric is above {@link RACING_OSCILLATION_COMMIT_THRESHOLD}.  It
 * is not multiplied by the metric; it is a fixed nudge that makes structural
 * commits slightly harder during unstable reward windows without creating a
 * death spiral for young networks.
 */
export const RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST = 0.03;

/**
 * Minimum live neuron count at which oscillation penalties and the growth-commit
 * threshold boost are applied.  Newborn and baby networks (< 200 neurons)
 * naturally oscillate while learning to steer, so the adaptation loop must not
 * penalize them until they reach the child tier.
 */
export const RACING_OSCILLATION_MIN_NEURONS = 200;

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
 * Resolves the oscillation-driven additive boost for the growth-commit
 * improvement threshold.  The boost is only returned when the network has
 * reached the child tier (200+ neurons) and the oscillation metric is above the
 * commit gate; otherwise it returns zero so young or smooth networks are not
 * saddled with an extra growth bar.
 *
 * @param oscillationMetric - Maximum of steering and score-window oscillation.
 * @param neuronCount - Live neuron count of the candidate network.
 * @returns Additive threshold boost (0 or
 *   {@link RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST}).
 */
export function resolveOscillationThresholdBoost(
  oscillationMetric: number,
  neuronCount: number,
): number {
  if (neuronCount < RACING_OSCILLATION_MIN_NEURONS) {
    return 0;
  }

  if (oscillationMetric <= RACING_OSCILLATION_COMMIT_THRESHOLD) {
    return 0;
  }

  return RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST;
}

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
    const previous = angles[i - 1]!;
    const current = angles[i]!;
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
function toDrivingQuality(entry: number | RacingQualitySignal): number {
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
  scores: readonly (number | RacingQualitySignal)[],
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
    const previous = values[i - 1]!;
    const current = values[i]!;
    const next = values[i + 1]!;
    if (previous < current && current > next) {
      peaks++;
    } else if (previous > current && current < next) {
      troughs++;
    }
  }

  return Math.min(1, (peaks + troughs) / (length * 0.5));
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
  scoreHistory: readonly (number | RacingQualitySignal)[],
): number[][] {
  const inputSize = network.input;
  if (inputSize <= 0) {
    return [];
  }

  const maxSamples = Math.min(MAX_FORWARD_PASS_SAMPLES, scoreHistory.length);
  const sampleIndices = resolveSampleIndices(scoreHistory.length, maxSamples);

  return sampleIndices.map((index) => {
    const entry = scoreHistory[index]!;
    const observation = resolveObservationVector(entry, inputSize);
    return [...network.activate(observation)];
  });
}

/**
 * Build a fresh candidate score window from the post-mutation network's
 * forward-pass outputs.
 *
 * Unlike the shared {@link RuntimeAdaptationTickInput.scoreHistory} (which
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
  evidenceWindow: readonly (number | RacingQualitySignal)[],
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
  entry: number | RacingQualitySignal,
  inputSize: number,
): number[] {
  if (typeof entry === 'number') {
    return Array(inputSize).fill(entry);
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

  const outputSize = outputs[0]!.length;
  const mean = new Array<number>(outputSize).fill(0);

  for (const output of outputs) {
    for (let i = 0; i < outputSize; i++) {
      mean[i] += output[i]!;
    }
  }

  for (let i = 0; i < outputSize; i++) {
    mean[i] /= outputs.length;
  }

  let variance = 0;
  for (const output of outputs) {
    for (let i = 0; i < outputSize; i++) {
      const deviation = output[i]! - mean[i]!;
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
 * {@link RacingQualitySignal} that carries track progress, forward speed,
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
  scoreHistory: readonly (number | RacingQualitySignal)[],
): number {
  if (scoreHistory.length === 0) {
    return 0;
  }

  const qualities = scoreHistory.map(toDrivingQuality);
  const scoreTrend = qualities.at(-1)! - qualities[0]!;
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

  // Penalize score-window oscillation so the trend evaluator prefers stable,
  // monotonic improvement over noisy reward swings.  The penalty is proportional
  // to the base score so it damps oscillation without inverting the score.  Like
  // the variant scorer, the penalty is gated to child-tier networks so young
  // networks are not punished for the natural learning oscillations that come
  // with learning to steer.
  const baseScore = scoreMean + scoreTrend * 0.5;
  const scoreOscillation = detectScoreWindowOscillation(scoreHistory);
  const applyOscillationPenalty =
    network.nodes.length >= RACING_OSCILLATION_MIN_NEURONS;
  const oscillationPenalty = applyOscillationPenalty
    ? baseScore * scoreOscillation * RACING_OSCILLATION_PENALTY_WEIGHT
    : 0;

  return baseScore + complexityBonus - oscillationPenalty;
}

function resolveCadenceOptions(
  cadence?: RuntimeAdaptationCadenceOptions,
): RuntimeAdaptationCadenceOptions {
  if (cadence === undefined) {
    return DEFAULT_CADENCE;
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

function resolveLimitOptions(
  limits?: Partial<RuntimeAdaptationLimits>,
): RuntimeAdaptationLimits {
  return {
    maxStructuralEditsPerStep: Math.max(
      0,
      Math.floor(
        limits?.maxStructuralEditsPerStep ??
          DEFAULT_LIMITS.maxStructuralEditsPerStep,
      ),
    ),
    maxNodes: Math.max(
      1,
      Math.floor(limits?.maxNodes ?? DEFAULT_LIMITS.maxNodes),
    ),
    maxConnections: Math.max(
      1,
      Math.floor(limits?.maxConnections ?? DEFAULT_LIMITS.maxConnections),
    ),
    mutationCooldownTicks: Math.max(
      0,
      Math.floor(
        limits?.mutationCooldownTicks ?? DEFAULT_LIMITS.mutationCooldownTicks,
      ),
    ),
    rollbackCooldownTicks: Math.max(
      0,
      Math.floor(
        limits?.rollbackCooldownTicks ?? DEFAULT_LIMITS.rollbackCooldownTicks,
      ),
    ),
  };
}

function resolveEvidenceWindow(
  scoreHistory: readonly (number | RacingQualitySignal)[],
): readonly (number | RacingQualitySignal)[] {
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

function isCadenceReady(
  cadence: RuntimeAdaptationCadenceOptions,
  tickInput: RuntimeAdaptationTickInput,
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
  evidenceWindow: readonly (number | RacingQualitySignal)[],
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
    moduleId: RUNTIME_MODULE_ID,
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
  limits: RuntimeAdaptationLimits,
): NgeGrowthBudget {
  return {
    maxNodes: limits.maxNodes,
    maxEdges: limits.maxConnections,
    maxEpisodicSlots: MAX_EPISODIC_SLOTS,
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

function passesSafetyChecks(
  networkSize: RuntimeNetworkSizeSnapshot,
  limits: RuntimeAdaptationLimits,
): boolean {
  return (
    networkSize.nodes <= limits.maxNodes &&
    networkSize.connections <= limits.maxConnections
  );
}

function resolveNetworkSizeSnapshot(
  network: Network,
): RuntimeNetworkSizeSnapshot {
  return {
    nodes: network.nodes.length,
    connections: network.connections.length,
  };
}

function createTelemetry(
  telemetry: Omit<
    RuntimeAdaptationTelemetry,
    'adaptationPhase' | 'stabilizationTicksSinceGrowth'
  >,
  adaptationPhase: 'growth' | 'stabilization' = 'growth',
  stabilizationTicksSinceGrowth: number = 0,
): RuntimeAdaptationTelemetry {
  return { ...telemetry, adaptationPhase, stabilizationTicksSinceGrowth };
}
