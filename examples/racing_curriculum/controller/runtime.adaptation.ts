import { Connection, Network } from '../../../src/browser-entry.ts';
import {
  restoreNetworkSnapshot,
  runNgeLifecycle,
} from '../../../src/neat/neat.nge-lifecycle';
import { advanceGrowthHysteresis } from '../../../src/neat/nge-juvenile/neat.nge-juvenile.ts';
import type { MorphApplyOutcome } from '../../../src/neat/nge-juvenile/neat.nge-juvenile.apply.ts';
import type {
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
} from '../../../src/neat/nge-juvenile/neat.nge-juvenile.types.ts';

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
   * @returns Deterministic-friendly telemetry for the decision.
   */
  adaptOnTick(
    tickInput: RuntimeAdaptationTickInput,
  ): RuntimeAdaptationTelemetry;
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
  maxStructuralEditsPerStep: 1,
  maxNodes: 8_000,
  maxConnections: 32_000,
  mutationCooldownTicks: 5,
  rollbackCooldownTicks: 5,
};

const DEFAULT_IMPROVEMENT_THRESHOLD = 0.01;
const DEFAULT_MINIMUM_EVIDENCE_WINDOW = 4;
const RUNTIME_MODULE_ID = 'racing:runtime';

/**
 * Maximum number of episodic growth slots the NGE lifecycle may allocate
 * during adaptation. A positive value enables the slot-expansion path in
 * `neat.nge-juvenile.grow.ts`; setting it to zero silently disables growth.
 */
const MAX_EPISODIC_SLOTS = 15;

/**
 * Node count above which the growth throttle engages.
 * Networks with more nodes than this threshold get progressively longer
 * back-off intervals to preserve real-time performance at scale.
 */
const LARGE_NETWORK_NODE_THRESHOLD = 1_000;

/**
 * Base throttle interval (in ticks) applied when the network exceeds the
 * large-network threshold. The effective interval scales with network size.
 */
const GROWTH_THROTTLE_BASE_INTERVAL_TICKS = 3;

/**
 * Resolve the adaptive hysteresis window count based on the live network
 * node count. Smaller networks use a lower threshold (2 consecutive
 * positive-quality windows) to accelerate early growth, while larger
 * networks require more sustained evidence (5 windows) before committing
 * to further structural expansion.
 *
 * @param nodeCount - Current total node count in the live network.
 * @returns Hysteresis window count: 2 for ≤ 200 nodes, 3 for ≤ 500, 5 for > 500.
 */
export function resolveAdaptiveHysteresis(nodeCount: number): number {
  if (nodeCount <= 200) {
    return 2;
  }
  if (nodeCount <= 500) {
    return 3;
  }
  return 5;
}

/**
 * Maximum number of quality-score entries retained for plateau detection.
 * The rolling window tracks the baseline score at each adaptation tick to
 * determine whether the network has stabilized before allowing growth.
 */
const PLATEAU_WINDOW_SIZE = 5;

/**
 * Variance threshold below which the quality score is considered plateaued.
 * When the rolling-window variance falls below this value, the network is
 * deemed to have learned to use its current structure and further growth
 * is permitted. A variance at or above this value indicates ongoing
 * learning — growth is blocked until the score stabilizes.
 */
const PLATEAU_VARIANCE_THRESHOLD = 0.1;

/**
 * Fraction of connections whose weights are perturbed during each
 * stabilization-phase adaptation tick. A moderate rate ensures enough
 * exploration without completely disrupting learned behavior.
 */
const WEIGHT_MUTATION_RATE = 0.3;

/**
 * Maximum magnitude of weight perturbation applied during stabilization.
 * Each selected connection's weight is shifted by a random value in
 * [-WEIGHT_MUTATION_MAGNITUDE, +WEIGHT_MUTATION_MAGNITUDE].
 */
const WEIGHT_MUTATION_MAGNITUDE = 0.1;

/**
 * Minimum stabilization ticks that must elapse after structural growth
 * before plateau detection can fire. This prevents premature growth
 * cycles by ensuring the network has time to learn its new structure.
 */
const MIN_STABILIZATION_TICKS = 5;

/**
 * Maximum stabilization ticks after which growth is forced to re-enter
 * even if the quality score has not plateaued. This time-box prevents
 * the network from getting stuck in an indefinitely long stabilization
 * phase when the score remains noisy.
 */
const MAX_STABILIZATION_TICKS = 25;

/**
 * Creates a reusable per-tick adaptation engine for racing runtime loops.
 *
 * @param options - Optional cadence, bounds, and evaluation policy.
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
  const randomSource = options.random ?? Math.random;

  return {
    adaptOnTick(
      tickInput: RuntimeAdaptationTickInput,
    ): RuntimeAdaptationTelemetry {
      const networkSizeBefore = resolveNetworkSizeSnapshot(tickInput.network);
      const evidenceWindow = resolveEvidenceWindow(tickInput.scoreHistory);
      const baselineScore = evaluateScore(tickInput.network, evidenceWindow);

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
      if (qualityScoreHistory.length > PLATEAU_WINDOW_SIZE) {
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

      // Step 3.6: Plateau detection — determine whether the network has
      //           stabilized enough for structural growth. Before the first
      //           growth, always allow growth. After growth, require the
      //           quality score to plateau (low variance) before allowing
      //           further structural changes. When plateau is not reached,
      //           run weight mutation stabilization to help the network
      //           learn to use its current structure.
      // plateau_not_reached: run weight stabilization instead of structural growth
      const plateauReached = isPlateauReached(
        qualityScoreHistory,
        hasGrownBefore,
        stabilizationTicksSinceGrowth,
      );

      if (!plateauReached) {
        currentPhase = 'stabilization';
        stabilizationTicksSinceGrowth++;

        const stabilizationInnovation = Connection.nextInnovation;
        const stabilizationSnapshot = tickInput.network.toJSON();
        const mutatedCount = applyWeightMutations(
          tickInput.network,
          randomSource,
        );

        if (mutatedCount === 0) {
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
        restoreNetworkSnapshot(
          tickInput.network,
          stabilizationSnapshot,
          stabilizationInnovation,
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

      // Step 4: Build lifecycle inputs from the runtime state.
      const metrics = buildModuleMetricsSnapshot(
        tickInput.network,
        evidenceWindow,
      );
      const growthBudget = buildGrowthBudget(tickInput.network, limits);
      const pruneBudget = buildPruneBudget(tickInput.network);

      // Step 4.5: Capture the preMutationBaselineScore BEFORE any lifecycle
      //           mutation is applied.  This represents the network's driving
      //           quality before the structural change and is used as the
      //           comparison anchor for the post-mutation candidate score.
      //           It is computed with buildCandidateScoreWindow on the
      //           pre-mutation network so the baseline and candidate scores
      //           use the SAME forward-pass scoring method, ensuring an
      //           apples-to-apples comparison that measures the actual
      //           topology-change impact rather than mixing historical
      //           quality scores with forward-pass candidate scores.
      const preMutationScoreWindow = buildCandidateScoreWindow(
        tickInput.network,
        evidenceWindow,
      );
      const preMutationBaselineScore = evaluateScore(
        tickInput.network,
        preMutationScoreWindow,
      );

      // Step 5: Snapshot the network and the global connection innovation
      //         counter so a rollback can restore both pieces of process state.
      const capturedInnovation = Connection.nextInnovation;
      const rollbackSnapshot = tickInput.network.toJSON();

      // Step 6: Call runNgeLifecycle to plan and apply growth morphs.
      //         For the very first growth (!hasGrownBefore), pre-satisfy the
      //         hysteresis gate so the lifecycle produces candidates
      //         immediately — the car hasn't driven long enough to accumulate
      //         5 consecutive positive-quality windows, but growth must happen
      //         first so the network has capacity to learn.
      const isFirstGrowth = !hasGrownBefore;
      const currentNodeCount = tickInput.network.nodes.length;
      const adaptiveHysteresis = resolveAdaptiveHysteresis(currentNodeCount);
      const lifecycleHysteresis = isFirstGrowth
        ? { ...hysteresis, growthPositiveWindowCount: adaptiveHysteresis }
        : hysteresis;
      const lifecycleResult = runNgeLifecycle({
        stage: 'juvenile',
        moduleId: RUNTIME_MODULE_ID,
        metrics,
        budget: growthBudget,
        config: {
          hysteresisWindowCount: adaptiveHysteresis,
          cooldownWindowCount: limits.mutationCooldownTicks,
        },
        hysteresis: lifecycleHysteresis,
        network: tickInput.network,
        pruneBudget,
      });

      // Step 7: Check applyOutcomes for applied morphs.
      const applyOutcomes = lifecycleResult.applyOutcomes ?? [];
      const operations = mapOutcomesToOperations(applyOutcomes);

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

      // Step 8: Evaluate candidate score after lifecycle mutation.
      //         The candidateScoreWindow is built from fresh forward-pass
      //         outputs of the post-mutation network, NOT from the shared
      //         scoreHistory/evidenceWindow.  This ensures the candidate score
      //         reflects the actual driving-quality impact of the structural
      //         mutation, not historical performance.
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
      const shouldCommit =
        safetyChecksPass &&
        (isFirstGrowth || improvement >= improvementThreshold);

      // Step 9: Commit or rollback based on score improvement.
      if (shouldCommit) {
        hysteresis = lifecycleResult.hysteresis ?? hysteresis;
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

      // Step 10: Rollback the network and the global innovation counter, then
      //          set the rollback cooldown.
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
    engines.set(carIndex, createRuntimeAdaptationEngine(options));
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
const RACING_COMPLEXITY_WEIGHT = 0.000_1;

/**
 * Maximum number of sample observations drawn from the score history for the
 * forward-pass evaluation.  Up to 5 evenly-spaced samples are taken to keep the
 * evaluation cheap while capturing enough behavioral variation to detect
 * non-trivial mutations.
 */
const MAX_FORWARD_PASS_SAMPLES = 5;

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
  return outputs.map((outputVector) => {
    if (outputVector.length === 0) {
      return 0;
    }
    const sum = outputVector.reduce(
      (accumulated, value) => accumulated + Math.abs(value),
      0,
    );
    return sum / outputVector.length;
  });
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
function resolveBehavioralComplexity(outputs: number[][]): number {
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

  return scoreMean + scoreTrend * 0.5 + complexityBonus;
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
    (entry.physicsReward ?? 0) * 0.3
  );
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

/**
 * Map lifecycle apply outcomes to runtime adaptation operations.
 *
 * @param outcomes - Apply outcomes from the lifecycle result.
 * @returns Runtime operations for telemetry, excluding skipped morphs.
 */
function mapOutcomesToOperations(
  outcomes: readonly MorphApplyOutcome[],
): RuntimeAdaptationOperation[] {
  const operations: RuntimeAdaptationOperation[] = [];
  for (const outcome of outcomes) {
    if (outcome.status !== 'applied') continue;
    if (outcome.kind === 'edgeDensify') operations.push('add_edge');
    else if (outcome.kind === 'nodeAdd') operations.push('add_node');
    else if (outcome.kind === 'edgePrune' || outcome.kind === 'compact')
      operations.push('prune_edge');
  }
  return operations;
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

/**
 * Compute whether the growth lifecycle should be throttled for the current tick.
 *
 * When the network exceeds {@link LARGE_NETWORK_NODE_THRESHOLD}, the effective
 * throttle interval scales with network size so that larger networks get
 * progressively longer back-off intervals. This preserves real-time
 * performance by preventing the lifecycle from running every tick at scale.
 *
 * @param network - Live controller network whose size determines throttling.
 * @param tick - Current fixed-timestep tick used for interval gating.
 * @returns Throttle decision with the computed interval.
 */
function computeGrowthThrottle(
  network: Network,
  tick: number,
): { shouldThrottle: boolean; interval: number } {
  const nodeCount = network.nodes.length;
  if (nodeCount <= LARGE_NETWORK_NODE_THRESHOLD) {
    return { shouldThrottle: false, interval: 1 };
  }

  // Scale the throttle interval based on network size budget.
  const sizeBudget = Math.ceil(nodeCount / LARGE_NETWORK_NODE_THRESHOLD);
  const interval = GROWTH_THROTTLE_BASE_INTERVAL_TICKS * sizeBudget;
  const shouldThrottle = tick % interval !== 0;

  return { shouldThrottle, interval };
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

/**
 * Determine whether the quality score has plateaued based on a rolling
 * window of recent baseline scores.
 *
 * Before the first structural growth, the function always returns `true` to
 * allow initial network development without waiting for a full score window.
 * After the first growth, the network is considered plateaued when the
 * rolling window is full and its variance falls below
 * {@link PLATEAU_VARIANCE_THRESHOLD}, indicating that learning has stabilized
 * and further structural growth is safe.
 *
 * Time-boxed stabilization: a minimum of {@link MIN_STABILIZATION_TICKS}
 * ticks must elapse before plateau can fire (preventing premature growth),
 * and a maximum of {@link MAX_STABILIZATION_TICKS} ticks forces growth
 * re-entry even if the variance remains above threshold.
 *
 * @param scoreWindow - Rolling window of recent baseline quality scores.
 * @param hasGrownBefore - Whether the network has already undergone at least
 *   one structural growth phase.
 * @param stabilizationTicksSinceGrowth - Ticks elapsed in the stabilization
 *   phase since the last structural growth.
 * @returns `true` when growth should proceed (first growth, stabilized
 *   plateau, or time-box cap exceeded), `false` when the network is still
 *   stabilizing after growth.
 */
function isPlateauReached(
  scoreWindow: readonly number[],
  hasGrownBefore: boolean,
  stabilizationTicksSinceGrowth: number,
): boolean {
  if (!hasGrownBefore) {
    return true;
  }

  // Time-box cap: after 25 stabilization ticks, force growth re-entry
  // even if the score has not plateaued. This prevents indefinite
  // stabilization when the quality signal remains noisy.
  if (stabilizationTicksSinceGrowth >= MAX_STABILIZATION_TICKS) {
    return true;
  }

  // Minimum guard: require at least 5 stabilization ticks before plateau
  // can fire. This gives the network time to learn its new structure
  // before allowing further structural growth.
  if (stabilizationTicksSinceGrowth < MIN_STABILIZATION_TICKS) {
    return false;
  }

  if (scoreWindow.length < PLATEAU_WINDOW_SIZE) {
    return false;
  }

  const mean =
    scoreWindow.reduce(
      (accumulatedScore, scoreValue) => accumulatedScore + scoreValue,
      0,
    ) / scoreWindow.length;

  let variance = 0;
  for (const score of scoreWindow) {
    const deviation = score - mean;
    variance += deviation * deviation;
  }
  variance /= scoreWindow.length;

  return variance < PLATEAU_VARIANCE_THRESHOLD;
}

/**
 * Apply random weight perturbations to existing connections.
 *
 * Each connection is independently selected for mutation with probability
 * {@link WEIGHT_MUTATION_RATE}. Selected connections have their weight
 * perturbed by a random amount in the range
 * [-{@link WEIGHT_MUTATION_MAGNITUDE}, +{@link WEIGHT_MUTATION_MAGNITUDE}].
 * This helps the network learn to use its current structure during the
 * stabilization phase between structural growth phases.
 *
 * @param network - The network whose connections to perturb.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The number of connections that were mutated.
 */
function applyWeightMutations(network: Network, random: () => number): number {
  let mutatedCount = 0;
  for (const connection of network.connections) {
    if (random() < WEIGHT_MUTATION_RATE) {
      const delta = (random() * 2 - 1) * WEIGHT_MUTATION_MAGNITUDE;
      connection.weight += delta;
      mutatedCount++;
    }
  }
  return mutatedCount;
}
