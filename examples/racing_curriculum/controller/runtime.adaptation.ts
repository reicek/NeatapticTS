import { Network } from '../../../src/browser-entry.ts';
import { runNgeLifecycle } from '../../../src/neat/neat.nge-lifecycle';
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
    | 'committed';
  /** Network size before candidate mutations were applied. */
  readonly networkSizeBefore: RuntimeNetworkSizeSnapshot;
  /** Network size after commit/rollback resolution. */
  readonly networkSizeAfter: RuntimeNetworkSizeSnapshot;
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
    scoreHistory: readonly number[],
  ) => number;
}

/** Per-tick input contract for adaptation checks. */
export interface RuntimeAdaptationTickInput {
  /** Current fixed-timestep tick. */
  readonly tick: number;
  /** Live mutable controller network. */
  readonly network: Network;
  /** Rolling score history assembled by the caller. */
  readonly scoreHistory: readonly number[];
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

const DEFAULT_CADENCE: RuntimeAdaptationCadenceOptions = {
  mode: 'every_tick',
};

const DEFAULT_LIMITS: RuntimeAdaptationLimits = {
  maxStructuralEditsPerStep: 1,
  maxNodes: 8_000,
  maxConnections: 32_000,
  mutationCooldownTicks: 0,
  rollbackCooldownTicks: 0,
};

const DEFAULT_IMPROVEMENT_THRESHOLD = 0;
const DEFAULT_MINIMUM_EVIDENCE_WINDOW = 4;
const RUNTIME_MODULE_ID = 'racing:runtime';

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
  const evaluateScore = options.evaluateScore ?? evaluateRollingScoreWindow;
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
        return createTelemetry({
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'cadence_not_reached',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        });
      }

      // Step 2: Guard against insufficient evidence and active cooldown windows.
      if (evidenceWindow.length < minimumEvidenceWindow) {
        return createTelemetry({
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'insufficient_evidence',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        });
      }

      // Step 3: Advance lifecycle hysteresis for the current evaluation window.
      const isPositiveFocusWindow =
        evidenceWindow.length >= 2 &&
        (evidenceWindow.at(-1) ?? 0) > (evidenceWindow[0] ?? 0);
      hysteresis = advanceGrowthHysteresis(hysteresis, isPositiveFocusWindow);

      if (hysteresis.cooldownWindowsRemaining > 0) {
        return createTelemetry({
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'mutation_cooldown_active',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        });
      }

      if (tickInput.tick < nextRollbackTick) {
        return createTelemetry({
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'rollback_cooldown_active',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        });
      }

      // Step 3.5: Size-based growth throttle — back off lifecycle runs when
      // the network has grown large to preserve real-time performance.
      const growthThrottle = computeGrowthThrottle(
        tickInput.network,
        tickInput.tick,
      );
      if (growthThrottle.shouldThrottle) {
        return createTelemetry({
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'growth_throttled',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        });
      }

      // Step 4: Build lifecycle inputs from the runtime state.
      const metrics = buildModuleMetricsSnapshot(
        tickInput.network,
        evidenceWindow,
      );
      const growthBudget = buildGrowthBudget(tickInput.network, limits);
      const pruneBudget = buildPruneBudget(tickInput.network);

      // Step 5: Snapshot the network for potential rollback.
      const rollbackSnapshot = tickInput.network.toJSON();

      // Step 6: Call runNgeLifecycle to plan and apply growth morphs.
      const lifecycleResult = runNgeLifecycle({
        stage: 'juvenile',
        moduleId: RUNTIME_MODULE_ID,
        metrics,
        budget: growthBudget,
        config: {
          hysteresisWindowCount: 0,
          cooldownWindowCount: limits.mutationCooldownTicks,
        },
        hysteresis,
        network: tickInput.network,
        pruneBudget,
      });

      // Step 7: Check applyOutcomes for applied morphs.
      const applyOutcomes = lifecycleResult.applyOutcomes ?? [];
      const operations = mapOutcomesToOperations(applyOutcomes);

      if (operations.length === 0) {
        return createTelemetry({
          tick: tickInput.tick,
          operations: [],
          scoreBefore: baselineScore,
          scoreAfter: baselineScore,
          committed: false,
          reason: 'no_candidate_operations',
          networkSizeBefore,
          networkSizeAfter: networkSizeBefore,
        });
      }

      // Step 8: Evaluate candidate score after lifecycle mutation.
      const rawCandidateSize = resolveNetworkSizeSnapshot(tickInput.network);
      const safetyChecksPass = passesSafetyChecks(rawCandidateSize, limits);
      const candidateScore = evaluateScore(tickInput.network, evidenceWindow);
      const improvement = candidateScore - baselineScore;
      const shouldCommit =
        safetyChecksPass && improvement >= improvementThreshold;

      // Step 9: Commit or rollback based on score improvement.
      if (shouldCommit) {
        hysteresis = lifecycleResult.hysteresis ?? hysteresis;
        return createTelemetry({
          tick: tickInput.tick,
          operations,
          scoreBefore: baselineScore,
          scoreAfter: candidateScore,
          committed: true,
          reason: 'committed',
          networkSizeBefore,
          networkSizeAfter: rawCandidateSize,
        });
      }

      // Step 10: Rollback the network and set rollback cooldown.
      restoreNetworkSnapshot(tickInput.network, rollbackSnapshot);
      nextRollbackTick = tickInput.tick + limits.rollbackCooldownTicks;
      const restoredSize = resolveNetworkSizeSnapshot(tickInput.network);

      return createTelemetry({
        tick: tickInput.tick,
        operations,
        scoreBefore: baselineScore,
        scoreAfter: candidateScore,
        committed: false,
        reason: safetyChecksPass
          ? 'improvement_below_threshold'
          : 'safety_checks_failed',
        networkSizeBefore,
        networkSizeAfter: restoredSize,
      });
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
  scoreHistory: readonly number[],
): readonly number[] {
  return scoreHistory.filter((scoreValue) => Number.isFinite(scoreValue));
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
  evidenceWindow: readonly number[],
): NgeModuleMetricsSnapshot {
  const scoreTrend =
    evidenceWindow.length >= 2
      ? (evidenceWindow.at(-1) ?? 0) - (evidenceWindow[0] ?? 0)
      : 0;
  const scoreMean =
    evidenceWindow.length > 0
      ? evidenceWindow.reduce(
          (accumulatedScore, scoreValue) => accumulatedScore + scoreValue,
          0,
        ) / evidenceWindow.length
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
    maxEpisodicSlots: 0,
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

function restoreNetworkSnapshot(
  network: Network,
  rollbackSnapshot: Record<string, unknown>,
): void {
  const restoredNetwork = Network.fromJSON(rollbackSnapshot);
  const mutableTarget = network as Record<string, unknown>;
  const mutableSource = restoredNetwork as unknown as Record<string, unknown>;

  Object.keys(mutableTarget).forEach((propertyName) => {
    delete mutableTarget[propertyName];
  });
  Object.assign(mutableTarget, mutableSource);
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
  telemetry: RuntimeAdaptationTelemetry,
): RuntimeAdaptationTelemetry {
  return telemetry;
}
