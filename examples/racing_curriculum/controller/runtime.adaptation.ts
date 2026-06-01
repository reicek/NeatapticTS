import { Network, methods } from '../../../src/browser-entry.ts';

/** Cadence modes supported by the runtime adaptation engine. */
export type RuntimeAdaptationCadenceMode =
  | 'every_tick'
  | 'every_n_ticks'
  | 'lap_boundary'
  | 'sector_boundary';

/** Candidate mutation operations supported by the runtime adaptation engine. */
export type RuntimeAdaptationOperation =
  | 'param_nudge'
  | 'add_edge'
  | 'add_node'
  | 'prune_edge';

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
  adaptOnTick(tickInput: RuntimeAdaptationTickInput): RuntimeAdaptationTelemetry;
  /** Resets cadence boundaries and cooldown state. */
  reset(): void;
}

const DEFAULT_CADENCE: RuntimeAdaptationCadenceOptions = {
  mode: 'every_tick',
};

const DEFAULT_LIMITS: RuntimeAdaptationLimits = {
  maxStructuralEditsPerStep: 1,
  maxNodes: 256,
  maxConnections: 1_024,
  mutationCooldownTicks: 0,
  rollbackCooldownTicks: 0,
};

const DEFAULT_IMPROVEMENT_THRESHOLD = 0;
const DEFAULT_MINIMUM_EVIDENCE_WINDOW = 4;

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
    Math.floor(options.minimumEvidenceWindow ?? DEFAULT_MINIMUM_EVIDENCE_WINDOW),
  );
  const improvementThreshold =
    options.improvementThreshold ?? DEFAULT_IMPROVEMENT_THRESHOLD;
  const random = options.random ?? Math.random;

  let nextMutationTick = Number.NEGATIVE_INFINITY;
  let nextRollbackTick = Number.NEGATIVE_INFINITY;
  let lastLapBoundary = Number.NEGATIVE_INFINITY;
  let lastSectorBoundary = Number.NEGATIVE_INFINITY;

  return {
    adaptOnTick(tickInput: RuntimeAdaptationTickInput): RuntimeAdaptationTelemetry {
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

      if (tickInput.tick < nextMutationTick) {
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

      // Step 3: Propose bounded candidate operations and snapshot rollback state.
      const proposedOperations = proposeCandidateOperations(
        tickInput.network,
        limits,
        random,
      );
      if (proposedOperations.length === 0) {
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

      const rollbackSnapshot = tickInput.network.toJSON();

      // Step 4: Apply candidate edits, evaluate, and commit/rollback by threshold + safety.
      applyOperations(tickInput.network, proposedOperations);
      const rawCandidateSize = resolveNetworkSizeSnapshot(tickInput.network);
      const safetyChecksPass = passesSafetyChecks(rawCandidateSize, limits);
      const candidateScore = evaluateScore(tickInput.network, evidenceWindow);
      const improvement = candidateScore - baselineScore;
      const shouldCommit =
        safetyChecksPass && improvement >= improvementThreshold;

      if (shouldCommit) {
        nextMutationTick = tickInput.tick + limits.mutationCooldownTicks;
        return createTelemetry({
          tick: tickInput.tick,
          operations: proposedOperations,
          scoreBefore: baselineScore,
          scoreAfter: candidateScore,
          committed: true,
          reason: 'committed',
          networkSizeBefore,
          networkSizeAfter: rawCandidateSize,
        });
      }

      restoreNetworkSnapshot(tickInput.network, rollbackSnapshot);
      nextRollbackTick = tickInput.tick + limits.rollbackCooldownTicks;
      const restoredSize = resolveNetworkSizeSnapshot(tickInput.network);

      return createTelemetry({
        tick: tickInput.tick,
        operations: proposedOperations,
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
      nextMutationTick = Number.NEGATIVE_INFINITY;
      nextRollbackTick = Number.NEGATIVE_INFINITY;
      lastLapBoundary = Number.NEGATIVE_INFINITY;
      lastSectorBoundary = Number.NEGATIVE_INFINITY;
    },
  };
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
  const boundaryInterval = Math.max(1, Math.floor(cadence.boundaryInterval ?? 1));

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
    maxNodes: Math.max(1, Math.floor(limits?.maxNodes ?? DEFAULT_LIMITS.maxNodes)),
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

function proposeCandidateOperations(
  network: Network,
  limits: RuntimeAdaptationLimits,
  random: () => number,
): RuntimeAdaptationOperation[] {
  const operationPlan: RuntimeAdaptationOperation[] = ['param_nudge'];
  const structuralPool = resolveStructuralPool(network, limits);

  if (limits.maxStructuralEditsPerStep === 0 || structuralPool.length === 0) {
    return operationPlan;
  }

  for (
    let structuralEditIndex = 0;
    structuralEditIndex < limits.maxStructuralEditsPerStep;
    structuralEditIndex++
  ) {
    const sampledIndex = Math.floor(random() * structuralPool.length);
    operationPlan.push(structuralPool[sampledIndex] ?? structuralPool[0]);
  }

  return operationPlan;
}

function resolveStructuralPool(
  network: Network,
  limits: RuntimeAdaptationLimits,
): RuntimeAdaptationOperation[] {
  const structuralPool: RuntimeAdaptationOperation[] = [];

  if (network.connections.length > 0) {
    structuralPool.push('prune_edge');
  }

  if (network.connections.length < limits.maxConnections) {
    structuralPool.push('add_edge');
  }

  if (network.nodes.length < limits.maxNodes) {
    structuralPool.push('add_node');
  }

  return structuralPool;
}

function applyOperations(
  network: Network,
  operations: readonly RuntimeAdaptationOperation[],
): void {
  operations.forEach((operation) => {
    if (operation === 'param_nudge') {
      network.mutate(methods.mutation.MOD_WEIGHT);
      return;
    }

    if (operation === 'add_edge') {
      network.mutate(methods.mutation.ADD_CONN);
      return;
    }

    if (operation === 'add_node') {
      network.mutate(methods.mutation.ADD_NODE);
      return;
    }

    network.mutate(methods.mutation.SUB_CONN);
  });
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

function resolveNetworkSizeSnapshot(network: Network): RuntimeNetworkSizeSnapshot {
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
