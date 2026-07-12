/**
 * NGE grow-stabilize cycle.
 *
 * This module owns the core grow-stabilize adaptation cycle extracted from the
 * racing curriculum's runtime adaptation engine. It provides pure decision
 * functions (plateau detection, adaptive hysteresis, weight mutations, growth
 * throttle) and a single orchestrator (`runNgeGrowStabilizeCycle`) that
 * sequences one adaptation tick.
 *
 * The orchestrator accepts plain numeric score history and a live mutable
 * network, keeping the core free of demo-specific types (e.g.
 * `RacingQualitySignal`). App layers convert composite signals to scalar
 * numbers before calling the core cycle.
 *
 * ## Determinism note
 *
 * When a deterministic `random` source is supplied, weight mutation selection
 * is reproducible. When a `lifecycleRunner` is injected, the caller controls
 * the lifecycle execution, enabling test doubles and cycle breaking.
 *
 * ## Background reading
 *
 * - Hysteresis in control systems:
 *   [Wikipedia — Hysteresis](https://en.wikipedia.org/wiki/Hysteresis).
 * - Plateau detection via rolling-window variance:
 *   [Wikipedia — Variance](https://en.wikipedia.org/wiki/Variance).
 *
 * ```mermaid
 * stateDiagram-v2
 *   [*] --> PlateauCheck
 *   PlateauCheck --> Stabilization : not plateaued
 *   PlateauCheck --> Growth : plateaued or first growth
 *   Stabilization --> [*] : weight mutations applied
 *   Growth --> [*] : lifecycle morphs applied
 * ```
 */

import type Network from '../../architecture/network';
import { runNgeLifecycle } from '../neat.nge-lifecycle';
import {
  NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
  NGE_GROW_STABILIZE_DEFAULT_MODULE_ID,
  NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS,
  NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD,
  NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
  NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS,
  NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS,
  NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD,
  NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE,
  NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE,
  NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE,
} from './neat.nge-juvenile.constants';
import type {
  NgeGrowthBudget,
  NgeGrowStabilizeConfig,
  NgeGrowStabilizeInput,
  NgeGrowStabilizeResult,
  NgeHysteresisState,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
} from './neat.nge-juvenile.types';

// ──────────────────────────────────────────────────────────────────────
// Pure decision functions
// ──────────────────────────────────────────────────────────────────────

/**
 * Resolve the adaptive hysteresis window count based on the live network
 * node count. Smaller networks use a lower threshold (2 consecutive
 * positive-quality windows) to accelerate early growth, while larger
 * networks require more sustained evidence (5 windows) before committing
 * to further structural expansion.
 *
 * @param nodeCount - Current total node count in the live network.
 * @returns Hysteresis window count: 2 for ≤ 200 nodes, 3 for ≤ 500, 5 for > 500.
 *
 * @example
 * ```ts
 * const hysteresis = resolveAdaptiveHysteresis(150);
 * console.log(hysteresis); // 2
 * ```
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
 * Determine whether the quality score has plateaued based on a rolling
 * window of recent baseline scores.
 *
 * Before the first structural growth, the function always returns `true` to
 * allow initial network development without waiting for a full score window.
 * After the first growth, the network is considered plateaued when the
 * rolling window is full and its variance falls below the threshold.
 *
 * Time-boxed stabilization: a minimum number of ticks must elapse before
 * plateau can fire (preventing premature growth), and a maximum number of
 * ticks forces growth re-entry even if the variance remains above threshold.
 *
 * @param scoreWindow - Rolling window of recent baseline quality scores.
 * @param hasGrownBefore - Whether the network has already undergone at least
 *   one structural growth phase.
 * @param stabilizationTicksSinceGrowth - Ticks elapsed in the stabilization
 *   phase since the last structural growth.
 * @returns `true` when growth should proceed (first growth, stabilized
 *   plateau, or time-box cap exceeded), `false` when the network is still
 *   stabilizing after growth.
 *
 * @example
 * ```ts
 * const plateaued = isPlateauReached([0.5, 0.51, 0.49, 0.5, 0.5], true, 10);
 * console.log(plateaued); // true (low variance after min ticks)
 * ```
 */
export function isPlateauReached(
  scoreWindow: readonly number[],
  hasGrownBefore: boolean,
  stabilizationTicksSinceGrowth: number,
): boolean {
  if (!hasGrownBefore) {
    return true;
  }

  // Time-box cap: after max stabilization ticks (25), force growth re-entry
  // even if the score has not plateaued. This prevents indefinite
  // stabilization when the quality signal remains noisy.
  if (
    stabilizationTicksSinceGrowth >= NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS
  ) {
    return true;
  }

  // Minimum guard: require at least min stabilization ticks before plateau
  // can fire. This gives the network time to learn its new structure
  // before allowing further structural growth.
  if (
    stabilizationTicksSinceGrowth < NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS
  ) {
    return false;
  }

  if (scoreWindow.length < NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE) {
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

  return variance < NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD;
}

/**
 * Apply random weight perturbations to existing connections.
 *
 * Each connection is independently selected for mutation with probability
 * equal to the weight mutation rate. Selected connections have their weight
 * perturbed by a random amount in the range
 * [-magnitude, +magnitude]. This helps the network learn to use its current
 * structure during the stabilization phase between structural growth phases.
 *
 * @param network - The network whose connections to perturb.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The number of connections that were mutated.
 *
 * @example
 * ```ts
 * const mutated = applyWeightMutations(network, Math.random);
 * console.log(mutated); // e.g. 3
 * ```
 */
export function applyWeightMutations(
  network: Network,
  random: () => number,
): number {
  let mutatedCount = 0;
  for (const connection of network.connections) {
    if (random() < NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE) {
      const delta =
        (random() * 2 - 1) * NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE;
      connection.weight += delta;
      mutatedCount++;
    }
  }
  return mutatedCount;
}

/**
 * Compute whether the growth lifecycle should be throttled for the current tick.
 *
 * When the network exceeds the large-network node threshold, the effective
 * throttle interval scales with network size so that larger networks get
 * progressively longer back-off intervals. This preserves real-time
 * performance by preventing the lifecycle from running every tick at scale.
 *
 * @param network - Live controller network whose size determines throttling.
 * @param tick - Current fixed-timestep tick used for interval gating.
 * @returns Throttle decision with the computed interval.
 *
 * @example
 * ```ts
 * const { shouldThrottle } = computeGrowthThrottle(network, 42);
 * console.log(shouldThrottle); // false for small networks
 * ```
 */
export function computeGrowthThrottle(
  network: Network,
  tick: number,
): { shouldThrottle: boolean; interval: number } {
  const nodeCount = network.nodes.length;
  if (nodeCount <= NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD) {
    return { shouldThrottle: false, interval: 1 };
  }

  // Scale the throttle interval based on network size budget.
  const sizeBudget = Math.ceil(
    nodeCount / NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD,
  );
  const interval =
    NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS * sizeBudget;
  const shouldThrottle = tick % interval !== 0;

  return { shouldThrottle, interval };
}

// ──────────────────────────────────────────────────────────────────────
// Orchestrator
// ──────────────────────────────────────────────────────────────────────

/**
 * Run one NGE grow-stabilize adaptation cycle.
 *
 * This orchestrator encapsulates the plateau-detection decision and either:
 *
 * - **Stabilization phase**: applies weight perturbations to existing
 *   connections so the network can learn to use its current structure.
 * - **Growth phase**: builds module metrics, a growth budget, and a prune
 *   budget from the live network state, then delegates to the NGE lifecycle
 *   runner to plan and apply structural morphs.
 *
 * For the very first growth (`hasGrownBefore` is `false`), the plateau check
 * is bypassed and the hysteresis gate is pre-satisfied so the lifecycle
 * produces candidate morphs immediately — the network needs capacity before
 * stabilization can tune it.
 *
 * The caller is responsible for pre-mutation score evaluation, network
 * snapshot/rollback, and post-mutation score evaluation. The cycle only
 * handles the core decision and mutation application; commit/rollback based
 * on score improvement remains the caller's responsibility.
 *
 * @param input - Grow-stabilize cycle input with required network,
 *   scoreHistory, hasGrownBefore, and stabilizationTicksSinceGrowth.
 * @returns Result describing whether the cycle committed, which phase it
 *   entered, and what operations were applied.
 *
 * @example
 * ```ts
 * const result = runNgeGrowStabilizeCycle({
 *   network,
 *   scoreHistory: [1, 2, 3, 4],
 *   hasGrownBefore: false,
 *   stabilizationTicksSinceGrowth: 0,
 * });
 * console.log(result.committed); // true (first growth)
 * ```
 */
export function runNgeGrowStabilizeCycle(
  input: NgeGrowStabilizeInput,
): NgeGrowStabilizeResult {
  const network = input.network;
  const hasGrownBefore = input.hasGrownBefore;
  const stabilizationTicksSinceGrowth = input.stabilizationTicksSinceGrowth;
  const random = input.random ?? Math.random;
  const runner = input.lifecycleRunner ?? runNgeLifecycle;
  const config = resolveGrowStabilizeConfig(input.config);

  // Step 1: Check whether the quality score has plateaued.
  const plateauReached = isPlateauReached(
    input.qualityScoreHistory ?? [],
    hasGrownBefore,
    stabilizationTicksSinceGrowth,
  );

  // Step 2: Stabilization phase — apply weight perturbations.
  if (!plateauReached) {
    const mutatedCount = applyWeightMutations(network, random);
    return {
      committed: mutatedCount > 0,
      phase: 'stabilization',
      reason:
        mutatedCount > 0 ? 'weight_mutation_committed' : 'no_weight_mutations',
      operations: mutatedCount > 0 ? ['param_nudge'] : [],
      stabilizationTicksSinceGrowth: stabilizationTicksSinceGrowth + 1,
      mutatedCount,
      networkSizeAfter: {
        nodes: network.nodes.length,
        connections: network.connections.length,
      },
    };
  }

  // Step 3: Growth phase — build lifecycle inputs and call the lifecycle runner.
  const metrics = buildDefaultMetrics(
    input.scoreHistory,
    network,
    config.moduleId,
  );
  const budget = buildDefaultBudget(network, config);
  const pruneBudget = buildDefaultPruneBudget(network);
  const adaptiveHysteresis = resolveAdaptiveHysteresis(network.nodes.length);
  const isFirstGrowth = !hasGrownBefore;

  // Pre-satisfy the hysteresis gate for first growth so the lifecycle
  // produces candidate morphs immediately without waiting for accumulated
  // positive-quality windows.
  const lifecycleHysteresis: NgeHysteresisState = isFirstGrowth
    ? {
        growthPositiveWindowCount: adaptiveHysteresis,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      }
    : (input.hysteresis ?? {
        growthPositiveWindowCount: 0,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      });

  const lifecycleResult = runner({
    stage: 'juvenile',
    moduleId: config.moduleId,
    metrics,
    budget,
    config: {
      hysteresisWindowCount: adaptiveHysteresis,
      cooldownWindowCount: 5,
      maxStructuralEditsPerStep: config.maxStructuralEditsPerStep,
    },
    hysteresis: lifecycleHysteresis,
    network,
    pruneBudget,
  });

  // Step 4: Map apply outcomes to operations.
  const applyOutcomes = lifecycleResult.applyOutcomes ?? [];
  const operations = mapOutcomesToOperations(applyOutcomes);

  return {
    committed: operations.length > 0,
    phase: 'growth',
    reason: operations.length > 0 ? 'committed' : 'no_candidate_operations',
    operations,
    stabilizationTicksSinceGrowth: 0,
    mutatedCount: 0,
    networkSizeAfter: {
      nodes: network.nodes.length,
      connections: network.connections.length,
    },
    hysteresis: lifecycleResult.hysteresis,
  };
}

// ──────────────────────────────────────────────────────────────────────
// Helpers (below the fold)
// ──────────────────────────────────────────────────────────────────────

/**
 * Resolve a partial grow-stabilize config with sensible defaults.
 *
 * @param partial - Caller-supplied config overrides.
 * @returns Fully resolved config.
 */
function resolveGrowStabilizeConfig(
  partial?: Partial<NgeGrowStabilizeConfig>,
): NgeGrowStabilizeConfig {
  return {
    maxStructuralEditsPerStep:
      partial?.maxStructuralEditsPerStep ??
      NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
    maxNodes: partial?.maxNodes ?? 8_000,
    maxConnections: partial?.maxConnections ?? 32_000,
    maxEpisodicSlots:
      partial?.maxEpisodicSlots ?? NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
    moduleId: partial?.moduleId ?? NGE_GROW_STABILIZE_DEFAULT_MODULE_ID,
  };
}

/**
 * Build default module metrics from numeric score history and live network state.
 *
 * @param scoreHistory - Rolling numeric score history.
 * @param network - Live controller network.
 * @param moduleId - Module identifier for the metrics snapshot.
 * @returns NGE module metrics for the lifecycle focus scorer.
 */
function buildDefaultMetrics(
  scoreHistory: readonly number[],
  network: Network,
  moduleId: string,
): NgeModuleMetricsSnapshot {
  const scoreTrend =
    scoreHistory.length >= 2 ? scoreHistory.at(-1)! - scoreHistory[0]! : 0;
  const scoreMean =
    scoreHistory.length > 0
      ? scoreHistory.reduce(
          (accumulatedScore, scoreValue) => accumulatedScore + scoreValue,
          0,
        ) / scoreHistory.length
      : 0;

  return {
    moduleId,
    utilization: Math.min(scoreMean, 1),
    rewardDelta: scoreTrend,
    novelty: 0,
    stabilityAge: 0,
    wiringCost: network.nodes.length + network.connections.length,
  };
}

/**
 * Build a default growth budget from the live network and resolved config.
 *
 * @param network - Live controller network.
 * @param config - Resolved grow-stabilize config.
 * @returns NGE growth budget for the lifecycle apply phase.
 */
function buildDefaultBudget(
  network: Network,
  config: NgeGrowStabilizeConfig,
): NgeGrowthBudget {
  return {
    maxNodes: config.maxNodes,
    maxEdges: config.maxConnections,
    maxEpisodicSlots: config.maxEpisodicSlots,
    currentNodeCount: network.nodes.length,
    currentEdgeCount: network.connections.length,
    currentEpisodicSlotCount: 0,
  };
}

/**
 * Build a default prune budget from the live network.
 *
 * @param network - Live controller network.
 * @returns NGE prune budget for the lifecycle apply phase.
 */
function buildDefaultPruneBudget(network: Network): NgePruneBudget {
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
 * Map lifecycle apply outcomes to operation name strings.
 *
 * @param outcomes - Apply outcomes from the lifecycle result.
 * @returns Operation strings for telemetry, excluding skipped morphs.
 */
function mapOutcomesToOperations(
  outcomes: readonly { status: string; kind: string }[],
): string[] {
  const operations: string[] = [];
  for (const outcome of outcomes) {
    if (outcome.status !== 'applied') continue;
    if (outcome.kind === 'edgeDensify') operations.push('add_edge');
    else if (outcome.kind === 'nodeAdd') operations.push('add_node');
    else if (outcome.kind === 'edgePrune' || outcome.kind === 'compact')
      operations.push('prune_edge');
  }
  return operations;
}
