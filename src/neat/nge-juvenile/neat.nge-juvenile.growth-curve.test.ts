/**
 * Red-test contract for Phase 4 Step 03: continuous growth toward 8,000 nodes.
 *
 * Encodes the observable contracts that the NGE juvenile growth pipeline must
 * satisfy before the growth-gap is considered fixed. These tests intentionally
 * fail against the current implementation:
 *
 * 1. ADD_CONN failures are reported as `skipped`, not falsely `applied`.
 * 2. A monotonic reward stream drives the network past the historical stall
 *    point (101 nodes / 388 edges).
 * 3. The same seed + the same experience stream reproducibly yields identical
 *    topology.
 *
 * No production code is modified here. Step 04 will make these tests green.
 */

import Network from '../../architecture/network';
import { advanceGrowthHysteresis } from './neat.nge-juvenile';
import { applyMorphDeltas } from './neat.nge-juvenile.apply';
import { runNgeLifecycle } from '../neat.nge-lifecycle';
import { resolveFocusConfig } from './neat.nge-juvenile.focus';
import type { MorphApplyOutcome } from './neat.nge-juvenile.apply';
import type {
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
} from './neat.nge-juvenile.types';

const DIAGNOSTIC_MODULE_ID = 'module:diagnostic';

const HISTORICAL_STALL_NODES = 101;
const HISTORICAL_STALL_EDGES = 388;
const MONOTONIC_WINDOW_COUNT = 80;
const DEFAULT_WINDOW_COUNT = 40;
const DEFAULT_SEED = 42;

interface GrowthTelemetry {
  windowIndex: number;
  rewardDelta: number;
  utilization: number;
  focusRawScore: number;
  focusNormalizedScore: number;
  deltasPlanned: string[];
  outcomes: MorphApplyOutcome[];
  nodesBefore: number;
  edgesBefore: number;
  nodesAfter: number;
  edgesAfter: number;
  appliedOutcomeKinds: string[];
  skippedOutcomeKinds: string[];
}

interface StreamConfig {
  name: string;
  buildMetrics: (
    windowIndex: number,
    network: Network,
  ) => NgeModuleMetricsSnapshot;
  windowCount: number;
}

function buildDefaultBudget(network: Network): NgeGrowthBudget {
  return {
    maxNodes: 8_000,
    maxEdges: 32_000,
    maxEpisodicSlots: 0,
    currentNodeCount: network.nodes.length,
    currentEdgeCount: network.connections.length,
    currentEpisodicSlotCount: 0,
  };
}

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

function runGrowthStream(
  network: Network,
  streamConfig: StreamConfig,
): GrowthTelemetry[] {
  const resolvedConfig = resolveFocusConfig({
    hysteresisWindowCount: 1,
    cooldownWindowCount: 0,
    focusWeights: {
      w_u: 0.25,
      w_r: 0.3,
      w_n: 0.2,
      w_s: 0.15,
      w_c: 0.1,
    },
  });

  let hysteresis: NgeHysteresisState = {
    growthPositiveWindowCount: 0,
    pruneUnderuseWindowCount: 0,
    lastMorphKind: 'none',
    cooldownWindowsRemaining: 0,
  };

  const telemetry: GrowthTelemetry[] = [];

  for (
    let windowIndex = 0;
    windowIndex < streamConfig.windowCount;
    windowIndex++
  ) {
    const metrics = streamConfig.buildMetrics(windowIndex, network);
    const isPositiveFocusWindow = metrics.rewardDelta > 0;
    hysteresis = advanceGrowthHysteresis(hysteresis, isPositiveFocusWindow);

    const nodesBefore = network.nodes.length;
    const edgesBefore = network.connections.length;

    const result = runNgeLifecycle({
      stage: 'juvenile',
      moduleId: DIAGNOSTIC_MODULE_ID,
      metrics,
      budget: buildDefaultBudget(network),
      config: resolvedConfig,
      hysteresis,
      network,
      pruneBudget: buildDefaultPruneBudget(network),
    });

    hysteresis = result.hysteresis ?? hysteresis;

    const nodesAfter = network.nodes.length;
    const edgesAfter = network.connections.length;

    const plannedKinds =
      result.juvenileResult?.deltas.map((delta) => delta.kind) ?? [];
    const outcomes = result.applyOutcomes ?? [];

    telemetry.push({
      windowIndex,
      rewardDelta: metrics.rewardDelta,
      utilization: metrics.utilization,
      focusRawScore: result.juvenileResult?.focusScore.rawScore ?? 0,
      focusNormalizedScore:
        result.juvenileResult?.focusScore.normalizedScore ?? 0,
      deltasPlanned: plannedKinds,
      outcomes,
      nodesBefore,
      edgesBefore,
      nodesAfter,
      edgesAfter,
      appliedOutcomeKinds: outcomes
        .filter((outcome) => outcome.status === 'applied')
        .map((outcome) => outcome.kind),
      skippedOutcomeKinds: outcomes
        .filter((outcome) => outcome.status === 'skipped')
        .map((outcome) => outcome.kind),
    });
  }

  return telemetry;
}

function summarizeTelemetry(telemetry: GrowthTelemetry[]): {
  finalNodes: number;
  finalEdges: number;
  totalAppliedReports: number;
  totalSkippedReports: number;
  actualNodeGrowth: number;
  actualEdgeGrowth: number;
  falsePositiveEdgeWindows: number;
} {
  const final = telemetry.at(-1);
  let totalAppliedReports = 0;
  let totalSkippedReports = 0;
  let falsePositiveEdgeWindows = 0;

  for (const window of telemetry) {
    for (const outcome of window.outcomes) {
      if (outcome.status === 'applied') {
        totalAppliedReports += 1;
      } else {
        totalSkippedReports += 1;
      }
    }

    const edgeApplied = window.outcomes.some(
      (outcome) =>
        outcome.kind === 'edgeDensify' && outcome.status === 'applied',
    );
    if (edgeApplied && window.edgesAfter === window.edgesBefore) {
      falsePositiveEdgeWindows += 1;
    }
  }

  return {
    finalNodes: final?.nodesAfter ?? 0,
    finalEdges: final?.edgesAfter ?? 0,
    totalAppliedReports,
    totalSkippedReports,
    actualNodeGrowth:
      (final?.nodesAfter ?? 0) - (telemetry[0]?.nodesBefore ?? 0),
    actualEdgeGrowth:
      (final?.edgesAfter ?? 0) - (telemetry[0]?.edgesBefore ?? 0),
    falsePositiveEdgeWindows,
  };
}

function makeMonotonicStream(): StreamConfig {
  return {
    name: 'monotonic',
    windowCount: MONOTONIC_WINDOW_COUNT,
    buildMetrics: (windowIndex, network) => ({
      moduleId: DIAGNOSTIC_MODULE_ID,
      utilization: 0.9,
      rewardDelta: 0.1 + windowIndex * 0.01,
      novelty: 0.5,
      stabilityAge: windowIndex,
      wiringCost: network.nodes.length + network.connections.length,
    }),
  };
}

function makeFlatStream(): StreamConfig {
  return {
    name: 'flat',
    windowCount: DEFAULT_WINDOW_COUNT,
    buildMetrics: (_, network) => ({
      moduleId: DIAGNOSTIC_MODULE_ID,
      utilization: 0.9,
      rewardDelta: 0,
      novelty: 0.5,
      stabilityAge: 1,
      wiringCost: network.nodes.length + network.connections.length,
    }),
  };
}

function makeNoisyStream(): StreamConfig {
  return {
    name: 'noisy',
    windowCount: DEFAULT_WINDOW_COUNT,
    buildMetrics: (windowIndex, network) => ({
      moduleId: DIAGNOSTIC_MODULE_ID,
      utilization: 0.9,
      rewardDelta: 0.05 + Math.sin(windowIndex) * 0.03,
      novelty: 0.5,
      stabilityAge: windowIndex,
      wiringCost: network.nodes.length + network.connections.length,
    }),
  };
}

/**
 * Capture a clone-safe, implementation-agnostic fingerprint of network topology.
 *
 * This compares node count, edge count, node roles, and connection innovation
 * IDs rather than object identity or internal state, so the contract stays at
 * the observable phenotype level.
 */
function topologyFingerprint(network: Network): Record<string, unknown> {
  return {
    nodeCount: network.nodes.length,
    edgeCount: network.connections.length,
    nodeTypes: network.nodes.map((node) => node.type).sort(),
    connectionInnovations: network.connections
      .map((conn) => String(conn.innovation))
      .sort(),
  };
}

describe('nge juvenile growth-curve red contracts', () => {
  describe('applyMorphDeltas edgeDensify truthfulness', () => {
    it('reports ADD_CONN failure as skipped when no edge is added', () => {
      // A 2x1 feedforward network is already saturated for forward ADD_CONN,
      // so the operator cannot produce a structural change. The contract is
      // that the applier must report this as skipped, not as applied.
      const network = new Network(2, 1, { seed: DEFAULT_SEED });
      const initialConnectionCount = network.connections.length;
      const initialNodeCount = network.nodes.length;

      const deltas = [
        {
          kind: 'edgeDensify' as const,
          targetModuleId: DIAGNOSTIC_MODULE_ID,
          detail: {
            currentEdgeCount: initialConnectionCount,
            proposedAdditions: 1,
            normalizedFocusScore: 0.9,
          },
          wiringCostDelta: 1,
        },
      ];

      const outcomes = applyMorphDeltas(network, deltas, {
        growth: buildDefaultBudget(network),
        prune: buildDefaultPruneBudget(network),
      });

      const edgeOutcome = outcomes.find(
        (outcome) => outcome.kind === 'edgeDensify',
      );
      const structuralChange = {
        nodes: network.nodes.length - initialNodeCount,
        edges: network.connections.length - initialConnectionCount,
        reportedStatus: edgeOutcome?.status,
      };

      expect(structuralChange).toEqual({
        nodes: 0,
        edges: 0,
        reportedStatus: 'skipped',
      });
    });
  });

  describe('runNgeLifecycle monotonic reward stream', () => {
    let monotonicTelemetry: GrowthTelemetry[];
    let monotonicSummary: ReturnType<typeof summarizeTelemetry>;

    beforeAll(() => {
      const network = new Network(4, 2, { seed: DEFAULT_SEED });
      monotonicTelemetry = runGrowthStream(network, makeMonotonicStream());
      monotonicSummary = summarizeTelemetry(monotonicTelemetry);
    });

    it('grows past the historical 101-node stall point', () => {
      expect(monotonicSummary.finalNodes).toBeGreaterThan(
        HISTORICAL_STALL_NODES,
      );
    });

    it('grows past the historical 388-edge stall point', () => {
      expect(monotonicSummary.finalEdges).toBeGreaterThan(
        HISTORICAL_STALL_EDGES,
      );
    });

    it('does not report edge-densify windows with no structural change', () => {
      expect(monotonicSummary.falsePositiveEdgeWindows).toBe(0);
    });
  });

  describe('runNgeLifecycle determinism', () => {
    it('reproduces identical topology from the same seed and experience stream', () => {
      const stream = makeMonotonicStream();

      const networkA = new Network(4, 2, { seed: DEFAULT_SEED });
      runGrowthStream(networkA, stream);

      const networkB = new Network(4, 2, { seed: DEFAULT_SEED });
      runGrowthStream(networkB, stream);

      expect(topologyFingerprint(networkA)).toEqual(
        topologyFingerprint(networkB),
      );
    });
  });

  describe('runNgeLifecycle baseline telemetry streams', () => {
    it('records growth telemetry for a flat reward stream', () => {
      const network = new Network(4, 2, { seed: DEFAULT_SEED });
      const stream = makeFlatStream();
      const telemetry = runGrowthStream(network, stream);
      const summary = summarizeTelemetry(telemetry);

      expect(summary.finalNodes).toBeGreaterThanOrEqual(0);
    });

    it('records growth telemetry for a noisy reward stream', () => {
      const network = new Network(4, 2, { seed: DEFAULT_SEED });
      const stream = makeNoisyStream();
      const telemetry = runGrowthStream(network, stream);
      const summary = summarizeTelemetry(telemetry);

      expect(summary.finalEdges).toBeGreaterThanOrEqual(0);
    });
  });
});
