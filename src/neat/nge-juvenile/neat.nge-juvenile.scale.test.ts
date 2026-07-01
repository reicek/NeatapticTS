/**
 * Red-test contract for Phase 7 Step 03: seed-driven scale verification.
 *
 * These tests encode the observable contracts the Step 04 implementation must
 * satisfy to prove NGE grows from a seed to 8,000+ neurons and 32,000+ edges
 * under continuous real-time adaptation, and that the same seed + experience
 * stream reproduces the same topology.
 *
 * These contracts capture the Phase 7 verification target. Against the current
 * implementation:
 *
 * 1. The tuned continuous-adaptation config already reaches 8,000+ nodes in 400
 *    windows, so the node-scale contract is green and protects the target
 *    from regression.
 * 2. The current `applyEdgeDensify` calls `ADD_CONN` in a tight loop, so the
 *    edge-densification fast path is still needed to reach 32,000 edges in
 *    bounded time. This is the active red contract.
 * 3. Topology determinism at 8,000-node scale is already reproducible for the
 *    growth-curve fingerprint, so the determinism contract is green and guards
 *    against Step 04 performance regressions.
 *
 * No production code is modified here. Step 04 will make the edge-scale test green.
 */

import Network from '../../architecture/network';
import { runNgeLifecycle } from '../neat.nge-lifecycle';
import { advanceGrowthHysteresis } from './neat.nge-juvenile';
import type { MorphApplyOutcome } from './neat.nge-juvenile.apply';
import {
  NGE_MAX_EDGE_CAPACITY,
  NGE_MAX_NODE_CAPACITY,
} from './neat.nge-juvenile.constants';
import { resolveFocusConfig } from './neat.nge-juvenile.focus';
import type {
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeJuvenilePhaseConfig,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
} from './neat.nge-juvenile.types';

const SCALE_TIMEOUT_MS = 180_000;

jest.setTimeout(SCALE_TIMEOUT_MS);

const DIAGNOSTIC_MODULE_ID = 'module:diagnostic';
const DEFAULT_SEED = 42;

const SCALE_NODE_WINDOW_COUNT = 400;
const SCALE_EDGE_WINDOW_COUNT = 300;
const SCALE_DETERMINISM_WINDOW_COUNT = 300;

interface ScaleTelemetry {
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

interface ScaleStreamConfig {
  name: string;
  windowCount: number;
  buildMetrics: (
    windowIndex: number,
    network: Network,
  ) => NgeModuleMetricsSnapshot;
}

function buildDefaultBudget(network: Network): NgeGrowthBudget {
  return {
    maxNodes: NGE_MAX_NODE_CAPACITY,
    // Keep the edge cap above the 32,000 verification target so the planner
    // does not throw a budget error while the network is approaching the
    // target from below. The assertion still checks the target itself.
    maxEdges: NGE_MAX_EDGE_CAPACITY + 1000,
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
  config: NgeJuvenilePhaseConfig,
  streamConfig: ScaleStreamConfig,
): ScaleTelemetry[] {
  let hysteresis: NgeHysteresisState = {
    growthPositiveWindowCount: 0,
    pruneUnderuseWindowCount: 0,
    lastMorphKind: 'none',
    cooldownWindowsRemaining: 0,
  };

  const telemetry: ScaleTelemetry[] = [];

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
      config,
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

function summarizeTelemetry(telemetry: ScaleTelemetry[]): {
  finalNodes: number;
  finalEdges: number;
  totalAppliedReports: number;
  totalSkippedReports: number;
  actualNodeGrowth: number;
  actualEdgeGrowth: number;
} {
  const final = telemetry.at(-1);
  let totalAppliedReports = 0;
  let totalSkippedReports = 0;

  for (const window of telemetry) {
    for (const outcome of window.outcomes) {
      if (outcome.status === 'applied') {
        totalAppliedReports += 1;
      } else {
        totalSkippedReports += 1;
      }
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
  };
}

function makeMonotonicStream(windowCount: number): ScaleStreamConfig {
  return {
    name: 'monotonic',
    windowCount,
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

/**
 * Capture a clone-safe, implementation-agnostic fingerprint of network topology.
 *
 * Compares node count, edge count, node roles, and connection innovation IDs
 * rather than object identity or internal state, so the contract stays at the
 * observable phenotype level.
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

/**
 * Tuned continuous-adaptation config that the Step 02 research brief showed
 * reaches 8,000+ nodes in ~400 windows when the growth pipeline is enabled.
 */
function buildTunedNodeGrowthConfig(): NgeJuvenilePhaseConfig {
  return resolveFocusConfig({
    hysteresisWindowCount: 1,
    cooldownWindowCount: 0,
    nodeAdditionCount: 20,
    edgeDensificationCount: 0,
  });
}

/**
 * Edge-densification override that requires the Step 04 `applyEdgeDensify`
 * batch fast path to reach 32,000+ edges in bounded time.
 */
function buildTunedEdgeDensifyConfig(): NgeJuvenilePhaseConfig {
  return resolveFocusConfig({
    hysteresisWindowCount: 1,
    cooldownWindowCount: 0,
    nodeAdditionCount: 20,
    edgeDensificationCount: 85,
  });
}

describe('nge juvenile scale verification red contracts', () => {
  describe('runNgeLifecycle seed-driven node scale', () => {
    let nodeTelemetry: ScaleTelemetry[];
    let nodeSummary: ReturnType<typeof summarizeTelemetry>;

    beforeAll(() => {
      const network = new Network(4, 2, { seed: DEFAULT_SEED });
      const config = buildTunedNodeGrowthConfig();
      const stream = makeMonotonicStream(SCALE_NODE_WINDOW_COUNT);
      nodeTelemetry = runGrowthStream(network, config, stream);
      nodeSummary = summarizeTelemetry(nodeTelemetry);
    }, 180_000);

    it('grows from seed to at least 8000 nodes in 400 windows with tuned continuous adaptation', () => {
      expect(nodeSummary.finalNodes).toBeGreaterThanOrEqual(
        NGE_MAX_NODE_CAPACITY,
      );
    }, 180_000);
  });

  describe('runNgeLifecycle seed-driven edge scale', () => {
    it('grows from seed to at least 32000 edges in 300 windows with edge densification', () => {
      const network = new Network(4, 2, { seed: DEFAULT_SEED });
      const config = buildTunedEdgeDensifyConfig();
      const stream = makeMonotonicStream(SCALE_EDGE_WINDOW_COUNT);

      runGrowthStream(network, config, stream);

      expect(network.connections.length).toBeGreaterThanOrEqual(
        NGE_MAX_EDGE_CAPACITY,
      );
    }, 180_000);
  });

  describe('runNgeLifecycle topology determinism at scale', () => {
    it('reproduces identical topology from the same seed and experience stream at scale', () => {
      const stream = makeMonotonicStream(SCALE_DETERMINISM_WINDOW_COUNT);
      const config = buildTunedNodeGrowthConfig();

      const networkA = new Network(4, 2, { seed: DEFAULT_SEED });
      runGrowthStream(networkA, config, stream);

      const networkB = new Network(4, 2, { seed: DEFAULT_SEED });
      runGrowthStream(networkB, config, stream);

      expect(topologyFingerprint(networkA)).toEqual(
        topologyFingerprint(networkB),
      );
    }, 180_000);
  });
});
