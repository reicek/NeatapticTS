import mutation from '../methods/mutation/mutation';
import Network from '../architecture/network';
import { resolveFocusConfig } from './nge-juvenile/neat.nge-juvenile.focus';
import type {
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
} from './nge-juvenile/neat.nge-juvenile.types';
import type { NgeJuvenileLifecycleInput } from './neat.nge-lifecycle';
import { runNgeLifecycle } from './neat.nge-lifecycle';

function createNetworkFixture(): Network {
  const network = new Network(3, 2, { seed: 42 });
  network.mutate(mutation.ADD_NODE);
  network.mutate(mutation.ADD_NODE);
  return network;
}

function createApplyFixture(): {
  network: Network;
  input: NgeJuvenileLifecycleInput;
} {
  const network = createNetworkFixture();

  const metrics: NgeModuleMetricsSnapshot = {
    moduleId: 'module:alpha',
    utilization: 1,
    rewardDelta: 1,
    novelty: 1,
    stabilityAge: 1,
    wiringCost: 0,
  };

  const budget: NgeGrowthBudget = {
    maxNodes: 100,
    maxEdges: 1000,
    maxEpisodicSlots: 100,
    currentNodeCount: network.nodes.length,
    currentEdgeCount: network.connections.length,
    currentEpisodicSlotCount: 0,
  };

  const pruneBudget: NgePruneBudget = {
    minEdges: 0,
    minNodes: 0,
    costExemptEdgeIds: [],
    currentEdgeCount: network.connections.length,
    currentNodeCount: network.nodes.length,
    currentWiringCost: 0,
  };

  const config = resolveFocusConfig({
    hysteresisWindowCount: 2,
    cooldownWindowCount: 3,
  });

  const hysteresis: NgeHysteresisState = {
    growthPositiveWindowCount: 2,
    pruneUnderuseWindowCount: 0,
    lastMorphKind: 'none',
    cooldownWindowsRemaining: 0,
  };

  const input: NgeJuvenileLifecycleInput = {
    stage: 'juvenile',
    moduleId: 'module:alpha',
    metrics,
    budget,
    config,
    hysteresis,
    network,
    pruneBudget,
  };

  return { network, input };
}

describe('nge lifecycle apply wiring', () => {
  describe('runNgeLifecycle with network and apply-phase', () => {
    it('mutates the network by applying planned growth morphs', () => {
      // Arrange
      const { network, input } = createApplyFixture();
      const initialNodeCount = network.nodes.length;

      // Act
      runNgeLifecycle(input);

      // Assert
      expect(network.nodes.length).toBeGreaterThan(initialNodeCount);
    });

    it('returns apply outcomes from applyMorphDeltas in the lifecycle result', () => {
      // Arrange
      const { input } = createApplyFixture();

      // Act
      const result = runNgeLifecycle(input);

      // Assert
      expect(
        result.applyOutcomes?.some(
          (o) => o.kind === 'slotExpand' && o.status === 'skipped',
        ),
      ).toBe(true);
    });

    it('updates hysteresis via commitGrowth after morph application', () => {
      // Arrange
      const { input } = createApplyFixture();

      // Act
      const result = runNgeLifecycle(input);

      // Assert
      expect(result.hysteresis?.cooldownWindowsRemaining).toBe(3);
    });

    it('passes the growth budget through to applyMorphDeltas', () => {
      // Arrange
      const { network, input } = createApplyFixture();
      const initialConnectionCount = network.connections.length;

      // Act
      runNgeLifecycle(input);

      // Assert
      expect(network.connections.length).toBeGreaterThan(
        initialConnectionCount,
      );
    });

    it('skips commitGrowth when no growth morph was applied', () => {
      // Arrange — craft a fixture where only slotExpand is planned (edgeDensify
      // and nodeAdd both fail their evidence/budget gates), so every apply
      // outcome is skipped and commitGrowth must not fire.
      const network = createNetworkFixture();

      const metrics: NgeModuleMetricsSnapshot = {
        moduleId: 'module:alpha',
        utilization: 1,
        rewardDelta: 0,
        novelty: 1,
        stabilityAge: 1,
        wiringCost: 0,
      };

      const budget: NgeGrowthBudget = {
        maxNodes: network.nodes.length,
        maxEdges: network.connections.length,
        maxEpisodicSlots: 100,
        currentNodeCount: network.nodes.length,
        currentEdgeCount: network.connections.length,
        currentEpisodicSlotCount: 0,
      };

      const pruneBudget: NgePruneBudget = {
        minEdges: 0,
        minNodes: 0,
        costExemptEdgeIds: [],
        currentEdgeCount: network.connections.length,
        currentNodeCount: network.nodes.length,
        currentWiringCost: 0,
      };

      const config = resolveFocusConfig({
        hysteresisWindowCount: 2,
        cooldownWindowCount: 3,
      });

      const hysteresis: NgeHysteresisState = {
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      };

      const input: NgeJuvenileLifecycleInput = {
        stage: 'juvenile',
        moduleId: 'module:alpha',
        metrics,
        budget,
        config,
        hysteresis,
        network,
        pruneBudget,
      };

      // Act
      const result = runNgeLifecycle(input);

      // Assert — hysteresis passes through unchanged because no growth morph
      // was committed.
      expect(result.hysteresis?.cooldownWindowsRemaining).toBe(0);
    });
  });
});
