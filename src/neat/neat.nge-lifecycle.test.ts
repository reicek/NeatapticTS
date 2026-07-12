import type { EquilibriumCandidate } from './nge-adult/neat.nge-adult.types';
import type {
  NgeAssimilationCandidate,
  NgeAssimilationPolicy,
  NgeAssimilationResult,
} from './nge-assimilation/neat.nge-assimilation.types';
import {
  computeFocusScores,
  resolveFocusConfig,
} from './nge-juvenile/neat.nge-juvenile.focus';
import { planGrowthMorphs } from './nge-juvenile/neat.nge-juvenile.grow';
import type {
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeJuvenilePhaseConfig,
  NgeModuleMetricsSnapshot,
} from './nge-juvenile/neat.nge-juvenile.types';
import { restoreNetworkSnapshot, runNgeLifecycle } from './neat.nge-lifecycle';
import Connection from '../architecture/connection';
import Network from '../architecture/network';

function createJuvenileFixture(): {
  metrics: NgeModuleMetricsSnapshot;
  budget: NgeGrowthBudget;
  config: NgeJuvenilePhaseConfig;
  hysteresis: NgeHysteresisState;
} {
  const metrics: NgeModuleMetricsSnapshot = {
    moduleId: 'module:alpha',
    utilization: 1,
    rewardDelta: 1,
    novelty: 1,
    stabilityAge: 1,
    wiringCost: 0,
  };
  const budget: NgeGrowthBudget = {
    maxNodes: 16,
    maxEdges: 32,
    maxEpisodicSlots: 8,
    currentNodeCount: 1,
    currentEdgeCount: 1,
    currentEpisodicSlotCount: 0,
  };
  const config = resolveFocusConfig({
    hysteresisWindowCount: 2,
    cooldownWindowCount: 0,
  });
  const hysteresis: NgeHysteresisState = {
    growthPositiveWindowCount: 2,
    pruneUnderuseWindowCount: 0,
    lastMorphKind: 'none',
    cooldownWindowsRemaining: 0,
  };

  return { metrics, budget, config, hysteresis };
}

function createAssimilationCandidate(
  equilibriumCandidate: EquilibriumCandidate,
): NgeAssimilationCandidate {
  return {
    equilibriumCandidate,
    sourceDnaFingerprint: 'fingerprint:alpha',
    sourceSchemaVersion: 'A.1.0',
    moduleDelta: {
      moduleId: 'module:alpha',
      zoneId: equilibriumCandidate.zoneId,
      ruleParameters: {
        replicationDepth: {
          currentValue: 10,
          targetValue: 30,
        },
      },
    },
  };
}

function createAssimilationPolicy(): NgeAssimilationPolicy {
  return {
    writeBackRate: 0.5,
    budgetGuardEnabled: true,
    encodingMode: 'lossless',
    maxNodes: 16,
    maxEdges: 32,
    maxBytes: 256,
  };
}

describe('nge lifecycle staging runner', () => {
  describe('runNgeLifecycle', () => {
    it('exists and exports a runner function', () => {
      expect(typeof runNgeLifecycle).toBe('function');
    });

    it('advances a juvenile module through a maturity window and returns stage adult', () => {
      // Arrange
      const { metrics, budget, config, hysteresis } = createJuvenileFixture();
      const focusVector = computeFocusScores([metrics], config);
      const focusScore = focusVector.scores[0];
      const deltas = planGrowthMorphs(
        'module:alpha',
        focusScore,
        metrics,
        budget,
        config,
        hysteresis,
      );

      // Act
      const result = runNgeLifecycle({
        stage: 'juvenile',
        moduleId: 'module:alpha',
        metrics,
        budget,
        config,
        hysteresis,
        focusScore,
        deltas,
      });

      // Assert
      expect(result.stage).toBe('adult');
    });

    it('re-seeds a live network when a seed is supplied', () => {
      // Arrange
      const { metrics, budget, config, hysteresis } = createJuvenileFixture();
      const focusVector = computeFocusScores([metrics], config);
      const focusScore = focusVector.scores[0];
      const deltas = planGrowthMorphs(
        'module:alpha',
        focusScore,
        metrics,
        budget,
        config,
        hysteresis,
      );
      const network = new Network(2, 1);

      // Act
      const result = runNgeLifecycle({
        stage: 'juvenile',
        moduleId: 'module:alpha',
        metrics,
        budget,
        config,
        hysteresis,
        focusScore,
        deltas,
        network,
        seed: 42,
        pruneBudget: {
          minEdges: 0,
          minNodes: 1,
          costExemptEdgeIds: [],
          currentEdgeCount: network.connections.length,
          currentNodeCount: network.nodes.length,
          currentWiringCost: network.nodes.length + network.connections.length,
        },
      });

      // Assert
      expect(result.stage).toBe('adult');
    });

    it('hands an emitted equilibrium candidate to assimilation and produces a structural-prior delta', () => {
      // Arrange
      const equilibriumCandidate: EquilibriumCandidate = {
        zoneId: 'zone:alpha',
        isGainStable: true,
        isPlateau: true,
      };
      const candidate = createAssimilationCandidate(equilibriumCandidate);
      const policy = createAssimilationPolicy();

      // Act
      const result = runNgeLifecycle({
        stage: 'adult',
        equilibriumCandidate,
        candidate,
        policy,
      });

      // Assert
      expect(
        (result.assimilationResult as NgeAssimilationResult).updatedModuleDelta,
      ).not.toBeNull();
    });
  });
});

describe('network snapshot rollback preserves global innovation counter', () => {
  afterEach(() => {
    Connection.resetInnovationCounter(1);
  });

  it('restores Connection.nextInnovation after rolling back a mutated network', () => {
    Connection.resetInnovationCounter(1000);
    const network = new Network(4, 2, { seed: 42 });
    const snapshot = network.toJSON();
    const initialInnovation = Connection.nextInnovation;

    network.connect(network.nodes[0], network.nodes[network.nodes.length - 1]);
    restoreNetworkSnapshot(network, snapshot, initialInnovation);

    expect(Connection.nextInnovation).toBe(initialInnovation);
  });
});
