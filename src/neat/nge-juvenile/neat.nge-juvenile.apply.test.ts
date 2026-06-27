/**
 * Red tests for the NGE juvenile morph applier.
 *
 * These tests define the expected contract for `applyMorphDeltas` before the
 * implementation exists. They cover all 5 morph kinds (edgeDensify, nodeAdd,
 * slotExpand, edgePrune, compact) plus budget overflow and prune floor
 * enforcement.
 *
 * The applier function signature:
 *   applyMorphDeltas(network: Network, deltas: NgeMorphDelta[], budget: MorphApplyBudget): MorphApplyOutcome[]
 *
 * Where MorphApplyBudget combines growth and prune budgets:
 *   { growth: NgeGrowthBudget; prune: NgePruneBudget }
 *
 * And MorphApplyOutcome is:
 *   { kind: NgeMorphDelta['kind']; status: 'applied' | 'skipped'; reason?: string }
 */

import mutation from '../../methods/mutation/mutation';
import Network from '../../architecture/network';
import { NgeJuvenile_BudgetError } from './neat.nge-juvenile.errors';
import { applyMorphDeltas } from './neat.nge-juvenile.apply';
import type {
  NgeGrowthBudget,
  NgeMorphDelta,
  NgePruneBudget,
} from './neat.nge-juvenile.types';

/**
 * Combined growth and prune budget consumed by the morph applier.
 * The applier re-validates both before mutating.
 */
interface MorphApplyBudget {
  growth: NgeGrowthBudget;
  prune: NgePruneBudget;
}

function countHiddenNodes(network: Network): number {
  return network.nodes.filter(
    (candidateNode) => candidateNode.type === 'hidden',
  ).length;
}

/**
 * Builds a permissive budget that allows any reasonable mutation.
 * The applier should re-validate against this before each morph.
 */
function buildPermissiveBudget(network: Network): MorphApplyBudget {
  const hiddenCount = countHiddenNodes(network);
  return {
    growth: {
      maxNodes: 100,
      maxEdges: 1000,
      maxEpisodicSlots: 10,
      currentNodeCount: network.nodes.length,
      currentEdgeCount: network.connections.length,
      currentEpisodicSlotCount: 0,
    },
    prune: {
      minEdges: 0,
      minNodes: 0,
      costExemptEdgeIds: [],
      currentEdgeCount: network.connections.length,
      currentNodeCount: hiddenCount,
      currentWiringCost: 0,
    },
  };
}

describe('nge juvenile morph applier', () => {
  describe('applyMorphDeltas', () => {
    describe('edgeDensify', () => {
      it('adds N connections where N equals detail.proposedAdditions', () => {
        // Arrange — create a network with hidden nodes so eligible
        // unconnected pairs exist for ADD_CONN to find.
        const network = new Network(3, 2, { seed: 42 });
        network.mutate(mutation.ADD_NODE);
        network.mutate(mutation.ADD_NODE);
        const initialConnectionCount = network.connections.length;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgeDensify',
            targetModuleId: 'module-A',
            detail: {
              currentEdgeCount: initialConnectionCount,
              proposedAdditions: 2,
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 2,
          },
        ];

        // Act
        applyMorphDeltas(network, deltas, buildPermissiveBudget(network));

        // Assert
        expect(network.connections.length).toBe(initialConnectionCount + 2);
      });
    });

    describe('nodeAdd', () => {
      it('adds a hidden node to the network', () => {
        // Arrange
        const network = new Network(2, 1, { seed: 42 });
        const initialHiddenCount = countHiddenNodes(network);
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'nodeAdd',
            targetModuleId: 'module-A',
            detail: {
              currentNodeCount: network.nodes.length,
              rewardDelta: 0.5,
            },
            wiringCostDelta: 0,
          },
        ];

        // Act
        applyMorphDeltas(network, deltas, buildPermissiveBudget(network));

        // Assert
        expect(countHiddenNodes(network)).toBe(initialHiddenCount + 1);
      });
    });

    describe('edgePrune', () => {
      it('disconnects the specific connection identified by detail.candidateId', () => {
        // Arrange — add hidden nodes so we have a hidden→output connection
        // to target by innovation ID.
        const network = new Network(2, 2, { seed: 42 });
        network.mutate(mutation.ADD_NODE);
        network.mutate(mutation.ADD_NODE);
        const targetConnection = network.connections.find(
          (conn) => conn.from.type === 'hidden' && conn.to.type === 'output',
        );
        if (!targetConnection) {
          throw new Error(
            'Test fixture requires a hidden-to-output connection',
          );
        }
        const candidateId = String(targetConnection.innovation);
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgePrune',
            targetModuleId: 'module-A',
            detail: {
              candidateId,
              wiringCost: 1.0,
              edgeLength: 1.0,
              currentEdgeCount: network.connections.length,
            },
            wiringCostDelta: -1.0,
          },
        ];

        // Act
        applyMorphDeltas(network, deltas, buildPermissiveBudget(network));

        // Assert — the targeted connection must no longer exist
        expect(
          network.connections.some(
            (conn) => conn.innovation === targetConnection.innovation,
          ),
        ).toBe(false);
      });
    });

    describe('slotExpand', () => {
      it('returns a skipped outcome without mutating the network', () => {
        // Arrange
        const network = new Network(2, 1, { seed: 42 });
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'slotExpand',
            targetModuleId: 'module-A',
            detail: {
              currentSlotCount: 0,
              proposedAdditions: 1,
              hitRate: 0.9,
              hitRateSource: 'metrics.utilization',
            },
            wiringCostDelta: 1,
          },
        ];

        // Act
        const results = applyMorphDeltas(
          network,
          deltas,
          buildPermissiveBudget(network),
        );

        // Assert — slotExpand has no NEAT mutation equivalent
        expect(results[0].status).toBe('skipped');
      });
    });

    describe('compact', () => {
      it('removes a hidden node from the network', () => {
        // Arrange — need multiple hidden nodes so SUB_NODE has one to remove
        const network = new Network(2, 1, { seed: 42 });
        network.mutate(mutation.ADD_NODE);
        network.mutate(mutation.ADD_NODE);
        network.mutate(mutation.ADD_NODE);
        const initialHiddenCount = countHiddenNodes(network);
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'compact',
            targetModuleId: 'module-A',
            detail: {
              currentNodeCount: network.nodes.length,
              currentWiringCost: 0,
            },
            wiringCostDelta: 0,
          },
        ];

        // Act
        applyMorphDeltas(network, deltas, buildPermissiveBudget(network));

        // Assert
        expect(countHiddenNodes(network)).toBe(initialHiddenCount - 1);
      });
    });

    describe('budget enforcement', () => {
      it('throws NgeJuvenile_BudgetError when edgeDensify would exceed maxEdges', () => {
        // Arrange — set maxEdges equal to current so any addition overflows
        const network = new Network(2, 1, { seed: 42 });
        const currentEdgeCount = network.connections.length;
        const budget = buildPermissiveBudget(network);
        budget.growth.maxEdges = currentEdgeCount;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgeDensify',
            targetModuleId: 'module-A',
            detail: {
              currentEdgeCount,
              proposedAdditions: 1,
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 1,
          },
        ];

        // Act & Assert
        expect(() => applyMorphDeltas(network, deltas, budget)).toThrow(
          NgeJuvenile_BudgetError,
        );
      });

      it('throws NgeJuvenile_BudgetError when edgePrune would drop below minEdges', () => {
        // Arrange — set minEdges equal to current so any prune violates floor
        const network = new Network(2, 1, { seed: 42 });
        const currentEdgeCount = network.connections.length;
        const budget = buildPermissiveBudget(network);
        budget.prune.minEdges = currentEdgeCount;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgePrune',
            targetModuleId: 'module-A',
            detail: {
              candidateId: 'nonexistent',
              wiringCost: 1.0,
              edgeLength: 1.0,
              currentEdgeCount,
            },
            wiringCostDelta: -1.0,
          },
        ];

        // Act & Assert
        expect(() => applyMorphDeltas(network, deltas, budget)).toThrow(
          NgeJuvenile_BudgetError,
        );
      });
    });
  });
});
