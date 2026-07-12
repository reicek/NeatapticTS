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
import type Node from '../../architecture/node';
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

      it('skips edgeDensify when proposedAdditions is zero', () => {
        // Arrange
        const network = new Network(3, 2, { seed: 42 });
        const initialConnectionCount = network.connections.length;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgeDensify',
            targetModuleId: 'module-A',
            detail: {
              currentEdgeCount: initialConnectionCount,
              proposedAdditions: 0,
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 0,
          },
        ];

        // Act
        const outcomes = applyMorphDeltas(
          network,
          deltas,
          buildPermissiveBudget(network),
        );

        // Assert
        expect(outcomes[0]).toEqual({
          kind: 'edgeDensify',
          status: 'skipped',
          reason:
            'ADD_CONN produced no net edges (saturated graph or sparsity budget pruning).',
        });
      });

      it('skips edgeDensify when the graph has no missing forward pairs', () => {
        // Arrange — a minimal feed-forward network with all possible
        // input-to-output edges already present.
        const network = new Network(2, 1, { seed: 42 });
        const initialConnectionCount = network.connections.length;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgeDensify',
            targetModuleId: 'module-A',
            detail: {
              currentEdgeCount: initialConnectionCount,
              proposedAdditions: 5,
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 5,
          },
        ];

        // Act
        const outcomes = applyMorphDeltas(
          network,
          deltas,
          buildPermissiveBudget(network),
        );

        // Assert
        expect(outcomes[0]?.status).toBe('skipped');
      });

      it('applies all available missing forward pairs when fewer exist than requested', () => {
        // Arrange — one input and two outputs with only one missing forward
        // pair. Requesting more edges than exist should apply exactly the one
        // missing pair, exercising the deduplication guard on repeated samples.
        const network = new Network(1, 2, { seed: 42 });
        network.disconnect(network.nodes[0] as Node, network.nodes[2] as Node);
        const afterDisconnect = network.connections.length;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgeDensify',
            targetModuleId: 'module-A',
            detail: {
              currentEdgeCount: afterDisconnect,
              proposedAdditions: 5,
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 5,
          },
        ];

        // Act
        applyMorphDeltas(network, deltas, buildPermissiveBudget(network));

        // Assert — exactly one new edge is added because the pool is exhausted.
        expect(network.connections.length).toBe(afterDisconnect + 1);
      });

      it('applies all available candidates when proposedAdditions exceed the candidate pool', () => {
        // Arrange — one hidden node leaves a small set of missing forward pairs.
        const network = new Network(2, 1, { seed: 42 });
        network.mutate(mutation.ADD_NODE);
        const initialConnectionCount = network.connections.length;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgeDensify',
            targetModuleId: 'module-A',
            detail: {
              currentEdgeCount: initialConnectionCount,
              proposedAdditions: 100,
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 100,
          },
        ];

        // Act
        const outcomes = applyMorphDeltas(
          network,
          deltas,
          buildPermissiveBudget(network),
        );

        // Assert — the delta applied at least one new edge, but not all 100.
        expect(outcomes[0]?.status).toBe('applied');
        expect(network.connections.length).toBeLessThan(
          initialConnectionCount + 100,
        );
      });

      it('defaults to one addition when detail.proposedAdditions is omitted', () => {
        // Arrange — hidden nodes ensure missing forward pairs exist.
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
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 1,
          },
        ];

        // Act
        applyMorphDeltas(network, deltas, buildPermissiveBudget(network));

        // Assert
        expect(network.connections.length).toBe(initialConnectionCount + 1);
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

      it('skips nodeAdd when the network sparsity budget denies the new connection', () => {
        // Arrange — cap total connections below the minimum room needed so
        // ADD_NODE cannot afford the extra connection it needs to split an
        // edge. The sparsity budget prunes to make room only when the planned
        // prune count stays above MIN_REMAINING_CONNECTION_COUNT; with only
        // one connection allowed and two already present, growth is denied.
        const network = new Network(2, 1, { seed: 42 });
        network.configureSparsityBudget({
          maxConnections: 1,
        });
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
        const results = applyMorphDeltas(
          network,
          deltas,
          buildPermissiveBudget(network),
        );

        // Assert
        expect(results[0]).toEqual({
          kind: 'nodeAdd',
          status: 'skipped',
          reason: 'ADD_NODE produced no net hidden nodes.',
        });
      });

      it('defaults to one addition when detail.proposedAdditions is omitted', () => {
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

      it('throws NgeJuvenile_BudgetError when nodeAdd would exceed maxNodes', () => {
        // Arrange — set maxNodes equal to current so any node addition overflows
        const network = new Network(2, 1, { seed: 42 });
        const currentNodeCount = network.nodes.length;
        const budget = buildPermissiveBudget(network);
        budget.growth.maxNodes = currentNodeCount;
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'nodeAdd',
            targetModuleId: 'module-A',
            detail: {
              currentNodeCount,
              proposedAdditions: 1,
              rewardDelta: 0.5,
            },
            wiringCostDelta: 0,
          },
        ];

        // Act & Assert
        expect(() => applyMorphDeltas(network, deltas, budget)).toThrow(
          NgeJuvenile_BudgetError,
        );
      });

      it('throws NgeJuvenile_BudgetError when compact would drop below minNodes', () => {
        // Arrange — add hidden nodes then set minNodes equal to the current
        // hidden count so any compact violates the floor.
        const network = new Network(2, 1, { seed: 42 });
        network.mutate(mutation.ADD_NODE);
        const hiddenCount = countHiddenNodes(network);
        const budget = buildPermissiveBudget(network);
        budget.prune.minNodes = hiddenCount;
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

        // Act & Assert
        expect(() => applyMorphDeltas(network, deltas, budget)).toThrow(
          NgeJuvenile_BudgetError,
        );
      });
    });

    describe('batch and empty input handling', () => {
      it('returns an empty outcome array when given no deltas', () => {
        // Arrange
        const network = new Network(2, 1, { seed: 42 });
        const deltas: NgeMorphDelta[] = [];

        // Act
        const outcomes = applyMorphDeltas(
          network,
          deltas,
          buildPermissiveBudget(network),
        );

        // Assert
        expect(outcomes).toEqual([]);
      });

      it('processes multiple deltas in one batch call preserving order', () => {
        // Arrange — add hidden nodes so both edgeDensify and nodeAdd have room
        const network = new Network(3, 2, { seed: 42 });
        network.mutate(mutation.ADD_NODE);
        network.mutate(mutation.ADD_NODE);
        const initialConnectionCount = network.connections.length;
        const initialHiddenCount = countHiddenNodes(network);
        const deltas: NgeMorphDelta[] = [
          {
            kind: 'edgeDensify',
            targetModuleId: 'module-A',
            detail: {
              currentEdgeCount: initialConnectionCount,
              proposedAdditions: 1,
              normalizedFocusScore: 0.8,
            },
            wiringCostDelta: 1,
          },
          {
            kind: 'nodeAdd',
            targetModuleId: 'module-B',
            detail: {
              currentNodeCount: network.nodes.length,
              rewardDelta: 0.5,
            },
            wiringCostDelta: 0,
          },
          {
            kind: 'slotExpand',
            targetModuleId: 'module-C',
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
        const outcomes = applyMorphDeltas(
          network,
          deltas,
          buildPermissiveBudget(network),
        );

        // Assert — three outcomes in input order: applied, applied, skipped
        expect(outcomes.map(({ kind }) => kind)).toEqual([
          'edgeDensify',
          'nodeAdd',
          'slotExpand',
        ]);
        expect(outcomes.map(({ status }) => status)).toEqual([
          'applied',
          'applied',
          'skipped',
        ]);
        expect(network.connections.length).toBe(initialConnectionCount + 2);
        expect(countHiddenNodes(network)).toBe(initialHiddenCount + 1);
      });
    });
  });
});
