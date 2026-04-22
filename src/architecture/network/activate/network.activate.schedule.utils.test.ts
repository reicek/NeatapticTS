import Network from '../network';
import type Node from '../../node';
import type { ActivationSchedule } from '../network.types';
import {
  resolveActivationTraversalNodes,
  resolveInputValuesByNodeId,
  resolveOrderedOutputNodes,
} from './network.activate.schedule.utils';

type ScheduleAwareRuntime = {
  _activationSchedule?: ActivationSchedule | null;
  _topoOrder?: Node[] | null;
};

function setActivationSchedule(
  network: Network,
  activationSchedule: ActivationSchedule | null,
): void {
  Reflect.set(
    network as unknown as ScheduleAwareRuntime,
    '_activationSchedule',
    activationSchedule,
  );
}

function setTopoOrder(network: Network, topoOrder: Node[] | null): void {
  Reflect.set(network as unknown as ScheduleAwareRuntime, '_topoOrder', topoOrder);
}

describe('network activate chapter', () => {
  describe('schedule helpers', () => {
    describe('given a compiled schedule mixes wave and recurrent steps', () => {
      describe('when traversal nodes are resolved', () => {
        it('flattens the scheduled order with recurrent iterations applied', () => {
          // Arrange
          const network = new Network(2, 1, { enforceAcyclic: true });
          const [inputLeft, inputRight, outputNode] = network.nodes;

          setActivationSchedule(network, {
            mode: 'recurrent',
            outputNodeIds: [outputNode.geneId],
            stateSemantics: 'carry',
            steps: [
              {
                kind: 'wave',
                nodeIds: [inputRight.geneId, inputLeft.geneId],
              },
              {
                kind: 'recurrent-component',
                nodeIds: [outputNode.geneId],
              },
              {
                kind: 'recurrent-component',
                iterations: 2,
                nodeIds: [outputNode.geneId],
              },
            ],
          });

          // Act
          const traversalNodeIds = resolveActivationTraversalNodes(network).map(
            (node) => node.geneId,
          );

          // Assert
          expect(traversalNodeIds).toEqual([
            inputRight.geneId,
            inputLeft.geneId,
            outputNode.geneId,
            outputNode.geneId,
            outputNode.geneId,
          ]);
        });
      });
    });

    describe('given the compiled schedule is absent but a topo-order cache exists', () => {
      describe('when traversal nodes are resolved', () => {
        it('returns the cached topo order', () => {
          // Arrange
          const network = new Network(2, 1, { enforceAcyclic: true });
          const [inputLeft, inputRight, outputNode] = network.nodes;
          setActivationSchedule(network, null);
          setTopoOrder(network, [outputNode, inputRight, inputLeft]);

          // Act
          const traversalNodeIds = resolveActivationTraversalNodes(network).map(
            (node) => node.geneId,
          );

          // Assert
          expect(traversalNodeIds).toEqual([
            outputNode.geneId,
            inputRight.geneId,
            inputLeft.geneId,
          ]);
        });
      });
    });

    describe('given the compiled schedule references a node that no longer exists', () => {
      describe('when traversal nodes are resolved', () => {
        it('falls back to the raw node storage order', () => {
          // Arrange
          const network = new Network(2, 1, { enforceAcyclic: true });

          setActivationSchedule(network, {
            mode: 'acyclic',
            outputNodeIds: [],
            steps: [
              {
                kind: 'wave',
                nodeIds: [999_999],
              },
            ],
          });
          setTopoOrder(network, null);

          // Act
          const traversalNodeIds = resolveActivationTraversalNodes(network).map(
            (node) => node.geneId,
          );

          // Assert
          expect(traversalNodeIds).toEqual(network.nodes.map((node) => node.geneId));
        });
      });
    });

    describe('given explicit input ids are valid while node storage order drifts', () => {
      describe('when input values are resolved by node id', () => {
        it('uses the explicit input-role order instead of raw node order', () => {
          // Arrange
          const network = new Network(2, 1, { enforceAcyclic: true });
          const [inputLeft, inputRight, outputNode] = network.nodes;

          network.nodes.splice(0, network.nodes.length, inputRight, outputNode, inputLeft);

          // Act
          const resolvedInputs = Array.from(
            resolveInputValuesByNodeId(network, [0.25, 0.75]).entries(),
          );

          // Assert
          expect(resolvedInputs).toEqual([
            [inputLeft.geneId, 0.25],
            [inputRight.geneId, 0.75],
          ]);
        });
      });
    });

    describe('given input vector width no longer matches the explicit input-role list', () => {
      describe('when input values are resolved by node id', () => {
        it('falls back to scanning input-role nodes in storage order', () => {
          // Arrange
          const network = new Network(2, 1, { enforceAcyclic: true });
          const [inputLeft, inputRight, outputNode] = network.nodes;

          network.nodes.splice(0, network.nodes.length, outputNode, inputLeft, inputRight);

          // Act
          const resolvedInputs = {
            longVector: Array.from(
              resolveInputValuesByNodeId(network, [0.2, 0.8, 0.4]).entries(),
            ),
            shortVector: Array.from(
              resolveInputValuesByNodeId(network, [0.9]).entries(),
            ),
          };

          // Assert
          expect(resolvedInputs).toEqual({
            longVector: [
              [inputLeft.geneId, 0.2],
              [inputRight.geneId, 0.8],
            ],
            shortVector: [[inputLeft.geneId, 0.9]],
          });
        });
      });
    });

    describe('given explicit output ids are valid in the compiled schedule', () => {
      describe('when output nodes are resolved', () => {
        it('returns outputs in the public output-vector order', () => {
          // Arrange
          const network = new Network(1, 2, { enforceAcyclic: true });
          const [, firstOutput, secondOutput] = network.nodes;

          setActivationSchedule(network, {
            mode: 'acyclic',
            outputNodeIds: [],
            steps: [],
          });
          const fallbackOutputIds = resolveOrderedOutputNodes(network).map(
            (node) => node.geneId,
          );

          setActivationSchedule(network, {
            mode: 'acyclic',
            outputNodeIds: [secondOutput.geneId, firstOutput.geneId],
            steps: [],
          });

          // Act
          const orderedOutputIds = resolveOrderedOutputNodes(network).map(
            (node) => node.geneId,
          );

          // Assert
          expect({
            fallbackOutputIds,
            orderedOutputIds,
          }).toEqual({
            fallbackOutputIds: [firstOutput.geneId, secondOutput.geneId],
            orderedOutputIds: [secondOutput.geneId, firstOutput.geneId],
          });
        });
      });
    });

    describe('given explicit output ids include non-output nodes', () => {
      describe('when output nodes are resolved', () => {
        it('falls back to filtering the raw output nodes', () => {
          // Arrange
          const network = new Network(1, 1, { enforceAcyclic: true });
          const [inputNode, outputNode] = network.nodes;

          setActivationSchedule(network, {
            mode: 'acyclic',
            outputNodeIds: [inputNode.geneId],
            steps: [],
          });

          // Act
          const orderedOutputIds = resolveOrderedOutputNodes(network).map(
            (node) => node.geneId,
          );

          // Assert
          expect(orderedOutputIds).toEqual([outputNode.geneId]);
        });
      });
    });
  });
});