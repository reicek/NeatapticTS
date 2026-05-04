import { Connection, Network } from '../../../neataptic';
import {
  asTopologyProps,
  applyIncomingEdgeCounts,
  clearCachedTopoOrder,
  createTopologyBuildContext,
  initializeAllNodeInDegreeCounts,
} from './network.topology.setup.utils';

describe('network topology setup helpers', () => {
  describe('clearCachedTopoOrder()', () => {
    describe('given cached topology state has already been populated', () => {
      describe('when the cache is cleared directly', () => {
        it('resets the cached schedule, order, diagnostics, and dirty flag', () => {
          // Arrange
          const network = new Network(1, 1, { enforceAcyclic: true });
          const internalTopologyProps = asTopologyProps(network);
          Reflect.set(
            internalTopologyProps,
            '_activationSchedulingDiagnostics',
            {
              topologyIntent: 'feed-forward',
            },
          );
          Reflect.set(internalTopologyProps, '_activationSchedule', {
            mode: 'acyclic',
          });
          Reflect.set(internalTopologyProps, '_topoOrder', [network.nodes[0]]);
          Reflect.set(internalTopologyProps, '_topoDirty', true);

          // Act
          clearCachedTopoOrder(internalTopologyProps);

          // Assert
          expect({
            activationSchedulingDiagnostics: Reflect.get(
              internalTopologyProps,
              '_activationSchedulingDiagnostics',
            ),
            activationSchedule: Reflect.get(
              internalTopologyProps,
              '_activationSchedule',
            ),
            topoOrder: Reflect.get(internalTopologyProps, '_topoOrder'),
            topoDirty: Reflect.get(internalTopologyProps, '_topoDirty'),
          }).toEqual({
            activationSchedulingDiagnostics: null,
            activationSchedule: null,
            topoOrder: null,
            topoDirty: false,
          });
        });
      });
    });
  });

  describe('applyIncomingEdgeCounts()', () => {
    describe('given a self-loop and a normal inbound edge are both present', () => {
      describe('when incoming edge counts are applied', () => {
        it('counts only the non-self inbound edge', () => {
          // Arrange
          const network = new Network(1, 1, { enforceAcyclic: false });
          const inputNode = network.nodes[0];
          const outputNode = network.nodes[1];

          network.connections = [
            new Connection(inputNode, outputNode, 1),
            new Connection(outputNode, outputNode, 1),
          ];

          const buildContext = createTopologyBuildContext(
            network,
            asTopologyProps(network),
          );
          initializeAllNodeInDegreeCounts(buildContext);

          // Act
          applyIncomingEdgeCounts(buildContext);

          // Assert
          expect(buildContext.inDegreeByNode.get(outputNode)).toBe(1);
        });
      });
    });

    describe('given the in-degree map starts empty', () => {
      describe('when incoming edge counts are applied directly', () => {
        it('seeds the missing entry from zero before incrementing it', () => {
          // Arrange
          const network = new Network(1, 1, { enforceAcyclic: false });
          const inputNode = network.nodes[0];
          const outputNode = network.nodes[1];

          network.connections = [new Connection(inputNode, outputNode, 1)];

          const buildContext = createTopologyBuildContext(
            network,
            asTopologyProps(network),
          );

          // Act
          applyIncomingEdgeCounts(buildContext);

          // Assert
          expect(buildContext.inDegreeByNode.get(outputNode)).toBe(1);
        });
      });
    });
  });
});
