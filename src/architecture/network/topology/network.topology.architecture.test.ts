import { Architect, Network, methods } from '../../../neataptic';

describe('network topology architecture descriptor', () => {
  describe('describeArchitecture()', () => {
    describe('given a perceptron topology has two hidden layers', () => {
      describe('when the descriptor is read', () => {
        it('reports both hidden-layer widths', () => {
          // Arrange
          const network = Architect.perceptron(12, 6, 6, 2);
          const expectedDescriptor = '6 - 6';

          // Act
          const descriptor = network.describeArchitecture();

          // Assert
          expect(descriptor.hiddenLayerSizes.join(' - ')).toBe(
            expectedDescriptor,
          );
        });
      });
    });

    describe('given a perceptron topology is serialized', () => {
      describe('when the JSON payload is inspected', () => {
        it('exports architecture metadata', () => {
          // Arrange
          const network = Architect.perceptron(4, 3, 2);

          // Act
          const jsonPayload = network.toJSON() as {
            architecture?: { hiddenLayerSizes?: number[] };
          };

          // Assert
          expect(
            Array.isArray(jsonPayload.architecture?.hiddenLayerSizes),
          ).toBe(true);
        });
      });
    });

    describe('given a hydrated inferred descriptor is present', () => {
      describe('when the live graph is still inferred-only', () => {
        it('reuses the hydrated hidden-layer widths', () => {
          // Arrange
          const baseNetwork = new Network(2, 1);
          const jsonPayload = baseNetwork.toJSON() as {
            architecture?: {
              hiddenLayerSizes: number[];
              hasCycles: boolean;
              source: 'layer-metadata' | 'graph-topology' | 'inferred';
              totalNodes: number;
              totalConnections: number;
            };
          };
          const expectedDescriptor = '6 - 6';

          jsonPayload.architecture = {
            hiddenLayerSizes: [6, 6],
            hasCycles: false,
            source: 'inferred',
            totalNodes: baseNetwork.nodes.length,
            totalConnections: baseNetwork.connections.length,
          };

          // Act
          const rebuiltNetwork = Network.fromJSON(
            jsonPayload as unknown as Record<string, unknown>,
          );
          const descriptor = rebuiltNetwork.describeArchitecture();

          // Assert
          expect(descriptor.hiddenLayerSizes.join(' - ')).toBe(
            expectedDescriptor,
          );
        });
      });
    });

    describe('given a cycle is introduced into the graph', () => {
      describe('when the descriptor is read', () => {
        it('reports that cycles are present', () => {
          // Arrange
          const network = new Network(2, 1);
          const outputNode = network.nodes.find(
            (node) => node.type === 'output',
          );

          if (!outputNode) {
            throw new Error(
              'Expected at least one output node to construct cycle',
            );
          }

          const inputNode = network.nodes[0];
          network.connect(outputNode, inputNode, 0.1);

          // Act
          const descriptor = network.describeArchitecture();

          // Assert
          expect(descriptor.hasCycles).toBe(true);
        });
      });
    });

    describe('given a factual descriptor is available', () => {
      describe('when the runtime cache is inspected after describing the network', () => {
        it('refreshes the cached descriptor away from inferred mode', () => {
          // Arrange
          const network = Architect.perceptron(12, 6, 6, 2);

          // Act
          network.describeArchitecture();
          const runtimeNetwork = network as unknown as {
            _serializedArchitectureDescriptor?: {
              source?: 'layer-metadata' | 'graph-topology' | 'inferred';
            };
          };

          // Assert
          expect(
            runtimeNetwork._serializedArchitectureDescriptor?.source,
          ).not.toBe('inferred');
        });
      });
    });

    describe('given a hydrated inferred descriptor becomes stale after mutation', () => {
      describe('when the descriptor is read again', () => {
        it('does not reuse the stale hidden-layer widths', () => {
          // Arrange
          const baseNetwork = new Network(2, 1);
          const jsonPayload = baseNetwork.toJSON() as {
            architecture?: {
              hiddenLayerSizes: number[];
              hasCycles: boolean;
              source: 'layer-metadata' | 'graph-topology' | 'inferred';
              totalNodes: number;
              totalConnections: number;
            };
          };
          const staleDescriptor = '6 - 6';

          jsonPayload.architecture = {
            hiddenLayerSizes: [6, 6],
            hasCycles: false,
            source: 'inferred',
            totalNodes: baseNetwork.nodes.length,
            totalConnections: baseNetwork.connections.length,
          };

          const rebuiltNetwork = Network.fromJSON(
            jsonPayload as unknown as Record<string, unknown>,
          );
          rebuiltNetwork.mutate(methods.mutation.ADD_NODE);

          // Act
          const descriptor = rebuiltNetwork.describeArchitecture();

          // Assert
          expect(descriptor.hiddenLayerSizes.join(' - ')).not.toBe(
            staleDescriptor,
          );
        });
      });
    });
  });
});
