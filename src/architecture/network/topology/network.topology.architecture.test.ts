import { Architect, Network, methods } from '../../../neataptic';
import { describeArchitecture } from './network.topology.architecture.utils';

type RuntimeNodeSnapshot = {
  index?: number;
  layer?: number;
  type?: string;
};

type RuntimeConnectionSnapshot = {
  enabled?: boolean;
  from?: { index?: number };
  to?: { index?: number };
};

function createArchitectureProbeNetwork(input: {
  connections?: RuntimeConnectionSnapshot[];
  nodes?: RuntimeNodeSnapshot[];
}): Network {
  const network = new Network(1, 1, {});

  Reflect.set(network, 'nodes', input.nodes);
  Reflect.set(network, 'connections', input.connections);

  return network;
}

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

    describe('given runtime node and connection arrays are missing entirely', () => {
      describe('when the descriptor is read directly from the helper', () => {
        it('falls back to an empty inferred descriptor', () => {
          // Arrange
          const network = createArchitectureProbeNetwork({});
          const expectedDescriptor = {
            hiddenLayerSizes: [],
            hasCycles: false,
            source: 'inferred',
            totalNodes: 0,
            totalConnections: 0,
          };

          // Act
          const descriptor = describeArchitecture(network);

          // Assert
          expect(descriptor).toEqual(expectedDescriptor);
        });
      });
    });

    describe('given explicit hidden-layer metadata is present on unordered runtime nodes', () => {
      describe('when the descriptor is read directly from the helper', () => {
        it('prefers the sorted layer-metadata widths over fallback inference', () => {
          // Arrange
          const network = createArchitectureProbeNetwork({
            nodes: [
              { type: 'hidden', layer: 2 },
              { type: 'input', layer: 0 },
              { type: 'hidden', layer: 1 },
              { type: 'constant', layer: 4 },
              { type: 'hidden', layer: 2 },
              { type: 'output', layer: 5 },
              { type: 'hidden' },
            ],
            connections: [],
          });
          const expectedDescriptor = {
            hiddenLayerSizes: [1, 2],
            hasCycles: false,
            source: 'layer-metadata',
            totalNodes: 7,
            totalConnections: 0,
          };

          // Act
          const descriptor = describeArchitecture(network);

          // Assert
          expect(descriptor).toEqual(expectedDescriptor);
        });
      });
    });

    describe('given a runtime node omits its type tag but keeps numeric layer metadata', () => {
      describe('when the descriptor is read directly from the helper', () => {
        it('treats the node as a hidden-layer metadata candidate', () => {
          // Arrange
          const network = createArchitectureProbeNetwork({
            nodes: [{ layer: 3 }],
            connections: [],
          });
          const expectedDescriptor = {
            hiddenLayerSizes: [1],
            hasCycles: false,
            source: 'layer-metadata',
            totalNodes: 1,
            totalConnections: 0,
          };

          // Act
          const descriptor = describeArchitecture(network);

          // Assert
          expect(descriptor).toEqual(expectedDescriptor);
        });
      });
    });

    describe('given graph topology includes malformed and ignored edge shapes', () => {
      describe('when the descriptor is read directly from the helper', () => {
        it('derives hidden widths from only the validated directed edges', () => {
          // Arrange
          const network = createArchitectureProbeNetwork({
            nodes: [
              { index: 10, type: 'input' },
              { index: 20, type: 'hidden' },
              { index: 30, type: 'output' },
            ],
            connections: [
              { enabled: false, from: { index: 10 }, to: { index: 20 } },
              { enabled: true, from: {}, to: { index: 20 } },
              { enabled: true, from: { index: 10 }, to: { index: 999 } },
              { enabled: true, from: { index: 20 }, to: { index: 20 } },
              { enabled: true, from: { index: 10 }, to: { index: 20 } },
              { enabled: true, from: { index: 20 }, to: { index: 30 } },
            ],
          });
          const expectedDescriptor = {
            hiddenLayerSizes: [1],
            hasCycles: false,
            source: 'graph-topology',
            totalNodes: 3,
            totalConnections: 6,
          };

          // Act
          const descriptor = describeArchitecture(network);

          // Assert
          expect(descriptor).toEqual(expectedDescriptor);
        });
      });
    });

    describe('given hidden nodes never receive a resolved parent depth', () => {
      describe('when the descriptor is read directly from the helper', () => {
        it('falls back to inferred hidden-node counting', () => {
          // Arrange
          const network = createArchitectureProbeNetwork({
            nodes: [
              { index: 0, type: 'hidden' },
              { index: 1, type: 'hidden' },
              { index: 2, type: 'output' },
            ],
            connections: [
              { enabled: true, from: { index: 0 }, to: { index: 1 } },
              { enabled: true, from: { index: 1 }, to: { index: 2 } },
            ],
          });
          const expectedDescriptor = {
            hiddenLayerSizes: [2],
            hasCycles: false,
            source: 'inferred',
            totalNodes: 3,
            totalConnections: 2,
          };

          // Act
          const descriptor = describeArchitecture(network);

          // Assert
          expect(descriptor).toEqual(expectedDescriptor);
        });
      });
    });
  });
});
