import { Architect, Network, methods } from '../../src/neataptic';

describe('Network Architecture Descriptor', () => {
  it('resolves hidden-layer widths for perceptron topology', () => {
    const network = Architect.perceptron(12, 6, 6, 2);

    const architectureDescriptor = network.describeArchitecture();

    expect(architectureDescriptor.hiddenLayerSizes.join(' - ')).toBe('6 - 6');
  });

  it('exports architecture metadata in JSON payload', () => {
    const network = Architect.perceptron(4, 3, 2);

    const jsonPayload = network.toJSON() as {
      architecture?: { hiddenLayerSizes?: number[] };
    };

    expect(Array.isArray(jsonPayload.architecture?.hiddenLayerSizes)).toBe(true);
  });

  it('hydrates serialized descriptor when live graph is inferred-only', () => {
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

    expect(rebuiltNetwork.describeArchitecture().hiddenLayerSizes.join(' - ')).toBe(
      '6 - 6',
    );
  });

  it('reports cycle presence in architecture descriptor', () => {
    const network = new Network(2, 1);
    network.mutate(methods.mutation.ADD_NODE);

    const outputNode = network.nodes[network.input];
    const inputNode = network.nodes[0];
    network.connect(outputNode, inputNode, 0.1);

    const architectureDescriptor = network.describeArchitecture();

    expect(architectureDescriptor.hasCycles).toBe(true);
  });
});
