import Network from '../network';
import { applyGradientClippingCore } from './network.training.gradient-clip.utils';

type GradientClipNetworkInternals = {
  _lastGradClipGroupCount?: number;
  layers?: Array<{ nodes?: Array<Network['nodes'][number] | null> } | null>;
};

function createSingleInputOutputNetwork(seed: number): {
  inputNode: Network['nodes'][number];
  network: Network;
  outputNode: Network['nodes'][number];
} {
  const network = new Network(1, 1, { seed });
  const inputNode = network.nodes[0];
  const outputNode = network.nodes.find((node) => node.type === 'output');

  if (!outputNode) {
    throw new Error('Expected output node to exist');
  }

  return { inputNode, network, outputNode };
}

function createSingleInputDualOutputNetwork(seed: number): {
  firstOutputNode: Network['nodes'][number];
  inputNode: Network['nodes'][number];
  network: Network;
  secondOutputNode: Network['nodes'][number];
} {
  const network = new Network(1, 2, { seed });
  const inputNode = network.nodes[0];
  const outputNodes = network.nodes.filter((node) => node.type === 'output');
  const firstOutputNode = outputNodes[0];
  const secondOutputNode = outputNodes[1];

  if (!firstOutputNode || !secondOutputNode) {
    throw new Error('Expected two output nodes to exist');
  }

  return { firstOutputNode, inputNode, network, secondOutputNode };
}

function getLastGradClipGroupCount(network: Network): number {
  return (
    Reflect.get(
      network as unknown as GradientClipNetworkInternals,
      '_lastGradClipGroupCount',
    ) as number | undefined
  ) ?? 0;
}

function setLayerMetadata(
  network: Network,
  layers: Array<{ nodes?: Array<Network['nodes'][number] | null> } | null>,
): void {
  Reflect.set(network as unknown as GradientClipNetworkInternals, 'layers', layers);
}

function clearGradientState(network: Network): void {
  network.nodes.forEach((node) => {
    node.connections.in.forEach((connection) => {
      Reflect.set(connection, 'totalDeltaWeight', undefined);
    });
    node.connections.self.forEach((connection) => {
      Reflect.set(connection, 'totalDeltaWeight', undefined);
    });
    Reflect.set(node, 'totalDeltaBias', undefined);
  });
}

function seedGradientState(options: {
  incomingDelta: number;
  network: Network;
  outputBias: number;
  outputNode: Network['nodes'][number];
  selfDelta: number;
}): {
  incomingConnection: ReturnType<Network['connect']>[number];
  selfConnection: ReturnType<Network['connect']>[number];
} {
  const inputNode = options.network.nodes[0];
  const [incomingConnection] = options.network.connect(
    inputNode,
    options.outputNode,
  );
  const [selfConnection] = options.network.connect(
    options.outputNode,
    options.outputNode,
  );

  incomingConnection.totalDeltaWeight = options.incomingDelta;
  selfConnection.totalDeltaWeight = options.selfDelta;
  Reflect.set(options.outputNode, 'totalDeltaBias', options.outputBias);

  return { incomingConnection, selfConnection };
}

describe('network training chapter', () => {
  describe('gradient clip utility helpers', () => {
    describe('given layerwise norm clipping sees explicit layer metadata', () => {
      describe('when clipping is applied', () => {
        it('skips empty layer entries and scales the populated layer gradients', () => {
          // Arrange
          const networkParts = createSingleInputOutputNetwork(201);
          const gradientState = seedGradientState({
            incomingDelta: 6,
            network: networkParts.network,
            outputBias: 8,
            outputNode: networkParts.outputNode,
            selfDelta: 10,
          });

          setLayerMetadata(networkParts.network, [
            null,
            {},
            { nodes: [networkParts.inputNode] },
            { nodes: [networkParts.inputNode, null, networkParts.outputNode] },
          ]);

          // Act
          applyGradientClippingCore(networkParts.network, {
            maxNorm: 1,
            mode: 'layerwiseNorm',
          });

          // Assert
          expect({
            groupCount: getLastGradClipGroupCount(networkParts.network),
            incomingScaled: Math.abs(gradientState.incomingConnection.totalDeltaWeight) < 6,
            selfScaled: Math.abs(gradientState.selfConnection.totalDeltaWeight) < 10,
          }).toEqual({
            groupCount: 1,
            incomingScaled: true,
            selfScaled: true,
          });
        });
      });
    });

    describe('given layerwise percentile clipping runs without explicit layer metadata', () => {
      describe('when clipping is applied', () => {
        it('collects self gradients from the node-local fallback groups', () => {
          // Arrange
          const networkParts = createSingleInputOutputNetwork(202);
          const gradientState = seedGradientState({
            incomingDelta: 0.5,
            network: networkParts.network,
            outputBias: 0.5,
            outputNode: networkParts.outputNode,
            selfDelta: 100,
          });

          // Act
          applyGradientClippingCore(networkParts.network, {
            mode: 'layerwisePercentile',
            percentile: 50,
          });

          // Assert
          expect({
            groupCount: getLastGradClipGroupCount(networkParts.network),
            selfDelta: gradientState.selfConnection.totalDeltaWeight,
          }).toEqual({
            groupCount: 1,
            selfDelta: 0.5,
          });
        });
      });
    });

    describe('given global norm clipping sees a self-connection gradient', () => {
      describe('when clipping is applied', () => {
        it('scales the self gradient inside the global clipping group', () => {
          // Arrange
          const networkParts = createSingleInputOutputNetwork(203);
          const gradientState = seedGradientState({
            incomingDelta: 0,
            network: networkParts.network,
            outputBias: 0,
            outputNode: networkParts.outputNode,
            selfDelta: 9,
          });

          // Act
          applyGradientClippingCore(networkParts.network, {
            maxNorm: 3,
            mode: 'norm',
          });

          // Assert
          expect({
            groupCount: getLastGradClipGroupCount(networkParts.network),
            selfDelta: gradientState.selfConnection.totalDeltaWeight,
          }).toEqual({
            groupCount: 1,
            selfDelta: 3,
          });
        });
      });
    });

    describe('given layerwise norm clipping receives mixed numeric and nonnumeric deltas', () => {
      describe('when clipping uses the default max norm', () => {
        it('clips only the owning layer while ignoring nonnumeric deltas', () => {
          // Arrange
          const networkParts = createSingleInputDualOutputNetwork(204);
          const [firstLayerInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.firstOutputNode,
          );
          const [firstLayerSelfConnection] = networkParts.network.connect(
            networkParts.firstOutputNode,
            networkParts.firstOutputNode,
          );
          const [secondLayerInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.secondOutputNode,
          );
          const [crossLayerConnection] = networkParts.network.connect(
            networkParts.firstOutputNode,
            networkParts.secondOutputNode,
          );
          const [secondLayerSelfConnection] = networkParts.network.connect(
            networkParts.secondOutputNode,
            networkParts.secondOutputNode,
          );

          firstLayerInputConnection.totalDeltaWeight = 5;
          secondLayerInputConnection.totalDeltaWeight = 0.25;
          Reflect.set(firstLayerSelfConnection, 'totalDeltaWeight', undefined);
          Reflect.set(crossLayerConnection, 'totalDeltaWeight', undefined);
          Reflect.set(secondLayerSelfConnection, 'totalDeltaWeight', undefined);
          Reflect.set(networkParts.firstOutputNode, 'totalDeltaBias', undefined);
          Reflect.set(networkParts.secondOutputNode, 'totalDeltaBias', undefined);

          setLayerMetadata(networkParts.network, [
            { nodes: [networkParts.firstOutputNode] },
            { nodes: [networkParts.secondOutputNode] },
          ]);

          // Act
          applyGradientClippingCore(networkParts.network, {
            mode: 'layerwiseNorm',
          });

          // Assert
          expect({
            firstLayerDelta: firstLayerInputConnection.totalDeltaWeight,
            secondLayerDelta: secondLayerInputConnection.totalDeltaWeight,
          }).toEqual({
            firstLayerDelta: 1,
            secondLayerDelta: 0.25,
          });
        });
      });
    });

    describe('given layerwise norm clipping groups nodes without layer metadata', () => {
      describe('when fallback grouping encounters nonnumeric deltas', () => {
        it('keeps the fallback node count while leaving the small node group unchanged', () => {
          // Arrange
          const networkParts = createSingleInputDualOutputNetwork(205);
          const [firstNodeInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.firstOutputNode,
          );
          const [secondNodeInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.secondOutputNode,
          );
          const [crossNodeConnection] = networkParts.network.connect(
            networkParts.firstOutputNode,
            networkParts.secondOutputNode,
          );
          const [secondNodeSelfConnection] = networkParts.network.connect(
            networkParts.secondOutputNode,
            networkParts.secondOutputNode,
          );

          firstNodeInputConnection.totalDeltaWeight = 2;
          secondNodeInputConnection.totalDeltaWeight = 0.25;
          Reflect.set(crossNodeConnection, 'totalDeltaWeight', undefined);
          Reflect.set(secondNodeSelfConnection, 'totalDeltaWeight', undefined);
          Reflect.set(networkParts.secondOutputNode, 'totalDeltaBias', undefined);

          // Act
          applyGradientClippingCore(networkParts.network, {
            mode: 'layerwiseNorm',
          });

          // Assert
          expect({
            groupCount: getLastGradClipGroupCount(networkParts.network),
            secondNodeDelta: secondNodeInputConnection.totalDeltaWeight,
          }).toEqual({
            groupCount: 2,
            secondNodeDelta: 0.25,
          });
        });
      });
    });

    describe('given global percentile clipping receives zero-valued gradients', () => {
      describe('when clipping uses the default percentile', () => {
        it('keeps the zero threshold group unchanged while ignoring nonnumeric deltas', () => {
          // Arrange
          const networkParts = createSingleInputDualOutputNetwork(206);
          const [firstNodeInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.firstOutputNode,
          );
          const [secondNodeInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.secondOutputNode,
          );
          const [crossNodeConnection] = networkParts.network.connect(
            networkParts.firstOutputNode,
            networkParts.secondOutputNode,
          );
          const [secondNodeSelfConnection] = networkParts.network.connect(
            networkParts.secondOutputNode,
            networkParts.secondOutputNode,
          );

          firstNodeInputConnection.totalDeltaWeight = 0;
          secondNodeInputConnection.totalDeltaWeight = 0;
          Reflect.set(crossNodeConnection, 'totalDeltaWeight', undefined);
          Reflect.set(secondNodeSelfConnection, 'totalDeltaWeight', undefined);
          Reflect.set(networkParts.firstOutputNode, 'totalDeltaBias', undefined);
          Reflect.set(networkParts.secondOutputNode, 'totalDeltaBias', undefined);

          // Act
          applyGradientClippingCore(networkParts.network, {
            mode: 'percentile',
          });

          // Assert
          expect({
            firstNodeDelta: firstNodeInputConnection.totalDeltaWeight,
            groupCount: getLastGradClipGroupCount(networkParts.network),
            secondNodeDelta: secondNodeInputConnection.totalDeltaWeight,
          }).toEqual({
            firstNodeDelta: 0,
            groupCount: 1,
            secondNodeDelta: 0,
          });
        });
      });
    });

    describe('given layerwise norm clipping sees a node without numeric fallback gradients', () => {
      describe('when clipping groups nodes without layer metadata', () => {
        it('omits the empty fallback node group from the recorded group count', () => {
          // Arrange
          const networkParts = createSingleInputDualOutputNetwork(207);

          clearGradientState(networkParts.network);

          const [firstNodeInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.firstOutputNode,
          );
          const [secondNodeInputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.secondOutputNode,
          );
          const [crossNodeConnection] = networkParts.network.connect(
            networkParts.firstOutputNode,
            networkParts.secondOutputNode,
          );
          const [secondNodeSelfConnection] = networkParts.network.connect(
            networkParts.secondOutputNode,
            networkParts.secondOutputNode,
          );

          firstNodeInputConnection.totalDeltaWeight = 2;
          Reflect.set(secondNodeInputConnection, 'totalDeltaWeight', undefined);
          Reflect.set(crossNodeConnection, 'totalDeltaWeight', undefined);
          Reflect.set(secondNodeSelfConnection, 'totalDeltaWeight', undefined);
          Reflect.set(networkParts.secondOutputNode, 'totalDeltaBias', undefined);

          // Act
          applyGradientClippingCore(networkParts.network, {
            mode: 'layerwiseNorm',
          });

          // Assert
          expect(getLastGradClipGroupCount(networkParts.network)).toBe(1);
        });
      });
    });

    describe('given clipping receives an unsupported runtime mode', () => {
      describe('when no numeric global gradients are available', () => {
        it('records zero collected groups and leaves the network unchanged', () => {
          // Arrange
          const networkParts = createSingleInputOutputNetwork(208);

          clearGradientState(networkParts.network);

          const [inputConnection] = networkParts.network.connect(
            networkParts.inputNode,
            networkParts.outputNode,
          );
          const [selfConnection] = networkParts.network.connect(
            networkParts.outputNode,
            networkParts.outputNode,
          );
          const runtimeConfig = {
            mode: 'unsupported-runtime-mode',
          } as unknown as Parameters<typeof applyGradientClippingCore>[1];

          Reflect.set(inputConnection, 'totalDeltaWeight', undefined);
          Reflect.set(selfConnection, 'totalDeltaWeight', undefined);
          Reflect.set(networkParts.outputNode, 'totalDeltaBias', undefined);

          // Act
          applyGradientClippingCore(networkParts.network, runtimeConfig);

          // Assert
          expect(getLastGradClipGroupCount(networkParts.network)).toBe(0);
        });
      });
    });

    describe('given percentile clipping sees a collected group cleared before threshold evaluation', () => {
      describe('when the runtime config empties the collected values', () => {
        it('takes the empty-group threshold path without changing the gradients', () => {
          // Arrange
          const networkParts = createSingleInputOutputNetwork(209);
          const gradientState = seedGradientState({
            incomingDelta: 5,
            network: networkParts.network,
            outputBias: 0,
            outputNode: networkParts.outputNode,
            selfDelta: 0,
          });
          const capturedGroups: number[][] = [];
          const originalPush = Array.prototype.push;
          const pushSpy = jest.spyOn(Array.prototype, 'push');
          const runtimeConfig = {
            mode: 'percentile',
            get percentile() {
              capturedGroups.forEach((groupValues) => {
                groupValues.length = 0;
              });

              return 50;
            },
          } as unknown as Parameters<typeof applyGradientClippingCore>[1];

          pushSpy.mockImplementation(function (
            this: unknown[],
            ...values: unknown[]
          ): number {
            if (values.length === 1 && Array.isArray(values[0])) {
              capturedGroups[capturedGroups.length] = values[0] as number[];
            }

            return Reflect.apply(originalPush, this, values);
          });

          try {
            // Act
            applyGradientClippingCore(networkParts.network, runtimeConfig);
          } finally {
            pushSpy.mockRestore();
          }

          // Assert
          expect({
            groupCount: getLastGradClipGroupCount(networkParts.network),
            incomingDelta: gradientState.incomingConnection.totalDeltaWeight,
          }).toEqual({
            groupCount: 1,
            incomingDelta: 5,
          });
        });
      });
    });
  });
});