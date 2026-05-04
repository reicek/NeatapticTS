import Network from '../network';

type MixedPrecisionConnection = Network['connections'][number] & {
  _fp32Weight?: number;
};

const LINEAR_DATASET = Array.from({ length: 8 }, (_, sampleIndex) => ({
  input: [sampleIndex],
  output: [2 * sampleIndex],
}));

function cloneNetworkParameters(
  sourceNetwork: Network,
  targetNetwork: Network,
): void {
  for (
    let connectionIndex = 0;
    connectionIndex < sourceNetwork.connections.length;
    connectionIndex++
  ) {
    targetNetwork.connections[connectionIndex].weight =
      sourceNetwork.connections[connectionIndex].weight;
  }

  for (let nodeIndex = 0; nodeIndex < sourceNetwork.nodes.length; nodeIndex++) {
    Reflect.set(
      targetNetwork.nodes[nodeIndex],
      'bias',
      Reflect.get(sourceNetwork.nodes[nodeIndex], 'bias'),
    );
  }
}

function createSingleConnectionNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

describe('network training chapter', () => {
  describe('gradient features', () => {
    describe('given accumulation steps mimic a larger batch update', () => {
      let accumulationDelta = 0;
      let largeBatchDelta = 0;

      beforeAll(() => {
        const accumulationNetwork = createSingleConnectionNetwork(180);
        const largeBatchNetwork = createSingleConnectionNetwork(181);
        cloneNetworkParameters(accumulationNetwork, largeBatchNetwork);

        const initialWeight = accumulationNetwork.connections[0].weight;
        const datasetSubset = LINEAR_DATASET.slice(0, 4);

        accumulationNetwork.train(datasetSubset, {
          iterations: 1,
          rate: 0.05,
          batchSize: 1,
          accumulationSteps: 4,
          optimizer: 'adam',
        });
        largeBatchNetwork.train(datasetSubset, {
          iterations: 1,
          rate: 0.05,
          batchSize: 4,
          accumulationSteps: 1,
          optimizer: 'adam',
        });

        accumulationDelta =
          accumulationNetwork.connections[0].weight - initialWeight;
        largeBatchDelta =
          largeBatchNetwork.connections[0].weight - initialWeight;
      });

      describe('when the update directions are compared', () => {
        it('keeps both updates moving in the same direction', () => {
          // Arrange
          const accumulationDirection = Math.sign(accumulationDelta);
          const largeBatchDirection = Math.sign(largeBatchDelta);

          // Act
          const directionsMatch = accumulationDirection === largeBatchDirection;

          // Assert
          expect(directionsMatch).toBe(true);
        });
      });

      describe('when the update magnitudes are compared', () => {
        it('keeps the magnitude ratio within a two-times band', () => {
          // Arrange
          const magnitudeRatio =
            Math.abs(accumulationDelta) /
            Math.max(1e-9, Math.abs(largeBatchDelta));

          // Act
          const staysWithinBand = magnitudeRatio < 2;

          // Assert
          expect(staysWithinBand).toBe(true);
        });
      });
    });

    describe('given global norm clipping runs during a large update', () => {
      describe('when the resulting weight is inspected', () => {
        it('keeps the weight magnitude bounded', () => {
          // Arrange
          const network = createSingleConnectionNetwork(182);

          network.train(LINEAR_DATASET, {
            iterations: 1,
            rate: 5,
            batchSize: 4,
            accumulationSteps: 1,
            optimizer: 'adam',
            gradientClip: { mode: 'norm', maxNorm: 0.01 },
          });

          // Act
          const weightMagnitude = Math.abs(network.connections[0].weight);

          // Assert
          expect(weightMagnitude).toBeLessThan(20);
        });
      });
    });

    describe('given percentile clipping runs during one optimizer step', () => {
      describe('when the resulting weight is inspected', () => {
        it('keeps the weight finite', () => {
          // Arrange
          const network = createSingleConnectionNetwork(183);

          network.train(LINEAR_DATASET, {
            iterations: 1,
            rate: 2,
            batchSize: 4,
            optimizer: 'adam',
            gradientClip: { mode: 'percentile', percentile: 90 },
          });

          // Act
          const updatedWeight = network.connections[0].weight;

          // Assert
          expect(Number.isFinite(updatedWeight)).toBe(true);
        });
      });
    });

    describe('given mixed precision runs with an explicit loss scale', () => {
      describe('when the connection state is inspected afterward', () => {
        it('stores an FP32 master weight copy', () => {
          // Arrange
          const network = createSingleConnectionNetwork(184);

          network.train(LINEAR_DATASET, {
            iterations: 1,
            rate: 0.01,
            batchSize: 4,
            optimizer: 'adam',
            mixedPrecision: { lossScale: 512 },
          });

          // Act
          const masterWeight = (
            network.connections[0] as MixedPrecisionConnection
          )._fp32Weight;

          // Assert
          expect(typeof masterWeight).toBe('number');
        });
      });
    });

    describe('given gradient clip has no mode, no maxNorm, and no percentile', () => {
      describe('when training completes without any clip mode applied', () => {
        it('runs one iteration without throwing even with an empty gradient clip config', () => {
          // Arrange – gradientClip with no mode/maxNorm/percentile → line 138 FALSE arm
          const network = createSingleConnectionNetwork(9_101);

          // Act & Assert
          expect(() =>
            network.train(LINEAR_DATASET, {
              iterations: 1,
              rate: 0.01,
              gradientClip: {},
            }),
          ).not.toThrow();
        });
      });
    });
  });
});
