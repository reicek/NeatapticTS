import Network from '../network';

type OptimizerConnection = Network['connections'][number] & {
  lookaheadShadowWeight?: number;
  secondMomentum?: number;
};

type TrainingDataset = Array<{ input: number[]; output: number[] }>;

function createDeterministicNetwork(seed: number): Network {
  const network = new Network(1, 1, { seed });

  network.connections.forEach((connection) => {
    connection.weight = 0.5;
  });

  network.nodes
    .filter((node) => node.type !== 'input')
    .forEach((node) => {
      node.bias = 0;
    });

  return network;
}

function createRegressionDataset(): TrainingDataset {
  return Array.from({ length: 5 }, (_, sampleIndex) => {
    return {
      input: [sampleIndex + 1],
      output: [2 * (sampleIndex + 1)],
    };
  });
}

function createSingleRegressionSample(): TrainingDataset {
  return [{ input: [1], output: [2] }];
}

function createZeroGradientDataset(): TrainingDataset {
  return [{ input: [0], output: [0] }];
}

describe('network training chapter', () => {
  describe('optimizer behavior', () => {
    describe('given adamw uses decoupled weight decay on a zero-gradient sample', () => {
      let initialWeight = 0;
      let updatedWeight = 0;

      beforeAll(() => {
        const network = createDeterministicNetwork(210);
        initialWeight = network.connections[0].weight;

        network.train(createZeroGradientDataset(), {
          iterations: 1,
          rate: 0.01,
          batchSize: 1,
          optimizer: { type: 'adamw', weightDecay: 0.1 },
        });

        updatedWeight = network.connections[0].weight;
      });

      describe('when the connection weight is inspected afterward', () => {
        it('decreases even without a gradient contribution', () => {
          // Arrange
          const weightWasDecayed = updatedWeight < initialWeight;

          // Act
          const didDecay = weightWasDecayed;

          // Assert
          expect(didDecay).toBe(true);
        });
      });
    });

    describe('given lion runs for a single optimizer step', () => {
      let absoluteWeightDelta = 0;

      beforeAll(() => {
        const network = createDeterministicNetwork(211);
        const initialWeight = network.connections[0].weight;

        network.train(createSingleRegressionSample(), {
          iterations: 1,
          rate: 0.01,
          error: 0,
          batchSize: 1,
          optimizer: { type: 'lion', beta1: 0.9, beta2: 0.99 },
          cost: {
            fn: (target: number[], output: number[]) => {
              return (output[0] - target[0]) ** 2;
            },
            calculate: (target: number[], output: number[]) => {
              return (output[0] - target[0]) ** 2;
            },
          },
        });

        absoluteWeightDelta = Math.abs(
          network.connections[0].weight - initialWeight,
        );
      });

      describe('when the first-step update magnitude is inspected', () => {
        it('matches the configured learning rate', () => {
          // Arrange
          const expectedStepMagnitude = 0.01;

          // Act
          const observedStepMagnitude = absoluteWeightDelta;

          // Assert
          expect(observedStepMagnitude).toBeCloseTo(expectedStepMagnitude, 12);
        });
      });
    });

    describe('given lookahead runs with an explicit sync cadence', () => {
      let syncedConnection: OptimizerConnection;

      beforeAll(() => {
        const network = createDeterministicNetwork(212);

        network.train(createRegressionDataset(), {
          iterations: 2,
          rate: 0.01,
          error: 0,
          batchSize: 1,
          optimizer: {
            type: 'lookahead',
            baseType: 'adam',
            la_k: 2,
            la_alpha: 0.5,
          },
          cost: {
            fn: (target: number[], output: number[]) => {
              return (output[0] - target[0]) ** 2;
            },
            calculate: (target: number[], output: number[]) => {
              return (output[0] - target[0]) ** 2;
            },
          },
        });

        syncedConnection = network.connections[0] as OptimizerConnection;
      });

      describe('when the optimizer state is inspected afterward', () => {
        it('stores a lookahead shadow weight', () => {
          // Arrange
          const shadowWeight = syncedConnection.lookaheadShadowWeight;

          // Act
          const hasShadowWeight = typeof shadowWeight === 'number';

          // Assert
          expect(hasShadowWeight).toBe(true);
        });
      });

      describe('when the fast weight is inspected on the sync step', () => {
        it('matches the shadow weight after blending', () => {
          // Arrange
          const shadowWeight = syncedConnection.lookaheadShadowWeight ?? NaN;

          // Act
          const fastWeight = syncedConnection.weight;

          // Assert
          expect(fastWeight).toBeCloseTo(shadowWeight, 12);
        });
      });
    });

    describe('given lookahead uses its default base optimizer settings', () => {
      let defaultLookaheadConnection: OptimizerConnection;

      beforeAll(() => {
        const network = createDeterministicNetwork(213);

        network.train(createRegressionDataset(), {
          iterations: 5,
          rate: 0.01,
          error: 0,
          batchSize: 1,
          optimizer: { type: 'lookahead' },
          cost: {
            fn: (target: number[], output: number[]) => {
              return (output[0] - target[0]) ** 2;
            },
            calculate: (target: number[], output: number[]) => {
              return (output[0] - target[0]) ** 2;
            },
          },
        });

        defaultLookaheadConnection = network
          .connections[0] as OptimizerConnection;
      });

      describe('when the first sync point is reached', () => {
        it('creates a default shadow weight', () => {
          // Arrange
          const shadowWeight = defaultLookaheadConnection.lookaheadShadowWeight;

          // Act
          const hasShadowWeight = typeof shadowWeight === 'number';

          // Assert
          expect(hasShadowWeight).toBe(true);
        });
      });
    });
  });
});
