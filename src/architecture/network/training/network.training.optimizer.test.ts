import Network from '../network';
import { NetworkTrainingUnknownOptimizerTypeError } from './network.training.errors';

type OptimizerConnection = Network['connections'][number] & {
  firstMoment?: number;
  secondMoment?: number;
  gradientAccumulator?: number;
};

type OptimizerName = 'sgd' | 'rmsprop' | 'adagrad' | 'adam' | 'adamw';

const OPTIMIZER_NAMES: OptimizerName[] = [
  'sgd',
  'rmsprop',
  'adagrad',
  'adam',
  'adamw',
];

const SINGLE_SAMPLE_DATASET = [{ input: [1], output: [0] }];

function createSingleConnectionNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

describe('network training chapter', () => {
  describe('optimizer state tracking', () => {
    OPTIMIZER_NAMES.forEach((optimizerName, optimizerIndex) => {
      describe(`given ${optimizerName} runs for one batch`, () => {
        let initialWeight = 0;
        let optimizerConnection: OptimizerConnection;

        beforeAll(() => {
          const network = createSingleConnectionNetwork(170 + optimizerIndex);
          optimizerConnection = network.connections[0] as OptimizerConnection;
          initialWeight = optimizerConnection.weight;

          network.train(SINGLE_SAMPLE_DATASET, {
            iterations: 1,
            rate: 0.2,
            batchSize: 1,
            optimizer: optimizerName,
          });
        });

        describe('when the connection weight is inspected afterward', () => {
          it('updates the weight', () => {
            // Arrange
            const updatedWeight = optimizerConnection.weight;

            // Act
            const weightChanged = updatedWeight !== initialWeight;

            // Assert
            expect(weightChanged).toBe(true);
          });
        });

        if (optimizerName === 'adam' || optimizerName === 'adamw') {
          describe('when first-moment state is inspected afterward', () => {
            it('stores the first-moment accumulator', () => {
              // Arrange
              const firstMoment = optimizerConnection.firstMoment;

              // Act
              const hasFirstMoment = typeof firstMoment === 'number';

              // Assert
              expect(hasFirstMoment).toBe(true);
            });
          });

          describe('when second-moment state is inspected afterward', () => {
            it('stores the second-moment accumulator', () => {
              // Arrange
              const secondMoment = optimizerConnection.secondMoment;

              // Act
              const hasSecondMoment = typeof secondMoment === 'number';

              // Assert
              expect(hasSecondMoment).toBe(true);
            });
          });
        }

        if (optimizerName === 'rmsprop' || optimizerName === 'adagrad') {
          describe('when accumulator state is inspected afterward', () => {
            it('stores the gradient accumulator', () => {
              // Arrange
              const gradientAccumulator =
                optimizerConnection.gradientAccumulator;

              // Act
              const hasGradientAccumulator =
                typeof gradientAccumulator === 'number';

              // Assert
              expect(hasGradientAccumulator).toBe(true);
            });
          });
        }
      });
    });

    describe('given optimizer is an object without a type field', () => {
      describe('when training runs with a typeless optimizer object', () => {
        it('leaves the type as-is and throws because the type is unknown', () => {
          // Arrange – optimizer object with no type string → line 191 FALSE arm
          // (typeof optimizerConfig.type !== 'string', so toLowerCase is skipped)
          const network = createSingleConnectionNetwork(9_201);

          // Act & Assert – unknown type throws after the lowercasing branch is skipped
          expect(() =>
            network.train(SINGLE_SAMPLE_DATASET, {
              iterations: 1,
              rate: 0.01,
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              optimizer: {} as any,
            }),
          ).toThrow(NetworkTrainingUnknownOptimizerTypeError);
        });
      });
    });
  });
});
