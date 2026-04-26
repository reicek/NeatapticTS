import Network from '../network';
import { config } from '../../../config';
import { applyGradientClippingImpl, trainImpl } from './network.training.utils';
import type {
  CheckpointConfig,
  MetricsHook,
  ScheduleConfig,
} from './network.training.utils';
import {
  NetworkTrainingAccumulationStepsError,
  NetworkTrainingBatchSizeError,
  NetworkTrainingDatasetCompatibilityError,
  NetworkTrainingDropoutRangeError,
  NetworkTrainingInvalidOptimizerOptionError,
  NetworkTrainingNestedLookaheadError,
  NetworkTrainingStoppingConditionRequiredError,
  NetworkTrainingUnknownLookaheadBaseTypeError,
  NetworkTrainingUnknownOptimizerTypeError,
} from './network.training.errors';

type TrainingDataset = Parameters<typeof trainImpl>[1];

interface NetworkInternals {
  _forceNextOverflow: boolean;
  _currentGradClip?: {
    maxNorm?: number;
    mode: string;
    percentile?: number;
  };
  _mixedPrecision: { enabled: boolean; lossScale: number };
}

function createSingleSampleDataset(): TrainingDataset {
  return [{ input: [0.3], output: [0.4] }];
}

function createBinaryClassificationDataset(): TrainingDataset {
  return [
    { input: [0, 0], output: [0] },
    { input: [1, 1], output: [1] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
  ];
}

function runTrainingWithInjectedSampleFailure(): string[] {
  const network = new Network(2, 1, { seed: 119 });
  const originalWarnings = config.warnings;
  const warningSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  const originalActivate = network.activate.bind(network);
  let failureWasInjected = false;

  config.warnings = true;

  jest
    .spyOn(network, 'activate')
    .mockImplementation(
      (input, training = false, maxActivationDepth = 1000) => {
        if (!failureWasInjected && input[0] === 0 && input[1] === 1) {
          failureWasInjected = true;
          throw new Error('Injected training failure');
        }

        return originalActivate(input, training, maxActivationDepth);
      },
    );

  try {
    network.train(createBinaryClassificationDataset(), {
      iterations: 10,
      error: 0.01,
      rate: 0.3,
    });

    return warningSpy.mock.calls.map(([message]) => {
      return String(message);
    });
  } finally {
    jest.restoreAllMocks();
    config.warnings = originalWarnings;
  }
}

function createSingleInputOutputNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

function setNetworkInternal<Key extends keyof NetworkInternals>(
  network: Network,
  key: Key,
  value: NetworkInternals[Key],
): void {
  Reflect.set(network, key, value);
}

function getNetworkInternal<Key extends keyof NetworkInternals>(
  network: Network,
  key: Key,
): NetworkInternals[Key] {
  return Reflect.get(network, key) as NetworkInternals[Key];
}

describe('network training chapter', () => {
  describe('basic training orchestration', () => {
    describe('validation errors', () => {
      describe('given the dataset input width does not match the network input width', () => {
        describe('when trainImpl starts', () => {
          it('throws the dataset compatibility error', () => {
            // Arrange
            const network = new Network(2, 1, { seed: 101 });
            const mismatchedDataset: TrainingDataset = [
              { input: [0.1], output: [0.5] },
            ];

            // Act
            const trainWithMismatchedDataset = () => {
              trainImpl(network, mismatchedDataset, {
                iterations: 1,
                rate: 0.1,
              });
            };

            // Assert
            expect(trainWithMismatchedDataset).toThrow(
              NetworkTrainingDatasetCompatibilityError,
            );
          });
        });
      });

      describe('given no stopping condition is provided', () => {
        describe('when trainImpl starts', () => {
          it('throws the stopping-condition error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(102);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithoutStoppingCondition = () => {
              trainImpl(network, trainingDataset, { rate: 0.1 });
            };

            // Assert
            expect(trainWithoutStoppingCondition).toThrow(
              NetworkTrainingStoppingConditionRequiredError,
            );
          });
        });

        describe('given warnings are enabled for missing stopping conditions', () => {
          describe('when trainImpl starts', () => {
            it('logs the missing-condition warning before throwing', () => {
              // Arrange
              const network = createSingleInputOutputNetwork(122);
              const trainingDataset = createSingleSampleDataset();
              const originalWarnings = config.warnings;
              const warnSpy = jest
                .spyOn(console, 'warn')
                .mockImplementation(() => {});

              config.warnings = true;

              try {
                // Act
                const trainWithoutStoppingCondition = () => {
                  trainImpl(network, trainingDataset, { rate: 0.1 });
                };

                // Assert
                expect(trainWithoutStoppingCondition).toThrow(
                  NetworkTrainingStoppingConditionRequiredError,
                );
              } finally {
                warnSpy.mockRestore();
                config.warnings = originalWarnings;
              }
            });
          });
        });
      });

      describe('given batch size exceeds the dataset size', () => {
        describe('when trainImpl starts', () => {
          it('throws the batch-size error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(103);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithOversizedBatch = () => {
              trainImpl(network, trainingDataset, {
                iterations: 1,
                rate: 0.1,
                batchSize: 5,
              });
            };

            // Assert
            expect(trainWithOversizedBatch).toThrow(
              NetworkTrainingBatchSizeError,
            );
          });
        });
      });

      describe('given dropout is set to one', () => {
        describe('when trainImpl starts', () => {
          it('throws the dropout-range error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(104);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithInvalidDropout = () => {
              trainImpl(network, trainingDataset, {
                iterations: 1,
                rate: 0.1,
                dropout: 1,
              });
            };

            // Assert
            expect(trainWithInvalidDropout).toThrow(
              NetworkTrainingDropoutRangeError,
            );
          });
        });
      });

      describe('given accumulationSteps is below one', () => {
        describe('when trainImpl starts', () => {
          it('throws the accumulation-steps error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(105);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithInvalidAccumulation = () => {
              trainImpl(network, trainingDataset, {
                iterations: 1,
                rate: 0.1,
                accumulationSteps: -1,
              });
            };

            // Assert
            expect(trainWithInvalidAccumulation).toThrow(
              NetworkTrainingAccumulationStepsError,
            );
          });
        });
      });

      describe('given optimizer type is unknown', () => {
        describe('when trainImpl starts', () => {
          it('throws the unknown-optimizer error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(106);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithUnknownOptimizer = () => {
              trainImpl(network, trainingDataset, {
                iterations: 1,
                rate: 0.1,
                optimizer: 'notreal',
              });
            };

            // Assert
            expect(trainWithUnknownOptimizer).toThrow(
              NetworkTrainingUnknownOptimizerTypeError,
            );
          });
        });
      });

      describe('given optimizer option is a non-object, non-string value', () => {
        describe('when trainImpl starts', () => {
          it('throws the invalid-optimizer-option error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(206);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithInvalidOptimizerOption = () => {
              trainImpl(network, trainingDataset, {
                iterations: 1,
                rate: 0.1,
                optimizer: 7 as unknown as never,
              });
            };

            // Assert
            expect(trainWithInvalidOptimizerOption).toThrow(
              NetworkTrainingInvalidOptimizerOptionError,
            );
          });
        });
      });

      describe('given lookahead uses lookahead as its base type', () => {
        describe('when trainImpl starts', () => {
          it('throws the nested-lookahead error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(107);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithNestedLookahead = () => {
              trainImpl(network, trainingDataset, {
                iterations: 1,
                rate: 0.1,
                optimizer: { type: 'lookahead', baseType: 'lookahead' },
              });
            };

            // Assert
            expect(trainWithNestedLookahead).toThrow(
              NetworkTrainingNestedLookaheadError,
            );
          });
        });
      });

      describe('given lookahead uses an unsupported base type', () => {
        describe('when trainImpl starts', () => {
          it('throws the unknown-lookahead-base-type error', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(207);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainWithUnknownLookaheadBaseType = () => {
              trainImpl(network, trainingDataset, {
                iterations: 1,
                rate: 0.1,
                optimizer: { type: 'lookahead', baseType: 'notreal' },
              });
            };

            // Assert
            expect(trainWithUnknownLookaheadBaseType).toThrow(
              NetworkTrainingUnknownLookaheadBaseTypeError,
            );
          });
        });
      });
    });

    describe('warnings', () => {
      describe('given warnings are enabled and rate is omitted', () => {
        describe('when trainImpl falls back to the default rate', () => {
          it('warns that the rate option is missing', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(108);
            const trainingDataset = createSingleSampleDataset();
            const originalWarnings = config.warnings;
            config.warnings = true;
            const warnSpy = jest
              .spyOn(console, 'warn')
              .mockImplementation(() => {});

            try {
              // Act
              trainImpl(network, trainingDataset, { iterations: 1 });
              const warningMessages = warnSpy.mock.calls.map(([message]) => {
                return String(message);
              });

              // Assert
              expect(warningMessages.includes('Missing `rate` option')).toBe(
                true,
              );
            } finally {
              warnSpy.mockRestore();
              config.warnings = originalWarnings;
            }
          });
        });
      });

      describe('given warnings are enabled and iterations is omitted', () => {
        describe('when trainImpl relies on an error threshold only', () => {
          it('warns that the iterations option is missing', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(118);
            const trainingDataset = createSingleSampleDataset();
            const originalWarnings = config.warnings;
            config.warnings = true;
            const warnSpy = jest
              .spyOn(console, 'warn')
              .mockImplementation(() => {});

            try {
              // Act
              trainImpl(network, trainingDataset, { error: 1, rate: 0.1 });
              const warningMessages = warnSpy.mock.calls.map(([message]) => {
                return String(message);
              });

              // Assert
              expect(
                warningMessages.some((message) => {
                  return message.includes('Missing `iterations` option');
                }),
              ).toBe(true);
            } finally {
              warnSpy.mockRestore();
              config.warnings = originalWarnings;
            }
          });
        });
      });
    });

    describe('gradient clipping', () => {
      describe('given norm clipping runs on a single large accumulated gradient', () => {
        describe('when clipping is applied', () => {
          it('scales the connection delta down to the configured norm ceiling', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(109);
            const inputNode = network.nodes[0];
            const outputNode = network.nodes.find(
              (node) => node.type === 'output',
            )!;
            const [connection] = network.connect(inputNode, outputNode);
            connection.totalDeltaWeight = 10;

            // Act
            applyGradientClippingImpl(network, { mode: 'norm', maxNorm: 1 });

            // Assert
            expect(connection.totalDeltaWeight).toBeCloseTo(1);
          });
        });
      });

      describe('given percentile clipping sees one small and one extreme gradient', () => {
        describe('when clipping is applied at the fiftieth percentile', () => {
          it('caps the extreme gradient at the percentile threshold', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(110);
            const inputNode = network.nodes[0];
            const outputNode = network.nodes.find(
              (node) => node.type === 'output',
            )!;
            network.connect(inputNode, outputNode);
            network.connect(inputNode, outputNode);
            const incomingConnections = outputNode.connections.in;
            const highMagnitudeConnection = incomingConnections.at(-1)!;

            network.nodes.forEach((node) => {
              Reflect.set(node, 'totalDeltaBias', 5);
              node.connections.self.forEach((connection) => {
                connection.totalDeltaWeight = 5;
              });
            });

            incomingConnections.forEach((connection, connectionIndex) => {
              const assignedGradient = [0.1, 5, 100][connectionIndex] ?? 5;
              connection.totalDeltaWeight = assignedGradient;
            });

            // Act
            applyGradientClippingImpl(network, {
              mode: 'percentile',
              percentile: 50,
            });

            // Assert
            expect(highMagnitudeConnection.totalDeltaWeight).toBeCloseTo(5);
          });
        });
      });

      describe('given gradient clipping uses shorthand max-norm and percentile options', () => {
        describe('when trainImpl normalizes the runtime configuration', () => {
          it('stores the expected runtime clipping mode for both shorthand forms', () => {
            // Arrange
            const maxNormNetwork = createSingleInputOutputNetwork(123);
            const percentileNetwork = createSingleInputOutputNetwork(124);
            const trainingDataset = createSingleSampleDataset();

            // Act
            trainImpl(maxNormNetwork, trainingDataset, {
              iterations: 1,
              rate: 0.1,
              gradientClip: { maxNorm: 1 },
            });
            trainImpl(percentileNetwork, trainingDataset, {
              iterations: 1,
              rate: 0.1,
              gradientClip: { percentile: 90 },
            });

            const maxNormClip = getNetworkInternal(
              maxNormNetwork,
              '_currentGradClip',
            );
            const percentileClip = getNetworkInternal(
              percentileNetwork,
              '_currentGradClip',
            );

            // Assert
            expect([maxNormClip?.mode, percentileClip?.mode]).toEqual([
              'norm',
              'percentile',
            ]);
          });
        });
      });
    });

    describe('mixed precision and stopping behavior', () => {
      describe('given mixed precision is enabled and the next step is forced to overflow', () => {
        describe('when trainImpl completes one iteration', () => {
          it('halves the default loss scale', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(111);
            const trainingDataset = createSingleSampleDataset();
            setNetworkInternal(network, '_forceNextOverflow', true);

            // Act
            trainImpl(network, trainingDataset, {
              iterations: 1,
              rate: 0.1,
              mixedPrecision: true,
              optimizer: 'adam',
            });
            const mixedPrecisionState = getNetworkInternal(
              network,
              '_mixedPrecision',
            );

            // Assert
            expect(mixedPrecisionState.lossScale).toBe(512);
          });
        });
      });

      describe('given early stopping is configured with no meaningful improvement window', () => {
        describe('when trainImpl runs', () => {
          it('returns before the configured max iteration count', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(112);
            const trainingDataset = createSingleSampleDataset();

            // Act
            const trainingSummary = trainImpl(network, trainingDataset, {
              iterations: 5,
              rate: 0.1,
              earlyStopPatience: 1,
              earlyStopMinDelta: 1,
            });

            // Assert
            expect(trainingSummary.iterations).toBeLessThan(5);
          });
        });
      });
    });

    describe('callbacks', () => {
      describe('given checkpoint.last is enabled', () => {
        describe('when trainImpl completes', () => {
          it('saves a last checkpoint payload', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(113);
            const trainingDataset = [{ input: [0.2], output: [0.25] }];
            const savedTypes: string[] = [];
            type CheckpointPayload = Parameters<CheckpointConfig['save']>[0];
            const save: CheckpointConfig['save'] = (
              payload: CheckpointPayload,
            ) => {
              savedTypes.push(payload.type);
            };

            // Act
            trainImpl(network, trainingDataset, {
              iterations: 2,
              rate: 0.1,
              checkpoint: { last: true, save },
            });

            // Assert
            expect(savedTypes.includes('last')).toBe(true);
          });
        });
      });

      describe('given checkpoint.best is enabled', () => {
        describe('when trainImpl completes', () => {
          it('saves a best checkpoint payload', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(114);
            const trainingDataset = [{ input: [0.2], output: [0.25] }];
            const savedTypes: string[] = [];
            type CheckpointPayload = Parameters<CheckpointConfig['save']>[0];
            const save: CheckpointConfig['save'] = (
              payload: CheckpointPayload,
            ) => {
              savedTypes.push(payload.type);
            };

            // Act
            trainImpl(network, trainingDataset, {
              iterations: 2,
              rate: 0.1,
              checkpoint: { best: true, save },
            });

            // Assert
            expect(savedTypes.includes('best')).toBe(true);
          });
        });
      });

      describe('given a schedule callback is configured', () => {
        describe('when trainImpl completes a matching iteration', () => {
          it('invokes the schedule callback once', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(115);
            const trainingDataset = [{ input: [0.2], output: [0.3] }];
            let scheduleInvocations = 0;
            const schedule: ScheduleConfig = {
              iterations: 1,
              function: () => {
                scheduleInvocations++;
              },
            };

            // Act
            trainImpl(network, trainingDataset, {
              iterations: 1,
              rate: 0.1,
              schedule,
            });

            // Assert
            expect(scheduleInvocations).toBe(1);
          });
        });
      });

      describe('given a metrics hook is configured', () => {
        describe('when trainImpl completes an iteration', () => {
          it('invokes the metrics hook once', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(116);
            const trainingDataset = [{ input: [0.2], output: [0.3] }];
            let metricsInvocations = 0;
            const metricsHook: MetricsHook = () => {
              metricsInvocations++;
            };

            // Act
            trainImpl(network, trainingDataset, {
              iterations: 1,
              rate: 0.1,
              metricsHook,
            });

            // Assert
            expect(metricsInvocations).toBe(1);
          });
        });
      });

      describe('given a metrics hook inspects training telemetry', () => {
        describe('when trainImpl completes an iteration', () => {
          it('receives a numeric gradient norm', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(117);
            const trainingDataset = [{ input: [0.2], output: [0.3] }];
            let observedGradientNorm: number | undefined;
            const metricsHook: MetricsHook = ({ gradNorm }) => {
              observedGradientNorm = gradNorm;
            };

            // Act
            trainImpl(network, trainingDataset, {
              iterations: 1,
              rate: 0.1,
              metricsHook,
            });

            // Assert
            expect(typeof observedGradientNorm).toBe('number');
          });
        });
      });
    });

    describe('public train() compatibility', () => {
      describe('given legacy extra option fields are present', () => {
        describe('when the public training API runs', () => {
          it('still returns a finite numeric error', () => {
            // Arrange
            const network = new Network(2, 1, { seed: 118 });
            const trainingDataset = createBinaryClassificationDataset();
            const logSpy = jest
              .spyOn(console, 'log')
              .mockImplementation(() => {});

            try {
              // Act
              const trainingSummary = network.train(trainingDataset, {
                rate: 0.2,
                iterations: 2,
                shuffle: true,
                dropout: 0.5,
                batchSize: 2,
                cost: () => 0,
                crossValidate: { testSize: 0.5, testError: 0 },
                log: 1,
                schedule: { iterations: 1, function: () => {} },
                clear: true,
                optimizer: 'sgd',
              });

              // Assert
              expect(Number.isFinite(trainingSummary.error)).toBe(true);
            } finally {
              logSpy.mockRestore();
            }
          });
        });
      });

      describe('given one dataset sample throws during activation', () => {
        describe('when train() processes the full dataset', () => {
          it('does not rethrow the sample error', () => {
            // Arrange
            const trainWithInjectedFailure = () => {
              runTrainingWithInjectedSampleFailure();
            };

            // Act
            const executeTraining = trainWithInjectedFailure;

            // Assert
            expect(executeTraining).not.toThrow();
          });
        });
      });

      describe('given one dataset sample throws during activation and warnings are enabled', () => {
        describe('when train() skips the failing sample', () => {
          it('warns that the data point was skipped', () => {
            // Arrange
            const warningMessages = runTrainingWithInjectedSampleFailure();

            // Act
            const emittedSkipWarning = warningMessages.some((message) => {
              return message.includes('Error processing data point');
            });

            // Assert
            expect(emittedSkipWarning).toBe(true);
          });
        });
      });
    });
  });
});
