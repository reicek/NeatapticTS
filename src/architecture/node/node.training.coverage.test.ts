import Connection from '../connection/connection';
import Activation from '../../methods/activation/activation';
import Node from './node';

function withSuppressedWarnings<T>(
  run: (warningSpy: jest.SpyInstance) => T,
): T {
  const warningSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

  try {
    return run(warningSpy);
  } finally {
    warningSpy.mockRestore();
  }
}

function withIgnoredWriteProperties<T>(
  targets: Array<{ propertyName: string; targetObject: object }>,
  run: () => T,
): T {
  const originalDescriptors = targets.map(({ propertyName, targetObject }) => ({
    originalDescriptor: Object.getOwnPropertyDescriptor(
      targetObject,
      propertyName,
    ),
    propertyName,
    targetObject,
  }));

  try {
    for (const { propertyName, targetObject } of targets) {
      Object.defineProperty(targetObject, propertyName, {
        configurable: true,
        get() {
          return undefined;
        },
        set() {
          return undefined;
        },
      });
    }

    return run();
  } finally {
    for (const {
      originalDescriptor,
      propertyName,
      targetObject,
    } of originalDescriptors) {
      if (originalDescriptor) {
        Object.defineProperty(targetObject, propertyName, originalDescriptor);
      } else {
        Reflect.deleteProperty(targetObject, propertyName);
      }
    }
  }
}

function createPropagationHarness(nodeType: string = 'output') {
  const trainingNode = new Node(nodeType);
  const sourceNode = new Node('input');
  const incomingConnection = sourceNode.connect(trainingNode, 1)[0];
  const selfConnection = trainingNode.connect(trainingNode, 1)[0];

  trainingNode.squash = Activation.identity;
  trainingNode.activation = 0;
  trainingNode.bias = 0;
  trainingNode.derivative = 1;
  trainingNode.mask = 1;
  sourceNode.activation = 1;
  incomingConnection.eligibility = 1;
  incomingConnection.xtrace = { nodes: [], values: [] };
  selfConnection.eligibility = 1;
  selfConnection.xtrace = { nodes: [], values: [] };

  return {
    trainingNode,
    sourceNode,
    incomingConnection,
    selfConnection,
  };
}

function createBatchOptimizerHarness(nodeType: string = 'hidden') {
  const trainingNode = new Node(nodeType);
  const sourceNode = new Node('input');
  const incomingConnection = sourceNode.connect(trainingNode, 1)[0];
  const selfConnection = trainingNode.connect(trainingNode, -1)[0];

  trainingNode.bias = 0;
  trainingNode.totalDeltaBias = 0;
  trainingNode.previousDeltaBias = 0;
  incomingConnection.totalDeltaWeight = 0;
  incomingConnection.previousDeltaWeight = 0;
  selfConnection.totalDeltaWeight = 0;
  selfConnection.previousDeltaWeight = 0;

  return {
    trainingNode,
    incomingConnection,
    selfConnection,
  };
}

type OptimizerOptions = Parameters<Node['applyBatchUpdatesWithOptimizer']>[0];

function runRegularizationCase(
  regularization:
    | number
    | { type: 'L1' | 'L2'; lambda: number }
    | ((weight: number) => number),
) {
  const { trainingNode, incomingConnection, selfConnection } =
    createPropagationHarness();

  trainingNode.activation = 1;
  incomingConnection.weight = 0.5;
  incomingConnection.eligibility = 2;
  selfConnection.weight = -0.5;
  selfConnection.eligibility = 3;

  trainingNode.propagate(1, 0, false, regularization, 2);

  return {
    incomingDeltaWeight: incomingConnection.totalDeltaWeight,
    selfDeltaWeight: selfConnection.totalDeltaWeight,
  };
}

function runOptimizerCase(optimizerOptions: OptimizerOptions) {
  const { trainingNode, incomingConnection, selfConnection } =
    createBatchOptimizerHarness('hidden');
  const optimizerNode = trainingNode as unknown as Record<string, number>;

  trainingNode.bias = 0.5;
  trainingNode.totalDeltaBias = 2;
  incomingConnection.totalDeltaWeight = 2;
  selfConnection.totalDeltaWeight = 2;

  trainingNode.applyBatchUpdatesWithOptimizer(optimizerOptions);

  return {
    biasAccumulatorCleared: trainingNode.totalDeltaBias === 0,
    biasFinite: Number.isFinite(trainingNode.bias),
    hasBiasFirstMoment: typeof optimizerNode.opt_mB === 'number',
    hasBiasInfinityNorm: typeof optimizerNode.opt_uB === 'number',
    hasBiasMaxSecondMoment: typeof optimizerNode.opt_vhatB === 'number',
    hasBiasSecondMoment: typeof optimizerNode.opt_vB === 'number',
    hasBiasSecondMomentum: typeof optimizerNode.opt_mB2 === 'number',
    hasConnectionFirstMoment:
      typeof incomingConnection.firstMoment === 'number',
    hasConnectionGradientAccumulator:
      typeof incomingConnection.gradientAccumulator === 'number',
    hasConnectionInfinityNorm:
      typeof incomingConnection.infinityNorm === 'number',
    hasConnectionMaxSecondMoment:
      typeof incomingConnection.maxSecondMoment === 'number',
    hasConnectionSecondMoment:
      typeof incomingConnection.secondMoment === 'number',
    hasConnectionSecondMomentum:
      typeof incomingConnection.secondMomentum === 'number',
    incomingAccumulatorCleared: incomingConnection.totalDeltaWeight === 0,
    incomingWeightChanged: incomingConnection.weight !== 1,
  };
}

describe('Node', () => {
  describe('training coverage helpers', () => {
    describe('given a hidden node that gates a recurrent target', () => {
      describe('when propagating the error signal', () => {
        it('combines projected and gated responsibility contributions', () => {
          // Arrange
          const gatingNode = new Node('hidden');
          const projectedTargetNode = new Node('hidden');
          const gateSourceNode = new Node('input');
          const gatedTargetNode = new Node('hidden');
          const projectedConnection = gatingNode.connect(
            projectedTargetNode,
            7,
          )[0];
          const gatedConnection = gateSourceNode.connect(gatedTargetNode, 4)[0];
          const gatedSelfConnection = gatedTargetNode.connect(
            gatedTargetNode,
            0.1,
          )[0];
          const ungatedSelfConnection = new Connection(
            gatedTargetNode,
            gatedTargetNode,
            0.2,
          );
          gatingNode.derivative = 1;
          projectedTargetNode.error.responsibility = 2;
          projectedConnection.gain = 0.5;
          gateSourceNode.activation = 3;
          gatedTargetNode.error.responsibility = 5;
          gatedTargetNode.old = 2;
          gatedTargetNode.connections.self.push(ungatedSelfConnection);
          gatingNode.gate(gatedConnection);
          gatedSelfConnection.gater = gatingNode;

          // Act
          gatingNode.propagate(1, 0, false, 0);

          // Assert
          expect({
            gated: gatingNode.error.gated,
            projected: gatingNode.error.projected,
            responsibility: gatingNode.error.responsibility,
          }).toStrictEqual({
            gated: 70,
            projected: 7,
            responsibility: 77,
          });
        });
      });
    });

    describe('given a constant node with accumulated gradients', () => {
      describe('when propagating the error signal', () => {
        it('returns before updating incoming, self, or bias accumulators', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createPropagationHarness('constant');
          trainingNode.derivative = 1;
          trainingNode.totalDeltaBias = 4;
          incomingConnection.totalDeltaWeight = 2;
          selfConnection.totalDeltaWeight = 3;

          // Act
          trainingNode.propagate(1, 0, false, 0);

          // Assert
          expect({
            biasAccumulator: trainingNode.totalDeltaBias,
            incomingDeltaWeight: incomingConnection.totalDeltaWeight,
            selfDeltaWeight: selfConnection.totalDeltaWeight,
          }).toStrictEqual({
            biasAccumulator: 4,
            incomingDeltaWeight: 2,
            selfDeltaWeight: 3,
          });
        });
      });
    });

    describe('given drop-connected incoming and self links', () => {
      describe('when propagating without an immediate update', () => {
        it('skips both gradient accumulation paths', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createPropagationHarness();
          trainingNode.activation = 1;
          incomingConnection.dcMask = 0;
          selfConnection.dcMask = 0;

          // Act
          trainingNode.propagate(1, 0, false, 0, 2);

          // Assert
          expect({
            incomingDeltaWeight: incomingConnection.totalDeltaWeight,
            selfDeltaWeight: selfConnection.totalDeltaWeight,
          }).toStrictEqual({
            incomingDeltaWeight: 0,
            selfDeltaWeight: 0,
          });
        });
      });
    });

    describe('given each supported regularization shape', () => {
      describe('when propagating an output node', () => {
        it('applies the matching penalty formula to incoming and self weights', () => {
          // Arrange
          const customCase = runRegularizationCase(() => 0.25);
          const l1Case = runRegularizationCase({ type: 'L1', lambda: 0.25 });
          const l2Case = runRegularizationCase({ type: 'L2', lambda: 2 });
          const numericCase = runRegularizationCase(2);

          // Act
          const regularizationSummary = {
            customCase,
            l1Case,
            l2Case,
            numericCase,
          };

          // Assert
          expect(regularizationSummary).toStrictEqual({
            customCase: {
              incomingDeltaWeight: 1.75,
              selfDeltaWeight: 2.75,
            },
            l1Case: {
              incomingDeltaWeight: 1.75,
              selfDeltaWeight: 3.25,
            },
            l2Case: {
              incomingDeltaWeight: 1,
              selfDeltaWeight: 4,
            },
            numericCase: {
              incomingDeltaWeight: 1,
              selfDeltaWeight: 4,
            },
          });
        });
      });
    });

    describe('given incoming and self xtrace entries', () => {
      describe('when propagating an output node', () => {
        it('adds the xtrace contributions into both gradients', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createPropagationHarness();
          const incomingTraceNode = new Node('hidden');
          const selfTraceNode = new Node('hidden');
          trainingNode.activation = 1;
          incomingConnection.eligibility = 2;
          incomingConnection.xtrace = {
            nodes: [incomingTraceNode],
            values: [0.5],
          };
          selfConnection.eligibility = 3;
          selfConnection.xtrace = {
            nodes: [selfTraceNode],
            values: [0.5],
          };
          incomingTraceNode.error.responsibility = 5;
          selfTraceNode.error.responsibility = 6;

          // Act
          trainingNode.propagate(1, 0, false, 0, 2);

          // Assert
          expect({
            incomingDeltaWeight: incomingConnection.totalDeltaWeight,
            selfDeltaWeight: selfConnection.totalDeltaWeight,
          }).toStrictEqual({
            incomingDeltaWeight: 4.5,
            selfDeltaWeight: 6,
          });
        });
      });
    });

    describe('given default regularization and in-range updates', () => {
      describe('when propagating with an immediate update and no momentum', () => {
        it('takes the unclamped update path for incoming, self, and bias values', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createPropagationHarness();
          trainingNode.activation = 1;
          trainingNode.bias = 0;
          incomingConnection.weight = 0.5;
          incomingConnection.eligibility = 2;
          selfConnection.weight = 0.5;
          selfConnection.eligibility = 3;

          // Act
          trainingNode.propagate(1, 0, true, undefined, 2);

          // Assert
          expect({
            bias: trainingNode.bias,
            incomingPreviousDeltaWeight: incomingConnection.previousDeltaWeight,
            incomingWeight: incomingConnection.weight,
            previousDeltaBias: trainingNode.previousDeltaBias,
            selfPreviousDeltaWeight: selfConnection.previousDeltaWeight,
            selfWeight: selfConnection.weight,
          }).toStrictEqual({
            bias: 1,
            incomingPreviousDeltaWeight: 2,
            incomingWeight: 2.5,
            previousDeltaBias: 1,
            selfPreviousDeltaWeight: 3,
            selfWeight: 3.5,
          });
        });
      });
    });

    describe('given an object regularization type outside L1 and L2', () => {
      describe('when propagating an output node', () => {
        it('falls back to a zero penalty term for incoming and self weights', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createPropagationHarness();
          const unsupportedRegularization = {
            type: 'none',
            lambda: 10,
          } as unknown as { type: 'L1' | 'L2'; lambda: number };
          trainingNode.activation = 1;
          incomingConnection.eligibility = 2;
          selfConnection.eligibility = 3;

          // Act
          trainingNode.propagate(1, 0, false, unsupportedRegularization, 2);

          // Assert
          expect({
            incomingDeltaWeight: incomingConnection.totalDeltaWeight,
            selfDeltaWeight: selfConnection.totalDeltaWeight,
          }).toStrictEqual({
            incomingDeltaWeight: 2,
            selfDeltaWeight: 3,
          });
        });
      });
    });

    describe('given non-finite propagate inputs and accumulators', () => {
      describe('when applying the update immediately', () => {
        it('resets incoming, self, and bias state back to zero', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createPropagationHarness();
          trainingNode.activation = 0;
          trainingNode.bias = Number.POSITIVE_INFINITY;
          trainingNode.previousDeltaBias = Number.POSITIVE_INFINITY;
          trainingNode.totalDeltaBias = Number.POSITIVE_INFINITY;
          incomingConnection.weight = Number.POSITIVE_INFINITY;
          incomingConnection.previousDeltaWeight = Number.POSITIVE_INFINITY;
          incomingConnection.totalDeltaWeight = Number.POSITIVE_INFINITY;
          selfConnection.weight = Number.POSITIVE_INFINITY;
          selfConnection.previousDeltaWeight = Number.POSITIVE_INFINITY;
          selfConnection.totalDeltaWeight = Number.POSITIVE_INFINITY;

          // Act
          const resetResult = withSuppressedWarnings((warningSpy) => {
            trainingNode.propagate(1, 1, true, 0, Number.POSITIVE_INFINITY);

            return {
              bias: trainingNode.bias,
              incomingPreviousDeltaWeight:
                incomingConnection.previousDeltaWeight,
              incomingTotalDeltaWeight: incomingConnection.totalDeltaWeight,
              incomingWeight: incomingConnection.weight,
              previousDeltaBias: trainingNode.previousDeltaBias,
              selfPreviousDeltaWeight: selfConnection.previousDeltaWeight,
              selfTotalDeltaWeight: selfConnection.totalDeltaWeight,
              selfWeight: selfConnection.weight,
              totalDeltaBias: trainingNode.totalDeltaBias,
              warningCount: warningSpy.mock.calls.length,
            };
          });

          // Assert
          expect(resetResult).toStrictEqual({
            bias: 0,
            incomingPreviousDeltaWeight: 0,
            incomingTotalDeltaWeight: 0,
            incomingWeight: 0,
            previousDeltaBias: 0,
            selfPreviousDeltaWeight: 0,
            selfTotalDeltaWeight: 0,
            selfWeight: 0,
            totalDeltaBias: 0,
            warningCount: 12,
          });
        });
      });
    });

    describe('given very large but finite propagate inputs', () => {
      describe('when applying the update immediately', () => {
        it('clips incoming, self, and bias updates to the configured limits', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createPropagationHarness();
          trainingNode.activation = 0;
          trainingNode.bias = 999_999.5;
          trainingNode.previousDeltaBias = 20;
          incomingConnection.weight = 999_999.5;
          incomingConnection.previousDeltaWeight = 20;
          selfConnection.weight = 999_999.5;
          selfConnection.previousDeltaWeight = 20;

          // Act
          trainingNode.propagate(1, 1, true, 0, 2_000);

          // Assert
          expect({
            bias: trainingNode.bias,
            incomingPreviousDeltaWeight: incomingConnection.previousDeltaWeight,
            incomingWeight: incomingConnection.weight,
            previousDeltaBias: trainingNode.previousDeltaBias,
            selfPreviousDeltaWeight: selfConnection.previousDeltaWeight,
            selfWeight: selfConnection.weight,
          }).toStrictEqual({
            bias: 1_000_000,
            incomingPreviousDeltaWeight: 1_000,
            incomingWeight: 1_000_000,
            previousDeltaBias: 1_000,
            selfPreviousDeltaWeight: 1_000,
            selfWeight: 1_000_000,
          });
        });
      });
    });

    describe('given accumulated batch deltas', () => {
      describe('when applying classic batch updates', () => {
        it('delegates to the SGD optimizer path', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createBatchOptimizerHarness('hidden');
          trainingNode.totalDeltaBias = 4;
          incomingConnection.totalDeltaWeight = 2;
          selfConnection.totalDeltaWeight = 3;

          // Act
          trainingNode.applyBatchUpdates(0.5);

          // Assert
          expect({
            bias: trainingNode.bias,
            incomingPreviousDeltaWeight: incomingConnection.previousDeltaWeight,
            incomingWeight: incomingConnection.weight,
            previousDeltaBias: trainingNode.previousDeltaBias,
            selfPreviousDeltaWeight: selfConnection.previousDeltaWeight,
            selfWeight: selfConnection.weight,
          }).toStrictEqual({
            bias: 4,
            incomingPreviousDeltaWeight: 2,
            incomingWeight: 3,
            previousDeltaBias: 4,
            selfPreviousDeltaWeight: 3,
            selfWeight: 2,
          });
        });
      });
    });

    describe('given optimizer type is omitted', () => {
      describe('when applying batch updates with optimizer options', () => {
        it('falls back to the default SGD path', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createBatchOptimizerHarness('hidden');
          trainingNode.bias = 0;
          trainingNode.totalDeltaBias = 3;
          incomingConnection.weight = 0;
          incomingConnection.totalDeltaWeight = 2;
          selfConnection.weight = 0;
          selfConnection.totalDeltaWeight = 0;

          // Act
          trainingNode.applyBatchUpdatesWithOptimizer(
            {} as unknown as OptimizerOptions,
          );

          // Assert
          expect({
            bias: trainingNode.bias,
            incomingWeight: incomingConnection.weight,
            selfWeight: selfConnection.weight,
          }).toStrictEqual({
            bias: 3,
            incomingWeight: 2,
            selfWeight: 0,
          });
        });
      });
    });

    describe('given the adaptive optimizer variants', () => {
      describe('when applying one batch step for each variant', () => {
        it('initializes the expected optimizer state and clears the accumulators', () => {
          // Arrange
          const optimizerSummary = {
            rmsprop: runOptimizerCase({ type: 'rmsprop' }),
            adagrad: runOptimizerCase({ type: 'adagrad' }),
            adam: runOptimizerCase({ type: 'adam' }),
            adamw: runOptimizerCase({ type: 'adamw', weightDecay: 0.1 }),
            amsgrad: runOptimizerCase({ type: 'amsgrad' }),
            adamax: runOptimizerCase({ type: 'adamax' }),
            nadam: runOptimizerCase({ type: 'nadam' }),
            radamRectified: runOptimizerCase({ type: 'radam', t: 10 }),
            radamWarmup: runOptimizerCase({ type: 'radam', t: 1 }),
            lion: runOptimizerCase({ type: 'lion' }),
            adabelief: runOptimizerCase({ type: 'adabelief' }),
          };

          // Act
          const summarizedOptimizers = optimizerSummary;

          // Assert
          expect(summarizedOptimizers).toStrictEqual({
            rmsprop: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: false,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: false,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: false,
              hasConnectionGradientAccumulator: true,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: false,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            adagrad: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: false,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: false,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: false,
              hasConnectionGradientAccumulator: true,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: false,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            adam: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: true,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            adamw: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: true,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            amsgrad: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: true,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: true,
              hasConnectionSecondMoment: true,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            adamax: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: true,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: true,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: false,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            nadam: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: true,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            radamRectified: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: true,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            radamWarmup: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: true,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            lion: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: true,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: false,
              hasConnectionSecondMomentum: true,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
            adabelief: {
              biasAccumulatorCleared: true,
              biasFinite: true,
              hasBiasFirstMoment: true,
              hasBiasInfinityNorm: false,
              hasBiasMaxSecondMoment: false,
              hasBiasSecondMoment: true,
              hasBiasSecondMomentum: false,
              hasConnectionFirstMoment: true,
              hasConnectionGradientAccumulator: false,
              hasConnectionInfinityNorm: false,
              hasConnectionMaxSecondMoment: false,
              hasConnectionSecondMoment: true,
              hasConnectionSecondMomentum: false,
              incomingAccumulatorCleared: true,
              incomingWeightChanged: true,
            },
          });
        });
      });
    });

    describe('given optimizer branches that depend on default and zero-gradient fallbacks', () => {
      describe('when applying targeted optimizer configurations', () => {
        it('covers the default lookahead, zero-gradient adamax and lion, and repeated amsgrad paths', () => {
          // Arrange
          const defaultLookaheadHarness = createBatchOptimizerHarness('hidden');
          defaultLookaheadHarness.trainingNode.bias = 2;
          defaultLookaheadHarness.trainingNode.totalDeltaBias = 0;
          defaultLookaheadHarness.incomingConnection.totalDeltaWeight = 1;
          defaultLookaheadHarness.selfConnection.totalDeltaWeight = 1;

          const repeatedLookaheadHarness =
            createBatchOptimizerHarness('hidden');
          const repeatedLookaheadNode =
            repeatedLookaheadHarness.trainingNode as unknown as Record<
              string,
              number
            >;
          repeatedLookaheadNode._la_alpha = 0.25;
          repeatedLookaheadNode._la_k = 1;
          repeatedLookaheadNode._la_shadowBias = 3;
          repeatedLookaheadNode._la_step = 0;
          repeatedLookaheadHarness.trainingNode.bias = 6;
          repeatedLookaheadHarness.trainingNode.totalDeltaBias = 1;
          repeatedLookaheadHarness.incomingConnection.weight = 7;
          repeatedLookaheadHarness.incomingConnection.lookaheadShadowWeight = 4;
          repeatedLookaheadHarness.incomingConnection.totalDeltaWeight = 1;
          repeatedLookaheadHarness.selfConnection.weight = 8;
          repeatedLookaheadHarness.selfConnection.lookaheadShadowWeight = 5;
          repeatedLookaheadHarness.selfConnection.totalDeltaWeight = 1;

          const zeroGradientAdamaxHarness =
            createBatchOptimizerHarness('hidden');
          zeroGradientAdamaxHarness.trainingNode.bias = 1;
          zeroGradientAdamaxHarness.trainingNode.totalDeltaBias = 0;
          zeroGradientAdamaxHarness.incomingConnection.totalDeltaWeight = 0;
          zeroGradientAdamaxHarness.selfConnection.totalDeltaWeight = 0;

          const zeroGradientLionHarness = createBatchOptimizerHarness('hidden');
          zeroGradientLionHarness.trainingNode.bias = 1;
          zeroGradientLionHarness.trainingNode.totalDeltaBias = 0;
          zeroGradientLionHarness.incomingConnection.totalDeltaWeight = 0;
          zeroGradientLionHarness.selfConnection.totalDeltaWeight = 0;

          const repeatedAmsgradHarness = createBatchOptimizerHarness('hidden');
          const repeatedAmsgradNode =
            repeatedAmsgradHarness.trainingNode as unknown as Record<
              string,
              number
            >;
          repeatedAmsgradHarness.trainingNode.totalDeltaBias = 2;
          repeatedAmsgradHarness.incomingConnection.totalDeltaWeight = 2;
          repeatedAmsgradHarness.selfConnection.totalDeltaWeight = 2;
          repeatedAmsgradHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'amsgrad',
          });
          repeatedAmsgradHarness.trainingNode.totalDeltaBias = 2;
          repeatedAmsgradHarness.incomingConnection.totalDeltaWeight = 2;
          repeatedAmsgradHarness.selfConnection.totalDeltaWeight = 2;

          const zeroDecayAdamwHarness = createBatchOptimizerHarness('hidden');
          zeroDecayAdamwHarness.trainingNode.totalDeltaBias = 2;
          zeroDecayAdamwHarness.incomingConnection.totalDeltaWeight = 2;
          zeroDecayAdamwHarness.selfConnection.totalDeltaWeight = 2;

          // Act
          defaultLookaheadHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'lookahead',
          });
          repeatedLookaheadHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'lookahead',
            baseType: 'sgd',
          });
          zeroGradientAdamaxHarness.trainingNode.applyBatchUpdatesWithOptimizer(
            {
              type: 'adamax',
            },
          );
          zeroGradientLionHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'lion',
          });
          repeatedAmsgradHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'amsgrad',
          });
          zeroDecayAdamwHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'adamw',
          });

          // Assert
          expect({
            defaultLookahead: {
              alpha: (
                defaultLookaheadHarness.trainingNode as unknown as Record<
                  string,
                  number
                >
              )._la_alpha,
              bias: defaultLookaheadHarness.trainingNode.bias,
              k: (
                defaultLookaheadHarness.trainingNode as unknown as Record<
                  string,
                  number
                >
              )._la_k,
              step: (
                defaultLookaheadHarness.trainingNode as unknown as Record<
                  string,
                  number
                >
              )._la_step,
            },
            repeatedAmsgrad: {
              incomingMaxSecondMoment: Number(
                repeatedAmsgradHarness.incomingConnection.maxSecondMoment?.toFixed(
                  6,
                ),
              ),
              optVhatBias: Number(repeatedAmsgradNode.opt_vhatB?.toFixed(6)),
            },
            repeatedLookahead: {
              bias: repeatedLookaheadHarness.trainingNode.bias,
              incomingShadowWeight:
                repeatedLookaheadHarness.incomingConnection
                  .lookaheadShadowWeight,
              selfShadowWeight:
                repeatedLookaheadHarness.selfConnection.lookaheadShadowWeight,
            },
            zeroDecayAdamw: {
              biasFinite: Number.isFinite(
                zeroDecayAdamwHarness.trainingNode.bias,
              ),
              weightFinite: Number.isFinite(
                zeroDecayAdamwHarness.incomingConnection.weight,
              ),
            },
            zeroGradientAdamax: {
              bias: zeroGradientAdamaxHarness.trainingNode.bias,
              incomingWeight:
                zeroGradientAdamaxHarness.incomingConnection.weight,
            },
            zeroGradientLion: {
              bias: zeroGradientLionHarness.trainingNode.bias,
              incomingWeight: zeroGradientLionHarness.incomingConnection.weight,
            },
          }).toStrictEqual({
            defaultLookahead: {
              alpha: 0.5,
              bias: 2,
              k: 5,
              step: 1,
            },
            repeatedAmsgrad: {
              incomingMaxSecondMoment: 0.007996,
              optVhatBias: 0.007996,
            },
            repeatedLookahead: {
              bias: 4,
              incomingShadowWeight: 5,
              selfShadowWeight: 6,
            },
            zeroDecayAdamw: {
              biasFinite: true,
              weightFinite: true,
            },
            zeroGradientAdamax: {
              bias: 1,
              incomingWeight: 1,
            },
            zeroGradientLion: {
              bias: 1,
              incomingWeight: 1,
            },
          });
        });
      });
    });

    describe('given optional optimizer scratch values that stay undefined', () => {
      describe('when applying targeted optimizer variants', () => {
        it('falls back to zeroed optional state while keeping the updates finite', () => {
          // Arrange
          const adamwZeroValueHarness = createBatchOptimizerHarness('hidden');
          adamwZeroValueHarness.trainingNode.bias = 0;
          adamwZeroValueHarness.trainingNode.totalDeltaBias = 2;
          adamwZeroValueHarness.incomingConnection.weight =
            undefined as unknown as number;
          adamwZeroValueHarness.incomingConnection.totalDeltaWeight = 2;
          adamwZeroValueHarness.selfConnection.weight =
            undefined as unknown as number;
          adamwZeroValueHarness.selfConnection.totalDeltaWeight = 2;

          const amsgradFallbackHarness = createBatchOptimizerHarness('hidden');
          amsgradFallbackHarness.trainingNode.totalDeltaBias = 2;
          amsgradFallbackHarness.incomingConnection.totalDeltaWeight = 2;
          amsgradFallbackHarness.selfConnection.totalDeltaWeight = 2;

          const adabeliefFallbackHarness =
            createBatchOptimizerHarness('hidden');
          adabeliefFallbackHarness.trainingNode.totalDeltaBias = 2;
          adabeliefFallbackHarness.incomingConnection.totalDeltaWeight = 2;
          adabeliefFallbackHarness.selfConnection.totalDeltaWeight = 2;

          const adamaxFallbackHarness = createBatchOptimizerHarness('hidden');
          adamaxFallbackHarness.trainingNode.totalDeltaBias = 2;
          adamaxFallbackHarness.incomingConnection.totalDeltaWeight = 2;
          adamaxFallbackHarness.selfConnection.totalDeltaWeight = 2;

          const lionFallbackHarness = createBatchOptimizerHarness('hidden');
          lionFallbackHarness.trainingNode.totalDeltaBias = 2;
          lionFallbackHarness.incomingConnection.totalDeltaWeight = 2;
          lionFallbackHarness.selfConnection.totalDeltaWeight = 2;

          const lookaheadFallbackHarness =
            createBatchOptimizerHarness('hidden');
          lookaheadFallbackHarness.trainingNode.bias = 4;
          lookaheadFallbackHarness.trainingNode.totalDeltaBias = 1;
          lookaheadFallbackHarness.incomingConnection.weight = 6;
          lookaheadFallbackHarness.incomingConnection.totalDeltaWeight = 1;
          lookaheadFallbackHarness.selfConnection.weight = 8;
          lookaheadFallbackHarness.selfConnection.totalDeltaWeight = 1;

          // Act
          adamwZeroValueHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'adamw',
            weightDecay: 0.1,
          });

          withIgnoredWriteProperties(
            [
              {
                propertyName: 'secondMoment',
                targetObject: amsgradFallbackHarness.incomingConnection,
              },
              {
                propertyName: 'maxSecondMoment',
                targetObject: amsgradFallbackHarness.incomingConnection,
              },
              {
                propertyName: 'secondMoment',
                targetObject: amsgradFallbackHarness.selfConnection,
              },
              {
                propertyName: 'maxSecondMoment',
                targetObject: amsgradFallbackHarness.selfConnection,
              },
              {
                propertyName: 'opt_mB',
                targetObject: amsgradFallbackHarness.trainingNode,
              },
              {
                propertyName: 'opt_vB',
                targetObject: amsgradFallbackHarness.trainingNode,
              },
              {
                propertyName: 'opt_vhatB',
                targetObject: amsgradFallbackHarness.trainingNode,
              },
            ],
            () => {
              amsgradFallbackHarness.trainingNode.applyBatchUpdatesWithOptimizer(
                {
                  type: 'amsgrad',
                },
              );
            },
          );

          withIgnoredWriteProperties(
            [
              {
                propertyName: 'opt_mB',
                targetObject: adabeliefFallbackHarness.trainingNode,
              },
              {
                propertyName: 'opt_vB',
                targetObject: adabeliefFallbackHarness.trainingNode,
              },
            ],
            () => {
              adabeliefFallbackHarness.trainingNode.applyBatchUpdatesWithOptimizer(
                {
                  type: 'adabelief',
                },
              );
            },
          );

          withIgnoredWriteProperties(
            [
              {
                propertyName: 'infinityNorm',
                targetObject: adamaxFallbackHarness.incomingConnection,
              },
              {
                propertyName: 'opt_uB',
                targetObject: adamaxFallbackHarness.trainingNode,
              },
            ],
            () => {
              adamaxFallbackHarness.trainingNode.applyBatchUpdatesWithOptimizer(
                {
                  type: 'adamax',
                },
              );
            },
          );

          withIgnoredWriteProperties(
            [
              {
                propertyName: 'firstMoment',
                targetObject: lionFallbackHarness.incomingConnection,
              },
              {
                propertyName: 'secondMomentum',
                targetObject: lionFallbackHarness.incomingConnection,
              },
              {
                propertyName: 'opt_mB',
                targetObject: lionFallbackHarness.trainingNode,
              },
              {
                propertyName: 'opt_mB2',
                targetObject: lionFallbackHarness.trainingNode,
              },
            ],
            () => {
              lionFallbackHarness.trainingNode.applyBatchUpdatesWithOptimizer({
                type: 'lion',
              });
            },
          );

          withIgnoredWriteProperties(
            [
              {
                propertyName: '_la_alpha',
                targetObject: lookaheadFallbackHarness.trainingNode,
              },
              {
                propertyName: '_la_k',
                targetObject: lookaheadFallbackHarness.trainingNode,
              },
              {
                propertyName: '_la_shadowBias',
                targetObject: lookaheadFallbackHarness.trainingNode,
              },
              {
                propertyName: '_la_step',
                targetObject: lookaheadFallbackHarness.trainingNode,
              },
            ],
            () => {
              lookaheadFallbackHarness.trainingNode.applyBatchUpdatesWithOptimizer(
                {
                  type: 'lookahead',
                },
              );
            },
          );

          // Assert
          expect({
            adabeliefBiasFinite: Number.isFinite(
              adabeliefFallbackHarness.trainingNode.bias,
            ),
            adamaxBiasFinite: Number.isFinite(
              adamaxFallbackHarness.trainingNode.bias,
            ),
            adamwBiasFinite: Number.isFinite(
              adamwZeroValueHarness.trainingNode.bias,
            ),
            adamwIncomingWeightFinite: Number.isFinite(
              adamwZeroValueHarness.incomingConnection.weight,
            ),
            amsgradBiasFinite: Number.isFinite(
              amsgradFallbackHarness.trainingNode.bias,
            ),
            lionBiasFinite: Number.isFinite(
              lionFallbackHarness.trainingNode.bias,
            ),
            lookaheadBias: lookaheadFallbackHarness.trainingNode.bias,
          }).toStrictEqual({
            adabeliefBiasFinite: true,
            adamaxBiasFinite: true,
            adamwBiasFinite: true,
            adamwIncomingWeightFinite: true,
            amsgradBiasFinite: true,
            lionBiasFinite: true,
            lookaheadBias: undefined,
          });
        });
      });
    });

    describe('given a lookahead optimizer step at the sync boundary', () => {
      describe('when applying the wrapper update', () => {
        it('stores lookahead state and blends the slow bias value', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createBatchOptimizerHarness('hidden');
          const optimizerNode = trainingNode as unknown as Record<
            string,
            number
          >;
          trainingNode.bias = 4;
          trainingNode.totalDeltaBias = 2;
          incomingConnection.weight = 6;
          incomingConnection.totalDeltaWeight = 2;
          selfConnection.weight = 8;
          selfConnection.totalDeltaWeight = 2;

          // Act
          trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'lookahead',
            baseType: 'sgd',
            la_alpha: 0.25,
            la_k: 1,
          });

          // Assert
          expect({
            bias: trainingNode.bias,
            incomingShadowWeight: incomingConnection.lookaheadShadowWeight,
            incomingWeight: incomingConnection.weight,
            selfShadowWeight: selfConnection.lookaheadShadowWeight,
            selfWeight: selfConnection.weight,
            step: optimizerNode._la_step,
            storedAlpha: optimizerNode._la_alpha,
            storedK: optimizerNode._la_k,
            storedShadowBias: optimizerNode._la_shadowBias,
          }).toStrictEqual({
            bias: 4.5,
            incomingShadowWeight: 8,
            incomingWeight: 8,
            selfShadowWeight: 10,
            selfWeight: 10,
            step: 1,
            storedAlpha: 0.25,
            storedK: 1,
            storedShadowBias: 4.5,
          });
        });
      });
    });

    describe('given an input node with queued bias deltas', () => {
      describe('when applying batch updates', () => {
        it('clears the bias accumulators without changing the bias', () => {
          // Arrange
          const inputNode = new Node('input');
          inputNode.bias = 7;
          inputNode.previousDeltaBias = 3;
          inputNode.totalDeltaBias = 4;

          // Act
          inputNode.applyBatchUpdatesWithOptimizer({
            type: 'sgd',
            momentum: 1,
          });

          // Assert
          expect({
            bias: inputNode.bias,
            previousDeltaBias: inputNode.previousDeltaBias,
            totalDeltaBias: inputNode.totalDeltaBias,
          }).toStrictEqual({
            bias: 7,
            previousDeltaBias: 0,
            totalDeltaBias: 0,
          });
        });
      });
    });

    describe('given adaptive bias updates with invalid and oversized results', () => {
      describe('when applying the adam optimizer', () => {
        it('resets invalid bias values and clips oversized ones', () => {
          // Arrange
          const invalidBiasHarness = createBatchOptimizerHarness('hidden');
          invalidBiasHarness.trainingNode.bias = Number.POSITIVE_INFINITY;
          invalidBiasHarness.trainingNode.totalDeltaBias = 1;
          invalidBiasHarness.incomingConnection.totalDeltaWeight = 1;

          const clippedBiasHarness = createBatchOptimizerHarness('hidden');
          clippedBiasHarness.trainingNode.bias = 999_999.75;
          clippedBiasHarness.trainingNode.totalDeltaBias = 10;
          clippedBiasHarness.incomingConnection.totalDeltaWeight = 1;

          // Act
          invalidBiasHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'adam',
          });
          clippedBiasHarness.trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'adam',
          });

          // Assert
          expect({
            clippedBias: clippedBiasHarness.trainingNode.bias,
            invalidBias: invalidBiasHarness.trainingNode.bias,
          }).toStrictEqual({
            clippedBias: 1_000_000,
            invalidBias: 0,
          });
        });
      });
    });

    describe('given non-finite SGD batch deltas and weights', () => {
      describe('when applying the optimizer', () => {
        it('resets incoming, self, and bias state to zero', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createBatchOptimizerHarness('hidden');
          trainingNode.bias = Number.POSITIVE_INFINITY;
          trainingNode.previousDeltaBias = Number.POSITIVE_INFINITY;
          trainingNode.totalDeltaBias = Number.POSITIVE_INFINITY;
          incomingConnection.weight = Number.POSITIVE_INFINITY;
          incomingConnection.previousDeltaWeight = Number.POSITIVE_INFINITY;
          incomingConnection.totalDeltaWeight = Number.POSITIVE_INFINITY;
          selfConnection.weight = Number.POSITIVE_INFINITY;
          selfConnection.previousDeltaWeight = Number.POSITIVE_INFINITY;
          selfConnection.totalDeltaWeight = Number.POSITIVE_INFINITY;

          // Act
          trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'sgd',
            momentum: 1,
          });

          // Assert
          expect({
            bias: trainingNode.bias,
            incomingPreviousDeltaWeight: incomingConnection.previousDeltaWeight,
            incomingWeight: incomingConnection.weight,
            previousDeltaBias: trainingNode.previousDeltaBias,
            selfPreviousDeltaWeight: selfConnection.previousDeltaWeight,
            selfWeight: selfConnection.weight,
          }).toStrictEqual({
            bias: 0,
            incomingPreviousDeltaWeight: 0,
            incomingWeight: 0,
            previousDeltaBias: 0,
            selfPreviousDeltaWeight: 0,
            selfWeight: 0,
          });
        });
      });
    });

    describe('given large finite SGD batch deltas and weights', () => {
      describe('when applying the optimizer', () => {
        it('clips the resulting weights and bias to the configured maximum', () => {
          // Arrange
          const { trainingNode, incomingConnection, selfConnection } =
            createBatchOptimizerHarness('hidden');
          trainingNode.bias = 999_999.5;
          trainingNode.previousDeltaBias = 20;
          trainingNode.totalDeltaBias = 2_000;
          incomingConnection.weight = 999_999.5;
          incomingConnection.previousDeltaWeight = 20;
          incomingConnection.totalDeltaWeight = 2_000;
          selfConnection.weight = 999_999.5;
          selfConnection.previousDeltaWeight = 20;
          selfConnection.totalDeltaWeight = 2_000;

          // Act
          trainingNode.applyBatchUpdatesWithOptimizer({
            type: 'sgd',
            momentum: 1,
          });

          // Assert
          expect({
            bias: trainingNode.bias,
            incomingPreviousDeltaWeight: incomingConnection.previousDeltaWeight,
            incomingWeight: incomingConnection.weight,
            previousDeltaBias: trainingNode.previousDeltaBias,
            selfPreviousDeltaWeight: selfConnection.previousDeltaWeight,
            selfWeight: selfConnection.weight,
          }).toStrictEqual({
            bias: 1_000_000,
            incomingPreviousDeltaWeight: 1_000,
            incomingWeight: 1_000_000,
            previousDeltaBias: 1_000,
            selfPreviousDeltaWeight: 1_000,
            selfWeight: 1_000_000,
          });
        });
      });
    });
  });
});
