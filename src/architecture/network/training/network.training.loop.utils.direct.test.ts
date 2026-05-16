import { config } from '../../../config';
import type Network from '../../network/network';
import { trainSetCore } from './network.training.loop.utils';

type LoopConnectionFixture = {
  previousDeltaWeight?: number;
  totalDeltaWeight?: number;
};

type LoopNodeFixture = {
  _fp32Bias?: number;
  applyBatchUpdatesWithOptimizer: jest.Mock;
  bias: number;
  connections: {
    in: LoopConnectionFixture[];
    self: LoopConnectionFixture[];
  };
  previousDeltaBias: number;
  propagate: jest.Mock;
  totalDeltaBias?: number;
  type: 'hidden' | 'input' | 'output';
};

type LoopNetworkFixture = {
  _accumulationReduction?: 'average' | 'sum';
  _currentGradClip?: unknown;
  _forceNextOverflow?: boolean;
  _gradAccumMicroBatches?: number;
  _lastGradNorm?: number;
  _lastOverflowStep?: number;
  _mixedPrecision: {
    enabled: boolean;
    lossScale: number;
  };
  _mixedPrecisionState: {
    badSteps: number;
    goodSteps: number;
    maxLossScale: number;
    minLossScale: number;
    overflowCount?: number;
    underflowCount?: number;
    lastUnderflowStep?: number;
    scaleDownEvents?: number;
    scaleUpEvents?: number;
  };
  _mpIncreaseEvery?: number;
  _optimizerStep?: number;
  activate: (input: number[], training: boolean) => number[];
  input: number;
  nodes: LoopNodeFixture[];
  output: number;
};

function createConnection(
  input?: Partial<LoopConnectionFixture>,
): LoopConnectionFixture {
  return {
    previousDeltaWeight: 0,
    totalDeltaWeight: 0,
    ...input,
  };
}

function createNode(input?: Partial<LoopNodeFixture>): LoopNodeFixture {
  return {
    applyBatchUpdatesWithOptimizer: jest.fn(),
    bias: 0,
    connections: {
      in: [],
      self: [],
    },
    previousDeltaBias: 0,
    propagate: jest.fn(),
    totalDeltaBias: 0,
    type: 'hidden',
    ...input,
  };
}

function createNetwork(
  input: Partial<LoopNetworkFixture> = {},
): LoopNetworkFixture {
  return {
    _accumulationReduction: 'sum',
    _currentGradClip: undefined,
    _forceNextOverflow: false,
    _gradAccumMicroBatches: 0,
    _lastGradNorm: undefined,
    _lastOverflowStep: -1,
    _mixedPrecision: {
      enabled: false,
      lossScale: 1,
    },
    _mixedPrecisionState: {
      badSteps: 0,
      goodSteps: 0,
      maxLossScale: 8,
      minLossScale: 1,
      overflowCount: 0,
      underflowCount: 0,
      lastUnderflowStep: -1,
      scaleDownEvents: 0,
      scaleUpEvents: 0,
    },
    _optimizerStep: 0,
    activate: jest.fn(() => [0]),
    input: 1,
    nodes: [createNode({ type: 'input' }), createNode({ type: 'output' })],
    output: 1,
    ...input,
  };
}

describe('network training loop chapter', () => {
  const originalWarnings = config.warnings;

  afterEach(() => {
    config.warnings = originalWarnings;
    jest.restoreAllMocks();
  });

  describe('trainSetCore', () => {
    describe('given an invalid cost descriptor and a dataset sample with mismatched dimensions', () => {
      it('warns once, skips the sample, and returns zero processed error', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createNetwork({ input: 2, output: 1 });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0,
          {},
          {} as never,
        );

        // Assert
        expect({
          meanError,
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          meanError: 0,
          warningCount: 1,
        });
      });
    });

    describe('given an invalid cost descriptor is used on a valid training sample', () => {
      it('falls back to a zero-valued cost function', () => {
        // Arrange
        const network = createNetwork({
          activate: jest.fn(() => [0.25]),
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0,
          {},
          {} as never,
        );

        // Assert
        expect(meanError).toBe(0);
      });
    });

    describe('given a deferred optimizer step averages accumulated gradients before applying updates', () => {
      it('uses hidden-node deferred propagation, averages self gradients, and scales loss with the default mixed-precision cadence', () => {
        // Arrange
        const hiddenSelfConnection = createConnection({
          previousDeltaWeight: 4,
          totalDeltaWeight: 6,
        });
        const hiddenInputConnection = createConnection({
          previousDeltaWeight: 3,
          totalDeltaWeight: 4,
        });
        const hiddenNode = createNode({
          connections: {
            in: [hiddenInputConnection],
            self: [hiddenSelfConnection],
          },
          previousDeltaBias: 5,
          totalDeltaBias: 8,
          type: 'hidden',
        });
        const outputNode = createNode({
          connections: {
            in: [
              createConnection({
                previousDeltaWeight: undefined,
                totalDeltaWeight: undefined,
              }),
            ],
            self: [
              createConnection({
                previousDeltaWeight: undefined,
                totalDeltaWeight: undefined,
              }),
            ],
          },
          totalDeltaBias: undefined,
          type: 'output',
        });
        const network = createNetwork({
          _accumulationReduction: 'average',
          _mixedPrecision: {
            enabled: true,
            lossScale: 2,
          },
          _mixedPrecisionState: {
            badSteps: 0,
            goodSteps: 199,
            maxLossScale: 8,
            minLossScale: 1,
            overflowCount: 0,
            scaleDownEvents: 0,
            scaleUpEvents: 0,
          },
          activate: jest.fn(() => [0.25]),
          nodes: [createNode({ type: 'input' }), hiddenNode, outputNode],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [
            { input: [1], output: [1] },
            { input: [1], output: [1] },
          ],
          1,
          2,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          deferredHiddenPropagation: hiddenNode.propagate.mock.calls[0]?.[2],
          gradNorm: network._lastGradNorm,
          hiddenBiasDelta: hiddenNode.totalDeltaBias,
          hiddenSelfDelta: hiddenSelfConnection.totalDeltaWeight,
          lossScale: network._mixedPrecision.lossScale,
          scaleUpEvents: network._mixedPrecisionState.scaleUpEvents,
        }).toEqual({
          deferredHiddenPropagation: false,
          gradNorm: 5,
          hiddenBiasDelta: 4,
          hiddenSelfDelta: 3,
          lossScale: 4,
          scaleUpEvents: 1,
        });
      });
    });

    describe('given activation throws a non-Error value while warnings are enabled', () => {
      it('stringifies the thrown value into the warning message and returns zero processed error', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createNetwork({
          activate: jest.fn(() => {
            throw 'non-error-overflow';
          }),
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0,
          {},
          () => 1,
        );

        // Assert
        expect({
          meanError,
          warnedWithThrownValue: String(warnSpy.mock.calls[0]?.[0]).includes(
            'non-error-overflow',
          ),
        }).toEqual({
          meanError: 0,
          warnedWithThrownValue: true,
        });
      });
    });

    describe('given activation throws an Error while warnings are enabled', () => {
      it('uses the Error message in the warning output and returns zero processed error', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createNetwork({
          activate: jest.fn(() => {
            throw new Error('finite-activation-guard');
          }),
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0,
          {},
          () => 1,
        );

        // Assert
        expect({
          meanError,
          warnedWithErrorMessage: String(warnSpy.mock.calls[0]?.[0]).includes(
            'finite-activation-guard',
          ),
        }).toEqual({
          meanError: 0,
          warnedWithErrorMessage: true,
        });
      });
    });

    describe('given warnings stay disabled across one mismatched sample and one thrown sample', () => {
      it('skips both bad samples without emitting warning output', () => {
        // Arrange
        config.warnings = false;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createNetwork({
          activate: jest.fn(() => {
            throw 'silent-training-failure';
          }),
          input: 2,
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [
            { input: [1], output: [1] },
            { input: [1, 1], output: [1] },
          ],
          1,
          1,
          0.1,
          0,
          {},
          () => 1,
        );

        // Assert
        expect({
          meanError,
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          meanError: 0,
          warningCount: 0,
        });
      });
    });

    describe('given activation returns a non-finite output while warnings are enabled', () => {
      it('warns once, skips propagation, and returns zero processed error', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const hiddenNode = createNode({ type: 'hidden' });
        const outputNode = createNode({ type: 'output' });
        const network = createNetwork({
          activate: jest.fn(() => [Number.NaN]),
          nodes: [createNode({ type: 'input' }), hiddenNode, outputNode],
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0,
          {},
          () => 1,
        );

        // Assert
        expect({
          hiddenPropagationCalls: hiddenNode.propagate.mock.calls.length,
          meanError,
          outputPropagationCalls: outputNode.propagate.mock.calls.length,
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          hiddenPropagationCalls: 0,
          meanError: 0,
          outputPropagationCalls: 0,
          warningCount: 1,
        });
      });
    });

    describe('given a training sample contains a non-finite target while warnings are enabled', () => {
      it('warns once, skips activation, and returns zero processed error', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createNetwork({
          activate: jest.fn(() => [0.5]),
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [Number.NaN] }],
          1,
          1,
          0.1,
          0,
          {},
          () => 1,
        );

        // Assert
        expect({
          activationCalls: jest.mocked(network.activate).mock.calls.length,
          meanError,
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          activationCalls: 0,
          meanError: 0,
          warningCount: 1,
        });
      });
    });

    describe('given warnings stay disabled for non-finite target and activation-output samples', () => {
      it('skips both samples without emitting warning output', () => {
        // Arrange
        config.warnings = false;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const outputNode = createNode({ type: 'output' });
        const network = createNetwork({
          activate: jest.fn().mockReturnValueOnce([Number.NaN]),
          nodes: [createNode({ type: 'input' }), outputNode],
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [
            { input: [1], output: [Number.NaN] },
            { input: [1], output: [1] },
          ],
          1,
          1,
          0.1,
          0,
          {},
          () => 1,
        );

        // Assert
        expect({
          activationCalls: jest.mocked(network.activate).mock.calls.length,
          meanError,
          outputPropagationCalls: outputNode.propagate.mock.calls.length,
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          activationCalls: 1,
          meanError: 0,
          outputPropagationCalls: 0,
          warningCount: 0,
        });
      });
    });

    describe('given plain SGD runs through one hidden node with a calculate-only cost descriptor', () => {
      it('uses the calculate callback and applies hidden propagation with immediate updates', () => {
        // Arrange
        const hiddenNode = createNode({
          connections: {
            in: [createConnection()],
            self: [createConnection()],
          },
          type: 'hidden',
        });
        const network = createNetwork({
          activate: jest.fn(() => [0.5]),
          nodes: [
            createNode({ type: 'input' }),
            hiddenNode,
            createNode({ type: 'output' }),
          ],
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          {
            calculate: () => 2,
          } as never,
          { type: 'sgd' },
        );

        // Assert
        expect({
          hiddenUsesImmediateUpdate: hiddenNode.propagate.mock.calls[0]?.[2],
          meanError,
        }).toEqual({
          hiddenUsesImmediateUpdate: true,
          meanError: 2,
        });
      });
    });

    describe('given a deferred optimizer step runs with mixed precision disabled', () => {
      it('keeps the loss scale unchanged while still completing the optimizer pass', () => {
        // Arrange
        const hiddenNode = createNode({
          connections: {
            in: [createConnection()],
            self: [createConnection()],
          },
          type: 'hidden',
        });
        const outputNode = createNode({ type: 'output' });
        const network = createNetwork({
          _currentGradClip: { maxNorm: 1, mode: 'norm' },
          activate: jest.fn(() => [0.5]),
          nodes: [createNode({ type: 'input' }), hiddenNode, outputNode],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          hiddenUsesDeferredPropagation: hiddenNode.propagate.mock.calls[0]?.[2],
          lossScale: network._mixedPrecision.lossScale,
          optimizerStep: network._optimizerStep,
          outputUsesDeferredPropagation: outputNode.propagate.mock.calls[0]?.[2],
          scaleUpEvents: network._mixedPrecisionState.scaleUpEvents,
        }).toEqual({
          hiddenUsesDeferredPropagation: false,
          lossScale: 1,
          optimizerStep: 1,
          outputUsesDeferredPropagation: false,
          scaleUpEvents: 0,
        });
      });
    });

    describe('given the next optimizer step is forced to overflow', () => {
      it('consumes the force flag and takes the overflow recovery path', () => {
        // Arrange
        const hiddenNode = createNode({
          connections: {
            in: [createConnection({ totalDeltaWeight: 5 })],
            self: [createConnection({ totalDeltaWeight: 7 })],
          },
          totalDeltaBias: 3,
          type: 'hidden',
        });
        const network = createNetwork({
          _forceNextOverflow: true,
          _mixedPrecision: {
            enabled: true,
            lossScale: 2,
          },
          _mixedPrecisionState: {
            badSteps: 0,
            goodSteps: 0,
            maxLossScale: 8,
            minLossScale: 1,
            overflowCount: 0,
            scaleDownEvents: 0,
            scaleUpEvents: 0,
          },
          activate: jest.fn(() => [0.5]),
          nodes: [
            createNode({ type: 'input' }),
            hiddenNode,
            createNode({ totalDeltaBias: undefined, type: 'output' }),
          ],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          forceNextOverflow: network._forceNextOverflow,
          gradNorm: network._lastGradNorm,
          lossScale: network._mixedPrecision.lossScale,
          overflowCount: network._mixedPrecisionState.overflowCount,
          scaleDownEvents: network._mixedPrecisionState.scaleDownEvents,
        }).toEqual({
          forceNextOverflow: false,
          gradNorm: 0,
          lossScale: 1,
          overflowCount: 1,
          scaleDownEvents: 1,
        });
      });
    });

    describe('given a deferred optimizer step flushes on the last sample with an existing optimizer step', () => {
      it('runs the deferred step even when the batch boundary is only reached by end-of-set', () => {
        // Arrange
        const hiddenNode = createNode({
          connections: {
            in: [createConnection()],
            self: [createConnection()],
          },
          type: 'hidden',
        });
        const outputNode = createNode({ type: 'output' });
        const network = createNetwork({
          _optimizerStep: 3,
          activate: jest.fn(() => [0.5]),
          nodes: [createNode({ type: 'input' }), hiddenNode, outputNode],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          2,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          hiddenUsesDeferredPropagation: hiddenNode.propagate.mock.calls[0]?.[2],
          optimizerStep: network._optimizerStep,
          outputUsesDeferredPropagation: outputNode.propagate.mock.calls[0]?.[2],
        }).toEqual({
          hiddenUsesDeferredPropagation: false,
          optimizerStep: 4,
          outputUsesDeferredPropagation: false,
        });
      });
    });

    describe('given mixed precision uses a custom increase cadence with finite fp32 bias', () => {
      it('avoids overflow and increases the loss scale through the configured cadence', () => {
        // Arrange
        const hiddenNode = createNode({
          _fp32Bias: 0,
          bias: 0,
          connections: {
            in: [createConnection({ totalDeltaWeight: 4 })],
            self: [createConnection({ totalDeltaWeight: 3 })],
          },
          totalDeltaBias: 2,
          type: 'hidden',
        });
        const network = createNetwork({
          _mixedPrecision: {
            enabled: true,
            lossScale: 2,
          },
          _mixedPrecisionState: {
            badSteps: 0,
            goodSteps: 0,
            maxLossScale: 8,
            minLossScale: 1,
            overflowCount: 0,
            underflowCount: 0,
            lastUnderflowStep: -1,
            scaleDownEvents: 0,
            scaleUpEvents: 0,
          },
          _mpIncreaseEvery: 1,
          activate: jest.fn(() => [0.5]),
          nodes: [
            createNode({ type: 'input' }),
            hiddenNode,
            createNode({ totalDeltaBias: undefined, type: 'output' }),
          ],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          lossScale: network._mixedPrecision.lossScale,
          overflowCount: network._mixedPrecisionState.overflowCount,
          scaleUpEvents: network._mixedPrecisionState.scaleUpEvents,
        }).toEqual({
          lossScale: 4,
          overflowCount: 0,
          scaleUpEvents: 1,
        });
      });
    });

    describe('given mixed precision reaches a custom increase cadence at the maximum loss scale', () => {
      it('avoids overflow without scaling past the configured maximum', () => {
        // Arrange
        const hiddenNode = createNode({
          _fp32Bias: 0,
          bias: 0,
          connections: {
            in: [createConnection({ totalDeltaWeight: 4 })],
            self: [createConnection({ totalDeltaWeight: 3 })],
          },
          totalDeltaBias: 2,
          type: 'hidden',
        });
        const network = createNetwork({
          _mixedPrecision: {
            enabled: true,
            lossScale: 8,
          },
          _mixedPrecisionState: {
            badSteps: 0,
            goodSteps: 0,
            maxLossScale: 8,
            minLossScale: 1,
            overflowCount: 0,
            underflowCount: 0,
            lastUnderflowStep: -1,
            scaleDownEvents: 0,
            scaleUpEvents: 0,
          },
          _mpIncreaseEvery: 1,
          activate: jest.fn(() => [0.5]),
          nodes: [
            createNode({ type: 'input' }),
            hiddenNode,
            createNode({ totalDeltaBias: undefined, type: 'output' }),
          ],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          lossScale: network._mixedPrecision.lossScale,
          overflowCount: network._mixedPrecisionState.overflowCount,
          scaleUpEvents: network._mixedPrecisionState.scaleUpEvents,
        }).toEqual({
          lossScale: 8,
          overflowCount: 0,
          scaleUpEvents: 0,
        });
      });
    });

    describe('given a cost descriptor exposes a fn callback on a valid training sample', () => {
      it('uses that fn callback as the active cost function', () => {
        // Arrange
        const network = createNetwork({
          activate: jest.fn(() => [0.25]),
        });

        // Act
        const meanError = trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0,
          {},
          {
            fn: () => 3,
          } as never,
        );

        // Assert
        expect(meanError).toBe(3);
      });
    });

    describe('given mixed precision detects a non-finite fp32 bias during an optimizer step', () => {
      it('zeros the accumulated self gradients and records a scale-down overflow event', () => {
        // Arrange
        const hiddenSelfConnection = createConnection({
          previousDeltaWeight: 2,
          totalDeltaWeight: 7,
        });
        const hiddenNode = createNode({
          _fp32Bias: 0,
          bias: Number.POSITIVE_INFINITY,
          connections: {
            in: [createConnection({ totalDeltaWeight: 5 })],
            self: [hiddenSelfConnection],
          },
          previousDeltaBias: 9,
          totalDeltaBias: 6,
          type: 'hidden',
        });
        const network = createNetwork({
          _mixedPrecision: {
            enabled: true,
            lossScale: 1,
          },
          _mixedPrecisionState: {
            badSteps: 0,
            goodSteps: 3,
            maxLossScale: 8,
            minLossScale: 0,
            overflowCount: 0,
            scaleDownEvents: 0,
            scaleUpEvents: 0,
          },
          activate: jest.fn(() => [0.5]),
          nodes: [
            createNode({ type: 'input' }),
            hiddenNode,
            createNode({ totalDeltaBias: undefined, type: 'output' }),
          ],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          lossScale: network._mixedPrecision.lossScale,
          overflowCount: network._mixedPrecisionState.overflowCount,
          previousDeltaBias: hiddenNode.previousDeltaBias,
          recordedOverflowStep: network._lastOverflowStep,
          scaleDownEvents: network._mixedPrecisionState.scaleDownEvents,
          selfGradientAfterOverflow: hiddenSelfConnection.totalDeltaWeight,
          totalDeltaBias: hiddenNode.totalDeltaBias,
        }).toEqual({
          lossScale: 1,
          overflowCount: 1,
          previousDeltaBias: 0,
          recordedOverflowStep: 1,
          scaleDownEvents: 1,
          selfGradientAfterOverflow: 0,
          totalDeltaBias: 0,
        });
      });
    });

    describe('given mixed precision sees only tiny accumulated gradients during an optimizer step', () => {
      it('records an underflow event and increases the loss scale', () => {
        // Arrange
        const hiddenNode = createNode({
          connections: {
            in: [createConnection({ totalDeltaWeight: 1e-16 })],
            self: [createConnection({ totalDeltaWeight: 5e-17 })],
          },
          totalDeltaBias: 2e-16,
          type: 'hidden',
        });
        const network = createNetwork({
          _mixedPrecision: {
            enabled: true,
            lossScale: 2,
          },
          _mixedPrecisionState: {
            badSteps: 0,
            goodSteps: 0,
            maxLossScale: 8,
            minLossScale: 1,
            overflowCount: 0,
            underflowCount: 0,
            lastUnderflowStep: -1,
            scaleDownEvents: 0,
            scaleUpEvents: 0,
          },
          activate: jest.fn(() => [0.5]),
          nodes: [
            createNode({ type: 'input' }),
            hiddenNode,
            createNode({ totalDeltaBias: undefined, type: 'output' }),
          ],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          lastUnderflowStep: network._mixedPrecisionState.lastUnderflowStep,
          lossScale: network._mixedPrecision.lossScale,
          scaleUpEvents: network._mixedPrecisionState.scaleUpEvents,
          underflowCount: network._mixedPrecisionState.underflowCount,
        }).toEqual({
          lastUnderflowStep: 1,
          lossScale: 4,
          scaleUpEvents: 1,
          underflowCount: 1,
        });
      });

      it('records the underflow without scaling past the configured maximum', () => {
        // Arrange
        const hiddenNode = createNode({
          connections: {
            in: [createConnection({ totalDeltaWeight: 1e-16 })],
            self: [createConnection({ totalDeltaWeight: 5e-17 })],
          },
          totalDeltaBias: 2e-16,
          type: 'hidden',
        });
        const network = createNetwork({
          _mixedPrecision: {
            enabled: true,
            lossScale: 8,
          },
          _mixedPrecisionState: {
            badSteps: 0,
            goodSteps: 0,
            maxLossScale: 8,
            minLossScale: 1,
            overflowCount: 0,
            underflowCount: 0,
            lastUnderflowStep: -1,
            scaleDownEvents: 0,
            scaleUpEvents: 0,
          },
          activate: jest.fn(() => [0.5]),
          nodes: [
            createNode({ type: 'input' }),
            hiddenNode,
            createNode({ totalDeltaBias: undefined, type: 'output' }),
          ],
        });

        // Act
        trainSetCore(
          network as unknown as Network,
          [{ input: [1], output: [1] }],
          1,
          1,
          0.1,
          0.9,
          {},
          () => 1,
          { type: 'adam' },
        );

        // Assert
        expect({
          lastUnderflowStep: network._mixedPrecisionState.lastUnderflowStep,
          lossScale: network._mixedPrecision.lossScale,
          scaleUpEvents: network._mixedPrecisionState.scaleUpEvents,
          underflowCount: network._mixedPrecisionState.underflowCount,
        }).toEqual({
          lastUnderflowStep: 1,
          lossScale: 8,
          scaleUpEvents: 0,
          underflowCount: 1,
        });
      });
    });
  });
});
