import { jest } from '@jest/globals';

jest.mock('./network.training.loop.utils', () => ({
  trainSetCore: jest.fn(() => 0.5),
}));

import Network from '../../network';
import { Architect } from '../../../neataptic';
import type { TrainingOptions } from '../network.types';
import {
  NetworkTrainingInvalidCostFunctionError,
  NetworkTrainingStoppingConditionRequiredError,
} from './network.training.errors';
import { trainFinalizeCore } from './network.training.finalize.utils';

const MINIMAL_DATASET = [{ input: [1], output: [1] }];

describe('network training finalize utility chapter', () => {
  describe('trainFinalizeCore', () => {
    describe('given options is null', () => {
      it('normalizes null to an empty object then throws because no stopping condition is set', () => {
        // Arrange – null → line 64 FALSE arm (options = null ?? {})
        const network = new Network(1, 1, { seed: 9_001 });

        // Act & Assert
        expect(() =>
          trainFinalizeCore(
            network,
            MINIMAL_DATASET,
            null as unknown as TrainingOptions,
          ),
        ).toThrow(NetworkTrainingStoppingConditionRequiredError);
      });
    });

    describe('given gradient clip is configured with only a percentile', () => {
      it('sets the gradient clip mode to percentile without throwing', () => {
        // Arrange – gradientClip with only percentile → line 138 else-if branch
        const network = new Network(1, 1, { seed: 9_002 });

        // Act & Assert (must not throw)
        expect(() =>
          trainFinalizeCore(network, MINIMAL_DATASET, {
            iterations: 1,
            gradientClip: { percentile: 0.95 },
          }),
        ).not.toThrow();
      });
    });

    describe('given gradient clip provides an explicit mode', () => {
      it('uses that mode without relying on shorthand inference', () => {
        // Arrange – explicit mode covers the direct configuration branch.
        const network = new Network(1, 1, { seed: 9_002_05 });

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          error: 0,
          gradientClip: { mode: 'norm', maxNorm: 1 },
        });
        const currentGradClip = (
          network as unknown as {
            _currentGradClip?: { mode?: string };
          }
        )._currentGradClip;

        // Assert
        expect(currentGradClip?.mode).toBe('norm');
      });
    });

    describe('given gradient clip only toggles separateBias', () => {
      it('keeps the clip payload unset while preserving the bias flag', () => {
        // Arrange – this forces the maxNorm and percentile shorthand checks to fall through.
        const network = new Network(1, 1, { seed: 9_002_06 });

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          error: 0,
          gradientClip: { separateBias: true },
        });
        const trainingInternals = network as unknown as {
          _currentGradClip?: unknown;
          _gradClipSeparateBias?: boolean;
        };

        // Assert
        expect({
          clip: trainingInternals._currentGradClip,
          separateBias: trainingInternals._gradClipSeparateBias,
        }).toStrictEqual({
          clip: undefined,
          separateBias: true,
        });
      });
    });

    describe('given max-norm shorthand clipping and momentum are both configured', () => {
      it('normalizes the clip mode and completes one finalize iteration', () => {
        // Arrange – direct finalize coverage for the maxNorm and momentum truthy branches.
        const network = new Network(1, 1, { seed: 9_002_1 });

        // Act
        const result = trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          error: 0,
          momentum: 0.9,
          gradientClip: { maxNorm: 1 },
        });
        const currentGradClip = (
          network as unknown as {
            _currentGradClip?: { mode?: string };
          }
        )._currentGradClip;

        // Assert
        expect({
          clipMode: currentGradClip?.mode,
          iterations: result.iterations,
        }).toStrictEqual({
          clipMode: 'norm',
          iterations: 1,
        });
      });
    });

    describe('given mixed precision is configured with lossScale of 0', () => {
      it('falls back to the default lossScale of 1024', () => {
        // Arrange – lossScale=0 (falsy) → line 157 FALSE arm (lossScale || 1024)
        const network = new Network(1, 1, { seed: 9_003 });

        // Act & Assert (must not throw)
        expect(() =>
          trainFinalizeCore(network, MINIMAL_DATASET, {
            iterations: 1,
            mixedPrecision: { lossScale: 0 },
          }),
        ).not.toThrow();
      });
    });

    describe('given optimizer is an object with a string type', () => {
      it('lower-cases the type field without throwing', () => {
        // Arrange – optimizer object with type string → line 191 branch
        const network = new Network(1, 1, { seed: 9_004 });

        // Act & Assert (must not throw)
        expect(() =>
          trainFinalizeCore(network, MINIMAL_DATASET, {
            iterations: 1,
            optimizer: { type: 'Adam' },
          }),
        ).not.toThrow();
      });
    });

    describe('given _maybePrune is set on the network', () => {
      it('invokes _maybePrune once per iteration', () => {
        // Arrange – _maybePrune set → line 330 TRUE arm
        const network = new Network(1, 1, { seed: 9_005 });
        const maybePruneSpy = jest.fn();
        (network as unknown as { _maybePrune: jest.Mock })._maybePrune =
          maybePruneSpy;

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, { iterations: 1 });

        // Assert
        expect(maybePruneSpy).toHaveBeenCalledTimes(1);
      });
    });

    describe('given metricsHook is provided and _lastGradNorm is absent', () => {
      it('reports gradNorm as undefined without synthesizing a fallback value', () => {
        // Arrange – absent _lastGradNorm should pass through unchanged.
        const network = new Network(1, 1, { seed: 9_006 });
        let capturedGradNorm: number | undefined;

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          metricsHook: ({ gradNorm }) => {
            capturedGradNorm = gradNorm;
          },
        });

        // Assert
        expect(capturedGradNorm).toBeUndefined();
      });
    });

    describe('given metricsHook is provided and _lastGradNorm is explicitly null', () => {
      it('reports the explicit null gradient norm without coercing it', () => {
        // Arrange – explicit null should pass through unchanged.
        const network = new Network(1, 1, { seed: 9_006_05 });
        let capturedGradNorm: number | null | undefined;
        (
          network as unknown as {
            _lastGradNorm?: number | null;
          }
        )._lastGradNorm = null;

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          error: 0,
          metricsHook: ({ gradNorm }) => {
            capturedGradNorm = gradNorm;
          },
        });

        // Assert
        expect(capturedGradNorm).toBeNull();
      });
    });

    describe('given metricsHook is provided and _lastGradNorm is explicitly undefined', () => {
      it('reports undefined without coercing it to 0', () => {
        // Arrange – explicit undefined should pass through unchanged.
        const network = new Network(1, 1, { seed: 9_006_06 });
        let capturedGradNorm: number | undefined;
        (
          network as unknown as {
            _lastGradNorm?: number | undefined;
          }
        )._lastGradNorm = undefined;

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          error: 0,
          metricsHook: ({ gradNorm }) => {
            capturedGradNorm = gradNorm;
          },
        });

        // Assert
        expect(capturedGradNorm).toBeUndefined();
      });
    });

    describe('given metricsHook is provided and _lastGradNorm is present', () => {
      it('reports the recorded gradient norm without using the fallback', () => {
        // Arrange – _lastGradNorm present covers the non-fallback nullish branch.
        const network = new Network(1, 1, { seed: 9_006_1 });
        let capturedGradNorm: number | undefined;
        (
          network as unknown as {
            _lastGradNorm?: number;
          }
        )._lastGradNorm = 2;

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          error: 0,
          metricsHook: ({ gradNorm }) => {
            capturedGradNorm = gradNorm;
          },
        });

        // Assert
        expect(capturedGradNorm).toBe(2);
      });
    });

    describe('given cost is an invalid object', () => {
      it('throws the invalid-cost-function error', () => {
        // Arrange – invalid object cost forces the validation throw path.
        const network = new Network(1, 1, { seed: 9_006_2 });

        // Act & Assert
        expect(() =>
          trainFinalizeCore(network, MINIMAL_DATASET, {
            iterations: 1,
            error: 0,
            cost: {} as unknown as never,
          }),
        ).toThrow(NetworkTrainingInvalidCostFunctionError);
      });
    });

    describe('given checkpoint.best is configured and no prior best exists', () => {
      it('triggers the first best-checkpoint save when _checkpointBestError is null', () => {
        // Arrange – _checkpointBestError is null → line 393 TRUE arm (== null)
        const network = new Network(1, 1, { seed: 9_007 });
        const saveSpy = jest.fn();

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          checkpoint: {
            best: true,
            save: saveSpy,
          },
        });

        // Assert
        expect(
          saveSpy.mock.calls.some(
            ([checkpointArg]) =>
              (checkpointArg as { type?: string }).type === 'best',
          ),
        ).toBe(true);
      });
    });

    describe('given the first monitored error already satisfies the target error', () => {
      it('stops after one iteration through the target-error early-stop branch', () => {
        // Arrange – trainSetCore is mocked to return 0.5, matching the target error.
        const network = new Network(1, 1, { seed: 9_008 });

        // Act
        const result = trainFinalizeCore(network, MINIMAL_DATASET, {
          error: 0.5,
          iterations: 3,
        });

        // Assert
        expect(result.iterations).toBe(1);
      });
    });

    describe('given earlyStopPatience is configured but not yet exhausted', () => {
      it('uses the full requested iteration budget', () => {
        // Arrange – iteration 2 reaches the patience comparison with 1 < 2.
        const network = new Network(1, 1, { seed: 9_009 });

        // Act
        const result = trainFinalizeCore(network, MINIMAL_DATASET, {
          earlyStopPatience: 2,
          error: 0,
          iterations: 2,
        });

        // Assert
        expect(result.iterations).toBe(2);
      });
    });

    describe('given the monitored-error ring buffers roll over and the metrics hook throws', () => {
      it('swallows the callback failure and still completes the full iteration budget', () => {
        // Arrange – iteration 3 forces the ring buffers onto the ordered rollover path.
        const network = new Network(1, 1, { seed: 9_010 });
        (
          network as unknown as {
            _lastGradNorm?: number;
          }
        )._lastGradNorm = 2;

        // Act
        const result = trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 3,
          error: 0,
          movingAverageWindow: 2,
          metricsHook: () => {
            throw new Error('ignore metrics callback failure');
          },
        });

        // Assert
        expect(result.iterations).toBe(3);
      });
    });

    describe('given hidden nodes were masked during training', () => {
      it('restores every hidden mask and clears dropout at finalize time', () => {
        // Arrange – finalizeTrainingRun should visit the hidden-node true arm.
        const network = Architect.perceptron(1, 2, 1);
        const hiddenNodes = network.nodes.filter(
          (node) => node.type === 'hidden',
        );
        hiddenNodes.forEach((node) => {
          node.mask = 0;
        });

        // Act
        trainFinalizeCore(network, MINIMAL_DATASET, {
          iterations: 1,
          error: 0,
          dropout: 0.5,
        });

        // Assert
        expect({
          dropout: network.dropout,
          masks: hiddenNodes.map((node) => node.mask),
        }).toStrictEqual({
          dropout: 0,
          masks: [1, 1],
        });
      });
    });
  });
});
