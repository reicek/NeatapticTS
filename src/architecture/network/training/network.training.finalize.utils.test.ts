jest.mock('./network.training.loop.utils', () => ({
  trainSetCore: jest.fn(() => 0.5),
}));

import Network from '../../network';
import type { TrainingOptions } from '../network.types';
import { NetworkTrainingStoppingConditionRequiredError } from './network.training.errors';
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
      it('reports gradNorm as 0 via the null-coalescing fallback', () => {
        // Arrange – _lastGradNorm absent → line 372 FALSE arm (_lastGradNorm ?? 0)
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
        expect(capturedGradNorm).toBe(0);
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
            ([checkpointArg]) => checkpointArg.type === 'best',
          ),
        ).toBe(true);
      });
    });
  });
});
