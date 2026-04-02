import * as methods from '../../../methods/methods';
import Network from '../network';
import type { MetricsHook } from './network.training.utils';

type TrainingSummary = { error: number; iterations: number };

const XOR_DATASET = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe('network training chapter', () => {
  describe('plateau smoothing separation', () => {
    let trainingSummary: TrainingSummary;
    let divergenceSeen = false;

    beforeAll(() => {
      const network = new Network(2, 1, { seed: 145 });
      const reduceOnPlateauPolicy = methods.Rate.reduceOnPlateau({
        patience: 3,
        factor: 0.5,
        minRate: 1e-4,
      });
      const metricsHook: MetricsHook = ({ iteration, error, plateauError }) => {
        if (
          iteration > 5 &&
          plateauError !== undefined &&
          plateauError !== error
        ) {
          divergenceSeen = true;
        }
      };

      trainingSummary = network.train(XOR_DATASET, {
        iterations: 80,
        rate: 0.2,
        movingAverageType: 'median',
        movingAverageWindow: 7,
        plateauMovingAverageType: 'ema',
        plateauMovingAverageWindow: 2,
        earlyStopPatience: 15,
        ratePolicy: reduceOnPlateauPolicy,
        metricsHook,
      });
    });

    describe('given plateau smoothing uses a faster EMA than the early-stop smoother', () => {
      describe('when training metrics are observed after the initial warm-up', () => {
        it('produces at least one plateau metric that diverges from the early-stop error', () => {
          // Assert
          expect(divergenceSeen).toBe(true);
        });
      });
    });

    describe('given training completes with the plateau-aware rate policy enabled', () => {
      describe('when the returned summary is inspected', () => {
        it('reports at least one completed iteration', () => {
          // Assert
          expect(trainingSummary.iterations).toBeGreaterThan(0);
        });

        it('reports a numeric final error', () => {
          // Assert
          expect(typeof trainingSummary.error).toBe('number');
        });
      });
    });
  });
});
