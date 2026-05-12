import Network from './network';
import {
  gaussianRand,
  rebuildConnectionSlabAsync,
  getRegularizationStats,
  testNetwork,
  createMLP,
  rebuildConnections,
  describeArchitecture,
  trainSetImpl,
  __trainingInternals,
  forwardWindowed,
  forwardWindowedAsync,
} from './network.utils';

describe('network utils barrel chapter', () => {
  describe('barrel re-export surface', () => {
    describe('given gaussianRand is imported through the barrel', () => {
      describe('when called with default rng', () => {
        it('returns a finite number', () => {
          const result = gaussianRand();
          expect(Number.isFinite(result)).toBe(true);
        });
      });
    });

    describe('given rebuildConnectionSlabAsync is imported through the barrel', () => {
      describe('when resolved from the barrel', () => {
        it('is a function', () => {
          expect(typeof rebuildConnectionSlabAsync).toBe('function');
        });
      });
    });

    describe('given getRegularizationStats is imported through the barrel', () => {
      describe('when called on a small network', () => {
        it('returns an object', () => {
          const network = new Network(2, 1);
          const stats = getRegularizationStats.call(network);
          expect(typeof stats).toBe('object');
        });
      });
    });

    describe('given testNetwork is imported through the barrel', () => {
      describe('when called on a small network with a dataset', () => {
        it('returns a result with an error property', () => {
          const network = new Network(1, 1);
          const result = testNetwork.call(network, [
            { input: [0.5], output: [0.5] },
          ]);
          expect(typeof result.error).toBe('number');
        });
      });
    });

    describe('given createMLP is imported through the barrel', () => {
      describe('when building a 2-2-1 network via Network.createMLP', () => {
        it('returns a Network instance', () => {
          const network = createMLP.call(Network, 2, [2], 1);
          expect(network).toBeInstanceOf(Network);
        });
      });
    });

    describe('given rebuildConnections is imported through the barrel', () => {
      describe('when called on a network', () => {
        it('runs without throwing', () => {
          const network = new Network(2, 1);
          expect(() => rebuildConnections(network)).not.toThrow();
        });
      });
    });

    describe('given describeArchitecture is imported through the barrel', () => {
      describe('when called on a small network', () => {
        it('returns a descriptor object', () => {
          const network = new Network(2, 1);
          const descriptor = describeArchitecture(network);
          expect(typeof descriptor).toBe('object');
        });
      });
    });

    describe('given trainSetImpl is imported through the barrel', () => {
      describe('when resolved from the barrel', () => {
        it('is a function', () => {
          expect(typeof trainSetImpl).toBe('function');
        });
      });
    });

    describe('given __trainingInternals is imported through the barrel', () => {
      describe('when accessed from the barrel', () => {
        it('is an object', () => {
          expect(typeof __trainingInternals).toBe('object');
        });
      });
    });

    describe('given forwardWindowed is imported through the barrel', () => {
      describe('when resolved from the barrel', () => {
        it('is a function', () => {
          expect(typeof forwardWindowed).toBe('function');
        });
      });
    });

    describe('given forwardWindowedAsync is imported through the barrel', () => {
      describe('when resolved from the barrel', () => {
        it('is a function', () => {
          expect(typeof forwardWindowedAsync).toBe('function');
        });
      });
    });
  });
});
