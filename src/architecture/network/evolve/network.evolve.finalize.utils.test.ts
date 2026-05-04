import Network from '../../network/network';
import {
  adoptBestGenomeOrWarn,
  buildEvolutionSummary,
  terminateWorkersSafely,
} from './network.evolve.finalize.utils';
import type { NeatRuntime } from '../network.types';

describe('network evolve finalize helpers', () => {
  describe('adoptBestGenomeOrWarn', () => {
    describe('given a best genome and clearState is disabled', () => {
      it('adopts the best genome nodes onto the evolved network', () => {
        // Arrange
        const network = new Network(1, 1);
        const bestGenome = new Network(2, 1);
        const neatRuntime = {} as NeatRuntime;

        // Act
        adoptBestGenomeOrWarn(network, neatRuntime, bestGenome, false);

        // Assert
        expect(network.nodes).toBe(bestGenome.nodes);
      });
    });

    describe('given a best genome and clearState is enabled', () => {
      it('clears network state after adoption', () => {
        // Arrange
        const network = new Network(1, 1);
        const bestGenome = new Network(2, 1);
        const clearSpy = jest
          .spyOn(network, 'clear')
          .mockImplementation(() => undefined);
        const neatRuntime = {} as NeatRuntime;

        // Act
        adoptBestGenomeOrWarn(network, neatRuntime, bestGenome, true);

        // Assert
        expect(clearSpy).toHaveBeenCalledTimes(1);
      });
    });

    describe('given no best genome and no warning hook', () => {
      it('returns without throwing', () => {
        // Arrange
        const network = new Network(1, 1);
        const neatRuntime = {} as NeatRuntime;

        // Act and Assert
        expect(() =>
          adoptBestGenomeOrWarn(network, neatRuntime, undefined, false),
        ).not.toThrow();
      });
    });

    describe('given no best genome and warning callback succeeds', () => {
      it('invokes the warning callback once', () => {
        // Arrange
        const network = new Network(1, 1);
        const warningSpy = jest.fn();
        const neatRuntime = {
          _warnIfNoBestGenome: warningSpy,
        } as unknown as NeatRuntime;

        // Act
        adoptBestGenomeOrWarn(network, neatRuntime, undefined, false);

        // Assert
        expect(warningSpy).toHaveBeenCalledTimes(1);
      });
    });

    describe('given no best genome and a warning callback throws', () => {
      it('swallows the warning callback failure after invoking it once', () => {
        // Arrange
        const network = new Network(1, 1);
        const throwingWarningSpy = jest.fn(() => {
          throw new Error('warn failure');
        });
        const neatRuntime = {
          _warnIfNoBestGenome: throwingWarningSpy,
        } as unknown as NeatRuntime;

        // Act
        adoptBestGenomeOrWarn(network, neatRuntime, undefined, false);

        // Assert
        expect(throwingWarningSpy).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('terminateWorkersSafely', () => {
    describe('given worker terminators complete normally', () => {
      it('invokes the terminator once', () => {
        // Arrange
        const terminatorSpy = jest.fn();

        // Act
        terminateWorkersSafely({
          _workerTerminators: terminatorSpy,
        });

        // Assert
        expect(terminatorSpy).toHaveBeenCalledTimes(1);
      });
    });

    describe('given worker terminators throw during shutdown', () => {
      it('swallows the termination failure after invoking the terminator once', () => {
        // Arrange
        const throwingTerminatorSpy = jest.fn(() => {
          throw new Error('terminate failure');
        });

        // Act
        terminateWorkersSafely({
          _workerTerminators: throwingTerminatorSpy,
        });

        // Assert
        expect(throwingTerminatorSpy).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('buildEvolutionSummary', () => {
    describe('given an evolve loop completion snapshot', () => {
      it('returns a summary with error and iteration values preserved', () => {
        // Arrange
        const error = 0.25;
        const iterations = 4;
        const loopStartTime = Date.now() - 20;

        // Act
        const summary = buildEvolutionSummary(error, iterations, loopStartTime);

        // Assert
        expect({
          error: summary.error,
          iterations: summary.iterations,
        }).toEqual({ error: 0.25, iterations: 4 });
      });
    });
  });
});
