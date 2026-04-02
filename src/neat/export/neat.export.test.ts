import Network from '../../architecture/network';
import Neat from '../../neat';
import { NeatExportStateBundleValidationError } from './neat.export.errors';
import type { NeatMetaJSON, NeatStateJSON } from './neat.export';

describe('neat export chapter', () => {
  describe('meta-only restore', () => {
    describe('given a run that has already advanced generation state', () => {
      const scoreByNodeCount = (network: Network) => network.nodes.length;

      let exportedMeta: NeatMetaJSON;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, {
          popsize: 5,
          seed: 802,
        });

        await neat.evaluate();
        await neat.evolve();

        // Act
        exportedMeta = neat.toJSON();
      });

      describe('when the controller is rebuilt from meta only', () => {
        let restoredGeneration: number;
        let restoredNextGlobalInnovation: number;

        beforeAll(() => {
          // Arrange
          const restored = Neat.fromJSON(exportedMeta, scoreByNodeCount);

          // Act
          restoredGeneration = restored.generation;
          restoredNextGlobalInnovation = restored.toJSON().nextGlobalInnovation;
        });

        it('restores the exported generation counter', () => {
          // Assert
          expect(restoredGeneration).toBe(exportedMeta.generation);
        });

        it('restores the exported next-global-innovation cursor', () => {
          // Assert
          expect(restoredNextGlobalInnovation).toBe(
            exportedMeta.nextGlobalInnovation,
          );
        });
      });
    });
  });

  describe('population-only snapshots', () => {
    const scoreByNodeCount = (network: Network) => network.nodes.length;

    describe('given the controller currently has no genomes', () => {
      it('exports an empty population array', () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 0 });
        neat.population = [];

        // Act
        const exportedPopulation = neat.export();

        // Assert
        expect(exportedPopulation).toEqual([]);
      });
    });

    describe('given a saved population is imported into another controller', () => {
      it('replaces the destination population with the exported genome count', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 2,
          seed: 21,
        });
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 22,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);

        // Assert
        expect(destinationController.population.length).toBe(
          exportedPopulation.length,
        );
      });
    });

    describe('given an empty array is imported', () => {
      it('sets both runtime population and configured popsize to zero', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2 });

        // Act
        await neat.import([]);

        // Assert
        expect({
          populationSize: neat.population.length,
          popsize: neat.options.popsize,
        }).toEqual({ populationSize: 0, popsize: 0 });
      });
    });
  });

  describe('full-state restore', () => {
    describe('given an invalid state bundle', () => {
      const scoreByNodeCount = (network: Network) => network.nodes.length;

      it('throws the export bundle validation error', async () => {
        // Arrange
        const importInvalidState = async () =>
          Neat.importState(
            undefined as unknown as NeatStateJSON,
            scoreByNodeCount,
          );

        // Act
        const invalidImport = importInvalidState();

        // Assert
        await expect(invalidImport).rejects.toThrow(
          NeatExportStateBundleValidationError,
        );
      });
    });

    describe('given a saved checkpoint from an evolved run', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let exportedState: NeatStateJSON;
      let restoredGeneration: number;
      let restoredPopulationSize: number;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 1, scoreByConnectionCount, {
          popsize: 6,
          seed: 11,
        });

        await neat.evolve();
        await neat.evolve();
        exportedState = neat.exportState();

        // Act
        const restored = await Neat.importState(
          exportedState,
          scoreByConnectionCount,
        );

        restoredGeneration = restored.generation;
        restoredPopulationSize = restored.population.length;
      });

      describe('when the saved generation is compared after restore', () => {
        it('keeps the exported generation value', () => {
          // Assert
          expect(restoredGeneration).toBe(exportedState.neat.generation);
        });
      });

      describe('when the restored pool is compared to the saved checkpoint', () => {
        it('keeps the exported population size', () => {
          // Assert
          expect(restoredPopulationSize).toBe(exportedState.population.length);
        });
      });
    });
  });
});
