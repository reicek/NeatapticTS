import Network from '../../../../architecture/network';
import Neat from '../../../../neat';
import type {
  SpeciesHistoryEntry,
  SpeciesHistoryStat,
  SpeciesHistoryStatExtended,
} from '../../../shared/neat.shared.types';

type SpeciesHistoryJsonLine = {
  stats?: unknown[];
};

function requireLatestSpeciesHistoryEntry(
  entries: SpeciesHistoryEntry[],
): SpeciesHistoryEntry {
  const latestEntry = entries.at(-1);
  if (!latestEntry) {
    throw new Error('Species history should contain at least one entry.');
  }
  return latestEntry;
}

function speciesStatHasInnovationRange(
  stat: SpeciesHistoryStat,
): stat is SpeciesHistoryStatExtended {
  return (stat as SpeciesHistoryStatExtended).innovationRange !== undefined;
}

function speciesStatHasEnabledRatio(
  stat: SpeciesHistoryStat,
): stat is SpeciesHistoryStatExtended {
  return (stat as SpeciesHistoryStatExtended).enabledRatio !== undefined;
}

describe('neat telemetry facade species chapter', () => {
  describe('getSpeciesHistory', () => {
    describe('given extended species history is enabled after one evaluation and evolution pass', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let latestSpeciesHistoryEntry: SpeciesHistoryEntry;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 1, scoreByConnectionCount, {
          popsize: 14,
          seed: 502,
          speciation: true,
          speciesAllocation: { extendedHistory: true },
        });

        await neat.evaluate();
        await neat.evolve();

        // Act
        latestSpeciesHistoryEntry = requireLatestSpeciesHistoryEntry(
          neat.getSpeciesHistory(),
        );
      });

      describe('when the latest history row is inspected', () => {
        it('exposes the innovation range extension on at least one species row', () => {
          // Assert
          expect(
            latestSpeciesHistoryEntry.stats.some(speciesStatHasInnovationRange),
          ).toBe(true);
        });

        it('exposes the enabled ratio extension on at least one species row', () => {
          // Assert
          expect(
            latestSpeciesHistoryEntry.stats.some(speciesStatHasEnabledRatio),
          ).toBe(true);
        });
      });
    });
  });
  describe('exportSpeciesHistoryCSV', () => {
    describe('given speciation history is recorded across several generations', () => {
      const scoreByNodeCount = (network: Network) => network.nodes.length;

      let speciesHistoryCsv = '';
      let headerRow = '';

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByNodeCount, {
          popsize: 30,
          seed: 5,
          speciation: true,
          speciesAllocation: { extendedHistory: true, minOffspring: 1 },
        });

        for (let generationIndex = 0; generationIndex < 6; generationIndex++) {
          await neat.evaluate();
          await neat.evolve();
        }

        // Act
        speciesHistoryCsv = neat.exportSpeciesHistoryCSV();
        [headerRow] = speciesHistoryCsv.split(/\r?\n/);
      });

      describe('when the exported species-history payload is inspected', () => {
        it('returns a non-empty CSV payload', () => {
          // Assert
          expect(speciesHistoryCsv.length).toBeGreaterThan(0);
        });

        it('includes the generation column in the header row', () => {
          // Assert
          expect(headerRow).toMatch(/generation/);
        });

        it('includes the species id column in the header row', () => {
          // Assert
          expect(headerRow).toMatch(/id/);
        });

        it('includes the size column in the header row', () => {
          // Assert
          expect(headerRow).toMatch(/size/);
        });
      });
    });
  });

  describe('exportSpeciesHistoryJSONL', () => {
    describe('given speciation history is recorded across several generations', () => {
      const scoreByNodeCount = (network: Network) => network.nodes.length;

      let parsedSpeciesHistoryLine: SpeciesHistoryJsonLine = {};

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByNodeCount, {
          popsize: 30,
          seed: 6,
          speciation: true,
          speciesAllocation: { extendedHistory: true, minOffspring: 1 },
        });

        for (let generationIndex = 0; generationIndex < 4; generationIndex++) {
          await neat.evaluate();
          await neat.evolve();
        }

        // Act
        parsedSpeciesHistoryLine = JSON.parse(
          neat.exportSpeciesHistoryJSONL().split(/\r?\n/).at(0) ?? '{}',
        ) as SpeciesHistoryJsonLine;
      });

      describe('when the exported species-history JSONL payload is inspected', () => {
        it('serializes at least one species stats array into the first JSONL row', () => {
          // Assert
          expect(Array.isArray(parsedSpeciesHistoryLine.stats)).toBe(true);
        });
      });
    });
  });
});
