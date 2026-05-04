import type Network from '../../../architecture/network/network';
import {
  archiveParetoFrontsIfEnabled,
  MAX_PARETO_ARCHIVE_FRONTS,
  MAX_PARETO_ARCHIVE_LENGTH,
} from './multiobjective.archive';
import type { NeatLikeWithMultiObjective } from '../shared/multiobjective.types';

function createArchiveHost(
  input: {
    enabled?: boolean;
    generation?: number;
    archive?: Array<{ generation?: number; fronts: number[][] }>;
  } = {},
): NeatLikeWithMultiObjective {
  return {
    _getObjectives: () => [],
    options: {
      multiObjective: {
        enabled: input.enabled ?? true,
      },
    },
    _paretoArchive: input.archive ?? [],
    generation: input.generation,
  };
}

function createFront(genomeIds: Array<number | undefined>): Network[] {
  return genomeIds.map((genomeId) => ({ _id: genomeId }) as unknown as Network);
}

describe('neat multiobjective archive chapter', () => {
  describe('archiveParetoFrontsIfEnabled', () => {
    describe('given multi-objective archiving is disabled', () => {
      it('leaves the pareto archive unchanged', () => {
        // Arrange
        const archiveHost = createArchiveHost({ enabled: false });

        // Act
        archiveParetoFrontsIfEnabled(archiveHost, [createFront([101])]);

        // Assert
        expect(archiveHost._paretoArchive).toEqual([]);
      });
    });

    describe('given more fronts are supplied than the compact archive keeps', () => {
      it('stores only the top fronts and normalizes missing genome ids to zero', () => {
        // Arrange
        const archiveHost = createArchiveHost({ generation: 27 });
        const fronts = [
          createFront([101, undefined]),
          createFront([202]),
          createFront([303]),
          createFront([404]),
        ];

        // Act
        archiveParetoFrontsIfEnabled(archiveHost, fronts);

        // Assert
        expect(archiveHost._paretoArchive).toEqual([
          {
            generation: 27,
            fronts: [[101, 0], [202], [303]].slice(
              0,
              MAX_PARETO_ARCHIVE_FRONTS,
            ),
          },
        ]);
      });
    });

    describe('given the rolling archive already sits at the maximum length', () => {
      it('drops the oldest snapshot after appending the new one', () => {
        // Arrange
        const existingArchive = Array.from(
          { length: MAX_PARETO_ARCHIVE_LENGTH },
          (_, archiveIndex) => ({
            generation: archiveIndex,
            fronts: [[archiveIndex]],
          }),
        );
        const archiveHost = createArchiveHost({
          generation: 999,
          archive: existingArchive,
        });

        // Act
        archiveParetoFrontsIfEnabled(archiveHost, [createFront([808])]);

        // Assert
        expect({
          archiveLength: archiveHost._paretoArchive.length,
          oldestGeneration: archiveHost._paretoArchive[0]?.generation,
          newestSnapshot: archiveHost._paretoArchive.at(-1),
        }).toEqual({
          archiveLength: MAX_PARETO_ARCHIVE_LENGTH,
          oldestGeneration: 1,
          newestSnapshot: {
            generation: 999,
            fronts: [[808]],
          },
        });
      });
    });
  });
});
