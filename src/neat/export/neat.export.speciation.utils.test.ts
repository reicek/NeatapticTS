import { createInnovationTracker } from '../innovation-tracker/innovation-tracker';
import {
  restoreSpeciationCheckpoint,
  serializeSpeciationCheckpoint,
} from './neat.export.speciation.utils';
import type {
  GenomeControllerCarrier,
  NeatControllerForExport,
  NetworkClass,
  SpeciationCheckpointJSON,
} from './neat.export.types';

function createGenomeCarrier(input: {
  genomeId?: number;
} = {}): GenomeControllerCarrier {
  return {
    _id: input.genomeId,
    toJSON: () => ({
      connections: [],
      nodes: [],
    }),
  };
}

function createExportController(
  overrides: Partial<NeatControllerForExport> = {},
): NeatControllerForExport {
  return {
    input: 2,
    output: 1,
    generation: 0,
    options: {},
    population: [],
    _innovationTracker: createInnovationTracker(),
    ...overrides,
  } as NeatControllerForExport;
}

function captureErrorSnapshot(runAction: () => void): {
  message: string;
  name: string;
} {
  try {
    runAction();
  } catch (error: unknown) {
    if (error instanceof Error) {
      return {
        message: error.message,
        name: error.name,
      };
    }

    return {
      message: String(error),
      name: 'UnknownError',
    };
  }

  return {
    message: 'No error thrown',
    name: 'NoError',
  };
}

const UNEXPECTED_NETWORK_RESTORE: NetworkClass = {
  fromJSON: () => {
    throw new Error('Representative restore was not expected in this scenario.');
  },
};

describe('neat export speciation utilities chapter', () => {
  describe('serializeSpeciationCheckpoint', () => {
    describe('given the controller has no live species registry or bookkeeping maps', () => {
      it('serializes empty collections while ignoring population genomes without stable ids', () => {
        // Arrange
        const controller = createExportController({
          population: [createGenomeCarrier()],
          _compatIntegral: 0.5,
          _compatSpeciesEMA: 2,
          _nextSpeciesId: 12,
        });

        // Act
        const checkpoint = serializeSpeciationCheckpoint(controller);

        // Assert
        expect(checkpoint).toEqual({
          compatIntegral: 0.5,
          compatSpeciesEMA: 2,
          nextSpeciesId: 12,
          prevSpeciesMembers: [],
          species: [],
          speciesCreated: [],
          speciesLastStats: [],
        });
      });
    });

    describe('given one species omits a representative anchor', () => {
      it('serializes the member ids without a representative genome reference', () => {
        // Arrange
        const memberGenome = createGenomeCarrier({ genomeId: 21 });
        const controller = createExportController({
          population: [memberGenome],
          _species: [
            {
              bestScore: 8,
              id: 7,
              members: [memberGenome],
            },
          ],
        });

        // Act
        const checkpoint = serializeSpeciationCheckpoint(controller);

        // Assert
        expect(checkpoint.species).toEqual([
          {
            avgSharedFitness: undefined,
            bestScore: 8,
            generation: undefined,
            id: 7,
            lastImproved: undefined,
            memberGenomeIds: [21],
            offspring: undefined,
            representativeGenome: undefined,
            representativeGenomeId: undefined,
            sharedFitness: undefined,
          },
        ]);
      });
    });

    describe('given one species keeps a detached representative anchor from the prior generation', () => {
      it('serializes the detached representative checkpoint and the populated bookkeeping maps', () => {
        // Arrange
        const liveMember = createGenomeCarrier({ genomeId: 71 });
        const detachedRepresentative = createGenomeCarrier({ genomeId: 72 });
        const controller = createExportController({
          _prevSpeciesMembers: new Map([[7, new Set([71, 72])]]),
          _species: [
            {
              avgSharedFitness: 4.5,
              bestScore: 9,
              generation: 3,
              id: 7,
              lastImproved: 2,
              members: [liveMember],
              offspring: 1,
              representative: detachedRepresentative,
              sharedFitness: 9,
            },
          ],
          _speciesCreated: new Map([[7, 1]]),
          _speciesLastStats: new Map([
            [7, { best: 9, meanConns: 4, meanNodes: 3 }],
          ]) as NonNullable<NeatControllerForExport['_speciesLastStats']>,
          population: [liveMember],
        });

        // Act
        const checkpoint = serializeSpeciationCheckpoint(controller);

        // Assert
        expect(checkpoint).toEqual({
          compatIntegral: undefined,
          compatSpeciesEMA: undefined,
          nextSpeciesId: undefined,
          prevSpeciesMembers: [[7, [71, 72]]],
          species: [
            {
              avgSharedFitness: 4.5,
              bestScore: 9,
              generation: 3,
              id: 7,
              lastImproved: 2,
              memberGenomeIds: [71],
              offspring: 1,
              representativeGenome: {
                connections: [],
                controllerMeta: {
                  genomeId: 72,
                },
                nodes: [],
              },
              representativeGenomeId: 72,
              sharedFitness: 9,
            },
          ],
          speciesCreated: [[7, 1]],
          speciesLastStats: [[7, { best: 9, meanConns: 4, meanNodes: 3 }]],
        });
      });
    });

    describe('given one species member is missing a stable genome id', () => {
      it('throws the full-checkpoint validation error', () => {
        // Arrange
        const memberGenome = createGenomeCarrier();
        const controller = createExportController({
          population: [memberGenome],
          _species: [
            {
              id: 5,
              members: [memberGenome],
            },
          ],
        });

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          serializeSpeciationCheckpoint(controller),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Cannot export species 5 member 0 without a stable genome id.',
          name: 'NeatExportStateBundleValidationError',
        });
      });
    });
  });

  describe('restoreSpeciationCheckpoint', () => {
    describe('given the checkpoint omits species rows and bookkeeping arrays', () => {
      it('restores empty speciation collections while preserving the current next-species id', () => {
        // Arrange
        const controller = createExportController({
          _nextSpeciesId: 77,
          population: [createGenomeCarrier({ genomeId: 31 })],
        });

        // Act
        restoreSpeciationCheckpoint(
          controller,
          {} as SpeciationCheckpointJSON,
          UNEXPECTED_NETWORK_RESTORE,
        );

        // Assert
        expect({
          nextSpeciesId: controller._nextSpeciesId,
          prevSpeciesMembers: Array.from(
            controller._prevSpeciesMembers ?? [],
            ([speciesId, memberIds]) => [speciesId, Array.from(memberIds)],
          ),
          species: controller._species,
          speciesCreated: Array.from(controller._speciesCreated ?? []),
          speciesLastStats: Array.from(controller._speciesLastStats ?? []),
        }).toEqual({
          nextSpeciesId: 77,
          prevSpeciesMembers: [],
          species: [],
          speciesCreated: [],
          speciesLastStats: [],
        });
      });
    });

    describe('given one restored genome is missing a stable id', () => {
      it('throws the stable-genome-id restore error', () => {
        // Arrange
        const controller = createExportController({
          population: [createGenomeCarrier()],
        });

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          restoreSpeciationCheckpoint(
            controller,
            {} as SpeciationCheckpointJSON,
            UNEXPECTED_NETWORK_RESTORE,
          ),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Full checkpoint restore requires stable genome ids on every imported genome.',
          name: 'NeatExportStateControllerRestoreError',
        });
      });
    });

    describe('given the restored population repeats one stable genome id', () => {
      it('throws the duplicate-genome-id restore error', () => {
        // Arrange
        const controller = createExportController({
          population: [
            createGenomeCarrier({ genomeId: 41 }),
            createGenomeCarrier({ genomeId: 41 }),
          ],
        });

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          restoreSpeciationCheckpoint(
            controller,
            {} as SpeciationCheckpointJSON,
            UNEXPECTED_NETWORK_RESTORE,
          ),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Full checkpoint restore encountered duplicate genome id 41.',
          name: 'NeatExportStateControllerRestoreError',
        });
      });
    });

    describe('given one species row references a representative genome id that cannot be rebound', () => {
      it('throws the missing-representative restore error', () => {
        // Arrange
        const controller = createExportController({
          population: [createGenomeCarrier({ genomeId: 51 })],
        });

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          restoreSpeciationCheckpoint(
            controller,
            {
              species: [
                {
                  id: 8,
                  memberGenomeIds: [],
                  representativeGenomeId: 999,
                },
              ],
            },
            UNEXPECTED_NETWORK_RESTORE,
          ),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Species checkpoint 8 references missing representative genome id 999. Export the checkpoint again with the current replay-aware contract before resuming this speciation boundary.',
          name: 'NeatExportStateControllerRestoreError',
        });
      });
    });

    describe('given one species row references a previous-generation member that is no longer live', () => {
      it('creates a placeholder member that still serializes the referenced stable id', () => {
        // Arrange
        const controller = createExportController({
          population: [createGenomeCarrier({ genomeId: 61 })],
        });

        // Act
        restoreSpeciationCheckpoint(
          controller,
          {
            species: [
              {
                id: 9,
                memberGenomeIds: [999],
              },
            ],
          },
          UNEXPECTED_NETWORK_RESTORE,
        );
        const placeholderPayload = controller._species?.[0]?.members[0]?.toJSON();

        // Assert
        expect(placeholderPayload).toEqual({
          controllerMeta: {
            genomeId: 999,
          },
        });
      });
    });

    describe('given one species row carries a detached representative checkpoint', () => {
      it('rehydrates the representative anchor and restores populated replay bookkeeping', () => {
        // Arrange
        const liveMember = createGenomeCarrier({ genomeId: 81 });
        const controller = createExportController({
          population: [liveMember],
        });
        const networkClass = {
          fromJSON: jest.fn(() => createGenomeCarrier()),
        } satisfies NetworkClass;

        // Act
        restoreSpeciationCheckpoint(
          controller,
          {
            compatIntegral: 0.25,
            compatSpeciesEMA: 1.5,
            nextSpeciesId: 14,
            prevSpeciesMembers: [[7, [81, 999]]],
            species: [
              {
                avgSharedFitness: 4,
                bestScore: 10,
                generation: 6,
                id: 7,
                lastImproved: 5,
                memberGenomeIds: [81, 999],
                offspring: 2,
                representativeGenome: {
                  connections: [],
                  controllerMeta: {
                    genomeId: 999,
                  },
                  nodes: [],
                },
                representativeGenomeId: 999,
                sharedFitness: 10,
              },
            ],
            speciesCreated: [[7, 2]],
            speciesLastStats: [[7, { best: 10, meanConns: 4, meanNodes: 3 }]],
          },
          networkClass,
        );

        // Assert
        expect({
          compatIntegral: controller._compatIntegral,
          compatSpeciesEMA: controller._compatSpeciesEMA,
          fromJsonCalls: networkClass.fromJSON.mock.calls,
          nextSpeciesId: controller._nextSpeciesId,
          prevSpeciesMembers: Array.from(
            controller._prevSpeciesMembers ?? [],
            ([speciesId, memberIds]) => [speciesId, Array.from(memberIds)],
          ),
          species: (controller._species ?? []).map((species) => ({
            id: species.id,
            memberGenomeIds: species.members.map((member) => member._id),
            representativeGenomeId: species.representative?._id,
          })),
          speciesCreated: Array.from(controller._speciesCreated ?? []),
        }).toEqual({
          compatIntegral: 0.25,
          compatSpeciesEMA: 1.5,
          fromJsonCalls: [[{ connections: [], nodes: [] }]],
          nextSpeciesId: 14,
          prevSpeciesMembers: [[7, [81, 999]]],
          species: [
            {
              id: 7,
              memberGenomeIds: [81, 999],
              representativeGenomeId: 999,
            },
          ],
          speciesCreated: [[7, 2]],
        });
      });
    });
  });
});