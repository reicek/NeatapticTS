import * as validateModule from '../validate/neat.validate';
import { NeatExportPopulationValidationError } from './neat.export.errors';
import {
  LEGACY_CHECKPOINT_FORMAT_VERSION,
  type GenomeControllerCarrier,
  type NeatMetaJSON,
  type NeatStateJSON,
} from './neat.export.types';
import {
  assertCheckpointGenomeIsNative,
  assertSerializedGenomeCarriesCheckpointIdentity,
  findNextGenomeIdFloor,
  resolveMetaFormatVersion,
  resolveStateFormatVersion,
} from './neat.export.utils';

function createGenomeCarrier(): GenomeControllerCarrier {
  return {
    toJSON() {
      return {};
    },
  };
}

function createValidCheckpointPayload(): Record<string, unknown> {
  return {
    nodes: [{ geneId: 1 }, { geneId: 2 }],
    connections: [{ innovation: 7, fromGeneId: 1, toGeneId: 2 }],
  };
}

function captureErrorSnapshot(runAction: () => void): {
  message: string;
  name: string;
} {
  try {
    runAction();
  } catch (error: unknown) {
    if (error instanceof Error) {
      return { message: error.message, name: error.name };
    }

    return { message: String(error), name: 'UnknownError' };
  }

  return { message: 'No error thrown', name: 'NoError' };
}

describe('neat export utility chapter', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('assertCheckpointGenomeIsNative()', () => {
    describe('when native validation accepts the genome', () => {
      it('returns without throwing', () => {
        // Arrange
        jest.spyOn(validateModule, 'validateNativeGenome').mockReturnValue({
          connectionCount: 0,
          genomeId: undefined,
          isValid: true,
          issues: [],
          nodeCount: 0,
          topologyIntent: 'feed-forward',
        });
        const genome = createGenomeCarrier();

        // Act
        const validateGenome = () =>
          assertCheckpointGenomeIsNative(genome, 2, 'export');

        // Assert
        expect(validateGenome).not.toThrow();
      });
    });

    describe('when native validation reports an explicit first issue', () => {
      it('throws the population validation error with that issue message', () => {
        // Arrange
        jest.spyOn(validateModule, 'validateNativeGenome').mockReturnValue({
          connectionCount: 0,
          genomeId: undefined,
          isValid: false,
          issues: [
            {
              code: 'missing-node-gene-id',
              message: 'Native genomes must assign a finite geneId to every runtime node.',
              path: 'nodes[0].geneId',
            },
          ],
          nodeCount: 0,
          topologyIntent: 'feed-forward',
        });
        const genome = createGenomeCarrier();

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertCheckpointGenomeIsNative(genome, 3, 'import'),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot import population genome 3: Native genomes must assign a finite geneId to every runtime node..',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });

    describe('when native validation reports no detailed issues', () => {
      it('falls back to the generic native validation message', () => {
        // Arrange
        jest.spyOn(validateModule, 'validateNativeGenome').mockReturnValue({
          connectionCount: 0,
          genomeId: undefined,
          isValid: false,
          issues: [],
          nodeCount: 0,
          topologyIntent: 'feed-forward',
        });
        const genome = createGenomeCarrier();

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertCheckpointGenomeIsNative(genome, 4, 'export'),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot export population genome 4: native genome validation failed.',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });
  });

  describe('assertSerializedGenomeCarriesCheckpointIdentity()', () => {
    describe('when the serialized payload carries every required identity field', () => {
      it('returns without throwing', () => {
        // Arrange
        const checkpointPayload = createValidCheckpointPayload();

        // Act
        const assertIdentity = () =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 0);

        // Assert
        expect(assertIdentity).not.toThrow();
      });
    });

    describe('when the serialized payload omits node and connection arrays', () => {
      it('treats the missing arrays as empty and returns without throwing', () => {
        // Arrange
        const checkpointPayload = {
          connections: null,
          nodes: null,
        } as unknown as Record<string, unknown>;

        // Act
        const assertIdentity = () =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 0);

        // Assert
        expect(assertIdentity).not.toThrow();
      });
    });

    describe('when a serialized node omits geneId', () => {
      it('throws the missing node identity error', () => {
        // Arrange
        const checkpointPayload = {
          ...createValidCheckpointPayload(),
          nodes: [{}],
        };

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 1),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot import population genome 1: serialized node 0 is missing an explicit geneId.',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });

    describe('when a serialized connection is not an object payload', () => {
      it('throws the object-payload validation error', () => {
        // Arrange
        const checkpointPayload = {
          ...createValidCheckpointPayload(),
          connections: [null],
        };

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 2),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot import population genome 2: serialized connection 0 must be an object payload.',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });

    describe('when a serialized connection omits innovation', () => {
      it('throws the missing innovation-id validation error', () => {
        // Arrange
        const checkpointPayload = {
          ...createValidCheckpointPayload(),
          connections: [{ fromGeneId: 1, toGeneId: 2 }],
        };

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 3),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot import population genome 3: serialized connection 0 is missing an explicit innovation id.',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });

    describe('when a serialized connection omits fromGeneId', () => {
      it('throws the missing source-gene validation error', () => {
        // Arrange
        const checkpointPayload = {
          ...createValidCheckpointPayload(),
          connections: [{ innovation: 7, toGeneId: 2 }],
        };

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 4),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot import population genome 4: serialized connection 0 is missing fromGeneId.',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });

    describe('when a serialized connection omits toGeneId', () => {
      it('throws the missing target-gene validation error', () => {
        // Arrange
        const checkpointPayload = {
          ...createValidCheckpointPayload(),
          connections: [{ innovation: 7, fromGeneId: 1 }],
        };

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 5),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot import population genome 5: serialized connection 0 is missing toGeneId.',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });

    describe('when a gated serialized connection omits gaterGeneId', () => {
      it('throws the missing gater-gene validation error', () => {
        // Arrange
        const checkpointPayload = {
          ...createValidCheckpointPayload(),
          connections: [
            {
              innovation: 7,
              fromGeneId: 1,
              gater: 3,
              toGeneId: 2,
            },
          ],
        };

        // Act
        const errorSnapshot = captureErrorSnapshot(() =>
          assertSerializedGenomeCarriesCheckpointIdentity(checkpointPayload, 6),
        );

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Cannot import population genome 6: serialized connection 0 is missing gaterGeneId.',
          name: 'NeatExportPopulationValidationError',
        });
      });
    });
  });

  describe('findNextGenomeIdFloor()', () => {
    describe('when only some genomes carry explicit ids', () => {
      it('returns one above the maximum observed genome id', () => {
        // Arrange
        const population = [
          { _id: 4 },
          {},
          { _id: 11 },
        ] as GenomeControllerCarrier[];

        // Act
        const nextGenomeIdFloor = findNextGenomeIdFloor(population);

        // Assert
        expect(nextGenomeIdFloor).toBe(12);
      });
    });
  });

  describe('resolveMetaFormatVersion()', () => {
    describe('when the checkpoint meta omits a format version', () => {
      it('falls back to the legacy checkpoint format version', () => {
        // Arrange
        const neatMeta = {} as NeatMetaJSON;

        // Act
        const formatVersion = resolveMetaFormatVersion(neatMeta);

        // Assert
        expect(formatVersion).toBe(LEGACY_CHECKPOINT_FORMAT_VERSION);
      });
    });

    describe('when the checkpoint meta carries an explicit format version', () => {
      it('returns that explicit format version', () => {
        // Arrange
        const neatMeta = { formatVersion: 7 } as NeatMetaJSON;

        // Act
        const formatVersion = resolveMetaFormatVersion(neatMeta);

        // Assert
        expect(formatVersion).toBe(7);
      });
    });
  });

  describe('resolveStateFormatVersion()', () => {
    describe('when the full checkpoint omits a format version', () => {
      it('falls back to the legacy checkpoint format version', () => {
        // Arrange
        const stateBundle = {} as NeatStateJSON;

        // Act
        const formatVersion = resolveStateFormatVersion(stateBundle);

        // Assert
        expect(formatVersion).toBe(LEGACY_CHECKPOINT_FORMAT_VERSION);
      });
    });

    describe('when the full checkpoint carries an explicit format version', () => {
      it('returns that explicit format version', () => {
        // Arrange
        const stateBundle = { formatVersion: 9 } as NeatStateJSON;

        // Act
        const formatVersion = resolveStateFormatVersion(stateBundle);

        // Assert
        expect(formatVersion).toBe(9);
      });
    });
  });
});