import {
  Architect,
  fromParameterVector,
  toParameterVector,
  type ParameterVector,
} from '../../../src/neataptic.ts';
import { importNeatChatSessionV2 } from './neatChat.snapshot.v2.services';

const MIN_SUPPORTED_VOCAB_SIZE = 300;
const MIN_SUPPORTED_HIDDEN_SIZE = 8;
const SMALL_MAPPING_VOCAB_SIZE = 5;
const SMALL_MAPPING_HIDDEN_SIZE = 2;
const SMALL_GRU_GATE_COUNT = 3;
const SMALL_LSTM_GATE_COUNT = 4;
const ROUND_TRIP_TOLERANCE = 1e-6;

type ExternalSeedLayerWeights = {
  readonly weightIh: readonly (readonly number[])[];
  readonly weightHh: readonly (readonly number[])[];
  readonly biasIh: readonly number[];
  readonly biasHh: readonly number[];
};

type ExternalSeedDescriptor = {
  readonly family: string;
  readonly hiddenSize: number;
  readonly layers: readonly ExternalSeedLayerWeights[];
  readonly linearBias?: readonly number[];
  readonly linearWeight?: readonly (readonly number[])[];
  readonly vocabSize: number;
};

type SeedImportErrorLike = Error & {
  readonly code?: unknown;
  readonly distillationSuggestion?: unknown;
};

type SeedImportServicesModule = {
  readonly buildSeedSnapshotFromExternalWeights: (
    descriptor: ExternalSeedDescriptor,
  ) => unknown;
  readonly mapExternalRecurrentWeightsToParameterVector: (
    network: Parameters<typeof fromParameterVector>[0],
    descriptor: ExternalSeedDescriptor,
  ) => ParameterVector;
  readonly validateNeatChatSeedFamily: (
    descriptor: ExternalSeedDescriptor,
  ) => unknown;
};

type SeedImportErrorsModule = {
  readonly NeatChatSeedImportError: new (...args: readonly unknown[]) => Error;
};

describe('neatChat seed import services', () => {
  describe('validateNeatChatSeedFamily', () => {
    it('rejects unsupported families with UNSUPPORTED_OPERATOR and distillation guidance', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const unsupportedFamilyDescriptor = createSupportedGruDescriptor({
        family: 'narx',
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.validateNeatChatSeedFamily(
          unsupportedFamilyDescriptor,
        ),
      );

      // Assert
      expect(errorSummary).toEqual({
        code: 'UNSUPPORTED_OPERATOR',
        hasDistillationSuggestion: true,
      });
    });

    it('rejects multi-layer descriptors with UNSUPPORTED_OPERATOR and distillation guidance', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const multiLayerDescriptor = createSupportedGruDescriptor({
        layers: [
          createExternalSeedLayerWeights(
            SMALL_GRU_GATE_COUNT,
            MIN_SUPPORTED_VOCAB_SIZE,
            MIN_SUPPORTED_HIDDEN_SIZE,
            0.01,
          ),
          createExternalSeedLayerWeights(
            SMALL_GRU_GATE_COUNT,
            MIN_SUPPORTED_VOCAB_SIZE,
            MIN_SUPPORTED_HIDDEN_SIZE,
            0.02,
          ),
        ],
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.validateNeatChatSeedFamily(multiLayerDescriptor),
      );

      // Assert
      expect(errorSummary).toEqual({
        code: 'UNSUPPORTED_OPERATOR',
        hasDistillationSuggestion: true,
      });
    });

    it('rejects vocabularies below the supported range with DIMENSION_MISMATCH', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const unsupportedVocabularyDescriptor = createSupportedGruDescriptor({
        vocabSize: MIN_SUPPORTED_VOCAB_SIZE - 1,
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.validateNeatChatSeedFamily(
          unsupportedVocabularyDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects hidden sizes below the supported range with DIMENSION_MISMATCH', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const unsupportedHiddenSizeDescriptor = createSupportedGruDescriptor({
        hiddenSize: MIN_SUPPORTED_HIDDEN_SIZE - 1,
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.validateNeatChatSeedFamily(
          unsupportedHiddenSizeDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects non-positive vocabularies with DIMENSION_MISMATCH', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const nonPositiveVocabularyDescriptor = createSupportedGruDescriptor({
        vocabSize: 0,
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.validateNeatChatSeedFamily(
          nonPositiveVocabularyDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects vocabularies above the supported range with DIMENSION_MISMATCH', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const unsupportedVocabularyDescriptor = createSupportedGruDescriptor({
        vocabSize: 3001,
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.validateNeatChatSeedFamily(
          unsupportedVocabularyDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects hidden sizes above the supported range with DIMENSION_MISMATCH', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const unsupportedHiddenSizeDescriptor = createSupportedGruDescriptor({
        hiddenSize: 129,
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.validateNeatChatSeedFamily(
          unsupportedHiddenSizeDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('accepts a supported single-layer GRU descriptor without throwing', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const supportedGruDescriptor = createSupportedGruDescriptor();

      // Act
      const validationSucceeded = completesWithoutThrow(() =>
        seedImportServices.validateNeatChatSeedFamily(supportedGruDescriptor),
      );

      // Assert
      expect(validationSucceeded).toBe(true);
    });

    it('accepts a supported single-layer LSTM descriptor without throwing', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const supportedLstmDescriptor = createSupportedLstmDescriptor();

      // Act
      const validationSucceeded = completesWithoutThrow(() =>
        seedImportServices.validateNeatChatSeedFamily(supportedLstmDescriptor),
      );

      // Assert
      expect(validationSucceeded).toBe(true);
    });
  });

  describe('mapExternalRecurrentWeightsToParameterVector', () => {
    it('returns a parameter vector whose values length matches the network layout length for GRU seeds', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const seedDescriptor = createTinyGruDescriptor();
      const baselineParameterVector = toParameterVector(network);

      // Act
      const mappedParameterVector =
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        );

      // Assert
      expect(mappedParameterVector.values.length).toBe(
        baselineParameterVector.values.length,
      );
    });

    it('preserves mapped GRU weights across a fromParameterVector and toParameterVector round-trip within tolerance', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const seedDescriptor = createTinyGruDescriptor();
      const mappedParameterVector =
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        );

      // Act
      fromParameterVector(network, mappedParameterVector);
      const roundTripParameterVector = toParameterVector(network);
      const maxAbsoluteDifference = computeMaxAbsoluteDifference(
        mappedParameterVector.values,
        roundTripParameterVector.values,
      );

      // Assert
      expect(maxAbsoluteDifference <= ROUND_TRIP_TOLERANCE).toBe(true);
    });

    it('rejects mismatched hidden sizes with DIMENSION_MISMATCH', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const mismatchedDescriptor = createTinyGruDescriptor({ hiddenSize: 3 });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          mismatchedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects native vocab layouts that do not match descriptor vocab size', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE + 1,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE + 1,
      );
      const seedDescriptor = createTinyGruDescriptor();

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects native GRU layouts with an input-to-output shortcut as unsupported', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
        { inputToOutput: true },
      );
      const seedDescriptor = createTinyGruDescriptor();

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary).toEqual({
        code: 'UNSUPPORTED_OPERATOR',
        hasDistillationSuggestion: true,
      });
    });

    it('rejects native bias layouts that do not match the descriptor family', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.lstm(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const seedDescriptor = createTinyGruDescriptor();

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary).toEqual({
        code: 'UNSUPPORTED_OPERATOR',
        hasDistillationSuggestion: true,
      });
    });

    it('rejects optional linear weights with the wrong row count', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const seedDescriptor = createTinyGruDescriptor({
        linearWeight: createDeterministicMatrix(
          SMALL_MAPPING_VOCAB_SIZE - 1,
          SMALL_MAPPING_HIDDEN_SIZE,
          0.9,
        ),
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects optional linear weights with the wrong column count', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const invalidLinearWeight = createDeterministicMatrix(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        0.9,
      ).with(0, [0.9]);
      const seedDescriptor = createTinyGruDescriptor({
        linearWeight: invalidLinearWeight,
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects optional linear weights with non-finite values', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const invalidLinearWeight = createDeterministicMatrix(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        0.9,
      ).with(0, [Number.POSITIVE_INFINITY, 0.901]);
      const seedDescriptor = createTinyGruDescriptor({
        linearWeight: invalidLinearWeight,
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects optional linear biases with the wrong length', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const seedDescriptor = createTinyGruDescriptor({
        linearBias: createDeterministicVector(
          SMALL_MAPPING_VOCAB_SIZE - 1,
          0.8,
        ),
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('rejects optional linear biases with non-finite values', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const seedDescriptor = createTinyGruDescriptor({
        linearBias: createDeterministicVector(
          SMALL_MAPPING_VOCAB_SIZE,
          0.8,
        ).with(0, Number.NaN),
      });

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });

    it('maps optional GRU linear readout weights and biases into deterministic ranks', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.gru(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const linearWeight = createDeterministicMatrix(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        0.9,
      );
      const linearBias = createDeterministicVector(
        SMALL_MAPPING_VOCAB_SIZE,
        0.8,
      );
      const seedDescriptor = createTinyGruDescriptor({
        linearBias,
        linearWeight,
      });
      const linearWeightStartIndex = computeGruLinearWeightStartIndex(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
      );

      // Act
      const mappedParameterVector =
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        );

      // Assert
      expect({
        firstLinearBias: mappedParameterVector.values[SMALL_MAPPING_VOCAB_SIZE],
        firstLinearWeight: mappedParameterVector.values[linearWeightStartIndex],
        lastLinearWeight:
          mappedParameterVector.values[
            linearWeightStartIndex +
              (SMALL_MAPPING_HIDDEN_SIZE - 1) * SMALL_MAPPING_VOCAB_SIZE +
              (SMALL_MAPPING_VOCAB_SIZE - 1)
          ],
      }).toEqual({
        firstLinearBias: linearBias[0],
        firstLinearWeight: linearWeight[0]![0],
        lastLinearWeight:
          linearWeight[SMALL_MAPPING_VOCAB_SIZE - 1]![
            SMALL_MAPPING_HIDDEN_SIZE - 1
          ],
      });
    });

    it('maps LSTM gate rows into deterministic native ranks', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const network = Architect.lstm(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
        SMALL_MAPPING_VOCAB_SIZE,
      );
      const seedDescriptor = createTinyLstmDescriptor();
      const inputGateBiasStartIndex = SMALL_MAPPING_VOCAB_SIZE * 2;
      const cellGateWeightStartIndex = computeLstmCellGateWeightStartIndex(
        SMALL_MAPPING_VOCAB_SIZE,
        SMALL_MAPPING_HIDDEN_SIZE,
      );

      // Act
      const mappedParameterVector =
        seedImportServices.mapExternalRecurrentWeightsToParameterVector(
          network,
          seedDescriptor,
        );

      // Assert
      expect({
        cellGateInputWeight:
          mappedParameterVector.values[cellGateWeightStartIndex],
        inputGateBias: mappedParameterVector.values[inputGateBiasStartIndex],
      }).toEqual({
        cellGateInputWeight:
          seedDescriptor.layers[0]!.weightIh[SMALL_MAPPING_HIDDEN_SIZE * 2]![0],
        inputGateBias:
          seedDescriptor.layers[0]!.biasIh[0]! +
          seedDescriptor.layers[0]!.biasHh[0]!,
      });
    });
  });

  describe('buildSeedSnapshotFromExternalWeights', () => {
    it('stores seed metadata in the v2 snapshot extension and returns an importable snapshot', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const supportedGruDescriptor = createSupportedGruDescriptor();

      // Act
      const seedSnapshot =
        seedImportServices.buildSeedSnapshotFromExternalWeights(
          supportedGruDescriptor,
        ) as {
          readonly extensions?: {
            readonly neatchat?: {
              readonly seedMetadata?: {
                readonly conversionSource?: string;
                readonly family?: string;
              };
            };
          };
          readonly formatVersion?: number;
        };
      const importedSession = importNeatChatSessionV2(seedSnapshot);

      // Assert
      expect({
        conversionSource:
          seedSnapshot.extensions?.neatchat?.seedMetadata?.conversionSource,
        family: seedSnapshot.extensions?.neatchat?.seedMetadata?.family,
        formatVersion: seedSnapshot.formatVersion,
        importSucceeded: importedSession.network.outputNodeIds.length > 0,
      }).toMatchObject({
        conversionSource: 'external-parameter-vector',
        family: supportedGruDescriptor.family,
        formatVersion: 2,
        importSucceeded: true,
      });
    });

    it('stores LSTM seed metadata in an importable v2 snapshot', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const supportedLstmDescriptor = createSupportedLstmDescriptor();

      // Act
      const seedSnapshot =
        seedImportServices.buildSeedSnapshotFromExternalWeights(
          supportedLstmDescriptor,
        ) as {
          readonly extensions?: {
            readonly neatchat?: {
              readonly seedMetadata?: {
                readonly conversionSource?: string;
                readonly family?: string;
              };
            };
          };
          readonly formatVersion?: number;
        };
      const importedSession = importNeatChatSessionV2(seedSnapshot);

      // Assert
      expect({
        conversionSource:
          seedSnapshot.extensions?.neatchat?.seedMetadata?.conversionSource,
        family: seedSnapshot.extensions?.neatchat?.seedMetadata?.family,
        formatVersion: seedSnapshot.formatVersion,
        importSucceeded: importedSession.network.outputNodeIds.length > 0,
      }).toMatchObject({
        conversionSource: 'external-parameter-vector',
        family: supportedLstmDescriptor.family,
        formatVersion: 2,
        importSucceeded: true,
      });
    });

    it('rejects retained-term reserves that leave no external token slots', () => {
      // Arrange
      const seedImportServices =
        loadSeedImportServicesModuleWithSpecialTokenReserve(
          MIN_SUPPORTED_VOCAB_SIZE + 1,
        );
      const supportedGruDescriptor = createSupportedGruDescriptor();

      // Act
      const errorSummary = captureSeedImportErrorSummary(() =>
        seedImportServices.buildSeedSnapshotFromExternalWeights(
          supportedGruDescriptor,
        ),
      );

      // Assert
      expect(errorSummary.code).toBe('DIMENSION_MISMATCH');
    });
  });

  describe('NeatChatSeedImportError', () => {
    it('inherits from Error and exposes code plus distillation guidance on unsupported direct imports', async () => {
      // Arrange
      const seedImportServices = await loadSeedImportServicesModule();
      const seedImportErrors = await loadSeedImportErrorsModule();
      const unsupportedFamilyDescriptor = createSupportedGruDescriptor({
        family: 'narx',
      });

      // Act
      const thrownError = captureThrownError(() =>
        seedImportServices.validateNeatChatSeedFamily(
          unsupportedFamilyDescriptor,
        ),
      );

      // Assert
      expect({
        code: readSeedImportErrorCode(thrownError),
        hasDistillationSuggestion:
          hasNonEmptyDistillationSuggestion(thrownError),
        isSeedImportError:
          thrownError instanceof seedImportErrors.NeatChatSeedImportError,
        isStandardError: thrownError instanceof Error,
      }).toEqual({
        code: 'UNSUPPORTED_OPERATOR',
        hasDistillationSuggestion: true,
        isSeedImportError: true,
        isStandardError: true,
      });
    });
  });
});

async function loadSeedImportServicesModule(): Promise<SeedImportServicesModule> {
  const modulePath = './neatChat.seed-import.services.ts';

  return (await import(modulePath)) as SeedImportServicesModule;
}

async function loadSeedImportErrorsModule(): Promise<SeedImportErrorsModule> {
  const modulePath = './neatChat.seed-import.errors.ts';

  return (await import(modulePath)) as SeedImportErrorsModule;
}

function loadSeedImportServicesModuleWithSpecialTokenReserve(
  specialTokenReserveCount: number,
): SeedImportServicesModule {
  let seedImportServices: SeedImportServicesModule | undefined;

  jest.resetModules();
  jest.doMock('./neatChat.constants', () => {
    const actualConstants = jest.requireActual(
      './neatChat.constants',
    ) as typeof import('./neatChat.constants');

    return {
      ...actualConstants,
      NEATCHAT_SPECIAL_TOKENS: Array.from(
        { length: specialTokenReserveCount },
        (_, tokenIndex) => `MOCK_SPECIAL_TOKEN_${String(tokenIndex)}`,
      ),
    };
  });

  jest.isolateModules(() => {
    seedImportServices = jest.requireActual(
      './neatChat.seed-import.services',
    ) as SeedImportServicesModule;
  });

  jest.dontMock('./neatChat.constants');
  jest.resetModules();

  return seedImportServices!;
}

function captureSeedImportErrorSummary(action: () => unknown): {
  readonly code: string | null;
  readonly hasDistillationSuggestion: boolean;
} {
  const thrownError = captureThrownError(action);

  return {
    code: readSeedImportErrorCode(thrownError),
    hasDistillationSuggestion: hasNonEmptyDistillationSuggestion(thrownError),
  };
}

function captureThrownError(action: () => unknown): unknown {
  try {
    action();
    return undefined;
  } catch (error) {
    return error;
  }
}

function completesWithoutThrow(action: () => unknown): boolean {
  return captureThrownError(action) == null;
}

function readSeedImportErrorCode(thrownError: unknown): string | null {
  if (!isSeedImportErrorLike(thrownError)) {
    return null;
  }

  return typeof thrownError.code === 'string' ? thrownError.code : null;
}

function hasNonEmptyDistillationSuggestion(thrownError: unknown): boolean {
  if (!isSeedImportErrorLike(thrownError)) {
    return false;
  }

  return (
    typeof thrownError.distillationSuggestion === 'string' &&
    thrownError.distillationSuggestion.length > 0
  );
}

function isSeedImportErrorLike(
  thrownError: unknown,
): thrownError is SeedImportErrorLike {
  return thrownError instanceof Error;
}

function computeMaxAbsoluteDifference(
  leftValues: ArrayLike<number>,
  rightValues: ArrayLike<number>,
): number {
  return Array.from(leftValues, (leftValue, valueIndex) =>
    Math.abs(leftValue - (rightValues[valueIndex] ?? 0)),
  ).reduce(
    (currentMaxAbsoluteDifference, absoluteDifference) =>
      Math.max(currentMaxAbsoluteDifference, absoluteDifference),
    0,
  );
}

function createTinyGruDescriptor(
  overrides: Partial<ExternalSeedDescriptor> = {},
): ExternalSeedDescriptor {
  return createGruDescriptor({
    hiddenSize: SMALL_MAPPING_HIDDEN_SIZE,
    vocabSize: SMALL_MAPPING_VOCAB_SIZE,
    ...overrides,
  });
}

function createTinyLstmDescriptor(
  overrides: Partial<ExternalSeedDescriptor> = {},
): ExternalSeedDescriptor {
  return createLstmDescriptor({
    hiddenSize: SMALL_MAPPING_HIDDEN_SIZE,
    vocabSize: SMALL_MAPPING_VOCAB_SIZE,
    ...overrides,
  });
}

function createSupportedGruDescriptor(
  overrides: Partial<ExternalSeedDescriptor> = {},
): ExternalSeedDescriptor {
  return createGruDescriptor({
    hiddenSize: MIN_SUPPORTED_HIDDEN_SIZE,
    vocabSize: MIN_SUPPORTED_VOCAB_SIZE,
    ...overrides,
  });
}

function createSupportedLstmDescriptor(
  overrides: Partial<ExternalSeedDescriptor> = {},
): ExternalSeedDescriptor {
  return createLstmDescriptor({
    hiddenSize: MIN_SUPPORTED_HIDDEN_SIZE,
    vocabSize: MIN_SUPPORTED_VOCAB_SIZE,
    ...overrides,
  });
}

function createGruDescriptor(
  overrides: Partial<ExternalSeedDescriptor> = {},
): ExternalSeedDescriptor {
  const hiddenSize = overrides.hiddenSize ?? MIN_SUPPORTED_HIDDEN_SIZE;
  const vocabSize = overrides.vocabSize ?? MIN_SUPPORTED_VOCAB_SIZE;

  return {
    family: overrides.family ?? 'gru',
    hiddenSize,
    layers: overrides.layers ?? [
      createExternalSeedLayerWeights(
        SMALL_GRU_GATE_COUNT,
        vocabSize,
        hiddenSize,
        0.005,
      ),
    ],
    linearBias: overrides.linearBias,
    linearWeight: overrides.linearWeight,
    vocabSize,
  };
}

function createLstmDescriptor(
  overrides: Partial<ExternalSeedDescriptor> = {},
): ExternalSeedDescriptor {
  const hiddenSize = overrides.hiddenSize ?? MIN_SUPPORTED_HIDDEN_SIZE;
  const vocabSize = overrides.vocabSize ?? MIN_SUPPORTED_VOCAB_SIZE;

  return {
    family: overrides.family ?? 'lstm',
    hiddenSize,
    layers: overrides.layers ?? [
      createExternalSeedLayerWeights(
        SMALL_LSTM_GATE_COUNT,
        vocabSize,
        hiddenSize,
        0.0075,
      ),
    ],
    linearBias: overrides.linearBias,
    linearWeight: overrides.linearWeight,
    vocabSize,
  };
}

function computeGruLinearWeightStartIndex(
  vocabSize: number,
  hiddenSize: number,
): number {
  const biasCount = 2 * vocabSize + SMALL_GRU_GATE_COUNT * 2 * hiddenSize;

  return (
    biasCount +
    5 * hiddenSize * hiddenSize +
    2 * hiddenSize +
    3 * vocabSize * hiddenSize
  );
}

function computeLstmCellGateWeightStartIndex(
  vocabSize: number,
  hiddenSize: number,
): number {
  const biasCount = 2 * vocabSize + (SMALL_LSTM_GATE_COUNT + 1) * hiddenSize;

  return biasCount + 4 * hiddenSize * hiddenSize + hiddenSize;
}

function createExternalSeedLayerWeights(
  gateCount: number,
  inputCount: number,
  hiddenCount: number,
  baseValue: number,
): ExternalSeedLayerWeights {
  const totalGateRows = gateCount * hiddenCount;

  return {
    weightIh: createDeterministicMatrix(totalGateRows, inputCount, baseValue),
    weightHh: createDeterministicMatrix(
      totalGateRows,
      hiddenCount,
      baseValue + 0.25,
    ),
    biasIh: createDeterministicVector(totalGateRows, baseValue + 0.5),
    biasHh: createDeterministicVector(totalGateRows, baseValue + 0.75),
  };
}

function createDeterministicMatrix(
  rowCount: number,
  columnCount: number,
  baseValue: number,
): readonly (readonly number[])[] {
  return Array.from({ length: rowCount }, (_, rowIndex) =>
    Array.from({ length: columnCount }, (_, columnIndex) =>
      Number((baseValue + rowIndex * 0.01 + columnIndex * 0.001).toFixed(6)),
    ),
  );
}

function createDeterministicVector(
  valueCount: number,
  baseValue: number,
): readonly number[] {
  return Array.from({ length: valueCount }, (_, valueIndex) =>
    Number((baseValue + valueIndex * 0.01).toFixed(6)),
  );
}
