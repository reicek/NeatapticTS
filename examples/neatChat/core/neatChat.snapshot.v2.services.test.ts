import {
  createNeatChatSeedNetwork,
  exportNeatChatPortablePayload as exportNeatChatPortablePayloadFromIndex,
  exportNeatChatSessionV2 as exportNeatChatSessionV2FromIndex,
  importNeatChatSessionV2 as importNeatChatSessionV2FromIndex,
  createNeatChatSession,
  runNeatChatExchange,
} from '../index.ts';
import {
  NeatChatNonNegativeIntegerValidationError,
  NeatChatPositiveIntegerValidationError,
  NeatChatSnapshotShapeError,
  NeatChatSnapshotVersionError,
} from './neatChat.errors';
import {
  exportPortableInferencePayload,
  fromParameterVector,
  toParameterVector,
} from '../../../src/neataptic.ts';
import {
  exportNeatChatPortablePayload,
  exportNeatChatSessionV2,
  importNeatChatSessionV2,
} from './neatChat.snapshot.v2.services';
import { createNeatChatEpisodicMemoryBank } from './neatChat.memory.services';

const SNAPSHOT_TEST_RETAINED_TERMS = [
  'hello',
  'there',
  'general',
  'kenobi',
  'friend',
  'response',
] as const;

describe('neatChat snapshot v2 services', () => {
  describe('exportNeatChatSessionV2', () => {
    it('returns a bundle with formatVersion 2', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const bundle = exportNeatChatSessionV2(session);

      // Assert
      expect(bundle.formatVersion).toBe(2);
    });

    it('preserves learnedExchangeCount', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const bundle = exportNeatChatSessionV2(session);

      // Assert
      expect(bundle.learnedExchangeCount).toBe(session.learnedExchangeCount);
    });

    it('preserves learnedTokenPairCount', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const bundle = exportNeatChatSessionV2(session);

      // Assert
      expect(bundle.learnedTokenPairCount).toBe(session.learnedTokenPairCount);
    });

    it('preserves seededTokenPairCount', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const bundle = exportNeatChatSessionV2(session);

      // Assert
      expect(bundle.seededTokenPairCount).toBe(session.seededTokenPairCount);
    });

    it('preserves contextWindowTokenCount', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const bundle = exportNeatChatSessionV2(session);

      // Assert
      expect(bundle.contextWindowTokenCount).toBe(
        session.contextWindowTokenCount,
      );
    });

    it('includes a non-empty parameterVector field', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const bundle = exportNeatChatSessionV2(session);

      // Assert
      expect({
        hasParameterVector: typeof bundle.parameterVector !== 'undefined',
        hasParameterValues: bundle.parameterVector.values.length > 0,
      }).toEqual({
        hasParameterVector: true,
        hasParameterValues: true,
      });
    });

    it('includes extensions.neatchat.vocabularySize', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const bundle = exportNeatChatSessionV2(session);

      // Assert
      expect(bundle.extensions?.neatchat?.vocabularySize).toBe(
        session.vocabulary.size,
      );
    });
  });

  describe('importNeatChatSessionV2', () => {
    it('round-trips vocabulary size', () => {
      // Arrange
      const session = createSnapshotSession();
      const bundle = exportNeatChatSessionV2(session);

      // Act
      const importedSession = importNeatChatSessionV2(bundle);

      // Assert
      expect(importedSession.vocabulary.size).toBe(session.vocabulary.size);
    });

    it('round-trips learnedExchangeCount', () => {
      // Arrange
      const session = createSnapshotSession();
      const bundle = exportNeatChatSessionV2(session);

      // Act
      const importedSession = importNeatChatSessionV2(bundle);

      // Assert
      expect(importedSession.learnedExchangeCount).toBe(
        session.learnedExchangeCount,
      );
    });

    it('round-trips retainedTerms length', () => {
      // Arrange
      const session = createSnapshotSession();
      const bundle = exportNeatChatSessionV2(session);

      // Act
      const importedSession = importNeatChatSessionV2(bundle);

      // Assert
      expect(importedSession.vocabulary.indexToTerm.length - 4).toBe(
        bundle.retainedTerms.length,
      );
    });

    it('round-trips indexToTerm symmetry', () => {
      // Arrange
      const session = createSnapshotSession();
      const bundle = exportNeatChatSessionV2(session);

      // Act
      const importedSession = importNeatChatSessionV2(bundle);

      // Assert
      expect(importedSession.vocabulary.indexToTerm[1]).toBe(
        session.vocabulary.indexToTerm[1],
      );
    });

    it('rejects formatVersion 1 with NeatChatSnapshotVersionError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        formatVersion: 1,
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotVersionError);
    });

    it('rejects missing retainedTerms with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        retainedTerms: undefined,
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects missing networkJson or parameterVector with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        networkJson: undefined,
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('round-trips non-empty memoryBank records and lastConsolidatedAt', () => {
      // Arrange
      const session = {
        ...createSnapshotSession(),
        memoryBank: {
          records: [
            {
              key: 'favorite color',
              value: 'blue',
              addedAt: 5,
              hitCount: 2,
            },
          ],
          maxRecords: 6,
          lastConsolidatedAt: 99,
        },
      };
      const bundle = exportNeatChatSessionV2(session);

      // Act
      const importedSession = importNeatChatSessionV2(bundle);

      // Assert
      expect(importedSession.memoryBank).toEqual(session.memoryBank);
    });

    it('defaults memoryBank when lastConsolidatedAt is not finite', () => {
      // Arrange
      const invalidBundle = createMemoryBankBundle({
        records: [],
        maxRecords: 6,
        lastConsolidatedAt: 'nope',
      });
      const defaultMemoryBank = createNeatChatEpisodicMemoryBank();

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.memoryBank).toEqual(defaultMemoryBank);
    });

    it('defaults memoryBank when maxRecords is not finite', () => {
      // Arrange
      const invalidBundle = createMemoryBankBundle({
        records: [],
        maxRecords: Number.POSITIVE_INFINITY,
        lastConsolidatedAt: null,
      });
      const defaultMemoryBank = createNeatChatEpisodicMemoryBank();

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.memoryBank).toEqual(defaultMemoryBank);
    });

    it('defaults memoryBank when one record is not an object', () => {
      // Arrange
      const invalidBundle = createMemoryBankBundle({
        records: [null],
        maxRecords: 6,
        lastConsolidatedAt: null,
      });
      const defaultMemoryBank = createNeatChatEpisodicMemoryBank();

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.memoryBank).toEqual(defaultMemoryBank);
    });

    it('defaults memoryBank when one record has non-finite counters', () => {
      // Arrange
      const invalidBundle = createMemoryBankBundle({
        records: [
          {
            key: 'favorite color',
            value: 'blue',
            addedAt: Number.POSITIVE_INFINITY,
            hitCount: 1,
          },
        ],
        maxRecords: 6,
        lastConsolidatedAt: null,
      });
      const defaultMemoryBank = createNeatChatEpisodicMemoryBank();

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.memoryBank).toEqual(defaultMemoryBank);
    });

    it('defaults candidateLog when extensions.neatchat.candidateLog is not an array', () => {
      // Arrange
      const invalidBundle = createCandidateLogBundle('nope');

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.candidateLog).toEqual([]);
    });

    it('defaults candidateLog when one candidate-log entry is not an object', () => {
      // Arrange
      const invalidBundle = createCandidateLogBundle([null]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.candidateLog).toEqual([]);
    });

    it('defaults candidateLog when evaluationScores is not a numeric record', () => {
      // Arrange
      const invalidBundle = createCandidateLogBundle([
        {
          decidedAt: 42,
          evaluationScores: null,
          status: 'promoted',
          trainedOnExchangeCount: 1,
        },
      ]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.candidateLog).toEqual([]);
    });

    it('preserves valid rejected candidateLog entries during import', () => {
      // Arrange
      const candidateLogEntry = {
        decidedAt: 42,
        evaluationScores: {
          heldOutAccuracy: 0.5,
          repetitionRate: 0.25,
        },
        status: 'rejected' as const,
        trainedOnExchangeCount: 1,
      };
      const validBundle = createCandidateLogBundle([candidateLogEntry]);

      // Act
      const importedSession = importNeatChatSessionV2(validBundle);

      // Assert
      expect(importedSession.candidateLog).toEqual([candidateLogEntry]);
    });

    it('preserves valid routingLog entries when responsesByPath is omitted', () => {
      // Arrange
      const routingLogEntry = {
        candidateCount: 1,
        comparedCandidatePaths: ['base'],
        decidedAt: 42,
        scores: { base: 0.5 },
        selectedPath: 'base' as const,
      };
      const validBundle = createRoutingLogBundle([routingLogEntry]);

      // Act
      const importedSession = importNeatChatSessionV2(validBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([routingLogEntry]);
    });

    it('defaults routingLog when one routing-log entry is not an object', () => {
      // Arrange
      const invalidBundle = createRoutingLogBundle([null]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });

    it('defaults routingLog when selectedPath is invalid', () => {
      // Arrange
      const invalidBundle = createRoutingLogBundle([
        {
          candidateCount: 1,
          comparedCandidatePaths: ['base'],
          decidedAt: 42,
          scores: { base: 0.5 },
          selectedPath: 'specialist',
        },
      ]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });

    it('defaults routingLog when comparedCandidatePaths contains an invalid path', () => {
      // Arrange
      const invalidBundle = createRoutingLogBundle([
        {
          candidateCount: 2,
          comparedCandidatePaths: ['base', 'specialist'],
          decidedAt: 42,
          scores: {
            base: 0.5,
            personalized: 0.25,
          },
          selectedPath: 'base',
        },
      ]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });

    it('defaults routingLog when candidateCount does not match comparedCandidatePaths length', () => {
      // Arrange
      const invalidBundle = createRoutingLogBundle([
        {
          candidateCount: 3,
          comparedCandidatePaths: ['base', 'personalized'],
          decidedAt: 42,
          scores: {
            base: 0.5,
            personalized: 0.25,
          },
          selectedPath: 'base',
        },
      ]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });

    it('defaults routingLog when retrievedMemoryKeysByPath is malformed', () => {
      // Arrange
      const invalidBundle = createRoutingLogBundle([
        {
          candidateCount: 1,
          comparedCandidatePaths: ['base'],
          decidedAt: 42,
          retrievedMemoryKeysByPath: {
            base: [1],
          },
          scores: { base: 0.5 },
          selectedPath: 'base',
        },
      ]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });

    it('defaults routingLog when retrievedMemoryCountByPath is malformed', () => {
      // Arrange
      const invalidBundle = createRoutingLogBundle([
        {
          candidateCount: 1,
          comparedCandidatePaths: ['base'],
          decidedAt: 42,
          retrievedMemoryCountByPath: {
            specialist: 1,
          },
          scores: { base: 0.5 },
          selectedPath: 'base',
        },
      ]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });

    it('defaults routingLog when responsesByPath is malformed', () => {
      // Arrange
      const invalidBundle = createRoutingLogBundle([
        {
          candidateCount: 1,
          comparedCandidatePaths: ['base'],
          decidedAt: 42,
          responsesByPath: {
            base: 7,
          },
          scores: { base: 0.5 },
          selectedPath: 'base',
        },
      ]);

      // Act
      const importedSession = importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });
  });

  describe('worker payload and parameter-vector seams', () => {
    it('re-exports the v2 snapshot surface from the public NEATchat index module', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const importedSession = importNeatChatSessionV2FromIndex(
        exportNeatChatSessionV2FromIndex(session),
      );
      const payload = exportNeatChatPortablePayloadFromIndex(session);

      // Assert
      expect({
        importedVocabularySize: importedSession.vocabulary.size,
        payloadInputCount: payload.inputCount,
      }).toEqual({
        importedVocabularySize: session.vocabulary.size,
        payloadInputCount: session.vocabulary.size,
      });
    });

    it('exportNeatChatPortablePayload matches the direct public worker payload export', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const payload = exportNeatChatPortablePayload(session);
      const directPayload = exportPortableInferencePayload(session.network);

      // Assert
      expect({
        inputCount: payload.inputCount,
        outputCount: payload.outputCount,
      }).toEqual({
        inputCount: directPayload.inputCount,
        outputCount: directPayload.outputCount,
      });
    });

    it('exportPortableInferencePayload on the session network exposes the vocabulary-sized inputCount', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const payload = exportPortableInferencePayload(session.network);

      // Assert
      expect(payload.inputCount).toBe(session.vocabulary.size);
    });

    it('toParameterVector on the LSTM session network returns non-empty values', () => {
      // Arrange
      const session = createSnapshotSession();

      // Act
      const parameterVector = toParameterVector(session.network);

      // Assert
      expect(parameterVector.values.length).toBeGreaterThan(0);
    });

    it('toParameterVector and fromParameterVector round-trip preserves network output', () => {
      // Arrange
      const session = createSnapshotSession();
      const activationInput = createOneHotInput(
        session.vocabulary.termToIndex.get('hello') ?? 0,
        session.vocabulary.size,
      );
      session.network.clear();
      const originalOutput = session.network.activate(activationInput);
      const parameterVector = toParameterVector(session.network);

      // Act
      fromParameterVector(session.network, parameterVector);
      session.network.clear();
      const restoredOutput = session.network.activate(activationInput);

      // Assert
      expect(
        Math.abs((restoredOutput[0] ?? 0) - (originalOutput[0] ?? 0)) < 1e-10,
      ).toBe(true);
    });

    it('importNeatChatSessionV2 accepts runtime ParameterVector objects from the public serializer surface', () => {
      // Arrange
      const session = createSnapshotSession();
      const exportedBundle = exportNeatChatSessionV2(session);
      const runtimeParameterVectorBundle = {
        ...exportedBundle,
        parameterVector: toParameterVector(session.network),
      };

      // Act
      const importedSession = importNeatChatSessionV2(
        runtimeParameterVectorBundle,
      );

      // Assert
      expect(importedSession.vocabulary.size).toBe(session.vocabulary.size);
    });
  });

  describe('validation guards', () => {
    it('rejects non-object bundles with NeatChatSnapshotShapeError', () => {
      // Arrange
      const importAction = () => importNeatChatSessionV2(null);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects exchanges that are not arrays with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        exchanges: 'nope',
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects non-object exchange entries with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        exchanges: [null],
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects exchange userTokens that are not arrays with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        exchanges: [
          {
            response: 'hello',
            responseTokens: [],
            trainedTokenPairCount: 1,
            userMessage: 'hi',
            userTokens: 'nope',
          },
        ],
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects exchange responseTokens that are not arrays with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        exchanges: [
          {
            response: 'hello',
            responseTokens: 'nope',
            trainedTokenPairCount: 1,
            userMessage: 'hi',
            userTokens: [],
          },
        ],
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects non-positive trainedTokenPairCount values with NeatChatPositiveIntegerValidationError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        exchanges: [
          {
            response: 'hello',
            responseTokens: ['hello'],
            trainedTokenPairCount: 0,
            userMessage: 'hi',
            userTokens: ['hi'],
          },
        ],
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatPositiveIntegerValidationError);
    });

    it('rejects missing extensions objects with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        extensions: undefined,
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects missing extensions.neatchat objects with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        extensions: {},
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects mismatched extensions.neatchat.vocabularySize values with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        extensions: {
          neatchat: {
            vocabularySize: 999,
          },
        },
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects non-object parameterVector payloads with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        parameterVector: 42,
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects parameterVector.layoutEntries values that are not arrays with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.layoutEntries = 'nope';
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects non-object parameterVector layout entries with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.layoutEntries = [null];
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects unsupported parameter-vector layout versions with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.layoutVersion = 2;
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects parameterVector values collections that are not arrays or Float64Array values with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.values = 'nope';
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects parameter vectors whose values length does not match layoutEntries length', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        const originalValues = bundle.parameterVector
          .values as readonly number[];

        bundle.parameterVector.values = originalValues.slice(1);
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects empty descriptor hashes with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.descriptorHash = '';
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects mismatched descriptor hashes with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.descriptorHash = 'deadbeef';
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects non-finite parameter values with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.values = [Infinity];
        bundle.parameterVector.layoutEntries = [{ kind: 'bias', nodeId: 1 }];
        bundle.parameterVector.descriptorHash = 'f72f51e1';
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects invalid layout entry kinds with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        bundle.parameterVector.layoutEntries = [{ kind: 'mystery', nodeId: 1 }];
        bundle.parameterVector.values = [1];
        bundle.parameterVector.descriptorHash = 'f72f51e1';
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects non-integer learnedExchangeCount values with NeatChatSnapshotShapeError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        learnedExchangeCount: 1.5,
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatSnapshotShapeError);
    });

    it('rejects weight layouts that drop innovation descriptors after hash validation', () => {
      // Arrange
      const invalidBundle = createStructuredBundleMutation((bundle) => {
        const layoutEntries = bundle.parameterVector
          .layoutEntries as TestParameterLayoutEntry[];
        const weightEntryIndex = layoutEntries.findIndex(
          (layoutEntry) => layoutEntry.kind === 'weight',
        );
        const weightEntry = layoutEntries[weightEntryIndex] as Extract<
          TestParameterLayoutEntry,
          { kind: 'weight' }
        >;

        layoutEntries[weightEntryIndex] = {
          kind: 'weight',
          from: weightEntry.from,
          to: weightEntry.to,
        };
        bundle.parameterVector.layoutEntries = layoutEntries;
        bundle.parameterVector.descriptorHash = createTestDescriptorHash(
          layoutEntries,
          bundle.parameterVector.layoutVersion,
        );
      });

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow('ParameterVector descriptor mismatch');
    });

    it('rejects negative learnedExchangeCount values with NeatChatNonNegativeIntegerValidationError', () => {
      // Arrange
      const invalidBundle = {
        ...createCandidateV2Bundle(),
        learnedExchangeCount: -1,
      };

      // Act
      const importAction = () => importNeatChatSessionV2(invalidBundle);

      // Assert
      expect(importAction).toThrow(NeatChatNonNegativeIntegerValidationError);
    });
  });
});

function createSnapshotSession() {
  const seededSession = createNeatChatSession({
    architectureFamily: 'lstm',
    contextWindowTokenCount: 6,
    corpusRetainedTerms: SNAPSHOT_TEST_RETAINED_TERMS,
    recurrentBlockSize: 8,
    seedConversationLines: ['hello there', 'general kenobi'],
  });

  return runNeatChatExchange(seededSession, 'hello friend').updatedSession;
}

function createCandidateV2Bundle() {
  const seedNetworkResult = createNeatChatSeedNetwork({
    architectureFamily: 'lstm',
    recurrentBlockSize: 8,
    vocabularySize: SNAPSHOT_TEST_RETAINED_TERMS.length,
  });
  const parameterVector = toParameterVector(seedNetworkResult.network);

  return {
    contextWindowTokenCount: 6,
    exchanges: [],
    extensions: {
      neatchat: {
        vocabularySize: seedNetworkResult.summary.effectiveVocabularySize,
      },
    },
    formatVersion: 2 as const,
    learnedExchangeCount: 0,
    learnedTokenPairCount: 0,
    networkJson: seedNetworkResult.network.toJSON(),
    parameterVector,
    retainedTerms: [...SNAPSHOT_TEST_RETAINED_TERMS],
    seededTokenPairCount: 0,
  };
}

function createStructuredBundleMutation(
  mutateBundle: (
    bundle: {
      parameterVector: {
        descriptorHash: string;
        layoutEntries: unknown;
        layoutVersion: number;
        values: unknown;
      };
    } & Record<string, unknown>,
  ) => void,
) {
  const bundle = exportNeatChatSessionV2(
    createSnapshotSession(),
  ) as unknown as {
    parameterVector: {
      descriptorHash: string;
      layoutEntries: unknown;
      layoutVersion: number;
      values: unknown;
    };
  } & Record<string, unknown>;

  mutateBundle(bundle);

  return bundle;
}

function createMemoryBankBundle(memoryBank: unknown) {
  const bundle = exportNeatChatSessionV2(
    createSnapshotSession(),
  ) as unknown as {
    readonly extensions: {
      readonly neatchat: {
        readonly vocabularySize: number;
      };
    };
  } & Record<string, unknown>;

  return {
    ...bundle,
    extensions: {
      neatchat: {
        vocabularySize: bundle.extensions.neatchat.vocabularySize,
        memoryBank,
      },
    },
  };
}

function createCandidateLogBundle(candidateLog: unknown) {
  const bundle = exportNeatChatSessionV2(
    createSnapshotSession(),
  ) as unknown as {
    readonly extensions: {
      readonly neatchat: {
        readonly vocabularySize: number;
      };
    };
  } & Record<string, unknown>;

  return {
    ...bundle,
    extensions: {
      neatchat: {
        vocabularySize: bundle.extensions.neatchat.vocabularySize,
        candidateLog,
      },
    },
  };
}

function createRoutingLogBundle(routingLog: unknown) {
  const bundle = exportNeatChatSessionV2(
    createSnapshotSession(),
  ) as unknown as {
    readonly extensions: {
      readonly neatchat: {
        readonly vocabularySize: number;
      };
    };
  } & Record<string, unknown>;

  return {
    ...bundle,
    extensions: {
      neatchat: {
        vocabularySize: bundle.extensions.neatchat.vocabularySize,
        routingLog,
      },
    },
  };
}

type TestParameterLayoutEntry =
  | {
      readonly kind: 'bias';
      readonly nodeId: number;
    }
  | {
      readonly kind: 'weight';
      readonly from: number;
      readonly to: number;
      readonly innovation?: number;
    };

function createTestDescriptorHash(
  layoutEntries: readonly TestParameterLayoutEntry[],
  layoutVersion: number,
): string {
  const descriptorSummary = [
    `version:${String(layoutVersion)}`,
    ...layoutEntries.map(summarizeTestLayoutEntry),
  ].join('|');
  let rollingHash = 2_166_136_261;

  for (const descriptorCharacter of descriptorSummary) {
    rollingHash ^= descriptorCharacter.charCodeAt(0);
    rollingHash = Math.imul(rollingHash, 16_777_619) >>> 0;
  }

  return rollingHash.toString(16).padStart(8, '0');
}

function summarizeTestLayoutEntry(
  layoutEntry: TestParameterLayoutEntry,
): string {
  if (layoutEntry.kind === 'bias') {
    return `bias:${String(layoutEntry.nodeId)}`;
  }

  const innovationSummary =
    layoutEntry.innovation == null ? 'none' : String(layoutEntry.innovation);

  return `weight:${String(layoutEntry.from)}->${String(layoutEntry.to)}:innovation:${innovationSummary}`;
}

function createOneHotInput(activeIndex: number, inputCount: number): number[] {
  return Array.from({ length: inputCount }, (_, inputIndex) =>
    inputIndex === activeIndex ? 1 : 0,
  );
}
