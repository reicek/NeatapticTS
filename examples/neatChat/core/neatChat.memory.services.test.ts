import {
  createNeatChatEpisodicMemoryBank,
  addNeatChatMemoryRecord,
  pruneNeatChatMemoryBank,
  retrieveNeatChatMemories,
} from './neatChat.memory.services';
import {
  createNeatChatSession,
  exportNeatChatSessionV2,
  importNeatChatSessionV2,
} from '../index';

type NeatChatMemoryRecordLike = {
  readonly key: string;
  readonly value: string;
  readonly addedAt: number;
  readonly hitCount: number;
};

type NeatChatEpisodicMemoryBankLike = {
  readonly records: readonly NeatChatMemoryRecordLike[];
  readonly maxRecords: number;
  readonly lastConsolidatedAt?: number;
};

type NeatChatSessionWithMemoryBankLike = ReturnType<
  typeof createNeatChatSession
> & {
  readonly memoryBank: NeatChatEpisodicMemoryBankLike;
};

const MEMORY_TEST_RETAINED_TERMS = [
  'favorite',
  'color',
  'blue',
  'ocean',
  'weekend',
  'pizza',
  'dinner',
  'pet',
  'name',
  'nova',
] as const;

describe('neatChat memory services', () => {
  describe('createNeatChatEpisodicMemoryBank', () => {
    it('creates an empty memory bank with the default session cap', () => {
      // Arrange
      const defaultSession = createNeatChatSession();
      const defaultSessionRecord = defaultSession as unknown as Record<
        string,
        unknown
      >;

      // Act
      const memoryBank = createNeatChatEpisodicMemoryBank();

      // Assert
      expect({
        records: memoryBank.records,
        maxRecords: memoryBank.maxRecords,
      }).toEqual({
        records: [],
        maxRecords: (
          defaultSessionRecord.memoryBank as
            | NeatChatEpisodicMemoryBankLike
            | undefined
        )?.maxRecords,
      });
    });

    it('accepts an explicit maxRecords override', () => {
      // Arrange
      const explicitMaxRecords = 5;

      // Act
      const memoryBank = createNeatChatEpisodicMemoryBank({
        maxRecords: explicitMaxRecords,
      });

      // Assert
      expect(memoryBank.maxRecords).toBe(explicitMaxRecords);
    });
  });

  describe('addNeatChatMemoryRecord', () => {
    it('returns a new bank and leaves the original bank unchanged', () => {
      // Arrange
      const originalBank = createNeatChatEpisodicMemoryBank({ maxRecords: 3 });
      const memoryRecord = createMemoryRecord({
        key: 'favorite color',
        value: 'blue',
      });

      // Act
      const updatedBank = addNeatChatMemoryRecord(originalBank, memoryRecord);

      // Assert
      expect({
        originalLength: originalBank.records.length,
        updatedLength: updatedBank.records.length,
      }).toEqual({
        originalLength: 0,
        updatedLength: 1,
      });
    });

    it('prunes the oldest lowest-hit record when the bank exceeds maxRecords', () => {
      // Arrange
      const originalBank: NeatChatEpisodicMemoryBankLike = {
        records: [
          createMemoryRecord({
            key: 'favorite color',
            value: 'blue',
            addedAt: 1,
            hitCount: 0,
          }),
          createMemoryRecord({
            key: 'pet name',
            value: 'nova',
            addedAt: 2,
            hitCount: 3,
          }),
        ],
        maxRecords: 2,
      };
      const incomingRecord = createMemoryRecord({
        key: 'weekend dinner',
        value: 'pizza',
        addedAt: 3,
        hitCount: 1,
      });

      // Act
      const updatedBank = addNeatChatMemoryRecord(originalBank, incomingRecord);

      // Assert
      expect(
        updatedBank.records.map(
          (memoryRecord: NeatChatMemoryRecordLike) => memoryRecord.key,
        ),
      ).toEqual(['pet name', 'weekend dinner']);
    });

    it('appends the incoming record when a zero-cap bank starts empty', () => {
      // Arrange
      const originalBank = createNeatChatEpisodicMemoryBank({ maxRecords: 0 });
      const incomingRecord = createMemoryRecord({
        key: 'favorite color',
        value: 'blue',
      });

      // Act
      const updatedBank = addNeatChatMemoryRecord(originalBank, incomingRecord);

      // Assert
      expect(updatedBank.records.map((memoryRecord) => memoryRecord.key)).toEqual(
        ['favorite color'],
      );
    });
  });

  describe('retrieveNeatChatMemories', () => {
    it('returns an empty array when the session memory bank has no records', () => {
      // Arrange
      const session = createSessionWithMemoryBank({
        records: [],
        maxRecords: 4,
      });

      // Act
      const retrievedMemories = retrieveNeatChatMemories(
        session,
        'favorite color blue',
      );

      // Assert
      expect(retrievedMemories).toEqual([]);
    });

    it('returns records ranked by descending token overlap with the prompt text', () => {
      // Arrange
      const session = createSessionWithMemoryBank({
        records: [
          createMemoryRecord({
            key: 'favorite color blue',
            value: 'ocean',
            addedAt: 1,
            hitCount: 1,
          }),
          createMemoryRecord({
            key: 'favorite color',
            value: 'weekend note',
            addedAt: 2,
            hitCount: 1,
          }),
          createMemoryRecord({
            key: 'pizza dinner',
            value: 'weekend',
            addedAt: 3,
            hitCount: 1,
          }),
        ],
        maxRecords: 4,
      });

      // Act
      const retrievedMemories = retrieveNeatChatMemories(
        session,
        'favorite color blue',
      );

      // Assert
      expect(
        retrievedMemories.map(
          (memoryRecord: NeatChatMemoryRecordLike) => memoryRecord.key,
        ),
      ).toEqual(['favorite color blue', 'favorite color', 'pizza dinner']);
    });

    it('places the highest token-overlap memory at index 0', () => {
      // Arrange
      const session = createSessionWithMemoryBank({
        records: [
          createMemoryRecord({
            key: 'weekend dinner',
            value: 'pizza',
            addedAt: 1,
            hitCount: 1,
          }),
          createMemoryRecord({
            key: 'favorite color blue',
            value: 'ocean',
            addedAt: 2,
            hitCount: 1,
          }),
          createMemoryRecord({
            key: 'pet name',
            value: 'nova',
            addedAt: 3,
            hitCount: 1,
          }),
        ],
        maxRecords: 4,
      });

      // Act
      const retrievedMemories = retrieveNeatChatMemories(
        session,
        'favorite color blue',
      );

      // Assert
      expect(retrievedMemories[0]?.key).toBe('favorite color blue');
    });

    it('respects the maxResults option', () => {
      // Arrange
      const session = createSessionWithMemoryBank({
        records: [
          createMemoryRecord({ key: 'favorite color blue', addedAt: 1 }),
          createMemoryRecord({ key: 'favorite color', addedAt: 2 }),
          createMemoryRecord({ key: 'favorite', addedAt: 3 }),
        ],
        maxRecords: 4,
      });

      // Act
      const retrievedMemories = retrieveNeatChatMemories(
        session,
        'favorite color blue',
        { maxResults: 2 },
      );

      // Assert
      expect(retrievedMemories.length).toBe(2);
    });

    it('breaks equal-overlap ties by earlier insertion time', () => {
      // Arrange
      const session = createSessionWithMemoryBank({
        records: [
          createMemoryRecord({
            key: 'favorite note',
            value: 'blue',
            addedAt: 1,
          }),
          createMemoryRecord({
            key: 'favorite update',
            value: 'ocean',
            addedAt: 2,
          }),
        ],
        maxRecords: 4,
      });

      // Act
      const retrievedMemories = retrieveNeatChatMemories(
        session,
        'favorite memory',
      );

      // Assert
      expect(retrievedMemories.map((memoryRecord) => memoryRecord.key)).toEqual([
        'favorite note',
        'favorite update',
      ]);
    });

    it('does not mutate the session network while retrieving memories', () => {
      // Arrange
      const session = createSessionWithMemoryBank({
        records: [
          createMemoryRecord({ key: 'favorite color blue', addedAt: 1 }),
          createMemoryRecord({ key: 'pet name nova', addedAt: 2 }),
        ],
        maxRecords: 4,
      });
      const originalNetworkJson = JSON.stringify(session.network.toJSON());

      // Act
      retrieveNeatChatMemories(session, 'favorite color blue');
      const retrievedNetworkJson = JSON.stringify(session.network.toJSON());

      // Assert
      expect(retrievedNetworkJson).toBe(originalNetworkJson);
    });
  });

  describe('pruneNeatChatMemoryBank', () => {
    it('returns the original bank when pruning is not needed', () => {
      // Arrange
      const originalBank: NeatChatEpisodicMemoryBankLike = {
        records: [createMemoryRecord({ key: 'favorite color', addedAt: 1 })],
        maxRecords: 1,
      };

      // Act
      const prunedBank = pruneNeatChatMemoryBank(originalBank);

      // Assert
      expect(prunedBank).toBe(originalBank);
    });

    it('removes the oldest low-hit record when the bank exceeds its cap', () => {
      // Arrange
      const overfilledBank: NeatChatEpisodicMemoryBankLike = {
        records: [
          createMemoryRecord({
            key: 'oldest low-hit',
            addedAt: 1,
            hitCount: 0,
          }),
          createMemoryRecord({
            key: 'newer low-hit',
            addedAt: 2,
            hitCount: 0,
          }),
          createMemoryRecord({
            key: 'high-hit keeper',
            addedAt: 3,
            hitCount: 2,
          }),
        ],
        maxRecords: 2,
      };

      // Act
      const prunedBank = pruneNeatChatMemoryBank(overfilledBank);

      // Assert
      expect(prunedBank.records.map((memoryRecord) => memoryRecord.key)).toEqual([
        'newer low-hit',
        'high-hit keeper',
      ]);
    });

    it('breaks equal-priority pruning ties by original index order', () => {
      // Arrange
      const overfilledBank: NeatChatEpisodicMemoryBankLike = {
        records: [
          createMemoryRecord({
            key: 'first equal-priority record',
            addedAt: 1,
            hitCount: 0,
          }),
          createMemoryRecord({
            key: 'second equal-priority record',
            addedAt: 1,
            hitCount: 0,
          }),
          createMemoryRecord({
            key: 'protected record',
            addedAt: 2,
            hitCount: 1,
          }),
        ],
        maxRecords: 2,
      };

      // Act
      const prunedBank = pruneNeatChatMemoryBank(overfilledBank);

      // Assert
      expect(prunedBank.records.map((memoryRecord) => memoryRecord.key)).toEqual([
        'second equal-priority record',
        'protected record',
      ]);
    });
  });

  describe('memory bank checkpoint integration', () => {
    it('round-trips memoryBank records length and maxRecords through v2 snapshots', () => {
      // Arrange
      const session = createSessionWithMemoryBank({
        records: [
          createMemoryRecord({
            key: 'favorite color',
            value: 'blue',
            addedAt: 1,
            hitCount: 2,
          }),
        ],
        maxRecords: 6,
      });

      // Act
      const exportedSnapshot = exportNeatChatSessionV2(session);
      const importedSession = importNeatChatSessionV2(exportedSnapshot);
      const importedMemoryBank = readMemoryBankLike(importedSession);

      // Assert
      expect({
        recordsLength: importedMemoryBank?.records.length,
        maxRecords: importedMemoryBank?.maxRecords,
      }).toEqual({
        recordsLength: session.memoryBank.records.length,
        maxRecords: session.memoryBank.maxRecords,
      });
    });

    it('imports legacy v2 snapshots without memoryBank as an empty default bank', () => {
      // Arrange
      const exportedSnapshot = exportNeatChatSessionV2(createNeatChatSession());
      const legacySnapshot = {
        ...structuredClone(exportedSnapshot),
        extensions: {
          ...exportedSnapshot.extensions,
          neatchat: {
            vocabularySize: exportedSnapshot.extensions.neatchat.vocabularySize,
          },
        },
      };

      // Act
      const importedSession = importNeatChatSessionV2(legacySnapshot);
      const importedMemoryBank = readMemoryBankLike(importedSession);
      const defaultMemoryBank = createNeatChatEpisodicMemoryBank();

      // Assert
      expect({
        records: importedMemoryBank?.records,
        maxRecords: importedMemoryBank?.maxRecords,
      }).toEqual({
        records: [],
        maxRecords: defaultMemoryBank.maxRecords,
      });
    });
  });

  describe('createNeatChatSession', () => {
    it('returns a session with a defined memoryBank field', () => {
      // Arrange
      const session = createNeatChatSession();
      const sessionRecord = session as unknown as Record<string, unknown>;

      // Act
      const hasMemoryBankField = typeof sessionRecord.memoryBank !== 'undefined';

      // Assert
      expect(hasMemoryBankField).toBe(true);
    });
  });
});

function createSessionWithMemoryBank(
  memoryBank: NeatChatEpisodicMemoryBankLike,
): NeatChatSessionWithMemoryBankLike {
  const session = createNeatChatSession({
    corpusRetainedTerms: MEMORY_TEST_RETAINED_TERMS,
    contextWindowTokenCount: 12,
  });

  return {
    ...session,
    memoryBank,
  };
}

function createMemoryRecord(
  overrides: Partial<NeatChatMemoryRecordLike> = {},
): NeatChatMemoryRecordLike {
  return {
    key: 'favorite color',
    value: 'blue',
    addedAt: 1,
    hitCount: 0,
    ...overrides,
  };
}

function readMemoryBankLike(
  session: ReturnType<typeof importNeatChatSessionV2>,
): NeatChatEpisodicMemoryBankLike | undefined {
  const sessionRecord = session as unknown as Record<string, unknown>;

  return sessionRecord.memoryBank as NeatChatEpisodicMemoryBankLike | undefined;
}