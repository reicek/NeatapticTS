import { MAX_DURABLE_ENTRIES } from './neatChat.memory.constants';
import {
  pruneMemory,
  retrieveMemoryContext,
  storeExchangeMemory,
} from './neatChat.memory.services';

type StoredMemoryEntryLike = {
  readonly entryId: string;
  readonly sessionId: string;
  readonly content: string;
  readonly tokens: readonly string[];
  readonly createdAt: number;
  readonly lastUsed: number;
};

type InMemoryAdapterLike = {
  readonly entries: readonly StoredMemoryEntryLike[];
  store(entry: Omit<StoredMemoryEntryLike, 'entryId'>): Promise<string>;
  list(sessionId: string): Promise<readonly StoredMemoryEntryLike[]>;
  remove(entryIds: readonly string[]): Promise<number>;
  snapshotEntryIds(): readonly string[];
};

describe('neatChat durable memory services', () => {
  describe('storeExchangeMemory', () => {
    it('stores an exchange entry and returns a truthy entry identifier', async () => {
      // Arrange
      const adapter = createInMemoryAdapter();

      // Act
      const entryId = await storeExchangeMemory({
        adapter,
        sessionId: 'session-1',
        userMessage: 'favorite color',
        response: 'blue ocean',
        now: 10,
      });

      // Assert
      expect(entryId).toBeTruthy();
    });
  });

  describe('retrieveMemoryContext', () => {
    it('returns entries ranked by descending BM25 and token overlap relevance', async () => {
      // Arrange
      const adapter = createInMemoryAdapter([
        createStoredMemoryEntry({
          entryId: 'entry-best',
          sessionId: 'session-1',
          content: 'favorite color blue ocean',
          tokens: ['favorite', 'color', 'blue', 'ocean'],
          createdAt: 1,
          lastUsed: 1,
        }),
        createStoredMemoryEntry({
          entryId: 'entry-middle',
          sessionId: 'session-1',
          content: 'favorite color weekend',
          tokens: ['favorite', 'color', 'weekend'],
          createdAt: 2,
          lastUsed: 2,
        }),
        createStoredMemoryEntry({
          entryId: 'entry-low',
          sessionId: 'session-1',
          content: 'pizza dinner plans',
          tokens: ['pizza', 'dinner', 'plans'],
          createdAt: 3,
          lastUsed: 3,
        }),
      ]);

      // Act
      const memoryResults = await retrieveMemoryContext({
        adapter,
        sessionId: 'session-1',
        query: 'favorite color blue',
        maxResults: 3,
      });

      // Assert
      expect(
        memoryResults.map(
          (memoryEntry: { readonly entryId: string }) => memoryEntry.entryId,
        ),
      ).toEqual(['entry-best', 'entry-middle', 'entry-low']);
    });

    it('returns an empty array without throwing when the memory store is empty', async () => {
      // Arrange
      const adapter = createInMemoryAdapter();

      // Act
      const memoryResults = await retrieveMemoryContext({
        adapter,
        sessionId: 'session-1',
        query: 'favorite color blue',
        maxResults: 3,
      });

      // Assert
      expect(memoryResults).toEqual([]);
    });
  });

  describe('pruneMemory', () => {
    it('removes the oldest entries first when the durable store exceeds the cap', async () => {
      // Arrange
      const adapter = createInMemoryAdapter(
        Array.from({ length: MAX_DURABLE_ENTRIES + 2 }, (_, index) =>
          createStoredMemoryEntry({
            entryId: `entry-${index + 1}`,
            sessionId: 'session-1',
            content: `memory ${index + 1}`,
            tokens: ['memory', String(index + 1)],
            createdAt: index + 1,
            lastUsed: index + 1,
          }),
        ),
      );

      // Act
      await pruneMemory({
        adapter,
        sessionId: 'session-1',
        maxEntries: MAX_DURABLE_ENTRIES,
      });

      // Assert
      expect(adapter.snapshotEntryIds().slice(0, 3)).toEqual([
        'entry-3',
        'entry-4',
        'entry-5',
      ]);
    });
  });
});

function createInMemoryAdapter(
  seedEntries: readonly StoredMemoryEntryLike[] = [],
): InMemoryAdapterLike {
  let nextEntryNumber = seedEntries.length + 1;
  let mutableEntries = [...seedEntries];

  return {
    get entries() {
      return mutableEntries;
    },
    async store(entry) {
      const entryId = `generated-${nextEntryNumber}`;
      nextEntryNumber += 1;
      mutableEntries = [...mutableEntries, { entryId, ...entry }];
      return entryId;
    },
    async list(sessionId) {
      return mutableEntries.filter(
        (memoryEntry) => memoryEntry.sessionId === sessionId,
      );
    },
    async remove(entryIds) {
      const entryIdSet = new Set(entryIds);
      const originalLength = mutableEntries.length;
      mutableEntries = mutableEntries.filter(
        (memoryEntry) => !entryIdSet.has(memoryEntry.entryId),
      );
      return originalLength - mutableEntries.length;
    },
    snapshotEntryIds() {
      return mutableEntries.map((memoryEntry) => memoryEntry.entryId);
    },
  };
}

function createStoredMemoryEntry(
  overrides: StoredMemoryEntryLike,
): StoredMemoryEntryLike {
  return overrides;
}
