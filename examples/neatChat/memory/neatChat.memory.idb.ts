import {
  IDB_MEMORY_DATABASE_NAME,
  IDB_MEMORY_OBJECT_STORE_NAME,
} from './neatChat.memory.constants';
import type {
  CreateStoredMemoryEntry,
  MemoryAdapter,
  StoredMemoryEntry,
} from './neatChat.memory.types';

type IdbStoredMemoryRecord = {
  entryId?: number;
  sessionId: string;
  entryType: StoredMemoryEntry['entryType'];
  content: string;
  tokens: string[];
  score: number;
  createdAt: number;
  lastUsed: number;
};

/** Options for creating the browser-side IndexedDB adapter. */
export interface CreateIdbMemoryAdapterOptions {
  /** Optional IndexedDB database name override. */
  readonly databaseName?: string;
  /** Optional object-store name override. */
  readonly objectStoreName?: string;
  /** Optional schema version override. */
  readonly version?: number;
}

/**
 * Raw IndexedDB durable memory adapter for browser NeatChat sessions.
 *
 * The adapter intentionally stays dependency-free and uses one object store
 * with session, recency, and entry-type indexes.
 */
export class IdbMemoryAdapter implements MemoryAdapter {
  private readonly databaseName: string;
  private readonly objectStoreName: string;
  private readonly version: number;
  private readonly databasePromise: Promise<IDBDatabase>;

  public constructor(options: CreateIdbMemoryAdapterOptions = {}) {
    this.databaseName = options.databaseName ?? IDB_MEMORY_DATABASE_NAME;
    this.objectStoreName =
      options.objectStoreName ?? IDB_MEMORY_OBJECT_STORE_NAME;
    this.version = options.version ?? 1;
    this.databasePromise = this.openDatabase();
  }

  /** @inheritdoc */
  public async store(entry: CreateStoredMemoryEntry): Promise<string> {
    const database = await this.databasePromise;

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(
        this.objectStoreName,
        'readwrite',
      );
      const objectStore = transaction.objectStore(this.objectStoreName);
      const request = objectStore.add({
        sessionId: entry.sessionId,
        entryType: entry.entryType,
        content: entry.content,
        tokens: [...entry.tokens],
        score: entry.score,
        createdAt: entry.createdAt,
        lastUsed: entry.lastUsed,
      } satisfies IdbStoredMemoryRecord);

      request.onerror = () => {
        reject(request.error);
      };
      request.onsuccess = () => {
        resolve(String(request.result));
      };
    });
  }

  /** @inheritdoc */
  public async list(sessionId: string): Promise<readonly StoredMemoryEntry[]> {
    const database = await this.databasePromise;

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(
        this.objectStoreName,
        'readonly',
      );
      const objectStore = transaction.objectStore(this.objectStoreName);
      const sessionIndex = objectStore.index('sessionId');
      const request = sessionIndex.getAll(IDBKeyRange.only(sessionId));

      request.onerror = () => {
        reject(request.error);
      };
      request.onsuccess = () => {
        const storedEntries = (
          request.result as readonly IdbStoredMemoryRecord[]
        )
          .map(mapIdbRecordToStoredMemoryEntry)
          .toSorted(
            (leftEntry, rightEntry) =>
              leftEntry.createdAt - rightEntry.createdAt,
          );

        resolve(storedEntries);
      };
    });
  }

  /** @inheritdoc */
  public async remove(entryIds: readonly string[]): Promise<number> {
    if (entryIds.length === 0) {
      return 0;
    }

    const database = await this.databasePromise;

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(
        this.objectStoreName,
        'readwrite',
      );
      const objectStore = transaction.objectStore(this.objectStoreName);
      let removedEntryCount = 0;
      let pendingDeleteCount = entryIds.length;

      transaction.onerror = () => {
        reject(transaction.error);
      };

      for (const entryId of entryIds) {
        const deleteRequest = objectStore.delete(resolveIdbKey(entryId));

        deleteRequest.onerror = () => {
          reject(deleteRequest.error);
        };
        deleteRequest.onsuccess = () => {
          removedEntryCount += 1;
          pendingDeleteCount -= 1;

          if (pendingDeleteCount === 0) {
            resolve(removedEntryCount);
          }
        };
      }
    });
  }

  /** @inheritdoc */
  public async close(): Promise<void> {
    const database = await this.databasePromise;

    database.close();
  }

  private openDatabase(): Promise<IDBDatabase> {
    if (!('indexedDB' in globalThis)) {
      return Promise.reject(
        new Error('IndexedDB is not available in this runtime.'),
      );
    }

    return new Promise((resolve, reject) => {
      const openRequest = globalThis.indexedDB.open(
        this.databaseName,
        this.version,
      );

      openRequest.onerror = () => {
        reject(openRequest.error);
      };
      openRequest.onupgradeneeded = () => {
        const database = openRequest.result;
        const objectStore = database.objectStoreNames.contains(
          this.objectStoreName,
        )
          ? openRequest.transaction?.objectStore(this.objectStoreName)
          : database.createObjectStore(this.objectStoreName, {
              keyPath: 'entryId',
              autoIncrement: true,
            });

        if (!objectStore) {
          return;
        }

        if (!objectStore.indexNames.contains('sessionId')) {
          objectStore.createIndex('sessionId', 'sessionId', { unique: false });
        }

        if (!objectStore.indexNames.contains('lastUsed')) {
          objectStore.createIndex('lastUsed', 'lastUsed', { unique: false });
        }

        if (!objectStore.indexNames.contains('entryType')) {
          objectStore.createIndex('entryType', 'entryType', { unique: false });
        }
      };
      openRequest.onsuccess = () => {
        resolve(openRequest.result);
      };
    });
  }
}

function mapIdbRecordToStoredMemoryEntry(
  record: IdbStoredMemoryRecord,
): StoredMemoryEntry {
  return {
    entryId: String(record.entryId ?? ''),
    sessionId: record.sessionId,
    entryType: record.entryType,
    content: record.content,
    tokens: record.tokens,
    score: record.score,
    createdAt: record.createdAt,
    lastUsed: record.lastUsed,
  };
}

function resolveIdbKey(entryId: string): IDBValidKey {
  const numericEntryId = Number(entryId);

  return Number.isNaN(numericEntryId) ? entryId : numericEntryId;
}
