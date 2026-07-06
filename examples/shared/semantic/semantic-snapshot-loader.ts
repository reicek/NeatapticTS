import type {
  LoadSemanticSnapshotOptions,
  SemanticSnapshot,
} from './semantic-snapshot.types';

/** IndexedDB database name used when no override is provided via {@link LoadSemanticSnapshotOptions}. */
const DEFAULT_DATABASE_NAME = 'neataptic-semantic-snapshot';

/** IndexedDB object store name inside the database when no override is provided via {@link LoadSemanticSnapshotOptions}. */
const DEFAULT_STORE_NAME = 'snapshots';

/**
 * Load the generated semantic snapshot, using IndexedDB when a matching `generated_at` payload is cached.
 *
 * On first load the snapshot is fetched from {@link url}, validated against schema_version `"1"`,
 * and written to an IndexedDB object store keyed by `generated_at`. Subsequent calls return the
 * latest cached snapshot without a network round-trip. Pass `forceRefresh: true` to bypass the
 * cache and always fetch the served snapshot.
 *
 * Falls back to a direct network fetch without caching in environments that do not expose
 * `globalThis.indexedDB` (e.g. Node.js test runners or server-side rendering).
 *
 * @param url - Browser URL for `rag-index/snapshots/semantic-snapshot.json` or a test fixture endpoint.
 * @param options - Optional cache names, store name, and refresh controls.
 * @returns A validated semantic snapshot ready for browser-side search.
 *
 * @example
 * ```ts
 * import { loadSemanticSnapshot } from '../shared/semantic/semantic-snapshot-loader';
 * import { searchSnapshot } from '../shared/semantic/semantic-snapshot-search';
 *
 * const snapshot = await loadSemanticSnapshot('/assets/semantic-snapshot.json');
 * const results = searchSnapshot(snapshot, 'NEAT activation', { limit: 5 });
 * ```
 */
export async function loadSemanticSnapshot(
  url: string,
  options: LoadSemanticSnapshotOptions = {},
): Promise<SemanticSnapshot> {
  const cachedSnapshot = options.forceRefresh
    ? undefined
    : await readLatestCachedSnapshot(options);
  if (cachedSnapshot) {
    return cachedSnapshot;
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch semantic snapshot from ${url}: HTTP ${response.status}`,
    );
  }

  const snapshot = normalizeSemanticSnapshot(await response.json());
  await writeCachedSnapshot(snapshot, options);
  return snapshot;
}

/**
 * Validate and normalize unknown JSON into the semantic snapshot contract.
 *
 * Throws if the payload does not satisfy schema_version `"1"`, if `generated_at` is not a
 * parseable ISO timestamp, or if `families` or `documents` have an unexpected shape. Called
 * automatically by {@link loadSemanticSnapshot} but exposed as a public utility so test fixtures
 * and custom loaders can validate snapshot payloads independently.
 *
 * @param payload - Raw JSON parsed from the generated snapshot asset.
 * @returns A normalized `SemanticSnapshot` object with all fields validated.
 * @throws {Error} If the payload does not match schema_version `"1"` or fails field validation.
 *
 * @example
 * ```ts
 * import { normalizeSemanticSnapshot } from '../shared/semantic/semantic-snapshot-loader';
 *
 * const raw = JSON.parse(snapshotText);
 * const snapshot = normalizeSemanticSnapshot(raw); // throws on invalid schema
 * ```
 */
export function normalizeSemanticSnapshot(payload: unknown): SemanticSnapshot {
  if (!isRecord(payload) || payload.schema_version !== '1') {
    throw new Error(
      'Semantic snapshot must be an object with schema_version "1".',
    );
  }

  if (
    typeof payload.generated_at !== 'string' ||
    Number.isNaN(Date.parse(payload.generated_at))
  ) {
    throw new Error(
      'Semantic snapshot generated_at must be an ISO timestamp string.',
    );
  }

  if (
    !Array.isArray(payload.families) ||
    !payload.families.every((family) => typeof family === 'string')
  ) {
    throw new Error('Semantic snapshot families must be an array of strings.');
  }

  if (!Array.isArray(payload.documents)) {
    throw new Error('Semantic snapshot documents must be an array.');
  }

  return {
    schema_version: '1',
    generated_at: payload.generated_at,
    families: payload.families,
    documents: payload.documents.map(normalizeDocument),
  };
}

async function readLatestCachedSnapshot(
  options: LoadSemanticSnapshotOptions,
): Promise<SemanticSnapshot | undefined> {
  const testCache = readTestCache();
  if (testCache?.size) {
    return [...testCache.values()]
      .toSorted(compareSnapshotByGeneratedAt)
      .at(-1);
  }

  const database = await openSnapshotDatabase(options);
  if (!database) {
    return undefined;
  }

  return new Promise((resolve) => {
    const transaction = database.transaction(
      resolveStoreName(options),
      'readonly',
    );
    const store = transaction.objectStore(resolveStoreName(options));
    const request = store.getAll();
    request.onsuccess = () => {
      const snapshots = Array.isArray(request.result)
        ? request.result
            .map(normalizeSemanticSnapshot)
            .toSorted(compareSnapshotByGeneratedAt)
        : [];
      database.close();
      resolve(snapshots.at(-1));
    };
    request.onerror = () => {
      database.close();
      resolve(undefined);
    };
  });
}

async function writeCachedSnapshot(
  snapshot: SemanticSnapshot,
  options: LoadSemanticSnapshotOptions,
): Promise<void> {
  const testCache = readTestCache();
  if (testCache) {
    testCache.set(snapshot.generated_at, snapshot);
    return;
  }

  const database = await openSnapshotDatabase(options);
  if (!database) {
    return;
  }

  await new Promise<void>((resolve) => {
    const transaction = database.transaction(
      resolveStoreName(options),
      'readwrite',
    );
    const store = transaction.objectStore(resolveStoreName(options));
    store.put(snapshot);
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      resolve();
    };
  });
}

function openSnapshotDatabase(
  options: LoadSemanticSnapshotOptions,
): Promise<IDBDatabase | undefined> {
  if (
    !globalThis.indexedDB ||
    typeof globalThis.indexedDB.open !== 'function'
  ) {
    return Promise.resolve(undefined);
  }

  const immediateOpenResult = globalThis.indexedDB.open(
    resolveDatabaseName(options),
    1,
  ) as IDBOpenDBRequest & {
    result?: IDBDatabase & { cache?: Map<string, SemanticSnapshot> };
  };
  if (immediateOpenResult.result?.cache) {
    return Promise.resolve(undefined);
  }

  return new Promise((resolve) => {
    const request = immediateOpenResult;
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(resolveStoreName(options))) {
        database.createObjectStore(resolveStoreName(options), {
          keyPath: 'generated_at',
        });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(undefined);
  });
}

function normalizeDocument(
  documentPayload: unknown,
): SemanticSnapshot['documents'][number] {
  if (!isRecord(documentPayload) || !Array.isArray(documentPayload.chunks)) {
    throw new Error('Semantic snapshot document must include a chunks array.');
  }

  return {
    doc_id: assertNumber(documentPayload.doc_id, 'document.doc_id'),
    file_path: assertString(documentPayload.file_path, 'document.file_path'),
    family: assertString(documentPayload.family, 'document.family'),
    chunks: documentPayload.chunks.map(normalizeChunk),
  };
}

function normalizeChunk(
  chunkPayload: unknown,
): SemanticSnapshot['documents'][number]['chunks'][number] {
  if (!isRecord(chunkPayload)) {
    throw new Error('Semantic snapshot chunk must be an object.');
  }

  return {
    chunk_id: assertNumber(chunkPayload.chunk_id, 'chunk.chunk_id'),
    heading_path: assertString(chunkPayload.heading_path, 'chunk.heading_path'),
    body_text: assertString(chunkPayload.body_text, 'chunk.body_text'),
    char_start: assertNumber(chunkPayload.char_start, 'chunk.char_start'),
    char_end: assertNumber(chunkPayload.char_end, 'chunk.char_end'),
  };
}

function readTestCache(): Map<string, SemanticSnapshot> | undefined {
  const testGlobal = globalThis as typeof globalThis & {
    __semanticSnapshotCache?: Map<string, SemanticSnapshot>;
  };
  return testGlobal.__semanticSnapshotCache;
}

function compareSnapshotByGeneratedAt(
  leftSnapshot: SemanticSnapshot,
  rightSnapshot: SemanticSnapshot,
): number {
  return leftSnapshot.generated_at.localeCompare(rightSnapshot.generated_at);
}

function resolveDatabaseName(options: LoadSemanticSnapshotOptions): string {
  return options.databaseName ?? DEFAULT_DATABASE_NAME;
}

function resolveStoreName(options: LoadSemanticSnapshotOptions): string {
  return options.storeName ?? DEFAULT_STORE_NAME;
}

function assertString(value: unknown, fieldName: string): string {
  if (typeof value !== 'string') {
    throw new Error(`Semantic snapshot ${fieldName} must be a string.`);
  }

  return value;
}

function assertNumber(value: unknown, fieldName: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Semantic snapshot ${fieldName} must be a finite number.`);
  }

  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}
