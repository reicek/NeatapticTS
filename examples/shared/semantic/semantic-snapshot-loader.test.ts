interface SnapshotChunk {
  chunk_id: number;
  heading_path: string;
  body_text: string;
  char_start: number;
  char_end: number;
}

interface SnapshotDocument {
  doc_id: number;
  file_path: string;
  family: string;
  chunks: SnapshotChunk[];
}

interface SemanticSnapshot {
  generated_at: string;
  schema_version: '1';
  families: string[];
  documents: SnapshotDocument[];
}

interface LoadSemanticSnapshotModule {
  loadSemanticSnapshot: (url: string) => Promise<SemanticSnapshot>;
}

describe('semantic browser snapshot', () => {
  describe('semantic-snapshot-loader.ts', () => {
    it('fetches a valid snapshot and stores it in IndexedDB by generated_at', async () => {
      const snapshot = createFixtureSnapshot('2026-05-23T00:00:00.000Z');
      const fetchCalls: string[] = [];
      installFetchMock(snapshot, fetchCalls);
      installIndexedDbMock();

      const { loadSemanticSnapshot } = await importLoaderModule();
      const loadedSnapshot = await loadSemanticSnapshot(
        '/assets/semantic-snapshot.json',
      );

      expect({
        loadedSnapshot,
        cachedSnapshot: readCachedSnapshot(snapshot.generated_at),
        fetchCalls,
      }).toEqual({
        loadedSnapshot: snapshot,
        cachedSnapshot: snapshot,
        fetchCalls: ['/assets/semantic-snapshot.json'],
      });
    });

    it('returns the cached snapshot without re-fetching when generated_at is already cached', async () => {
      const snapshot = createFixtureSnapshot('2026-05-23T00:00:00.000Z');
      const fetchCalls: string[] = [];
      installFetchMock(snapshot, fetchCalls);
      installIndexedDbMock(new Map([[snapshot.generated_at, snapshot]]));

      const { loadSemanticSnapshot } = await importLoaderModule();
      const firstSnapshot = await loadSemanticSnapshot(
        '/assets/semantic-snapshot.json',
      );
      const secondSnapshot = await loadSemanticSnapshot(
        '/assets/semantic-snapshot.json',
      );

      expect({ firstSnapshot, secondSnapshot, fetchCalls }).toEqual({
        firstSnapshot: snapshot,
        secondSnapshot: snapshot,
        fetchCalls: [],
      });
    });
  });
});

async function importLoaderModule(): Promise<LoadSemanticSnapshotModule> {
  const modulePath = './semantic-snapshot-loader';
  return import(modulePath) as Promise<LoadSemanticSnapshotModule>;
}

function createFixtureSnapshot(generatedAt: string): SemanticSnapshot {
  return {
    generated_at: generatedAt,
    schema_version: '1',
    families: ['readme'],
    documents: [
      {
        doc_id: 1,
        file_path: 'README.md',
        family: 'readme',
        chunks: [
          {
            chunk_id: 1,
            heading_path: '# NEAT',
            body_text: 'NEAT snapshot loader fixture.',
            char_start: 0,
            char_end: 29,
          },
        ],
      },
    ],
  };
}

function installFetchMock(snapshot: SemanticSnapshot, calls: string[]): void {
  const fetchMock = async (url: RequestInfo | URL): Promise<Response> => {
    calls.push(String(url));
    return new Response(JSON.stringify(snapshot), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  Object.defineProperty(globalThis, 'fetch', {
    value: fetchMock,
    configurable: true,
  });
}

function installIndexedDbMock(
  seed = new Map<string, SemanticSnapshot>(),
): void {
  Object.defineProperty(globalThis, '__semanticSnapshotCache', {
    value: seed,
    configurable: true,
  });
  Object.defineProperty(globalThis, 'indexedDB', {
    value: {
      open: () => ({ result: { cache: seed } }),
    },
    configurable: true,
  });
}

function readCachedSnapshot(generatedAt: string): SemanticSnapshot | undefined {
  const globalWithCache = globalThis as typeof globalThis & {
    __semanticSnapshotCache: Map<string, SemanticSnapshot>;
  };

  return globalWithCache.__semanticSnapshotCache.get(generatedAt);
}

export {};
