/**
 * @module build-entity-graph.test
 * @description 100% coverage tests for build-entity-graph.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'build-entity-graph.mjs');

// Shared mock functions
const mockFg = jest.fn();
const mockReadFile = jest.fn();
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockToRepoRelative = jest.fn();
const mockWriteJsonOrText = jest.fn();
const mockGetFreshnessProof = jest.fn();
const mockIsFreshDocument = jest.fn();
const mockInitSemanticIndex = jest.fn();
const mockExtractCodeEntities = jest.fn();
const mockDeriveModulePath = jest.fn();
const mockExtractDocEntities = jest.fn();
const mockMapFamilyToEntityType = jest.fn();
const mockExtractCrossRefs = jest.fn();

jest.unstable_mockModule('fast-glob', () => ({ default: mockFg }));
jest.unstable_mockModule('node:fs/promises', () => ({ readFile: mockReadFile }));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  toRepoRelative: mockToRepoRelative,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./freshness.mjs', () => ({
  getFreshnessProof: mockGetFreshnessProof,
  isFreshDocument: mockIsFreshDocument,
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: '/fake/default-db.sqlite',
  initSemanticIndex: mockInitSemanticIndex,
  repoRoot: '/fake/repo',
}));
jest.unstable_mockModule('./extract-code-entities.mjs', () => ({
  extractCodeEntities: mockExtractCodeEntities,
  deriveModulePath: mockDeriveModulePath,
}));
jest.unstable_mockModule('./extract-doc-entities.mjs', () => ({
  extractDocEntities: mockExtractDocEntities,
  mapFamilyToEntityType: mockMapFamilyToEntityType,
}));
jest.unstable_mockModule('./extract-cross-refs.mjs', () => ({
  extractCrossRefs: mockExtractCrossRefs,
}));

const { buildEntityGraph } = await import('./build-entity-graph.mjs');

function makeMockClient({ docRows = [], chunkRows = [], entityRows = [] } = {}) {
  return {
    execute: jest.fn().mockImplementation((params) => {
      const sql = typeof params === 'string' ? params : params.sql;
      if (sql.includes('DELETE FROM edges')) return Promise.resolve({});
      if (sql.includes('DELETE FROM entities')) return Promise.resolve({});
      if (sql.includes('SELECT doc_id, file_path FROM documents'))
        return Promise.resolve({ rows: docRows });
      if (sql.includes('SELECT doc_id, chunk_id, symbol_name, heading_path'))
        return Promise.resolve({ rows: chunkRows });
      if (sql.includes('SELECT entity_id, qualified_name FROM entities'))
        return Promise.resolve({ rows: entityRows });
      return Promise.resolve({ rows: [] });
    }),
    batch: jest.fn().mockResolvedValue([]),
    close: jest.fn().mockResolvedValue(undefined),
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  mockToRepoRelative.mockImplementation((p) => p);
  mockFg.mockResolvedValue([]);
  mockReadFile.mockResolvedValue('content');
  mockExtractCodeEntities.mockResolvedValue({
    entities: [],
    edges: [],
    symbolEntityMap: new Map(),
    moduleEntityMap: new Map(),
  });
  mockExtractDocEntities.mockResolvedValue({
    entities: [],
    edges: [],
    entityMap: new Map(),
  });
  mockExtractCrossRefs.mockReturnValue({ edges: [] });
  mockInitSemanticIndex.mockResolvedValue(makeMockClient());
});

// ---------------------------------------------------------------------------
// buildEntityGraph — dry-run mode
// ---------------------------------------------------------------------------

describe('build-entity-graph: dry-run mode', () => {
  it('returns summary with entity and edge counts', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts', module_path: 'src/test',
    };
    const codeEdge = {
      source_qualified_name: 'foo', target_qualified_name: 'bar',
      relationship: 'calls',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [codeEdge],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const docEntity = {
      entity_type: 'plan', name: 'myplan', qualified_name: 'plans/myplan',
      file_path: 'plans/myplan.md',
    };
    mockExtractDocEntities.mockResolvedValue({
      entities: [docEntity],
      edges: [],
      entityMap: new Map(),
    });
    const result = await buildEntityGraph({ dryRun: true });
    expect(result.dryRun).toBe(true);
    expect(result.entities).toBe(2);
    expect(result.edges).toBe(1);
    expect(result.codeEntityCount).toBe(1);
    expect(result.docEntityCount).toBe(1);
    expect(result.codeEdgeCount).toBe(1);
    expect(result.crossRefEdgeCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// buildEntityGraph — non-dry-run mode
// ---------------------------------------------------------------------------

describe('build-entity-graph: non-dry-run mode', () => {
  it('creates client via initSemanticIndex when not provided', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      docRows: [{ doc_id: 1, file_path: 'src/test.ts' }],
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({});
    expect(result.entities).toBe(1);
    expect(result.edges).toBe(0);
    expect(mockInitSemanticIndex).toHaveBeenCalledWith({
      databasePath: expect.any(String),
    });
  });

  it('uses provided client and does not create new one', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      docRows: [{ doc_id: 1, file_path: 'src/test.ts' }],
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({ client });
    expect(result.entities).toBe(1);
    expect(mockInitSemanticIndex).toHaveBeenCalledWith({ client });
  });

  it('deletes existing edges and entities before insert', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    await buildEntityGraph({ client });
    const deleteCalls = client.execute.mock.calls.filter(
      (c) => {
        const sql = typeof c[0] === 'string' ? c[0] : c[0].sql;
        return sql.includes('DELETE FROM');
      },
    );
    expect(deleteCalls.length).toBe(2);
  });

  it('resolves doc_id and chunk_id for entities', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      docRows: [{ doc_id: 1, file_path: 'src/test.ts' }],
      chunkRows: [{ doc_id: 1, chunk_id: 100, symbol_name: 'foo', heading_path: null }],
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({ client });
    expect(result.entities).toBe(1);
    // Check entity batch includes the chunkId
    const entityBatchCall = client.batch.mock.calls.find(
      (c) => {
        const sql = c[0][0].sql;
        return sql.includes('INSERT INTO entities');
      },
    );
    expect(entityBatchCall).toBeDefined();
    expect(entityBatchCall[0][0].args[4]).toBe(100); // chunkId
  });

  it('sets chunkId to null for module entities', async () => {
    const moduleEntity = {
      entity_type: 'module', name: 'test', qualified_name: 'src/test',
      file_path: 'src/test.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [moduleEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      docRows: [{ doc_id: 1, file_path: 'src/test.ts' }],
      chunkRows: [{ doc_id: 1, chunk_id: 100, symbol_name: 'test', heading_path: null }],
      entityRows: [{ entity_id: 10, qualified_name: 'src/test' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    await buildEntityGraph({ client });
    const entityBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT INTO entities'),
    );
    expect(entityBatchCall[0][0].args[4]).toBe(null); // chunkId null for module
  });

  it('sets docId and chunkId to null when not in maps', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'unknown.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      docRows: [], // no documents
      chunkRows: [],
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    await buildEntityGraph({ client });
    const entityBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT INTO entities'),
    );
    expect(entityBatchCall[0][0].args[3]).toBe(null); // docId null
    expect(entityBatchCall[0][0].args[4]).toBe(null); // chunkId null
  });

  it('uses default values for optional entity fields', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts',
      // No module_path, signature_text, char_start, char_end, extra_metadata
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    await buildEntityGraph({ client });
    const entityBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT INTO entities'),
    );
    const args = entityBatchCall[0][0].args;
    expect(args[5]).toBe(null); // module_path
    expect(args[6]).toBe(null); // signature_text
    expect(args[8]).toBe(null); // char_start
    expect(args[9]).toBe(null); // char_end
    expect(args[10]).toBe('{}'); // extra_metadata default
  });

  it('uses provided values for optional entity fields', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts',
      module_path: 'src/test',
      signature_text: 'function foo(): void',
      char_start: 10,
      char_end: 50,
      extra_metadata: '{"custom":true}',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    await buildEntityGraph({ client });
    const entityBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT INTO entities'),
    );
    const args = entityBatchCall[0][0].args;
    expect(args[5]).toBe('src/test');
    expect(args[6]).toBe('function foo(): void');
    expect(args[8]).toBe(10);
    expect(args[9]).toBe(50);
    expect(args[10]).toBe('{"custom":true}');
  });
});

// ---------------------------------------------------------------------------
// Edge resolution and insertion
// ---------------------------------------------------------------------------

describe('build-entity-graph: edge resolution', () => {
  it('resolves and inserts edges with default confidence and metadata', async () => {
    const codeEdge = {
      source_qualified_name: 'foo', target_qualified_name: 'bar',
      relationship: 'calls',
      // No confidence or extra_metadata → defaults
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [],
      edges: [codeEdge],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [
        { entity_id: 1, qualified_name: 'foo' },
        { entity_id: 2, qualified_name: 'bar' },
      ],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({ client });
    expect(result.edges).toBe(1);
    const edgeBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT OR IGNORE INTO edges'),
    );
    expect(edgeBatchCall).toBeDefined();
    const args = edgeBatchCall[0][0].args;
    expect(args[0]).toBe(1); // sourceId
    expect(args[1]).toBe(2); // targetId
    expect(args[2]).toBe('calls'); // relationship
    expect(args[3]).toBe('high'); // confidence default
    expect(args[4]).toBe('{}'); // extra_metadata default
  });

  it('uses provided confidence and extra_metadata for edges', async () => {
    const codeEdge = {
      source_qualified_name: 'foo', target_qualified_name: 'bar',
      relationship: 'imports', confidence: 'medium',
      extra_metadata: '{"source":"test"}',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [],
      edges: [codeEdge],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [
        { entity_id: 1, qualified_name: 'foo' },
        { entity_id: 2, qualified_name: 'bar' },
      ],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    await buildEntityGraph({ client });
    const edgeBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT OR IGNORE INTO edges'),
    );
    const args = edgeBatchCall[0][0].args;
    expect(args[3]).toBe('medium');
    expect(args[4]).toBe('{"source":"test"}');
  });

  it('skips edges with unresolved source or target', async () => {
    const edges = [
      { source_qualified_name: 'foo', target_qualified_name: 'unknown', relationship: 'calls' },
      { source_qualified_name: 'unknown', target_qualified_name: 'bar', relationship: 'calls' },
      { source_qualified_name: 'unknown1', target_qualified_name: 'unknown2', relationship: 'calls' },
    ];
    mockExtractCodeEntities.mockResolvedValue({
      entities: [],
      edges,
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [
        { entity_id: 1, qualified_name: 'foo' },
        { entity_id: 2, qualified_name: 'bar' },
      ],
    });

    const result = await buildEntityGraph({ client });
    expect(result.edges).toBe(0);
    // No edge batch should be called (0 resolved edges)
    const edgeBatchCalls = client.batch.mock.calls.filter(
      (c) => c[0][0].sql.includes('INSERT OR IGNORE INTO edges'),
    );
    expect(edgeBatchCalls.length).toBe(0);
  });

  it('combines code, doc, and cross-ref edges', async () => {
    const codeEdge = {
      source_qualified_name: 'foo', target_qualified_name: 'bar',
      relationship: 'calls',
    };
    const docEdge = {
      source_qualified_name: 'foo', target_qualified_name: 'bar',
      relationship: 'documents',
    };
    const crossRefEdge = {
      source_qualified_name: 'foo', target_qualified_name: 'bar',
      relationship: 'references',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [],
      edges: [codeEdge],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    mockExtractDocEntities.mockResolvedValue({
      entities: [],
      edges: [docEdge],
      entityMap: new Map(),
    });
    mockExtractCrossRefs.mockReturnValue({ edges: [crossRefEdge] });
    const client = makeMockClient({
      entityRows: [
        { entity_id: 1, qualified_name: 'foo' },
        { entity_id: 2, qualified_name: 'bar' },
      ],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({ client });
    expect(result.edges).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// collectAllEntities — dedup by qualified_name
// ---------------------------------------------------------------------------

describe('build-entity-graph: collectAllEntities dedup', () => {
  it('deduplicates entities by qualified_name', async () => {
    const codeEntity = {
      entity_type: 'function', name: 'foo', qualified_name: 'shared',
      file_path: 'src/test.ts',
    };
    const docEntity = {
      entity_type: 'plan', name: 'shared', qualified_name: 'shared',
      file_path: 'plans/shared.md',
    };
    const uniqueEntity = {
      entity_type: 'function', name: 'bar', qualified_name: 'bar',
      file_path: 'src/bar.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [codeEntity, uniqueEntity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    mockExtractDocEntities.mockResolvedValue({
      entities: [docEntity],
      edges: [],
      entityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [
        { entity_id: 1, qualified_name: 'shared' },
        { entity_id: 2, qualified_name: 'bar' },
      ],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({ client });
    expect(result.entities).toBe(2);
    // Entity batch should have 2 entities (deduped from 3)
    const entityBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT INTO entities'),
    );
    expect(entityBatchCall[0].length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// loadChunkIdMap — branch coverage
// ---------------------------------------------------------------------------

describe('build-entity-graph: loadChunkIdMap branches', () => {
  it('handles symbol_name, heading_path, both, duplicates, and nulls', async () => {
    const entity = {
      entity_type: 'function', name: 'foo', qualified_name: 'foo',
      file_path: 'src/test.ts',
    };
    mockExtractCodeEntities.mockResolvedValue({
      entities: [entity],
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      docRows: [{ doc_id: 1, file_path: 'src/test.ts' }],
      chunkRows: [
        // symbol_name set, heading_path null → bySymbol
        { doc_id: 1, chunk_id: 100, symbol_name: 'foo', heading_path: null },
        // duplicate symbol_name → skip (!map.has is false)
        { doc_id: 1, chunk_id: 101, symbol_name: 'foo', heading_path: null },
        // heading_path set, symbol_name null → byHeading
        { doc_id: 1, chunk_id: 200, symbol_name: null, heading_path: 'bar' },
        // duplicate heading_path → skip
        { doc_id: 1, chunk_id: 201, symbol_name: null, heading_path: 'bar' },
        // both set → both added
        { doc_id: 1, chunk_id: 300, symbol_name: 'baz', heading_path: 'qux' },
        // both null → neither added
        { doc_id: 1, chunk_id: 400, symbol_name: null, heading_path: null },
      ],
      entityRows: [{ entity_id: 10, qualified_name: 'foo' }],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    await buildEntityGraph({ client });
    // Entity 'foo' should get chunkId 100 (first match)
    const entityBatchCall = client.batch.mock.calls.find(
      (c) => c[0][0].sql.includes('INSERT INTO entities'),
    );
    expect(entityBatchCall[0][0].args[4]).toBe(100);
  });
});

// ---------------------------------------------------------------------------
// Entity batch flush (>250 entities)
// ---------------------------------------------------------------------------

describe('build-entity-graph: entity batch flush', () => {
  it('flushes entity batch at ENTITY_INSERT_BATCH_SIZE=250', async () => {
    const entities = [];
    const entityRows = [];
    for (let i = 0; i < 260; i++) {
      entities.push({
        entity_type: 'function', name: `fn${i}`, qualified_name: `qn${i}`,
        file_path: `src/file${i}.ts`,
      });
      entityRows.push({ entity_id: i + 1, qualified_name: `qn${i}` });
    }
    mockExtractCodeEntities.mockResolvedValue({
      entities,
      edges: [],
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({ entityRows });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({ client });
    expect(result.entities).toBe(260);
    // Should have 2 entity batch calls: one of 250, one of 10
    const entityBatchCalls = client.batch.mock.calls.filter(
      (c) => c[0][0].sql.includes('INSERT INTO entities'),
    );
    expect(entityBatchCalls.length).toBe(2);
    expect(entityBatchCalls[0][0].length).toBe(250);
    expect(entityBatchCalls[1][0].length).toBe(10);
  });
});

// ---------------------------------------------------------------------------
// Edge batch flush (>500 edges)
// ---------------------------------------------------------------------------

describe('build-entity-graph: edge batch flush', () => {
  it('flushes edge batch at EDGE_INSERT_BATCH_SIZE=500', async () => {
    const edges = [];
    for (let i = 0; i < 510; i++) {
      edges.push({
        source_qualified_name: 'foo', target_qualified_name: 'bar',
        relationship: 'calls',
      });
    }
    mockExtractCodeEntities.mockResolvedValue({
      entities: [
        { entity_type: 'function', name: 'foo', qualified_name: 'foo', file_path: 'a.ts' },
        { entity_type: 'function', name: 'bar', qualified_name: 'bar', file_path: 'b.ts' },
      ],
      edges,
      symbolEntityMap: new Map(),
      moduleEntityMap: new Map(),
    });
    const client = makeMockClient({
      entityRows: [
        { entity_id: 1, qualified_name: 'foo' },
        { entity_id: 2, qualified_name: 'bar' },
      ],
    });
    mockInitSemanticIndex.mockResolvedValue(client);

    const result = await buildEntityGraph({ client });
    expect(result.edges).toBe(510);
    const edgeBatchCalls = client.batch.mock.calls.filter(
      (c) => c[0][0].sql.includes('INSERT OR IGNORE INTO edges'),
    );
    expect(edgeBatchCalls.length).toBe(2);
    expect(edgeBatchCalls[0][0].length).toBe(500);
    expect(edgeBatchCalls[1][0].length).toBe(10);
  });
});

// ---------------------------------------------------------------------------
// getParentQualifiedName — all branches (via buildHeadingTextMap)
// ---------------------------------------------------------------------------

describe('build-entity-graph: getParentQualifiedName branches', () => {
  it('covers no-slash, no-dot, slug, uppercase, and no-hyphen branches', async () => {
    const docEntities = [
      // 1. No slash → returns as-is
      { entity_type: 'plan', name: 'simple', qualified_name: 'simple', file_path: 'f1.md', char_start: 0, char_end: 3 },
      // 2. No dot after slash → returns as-is
      { entity_type: 'plan', name: 'myplan', qualified_name: 'plans/myplan', file_path: 'f2.md', char_start: 0, char_end: 3 },
      // 3. Slug (lowercase + hyphen) → returns prefix
      { entity_type: 'plan', name: 'scope', qualified_name: 'plans/myplan.scope-name', file_path: 'f3.md', char_start: 0, char_end: 3 },
      // 4. Uppercase → returns as-is (non-slug)
      { entity_type: 'plan', name: 'Scope', qualified_name: 'plans/myplan.ScopeName', file_path: 'f4.md', char_start: 0, char_end: 3 },
      // 5. No hyphen → returns as-is (non-slug)
      { entity_type: 'plan', name: 'nohyphen', qualified_name: 'plans/myplan.nohyphen', file_path: 'f5.md', char_start: 0, char_end: 3 },
    ];
    mockExtractDocEntities.mockResolvedValue({
      entities: docEntities,
      edges: [],
      entityMap: new Map(),
    });
    mockReadFile.mockResolvedValue('abcdef');
    const client = makeMockClient({
      entityRows: docEntities.map((e, i) => ({ entity_id: i + 1, qualified_name: e.qualified_name })),
    });

    // Use dryRun to avoid DB complexity; buildHeadingTextMap is called before dryRun check
    const result = await buildEntityGraph({ dryRun: true });
    expect(result.entities).toBe(5);
    // readFile called for each doc entity (5 times)
    expect(mockReadFile).toHaveBeenCalledTimes(5);
  });
});

// ---------------------------------------------------------------------------
// buildHeadingTextMap — readFile catch
// ---------------------------------------------------------------------------

describe('build-entity-graph: buildHeadingTextMap readFile catch', () => {
  it('skips unreadable files without throwing', async () => {
    const docEntities = [
      { entity_type: 'plan', name: 'good', qualified_name: 'plans/good', file_path: 'good.md', char_start: 0, char_end: 3 },
      { entity_type: 'plan', name: 'bad', qualified_name: 'plans/bad', file_path: 'bad.md', char_start: 0, char_end: 3 },
    ];
    mockExtractDocEntities.mockResolvedValue({
      entities: docEntities,
      edges: [],
      entityMap: new Map(),
    });
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.includes('bad.md')) {
        return Promise.reject(new Error('ENOENT'));
      }
      return Promise.resolve('content');
    });

    const result = await buildEntityGraph({ dryRun: true });
    expect(result.entities).toBe(2);
    // Should not throw — readFile catch handled the error
  });
});

// ---------------------------------------------------------------------------
// collectDocDocuments — via fast-glob
// ---------------------------------------------------------------------------

describe('build-entity-graph: collectDocDocuments', () => {
  it('scans doc entity sources via fast-glob', async () => {
    mockFg.mockImplementation((patterns) => {
      if (patterns.includes('plans/**/*.md')) return Promise.resolve(['plans/my-plan.md']);
      if (patterns.includes('examples/**/README.md')) return Promise.resolve(['examples/demo/README.md']);
      return Promise.resolve([]);
    });
    mockExtractDocEntities.mockImplementation(({ documents }) => {
      return Promise.resolve({
        entities: documents.map((d) => ({
          entity_type: 'plan', name: d.filePath, qualified_name: d.filePath,
          file_path: d.filePath,
        })),
        edges: [],
        entityMap: new Map(),
      });
    });

    const result = await buildEntityGraph({ dryRun: true });
    expect(result.docEntityCount).toBe(2);
    // fg should be called 7 times (7 DOC_ENTITY_SOURCES)
    expect(mockFg).toHaveBeenCalledTimes(7);
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('build-entity-graph: main()', () => {
  const origArgv1 = process.argv[1];
  const origExitCode = process.exitCode;

  afterEach(() => {
    process.argv[1] = origArgv1;
    process.exitCode = origExitCode;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-entity-graph.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('writes text output on success', async () => {
    mockParseCliArgs.mockReturnValue({
      json: false,
      'dry-run': false,
      force: false,
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-entity-graph.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(false);
  });

  it('writes JSON output on success', async () => {
    mockParseCliArgs.mockReturnValue({
      json: true,
      'dry-run': true,
      force: false,
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-entity-graph.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(true);
  });

  it('calls fail on Error instance', async () => {
    mockParseCliArgs.mockReturnValue({ json: false });
    mockExtractCodeEntities.mockRejectedValue(new Error('extract failed'));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-entity-graph.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('extract failed');
  });

  it('calls fail with String(error) for non-Error', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    mockExtractCodeEntities.mockRejectedValue('string error');
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-entity-graph.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
    expect(mockFail.mock.calls[0][1]).toBe(true);
  });
});