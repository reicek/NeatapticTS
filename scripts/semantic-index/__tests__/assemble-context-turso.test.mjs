/**
 * @module assemble-context-turso.test
 * @description Red tests for Phase 5 Step 05 — server-side context assembly
 * (SQL JOIN enrichment, SQL dedup, server-side ordering, budget preservation).
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because the server-side SQL implementation does not exist yet (not because of
 * syntax errors). Budget enforcement tests are regression guards and should PASS.
 *
 * Coverage targets:
 * - assemble-context.mjs uses SQL JOIN with documents for chunk enrichment
 * - assemble-context.mjs uses SQL JOIN with entities for chunk enrichment
 * - assemble-context.mjs enrichment is a single SQL SELECT with chunks + documents
 * - assemble-context.mjs dedup uses SQL DISTINCT or GROUP BY for hash-based dedup
 * - assemble-context.mjs dedup uses chunk_sha256 column for server-side hash dedup
 * - assemble-context.mjs removes JS-side sha256Hex computation in enrichChunks
 * - assemble-context.mjs removes JS-side buildContextHeader call in enrichChunks
 * - assemble-context.mjs removes JS-side hashGroups Map for deduplication
 * - assemble-context.mjs removes JS-side collapseNearDuplicates function
 * - assemble-context.mjs removes JS-side collapseParentChild function
 * - assemble-context.mjs removes JS-side selectNearDuplicateWinner function
 * - assemble-context.mjs uses SQL ORDER BY for chunk ordering
 * - search-context.mjs passes database client to assembleContext
 * - enforceBudget selects chunks within token budget (regression)
 * - enforceBudget truncates essential chunks that exceed budget (regression)
 * - enforceBudget drops supplementary chunks that do not fit (regression)
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { closeTursoClient } from '../../mcp-semantic/tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ASSEMBLE_CONTEXT_PATH = path.resolve(
  __dirname,
  '..',
  'assemble-context.mjs',
);
const SEARCH_CONTEXT_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'mcp-semantic',
  'tools',
  'search-context.mjs',
);

/**
 * Read a source file as UTF-8 text.
 *
 * @param {string} filePath - Absolute path to the source file.
 * @returns {Promise<string>} File contents.
 */
async function readSource(filePath) {
  return readFile(filePath, 'utf8');
}

/**
 * Dynamically import the assemble-context module for functional tests.
 *
 * @returns {Promise<Object>} The module's named exports.
 */
function loadAssembleContext() {
  return import(pathToFileURL(ASSEMBLE_CONTEXT_PATH).href);
}

/**
 * Build a minimal chunk fixture for budget enforcement tests.
 *
 * @param {Object} [overrides] - Properties to override on the base chunk.
 * @returns {Object} A chunk object compatible with enforceBudget.
 */
function makeChunk(overrides = {}) {
  return {
    chunk_id: 1,
    file_path: 'src/foo.ts',
    family: 'ts-source',
    heading_path: 'Foo',
    char_start: 0,
    char_end: 19,
    score: 0.5,
    body_text: 'function foo() {}',
    embedding: null,
    parent_chunk_id: null,
    context_header: null,
    sha256: null,
    tier: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Teardown — close any Turso clients that may have been opened
// ---------------------------------------------------------------------------

afterEach(async () => {
  await closeTursoClient();
});

// ---------------------------------------------------------------------------
// Server-side JOIN enrichment
// ---------------------------------------------------------------------------

describe('assemble-context-turso: server-side JOIN enrichment', () => {
  it('assemble-context.mjs uses SQL JOIN with documents for chunk enrichment', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/JOIN\s+documents/i);
  });

  it('assemble-context.mjs uses SQL JOIN with entities for chunk enrichment', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/JOIN\s+entities/i);
  });

  it('assemble-context.mjs enrichment uses SQL SELECT query with chunks table', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/SELECT[\s\S]*FROM\s+chunks/i);
  });

  it('assemble-context.mjs enrichment query joins chunks with documents in single SQL query', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/FROM\s+chunks[\s\S]*JOIN\s+documents/i);
  });
});

// ---------------------------------------------------------------------------
// Dedup via SQL
// ---------------------------------------------------------------------------

describe('assemble-context-turso: dedup via SQL', () => {
  it('assemble-context.mjs uses SQL DISTINCT or GROUP BY for hash-based deduplication', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/(DISTINCT|GROUP\s+BY)/i);
  });

  it('assemble-context.mjs dedup uses chunk_sha256 column for server-side hash deduplication', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/chunk_sha256/i);
  });
});

// ---------------------------------------------------------------------------
// Old JS-side enrichment code removed
// ---------------------------------------------------------------------------

describe('assemble-context-turso: old JS-side enrichment removed', () => {
  it('assemble-context.mjs removes JS-side sha256Hex computation in enrichChunks', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).not.toMatch(/enrichChunks[\s\S]*sha256Hex/i);
  });

  it('assemble-context.mjs removes JS-side buildContextHeader call in enrichChunks', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).not.toMatch(/enrichChunks[\s\S]*buildContextHeader/i);
  });
});

// ---------------------------------------------------------------------------
// Old JS-side dedup code removed
// ---------------------------------------------------------------------------

describe('assemble-context-turso: old JS-side dedup removed', () => {
  it('assemble-context.mjs removes JS-side hashGroups Map for deduplication', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).not.toMatch(/hashGroups\s*=\s*new\s+Map/i);
  });

  it('assemble-context.mjs removes JS-side collapseNearDuplicates function', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).not.toMatch(/function\s+collapseNearDuplicates/i);
  });

  it('assemble-context.mjs removes JS-side collapseParentChild function', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).not.toMatch(/function\s+collapseParentChild/i);
  });

  it('assemble-context.mjs removes JS-side selectNearDuplicateWinner function', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).not.toMatch(/function\s+selectNearDuplicateWinner/i);
  });
});

// ---------------------------------------------------------------------------
// Server-side ordering
// ---------------------------------------------------------------------------

describe('assemble-context-turso: server-side ordering', () => {
  it('assemble-context.mjs uses SQL ORDER BY for chunk ordering', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/ORDER\s+BY/i);
  });
});

// ---------------------------------------------------------------------------
// search-context.mjs integration
// ---------------------------------------------------------------------------

describe('assemble-context-turso: search-context passes database client', () => {
  it('search-context.mjs passes database client to assembleContext for server-side SQL queries', async () => {
    const source = await readSource(SEARCH_CONTEXT_PATH);
    // After implementation, the assembleContext call should include a client
    // parameter so SQL queries can be executed server-side. The regex checks
    // that 'client' appears within the assembleContext(...) call arguments.
    expect(source).toMatch(/assembleContext\([^)]*client/i);
  });
});

// ---------------------------------------------------------------------------
// Budget enforcement still works (regression guards — should pass)
// ---------------------------------------------------------------------------

describe('assemble-context-turso: budget enforcement preserved', () => {
  it('enforceBudget selects chunks within token budget', async () => {
    const { enforceBudget } = await loadAssembleContext();
    const chunks = [
      makeChunk({ chunk_id: 1, tier: 'essential', body_text: 'hello world' }),
      makeChunk({ chunk_id: 2, tier: 'supporting', body_text: 'foo bar baz' }),
    ];
    const result = enforceBudget(chunks, { budget: 100 });
    expect(result.selectedChunks.map((c) => c.chunk_id)).toEqual([1, 2]);
  });

  it('enforceBudget truncates essential chunks that exceed budget', async () => {
    const { enforceBudget } = await loadAssembleContext();
    const longBody = 'a '.repeat(5000);
    const chunks = [
      makeChunk({ chunk_id: 1, tier: 'essential', body_text: longBody }),
    ];
    const result = enforceBudget(chunks, { budget: 10 });
    expect(result.selectedChunks[0].truncated).toBe(true);
  });

  it('enforceBudget drops supplementary chunks that do not fit', async () => {
    const { enforceBudget } = await loadAssembleContext();
    const chunks = [
      makeChunk({
        chunk_id: 1,
        tier: 'essential',
        body_text: 'essential body text',
      }),
      makeChunk({
        chunk_id: 2,
        tier: 'supplementary',
        body_text: 'x'.repeat(100),
      }),
      makeChunk({
        chunk_id: 3,
        tier: 'supplementary',
        body_text: 'fits',
      }),
    ];
    const result = enforceBudget(chunks, { budget: 10 });
    expect(result.selectedChunks.map((c) => c.chunk_id)).toEqual([1, 3]);
  });
});
