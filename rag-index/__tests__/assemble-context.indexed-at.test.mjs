/**
 * @module assemble-context.indexed-at.test
 * @description Red tests for RAG Index Freshness Strategy Phase 3 slice
 * P3-S1-A (AC-011): assemble-context.mjs joins chunks → documents and exposes
 * the document row's `indexed_at` on each chunk in the output payload so
 * cross-family queries can show staleness per result.
 *
 * Determinism note (Level 2, content-deterministic): tests assert `indexed_at`
 * presence and that it matches the mocked document row value — never a
 * Date.now()-derived value — because `indexed_at` is intentionally
 * non-deterministic metadata excluded from freshness equality (plan risk R6).
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ASSEMBLE_CONTEXT_PATH = path.resolve(
  __dirname,
  '..',
  'assemble-context.mjs',
);

/** Fixed mock document `indexed_at` values (ms epoch), never Date.now(). */
const DOC_A_INDEXED_AT = 1_727_300_000_000;
const DOC_B_INDEXED_AT = 1_727_400_000_000;

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
 * Build a minimal chunk fixture for the assembly pipeline.
 *
 * @param {Object} [overrides] - Properties to override on the base chunk.
 * @returns {Object} A chunk object compatible with the pipeline stages.
 */
function makeChunk(overrides = {}) {
  return {
    chunk_id: 1,
    file_path: 'src/foo.ts',
    family: 'ts-source',
    heading_path: 'Foo',
    char_start: 0,
    char_end: 19,
    score: 0.9,
    body_text: 'function foo() {}',
    embedding: null,
    parent_chunk_id: null,
    context_header: null,
    sha256: null,
    tier: null,
    ...overrides,
  };
}

/**
 * Build a document-side enrichment row as returned by the SQL JOIN.
 *
 * @param {Object} [overrides] - Properties to override on the base row.
 * @returns {Object} A JOIN result row for one chunk.
 */
function makeEnrichmentRow(overrides = {}) {
  return {
    chunk_id: 1,
    chunk_index: 0,
    heading_path: 'Foo',
    body_text: 'function foo() {}',
    char_start: 0,
    char_end: 17,
    parent_chunk_id: null,
    depth: 0,
    context_header: 'src/foo.ts > Foo',
    symbol_name: null,
    signature_text: null,
    jsdoc_text: null,
    export_type: null,
    module_path: 'src/foo.ts',
    arch_layer: null,
    chunk_sha256: 'hash-foo',
    file_path: 'src/foo.ts',
    family: 'ts-source',
    indexed_at: DOC_A_INDEXED_AT,
    entity_type: null,
    entity_name: null,
    ...overrides,
  };
}

/**
 * Build a database client stub that returns the given rows from execute().
 *
 * @param {Object[]} rows - Rows the stubbed execute() call returns.
 * @returns {{ execute: Function, calls: Object[] }} Client stub plus captured calls.
 */
function makeClientStub(rows) {
  const calls = [];
  return {
    calls,
    execute: async (request) => {
      calls.push(request);
      return { rows };
    },
  };
}

// ---------------------------------------------------------------------------
// SQL surface — the enrichment JOIN must select documents.indexed_at
// ---------------------------------------------------------------------------

describe('assemble-context indexed_at: SQL surface (AC-011)', () => {
  it('enrichment JOIN query selects documents.indexed_at', async () => {
    const source = await readSource(ASSEMBLE_CONTEXT_PATH);
    expect(source).toMatch(/d\.indexed_at/);
  });
});

// ---------------------------------------------------------------------------
// Enrichment output — indexed_at from the document row lands on each chunk
// ---------------------------------------------------------------------------

describe('assemble-context indexed_at: enrichment output', () => {
  it('exposes indexed_at matching the joined document row', async () => {
    const { enrichChunks } = await loadAssembleContext();
    const client = makeClientStub([makeEnrichmentRow()]);
    const [enriched] = await enrichChunks([makeChunk()], {
      client,
      query_class: 'code',
    });
    expect(enriched.indexed_at).toBe(DOC_A_INDEXED_AT);
    expect(Number.isInteger(enriched.indexed_at)).toBe(true);
  });

  it('maps each chunk to its own document indexed_at', async () => {
    const { enrichChunks } = await loadAssembleContext();
    const client = makeClientStub([
      makeEnrichmentRow({
        chunk_id: 1,
        chunk_sha256: 'hash-a',
        file_path: 'src/a.ts',
        indexed_at: DOC_A_INDEXED_AT,
      }),
      makeEnrichmentRow({
        chunk_id: 2,
        chunk_sha256: 'hash-b',
        file_path: 'src/b.ts',
        indexed_at: DOC_B_INDEXED_AT,
      }),
    ]);
    const chunks = [
      makeChunk({ chunk_id: 1, file_path: 'src/a.ts' }),
      makeChunk({ chunk_id: 2, file_path: 'src/b.ts' }),
    ];
    const result = await enrichChunks(chunks, { client, query_class: 'code' });
    expect(result.map((chunk) => chunk.indexed_at)).toEqual([
      DOC_A_INDEXED_AT,
      DOC_B_INDEXED_AT,
    ]);
  });

  it('fallback enrichment preserves an indexed_at already present on the chunk', async () => {
    const { enrichChunks } = await loadAssembleContext();
    const [enriched] = await enrichChunks([
      makeChunk({ chunk_id: null, indexed_at: DOC_A_INDEXED_AT }),
    ]);
    expect(enriched.indexed_at).toBe(DOC_A_INDEXED_AT);
  });
});

// ---------------------------------------------------------------------------
// Full pipeline — indexed_at survives into the assembled output payload
// ---------------------------------------------------------------------------

describe('assemble-context indexed_at: assembled output payload', () => {
  it('assembleContext output payload carries indexed_at on every selected chunk', async () => {
    const { assembleContext } = await loadAssembleContext();
    const client = makeClientStub([
      makeEnrichmentRow({
        chunk_id: 1,
        chunk_sha256: 'hash-a',
        file_path: 'src/a.ts',
        indexed_at: DOC_A_INDEXED_AT,
      }),
      makeEnrichmentRow({
        chunk_id: 2,
        chunk_sha256: 'hash-b',
        file_path: 'src/b.ts',
        heading_path: 'Bar',
        indexed_at: DOC_B_INDEXED_AT,
      }),
    ]);
    const chunks = [
      makeChunk({ chunk_id: 1, file_path: 'src/a.ts' }),
      makeChunk({ chunk_id: 2, file_path: 'src/b.ts' }),
    ];
    const result = await assembleContext(chunks, {
      client,
      budget: 4096,
      query_class: 'code',
    });
    expect(
      result.selectedChunks.map((chunk) => chunk.indexed_at),
    ).toEqual([DOC_A_INDEXED_AT, DOC_B_INDEXED_AT]);
  });
});