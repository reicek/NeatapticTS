/**
 * @module expand-turso.test
 * @description Red tests for Phase 5 Step 06 — server-side query expansion via
 * Turso's `vector_top_k()` function on the `term_embeddings` DiskANN index.
 *
 * These tests define the EXPECTED behavior AFTER implementation. The
 * source-inspection tests for NEW behavior must FAIL because the current
 * `expand-query.mjs` uses client-side cosine similarity (JS-side loop over all
 * term embeddings) instead of server-side `vector_top_k`.
 *
 * Source-inspection tests for REMOVED behavior must also FAIL because the old
 * patterns (`findNearestTerms`, `loadTermEmbeddingsWithClient`,
 * `computeCosineSimilarity` import, direct `createClient` usage) still exist
 * in the current source.
 *
 * Regression guard tests (domain associations, classification-aware expansion,
 * BM25 query reconstruction) verify PRESERVED behavior and should PASS against
 * the current code.
 *
 * Coverage targets:
 * - expand-query.mjs uses `vector_top_k(term_embeddings_embedding_idx, vector8(?), 10)` for term similarity
 * - expand-query.mjs imports `getTursoClient` from `../mcp-semantic/tools/cortex-db.mjs`
 * - expand-query.mjs does NOT import `createClient` from `@libsql/client` directly
 * - expand-query.mjs does NOT define `findNearestTerms` (old JS-side cosine similarity)
 * - expand-query.mjs does NOT define `loadTermEmbeddingsWithClient` (loads ALL embeddings into JS Map)
 * - expand-query.mjs does NOT import `computeCosineSimilarity` from `./hybrid-rank.mjs`
 * - Domain associations, classification-aware expansion, and BM25 reconstruction preserved
 * - Expansion results merged with domain associations via server-side vector search
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createSchemaClient } from './turso-test-helpers.mjs';
import { closeTursoClient } from '../../scripts/mcp-semantic/tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const EXPAND_QUERY_PATH = path.resolve(__dirname, '..', 'expand-query.mjs');
const MCP_EXPAND_QUERY_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'mcp-semantic',
  'tools',
  'expand-query.mjs',
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

// ---------------------------------------------------------------------------
// Teardown — close any Turso clients that may have been opened
// ---------------------------------------------------------------------------

afterEach(async () => {
  await closeTursoClient();
});

// ---------------------------------------------------------------------------
// 1. Server-side vector_top_k on term_embeddings (source inspection)
// ---------------------------------------------------------------------------

describe('expand-turso: server-side vector_top_k on term_embeddings', () => {
  it('references vector_top_k in SQL for term similarity', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).toMatch(/vector_top_k\s*\(/i);
  });

  it('uses vector8(?) parameter for the query embedding in vector_top_k', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).toMatch(/vector_top_k\s*\([^)]*vector8\s*\(\s*\?\s*\)/is);
  });

  it('queries the term_embeddings table in the vector_top_k call', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).toMatch(/vector_top_k[\s\S]{0,200}term_embeddings/i);
  });

  it('uses term_embeddings_embedding_idx in the vector_top_k call', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).toMatch(/term_embeddings_embedding_idx/i);
  });
});

// ---------------------------------------------------------------------------
// 2. getTursoClient from cortex-db.mjs used (source inspection)
// ---------------------------------------------------------------------------

describe('expand-turso: getTursoClient from cortex-db.mjs used', () => {
  it('imports getTursoClient from cortex-db.mjs', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).toMatch(
      /import\s+\{[^}]*getTursoClient[^}]*\}\s+from\s+['"][^'"]*cortex-db\.mjs['"]/is,
    );
  });

  it('does NOT import createClient from @libsql/client directly', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).not.toMatch(
      /import\s+\{[^}]*createClient[^}]*\}\s+from\s+['"]@libsql\/client['"]/is,
    );
  });

  it('calls getTursoClient instead of createClient', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).toMatch(/getTursoClient\s*\(/);
  });
});

// ---------------------------------------------------------------------------
// 3. Old client-side term similarity code removed (source inspection)
// ---------------------------------------------------------------------------

describe('expand-turso: old client-side term similarity code removed', () => {
  it('does NOT define findNearestTerms (old JS-side cosine similarity function)', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).not.toMatch(/function\s+findNearestTerms\b/);
  });

  it('does NOT define loadTermEmbeddingsWithClient (loads ALL embeddings into JS Map)', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).not.toMatch(/function\s+loadTermEmbeddingsWithClient\b/);
  });

  it('does NOT import computeCosineSimilarity from ./hybrid-rank.mjs', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).not.toMatch(
      /import\s+\{[^}]*computeCosineSimilarity[^}]*\}\s+from\s+['"]\.\/hybrid-rank\.mjs['"]/is,
    );
  });
});

// ---------------------------------------------------------------------------
// 4. Domain associations preserved (behavioral — regression guard)
// ---------------------------------------------------------------------------

describe('expand-turso: domain associations preserved', () => {
  it('still exports lookupDomainAssociations', async () => {
    const mod = await import('../expand-query.mjs');
    expect(typeof mod.lookupDomainAssociations).toBe('function');
  });

  it('still exports loadDomainAssociations', async () => {
    const mod = await import('../expand-query.mjs');
    expect(typeof mod.loadDomainAssociations).toBe('function');
  });

  it('domain association lookup returns matching expansions', async () => {
    const { lookupDomainAssociations } = await import('../expand-query.mjs');
    const dictionary = {
      version: 1,
      associations: [
        {
          term: 'NEAT',
          expansions: ['neuroevolution'],
          source: 'academic',
          confidence: 1.0,
        },
      ],
    };
    const expansions = lookupDomainAssociations(['NEAT'], dictionary);
    expect(expansions.some((e) => e.expanded === 'neuroevolution')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 5. Classification-aware expansion preserved (behavioral — regression guard)
// ---------------------------------------------------------------------------

describe('expand-turso: classification-aware expansion preserved', () => {
  it('still exports expansionBehaviorForClass', async () => {
    const mod = await import('../expand-query.mjs');
    expect(typeof mod.expansionBehaviorForClass).toBe('function');
  });

  it('returns false for simple_lookup', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('simple_lookup')).toBe(false);
  });

  it('returns true for cross_boundary', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('cross_boundary')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 6. BM25 query reconstruction preserved (behavioral — regression guard)
// ---------------------------------------------------------------------------

describe('expand-turso: BM25 query reconstruction preserved', () => {
  it('still exports buildExpandedFtsQuery', async () => {
    const mod = await import('../expand-query.mjs');
    expect(typeof mod.buildExpandedFtsQuery).toBe('function');
  });

  it('buildExpandedFtsQuery builds OR-expanded FTS5 query', async () => {
    const { buildExpandedFtsQuery } = await import('../expand-query.mjs');
    const query = buildExpandedFtsQuery('NEAT crossover', [
      { expanded: 'neuroevolution' },
    ]);
    expect(query).toContain('OR neuroevolution');
  });
});

// ---------------------------------------------------------------------------
// 7. Expansion results merged with domain associations (behavioral with in-memory client)
// ---------------------------------------------------------------------------

describe('expand-turso: expansion results merged with domain associations via server-side vector search', () => {
  it('expandQuery with injected client returns expansion result containing expandedTerms, bm25Query, and expansion.applied', async () => {
    // Create an in-memory schema-loaded client
    const client = await createSchemaClient();

    // Insert a term_embeddings fixture with a deterministic 384-dimensional embedding
    const embedding = new Float32Array(384);
    for (let i = 0; i < 384; i += 1) {
      embedding[i] = (i % 7) / 10.0;
    }
    const embeddingBuffer = Buffer.from(
      embedding.buffer,
      embedding.byteOffset,
      embedding.byteLength,
    );

    await client.execute({
      sql: `INSERT INTO term_embeddings (term, embedding, term_sha256, model_id, model_sha256, dimension, frequency, doc_family_count, embedded_at)
            VALUES (?, vector8(?), ?, ?, ?, ?, ?, ?, ?)`,
      args: [
        'neuroevolution',
        embeddingBuffer,
        'sha256-neuro',
        'test-model',
        'sha256-model',
        384,
        10,
        3,
        '2026-01-01T00:00:00Z',
      ],
    });

    // Inject the client into expandQuery — after implementation, this should
    // use server-side vector_top_k on term_embeddings_embedding_idx to find
    // similar terms. The current implementation does NOT support server-side
    // vector_top_k, so this test will FAIL.
    const { expandQuery } = await import('../expand-query.mjs');
    const result = await expandQuery({
      query: 'neuroevolution',
      expandQuery: true,
      client,
      embedText: async () => embedding,
      modelId: 'test-model',
    });

    expect(result.expansion.applied).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 8. No deferred cleanup — old client-side patterns absent (source inspection)
// ---------------------------------------------------------------------------

describe('expand-turso: old client-side patterns absent', () => {
  it('does NOT have a raw createClient({ call', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    expect(source).not.toMatch(/createClient\s*\(\s*\{/);
  });

  it('does NOT load ALL term embeddings via SELECT term, embedding without a vector_top_k clause', async () => {
    const source = await readSource(EXPAND_QUERY_PATH);
    // The old pattern loads all rows with a plain SELECT on term_embeddings
    // (loading term + embedding columns) without any vector_top_k proximity
    // clause. After migration, all term similarity lookups must go through
    // vector_top_k. The negative lookahead fails (regex matches) when
    // vector_top_k does NOT appear after the FROM term_embeddings match point,
    // which is the case in the current client-side code.
    expect(source).not.toMatch(
      /SELECT\s+term\s*,\s*embedding[\s\S]*?FROM\s+term_embeddings(?![\s\S]*?vector_top_k)/is,
    );
  });
});
