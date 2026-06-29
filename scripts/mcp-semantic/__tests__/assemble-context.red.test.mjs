/**
 * @module assemble-context.red.test
 * @description Red tests for the five-stage assembleContext pipeline
 * (enrich, deduplicate, order, budget, stitch) and the search_context MCP tool contract.
 */

import { createHash } from 'node:crypto';

const ASSEMBLE_CONTEXT_PATH = '../../../rag-index/assemble-context.mjs';
const SEARCH_CONTEXT_PATH = '../tools/search-context.mjs';

function loadAssembleContext() {
  return import(ASSEMBLE_CONTEXT_PATH);
}

function loadSearchContext() {
  return import(SEARCH_CONTEXT_PATH);
}

function makeChunk(overrides = {}) {
  return {
    chunk_id: 1,
    file_path: 'src/foo.ts',
    doc_family: 'src',
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

function sha256(text) {
  return createHash('sha256').update(text).digest('hex');
}

function vector(dim, fill) {
  return Array.from({ length: dim }, () => fill);
}

describe('assemble-context pipeline', () => {
  describe('enrichChunks', () => {
    it('adds a SHA-256 hash and context header to each chunk', async () => {
      const { enrichChunks } = await loadAssembleContext();
      const chunk = makeChunk({
        chunk_id: 10,
        file_path: 'src/bar.ts',
        heading_path: 'Bar',
        body_text: 'const x = 1;',
      });
      const [enriched] = await enrichChunks([chunk], { query_class: 'code' });
      expect(enriched).toEqual(
        expect.objectContaining({
          chunk_id: 10,
          sha256: sha256('const x = 1;'),
          context_header: 'src/bar.ts > Bar',
        }),
      );
    });

    it('preserves existing embeddings when they are already present', async () => {
      const { enrichChunks } = await loadAssembleContext();
      const embedding = vector(3, 0.1);
      const chunk = makeChunk({ chunk_id: 20, embedding });
      const [enriched] = await enrichChunks([chunk], { query_class: 'code' });
      expect(enriched.embedding).toBe(embedding);
    });

    it('is idempotent when called twice on the same inputs', async () => {
      const { enrichChunks } = await loadAssembleContext();
      const chunks = [makeChunk({ chunk_id: 30, body_text: 'let a = 2;' })];
      const first = await enrichChunks(chunks, { query_class: 'code' });
      const second = await enrichChunks(first, { query_class: 'code' });
      expect(second).toEqual(first);
    });
  });

  describe('deduplicateChunks', () => {
    it('removes chunks with identical SHA-256 hashes', async () => {
      const { deduplicateChunks } = await loadAssembleContext();
      const sharedText = 'duplicate body';
      const chunks = [
        makeChunk({ chunk_id: 1, body_text: sharedText }),
        makeChunk({ chunk_id: 2, body_text: sharedText }),
        makeChunk({ chunk_id: 3, body_text: 'unique body' }),
      ];
      const result = deduplicateChunks(chunks, { cosineThreshold: 0.95 });
      expect(result.map((c) => c.chunk_id)).toEqual([1, 3]);
    });

    it('removes near-duplicate chunks when cosine similarity is >= 0.95', async () => {
      const { deduplicateChunks } = await loadAssembleContext();
      const embedding = vector(4, 0.5);
      const nearDuplicate = embedding.map((v, index) =>
        index === 0 ? v + 0.02 : v,
      );
      const chunks = [
        makeChunk({ chunk_id: 1, embedding }),
        makeChunk({ chunk_id: 2, embedding: nearDuplicate }),
        makeChunk({ chunk_id: 3, embedding: vector(4, -0.5) }),
      ];
      const result = deduplicateChunks(chunks, { cosineThreshold: 0.95 });
      expect(result.map((c) => c.chunk_id)).toEqual([1, 3]);
    });

    it('keeps near-duplicate chunks when embeddings are cold', async () => {
      const { deduplicateChunks } = await loadAssembleContext();
      const chunks = [
        makeChunk({ chunk_id: 1, embedding: null }),
        makeChunk({ chunk_id: 2, embedding: null }),
      ];
      const result = deduplicateChunks(chunks, { cosineThreshold: 0.95 });
      expect(result.map((c) => c.chunk_id)).toEqual([1, 2]);
    });

    it('collapses parents in favor of their children', async () => {
      const { deduplicateChunks } = await loadAssembleContext();
      const parent = makeChunk({ chunk_id: 1, parent_chunk_id: null });
      const child = makeChunk({
        chunk_id: 2,
        parent_chunk_id: 1,
        body_text: 'child body',
      });
      const result = deduplicateChunks([parent, child], {
        cosineThreshold: 0.95,
      });
      expect(result.map((c) => c.chunk_id)).toEqual([2]);
    });

    it('does not remove unrelated siblings', async () => {
      const { deduplicateChunks } = await loadAssembleContext();
      const chunks = [
        makeChunk({ chunk_id: 1, parent_chunk_id: null }),
        makeChunk({ chunk_id: 2, parent_chunk_id: 1, body_text: 'alpha' }),
        makeChunk({ chunk_id: 3, parent_chunk_id: 1, body_text: 'beta' }),
      ];
      const result = deduplicateChunks(chunks, { cosineThreshold: 0.95 });
      expect(result.map((c) => c.chunk_id)).toEqual([2, 3]);
    });
  });

  describe('orderChunks', () => {
    it('assigns essential tier to chunks with score >= 0.7', async () => {
      const { orderChunks } = await loadAssembleContext();
      const chunks = [makeChunk({ chunk_id: 1, score: 0.75 })];
      const [ordered] = orderChunks(chunks);
      expect(ordered.tier).toBe('essential');
    });

    it('assigns supporting tier to chunks with score between 0.4 and 0.7', async () => {
      const { orderChunks } = await loadAssembleContext();
      const chunks = [makeChunk({ chunk_id: 1, score: 0.5 })];
      const [ordered] = orderChunks(chunks);
      expect(ordered.tier).toBe('supporting');
    });

    it('assigns supplementary tier to chunks with score < 0.4', async () => {
      const { orderChunks } = await loadAssembleContext();
      const chunks = [makeChunk({ chunk_id: 1, score: 0.3 })];
      const [ordered] = orderChunks(chunks);
      expect(ordered.tier).toBe('supplementary');
    });

    it('groups files by their maximum score descending', async () => {
      const { orderChunks } = await loadAssembleContext();
      const chunks = [
        makeChunk({ chunk_id: 1, file_path: 'src/low.ts', score: 0.3 }),
        makeChunk({ chunk_id: 2, file_path: 'src/high.ts', score: 0.8 }),
        makeChunk({ chunk_id: 3, file_path: 'src/mid.ts', score: 0.5 }),
      ];
      const result = orderChunks(chunks);
      expect(result.map((c) => c.file_path)).toEqual([
        'src/high.ts',
        'src/mid.ts',
        'src/low.ts',
      ]);
    });

    it('orders chunks within a file by char_start ascending', async () => {
      const { orderChunks } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          file_path: 'src/a.ts',
          char_start: 40,
          score: 0.6,
        }),
        makeChunk({
          chunk_id: 2,
          file_path: 'src/a.ts',
          char_start: 10,
          score: 0.6,
        }),
        makeChunk({
          chunk_id: 3,
          file_path: 'src/a.ts',
          char_start: 25,
          score: 0.6,
        }),
      ];
      const result = orderChunks(chunks);
      expect(result.map((c) => c.chunk_id)).toEqual([2, 3, 1]);
    });

    it('prefers family priority when scores are equal', async () => {
      const { orderChunks } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          file_path: 'src/a.ts',
          score: 0.6,
          doc_family: 'test',
        }),
        makeChunk({
          chunk_id: 2,
          file_path: 'src/b.ts',
          score: 0.6,
          doc_family: 'src',
        }),
      ];
      const result = orderChunks(chunks, { familyPriority: ['src', 'test'] });
      expect(result.map((c) => c.chunk_id)).toEqual([2, 1]);
    });
  });

  describe('enforceBudget', () => {
    it('defaults to a 4096 token budget', async () => {
      const { enforceBudget } = await loadAssembleContext();
      const chunks = [makeChunk({ chunk_id: 1, body_text: 'tiny' })];
      const result = enforceBudget(chunks);
      expect(result.budget).toBe(4096);
    });

    it('includes all essential chunks even when over budget', async () => {
      const { enforceBudget } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          tier: 'essential',
          body_text: 'a '.repeat(5000),
        }),
      ];
      const result = enforceBudget(chunks, { budget: 10 });
      expect(result.selectedChunks.map((c) => c.chunk_id)).toEqual([1]);
    });

    it('truncates supplementary chunks before supporting chunks', async () => {
      const { enforceBudget } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          tier: 'supplementary',
          body_text: 'extra content here',
        }),
        makeChunk({
          chunk_id: 2,
          tier: 'supporting',
          body_text: 'supporting content',
        }),
      ];
      const result = enforceBudget(chunks, { budget: 5 });
      expect(result.selectedChunks.map((c) => c.chunk_id)).toEqual([2]);
    });

    it('reports token_count not exceeding the budget', async () => {
      const { enforceBudget } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          tier: 'essential',
          body_text: 'one two three four five',
        }),
        makeChunk({
          chunk_id: 2,
          tier: 'supplementary',
          body_text: 'six seven eight nine ten',
        }),
      ];
      const result = enforceBudget(chunks, { budget: 4 });
      expect(result.tokenCount).toBeLessThanOrEqual(4);
    });

    it('truncates at the last sentence boundary when possible', async () => {
      const { enforceBudget } = await loadAssembleContext();
      const body = 'First sentence. Second sentence. Third sentence.';
      const chunks = [
        makeChunk({ chunk_id: 1, tier: 'essential', body_text: body }),
      ];
      const result = enforceBudget(chunks, { budget: 5 });
      expect(result.selectedChunks[0].body_text).toMatch(/\.$/);
    });
  });

  describe('stitchContext', () => {
    it('emits context headers in markdown format', async () => {
      const { stitchContext } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          file_path: 'src/a.ts',
          heading_path: 'A',
          context_header: 'src/a.ts > A',
          body_text: 'body a',
        }),
      ];
      const result = stitchContext(chunks, { context_format: 'markdown' });
      expect(result).toMatch(/\[src\/a\.ts > A\]\s*\n?body a/);
    });

    it('does not repeat the same file header for contiguous same-file chunks', async () => {
      const { stitchContext } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          file_path: 'src/a.ts',
          heading_path: 'A',
          context_header: 'src/a.ts > A',
          body_text: 'one',
        }),
        makeChunk({
          chunk_id: 2,
          file_path: 'src/a.ts',
          heading_path: 'A',
          context_header: 'src/a.ts > A',
          body_text: 'two',
        }),
      ];
      const result = stitchContext(chunks, { context_format: 'markdown' });
      const headerCount = (result.match(/\[src\/a\.ts > A\]/g) || []).length;
      expect(headerCount).toBe(1);
    });

    it('emits JSON output when context_format is json', async () => {
      const { stitchContext } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          context_header: 'src/a.ts > A',
          body_text: 'body a',
        }),
      ];
      const result = stitchContext(chunks, { context_format: 'json' });
      expect(result).toEqual(
        expect.objectContaining({
          chunks: expect.any(Array),
          context: expect.any(String),
          tokenCount: expect.any(Number),
        }),
      );
    });

    it('re-emits a header when the file changes', async () => {
      const { stitchContext } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          file_path: 'src/a.ts',
          heading_path: 'A',
          context_header: 'src/a.ts > A',
          body_text: 'a',
        }),
        makeChunk({
          chunk_id: 2,
          file_path: 'src/b.ts',
          heading_path: 'B',
          context_header: 'src/b.ts > B',
          body_text: 'b',
        }),
      ];
      const result = stitchContext(chunks, { context_format: 'markdown' });
      expect(result).toMatch(/\[src\/a\.ts > A\]/);
      expect(result).toMatch(/\[src\/b\.ts > B\]/);
    });
  });

  describe('assembleContext', () => {
    it('runs the full pipeline and returns context and metadata', async () => {
      const { assembleContext } = await loadAssembleContext();
      const chunks = [
        makeChunk({
          chunk_id: 1,
          file_path: 'src/a.ts',
          score: 0.8,
          body_text: 'important.',
        }),
        makeChunk({
          chunk_id: 2,
          file_path: 'src/a.ts',
          score: 0.2,
          body_text: 'noise.',
        }),
      ];
      const result = await assembleContext(chunks, {
        budget: 100,
        context_format: 'markdown',
        query_class: 'code',
      });
      expect(result).toEqual(
        expect.objectContaining({
          context: expect.any(String),
          tokenCount: expect.any(Number),
          tierCounts: expect.any(Object),
        }),
      );
    });

    it('returns the same result for identical inputs', async () => {
      const { assembleContext } = await loadAssembleContext();
      const chunks = [
        makeChunk({ chunk_id: 1, score: 0.6, body_text: 'stable.' }),
      ];
      const options = {
        budget: 50,
        context_format: 'markdown',
        query_class: 'code',
      };
      const first = await assembleContext(chunks, options);
      const second = await assembleContext(chunks, options);
      expect(second).toEqual(first);
    });
  });
});

describe('search-context MCP tool', () => {
  describe('input validation', () => {
    it('requires a non-empty query string', async () => {
      const { searchContext } = await loadSearchContext();
      await expect(searchContext({ query: '' })).rejects.toThrow('query');
    });

    it('rejects an unsupported context_format', async () => {
      const { searchContext } = await loadSearchContext();
      await expect(
        searchContext({ query: 'NEAT', context_format: 'xml' }),
      ).rejects.toThrow('context_format');
    });

    it('rejects a non-positive token budget', async () => {
      const { searchContext } = await loadSearchContext();
      await expect(searchContext({ query: 'NEAT', budget: 0 })).rejects.toThrow(
        'budget',
      );
    });
  });

  describe('output contract', () => {
    it('returns context, token_count, tier_counts, and dense_state', async () => {
      const { searchContext } = await loadSearchContext();
      const result = await searchContext({
        query: 'NEAT',
        limit: 3,
        budget: 100,
      });
      expect(result).toEqual(
        expect.objectContaining({
          context: expect.any(String),
          token_count: expect.any(Number),
          tier_counts: expect.any(Object),
          dense_state: expect.any(String),
        }),
      );
    });

    it('returns dense_degraded true when embeddings are cold', async () => {
      const { searchContext } = await loadSearchContext();
      const result = await searchContext({ query: 'NEAT', use_dense: true });
      expect(result.dense_degraded).toBe(true);
    });

    it('defaults context_format to markdown', async () => {
      const { searchContext } = await loadSearchContext();
      const result = await searchContext({ query: 'NEAT' });
      expect(result.context_format).toBe('markdown');
    });
  });
});
