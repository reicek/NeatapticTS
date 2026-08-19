/**
 * @module search-advanced.test
 * @description Coverage tests for search-advanced.mjs — full pipeline:
 * expansion, retrieval, context assembly, fallback, compact, top_result,
 * follow_up_refs, timeout, and error handling.
 */
import { jest } from '@jest/globals';
import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_CHUNK_ID,
  TEST_PARENT_CHUNK_ID,
  TEST_FILE_PATH,
  TEST_FAMILY,
} from '../__tests__/turso-test-helpers.mjs';

// Mock upstream modules
const mockExpandQuery = jest.fn();
const mockAssembleContext = jest.fn();
const mockClassifyAndRoute = jest.fn();
const mockSearchCorpus = jest.fn();
const mockBuildRankingExplanation = jest.fn();
const mockBuildResponseFreshness = jest.fn();

jest.unstable_mockModule('../../../rag-index/expand-query.mjs', () => ({
  expandQuery: mockExpandQuery,
  __esModule: true,
}));

jest.unstable_mockModule('../../../rag-index/assemble-context.mjs', () => ({
  assembleContext: mockAssembleContext,
  __esModule: true,
}));

jest.unstable_mockModule('../../../rag-index/routing-table.mjs', () => ({
  classifyAndRoute: mockClassifyAndRoute,
  __esModule: true,
}));

jest.unstable_mockModule('./search-corpus.mjs', () => ({
  searchCorpus: mockSearchCorpus,
  buildRankingExplanation: mockBuildRankingExplanation,
  buildResponseFreshness: mockBuildResponseFreshness,
  __esModule: true,
}));

const { searchAdvanced, searchAdvancedTool } =
  await import('./search-advanced.mjs');

/**
 * Build a mock search result row.
 * @param {object} overrides
 * @returns {object}
 */
function mockResult(overrides = {}) {
  return {
    chunk_id: 1,
    doc_id: 999999,
    file_path: TEST_FILE_PATH,
    family: TEST_FAMILY,
    chunk_index: 0,
    heading_path: 'Test',
    text: 'test result text content',
    body_text: 'test result text content',
    score: 0.9,
    ...overrides,
  };
}

describe('search-advanced', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;

  beforeEach(async () => {
    saveEnv();
    mockExpandQuery.mockReset();
    mockAssembleContext.mockReset();
    mockClassifyAndRoute.mockReset();
    mockSearchCorpus.mockReset();
    mockBuildRankingExplanation.mockReset();
    mockBuildResponseFreshness.mockReset();

    client = await createSchemaClient();
    await insertTestFixtures(client);

    // Default mocks
    mockClassifyAndRoute.mockReturnValue({
      query_class: 'cross_boundary',
      confidence: 0.9,
      strategy: { family: null },
    });
    mockSearchCorpus.mockResolvedValue({
      results: [mockResult()],
      dense_state: 'cold',
      rerank_state: 'cold',
      diskann_used: false,
      rrf_used: false,
    });
    mockBuildResponseFreshness.mockResolvedValue({
      timestamp: Date.now(),
      stale: false,
    });
  });

  afterEach(async () => {
    restoreEnv();
    if (client) {
      try {
        await client.close();
      } catch {
        /* noop */
      }
    }
  });

  describe('validation', () => {
    it('throws on empty query', async () => {
      await expect(searchAdvanced({ query: '' })).rejects.toThrow(
        'query is required',
      );
    });

    it('throws on whitespace-only query', async () => {
      await expect(searchAdvanced({ query: '   ' })).rejects.toThrow(
        'query is required',
      );
    });

    it('throws on non-string query', async () => {
      await expect(searchAdvanced({ query: 123 })).rejects.toThrow(
        'query is required',
      );
    });

    it('throws on null query', async () => {
      await expect(searchAdvanced({ query: null })).rejects.toThrow(
        'query is required',
      );
    });

    it('throws on invalid limit (non-number)', async () => {
      await expect(
        searchAdvanced({ query: 'test', limit: 'abc' }),
      ).rejects.toThrow('limit must be a number');
    });

    it('throws on invalid budget (non-number)', async () => {
      await expect(
        searchAdvanced({ query: 'test', budget: 'abc' }),
      ).rejects.toThrow('budget must be a positive number');
    });

    it('throws on invalid budget (zero)', async () => {
      await expect(
        searchAdvanced({ query: 'test', budget: 0 }),
      ).rejects.toThrow('budget must be a positive number');
    });

    it('throws on invalid budget (negative)', async () => {
      await expect(
        searchAdvanced({ query: 'test', budget: -1 }),
      ).rejects.toThrow('budget must be a positive number');
    });
  });

  describe('basic pipeline', () => {
    it('runs the full pipeline and returns results', async () => {
      mockExpandQuery.mockResolvedValue({
        originalQuery: 'test',
        expandedTerms: ['expanded'],
        bm25Query: 'test OR expanded',
        expansion: { applied: true, reason: 'ok' },
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.query).toBe('test');
      expect(result.query_class).toBe('cross_boundary');
      expect(result.results).toHaveLength(1);
      expect(result.expansion.applied).toBe(true);
      expect(result.dense_state).toBe('cold');
      expect(result.rerank_state).toBe('cold');
      expect(result.response_tokens).toBeDefined();
      expect(result.follow_up_refs).toBeDefined();
    });

    it('uses explicit query_class when provided', async () => {
      const result = await searchAdvanced({
        query: 'test',
        query_class: 'simple_lookup',
        client,
      });

      expect(result.query_class).toBe('simple_lookup');
      // classifyAndRoute is called once during retrieval routing (runRetrieval),
      // but NOT for initial classification since an explicit query_class was provided.
      expect(mockClassifyAndRoute).toHaveBeenCalledTimes(1);
    });

    it('uses classifyAndRoute when no explicit query_class', async () => {
      await searchAdvanced({ query: 'test', client });
      expect(mockClassifyAndRoute).toHaveBeenCalledWith('test');
    });
  });

  describe('expansion stage', () => {
    it('skips expansion when expand_query is false', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'simple_lookup',
        confidence: 0.9,
        strategy: { family: null },
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.expansion.applied).toBe(false);
      expect(mockExpandQuery).not.toHaveBeenCalled();
    });

    it('runs expansion when expand_query is true', async () => {
      mockExpandQuery.mockResolvedValue({
        originalQuery: 'test',
        expandedTerms: ['term1'],
        bm25Query: 'test OR term1',
        expansion: { applied: true, reason: 'ok' },
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(mockExpandQuery).toHaveBeenCalledWith({
        query: 'test',
        expandQuery: true,
      });
      expect(result.expansion.applied).toBe(true);
    });

    it('handles expansion error gracefully', async () => {
      mockExpandQuery.mockRejectedValue(new Error('ONNX unavailable'));

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.expansion.applied).toBe(false);
      expect(result.expansion.degraded).toBe(true);
      expect(result.expansion.reason).toContain('Expansion failed');
    });

    it('handles expansion error with non-Error value', async () => {
      mockExpandQuery.mockRejectedValue('string error');

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.expansion.degraded).toBe(true);
      expect(result.expansion.reason).toContain('string error');
    });
  });

  describe('timeout handling', () => {
    it('returns timeout error after expansion when deadline exceeded', async () => {
      mockExpandQuery.mockImplementation(async () => {
        await new Promise((r) => setTimeout(r, 10));
        return {
          originalQuery: 'test',
          expandedTerms: [],
          bm25Query: null,
          expansion: { applied: true, reason: 'ok' },
        };
      });

      const result = await searchAdvanced({
        query: 'test',
        timeout_ms: 1,
        client,
      });

      expect(result.isError).toBe(true);
      expect(result.structuredContent.error).toContain('CORTEX_TIMEOUT');
      expect(result.structuredContent.expansion).toBeDefined();
      expect(result.structuredContent.results).toEqual([]);
    });

    it('returns timeout error after retrieval when deadline exceeded', async () => {
      mockExpandQuery.mockResolvedValue({
        originalQuery: 'test',
        expandedTerms: [],
        bm25Query: null,
        expansion: { applied: false, reason: 'disabled' },
      });
      mockSearchCorpus.mockImplementation(async () => {
        await new Promise((r) => setTimeout(r, 10));
        return { results: [mockResult()], dense_state: 'cold' };
      });

      const result = await searchAdvanced({
        query: 'test',
        timeout_ms: 5,
        client,
      });

      expect(result.isError).toBe(true);
      expect(result.structuredContent.error).toContain('CORTEX_TIMEOUT');
      expect(result.structuredContent.results).toHaveLength(1);
    });
  });

  describe('include_code_only', () => {
    it('filters out readme and generated-readme families by default for code_specific', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'code_specific',
        confidence: 0.9,
        strategy: { family: null },
      });
      mockSearchCorpus.mockResolvedValue({
        results: [
          mockResult({ chunk_id: 1, family: 'src' }),
          mockResult({ chunk_id: 2, family: 'readme' }),
          mockResult({ chunk_id: 3, family: 'generated-readme' }),
        ],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.include_code_only).toBe(true);
      expect(result.results).toHaveLength(1);
      expect(result.results[0].family).toBe('src');
    });

    it('does NOT filter when include_code_only is false', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'code_specific',
        confidence: 0.9,
        strategy: { family: null },
      });
      mockSearchCorpus.mockResolvedValue({
        results: [
          mockResult({ chunk_id: 1, family: 'src' }),
          mockResult({ chunk_id: 2, family: 'readme' }),
        ],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({
        query: 'test',
        include_code_only: false,
        client,
      });

      expect(result.results).toHaveLength(2);
    });

    it('does NOT filter for non-code_specific classes by default', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [mockResult({ chunk_id: 1, family: 'readme' })],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.include_code_only).toBe(false);
      expect(result.results).toHaveLength(1);
    });
  });

  describe('auto_fallback', () => {
    it('triggers native fallback when results are empty', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({
        query: 'tursouniqueword',
        auto_fallback: true,
        client,
      });

      expect(result.fallback_triggered).toBe(true);
      // The fallback should find results from the test fixtures
      expect(result.results.length).toBeGreaterThan(0);
    });

    it('triggers native fallback when confidence is low', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'cross_boundary',
        confidence: 0.3,
        strategy: { family: null },
      });
      mockSearchCorpus.mockResolvedValue({
        results: [mockResult()],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({
        query: 'tursouniqueword',
        auto_fallback: true,
        client,
      });

      expect(result.fallback_triggered).toBe(true);
    });

    it('does NOT trigger fallback when auto_fallback is false', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({
        query: 'test',
        auto_fallback: false,
        client,
      });

      expect(result.fallback_triggered).toBe(false);
    });
  });

  describe('explain_ranking', () => {
    it('adds ranking_explanation to results', async () => {
      mockBuildRankingExplanation.mockReturnValue({
        score: 0.9,
        components: { bm25: 0.8 },
      });

      const result = await searchAdvanced({
        query: 'test',
        explain_ranking: true,
        client,
      });

      expect(mockBuildRankingExplanation).toHaveBeenCalled();
      expect(result.results[0].ranking_explanation).toBeDefined();
    });
  });

  describe('context assembly', () => {
    it('assembles context when budget is provided', async () => {
      mockAssembleContext.mockResolvedValue({
        context: 'assembled context text',
        tokenCount: 50,
        tierCounts: { essential: 1, supporting: 0, supplementary: 0 },
        selectedChunks: [{ chunk_id: 1 }],
      });

      const result = await searchAdvanced({
        query: 'test',
        budget: 512,
        client,
      });

      expect(result.context).toBe('assembled context text');
      expect(result.token_count).toBe(50);
      expect(result.tier_counts).toEqual({
        essential: 1,
        supporting: 0,
        supplementary: 0,
      });
      expect(result.chunks_in_context).toBe(1);
    });

    it('assembles context when context_budget is provided', async () => {
      mockAssembleContext.mockResolvedValue({
        context: 'ctx',
        tokenCount: 10,
        tierCounts: { essential: 0, supporting: 0, supplementary: 0 },
        selectedChunks: [],
      });

      const result = await searchAdvanced({
        query: 'test',
        context_budget: 256,
        client,
      });

      expect(result.context).toBe('ctx');
    });

    it('does NOT assemble context when no budget provided', async () => {
      const result = await searchAdvanced({ query: 'test', client });

      expect(result.context).toBeUndefined();
      expect(mockAssembleContext).not.toHaveBeenCalled();
    });
  });

  describe('compact mode', () => {
    it('compacts results when compact=true', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [
          mockResult({
            text: 'x'.repeat(500),
            body_text: 'x'.repeat(500),
          }),
        ],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({
        query: 'test',
        compact: true,
        client,
      });

      expect(result.results[0].text.length).toBeLessThanOrEqual(300);
      expect(result.results[0].text).toContain('…');
    });

    it('preserves text under threshold in compact mode', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [mockResult({ text: 'short text', body_text: 'short text' })],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({
        query: 'test',
        compact: true,
        client,
      });

      expect(result.results[0].text).toBe('short text');
    });
  });

  describe('read_top_result', () => {
    it('builds top_result when read_top_result=true', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [
          mockResult({
            chunk_id: TEST_CHUNK_ID,
            body_text: 'full body text',
          }),
        ],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({
        query: 'test',
        read_top_result: true,
        client,
      });

      expect(result.top_result).toBeDefined();
      expect(result.top_result.chunk_id).toBe(TEST_CHUNK_ID);
      expect(result.top_result.text).toBe('full body text');
    });

    it('does NOT build top_result by default', async () => {
      const result = await searchAdvanced({ query: 'test', client });
      expect(result.top_result).toBeUndefined();
    });
  });

  describe('follow_up_refs', () => {
    it('builds follow_up_refs with load_chunk and search_advanced', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [
          mockResult({ chunk_id: TEST_PARENT_CHUNK_ID, doc_id: 500001 }),
        ],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.follow_up_refs).toHaveLength(3); // load_chunk, next chunk, search_advanced
      expect(result.follow_up_refs[0].tool).toBe('load_chunk');
      expect(result.follow_up_refs[2].tool).toBe('search_advanced');
    });

    it('returns empty follow_up_refs when no results', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [],
        dense_state: 'cold',
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.follow_up_refs).toEqual([]);
    });
  });

  describe('rerank candidates', () => {
    it('includes rerank_candidates_count when use_rerank is true', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'cross_boundary',
        confidence: 0.9,
        strategy: { family: null },
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.rerank_candidates_count).toBeDefined();
    });

    it('does NOT include rerank_candidates_count when use_rerank is false', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'simple_lookup',
        confidence: 0.9,
        strategy: { family: null },
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.rerank_candidates_count).toBeUndefined();
    });

    it('uses custom rerank_candidates_count when provided as valid number', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'cross_boundary',
        confidence: 0.9,
        strategy: { family: null },
      });

      const result = await searchAdvanced({
        query: 'test',
        client,
        rerank_candidates_count: 20,
      });

      expect(result.rerank_candidates_count).toBe(20);
    });
  });

  describe('dense_degraded', () => {
    it('includes dense_degraded when searchCorpus reports it', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [mockResult()],
        dense_state: 'cold',
        dense_degraded: true,
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.dense_degraded).toBe(true);
    });
  });

  describe('turso_native_features', () => {
    it('reports diskann_used and rrf_used from searchCorpus', async () => {
      mockSearchCorpus.mockResolvedValue({
        results: [mockResult()],
        dense_state: 'warm',
        diskann_used: true,
        rrf_used: true,
      });

      const result = await searchAdvanced({
        query: 'test',
        use_dense: true,
        client,
      });

      expect(result.turso_native_features.diskann_used).toBe(true);
      expect(result.turso_native_features.rrf_used).toBe(true);
      expect(result.turso_native_features.vector_search_used).toBe(true);
    });
  });

  describe('searchAdvancedTool alias', () => {
    it('is the same function as searchAdvanced', () => {
      expect(searchAdvancedTool).toBe(searchAdvanced);
    });
  });

  describe('config resolution', () => {
    it('uses custom alpha clamped to [0,1]', async () => {
      const result = await searchAdvanced({
        query: 'test',
        alpha: 1.5,
        client,
      });

      expect(result.alpha).toBe(1);
    });

    it('uses custom alpha=0', async () => {
      const result = await searchAdvanced({
        query: 'test',
        alpha: 0,
        client,
      });

      expect(result.alpha).toBe(0);
    });

    it('respects explicit use_rerank override', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'simple_lookup',
        confidence: 0.9,
        strategy: { family: null },
      });

      const result = await searchAdvanced({
        query: 'test',
        use_rerank: true,
        client,
      });

      expect(result.use_rerank).toBe(true);
      expect(result.rerank_candidates_count).toBeDefined();
    });

    it('respects explicit use_dense override', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'cross_boundary',
        confidence: 0.9,
        strategy: { family: null },
      });

      const result = await searchAdvanced({
        query: 'test',
        use_dense: false,
        client,
      });

      expect(result.use_dense).toBe(false);
    });

    it('uses fallback defaults for unknown query_class', async () => {
      mockClassifyAndRoute.mockReturnValue({
        query_class: 'unknown_class',
        confidence: 0.9,
        strategy: { family: null },
      });

      const result = await searchAdvanced({ query: 'test', client });

      expect(result.alpha).toBe(0.5);
      expect(result.expand_query).toBe(false);
    });
  });

  describe('limit normalization', () => {
    it('clamps limit to [1, 100]', async () => {
      const result = await searchAdvanced({
        query: 'test',
        limit: 200,
        client,
      });

      expect(result.limit).toBe(100);
    });

    it('enforces minimum limit of 1', async () => {
      const result = await searchAdvanced({
        query: 'test',
        limit: 0,
        client,
      });

      expect(result.limit).toBe(1);
    });
  });

  describe('timeout_ms default', () => {
    it('uses default timeout when not provided', async () => {
      const result = await searchAdvanced({ query: 'test', client });
      // Should not time out with default 30s
      expect(result.isError).toBeUndefined();
    });

    it('uses default timeout when timeout_ms is invalid', async () => {
      const result = await searchAdvanced({
        query: 'test',
        timeout_ms: -1,
        client,
      });

      expect(result.isError).toBeUndefined();
    });
  });
});
