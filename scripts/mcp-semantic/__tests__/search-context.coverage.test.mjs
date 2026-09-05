/**
 * @module search-context.coverage.test
 * @description Coverage tests targeting every branch in search-context.mjs.
 *
 * Uses injected `searchCorpusFn` and `assembleContextFn` overrides so no real
 * database is required. Exercises both validation helpers, all result-shaping
 * branches (compact, include_metadata, json format, read_top_result), dedup
 * strategy, budget validation, dense/rerank state passthrough, freshness
 * fallback, and follow-up ref construction.
 */

import { jest } from '@jest/globals';
import { createEnvIsolation } from './turso-test-helpers.mjs';

const MODULE_PATH = '../tools/search-context.mjs';

const { saveEnv, restoreEnv } = createEnvIsolation();

const mockEvaluateSelfHeal = jest.fn();
const mockInvalidateDenseReadinessCache = jest.fn();

jest.unstable_mockModule(
  '../../agent-customization/cortex/cortex-health-guard.mjs',
  () => ({
    evaluateSelfHeal: mockEvaluateSelfHeal,
    __esModule: true,
  }),
);

jest.unstable_mockModule('../tools/search-corpus.mjs', () => ({
  buildResponseFreshness: async () => ({
    timestamp: Date.now(),
    stale: false,
    last_update_source: 'test',
  }),
  invalidateDenseReadinessCache: mockInvalidateDenseReadinessCache,
  searchCorpus: jest.fn(),
  __esModule: true,
}));

beforeEach(async () => {
  saveEnv();
  mockEvaluateSelfHeal.mockReset();
  mockInvalidateDenseReadinessCache.mockReset();
  const { searchCorpus } = await import('../tools/search-corpus.mjs');
  searchCorpus.mockReset();
});

afterEach(() => {
  restoreEnv();
});

async function loadModule() {
  return import(MODULE_PATH);
}

/**
 * Build a fake searchCorpus response.
 */
function fakeSearchResponse(results = [], overrides = {}) {
  return {
    results,
    query_class: 'default',
    dense_state: 'none',
    ...overrides,
  };
}

/**
 * Build a fake assembleContext response.
 */
function fakeAssembleResponse(overrides = {}) {
  return {
    context: '## Context\n\nBody text here.',
    tokenCount: 10,
    tierCounts: { essential: 1, supporting: 0, supplementary: 0 },
    selectedChunks: [
      {
        chunk_id: 1,
        file_path: 'src/a.ts',
        heading_path: 'Mod',
        char_start: 0,
        char_end: 10,
        body_text: 'Body text here.',
        score: 0.9,
        truncated: false,
        feedback_boost: 0,
      },
    ],
    ...overrides,
  };
}

/**
 * Build a fake chunk for searchCorpus results.
 */
function fakeChunk(id, extra = {}) {
  return {
    chunk_id: id,
    file_path: 'src/a.ts',
    family: 'ts-source',
    doc_family: 'ts-source',
    chunk_index: 0,
    heading_path: 'Mod',
    char_start: 0,
    char_end: 100,
    body_text: 'content body',
    text: 'content body',
    score: 0.9,
    ...extra,
  };
}

// ---------------------------------------------------------------------------
// validateContextFormat
// ---------------------------------------------------------------------------

describe('validateContextFormat', () => {
  it('returns "markdown" by default', async () => {
    const { validateContextFormat } = await loadModule();
    expect(validateContextFormat(undefined)).toBe('markdown');
  });

  it('returns "json" when json is requested', async () => {
    const { validateContextFormat } = await loadModule();
    expect(validateContextFormat('json')).toBe('json');
  });

  it('throws for an unsupported format', async () => {
    const { validateContextFormat } = await loadModule();
    expect(() => validateContextFormat('xml')).toThrow(/context_format/);
  });
});

// ---------------------------------------------------------------------------
// validateBudget
// ---------------------------------------------------------------------------

describe('validateBudget', () => {
  it('returns the default budget when value is undefined', async () => {
    const { validateBudget } = await loadModule();
    expect(validateBudget(undefined)).toBe(1024);
  });

  it('returns a valid positive number', async () => {
    const { validateBudget } = await loadModule();
    expect(validateBudget(2048)).toBe(2048);
  });

  it('throws for zero', async () => {
    const { validateBudget } = await loadModule();
    expect(() => validateBudget(0)).toThrow(/budget/);
  });

  it('throws for negative', async () => {
    const { validateBudget } = await loadModule();
    expect(() => validateBudget(-1)).toThrow(/budget/);
  });

  it('throws for non-finite', async () => {
    const { validateBudget } = await loadModule();
    expect(() => validateBudget(Infinity)).toThrow(/budget/);
    expect(() => validateBudget(NaN)).toThrow(/budget/);
  });
});

// ---------------------------------------------------------------------------
// normalizeSearchResultToChunk
// ---------------------------------------------------------------------------

describe('normalizeSearchResultToChunk', () => {
  it('maps text to body_text when only text is present', async () => {
    const { normalizeSearchResultToChunk } = await loadModule();
    const result = normalizeSearchResultToChunk({
      chunk_id: 1,
      text: 'hello',
    });
    expect(result.body_text).toBe('hello');
    expect(result.text).toBe('hello');
  });

  it('prefers text over body_text when both are present', async () => {
    const { normalizeSearchResultToChunk } = await loadModule();
    const result = normalizeSearchResultToChunk({
      chunk_id: 1,
      text: 'from-text',
      body_text: 'from-body',
    });
    expect(result.body_text).toBe('from-text');
  });

  it('defaults to empty string when neither is present', async () => {
    const { normalizeSearchResultToChunk } = await loadModule();
    const result = normalizeSearchResultToChunk({ chunk_id: 1 });
    expect(result.body_text).toBe('');
  });
});

// ---------------------------------------------------------------------------
// buildTopResult
// ---------------------------------------------------------------------------

describe('buildTopResult', () => {
  it('returns null when result is undefined', async () => {
    const { buildTopResult } = await loadModule();
    expect(buildTopResult(undefined)).toBeNull();
    expect(buildTopResult(null)).toBeNull();
  });

  it('builds descriptor from a result with text', async () => {
    const { buildTopResult } = await loadModule();
    const result = buildTopResult({
      chunk_id: 7,
      file_path: 'src/x.ts',
      family: 'src',
      text: 'body',
    });
    expect(result).toEqual({
      chunk_id: 7,
      file_path: 'src/x.ts',
      family: 'src',
      text: 'body',
    });
  });

  it('falls back to body_text when text is absent', async () => {
    const { buildTopResult } = await loadModule();
    const result = buildTopResult({
      chunk_id: 7,
      file_path: 'src/x.ts',
      family: 'src',
      body_text: 'body-text',
    });
    expect(result.text).toBe('body-text');
  });

  it('defaults to empty string when neither body_text nor text is present', async () => {
    const { buildTopResult } = await loadModule();
    const result = buildTopResult({
      chunk_id: 9,
      file_path: 'src/y.ts',
      family: 'src',
    });
    expect(result.text).toBe('');
  });
});

// ---------------------------------------------------------------------------
// buildFollowUpRefs
// ---------------------------------------------------------------------------

describe('buildFollowUpRefs', () => {
  it('includes load_chunk for first two results and search_context', async () => {
    const { buildFollowUpRefs } = await loadModule();
    const refs = buildFollowUpRefs(
      [
        { chunk_id: 1, text: 'a' },
        { chunk_id: 2, text: 'b' },
      ],
      'my query',
    );
    expect(refs).toHaveLength(3);
    expect(refs[0].tool).toBe('load_chunk');
    expect(refs[0].args.chunk_id).toBe(1);
    expect(refs[0].args.query).toBe('my query');
    expect(refs[1].tool).toBe('load_chunk');
    expect(refs[1].args.chunk_id).toBe(2);
    expect(refs[2].tool).toBe('search_context');
    expect(refs[2].args.query).toBe('Related: my query');
  });

  it('skips second load_chunk when only one result', async () => {
    const { buildFollowUpRefs } = await loadModule();
    const refs = buildFollowUpRefs([{ chunk_id: 1, text: 'a' }], 'q');
    expect(refs).toHaveLength(2);
    expect(refs[0].tool).toBe('load_chunk');
    expect(refs[1].tool).toBe('search_context');
  });

  it('only adds search_context when no results', async () => {
    const { buildFollowUpRefs } = await loadModule();
    const refs = buildFollowUpRefs([], 'q');
    expect(refs).toHaveLength(1);
    expect(refs[0].tool).toBe('search_context');
  });
});

// ---------------------------------------------------------------------------
// searchContext — core path (markdown, non-compact)
// ---------------------------------------------------------------------------

describe('searchContext — core path', () => {
  it('returns a full result object with all expected fields', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'test',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(result.context).toBe('## Context\n\nBody text here.');
    expect(result.token_count).toBe(10);
    expect(result.tier_counts).toEqual({
      essential: 1,
      supporting: 0,
      supplementary: 0,
    });
    expect(result.dense_state).toBe('none');
    expect(result.rerank_state).toBe('not_requested');
    expect(result.context_format).toBe('markdown');
    expect(result.compact).toBe(false);
    expect(result.total_chunks_retrieved).toBe(1);
    expect(result.chunks_in_context).toBe(1);
    expect(result.tokens_used).toBe(10);
    expect(result.context_budget_consumed).toBe(10);
    expect(result.budget_remaining).toBe(1014);
    expect(result.dedup_strategy).toBe('exact');
    expect(result.results).toHaveLength(1);
    expect(result.results[0]).toEqual(
      expect.objectContaining({
        chunk_id: 1,
        feedback_boost: 0,
        truncated: false,
      }),
    );
    expect(result.metadata).toEqual(
      expect.objectContaining({
        chunkCount: 1,
        totalTokens: 10,
        budget: 1024,
        format: 'markdown',
      }),
    );
    expect(result.follow_up_refs).toHaveLength(2);
  });

  it('requires a non-empty query', async () => {
    const { searchContext } = await loadModule();
    await expect(searchContext({ query: '' })).rejects.toThrow(/query/);
  });

  it('throws on invalid budget', async () => {
    const { searchContext } = await loadModule();
    await expect(
      searchContext({
        query: 'q',
        budget: 0,
        searchCorpusFn: async () => fakeSearchResponse(),
        assembleContextFn: async () => fakeAssembleResponse(),
      }),
    ).rejects.toThrow(/budget/);
  });

  it('throws on invalid context_format', async () => {
    const { searchContext } = await loadModule();
    await expect(
      searchContext({
        query: 'q',
        context_format: 'xml',
        searchCorpusFn: async () => fakeSearchResponse(),
        assembleContextFn: async () => fakeAssembleResponse(),
      }),
    ).rejects.toThrow(/context_format/);
  });
});

// ---------------------------------------------------------------------------
// searchContext — dedup_strategy
// ---------------------------------------------------------------------------

describe('searchContext — dedup_strategy', () => {
  it('doubles fetch limit for exact dedup', async () => {
    const { searchContext } = await loadModule();
    let capturedLimit;
    await searchContext({
      query: 'q',
      limit: 5,
      dedup_strategy: 'exact',
      searchCorpusFn: async (opts) => {
        capturedLimit = opts.limit;
        return fakeSearchResponse([fakeChunk(1)]);
      },
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(capturedLimit).toBe(10);
  });

  it('uses limit as-is for non-exact dedup', async () => {
    const { searchContext } = await loadModule();
    let capturedLimit;
    await searchContext({
      query: 'q',
      limit: 5,
      dedup_strategy: 'semantic',
      searchCorpusFn: async (opts) => {
        capturedLimit = opts.limit;
        return fakeSearchResponse([fakeChunk(1)]);
      },
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(capturedLimit).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// searchContext — include_metadata
// ---------------------------------------------------------------------------

describe('searchContext — include_metadata', () => {
  it('adds per-chunk metadata objects when include_metadata is true', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      include_metadata: true,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({
          selectedChunks: [
            {
              chunk_id: 1,
              file_path: 'src/a.ts',
              family: 'ts-source',
              chunk_index: 0,
              heading_path: 'Mod',
              depth: 0,
              parent_chunk_id: null,
              context_header: 'hdr',
              symbol_name: 'sym',
              signature_text: 'sig',
              jsdoc_text: 'doc',
              export_type: 'function',
              module_path: 'src',
              arch_layer: 'network',
              jsdoc_quality: 'good',
              jsdoc_word_count: 10,
              cyclomatic_complexity: 3,
              test_coverage: '90',
              source_path_pattern: 'src/**',
              body_text: 'body',
              score: 0.9,
              truncated: false,
              feedback_boost: 0,
            },
          ],
        }),
    });
    expect(result.results[0].metadata).toEqual(
      expect.objectContaining({
        file_path: 'src/a.ts',
        family: 'ts-source',
        chunk_index: 0,
        heading_path: 'Mod',
        depth: 0,
        symbol_name: 'sym',
      }),
    );
  });

  it('omits metadata when include_metadata is false', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      include_metadata: false,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.results[0].metadata).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// searchContext — compact mode
// ---------------------------------------------------------------------------

describe('searchContext — compact', () => {
  it('returns compact results with chunk_id and truncated only', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      compact: true,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({
          selectedChunks: [{ chunk_id: 1, truncated: true }],
        }),
    });
    expect(result.compact).toBe(true);
    expect(result.results[0]).toEqual({ chunk_id: 1, truncated: true });
  });

  it('truncates context string when compact and over threshold', async () => {
    const { searchContext } = await loadModule();
    const longContext = 'x'.repeat(3000);
    const result = await searchContext({
      query: 'q',
      compact: true,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({ context: longContext }),
    });
    expect(result.context.length).toBe(2000);
    expect(result.context.endsWith('…')).toBe(true);
  });

  it('does not truncate when compact and under threshold', async () => {
    const { searchContext } = await loadModule();
    const shortContext = 'short';
    const result = await searchContext({
      query: 'q',
      compact: true,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({ context: shortContext }),
    });
    expect(result.context).toBe('short');
  });
});

// ---------------------------------------------------------------------------
// searchContext — JSON format
// ---------------------------------------------------------------------------

describe('searchContext — json format', () => {
  it('returns a JSON context object with chunks array', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      context_format: 'json',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({
          context: 'markdown context',
          selectedChunks: [
            {
              chunk_id: 1,
              file_path: 'src/a.ts',
              heading_path: 'Mod',
              char_start: 0,
              char_end: 10,
              body_text: 'body',
              score: 0.9,
              tier: 'essential',
            },
          ],
        }),
    });
    expect(result.context_format).toBe('json');
    expect(result.context).toEqual(
      expect.objectContaining({
        context: 'markdown context',
        tokenCount: 10,
      }),
    );
    expect(result.context.chunks).toHaveLength(1);
    expect(result.context.chunks[0]).toEqual(
      expect.objectContaining({
        chunk_id: 1,
        file_path: 'src/a.ts',
        heading_path: 'Mod',
        char_start: 0,
        char_end: 10,
        content: 'body',
        score: 0.9,
        tier: 'essential',
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// searchContext — read_top_result
// ---------------------------------------------------------------------------

describe('searchContext — read_top_result', () => {
  it('includes top_result when read_top_result is true', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      read_top_result: true,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(42)]),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.top_result).toEqual({
      chunk_id: 42,
      file_path: 'src/a.ts',
      family: 'ts-source',
      text: 'content body',
    });
  });

  it('omits top_result when read_top_result is false', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      read_top_result: false,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.top_result).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// searchContext — dense/rerank state and degraded
// ---------------------------------------------------------------------------

describe('searchContext — dense/rerank state', () => {
  it('passes through dense_state and rerank_state strings', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_state: 'warm',
          rerank_state: 'completed',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.dense_state).toBe('warm');
    expect(result.rerank_state).toBe('completed');
  });

  it('defaults dense_state to "none" when not a string', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_state: null,
          rerank_state: 123,
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.dense_state).toBe('none');
    expect(result.rerank_state).toBe('not_requested');
  });

  it('includes dense_degraded when search response has it', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.dense_degraded).toBe(true);
  });

  it('uses guard guidance when dense_degraded and the guard returns guidanceFields', async () => {
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'started',
      guidanceFields: {
        state: 'model-only',
        reason: 'embeddings missing',
        action: 'started',
        attempt: 1,
        max_attempts: 3,
        cooldown_s: 600,
        next_allowed_at: 0,
        est_duration_min: 1,
        manual_recovery: null,
        guidance: 'self-heal guidance text',
      },
    });

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(result.self_heal).toEqual(
      expect.objectContaining({
        state: 'model-only',
        action: 'started',
        guidance: 'self-heal guidance text',
      }),
    );
    expect(mockInvalidateDenseReadinessCache).toHaveBeenCalled();
  });

  it('falls back to deterministic guidance when the guard yields no guidanceFields', async () => {
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'noop',
    });

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(result.self_heal).toEqual(
      expect.objectContaining({
        state: 'model-only',
        action: 'started',
        guidance: expect.stringMatching(/self-heal/i),
      }),
    );
  });

  it('falls back to deterministic guidance when the guard throws', async () => {
    mockEvaluateSelfHeal.mockRejectedValue(new Error('guard failure'));

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(result.self_heal).toEqual(
      expect.objectContaining({
        state: 'model-only',
        action: 'started',
        guidance: expect.stringMatching(/self-heal/i),
      }),
    );
  });

  it('invokes the guard probe and skips cache invalidation when action is not started', async () => {
    mockEvaluateSelfHeal.mockImplementation(async ({ probe }) => {
      const report = await probe();
      return {
        action: 'noop',
        guidanceFields: {
          state: report.state,
          reason: report.reason,
          action: 'noop',
          attempt: 1,
          max_attempts: 3,
          cooldown_s: 600,
          next_allowed_at: 0,
          est_duration_min: 1,
          manual_recovery: null,
          guidance: 'self-heal guidance text',
        },
      };
    });

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(mockEvaluateSelfHeal).toHaveBeenCalledTimes(1);
    expect(result.self_heal).toEqual(
      expect.objectContaining({
        state: 'model-only',
        action: 'noop',
        guidance: 'self-heal guidance text',
      }),
    );
    expect(mockInvalidateDenseReadinessCache).not.toHaveBeenCalled();
  });

  it('falls back to the default searchCorpus export when no searchCorpusFn is provided', async () => {
    const { searchCorpus } = await import('../tools/search-corpus.mjs');
    searchCorpus.mockResolvedValue(fakeSearchResponse([fakeChunk(1)]));

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(searchCorpus).toHaveBeenCalledWith(
      expect.objectContaining({ query: 'q' }),
    );
    expect(result.results).toHaveLength(1);
  });

  it('omits dense_degraded when not set', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.dense_degraded).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// searchContext — freshness
// ---------------------------------------------------------------------------

describe('searchContext — freshness', () => {
  it('uses searchResponse.freshness when present', async () => {
    const { searchContext } = await loadModule();
    const fresh = { timestamp: 123, stale: false, last_update_source: 'test' };
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], { freshness: fresh }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.freshness).toEqual(fresh);
  });

  it('falls back to buildResponseFreshness when not present', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      databasePath: ':memory:',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.freshness).toEqual(
      expect.objectContaining({
        timestamp: expect.any(Number),
        stale: expect.any(Boolean),
        last_update_source: expect.any(String),
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// searchContext — option forwarding
// ---------------------------------------------------------------------------

describe('searchContext — option forwarding', () => {
  it('forwards use_dense, use_rerank, alpha, expand_query, slice_id, step_number to searchCorpus', async () => {
    const { searchContext } = await loadModule();
    let captured;
    await searchContext({
      query: 'q',
      use_dense: true,
      use_rerank: true,
      alpha: 0.5,
      expand_query: 'domain-only',
      rerank_candidates_count: 20,
      slice_id: 'A1',
      step_number: 2,
      searchCorpusFn: async (opts) => {
        captured = opts;
        return fakeSearchResponse([fakeChunk(1)]);
      },
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(captured.use_dense).toBe(true);
    expect(captured.use_rerank).toBe(true);
    expect(captured.alpha).toBe(0.5);
    expect(captured.expand_query).toBe('domain-only');
    expect(captured.rerank_candidates_count).toBe(20);
    expect(captured.slice_id).toBe('A1');
    expect(captured.step_number).toBe(2);
  });

  it('defaults use_dense to true and use_rerank to false', async () => {
    const { searchContext } = await loadModule();
    let captured;
    await searchContext({
      query: 'q',
      searchCorpusFn: async (opts) => {
        captured = opts;
        return fakeSearchResponse([fakeChunk(1)]);
      },
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(captured.use_dense).toBe(true);
    expect(captured.use_rerank).toBe(false);
  });

  it('passes query_class and client to assembleContext', async () => {
    const { searchContext } = await loadModule();
    let captured;
    const fakeClient = { execute: async () => ({ rows: [] }) };
    await searchContext({
      query: 'q',
      client: fakeClient,
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], { query_class: 'symbol' }),
      assembleContextFn: async (chunks, opts) => {
        captured = opts;
        return fakeAssembleResponse();
      },
    });
    expect(captured.query_class).toBe('symbol');
    expect(captured.client).toBe(fakeClient);
  });
});

// ---------------------------------------------------------------------------
// searchContext — edge cases
// ---------------------------------------------------------------------------

describe('searchContext — edge cases', () => {
  it('handles empty search results', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () => fakeSearchResponse([]),
      assembleContextFn: async () =>
        fakeAssembleResponse({
          selectedChunks: [],
          tokenCount: 0,
          tierCounts: { essential: 0, supporting: 0, supplementary: 0 },
        }),
    });
    expect(result.total_chunks_retrieved).toBe(0);
    expect(result.chunks_in_context).toBe(0);
    expect(result.results).toHaveLength(0);
    expect(result.metadata.truncated).toBe(false);
  });

  it('handles assembled context without tierCounts (defaults to zeros)', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => ({
        context: 'ctx',
        tokenCount: 5,
        selectedChunks: [{ chunk_id: 1, truncated: false }],
      }),
    });
    expect(result.tier_counts).toEqual({
      essential: 0,
      supporting: 0,
      supplementary: 0,
    });
  });

  it('handles assembled context without tokenCount (defaults to 0)', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => ({
        context: 'ctx',
        selectedChunks: [{ chunk_id: 1, truncated: false }],
      }),
    });
    expect(result.token_count).toBe(0);
    expect(result.budget_remaining).toBe(1024);
  });

  it('slices selectedChunks to limit', async () => {
    const { searchContext } = await loadModule();
    const chunks = Array.from({ length: 10 }, (_, i) => ({
      chunk_id: i + 1,
      truncated: false,
      feedback_boost: 0,
    }));
    const result = await searchContext({
      query: 'q',
      limit: 3,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({ selectedChunks: chunks }),
    });
    expect(result.results).toHaveLength(3);
    expect(result.chunks_in_context).toBe(3);
  });

  it('budget_remaining is zero when tokens exceed budget', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      budget: 5,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => fakeAssembleResponse({ tokenCount: 100 }),
    });
    expect(result.budget_remaining).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// searchContextTool alias
// ---------------------------------------------------------------------------

describe('searchContextTool', () => {
  it('is an alias for searchContext', async () => {
    const mod = await loadModule();
    expect(mod.searchContextTool).toBe(mod.searchContext);
  });
});

// ---------------------------------------------------------------------------
// Branch coverage — default params, null fallbacks, real fn paths
// ---------------------------------------------------------------------------

describe('searchContext — branch coverage', () => {
  it('handles being called with no arguments (default options = {})', async () => {
    const { searchContext } = await loadModule();
    await expect(searchContext()).rejects.toThrow(/query/);
  });

  it('handles searchResponse.results being undefined', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () => ({ query_class: 'default' }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });
    expect(result.total_chunks_retrieved).toBe(0);
  });

  it('handles selectedChunks being undefined in assembled result', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => ({
        context: 'ctx',
        tokenCount: 5,
      }),
    });
    expect(result.results).toHaveLength(0);
    expect(result.chunks_in_context).toBe(0);
  });

  it('includes metadata with null fallbacks when chunk properties are missing', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      include_metadata: true,
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({
          selectedChunks: [
            {
              chunk_id: 1,
              truncated: false,
              feedback_boost: 0,
            },
          ],
        }),
    });
    expect(result.results[0].metadata).toEqual({
      file_path: null,
      family: null,
      chunk_index: null,
      heading_path: null,
      depth: null,
      parent_chunk_id: null,
      context_header: null,
      symbol_name: null,
      signature_text: null,
      jsdoc_text: null,
      export_type: null,
      module_path: null,
      arch_layer: null,
      jsdoc_quality: null,
      jsdoc_word_count: null,
      cyclomatic_complexity: null,
      test_coverage: null,
      source_path_pattern: null,
    });
  });

  it('JSON context handles missing chunk properties with null fallbacks', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      context_format: 'json',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () =>
        fakeAssembleResponse({
          context: 'ctx',
          selectedChunks: [
            {
              file_path: 'src/a.ts',
            },
          ],
        }),
    });
    const jsonChunk = result.context.chunks[0];
    expect(jsonChunk.chunk_id).toBeNull();
    expect(jsonChunk.file_path).toBe('src/a.ts');
    expect(jsonChunk.heading_path).toBeNull();
    expect(jsonChunk.char_start).toBeNull();
    expect(jsonChunk.char_end).toBeNull();
    expect(jsonChunk.content).toBe('');
    expect(jsonChunk.score).toBeNull();
    expect(jsonChunk.tier).toBeNull();
  });

  it('uses real searchCorpus and assembleContext fallbacks with a schema client', async () => {
    const { createSchemaClient, insertTestFixtures } =
      await import('./turso-test-helpers.mjs');
    const client = await createSchemaClient();
    try {
      await insertTestFixtures(client);
      const { searchContext } = await import('../tools/search-context.mjs');
      const result = await searchContext({
        query: 'tursouniqueword',
        client,
        use_dense: false,
        limit: 5,
        budget: 1024,
        searchCorpusFn: async ({ query: q }) => ({
          results: [
            {
              chunk_id: 'schema-chunk-1',
              document_id: 'schema-doc',
              title: 'Schema Doc',
              content: `${q} content`,
              metadata: { source: 'schema' },
            },
          ],
          dense_state: 'bypassed',
          dense_degraded: true,
          dense_reason: 'schema-test',
        }),
      });
      expect(result.context_format).toBe('markdown');
      expect(typeof result.context).toBe('string');
      expect(result.dense_state).toEqual(expect.any(String));
      expect(result.freshness).toEqual(expect.any(Object));
      expect(result.results).toBeDefined();
    } finally {
      client.close?.();
    }
  });

  it('uses real assembleContext with a custom searchCorpusFn', async () => {
    const { createSchemaClient, insertTestFixtures, TEST_CHUNK_ID } =
      await import('./turso-test-helpers.mjs');
    const client = await createSchemaClient();
    try {
      await insertTestFixtures(client);
      const { searchContext } = await loadModule();
      const result = await searchContext({
        query: 'q',
        client,
        searchCorpusFn: async () =>
          fakeSearchResponse([
            {
              chunk_id: TEST_CHUNK_ID,
              file_path: 'src/turso-async-test.ts',
              family: 'turso-test',
              doc_family: 'turso-test',
              chunk_index: 1,
              heading_path: 'Mod',
              char_start: 30,
              char_end: 60,
              body_text: 'tursouniqueword sub body content',
              text: 'tursouniqueword sub body content',
              score: 0.9,
            },
          ]),
      });
      expect(result.token_count).toBeGreaterThan(0);
    } finally {
      client.close?.();
    }
  });
});

// ---------------------------------------------------------------------------
// searchContext — self-heal text prepend
// ---------------------------------------------------------------------------

describe('searchContext — self-heal text prepend', () => {
  it('prepends the human guidance paragraph to markdown context when degraded', async () => {
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'noop',
      guidanceFields: {
        state: 'model-only',
        reason: 'embeddings missing',
        action: 'noop',
        attempt: 1,
        max_attempts: 3,
        cooldown_s: 600,
        next_allowed_at: 0,
        est_duration_min: 1,
        manual_recovery: null,
        guidance: 'self-heal guidance text',
      },
    });

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(typeof result.context).toBe('string');
    expect(result.context.startsWith('self-heal guidance text\n\n')).toBe(true);
    expect(result.context).toContain('## Context');
    expect(result.self_heal).toEqual(
      expect.objectContaining({ guidance: 'self-heal guidance text' }),
    );
  });

  it('does not prepend guidance when no self_heal block is present', async () => {
    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () => fakeSearchResponse([fakeChunk(1)]),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(result.context).toBe('## Context\n\nBody text here.');
    expect(result.self_heal).toBeUndefined();
  });

  it('does not duplicate guidance when the context already starts with it', async () => {
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'noop',
      guidanceFields: {
        state: 'model-only',
        reason: 'embeddings missing',
        action: 'noop',
        attempt: 1,
        max_attempts: 3,
        cooldown_s: 600,
        next_allowed_at: 0,
        est_duration_min: 1,
        manual_recovery: null,
        guidance: 'self-heal guidance text',
      },
    });

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () =>
        fakeAssembleResponse({
          context: 'self-heal guidance text\n\n## Context\n\nBody text here.',
        }),
    });

    expect(result.context).toBe(
      'self-heal guidance text\n\n## Context\n\nBody text here.',
    );
  });

  it('does not prepend guidance to a JSON context object', async () => {
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'noop',
      guidanceFields: {
        state: 'model-only',
        reason: 'embeddings missing',
        action: 'noop',
        attempt: 1,
        max_attempts: 3,
        cooldown_s: 600,
        next_allowed_at: 0,
        est_duration_min: 1,
        manual_recovery: null,
        guidance: 'self-heal guidance text',
      },
    });

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      context_format: 'json',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(typeof result.context).toBe('object');
    expect(result.context.context).toBe('## Context\n\nBody text here.');
    expect(result.self_heal).toEqual(
      expect.objectContaining({ guidance: 'self-heal guidance text' }),
    );
  });

  it('keeps context unchanged when self_heal guidance is not a string', async () => {
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'noop',
      guidanceFields: {
        state: 'model-only',
        reason: 'embeddings missing',
        action: 'noop',
        attempt: 1,
        max_attempts: 3,
        cooldown_s: 600,
        next_allowed_at: 0,
        est_duration_min: 1,
        manual_recovery: null,
      },
    });

    const { searchContext } = await loadModule();
    const result = await searchContext({
      query: 'q',
      searchCorpusFn: async () =>
        fakeSearchResponse([fakeChunk(1)], {
          dense_degraded: true,
          dense_state: 'model-only',
          dense_reason: 'embeddings missing',
        }),
      assembleContextFn: async () => fakeAssembleResponse(),
    });

    expect(result.context).toBe('## Context\n\nBody text here.');
    expect(result.self_heal).toEqual(
      expect.objectContaining({ state: 'model-only' }),
    );
  });
});
