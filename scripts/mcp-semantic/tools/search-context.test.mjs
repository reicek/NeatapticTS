/**
 * @module search-context.test
 * @description Red tests for self_heal propagation in search-context.mjs.
 *
 * Exercises the P3-S02-A contract that `searchContext` must either forward
 * the `self_heal` block from the underlying corpus search or, when the corpus
 * response is degraded but lacks one, augment the context response with a
 * guard-derived `self_heal` block.
 */
import { jest } from '@jest/globals';

const mockEvaluateSelfHeal = jest.fn();

jest.unstable_mockModule(
  '../../agent-customization/cortex/cortex-health-guard.mjs',
  () => ({
    evaluateSelfHeal: mockEvaluateSelfHeal,
    __esModule: true,
  }),
);

const { searchContext } = await import('./search-context.mjs');

describe('search-context self-heal red tests', () => {
  const FRESHNESS = {
    timestamp: Date.now(),
    stale: false,
    last_update_source: 'corpus_index',
  };

  function makeCorpusResult(overrides = {}) {
    return {
      chunk_id: 424242,
      file_path: 'src/turso-async-test.ts',
      family: 'turso-test',
      chunk_index: 0,
      heading_path: 'TursoAsyncTestModule.TursoAsyncMethod',
      body_text: 'turso test result body',
      char_start: 0,
      char_end: 24,
      parent_chunk_id: null,
      depth: 0,
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
      score: 1.0,
      feedback_boost: 0,
      feedback_signals: {
        total_positive: 0,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      },
      ...overrides,
    };
  }

  function makeSearchCorpusResponse(overrides = {}) {
    return {
      query: 'turso test',
      limit: 5,
      use_dense: false,
      query_class: 'simple_lookup',
      confidence: 0.8,
      classification_fallback: false,
      family_fallback: false,
      results: [makeCorpusResult()],
      freshness: FRESHNESS,
      ...overrides,
    };
  }

  function makeAssembleResponse(overrides = {}) {
    return {
      context: '# Turso test context',
      tokenCount: 10,
      tierCounts: { essential: 1, supporting: 0, supplementary: 0 },
      selectedChunks: [makeCorpusResult()],
      ...overrides,
    };
  }

  it('forwards a self_heal block from the underlying corpus search', async () => {
    const selfHeal = {
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
    };
    const searchCorpusFn = jest
      .fn()
      .mockResolvedValue(makeSearchCorpusResponse({ self_heal: selfHeal }));
    const assembleContextFn = jest
      .fn()
      .mockResolvedValue(makeAssembleResponse());

    const response = await searchContext({
      query: 'turso test',
      searchCorpusFn,
      assembleContextFn,
    });

    expect(response.self_heal).toEqual(selfHeal);
    expect(response.self_heal.action).toBe('started');
  });

  it('augments a self_heal block when the corpus response is degraded without one', async () => {
    const searchCorpusFn = jest.fn().mockResolvedValue(
      makeSearchCorpusResponse({
        dense_degraded: true,
        dense_state: 'model-only',
        dense_reason: 'embeddings missing',
      }),
    );
    const assembleContextFn = jest
      .fn()
      .mockResolvedValue(makeAssembleResponse());

    const response = await searchContext({
      query: 'turso test',
      searchCorpusFn,
      assembleContextFn,
    });

    expect(response.self_heal).toBeDefined();
    expect(response.self_heal.action).toMatch(
      /started|in_flight|cooldown|exhausted|disabled/,
    );
  });
});
