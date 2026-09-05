/**
 * @module search-corpus.test
 * @description Red tests for response augmentation (self_heal field) and
 * memo invalidation in search-corpus.mjs.
 *
 * These tests exercise the P3-S02-A contract:
 *  - degraded dense-search responses include a populated `self_heal` block
 *    produced by `evaluateSelfHeal` from cortex-health-guard.mjs;
 *  - an exception inside the guard does not fail the search request;
 *  - an exported cache-invalidation seam lets a repaired dense readiness
 *    state be observed without restarting the process.
 */
import { jest } from '@jest/globals';
import {
  createSchemaClient,
  insertTestFixtures,
  UNIQUE_QUERY_TERM,
} from '../__tests__/turso-test-helpers.mjs';

const MODEL_ONLY_GUIDANCE = {
  state: 'model-only',
  reason:
    'Dense embeddings are unavailable because the embeddings database is missing or incomplete.',
  action: 'started',
  attempt: 1,
  max_attempts: 3,
  cooldown_s: 600,
  next_allowed_at: 0,
  est_duration_min: 1,
  manual_recovery: null,
  guidance:
    'Cortex dense search is degraded and a background self-heal repair has been started. ' +
    'BM25-only results are being returned; continue with reduced recall.',
};

const mockEvaluateSelfHeal = jest.fn();

jest.unstable_mockModule(
  '../../agent-customization/cortex/cortex-health-guard.mjs',
  () => ({
    evaluateSelfHeal: mockEvaluateSelfHeal,
    __esModule: true,
  }),
);

const {
  searchCorpus,
  getDenseReadiness,
  resetReadinessCaches,
  invalidateDenseReadinessCache,
} = await import('./search-corpus.mjs');

describe('search-corpus self-heal red tests', () => {
  let client;
  const originalDenseForceState = process.env.DENSE_FORCE_STATE;
  const originalRerankerForceState = process.env.RERANKER_FORCE_STATE;

  beforeEach(async () => {
    resetReadinessCaches();
    mockEvaluateSelfHeal.mockReset();
    client = await createSchemaClient();
    await insertTestFixtures(client);
  });

  afterEach(async () => {
    if (client) {
      try {
        await client.close();
      } catch {
        // best-effort cleanup
      }
    }
    process.env.DENSE_FORCE_STATE = originalDenseForceState;
    process.env.RERANKER_FORCE_STATE = originalRerankerForceState;
  });

  it('augments a degraded model-only response with a self_heal block from the guard', async () => {
    process.env.DENSE_FORCE_STATE = 'model-only';
    process.env.RERANKER_FORCE_STATE = 'cold';
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'started',
      guidanceFields: MODEL_ONLY_GUIDANCE,
      spawnDecision: { pid: 123, command: 'node cortex-self-heal.mjs' },
    });

    const response = await searchCorpus({
      client,
      query: UNIQUE_QUERY_TERM,
      use_dense: true,
      limit: 5,
    });

    expect(mockEvaluateSelfHeal).toHaveBeenCalled();
    expect(response.dense_degraded).toBe(true);
    expect(response.dense_state).toBe('model-only');
    expect(response.self_heal).toEqual(MODEL_ONLY_GUIDANCE);
    expect(response.self_heal.guidance).toMatch(/self-heal/i);
  });

  it('still returns plain BM25 results when the guard throws', async () => {
    process.env.DENSE_FORCE_STATE = 'model-only';
    process.env.RERANKER_FORCE_STATE = 'cold';
    mockEvaluateSelfHeal.mockRejectedValue(new Error('guard decision failed'));

    const response = await searchCorpus({
      client,
      query: UNIQUE_QUERY_TERM,
      use_dense: true,
      limit: 5,
    });

    expect(mockEvaluateSelfHeal).toHaveBeenCalled();
    expect(Array.isArray(response.results)).toBe(true);
    expect(response.results.length).toBeGreaterThan(0);
    expect(response.error).toBeUndefined();
  });

  it('returns plain BM25 results with no self_heal when the guard returns a decision without guidanceFields', async () => {
    process.env.DENSE_FORCE_STATE = 'model-only';
    process.env.RERANKER_FORCE_STATE = 'cold';
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'skipped',
      spawnDecision: { reason: 'cooldown active' },
    });

    const response = await searchCorpus({
      client,
      query: UNIQUE_QUERY_TERM,
      use_dense: true,
      limit: 5,
    });

    expect(mockEvaluateSelfHeal).toHaveBeenCalled();
    expect(Array.isArray(response.results)).toBe(true);
    expect(response.results.length).toBeGreaterThan(0);
    expect(response.self_heal).toBeUndefined();
    expect(response.error).toBeUndefined();
  });

  it('invalidates the dense-readiness cache so a repaired probe is observed without restart', async () => {
    // Remove the environment force so the cache is actually consulted.
    process.env.DENSE_FORCE_STATE = '';
    process.env.RERANKER_FORCE_STATE = '';

    const warmProbe = jest.fn().mockResolvedValue({
      state: 'warm',
      ready: true,
      reason: 'embeddings present',
      chunk_count: 100,
      embedding_count: 100,
    });
    const warmReport = await getDenseReadiness({ readinessProbe: warmProbe });
    expect(warmReport.state).toBe('warm');

    expect(invalidateDenseReadinessCache).toBeInstanceOf(Function);
    invalidateDenseReadinessCache();

    const repairedProbe = jest.fn().mockResolvedValue({
      state: 'model-only',
      ready: false,
      reason: 'embeddings incomplete after repair attempt',
      chunk_count: 100,
      embedding_count: 0,
    });
    const postRepairReport = await getDenseReadiness({
      readinessProbe: repairedProbe,
    });
    expect(postRepairReport.state).toBe('model-only');
  });
});
