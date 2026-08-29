import { jest } from '@jest/globals';
import { runCortexIndexGate } from '../cortex-index.gate.mjs';
import { resolveFixHint } from '../cortex-index.gate.runtime.mjs';

function makeFamilyFresh(overrides = {}) {
  return {
    readme: { fresh: true, stalePaths: [] },
    'ts-source': { fresh: true, stalePaths: [] },
    plan: { fresh: true, stalePaths: [] },
    demo: { fresh: true, stalePaths: [] },
    benchmark: { fresh: true, stalePaths: [] },
    'root-doc': { fresh: true, stalePaths: [] },
    'completed-plan': { fresh: true, stalePaths: [] },
    ...overrides,
  };
}

function makeDeps(overrides = {}) {
  return {
    databasePath: 'test.sqlite',
    snapshotPath: 'snapshot.json',
    snapshotMaxAgeMs: 86_400_000,
    timeoutMs: 60_000,
    workflowPlanPath: 'plans/test.plans.md',
    indexValidator: jest.fn().mockResolvedValue({
      pass: true,
      documents: 7,
      family_fresh: makeFamilyFresh(),
    }),
    mcpSmoke: jest.fn().mockResolvedValue({ pass: true }),
    workflowMcpCheck: jest.fn().mockResolvedValue({ pass: true }),
    snapshotCurrency: jest.fn().mockResolvedValue({
      pass: true,
      indexDocuments: 7,
      snapshotAgeSeconds: 10,
      snapshotIndexedAt: '2024-01-01T00:00:00.000Z',
    }),
    rebuildIndex: jest.fn().mockResolvedValue({ success: true }),
    resolveFixHint: jest.fn().mockReturnValue('mock fix hint'),
    ...overrides,
  };
}

describe('cortex-index gate per-family freshness', () => {
  test('exposes family_fresh map and index_fresh alias in evidence', async () => {
    const familyFresh = makeFamilyFresh();
    const deps = makeDeps({
      indexValidator: jest.fn().mockResolvedValue({
        pass: true,
        documents: 7,
        family_fresh: familyFresh,
      }),
    });

    const result = await runCortexIndexGate({}, deps);

    expect(result.pass).toBe(true);
    expect(result.evidence.family_fresh).toEqual(familyFresh);
    expect(result.evidence.index_fresh).toBe(true);
  });

  test('completed-plan staleness does not fail the gate', async () => {
    const familyFresh = makeFamilyFresh({
      'completed-plan': {
        fresh: false,
        stalePaths: ['plans/completed.plans.md'],
      },
    });
    const deps = makeDeps({
      indexValidator: jest.fn().mockResolvedValue({
        pass: false,
        documents: 7,
        family_fresh: familyFresh,
      }),
    });

    const result = await runCortexIndexGate({}, deps);

    expect(result.pass).toBe(true);
    expect(result.evidence.index_fresh).toBe(true);
    expect(result.evidence.family_fresh['completed-plan'].fresh).toBe(false);
  });

  test('a gated stale family fails the gate', async () => {
    const familyFresh = makeFamilyFresh({
      plan: { fresh: false, stalePaths: ['plans/active.plans.md'] },
    });
    const deps = makeDeps({
      indexValidator: jest.fn().mockResolvedValue({
        pass: false,
        documents: 7,
        family_fresh: familyFresh,
      }),
    });

    const result = await runCortexIndexGate({}, deps);

    expect(result.pass).toBe(false);
    expect(result.evidence.index_fresh).toBe(false);
    expect(result.evidence.family_fresh.plan.fresh).toBe(false);
    expect(result.evidence.family_fresh.readme.fresh).toBe(true);
  });

  test('missing family_fresh fails closed with missing-manifest fix hint', async () => {
    const deps = makeDeps({
      indexValidator: jest.fn().mockResolvedValue({
        pass: true,
        documents: 7,
      }),
    });

    const result = await runCortexIndexGate({}, deps);

    expect(result.pass).toBe(false);
    expect(result.evidence.family_fresh).toBeNull();
    expect(result.evidence.index_fresh).toBe(false);
    expect(result.fixHint).toBe(
      'freshness manifest missing — run validate-index',
    );
  });
});

describe('cortex-index gate resolveFixHint', () => {
  test('uses stale-plan hint when only plan paths are stale and nothing is missing', () => {
    const stalePaths = ['plans/RAG_Index_Freshness_Strategy.plans.md'];
    const customPlanHint = 'stale plan hint';
    const hint = resolveFixHint(
      {
        indexReport: {
          pass: false,
          missing_paths: [],
          stale_paths: stalePaths,
        },
      },
      {
        resolveStalePlanFixHint: (paths) =>
          `${customPlanHint}: ${paths.join(',')}`,
      },
    );

    expect(hint).toBe(`${customPlanHint}: ${stalePaths.join(',')}`);
  });

  test('returns generic rebuild hint when missing paths exist', () => {
    const hint = resolveFixHint({
      indexReport: {
        pass: false,
        missing_paths: ['src/missing.ts'],
        stale_paths: ['src/foo.ts'],
      },
    });

    expect(hint).toBe(
      'Run: node rag-index/build-index.mjs to rebuild stale index',
    );
  });
});
