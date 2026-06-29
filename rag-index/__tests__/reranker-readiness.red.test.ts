/**
 * @module reranker-readiness.red.test
 * @description Red tests for the cross-encoder reranker readiness probe.
 *
 * Uses runModuleEvaluation to call .mjs module functions from .ts test files,
 * matching the established pattern in classify-query.red.test.ts.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Types for readiness results
// ---------------------------------------------------------------------------

interface ReadinessReport {
  state: 'cold' | 'model-only' | 'warm';
  ready: boolean;
  reason: string;
  model_id?: string | null;
  max_sequence_length?: number | null;
}

// ---------------------------------------------------------------------------
// Helper: evaluate .mjs modules via subprocess (matching established pattern)
// ---------------------------------------------------------------------------

const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: path.resolve(process.cwd()),
      encoding: 'utf8',
    },
  );
  return JSON.parse(output) as Result;
};

// ---------------------------------------------------------------------------
// checkRerankerReadiness — cold state
// ---------------------------------------------------------------------------

describe('checkRerankerReadiness — cold state', () => {
  it('returns cold when model.onnx is absent', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/reranker/model/path',
      });
      console.log(JSON.stringify(report));
    `);
    expect(result.state).toBe('cold');
    expect(result.ready).toBe(false);
  });

  it('returns cold when RERANKER_FORCE_STATE is cold', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/path',
        forceState: 'cold',
      });
      console.log(JSON.stringify(report));
    `);
    expect(result.state).toBe('cold');
    expect(result.ready).toBe(false);
  });

  it('reports absent model assets in reason for cold state', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/reranker/model/path',
      });
      console.log(JSON.stringify(report));
    `);
    expect(result.reason).toContain('absent');
  });
});

// ---------------------------------------------------------------------------
// checkRerankerReadiness — model-only state
// ---------------------------------------------------------------------------

describe('checkRerankerReadiness — model-only state', () => {
  it('returns model-only when RERANKER_FORCE_STATE is model-only', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/path',
        forceState: 'model-only',
      });
      console.log(JSON.stringify(report));
    `);
    expect(result.state).toBe('model-only');
    expect(result.ready).toBe(false);
  });

  it('reports session creation failure in reason for model-only state', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/path',
        forceState: 'model-only',
      });
      console.log(JSON.stringify(report));
    `);
    expect(result.reason).toContain('model-only');
  });
});

// ---------------------------------------------------------------------------
// checkRerankerReadiness — forced state normalization
// ---------------------------------------------------------------------------

describe('checkRerankerReadiness — forced state normalization', () => {
  it('ignores invalid forced state values', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/path',
        forceState: 'invalid-state',
      });
      console.log(JSON.stringify(report));
    `);
    // Invalid forced state should fall through to actual filesystem check
    expect(result.state).toBe('cold');
  });

  it('trims whitespace from forced state values', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/path',
        forceState: '  cold  ',
      });
      console.log(JSON.stringify(report));
    `);
    expect(result.state).toBe('cold');
  });
});

// ---------------------------------------------------------------------------
// checkRerankerReadiness — model-meta validation
// ---------------------------------------------------------------------------

describe('checkRerankerReadiness — model-meta validation', () => {
  it('does not include dimension field in readiness report (cross-encoders output scalars)', () => {
    const result = runModuleEvaluation<ReadinessReport>(`
      import { checkRerankerReadiness } from './rag-index/reranker-readiness.mjs';
      const report = await checkRerankerReadiness({
        rerankerModelDirectory: '/nonexistent/path',
        forceState: 'cold',
      });
      console.log(JSON.stringify(report));
    `);
    // Cross-encoders output a scalar relevance score, not a vector
    expect(result).not.toHaveProperty('dimension');
  });
});

// ---------------------------------------------------------------------------
// DEFAULT constants
// ---------------------------------------------------------------------------

describe('reranker-readiness constants', () => {
  it('exports DEFAULT_RERANKER_MODEL_ID as cross-encoder/ms-marco-MiniLM-L-6-v2', () => {
    const result = runModuleEvaluation<{ modelId: string }>(`
      import { DEFAULT_RERANKER_MODEL_ID } from './rag-index/reranker-readiness.mjs';
      console.log(JSON.stringify({ modelId: DEFAULT_RERANKER_MODEL_ID }));
    `);
    expect(result.modelId).toBe('cross-encoder/ms-marco-MiniLM-L-6-v2');
  });

  it('exports DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH as 512', () => {
    const result = runModuleEvaluation<{ maxLength: number }>(`
      import { DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH } from './rag-index/reranker-readiness.mjs';
      console.log(JSON.stringify({ maxLength: DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH }));
    `);
    expect(result.maxLength).toBe(512);
  });
});
