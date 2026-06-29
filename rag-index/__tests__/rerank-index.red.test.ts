/**
 * @module rerank-index.red.test
 * @description Red tests for the cross-encoder re-ranking pipeline.
 *
 * Uses runModuleEvaluation to call .mjs module functions from .ts test files,
 * matching the established pattern in classify-query.red.test.ts.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Types for rerank results
// ---------------------------------------------------------------------------

interface RerankCandidate {
  chunk_id: number;
  file_path: string;
  family: string;
  heading_path: string | null;
  body_text: string;
  score: number;
  rerank_score: number;
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
// createRerankInput — tokenization contract
// ---------------------------------------------------------------------------

describe('createRerankInput — tokenization contract', () => {
  it('produces input_ids, attention_mask, and token_type_ids arrays of equal length', () => {
    const result = runModuleEvaluation<{
      inputIdsLength: number;
      attentionMaskLength: number;
      tokenTypeIdsLength: number;
    }>(`
      import { createRerankInput } from './rag-index/rerank-index.mjs';
      import { Tokenizer } from '@huggingface/tokenizers';
      import path from 'node:path';
      import { readFile } from 'node:fs/promises';

      // For this test, we only verify the function signature exists and accepts the right params
      // Full integration test requires the actual model tokenizer
      const input = createRerankInput('test query', 'test document', {
        input_ids: [101, 2342, 3456, 102, 4567, 5678, 102],
        attention_mask: [1, 1, 1, 1, 1, 1, 1],
        token_type_ids: [0, 0, 0, 0, 1, 1, 1],
      }, 512);
      console.log(JSON.stringify({
        inputIdsLength: input.input_ids.length,
        attentionMaskLength: input.attention_mask.length,
        tokenTypeIdsLength: input.token_type_ids.length,
      }));
    `);
    expect(result.inputIdsLength).toBe(result.attentionMaskLength);
    expect(result.inputIdsLength).toBe(result.tokenTypeIdsLength);
  });

  it('truncates to max_sequence_length when provided', () => {
    const result = runModuleEvaluation<{
      inputIdsLength: number;
      maxLength: number;
    }>(`
      import { createRerankInput } from './rag-index/rerank-index.mjs';
      const input = createRerankInput('test query', 'test document', {
        input_ids: new Array(600).fill(0).map((_, i) => i),
        attention_mask: new Array(600).fill(1),
        token_type_ids: new Array(600).fill(0),
      }, 128);
      console.log(JSON.stringify({ inputIdsLength: input.input_ids.length, maxLength: 128 }));
    `);
    expect(result.inputIdsLength).toBeLessThanOrEqual(result.maxLength);
  });
});

// ---------------------------------------------------------------------------
// normalizeRerankCandidates — parameter normalization
// ---------------------------------------------------------------------------

describe('normalizeRerankCandidates — parameter normalization', () => {
  it('defaults to 50 when no value is provided', () => {
    const result = runModuleEvaluation<{ candidates: number }>(`
      import { normalizeRerankCandidates } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ candidates: normalizeRerankCandidates() }));
    `);
    expect(result.candidates).toBe(50);
  });

  it('clamps minimum to 1', () => {
    const result = runModuleEvaluation<{ candidates: number }>(`
      import { normalizeRerankCandidates } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ candidates: normalizeRerankCandidates(0) }));
    `);
    expect(result.candidates).toBe(1);
  });

  it('clamps negative values to 1', () => {
    const result = runModuleEvaluation<{ candidates: number }>(`
      import { normalizeRerankCandidates } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ candidates: normalizeRerankCandidates(-5) }));
    `);
    expect(result.candidates).toBe(1);
  });

  it('clamps maximum to 200', () => {
    const result = runModuleEvaluation<{ candidates: number }>(`
      import { normalizeRerankCandidates } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ candidates: normalizeRerankCandidates(500) }));
    `);
    expect(result.candidates).toBe(200);
  });

  it('returns valid values unchanged', () => {
    const result = runModuleEvaluation<{ candidates: number }>(`
      import { normalizeRerankCandidates } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ candidates: normalizeRerankCandidates(25) }));
    `);
    expect(result.candidates).toBe(25);
  });

  it('coerces non-integer values to integers', () => {
    const result = runModuleEvaluation<{ candidates: number }>(`
      import { normalizeRerankCandidates } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ candidates: normalizeRerankCandidates(30.7) }));
    `);
    expect(result.candidates).toBe(30);
  });
});

// ---------------------------------------------------------------------------
// softmax — output interpretation
// ---------------------------------------------------------------------------

describe('softmax — output interpretation', () => {
  it('converts logits to probabilities that sum to 1', () => {
    const result = runModuleEvaluation<{
      probRelevant: number;
      probNotRelevant: number;
      sum: number;
    }>(`
      import { softmax } from './rag-index/rerank-index.mjs';
      const probs = softmax([2.0, 5.0]);
      console.log(JSON.stringify({
        probRelevant: probs[1],
        probNotRelevant: probs[0],
        sum: probs[0] + probs[1],
      }));
    `);
    expect(result.probRelevant).toBeGreaterThan(result.probNotRelevant);
    expect(result.sum).toBeCloseTo(1.0, 5);
  });

  it('handles equal logits producing near-equal probabilities', () => {
    const result = runModuleEvaluation<{
      probRelevant: number;
      probNotRelevant: number;
    }>(`
      import { softmax } from './rag-index/rerank-index.mjs';
      const probs = softmax([1.0, 1.0]);
      console.log(JSON.stringify({
        probRelevant: probs[1],
        probNotRelevant: probs[0],
      }));
    `);
    expect(result.probRelevant).toBeCloseTo(result.probNotRelevant, 5);
  });

  it('produces high probability for strongly relevant logits', () => {
    const result = runModuleEvaluation<{ probRelevant: number }>(`
      import { softmax } from './rag-index/rerank-index.mjs';
      const probs = softmax([-5.0, 10.0]);
      console.log(JSON.stringify({ probRelevant: probs[1] }));
    `);
    expect(result.probRelevant).toBeGreaterThan(0.99);
  });
});

// ---------------------------------------------------------------------------
// scorePair — relevance scoring contract
// ---------------------------------------------------------------------------

describe('scorePair — relevance scoring contract', () => {
  it('returns a number between 0 and 1', () => {
    // This test requires the mock scoring function; actual ONNX inference
    // is tested in integration. Here we verify the scoring helper normalizes.
    const result = runModuleEvaluation<{ score: number }>(`
      import { scorePairFromLogits } from './rag-index/rerank-index.mjs';
      const score = scorePairFromLogits([0.5, 2.5]);
      console.log(JSON.stringify({ score }));
    `);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
  });

  it('gives higher score for relevant-biased logits', () => {
    const relevantResult = runModuleEvaluation<{ score: number }>(`
      import { scorePairFromLogits } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ score: scorePairFromLogits([-2.0, 5.0]) }));
    `);
    const irrelevantResult = runModuleEvaluation<{ score: number }>(`
      import { scorePairFromLogits } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ score: scorePairFromLogits([5.0, -2.0]) }));
    `);
    expect(relevantResult.score).toBeGreaterThan(irrelevantResult.score);
  });
});

// ---------------------------------------------------------------------------
// DEFAULT constants
// ---------------------------------------------------------------------------

describe('rerank-index constants', () => {
  it('exports DEFAULT_RERANKER_CANDIDATES_COUNT as 50', () => {
    const result = runModuleEvaluation<{ count: number }>(`
      import { DEFAULT_RERANKER_CANDIDATES_COUNT } from './rag-index/rerank-index.mjs';
      console.log(JSON.stringify({ count: DEFAULT_RERANKER_CANDIDATES_COUNT }));
    `);
    expect(result.count).toBe(50);
  });
});

// ---------------------------------------------------------------------------
// releaseRerankSession — session lifecycle
// ---------------------------------------------------------------------------

describe('releaseRerankSession — session lifecycle', () => {
  it('does not throw when called without an active session', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { releaseRerankSession } from './rag-index/rerank-index.mjs';
      await releaseRerankSession();
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });
});
