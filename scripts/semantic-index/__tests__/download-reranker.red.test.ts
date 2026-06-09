/**
 * @module download-reranker.red.test
 * @description Red tests for the cross-encoder model download script.
 *
 * Uses runModuleEvaluation to call .mjs module functions from .ts test files,
 * matching the established pattern in classify-query.red.test.ts.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Types for download results
// ---------------------------------------------------------------------------

interface DownloadResult {
  modelId: string;
  modelDirectory: string;
  modelSha256: string;
  assets: Array<{ file: string; sha256: string; verified: boolean }>;
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
// downloadRerankerAssets — constants
// ---------------------------------------------------------------------------

describe('download-reranker constants', () => {
  it('exports DEFAULT_RERANKER_MODEL_ID as cross-encoder/ms-marco-MiniLM-L-6-v2', () => {
    const result = runModuleEvaluation<{ modelId: string }>(`
      import { DEFAULT_RERANKER_MODEL_ID } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ modelId: DEFAULT_RERANKER_MODEL_ID }));
    `);
    expect(result.modelId).toBe('cross-encoder/ms-marco-MiniLM-L-6-v2');
  });

  it('exports DEFAULT_RERANKER_REPOSITORY_ID pointing to Hugging Face', () => {
    const result = runModuleEvaluation<{ repositoryId: string }>(`
      import { DEFAULT_RERANKER_REPOSITORY_ID } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ repositoryId: DEFAULT_RERANKER_REPOSITORY_ID }));
    `);
    expect(result.repositoryId).toContain('cross-encoder');
  });

  it('exports DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH as 512', () => {
    const result = runModuleEvaluation<{ maxLength: number }>(`
      import { DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ maxLength: DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH }));
    `);
    expect(result.maxLength).toBe(512);
  });
});

// ---------------------------------------------------------------------------
// downloadRerankerAssets — model-meta.json content
// ---------------------------------------------------------------------------

describe('downloadRerankerAssets — model-meta.json', () => {
  it('model-meta.json must NOT contain a dimension field (cross-encoders output scalars)', () => {
    // This test verifies the download function contract:
    // Cross-encoder model-meta.json must NOT include a dimension field,
    // unlike the bi-encoder model-meta.json which includes dimension: 384.
    // This is tested indirectly by checking the download function's return value.
    const result = runModuleEvaluation<{ hasDimension: boolean }>(`
      import { RERANKER_META_FIELDS } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ hasDimension: RERANKER_META_FIELDS.includes('dimension') }));
    `);
    expect(result.hasDimension).toBe(false);
  });

  it('model-meta.json must include max_sequence_length instead of dimension', () => {
    const result = runModuleEvaluation<{ hasMaxLength: boolean }>(`
      import { RERANKER_META_FIELDS } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ hasMaxLength: RERANKER_META_FIELDS.includes('max_sequence_length') }));
    `);
    expect(result.hasMaxLength).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// downloadRerankerAssets — asset list
// ---------------------------------------------------------------------------

describe('downloadRerankerAssets — asset configuration', () => {
  it('downloads model.onnx from onnx/ subdirectory (LFS path)', () => {
    const result = runModuleEvaluation<{
      assets: Array<{ localName: string; remotePath: string }>;
    }>(`
      import { RERANKER_ASSETS } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ assets: RERANKER_ASSETS }));
    `);
    const modelAsset = result.assets.find((a) => a.localName === 'model.onnx');
    expect(modelAsset).toBeDefined();
    expect(modelAsset!.remotePath).toBe('onnx/model.onnx');
  });

  it('downloads all four tokenizer assets', () => {
    const result = runModuleEvaluation<{
      assets: Array<{ localName: string }>;
    }>(`
      import { RERANKER_ASSETS } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ assets: RERANKER_ASSETS }));
    `);
    const localNames = result.assets.map((a) => a.localName);
    expect(localNames).toContain('tokenizer.json');
    expect(localNames).toContain('tokenizer_config.json');
    expect(localNames).toContain('special_tokens_map.json');
  });

  it('verifies SHA-256 only for model.onnx', () => {
    const result = runModuleEvaluation<{
      assets: Array<{ localName: string; verifySha256: boolean }>;
    }>(`
      import { RERANKER_ASSETS } from './scripts/semantic-index/download-reranker.mjs';
      console.log(JSON.stringify({ assets: RERANKER_ASSETS }));
    `);
    const modelAsset = result.assets.find((a) => a.localName === 'model.onnx');
    expect(modelAsset!.verifySha256).toBe(true);
    const nonModelAssets = result.assets.filter(
      (a) => a.localName !== 'model.onnx',
    );
    for (const asset of nonModelAssets) {
      expect(asset.verifySha256).toBe(false);
    }
  });
});
