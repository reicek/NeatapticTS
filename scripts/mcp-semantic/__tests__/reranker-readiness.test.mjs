/**
 * @module reranker-readiness.test
 * @description Coverage tests for rag-index/reranker-readiness.mjs — reranker readiness probe.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { mkdtemp, writeFile, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';

// Mock onnxruntime-node to prevent native bindings from loading under
// Jest ESM VM modules, which causes a native crash during cleanup.
jest.unstable_mockModule('onnxruntime-node', () => ({
  InferenceSession: {
    create: async () => {
      throw new Error(
        'Mocked: onnxruntime-node is not available in test environment',
      );
    },
  },
  Tensor: class MockTensor {},
  __esModule: true,
}));

const {
  checkRerankerReadiness,
  DEFAULT_RERANKER_MODEL_ID,
  DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH,
  DEFAULT_RERANKER_MODEL_DIRECTORY,
} = await import('../../../rag-index/reranker-readiness.mjs');

let tempDir;

beforeEach(async () => {
  tempDir = await mkdtemp(path.join(tmpdir(), 'reranker-test-'));
});

afterEach(async () => {
  await rm(tempDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe('reranker-readiness: constants', () => {
  it('exports DEFAULT_RERANKER_MODEL_ID', () => {
    expect(DEFAULT_RERANKER_MODEL_ID).toBe(
      'cross-encoder/ms-marco-MiniLM-L-6-v2',
    );
  });

  it('exports DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH', () => {
    expect(DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH).toBe(512);
  });

  it('exports DEFAULT_RERANKER_MODEL_DIRECTORY as a string path', () => {
    expect(typeof DEFAULT_RERANKER_MODEL_DIRECTORY).toBe('string');
    expect(DEFAULT_RERANKER_MODEL_DIRECTORY.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// checkRerankerReadiness with forced states
// ---------------------------------------------------------------------------

describe('reranker-readiness: forced states', () => {
  it('returns cold state when forceState is cold', async () => {
    const report = await checkRerankerReadiness({ forceState: 'cold' });
    expect(report.ready).toBe(false);
    expect(report.state).toBe('cold');
    expect(report.reason).toContain('forced cold');
    expect(report.model_id).toBe(null);
    expect(report.max_sequence_length).toBe(null);
  });

  it('returns model-only state when forceState is model-only', async () => {
    const report = await checkRerankerReadiness({ forceState: 'model-only' });
    expect(report.ready).toBe(false);
    expect(report.state).toBe('model-only');
    expect(report.reason).toContain('forced model-only');
  });

  it('returns cold state from RERANKER_FORCE_STATE env var', async () => {
    const saved = process.env.RERANKER_FORCE_STATE;
    process.env.RERANKER_FORCE_STATE = 'cold';
    try {
      const report = await checkRerankerReadiness();
      expect(report.state).toBe('cold');
    } finally {
      if (saved === undefined) delete process.env.RERANKER_FORCE_STATE;
      else process.env.RERANKER_FORCE_STATE = saved;
    }
  });

  it('ignores invalid forced state values', async () => {
    const report = await checkRerankerReadiness({
      forceState: 'invalid-state',
    });
    // Should not be forced — should proceed to actual filesystem check
    expect(report.state).not.toBe('invalid-state');
  });

  it('ignores whitespace-wrapped forced state values after trim', async () => {
    const report = await checkRerankerReadiness({ forceState: '  cold  ' });
    expect(report.state).toBe('cold');
  });

  it('ignores undefined forced state', async () => {
    const report = await checkRerankerReadiness({ forceState: undefined });
    expect(report.state).not.toBe('undefined');
  });
});

// ---------------------------------------------------------------------------
// checkRerankerReadiness cold path (model.onnx absent)
// ---------------------------------------------------------------------------

describe('reranker-readiness: cold path', () => {
  let savedForceState;

  beforeEach(() => {
    savedForceState = process.env.RERANKER_FORCE_STATE;
    delete process.env.RERANKER_FORCE_STATE;
  });

  afterEach(() => {
    if (savedForceState === undefined) delete process.env.RERANKER_FORCE_STATE;
    else process.env.RERANKER_FORCE_STATE = savedForceState;
  });

  it('returns cold when model.onnx does not exist', async () => {
    const emptyDir = path.join(tempDir, 'empty-model');
    await mkdir(emptyDir, { recursive: true });
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: emptyDir,
    });
    expect(report.ready).toBe(false);
    expect(report.state).toBe('cold');
    expect(report.model_id).toBe(null);
    expect(report.reason).toBe('Reranker model assets are absent.');
  });
});

// ---------------------------------------------------------------------------
// checkRerankerReadiness model-only path (model exists but meta missing/invalid)
// ---------------------------------------------------------------------------

describe('reranker-readiness: model-only path', () => {
  let savedForceState;

  beforeEach(() => {
    savedForceState = process.env.RERANKER_FORCE_STATE;
    delete process.env.RERANKER_FORCE_STATE;
  });

  afterEach(() => {
    if (savedForceState === undefined) delete process.env.RERANKER_FORCE_STATE;
    else process.env.RERANKER_FORCE_STATE = savedForceState;
  });

  it('returns model-only when model.onnx exists but model-meta.json is missing', async () => {
    const modelDir = path.join(tempDir, 'model-only');
    await mkdir(modelDir, { recursive: true });
    const onnxPath = path.join(modelDir, 'model.onnx');
    await writeFile(onnxPath, 'fake-onnx');
    expect(existsSync(onnxPath)).toBe(true);
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: modelDir,
    });
    expect(report.ready).toBe(false);
    expect(report.state).toBe('model-only');
    expect(report.reason).toContain('model-meta.json');
  });

  it('returns model-only when model-meta.json is not valid JSON', async () => {
    const modelDir = path.join(tempDir, 'bad-meta');
    await mkdir(modelDir, { recursive: true });
    await writeFile(path.join(modelDir, 'model.onnx'), 'fake-onnx');
    await writeFile(path.join(modelDir, 'model-meta.json'), 'not-json{');
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: modelDir,
    });
    expect(report.ready).toBe(false);
    expect(report.state).toBe('model-only');
  });

  it('returns model-only when model-meta.json is not an object', async () => {
    const modelDir = path.join(tempDir, 'non-object-meta');
    await mkdir(modelDir, { recursive: true });
    await writeFile(path.join(modelDir, 'model.onnx'), 'fake-onnx');
    await writeFile(
      path.join(modelDir, 'model-meta.json'),
      '"string-not-object"',
    );
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: modelDir,
    });
    expect(report.ready).toBe(false);
    expect(report.state).toBe('model-only');
    expect(report.reason).toContain('not a valid object');
  });
});

// ---------------------------------------------------------------------------
// checkRerankerReadiness warm/session-creation path
// ---------------------------------------------------------------------------

describe('reranker-readiness: warm/session path', () => {
  let savedForceState;

  beforeEach(() => {
    savedForceState = process.env.RERANKER_FORCE_STATE;
    delete process.env.RERANKER_FORCE_STATE;
  });

  afterEach(() => {
    if (savedForceState === undefined) delete process.env.RERANKER_FORCE_STATE;
    else process.env.RERANKER_FORCE_STATE = savedForceState;
  });

  it('returns model-only when ONNX session creation fails (no real ONNX model)', async () => {
    const modelDir = path.join(tempDir, 'fake-session');
    await mkdir(modelDir, { recursive: true });
    await writeFile(path.join(modelDir, 'model.onnx'), 'fake-onnx');
    await writeFile(
      path.join(modelDir, 'model-meta.json'),
      JSON.stringify({
        model_id: 'test-model',
        max_sequence_length: 128,
      }),
    );
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: modelDir,
    });
    // Session creation will fail because the ONNX file is fake
    expect(report.ready).toBe(false);
    expect(report.state).toBe('model-only');
    expect(report.model_id).toBe('test-model');
    expect(report.max_sequence_length).toBe(128);
  });

  it('uses default max_sequence_length when meta omits it', async () => {
    const modelDir = path.join(tempDir, 'no-max-seq');
    await mkdir(modelDir, { recursive: true });
    await writeFile(path.join(modelDir, 'model.onnx'), 'fake-onnx');
    await writeFile(
      path.join(modelDir, 'model-meta.json'),
      JSON.stringify({ model_id: 'test-model' }),
    );
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: modelDir,
    });
    expect(report.max_sequence_length).toBe(
      DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH,
    );
  });

  it('uses default model_id when meta omits it', async () => {
    const modelDir = path.join(tempDir, 'no-model-id');
    await mkdir(modelDir, { recursive: true });
    await writeFile(path.join(modelDir, 'model.onnx'), 'fake-onnx');
    await writeFile(
      path.join(modelDir, 'model-meta.json'),
      JSON.stringify({ max_sequence_length: 64 }),
    );
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: modelDir,
      rerankerModelId: 'custom-id',
    });
    expect(report.model_id).toBe('custom-id');
  });
});
