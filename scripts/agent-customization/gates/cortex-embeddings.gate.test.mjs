/**
 * @module cortex-embeddings.gate.test
 * @description Coverage tests for cortex-embeddings.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/cortex-embeddings.gate.mjs',
);

let mockExistsSync;
let mockValidationReport;
let mockEvaluationReport;

jest.unstable_mockModule('node:fs', () => ({
  existsSync: (...args) => mockExistsSync(...args),
}));
jest.unstable_mockModule('../../../rag-index/validate-embeddings.mjs', () => ({
  validateEmbeddings: async () => mockValidationReport,
}));
jest.unstable_mockModule('../../../rag-index/embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/mock/models',
}));
jest.unstable_mockModule('../../../rag-index/eval-embeddings.mjs', () => ({
  evaluateEmbeddings: async () => mockEvaluationReport,
}));

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function importGateMain(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    await withArgv([process.execPath, GATE_PATH, ...argv], async () => {
      await import('./cortex-embeddings.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('cortex-embeddings gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockExistsSync = () => true;
    mockValidationReport = { evidence: [] };
    mockEvaluationReport = {
      bm25MrrAt5: 0.5,
      chunkCount: 10,
      embeddingCount: 10,
      hybridMrrAt5: 0.6,
    };
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = 0;
  });

  describe('evaluateCortexEmbeddingsGate', () => {
    it('returns pass=true when embeddings present and hybrid improves', async () => {
      const { evaluateCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = evaluateCortexEmbeddingsGate({
        chunkCount: 10,
        embeddingCount: 10,
        bm25MrrAt5: 0.5,
        hybridMrrAt5: 0.6,
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.length, 0);
      assert.equal(result.fixHint, null);
      assert.equal(result.owner, '05-green-testing');
    });

    it('flags no usable embeddings when embeddingCount is 0', async () => {
      const { evaluateCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = evaluateCortexEmbeddingsGate({
        chunkCount: 8,
        embeddingCount: 0,
        bm25MrrAt5: 0.5,
        hybridMrrAt5: 0.6,
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.length, 1);
      assert.equal(result.evidence[0].issue, 'no usable embeddings for model');
      assert.equal(result.evidence[0].expected, 8);
    });

    it('flags hybrid MRR below threshold', async () => {
      const { evaluateCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = evaluateCortexEmbeddingsGate({
        chunkCount: 10,
        embeddingCount: 10,
        bm25MrrAt5: 0.5,
        hybridMrrAt5: 0.51,
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.length, 1);
      assert.equal(result.evidence[0].issue, 'hybrid MRR@5 improvement below threshold');
      assert.equal(result.evidence[0].minHybridImprovement, 0.02);
    });

    it('flags both issues at once', async () => {
      const { evaluateCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = evaluateCortexEmbeddingsGate({
        chunkCount: 4,
        embeddingCount: 0,
        bm25MrrAt5: 0.5,
        hybridMrrAt5: 0.5,
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.length, 2);
    });

    it('respects custom minHybridImprovement that passes', async () => {
      const { evaluateCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = evaluateCortexEmbeddingsGate({
        chunkCount: 10,
        embeddingCount: 10,
        bm25MrrAt5: 0.5,
        hybridMrrAt5: 0.55,
        minHybridImprovement: 0.04,
      });
      assert.equal(result.pass, true);
    });

    it('uses defaults when options absent', async () => {
      const { evaluateCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = evaluateCortexEmbeddingsGate();
      assert.equal(result.pass, false);
      assert.equal(result.evidence.length, 2);
    });
  });

  describe('runCortexEmbeddingsGate', () => {
    it('returns pass=true when assets present and validation/eval clean', async () => {
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      mockExistsSync = () => true;
      const result = await runCortexEmbeddingsGate({
        evaluationReport: {
          chunkCount: 10,
          embeddingCount: 10,
          bm25MrrAt5: 0.5,
          hybridMrrAt5: 0.6,
        },
      });
      assert.equal(result.pass, true);
    });

    it('uses default options when called with no args', async () => {
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      mockExistsSync = () => true;
      const result = await runCortexEmbeddingsGate();
      assert.equal(result.pass, true);
    });

    it('flags model assets missing when both files absent', async () => {
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      mockExistsSync = () => false;
      const result = await runCortexEmbeddingsGate({
        evaluationReport: {
          chunkCount: 10,
          embeddingCount: 10,
          bm25MrrAt5: 0.5,
          hybridMrrAt5: 0.6,
        },
      });
      assert.equal(result.pass, false);
      assert.ok(
        result.evidence.some((e) => e.issue === 'model assets missing'),
      );
    });

    it('flags model assets missing when only onnx absent (evaluates both existsSync calls)', async () => {
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      mockExistsSync = (p) => p.endsWith('model-meta.json');
      const result = await runCortexEmbeddingsGate({
        evaluationReport: {
          chunkCount: 10,
          embeddingCount: 10,
          bm25MrrAt5: 0.5,
          hybridMrrAt5: 0.6,
        },
      });
      assert.equal(result.pass, false);
      assert.ok(
        result.evidence.some((e) => e.issue === 'model assets missing'),
      );
    });

    it('catches Error thrown by validateEmbeddings', async () => {
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      jest.resetModules();
      jest.unstable_mockModule('../../../rag-index/validate-embeddings.mjs', () => ({
        validateEmbeddings: async () => {
          throw new Error('boom');
        },
      }));
      const { runCortexEmbeddingsGate: run } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = await run({
        evaluationReport: {
          chunkCount: 10,
          embeddingCount: 10,
          bm25MrrAt5: 0.5,
          hybridMrrAt5: 0.6,
        },
      });
      assert.equal(result.pass, false);
      const errEv = result.evidence.find((e) => e.issue === 'gate execution error');
      assert.ok(errEv);
      assert.equal(errEv.message, 'boom');
    });

    it('catches non-Error throw and stringifies it', async () => {
      jest.resetModules();
      jest.unstable_mockModule('../../../rag-index/validate-embeddings.mjs', () => ({
        validateEmbeddings: async () => {
          throw 'string failure';
        },
      }));
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      const result = await runCortexEmbeddingsGate({
        evaluationReport: {
          chunkCount: 10,
          embeddingCount: 10,
          bm25MrrAt5: 0.5,
          hybridMrrAt5: 0.6,
        },
      });
      assert.equal(result.pass, false);
      const errEv = result.evidence.find((e) => e.issue === 'gate execution error');
      assert.ok(errEv);
      assert.equal(errEv.message, 'string failure');
    });

    it('loads evaluationReport via dynamic import when not provided', async () => {
      jest.resetModules();
      jest.unstable_mockModule('../../../rag-index/validate-embeddings.mjs', () => ({
        validateEmbeddings: async () => ({ evidence: [] }),
      }));
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      mockExistsSync = () => true;
      const result = await runCortexEmbeddingsGate({});
      assert.equal(result.pass, true);
    });

    it('respects custom modelDirectory option', async () => {
      const { runCortexEmbeddingsGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-embeddings.gate.mjs'),
      );
      let seenPaths = [];
      mockExistsSync = (p) => {
        seenPaths.push(p);
        return true;
      };
      await runCortexEmbeddingsGate({
        modelDirectory: '/custom/models',
        evaluationReport: {
          chunkCount: 10,
          embeddingCount: 10,
          bm25MrrAt5: 0.5,
          hybridMrrAt5: 0.6,
        },
      });
      assert.ok(seenPaths.every((p) => p.startsWith(path.resolve('/custom/models'))));
    });
  });

  describe('main via import.meta.url guard', () => {
    it('prints usage with --help', async () => {
      const logs = await importGateMain(['--help']);
      assert.ok(logs.some((l) => l.includes('Cortex embeddings gate')));
    });

    it('emits JSON with --json when pass=true', async () => {
      mockExistsSync = () => true;
      mockValidationReport = { evidence: [] };
      const logs = await importGateMain(['--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    });

    it('emits PASS text without --json when pass=true', async () => {
      mockExistsSync = () => true;
      mockValidationReport = { evidence: [] };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.equal(process.exitCode, 0);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockExistsSync = () => false;
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });

    it('emits FAIL text without --json when pass=false', async () => {
      mockExistsSync = () => false;
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.equal(process.exitCode, 1);
    });

    it('parses all CLI flag overrides with --json', async () => {
      mockExistsSync = () => true;
      mockValidationReport = { evidence: [] };
      mockEvaluationReport = {
        bm25MrrAt5: 0.5,
        chunkCount: 10,
        embeddingCount: 10,
        hybridMrrAt5: 0.6,
      };
      const logs = await importGateMain([
        '--json',
        '--database=custom/db.sqlite',
        '--model-directory=/custom/models',
        '--model-id=test-model',
        '--query-file=custom-queries.json',
        '--min-hybrid-improvement=0.03',
        '--unknown-flag',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    });
  });

  describe('import.meta.url guard', () => {
    it('does not run main when argv[1] does not match', async () => {
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        jest.resetModules();
        await withArgv([process.execPath, 'dummy'], async () => {
          await import('./cortex-embeddings.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});