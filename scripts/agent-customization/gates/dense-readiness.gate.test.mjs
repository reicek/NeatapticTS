/**
 * @module dense-readiness.gate.test
 * @description Coverage tests for dense-readiness.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/dense-readiness.gate.mjs',
);

let mockReadinessReport;
let mockParsedArgs;
let capturedHelp;
let capturedWrite;

jest.unstable_mockModule('../../../rag-index/cli-utils.mjs', () => ({
  parseCliArgs: () => mockParsedArgs,
  printHelp: (opts) => {
    capturedHelp = opts;
  },
  writeJsonOrText: (payload, json, formatText) => {
    capturedWrite = { payload, json, formatText };
  },
}));
jest.unstable_mockModule('../../../rag-index/dense-readiness.mjs', () => ({
  checkDenseReadiness: async () => mockReadinessReport,
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
  jest.resetModules();
  await withArgv([process.execPath, GATE_PATH, ...argv], async () => {
    await import('./dense-readiness.gate.mjs');
    await new Promise((r) => setTimeout(r, 200));
  });
}

describe('dense-readiness gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockReadinessReport = {
      state: 'warm',
      chunk_count: 10,
      embedding_count: 5,
    };
    mockParsedArgs = {};
    capturedHelp = null;
    capturedWrite = null;
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('evaluateDenseReadinessGate', () => {
    it('returns pass=true for warm state with counts', async () => {
      const { evaluateDenseReadinessGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./dense-readiness.gate.mjs'),
      );
      const result = await evaluateDenseReadinessGate({
        readinessProbe: async () => ({
          state: 'warm',
          chunk_count: 12,
          embedding_count: 7,
        }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.state, 'warm');
      assert.equal(result.evidence.chunk_count, 12);
      assert.equal(result.evidence.embedding_count, 7);
      assert.equal(result.fixHint, null);
      assert.equal(result.owner, '01-planning');
    });

    it('coalesces missing warm counts to null', async () => {
      const { evaluateDenseReadinessGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./dense-readiness.gate.mjs'),
      );
      const result = await evaluateDenseReadinessGate({
        readinessProbe: async () => ({ state: 'warm' }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.chunk_count, null);
      assert.equal(result.evidence.embedding_count, null);
    });

    it('coalesces explicit null warm counts to null', async () => {
      const { evaluateDenseReadinessGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./dense-readiness.gate.mjs'),
      );
      const result = await evaluateDenseReadinessGate({
        readinessProbe: async () => ({
          state: 'warm',
          chunk_count: null,
          embedding_count: null,
        }),
      });
      assert.equal(result.evidence.chunk_count, null);
      assert.equal(result.evidence.embedding_count, null);
    });

    it('returns pass=false for cold state', async () => {
      const { evaluateDenseReadinessGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./dense-readiness.gate.mjs'),
      );
      const result = await evaluateDenseReadinessGate({
        readinessProbe: async () => ({ state: 'cold' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.state, 'cold');
      assert.ok(result.fixHint.includes('index:prewarm'));
    });

    it('returns pass=false for model-only state', async () => {
      const { evaluateDenseReadinessGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./dense-readiness.gate.mjs'),
      );
      const result = await evaluateDenseReadinessGate({
        readinessProbe: async () => ({ state: 'model-only' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.state, 'model-only');
      assert.ok(result.fixHint.includes('index:prewarm'));
    });

    it('passes options through to readinessProbe', async () => {
      const { evaluateDenseReadinessGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./dense-readiness.gate.mjs'),
      );
      let received;
      const result = await evaluateDenseReadinessGate({
        corpusDatabasePath: '/db',
        modelDirectory: '/mdl',
        modelId: 'mid',
        readinessProbe: async (opts) => {
          received = opts;
          return { state: 'warm' };
        },
      });
      assert.equal(result.pass, true);
      assert.equal(received.corpusDatabasePath, '/db');
      assert.equal(received.modelDirectory, '/mdl');
      assert.equal(received.modelId, 'mid');
    });

    it('uses default readinessProbe when no options provided', async () => {
      const { evaluateDenseReadinessGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./dense-readiness.gate.mjs'),
      );
      const result = await evaluateDenseReadinessGate();
      assert.equal(result.pass, true);
    });
  });

  describe('main via import.meta.url guard', () => {
    it('prints help with --help', async () => {
      mockParsedArgs = { help: true };
      await importGateMain(['--help']);
      assert.equal(capturedHelp.title, 'Dense readiness gate');
      assert.equal(capturedWrite, null);
    });

    it('emits JSON with --json when pass=true', async () => {
      mockParsedArgs = { json: true };
      await importGateMain(['--json']);
      assert.notEqual(capturedWrite, null);
      assert.equal(capturedWrite.json, true);
      assert.equal(capturedWrite.payload.pass, true);
    });

    it('emits text without --json when pass=true', async () => {
      mockParsedArgs = {};
      await importGateMain([]);
      assert.notEqual(capturedWrite, null);
      assert.equal(capturedWrite.json, false);
      assert.equal(capturedWrite.formatText(capturedWrite.payload), 'PASS dense-readiness.gate');
    });

    it('emits JSON with --json when pass=false', async () => {
      mockParsedArgs = { json: true };
      mockReadinessReport = { state: 'cold' };
      await importGateMain(['--json']);
      assert.equal(capturedWrite.json, true);
      assert.equal(capturedWrite.payload.pass, false);
      assert.equal(process.exitCode, 1);
    });

    it('emits text without --json when pass=false and sets exitCode', async () => {
      mockParsedArgs = {};
      mockReadinessReport = { state: 'model-only' };
      await importGateMain([]);
      assert.equal(capturedWrite.json, false);
      assert.equal(capturedWrite.formatText(capturedWrite.payload), 'FAIL dense-readiness.gate');
      assert.equal(process.exitCode, 1);
    });

    it('forwards parsed args into gate options', async () => {
      mockParsedArgs = {
        json: false,
        database: '/db',
        'model-directory': '/mdl',
        'model-id': 'mid',
      };
      mockReadinessReport = { state: 'warm' };
      let received;
      jest.resetModules();
      jest.unstable_mockModule('../../../rag-index/dense-readiness.mjs', () => ({
        checkDenseReadiness: async (opts) => {
          received = opts;
          return mockReadinessReport;
        },
      }));
      await withArgv([process.execPath, GATE_PATH], async () => {
        await import('./dense-readiness.gate.mjs');
        await new Promise((r) => setTimeout(r, 200));
      });
      assert.equal(received.corpusDatabasePath, '/db');
      assert.equal(received.modelDirectory, '/mdl');
      assert.equal(received.modelId, 'mid');
    });
  });

  describe('import.meta.url guard', () => {
    it('does not run main when argv[1] does not match', async () => {
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        jest.resetModules();
        capturedHelp = null;
        capturedWrite = null;
        await withArgv([process.execPath, 'dummy'], async () => {
          await import('./dense-readiness.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
      assert.equal(capturedHelp, null);
      assert.equal(capturedWrite, null);
    });
  });
});