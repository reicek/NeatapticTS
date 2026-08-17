/**
 * @module prewarm-dense.test
 * @description 100% coverage tests for prewarm-dense.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'prewarm-dense.mjs');

// Shared mock functions
const mockSpawnSync = jest.fn();
const mockExistsSync = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();

jest.unstable_mockModule('node:child_process', () => ({
  spawnSync: mockSpawnSync,
}));
jest.unstable_mockModule('node:fs', () => ({
  existsSync: mockExistsSync,
}));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/fake/models',
}));
jest.unstable_mockModule('./reranker-readiness.mjs', () => ({
  DEFAULT_RERANKER_MODEL_DIRECTORY: '/fake/reranker',
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  repoRoot: '/fake/repo',
}));

const { runDensePrewarm } = await import('./prewarm-dense.mjs');

beforeEach(() => {
  jest.clearAllMocks();
  mockSpawnSync.mockReset();
  mockExistsSync.mockReset();
});

// ---------------------------------------------------------------------------
// runDensePrewarm: dry-run mode
// ---------------------------------------------------------------------------

describe('prewarm-dense: runDensePrewarm dry-run', () => {
  it('logs all steps as dry-run when no models exist', async () => {
    const logger = jest.fn();
    const result = await runDensePrewarm({
      dryRun: true,
      modelExists: () => false,
      rerankerModelExists: () => false,
      logger,
    });
    expect(result.exitCode).toBe(0);
    expect(result.report.pass).toBe(true);
    expect(result.report.steps).toHaveLength(0);
    // All 5 steps logged as dry-run
    expect(logger).toHaveBeenCalledTimes(5);
    expect(logger.mock.calls[0][0]).toBe('download-model: dry-run');
  });

  it('logs download-model as skipped when model exists', async () => {
    const logger = jest.fn();
    const result = await runDensePrewarm({
      dryRun: true,
      modelExists: () => true,
      rerankerModelExists: () => false,
      logger,
    });
    expect(logger.mock.calls[0][0]).toBe(
      'download-model: model present, skipping download',
    );
  });

  it('logs download-reranker as skipped when reranker model exists', async () => {
    const logger = jest.fn();
    await runDensePrewarm({
      dryRun: true,
      modelExists: () => false,
      rerankerModelExists: () => true,
      logger,
    });
    const rerankerLog = logger.mock.calls.find(
      (call) => call[0] === 'download-reranker: model present, skipping download',
    );
    expect(rerankerLog).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// runDensePrewarm: normal mode, all steps succeed
// ---------------------------------------------------------------------------

describe('prewarm-dense: runDensePrewarm normal mode', () => {
  it('runs all steps when no models exist and all succeed', async () => {
    const logger = jest.fn();
    const commandRunner = jest
      .fn()
      .mockReturnValue({ status: 0, stdout: '', stderr: '' });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => false,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.exitCode).toBe(0);
    expect(result.report.pass).toBe(true);
    expect(result.report.steps).toHaveLength(5);
    expect(result.report.steps.every((s) => s.status === 'ok')).toBe(true);
    expect(commandRunner).toHaveBeenCalledTimes(5);
  });

  it('skips download-model when model exists', async () => {
    const logger = jest.fn();
    const commandRunner = jest
      .fn()
      .mockReturnValue({ status: 0, stdout: '', stderr: '' });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => true,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.exitCode).toBe(0);
    const downloadStep = result.report.steps.find(
      (s) => s.name === 'download-model',
    );
    expect(downloadStep.status).toBe('skipped');
    expect(commandRunner).toHaveBeenCalledTimes(4);
  });

  it('skips download-reranker when reranker model exists', async () => {
    const logger = jest.fn();
    const commandRunner = jest
      .fn()
      .mockReturnValue({ status: 0, stdout: '', stderr: '' });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => false,
      rerankerModelExists: () => true,
      commandRunner,
      logger,
    });
    expect(result.exitCode).toBe(0);
    const rerankerStep = result.report.steps.find(
      (s) => s.name === 'download-reranker',
    );
    expect(rerankerStep.status).toBe('skipped');
    expect(commandRunner).toHaveBeenCalledTimes(4);
  });

  it('skips both downloads when both models exist', async () => {
    const logger = jest.fn();
    const commandRunner = jest
      .fn()
      .mockReturnValue({ status: 0, stdout: '', stderr: '' });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => true,
      rerankerModelExists: () => true,
      commandRunner,
      logger,
    });
    expect(result.exitCode).toBe(0);
    expect(commandRunner).toHaveBeenCalledTimes(3);
  });
});

// ---------------------------------------------------------------------------
// runDensePrewarm: step failure
// ---------------------------------------------------------------------------

describe('prewarm-dense: runDensePrewarm step failure', () => {
  it('returns failure with stderr message when step fails', async () => {
    const logger = jest.fn();
    const commandRunner = jest.fn().mockReturnValue({
      status: 1,
      stderr: 'some error',
      stdout: '',
    });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => false,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.exitCode).toBe(1);
    expect(result.report.pass).toBe(false);
    expect(result.report.failedStep).toBe('download-model');
    expect(result.report.error).toBe('some error');
  });

  it('returns failure with stdout message when stderr is empty', async () => {
    const logger = jest.fn();
    const commandRunner = jest.fn().mockReturnValue({
      status: 1,
      stderr: '',
      stdout: 'stdout error',
    });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => false,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.report.error).toBe('stdout error');
  });

  it('returns failure with default message when both stdout and stderr are empty', async () => {
    const logger = jest.fn();
    const commandRunner = jest.fn().mockReturnValue({
      status: 1,
      stderr: '',
      stdout: '',
    });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => false,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.report.error).toBe(
      'download-model exited with a non-zero status.',
    );
  });

  it('returns failure when stepResult status is null (defaults to 1)', async () => {
    const logger = jest.fn();
    const commandRunner = jest.fn().mockReturnValue({ status: null });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => false,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.exitCode).toBe(1);
  });

  it('returns failure when stepResult is undefined', async () => {
    const logger = jest.fn();
    const commandRunner = jest.fn().mockReturnValue(undefined);
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => false,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.exitCode).toBe(1);
  });

  it('fails on a non-first step', async () => {
    const logger = jest.fn();
    const commandRunner = jest
      .fn()
      .mockReturnValueOnce({ status: 0 })
      .mockReturnValue({ status: 1, stderr: 'validate failed' });
    const result = await runDensePrewarm({
      dryRun: false,
      modelExists: () => true,
      rerankerModelExists: () => false,
      commandRunner,
      logger,
    });
    expect(result.report.failedStep).toBe('validate-embeddings');
  });
});

// ---------------------------------------------------------------------------
// runDensePrewarm: default logger
// ---------------------------------------------------------------------------

describe('prewarm-dense: runDensePrewarm default logger', () => {
  it('uses console.log as default logger', async () => {
    const origLog = console.log;
    const logs = [];
    console.log = (msg) => logs.push(msg);
    try {
      const commandRunner = jest.fn().mockReturnValue({ status: 0 });
      await runDensePrewarm({
        dryRun: true,
        modelExists: () => false,
        rerankerModelExists: () => false,
        commandRunner,
      });
      expect(logs.length).toBe(5);
    } finally {
      console.log = origLog;
    }
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('prewarm-dense: main()', () => {
  const origArgv1 = process.argv[1];
  const origExitCode = process.exitCode;

  afterEach(() => {
    process.argv[1] = origArgv1;
    process.exitCode = origExitCode;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./prewarm-dense.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText).not.toHaveBeenCalled();
  });

  it('runs dry-run and writes text output', async () => {
    mockParseCliArgs.mockReturnValue({ 'dry-run': true, json: false });
    mockExistsSync.mockReturnValue(false);
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./prewarm-dense.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    const payload = mockWriteJsonOrText.mock.calls[0][0];
    expect(payload.pass).toBe(true);
  });

  it('runs dry-run and writes json output', async () => {
    mockParseCliArgs.mockReturnValue({ 'dry-run': true, json: true });
    mockExistsSync.mockReturnValue(true);
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./prewarm-dense.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    const jsonFlag = mockWriteJsonOrText.mock.calls[0][1];
    expect(jsonFlag).toBe(true);
  });

  it('sets process.exitCode when step fails', async () => {
    mockParseCliArgs.mockReturnValue({ json: false });
    mockExistsSync.mockReturnValue(false);
    mockSpawnSync.mockReturnValue({ status: 1, stderr: 'fail' });
    process.argv[1] = sourceFilePath;
    process.exitCode = undefined;
    await jest.isolateModulesAsync(async () => {
      await import('./prewarm-dense.mjs');
    });
    expect(process.exitCode).toBe(1);
    const payload = mockWriteJsonOrText.mock.calls[0][0];
    expect(payload.pass).toBe(false);
  });
});