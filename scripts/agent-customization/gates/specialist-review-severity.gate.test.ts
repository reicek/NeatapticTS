/**
 * @module specialist-review-severity.gate.test
 * @description Green tests for the specialist-review-severity gate contract and
 *   CLI entry point.
 */
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const REPO_ROOT = path.resolve();
const GATE_PATH = './specialist-review-severity.gate.mjs';
const GATE_ABS_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/specialist-review-severity.gate.mjs',
);

type LogCapture = {
  logs: unknown[][];
  errors: unknown[][];
};

/**
 * Re-import the gate module while pretending it is the main Node entry point.
 *
 * Because the gate's CLI code runs at top-level, this is the only way to
 * exercise the `import.meta.url === pathToFileURL(process.argv[1])` guard and
 * the wrapper helpers (`printUsage`, `loadChangedFiles`) inside Jest's
 * coverage context. Console output is captured and `process.argv` /
 * `process.exitCode` are restored afterwards.
 *
 * We use `jest.resetModules()` rather than `jest.isolateModulesAsync()` because
 * the .mjs gate is loaded through a dynamic `import()` from this TypeScript
 * test. Jest's CJS transform for the test file makes `isolateModulesAsync` a
 * poor fit for repeated ESM gate imports, while `resetModules()` reliably
 * re-evaluates the gate's top-level CLI guard across invocations.
 */
async function importGateAsMain(argv: string[]): Promise<LogCapture> {
  const originalArgv = process.argv;
  const originalExitCode = process.exitCode;
  const logs: unknown[][] = [];
  const errors: unknown[][] = [];

  process.argv = ['node', GATE_ABS_PATH, ...argv];
  jest.resetModules();

  const logSpy = jest
    .spyOn(console, 'log')
    .mockImplementation((...args: unknown[]) => {
      logs.push(args);
    });
  const errorSpy = jest
    .spyOn(console, 'error')
    .mockImplementation((...args: unknown[]) => {
      errors.push(args);
    });

  try {
    await import(GATE_PATH);
  } finally {
    logSpy.mockRestore();
    errorSpy.mockRestore();
    process.argv = originalArgv;
    process.exitCode = originalExitCode;
  }

  return { logs, errors };
}

describe('specialist-review-severity.gate.mjs', () => {
  describe('classifySeverity', () => {
    it('exposes a classifySeverity function', async () => {
      const gate = await import(GATE_PATH);

      expect(typeof gate.classifySeverity).toBe('function');
    });

    it('returns severity TRIVIAL when every changed file is a test file', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity([
        'testing/architecture/network.test.ts',
      ]);

      expect(result.severity).toBe('TRIVIAL');
    });

    it('returns severity FULL when a src runtime source file changes', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity(['src/foo.ts']);

      expect(result.severity).toBe('FULL');
    });

    it('returns severity FULL when an examples runtime source file changes', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity(['examples/bar.mjs']);

      expect(result.severity).toBe('FULL');
    });

    it('returns severity FULL and partitions trivial and non-trivial files for mixed changes', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity(['src/foo.ts', 'src/foo.test.ts']);

      expect(result).toEqual({
        severity: 'FULL',
        trivialFiles: ['src/foo.test.ts'],
        nonTrivialFiles: ['src/foo.ts'],
        specialistCount: 1,
      });
    });

    it('throws a TypeError when the input is not an array', async () => {
      const gate = await import(GATE_PATH);

      expect(() =>
        gate.classifySeverity('src/foo.ts' as unknown as string[]),
      ).toThrow('classifySeverity expects an array of file paths');
    });

    it('throws a TypeError when a file path is not a string', async () => {
      const gate = await import(GATE_PATH);

      expect(() =>
        gate.classifySeverity(['src/foo.ts', 42 as unknown as string]),
      ).toThrow('classifySeverity expects file paths to be strings');
    });

    it('returns TRIVIAL for an empty input array', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity([]);

      expect(result).toEqual({
        severity: 'TRIVIAL',
        trivialFiles: [],
        nonTrivialFiles: [],
        specialistCount: 0,
      });
    });

    it('classifies Markdown files as trivial', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity(['README.md']);

      expect(result.severity).toBe('TRIVIAL');
    });

    it('classifies formatting config files as trivial', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity(['.prettierrc.json', '.gitignore']);

      expect(result.severity).toBe('TRIVIAL');
    });

    it('classifies lock files as trivial', async () => {
      const gate = await import(GATE_PATH);
      const result = gate.classifySeverity(['package-lock.json']);

      expect(result.severity).toBe('TRIVIAL');
    });
  });

  describe('CLI helpers', () => {
    let tempDir: string;

    beforeEach(() => {
      tempDir = mkdtempSync(path.join(os.tmpdir(), 'severity-gate-'));
    });

    afterEach(() => {
      rmSync(tempDir, { recursive: true, force: true });
    });

    it('reads changed files from an @file-list', async () => {
      const listPath = path.join(tempDir, 'files.txt');
      writeFileSync(
        listPath,
        'src/foo.ts\n# comment\n\ntesting/foo.test.ts\n',
        'utf8',
      );

      const { logs } = await importGateAsMain([
        '--json',
        `--input=@${listPath}`,
      ]);
      const parsed = JSON.parse(String(logs[0][0]));

      expect(parsed).toMatchObject({
        pass: true,
        evidence: {
          classification: {
            severity: 'FULL',
            trivialFiles: ['testing/foo.test.ts'],
            nonTrivialFiles: ['src/foo.ts'],
          },
        },
      });
    });

    it('reads comma-separated changed files from --input', async () => {
      const { logs } = await importGateAsMain([
        '--json',
        '--input=src/foo.ts,testing/foo.test.ts',
      ]);
      const parsed = JSON.parse(String(logs[0][0]));

      expect(parsed).toMatchObject({
        evidence: {
          classification: {
            severity: 'FULL',
            trivialFiles: ['testing/foo.test.ts'],
          },
        },
      });
    });

    it('filters blank entries from comma-separated input', async () => {
      const { logs } = await importGateAsMain([
        '--json',
        '--input=src/foo.ts,,testing/foo.test.ts',
      ]);
      const parsed = JSON.parse(String(logs[0][0]));

      expect(parsed.evidence.classification.nonTrivialFiles).toEqual([
        'src/foo.ts',
      ]);
    });

    it('prints usage text with the --help flag', async () => {
      const { logs } = await importGateAsMain(['--help']);
      const output = logs.map((args) => args.join(' ')).join('\n');

      expect(output).toMatch(
        /specialist-review-severity gate[\s\S]*Usage:[\s\S]*Options:/,
      );
    });

    it('executes the main-module guard and runs the CLI with no arguments', async () => {
      const { logs } = await importGateAsMain([]);
      const output = logs.map((args) => args.join(' ')).join('\n');

      expect(output).toMatch(/TRIVIAL[\s\S]*no changed files supplied/);
    });

    it('writes JSON output with the --json flag', async () => {
      const { logs } = await importGateAsMain([
        '--json',
        '--input=testing/foo.test.ts',
      ]);
      const parsed = JSON.parse(String(logs[0][0]));

      expect(parsed).toMatchObject({
        pass: true,
        evidence: {
          classification: {
            severity: 'TRIVIAL',
          },
        },
        owner:
          'scripts/agent-customization/gates/specialist-review-severity.gate.mjs',
      });
    });

    it('writes human-readable output and lists non-trivial files for FULL fixes', async () => {
      const { logs } = await importGateAsMain(['--input=src/foo.ts']);
      const output = logs.map((args) => args.join(' ')).join('\n');

      expect(output).toMatch(
        /FULL[\s\S]*non-trivial files:[\s\S]*src\/foo\.ts/,
      );
    });
  });

  describe('CLI integration', () => {
    it('exits successfully from a real JSON subprocess invocation', () => {
      const result = spawnSync(
        process.execPath,
        [GATE_ABS_PATH, '--json', '--input=testing/foo.test.ts'],
        { cwd: REPO_ROOT, encoding: 'utf8' },
      );

      expect(result.status).toBe(0);
    });

    it('emits a TRIVIAL JSON classification from a real subprocess invocation', () => {
      const result = spawnSync(
        process.execPath,
        [GATE_ABS_PATH, '--json', '--input=testing/foo.test.ts'],
        { cwd: REPO_ROOT, encoding: 'utf8' },
      );

      const parsed = JSON.parse(result.stdout);

      expect(parsed).toMatchObject({
        pass: true,
        evidence: {
          classification: {
            severity: 'TRIVIAL',
          },
        },
      });
    });

    it('exits successfully when --help is invoked as a subprocess', () => {
      const result = spawnSync(process.execPath, [GATE_ABS_PATH, '--help'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });

      expect(result.status).toBe(0);
    });

    it('emits usage help text from a real subprocess invocation', () => {
      const result = spawnSync(process.execPath, [GATE_ABS_PATH, '--help'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });

      expect(result.stdout).toMatch(/Usage:[\s\S]*Options:/);
    });
  });
});
