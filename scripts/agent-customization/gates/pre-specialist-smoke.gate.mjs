#!/usr/bin/env node
/**
 * Tier-1 gate: pre-specialist-smoke
 *
 * Runs the narrowest Jest selection against the test files implied by a list of
 * changed source files. The orchestrator calls this gate after `04-implementing`
 * returns and before dispatching Tier-3 specialists, so simple test/fixture
 * failures loop back to implementation in seconds instead of wasting specialist
 * dispatches.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string|null, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs --json
 *   node scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs --json --changed-files=src/foo.ts,src/bar.ts
 */

import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

import { parseArgs, repoRoot } from '../customization-utils.mjs';
import { deriveTestFiles } from './gate-test-utils.mjs';

const OWNER = 'pre-specialist-smoke';
const DEFAULT_JEST_CONFIG = 'jest.config.mjs';

/* istanbul ignore next */
const isMain = import.meta.url === pathToFileURL(process.argv[1] ?? '').href;

/**
 * CLI entry point for the pre-specialist smoke gate.
 *
 * @param {string[]} [argv] - Raw CLI arguments (defaults to `process.argv.slice(2)`).
 * @returns {Promise<object>} The standard gate contract.
 */
export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);

  if (options.help) {
    console.log(`pre-specialist-smoke gate

Usage:
  node scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs --json
  node scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs --json --changed-files=src/foo.ts,src/bar.ts

Options:
  --json               Write machine-readable JSON to stdout.
  --changed-files      Comma or newline separated repo-relative source paths.
  --help, -h           Show this help text.`);
    process.exit(0);
  }

  const changedFiles = parseChangedFiles(argv);
  const result = await runPreSpecialistSmokeGate({ changedFiles });

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result.pass ? 'PASS' : 'FAIL', `${OWNER} gate`);
    if (!result.pass) console.log('fixHint:', result.fixHint);
  }

  process.exitCode = result.pass ? 0 : 1;
  return result;
}

/* istanbul ignore next */
if (isMain) {
  main();
}

// ---------------------------------------------------------------------------

/**
 * Runs the pre-specialist smoke gate.
 *
 * Maps each changed file to the narrowest set of owner-local test files, then
 * runs them with Jest. If no test files are implied by the changed list, the
 * gate passes immediately (nothing to smoke). If the runner reports a non-zero
 * exit, the gate fails with the captured stdout/stderr as evidence so the
 * orchestrator can loop back to `04-implementing`.
 *
 * @param {object} [params] - Gate parameters.
 * @param {string[]} [params.changedFiles] - Repo-relative changed file paths.
 * @param {function} [params.runner] - Optional async runner for tests; called
 *   with `{ testFiles: string[], changedFiles: string[] }`. The default runner
 *   spawns `npx jest --config=jest.config.mjs --no-cache --runInBand <files>`.
 * @returns {Promise<object>} Standard gate contract.
 */
export async function runPreSpecialistSmokeGate(params = {}) {
  const changedFiles = params.changedFiles ?? [];

  // CLI descriptor mode: when invoked without changed files, just identify
  // the gate so callers can list it without running tests.
  if (changedFiles.length === 0) {
    return {
      pass: true,
      evidence: {
        mode: 'standalone-descriptor',
        owner: OWNER,
        description:
          'Pre-specialist smoke gate: runs the narrowest Jest selection for changed files before dispatching specialists.',
      },
      fixHint: 'n/a',
      owner: OWNER,
    };
  }

  const testFiles = deriveTestFiles(changedFiles);

  if (testFiles.length === 0) {
    return {
      pass: true,
      evidence: {
        changedFiles,
        testFiles: [],
        message: 'No owner-local test files selected for smoke run.',
      },
      fixHint: null,
      owner: OWNER,
    };
  }

  const runner = params.runner ?? defaultJestRunner;
  const runResult = await runner({ testFiles, changedFiles });

  if (runResult.status !== 0) {
    return {
      pass: false,
      evidence: {
        changedFiles,
        testFiles,
        exitStatus: runResult.status,
        stdout: runResult.stdout,
        stderr: runResult.stderr,
      },
      fixHint:
        'Smoke tests failed before specialist review. Route back to 04-implementing, fix the failing tests or fixtures, and re-run this gate.',
      owner: OWNER,
    };
  }

  return {
    pass: true,
    evidence: {
      changedFiles,
      testFiles,
      exitStatus: runResult.status,
      stdout: runResult.stdout,
      stderr: runResult.stderr,
    },
    fixHint: null,
    owner: OWNER,
  };
}

/**
 * Default Jest runner used in production. Spawns `npx jest` with the narrowest
 * file selection so the smoke run completes quickly.
 *
 * @param {object} params - Runner parameters.
 * @param {string[]} params.testFiles - Repo-relative test file paths.
 * @returns {Promise<object>} `{ status: number, stdout: string, stderr: string }`.
 */
async function defaultJestRunner({ testFiles }) {
  const absoluteFiles = testFiles.map((file) => path.join(repoRoot, file));
  const useShell = process.platform === 'win32';
  const evalScript =
    `import { spawnSync } from 'node:child_process'; ` +
    `const r = spawnSync('npx', ['jest', '--config=${DEFAULT_JEST_CONFIG}', '--no-cache', '--runInBand', ...${JSON.stringify(absoluteFiles)}], { cwd: ${JSON.stringify(repoRoot)}, encoding: 'utf8', timeout: 600_000, stdio: ['ignore', 'pipe', 'pipe'], shell: ${JSON.stringify(useShell)} }); ` +
    `console.log(JSON.stringify({ status: r.status ?? (r.signal ? 1 : 0), stdout: r.stdout ?? '', stderr: r.stderr ?? '' }));`;
  const result = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', evalScript],
    {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: 650_000,
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );

  const raw = result.stdout?.trim() ?? '';
  try {
    return JSON.parse(raw);
  } catch {
    const status = result.status ?? 1;
    const stdout = result.stdout ?? '';
    const stderr = result.stderr ?? '';
    return { status, stdout, stderr };
  }
}

/**
 * Parses `--changed-files` from raw CLI arguments, accepting comma or newline
 * separated repo-relative paths.
 *
 * @param {string[]} argv - Raw CLI arguments.
 * @returns {string[]} Non-empty repo-relative paths.
 */
function parseChangedFiles(argv) {
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    let value;
    if (arg.startsWith('--changed-files=')) {
      value = arg.slice('--changed-files='.length);
    } else if (arg === '--changed-files' && argv[index + 1] !== undefined) {
      value = argv[++index];
    }

    if (value !== undefined) {
      return value
        .split(/[,\r\n]+/u)
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);
    }
  }
  return [];
}
