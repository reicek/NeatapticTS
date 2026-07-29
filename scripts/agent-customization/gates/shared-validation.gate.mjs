#!/usr/bin/env node
/**
 * Tier-1 gate: shared-validation
 *
 * Runs tests, build, and lint once for a set of changed files, then writes a
 * structured JSON artifact that every Tier-3 specialist reviewer receives.
 * This eliminates the duplicate work of three specialists independently
 * running the same suites and build.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string|null, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/shared-validation.gate.mjs --json
 *   node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=src/foo.ts,src/bar.ts
 *   node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=src/foo.ts --artifact-path=artifacts/shared-validation.json
 */

import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

import { parseArgs, repoRoot } from '../customization-utils.mjs';
import { deriveTestFiles } from './gate-test-utils.mjs';

const OWNER = 'shared-validation';
const DEFAULT_JEST_CONFIG = 'jest.config.mjs';

/* istanbul ignore next */
const isMain = import.meta.url === pathToFileURL(process.argv[1] ?? '').href;

/**
 * CLI entry point for the shared validation gate.
 *
 * @param {string[]} [argv] - Raw CLI arguments (defaults to `process.argv.slice(2)`).
 * @returns {Promise<object>} The standard gate contract.
 */
export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);

  if (options.help) {
    console.log(`shared-validation gate

Usage:
  node scripts/agent-customization/gates/shared-validation.gate.mjs --json
  node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=src/foo.ts,src/bar.ts
  node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=src/foo.ts --artifact-path=artifacts/shared-validation.json

Options:
  --json               Write machine-readable JSON to stdout.
  --changed-files      Comma or newline separated repo-relative source paths.
  --artifact-path      Repo-relative path for the JSON artifact (default: artifacts/shared-validation.json).
  --help, -h           Show this help text.`);
    process.exit(0);
  }

  const changedFiles = parseChangedFiles(argv);
  const artifactPath = parseArtifactPath(argv);
  const result = await runSharedValidationGate({ changedFiles, artifactPath });

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
 * Runs the shared validation gate.
 *
 * Executes tests, build, and lint once for the supplied changed files, writes
 * a structured JSON artifact, and returns a standard gate contract. The artifact
 * contains the changed file list and the captured stdout/stderr for each
 * runner so that specialist reviewers can read the shared results instead of
 * re-running the suites themselves.
 *
 * @param {object} [params] - Gate parameters.
 * @param {string[]} [params.changedFiles] - Repo-relative changed file paths.
 * @param {string} [params.artifactPath] - Repo-relative path for the JSON artifact.
 * @param {function} [params.testRunner] - Optional async runner for tests.
 * @param {function} [params.buildRunner] - Optional async runner for build.
 * @param {function} [params.lintRunner] - Optional async runner for lint.
 * @returns {Promise<object>} Standard gate contract.
 */
export async function runSharedValidationGate(params = {}) {
  const changedFiles = params.changedFiles ?? [];

  // CLI descriptor mode: when invoked without changed files, just identify
  // the gate so callers can list it without running expensive suites.
  if (changedFiles.length === 0) {
    return {
      pass: true,
      evidence: {
        mode: 'standalone-descriptor',
        owner: OWNER,
        description:
          'Shared validation gate: runs tests, build, and lint once for changed files and writes a JSON artifact for specialist reviewers.',
      },
      fixHint: 'n/a',
      owner: OWNER,
    };
  }

  const artifactPath =
    params.artifactPath ?? 'artifacts/shared-validation.json';
  const testFiles = deriveTestFiles(changedFiles);

  const testRunner = params.testRunner ?? defaultTestRunner;
  const buildRunner = params.buildRunner ?? defaultBuildRunner;
  const lintRunner = params.lintRunner ?? defaultLintRunner;

  const testResult = await testRunner({ changedFiles, testFiles });
  const buildResult = await buildRunner({ changedFiles });
  const lintResult = await lintRunner({ changedFiles });

  const artifact = {
    changedFiles: Array.from(changedFiles),
    testResult,
    buildResult,
    lintResult,
    timestamp: new Date().toISOString(),
  };

  const finalArtifactPath = path.isAbsolute(artifactPath)
    ? artifactPath
    : path.join(repoRoot, artifactPath);
  const artifactDir = path.dirname(finalArtifactPath);
  if (!existsSync(artifactDir)) {
    mkdirSync(artifactDir, { recursive: true });
  }

  writeFileSync(finalArtifactPath, JSON.stringify(artifact, null, 2), 'utf8');

  const allPassed =
    testResult.status === 0 &&
    buildResult.status === 0 &&
    lintResult.status === 0;

  if (!allPassed) {
    const failed = [
      testResult.status !== 0 ? 'tests' : '',
      buildResult.status !== 0 ? 'build' : '',
      lintResult.status !== 0 ? 'lint' : '',
    ]
      .filter(Boolean)
      .join(', ');

    return {
      pass: false,
      evidence: artifact,
      fixHint: `shared validation failed: ${failed}. Fix the failing runner(s), then re-run the shared-validation gate.`,
      owner: OWNER,
    };
  }

  return {
    pass: true,
    evidence: artifact,
    fixHint: null,
    owner: OWNER,
  };
}

// ---------------------------------------------------------------------------
// Helpers

/**
 * Default test runner used in production. Spawns `npx jest` with the narrowest
 * file selection so the shared test run completes quickly.
 *
 * @param {object} params - Runner parameters.
 * @param {string[]} params.testFiles - Repo-relative test file paths.
 * @returns {Promise<object>} `{ status: number, stdout: string, stderr: string }`.
 */
async function defaultTestRunner({ testFiles }) {
  if (testFiles.length === 0) {
    return { status: 0, stdout: 'No test files selected.', stderr: '' };
  }

  const absoluteFiles = testFiles.map((file) => path.join(repoRoot, file));
  const useShell = process.platform === 'win32';
  const evalScript =
    `import { spawnSync } from 'node:child_process'; ` +
    `const r = spawnSync('npx', ['jest', '--config=${DEFAULT_JEST_CONFIG}', '--no-cache', '--runInBand', ...${JSON.stringify(absoluteFiles)}], { cwd: ${JSON.stringify(repoRoot)}, encoding: 'utf8', timeout: 600_000, stdio: ['ignore', 'pipe', 'pipe'], shell: ${JSON.stringify(useShell)} }); ` +
    `console.log(JSON.stringify({ status: r.status ?? (r.signal ? 1 : 0), stdout: r.stdout ?? '', stderr: r.stderr ?? '', error: r.error ? String(r.error) : undefined }));`;

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
    const parsed = JSON.parse(raw);
    if (parsed.error) {
      parsed.status = parsed.status ?? 1;
    }
    return parsed;
  } catch {
    const status = result.status ?? 1;
    const stdout = result.stdout ?? '';
    const stderr = result.stderr ?? '';
    return { status, stdout, stderr };
  }
}

/**
 * Default build runner used in production. Spawns `npm run build`.
 *
 * @returns {Promise<object>} `{ status: number, stdout: string, stderr: string }`.
 */
async function defaultBuildRunner() {
  const result = spawnSync('npm', ['run', 'build'], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 300_000,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: process.platform === 'win32',
  });

  return {
    status: result.status ?? (result.error ? 1 : result.signal ? 1 : 0),
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
  };
}

/**
 * Default lint runner used in production. Spawns `npm run lint`.
 *
 * @returns {Promise<object>} `{ status: number, stdout: string, stderr: string }`.
 */
async function defaultLintRunner() {
  const result = spawnSync('npm', ['run', 'lint'], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 300_000,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: process.platform === 'win32',
  });

  return {
    status: result.status ?? (result.error ? 1 : result.signal ? 1 : 0),
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
  };
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

/**
 * Parses `--artifact-path` from raw CLI arguments.
 *
 * @param {string[]} argv - Raw CLI arguments.
 * @returns {string} Repo-relative artifact path.
 */
function parseArtifactPath(argv) {
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg.startsWith('--artifact-path=')) {
      return arg.slice('--artifact-path='.length);
    }
    if (arg === '--artifact-path' && argv[index + 1] !== undefined) {
      return argv[++index];
    }
  }
  return 'artifacts/shared-validation.json';
}
