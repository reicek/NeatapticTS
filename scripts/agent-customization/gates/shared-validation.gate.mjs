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
 *   node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=src/foo.ts --artifact-path=coverage/shared-validation.json
 *   node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=src/foo.ts --full-validation
 *
 * By default the build and lint runners are SCOPED to the changed files:
 *   - the build runs only when a changed file can change the build output;
 *   - lint runs over the changed files that live inside the roots covered by
 *     `npm run lint`.
 * `--full-validation` restores the unscoped `npm run build` + `npm run lint`
 * behaviour for release-grade runs.
 */

import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

import { parseArgs, repoRoot } from '../customization-utils.mjs';
import {
  GATE_MAX_BUFFER_BYTES,
  SHARED_VALIDATION_RUNNER_TIMEOUTS,
} from '../gate-timeout-budgets.mjs';
import { deriveTestFiles } from './gate-test-utils.mjs';

const OWNER = 'shared-validation';
const DEFAULT_ARTIFACT_PATH = 'coverage/shared-validation.json';
const DEFAULT_JEST_CONFIG = 'jest.config.mjs';

/**
 * Repo-relative roots whose contents can change the compiled build output.
 *
 * `npm run build` runs `webpack --config webpack.config.js` (entry
 * `./src/neataptic.ts`) followed by `tsc`, and `tsconfig.json` includes only
 * TypeScript files under `src`. A change anywhere else cannot alter
 * `dist/`, so the build does not need to run for such a slice.
 */
const BUILD_RELEVANT_ROOTS = ['src/'];

/**
 * Individual repo-relative files that change how the build is produced.
 */
const BUILD_RELEVANT_FILES = [
  'package.json',
  'tsconfig.json',
  'webpack.config.js',
];

/**
 * Repo-relative roots covered by `npm run lint`
 * (`eslint src/ testing/ benchmarks/ examples/`). Scoped linting reuses the
 * same roots so a scoped run covers exactly the same files as the full run,
 * never fewer and never more.
 */
const LINT_RELEVANT_ROOTS = ['benchmarks/', 'examples/', 'src/', 'testing/'];

/**
 * File extensions the ESLint configuration can parse.
 */
const LINTABLE_EXTENSIONS = ['.cjs', '.js', '.jsx', '.mjs', '.ts', '.tsx'];

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
  node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=src/foo.ts --artifact-path=coverage/shared-validation.json

Options:
  --json               Write machine-readable JSON to stdout.
  --changed-files      Comma or newline separated repo-relative source paths.
  --artifact-path      Repo-relative path for the JSON artifact (default: coverage/shared-validation.json).
  --full-validation    Force the unscoped full build and lint instead of scoping
                       them to --changed-files.
  --help, -h           Show this help text.`);
    process.exit(0);
  }

  const changedFiles = parseChangedFiles(argv);
  const artifactPath = parseArtifactPath(argv);
  const fullValidation = parseFullValidation(argv);
  const result = await runSharedValidationGate({
    changedFiles,
    artifactPath,
    fullValidation,
  });

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
 * @param {boolean} [params.fullValidation] - Force the unscoped full build and lint.
 * @param {function} [params.testRunner] - Optional async runner for tests.
 * @param {function} [params.buildRunner] - Optional async runner for build.
 * @param {function} [params.lintRunner] - Optional async runner for lint.
 * @returns {Promise<object>} Standard gate contract.
 */
export async function runSharedValidationGate(params = {}) {
  const changedFiles = params.changedFiles ?? [];
  const fullValidation = params.fullValidation === true;

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

  const artifactPath = params.artifactPath ?? DEFAULT_ARTIFACT_PATH;
  const testFiles = deriveTestFiles(changedFiles);

  const testRunner = params.testRunner ?? defaultTestRunner;
  const buildRunner = params.buildRunner ?? defaultBuildRunner;
  const lintRunner = params.lintRunner ?? defaultLintRunner;

  const testResult = await testRunner({ changedFiles, testFiles });
  const buildResult = await buildRunner({ changedFiles, fullValidation });
  const lintResult = await lintRunner({ changedFiles, fullValidation });

  const artifact = {
    changedFiles: Array.from(changedFiles),
    fullValidation,
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

  const existingFiles = testFiles.filter((file) =>
    existsSync(path.join(repoRoot, file)),
  );
  if (existingFiles.length === 0) {
    return { status: 0, stdout: 'No test files selected.', stderr: '' };
  }

  const mjsFiles = existingFiles.filter((file) => file.endsWith('.mjs'));
  const otherFiles = existingFiles.filter((file) => !file.endsWith('.mjs'));

  const groups = [];
  if (mjsFiles.length > 0) {
    groups.push({
      files: mjsFiles,
      nodeOptions: '--experimental-vm-modules',
    });
  }
  if (otherFiles.length > 0) {
    groups.push({ files: otherFiles });
  }

  let overallStatus = 0;
  const outputs = [];
  const errors = [];

  for (const group of groups) {
    const groupResult = await runJestGroup(group.files, group.nodeOptions);
    if (groupResult.status !== 0) {
      overallStatus = groupResult.status || 1;
    }
    outputs.push(`[${group.files.join(', ')}]\nstdout:\n${groupResult.stdout}`);
    errors.push(`[${group.files.join(', ')}]\nstderr:\n${groupResult.stderr}`);
  }

  return {
    status: overallStatus,
    stdout: outputs.join('\n---\n'),
    stderr: errors.join('\n---\n'),
  };
}

/**
 * Run Jest for a single homogeneous group of test files.
 *
 * `.mjs` test files require `--experimental-vm-modules`; other files must not
 * inherit that flag (it breaks ts-jest). The runner spawns a fresh Node
 * process with the appropriate NODE_OPTIONS for the group.
 *
 * @param {string[]} files - Absolute paths to test files.
 * @param {string|undefined} nodeOptions - NODE_OPTIONS value for the group.
 * @returns {Promise<{status: number, stdout: string, stderr: string}>} Group result.
 */
async function runJestGroup(files, nodeOptions) {
  const absoluteFiles = files.map((file) => path.join(repoRoot, file));
  const useShell = process.platform === 'win32';
  const env = {
    ...process.env,
    NODE_OPTIONS: nodeOptions ?? '',
  };
  // The INNER jest spawn gets a smaller ceiling than the OUTER wrapper that
  // supervises it, so the inner layer always reports a clean runner failure
  // before the outer wrapper could truncate it.
  const innerJestTimeoutMs = SHARED_VALIDATION_RUNNER_TIMEOUTS.tests - 50_000;
  const evalScript =
    `import { spawnSync } from 'node:child_process'; ` +
    `const r = spawnSync('npx', ['jest', '--config=${DEFAULT_JEST_CONFIG}', '--no-cache', '--runInBand', ...${JSON.stringify(absoluteFiles)}], { cwd: ${JSON.stringify(repoRoot)}, encoding: 'utf8', timeout: ${innerJestTimeoutMs}, maxBuffer: ${GATE_MAX_BUFFER_BYTES}, stdio: ['ignore', 'pipe', 'pipe'], shell: ${JSON.stringify(useShell)}, env: ${JSON.stringify(env)} }); ` +
    `console.log(JSON.stringify({ status: r.status ?? (r.signal ? 1 : 0), stdout: r.stdout ?? '', stderr: r.stderr ?? '', error: r.error ? String(r.error) : undefined }));`;

  const result = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', evalScript],
    {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: SHARED_VALIDATION_RUNNER_TIMEOUTS.tests,
      maxBuffer: GATE_MAX_BUFFER_BYTES,
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
 * Default build runner used in production.
 *
 * Runs `npm run build` when the changed files can actually change the build
 * output, and records an explicit skip otherwise. The full build cannot be
 * narrowed to a subset of files (webpack bundles the whole `src/` graph and
 * `tsc` type-checks the whole project), so the only lever available for a
 * slice that touches no build input is to skip the run and say why.
 *
 * @param {object} [params] - Runner parameters.
 * @param {string[]} [params.changedFiles] - Repo-relative changed file paths.
 * @param {boolean} [params.fullValidation] - Force the unscoped full build.
 * @returns {Promise<object>} `{ status, stdout, stderr, skipped, scope }`.
 */
async function defaultBuildRunner({ changedFiles = [], fullValidation = false } = {}) {
  const scope = resolveBuildScope(changedFiles);
  const scopeRecord = {
    mode: fullValidation ? 'full' : 'scoped',
    required: fullValidation ? true : scope.required,
    relevantFiles: scope.relevantFiles,
    reason: fullValidation
      ? 'Full build forced (--full-validation).'
      : scope.reason,
  };

  if (!fullValidation && !scope.required) {
    return {
      status: 0,
      stdout: `Build skipped. ${scope.reason}`,
      stderr: '',
      skipped: true,
      durationMs: 0,
      scope: scopeRecord,
    };
  }

  const startedAt = Date.now();
  const result = spawnSync('npm', ['run', 'build'], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: SHARED_VALIDATION_RUNNER_TIMEOUTS.build,
    maxBuffer: GATE_MAX_BUFFER_BYTES,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: process.platform === 'win32',
  });

  return {
    status: result.status ?? (result.error ? 1 : result.signal ? 1 : 0),
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
    skipped: false,
    durationMs: Date.now() - startedAt,
    scope: scopeRecord,
  };
}

/**
 * Default lint runner used in production.
 *
 * `npm run lint` lints the whole repository and dominates the shared
 * validation wall-clock budget. When the changed files are known, lint
 * exactly those files instead, restricted to the same roots `npm run lint`
 * covers so the scoped run has identical coverage of the changed surface.
 *
 * ESLint's own ignore rules still apply: a changed file that the flat config
 * ignores (`scripts/`, `plans/`, `docs/`, generated bundles) is skipped
 * silently via `--no-warn-ignored` and does not fail the run.
 *
 * @param {object} [params] - Runner parameters.
 * @param {string[]} [params.changedFiles] - Repo-relative changed file paths.
 * @param {boolean} [params.fullValidation] - Force the unscoped full lint.
 * @returns {Promise<object>} `{ status, stdout, stderr, skipped, scope }`.
 */
async function defaultLintRunner({ changedFiles = [], fullValidation = false } = {}) {
  const scope = resolveLintScope(changedFiles);
  const scopeRecord = {
    mode: fullValidation ? 'full' : 'scoped',
    files: scope.files,
    outOfScopeFiles: scope.outOfScopeFiles,
    missingFiles: scope.missingFiles,
    reason: fullValidation
      ? 'Full lint forced (--full-validation): the whole repository is linted.'
      : scope.reason,
  };

  if (fullValidation) {
    const startedAt = Date.now();
    const result = spawnSync('npm', ['run', 'lint'], {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: SHARED_VALIDATION_RUNNER_TIMEOUTS.lint,
      maxBuffer: GATE_MAX_BUFFER_BYTES,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: process.platform === 'win32',
    });

    return {
      status: result.status ?? (result.error ? 1 : result.signal ? 1 : 0),
      stdout: result.stdout ?? '',
      stderr: result.stderr ?? '',
      skipped: false,
      durationMs: Date.now() - startedAt,
      scope: scopeRecord,
    };
  }

  if (scope.files.length === 0) {
    return {
      status: 0,
      stdout: `Lint skipped. ${scope.reason}`,
      stderr: '',
      skipped: true,
      durationMs: 0,
      scope: scopeRecord,
    };
  }

  const startedAt = Date.now();
  const result = spawnSync(
    'npx',
    ['eslint', ...scope.files, '--no-warn-ignored'],
    {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: SHARED_VALIDATION_RUNNER_TIMEOUTS.lint,
      maxBuffer: GATE_MAX_BUFFER_BYTES,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: process.platform === 'win32',
    },
  );

  return {
    status: result.status ?? (result.error ? 1 : result.signal ? 1 : 0),
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
    skipped: false,
    durationMs: Date.now() - startedAt,
    scope: scopeRecord,
  };
}

/**
 * Converts an arbitrary path (absolute or repo-relative) into a
 * repo-relative POSIX-style path.
 *
 * @param {string} filePath - Absolute or repo-relative path.
 * @returns {string} Repo-relative path, or an empty string for blank input.
 */
export function toRepoRelative(filePath) {
  const raw = String(filePath ?? '').trim();
  if (raw.length === 0) return '';

  const absolute = path.isAbsolute(raw) ? raw : path.resolve(repoRoot, raw);
  const relative = path.relative(repoRoot, absolute);
  if (relative.startsWith('..')) {
    // Outside the repository: keep the original value so it is reported as
    // out of scope rather than silently mapped onto an unrelated file.
    return raw.replace(/\\/gu, '/');
  }
  return relative.split(path.sep).join('/');
}

/**
 * Decides whether the changed files can affect the compiled build output.
 *
 * @param {string[]} changedFiles - Repo-relative changed file paths.
 * @returns {{ required: boolean, relevantFiles: string[], reason: string }} Scope decision.
 */
export function resolveBuildScope(changedFiles) {
  const normalized = (changedFiles ?? [])
    .map(toRepoRelative)
    .filter((entry) => entry.length > 0);

  const relevantFiles = normalized.filter(
    (file) =>
      BUILD_RELEVANT_ROOTS.some((root) => file.startsWith(root)) ||
      BUILD_RELEVANT_FILES.includes(file),
  );

  if (relevantFiles.length > 0) {
    return {
      required: true,
      relevantFiles,
      reason: `Build required: ${relevantFiles.join(', ')} can change the build output.`,
    };
  }

  return {
    required: false,
    relevantFiles,
    reason:
      `Build skipped: no changed file affects the build output ` +
      `(build inputs are ${BUILD_RELEVANT_ROOTS.join(', ')} and ${BUILD_RELEVANT_FILES.join(', ')}).`,
  };
}

/**
 * Resolves the set of changed files that scoped linting must cover.
 *
 * Only files inside the roots covered by `npm run lint` are eligible, and
 * only when they exist on disk and carry a lintable extension. Files that
 * ESLint itself ignores are still passed to ESLint so that the ignore rules
 * remain the single source of truth.
 *
 * @param {string[]} changedFiles - Repo-relative changed file paths.
 * @returns {{ files: string[], outOfScopeFiles: string[], missingFiles: string[], reason: string }} Scope decision.
 */
export function resolveLintScope(changedFiles) {
  const normalized = (changedFiles ?? [])
    .map(toRepoRelative)
    .filter((entry) => entry.length > 0);

  const inRoot = normalized.filter((file) =>
    LINT_RELEVANT_ROOTS.some((root) => file.startsWith(root)),
  );
  const lintable = inRoot.filter((file) =>
    LINTABLE_EXTENSIONS.some((extension) => file.endsWith(extension)),
  );
  // Every changed file lands in exactly one bucket: lintable-and-present,
  // lintable-but-missing, or not lintable at all.
  const outOfScopeFiles = normalized.filter((file) => !lintable.includes(file));
  const files = lintable.filter((file) =>
    existsSync(path.join(repoRoot, file)),
  );
  const missingFiles = lintable.filter((file) => !files.includes(file));

  const reason =
    files.length > 0
      ? `Lint scoped to ${files.length} changed file(s): ${files.join(', ')}.`
      : 'Lint skipped: no changed file is inside a lint root ' +
        `(${LINT_RELEVANT_ROOTS.join(', ')}) with a lintable extension that exists on disk.`;

  return { files, outOfScopeFiles, missingFiles, reason };
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
 * Parses `--full-validation` from raw CLI arguments.
 *
 * When present, the build and lint runners fall back to the unscoped
 * `npm run build` / `npm run lint` invocations instead of scoping themselves
 * to the changed files.
 *
 * @param {string[]} argv - Raw CLI arguments.
 * @returns {boolean} True when the unscoped full validation was requested.
 */
function parseFullValidation(argv) {
  return argv.some(
    (arg) => arg === '--full-validation' || arg.startsWith('--full-validation='),
  );
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
  return DEFAULT_ARTIFACT_PATH;
}
