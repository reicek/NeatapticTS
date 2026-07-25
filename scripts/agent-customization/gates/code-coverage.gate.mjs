#!/usr/bin/env node
/**
 * @description Tier-1 gate: code-coverage.
 *
 * Checks that changed source files under `src/` and `scripts/agent-customization/`
 * have 100 % line, statement, function, and branch coverage according to
 * `coverage/coverage-summary.json`.
 *
 * The gate accepts an optional `coverageSummaryPath` and explicit file lists for
 * unit testing. When no file list is supplied, changed files are derived from
 * `git status --porcelain` and filtered to the two coverage-relevant directories
 * and source-file extensions.
 *
 * Gate contract: `{ pass: boolean, evidence: object, fixHint: string|null, owner: string }`
 *
 * Usage:
 *   node scripts/agent-customization/gates/code-coverage.gate.mjs [--json]
 *   node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=src/foo.ts,src/bar.ts
 *   node scripts/agent-customization/gates/code-coverage.gate.mjs --json --scripts=scripts/agent-customization/gates/code-coverage.gate.mjs
 *   node scripts/agent-customization/gates/code-coverage.gate.mjs --json --coverage-summary-path=coverage/coverage-summary.json --changed-files=src/foo.ts
 *   node scripts/agent-customization/gates/code-coverage.gate.mjs --json --exemptions=exemptions.json --changed-files=src/foo.ts,src/bar.ts
 *   node scripts/agent-customization/gates/code-coverage.gate.mjs --json --exemptions='{"src/types.ts":"type-only","src/legacy.ts":"legacy-dominant"}' --changed-files=src/types.ts
 *
 * @param {boolean} [--json] - Emit the standard gate JSON contract.
 * @param {string} [--changed-files=<paths>] - Comma or newline separated repo-relative paths.
 * @param {string} [--scripts=<paths>] - Alias for --changed-files.
 * @param {string} [--coverage-summary-path=<path>] - Repo-relative path to the Istanbul summary JSON.
 * @param {string} [--coverage-baseline-path=<path>] - Repo-relative path to the committed coverage baseline JSON.
 * @param {string} [--exemptions=<json-or-path>] - JSON object or path to a JSON file mapping repo-relative paths to exemption kinds (`type-only` or `legacy-dominant`).
 * @returns {void} Exits 0 when coverage is green, 1 otherwise.
 */
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

import { parseArgs, repoRoot } from '../customization-utils.mjs';

const OWNER = 'code-coverage';
const COVERAGE_SUMMARY_PATH = 'coverage/coverage-summary.json';
const COVERAGE_BASELINE_PATH = 'coverage/coverage-baseline.json';
const COVERAGE_DIRS = [
  'src/',
  'scripts/agent-customization/',
  'scripts/mcp-semantic/',
];
const SOURCE_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.mjs']);
const REQUIRED_METRICS = ['lines', 'statements', 'functions', 'branches'];
const TEST_FILE_RE = /\.(test|spec)\.[cm]?[jt]sx?$/iu;
const TEST_DIR_RE = /(^|\/)__tests__\//iu;

/**
 * Runs the code-coverage gate.
 *
 * The gate checks every changed source file against the current coverage
 * summary. Every changed source file must be 100 % covered on lines, statements,
 * functions, and branches. Legacy files whose baseline coverage is below 100 %
 * are allowed to match their baseline instead of 100 %, so the gate enforces
 * "do not regress" rather than requiring an immediate full-coverage rewrite.
 *
 * @param {object} [options] - Optional configuration.
 * @param {string} [options.coverageSummaryPath] - Repo-relative path to the
 *   Istanbul coverage summary JSON. Defaults to `coverage/coverage-summary.json`.
 * @param {string} [options.coverageBaselinePath] - Repo-relative path to the
 *   committed coverage baseline JSON. Defaults to
 *   `coverage/coverage-baseline.json`.
 * @param {string[]} [options.changedFiles] - Explicit repo-relative source paths
 *   to check. When omitted, the gate derives the list from `git status`.
 * @param {Record<string, string|{kind: string}>} [options.exemptions] - Optional
 *   file-level coverage exemptions. `type-only` skips a file because it contains
 *   no executable code. `legacy-dominant` accepts the file's current coverage as
 *   the threshold when no baseline exists, preserving the existing baseline
 *   contract when one is present.
 * @returns {Promise<object>} Standard gate contract:
 *   `{ pass, evidence, fixHint, owner }`.
 */
export async function runCodeCoverageGate(options = {}) {
  const coverageSummaryPath =
    options.coverageSummaryPath ?? COVERAGE_SUMMARY_PATH;
  const coverageBaselinePath =
    options.coverageBaselinePath ?? COVERAGE_BASELINE_PATH;
  const exemptions = options.exemptions
    ? normalizeExemptions(options.exemptions)
    : {};
  const changedFiles =
    options.changedFiles ?? (await deriveChangedSourceFiles());
  const targetFiles = filterTargetFiles(changedFiles).filter(
    (file) => !isTypeOnlyExempt(file, exemptions),
  );

  let coverageSummary = null;
  try {
    coverageSummary = JSON.parse(
      await readFile(path.join(repoRoot, coverageSummaryPath), 'utf8'),
    );
  } catch (error) {
    return {
      pass: false,
      evidence: {
        coverageSummaryPath,
        error: String(error),
        targetFiles,
      },
      fixHint: `Run the test suite with coverage to generate ${coverageSummaryPath}, then merge per-project finals with merge-coverage-summaries.mjs.`,
      owner: OWNER,
    };
  }

  let baselineSummary = null;
  try {
    baselineSummary = JSON.parse(
      await readFile(path.join(repoRoot, coverageBaselinePath), 'utf8'),
    );
  } catch {
    // No baseline yet: every target file will be treated as new and must reach
    // 100 %. This is the strict bootstrap mode.
    baselineSummary = {};
  }

  if (targetFiles.length === 0) {
    return {
      pass: true,
      evidence: {
        coverageSummaryPath,
        coverageBaselinePath,
        targetFiles: [],
        message: 'No coverage-relevant source files changed.',
      },
      fixHint: null,
      owner: OWNER,
    };
  }

  const fileReports = [];
  const missingFiles = [];
  const failedFiles = [];
  const exemptFiles = Object.entries(exemptions).map(([file, exemption]) => ({
    file,
    kind: typeof exemption === 'string' ? exemption : exemption.kind,
  }));

  for (const targetFile of targetFiles) {
    const relativeKey = targetFile;
    const entry = lookupCoverageEntry(coverageSummary, relativeKey);
    const baselineEntry = lookupCoverageEntry(baselineSummary, relativeKey);

    const isLegacyDominant = isLegacyDominantExempt(targetFile, exemptions);

    if (!entry) {
      // A changed source file that is absent from the current coverage summary
      // is treated as 0 % covered. If the baseline records 0 % as well, the
      // file passes (legacy not-run code). Legacy-dominant files with no
      // baseline are also accepted because only their new/changed surface is in
      // scope, and a missing summary entry means Istanbul found no executable
      // code for them in this run.
      const missingMetrics = {
        lines: 0,
        statements: 0,
        functions: 0,
        branches: 0,
      };
      const missingThresholds = Object.fromEntries(
        REQUIRED_METRICS.map((metric) => [
          metric,
          resolveThreshold(metric, baselineEntry, null, isLegacyDominant),
        ]),
      );
      const missingAllCovered = REQUIRED_METRICS.every(
        (metric) => missingMetrics[metric] >= missingThresholds[metric],
      );
      missingFiles.push(targetFile);
      if (!missingAllCovered) {
        failedFiles.push(targetFile);
      }
      fileReports.push({
        file: targetFile,
        found: false,
        exempt: isLegacyDominant ? 'legacy-dominant' : null,
        baseline: baselineEntry
          ? Object.fromEntries(
              REQUIRED_METRICS.map((metric) => [
                metric,
                baselineEntry[metric]?.pct ?? 0,
              ]),
            )
          : null,
        metrics: missingMetrics,
        thresholds: missingThresholds,
        allCovered: missingAllCovered,
      });
      continue;
    }

    const metrics = Object.fromEntries(
      REQUIRED_METRICS.map((metric) => [metric, entry[metric]?.pct ?? 0]),
    );
    const thresholds = Object.fromEntries(
      REQUIRED_METRICS.map((metric) => [
        metric,
        resolveThreshold(metric, baselineEntry, entry, isLegacyDominant),
      ]),
    );
    const allCovered = REQUIRED_METRICS.every(
      (metric) => metrics[metric] >= thresholds[metric],
    );

    fileReports.push({
      file: targetFile,
      found: true,
      exempt: isLegacyDominant ? 'legacy-dominant' : null,
      baseline: baselineEntry
        ? Object.fromEntries(
            REQUIRED_METRICS.map((metric) => [
              metric,
              baselineEntry[metric]?.pct ?? 0,
            ]),
          )
        : null,
      metrics,
      thresholds,
      allCovered,
    });

    if (!allCovered) {
      failedFiles.push(targetFile);
    }
  }

  const pass = failedFiles.length === 0;

  return {
    pass,
    failedFiles,
    missingFiles,
    evidence: {
      coverageSummaryPath,
      coverageBaselinePath,
      targetFiles,
      exemptFiles,
      fileReports,
      missingFiles,
      failedFiles,
    },
    fixHint: pass ? null : buildFixHint(missingFiles, failedFiles),
    owner: OWNER,
  };
}

/**
 * Parses CLI arguments for this gate. Extends the shared `parseArgs` helper
 * with `--changed-files` and `--scripts` support.
 *
 * @param {string[]} argv - Argument strings from `process.argv.slice(2)`.
 * @returns {object} Parsed options including `json`, `changedFiles`, and `scripts`.
 */
function parseGateArgs(argv) {
  const base = parseArgs(argv);
  const changedFiles = [];
  let coverageSummaryPath = null;
  let coverageBaselinePath = null;
  let exemptions = null;

  for (let index = 0; index < argv.length; index++) {
    const rawArg = argv[index];
    if (rawArg.startsWith('--changed-files=')) {
      changedFiles.push(
        ...splitPathList(rawArg.slice('--changed-files='.length)),
      );
    } else if (rawArg.startsWith('--scripts=')) {
      changedFiles.push(...splitPathList(rawArg.slice('--scripts='.length)));
    } else if (rawArg === '--changed-files' || rawArg === '--scripts') {
      const nextArg = argv[index + 1];
      if (nextArg !== undefined && !nextArg.startsWith('--')) {
        changedFiles.push(...splitPathList(nextArg));
        index += 1;
      }
    } else if (rawArg.startsWith('--coverage-summary-path=')) {
      coverageSummaryPath = rawArg.slice('--coverage-summary-path='.length);
    } else if (rawArg === '--coverage-summary-path') {
      const nextArg = argv[index + 1];
      if (nextArg !== undefined && !nextArg.startsWith('--')) {
        coverageSummaryPath = nextArg;
        index += 1;
      }
    } else if (rawArg.startsWith('--coverage-baseline-path=')) {
      coverageBaselinePath = rawArg.slice('--coverage-baseline-path='.length);
    } else if (rawArg === '--coverage-baseline-path') {
      const nextArg = argv[index + 1];
      if (nextArg !== undefined && !nextArg.startsWith('--')) {
        coverageBaselinePath = nextArg;
        index += 1;
      }
    } else if (rawArg.startsWith('--exemptions=')) {
      exemptions = rawArg.slice('--exemptions='.length);
    } else if (rawArg === '--exemptions') {
      const nextArg = argv[index + 1];
      if (nextArg !== undefined && !nextArg.startsWith('--')) {
        exemptions = nextArg;
        index += 1;
      }
    }
  }

  return {
    ...base,
    changedFiles: changedFiles.length > 0 ? changedFiles : null,
    coverageSummaryPath,
    coverageBaselinePath,
    exemptions,
  };
}

/**
 * Splits a comma- or newline-separated path list into normalized repo-relative
 * entries.
 *
 * @param {string} value - Raw path list.
 * @returns {string[]} Non-empty normalized paths.
 */
function splitPathList(value) {
  return value
    .split(/[\n,]+/)
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => entry.replace(/\\/g, '/'));
}

/**
 * Resolves the required coverage threshold for a single metric.
 *
 * The default contract is 100 % for new files and the recorded baseline
 * percentage for legacy files. When a file is marked as `legacy-dominant`, the
 * threshold falls back to its current coverage percentage if no baseline
 * exists, so the gate verifies that the newly changed surface does not regress
 * the file rather than demanding whole-file 100 % coverage immediately.
 *
 * @param {string} metric - Istanbul metric name (`lines`, `statements`,
 *   `functions`, or `branches`).
 * @param {object|undefined} baselineEntry - Baseline summary entry for the file.
 * @param {object|undefined} entry - Current coverage summary entry for the file.
 * @param {boolean} isLegacyDominant - Whether the file has a legacy-dominant
 *   exemption.
 * @returns {number} Required percentage for the metric.
 */
function resolveThreshold(metric, baselineEntry, entry, isLegacyDominant) {
  if (baselineEntry) {
    return baselineEntry[metric]?.pct ?? 0;
  }
  if (isLegacyDominant && entry) {
    return entry[metric]?.pct ?? 0;
  }
  return 100;
}

/**
 * Loads and parses an exemptions value.
 *
 * Accepts either a repo-relative path to a JSON file or an inline JSON object.
 * The JSON must map repo-relative source paths to an exemption kind string
 * (`type-only` or `legacy-dominant`) or to an object with a `kind` property.
 *
 * @param {string} value - Path to a JSON file or an inline JSON object string.
 * @returns {Promise<Record<string, string|{kind: string}>>} Parsed exemptions.
 * @throws {Error} When the value is neither a readable file nor valid JSON.
 */
async function loadExemptions(value) {
  const normalized = value.replace(/\\/g, '/').trim();
  const candidatePath = path.join(repoRoot, normalized);

  let stats;
  try {
    stats = await stat(candidatePath);
  } catch {
    stats = null;
  }

  if (stats?.isFile()) {
    const contents = await readFile(candidatePath, 'utf8');
    return JSON.parse(contents);
  }

  // Inline JSON: support both single-quoted shell wrappers and bare JSON.
  const json = normalized.replace(/^'|'$/g, '');
  return JSON.parse(json);
}

/**
 * Normalizes exemptions into a consistent map of repo-relative paths to kind
 * strings.
 *
 * @param {Record<string, string|{kind: string}>} exemptions - Raw exemption map.
 * @returns {Record<string, string>} Normalized map of path → kind.
 */
function normalizeExemptions(exemptions) {
  const normalized = {};
  for (const [rawPath, value] of Object.entries(exemptions)) {
    const file = rawPath.replace(/\\/g, '/');
    const kind = typeof value === 'string' ? value : value?.kind;
    if (kind !== 'type-only' && kind !== 'legacy-dominant') {
      throw new Error(
        `Unsupported exemption kind "${kind}" for ${file}. Use "type-only" or "legacy-dominant".
`,
      );
    }
    normalized[file] = kind;
  }
  return normalized;
}

/**
 * Checks whether a file is exempt as type-only.
 *
 * @param {string} filePath - Repo-relative path.
 * @param {Record<string, string>} exemptions - Normalized exemption map.
 * @returns {boolean} True when the file is declared type-only.
 */
function isTypeOnlyExempt(filePath, exemptions) {
  return exemptions[filePath] === 'type-only';
}

/**
 * Checks whether a file is exempt as legacy-dominant.
 *
 * @param {string} filePath - Repo-relative path.
 * @param {Record<string, string>} exemptions - Normalized exemption map.
 * @returns {boolean} True when the file is declared legacy-dominant.
 */
function isLegacyDominantExempt(filePath, exemptions) {
  return exemptions[filePath] === 'legacy-dominant';
}

/**
 * Looks up a file entry in a coverage-style summary by repo-relative key,
 * falling back to the legacy absolute key for backwards compatibility.
 *
 * Istanbul/Jest historically emitted absolute paths; the merged summary and
 * baseline now use repo-relative forward-slash keys. This helper lets the gate
 * consume both formats during the transition.
 *
 * @param {Record<string, object>} summary - Coverage or baseline summary.
 * @param {string} relativeKey - Repo-relative path with forward slashes.
 * @returns {object|undefined} The matching entry, if any.
 */
function lookupCoverageEntry(summary, relativeKey) {
  return summary[relativeKey] ?? summary[path.resolve(repoRoot, relativeKey)];
}

/**
 * Derives changed source files from `git status --porcelain` and filters them
 * to the coverage-relevant directories and source extensions.
 *
 * @returns {Promise<string[]>} Repo-relative source paths.
 */
async function deriveChangedSourceFiles() {
  const spawned = spawnSync('git', ['status', '--porcelain'], {
    encoding: 'utf8',
    cwd: repoRoot,
  });

  if (spawned.status !== 0) {
    return [];
  }

  return spawned.stdout
    .split(/\r?\n/)
    .map((line) => line.slice(3).trim())
    .filter(Boolean)
    .map((entry) => entry.replace(/\\/g, '/'))
    .filter((entry) => COVERAGE_DIRS.some((dir) => entry.startsWith(dir)))
    .filter(isSourceFile);
}

/**
 * Filters an explicit file list to source files only, preserving the input order.
 *
 * @param {string[]} files - Repo-relative paths.
 * @returns {string[]} Source files with normalized slashes.
 */
function filterTargetFiles(files) {
  return files.map((entry) => entry.replace(/\\/g, '/')).filter(isSourceFile);
}

/**
 * Determines whether a repo-relative path is a recognized source file.
 *
 * Test files (`*.test.*`, `*.spec.*`, and files under `__tests__/`) are excluded
 * because the coverage summary only tracks production source coverage.
 *
 * @param {string} filePath - Repo-relative path.
 * @returns {boolean} True when the extension is `.ts`, `.tsx`, `.js`, or `.mjs`
 *   and the path does not denote a test file.
 */
function isSourceFile(filePath) {
  return (
    SOURCE_EXTENSIONS.has(path.extname(filePath)) &&
    !TEST_FILE_RE.test(filePath) &&
    !TEST_DIR_RE.test(filePath)
  );
}

/**
 * Builds a human-readable fix hint naming missing or under-covered files.
 *
 * @param {string[]} missingFiles - Files absent from the coverage summary.
 * @param {string[]} failedFiles - Files with coverage below 100 %.
 * @returns {string} Fix hint string.
 */
export function buildFixHint(missingFiles, failedFiles) {
  const parts = [];
  if (missingFiles.length > 0) {
    parts.push(
      `Missing from coverage summary: ${missingFiles.join(', ')}. Run the test suite with coverage.`,
    );
  }
  if (failedFiles.length > 0) {
    parts.push(
      `Files below 100% coverage: ${failedFiles.join(', ')}. Add focused unit tests until lines/statements/functions/branches are all 100%.`,
    );
  }
  return parts.join(' ');
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseGateArgs(argv);
  const exemptions = options.exemptions
    ? await loadExemptions(options.exemptions)
    : null;
  const report = await runCodeCoverageGate({
    changedFiles: options.changedFiles ?? undefined,
    coverageSummaryPath: options.coverageSummaryPath ?? undefined,
    coverageBaselinePath: options.coverageBaselinePath ?? undefined,
    exemptions,
  });

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(report.pass ? 'PASS' : 'FAIL', 'code-coverage gate');
    if (!report.pass) console.log('fixHint:', report.fixHint);
  }

  process.exitCode = report.pass ? 0 : 1;
  return report;
}

/* istanbul ignore next */
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  main();
}
