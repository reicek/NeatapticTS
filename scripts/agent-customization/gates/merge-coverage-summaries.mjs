/**
 * @description Merge per-project Jest coverage-summary.json files into a single
 *   coverage/coverage-summary.json without re-running tests, and optionally
 *   emit a coverage baseline for changed source files.
 *
 * Jest writes each project's `coverage-summary.json` under its own
 * `coverageDirectory`. This utility recursively scans the coverage root for
 * every `coverage-summary.json` file (except the output file itself), combines
 * per-file metrics by keeping the most favourable entry across projects,
 * recomputes the top-level total, and emits a single summary for the
 * `code-coverage` gate.
 *
 * When `--baseline[=PATH]` is passed, it also runs `git status --porcelain`,
 * extracts changed source files under `src/`, `scripts/agent-customization/`,
 * and `scripts/mcp-semantic/`, and writes a per-file coverage baseline JSON
 * preserving current coverage or recording 0% for files not present in the
 * merged summary. An explicit comma/newline separated `--source-files` list
 * bypasses `git status` and is useful when regenerating the baseline for files
 * that are already committed.
 *
 * Usage:
 *   node scripts/agent-customization/gates/merge-coverage-summaries.mjs
 *   node scripts/agent-customization/gates/merge-coverage-summaries.mjs --baseline
 *
 * @returns {void} Exits 0 on success, 1 on failure.
 */
import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { repoRoot } from '../customization-utils.mjs';

const DEFAULT_COVERAGE_DIR = path.join(repoRoot, 'coverage');
const DEFAULT_SUMMARY_PATH = path.join(
  DEFAULT_COVERAGE_DIR,
  'coverage-summary.json',
);
const SUMMARY_NAME = 'coverage-summary.json';
const METRICS = ['lines', 'statements', 'functions', 'branches'];
const COVERAGE_DIRS = [
  'src/',
  'scripts/agent-customization/',
  'scripts/mcp-semantic/',
];
const SOURCE_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.mjs']);
const TEST_FILE_RE = /\.(test|spec)\.[cm]?[jt]sx?$/iu;
const TEST_DIR_RE = /(^|\/)__tests__\//iu;

/**
 * Recursively collect every `coverage-summary.json` path under a directory,
 * excluding the eventual output path so a previous merged summary is not
 * re-merged into itself.
 *
 * @param {string} coverageDir - Absolute path to the coverage root.
 * @param {string} summaryPath - Absolute path to the merged summary being produced.
 * @returns {Promise<string[]>} Sorted list of project summary paths.
 */
async function collectSummaryPaths(coverageDir, summaryPath) {
  const summaryPaths = [];
  const queue = [coverageDir];

  while (queue.length > 0) {
    const currentDir = queue.shift();
    const entries = await readdir(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const entryPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        queue.push(entryPath);
      } else if (
        entry.isFile() &&
        entry.name === SUMMARY_NAME &&
        entryPath !== summaryPath
      ) {
        summaryPaths.push(entryPath);
      }
    }
  }

  return summaryPaths.sort();
}

/**
 * Convert an Istanbul/Jest coverage key (usually an absolute path) to a
 * repo-relative forward-slash key. The special `total` key is preserved.
 *
 * @param {string} key - Original coverage summary key.
 * @returns {string} Repo-relative key with forward slashes.
 */
export function toRepoRelativeKey(key) {
  if (key === 'total') {
    return key;
  }
  return path.relative(repoRoot, key).replace(/\\/g, '/');
}

/**
 * Determine whether a repository-relative path is a source file that should
 * participate in coverage gating.
 *
 * @param {string} filePath - Repository-relative file path.
 * @returns {boolean} True for source files under coverage roots, excluding tests.
 */
function isSourceFile(filePath) {
  const extension = path.extname(filePath);
  return (
    SOURCE_EXTENSIONS.has(extension) &&
    !TEST_FILE_RE.test(filePath) &&
    !TEST_DIR_RE.test(filePath)
  );
}

/**
 * Merge per-project Jest coverage-summary.json files found under the coverage root
 * and emit a single combined json-summary report.
 *
 * @param {object} [options] - Optional configuration.
 * @param {string} [options.coverageDir] - Absolute path to the coverage root.
 *   Defaults to repoRoot/coverage.
 * @param {string} [options.summaryPath] - Absolute path for the emitted summary.
 *   Defaults to coverageDir/coverage-summary.json.
 * @returns {Promise<{mergedFiles: string[], summaryPath: string}>}
 */
export async function mergeCoverageSummaries(options = {}) {
  const coverageDir = options.coverageDir ?? DEFAULT_COVERAGE_DIR;
  const summaryPath =
    options.summaryPath ??
    path.join(coverageDir, path.basename(DEFAULT_SUMMARY_PATH));
  const readFileImpl = options.readFile ?? readFile;
  const summaryPaths = await collectSummaryPaths(coverageDir, summaryPath);
  const mergedFiles = [];
  /** @type {Record<string, Record<string, {total: number, covered: number, skipped: number, pct: number}>>} */
  const fileEntries = {};

  for (const projectSummaryPath of summaryPaths) {
    let raw;
    try {
      raw = await readFileImpl(projectSummaryPath, 'utf8');
    } catch {
      // Project coverage may not have been generated yet; skip it.
      continue;
    }

    const data = JSON.parse(raw);
    mergedFiles.push(path.relative(repoRoot, projectSummaryPath));

    for (const [key, entry] of Object.entries(data)) {
      if (key === 'total') {
        continue;
      }
      const relativeKey = toRepoRelativeKey(key);
      const existing = fileEntries[relativeKey];
      if (!existing) {
        fileEntries[relativeKey] = entry;
        continue;
      }
      if (isBetterCoverage(entry, existing)) {
        fileEntries[relativeKey] = entry;
      }
    }
  }

  if (mergedFiles.length === 0) {
    throw new Error(
      `No ${SUMMARY_NAME} files found under ${coverageDir}. Run Jest with coverage first.`,
    );
  }

  await mkdir(coverageDir, { recursive: true });

  const total = computeTotal(fileEntries);
  const merged = { total, ...fileEntries };

  await writeFile(summaryPath, JSON.stringify(merged, null, 2));

  return {
    mergedFiles,
    summaryPath: path.relative(repoRoot, summaryPath),
  };
}

/**
 * Generate a per-file coverage baseline JSON for source files reported as
 * changed by `git status --porcelain`.
 *
 * @param {object} [options] - Optional configuration.
 * @param {string} [options.coverageSummaryPath] - Path to the merged coverage summary.
 * @param {string} [options.baselinePath] - Path for the emitted baseline JSON.
 * @param {Function} [options.spawnSync] - `child_process.spawnSync` seam for testing.
 * @param {Function} [options.readFile] - `fs.promises.readFile` seam for testing.
 * @param {Function} [options.writeFile] - `fs.promises.writeFile` seam for testing.
 * @param {string} [options.sourceFiles] - Comma/newline separated repo-relative
 *   source paths. When provided, `git status` is skipped.
 * @returns {Promise<{baselinePath: string, files: number, zeroFiles: number}>}
 */
export async function generateCoverageBaseline(options = {}) {
  const coverageSummaryPath =
    options.coverageSummaryPath ?? DEFAULT_SUMMARY_PATH;
  const baselinePath =
    options.baselinePath ??
    path.join(DEFAULT_COVERAGE_DIR, 'coverage-baseline.json');
  const spawnSyncImpl = options.spawnSync ?? spawnSync;
  const readFileImpl = options.readFile ?? readFile;
  const writeFileImpl = options.writeFile ?? writeFile;

  let changedFiles;
  if (options.sourceFiles) {
    changedFiles = splitSourceFiles(options.sourceFiles);
  } else {
    const spawned = spawnSyncImpl('git', ['status', '--porcelain'], {
      cwd: repoRoot,
      encoding: 'utf8',
    });

    if (spawned.status !== 0) {
      throw new Error(
        `git status failed: ${spawned.stderr?.trim() ?? 'unknown error'}`,
      );
    }

    changedFiles = spawned.stdout
      .split(/\r?\n/)
      .map((line) => line.slice(3).trim())
      .filter(Boolean)
      .map((entry) => entry.replace(/\\/g, '/'))
      .filter((entry) => COVERAGE_DIRS.some((dir) => entry.startsWith(dir)))
      .filter(isSourceFile);
  }

  const summary = JSON.parse(await readFileImpl(coverageSummaryPath, 'utf8'));

  /** @type {Record<string, Record<string, {total: number, covered: number, skipped: number, pct: number}>>} */
  const baseline = {};
  let zeroFiles = 0;

  for (const file of changedFiles) {
    const relativeKey = file;
    const entry = summary[relativeKey] ?? summary[path.resolve(repoRoot, file)];
    if (entry) {
      baseline[relativeKey] = {};
      for (const metric of METRICS) {
        const metricEntry = entry[metric];
        baseline[relativeKey][metric] = {
          total: metricEntry?.total ?? 0,
          covered: metricEntry?.covered ?? 0,
          skipped: metricEntry?.skipped ?? 0,
          pct: metricEntry?.pct ?? 0,
        };
      }
    } else {
      zeroFiles += 1;
      baseline[relativeKey] = {};
      for (const metric of METRICS) {
        baseline[relativeKey][metric] = {
          total: 0,
          covered: 0,
          skipped: 0,
          pct: 0,
        };
      }
    }
  }

  await writeFileImpl(baselinePath, JSON.stringify(baseline, null, 2));

  return {
    baselinePath: path.relative(repoRoot, baselinePath),
    files: changedFiles.length,
    zeroFiles,
  };
}

/**
 * Split an explicit `--source-files` value into normalized repo-relative paths.
 *
 * @param {string} value - Comma or newline separated path list.
 * @returns {string[]} Non-empty normalized paths.
 */
function splitSourceFiles(value) {
  return value
    .split(/[\n,]+/)
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => entry.replace(/\\/g, '/'));
}

/**
 * Determine whether `candidate` coverage is better than `current`.
 *
 * "Better" means a higher statement percentage, breaking ties by higher line
 * percentage. This is intentionally conservative: a file appearing in multiple
 * projects should use the most favourable project view.
 *
 * @param {Record<string, {pct: number}>} candidate - Candidate file entry.
 * @param {Record<string, {pct: number}>} current - Existing merged entry.
 * @returns {boolean} True when the candidate should replace the current entry.
 */
function isBetterCoverage(candidate, current) {
  const candidateStatements = candidate.statements?.pct ?? 0;
  const currentStatements = current.statements?.pct ?? 0;
  if (candidateStatements !== currentStatements) {
    return candidateStatements > currentStatements;
  }
  return (candidate.lines?.pct ?? 0) > (current.lines?.pct ?? 0);
}

/**
 * Recompute the top-level total by summing counts from every merged file entry.
 *
 * @param {Record<string, Record<string, {total: number, covered: number, skipped: number, pct: number}>>} fileEntries
 * @returns {Record<string, {total: number, covered: number, skipped: number, pct: number}>}
 */
function computeTotal(fileEntries) {
  const total = {};
  for (const metric of METRICS) {
    total[metric] = { total: 0, covered: 0, skipped: 0, pct: 100 };
  }

  for (const entry of Object.values(fileEntries)) {
    for (const metric of METRICS) {
      const metricEntry = entry[metric];
      if (!metricEntry) {
        continue;
      }
      total[metric].total += metricEntry.total ?? 0;
      total[metric].covered += metricEntry.covered ?? 0;
      total[metric].skipped += metricEntry.skipped ?? 0;
    }
  }

  for (const metric of METRICS) {
    const { total: t, covered: c } = total[metric];
    total[metric].pct = t === 0 ? 100 : Math.floor((c / t) * 10000) / 100;
  }

  total.branchesTrue = { total: 0, covered: 0, skipped: 0, pct: 100 };
  return total;
}

export function parseCliOptions(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const rawArg = argv[index];
    if (rawArg === '--coverage-dir' && argv[index + 1] !== undefined) {
      options.coverageDir = path.resolve(argv[++index]);
    } else if (rawArg.startsWith('--coverage-dir=')) {
      options.coverageDir = path.resolve(
        rawArg.slice('--coverage-dir='.length),
      );
    } else if (rawArg === '--summary-path' && argv[index + 1] !== undefined) {
      options.summaryPath = path.resolve(argv[++index]);
    } else if (rawArg.startsWith('--summary-path=')) {
      options.summaryPath = path.resolve(
        rawArg.slice('--summary-path='.length),
      );
    } else if (rawArg === '--baseline') {
      const nextArg = argv[index + 1];
      if (nextArg !== undefined && !nextArg.startsWith('--')) {
        options.baselinePath = path.resolve(nextArg);
        index += 1;
      } else {
        options.baselinePath = path.join(
          DEFAULT_COVERAGE_DIR,
          'coverage-baseline.json',
        );
      }
    } else if (rawArg.startsWith('--baseline=')) {
      options.baselinePath = path.resolve(rawArg.slice('--baseline='.length));
    } else if (rawArg === '--source-files' && argv[index + 1] !== undefined) {
      options.sourceFiles = argv[++index];
    } else if (rawArg.startsWith('--source-files=')) {
      options.sourceFiles = rawArg.slice('--source-files='.length);
    }
  }
  return options;
}

const isMainEntry =
  process.argv[1] !== undefined &&
  pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;

export async function main(argv = process.argv.slice(2), deps = {}) {
  const cliOptions = parseCliOptions(argv);
  const mergeResult = await mergeCoverageSummaries(cliOptions);
  if (cliOptions.baselinePath !== undefined) {
    await generateCoverageBaseline({
      coverageSummaryPath: path.resolve(repoRoot, mergeResult.summaryPath),
      baselinePath: cliOptions.baselinePath,
      spawnSync: deps.spawnSync,
      readFile: deps.readFile,
      writeFile: deps.writeFile,
      sourceFiles: cliOptions.sourceFiles,
    });
  }
  return mergeResult;
}

export function printMergeResult(result) {
  console.log(JSON.stringify(result, null, 2));
}

export function printMergeError(error) {
  console.error(error.message);
  process.exitCode = 1;
}

/* istanbul ignore next - CLI entry path is exercised in subprocess integration tests and cannot be re-evaluated under Jest's ESM cache. */
if (isMainEntry) {
  main().then(printMergeResult, printMergeError);
}
