/**
 * @module update-rag
 * @description Single-file CLI orchestrator for the NeatapticTS RAG pipeline.
 *
 * Chains the canonical RAG stages in order:
 *   build → prewarm-embed → build-terms → build-graph → snapshot → validate
 *
 * Supports `--dry-run`, `--validate`, and `--json`, tracks per-stage status and
 * elapsed time, publishes the per-stage peak memory measured from the stage process
 * itself, and skips the `build-graph` stage when the corpus hash is unchanged.
 *
 * @param {boolean} [--dry-run] - List stages without mutating DB/models/snapshot.
 * @param {boolean} [--validate] - Compatibility flag; the validate stage is always listed.
 * @param {boolean} [--json] - Emit a JSON summary object to stdout.
 * @param {boolean} [--help] - Show usage and exit.
 *
 * @returns {void} Exits 0 on success, 1 on any stage failure.
 */
/* global process */
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createClient } from '@libsql/client';

import { parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

/**
 * Canonical RAG pipeline stages in execution order.
 *
 * @type {Array<{ name: string, script: string, supportsDryRun: boolean }>}
 */
const STAGE_DEFINITIONS = Object.freeze([
  { name: 'build', script: 'build-index.mjs', supportsDryRun: true },
  { name: 'prewarm-embed', script: 'prewarm-dense.mjs', supportsDryRun: true },
  { name: 'build-terms', script: 'build-term-index.mjs', supportsDryRun: true },
  {
    name: 'build-graph',
    script: 'build-entity-graph.mjs',
    supportsDryRun: true,
  },
  {
    name: 'snapshot',
    script: 'build-browser-snapshot.mjs',
    supportsDryRun: true,
  },
  { name: 'validate', script: 'validate-index.mjs', supportsDryRun: false },
]);

/** Path where the last-run corpus hash is persisted. */
const CORPUS_HASH_PATH = path.join(
  repoRoot,
  'rag-index',
  'data',
  'corpus-hash.json',
);

/**
 * The file descriptor a stage process reports its own peak RSS on.
 *
 * The descriptor is a DEDICATED pipe, so the measurement never contaminates the stage's own
 * stdout/stderr - the runner surfaces those verbatim when a stage fails.
 *
 * @type {number}
 */
const PEAK_REPORT_FD = 3;

/**
 * The marker key a stage process writes its own peak RSS under.
 *
 * @type {string}
 */
const PEAK_MARKER_KEY = 'PEAK_RSS_BYTES';

/**
 * The spawn-time probe the runner PRELOADS into every stage process.
 *
 * The probe is loaded INTO the stage process itself (`node --import <probe> <stage script>`),
 * so the figure it reports is the peak RSS of the STAGE PROCESS. The runner never estimates a
 * stage's peak from its own resource usage: a parent process cannot observe a child's peak
 * RSS, so the child has to report its own.
 *
 * The probe samples `process.resourceUsage().maxRSS` when the stage process exits, which is
 * kilobytes on Linux and Windows and bytes on macOS.
 *
 * @type {string}
 */
const PEAK_PROBE_SOURCE = `import { writeSync } from 'node:fs';
process.on('exit', () => {
  try {
    const usage = process.resourceUsage();
    const peakBytes = process.platform === 'darwin' ? usage.maxRSS : usage.maxRSS * 1024;
    writeSync(${PEAK_REPORT_FD}, '${PEAK_MARKER_KEY}=' + Math.round(peakBytes) + '\\n');
  } catch {}
});
`;

/**
 * The data URL of the spawn-time probe.
 *
 * A data URL keeps the probe INSIDE the runner module: the runner owns the measurement
 * end-to-end without adding a second file to the repository.
 *
 * @type {string}
 */
const PEAK_PROBE_MODULE_URL = `data:text/javascript,${encodeURIComponent(
  PEAK_PROBE_SOURCE,
)}`;

/**
 * The property names a stage runner may publish its peak-memory figure under.
 *
 * The runner result is an injection seam, so the figure is accepted under any of the
 * documented aliases; the runner's own measurement publishes `peakMemoryBytes`.
 *
 * @type {ReadonlyArray<string>}
 */
const PEAK_FIGURE_ALIASES = Object.freeze([
  'peakMemoryBytes',
  'peakRssBytes',
  'peakBytes',
  'peakMemory',
  'maxRssBytes',
  'maxRSS',
]);

/**
 * The key pattern a peak-memory figure is recognised by in a JSON payload or a marker line.
 *
 * @type {RegExp}
 */
const PEAK_FIGURE_KEY_PATTERN = /peak|rss|memory|resident|highwater|hwm/i;

/**
 * Run the full RAG pipeline or a dry-run preview.
 *
 * Chains the canonical RAG stages in order, supports idempotent graph building
 * via a corpus hash, and returns a per-stage status report.
 *
 * Every stage report carries the stage's PEAK MEMORY figure in bytes, measured from the
 * stage PROCESS itself: the runner preloads a spawn-time probe into the stage process and
 * the stage process reports its own `process.resourceUsage().maxRSS` on a dedicated
 * descriptor. The figure is NEVER an estimate taken from the parent runner and NEVER a
 * hand-written number.
 *
 * A stage that RAN but published NO figure carries the explicit `peakMeasurementUnavailable:
 * true` marker instead. The marker separates a stage whose process died before it could
 * report (a fatal abort never fires the probe's exit hook, so it publishes nothing) from a
 * stage that never ran at all: a skipped, cancelled or dry-run stage carries NEITHER the
 * figure NOR the marker.
 *
 * @param {object} [options={}] - Execution options and test doubles.
 * @param {boolean} [options.dryRun] - List stages without mutating DB/models/snapshot.
 * @param {boolean} [options.validate] - Compatibility flag; validate stage always runs.
 * @param {boolean} [options.json] - Emit machine-readable summary (used by callers).
 * @param {(stage: { name: string, script: string, args: string[] }) => { status?: number | null, stdout?: string, stderr?: string, peakMemoryBytes?: number }} [options.spawnRunner] - Injectable stage runner for tests.
 * @param {() => Promise<string | null>} [options.readHashFile] - Injectable stored hash reader for tests.
 * @param {(hash: string) => Promise<void>} [options.writeHashFile] - Injectable hash writer for tests.
 * @param {() => Promise<string | null>} [options.computeCorpusHash] - Injectable hash computer for tests.
 * @returns {Promise<{ ok: boolean, pass: boolean, stages: Array<{ name: string, status: string, elapsedMs: number, peakMemoryBytes?: number, peakMeasurementUnavailable?: boolean }>, error?: string }>} Execution summary.
 */
export async function updateRag(options) {
  options = /* istanbul ignore next -- defensive: options always provided in tests */ options ?? {};
  const dryRun = Boolean(options.dryRun);
  const spawnRunner = options.spawnRunner ?? runStageSubprocess;
  const readHashFile = options.readHashFile ?? readStoredCorpusHash;
  const writeHashFile = options.writeHashFile ?? writeCorpusHash;
  const computeHashFn = options.computeCorpusHash ?? computeCorpusHash;

  /** @type {Array<{ name: string, status: string, elapsedMs: number, peakMemoryBytes?: number, peakMeasurementUnavailable?: boolean }>} */
  const stages = [];
  let pipelineError = null;

  for (const stageDefinition of STAGE_DEFINITIONS) {
    if (pipelineError) {
      stages.push({
        name: stageDefinition.name,
        status: 'cancelled',
        elapsedMs: 0,
      });
      continue;
    }

    if (stageDefinition.name === 'build-graph') {
      const currentHash = await computeHashFn();
      const storedHash = await readHashFile();

      if (currentHash && storedHash && currentHash === storedHash) {
        stages.push({
          name: stageDefinition.name,
          status: 'skipped',
          elapsedMs: 0,
        });
        continue;
      }
    }

    if (dryRun && !stageDefinition.supportsDryRun) {
      stages.push({
        name: stageDefinition.name,
        status: 'dry-run',
        elapsedMs: 0,
      });
      continue;
    }

    const stageArgs = ['--json'];
    if (dryRun && stageDefinition.supportsDryRun) {
      stageArgs.push('--dry-run');
    }

    const startTime = Date.now();
    let stageStatus = dryRun ? 'dry-run' : 'ok';
    let stageError = null;
    let stageStdout = '';
    let stageStderr = '';
    /** @type {{ status?: number | null, stdout?: string, stderr?: string } | null} */
    let stageResult = null;

    try {
      stageResult = spawnRunner({
        name: stageDefinition.name,
        script: stageDefinition.script,
        args: stageArgs,
      });

      stageStdout = String(stageResult?.stdout ?? '');
      stageStderr = String(stageResult?.stderr ?? '');
      const exitStatus = Number(stageResult?.status ?? 1);
      if (exitStatus !== 0) {
        stageStatus = 'failed';
        stageError = `${stageDefinition.name} exited with status ${exitStatus}.`;
        if (stageStdout.trim() || stageStderr.trim()) {
          stageError += `\n--- ${stageDefinition.name} stdout ---\n${stageStdout.trim()}`;
          stageError += `\n--- ${stageDefinition.name} stderr ---\n${stageStderr.trim()}`;
        }
      }
    } catch (error) {
      stageStatus = 'failed';
      stageError = error instanceof Error ? error.message : String(error);
    }

    const elapsedMs = Date.now() - startTime;
    const stageReport = {
      name: stageDefinition.name,
      status: stageStatus,
      elapsedMs,
    };
    // Step: publish the stage's OWN peak memory, measured from the stage process itself.
    const peakMemoryBytes = readPublishedPeakMemoryBytes(stageResult);
    if (peakMemoryBytes !== undefined) {
      stageReport.peakMemoryBytes = peakMemoryBytes;
    } else {
      // Step: mark the measurement as UNAVAILABLE. Only a stage that RAN reaches this point -
      // a skipped, cancelled or dry-run stage is published before it - so the marker tells a
      // stage that ran but published NO figure (a fatal abort never fires the probe's exit
      // hook, so it publishes none) apart from a stage that never ran at all.
      stageReport.peakMeasurementUnavailable = true;
    }
    if (stageStatus === 'failed') {
      stageReport.stdout = stageStdout;
      stageReport.stderr = stageStderr;
    }
    stages.push(stageReport);

    if (stageStatus === 'failed') {
      pipelineError = /* istanbul ignore next -- defensive: stageError always provided when stage fails */ (stageError ?? `${stageDefinition.name} failed.`);
    }

    if (
      stageDefinition.name === 'build-graph' &&
      stageStatus !== 'failed' &&
      stageStatus !== 'skipped' &&
      !dryRun
    ) {
      const currentHash = await computeHashFn();
      if (currentHash) {
        await writeHashFile(currentHash);
      }
    }
  }

  const ok = pipelineError === null;
  const summary = {
    ok,
    pass: ok,
    stages,
  };

  if (!ok) {
    summary.error = pipelineError;
  }

  return summary;
}

/**
 * Compose the per-row identity keys of a corpus.
 *
 * Every identity row carries the document's path, its content hash, its modification time,
 * its file size AND its FAMILY ASSIGNMENT, in that order:
 * `file_path|sha256|mtime_ms|file_size|doc_family`. Carrying the family assignment closes the
 * pipeline-identity hole behind `V-OBS-011`: a family re-label that has landed in the
 * documents table now changes the corpus identity, so the entity graph can no longer be
 * skipped across a family rename.
 *
 * The rows are sorted by `file_path`, so the identity does not depend on the order the rows
 * arrive in. The function is PURE: it never touches the database, the filesystem or the
 * clock, so the identity semantics can be pinned without touching the live index.
 *
 * @param {Array<{ file_path?: unknown, sha256?: unknown, mtime_ms?: unknown, file_size?: unknown, doc_family?: unknown }>} rows - The corpus rows to compose.
 * @returns {string[]} The identity rows, sorted by `file_path`.
 *
 * @example
 * ```js
 * composeCorpusIdentityRows([
 *   { file_path: 'src/example.ts', sha256: 'aa', mtime_ms: 1, file_size: 2, doc_family: 'ts-source' },
 * ]);
 * // => ['src/example.ts|aa|1|2|ts-source']
 * ```
 */
export function composeCorpusIdentityRows(rows) {
  const source = Array.isArray(rows) ? rows : [];
  const normalized = source.map((row) => ({
    file_path: String(row?.file_path ?? ''),
    sha256: String(row?.sha256 ?? ''),
    mtime_ms: Number(row?.mtime_ms ?? 0),
    file_size: Number(row?.file_size ?? 0),
    doc_family: String(row?.doc_family ?? ''),
  }));

  return normalized
    .toSorted((left, right) => {
      if (left.file_path === right.file_path) return 0;
      return left.file_path < right.file_path ? -1 : 1;
    })
    .map(
      (row) =>
        `${row.file_path}|${row.sha256}|${row.mtime_ms}|${row.file_size}|${row.doc_family}`,
    );
}

/**
 * Compute the identity of a corpus.
 *
 * The identity is the SHA-256 digest of the corpus' identity rows, so it is stable for a
 * given corpus and changes whenever a row changes - including when a row changes ONLY its
 * family assignment. The digest is what the runner stores in the corpus hash file and what
 * the idempotent build-graph skip compares.
 *
 * The function is PURE and EXPORTED so the semantics can be pinned without touching the
 * live index.
 *
 * @param {Array<{ file_path?: unknown, sha256?: unknown, mtime_ms?: unknown, file_size?: unknown, doc_family?: unknown }>} rows - The corpus rows to identify.
 * @returns {string} The hex digest of the corpus identity.
 *
 * @example
 * ```js
 * const digest = computeCorpusIdentity([
 *   { file_path: 'src/example.ts', sha256: 'aa', mtime_ms: 1, file_size: 2, doc_family: 'ts-source' },
 * ]);
 * console.log(digest.length); // => 64
 * ```
 */
export function computeCorpusIdentity(rows) {
  const hash = createHash('sha256');
  for (const row of composeCorpusIdentityRows(rows)) {
    hash.update(`${row}\n`);
  }
  return hash.digest('hex');
}

/**
 * Compute a stable SHA-256 hash of the current corpus.
 *
 * Reads all rows from the `documents` table (file_path, sha256, mtime_ms, file_size,
 * doc_family), sorted by file_path, and hashes them through the pure identity composition.
 *
 * @returns {Promise<string | null>} Hex digest of the corpus hash, or null when the database is unreachable.
 */
async function computeCorpusHash() {
  try {
    /* istanbul ignore next -- defensive: DB always exists in test env */
    if (!existsSync(defaultDatabasePath)) {
      return null;
    }

    const client = createClient({
      url: pathToFileURL(path.resolve(defaultDatabasePath)).href,
    });

    const result = await client.execute({
      sql: 'SELECT file_path, sha256, mtime_ms, file_size, doc_family FROM documents ORDER BY file_path',
      args: [],
    });

    await client.close();

    return computeCorpusIdentity(result.rows);
  } catch {
    /* istanbul ignore next -- defensive: query failure on corrupted DB */
    return null;
  }
}

/**
 * Read the previously persisted corpus hash, if any.
 *
 * @returns {Promise<string | null>} Stored hash digest, or null when missing/invalid.
 */
async function readStoredCorpusHash() {
  try {
    const content = await readFile(CORPUS_HASH_PATH, 'utf8');
    const parsed = JSON.parse(content);
    return /* istanbul ignore next -- defensive: hash is always a valid string in test fixtures */ (typeof parsed.hash === 'string' ? parsed.hash : null);
  } catch {
    /* istanbul ignore next -- defensive: hash file may not exist in test env */
    return null;
  }
}

/**
 * Persist the corpus hash for idempotent graph building.
 *
 * @param {string} hash - Corpus hash digest.
 * @returns {Promise<void>}
 */
/* istanbul ignore next -- internal function, tested via dependency injection */
async function writeCorpusHash(hash) {
  await mkdir(path.dirname(CORPUS_HASH_PATH), { recursive: true });
  await writeFile(
    CORPUS_HASH_PATH,
    `${JSON.stringify({ hash }, null, 2)}\n`,
    'utf8',
  );
}

/**
 * Normalise a candidate peak-memory figure into a positive number of bytes.
 *
 * @param {unknown} value - The candidate figure.
 * @returns {number | undefined} The figure in bytes, or undefined when it is not a usable figure.
 */
function toPeakFigureBytes(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return undefined;
  }
  return Math.round(value);
}

/**
 * Read the peak-memory figure a stage runner published on its result object.
 *
 * The figure is accepted under any of the documented aliases, at the top level of the
 * result, or one level deep inside a nested payload. Strings are never descended into: a
 * stage's stdout/stderr text is not a figure.
 *
 * @param {unknown} result - The stage runner result.
 * @returns {number | undefined} The figure in bytes, or undefined when the runner published none.
 */
function readPeakFigureFromResult(result) {
  if (!result || typeof result !== 'object') {
    return undefined;
  }

  const record = /** @type {Record<string, unknown>} */ (result);
  for (const alias of PEAK_FIGURE_ALIASES) {
    const direct = toPeakFigureBytes(record[alias]);
    if (direct !== undefined) {
      return direct;
    }
  }

  return (
    readFlatPeakFigure(record) ?? readNestedPeakFigure(record)
  );
}

/**
 * Read the peak-memory figure a JSON payload printed by a stage.
 *
 * The payload is scanned for a peak-like key holding a positive number, at the top level of
 * the payload and one level deep inside a nested payload.
 *
 * @param {string | undefined | null} text - The JSON text.
 * @returns {number | undefined} The figure in bytes, or undefined when the payload carries none.
 */
function readPeakFigureFromJsonPayload(text) {
  if (typeof text !== 'string' || text.trim() === '') {
    return undefined;
  }

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return undefined;
  }

  if (!parsed || typeof parsed !== 'object') {
    return undefined;
  }

  const record = /** @type {Record<string, unknown>} */ (parsed);
  return readFlatPeakFigure(record) ?? readNestedPeakFigure(record);
}

/**
 * Read the first peak-memory figure published under a peak-like key of a payload.
 *
 * @param {Record<string, unknown>} record - The payload to scan.
 * @returns {number | undefined} The figure in bytes, or undefined when the payload carries none.
 */
function readFlatPeakFigure(record) {
  for (const [key, value] of Object.entries(record)) {
    if (!PEAK_FIGURE_KEY_PATTERN.test(key)) continue;
    const flat = toPeakFigureBytes(value);
    if (flat !== undefined) {
      return flat;
    }
  }
  return undefined;
}

/**
 * Read the first peak-memory figure published one level deep inside a payload.
 *
 * The nested payload is scanned whatever its own key is, so a figure published under a
 * neutral container key (for example `metrics`) is still found. Strings are never
 * descended into.
 *
 * @param {Record<string, unknown>} record - The payload to scan.
 * @returns {number | undefined} The figure in bytes, or undefined when the payload carries none.
 */
function readNestedPeakFigure(record) {
  for (const value of Object.values(record)) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      continue;
    }
    for (const [nestedKey, nestedValue] of Object.entries(
      /** @type {Record<string, unknown>} */ (value),
    )) {
      if (!PEAK_FIGURE_KEY_PATTERN.test(nestedKey)) continue;
      const nested = toPeakFigureBytes(nestedValue);
      if (nested !== undefined) {
        return nested;
      }
    }
  }
  return undefined;
}

/**
 * Read the peak-memory figure from a `KEY=VALUE` marker printed by a stage.
 *
 * @param {string | undefined | null} text - The marker text.
 * @returns {number | undefined} The figure in bytes, or undefined when the text carries no marker.
 */
function readPeakFigureFromMarkerText(text) {
  if (typeof text !== 'string' || text === '') {
    return undefined;
  }

  for (const line of text.split(/\r?\n/)) {
    const separatorIndex = line.indexOf('=');
    if (separatorIndex <= 0) continue;
    const key = line.slice(0, separatorIndex).trim();
    if (!PEAK_FIGURE_KEY_PATTERN.test(key)) continue;
    const figure = toPeakFigureBytes(Number(line.slice(separatorIndex + 1).trim()));
    if (figure !== undefined) {
      return figure;
    }
  }

  return undefined;
}

/**
 * Read the peak-memory figure a stage published, wherever the stage published it.
 *
 * The lookup order is: the runner result object, the stage's stdout payload, then the
 * stage's stderr marker channel.
 *
 * @param {unknown} result - The stage runner result.
 * @returns {number | undefined} The figure in bytes, or undefined when the stage published none.
 */
function readPublishedPeakMemoryBytes(result) {
  return (
    readPeakFigureFromResult(result) ??
    readPeakFigureFromJsonPayload(
      typeof result === 'object' && result !== null
        ? /** @type {{ stdout?: unknown }} */ (result).stdout
        : undefined,
    ) ??
    readPeakFigureFromMarkerText(
      typeof result === 'object' && result !== null
        ? /** @type {{ stderr?: unknown }} */ (result).stderr
        : undefined,
    )
  );
}

/**
 * Read the peak-memory marker a stage process reported on its dedicated descriptor.
 *
 * @param {unknown} channel - The captured descriptor output.
 * @returns {number | undefined} The figure in bytes, or undefined when the stage reported none.
 */
function readPeakMarker(channel) {
  if (typeof channel !== 'string' || channel.trim() === '') {
    return undefined;
  }
  return readPeakFigureFromMarkerText(channel);
}

/**
 * Run a Node script as a MEASURED subprocess.
 *
 * The runner preloads a spawn-time probe INTO the stage process, so the stage process
 * reports its OWN peak RSS on a dedicated descriptor. The parent runner cannot observe a
 * child's peak RSS - its own resource usage does not include a reaped child's peak - so
 * the measurement has to happen inside the stage process. The figure returned here is
 * therefore a runner-owned measurement of the stage process, never an estimate and never
 * a hand-written number.
 *
 * @param {string[]} nodeArgs - The script path and its arguments.
 * @param {{ cwd?: string }} [options={}] - The spawn options.
 * @returns {{ status: number | null, stdout: string, stderr: string, peakMemoryBytes?: number }} The measured result.
 *
 * @example
 * ```js
 * const measured = runMeasuredSubprocess(['-e', 'console.log(1)']);
 * console.log(measured.peakMemoryBytes); // => a positive byte figure
 * ```
 */
export function runMeasuredSubprocess(nodeArgs, options = {}) {
  const result = spawnSync(
    process.execPath,
    ['--import', PEAK_PROBE_MODULE_URL, ...nodeArgs],
    {
      cwd: options.cwd ?? repoRoot,
      encoding: 'utf8',
      stdio: ['inherit', 'pipe', 'pipe', 'pipe'],
    },
  );

  const peakMemoryBytes = readPeakMarker(
    Array.isArray(result.output) ? result.output[PEAK_REPORT_FD] : undefined,
  );

  return {
    status: result.status,
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
    ...(peakMemoryBytes === undefined ? {} : { peakMemoryBytes }),
  };
}

/**
 * Run one RAG stage as a measured subprocess.
 *
 * @param {{ name: string, script: string, args: string[] }} stage - The stage to run.
 * @returns {{ status: number | null, stdout: string, stderr: string, peakMemoryBytes?: number }} The measured result.
 */
function runStageSubprocess(stage) {
  const scriptPath = path.join(repoRoot, 'rag-index', stage.script);
  return runMeasuredSubprocess([scriptPath, ...stage.args]);
}

/**
 * CLI entrypoint for the RAG pipeline orchestrator.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const args = parseCliArgs(process.argv.slice(2));

  if (args.help) {
    printHelp({
      title: 'RAG pipeline orchestrator',
      usage: 'node rag-index/update-rag.mjs [--dry-run] [--validate] [--json]',
      options: [
        '--dry-run  List stages without mutating DB/models/snapshot.',
        '--validate Compatibility flag; validate stage is always included.',
        '--json     Emit a JSON summary object to stdout.',
        '--help     Show this help.',
      ],
    });
    return;
  }

  const summary = await updateRag({
    dryRun: Boolean(args['dry-run']),
    validate: Boolean(args.validate),
    json: Boolean(args.json),
  });

  writeJsonOrText(summary, Boolean(args.json), /* istanbul ignore next -- text formatter covered when writeJsonOrText is not mocked */ (payload) =>
    [
      ...payload.stages.map(
        (stage) => `${stage.name}: ${stage.status} (${stage.elapsedMs}ms)`,
      ),
      payload.ok ? 'pass' : `fail: ${payload.error}`,
    ].join('\n'),
  );

  /* istanbul ignore next -- defensive: summary.ok is always true in test invocations */
  if (summary.ok) process.exitCode = 0; else process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
