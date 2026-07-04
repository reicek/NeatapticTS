/**
 * @module update-rag
 * @description Single-file CLI orchestrator for the NeatapticTS RAG pipeline.
 *
 * Chains the canonical RAG stages in order:
 *   build → prewarm-embed → build-terms → build-graph → snapshot → validate
 *
 * Supports `--dry-run`, `--validate`, and `--json`, tracks per-stage status and
 * elapsed time, and skips the `build-graph` stage when the corpus hash is
 * unchanged.
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
 * Run the full RAG pipeline or a dry-run preview.
 *
 * Chains the canonical RAG stages in order, supports idempotent graph building
 * via a corpus hash, and returns a per-stage status report.
 *
 * @param {object} [options={}] - Execution options and test doubles.
 * @param {boolean} [options.dryRun] - List stages without mutating DB/models/snapshot.
 * @param {boolean} [options.validate] - Compatibility flag; validate stage always runs.
 * @param {boolean} [options.json] - Emit machine-readable summary (used by callers).
 * @param {(stage: { name: string, script: string, args: string[] }) => { status?: number | null, error?: Error }} [options.spawnRunner] - Injectable stage runner for tests.
 * @param {() => Promise<string | null>} [options.readHashFile] - Injectable stored hash reader for tests.
 * @param {(hash: string) => Promise<void>} [options.writeHashFile] - Injectable hash writer for tests.
 * @param {() => Promise<string | null>} [options.computeCorpusHash] - Injectable hash computer for tests.
 * @returns {Promise<{ ok: boolean, pass: boolean, stages: Array<{ name: string, status: string, elapsedMs: number }>, error?: string }>} Execution summary.
 */
export async function updateRag(options = {}) {
  const dryRun = Boolean(options.dryRun);
  const spawnRunner = options.spawnRunner ?? runStageSubprocess;
  const readHashFile = options.readHashFile ?? readStoredCorpusHash;
  const writeHashFile = options.writeHashFile ?? writeCorpusHash;
  const computeHashFn = options.computeCorpusHash ?? computeCorpusHash;

  /** @type {Array<{ name: string, status: string, elapsedMs: number }>} */
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

    try {
      const result = spawnRunner({
        name: stageDefinition.name,
        script: stageDefinition.script,
        args: stageArgs,
      });

      const exitStatus = Number(result?.status ?? 1);
      if (exitStatus !== 0) {
        stageStatus = 'failed';
        stageError = `${stageDefinition.name} exited with status ${exitStatus}.`;
      }
    } catch (error) {
      stageStatus = 'failed';
      stageError = error instanceof Error ? error.message : String(error);
    }

    const elapsedMs = Date.now() - startTime;
    stages.push({
      name: stageDefinition.name,
      status: stageStatus,
      elapsedMs,
    });

    if (stageStatus === 'failed') {
      pipelineError = stageError ?? `${stageDefinition.name} failed.`;
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
 * Compute a stable SHA-256 hash of the current corpus.
 *
 * Reads all rows from the `documents` table (file_path, sha256, mtime_ms,
 * file_size), sorted by file_path, and hashes a stable concatenation.
 *
 * @returns {Promise<string | null>} Hex digest of the corpus hash, or null when the database is unreachable.
 */
async function computeCorpusHash() {
  try {
    if (!existsSync(defaultDatabasePath)) {
      return null;
    }

    const client = createClient({
      url: pathToFileURL(path.resolve(defaultDatabasePath)).href,
    });

    const result = await client.execute({
      sql: 'SELECT file_path, sha256, mtime_ms, file_size FROM documents ORDER BY file_path',
      args: [],
    });

    await client.close();

    const hash = createHash('sha256');
    for (const row of result.rows) {
      hash.update(
        `${String(row.file_path)}|${String(row.sha256)}|${Number(row.mtime_ms)}|${Number(row.file_size)}\n`,
      );
    }

    return hash.digest('hex');
  } catch {
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
    return typeof parsed.hash === 'string' ? parsed.hash : null;
  } catch {
    return null;
  }
}

/**
 * Persist the corpus hash for idempotent graph building.
 *
 * @param {string} hash - Corpus hash digest.
 * @returns {Promise<void>}
 */
async function writeCorpusHash(hash) {
  await mkdir(path.dirname(CORPUS_HASH_PATH), { recursive: true });
  await writeFile(
    CORPUS_HASH_PATH,
    `${JSON.stringify({ hash }, null, 2)}\n`,
    'utf8',
  );
}

/**
 * Run a single pipeline stage as a Node subprocess.
 *
 * Captures stdout/stderr so the child output does not leak to the orchestrator.
 *
 * @param {{ name: string, script: string, args: string[] }} stage - Stage to run.
 * @returns {{ status: number | null }} Subprocess result.
 */
function runStageSubprocess(stage) {
  const scriptPath = path.join(repoRoot, 'rag-index', stage.script);
  return spawnSync(process.execPath, [scriptPath, ...stage.args], {
    cwd: repoRoot,
    encoding: 'utf8',
    stdio: ['inherit', 'pipe', 'pipe'],
  });
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

  writeJsonOrText(summary, Boolean(args.json), (payload) =>
    [
      ...payload.stages.map(
        (stage) => `${stage.name}: ${stage.status} (${stage.elapsedMs}ms)`,
      ),
      payload.ok ? 'pass' : `fail: ${payload.error}`,
    ].join('\n'),
  );

  process.exitCode = summary.ok ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
