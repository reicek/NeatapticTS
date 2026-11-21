/**
 * @module build-memory-probe
 * @description Captures the PRE-CHANGE build memory baseline for the NeatapticTS semantic
 * corpus. The probe runs the corpus build IN-PROCESS against a TEMPORARY database path, so
 * the LIVE index (`rag-index/data/turso-replica.sqlite`) is never opened, written or replaced
 * by the probe.
 *
 * The captured record holds the four numbers the corpus contract compares against:
 * the peak RSS of the building process (`process.resourceUsage().maxRSS`), the byte size of
 * the temporary replica, the indexed document count and the indexed chunk count. It also
 * records the identity of the corpus that produced them - the per-family document census read
 * back from the temporary replica - so a later post-change measurement can be proven
 * like-for-like instead of being compared against a hand-written number.
 *
 * Usage (capture or refresh the baseline):
 *   node rag-index/__tests__/fixtures/build-memory-probe.mjs \
 *     --out=rag-index/__tests__/fixtures/build-memory-baseline.json
 */

import { createHash } from 'node:crypto';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createClient } from '@libsql/client';

import { defaultDatabasePath } from '../../init-schema.mjs';

/** Schema identifier of the recorded baseline fixture. */
export const BASELINE_SCHEMA = 'rag-index/build-memory-baseline/v1';

const moduleDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(moduleDirectory, '..', '..', '..');

/** Default location of the recorded baseline fixture. */
export const DEFAULT_BASELINE_PATH = path.join(
  moduleDirectory,
  'build-memory-baseline.json',
);

/**
 * Run the corpus build against a temporary database path and measure it.
 *
 * The build runs in-process so the peak RSS is the RSS of THIS process, and the replica is
 * written to a fresh temporary directory that is removed when the measurement completes.
 *
 * @param {object} [options] - Probe options.
 * @param {string} [options.databaseDirectory] - Directory to host the temporary replica. When
 *   omitted a fresh temporary directory is created and removed afterwards.
 * @returns {Promise<object>} The baseline record (see `BASELINE_SCHEMA`).
 */
export async function captureBuildMemoryBaseline(options = {}) {
  const ownsDirectory = !options.databaseDirectory;
  const databaseDirectory = options.databaseDirectory
    ? path.resolve(options.databaseDirectory)
    : await mkdtemp(path.join(tmpdir(), 'rag-build-memory-probe-'));
  const databasePath = path.join(databaseDirectory, 'corpus.sqlite');

  let record = null;

  try {
    const { buildSemanticIndex } = await import('../../build-index.mjs');
    const summary = await buildSemanticIndex({ databasePath });
    const replica = await stat(databasePath);
    const census = await readIndexCensus(databasePath);
    const usage = process.resourceUsage();
    const peakRssUnit = process.platform === 'darwin' ? 'bytes' : 'kilobytes';

    record = {
      schema: BASELINE_SCHEMA,
      capturedAt: new Date().toISOString(),
      corpus: {
        identity: census.identity,
        familyCount: census.families.length,
        families: census.families,
        documentCount: census.documentCount,
      },
      measurements: {
        peakRssBytes:
          peakRssUnit === 'bytes' ? usage.maxRSS : usage.maxRSS * 1024,
        peakRssRaw: usage.maxRSS,
        peakRssUnit,
        replicaBytes: replica.size,
        documentCount: census.documentCount,
        chunkCount: census.chunkCount,
      },
      build: {
        scanned: summary.scanned,
        indexed: summary.indexed,
        chunks: summary.chunks,
        purged: summary.purged,
        elapsedMs: summary.elapsedMs,
        replica: '<temporary-database-path>',
        liveIndexPath: defaultDatabasePath,
        liveIndexUntouched: true,
        temporaryDirectoryRemoved: false,
      },
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        repositoryRoot,
      },
    };
  } finally {
    if (ownsDirectory && record) {
      record.build.temporaryDirectoryRemoved =
        await removeTemporaryDirectory(databaseDirectory);
    }
  }

  return record;
}

/**
 * Remove the temporary directory that hosted the measured replica.
 *
 * Windows keeps a freshly closed libSQL file locked for a short while, so the removal is
 * retried and, as a last resort, abandoned: a leftover directory in the OS temp folder must
 * never invalidate an otherwise complete measurement.
 *
 * @param {string} directory - Directory to remove.
 * @returns {Promise<boolean>} True when the directory was removed.
 */
async function removeTemporaryDirectory(directory) {
  try {
    await rm(directory, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 250,
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Read the per-family document census, the document count and the chunk count back from the
 * temporary replica, and derive the corpus identity from that census.
 *
 * @param {string} databasePath - Path to the temporary replica.
 * @returns {Promise<{families: Array<{family: string, documentCount: number}>, documentCount: number, chunkCount: number, identity: string}>}
 *   The measured census of the replica.
 */
export async function readIndexCensus(databasePath) {
  const client = createClient({ url: pathToFileURL(databasePath).href });
  try {
    const familyRows = await client.execute(
      'SELECT doc_family AS family, COUNT(*) AS count FROM documents GROUP BY doc_family ORDER BY doc_family',
    );
    const families = familyRows.rows.map((row) => ({
      family: String(row.family),
      documentCount: Number(row.count),
    }));
    const documentRow = await client.execute(
      'SELECT COUNT(*) AS count FROM documents',
    );
    const chunkRow = await client.execute('SELECT COUNT(*) AS count FROM chunks');
    const identity = createHash('sha256')
      .update(
        families
          .map((entry) => `${entry.family}:${entry.documentCount}`)
          .join('\n'),
      )
      .digest('hex');

    return {
      families,
      documentCount: Number(documentRow.rows[0].count),
      chunkCount: Number(chunkRow.rows[0].count),
      identity,
    };
  } finally {
    await client.close();
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  const outArgument = process.argv
    .slice(2)
    .find((argument) => argument.startsWith('--out='));
  const outputPath = outArgument
    ? path.resolve(outArgument.slice('--out='.length))
    : DEFAULT_BASELINE_PATH;
  const record = await captureBuildMemoryBaseline();
  await writeFile(outputPath, `${JSON.stringify(record, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify(record, null, 2)}\n`);
}
