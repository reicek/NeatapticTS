import { spawnSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface EmbeddingIndexContractReport {
  firstSummary: { embedded: number; skipped: number };
  purgeSummary: { embedded: number; purged: number; skipped: number };
  rows: { blobBytes: number[]; count: number; dimensions: number[] };
  secondSummary: { embedded: number; skipped: number };
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('embed-index.mjs', () => {
  describe('red embedding builder contract', () => {
    it('stores BLOB vectors for a seeded corpus and skips unchanged chunks incrementally', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'semantic-embed-index-red-'));
      const corpusDatabasePath = path.join(fixtureDirectory, 'semantic-index.sqlite');
      const embeddingsDatabasePath = path.join(fixtureDirectory, 'embeddings.sqlite');

      // Act
      const result = runModuleEvaluation<EmbeddingIndexContractReport>(`
        import Database from 'better-sqlite3';
        import { buildEmbeddingIndex } from './scripts/semantic-index/embed-index.mjs';

        const corpusDatabase = new Database(${JSON.stringify(corpusDatabasePath)});
        corpusDatabase.exec(\`
          CREATE TABLE documents (
            doc_id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            doc_family TEXT NOT NULL,
            mtime_ms INTEGER NOT NULL,
            file_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            indexed_at INTEGER NOT NULL
          );
          CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            heading_path TEXT,
            body_text TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL
          );
        \`);
        const documentInsert = corpusDatabase.prepare(
          'INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at) VALUES (?, ?, ?, ?, ?, ?)'
        );
        const chunkInsert = corpusDatabase.prepare(
          'INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end) VALUES (?, ?, ?, ?, ?, ?, ?)'
        );
        const documentId = Number(documentInsert.run('fixture.md', 'plan', 1, 100, 'fixture-doc-sha', 1).lastInsertRowid);
        chunkInsert.run(1, documentId, 0, 'Embedding setup', 'alpha dense retrieval note', 0, 26);
        chunkInsert.run(2, documentId, 1, 'Hybrid search', 'beta weighted rank note', 27, 51);
        chunkInsert.run(3, documentId, 2, 'Validation', 'gamma count check note', 52, 74);
        corpusDatabase.close();

        const vectorsByChunkId = new Map([
          [1, [1, 0, 0]],
          [2, [0, 1, 0]],
          [3, [0, 0, 1]],
        ]);
        const embedText = async ({ chunkId }) => new Float32Array(vectorsByChunkId.get(Number(chunkId)));
        const sharedOptions = {
          corpusDatabasePath: ${JSON.stringify(corpusDatabasePath)},
          dimension: 3,
          embedText,
          embeddingsDatabasePath: ${JSON.stringify(embeddingsDatabasePath)},
          modelId: 'all-MiniLM-L6-v2',
          modelSha256: 'fixture-model-sha256',
        };
        const firstSummary = await buildEmbeddingIndex(sharedOptions);
        const secondSummary = await buildEmbeddingIndex(sharedOptions);

        const orphanedDatabase = new Database(${JSON.stringify(embeddingsDatabasePath)});
        orphanedDatabase.prepare(
          'INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding, chunk_sha256, model_id, model_sha256, dimension, embedded_at) VALUES (?, ?, ?, ?, ?, ?, ?)'
        ).run(
          999,
          Buffer.from(new Float32Array([1, 1, 1]).buffer),
          'orphaned-sha',
          'all-MiniLM-L6-v2',
          'fixture-model-sha256',
          3,
          '2026-05-24T00:00:00.000Z'
        );
        orphanedDatabase.close();

        const purgeSummary = await buildEmbeddingIndex(sharedOptions);

        const embeddingsDatabase = new Database(${JSON.stringify(embeddingsDatabasePath)}, { readonly: true });
        const rows = embeddingsDatabase.prepare(\`
          SELECT COUNT(*) AS count,
                 GROUP_CONCAT(dimension) AS dimensions,
                 GROUP_CONCAT(blob_bytes) AS blobBytes
          FROM (
            SELECT dimension, length(embedding) AS blob_bytes
            FROM chunk_embeddings
            ORDER BY chunk_id
          )
        \`).get();
        embeddingsDatabase.close();

        console.log(JSON.stringify({
          firstSummary: { embedded: firstSummary.embedded, skipped: firstSummary.skipped },
          purgeSummary: { embedded: purgeSummary.embedded, purged: purgeSummary.purged, skipped: purgeSummary.skipped },
          secondSummary: { embedded: secondSummary.embedded, skipped: secondSummary.skipped },
          rows: {
            blobBytes: String(rows.blobBytes).split(',').map(Number),
            count: rows.count,
            dimensions: String(rows.dimensions).split(',').map(Number),
          },
        }));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true });

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          firstSummary: { embedded: 3, skipped: 0 },
          purgeSummary: { embedded: 0, purged: 1, skipped: 3 },
          rows: { blobBytes: [12, 12, 12], count: 3, dimensions: [3, 3, 3] },
          secondSummary: { embedded: 0, skipped: 3 },
        },
        status: 0,
      }));
    });
  });
});

function runModuleEvaluation<ReportType>(source: string): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(process.execPath, ['--input-type=module', '--eval', source], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });

  return {
    report: tryParseJson<ReportType>(spawned.stdout ?? ''),
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout: spawned.stdout ?? '',
  };
}

function tryParseJson<ReportType>(stdout: string): ReportType | null {
  if (!stdout.trim()) return null;

  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}