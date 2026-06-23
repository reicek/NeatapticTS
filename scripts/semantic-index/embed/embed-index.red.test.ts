import { spawnSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface EmbeddingIndexContractReport {
  firstSummary: { embedded: number; skipped: number };
  rows: { blobBytes: number[]; count: number };
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
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-embed-index-red-'),
      );
      const corpusDatabasePath = path.join(fixtureDirectory, 'corpus.sqlite');

      // Act
      const result = runModuleEvaluation<EmbeddingIndexContractReport>(`
        import { createClient } from '@libsql/client';
        import { pathToFileURL } from 'node:url';
        import { buildEmbeddingIndex } from './scripts/semantic-index/embed-index.mjs';

        const corpusClient = createClient({ url: pathToFileURL(${JSON.stringify(corpusDatabasePath)}).href });
        await corpusClient.execute(\`
          CREATE TABLE documents (
           doc_id INTEGER PRIMARY KEY,
           file_path TEXT NOT NULL UNIQUE,
           doc_family TEXT NOT NULL,
           mtime_ms INTEGER NOT NULL,
           file_size INTEGER NOT NULL,
           sha256 TEXT NOT NULL,
           indexed_at INTEGER NOT NULL,
           arch_layer TEXT,
           test_coverage TEXT CHECK(test_coverage IN ('full', 'partial', 'none', 'unknown')),
           source_path_pattern TEXT
          );
        \`);
        await corpusClient.execute(\`
          CREATE TABLE chunks (
           chunk_id INTEGER PRIMARY KEY,
           doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
           chunk_index INTEGER NOT NULL,
           heading_path TEXT,
           body_text TEXT NOT NULL,
           char_start INTEGER NOT NULL,
           char_end INTEGER NOT NULL,
           parent_chunk_id INTEGER,
           depth INTEGER NOT NULL DEFAULT 0,
           context_header TEXT,
           symbol_name TEXT,
           signature_text TEXT,
           jsdoc_text TEXT,
           export_type TEXT,
           module_path TEXT,
           arch_layer TEXT,
           jsdoc_quality TEXT CHECK(jsdoc_quality IN ('none', 'weak', 'adequate', 'good')),
           jsdoc_word_count INTEGER,
           cyclomatic_complexity INTEGER,
           test_coverage TEXT CHECK(test_coverage IN ('full', 'partial', 'none', 'unknown')),
           source_path_pattern TEXT,
           embedding BLOB,
           embedding_model TEXT,
           chunk_sha256 TEXT,
           embedded_at INTEGER,
           UNIQUE(doc_id, chunk_index)
          );
        \`);
        const documentResult = await corpusClient.execute({
          sql: 'INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer, test_coverage, source_path_pattern) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
          args: ['fixture.md', 'plan', 1, 100, 'fixture-doc-sha', 1, 'doc', 'unknown', 'docs/**'],
        });
        const documentId = Number(documentResult.lastInsertRowid);
        const chunkInsertSql = 'INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth, context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path, arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)';
        await corpusClient.execute({ sql: chunkInsertSql, args: [1, documentId, 0, 'Embedding setup', 'alpha dense retrieval note', 0, 26, null, 0, null, null, null, null, null, null, 'doc', 'none', null, null, 'unknown', 'docs/**'] });
        await corpusClient.execute({ sql: chunkInsertSql, args: [2, documentId, 1, 'Hybrid search', 'beta weighted rank note', 27, 51, null, 1, null, null, null, null, null, null, 'doc', 'none', null, null, 'unknown', 'docs/**'] });
        await corpusClient.execute({ sql: chunkInsertSql, args: [3, documentId, 2, 'Validation', 'gamma count check note', 52, 74, null, 2, null, null, null, null, null, null, 'doc', 'none', null, null, 'unknown', 'docs/**'] });
        await corpusClient.close();

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
          modelId: 'all-MiniLM-L6-v2',
          modelSha256: 'fixture-model-sha256',
        };
        const firstSummary = await buildEmbeddingIndex(sharedOptions);
        const secondSummary = await buildEmbeddingIndex(sharedOptions);

        const rowsResult = await createClient({ url: pathToFileURL(${JSON.stringify(corpusDatabasePath)}).href }).execute(\`
          SELECT COUNT(*) AS count,
                GROUP_CONCAT(length(embedding)) AS blobBytes
          FROM chunks
          ORDER BY chunk_id
        \`);
        const rows = rowsResult.rows[0];

        console.log(JSON.stringify({
          firstSummary: { embedded: firstSummary.embedded, skipped: firstSummary.skipped },
          secondSummary: { embedded: secondSummary.embedded, skipped: secondSummary.skipped },
          rows: {
           blobBytes: String(rows.blobBytes).split(',').map(Number),
           count: rows.count,
          },
        }));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true });

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: {
            firstSummary: { embedded: 3, skipped: 0 },
            rows: { blobBytes: [15, 15, 15], count: 3 },
            secondSummary: { embedded: 0, skipped: 3 },
          },
          status: 0,
        }),
      );
    });
  });
});

function runModuleEvaluation<ReportType>(
  source: string,
): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );

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
