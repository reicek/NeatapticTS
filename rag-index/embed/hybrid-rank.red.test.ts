import { spawnSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface QueryDenseContractReport {
  topChunkId: number;
  topHeadingPath: string | null;
  useDense: boolean;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('hybrid-rank.mjs', () => {
  describe('dense candidate source contract', () => {
    it('adds global dense candidates that do not match the BM25 candidate pool', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-query-dense-red-'),
      );
      const corpusDatabasePath = path.join(fixtureDirectory, 'corpus.sqlite');

      try {
        // Act
        const result = runModuleEvaluation<QueryDenseContractReport>(`
          import { createClient } from '@libsql/client';
          import { pathToFileURL } from 'node:url';
          import { queryDenseIndex } from './rag-index/query-dense.mjs';

          const corpusClient = createClient({ url: pathToFileURL(${JSON.stringify(corpusDatabasePath)}).href });
          await corpusClient.execute(\`
            CREATE TABLE documents (
              doc_id INTEGER PRIMARY KEY,
              file_path TEXT NOT NULL UNIQUE,
              doc_family TEXT NOT NULL,
              mtime_ms INTEGER NOT NULL,
              file_size INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              indexed_at INTEGER NOT NULL
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
              embedding BLOB,
              embedding_model TEXT,
              chunk_sha256 TEXT,
              embedded_at INTEGER,
              UNIQUE(doc_id, chunk_index)
            );
          \`);
          await corpusClient.execute(\`
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
              body_text,
              heading_path,
              content='chunks',
              content_rowid='chunk_id',
              tokenize='porter unicode61'
            );
          \`);
          const toBlob = (values) => Buffer.from(Float32Array.from(values).buffer);
          await corpusClient.execute({ sql: 'INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)', args: [1, 'bm25.md', 'plan', 0, 1, 'a', 0] });
          await corpusClient.execute({ sql: 'INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)', args: [2, 'semantic.md', 'readme', 0, 1, 'b', 0] });
          await corpusClient.execute({ sql: 'INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, embedding, embedding_model) VALUES (?, ?, ?, ?, ?, ?, ?, vector8(?), ?)', args: [1, 1, 0, 'Exact lexical decoy', 'alpha query exact tokens', 0, 24, toBlob([0, 1]), 'fixture-model'] });
          await corpusClient.execute({ sql: 'INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, embedding, embedding_model) VALUES (?, ?, ?, ?, ?, ?, ?, vector8(?), ?)', args: [2, 2, 0, 'Semantic target', 'orthogonal paraphrase content', 0, 28, toBlob([1, 0]), 'fixture-model'] });
          await corpusClient.execute(\`INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');\`);
          await corpusClient.close();

          const report = await queryDenseIndex({
            alpha: 0.25,
            corpusDatabasePath: ${JSON.stringify(corpusDatabasePath)},
            dense: true,
            embedText: async () => new Float32Array([1, 0]),
            limit: 1,
            modelId: 'fixture-model',
            modelMeta: { dimension: 2, model_id: 'fixture-model' },
            query: 'alpha query',
          });

          console.log(JSON.stringify({
            topChunkId: report.results[0]?.chunk_id,
            topHeadingPath: report.results[0]?.heading_path,
            useDense: report.use_dense,
          }));
        `);

        // Assert
        expect(result).toEqual(
          expect.objectContaining({
            report: {
              topChunkId: 1,
              topHeadingPath: 'Exact lexical decoy',
              useDense: true,
            },
            status: 0,
          }),
        );
      } finally {
        await rm(fixtureDirectory, { force: true, recursive: true });
      }
    });

    it('searches dense candidates when BM25 has no lexical candidate', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-query-dense-empty-red-'),
      );
      const corpusDatabasePath = path.join(fixtureDirectory, 'corpus.sqlite');

      try {
        // Act
        const result = runModuleEvaluation<QueryDenseContractReport>(`
          import { createClient } from '@libsql/client';
          import { pathToFileURL } from 'node:url';
          import { queryDenseIndex } from './rag-index/query-dense.mjs';

          const corpusClient = createClient({ url: pathToFileURL(${JSON.stringify(corpusDatabasePath)}).href });
          await corpusClient.execute(\`
            CREATE TABLE documents (
              doc_id INTEGER PRIMARY KEY,
              file_path TEXT NOT NULL UNIQUE,
              doc_family TEXT NOT NULL,
              mtime_ms INTEGER NOT NULL,
              file_size INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              indexed_at INTEGER NOT NULL
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
              embedding BLOB,
              embedding_model TEXT,
              chunk_sha256 TEXT,
              embedded_at INTEGER,
              UNIQUE(doc_id, chunk_index)
            );
          \`);
          await corpusClient.execute(\`
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
              body_text,
              heading_path,
              content='chunks',
              content_rowid='chunk_id',
              tokenize='porter unicode61'
            );
          \`);
          await corpusClient.execute({ sql: 'INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)', args: [1, 'semantic.md', 'readme', 0, 1, 'a', 0] });
          await corpusClient.execute({ sql: 'INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, embedding, embedding_model) VALUES (?, ?, ?, ?, ?, ?, ?, vector8(?), ?)', args: [1, 1, 0, 'Dense-only target', 'orthogonal paraphrase content', 0, 28, Buffer.from(Float32Array.from([1, 0]).buffer), 'fixture-model'] });
          await corpusClient.execute(\`INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');\`);
          await corpusClient.close();

          const report = await queryDenseIndex({
            alpha: 0.5,
            corpusDatabasePath: ${JSON.stringify(corpusDatabasePath)},
            dense: true,
            embedText: async () => new Float32Array([1, 0]),
            limit: 1,
            modelId: 'fixture-model',
            modelMeta: { dimension: 2, model_id: 'fixture-model' },
            query: 'missing lexical tokens',
          });

          console.log(JSON.stringify({
            topChunkId: report.results[0]?.chunk_id,
            topHeadingPath: report.results[0]?.heading_path,
            useDense: report.use_dense,
          }));
        `);

        // Assert
        expect(result).toEqual(
          expect.objectContaining({
            report: {
              topChunkId: 1,
              topHeadingPath: 'Dense-only target',
              useDense: true,
            },
            status: 0,
          }),
        );
      } finally {
        await rm(fixtureDirectory, { force: true, recursive: true });
      }
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
