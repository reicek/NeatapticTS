import { spawnSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface HybridRankContractReport {
  rankedChunkIds: string[];
  roundedScores: number[];
}

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
  describe('red weighted-rank contract', () => {
    it('combines min-max BM25 and cosine scores with configurable alpha', () => {
      // Arrange and Act
      const result = runModuleEvaluation<HybridRankContractReport>(`
        import { rankHybridResults } from './scripts/semantic-index/hybrid-rank.mjs';

        const ranked = rankHybridResults({
          alpha: 0.25,
          candidates: [
            { bm25_score: 10, chunk_id: 'chunk-bm25', embedding: new Float32Array([0, 1]) },
            { bm25_score: 2, chunk_id: 'chunk-semantic', embedding: new Float32Array([1, 0]) },
            { bm25_score: 6, chunk_id: 'chunk-balanced', embedding: new Float32Array([0.6, 0.8]) },
          ],
          queryEmbedding: new Float32Array([1, 0]),
        });

        console.log(JSON.stringify({
          rankedChunkIds: ranked.map(({ chunk_id }) => chunk_id),
          roundedScores: ranked.map(({ score }) => Number(score.toFixed(3))),
        }));
      `);

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: {
            rankedChunkIds: ['chunk-semantic', 'chunk-balanced', 'chunk-bm25'],
            roundedScores: [0.75, 0.575, 0.25],
          },
          status: 0,
        }),
      );
    });
  });

  describe('dense candidate source contract', () => {
    it('adds global dense candidates that do not match the BM25 candidate pool', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-query-dense-red-'),
      );
      const corpusDatabasePath = path.join(
        fixtureDirectory,
        'semantic-index.sqlite',
      );
      const embeddingsDatabasePath = path.join(
        fixtureDirectory,
        'embeddings.sqlite',
      );

      try {
        // Act
        const result = runModuleEvaluation<QueryDenseContractReport>(`
          import Database from 'better-sqlite3';
          import { queryDenseIndex } from './scripts/semantic-index/query-dense.mjs';

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
              doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
              chunk_index INTEGER NOT NULL,
              heading_path TEXT,
              body_text TEXT NOT NULL,
              char_start INTEGER NOT NULL,
              char_end INTEGER NOT NULL,
              UNIQUE(doc_id, chunk_index)
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
              body_text,
              heading_path,
              content='chunks',
              content_rowid='chunk_id',
              tokenize='porter unicode61'
            );
          \`);
          corpusDatabase.prepare('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)').run(1, 'bm25.md', 'plan', 0, 1, 'a', 0);
          corpusDatabase.prepare('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)').run(2, 'semantic.md', 'readme', 0, 1, 'b', 0);
          corpusDatabase.prepare('INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)').run(1, 1, 0, 'Exact lexical decoy', 'alpha query exact tokens', 0, 24);
          corpusDatabase.prepare('INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)').run(2, 2, 0, 'Semantic target', 'orthogonal paraphrase content', 0, 28);
          corpusDatabase.exec(\`INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');\`);
          corpusDatabase.close();

          const embeddingsDatabase = new Database(${JSON.stringify(embeddingsDatabasePath)});
          embeddingsDatabase.exec(\`
            CREATE TABLE chunk_embeddings (
              chunk_id INTEGER PRIMARY KEY,
              embedding BLOB NOT NULL,
              chunk_sha256 TEXT NOT NULL,
              model_id TEXT NOT NULL,
              model_sha256 TEXT NOT NULL,
              dimension INTEGER NOT NULL,
              embedded_at TEXT NOT NULL
            );
          \`);
          const toBlob = (values) => Buffer.from(Float32Array.from(values).buffer);
          const insertEmbedding = embeddingsDatabase.prepare('INSERT INTO chunk_embeddings VALUES (?, ?, ?, ?, ?, ?, ?)');
          insertEmbedding.run(1, toBlob([0, 1]), 'sha-1', 'fixture-model', 'model-sha', 2, '2026-05-23T00:00:00.000Z');
          insertEmbedding.run(2, toBlob([1, 0]), 'sha-2', 'fixture-model', 'model-sha', 2, '2026-05-23T00:00:00.000Z');
          embeddingsDatabase.close();

          const report = await queryDenseIndex({
            alpha: 0.25,
            corpusDatabasePath: ${JSON.stringify(corpusDatabasePath)},
            dense: true,
            embeddingsDatabasePath: ${JSON.stringify(embeddingsDatabasePath)},
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
              topChunkId: 2,
              topHeadingPath: 'Semantic target',
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
      const corpusDatabasePath = path.join(
        fixtureDirectory,
        'semantic-index.sqlite',
      );
      const embeddingsDatabasePath = path.join(
        fixtureDirectory,
        'embeddings.sqlite',
      );

      try {
        // Act
        const result = runModuleEvaluation<QueryDenseContractReport>(`
          import Database from 'better-sqlite3';
          import { queryDenseIndex } from './scripts/semantic-index/query-dense.mjs';

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
              doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
              chunk_index INTEGER NOT NULL,
              heading_path TEXT,
              body_text TEXT NOT NULL,
              char_start INTEGER NOT NULL,
              char_end INTEGER NOT NULL,
              UNIQUE(doc_id, chunk_index)
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
              body_text,
              heading_path,
              content='chunks',
              content_rowid='chunk_id',
              tokenize='porter unicode61'
            );
          \`);
          corpusDatabase.prepare('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)').run(1, 'semantic.md', 'readme', 0, 1, 'a', 0);
          corpusDatabase.prepare('INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)').run(1, 1, 0, 'Dense-only target', 'orthogonal paraphrase content', 0, 28);
          corpusDatabase.exec(\`INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');\`);
          corpusDatabase.close();

          const embeddingsDatabase = new Database(${JSON.stringify(embeddingsDatabasePath)});
          embeddingsDatabase.exec(\`
            CREATE TABLE chunk_embeddings (
              chunk_id INTEGER PRIMARY KEY,
              embedding BLOB NOT NULL,
              chunk_sha256 TEXT NOT NULL,
              model_id TEXT NOT NULL,
              model_sha256 TEXT NOT NULL,
              dimension INTEGER NOT NULL,
              embedded_at TEXT NOT NULL
            );
          \`);
          embeddingsDatabase.prepare('INSERT INTO chunk_embeddings VALUES (?, ?, ?, ?, ?, ?, ?)')
            .run(1, Buffer.from(Float32Array.from([1, 0]).buffer), 'sha-1', 'fixture-model', 'model-sha', 2, '2026-05-23T00:00:00.000Z');
          embeddingsDatabase.close();

          const report = await queryDenseIndex({
            alpha: 0.5,
            corpusDatabasePath: ${JSON.stringify(corpusDatabasePath)},
            dense: true,
            embeddingsDatabasePath: ${JSON.stringify(embeddingsDatabasePath)},
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
