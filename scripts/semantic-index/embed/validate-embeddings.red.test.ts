import { spawnSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface ValidateEmbeddingsContractReport {
  issues: string[];
  passes: boolean[];
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('validate-embeddings.mjs', () => {
  describe('red count-validation contract', () => {
    it('returns pass false when counts mismatch and pass true when counts match', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'semantic-validate-embeddings-red-'));
      const corpusDatabasePath = path.join(fixtureDirectory, 'semantic-index.sqlite');
      const mismatchEmbeddingsPath = path.join(fixtureDirectory, 'mismatch-embeddings.sqlite');
      const matchedEmbeddingsPath = path.join(fixtureDirectory, 'matched-embeddings.sqlite');

      // Act
      const result = runModuleEvaluation<ValidateEmbeddingsContractReport>(`
        import Database from 'better-sqlite3';
        import { validateEmbeddings } from './scripts/semantic-index/validate-embeddings.mjs';

        const corpusDatabase = new Database(${JSON.stringify(corpusDatabasePath)});
        corpusDatabase.exec(\`
          CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, body_text TEXT NOT NULL);
          INSERT INTO chunks (chunk_id, body_text) VALUES (1, 'first chunk'), (2, 'second chunk');
        \`);
        corpusDatabase.close();

        for (const [databasePath, rowCount] of [
          [${JSON.stringify(mismatchEmbeddingsPath)}, 1],
          [${JSON.stringify(matchedEmbeddingsPath)}, 2],
        ]) {
          const embeddingsDatabase = new Database(databasePath);
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
          const insertEmbedding = embeddingsDatabase.prepare(
            'INSERT INTO chunk_embeddings (chunk_id, embedding, chunk_sha256, model_id, model_sha256, dimension, embedded_at) VALUES (?, ?, ?, ?, ?, ?, ?)'
          );
          for (let chunkNumber = 1; chunkNumber <= rowCount; chunkNumber += 1) {
            insertEmbedding.run(
              chunkNumber,
              Buffer.from(new Float32Array([1, 0, 0]).buffer),
              \`chunk-sha-\${chunkNumber}\`,
              'all-MiniLM-L6-v2',
              'fixture-model-sha256',
              3,
              '2026-05-23T00:00:00.000Z'
            );
          }
          embeddingsDatabase.close();
        }

        const mismatchReport = await validateEmbeddings({
          corpusDatabasePath: ${JSON.stringify(corpusDatabasePath)},
          embeddingsDatabasePath: ${JSON.stringify(mismatchEmbeddingsPath)},
          modelId: 'all-MiniLM-L6-v2',
        });
        const matchedReport = await validateEmbeddings({
          corpusDatabasePath: ${JSON.stringify(corpusDatabasePath)},
          embeddingsDatabasePath: ${JSON.stringify(matchedEmbeddingsPath)},
          modelId: 'all-MiniLM-L6-v2',
        });

        console.log(JSON.stringify({
          issues: mismatchReport.evidence.map(({ issue }) => issue),
          passes: [mismatchReport.pass, matchedReport.pass],
        }));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true });

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          issues: ['embedding count mismatch'],
          passes: [false, true],
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