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
    it('returns pass false when no usable embeddings and pass true when embeddings are present', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-validate-embeddings-red-'),
      );
      const emptyCorpusDatabasePath = path.join(
        fixtureDirectory,
        'empty-corpus.sqlite',
      );
      const embeddedCorpusDatabasePath = path.join(
        fixtureDirectory,
        'embedded-corpus.sqlite',
      );

      // Act
      const result = runModuleEvaluation<ValidateEmbeddingsContractReport>(`
        import { createClient } from '@libsql/client';
        import { pathToFileURL } from 'node:url';
        import { validateEmbeddings } from './rag-index/validate-embeddings.mjs';

        const toBlob = (values) => Buffer.from(Float32Array.from(values).buffer);
        const createChunksTable = async (client) => {
          await client.execute(\`
            CREATE TABLE chunks (
              chunk_id INTEGER PRIMARY KEY,
              doc_id INTEGER NOT NULL,
              chunk_index INTEGER NOT NULL,
              heading_path TEXT,
              body_text TEXT NOT NULL,
              char_start INTEGER NOT NULL,
              char_end INTEGER NOT NULL,
              embedding BLOB,
              embedding_model TEXT,
              chunk_sha256 TEXT,
              embedded_at INTEGER
            );
          \`);
        };

        const emptyClient = createClient({ url: pathToFileURL(${JSON.stringify(emptyCorpusDatabasePath)}).href });
        await createChunksTable(emptyClient);
        await emptyClient.execute({
          sql: "INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
          args: [1, 1, 0, null, 'first chunk', 0, 11],
        });
        await emptyClient.close();

        const embeddedClient = createClient({ url: pathToFileURL(${JSON.stringify(embeddedCorpusDatabasePath)}).href });
        await createChunksTable(embeddedClient);
        await embeddedClient.execute({
          sql: 'INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, embedding, embedding_model) VALUES (?, ?, ?, ?, ?, ?, ?, vector8(?), ?)',
          args: [1, 1, 0, null, 'first chunk', 0, 11, toBlob([1, 0, 0]), 'all-MiniLM-L6-v2'],
        });
        await embeddedClient.close();

        const emptyReport = await validateEmbeddings({
          corpusDatabasePath: ${JSON.stringify(emptyCorpusDatabasePath)},
          modelId: 'all-MiniLM-L6-v2',
        });
        const embeddedReport = await validateEmbeddings({
          corpusDatabasePath: ${JSON.stringify(embeddedCorpusDatabasePath)},
          modelId: 'all-MiniLM-L6-v2',
        });

        console.log(JSON.stringify({
          issues: emptyReport.evidence.map(({ issue }) => issue),
          passes: [emptyReport.pass, embeddedReport.pass],
        }));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true });

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: {
            issues: ['no embeddings for model'],
            passes: [false, true],
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
