import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface DenseReadinessReport {
  chunk_count?: number | null;
  embedding_count?: number | null;
  ready: boolean;
  reason: string;
  state: 'cold' | 'model-only' | 'warm';
}

interface DenseReadinessStateSequenceReport {
  chunkCounts?: Array<number | null>;
  embeddingCounts?: Array<number | null>;
  readyValues: boolean[];
  states: string[];
}

interface DenseReadinessGateReport {
  evidence: Record<string, string | number | null>;
  fixHint: string | null;
  owner: string;
  pass: boolean;
}

interface DenseReadinessGatePairReport {
  cold: DenseReadinessGateReport;
  warm: DenseReadinessGateReport;
}

interface SearchCorpusColdReport {
  denseReasonHasContent: boolean;
  denseQueryCalls: number;
  dense_degraded?: boolean;
  dense_state?: string;
  resultCount: number;
  use_dense?: boolean;
}

interface SearchCorpusWarmReport {
  denseQueryCalls: number;
  dense_state?: string;
  hasDenseDegraded: boolean;
  resultCount: number;
  use_dense?: boolean;
}

interface SearchCorpusCacheReport {
  denseQueryCalls: number;
  readinessCalls: number;
}

interface SearchCorpusSchemaReport {
  defaultValue: boolean | null;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('dense-readiness.mjs', () => {
  describe('red readiness state contract', () => {
    it('reports cold when the model and embeddings database are absent', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DenseReadinessReport>(`
        import { mkdtemp } from 'node:fs/promises';
        import { tmpdir } from 'node:os';
        import path from 'node:path';
        import { checkDenseReadiness } from './scripts/semantic-index/dense-readiness.mjs';

        const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'dense-readiness-cold-'));
        const report = await checkDenseReadiness({
          corpusDatabasePath: path.join(fixtureDirectory, 'semantic-index.sqlite'),
          embeddingsDatabasePath: path.join(fixtureDirectory, 'embeddings.sqlite'),
          modelDirectory: path.join(fixtureDirectory, 'models'),
        });

        console.log(JSON.stringify(report));
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: expect.objectContaining({
          ready: false,
          reason: expect.stringMatching(/\S/u),
          state: 'cold',
        }),
        status: 0,
      }));
    });

    it('reports model-only when model.onnx is present and the embeddings database is absent', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DenseReadinessReport>(`
        import { mkdir, writeFile } from 'node:fs/promises';
        import { mkdtemp } from 'node:fs/promises';
        import { tmpdir } from 'node:os';
        import path from 'node:path';
        import { checkDenseReadiness } from './scripts/semantic-index/dense-readiness.mjs';

        const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'dense-readiness-model-only-'));
        const modelDirectory = path.join(fixtureDirectory, 'models');
        await mkdir(modelDirectory, { recursive: true });
        await writeFile(path.join(modelDirectory, 'model.onnx'), 'fixture model');
        const report = await checkDenseReadiness({
          corpusDatabasePath: path.join(fixtureDirectory, 'semantic-index.sqlite'),
          embeddingsDatabasePath: path.join(fixtureDirectory, 'embeddings.sqlite'),
          modelDirectory,
        });

        console.log(JSON.stringify(report));
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: expect.objectContaining({
          ready: false,
          reason: expect.stringMatching(/\S/u),
          state: 'model-only',
        }),
        status: 0,
      }));
    });

    it('distinguishes partial embeddings from matching warm embeddings', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DenseReadinessStateSequenceReport>(`
        import Database from 'better-sqlite3';
        import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
        import { tmpdir } from 'node:os';
        import path from 'node:path';
        import { checkDenseReadiness } from './scripts/semantic-index/dense-readiness.mjs';

        const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'dense-readiness-counts-'));
        const modelDirectory = path.join(fixtureDirectory, 'models');
        const corpusDatabasePath = path.join(fixtureDirectory, 'semantic-index.sqlite');
        const partialEmbeddingsPath = path.join(fixtureDirectory, 'partial-embeddings.sqlite');
        const warmEmbeddingsPath = path.join(fixtureDirectory, 'warm-embeddings.sqlite');
        await mkdir(modelDirectory, { recursive: true });
        await writeFile(path.join(modelDirectory, 'model.onnx'), 'fixture model');
        createCorpusDatabase(corpusDatabasePath, 2);
        createEmbeddingsDatabase(partialEmbeddingsPath, 1);
        createEmbeddingsDatabase(warmEmbeddingsPath, 2);

        const partial = await checkDenseReadiness({ corpusDatabasePath, embeddingsDatabasePath: partialEmbeddingsPath, modelDirectory });
        const warm = await checkDenseReadiness({ corpusDatabasePath, embeddingsDatabasePath: warmEmbeddingsPath, modelDirectory });

        console.log(JSON.stringify({
          chunkCounts: [partial.chunk_count, warm.chunk_count],
          embeddingCounts: [partial.embedding_count, warm.embedding_count],
          readyValues: [partial.ready, warm.ready],
          states: [partial.state, warm.state],
        }));

        function createCorpusDatabase(databasePath, chunkCount) {
          const database = new Database(databasePath);
          database.exec('CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, body_text TEXT NOT NULL);');
          const insertChunk = database.prepare('INSERT INTO chunks (chunk_id, body_text) VALUES (?, ?)');
          for (let chunkNumber = 1; chunkNumber <= chunkCount; chunkNumber += 1) insertChunk.run(chunkNumber, 'fixture chunk');
          database.close();
        }

        function createEmbeddingsDatabase(databasePath, embeddingCount) {
          const database = new Database(databasePath);
          database.exec(\`
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
          const insertEmbedding = database.prepare(
            'INSERT INTO chunk_embeddings (chunk_id, embedding, chunk_sha256, model_id, model_sha256, dimension, embedded_at) VALUES (?, ?, ?, ?, ?, ?, ?)'
          );
          for (let chunkNumber = 1; chunkNumber <= embeddingCount; chunkNumber += 1) {
            insertEmbedding.run(chunkNumber, Buffer.from(new Float32Array([1, 0, 0]).buffer), 'sha', 'all-MiniLM-L6-v2', 'model-sha', 3, '2026-05-23T00:00:00.000Z');
          }
          database.close();
        }
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          chunkCounts: [2, 2],
          embeddingCounts: [1, 2],
          readyValues: [false, true],
          states: ['model-only', 'warm'],
        },
        status: 0,
      }));
    });

    it('lets DENSE_FORCE_STATE override filesystem state for cold and model-only', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DenseReadinessStateSequenceReport>(`
        import { checkDenseReadiness } from './scripts/semantic-index/dense-readiness.mjs';

        process.env.DENSE_FORCE_STATE = 'cold';
        const cold = await checkDenseReadiness({});
        process.env.DENSE_FORCE_STATE = 'model-only';
        const modelOnly = await checkDenseReadiness({});
        delete process.env.DENSE_FORCE_STATE;

        console.log(JSON.stringify({
          readyValues: [cold.ready, modelOnly.ready],
          states: [cold.state, modelOnly.state],
        }));
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          readyValues: [false, false],
          states: ['cold', 'model-only'],
        },
        status: 0,
      }));
    });
  });

  describe('red readiness gate contract', () => {
    it('maps warm and cold probe results to the standard gate shape', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DenseReadinessGatePairReport>(`
        import { evaluateDenseReadinessGate } from './scripts/agent-customization/gates/dense-readiness.gate.mjs';

        const warm = await evaluateDenseReadinessGate({
          readinessProbe: async () => ({
            chunk_count: 7,
            embedding_count: 7,
            ready: true,
            reason: 'All chunks have embeddings.',
            state: 'warm',
          }),
        });
        const cold = await evaluateDenseReadinessGate({
          readinessProbe: async () => ({
            chunk_count: null,
            embedding_count: null,
            ready: false,
            reason: 'Model and embeddings are absent.',
            state: 'cold',
          }),
        });

        console.log(JSON.stringify({ cold, warm }));
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          cold: {
            evidence: { state: 'cold' },
            fixHint: 'Run `npm run index:prewarm` to build the embedding index.',
            owner: '01-planning',
            pass: false,
          },
          warm: {
            evidence: { chunk_count: 7, embedding_count: 7, state: 'warm' },
            fixHint: null,
            owner: '01-planning',
            pass: true,
          },
        },
        status: 0,
      }));
    });
  });

  describe('red MCP search_corpus dense default and degradation contract', () => {
    it('returns BM25-only degraded output with cold readiness', () => {
      // Arrange and Act
      const result = runModuleEvaluation<SearchCorpusColdReport>(`
        import Database from 'better-sqlite3';
        import { mkdtemp } from 'node:fs/promises';
        import { tmpdir } from 'node:os';
        import path from 'node:path';
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';

        const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'dense-readiness-search-cold-'));
        const databasePath = path.join(fixtureDirectory, 'semantic-index.sqlite');
        createSearchCorpusDatabase(databasePath);
        let denseQueryCalls = 0;
        const result = await searchCorpus({
          databasePath,
          denseQuery: async () => {
            denseQueryCalls += 1;
            throw new Error('Dense query must not run while readiness is cold.');
          },
          limit: 1,
          query: 'NEAT activation',
          readinessProbe: async () => ({ ready: false, state: 'cold', reason: 'Embeddings are absent.' }),
          use_dense: true,
        });

        console.log(JSON.stringify({
          denseReasonHasContent: typeof result.dense_reason === 'string' && result.dense_reason.trim().length >= 1,
          denseQueryCalls,
          dense_degraded: result.dense_degraded,
          dense_state: result.dense_state,
          resultCount: result.results.length,
          use_dense: result.use_dense,
        }));

        function createSearchCorpusDatabase(targetPath) {
          const database = new Database(targetPath);
          database.exec(\`
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
            CREATE VIRTUAL TABLE chunks_fts USING fts5(body_text);
            INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
              VALUES (1, 'fixture.md', 'plan', 1, 100, 'fixture-sha', 1);
            INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end)
              VALUES (1, 1, 0, 'Fixture', 'NEAT activation retrieval contract', 0, 34);
            INSERT INTO chunks_fts (rowid, body_text) VALUES (1, 'NEAT activation retrieval contract');
          \`);
          database.close();
        }
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          denseReasonHasContent: true,
          denseQueryCalls: 0,
          dense_degraded: true,
          dense_state: 'cold',
          resultCount: 1,
          use_dense: false,
        },
        status: 0,
      }));
    });

    it('treats omitted use_dense as true and proves the warm dense path', () => {
      // Arrange and Act
      const result = runModuleEvaluation<SearchCorpusWarmReport>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';

        let denseQueryCalls = 0;
        const result = await searchCorpus({
          denseQuery: async () => {
            denseQueryCalls += 1;
            return {
              alpha: 0.5,
              limit: 1,
              query: 'NEAT activation',
              results: [{ chunk_id: 1, score: 0.99, text: 'NEAT activation retrieval contract' }],
              use_dense: true,
            };
          },
          limit: 1,
          query: 'NEAT activation',
          readinessProbe: async () => ({ ready: true, state: 'warm', reason: 'All chunks have embeddings.' }),
        });

        console.log(JSON.stringify({
          denseQueryCalls,
          dense_state: result.dense_state,
          hasDenseDegraded: Object.hasOwn(result, 'dense_degraded'),
          resultCount: result.results.length,
          use_dense: result.use_dense,
        }));
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          denseQueryCalls: 1,
          dense_state: 'warm',
          hasDenseDegraded: false,
          resultCount: 1,
          use_dense: true,
        },
        status: 0,
      }));
    });

    it('declares the search_corpus use_dense input schema default as true', () => {
      // Arrange and Act
      const result = runModuleEvaluation<SearchCorpusSchemaReport>(`
        import { createRepoCortexTools } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';

        const searchTool = createRepoCortexTools('./data/semantic-index.sqlite').find((tool) => tool.name === 'search_corpus');
        const defaultValue = searchTool?.inputSchema?.properties?.use_dense?.default ?? null;

        console.log(JSON.stringify({ defaultValue }));
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: { defaultValue: true },
        status: 0,
      }));
    });

    it('caches one warm readiness probe across two dense searches', () => {
      // Arrange and Act
      const result = runModuleEvaluation<SearchCorpusCacheReport>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';

        let denseQueryCalls = 0;
        let readinessCalls = 0;
        const sharedOptions = {
          denseQuery: async () => {
            denseQueryCalls += 1;
            return {
              alpha: 0.5,
              limit: 1,
              query: 'NEAT activation',
              results: [{ chunk_id: 1, score: 0.99, text: 'NEAT activation retrieval contract' }],
              use_dense: true,
            };
          },
          limit: 1,
          query: 'NEAT activation',
          readinessProbe: async () => {
            readinessCalls += 1;
            return { ready: true, state: 'warm', reason: 'All chunks have embeddings.' };
          },
          use_dense: true,
        };

        await searchCorpus(sharedOptions);
        await searchCorpus(sharedOptions);

        console.log(JSON.stringify({ denseQueryCalls, readinessCalls }));
      `);

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          denseQueryCalls: 2,
          readinessCalls: 1,
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