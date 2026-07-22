/**
 * @module embed-index.test
 * @description Red tests for Phase A Step A1: schema migration and plan-chunk
 * slice-metadata enrichment.
 *
 * These tests encode the contract that the A1-schema-enrichment slice must:
 *   - add `slice_id`, `step_number`, `phase`, and `status` columns to the
 *     `chunks` table via the schema migration;
 *   - parse step-packet YAML in plan-family chunks and populate those columns;
 *   - leave the semantic index healthy (validateDatabase passes) after the
 *     re-index, with slice metadata present.
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const REPO_ROOT = path.resolve();

interface SchemaColumnsReport {
  columns: string[];
}

interface SliceEnrichmentReport {
  slice_id: string | null;
  step_number: number | null;
  phase: string | null;
  status: string | null;
}

interface HealthAfterReindexReport {
  pass: boolean;
  columns: string[];
  slice_id: string | null;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const PLAN_CHUNK_BODY = `#### Step A1: Schema migration and embed-index enrichment [PLANNED]

\`\`\`yaml
phase: A
step: 1
title: 'Schema migration and embed-index slice metadata parsing'
status: '[PLANNED]'
goal: 'implementing'
slices:
  - slice_id: 'A1-red-tests'
    title: 'Write red tests for schema migration and embed-index slice metadata parsing'
    status: '[PLANNED]'
    goal: 'red-testing'
\`\`\`
`;

function runModuleEvaluation<ReportType>(
  source: string,
): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
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

async function makeSchemaFixture(): Promise<{
  databasePath: string;
  tempDir: string;
}> {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-index-schema-'));
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  return { databasePath, tempDir };
}

async function makePlanEnrichmentFixture(): Promise<{
  databasePath: string;
  tempDir: string;
}> {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-index-enrich-'));
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  // Keep the fixture setup in a child-process ESM evaluation, matching the
  // repo convention for exercising .mjs modules from Jest TypeScript tests.
  // Direct dynamic imports of .mjs sources fail in the rag-index-scripts Jest
  // project because the project runs under --experimental-vm-modules.
  const result = runModuleEvaluation<{ ok: boolean }>(`
    import { initSemanticIndex } from './rag-index/init-schema.mjs';
    import { createClient } from '@libsql/client';

    const databasePath = ${JSON.stringify(databasePath)};
    const planBody = ${JSON.stringify(PLAN_CHUNK_BODY)};
    const client = await initSemanticIndex({ databasePath });
    try {
      await client.execute({
        sql: \`INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
              VALUES (1, ?, 'plan', 1, 100, 'fake', 1)\`,
        args: ['plans/example.plans.md'],
      });
      await client.execute({
        sql: \`INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end)
              VALUES (1, 1, 0, ?, 0, ?)\`,
        args: [planBody, planBody.length],
      });
      console.log(JSON.stringify({ ok: true }));
    } finally {
      await client.close();
    }
  `);
  if (result.status !== 0) {
    throw new Error(
      `makePlanEnrichmentFixture failed: status=${result.status}, stderr=${result.stderr}, stdout=${result.stdout}`,
    );
  }
  return { databasePath, tempDir };
}

interface TargetedReindexSummary {
  dryRun?: boolean;
  embedded: number;
  queued?: number;
  skipped: number;
}

interface DocumentFreshnessReport {
  indexed_at: number;
}

async function makeTargetedReindexFixture(): Promise<{
  databasePath: string;
  tempDir: string;
}> {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'embed-index-targeted-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const result = runModuleEvaluation<{ ok: boolean }>(`
    import { initSemanticIndex } from './rag-index/init-schema.mjs';
    import { createClient } from '@libsql/client';

    const databasePath = ${JSON.stringify(databasePath)};
    const client = await initSemanticIndex({ databasePath });
    try {
      await client.execute({
        sql: \`INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at) VALUES (1, ?, 'plan', 1, 100, 'fake1', 1)\`,
        args: ['plans/example.plans.md'],
      });
      await client.execute({
        sql: \`INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end) VALUES (1, 1, 0, 'plan chunk body', 0, 17)\`,
      });
      await client.execute({
        sql: \`INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at) VALUES (2, ?, 'source', 1, 100, 'fake2', 1)\`,
        args: ['src/foo.ts'],
      });
      await client.execute({
        sql: \`INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end) VALUES (2, 2, 0, 'source chunk body', 0, 18)\`,
      });
      console.log(JSON.stringify({ ok: true }));
    } finally {
      await client.close();
    }
  `);
  if (result.status !== 0) {
    throw new Error(
      'makeTargetedReindexFixture failed: status=' +
        result.status +
        ', stderr=' +
        result.stderr +
        ', stdout=' +
        result.stdout,
    );
  }
  return { databasePath, tempDir };
}

describe('embed-index.mjs A1 schema migration', () => {
  describe('chunks table columns', () => {
    it('has slice_id, step_number, phase, and status columns after schema initialization', async () => {
      const { databasePath, tempDir } = await makeSchemaFixture();
      try {
        const result = runModuleEvaluation<SchemaColumnsReport>(`
          import { initSemanticIndex } from './rag-index/init-schema.mjs';
          import { createClient } from '@libsql/client';

          const databasePath = ${JSON.stringify(databasePath)};
          await initSemanticIndex({ databasePath });
          const client = createClient({ url: 'file:' + databasePath });
          const info = await client.execute("PRAGMA table_info(chunks)");
          const columns = info.rows.map(row => row.name).toSorted();
          await client.close();
          console.log(JSON.stringify({ columns }));
        `);

        expect(result.report).toEqual(
          expect.objectContaining({
            columns: expect.arrayContaining([
              'slice_id',
              'step_number',
              'phase',
              'status',
            ]),
          }),
        );
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('plan chunk slice metadata enrichment', () => {
    it('populates slice_id, step_number, phase, and status from the step packet YAML', async () => {
      const { databasePath, tempDir } = await makePlanEnrichmentFixture();
      try {
        const result = runModuleEvaluation<SliceEnrichmentReport>(`
          import { buildEmbeddingIndex } from './rag-index/embed-index.mjs';
          import { createClient } from '@libsql/client';

          const databasePath = ${JSON.stringify(databasePath)};
          const client = createClient({ url: 'file:' + databasePath });
          const embedText = async ({ text }) => new Float32Array(384).fill(0.1);
          await buildEmbeddingIndex({
            client,
            embedText,
            dimension: 384,
            modelSha256: 'fake-sha256',
            modelId: 'fake-model',
          });
          const rowResult = await client.execute(
            'SELECT slice_id, step_number, phase, status FROM chunks WHERE chunk_id = 1'
          );
          await client.close();
          const r = rowResult.rows[0] ?? {};
          console.log(JSON.stringify({
            slice_id: r.slice_id ?? null,
            step_number: r.step_number ?? null,
            phase: r.phase ?? null,
            status: r.status ?? null,
          }));
        `);

        expect(result.report).toEqual({
          slice_id: 'A1-red-tests',
          step_number: 1,
          phase: 'A',
          status: '[PLANNED]',
        });
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('cortex-index health after re-index', () => {
    it('passes semantic validation and exposes slice metadata after re-indexing a plan', async () => {
      const { databasePath, tempDir } = await makeSchemaFixture();
      try {
        const result = runModuleEvaluation<HealthAfterReindexReport>(`
          import path from 'node:path';
          import { createClient } from '@libsql/client';
          import { initSemanticIndex, repoRoot } from './rag-index/init-schema.mjs';
          import { buildEmbeddingIndex } from './rag-index/embed-index.mjs';
          import { validateDatabase } from './rag-index/validate-index.mjs';
          import { getFreshnessProof } from './rag-index/freshness.mjs';

          const databasePath = ${JSON.stringify(databasePath)};
          const planFilePath = 'plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md';
          const absolutePlanPath = path.join(repoRoot, planFilePath);
          const proof = await getFreshnessProof(absolutePlanPath);

          const client = await initSemanticIndex({ databasePath });
          await client.execute({
            sql: \`INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
                  VALUES (1, ?, 'plan', ?, ?, ?, ?)\`,
            args: [planFilePath, proof.mtime_ms, proof.size, proof.sha256, Date.now()],
          });
          const planBody = ${JSON.stringify(PLAN_CHUNK_BODY)};
          await client.execute({
            sql: \`INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end)
                  VALUES (1, 1, 0, ?, 0, ?)\`,
            args: [planBody, planBody.length],
          });
          const embedText = async ({ text }) => new Float32Array(384).fill(0.1);
          await buildEmbeddingIndex({
            client,
            embedText,
            dimension: 384,
            modelSha256: 'fake-sha256',
            modelId: 'fake-model',
          });
          const health = await validateDatabase({
            client,
            minDocuments: 1,
            minChunks: 1,
            maxStalenessMs: 86400000,
          });
          const info = await client.execute("PRAGMA table_info(chunks)");
          const columns = info.rows.map(row => row.name);
          const rowResult = await client.execute(
            'SELECT slice_id FROM chunks WHERE chunk_id = 1'
          );
          const sliceId = rowResult.rows[0]?.slice_id ?? null;
          await client.close();
          console.log(JSON.stringify({ pass: health.pass, columns, slice_id: sliceId }));
        `);

        expect(result.report).toMatchObject({
          pass: true,
          columns: expect.arrayContaining([
            'slice_id',
            'step_number',
            'phase',
            'status',
          ]),
          slice_id: expect.any(String),
        });
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });
});

describe('embed-index.mjs E1 --files targeted re-index', () => {
  describe('buildEmbeddingIndex chunk filtering', () => {
    it('only processes chunks whose file_path matches the provided files list', async () => {
      const { databasePath, tempDir } = await makeTargetedReindexFixture();
      try {
        const result = runModuleEvaluation<TargetedReindexSummary>(`
          import { initSemanticIndex } from './rag-index/init-schema.mjs';
          import { buildEmbeddingIndex } from './rag-index/embed-index.mjs';
          import { createClient } from '@libsql/client';

          const databasePath = ${JSON.stringify(databasePath)};
          const client = await initSemanticIndex({ databasePath });
          try {
            const embedText = async ({ text }) => new Float32Array(384).fill(0.1);
            const summary = await buildEmbeddingIndex({
              client,
              embedText,
              dimension: 384,
              modelSha256: 'fake-sha256',
              modelId: 'fake-model',
              files: ['plans/example.plans.md'],
            });
            await client.close();
            console.log(JSON.stringify(summary));
          } catch (error) {
            await client.close();
            throw error;
          }
        `);

        expect(result.report).toEqual(
          expect.objectContaining({ embedded: 1, skipped: 1 }),
        );
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('CLI --files parsing', () => {
    // This contract targets the parser configuration the CLI entrypoint must
    // use. The current embed-index.mjs main() calls parseCliArgs() with no
    // options, so repeated --files values overwrite each other instead of
    // being collected. The E1 implementation must pass
    // { repeatableFlags: ['files'] } and forward the resulting array to
    // buildEmbeddingIndex. The actual forwarding behavior is covered by the
    // buildEmbeddingIndex chunk-filtering test above; this assertion proves
    // the parser is configured to collect repeated --files values.
    it('collects repeated --files values when repeatableFlags includes files', async () => {
      const result = runModuleEvaluation<{ files: string | string[] }>(`
        import { parseCliArgs } from './rag-index/cli-utils.mjs';

        const flags = parseCliArgs([
          '--files=plans/example.plans.md',
          '--files=src/foo.ts',
        ], { repeatableFlags: ['files'] });
        console.log(JSON.stringify({ files: flags.files }));
      `);

      expect(result.report?.files).toEqual([
        'plans/example.plans.md',
        'src/foo.ts',
      ]);
    });
  });

  describe('affected document freshness', () => {
    it('updates documents.indexed_at for every document whose chunks were re-indexed', async () => {
      const { databasePath, tempDir } = await makeTargetedReindexFixture();
      try {
        const result = runModuleEvaluation<DocumentFreshnessReport>(`
          import { initSemanticIndex } from './rag-index/init-schema.mjs';
          import { buildEmbeddingIndex } from './rag-index/embed-index.mjs';
          import { createClient } from '@libsql/client';

          const databasePath = ${JSON.stringify(databasePath)};
          const client = await initSemanticIndex({ databasePath });
          try {
            const embedText = async ({ text }) => new Float32Array(384).fill(0.1);
            await buildEmbeddingIndex({
              client,
              embedText,
              dimension: 384,
              modelSha256: 'fake-sha256',
              modelId: 'fake-model',
              files: ['plans/example.plans.md'],
            });
            const rowResult = await client.execute(
              'SELECT indexed_at FROM documents WHERE doc_id = 1',
            );
            const row = rowResult.rows[0] ?? {};
            await client.close();
            console.log(JSON.stringify({ indexed_at: Number(row.indexed_at ?? 0) }));
          } catch (error) {
            await client.close();
            throw error;
          }
        `);

        expect(result.report?.indexed_at).toBeGreaterThan(1);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('full corpus re-index', () => {
    it('processes all chunks when files is undefined or empty', async () => {
      const { databasePath, tempDir } = await makeTargetedReindexFixture();
      try {
        const result = runModuleEvaluation<TargetedReindexSummary>(`
          import { initSemanticIndex } from './rag-index/init-schema.mjs';
          import { buildEmbeddingIndex } from './rag-index/embed-index.mjs';
          import { createClient } from '@libsql/client';

          const databasePath = ${JSON.stringify(databasePath)};
          const client = await initSemanticIndex({ databasePath });
          try {
            const embedText = async ({ text }) => new Float32Array(384).fill(0.1);
            const summary = await buildEmbeddingIndex({
              client,
              embedText,
              dimension: 384,
              modelSha256: 'fake-sha256',
              modelId: 'fake-model',
            });
            await client.close();
            console.log(JSON.stringify(summary));
          } catch (error) {
            await client.close();
            throw error;
          }
        `);

        expect(result.report).toEqual(
          expect.objectContaining({ embedded: 2, skipped: 0 }),
        );
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('dry-run read-only', () => {
    it('does not update documents.indexed_at during a dry run', async () => {
      const { databasePath, tempDir } = await makeTargetedReindexFixture();
      try {
        const result = runModuleEvaluation<DocumentFreshnessReport>(`
          import { initSemanticIndex } from './rag-index/init-schema.mjs';
          import { buildEmbeddingIndex } from './rag-index/embed-index.mjs';
          import { createClient } from '@libsql/client';

          const databasePath = ${JSON.stringify(databasePath)};
          const client = await initSemanticIndex({ databasePath });
          try {
            const embedText = async ({ text }) => new Float32Array(384).fill(0.1);
            await buildEmbeddingIndex({
              client,
              embedText,
              dimension: 384,
              modelSha256: 'fake-sha256',
              modelId: 'fake-model',
              files: ['plans/example.plans.md'],
              dryRun: true,
            });
            const rowResult = await client.execute(
              'SELECT indexed_at FROM documents WHERE doc_id = 1',
            );
            const row = rowResult.rows[0] ?? {};
            await client.close();
            console.log(JSON.stringify({ indexed_at: Number(row.indexed_at ?? 0) }));
          } catch (error) {
            await client.close();
            throw error;
          }
        `);

        expect(result.report?.indexed_at).toBe(1);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('non-existent path warning', () => {
    async function runNonExistentPathTest(): Promise<
      SpawnedJsonResult<TargetedReindexSummary>
    > {
      const { databasePath, tempDir } = await makeTargetedReindexFixture();
      try {
        const result = runModuleEvaluation<TargetedReindexSummary>(`
          import { initSemanticIndex } from './rag-index/init-schema.mjs';
          import { buildEmbeddingIndex } from './rag-index/embed-index.mjs';
          import { createClient } from '@libsql/client';

          const databasePath = ${JSON.stringify(databasePath)};
          const client = await initSemanticIndex({ databasePath });
          try {
            const embedText = async ({ text }) => new Float32Array(384).fill(0.1);
            const summary = await buildEmbeddingIndex({
              client,
              embedText,
              dimension: 384,
              modelSha256: 'fake-sha256',
              modelId: 'fake-model',
              files: ['does-not-exist.md'],
            });
            await client.close();
            console.log(JSON.stringify(summary));
          } catch (error) {
            await client.close();
            throw error;
          }
        `);
        return result;
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    }

    it('reports zero embedded and all skipped when none of the requested files exist', async () => {
      const result = await runNonExistentPathTest();
      expect(result.report).toEqual(
        expect.objectContaining({ embedded: 0, skipped: 2 }),
      );
    });

    it('emits a stderr warning naming the missing file', async () => {
      const result = await runNonExistentPathTest();
      expect(result.stderr).toMatch(
        /No matching documents found for: does-not-exist\.md/,
      );
    });
  });
});
