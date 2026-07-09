import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createClient } from '@libsql/client';
import { pathToFileURL } from 'node:url';

// Tests run from the repository root, so cwd is a stable anchor for repo-relative
// paths without needing import.meta.url (which currently breaks ts-jest for the
// semantic-index-scripts project).
const REPO_ROOT = path.resolve();

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 * Used to import `.mjs` implementation modules that do not yet exist in the
 * red phase and to run them against real fixtures.
 *
 * @param source - ESM source string executed as `--input-type=module --eval`.
 * @returns The JSON-parsed value the snippet wrote to stdout.
 */
const runModuleEvaluation = <Result>(source: string): Result => {
  let stderr = '';
  try {
    const output = execFileSync(
      process.execPath,
      ['--input-type=module', '--eval', source],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 60000,
        stdio: ['pipe', 'pipe', 'pipe'],
      },
    );
    const trimmed = output.trim();
    if (trimmed.length === 0) {
      throw new Error('Module evaluation produced empty output');
    }
    return JSON.parse(trimmed) as Result;
  } catch (error) {
    if (error instanceof Error && 'stderr' in error && typeof error.stderr === 'string') {
      stderr = error.stderr;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Module evaluation failed: ${message}${stderr ? `\nstderr: ${stderr}` : ''}`);
  }
};

interface TempFixture {
  /** Absolute filesystem path of the temp directory. */
  dir: string;
  /** Absolute filesystem path of the fixture file inside the temp directory. */
  absolutePath: string;
  /** Repo-relative POSIX path of the fixture file. */
  filePath: string;
}

/**
 * Create an isolated temp directory inside the repo root so that repo-relative
 * paths remain valid for `build-index.mjs` and the freshness hook.
 *
 * @param prefix - Directory name prefix, e.g. `'freshness-bm25'`.
 * @returns Fixture descriptor. The caller is responsible for deleting `dir`.
 */
async function createTempFixture(prefix: string): Promise<TempFixture> {
  const dir = await mkdtemp(path.join(REPO_ROOT, `${prefix}-`));
  const absolutePath = path.join(dir, 'fixture.md');
  const filePath = path.relative(REPO_ROOT, absolutePath).replaceAll('\\', '/');
  return { absolutePath, dir, filePath };
}

/**
 * Seed a single markdown document into a fresh semantic-index database.
 * Forces a full index so the document row and chunks are present for later
 * incremental hook tests.
 *
 * @param databasePath - Absolute path to the temporary corpus database.
 * @param fixture - Fixture descriptor created by {@link createTempFixture}.
 * @param content - Markdown content to write to the fixture file.
 */
async function seedDocument(
  databasePath: string,
  fixture: TempFixture,
  content: string,
): Promise<void> {
  await writeFile(fixture.absolutePath, content, 'utf8');
  runModuleEvaluation<{ indexed: number }>(`
    import { buildSemanticIndex } from './rag-index/build-index.mjs';
    const summary = await buildSemanticIndex({
      databasePath: ${JSON.stringify(databasePath)},
      corpusDocuments: [
        { filePath: ${JSON.stringify(fixture.filePath)}, family: 'root-doc' },
      ],
      force: true,
    });
    console.log(JSON.stringify({ indexed: summary.indexed, chunks: summary.chunks }));
  `);
}

/**
 * Read the stored freshness proof and `indexed_at` for a single document row.
 *
 * @param databasePath - Absolute path to the corpus database.
 * @param filePath - Repo-relative path of the document.
 * @returns The stored row, or `null` if the document is missing.
 */
async function getDocumentRow(databasePath: string, filePath: string) {
  const client = createClient({ url: pathToFileURL(databasePath).href });
  try {
    const result = await client.execute({
      sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents WHERE file_path = ?',
      args: [filePath],
    });
    return (
      (result.rows[0] as unknown as
        | {
            file_path: string;
            mtime_ms: number;
            file_size: number;
            sha256: string;
            indexed_at: number;
          }
        | undefined) ?? null
    );
  } finally {
    await client.close();
  }
}

/**
 * Compute the freshness proof for a file on disk. Mirrors `freshness.mjs`
 * so tests can assert that the stored DB proof matches the new on-disk state
 * after the hook runs.
 *
 * @param absolutePath - Absolute filesystem path.
 * @returns Freshness proof object.
 */
async function computeFreshnessProof(absolutePath: string) {
  const [fileStats, fileBuffer] = await Promise.all([
    stat(absolutePath),
    readFile(absolutePath),
  ]);
  return {
    mtime_ms: Math.trunc(fileStats.mtimeMs),
    file_size: fileBuffer.byteLength,
    sha256: createHash('sha256').update(fileBuffer).digest('hex'),
  };
}

describe('freshness-hooks.mjs', () => {
  describe('createFreshnessHook', () => {
    describe('trigger behavior', () => {
      it('runs an incremental update after a file write notification', () => {
        const result = runModuleEvaluation<string[][]>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const calls = [];
          const hook = createFreshnessHook({
            debounce_ms: 0,
            runIncrementalBuild: async (paths) => {
              calls.push(paths);
              return { updated: paths, failed: [] };
            },
          });
          hook.notifyWrite('src/foo.ts');
          await hook.flush();
          console.log(JSON.stringify(calls));
        `);

        expect(result).toEqual([['src/foo.ts']]);
      });

      it('runs an incremental update after a file rename notification', () => {
        const result = runModuleEvaluation<string[][]>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const calls = [];
          const hook = createFreshnessHook({
            debounce_ms: 0,
            runIncrementalBuild: async (paths) => {
              calls.push(paths);
              return { updated: paths, failed: [] };
            },
          });
          hook.notifyRename('src/old.ts', 'src/new.ts');
          await hook.flush();
          console.log(JSON.stringify(calls));
        `);

        expect(result).toEqual([['src/old.ts', 'src/new.ts']]);
      });

      it('runs an incremental update after a file delete notification', () => {
        const result = runModuleEvaluation<string[][]>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const calls = [];
          const hook = createFreshnessHook({
            debounce_ms: 0,
            runIncrementalBuild: async (paths) => {
              calls.push(paths);
              return { updated: paths, failed: [] };
            },
          });
          hook.notifyDelete('src/removed.ts');
          await hook.flush();
          console.log(JSON.stringify(calls));
        `);

        expect(result).toEqual([['src/removed.ts']]);
      });
    });

    describe('debounce and batching', () => {
      it('batches two rapid writes into a single incremental run', async () => {
        const result = runModuleEvaluation<string[][]>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const calls = [];
          const hook = createFreshnessHook({
            debounce_ms: 50,
            runIncrementalBuild: async (paths) => {
              calls.push(paths);
              return { updated: paths, failed: [] };
            },
          });
          hook.notifyWrite('src/a.ts');
          hook.notifyWrite('src/b.ts');
          await hook.flush();
          console.log(JSON.stringify(calls));
        `);

        expect(result).toEqual([['src/a.ts', 'src/b.ts']]);
      });
    });

    describe('BM25 immediacy and embedding queue', () => {
      it('reflects changed file content in BM25 search immediately after flush', async () => {
        const fixture = await createTempFixture('freshness-bm25');
        const databasePath = path.join(fixture.dir, 'corpus.sqlite');
        const uniqueToken = 'UniqueRedTokenBm25';

        try {
          await seedDocument(
            databasePath,
            fixture,
            '# Original\n\nOriginal fixture content.',
          );
          await new Promise((resolve) => setTimeout(resolve, 20));
          await writeFile(
            fixture.absolutePath,
            `# Updated\n\n${uniqueToken} appears in the updated fixture.`,
            'utf8',
          );

          runModuleEvaluation<{ flushed: true }>(`
            import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
            const hook = createFreshnessHook({
              databasePath: ${JSON.stringify(databasePath)},
              debounce_ms: 0,
              skip_ann: true,
            });
            hook.notifyWrite(${JSON.stringify(fixture.filePath)});
            await hook.flush();
            console.log(JSON.stringify({ flushed: true }));
          `);

          const search = runModuleEvaluation<{
            results: Array<{ file_path: string; text: string }>;
          }>(`
            import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
            const result = await searchCorpus({
              query: ${JSON.stringify(uniqueToken)},
              databasePath: ${JSON.stringify(databasePath)},
              use_dense: false,
              limit: 10,
            });
            console.log(JSON.stringify({ results: result.results.map(r => ({ file_path: r.file_path, text: r.text })) }));
          `);

          const matchedFilePaths = search.results.map((r) => r.file_path);
          expect(matchedFilePaths).toContain(fixture.filePath);
        } finally {
          await rm(fixture.dir, {
            recursive: true,
            force: true,
            maxRetries: 5,
            retryDelay: 200,
          });
        }
      });

      it('does not block flush while embedding updates are queued', async () => {
        const fixture = await createTempFixture('freshness-skip-ann');
        const databasePath = path.join(fixture.dir, 'corpus.sqlite');

        try {
          await seedDocument(
            databasePath,
            fixture,
            '# Skip Ann\n\nEmbedding updates should be queued.',
          );
          await new Promise((resolve) => setTimeout(resolve, 20));
          await writeFile(
            fixture.absolutePath,
            '# Skip Ann Updated\n\nEmbedding updates are still queued.',
            'utf8',
          );

          const result = runModuleEvaluation<{
            elapsedMs: number;
            flushed: true;
          }>(`
            import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
            const hook = createFreshnessHook({
              databasePath: ${JSON.stringify(databasePath)},
              debounce_ms: 0,
              skip_ann: true,
            });
            const start = Date.now();
            hook.notifyWrite(${JSON.stringify(fixture.filePath)});
            await hook.flush();
            console.log(JSON.stringify({ elapsedMs: Date.now() - start, flushed: true }));
          `);

          expect(result.elapsedMs).toBeLessThan(5000);
        } finally {
          await rm(fixture.dir, {
            recursive: true,
            force: true,
            maxRetries: 5,
            retryDelay: 200,
          });
        }
      });
    });

    describe('freshness proof', () => {
      it('updates stored indexed_at and freshness proof after a successful update', async () => {
        const fixture = await createTempFixture('freshness-proof');
        const databasePath = path.join(fixture.dir, 'corpus.sqlite');

        try {
          await seedDocument(
            databasePath,
            fixture,
            '# First\n\nFirst revision of the fixture.',
          );
          const before = await getDocumentRow(databasePath, fixture.filePath);
          expect(before).not.toBeNull();

          await new Promise((resolve) => setTimeout(resolve, 30));
          await writeFile(
            fixture.absolutePath,
            '# Second\n\nSecond revision of the fixture.',
            'utf8',
          );
          const diskProof = await computeFreshnessProof(fixture.absolutePath);

          runModuleEvaluation<{ flushed: true }>(`
            import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
            const hook = createFreshnessHook({
              databasePath: ${JSON.stringify(databasePath)},
              debounce_ms: 0,
              skip_ann: true,
            });
            hook.notifyWrite(${JSON.stringify(fixture.filePath)});
            await hook.flush();
            console.log(JSON.stringify({ flushed: true }));
          `);

          const after = await getDocumentRow(databasePath, fixture.filePath);
          expect(after).toEqual(
            expect.objectContaining({
              file_path: fixture.filePath,
              mtime_ms: diskProof.mtime_ms,
              file_size: diskProof.file_size,
              sha256: diskProof.sha256,
            }),
          );
          expect(Number(after?.indexed_at)).toBeGreaterThan(
            Number(before?.indexed_at),
          );
        } finally {
          await rm(fixture.dir, {
            recursive: true,
            force: true,
            maxRetries: 5,
            retryDelay: 200,
          });
        }
      });
    });

    describe('failure handling', () => {
      it('logs a warning and resolves without throwing when the incremental update fails', () => {
        const result = runModuleEvaluation<string[]>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const warnings = [];
          const hook = createFreshnessHook({
            debounce_ms: 0,
            runIncrementalBuild: async () => {
              throw new Error('simulated build failure');
            },
            logWarning: (message) => warnings.push(message),
          });
          hook.notifyWrite('src/fail.ts');
          await hook.flush();
          console.log(JSON.stringify(warnings));
        `);

        expect(result).toEqual([
          expect.stringContaining('simulated build failure'),
        ]);
      });
    });

    describe('configuration', () => {
      it('honors debounce_ms before flushing', () => {
        const result = runModuleEvaluation<{ elapsedMs: number }>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const calls = [];
          const hook = createFreshnessHook({
            debounce_ms: 100,
            runIncrementalBuild: async (paths) => {
              calls.push(paths);
              return { updated: paths, failed: [] };
            },
          });
          const start = Date.now();
          hook.notifyWrite('src/delayed.ts');
          // Intentionally do not await flush; the debounce window should pass before the build runs.
          await new Promise(resolve => setTimeout(resolve, 200));
          console.log(JSON.stringify({ elapsedMs: Date.now() - start, calls }));
        `);

        expect(result.elapsedMs).toBeGreaterThanOrEqual(90);
      });

      it('ignores file paths that do not match changed_file_globs', () => {
        const result = runModuleEvaluation<string[][]>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const calls = [];
          const hook = createFreshnessHook({
            debounce_ms: 0,
            changed_file_globs: ['src/**/*.ts'],
            runIncrementalBuild: async (paths) => {
              calls.push(paths);
              return { updated: paths, failed: [] };
            },
          });
          hook.notifyWrite('docs/readme.md');
          hook.notifyWrite('src/match.ts');
          await hook.flush();
          console.log(JSON.stringify(calls));
        `);

        expect(result).toEqual([['src/match.ts']]);
      });

      it('passes skip_ann through to the incremental build path', () => {
        const result = runModuleEvaluation<boolean[]>(`
          import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
          const flags = [];
          const hook = createFreshnessHook({
            debounce_ms: 0,
            skip_ann: true,
            runIncrementalBuild: async (paths, options) => {
              flags.push(Boolean(options?.skip_ann));
              return { updated: paths, failed: [] };
            },
          });
          hook.notifyWrite('src/no-embed.ts');
          await hook.flush();
          console.log(JSON.stringify(flags));
        `);

        expect(result).toEqual([true]);
      });
    });

    describe('regression search', () => {
      it('returns updated chunk content via searchCorpus after a tracked file is edited and flushed', async () => {
        const fixture = await createTempFixture('freshness-regression');
        const databasePath = path.join(fixture.dir, 'corpus.sqlite');
        const uniqueToken = 'UniqueRedTokenRegression';

        try {
          await seedDocument(
            databasePath,
            fixture,
            '# Stable\n\nStable fixture content.',
          );
          await new Promise((resolve) => setTimeout(resolve, 20));
          await writeFile(
            fixture.absolutePath,
            `# Changed\n\n${uniqueToken} is now present.`,
            'utf8',
          );

          runModuleEvaluation<{ flushed: true }>(`
            import { createFreshnessHook } from './rag-index/freshness-hooks/freshness-hooks.mjs';
            const hook = createFreshnessHook({
              databasePath: ${JSON.stringify(databasePath)},
              debounce_ms: 0,
              skip_ann: true,
            });
            hook.notifyWrite(${JSON.stringify(fixture.filePath)});
            await hook.flush();
            console.log(JSON.stringify({ flushed: true }));
          `);

          const search = runModuleEvaluation<{
            results: Array<{ text: string }>;
          }>(`
            import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
            const result = await searchCorpus({
              query: ${JSON.stringify(uniqueToken)},
              databasePath: ${JSON.stringify(databasePath)},
              use_dense: false,
              limit: 10,
            });
            console.log(JSON.stringify({ results: result.results.map(r => ({ text: r.text })) }));
          `);

          const match = search.results.find((r) =>
            r.text.includes(uniqueToken),
          );
          expect(match).toBeTruthy();
        } finally {
          await rm(fixture.dir, {
            recursive: true,
            force: true,
            maxRetries: 5,
            retryDelay: 200,
          });
        }
      });
    });
  });
});
