import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

interface MarkdownChunk {
  heading_path: string;
}

interface DeletionCleanupResult {
  firstSummary: { scanned: number; indexed: number; purged: number };
  secondSummary: { scanned: number; indexed: number; purged: number };
  counts: { documents: number; chunks: number; ftsRows: number };
}

const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: path.resolve(__dirname, '..', '..', '..'),
      encoding: 'utf8',
    },
  );

  return JSON.parse(output) as Result;
};

describe('semantic-index red contracts', () => {
  describe('cli-utils.mjs', () => {
    it('accumulates repeatable source flags when requested', () => {
      const parsedArgs = runModuleEvaluation<{
        _: string[];
        source: string[];
      }>(`
        import { parseCliArgs } from './scripts/semantic-index/cli-utils.mjs';

        const parsedArgs = parseCliArgs([
          '--json',
          '--source',
          'scripts/agent-customization/mcp/mcp-utils.mjs',
          '--source=scripts/agent-customization/mcp/mcp-plan-utils.mjs',
        ], {
          repeatableFlags: ['source'],
        });
        console.log(JSON.stringify(parsedArgs));
      `);

      expect(parsedArgs).toEqual(
        expect.objectContaining({
          source: [
            'scripts/agent-customization/mcp/mcp-utils.mjs',
            'scripts/agent-customization/mcp/mcp-plan-utils.mjs',
          ],
        }),
      );
    });

    it('preserves scalar flag parsing when repeatable mode is not enabled', () => {
      const parsedArgs = runModuleEvaluation<{ _: string[]; source: string }>(`
        import { parseCliArgs } from './scripts/semantic-index/cli-utils.mjs';

        const parsedArgs = parseCliArgs(['--source', 'scripts/semantic-index/cli-utils.mjs']);
        console.log(JSON.stringify(parsedArgs));
      `);

      expect(parsedArgs).toEqual(
        expect.objectContaining({
          source: 'scripts/semantic-index/cli-utils.mjs',
        }),
      );
    });
  });

  describe('freshness.mjs', () => {
    it('returns mtime_ms, size, and sha256 for a known file', async () => {
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-index-freshness-'),
      );
      const fixturePath = path.join(fixtureDirectory, 'known.md');
      const fixtureText = '# Known\n\nFreshness proof fixture.\n';
      await writeFile(fixturePath, fixtureText, 'utf8');
      const fileStats = await stat(fixturePath);
      const expectedProof = {
        mtime_ms: Math.trunc(fileStats.mtimeMs),
        size: Buffer.byteLength(fixtureText),
        sha256: createHash('sha256').update(fixtureText).digest('hex'),
      };

      const proof = runModuleEvaluation<typeof expectedProof>(`
        import { getFreshnessProof } from './scripts/semantic-index/freshness.mjs';
        const proof = await getFreshnessProof(${JSON.stringify(fixturePath)});
        console.log(JSON.stringify(proof));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });

      expect(proof).toEqual(expectedProof);
    });
  });

  describe('chunker.mjs', () => {
    it('emits stable chunk count and heading paths for markdown sections', async () => {
      const sampleMarkdown = [
        '# Semantic Index',
        '',
        'Opening summary for the corpus.',
        '',
        '## Scope',
        '',
        'The scanner captures generated docs and plans.',
        '',
        '### Freshness',
        '',
        'Freshness proofs prevent stale search results.',
      ].join('\n');
      const chunks = runModuleEvaluation<MarkdownChunk[]>(`
        import { chunkMarkdown } from './scripts/semantic-index/chunker.mjs';
        const chunks = chunkMarkdown(${JSON.stringify(sampleMarkdown)}, { maxChars: 90, overlapChars: 0 });
        console.log(JSON.stringify(chunks));
      `);

      expect(
        chunks.map(({ heading_path }: MarkdownChunk) => heading_path),
      ).toEqual([
        '# Semantic Index',
        '# Semantic Index > ## Scope',
        '# Semantic Index > ## Scope > ### Freshness',
      ]);
    });
  });

  describe('validate-index.mjs', () => {
    it('reports empty and stale indexes as invalid while accepting a fresh index', async () => {
      const validationCases = runModuleEvaluation<Array<{ ok: boolean }>>(`
        import { validateSemanticIndex } from './scripts/semantic-index/validate-index.mjs';
        const validationCases = [
          await validateSemanticIndex({ documents: [], freshnessChecks: [] }),
          await validateSemanticIndex({
            documents: [{ file_path: 'README.md', mtime_ms: 1, file_size: 1, sha256: 'stale' }],
            freshnessChecks: [{ file_path: 'README.md', mtime_ms: 2, file_size: 1, sha256: 'fresh' }],
          }),
          await validateSemanticIndex({
            documents: [{ file_path: 'README.md', mtime_ms: 2, file_size: 1, sha256: 'fresh' }],
            freshnessChecks: [{ file_path: 'README.md', mtime_ms: 2, file_size: 1, sha256: 'fresh' }],
          }),
        ];
        console.log(JSON.stringify(validationCases));
      `);

      expect(validationCases.map(({ ok }) => ok)).toEqual([false, false, true]);
    });
  });

  describe('build-index.mjs', () => {
    it('purges document, chunk, and FTS rows when a previously indexed file disappears from the scan', async () => {
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-index-build-'),
      );
      const databasePath = path.join(fixtureDirectory, 'corpus.sqlite');

      const deletionCleanup = runModuleEvaluation<DeletionCleanupResult>(`
        import { createClient } from '@libsql/client';
        import { pathToFileURL } from 'node:url';
        import { buildSemanticIndex } from './scripts/semantic-index/build-index.mjs';

        const databasePath = ${JSON.stringify(databasePath)};
        const firstSummary = await buildSemanticIndex({
          databasePath,
          force: true,
          corpusDocuments: [{ filePath: 'README.md', family: 'root-doc' }],
        });
        const secondSummary = await buildSemanticIndex({ databasePath, force: true, corpusDocuments: [] });

        const client = createClient({ url: pathToFileURL(databasePath).href });
        const documentsResult = await client.execute('SELECT COUNT(*) AS documents FROM documents');
        const chunksResult = await client.execute('SELECT COUNT(*) AS chunks FROM chunks');
        const ftsRowsResult = await client.execute('SELECT COUNT(*) AS ftsRows FROM chunks_fts');
        const [{ documents }] = documentsResult.rows;
        const [{ chunks }] = chunksResult.rows;
        const [{ ftsRows }] = ftsRowsResult.rows;
        await client.close();

        console.log(JSON.stringify({ firstSummary, secondSummary, counts: { documents, chunks, ftsRows } }));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });

      expect(deletionCleanup).toEqual({
        firstSummary: expect.objectContaining({
          scanned: 1,
          indexed: 1,
          purged: 0,
        }),
        secondSummary: expect.objectContaining({
          scanned: 0,
          indexed: 0,
          purged: 1,
        }),
        counts: { documents: 0, chunks: 0, ftsRows: 0 },
      });
    });
  });
});
