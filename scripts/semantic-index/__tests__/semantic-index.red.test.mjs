import { createHash } from 'node:crypto';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

describe('semantic-index red contracts', () => {
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

      const { getFreshnessProof } = await import('../freshness.mjs');
      const proof = await getFreshnessProof(fixturePath);
      await rm(fixtureDirectory, { recursive: true, force: true });

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
      const { chunkMarkdown } = await import('../chunker.mjs');

      const chunks = chunkMarkdown(sampleMarkdown, {
        maxChars: 90,
        overlapChars: 0,
      });

      expect(chunks.map(({ heading_path }) => heading_path)).toEqual([
        '# Semantic Index',
        '# Semantic Index > ## Scope',
        '# Semantic Index > ## Scope > ### Freshness',
      ]);
    });
  });

  describe('validate-index.mjs', () => {
    it('reports empty and stale indexes as invalid while accepting a fresh index', async () => {
      const { validateSemanticIndex } = await import('../validate-index.mjs');
      const validationCases = [
        await validateSemanticIndex({ documents: [], freshnessChecks: [] }),
        await validateSemanticIndex({
          documents: [
            {
              file_path: 'README.md',
              mtime_ms: 1,
              file_size: 1,
              sha256: 'stale',
            },
          ],
          freshnessChecks: [
            {
              file_path: 'README.md',
              mtime_ms: 2,
              file_size: 1,
              sha256: 'fresh',
            },
          ],
        }),
        await validateSemanticIndex({
          documents: [
            {
              file_path: 'README.md',
              mtime_ms: 2,
              file_size: 1,
              sha256: 'fresh',
            },
          ],
          freshnessChecks: [
            {
              file_path: 'README.md',
              mtime_ms: 2,
              file_size: 1,
              sha256: 'fresh',
            },
          ],
        }),
      ];

      expect(validationCases.map(({ ok }) => ok)).toEqual([false, false, true]);
    });
  });
});
