/**
 * @module index-stats-extended.red.test
 * @description Red tests for the Step 21 extension to index_stats that adds
 * optional metadata coverage statistics.
 */

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createFileSchemaClient } from './turso-test-helpers.mjs';

async function setupDb() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'index-stats-ext-test-'));
  const dbPath = path.join(tempDir, 'test.sqlite');
  const client = await createFileSchemaClient(dbPath);
  await client.execute(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer, test_coverage, source_path_pattern)
    VALUES
      ('src/network.ts', 'ts-source', 0, 100, 'a', 1, 'network', 'full', 'src'),
      ('src/methods.ts', 'ts-source', 0, 100, 'b', 1, 'methods', 'partial', 'src'),
      ('README.md', 'docs', 0, 100, 'c', 1, NULL, 'unknown', 'root');
  `);
  const docsResult = await client.execute('SELECT doc_id FROM documents');
  const docs = docsResult.rows;
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth, arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern)
          VALUES (?, 0, 'body', 0, 4, 0, 'network', 'good', 12, 3, 'full', 'src')`,
    args: [docs[0].doc_id],
  });
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth, arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern)
          VALUES (?, 0, 'body', 0, 4, 0, NULL, 'none', NULL, NULL, 'unknown', 'src')`,
    args: [docs[1].doc_id],
  });
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth, arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern)
          VALUES (?, 0, 'body', 0, 4, 0, 'methods', 'weak', 5, 1, 'partial', 'root')`,
    args: [docs[2].doc_id],
  });
  return { client, tempDir };
}

async function teardown(client, tempDir) {
  await client.close();
  await rm(tempDir, {
    recursive: true,
    force: true,
    maxRetries: 10,
    retryDelay: 200,
  });
}

describe('index-stats extended', () => {
  describe('metadata_coverage', () => {
    it('does not include metadata_coverage by default', async () => {
      const { indexStats } = await import('../tools/index-stats.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await indexStats({ client });

        expect(result).not.toHaveProperty('metadata_coverage');
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('returns metadata_coverage when include_metadata_coverage is true', async () => {
      const { indexStats } = await import('../tools/index-stats.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await indexStats({
          client,
          include_metadata_coverage: true,
        });

        expect(result).toEqual(
          expect.objectContaining({
            metadata_coverage: expect.any(Object),
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('reports chunk-level column coverage with total and percent', async () => {
      const { indexStats } = await import('../tools/index-stats.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await indexStats({
          client,
          include_metadata_coverage: true,
        });

        expect(result.metadata_coverage.chunks).toEqual(
          expect.objectContaining({
            arch_layer: expect.objectContaining({
              total: expect.any(Number),
              percent: expect.any(Number),
            }),
            jsdoc_quality: expect.objectContaining({
              total: expect.any(Number),
              percent: expect.any(Number),
            }),
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('reports document-level column coverage with distribution', async () => {
      const { indexStats } = await import('../tools/index-stats.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await indexStats({
          client,
          include_metadata_coverage: true,
        });

        expect(result.metadata_coverage.documents).toEqual(
          expect.objectContaining({
            arch_layer: expect.objectContaining({
              total: expect.any(Number),
              percent: expect.any(Number),
              distribution: expect.any(Object),
            }),
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('keeps the default response identical to the pre-extension shape', async () => {
      const { indexStats } = await import('../tools/index-stats.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await indexStats({ client });

        expect(Object.keys(result).sort()).toEqual([
          'feedback_stats',
          'last_build_timestamp',
          'total_chunks',
          'total_documents',
          'total_families',
        ]);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });
});
