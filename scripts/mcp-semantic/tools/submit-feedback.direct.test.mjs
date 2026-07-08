/**
 * @module submit-feedback.direct.test
 * @description Direct-import coverage tests for submit-feedback.mjs.
 *
 * Runs in the mcp-semantic-mjs project so Jest can instrument the native ESM
 * source file via the V8 coverage provider.
 */
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { createClient } from '@libsql/client';
import { getTursoClient, closeTursoClient } from './cortex-db.mjs';

const REPO_ROOT = path.resolve();
const SHARED_TEMP_DIR = path.join(
  os.tmpdir(),
  'neat-submit-feedback-direct-mjs',
);

async function makeSharedFixture() {
  fs.rmSync(SHARED_TEMP_DIR, { recursive: true, force: true });
  fs.mkdirSync(SHARED_TEMP_DIR, { recursive: true });
  const databasePath = path.join(SHARED_TEMP_DIR, 'corpus.sqlite');
  const client = createClient({ url: 'file:' + databasePath });
  await client.executeMultiple(
    fs.readFileSync(
      path.resolve(REPO_ROOT, './rag-index/schema-turso.sql'),
      'utf8',
    ),
  );
  await client.executeMultiple(`
    INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
      VALUES (1, 'src/network.ts', 'ts-source', 1, 100, 'sha', 1);
    INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, depth)
      VALUES (42, 1, 0, 'activate', 'fixture body', 0, 12, 0);
  `);
  return { databasePath, client };
}

describe('submit-feedback.mjs direct import coverage', () => {
  let databasePath;
  let client;

  beforeAll(async () => {
    const fixture = await makeSharedFixture();
    databasePath = fixture.databasePath;
    client = fixture.client;
  });

  afterAll(async () => {
    await client.close();
  });

  it('clamps an explicit positive strength to the upper bound', async () => {
    const { submitFeedback } = await import('./submit-feedback.mjs');
    const response = await submitFeedback({
      chunk_id: 42,
      signal_type: 'positive',
      signal_strength: 50,
      agent_id: 'agent-direct',
      query: 'direct positive clamp',
      databasePath,
      client,
    });
    expect(response.signal_strength).toBeLessThanOrEqual(1);
    expect(response.feedback_score).not.toBeNull();
  });

  it('clamps an explicit negative strength to the lower bound', async () => {
    const { submitFeedback } = await import('./submit-feedback.mjs');
    const response = await submitFeedback({
      chunk_id: 42,
      signal_type: 'negative',
      signal_strength: -50,
      agent_id: 'agent-direct',
      query: 'direct negative clamp',
      databasePath,
      client,
    });
    expect(response.signal_strength).toBeGreaterThanOrEqual(-1);
  });

  it('throws MISSING_CHUNK_ID for a non-existent chunk', async () => {
    const { submitFeedback } = await import('./submit-feedback.mjs');
    await expect(
      submitFeedback({
        chunk_id: 999,
        signal_type: 'positive',
        databasePath,
        client,
      }),
    ).rejects.toThrow(/MISSING_CHUNK_ID/);
  });

  it('throws INVALID_SIGNAL_TYPE for an unsupported signal_type', async () => {
    const { submitFeedback } = await import('./submit-feedback.mjs');
    await expect(
      submitFeedback({
        chunk_id: 42,
        signal_type: 'bogus',
        databasePath,
        client,
      }),
    ).rejects.toThrow(/INVALID_SIGNAL_TYPE/);
  });

  it('throws MISSING_CHUNK_ID when called with no arguments', async () => {
    const { submitFeedback } = await import('./submit-feedback.mjs');
    await expect(submitFeedback()).rejects.toThrow(/MISSING_CHUNK_ID/);
  });

  it('falls back to getTursoClient when no client is provided', async () => {
    const memoryClient = await getTursoClient(':memory:');
    try {
      await memoryClient.executeMultiple(
        fs.readFileSync(
          path.resolve(REPO_ROOT, './rag-index/schema-turso.sql'),
          'utf8',
        ),
      );
      await memoryClient.executeMultiple(`
        INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
          VALUES (1, 'src/network.ts', 'ts-source', 1, 100, 'sha', 1);
        INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, depth)
          VALUES (42, 1, 0, 'activate', 'fixture body', 0, 12, 0);
      `);

      const { submitFeedback } = await import('./submit-feedback.mjs');
      const response = await submitFeedback({
        chunk_id: 42,
        signal_type: 'positive',
        databasePath: ':memory:',
      });
      expect(response.signal_strength).toBeLessThanOrEqual(1);
      expect(response.feedback_score).not.toBeNull();
    } finally {
      await closeTursoClient(':memory:');
    }
  });

  it('uses the default signal strength when none is provided', async () => {
    const { submitFeedback } = await import('./submit-feedback.mjs');
    const response = await submitFeedback({
      chunk_id: 42,
      signal_type: 'reference',
      databasePath,
      client,
    });
    expect(response.signal_strength).toBeGreaterThanOrEqual(-1);
    expect(response.signal_strength).toBeLessThanOrEqual(1);
  });
});
