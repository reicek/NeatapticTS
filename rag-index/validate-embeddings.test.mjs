import { jest } from '@jest/globals';
import path from 'node:path';
import { validateEmbeddings } from './validate-embeddings.mjs';

function createMockClient(chunkCount, embeddingCount) {
  const calls = [];
  const client = {
    execute: jest.fn(async ({ sql }) => {
      calls.push(sql);
      if (sql.includes('WHERE embedding IS NOT NULL')) {
        return { rows: [{ count: embeddingCount }] };
      }
      return { rows: [{ count: chunkCount }] };
    }),
    close: jest.fn(async () => {}),
  };
  client._calls = calls;
  return client;
}

describe('validate-embeddings.mjs', () => {
  describe('validateEmbeddings', () => {
    it('returns pass=true when embeddings exist for the model', async () => {
      const client = createMockClient(100, 50);
      const result = await validateEmbeddings({ client, modelId: 'test-model' });
      expect(result.pass).toBe(true);
      expect(result.chunk_count).toBe(100);
      expect(result.embedding_count).toBe(50);
      expect(result.evidence).toEqual([]);
      expect(result.fixHint).toBe(null);
      expect(result.owner).toBe('05-green-testing');
    });

    it('returns pass=false when no embeddings exist for the model', async () => {
      const client = createMockClient(100, 0);
      const result = await validateEmbeddings({ client, modelId: 'test-model' });
      expect(result.pass).toBe(false);
      expect(result.chunk_count).toBe(100);
      expect(result.embedding_count).toBe(0);
      expect(result.evidence).toHaveLength(1);
      expect(result.evidence[0].issue).toBe('no embeddings for model');
      expect(result.evidence[0].model_id).toBe('test-model');
      expect(result.evidence[0].actual).toBe(0);
      expect(result.evidence[0].expected).toBe(100);
      expect(result.fixHint).toBe('Run: node rag-index/embed-index.mjs');
    });

    it('defaults modelId to all-MiniLM-L6-v2', async () => {
      const client = createMockClient(10, 5);
      const result = await validateEmbeddings({ client });
      expect(result.pass).toBe(true);
      // Verify the model id was used in the query
      expect(client.execute).toHaveBeenCalledWith(
        expect.objectContaining({ args: ['all-MiniLM-L6-v2'] }),
      );
    });

    it('handles zero chunks', async () => {
      const client = createMockClient(0, 0);
      const result = await validateEmbeddings({ client });
      expect(result.pass).toBe(false);
      expect(result.chunk_count).toBe(0);
    });
  });

  describe('main (CLI entry point)', () => {
    let originalArgv;
    let originalExitCode;

    beforeEach(() => {
      originalArgv = process.argv;
      originalExitCode = process.exitCode;
    });

    afterEach(() => {
      process.argv = originalArgv;
      process.exitCode = originalExitCode;
    });

    it('prints help when --help is passed', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-embeddings.mjs',
      );
      process.argv = ['node', scriptPath, '--help'];

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => undefined);
      await import(`./validate-embeddings.mjs?cli-test=${Date.now()}`);

      const output = logSpy.mock.calls.map((c) => c[0]).join('');
      expect(output).toContain('Embeddings validator');
      logSpy.mockRestore();
    });

    it('runs validation and outputs text result', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-embeddings.mjs',
      );
      // Use a non-existent database to trigger error path
      process.argv = ['node', scriptPath, '--database=nonexistent-db.sqlite'];

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => undefined);
      const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => undefined);
      await import(`./validate-embeddings.mjs?cli-test=${Date.now()}-2`);

      const output = [...logSpy.mock.calls, ...errorSpy.mock.calls].map((c) => String(c[0])).join('');
      expect(output).toBeTruthy();
      logSpy.mockRestore();
      errorSpy.mockRestore();
    });

    it('runs validation with --json flag', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-embeddings.mjs',
      );
      process.argv = ['node', scriptPath, '--json', '--database=nonexistent-db.sqlite'];

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => undefined);
      await import(`./validate-embeddings.mjs?cli-test=${Date.now()}-3`);

      const output = logSpy.mock.calls.map((c) => String(c[0])).join('');
      expect(output).toBeTruthy();
      logSpy.mockRestore();
    });
  });
});