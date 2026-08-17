import { jest } from '@jest/globals';
import path from 'node:path';
import { updateRag } from './update-rag.mjs';

describe('update-rag.mjs additional coverage', () => {
  describe('updateRag — error and cancellation paths', () => {
    it('catches exceptions thrown by spawnRunner and marks stage as failed', async () => {
      const summary = await updateRag({
        spawnRunner: () => {
          throw new Error('boom');
        },
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.ok).toBe(false);
      const buildStage = summary.stages.find((s) => s.name === 'build');
      expect(buildStage.status).toBe('failed');
      expect(summary.error).toContain('boom');
    });

    it('catches non-Error exceptions thrown by spawnRunner', async () => {
      const summary = await updateRag({
        spawnRunner: () => {
          throw 'string error';
        },
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.ok).toBe(false);
      expect(summary.error).toContain('string error');
    });

    it('cancels subsequent stages after a failure', async () => {
      const summary = await updateRag({
        spawnRunner: ({ name }) => {
          if (name === 'build') return { status: 1 };
          return { status: 0 };
        },
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      const prewarmStage = summary.stages.find((s) => s.name === 'prewarm-embed');
      expect(prewarmStage.status).toBe('cancelled');
      const graphStage = summary.stages.find((s) => s.name === 'build-graph');
      expect(graphStage.status).toBe('cancelled');
    });

    it('reports failed stage with no stdout/stderr (no extra error detail)', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 1 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.ok).toBe(false);
      // Error should NOT contain stdout/stderr sections
      const buildStage = summary.stages.find((s) => s.name === 'build');
      expect(buildStage.status).toBe('failed');
      expect(summary.error).not.toContain('--- build stdout ---');
    });

    it('reports failed stage with stdout/stderr in error detail', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 1, stdout: 'out msg', stderr: 'err msg' }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.error).toContain('--- build stdout ---');
      expect(summary.error).toContain('out msg');
      expect(summary.error).toContain('--- build stderr ---');
      expect(summary.error).toContain('err msg');
    });

    it('writes corpus hash after successful build-graph in non-dry-run mode', async () => {
      let writtenHash = null;
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0 }),
        readHashFile: async () => null,
        writeHashFile: async (hash) => {
          writtenHash = hash;
        },
        computeCorpusHash: async () => 'new-hash-abc',
      });

      const graphStage = summary.stages.find((s) => s.name === 'build-graph');
      expect(graphStage.status).toBe('ok');
      expect(writtenHash).toBe('new-hash-abc');
    });

    it('does not write corpus hash when computeCorpusHash returns null', async () => {
      let hashWritten = false;
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0 }),
        readHashFile: async () => null,
        writeHashFile: async () => {
          hashWritten = true;
        },
        computeCorpusHash: async () => null,
      });

      const graphStage = summary.stages.find((s) => s.name === 'build-graph');
      expect(graphStage.status).toBe('ok');
      expect(hashWritten).toBe(false);
    });

    it('does not write hash in dry-run mode', async () => {
      let hashWritten = false;
      const summary = await updateRag({
        dryRun: true,
        spawnRunner: () => ({ status: 0 }),
        readHashFile: async () => null,
        writeHashFile: async () => {
          hashWritten = true;
        },
        computeCorpusHash: async () => 'hash',
      });

      expect(hashWritten).toBe(false);
    });

    it('does not write hash when build-graph is skipped', async () => {
      let hashWritten = false;
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0 }),
        readHashFile: async () => 'same-hash',
        writeHashFile: async () => {
          hashWritten = true;
        },
        computeCorpusHash: async () => 'same-hash',
      });

      const graphStage = summary.stages.find((s) => s.name === 'build-graph');
      expect(graphStage.status).toBe('skipped');
      expect(hashWritten).toBe(false);
    });

    it('does not write hash when build-graph fails', async () => {
      let hashWritten = false;
      const summary = await updateRag({
        spawnRunner: ({ name }) => {
          if (name === 'build-graph') return { status: 1 };
          return { status: 0 };
        },
        readHashFile: async () => null,
        writeHashFile: async () => {
          hashWritten = true;
        },
        computeCorpusHash: async () => 'hash',
      });

      expect(hashWritten).toBe(false);
    });

    it('skips build-graph when stored hash equals current hash but other stages fail', async () => {
      const summary = await updateRag({
        spawnRunner: ({ name }) => {
          if (name === 'build') return { status: 1 };
          return { status: 0 };
        },
        readHashFile: async () => 'same-hash',
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'same-hash',
      });

      const graphStage = summary.stages.find((s) => s.name === 'build-graph');
      // build fails, build-graph is cancelled (not skipped)
      expect(graphStage.status).toBe('cancelled');
    });

    it('handles status being null from spawnRunner', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: null }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash',
      });

      // null status → null ?? 1 = 1 → Number(1) = 1 → 1 !== 0 → failed
      expect(summary.ok).toBe(false);
    });

    it('handles status being undefined from spawnRunner', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({}),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash',
      });

      // undefined status → Number(undefined) = NaN → NaN !== 0 → failed
      // Actually, Number(undefined) = NaN, and NaN !== 0 is true → failed
      // Wait, let me re-check: `const exitStatus = Number(result?.status ?? 1);`
      // result?.status is undefined, undefined ?? 1 = 1, Number(1) = 1, 1 !== 0 → failed
      expect(summary.ok).toBe(false);
    });
  });

  describe('updateRag — default (non-injected) paths', () => {
    it('uses real computeCorpusHash and readStoredCorpusHash when not injected', async () => {
      // This test exercises the real internal functions.
      // computeCorpusHash checks existsSync(defaultDatabasePath) → likely false in test env → returns null
      // readStoredCorpusHash reads CORPUS_HASH_PATH → likely doesn't exist → returns null
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0 }),
        // Don't inject readHashFile, writeHashFile, computeCorpusHash
      });

      // Should still complete; build-graph may be 'ok' or 'skipped' depending on
      // whether the real corpus-hash.json matches the current corpus hash
      const graphStage = summary.stages.find((s) => s.name === 'build-graph');
      expect(['ok', 'skipped']).toContain(graphStage.status);
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
        'update-rag.mjs',
      );
      process.argv = ['node', scriptPath, '--help'];

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./update-rag.mjs?cli-test=${Date.now()}`);

      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).toContain('RAG pipeline orchestrator');
      logSpy.mockRestore();
    });

    it('runs pipeline in dry-run mode', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'update-rag.mjs',
      );
      process.argv = ['node', scriptPath, '--dry-run'];

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      process.exitCode = undefined;
      await import(`./update-rag.mjs?cli-test=${Date.now()}-2`);

      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).toContain('build');
      logSpy.mockRestore();
    });

    it('runs pipeline with --json flag', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'update-rag.mjs',
      );
      process.argv = ['node', scriptPath, '--json', '--dry-run'];

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./update-rag.mjs?cli-test=${Date.now()}-3`);

      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).toContain('"stages"');
      logSpy.mockRestore();
    });
  });
});