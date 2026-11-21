import { jest } from '@jest/globals';
import path from 'node:path';
import {
  composeCorpusIdentityRows,
  computeCorpusIdentity,
  runMeasuredSubprocess,
  updateRag,
} from './update-rag.mjs';

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
      await updateRag({
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
      await updateRag({
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
      // writeHashFile IS injected: the real writer would rewrite the production
      // rag-index/data/corpus-hash.json from the test suite (V-OBS-047).
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0 }),
        writeHashFile: async () => {},
        // Don't inject readHashFile, writeHashFile, computeCorpusHash
      });

      // Should still complete; build-graph may be 'ok' or 'skipped' depending on
      // whether the real corpus-hash.json matches the current corpus hash
      const graphStage = summary.stages.find((s) => s.name === 'build-graph');
      expect(['ok', 'skipped']).toContain(graphStage.status);
    });
  });

  describe('updateRag — per-stage peak memory publication (AC-203-C)', () => {
    it('publishes a peak-memory figure for every stage it runs', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0, peakMemoryBytes: 563331072 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(
        summary.stages.map((stage) => [stage.name, stage.peakMemoryBytes]),
      ).toEqual([
        ['build', 563331072],
        ['prewarm-embed', 563331072],
        ['build-terms', 563331072],
        ['build-graph', 563331072],
        ['snapshot', 563331072],
        ['validate', 563331072],
      ]);
    });

    it('omits the figure when a stage publishes none', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(
        summary.stages.map((stage) => stage.peakMemoryBytes),
      ).toEqual([undefined, undefined, undefined, undefined, undefined, undefined]);
    });

    it('publishes the figure of a stage that failed', async () => {
      const summary = await updateRag({
        spawnRunner: ({ name }) =>
          name === 'build'
            ? { status: 1, stdout: 'out msg', stderr: 'err msg', peakRssBytes: 4242 }
            : { status: 0 },
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages.find((stage) => stage.name === 'build').peakMemoryBytes).toBe(
        4242,
      );
    });

    it('publishes no figure for a stage that was skipped or cancelled', async () => {
      const summary = await updateRag({
        spawnRunner: ({ name }) => {
          if (name === 'build') return { status: 1 };
          return { status: 0, peakMemoryBytes: 999 };
        },
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(
        summary.stages
          .filter((stage) => stage.status === 'cancelled')
          .map((stage) => stage.peakMemoryBytes),
      ).toEqual([undefined, undefined, undefined, undefined, undefined]);
    });

    it('publishes a figure published under a documented alias', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0, peakBytes: 2048 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBe(2048);
    });

    it('publishes a figure published under a peak-like key outside the documented aliases', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0, residentPeakBytes: 4096 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBe(4096);
    });

    it('publishes a figure nested one level deep in the stage result', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0, memory: { peakRssBytes: 8192 } }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBe(8192);
    });

    it('publishes a figure a stage printed as a JSON payload on stdout', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({
          status: 0,
          stdout: JSON.stringify({ peakMemoryBytes: 16384 }),
        }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBe(16384);
    });

    it('publishes a figure nested one level deep in the stdout JSON payload', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({
          status: 0,
          stdout: JSON.stringify({ metrics: { peakRssBytes: 512 } }),
        }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBe(512);
    });

    it('publishes no figure for a stdout JSON payload without a peak figure', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({
          status: 0,
          stdout: JSON.stringify({ documentsIndexed: 12 }),
        }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBeUndefined();
    });

    it('publishes no figure for a stdout payload that is not a JSON object', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0, stdout: '123' }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBeUndefined();
    });

    it('publishes a figure a stage printed as a marker on stderr', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({
          status: 0,
          stderr: 'PEAK_RSS_BYTES=32768\n',
        }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(summary.stages[0].peakMemoryBytes).toBe(32768);
    });
  });

  describe('runMeasuredSubprocess — runner-owned peak measurement (AC-203-C)', () => {
    it('measures the peak memory of the measured process itself', () => {
      const trivial = runMeasuredSubprocess(['-e', 'process.exit(0)']);
      const hungry = runMeasuredSubprocess([
        '-e',
        'const blocks = []; for (let i = 0; i < 2000000; i += 1) blocks.push({ i });',
      ]);

      expect({
        trivialStatus: trivial.status,
        trivialIsPositive:
          Number.isFinite(trivial.peakMemoryBytes) && trivial.peakMemoryBytes > 0,
        hungryStatus: hungry.status,
        hungryIsPositive:
          Number.isFinite(hungry.peakMemoryBytes) && hungry.peakMemoryBytes > 0,
        hungryExceedsTrivial: hungry.peakMemoryBytes > trivial.peakMemoryBytes,
      }).toEqual({
        trivialStatus: 0,
        trivialIsPositive: true,
        hungryStatus: 0,
        hungryIsPositive: true,
        hungryExceedsTrivial: true,
      });
    });

    it('captures the stdout and stderr of the measured process', () => {
      const measured = runMeasuredSubprocess([
        '-e',
        'process.stdout.write("out"); process.stderr.write("err");',
      ]);

      expect({ stdout: measured.stdout, stderr: measured.stderr }).toEqual({
        stdout: 'out',
        stderr: 'err',
      });
    });
  });

  describe('runMeasuredSubprocess — REAL-subprocess lanes (AC-203-C)', () => {
    it('publishes a figure for a real child that exits with a non-zero status', () => {
      const measured = runMeasuredSubprocess(['-e', 'process.exit(3)']);

      expect({
        status: measured.status,
        publishedPositiveFigure:
          Number.isFinite(measured.peakMemoryBytes) &&
          measured.peakMemoryBytes > 0,
      }).toEqual({
        status: 3,
        publishedPositiveFigure: true,
      });
    });

    it('publishes the figure of a real child that exited with status 3 on the failed stage report', async () => {
      const summary = await updateRag({
        spawnRunner: ({ name }) =>
          name === 'build'
            ? runMeasuredSubprocess(['-e', 'process.exit(3)'])
            : runMeasuredSubprocess(['-e', 'process.exit(0)']),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      const buildStage = summary.stages.find((stage) => stage.name === 'build');
      expect({
        status: buildStage.status,
        publishedPositiveFigure:
          Number.isFinite(buildStage.peakMemoryBytes) &&
          buildStage.peakMemoryBytes > 0,
        hasMarkerKey: Object.hasOwn(buildStage, 'peakMeasurementUnavailable'),
      }).toEqual({
        status: 'failed',
        publishedPositiveFigure: true,
        hasMarkerKey: false,
      });
    });

    it('publishes NO figure for a real child killed by a fatal abort and marks the measurement unavailable', async () => {
      const summary = await updateRag({
        spawnRunner: ({ name }) =>
          name === 'build'
            ? runMeasuredSubprocess(['-e', 'process.abort()'])
            : runMeasuredSubprocess(['-e', 'process.exit(0)']),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      const buildStage = summary.stages.find((stage) => stage.name === 'build');
      expect({
        status: buildStage.status,
        hasFigureKey: Object.hasOwn(buildStage, 'peakMemoryBytes'),
        hasMarkerKey: Object.hasOwn(buildStage, 'peakMeasurementUnavailable'),
        markerValue: buildStage.peakMeasurementUnavailable,
      }).toEqual({
        status: 'failed',
        hasFigureKey: false,
        hasMarkerKey: true,
        markerValue: true,
      });
    });
  });

  describe('updateRag — the peak-measurement availability marker (AC-203-C)', () => {
    it('marks a stage that ran but published no figure as measurement-unavailable', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(
        summary.stages.map((stage) => [
          stage.name,
          stage.status,
          stage.peakMeasurementUnavailable,
        ]),
      ).toEqual([
        ['build', 'ok', true],
        ['prewarm-embed', 'ok', true],
        ['build-terms', 'ok', true],
        ['build-graph', 'ok', true],
        ['snapshot', 'ok', true],
        ['validate', 'ok', true],
      ]);
    });

    it('does not mark a stage that published a figure as measurement-unavailable', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0, peakMemoryBytes: 4096 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      expect(
        summary.stages.map((stage) => stage.peakMeasurementUnavailable),
      ).toEqual([
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
      ]);
    });

    it('does not mark a skipped stage as measurement-unavailable', async () => {
      const summary = await updateRag({
        spawnRunner: () => ({ status: 0, peakMemoryBytes: 4096 }),
        readHashFile: async () => 'same-hash',
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'same-hash',
      });

      const graphStage = summary.stages.find(
        (stage) => stage.name === 'build-graph',
      );
      expect({
        status: graphStage.status,
        hasFigureKey: Object.hasOwn(graphStage, 'peakMemoryBytes'),
        hasMarkerKey: Object.hasOwn(graphStage, 'peakMeasurementUnavailable'),
      }).toEqual({
        status: 'skipped',
        hasFigureKey: false,
        hasMarkerKey: false,
      });
    });

    it('does not mark a cancelled stage as measurement-unavailable', async () => {
      const summary = await updateRag({
        spawnRunner: ({ name }) =>
          name === 'build'
            ? { status: 1 }
            : { status: 0, peakMemoryBytes: 4096 },
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      const cancelledStages = summary.stages.filter(
        (stage) => stage.status === 'cancelled',
      );
      expect({
        cancelledCount: cancelledStages.length,
        markerKeys: cancelledStages.map((stage) =>
          Object.hasOwn(stage, 'peakMeasurementUnavailable'),
        ),
      }).toEqual({
        cancelledCount: 5,
        markerKeys: [false, false, false, false, false],
      });
    });

    it('does not mark a dry-run stage that never ran as measurement-unavailable', async () => {
      const summary = await updateRag({
        dryRun: true,
        spawnRunner: () => ({ status: 0, peakMemoryBytes: 4096 }),
        readHashFile: async () => null,
        writeHashFile: async () => {},
        computeCorpusHash: async () => 'hash-1',
      });

      const validateStage = summary.stages.find(
        (stage) => stage.name === 'validate',
      );
      expect({
        status: validateStage.status,
        hasFigureKey: Object.hasOwn(validateStage, 'peakMemoryBytes'),
        hasMarkerKey: Object.hasOwn(
          validateStage,
          'peakMeasurementUnavailable',
        ),
      }).toEqual({
        status: 'dry-run',
        hasFigureKey: false,
        hasMarkerKey: false,
      });
    });
  });

  describe('computeCorpusIdentity — family-sensitive corpus identity (AC-203-E)', () => {
    const baseRows = [
      {
        file_path: 'src/alpha.ts',
        sha256: 'aaaa',
        mtime_ms: 111,
        file_size: 22,
        doc_family: 'ts-source',
      },
      {
        file_path: 'src/beta.ts',
        sha256: 'bbbb',
        mtime_ms: 333,
        file_size: 44,
        doc_family: 'ts-source',
      },
    ];

    it('composes identity rows that carry the doc family', () => {
      expect(composeCorpusIdentityRows(baseRows)).toEqual([
        'src/alpha.ts|aaaa|111|22|ts-source',
        'src/beta.ts|bbbb|333|44|ts-source',
      ]);
    });

    it('composes identity rows sorted by file path', () => {
      expect(composeCorpusIdentityRows([...baseRows].reverse())).toEqual([
        'src/alpha.ts|aaaa|111|22|ts-source',
        'src/beta.ts|bbbb|333|44|ts-source',
      ]);
    });

    it('is identical for identical rows regardless of the row order', () => {
      expect(computeCorpusIdentity(baseRows)).toBe(
        computeCorpusIdentity([...baseRows].reverse()),
      );
    });

    it('changes when only the doc family changes', () => {
      const relabelled = baseRows.map((row) => ({
        ...row,
        doc_family: 'ts-relabelled',
      }));

      expect(computeCorpusIdentity(baseRows)).not.toBe(
        computeCorpusIdentity(relabelled),
      );
    });

    it('changes when a row content changes', () => {
      const edited = baseRows.map((row, index) =>
        index === 0 ? { ...row, sha256: 'cccc' } : row,
      );

      expect(computeCorpusIdentity(baseRows)).not.toBe(
        computeCorpusIdentity(edited),
      );
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