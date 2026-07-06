/**
 * @module update-rag.test
 * @description Unit tests for the `rag-index/update-rag.mjs` pipeline
 * orchestrator.
 *
 * These tests exercise the exported `updateRag` orchestrator with injected
 * stage runners and hash storage so the contracts are fast, deterministic, and
 * independent of the current on-disk corpus state.
 *
 * Stable stage names used by these contracts:
 *   build, prewarm-embed, build-terms, build-graph, snapshot, validate
 */

import { updateRag } from '../update-rag.mjs';

/** Canonical stage order, including the optional validate stage last. */
const EXPECTED_STAGE_ORDER = [
  'build',
  'prewarm-embed',
  'build-terms',
  'build-graph',
  'snapshot',
  'validate',
];

/**
 * Create an injectable stage runner that records every invocation and
 * succeeds by default.
 *
 * @param {Record<string, number>} [exitByStage] - Optional exit status overrides keyed by stage name.
 * @returns {{ runner: function, calls: Array<object> }} Runner and its call log.
 */
function createMockRunner(exitByStage = {}) {
  const calls = [];
  const runner = ({ name, script, args }) => {
    calls.push({ name, script, args });
    return { status: exitByStage[name] ?? 0 };
  };
  return { runner, calls };
}

/**
 * Create an injectable corpus-hash store for idempotency tests.
 *
 * @param {string | null} [initialHash] - Hash returned by the first read.
 * @returns {{ readHashFile: function, writeHashFile: function, computeCorpusHash: function, storedHash: string | null }}
 */
function createMockHashStore(initialHash = null) {
  let storedHash = initialHash;
  return {
    readHashFile: async () => storedHash,
    writeHashFile: async (hash) => {
      storedHash = hash;
    },
    computeCorpusHash: async () => 'stable-corpus-hash',
    get storedHash() {
      return storedHash;
    },
  };
}

// ---------------------------------------------------------------------------
// Dry-run contract
// ---------------------------------------------------------------------------

describe('update-rag.mjs CLI orchestrator', () => {
  describe('dry-run mode', () => {
    it('returns a passing summary when invoked with dryRun and json options', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        dryRun: true,
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });

      expect(summary.ok).toBe(true);
    });

    it('emits a non-empty stages array in dry-run mode', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        dryRun: true,
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });

      expect(summary.stages.length > 0).toBe(true);
    });

    it('does not report mutation side effects in dry-run stage statuses', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        dryRun: true,
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });
      const buildStage = summary.stages.find((stage) => stage.name === 'build');

      expect({
        buildStageExists: Boolean(buildStage),
        buildStatusIsNonMutating: buildStage
          ? !/^(mutated|written)$/.test(buildStage.status)
          : false,
      }).toEqual({
        buildStageExists: true,
        buildStatusIsNonMutating: true,
      });
    });
  });

  // ---------------------------------------------------------------------------
  // --validate contract
  // ---------------------------------------------------------------------------

  describe('--validate mode', () => {
    it('returns a passing summary when validate is requested on a healthy corpus', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        validate: true,
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });

      expect(summary.ok).toBe(true);
    });

    it('includes a validate stage in the summary when validate is requested', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        validate: true,
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });
      const hasValidateStage = summary.stages.some(
        (stage) => stage.name === 'validate',
      );

      expect(hasValidateStage).toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // Stage ordering contract
  // ---------------------------------------------------------------------------

  describe('stage ordering', () => {
    it('lists stages in canonical order in dry-run mode', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        dryRun: true,
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });
      const stageNames = summary.stages.map((stage) => stage.name);

      expect(stageNames).toEqual(EXPECTED_STAGE_ORDER);
    });
  });

  // ---------------------------------------------------------------------------
  // Idempotency contract
  // ---------------------------------------------------------------------------

  describe('idempotency', () => {
    it('skips the build-graph stage on a second unchanged run', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      await updateRag({
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });
      const secondSummary = await updateRag({
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });
      const graphStage = secondSummary.stages.find(
        (stage) => stage.name === 'build-graph',
      );

      expect({
        graphStageExists: Boolean(graphStage),
        graphStatusIsSkipped: graphStage?.status === 'skipped',
      }).toEqual({
        graphStageExists: true,
        graphStatusIsSkipped: true,
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Per-stage status reporting contract
  // ---------------------------------------------------------------------------

  describe('per-stage status reporting', () => {
    it('reports every stage with name, status, and elapsedMs in dry-run output', async () => {
      const { runner } = createMockRunner();
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        dryRun: true,
        json: true,
        spawnRunner: runner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });
      const wellFormedStageCount = summary.stages.filter(
        (stage) =>
          typeof stage.name === 'string' &&
          typeof stage.status === 'string' &&
          typeof stage.elapsedMs === 'number',
      ).length;

      expect({
        nonEmpty: summary.stages.length > 0,
        allWellFormed: wellFormedStageCount === summary.stages.length,
      }).toEqual({
        nonEmpty: true,
        allWellFormed: true,
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Failure log capture contract
  // ---------------------------------------------------------------------------

  describe('failure log capture', () => {
    it('surfaces stdout and stderr in the failed stage report', async () => {
      const { runner, calls } = createMockRunner({
        validate: 1,
      });
      runner.mockOutput = { stdout: 'validate stdout', stderr: 'validate stderr' };
      const wrappedRunner = (stage) => {
        const result = runner(stage);
        return {
          status: result.status,
          stdout: runner.mockOutput.stdout,
          stderr: runner.mockOutput.stderr,
        };
      };
      const hashStore = createMockHashStore();

      const summary = await updateRag({
        json: true,
        spawnRunner: wrappedRunner,
        readHashFile: hashStore.readHashFile,
        writeHashFile: hashStore.writeHashFile,
        computeCorpusHash: hashStore.computeCorpusHash,
      });
      const validateStage = summary.stages.find(
        (stage) => stage.name === 'validate',
      );

      expect(summary.ok).toBe(false);
      expect(validateStage).toEqual(
        expect.objectContaining({
          status: 'failed',
          stdout: 'validate stdout',
          stderr: 'validate stderr',
        }),
      );
      expect(summary.error).toContain('validate stdout');
      expect(summary.error).toContain('validate stderr');
    });
  });
});
