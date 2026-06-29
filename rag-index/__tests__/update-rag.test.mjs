/**
 * @module update-rag.test
 * @description Red tests for the `rag-index/update-rag.mjs` CLI orchestrator.
 *
 * The orchestrator does not yet exist; these tests define the observable
 * contracts it must satisfy once implemented. They are expected to fail today
 * because the module is missing (module/file not found).
 *
 * Stable stage names used by these contracts:
 *   build, prewarm-embed, build-terms, build-graph, snapshot, validate
 */

import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const scriptPath = path.join(repoRoot, 'rag-index', 'update-rag.mjs');

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
 * Spawn the update-rag CLI with the given arguments from the repo root.
 *
 * @param {string[]} args - CLI arguments.
 * @returns {Promise<{ exitCode: number | null, stdout: string, stderr: string }>}
 */
function runUpdateRag(args = []) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [scriptPath, ...args], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += String(data);
    });
    child.stderr.on('data', (data) => {
      stderr += String(data);
    });

    child.on('error', reject);
    child.on('close', (exitCode) => {
      resolve({ exitCode, stdout, stderr });
    });
  });
}

/**
 * Safely parse the JSON stages array from stdout.
 * Returns null when stdout is not valid JSON or has no stages array.
 *
 * @param {string} stdout
 * @returns {Array<{ name: string, status: string, elapsedMs: number }> | null}
 */
function parseJsonStages(stdout) {
  try {
    const parsed = JSON.parse(stdout);
    return Array.isArray(parsed.stages) ? parsed.stages : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Dry-run contract
// ---------------------------------------------------------------------------

describe('update-rag.mjs CLI orchestrator', () => {
  describe('dry-run mode', () => {
    it('exits 0 when invoked with --dry-run --json', async () => {
      const result = await runUpdateRag(['--dry-run', '--json']);

      expect(result.exitCode).toBe(0);
    });

    it('emits a JSON summary with a non-empty stages array in dry-run mode', async () => {
      const result = await runUpdateRag(['--dry-run', '--json']);
      const stages = parseJsonStages(result.stdout);

      expect(stages?.length > 0).toBe(true);
    });

    it('does not report mutation side effects in dry-run stage statuses', async () => {
      const result = await runUpdateRag(['--dry-run', '--json']);
      const stages = parseJsonStages(result.stdout) ?? [];
      const buildStage = stages.find((stage) => stage.name === 'build');

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
    it('exits 0 when invoked with --validate --json on a healthy corpus', async () => {
      const result = await runUpdateRag(['--validate', '--json']);

      expect(result.exitCode).toBe(0);
    });

    it('includes a validate stage in the JSON summary when --validate is passed', async () => {
      const result = await runUpdateRag(['--validate', '--json']);
      const stages = parseJsonStages(result.stdout) ?? [];
      const hasValidateStage = stages.some(
        (stage) => stage.name === 'validate',
      );

      expect(hasValidateStage).toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // Stage ordering contract
  // ---------------------------------------------------------------------------

  describe('stage ordering', () => {
    it('lists stages in canonical order with --dry-run --json', async () => {
      const result = await runUpdateRag(['--dry-run', '--json']);
      const stages = parseJsonStages(result.stdout) ?? [];
      const stageNames = stages.map((stage) => stage.name);

      expect(stageNames).toEqual(EXPECTED_STAGE_ORDER);
    });
  });

  // ---------------------------------------------------------------------------
  // Idempotency contract
  // ---------------------------------------------------------------------------

  describe('idempotency', () => {
    it('skips the build-graph stage on a second unchanged run', async () => {
      await runUpdateRag(['--json']);
      const secondRun = await runUpdateRag(['--json']);
      const stages = parseJsonStages(secondRun.stdout) ?? [];
      const graphStage = stages.find((stage) => stage.name === 'build-graph');

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
      const result = await runUpdateRag(['--dry-run', '--json']);
      const stages = parseJsonStages(result.stdout) ?? [];
      const wellFormedStageCount = stages.filter(
        (stage) =>
          typeof stage.name === 'string' &&
          typeof stage.status === 'string' &&
          typeof stage.elapsedMs === 'number',
      ).length;

      expect({
        nonEmpty: stages.length > 0,
        allWellFormed: wellFormedStageCount === stages.length,
      }).toEqual({
        nonEmpty: true,
        allWellFormed: true,
      });
    });
  });
});
