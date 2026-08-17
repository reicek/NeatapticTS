import { rm, readFile } from 'node:fs/promises';
import path from 'node:path';
import { writeDocsQualityRunArtifacts } from '../../../rag-index/docs-quality/docs-quality.artifacts.mjs';

const ARTIFACTS_BASE = path.resolve(
  process.cwd(),
  'artifacts',
  'docs-quality',
  'runs',
);

async function cleanupRun(runId) {
  const runDir = path.join(ARTIFACTS_BASE, runId);
  await rm(runDir, { recursive: true, force: true });
}

describe('docs-quality.artifacts.mjs coverage', () => {
  describe('writeDocsQualityRunArtifacts — normalizeRunId branches', () => {
    it('uses provided runId (non-nullish, non-empty after sanitize)', async () => {
      const runId = 'cov-test-run-1';
      await cleanupRun(runId);

      const result = await writeDocsQualityRunArtifacts({
        runId,
        summary: { pass: true },
        evidence: { rows: [] },
        manifest: { version: 2 },
      });

      expect(result.runDirectory).toContain(runId);
      expect(result.summaryPath).toBe(path.join(result.runDirectory, 'summary.json'));
      expect(result.evidencePath).toBe(path.join(result.runDirectory, 'evidence.json'));
      expect(result.manifestPath).toBe(path.join(result.runDirectory, 'manifest.json'));

      await cleanupRun(runId);
    });

    it('sanitizes runId with special characters', async () => {
      const rawRunId = 'cov/test/run#2';
      const sanitizedRunId = 'cov-test-run-2';
      await cleanupRun(sanitizedRunId);

      const result = await writeDocsQualityRunArtifacts({
        runId: rawRunId,
        summary: {},
        evidence: {},
        manifest: {},
      });

      expect(result.runDirectory).toContain(sanitizedRunId);

      await cleanupRun(sanitizedRunId);
    });

    it('defaults to "default" when runId is undefined (nullish branch)', async () => {
      await cleanupRun('default');

      const result = await writeDocsQualityRunArtifacts({
        summary: {},
        evidence: {},
        manifest: {},
      });

      expect(result.runDirectory).toContain('default');

      await cleanupRun('default');
    });

    it('defaults to "default" when runId is null (nullish branch)', async () => {
      await cleanupRun('default');

      const result = await writeDocsQualityRunArtifacts({
        runId: null,
        summary: {},
        evidence: {},
        manifest: {},
      });

      expect(result.runDirectory).toContain('default');

      await cleanupRun('default');
    });

    it('defaults to "default" when runId sanitizes to empty string (ternary false branch)', async () => {
      await cleanupRun('default');

      const result = await writeDocsQualityRunArtifacts({
        runId: '   ',
        summary: {},
        evidence: {},
        manifest: {},
      });

      expect(result.runDirectory).toContain('default');

      await cleanupRun('default');
    });

    it('writes files with correct content', async () => {
      const runId = 'cov-test-content';
      await cleanupRun(runId);

      const summary = { pass: false, evidenceCount: 3 };
      const evidence = { rows: [{ file: 'a.ts' }], digest: 'abc' };
      const manifest = { metricVersion: 2, pass: false };

      const result = await writeDocsQualityRunArtifacts({
        runId,
        summary,
        evidence,
        manifest,
      });

      const writtenSummary = JSON.parse(await readFile(result.summaryPath, 'utf8'));
      const writtenEvidence = JSON.parse(await readFile(result.evidencePath, 'utf8'));
      const writtenManifest = JSON.parse(await readFile(result.manifestPath, 'utf8'));

      expect(writtenSummary).toEqual(summary);
      expect(writtenEvidence).toEqual(evidence);
      expect(writtenManifest).toEqual(manifest);

      await cleanupRun(runId);
    });
  });
});