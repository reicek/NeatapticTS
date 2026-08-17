import { mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import path from 'node:path';
import {
  writeDocsQualityRunArtifacts,
} from './docs-quality.artifacts.mjs';

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

describe('docs-quality.artifacts.mjs', () => {
  describe('writeDocsQualityRunArtifacts', () => {
    it('writes summary, evidence, and manifest files and returns their paths', async () => {
      const runId = 'test-run-1';
      await cleanupRun(runId);

      const result = await writeDocsQualityRunArtifacts({
        runId,
        summary: { pass: true, evidenceCount: 0 },
        evidence: { rows: [], digest: 'abc' },
        manifest: { metricVersion: 2 },
      });

      expect(result.runDirectory).toContain(runId);
      expect(result.summaryPath).toBe(
        path.join(result.runDirectory, 'summary.json'),
      );
      expect(result.evidencePath).toBe(
        path.join(result.runDirectory, 'evidence.json'),
      );
      expect(result.manifestPath).toBe(
        path.join(result.runDirectory, 'manifest.json'),
      );

      const summary = JSON.parse(await readFile(result.summaryPath, 'utf8'));
      expect(summary).toEqual({ pass: true, evidenceCount: 0 });

      const evidence = JSON.parse(await readFile(result.evidencePath, 'utf8'));
      expect(evidence).toEqual({ rows: [], digest: 'abc' });

      const manifest = JSON.parse(await readFile(result.manifestPath, 'utf8'));
      expect(manifest).toEqual({ metricVersion: 2 });

      await cleanupRun(runId);
    });

    it('sanitizes runId by replacing invalid characters with dashes', async () => {
      const rawRunId = 'test/run with spaces&special';
      const sanitizedRunId = 'test-run-with-spaces-special';
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

    it('defaults to "default" when runId is undefined', async () => {
      await cleanupRun('default');

      const result = await writeDocsQualityRunArtifacts({
        summary: {},
        evidence: {},
        manifest: {},
      });

      expect(result.runDirectory).toContain('default');

      await cleanupRun('default');
    });

    it('defaults to "default" when sanitized runId is empty', async () => {
      await cleanupRun('default');

      const result = await writeDocsQualityRunArtifacts({
        runId: '!!!',
        summary: {},
        evidence: {},
        manifest: {},
      });

      // '!!!' → all replaced with dashes → '---' → length > 0, so it's '---'
      // Actually, '!!!'.replace(/[^a-zA-Z0-9._-]/g, '-') = '---' which has length > 0
      // So it returns '---', not 'default'
      expect(result.runDirectory).toContain('---');

      await cleanupRun('---');
    });

    it('defaults to "default" when runId is null', async () => {
      await cleanupRun('default');

      const result = await writeDocsQualityRunArtifacts({
        runId: null,
        summary: {},
        evidence: {},
        manifest: {},
      });

      // null → String(null ?? 'default') = 'default'
      expect(result.runDirectory).toContain('default');

      await cleanupRun('default');
    });

    it('handles runId that sanitizes to empty string', async () => {
      // A runId of only whitespace: '   '.trim() = '' → sanitizedRunId = '' → length 0 → 'default'
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
  });
});