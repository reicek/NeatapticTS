import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * Persist canonical docs-quality run artifacts for one run id.
 *
 * @param {{ evidence: Record<string, unknown>, manifest: Record<string, unknown>, runId: string, summary: Record<string, unknown> }} payload - Artifact payload.
 * @returns {Promise<{ evidencePath: string, manifestPath: string, runDirectory: string, summaryPath: string }>} Absolute artifact paths.
 */
export async function writeDocsQualityRunArtifacts(payload) {
  const runId = normalizeRunId(payload.runId);
  const runDirectory = path.resolve(
    process.cwd(),
    'artifacts',
    'docs-quality',
    'runs',
    runId,
  );
  const summaryPath = path.join(runDirectory, 'summary.json');
  const evidencePath = path.join(runDirectory, 'evidence.json');
  const manifestPath = path.join(runDirectory, 'manifest.json');

  await mkdir(runDirectory, { recursive: true });
  await writeFile(
    summaryPath,
    `${JSON.stringify(payload.summary, null, 2)}\n`,
    'utf8',
  );
  await writeFile(
    evidencePath,
    `${JSON.stringify(payload.evidence, null, 2)}\n`,
    'utf8',
  );
  await writeFile(
    manifestPath,
    `${JSON.stringify(payload.manifest, null, 2)}\n`,
    'utf8',
  );

  return {
    evidencePath,
    manifestPath,
    runDirectory,
    summaryPath,
  };
}

function normalizeRunId(runId) {
  const rawRunId = String(runId ?? 'default').trim();
  const sanitizedRunId = rawRunId.replace(/[^a-zA-Z0-9._-]/g, '-');
  return sanitizedRunId.length > 0 ? sanitizedRunId : 'default';
}
