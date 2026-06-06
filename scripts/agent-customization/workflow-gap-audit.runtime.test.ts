import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const LEARNING_LOG_PATH = path.join(REPO_ROOT, '.github', 'ai-learning', 'learning-log.jsonl');
const WORKFLOW_GAP_AUDIT_PATH = path.join(REPO_ROOT, 'scripts', 'agent-customization', 'workflow-gap-audit.mjs');

describe('workflow-gap-audit runtime enforcement evidence', () => {
  let originalLearningLog = '';

  beforeEach(async () => {
    try {
      originalLearningLog = await readFile(LEARNING_LOG_PATH, 'utf8');
    } catch {
      originalLearningLog = '';
    }
    await mkdir(path.dirname(LEARNING_LOG_PATH), { recursive: true });
    await writeFile(
      LEARNING_LOG_PATH,
      [
        JSON.stringify({
          timestamp: '2026-06-03T00:00:00.000Z',
          eventType: 'runtime-action-prepass',
          sessionId: 'runtime-audit-test-session',
          actionId: 'action-pre-only',
        }),
        JSON.stringify({
          timestamp: '2026-06-03T00:01:00.000Z',
          eventType: 'runtime-action-postpass',
          sessionId: 'runtime-audit-test-session',
          actionId: 'action-post-only',
        }),
        JSON.stringify({
          timestamp: '2026-06-03T00:02:00.000Z',
          eventType: 'runtime-proof-mismatch',
          sessionId: 'runtime-audit-test-session',
          reason: 'Missing runtime enforcement context carrier.',
        }),
      ].join('\n') + '\n',
      'utf8',
    );
  });

  afterEach(async () => {
    await writeFile(LEARNING_LOG_PATH, originalLearningLog, 'utf8');
  });

  it('reports missing hook pairs and runtime proof mismatches in JSON output', () => {
    const auditResult = spawnSync(
      process.execPath,
      [WORKFLOW_GAP_AUDIT_PATH, '--json', '--window=30'],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 120000,
      },
    );
    const parsedReport = JSON.parse(auditResult.stdout) as {
      runtimeEnforcementEvidence: {
        preActionPasses: number;
        postActionPasses: number;
        proofMismatches: number;
        blockedActions: number;
        missingPostActionPairs: string[];
        postWithoutPrePairs: string[];
      };
    };

    expect(parsedReport.runtimeEnforcementEvidence).toEqual({
      preActionPasses: 1,
      postActionPasses: 1,
      proofMismatches: 1,
      blockedActions: 1,
      missingPostActionPairs: ['action-pre-only'],
      postWithoutPrePairs: ['action-post-only'],
    });
  });
});
