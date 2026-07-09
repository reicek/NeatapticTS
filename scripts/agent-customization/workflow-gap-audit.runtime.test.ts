import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const WORKFLOW_GAP_AUDIT_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'workflow-gap-audit.mjs',
);

describe('workflow-gap-audit runtime enforcement evidence', () => {
  let tempDir: string;
  let tempLogPath: string;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'neataptic-workflow-audit-'));
    tempLogPath = path.join(tempDir, 'learning-log.jsonl');
    writeFileSync(
      tempLogPath,
      [
        JSON.stringify({
          eventType: 'runtime-action-prepass',
          sessionId: 'runtime-audit-test-session',
          actionId: 'action-pre-only',
        }),
        JSON.stringify({
          eventType: 'runtime-action-postpass',
          sessionId: 'runtime-audit-test-session',
          actionId: 'action-post-only',
        }),
        JSON.stringify({
          eventType: 'runtime-proof-mismatch',
          sessionId: 'runtime-audit-test-session',
          reason: 'Missing runtime enforcement context carrier.',
        }),
      ].join('\n') + '\n',
      'utf8',
    );
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('reports missing hook pairs and runtime proof mismatches in JSON output', () => {
    const auditResult = spawnSync(
      process.execPath,
      [WORKFLOW_GAP_AUDIT_PATH, '--json', '--window=30'],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 120000,
        env: {
          ...process.env,
          NEATAPTIC_LEARNING_LOG_PATH: tempLogPath,
        },
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
