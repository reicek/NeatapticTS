import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const RECORD_GATE_EXCEPTION_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
  'record-gate-exception.mjs',
);
const GATE_EXCEPTION_COUNTER_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
  'gate-exception-counter.mjs',
);
const TEST_SESSION_ID = 'runtime-escalation-test-session';

describe('runtime gate escalation persistence', () => {
  let tempDir: string;
  let tempLogPath: string;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'neataptic-escalation-'));
    tempLogPath = path.join(tempDir, 'learning-log.jsonl');
    writeFileSync(tempLogPath, '', 'utf8');
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('appends a gate escalation event after three consecutive gate exceptions', () => {
    const childEnv = {
      ...process.env,
      NEATAPTIC_LEARNING_LOG_PATH: tempLogPath,
    };

    for (const gateId of ['first', 'second', 'third']) {
      spawnSync(
        process.execPath,
        [
          RECORD_GATE_EXCEPTION_PATH,
          '--json',
          `--gate-id=${gateId}`,
          '--agent=05-green-testing',
          `--session-id=${TEST_SESSION_ID}`,
          '--evidence={"reason":"runtime-test"}',
        ],
        {
          cwd: REPO_ROOT,
          encoding: 'utf8',
          timeout: 120000,
          env: childEnv,
        },
      );
    }

    const counterResult = spawnSync(
      process.execPath,
      [
        GATE_EXCEPTION_COUNTER_PATH,
        '--json',
        `--session-id=${TEST_SESSION_ID}`,
        '--derive-from-learning-log',
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 120000,
        env: childEnv,
      },
    );
    const parsedCounter = JSON.parse(counterResult.stdout) as {
      escalationTriggered: boolean;
      failureCount: number;
    };
    const learningLogText = readFileSync(tempLogPath, 'utf8');
    const hasEscalationEvent = learningLogText.includes(
      '"eventType":"gate-escalation"',
    );

    expect({
      escalationTriggered: parsedCounter.escalationTriggered,
      failureCount: parsedCounter.failureCount,
      hasEscalationEvent,
    }).toEqual({
      escalationTriggered: true,
      failureCount: 3,
      hasEscalationEvent: true,
    });
  });
});
