import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const LEARNING_LOG_PATH = path.join(
  REPO_ROOT,
  '.github',
  'ai-learning',
  'learning-log.jsonl',
);
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
  let originalLearningLog = '';

  beforeEach(async () => {
    try {
      originalLearningLog = await readFile(LEARNING_LOG_PATH, 'utf8');
    } catch {
      originalLearningLog = '';
    }
    await mkdir(path.dirname(LEARNING_LOG_PATH), { recursive: true });
    await writeFile(LEARNING_LOG_PATH, '', 'utf8');
  });

  afterEach(async () => {
    await writeFile(LEARNING_LOG_PATH, originalLearningLog, 'utf8');
  });

  it('appends a gate escalation event after three consecutive gate exceptions', async () => {
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
      },
    );
    const parsedCounter = JSON.parse(counterResult.stdout) as {
      escalationTriggered: boolean;
      failureCount: number;
    };
    const learningLogText = await readFile(LEARNING_LOG_PATH, 'utf8');
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
