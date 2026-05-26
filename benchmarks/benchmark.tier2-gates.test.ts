/**
 * Red-phase contract tests for Tier-2 gate scripts, `record_gate_exception` helper,
 * and three-exceptions-trigger-00-helping escalation logic.
 *
 * These tests are intentionally red until Phase 3 Step 04 implements:
 * - `scripts/agent-customization/gates/{output-resolution-evidence, planning-output-contract,
 *    research-findings-evidence, red-test-confirmation, implementation-artifact-paths,
 *    green-validation-evidence, docs-artifact-reference, log-completion-marker}.gate.mjs`
 * - `scripts/agent-customization/gates/record-gate-exception.mjs`
 * - `scripts/agent-customization/gates/gate-exception-counter.mjs`
 *
 * Tier-2 gate contract (from Agentic_Flows_and_Gates_Upgrade.plans.md Phase 3 Step 01):
 *   pass: boolean, evidence: object, fixHint: string, owner: string
 *
 * record_gate_exception contract:
 *   timestamp: ISO-8601 string, gate-id: string, exception-evidence: object,
 *   agent: string, session-id: string
 */
import { spawnSync } from 'child_process';
import path from 'path';

const ROOT = path.resolve(__dirname, '..');
const GATES_DIR = path.join(ROOT, 'scripts', 'agent-customization', 'gates');

/**
 * Run a gate script with --json flag and return parsed stdout.
 * Throws if the script exits non-zero or stdout is not valid JSON — intentionally red
 * when the script does not exist.
 */
function runGateJson(scriptName: string): Record<string, unknown> {
  const scriptPath = path.join(GATES_DIR, scriptName);
  const result = spawnSync('node', [scriptPath, '--json'], {
    encoding: 'utf8',
    timeout: 10_000,
  });
  return JSON.parse(result.stdout) as Record<string, unknown>;
}

/**
 * Run a helper script with extra args and return parsed stdout.
 * Throws if stdout is not valid JSON — intentionally red when the script does not exist.
 */
function runHelperJson(
  scriptName: string,
  args: string[],
): Record<string, unknown> {
  const scriptPath = path.join(GATES_DIR, scriptName);
  const result = spawnSync('node', [scriptPath, '--json', ...args], {
    encoding: 'utf8',
    timeout: 10_000,
  });
  return JSON.parse(result.stdout) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Tier-2 gate output contracts — one gate per numbered agent (00–07)
// ---------------------------------------------------------------------------

describe('Tier-2 gate scripts — output contract', () => {
  // 00-helping: output-resolution-evidence gate
  describe('00-helping: output-resolution-evidence gate', () => {
    it('returns a pass boolean', () => {
      expect(
        typeof runGateJson('output-resolution-evidence.gate.mjs')['pass'],
      ).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('output-resolution-evidence.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('output-resolution-evidence.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('output-resolution-evidence.gate.mjs')['owner'],
      ).toBe('string');
    });
  });

  // 01-planning: planning-output-contract gate
  describe('01-planning: planning-output-contract gate', () => {
    it('returns a pass boolean', () => {
      expect(
        typeof runGateJson('planning-output-contract.gate.mjs')['pass'],
      ).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('planning-output-contract.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('planning-output-contract.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('planning-output-contract.gate.mjs')['owner'],
      ).toBe('string');
    });
  });

  // 02-researching: research-findings-evidence gate
  describe('02-researching: research-findings-evidence gate', () => {
    it('returns a pass boolean', () => {
      expect(
        typeof runGateJson('research-findings-evidence.gate.mjs')['pass'],
      ).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('research-findings-evidence.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('research-findings-evidence.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('research-findings-evidence.gate.mjs')['owner'],
      ).toBe('string');
    });
  });

  // 03-red-testing: red-test-confirmation gate
  describe('03-red-testing: red-test-confirmation gate', () => {
    it('returns a pass boolean', () => {
      expect(typeof runGateJson('red-test-confirmation.gate.mjs')['pass']).toBe(
        'boolean',
      );
    });

    it('returns an evidence object', () => {
      const output = runGateJson('red-test-confirmation.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('red-test-confirmation.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('red-test-confirmation.gate.mjs')['owner'],
      ).toBe('string');
    });
  });

  // 04-implementing: implementation-artifact-paths gate
  describe('04-implementing: implementation-artifact-paths gate', () => {
    it('returns a pass boolean', () => {
      expect(
        typeof runGateJson('implementation-artifact-paths.gate.mjs')['pass'],
      ).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('implementation-artifact-paths.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('implementation-artifact-paths.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('implementation-artifact-paths.gate.mjs')['owner'],
      ).toBe('string');
    });
  });

  // 05-green-testing: green-validation-evidence gate
  describe('05-green-testing: green-validation-evidence gate', () => {
    it('returns a pass boolean', () => {
      expect(
        typeof runGateJson('green-validation-evidence.gate.mjs')['pass'],
      ).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('green-validation-evidence.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('green-validation-evidence.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('green-validation-evidence.gate.mjs')['owner'],
      ).toBe('string');
    });
  });

  // 06-documenting: docs-artifact-reference gate
  describe('06-documenting: docs-artifact-reference gate', () => {
    it('returns a pass boolean', () => {
      expect(
        typeof runGateJson('docs-artifact-reference.gate.mjs')['pass'],
      ).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('docs-artifact-reference.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('docs-artifact-reference.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('docs-artifact-reference.gate.mjs')['owner'],
      ).toBe('string');
    });
  });

  // 07-logging: log-completion-marker gate
  describe('07-logging: log-completion-marker gate', () => {
    it('returns a pass boolean', () => {
      expect(typeof runGateJson('log-completion-marker.gate.mjs')['pass']).toBe(
        'boolean',
      );
    });

    it('returns an evidence object', () => {
      const output = runGateJson('log-completion-marker.gate.mjs');
      expect(
        output['evidence'] !== null && typeof output['evidence'] === 'object',
      ).toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(
        typeof runGateJson('log-completion-marker.gate.mjs')['fixHint'],
      ).toBe('string');
    });

    it('returns an owner string', () => {
      expect(
        typeof runGateJson('log-completion-marker.gate.mjs')['owner'],
      ).toBe('string');
    });
  });
});

// ---------------------------------------------------------------------------
// record_gate_exception helper — output field contract
// ---------------------------------------------------------------------------

describe('record_gate_exception helper — output contract', () => {
  const EXCEPTION_ARGS = [
    '--gate-id=plan-sync',
    '--agent=01-planning',
    '--session-id=test-session-001',
    '--evidence={"reason":"test-red-phase"}',
  ];

  it('returns a timestamp ISO-8601 string', () => {
    const output = runHelperJson('record-gate-exception.mjs', EXCEPTION_ARGS);
    expect(typeof output['timestamp']).toBe('string');
  });

  it('returns a gate-id string', () => {
    const output = runHelperJson('record-gate-exception.mjs', EXCEPTION_ARGS);
    expect(typeof output['gate-id']).toBe('string');
  });

  it('returns an exception-evidence object', () => {
    const output = runHelperJson('record-gate-exception.mjs', EXCEPTION_ARGS);
    expect(
      output['exception-evidence'] !== null &&
        typeof output['exception-evidence'] === 'object',
    ).toBe(true);
  });

  it('returns an agent string', () => {
    const output = runHelperJson('record-gate-exception.mjs', EXCEPTION_ARGS);
    expect(typeof output['agent']).toBe('string');
  });

  it('returns a session-id string', () => {
    const output = runHelperJson('record-gate-exception.mjs', EXCEPTION_ARGS);
    expect(typeof output['session-id']).toBe('string');
  });
});

// ---------------------------------------------------------------------------
// gate-exception-counter — three-exceptions escalation rule
// ---------------------------------------------------------------------------

describe('gate-exception-counter — three-exceptions escalation rule', () => {
  const SESSION_ID = 'test-escalation-session-001';

  it('does not trigger escalation on the first gate failure', () => {
    const output = runHelperJson('gate-exception-counter.mjs', [
      '--gate-id=plan-sync',
      '--agent=01-planning',
      `--session-id=${SESSION_ID}`,
      '--failure-count=1',
    ]);
    expect(output['escalationTriggered']).toBe(false);
  });

  it('does not trigger escalation on the second consecutive gate failure', () => {
    const output = runHelperJson('gate-exception-counter.mjs', [
      '--gate-id=plan-sync',
      '--agent=01-planning',
      `--session-id=${SESSION_ID}`,
      '--failure-count=2',
    ]);
    expect(output['escalationTriggered']).toBe(false);
  });

  it('triggers 00-helping escalation on the third consecutive gate failure', () => {
    const output = runHelperJson('gate-exception-counter.mjs', [
      '--gate-id=plan-sync',
      '--agent=01-planning',
      `--session-id=${SESSION_ID}`,
      '--failure-count=3',
    ]);
    expect(output['escalationTriggered']).toBe(true);
  });

  it('names 00-helping as the suggested escalation target', () => {
    const output = runHelperJson('gate-exception-counter.mjs', [
      '--gate-id=plan-sync',
      '--agent=01-planning',
      `--session-id=${SESSION_ID}`,
      '--failure-count=3',
    ]);
    expect(output['suggestedAgent']).toBe('00-helping');
  });

  it('resets escalation trigger to false when failure-count drops to zero', () => {
    const output = runHelperJson('gate-exception-counter.mjs', [
      '--gate-id=plan-sync',
      '--agent=01-planning',
      `--session-id=${SESSION_ID}`,
      '--failure-count=0',
    ]);
    expect(output['escalationTriggered']).toBe(false);
  });
});
