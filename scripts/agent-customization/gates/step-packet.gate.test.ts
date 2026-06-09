import { spawnSync } from 'node:child_process';
import { unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * Step-packet gate tests for goal-based routing validation.
 *
 * After removing backward compatibility for the 'agent' field:
 * - 'goal' is required (not optional, not replaceable by 'agent')
 * - 'agent' is rejected as a deprecated field violation
 * - 'tdd_sequence' optional field validation
 * - 'goal' value validation against allowed set
 * - 'fixHint' mentions 'goal' when goal is missing
 *
 * Plan: plans/Step_Packet_Goal_Redesign.plans.md — Phase 2 Step 05
 */

interface StepPacketViolation {
  stepId: string;
  missingField?: string;
  missingSection?: string;
  invalidField?: string;
  invalidValue?: string;
  allowedValues?: string[];
  deprecatedField?: string;
  message?: string;
  [key: string]: unknown;
}

interface StepPacketGateResult {
  pass: boolean;
  evidence: {
    stepsChecked: string[];
    violations: StepPacketViolation[];
    plansScanned: number;
    [key: string]: unknown;
  };
  fixHint: string;
  owner: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const GATE_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
  'step-packet.gate.mjs',
);
const PLANS_DIR = path.join(REPO_ROOT, 'plans');

const VALID_GOAL_VALUES = [
  'planning',
  'researching',
  'red-testing',
  'implementing',
  'green-testing',
  'documenting',
  'logging',
  'helping',
] as const;

/**
 * Creates a temporary WIP plan file in plans/ with the given YAML fields.
 * Returns the file name (relative to plans/) for cleanup and violation filtering.
 *
 * The generated file includes the required "Stop conditions" and
 * "Required validation" prose sections so the gate does not flag
 * missing sections — only the YAML fields under test will cause violations.
 */
async function createTempPlanFile(
  yamlFields: Record<string, string>,
  testId: string,
): Promise<string> {
  const fileName = `_step-packet-goal-test-${testId}.plans.md`;
  const filePath = path.join(PLANS_DIR, fileName);

  const yamlLines = Object.entries(yamlFields)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n');

  const content = [
    `# Temp Test Plan for step-packet goal validation (${testId})`,
    '',
    '**Status:** [WIP]',
    '',
    '## Test Step',
    '',
    '```yaml',
    yamlLines,
    '```',
    '',
    '**Stop conditions:** All gate validations pass.',
    '',
    '**Required validation:** Gate passes without violations for this step.',
  ].join('\n');

  await writeFile(filePath, content, 'utf8');
  return fileName;
}

/** Removes the temporary plan file created by {@link createTempPlanFile}. */
async function cleanupTempPlanFile(fileName: string): Promise<void> {
  const filePath = path.join(PLANS_DIR, fileName);
  try {
    await unlink(filePath);
  } catch {
    // Ignore cleanup errors — temp files are non-essential artifacts
  }
}

/** Spawns the step-packet gate with --json and parses the structured result. */
function runGate(): StepPacketGateResult {
  const result = spawnSync(process.execPath, [GATE_PATH, '--json'], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 30000,
  });

  const stdout = result.stdout ?? '';
  try {
    return JSON.parse(stdout) as StepPacketGateResult;
  } catch {
    throw new Error(
      `Failed to parse gate output:\n${stdout.substring(0, 500)}`,
    );
  }
}

/** Filters gate violations to those originating from the given plan file. */
function getViolationsForFile(
  result: StepPacketGateResult,
  fileName: string,
): StepPacketViolation[] {
  return result.evidence.violations.filter((v) =>
    v.stepId.startsWith(`plans/${fileName}:`),
  );
}

/**
 * Base YAML fields that satisfy the current gate's REQUIRED_YAML_FIELDS.
 * Used in tests that need the step to pass current validation so that
 * only NEW validation rules cause failures.
 */
const BASE_YAML_FIELDS: Record<string, string> = {
  phase: '1',
  step: '2',
  goal: "'implementing'",
  status: '"[WIP]"',
  next_step: "'Step 03 — Update routing table'",
};

describe('step-packet.gate.mjs', () => {
  describe('goal field validation', () => {
    // -----------------------------------------------------------------------
    // (a) Neither goal nor agent present → fixHint must mention goal
    // Current: fixHint mentions "agent" only → RED
    // -----------------------------------------------------------------------
    it('mentions goal in fixHint when neither goal nor agent is present', async () => {
      const fileName = await createTempPlanFile(
        {
          phase: '1',
          step: '2',
          status: '"[WIP]"',
          next_step: "'Step 03 — Update routing table'",
        },
        'no-goal-no-agent',
      );

      try {
        const result = runGate();
        // The fixHint should mention "goal" as the required routing field
        // Currently it only mentions "agent"
        expect(result.fixHint).toContain('goal');
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    // -----------------------------------------------------------------------
    // (b) Agent without goal → violation for deprecated agent and missing goal
    // Previously: deprecation warning → now: hard violation
    // -----------------------------------------------------------------------
    it('rejects agent field without goal as violation', async () => {
      const fileName = await createTempPlanFile(
        {
          phase: '1',
          step: '2',
          agent: "'04-implementing'",
          status: '"[WIP]"',
          next_step: "'Step 03 — Update routing table'",
        },
        'agent-no-goal',
      );

      try {
        const result = runGate();
        const fileViolations = getViolationsForFile(result, fileName);
        // agent without goal is now a violation (missing goal + deprecated agent)
        expect(fileViolations.length).toBeGreaterThanOrEqual(1);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    // -----------------------------------------------------------------------
    // (c) Both goal and agent present → violation for deprecated agent field
    // Previously: deprecation warning → now: hard violation
    // -----------------------------------------------------------------------
    it('rejects deprecated agent field when both goal and agent are present', async () => {
      const fileName = await createTempPlanFile(
        {
          phase: '1',
          step: '2',
          goal: "'implementing'",
          agent: "'04-implementing'",
          status: '"[WIP]"',
          next_step: "'Step 03 — Update routing table'",
        },
        'both-goal-and-agent',
      );

      try {
        const result = runGate();
        const fileViolations = getViolationsForFile(result, fileName);
        // agent field is now a violation even when goal is present
        expect(fileViolations.some((v) => v.deprecatedField === 'agent')).toBe(
          true,
        );
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    // -----------------------------------------------------------------------
    // (d) Goal without agent → should satisfy routing requirement
    // Current: agent is required → violation for missing agent → RED
    // -----------------------------------------------------------------------
    it('accepts goal as primary routing field without agent', async () => {
      const fileName = await createTempPlanFile(
        {
          phase: '1',
          step: '2',
          goal: "'implementing'",
          status: '"[WIP]"',
          next_step: "'Step 03 — Update routing table'",
        },
        'goal-without-agent',
      );

      try {
        const result = runGate();
        const fileViolations = getViolationsForFile(result, fileName);
        // Should have no violations — goal satisfies the routing field requirement
        // Currently: violation for missing "agent" field
        expect(fileViolations.length).toBe(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    // -----------------------------------------------------------------------
    // (e) Invalid goal value → should be rejected
    // Current: goal values not validated → no violation → RED
    // -----------------------------------------------------------------------
    it('rejects invalid goal value', async () => {
      const fileName = await createTempPlanFile(
        {
          ...BASE_YAML_FIELDS,
          goal: "'invalid-goal'",
        },
        'invalid-goal',
      );

      try {
        const result = runGate();
        const fileViolations = getViolationsForFile(result, fileName);
        // Should have a violation for invalid goal value
        // Currently: passes because goal values are not validated
        expect(fileViolations.length).toBeGreaterThan(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    // -----------------------------------------------------------------------
    // (f) Invalid tdd_sequence value → should be rejected
    // Current: tdd_sequence not validated → no violation → RED
    // -----------------------------------------------------------------------
    it('rejects invalid tdd_sequence value', async () => {
      const fileName = await createTempPlanFile(
        {
          ...BASE_YAML_FIELDS,
          goal: "'implementing'",
          tdd_sequence: "'invalid-sequence'",
        },
        'invalid-tdd-sequence',
      );

      try {
        const result = runGate();
        const fileViolations = getViolationsForFile(result, fileName);
        // Should have a violation for invalid tdd_sequence value
        // Currently: passes because tdd_sequence is not validated
        expect(fileViolations.length).toBeGreaterThan(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    // -----------------------------------------------------------------------
    // (g) Each valid goal value should be accepted without agent
    // Current: agent is required → violation for missing agent → RED
    // -----------------------------------------------------------------------
    it.each(VALID_GOAL_VALUES)(
      'accepts valid goal value "%s" without agent',
      async (goalValue: string) => {
        const fileName = await createTempPlanFile(
          {
            phase: '1',
            step: '2',
            goal: `'${goalValue}'`,
            status: '"[WIP]"',
            next_step: "'Next step'",
          },
          `valid-goal-${goalValue}`,
        );

        try {
          const result = runGate();
          const fileViolations = getViolationsForFile(result, fileName);
          // Should have no violations — valid goal satisfies routing requirement
          // Currently: violation for missing "agent" field
          expect(fileViolations.length).toBe(0);
        } finally {
          await cleanupTempPlanFile(fileName);
        }
      },
    );

    // -----------------------------------------------------------------------
    // (h) Invalid goal value that resembles an agent name → should be rejected
    // Ensures that agent names like "04-implementing" are not valid goal values
    // Current: goal values not validated → no violation → RED
    // -----------------------------------------------------------------------
    it('rejects agent-style goal value like "04-implementing"', async () => {
      const fileName = await createTempPlanFile(
        {
          ...BASE_YAML_FIELDS,
          goal: "'04-implementing'",
        },
        'agent-style-goal',
      );

      try {
        const result = runGate();
        const fileViolations = getViolationsForFile(result, fileName);
        // Should have a violation because "04-implementing" is not a valid goal
        // Valid goals are: planning, researching, red-testing, implementing,
        //   green-testing, documenting, logging, helping
        // Currently: passes because goal values are not validated
        expect(fileViolations.length).toBeGreaterThan(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });
  });
});
