import { spawnSync } from 'node:child_process';
import { readFile, readdir, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * Step-packet gate tests for the mandatory plan-phase-step workflow format.
 *
 * Covers:
 * - required YAML fields for WIP step and phase packets
 * - goal-based routing and rejection of agent:/agent_file:
 * - slice expansion validation (auto_expand, tdd_sequence, slice schema)
 */

interface StepPacketViolation {
  blockId?: string;
  sliceId?: string;
  missingField?: string;
  invalidField?: string;
  invalidValue?: unknown;
  expected?: unknown;
  allowedValues?: string[];
  deprecatedField?: string;
  message?: string;
  legacy?: boolean;
  parseError?: string;
  level?: string;
}

interface PreExecuteHookEvidence {
  blockId: string;
  tool: string;
  args: Record<string, unknown>;
}

interface StepPacketGateResult {
  pass: boolean;
  evidence: {
    blocksChecked: string[];
    violations: StepPacketViolation[];
    plansScanned: number;
    preExecuteHooks?: PreExecuteHookEvidence[];
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
const TEMP_FILE_PREFIX = '_step-packet-gate-test-';

/** Minimal YAML serializer used for test fixtures. */
function quoteScalar(value: unknown): string {
  if (value === true) return 'true';
  if (value === false) return 'false';
  if (value === null) return 'null';
  const str = String(value);
  if (/^-?\d+(\.\d+)?$/u.test(str)) return str;
  if (/^[A-Za-z0-9_./:@#\-]+$/u.test(str) && !str.includes("'")) return str;
  return `'${str.replace(/'/gu, "''")}'`;
}

function serializeYamlLines(
  metadata: Record<string, unknown>,
  indent = 0,
): string[] {
  const lines: string[] = [];
  const prefix = ' '.repeat(indent);

  for (const [key, value] of Object.entries(metadata)) {
    if (Array.isArray(value)) {
      lines.push(`${prefix}${key}:`);
      for (const item of value) {
        if (item && typeof item === 'object' && !Array.isArray(item)) {
          const entries = Object.entries(item);
          if (entries.length === 0) {
            lines.push(`${prefix}  - `);
            continue;
          }
          const [[firstKey, firstValue], ...rest] = entries;
          lines.push(`${prefix}  - ${firstKey}: ${quoteScalar(firstValue)}`);
          for (const [subKey, subValue] of rest) {
            lines.push(
              ...serializeYamlLines({ [subKey]: subValue }, indent + 4),
            );
          }
        } else {
          lines.push(`${prefix}  - ${quoteScalar(item)}`);
        }
      }
      continue;
    }

    if (value && typeof value === 'object') {
      lines.push(`${prefix}${key}:`);
      for (const [subKey, subValue] of Object.entries(value)) {
        lines.push(...serializeYamlLines({ [subKey]: subValue }, indent + 2));
      }
      continue;
    }

    lines.push(`${prefix}${key}: ${quoteScalar(value)}`);
  }

  return lines;
}

function serializeYaml(metadata: Record<string, unknown>): string {
  return serializeYamlLines(metadata).join('\n');
}

interface Slice {
  slice_id: string;
  title: string;
  status: string;
  goal: string;
  estimate_hours: number | string;
  files_to_change: string[];
  acceptance_criteria: string[];
  parallelizable: boolean;
  dependencies: string[];
  next_slice?: string;
}

interface StepMetadata extends Record<string, unknown> {
  phase: number;
  step: number;
  title: string;
  status: string;
  goal: string;
  tdd_sequence?: string;
  expansion: string;
  auto_expand: boolean;
  mode: string;
  source_of_truth: string;
  copy_paste: boolean;
  next_step: string;
  skills: string[];
  validation: string[];
  acceptance_criteria: string[];
  slices: Slice[];
}

function defaultSlices(stepNumber: number): Slice[] {
  const prefix = `step-${stepNumber}`;
  return [
    {
      slice_id: `${prefix}-red-tests`,
      title: 'Write red tests',
      status: '[PLANNED]',
      goal: 'red-testing',
      estimate_hours: 4,
      files_to_change: ['TBD'],
      acceptance_criteria: ['Red tests fail for expected behavior'],
      parallelizable: false,
      dependencies: [],
      next_slice: `${prefix}-core`,
    },
    {
      slice_id: `${prefix}-core`,
      title: 'Implement core behavior',
      status: '[PLANNED]',
      goal: 'implementing',
      estimate_hours: 3,
      files_to_change: ['TBD'],
      acceptance_criteria: ['Implementation satisfies red tests'],
      parallelizable: false,
      dependencies: [`${prefix}-red-tests`],
      next_slice: `${prefix}-green`,
    },
    {
      slice_id: `${prefix}-green`,
      title: 'Green validation and coverage guard',
      status: '[PLANNED]',
      goal: 'green-testing',
      estimate_hours: 4,
      files_to_change: ['TBD'],
      acceptance_criteria: ['All tests pass and coverage guard is satisfied'],
      parallelizable: false,
      dependencies: [`${prefix}-core`],
    },
  ];
}

function baseValidStep(testId: string, stepNumber = 1): StepMetadata {
  return {
    phase: 2,
    step: stepNumber,
    title: 'Implement test feature',
    status: '[WIP]',
    goal: 'implementing',
    tdd_sequence: 'red-green',
    expansion: 'slices',
    auto_expand: true,
    mode: 'fresh-session',
    source_of_truth: `plans/${TEMP_FILE_PREFIX}${testId}.plans.md`,
    copy_paste: true,
    next_step: 'null',
    skills: ['implementation-standards'],
    validation: [
      `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/${TEMP_FILE_PREFIX}${testId}.plans.md`,
    ],
    acceptance_criteria: ['Slice workflow passes the step-packet gate'],
    slices: defaultSlices(stepNumber),
  };
}

function baseValidPhase(testId: string) {
  return {
    phase: 1,
    title: 'Design phase',
    status: '[WIP]',
    goal: 'planning',
    expansion: 'steps',
    auto_expand: false,
    mode: 'fresh-session',
    source_of_truth: `plans/${TEMP_FILE_PREFIX}${testId}.plans.md`,
    copy_paste: true,
    next_phase: 'null',
    skills: ['plan-alignment'],
    validation: [
      `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/${TEMP_FILE_PREFIX}${testId}.plans.md`,
    ],
    acceptance_criteria: ['Phase kickoff packet passes the step-packet gate'],
    placeholder_steps: ['Step 01 — Plan the phase'],
  };
}

/**
 * Creates a temporary plan file in plans/ containing a single YAML block.
 */
async function createTempPlanFile(
  metadata: Record<string, unknown>,
  testId: string,
): Promise<string> {
  const fileName = `${TEMP_FILE_PREFIX}${testId}.plans.md`;
  const filePath = path.join(PLANS_DIR, fileName);
  const content = [
    `# Temp plan for step-packet gate test (${testId})`,
    '',
    '```yaml',
    serializeYaml(metadata),
    '```',
    '',
    '**Stop conditions:** Gate validation passes.',
    '',
    '**Required validation:** Gate runs without violations.',
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
    // ignore cleanup errors
  }
}

/** Removes any leftover temp plan files from a prior interrupted run. */
async function cleanupAllTempPlanFiles(): Promise<void> {
  let entries: string[] = [];
  try {
    entries = await readdir(PLANS_DIR);
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.startsWith(TEMP_FILE_PREFIX) && entry.endsWith('.plans.md')) {
      await cleanupTempPlanFile(entry);
    }
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
    v.blockId?.startsWith(`plans/${fileName}:`),
  );
}

beforeAll(cleanupAllTempPlanFiles);
afterAll(cleanupAllTempPlanFiles);

describe('step-packet.gate.mjs', () => {
  describe('valid WIP packets pass', () => {
    it('accepts a valid step packet with slices', async () => {
      const testId = 'valid-step-with-slices';
      const fileName = await createTempPlanFile(baseValidStep(testId), testId);

      try {
        const result = runGate();
        expect(getViolationsForFile(result, fileName).length).toBe(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('accepts a valid phase kickoff packet', async () => {
      const testId = 'valid-phase-kickoff';
      const fileName = await createTempPlanFile(baseValidPhase(testId), testId);

      try {
        const result = runGate();
        expect(getViolationsForFile(result, fileName).length).toBe(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });
  });

  describe('routing field validation', () => {
    it('rejects a WIP step packet without goal', async () => {
      const testId = 'missing-goal';
      const meta = baseValidStep(testId);
      delete (meta as Record<string, unknown>).goal;
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.missingField === 'goal' || v.legacy),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects deprecated agent field', async () => {
      const testId = 'agent-field';
      const meta = { ...baseValidStep(testId), agent: '04-implementing' };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.deprecatedField === 'agent' || v.legacy),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects deprecated agent_file field', async () => {
      const testId = 'agent-file-field';
      const meta = {
        ...baseValidStep(testId),
        agent_file: '.github/agents/04-implementing.agent.md',
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) => v.deprecatedField === 'agent_file' || v.legacy,
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects invalid goal value', async () => {
      const testId = 'invalid-goal';
      const meta = { ...baseValidStep(testId), goal: 'invalid-goal' };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) =>
              v.invalidField === 'goal' && v.invalidValue === 'invalid-goal',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it.each([
      'planning',
      'researching',
      'red-testing',
      'implementing',
      'green-testing',
      'documenting',
      'logging',
      'helping',
    ] as const)('accepts valid goal "%s" without agent', async (goalValue) => {
      const testId = `valid-goal-${goalValue}`;
      const meta = { ...baseValidStep(testId), goal: goalValue };
      if (goalValue !== 'implementing') {
        (meta as Record<string, unknown>).expansion = 'none';
        delete (meta as Record<string, unknown>).tdd_sequence;
        delete (meta as Record<string, unknown>).slices;
        (meta as Record<string, unknown>).auto_expand = false;
      }
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        expect(getViolationsForFile(result, fileName).length).toBe(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });
  });

  describe('slice expansion validation', () => {
    it('requires auto_expand: true when expansion is slices', async () => {
      const testId = 'slices-no-auto-expand';
      const meta = { ...baseValidStep(testId), auto_expand: false };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) => v.invalidField === 'auto_expand' && v.expected === true,
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('requires tdd_sequence when expansion is slices', async () => {
      const testId = 'slices-no-tdd';
      const meta = baseValidStep(testId);
      delete (meta as Record<string, unknown>).tdd_sequence;
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(violations.some((v) => v.missingField === 'tdd_sequence')).toBe(
          true,
        );
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects invalid tdd_sequence value', async () => {
      const testId = 'invalid-tdd';
      const meta = {
        ...baseValidStep(testId),
        tdd_sequence: 'invalid-sequence',
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) =>
              v.invalidField === 'tdd_sequence' &&
              v.invalidValue === 'invalid-sequence',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects a slice missing required keys', async () => {
      const testId = 'slice-missing-keys';
      const meta = baseValidStep(testId);
      meta.slices = [
        {
          slice_id: 'step-1-bad',
          title: 'Bad slice',
        },
      ] as Slice[];
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) => v.sliceId === 'step-1-bad' && v.missingField === 'goal',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects a slice with an invalid goal', async () => {
      const testId = 'slice-invalid-goal';
      const meta = baseValidStep(testId);
      meta.slices = defaultSlices(1);
      (meta.slices[0] as unknown as Record<string, unknown>).goal = 'planning';
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) =>
              v.sliceId === 'step-1-red-tests' &&
              v.invalidField === 'goal' &&
              v.invalidValue === 'planning',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects a slice with a non-numeric estimate_hours', async () => {
      const testId = 'slice-bad-estimate';
      const meta = baseValidStep(testId);
      meta.slices = defaultSlices(1);
      (meta.slices[1] as unknown as Record<string, unknown>).estimate_hours =
        'six';
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) =>
              v.sliceId === 'step-1-core' &&
              v.invalidField === 'estimate_hours',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects a slice with wrong goal order for red-green sequence', async () => {
      const testId = 'slice-wrong-order';
      const meta = baseValidStep(testId);
      meta.slices = defaultSlices(1);
      (meta.slices[0] as unknown as Record<string, unknown>).goal =
        'implementing';
      (meta.slices[1] as unknown as Record<string, unknown>).goal =
        'red-testing';
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) =>
              v.sliceId === 'step-1-red-tests' &&
              v.invalidField === 'goal' &&
              v.expected === 'red-testing',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });
  });

  describe('phase kickoff validation', () => {
    it('requires auto_expand: false for phase packets', async () => {
      const testId = 'phase-auto-expand-true';
      const meta = { ...baseValidPhase(testId), auto_expand: true };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) => v.invalidField === 'auto_expand' && v.expected === false,
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('requires expansion: steps for phase packets', async () => {
      const testId = 'phase-expansion-slices';
      const meta = { ...baseValidPhase(testId), expansion: 'slices' };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) => v.invalidField === 'expansion' && v.expected === 'steps',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('requires goal: planning for phase packets', async () => {
      const testId = 'phase-goal-implementing';
      const meta = { ...baseValidPhase(testId), goal: 'implementing' };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some(
            (v) => v.invalidField === 'goal' && v.expected === 'planning',
          ),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });
  });

  describe('pre_execute_hook validation', () => {
    it('passes when pre_execute_hook is omitted', async () => {
      const testId = 'hook-omitted';
      const meta = baseValidStep(testId);
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        expect(getViolationsForFile(result, fileName).length).toBe(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('passes when pre_execute_hook has a valid shape', async () => {
      const testId = 'hook-valid-shape';
      const meta = {
        ...baseValidStep(testId),
        pre_execute_hook: { tool: 'fetchContext', args: { query: 'mutation' } },
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        expect(getViolationsForFile(result, fileName).length).toBe(0);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('surfaces a valid pre_execute_hook in gate evidence', async () => {
      const testId = 'hook-surfaced';
      const hook = { tool: 'fetchContext', args: { query: 'mutation' } };
      const meta = { ...baseValidStep(testId), pre_execute_hook: hook };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        expect(result.evidence.preExecuteHooks).toContainEqual({
          blockId: expect.stringContaining(fileName),
          tool: hook.tool,
          args: hook.args,
        });
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects pre_execute_hook that is not an object', async () => {
      const testId = 'hook-scalar';
      const meta = {
        ...baseValidStep(testId),
        pre_execute_hook: 'fetchContext',
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.invalidField === 'pre_execute_hook'),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects pre_execute_hook missing tool', async () => {
      const testId = 'hook-missing-tool';
      const meta = {
        ...baseValidStep(testId),
        pre_execute_hook: { args: { query: 'mutation' } },
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.invalidField === 'pre_execute_hook'),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects pre_execute_hook with non-string tool', async () => {
      const testId = 'hook-non-string-tool';
      const meta = {
        ...baseValidStep(testId),
        pre_execute_hook: { tool: 123, args: {} },
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.invalidField === 'pre_execute_hook'),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects pre_execute_hook with empty tool', async () => {
      const testId = 'hook-empty-tool';
      const meta = {
        ...baseValidStep(testId),
        pre_execute_hook: { tool: '', args: {} },
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.invalidField === 'pre_execute_hook'),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects pre_execute_hook missing args', async () => {
      const testId = 'hook-missing-args';
      const meta = {
        ...baseValidStep(testId),
        pre_execute_hook: { tool: 'fetchContext' },
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.invalidField === 'pre_execute_hook'),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });

    it('rejects pre_execute_hook with non-object args', async () => {
      const testId = 'hook-non-object-args';
      const meta = {
        ...baseValidStep(testId),
        pre_execute_hook: { tool: 'fetchContext', args: 'query' },
      };
      const fileName = await createTempPlanFile(meta, testId);

      try {
        const result = runGate();
        const violations = getViolationsForFile(result, fileName);
        expect(
          violations.some((v) => v.invalidField === 'pre_execute_hook'),
        ).toBe(true);
      } finally {
        await cleanupTempPlanFile(fileName);
      }
    });
  });
});

describe('C2 pre_execute_hook contracts', () => {
  describe('04-implementing agent body', () => {
    it('contains pre_execute_hook instructions in its body', async () => {
      const agentPath = path.join(
        REPO_ROOT,
        '.github',
        'agents',
        '04-implementing.agent.md',
      );
      const content = await readFile(agentPath, 'utf-8');
      const body = content.replace(/^---\n[\s\S]*?\n---\n/, '');
      expect(body).toMatch(/pre_execute_hook/);
    });
  });

  describe('plan sample packet', () => {
    const PLAN_PATH = path.join(
      REPO_ROOT,
      'plans',
      'Cortex_Orchestration_Single_Source_of_Truth.plans.md',
    );
    const SAMPLE_MARKER = 'Sample step packet using the new convention';

    async function readSampleBlock(): Promise<string> {
      const content = await readFile(PLAN_PATH, 'utf-8');
      const markerIndex = content.indexOf(SAMPLE_MARKER);
      if (markerIndex === -1) return '';
      const afterMarker = content.slice(markerIndex + SAMPLE_MARKER.length);
      const match = afterMarker.match(/```yaml\r?\n([\s\S]*?)```/);
      return match?.[1] ?? '';
    }

    it('declares a pre_execute_hook', async () => {
      const sampleBlock = await readSampleBlock();
      expect(sampleBlock).toMatch(/pre_execute_hook:/);
    });

    it('references neataptic-workflow-mcp/get_slice_context in the hook', async () => {
      const sampleBlock = await readSampleBlock();
      expect(sampleBlock).toMatch(
        /pre_execute_hook:\s*\n\s*tool:\s*['"]neataptic-workflow-mcp\/get_slice_context['"]/,
      );
    });
  });
});
