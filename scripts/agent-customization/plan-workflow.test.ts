import { spawnSync } from 'node:child_process';
import {
  mkdir,
  readdir,
  readFile,
  rm,
  unlink,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';

/**
 * Tests for the plan-phase-step workflow tooling:
 * - migrate-plan-format.mjs
 * - legacy-plan-format.gate.mjs
 * - validate-plan-phase-packets.mjs
 */

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SCRIPTS_DIR = path.join(REPO_ROOT, 'scripts', 'agent-customization');
const TEMP_DIR = path.join(REPO_ROOT, 'plans', '_test-temp');
const TEMP_PREFIX = '_plan-workflow-test-';

interface MigrationReport {
  ok: boolean;
  dryRun?: boolean;
  planFile?: string;
  changed?: boolean;
  changedBlocks?: Array<{ type: string; title: string; action: string }>;
  details?: MigrationReport[];
  plansProcessed?: number;
  plansChanged?: number;
}

interface GateResult {
  pass: boolean;
  evidence: {
    plansScanned?: number;
    blocksChecked?: number;
    legacyBlocks: Array<{
      planFile: string;
      blockIndex: number;
      reasons: string[];
    }>;
  };
  fixHint: string;
  owner: string;
}

interface ValidatorReport {
  name: string;
  ok: boolean;
  issues: Array<{ severity: string; path: string; message: string }>;
  counts: { errors: number; warnings: number };
}

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

const STEP_KEY_ORDER = [
  'phase',
  'step',
  'title',
  'status',
  'goal',
  'tdd_sequence',
  'expansion',
  'auto_expand',
  'mode',
  'source_of_truth',
  'copy_paste',
  'next_step',
  'skills',
  'validation',
  'acceptance_criteria',
  'specialists',
  'slices',
];
const PHASE_KEY_ORDER = [
  'phase',
  'title',
  'status',
  'goal',
  'expansion',
  'auto_expand',
  'mode',
  'source_of_truth',
  'copy_paste',
  'next_phase',
  'skills',
  'validation',
  'acceptance_criteria',
  'placeholder_steps',
];
const SLICE_KEY_ORDER = [
  'slice_id',
  'title',
  'status',
  'goal',
  'estimate_hours',
  'files_to_change',
  'acceptance_criteria',
  'parallelizable',
  'dependencies',
  'next_slice',
];

function sortKeys(keys: string[], order: string[]): string[] {
  const orderIndex = new Map(order.map((key, index) => [key, index]));
  return [...keys].sort((a, b) => {
    const aIndex = orderIndex.get(a);
    const bIndex = orderIndex.get(b);
    if (aIndex !== undefined && bIndex !== undefined) return aIndex - bIndex;
    if (aIndex !== undefined) return -1;
    if (bIndex !== undefined) return 1;
    return a.localeCompare(b);
  });
}

function serializeLines(
  metadata: Record<string, unknown>,
  indent = 0,
  keyOrder?: string[],
): string[] {
  const lines: string[] = [];
  const prefix = ' '.repeat(indent);

  let keys = Object.keys(metadata);
  if (keyOrder) {
    keys = sortKeys(keys, keyOrder);
  }

  for (const key of keys) {
    const value = metadata[key];
    if (Array.isArray(value)) {
      lines.push(`${prefix}${key}:`);
      for (const item of value) {
        if (item && typeof item === 'object' && !Array.isArray(item)) {
          const entries = Object.entries(item);
          if (entries.length === 0) {
            lines.push(`${prefix}  - `);
            continue;
          }
          const sortedEntries: [string, unknown][] = keyOrder
            ? sortKeys(
                entries.map(([k]) => k),
                SLICE_KEY_ORDER,
              ).map((k) => [k, (item as Record<string, unknown>)[k]])
            : entries;
          const [[firstKey, firstValue], ...rest] = sortedEntries;
          lines.push(`${prefix}  - ${firstKey}: ${quoteScalar(firstValue)}`);
          for (const [subKey, subValue] of rest) {
            lines.push(...serializeLines({ [subKey]: subValue }, indent + 4));
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
        lines.push(...serializeLines({ [subKey]: subValue }, indent + 2));
      }
      continue;
    }

    lines.push(`${prefix}${key}: ${quoteScalar(value)}`);
  }

  return lines;
}

function serializeYaml(
  metadata: Record<string, unknown>,
  kind: 'step' | 'phase' = 'step',
): string {
  const keyOrder = kind === 'phase' ? PHASE_KEY_ORDER : STEP_KEY_ORDER;
  return serializeLines(metadata, 0, keyOrder).join('\n');
}

async function ensureTempDir(): Promise<void> {
  await mkdir(TEMP_DIR, { recursive: true });
  const rootPlanDir = path.join(REPO_ROOT, 'plans');
  const leftover = await readdir(rootPlanDir);
  for (const entry of leftover) {
    if (entry.startsWith(TEMP_PREFIX) && entry.endsWith('.plans.md')) {
      await unlink(path.join(rootPlanDir, entry)).catch(() => undefined);
    }
  }
}

async function cleanupTempDir(): Promise<void> {
  try {
    await rm(TEMP_DIR, { recursive: true, force: true });
  } catch {
    // ignore cleanup errors
  }
  const rootPlanDir = path.join(REPO_ROOT, 'plans');
  try {
    const leftover = await readdir(rootPlanDir);
    for (const entry of leftover) {
      if (entry.startsWith(TEMP_PREFIX) && entry.endsWith('.plans.md')) {
        await unlink(path.join(rootPlanDir, entry));
      }
    }
  } catch {
    // ignore
  }
}

async function createTempPlanFile(
  metadata: Record<string, unknown>,
  testId: string,
  options: { heading?: string; inPlansRoot?: boolean } = {},
): Promise<string> {
  const fileName = `${TEMP_PREFIX}${testId}.plans.md`;
  const dir = options.inPlansRoot ? path.join(REPO_ROOT, 'plans') : TEMP_DIR;
  const filePath = path.join(dir, fileName);
  const relativePath = path.relative(REPO_ROOT, filePath).replace(/\\/gu, '/');
  const heading = options.heading ?? '#### Step 01 — Test step [WIP]';
  const content = [
    heading,
    '',
    '```yaml',
    serializeYaml(metadata),
    '```',
    '',
    '**Stop conditions:** Validation passes.',
    '',
    '**Required validation:** Workflow tooling runs cleanly.',
  ].join('\n');
  await writeFile(filePath, content, 'utf8');
  return relativePath;
}

async function removeTempPlanFile(fileName: string): Promise<void> {
  try {
    await unlink(path.join(REPO_ROOT, fileName));
  } catch {
    // ignore
  }
}

function runScript(script: string, args: string[] = []): string {
  const scriptPath = path.join(SCRIPTS_DIR, script);
  const result = spawnSync(process.execPath, [scriptPath, ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 30000,
  });
  return result.stdout ?? '';
}

function runMigration(args: string[]): MigrationReport {
  const stdout = runScript('migrate-plan-format.mjs', args);
  try {
    return JSON.parse(stdout) as MigrationReport;
  } catch {
    throw new Error(
      `Failed to parse migration output:\n${stdout.slice(0, 500)}`,
    );
  }
}

function runLegacyGate(): GateResult {
  const stdout = runScript(path.join('gates', 'legacy-plan-format.gate.mjs'), [
    '--json',
  ]);
  try {
    return JSON.parse(stdout) as GateResult;
  } catch {
    throw new Error(
      `Failed to parse legacy gate output:\n${stdout.slice(0, 500)}`,
    );
  }
}

function runValidator(planFile: string): ValidatorReport {
  const stdout = runScript('validate-plan-phase-packets.mjs', [
    '--json',
    `--plan=${planFile}`,
  ]);
  try {
    return JSON.parse(stdout) as ValidatorReport;
  } catch {
    throw new Error(
      `Failed to parse validator output:\n${stdout.slice(0, 500)}`,
    );
  }
}

beforeAll(ensureTempDir);
afterAll(cleanupTempDir);

describe('migrate-plan-format.mjs', () => {
  it('reports no change for a conforming new-format plan', async () => {
    const planFile = await createTempPlanFile(
      {
        phase: 1,
        step: 1,
        title: 'Plan the phase',
        status: '[WIP]',
        goal: 'planning',
        expansion: 'none',
        auto_expand: false,
        mode: 'fresh-session',
        source_of_truth: 'plans/_test-temp/placeholder.plans.md',
        copy_paste: true,
        next_step: 'null',
        skills: ['plan-alignment'],
        validation: [
          'node scripts/agent-customization/validate-plan-phase-packets.mjs',
        ],
        acceptance_criteria: ['Plan validates'],
      },
      'conforming',
    );

    try {
      const filePath = path.join(REPO_ROOT, planFile);
      const content = await readFile(filePath, 'utf8');
      await writeFile(
        filePath,
        content.replace(
          /plans\/_test-temp\/placeholder\.plans\.md/gu,
          planFile,
        ),
        'utf8',
      );
      const report = runMigration(['--plan', planFile, '--dry-run', '--json']);
      expect(report.changed).toBe(false);
    } finally {
      await removeTempPlanFile(planFile);
    }
  });

  it('rewrites a legacy step block in dry-run', async () => {
    const planFile = await createTempPlanFile(
      {
        phase: 1,
        step: 2,
        agent: '04-implementing',
        agent_file: '.github/agents/04-implementing.agent.md',
        status: '[WIP]',
        mode: 'fresh-session',
        source_of_truth: 'plans/_test-temp/placeholder.plans.md',
        copy_paste: true,
        next_step: 'null',
        skills: ['implementation-standards'],
        validation: ['echo ok'],
        acceptance_criteria: ['OK'],
      },
      'legacy-step',
    );

    try {
      const report = runMigration(['--plan', planFile, '--dry-run', '--json']);
      expect(report.changed).toBe(true);
      expect(report.changedBlocks?.length).toBeGreaterThan(0);
    } finally {
      await removeTempPlanFile(planFile);
    }
  });

  it('migrates a legacy block when not in dry-run', async () => {
    const planFile = await createTempPlanFile(
      {
        phase: 1,
        step: 2,
        agent: '04-implementing',
        status: '[WIP]',
        mode: 'fresh-session',
        source_of_truth: 'plans/_test-temp/placeholder.plans.md',
        copy_paste: true,
        next_step: 'null',
        skills: ['implementation-standards'],
        validation: ['echo ok'],
        acceptance_criteria: ['OK'],
      },
      'legacy-mutate',
    );

    try {
      const report = runMigration(['--plan', planFile, '--json']);
      expect(report.changed).toBe(true);

      const updated = await readFile(path.join(REPO_ROOT, planFile), 'utf8');
      expect(updated).toContain('goal:');
      expect(updated).toContain('goal: implementing');
      expect(updated).toContain('expansion:');
      expect(updated).not.toContain('\nagent:');
    } finally {
      await removeTempPlanFile(planFile);
    }
  });
});

describe('legacy-plan-format.gate.mjs', () => {
  it('passes after migration', () => {
    const result = runLegacyGate();
    expect(result.pass).toBe(true);
  });

  it('flags a legacy active block', async () => {
    const planFile = await createTempPlanFile(
      {
        phase: 1,
        step: 2,
        agent: '04-implementing',
        status: '[WIP]',
      },
      'legacy-active',
      { heading: '#### Step 02 — Legacy step [WIP]', inPlansRoot: true },
    );

    try {
      const result = runLegacyGate();
      expect(result.pass).toBe(false);
      expect(
        result.evidence.legacyBlocks.some(
          (block) => block.planFile === planFile,
        ),
      ).toBe(true);
    } finally {
      await removeTempPlanFile(planFile);
    }
  });
});

describe('validate-plan-phase-packets.mjs', () => {
  it('passes on the RAG plan', () => {
    const report = runValidator(
      'plans/Cortex_RAG_Premium_Primary_Search.plans.md',
    );
    expect(report.ok).toBe(true);
    expect(report.counts.errors).toBe(0);
  });

  it('reports errors for an invalid WIP step', async () => {
    const planFile = await createTempPlanFile(
      {
        phase: 1,
        step: 1,
        title: 'Bad step',
        status: '[WIP]',
        goal: 'invalid-goal',
        expansion: 'none',
        auto_expand: false,
        mode: 'fresh-session',
        source_of_truth: 'plans/_test-temp/placeholder.plans.md',
        copy_paste: true,
        next_step: 'null',
        skills: ['plan-alignment'],
        validation: ['echo ok'],
        acceptance_criteria: ['OK'],
      },
      'invalid-step',
      { heading: '#### Step 01 — Bad step [WIP]' },
    );

    try {
      const report = runValidator(planFile);
      expect(report.ok).toBe(false);
      expect(report.counts.errors).toBeGreaterThan(0);
    } finally {
      await removeTempPlanFile(planFile);
    }
  });
});
