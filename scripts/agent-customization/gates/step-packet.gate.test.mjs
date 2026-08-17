/**
 * @module step-packet.gate.test
 * @description Coverage tests for step-packet.gate.mjs (top-level await, no exports).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/step-packet.gate.mjs',
);

// --- Mock state (captured by mock factory closures) ---
let mockReaddirEntries;
let mockReaddirError;
let mockPlanContents;
let mockReadThrow;
let mockParsedYamlQueue;
let mockParseError;

jest.unstable_mockModule('node:fs/promises', () => ({
  readdir: async () => {
    if (mockReaddirError) throw mockReaddirError;
    return mockReaddirEntries;
  },
  readFile: async (filePath) => {
    const rel = path.relative(REPO_ROOT, filePath).replace(/\\/g, '/');
    if (mockReadThrow && mockReadThrow.has(rel)) throw new Error('readerr');
    return mockPlanContents[rel] ?? '';
  },
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => ({ json: argv.includes('--json') }),
  parsePlanYamlBlock: () => {
    if (mockParseError) throw mockParseError;
    return mockParsedYamlQueue.shift() ?? {};
  },
  repoRoot: REPO_ROOT,
}));

// --- Helpers ---

/** Builds a ```yaml fenced block with the given status line. */
function yamlBlock(statusLine, body = '') {
  return '```yaml\n' + statusLine + '\n' + body + '```\n';
}

/** A minimal valid step block metadata (expansion: none). */
function validStep(overrides = {}) {
  return {
    phase: 1,
    step: 1,
    title: 'Test step',
    status: 'WIP',
    goal: 'implementing',
    mode: 'm',
    source_of_truth: 's',
    copy_paste: 'c',
    next_step: 'next',
    skills: ['s'],
    validation: ['v'],
    acceptance_criteria: ['a'],
    expansion: 'none',
    ...overrides,
  };
}

/** A minimal valid phase block metadata. */
function validPhase(overrides = {}) {
  return {
    phase: 1,
    title: 'Test phase',
    status: 'WIP',
    goal: 'planning',
    expansion: 'steps',
    auto_expand: false,
    mode: 'm',
    source_of_truth: 's',
    copy_paste: 'c',
    next_phase: 'next',
    skills: ['s'],
    validation: ['v'],
    acceptance_criteria: ['a'],
    placeholder_steps: [],
    ...overrides,
  };
}

/** A valid slice for the given goal. */
function slice(goal, overrides = {}) {
  return {
    slice_id: 's1',
    title: 't',
    status: 'WIP',
    goal,
    estimate_hours: 2,
    files_to_change: ['f'],
    acceptance_criteria: ['a'],
    parallelizable: false,
    dependencies: [],
    ...overrides,
  };
}

/**
 * Imports the gate (triggering top-level execution) with the given argv and
 * mock state. Returns the parsed JSON result (if --json) or the raw log lines.
 */
async function runGate(argv, { plans, queue, parseError, readdirError, readThrow } = {}) {
  mockReaddirEntries = plans ? Object.keys(plans).map((k) => k.replace('plans/', '')) : [];
  mockReaddirError = readdirError ?? null;
  mockPlanContents = plans ?? {};
  mockReadThrow = readThrow ?? null;
  mockParsedYamlQueue = queue ? [...queue] : [];
  mockParseError = parseError ?? null;

  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  const originalArgv = process.argv;
  try {
    jest.resetModules();
    process.argv = [process.execPath, GATE_PATH, ...argv];
    await import('./step-packet.gate.mjs');
    await new Promise((r) => setTimeout(r, 150));
  } finally {
    console.log = originalLog;
    process.argv = originalArgv;
  }
  return logs;
}

/** Runs the gate with --json and returns the parsed result object. */
async function runJson(config) {
  const logs = await runGate(['--json'], config);
  return JSON.parse(logs[0]);
}

/** Plan text builder: wraps YAML blocks and optional evidence section. */
function planText(blocks, evidence = '') {
  return blocks.join('\n\n') + (evidence ? '\n\n' + evidence : '');
}

const EVIDENCE_GREEN = '## Latest validation evidence\ngreen-light: true';
const EVIDENCE_STATUS = '## Latest validation evidence\nstatus: green-light';
const EVIDENCE_NONE = '## Latest validation evidence\nstatus: failed';

describe('step-packet gate', () => {
  let originalExitCode;

  beforeEach(() => {
    jest.resetModules();
    originalExitCode = process.exitCode;
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  // ---- runStepPacketGate: readdir / readFile / no-plans ----

  it('fails when readdir throws', async () => {
    const result = await runJson({
      plans: {},
      queue: [],
      readdirError: new Error('ENOENT plans'),
    });
    assert.equal(result.pass, false);
    assert.ok(result.evidence.error.includes('ENOENT'));
    assert.ok(result.fixHint.includes('plans/ directory'));
  });

  it('passes with no plan files', async () => {
    const result = await runJson({
      plans: {},
      queue: [],
    });
    assert.equal(result.pass, true);
    assert.equal(result.evidence.plansScanned, 0);
  });

  it('skips plan when readFile throws', async () => {
    const result = await runJson({
      plans: { 'plans/bad.plans.md': 'irrelevant' },
      queue: [],
      readThrow: new Set(['plans/bad.plans.md']),
    });
    assert.equal(result.pass, true);
    assert.equal(result.evidence.blocksChecked.length, 0);
  });

  // ---- extractStatusFromYaml: non-WIP statuses are skipped ----

  it('skips blocks with PLANNED, DONE, and missing status', async () => {
    const text = planText([
      yamlBlock('status: PLANNED'),
      yamlBlock('status: DONE'),
      yamlBlock('# no status here'),
    ]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [],
    });
    assert.equal(result.pass, true);
    assert.equal(result.evidence.blocksChecked.length, 0);
  });

  it('accepts bracketed, quoted, and commented WIP status', async () => {
    const text = planText([
      yamlBlock('status: [WIP]'),
      yamlBlock("status: 'WIP'"),
      yamlBlock('status: WIP # comment'),
    ]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [
        validStep(),
        validStep(),
        validStep(),
      ],
    });
    assert.equal(result.pass, true);
    assert.equal(result.evidence.blocksChecked.length, 3);
  });

  // ---- parse error ----

  it('records parseError when parsePlanYamlBlock throws', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [],
      parseError: new Error('yaml parse boom'),
    });
    assert.equal(result.pass, false);
    assert.equal(result.evidence.violations.length, 1);
    assert.ok(result.evidence.violations[0].parseError.includes('yaml parse boom'));
  });

  // ---- isLegacyBlock ----

  it('flags legacy block with no expansion field', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [{ status: 'WIP' }],
    });
    assert.equal(result.pass, false);
    assert.equal(result.evidence.violations[0].legacy, true);
  });

  it('flags legacy block with agent field', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [{ expansion: 'slices', agent: '04-implementing' }],
    });
    assert.equal(result.evidence.violations[0].legacy, true);
  });

  it('flags legacy block with agent_file field', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [{ expansion: 'slices', agent_file: 'foo.agent.md' }],
    });
    assert.equal(result.evidence.violations[0].legacy, true);
  });

  it('records missingField=phase for non-object metadata (string)', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: ['not-an-object'],
    });
    assert.equal(result.pass, false);
    assert.equal(result.evidence.violations[0].missingField, 'phase');
  });

  // ---- validatePhaseBlock ----

  it('passes with a fully valid phase block', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validPhase()],
    });
    assert.equal(result.pass, true);
  });

  it('records all phase block violations', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    // Build a phase block that is missing acceptance_criteria and placeholder_steps
    // (validPhase provides them by default, so we must explicitly delete them).
    const phaseData = validPhase({
      goal: 'implementing',
      expansion: 'slices',
      auto_expand: true,
      skills: [],
      validation: 'notarray',
    });
    delete phaseData.acceptance_criteria;
    delete phaseData.placeholder_steps;
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [phaseData],
    });
    const v = result.evidence.violations;
    assert.equal(result.pass, false);
    // missing acceptance_criteria + placeholder_steps
    assert.ok(v.some((x) => x.missingField === 'acceptance_criteria' && x.level === 'phase'));
    assert.ok(v.some((x) => x.missingField === 'placeholder_steps'));
    assert.ok(v.some((x) => x.invalidField === 'goal' && x.level === 'phase'));
    assert.ok(v.some((x) => x.invalidField === 'expansion' && x.level === 'phase'));
    assert.ok(v.some((x) => x.invalidField === 'auto_expand' && x.level === 'phase'));
    assert.ok(v.some((x) => x.message && x.message.includes('skills must be a non-empty list')));
    assert.ok(v.some((x) => x.message && x.message.includes('validation must be a non-empty list')));
  });

  it('records skills non-array violation for phase', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validPhase({ skills: 'notarray' })],
    });
    assert.ok(result.evidence.violations.some((x) => x.message.includes('skills must be a non-empty list')));
  });

  // ---- validateStepBlock: valid ----

  it('passes with a fully valid step block (expansion: none)', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep()],
    });
    assert.equal(result.pass, true);
  });

  it('records missing fields for step block', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [{ step: 1, expansion: 'none' }],
    });
    const v = result.evidence.violations;
    assert.ok(v.some((x) => x.missingField === 'phase' && x.level === 'step'));
    assert.ok(v.some((x) => x.missingField === 'title' && x.level === 'step'));
    assert.ok(v.some((x) => x.missingField === 'goal' && x.level === 'step'));
  });

  it('records deprecated agent and agent_file fields', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    // Blocks with agent/agent_file are caught by isLegacyBlock() before
    // validateStepBlock can flag them as deprecated, so they are recorded
    // as legacy violations with the migration hint message.
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ agent: '04', agent_file: 'foo.agent.md' })],
    });
    const v = result.evidence.violations;
    assert.ok(v.some((x) => x.legacy === true));
    assert.ok(v.some((x) => x.message && x.message.includes('Legacy format detected')));
  });

  it('records invalid goal violation', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ goal: 'bogus' })],
    });
    assert.ok(result.evidence.violations.some((x) => x.invalidField === 'goal'));
  });

  it('records invalid tdd_sequence violation', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ tdd_sequence: 'bogus' })],
    });
    assert.ok(result.evidence.violations.some((x) => x.invalidField === 'tdd_sequence'));
  });

  it('records invalid expansion violation for step', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ expansion: 'bogus' })],
    });
    assert.ok(result.evidence.violations.some((x) => x.invalidField === 'expansion'));
  });

  it('records empty skills, validation, and acceptance_criteria violations', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ skills: [], validation: [], acceptance_criteria: [] })],
    });
    const v = result.evidence.violations;
    assert.ok(v.some((x) => x.message.includes('skills must be a non-empty list')));
    assert.ok(v.some((x) => x.message.includes('validation must be a non-empty list')));
    assert.ok(v.some((x) => x.message.includes('acceptance_criteria must be a non-empty list')));
  });

  it('records non-array skills, validation, and acceptance_criteria violations', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ skills: 'x', validation: 'x', acceptance_criteria: 'x' })],
    });
    const v = result.evidence.violations;
    assert.ok(v.some((x) => x.message.includes('skills must be a non-empty list')));
    assert.ok(v.some((x) => x.message.includes('validation must be a non-empty list')));
    assert.ok(v.some((x) => x.message.includes('acceptance_criteria must be a non-empty list')));
  });

  // ---- validatePreExecuteHook ----

  it('accepts a valid pre_execute_hook and collects it', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        pre_execute_hook: { tool: 'cortex:search', args: { query: 'foo' } },
      })],
    });
    assert.equal(result.pass, true);
    assert.equal(result.evidence.preExecuteHooks.length, 1);
    assert.equal(result.evidence.preExecuteHooks[0].tool, 'cortex:search');
  });

  // Pre-execute hook failure modes — combine into one plan with multiple blocks
  it('rejects all invalid pre_execute_hook shapes', async () => {
    const blocks = Array.from({ length: 8 }, () => yamlBlock('status: WIP'));
    const text = planText(blocks);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [
        validStep({ pre_execute_hook: [] }),                 // array → not object
        validStep({ pre_execute_hook: null }),                // null → not object
        validStep({ pre_execute_hook: {} }),                  // missing tool
        validStep({ pre_execute_hook: { tool: 42 } }),        // tool not string
        validStep({ pre_execute_hook: { tool: '' } }),        // tool empty
        validStep({ pre_execute_hook: { tool: 'ok' } }),      // missing args
        validStep({ pre_execute_hook: { tool: 'ok', args: [] } }), // args array
        validStep({ pre_execute_hook: { tool: 'ok', args: null } }), // args null
      ],
    });
    const v = result.evidence.violations;
    assert.equal(v.length, 8);
    assert.ok(v.every((x) => x.invalidField === 'pre_execute_hook'));
  });

  // ---- expansion: slices ----

  it('rejects expansion=slices with auto_expand != true', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: false,
        tdd_sequence: 'red-green',
        slices: [slice('red-testing'), slice('implementing'), slice('green-testing')],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.invalidField === 'auto_expand'));
  });

  it('rejects expansion=slices with missing tdd_sequence', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        slices: [slice('red-testing'), slice('implementing'), slice('green-testing')],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.missingField === 'tdd_sequence'));
  });

  it('rejects expansion=slices with empty slices list', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.missingField === 'slices'));
  });

  it('rejects expansion=slices with non-array slices', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: 'notarray',
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.missingField === 'slices'));
  });

  // ---- validateSlices: count violations ----

  it('rejects >5 slices', async () => {
    const slices6 = [
      slice('red-testing', { slice_id: 's1' }),
      slice('implementing', { slice_id: 's2' }),
      slice('implementing', { slice_id: 's3' }),
      slice('implementing', { slice_id: 's4' }),
      slice('implementing', { slice_id: 's5' }),
      slice('green-testing', { slice_id: 's6' }),
    ];
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: slices6,
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.message.includes('exceeding the 5-slice')));
  });

  it('rejects green-only with <2 slices', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'green-only',
        slices: [slice('implementing')],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.message.includes('green-only tdd_sequence requires at least 2')));
  });

  it('rejects red-green with <3 slices', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [slice('red-testing'), slice('green-testing')],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.message.includes('red-green tdd_sequence requires at least 3')));
  });

  // ---- validateSlices: per-slice validations ----

  it('rejects slice with missing keys', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [{ slice_id: 's1' }],
      })],
    });
    const v = result.evidence.violations;
    assert.ok(v.some((x) => x.missingField === 'title' && x.sliceId === 's1'));
    assert.ok(v.some((x) => x.missingField === 'goal' && x.sliceId === 's1'));
  });

  it('rejects slice with invalid goal', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [
          slice('planning', { slice_id: 's1' }),
          slice('implementing', { slice_id: 's2' }),
          slice('green-testing', { slice_id: 's3' }),
        ],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.invalidField === 'goal' && x.sliceId === 's1' && x.message.includes('invalid goal')));
  });

  it('rejects wrong position goal for green-only', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'green-only',
        slices: [
          slice('green-testing', { slice_id: 's1' }),
          slice('implementing', { slice_id: 's2' }),
        ],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.expected === 'implementing' && x.sliceId === 's1'));
    assert.ok(result.evidence.violations.some((x) => x.expected === 'green-testing' && x.sliceId === 's2'));
  });

  it('rejects wrong position goal for red-green', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [
          slice('implementing', { slice_id: 's1' }),
          slice('red-testing', { slice_id: 's2' }),
          slice('implementing', { slice_id: 's3' }),
        ],
      })],
    });
    const v = result.evidence.violations;
    assert.ok(v.some((x) => x.expected === 'red-testing' && x.sliceId === 's1'));
    assert.ok(v.some((x) => x.expected === 'implementing' && x.sliceId === 's2'));
    assert.ok(v.some((x) => x.expected === 'green-testing' && x.sliceId === 's3'));
  });

  it('rejects slice with invalid field types', async () => {
    const badSlice = {
      slice_id: 's1',
      title: 't',
      status: 'WIP',
      goal: 'red-testing',
      estimate_hours: 'notnum',
      files_to_change: 'notarray',
      acceptance_criteria: 'notarray',
      parallelizable: 'notbool',
      dependencies: 'notarray',
    };
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [badSlice, slice('implementing', { slice_id: 's2' }), slice('green-testing', { slice_id: 's3' })],
      })],
    });
    const v = result.evidence.violations;
    assert.ok(v.some((x) => x.invalidField === 'files_to_change'));
    assert.ok(v.some((x) => x.invalidField === 'acceptance_criteria'));
    assert.ok(v.some((x) => x.invalidField === 'parallelizable'));
    assert.ok(v.some((x) => x.invalidField === 'dependencies'));
    assert.ok(v.some((x) => x.invalidField === 'estimate_hours' && x.message.includes('must be a number')));
  });

  it('rejects slice with estimate_hours > 4', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [
          slice('red-testing', { slice_id: 's1', estimate_hours: 5 }),
          slice('implementing', { slice_id: 's2' }),
          slice('green-testing', { slice_id: 's3' }),
        ],
      })],
    });
    assert.ok(result.evidence.violations.some((x) => x.invalidField === 'estimate_hours' && x.invalidValue === 5));
  });

  it('handles null/falsy slice entries', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [null, slice('implementing', { slice_id: 's2' }), slice('green-testing', { slice_id: 's3' })],
      })],
    });
    const v = result.evidence.violations;
    // null slice: slice_id defaults to 'slice-0', all keys missing
    assert.ok(v.some((x) => x.sliceId === 'slice-0'));
  });

  it('passes with valid green-only slices (2)', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'green-only',
        slices: [
          slice('implementing', { slice_id: 's1' }),
          slice('green-testing', { slice_id: 's2' }),
        ],
      })],
    });
    assert.equal(result.pass, true);
  });

  it('passes with valid red-green slices (3)', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        expansion: 'slices',
        auto_expand: true,
        tdd_sequence: 'red-green',
        slices: [
          slice('red-testing', { slice_id: 's1' }),
          slice('implementing', { slice_id: 's2' }),
          slice('green-testing', { slice_id: 's3' }),
        ],
      })],
    });
    assert.equal(result.pass, true);
  });

  // ---- planReadinessWarnings / checkPlanGreenLight ----

  it('emits readiness warning for red-testing goal without green-light', async () => {
    const text = planText([yamlBlock('status: WIP')], EVIDENCE_NONE);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ goal: 'red-testing' })],
    });
    assert.ok(result.evidence.planReadinessWarnings.some((w) => w.goal === 'red-testing'));
  });

  it('emits readiness warning for implementing goal without green-light', async () => {
    const text = planText([yamlBlock('status: WIP')], EVIDENCE_NONE);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ goal: 'implementing' })],
    });
    assert.ok(result.evidence.planReadinessWarnings.some((w) => w.goal === 'implementing'));
  });

  it('does not emit readiness warning when green-light is present (green-light: true)', async () => {
    const text = planText([yamlBlock('status: WIP')], EVIDENCE_GREEN);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ goal: 'red-testing' })],
    });
    assert.equal(result.evidence.planReadinessWarnings.length, 0);
  });

  it('does not emit readiness warning when green-light via status: green-light', async () => {
    const text = planText([yamlBlock('status: WIP')], EVIDENCE_STATUS);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ goal: 'implementing' })],
    });
    assert.equal(result.evidence.planReadinessWarnings.length, 0);
  });

  it('does not emit readiness warning for planning goal without green-light', async () => {
    const text = planText([yamlBlock('status: WIP')], EVIDENCE_NONE);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ goal: 'planning' })],
    });
    assert.equal(result.evidence.planReadinessWarnings.length, 0);
  });

  it('does not emit readiness warning when no evidence section exists', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({ goal: 'red-testing' })],
    });
    assert.ok(result.evidence.planReadinessWarnings.some((w) => w.goal === 'red-testing'));
  });

  // ---- Neither step nor phase → missingField=phase ----

  it('records missingField=phase when neither step nor phase defined', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [{ expansion: 'none', title: 't', status: 'WIP' }],
    });
    assert.equal(result.evidence.violations[0].missingField, 'phase');
  });

  // ---- output formats ----

  it('emits PASS text without --json when pass=true', async () => {
    const logs = await runGate([], {
      plans: { 'plans/test.plans.md': planText([yamlBlock('status: WIP')]) },
      queue: [validStep()],
    });
    assert.ok(logs.some((l) => l.includes('PASS')));
    assert.equal(process.exitCode, 0);
  });

  it('emits FAIL text with fixHint without --json when pass=false', async () => {
    const logs = await runGate([], {
      plans: { 'plans/test.plans.md': planText([yamlBlock('status: WIP')]) },
      queue: [{ status: 'WIP' }],
    });
    assert.ok(logs.some((l) => l.includes('FAIL')));
    assert.ok(logs.some((l) => l.includes('fixHint:')));
    assert.equal(process.exitCode, 1);
  });

  it('emits JSON with --json when pass=false (fixHint mapping coverage)', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep({
        agent: '04',
        goal: 'bogus',
        tdd_sequence: 'bogus',
        expansion: 'bogus',
        skills: [],
        validation: [],
        acceptance_criteria: [],
      })],
    });
    assert.equal(result.pass, false);
    // fixHint should contain various field references
    assert.ok(result.fixHint.includes('Fix new-format violations'));
    assert.equal(process.exitCode, 1);
  });

  it('emits JSON with --json when pass=true', async () => {
    const text = planText([yamlBlock('status: WIP')]);
    const result = await runJson({
      plans: { 'plans/test.plans.md': text },
      queue: [validStep()],
    });
    assert.equal(result.pass, true);
    assert.ok(result.fixHint.includes('All active WIP'));
    assert.equal(process.exitCode, 0);
  });

  // ---- multiple plans ----

  it('processes multiple plan files', async () => {
    const result = await runJson({
      plans: {
        'plans/a.plans.md': planText([yamlBlock('status: WIP')]),
        'plans/b.plans.md': planText([yamlBlock('status: WIP')]),
      },
      queue: [validStep(), validStep()],
    });
    assert.equal(result.pass, true);
    assert.equal(result.evidence.plansScanned, 2);
    assert.equal(result.evidence.blocksChecked.length, 2);
  });
});