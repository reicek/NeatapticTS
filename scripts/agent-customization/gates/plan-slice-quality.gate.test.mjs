/**
 * @module plan-slice-quality.gate.test
 * @description Coverage tests for plan-slice-quality.gate.mjs (top-level await gate).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

let mockOptions;
let mockReaddirResult;
let mockReaddirThrows;
let mockReadFileFn;
let mockParsePlanYamlBlockFn;
let mockRepoRoot;

jest.unstable_mockModule('node:fs/promises', () => ({
  readdir: async () => {
    if (mockReaddirThrows) throw mockReaddirThrows;
    return mockReaddirResult;
  },
  readFile: async (filePath) => mockReadFileFn(filePath),
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: () => mockOptions,
  parsePlanYamlBlock: (...args) => mockParsePlanYamlBlockFn(...args),
  repoRoot: mockRepoRoot,
}));

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function importGate(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    await withArgv([process.execPath, 'dummy', ...argv], async () => {
      await import('./plan-slice-quality.gate.mjs');
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

const WIP_BLOCK = '```yaml\nstatus: WIP\nstep: 1\n```';
const DONE_BLOCK = '```yaml\nstatus: DONE\nstep: 2\n```';
const PLANNED_BLOCK = '```yaml\nstatus: PLANNED\nstep: 3\n```';
const NO_STATUS_BLOCK = '```yaml\nstep: 4\n```';

function planWith(...blocks) {
  return `# Plan\n\n${blocks.join('\n\n')}\n`;
}

describe('plan-slice-quality gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockOptions = { json: false, help: false };
    mockReaddirResult = [];
    mockReaddirThrows = null;
    mockReadFileFn = async () => '';
    mockParsePlanYamlBlockFn = () => ({});
    mockRepoRoot = '/fake/repo';
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  it('fails when readdir throws', async () => {
    mockReaddirThrows = new Error('ENOENT');
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.ok(parsed.evidence.error);
    assert.equal(parsed.fixHint, 'Ensure the plans/ directory is readable.');
    assert.equal(process.exitCode, 1);
  });

  it('passes when no plan files exist', async () => {
    mockReaddirResult = [];
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
    assert.equal(process.exitCode, 0);
  });

  it('passes with WIP block and valid slices', async () => {
    mockReaddirResult = ['test.plans.md', 'readme.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [{ slice_id: 'A1', estimate_hours: 2 }],
      status: 'WIP',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.plansChecked.length, 1);
    assert.equal(parsed.evidence.violations.length, 0);
    assert.equal(process.exitCode, 0);
  });

  it('flags slice with estimate_hours > 4', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [{ slice_id: 'A1', estimate_hours: 5 }],
      status: 'WIP',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.violations.length, 1);
    assert.ok(parsed.evidence.violations[0].message.includes('4-hour limit'));
    assert.equal(process.exitCode, 1);
  });

  it('flags step with > 5 slices', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [
        { slice_id: 'A1', estimate_hours: 1 },
        { slice_id: 'A2', estimate_hours: 1 },
        { slice_id: 'A3', estimate_hours: 1 },
        { slice_id: 'A4', estimate_hours: 1 },
        { slice_id: 'A5', estimate_hours: 1 },
        { slice_id: 'A6', estimate_hours: 1 },
      ],
      status: 'WIP',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.violations.length, 1);
    assert.ok(
      parsed.evidence.violations[0].message.includes('5-slice-per-step'),
    );
    assert.equal(process.exitCode, 1);
  });

  it('skips non-WIP status blocks (DONE)', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(DONE_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 2,
      slices: [{ slice_id: 'B1', estimate_hours: 10 }],
      status: 'DONE',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('skips PLANNED status blocks', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(PLANNED_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 3,
      slices: [{ slice_id: 'C1', estimate_hours: 10 }],
      status: 'PLANNED',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('skips blocks with no matching status line (null statusMatch)', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(NO_STATUS_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 4,
      slices: [{ slice_id: 'D1', estimate_hours: 10 }],
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('skips when parsePlanYamlBlock throws', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => {
      throw new Error('YAML parse error');
    };
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('skips when step is undefined', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: undefined,
      slices: [{ slice_id: 'A1', estimate_hours: 10 }],
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('skips when slices is not an array', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: 'not an array',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('skips when slices is empty array', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [],
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('skips plan file when readFile throws', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => {
      throw new Error('read error');
    };
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.plansChecked.length, 0);
  });

  it('handles slice without slice_id (uses unknown)', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [{ estimate_hours: 5 }],
      status: 'WIP',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.violations[0].sliceId, 'unknown');
  });

  it('handles slice with non-number estimate_hours (no violation)', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [{ slice_id: 'A1', estimate_hours: 'big' }],
      status: 'WIP',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('handles slice with no estimate_hours (no violation)', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [{ slice_id: 'A1' }],
      status: 'WIP',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('emits PASS text without --json when no violations', async () => {
    mockReaddirResult = [];
    mockOptions = { json: false, help: false };
    const logs = await importGate([]);
    assert.ok(logs.some((l) => l.includes('PASS')));
    assert.equal(process.exitCode, 0);
  });

  it('emits FAIL text with fixHint without --json when violations exist', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [{ slice_id: 'A1', estimate_hours: 5 }],
      status: 'WIP',
    });
    mockOptions = { json: false, help: false };
    const logs = await importGate([]);
    assert.ok(logs.some((l) => l.includes('FAIL')));
    assert.ok(logs.some((l) => l.includes('fixHint:')));
    assert.equal(process.exitCode, 1);
  });

  it('handles multiple YAML blocks in one plan (WIP + DONE)', async () => {
    mockReaddirResult = ['test.plans.md'];
    mockReadFileFn = async () => planWith(WIP_BLOCK, DONE_BLOCK);
    let callCount = 0;
    mockParsePlanYamlBlockFn = () => {
      callCount++;
      if (callCount === 1) {
        return {
          step: 1,
          slices: [{ slice_id: 'A1', estimate_hours: 2 }],
          status: 'WIP',
        };
      }
      return {
        step: 2,
        slices: [{ slice_id: 'B1', estimate_hours: 10 }],
        status: 'DONE',
      };
    };
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.violations.length, 0);
  });

  it('filters non-.plans.md entries from readdir', async () => {
    mockReaddirResult = ['test.plans.md', 'readme.md', 'config.json'];
    mockReadFileFn = async () => planWith(WIP_BLOCK);
    mockParsePlanYamlBlockFn = () => ({
      step: 1,
      slices: [{ slice_id: 'A1', estimate_hours: 2 }],
      status: 'WIP',
    });
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.evidence.plansChecked.length, 1);
  });
});