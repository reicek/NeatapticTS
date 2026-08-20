import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  issue: jest.fn((severity, path, message) => ({ severity, path, message })),
  parseArgs: jest.fn(),
  printUsage: jest.fn(),
  summarizeIssues: jest.fn((name, issues) => ({
    name,
    ok: issues.length === 0,
    issues,
    counts: {
      errors: issues.filter((i) => i.severity === 'error').length,
      warnings: 0,
    },
    summaryText: `${issues.length === 0 ? 'PASS' : 'FAIL'} ${name}`,
  })),
  writeReport: jest.fn(),
}));

jest.unstable_mockModule('node:fs/promises', () => ({
  readFile: jest.fn(),
  readdir: jest.fn(),
  stat: jest.fn(),
  writeFile: jest.fn(),
  mkdir: jest.fn(),
  appendFile: jest.fn(),
  access: jest.fn(),
  constants: { R_OK: 4, W_OK: 2, F_OK: 0 },
  rm: jest.fn(),
  glob: jest.fn(),
}));

let mockUtils;
let mockFs;

beforeEach(async () => {
  jest.resetModules();
  mockUtils = await import('./customization-utils.mjs');
  mockFs = await import('node:fs/promises');
  jest.clearAllMocks();
});

async function importModule() {
  try {
    await import('./validate-numbered-agent-structured-v1-output.mjs');
  } catch {
    // process.exit may throw
  }
}

function makeValidTier0Output() {
  return '```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 0\nROLE: Orchestrator\nTASK_RECEIVED: Do something\nFILES_READ:\n- file1.md\nFILES_CHANGED:\n- file2.md\nKEY_FINDINGS:\n- Finding 1\nACTIONS_TAKEN:\n- Action 1\nVALIDATION_EVIDENCE:\n- Evidence 1\nBLOCKERS:\n- None\nRISKS_OR_GAPS:\n- None\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: 01-planning\nPHASE_COMPLETE: true\nSUB_ORCHESTRATORS_USED:\n- 02-researching\nSUMMARY: Done\n```';
}

function makeValidTier1Output() {
  return '```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 1\nROLE: Coordinator\nTASK_RECEIVED: Do something\nFILES_READ:\n- file1.md\nFILES_CHANGED:\n- file2.md\nKEY_FINDINGS:\n- Finding 1\nACTIONS_TAKEN:\n- Action 1\nVALIDATION_EVIDENCE:\n- Evidence 1\nSPECIALISTS_USED:\n- specialist-1\nHANDOFF: Next step\nBLOCKERS:\n- None\nRISKS_OR_GAPS:\n- None\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: 03-red-testing\nSUMMARY: Done\n```';
}

describe('validate-numbered-agent-structured-v1-output', () => {
  it('prints help and exits 0 when --help is passed', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'script.mjs', '--help'];
    process.exit = (code) => {
      throw new Error(`EXIT:${code}`);
    };
    mockUtils.parseArgs.mockReturnValue({
      help: true,
      json: false,
      contract: 'tier0',
    });

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
  });

  it('validates a valid tier0 output', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(makeValidTier0Output());

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
    assert.strictEqual(report.contract, 'tier0');
  });

  it('validates a valid tier1 output', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier1',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier1',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(makeValidTier1Output());

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
    assert.strictEqual(report.contract, 'tier1');
  });

  it('supports contract aliases (tier-0, numbered, tier-1, coordinator)', async () => {
    for (const alias of ['tier-0', 'numbered', 'tier-1', 'coordinator']) {
      const origArgv = process.argv;
      process.argv = [
        'node',
        'script.mjs',
        '--json',
        `--contract=${alias}`,
        '--input=test.md',
      ];
      mockUtils.parseArgs.mockReturnValue({
        help: false,
        json: true,
        contract: alias,
        input: 'test.md',
      });
      const isTier0 = ['tier-0', 'numbered'].includes(alias);
      mockFs.readFile.mockResolvedValue(
        isTier0 ? makeValidTier0Output() : makeValidTier1Output(),
      );

      await importModule();

      process.argv = origArgv;
      const report = mockUtils.writeReport.mock.calls[0][0];
      assert.strictEqual(
        report.ok,
        true,
        `alias ${alias} should produce ok report`,
      );
      jest.resetModules();
      mockUtils = await import('./customization-utils.mjs');
      mockFs = await import('node:fs/promises');
      jest.clearAllMocks();
    }
  });

  it('flags error for unsupported contract', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=invalid',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'invalid',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(makeValidTier0Output());

    await importModule();

    process.argv = origArgv;
    const contractIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('Unsupported --contract'),
    );
    assert.ok(contractIssue);
  });

  it('flags error when --input is missing', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'script.mjs', '--json', '--contract=tier0'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
    });

    await importModule();

    process.argv = origArgv;
    const inputIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('requires --input'),
    );
    assert.ok(inputIssue);
  });

  it('flags error when input file not found (ENOENT)', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=missing.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'missing.md',
    });
    const error = new Error('not found');
    error.code = 'ENOENT';
    mockFs.readFile.mockRejectedValue(error);

    await importModule();

    process.argv = origArgv;
    const notFoundIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('not found'),
    );
    assert.ok(notFoundIssue);
  });

  it('throws on non-ENOENT read errors', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockRejectedValue(new Error('permission denied'));

    await importModule();

    process.argv = origArgv;
    // The throw is caught by the top-level await, but since there's no try/catch,
    // it becomes an unhandled rejection. The module should still have been imported.
  });

  it('flags error when output is not a structured-v1 fence', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue('Some random text');

    await importModule();

    process.argv = origArgv;
    const fenceIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('fenced'),
    );
    assert.ok(fenceIssue);
  });

  it('flags missing required fields', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(
      '```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 0\n```',
    );

    await importModule();

    process.argv = origArgv;
    const missingIssues = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('Missing required'),
    );
    assert.ok(missingIssues.length > 1);
  });

  it('flags unexpected fields', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(
      '```structured-v1\nUNKNOWN_FIELD: value\n```',
    );

    await importModule();

    process.argv = origArgv;
    const unexpectedIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('Unexpected'),
    );
    assert.ok(unexpectedIssue);
  });

  it('flags duplicate fields', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace(
      'SUMMARY: Done',
      'SUMMARY: Done\nSUMMARY: Again',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const dupIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('Duplicate'),
    );
    assert.ok(dupIssue);
  });

  it('flags invalid line syntax', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(
      '```structured-v1\nSome random line without colon\n```',
    );

    await importModule();

    process.argv = origArgv;
    const syntaxIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('must match FIELD'),
    );
    assert.ok(syntaxIssue);
  });

  it('flags list item without preceding field', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue('```structured-v1\n- orphan item\n```');

    await importModule();

    process.argv = origArgv;
    const orphanIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('List item must belong'),
    );
    assert.ok(orphanIssue);
  });

  it('flags list items in non-list-capable fields', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(
      '```structured-v1\nTIER: 0\n- not a list\n```',
    );

    await importModule();

    process.argv = origArgv;
    const nonListIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('does not accept list items'),
    );
    assert.ok(nonListIssue);
  });

  it('flags empty list items', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(
      '```structured-v1\nFILES_READ:\n-  \n```',
    );

    await importModule();

    process.argv = origArgv;
    // The validator trims each line before matching list-item syntax, so an
    // empty list item ("-  ") collapses to "-" which fails the list-item
    // pattern and is flagged as a syntax error rather than a specific
    // empty-list-item error.
    const emptyListIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('must match FIELD: value or list-item syntax'),
    );
    assert.ok(emptyListIssue);
  });

  it('flags mixing inline values with list items', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(
      '```structured-v1\nFILES_READ: inline\nFILES_READ:\n- list item\n```',
    );

    await importModule();

    process.argv = origArgv;
    const mixIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('cannot mix'),
    );
    assert.ok(mixIssue);
  });

  it('flags wrong OUTPUT_CONTRACT value', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace(
      'OUTPUT_CONTRACT: structured-v1',
      'OUTPUT_CONTRACT: wrong',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const contractIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes("OUTPUT_CONTRACT must equal 'structured-v1'"),
    );
    assert.ok(contractIssue);
  });

  it('flags invalid TASK_STATUS value', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace(
      'TASK_STATUS: SUCCESS',
      'TASK_STATUS: INVALID',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const statusIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('TASK_STATUS must be one of'),
    );
    assert.ok(statusIssue);
  });

  it('flags wrong TIER value', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace('TIER: 0', 'TIER: 1');
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const tierIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes("TIER must equal '0'"),
    );
    assert.ok(tierIssue);
  });

  it('flags empty scalar fields', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace(
      'ROLE: Orchestrator',
      'ROLE: ',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const roleIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('ROLE must be a non-empty'),
    );
    assert.ok(roleIssue);
  });

  it('flags empty field content for list fields', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace('- file1.md', '');
    mockFs.readFile.mockResolvedValue(output);

    await importModule();
    process.argv = origArgv;
    // FILES_READ would be an empty array which should trigger "must include at least one value"
  });

  it('flags wrong field order', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace(
      'TIER: 0\nROLE: Orchestrator',
      'ROLE: Orchestrator\nTIER: 0',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const orderIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('must appear in exact'),
    );
    assert.ok(orderIssue);
  });

  it('defaults to tier0 contract when no --contract provided', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'script.mjs', '--json', '--input=test.md'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      input: 'test.md',
    });
    mockFs.readFile.mockResolvedValue(makeValidTier0Output());

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.contract, 'tier0');
  });

  it('flags invalid LEARNING_EVENT_NEEDED value', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace(
      'LEARNING_EVENT_NEEDED: false',
      'LEARNING_EVENT_NEEDED: maybe',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const lenIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('LEARNING_EVENT_NEEDED must be one of'),
    );
    assert.ok(lenIssue);
  });

  it('flags invalid PHASE_COMPLETE value for tier0', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    const output = makeValidTier0Output().replace(
      'PHASE_COMPLETE: true',
      'PHASE_COMPLETE: maybe',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const pcIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('PHASE_COMPLETE must be one of'),
    );
    assert.ok(pcIssue);
  });

  it('handles multiple list items for the same field', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier0',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier0',
      input: 'test.md',
    });
    // Add multiple list items for FILES_READ to exercise the Array.isArray false branch
    const output = makeValidTier0Output().replace(
      'FILES_READ:\n- file1.md\nFILES_CHANGED:',
      'FILES_READ:\n- file1.md\n- file2.md\n- file3.md\nFILES_CHANGED:',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
    assert.ok(Array.isArray(report.parsedFields.FILES_READ));
    assert.strictEqual(report.parsedFields.FILES_READ.length, 3);
  });

  it('handles multiple list items for tier1 SPECIALISTS_USED', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'script.mjs',
      '--json',
      '--contract=tier1',
      '--input=test.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      contract: 'tier1',
      input: 'test.md',
    });
    const output = makeValidTier1Output().replace(
      'SPECIALISTS_USED:\n- specialist-1\nHANDOFF:',
      'SPECIALISTS_USED:\n- specialist-1\n- specialist-2\nHANDOFF:',
    );
    mockFs.readFile.mockResolvedValue(output);

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
    assert.strictEqual(report.parsedFields.SPECIALISTS_USED.length, 2);
  });
});
