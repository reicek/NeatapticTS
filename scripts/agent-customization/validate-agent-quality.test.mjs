/**
 * @module validate-agent-quality.test
 * @description Native-ESM coverage tests for validate-agent-quality.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the script.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { writeFile, rm } from 'node:fs/promises';

const REPO_ROOT = path.resolve();
const SCRIPT_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/validate-agent-quality.mjs',
);
const AGENTS_DIR = path.join(REPO_ROOT, '.github/agents');

/**
 * Temporarily sets `process.argv` for the duration of `fn`.
 *
 * @param {string[]} argv - Full argv array (typically `[execPath, scriptPath, ...flags]`).
 * @param {() => Promise<unknown>} fn - Async function to run under the override.
 * @returns {Promise<unknown>} Whatever `fn` returns.
 */
async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

/**
 * Captures all `process.stdout.write` calls for the duration of `fn`.
 *
 * Needed because `writeReport` in `customization-utils.mjs` uses
 * `process.stdout.write` directly (not `console.log`).
 *
 * @param {() => Promise<unknown>} fn - Async function whose stdout should be captured.
 * @returns {Promise<{result: unknown, output: string}>} Captured result and concatenated output.
 */
async function captureStdout(fn) {
  const chunks = [];
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (chunk, _encoding) => {
    chunks.push(typeof chunk === 'string' ? chunk : chunk.toString('utf8'));
    return true;
  };
  try {
    const result = await fn();
    return { result, output: chunks.join('') };
  } finally {
    process.stdout.write = originalWrite;
  }
}

/**
 * Suppresses `console.log` and `console.error` for the duration of `fn`.
 *
 * @param {() => Promise<unknown>} fn - Async function to run with silenced logs.
 * @returns {Promise<unknown>} Whatever `fn` returns.
 */
async function swallowLogs(fn) {
  const originalLog = console.log;
  const originalErr = console.error;
  console.log = () => {};
  console.error = () => {};
  try {
    return await fn();
  } finally {
    console.log = originalLog;
    console.error = originalErr;
  }
}

/**
 * Stubs `process.exit` so it sets a flag instead of terminating. The stub
 * throws to prevent subsequent code from executing (mimicking real exit).
 *
 * @param {() => Promise<unknown>} fn - Async function receiving `{ called }`.
 * @returns {Promise<unknown>} Whatever `fn` returns.
 */
async function withExitStub(fn) {
  const original = process.exit;
  let called = false;
  process.exit = () => {
    called = true;
    throw new Error('exit intercepted');
  };
  try {
    return await fn({ called: () => called });
  } finally {
    process.exit = original;
  }
}

/**
 * Imports a module inside `jest.isolateModulesAsync` so top-level code runs
 * fresh for each test. A 150ms delay lets unawaited top-level promises settle.
 *
 * @param {() => Promise<void>} callback - Import callback executed inside the isolate.
 * @returns {Promise<void>} Resolves after the isolate and delay complete.
 */
async function importInIsolation(callback) {
  return jest.isolateModulesAsync(async () => {
    await callback();
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
}

/**
 * Creates a temp agent file in `.github/agents/`.
 *
 * @param {string} name - File name (e.g. `zz-quality-test.agent.md`).
 * @param {string} content - Full file content including frontmatter.
 * @returns {Promise<string>} Absolute path to the created file.
 */
async function createTempAgent(name, content) {
  const filePath = path.join(AGENTS_DIR, name);
  await writeFile(filePath, content, 'utf8');
  return filePath;
}

// Import the public API directly for unit-style assertions.
import { runValidateAgentQuality } from './validate-agent-quality.mjs';

describe('validate-agent-quality export', () => {
  it('returns a passing report for real agents', async () => {
    const report = await runValidateAgentQuality();
    assert.equal(report.ok, true);
    assert.equal(report.name, 'agent quality');
    assert.ok(Array.isArray(report.agents));
    assert.ok(report.agents.length > 0);
  });
});

describe('validate-agent-quality CLI entry point', () => {
  it('runs the main flow with --json and produces valid JSON', async () => {
    const { output } = await captureStdout(() =>
      withArgv([process.execPath, SCRIPT_PATH, '--json'], () =>
        importInIsolation(async () => {
          await import(SCRIPT_PATH);
        }),
      ),
    );
    const parsed = JSON.parse(output);
    assert.equal(parsed.ok, true);
    assert.equal(parsed.name, 'agent quality');
    assert.ok(Array.isArray(parsed.agents));
    process.exitCode = 0;
  });

  it('runs the main flow without --json and produces human-readable output', async () => {
    const { output } = await captureStdout(() =>
      withArgv([process.execPath, SCRIPT_PATH], () =>
        importInIsolation(async () => {
          await import(SCRIPT_PATH);
        }),
      ),
    );
    assert.ok(output.includes('PASS'), `expected PASS in output: ${output}`);
    assert.equal(process.exitCode, 0);
    process.exitCode = 0;
  });

  it('covers the --help branch when run as the entry module', async () => {
    const originalExit = process.exit;
    let called = false;
    process.exit = () => {
      called = true;
    };
    try {
      await withArgv([process.execPath, SCRIPT_PATH, '--help'], async () => {
        await swallowLogs(() =>
          importInIsolation(async () => {
            await import(SCRIPT_PATH);
          }),
        );
        assert.ok(called);
      });
    } finally {
      process.exit = originalExit;
    }
  });

  it('covers the --fix branch without rewriting real agents', async () => {
    const mockRunFix = jest.fn().mockResolvedValue({
      name: 'agent-quality-footer-fix',
      ok: true,
      issues: [],
      fixed: [],
      unchanged: [],
      counts: { fixed: 0, unchanged: 0, checked: 0 },
    });

    jest.unstable_mockModule('./validate-agent-quality.fix.mjs', () => ({
      runFix: mockRunFix,
    }));

    const originalExit = process.exit;
    let called = false;
    process.exit = () => {
      called = true;
    };
    try {
      await withArgv(
        [process.execPath, SCRIPT_PATH, '--json', '--fix'],
        async () => {
          await captureStdout(() =>
            importInIsolation(async () => {
              await import(SCRIPT_PATH);
            }),
          );
        },
      );
      assert.ok(called);
      assert.ok(mockRunFix.mock.calls.length > 0);
    } finally {
      process.exit = originalExit;
      jest.dontMock('./validate-agent-quality.fix.mjs');
    }
  });

  it('covers the --fix branch when the fixer reports failure', async () => {
    const mockRunFix = jest.fn().mockResolvedValue({
      name: 'agent-quality-footer-fix',
      ok: false,
      issues: [
        {
          severity: 'error',
          path: '.github/agents/mock.agent.md',
          message: 'mock fix failure',
        },
      ],
      fixed: [],
      unchanged: [],
      counts: { fixed: 0, unchanged: 0, checked: 0 },
    });

    jest.unstable_mockModule('./validate-agent-quality.fix.mjs', () => ({
      runFix: mockRunFix,
    }));

    const originalExit = process.exit;
    let called = false;
    process.exit = () => {
      called = true;
    };
    try {
      await withArgv(
        [process.execPath, SCRIPT_PATH, '--json', '--fix'],
        async () => {
          await captureStdout(() =>
            importInIsolation(async () => {
              await import(SCRIPT_PATH);
            }),
          );
        },
      );
      assert.ok(called);
    } finally {
      process.exit = originalExit;
      jest.dontMock('./validate-agent-quality.fix.mjs');
    }
  });

  it('exits with code 1 when validation fails in main flow', async () => {
    const tempPath = await createTempAgent(
      'zz-quality-main-fail.agent.md',
      '---\nname: zz-quality-main-fail\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 2\nROLE: zz-quality-main-fail\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    try {
      await withArgv([process.execPath, SCRIPT_PATH, '--json'], async () => {
        await captureStdout(() =>
          importInIsolation(async () => {
            await import(SCRIPT_PATH);
          }),
        );
      });
      assert.equal(process.exitCode, 1);
    } finally {
      await rm(tempPath, { force: true });
      process.exitCode = 0;
    }
  });

  it('passes via CLI with --json', () => {
    const result = spawnSync(process.execPath, [SCRIPT_PATH, '--json'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    });
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.ok, true);
    assert.ok(Array.isArray(parsed.agents));
  });
});

describe('validate-agent-quality error-path coverage', () => {
  /**
   * Runs the validator and returns the parsed JSON report.
   * Cleans up the temp file after the run.
   *
   * @param {string} name - Temp agent file name.
   * @param {string} content - Agent file content.
   * @returns {Promise<object>} Parsed report.
   */
  async function runWithTempAgent(name, content) {
    const tempPath = await createTempAgent(name, content);
    try {
      return await runValidateAgentQuality();
    } finally {
      await rm(tempPath, { force: true });
    }
  }

  it('flags an agent with a missing tier', async () => {
    const report = await runWithTempAgent(
      'zz-quality-missing-tier.agent.md',
      '---\nname: zz-quality-missing-tier\ndescription: test\n---\n## Mission\n## Constraints\n## Default Flow\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 1\nROLE: zz-quality-missing-tier\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nPHASE_COMPLETE: false\nSUB_ORCHESTRATORS_USED:\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-missing-tier.agent.md' &&
          i.message.includes('valid numeric tier'),
      ),
    );
  });

  it('flags an agent with an invalid tier', async () => {
    // normalizeTier only returns 1-4 or null, so tier: 9 is treated as a missing tier.
    const report = await runWithTempAgent(
      'zz-quality-invalid-tier.agent.md',
      '---\nname: zz-quality-invalid-tier\ndescription: test\ntier: 9\n---\n## Mission\n## Constraints\n## Default Flow\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 9\nROLE: zz-quality-invalid-tier\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nPHASE_COMPLETE: false\nSUB_ORCHESTRATORS_USED:\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-invalid-tier.agent.md' &&
          i.message.includes('valid numeric tier'),
      ),
    );
  });

  it('flags an agent missing the Output Format section', async () => {
    const report = await runWithTempAgent(
      'zz-quality-no-output-format.agent.md',
      '---\nname: zz-quality-no-output-format\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\nbody\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-no-output-format.agent.md' &&
          i.message.includes('## Output Format'),
      ),
    );
  });

  it('flags an agent with no structured-v1 fence', async () => {
    const report = await runWithTempAgent(
      'zz-quality-no-fence.agent.md',
      '---\nname: zz-quality-no-fence\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\njust text\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-no-fence.agent.md' &&
          i.message.includes('exactly one fenced'),
      ),
    );
  });

  it('flags an agent with multiple structured-v1 fences', async () => {
    const report = await runWithTempAgent(
      'zz-quality-multi-fence.agent.md',
      '---\nname: zz-quality-multi-fence\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-multi-fence\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-multi-fence\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-multi-fence.agent.md' &&
          i.message.includes('exactly one fenced'),
      ),
    );
  });

  it('flags an agent with content after the structured-v1 fence', async () => {
    const report = await runWithTempAgent(
      'zz-quality-tail-content.agent.md',
      '---\nname: zz-quality-tail-content\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-tail-content\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\ntrailing content\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-tail-content.agent.md' &&
          i.message.includes('final content'),
      ),
    );
  });

  it('flags an agent with the wrong first line inside the fence', async () => {
    const report = await runWithTempAgent(
      'zz-quality-wrong-first.agent.md',
      '---\nname: zz-quality-wrong-first\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-wrong-first\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-wrong-first.agent.md' &&
          i.message.includes('first non-blank line'),
      ),
    );
  });

  it('flags an agent with the wrong field order', async () => {
    const report = await runWithTempAgent(
      'zz-quality-wrong-order.agent.md',
      '---\nname: zz-quality-wrong-order\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nROLE: zz-quality-wrong-order\nTIER: 3\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-wrong-order.agent.md' &&
          i.message.includes('exact order'),
      ),
    );
  });

  it('flags an agent with missing required fields', async () => {
    const report = await runWithTempAgent(
      'zz-quality-missing-fields.agent.md',
      '---\nname: zz-quality-missing-fields\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-missing-fields\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-missing-fields.agent.md' &&
          i.message.includes('missing required field'),
      ),
    );
  });

  it('flags an agent whose structured-v1 TIER does not match frontmatter', async () => {
    const report = await runWithTempAgent(
      'zz-quality-tier-mismatch.agent.md',
      '---\nname: zz-quality-tier-mismatch\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 2\nROLE: zz-quality-tier-mismatch\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-tier-mismatch.agent.md' &&
          i.message.includes('TIER: 3'),
      ),
    );
  });

  it('flags an agent whose structured-v1 ROLE does not match frontmatter name', async () => {
    const report = await runWithTempAgent(
      'zz-quality-role-mismatch.agent.md',
      '---\nname: zz-quality-role-mismatch\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: wrong-role\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path === '.github/agents/zz-quality-role-mismatch.agent.md' &&
          i.message.includes('ROLE: zz-quality-role-mismatch'),
      ),
    );
  });

  it('flags an agent with the wrong OUTPUT_CONTRACT value', async () => {
    const report = await runWithTempAgent(
      'zz-quality-wrong-contract-value.agent.md',
      '---\nname: zz-quality-wrong-contract-value\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: wrong-value\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-wrong-contract-value\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, false);
    assert.ok(
      report.issues.some(
        (i) =>
          i.path ===
            '.github/agents/zz-quality-wrong-contract-value.agent.md' &&
          i.message.includes('OUTPUT_CONTRACT: structured-v1'),
      ),
    );
  });

  it('passes a clean tier-4 agent', async () => {
    const report = await runWithTempAgent(
      'zz-quality-clean-t4.agent.md',
      '---\nname: zz-quality-clean-t4\ndescription: test\ntier: 4\n---\n## Mission\n## Constraints\n## Default Flow\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 4\nROLE: zz-quality-clean-t4\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, true);
    const t4Agent = report.agents.find(
      (a) => a.path === '.github/agents/zz-quality-clean-t4.agent.md',
    );
    assert.ok(t4Agent);
    assert.equal(t4Agent.counts.errors, 0);
  });

  it('covers the name fallback when frontmatter omits name', async () => {
    const report = await runWithTempAgent(
      'zz-quality-no-name.agent.md',
      '---\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-no-name\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    const agent = report.agents.find(
      (a) => a.path === '.github/agents/zz-quality-no-name.agent.md',
    );
    assert.ok(agent);
    assert.equal(agent.name, 'zz-quality-no-name');
  });

  it('covers non-field lines inside the structured-v1 fence', async () => {
    const report = await runWithTempAgent(
      'zz-quality-non-field.agent.md',
      '---\nname: zz-quality-non-field\ndescription: test\ntier: 3\n---\n## Mission\n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-non-field\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nthis line is ignored\nSUMMARY: test\n```\n',
    );
    assert.equal(report.ok, true);
    const agent = report.agents.find(
      (a) => a.path === '.github/agents/zz-quality-non-field.agent.md',
    );
    assert.ok(agent);
    assert.equal(agent.counts.errors, 0);
  });

  it('covers an empty section name in extractSections', async () => {
    const report = await runWithTempAgent(
      'zz-quality-empty-section.agent.md',
      '---\nname: zz-quality-empty-section\ndescription: test\ntier: 3\n---\n## Mission\n##   \n## Constraints\n## Approach\n## If Blocked\n## Output Format\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 3\nROLE: zz-quality-empty-section\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nHANDOFF:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nSUMMARY: test\n```\n',
    );
    const agent = report.agents.find(
      (a) => a.path === '.github/agents/zz-quality-empty-section.agent.md',
    );
    assert.ok(agent);
  });
});
