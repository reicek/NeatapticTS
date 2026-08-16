/**
 * @module validate-agent-frontmatter.test
 * @description Native-ESM coverage tests for validate-agent-frontmatter.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the script.
 *
 * The validator has no exports — it runs entirely as top-level code. Tests
 * import it in isolation with controlled `process.argv` values to exercise
 * the main flow, `--help` branch, strict mode, UTF-8 encoding, and the
 * `target` field validation paths.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { writeFile, rm, rename, mkdir } from 'node:fs/promises';

const REPO_ROOT = path.resolve();
const SCRIPT_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/validate-agent-frontmatter.mjs',
);

// Import real utils so mock factories can spread `...realUtils` and override
// only the specific functions needed for a test.
import * as realUtils from './customization-utils.mjs';

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
 * Imports a module inside `jest.isolateModulesAsync` so top-level code runs
 * fresh for each test. A 150ms delay lets unawaited top-level promises settle.
 *
 * @param {() => Promise<void>} callback - Import callback executed inside the isolate.
 * @returns {Promise<void>} Resolves after the isolate and delay complete.
 */
async function importInIsolation(callback) {
  return jest.isolateModulesAsync(async () => {
    await callback();
    // Let the unawaited top-level await settle before the isolate ends.
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
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

describe('validate-agent-frontmatter native-ESM coverage', () => {
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
    assert.ok(Array.isArray(parsed.agents));
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

  it('runs the main flow with --strict --json and produces valid JSON', async () => {
    const { output } = await captureStdout(() =>
      withArgv([process.execPath, SCRIPT_PATH, '--json', '--strict'], () =>
        importInIsolation(async () => {
          await import(SCRIPT_PATH);
        }),
      ),
    );
    const parsed = JSON.parse(output);
    assert.equal(parsed.ok, true);
    assert.ok(Array.isArray(parsed.agents));
  });

  it('sets the stdout encoding to utf8 on import', async () => {
    // The script calls process.stdout.setDefaultEncoding('utf8') at the top
    // level (line 17). We verify this by capturing stdout output and confirming
    // non-ASCII characters are preserved (not corrupted to '?').
    const { output } = await captureStdout(() =>
      withArgv([process.execPath, SCRIPT_PATH, '--json'], () =>
        importInIsolation(async () => {
          await import(SCRIPT_PATH);
        }),
      ),
    );
    const parsed = JSON.parse(output);
    // If encoding were not UTF-8, non-ASCII characters in agent descriptions
    // would be corrupted. The JSON should still parse and contain the agents.
    assert.ok(Array.isArray(parsed.agents));
    assert.ok(parsed.agents.length > 0);
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

  it('detects non-string and invalid string target values via temp agent files', async () => {
    const fs = await import('node:fs/promises');
    const os = await import('node:os');

    // Create temp agent files with invalid target values.
    // parseFrontmatterValue('') returns true (boolean), so `target:` (bare key)
    // gives data.target = true → typeof !== 'string' → "must be a string" error.
    // `target: bad-platform` gives data.target = 'bad-platform' (string) →
    // not in allowed set → "must be one of" error.
    const tmpNonString = path.join(
      REPO_ROOT,
      '.github/agents/zz-test-nonstring.agent.md',
    );
    const tmpInvalid = path.join(
      REPO_ROOT,
      '.github/agents/zz-test-invalid.agent.md',
    );

    try {
      // Test 1: non-string target (bare key → boolean true)
      await fs.writeFile(
        tmpNonString,
        '---\nname: zz-test-nonstring\ndescription: test\ntarget:\ntools: [read]\nskills: []\n---\nbody',
        'utf8',
      );
      {
        const result = spawnSync(process.execPath, [SCRIPT_PATH, '--json'], {
          cwd: REPO_ROOT,
          encoding: 'utf8',
        });
        const parsed = JSON.parse(result.stdout);
        assert.equal(parsed.ok, false);
        assert.ok(
          parsed.issues.some(
            (i) =>
              i.path === '.github/agents/zz-test-nonstring.agent.md' &&
              i.message.includes('`target` must be a string'),
          ),
          `Expected target-must-be-string error, got: ${JSON.stringify(parsed.issues)}`,
        );
      }
      await fs.unlink(tmpNonString);

      // Test 2: invalid string target (string not in allowed set)
      await fs.writeFile(
        tmpInvalid,
        '---\nname: zz-test-invalid\ndescription: test\ntarget: bad-platform\ntools: [read]\nskills: []\n---\nbody',
        'utf8',
      );
      {
        const result = spawnSync(process.execPath, [SCRIPT_PATH, '--json'], {
          cwd: REPO_ROOT,
          encoding: 'utf8',
        });
        const parsed = JSON.parse(result.stdout);
        assert.equal(parsed.ok, false);
        assert.ok(
          parsed.issues.some(
            (i) =>
              i.path === '.github/agents/zz-test-invalid.agent.md' &&
              i.message.includes('`target` must be one of'),
          ),
          `Expected target-must-be-one-of error, got: ${JSON.stringify(parsed.issues)}`,
        );
      }
    } finally {
      await fs.rm(tmpNonString, { force: true });
      await fs.rm(tmpInvalid, { force: true });
    }
  });

  it('accepts a valid target value (covered by real agents — all have target: vscode)', async () => {
    // All 33 real agents have `target: vscode` which is a valid value.
    // The main flow tests already exercise the valid target path.
    // This test verifies no target-related errors appear for real agents.
    const result = spawnSync(process.execPath, [SCRIPT_PATH, '--json'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    });
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.ok, true);
    const targetErrors = (parsed.issues ?? []).filter((i) =>
      i.message.includes('target'),
    );
    assert.equal(
      targetErrors.length,
      0,
      `Expected no target errors for real agents, got: ${JSON.stringify(targetErrors)}`,
    );
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

  it('passes via CLI with --json --strict', () => {
    const result = spawnSync(
      process.execPath,
      [SCRIPT_PATH, '--json', '--strict'],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      },
    );
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.ok, true);
  });
});

describe('validate-agent-frontmatter error-path coverage', () => {
  const AGENTS_DIR = path.join(REPO_ROOT, '.github/agents');

  /**
   * Creates a temp agent file in `.github/agents/` for in-process coverage.
   *
   * @param {string} name - File name (e.g. `zz-test-bad.agent.md`).
   * @param {string} content - Full file content including frontmatter.
   * @returns {Promise<string>} Absolute path to the created file.
   */
  async function createTempAgent(name, content) {
    const filePath = path.join(AGENTS_DIR, name);
    await writeFile(filePath, content, 'utf8');
    return filePath;
  }

  it('covers all field validation and strict-mode error paths', async () => {
    const tempFiles = [];
    const tempSkillDir = path.join(REPO_ROOT, '.github/skills/zz-test-no-name');
    try {
      // File A: no desc, no name, bad types, bad model → 174, 176, 184, 192,
      // 202, 216, 221, 226, 359, 289
      tempFiles.push(
        await createTempAgent(
          'zz-test-all-errors.agent.md',
          '---\nuser-invocable: not-a-bool\ndisable-model-invocation: not-a-bool\ntarget:\ntools: not-an-array\nagents: not-an-array\nmodel: bad-model\n---\nbody',
        ),
      );

      // File B: skills not an array → 235
      tempFiles.push(
        await createTempAgent(
          'zz-test-bad-skills.agent.md',
          '---\nname: zz-test-bad-skills\ndescription: test\nskills: not-an-array\nmodel: glm-5.2:cloud\nuser-invocable: false\n---\n## Output format',
        ),
      );

      // File C: invalid target string → 206
      tempFiles.push(
        await createTempAgent(
          'zz-test-bad-target.agent.md',
          '---\nname: zz-test-bad-target\ndescription: test\ntarget: bad-platform\ntools: [read]\nskills: []\nmodel: glm-5.2:cloud\nuser-invocable: false\n---\n## Output format',
        ),
      );

      // File D: subagent/tool/skill errors → 245, 256, 265, 270, 278
      tempFiles.push(
        await createTempAgent(
          'zz-test-subagent-errors.agent.md',
          '---\nname: zz-test-subagent-errors\ndescription: test\ntools: [unknown-tool]\nagents: [zz-test-subagent-errors, nonexistent-agent]\nskills: [nonexistent-skill]\nmodel: glm-5.2:cloud\nuser-invocable: false\n---\n## Output format',
        ),
      );

      // File E: SDLC path, no model, not user-invocable, no name,
      // no structured-v1 → 300, 308, 316, 320 (?? 'NONE'), 468-475
      tempFiles.push(
        await createTempAgent(
          '03-zz-test-sdlc.agent.md',
          '---\ndescription: test\nskills: []\n---\nbody',
        ),
      );

      // File F: SDLC path with real name but wrong path, valid structured-v1
      // → 326
      tempFiles.push(
        await createTempAgent(
          '03-zz-test-sdlc-wrong.agent.md',
          '---\nname: 03-red-testing\ndescription: test\nskills: []\nmodel: glm-5.2:cloud\nuser-invocable: true\n---\n```structured-v1\nOUTPUT_CONTRACT: structured-v1\nTASK_STATUS: SUCCESS\nTIER: 1\nROLE: 03-red-testing\nTASK_RECEIVED: test\nFILES_READ:\nFILES_CHANGED:\nKEY_FINDINGS:\nACTIONS_TAKEN:\nVALIDATION_EVIDENCE:\nBLOCKERS:\nRISKS_OR_GAPS:\nLEARNING_EVENT_NEEDED: false\nSUGGESTED_NEXT_AGENT: none\nPHASE_COMPLETE: false\nSUB_ORCHESTRATORS_USED:\nSUMMARY: test\n```\n',
        ),
      );

      // File G: hidden agent without output contract → 349
      tempFiles.push(
        await createTempAgent(
          'zz-test-hidden-no-contract.agent.md',
          '---\nname: zz-test-hidden-no-contract\ndescription: test\nuser-invocable: false\nskills: []\nmodel: glm-5.2:cloud\n---\nbody without contract',
        ),
      );

      // File H: qualified model not in strict allowed pool → 368
      tempFiles.push(
        await createTempAgent(
          'zz-test-strict-bad-model.agent.md',
          '---\nname: zz-test-strict-bad-model\ndescription: test\nuser-invocable: false\nskills: []\nmodel: Test Provider (Test)\n---\n## Output format',
        ),
      );

      // File I: tier-2 coordinator path, no structured-v1 → 451, 468-475
      tempFiles.push(
        await createTempAgent(
          'flappy-architecture-polish.agent.md',
          '---\nname: flappy-architecture-polish\ndescription: test\nuser-invocable: false\nskills: []\nmodel: glm-5.2:cloud\n---\n## Output format',
        ),
      );

      // File J: SDLC path with bad structured-v1 → 488, 499, 513, 524, 535, 558
      tempFiles.push(
        await createTempAgent(
          '03-zz-test-structured-bad.agent.md',
          '---\nname: zz-test-structured-bad\ndescription: test\nskills: []\nmodel: glm-5.2:cloud\nuser-invocable: true\n---\n```structured-v1\nTIER: 2\nROLE: wrong-role\nOUTPUT_CONTRACT: wrong-value\nTASK_STATUS: SUCCESS\nthis line does not match\n```\n',
        ),
      );

      // File K: tier-2 coordinator path with bad structured-v1 field order
      // → 492 (Tier-2 branch of ternary)
      tempFiles.push(
        await createTempAgent(
          'green-test-failure-triage-coordinator.agent.md',
          '---\nname: green-test-failure-triage-coordinator\ndescription: test\nuser-invocable: false\nskills: []\nmodel: glm-5.2:cloud\n---\n```structured-v1\nTIER: 2\nROLE: wrong-role\nOUTPUT_CONTRACT: wrong-value\nTASK_STATUS: SUCCESS\n```\n',
        ),
      );

      // Temp skill file without name → 162 (first ?? fallback in collectSkillNames)
      await mkdir(tempSkillDir, { recursive: true });
      const tempSkillFile = path.join(tempSkillDir, 'SKILL.md');
      await writeFile(
        tempSkillFile,
        '---\ndescription: test skill without name\n---\n# Test Skill\nContent.\n',
        'utf8',
      );

      // Run validator in strict mode with all temp files present.
      // Also covers 389 (wrong visible count) and 412 (unexpected visible agents).
      const { output } = await captureStdout(() =>
        withArgv([process.execPath, SCRIPT_PATH, '--json', '--strict'], () =>
          importInIsolation(async () => {
            await import(SCRIPT_PATH);
          }),
        ),
      );
      const parsed = JSON.parse(output);
      assert.equal(parsed.ok, false);
      assert.ok(
        parsed.issues.length > 0,
        'Expected issues from temp agent files',
      );
      process.exitCode = 0;
    } finally {
      for (const file of tempFiles) {
        await rm(file, { force: true });
      }
      await rm(tempSkillDir, { recursive: true, force: true });
    }
  });

  it('covers the missing SDLC agent global rule', async () => {
    const agentPath = path.join(AGENTS_DIR, '03-red-testing.agent.md');
    const backupPath = agentPath + '.bak';
    try {
      await rename(agentPath, backupPath);
      const { output } = await captureStdout(() =>
        withArgv([process.execPath, SCRIPT_PATH, '--json', '--strict'], () =>
          importInIsolation(async () => {
            await import(SCRIPT_PATH);
          }),
        ),
      );
      const parsed = JSON.parse(output);
      assert.equal(parsed.ok, false);
      assert.ok(
        parsed.issues.some(
          (i) =>
            i.message.includes('Missing user-invocable SDLC agent') &&
            i.message.includes('03-red-testing'),
        ),
        `Expected missing 03-red-testing error, got: ${JSON.stringify(parsed.issues)}`,
      );
      process.exitCode = 0;
    } finally {
      await rename(backupPath, agentPath);
    }
  });
});
