/**
 * @module inventory-customizations.gate.test
 * @description Native-ESM coverage tests for inventory-customizations.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the script.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const SCRIPT_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/inventory-customizations.mjs',
);

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

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

async function importInIsolation(callback) {
  return jest.isolateModulesAsync(async () => {
    await callback();
    // Let the unawaited top-level main() promise settle before the isolate ends.
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
}

describe('inventory-customizations native-ESM coverage', () => {
  it('imports the module and exposes the runner', async () => {
    const mod = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./inventory-customizations.mjs'),
    );
    assert.equal(typeof mod.runCustomizationInventory, 'function');
    assert.equal(typeof mod.main, 'function');
  });

  it('runCustomizationInventory returns ok inventory', async () => {
    const { runCustomizationInventory } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./inventory-customizations.mjs'),
    );
    const result = await runCustomizationInventory();
    assert.equal(result.ok, true);
    assert.ok(result.summary.agents >= 1);
    assert.ok(result.summary.skills >= 1);
  });

  it('main prints JSON output', async () => {
    const logs = [];
    const originalLog = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    try {
      await withArgv([process.execPath, 'dummy-runner', '--json'], async () => {
        await importInIsolation(async () => {
          const { main } = await import('./inventory-customizations.mjs');
          await main();
        });
      });
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.ok, true);
    } finally {
      console.log = originalLog;
    }
  });

  it('main prints human-readable output', async () => {
    const logs = [];
    const originalLog = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    try {
      await withArgv([process.execPath, 'dummy-runner'], async () => {
        await importInIsolation(async () => {
          const { main } = await import('./inventory-customizations.mjs');
          await main();
        });
      });
      assert.ok(
        logs.some((line) => line.includes('PASS customization inventory')),
      );
    } finally {
      console.log = originalLog;
    }
  });

  it('covers the top-level help branch when run as the entry module', async () => {
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

  it('covers the main entry-point guard when run as the entry module', async () => {
    await withArgv([process.execPath, SCRIPT_PATH], async () => {
      await swallowLogs(() =>
        importInIsolation(async () => {
          await import(SCRIPT_PATH);
        }),
      );
    });
  });

  it('covers the main entry-point error handler', async () => {
    const errors = [];
    const originalStderr = console.error;
    console.error = (...args) => errors.push(args.join(' '));
    try {
      const { handleMainError } =
        await import('./inventory-customizations.mjs');
      await handleMainError(new Error('inventory boom'));
      assert.ok(errors.some((line) => line.includes('inventory boom')));
      assert.equal(process.exitCode, 1);
    } finally {
      console.error = originalStderr;
      process.exitCode = 0;
    }
  });

  it('readAgent and readSkill fill in defaults and preserve provided fields', async () => {
    const utils = await import('./customization-utils.mjs');
    const minimalAgent = '---\n---\nbody\n';
    const fullAgent =
      '---\nname: Foo\ndescription: desc\ntier: 1\ntools: [t]\nagents: [a]\nskills: [s]\nmodel: m\nuser-invocable: false\ndisable-model-invocation: true\nhandoffs:\n  - x\n---\nbody\n';
    const minimalSkill = '---\n---\nbody\n';
    const fullSkill =
      '---\nname: Bar\ndescription: d\nargument-hint: arg\nuser-invocable: false\ndisable-model-invocation: true\ncontext: c\nlicense: l\n---\nline1\nline2\n';
    jest.unstable_mockModule('./customization-utils.mjs', () => ({
      ...utils,
      readWorkspaceFile: jest.fn((relativePath) => {
        if (relativePath.includes('minimal.agent.md')) {
          return Promise.resolve(minimalAgent);
        }
        if (relativePath.includes('full.agent.md')) {
          return Promise.resolve(fullAgent);
        }
        if (relativePath.includes('minimal/SKILL.md')) {
          return Promise.resolve(minimalSkill);
        }
        if (relativePath.includes('full/SKILL.md')) {
          return Promise.resolve(fullSkill);
        }
        return utils.readWorkspaceFile(relativePath);
      }),
    }));
    try {
      await withArgv([process.execPath, 'dummy-runner'], async () => {
        await importInIsolation(async () => {
          const { readAgent, readSkill } =
            await import('./inventory-customizations.mjs');

          const minimalAgentResult = await readAgent('agents/minimal.agent.md');
          assert.equal(minimalAgentResult.name, 'minimal');
          assert.equal(minimalAgentResult.description, '');
          assert.equal(minimalAgentResult.tier, null);
          assert.deepEqual(minimalAgentResult.tools, []);
          assert.deepEqual(minimalAgentResult.agents, []);
          assert.deepEqual(minimalAgentResult.skills, []);
          assert.equal(minimalAgentResult.model, null);
          assert.equal(minimalAgentResult.handoffs, false);
          assert.equal(minimalAgentResult.userInvocable, true);
          assert.equal(minimalAgentResult.disableModelInvocation, false);

          const fullAgentResult = await readAgent('agents/full.agent.md');
          assert.equal(fullAgentResult.name, 'Foo');
          assert.equal(fullAgentResult.description, 'desc');
          assert.equal(fullAgentResult.tier, '1');
          assert.deepEqual(fullAgentResult.tools, ['t']);
          assert.deepEqual(fullAgentResult.agents, ['a']);
          assert.deepEqual(fullAgentResult.skills, ['s']);
          assert.equal(fullAgentResult.model, 'm');
          assert.equal(fullAgentResult.handoffs, true);
          assert.equal(fullAgentResult.userInvocable, false);
          assert.equal(fullAgentResult.disableModelInvocation, true);

          const minimalSkillResult = await readSkill('skills/minimal/SKILL.md');
          assert.equal(minimalSkillResult.name, '');
          assert.equal(minimalSkillResult.description, '');
          assert.equal(minimalSkillResult.argumentHint, null);
          assert.equal(minimalSkillResult.userInvocable, true);
          assert.equal(minimalSkillResult.disableModelInvocation, false);
          assert.equal(minimalSkillResult.context, null);
          assert.equal(minimalSkillResult.license, null);

          const fullSkillResult = await readSkill('skills/full/SKILL.md');
          assert.equal(fullSkillResult.name, 'Bar');
          assert.equal(fullSkillResult.description, 'd');
          assert.equal(fullSkillResult.argumentHint, 'arg');
          assert.equal(fullSkillResult.userInvocable, false);
          assert.equal(fullSkillResult.disableModelInvocation, true);
          assert.equal(fullSkillResult.context, 'c');
          assert.equal(fullSkillResult.license, 'l');
          assert.equal(fullSkillResult.bodyLines, 3);
        });
      });
    } finally {
      jest.unstable_mockModule('./customization-utils.mjs', () => utils);
    }
  });

  it('passes via CLI with --json', () => {
    const output = execFileSync(process.execPath, [SCRIPT_PATH, '--json'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    });
    const parsed = JSON.parse(output);
    assert.equal(parsed.ok, true);
  });
});
