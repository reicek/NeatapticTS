/**
 * @module generate-agent-skill-routing-table.gate.test
 * @description Native-ESM coverage tests for generate-agent-skill-routing-table.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the generator.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const SCRIPT_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/generate-agent-skill-routing-table.mjs',
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

describe('generate-agent-skill-routing-table native-ESM coverage', () => {
  it('imports the module and exposes helpers', async () => {
    const {
      collectCustomizationRoutingTable,
      runGenerateCustomizationRoutingTable,
      main,
    } = await withArgv([process.execPath, 'dummy-runner'], () =>
      import('./generate-agent-skill-routing-table.mjs'),
    );
    assert.equal(typeof collectCustomizationRoutingTable, 'function');
    assert.equal(typeof runGenerateCustomizationRoutingTable, 'function');
    assert.equal(typeof main, 'function');
  });

  it('collectCustomizationRoutingTable returns table metadata', async () => {
    const { collectCustomizationRoutingTable } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./generate-agent-skill-routing-table.mjs'),
    );
    const table = await collectCustomizationRoutingTable();
    assert.equal(typeof table.sourceHash, 'string');
    assert.ok(Array.isArray(table.sourceFiles));
    assert.ok(table.sourceFiles.length > 0);
    assert.ok(Array.isArray(table.agentRows));
    assert.ok(Array.isArray(table.skillRows));
  });

  it('runGenerateCustomizationRoutingTable reports ok and unchanged', async () => {
    const { runGenerateCustomizationRoutingTable } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./generate-agent-skill-routing-table.mjs'),
    );
    const result = await runGenerateCustomizationRoutingTable({ write: false });
    assert.equal(result.ok, true);
    assert.equal(result.changed, false);
  });

  it('writes the routing table when changed', async () => {
    const { runGenerateCustomizationRoutingTable } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./generate-agent-skill-routing-table.mjs'),
    );
    const tablePath = path.join(REPO_ROOT, '.github/agent-skill-routing-table.md');
    const original = readFileSync(tablePath, 'utf8');
    writeFileSync(tablePath, `${original}\n<!-- temporary coverage perturbation -->\n`, 'utf8');
    try {
      const result = await runGenerateCustomizationRoutingTable({ write: true });
      assert.equal(result.ok, true);
      assert.equal(result.changed, true);
      const restored = readFileSync(tablePath, 'utf8');
      assert.equal(restored, original);
    } finally {
      writeFileSync(tablePath, original, 'utf8');
    }
  });

  it('extractRoutingTableSourceHash returns the embedded hash or null', async () => {
    const { extractRoutingTableSourceHash } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./generate-agent-skill-routing-table.mjs'),
    );
    assert.equal(
      extractRoutingTableSourceHash('<!-- source-hash: abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789 -->'),
      'abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789',
    );
    assert.equal(extractRoutingTableSourceHash('no hash here'), null);
  });

  it('formatModel formats scalar and array models', async () => {
    const { formatModel } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./generate-agent-skill-routing-table.mjs'),
    );
    assert.equal(formatModel('gpt-4o'), 'gpt-4o');
    assert.equal(formatModel(null), '-');
    assert.equal(formatModel(['a', 'b']), 'a<br>b');
  });

  it('formatList joins items or returns a dash', async () => {
    const { formatList } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./generate-agent-skill-routing-table.mjs'),
    );
    assert.equal(formatList([]), '-');
    assert.equal(formatList(['x', 'y']), 'x<br>y');
  });

  it('createAgentRows fills in tier and createSkillRows handles unused skills', async () => {
    const { createAgentRows, createSkillRows } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./generate-agent-skill-routing-table.mjs'),
    );
    const agents = createAgentRows([
      { name: 'a1', tier: 1, model: ['m1', 'm2'], agents: [], skills: ['s1'] },
      { name: 'a2', model: 'single', agents: ['a1'], skills: [] },
    ]);
    assert.equal(agents[0].tier, 1);
    assert.equal(agents[0].model, 'm1<br>m2');
    assert.equal(agents[1].tier, '-');
    assert.equal(agents[1].skills, '-');

    const skills = createSkillRows(
      [{ name: 's1' }, { name: 's2' }],
      [
        { name: 'a1', skills: ['s1'] },
        { name: 'a2', skills: [] },
      ],
    );
    assert.equal(skills[0].name, 's1');
    assert.equal(skills[0].agents, 'a1');
    assert.equal(skills[1].agents, '-');
  });

  it('writes the routing table when it does not exist', async () => {
    const utils = await import('./customization-utils.mjs');
    jest.unstable_mockModule('./customization-utils.mjs', () => ({
      ...utils,
      fileExists: jest.fn(() => Promise.resolve(false)),
    }));
    const tablePath = path.join(REPO_ROOT, '.github/agent-skill-routing-table.md');
    const original = readFileSync(tablePath, 'utf8');
    try {
      await withArgv(
        [process.execPath, 'dummy-runner'],
        async () => {
          await importInIsolation(async () => {
            const { runGenerateCustomizationRoutingTable } = await import(
              './generate-agent-skill-routing-table.mjs'
            );
            const result = await runGenerateCustomizationRoutingTable({ write: true });
            assert.equal(result.ok, true);
            assert.equal(result.changed, true);
          });
        },
      );
      const written = readFileSync(tablePath, 'utf8');
      assert.ok(written.includes('# Canonical Agent and Skill Routing Table'));
    } finally {
      writeFileSync(tablePath, original, 'utf8');
      jest.unstable_mockModule('./customization-utils.mjs', () => utils);
    }
  });

  it('main prints JSON output', async () => {
    const logs = [];
    const originalLog = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    try {
      await withArgv(
        [process.execPath, 'dummy-runner', '--json'],
        async () => {
          await importInIsolation(async () => {
            const { main } = await import('./generate-agent-skill-routing-table.mjs');
            await main();
          });
        },
      );
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
      await withArgv(
        [process.execPath, 'dummy-runner'],
        async () => {
          await importInIsolation(async () => {
            const { main } = await import('./generate-agent-skill-routing-table.mjs');
            await main();
          });
        },
      );
      assert.ok(logs.some((line) => line.includes('PASS customization routing table')));
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
      const { handleMainError } = await import('./generate-agent-skill-routing-table.mjs');
      await handleMainError(new Error('boom'));
      assert.ok(errors.some((line) => line.includes('boom')));
      assert.equal(process.exitCode, 1);
    } finally {
      console.error = originalStderr;
      process.exitCode = 0;
    }
  });

  it('passes via CLI with --json', () => {
    const output = execFileSync(process.execPath, [SCRIPT_PATH, '--json'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    });
    const parsed = JSON.parse(output);
    assert.equal(parsed.ok, true);
    assert.equal(parsed.changed, false);
  });
});
