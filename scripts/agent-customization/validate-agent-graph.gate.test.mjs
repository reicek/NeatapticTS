/**
 * @module validate-agent-graph.gate.test
 * @description Native-ESM coverage tests for validate-agent-graph.mjs.
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
  'scripts/agent-customization/validate-agent-graph.mjs',
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

describe('validate-agent-graph native-ESM coverage', () => {
  it('imports the module and re-exports runValidateAgentGraph', async () => {
    const mod = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./validate-agent-graph.mjs'),
    );
    assert.equal(typeof mod.runValidateAgentGraph, 'function');
    assert.equal(typeof mod.main, 'function');
  });

  it('runValidateAgentGraph returns ok', async () => {
    const { runValidateAgentGraph } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./validate-agent-graph.mjs'),
    );
    const result = await runValidateAgentGraph();
    assert.equal(result.ok, true);
  });

  it('main prints JSON output', async () => {
    const logs = [];
    const originalLog = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    try {
      await withArgv([process.execPath, 'dummy-runner', '--json'], async () => {
        await importInIsolation(async () => {
          const { main } = await import('./validate-agent-graph.mjs');
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
          const { main } = await import('./validate-agent-graph.mjs');
          await main();
        });
      });
      assert.ok(logs.some((line) => line.includes('PASS')));
      assert.equal(process.exitCode, 0);
    } finally {
      console.log = originalLog;
      process.exitCode = 0;
    }
  });

  it('main sets non-zero exit code when validation fails', async () => {
    jest.unstable_mockModule('./tier-graph-utils.mjs', () => ({
      collectTierInventory: jest.fn(() => Promise.resolve({})),
      runValidateAgentGraph: jest.fn(() =>
        Promise.resolve({
          name: 'agent graph',
          ok: false,
          issues: [{ severity: 'high', path: 'x', message: 'bad' }],
        }),
      ),
    }));
    const logs = [];
    const originalLog = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    try {
      await withArgv([process.execPath, 'dummy-runner'], async () => {
        await importInIsolation(async () => {
          const { main } = await import('./validate-agent-graph.mjs');
          await main();
        });
      });
      assert.equal(process.exitCode, 1);
      assert.ok(logs.some((line) => line.includes('FAIL')));
    } finally {
      console.log = originalLog;
      process.exitCode = 0;
      jest.unstable_mockModule('./tier-graph-utils.mjs', () => ({
        collectTierInventory: jest.fn(),
        runValidateAgentGraph: jest.fn(),
      }));
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

  it('covers main help branch without exiting the process', async () => {
    await withExitStub((stub) =>
      withArgv([process.execPath, 'dummy-runner', '--help'], async () => {
        const { main } = await import('./validate-agent-graph.mjs');
        try {
          await main();
          assert.fail('expected main help to exit');
        } catch (error) {
          assert.ok(stub.called());
        }
      }),
    );
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
      const { handleMainError } = await import('./validate-agent-graph.mjs');
      await handleMainError(new Error('graph boom'));
      assert.ok(errors.some((line) => line.includes('graph boom')));
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
  });
});
