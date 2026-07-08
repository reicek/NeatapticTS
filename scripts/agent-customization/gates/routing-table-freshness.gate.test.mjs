/**
 * @module routing-table-freshness.gate.test
 * @description Native-ESM coverage tests for routing-table-freshness.gate.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the gate script.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/routing-table-freshness.gate.mjs',
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

describe('routing-table-freshness gate native-ESM coverage', () => {
  it('imports the module and exposes the runner', async () => {
    const gate = await withArgv([process.execPath, 'dummy-runner'], () =>
      import('./routing-table-freshness.gate.mjs'),
    );
    assert.equal(typeof gate.runRoutingTableFreshnessGate, 'function');
    assert.equal(typeof gate.main, 'function');
  });

  it('passes when the routing table is fresh', async () => {
    const { runRoutingTableFreshnessGate } = await withArgv(
      [process.execPath, 'dummy-runner'],
      () => import('./routing-table-freshness.gate.mjs'),
    );
    const result = await runRoutingTableFreshnessGate();
    assert.equal(result.pass, true);
    assert.equal(result.evidence.exists, true);
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
            const { main } = await import('./routing-table-freshness.gate.mjs');
            await main();
          });
        },
      );
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    } finally {
      console.log = originalLog;
      process.exitCode = 0;
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
            const { main } = await import('./routing-table-freshness.gate.mjs');
            await main();
          });
        },
      );
      assert.ok(logs.some((line) => line.includes('PASS')));
      assert.equal(process.exitCode, 0);
    } finally {
      console.log = originalLog;
      process.exitCode = 0;
    }
  });

  it('main reports failure and fix hint when the table is stale', async () => {
    jest.unstable_mockModule('../generate-agent-skill-routing-table.mjs', () => ({
      ROUTING_TABLE_PATH: '.github/agent-skill-routing-table.md',
      collectCustomizationRoutingTable: jest.fn(() =>
        Promise.resolve({
          sourceHash: 'stale-hash',
          sourceFiles: ['a'],
          agentRows: [],
          skillRows: [],
          markdown: 'stale',
        }),
      ),
      extractRoutingTableSourceHash: jest.fn(() => 'current-hash'),
    }));
    const logs = [];
    const originalLog = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    try {
      await withArgv(
        [process.execPath, 'dummy-runner'],
        async () => {
          await importInIsolation(async () => {
            const { main } = await import('./routing-table-freshness.gate.mjs');
            await main();
          });
        },
      );
      assert.equal(process.exitCode, 1);
      assert.ok(logs.some((line) => line.includes('FAIL')));
      assert.ok(logs.some((line) => line.includes('fixHint')));
    } finally {
      console.log = originalLog;
      process.exitCode = 0;
      jest.unstable_mockModule('../generate-agent-skill-routing-table.mjs', () => ({
        ROUTING_TABLE_PATH: '.github/agent-skill-routing-table.md',
        collectCustomizationRoutingTable: jest.fn(),
        extractRoutingTableSourceHash: jest.fn(),
      }));
    }
  });

  it('covers the main entry-point guard when run as the entry module', async () => {
    await withArgv([process.execPath, GATE_PATH], async () => {
      await swallowLogs(() =>
        importInIsolation(async () => {
          await import(GATE_PATH);
        }),
      );
    });
  });

  it('covers the main entry-point error handler', async () => {
    const errors = [];
    const originalStderr = console.error;
    console.error = (...args) => errors.push(args.join(' '));
    try {
      const { handleMainError } = await import('./routing-table-freshness.gate.mjs');
      await handleMainError(new Error('freshness boom'));
      assert.ok(errors.some((line) => line.includes('freshness boom')));
      assert.equal(process.exitCode, 1);
    } finally {
      console.error = originalStderr;
      process.exitCode = 0;
    }
  });

  it('reports failure when the routing table is missing', async () => {
    const utils = await import('../customization-utils.mjs');
    jest.unstable_mockModule('../customization-utils.mjs', () => ({
      ...utils,
      fileExists: jest.fn(() => Promise.resolve(false)),
    }));
    try {
      await withArgv(
        [process.execPath, 'dummy-runner'],
        async () => {
          await importInIsolation(async () => {
            const { runRoutingTableFreshnessGate } = await import(
              './routing-table-freshness.gate.mjs'
            );
            const result = await runRoutingTableFreshnessGate();
            assert.equal(result.pass, false);
            assert.equal(result.evidence.exists, false);
          });
        },
      );
    } finally {
      jest.unstable_mockModule('../customization-utils.mjs', () => utils);
    }
  });

  it('passes via CLI with --json', () => {
    const output = execFileSync(process.execPath, [GATE_PATH, '--json'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    });
    const parsed = JSON.parse(output);
    assert.equal(parsed.pass, true);
  });
});
