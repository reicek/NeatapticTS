/**
 * @module cortex-self-heal.test
 * @description Red-phase unit tests for the Cortex self-heal orchestrator CLI
 * seams: lock lifecycle with pid-liveness reclaim, the atomic write-then-rename
 * state persistence, the classifier-first dry-run command lists (never
 * --force), and the runRepair coordination flow.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { spawn, spawnSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';

// Capture the REAL fs namespaces before mock registration so test fixtures
// themselves stay uncounted; only the module under test observes the counters.
const fsReal = await import('node:fs');
const fsPromisesReal = await import('node:fs/promises');

const fsCounts = { total: 0, rename: 0 };

function countedNamespace(namespace) {
  return new Proxy(namespace, {
    get(target, property) {
      if (property === 'default') {
        const defaultValue = target[property];
        if (defaultValue && typeof defaultValue === 'object') {
          return countedNamespace(defaultValue);
        }
        return defaultValue;
      }
      const value = target[property];
      if (typeof value === 'function') {
        return (...args) => {
          fsCounts.total += 1;
          if (property === 'rename' || property === 'renameSync') {
            fsCounts.rename += 1;
          }
          return value(...args);
        };
      }
      if (value && typeof value === 'object') {
        return countedNamespace(value);
      }
      return value;
    },
  });
}

jest.unstable_mockModule('node:fs', () => countedNamespace(fsReal));
jest.unstable_mockModule('node:fs/promises', () =>
  countedNamespace(fsPromisesReal),
);

const {
  runRepair,
  createLock,
  refreshLock,
  releaseLock,
  reclaimStaleLock,
  dryRunReport,
} = await import('./cortex-self-heal.mjs');

const BASE_NOW = 1_000_000_000_000;

function makeTmpDir() {
  return mkdtempSync(path.join(os.tmpdir(), 'cortex-self-heal-'));
}

function rmTmp(dir) {
  try {
    rmSync(dir, { recursive: true, force: true });
  } catch {
    // ignore cleanup failures
  }
}

function lockPathIn(dir) {
  return path.join(dir, 'cortex-self-heal.repair.lock');
}

function statePathIn(dir) {
  return path.join(dir, 'cortex-self-heal-state.json');
}

function pidAlive(pid) {
  const probe = spawnSync(process.execPath, ['-e', ''], {
    timeout: 15_000,
    windowsHide: true,
  });
  if (pid === probe.pid) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === 'EPERM';
  }
}

function deadPid() {
  const probe = spawnSync(process.execPath, ['-e', ''], {
    timeout: 15_000,
    windowsHide: true,
  });
  return probe.pid + 10_000;
}

describe('dryRunReport — classifier-first command lists (AC-004)', () => {
  it('lists the full classifier-first repair sequence without any --force flag', () => {
    const commands = dryRunReport({
      state: { last_probe: { state: 'model-only' } },
    });

    assert.deepStrictEqual(commands, [
      'node rag-index/validate-index.mjs --json',
      'node rag-index/build-index.mjs --json',
      'npm run index:build-snapshot',
      'npm run index:prewarm',
      'node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json',
      'node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json',
    ]);
    for (const command of commands) {
      assert.ok(
        !command.includes('--force'),
        `unexpected --force in: ${command}`,
      );
    }
  });

  it('skips the snapshot and prewarm steps in the warm state', () => {
    const commands = dryRunReport({
      state: { last_probe: { state: 'warm' } },
    });

    assert.deepStrictEqual(commands, [
      'node rag-index/validate-index.mjs --json',
      'node rag-index/build-index.mjs --json',
      'node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json',
      'node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json',
    ]);
  });
});

describe('createLock — exclusive lock creation (AC-003)', () => {
  it('creates the lock file with the plan record schema', () => {
    const tmpDir = makeTmpDir();
    try {
      const pid = spawnSync(process.execPath, ['-e', '']).pid;
      const lock = createLock({
        stateDir: tmpDir,
        pid,
        now: () => BASE_NOW,
        attempt: 1,
        reason: 'model-only',
      });

      assert.deepStrictEqual(lock, {
        pid,
        started_at: BASE_NOW,
        heartbeat_at: BASE_NOW,
        attempt: 1,
        reason: 'model-only',
      });
      const onDisk = JSON.parse(readFileSync(lockPathIn(tmpDir), 'utf8'));
      assert.strictEqual(onDisk.pid, pid);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('returns null when another holder still owns the lock', () => {
    const tmpDir = makeTmpDir();
    try {
      const pid = spawnSync(process.execPath, ['-e', '']).pid;
      createLock({
        stateDir: tmpDir,
        pid,
        now: () => BASE_NOW,
        attempt: 1,
        reason: 'model-only',
      });

      const second = createLock({
        stateDir: tmpDir,
        pid,
        now: () => BASE_NOW + 1_000,
        attempt: 2,
        reason: 'model-only',
      });

      assert.strictEqual(second, null);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('refreshLock / releaseLock — lock lifecycle (AC-003)', () => {
  it('bumps only the heartbeat timestamp on refresh', () => {
    const tmpDir = makeTmpDir();
    try {
      const pid = spawnSync(process.execPath, ['-e', '']).pid;
      createLock({
        stateDir: tmpDir,
        pid,
        now: () => BASE_NOW,
        attempt: 1,
        reason: 'model-only',
      });

      const refreshed = refreshLock({
        stateDir: tmpDir,
        now: () => BASE_NOW + 30_000,
      });

      assert.strictEqual(refreshed.heartbeat_at, BASE_NOW + 30_000);
      assert.strictEqual(refreshed.started_at, BASE_NOW);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('removes the lock file on release', () => {
    const tmpDir = makeTmpDir();
    try {
      const pid = spawnSync(process.execPath, ['-e', '']).pid;
      createLock({
        stateDir: tmpDir,
        pid,
        now: () => BASE_NOW,
        attempt: 1,
        reason: 'model-only',
      });

      releaseLock({ stateDir: tmpDir, pid });

      assert.strictEqual(existsSync(lockPathIn(tmpDir)), false);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('refuses to release a lock held by a foreign pid', () => {
    const tmpDir = makeTmpDir();
    try {
      const pid = spawnSync(process.execPath, ['-e', '']).pid;
      createLock({
        stateDir: tmpDir,
        pid,
        now: () => BASE_NOW,
        attempt: 1,
        reason: 'model-only',
      });

      releaseLock({ stateDir: tmpDir, pid: pid + 1 });

      assert.strictEqual(existsSync(lockPathIn(tmpDir)), true);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('reclaimStaleLock — stale-lock takeover (AC-003, AC-006)', () => {
  it('reclaims a heartbeat-stale lock by age', () => {
    const tmpDir = makeTmpDir();
    try {
      const pid = spawnSync(process.execPath, ['-e', '']).pid;
      createLock({
        stateDir: tmpDir,
        pid,
        now: () => BASE_NOW,
        attempt: 1,
        reason: 'model-only',
      });

      const reclaimed = reclaimStaleLock({
        stateDir: tmpDir,
        now: () => BASE_NOW + 2_700_000,
        pid,
      });

      assert.notStrictEqual(reclaimed, null);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('reclaims a dead-pid lock regardless of heartbeat freshness', () => {
    const tmpDir = makeTmpDir();
    try {
      const dead = deadPid();
      fsRealWriteLock(tmpDir, {
        pid: dead,
        started_at: BASE_NOW - 1_000,
        heartbeat_at: BASE_NOW - 1_000,
        attempt: 1,
        reason: 'model-only',
      });

      const reclaimed = reclaimStaleLock({
        stateDir: tmpDir,
        now: () => BASE_NOW,
        pid: process.pid,
      });

      assert.notStrictEqual(reclaimed, null);
      assert.strictEqual(reclaimed.pid, process.pid);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('does not reclaim a lock held by a live pid with a fresh heartbeat', () => {
    const tmpDir = makeTmpDir();
    // A genuinely-live holder: the child idles on an interval until it is
    // killed in `finally` (spawnSync would block until exit, leaving a
    // dead-pid fixture that must be reclaimed). Declared before `try` so the
    // `finally` block can reach the handle for guaranteed cleanup.
    const holderChild = spawn(
      process.execPath,
      ['-e', 'setInterval(() => {}, 1_000)'],
      { stdio: 'ignore', windowsHide: true },
    );
    try {
      const holder = holderChild.pid;
      fsRealWriteLock(tmpDir, {
        pid: holder,
        started_at: BASE_NOW - 1_000,
        heartbeat_at: BASE_NOW - 1_000,
        attempt: 1,
        reason: 'model-only',
      });

      const reclaimed = reclaimStaleLock({
        stateDir: tmpDir,
        now: () => BASE_NOW,
        pid: process.pid,
      });

      assert.strictEqual(reclaimed, null);
    } finally {
      holderChild.kill();
      rmTmp(tmpDir);
    }
  });

  it('reclaims at most once per caller — a second reclaim finds a fresh holder', () => {
    const tmpDir = makeTmpDir();
    try {
      const dead = deadPid();
      fsRealWriteLock(tmpDir, {
        pid: dead,
        started_at: BASE_NOW - 3_600_000,
        heartbeat_at: BASE_NOW - 3_600_000,
        attempt: 1,
        reason: 'model-only',
      });

      const first = reclaimStaleLock({
        stateDir: tmpDir,
        now: () => BASE_NOW,
        pid: process.pid,
      });
      const second = reclaimStaleLock({
        stateDir: tmpDir,
        now: () => BASE_NOW + 1_000,
        pid: process.pid + 1,
      });

      assert.notStrictEqual(first, null);
      assert.strictEqual(second, null);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('runRepair — coordination flow (AC-005, AC-008)', () => {
  it('executes the classifier-first sequence under the lock and releases it on success', async () => {
    const tmpDir = makeTmpDir();
    try {
      const executed = [];
      const spawner = async (command) => {
        executed.push(command);
        // The lock must exist for the whole repair window.
        if (!existsSync(lockPathIn(tmpDir))) {
          throw new Error('lock missing during repair');
        }
        return { code: 0 };
      };

      const result = await runRepair({
        stateDir: tmpDir,
        now: () => BASE_NOW,
        spawner,
        state: { last_probe: { state: 'model-only' } },
      });

      assert.strictEqual(result.success, true);
      assert.strictEqual(executed.length, 6);
      assert.strictEqual(existsSync(lockPathIn(tmpDir)), false);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('releases the lock and records failure when a command fails', async () => {
    const tmpDir = makeTmpDir();
    try {
      const spawner = async (command) => {
        if (command.includes('prewarm')) {
          return { code: 1, stderr: 'prewarm timeout' };
        }
        return { code: 0 };
      };

      const result = await runRepair({
        stateDir: tmpDir,
        now: () => BASE_NOW,
        spawner,
        state: { last_probe: { state: 'model-only' } },
      });

      assert.strictEqual(result.success, false);
      assert.strictEqual(existsSync(lockPathIn(tmpDir)), false);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('appends the repair outcome to the state history', async () => {
    const tmpDir = makeTmpDir();
    try {
      await runRepair({
        stateDir: tmpDir,
        now: () => BASE_NOW,
        spawner: async () => ({ code: 0 }),
        state: { last_probe: { state: 'model-only' } },
      });

      const onDisk = JSON.parse(readFileSync(statePathIn(tmpDir), 'utf8'));
      assert.strictEqual(onDisk.history.length, 1);
      assert.strictEqual(onDisk.history[0].success, true);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('persists state atomically via write-then-rename', async () => {
    const tmpDir = makeTmpDir();
    try {
      await runRepair({
        stateDir: tmpDir,
        now: () => BASE_NOW,
        spawner: async () => ({ code: 0 }),
        state: { last_probe: { state: 'model-only' } },
      });

      assert.ok(fsCounts.rename >= 1, 'expected at least one atomic rename');
    } finally {
      rmTmp(tmpDir);
    }
  });
});

/** Writes a raw lock fixture bypassing the module under test. */
function fsRealWriteLock(dir, lock) {
  rawWriteFileSync(lockPathIn(dir), JSON.stringify(lock, null, 2));
}

// The raw-fs helper resolves the REAL node:fs writeFileSync (imported before
// any mock registration) so lock fixtures are never proxied/counted.
import { writeFileSync as rawWriteFileSync } from 'node:fs';

// Keep the pidAlive helper referenced so liveness semantics stay documented
// even though the dead-pid path uses an offset pid.
void pidAlive;
