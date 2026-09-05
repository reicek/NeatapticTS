/**
 * @module cortex-health-guard.test
 * @description Red-phase unit tests for the Cortex self-heal guard decision engine.
 *
 * The guard decides whether a degraded dense-readiness signal should trigger
 * repair, coordinates the shared state file and repair lock, applies cooldown,
 * backoff, and the attempt ceiling, and renders the model-facing self_heal
 * guidance block. All external effects are injected (clock, probe, spawner,
 * temp state dir) and the warm happy path must stay free.
 *
 * Module contract under test (per the plan's Phase-2 Step-02 packet):
 *   evaluateSelfHeal({ probe, now, stateDir, spawner })
 *     -> { action, guidanceFields, spawnDecision }
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
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

const { evaluateSelfHeal } = await import('./cortex-health-guard.mjs');

const BASE_NOW = 1_000_000_000_000;

const SELFHEAL_ENV_KEYS = [
  'CORTEX_SELFHEAL_COOLDOWN_S',
  'CORTEX_SELFHEAL_BACKOFF_FACTOR',
  'CORTEX_SELFHEAL_MAX_ATTEMPTS',
  'CORTEX_SELFHEAL_ATTEMPT_WINDOW_S',
  'CORTEX_SELFHEAL_LOCK_STALE_S',
  'CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS',
  'CORTEX_SELFHEAL_DISABLE',
];

let savedEnv;

beforeEach(() => {
  savedEnv = { ...process.env };
  for (const key of SELFHEAL_ENV_KEYS) delete process.env[key];
});

afterEach(() => {
  for (const key of SELFHEAL_ENV_KEYS) {
    const value = savedEnv[key];
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
});

/**
 * Re-import the guard module with a clean module registry so env knobs read
 * at import time and per-process probe-cost memos are isolated per test.
 */
async function freshImport() {
  jest.resetModules();
  jest.unstable_mockModule('node:fs', () => countedNamespace(fsReal));
  jest.unstable_mockModule('node:fs/promises', () =>
    countedNamespace(fsPromisesReal),
  );
  return import('./cortex-health-guard.mjs');
}

function makeTmpDir() {
  return mkdtempSync(path.join(os.tmpdir(), 'cortex-guard-'));
}

function rmTmp(dir) {
  try {
    rmSync(dir, { recursive: true, force: true });
  } catch {
    // ignore cleanup failures
  }
}

function statePathIn(dir) {
  return path.join(dir, 'cortex-self-heal-state.json');
}

function lockPathIn(dir) {
  return path.join(dir, 'cortex-self-heal.repair.lock');
}

function modelOnlyReport(chunkCount = 24957, embeddingCount = 0) {
  return {
    ready: false,
    state: 'model-only',
    reason: `Embeddings are incomplete: expected ${chunkCount}, found ${embeddingCount}.`,
    chunk_count: chunkCount,
    embedding_count: embeddingCount,
  };
}

function warmReport() {
  return {
    ready: true,
    state: 'warm',
    reason: '24957 chunks have embeddings.',
    chunk_count: 24957,
    embedding_count: 24957,
  };
}

/**
 * Writes a state file in the exact plan schema so the guard can be exercised
 * against pre-existing orchestrator-written state (single writer is the
 * orchestrator; the guard only reads or updates it on trigger).
 */
function writeStateFixture(dir, overrides = {}) {
  const state = {
    version: 1,
    last_probe: {
      state: 'model-only',
      dense_reason: 'Embeddings are incomplete: expected 24957, found 0.',
      chunk_count: 24957,
      embedding_count: 0,
      probed_at: BASE_NOW - 60_000,
    },
    cooldown: { cooldown_s: 600, last_trigger_at: null },
    backoff: {
      attempts_in_window: 0,
      window_started_at: BASE_NOW,
      backoff_factor: 2,
      next_allowed_at: 0,
    },
    max_attempts: 3,
    in_flight: null,
    history: [],
    ...overrides,
  };
  fsReal.writeFileSync(statePathIn(dir), JSON.stringify(state, null, 2));
  return state;
}

describe('evaluateSelfHeal — first degraded trigger (AC-001)', () => {
  it('starts the repair with T1 guidance on the first degraded evaluation', async () => {
    const tmpDir = makeTmpDir();
    try {
      const spawner = jest.fn().mockResolvedValue({ pid: 1234 });

      const decision = await evaluateSelfHeal({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner,
      });

      assert.strictEqual(decision.action, 'started');
      assert.match(
        decision.guidanceFields.guidance,
        /self-heal has been STARTED/i,
      );
      assert.strictEqual(spawner.mock.calls.length, 1);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('creates the repair lock atomically with the plan record schema', async () => {
    const tmpDir = makeTmpDir();
    try {
      await evaluateSelfHeal({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner: async () => ({ pid: 1234 }),
      });

      const onDisk = JSON.parse(readFileSync(lockPathIn(tmpDir), 'utf8'));
      assert.deepStrictEqual(Object.keys(onDisk).sort(), [
        'attempt',
        'heartbeat_at',
        'pid',
        'reason',
        'started_at',
      ]);
      assert.strictEqual(onDisk.pid, 1234);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('persists the state file with the exact plan schema', async () => {
    const tmpDir = makeTmpDir();
    try {
      await evaluateSelfHeal({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner: async () => ({ pid: 1234 }),
      });

      const onDisk = JSON.parse(readFileSync(statePathIn(tmpDir), 'utf8'));
      assert.deepStrictEqual(Object.keys(onDisk).sort(), [
        'backoff',
        'cooldown',
        'history',
        'in_flight',
        'last_probe',
        'max_attempts',
        'version',
      ]);
      assert.deepStrictEqual(onDisk.last_probe, {
        state: 'model-only',
        dense_reason: 'Embeddings are incomplete: expected 24957, found 0.',
        chunk_count: 24957,
        embedding_count: 0,
        probed_at: BASE_NOW,
      });
      assert.strictEqual(onDisk.in_flight.pid, 1234);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('evaluateSelfHeal — in-flight and cooldown (AC-002)', () => {
  it('reports T2 in-flight guidance without respawning during an active repair', async () => {
    const tmpDir = makeTmpDir();
    try {
      const spawner = jest.fn().mockResolvedValue({ pid: 1234 });
      const consult = () =>
        evaluateSelfHeal({
          probe: async () => modelOnlyReport(),
          now: () => BASE_NOW,
          stateDir: tmpDir,
          spawner,
        });

      await consult();
      const second = await consult();

      assert.strictEqual(second.action, 'in_flight');
      assert.match(second.guidanceFields.guidance, /ALREADY RUNNING/i);
      assert.strictEqual(spawner.mock.calls.length, 1);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('never spawns a second repair inside the cooldown window', async () => {
    const tmpDir = makeTmpDir();
    try {
      writeStateFixture(tmpDir, {
        cooldown: { cooldown_s: 600, last_trigger_at: BASE_NOW - 10_000 },
      });

      const spawner = jest.fn().mockResolvedValue({ pid: 1234 });
      const decision = await evaluateSelfHeal({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner,
      });

      assert.notStrictEqual(decision.action, 'started');
      assert.strictEqual(spawner.mock.calls.length, 0);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('evaluateSelfHeal — backoff and attempt ceiling (AC-002)', () => {
  it('doubles next_allowed_at via the backoff factor when re-triggering after a failure', async () => {
    const tmpDir = makeTmpDir();
    try {
      writeStateFixture(tmpDir, {
        cooldown: { cooldown_s: 600, last_trigger_at: BASE_NOW - 3_600_000 },
        backoff: {
          attempts_in_window: 1,
          window_started_at: BASE_NOW,
          backoff_factor: 2,
          next_allowed_at: 0,
        },
        history: [
          {
            success: false,
            started_at: BASE_NOW - 3_600_000,
            error: 'prewarm timeout',
          },
        ],
      });

      const decision = await evaluateSelfHeal({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner: jest.fn().mockResolvedValue({ pid: 1234 }),
      });

      assert.strictEqual(decision.action, 'started');
      assert.strictEqual(
        decision.guidanceFields.next_allowed_at,
        BASE_NOW + 1_200_000,
      );
      const onDisk = JSON.parse(readFileSync(statePathIn(tmpDir), 'utf8'));
      assert.strictEqual(onDisk.backoff.next_allowed_at, BASE_NOW + 1_200_000);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('resets the attempt window and allows a new repair after the window elapses', async () => {
    const tmpDir = makeTmpDir();
    try {
      writeStateFixture(tmpDir, {
        cooldown: { cooldown_s: 600, last_trigger_at: BASE_NOW - 2_000_000 },
        backoff: {
          attempts_in_window: 3,
          window_started_at: BASE_NOW - 2 * 86_400_000,
          backoff_factor: 2,
          next_allowed_at: 0,
        },
        history: [
          { success: false, started_at: BASE_NOW - 2 * 86_400_000 },
          { success: false, started_at: BASE_NOW - 2 * 86_400_000 + 1 },
          { success: false, started_at: BASE_NOW - 2 * 86_400_000 + 2 },
        ],
      });

      const decision = await evaluateSelfHeal({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner: jest.fn().mockResolvedValue({ pid: 1234 }),
      });

      assert.strictEqual(decision.action, 'started');
      const onDisk = JSON.parse(readFileSync(statePathIn(tmpDir), 'utf8'));
      assert.strictEqual(onDisk.backoff.attempts_in_window, 1);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('emits T3 pause-and-ask guidance once the attempt ceiling is reached', async () => {
    const tmpDir = makeTmpDir();
    try {
      writeStateFixture(tmpDir, {
        cooldown: { cooldown_s: 600, last_trigger_at: BASE_NOW - 3_600_000 },
        backoff: {
          attempts_in_window: 3,
          window_started_at: BASE_NOW,
          backoff_factor: 2,
          next_allowed_at: 0,
        },
        history: [
          { success: false, started_at: BASE_NOW - 3_600_000 },
          { success: false, started_at: BASE_NOW - 1_800_000 },
          { success: false, started_at: BASE_NOW - 900_000 },
        ],
      });

      const decision = await evaluateSelfHeal({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner: jest.fn().mockResolvedValue({ pid: 1234 }),
      });

      assert.strictEqual(decision.action, 'exhausted');
      assert.match(
        decision.guidanceFields.guidance,
        /automatic self-heal has FAILED/i,
      );
      assert.deepStrictEqual(decision.guidanceFields.manual_recovery, [
        'npm run index:session-start',
        'npm run index:prewarm',
        'node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json',
      ]);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('honors CORTEX_SELFHEAL_MAX_ATTEMPTS for the ceiling', async () => {
    const tmpDir = makeTmpDir();
    try {
      process.env.CORTEX_SELFHEAL_MAX_ATTEMPTS = '1';
      const { evaluateSelfHeal: isolated } = await freshImport();

      writeStateFixture(tmpDir, {
        backoff: {
          attempts_in_window: 1,
          window_started_at: BASE_NOW,
          backoff_factor: 2,
          next_allowed_at: 0,
        },
      });

      const decision = await isolated({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner: jest.fn().mockResolvedValue({ pid: 1234 }),
      });

      assert.strictEqual(decision.action, 'exhausted');
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('honors CORTEX_SELFHEAL_COOLDOWN_S for the cooldown window', async () => {
    const tmpDir = makeTmpDir();
    try {
      process.env.CORTEX_SELFHEAL_COOLDOWN_S = '120';
      const { evaluateSelfHeal: isolated } = await freshImport();

      const decision = await isolated({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner: jest.fn().mockResolvedValue({ pid: 1234 }),
      });

      assert.strictEqual(decision.guidanceFields.cooldown_s, 120);
      const onDisk = JSON.parse(readFileSync(statePathIn(tmpDir), 'utf8'));
      assert.strictEqual(onDisk.cooldown.cooldown_s, 120);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('evaluateSelfHeal — kill switch (AC-002)', () => {
  it('blocks triggering and spawning under CORTEX_SELFHEAL_DISABLE=1 while degraded guidance still reports', async () => {
    const tmpDir = makeTmpDir();
    try {
      process.env.CORTEX_SELFHEAL_DISABLE = '1';
      const { evaluateSelfHeal: isolated } = await freshImport();

      const spawner = jest.fn().mockResolvedValue({ pid: 1234 });
      const decision = await isolated({
        probe: async () => modelOnlyReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner,
      });

      assert.strictEqual(decision.action, 'disabled');
      assert.strictEqual(decision.guidanceFields.state, 'model-only');
      assert.strictEqual(spawner.mock.calls.length, 0);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('evaluateSelfHeal — probe-cost fallback (AC-002)', () => {
  it('switches subsequent evaluations to state-file-only checks when the probe exceeds the cost threshold', async () => {
    const tmpDir = makeTmpDir();
    try {
      writeStateFixture(tmpDir, {
        in_flight: {
          pid: process.pid,
          started_at: BASE_NOW - 60_000,
          attempt: 1,
          reason: 'model-only',
          est_duration_min: 5,
        },
      });

      let currentTime = BASE_NOW;
      const probe = jest.fn(async () => {
        currentTime += 700; // 700 ms measured > 600 ms default threshold
        return modelOnlyReport();
      });
      const spawner = jest.fn().mockResolvedValue({ pid: 1234 });
      const { evaluateSelfHeal: isolated } = await freshImport();

      const first = await isolated({
        probe,
        now: () => currentTime,
        stateDir: tmpDir,
        spawner,
      });
      const second = await isolated({
        probe,
        now: () => currentTime,
        stateDir: tmpDir,
        spawner,
      });

      assert.strictEqual(first.action, 'in_flight');
      assert.strictEqual(probe.mock.calls.length, 1);
      assert.strictEqual(second.action, 'in_flight');
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('keeps the live-probe path when the probe cost stays under the threshold', async () => {
    const tmpDir = makeTmpDir();
    try {
      writeStateFixture(tmpDir, {
        in_flight: {
          pid: process.pid,
          started_at: BASE_NOW - 60_000,
          attempt: 1,
          reason: 'model-only',
          est_duration_min: 5,
        },
      });

      const probe = jest.fn(async () => modelOnlyReport());
      const { evaluateSelfHeal: isolated } = await freshImport();
      const consult = () =>
        isolated({
          probe,
          now: () => BASE_NOW,
          stateDir: tmpDir,
          spawner: jest.fn().mockResolvedValue({ pid: 1234 }),
        });

      await consult();
      await consult();

      assert.strictEqual(probe.mock.calls.length, 2);
    } finally {
      rmTmp(tmpDir);
    }
  });

  it('honors CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS for the cost fallback', async () => {
    const tmpDir = makeTmpDir();
    try {
      process.env.CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS = '100';
      const { evaluateSelfHeal: isolated } = await freshImport();

      writeStateFixture(tmpDir, {
        in_flight: {
          pid: process.pid,
          started_at: BASE_NOW - 60_000,
          attempt: 1,
          reason: 'model-only',
          est_duration_min: 5,
        },
      });

      let currentTime = BASE_NOW;
      const probe = jest.fn(async () => {
        currentTime += 150; // 150 ms > 100 ms overridden threshold
        return modelOnlyReport();
      });
      const consult = () =>
        isolated({
          probe,
          now: () => currentTime,
          stateDir: tmpDir,
          spawner: jest.fn().mockResolvedValue({ pid: 1234 }),
        });

      await consult();
      await consult();

      assert.strictEqual(probe.mock.calls.length, 1);
    } finally {
      rmTmp(tmpDir);
    }
  });
});

describe('evaluateSelfHeal — warm happy path (AC-007)', () => {
  it('keeps the warm path free: zero fs operations and zero spawns', async () => {
    const tmpDir = makeTmpDir();
    try {
      const spawner = jest.fn().mockResolvedValue({ pid: 1234 });
      const fsBefore = fsCounts.total;

      const decision = await evaluateSelfHeal({
        probe: async () => warmReport(),
        now: () => BASE_NOW,
        stateDir: tmpDir,
        spawner,
      });

      assert.strictEqual(fsCounts.total - fsBefore, 0);
      assert.deepStrictEqual(decision, {
        action: null,
        guidanceFields: null,
        spawnDecision: null,
      });
      assert.strictEqual(spawner.mock.calls.length, 0);
    } finally {
      rmTmp(tmpDir);
    }
  });
});
