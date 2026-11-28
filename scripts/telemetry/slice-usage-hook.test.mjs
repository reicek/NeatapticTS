/**
 * @module slice-usage-hook.test
 * @description RED contract home for the tool-execution observer of the vertical slice B.
 *
 * Slice: OT-V2-1 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria pinned by this home:
 *   - AC-V2-A (the observer home), which carries forward AC-011-D, AC-004-E,
 *     AC-003-F, AC-011-E and the anti-vacuity lane of the failure event key.
 *
 * THE SUBJECT UNDER TEST IS `scripts/agent-customization/hooks/slice-usage-hook.mjs`,
 * THE REPO-OWNED TOOL-EXECUTION OBSERVER, AND THE TRACKED MANIFEST
 * `.github/hooks/cortex-refresh.json` THAT REGISTERS IT. Every lane drives the
 * REAL observer as a REAL child process over REAL temporary artefacts. Nothing
 * is mocked: not the observer, not the manifest, not the dispatch record.
 *
 * WHY THE OBSERVER IS DRIVEN AS A CHILD PROCESS: the observer is a HOOK, its
 * contract IS its standard input, its standard output, its exit code and the
 * bytes it appends. A hook that is driven through an import cannot prove that
 * it stays fail-open, that it appends exactly one complete line, or that it
 * survives a concurrent writer. The observer does not exist yet, so every lane
 * is RED BY ASSERTION: the lanes fail on the exit code, on the missing line,
 * on the missing manifest entry or on the leaked content - never on an import
 * error that would abort the file.
 *
 * THE SEAMS THIS HOME PINS (the implementer of OT-V2-2 owns these):
 *   - the observer reads the host payload as JSON on STANDARD INPUT and exits
 *     0 (it is FAIL-OPEN: a failed observation never fails the tool call);
 *   - it resolves the transient dispatch-record directory from
 *     `NEATAPTIC_DISPATCH_RECORD_DIR` (the seam the repository already uses)
 *     and falls back to `logs/slice-ledger/dispatch` under the repo root;
 *   - it resolves the observation log from `NEATAPTIC_OBSERVATION_LOG_PATH`
 *     and falls back to the sibling `tool-executions.jsonl` of the resolved
 *     dispatch-record directory (which is the repository path
 *     `logs/slice-ledger/tool-executions.jsonl`);
 *   - it appends ONE JSON line per observed execution carrying EXACTLY five
 *     fields: `seq`, `slice_id`, `session_key`, `bucket` and `outcome`;
 *   - `outcome` is `failed` when the payload carries a non-empty `error` (in
 *     either payload spelling) and `completed` otherwise;
 *   - `seq` is the append-time LINE INDEX of that line in the log;
 *   - `session_key` is a TRUNCATED SHA-256 digest of the raw session id;
 *   - `bucket` is one of the five safe buckets of
 *     `scripts/telemetry/telemetry-contract.mjs`;
 *   - it writes NOTHING when there is no open dispatch record, and it never
 *     attributes an execution to a GUESSED slice: when the dispatch directory
 *     holds more than one record the open slice is ambiguous and the observer
 *     writes nothing.
 *
 * THE MANIFEST CONTRACT (the tracked registration):
 *   - the observer is registered under BOTH post event keys, `PostToolUse`
 *     (a successful completion) and `PostToolUseFailure` (a failed
 *     completion), because a failing call SKIPS the success key entirely;
 *   - both entries run the SAME command;
 *   - NEITHER entry carries a `matcher` field, because the correct way to
 *     receive every tool is to omit the matcher, and an invalid pattern makes
 *     the host SKIP the hook entry silently;
 *   - every matcher that IS present anywhere in the manifest compiles under
 *     `^(?:PATTERN)$` and is never the bare `*`, which is not a valid
 *     post-event pattern.
 *
 * THE HOOK-LOAD BOUNDARY (AC-011-E): the repository loads its hooks at
 * STARTUP, so a manifest change is not live in the session that wrote it. The
 * end-to-end proof therefore has to run in a FRESH session and has to PROVE
 * that the observer actually fired. The lane below drives the command the
 * MANIFEST NAMES, in a FRESH child process, and asserts that the line the
 * manifest's command produced is in the log: a manifest that names a command
 * which does not append the line is not a registration.
 *
 * THE DENIED CALL (R-13): a tool call denied by a `preToolUse` hook never
 * completes, so NEITHER post event fires for it and it is never observed. The
 * observable consequence is the persisted outcome VOCABULARY: the log carries
 * `completed` and `failed` and nothing else, so a denied call can never be
 * counted into either counter.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { spawn, spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { scanPersistedText, TOOL_EXECUTION_BUCKETS, writeDispatchRecord } from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/** The repo-relative path of the observer this home owns. */
const HOOK_RELPATH = 'scripts/agent-customization/hooks/slice-usage-hook.mjs';
/** The absolute path of the observer this home owns. */
const HOOK_PATH = path.join(REPO_ROOT, 'scripts', 'agent-customization', 'hooks', 'slice-usage-hook.mjs');
/** The repo-relative path of the tracked manifest that registers the observer. */
const MANIFEST_RELPATH = '.github/hooks/cortex-refresh.json';
/** The absolute path of the tracked manifest that registers the observer. */
const MANIFEST_PATH = path.join(REPO_ROOT, '.github', 'hooks', 'cortex-refresh.json');
/** The seam the repository already uses for the transient dispatch-record directory. */
const DISPATCH_DIR_ENV = 'NEATAPTIC_DISPATCH_RECORD_DIR';
/** The seam this home pins for the observation log. */
const OBSERVATION_LOG_ENV = 'NEATAPTIC_OBSERVATION_LOG_PATH';

const PLAN_ID = 'fixture-plan';
const SLICE_ID = 'OT-V2-9';
const OTHER_SLICE_ID = 'OT-V2-10';

/** The five fields the observation log line carries, and nothing else. */
const OBSERVATION_FIELDS = ['bucket', 'outcome', 'seq', 'session_key', 'slice_id'];

/** The camelCase spelling of the host payload. */
const CAMEL_PAYLOAD = {
  sessionId: 'session-camel-9f3a7c21',
  toolName: 'functions.view',
  toolArgs: { path: 'ARG-SENTINEL-7b21' },
  toolResult: 'RESULT-SENTINEL-4d18',
};

/** The VS Code spelling of the host payload. */
const VSCODE_PAYLOAD = {
  session_id: 'session-vscode-5c02be77',
  tool_name: 'functions.view',
  tool_input: { path: 'ARG-SENTINEL-7b21' },
  tool_result: 'RESULT-SENTINEL-4d18',
};

/** The failure payload: the host reports a failed completion with an error. */
const FAILURE_PAYLOAD = {
  ...VSCODE_PAYLOAD,
  error: 'ERROR-SENTINEL-2f66',
};

/** Every sentinel the persisted line must never carry. */
const LEAKED_SENTINELS = [
  'ARG-SENTINEL-7b21',
  'RESULT-SENTINEL-4d18',
  'ERROR-SENTINEL-2f66',
  'functions.view',
];

/** The temporary workspace of the running lane. */
let workspace;

/**
 * Creates the temporary workspace: a transient dispatch-record directory and
 * the observation log path that belongs to it.
 *
 * @returns {{ root: string, dispatchDir: string, logPath: string }} The workspace.
 */
function createWorkspace() {
  const root = mkdtempSync(path.join(tmpdir(), 'slice-usage-hook-'));
  const dispatchDir = path.join(root, 'dispatch');
  mkdirSync(dispatchDir, { recursive: true });
  return {
    root,
    dispatchDir,
    logPath: path.join(root, 'tool-executions.jsonl'),
  };
}

/**
 * Writes a transient dispatch record with the REAL producer helper, so the
 * observer reads the record in exactly the shape the repository writes it.
 *
 * @param {string} sliceId The slice id the record names.
 * @param {string} [directory] The dispatch-record directory.
 * @returns {string} The record path.
 */
function writeOpenDispatchRecord(sliceId, directory = workspace.dispatchDir) {
  return writeDispatchRecord(
    {
      plan_id: PLAN_ID,
      slice_id: sliceId,
      context_tokens_delivered: 1200,
      dispatch_seq: 0,
      started_monotonic_ns: '0x0',
    },
    { directory },
  );
}

/**
 * Runs the REAL observer as a REAL child process with the host payload on
 * standard input.
 *
 * @param {object} payload The host payload.
 * @param {object} [options] The overrides.
 * @param {string} [options.logPath] The observation log path.
 * @param {string} [options.dispatchDir] The dispatch-record directory.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string, error?: Error }} The run result.
 */
function runObserver(payload, options = {}) {
  const result = spawnSync(process.execPath, [HOOK_PATH], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 60000,
    input: JSON.stringify(payload),
    env: {
      ...process.env,
      [DISPATCH_DIR_ENV]: options.dispatchDir ?? workspace.dispatchDir,
      [OBSERVATION_LOG_ENV]: options.logPath ?? workspace.logPath,
    },
  });
  const stdout = result.stdout ?? '';
  const stderr = result.stderr ?? '';
  return { status: result.status, stdout, stderr, output: `${stdout}\n${stderr}`, error: result.error };
}

/**
 * Runs the REAL observer asynchronously, so two observers can append at the
 * same time.
 *
 * @param {object} payload The host payload.
 * @returns {Promise<{ status: number|null, stdout: string, stderr: string, output: string }>} The run result.
 */
function runObserverAsync(payload) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [HOOK_PATH], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        [DISPATCH_DIR_ENV]: workspace.dispatchDir,
        [OBSERVATION_LOG_ENV]: workspace.logPath,
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    child.on('close', (status) => {
      resolve({ status, stdout, stderr, output: `${stdout}\n${stderr}` });
    });
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

/**
 * Tokenises a command line, so the command the MANIFEST names can be executed
 * exactly as the host would execute it.
 *
 * @param {string} command The command line.
 * @returns {string[]} The tokens.
 */
function tokenizeCommand(command) {
  const tokens = [];
  const pattern = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let match = pattern.exec(command);
  while (match !== null) {
    tokens.push(match[1] ?? match[2] ?? match[3]);
    match = pattern.exec(command);
  }
  return tokens;
}

/**
 * Executes the command the MANIFEST names, in a FRESH child process, with the
 * host payload on standard input.
 *
 * @param {string} command The manifest command.
 * @param {object} payload The host payload.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string, error?: Error }} The run result.
 */
function runManifestCommand(command, payload) {
  const tokens = tokenizeCommand(command);
  const [binary, ...args] = tokens;
  const result = spawnSync(binary === 'node' ? process.execPath : binary, args, {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 60000,
    input: JSON.stringify(payload),
    env: {
      ...process.env,
      [DISPATCH_DIR_ENV]: workspace.dispatchDir,
      [OBSERVATION_LOG_ENV]: workspace.logPath,
    },
  });
  const stdout = result.stdout ?? '';
  const stderr = result.stderr ?? '';
  return { status: result.status, stdout, stderr, output: `${stdout}\n${stderr}`, error: result.error };
}

/**
 * Reads the tracked manifest.
 *
 * @returns {Record<string, unknown>} The parsed manifest.
 */
function readManifest() {
  return JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
}

/**
 * Returns the manifest entries of an event key that register the observer.
 *
 * @param {Record<string, unknown>} manifest The parsed manifest.
 * @param {string} eventKey The hook event key.
 * @returns {Array<Record<string, unknown>>} The observer entries.
 */
function observerEntries(manifest, eventKey) {
  const hooks = manifest.hooks ?? {};
  const entries = Array.isArray(hooks[eventKey]) ? hooks[eventKey] : [];
  return entries.filter((entry) => String(entry?.command ?? '').includes('slice-usage-hook.mjs'));
}

/**
 * Reads the observation log as its non-empty lines.
 *
 * @param {string} [logPath] The log path.
 * @returns {string[]} The lines.
 */
function readObservationLines(logPath = workspace.logPath) {
  if (!existsSync(logPath)) return [];
  return readFileSync(logPath, 'utf8')
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
}

/**
 * Reads the observation log as its parsed lines.
 *
 * @param {string} [logPath] The log path.
 * @returns {Record<string, unknown>[]} The parsed lines.
 */
function readObservationRecords(logPath = workspace.logPath) {
  return readObservationLines(logPath).map((line) => JSON.parse(line));
}

/**
 * Asserts that the observer is on disk, so a lane that fails afterwards fails
 * on the OBSERVER's behaviour rather than on a missing file.
 */
function expectObserverAvailable() {
  expect({ observer: HOOK_RELPATH, exists: existsSync(HOOK_PATH) }).toEqual({
    observer: HOOK_RELPATH,
    exists: true,
  });
}

beforeEach(() => {
  workspace = createWorkspace();
});

afterEach(() => {
  rmSync(workspace.root, { recursive: true, force: true });
});

describe('OT-V2-1: the tracked registration of the observer', () => {
  it('registers the observer under BOTH post event keys', () => {
    const manifest = readManifest();
    const success = observerEntries(manifest, 'PostToolUse');
    const failure = observerEntries(manifest, 'PostToolUseFailure');
    expect({ success: success.length, failure: failure.length }).toEqual({ success: 1, failure: 1 });
  });

  it('runs the SAME command for the success key and for the failure key', () => {
    const manifest = readManifest();
    const [success] = observerEntries(manifest, 'PostToolUse');
    const [failure] = observerEntries(manifest, 'PostToolUseFailure');
    expect(success.command).toBe(failure.command);
  });

  it('carries NO matcher on either observer entry', () => {
    const manifest = readManifest();
    const entries = [
      ...observerEntries(manifest, 'PostToolUse'),
      ...observerEntries(manifest, 'PostToolUseFailure'),
    ];
    expect(entries).toHaveLength(2);
    for (const entry of entries) {
      expect(Object.prototype.hasOwnProperty.call(entry, 'matcher')).toBe(false);
    }
  });

  it('compiles every matcher in the manifest under ^(?:PATTERN)$ and never uses the bare `*`', () => {
    const manifest = readManifest();
    // The guard is checked against the manifest the contract describes: the
    // one that carries BOTH observer entries. Checked against a manifest that
    // does not carry them yet, the rule would hold vacuously and the lane
    // would prove nothing.
    expect(observerEntries(manifest, 'PostToolUse')).toHaveLength(1);
    expect(observerEntries(manifest, 'PostToolUseFailure')).toHaveLength(1);
    const matchers = [];
    for (const entries of Object.values(manifest.hooks ?? {})) {
      for (const entry of Array.isArray(entries) ? entries : []) {
        if (typeof entry?.matcher === 'string') matchers.push(entry.matcher);
      }
    }
    for (const matcher of matchers) {
      expect(() => new RegExp(`^(?:${matcher})$`)).not.toThrow();
      expect(matcher).not.toBe('*');
    }
  });

  it('names a command whose script EXISTS for every hook entry, so no entry is silently skipped', () => {
    // The same defect class as an invalid matcher: an entry that names a
    // script that is not on disk is SKIPPED, and the observer would then
    // never fire.
    const manifest = readManifest();
    // The manifest the contract describes carries BOTH observer entries; only
    // against that manifest can this lane say anything about the observer.
    expect(observerEntries(manifest, 'PostToolUse')).toHaveLength(1);
    expect(observerEntries(manifest, 'PostToolUseFailure')).toHaveLength(1);
    const commands = [];
    for (const entries of Object.values(manifest.hooks ?? {})) {
      for (const entry of Array.isArray(entries) ? entries : []) {
        if (typeof entry?.command === 'string') commands.push(entry.command);
      }
    }
    expect(commands.length).toBeGreaterThan(0);
    const missing = commands
      .map((command) => command.trim().split(/\s+/).filter((token) => !token.startsWith('-'))[1] ?? '')
      .filter((script) => !existsSync(path.join(REPO_ROOT, script)));
    expect(missing).toEqual([]);
  });

  it('executes the command the manifest names, in a FRESH session, and the observer appends the line', () => {
    // THE HOOK-LOAD BOUNDARY (AC-011-E): the manifest names the command, the
    // command is executed in a FRESH child process, and the line the command
    // produced is in the log. A manifest that names a command which never
    // appends the line is not a registration.
    const manifest = readManifest();
    const [success] = observerEntries(manifest, 'PostToolUse');
    expect(success).toBeDefined();
    writeOpenDispatchRecord(SLICE_ID);
    const result = runManifestCommand(String(success.command), CAMEL_PAYLOAD);
    expect(result.status).toBe(0);
    expect(readObservationLines()).toHaveLength(1);
  });
});

describe('OT-V2-1: the observation the observer persists', () => {
  it('exists at the path the contract declares', () => {
    expectObserverAvailable();
  });

  it('writes exactly ONE line for ONE observed execution', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    const result = runObserver(CAMEL_PAYLOAD);
    expect(result.status).toBe(0);
    expect(readObservationLines()).toHaveLength(1);
  });

  it('persists EXACTLY the five contract fields and nothing else', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    const [record] = readObservationRecords();
    expect(Object.keys(record).sort()).toEqual(OBSERVATION_FIELDS);
  });

  it('stamps the slice id of the OPEN dispatch record', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    const [record] = readObservationRecords();
    expect(record.slice_id).toBe(SLICE_ID);
  });

  it('derives the bucket from the tool name WITHOUT persisting the tool name', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    const [record] = readObservationRecords();
    expect(TOOL_EXECUTION_BUCKETS).toContain(record.bucket);
    expect(readObservationLines()[0]).not.toContain('functions.view');
  });

  it('persists a TRUNCATED session digest and NEVER the raw session id', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    const [record] = readObservationRecords();
    expect(typeof record.session_key).toBe('string');
    expect(record.session_key.length).toBeGreaterThan(0);
    expect(record.session_key.length).toBeLessThan(CAMEL_PAYLOAD.sessionId.length);
    expect(record.session_key).not.toBe(CAMEL_PAYLOAD.sessionId);
    expect(readObservationLines()[0]).not.toContain(CAMEL_PAYLOAD.sessionId);
  });

  it('persists NO argument, NO result and NO content of a successful execution', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    const line = readObservationLines()[0];
    for (const sentinel of LEAKED_SENTINELS) {
      expect(line).not.toContain(sentinel);
    }
  });

  it('persists NO error content of a failed execution - the failure path is no less private', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(FAILURE_PAYLOAD);
    const line = readObservationLines()[0];
    for (const sentinel of LEAKED_SENTINELS) {
      expect(line).not.toContain(sentinel);
    }
  });

  it('writes no forbidden key and no forbidden value into the observation log', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    runObserver(FAILURE_PAYLOAD);
    expect(scanPersistedText(readFileSync(workspace.logPath, 'utf8'))).toEqual([]);
  });

  it('names the two completion outcomes SEPARATELY: `completed` and `failed`', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    runObserver(FAILURE_PAYLOAD);
    const outcomes = readObservationRecords().map((record) => record.outcome);
    expect(outcomes).toEqual(['completed', 'failed']);
  });

  it('persists the outcome vocabulary `completed` and `failed` and nothing else', () => {
    // A tool call DENIED before execution never completes, so NEITHER post
    // event fires for it and it is never observed: the persisted vocabulary is
    // the proof that a denied call can land in neither counter.
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    runObserver(FAILURE_PAYLOAD);
    const outcomes = readObservationRecords().map((record) => record.outcome);
    expect([...new Set(outcomes)].sort()).toEqual(['completed', 'failed']);
  });

  it('accepts the camelCase payload and the VS Code payload with the same result', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    runObserver(VSCODE_PAYLOAD);
    const records = readObservationRecords();
    expect(records).toHaveLength(2);
    for (const record of records) {
      expect(Object.keys(record).sort()).toEqual(OBSERVATION_FIELDS);
      expect(record.slice_id).toBe(SLICE_ID);
      expect(TOOL_EXECUTION_BUCKETS).toContain(record.bucket);
    }
    expect(records[0].session_key).not.toBe(records[1].session_key);
  });

  it('derives the seq from the append-time LINE INDEX', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    runObserver(CAMEL_PAYLOAD);
    runObserver(VSCODE_PAYLOAD);
    runObserver(FAILURE_PAYLOAD);
    const seqs = readObservationRecords().map((record) => record.seq);
    for (const seq of seqs) {
      expect(Number.isInteger(seq)).toBe(true);
      expect(seq).toBeGreaterThanOrEqual(0);
    }
    expect(seqs[1]).toBe(seqs[0] + 1);
    expect(seqs[2]).toBe(seqs[1] + 1);
  });

  it('continues the seq from a PRE-SEEDED observation log', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    const seeded = [
      '{"bucket":"read","outcome":"completed","seq":0,"session_key":"seed-0","slice_id":"OT-V2-1"}',
      '{"bucket":"write","outcome":"failed","seq":1,"session_key":"seed-1","slice_id":"OT-V2-2"}',
    ].join('\n');
    writeFileSync(workspace.logPath, `${seeded}\n`, 'utf8');
    runObserver(CAMEL_PAYLOAD);
    const records = readObservationRecords();
    expect(records).toHaveLength(3);
    expect(records[2].seq).toBe(2);
  });

  it('writes ONE well-formed line per execution under CONCURRENT writers', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    return Promise.all([
      runObserverAsync(CAMEL_PAYLOAD),
      runObserverAsync(VSCODE_PAYLOAD),
      runObserverAsync(FAILURE_PAYLOAD),
      runObserverAsync(CAMEL_PAYLOAD),
      runObserverAsync(VSCODE_PAYLOAD),
      runObserverAsync(FAILURE_PAYLOAD),
    ]).then(() => {
      const lines = readObservationLines();
      expect(lines).toHaveLength(6);
      for (const line of lines) {
        const record = JSON.parse(line);
        expect(Object.keys(record).sort()).toEqual(OBSERVATION_FIELDS);
        expect(Number.isInteger(record.seq)).toBe(true);
      }
    });
  });

  it('writes NO line when no dispatch record is open', () => {
    expectObserverAvailable();
    const result = runObserver(CAMEL_PAYLOAD);
    expect(result.status).toBe(0);
    expect(readObservationLines()).toHaveLength(0);
  });

  it('never attributes an execution to a GUESSED slice when the open slice is ambiguous', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    writeOpenDispatchRecord(OTHER_SLICE_ID);
    const result = runObserver(CAMEL_PAYLOAD);
    expect(result.status).toBe(0);
    expect(readObservationLines()).toHaveLength(0);
  });

  it('stays FAIL-OPEN when the observation log cannot be written', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    // The parent of the observation log is a FILE, so every append fails.
    const blocker = path.join(workspace.root, 'blocker');
    writeFileSync(blocker, 'not a directory', 'utf8');
    const result = runObserver(CAMEL_PAYLOAD, { logPath: path.join(blocker, 'tool-executions.jsonl') });
    expect(result.status).toBe(0);
    expect(result.error).toBeUndefined();
    expect(readObservationLines()).toHaveLength(0);
  });

  it('reports NOTHING about the observed execution outside the log when the write fails', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    const blocker = path.join(workspace.root, 'blocker');
    writeFileSync(blocker, 'not a directory', 'utf8');
    const result = runObserver(CAMEL_PAYLOAD, { logPath: path.join(blocker, 'tool-executions.jsonl') });
    for (const sentinel of LEAKED_SENTINELS) {
      expect(result.output).not.toContain(sentinel);
    }
    expect(result.output).not.toContain(CAMEL_PAYLOAD.sessionId);
  });

  it('retries a TRANSIENT write failure a BOUNDED number of times and then stays fail-open', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    const blocker = path.join(workspace.root, 'blocker');
    writeFileSync(blocker, 'not a directory', 'utf8');
    const result = runObserver(CAMEL_PAYLOAD, { logPath: path.join(blocker, 'tool-executions.jsonl') });
    // The observer TERMINATED on its own: an unbounded retry would have hit
    // the spawn timeout and reported an error instead of an exit code.
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
  });

  it('creates the observation log under the repository path when no override is given', () => {
    // The fallback path is the repository path the workflow MCP already owns:
    // `logs/slice-ledger/tool-executions.jsonl`, which is the sibling of the
    // resolved dispatch-record directory.
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    const env = { ...process.env, [DISPATCH_DIR_ENV]: workspace.dispatchDir };
    delete env[OBSERVATION_LOG_ENV];
    const result = spawnSync(process.execPath, [HOOK_PATH], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 60000,
      input: JSON.stringify(CAMEL_PAYLOAD),
      env,
    });
    expect(result.status).toBe(0);
    const fallback = path.join(workspace.root, 'tool-executions.jsonl');
    expect(readObservationLines(fallback)).toHaveLength(1);
  });
});

describe('OT-V2-1: the observer never touches the repository state', () => {
  it('writes nothing outside the observation log it was pointed at', () => {
    expectObserverAvailable();
    writeOpenDispatchRecord(SLICE_ID);
    const entries = () =>
      readdirSync(workspace.root)
        .filter((name) => name !== path.basename(workspace.logPath))
        .sort();
    const before = entries();
    runObserver(CAMEL_PAYLOAD);
    expect(entries()).toEqual(before);
    expect(statSync(workspace.logPath).isFile()).toBe(true);
  });
});
