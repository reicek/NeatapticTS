/**
 * RED home for the vertical slice A telemetry exporter (slice OT-V1-1 of
 * `plans/opentelemetry-slice-telemetry.plans.md`).
 *
 * THE CONTRACT THIS HOME DEMANDS - the vertical slice A telemetry path is
 * built from three modules under `src/observability/telemetry/`:
 *
 *   1. `telemetry.policy.ts` exports the telemetry constants:
 *        - `TELEMETRY_SINK_ROOT`            the sink root, `'logs/telemetry'`
 *        - `TELEMETRY_NODE_FLOOR`           the telemetry engine floor, which
 *                                           declares `22.15.0`
 *        - `REASON_NODE_BELOW_TELEMETRY_FLOOR`
 *                                           the machine-stable reason token
 *                                           `'node_below_telemetry_floor'`
 *        - `TELEMETRY_ALLOW_LIST`           one entry per signal (`trace`,
 *                                           `metric`, `log`), each with
 *                                           `recordNames` and
 *                                           `attributeNames`
 *        - `FORBIDDEN_KEYS`                 the forbidden key names
 *        - `FORBIDDEN_VALUE_PATTERNS`       the forbidden value patterns
 *   2. `telemetry.jsonl-exporters.ts` exports `createJsonlExporter(options)`
 *      returning an exporter with `sinkDirectory`, `exportRecord(record)`,
 *      `forceFlush()` and `shutdown()`. The options are `sinkDirectory`,
 *      `signal` (`'trace' | 'metric' | 'log'`) and `warn`. A record is
 *      `{ name, attributes?, interval?, durationMs?, startTime?, endTime?,
 *      timeUnixNano?, observedTimestamp? }` and the monotonic interval is
 *      `{ startMonotonicMs, endMonotonicMs }`.
 *   3. `telemetry.bootstrap.ts` exports `loadOptionalPeer(specifier, warn?)`
 *      and `checkNodeFloor(nodeVersion?)`.
 *
 * THE RULES THIS HOME ENFORCES:
 *   - the written JSONL carries NO forbidden key name and NO calendar date;
 *   - it carries EXACTLY ONE duration field, `duration_ms`, and the value is
 *     derived from the monotonic interval - a caller supplied duration never
 *     becomes a second duration field;
 *   - an attribute outside the allow-list is dropped, and so is an
 *     allow-listed attribute whose value is an absolute path or a non-scalar
 *     bag of tool arguments or file content;
 *   - the meter scope exposes NO `neat.acceleration` instrument, so a metric
 *     record outside the allow-list is never written;
 *   - a sink that cannot be written, an attribute accessor that throws and a
 *     missing optional peer each produce a WARNING and never an exception
 *     into the caller, and every warning is a short machine-stable token;
 *   - the sink directory is created on demand;
 *   - the policy's forbidden key set and value patterns are IDENTICAL to the
 *     ones the single contract module owns (`scripts/telemetry/
 *     telemetry-contract.mjs`) - the drift guard.
 *
 * WHY EVERY MODULE IS LOADED LAZILY: the modules do not exist yet, and a
 * static import of a missing module is a TypeScript compile error
 * (TS2307) - that is an IMPORT failure, not a behavioural one. The
 * specifiers are therefore typed as `string` so the compiler does not try to
 * resolve them, and the loader turns a resolution failure into an
 * ASSERTION failure. The lane "resolves the modules lazily" proves the
 * loader really attempts a resolution, so no lane below it can be vacuous.
 *
 * WHY NOTHING IS MOCKED: every lane drives the REAL module over a REAL
 * temporary sink directory. The exporter, the policy, the sanitiser and the
 * bootstrap are never stubbed, and the contract module is read through a
 * child process rather than imported, because a `.mjs` import from a
 * TypeScript test would be a module-resolution failure of the test's own
 * making.
 *
 * THE ONE LANE THAT WRITES OUTSIDE A TEMPORARY DIRECTORY: the "REAL local
 * JSONL" lane deliberately writes into the REAL sink root (`logs/telemetry`)
 * because the privacy sweep has to run over a REAL artefact - a sweep over
 * a temporary directory proves nothing about what the vertical persists.
 * The lane is additive (it appends sanitized records), the sink root is
 * git-ignored, and the lane asserts the artefact against the repo's own
 * contract scanner.
 *
 * WHY THE ALLOW-LIST INJECTION SEAM EXISTS: the deny pass of the sanitiser
 * is unreachable through the public allow-lists, because every signal
 * allow-lists `neat.telemetry.*` names and none of them is forbidden. A
 * sanitiser whose deny pass were deleted would therefore still pass every
 * allow-list lane. The `allowedAttributeNames` seam injects a forbidden
 * name INTO the allow-list so the deny pass is provably live.
 *
 * @see plans/opentelemetry-slice-telemetry.plans.md - "Vertical slice A"
 */

import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';

type TelemetrySignal = 'trace' | 'metric' | 'log';

interface AllowListEntry {
  readonly recordNames: readonly string[];
  readonly attributeNames: readonly string[];
}

interface PolicyModule {
  readonly TELEMETRY_SINK_ROOT: string;
  readonly TELEMETRY_NODE_FLOOR: string;
  readonly REASON_NODE_BELOW_TELEMETRY_FLOOR: string;
  readonly TELEMETRY_ALLOW_LIST: {
    readonly trace: AllowListEntry;
    readonly metric: AllowListEntry;
    readonly log: AllowListEntry;
  };
  readonly FORBIDDEN_KEYS: readonly string[];
  readonly FORBIDDEN_VALUE_PATTERNS: readonly RegExp[];
}

interface TelemetryRecordInput {
  readonly name: string;
  readonly attributes?: Record<string, unknown>;
  readonly interval?: {
    readonly startMonotonicMs: number;
    readonly endMonotonicMs: number;
  };
  readonly durationMs?: number;
  readonly startTime?: unknown;
  readonly endTime?: unknown;
  readonly timeUnixNano?: unknown;
  readonly observedTimestamp?: unknown;
}

interface JsonlExporter {
  readonly sinkDirectory: string;
  exportRecord(record: TelemetryRecordInput): void | Promise<void>;
  forceFlush(): void | Promise<void>;
  shutdown(): void | Promise<void>;
}

interface JsonlExporterOptions {
  readonly sinkDirectory?: string;
  readonly signal?: TelemetrySignal;
  readonly warn?: (message: string) => void;
}

interface ExporterModule {
  createJsonlExporter(
    options?: JsonlExporterOptions,
  ): JsonlExporter | Promise<JsonlExporter>;
}

interface BootstrapModule {
  loadOptionalPeer(
    specifier: string,
    warn?: (message: string) => void,
  ): Promise<unknown | null>;
  checkNodeFloor(
    nodeVersion?: string,
  ): { readonly supported: boolean; readonly reason: string | null };
}

/**
 * The sanitiser module, with its `allowedAttributeNames` INJECTION SEAM.
 *
 * The seam is the anti-vacuity lever: a name that is BOTH allow-listed AND
 * forbidden is the ONLY way to reach the deny pass, because the allow-list
 * of every signal holds `neat.telemetry.*` names and none of them is
 * forbidden.
 */
interface SanitizeModule {
  sanitizeAttributes(
    attributes: Record<string, unknown> | undefined | null,
    options?: {
      readonly allowedAttributeNames?: readonly string[];
      readonly signal?: TelemetrySignal;
      readonly warn?: (message: string) => void;
      readonly strict?: boolean;
    },
  ): {
    readonly attributes: Record<string, unknown>;
    readonly dropped: string[];
  };
}

/**
 * The module specifiers are typed as `string` on purpose: the compiler must
 * not try to resolve them, because the modules do not exist yet and a
 * resolution error at compile time would be an import failure rather than a
 * behavioural one.
 */
const POLICY_MODULE_SPECIFIER: string = './telemetry.policy';
const EXPORTER_MODULE_SPECIFIER: string = './telemetry.jsonl-exporters';
const BOOTSTRAP_MODULE_SPECIFIER: string = './telemetry.bootstrap';
const SANITIZE_MODULE_SPECIFIER: string = './telemetry.sanitize';

/**
 * The warning shape: a warning is a non-empty string, and it carries NO
 * absolute path and NO calendar date - a warning is persisted in a log like
 * any other artefact, so the privacy rules apply to it too.
 */
const ABSOLUTE_PATH_PATTERN =
  /[A-Za-z]:[\\/]|\/(?:Users|home|tmp|var|etc|opt|private)\//;
const CALENDAR_DATE_PATTERN = /\d{4}-\d{2}-\d{2}/;

/**
 * Walks up from the current working directory to the repository root, so the
 * home does not depend on `__dirname` (which an ES module does not have) nor
 * on the working directory the runner happened to choose.
 *
 * @returns The absolute path of the repository root.
 */
function findRepoRoot(): string {
  let current = process.cwd();
  for (let depth = 0; depth < 8; depth += 1) {
    if (existsSync(path.join(current, 'package.json'))) return current;
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  throw new Error('could not locate the repository root');
}

const REPO_ROOT = findRepoRoot();

/**
 * Loads a module lazily. A resolution failure is turned into an ASSERTION
 * failure: the lane then reports "the module is missing" as a behavioural
 * result instead of crashing the suite with an import error.
 *
 * @param specifier The module specifier.
 * @returns The loaded module.
 */
async function loadModule<T>(specifier: string): Promise<T> {
  let loaded: unknown;
  let failure: string | null = null;
  try {
    loaded = await import(specifier);
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  }
  expect({ specifier, failure }).toEqual({ specifier, failure: null });
  return loaded as T;
}

/**
 * Reads every file under a sink directory and joins their contents.
 *
 * @param directory The sink directory.
 * @returns The concatenated sink text.
 */
function readSinkText(directory: string): string {
  if (!existsSync(directory)) return '';
  const chunks: string[] = [];
  const walk = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) chunks.push(readFileSync(full, 'utf8'));
    }
  };
  walk(directory);
  return chunks.join('\n');
}

/**
 * Parses the JSONL sink into records.
 *
 * @param directory The sink directory.
 * @returns The parsed records.
 */
function readSinkRecords(directory: string): Record<string, unknown>[] {
  return readSinkText(directory)
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

/**
 * Creates a throwaway sink directory that does NOT exist yet, so the
 * "created on demand" behaviour is exercised for real.
 *
 * @returns The absolute path of the sink directory.
 */
function createMissingSinkDirectory(): string {
  const root = mkdtempSync(path.join(tmpdir(), 'telemetry-sink-'));
  return path.join(root, 'nested', 'sink');
}

interface SignalFixture {
  readonly exporter: JsonlExporter;
  readonly sinkDirectory: string;
  readonly warnings: string[];
  readonly recordName: string;
  readonly attributeName: string;
  readonly otherAttributeName: string;
}

/**
 * Creates the REAL exporter over a REAL sink directory, and reads the
 * allow-listed names from the REAL policy so the lanes never hard-code an
 * attribute name that the policy does not actually allow.
 *
 * @param signal The telemetry signal.
 * @param sinkDirectory The sink directory.
 * @returns The fixture.
 */
async function createSignalFixture(
  signal: TelemetrySignal,
  sinkDirectory: string,
): Promise<SignalFixture> {
  const [exporters, policy] = await Promise.all([
    loadModule<ExporterModule>(EXPORTER_MODULE_SPECIFIER),
    loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER),
  ]);
  const warnings: string[] = [];
  const exporter = await exporters.createJsonlExporter({
    sinkDirectory,
    signal,
    warn: (message: string) => {
      warnings.push(message);
    },
  });
  const entry = policy.TELEMETRY_ALLOW_LIST[signal];
  return {
    exporter,
    sinkDirectory,
    warnings,
    recordName: entry.recordNames[0],
    attributeName: entry.attributeNames[0],
    otherAttributeName:
      entry.attributeNames.length > 1
        ? entry.attributeNames[1]
        : entry.attributeNames[0],
  };
}

/**
 * Exports one record and returns the parsed sink records.
 *
 * @param fixture The fixture.
 * @param record The record.
 * @returns The parsed sink records.
 */
async function exportAndRead(
  fixture: SignalFixture,
  record: TelemetryRecordInput,
): Promise<Record<string, unknown>[]> {
  await fixture.exporter.exportRecord(record);
  await fixture.exporter.forceFlush();
  return readSinkRecords(fixture.sinkDirectory);
}

describe('OT-V1-1: the telemetry exporter (vertical slice A)', () => {
  describe('the lazy module loader', () => {
    it('resolves a REAL TypeScript module through the same mechanism', async () => {
      // The loader is the mechanism every other lane depends on, so it is
      // proven on a module that EXISTS: if this lane cannot load a real
      // module, every lane below it would be vacuous.
      const specifier: string = '../../acceleration/acceleration.types';
      const loaded = await loadModule<unknown>(specifier);
      expect(loaded).not.toBeUndefined();
    });

    it('reports a missing module as a resolution failure, never a silent success', async () => {
      const specifier: string = './telemetry.module-that-does-not-exist';
      let failure: string | null = null;
      try {
        await import(specifier);
      } catch (error) {
        failure = error instanceof Error ? error.message : String(error);
      }
      expect(failure).not.toBeNull();
    });
  });

  describe('the policy module', () => {
    it('exposes the telemetry sink root, the node floor and the reason token', async () => {
      const policy = await loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER);
      expect(policy.TELEMETRY_SINK_ROOT.replace(/\\/g, '/')).toBe(
        'logs/telemetry',
      );
      expect(policy.TELEMETRY_NODE_FLOOR).toContain('22.15.0');
      expect(policy.REASON_NODE_BELOW_TELEMETRY_FLOOR).toBe(
        'node_below_telemetry_floor',
      );
    });

    it('exposes an allow-list per signal with record names and attribute names', async () => {
      const policy = await loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER);
      const allowList = policy.TELEMETRY_ALLOW_LIST;
      expect(Object.keys(allowList).sort()).toEqual(['log', 'metric', 'trace']);
      for (const signal of ['trace', 'metric', 'log'] as const) {
        expect(Array.isArray(allowList[signal].recordNames)).toBe(true);
        expect(Array.isArray(allowList[signal].attributeNames)).toBe(true);
        expect(allowList[signal].recordNames.length).toBeGreaterThan(0);
        expect(allowList[signal].attributeNames.length).toBeGreaterThan(0);
      }
    });

    it('exposes no neat.acceleration instrument in the metric allow-list', async () => {
      const policy = await loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER);
      const metric = policy.TELEMETRY_ALLOW_LIST.metric;
      const acceleration = [
        ...metric.recordNames,
        ...metric.attributeNames,
      ].filter((name) => name.startsWith('neat.acceleration'));
      expect(acceleration).toEqual([]);
    });

    it('carries the SAME forbidden keys and value patterns as the telemetry contract', async () => {
      const policy = await loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER);
      const contractUrl = pathToFileURL(
        path.join(
          REPO_ROOT,
          'scripts',
          'telemetry',
          'telemetry-contract.mjs',
        ),
      ).href;
      // The contract module is a `.mjs` file, so it is read through a child
      // process rather than imported: a `.mjs` import from a TypeScript test
      // is a module-resolution failure of the test's own making.
      const script = [
        'const mod = await import(',
        JSON.stringify(contractUrl),
        ');',
        'process.stdout.write(JSON.stringify({',
        '  keys: [...mod.FORBIDDEN_KEYS].sort(),',
        '  patterns: mod.FORBIDDEN_VALUE_PATTERNS',
        '    .map((pattern) => ({ source: pattern.source, flags: pattern.flags }))',
        '    .sort((left, right) => left.source.localeCompare(right.source)),',
        '}));',
      ].join('');
      const probe = spawnSync(
        process.execPath,
        ['--input-type=module', '-e', script],
        { cwd: REPO_ROOT, encoding: 'utf8' },
      );
      expect({ exitCode: probe.status }).toEqual({ exitCode: 0 });
      const contract = JSON.parse(probe.stdout) as {
        keys: string[];
        patterns: { source: string; flags: string }[];
      };
      expect({
        keys: [...policy.FORBIDDEN_KEYS].sort(),
        patterns: [...policy.FORBIDDEN_VALUE_PATTERNS]
          .map((pattern) => ({
            source: pattern.source,
            flags: pattern.flags,
          }))
          .sort((left, right) => left.source.localeCompare(right.source)),
      }).toEqual({
        keys: contract.keys,
        patterns: contract.patterns,
      });
    });
  });

  describe('the JSONL exporter', () => {
    it('creates the sink directory on demand when a record is exported', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      expect(existsSync(sinkDirectory)).toBe(false);
      const fixture = await createSignalFixture('trace', sinkDirectory);
      await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: { [fixture.attributeName]: 'probe' },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      expect(existsSync(sinkDirectory)).toBe(true);
      expect(readSinkText(sinkDirectory).length).toBeGreaterThan(0);
    });

    it('resolves the default sink directory under the telemetry sink root', async () => {
      const exporters = await loadModule<ExporterModule>(
        EXPORTER_MODULE_SPECIFIER,
      );
      const exporter = await exporters.createJsonlExporter({ signal: 'trace' });
      expect(
        exporter.sinkDirectory.replace(/\\/g, '/').endsWith('logs/telemetry'),
      ).toBe(true);
    });

    it('writes no forbidden key name into the sink', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: {
          [fixture.attributeName]: 'probe',
          startTime: '2024-01-01T00:00:00.000Z',
          endTime: '2024-01-01T00:00:01.000Z',
          timeUnixNano: '1704067200000000000',
          observedTimestamp: '2024-01-01T00:00:00.000Z',
          timestamp: '2024-01-01T00:00:00.000Z',
        },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      const text = readSinkText(sinkDirectory);
      for (const forbidden of [
        'startTime',
        'endTime',
        'timeUnixNano',
        'observedTimestamp',
        'timestamp',
      ]) {
        expect({ forbidden, present: text.includes(forbidden) }).toEqual({
          forbidden,
          present: false,
        });
      }
    });

    it('writes no calendar date and no epoch into the sink', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: { [fixture.attributeName]: 'probe' },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      const text = readSinkText(sinkDirectory);
      expect(text).not.toMatch(/\d{4}-\d{2}-\d{2}/);
      expect(text).not.toMatch(/\b\d{13}\b/);
      expect(text).not.toMatch(/\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}/);
    });

    it('writes exactly ONE duration field, derived from the monotonic interval', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      const records = await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: { [fixture.attributeName]: 'probe' },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      expect(records).toHaveLength(1);
      const record = records[0];
      const durationKeys = Object.keys(record).filter((key) =>
        key.toLowerCase().includes('duration'),
      );
      expect(durationKeys).toEqual(['duration_ms']);
      expect(record.duration_ms).toBe(250);
    });

    it('ignores a caller supplied duration and still writes one derived duration', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      const records = await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: { [fixture.attributeName]: 'probe' },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
        durationMs: 999999,
      });
      expect(records).toHaveLength(1);
      const record = records[0];
      const durationKeys = Object.keys(record).filter((key) =>
        key.toLowerCase().includes('duration'),
      );
      expect(durationKeys).toEqual(['duration_ms']);
      expect(record.duration_ms).toBe(250);
    });

    it('keeps an allow-listed attribute and drops an attribute outside the allow-list', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      const records = await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: {
          [fixture.attributeName]: 'kept',
          'not.in.the.allow.list': 'dropped',
        },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      expect(records).toHaveLength(1);
      const serialized = JSON.stringify(records[0]);
      expect(serialized).toContain('kept');
      expect(serialized).not.toContain('not.in.the.allow.list');
      expect(serialized).not.toContain('dropped');
    });

    it('drops an absolute path value even on an allow-listed attribute', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      const records = await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: {
          [fixture.attributeName]: 'C:\\Users\\someone\\secret.txt',
        },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      const serialized = JSON.stringify(records);
      expect(serialized).not.toContain('secret.txt');
    });

    it('drops a non-scalar attribute value even on an allow-listed attribute', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      const records = await exportAndRead(fixture, {
        name: fixture.recordName,
        attributes: {
          [fixture.otherAttributeName]: {
            command: 'rm -rf',
            content: 'SECRETFILECONTENT',
          },
        },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      const serialized = JSON.stringify(records);
      expect(serialized).not.toContain('SECRETFILECONTENT');
      expect(serialized).not.toContain('rm -rf');
    });

    it('writes nothing for a record the allow-list does not cover', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('metric', sinkDirectory);
      await exportAndRead(fixture, {
        name: 'neat.acceleration.inference_ms',
        attributes: { [fixture.attributeName]: 'probe' },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      expect(readSinkText(sinkDirectory).trim()).toBe('');
    });

    it('writes exactly one JSON record per exported record', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      await fixture.exporter.exportRecord({
        name: fixture.recordName,
        attributes: { [fixture.attributeName]: 'first' },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      await fixture.exporter.exportRecord({
        name: fixture.recordName,
        attributes: { [fixture.attributeName]: 'second' },
        interval: { startMonotonicMs: 2000, endMonotonicMs: 2250 },
      });
      await fixture.exporter.forceFlush();
      const lines = readSinkText(sinkDirectory)
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line.length > 0);
      expect(lines).toHaveLength(2);
      for (const line of lines) {
        expect(() => JSON.parse(line)).not.toThrow();
      }
    });

    it('warns and never throws when the sink cannot be written', async () => {
      const root = mkdtempSync(path.join(tmpdir(), 'telemetry-blocked-'));
      const blocked = path.join(root, 'blocked');
      writeFileSync(blocked, 'this is a file, not a directory', 'utf8');
      // The allow-listed names are read from the REAL policy, and the REAL
      // exporter is constructed INSIDE the try block: an exporter that
      // creates the sink eagerly must warn instead of throwing.
      const [exporters, policy] = await Promise.all([
        loadModule<ExporterModule>(EXPORTER_MODULE_SPECIFIER),
        loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER),
      ]);
      const entry = policy.TELEMETRY_ALLOW_LIST.trace;
      const warnings: string[] = [];
      let thrown: string | null = null;
      let exported = false;
      try {
        const exporter = await exporters.createJsonlExporter({
          sinkDirectory: blocked,
          signal: 'trace',
          warn: (message: string) => {
            warnings.push(message);
          },
        });
        await exporter.exportRecord({
          name: entry.recordNames[0],
          attributes: { [entry.attributeNames[0]]: 'probe' },
          interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
        });
        await exporter.forceFlush();
        exported = true;
      } catch (error) {
        thrown = error instanceof Error ? error.message : String(error);
      }
      expect(thrown).toBeNull();
      expect(exported).toBe(true);
      expect(warnings.length).toBeGreaterThan(0);
      for (const warning of warnings) {
        expect(warning.length).toBeGreaterThan(0);
        expect(warning).not.toMatch(ABSOLUTE_PATH_PATTERN);
        expect(warning).not.toMatch(CALENDAR_DATE_PATTERN);
      }
    });

    it('warns and never throws when an attribute accessor throws', async () => {
      const sinkDirectory = createMissingSinkDirectory();
      const fixture = await createSignalFixture('trace', sinkDirectory);
      const attributes: Record<string, unknown> = {};
      Object.defineProperty(attributes, fixture.attributeName, {
        enumerable: true,
        get() {
          throw new Error('attribute accessor exploded');
        },
      });
      let thrown: string | null = null;
      try {
        await fixture.exporter.exportRecord({
          name: fixture.recordName,
          attributes,
          interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
        });
        await fixture.exporter.forceFlush();
      } catch (error) {
        thrown = error instanceof Error ? error.message : String(error);
      }
      expect(thrown).toBeNull();
      expect(fixture.warnings.length).toBeGreaterThan(0);
      for (const warning of fixture.warnings) {
        expect(warning.length).toBeGreaterThan(0);
        expect(warning).not.toMatch(ABSOLUTE_PATH_PATTERN);
        expect(warning).not.toMatch(CALENDAR_DATE_PATTERN);
      }
    });
  });

  describe('the optional-peer loader', () => {
    it('returns null and warns when the peer is missing', async () => {
      const bootstrap = await loadModule<BootstrapModule>(
        BOOTSTRAP_MODULE_SPECIFIER,
      );
      const warnings: string[] = [];
      const peer = await bootstrap.loadOptionalPeer(
        '@opentelemetry/definitely-not-installed',
        (message: string) => {
          warnings.push(message);
        },
      );
      expect(peer).toBeNull();
      expect(warnings.length).toBeGreaterThan(0);
      for (const warning of warnings) {
        expect(warning.length).toBeGreaterThan(0);
        expect(warning).not.toMatch(ABSOLUTE_PATH_PATTERN);
        expect(warning).not.toMatch(CALENDAR_DATE_PATTERN);
      }
    });

    it('loads an installed optional peer', async () => {
      const bootstrap = await loadModule<BootstrapModule>(
        BOOTSTRAP_MODULE_SPECIFIER,
      );
      const peer = await bootstrap.loadOptionalPeer('@opentelemetry/api');
      expect(peer).not.toBeNull();
      expect(peer).toBeTruthy();
    });
  });

  describe('the node floor guard', () => {
    it('reports the machine-stable reason token below the telemetry floor', async () => {
      const [bootstrap, policy] = await Promise.all([
        loadModule<BootstrapModule>(BOOTSTRAP_MODULE_SPECIFIER),
        loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER),
      ]);
      const supported = bootstrap.checkNodeFloor('22.15.0');
      expect(supported.supported).toBe(true);
      expect(supported.reason).toBeNull();
      const unsupported = bootstrap.checkNodeFloor('22.14.0');
      expect(unsupported.supported).toBe(false);
      expect(unsupported.reason).toBe(
        policy.REASON_NODE_BELOW_TELEMETRY_FLOOR,
      );
    });
  });

  describe('the allow-list-injection anti-vacuity lane', () => {
    it('drops a forbidden key that was INJECTED INTO THE ALLOW-LIST', async () => {
      const sanitize = await loadModule<SanitizeModule>(
        SANITIZE_MODULE_SPECIFIER,
      );
      // THE VALUE IS DELIBERATELY BENIGN: a forbidden VALUE would be dropped by
      // the value pass, so the lane would stay green even when the deny pass
      // was gone. Only the deny pass can drop a forbidden KEY whose value
      // carries no forbidden shape, so this lane is a witness of the deny pass.
      const result = sanitize.sanitizeAttributes(
        {
          timestamp: 'later',
          'probe.attribute': 'kept',
        },
        { allowedAttributeNames: ['timestamp', 'probe.attribute'] },
      );
      expect(result.attributes).not.toHaveProperty('timestamp');
      expect(result.dropped).toContain('timestamp');
      expect(result.attributes['probe.attribute']).toBe('kept');
    });

    it('drops a reserved record key that was INJECTED INTO THE ALLOW-LIST', async () => {
      const sanitize = await loadModule<SanitizeModule>(
        SANITIZE_MODULE_SPECIFIER,
      );
      const result = sanitize.sanitizeAttributes(
        { duration_ms: 999999, name: 'hijacked' },
        { allowedAttributeNames: ['duration_ms', 'name'] },
      );
      expect(result.attributes).not.toHaveProperty('duration_ms');
      expect(result.attributes).not.toHaveProperty('name');
      expect(result.dropped).toEqual(
        expect.arrayContaining(['duration_ms', 'name']),
      );
    });

    it('drops a forbidden VALUE on a name that was INJECTED INTO THE ALLOW-LIST', async () => {
      const sanitize = await loadModule<SanitizeModule>(
        SANITIZE_MODULE_SPECIFIER,
      );
      const result = sanitize.sanitizeAttributes(
        { 'probe.attribute': '2024-01-01T00:00:00.000Z' },
        { allowedAttributeNames: ['probe.attribute'] },
      );
      expect(result.attributes).not.toHaveProperty('probe.attribute');
      expect(result.dropped).toContain('probe.attribute');
    });

    it('keeps an ordinary name that was INJECTED INTO THE ALLOW-LIST', async () => {
      const sanitize = await loadModule<SanitizeModule>(
        SANITIZE_MODULE_SPECIFIER,
      );
      const result = sanitize.sanitizeAttributes(
        { 'probe.attribute': 'probe' },
        { allowedAttributeNames: ['probe.attribute'] },
      );
      expect(result.attributes['probe.attribute']).toBe('probe');
      expect(result.dropped).toEqual([]);
    });
  });

  describe('the REAL local JSONL the privacy sweep enumerates', () => {
    it('writes a sanitized JSONL into the REAL sink root and the repo contract scanner finds NO violation in it', async () => {
      const [exporters, policy] = await Promise.all([
        loadModule<ExporterModule>(EXPORTER_MODULE_SPECIFIER),
        loadModule<PolicyModule>(POLICY_MODULE_SPECIFIER),
      ]);
      const defaultExporter = await exporters.createJsonlExporter({
        signal: 'trace',
      });
      const sinkDirectory = defaultExporter.sinkDirectory;
      expect(sinkDirectory.replace(/\\/g, '/')).toMatch(/\/logs\/telemetry$/);

      for (const signal of ['trace', 'metric', 'log'] as const) {
        const exporter = await exporters.createJsonlExporter({ signal });
        const entry = policy.TELEMETRY_ALLOW_LIST[signal];
        await exporter.exportRecord({
          name: entry.recordNames[0],
          attributes: { [entry.attributeNames[0]]: signal },
          ...(signal === 'trace'
            ? { interval: { startMonotonicMs: 0, endMonotonicMs: 250 } }
            : {}),
        });
        await exporter.forceFlush();
      }

      const files = readdirSync(sinkDirectory, { withFileTypes: true })
        .filter((entry) => entry.isFile() && entry.name.endsWith('.jsonl'))
        .map((entry) => path.join(sinkDirectory, entry.name))
        .sort();
      expect(files.length).toBeGreaterThanOrEqual(3);
      for (const file of files) {
        const text = readFileSync(file, 'utf8');
        const lines = text
          .split('\n')
          .map((line) => line.trim())
          .filter((line) => line.length > 0);
        expect(lines.length).toBeGreaterThan(0);
        for (const line of lines) {
          expect(() => JSON.parse(line)).not.toThrow();
          expect(line).not.toMatch(/\d{4}-\d{2}-\d{2}/);
          expect(line).not.toMatch(/\b\d{13}\b/);
          expect(line).not.toMatch(ABSOLUTE_PATH_PATTERN);
        }
      }

      const traceRecords = readFileSync(
        path.join(sinkDirectory, 'trace.jsonl'),
        'utf8',
      )
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line) as Record<string, unknown>);
      const lastTrace = traceRecords[traceRecords.length - 1];
      expect(lastTrace.name).toBe('neat.telemetry.trace');
      expect(lastTrace.duration_ms).toBe(250);
      expect(
        Object.keys(lastTrace).filter((key) => key.toLowerCase().includes('duration')),
      ).toEqual(['duration_ms']);

      const contractUrl = pathToFileURL(
        path.join(REPO_ROOT, 'scripts', 'telemetry', 'telemetry-contract.mjs'),
      ).href;
      const script = [
        'const mod = await import(',
        JSON.stringify(contractUrl),
        ');',
        'const { readFileSync } = await import("node:fs");',
        'const files = ',
        JSON.stringify(files),
        ';',
        'const violations = [];',
        'for (const file of files) {',
        '  for (const violation of mod.scanPersistedText(readFileSync(file, "utf8"))) {',
        '    violations.push({ file, kind: violation.kind, key: violation.key, pattern: violation.pattern, sample: violation.sample, line: violation.line });',
        '  }',
        '}',
        'process.stdout.write(JSON.stringify({ violations }));',
      ].join('');
      const probe = spawnSync(
        process.execPath,
        ['--input-type=module', '-e', script],
        { cwd: REPO_ROOT, encoding: 'utf8' },
      );
      expect({ exitCode: probe.status }).toEqual({ exitCode: 0 });
      const scanned = JSON.parse(probe.stdout) as { violations: unknown[] };
      expect(scanned.violations).toEqual([]);
    });
  });
});
