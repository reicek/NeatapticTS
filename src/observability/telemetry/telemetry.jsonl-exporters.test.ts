/**
 * GREEN home for the local JSONL exporters (slice OT-V1-2 of
 * `plans/opentelemetry-slice-telemetry.plans.md`).
 *
 * WHAT THIS HOME DEMANDS: the three local exporters are the ONLY thing
 * layer A writes to disk, so every rule they declare is proved here over
 * the REAL module:
 *
 *   - the default sink directory is the telemetry sink root under the
 *     repository root, and it is resolved from a working directory that is
 *     the repository root, a DEEP SUBDIRECTORY of it and the DRIVE ROOT -
 *     so the sink never depends on the directory a runner happened to
 *     choose;
 *   - an exporter that is constructed with NO options writes the `trace`
 *     signal, and its default warning sink reaches `console.warn`;
 *   - a record the allow-list does not cover is NOT written at all, and it
 *     warns with the machine-stable token;
 *   - the monotonic interval becomes the single `duration_ms` field, and a
 *     record with NO interval, with a null interval, with a negative
 *     duration, with a non-finite bound or with an overflowing duration
 *     carries NO duration field at all;
 *   - a sink that cannot be created or written warns with the
 *     machine-stable token and never throws into the caller;
 *   - an exporter that was shut down writes NOTHING, and the shutdown is
 *     what closes it.
 *
 * WHY EVERY SINK IS A REAL TEMPORARY DIRECTORY: the exporter creates its
 * sink on demand, so a lane that asserted against a mock could not prove
 * that the directory is created exactly when the first record is written.
 *
 * @see plans/opentelemetry-slice-telemetry.plans.md - "Vertical slice A"
 */

import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import {
  REASON_RECORD_NOT_ALLOWED,
  REASON_SINK_UNWRITABLE,
  createJsonlExporter,
  resolveDefaultSinkDirectory,
} from './telemetry.jsonl-exporters';

/**
 * Walks up from the working directory to the repository root, so the home
 * does not depend on the directory the runner happened to choose.
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
 * Creates a throwaway sink directory that does NOT exist yet, so the
 * "created on demand" behaviour is exercised for real.
 *
 * @returns The absolute path of the sink directory.
 */
function createTempSinkDirectory(): string {
  const root = mkdtempSync(path.join(tmpdir(), 'telemetry-exporter-'));
  return path.join(root, 'nested', 'sink');
}

/**
 * Reads every JSONL record of a sink directory.
 *
 * @param directory The sink directory.
 * @returns The parsed records.
 */
function readSinkRecords(directory: string): Record<string, unknown>[] {
  if (!existsSync(directory)) return [];
  const records: Record<string, unknown>[] = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith('.jsonl')) continue;
    const text = readFileSync(path.join(directory, entry.name), 'utf8');
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.length === 0) continue;
      records.push(JSON.parse(trimmed) as Record<string, unknown>);
    }
  }
  return records;
}

/**
 * The POSIX-style form of a path, so the lanes compare the SAME shape on
 * every platform.
 *
 * @param value The path.
 * @returns The path with forward slashes.
 */
function toPosix(value: string): string {
  return value.replace(/\\/g, '/');
}

/**
 * Lists the files of a sink directory, so a lane can prove that an
 * exporter wrote NOTHING without depending on the directory being empty.
 *
 * @param directory The sink directory.
 * @returns The sorted file names.
 */
function listSinkFiles(directory: string): string[] {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort();
}

/**
 * Creates a throwaway repository root whose telemetry sink root exists but
 * whose local settings do NOT, so a lane can write settings of its own.
 *
 * @returns The absolute path of the repository root.
 */
function createTempRepositoryRoot(): string {
  const root = mkdtempSync(path.join(tmpdir(), 'telemetry-settings-root-'));
  mkdirSync(path.join(root, 'logs', 'telemetry'), { recursive: true });
  return root;
}

/**
 * Writes the local settings of a throwaway repository root.
 *
 * @param root The repository root.
 * @param content The raw file content.
 */
function writeLocalSettings(root: string, content: string): void {
  const settingsPath = path.join(root, 'logs', 'telemetry', 'settings.json');
  writeFileSync(settingsPath, content, 'utf8');
}

describe('OT-V1-2: the local JSONL exporters', () => {
  describe('the default sink directory', () => {
    it('resolves the telemetry sink root under the repository root', () => {
      const cwd = jest.spyOn(process, 'cwd').mockReturnValue(REPO_ROOT);
      try {
        expect(toPosix(resolveDefaultSinkDirectory())).toBe(
          `${toPosix(REPO_ROOT)}/logs/telemetry`,
        );
      } finally {
        cwd.mockRestore();
      }
    });

    it('walks up to the repository root from a deep subdirectory', () => {
      const cwd = jest
        .spyOn(process, 'cwd')
        .mockReturnValue(path.join(REPO_ROOT, 'src', 'observability', 'telemetry'));
      try {
        expect(toPosix(resolveDefaultSinkDirectory())).toBe(
          `${toPosix(REPO_ROOT)}/logs/telemetry`,
        );
      } finally {
        cwd.mockRestore();
      }
    });

    it('falls back to the working directory at the drive root', () => {
      const driveRoot = path.parse(process.cwd()).root;
      const cwd = jest.spyOn(process, 'cwd').mockReturnValue(driveRoot);
      try {
        expect(toPosix(resolveDefaultSinkDirectory())).toBe(
          `${toPosix(driveRoot)}logs/telemetry`,
        );
      } finally {
        cwd.mockRestore();
      }
    });
  });

  describe('the exporter built with no options at all', () => {
    it('targets the telemetry sink root', () => {
      const exporter = createJsonlExporter();
      expect(toPosix(exporter.sinkDirectory)).toBe(
        `${toPosix(REPO_ROOT)}/logs/telemetry`,
      );
    });

    it('warns through the default warning sink when the record is not allowed', () => {
      const warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
      try {
        const exporter = createJsonlExporter();
        const before = listSinkFiles(exporter.sinkDirectory);
        exporter.exportRecord({ name: 'not.allowed' });
        expect(warn).toHaveBeenCalledWith(REASON_RECORD_NOT_ALLOWED);
        expect(listSinkFiles(exporter.sinkDirectory)).toEqual(before);
      } finally {
        warn.mockRestore();
      }
    });
  });

  describe('the record that is written', () => {
    it('creates the sink directory on demand and derives the single duration field', () => {
      const sinkDirectory = createTempSinkDirectory();
      expect(existsSync(sinkDirectory)).toBe(false);
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        attributes: { 'neat.telemetry.signal': 'trace' },
        interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
      });
      expect(existsSync(sinkDirectory)).toBe(true);
      expect(readSinkRecords(sinkDirectory)).toEqual([
        {
          name: 'neat.telemetry.trace',
          duration_ms: 250,
          'neat.telemetry.signal': 'trace',
        },
      ]);
    });

    it('writes the signal the exporter was built for', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({
        sinkDirectory,
        signal: 'metric',
        enabled: true,
      });
      exporter.exportRecord({
        name: 'neat.telemetry.metric',
        attributes: { 'neat.telemetry.signal': 'metric' },
        interval: { startMonotonicMs: 0, endMonotonicMs: 5 },
      });
      expect(listSinkFiles(sinkDirectory)).toEqual(['metric.jsonl']);
      expect(readSinkRecords(sinkDirectory)).toEqual([
        {
          name: 'neat.telemetry.metric',
          duration_ms: 5,
          'neat.telemetry.signal': 'metric',
        },
      ]);
    });

    it('writes NO duration field when the record carries no interval', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      exporter.exportRecord({ name: 'neat.telemetry.trace' });
      const records = readSinkRecords(sinkDirectory);
      expect(records).toHaveLength(1);
      expect(records[0]).not.toHaveProperty('duration_ms');
    });

    it('writes NO duration field when the interval is null', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: null as unknown as {
          startMonotonicMs: number;
          endMonotonicMs: number;
        },
      });
      const records = readSinkRecords(sinkDirectory);
      expect(records).toHaveLength(1);
      expect(records[0]).not.toHaveProperty('duration_ms');
    });

    it('writes NO duration field when the duration is negative', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: 100, endMonotonicMs: 50 },
      });
      const records = readSinkRecords(sinkDirectory);
      expect(records).toHaveLength(1);
      expect(records[0]).not.toHaveProperty('duration_ms');
    });

    it('writes NO duration field when a bound of the interval is not finite', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: Number.NaN, endMonotonicMs: 50 },
      });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: 0, endMonotonicMs: Number.POSITIVE_INFINITY },
      });
      const records = readSinkRecords(sinkDirectory);
      expect(records).toHaveLength(2);
      for (const record of records) {
        expect(record).not.toHaveProperty('duration_ms');
      }
    });

    it('writes NO duration field when the derived duration overflows', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: -1e308, endMonotonicMs: 1e308 },
      });
      const records = readSinkRecords(sinkDirectory);
      expect(records).toHaveLength(1);
      expect(records[0]).not.toHaveProperty('duration_ms');
    });

    it('writes nothing at all for a record the allow-list does not cover', () => {
      const sinkDirectory = createTempSinkDirectory();
      const warnings: string[] = [];
      const exporter = createJsonlExporter({
        sinkDirectory,
        enabled: true,
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      exporter.exportRecord({ name: 'not.allowed' });
      expect(warnings).toEqual([REASON_RECORD_NOT_ALLOWED]);
      expect(readSinkRecords(sinkDirectory)).toEqual([]);
    });

    it('flushes without throwing and without writing anything', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      expect(() => exporter.forceFlush()).not.toThrow();
      expect(readSinkRecords(sinkDirectory)).toEqual([]);
    });
  });

  describe('the sink that cannot be written', () => {
    it('warns with the machine-stable token and never throws', () => {
      const root = mkdtempSync(path.join(tmpdir(), 'telemetry-blocked-'));
      const blocked = path.join(root, 'blocked');
      writeFileSync(blocked, 'this is a file, not a directory', 'utf8');
      const warnings: string[] = [];
      const exporter = createJsonlExporter({
        sinkDirectory: blocked,
        enabled: true,
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      expect(() =>
        exporter.exportRecord({
          name: 'neat.telemetry.trace',
          interval: { startMonotonicMs: 0, endMonotonicMs: 1 },
        }),
      ).not.toThrow();
      expect(warnings).toEqual([REASON_SINK_UNWRITABLE]);
    });
  });

  describe('the exporter that was shut down', () => {
    it('writes NOTHING after the shutdown', () => {
      const sinkDirectory = createTempSinkDirectory();
      const exporter = createJsonlExporter({ sinkDirectory, enabled: true });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: 0, endMonotonicMs: 1 },
      });
      expect(readSinkRecords(sinkDirectory)).toHaveLength(1);
      exporter.shutdown();
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: 0, endMonotonicMs: 1 },
      });
      expect(readSinkRecords(sinkDirectory)).toHaveLength(1);
    });
  });

  describe('the local settings that govern the sinks', () => {
    it('takes the sink root from the exporter path of the local settings', () => {
      const root = createTempRepositoryRoot();
      writeLocalSettings(root, JSON.stringify({ exporter: { path: '.' } }));
      const exporter = createJsonlExporter({ repositoryRoot: root });
      expect(toPosix(exporter.sinkDirectory)).toBe(
        `${toPosix(root)}/logs/telemetry`,
      );
    });

    it('takes the sink root from a relative exporter path of the local settings', () => {
      const root = createTempRepositoryRoot();
      writeLocalSettings(root, JSON.stringify({ exporter: { path: 'sinks' } }));
      const exporter = createJsonlExporter({ repositoryRoot: root });
      expect(toPosix(exporter.sinkDirectory)).toBe(
        `${toPosix(root)}/logs/telemetry/sinks`,
      );
    });

    it('refuses an absolute exporter path and uses the declared default root', () => {
      const root = createTempRepositoryRoot();
      writeLocalSettings(
        root,
        JSON.stringify({ exporter: { path: path.join(root, 'elsewhere') } }),
      );
      const exporter = createJsonlExporter({ repositoryRoot: root });
      expect(toPosix(exporter.sinkDirectory)).toBe(
        `${toPosix(root)}/logs/telemetry`,
      );
    });

    it('writes NOTHING when the local settings disable the layer', () => {
      const root = createTempRepositoryRoot();
      writeLocalSettings(root, JSON.stringify({ enabled: false }));
      const exporter = createJsonlExporter({ repositoryRoot: root });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: 0, endMonotonicMs: 1 },
      });
      expect(
        listSinkFiles(exporter.sinkDirectory).filter((name) =>
          name.endsWith('.jsonl'),
        ),
      ).toEqual([]);
      expect(readSinkRecords(exporter.sinkDirectory)).toEqual([]);
    });

    it('writes into the sink when the local settings enable the layer', () => {
      const root = createTempRepositoryRoot();
      writeLocalSettings(root, JSON.stringify({ enabled: true }));
      const exporter = createJsonlExporter({ repositoryRoot: root });
      exporter.exportRecord({
        name: 'neat.telemetry.trace',
        interval: { startMonotonicMs: 0, endMonotonicMs: 1 },
      });
      expect(readSinkRecords(exporter.sinkDirectory)).toHaveLength(1);
    });
  });
});
