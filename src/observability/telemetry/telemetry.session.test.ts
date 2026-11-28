/**
 * GREEN home for the layer-A session lifecycle of the local OpenTelemetry
 * signal layer (slice OT-V1-2 of
 * `plans/opentelemetry-slice-telemetry.plans.md`).
 *
 * WHAT THIS HOME DEMANDS, over the REAL module:
 *
 *   - a session hands the sanitized snapshot it was given to the three local
 *     exporters, one file per signal, and it NEVER writes a record the
 *     snapshot did not carry;
 *   - a session that starts on a runtime BELOW the telemetry floor is
 *     warn-only: it reports the machine-stable token and loads no peer;
 *   - a session whose peers are missing is DEGRADED, not broken, and it
 *     reports the machine-stable peer token;
 *   - an emission, a flush and a shutdown that FAIL warn with the
 *     machine-stable token and never throw into the caller;
 *   - the SAME warning is reported ONCE for the whole session, so a sink
 *     that cannot be written does not turn into a warning storm;
 *   - the active session is the session that was started LAST, and only the
 *     ACTIVE session is closed by `shutdownTelemetry()`.
 *
 * WHY THE LOCAL EXPORTERS ARE MOCKED: the exporter module is replaced by a
 * seam that DELEGATES to the real implementation and can be told to fail.
 * The real implementation is what writes the JSONL, so the artefact is
 * proved for real; the failing seam is what proves the fail-open contract,
 * which the real exporter - that never throws - cannot demonstrate.
 *
 * @see plans/opentelemetry-slice-telemetry.plans.md - "Vertical slice A"
 */

import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import type {
  JsonlExporter,
  JsonlExporterOptions,
} from './telemetry.jsonl-exporters';
import type { TelemetrySnapshot } from './telemetry.session';

type SessionModule = typeof import('./telemetry.session');

/**
 * The behaviour the mocked exporter seam shall exhibit.
 */
const behaviour: {
  failExport: boolean;
  failFlush: boolean;
  failShutdown: boolean;
} = {
  failExport: false,
  failFlush: false,
  failShutdown: false,
};

/**
 * Creates a throwaway sink directory for the local exporters.
 *
 * @returns The absolute path of the sink directory.
 */
function createTempSinkDirectory(): string {
  const root = mkdtempSync(path.join(tmpdir(), 'telemetry-session-'));
  return path.join(root, 'sink');
}

/**
 * Reads the records of one JSONL sink file.
 *
 * @param sinkDirectory The sink directory.
 * @param fileName The name of the file.
 * @returns The parsed records.
 */
function readSinkRecords(
  sinkDirectory: string,
  fileName: string,
): Record<string, unknown>[] {
  const file = path.join(sinkDirectory, fileName);
  if (!existsSync(file)) return [];
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

/**
 * The exporter that fails on demand.
 *
 * @param options The exporter options.
 * @returns The failing exporter.
 */
function createFailingExporter(options: JsonlExporterOptions): JsonlExporter {
  return {
    sinkDirectory: options.sinkDirectory ?? 'failing-sink',
    exportRecord(): void {
      if (behaviour.failExport) throw new Error('the exporter refused the record');
    },
    async forceFlush(): Promise<void> {
      if (behaviour.failFlush) throw new Error('the exporter refused the flush');
    },
    async shutdown(): Promise<void> {
      if (behaviour.failShutdown) {
        throw new Error('the exporter refused the shutdown');
      }
    },
  };
}

/**
 * Loads a FRESH copy of the session module, so the seams that are in force
 * at this moment are the ones the module under test resolves.
 *
 * @returns The freshly loaded session module.
 */
async function loadSession(): Promise<SessionModule> {
  return await import('./telemetry.session');
}

describe('OT-V1-2: the layer-A session lifecycle', () => {
  beforeEach(() => {
    jest.resetModules();
    behaviour.failExport = false;
    behaviour.failFlush = false;
    behaviour.failShutdown = false;
    jest.doMock('./telemetry.jsonl-exporters', () => {
      const actual = jest.requireActual(
        './telemetry.jsonl-exporters',
      ) as typeof import('./telemetry.jsonl-exporters');
      return {
        ...actual,
        createJsonlExporter: (options: JsonlExporterOptions): JsonlExporter => {
          const failing =
            behaviour.failExport ||
            behaviour.failFlush ||
            behaviour.failShutdown;
          return failing
            ? createFailingExporter(options)
            : actual.createJsonlExporter(options);
        },
      };
    });
    jest.doMock('@opentelemetry/sdk-trace', () => ({
      TracerProvider: class {},
      BatchSpanProcessor: class {},
    }));
    jest.doMock('@opentelemetry/sdk-metrics', () => ({
      MeterProvider: class {},
      PeriodicExportingMetricReader: class {},
    }));
    jest.doMock('@opentelemetry/sdk-logs', () => ({
      LoggerProvider: class {},
      BatchLogRecordProcessor: class {},
    }));
  });

  describe('the session that starts', () => {
    it('starts with no options at all on the default sink root', async () => {
      const session = await loadSession();
      const started = await session.startTelemetrySession();
      expect(started.degraded).toBe(false);
      expect(started.reason).toBeNull();
      expect(started.sinkDirectory.replace(/\\/g, '/')).toMatch(
        /\/logs\/telemetry$/,
      );
    });

    it('reports the floor token and loads no peer below the telemetry floor', async () => {
      const descriptor = Object.getOwnPropertyDescriptor(process, 'versions');
      Object.defineProperty(process, 'versions', {
        value: { ...process.versions, node: '20.11.0' },
        configurable: true,
        writable: true,
      });
      try {
        const session = await loadSession();
        const warnings: string[] = [];
        const started = await session.startTelemetrySession({
          warn: (message: string) => {
            warnings.push(message);
          },
        });
        expect(started.degraded).toBe(true);
        expect(started.reason).toBe('node_below_telemetry_floor');
        expect(warnings).toEqual(['node_below_telemetry_floor']);
      } finally {
        if (descriptor) Object.defineProperty(process, 'versions', descriptor);
      }
    });

    it('reports the peer token when a peer is missing', async () => {
      jest.doMock('@opentelemetry/sdk-logs', () => {
        throw new Error('peer unavailable');
      });
      const session = await loadSession();
      const warnings: string[] = [];
      const started = await session.startTelemetrySession({
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      expect(started.degraded).toBe(true);
      expect(started.reason).toBe('telemetry_peer_unavailable');
      expect(warnings).toContain('telemetry_peer_unavailable');
    });
  });

  describe('the snapshot the session emits', () => {
    it('writes every record of the snapshot into the file of its signal', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const snapshot: TelemetrySnapshot = {
        trace: [
          {
            name: 'neat.telemetry.trace',
            attributes: { 'neat.telemetry.signal': 'trace' },
            interval: { startMonotonicMs: 0, endMonotonicMs: 12 },
          },
        ],
        metric: [
          {
            name: 'neat.telemetry.metric',
            attributes: { 'neat.telemetry.signal': 'metric' },
          },
        ],
        log: [
          {
            name: 'neat.telemetry.log',
            attributes: { 'neat.telemetry.signal': 'log' },
          },
        ],
      };
      const session = await loadSession();
      await session.startTelemetrySession({ sinkDirectory, snapshot });
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toEqual([
        {
          name: 'neat.telemetry.trace',
          duration_ms: 12,
          'neat.telemetry.signal': 'trace',
        },
      ]);
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([
        {
          name: 'neat.telemetry.metric',
          'neat.telemetry.signal': 'metric',
        },
      ]);
      expect(readSinkRecords(sinkDirectory, 'log.jsonl')).toEqual([
        { name: 'neat.telemetry.log', 'neat.telemetry.signal': 'log' },
      ]);
    });

    it('writes nothing for a snapshot that carries one signal only', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const session = await loadSession();
      await session.startTelemetrySession({
        sinkDirectory,
        snapshot: {
          trace: [
            { name: 'neat.telemetry.trace', attributes: { 'neat.telemetry.signal': 'trace' } },
          ],
        },
      });
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toHaveLength(1);
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([]);
      expect(readSinkRecords(sinkDirectory, 'log.jsonl')).toEqual([]);
    });

    it('writes nothing for a snapshot that carries no trace records at all', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const session = await loadSession();
      await session.startTelemetrySession({
        sinkDirectory,
        snapshot: {
          metric: [
            {
              name: 'neat.telemetry.metric',
              attributes: { 'neat.telemetry.signal': 'metric' },
            },
          ],
        },
      });
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toEqual([]);
      expect(readSinkRecords(sinkDirectory, 'log.jsonl')).toEqual([]);
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([
        {
          name: 'neat.telemetry.metric',
          'neat.telemetry.signal': 'metric',
        },
      ]);
    });
  });

  describe('the emission of a record', () => {
    it('writes a record that is emitted after the start', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const session = await loadSession();
      const started = await session.startTelemetrySession({ sinkDirectory });
      started.emitTrace({
        name: 'neat.telemetry.trace',
        attributes: { 'neat.telemetry.signal': 'trace' },
      });
      started.emitMetric({
        name: 'neat.telemetry.metric',
        attributes: { 'neat.telemetry.signal': 'metric' },
      });
      started.emitLog({
        name: 'neat.telemetry.log',
        attributes: { 'neat.telemetry.signal': 'log' },
      });
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toHaveLength(1);
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toHaveLength(1);
      expect(readSinkRecords(sinkDirectory, 'log.jsonl')).toHaveLength(1);
    });

    it('warns instead of throwing when the emission fails', async () => {
      behaviour.failExport = true;
      const session = await loadSession();
      const warnings: string[] = [];
      const started = await session.startTelemetrySession({
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      expect(() =>
        started.emitTrace({ name: 'neat.telemetry.trace' }),
      ).not.toThrow();
      expect(warnings).toEqual(['telemetry_emission_failed']);
    });
  });

  describe('the flush and the shutdown', () => {
    it('flushes the layer without throwing', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const session = await loadSession();
      const started = await session.startTelemetrySession({ sinkDirectory });
      await expect(started.flush()).resolves.toBeUndefined();
    });

    it('warns instead of throwing when the flush fails', async () => {
      behaviour.failFlush = true;
      const session = await loadSession();
      const warnings: string[] = [];
      const started = await session.startTelemetrySession({
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      await expect(started.flush()).resolves.toBeUndefined();
      expect(warnings).toEqual(['telemetry_emission_failed']);
    });

    it('warns instead of throwing when the shutdown fails', async () => {
      behaviour.failShutdown = true;
      const session = await loadSession();
      const warnings: string[] = [];
      const started = await session.startTelemetrySession({
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      await expect(started.shutdown()).resolves.toBeUndefined();
      expect(warnings).toEqual(['telemetry_emission_failed']);
    });

    it('writes NOTHING after the shutdown', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const session = await loadSession();
      const started = await session.startTelemetrySession({ sinkDirectory });
      started.emitTrace({
        name: 'neat.telemetry.trace',
        attributes: { 'neat.telemetry.signal': 'trace' },
      });
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toHaveLength(1);
      await started.shutdown();
      started.emitTrace({
        name: 'neat.telemetry.trace',
        attributes: { 'neat.telemetry.signal': 'trace' },
      });
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toHaveLength(1);
    });
  });

  describe('the warning that is reported once', () => {
    it('reports a sink that cannot be written ONCE for the whole session', async () => {
      const root = mkdtempSync(path.join(tmpdir(), 'telemetry-blocked-'));
      const blocked = path.join(root, 'blocked');
      writeFileSync(blocked, 'this is a file, not a directory', 'utf8');
      const session = await loadSession();
      const warnings: string[] = [];
      const started = await session.startTelemetrySession({
        sinkDirectory: blocked,
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      started.emitTrace({ name: 'neat.telemetry.trace' });
      started.emitMetric({ name: 'neat.telemetry.metric' });
      started.emitLog({ name: 'neat.telemetry.log' });
      expect(warnings).toEqual(['telemetry_sink_unwritable']);
    });
  });

  describe('the active session', () => {
    it('closes the session that was started LAST', async () => {
      const firstSink = createTempSinkDirectory();
      const secondSink = createTempSinkDirectory();
      const session = await loadSession();
      const first = await session.startTelemetrySession({
        sinkDirectory: firstSink,
      });
      const second = await session.startTelemetrySession({
        sinkDirectory: secondSink,
      });
      await first.shutdown();
      first.emitTrace({
        name: 'neat.telemetry.trace',
        attributes: { 'neat.telemetry.signal': 'trace' },
      });
      expect(readSinkRecords(firstSink, 'trace.jsonl')).toEqual([]);
      await session.shutdownTelemetry();
      second.emitTrace({
        name: 'neat.telemetry.trace',
        attributes: { 'neat.telemetry.signal': 'trace' },
      });
      expect(readSinkRecords(secondSink, 'trace.jsonl')).toEqual([]);
    });

    it('is a no-op when no session was ever started', async () => {
      const session = await loadSession();
      await expect(session.shutdownTelemetry()).resolves.toBeUndefined();
    });
  });
});
