/**
 * GREEN home for the provider bootstrap of the local OpenTelemetry signal
 * layer (slice OT-V1-2 of `plans/opentelemetry-slice-telemetry.plans.md`).
 *
 * WHAT THIS HOME DEMANDS, over the REAL module:
 *
 *   - the engine floor guard accepts a runtime AT the floor, ABOVE it and
 *     BELOW it, and it refuses a version string it cannot parse;
 *   - the warning of a peer that could not be loaded carries the SPECIFIER
 *     of the peer and never an absolute path, so the warning is safe to
 *     persist;
 *   - the optional-peer loader turns an absent peer into `null` plus ONE
 *     warning, and an absent peer without a warning sink stays silent;
 *   - the provider construction is DEGRADED, NOT BROKEN: a missing peer
 *     leaves its provider null and the other providers alive;
 *   - a runtime BELOW the floor never loads a peer at all - the layer is
 *     warn-only;
 *   - the three SDK adapters hand the measured spans, the metric data
 *     points and the log records to the local exporters, they report the
 *     result code of the SDK and they NEVER throw into the SDK.
 *
 * WHY THE PEERS ARE MOCKED: the three OpenTelemetry SDKs are OPTIONAL peers
 * of this repository, and the layer must behave the same whether they are
 * installed or not. A lane that only ever saw the installed packages could
 * not prove the degraded mode, and a lane that used the real processors
 * would leave a timer running behind every assertion.
 *
 * @see plans/opentelemetry-slice-telemetry.plans.md - "Vertical slice A"
 */

import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';

type BootstrapModule = typeof import('./telemetry.bootstrap');

/**
 * The SDK-facing exporter the bootstrap hands to an SDK processor.
 */
interface SdkExporter {
  export(items: unknown, resultCallback?: (result: unknown) => void): void;
  forceFlush(): Promise<void>;
  shutdown(): Promise<void>;
}

/**
 * The state of one optional peer: whether it resolves, and what it exports.
 */
interface PeerState {
  available: boolean;
  exports: Record<string, unknown>;
}

/**
 * Everything the mocked SDK constructors hand back to the lanes.
 */
const captured: {
  span?: SdkExporter;
  metric?: SdkExporter;
  log?: SdkExporter;
  tracerOptions?: unknown;
  meterOptions?: unknown;
  loggerOptions?: unknown;
  readerOptions?: unknown;
} = {};

/**
 * The state of the three optional peers, swapped per lane.
 */
const peerState: {
  trace: PeerState;
  metrics: PeerState;
  logs: PeerState;
} = {
  trace: { available: true, exports: {} },
  metrics: { available: true, exports: {} },
  logs: { available: true, exports: {} },
};

/**
 * The trace peer: a provider and the processor that takes the exporter.
 *
 * @returns The exports of the mocked trace peer.
 */
function defaultTraceExports(): Record<string, unknown> {
  return {
    TracerProvider: class {
      constructor(options: unknown) {
        captured.tracerOptions = options;
      }
    },
    BatchSpanProcessor: class {
      constructor(exporter: unknown) {
        captured.span = exporter as SdkExporter;
      }
    },
  };
}

/**
 * The metrics peer: a provider and the reader that takes the exporter.
 *
 * @returns The exports of the mocked metrics peer.
 */
function defaultMetricExports(): Record<string, unknown> {
  return {
    MeterProvider: class {
      constructor(options: unknown) {
        captured.meterOptions = options;
      }
    },
    PeriodicExportingMetricReader: class {
      constructor(options: unknown) {
        captured.readerOptions = options;
        captured.metric = (options as { exporter: SdkExporter }).exporter;
      }
    },
  };
}

/**
 * The logs peer: a provider and the processor that takes the exporter.
 *
 * @returns The exports of the mocked logs peer.
 */
function defaultLogExports(): Record<string, unknown> {
  return {
    LoggerProvider: class {
      constructor(options: unknown) {
        captured.loggerOptions = options;
      }
    },
    BatchLogRecordProcessor: class {
      constructor(exporter: unknown) {
        captured.log = exporter as SdkExporter;
      }
    },
  };
}

/**
 * Loads a FRESH copy of the bootstrap, so the peer mocks that are in force
 * at this moment are the ones the module under test resolves.
 *
 * @returns The freshly loaded bootstrap module.
 */
async function loadBootstrap(): Promise<BootstrapModule> {
  return await import('./telemetry.bootstrap');
}

/**
 * The captured span exporter, or a loud failure when the lane never got one.
 *
 * @returns The span exporter the bootstrap handed to the SDK.
 */
function spanExporter(): SdkExporter {
  if (!captured.span) throw new Error('no span exporter was captured');
  return captured.span;
}

/**
 * The captured metric exporter, or a loud failure when the lane never got one.
 *
 * @returns The metric exporter the bootstrap handed to the SDK.
 */
function metricExporter(): SdkExporter {
  if (!captured.metric) throw new Error('no metric exporter was captured');
  return captured.metric;
}

/**
 * The captured log exporter, or a loud failure when the lane never got one.
 *
 * @returns The log exporter the bootstrap handed to the SDK.
 */
function logExporter(): SdkExporter {
  if (!captured.log) throw new Error('no log exporter was captured');
  return captured.log;
}

/**
 * Creates a throwaway sink directory for the local exporters.
 *
 * @returns The absolute path of the sink directory.
 */
function createTempSinkDirectory(): string {
  const root = mkdtempSync(path.join(tmpdir(), 'telemetry-bootstrap-'));
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
 * The pattern an absolute path would match in a persisted artefact.
 */
const ABSOLUTE_PATH_PATTERN = /[A-Za-z]:[\\/]|\/(?:Users|home|tmp|var|etc|opt|private)\//;

describe('OT-V1-2: the provider bootstrap', () => {
  beforeEach(() => {
    jest.resetModules();
    captured.span = undefined;
    captured.metric = undefined;
    captured.log = undefined;
    captured.tracerOptions = undefined;
    captured.meterOptions = undefined;
    captured.loggerOptions = undefined;
    captured.readerOptions = undefined;
    peerState.trace = { available: true, exports: defaultTraceExports() };
    peerState.metrics = { available: true, exports: defaultMetricExports() };
    peerState.logs = { available: true, exports: defaultLogExports() };
    jest.doMock('@opentelemetry/sdk-trace', () => {
      if (!peerState.trace.available) throw new Error('peer unavailable');
      return peerState.trace.exports;
    });
    jest.doMock('@opentelemetry/sdk-metrics', () => {
      if (!peerState.metrics.available) throw new Error('peer unavailable');
      return peerState.metrics.exports;
    });
    jest.doMock('@opentelemetry/sdk-logs', () => {
      if (!peerState.logs.available) throw new Error('peer unavailable');
      return peerState.logs.exports;
    });
  });

  describe('the engine floor guard', () => {
    it('accepts a runtime that sits exactly on the floor', async () => {
      const bootstrap = await loadBootstrap();
      expect(bootstrap.checkNodeFloor('22.15.0')).toEqual({
        supported: true,
        reason: null,
      });
    });

    it('accepts a runtime above the floor', async () => {
      const bootstrap = await loadBootstrap();
      expect(bootstrap.checkNodeFloor('22.15.1').supported).toBe(true);
      expect(bootstrap.checkNodeFloor('23.0.0').supported).toBe(true);
    });

    it('refuses a runtime below the floor with the machine-stable token', async () => {
      const bootstrap = await loadBootstrap();
      expect(bootstrap.checkNodeFloor('22.14.9')).toEqual({
        supported: false,
        reason: 'node_below_telemetry_floor',
      });
      expect(bootstrap.checkNodeFloor('20.11.0').supported).toBe(false);
    });

    it('refuses a version string it cannot parse', async () => {
      const bootstrap = await loadBootstrap();
      expect(bootstrap.checkNodeFloor('not-a-version')).toEqual({
        supported: false,
        reason: 'node_below_telemetry_floor',
      });
    });

    it('reads the running runtime when no version is given', async () => {
      const bootstrap = await loadBootstrap();
      const verdict = bootstrap.checkNodeFloor();
      expect(verdict.supported).toBe(
        Number(process.versions.node.split('.')[0]) > 22 ||
          (Number(process.versions.node.split('.')[0]) === 22 &&
            Number(process.versions.node.split('.')[1]) >= 15),
      );
    });
  });

  describe('the warning of a peer that could not be loaded', () => {
    it('carries the specifier of the peer', async () => {
      const bootstrap = await loadBootstrap();
      expect(bootstrap.describeMissingPeer('@opentelemetry/sdk-trace')).toBe(
        'telemetry_peer_unavailable:@opentelemetry/sdk-trace',
      );
      expect(bootstrap.describeMissingPeer('opentelemetry-api')).toBe(
        'telemetry_peer_unavailable:opentelemetry-api',
      );
    });

    it('never carries an absolute path', async () => {
      const bootstrap = await loadBootstrap();
      const warning = bootstrap.describeMissingPeer(
        'C:\\Users\\someone\\node_modules\\peer\\index.js',
      );
      expect(warning).toBe('telemetry_peer_unavailable:peer');
      expect(warning).not.toMatch(ABSOLUTE_PATH_PATTERN);
    });
  });

  describe('the optional peer loader', () => {
    it('resolves an installed peer to its module namespace', async () => {
      const bootstrap = await loadBootstrap();
      const loaded = await bootstrap.loadOptionalPeer('node:path');
      expect(loaded).not.toBeNull();
      expect(typeof loaded).toBe('object');
    });

    it('resolves an absent peer to null and warns once', async () => {
      const bootstrap = await loadBootstrap();
      const warnings: string[] = [];
      const loaded = await bootstrap.loadOptionalPeer(
        '@opentelemetry/does-not-exist',
        (message: string) => {
          warnings.push(message);
        },
      );
      expect(loaded).toBeNull();
      expect(warnings).toEqual([
        'telemetry_peer_unavailable:@opentelemetry/does-not-exist',
      ]);
    });

    it('stays silent when no warning sink was handed in', async () => {
      const bootstrap = await loadBootstrap();
      const loaded = await bootstrap.loadOptionalPeer(
        '@opentelemetry/does-not-exist',
      );
      expect(loaded).toBeNull();
    });
  });

  describe('the provider construction', () => {
    it('builds the three providers when every peer is installed', async () => {
      const bootstrap = await loadBootstrap();
      const providers = await bootstrap.createTelemetryProviders();
      expect(providers.traceProvider).not.toBeNull();
      expect(providers.meterProvider).not.toBeNull();
      expect(providers.loggerProvider).not.toBeNull();
      expect(providers.degraded).toBe(false);
      expect(providers.reason).toBeNull();
    });

    it('degrades, without breaking, when one peer is missing', async () => {
      peerState.logs.available = false;
      const bootstrap = await loadBootstrap();
      const warnings: string[] = [];
      const providers = await bootstrap.createTelemetryProviders({
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      expect(providers.traceProvider).not.toBeNull();
      expect(providers.meterProvider).not.toBeNull();
      expect(providers.loggerProvider).toBeNull();
      expect(providers.degraded).toBe(true);
      expect(providers.reason).toBe('telemetry_peer_unavailable');
      expect(warnings).toEqual([
        'telemetry_peer_unavailable:@opentelemetry/sdk-logs',
      ]);
    });

    it('warns once per missing peer when every peer is missing', async () => {
      peerState.trace.available = false;
      peerState.metrics.available = false;
      peerState.logs.available = false;
      const bootstrap = await loadBootstrap();
      const warnings: string[] = [];
      const providers = await bootstrap.createTelemetryProviders({
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      expect(providers.degraded).toBe(true);
      expect(warnings).toEqual([
        'telemetry_peer_unavailable:@opentelemetry/sdk-trace',
        'telemetry_peer_unavailable:@opentelemetry/sdk-metrics',
        'telemetry_peer_unavailable:@opentelemetry/sdk-logs',
      ]);
    });

    it('never loads a peer at all when the runtime is below the floor', async () => {
      const descriptor = Object.getOwnPropertyDescriptor(process, 'versions');
      Object.defineProperty(process, 'versions', {
        value: { ...process.versions, node: '20.11.0' },
        configurable: true,
        writable: true,
      });
      try {
        const bootstrap = await loadBootstrap();
        const warnings: string[] = [];
        const providers = await bootstrap.createTelemetryProviders({
          warn: (message: string) => {
            warnings.push(message);
          },
        });
        expect(providers.traceProvider).toBeNull();
        expect(providers.meterProvider).toBeNull();
        expect(providers.loggerProvider).toBeNull();
        expect(providers.degraded).toBe(true);
        expect(providers.reason).toBe('node_below_telemetry_floor');
        expect(warnings).toEqual(['node_below_telemetry_floor']);
      } finally {
        if (descriptor) Object.defineProperty(process, 'versions', descriptor);
      }
    });

    it('leaves a provider null when its peer exposes no processor', async () => {
      peerState.trace.exports = {
        TracerProvider: class {
          constructor(options: unknown) {
            captured.tracerOptions = options;
          }
        },
      };
      const bootstrap = await loadBootstrap();
      const providers = await bootstrap.createTelemetryProviders();
      expect(providers.traceProvider).toBeNull();
      expect(providers.meterProvider).not.toBeNull();
      expect(providers.loggerProvider).not.toBeNull();
    });

    it('leaves a provider null when its constructor throws', async () => {
      peerState.trace.exports = {
        TracerProvider: class {
          constructor() {
            throw new Error('the provider refused to start');
          }
        },
        BatchSpanProcessor: class {
          constructor(exporter: unknown) {
            captured.span = exporter as SdkExporter;
          }
        },
      };
      const bootstrap = await loadBootstrap();
      const providers = await bootstrap.createTelemetryProviders();
      expect(providers.traceProvider).toBeNull();
      expect(providers.meterProvider).not.toBeNull();
      expect(providers.loggerProvider).not.toBeNull();
    });

    it('leaves the meter provider null when its constructor throws', async () => {
      peerState.metrics.exports = {
        MeterProvider: class {
          constructor() {
            throw new Error('the provider refused to start');
          }
        },
        PeriodicExportingMetricReader: class {
          constructor(options: unknown) {
            captured.readerOptions = options;
            captured.metric = (options as { exporter: SdkExporter }).exporter;
          }
        },
      };
      const bootstrap = await loadBootstrap();
      const providers = await bootstrap.createTelemetryProviders();
      expect(providers.meterProvider).toBeNull();
      expect(providers.traceProvider).not.toBeNull();
      expect(providers.loggerProvider).not.toBeNull();
    });

    it('leaves the log provider null when its constructor throws', async () => {
      peerState.logs.exports = {
        LoggerProvider: class {
          constructor() {
            throw new Error('the provider refused to start');
          }
        },
        BatchLogRecordProcessor: class {
          constructor(exporter: unknown) {
            captured.log = exporter as SdkExporter;
          }
        },
      };
      const bootstrap = await loadBootstrap();
      const providers = await bootstrap.createTelemetryProviders();
      expect(providers.loggerProvider).toBeNull();
      expect(providers.traceProvider).not.toBeNull();
      expect(providers.meterProvider).not.toBeNull();
    });
  });

  describe('the span adapter', () => {
    it('writes the measured span as one local record', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.span?.export(
        [
          {
            duration: [1, 500000],
            attributes: { 'neat.telemetry.signal': 'trace' },
          },
        ],
        (result: unknown) => {
          codes.push(result);
        },
      );
      expect(codes).toEqual([{ code: 0 }]);
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toEqual([
        {
          name: 'neat.telemetry.trace',
          duration_ms: 1000.5,
          'neat.telemetry.signal': 'trace',
        },
      ]);
    });

    it('writes NO duration when the span carries no measured duration', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      captured.span?.export([
        { attributes: { 'neat.telemetry.signal': 'trace' } },
      ]);
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toEqual([
        { name: 'neat.telemetry.trace', 'neat.telemetry.signal': 'trace' },
      ]);
    });

    it('writes NO duration when the measured duration is not finite', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      captured.span?.export([
        { duration: [Number.NaN, 0], attributes: { 'neat.telemetry.signal': 'trace' } },
      ]);
      const records = readSinkRecords(sinkDirectory, 'trace.jsonl');
      expect(records).toHaveLength(1);
      expect(records[0]).not.toHaveProperty('duration_ms');
    });

    it('writes a record with no attributes at all', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      captured.span?.export([{}]);
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toEqual([
        { name: 'neat.telemetry.trace' },
      ]);
    });

    it('writes nothing when the SDK hands over no span at all', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.span?.export(undefined, (result: unknown) => {
        codes.push(result);
      });
      expect(codes).toEqual([{ code: 0 }]);
      expect(readSinkRecords(sinkDirectory, 'trace.jsonl')).toEqual([]);
    });

    it('reports a failure to the SDK instead of throwing into it', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.span?.export(5, (result: unknown) => {
        codes.push(result);
      });
      expect(codes).toEqual([{ code: 1 }]);
    });

    it('swallows a failure when the SDK handed over no callback', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      expect(() => spanExporter().export(5)).not.toThrow();
    });

    it('flushes and shuts the local exporter down', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      await expect(spanExporter().forceFlush()).resolves.toBeUndefined();
      await expect(spanExporter().shutdown()).resolves.toBeUndefined();
    });
  });

  describe('the metric adapter', () => {
    it('writes every metric data point as one local record', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.metric?.export(
        {
          scopeMetrics: [
            {
              scope: { name: 'neat' },
              metrics: [
                {
                  dataPoints: [
                    { attributes: { 'neat.telemetry.signal': 'metric' } },
                  ],
                },
              ],
            },
          ],
        },
        (result: unknown) => {
          codes.push(result);
        },
      );
      expect(codes).toEqual([{ code: 0 }]);
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([
        {
          name: 'neat.telemetry.metric',
          'neat.telemetry.signal': 'metric',
        },
      ]);
    });

    it('writes nothing when the SDK hands over no metric at all', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.metric?.export(undefined, (result: unknown) => {
        codes.push(result);
      });
      expect(codes).toEqual([{ code: 0 }]);
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([]);
    });

    it('writes nothing when a scope carries no metric', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      metricExporter().export({ scopeMetrics: [{ scope: { name: 'neat' } }] });
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([]);
    });

    it('writes nothing when a metric carries no data point', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      metricExporter().export({ scopeMetrics: [{ metrics: [{}] }] });
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([]);
    });

    it('writes a data point with no attributes at all', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      captured.metric?.export({
        scopeMetrics: [{ metrics: [{ dataPoints: [{}] }] }],
      });
      expect(readSinkRecords(sinkDirectory, 'metric.jsonl')).toEqual([
        { name: 'neat.telemetry.metric' },
      ]);
    });

    it('reports a failure to the SDK instead of throwing into it', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.metric?.export(
        { scopeMetrics: [{ metrics: [{ dataPoints: [null] }] }] },
        (result: unknown) => {
          codes.push(result);
        },
      );
      expect(codes).toEqual([{ code: 1 }]);
    });

    it('swallows a failure when the SDK handed over no callback', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      expect(() =>
        metricExporter().export({ scopeMetrics: [{ metrics: [{ dataPoints: [null] }] }] }),
      ).not.toThrow();
    });

    it('flushes and shuts the local exporter down', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      await expect(metricExporter().forceFlush()).resolves.toBeUndefined();
      await expect(metricExporter().shutdown()).resolves.toBeUndefined();
    });
  });

  describe('the log adapter', () => {
    it('writes every log record as one local record', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.log?.export(
        [{ attributes: { 'neat.telemetry.signal': 'log' } }],
        (result: unknown) => {
          codes.push(result);
        },
      );
      expect(codes).toEqual([{ code: 0 }]);
      expect(readSinkRecords(sinkDirectory, 'log.jsonl')).toEqual([
        { name: 'neat.telemetry.log', 'neat.telemetry.signal': 'log' },
      ]);
    });

    it('writes nothing when the SDK hands over no log at all', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.log?.export(undefined, (result: unknown) => {
        codes.push(result);
      });
      expect(codes).toEqual([{ code: 0 }]);
      expect(readSinkRecords(sinkDirectory, 'log.jsonl')).toEqual([]);
    });

    it('writes a log record with no attributes at all', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      captured.log?.export([{}]);
      expect(readSinkRecords(sinkDirectory, 'log.jsonl')).toEqual([
        { name: 'neat.telemetry.log' },
      ]);
    });

    it('reports a failure to the SDK instead of throwing into it', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      const codes: unknown[] = [];
      captured.log?.export([null], (result: unknown) => {
        codes.push(result);
      });
      expect(codes).toEqual([{ code: 1 }]);
    });

    it('swallows a failure when the SDK handed over no callback', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      expect(() => logExporter().export([null])).not.toThrow();
    });

    it('flushes and shuts the local exporter down', async () => {
      const sinkDirectory = createTempSinkDirectory();
      const bootstrap = await loadBootstrap();
      await bootstrap.createTelemetryProviders({ sinkDirectory });
      await expect(logExporter().forceFlush()).resolves.toBeUndefined();
      await expect(logExporter().shutdown()).resolves.toBeUndefined();
    });
  });
});
