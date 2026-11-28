/**
 * The provider bootstrap of the local OpenTelemetry signal layer: the
 * optional-peer loader, the telemetry engine floor guard and the construction
 * of the three local providers.
 *
 * WHY EVERY PEER IS LOADED THROUGH A DYNAMIC IMPORT: the six OpenTelemetry
 * packages are OPTIONAL peers. A consumer that never enables the telemetry
 * path does not install them, and a static import would turn their absence
 * into a load-time crash of the whole library. A dynamic import turns their
 * absence into a DEGRADED no-op mode: one warning per missing peer, and every
 * API call of the layer still succeeds.
 *
 * WHY THE WARNING NEVER CARRIES THE ERROR MESSAGE: a failed module
 * resolution carries the absolute path of the module the loader looked for,
 * and a warning is persisted like any other artefact. The warning is
 * therefore a short machine-stable token plus the SPECIFIER of the peer (a
 * package name, never a path).
 *
 * WHY THE ENGINE FLOOR IS ENFORCED HERE: the root manifest floor stays at
 * `>= 22.0.0` because the OpenTelemetry packages are optional, and a
 * manifest cannot express two floors for one package. The telemetry path
 * declares and enforces its OWN, higher floor, and a runtime below it
 * degrades the layer warn-only with the machine-stable reason token
 * `node_below_telemetry_floor` instead of failing the caller's work.
 *
 * WHY THE PROVIDER COLUMN NAME IS `TracerProvider`: the ruled 3.1.1 line
 * exposes `TracerProvider` and the 2.x `BasicTracerProvider` is GONE. The
 * bootstrap resolves the constructor by name and never mentions the retired
 * one, so a dependency that still ships the retired name cannot be picked up
 * by accident.
 *
 * LAYER INDEPENDENCE: this module imports NOTHING from the repo-owned
 * measurement tooling in `scripts/telemetry` and NOTHING from the
 * NEAT-algorithm telemetry tree in `src/neat/telemetry`. It measures nothing
 * of a slice window and it owns no slice instant.
 *
 * @example
 * const peer = await loadOptionalPeer('@opentelemetry/api');
 * checkNodeFloor('22.15.0');
 * // => { supported: true, reason: null }
 *
 * @module telemetry.bootstrap
 */

import {
  REASON_NODE_BELOW_TELEMETRY_FLOOR,
  TELEMETRY_NODE_FLOOR,
} from './telemetry.policy';
import {
  createJsonlExporter,
  type JsonlExporter,
  type TelemetryRecordInput,
} from './telemetry.jsonl-exporters';

/**
 * The warning token for a peer that could not be loaded.
 *
 * @type {string}
 */
export const REASON_PEER_UNAVAILABLE = 'telemetry_peer_unavailable';

/**
 * The specifier of the API package the layer always needs.
 *
 * @type {string}
 */
export const PEER_API = '@opentelemetry/api';

/**
 * The specifier of the ruled trace SDK. The trace SDK is an OPTIONAL peer: a
 * consumer that never enables the telemetry path does not install it.
 *
 * @type {string}
 */
export const PEER_TRACE_SDK = '@opentelemetry/sdk-trace';

/**
 * The specifier of the ruled metrics SDK. The metrics SDK is an OPTIONAL
 * peer: a consumer that never enables the telemetry path does not install
 * it.
 *
 * @type {string}
 */
export const PEER_METRICS_SDK = '@opentelemetry/sdk-metrics';

/**
 * The specifier of the ruled logs SDK. The logs SDK is an OPTIONAL peer: a
 * consumer that never enables the telemetry path does not install it.
 *
 * @type {string}
 */
export const PEER_LOGS_SDK = '@opentelemetry/sdk-logs';

/**
 * The numeric floor the telemetry path enforces, taken from the declared
 * floor of the policy.
 *
 * @type {string}
 */
const FLOOR_VERSION: string = TELEMETRY_NODE_FLOOR.replace(/[^0-9.]/g, '');

/**
 * The verdict of the engine floor guard. A supported runtime carries NO
 * reason token, and an unsupported one carries the machine-stable token that
 * names the floor it fell below.
 *
 * @interface NodeFloorVerdict
 * @property {boolean} supported - Whether the runtime is at or above the telemetry floor.
 * @property {string | null} reason - The machine-stable reason token, or null when the runtime is supported.
 */
export interface NodeFloorVerdict {
  readonly supported: boolean;
  readonly reason: string | null;
}

/**
 * Parses a `major.minor.patch` version out of a version string.
 *
 * @param {string} value - The version string.
 * @returns {number[] | null} The three components, or null when the string carries none.
 */
function parseVersion(value: string): number[] | null {
  const match = /(\d+)\.(\d+)\.(\d+)/.exec(String(value));
  if (!match) return null;
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

/**
 * Tells whether a runtime is at or above the telemetry engine floor.
 *
 * @param {string} [nodeVersion] - The runtime version; defaults to the running one.
 * @returns {NodeFloorVerdict} The verdict and its reason token.
 * @example
 * checkNodeFloor('22.15.0'); // => { supported: true, reason: null }
 * checkNodeFloor('22.14.0'); // => { supported: false, reason: 'node_below_telemetry_floor' }
 */
export function checkNodeFloor(nodeVersion?: string): NodeFloorVerdict {
  const current = parseVersion(nodeVersion ?? process.versions.node);
  const floor = parseVersion(FLOOR_VERSION);
  if (!current || !floor) {
    return { supported: false, reason: REASON_NODE_BELOW_TELEMETRY_FLOOR };
  }
  for (let index = 0; index < 3; index += 1) {
    if (current[index] === floor[index]) continue;
    if (current[index] > floor[index]) {
      return { supported: true, reason: null };
    }
    return { supported: false, reason: REASON_NODE_BELOW_TELEMETRY_FLOOR };
  }
  return { supported: true, reason: null };
}

/**
 * Builds the warning text of a peer that could not be loaded. The text is a
 * machine-stable token plus the specifier of the peer, and it carries no
 * absolute path, because a warning is persisted like any other artefact.
 *
 * @param {string} specifier - The specifier of the peer.
 * @returns {string} The warning text.
 */
export function describeMissingPeer(specifier: string): string {
  const safe =
    /^(@[a-z0-9-~][a-z0-9-._~]*\/)?[a-z0-9-~][a-z0-9-._~]*$/i.test(specifier)
      ? specifier
      : 'peer';
  return `${REASON_PEER_UNAVAILABLE}:${safe}`;
}

/**
 * Loads an OPTIONAL peer through a dynamic import. A peer that is not
 * installed yields NULL and one warning, instead of a load-time crash of the
 * whole library.
 *
 * @param {string} specifier - The specifier of the peer.
 * @param {(message: string) => void} [warn] - The warning sink.
 * @returns {Promise<unknown | null>} The loaded module, or null when it is not installed.
 * @example
 * await loadOptionalPeer('@opentelemetry/api');
 * // => the module namespace
 */
export async function loadOptionalPeer(
  specifier: string,
  warn?: (message: string) => void,
): Promise<unknown | null> {
  try {
    const loaded: unknown = await import(specifier);
    // A dynamic import ALWAYS yields a module namespace object, so the
    // `?? null` that used to stand here was DEAD CODE: the namespace can
    // never be null or undefined. The "null when the peer is not installed"
    // contract is carried by the catch below, so the redundant coalesce is
    // removed instead of being kept as an uncoverable branch.
    return loaded;
  } catch {
    if (warn) warn(describeMissingPeer(specifier));
    return null;
  }
}

/**
 * Picks a constructor out of a loaded peer.
 *
 * @param {unknown} peer - The loaded peer.
 * @param {string} name - The name of the constructor.
 * @returns {unknown} The constructor, or null when the peer does not expose it.
 */
function pickConstructor(peer: unknown, name: string): unknown {
  if (peer === null || typeof peer !== 'object') return null;
  const candidate = (peer as Record<string, unknown>)[name];
  return typeof candidate === 'function' ? candidate : null;
}

/**
 * The three providers of the local layer, each of them null when its peer is
 * not installed.
 *
 * @interface TelemetryProviders
 * @property {unknown} traceProvider - The trace provider, or null.
 * @property {unknown} meterProvider - The metric provider, or null.
 * @property {unknown} loggerProvider - The log provider, or null.
 * @property {boolean} degraded - Whether the layer runs in the degraded no-op mode.
 * @property {string | null} reason - The machine-stable reason token of the degradation.
 */
export interface TelemetryProviders {
  readonly traceProvider: unknown;
  readonly meterProvider: unknown;
  readonly loggerProvider: unknown;
  readonly degraded: boolean;
  readonly reason: string | null;
}

/**
 * The options of the provider construction: the sink directory the local
 * exporters write into, and the warning sink that receives the warnings the
 * construction emits.
 *
 * @interface TelemetryProviderOptions
 * @property {string} [sinkDirectory] - The sink directory of the local exporters.
 * @property {(message: string) => void} [warn] - The warning sink.
 */
export interface TelemetryProviderOptions {
  readonly sinkDirectory?: string;
  readonly warn?: (message: string) => void;
}

/**
 * The adapter that hands the spans of the trace SDK to the local JSONL
 * exporter. The SDK hands a MEASURED duration, and the exporter derives its
 * single duration field from an interval, so the adapter carries the measured
 * duration as a zero-based interval. No instant is ever carried over.
 *
 * @param exporter - The local exporter.
 * @returns {unknown} The SDK-facing exporter.
 */
function createSpanExporter(exporter: JsonlExporter): unknown {
  return {
    export(spans: unknown, resultCallback?: (result: unknown) => void): void {
      try {
        for (const span of (spans ?? []) as Array<Record<string, unknown>>) {
          const duration = Array.isArray(span.duration)
            ? Number(span.duration[0]) * 1000 + Number(span.duration[1]) / 1e6
            : null;
          const record: TelemetryRecordInput = {
            name: 'neat.telemetry.trace',
            attributes: (span.attributes ?? {}) as Record<string, unknown>,
            ...(duration !== null && Number.isFinite(duration)
              ? { interval: { startMonotonicMs: 0, endMonotonicMs: duration } }
              : {}),
          };
          exporter.exportRecord(record);
        }
        if (typeof resultCallback === 'function') resultCallback({ code: 0 });
      } catch {
        if (typeof resultCallback === 'function') resultCallback({ code: 1 });
      }
    },
    async shutdown(): Promise<void> {
      await exporter.shutdown();
    },
    async forceFlush(): Promise<void> {
      await exporter.forceFlush();
    },
  };
}

/**
 * The adapter that hands the metric data points of the metrics SDK to the
 * local JSONL exporter. A metric data point carries no measured duration, so
 * the record carries no duration field at all.
 *
 * @param exporter - The local exporter.
 * @returns {unknown} The SDK-facing exporter.
 */
function createMetricExporter(exporter: JsonlExporter): unknown {
  return {
    export(metrics: unknown, resultCallback?: (result: unknown) => void): void {
      try {
        const resourceMetrics = metrics as
          | { scopeMetrics?: Array<{ scope?: { name?: string }; metrics?: unknown[] }> }
          | undefined;
        for (const scopeMetrics of resourceMetrics?.scopeMetrics ?? []) {
          for (const metric of scopeMetrics.metrics ?? []) {
            const dataPoints = (
              metric as { dataPoints?: Array<Record<string, unknown>> }
            ).dataPoints ?? [];
            for (const dataPoint of dataPoints) {
              exporter.exportRecord({
                name: 'neat.telemetry.metric',
                attributes: (dataPoint.attributes ?? {}) as Record<string, unknown>,
              });
            }
          }
        }
        if (typeof resultCallback === 'function') resultCallback({ code: 0 });
      } catch {
        if (typeof resultCallback === 'function') resultCallback({ code: 1 });
      }
    },
    async forceFlush(): Promise<void> {
      await exporter.forceFlush();
    },
    async shutdown(): Promise<void> {
      await exporter.shutdown();
    },
  };
}

/**
 * The adapter that hands the log records of the logs SDK to the local JSONL
 * exporter.
 *
 * @param exporter - The local exporter.
 * @returns {unknown} The SDK-facing exporter.
 */
function createLogExporter(exporter: JsonlExporter): unknown {
  return {
    export(logs: unknown, resultCallback?: (result: unknown) => void): void {
      try {
        for (const log of (logs ?? []) as Array<Record<string, unknown>>) {
          exporter.exportRecord({
            name: 'neat.telemetry.log',
            attributes: (log.attributes ?? {}) as Record<string, unknown>,
          });
        }
        if (typeof resultCallback === 'function') resultCallback({ code: 0 });
      } catch {
        if (typeof resultCallback === 'function') resultCallback({ code: 1 });
      }
    },
    async forceFlush(): Promise<void> {
      await exporter.forceFlush();
    },
    async shutdown(): Promise<void> {
      await exporter.shutdown();
    },
  };
}

/**
 * Constructs the three local providers out of the optional peers.
 *
 * A peer that is not installed leaves its provider null and the layer
 * DEGRADED: EACH missing peer emits ONE warning that names it, and every
 * API call of the layer still succeeds through the local exporters.
 *
 * @param {TelemetryProviderOptions} [options] - The construction options.
 * @returns {Promise<TelemetryProviders>} The providers and the degradation verdict.
 * @example
 * const providers = await createTelemetryProviders();
 * providers.degraded; // => false when the peers are installed
 */
export async function createTelemetryProviders(
  options: TelemetryProviderOptions = {},
): Promise<TelemetryProviders> {
  // WHY THERE IS NO WARN-ONCE DEDUPE HERE: every warning site of this
  // construction emits a DISTINCT message - the floor token, or the peer
  // token plus the specifier of the peer that could not be loaded - so a
  // dedupe by message could never fire and would stay an uncoverable branch.
  // The construction therefore warns ONCE per EVENT: once for the floor and
  // once per missing peer. The layer stays SILENT when no warning sink was
  // handed in, because a fail-open layer never writes to the console on its
  // own.
  const warn = (message: string): void => {
    options.warn?.(message);
  };
  const floor = checkNodeFloor();
  // The reason token is NORMALISED OUTSIDE the guard on purpose: the floor
  // verdict carries a reason only when it is unsupported, so a coalesce that
  // lived inside the guard could never reach its right-hand side and would
  // stay uncoverable. Normalising it first keeps both sides of the coalesce
  // reachable and leaves the reported token unchanged.
  const floorReason: string = floor.reason ?? REASON_NODE_BELOW_TELEMETRY_FLOOR;
  if (!floor.supported) {
    warn(floorReason);
    return {
      traceProvider: null,
      meterProvider: null,
      loggerProvider: null,
      degraded: true,
      reason: floor.reason,
    };
  }
  const sinkDirectory = options.sinkDirectory;
  const [tracePeer, metricsPeer, logsPeer] = await Promise.all([
    loadOptionalPeer(PEER_TRACE_SDK, warn),
    loadOptionalPeer(PEER_METRICS_SDK, warn),
    loadOptionalPeer(PEER_LOGS_SDK, warn),
  ]);
  const degraded = tracePeer === null || metricsPeer === null || logsPeer === null;

  let traceProvider: unknown = null;
  const traceConstructor = pickConstructor(tracePeer, 'TracerProvider');
  const spanProcessorConstructor = pickConstructor(tracePeer, 'BatchSpanProcessor');
  if (traceConstructor && spanProcessorConstructor) {
    try {
      const exporter = createJsonlExporter({ sinkDirectory, signal: 'trace', warn });
      const Ctor = traceConstructor as new (options: unknown) => unknown;
      const Processor = spanProcessorConstructor as new (exporter: unknown) => unknown;
      traceProvider = new Ctor({
        spanProcessors: [new Processor(createSpanExporter(exporter))],
      });
    } catch {
      traceProvider = null;
    }
  }

  let meterProvider: unknown = null;
  const meterConstructor = pickConstructor(metricsPeer, 'MeterProvider');
  const readerConstructor = pickConstructor(
    metricsPeer,
    'PeriodicExportingMetricReader',
  );
  if (meterConstructor && readerConstructor) {
    try {
      const exporter = createJsonlExporter({ sinkDirectory, signal: 'metric', warn });
      const Ctor = meterConstructor as new (options: unknown) => unknown;
      const Reader = readerConstructor as new (options: unknown) => unknown;
      const reader = new Reader({
        exporter: createMetricExporter(exporter),
        exportIntervalMillis: 60000,
      });
      meterProvider = new Ctor({ readers: [reader] });
    } catch {
      meterProvider = null;
    }
  }

  let loggerProvider: unknown = null;
  const loggerConstructor = pickConstructor(logsPeer, 'LoggerProvider');
  const logProcessorConstructor = pickConstructor(
    logsPeer,
    'BatchLogRecordProcessor',
  );
  if (loggerConstructor && logProcessorConstructor) {
    try {
      const exporter = createJsonlExporter({ sinkDirectory, signal: 'log', warn });
      const Ctor = loggerConstructor as new (options: unknown) => unknown;
      const Processor = logProcessorConstructor as new (exporter: unknown) => unknown;
      loggerProvider = new Ctor({
        processors: [new Processor(createLogExporter(exporter))],
      });
    } catch {
      loggerProvider = null;
    }
  }

  return {
    traceProvider,
    meterProvider,
    loggerProvider,
    degraded,
    reason: degraded ? REASON_PEER_UNAVAILABLE : null,
  };
}
