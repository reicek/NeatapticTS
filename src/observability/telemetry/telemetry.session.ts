/**
 * The session lifecycle of the local OpenTelemetry signal layer: it starts the
 * session, emits the three signals for the sanitized snapshot it is HANDED,
 * flushes deterministically and shuts the layer down.
 *
 * WHAT THIS MODULE OWNS, AND WHAT IT MUST NOT OWN:
 *
 *   IT OWNS: the exporters, the provider lifecycle, the emission of the
 *   snapshot it is handed, the deterministic flush and the shutdown.
 *
 *   IT DOES NOT OWN: the slice begin/end instants, the slice-window
 *   duration, the usage deltas, the transient dispatch record of the
 *   repo-owned measurement tooling and the ledger block. Those belong to the
 *   repo-owned per-slice ledger, which PRODUCES the measurements; the signal
 *   layer may only CONSUME a finished snapshot.
 *   This module therefore imports NO `scripts/telemetry` module, references
 *   NO dispatch-record path and computes NO slice-window duration, so the
 *   ownership inversion cannot reappear inside the fail-open layer.
 *
 * WHY THE SNAPSHOT IS HANDED IN: the caller owns the facts. The session
 * emits what it is given and never goes looking for a measurement of its
 * own, because a layer that measures its own window would report a duration
 * it cannot evidence.
 *
 * WHY EVERY CALL IS FAIL-OPEN: an emission that fails warns with a short
 * machine-stable token and returns. The work the telemetry path observes must
 * never fail because of the telemetry path.
 *
 * @example
 * const session = await startTelemetrySession({
 *   snapshot: {
 *     trace: [{ name: 'neat.telemetry.trace', attributes: { 'neat.telemetry.signal': 'trace' } }],
 *   },
 * });
 * await session.flush();
 * await session.shutdown();
 *
 * @module telemetry.session
 */

import {
  createJsonlExporter,
  type JsonlExporter,
  type TelemetryRecordInput,
} from './telemetry.jsonl-exporters';
import {
  REASON_PEER_UNAVAILABLE,
  checkNodeFloor,
  createTelemetryProviders,
} from './telemetry.bootstrap';
import { REASON_NODE_BELOW_TELEMETRY_FLOOR } from './telemetry.policy';

/**
 * The warning token for an emission that failed. It is a TOKEN, not a
 * sentence, so a persisted warning stays greppable and stable across
 * releases.
 *
 * @type {string}
 */
export const REASON_EMISSION_FAILED = 'telemetry_emission_failed';

/**
 * The sanitized snapshot the session emits: one list of records per signal.
 *
 * @interface TelemetrySnapshot
 * @property {readonly TelemetryRecordInput[]} [trace] - The trace records.
 * @property {readonly TelemetryRecordInput[]} [metric] - The metric records.
 * @property {readonly TelemetryRecordInput[]} [log] - The log records.
 */
export interface TelemetrySnapshot {
  readonly trace?: readonly TelemetryRecordInput[];
  readonly metric?: readonly TelemetryRecordInput[];
  readonly log?: readonly TelemetryRecordInput[];
}

/**
 * The session handle. It exposes the degradation verdict of the layer, the
 * three emit calls and the two lifecycle calls. Every one of them is
 * fail-open, so the work the telemetry path observes never fails because of
 * the telemetry path.
 *
 * @interface TelemetrySession
 * @property {boolean} degraded - Whether the layer runs in the degraded no-op mode.
 * @property {string | null} reason - The machine-stable reason token of the degradation.
 * @property {string} sinkDirectory - The absolute sink directory of the local exporters.
 * @property {(record: TelemetryRecordInput) => void} emitTrace - Emits one trace record.
 * @property {(record: TelemetryRecordInput) => void} emitMetric - Emits one metric record.
 * @property {(record: TelemetryRecordInput) => void} emitLog - Emits one log record.
 * @property {() => Promise<void>} flush - Flushes the layer deterministically.
 * @property {() => Promise<void>} shutdown - Shuts the layer down.
 */
export interface TelemetrySession {
  readonly degraded: boolean;
  readonly reason: string | null;
  readonly sinkDirectory: string;
  emitTrace(record: TelemetryRecordInput): void;
  emitMetric(record: TelemetryRecordInput): void;
  emitLog(record: TelemetryRecordInput): void;
  flush(): Promise<void>;
  shutdown(): Promise<void>;
}

/**
 * The options of a session start: the sanitized snapshot the session emits,
 * the sink directory of the local exporters, and the warning sink that
 * receives the fail-open warnings.
 *
 * @interface StartTelemetrySessionOptions
 * @property {TelemetrySnapshot} [snapshot] - The sanitized snapshot the session emits.
 * @property {string} [sinkDirectory] - The sink directory of the local exporters.
 * @property {(message: string) => void} [warn] - The warning sink.
 */
export interface StartTelemetrySessionOptions {
  readonly snapshot?: TelemetrySnapshot;
  readonly sinkDirectory?: string;
  readonly warn?: (message: string) => void;
}

/**
 * The ACTIVE session, so the library entry point `shutdownTelemetry()` can
 * close the session a caller started.
 *
 * @type {TelemetrySession | null}
 */
let activeSession: TelemetrySession | null = null;

/**
 * Starts a session of the signal layer. The returned handle is always
 * usable: when a peer SDK or the runtime floor is missing, the session still
 * starts and reports the degradation through its verdict.
 *
 * @param {StartTelemetrySessionOptions} [options] - The session options.
 * @returns {Promise<TelemetrySession>} The session handle.
 * @example
 * const session = await startTelemetrySession();
 * await session.shutdown();
 */
export async function startTelemetrySession(
  options: StartTelemetrySessionOptions = {},
): Promise<TelemetrySession> {
  const warnings: string[] = [];
  const warn = (message: string): void => {
    if (warnings.includes(message)) return;
    warnings.push(message);
    options.warn?.(message);
  };
  const floor = checkNodeFloor();
  const traceExporter = createJsonlExporter({
    sinkDirectory: options.sinkDirectory,
    signal: 'trace',
    warn,
  });
  const metricExporter = createJsonlExporter({
    sinkDirectory: options.sinkDirectory,
    signal: 'metric',
    warn,
  });
  const logExporter = createJsonlExporter({
    sinkDirectory: options.sinkDirectory,
    signal: 'log',
    warn,
  });
  const exporters: Record<'trace' | 'metric' | 'log', JsonlExporter> = {
    trace: traceExporter,
    metric: metricExporter,
    log: logExporter,
  };

  let degraded = !floor.supported;
  let reason: string | null = floor.supported ? null : floor.reason;
  // The reason token is NORMALISED OUTSIDE the guard on purpose: the floor
  // verdict carries a reason only when it is unsupported, so a coalesce that
  // lived inside the guard could never reach its right-hand side and would
  // stay uncoverable. Normalising it first keeps both sides of the coalesce
  // reachable and leaves the reported token unchanged.
  const floorReason: string = floor.reason ?? REASON_NODE_BELOW_TELEMETRY_FLOOR;
  if (!floor.supported) {
    warn(floorReason);
  } else {
    const providers = await createTelemetryProviders({
      sinkDirectory: options.sinkDirectory,
      warn,
    });
    degraded = providers.degraded;
    reason = providers.reason;
    if (providers.degraded) warn(REASON_PEER_UNAVAILABLE);
  }

  /**
   * Emits one record through the exporter of a signal. A failure is a
   * warning: the session never throws into the caller.
   *
   * @param signal - The signal.
   * @param record - The record.
   */
  const emit = (
    signal: 'trace' | 'metric' | 'log',
    record: TelemetryRecordInput,
  ): void => {
    try {
      exporters[signal].exportRecord(record);
    } catch {
      warn(REASON_EMISSION_FAILED);
    }
  };

  const session: TelemetrySession = {
    degraded,
    reason,
    sinkDirectory: traceExporter.sinkDirectory,
    emitTrace(record: TelemetryRecordInput): void {
      emit('trace', record);
    },
    emitMetric(record: TelemetryRecordInput): void {
      emit('metric', record);
    },
    emitLog(record: TelemetryRecordInput): void {
      emit('log', record);
    },
    async flush(): Promise<void> {
      for (const exporter of Object.values(exporters)) {
        try {
          await exporter.forceFlush();
        } catch {
          warn(REASON_EMISSION_FAILED);
        }
      }
    },
    async shutdown(): Promise<void> {
      for (const exporter of Object.values(exporters)) {
        try {
          await exporter.shutdown();
        } catch {
          warn(REASON_EMISSION_FAILED);
        }
      }
      if (activeSession === session) activeSession = null;
    },
  };

  const snapshot = options.snapshot;
  if (snapshot) {
    for (const record of snapshot.trace ?? []) session.emitTrace(record);
    for (const record of snapshot.metric ?? []) session.emitMetric(record);
    for (const record of snapshot.log ?? []) session.emitLog(record);
  }

  activeSession = session;
  return session;
}

/**
 * Shuts the ACTIVE session of the signal layer down. A session that was never
 * started is not an error: the shutdown is idempotent and fail-open.
 *
 * @returns {Promise<void>} The completion of the shutdown.
 * @example
 * await shutdownTelemetry();
 */
export async function shutdownTelemetry(): Promise<void> {
  const session = activeSession;
  if (!session) return;
  await session.shutdown();
}
