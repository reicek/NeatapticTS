import {
  NGE_JUVENILE_DEFAULT_GATING_EDGE_LENGTH_THRESHOLD,
  NGE_JUVENILE_DEFAULT_LESION_SEVERITY,
  NGE_JUVENILE_DEFAULT_NOISE_SIGMA,
  NGE_JUVENILE_DEFAULT_PROBE_CADENCE_EPOCHS,
  NGE_JUVENILE_DEFAULT_PROBE_KINDS,
  NGE_JUVENILE_DEFAULT_PROBE_MAX_LEDGER_ENTRIES,
} from './neat.nge-juvenile.constants';
import { NgeJuvenile_ProbeError } from './neat.nge-juvenile.errors';
import type {
  NgeProbeDecision,
  NgeProbeKind,
  NgeProbeLedgerEntry,
  NgeProbeSchedulerConfig,
  NgeProbeSchedulerState,
} from './neat.nge-juvenile.types';

/**
 * Resolve a partial probe scheduler config against the seeded plan defaults.
 *
 * @param partial - Partial config whose omitted fields should resolve conservatively.
 * @returns A fully resolved probe scheduler config packet.
 * @throws {NgeJuvenile_ProbeError} When the probe kinds list is empty.
 */
export function resolveProbeSchedulerConfig(
  partial: Partial<NgeProbeSchedulerConfig>,
): NgeProbeSchedulerConfig {
  if (partial.probeKinds?.length === 0) {
    throw new NgeJuvenile_ProbeError(
      'Probe scheduler requires at least one probe kind.',
    );
  }

  return {
    probeKinds: [...(partial.probeKinds ?? NGE_JUVENILE_DEFAULT_PROBE_KINDS)],
    cadenceEpochs:
      partial.cadenceEpochs ?? NGE_JUVENILE_DEFAULT_PROBE_CADENCE_EPOCHS,
    maxLedgerEntries:
      partial.maxLedgerEntries ?? NGE_JUVENILE_DEFAULT_PROBE_MAX_LEDGER_ENTRIES,
    lesionSeverity:
      partial.lesionSeverity ?? NGE_JUVENILE_DEFAULT_LESION_SEVERITY,
    noiseSigma: partial.noiseSigma ?? NGE_JUVENILE_DEFAULT_NOISE_SIGMA,
    gatingEdgeLengthThreshold:
      partial.gatingEdgeLengthThreshold ??
      NGE_JUVENILE_DEFAULT_GATING_EDGE_LENGTH_THRESHOLD,
  };
}

/**
 * Build the zeroed scheduler state used before any probe has fired.
 *
 * @returns A JSON-safe scheduler state packet for one episode.
 */
export function defaultProbeSchedulerState(): NgeProbeSchedulerState {
  return {
    lastProbeEpoch: -1,
    nextProbeKindIndex: 0,
    ledger: [],
  };
}

/**
 * Decide whether the current epoch may execute one expensive perturbation probe.
 *
 * @param epochIndex - Current training or evaluation epoch.
 * @param state - Current scheduler state.
 * @param config - Fully resolved scheduler config.
 * @returns A pure decision packet describing cadence and the selected probe kind.
 */
export function decideProbe(
  epochIndex: number,
  state: NgeProbeSchedulerState,
  config: NgeProbeSchedulerConfig,
): NgeProbeDecision {
  const shouldRun =
    state.lastProbeEpoch < 0 ||
    epochIndex - state.lastProbeEpoch >= config.cadenceEpochs;

  if (!shouldRun) {
    return {
      shouldRun: false,
      probeKind: undefined,
    };
  }

  return {
    shouldRun: true,
    probeKind:
      config.probeKinds[state.nextProbeKindIndex % config.probeKinds.length],
  };
}

/**
 * Advance the scheduler state after one measured probe result is available.
 *
 * @param state - Previous scheduler state.
 * @param epochIndex - Epoch that produced the new probe result.
 * @param entry - Append-only ledger entry describing the probe outcome.
 * @param config - Fully resolved scheduler config.
 * @returns A fresh scheduler state with updated cadence metadata and ledger.
 */
export function advanceSchedulerState(
  state: NgeProbeSchedulerState,
  epochIndex: number,
  entry: NgeProbeLedgerEntry,
  config: NgeProbeSchedulerConfig,
): NgeProbeSchedulerState {
  return {
    lastProbeEpoch: epochIndex,
    nextProbeKindIndex:
      (state.nextProbeKindIndex + 1) % config.probeKinds.length,
    ledger: appendProbeLedgerEntry(
      state.ledger,
      entry,
      config.maxLedgerEntries,
    ),
  };
}

/**
 * Build one append-only probe ledger entry from caller-measured before and after reward readings.
 *
 * @param kind - Probe kind applied to the target module.
 * @param targetModuleId - Module whose local behavior was perturbed.
 * @param epochIndex - Epoch that recorded the probe result.
 * @param rewardBefore - Reward measured before the perturbation.
 * @param rewardAfter - Reward measured after the perturbation.
 * @returns One append-only probe ledger entry.
 */
export function buildProbeLedgerEntry(
  kind: NgeProbeKind,
  targetModuleId: string,
  epochIndex: number,
  rewardBefore: number,
  rewardAfter: number,
): NgeProbeLedgerEntry {
  return {
    probeKind: kind,
    targetModuleId,
    rewardBefore,
    rewardAfter,
    delta: rewardAfter - rewardBefore,
    epochIndex,
  };
}

/**
 * Append one probe entry while preserving immutability and the bounded ledger cap.
 *
 * @param ledger - Existing append-only probe ledger.
 * @param entry - New entry to append.
 * @param maxEntries - Maximum number of entries preserved in the returned ledger.
 * @returns A new bounded ledger with the newest entry preserved.
 */
export function appendProbeLedgerEntry(
  ledger: NgeProbeLedgerEntry[],
  entry: NgeProbeLedgerEntry,
  maxEntries: number,
): NgeProbeLedgerEntry[] {
  const nextLedger = [...ledger, entry];

  if (nextLedger.length <= maxEntries) {
    return nextLedger;
  }

  return nextLedger.slice(Math.max(nextLedger.length - maxEntries, 0));
}

/**
 * Compute the mean signed probe delta for one module across the current ledger.
 *
 * @param ledger - Append-only probe ledger.
 * @param moduleId - Module whose probe deltas should be averaged.
 * @returns Mean signed reward delta, or `0` when no matching probes exist.
 */
export function computeProbeRewardDelta(
  ledger: NgeProbeLedgerEntry[],
  moduleId: string,
): number {
  const matchingEntries = ledger.filter(
    ({ targetModuleId }) => targetModuleId === moduleId,
  );

  if (matchingEntries.length === 0) {
    return 0;
  }

  const totalDelta = matchingEntries.reduce(
    (currentTotal, { delta }) => currentTotal + delta,
    0,
  );

  return totalDelta / matchingEntries.length;
}

/**
 * Serialize the append-only probe ledger into a stable JSON string for checkpoint storage.
 *
 * @param ledger - Probe ledger to serialize.
 * @returns JSON string containing the ledger entries in order.
 */
export function serializeLedger(ledger: NgeProbeLedgerEntry[]): string {
  return JSON.stringify(ledger);
}

/**
 * Deserialize one JSON-serialized probe ledger previously produced by `serializeLedger`.
 * Throws `NgeJuvenile_ProbeError` when the payload cannot be parsed or is not a JSON array.
 *
 * @param json - JSON string previously produced by `serializeLedger`.
 * @returns Parsed probe ledger entries when the payload is a valid JSON array.
 * @throws {NgeJuvenile_ProbeError} When the payload cannot be parsed or is not a JSON array.
 */
export function deserializeLedger(json: string): NgeProbeLedgerEntry[] {
  try {
    const parsedValue: unknown = JSON.parse(json);

    if (!Array.isArray(parsedValue)) {
      throw new NgeJuvenile_ProbeError(
        'Probe ledger JSON must deserialize to an array.',
      );
    }

    return parsedValue as NgeProbeLedgerEntry[];
  } catch (error) {
    if (error instanceof NgeJuvenile_ProbeError) {
      throw error;
    }

    throw new NgeJuvenile_ProbeError('Probe ledger JSON could not be parsed.', {
      cause: error,
    });
  }
}
