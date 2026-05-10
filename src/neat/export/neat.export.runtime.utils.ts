import Connection from '../../architecture/connection';
import Node from '../../architecture/node';
import { exportRngState, restoreRngState } from '../rng/rng';
import type {
  NeatControllerForExport,
  NeatRuntimeMetaJSON,
} from './neat.export.types';

/**
 * Serialize controller runtime state that is independent of the live population.
 *
 * These fields can travel with the meta-only checkpoint surface because they do
 * not hold direct references to live genome instances.
 *
 * Deterministic replay note:
 *
 * The controller's replay-safe resume path requires more than the innovation
 * tracker payload. The architecture layer also owns monotonic counters for
 * allocating connection innovations, node gene ids, and runtime node indexes.
 * If those counters are not restored, a resumed run can produce valid networks
 * that still diverge immediately because the next structural allocation chooses
 * different identities.
 *
 * @param internal - Live controller host.
 * @returns Serializable runtime metadata payload.
 */
export function serializeRuntimeMeta(
  internal: NeatControllerForExport,
): NeatRuntimeMetaJSON {
  const runtime: NeatRuntimeMetaJSON = {};
  const architectureCounters = readArchitectureCounters();

  if (typeof internal._nextGenomeId === 'number') {
    runtime.nextGenomeId = internal._nextGenomeId;
  }
  runtime.nextConnectionInnovation =
    architectureCounters.nextConnectionInnovation;
  runtime.nextNodeGeneId = architectureCounters.nextNodeGeneId;
  runtime.nextNodeIndex = architectureCounters.nextNodeIndex;
  if (typeof internal._lineageEnabled === 'boolean') {
    runtime.lineageEnabled = internal._lineageEnabled;
  }
  const rngState = exportRngState(internal);
  if (typeof rngState === 'number') {
    runtime.rngState = rngState;
  }
  if (typeof internal._lastInbreedingCount === 'number') {
    runtime.lastInbreedingCount = internal._lastInbreedingCount;
  }
  if (typeof internal._lastGlobalImproveGeneration === 'number') {
    runtime.lastGlobalImproveGeneration = internal._lastGlobalImproveGeneration;
  }
  if (typeof internal._adaptivePruneLevel === 'number') {
    runtime.adaptivePruneLevel = internal._adaptivePruneLevel;
  }
  if (typeof internal._adaptivePruneBaseline === 'number') {
    runtime.adaptivePruneBaseline = internal._adaptivePruneBaseline;
  }
  if (
    Array.isArray(internal._noveltyArchive) &&
    internal._noveltyArchive.length > 0
  ) {
    runtime.noveltyArchive = structuredClone(internal._noveltyArchive);
  }
  if (Array.isArray(internal._speciesHistory)) {
    runtime.speciesHistory = structuredClone(internal._speciesHistory);
  }
  if (
    internal._operatorStats instanceof Map &&
    internal._operatorStats.size > 0
  ) {
    runtime.operatorStats = Array.from(
      internal._operatorStats.entries(),
      ([methodName, operatorStats]) => [
        methodName,
        structuredClone(operatorStats),
      ],
    );
  }

  return runtime;
}

/**
 * Restore controller runtime state from meta-only checkpoint data.
 *
 * This helper only applies fields that remain meaningful without the live
 * population object graph; species registries themselves stay reserved for the
 * full checkpoint path.
 *
 * @param neatInstance - Fresh controller instance being restored.
 * @param runtimeMeta - Optional runtime metadata payload from persistence.
 * @returns Nothing.
 */
export function restoreRuntimeMeta(
  neatInstance: NeatControllerForExport,
  runtimeMeta: NeatRuntimeMetaJSON | undefined,
): void {
  if (!runtimeMeta || typeof runtimeMeta !== 'object') {
    return;
  }

  if (typeof runtimeMeta.nextGenomeId === 'number') {
    neatInstance._nextGenomeId = runtimeMeta.nextGenomeId;
  }
  restoreArchitectureCounters(runtimeMeta);
  if (typeof runtimeMeta.lineageEnabled === 'boolean') {
    neatInstance._lineageEnabled = runtimeMeta.lineageEnabled;
  }
  if (typeof runtimeMeta.rngState === 'number') {
    restoreRngState(neatInstance, runtimeMeta.rngState);
  }
  if (typeof runtimeMeta.lastInbreedingCount === 'number') {
    neatInstance._lastInbreedingCount = runtimeMeta.lastInbreedingCount;
  }
  if (typeof runtimeMeta.lastGlobalImproveGeneration === 'number') {
    neatInstance._lastGlobalImproveGeneration =
      runtimeMeta.lastGlobalImproveGeneration;
  }
  if (typeof runtimeMeta.adaptivePruneLevel === 'number') {
    neatInstance._adaptivePruneLevel = runtimeMeta.adaptivePruneLevel;
  }
  if (typeof runtimeMeta.adaptivePruneBaseline === 'number') {
    neatInstance._adaptivePruneBaseline = runtimeMeta.adaptivePruneBaseline;
  }
  if (Array.isArray(runtimeMeta.noveltyArchive)) {
    neatInstance._noveltyArchive = structuredClone(runtimeMeta.noveltyArchive);
  }
  if (Array.isArray(runtimeMeta.speciesHistory)) {
    neatInstance._speciesHistory = structuredClone(runtimeMeta.speciesHistory);
  }
  if (Array.isArray(runtimeMeta.operatorStats)) {
    neatInstance._operatorStats = new Map(
      runtimeMeta.operatorStats.map(([methodName, operatorStats]) => [
        methodName,
        structuredClone(operatorStats),
      ]),
    );
  }
}

function readArchitectureCounters(): {
  nextConnectionInnovation: number;
  nextNodeGeneId: number;
  nextNodeIndex: number;
} {
  return {
    nextConnectionInnovation: (
      Connection as unknown as { _nextInnovation: number }
    )._nextInnovation,
    nextNodeGeneId: (Node as unknown as { _nextGeneId: number })._nextGeneId,
    nextNodeIndex: (Node as unknown as { _globalNodeIndex: number })
      ._globalNodeIndex,
  };
}

function restoreArchitectureCounters(runtimeMeta: NeatRuntimeMetaJSON): void {
  // Step 1: Restore monotonic allocators used by the architecture runtime.
  // These counters are intentionally stored as part of controller runtime meta
  // so a full checkpoint can keep allocating unique, deterministic identity
  // values after restore.
  if (typeof runtimeMeta.nextConnectionInnovation === 'number') {
    (Connection as unknown as { _nextInnovation: number })._nextInnovation =
      runtimeMeta.nextConnectionInnovation;
  }
  if (typeof runtimeMeta.nextNodeGeneId === 'number') {
    (Node as unknown as { _nextGeneId: number })._nextGeneId =
      runtimeMeta.nextNodeGeneId;
  }
  if (typeof runtimeMeta.nextNodeIndex === 'number') {
    (Node as unknown as { _globalNodeIndex: number })._globalNodeIndex =
      runtimeMeta.nextNodeIndex;
  }
}
