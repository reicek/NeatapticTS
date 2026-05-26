import Connection from '../../architecture/connection';
import Node from '../../architecture/node';
import { exportRngState, restoreRngState } from '../rng/rng';
import type {
  NeatControllerForExport,
  NeatRuntimeMetaJSON,
} from './neat.export.types';

type NumericRuntimeFieldName =
  | 'nextGenomeId'
  | 'nextConnectionInnovation'
  | 'nextNodeGeneId'
  | 'nextNodeIndex'
  | 'rngState'
  | 'lastInbreedingCount'
  | 'lastGlobalImproveGeneration'
  | 'adaptivePruneLevel'
  | 'adaptivePruneBaseline';

type BooleanRuntimeFieldName = 'lineageEnabled';

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

  assignOptionalRuntimeNumber(runtime, 'nextGenomeId', internal._nextGenomeId);
  assignOptionalRuntimeNumber(
    runtime,
    'nextConnectionInnovation',
    architectureCounters.nextConnectionInnovation,
  );
  assignOptionalRuntimeNumber(
    runtime,
    'nextNodeGeneId',
    architectureCounters.nextNodeGeneId,
  );
  assignOptionalRuntimeNumber(
    runtime,
    'nextNodeIndex',
    architectureCounters.nextNodeIndex,
  );
  assignOptionalRuntimeBoolean(
    runtime,
    'lineageEnabled',
    internal._lineageEnabled,
  );
  assignOptionalRuntimeNumber(runtime, 'rngState', exportRngState(internal));
  assignOptionalRuntimeNumber(
    runtime,
    'lastInbreedingCount',
    internal._lastInbreedingCount,
  );
  assignOptionalRuntimeNumber(
    runtime,
    'lastGlobalImproveGeneration',
    internal._lastGlobalImproveGeneration,
  );
  assignOptionalRuntimeNumber(
    runtime,
    'adaptivePruneLevel',
    internal._adaptivePruneLevel,
  );
  assignOptionalRuntimeNumber(
    runtime,
    'adaptivePruneBaseline',
    internal._adaptivePruneBaseline,
  );
  serializeRuntimeCollections(runtime, internal);
  serializeRuntimeOperatorStats(runtime, internal);

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

  restoreOptionalRuntimeNumber(runtimeMeta.nextGenomeId, (nextGenomeId) => {
    neatInstance._nextGenomeId = nextGenomeId;
  });
  restoreArchitectureCounters(runtimeMeta);
  restoreOptionalRuntimeBoolean(
    runtimeMeta.lineageEnabled,
    (lineageEnabled) => {
      neatInstance._lineageEnabled = lineageEnabled;
    },
  );
  restoreOptionalRuntimeNumber(runtimeMeta.rngState, (rngState) => {
    restoreRngState(neatInstance, rngState);
  });
  restoreOptionalRuntimeNumber(
    runtimeMeta.lastInbreedingCount,
    (lastInbreedingCount) => {
      neatInstance._lastInbreedingCount = lastInbreedingCount;
    },
  );
  restoreOptionalRuntimeNumber(
    runtimeMeta.lastGlobalImproveGeneration,
    (lastGlobalImproveGeneration) => {
      neatInstance._lastGlobalImproveGeneration = lastGlobalImproveGeneration;
    },
  );
  restoreOptionalRuntimeNumber(
    runtimeMeta.adaptivePruneLevel,
    (adaptivePruneLevel) => {
      neatInstance._adaptivePruneLevel = adaptivePruneLevel;
    },
  );
  restoreOptionalRuntimeNumber(
    runtimeMeta.adaptivePruneBaseline,
    (adaptivePruneBaseline) => {
      neatInstance._adaptivePruneBaseline = adaptivePruneBaseline;
    },
  );
  restoreRuntimeCollections(neatInstance, runtimeMeta);
}

function assignOptionalRuntimeNumber(
  runtime: NeatRuntimeMetaJSON,
  fieldName: NumericRuntimeFieldName,
  value: number | undefined,
): void {
  if (typeof value === 'number') {
    runtime[fieldName] = value;
  }
}

function assignOptionalRuntimeBoolean(
  runtime: NeatRuntimeMetaJSON,
  fieldName: BooleanRuntimeFieldName,
  value: boolean | undefined,
): void {
  if (typeof value === 'boolean') {
    runtime[fieldName] = value;
  }
}

function serializeRuntimeCollections(
  runtime: NeatRuntimeMetaJSON,
  internal: NeatControllerForExport,
): void {
  if (
    Array.isArray(internal._noveltyArchive) &&
    internal._noveltyArchive.length > 0
  ) {
    runtime.noveltyArchive = structuredClone(internal._noveltyArchive);
  }
  if (Array.isArray(internal._speciesHistory)) {
    runtime.speciesHistory = structuredClone(internal._speciesHistory);
  }
}

function serializeRuntimeOperatorStats(
  runtime: NeatRuntimeMetaJSON,
  internal: NeatControllerForExport,
): void {
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
}

function restoreOptionalRuntimeNumber(
  value: number | undefined,
  apply: (value: number) => void,
): void {
  if (typeof value === 'number') {
    apply(value);
  }
}

function restoreOptionalRuntimeBoolean(
  value: boolean | undefined,
  apply: (value: boolean) => void,
): void {
  if (typeof value === 'boolean') {
    apply(value);
  }
}

function restoreRuntimeCollections(
  neatInstance: NeatControllerForExport,
  runtimeMeta: NeatRuntimeMetaJSON,
): void {
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
