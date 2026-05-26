import { NeatExportPopulationValidationError } from './neat.export.errors';
import type {
  GenomeControllerCarrier,
  GenomeControllerMetaJSON,
  GenomeJSON,
} from './neat.export.types';

type NumericGenomeMetaAssignment = {
  value: number | undefined;
  apply(value: number): void;
};

/**
 * Population genome checkpoint helpers.
 *
 * A "population snapshot" is primarily the network structure (nodes +
 * connections). However, the NEAT controller also owns per-genome annotations
 * such as scores, lineage metadata, multi-objective ranks, novelty, and stable
 * genome ids.
 *
 * This file implements a small contract:
 *
 * - `controllerMeta` is the only reserved controller-owned key in a serialized
 *   genome payload.
 * - Everything else is treated as raw network JSON and is passed through to
 *   `Network.fromJSON()`.
 * - Stable genome ids must be unique within a restored population.
 *
 * That last point matters for deterministic replay: a full checkpoint restores
 * speciation state by stable genome id, not by array position.
 */

/**
 * Serialize one live genome into a checkpoint payload.
 *
 * The network serializer owns structural graph fields, while this helper adds
 * the reserved `controllerMeta` pocket only when the controller has metadata
 * worth preserving beside the network JSON.
 *
 * @param genome - Live genome from the controller population.
 * @returns Serialized genome payload with optional controller metadata.
 */
export function serializeGenomeCheckpoint(
  genome: GenomeControllerCarrier,
  networkPayload: GenomeJSON = genome.toJSON(),
): GenomeJSON {
  const controllerMeta = buildGenomeControllerMeta(genome);

  if (!Object.keys(controllerMeta).length) {
    return networkPayload;
  }

  return {
    ...networkPayload,
    controllerMeta,
  };
}

/**
 * Split one serialized genome into controller metadata and network payload.
 *
 * Import paths treat `controllerMeta` as the only reserved export-owned field;
 * everything else is passed through to `Network.fromJSON()` as the raw network
 * payload.
 *
 * @param serializedGenome - Checkpoint genome object read from persistence.
 * @returns Reserved controller metadata plus the raw network payload.
 */
export function splitSerializedGenomeCheckpoint(serializedGenome: GenomeJSON): {
  controllerMeta?: GenomeControllerMetaJSON;
  networkPayload: Record<string, unknown>;
} {
  const { controllerMeta, ...networkPayload } = serializedGenome;

  return {
    controllerMeta:
      controllerMeta &&
      typeof controllerMeta === 'object' &&
      !Array.isArray(controllerMeta)
        ? (controllerMeta as GenomeControllerMetaJSON)
        : undefined,
    networkPayload,
  };
}

/**
 * Rehydrate controller-owned genome metadata after network restore.
 *
 * This helper restores score- and lineage-side annotations, enforces unique
 * stable genome ids within the imported population, and returns the next id
 * floor that later imports or offspring generation must stay above.
 *
 * @param genome - Rehydrated live genome instance.
 * @param controllerMeta - Optional controller metadata from the checkpoint.
 * @param seenGenomeIds - Set tracking stable genome ids already assigned.
 * @param nextAssignedGenomeId - Next fallback genome id when the payload lacks one.
 * @returns Updated next genome id floor after this genome is restored.
 */
export function hydrateGenomeControllerMeta(
  genome: GenomeControllerCarrier,
  controllerMeta: GenomeControllerMetaJSON | undefined,
  seenGenomeIds: Set<number>,
  nextAssignedGenomeId: number,
): number {
  restoreNumericGenomeMeta(genome, controllerMeta);
  restoreGenomeNetworkRngState(genome, controllerMeta);
  restoreGenomeParents(genome, controllerMeta);
  restoreOptionalGenomeString(
    controllerMeta?.compatInnovationMode,
    (compatInnovationMode) => {
      genome._compatInnovationMode = compatInnovationMode;
    },
  );

  const restoredGenomeId = resolveRestoredGenomeId(
    controllerMeta,
    nextAssignedGenomeId,
  );

  assignRestoredGenomeId(genome, restoredGenomeId, seenGenomeIds);

  return resolveNextAssignedGenomeId(
    controllerMeta,
    nextAssignedGenomeId,
    restoredGenomeId,
  );
}

function restoreNumericGenomeMeta(
  genome: GenomeControllerCarrier,
  controllerMeta: GenomeControllerMetaJSON | undefined,
): void {
  applyNumericGenomeMetaAssignments([
    {
      value: controllerMeta?.score,
      apply(value) {
        genome.score = value;
      },
    },
    {
      value: controllerMeta?.mutationRate,
      apply(value) {
        genome._mutRate = value;
      },
    },
    {
      value: controllerMeta?.mutationAmount,
      apply(value) {
        genome._mutAmount = value;
      },
    },
    {
      value: controllerMeta?.sharedFitness,
      apply(value) {
        genome._sharedFitness = value;
      },
    },
    {
      value: controllerMeta?.crowdingDistance,
      apply(value) {
        genome._crowdingDistance = value;
      },
    },
    {
      value: controllerMeta?.frontRank,
      apply(value) {
        genome._frontRank = value;
      },
    },
    {
      value: controllerMeta?.structuralEntropy,
      apply(value) {
        genome._structuralEntropy = value;
      },
    },
    {
      value: controllerMeta?.multiObjectiveRank,
      apply(value) {
        genome._moRank = value;
      },
    },
    {
      value: controllerMeta?.multiObjectiveCrowding,
      apply(value) {
        genome._moCrowd = value;
      },
    },
    {
      value: controllerMeta?.depth,
      apply(value) {
        genome._depth = value;
      },
    },
    {
      value: controllerMeta?.reenableProb,
      apply(value) {
        genome._reenableProb = value;
      },
    },
    {
      value: controllerMeta?.reenableSuccess,
      apply(value) {
        genome._reenableSuccess = value;
      },
    },
    {
      value: controllerMeta?.reenableAttempts,
      apply(value) {
        genome._reenableAttempts = value;
      },
    },
    {
      value: controllerMeta?.novelty,
      apply(value) {
        genome._novelty = value;
      },
    },
  ]);
}

function applyNumericGenomeMetaAssignments(
  assignments: NumericGenomeMetaAssignment[],
): void {
  for (const assignment of assignments) {
    if (typeof assignment.value === 'number') {
      assignment.apply(assignment.value);
    }
  }
}

function restoreGenomeNetworkRngState(
  genome: GenomeControllerCarrier,
  controllerMeta: GenomeControllerMetaJSON | undefined,
): void {
  if (
    typeof controllerMeta?.networkRngState === 'number' &&
    typeof genome.setRNGState === 'function'
  ) {
    genome.setRNGState(controllerMeta.networkRngState);
  }
}

function restoreGenomeParents(
  genome: GenomeControllerCarrier,
  controllerMeta: GenomeControllerMetaJSON | undefined,
): void {
  if (Array.isArray(controllerMeta?.parents)) {
    genome._parents = controllerMeta.parents.filter(
      (parentId): parentId is number => typeof parentId === 'number',
    );
  }
}

function restoreOptionalGenomeString<T extends string>(
  value: T | undefined,
  apply: (value: T) => void,
): void {
  if (typeof value === 'string') {
    apply(value as T);
  }
}

function resolveRestoredGenomeId(
  controllerMeta: GenomeControllerMetaJSON | undefined,
  nextAssignedGenomeId: number,
): number {
  return typeof controllerMeta?.genomeId === 'number'
    ? controllerMeta.genomeId
    : nextAssignedGenomeId;
}

function assignRestoredGenomeId(
  genome: GenomeControllerCarrier,
  restoredGenomeId: number,
  seenGenomeIds: Set<number>,
): void {
  if (seenGenomeIds.has(restoredGenomeId)) {
    throw new NeatExportPopulationValidationError(
      `Population snapshots must not reuse genome id ${restoredGenomeId}.`,
    );
  }

  genome._id = restoredGenomeId;
  seenGenomeIds.add(restoredGenomeId);
}

function resolveNextAssignedGenomeId(
  controllerMeta: GenomeControllerMetaJSON | undefined,
  nextAssignedGenomeId: number,
  restoredGenomeId: number,
): number {
  return typeof controllerMeta?.genomeId === 'number'
    ? Math.max(nextAssignedGenomeId, restoredGenomeId + 1)
    : nextAssignedGenomeId + 1;
}

/**
 * Build the controller-owned metadata pocket for one genome.
 *
 * Only fields that are currently present on the live genome are copied into the
 * checkpoint metadata so exported payloads stay compact and omission remains
 * meaningful.
 *
 * @param genome - Live genome carrying controller-owned annotations.
 * @returns Controller metadata object for checkpoint export.
 */
function buildGenomeControllerMeta(
  genome: GenomeControllerCarrier,
): GenomeControllerMetaJSON {
  const controllerMeta: GenomeControllerMetaJSON = {};

  if (typeof genome.score === 'number') controllerMeta.score = genome.score;
  if (typeof genome._id === 'number') controllerMeta.genomeId = genome._id;
  if (typeof genome._mutRate === 'number') {
    controllerMeta.mutationRate = genome._mutRate;
  }
  if (typeof genome._mutAmount === 'number') {
    controllerMeta.mutationAmount = genome._mutAmount;
  }
  const networkRngState = genome.getRNGState?.();
  if (typeof networkRngState === 'number') {
    controllerMeta.networkRngState = networkRngState;
  }
  if (typeof genome._sharedFitness === 'number') {
    controllerMeta.sharedFitness = genome._sharedFitness;
  }
  if (typeof genome._crowdingDistance === 'number') {
    controllerMeta.crowdingDistance = genome._crowdingDistance;
  }
  if (typeof genome._frontRank === 'number') {
    controllerMeta.frontRank = genome._frontRank;
  }
  if (typeof genome._structuralEntropy === 'number') {
    controllerMeta.structuralEntropy = genome._structuralEntropy;
  }
  if (typeof genome._moRank === 'number') {
    controllerMeta.multiObjectiveRank = genome._moRank;
  }
  if (typeof genome._moCrowd === 'number') {
    controllerMeta.multiObjectiveCrowding = genome._moCrowd;
  }
  if (Array.isArray(genome._parents)) {
    controllerMeta.parents = genome._parents.slice();
  }
  if (typeof genome._depth === 'number') controllerMeta.depth = genome._depth;
  if (typeof genome._reenableProb === 'number') {
    controllerMeta.reenableProb = genome._reenableProb;
  }
  if (typeof genome._reenableSuccess === 'number') {
    controllerMeta.reenableSuccess = genome._reenableSuccess;
  }
  if (typeof genome._reenableAttempts === 'number') {
    controllerMeta.reenableAttempts = genome._reenableAttempts;
  }
  if (typeof genome._compatInnovationMode === 'string') {
    controllerMeta.compatInnovationMode = genome._compatInnovationMode;
  }
  if (typeof genome._novelty === 'number') {
    controllerMeta.novelty = genome._novelty;
  }

  return controllerMeta;
}
