import { NeatExportPopulationValidationError } from './neat.export.errors';
import type {
  GenomeControllerCarrier,
  GenomeControllerMetaJSON,
  GenomeJSON,
} from './neat.export.types';

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
  if (typeof controllerMeta?.score === 'number') genome.score = controllerMeta.score;
  if (
    typeof controllerMeta?.networkRngState === 'number' &&
    typeof genome.setRNGState === 'function'
  ) {
    genome.setRNGState(controllerMeta.networkRngState);
  }
  if (typeof controllerMeta?.sharedFitness === 'number') {
    genome._sharedFitness = controllerMeta.sharedFitness;
  }
  if (typeof controllerMeta?.crowdingDistance === 'number') {
    genome._crowdingDistance = controllerMeta.crowdingDistance;
  }
  if (typeof controllerMeta?.frontRank === 'number') {
    genome._frontRank = controllerMeta.frontRank;
  }
  if (typeof controllerMeta?.structuralEntropy === 'number') {
    genome._structuralEntropy = controllerMeta.structuralEntropy;
  }
  if (typeof controllerMeta?.multiObjectiveRank === 'number') {
    genome._moRank = controllerMeta.multiObjectiveRank;
  }
  if (typeof controllerMeta?.multiObjectiveCrowding === 'number') {
    genome._moCrowd = controllerMeta.multiObjectiveCrowding;
  }
  if (Array.isArray(controllerMeta?.parents)) {
    genome._parents = controllerMeta.parents.filter(
      (parentId): parentId is number => typeof parentId === 'number',
    );
  }
  if (typeof controllerMeta?.depth === 'number') genome._depth = controllerMeta.depth;
  if (typeof controllerMeta?.reenableProb === 'number') {
    genome._reenableProb = controllerMeta.reenableProb;
  }
  if (typeof controllerMeta?.reenableSuccess === 'number') {
    genome._reenableSuccess = controllerMeta.reenableSuccess;
  }
  if (typeof controllerMeta?.reenableAttempts === 'number') {
    genome._reenableAttempts = controllerMeta.reenableAttempts;
  }
  if (typeof controllerMeta?.compatInnovationMode === 'string') {
    genome._compatInnovationMode = controllerMeta.compatInnovationMode;
  }
  if (typeof controllerMeta?.novelty === 'number') {
    genome._novelty = controllerMeta.novelty;
  }

  const restoredGenomeId =
    typeof controllerMeta?.genomeId === 'number'
      ? controllerMeta.genomeId
      : nextAssignedGenomeId;

  if (seenGenomeIds.has(restoredGenomeId)) {
    throw new NeatExportPopulationValidationError(
      `Population snapshots must not reuse genome id ${restoredGenomeId}.`,
    );
  }

  genome._id = restoredGenomeId;
  seenGenomeIds.add(restoredGenomeId);

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