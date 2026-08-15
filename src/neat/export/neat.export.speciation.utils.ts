import {
  NeatExportStateBundleValidationError,
  NeatExportStateControllerRestoreError,
} from './neat.export.errors';
import {
  hydrateGenomeControllerMeta,
  serializeGenomeCheckpoint,
  splitSerializedGenomeCheckpoint,
} from './neat.export.population.utils';
import type {
  GenomeControllerCarrier,
  NeatControllerForExport,
  NetworkClass,
  SpeciationCheckpointJSON,
  SpeciesCheckpointJSON,
} from './neat.export.types';
import type { SpeciesLastStats } from '../shared/neat.shared.types';

/**
 * Full-checkpoint speciation helpers.
 *
 * Speciation is replay-sensitive because it binds historical context to the
 * current generation: species membership, representative anchors, and
 * compatibility threshold state all influence future reproduction.
 *
 * For deterministic restore, we serialize species membership by stable genome
 * id rather than by array index.
 *
 * ```mermaid
 * flowchart LR
 *   A[Restored population] -->|build| B[(genomesById)]
 *   C[Species rows\nmemberGenomeIds] -->|rebind| D[Live species registry]
 *   B --> D
 * ```
 */

/**
 * Serialize the live speciation registry into a full-checkpoint payload.
 *
 * Species membership is written by stable genome id so restore can rebind the
 * registry onto freshly rehydrated network instances without relying on array
 * position or permissive structural matching.
 *
 * @param internal - Live controller host.
 * @returns Serializable speciation checkpoint payload.
 * @throws {NeatExportStateControllerRestoreError} When a species member or representative lacks a stable genome id.
 */
export function serializeSpeciationCheckpoint(
  internal: NeatControllerForExport,
): SpeciationCheckpointJSON {
  const livePopulationIds = collectLivePopulationIds(internal);

  return {
    nextSpeciesId: internal._nextSpeciesId,
    species: serializeSpeciesCheckpointRows(internal, livePopulationIds),
    speciesCreated: Array.from(internal._speciesCreated?.entries() ?? []),
    prevSpeciesMembers: serializePrevSpeciesMembers(internal),
    speciesLastStats: serializeSpeciesLastStats(internal),
    compatIntegral: internal._compatIntegral,
    compatSpeciesEMA: internal._compatSpeciesEMA,
  };
}

/**
 * Collect the set of stable genome ids present in the live population.
 *
 * This set drives the decision of whether a species representative must be
 * serialized as a detached anchor (when it no longer lives in the current
 * population).
 *
 * @param internal - Live controller host.
 * @returns Set of stable genome ids in the current population.
 */
function collectLivePopulationIds(
  internal: NeatControllerForExport,
): Set<number> {
  return new Set(
    internal.population.flatMap((genome) =>
      typeof genome._id === 'number' ? [genome._id] : [],
    ),
  );
}

/**
 * Serialize all species rows from the live registry.
 *
 * Each row writes member ids by stable genome id and optionally carries a
 * detached representative anchor when the representative is no longer in the
 * live population.
 *
 * @param internal - Live controller host.
 * @param livePopulationIds - Stable genome ids present in the current population.
 * @returns Array of serialized species checkpoint rows.
 */
function serializeSpeciesCheckpointRows(
  internal: NeatControllerForExport,
  livePopulationIds: Set<number>,
): SpeciesCheckpointJSON[] {
  return (internal._species ?? []).map((species) => ({
    id: species.id,
    memberGenomeIds: species.members.map((member, memberIndex) =>
      readRequiredGenomeId(
        member,
        `species ${species.id} member ${memberIndex}`,
      ),
    ),
    representativeGenomeId: species.representative
      ? readRequiredGenomeId(
          species.representative,
          `species ${species.id} representative`,
        )
      : undefined,
    representativeGenome:
      species.representative &&
      shouldSerializeRepresentativeAnchor(
        species.representative,
        livePopulationIds,
      )
        ? serializeGenomeCheckpoint(species.representative)
        : undefined,
    bestScore: species.bestScore,
    lastImproved: species.lastImproved,
    sharedFitness: species.sharedFitness,
    avgSharedFitness: species.avgSharedFitness,
    offspring: species.offspring,
    generation: species.generation,
  }));
}

/**
 * Serialize the previous-generation species member id map for checkpoint continuity.
 *
 * @param internal - Live controller host.
 * @returns Array of species-id to member-id-list pairs.
 */
function serializePrevSpeciesMembers(
  internal: NeatControllerForExport,
): Array<[number, number[]]> {
  return Array.from(
    internal._prevSpeciesMembers?.entries() ?? [],
    ([speciesId, memberIds]) => [speciesId, Array.from(memberIds)],
  );
}

/**
 * Serialize per-species rolling statistics with deep-cloned payloads.
 *
 * @param internal - Live controller host.
 * @returns Array of species-id to cloned statistics pairs.
 */
function serializeSpeciesLastStats(
  internal: NeatControllerForExport,
): Array<[number, SpeciesLastStats]> {
  return Array.from(
    internal._speciesLastStats?.entries() ?? [],
    ([speciesId, speciesStats]) => [speciesId, structuredClone(speciesStats)],
  );
}

/**
 * Restore the live species registry from a full-checkpoint payload.
 *
 * The restore path first rebuilds a lookup of imported genomes by stable id,
 * then rebinds each checkpoint species row onto those live instances while
 * restoring the related speciation bookkeeping maps and threshold state.
 *
 * @param neatInstance - Controller instance whose population is already restored.
 * @param speciationCheckpoint - Serialized speciation payload from persistence.
 * @param networkClass - Network constructor class for rehydrating representative anchors.
 * @returns Nothing.
 * @throws {NeatExportStateControllerRestoreError} When species references are invalid.
 */
export function restoreSpeciationCheckpoint(
  neatInstance: NeatControllerForExport,
  speciationCheckpoint: SpeciationCheckpointJSON,
  networkClass: NetworkClass,
): void {
  const genomesById = buildGenomesById(neatInstance.population);

  restoreNextSpeciesId(neatInstance, speciationCheckpoint.nextSpeciesId);
  neatInstance._species = restoreSpeciesRows(
    speciationCheckpoint.species ?? [],
    genomesById,
    networkClass,
  );
  restoreSpeciationCollections(neatInstance, speciationCheckpoint);
  restoreSpeciationThresholdState(neatInstance, speciationCheckpoint);
}

function buildGenomesById(
  population: GenomeControllerCarrier[],
): Map<number, GenomeControllerCarrier> {
  const genomesById = new Map<number, GenomeControllerCarrier>();

  for (const genome of population) {
    if (typeof genome._id !== 'number') {
      throw new NeatExportStateControllerRestoreError(
        'Full checkpoint restore requires stable genome ids on every imported genome.',
      );
    }
    if (genomesById.has(genome._id)) {
      throw new NeatExportStateControllerRestoreError(
        `Full checkpoint restore encountered duplicate genome id ${genome._id}.`,
      );
    }

    genomesById.set(genome._id, genome);
  }

  return genomesById;
}

function restoreNextSpeciesId(
  neatInstance: NeatControllerForExport,
  nextSpeciesId: number | undefined,
): void {
  if (typeof nextSpeciesId === 'number') {
    neatInstance._nextSpeciesId = nextSpeciesId;
  }
}

function restoreSpeciesRows(
  speciesRows: NonNullable<SpeciationCheckpointJSON['species']>,
  genomesById: Map<number, GenomeControllerCarrier>,
  networkClass: NetworkClass,
) {
  return speciesRows.map((species) =>
    restoreSpeciesRow(species, genomesById, networkClass),
  );
}

function restoreSpeciesRow(
  species: NonNullable<SpeciationCheckpointJSON['species']>[number],
  genomesById: Map<number, GenomeControllerCarrier>,
  networkClass: NetworkClass,
) {
  const representativeAnchor = restoreRepresentativeAnchor(
    species.representativeGenome,
    genomesById,
    networkClass,
  );
  const members = restoreSpeciesMembers(
    species.memberGenomeIds,
    genomesById,
    representativeAnchor,
  );
  const representative = resolveRestoredSpeciesRepresentative(
    species,
    genomesById,
    representativeAnchor,
    members,
  );

  assertRepresentativeGenomeResolved(
    species.id,
    species.representativeGenomeId,
    representative,
  );

  return {
    id: species.id,
    members,
    representative,
    bestScore: species.bestScore,
    lastImproved: species.lastImproved,
    sharedFitness: species.sharedFitness,
    avgSharedFitness: species.avgSharedFitness,
    offspring: species.offspring,
    generation: species.generation,
  };
}

function restoreSpeciesMembers(
  memberGenomeIds: number[],
  genomesById: Map<number, GenomeControllerCarrier>,
  representativeAnchor: GenomeControllerCarrier | undefined,
): GenomeControllerCarrier[] {
  return memberGenomeIds.map((genomeId) =>
    resolveRestoredSpeciesMember(genomeId, genomesById, representativeAnchor),
  );
}

function resolveRestoredSpeciesMember(
  genomeId: number,
  genomesById: Map<number, GenomeControllerCarrier>,
  representativeAnchor: GenomeControllerCarrier | undefined,
): GenomeControllerCarrier {
  const liveMember = genomesById.get(genomeId);

  if (liveMember) {
    return liveMember;
  }
  if (representativeAnchor && representativeAnchor._id === genomeId) {
    return representativeAnchor;
  }

  return createCheckpointMemberPlaceholder(genomeId);
}

function resolveRestoredSpeciesRepresentative(
  species: NonNullable<SpeciationCheckpointJSON['species']>[number],
  genomesById: Map<number, GenomeControllerCarrier>,
  representativeAnchor: GenomeControllerCarrier | undefined,
  members: GenomeControllerCarrier[],
): GenomeControllerCarrier | undefined {
  return typeof species.representativeGenomeId === 'number'
    ? (genomesById.get(species.representativeGenomeId) ?? representativeAnchor)
    : members[0];
}

function assertRepresentativeGenomeResolved(
  speciesId: number,
  representativeGenomeId: number | undefined,
  representative: GenomeControllerCarrier | undefined,
): void {
  if (representativeGenomeId != null && !representative) {
    throw new NeatExportStateControllerRestoreError(
      `Species checkpoint ${speciesId} references missing representative genome id ${representativeGenomeId}. Export the checkpoint again with the current replay-aware contract before resuming this speciation boundary.`,
    );
  }
}

function restoreSpeciationCollections(
  neatInstance: NeatControllerForExport,
  speciationCheckpoint: SpeciationCheckpointJSON,
): void {
  neatInstance._speciesCreated = new Map(
    speciationCheckpoint.speciesCreated ?? [],
  );
  neatInstance._prevSpeciesMembers = new Map(
    (speciationCheckpoint.prevSpeciesMembers ?? []).map(
      ([speciesId, memberIds]) => [speciesId, new Set(memberIds)],
    ),
  );
  neatInstance._speciesLastStats = new Map(
    speciationCheckpoint.speciesLastStats ?? [],
  );
}

function restoreSpeciationThresholdState(
  neatInstance: NeatControllerForExport,
  speciationCheckpoint: SpeciationCheckpointJSON,
): void {
  if (typeof speciationCheckpoint.compatIntegral === 'number') {
    neatInstance._compatIntegral = speciationCheckpoint.compatIntegral;
  }
  if (typeof speciationCheckpoint.compatSpeciesEMA === 'number') {
    neatInstance._compatSpeciesEMA = speciationCheckpoint.compatSpeciesEMA;
  }
}

/**
 * Read the stable genome id required by checkpointed species state.
 *
 * Full checkpoints refer back to live genomes by id, so species export must
 * fail immediately when a referenced genome does not carry one.
 *
 * @param genome - Live genome referenced by checkpoint state.
 * @param contextLabel - Human-readable export context for diagnostics.
 * @returns Stable genome id.
 * @throws {NeatExportStateBundleValidationError} When the genome lacks an id.
 */
function readRequiredGenomeId(
  genome: GenomeControllerCarrier,
  contextLabel: string,
): number {
  if (typeof genome._id === 'number') {
    return genome._id;
  }

  throw new NeatExportStateBundleValidationError(
    `Cannot export ${contextLabel} without a stable genome id.`,
  );
}

/**
 * Decide whether the checkpoint must carry a detached representative anchor.
 *
 * Generation-boundary checkpoints can legally keep a previous-generation
 * representative even after the live population has already been replaced.
 * When that happens, replay needs the representative's full structure because
 * the next speciation pass compares the new population against that anchor.
 *
 * @param representative - Live representative genome.
 * @param livePopulationIds - Stable genome ids present in the current population.
 * @returns Whether the representative must be serialized explicitly.
 */
function shouldSerializeRepresentativeAnchor(
  representative: GenomeControllerCarrier,
  livePopulationIds: Set<number>,
): boolean {
  return (
    typeof representative._id === 'number' &&
    !livePopulationIds.has(representative._id)
  );
}

/**
 * Restore a detached representative anchor from checkpoint JSON.
 *
 * The representative snapshot is only needed when the controller checkpoint was
 * taken between generations and the live species registry still points at a
 * prior-generation anchor. That anchor must keep its structural graph so the
 * next speciation pass can compare new genomes against the same reference.
 *
 * @param representativeGenome - Optional serialized representative checkpoint.
 * @param liveGenomesById - Restored current-population genomes keyed by id.
 * @param networkClass - Network class used to rebuild serialized genomes.
 * @returns Restored representative anchor when one was exported.
 */
function restoreRepresentativeAnchor(
  representativeGenome: Record<string, unknown> | undefined,
  liveGenomesById: Map<number, GenomeControllerCarrier>,
  networkClass: NetworkClass,
): GenomeControllerCarrier | undefined {
  if (!representativeGenome) {
    return undefined;
  }

  const { controllerMeta, networkPayload } =
    splitSerializedGenomeCheckpoint(representativeGenome);
  const representativeAnchor = networkClass.fromJSON(networkPayload);
  const seenGenomeIds = new Set(liveGenomesById.keys());

  hydrateGenomeControllerMeta(
    representativeAnchor,
    controllerMeta,
    seenGenomeIds,
    1,
  );

  return representativeAnchor;
}

/**
 * Create a lightweight placeholder for a previous-generation species member.
 *
 * Full replay only needs these historical members for their stable ids because
 * `_speciate()` snapshots member ids before immediately clearing the live member
 * arrays. Detached placeholders therefore preserve continuity without forcing
 * the checkpoint to duplicate every historical genome payload.
 *
 * @param genomeId - Stable genome id referenced by the historical species row.
 * @returns Minimal carrier exposing the required stable id.
 */
function createCheckpointMemberPlaceholder(
  genomeId: number,
): GenomeControllerCarrier {
  return {
    _id: genomeId,
    toJSON: () => ({
      controllerMeta: {
        genomeId,
      },
    }),
  };
}
