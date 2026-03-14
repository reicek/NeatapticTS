import type {
  GenomeDetailed,
  SpeciesLike,
  SpeciationOptions,
  SpeciationHarnessContext,
} from '../../neat.types';
import {
  DEFAULT_COMPATIBILITY_THRESHOLD,
  NEGATIVE_INFINITY,
} from '../shared/speciation.shared';

/**
 * Speciation assignment mechanics.
 *
 * This chapter covers the part of speciation that decides where genomes go:
 * snapshot the old memberships, clear the species, match genomes against
 * representatives, create new species when needed, and refresh representatives
 * once reassignment is complete.
 */

/**
 * Snapshot current species memberships for telemetry.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function snapshotPreviousMembers<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Ensure the previous map exists and is cleared.
  speciationContext._prevSpeciesMembers =
    speciationContext._prevSpeciesMembers ?? new Map();
  speciationContext._prevSpeciesMembers.clear();
  // Step 2: Capture member IDs per species.
  for (const species of speciationContext._species) {
    const previousMembers = new Set<number>();
    for (const member of species.members as GenomeDetailed[]) {
      previousMembers.add(member._id);
    }
    speciationContext._prevSpeciesMembers.set(species.id, previousMembers);
  }
}

/**
 * Clear member lists for all species.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function resetSpeciesMembers<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Empty each species membership list.
  speciationContext._species.forEach((species: SpeciesLike) => {
    species.members = [];
  });
}

/**
 * Assign each genome in the population to a compatible species.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @returns Nothing.
 */
export function assignPopulationToSpecies<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void {
  // Step 1: Assign each genome to an existing species when compatible.
  for (const genome of speciationContext.population) {
    const matchedSpecies = findCompatibleSpecies(
      speciationContext,
      options,
      genome,
    );
    if (matchedSpecies) {
      matchedSpecies.members.push(genome);
      continue;
    }
    // Step 2: Create a new species when no match is found.
    createSpeciesForGenome(speciationContext, genome);
  }
}

/**
 * Refresh representatives and remove empty species.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function refreshSpeciesRepresentatives<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Remove empty species.
  speciationContext._species = speciationContext._species.filter(
    (species: SpeciesLike) => species.members.length > 0,
  );
  // Step 2: Assign representatives from member lists.
  speciationContext._species.forEach((species: SpeciesLike) => {
    species.representative = species.members[0] as GenomeDetailed;
  });
}

/**
 * Find a compatible species representative for the given genome.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @param genome - Genome to match.
 * @returns Matching species or undefined.
 */
function findCompatibleSpecies<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  genome: GenomeDetailed,
): SpeciesLike | undefined {
  // Step 1: Use the current threshold to match against representatives.
  const compatibilityThreshold =
    options.compatibilityThreshold ?? DEFAULT_COMPATIBILITY_THRESHOLD;
  for (const species of speciationContext._species) {
    const compatibilityDistance = speciationContext._compatibilityDistance(
      genome,
      species.representative as GenomeDetailed,
    );
    if (compatibilityDistance < compatibilityThreshold) return species;
  }
  return undefined;
}

/**
 * Create a new species for the provided genome.
 *
 * @param speciationContext - Speciation harness context.
 * @param genome - Genome that starts a new species.
 * @returns Nothing.
 */
function createSpeciesForGenome<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  genome: GenomeDetailed,
): void {
  // Step 1: Allocate a fresh species id.
  const newSpeciesId = speciationContext._nextSpeciesId++;
  // Step 2: Seed the new species with this genome.
  speciationContext._species.push({
    id: newSpeciesId,
    members: [genome],
    representative: genome,
    lastImproved: speciationContext.generation,
    bestScore: genome.score ?? NEGATIVE_INFINITY,
  });
  speciationContext._speciesCreated.set(
    newSpeciesId,
    speciationContext.generation,
  );
}