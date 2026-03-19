import type {
  GenomeDetailed,
  SpeciesLike,
  SpeciationOptions,
  SpeciationHarnessContext,
} from '../../shared/neat.shared.types';
import {
  DEFAULT_COMPATIBILITY_THRESHOLD,
  NEGATIVE_INFINITY,
} from '../shared/speciation.shared';

/**
 * Speciation assignment mechanics.
 *
 * This chapter owns the population-to-species remap that sits at the front of
 * every speciation pass. The root speciation chapter explains why the
 * controller keeps species at all; this file explains how the live registry is
 * rebuilt once a new generation is ready to be grouped.
 *
 * Read the assignment flow in four stages:
 *
 * 1. preserve the previous membership picture so telemetry and history can
 *    compare "before" and "after",
 * 2. clear live member arrays without discarding the long-lived species
 *    records themselves,
 * 3. walk the population and either match each genome to an existing
 *    representative or create a fresh species,
 * 4. refresh representatives so later threshold, history, sharing, and
 *    stagnation passes all read a coherent post-assignment registry.
 *
 * The boundary stays intentionally narrow. These helpers decide membership,
 * but they do not tune the compatibility threshold, normalize scores, or write
 * history rows. That separation keeps the speciation pipeline legible: this
 * file answers "where does each genome belong right now?" and the neighboring
 * chapters handle what happens after that answer exists.
 *
 * ```mermaid
 * flowchart TD
 *   Population[Current population]
 *   Snapshot[Snapshot previous memberships]
 *   Reset[Clear live species members]
 *   Match[Match genomes to representatives]
 *   Create[Create new species when no match exists]
 *   Refresh[Refresh representatives and drop empty species]
 *   Downstream[Threshold, history, and sharing passes]
 *
 *   Population --> Snapshot
 *   Snapshot --> Reset
 *   Reset --> Match
 *   Match --> Create
 *   Create --> Refresh
 *   Refresh --> Downstream
 * ```
 */

/**
 * Snapshot current species memberships for telemetry.
 *
 * This preserves the "before reassignment" view of the registry so later
 * telemetry, history, and species-reporting code can compare how memberships
 * moved across the current speciation pass. The helper records only species ids
 * and genome ids because assignment does not need to duplicate full genome
 * state in order to preserve that continuity signal.
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
 * Assignment keeps the existing species records, ids, and representatives long
 * enough to reuse them as comparison anchors for the incoming population. What
 * must be cleared is only the live member list, so the next pass can rebuild
 * memberships from scratch instead of accidentally accumulating stale members.
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
 * This is the main assignment walk. Each genome gets one chance to join an
 * existing species by comparing against the current representatives. When no
 * representative falls within the active compatibility threshold, the helper
 * seeds a new species immediately so later genomes can also match against that
 * new lineage during the same pass.
 *
 * That "match or create" rule is what keeps the registry coherent for later
 * threshold adaptation and score sharing: by the time this helper finishes,
 * every genome belongs to exactly one live species.
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
 * After assignment, some prior species may have lost every member and some
 * surviving species need a new representative taken from their rebuilt member
 * list. This helper performs that cleanup so downstream passes do not have to
 * reason about empty shells or stale representative pointers.
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
 * This is the local "does this genome still belong here?" decision at the
 * heart of assignment. The helper compares the genome against each current
 * representative using the controller's compatibility distance and returns the
 * first species that falls inside the active threshold.
 *
 * It deliberately does not rank all possible matches or try to optimize global
 * placement. The assignment contract is smaller: scan the existing registry in
 * deterministic order, accept the first compatible home, otherwise signal that
 * a new species should be created.
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
 * New species creation is the explicit fallback for genomes that do not fit any
 * current representative. The helper allocates a fresh species id, seeds the
 * first member and representative from the incoming genome, initializes the
 * best-score view for later stagnation logic, and records the creation
 * generation for age-aware downstream behavior.
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
