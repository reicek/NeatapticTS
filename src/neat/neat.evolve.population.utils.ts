/*
 * ESLint configuration for intentional `any` usage in NEAT evolution population utils
 *
 * This file mirrors the evolution module's runtime metadata handling,
 * where dynamic properties are attached to genomes/species at runtime.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import Network from '../architecture/network';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
} from './neat.evolve.types';
import {
  promoteGenomeToFeedForwardIntentWhenEligible,
  usesFeedForwardMutationPolicy,
} from './neat.topology-intent.utils';

/**
 * Build the next population (elitism, provenance, offspring).
 * @param internal - NEAT controller instance.
 * @param helpers - Helper callbacks for population construction.
 * @param helpers.applyElitism - Elitism helper.
 * @param helpers.applyProvenance - Provenance helper.
 * @param helpers.addOffspring - Offspring helper.
 * @returns next population array.
 */
export async function buildNextPopulation(
  internal: NeatControllerForEvolution,
  helpers: {
    applyElitism: (nextPopulation: Network[]) => void;
    applyProvenance: (nextPopulation: Network[]) => void;
    addOffspring: (nextPopulation: Network[]) => Promise<void>;
  },
): Promise<Network[]> {
  // Step 1: Initialize the population container.
  const nextPopulation: Network[] = [];
  // Step 2: Add elites and provenance genomes.
  helpers.applyElitism(nextPopulation);
  helpers.applyProvenance(nextPopulation);
  // Step 3: Fill remaining slots with offspring.
  await helpers.addOffspring(nextPopulation);
  return nextPopulation;
}

/**
 * Ensure new population meets structural constraints.
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Population to validate.
 * @returns void.
 */
export async function enforcePopulationConstraints(
  internal: NeatControllerForEvolution,
  nextPopulation: Network[],
): Promise<void> {
  // Step 1: Ensure minimum hidden nodes and no dead ends.
  for (const genome of nextPopulation) {
    if (!genome) continue;
    await internal.ensureMinHiddenNodes?.(genome as never);
    await internal.ensureNoDeadEnds?.(genome as never);
  }
}

/**
 * Apply elitism for the next generation.
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @returns void.
 */
export function applyElitism(
  internal: NeatControllerForEvolution,
  nextPopulation: Network[],
): void {
  // Step 1: Compute elitism count.
  const elitismCount = Math.max(
    0,
    Math.min(internal.options.elitism || 0, internal.population.length),
  );
  // Step 2: Copy elites.
  for (let index = 0; index < elitismCount; index++) {
    const elite = internal.population[index];
    if (elite) nextPopulation.push(elite as never);
  }
}

/**
 * Add provenance genomes into the next population.
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @returns void.
 */
export function applyProvenance(
  internal: NeatControllerForEvolution,
  nextPopulation: Network[],
): void {
  // Step 1: Compute desired population and remaining slots.
  const desiredPopulation = Math.max(0, internal.options.popsize || 0);
  const remainingSlotsAfterElites = Math.max(
    0,
    desiredPopulation - nextPopulation.length,
  );
  const provenanceCount = Math.max(
    0,
    Math.min(internal.options.provenance || 0, remainingSlotsAfterElites),
  );
  const shouldPromoteFeedForwardIntent = usesFeedForwardMutationPolicy(
    internal.options.mutation,
  );

  // Step 2: Insert provenance genomes.
  for (let index = 0; index < provenanceCount; index++) {
    if (internal.options.network) {
      const provenanceGenome = Network.fromJSON(internal.options.network.toJSON());

      // Step 2.1: Preserve feed-forward intent when the seed topology is eligible.
      promoteGenomeToFeedForwardIntentWhenEligible(
        provenanceGenome,
        shouldPromoteFeedForwardIntent,
      );

      nextPopulation.push(provenanceGenome);
    } else {
      const provenanceGenome = new Network(internal.input, internal.output, {
        minHidden: internal.options.minHidden,
      });

      // Step 2.2: Promote fresh provenance genomes when FFW is the active contract.
      promoteGenomeToFeedForwardIntentWhenEligible(
        provenanceGenome,
        shouldPromoteFeedForwardIntent,
      );

      nextPopulation.push(provenanceGenome);
    }
  }
}

/**
 * Add offspring to fill remaining population slots.
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @param helpers - Helper callbacks for offspring selection.
 * @param helpers.addSpeciatedOffspring - Speciated offspring helper.
 * @param helpers.addUnspeciatedOffspring - Unspeciated offspring helper.
 * @returns void.
 */
export async function addOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: Network[],
  helpers: {
    addSpeciatedOffspring: (
      nextPopulation: Network[],
      remainingSlots: number,
    ) => Promise<void>;
    addUnspeciatedOffspring: (
      nextPopulation: Network[],
      remainingSlots: number,
    ) => Promise<void>;
  },
): Promise<void> {
  // Step 1: Compute remaining slots.
  const desiredPopulation = Math.max(0, internal.options.popsize || 0);
  const remainingSlots = Math.max(0, desiredPopulation - nextPopulation.length);
  if (remainingSlots <= 0) return;
  // Step 2: Branch based on speciation.
  if (internal.options.speciation && (internal._species?.length ?? 0) > 0) {
    internal._suppressTournamentError = true;
    await helpers.addSpeciatedOffspring(nextPopulation, remainingSlots);
    internal._suppressTournamentError = false;
    return;
  }
  internal._suppressTournamentError = true;
  await helpers.addUnspeciatedOffspring(nextPopulation, remainingSlots);
  internal._suppressTournamentError = false;
}

/**
 * Add offspring when speciation is enabled.
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @param remainingSlots - Slots remaining to fill.
 * @param config - Offspring allocation constants.
 * @returns void.
 */
export async function addSpeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: Network[],
  remainingSlots: number,
  config: {
    minOffspringDefault: number;
    survivalThresholdDefault: number;
    youngThresholdDefault: number;
    youngMultiplierDefault: number;
    oldThresholdDefault: number;
    oldMultiplierDefault: number;
    crossSpeciesGuardLimit: number;
  },
): Promise<void> {
  // Step 1: Allocate offspring counts by species.
  const offspringAllocation = computeOffspringAllocation(
    internal,
    remainingSlots,
    {
      minOffspringDefault: config.minOffspringDefault,
      youngThresholdDefault: config.youngThresholdDefault,
      youngMultiplierDefault: config.youngMultiplierDefault,
      oldThresholdDefault: config.oldThresholdDefault,
      oldMultiplierDefault: config.oldMultiplierDefault,
    },
  );
  // Step 2: Record allocations for telemetry.
  internal._lastOffspringAlloc = (internal._species ?? []).map(
    (species: any, speciesIndex: number) => ({
      id: species.id,
      alloc: offspringAllocation[speciesIndex] || 0,
    }),
  );
  // Step 3: Breed within each species.
  internal._prevInbreedingCount = internal._lastInbreedingCount;
  internal._lastInbreedingCount = 0;
  offspringAllocation.forEach((count, speciesIndex) => {
    if (count <= 0) return;
    const species = internal._species?.[speciesIndex];
    if (!species) return;
    internal._sortSpeciesMembers?.(species);
    const survivors = species.members.slice(
      0,
      Math.max(
        1,
        Math.floor(
          species.members.length *
            (internal.options.survivalThreshold ??
              config.survivalThresholdDefault),
        ),
      ),
    );
    for (let offspringIndex = 0; offspringIndex < count; offspringIndex++) {
      const offspring = buildSpeciesOffspring(
        internal,
        survivors,
        speciesIndex,
        internal.options.crossSpeciesMatingProb || 0,
        config.crossSpeciesGuardLimit,
        config.survivalThresholdDefault,
      );
      nextPopulation.push(offspring as never);
    }
  });
}

/**
 * Add offspring when speciation is disabled.
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @param remainingSlots - Slots remaining to fill.
 * @returns void.
 */
export async function addUnspeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: Network[],
  remainingSlots: number,
): Promise<void> {
  // Step 1: Generate offspring via global selection.
  for (
    let offspringIndex = 0;
    offspringIndex < remainingSlots;
    offspringIndex++
  ) {
    nextPopulation.push((await internal.getOffspring?.()) as never);
  }
}

/**
 * Compute offspring allocation per species.
 * @param internal - NEAT controller instance.
 * @param remainingSlots - Slots remaining to fill.
 * @param config - Allocation constants.
 * @returns allocation per species index.
 */
function computeOffspringAllocation(
  internal: NeatControllerForEvolution,
  remainingSlots: number,
  config: {
    minOffspringDefault: number;
    youngThresholdDefault: number;
    youngMultiplierDefault: number;
    oldThresholdDefault: number;
    oldMultiplierDefault: number;
  },
): number[] {
  // Step 1: Resolve age bonus configuration.
  const ageConfig = internal.options.speciesAgeBonus || {};
  const youngThreshold =
    ageConfig.youngThreshold ?? config.youngThresholdDefault;
  const youngMultiplier =
    ageConfig.youngMultiplier ?? config.youngMultiplierDefault;
  const oldThreshold = ageConfig.oldThreshold ?? config.oldThresholdDefault;
  const oldMultiplier = ageConfig.oldMultiplier ?? config.oldMultiplierDefault;
  // Step 2: Compute adjusted fitness per species.
  const speciesAdjusted = (internal._species ?? []).map((species: any) => {
    const base = species.members.reduce(
      (sum: number, member: any) => sum + (member.score || 0),
      0,
    );
    const age = internal.generation - species.lastImproved;
    if (age <= youngThreshold) return base * youngMultiplier;
    if (age >= oldThreshold) return base * oldMultiplier;
    return base;
  });
  // Step 3: Compute raw shares.
  const totalAdjusted =
    speciesAdjusted.reduce((sum: number, value: number) => sum + value, 0) || 1;
  const rawShares = (internal._species ?? []).map(
    (_: any, speciesIndex: number) =>
      (speciesAdjusted[speciesIndex] / totalAdjusted) * remainingSlots,
  );
  // Step 4: Floor allocations.
  const offspringAllocation = rawShares.map((share) => Math.floor(share));
  enforceMinimumOffspring(
    internal,
    offspringAllocation,
    remainingSlots,
    config.minOffspringDefault,
  );
  distributeRemainingSlots(offspringAllocation, rawShares, remainingSlots);
  trimOversubscription(
    internal,
    offspringAllocation,
    remainingSlots,
    config.minOffspringDefault,
  );
  return offspringAllocation;
}

/**
 * Enforce minimum offspring per species when possible.
 * @param internal - NEAT controller instance.
 * @param allocation - Allocation array to adjust.
 * @param remainingSlots - Total slots available.
 * @param minOffspringDefault - Default minimum offspring.
 * @returns void.
 */
function enforceMinimumOffspring(
  internal: NeatControllerForEvolution,
  allocation: number[],
  remainingSlots: number,
  minOffspringDefault: number,
): void {
  // Step 1: Resolve minimum offspring policy.
  const minOffspring =
    internal.options.speciesAllocation?.minOffspring ?? minOffspringDefault;
  const speciesCount = internal._species?.length ?? 0;
  if (remainingSlots < speciesCount * minOffspring) return;
  // Step 2: Enforce minimum for each species.
  for (let speciesIndex = 0; speciesIndex < allocation.length; speciesIndex++) {
    if (allocation[speciesIndex] < minOffspring)
      allocation[speciesIndex] = minOffspring;
  }
}

/**
 * Distribute leftover slots by fractional remainders.
 * @param allocation - Allocation array to adjust.
 * @param rawShares - Raw fractional shares.
 * @param remainingSlots - Total slots available.
 * @returns void.
 */
function distributeRemainingSlots(
  allocation: number[],
  rawShares: number[],
  remainingSlots: number,
): void {
  // Step 1: Compute slots left after flooring.
  const allocated = allocation.reduce((sum, value) => sum + value, 0);
  let slotsLeft = remainingSlots - allocated;
  if (slotsLeft <= 0) return;
  // Step 2: Distribute by largest fractional remainder.
  const remainders = rawShares
    .map((share, speciesIndex) => ({
      speciesIndex,
      fraction: share - Math.floor(share),
    }))
    .toSorted((left, right) => right.fraction - left.fraction);
  if (remainders.length === 0) return;
  let remainderIndex = 0;
  while (slotsLeft > 0) {
    const targetSpeciesIndex = remainders[remainderIndex].speciesIndex;
    allocation[targetSpeciesIndex]++;
    slotsLeft--;
    remainderIndex = (remainderIndex + 1) % remainders.length;
  }
}

/**
 * Trim allocations when oversubscribed.
 * @param internal - NEAT controller instance.
 * @param allocation - Allocation array to adjust.
 * @param remainingSlots - Total slots available.
 * @param minOffspringDefault - Default minimum offspring.
 * @returns void.
 */
function trimOversubscription(
  internal: NeatControllerForEvolution,
  allocation: number[],
  remainingSlots: number,
  minOffspringDefault: number,
): void {
  // Step 1: Compute oversubscription.
  const allocated = allocation.reduce((sum, value) => sum + value, 0);
  let slotsLeft = remainingSlots - allocated;
  if (slotsLeft >= 0) return;
  // Step 2: Trim from largest allocations while respecting minimum.
  const minOffspring =
    internal.options.speciesAllocation?.minOffspring ?? minOffspringDefault;
  const order = allocation
    .map((value, speciesIndex) => ({ speciesIndex, value }))
    .toSorted((left, right) => right.value - left.value);
  if (order.length === 0) return;
  let didTrim = true;
  while (slotsLeft < 0 && didTrim) {
    didTrim = false;
    for (const entry of order) {
      if (slotsLeft === 0) break;
      if (allocation[entry.speciesIndex] > minOffspring) {
        allocation[entry.speciesIndex]--;
        slotsLeft++;
        didTrim = true;
      }
    }
  }
}

/**
 * Build a single offspring within a species.
 * @param internal - NEAT controller instance.
 * @param survivors - Survivors pool for selection.
 * @param speciesIndex - Species index.
 * @param crossSpeciesProbability - Cross-species mating probability.
 * @param crossSpeciesGuardLimit - Retry guard for cross-species selection.
 * @returns offspring genome.
 */
function buildSpeciesOffspring(
  internal: NeatControllerForEvolution,
  survivors: GenomeWithMetadata[],
  speciesIndex: number,
  crossSpeciesProbability: number,
  crossSpeciesGuardLimit: number,
  survivalThresholdDefault: number,
): GenomeWithMetadata {
  // Step 1: Select first parent.
  const parentA =
    survivors[Math.floor(internal._getRNG()() * survivors.length)];
  // Step 2: Select second parent.
  const parentB = selectSecondParent(
    internal,
    survivors,
    speciesIndex,
    crossSpeciesProbability,
    crossSpeciesGuardLimit,
    survivalThresholdDefault,
  );
  // Step 3: Cross over and assign lineage metadata.
  const child = Network.crossOver(
    parentA as never,
    parentB as never,
    internal.options.equal || false,
  ) as never as GenomeWithMetadata;
  child._reenableProb = internal.options.reenableProb;
  child._id = internal._nextGenomeId++;
  if (internal._lineageEnabled) {
    child._parents = [
      (parentA as never as GenomeWithMetadata)._id,
      (parentB as any)._id,
    ];
    const depthA = (parentA as any)._depth ?? 0;
    const depthB = (parentB as any)._depth ?? 0;
    (child as any)._depth = 1 + Math.max(depthA, depthB);
    if (
      (parentA as never as GenomeWithMetadata)._id ===
      (parentB as never as GenomeWithMetadata)._id
    ) {
      internal._lastInbreedingCount++;
    }
  }
  return child;
}

/**
 * Select a second parent, optionally from another species.
 * @param internal - NEAT controller instance.
 * @param survivors - Survivors pool from the current species.
 * @param speciesIndex - Current species index.
 * @param crossSpeciesProbability - Probability to cross species.
 * @param crossSpeciesGuardLimit - Retry guard for cross-species selection.
 * @returns chosen parent genome.
 */
function selectSecondParent(
  internal: NeatControllerForEvolution,
  survivors: GenomeWithMetadata[],
  speciesIndex: number,
  crossSpeciesProbability: number,
  crossSpeciesGuardLimit: number,
  survivalThresholdDefault: number,
): GenomeWithMetadata {
  // Step 1: Determine whether to cross species.
  const shouldCross =
    crossSpeciesProbability > 0 &&
    (internal._species?.length ?? 0) > 1 &&
    internal._getRNG()() < crossSpeciesProbability;
  if (!shouldCross) {
    return survivors[
      Math.floor(internal._getRNG()() * survivors.length)
    ] as never;
  }
  // Step 2: Choose another species (bounded retries).
  let otherIndex = speciesIndex;
  let guard = 0;
  while (otherIndex === speciesIndex && guard++ < crossSpeciesGuardLimit) {
    otherIndex = Math.floor(
      internal._getRNG()() * (internal._species?.length ?? 1),
    );
  }
  const otherSpecies = internal._species?.[otherIndex];
  if (!otherSpecies) {
    return survivors[
      Math.floor(internal._getRNG()() * survivors.length)
    ] as never;
  }
  // Step 3: Select parent from the other species.
  internal._sortSpeciesMembers?.(otherSpecies);
  const otherSurvivors = otherSpecies.members.slice(
    0,
    Math.max(
      1,
      Math.floor(
        otherSpecies.members.length *
          (internal.options.survivalThreshold ?? survivalThresholdDefault),
      ),
    ),
  );
  return otherSurvivors[
    Math.floor(internal._getRNG()() * otherSurvivors.length)
  ] as never;
}
