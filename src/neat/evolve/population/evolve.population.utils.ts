/*
 * ESLint configuration for intentional `any` usage in NEAT evolution population utils
 *
 * This file mirrors the evolution module's runtime metadata handling,
 * where dynamic properties are attached to genomes/species at runtime.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import Network from '../../../architecture/network/network';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
} from '../evolve.types';
import {
  promoteGenomeToFeedForwardIntentWhenEligible,
  usesFeedForwardMutationPolicy,
} from '../../topology-intent/neat.topology-intent';

/**
 * The evolve-time population boundary assembles the next generation after the
 * current one has already been ranked, summarized, and snapshotted.
 *
 * The root `evolve/` chapter explains the full generation lifecycle, and the
 * `offspring/` chapter explains how one child is crossed and normalized. This
 * file owns the layer between them: once evolve has decided it is time to
 * rebuild, how does the controller preserve elites, inject provenance seeds,
 * allocate the remaining budget across species when needed, and finish with a
 * population that still respects the controller's structural expectations?
 *
 * Read this chapter when you want to understand:
 *
 * - why next-generation assembly stays separate from the broader evolve spine,
 * - how elitism, provenance, and offspring fill the population in a fixed order,
 * - where speciated allocation decides how many children each lineage receives,
 * - why structural cleanup happens after assembly rather than inside every
 *   earlier helper.
 *
 * The helper flow is easiest to retain as four responsibilities:
 *
 * 1. reserve deterministic slots for elites,
 * 2. add fresh provenance genomes when configured,
 * 3. fill the remaining budget through speciated or unspeciated offspring,
 * 4. reapply minimum hidden-node and dead-end constraints to the final result.
 *
 * The two questions that usually make this chapter feel denser than the rest
 * of `evolve/` are:
 *
 * 1. where the remaining population budget actually goes once elites and
 *    provenance have already claimed space,
 * 2. why species-aware offspring filling is split into allocation math first
 *    and child construction second.
 *
 * Read the first chart as the outer generation-building spine. Read the second
 * chart as the inner reproduction-budget story that only activates when a live
 * species registry exists.
 *
 * ```mermaid
 * flowchart TD
 *   Ranked[Ranked current generation] --> Elites[Copy elites]
 *   Elites --> Provenance[Add provenance seeds]
 *   Provenance --> Branch{Speciation active?}
 *   Branch -- Yes --> Allocate[Allocate offspring by species]
 *   Branch -- No --> Global[Fill remaining slots globally]
 *   Allocate --> Offspring[Add offspring]
 *   Global --> Offspring
 *   Offspring --> Constraints[Enforce structural constraints]
 *   Constraints --> Ready[Next population ready for mutation]
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   Budget[Remaining population slots]
 *   Fitness[Adjusted species fitness]
 *   Floor[Floor raw offspring shares]
 *   Minimum[Protect minimum offspring when affordable]
 *   Remainders[Spend leftover slots by remainder]
 *   Trim[Trim oversubscription if guarantees went too far]
 *   Breed[Breed species-local offspring]
 *
 *   Budget --> Fitness
 *   Fitness --> Floor
 *   Floor --> Minimum
 *   Minimum --> Remainders
 *   Remainders --> Trim
 *   Trim --> Breed
 * ```
 */

/* Module introduction boundary for generated README output. */

/**
 * Build the next population (elitism, provenance, offspring).
 *
 * This helper is the orchestration entrypoint for next-generation assembly.
 * It deliberately reads like a short collect-and-fill pipeline: start with an
 * empty container, reserve the slots that should bypass parent selection, then
 * spend the remaining capacity on offspring generation. The mutation and prune
 * phases happen later; this boundary only answers how the raw next population is
 * assembled before those later transforms run.
 *
 * Pedagogically, this is the chapter's "packing list" helper. It does not yet
 * ask whether the chosen genomes are structurally clean enough for the next
 * loop. It only decides which genomes enter the first draft of the next
 * population, and in which order those admission rules are applied.
 *
 * Example:
 *
 * ```ts
 * const nextPopulation = await buildNextPopulation(internal, {
 *   applyElitism: (population) => applyElitism(internal, population),
 *   applyProvenance: (population) => applyProvenance(internal, population),
 *   addOffspring: (population) => addOffspring(internal, population, helpers),
 * });
 * ```
 *
 * @param internal - NEAT controller instance.
 * @param helpers - Helper callbacks for population construction.
 * @param helpers.applyElitism - Elitism helper.
 * @param helpers.applyProvenance - Provenance helper.
 * @param helpers.addOffspring - Offspring helper.
 * @returns Next population array before later mutation and pruning phases.
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
 *
 * Population assembly intentionally separates slot-filling from structural
 * cleanup. Elites may already be valid, provenance genomes may come from a
 * seed network or a fresh constructor path, and offspring may arrive from
 * crossover with small topology issues that the controller routinely repairs.
 * Running those repairs here keeps later evolve code free to assume the new
 * population already satisfies the controller's minimum hidden-node and
 * dead-end expectations.
 *
 * Keeping this repair pass at the end is a deliberate architecture choice. If
 * every earlier helper tried to repair genomes inline, the chapter would blur
 * slot-allocation policy together with structural-safety policy. Centralizing
 * cleanup here keeps the earlier helpers focused on population composition.
 *
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Population to validate.
 * @returns A promise that resolves after best-effort structural cleanup.
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
 *
 * Elitism reserves the deterministic carry-over portion of the population.
 * These genomes bypass parent selection entirely so the best ranked candidates
 * from the current generation survive into the next one unchanged.
 *
 * Read this as the chapter's continuity rule. Before the controller starts
 * gambling on new offspring, it preserves a small slice of already-proven
 * genomes so the next generation cannot forget the current best evidence.
 *
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @returns Nothing.
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
 *
 * Provenance is the population builder's controlled source of fresh starting
 * material. Unlike offspring, these genomes do not depend on current parent
 * selection pressure. They either clone the configured seed network or create a
 * new minimal network, then optionally preserve feed-forward intent so the
 * resulting generation stays aligned with the runtime topology contract.
 *
 * This makes provenance the chapter's controlled exploration valve. Elites
 * preserve what is already working; provenance reintroduces known-safe or fresh
 * starting material without asking the current parent pool for permission.
 *
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @returns Nothing.
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
      const provenanceGenome = Network.fromJSON(
        internal.options.network.toJSON(),
      );

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
 *
 * This helper spends whatever population budget remains after elitism and
 * provenance have claimed their slots. Its main job is not to create children
 * itself, but to choose the correct filling strategy: species-aware allocation
 * when the controller currently maintains a species registry, or global parent
 * selection when it does not.
 *
 * That branch is the main conceptual seam in the chapter. Everything before
 * this point is deterministic packing. This helper is where the controller asks
 * whether the remaining search budget should respect live species boundaries or
 * whether it should fall back to one global parent pool.
 *
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @param helpers - Helper callbacks for offspring selection.
 * @param helpers.addSpeciatedOffspring - Speciated offspring helper.
 * @param helpers.addUnspeciatedOffspring - Unspeciated offspring helper.
 * @returns A promise that resolves after the remaining population budget is filled.
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
 *
 * This is the species-aware branch of population filling. It converts the
 * remaining population budget into per-species child counts, records those
 * counts for later telemetry or diagnostics reads, then breeds within each
 * species using the narrower offspring mechanics described in `offspring/`.
 *
 * The helper stays intentionally focused on allocation and local survivor
 * pools. It does not re-run speciation or mutate the produced children.
 *
 * The important teaching split is that this helper does two different jobs in
 * sequence:
 *
 * 1. decide how much reproductive budget each species deserves,
 * 2. spend each species-local budget through survivor-based crossover.
 *
 * Keeping those jobs together makes the generated chapter longer, but it also
 * keeps the species-aware branch readable in one place instead of scattering the
 * allocation rationale across several tiny helpers.
 *
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @param remainingSlots - Slots remaining to fill.
 * @param config - Offspring allocation constants.
 * @returns A promise that resolves after species-aware offspring have been added.
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
 *
 * When no species registry is active, the population builder falls back to the
 * controller's global offspring-selection path. This keeps the no-speciation
 * branch small and makes the contrast with the species-aware allocator easy to
 * read in the generated chapter.
 *
 * @param internal - NEAT controller instance.
 * @param nextPopulation - Target population array.
 * @param remainingSlots - Slots remaining to fill.
 * @returns A promise that resolves after all remaining slots have been filled.
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
 *
 * Allocation is where the ranked generation turns into concrete reproduction
 * budget. The helper converts species-level adjusted fitness into integer child
 * counts, then layers in minimum-offspring protection plus remainder handling so
 * the final distribution stays both policy-aware and population-size safe.
 *
 * Read this as a small budgeting pipeline rather than one opaque formula:
 *
 * 1. adjust each species' effective fitness with age-sensitive multipliers,
 * 2. translate those adjusted values into fractional offspring shares,
 * 3. turn the shares into integers without losing all protection for small but
 *    still-viable species,
 * 4. repair rounding drift so the final counts still match the remaining slot
 *    budget exactly.
 *
 * @param internal - NEAT controller instance.
 * @param remainingSlots - Slots remaining to fill.
 * @param config - Allocation constants.
 * @returns Offspring allocation per species index.
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
 *
 * This rule prevents species allocation from collapsing entirely onto a few
 * dominant lineages when the remaining slot budget is large enough to preserve
 * a broader search frontier.
 *
 * In other words, this is the chapter's anti-monoculture guard. It only runs
 * when the slot budget is big enough to afford that diversity protection.
 *
 * @param internal - NEAT controller instance.
 * @param allocation - Allocation array to adjust.
 * @param remainingSlots - Total slots available.
 * @param minOffspringDefault - Default minimum offspring.
 * @returns Nothing.
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
 *
 * Flooring raw shares rarely sums exactly to the remaining slot budget. This
 * helper spends the leftover capacity by largest remainder so the final integer
 * allocation stays as close as possible to the original fractional intent.
 *
 * This is the allocation chapter's rounding-fairness step. Without it, small
 * systematic flooring losses would quietly bias the final child counts away
 * from the fractional budget that the controller just computed.
 *
 * @param allocation - Allocation array to adjust.
 * @param rawShares - Raw fractional shares.
 * @param remainingSlots - Total slots available.
 * @returns Nothing.
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
 *
 * Minimum-offspring guarantees can occasionally oversubscribe the remaining
 * budget. This helper trims from the largest allocations first while still
 * respecting the minimum line preserved for each surviving species.
 *
 * Read it as the final safety rail after the diversity protections have done
 * their work. The helper is not changing the policy goal; it is only forcing
 * the final integer allocation back inside the available slot budget.
 *
 * @param internal - NEAT controller instance.
 * @param allocation - Allocation array to adjust.
 * @param remainingSlots - Total slots available.
 * @param minOffspringDefault - Default minimum offspring.
 * @returns Nothing.
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
 *
 * This helper is the point where species-local survivor selection turns into
 * one actual child. It chooses both parents, performs crossover, and annotates
 * runtime lineage metadata so the resulting genome is ready for later telemetry,
 * lineage, and inbreeding reads.
 *
 * Conceptually, this is where the abstract allocation budget becomes one real
 * experiment. Everything above this helper is still about counts and survivor
 * pools; this helper is where the controller finally spends one unit of that
 * budget on one concrete child genome.
 *
 * @param internal - NEAT controller instance.
 * @param survivors - Survivors pool for selection.
 * @param speciesIndex - Species index.
 * @param crossSpeciesProbability - Cross-species mating probability.
 * @param crossSpeciesGuardLimit - Retry guard for cross-species selection.
 * @param survivalThresholdDefault - Default survivor-window policy used when cross-species selection samples another species.
 * @returns Offspring genome carrying runtime metadata.
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
 *
 * Cross-species mating stays bounded and opportunistic. The helper first asks
 * whether the controller should attempt a cross-species parent at all, then
 * applies a retry guard so the search for another species cannot spiral in edge
 * cases where the registry is sparse or unstable.
 *
 * This keeps cross-species mating opportunistic instead of dominant. The helper
 * first treats inter-species mating as an exception worth asking for, then
 * bounds the search so that a sparse registry cannot trap population assembly in
 * an expensive parent hunt.
 *
 * @param internal - NEAT controller instance.
 * @param survivors - Survivors pool from the current species.
 * @param speciesIndex - Current species index.
 * @param crossSpeciesProbability - Probability to cross species.
 * @param crossSpeciesGuardLimit - Retry guard for cross-species selection.
 * @param survivalThresholdDefault - Default survivor-window policy used when sampling another species.
 * @returns Chosen parent genome from the current or another species.
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
