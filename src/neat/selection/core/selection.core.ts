import type {
  GenomeWithScore,
  NeatLikeWithSelection,
  SelectionContext,
} from './selection.types';

/** Default power exponent for POWER selection when none is configured. */
export const DEFAULT_POWER = 1;

/** Default tournament size when none is configured. */
export const DEFAULT_TOURNAMENT_SIZE = 2;

/** Default tournament win probability when none is configured. */
export const DEFAULT_TOURNAMENT_PROBABILITY = 0.5;

/** Default score when a genome has no explicit score. */
export const DEFAULT_SCORE = 0;

/** Index of the first element in an array. */
export const FIRST_INDEX = 0;

/** Index of the second element in an array. */
export const SECOND_INDEX = 1;

/** Offset for retrieving the last element via length arithmetic. */
export const LAST_INDEX_OFFSET = 1;

/** Step size for index-based loops. */
export const LOOP_INDEX_INCREMENT = 1;

/** Index used with `at()` to access the last element. */
export const LAST_ELEMENT_INDEX = -1;

/** Initial total fitness accumulator value. */
export const INITIAL_TOTAL_FITNESS = 0;

/** Initial most-negative score sentinel for fitness scans. */
export const INITIAL_MOST_NEGATIVE_SCORE = 0;

/** Initial cumulative fitness value for threshold scans. */
export const INITIAL_CUMULATIVE_FITNESS = 0;

/**
 * Selection mechanics used by the NEAT controller.
 *
 * This chapter collects the shared constants, ordering helpers, and the three
 * built-in parent-selection strategies.
 */

/**
 * Select a parent genome according to the configured selection strategy.
 *
 * @param internal - NEAT host containing population, options, and RNG access.
 * @returns A genome chosen according to the active selection strategy.
 */
export function selectParentByStrategy(
  internal: NeatLikeWithSelection,
): GenomeWithScore {
  const selectionOptions = internal.options.selection;
  const selectionName = selectionOptions?.name;
  const getRngFactory = internal._getRNG.bind(internal);
  const population = internal.population;

  const selectionContext: SelectionContext = {
    internal,
    population,
    selectionOptions,
    getRngFactory,
  };

  switch (selectionName) {
    case 'POWER':
      return selectParentByPower(selectionContext);
    case 'FITNESS_PROPORTIONATE':
      return selectParentByFitnessProportionate(selectionContext);
    case 'TOURNAMENT':
      return selectParentByTournament(selectionContext);
    default:
      return population[FIRST_INDEX];
  }
}

/**
 * Ensure population scores exist by running evaluation if needed.
 *
 * @param internal - NEAT host containing population and evaluation support.
 * @returns Nothing. Evaluation is triggered only when the population is unevaluated.
 */
export function ensurePopulationEvaluated(
  internal: NeatLikeWithSelection,
): void {
  const lastGenome = internal.population.at(LAST_ELEMENT_INDEX);
  const isUnevaluated = lastGenome?.score === undefined;
  if (isUnevaluated) {
    (internal as unknown as { evaluate: () => void }).evaluate();
  }
}

/**
 * Ensure the population is sorted descending by score when out of order.
 *
 * @param internal - NEAT host containing the current population.
 * @returns Nothing. Sorting only runs when the first two scores are out of order.
 */
export function ensurePopulationSortedDescending(
  internal: NeatLikeWithSelection,
): void {
  const population = internal.population;
  const hasSecondGenome = population[SECOND_INDEX] !== undefined;
  const firstScore = population[FIRST_INDEX]?.score ?? DEFAULT_SCORE;
  const secondScore = population[SECOND_INDEX]?.score ?? DEFAULT_SCORE;
  const isOutOfOrder = hasSecondGenome && firstScore < secondScore;

  if (isOutOfOrder) {
    internal.sort();
  }
}

/**
 * Calculate the total fitness across the population.
 *
 * @param population - Genomes in the current population.
 * @returns Sum of all scores with missing scores treated as zero.
 */
export function calculateTotalScore(population: GenomeWithScore[]): number {
  return population.reduce(
    (sum, genome) => sum + (genome.score ?? DEFAULT_SCORE),
    INITIAL_TOTAL_FITNESS,
  );
}

/**
 * Select a parent by power-law distribution on the sorted population.
 *
 * @param selectionContext - Shared selection state.
 * @returns The chosen parent genome.
 */
function selectParentByPower(
  selectionContext: SelectionContext,
): GenomeWithScore {
  // Step 1: Ensure the population is sorted when needed.
  ensurePopulationSortedDescendingForPower(selectionContext);

  // Step 2: Compute a power-law-biased index.
  const power = selectionContext.selectionOptions?.power ?? DEFAULT_POWER;
  const selectedIndex = Math.floor(
    Math.pow(selectionContext.getRngFactory()(), power) *
      selectionContext.population.length,
  );

  return selectionContext.population[selectedIndex];
}

/**
 * Select a parent using roulette-wheel fitness proportionate selection.
 *
 * @param selectionContext - Shared selection state.
 * @returns The chosen parent genome.
 */
function selectParentByFitnessProportionate(
  selectionContext: SelectionContext,
): GenomeWithScore {
  // Step 1: Compute shifted fitness totals.
  const fitnessTotals = calculateFitnessTotals(selectionContext.population);

  // Step 2: Select by shifted cumulative fitness.
  const selectionThreshold =
    selectionContext.getRngFactory()() * fitnessTotals.totalFitness;
  const candidate = pickByShiftedThreshold(
    selectionContext.population,
    selectionThreshold,
    fitnessTotals.minFitnessShift,
  );

  // Step 3: Fall back to a random parent if the threshold scan misses.
  return candidate ?? getRandomPopulationMember(selectionContext);
}

/**
 * Select a parent by tournament selection.
 *
 * @param selectionContext - Shared selection state.
 * @returns The chosen parent genome.
 */
function selectParentByTournament(
  selectionContext: SelectionContext,
): GenomeWithScore {
  // Step 1: Validate the tournament size before sampling.
  const tournamentSize =
    selectionContext.selectionOptions?.size ?? DEFAULT_TOURNAMENT_SIZE;
  if (tournamentSize > selectionContext.population.length) {
    return resolveTournamentOverflow(selectionContext);
  }

  // Step 2: Sample and sort the tournament participants.
  const tournamentParticipants = sampleTournamentParticipants(
    selectionContext,
    tournamentSize,
  );
  const sortedParticipants = tournamentParticipants.toSorted(
    (leftGenome, rightGenome) =>
      (rightGenome.score ?? DEFAULT_SCORE) -
      (leftGenome.score ?? DEFAULT_SCORE),
  );

  // Step 3: Walk probabilistically to choose the winner.
  return pickTournamentWinner(selectionContext, sortedParticipants);
}

/**
 * Ensure the population is sorted descending by score for POWER selection.
 *
 * @param selectionContext - Shared selection state.
 * @returns Nothing. Sorting only runs when the first two entries are out of order.
 */
function ensurePopulationSortedDescendingForPower(
  selectionContext: SelectionContext,
): void {
  const population = selectionContext.population;
  if (
    population[FIRST_INDEX]?.score !== undefined &&
    population[SECOND_INDEX]?.score !== undefined &&
    population[FIRST_INDEX].score < population[SECOND_INDEX].score
  ) {
    selectionContext.internal.sort();
  }
}

/**
 * Compute the total fitness and minimal shift used by roulette selection.
 *
 * @param population - Genomes in the current population.
 * @returns Aggregate fitness totals with the negative-score shift.
 */
function calculateFitnessTotals(population: GenomeWithScore[]): {
  totalFitness: number;
  minFitnessShift: number;
} {
  const initialTotals = {
    totalFitness: INITIAL_TOTAL_FITNESS,
    mostNegativeScore: INITIAL_MOST_NEGATIVE_SCORE,
  };

  const totals = population.reduce((aggregate, individual) => {
    const score = individual.score ?? DEFAULT_SCORE;
    return {
      totalFitness: aggregate.totalFitness + score,
      mostNegativeScore: Math.min(aggregate.mostNegativeScore, score),
    };
  }, initialTotals);

  const minFitnessShift = Math.abs(totals.mostNegativeScore);
  const totalFitness =
    totals.totalFitness + minFitnessShift * population.length;

  return { totalFitness, minFitnessShift };
}

/**
 * Pick the first genome whose shifted cumulative fitness exceeds the threshold.
 *
 * @param population - Genomes in the current population.
 * @param selectionThreshold - Random threshold in shifted fitness space.
 * @param minFitnessShift - Amount added to each score to shift negatives.
 * @returns The chosen genome when a threshold crossing occurs.
 */
function pickByShiftedThreshold(
  population: GenomeWithScore[],
  selectionThreshold: number,
  minFitnessShift: number,
): GenomeWithScore | undefined {
  let cumulativeFitness = INITIAL_CUMULATIVE_FITNESS;

  for (const individual of population) {
    cumulativeFitness += (individual.score ?? DEFAULT_SCORE) + minFitnessShift;
    if (selectionThreshold < cumulativeFitness) {
      return individual;
    }
  }

  return undefined;
}

/**
 * Resolve what happens when tournament size exceeds population size.
 *
 * @param selectionContext - Shared selection state.
 * @returns A fallback parent genome.
 */
function resolveTournamentOverflow(
  selectionContext: SelectionContext,
): GenomeWithScore {
  if (!selectionContext.internal._suppressTournamentError) {
    throw new Error('Tournament size must be less than population size.');
  }

  return getRandomPopulationMember(selectionContext);
}

/**
 * Sample tournament participants with possible repeats.
 *
 * @param selectionContext - Shared selection state.
 * @param tournamentSize - Number of competitors to sample.
 * @returns Sampled participants.
 */
function sampleTournamentParticipants(
  selectionContext: SelectionContext,
  tournamentSize: number,
): GenomeWithScore[] {
  return Array.from({ length: tournamentSize }, () =>
    getRandomPopulationMember(selectionContext),
  );
}

/**
 * Select a winner from sorted tournament participants.
 *
 * @param selectionContext - Shared selection state.
 * @param sortedParticipants - Participants sorted by descending score.
 * @returns The chosen tournament winner.
 */
function pickTournamentWinner(
  selectionContext: SelectionContext,
  sortedParticipants: GenomeWithScore[],
): GenomeWithScore {
  const probability =
    selectionContext.selectionOptions?.probability ??
    DEFAULT_TOURNAMENT_PROBABILITY;

  for (
    let participantIndex = FIRST_INDEX;
    participantIndex < sortedParticipants.length;
    participantIndex += LOOP_INDEX_INCREMENT
  ) {
    const isLastParticipant =
      participantIndex === sortedParticipants.length - LAST_INDEX_OFFSET;
    const isWinner =
      selectionContext.getRngFactory()() < probability || isLastParticipant;
    if (isWinner) {
      return sortedParticipants[participantIndex];
    }
  }

  return sortedParticipants[FIRST_INDEX];
}

/**
 * Select a random population member using the configured RNG.
 *
 * @param selectionContext - Shared selection state.
 * @returns Randomly chosen genome from the current population.
 */
function getRandomPopulationMember(
  selectionContext: SelectionContext,
): GenomeWithScore {
  const randomIndex = Math.floor(
    selectionContext.getRngFactory()() * selectionContext.population.length,
  );
  return selectionContext.population[randomIndex];
}