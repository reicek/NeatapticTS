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
 * This is the narrow mechanics layer beneath the controller-facing selection
 * chapter. The root selection helpers answer high-level questions such as
 * "who is the current champion?" or "which parent should breed next?" This
 * core chapter explains how those answers stay stable.
 *
 * Three responsibilities live here:
 *
 * 1. evaluation guards make sure score-dependent reads do not quietly operate
 *    on an unevaluated population,
 * 2. ordering guards restore descending-score order before strategies that
 *    depend on best-first traversal,
 * 3. strategy helpers implement the distinct parent-selection flows for POWER,
 *    FITNESS_PROPORTIONATE, and TOURNAMENT.
 *
 * Keeping that work in `core/` preserves a clean split. The root selection
 * chapter stays readable and controller-facing, the facade chapter keeps the
 * stable `Neat` wrappers thin, and this layer owns the exact mechanics and
 * fallback rules those higher surfaces rely on.
 *
 * Read the chapter in this order: start with the evaluation and ordering
 * guards, then the strategy dispatcher, then the strategy-specific helpers for
 * sorted bias, shifted-fitness roulette, and tournament sampling.
 *
 * ```mermaid
 * flowchart TD
 *   Caller[Root selection helper or Neat facade]
 *   Guards[Evaluation and ordering guards]
 *   Dispatch[selectParentByStrategy]
 *   Power[POWER\nbest-first bias]
 *   Fitness[FITNESS_PROPORTIONATE\nshifted roulette]
 *   Tournament[TOURNAMENT\nsample and walk bracket]
 *   Parent[Chosen parent]
 *
 *   Caller --> Guards
 *   Guards --> Dispatch
 *   Dispatch --> Power
 *   Dispatch --> Fitness
 *   Dispatch --> Tournament
 *   Power --> Parent
 *   Fitness --> Parent
 *   Tournament --> Parent
 * ```
 */

/**
 * Select a parent genome according to the configured selection strategy.
 *
 * This is the single dispatch point shared by the controller-facing selection
 * helpers. It resolves the active strategy once, builds a compact
 * {@link SelectionContext}, and then hands off to the concrete algorithm.
 *
 * The fallback to the first population entry is deliberate. When callers have
 * configured an unknown strategy name, selection still returns a deterministic
 * candidate instead of failing unexpectedly deep inside crossover or mutation
 * flow.
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
 * Selection treats score production as an upstream responsibility, but the
 * controller still needs a safe guard at the point where score-dependent reads
 * actually happen. This helper preserves that assumption: callers can ask for
 * fittest genomes, averages, or parents without manually remembering whether
 * evaluation already ran this generation.
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
 * Several selection reads assume best-first order but should not pay the cost
 * of sorting when the population is already in the expected shape. This helper
 * preserves that cheap guard by checking only the leading edge before calling
 * the shared in-place sort hook.
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
 * This is the simplest "treat the generation as one pool" fold used by summary
 * reads. Missing scores are interpreted with the shared selection fallback so
 * total-score calculations stay aligned with the rest of the chapter.
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
 * POWER selection assumes best-first ordering and then biases random choice
 * toward the front of that ranking. Lower random samples stay near the leading
 * genomes, while the configured exponent controls how quickly the chance falls
 * away from the champion.
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
 * This path turns the current population into a weighted threshold scan. When
 * some genomes have negative scores, the helper first shifts the whole fitness
 * space upward so every participant still occupies a non-negative span on the
 * roulette wheel.
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
 * Tournament selection samples a temporary bracket, orders it by descending
 * score, then walks from strongest to weakest using the configured win
 * probability. This gives the controller a middle ground between pure
 * best-first bias and fully score-proportional roulette.
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
 * POWER selection is the only built-in strategy that depends on the population
 * already being rank-ordered before index sampling. The narrower guard lives
 * here instead of in the root chapter so the strategy can preserve that rule
 * without forcing unrelated selection paths to sort first.
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
 * Fitness-proportionate selection must work even when a population contains
 * negative scores. This helper records the most-negative value, converts that
 * into a uniform upward shift, and returns the adjusted total that the later
 * threshold scan uses.
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
 * Read this as the second half of roulette selection. Once the threshold has
 * been sampled, the helper walks the population once, expanding a cumulative
 * shifted-fitness window until the threshold lands inside one genome's slice.
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
 * Oversized tournaments usually indicate a configuration mistake, so the
 * default behavior is to fail loudly. Tests and a few tolerant call sites can
 * opt into the host-level suppression flag when "best effort" random fallback
 * is more useful than a hard error.
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
 * Repeats are allowed because this helper models independent random draws from
 * the current population rather than a unique bracket seeding pass. That keeps
 * the implementation small and preserves the controller's existing stochastic
 * behavior.
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
 * After participants are sorted best-first, this helper walks the list from the
 * front and gives each participant a chance to win immediately. The configured
 * probability therefore controls how often the top entrant wins outright versus
 * how often weaker entrants remain reachable later in the walk.
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
 * This is the small shared fallback used by roulette misses and suppressed
 * tournament overflow. Centralizing it here keeps every selection path tied to
 * the same controller RNG stream.
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
