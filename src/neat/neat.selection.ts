import { NeatLike } from './neat.types';

/**
 * Sorts the internal population in place by descending fitness.
 *
 * This method mutates the `population` array on the Neat instance so that
 * the genome with the highest `score` appears at index 0. It treats missing
 * scores as 0.
 *
 * Example:
 * const neat = new Neat(...);
 * neat.sort();
 * console.log(neat.population[0].score); // highest score
 *
 * Notes for documentation generators: this is a small utility used by many
 * selection and evaluation routines; it intentionally sorts in-place for
 * performance and to preserve references to genome objects.
 *
 * @this NeatLike - the Neat instance with `population` to sort
 */
export function sort(this: NeatLike): void {
  // Sort population descending by score (highest score first). Missing
  // scores (undefined/null) are treated as 0 using the nullish coalescing operator.
  (this as any).population.sort(
    (a: any, b: any) => (b.score ?? 0) - (a.score ?? 0)
  );
}

/**
 * Select a parent genome according to the configured selection strategy.
 *
 * Supported strategies (via `options.selection.name`):
 * - 'POWER'              : biased power-law selection (exploits best candidates)
 * - 'FITNESS_PROPORTIONATE': roulette-wheel style selection proportional to fitness
 * - 'TOURNAMENT'         : pick N random competitors and select the best with probability p
 *
 * This function intentionally makes no changes to the population except in
 * the POWER path where a quick sort may be triggered to ensure descending
 * order.
 *
 * Examples:
 * // POWER selection (higher power => more exploitation)
 * neat.options.selection = { name: 'POWER', power: 2 };
 * const parent = neat.getParent();
 *
 * // Tournament selection (size 3, 75% probability to take top of tournament)
 * neat.options.selection = { name: 'TOURNAMENT', size: 3, probability: 0.75 };
 * const parent2 = neat.getParent();
 *
 * @this NeatLike - the Neat instance containing `population`, `options`, and `_getRNG`
 * @returns A genome object chosen as the parent according to the selection strategy
 */
export function getParent(this: NeatLike) {
  /**
   * The configured selection options for this Neat instance. It controls the
   * algorithm used to pick parents.
   * @type {any}
   */
  const selectionOptions = (this as any).options.selection;

  /**
   * The selection strategy identifier (e.g. 'POWER', 'FITNESS_PROPORTIONATE', 'TOURNAMENT').
   * @type {string|undefined}
   */
  const selectionName = selectionOptions?.name;

  /**
   * Bound factory that yields a random number generator function when called.
   * Many parts of the codebase use the pattern `_getRNG()()` to obtain a
   * uniform RNG in [0, 1). We preserve that behaviour via getRngFactory.
   * @type {() => () => number}
   */
  const getRngFactory = (this as any)._getRNG.bind(this);

  /**
   * Local reference to the population array of genomes on this Neat instance.
   * @type {any[]}
   */
  const population = (this as any).population;

  switch (selectionName) {
    case 'POWER':
      // Ensure population sorted descending when necessary. The POWER strategy
      // expects the best genomes to be at the front so we check and sort.
      if (
        population[0]?.score !== undefined &&
        population[1]?.score !== undefined &&
        population[0].score < population[1].score
      ) {
        (this as any).sort();
      }

      /**
       * Compute the selected index using a power-law distribution. `power`
       * > 1 biases selection toward the start of the sorted population.
       * @type {number}
       */
      const selectedIndex = Math.floor(
        Math.pow(getRngFactory()(), selectionOptions.power || 1) *
          population.length
      );

      // Return the genome at the chosen index.
      return population[selectedIndex];

    case 'FITNESS_PROPORTIONATE':
      // --- Compute total fitness and shift negative fitnesses ---
      /**
       * Accumulator for sum of fitness values (before shifting negatives).
       * @type {number}
       */
      let totalFitness = 0;

      /**
       * Track the most negative score to shift all scores into positive space.
       * This avoids problems when fitness values are negative.
       * @type {number}
       */
      let mostNegativeScore = 0;

      // Aggregate total fitness and discover minimal score in one loop.
      population.forEach((individual: any) => {
        mostNegativeScore = Math.min(mostNegativeScore, individual.score ?? 0);
        totalFitness += individual.score ?? 0;
      });

      // Convert the most negative score into a non-negative shift value.
      const minFitnessShift = Math.abs(mostNegativeScore);

      // Add the shift for every member so the totalFitness accounts for shifting.
      totalFitness += minFitnessShift * population.length;

      /**
       * Random threshold used to perform roulette-wheel selection over shifted fitness.
       * @type {number}
       */
      const threshold = getRngFactory()() * totalFitness;

      /**
       * Running cumulative total while iterating to find where `threshold` falls.
       * @type {number}
       */
      let cumulative = 0;

      // Walk the population adding shifted scores until threshold is exceeded.
      for (const individual of population) {
        cumulative += (individual.score ?? 0) + minFitnessShift;
        if (threshold < cumulative) return individual;
      }

      // Fallback in the unlikely event the loop did not return: choose random.
      return population[Math.floor(getRngFactory()() * population.length)];

    case 'TOURNAMENT':
      // Validate tournament size vs population and handle fallback/exception.
      if ((selectionOptions.size || 2) > population.length) {
        // Only throw when not in internal reproduction path (flag set by evolve to suppress)
        if (!(this as any)._suppressTournamentError) {
          throw new Error('Tournament size must be less than population size.');
        }
        // Fallback: degrade to random parent
        return population[Math.floor(getRngFactory()() * population.length)];
      }

      /**
       * Number of competitors to sample for the tournament.
       * @type {number}
       */
      const tournamentSize = selectionOptions.size || 2;

      /**
       * Temporary list of randomly sampled tournament participants.
       * @type {any[]}
       */
      const tournamentParticipants: any[] = [];

      // Sample `tournamentSize` random individuals (with possible repeats).
      for (let i = 0; i < tournamentSize; i++) {
        tournamentParticipants.push(
          population[Math.floor(getRngFactory()() * population.length)]
        );
      }

      // Sort participants descending by fitness so index 0 is the best.
      tournamentParticipants.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

      // Walk through the sorted tournament and pick a winner probabilistically.
      for (let i = 0; i < tournamentParticipants.length; i++) {
        if (
          getRngFactory()() < (selectionOptions.probability ?? 0.5) ||
          i === tournamentParticipants.length - 1
        )
          return tournamentParticipants[i];
      }
      break;

    default:
      // Legacy fallback: return the first population member as a safe default.
      return population[0];
  }
  // Extra safety fallback.
  return population[0];
}

/**
 * Return the fittest genome in the population.
 *
 * This will trigger an `evaluate()` if genomes have not been scored yet, and
 * will ensure the population is sorted so index 0 contains the fittest.
 *
 * Example:
 * const best = neat.getFittest();
 * console.log(best.score);
 *
 * @this NeatLike - the Neat instance containing `population` and `evaluate`.
 * @returns The genome object judged to be the fittest (highest score).
 */
export function getFittest(this: NeatLike) {
  /**
   * Local reference to the population array of genomes.
   * @type {any[]}
   */
  const population = (this as any).population;

  // If the last element doesn't have a score then evaluation hasn't run yet.
  if (population[population.length - 1].score === undefined) {
    (this as any).evaluate();
  }

  // If the population isn't sorted descending by score, sort it.
  if (
    population[1] &&
    (population[0].score ?? 0) < (population[1].score ?? 0)
  ) {
    (this as any).sort();
  }

  // Return the genome at index 0 which should be the fittest.
  return population[0];
}

/**
 * Compute the average (mean) fitness across the population.
 *
 * If genomes have not been evaluated yet this will call `evaluate()` so
 * that scores exist. Missing scores are treated as 0.
 *
 * Example:
 * const avg = neat.getAverage();
 * console.log(`Average fitness: ${avg}`);
 *
 * @this NeatLike - the Neat instance containing `population` and `evaluate`.
 * @returns The mean fitness as a number.
 */
export function getAverage(this: NeatLike) {
  const population = (this as any).population;

  // Ensure all genomes have been evaluated before computing the mean.
  if (population[population.length - 1].score === undefined) {
    (this as any).evaluate();
  }

  // Sum all scores treating undefined as 0 and divide by population size.
  const totalScore = population.reduce(
    (sum: number, genome: any) => sum + (genome.score ?? 0),
    0
  );
  return totalScore / population.length;
}
