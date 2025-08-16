/**
 * Multi-objective helpers (fast non-dominated sorting + crowding distance).
 * Extracted from `neat.ts` to keep the core class slimmer.
 */
import type Network from '../architecture/network';

/**
 * Shape of an objective descriptor used by the Neat instance.
 * - `accessor` extracts a numeric objective from a genome
 * - `direction` optionally indicates whether the objective is maximized or
 *   minimized (defaults to 'max')
 */
type ObjectiveDescriptor = {
  accessor: (genome: Network) => number;
  direction?: 'max' | 'min';
};

/**
 * Perform fast non-dominated sorting and compute crowding distances for a
 * population of networks (genomes). This implements a standard NSGA-II style
 * non-dominated sorting followed by crowding distance assignment.
 *
 * The function annotates genomes with two fields used elsewhere in the codebase:
 * - `_moRank`: integer Pareto front rank (0 = best/frontier)
 * - `_moCrowd`: numeric crowding distance (higher is better; Infinity for
 *   boundary solutions)
 *
 * Example
 * ```ts
 * // inside a Neat class that exposes `_getObjectives()` and `options`
 * const fronts = fastNonDominated.call(neatInstance, population);
 * // fronts[0] is the Pareto-optimal set
 * ```
 *
 * Notes for documentation generation:
 * - Each objective descriptor returned by `_getObjectives()` must have an
 *   `accessor(genome: Network): number` function and may include
 *   `direction: 'max' | 'min'` to indicate optimization direction.
 * - Accessor failures are guarded and will yield a default value of 0.
 *
 * @param this - Neat instance providing `_getObjectives()`, `options` and
 *   `_paretoArchive` fields (function is meant to be invoked using `.call`)
 * @param pop - population array of `Network` genomes to be ranked
 * @returns Array of Pareto fronts; each front is an array of `Network` genomes.
 */
export function fastNonDominated(this: any, pop: Network[]): Network[][] {
  /**
   * const: objective descriptors array
   * Short description: descriptors returned by the Neat instance that define
   * how to extract objective values from genomes and whether each objective is
   * maximized or minimized.
   *
   * Each descriptor must provide:
   * - `accessor(genome: Network): number` — returns numeric score for genome
   * - `direction?: 'max' | 'min'` — optional optimization direction (default 'max')
   */
  const objectiveDescriptors: ObjectiveDescriptor[] = this._getObjectives();

  /**
   * const: objective values matrix
   * Short description: precomputed numeric values of each objective for every
   * genome in the population. This avoids repeated accessor calls during
   * pairwise domination checks.
   *
   * Shape: `[population.length][objectives.length]` where row i contains the
   * objective vector for `pop[i]`.
   */
  const valuesMatrix: number[][] = pop.map((genomeItem: Network) =>
    objectiveDescriptors.map((descriptor: any) => {
      try {
        return descriptor.accessor(genomeItem);
      } catch {
        // If an objective accessor fails, treat the value as neutral (0).
        return 0;
      }
    })
  );

  /**
   * const: dominance predicate
   * Short description: returns true when vector `valuesA` Pareto-dominates
   * vector `valuesB`.
   *
   * Detailed behavior:
   * - For each objective the comparator respects the objective's `direction`.
   * - `a` must be at least as good in all objectives and strictly better in at
   *   least one objective to be considered dominating.
   *
   * @param valuesA - objective value vector for candidate A
   * @param valuesB - objective value vector for candidate B
   * @returns boolean whether A dominates B
   *
   * Example:
   * ```ts
   * vectorDominates([1,2], [1,3]) // false when both objectives are 'max'
   * ```
   */
  const vectorDominates = (valuesA: number[], valuesB: number[]) => {
    let strictlyBetter = false;
    // Compare each objective, honoring the objective's optimization direction.
    for (
      let objectiveIndex = 0;
      objectiveIndex < valuesA.length;
      objectiveIndex++
    ) {
      const direction = objectiveDescriptors[objectiveIndex].direction || 'max';
      if (direction === 'max') {
        // For maximization, higher is better.
        if (valuesA[objectiveIndex] < valuesB[objectiveIndex]) return false;
        if (valuesA[objectiveIndex] > valuesB[objectiveIndex])
          strictlyBetter = true;
      } else {
        // For minimization, lower is better.
        if (valuesA[objectiveIndex] > valuesB[objectiveIndex]) return false;
        if (valuesA[objectiveIndex] < valuesB[objectiveIndex])
          strictlyBetter = true;
      }
    }
    return strictlyBetter;
  };

  /**
   * const: paretoFronts
   * Short description: accumulates discovered Pareto fronts during sorting.
   *
   * Each element is a front (array of `Network`) in ascending rank order.
   */
  const paretoFronts: Network[][] = [];

  /**
   * const: dominationCounts
   * Short description: for each genome index, the count of other genomes that
   * currently dominate it (used to detect front membership when count reaches 0).
   */
  const dominationCounts: number[] = new Array(pop.length).fill(0);

  /**
   * const: dominatesIndicesList
   * Short description: adjacency list where index p maps to a list of indices q
   * such that genome p dominates genome q. This accelerates propagation when
   * removing a front.
   */
  const dominatedIndicesByIndex: number[][] = pop.map(() => []);

  /**
   * const: nonDominatedIndices
   * Short description: temporary buffer containing indices of genomes that are
   * not dominated by any other genome — i.e., members of the first front.
   */
  const firstFrontIndices: number[] = [];

  // Build domination relationships between every pair of genomes.
  // Step: for each pair (p,q) compute who (if any) dominates who, using the
  // precomputed valuesMatrix to avoid repeated accessor calls.
  for (let pIndex = 0; pIndex < pop.length; pIndex++) {
    for (let qIndex = 0; qIndex < pop.length; qIndex++) {
      if (pIndex === qIndex) continue;
      if (vectorDominates(valuesMatrix[pIndex], valuesMatrix[qIndex]))
        dominatedIndicesByIndex[pIndex].push(qIndex);
      else if (vectorDominates(valuesMatrix[qIndex], valuesMatrix[pIndex]))
        dominationCounts[pIndex]++;
    }
    if (dominationCounts[pIndex] === 0) firstFrontIndices.push(pIndex);
  }

  // Assign genomes to Pareto fronts using a breadth-like ranking algorithm.
  let currentFrontIndices = firstFrontIndices;
  let currentFrontRank = 0;
  while (currentFrontIndices.length) {
    const nextFrontIndices: number[] = [];
    for (const pIndex of currentFrontIndices) {
      // Annotate genome with its multi-objective rank for downstream use.
      (pop[pIndex] as any)._moRank = currentFrontRank;
      // For every genome q dominated by p, reduce its domination count.
      for (const qIndex of dominatedIndicesByIndex[pIndex]) {
        dominationCounts[qIndex]--;
        if (dominationCounts[qIndex] === 0) nextFrontIndices.push(qIndex);
      }
    }
    // Add the actual genomes (not indices) as a front.
    paretoFronts.push(currentFrontIndices.map((i) => pop[i]));
    currentFrontIndices = nextFrontIndices;
    currentFrontRank++;
    // Safety: prevent pathological runs in degenerate cases.
    if (currentFrontRank > 50) break;
  }

  // Crowding distance calculation: measures density around solutions in each front.
  for (const front of paretoFronts) {
    if (front.length === 0) continue;
    // Initialize crowding distance for each genome in the front.
    for (const genomeItem of front) (genomeItem as any)._moCrowd = 0;

    // For each objective, sort the front and accumulate normalized distances.
    for (
      let objectiveIndex = 0;
      objectiveIndex < objectiveDescriptors.length;
      objectiveIndex++
    ) {
      // Sort ascending by the objective value so that boundary solutions
      // (min and max) fall at the ends of the sorted array — this is needed
      // to mark boundary genomes with infinite crowding distance.
      const sortedByCurrentObjective = front
        .slice()
        .sort((genomeA, genomeB) => {
          const valA = objectiveDescriptors[objectiveIndex].accessor(genomeA);
          const valB = objectiveDescriptors[objectiveIndex].accessor(genomeB);
          return valA - valB;
        });

      // Boundary solutions get infinite crowding so they are always preferred.
      (sortedByCurrentObjective[0] as any)._moCrowd = Infinity;
      (sortedByCurrentObjective[
        sortedByCurrentObjective.length - 1
      ] as any)._moCrowd = Infinity;

      const minVal = objectiveDescriptors[objectiveIndex].accessor(
        sortedByCurrentObjective[0]
      );
      const maxVal = objectiveDescriptors[objectiveIndex].accessor(
        sortedByCurrentObjective[sortedByCurrentObjective.length - 1]
      );
      // Avoid division by zero when all values are equal.
      const valueRange = maxVal - minVal || 1;

      // For non-boundary genomes, add normalized distance between neighbors.
      for (
        let sortedIndex = 1;
        sortedIndex < sortedByCurrentObjective.length - 1;
        sortedIndex++
      ) {
        const prevVal = objectiveDescriptors[objectiveIndex].accessor(
          sortedByCurrentObjective[sortedIndex - 1]
        );
        const nextVal = objectiveDescriptors[objectiveIndex].accessor(
          sortedByCurrentObjective[sortedIndex + 1]
        );
        (sortedByCurrentObjective[sortedIndex] as any)._moCrowd +=
          (nextVal - prevVal) / valueRange;
      }
    }
  }

  // Optionally archive a compact Pareto history for visualization/debugging.
  // The archive stores the current generation and the IDs of genomes in the
  // top N fronts (here we keep up to 3 fronts). This is deliberately compact
  // (IDs only) to keep the archive small for long runs.
  if (this.options.multiObjective?.enabled) {
    this._paretoArchive.push({
      generation: this.generation,
      fronts: paretoFronts.slice(0, 3).map((front) =>
        // map each front (array of Network) to an array of genome IDs
        front.map((genome) => (genome as any)._id)
      ),
    });
    if (this._paretoArchive.length > 100) this._paretoArchive.shift();
  }

  return paretoFronts;
}
