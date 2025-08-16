/**
 * Generate a deterministic fallback innovation id for a connection when the
 * connection does not provide an explicit innovation number.
 *
 * This function encodes the (from.index, to.index) pair into a single number
 * by multiplying the `from` index by a large base and adding the `to` index.
 * The large base reduces collisions between different pairs and keeps the id
 * stable and deterministic across runs. It is intended as a fallback only —
 * explicit innovation numbers (when present) should be preferred.
 *
 * Example:
 * const conn = { from: { index: 2 }, to: { index: 5 } };
 * const id = _fallbackInnov.call(neatContext, conn); // 200005
 *
 * Notes:
 * - Not globally guaranteed unique, but deterministic for the same indices.
 * - Useful during compatibility checks when some connections are missing innovation ids.
 *
 * @param this - The NEAT instance / context (kept for symmetry with other helpers).
 * @param connection - Connection object expected to contain `from.index` and `to.index`.
 * @returns A numeric innovation id derived from the (from, to) index pair.
 */
export function _fallbackInnov(this: any, connection: any): number {
  // Read the source and target node indices, defaulting to 0 if missing.
  const fromIndex = connection.from?.index ?? 0;
  const toIndex = connection.to?.index ?? 0;

  // Encode the pair deterministically using a large multiplier to reduce collisions.
  return fromIndex * 100000 + toIndex;
}
/**
 * Compute the NEAT compatibility distance between two genomes (networks).
 *
 * The compatibility distance is used for speciation in NEAT. It combines the
 * number of excess and disjoint genes with the average weight difference of
 * matching genes. A generation-scoped cache is used to avoid recomputing the
 * same pair distances repeatedly within a generation.
 *
 * Formula:
 * distance = (c1 * excess + c2 * disjoint) / N + c3 * avgWeightDiff
 * where N = max(number of genes in genomeA, number of genes in genomeB)
 * and c1,c2,c3 are coefficients provided in `this.options`.
 *
 * Example:
 * const d = _compatibilityDistance.call(neatInstance, genomeA, genomeB);
 * if (d < neatInstance.options.compatibilityThreshold) { // same species }
 *
 * @param this - The NEAT instance / context which holds generation, options, and caches.
 * @param genomeA - First genome (network) to compare. Expected to expose `_id` and `connections`.
 * @param genomeB - Second genome (network) to compare. Expected to expose `_id` and `connections`.
 * @returns A numeric compatibility distance; lower means more similar.
 */
export function _compatibilityDistance(
  this: any,
  genomeA: any,
  genomeB: any
): number {
  // Ensure a generation-scoped cache exists and reset it at generation boundaries.
  if (!this._compatCacheGen || this._compatCacheGen !== this.generation) {
    this._compatCacheGen = this.generation;
    this._compatDistCache = new Map<string, number>();
  }

  /**
   * Short description: Stable cache key for the genome pair in the form "minId|maxId".
   */
  const key =
    (genomeA as any)._id < (genomeB as any)._id
      ? `${(genomeA as any)._id}|${(genomeB as any)._id}`
      : `${(genomeB as any)._id}|${(genomeA as any)._id}`;

  /** Short description: Map storing cached distances for genome pairs this generation. */
  const cacheMap: Map<string, number> = this._compatDistCache;

  // If we've already computed this pair this generation, return it immediately.
  if (cacheMap.has(key)) return cacheMap.get(key)!;

  /**
   * Short description: Retrieve or build a sorted innovation list for a genome.
   * Returns an array of [innovationNumber, weight] sorted by innovationNumber.
   */
  const getCache = (network: any) => {
    if (!network._compatCache) {
      // Build a list of pairs [innovation, weight] using connection.innovation
      // if present, otherwise falling back to a deterministic id.
      const list: [number, number][] = network.connections.map((conn: any) => [
        conn.innovation ?? this._fallbackInnov(conn),
        conn.weight,
      ]);

      // Sort by innovation id so we can do a linear merge to compare genomes.
      list.sort((x, y) => x[0] - y[0]);
      network._compatCache = list;
    }
    return network._compatCache as [number, number][];
  };

  // Sorted innovation lists for both genomes.
  const aList = getCache(genomeA);
  const bList = getCache(genomeB);

  // Indices used to iterate the sorted lists.
  let indexA = 0,
    indexB = 0;

  // Counters for types of gene comparisons.
  let matchingCount = 0,
    disjoint = 0,
    excess = 0;

  // Accumulator for absolute weight differences of matching genes.
  let weightDifferenceSum = 0;

  // Highest innovation id present in each list (0 if empty).
  const maxInnovA = aList.length ? aList[aList.length - 1][0] : 0;
  const maxInnovB = bList.length ? bList[bList.length - 1][0] : 0;

  // Step through both sorted innovation lists once, counting matches/disjoint/excess.
  while (indexA < aList.length && indexB < bList.length) {
    const [innovA, weightA] = aList[indexA];
    const [innovB, weightB] = bList[indexB];

    if (innovA === innovB) {
      // Matching innovation ids: accumulate weight difference.
      matchingCount++;
      weightDifferenceSum += Math.abs(weightA - weightB);
      indexA++;
      indexB++;
    } else if (innovA < innovB) {
      // Genome A has a gene with a lower innovation id.
      if (innovA > maxInnovB) excess++;
      else disjoint++;
      indexA++;
    } else {
      // Genome B has a gene with a lower innovation id.
      if (innovB > maxInnovA) excess++;
      else disjoint++;
      indexB++;
    }
  }

  // Any remaining genes after one list is exhausted are all excess genes.
  if (indexA < aList.length) excess += aList.length - indexA;
  if (indexB < bList.length) excess += bList.length - indexB;

  // Normalization factor: use the larger genome size but at least 1 to avoid div0.
  const N = Math.max(1, Math.max(aList.length, bList.length));

  // Average weight difference across matching genes.
  const avgWeightDiff = matchingCount ? weightDifferenceSum / matchingCount : 0;

  /** Short description: Local alias for NEAT options (coefficients for the formula). */
  const opts = this.options;

  /** Short description: Final compatibility distance computed from components. */
  const dist =
    (opts.excessCoeff! * excess) / N +
    (opts.disjointCoeff! * disjoint) / N +
    opts.weightDiffCoeff! * avgWeightDiff;

  // Cache the result for this generation and return.
  cacheMap.set(key, dist);
  return dist;
}
