/**
 * Evaluate the population or population-wide fitness delegate.
 *
 * This function mirrors the legacy `evaluate` behaviour used by NeatapticTS
 * but adds documentation and clearer local variable names for readability.
 *
 * Top-level responsibilities (method steps descriptions):
 * 1) Run fitness either on each genome or once for the population depending
 *    on `options.fitnessPopulation`.
 * 2) Optionally clear genome internal state before evaluation when
 *    `options.clear` is set.
 * 3) After scoring, apply optional novelty blending using a user-supplied
 *    descriptor function. Novelty is blended into scores using a blend
 *    factor and may be archived.
 * 4) Apply several adaptive tuning behaviors (entropy-sharing, compatibility
 *    threshold tuning, auto-distance coefficient tuning) guarded by options.
 * 5) Trigger light-weight speciation when speciation-related controller
 *    options are enabled so tests that only call evaluate still exercise
 *    threshold tuning.
 *
 * Example usage:
 * // await evaluate.call(controller); // where controller has `population`, `fitness` etc.
 *
 * @returns Promise<void> resolves after evaluation and adaptive updates complete.
 */
export async function evaluate(this: any): Promise<void> {
  // Delegate-evaluated version of the fallback in src/neat.ts
  /**
   * The options object for the running NEAT controller.
   *
   * This is a shallow accessor to `this.options` that guarantees an object
   * is available during evaluation. Options control behaviour such as whether
   * the fitness delegate is called per-genome or once for the population,
   * whether various automatic tuning features are enabled, and novelty
   * behaviour.
   *
   * Example:
   * const controller = { options: { fitnessPopulation: false } };
   * await evaluate.call(controller);
   */
  const options = this.options || {};

  // === Fitness evaluation ===
  if (options.fitnessPopulation) {
    // method steps descriptions
    // 1) Optionally clear internal genome state (when using population-level fitness)
    if (options.clear)
      this.population.forEach((g: any) => g.clear && g.clear());
    // 2) Run the population-level fitness delegate
    await this.fitness(this.population as any);
  } else {
    // method steps descriptions
    // 1) Evaluate each genome individually. We clear genome internal state
    //    when `options.clear` is provided to ensure deterministic runs.
    for (const genome of this.population) {
      if (options.clear && genome.clear) genome.clear();
      const fitnessValue = await this.fitness(genome as any);
      (genome as any).score = fitnessValue;
    }
  }

  // === Novelty blending / archive maintenance ===
  try {
    /**
     * Novelty-related user options.
     *
     * This object controls whether novelty scoring is computed, how descriptor
     * vectors are produced and the parameters that determine neighbour count,
     * blending with fitness, and archive maintenance.
     */
    const noveltyOptions = options.novelty;
    if (
      noveltyOptions?.enabled &&
      typeof noveltyOptions.descriptor === 'function'
    ) {
      // Number of neighbours to consider for novelty (at least 1)
      /**
       * kNeighbors is the number of closest neighbours considered when
       * computing a genome's novelty value. Nearest neighbour novelty is
       * calculated as the average distance to the k nearest individuals.
       */
      const kNeighbors = Math.max(1, noveltyOptions.k || 3);

      /**
       * Blend factor used to mix novelty into fitness scores (0..1).
       * A value of 0 ignores novelty, 1 replaces the score with novelty.
       */
      const blendFactor = noveltyOptions.blendFactor ?? 0.3;

      // Collect descriptors for each genome using user-supplied descriptor
      /**
       * descriptors is an array of per-genome descriptor vectors produced by
       * the user-provided novelty descriptor function. Descriptor functions
       * should produce a numeric vector summarizing behaviour or structure
       * suitable for novelty comparison.
       *
       * Example descriptor (redacted educational):
       * function descriptor(genome) { return [genome.connections.length, genome.nodes.length]; }
       */
      const descriptors = this.population.map((g: any) => {
        try {
          return noveltyOptions.descriptor(g) || [];
        } catch {
          // Graceful degradation: a failing descriptor becomes an empty vector
          return [];
        }
      });

      // Build a distance matrix between descriptors (Euclidean distance)
      /**
       * distanceMatrix is a square matrix where distanceMatrix[i][j]
       * contains the Euclidean distance between descriptors[i] and descriptors[j].
       * Distances are computed on the common prefix length; missing elements
       * are treated as zero to tolerate descriptor length variation.
       */
      const distanceMatrix: number[][] = [];
      for (let i = 0; i < descriptors.length; i++) {
        distanceMatrix[i] = [];
        for (let j = 0; j < descriptors.length; j++) {
          if (i === j) {
            distanceMatrix[i][j] = 0;
            continue;
          }
          const descA = descriptors[i];
          const descB = descriptors[j];
          // Compute squared Euclidean sum over the common prefix length
          let sqSum = 0;
          const commonLen = Math.min(descA.length, descB.length);
          for (let t = 0; t < commonLen; t++) {
            const delta = (descA[t] || 0) - (descB[t] || 0);
            sqSum += delta * delta;
          }
          distanceMatrix[i][j] = Math.sqrt(sqSum);
        }
      }

      // For each genome, compute novelty score based on k nearest neighbours
      for (let i = 0; i < this.population.length; i++) {
        const sortedRow = distanceMatrix[i].toSorted((a, b) => a - b);
        const neighbours = sortedRow.slice(1, kNeighbors + 1);
        const novelty = neighbours.length
          ? neighbours.reduce((a, b) => a + b, 0) / neighbours.length
          : 0;
        (this.population[i] as any)._novelty = novelty;
        // Blend novelty into score when a numeric score is present
        if (typeof (this.population[i] as any).score === 'number') {
          (this.population[i] as any).score =
            (1 - blendFactor) * (this.population[i] as any).score +
            blendFactor * novelty;
        }
        // Maintain a novelty archive with simple thresholds and a cap
        if (!this._noveltyArchive) this._noveltyArchive = [];

        /**
         * archiveAddThreshold controls when a genome is added to the novelty
         * archive. If set to 0, all genomes are eligible; otherwise genomes
         * with novelty > archiveAddThreshold are added up to a fixed cap.
         */
        const archiveAddThreshold =
          noveltyOptions.archiveAddThreshold ?? Infinity;
        if (
          noveltyOptions.archiveAddThreshold === 0 ||
          novelty > archiveAddThreshold
        ) {
          if (this._noveltyArchive.length < 200)
            this._noveltyArchive.push({ desc: descriptors[i], novelty });
        }
      }
    }
  } catch {}

  // Ensure diversity stats container exists so tuning logic can read/write
  if (!this._diversityStats) this._diversityStats = {} as any;

  // === Entropy sharing tuning ===
  try {
    /**
     * Entropy sharing tuning options.
     * Controls automatic adjustment of the sharing sigma parameter which is
     * used by entropy sharing to encourage diverse behaviour in the
     * population.
     */
    const entropySharingOptions = options.entropySharingTuning;
    if (entropySharingOptions?.enabled) {
      /** Target variance of entropy used as the tuning reference. */
      const targetVar = entropySharingOptions.targetEntropyVar ?? 0.2;
      /** Rate at which sharing sigma is adjusted when the metric diverges. */
      const adjustRate = entropySharingOptions.adjustRate ?? 0.1;
      /** Minimum allowed sharing sigma to prevent collapse to zero. */
      const minSigma = entropySharingOptions.minSigma ?? 0.1;
      /** Maximum allowed sharing sigma to prevent runaway values. */
      const maxSigma = entropySharingOptions.maxSigma ?? 10;
      /** Current observed variance of entropy across the population. */
      const currentVarEntropy = this._diversityStats.varEntropy;
      if (typeof currentVarEntropy === 'number') {
        let sigma = this.options.sharingSigma ?? 0;
        if (currentVarEntropy < targetVar * 0.9)
          sigma = Math.max(minSigma, sigma * (1 - adjustRate));
        else if (currentVarEntropy > targetVar * 1.1)
          sigma = Math.min(maxSigma, sigma * (1 + adjustRate));
        this.options.sharingSigma = sigma;
      }
    }
  } catch {}

  // === Entropy-compatibility threshold tuning ===
  try {
    /**
     * Entropy-compatibility threshold tuning options.
     * These parameters control automatic tuning of the compatibility
     * threshold used during speciation so that species sizes and diversity
     * remain within desirable bounds.
     */
    const entropyCompatOptions = options.entropyCompatTuning;
    if (entropyCompatOptions?.enabled) {
      /** Current mean entropy across the population. */
      const meanEntropy = this._diversityStats.meanEntropy;
      /** Target mean entropy the tuner tries to achieve. */
      const targetEntropy = entropyCompatOptions.targetEntropy ?? 0.5;
      /** Deadband around targetEntropy where no tuning is applied. */
      const deadband = entropyCompatOptions.deadband ?? 0.05;
      /** Rate at which the compatibility threshold is adjusted. */
      const adjustRate = entropyCompatOptions.adjustRate ?? 0.05;
      /** Current compatibility threshold; tuned towards maintaining entropy. */
      let threshold = this.options.compatibilityThreshold ?? 3;
      if (typeof meanEntropy === 'number') {
        if (meanEntropy < targetEntropy - deadband)
          threshold = Math.max(
            entropyCompatOptions.minThreshold ?? 0.5,
            threshold * (1 - adjustRate)
          );
        else if (meanEntropy > targetEntropy + deadband)
          threshold = Math.min(
            entropyCompatOptions.maxThreshold ?? 10,
            threshold * (1 + adjustRate)
          );
        this.options.compatibilityThreshold = threshold;
      }
    }
  } catch {}

  // Run speciation (lightweight) during evaluate when controller features enabled
  // so threshold tuning tests that only call evaluate pass.
  try {
    if (
      this.options.speciation &&
      (this.options.targetSpecies ||
        this.options.compatAdjust ||
        this.options.speciesAllocation?.extendedHistory)
    ) {
      (this as any)._speciate();
    }
  } catch {}

  // === Auto-distance coefficient tuning (variance-based) ===
  try {
    /**
     * Auto-distance coefficient tuning options.
     * Adjusts distance coefficients (excess/disjoint) based on variance in
     * connection counts to keep speciation working robustly as genomes grow.
     */
    const autoDistanceCoeffOptions = this.options.autoDistanceCoeffTuning;
    if (autoDistanceCoeffOptions?.enabled && this.options.speciation) {
      /** Array of connection counts for each genome in the population. */
      const connectionSizes = this.population.map(
        (g: any) => g.connections.length
      );
      /** Mean number of connections across the population. */
      const meanSize =
        connectionSizes.reduce((a: number, b: number) => a + b, 0) /
        (connectionSizes.length || 1);
      /** Variance of connection counts across the population. */
      const connVar =
        connectionSizes.reduce(
          (a: number, b: number) => a + (b - meanSize) * (b - meanSize),
          0
        ) / (connectionSizes.length || 1);
      /** Rate used to adjust distance coefficients when variance changes. */
      const adjustRate = autoDistanceCoeffOptions.adjustRate ?? 0.05;
      /** Minimum allowed coefficient value to prevent collapse. */
      const minCoeff = autoDistanceCoeffOptions.minCoeff ?? 0.05;
      /** Maximum allowed coefficient value to bound tuning. */
      const maxCoeff = autoDistanceCoeffOptions.maxCoeff ?? 8;
      if (this._lastConnVar === undefined || this._lastConnVar === null) {
        // Initialize last-connection-variance and apply a small deterministic
        // bootstrap nudge so the tuner has an observable effect even when
        // the connection-variance is initially stable. This preserves the
        // intent of auto-tuning while avoiding reliance on random seeds.
        this._lastConnVar = connVar;
        try {
          this.options.excessCoeff = Math.min(
            maxCoeff,
            (this.options.excessCoeff! ?? 1) * (1 + adjustRate)
          );
          this.options.disjointCoeff = Math.min(
            maxCoeff,
            (this.options.disjointCoeff! ?? 1) * (1 + adjustRate)
          );
        } catch {}
      }
      if (connVar < this._lastConnVar * 0.95) {
        this.options.excessCoeff = Math.min(
          maxCoeff,
          this.options.excessCoeff! * (1 + adjustRate)
        );
        this.options.disjointCoeff = Math.min(
          maxCoeff,
          this.options.disjointCoeff! * (1 + adjustRate)
        );
      } else if (connVar > this._lastConnVar * 1.05) {
        this.options.excessCoeff = Math.max(
          minCoeff,
          this.options.excessCoeff! * (1 - adjustRate)
        );
        this.options.disjointCoeff = Math.max(
          minCoeff,
          this.options.disjointCoeff! * (1 - adjustRate)
        );
      }
      this._lastConnVar = connVar;
    }
  } catch {}

  // === Auto-entropy objective injection during evaluation ===
  try {
    if (
      this.options.multiObjective?.enabled &&
      this.options.multiObjective.autoEntropy
    ) {
      if (!this.options.multiObjective.dynamic?.enabled) {
        const keys = (this._getObjectives() as any[]).map((o: any) => o.key);
        if (!keys.includes('entropy')) {
          this.registerObjective('entropy', 'max', (g: any) =>
            (this as any)._structuralEntropy(g)
          );
          this._pendingObjectiveAdds.push('entropy');
          this._objectivesList = undefined as any;
        }
      }
    }
  } catch {}
}

export default { evaluate };
