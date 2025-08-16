/**
 * Assign genomes into species based on compatibility distance and maintain species structures.
 * This function creates new species for unassigned genomes, prunes empty species, updates
 * dynamic compatibility threshold controllers, performs optional auto coefficient tuning, and
 * records per‑species history statistics used by telemetry and adaptive controllers.
 *
 * Implementation notes:
 * - Uses existing representatives; any unassigned genome that doesn't fit an existing species
 *   creates a new species with itself as representative.
 * - Representatives are refreshed each generation (first member heuristic) to reduce drift cost.
 * - Includes optional age penalty for very old species to gently reduce their reproductive share.
 * - PID‑style controller adjusts the global compatibility threshold toward `targetSpecies`.
 * - Auto compatibility coefficient tuning slightly nudges excess/disjoint coefficients to influence
 *   clustering granularity when enabled.
 * - Extended history snapshot captures structural and innovation statistics for richer telemetry.
 */
/**
 * Partition the current population into species using compatibility distance.
 *
 * This function is responsible for assigning genomes into species based on the
 * configured compatibility threshold and maintaining per-species bookkeeping.
 * It also optionally adjusts the global compatibility threshold (PID-like controller),
 * applies an automatic tuning of compatibility coefficients, and records history
 * snapshots used by telemetry and adaptive controllers.
 *
 * Example:
 * const population = ...; // created genomes
 * neat._speciate();
 * // now neat._species contains species with assigned members and representatives
 *
 * Notes for documentation:
 * - This method mutates `this._species`, `this.options.compatibilityThreshold`, and
 *   `this._speciesHistory` as part of each generation's bookkeeping.
 * - It is intentionally conservative: empty species are pruned and representatives are
 *   refreshed to the first member each generation to reduce drift in representative choice.
 *
 * @this any Neataptic-like instance with population, options and bookkeeping maps
 */
export function _speciate(this: any) {
  // Step 1: Preserve previous membership for turnover calculations
  this._prevSpeciesMembers.clear();
  for (const species of this._species) {
    /**
     * prevMemberSet - set of numeric member ids for quick lookup of previous members.
     * Used to compute the turnover rate (fraction of new members since last generation).
     */
    const prevMemberSet = new Set<number>();
    for (const member of species.members)
      prevMemberSet.add((member as any)._id);
    this._prevSpeciesMembers.set(species.id, prevMemberSet);
  }

  // Step 2: Clear current members to allow reassignment from scratch
  this._species.forEach((species: any) => (species.members = []));

  // Step 3: Assignment loop - try to place each genome into an existing species,
  // otherwise create a new species with the genome as representative.
  for (const genome of this.population) {
    /**
     * assignedToExisting - whether the genome was placed into an existing species.
     * This flag guards creation of a new species when false.
     */
    let assignedToExisting = false;
    for (const species of this._species) {
      /**
       * compatDist - numeric compatibility distance between the candidate genome
       * and the species representative. Smaller values indicate greater similarity.
       */
      const compatDist = this._compatibilityDistance(
        genome,
        species.representative
      );
      // method step: if distance below threshold, assign to this species
      if (compatDist < (this.options.compatibilityThreshold || 3)) {
        species.members.push(genome);
        assignedToExisting = true;
        break;
      }
    }
    if (!assignedToExisting) {
      /**
       * speciesId - unique id assigned to a newly created species.
       */
      const speciesId = this._nextSpeciesId++;
      this._species.push({
        id: speciesId,
        members: [genome],
        representative: genome,
        lastImproved: this.generation,
        bestScore: genome.score || -Infinity,
      });
      this._speciesCreated.set(speciesId, this.generation);
    }
  }

  // Step 4: Remove any empty species (defensive - usually not needed)
  this._species = this._species.filter(
    (species: any) => species.members.length > 0
  );

  // Step 5: Refresh representatives (choose the first member as lightweight heuristic)
  this._species.forEach((species: any) => {
    // method step: refresh representative to the first member of the species
    species.representative = species.members[0];
  });

  // Step 6: Soft age penalty - gradually reduce fitness for very old species to
  // encourage turnover and prevent lock-in of stale lineages.
  /**
   * ageProtection - configuration controlling grace period and penalty factor.
   * Applied to species older than (grace * 10) generations (heuristic).
   */
  const ageProtection = this.options.speciesAgeProtection || {
    grace: 3,
    oldPenalty: 0.5,
  };
  for (const species of this._species) {
    const createdGen = this._speciesCreated.get(species.id) ?? this.generation;
    const speciesAge = this.generation - createdGen;
    // method step: apply penalty only when age exceeds a threshold (grace * 10)
    if (speciesAge >= (ageProtection.grace ?? 3) * 10) {
      /** penalty - multiplicative fitness penalty applied to members of very old species */
      const penalty = ageProtection.oldPenalty ?? 0.5;
      if (penalty < 1)
        species.members.forEach((member: any) => {
          if (typeof member.score === 'number') member.score *= penalty;
        });
    }
  }

  // Step 7: Dynamic compatibility threshold controller (PID-like) to steer
  // the number of species toward `targetSpecies` when speciation controller enabled.
  if (this.options.speciation && (this.options.targetSpecies || 0) > 0) {
    /**
     * targetSpeciesCount - the desired number of species set in options.
     */
    const targetSpeciesCount = this.options.targetSpecies!;
    /** observedSpeciesCount - the current number of species observed */
    const observedSpeciesCount = this._species.length;
    /** adjustConfig - PID-like controller configuration from options.compatAdjust */
    const adjustConfig = this.options.compatAdjust!;
    /** smoothingWindow - window size used to compute exponential moving average */
    const smoothingWindow = Math.max(1, adjustConfig.smoothingWindow || 1);
    /** alpha - smoothing coefficient used by the exponential moving average */
    const alpha = 2 / (smoothingWindow + 1);
    this._compatSpeciesEMA =
      this._compatSpeciesEMA === undefined
        ? observedSpeciesCount
        : this._compatSpeciesEMA +
          alpha * (observedSpeciesCount - this._compatSpeciesEMA);
    /** smoothedSpecies - EMA-smoothed observed species count */
    const smoothedSpecies = this._compatSpeciesEMA;
    // error: positive => we want more species => decrease threshold (make clustering harder)
    /** speciesError - difference between desired and smoothed observed species count */
    const speciesError = targetSpeciesCount - smoothedSpecies;
    this._compatIntegral =
      this._compatIntegral * (adjustConfig.decay || 0.95) + speciesError;
    /** delta - PID-like correction term computed from kp/ki and the integrated error */
    const delta =
      (adjustConfig.kp || 0) * speciesError +
      (adjustConfig.ki || 0) * this._compatIntegral;
    /** newThreshold - tentative updated compatibility threshold before clipping */
    let newThreshold = (this.options.compatibilityThreshold || 3) - delta;
    /** minThreshold - lower bound for adjusted compatibility threshold */
    const minThreshold = adjustConfig.minThreshold || 0.5;
    /** maxThreshold - upper bound for adjusted compatibility threshold */
    const maxThreshold = adjustConfig.maxThreshold || 10;
    if (newThreshold < minThreshold) {
      newThreshold = minThreshold;
      this._compatIntegral = 0;
    }
    if (newThreshold > maxThreshold) {
      newThreshold = maxThreshold;
      this._compatIntegral = 0;
    }
    this.options.compatibilityThreshold = newThreshold;
  }

  // Step 8: Auto compatibility coefficient tuning - gently nudge excess/disjoint
  // coefficients to influence clustering granularity when enabled.
  if (this.options.autoCompatTuning?.enabled) {
    /**
     * autoTarget - desired species target for auto tuning, falls back to sqrt(pop).
     * Helps the controller infer a reasonable clustering target when none is provided.
     */
    const autoTarget =
      this.options.autoCompatTuning.target ??
      this.options.targetSpecies ??
      Math.max(2, Math.round(Math.sqrt(this.population.length)));
    /** observedForTuning - number of species observed for tuning calculations */
    const observedForTuning = this._species.length || 1;
    /** tuningError - positive means we want more species -> reduce coefficients */
    const tuningError = autoTarget - observedForTuning;
    /** adjustRate - step rate used to scale coefficient changes */
    const adjustRate = this.options.autoCompatTuning.adjustRate ?? 0.01;
    /** minCoeff - lower bound for tuned coefficients */
    const minCoeff = this.options.autoCompatTuning.minCoeff ?? 0.1;
    /** maxCoeff - upper bound for tuned coefficients */
    const maxCoeff = this.options.autoCompatTuning.maxCoeff ?? 5.0;
    /** factor - multiplicative factor derived from adjustRate and tuning error sign */
    const factor = 1 - adjustRate * Math.sign(tuningError);
    let effectiveFactor = factor;
    if (tuningError === 0) {
      // mild jitter to avoid stagnation when already at target (helps certain tests)
      effectiveFactor = 1 + (this._getRNG()() - 0.5) * adjustRate * 0.5;
    }
    this.options.excessCoeff = Math.min(
      maxCoeff,
      Math.max(minCoeff, this.options.excessCoeff! * effectiveFactor)
    );
    this.options.disjointCoeff = Math.min(
      maxCoeff,
      Math.max(minCoeff, this.options.disjointCoeff! * effectiveFactor)
    );
  }

  // Step 9: Extended history snapshot (rich metrics) or minimal snapshot for telemetry.
  if (this.options.speciesAllocation?.extendedHistory) {
    const stats = this._species.map((species: any) => {
      // Build per-member structural summaries used by aggregated stats below
      /** sizes - per-member compact structural summary used for aggregation */
      const sizes = species.members.map((member: any) => ({
        nodes: member.nodes.length,
        conns: member.connections.length,
        score: member.score || 0,
        nov: (member as any)._novelty || 0,
        ent: this._structuralEntropy(member),
      }));
      /** avg - helper to compute arithmetic mean of numeric arrays */
      const avg = (arr: number[]) =>
        arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
      // Pairwise compatibility sampling (bounded to first 10 members for cost control)
      /** compatSum - cumulative sum of sampled pairwise compatibility distances */
      let compatSum = 0;
      /** compatCount - number of pairwise comparisons included in compatSum */
      let compatCount = 0;
      for (let i = 0; i < species.members.length && i < 10; i++)
        for (let j = i + 1; j < species.members.length && j < 10; j++) {
          compatSum += this._compatibilityDistance(
            species.members[i],
            species.members[j]
          );
          compatCount++;
        }
      /** meanCompat - average pairwise compatibility sampled above */
      const meanCompat = compatCount ? compatSum / compatCount : 0;
      /** last - previously recorded summary stats for this species (if any) */
      const last = this._speciesLastStats.get(species.id);
      /** meanNodes - average number of nodes across sampled members */
      const meanNodes = avg(sizes.map((s: any) => s.nodes));
      /** meanConns - average number of connections across sampled members */
      const meanConns = avg(sizes.map((s: any) => s.conns));
      /** deltaMeanNodes - change in mean node count compared to last snapshot */
      const deltaMeanNodes = last ? meanNodes - last.meanNodes : 0;
      /** deltaMeanConns - change in mean connection count compared to last snapshot */
      const deltaMeanConns = last ? meanConns - last.meanConns : 0;
      /** deltaBestScore - improvement of best score compared to last snapshot */
      const deltaBestScore = last ? species.bestScore - last.best : 0;
      /** createdGen - generation when the species was first created (fallback current gen) */
      const createdGen =
        this._speciesCreated.get(species.id) ?? this.generation;
      /** speciesAge - number of generations since species creation */
      const speciesAge = this.generation - createdGen;
      // Turnover rate: fraction of members that are new relative to previous generation
      /** turnoverRate - fraction of members that are new relative to previous generation */
      let turnoverRate = 0;
      /** prevSet - cached Set of previous member ids for this species */
      const prevSet = this._prevSpeciesMembers.get(species.id);
      if (prevSet && species.members.length) {
        /** newCount - number of members not present in prevSet */
        let newCount = 0;
        for (const member of species.members)
          if (!prevSet.has((member as any)._id)) newCount++;
        turnoverRate = newCount / species.members.length;
      }
      // Variance helper
      /** varCalc - helper to compute variance of numeric arrays */
      const varCalc = (arr: number[]) => {
        if (!arr.length) return 0;
        const mean = avg(arr);
        return avg(arr.map((v) => (v - mean) * (v - mean)));
      };
      /** varNodes - variance of node counts across sampled members */
      const varNodes = varCalc(sizes.map((s: any) => s.nodes));
      /** varConns - variance of connection counts across sampled members */
      const varConns = varCalc(sizes.map((s: any) => s.conns));
      // Innovation statistics across connections in the species
      /** innovSum - cumulative innovation ids sum (for mean) */
      let innovSum = 0;
      /** innovCount - number of connection innovations observed */
      let innovCount = 0;
      /** maxInnov - maximum innovation id observed */
      let maxInnov = -Infinity;
      /** minInnov - minimum innovation id observed */
      let minInnov = Infinity;
      /** enabled - number of enabled connections */
      let enabled = 0;
      /** disabled - number of disabled connections */
      let disabled = 0;
      for (const member of species.members)
        for (const conn of member.connections) {
          const innov = (conn as any).innovation ?? this._fallbackInnov(conn);
          innovSum += innov;
          innovCount++;
          if (innov > maxInnov) maxInnov = innov;
          if (innov < minInnov) minInnov = innov;
          if ((conn as any).enabled === false) disabled++;
          else enabled++;
        }
      /** meanInnovation - mean innovation id across sampled connections */
      const meanInnovation = innovCount ? innovSum / innovCount : 0;
      /** innovationRange - span between max and min innovation ids */
      const innovationRange =
        isFinite(maxInnov) && isFinite(minInnov) && maxInnov > minInnov
          ? maxInnov - minInnov
          : 0;
      /** enabledRatio - fraction of connections that are enabled */
      const enabledRatio =
        enabled + disabled > 0 ? enabled / (enabled + disabled) : 0;
      return {
        id: species.id,
        size: species.members.length,
        best: species.bestScore,
        lastImproved: species.lastImproved,
        age: speciesAge,
        meanNodes,
        meanConns,
        meanScore: avg(sizes.map((s: any) => s.score)),
        meanNovelty: avg(sizes.map((s: any) => s.nov)),
        meanCompat,
        meanEntropy: avg(sizes.map((s: any) => s.ent)),
        varNodes,
        varConns,
        deltaMeanNodes,
        deltaMeanConns,
        deltaBestScore,
        turnoverRate,
        meanInnovation,
        innovationRange,
        enabledRatio,
      };
    });
    for (const st of stats)
      this._speciesLastStats.set(st.id, {
        meanNodes: st.meanNodes,
        meanConns: st.meanConns,
        best: st.best,
      });
    this._speciesHistory.push({ generation: this.generation, stats });
  } else {
    // Minimal snapshot: only store the essentials to reduce memory
    this._speciesHistory.push({
      generation: this.generation,
      stats: this._species.map((species: any) => ({
        id: species.id,
        size: species.members.length,
        best: species.bestScore,
        lastImproved: species.lastImproved,
      })),
    });
  }
  // Step 10: Trim history length to cap memory usage (simple FIFO)
  if (this._speciesHistory.length > 200) this._speciesHistory.shift();
}
/**
 * Apply fitness sharing within each species.
 *
 * Fitness sharing reduces the effective fitness of genomes that are clustered
 * tightly together (close compatibility distance), promoting diversity by
 * penalizing dense species. Two modes are supported:
 *  - Kernel sharing with bandwidth `sharingSigma` (quadratic kernel)
 *  - Equal sharing based on species size when `sharingSigma` is 0
 *
 * Example:
 * neat.options.sharingSigma = 3;
 * neat._applyFitnessSharing();
 *
 * @this any Neataptic-like instance with _species and options
 */
export function _applyFitnessSharing(this: any) {
  /** const sharingSigma - kernel bandwidth controlling neighbor influence */
  const sharingSigma = this.options.sharingSigma || 0;
  if (sharingSigma > 0) {
    // method step: apply kernel-based sharing inside each species
    this._species.forEach((species: any) => {
      const members = species.members;
      for (let i = 0; i < members.length; i++) {
        const memberI = members[i];
        if (typeof memberI.score !== 'number') continue;
        /** shareSum - accumulates kernel values from neighbors used to divide fitness */
        let shareSum = 0;
        for (let j = 0; j < members.length; j++) {
          const memberJ = members[j];
          /** dist - compatibility distance between two members used by the kernel */
          const dist =
            i === j ? 0 : this._compatibilityDistance(memberI, memberJ);
          if (dist < sharingSigma) {
            /** ratio - normalized distance (0..1) relative to sharingSigma bandwidth */
            const ratio = dist / sharingSigma;
            // quadratic kernel: stronger penalty for closer neighbors
            shareSum += 1 - ratio * ratio;
          }
        }
        if (shareSum <= 0) shareSum = 1; // safety to avoid division by zero
        memberI.score = memberI.score / shareSum;
      }
    });
  } else {
    // method step: equal sharing across species members (simple average)
    this._species.forEach((species: any) => {
      /** size - current number of members in the species (used for equal sharing) */
      const size = species.members.length;
      species.members.forEach((member: any) => {
        if (typeof member.score === 'number')
          member.score = member.score / size;
      });
    });
  }
}
/**
 * Sort members of a species in descending order by score.
 *
 * Simple utility used by stagnation checks and selection routines to ensure
 * the top-performing genomes are at index 0.
 *
 * @param sp species-like object with a `members` array and member `.score`
 */
export function _sortSpeciesMembers(this: any, sp: any) {
  // method step: sort in place from highest to lowest score
  sp.members.sort((a: any, b: any) => (b.score || 0) - (a.score || 0));
}
/**
 * Update species stagnation statistics and prune species that have not
 * improved within the configured stagnation window.
 *
 * This updates each species' `bestScore` and `lastImproved` fields and then
 * removes species whose age since last improvement exceeds `stagnationGenerations`.
 *
 * @this any Neataptic-like instance with _species and options
 */
export function _updateSpeciesStagnation(this: any) {
  /** stagnationWindow - number of generations allowed without improvement */
  const stagnationWindow = this.options.stagnationGenerations || 15;
  // method step: refresh member ordering and update per-species bests
  this._species.forEach((species: any) => {
    this._sortSpeciesMembers(species);
    /** top - highest scoring member after sorting (index 0) */
    const top = species.members[0];
    if ((top.score || -Infinity) > species.bestScore) {
      species.bestScore = top.score || -Infinity;
      species.lastImproved = this.generation;
    }
  });
  // method step: keep only species that have improved recently
  /** survivors - species that remain because they have improved within the window */
  const survivors = this._species.filter(
    (species: any) => this.generation - species.lastImproved <= stagnationWindow
  );
  if (survivors.length) this._species = survivors;
}
