/**
 * Apply complexity budget scheduling to the evolving population.
 *
 * This routine updates `this.options.maxNodes` (and optionally
 * `this.options.maxConns`) according to a configured complexity budget
 * strategy. Two modes are supported:
 *
 * - `adaptive`: reacts to recent population improvement (or stagnation)
 *   by increasing or decreasing the current complexity cap using
 *   heuristics such as slope (linear trend) of recent best scores,
 *   novelty, and configured increase/stagnation factors.
 * - `linear` (default behaviour when not `adaptive`): linearly ramps
 *   the budget from `maxNodesStart` to `maxNodesEnd` over a horizon.
 *
 * Internal state used/maintained on the `this` object:
 * - `_cbHistory`: rolling window of best scores used to compute trends.
 * - `_cbMaxNodes`: current complexity budget for nodes.
 * - `_cbMaxConns`: current complexity budget for connections (optional).
 *
 * The method is intended to be called on the NEAT engine instance with
 * `this` bound appropriately (i.e. a NeatapticTS `Neat`-like object).
 *
 * @this {{
 *   options: any,
 *   population: Array<{score?: number}>,
 *   input: number,
 *   output: number,
 *   generation: number,
 *   _noveltyArchive?: any[]
 * }} NeatEngine
 *
 * @returns {void} Updates `this.options.maxNodes` and possibly
 * `this.options.maxConns` in-place; no value is returned.
 *
 * @example
 * // inside a training loop where `engine` is your Neat instance:
 * engine.applyComplexityBudget();
 * // engine.options.maxNodes now holds the adjusted complexity cap
 *
 * @remarks
 * This method intentionally uses lightweight linear-regression slope
 * estimation to detect improvement trends. It clamps growth/decay and
 * respects explicit `minNodes`/`maxNodesEnd` if provided. When used in
 * an educational setting, this helps learners observe how stronger
 * selection pressure or novelty can influence allowed network size.
 */
export function applyComplexityBudget(this: any) {
  if (!this.options.complexityBudget?.enabled) return;
  /**
   * Complexity budget configuration object taken from `this.options`.
   * Fields (documented here for doc generators):
   * - mode: 'adaptive'|'linear' — selects scheduling strategy.
   * - improvementWindow: history length used to estimate improvement trends.
   * - increaseFactor / stagnationFactor: multiplicative nudges for growth/shrink.
   * - maxNodesStart / maxNodesEnd / minNodes: explicit clamps for node budget.
   * - maxConnsStart / maxConnsEnd: optional connection-budget clamps.
   * - horizon: generations over which a linear schedule ramps (when mode='linear').
   */
  const complexityBudget = this.options.complexityBudget;
  if (complexityBudget.mode === 'adaptive') {
    if (!this._cbHistory) this._cbHistory = [];
    // method step: record current best score to history for trend analysis
    this._cbHistory.push(this.population[0]?.score || 0);
    /**
     * windowSize: number — number of recent generations to retain for trend
     * estimation. A rolling window is used to smooth noisy fitness
     * signals; larger values reduce variance but react slower.
     */
    /**
     * Number of recent generations to retain for trend estimation.
     */
    const windowSize = complexityBudget.improvementWindow ?? 10;
    if (this._cbHistory.length > windowSize) this._cbHistory.shift();
    /**
     * history: retained numeric history of the best score each recorded
     * generation. Used for computing improvement and slope estimates.
     */
    /**
     * Rolling history of best scores used for improvement and slope estimates.
     */
    const history: number[] = this._cbHistory;
    // method step: compute simple improvement over the retained window
    const improvement =
      history.length > 1 ? history[history.length - 1] - history[0] : 0;
    let slope = 0;
    if (history.length > 2) {
      // method step: estimate linear trend (slope) of the score series
      // using an ordinary least-squares formula. The slope describes the
      // average per-generation change in best-score across the window.
      /**
       * n: number — number of samples in the retained history window.
       * Used by the small OLS slope estimator.
       */
      /** Number of samples in the retained history window. */
      const count = history.length;
      let sumIndices = 0,
        sumScores = 0,
        sumIndexScore = 0,
        sumIndexSquared = 0;
      for (let idx = 0; idx < count; idx++) {
        sumIndices += idx;
        sumScores += history[idx];
        sumIndexScore += idx * history[idx];
        sumIndexSquared += idx * idx;
      }
      // denom could be zero in degenerate cases; default to 1 to avoid NaN
      const denom = count * sumIndexSquared - sumIndices * sumIndices || 1;
      slope = (count * sumIndexScore - sumIndices * sumScores) / denom;
    }
    /**
     * _cbMaxNodes: mutable state on the engine that holds the current
     * node budget. Initialized from config or minimal topology size.
     */
    if (this._cbMaxNodes === undefined)
      this._cbMaxNodes =
        complexityBudget.maxNodesStart ?? this.input + this.output + 2;
    /** Base multiplicative factor used to increase the node budget. */
    const baseInc = complexityBudget.increaseFactor ?? 1.1;
    /** Base multiplicative factor used to shrink the node budget on stagnation. */
    const baseStag = complexityBudget.stagnationFactor ?? 0.95;
    /**
     * slopeMag: normalized slope magnitude clamped to [-2,2]. Used to
     * scale how much the baseInc/baseStag should be nudged by recent
     * improvement trends.
     */
    /** Normalized slope magnitude used to scale growth/shrink nudges. */
    const slopeMag = Math.min(
      2,
      Math.max(-2, slope / (Math.abs(history[0]) + 1e-9))
    );
    // method step: compute final increase and stagnation multipliers
    /**
     * incF: final multiplicative increase factor to apply when scores
     * improve. stagF: final multiplicative decay factor to apply on
     * stagnation. Both combine base factors with slope-derived tweaks.
     */
    /** Final increase multiplier after mixing baseInc and trend signals. */
    const incF = baseInc + 0.05 * Math.max(0, slopeMag);
    /** Final stagnation multiplier after mixing baseStag and trend signals. */
    const stagF = baseStag - 0.03 * Math.max(0, -slopeMag);
    // Constant description: noveltyFactor reduces growth slightly when the
    // novelty archive is small. This dampens expansion for low-novelty
    // situations where exploration is limited.
    /**
     * noveltyFactor: soft multiplier reducing growth when the novelty
     * archive is small; encourages slower expansion if exploration is
     * limited.
     */
    /** Soft multiplier reducing growth if the novelty archive is small. */
    const noveltyFactor = this._noveltyArchive.length > 5 ? 1 : 0.9;
    // method step: expand or contract the node budget depending on trend
    if (improvement > 0 || slope > 0)
      this._cbMaxNodes = Math.min(
        complexityBudget.maxNodesEnd ?? this._cbMaxNodes * 4,
        Math.floor(this._cbMaxNodes * incF * noveltyFactor)
      );
    else if (history.length === windowSize)
      this._cbMaxNodes = Math.max(
        complexityBudget.minNodes ?? this.input + this.output + 2,
        Math.floor(this._cbMaxNodes * stagF)
      );
    // Final clamp to explicit minNodes if provided (safety to avoid too-small nets)
    if (complexityBudget.minNodes !== undefined)
      this._cbMaxNodes = Math.max(complexityBudget.minNodes, this._cbMaxNodes);
    this.options.maxNodes = this._cbMaxNodes;
    if (complexityBudget.maxConnsStart) {
      if (this._cbMaxConns === undefined)
        this._cbMaxConns = complexityBudget.maxConnsStart;
      // method step: apply same expansion/contraction logic to connection budget
      if (improvement > 0 || slope > 0)
        this._cbMaxConns = Math.min(
          complexityBudget.maxConnsEnd ?? this._cbMaxConns * 4,
          Math.floor(this._cbMaxConns * incF * noveltyFactor)
        );
      else if (history.length === windowSize)
        this._cbMaxConns = Math.max(
          complexityBudget.maxConnsStart,
          Math.floor(this._cbMaxConns * stagF)
        );
      this.options.maxConns = this._cbMaxConns;
    }
  } else {
    // method step: linear schedule from start to end across horizon
    // Default start is minimal topology with input+output+2
    /** Linear-schedule starting node budget. */
    const maxStart =
      complexityBudget.maxNodesStart ?? this.input + this.output + 2;
    /** Linear-schedule ending node budget. */
    const maxEnd = complexityBudget.maxNodesEnd ?? maxStart * 4;
    /** Horizon (in generations) over which the linear ramp completes. */
    const horizon = complexityBudget.horizon ?? 100;
    /** Normalized time fraction used by the linear ramp (0..1). */
    const t = Math.min(1, this.generation / horizon);
    this.options.maxNodes = Math.floor(maxStart + (maxEnd - maxStart) * t);
  }
}
/**
 * Toggle phased complexity mode between 'complexify' and 'simplify'.
 *
 * Phased complexity supports alternating periods where the algorithm
 * is encouraged to grow (complexify) or shrink (simplify) network
 * structures. This can help escape local minima or reduce bloat.
 *
 * The current phase and its start generation are stored on `this` as
 * `_phase` and `_phaseStartGeneration` so the state persists across
 * generations.
 *
 * @this {{ options: any, generation: number }} NeatEngine
 * @returns {void} Mutates `this._phase` and `this._phaseStartGeneration`.
 *
 * @example
 * // Called once per generation to update the phase state
 * engine.applyPhasedComplexity();
 */
export function applyPhasedComplexity(this: any) {
  if (!this.options.phasedComplexity?.enabled) return;
  /**
   * phaseLength: number — how many generations each phase ('complexify' or
   * 'simplify') lasts before toggling. Shorter lengths yield faster
   * alternation; longer lengths let the population settle.
   */
  const len = this.options.phasedComplexity.phaseLength ?? 10;
  if (!this._phase) {
    // method step: initialize phase tracking state on first call
    // Default start is 'complexify' (allow growth first) unless
    // explicitly configured otherwise by the caller.
    this._phase = this.options.phasedComplexity.initialPhase ?? 'complexify';
    this._phaseStartGeneration = this.generation;
  }
  if (this.generation - this._phaseStartGeneration >= len) {
    // method step: toggle phase and reset start generation
    this._phase = this._phase === 'complexify' ? 'simplify' : 'complexify';
    this._phaseStartGeneration = this.generation;
  }
}
/**
 * Apply adaptive minimal criterion (MC) acceptance.
 *
 * This method maintains an MC threshold used to decide whether an
 * individual genome is considered acceptable. It adapts the threshold
 * based on the proportion of the population that meets the current
 * threshold, trying to converge to a target acceptance rate.
 *
 * Behavior summary:
 * - Initializes `_mcThreshold` from configuration if undefined.
 * - Computes the proportion of genomes with score >= threshold.
 * - Adjusts threshold multiplicatively by `adjustRate` to move the
 *   observed proportion towards `targetAcceptance`.
 * - Sets `g.score = 0` for genomes that fall below the final threshold
 *   — effectively rejecting them from selection.
 *
 * @this {{ options: any, population: Array<{score?: number}>, _mcThreshold?: number }} NeatEngine
 * @returns {void}
 *
 * @example
 * // Example config snippet used by the engine
 * // options.minimalCriterionAdaptive = { enabled: true, initialThreshold: 0.1, targetAcceptance: 0.5, adjustRate: 0.1 }
 * engine.applyMinimalCriterionAdaptive();
 *
 * @notes
 * Use MC carefully: setting an overly high initial threshold can cause
 * mass-rejection early in evolution. The multiplicative update keeps
 * changes smooth and conservative.
 */
export function applyMinimalCriterionAdaptive(this: any) {
  if (!this.options.minimalCriterionAdaptive?.enabled) return;
  /** Minimal criterion adaptive configuration attached to options. */
  const mcCfg = this.options.minimalCriterionAdaptive;
  /**
   * initialThreshold: optional number — starting value for the MC
   * acceptance threshold. If not provided, starts at 0 (permissive).
   */
  if (this._mcThreshold === undefined)
    this._mcThreshold = mcCfg.initialThreshold ?? 0;
  // method step: compute current acceptance proportion
  /** Population fitness scores snapshot used to compute acceptance proportion. */
  const scores = this.population.map((g: any) => g.score || 0);
  /** Count of genomes meeting or exceeding the current MC threshold. */
  const accepted = scores.filter((s: number) => s >= this._mcThreshold).length;
  /** Observed acceptance proportion in the current population. */
  const prop = scores.length ? accepted / scores.length : 0;
  /**
   * targetAcceptance: desired fraction of genomes to accept (0..1).
   * adjustRate: multiplicative adjustment step applied when acceptance
   * deviates from the target (e.g. 0.1 for 10% change per adaptation).
   */
  /** Target fraction of the population to accept under MC. */
  const targetAcceptance = mcCfg.targetAcceptance ?? 0.5;
  /** Multiplicative adjustment rate applied to the threshold per adaptation. */
  const adjustRate = mcCfg.adjustRate ?? 0.1;
  // method step: adapt threshold multiplicatively to reach target acceptance
  if (prop > targetAcceptance * 1.05) this._mcThreshold *= 1 + adjustRate;
  else if (prop < targetAcceptance * 0.95) this._mcThreshold *= 1 - adjustRate;
  // method step: apply rejection by zeroing scores below threshold
  for (const g of this.population)
    if ((g.score || 0) < this._mcThreshold) g.score = 0;
}
/**
 * Adaptive adjustments based on ancestor uniqueness telemetry.
 *
 * This helper inspects the most recent telemetry lineage block (if
 * available) for an `ancestorUniq` metric indicating how unique
 * ancestry is across the population. If ancestry uniqueness drifts
 * outside configured thresholds, the method will adjust either the
 * multi-objective dominance epsilon (if `mode === 'epsilon'`) or the
 * lineage pressure strength (if `mode === 'lineagePressure'`).
 *
 * Typical usage: keep population lineage diversity within a healthy
 * band. Low ancestor uniqueness means too many genomes share ancestors
 * (risking premature convergence); high uniqueness might indicate
 * excessive divergence.
 *
 * @this {{ options: any, generation: number, _telemetry?: any[], _lastAncestorUniqAdjustGen?: number }} NeatEngine
 * @returns {void}
 *
 * @example
 * // Adjusts `options.multiObjective.dominanceEpsilon` when configured
 * engine.applyAncestorUniqAdaptive();
 *
 * @remarks
 * This method respects a `cooldown` so adjustments are not made every
 * generation. Values are adjusted multiplicatively for gentle change.
 */
export function applyAncestorUniqAdaptive(this: any) {
  if (!this.options.ancestorUniqAdaptive?.enabled) return;
  /** Ancestor uniqueness adaptive configuration object from options. */
  const ancestorCfg = this.options.ancestorUniqAdaptive;
  /**
   * cooldown: number — number of generations to wait between adjustments
   * to avoid fast oscillations of the adjusted parameter(s).
   */
  /** Cooldown (in generations) between successive ancestor-uniqueness adjustments. */
  const cooldown = ancestorCfg.cooldown ?? 5;
  if (this.generation - this._lastAncestorUniqAdjustGen < cooldown) return;
  // method step: fetch latest lineage telemetry block and extract ancestor uniqueness
  const lineageBlock = this._telemetry[this._telemetry.length - 1]?.lineage;
  const ancUniq = lineageBlock ? lineageBlock.ancestorUniq : undefined;
  if (typeof ancUniq !== 'number') return;
  /**
   * lowThreshold/highThreshold: bounds (0..1) defining the acceptable
   * range for ancestor uniqueness. Falling below lowT signals too much
   * shared ancestry; exceeding highT suggests large divergence.
   * adjust: magnitude of the parameter nudge applied when thresholds
   * are crossed.
   */
  /** Lower bound of acceptable ancestor uniqueness (below => increase diversity pressure). */
  const lowT = ancestorCfg.lowThreshold ?? 0.25;
  /** Upper bound of acceptable ancestor uniqueness (above => reduce diversity pressure). */
  const highT = ancestorCfg.highThreshold ?? 0.55;
  /** Adjustment magnitude used when nudging controlled parameters (epsilon/lineage strength). */
  const adj = ancestorCfg.adjust ?? 0.01;
  if (
    ancestorCfg.mode === 'epsilon' &&
    this.options.multiObjective?.adaptiveEpsilon?.enabled
  ) {
    // method step: gently increase or decrease dominance epsilon to
    // encourage/discourage Pareto dominance sensitivity
    if (ancUniq < lowT) {
      this.options.multiObjective.dominanceEpsilon =
        (this.options.multiObjective.dominanceEpsilon || 0) + adj;
      this._lastAncestorUniqAdjustGen = this.generation;
    } else if (ancUniq > highT) {
      this.options.multiObjective.dominanceEpsilon = Math.max(
        0,
        (this.options.multiObjective.dominanceEpsilon || 0) - adj
      );
      this._lastAncestorUniqAdjustGen = this.generation;
    }
  } else if (ancestorCfg.mode === 'lineagePressure') {
    if (!this.options.lineagePressure)
      this.options.lineagePressure = {
        enabled: true,
        mode: 'spread',
        strength: 0.01,
      } as any;
    const lpRef = this.options.lineagePressure!;
    // method step: adjust lineage pressure strength to push populations
    // toward more spread (if ancUniq low) or less (if ancUniq high)
    if (ancUniq < lowT) {
      lpRef.strength = (lpRef.strength || 0.01) * 1.15;
      lpRef.mode = 'spread';
      this._lastAncestorUniqAdjustGen = this.generation;
    } else if (ancUniq > highT) {
      lpRef.strength = (lpRef.strength || 0.01) * 0.9;
      this._lastAncestorUniqAdjustGen = this.generation;
    }
  }
}
/**
 * Self-adaptive per-genome mutation tuning.
 *
 * This function implements several strategies to adjust each genome's
 * internal mutation rate (`g._mutRate`) and optionally its mutation
 * amount (`g._mutAmount`) over time. Strategies include:
 * - `twoTier`: push top and bottom halves in opposite directions to
 *   create exploration/exploitation balance.
 * - `exploreLow`: preferentially increase mutation for lower-scoring
 *   genomes to promote exploration.
 * - `anneal`: gradually reduce mutation deltas over time.
 *
 * The method reads `this.options.adaptiveMutation` for configuration
 * and mutates genomes in-place.
 *
 * @this {{ options: any, population: Array<any>, generation: number, _getRNG: () => () => number }} NeatEngine
 * @returns {void}
 *
 * @example
 * // configuration example:
 * // options.adaptiveMutation = { enabled: true, initialRate: 0.5, adaptEvery: 1, strategy: 'twoTier', minRate: 0.01, maxRate: 1 }
 * engine.applyAdaptiveMutation();
 *
 * @notes
 * - Each genome must already expose `_mutRate` to be adapted. The
 *   function leaves genomes without `_mutRate` untouched.
 * - Randomness is used to propose changes; seeding the RNG allows for
 *   reproducible experiments.
 */
export function applyAdaptiveMutation(this: any) {
  if (!this.options.adaptiveMutation?.enabled) return;
  const adaptCfg = this.options.adaptiveMutation;
  /**
   * adaptEvery: number — adapt mutation parameters every N generations.
   * If 1 (default) adapt every generation; larger values throttle updates.
   */
  const every = adaptCfg.adaptEvery ?? 1;
  if (!(every <= 1 || this.generation % every === 0)) return;
  const scored = this.population.filter(
    (g: any) => typeof g.score === 'number'
  );
  scored.sort((a: any, b: any) => (a.score || 0) - (b.score || 0));
  // method step: partition scored genomes into top/bottom halves used by strategies
  const mid = Math.floor(scored.length / 2);
  const topHalf = scored.slice(mid);
  const bottomHalf = scored.slice(0, mid);
  /** Base scale for random perturbations applied to each genome's mutation rate. */
  const sigmaBase = (adaptCfg.sigma ?? 0.05) * 1.5;
  /** Minimum allowed per-genome mutation rate (clamp lower bound). */
  const minR = adaptCfg.minRate ?? 0.01;
  /** Maximum allowed per-genome mutation rate (clamp upper bound). */
  const maxR = adaptCfg.maxRate ?? 1;
  /** Strategy used to adapt per-genome mutation rates: 'twoTier'|'exploreLow'|'anneal'. */
  const strategy = adaptCfg.strategy || 'twoTier';
  let anyUp = false,
    anyDown = false;
  for (let index = 0; index < this.population.length; index++) {
    const genome = this.population[index];
    if (genome._mutRate === undefined) continue;
    let rate = genome._mutRate;
    // method step: propose a signed delta from RNG and scale it. Values
    // are in [-1,1] then multiplied by sigmaBase to control magnitude.
    let delta = this._getRNG()() * 2 - 1; // base unit in [-1,1]
    delta *= sigmaBase;
    if (strategy === 'twoTier') {
      if (topHalf.length === 0 || bottomHalf.length === 0)
        delta = index % 2 === 0 ? Math.abs(delta) : -Math.abs(delta);
      else if (topHalf.includes(genome)) delta = -Math.abs(delta);
      else if (bottomHalf.includes(genome)) delta = Math.abs(delta);
    } else if (strategy === 'exploreLow') {
      delta = bottomHalf.includes(genome)
        ? Math.abs(delta * 1.5)
        : -Math.abs(delta * 0.5);
    } else if (strategy === 'anneal') {
      const progress = Math.min(
        1,
        this.generation / (50 + this.population.length)
      );
      delta *= 1 - progress;
    }
    // method step: apply delta and clamp to allowed [minR, maxR]
    rate += delta;
    if (rate < minR) rate = minR;
    if (rate > maxR) rate = maxR;
    if (rate > (this.options.adaptiveMutation!.initialRate ?? 0.5))
      anyUp = true;
    if (rate < (this.options.adaptiveMutation!.initialRate ?? 0.5))
      anyDown = true;
    genome._mutRate = rate;
    if (adaptCfg.adaptAmount) {
      /** Scale used when perturbing per-genome discrete mutation amount. */
      const aSigma = adaptCfg.amountSigma ?? 0.25;
      // method step: propose and apply an amount delta if requested
      let aDelta = (this._getRNG()() * 2 - 1) * aSigma;
      if (strategy === 'twoTier') {
        if (topHalf.length === 0 || bottomHalf.length === 0)
          aDelta = index % 2 === 0 ? Math.abs(aDelta) : -Math.abs(aDelta);
        else
          aDelta = bottomHalf.includes(genome)
            ? Math.abs(aDelta)
            : -Math.abs(aDelta);
      }
      // method step: update discrete mutation amount and clamp
      let amt = genome._mutAmount ?? (this.options.mutationAmount || 1);
      amt += aDelta;
      amt = Math.round(amt);
      /** Minimum allowed mutation-amount (discrete clamp). */
      const minA = adaptCfg.minAmount ?? 1;
      /** Maximum allowed mutation-amount (discrete clamp). */
      const maxA = adaptCfg.maxAmount ?? 10;
      if (amt < minA) amt = minA;
      if (amt > maxA) amt = maxA;
      genome._mutAmount = amt;
    }
  }
  if (strategy === 'twoTier' && !(anyUp && anyDown)) {
    const baseline = this.options.adaptiveMutation!.initialRate ?? 0.5;
    const half = Math.floor(this.population.length / 2);
    for (let i = 0; i < this.population.length; i++) {
      const genome = this.population[i];
      if (genome._mutRate === undefined) continue;
      // method step: fallback balancing to ensure some genomes go up and some down
      if (i < half) genome._mutRate = Math.min(genome._mutRate + sigmaBase, 1);
      else genome._mutRate = Math.max(genome._mutRate - sigmaBase, 0.01);
    }
  }
}
/**
 * Decay operator adaptation statistics (success/attempt counters).
 *
 * Many adaptive operator-selection schemes keep running tallies of how
 * successful each operator has been. This helper applies an exponential
 * moving-average style decay to those counters so older outcomes
 * progressively matter less.
 *
 * The `_operatorStats` map on `this` is expected to contain values of
 * the shape `{ success: number, attempts: number }` keyed by operator
 * id/name.
 *
 * @this {{ options: any, _operatorStats: Map<any, {success:number,attempts:number}> }} NeatEngine
 * @returns {void}
 *
 * @example
 * engine.applyOperatorAdaptation();
 */
export function applyOperatorAdaptation(this: any) {
  if (!this.options.operatorAdaptation?.enabled) return;
  const decay = this.options.operatorAdaptation.decay ?? 0.9;
  // method step: apply exponential decay to operator success/attempt tallies
  for (const [k, stat] of this._operatorStats.entries()) {
    stat.success *= decay;
    stat.attempts *= decay;
    this._operatorStats.set(k, stat);
  }
}
