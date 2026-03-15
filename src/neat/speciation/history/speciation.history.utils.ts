import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciesLike,
  SpeciationOptions,
  SpeciationHarnessContext,
} from '../../shared/neat.shared.types';
import {
  DEFAULT_SCORE_FALLBACK,
  DEFAULT_SPECIES_AGE_GRACE,
  DEFAULT_SPECIES_OLD_PENALTY,
  HISTORY_BUFFER_MAX_ENTRIES,
  NEGATIVE_INFINITY,
  PENALTY_NO_EFFECT_THRESHOLD,
  SPECIES_AGE_GRACE_MULTIPLIER,
} from '../shared/speciation.shared';
import type { InnovationAccumulator } from '../shared/speciation.shared';

/**
 * History and telemetry mechanics for speciation.
 *
 * This chapter captures what happened after speciation ran: age-based species
 * protection, per-generation history snapshots, and the extended innovation
 * statistics used by teaching and telemetry surfaces.
 */

/**
 * Apply age protection penalties to old species.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @returns Nothing.
 */
export function applyAgeProtection<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void {
  // Step 1: Resolve configured protection policy.
  const ageProtection = options.speciesAgeProtection ?? {
    grace: DEFAULT_SPECIES_AGE_GRACE,
    oldPenalty: DEFAULT_SPECIES_OLD_PENALTY,
  };
  // Step 2: Apply penalties to old species when configured.
  for (const species of speciationContext._species) {
    const createdGeneration =
      speciationContext._speciesCreated.get(species.id) ??
      speciationContext.generation;
    const speciesAge = speciationContext.generation - createdGeneration;
    const graceMultiplier =
      (ageProtection.grace ?? DEFAULT_SPECIES_AGE_GRACE) *
      SPECIES_AGE_GRACE_MULTIPLIER;
    if (speciesAge < graceMultiplier) continue;
    const penalty = ageProtection.oldPenalty ?? DEFAULT_SPECIES_OLD_PENALTY;
    if (penalty >= PENALTY_NO_EFFECT_THRESHOLD) continue;
    (species.members as GenomeDetailed[]).forEach((member) => {
      if (typeof member.score === 'number') member.score *= penalty;
    });
  }
}

/**
 * Record the current species history snapshot.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @returns Nothing.
 */
export function recordHistory<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void {
  // Step 1: Select history format based on configuration.
  if (options.speciesAllocation?.extendedHistory) {
    const speciesStats = speciationContext._species.map(
      (species: SpeciesLike) =>
        buildExtendedHistoryStats(speciationContext, species),
    );
    speciationContext._speciesHistory.push({
      generation: speciationContext.generation,
      stats: speciesStats,
    });
    return;
  }
  speciationContext._speciesHistory.push({
    generation: speciationContext.generation,
    stats: speciationContext._species.map((species: SpeciesLike) => ({
      id: species.id,
      size: species.members.length,
      best: species.bestScore,
    })),
  });
}

/**
 * Trim species history to the maximum buffer size.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function trimHistory<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Drop the oldest history entry when the buffer is too large.
  if (speciationContext._speciesHistory.length > HISTORY_BUFFER_MAX_ENTRIES)
    speciationContext._speciesHistory.shift();
}

/**
 * Build extended history stats for a species.
 *
 * @param speciationContext - Speciation harness context.
 * @param species - Species to snapshot.
 * @returns Extended history entry.
 */
function buildExtendedHistoryStats<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  species: SpeciesLike,
): Record<string, unknown> {
  // Step 1: Snapshot members for structural and innovation summaries.
  const members = species.members as GenomeDetailed[];
  // Step 2: Compute structural sizes and their averages.
  const structuralSizes = computeMemberStructuralSizes(
    speciationContext,
    members,
  );
  const meanNodes = averageNumbers(structuralSizes.map((entry) => entry.nodes));
  const meanConnections = averageNumbers(
    structuralSizes.map((entry) => entry.connections),
  );
  // Step 3: Aggregate innovation statistics.
  const innovationStats = summarizeInnovations(speciationContext, members);
  // Step 4: Fold into the history stats payload.
  return {
    id: species.id,
    size: species.members.length,
    best: species.bestScore,
    meanNodes,
    meanConns: meanConnections,
    meanInnovation: innovationStats.meanInnovation,
    innovationRange: innovationStats.innovationRange,
    enabledRatio: innovationStats.enabledRatio,
  } as Record<string, unknown>;

  /**
   * @param context - Speciation harness context.
   * @param memberList - Members to summarize.
   * @returns Structural size stats per member.
   */
  function computeMemberStructuralSizes(
    context: SpeciationHarnessContext<TOptions>,
    memberList: GenomeDetailed[],
  ): Array<{
    nodes: number;
    connections: number;
    score: number;
    entropy: number;
  }> {
    // Step 1: Map members into structural summary rows.
    return memberList.map((member) => ({
      nodes: member.nodes.length,
      connections: member.connections.length,
      score: member.score ?? DEFAULT_SCORE_FALLBACK,
      entropy: context._structuralEntropy(member),
    }));
  }
}

/**
 * Average a list of numbers, returning zero when empty.
 *
 * @param values - Numeric values to average.
 * @returns Mean of the values or zero.
 */
function averageNumbers(values: number[]): number {
  // Step 1: Compute mean with a safe empty fallback.
  if (!values.length) return DEFAULT_SCORE_FALLBACK;
  const sum = values.reduce((total, value) => total + value, 0);
  return sum / values.length;
}

/**
 * Summarize innovation statistics for a set of members.
 *
 * @param speciationContext - Speciation harness context.
 * @param members - Members to summarize.
 * @returns Innovation summary statistics.
 */
function summarizeInnovations<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  members: GenomeDetailed[],
): {
  meanInnovation: number;
  innovationRange: number;
  enabledRatio: number;
} {
  // Step 1: Accumulate innovation data across all member connections.
  const innovationAccumulation = accumulateInnovationStats(
    speciationContext,
    members,
  );
  // Step 2: Compute mean, range, and enabled ratio from the accumulation.
  const meanInnovation = computeInnovationMean(innovationAccumulation);
  const innovationRange = computeInnovationRange(innovationAccumulation);
  const enabledRatio = computeEnabledRatio(innovationAccumulation);
  return { meanInnovation, innovationRange, enabledRatio };

  /**
   * @param context - Speciation harness context.
   * @param memberList - Members to summarize.
   * @returns Accumulated innovation stats.
   */
  function accumulateInnovationStats(
    context: SpeciationHarnessContext<TOptions>,
    memberList: GenomeDetailed[],
  ): InnovationAccumulator {
    // Step 1: Seed accumulator defaults.
    const accumulator: InnovationAccumulator = {
      innovationSum: 0,
      innovationCount: 0,
      maxInnovation: NEGATIVE_INFINITY,
      minInnovation: Infinity,
      enabledCount: 0,
      disabledCount: 0,
    };
    // Step 2: Walk every connection and update the counters.
    for (const member of memberList) {
      for (const connection of member.connections as ConnectionLike[]) {
        applyConnectionInnovation(context, accumulator, connection);
      }
    }
    return accumulator;
  }

  /**
   * @param context - Speciation harness context.
   * @param accumulator - Mutable innovation accumulator.
   * @param connection - Connection to process.
   * @returns Nothing.
   */
  function applyConnectionInnovation(
    context: SpeciationHarnessContext<TOptions>,
    accumulator: InnovationAccumulator,
    connection: ConnectionLike,
  ): void {
    // Step 1: Resolve the innovation id for this connection.
    const innovation =
      connection.innovation ?? context._fallbackInnov(connection);
    // Step 2: Update numeric aggregates.
    accumulator.innovationSum += innovation;
    accumulator.innovationCount += 1;
    if (innovation > accumulator.maxInnovation)
      accumulator.maxInnovation = innovation;
    if (innovation < accumulator.minInnovation)
      accumulator.minInnovation = innovation;
    // Step 3: Track enabled and disabled totals.
    if (connection.enabled === false) accumulator.disabledCount += 1;
    else accumulator.enabledCount += 1;
  }

  /**
   * @param accumulator - Innovation accumulator.
   * @returns Mean innovation or fallback.
   */
  function computeInnovationMean(accumulator: InnovationAccumulator): number {
    // Step 1: Return the mean when there are observations.
    if (!accumulator.innovationCount) return DEFAULT_SCORE_FALLBACK;
    return accumulator.innovationSum / accumulator.innovationCount;
  }

  /**
   * @param accumulator - Innovation accumulator.
   * @returns Range of innovation ids or fallback.
   */
  function computeInnovationRange(accumulator: InnovationAccumulator): number {
    // Step 1: Validate min and max and compute the range.
    if (
      !Number.isFinite(accumulator.maxInnovation) ||
      !Number.isFinite(accumulator.minInnovation) ||
      accumulator.maxInnovation <= accumulator.minInnovation
    ) {
      return DEFAULT_SCORE_FALLBACK;
    }
    return accumulator.maxInnovation - accumulator.minInnovation;
  }

  /**
   * @param accumulator - Innovation accumulator.
   * @returns Ratio of enabled connections or fallback.
   */
  function computeEnabledRatio(accumulator: InnovationAccumulator): number {
    // Step 1: Compute ratio when any connections were observed.
    const enabledTotal = accumulator.enabledCount + accumulator.disabledCount;
    if (!enabledTotal) return DEFAULT_SCORE_FALLBACK;
    return accumulator.enabledCount / enabledTotal;
  }
}
