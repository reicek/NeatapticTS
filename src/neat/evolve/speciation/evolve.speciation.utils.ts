/*
 * ESLint configuration for intentional `any` usage in NEAT evolution speciation utils
 *
 * This file mirrors the evolution module's runtime metadata handling,
 * where dynamic properties are attached to genomes/species at runtime.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import Network from '../../../architecture/network';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
} from '../evolve.types';

/**
 * Apply speciation, fitness sharing, and related side effects.
 * @param internal - NEAT controller instance.
 * @param helpers - Helper callbacks used for tuning and history.
 * @param helpers.applyAutoCompatibilityTuning - Auto-compatibility adjustment helper.
 * @param helpers.recordSpeciesHistorySnapshot - Species history snapshot helper.
 * @returns void.
 */
export async function applySpeciationAndSharingIfEnabled(
  internal: NeatControllerForEvolution,
  helpers: {
    applyAutoCompatibilityTuning: () => void;
    recordSpeciesHistorySnapshot: () => void;
  },
): Promise<void> {
  // Step 1: Exit early when speciation is disabled.
  if (!internal.options.speciation) return;
  // Step 2: Perform speciation and fitness sharing.
  try {
    internal._speciate?.();
  } catch {
    // Empty catch: speciation is optional and may fail in edge cases.
  }
  try {
    internal._applyFitnessSharing?.();
  } catch {
    // Empty catch: fitness sharing is optional and may fail if species data is incomplete.
  }
  // Step 3: Apply auto compatibility tuning.
  helpers.applyAutoCompatibilityTuning();
  // Step 4: Re-sort after sharing adjustments.
  internal.sort?.();
  // Step 5: Record species history snapshots.
  helpers.recordSpeciesHistorySnapshot();
}

/**
 * Build a fresh genome for stagnation injection.
 * @param internal - NEAT controller instance.
 * @returns new genome with minimum constraints.
 */
export async function buildFreshGenomeForStagnation(
  internal: NeatControllerForEvolution,
): Promise<GenomeWithMetadata> {
  // Step 1: Create a new random network.
  const fresh = new Network(internal.input, internal.output, {
    minHidden: internal.options.minHidden,
  }) as never as GenomeWithMetadata;
  fresh.score = undefined;
  fresh._reenableProb = internal.options.reenableProb;
  fresh._id = internal._nextGenomeId++;
  if (internal._lineageEnabled) {
    fresh._parents = [];
    fresh._depth = 0;
  }
  // Step 2: Enforce constraints and inject variance if needed.
  try {
    await internal.ensureMinHiddenNodes?.(fresh);
    await internal.ensureNoDeadEnds?.(fresh);
    await ensureHiddenNodeVariance(internal, fresh);
  } catch {
    // Empty catch: constraints are best-effort during stagnation injection.
  }
  return fresh;
}

/**
 * Ensure a minimal hidden-node variance in injected genomes.
 * @param internal - NEAT controller instance.
 * @param genome - Genome to adjust.
 * @returns void.
 */
async function ensureHiddenNodeVariance(
  internal: NeatControllerForEvolution,
  genome: GenomeWithMetadata,
): Promise<void> {
  // Step 1: Check hidden node count.
  const hiddenCount = genome.nodes.filter(
    (node: any) => node.type === 'hidden',
  ).length;
  if (hiddenCount !== 0) return;
  // Step 2: Insert a hidden node and connect it.
  const { default: NodeCls } = await import('../../../architecture/node');
  const newNode = new NodeCls('hidden');
  genome.nodes.splice(genome.nodes.length - internal.output, 0, newNode);
  const inputNodes = genome.nodes.filter((node: any) => node.type === 'input');
  const outputNodes = genome.nodes.filter(
    (node: any) => node.type === 'output',
  );
  if (!inputNodes.length || !outputNodes.length) return;
  try {
    (genome as never as Network).connect(
      inputNodes[0] as never,
      newNode as never,
      1,
    );
  } catch {
    // Empty catch: connection may fail if constraints prevent it.
  }
  try {
    (genome as never as Network).connect(
      newNode as never,
      outputNodes[0] as never,
      1,
    );
  } catch {
    // Empty catch: connection may fail if constraints prevent it.
  }
}

/**
 * Record a species history snapshot when needed.
 * @param internal - NEAT controller instance.
 * @param maxHistory - Maximum history length.
 * @returns void.
 */
export function recordSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void {
  // Step 1: Skip when extended history is active.
  if (internal.options.speciesAllocation?.extendedHistory) return;
  // Step 2: Append a minimal snapshot if needed.
  try {
    if (!internal._speciesHistory) internal._speciesHistory = [];
    if (
      internal._speciesHistory.length === 0 ||
      internal._speciesHistory.at(-1)?.generation !== internal.generation
    ) {
      internal._speciesHistory.push({
        generation: internal.generation,
        stats: (internal._species ?? []).map((species: any) => ({
          id: species.id,
          size: species.members.length,
          best: species.bestScore,
          lastImproved: species.lastImproved,
        })),
      });
      if (internal._speciesHistory.length > maxHistory)
        internal._speciesHistory.shift();
    }
  } catch {
    // Empty catch: species history tracking is optional.
  }
}

/**
 * Update species stagnation status when speciation enabled.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function updateSpeciesStagnationIfEnabled(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Skip when speciation disabled.
  if (!internal.options.speciation) return;
  // Step 2: Update stagnation counters.
  internal._updateSpeciesStagnation?.();
}

/**
 * Apply global stagnation injection if configured.
 * @param internal - NEAT controller instance.
 * @param helpers - Helper callbacks for stagnation injection.
 * @param helpers.buildFreshGenomeForStagnation - Genome builder for injection.
 * @returns void.
 */
export async function applyGlobalStagnationInjectionIfNeeded(
  internal: NeatControllerForEvolution,
  helpers: {
    buildFreshGenomeForStagnation: () => Promise<
      import('../evolve.types').GenomeWithMetadata
    >;
    replaceFraction: number;
  },
): Promise<void> {
  // Step 1: Guard against disabled configuration.
  const threshold = internal.options.globalStagnationGenerations || 0;
  if (threshold <= 0) return;
  if (
    internal.generation - (internal._lastGlobalImproveGeneration ?? 0) <=
    threshold
  ) {
    return;
  }
  // Step 2: Replace worst fraction with fresh genomes.
  const startIndex = Math.max(
    internal.options.elitism || 0,
    Math.floor(internal.population.length * (1 - helpers.replaceFraction)),
  );
  for (
    let populationIndex = startIndex;
    populationIndex < internal.population.length;
    populationIndex++
  ) {
    const freshGenome = await helpers.buildFreshGenomeForStagnation();
    internal.population[populationIndex] = freshGenome as never;
  }
  // Step 3: Reset stagnation window.
  internal._lastGlobalImproveGeneration = internal.generation;
}

/**
 * Ensure a minimal species history snapshot exists for exports.
 * @param internal - NEAT controller instance.
 * @param maxHistory - Maximum history length.
 * @returns void.
 */
export function ensureSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void {
  // Step 1: Skip when extended history is active.
  if (internal.options.speciesAllocation?.extendedHistory) return;
  // Step 2: Build a minimal snapshot when needed.
  try {
    if (!internal._speciesHistory) internal._speciesHistory = [];
    if (
      internal._speciesHistory.length === 0 ||
      internal._speciesHistory.at(-1)?.generation !== internal.generation
    ) {
      internal._speciesHistory.push({
        generation: internal.generation,
        stats: (internal._species ?? []).map((species: any) => ({
          id: species.id,
          size: species.members.length,
          best: species.bestScore,
          lastImproved: species.lastImproved,
        })),
      });
      if (internal._speciesHistory.length > maxHistory)
        internal._speciesHistory.shift();
    }
  } catch {
    // Empty catch: species history tracking is optional telemetry.
  }
}
