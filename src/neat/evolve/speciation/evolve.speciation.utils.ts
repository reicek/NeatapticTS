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
 * The evolve-time speciation bridge keeps the ranked generation coherent before
 * parent allocation and next-population construction begin.
 *
 * The root {@link ../../../README.md | evolve chapter} explains the full generation
 * lifecycle, but this file owns one narrower question inside that lifecycle:
 * once evaluation and adaptive policy updates are finished, how does evolve
 * refresh species structure, record minimal history, and recover from prolonged
 * global stagnation without widening into a full population-rebuild layer?
 *
 * Read this chapter when you want to understand:
 *
 * - why speciation and sharing are triggered here instead of inside population construction,
 * - which helpers preserve current-generation evidence before offspring allocation starts,
 * - how minimal species-history snapshots stay available even when extended history is disabled,
 * - how global-stagnation rescue injects constrained fresh genomes without replacing the whole
 *   rebuild pipeline.
 *
 * The helper flow is easiest to retain as three small responsibilities:
 *
 * 1. refresh species assignments, sharing, tuning, and ordering for the evaluated population,
 * 2. preserve species-history evidence needed for later export and diagnostics reads,
 * 3. inject a bounded fraction of fresh genomes when long-run stagnation says the current search
 *    basin has gone flat.
 *
 * ```mermaid
 * flowchart TD
 *   Scores[Fresh evaluated population] --> Speciate[Refresh species and sharing]
 *   Speciate --> Tune[Retune compatibility and re-sort]
 *   Tune --> History[Capture minimal species history when needed]
 *   History --> Check{Global stagnation threshold exceeded?}
 *   Check -- No --> Continue[Continue into offspring allocation]
 *   Check -- Yes --> Inject[Inject bounded fresh genomes]
 *   Inject --> Continue
 * ```
 */

/* Module introduction boundary for generated README output. */

/**
 * Apply speciation, fitness sharing, and related side effects.
 *
 * This is the evolve-stage bridge back into the stronger `speciation/` chapter.
 * It assumes the current population already carries fresh evaluation evidence,
 * then refreshes species membership, applies post-assignment sharing pressure,
 * lets the caller retune compatibility settings, restores deterministic best-first
 * ordering, and finally records the lightest history snapshot needed for later
 * telemetry or export reads.
 *
 * The helper intentionally stays narrow. It does not build offspring, decide
 * species quotas, or mutate the next population. Its job is to make the current
 * ranked generation internally coherent before the evolve loop moves on.
 *
 * @param internal - NEAT controller instance.
 * @param helpers - Helper callbacks used for tuning and history.
 * @param helpers.applyAutoCompatibilityTuning - Auto-compatibility adjustment helper.
 * @param helpers.recordSpeciesHistorySnapshot - Species history snapshot helper.
 * @returns A promise that resolves after the current generation has been refreshed
 * and post-speciation side effects have been applied.
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
 *
 * Global-stagnation rescue needs genomes that are genuinely new search seeds
 * but still obey the controller's minimum structural expectations. This helper
 * creates that bounded replacement candidate: a fresh network with controller
 * metadata, replay-related fields, and best-effort structural cleanup so the
 * injected genome can enter the population without widening into a bespoke
 * rebuild path.
 *
 * @param internal - NEAT controller instance.
 * @returns A new genome prepared for bounded stagnation rescue.
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
 *
 * Fresh stagnation-recovery genomes can otherwise collapse into the smallest
 * legal topology and fail to contribute any structural novelty. This helper adds
 * one conservative hidden-node bridge when the injected genome has no hidden
 * layer at all, preserving the idea that rescue should re-open search space
 * rather than only reshuffle minimal direct input-output paths.
 *
 * @param internal - NEAT controller instance.
 * @param genome - Genome to adjust.
 * @returns A promise that resolves after best-effort variance injection.
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
 *
 * This helper preserves a minimal per-generation history row for downstream
 * readers that expect species-history evidence even when the heavier extended
 * history path is disabled. It deliberately records only lightweight summary
 * fields, keeps one row per generation, and trims to a bounded rolling window
 * so evolve can maintain export-friendly evidence without turning this bridge
 * into the full history-enrichment layer.
 *
 * @param internal - NEAT controller instance.
 * @param maxHistory - Maximum history length.
 * @returns Nothing.
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
 *
 * The stagnation update remains optional because some evolve configurations use
 * the broader selection and replacement machinery without long-lived species
 * maintenance. When speciation is active, this helper advances the species-side
 * stagnation counters so later allocation and pruning decisions can distinguish
 * between active lineages and species that have stopped improving.
 *
 * @param internal - NEAT controller instance.
 * @returns Nothing.
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
 *
 * This is the evolve loop's constrained recovery valve for long periods without
 * global improvement. Instead of discarding the whole population or rebuilding
 * the generation logic from scratch, the helper replaces only the worst-ranked
 * fraction beyond elitism with fresh genomes, then resets the stagnation window
 * so the controller can test whether the new search seeds reopen progress.
 *
 * The design is intentionally conservative:
 *
 * - elites are preserved,
 * - the replacement fraction is bounded by the caller,
 * - injected genomes still pass through the normal later evolution pipeline.
 *
 * @param internal - NEAT controller instance.
 * @param helpers - Helper callbacks for stagnation injection.
 * @param helpers.buildFreshGenomeForStagnation - Genome builder for injection.
 * @returns A promise that resolves after any bounded replacements are complete.
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
 *
 * Export code may ask for species history after a generation that never took
 * the heavier history-recording path. This helper backfills the same minimal
 * summary shape used by {@link recordSpeciesHistorySnapshot} so export and
 * inspection code can still rely on one bounded row per generation without
 * forcing extended history to stay on permanently.
 *
 * @param internal - NEAT controller instance.
 * @param maxHistory - Maximum history length.
 * @returns Nothing.
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
