/**
 * Deterministic ranking helpers for trainer populations.
 *
 * These utilities keep score extraction and descending-order selection in one
 * place so the staged evaluation services do not each reinvent the same sorting
 * logic with slightly different fallback rules.
 */
import type { FlappyTrainerNetwork, ScoredGenomeEntry } from './trainer.types';

/**
 * Returns top genomes ordered by current provisional score.
 *
 * @param population - Current trainer population.
 * @param provisionalScoresByGenome - Optional map of staged provisional scores.
 * @param targetCount - Maximum number of genomes to return.
 * @returns Highest-scoring genomes in descending score order.
 */
export function selectTopGenomesByScore(
  population: readonly FlappyTrainerNetwork[],
  provisionalScoresByGenome: ReadonlyMap<FlappyTrainerNetwork, number>,
  targetCount: number,
): FlappyTrainerNetwork[] {
  const scoredEntries = buildScoredGenomeEntries(
    population,
    provisionalScoresByGenome,
  );

  scoredEntries.sort(compareScoredGenomeEntriesDescending);

  const selectedGenomes: FlappyTrainerNetwork[] = [];
  const maximumSelectionCount = Math.max(
    0,
    Math.min(targetCount, scoredEntries.length),
  );

  let selectedIndex = 0;
  while (selectedIndex < maximumSelectionCount) {
    const scoredEntry = scoredEntries[selectedIndex];
    if (scoredEntry) {
      selectedGenomes.push(scoredEntry.genome);
    }
    selectedIndex += 1;
  }

  return selectedGenomes;
}

/**
 * Resolves the best genome by current score.
 *
 * This helper is intentionally tiny, but it gives the rest of the trainer a
 * single vocabulary term for "the current best genome under whatever score shelf
 * is currently populated."
 *
 * @param population - Current trainer population.
 * @returns Highest-scoring genome or `undefined` when population is empty.
 */
export function resolveBestGenomeByScore(
  population: readonly FlappyTrainerNetwork[],
): FlappyTrainerNetwork | undefined {
  const scoredGenomes = buildScoredGenomeEntries(population, undefined);
  scoredGenomes.sort(compareScoredGenomeEntriesDescending);
  return scoredGenomes[0]?.genome;
}

function buildScoredGenomeEntries(
  population: readonly FlappyTrainerNetwork[],
  provisionalScoresByGenome:
    ReadonlyMap<FlappyTrainerNetwork, number> | undefined,
): ScoredGenomeEntry[] {
  const scoredEntries: ScoredGenomeEntry[] = [];

  for (const genome of population) {
    const score = resolveGenomeScore(genome, provisionalScoresByGenome);
    scoredEntries.push({ genome, score });
  }

  return scoredEntries;
}

function resolveGenomeScore(
  genome: FlappyTrainerNetwork,
  provisionalScoresByGenome:
    ReadonlyMap<FlappyTrainerNetwork, number> | undefined,
): number {
  if (provisionalScoresByGenome) {
    return provisionalScoresByGenome.get(genome) ?? Number.NEGATIVE_INFINITY;
  }

  return genome.score ?? Number.NEGATIVE_INFINITY;
}

function compareScoredGenomeEntriesDescending(
  leftEntry: ScoredGenomeEntry,
  rightEntry: ScoredGenomeEntry,
): number {
  return rightEntry.score - leftEntry.score;
}
