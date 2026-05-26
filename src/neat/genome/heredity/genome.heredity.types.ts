import type { NeatGenome, NeatGenomeConnectionGene } from '../genome.types';

/**
 * Stable string literal labels identifying which parent contributed a given connection gene during the genome-owned heredity selection pass.
 */
export type GenomeHereditySourceParent = 'parent1' | 'parent2';

/**
 * Pure selection context for innovation-aligned genome heredity.
 *
 * This contract keeps the heredity pass structural-first: two strict genomes,
 * the fitness or equality policy that decides disjoint inheritance, and the
 * deterministic RNG that resolves matching-gene choices plus disabled-gene
 * re-enable behavior.
 */
export interface GenomeHereditySelectionContext {
  /** First parent genome. */
  parent1Genome: NeatGenome;
  /** Second parent genome. */
  parent2Genome: NeatGenome;
  /** First parent score. */
  parent1Score: number;
  /** Second parent score. */
  parent2Score: number;
  /** Optional first-parent disabled-gene re-enable probability. */
  parent1ReenableProbability?: number;
  /** Optional second-parent disabled-gene re-enable probability. */
  parent2ReenableProbability?: number;
  /** True when disjoint or excess genes should stay symmetric. */
  equal: boolean;
  /** Deterministic random generator for heredity decisions. */
  randomGenerator: () => number;
}

/**
 * One inherited connection gene selected by the genome-owned heredity pass.
 *
 * The source-parent label preserves parent provenance for runtime adapters and
 * future narrow seams without pushing runtime node indexing into the genome
 * surface.
 */
export interface SelectedGenomeConnectionGene {
  /** Cloned inherited connection gene. */
  connectionGene: NeatGenomeConnectionGene;
  /** Parent that contributed the inherited structural gene. */
  sourceParent: GenomeHereditySourceParent;
}
