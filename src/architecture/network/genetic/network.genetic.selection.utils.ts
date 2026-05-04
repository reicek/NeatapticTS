import type {
  ConnectionGene,
  GeneticNetwork,
  ParentMetrics,
} from '../network.types';
import {
  createGenomeFromNetwork,
  selectGenomeHeredityConnectionGenes,
  type GenomeHereditySelectionContext,
  type NeatGenome,
  type SelectedGenomeConnectionGene,
} from '../../../neat/genome/genome';
import type { RandomGenerator } from './network.genetic.utils.types';

/**
 * Adapts genome-owned heredity selection back into runtime crossover genes.
 *
 * Step 7.2b keeps this runtime shelf as one thin bridge that:
 *
 * 1. projects each parent into the strict genome contract,
 * 2. asks the genome boundary which structural genes survive, and
 * 3. strips transient runtime node-index hints so the phenotype materializer
 *    reads only stable gene ids plus the inherited weight, enabled state, and
 *    innovation identity.
 *
 * Runtime node scaffolding, topology pruning, and gating reattachment remain
 * outside this adapter.
 *
 * @param parent1 First runtime parent.
 * @param parent2 Second runtime parent.
 * @param parentMetrics Shared score summary used by heredity policy.
 * @param equal Equal-treatment mode flag.
 * @param randomGenerator Deterministic crossover RNG.
 * @returns Chosen genes for runtime materialization.
 */
export function chooseConnectionGenes(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: RandomGenerator,
): ConnectionGene[] {
  const parent1Genome = createGenomeFromNetwork(parent1);
  const parent2Genome = createGenomeFromNetwork(parent2);
  const selectionContext = createGenomeSelectionContext(
    parentMetrics,
    equal,
    randomGenerator,
    parent1Genome,
    parent2Genome,
    parent1,
    parent2,
  );
  return selectGenomeHeredityConnectionGenes(selectionContext).map(
    createMaterializationConnectionGene,
  );
}

function createMaterializationConnectionGene(
  selectedGenomeGene: SelectedGenomeConnectionGene,
): ConnectionGene {
  const { connectionGene } = selectedGenomeGene;

  return {
    weight: connectionGene.weight,
    innovation: connectionGene.innovation,
    fromGeneId: connectionGene.fromGeneId,
    toGeneId: connectionGene.toGeneId,
    gaterGeneId: connectionGene.gaterGeneId,
    enabled: connectionGene.enabled,
  };
}

function createGenomeSelectionContext(
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: RandomGenerator,
  parent1Genome: NeatGenome,
  parent2Genome: NeatGenome,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
): GenomeHereditySelectionContext {
  return {
    parent1Genome,
    parent2Genome,
    parent1Score: parentMetrics.score1,
    parent2Score: parentMetrics.score2,
    parent1ReenableProbability: parent1._reenableProb,
    parent2ReenableProbability: parent2._reenableProb,
    equal,
    randomGenerator,
  };
}
