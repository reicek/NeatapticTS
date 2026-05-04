/**
 * First-class genome boundary for the proper-NEAT Step 7.1 lift.
 *
 * This chapter owns the strict structural contract that sits between the NEAT
 * controller and the executable `Network` phenotype. Its job is deliberately
 * narrow in the first pass:
 *
 * 1. define the pure node-gene and connection-gene contract,
 * 2. provide validated phenotype-to-genome and genome-to-phenotype adapters,
 * 3. expose a pure genome-facing validation seam that later chapters can adopt
 *    without widening the public runtime `Network` surface.
 *
 * Step 7.2a adds the first heredity-specific subchapter under
 * `genome/heredity/`, where innovation-aligned connection-gene selection can
 * move off the runtime crossover shelf while phenotype scaffolding stays in the
 * `Network` layer.
 *
 * Runtime activation state, species membership, controller replay metadata,
 * and architecture allocation counters remain outside this boundary.
 */

export type {
  NeatGenomeCaptureOptions,
  NeatGenomeGatedBlockDescriptor,
  GenomeMaterializationRuntimeHints,
  NeatGenome,
  NeatGenomeConnectionGene,
  NeatGenomeExtensionValues,
  NeatGenomeExtensions,
  NeatGenomeNodeGene,
  NeatGenomeNodeType,
  NeatGenomeRecurrentModuleDescriptor,
  NeatGenomeRecurrentModuleKind,
  NeatGenomeValidationIssue,
  NeatGenomeValidationIssueCode,
  NeatGenomeValidationReport,
} from './genome.types';
export type {
  GenomeHereditySelectionContext,
  GenomeHereditySourceParent,
  SelectedGenomeConnectionGene,
} from './heredity/genome.heredity.types';
export {
  assertValidGenomeContract,
  createCompatibilityGenomeView,
  createGenomeFromNetwork,
  createGenomeFromNetworkJson,
  createNetworkFromGenome,
  createNetworkJsonFromGenome,
  validateGenomeContract,
} from './genome.utils';
export { selectGenomeHeredityConnectionGenes } from './heredity/genome.heredity';
export {
  NeatGenomeConversionError,
  NeatGenomeValidationError,
} from './genome.errors';
