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
import {
  assertValidGenomeContract as assertValidGenomeContractImpl,
  createCompatibilityGenomeView as createCompatibilityGenomeViewImpl,
  createGenomeFromNetwork as createGenomeFromNetworkImpl,
  createGenomeFromNetworkJson as createGenomeFromNetworkJsonImpl,
  createNetworkFromGenome as createNetworkFromGenomeImpl,
  createNetworkJsonFromGenome as createNetworkJsonFromGenomeImpl,
  validateGenomeContract as validateGenomeContractImpl,
} from './genome.utils';
export { selectGenomeHeredityConnectionGenes } from './heredity/genome.heredity';
export {
  NeatGenomeConversionError,
  NeatGenomeValidationError,
} from './genome.errors';

/**
 * Assert that one strict genome contract is structurally valid before crossover, mutation, import, replay, or serialization boundaries consume it.
 * This forwarding seam keeps barrel-level API docs explicit while implementation details remain in the utility chapter.
 */
export const assertValidGenomeContract = assertValidGenomeContractImpl;

/**
 * Create the compatibility-layer genome view used by legacy paths that still bridge strict genome contracts and runtime phenotypes safely.
 * The view preserves adapter behavior while allowing strict-genome-first internals to evolve independently.
 */
export const createCompatibilityGenomeView = createCompatibilityGenomeViewImpl;

/**
 * Capture one runtime phenotype network into the strict genome contract while preserving only explicitly modeled extension families and invariants.
 * This documented export keeps genome capture semantics discoverable from the chapter entrypoint.
 */
export const createGenomeFromNetwork = createGenomeFromNetworkImpl;

/**
 * Convert one versioned network JSON payload into a validated strict genome contract for deterministic NEAT-core heredity and evaluation workflows.
 * The conversion route is intentionally explicit at the barrel surface for docs consumers.
 */
export const createGenomeFromNetworkJson = createGenomeFromNetworkJsonImpl;

/**
 * Materialize one executable runtime phenotype from a validated strict genome contract plus optional runtime-only hints and diagnostics metadata.
 * Keeping this forwarder documented helps users discover genome-to-network materialization from the public chapter.
 */
export const createNetworkFromGenome = createNetworkFromGenomeImpl;

/**
 * Convert one strict genome contract back into the versioned network JSON payload consumed by runtime serializers and import seams consistently.
 * This export documents the genome-to-json bridge as a first-class persistence boundary.
 */
export const createNetworkJsonFromGenome = createNetworkJsonFromGenomeImpl;

/**
 * Validate one strict genome contract and return a structured report that callers can inspect before deciding to throw validation errors.
 * This forwarding seam supports diagnostics-first workflows while sharing one canonical validator implementation.
 */
export const validateGenomeContract = validateGenomeContractImpl;
