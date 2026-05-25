import type {
  NetworkArchitectureDescriptor,
  NetworkTopologyIntent,
} from '../../architecture/network/network.types';

/**
 * Canonical node-role literals supported by the first-pass NEAT genome boundary for input, hidden, and output nodes.
 */
export type NeatGenomeNodeType = 'input' | 'hidden' | 'output';

/**
 * Supported recurrent-module family identifiers tracked by the Step 7.4 temporal extension lane for LSTM, GRU, and NARX.
 */
export type NeatGenomeRecurrentModuleKind = 'lstm' | 'gru' | 'narx-memory';

/**
 * Descriptor for one explicit recurrent module stored in the extension bag.
 *
 * The executable graph still lives in canonical node and connection genes.
 * This additive descriptor preserves the higher-level block identity so
 * checkpoints and later mutation passes can tell an intentional recurrent
 * module from an arbitrary cyclic subgraph. Disabling one referenced
 * connection gene does not retire the module by itself; the descriptor stays
 * valid until the referenced genes or gating ownership disappear structurally.
 */
export interface NeatGenomeRecurrentModuleDescriptor {
  /** Stable module identifier within the extension bag. */
  moduleId: string;
  /** Supported recurrent-module family. */
  kind: NeatGenomeRecurrentModuleKind;
  /** Role-to-node-gene map describing the block's internal node groups. */
  nodeGeneIdsByRole: Record<string, number[]>;
  /** Connection innovations that belong to the block. */
  connectionInnovations: number[];
}

/**
 * Descriptor for one explicit gated block stored in the extension bag.
 *
 * Gated blocks preserve grouped gating intent without changing canonical
 * connection-gene ownership. The descriptor is therefore about identity and
 * checkpoint semantics, not about replacing individual connection genes.
 * Disabled gated connection genes still count as dormant block structure until
 * the connection or gater identity is removed from the genome.
 */
export interface NeatGenomeGatedBlockDescriptor {
  /** Stable block identifier within the extension bag. */
  blockId: string;
  /** Gater node-gene ids that govern the block's connections. */
  gaterGeneIds: number[];
  /** Connection innovations controlled by the gated block. */
  connectionInnovations: number[];
}

/**
 * Typed extension payload reserved for additive beyond-paper genome traits that augment canonical genes without widening the base connection or node contracts.
 */
export interface NeatGenomeExtensionValues extends Record<string, unknown> {
  /**
   * Non-neutral baseline gains keyed by connection innovation.
   *
   * The first extension pass keeps this map scoped to ungated connections so
   * live gating state stays phenotype-owned rather than leaking into genotype
   * state.
   */
  connectionGainByInnovation?: Record<string, number>;
  /**
   * Non-neutral response multipliers keyed by node gene id.
   *
   * The response trait stays additive so canonical node genes keep their
   * original structural contract while extension-aware checkpoints can still
   * recover explicit slope-style behavior.
   */
  nodeResponseByGeneId?: Record<string, number>;
  /**
   * Explicit disabled-connection re-enable probability for heredity flows.
   *
   * This keeps the enable/disable policy versioned and checkpoint-safe instead
   * of relying only on controller-side runtime metadata.
   */
  disabledConnectionReenableProbability?: number;
  /**
   * Explicit recurrent-module descriptors for the Step 7.4 extension lane.
   */
  recurrentModules?: NeatGenomeRecurrentModuleDescriptor[];
  /**
   * Explicit gated-block descriptors for the Step 7.4 extension lane.
   */
  gatedBlocks?: NeatGenomeGatedBlockDescriptor[];
}

/**
 * Versioned extension bag reserved for beyond-paper genome traits.
 *
 * Step 7.1 keeps this bag optional and empty by default so the structural
 * identity contract can land without immediately committing to extension
 * semantics. Later phases can add opt-in traits here without widening the core
 * node-gene and connection-gene shapes.
 */
export interface NeatGenomeExtensions {
  /** Monotonic extension-bag version used by future additive traits. */
  version: number;
  /** Opaque extension payload reserved for opt-in features. */
  values: NeatGenomeExtensionValues;
}

/**
 * Opt-in capture settings used when projecting runtime payloads into the.
 * strict genome contract.
 */
export interface NeatGenomeCaptureOptions {
  /** Capture non-neutral ungated connection gain as explicit extension state. */
  connectionGain?: boolean;
  /** Capture non-neutral node response as explicit extension state. */
  nodeResponse?: boolean;
  /** Capture disabled-connection re-enable probability as explicit extension state. */
  disabledConnectionReenableProbability?: boolean;
}

/**
 * Pure node-gene contract owned by the NEAT subtree.
 *
 * Array order carries the canonical interface ordering. Runtime node indexes do
 * not belong here because they are phenotype-only bookkeeping rebuilt during
 * materialization.
 */
export interface NeatGenomeNodeGene {
  /** Stable historical node identity. */
  geneId: number;
  /** Canonical node role. */
  type: NeatGenomeNodeType;
  /** Persisted node bias. */
  bias: number;
  /** Stable activation identifier. */
  squash: string;
}

/**
 * Pure structural connection-gene contract owned by the NEAT genome subtree, carrying innovation identity, endpoint gene ids, weight, enabled state, and optional gater.
 */
export interface NeatGenomeConnectionGene {
  /** Stable historical connection identity. */
  innovation: number;
  /** Stable source node-gene id. */
  fromGeneId: number;
  /** Stable target node-gene id. */
  toGeneId: number;
  /** Persisted connection weight. */
  weight: number;
  /** Explicit enabled state. */
  enabled: boolean;
  /** Stable gater node-gene id when one exists. */
  gaterGeneId: number | null;
}

/**
 * First-pass genotype contract for the proper-NEAT lift.
 *
 * This is intentionally structural only. Replay state, species membership,
 * controller-owned genome ids, caches, runtime node indexes, and activation
 * traces stay outside this contract.
 */
export interface NeatGenome {
  /** Input count of the executable phenotype this genome can materialize. */
  input: number;
  /** Output count of the executable phenotype this genome can materialize. */
  output: number;
  /** Public topology contract carried by the genome. */
  topologyIntent?: NetworkTopologyIntent;
  /** Ordered node-gene list. */
  nodeGenes: NeatGenomeNodeGene[];
  /** Connection-gene list aligned by historical identity. */
  connectionGenes: NeatGenomeConnectionGene[];
  /** Optional beyond-paper extension bag. */
  extensions?: NeatGenomeExtensions;
}

/**
 * Optional phenotype-only hints applied when materializing a `Network` from one
 * strict genome contract.
 *
 * These values do not become genome state. They exist so checkpoint import and
 * other bridges can preserve non-genetic runtime metadata while the genotype
 * boundary stays narrow.
 */
export interface GenomeMaterializationRuntimeHints {
  /** Optional dropout value restored onto the phenotype payload. */
  dropout?: number;
  /** Optional architecture descriptor used by diagnostics consumers. */
  architecture?: NetworkArchitectureDescriptor;
}

/**
 * Stable machine-readable issue codes produced by the pure genome validator for each detected structural violation.
 */
export type NeatGenomeValidationIssueCode =
  | 'invalid-input-count'
  | 'invalid-output-count'
  | 'insufficient-node-count'
  | 'missing-node-gene-id'
  | 'duplicate-node-gene-id'
  | 'invalid-node-type'
  | 'invalid-node-bias'
  | 'invalid-node-squash'
  | 'input-node-order-mismatch'
  | 'output-node-order-mismatch'
  | 'missing-connection-innovation'
  | 'duplicate-connection-innovation'
  | 'invalid-connection-weight'
  | 'unknown-connection-endpoint'
  | 'unknown-gater-node'
  | 'feed-forward-recurrent-connection'
  | 'invalid-connection-gain-extension'
  | 'invalid-node-response-extension'
  | 'invalid-connection-reenable-extension'
  | 'invalid-recurrent-module-extension'
  | 'invalid-gated-block-extension'
  | 'invalid-extensions-bag';

/**
 * One structured finding produced by the pure genome validator carrying a stable code, path, and human-readable message.
 */
export interface NeatGenomeValidationIssue {
  /** Stable machine-readable issue code. */
  code: NeatGenomeValidationIssueCode;
  /** Structural path pointing to the failing genome field. */
  path: string;
  /** Human-readable explanation of the failure. */
  message: string;
  /** Optional structured diagnostics for tests and tooling. */
  detail?: Record<string, unknown>;
}

/**
 * Complete validation report returned for one strict genome contract containing all structural findings and summary counts.
 */
export interface NeatGenomeValidationReport {
  /** Whether the strict genome contract passed all checks. */
  isValid: boolean;
  /** Input count carried by the contract. */
  input: number;
  /** Output count carried by the contract. */
  output: number;
  /** Public topology contract carried by the genome. */
  topologyIntent?: NetworkTopologyIntent;
  /** Node-gene count. */
  nodeCount: number;
  /** Connection-gene count. */
  connectionCount: number;
  /** Structured findings, empty when valid. */
  issues: NeatGenomeValidationIssue[];
}
