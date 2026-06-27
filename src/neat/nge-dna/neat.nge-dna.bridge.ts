/**
 * Canonical envelope bridge between the NGE developmental pipeline and the
 * runtime {@link Network} substrate.
 *
 * The NGE developmental pipeline produces a fully realized phenotype
 * descriptor — a JSON-serializable blueprint of modules and edges — through a
 * deterministic sequence of rule execution, CPPN evaluation, and substrate
 * placement. The bridge is the final translation step: it turns that blueprint
 * into a live, mutable `Network` that the rest of the NeatapticTS runtime can
 * train, evaluate, and serialize exactly like any classic-NEAT network.
 *
 * The bridge also provides the reverse path: extracting the canonical NGE DNA
 * envelope back out of a serialized `Network` so that round-trips preserve
 * identity, substrate metadata, and fingerprint integrity.
 *
 * ## Why the bridge routes through NetworkJSON
 *
 * Rather than constructing `Node` and `Connection` objects directly, the
 * bridge assembles a {@link NetworkJSON} intermediate and delegates hydration
 * to the canonical serializer (`fromJSONImpl`). This mirrors the genome bridge
 * pattern used in `genome.utils.ts` and ensures that historical identity
 * restoration, extension-bag attachment, and node/connection lifecycle all flow
 * through a single, well-tested code path. Two materializations of the same
 * DNA produce byte-identical `toJSON()` payloads regardless of global counter
 * drift because `geneId` and `innovation` are injected deterministically.
 *
 * ## Determinism contract
 *
 * - **Node `geneId`** = module index + 1 (1-based, position-stable).
 * - **Connection `innovation`** = djb2-style hash of `sourceModuleId|targetModuleId`,
 *   always positive. Identical adjacencies always restore the same historical
 *   identity across round-trips.
 * - **Zone → node type**: `z=0` → `input`, `z=1` → `output`, `z>1` → `hidden`.
 * - **Squash mapping** (NGE enabled): `DenseFeedForward→relu`,
 *   `AttentionHead→sigmoid`, `GatedRecurrentCell→tanh`,
 *   `EpisodicSlot→identity`, `ModulatorBroadcaster→identity`,
 *   `GatingRouter→sigmoid`.
 *
 * ## Opt-in isolation
 *
 * When `runtimeHints.ngeEnabled` is not `true`, the bridge omits the NGE
 * extension bag entirely and collapses every squash function to `identity`.
 * Classic-NEAT consumers therefore see a plain `Network` with no NGE
 * properties, preserving full backward compatibility. The NGE surface is
 * strictly opt-in.
 *
 * ## Extension carrier
 *
 * The NGE extension carrier is stored inside `NetworkJSONExtensions.values` as:
 * `{ version: 1, ngeDescriptor, ngeEnvelope }`. The outer
 * `NetworkJSONExtensions.version` tracks the bag wrapper; the inner
 * `version` tracks the carrier schema so downstream consumers can migrate
 * independently.
 *
 * ```mermaid
 * flowchart LR
 *   classDef dna fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef plan fill:#0f2233,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef pheno fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *   classDef bridge fill:#1a0f1a,stroke:#ff6b9d,color:#ffd6e5,stroke-width:1.5px;
 *   classDef net fill:#08131f,stroke:#06d6a0,color:#bff7e6,stroke-width:1.5px;
 *
 *   env["NgeDnaCanonicalEnvelope"]:::dna
 *   plan["buildVirtualPlan"]:::plan
 *   real["realizePhenotype"]:::plan
 *   desc["NgeRealizedPhenotypeDescriptor"]:::pheno
 *   mat["materializeNetworkFromPhenotype"]:::bridge
 *   json["NetworkJSON intermediate"]:::bridge
 *   net["Network"]:::net
 *
 *   env -->|"seed"| plan
 *   plan -->|"plan, seed"| real
 *   real --> desc
 *   desc --> mat
 *   env --> mat
 *   mat --> json
 *   json -->|"fromJSONImpl"| net
 *
 *   ext["extractCanonicalEnvelopeFromNetwork"]:::bridge
 *   net -->|"toJSON"| ext
 *   ext -->|"structuredClone"| env
 * ```
 *
 * ## Background reading
 *
 * The NEAT algorithm — speciation through historical markings and
 * complexification of topologies — is described in Stanley and Miikkulainen,
 * [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02).
 * The CPPN substrate-encoding idea that NGE builds on is explained in
 * Wikipedia contributors,
 * [Compositional pattern-producing network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network),
 * and the broader neuroevolution context is covered in Wikipedia contributors,
 * [Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution).
 *
 * @remarks
 * This module is not yet re-exported from the `nge-experimental` public barrel.
 * The bridge functions are consumed internally during NGE lifecycle staging and
 * will graduate to the public API once the NGE workstream stabilizes.
 */

import Network from '../../architecture/network/network';
import type {
  NetworkJSON,
  NetworkJSONConnection,
  NetworkJSONExtensions,
  NetworkJSONNode,
} from '../../architecture/network/network.types';
import { NETWORK_JSON_FORMAT_VERSION } from '../../architecture/network/serialize/network.serialize.utils.types';
import type {
  GenomeMaterializationRuntimeHints,
  NeatGenomeComputationType,
} from '../genome/genome.types';
import { NGE_DNA_BridgeError } from './neat.nge-dna.errors';
import type {
  NgeDnaCanonicalEnvelope,
  NgeRealizedEdge,
  NgeRealizedModule,
  NgeRealizedPhenotypeDescriptor,
  NgeVirtualModulePlan,
} from './neat.nge-dna.types';

/**
 * Monotonic version of the extension bag wrapper emitted by the bridge.
 * Increment when the outer {@link NetworkJSONExtensions} contract changes.
 */
const BRIDGE_EXTENSION_VERSION = 1;

/**
 * Schema version recorded inside `NetworkJSONExtensions.values` so downstream
 * consumers can migrate the NGE carrier shape independently of the bag version.
 */
const BRIDGE_EXTENSION_SCHEMA_VERSION = 1;

const DEFAULT_NODE_BIAS = 0;
const DEFAULT_CONNECTION_GAIN = 1;
const DEFAULT_CONNECTION_ENABLED = true;
const DEFAULT_DROPOUT = 0;
const NEUTRAL_GATER = null;
const SQUASH_WHEN_NGE_DISABLED = 'identity';
const INPUT_ZONE_COORDINATE = 0;
const OUTPUT_ZONE_COORDINATE = 1;
const DETERMINISTIC_HASH_OFFSET = 5;
const DETERMINISTIC_HASH_POSITIVE_FLOOR = 1;

/**
 * Fixed mapping from a realized module's computation motif to the runtime
 * squash function name. The bridge always reads from this single table so the
 * materialization pass stays declarative and extension-safe.
 */
const COMPUTATION_TYPE_TO_SQUASH: Readonly<
  Record<NeatGenomeComputationType, string>
> = {
  DenseFeedForward: 'relu',
  AttentionHead: 'sigmoid',
  GatedRecurrentCell: 'tanh',
  EpisodicSlot: 'identity',
  ModulatorBroadcaster: 'identity',
  GatingRouter: 'sigmoid',
};

/** Lookup table from the unit-cube zone coordinate to the runtime node type. */
type ModuleIdToNodeIndex = Map<string, number>;

/** Shape of the NGE extension carrier stored inside `NetworkJSONExtensions.values`. */
interface NgeBridgeExtensionValues {
  version: number;
  ngeDescriptor: NgeRealizedPhenotypeDescriptor;
  ngeEnvelope: NgeDnaCanonicalEnvelope;
}

/**
 * Materialize a runtime {@link Network} from one realized NGE phenotype descriptor.
 *
 * This is the forward direction of the canonical envelope bridge: it translates a
 * fully realized phenotype descriptor (modules + edges) into a live, mutable
 * `Network` that integrates with the rest of the NeatapticTS runtime. The bridge
 * routes through a {@link NetworkJSON} intermediate so the canonical network
 * serializer (`fromJSONImpl`) owns node/connection hydration, historical identity
 * restoration, and extension-bag attachment — the same pattern used by the genome
 * bridge in `genome.utils.ts`.
 *
 * Deterministic `geneId` and `innovation` values are injected explicitly so two
 * materializations of the same DNA produce byte-identical `toJSON()` payloads
 * regardless of global counter drift. The `geneId` is the 1-based module index;
 * the `innovation` is a djb2-style hash of `sourceModuleId|targetModuleId`.
 *
 * When `runtimeHints.ngeEnabled` is `true`, the NGE extension carrier — containing
 * a structured clone of both the descriptor and the envelope — is attached to the
 * `NetworkJSON` so downstream consumers can extract it via
 * {@link extractCanonicalEnvelopeFromNetwork}. When `ngeEnabled` is not `true`,
 * the extension bag is omitted and every squash function collapses to `identity`,
 * producing a plain classic-NEAT network with no NGE surface.
 *
 * @param envelope - Canonical NGE DNA envelope carried alongside the descriptor.
 * Stored inside the extension bag when NGE is enabled.
 * @param _plan - Virtual module plan that produced the descriptor (reserved for
 * future governance checks; not required for topology materialization).
 * @param descriptor - Realized phenotype descriptor whose modules and edges are
 * translated into network nodes and connections. Must contain at least one module.
 * @param runtimeHints - Optional materialization hints. When `ngeEnabled` is not
 * `true` the NGE extension bag is omitted so classic NEAT consumers see a plain
 * Network. Defaults to `{}`.
 * @returns A runtime Network whose nodes and connections mirror the descriptor.
 * @throws {NGE_DNA_BridgeError} when the descriptor has zero modules.
 *
 * @example
 * ```ts
 * const network = materializeNetworkFromPhenotype(envelope, plan, descriptor, {
 *   ngeEnabled: true,
 * });
 * console.log(network.nodes.length); // descriptor.modules.length
 * ```
 *
 * @example
 * ```ts
 * // Classic-NEAT mode: no NGE extension bag, all squashes collapse to identity.
 * const classicNet = materializeNetworkFromPhenotype(envelope, plan, descriptor);
 * // extractCanonicalEnvelopeFromNetwork(classicNet) would throw NGE_DNA_BridgeError.
 * ```
 */
export function materializeNetworkFromPhenotype(
  envelope: NgeDnaCanonicalEnvelope,
  _plan: NgeVirtualModulePlan,
  descriptor: NgeRealizedPhenotypeDescriptor,
  runtimeHints: GenomeMaterializationRuntimeHints = {},
): Network {
  // Step 1: Reject empty descriptors before building any topology.
  if (descriptor.modules.length === 0) {
    throw new NGE_DNA_BridgeError(
      'Cannot materialize a Network from a phenotype descriptor with zero modules.',
    );
  }

  // Step 2: Map realized modules to node entries and record module-id → index.
  const moduleIdToNodeIndex: ModuleIdToNodeIndex = new Map();
  const nodes = descriptor.modules.map((module, index) => {
    moduleIdToNodeIndex.set(module.moduleId, index);
    return buildNodeJsonEntry(module, index, runtimeHints);
  });

  // Step 3: Map realized edges to connection entries using the index lookup.
  const connections = descriptor.edges.map((edge) =>
    buildConnectionJsonEntry(edge, moduleIdToNodeIndex),
  );

  // Step 4: Assemble the NetworkJSON intermediate and attach the NGE extension bag
  // only when the caller has opted in via runtimeHints.ngeEnabled.
  const networkJson: NetworkJSON = {
    formatVersion: NETWORK_JSON_FORMAT_VERSION,
    input: nodes.filter((node) => node.type === 'input').length,
    output: nodes.filter((node) => node.type === 'output').length,
    dropout: DEFAULT_DROPOUT,
    nodes,
    connections,
  };
  if (runtimeHints.ngeEnabled === true) {
    networkJson.extensions = buildNgeExtensions(envelope, descriptor);
  }

  // Step 5: Hydrate the runtime Network through the canonical serializer.
  return Network.fromJSON(networkJson as unknown as Record<string, unknown>);
}

/**
 * Extract the canonical NGE DNA envelope previously attached to a runtime Network.
 *
 * This is the reverse direction of the canonical envelope bridge: it reads the
 * NGE extension carrier from the network's serialized extension bag and returns a
 * structured clone so callers cannot mutate the stored envelope. The envelope's
 * `schemaVersion` and `fingerprint` survive the round-trip unchanged, which means
 * a `materialize → extract` cycle is a lossless identity-preserving operation.
 *
 * The function throws when the Network carries no NGE extension bag — for
 * example, when the network was materialized without `ngeEnabled: true`, or when
 * it is a classic-NEAT network that was never touched by the NGE pipeline.
 *
 * @param network - Runtime Network produced by {@link materializeNetworkFromPhenotype}
 * with `ngeEnabled: true`, or any Network whose serialized extension bag contains
 * an `ngeEnvelope` carrier.
 * @returns A structured clone of the stored canonical NGE DNA envelope.
 * @throws {NGE_DNA_BridgeError} when the Network carries no NGE extension bag.
 *
 * @example
 * ```ts
 * const network = materializeNetworkFromPhenotype(envelope, plan, descriptor, {
 *   ngeEnabled: true,
 * });
 * const extracted = extractCanonicalEnvelopeFromNetwork(network);
 * console.log(extracted.fingerprint === envelope.fingerprint); // true
 * ```
 *
 * @example
 * ```ts
 * // Classic-NEAT network without NGE extension → throws.
 * const plainNet = materializeNetworkFromPhenotype(envelope, plan, descriptor);
 * try {
 *   extractCanonicalEnvelopeFromNetwork(plainNet);
 * } catch (err) {
 *   console.log(err instanceof NGE_DNA_BridgeError); // true
 * }
 * ```
 */
export function extractCanonicalEnvelopeFromNetwork(
  network: Network,
): NgeDnaCanonicalEnvelope {
  const json = network.toJSON() as unknown as NetworkJSON;
  const values = json.extensions?.values as
    NgeBridgeExtensionValues | undefined;
  const ngeEnvelope = values?.ngeEnvelope;
  if (!ngeEnvelope) {
    throw new NGE_DNA_BridgeError(
      'The Network carries no NGE canonical envelope extension to extract.',
    );
  }
  return structuredClone(ngeEnvelope);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers below the fold
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build one {@link NetworkJSONNode} from a realized module.
 * The `geneId` is the deterministic 1-based module index so round-trips stay stable.
 */
function buildNodeJsonEntry(
  module: NgeRealizedModule,
  index: number,
  runtimeHints: GenomeMaterializationRuntimeHints,
): NetworkJSONNode {
  return {
    type: resolveNodeType(module.coordinate[2]),
    bias: DEFAULT_NODE_BIAS,
    squash: resolveSquash(module.computationType, runtimeHints),
    index,
    geneId: index + 1,
  };
}

/** Resolve the runtime node type from the unit-cube zone coordinate. */
function resolveNodeType(zoneCoordinate: number): string {
  if (zoneCoordinate === INPUT_ZONE_COORDINATE) {
    return 'input';
  }
  if (zoneCoordinate === OUTPUT_ZONE_COORDINATE) {
    return 'output';
  }
  return 'hidden';
}

/** Resolve the squash function name from the computation motif and runtime hints. */
function resolveSquash(
  computationType: NeatGenomeComputationType,
  runtimeHints: GenomeMaterializationRuntimeHints,
): string {
  if (runtimeHints.ngeEnabled !== true) {
    return SQUASH_WHEN_NGE_DISABLED;
  }
  return COMPUTATION_TYPE_TO_SQUASH[computationType];
}

/**
 * Build one {@link NetworkJSONConnection} from a realized edge.
 * The `innovation` is a deterministic positive hash of the module-id pair so the
 * same adjacency always restores the same historical identity.
 */
function buildConnectionJsonEntry(
  edge: NgeRealizedEdge,
  moduleIdToNodeIndex: ModuleIdToNodeIndex,
): NetworkJSONConnection {
  return {
    from: moduleIdToNodeIndex.get(edge.sourceModuleId) as number,
    to: moduleIdToNodeIndex.get(edge.targetModuleId) as number,
    weight: edge.weight,
    gain: DEFAULT_CONNECTION_GAIN,
    gater: NEUTRAL_GATER,
    enabled: DEFAULT_CONNECTION_ENABLED,
    innovation: computeStableInnovation(
      edge.sourceModuleId,
      edge.targetModuleId,
    ),
  };
}

/**
 * Compute a deterministic positive integer hash from a source/target module-id pair.
 * Uses a djb2-style fold so identical adjacencies always yield the same innovation
 * number across round-trips, independent of global counter state.
 */
function computeStableInnovation(
  sourceModuleId: string,
  targetModuleId: string,
): number {
  const key = `${sourceModuleId}|${targetModuleId}`;
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = ((hash << DETERMINISTIC_HASH_OFFSET) - hash + key.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) + DETERMINISTIC_HASH_POSITIVE_FLOOR;
}

/** Build the NGE extension bag carried inside `NetworkJSONExtensions`. */
function buildNgeExtensions(
  envelope: NgeDnaCanonicalEnvelope,
  descriptor: NgeRealizedPhenotypeDescriptor,
): NetworkJSONExtensions {
  return {
    version: BRIDGE_EXTENSION_VERSION,
    values: {
      version: BRIDGE_EXTENSION_SCHEMA_VERSION,
      ngeDescriptor: structuredClone(descriptor),
      ngeEnvelope: structuredClone(envelope),
    },
  };
}
