import type { NeatGenomeComputationType } from '../genome/genome.types';
import type {
  NgeMainAgentAdult,
  NgeMainAgentEmbryo,
  NgeMainAgentJuvenile,
  NgeMainAgentReproducing,
} from './neat.nge-main-agent.types';

import Connection from '../../architecture/connection';
import Node from '../../architecture/node/node';
import Network from '../../architecture/network/network';

/**
 * Computation motifs that introduce recurrence (self-feedback or cyclic wiring)
 * in the materialized network topology.
 *
 * When at least one archetype in a lifecycle state carries one of these motifs,
 * the bridge guarantees at least one self-connection appears in the output
 * `network.selfconns` array.
 */
const RECURRENT_MOTIF_TYPES: ReadonlySet<NeatGenomeComputationType> = new Set([
  'GatedRecurrentCell',
  'EpisodicSlot',
]);

/**
 * Fixed weight assigned to every materialized connection.
 *
 * A constant weight keeps the bridge fully deterministic without relying on
 * `Math.random`. Later NGE evolution phases mutate these weights; the
 * materialization bridge only needs to produce a structurally correct graph.
 */
const MATERIALIZE_CONNECTION_WEIGHT = 0.5;

const INPUT_NODE_LABEL_PREFIX = 'materialize-input-';
const OUTPUT_NODE_LABEL_PREFIX = 'materialize-output-';
const DEFAULT_INPUT_COUNT = 1;
const DEFAULT_OUTPUT_COUNT = 1;

/**
 * Optional I/O configuration for {@link materializeFromNgeState}.
 *
 * Allows the caller to specify how many of the materialized nodes should be
 * designated as input and output nodes, making the bridge a drop-in replacement
 * for `new Network(inputCount, outputCount)`.
 */
export interface MaterializeOptions {
  /** Number of input nodes (the first `inputCount` nodes). Defaults to 1. */
  inputCount?: number;
  /** Number of output nodes (the last `outputCount` nodes). Defaults to 1. */
  outputCount?: number;
}

/**
 * Union of every NGE main-agent lifecycle stage that carries enough topology
 * metadata to materialize a `Network`.
 *
 * All four stages share `nodeCount`, `edgeCount`, `seed`, and `archetypes`
 * (each with a `computationType`), which are the only fields the bridge reads.
 */
export type NgeMaterializableState =
  | NgeMainAgentEmbryo
  | NgeMainAgentJuvenile
  | NgeMainAgentAdult
  | NgeMainAgentReproducing;

/**
 * Materialize a live {@link Network} runtime from an NGE main-agent lifecycle state.
 *
 * The bridge reads the deterministic topology counts (`nodeCount`, `edgeCount`),
 * motif archetypes, and seed from any lifecycle stage — embryo, juvenile, adult,
 * or reproducing — and produces a fully wired `Network` instance whose:
 *
 * - `nodes.length` equals `state.nodeCount`,
 * - `connections.length + selfconns.length` equals `state.edgeCount`
 *   (self-connections remain in `selfconns`, matching the `Network.construct`
 *   invariant), and
 * - `connections` or `selfconns` contains at least one self-connection when the
 *   state's archetypes include a recurrent motif (`GatedRecurrentCell` or
 *   `EpisodicSlot`).
 *
 * The same state always produces the same node and connection counts, and the
 * same innovation IDs, making the bridge reproducible across calls.
 *
 * @param state - Any materializable NGE main-agent lifecycle state.
 * @param options - Optional I/O configuration. When omitted the bridge defaults
 *   to 1 input and 1 output node (the first and last nodes respectively). Pass
 *   `{ inputCount, outputCount }` to produce a network with the correct number
 *   of input and output nodes — e.g. to be a drop-in replacement for
 *   `new Network(22, 5)`.
 * @returns A `Network` instance wired with the state's topology.
 *
 * @example
 * ```ts
 * const embryo = buildMainAgentEmbryo({ seed: 42, maxNodes: 1024, maxEdges: 4096 });
 * const network = materializeFromNgeState(embryo);
 * console.log(network.nodes.length);       // 3
 * console.log(network.connections.length); // non-self edges
 * console.log(network.selfconns.length);   // self-edges
 * ```
 */
export function materializeFromNgeState(
  state: NgeMaterializableState,
  options?: MaterializeOptions,
): Network {
  const inputCount = options?.inputCount ?? DEFAULT_INPUT_COUNT;
  const outputCount = options?.outputCount ?? DEFAULT_OUTPUT_COUNT;
  const nodes = createMaterializationNodes(
    state.nodeCount,
    state.seed,
    inputCount,
    outputCount,
  );
  const hasRecurrentMotif = state.archetypes.some((archetype) =>
    RECURRENT_MOTIF_TYPES.has(archetype.computationType),
  );
  wireMaterializationEdges(
    nodes,
    state.edgeCount,
    hasRecurrentMotif,
    inputCount,
  );
  return assembleNetworkFromParts(nodes, state.seed);
}

// --- Helpers ----------------------------------------------------------------

/**
 * Create `nodeCount` deterministic `Node` instances with role-based labels.
 *
 * The first `inputCount` nodes are typed `'input'`, the last `outputCount`
 * nodes `'output'`, and every interior node `'hidden'`. A seeded PRNG supplies
 * bias values so the same seed always produces the same initial biases.
 *
 * @param nodeCount - Total nodes to create (must be ≥ 2 for a valid I/O pair).
 * @param seed - Determinism seed forwarded to the PRNG.
 * @param inputCount - Number of input nodes (first `inputCount` indices).
 * @param outputCount - Number of output nodes (last `outputCount` indices).
 * @returns An ordered array of `Node` instances.
 */
function createMaterializationNodes(
  nodeCount: number,
  seed: number,
  inputCount: number,
  outputCount: number,
): Node[] {
  const rng = createSeededRng(seed);
  const nodes: Node[] = [];

  for (let index = 0; index < nodeCount; index++) {
    const role = resolveNodeRole(index, nodeCount, inputCount, outputCount);
    const node = new Node(role, undefined, rng);
    const label = resolveNodeLabel(role, index);
    if (label !== null) {
      node.describe({ label });
    }
    nodes.push(node);
  }

  return nodes;
}

/**
 * Resolve the runtime role string for a node at a given index.
 *
 * The first `inputCount` nodes are inputs and the last `outputCount` nodes are
 * outputs. When the two ranges would overlap (`inputCount + outputCount >
 * nodeCount`) the output range takes precedence so every requested output node
 * is guaranteed a slot.
 *
 * @param index - Zero-based node position.
 * @param nodeCount - Total node count in the materialization set.
 * @param inputCount - Number of input nodes.
 * @param outputCount - Number of output nodes.
 * @returns `'input'`, `'output'`, or `'hidden'`.
 */
function resolveNodeRole(
  index: number,
  nodeCount: number,
  inputCount: number,
  outputCount: number,
): string {
  const outputStart = Math.max(inputCount, nodeCount - outputCount);
  if (index < inputCount && index < outputStart) {
    return 'input';
  }
  if (index >= outputStart) {
    return 'output';
  }
  return 'hidden';
}

/**
 * Resolve a human-readable label for a node so it can be referenced by string
 * id in `Network.construct`.
 *
 * @param role - Runtime role assigned to the node.
 * @param index - Zero-based node position.
 * @returns A label string for input/output nodes, or `null` for hidden nodes.
 */
function resolveNodeLabel(role: string, index: number): string | null {
  if (role === 'input') {
    return `${INPUT_NODE_LABEL_PREFIX}${index}`;
  }
  if (role === 'output') {
    return `${OUTPUT_NODE_LABEL_PREFIX}${index}`;
  }
  return null;
}

/**
 * Wire `edgeCount` deterministic connections between the provided nodes.
 *
 * Candidate edges are enumerated in a stable order. When `hasRecurrentMotif`
 * is true, self-connections are prioritized so at least one recurrent edge is
 * selected (provided `edgeCount` ≥ 1). Otherwise, non-self edges come first and
 * self-connections fill remaining slots only when needed.
 *
 * The first `inputCount` nodes are never targets of any edge, preserving the
 * pure-source invariant required by `Network.construct`.
 *
 * @param nodes - Ordered node array to wire.
 * @param edgeCount - Number of edges to create.
 * @param hasRecurrentMotif - Whether recurrent motifs are present in the state.
 * @param inputCount - Number of input nodes to protect from incoming edges.
 */
function wireMaterializationEdges(
  nodes: Node[],
  edgeCount: number,
  hasRecurrentMotif: boolean,
  inputCount: number,
): void {
  const candidates = enumerateCandidateEdges(
    nodes.length,
    hasRecurrentMotif,
    inputCount,
  );
  if (edgeCount > candidates.length) {
    throw new Error(
      `materializeFromNgeState: requested ${edgeCount} edges but only ${candidates.length} candidate edges are available for ${nodes.length} nodes`,
    );
  }
  const selected = candidates.slice(0, edgeCount);

  // Reset the global innovation counter so the same state+seed always produces
  // identical innovation IDs across calls, keeping the bridge fully reproducible.
  Connection.resetInnovationCounter(1);

  for (const [sourceIndex, targetIndex] of selected) {
    nodes[sourceIndex].connect(
      nodes[targetIndex],
      MATERIALIZE_CONNECTION_WEIGHT,
    );
  }
}

/**
 * Enumerate all candidate directed edges for `nodeCount` nodes in deterministic
 * order.
 *
 * Self-edges are listed separately from non-self edges. When `recurrentFirst`
 * is true, self-edges precede non-self edges so that a `slice(0, edgeCount)`
 * selection guarantees at least one recurrent connection.
 *
 * @param nodeCount - Total number of nodes.
 * @param recurrentFirst - Whether to prioritize self-connections.
 * @param inputCount - Number of input nodes (indices 0..inputCount-1) to
 *   protect from incoming edges.
 * @returns An ordered array of `[sourceIndex, targetIndex]` pairs.
 */
function enumerateCandidateEdges(
  nodeCount: number,
  recurrentFirst: boolean,
  inputCount: number,
): Array<[number, number]> {
  const nonSelfEdges: Array<[number, number]> = [];
  const selfEdges: Array<[number, number]> = [];

  for (let source = 0; source < nodeCount; source++) {
    // Target starts at inputCount to enforce the input-pure-source invariant:
    // the first inputCount nodes (inputs) must never receive an incoming edge.
    for (let target = inputCount; target < nodeCount; target++) {
      if (source === target) {
        selfEdges.push([source, target]);
      } else {
        nonSelfEdges.push([source, target]);
      }
    }
  }

  return recurrentFirst
    ? [...selfEdges, ...nonSelfEdges]
    : [...nonSelfEdges, ...selfEdges];
}

/**
 * Assemble a `Network` from pre-wired nodes via the construct-from-parts API.
 *
 * Input and output node labels are collected from the node array and passed to
 * `Network.construct` with `mode: 'recurrent'` and relaxed output-edge
 * validation so that output self-connections and feedback edges are permitted.
 *
 * `Network.construct` separates self-connections into `network.selfconns` and
 * non-self connections into `network.connections`. The bridge preserves that
 * invariant — callers should inspect both arrays. The total edge count is
 * `network.connections.length + network.selfconns.length`, matching the bridge
 * contract that this sum equals `state.edgeCount`.
 *
 * @param nodes - Pre-wired node array with labels on I/O nodes.
 * @param seed - Determinism seed forwarded to the network constructor.
 * @returns A materialized `Network` instance.
 */
function assembleNetworkFromParts(nodes: Node[], seed: number): Network {
  const inputLabels = nodes
    .filter((node) => node.type === 'input')
    .map((node) => node.label!);

  const outputLabels = nodes
    .filter((node) => node.type === 'output')
    .map((node) => node.label!);

  const result = Network.construct(nodes, {
    inputNodes: inputLabels,
    outputNodes: outputLabels,
    mode: 'recurrent',
    allowIsolatedHiddenNodes: true,
    validate: { allowOutputNodeOutgoingEdges: true },
    seed,
  });

  return result.network;
}

/**
 * Create a deterministic mulberry32 PRNG from a numeric seed.
 *
 * The same seed always produces the same sequence of pseudo-random floats in
 * `[0, 1)`, which keeps node bias initialization reproducible.
 *
 * @param seed - Numeric seed (coerced to a 32-bit unsigned integer).
 * @returns A stateful RNG function returning floats in `[0, 1)`.
 */
function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let temp = state;
    temp = Math.imul(temp ^ (temp >>> 15), temp | 1);
    temp ^= temp + Math.imul(temp ^ (temp >>> 7), temp | 61);
    return ((temp ^ (temp >>> 14)) >>> 0) / 4294967296;
  };
}
