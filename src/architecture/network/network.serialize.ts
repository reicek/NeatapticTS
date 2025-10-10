import type Network from '../network';
import Node from '../node';
import Connection from '../connection';
import * as methods from '../../methods/methods';

/**
 * Runtime interface for accessing Network internal properties.
 */
interface NetworkInternals {
  nodes: Node[];
  connections: Connection[];
  selfconns: Connection[];
  gates: Connection[];
  input: number;
  output: number;
  connect: (from: Node, to: Node, weight: number) => Connection[];
  gate: (gater: Node, connection: Connection) => void;
}

/**
 * Runtime interface for accessing Node internal properties.
 */
interface NodeInternals {
  index: number;
  activation: number;
  state: number;
  squash: ((x: number, derivate?: boolean) => number) & { name: string };
}

/**
 * Serialized connection representation.
 */
export interface SerializedConnection {
  from: number;
  to: number;
  weight: number;
  gater: number | null;
}

/**
 * Verbose JSON format for network serialization.
 */
interface NetworkJSON {
  formatVersion: number;
  input: number;
  output: number;
  dropout: number;
  nodes: {
    type: string;
    bias: number;
    squash: string;
    index: number;
    geneId?: number;
  }[];
  connections: {
    from: number;
    to: number;
    weight: number;
    gater: number | null;
    enabled: boolean;
  }[];
}

/**
 * Serialization & deserialization helpers for Network instances.
 *
 * Provides two independent formats:
 *  1. Compact tuple (serialize/deserialize): optimized for fast structured clone / worker transfer.
 *  2. Verbose JSON (toJSONImpl/fromJSONImpl): stable, versioned representation retaining structural genes.
 *
 * Compact tuple format layout:
 *  [ activations: number[], states: number[], squashes: string[],
 *    connections: { from:number; to:number; weight:number; gater:number|null }[],
 *    inputSize: number, outputSize: number ]
 *
 * Design Principles:
 *  - Avoid deep nested objects to reduce serialization overhead.
 *  - Use current node ordering as canonical index mapping (caller must keep ordering stable between peers).
 *  - Include current activation/state for scenarios resuming partially evaluated populations.
 *  - Self connections placed in the same array as normal connections for uniform reconstruction.
 *
 * Verbose JSON (formatVersion = 2) adds:
 *  - Enabled flag for connections (innovation toggling).
 *  - Stable geneId (if tracked) on nodes.
 *  - Dropout probability.
 *
 * Future Ideas:
 *  - Delta / patch serialization for large evolving populations.
 *  - Compressed binary packing (e.g., Float32Array segments) for WASM pipelines.
 */

/**
 * Instance-level lightweight serializer used primarily for fast inter-thread (WebWorker) transfer.
 * Produces a compact tuple style array instead of a verbose object graph.
 *
 * Layout:
 *  [ activations: number[], states: number[], squashes: string[],
 *    connections: { from:number; to:number; weight:number; gater:number|null }[],
 *    inputSize: number, outputSize: number ]
 *
 * Design notes:
 *  - Only minimal dynamic runtime values are captured (activation/state and current squash fn name).
 *  - Self connections are appended alongside normal connections (caller rehydrates uniformly).
 *  - Indices are derived from current node ordering; caller must ensure consistent ordering across workers.
 */
export function serialize(
  this: Network,
): [number[], number[], string[], SerializedConnection[], number, number] {
  const networkInternal = this as unknown as NetworkInternals;
  // Ensure indices are refreshed (fast paths may leave stale indices for performance; we enforce consistency here).
  networkInternal.nodes.forEach((nodeRef, nodeIndex: number) => {
    const nodeInternal = nodeRef as unknown as NodeInternals;
    nodeInternal.index = nodeIndex;
  });
  // At this point each node.index becomes our canonical ID used throughout the serialization.
  // Indices are intentionally positional so the resulting arrays remain tightly packed and cache‑friendly.
  /** Current activation values per node (index-aligned). */
  const activations = networkInternal.nodes.map(
    (nodeRef) => (nodeRef as unknown as NodeInternals).activation,
  );
  // activations[] captures the post-squash output of each neuron; when deserialized we can resume
  // a simulation mid-stream (e.g. during evolutionary evaluation) if desired.
  /** Current membrane/accumulator state per node. */
  const states = networkInternal.nodes.map(
    (nodeRef) => (nodeRef as unknown as NodeInternals).state,
  );
  // states[] represent the pre-activation internal sum (or evolving state for recurrent / gated constructs).
  /** Squash (activation function) names per node for later rehydration. */
  const squashes = networkInternal.nodes.map(
    (nodeRef) => (nodeRef as unknown as NodeInternals).squash.name,
  );
  // Instead of serializing function references we store the human-readable name; on import we map name->fn.
  /** Combined forward + self connections flattened to plain indices + weights. */
  const serializedConnections = networkInternal.connections
    .concat(networkInternal.selfconns)
    .map(
      (connInstance): SerializedConnection => ({
        from: (connInstance.from as unknown as NodeInternals).index,
        to: (connInstance.to as unknown as NodeInternals).index,
        weight: connInstance.weight,
        gater: connInstance.gater
          ? (connInstance.gater as unknown as NodeInternals).index
          : null,
      }),
    );
  // A single linear pass is used; order of connections is not semantically important because reconstruction
  // will look up by (from,to) pairs. Self connections are treated uniformly (from === to) for simplicity.
  /** Input layer size captured for reconstruction. */
  const inputSize = networkInternal.input;
  /** Output layer size captured for reconstruction. */
  const outputSize = networkInternal.output;
  // We intentionally return a plain Array rather than an object literal to minimize JSON overhead and
  // reduce property name duplication during stringify/structuredClone operations.
  return [
    activations,
    states,
    squashes,
    serializedConnections,
    inputSize,
    outputSize,
  ];
}

/**
 * Static counterpart to {@link serialize}. Rebuilds a Network from the compact tuple form.
 * Accepts optional explicit input/output size overrides (useful when piping through evolvers that trim IO).
 */
export const deserialize = (
  data: [number[], number[], string[], SerializedConnection[], number, number],
  inputSize?: number,
  outputSize?: number,
): Network => {
  /** Destructured compact tuple payload produced by serialize(). */
  const [
    activations,
    states,
    squashes,
    connections,
    serializedInput,
    serializedOutput,
  ] = data;
  /** Effective input size (override takes precedence). */
  const input =
    typeof inputSize === 'number' ? inputSize : serializedInput || 0;
  /** Effective output size (override takes precedence). */
  const output =
    typeof outputSize === 'number' ? outputSize : serializedOutput || 0;
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- Dynamic require needed to avoid circular dependency
  const { default: NetworkConstructor } = require('../network');
  /** Newly constructed network shell with IO sizes. */
  const net = new NetworkConstructor(input, output) as Network;
  const netInternal = net as unknown as NetworkInternals;
  netInternal.nodes = [];
  netInternal.connections = [];
  netInternal.selfconns = [];
  netInternal.gates = [];
  // Phase 1: Recreate nodes in positional order. We intentionally rebuild even input/output nodes so that
  // any evolution-time modifications (bias, activation) are preserved.
  activations.forEach((activation: number, nodeIndex: number) => {
    /** Node type derived from index relative to IO spans. */
    let type: string;
    if (nodeIndex < input) type = 'input';
    else if (nodeIndex >= activations.length - output) type = 'output';
    else type = 'hidden';
    /** Rehydrated node instance. */
    const node = new Node(type);
    const nodeInternal = node as unknown as NodeInternals;
    nodeInternal.activation = activation;
    nodeInternal.state = states[nodeIndex];
    /** Activation function name captured during serialization. */
    const squashName = squashes[nodeIndex] as keyof typeof methods.Activation;
    if (!methods.Activation[squashName]) {
      console.warn(
        `Unknown squash function '${String(
          squashName,
        )}' encountered during deserialize. Falling back to identity.`,
      );
    }
    nodeInternal.squash =
      methods.Activation[squashName] || methods.Activation.identity;
    nodeInternal.index = nodeIndex;
    netInternal.nodes.push(node);
  });
  // Phase 2: Recreate connections. We iterate the flat connection list and re-establish edges using indices.
  // Self connections are seamlessly handled when from === to. Gating is re-applied after connection creation.
  connections.forEach((serializedConn) => {
    if (
      serializedConn.from < netInternal.nodes.length &&
      serializedConn.to < netInternal.nodes.length
    ) {
      /** Source node for reconstructed connection. */
      const sourceNode = netInternal.nodes[serializedConn.from];
      /** Target node for reconstructed connection. */
      const targetNode = netInternal.nodes[serializedConn.to];
      /** Newly created connection (array return from connect). */
      const createdConnection = netInternal.connect(
        sourceNode,
        targetNode,
        serializedConn.weight,
      )[0];
      if (createdConnection && serializedConn.gater != null) {
        if (serializedConn.gater < netInternal.nodes.length) {
          // Only gate if the gater index is valid—defensive against older or pruned models.
          netInternal.gate(
            netInternal.nodes[serializedConn.gater],
            createdConnection,
          );
        } else {
          console.warn(
            'Invalid gater index encountered during deserialize; skipping gater assignment.',
          );
        }
      }
    } else {
      console.warn(
        'Invalid connection indices encountered during deserialize; skipping connection.',
      );
    }
  });
  // Note: We intentionally do NOT rebuild any cached topological ordering here; callers invoking activation
  // or mutation operations will trigger those lazy recomputations.
  return net;
};

/**
 * Verbose JSON export (stable formatVersion). Omits transient runtime fields but keeps structural genetics.
 * formatVersion=2 adds: enabled flags, stable geneId (if present), dropout value.
 */
export function toJSONImpl(this: Network): NetworkJSON {
  const networkInternal = this as unknown as NetworkInternals & {
    dropout?: number;
  };
  /** Accumulated verbose JSON representation (formatVersion = 2). */
  const json: NetworkJSON = {
    formatVersion: 2,
    input: networkInternal.input,
    output: networkInternal.output,
    dropout: networkInternal.dropout || 0,
    nodes: [],
    connections: [],
  };
  // Node pass: capture minimal structural genetics (bias, activation, geneId) but exclude transient runtime state.
  networkInternal.nodes.forEach((node, nodeIndex: number) => {
    const nodeInternal = node as unknown as NodeInternals & {
      bias: number;
      geneId?: number;
      connections: { self: Connection[] };
    };
    nodeInternal.index = nodeIndex; // refresh index for safety
    json.nodes.push({
      type: node.type,
      bias: nodeInternal.bias,
      squash: nodeInternal.squash.name,
      index: nodeIndex,
      geneId: nodeInternal.geneId,
    });
    if (nodeInternal.connections.self.length > 0) {
      /** Self connection reference (at most one). */
      const selfConn = nodeInternal.connections.self[0];
      const selfConnInternal = selfConn as Connection & { enabled?: boolean };
      json.connections.push({
        from: nodeIndex,
        to: nodeIndex,
        weight: selfConn.weight,
        gater: selfConn.gater
          ? (selfConn.gater as unknown as NodeInternals).index
          : null,
        enabled: selfConnInternal.enabled !== false,
      });
    }
  });
  // Connection pass: append forward connections preserving enabled state & gating relationships.
  networkInternal.connections.forEach((connInstance) => {
    const fromInternal = connInstance.from as unknown as NodeInternals;
    const toInternal = connInstance.to as unknown as NodeInternals;
    if (
      typeof fromInternal.index !== 'number' ||
      typeof toInternal.index !== 'number'
    )
      return;
    const connInternal = connInstance as Connection & { enabled?: boolean };
    json.connections.push({
      from: fromInternal.index,
      to: toInternal.index,
      weight: connInstance.weight,
      gater: connInstance.gater
        ? (connInstance.gater as unknown as NodeInternals).index
        : null,
      enabled: connInternal.enabled !== false,
    });
  });
  // The resulting JSON is stable: ordering of nodes is deterministic, and connections list order derives from existing array ordering.
  return json;
}

/**
 * Reconstruct a Network from the verbose JSON produced by {@link toJSONImpl} (formatVersion 2).
 * Defensive parsing retains forward compatibility (warns on unknown versions rather than aborting).
 */
export const fromJSONImpl = (json: NetworkJSON): Network => {
  if (!json || typeof json !== 'object')
    throw new Error('Invalid JSON for network.');
  if (json.formatVersion !== 2)
    console.warn('fromJSONImpl: Unknown formatVersion, attempting import.');
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- Dynamic require needed to avoid circular dependency
  const { default: NetworkConstructor } = require('../network');
  /** New network shell with recorded IO sizes. */
  const net = new NetworkConstructor(json.input, json.output) as Network;
  const netInternal = net as unknown as NetworkInternals & { dropout?: number };
  netInternal.dropout = json.dropout || 0;
  netInternal.nodes = [];
  netInternal.connections = [];
  netInternal.selfconns = [];
  netInternal.gates = [];
  // Rebuild nodes first so that index-based connection references become valid.
  json.nodes.forEach((nodeJson, nodeIndex: number) => {
    /** Rehydrated node from JSON. */
    const node = new Node(nodeJson.type);
    const nodeInternal = node as unknown as NodeInternals & {
      bias: number;
      geneId?: number;
    };
    nodeInternal.bias = nodeJson.bias;
    const squashName = nodeJson.squash as keyof typeof methods.Activation;
    nodeInternal.squash =
      methods.Activation[squashName] || methods.Activation.identity;
    nodeInternal.index = nodeIndex;
    if (typeof nodeJson.geneId === 'number')
      nodeInternal.geneId = nodeJson.geneId;
    netInternal.nodes.push(node);
  });
  // Then recreate connections, applying gating and enabled status (innovation tracking) if present.
  json.connections.forEach((connJson) => {
    if (typeof connJson.from !== 'number' || typeof connJson.to !== 'number')
      return;

    // Defensive bounds check: ensure indices reference existing nodes before attempting to connect.
    const nodesLength = netInternal.nodes.length;
    if (
      connJson.from < 0 ||
      connJson.to < 0 ||
      connJson.from >= nodesLength ||
      connJson.to >= nodesLength
    ) {
      console.warn(
        'Invalid connection indices encountered during fromJSONImpl; skipping connection.',
      );
      return;
    }

    /** Source node for connection gene. */
    const sourceNode = netInternal.nodes[connJson.from];
    /** Destination node for connection gene. */
    const targetNode = netInternal.nodes[connJson.to];
    /** Newly established connection instance. */
    const createdConnection = netInternal.connect(
      sourceNode,
      targetNode,
      connJson.weight,
    )[0];
    if (
      createdConnection &&
      connJson.gater != null &&
      typeof connJson.gater === 'number'
    ) {
      if (connJson.gater >= 0 && connJson.gater < nodesLength) {
        netInternal.gate(netInternal.nodes[connJson.gater], createdConnection);
      } else {
        console.warn(
          'Invalid gater index encountered during fromJSONImpl; skipping gater assignment.',
        );
      }
    }
    if (createdConnection && typeof connJson.enabled !== 'undefined') {
      const connInternal = createdConnection as Connection & {
        enabled?: boolean;
      };
      connInternal.enabled = connJson.enabled;
    }
  });
  // As with deserialize(), we defer recalculating any cached orderings until first operational use.
  return net;
};

export { Connection }; // re-export for potential external tooling needing innovation IDs
