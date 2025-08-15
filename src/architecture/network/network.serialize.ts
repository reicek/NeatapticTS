import type Network from '../network';
import Node from '../node';
import Connection from '../connection';
import * as methods from '../../methods/methods';

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
export function serialize(this: Network): any[] {
  // Ensure indices are refreshed (fast paths may leave stale indices for performance; we enforce consistency here).
  (this as any).nodes.forEach(
    (nodeRef: any, nodeIndex: number) => (nodeRef.index = nodeIndex)
  );
  // At this point each node.index becomes our canonical ID used throughout the serialization.
  // Indices are intentionally positional so the resulting arrays remain tightly packed and cache‑friendly.
  /** Current activation values per node (index-aligned). */
  const activations = (this as any).nodes.map(
    (nodeRef: any) => nodeRef.activation
  );
  // activations[] captures the post-squash output of each neuron; when deserialized we can resume
  // a simulation mid-stream (e.g. during evolutionary evaluation) if desired.
  /** Current membrane/accumulator state per node. */
  const states = (this as any).nodes.map((nodeRef: any) => nodeRef.state);
  // states[] represent the pre-activation internal sum (or evolving state for recurrent / gated constructs).
  /** Squash (activation function) names per node for later rehydration. */
  const squashes = (this as any).nodes.map(
    (nodeRef: any) => nodeRef.squash.name
  );
  // Instead of serializing function references we store the human-readable name; on import we map name->fn.
  /** Combined forward + self connections flattened to plain indices + weights. */
  const serializedConnections = (this as any).connections
    .concat((this as any).selfconns)
    .map((connInstance: any) => ({
      from: connInstance.from.index,
      to: connInstance.to.index,
      weight: connInstance.weight,
      gater: connInstance.gater ? connInstance.gater.index : null,
    }));
  // A single linear pass is used; order of connections is not semantically important because reconstruction
  // will look up by (from,to) pairs. Self connections are treated uniformly (from === to) for simplicity.
  /** Input layer size captured for reconstruction. */
  const inputSize = (this as any).input;
  /** Output layer size captured for reconstruction. */
  const outputSize = (this as any).output;
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
export function deserialize(
  data: any[],
  inputSize?: number,
  outputSize?: number
): Network {
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
  /** Newly constructed network shell with IO sizes. */
  const net = new (require('../network').default)(input, output) as Network; // dynamic require to avoid circular dependency timing
  (net as any).nodes = [];
  (net as any).connections = [];
  (net as any).selfconns = [];
  (net as any).gates = [];
  // Phase 1: Recreate nodes in positional order. We intentionally rebuild even input/output nodes so that
  // any evolution-time modifications (bias, activation) are preserved.
  activations.forEach((activation: number, nodeIndex: number) => {
    /** Node type derived from index relative to IO spans. */
    let type: string;
    if (nodeIndex < input) type = 'input';
    else if (nodeIndex >= (activations as any).length - output) type = 'output';
    else type = 'hidden';
    /** Rehydrated node instance. */
    const node: any = new Node(type);
    node.activation = activation;
    node.state = states[nodeIndex];
    /** Activation function name captured during serialization. */
    const squashName = squashes[nodeIndex] as keyof typeof methods.Activation;
    if (!(methods.Activation as any)[squashName]) {
      console.warn(
        `Unknown squash function '${String(
          squashName
        )}' encountered during deserialize. Falling back to identity.`
      );
    }
    node.squash =
      (methods.Activation as any)[squashName] || methods.Activation.identity;
    node.index = nodeIndex;
    (net as any).nodes.push(node);
  });
  // Phase 2: Recreate connections. We iterate the flat connection list and re-establish edges using indices.
  // Self connections are seamlessly handled when from === to. Gating is re-applied after connection creation.
  connections.forEach((serializedConn: any) => {
    if (
      serializedConn.from < (net as any).nodes.length &&
      serializedConn.to < (net as any).nodes.length
    ) {
      /** Source node for reconstructed connection. */
      const sourceNode = (net as any).nodes[serializedConn.from];
      /** Target node for reconstructed connection. */
      const targetNode = (net as any).nodes[serializedConn.to];
      /** Newly created connection (array return from connect). */
      const createdConnection = (net as any).connect(
        sourceNode,
        targetNode,
        serializedConn.weight
      )[0];
      if (createdConnection && serializedConn.gater != null) {
        if (serializedConn.gater < (net as any).nodes.length) {
          // Only gate if the gater index is valid—defensive against older or pruned models.
          (net as any).gate(
            (net as any).nodes[serializedConn.gater],
            createdConnection
          );
        } else {
          console.warn(
            'Invalid gater index encountered during deserialize; skipping gater assignment.'
          );
        }
      }
    } else {
      console.warn(
        'Invalid connection indices encountered during deserialize; skipping connection.'
      );
    }
  });
  // Note: We intentionally do NOT rebuild any cached topological ordering here; callers invoking activation
  // or mutation operations will trigger those lazy recomputations.
  return net;
}

/**
 * Verbose JSON export (stable formatVersion). Omits transient runtime fields but keeps structural genetics.
 * formatVersion=2 adds: enabled flags, stable geneId (if present), dropout value.
 */
export function toJSONImpl(this: Network): object {
  /** Accumulated verbose JSON representation (formatVersion = 2). */
  const json: any = {
    formatVersion: 2,
    input: (this as any).input,
    output: (this as any).output,
    dropout: (this as any).dropout,
    nodes: [],
    connections: [],
  };
  // Node pass: capture minimal structural genetics (bias, activation, geneId) but exclude transient runtime state.
  (this as any).nodes.forEach((node: any, nodeIndex: number) => {
    node.index = nodeIndex; // refresh index for safety
    json.nodes.push({
      type: node.type,
      bias: node.bias,
      squash: node.squash.name,
      index: nodeIndex,
      geneId: (node as any).geneId,
    });
    if (node.connections.self.length > 0) {
      /** Self connection reference (at most one). */
      const selfConn = node.connections.self[0];
      json.connections.push({
        from: nodeIndex,
        to: nodeIndex,
        weight: selfConn.weight,
        gater: selfConn.gater ? selfConn.gater.index : null,
        enabled: (selfConn as any).enabled !== false,
      });
    }
  });
  // Connection pass: append forward connections preserving enabled state & gating relationships.
  (this as any).connections.forEach((connInstance: any) => {
    if (
      typeof connInstance.from.index !== 'number' ||
      typeof connInstance.to.index !== 'number'
    )
      return;
    json.connections.push({
      from: connInstance.from.index,
      to: connInstance.to.index,
      weight: connInstance.weight,
      gater: connInstance.gater ? connInstance.gater.index : null,
      enabled: (connInstance as any).enabled !== false,
    });
  });
  // The resulting JSON is stable: ordering of nodes is deterministic, and connections list order derives from existing array ordering.
  return json;
}

/**
 * Reconstruct a Network from the verbose JSON produced by {@link toJSONImpl} (formatVersion 2).
 * Defensive parsing retains forward compatibility (warns on unknown versions rather than aborting).
 */
export function fromJSONImpl(json: any): Network {
  if (!json || typeof json !== 'object')
    throw new Error('Invalid JSON for network.');
  if (json.formatVersion !== 2)
    console.warn('fromJSONImpl: Unknown formatVersion, attempting import.');
  /** New network shell with recorded IO sizes. */
  const net = new (require('../network').default)(
    json.input,
    json.output
  ) as Network;
  (net as any).dropout = json.dropout || 0;
  (net as any).nodes = [];
  (net as any).connections = [];
  (net as any).selfconns = [];
  (net as any).gates = [];
  // Rebuild nodes first so that index-based connection references become valid.
  json.nodes.forEach((nodeJson: any, nodeIndex: number) => {
    /** Rehydrated node from JSON. */
    const node: any = new Node(nodeJson.type);
    node.bias = nodeJson.bias;
    node.squash =
      (methods.Activation as any)[nodeJson.squash] ||
      methods.Activation.identity;
    node.index = nodeIndex;
    if (typeof nodeJson.geneId === 'number')
      (node as any).geneId = nodeJson.geneId;
    (net as any).nodes.push(node);
  });
  // Then recreate connections, applying gating and enabled status (innovation tracking) if present.
  json.connections.forEach((connJson: any) => {
    if (typeof connJson.from !== 'number' || typeof connJson.to !== 'number')
      return;
    /** Source node for connection gene. */
    const sourceNode = (net as any).nodes[connJson.from];
    /** Destination node for connection gene. */
    const targetNode = (net as any).nodes[connJson.to];
    /** Newly established connection instance. */
    const createdConnection = (net as any).connect(
      sourceNode,
      targetNode,
      connJson.weight
    )[0];
    if (
      createdConnection &&
      connJson.gater != null &&
      typeof connJson.gater === 'number' &&
      (net as any).nodes[connJson.gater]
    ) {
      (net as any).gate((net as any).nodes[connJson.gater], createdConnection);
    }
    if (createdConnection && typeof connJson.enabled !== 'undefined')
      (createdConnection as any).enabled = connJson.enabled;
  });
  // As with deserialize(), we defer recalculating any cached orderings until first operational use.
  return net;
}

export { Connection }; // re-export for potential external tooling needing innovation IDs
