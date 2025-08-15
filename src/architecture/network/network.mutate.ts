import type Network from '../network';
import Node from '../node';
import mutation from '../../methods/mutation';
import { config } from '../../config';

/**
 * Network structural & parametric mutation utilities.
 *
 * This module exposes {@link mutateImpl} which delegates to small, focused internal helper
 * functions (one per mutation type). Extracting each case into its own function improves
 * readability, testability, and allows rich per-operator documentation.
 *
 * Mutations supported (see individual helper docs):
 *  - Topology: add/remove nodes, forward connections, backward connections, self connections.
 *  - Parameters: modify weights, biases, activations; swap node params.
 *  - Gating: add/remove gates.
 *  - Recurrent blocks: insert minimal LSTM / GRU macro-nodes.
 *
 * Internal helpers are intentionally un-exported (private to module) and are named with an
 * underscore prefix, e.g. {@link _addNode}.
 *
 * @module network.mutate
 */

/**
 * Dispatcher from mutation identity -> implementation.
 *
 * Why a map instead of a giant switch?
 *  - O(1) lookup keeps code flatter and makes tree‑shaking friendlier.
 *  - Enables meta‑programming (e.g. listing supported mutations) in tooling/docs.
 */
const MUTATION_DISPATCH: Record<
  string,
  (this: Network, method?: any) => void
> = {
  ADD_NODE: _addNode,
  SUB_NODE: _subNode,
  ADD_CONN: _addConn,
  SUB_CONN: _subConn,
  MOD_WEIGHT: _modWeight,
  MOD_BIAS: _modBias,
  MOD_ACTIVATION: _modActivation,
  ADD_SELF_CONN: _addSelfConn,
  SUB_SELF_CONN: _subSelfConn,
  ADD_GATE: _addGate,
  SUB_GATE: _subGate,
  ADD_BACK_CONN: _addBackConn,
  SUB_BACK_CONN: _subBackConn,
  SWAP_NODES: _swapNodes,
  ADD_LSTM_NODE: _addLSTMNode,
  ADD_GRU_NODE: _addGRUNode,
  REINIT_WEIGHT: _reinitWeight,
  BATCH_NORM: _batchNorm,
};

/**
 * Public entry point: apply a single mutation operator to the network.
 *
 * Steps:
 *  1. Validate the supplied method (enum value or descriptor object).
 *  2. Resolve helper implementation from the dispatch map (supports objects exposing name/type/identity).
 *  3. Invoke helper (passing through method for parameterized operators).
 *  4. Flag topology caches dirty so ordering / slabs rebuild lazily.
 *
 * Accepts either the raw enum value (e.g. `mutation.ADD_NODE`) or an object carrying an
 * identifying `name | type | identity` field allowing future parameterization without breaking call sites.
 *
 * @param this Network instance (bound).
 * @param method Mutation enum value or descriptor object.
 */
export function mutateImpl(this: Network, method: any): void {
  if (method == null) throw new Error('No (correct) mutate method given!');

  // Some mutation method objects may contain additional config but carry an identity equal to enum value.
  let key: string | undefined;
  if (typeof method === 'string') key = method;
  else key = method?.name ?? method?.type ?? method?.identity;
  if (!key) {
    // Fallback: identity match against exported mutation objects
    for (const k in mutation) {
      if (method === (mutation as any)[k]) {
        key = k;
        break;
      }
    }
  }
  const fn = key ? MUTATION_DISPATCH[key] : undefined;
  if (!fn) {
    if (config.warnings) {
      // eslint-disable-next-line no-console
      console.warn('[mutate] Unknown mutation method ignored:', key);
    }
    return; // graceful no-op for invalid method objects
  }
  fn.call(this, method);
  (this as any)._topoDirty = true; // Mark topology/order caches invalid.
}

// ======================= Individual mutation helpers ======================= //

/**
 * ADD_NODE: Insert a new hidden node by splitting an existing connection.
 *
 * Deterministic test mode (config.deterministicChainMode):
 *  - Maintain an internal linear chain (input → hidden* → output).
 *  - Always split the chain's terminal edge, guaranteeing depth +1 per call.
 *  - Prune side edges from chain nodes to keep depth measurement unambiguous.
 *
 * Standard evolutionary mode:
 *  - Sample a random existing connection and perform the classical NEAT split.
 *
 * Core algorithm (stochastic variant):
 *  1. Pick connection (random).
 *  2. Disconnect it (preserve any gater reference).
 *  3. Create hidden node (random activation mutation).
 *  4. Insert before output tail to preserve ordering invariants.
 *  5. Connect source→hidden and hidden→target.
 *  6. Reassign gater uniformly to one of the new edges.
 */
function _addNode(this: Network): void {
  const internal = this as any;
  if (internal._enforceAcyclic) internal._topoDirty = true;

  // Deterministic linear chain growth: always split the terminal edge of a persisted chain.
  if (config.deterministicChainMode) {
    const inputNode = this.nodes.find((n) => n.type === 'input');
    const outputNode = this.nodes.find((n) => n.type === 'output');
    if (!inputNode || !outputNode) return;
    // Initialize chain & seed direct edge only once (first invocation) so subsequent splits extend depth.
    if (!internal._detChain) {
      if (
        !this.connections.some(
          (c) => c.from === inputNode && c.to === outputNode
        )
      ) {
        this.connect(inputNode, outputNode);
      }
      internal._detChain = [inputNode]; // store chain nodes (excluding output)
    }
    const chain: any[] = internal._detChain;
    const tail = chain[chain.length - 1];
    // Ensure tail -> output edge exists (recreate if pruned earlier)
    let terminal = this.connections.find(
      (c) => c.from === tail && c.to === outputNode
    );
    if (!terminal) terminal = this.connect(tail, outputNode)[0];
    const prevGater = terminal.gater;
    this.disconnect(terminal.from, terminal.to);
    const hidden = new Node('hidden', undefined, internal._rand);
    hidden.mutate(mutation.MOD_ACTIVATION);
    const outIndex = this.nodes.indexOf(outputNode);
    const insertIndex = Math.min(outIndex, this.nodes.length - this.output);
    this.nodes.splice(insertIndex, 0, hidden);
    internal._nodeIndexDirty = true;
    const c1 = this.connect(tail, hidden)[0];
    const c2 = this.connect(hidden, outputNode)[0];
    chain.push(hidden);
    internal._preferredChainEdge = c2; // maintain legacy pointer for opportunistic logic elsewhere
    if (prevGater) this.gate(prevGater, internal._rand() >= 0.5 ? c1 : c2);
    // Prune any extra outgoing edges from chain nodes so path stays linear & depth metric stable.
    for (let i = 0; i < chain.length; i++) {
      const node = chain[i];
      const target = i + 1 < chain.length ? chain[i + 1] : outputNode;
      const keep = node.connections.out.find((e: any) => e.to === target);
      if (keep) {
        for (const extra of node.connections.out.slice()) {
          if (extra !== keep) {
            try {
              this.disconnect(extra.from, extra.to);
            } catch {}
          }
        }
      }
    }
    return; // done deterministic path
  }

  // Non-deterministic (original) behaviour: split a random connection. Abort if no connections yet.
  if (this.connections.length === 0) {
    // If no connections (fresh network), proactively create a random input->output edge to enable future splits.
    const input = this.nodes.find((n) => n.type === 'input');
    const output = this.nodes.find((n) => n.type === 'output');
    if (input && output) this.connect(input, output);
    else return;
  }
  const connection = this.connections[
    Math.floor(internal._rand() * this.connections.length)
  ];
  if (!connection) return;
  const prevGater = connection.gater;
  this.disconnect(connection.from, connection.to);
  const hidden = new Node('hidden', undefined, internal._rand);
  hidden.mutate(mutation.MOD_ACTIVATION);
  const targetIndex = this.nodes.indexOf(connection.to);
  const insertIndex = Math.min(targetIndex, this.nodes.length - this.output);
  this.nodes.splice(insertIndex, 0, hidden);
  internal._nodeIndexDirty = true;
  const c1 = this.connect(connection.from, hidden)[0];
  const c2 = this.connect(hidden, connection.to)[0];
  internal._preferredChainEdge = c2;
  if (prevGater) this.gate(prevGater, internal._rand() >= 0.5 ? c1 : c2);
}

/**
 * SUB_NODE: Remove a random hidden node (if any remain).
 * After removal a tiny deterministic weight nudge encourages observable phenotype change in tests.
 */
function _subNode(this: Network): void {
  const hidden = this.nodes.filter((n) => n.type === 'hidden');
  if (hidden.length === 0) {
    if (config.warnings) console.warn('No hidden nodes left to remove!');
    return;
  }
  const internal = this as any;
  const victim = hidden[Math.floor(internal._rand() * hidden.length)];
  this.remove(victim);
  // Nudge a weight slightly so tests expecting output change are robust.
  const anyConn = this.connections[0];
  if (anyConn) anyConn.weight += 1e-4;
}

/**
 * ADD_CONN: Add a new forward (acyclic) connection between two previously unconnected nodes.
 * Recurrent edges are handled separately by ADD_BACK_CONN.
 */
function _addConn(this: Network): void {
  const netInternal = this as any;
  if (netInternal._enforceAcyclic) netInternal._topoDirty = true;
  /** Candidate pairs [source,target]. */
  const forwardConnectionCandidates: Array<[any, any]> = [];
  for (
    let sourceIndex = 0;
    sourceIndex < this.nodes.length - this.output;
    sourceIndex++
  ) {
    const sourceNode = this.nodes[sourceIndex];
    for (
      let targetIndex = Math.max(sourceIndex + 1, this.input);
      targetIndex < this.nodes.length;
      targetIndex++
    ) {
      const targetNode = this.nodes[targetIndex];
      if (!sourceNode.isProjectingTo(targetNode))
        forwardConnectionCandidates.push([sourceNode, targetNode]);
    }
  }
  if (forwardConnectionCandidates.length === 0) return;
  /** Selected pair to connect. */
  const selectedPair =
    forwardConnectionCandidates[
      Math.floor(netInternal._rand() * forwardConnectionCandidates.length)
    ];
  this.connect(selectedPair[0], selectedPair[1]);
}

/**
 * SUB_CONN: Remove a forward connection chosen under redundancy heuristics to avoid disconnects.
 */
function _subConn(this: Network): void {
  const netInternal = this as any;
  /** Candidate removable forward connections. */
  const removableForwardConnections = this.connections.filter(
    (candidateConn) => {
      const sourceHasMultipleOutgoing =
        candidateConn.from.connections.out.length > 1;
      const targetHasMultipleIncoming =
        candidateConn.to.connections.in.length > 1;
      const targetLayerPeers = this.nodes.filter(
        (n) =>
          n.type === candidateConn.to.type &&
          Math.abs(
            this.nodes.indexOf(n) - this.nodes.indexOf(candidateConn.to)
          ) < Math.max(this.input, this.output)
      );
      let wouldDisconnectLayerPeerGroup = false;
      if (targetLayerPeers.length > 0) {
        const peerConnectionsFromSource = this.connections.filter(
          (c) =>
            c.from === candidateConn.from && targetLayerPeers.includes(c.to)
        );
        if (peerConnectionsFromSource.length <= 1)
          wouldDisconnectLayerPeerGroup = true;
      }
      return (
        sourceHasMultipleOutgoing &&
        targetHasMultipleIncoming &&
        this.nodes.indexOf(candidateConn.to) >
          this.nodes.indexOf(candidateConn.from) &&
        !wouldDisconnectLayerPeerGroup
      );
    }
  );
  if (removableForwardConnections.length === 0) return;
  /** Connection chosen for removal. */
  const connectionToRemove =
    removableForwardConnections[
      Math.floor(netInternal._rand() * removableForwardConnections.length)
    ];
  this.disconnect(connectionToRemove.from, connectionToRemove.to);
}

/**
 * MOD_WEIGHT: Perturb a single (possibly self) connection weight by uniform delta in [min,max].
 */
function _modWeight(this: Network, method: any): void {
  /** Combined list of normal and self connections. */
  const allConnections = this.connections.concat(this.selfconns);
  if (allConnections.length === 0) return;
  /** Random connection to perturb. */
  const connectionToPerturb =
    allConnections[Math.floor((this as any)._rand() * allConnections.length)];
  /** Delta sampled uniformly from [min,max]. */
  const modification =
    (this as any)._rand() * (method.max - method.min) + method.min;
  connectionToPerturb.weight += modification;
}

/**
 * MOD_BIAS: Delegate to node.mutate to adjust bias of a random non‑input node.
 */
function _modBias(this: Network, method: any): void {
  if (this.nodes.length <= this.input) return;
  /** Index of target node (excluding inputs). */
  const targetNodeIndex = Math.floor(
    (this as any)._rand() * (this.nodes.length - this.input) + this.input
  );
  /** Selected node for bias mutation. */
  const nodeForBiasMutation = this.nodes[targetNodeIndex];
  nodeForBiasMutation.mutate(method);
}

/**
 * MOD_ACTIVATION: Swap activation (squash) of a random eligible node; may exclude outputs.
 */
function _modActivation(this: Network, method: any): void {
  /** Whether output nodes may be mutated. */
  const canMutateOutput = method.mutateOutput ?? true;
  /** Count of nodes available for mutation. */
  const numMutableNodes =
    this.nodes.length - this.input - (canMutateOutput ? 0 : this.output);
  if (numMutableNodes <= 0) {
    if (config.warnings)
      console.warn(
        'No nodes available for activation function mutation based on config.'
      );
    return;
  }
  /** Index of chosen node. */
  const targetNodeIndex = Math.floor(
    (this as any)._rand() * numMutableNodes + this.input
  );
  /** Target node. */
  const targetNode = this.nodes[targetNodeIndex];
  targetNode.mutate(method);
}

/**
 * ADD_SELF_CONN: Add a self loop to a random eligible node (only when cycles allowed).
 */
function _addSelfConn(this: Network): void {
  const netInternal = this as any;
  if (netInternal._enforceAcyclic) return;
  /** Nodes without an existing self connection (excluding inputs). */
  const nodesWithoutSelfLoop = this.nodes.filter(
    (n, idx) => idx >= this.input && n.connections.self.length === 0
  );
  if (nodesWithoutSelfLoop.length === 0) {
    if (config.warnings)
      console.warn('All eligible nodes already have self-connections.');
    return;
  }
  /** Node selected to receive self loop. */
  const nodeReceivingSelfLoop =
    nodesWithoutSelfLoop[
      Math.floor(netInternal._rand() * nodesWithoutSelfLoop.length)
    ];
  this.connect(nodeReceivingSelfLoop, nodeReceivingSelfLoop);
}

/**
 * SUB_SELF_CONN: Remove a random existing self loop.
 */
function _subSelfConn(this: Network): void {
  if (this.selfconns.length === 0) {
    if (config.warnings) console.warn('No self-connections exist to remove.');
    return;
  }
  /** Chosen self connection for removal. */
  const selfConnectionToRemove = this.selfconns[
    Math.floor((this as any)._rand() * this.selfconns.length)
  ];
  this.disconnect(selfConnectionToRemove.from, selfConnectionToRemove.to);
}

/**
 * ADD_GATE: Assign a random (hidden/output) node to gate a random ungated connection.
 */
function _addGate(this: Network): void {
  const netInternal = this as any;
  /** All connections (including self connections). */
  const allConnectionsIncludingSelf = this.connections.concat(this.selfconns);
  /** Ungated connection candidates. */
  const ungatedConnectionCandidates = allConnectionsIncludingSelf.filter(
    (c: any) => c.gater === null
  );
  if (
    ungatedConnectionCandidates.length === 0 ||
    this.nodes.length <= this.input
  ) {
    if (config.warnings) console.warn('All connections are already gated.');
    return;
  }
  /** Index for gating node (hidden or output). */
  const gatingNodeIndex = Math.floor(
    netInternal._rand() * (this.nodes.length - this.input) + this.input
  );
  /** Gating node. */
  const gatingNode = this.nodes[gatingNodeIndex];
  /** Connection to gate. */
  const connectionToGate =
    ungatedConnectionCandidates[
      Math.floor(netInternal._rand() * ungatedConnectionCandidates.length)
    ];
  this.gate(gatingNode, connectionToGate);
}

/**
 * SUB_GATE: Remove gating from a random previously gated connection.
 */
function _subGate(this: Network): void {
  if (this.gates.length === 0) {
    if (config.warnings) console.warn('No gated connections to ungate.');
    return;
  }
  /** Random gated connection reference. */
  const gatedConnectionIndex = Math.floor(
    (this as any)._rand() * this.gates.length
  );
  const gatedConnection = this.gates[gatedConnectionIndex];
  this.ungate(gatedConnection);
}

/**
 * ADD_BACK_CONN: Add a backward (recurrent) connection (acyclic mode must be off).
 */
function _addBackConn(this: Network): void {
  const netInternal = this as any;
  if (netInternal._enforceAcyclic) return;
  /** Candidate backward pairs [laterNode, earlierNode]. */
  const backwardConnectionCandidates: Array<[any, any]> = [];
  for (
    let laterIndex = this.input;
    laterIndex < this.nodes.length;
    laterIndex++
  ) {
    const laterNode = this.nodes[laterIndex];
    for (
      let earlierIndex = this.input;
      earlierIndex < laterIndex;
      earlierIndex++
    ) {
      const earlierNode = this.nodes[earlierIndex];
      if (!laterNode.isProjectingTo(earlierNode))
        backwardConnectionCandidates.push([laterNode, earlierNode]);
    }
  }
  if (backwardConnectionCandidates.length === 0) return;
  /** Chosen backward pair. */
  const selectedBackwardPair =
    backwardConnectionCandidates[
      Math.floor(netInternal._rand() * backwardConnectionCandidates.length)
    ];
  this.connect(selectedBackwardPair[0], selectedBackwardPair[1]);
}

/**
 * SUB_BACK_CONN: Remove a backward connection meeting redundancy heuristics.
 */
function _subBackConn(this: Network): void {
  /** Candidate backward connections to remove. */
  const removableBackwardConnections = this.connections.filter(
    (candidateConn) =>
      candidateConn.from.connections.out.length > 1 &&
      candidateConn.to.connections.in.length > 1 &&
      this.nodes.indexOf(candidateConn.from) >
        this.nodes.indexOf(candidateConn.to)
  );
  if (removableBackwardConnections.length === 0) return;
  /** Selected backward connection. */
  const backwardConnectionToRemove =
    removableBackwardConnections[
      Math.floor((this as any)._rand() * removableBackwardConnections.length)
    ];
  this.disconnect(
    backwardConnectionToRemove.from,
    backwardConnectionToRemove.to
  );
}

/**
 * SWAP_NODES: Exchange bias & activation function between two random eligible nodes.
 */
function _swapNodes(this: Network, method: any): void {
  const netInternal = this as any;
  /** Whether output nodes may be included. */
  const canSwapOutput = method.mutateOutput ?? true;
  /** Number of nodes eligible for swapping. */
  const numSwappableNodes =
    this.nodes.length - this.input - (canSwapOutput ? 0 : this.output);
  if (numSwappableNodes < 2) return;
  /** First random index. */
  let firstNodeIndex = Math.floor(
    netInternal._rand() * numSwappableNodes + this.input
  );
  /** Second random index (distinct). */
  let secondNodeIndex = Math.floor(
    netInternal._rand() * numSwappableNodes + this.input
  );
  while (firstNodeIndex === secondNodeIndex)
    secondNodeIndex = Math.floor(
      netInternal._rand() * numSwappableNodes + this.input
    );
  /** First node. */
  const firstNode = this.nodes[firstNodeIndex];
  /** Second node. */
  const secondNode = this.nodes[secondNodeIndex];
  /** Temporary store for bias before swap. */
  const tempBias = firstNode.bias;
  /** Temporary store for activation function before swap. */
  const tempSquash = firstNode.squash;
  firstNode.bias = secondNode.bias;
  firstNode.squash = secondNode.squash;
  secondNode.bias = tempBias;
  secondNode.squash = tempSquash;
}

/**
 * ADD_LSTM_NODE: Replace a random connection with a minimal 1‑unit LSTM block (macro mutation).
 */
function _addLSTMNode(this: Network): void {
  const netInternal = this as any;
  if (netInternal._enforceAcyclic) return;
  if (this.connections.length === 0) return;
  /** Connection selected to expand into an LSTM block. */
  const connectionToExpand = this.connections[
    Math.floor(Math.random() * this.connections.length)
  ];
  /** Original gater to reapply to new outgoing edge. */
  const gaterLSTM = connectionToExpand.gater;
  this.disconnect(connectionToExpand.from, connectionToExpand.to);
  // Dynamic import of layer factory (kept lazy to avoid circular refs if any).
  const Layer = require('../layer').default;
  const lstmLayer = Layer.lstm(1);
  // Convert produced layer's nodes to hidden and append to network node list.
  lstmLayer.nodes.forEach((n: any) => {
    n.type = 'hidden';
    this.nodes.push(n);
  });
  // Reconnect using first internal node as entry & layer output node as exit.
  this.connect(connectionToExpand.from, lstmLayer.nodes[0]);
  this.connect(lstmLayer.output.nodes[0], connectionToExpand.to);
  if (gaterLSTM)
    this.gate(gaterLSTM, this.connections[this.connections.length - 1]);
}

/**
 * ADD_GRU_NODE: Replace a random connection with a minimal 1‑unit GRU block.
 */
function _addGRUNode(this: Network): void {
  const netInternal = this as any;
  if (netInternal._enforceAcyclic) return;
  if (this.connections.length === 0) return;
  /** Connection selected to expand into a GRU block. */
  const connectionToExpand = this.connections[
    Math.floor(Math.random() * this.connections.length)
  ];
  /** Original gater (if any). */
  const gaterGRU = connectionToExpand.gater;
  this.disconnect(connectionToExpand.from, connectionToExpand.to);
  const Layer = require('../layer').default;
  const gruLayer = Layer.gru(1);
  gruLayer.nodes.forEach((n: any) => {
    n.type = 'hidden';
    this.nodes.push(n);
  });
  this.connect(connectionToExpand.from, gruLayer.nodes[0]);
  this.connect(gruLayer.output.nodes[0], connectionToExpand.to);
  if (gaterGRU)
    this.gate(gaterGRU, this.connections[this.connections.length - 1]);
}

/**
 * REINIT_WEIGHT: Reinitialize all incoming/outgoing/self connection weights for a random node.
 * Useful as a heavy mutation to escape local minima. Falls back silently if no eligible node.
 */
function _reinitWeight(this: Network, method: any): void {
  if (this.nodes.length <= this.input) return;
  const internal = this as any;
  const idx = Math.floor(
    internal._rand() * (this.nodes.length - this.input) + this.input
  );
  const node = this.nodes[idx];
  const min = method?.min ?? -1;
  const max = method?.max ?? 1;
  const sample = () => internal._rand() * (max - min) + min;
  // Incoming
  for (const c of node.connections.in) c.weight = sample();
  // Outgoing
  for (const c of node.connections.out) c.weight = sample();
  // Self
  for (const c of node.connections.self) c.weight = sample();
}

/**
 * BATCH_NORM: Placeholder mutation – marks a random hidden node with a flag for potential
 * future batch normalization integration. Currently a no-op beyond tagging.
 */
function _batchNorm(this: Network): void {
  const hidden = this.nodes.filter((n) => n.type === 'hidden');
  if (!hidden.length) return;
  const internal = this as any;
  const node = hidden[Math.floor(internal._rand() * hidden.length)] as any;
  node._batchNorm = true; // simple tag; downstream training code could act on this.
}
