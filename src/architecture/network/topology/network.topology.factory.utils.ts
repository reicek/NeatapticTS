import type Network from '../../network/network';
import type Connection from '../../connection';
import Node from '../../node';
import type { TopologyNetworkProps } from '../network.types';
import { setTopologyIntent } from './network.topology.contract.utils';

type NetworkConstructor = new (input: number, output: number) => Network;

interface MlpNodeLayers {
  inputNodes: Node[];
  hiddenLayers: Node[][];
  outputNodes: Node[];
}

/**
 * Build a strictly layered, fully connected MLP network from layer sizes.
 *
 * @param this Network constructor.
 * @param inputCount Number of input nodes.
 * @param hiddenCounts Hidden-layer node counts.
 * @param outputCount Number of output nodes.
 * @returns Newly created MLP network.
 */
/**
 * Contract for createMLP.
 */
export function createMLP(
  this: new (input: number, output: number) => Network,
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): Network {
  // Step 1: Build each layer's nodes.
  const mlpNodeLayers = createMlpNodeLayers(
    inputCount,
    hiddenCounts,
    outputCount,
  );

  // Step 2: Create the network instance.
  const networkInstance = instantiateNetwork(this, inputCount, outputCount);

  // Step 3: Assign canonical node ordering.
  assignNetworkNodes(networkInstance, mlpNodeLayers);

  // Step 4: Fully connect adjacent layers.
  connectMlpLayers(mlpNodeLayers);

  // Step 5: Rebuild canonical connection storage.
  rebuildConnections(networkInstance);

  // Step 6: Mark topology as requiring refresh.
  markTopologyDirty(networkInstance);

  // Step 7: Preserve the public feed-forward contract for layered MLP builders.
  setTopologyIntent.call(networkInstance, 'feed-forward');

  return networkInstance;
}

/**
 * Rebuild the canonical connection array from all per-node outgoing lists.
 *
 * @param networkInstance Target network.
 */
export function rebuildConnections(networkInstance: Network): void {
  // Step 1: Collect deduplicated outgoing connections.
  const uniqueConnections = collectUniqueOutgoingConnections(networkInstance);

  // Step 2: Store a canonical array on the network.
  networkInstance.connections = convertConnectionSetToArray(uniqueConnections);
}

/**
 * Build input, hidden, and output node layers for an MLP topology.
 *
 * @param inputCount Number of input nodes.
 * @param hiddenCounts Hidden-layer node counts.
 * @param outputCount Number of output nodes.
 * @returns Grouped node layers for MLP assembly.
 */
function createMlpNodeLayers(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): MlpNodeLayers {
  return {
    inputNodes: createNodesOfType(inputCount, 'input'),
    hiddenLayers: createHiddenLayers(hiddenCounts),
    outputNodes: createNodesOfType(outputCount, 'output'),
  };
}

/**
 * Instantiate a new network using the runtime constructor.
 *
 * @param networkFactory Network constructor function.
 * @param inputCount Number of input nodes.
 * @param outputCount Number of output nodes.
 * @returns Newly instantiated network.
 */
function instantiateNetwork(
  networkFactory: NetworkConstructor,
  inputCount: number,
  outputCount: number,
): Network {
  return new networkFactory(inputCount, outputCount);
}

/**
 * Assign ordered nodes to the network instance.
 *
 * @param networkInstance Target network.
 * @param mlpNodeLayers Grouped node layers for this MLP.
 */
function assignNetworkNodes(
  networkInstance: Network,
  mlpNodeLayers: MlpNodeLayers,
): void {
  networkInstance.nodes = createOrderedNodeList(mlpNodeLayers);
  networkInstance.refreshExplicitIORoles();
}

/**
 * Create all nodes for a single fixed node type.
 *
 * @param nodeCount Number of nodes to create.
 * @param nodeType Node type identifier.
 * @returns Node list of the requested type.
 */
function createNodesOfType(
  nodeCount: number,
  nodeType: 'input' | 'hidden' | 'output',
): Node[] {
  const nodes: Node[] = [];
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
    nodes.push(new Node(nodeType));
  }
  return nodes;
}

/**
 * Create all hidden layers for an MLP topology.
 *
 * @param hiddenCounts Hidden-layer node counts.
 * @returns Hidden layers in forward order.
 */
function createHiddenLayers(hiddenCounts: number[]): Node[][] {
  const hiddenLayers: Node[][] = [];
  for (const hiddenNodeCount of hiddenCounts) {
    hiddenLayers.push(createNodesOfType(hiddenNodeCount, 'hidden'));
  }
  return hiddenLayers;
}

/**
 * Build the canonical ordered node list used by the network.
 *
 * @param mlpNodeLayers Grouped node layers for this MLP.
 * @returns Ordered node list: input, hidden, then output.
 */
function createOrderedNodeList(mlpNodeLayers: MlpNodeLayers): Node[] {
  return [
    ...mlpNodeLayers.inputNodes,
    ...flattenNodeLayers(mlpNodeLayers.hiddenLayers),
    ...mlpNodeLayers.outputNodes,
  ];
}

/**
 * Flatten layered node collections into a single ordered list.
 *
 * @param nodeLayers Layered node collections.
 * @returns Flattened node list.
 */
function flattenNodeLayers(nodeLayers: Node[][]): Node[] {
  const flattenedNodes: Node[] = [];
  for (const nodeLayer of nodeLayers) {
    for (const node of nodeLayer) {
      flattenedNodes.push(node);
    }
  }
  return flattenedNodes;
}

/**
 * Fully connect each adjacent layer in MLP order.
 *
 * @param mlpNodeLayers Grouped node layers for this MLP.
 */
function connectMlpLayers(mlpNodeLayers: MlpNodeLayers): void {
  let previousLayer = mlpNodeLayers.inputNodes;
  for (const hiddenLayer of mlpNodeLayers.hiddenLayers) {
    connectLayerPair(previousLayer, hiddenLayer);
    previousLayer = hiddenLayer;
  }
  connectLayerPair(previousLayer, mlpNodeLayers.outputNodes);
}

/**
 * Fully connect every source node to every target node.
 *
 * @param sourceLayer Source layer.
 * @param targetLayer Target layer.
 */
function connectLayerPair(sourceLayer: Node[], targetLayer: Node[]): void {
  for (const targetNode of targetLayer) {
    for (const sourceNode of sourceLayer) {
      sourceNode.connect(targetNode);
    }
  }
}

/**
 * Mark a network topology as dirty after structural edits.
 *
 * @param networkInstance Network instance to mark.
 */
function markTopologyDirty(networkInstance: Network): void {
  (networkInstance as unknown as TopologyNetworkProps)._topoDirty = true;
}

/**
 * Collect unique outgoing connections across all network nodes.
 *
 * @param networkInstance Target network.
 * @returns Set of unique outgoing connections.
 */
function collectUniqueOutgoingConnections(
  networkInstance: Network,
): Set<Connection> {
  const allConnections = new Set<Connection>();
  for (const node of networkInstance.nodes) {
    addOutgoingConnectionsToSet(node.connections.out, allConnections);
  }
  return allConnections;
}

/**
 * Add all outgoing connections to a deduplication set.
 *
 * @param outgoingConnections Outgoing connections from one node.
 * @param allConnections Deduplication set for network connections.
 */
function addOutgoingConnectionsToSet(
  outgoingConnections: Connection[],
  allConnections: Set<Connection>,
): void {
  for (const connection of outgoingConnections) {
    allConnections.add(connection);
  }
}

/**
 * Convert a connection set into the canonical array format.
 *
 * @param uniqueConnections Unique network connections.
 * @returns Array of network connections.
 */
function convertConnectionSetToArray(
  uniqueConnections: Set<Connection>,
): Connection[] {
  return Array.from(uniqueConnections);
}
