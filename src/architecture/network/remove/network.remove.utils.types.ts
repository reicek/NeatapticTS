import type Network from '../../network/network';
import type Node from '../../node';
import type Connection from '../../connection';

/**
 * Node type literal for input anchors that cannot be removed from the network topology.
 */
export const NODE_TYPE_INPUT: Node['type'] = 'input';

/**
 * Node type literal for output anchors that cannot be removed from the network topology.
 */
export const NODE_TYPE_OUTPUT: Node['type'] = 'output';

/**
 * Error emitted when the target node passed to remove is not present in the network node list.
 */
export const ERROR_NODE_NOT_IN_NETWORK = 'Node not in network';

/**
 * Error emitted when a caller attempts to remove an input or output anchor node from the network topology.
 */
export const ERROR_CANNOT_REMOVE_ANCHOR_NODE =
  'Cannot remove input or output node from the network.';

/**
 * Sentinel index value returned when a node search yields no match in the network node list.
 */
export const NODE_NOT_FOUND_INDEX = -1;

/**
 * Array index used to retrieve the first element spliced from the node list during a single-node removal operation.
 */
export const FIRST_REMOVED_NODE_INDEX = 0;

/** Internal network properties accessed by the remove utilities to manage dirty-state flags after node removal. */
export interface NetworkRemoveProps {
  /** Topology dirty marker. */
  _topoDirty?: boolean;
  /** Node-index dirty marker. */
  _nodeIndexDirty?: boolean;
  /** Slab dirty marker. */
  _slabDirty?: boolean;
  /** Adjacency dirty marker. */
  _adjDirty?: boolean;
}

/** Immutable context object used to carry one validated node-removal orchestration request. */
export interface NodeRemovalContext {
  /** Owning network instance. */
  network: Network;
  /** Internal mutable network flags. */
  internalNetwork: NetworkRemoveProps;
  /** Node requested for removal. */
  targetNode: Node;
  /** Index of target node in network list. */
  targetNodeIndex: number;
}

/** Snapshot of all node adjacency connection lists captured prior to removal. */
export interface NodeConnectionSnapshotContext {
  /** Incoming connections to removed node. */
  inboundConnections: Connection[];
  /** Outgoing connections from removed node. */
  outboundConnections: Connection[];
  /** Number of removed self-connections. */
  selfConnectionCount: number;
}

/** Endpoint pair describing source and target nodes for reconnecting bridged paths. */
export interface ReconnectEndpointPairContext {
  /** Source node of candidate reconnect edge. */
  sourceNode: Node;
  /** Target node of candidate reconnect edge. */
  targetNode: Node;
}
