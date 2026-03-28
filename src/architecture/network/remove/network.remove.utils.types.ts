import type Network from '../../network/network';
import type Node from '../../node';
import type Connection from '../../connection';

/**
 * Node type literal for input anchors.
 */
export const NODE_TYPE_INPUT: Node['type'] = 'input';

/**
 * Node type literal for output anchors.
 */
export const NODE_TYPE_OUTPUT: Node['type'] = 'output';

/**
 * Error emitted when target node is not part of the network.
 */
export const ERROR_NODE_NOT_IN_NETWORK = 'Node not in network';

/**
 * Error emitted when trying to remove structural anchor nodes.
 */
export const ERROR_CANNOT_REMOVE_ANCHOR_NODE =
  'Cannot remove input or output node from the network.';

/**
 * Sentinel index used when node is not found.
 */
export const NODE_NOT_FOUND_INDEX = -1;

/**
 * Index for selecting first spliced node.
 */
export const FIRST_REMOVED_NODE_INDEX = 0;

/** Internal network properties accessed during remove operations. */
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

/** Immutable context for validated node-removal request. */
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

/** Snapshot of node adjacency prior to removal. */
export interface NodeConnectionSnapshotContext {
  /** Incoming connections to removed node. */
  inboundConnections: Connection[];
  /** Outgoing connections from removed node. */
  outboundConnections: Connection[];
  /** Number of removed self-connections. */
  selfConnectionCount: number;
}

/** Endpoint pair for reconnecting bridged paths. */
export interface ReconnectEndpointPairContext {
  /** Source node of candidate reconnect edge. */
  sourceNode: Node;
  /** Target node of candidate reconnect edge. */
  targetNode: Node;
}
