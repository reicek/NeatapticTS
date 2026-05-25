import type Connection from '../../connection';
import type Node from '../../node';
import type {
  GatingNetworkProps as NetworkGatingProps,
  SubNodeMutationConfig,
} from '../network.types';

/**
 * Normalized SUB_NODE mutation configuration resolved before node-removal rewiring begins; undefined when no mutation config is present in the caller context.
 */
export type NodeRemovalMutationConfig = SubNodeMutationConfig | undefined;

/**
 * Mutable collection of gater nodes retained during hidden-node removal so they can be reassigned to the bridging connections created by the rewiring pass.
 */
export type PreservedGaters = Node[];

/**
 * Ordered list of predecessor or successor nodes collected during bridge construction when a hidden node is removed from the network graph.
 */
export type ConnectedNodeList = Node[];

/**
 * List of newly created bridging connections that can each receive a reassigned gater from the node-removal rewiring operation.
 */
export type BridgingConnectionList = Connection[];

/**
 * Network shape extension that exposes the mutable flag used to invalidate the node index cache after structural gating changes alter the graph.
 */
export type MutableNetworkGatingProps = NetworkGatingProps;
