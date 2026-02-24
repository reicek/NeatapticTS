import type Connection from '../../connection';
import type Node from '../../node';
import type {
  GatingNetworkProps as NetworkGatingProps,
  SubNodeMutationConfig,
} from '../network.types';

/**
 * Normalized SUB_NODE mutation configuration used during node-removal rewiring.
 */
export type NodeRemovalMutationConfig = SubNodeMutationConfig | undefined;

/**
 * Mutable gater collection retained while removing a hidden node.
 */
export type PreservedGaters = Node[];

/**
 * Predecessor or successor node collection used during bridge construction.
 */
export type ConnectedNodeList = Node[];

/**
 * Newly created connections that can be assigned preserved gaters.
 */
export type BridgingConnectionList = Connection[];

/**
 * Network shape extension used to flag node index cache invalidation.
 */
export type MutableNetworkGatingProps = NetworkGatingProps;
