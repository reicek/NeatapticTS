import type Network from '../../network/network';
import type Node from '../../node';
import type {
  PathSearchContext as NetworkPathSearchContext,
  TopologyBuildContext as NetworkTopologyBuildContext,
  TopologyNetworkProps as NetworkTopologyProps,
} from '../network.types';

/** Input node-type discriminator used for queue seeding. */
export const INPUT_NODE_TYPE = 'input';

/** Zero baseline used for degree counts and empty-size checks. */
export const ZERO_COUNT = 0;

/** Unit decrement/increment used for in-degree tally updates. */
export const IN_DEGREE_DECREMENT = 1;

/** Internal topology state view carried across helper groups. */
export type TopologyNetworkProps = NetworkTopologyProps;

/** Mutable context used while building Kahn topological order. */
export type TopologyBuildContext = NetworkTopologyBuildContext;

/** Mutable context used while running iterative DFS reachability checks. */
export type PathSearchContext = NetworkPathSearchContext;

/** Network instance type used by topology helpers. */
export type TopologyNetwork = Network;

/** Node instance type used by topology helpers. */
export type TopologyNode = Node;
