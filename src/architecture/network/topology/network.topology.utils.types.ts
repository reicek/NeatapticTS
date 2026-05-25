import type Network from '../../network/network';
import type Node from '../../node';
import type {
  ActivationSchedule as NetworkActivationSchedule,
  ActivationSchedulingDiagnostics as NetworkActivationSchedulingDiagnostics,
  ActivationScheduleStep as NetworkActivationScheduleStep,
  PathSearchContext as NetworkPathSearchContext,
  TopologyBuildContext as NetworkTopologyBuildContext,
  TopologyNetworkProps as NetworkTopologyProps,
} from '../network.types';

/** Input node-type discriminator used to seed the Kahn topological sort with source nodes that have no predecessors. */
export const INPUT_NODE_TYPE = 'input';

/** Zero baseline value used to initialize degree counters and empty-size comparisons during topological scheduling. */
export const ZERO_COUNT = 0;

/** Unit step value applied when decrementing or incrementing in-degree tally entries during Kahn queue processing. */
export const IN_DEGREE_DECREMENT = 1;

/** Internal topology state view carrying network node and connection lists across all topology helper groups consistently. */
export type TopologyNetworkProps = NetworkTopologyProps;

/** Mutable scratch context allocated and carried while building the Kahn-algorithm topological activation order. */
export type TopologyBuildContext = NetworkTopologyBuildContext;

/** Mutable scratch context allocated and carried while running iterative depth-first reachability checks across the graph. */
export type PathSearchContext = NetworkPathSearchContext;

/** Deterministic activation schedule type produced by topology helpers and consumed by the activation chapter. */
export type ActivationSchedule = NetworkActivationSchedule;

/** Human-readable activation scheduling diagnostics type carrying node counts, depth, and coverage metadata for inspection. */
export type ActivationSchedulingDiagnostics =
  NetworkActivationSchedulingDiagnostics;

/** One ordered step in the deterministic activation schedule produced by the topology sort and consumed at inference time. */
export type ActivationScheduleStep = NetworkActivationScheduleStep;

/** Network instance type alias used by topology helpers to avoid direct runtime imports at the utility boundary. */
export type TopologyNetwork = Network;

/** Node instance type alias used by topology helpers to avoid direct runtime imports at the utility boundary. */
export type TopologyNode = Node;
