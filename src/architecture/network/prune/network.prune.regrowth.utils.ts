import type Network from '../../network/network';
import type Node from '../../node';
import type {
  NetworkPruningProps,
  RegrowthExecutionContext,
  RegrowthPlan,
  RegrowthPlanContext,
} from '../network.types';
import { REGROW_ATTEMPT_MULTIPLIER } from './network.prune.utils.types';

/**
 * Build and execute a bounded connection-regrowth plan when regrowth is enabled.
 * @param currentNetwork - Network to regrow.
 * @param context - Inputs describing regrowth intent.
 * @returns Nothing.
 */
export function maybeRunRegrowth(
  currentNetwork: Network,
  context: RegrowthPlanContext,
): void {
  // Step 1: Build a bounded regrowth execution plan.
  const regrowthPlan = buildRegrowthPlan(context);
  if (!regrowthPlan) return;

  // Step 2: Execute random valid edge additions.
  executeRegrowthAttempts({
    network: currentNetwork,
    desiredRemainingConnections: regrowthPlan.desiredRemainingConnections,
    maxAttempts: regrowthPlan.maxAttempts,
  });
}

/**
 * Convert regrowth intent into a bounded execution plan.
 * @param context - Regrowth planning inputs.
 * @returns A plan when regrowth is meaningful; otherwise null.
 */
function buildRegrowthPlan(context: RegrowthPlanContext): RegrowthPlan | null {
  // Step 1: Skip disabled regrowth.
  if (context.regrowFraction <= 0) return null;

  // Step 2: Compute requested number of regrown connections.
  const intendedRegrowCount = Math.floor(
    context.prunedConnectionCount * context.regrowFraction,
  );
  if (intendedRegrowCount <= 0) return null;

  // Step 3: Translate requested count into attempt budget.
  return {
    desiredRemainingConnections: context.desiredRemainingConnections,
    maxAttempts: intendedRegrowCount * REGROW_ATTEMPT_MULTIPLIER,
  };
}

/**
 * Execute bounded stochastic regrowth attempts.
 * @param context - Regrowth execution settings.
 * @returns Nothing.
 */
function executeRegrowthAttempts(context: RegrowthExecutionContext): void {
  // Step 1: Track how many tries have been consumed.
  let attemptedRegrowthCount = 0;

  // Step 2: Keep trying until target density or attempt cap is reached.
  while (
    shouldContinueRegrowth(
      context.network,
      context.desiredRemainingConnections,
      attemptedRegrowthCount,
      context.maxAttempts,
    )
  ) {
    // Step 3: Consume one attempt and try to add one valid edge.
    attemptedRegrowthCount += 1;
    tryRegrowConnection(context.network);
  }
}

/**
 * Decide whether another regrowth attempt is allowed.
 * @param currentNetwork - Network being regrown.
 * @param desiredRemainingConnections - Target remaining connection count.
 * @param attemptedRegrowthCount - Number of attempts already used.
 * @param maxAttempts - Maximum attempts allowed.
 * @returns True when another attempt should run.
 */
function shouldContinueRegrowth(
  currentNetwork: Network,
  desiredRemainingConnections: number,
  attemptedRegrowthCount: number,
  maxAttempts: number,
): boolean {
  return (
    currentNetwork.connections.length < desiredRemainingConnections &&
    attemptedRegrowthCount < maxAttempts
  );
}

/**
 * Attempt one random valid connection addition.
 * @param currentNetwork - Network being regrown.
 * @returns Nothing.
 */
function tryRegrowConnection(currentNetwork: Network): void {
  // Step 1: Build a valid random source-target pair.
  const regrowthPair = buildRegrowthCandidatePair(currentNetwork);
  if (!regrowthPair) return;

  // Step 2: Materialize the new edge.
  currentNetwork.connect(regrowthPair.sourceNode, regrowthPair.targetNode);
}

/**
 * Build one random regrowth candidate pair if valid.
 * @param currentNetwork - Network being regrown.
 * @returns Candidate node pair or null when invalid.
 */
function buildRegrowthCandidatePair(
  currentNetwork: Network,
): { sourceNode: Node; targetNode: Node } | null {
  // Step 1: Draw random source and target nodes.
  const sourceNode = pickRandomNode(currentNetwork);
  const targetNode = pickRandomNode(currentNetwork);
  if (!sourceNode || !targetNode) return null;

  // Step 2: Validate candidate constraints.
  if (isInvalidRegrowthPair(currentNetwork, sourceNode, targetNode)) {
    return null;
  }

  return { sourceNode, targetNode };
}

/**
 * Pick a random node using the network RNG.
 * @param currentNetwork - Network providing node set and RNG.
 * @returns Random node or undefined when the node list is empty.
 */
function pickRandomNode(currentNetwork: Network): Node | undefined {
  // Step 1: Resolve deterministic RNG when available.
  const randomSource =
    (currentNetwork as unknown as NetworkPruningProps)._rand ?? Math.random;

  // Step 2: Sample one node index uniformly.
  const randomIndex = Math.floor(randomSource() * currentNetwork.nodes.length);
  return currentNetwork.nodes[randomIndex];
}

/**
 * Validate whether a candidate regrowth pair is acceptable.
 * @param currentNetwork - Network being regrown.
 * @param sourceNode - Proposed source node.
 * @param targetNode - Proposed target node.
 * @returns True when the pair must be rejected.
 */
function isInvalidRegrowthPair(
  currentNetwork: Network,
  sourceNode: Node,
  targetNode: Node,
): boolean {
  if (sourceNode === targetNode) return true;
  if (connectionAlreadyExists(currentNetwork, sourceNode, targetNode)) {
    return true;
  }
  return violatesAcyclicConstraint(currentNetwork, sourceNode, targetNode);
}

/**
 * Check whether a connection already exists.
 * @param currentNetwork - Network being regrown.
 * @param sourceNode - Proposed source node.
 * @param targetNode - Proposed target node.
 * @returns True when the edge already exists.
 */
function connectionAlreadyExists(
  currentNetwork: Network,
  sourceNode: Node,
  targetNode: Node,
): boolean {
  return currentNetwork.connections.some(
    (connection) =>
      connection.from === sourceNode && connection.to === targetNode,
  );
}

/**
 * Check whether a pair violates forward-only acyclic ordering.
 * @param currentNetwork - Network being regrown.
 * @param sourceNode - Proposed source node.
 * @param targetNode - Proposed target node.
 * @returns True when acyclic ordering would be violated.
 */
function violatesAcyclicConstraint(
  currentNetwork: Network,
  sourceNode: Node,
  targetNode: Node,
): boolean {
  // Step 1: Skip ordering checks when acyclic enforcement is disabled.
  const networkProperties = currentNetwork as unknown as NetworkPruningProps;
  if (!networkProperties._enforceAcyclic) return false;

  // Step 2: Reject backward edges by node index ordering.
  return (
    currentNetwork.nodes.indexOf(sourceNode) >
    currentNetwork.nodes.indexOf(targetNode)
  );
}
