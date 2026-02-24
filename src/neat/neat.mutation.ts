import type { NeatLike } from './neat.types';
import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
} from './neat.mutation.types';
import * as MutateHelpers from './neat.mutation.utils';

/** Default connection weight used for bootstrap and split in-edges. */
export const DEFAULT_CONNECTION_WEIGHT = 1;
/** Default gene id value when a node has no gene id. */
export const DEFAULT_GENE_ID = 0;
/** Default innovation id value when a connection has none. */
export const DEFAULT_INNOVATION_ID = 0;

/**
 * Mutate every genome in the population according to configured policies.
 *
 * This is the high-level mutation driver used by NeatapticTS. It iterates the
 * current population and, depending on the configured mutation rate and
 * (optional) adaptive mutation controller, applies one or more mutation
 * operators to each genome.
 *
 * Educational notes:
 * - Adaptive mutation allows per-genome mutation rates/amounts to evolve so
 *   that successful genomes can reduce or increase plasticity over time.
 * - Structural mutations (ADD_NODE, ADD_CONN, etc.) may update global
 *   innovation bookkeeping; this function attempts to reuse specialized
 *   helper routines that preserve innovation ids across the population.
 *
 * Example:
 *
 * ```ts
 * // called on a Neat instance after a generation completes
 * neat.mutate();
 * ```
 *
 * @this NeatLike - instance of a Neat controller with population and options
 */
export async function mutate(this: NeatLike): Promise<void> {
  const internal = this as unknown as NeatControllerForMutation;
  const methods = await import('../methods/methods');
  const population = internal.population;

  for (const genome of population) {
    await MutateHelpers.mutateGenome(genome, internal, methods);
  }
}

/**
 * Split a randomly chosen enabled connection and insert a hidden node.
 *
 * This routine attempts to reuse a historical "node split" innovation record
 * so that identical splits across different genomes share the same
 * innovation ids. This preservation of innovation information is important
 * for NEAT-style speciation and genome alignment.
 *
 * Method steps (high-level):
 * - If the genome has no connections, connect an input to an output to
 *   bootstrap connectivity.
 * - Filter enabled connections and choose one at random.
 * - Disconnect the chosen connection and either reuse an existing split
 *   innovation record or create a new hidden node + two connecting
 *   connections (in->new, new->out) assigning new innovation ids.
 * - Insert the newly created node into the genome's node list at the
 *   deterministic position to preserve ordering for downstream algorithms.
 *
 * Example:
 *
 * ```ts
 * neat._mutateAddNodeReuse(genome);
 * ```
 *
 * @this NeatLike - neat controller context (holds innovation tables)
 * @param genome - genome to modify in-place
 */
export async function mutateAddNodeReuse(
  this: NeatLike,
  genome: GenomeWithMetadata,
): Promise<void> {
  const internal = this as unknown as NeatControllerForMutation;

  // Step 1: bootstrap connectivity when no connections exist.
  MutateHelpers.ensureBootstrapConnection(genome, internal);

  // Step 2: pick an enabled connection to split.
  const enabledConnections = MutateHelpers.collectEnabledConnections(genome);
  const chosenConnection = MutateHelpers.chooseConnectionForSplit(
    enabledConnections,
    internal,
  );
  if (!chosenConnection) return;

  // Step 3: compute split metadata and remove the original connection.
  const splitDescriptor = MutateHelpers.buildSplitDescriptor(chosenConnection);
  MutateHelpers.disconnectOriginalConnection(genome, chosenConnection);

  // Step 4: resolve split record and create split node.
  const splitRecord = internal._nodeSplitInnovations.get(
    splitDescriptor.splitKey,
  );
  const { default: NodeClass } = await import('../architecture/node');

  if (splitRecord) {
    MutateHelpers.applySplitWithExistingRecord(
      genome,
      chosenConnection,
      splitDescriptor,
      splitRecord,
      NodeClass,
    );
    return;
  }

  MutateHelpers.applySplitWithNewRecord(
    genome,
    chosenConnection,
    splitDescriptor,
    NodeClass,
    internal,
  );
}

/**
 * Add a connection between two previously unconnected nodes, reusing a
 * stable innovation id per unordered node pair when possible.
 *
 * Notes on behavior:
 * - The search space consists of node pairs (from, to) where `from` is not
 *   already projecting to `to` and respects the input/output ordering used by
 *   the genome representation.
 * - When a historical innovation exists for the unordered pair, the
 *   previously assigned innovation id is reused to keep different genomes
 *   compatible for downstream crossover and speciation.
 *
 * Steps:
 * - Build a list of all legal (from,to) pairs that don't currently have a
 *   connection.
 * - Prefer pairs which already have a recorded innovation id (reuse
 *   candidates) to maximize reuse; otherwise use the full set.
 * - If the genome enforces acyclicity, simulate whether adding the connection
 *   would create a cycle; abort if it does.
 * - Create the connection and set its innovation id, either from the
 *   historical table or by allocating a new global innovation id.
 *
 * @this NeatLike - neat controller context (holds innovation tables)
 * @param genome - genome to modify in-place
 */
export function mutateAddConnReuse(
  this: NeatLike,
  genome: GenomeWithMetadata,
): void {
  const internal = this as unknown as NeatControllerForMutation;

  // Step 1: build candidate node pairs.
  const candidatePairs = MutateHelpers.collectCandidatePairsForConn(genome);
  if (!candidatePairs.length) return;

  // Step 2: choose the candidate pool based on reuse and hidden-pair rules.
  const reuseCandidates = MutateHelpers.filterPairsWithInnovations(
    candidatePairs,
    internal,
  );
  const selectionPool = MutateHelpers.selectPairPool(
    candidatePairs,
    reuseCandidates,
  );

  // Step 3: pick a concrete pair to connect.
  const chosenPair = MutateHelpers.choosePairForConn(selectionPool, internal);
  if (!chosenPair) return;

  // Step 4: evaluate acyclic constraints if configured.
  const pairNodes = MutateHelpers.resolvePairNodes(chosenPair);
  if (MutateHelpers.shouldAbortForCycle(genome, pairNodes)) return;

  // Step 5: create the connection and assign an innovation id.
  const connection = MutateHelpers.connectChosenPair(genome, pairNodes);
  if (!connection) return;
  MutateHelpers.assignInnovationForConnection(connection, pairNodes, internal);
}

/**
 * Ensure the network has a minimum number of hidden nodes and connectivity.
 */
export async function ensureMinHiddenNodes(
  this: NeatLike,
  network: GenomeWithMetadata,
  multiplierOverride?: number,
): Promise<void> {
  const internal = this as unknown as NeatControllerForMutation;

  // Step 1: resolve node groups and size constraints.
  const nodeGroups = MutateHelpers.collectNodeGroupsForMinHidden(network);
  const maxNodes = MutateHelpers.resolveMaxNodesForMinHidden(internal);
  const minHidden = MutateHelpers.resolveMinHiddenForMinHidden(
    network,
    maxNodes,
    multiplierOverride,
    internal,
  );

  // Step 2: validate network inputs/outputs.
  if (!MutateHelpers.hasRequiredEndpointsForMinHidden(nodeGroups)) {
    MutateHelpers.warnMissingEndpointsForMinHidden();
    return;
  }

  // Step 3: ensure the minimum number of hidden nodes exist.
  await MutateHelpers.ensureHiddenNodeCountForMinHidden(
    network,
    nodeGroups,
    minHidden,
    maxNodes,
  );

  // Step 4: ensure hidden nodes have at least one incoming and outgoing edge.
  MutateHelpers.ensureHiddenConnectivityForMinHidden(
    network,
    nodeGroups,
    internal,
  );

  // Step 5: rebuild cached connection structures.
  await MutateHelpers.rebuildNetworkConnectionsForMinHidden(network);
  return;
}

/**
 * Ensure there are no dead-end nodes (input/output isolation) in the network.
 */
export function ensureNoDeadEnds(
  this: NeatLike,
  network: GenomeWithMetadata,
): void {
  const internal = this as unknown as NeatControllerForMutation;

  // Step 1: gather node groups for connectivity repair.
  const nodeGroups = MutateHelpers.collectNodeGroupsForDeadEnds(network);

  // Step 2: repair input nodes that lack outgoing connections.
  MutateHelpers.ensureInputConnectivityForDeadEnds(
    network,
    nodeGroups,
    internal,
  );

  // Step 3: repair output nodes that lack incoming connections.
  MutateHelpers.ensureOutputConnectivityForDeadEnds(
    network,
    nodeGroups,
    internal,
  );

  // Step 4: repair hidden nodes with missing in/out connections.
  MutateHelpers.ensureHiddenConnectivityForDeadEnds(
    network,
    nodeGroups,
    internal,
  );
  return;
}

/**
 * Select a mutation method respecting structural constraints and adaptive controllers.
 * Mirrors legacy implementation from `neat.ts` to preserve test expectations.
 * `rawReturnForTest` retains historical behavior where the full FFW array is
 * returned for identity checks in tests.
 */
export async function selectMutationMethod(
  this: NeatLike,
  genome: GenomeWithMetadata,
  rawReturnForTest: boolean = true,
): Promise<MutationMethod | MutationMethod[] | null> {
  const internal = this as unknown as NeatControllerForMutation;

  /** Methods module used to access named mutation operator descriptors. */
  const methods = await import('../methods/methods');

  // Step 1: resolve FFW-specific policy behavior.
  const ffwPolicy = MutateHelpers.resolveFFWPolicyForSelect(
    internal,
    methods,
    rawReturnForTest,
  );
  if (ffwPolicy) return ffwPolicy;

  // Step 2: build the operator pool for selection.
  const basePool = MutateHelpers.normalizeMutationPoolForSelect(
    internal,
    methods,
    rawReturnForTest,
  );
  const phasedPool = MutateHelpers.applyPhasedComplexityForSelect(
    basePool,
    internal,
  );
  const adaptedPool = MutateHelpers.applyOperatorAdaptationForSelect(
    phasedPool,
    internal,
  );

  // Step 3: select a candidate method from the pool.
  const sampledMethod = MutateHelpers.sampleFromPoolForSelect(
    adaptedPool,
    internal,
  );
  if (!sampledMethod) return null;
  if (
    MutateHelpers.isBlockedByStructuralLimitsForSelect(
      sampledMethod,
      genome,
      internal,
      methods,
    )
  )
    return null;

  // Step 4: apply operator bandit selection when enabled.
  const banditMethod = MutateHelpers.applyOperatorBanditForSelect(
    adaptedPool,
    sampledMethod,
    internal,
  );
  if (
    MutateHelpers.isBlockedByRecurrentPolicyForSelect(
      banditMethod,
      internal,
      methods,
    )
  )
    return null;

  return banditMethod;
}
