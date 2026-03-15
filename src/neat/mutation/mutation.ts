import type { NeatLike } from '../shared/neat.shared.types';
import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
} from './shared/mutation.types';
import * as mutationAddConn from './add-conn/mutation.add-conn';
import * as mutationAddNode from './add-node/mutation.add-node';
import * as mutationFlow from './flow/mutation.flow';
import * as mutationDeadEnds from './repair/mutation.dead-ends';
import * as mutationMinHidden from './repair/mutation.min-hidden';
import * as mutationSelect from './select/mutation.select';

/** Default connection weight used for bootstrap and split in-edges. */
export const DEFAULT_CONNECTION_WEIGHT = 1;
/** Default gene id value when a node has no gene id. */
export const DEFAULT_GENE_ID = 0;
/** Default innovation id value when a connection has none. */
export const DEFAULT_INNOVATION_ID = 0;

/**
 * Root orchestration for NEAT mutation operations.
 *
 * This chapter keeps the public mutation flow readable: mutate every genome,
 * reuse structural innovations for add-node and add-connection operators,
 * repair minimum hidden-node structure, and preserve the stable mutation-method
 * selection surface used by the main `Neat` controller.
 *
 * The neighboring `flow/`, `select/`, `add-node/`, `add-conn/`, and
 * `repair/` chapters own the narrower mechanics.
 */

/**
 * Mutate every genome in the population according to configured policies.
 *
 * This is the high-level mutation driver used by NeatapticTS. It iterates the
 * current population and, depending on the configured mutation rate and
 * (optional) adaptive mutation controller, applies one or more mutation
 * operators to each genome. The sibling `maintenance/facade/` chapter keeps the
 * stable `Neat` class wrappers for maintenance-oriented callers, while this
 * mutation chapter continues to own the actual repair and mutation mechanics.
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
  const methods = await import('../../methods/methods');
  const population = internal.population;

  for (const genome of population) {
    await mutationFlow.mutateGenome(genome, internal, methods);
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
  mutationAddNode.ensureBootstrapConnection(genome, internal);

  // Step 2: pick an enabled connection to split.
  const enabledConnections = mutationAddNode.collectEnabledConnections(genome);
  const chosenConnection = mutationAddNode.chooseConnectionForSplit(
    enabledConnections,
    internal,
  );
  if (!chosenConnection) return;

  // Step 3: compute split metadata and remove the original connection.
  const splitDescriptor =
    mutationAddNode.buildSplitDescriptor(chosenConnection);
  mutationAddNode.disconnectOriginalConnection(genome, chosenConnection);

  // Step 4: resolve split record and create split node.
  const splitRecord = internal._nodeSplitInnovations.get(
    splitDescriptor.splitKey,
  );
  const { default: NodeClass } = await import('../../architecture/node');

  if (splitRecord) {
    mutationAddNode.applySplitWithExistingRecord(
      genome,
      chosenConnection,
      splitDescriptor,
      splitRecord,
      NodeClass,
    );
    return;
  }

  mutationAddNode.applySplitWithNewRecord(
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
  const candidatePairs = mutationAddConn.collectCandidatePairsForConn(genome);
  if (!candidatePairs.length) return;

  // Step 2: choose the candidate pool based on reuse and hidden-pair rules.
  const reuseCandidates = mutationAddConn.filterPairsWithInnovations(
    candidatePairs,
    internal,
  );
  const selectionPool = mutationAddConn.selectPairPool(
    candidatePairs,
    reuseCandidates,
  );

  // Step 3: pick a concrete pair to connect.
  const chosenPair = mutationAddConn.choosePairForConn(selectionPool, internal);
  if (!chosenPair) return;

  // Step 4: evaluate acyclic constraints if configured.
  const pairNodes = mutationAddConn.resolvePairNodes(chosenPair);
  if (mutationAddConn.shouldAbortForCycle(genome, pairNodes)) return;

  // Step 5: create the connection and assign an innovation id.
  const connection = mutationAddConn.connectChosenPair(genome, pairNodes);
  if (!connection) return;
  mutationAddConn.assignInnovationForConnection(
    connection,
    pairNodes,
    internal,
  );
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
  const nodeGroups = mutationMinHidden.collectNodeGroupsForMinHidden(network);
  const maxNodes = mutationMinHidden.resolveMaxNodesForMinHidden(internal);
  const minHidden = mutationMinHidden.resolveMinHiddenForMinHidden(
    network,
    maxNodes,
    multiplierOverride,
    internal,
  );

  // Step 2: validate network inputs/outputs.
  if (!mutationMinHidden.hasRequiredEndpointsForMinHidden(nodeGroups)) {
    mutationMinHidden.warnMissingEndpointsForMinHidden();
    return;
  }

  // Step 3: ensure the minimum number of hidden nodes exist.
  await mutationMinHidden.ensureHiddenNodeCountForMinHidden(
    network,
    nodeGroups,
    minHidden,
    maxNodes,
  );

  // Step 4: ensure hidden nodes have at least one incoming and outgoing edge.
  mutationMinHidden.ensureHiddenConnectivityForMinHidden(
    network,
    nodeGroups,
    internal,
  );

  // Step 5: rebuild cached connection structures.
  await mutationMinHidden.rebuildNetworkConnectionsForMinHidden(network);
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
  const nodeGroups = mutationDeadEnds.collectNodeGroupsForDeadEnds(network);

  // Step 2: repair input nodes that lack outgoing connections.
  mutationDeadEnds.ensureInputConnectivityForDeadEnds(
    network,
    nodeGroups,
    internal,
  );

  // Step 3: repair output nodes that lack incoming connections.
  mutationDeadEnds.ensureOutputConnectivityForDeadEnds(
    network,
    nodeGroups,
    internal,
  );

  // Step 4: repair hidden nodes with missing in/out connections.
  mutationDeadEnds.ensureHiddenConnectivityForDeadEnds(
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
  const methods = await import('../../methods/methods');

  // Step 1: resolve FFW-specific policy behavior.
  const ffwPolicy = mutationSelect.resolveFFWPolicyForSelect(
    internal,
    methods,
    rawReturnForTest,
  );
  if (ffwPolicy) return ffwPolicy;

  // Step 2: build the operator pool for selection.
  const basePool = mutationSelect.normalizeMutationPoolForSelect(
    internal,
    methods,
    rawReturnForTest,
  );
  const phasedPool = mutationSelect.applyPhasedComplexityForSelect(
    basePool,
    internal,
  );
  const adaptedPool = mutationSelect.applyOperatorAdaptationForSelect(
    phasedPool,
    internal,
  );

  // Step 3: select a candidate method from the pool.
  const sampledMethod = mutationSelect.sampleFromPoolForSelect(
    adaptedPool,
    internal,
  );
  if (!sampledMethod) return null;
  if (
    mutationSelect.isBlockedByStructuralLimitsForSelect(
      sampledMethod,
      genome,
      internal,
      methods,
    )
  ) {
    return null;
  }

  // Step 4: apply operator bandit selection when enabled.
  const banditMethod = mutationSelect.applyOperatorBanditForSelect(
    adaptedPool,
    sampledMethod,
    internal,
  );
  if (
    mutationSelect.isBlockedByRecurrentPolicyForSelect(
      banditMethod,
      internal,
      methods,
    )
  ) {
    return null;
  }

  return banditMethod;
}
