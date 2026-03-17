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

/**
 * Default connection weight used when mutation must create a structural edge from scratch.
 *
 * This keeps bootstrap connections and split in-edges deterministic at the
 * mutation boundary before later weight mutations or evaluation passes tune the
 * value more precisely.
 */
export const DEFAULT_CONNECTION_WEIGHT = 1;
/**
 * Default gene id used when mutation needs a stable fallback for node metadata.
 *
 * The value is intentionally simple because it acts as compatibility padding,
 * not as a semantic innovation marker.
 */
export const DEFAULT_GENE_ID = 0;
/**
 * Default innovation id used when a connection lacks recorded innovation metadata.
 *
 * This fallback prevents root mutation helpers from depending on missing ids
 * while the real innovation-tracking paths decide whether to reuse or allocate
 * new structural records.
 */
export const DEFAULT_INNOVATION_ID = 0;

/**
 * Root orchestration for NEAT mutation operations.
 *
 * Mutation is the controller's "change the structure on purpose" chapter.
 * It owns the whole-population edit pass that happens after scoring and before
 * the next generation settles into its new topology.
 *
 * The root file answers four controller-facing questions:
 *
 * - when should each genome be offered mutation work at all?
 * - how is one concrete operator chosen from the configured policy?
 * - when should structural edits reuse innovation history instead of inventing
 *   brand-new ids?
 * - which maintenance repairs run so newly mutated genomes stay usable by the
 *   next evaluation, speciation, and crossover passes?
 *
 * The root chapter stays orchestration-first because callers usually need the
 * whole structure-editing story, not just one isolated operator. The helper
 * folders own the narrower mechanics:
 *
 * - `flow/` runs the per-genome mutation loop and keeps operator-side effects coherent.
 * - `select/` resolves which operator is even allowed or favored right now.
 * - `add-node/` and `add-conn/` own structural reuse and innovation bookkeeping.
 * - `repair/` keeps mutated networks connected enough to remain valid training candidates.
 *
 * Read this root chapter when you want the controller view of mutation.
 * Follow `flow/` for the actual per-genome loop, `select/` for policy and
 * bandit-weighted operator choice, and `repair/` when you need to understand
 * why mutation sometimes adds maintenance edges after the main structural edit.
 *
 * ```mermaid
 * flowchart TD
 *   Start["generation ready for structure edits"] --> Mutate["mutate()\nwalk every genome"]
 *   Mutate --> Select["selectMutationMethod()\nresolve allowed operator"]
 *   Select --> Operator{"Which structural path?"}
 *   Operator -->|ADD_NODE| AddNode["mutateAddNodeReuse()\nreuse or allocate split innovations"]
 *   Operator -->|ADD_CONN| AddConn["mutateAddConnReuse()\nreuse or allocate connection innovations"]
 *   Operator -->|repair needed| Repair["ensureMinHiddenNodes() / ensureNoDeadEnds()"]
 *   AddNode --> Population["genome structure updated in place"]
 *   AddConn --> Population
 *   Repair --> Population
 *   Population --> Bookkeeping["innovation tables, caches, and operator stats stay aligned"]
 * ```
 */

/**
 * Mutate every genome in the population according to configured policies.
 *
 * This is the controller's whole-population mutation pass. It does not decide
 * one fixed operator up front for the entire generation. Instead it walks the
 * current population, lets the lower-level flow helpers resolve each genome's
 * effective mutation policy, and then applies whatever operator mix is valid
 * for that specific genome at that moment.
 *
 * In practice, this is where several otherwise separate mutation concerns meet:
 * adaptive per-genome rates, structural innovation reuse, optional extra
 * structural exploration, and post-edit repair work. That is why the root
 * mutation chapter exists at all: the controller needs one place where those
 * per-genome edits stay coherent before later stages such as evaluation or
 * speciation inspect the changed population.
 *
 * Educational notes:
 * - Adaptive mutation allows per-genome mutation rates/amounts to evolve so
 *   that successful genomes can reduce or increase plasticity over time.
 * - Structural mutations (ADD_NODE, ADD_CONN, etc.) may update global
 *   innovation bookkeeping; this function attempts to reuse specialized
 *   helper routines that preserve innovation ids across the population.
 * - Cache invalidation and operator statistics also live downstream of this
 *   pass, so `mutate()` is about consistency as much as randomness.
 *
 * @example
 * ```ts
 * // Called after evaluation and before the next generation is consumed.
 * neat.mutate();
 * ```
 *
 * @returns Promise that resolves after every genome has gone through the mutation flow for this pass.
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
 * Use this helper when the controller wants a structural growth mutation that
 * stays compatible with prior history. The important state change is not only
 * the new hidden node inside one genome, but also the possible update to the
 * controller's split-innovation table when this exact split has never been seen
 * before.
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
 * @example
 * ```ts
 * neat._mutateAddNodeReuse(genome);
 * ```
 *
 * @this NeatLike - neat controller context (holds innovation tables)
 * @param genome - genome to modify in-place
 * @returns Promise that resolves after the split has either reused an existing innovation record or created a new one.
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
 * This is the connection-growth companion to `mutateAddNodeReuse()`. Its main
 * controller value is innovation consistency: if two genomes discover the same
 * structural pair across time, the mutation system tries to keep that edit
 * comparable for later crossover and speciation rather than treating it as a
 * completely unrelated event.
 *
 * @this NeatLike - neat controller context (holds innovation tables)
 * @param genome - genome to modify in-place
 * @returns Nothing. The genome may gain one new connection and the controller innovation map may be consulted or extended.
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
 *
 * This repair helper runs after structural edits when the controller wants to
 * keep a mutated genome above a minimum hidden-capacity floor. It is less about
 * exploration than about preserving a usable topology budget so later mutation,
 * evaluation, and selection steps do not inherit a trivially underbuilt graph.
 *
 * The helper may add hidden nodes, wire missing edges, and rebuild cached
 * connection structures, so callers should treat it as a topology-maintenance
 * pass rather than a tiny invariant check.
 *
 * @param network - Genome whose hidden-node budget and connectivity should be repaired.
 * @param multiplierOverride - Optional override for the configured hidden-node multiplier.
 * @returns Promise that resolves after hidden-node and connectivity repairs have completed.
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
 *
 * Mutation can produce temporarily awkward graphs, especially after structural
 * growth or pruning-like simplification. This repair pass reconnects stranded
 * input, output, or hidden nodes so the genome remains a sensible candidate for
 * later evaluation and does not carry obviously broken topology into the next
 * controller stage.
 *
 * @param network - Genome whose endpoint and hidden-node connectivity should be repaired.
 * @returns Nothing. The network may gain repair connections in place.
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
 *
 * This is the policy gateway between "a genome is eligible for mutation" and
 * "here is the concrete operator to apply." The selector folds together legacy
 * fast-feed-forward behavior, phased-complexity biasing, adaptive operator
 * weighting, structural safety checks, and optional bandit-style operator
 * choice so the rest of the mutation flow can work with one resolved answer.
 *
 * Mirrors legacy implementation from `neat.ts` to preserve test expectations.
 * `rawReturnForTest` retains historical behavior where the full FFW array is
 * returned for identity checks in tests.
 *
 * @example
 * ```ts
 * const method = await neat.selectMutationMethod(genome, false);
 * // The result already reflects policy gates such as phased complexity and
 * // structural limits, not just a random sample from the raw configured pool.
 * ```
 *
 * @param genome - Genome whose current structure constrains which operators are legal.
 * @param rawReturnForTest - Preserves legacy array-return behavior for test-only FFW checks.
 * @returns Resolved mutation method, legacy FFW array for compatibility tests, or `null` when no operator should run.
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
