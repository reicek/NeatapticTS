import type { InnovationTracker } from '../../innovation-tracker/innovation-tracker.types';

/**
 * Shared contracts for the NEAT mutation chapter.
 *
 * These types keep the mutation subtree decoupled from the full controller and
 * network implementations while the direct-path split continues. Every chapter
 * under `mutation/` depends on this file as a leaf contract surface.
 *
 * This file is the contract map for the whole mutation subtree. It explains
 * which parts of a genome mutation helpers are allowed to touch, which pieces
 * of controller state they can rely on, and which historical stores exist to
 * preserve structural identity across generations.
 *
 * The contracts fall into four groups:
 *
 * 1. runtime genome shape: `GenomeWithMetadata`, `NodeWithMetadata`, and
 *    `ConnectionWithMetadata`,
 * 2. operator description and adaptation state: `MutationMethod` and
 *    `OperatorStats`,
 * 3. controller-owned innovation tracking: `InnovationTracker`,
 * 4. host controller seam: `NeatControllerForMutation`.
 *
 * Read this chapter before diving into the helper folders when you need to know
 * which state is local to a genome, which state belongs to the controller, and
 * which caches or innovation stores mutation helpers are allowed to mutate.
 *
 * ```mermaid
 * flowchart TD
 *   Genome[Genome runtime shape] --> Nodes[Node and connection metadata]
 *   Genome --> Operators[Mutation method descriptors]
 *   Operators --> Stats[Operator success tracking]
 *   Genome --> Controller[Mutation host controller seam]
 *   Controller --> Tracker[Generation-scoped innovation tracker]
 *   Controller --> Flow[Used by flow, select, add-node, and add-conn helpers]
 * ```
 */

/**
 * Runtime interface for a genome with mutation-related metadata.
 *
 * This is the central runtime contract for the mutation subtree. It is smaller
 * than the full network implementation on purpose: helpers only see the node
 * list, connection list, a few topology flags, and the mutation-oriented hooks
 * they actually need to perform structural edits.
 *
 * The shape makes an architectural distinction explicit. Per-genome mutable
 * state such as `_mutRate` and `_mutAmount` travels with the genome itself,
 * while global innovation tables and policy controls stay on the controller
 * host contract.
 */
export interface GenomeWithMetadata {
  /** Ordered list of nodes in the genome. */
  nodes: NodeWithMetadata[];
  /** All directed connections between nodes. */
  connections: ConnectionWithMetadata[];
  /** Gate metadata used by gated connections. */
  gates: unknown[];
  /** Count of input nodes in the genome. */
  input: number;
  /** Count of output nodes in the genome. */
  output: number;
  /** Flag indicating whether cycles are disallowed. */
  _enforceAcyclic?: boolean;
  /** Optional public topology-intent accessor used by policy bridges. */
  getTopologyIntent?: () => 'feed-forward' | 'unconstrained';
  /** Per-genome adaptive mutation rate. */
  _mutRate?: number;
  /** Per-genome adaptive mutation amount. */
  _mutAmount?: number;
  /** Applies a mutation operator to the genome. */
  mutate?: (method: MutationMethod) => void;
  /** Creates one or more connections between nodes. */
  connect?: (
    from: NodeWithMetadata,
    to: NodeWithMetadata,
    weight?: number,
  ) => ConnectionWithMetadata[];
  /** Removes a connection between nodes. */
  disconnect?: (from: NodeWithMetadata, to: NodeWithMetadata) => void;
}

/**
 * Runtime interface for a node within a genome.
 *
 * Mutation helpers only need a narrow node view: topology category, optional
 * gene identity, adjacency lists, and one projection check. That narrowness is
 * what lets the structural chapters reason about connection growth and cycle
 * checks without depending on the whole node implementation.
 */
export interface NodeWithMetadata {
  /** Node category used by topology constraints. */
  type: 'input' | 'output' | 'hidden';
  /** Stable identifier used for innovation alignment. */
  geneId?: number;
  /** Incoming and outgoing connection lists. */
  connections: {
    /** Incoming connections to this node. */
    in: ConnectionWithMetadata[];
    /** Outgoing connections from this node. */
    out: ConnectionWithMetadata[];
  };
  /** Checks whether this node already projects to a target. */
  isProjectingTo?: (target: NodeWithMetadata) => boolean;
}

/**
 * Runtime interface for a connection within a genome.
 *
 * This is the minimum edge metadata required by the structural mutation paths.
 * The add-node and add-conn chapters use it to preserve weights, inspect edge
 * activation, and attach innovation ids that later support alignment-based
 * crossover and speciation.
 */
export interface ConnectionWithMetadata {
  /** Source node of the directed edge. */
  from: NodeWithMetadata;
  /** Target node of the directed edge. */
  to: NodeWithMetadata;
  /** Weight applied along the connection. */
  weight: number;
  /** Gater node when this edge is multiplicatively controlled. */
  gater?: NodeWithMetadata | null;
  /** Whether the connection is active. */
  enabled?: boolean;
  /** Innovation identifier for NEAT alignment. */
  innovation?: number;
}

/**
 * Runtime interface for a mutation method descriptor.
 *
 * Mutation operators are represented as descriptive runtime objects rather than
 * as enum literals alone. That gives selection helpers room to reason about
 * operator families, sample-count hints, and output-mutation behavior without
 * having to know the concrete implementation of each operator.
 */
export interface MutationMethod {
  /** Symbolic operator name (e.g., ADD_NODE). */
  name: string;
  /** Minimum number of operators to sample in a policy. */
  min?: number;
  /** Maximum number of operators to sample in a policy. */
  max?: number;
  /** Whether existing gates should be preserved. */
  keep_gates?: boolean;
  /** Whether mutation may alter output structure. */
  mutateOutput?: boolean;
  /** Optional predicate list to filter candidate values. */
  allowed?: Array<(x: number) => number>;
}

/**
 * Runtime interface for operator statistics tracking.
 *
 * These counters support the adaptive side of mutation policy. They do not try
 * to capture full fitness impact; instead they record a cheap local proxy for
 * whether an attempted operator actually produced structure often enough to be
 * favored later by adaptation or bandit logic.
 */
export interface OperatorStats {
  /** Count of times the operator improved structure. */
  success: number;
  /** Total times the operator was attempted. */
  attempts: number;
}

/**
 * Runtime interface for the NEAT controller used in mutation operations.
 *
 * This host contract is the bridge between local genome edits and global NEAT
 * state. It exposes configuration, RNG access, innovation stores, operator
 * statistics, and the narrow controller callbacks that the mutation helpers need.
 *
 * The important boundary is that mutation chapters can mutate controller-owned
 * bookkeeping through this interface, but they do not need the entire `Neat`
 * class surface to do their work.
 */
export interface NeatControllerForMutation {
  /** Current population of genomes. */
  population: GenomeWithMetadata[];
  /** Mutation and constraint configuration. */
  options: {
    /** Adaptive mutation configuration (per-genome rates/amounts). */
    adaptiveMutation?: {
      /** Whether adaptive mutation is enabled. */
      enabled?: boolean;
      /** Initial per-genome mutation rate if not yet assigned. */
      initialRate?: number;
      /** Whether per-genome mutation amounts are adapted. */
      adaptAmount?: boolean;
    };
    /** Global mutation rate if adaptive mutation is disabled. */
    mutationRate?: number;
    /** Global mutation amount if adaptive mutation is disabled. */
    mutationAmount?: number;
    /** Mutation operator policy (array or legacy descriptors). */
    mutation?: MutationMethod[] | unknown;
    /** Phased complexity configuration. */
    phasedComplexity?: {
      /** Whether phased complexity is enabled. */
      enabled?: boolean;
    };
    /** Operator adaptation configuration. */
    operatorAdaptation?: {
      /** Whether operator adaptation is enabled. */
      enabled?: boolean;
      /** Boost factor for successful operators. */
      boost?: number;
    };
    /** Operator bandit configuration. */
    operatorBandit?: {
      /** Whether operator bandit selection is enabled. */
      enabled?: boolean;
      /** Exploration coefficient. */
      c?: number;
      /** Minimum attempts before normalizing operator scores. */
      minAttempts?: number;
    };
    /** Maximum number of nodes allowed in a genome. */
    maxNodes?: number;
    /** Maximum number of gates allowed in a genome. */
    maxGates?: number;
    /** Maximum number of connections allowed in a genome. */
    maxConns?: number;
    /** Whether recurrent connections are permitted. */
    allowRecurrent?: boolean;
  };
  /** Returns the RNG function used by the controller. */
  _getRNG: () => () => number;
  /** Picks a mutation operator for a genome. */
  selectMutationMethod: (
    genome: GenomeWithMetadata,
    crossover: boolean,
  ) => Promise<MutationMethod | MutationMethod[] | null>;
  /** Adds a node with innovation reuse. */
  _mutateAddNodeReuse: (genome: GenomeWithMetadata) => Promise<void>;
  /** Adds a connection with innovation reuse. */
  _mutateAddConnReuse: (genome: GenomeWithMetadata) => void;
  /** Clears cached connection structures. */
  _invalidateGenomeCaches: (genome: GenomeWithMetadata) => void;
  /** Per-operator success/attempt statistics. */
  _operatorStats: Map<string, OperatorStats>;
  /** Explicit innovation tracker for structural mutation identity. */
  _innovationTracker: InnovationTracker;
  /** Current phased-complexity mode. */
  _phase?: 'simplify' | 'complexify';
  /** Computes minimum hidden node target. */
  getMinimumHiddenSize?: (multiplierOverride?: number) => number;
}
