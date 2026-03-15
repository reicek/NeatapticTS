/**
 * Shared contracts for the NEAT mutation chapter.
 *
 * These types keep the mutation subtree decoupled from the full controller and
 * network implementations while the direct-path split continues. Every chapter
 * under `mutation/` depends on this file as a leaf contract surface.
 */

/**
 * Runtime interface for a genome with mutation-related metadata.
 * Avoids circular dependencies by defining only the properties accessed in mutation modules.
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
 */
export interface ConnectionWithMetadata {
  /** Source node of the directed edge. */
  from: NodeWithMetadata;
  /** Target node of the directed edge. */
  to: NodeWithMetadata;
  /** Weight applied along the connection. */
  weight: number;
  /** Whether the connection is active. */
  enabled?: boolean;
  /** Innovation identifier for NEAT alignment. */
  innovation?: number;
}

/**
 * Runtime interface for a mutation method descriptor.
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
 */
export interface OperatorStats {
  /** Count of times the operator improved structure. */
  success: number;
  /** Total times the operator was attempted. */
  attempts: number;
}

/**
 * Runtime interface for node-split innovation records.
 */
export interface NodeSplitRecord {
  /** Gene id assigned to the inserted node. */
  newNodeGeneId: number;
  /** Innovation id for the incoming split connection. */
  inInnov: number;
  /** Innovation id for the outgoing split connection. */
  outInnov: number;
}

/**
 * Runtime interface for the NEAT controller used in mutation operations.
 * Avoids circular dependencies by defining only properties accessed in mutation modules.
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
  /** Cache of node split innovation records. */
  _nodeSplitInnovations: Map<string, NodeSplitRecord>;
  /** Cache of connection innovation ids. */
  _connInnovations: Map<string, number>;
  /** Next global innovation id. */
  _nextGlobalInnovation: number;
  /** Current phased-complexity mode. */
  _phase?: 'simplify' | 'complexify';
  /** Computes minimum hidden node target. */
  getMinimumHiddenSize?: (multiplierOverride?: number) => number;
}
