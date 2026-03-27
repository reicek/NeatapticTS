import Activation from '../activation/activation';

/**
 * Configuration shape for one mutation operator.
 *
 * Each mutation method carries a small policy object describing what kind of
 * structural or parametric change it performs and the narrow knobs that shape
 * that change. Read the fields as metadata for the evolutionary controller,
 * not as a full runtime implementation.
 */
interface MutationConfig {
  name: string;
  min?: number;
  max?: number;
  keep_gates?: boolean;
  mutateOutput?: boolean;
  allowed?: Array<(x: number) => number>;
  [key: string]: unknown;
}

/**
 * Defines various mutation methods used in neuroevolution algorithms.
 *
 * Mutation introduces genetic diversity into the population by randomly
 * altering parts of an individual's genome (the neural network structure or parameters).
 * This is crucial for exploring the search space and escaping local optima.
 *
 * Common mutation strategies include adding or removing nodes and connections,
 * modifying connection weights and node biases, and changing node activation functions.
 * These operations allow the network topology and parameters to adapt over generations.
 *
 * The methods listed here are inspired by techniques used in algorithms like NEAT
 * and particularly the Instinct algorithm, providing a comprehensive set of tools
 * for evolving network architectures.
 *
 * Read this file as a mutation toolbox organized by what kind of change you
 * want evolution to make:
 *
 * - topology-growth operators such as `ADD_NODE`, `ADD_CONN`,
 *   `ADD_SELF_CONN`, and `ADD_BACK_CONN` make the graph more expressive,
 * - topology-pruning operators such as `SUB_NODE`, `SUB_CONN`,
 *   `SUB_SELF_CONN`, and `SUB_BACK_CONN` remove structure and can simplify an
 *   overgrown search,
 * - parameter-tuning operators such as `MOD_WEIGHT`, `MOD_BIAS`, and
 *   `REINIT_WEIGHT` change numeric behavior without rewriting the graph,
 * - behavior-shaping operators such as `MOD_ACTIVATION`, `ADD_GATE`,
 *   `SUB_GATE`, and `SWAP_NODES` change how existing structure computes,
 * - architecture-expansion operators such as `ADD_LSTM_NODE` and
 *   `ADD_GRU_NODE` introduce memory-oriented building blocks.
 *
 * A practical reading order is:
 *
 * 1. start with `MOD_WEIGHT` and `MOD_BIAS` to understand the gentlest search
 *    moves,
 * 2. then compare `ADD_CONN` and `ADD_NODE` to see how structure starts to
 *    grow,
 * 3. then read the recurrent and gating operators when you want temporal
 *    behavior or context-sensitive routing,
 * 4. finish with `ALL` and `FFW`, which summarize which operators belong in a
 *    broad search versus a strictly feedforward one.
 *
 * A practical chooser for first experiments:
 *
 * - begin with weight and bias mutations when the topology is already plausible
 *   and you mainly want numeric refinement,
 * - allow `ADD_CONN` and `ADD_NODE` when the current architecture feels too
 *   rigid or too shallow,
 * - enable gating or back-connections only when temporal memory or dynamic
 *   routing is actually part of the task,
 * - prefer `FFW` as the safe shelf when a run must remain strictly
 *   feedforward.
 *
 * ```mermaid
 * flowchart TD
 *   Mutation[Mutation toolbox] --> Grow[Grow structure]
 *   Mutation --> Prune[Prune structure]
 *   Mutation --> Tune[Tune parameters]
 *   Mutation --> Shape[Reshape behavior]
 *   Mutation --> Memory[Add memory blocks]
 *   Grow --> GrowItems[ADD_NODE ADD_CONN ADD_SELF_CONN ADD_BACK_CONN]
 *   Prune --> PruneItems[SUB_NODE SUB_CONN SUB_SELF_CONN SUB_BACK_CONN]
 *   Tune --> TuneItems[MOD_WEIGHT MOD_BIAS REINIT_WEIGHT]
 *   Shape --> ShapeItems[MOD_ACTIVATION ADD_GATE SUB_GATE SWAP_NODES]
 *   Memory --> MemoryItems[ADD_LSTM_NODE ADD_GRU_NODE]
 * ```
 *
 * Minimal workflow:
 *
 * ```ts
 * const safeFeedforwardShelf = mutation.FFW;
 *
 * const structuralSearchShelf = [
 *   mutation.ADD_CONN,
 *   mutation.ADD_NODE,
 *   mutation.MOD_WEIGHT,
 *   mutation.MOD_BIAS,
 * ];
 *
 * const recurrentSearchShelf = [
 *   ...structuralSearchShelf,
 *   mutation.ADD_GATE,
 *   mutation.ADD_BACK_CONN,
 * ];
 * ```
 *
 * Supported mutation families:
 *
 * - `ADD_NODE`: Adds a new node by splitting an existing connection.
 * - `SUB_NODE`: Removes a hidden node and its connections.
 * - `ADD_CONN`: Adds a new connection between two unconnected nodes.
 * - `SUB_CONN`: Removes an existing connection.
 * - `MOD_WEIGHT`: Modifies the weight of an existing connection.
 * - `MOD_BIAS`: Modifies the bias of a node.
 * - `MOD_ACTIVATION`: Changes the activation function of a node.
 * - `ADD_SELF_CONN`: Adds a self-connection (recurrent loop) to a node.
 * - `SUB_SELF_CONN`: Removes a self-connection from a node.
 * - `ADD_GATE`: Adds a gating mechanism to a connection.
 * - `SUB_GATE`: Removes a gating mechanism from a connection.
 * - `ADD_BACK_CONN`: Adds a recurrent (backward) connection between nodes.
 * - `SUB_BACK_CONN`: Removes a recurrent (backward) connection.
 * - `SWAP_NODES`: Swaps the roles (bias and activation) of two nodes.
 * - `REINIT_WEIGHT`: Reinitializes all weights for a node.
 * - `BATCH_NORM`: Marks a node for batch normalization (stub).
 * - `ADD_LSTM_NODE`: Adds a new LSTM node (memory cell with gates).
 * - `ADD_GRU_NODE`: Adds a new GRU node (gated recurrent unit).
 *
 * Summary shelves:
 * - `ALL`: all mutation methods, including recurrent and memory-oriented ones.
 * - `FFW`: the feedforward-safe subset that avoids recurrence and gating.
 *
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
 * @see {@link https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm) Mutation (Genetic Algorithm) - Wikipedia}
 * @see {@link https://en.wikipedia.org/wiki/Neuroevolution Neuroevolution - Wikipedia}
 * @see {@link http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf NEAT Paper (Relevant concepts)}
 */
export const mutation: {
  ADD_NODE: MutationConfig;
  SUB_NODE: MutationConfig;
  ADD_CONN: MutationConfig;
  SUB_CONN: MutationConfig;
  MOD_WEIGHT: MutationConfig;
  MOD_BIAS: MutationConfig;
  MOD_ACTIVATION: MutationConfig;
  ADD_GATE: MutationConfig;
  SUB_GATE: MutationConfig;
  ADD_SELF_CONN: MutationConfig;
  SUB_SELF_CONN: MutationConfig;
  ADD_BACK_CONN: MutationConfig;
  SUB_BACK_CONN: MutationConfig;
  SWAP_NODES: MutationConfig;
  REINIT_WEIGHT: MutationConfig;
  BATCH_NORM: MutationConfig;
  ADD_LSTM_NODE: MutationConfig;
  ADD_GRU_NODE: MutationConfig;
  ALL: MutationConfig[];
  FFW: MutationConfig[];
} = {
  /**
   * Adds a new node to the network by splitting an existing connection.
   * The original connection is disabled, and two new connections are created:
   * one from the original source to the new node, and one from the new node
   * to the original target. This increases network complexity, potentially
   * allowing for more sophisticated computations.
   */
  ADD_NODE: {
    name: 'ADD_NODE',
    /**
     * @see Instinct Algorithm - Section 3.1 Add Node Mutation
     */
  },
  /**
   * Removes a hidden node from the network. Connections to and from the
   * removed node are also removed. This simplifies the network topology.
   */
  SUB_NODE: {
    name: 'SUB_NODE',
    /** If true, attempts to preserve gating connections associated with the removed node. */
    keep_gates: true,
    /**
     * @see Instinct Algorithm - Section 3.7 Remove Node Mutation
     */
  },
  /**
   * Adds a new connection between two previously unconnected nodes.
   * This increases network connectivity, potentially creating new pathways
   * for information flow.
   */
  ADD_CONN: {
    name: 'ADD_CONN',
    /**
     * @see Instinct Algorithm - Section 3.2 Add Connection Mutation
     */
  },
  /**
   * Removes an existing connection between two nodes.
   * This prunes the network, potentially removing redundant or detrimental pathways.
   */
  SUB_CONN: {
    name: 'SUB_CONN',
    /**
     * @see Instinct Algorithm - Section 3.8 Remove Connection Mutation
     */
  },
  /**
   * Modifies the weight of an existing connection by adding a random value
   * or multiplying by a random factor. This fine-tunes the strength of
   * the connection.
   */
  MOD_WEIGHT: {
    name: 'MOD_WEIGHT',
    /** Minimum value for the random modification factor/offset. */
    min: -1,
    /** Maximum value for the random modification factor/offset. */
    max: 1,
    /**
     * @see Instinct Algorithm - Section 3.4 Modify Weight Mutation
     */
  },
  /**
   * Modifies the bias of a node (excluding input nodes) by adding a random value.
   * This adjusts the node's activation threshold, influencing its firing behavior.
   */
  MOD_BIAS: {
    name: 'MOD_BIAS',
    /** Minimum value for the random modification offset. */
    min: -1,
    /** Maximum value for the random modification offset. */
    max: 1,
    /**
     * @see Instinct Algorithm - Section 3.5 Modify Bias Mutation
     */
  },
  /**
   * Randomly changes the activation function of a node (excluding input nodes).
   * This allows nodes to specialize their response characteristics during evolution.
   */
  MOD_ACTIVATION: {
    name: 'MOD_ACTIVATION',
    /** If true, allows mutation of activation functions in output nodes. */
    mutateOutput: true,
    /** A list of allowed activation functions to choose from during mutation. */
    allowed: [
      Activation.logistic,
      Activation.tanh,
      Activation.relu,
      Activation.identity,
      Activation.step,
      Activation.softsign,
      Activation.sinusoid,
      Activation.gaussian,
      Activation.bentIdentity,
      Activation.bipolar,
      Activation.bipolarSigmoid,
      Activation.hardTanh,
      Activation.absolute,
      Activation.inverse,
      Activation.selu,
      Activation.softplus,
      Activation.swish,
      Activation.gelu,
      Activation.mish,
    ],
    /**
     * @see Instinct Algorithm - Section 3.6 Modify Squash Mutation
     */
  },
  /**
   * Adds a self-connection (recurrent connection from a node to itself).
   * This allows a node to retain information about its previous state,
   * introducing memory capabilities at the node level. Only applicable
   * to hidden and output nodes.
   */
  ADD_SELF_CONN: {
    name: 'ADD_SELF_CONN',
  },
  /**
   * Removes a self-connection from a node.
   * This removes the node's direct recurrent loop.
   */
  SUB_SELF_CONN: {
    name: 'SUB_SELF_CONN',
  },
  /**
   * Adds a gating mechanism to an existing connection. A new node (the gater)
   * is selected to control the flow of information through the gated connection.
   * This introduces multiplicative interactions, similar to LSTM or GRU units,
   * enabling more complex temporal processing or conditional logic.
   */
  ADD_GATE: {
    name: 'ADD_GATE',
  },
  /**
   * Removes a gating mechanism from a connection.
   * This simplifies the network by removing the modulatory influence of the gater node.
   */
  SUB_GATE: {
    name: 'SUB_GATE',
  },
  /**
   * Adds a recurrent connection between two nodes, potentially creating cycles
   * in the network graph (e.g., connecting a node to a node in a previous layer
   * or a non-adjacent node). This enables the network to maintain internal state
   * and process temporal dependencies.
   */
  ADD_BACK_CONN: {
    name: 'ADD_BACK_CONN',
  },
  /**
   * Removes a recurrent connection (that is not a self-connection).
   * This simplifies the recurrent topology of the network.
   */
  SUB_BACK_CONN: {
    name: 'SUB_BACK_CONN',
  },
  /**
   * Swaps the roles (bias and activation function) of two nodes (excluding input nodes).
   * Connections are generally preserved relative to the node indices.
   * This mutation alters the network's internal processing without changing
   * the overall node count or connection density.
   */
  SWAP_NODES: {
    name: 'SWAP_NODES',
    /** If true, allows swapping involving output nodes. */
    mutateOutput: true,
  },
  /**
   * Reinitializes the weights of all incoming, outgoing, and self connections for a node.
   * This can help escape local minima or inject diversity during evolution.
   */
  REINIT_WEIGHT: {
    name: 'REINIT_WEIGHT',
    /** Range for random reinitialization. */
    min: -1,
    max: 1,
  },
  /**
   * Marks a node for batch normalization. (Stub: actual normalization requires architectural support.)
   * This mutation can be used to toggle batch normalization on a node or layer.
   */
  BATCH_NORM: {
    name: 'BATCH_NORM',
  },
  /**
   * Adds a new LSTM node (memory cell with gates) to the network.
   * This enables the evolution of memory-augmented architectures.
   */
  ADD_LSTM_NODE: {
    name: 'ADD_LSTM_NODE',
    // Additional config can be added here if needed
  },
  /**
   * Adds a new GRU node (gated recurrent unit) to the network.
   * This enables the evolution of memory-augmented architectures.
   */
  ADD_GRU_NODE: {
    name: 'ADD_GRU_NODE',
    // Additional config can be added here if needed
  },
  /** Placeholder for the list of all mutation methods. */
  ALL: [],
  /** Placeholder for the list of mutation methods suitable for feedforward networks. */
  FFW: [],
};

/**
 * A list containing all defined mutation methods.
 *
 * Use this when the search is allowed to explore the whole mutation vocabulary,
 * including recurrent, gating, and memory-oriented operators.
 */
mutation.ALL = [
  mutation.ADD_NODE as MutationConfig,
  mutation.SUB_NODE as MutationConfig,
  mutation.ADD_CONN as MutationConfig,
  mutation.SUB_CONN as MutationConfig,
  mutation.MOD_WEIGHT as MutationConfig,
  mutation.MOD_BIAS as MutationConfig,
  mutation.MOD_ACTIVATION as MutationConfig,
  mutation.ADD_GATE as MutationConfig,
  mutation.SUB_GATE as MutationConfig,
  mutation.ADD_SELF_CONN as MutationConfig,
  mutation.SUB_SELF_CONN as MutationConfig,
  mutation.ADD_BACK_CONN as MutationConfig,
  mutation.SUB_BACK_CONN as MutationConfig,
  mutation.SWAP_NODES as MutationConfig,
  mutation.REINIT_WEIGHT as MutationConfig,
  mutation.BATCH_NORM as MutationConfig,
  mutation.ADD_LSTM_NODE as MutationConfig, // Added
  mutation.ADD_GRU_NODE as MutationConfig, // Added
];

/**
 * A list containing mutation methods suitable for purely feedforward networks.
 * Excludes mutations that introduce recurrence (ADD_SELF_CONN, ADD_BACK_CONN, ADD_GATE)
 * and related removal operations (SUB_SELF_CONN, SUB_BACK_CONN, SUB_GATE),
 * as these would violate the feedforward structure.
 *
 * This is the safest default shelf when the user intent is clearly feedforward
 * and the runtime should never grow temporal loops or gated recurrent paths.
 */
mutation.FFW = [
  mutation.ADD_NODE as MutationConfig,
  mutation.SUB_NODE as MutationConfig,
  mutation.ADD_CONN as MutationConfig,
  mutation.SUB_CONN as MutationConfig,
  mutation.MOD_WEIGHT as MutationConfig,
  mutation.MOD_BIAS as MutationConfig,
  mutation.MOD_ACTIVATION as MutationConfig,
  mutation.SWAP_NODES as MutationConfig,
  mutation.REINIT_WEIGHT as MutationConfig,
  mutation.BATCH_NORM as MutationConfig,
];

export default mutation;
