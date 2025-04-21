import Activation from './activation';

/**
 * Mutation methods for genetic algorithms.
 *
 * Mutation introduces diversity into the population by altering genes.
 * Strategies like adding/removing nodes, modifying weights, and changing
 * activation functions enable adaptation to complex datasets.
 *
 * Mutation is inspired by biological evolution and is essential for
 * escaping local optima in optimization problems.
 *
 * These methods implement the mutation strategies described in the Instinct algorithm,
 * including adding/removing nodes and connections, modifying weights and biases, and
 * changing activation functions. These mutations allow the genome to evolve dynamically
 * and adapt to complex datasets.
 *
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
 * @see {@link https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)}
 * @see {@link https://en.wikipedia.org/wiki/Evolutionary_algorithm}
 */
export const mutation: { [key: string]: any } = {
  ADD_NODE: {
    /** Adds a new node to the network by splitting an existing connection. */
    name: 'ADD_NODE',
    /**
     * Inserts a new node between two connected nodes, creating two new connections.
     * @see Instinct Algorithm - Section 3.1 Add Node Mutation
     */
  },
  SUB_NODE: {
    /** Removes an existing node from the network. */
    name: 'SUB_NODE',
    keep_gates: true,
    /**
     * Removes a hidden node and reconnects its incoming and outgoing connections.
     * @see Instinct Algorithm - Section 3.7 Remove Node Mutation
     */
  },
  ADD_CONN: {
    /** Adds a new connection between nodes. */
    name: 'ADD_CONN',
    /**
     * Generates a list of all possible connections that do not yet exist and adds one randomly.
     * @see Instinct Algorithm - Section 3.2 Add Connection Mutation
     */
  },
  SUB_CONN: {
    /** Removes an existing connection between nodes. */
    name: 'SUB_CONN',
    /**
     * Ensures that nodes retain at least one incoming and one outgoing connection.
     * @see Instinct Algorithm - Section 3.8 Remove Connection Mutation
     */
  },
  MOD_WEIGHT: {
    /** Modifies the weight of an existing connection. */
    name: 'MOD_WEIGHT',
    min: -1,
    max: 1,
    /**
     * Adjusts the weight of a connection by adding a random value within a fixed range.
     * @see Instinct Algorithm - Section 3.4 Modify Weight Mutation
     */
  },
  MOD_BIAS: {
    /** Modifies the bias of a node. */
    name: 'MOD_BIAS',
    min: -1,
    max: 1,
    /**
     * Adjusts the bias of a node by adding a random value within a fixed range.
     * @see Instinct Algorithm - Section 3.5 Modify Bias Mutation
     */
  },
  MOD_ACTIVATION: {
    /** Changes the activation function of a node. */
    name: 'MOD_ACTIVATION',
    mutateOutput: true,
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
    ],
    /**
     * Allows nodes to mutate their activation functions, enabling adaptation to different datasets.
     * @see Instinct Algorithm - Section 3.6 Modify Squash Mutation
     */
  },
  ADD_SELF_CONN: {
    /** Adds a self-connection to a node. */
    name: 'ADD_SELF_CONN',
  },
  SUB_SELF_CONN: {
    /** Removes a self-connection from a node. */
    name: 'SUB_SELF_CONN',
  },
  ADD_GATE: {
    /** Adds a gate to a connection. */
    name: 'ADD_GATE',
  },
  SUB_GATE: {
    /** Removes a gate from a connection. */
    name: 'SUB_GATE',
  },
  ADD_BACK_CONN: {
    /** Adds a recurrent connection to the network. */
    name: 'ADD_BACK_CONN',
  },
  SUB_BACK_CONN: {
    /** Removes a recurrent connection from the network. */
    name: 'SUB_BACK_CONN',
  },
  SWAP_NODES: {
    /** Swaps two nodes in the network. */
    name: 'SWAP_NODES',
    mutateOutput: true,
  },
  ALL: [],
  FFW: [],
};

/**
 * List of all mutation methods.
 * Includes all possible mutations for neural networks.
 */
mutation.ALL = [
  mutation.ADD_NODE,
  mutation.SUB_NODE,
  mutation.ADD_CONN,
  mutation.SUB_CONN,
  mutation.MOD_WEIGHT,
  mutation.MOD_BIAS,
  mutation.MOD_ACTIVATION,
  mutation.ADD_GATE,
  mutation.SUB_GATE,
  mutation.ADD_SELF_CONN,
  mutation.SUB_SELF_CONN,
  mutation.ADD_BACK_CONN,
  mutation.SUB_BACK_CONN,
  mutation.SWAP_NODES,
];

/**
 * List of feedforward-compatible mutation methods.
 * Excludes mutations incompatible with feedforward networks.
 */
mutation.FFW = [
  mutation.ADD_NODE,
  mutation.SUB_NODE,
  mutation.ADD_CONN,
  mutation.SUB_CONN,
  mutation.MOD_WEIGHT,
  mutation.MOD_BIAS,
  mutation.MOD_ACTIVATION,
  mutation.SWAP_NODES,
];

export default mutation;
