import Activation from './activation';

/**
 * Mutation methods for genetic algorithms.
 * @see {@link https://en.wikipedia.org/wiki/mutation_(genetic_algorithm)}
 */
export const mutation: { [key: string]: any } = {
  ADD_NODE: {
    /** Adds a new node to the network. */
    name: 'ADD_NODE',
  },
  SUB_NODE: {
    /** Removes an existing node from the network. */
    name: 'SUB_NODE',
    keep_gates: true,
  },
  ADD_CONN: {
    /** Adds a new connection between nodes. */
    name: 'ADD_CONN',
  },
  SUB_CONN: {
    /** Removes an existing connection between nodes. */
    name: 'SUB_CONN',
  },
  MOD_WEIGHT: {
    /** Modifies the weight of an existing connection. */
    name: 'MOD_WEIGHT',
    min: -1,
    max: 1,
  },
  MOD_BIAS: {
    /** Modifies the bias of a node. */
    name: 'MOD_BIAS',
    min: -1,
    max: 1,
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
