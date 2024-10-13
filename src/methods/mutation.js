import Activation from './Activation';

/** @see {@link https://en.wikipedia.org/wiki/mutation_(genetic_algorithm)} */
export const mutation = {
  ADD_NODE: {
    name: 'ADD_NODE',
  },
  SUB_NODE: {
    name: 'SUB_NODE',
    keep_gates: true,
  },
  ADD_CONN: {
    name: 'ADD_CONN',
  },
  SUB_CONN: {
    name: 'REMOVE_CONN',
  },
  MOD_WEIGHT: {
    name: 'MOD_WEIGHT',
    min: -1,
    max: 1,
  },
  MOD_BIAS: {
    name: 'MOD_BIAS',
    min: -1,
    max: 1,
  },
  MOD_ACTIVATION: {
    name: 'MOD_ACTIVATION',
    mutateOutput: true,
    allowed: [
      Activation.LOGISTIC,
      Activation.TANH,
      Activation.RELU,
      Activation.IDENTITY,
      Activation.STEP,
      Activation.SOFTSIGN,
      Activation.SINUSOID,
      Activation.GAUSSIAN,
      Activation.BENT_IDENTITY,
      Activation.BIPOLAR,
      Activation.BIPOLAR_SIGMOID,
      Activation.HARD_TANH,
      Activation.ABSOLUTE,
      Activation.INVERSE,
      Activation.SELU,
    ],
  },
  ADD_SELF_CONN: {
    name: 'ADD_SELF_CONN',
  },
  SUB_SELF_CONN: {
    name: 'SUB_SELF_CONN',
  },
  ADD_GATE: {
    name: 'ADD_GATE',
  },
  SUB_GATE: {
    name: 'SUB_GATE',
  },
  ADD_BACK_CONN: {
    name: 'ADD_BACK_CONN',
  },
  SUB_BACK_CONN: {
    name: 'SUB_BACK_CONN',
  },
  SWAP_NODES: {
    name: 'SWAP_NODES',
    mutateOutput: true,
  },
};

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
