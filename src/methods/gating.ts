/**
 * Specifies how to gate a connection between two groups of multiple neurons.
 *
 * Gating mechanisms control the flow of information in neural networks,
 * enabling complex behaviors and memory. They are inspired by biological
 * neural systems and are crucial for tasks requiring selective information
 * routing.
 *
 * For example, gating can be used to regulate recurrent connections in
 * recurrent neural networks (RNNs), allowing the network to retain or
 * discard information over time.
 *
 * @see {@link https://en.wikipedia.org/wiki/Artificial_neural_network#Gating_mechanisms}
 */
export const gating = {
  /**
   * Gate the output of the connection.
   * @property {string} name - The name of the gating method.
   */
  OUTPUT: {
    name: 'OUTPUT',
  },

  /**
   * Gate the input of the connection.
   * @property {string} name - The name of the gating method.
   */
  INPUT: {
    name: 'INPUT',
  },

  /**
   * Gate the connection itself (self-gating).
   * @property {string} name - The name of the gating method.
   */
  SELF: {
    name: 'SELF',
  },
};
