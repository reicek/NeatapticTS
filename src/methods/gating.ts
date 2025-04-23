/**
 * Defines different methods for gating connections between neurons or groups of neurons.
 *
 * Gating mechanisms dynamically control the flow of information through connections
 * in a neural network. This allows the network to selectively route information,
 * enabling more complex computations, memory functions, and adaptive behaviors.
 * These mechanisms are inspired by biological neural processes where certain neurons
 * can modulate the activity of others. Gating is particularly crucial in recurrent
 * neural networks (RNNs) for managing information persistence over time.
 *
 * @see {@link https://en.wikipedia.org/wiki/Artificial_neural_network#Gating_mechanisms}
 */
export const gating = {
  /**
   * Output Gating: The gating neuron(s) control the activation flowing *out*
   * of the connection's target neuron(s). The connection's weight remains static,
   * but the output signal from the target neuron is modulated by the gater's state.
   * @property {string} name - Identifier for the output gating method.
   */
  OUTPUT: {
    name: 'OUTPUT',
  },

  /**
   * Input Gating: The gating neuron(s) control the activation flowing *into*
   * the connection's target neuron(s). The connection effectively transmits
   * `connection_weight * source_activation * gater_activation` to the target neuron.
   * @property {string} name - Identifier for the input gating method.
   */
  INPUT: {
    name: 'INPUT',
  },

  /**
   * Self Gating: The gating neuron(s) directly modulate the *weight* or strength
   * of the connection itself. The connection's effective weight becomes dynamic,
   * influenced by the gater's activation state (`effective_weight = connection_weight * gater_activation`).
   * @property {string} name - Identifier for the self-gating method.
   */
  SELF: {
    name: 'SELF',
  },
};
