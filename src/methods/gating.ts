/**
 * Specifies how to gate a connection between two groups of multiple neurons.
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
