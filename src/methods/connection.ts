/**
 * Specifies the manner in which two groups of nodes are connected.
 */
export const connection = {
  /**
   * Connects all nodes in one group to all nodes in another group.
   */
  ALL_TO_ALL: {
    name: 'OUTPUT',
  },

  /**
   * Connects all nodes in one group to all nodes in another group, except self-connections.
   */
  ALL_TO_ELSE: {
    name: 'INPUT',
  },

  /**
   * Connects each node in one group to exactly one node in another group.
   */
  ONE_TO_ONE: {
    name: 'SELF',
  },
};

/**
 * Export the connection object as the default export.
 */
export default connection;
