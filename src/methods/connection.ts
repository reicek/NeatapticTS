/**
 * Specifies the manner in which two groups of nodes are connected.
 */
export const groupConnection = Object.freeze({
  // Renamed export
  /**
   * Connects all nodes in the source group to all nodes in the target group.
   */
  ALL_TO_ALL: Object.freeze({
    name: 'ALL_TO_ALL', // Renamed name
  }),

  /**
   * Connects all nodes in the source group to all nodes in the target group, excluding self-connections (if groups are identical).
   */
  ALL_TO_ELSE: Object.freeze({
    name: 'ALL_TO_ELSE', // Renamed name
  }),

  /**
   * Connects each node in the source group to the node at the same index in the target group. Requires groups to be the same size.
   */
  ONE_TO_ONE: Object.freeze({
    name: 'ONE_TO_ONE', // Renamed name
  }),
});

/**
 * Export the connection object as the default export.
 */
export default groupConnection; // Export renamed object
