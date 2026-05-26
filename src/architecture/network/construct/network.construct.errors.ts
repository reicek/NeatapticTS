/**
 * Raised when construct-from-parts cannot resolve an explicit input or output node id.
 */
export class NetworkConstructNodeIdResolutionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructNodeIdResolutionError';
  }
}

/**
 * Raised when a single explicit string node id matches more than one labeled node in the provided parts list during network construction.
 */
export class NetworkConstructAmbiguousNodeIdError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructAmbiguousNodeIdError';
  }
}

/**
 * Raised when collected connections reference nodes that were not included in the provided parts.
 */
export class NetworkConstructMissingReferencedNodeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructMissingReferencedNodeError';
  }
}

/**
 * Raised when the same source-to-target node pair appears more than once while duplicate edges are explicitly disallowed by the construction policy.
 */
export class NetworkConstructDuplicateEdgeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructDuplicateEdgeError';
  }
}

/**
 * Raised when a connection loops from a node back to itself while self-edges are explicitly forbidden by the active construction policy.
 */
export class NetworkConstructSelfEdgeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructSelfEdgeError';
  }
}

/**
 * Raised when the construction pass finds hidden nodes with no connections while the active construction policy forbids isolated hidden neurons in the graph.
 */
export class NetworkConstructIsolatedHiddenNodeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructIsolatedHiddenNodeError';
  }
}

/**
 * Raised when the network construction pass cannot find any nodes classified with an input role in the provided node list.
 */
export class NetworkConstructNoInputNodesError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructNoInputNodesError';
  }
}

/**
 * Raised when an input-role node is wired as a connection target while the active construction policy forbids incoming edges on input nodes.
 */
export class NetworkConstructInputNodeIncomingEdgeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructInputNodeIncomingEdgeError';
  }
}

/**
 * Raised when the network construction pass cannot find any nodes classified with an output role in the provided node list.
 */
export class NetworkConstructNoOutputNodesError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructNoOutputNodesError';
  }
}

/**
 * Raised when a public output node emits one or more outgoing edges while sink-only output validation is enabled.
 */
export class NetworkConstructOutputNodeOutgoingEdgeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructOutputNodeOutgoingEdgeError';
  }
}

/**
 * Raised when a public output node gates one or more connections while sink-only output validation is enabled.
 */
export class NetworkConstructOutputNodeGatedConnectionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructOutputNodeGatedConnectionError';
  }
}

/**
 * Raised when a cyclic graph is compiled while acyclic mode is required.
 */
export class NetworkConstructCycleModeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructCycleModeError';
  }
}
