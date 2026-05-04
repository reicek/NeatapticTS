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
 * Raised when one explicit string node id matches multiple labeled nodes.
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
 * Raised when duplicate source-to-target edges are forbidden during construction.
 */
export class NetworkConstructDuplicateEdgeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructDuplicateEdgeError';
  }
}

/**
 * Raised when self edges are forbidden during construction.
 */
export class NetworkConstructSelfEdgeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructSelfEdgeError';
  }
}

/**
 * Raised when hidden nodes are disconnected while isolated hidden nodes are disallowed.
 */
export class NetworkConstructIsolatedHiddenNodeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructIsolatedHiddenNodeError';
  }
}

/**
 * Raised when construction cannot identify any input-role nodes.
 */
export class NetworkConstructNoInputNodesError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructNoInputNodesError';
  }
}

/**
 * Raised when a public input node receives one or more incoming edges.
 */
export class NetworkConstructInputNodeIncomingEdgeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructInputNodeIncomingEdgeError';
  }
}

/**
 * Raised when construction cannot identify any output-role nodes.
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
