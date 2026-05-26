/**
 * Raised when node removal targets a node that is not in the network.
 */
export class NetworkRemoveNodeNotFoundError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRemoveNodeNotFoundError';
  }
}

/**
 * Raised when node removal targets an input or output anchor node.
 */
export class NetworkRemoveStructuralAnchorError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRemoveStructuralAnchorError';
  }
}
