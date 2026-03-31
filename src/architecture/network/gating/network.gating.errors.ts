/**
 * Raised when a gating node does not belong to the target network.
 */
export class NetworkGatingNodeMembershipError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkGatingNodeMembershipError';
  }
}

/**
 * Raised when a gating removal request targets a structural anchor node.
 */
export class NetworkGatingStructuralAnchorRemovalError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkGatingStructuralAnchorRemovalError';
  }
}

/**
 * Raised when a gating removal request targets a node that is not in the network.
 */
export class NetworkGatingRemovalNodeNotFoundError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkGatingRemovalNodeNotFoundError';
  }
}
