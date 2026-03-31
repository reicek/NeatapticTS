/**
 * Raised when a node mutation call receives a null or undefined method.
 */
export class NodeMutationMethodRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NodeMutationMethodRequiredError';
  }
}

/**
 * Raised when a node mutation method name is unknown.
 */
export class NodeUnknownMutationMethodError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NodeUnknownMutationMethodError';
  }
}

/**
 * Raised when a known mutation call reaches an unsupported mutation branch.
 */
export class NodeUnsupportedMutationMethodError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NodeUnsupportedMutationMethodError';
  }
}

/**
 * Raised when a node connection target is missing.
 */
export class NodeUndefinedConnectionTargetError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NodeUndefinedConnectionTargetError';
  }
}

/**
 * Raised when a node connection target is neither a node nor a group-like object.
 */
export class NodeInvalidConnectionTargetTypeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NodeInvalidConnectionTargetTypeError';
  }
}
