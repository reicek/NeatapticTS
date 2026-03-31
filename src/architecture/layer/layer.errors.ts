/**
 * Raised when caller-provided layer values do not match the number of nodes.
 */
export class LayerSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'LayerSizeMismatchError';
  }
}

/**
 * Raised when a layer output group is missing during connect operations.
 */
export class LayerOutputConnectUnavailableError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'LayerOutputConnectUnavailableError';
  }
}

/**
 * Raised when a layer output group is missing during gate operations.
 */
export class LayerOutputGateUnavailableError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'LayerOutputGateUnavailableError';
  }
}

/**
 * Raised when a layer target output group is missing during input wiring.
 */
export class LayerInputTargetUnavailableError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'LayerInputTargetUnavailableError';
  }
}

/**
 * Raised when a layer source output group is missing during input wiring.
 */
export class LayerInputSourceUnavailableError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'LayerInputSourceUnavailableError';
  }
}

/**
 * Raised when a recurrent memory layer cannot resolve a group-like input block.
 */
export class LayerMemoryInputBlockTypeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'LayerMemoryInputBlockTypeError';
  }
}

/**
 * Raised when recurrent memory source and target block sizes do not match.
 */
export class LayerMemoryInputSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'LayerMemoryInputSizeMismatchError';
  }
}
