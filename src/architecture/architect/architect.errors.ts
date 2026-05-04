/**
 * Raised when architect construction cannot infer input/output nodes from supplied primitives.
 */
export class ArchitectInputOutputTypeResolutionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectInputOutputTypeResolutionError';
  }
}

/**
 * Raised when architect construction produces a network with zero inputs or outputs.
 */
export class ArchitectZeroInputOutputNodesError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectZeroInputOutputNodesError';
  }
}

/**
 * Raised when an MLP builder receives too few layer sizes.
 */
export class ArchitectInvalidPerceptronConfigurationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectInvalidPerceptronConfigurationError';
  }
}

/**
 * Raised when LSTM builder arguments contain invalid layer-size values.
 */
export class ArchitectInvalidLstmLayerArgumentsError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectInvalidLstmLayerArgumentsError';
  }
}

/**
 * Raised when an LSTM builder receives too few layer sizes.
 */
export class ArchitectInvalidLstmConfigurationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectInvalidLstmConfigurationError';
  }
}

/**
 * Raised when GRU builder arguments contain invalid layer-size values.
 */
export class ArchitectInvalidGruLayerArgumentsError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectInvalidGruLayerArgumentsError';
  }
}

/**
 * Raised when a GRU builder receives too few layer sizes.
 */
export class ArchitectInvalidGruConfigurationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectInvalidGruConfigurationError';
  }
}

/**
 * Raised when a sparse architect builder receives invalid dimensions or
 * requests more structural edits than the graph can satisfy.
 */
export class ArchitectInvalidRandomSparseConfigurationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'ArchitectInvalidRandomSparseConfigurationError';
  }
}
