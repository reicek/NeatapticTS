/**
 * Raised when caller-provided group values do not match the number of nodes.
 */
export class GroupSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'GroupSizeMismatchError';
  }
}

/**
 * Raised when ONE_TO_ONE group connections are requested for groups of different sizes.
 */
export class GroupOneToOneSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'GroupOneToOneSizeMismatchError';
  }
}

/**
 * Raised when gating is requested without specifying a gating method.
 */
export class GroupGatingMethodRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'GroupGatingMethodRequiredError';
  }
}
