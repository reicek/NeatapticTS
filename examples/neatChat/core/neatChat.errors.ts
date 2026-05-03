/**
 * Base class for all NEATchat boundary errors.
 *
 * The split keeps NEATchat-specific validation failures grouped behind one
 * parent class so callers can catch one boundary-owned error family while
 * preserving precise subclasses for user-facing guidance.
 */
export class NeatChatError extends Error {
  /**
   * @param message - Human-readable error message.
   */
  public constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/**
 * Thrown when a numeric option must be a positive integer.
 *
 * @example
 * ```ts
 * try {
 *   resolvePositiveInteger(0, 'topWordLimit');
 * } catch (error) {
 *   if (error instanceof NeatChatPositiveIntegerValidationError) {
 *     console.error(error.message);
 *   }
 * }
 * ```
 */
export class NeatChatPositiveIntegerValidationError extends NeatChatError {}

/**
 * Thrown when a snapshot counter must be a non-negative integer.
 */
export class NeatChatNonNegativeIntegerValidationError extends NeatChatError {}

/**
 * Thrown when a snapshot payload uses an unsupported format version.
 */
export class NeatChatSnapshotVersionError extends NeatChatError {}

/**
 * Thrown when required snapshot payload fields are missing or malformed.
 */
export class NeatChatSnapshotShapeError extends NeatChatError {}
