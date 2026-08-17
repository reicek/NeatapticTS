/**
 * Shared interpolation type definitions for the Neatenstein render-side state
 * interpolation.
 *
 * Centralises the generic numeric state shape consumed by
 * {@link module:./interpolate}.
 *
 * @module
 */

/**
 * Object shape accepted by the linear interpolation helper.
 *
 * The generic state is expected to expose own enumerable string keys whose
 * values are finite numbers at runtime.
 */
export type NeatensteinNumericState = Record<string, number>;
