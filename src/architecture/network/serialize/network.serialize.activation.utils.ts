import * as methods from '../../../methods/methods';
import type { ActivationFunction } from '../../../methods/activation/activation.utils';
import {
  FALLBACK_ACTIVATION_KEY,
  WARNING_UNKNOWN_SQUASH_PREFIX,
  WARNING_UNKNOWN_SQUASH_SUFFIX,
} from './network.serialize.utils.types';

/**
 * Resolves a canonical activation key from a runtime activation function reference.
 *
 * Resolution order is: direct registry reference match, then the function name,
 * then a stable identity fallback key.
 *
 * @param squashFunction - Activation function instance.
 * @returns Activation key.
 * @remarks
 * Canonical keys make serialized payloads portable across runtimes where function references
 * are not comparable after reload.
 * @example
 * ```ts
 * import * as methods from '../../../methods/methods';
 *
 * const key = resolveActivationKey(methods.Activation.tanh);
 * ```
 */
export function resolveActivationKey(
  squashFunction: ActivationFunction,
): string {
  const activationEntry = findActivationEntryByReference(squashFunction);
  if (activationEntry) {
    return activationEntry[0];
  }

  const activationName = resolveNamedActivationFromFunction(squashFunction);
  if (activationName) {
    return activationName;
  }

  return FALLBACK_ACTIVATION_KEY;
}

/**
 * Resolves an activation function from a stored key or function name.
 *
 * Unknown values produce a warning and return the identity activation to keep
 * deserialization deterministic and non-throwing.
 *
 * @param squashName - Activation key or function name.
 * @returns Activation function.
 * @remarks
 * This fallback strategy favors recoverability over strict failure.
 * Applications that require strict schema validation should reject unknown squash names before import.
 * @example
 * ```ts
 * const squashFunction = resolveActivationFunction('relu');
 * ```
 */
export function resolveActivationFunction(
  squashName: string | undefined,
): ActivationFunction {
  const activationByKey = findActivationByKey(squashName);
  if (activationByKey) {
    return activationByKey;
  }

  const activationByName = findActivationByFunctionName(squashName);
  if (activationByName) {
    return activationByName;
  }

  warnUnknownSquashName(squashName);
  return methods.Activation.identity;
}

/**
 * Finds activation entry by function reference.
 *
 * @param squashFunction - Activation function instance.
 * @returns Activation entry or undefined.
 */
function findActivationEntryByReference(
  squashFunction: ActivationFunction,
): [string, ActivationFunction] | undefined {
  return Object.entries(methods.Activation).find(
    ([, activationFunction]) => activationFunction === squashFunction,
  );
}

/**
 * Resolves activation name from function.name when non-empty.
 *
 * @param squashFunction - Activation function instance.
 * @returns Activation name or undefined.
 */
function resolveNamedActivationFromFunction(
  squashFunction: ActivationFunction,
): string | undefined {
  if (
    typeof squashFunction?.name === 'string' &&
    squashFunction.name.length > 0
  ) {
    return squashFunction.name;
  }
  return undefined;
}

/**
 * Resolves activation by direct key lookup.
 *
 * @param squashName - Activation key.
 * @returns Activation function or undefined.
 */
function findActivationByKey(
  squashName: string | undefined,
): ActivationFunction | undefined {
  if (!squashName) {
    return undefined;
  }
  return methods.Activation[squashName];
}

/**
 * Resolves activation by matching function.name.
 *
 * @param squashName - Activation function name.
 * @returns Activation function or undefined.
 */
function findActivationByFunctionName(
  squashName: string | undefined,
): ActivationFunction | undefined {
  const activationEntry = Object.entries(methods.Activation).find(
    ([, activationFunction]) => activationFunction.name === squashName,
  );
  return activationEntry?.[1];
}

/**
 * Warns about unknown activation and fallback to identity.
 *
 * @param squashName - Unknown activation name.
 * @returns Nothing.
 */
function warnUnknownSquashName(squashName: string | undefined): void {
  console.warn(
    `${WARNING_UNKNOWN_SQUASH_PREFIX} '${String(squashName)}' ${WARNING_UNKNOWN_SQUASH_SUFFIX}`,
  );
}
