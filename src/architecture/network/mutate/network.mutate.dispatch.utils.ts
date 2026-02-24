import mutation from '../../../methods/mutation';
import { config } from '../../../config';
import type { MutationMethod, MutationMethodObject } from '../network.types';
import { UNKNOWN_MUTATION_WARNING_PREFIX } from './network.mutate.utils.types';

/**
 * Mutation-key normalization and warning helpers used by the mutate orchestrator.
 *
 * Responsibilities:
 * - Normalize string/object/reference mutation inputs into a dispatch key.
 * - Emit unknown-mutation warnings when warning mode is enabled.
 *
 * The helpers in this module are intentionally side-effect-light, except for optional
 * warning emission, so orchestration code can remain deterministic and easy to inspect.
 *
 * @module network.mutate.dispatch
 */

/**
 * Resolves a mutation dispatch key from user-provided method input.
 *
 * Resolution order:
 * 1. Use string input directly.
 * 2. Use direct object identity fields (`name`, `type`, `identity`).
 * 3. Fall back to identity-reference comparison against known mutation objects.
 *
 * @param method - Mutation method input.
 * @returns Dispatch key or undefined.
 *
 * @example
 * ```ts
 * const key = resolveMutationKey('ADD_NODE');
 * ```
 */
export function resolveMutationKey(method: MutationMethod): string | undefined {
  if (isMutationMethodKeyString(method)) {
    return method;
  }

  return resolveMutationKeyFromObject(method);
}

/**
 * Emits unknown-mutation warning when configured.
 *
 * This helper intentionally no-ops when warnings are disabled so callers can invoke it
 * without repeating feature-flag checks.
 *
 * @param mutationKey - Resolved mutation key.
 * @returns Nothing.
 */
export function warnUnknownMutation(mutationKey?: string): void {
  if (!config.warnings) {
    return;
  }
  console.warn(UNKNOWN_MUTATION_WARNING_PREFIX, mutationKey);
}

/**
 * Checks whether mutation input is already a direct key string.
 *
 * @param method - Mutation method input.
 * @returns True when method is a key string.
 */
function isMutationMethodKeyString(method: MutationMethod): method is string {
  return typeof method === 'string';
}

/**
 * Resolves mutation key from object-form descriptor.
 *
 * @param methodObject - Mutation method object.
 * @returns Dispatch key or undefined.
 */
function resolveMutationKeyFromObject(
  methodObject: MutationMethodObject,
): string | undefined {
  const directKey = resolveDirectMutationKey(methodObject);
  if (directKey) {
    return directKey;
  }

  return findMutationKeyByIdentityReference(methodObject);
}

/**
 * Resolves direct object fields that can represent a mutation key.
 *
 * @param methodObject - Mutation method object.
 * @returns Direct key or undefined.
 */
function resolveDirectMutationKey(
  methodObject: MutationMethodObject,
): string | undefined {
  return methodObject.name ?? methodObject.type ?? methodObject.identity;
}

/**
 * Resolves a mutation key by direct identity-reference comparison.
 *
 * @param method - Mutation object reference.
 * @returns Matching mutation key or undefined.
 */
function findMutationKeyByIdentityReference(
  method: MutationMethod,
): string | undefined {
  const mutationMethods = mutation as Record<string, unknown>;
  const mutationKeys = Object.keys(mutationMethods);

  for (
    let mutationKeyIndex = 0;
    mutationKeyIndex < mutationKeys.length;
    mutationKeyIndex++
  ) {
    const mutationKey = mutationKeys[mutationKeyIndex];
    if (method === mutationMethods[mutationKey]) {
      return mutationKey;
    }
  }

  return undefined;
}
