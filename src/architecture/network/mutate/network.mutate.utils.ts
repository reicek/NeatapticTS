import type Network from '../../network/network';
import type { MutationHandler, MutationMethod } from '../network.types';
import {
  resolveMutationKey,
  warnUnknownMutation,
} from './network.mutate.dispatch.utils';
import {
  addBackConn,
  addConn,
  addGate,
  addGRUNode,
  addLSTMNode,
  addNode,
  addSelfConn,
  batchNorm,
  modActivation,
  modBias,
  modWeight,
  reinitWeight,
  subBackConn,
  subConn,
  subGate,
  subNode,
  subSelfConn,
  swapNodes,
} from './network.mutate.handlers.utils';
import { ERROR_NO_MUTATE_METHOD } from './network.mutate.utils.types';

export type { MutationMethod } from '../network.types';

/**
 * Mutation orchestration entrypoint for network-level structural and parametric edits.
 *
 * This module intentionally stays lightweight:
 * - It resolves the incoming mutation request into a dispatch key.
 * - It selects a concrete handler from the dispatch table.
 * - It delegates execution and marks topology caches dirty after successful handling.
 *
 * Handler-specific logic lives in dedicated helper files so this module remains a stable,
 * high-level control surface for mutation flow.
 *
 * @module network.mutate
 */

/**
 * Mutation dispatch table keyed by mutation identity.
 *
 * The table maps normalized mutation keys to bound handler functions that run against
 * the network instance via Function.call. Keeping this mapping centralized makes
 * mutation routing explicit and easy to audit.
 */
const MUTATION_DISPATCH: Record<string, MutationHandler> = {
  ADD_NODE: addNode,
  SUB_NODE: subNode,
  ADD_CONN: addConn,
  SUB_CONN: subConn,
  MOD_WEIGHT: modWeight,
  MOD_BIAS: modBias,
  MOD_ACTIVATION: modActivation,
  ADD_SELF_CONN: addSelfConn,
  SUB_SELF_CONN: subSelfConn,
  ADD_GATE: addGate,
  SUB_GATE: subGate,
  ADD_BACK_CONN: addBackConn,
  SUB_BACK_CONN: subBackConn,
  SWAP_NODES: swapNodes,
  ADD_LSTM_NODE: addLSTMNode,
  ADD_GRU_NODE: addGRUNode,
  REINIT_WEIGHT: reinitWeight,
  BATCH_NORM: batchNorm,
};

/**
 * Public entry point: apply a single mutation operator to the network.
 *
 * Runtime flow:
 * 1. Validate mutation input.
 * 2. Resolve the mutation key from string/object/reference forms.
 * 3. Resolve a concrete handler from the dispatch table.
 * 4. Delegate execution and mark topology-derived caches dirty.
 *
 * Error and warning behavior:
 * - Throws when no method is provided.
 * - Emits a warning and no-ops when an unknown method key is received.
 *
 * @param this - Network instance.
 * @param method - Mutation enum value or descriptor object.
 * @returns Nothing.
 *
 * @example
 * ```ts
 * network.mutate('ADD_NODE');
 * network.mutate({ name: 'MOD_WEIGHT', min: -0.1, max: 0.1 });
 * ```
 */
export function mutateImpl(this: Network, method?: MutationMethod): void {
  // Step 1: Validate mutation input and resolve mutation key.
  if (method == null) {
    throw new Error(ERROR_NO_MUTATE_METHOD);
  }

  const mutationKey = resolveMutationKey(method);
  const mutationHandler = mutationKey
    ? MUTATION_DISPATCH[mutationKey]
    : undefined;

  // Step 2: Handle unknown mutation keys when warnings are enabled.
  if (!mutationHandler) {
    warnUnknownMutation(mutationKey);
    return;
  }

  // Step 3: Delegate to handler and mark topology cache as dirty.
  mutationHandler.call(this, method);
  (this as unknown as { _topoDirty?: boolean })._topoDirty = true;
}
