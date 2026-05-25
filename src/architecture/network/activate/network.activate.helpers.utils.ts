/**
 * Re-export no-trace activation orchestration for callers that want output values
 * without retaining per-node trace state.
 */
export { executeNoTraceActivation } from './network.activate.notrace.utils';

/**
 * Re-export raw activation orchestration for callers that need explicit control
 * over training traces and activation-depth safeguards.
 */
export { executeRawActivation } from './network.activate.raw.utils';

/**
 * Re-export batch activation orchestration for matrix-style inference and
 * training workloads.
 */
export { executeBatchActivation } from './network.activate.batch.utils';

/**
 * Re-export helper that assembles the shared context consumed by no-trace
 * activation helpers.
 */
export { createNoTraceActivationContext } from './network.activate.contexts.utils';

/**
 * Re-export helper that assembles the shared context consumed by raw
 * activation helpers.
 */
export { createRawActivationContext } from './network.activate.contexts.utils';

/**
 * Re-export helper that assembles the shared context consumed by batch
 * activation helpers.
 */
export { createBatchActivationContext } from './network.activate.contexts.utils';
