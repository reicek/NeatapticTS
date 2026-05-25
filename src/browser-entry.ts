/**
 * Browser-safe public entry for NeatapticTS bundles.
 *
 * This entry keeps the core learning and architecture surface available to
 * browser builds while leaving out the mixed-runtime multithreading facade.
 * The Node worker helpers stay on the main root entry until the environment
 * adapter step narrows that boundary further.
 *
 * @example
 * ```ts
 * import { Neat, Network, methods } from './browser-entry';
 *
 * const network = new Network(2, 1);
 * const output = network.activate([0.5, 0.5]);
 * console.log(output, methods.activation.LOGISTIC);
 * ```
 */
export { default as Neat } from './neat';
export { default as Network } from './architecture/network';
export { formatConstructSummary } from './architecture/network';
export { default as Node } from './architecture/node';
export { default as Layer } from './architecture/layer';
export { default as Group } from './architecture/group';
export { default as Connection } from './architecture/connection';
export { default as Architect } from './architecture/architect';
/** Activation, cost, crossover, mutation, and selection method objects. Stateless algorithm namespaces for use with {@link Network} and {@link Neat}. */
export * as methods from './methods/methods';
/** Global library configuration namespace. Controls backend precision, debug flags, and runtime behavior for browser builds. */
export * as config from './config';
