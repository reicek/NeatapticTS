/**
 * Root compatibility surface for activation-array pooling.
 *
 * The implementation chapter lives under `activationArrayPool/`, but these
 * exports stay at the architecture root so hot-path allocation helpers remain
 * easy to discover from the public API.
 */
export type { ActivationArray } from './activationArrayPool/activationArrayPool';
export { activationArrayPool } from './activationArrayPool/activationArrayPool';
