/**
 * Root compatibility surface for node-pool helpers.
 *
 * The `nodePool/` chapter owns the concrete pooling behavior; this file keeps
 * the convenience exports grouped at the architecture root for callers that do
 * not need to know the internal folder layout.
 */
export {
  acquireNode,
  nodePoolStats,
  releaseNode,
  resetNodePool,
} from './nodePool/nodePool';

export type { AcquireNodeOptions } from './nodePool/nodePool';

export { default } from './nodePool/nodePool';
