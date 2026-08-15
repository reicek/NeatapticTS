/**
 * Worker protocol constant re-exports for the Neatenstein host layer.
 *
 * Re-exports the shared worker message-type and render-tier constants from
 * {@link ../constants.ts} so host modules can import them from a single
 * host-local module instead of reaching up to the browser-entry level.
 *
 * @module
 */

export {
  /** Worker `init` message type. */
  WORKER_MSG_INIT,
  /** Worker `resize` message type. */
  WORKER_MSG_RESIZE,
  /** Worker simulation-state message type. */
  WORKER_MSG_SIM_STATE,
  /** Worker `initialized` acknowledgment message type. */
  WORKER_MSG_INITIALIZED,
  /** Worker rendered-frame message type. */
  WORKER_MSG_FRAME,
} from '../constants';

export {
  /** Render tier identifier for the OffscreenCanvas worker path. */
  RENDER_TIER_WORKER,
  /** Render tier identifier for the CPU fallback path. */
  RENDER_TIER_CPU,
  /** Render tier identifier for the GPU premium path. */
  RENDER_TIER_GPU,
} from '../constants';