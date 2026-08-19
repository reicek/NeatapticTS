/**
 * Message-handler orchestrator for the Neatenstein display worker.
 *
 * Extracted from `display.worker.ts` as part of the B2 architecture-debt
 * refactoring.  This module documents the compositing pipeline and
 * re-exports the individual case handlers from
 * {@link ./display.worker.message-handler.utils}.
 *
 * @module
 */

/// <reference lib="webworker" />

import {
  INIT_TYPE,
  RESIZE_TYPE,
  SIM_STATE_TYPE,
  INPUT_TYPE,
} from './display.worker.message-handler.utils';

/**
 * Compositing order for the Neatenstein ray-caster frame.
 *
 * The renderer paints layers in the following back-to-front order:
 *
 * 1. **Floor** — flat floor cast at the bottom half of the screen.
 * 2. **Ceiling** — flat ceiling cast at the top half of the screen.
 * 3. **Walls** — ray-cast wall slices with depth-sorted sprite occlusion.
 * 4. **Sprites** — billboarded enemy and item sprites, depth-sorted.
 * 5. **Pulses** — energy pulse effects rendered additively on top.
 * 6. **Bolts** — projectile bolts rendered as the final overlay layer.
 *
 * This ordering ensures correct occlusion: floor and ceiling form the
 * background, walls provide mid-depth geometry, sprites are placed in
 * front of walls at the correct depth, and effects (pulses and bolts)
 * are composited on top of the scene.
 *
 * Re-exported from {@link ./display.worker.render.utils} to avoid
 * duplicating the canonical {@link RENDER_COMPOSITING_ORDER} constant.
 */
export { RENDER_COMPOSITING_ORDER as COMPOSITING_ORDER } from './display.worker.render.utils';

/** Message-type labels handled by the orchestrator. */
export const HANDLED_MESSAGE_TYPES = {
  init: INIT_TYPE,
  resize: RESIZE_TYPE,
  simState: SIM_STATE_TYPE,
  input: INPUT_TYPE,
} as const;

export {
  handleInitMessage,
  handleSimStateMessage,
  handleInputMessage,
  isWorkerMessage,
  resolveDisplayTier,
  resolveMapSeed,
} from './display.worker.message-handler.utils';
