import { _reindexNodes } from './network.slab.shared.helpers.utils';
import type { SlabBuildContext } from './network.slab.utils.types';

/**
 * Applies all required prerequisite normalization steps before starting slab rebuild passes.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
export function _prepareSlabBuildPreconditions(
  buildContext: SlabBuildContext,
): void {
  // Step 1: Rebuild node indices when structural mutations invalidated ordering.
  if (buildContext.internalNet._nodeIndexDirty) {
    _reindexNodes(buildContext.network);
  }
}
