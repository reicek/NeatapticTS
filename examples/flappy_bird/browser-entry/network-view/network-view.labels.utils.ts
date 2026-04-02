import {
  FLAPPY_LIGHT_NEON_RAMP,
  FLAPPY_MEMORY_CORE_FEATURE_COUNT,
  FLAPPY_MEMORY_STACKED_FRAME_COUNT,
} from '../../constants/constants';
import { FLAPPY_INPUT_GROUP_LABELS } from './network-view.constants';
import type { InputGroupLabelBand } from './network-view.types';

/**
 * Semantic input-label helpers for the network-view panel.
 *
 * The Flappy controller input layer is not just a list of anonymous scalars; it
 * is organized into stacked observation frames plus action-history channels.
 * These helpers recover that grouping for visual annotation.
 */

/**
 * Resolves input-layer semantic label bands for Flappy temporal observation channels.
 *
 * When the input size matches the expected temporal-memory layout, the view can
 * annotate groups such as stacked frames and action channels directly beside the
 * input layer.
 *
 * @param inputNodeCount - Input-layer node count.
 * @returns Group label ranges with band colors.
 */
export function resolveInputGroupLabelBands(
  inputNodeCount: number,
): InputGroupLabelBand[] {
  const perFrameFeatureCount = FLAPPY_MEMORY_CORE_FEATURE_COUNT;
  const stackedFrameCount = FLAPPY_MEMORY_STACKED_FRAME_COUNT;
  const temporalFeatureCount = perFrameFeatureCount * stackedFrameCount;
  const actionChannelsCount = 2;
  const expectedInputNodeCount = temporalFeatureCount + actionChannelsCount;
  if (inputNodeCount !== expectedInputNodeCount) {
    return [];
  }

  const groupedCounts = [
    perFrameFeatureCount,
    perFrameFeatureCount,
    perFrameFeatureCount,
    1,
    1,
  ];
  const groupedColors = [
    FLAPPY_LIGHT_NEON_RAMP[0],
    FLAPPY_LIGHT_NEON_RAMP[2],
    FLAPPY_LIGHT_NEON_RAMP[4],
    FLAPPY_LIGHT_NEON_RAMP[6],
    FLAPPY_LIGHT_NEON_RAMP[8],
  ];
  const groupedOrientations: Array<'vertical' | 'horizontal'> = [
    'vertical',
    'vertical',
    'vertical',
    'horizontal',
    'horizontal',
  ];

  const groupBands: InputGroupLabelBand[] = [];
  let runningNodeIndex = 0;
  groupedCounts.forEach((groupCount, groupIndex) => {
    const startNodeIndex = runningNodeIndex;
    const endNodeIndex = runningNodeIndex + groupCount - 1;
    groupBands.push({
      label: FLAPPY_INPUT_GROUP_LABELS[groupIndex] ?? `GROUP ${groupIndex + 1}`,
      startNodeIndex,
      endNodeIndex,
      backgroundColor:
        groupedColors[groupIndex] ??
        FLAPPY_LIGHT_NEON_RAMP.at(-1) ??
        FLAPPY_LIGHT_NEON_RAMP[0],
      orientation: groupedOrientations[groupIndex] ?? 'vertical',
    });
    runningNodeIndex += groupCount;
  });

  return groupBands;
}
