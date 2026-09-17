import {
  NETWORK_INPUT_DESCRIPTION_CHARACTER_WIDTH_PX,
  NETWORK_INPUT_DESCRIPTION_MIN_WIDTH_PX,
  NETWORK_INPUT_DESCRIPTION_TEXT_PADDING_PX,
} from '../network-visualization.constants';
import type {
  InputLabelGroupDefinition,
  InputGroupLabelBand,
  InputNodeDescriptionLabel,
} from './network-view.types';

/**
 * Semantic input-label helpers for the network-view panel.
 *
 * A host can pass a compact current-frame shelf as input-label group
 * definitions, but the panel still needs to teach what each row means. These
 * helpers recover both the broader semantic families and the per-input
 * descriptions from the provided groups.
 */

/**
 * Resolves input-layer semantic label bands from the provided input-label
 * group definitions.
 *
 * When the input size matches the total node count described by the groups, the
 * view can annotate the full input band directly beside the input layer.
 *
 * @param inputNodeCount - Input-layer node count.
 * @param inputLabelGroupDefinitions - Optional semantic input-label group definitions.
 * @returns Group label ranges with band colors.
 */
export function resolveInputGroupLabelBands(
  inputNodeCount: number,
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[],
): InputGroupLabelBand[] {
  const resolvedInputLabelGroupDefinitions = resolveApplicableInputLabelGroups(
    inputNodeCount,
    inputLabelGroupDefinitions,
  );
  if (resolvedInputLabelGroupDefinitions.length === 0) {
    return [];
  }

  const groupBands: InputGroupLabelBand[] = [];
  let runningNodeIndex = 0;
  resolvedInputLabelGroupDefinitions.forEach((groupDefinition) => {
    const groupCount = groupDefinition.nodeDescriptionDefinitions.length;
    const startNodeIndex = runningNodeIndex;
    const endNodeIndex = runningNodeIndex + groupCount - 1;
    groupBands.push({
      label: groupDefinition.label,
      labelLines: groupDefinition.labelLines,
      tooltipHeading: groupDefinition.tooltipHeading,
      tooltipBodyParagraphs: groupDefinition.tooltipBodyParagraphs,
      startNodeIndex,
      endNodeIndex,
      backgroundColor: groupDefinition.backgroundColor,
      orientation: groupDefinition.orientation,
    });
    runningNodeIndex += groupCount;
  });

  return groupBands;
}

/**
 * Resolves one short horizontal description for each provided input-label
 * group node.
 *
 * These descriptions sit between the group bands and the network so each input
 * row can be read directly from the browser visualizer.
 *
 * @param inputNodeCount - Input-layer node count.
 * @param inputLabelGroupDefinitions - Optional semantic input-label group definitions.
 * @returns Ordered node descriptions for the input shelf.
 */
export function resolveInputNodeDescriptionLabels(
  inputNodeCount: number,
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[],
): InputNodeDescriptionLabel[] {
  const resolvedInputLabelGroupDefinitions = resolveApplicableInputLabelGroups(
    inputNodeCount,
    inputLabelGroupDefinitions,
  );
  if (resolvedInputLabelGroupDefinitions.length === 0) {
    return [];
  }

  const nodeDescriptions: InputNodeDescriptionLabel[] = [];
  let runningNodeIndex = 0;
  resolvedInputLabelGroupDefinitions.forEach((groupDefinition) => {
    groupDefinition.nodeDescriptionDefinitions.forEach(
      ({ labelLines, tooltipBodyParagraphs, tooltipHeading }) => {
        nodeDescriptions.push({
          labelLines,
          tooltipHeading,
          tooltipBodyParagraphs,
          nodeIndex: runningNodeIndex,
        });
        runningNodeIndex += 1;
      },
    );
  });

  return nodeDescriptions;
}

/**
 * Resolves the maximum width required by the current input-description column.
 *
 * The layout shelf should reserve enough space for the widest chip so the
 * semantic group bands never get pushed off the left edge of the canvas.
 *
 * @param inputNodeCount - Input-layer node count.
 * @param inputLabelGroupDefinitions - Optional semantic input-label group definitions.
 * @returns Maximum chip width needed by the current input-description column.
 */
export function resolveInputDescriptionColumnWidthPx(
  inputNodeCount: number,
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[],
): number {
  const inputDescriptions = resolveInputNodeDescriptionLabels(
    inputNodeCount,
    inputLabelGroupDefinitions,
  );

  if (inputDescriptions.length === 0) {
    return 0;
  }

  return Math.max(
    ...inputDescriptions.map((inputDescription) =>
      resolveInputDescriptionChipWidthPx(inputDescription.labelLines),
    ),
  );
}

function resolveApplicableInputLabelGroups(
  inputNodeCount: number,
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[],
): readonly InputLabelGroupDefinition[] {
  const resolvedInputLabelGroupDefinitions = inputLabelGroupDefinitions ?? [];
  const totalDefinedNodeCount = resolvedInputLabelGroupDefinitions.reduce(
    (runningNodeCount, inputLabelGroupDefinition) =>
      runningNodeCount +
      inputLabelGroupDefinition.nodeDescriptionDefinitions.length,
    0,
  );

  return totalDefinedNodeCount === inputNodeCount
    ? resolvedInputLabelGroupDefinitions
    : [];
}

/**
 * Resolves the content-driven width of one input-description chip.
 *
 * @param labelLines - Human-readable label lines shown inside the chip.
 * @returns Pixel width needed to render the chip without clipping.
 */
export function resolveInputDescriptionChipWidthPx(
  labelLines: readonly string[],
): number {
  const widestLineCharacterCount = Math.max(
    0,
    ...labelLines.map((labelLine) => labelLine.length),
  );

  return Math.max(
    NETWORK_INPUT_DESCRIPTION_MIN_WIDTH_PX,
    widestLineCharacterCount *
      NETWORK_INPUT_DESCRIPTION_CHARACTER_WIDTH_PX +
      NETWORK_INPUT_DESCRIPTION_TEXT_PADDING_PX * 2,
  );
}
