import {
  FLAPPY_LIGHT_NEON_RAMP,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_CHARACTER_WIDTH_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_MIN_WIDTH_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_PADDING_PX,
} from '../../constants/constants';
import { FLAPPY_INPUT_GROUP_LABELS } from './network-view.constants';
import type {
  InputLabelGroupDefinition,
  InputGroupLabelBand,
  InputNodeDescriptionLabel,
} from './network-view.types';

const FLAPPY_INPUT_LABEL_GROUP_DEFINITIONS: readonly InputLabelGroupDefinition[] =
  [
    {
      label: FLAPPY_INPUT_GROUP_LABELS[0] ?? 'BIRD STATE',
      labelLines: ['BIRD', 'STATE'],
      tooltipHeading: 'Bird State',
      tooltipBodyParagraphs: [
        "This group tells the network where the bird is and how fast it is already moving. It is the controller's body-state check before any pipe geometry matters.",
        'Bird height says where the bird sits in the tunnel, while vertical speed says whether it is already rising or falling into trouble.',
        'Together these channels answer a control question that every later feature depends on: would a flap correct the current motion, or would it overreact?',
      ],
      nodeDescriptionDefinitions: [
        {
          labelLines: ['Bird height'],
          tooltipHeading: 'Bird Height',
          tooltipBodyParagraphs: [
            "Bird height measures the bird's current vertical position inside the playfield, so the controller knows whether it is drifting toward the floor or ceiling.",
            'It does not say when to flap by itself, but it anchors every comparison with the next and following gaps.',
          ],
        },
        {
          labelLines: ['Vertical speed'],
          tooltipHeading: 'Vertical Speed',
          tooltipBodyParagraphs: [
            'Vertical speed measures how fast the bird is already rising or falling when this frame begins.',
            'That matters because a flap adds to current motion instead of resetting it, so the policy can avoid late oscillating corrections.',
          ],
        },
      ],
      backgroundColor: FLAPPY_LIGHT_NEON_RAMP[0],
      orientation: 'vertical',
    },
    {
      label: FLAPPY_INPUT_GROUP_LABELS[1] ?? 'NEXT GAP',
      labelLines: ['NEXT', 'GAP'],
      tooltipHeading: 'Next Gap',
      tooltipBodyParagraphs: [
        'This group describes the very next opening the bird must survive. It turns the obstacle into a compact navigation target instead of a wall of pixels.',
        'Distance tells the controller how much time is left, while offset, top, and bottom explain where the safe corridor sits around the bird.',
        'That separation matters because urgency and alignment are different problems: the policy needs to know both how soon to react and where to steer.',
      ],
      nodeDescriptionDefinitions: [
        {
          labelLines: ['Next pipe distance'],
          tooltipHeading: 'Next Pipe Distance',
          tooltipBodyParagraphs: [
            'Next pipe distance acts like a countdown to the next real steering deadline.',
            'Large values allow calm setup, while small values tell the network that any remaining alignment error must be corrected quickly.',
          ],
        },
        {
          labelLines: ['Next gap offset'],
          tooltipHeading: 'Next Gap Offset',
          tooltipBodyParagraphs: [
            "Next gap offset measures the bird's vertical difference from the center of the next safe opening.",
            'It is one of the clearest steer-here signals because it says both how far off the bird is and in which direction.',
          ],
        },
        {
          labelLines: ['Next gap top'],
          tooltipHeading: 'Next Gap Top',
          tooltipBodyParagraphs: [
            'Next gap top marks the upper boundary of the next opening rather than just its center.',
            'When combined with bird height, it exposes how much ceiling-side safety margin is left.',
          ],
        },
        {
          labelLines: ['Next gap bottom'],
          tooltipHeading: 'Next Gap Bottom',
          tooltipBodyParagraphs: [
            'Next gap bottom marks the lower boundary of the next opening and completes the safe corridor geometry.',
            'When combined with bird height, it exposes how much floor-side safety margin remains.',
          ],
        },
      ],
      backgroundColor: FLAPPY_LIGHT_NEON_RAMP[2],
      orientation: 'vertical',
    },
    {
      label: FLAPPY_INPUT_GROUP_LABELS[2] ?? 'LOOK AHEAD',
      labelLines: ['LOOK', 'AHEAD'],
      tooltipHeading: 'Look Ahead',
      tooltipBodyParagraphs: [
        'This group gives the network three planning-oriented signals that complement the immediate next-gap geometry.',
        'The pipe entrance distance crosses zero the moment the bird enters the pipe body and goes negative while the bird is traversing it — a signal the other channels cannot provide.',
        'Gap clearance says how well-centred the bird currently is inside the opening, creating pressure to maintain position rather than drift.',
        'The second-gap offset introduces a lookahead horizon: if the next obstacle sits at a different height the network can start repositioning early instead of reacting late.',
      ],
      nodeDescriptionDefinitions: [
        {
          labelLines: ['Pipe entrance dist'],
          tooltipHeading: 'Pipe Entrance Distance',
          tooltipBodyParagraphs: [
            'Pipe entrance distance measures the signed gap between the front of the bird and the left edge of the next pipe.',
            'The value is positive while the pipe is still ahead, crosses zero when the bird enters, and goes negative while the bird is inside the pipe body.',
            "That negative region gives the controller an unambiguous 'currently traversing' signal that the pipe-exit distance and gap-offset channels alone cannot supply.",
          ],
        },
        {
          labelLines: ['Gap clearance'],
          tooltipHeading: 'Gap Clearance',
          tooltipBodyParagraphs: [
            'Gap clearance measures how centred the bird currently is inside the next gap opening.',
            'A value near +1 means the bird is well inside the safe corridor; a value near 0 means it is on the edge; a negative value means it has already crossed the gap boundary.',
            'Unlike the gap-offset channel, clearance is symmetric around the corridor centre so the controller gets a direct safety margin reading rather than a directional correction signal.',
          ],
        },
        {
          labelLines: ['2nd gap offset'],
          tooltipHeading: 'Second Gap Offset',
          tooltipBodyParagraphs: [
            'Second gap offset measures the signed vertical difference between the bird and the centre of the second upcoming pipe opening.',
            'When the two gaps are at similar heights this channel is near zero and the controller can safely hold position.',
            'When the second gap sits noticeably higher or lower, this signal motivates early repositioning before the first pipe is even cleared — the key missing ingredient for smooth sequential navigation.',
          ],
        },
      ],
      backgroundColor: FLAPPY_LIGHT_NEON_RAMP[4],
      orientation: 'vertical',
    },
  ] as const;

/**
 * Semantic input-label helpers for the network-view panel.
 *
 * The Flappy controller now uses a compact current-frame shelf, but the panel
 * still needs to teach what each row means. These helpers recover both the
 * broader semantic families and the per-input descriptions.
 */

/**
 * Resolves input-layer semantic label bands for the simplified Flappy inputs.
 *
 * When the input size matches the current-frame layout, the view can annotate
 * the full input band directly beside the input layer.
 *
 * @param inputNodeCount - Input-layer node count.
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
 * Resolves one short horizontal description for each simplified Flappy input.
 *
 * These descriptions sit between the group bands and the network so each input
 * row can be read directly from the browser visualizer.
 *
 * @param inputNodeCount - Input-layer node count.
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
  const resolvedInputLabelGroupDefinitions =
    inputLabelGroupDefinitions ?? FLAPPY_INPUT_LABEL_GROUP_DEFINITIONS;
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
    FLAPPY_NETWORK_INPUT_DESCRIPTION_MIN_WIDTH_PX,
    widestLineCharacterCount *
      FLAPPY_NETWORK_INPUT_DESCRIPTION_CHARACTER_WIDTH_PX +
      FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_PADDING_PX * 2,
  );
}
