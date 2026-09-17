/**
 * Flappy-branded network visualization settings.
 *
 * This module is the single source of demo-branded values injected into the
 * shared network visualizer from the host. It lets the shared overlay stay
 * generic while the Flappy host retains its neon/chrome identity.
 *
 * @example
 * ```ts
 * drawNetworkVisualization(
 *   canvasContext,
 *   bestNetwork,
 *   12,
 *   2,
 *   undefined,
 *   undefined,
 *   FLAPPY_HOST_NETWORK_VIEW_SETTINGS,
 * );
 * ```
 */

import {
  FLAPPY_CENTER_BLUE_RAMP,
  FLAPPY_LIGHT_NEON_RAMP,
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
  FLAPPY_REGULAR_NEON_RAMP,
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
  FLAPPY_NETWORK_HEADER_TEXT_COLOR,
  FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR,
  FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR,
  FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR,
  FLAPPY_NETWORK_LEGEND_HEADER_COLOR,
  FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR,
  FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR,
  FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR,
} from '../../constants/constants';
import { NETWORK_LEGEND_BACKGROUND } from '../../../shared/network-visualization/network-visualization.constants';
import type {
  InputLabelGroupDefinition,
  NetworkVisualizationSettings,
} from '../../../shared/network-visualization/network-view/network-view.types';
import { NETWORK_HOVER_TRANSITION_DURATION_MS } from '../../../shared/network-visualization/visualization/visualization.constants';

/**
 * Flappy-branded input-label group fixture passed to the shared network
 * visualizer from the host.
 */
export const FLAPPY_INPUT_LABEL_GROUP_DEFINITIONS: readonly InputLabelGroupDefinition[] =
  [
    {
      label: 'BIRD STATE',
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
      label: 'NEXT GAP',
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
      label: 'LOOK AHEAD',
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

/** Flappy-branded settings object passed to {@link drawNetworkVisualization}. */
export const FLAPPY_HOST_NETWORK_VIEW_SETTINGS: NetworkVisualizationSettings = {
  inputLabelGroupDefinitions: FLAPPY_INPUT_LABEL_GROUP_DEFINITIONS,
  canvasBackground: FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  palette: {
    currentRunText: FLAPPY_NEON_PALETTE.currentRunText,
    statusText: FLAPPY_NEON_PALETTE.statusText,
  },
  fontFamily: FLAPPY_MONOSPACE_FONT_FAMILY,
  overlayHiddenBreakpointPx: FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
  hoverTransitionDurationMs: NETWORK_HOVER_TRANSITION_DURATION_MS,
  lightNeonRamp: FLAPPY_LIGHT_NEON_RAMP,
  regularNeonRamp: FLAPPY_REGULAR_NEON_RAMP,
  centerBlueRamp: FLAPPY_CENTER_BLUE_RAMP,
  theme: {
    headerText: FLAPPY_NETWORK_HEADER_TEXT_COLOR,
    nodeLabelFill: FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR,
    hiddenNodeStroke: FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR,
    outputNodeStroke: FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR,
    outputNodeFill: FLAPPY_NEON_PALETTE.currentRunText,
    legendBackground: NETWORK_LEGEND_BACKGROUND,
    legendStroke: FLAPPY_NEON_PALETTE.statusText,
    legendHeader: FLAPPY_NETWORK_LEGEND_HEADER_COLOR,
    legendConnectionTitle: FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR,
    legendBiasTitle: FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR,
    legendRowText: FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR,
  },
};
