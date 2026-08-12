/**
 * Visual constants and input/output label definitions for the racing curriculum
 * network-view panel.
 *
 * The Tier 1 racing controller consumes a 70-element observation vector and
 * produces a 2-element action vector (throttle and steer). Higher tiers widen
 * the same base vector in-place:
 *   Tier 2 — 77 inputs (70 base + 7 self-radio channels).
 *   Tier 3 — 91 inputs (70 base + 21 teammate-radio channels).
 *   Tier 4/5 — 103 inputs (91 Tier 3 + 4 tire-health + 8 pit/strategy channels).
 *   Tier 6 — 124 inputs (103 Tier 4/5 + 21 opponent-perception channels).
 *
 * These constants describe the semantic input groups, per-node chip labels,
 * hover tooltip copy, and the short output tags shown in the right-sidebar
 * network visualizer.
 *
 * Tier 1 observation vector layout (in order):
 *   0..19   Car state scalars — position, heading, speed, yaw, slip angle,
 *           progress, boundary distances, hazard/waypoint distances, optimal-line
 *           offset, target speed, etc.
 *   20..59  Five look-ahead track segments, 8 channels each — relX, relY,
 *           nextRelX, nextRelY, sinTangent, cosTangent, trackWidth, distance.
 *   60..69  Recurrent memory trace — ten channels of self-feedback state.
 *
 * Tier 6 tail layout:
 *   70..76   Self-radio — 7 learned self-communication channels (Tier 2 only).
 *   77..97   Teammate radio — 3 slots × 7 channels each.
 *   98..101  Own-car tire health — frontLeft, frontRight, rearLeft, rearRight.
 *  102..109  Pit/strategy state — 8 channels.
 *  110..130  Opponent perception — 3 slots × 7 ego-relative channels each.
 *
 * Action vector layout:
 *   0  throttle (THR)
 *   1  steer     (STR)
 *   2..8  radio-write channels for Tier 3+ controllers.
 */

import {
  TIER6_TOTAL_INPUT_SIZE,
  TOTAL_TIER4_INPUT_SIZE,
} from '../../controller/observation.assembler';

/** Number of observation inputs consumed by the Tier 1 racing network. */
export const RACING_INPUT_SIZE = 70;

/** Number of channels in the Tier 2 self-radio tail. */
const TIER_TWO_SELF_RADIO_CHANNEL_COUNT = 7;

/** Number of action outputs produced by the Tier 1 racing network. */
export const RACING_OUTPUT_SIZE = 2;

/** Hidden-node count above which the racing network view switches to an abstract cluster/density LOD. */
export const RACING_NETWORK_LOD_HIDDEN_NODE_THRESHOLD = 2048;

/** Maximum number of local nodes rendered when hovering a hidden node in the LOD network view. */
export const RACING_NETWORK_LOD_HOVER_MAX_LOCAL_NODES = 64;

/** Number of abstract hidden clusters/density bins in the LOD view. */
export const RACING_NETWORK_LOD_HIDDEN_CLUSTER_COUNT = 4;

/** Total left padding reserved for the input label panel (band + chip + gap). */
export const RACING_LABEL_LEFT_PADDING_PX = 176;

/** Width of the colored group band rectangle. */
export const RACING_LABEL_BAND_WIDTH_PX = 12;

/** X offset where the group band starts. */
export const RACING_LABEL_BAND_START_PX = 3;

/** X offset where per-node chip labels start. */
export const RACING_LABEL_CHIP_LEFT_PX = 19;

/** Gap between the right edge of a chip and the node's left edge. */
export const RACING_LABEL_CHIP_RIGHT_GAP_PX = 8;

/** Font size for chip label text. */
export const RACING_LABEL_CHIP_FONT_SIZE_PX = 9;

/** Corner radius for chip label rectangles. */
export const RACING_LABEL_CHIP_RADIUS_PX = 3;

/** Vertical breathing room added above and below a group's node span. */
export const RACING_LABEL_CHIP_VERTICAL_GAP_PX = 4;

/** Top offset of the weight legend from the canvas edge. */
export const RACING_LEGEND_TOP_OFFSET_PX = 8;

/** Right offset of the weight legend from the canvas edge. */
export const RACING_LEGEND_RIGHT_OFFSET_PX = 10;

/** Height of each legend item row. */
export const RACING_LEGEND_ITEM_HEIGHT_PX = 13;

/** Vertical gap between legend items. */
export const RACING_LEGEND_ITEM_GAP_PX = 4;

/** Size of the colored swatch square inside each legend row. */
export const RACING_LEGEND_SWATCH_SIZE_PX = 10;

/** Font size for legend text labels. */
export const RACING_LEGEND_TEXT_FONT_SIZE_PX = 9;

/** Monospace font stack used throughout the label panel. */
export const RACING_MONOSPACE_FONT = '"Courier New", Courier, monospace';

/** Bright neon underlay color used behind racing network connection strokes. */
export const RACING_NETWORK_CONNECTION_UNDERLAY_COLOR = '#0f5a8a';

/**
 * Racing-specific connection layer style override.
 *
 * The shared Flappy visualizer defaults to a low default alpha (0.3) so dense
 * Flappy networks stay legible. Racing's right-sidebar panel uses a much
 * smaller controller snapshot against the same dark background, so connections
 * are intentionally brighter and slightly thicker so the topology reads as a
 * vivid neon graph instead of disappearing into the background.
 */
export const RACING_NETWORK_CONNECTION_LAYER_STYLE = {
  lineWidthPx: 2.2,
  defaultConnectionOpacity: 0.72,
  dimmedConnectionOpacity: 0.28,
  highlightConnectionOpacity: 1,
  underlayColor: RACING_NETWORK_CONNECTION_UNDERLAY_COLOR,
  underlayOpacity: 0.42,
  underlayLineWidthPx: 5,
} as const;

/**
 * Light neon ramp used for group band fills.
 *
 * Reuses the same Flappy Bird light neon ramp so the racing panel feels visually
 * consistent with the other browser demos. Each group picks a deterministic
 * color from this ramp in index order.
 */
export const RACING_GROUP_COLORS: readonly {
  bandFill: string;
  accent: string;
}[] = [
  { bandFill: '#7dffd2', accent: '#4dffc2' },
  { bandFill: '#8dffb7', accent: '#5dff97' },
  { bandFill: '#b8ff8a', accent: '#98ff5a' },
  { bandFill: '#ddff8a', accent: '#c9ff5a' },
  { bandFill: '#fff38a', accent: '#ffea5a' },
  { bandFill: '#ffd98a', accent: '#ffca5a' },
  { bandFill: '#ffc18a', accent: '#ffac5a' },
  { bandFill: '#ffaa8a', accent: '#ff8a6a' },
  { bandFill: '#ff8a9a', accent: '#ff5a7a' },
  { bandFill: '#d78aff', accent: '#c05aff' },
  { bandFill: '#8aa2ff', accent: '#5a82ff' },
  { bandFill: '#8ae8ff', accent: '#5adfff' },
  { bandFill: '#8affc1', accent: '#5affb1' },
  { bandFill: '#aaff8a', accent: '#7aff5a' },
] as const;

/** Definition for one per-node chip label, including hover tooltip content. */
export interface RacingNodeDescDef {
  /** Short label displayed inside the chip (max ~14 chars). */
  labelLines: readonly string[];
  /** Tooltip heading shown on hover. */
  tooltipHeading: string;
  /** Tooltip body paragraphs shown on hover. */
  tooltipBodyParagraphs: readonly string[];
}

/** Definition for one input semantic group, covering band and chip metadata. */
export interface RacingInputGroupDef {
  /** Short uppercase group title. */
  label: string;
  /** Lines used for the rotated band text. */
  labelLines: readonly string[];
  /** Number of consecutive input nodes belonging to this group. */
  nodeCount: number;
  /** Tooltip heading for the group band. */
  tooltipHeading: string;
  /** Tooltip body for the group band. */
  tooltipBodyParagraphs: readonly string[];
  /** Per-node label + tooltip definitions ordered to match the input vector. */
  nodeDescriptions: readonly RacingNodeDescDef[];
}

/**
 * Ordered input group definitions for the Tier 1 racing observation vector.
 *
 * The seven groups correspond to CAR STATE, five LOOK AHEAD segments, and the
 * recurrent MEMORY trace. Group ordering matches the observation vector so the
 * visualizer bands align exactly with the network's input shelf.
 */
export const RACING_INPUT_GROUP_DEFS: readonly RacingInputGroupDef[] = [
  {
    label: 'CAR STATE',
    labelLines: ['CAR', 'STATE'],
    nodeCount: 20,
    tooltipHeading: 'Car State Scalars',
    tooltipBodyParagraphs: [
      'The car state group encodes the immediate driving context as twenty normalized scalars.',
      'These channels give the network a compact but complete snapshot of where the car is, how fast it is moving, and how it is oriented relative to the track and the optimal racing line.',
    ],
    nodeDescriptions: [
      {
        labelLines: ['carX'],
        tooltipHeading: 'carX',
        tooltipBodyParagraphs: [
          'Normalized world X coordinate of the car center.',
        ],
      },
      {
        labelLines: ['carY'],
        tooltipHeading: 'carY',
        tooltipBodyParagraphs: [
          'Normalized world Y coordinate of the car center.',
        ],
      },
      {
        labelLines: ['sinHeading'],
        tooltipHeading: 'sinHeading',
        tooltipBodyParagraphs: [
          'Sine of the car heading angle, keeping the angular signal continuous and easy to learn.',
        ],
      },
      {
        labelLines: ['cosHeading'],
        tooltipHeading: 'cosHeading',
        tooltipBodyParagraphs: [
          'Cosine of the car heading angle, paired with sinHeading to encode orientation without wrap discontinuities.',
        ],
      },
      {
        labelLines: ['fwdSpeed'],
        tooltipHeading: 'forwardSpeed',
        tooltipBodyParagraphs: [
          'Forward speed along the car body axis, normalized to world units per second.',
        ],
      },
      {
        labelLines: ['latSpeed'],
        tooltipHeading: 'lateralSpeed',
        tooltipBodyParagraphs: [
          'Lateral speed perpendicular to the car body axis; high magnitudes indicate sliding or drifting.',
        ],
      },
      {
        labelLines: ['speed'],
        tooltipHeading: 'speed',
        tooltipBodyParagraphs: [
          'Overall scalar speed of the car, combining forward and lateral components.',
        ],
      },
      {
        labelLines: ['yawRate'],
        tooltipHeading: 'yawRate',
        tooltipBodyParagraphs: [
          'Rate of change of the heading angle; useful for recognizing spin-outs and over-rotation.',
        ],
      },
      {
        labelLines: ['slipAngle'],
        tooltipHeading: 'slipAngle',
        tooltipBodyParagraphs: [
          'Angle between the car heading and the velocity vector; large values mean the tires are sliding.',
        ],
      },
      {
        labelLines: ['progress'],
        tooltipHeading: 'progress',
        tooltipBodyParagraphs: [
          'Normalized longitudinal progress along the current lap, used as a coarse position signal.',
        ],
      },
      {
        labelLines: ['lapProgress'],
        tooltipHeading: 'lapProgress',
        tooltipBodyParagraphs: [
          'Fine-grained lap progress fraction, updated every physics tick.',
        ],
      },
      {
        labelLines: ['bndLeft'],
        tooltipHeading: 'boundaryLeft',
        tooltipBodyParagraphs: [
          'Distance from the car to the left full-road boundary in world units.',
        ],
      },
      {
        labelLines: ['bndRight'],
        tooltipHeading: 'boundaryRight',
        tooltipBodyParagraphs: [
          'Distance from the car to the right full-road boundary in world units.',
        ],
      },
      {
        labelLines: ['bndBal'],
        tooltipHeading: 'boundaryBalance',
        tooltipBodyParagraphs: [
          'Signed balance between left and right full-road boundary distances; zero means centered across the whole road.',
        ],
      },
      {
        labelLines: ['hazDist'],
        tooltipHeading: 'hazardDistance',
        tooltipBodyParagraphs: [
          'Distance to the nearest hazard or obstacle ahead on the track.',
        ],
      },
      {
        labelLines: ['wpDist'],
        tooltipHeading: 'waypointDistance',
        tooltipBodyParagraphs: [
          'Distance to the next waypoint or progression gate, used for target-seeking behavior.',
        ],
      },
      {
        labelLines: ['optOffset'],
        tooltipHeading: 'optimalLineLateralOffset',
        tooltipBodyParagraphs: [
          'Lateral offset from the inner-lane centerline (optimal racing line); negative is left, positive is right.',
        ],
      },
      {
        labelLines: ['optHead'],
        tooltipHeading: 'optimalLineHeadingError',
        tooltipBodyParagraphs: [
          'Heading error relative to the inner-lane centerline tangent; the controller uses this to align the car.',
        ],
      },
      {
        labelLines: ['tgtSpeed'],
        tooltipHeading: 'targetSpeed',
        tooltipBodyParagraphs: [
          'Desired speed for the current track segment, giving the throttle network a clear setpoint.',
        ],
      },
      {
        labelLines: ['spdDelta'],
        tooltipHeading: 'speedDelta',
        tooltipBodyParagraphs: [
          'Difference between current speed and target speed; negative means the car should accelerate.',
        ],
      },
    ],
  },
  ...buildLookAheadGroupDefinitions(0),
  ...buildLookAheadGroupDefinitions(1),
  ...buildLookAheadGroupDefinitions(2),
  ...buildLookAheadGroupDefinitions(3),
  ...buildLookAheadGroupDefinitions(4),
  {
    label: 'MEMORY',
    labelLines: ['MEMORY'],
    nodeCount: 10,
    tooltipHeading: 'Memory Trace',
    tooltipBodyParagraphs: [
      'The memory group preserves ten recurrent self-feedback channels from the previous step.',
      'These channels let the controller maintain short-term state such as recent speed history, steering tendency, and progress momentum without relying only on feed-forward context.',
    ],
    nodeDescriptions: Array.from(
      { length: 10 },
      (_unusedValue, memoryIndex) => ({
        labelLines: [`mem ${memoryIndex}`],
        tooltipHeading: `Memory ${memoryIndex}`,
        tooltipBodyParagraphs: [
          `Recurrent memory channel ${memoryIndex} feeds the network's own prior-step activation back into the current observation.`,
        ],
      }),
    ),
  },
] as const;

/**
 * Short labels for the action output nodes, ordered to match the action vector.
 * Tier 1–2 controllers only use the first two entries (throttle and steer);
 * Tier 3+ controllers append seven radio-write channels.
 */
export const RACING_OUTPUT_LABELS: readonly string[] = [
  'THR',
  'STR',
  'R0',
  'R1',
  'R2',
  'R3',
  'R4',
  'R5',
  'R6',
] as const;

/**
 * Builds one look-ahead input group definition for the given segment index.
 *
 * Each of the five look-ahead segments occupies eight consecutive channels and
 * describes the track geometry ahead of the car in a local frame.
 *
 * @param segmentIndex - Zero-based look-ahead segment (0..4).
 * @returns One group definition with eight node descriptions.
 */
function buildLookAheadGroupDefinitions(
  segmentIndex: number,
): readonly RacingInputGroupDef[] {
  return [
    {
      label: `LOOK AHEAD ${segmentIndex}`,
      labelLines: ['LOOK', `AHEAD ${segmentIndex}`],
      nodeCount: 8,
      tooltipHeading: `Look-Ahead Segment ${segmentIndex}`,
      tooltipBodyParagraphs: [
        `Look-ahead segment ${segmentIndex} encodes track geometry ${segmentIndex + 1} sample(s) ahead of the car in a local coordinate frame.`,
        'Together these five segments give the network a short-term preview of upcoming curvature, width changes, and distance to the next segment.',
      ],
      nodeDescriptions: [
        {
          labelLines: ['relX'],
          tooltipHeading: 'relX',
          tooltipBodyParagraphs: [
            'Local X coordinate of the look-ahead segment relative to the car.',
          ],
        },
        {
          labelLines: ['relY'],
          tooltipHeading: 'relY',
          tooltipBodyParagraphs: [
            'Local Y coordinate of the look-ahead segment relative to the car.',
          ],
        },
        {
          labelLines: ['nextRelX'],
          tooltipHeading: 'nextRelX',
          tooltipBodyParagraphs: [
            'Local X coordinate of the segment endpoint, used to infer the approaching direction.',
          ],
        },
        {
          labelLines: ['nextRelY'],
          tooltipHeading: 'nextRelY',
          tooltipBodyParagraphs: [
            'Local Y coordinate of the segment endpoint, used to infer the approaching direction.',
          ],
        },
        {
          labelLines: ['sinTng'],
          tooltipHeading: 'sinTangent',
          tooltipBodyParagraphs: [
            'Sine of the track tangent angle at the look-ahead point, encoding upcoming heading change.',
          ],
        },
        {
          labelLines: ['cosTng'],
          tooltipHeading: 'cosTangent',
          tooltipBodyParagraphs: [
            'Cosine of the track tangent angle at the look-ahead point, paired with sinTangent for smooth orientation encoding.',
          ],
        },
        {
          labelLines: ['width'],
          tooltipHeading: 'trackWidth',
          tooltipBodyParagraphs: [
            'Track width at the look-ahead point; narrow sections reward cautious steering.',
          ],
        },
        {
          labelLines: ['dist'],
          tooltipHeading: 'distance',
          tooltipBodyParagraphs: [
            'Distance along the track spline from the car to the look-ahead point.',
          ],
        },
      ],
    },
  ];
}

/** Number of teammate-radio slots encoded in the Tier 6 observation tail. */
const TEAMMATE_RADIO_SLOT_COUNT = 3;
/** Number of opponent-perception slots encoded in the Tier 6 observation tail. */
const OPPONENT_SLOT_COUNT = 3;
/** Number of channels per teammate or opponent radio/perception slot. */
const SLOT_CHANNEL_COUNT = 7;
/** Number of own-car tire-health channels in the Tier 6 observation tail. */
const TIRE_HEALTH_CHANNEL_COUNT = 4;
/** Number of pit/strategy channels in the Tier 6 observation tail. */
const PIT_STRATEGY_CHANNEL_COUNT = 8;

/** Ordered short labels for one teammate-radio slot. */
const TEAMMATE_RADIO_CHANNEL_NAMES = [
  'posX',
  'posY',
  'sinHead',
  'speed',
  'relOffX',
  'relOffY',
  'relHeadSin',
] as const;

/** Tooltip copy for each channel in a teammate-radio slot. */
const TEAMMATE_RADIO_CHANNEL_TOOLTIPS = [
  'Absolute world X coordinate of the teammate, normalized to track position scale.',
  'Absolute world Y coordinate of the teammate, normalized to track position scale.',
  'Sine of the teammate heading angle.',
  'Scalar speed of the teammate, normalized to world units per second.',
  'Ego-relative X offset from the focal car to the teammate, normalized to track position scale.',
  'Ego-relative Y offset from the focal car to the teammate, normalized to track position scale.',
  'Sine of the heading difference between the teammate and the focal car.',
] as const;

/** Ordered short labels for the Tier 2 self-radio group. */
const SELF_RADIO_CHANNEL_NAMES = [
  'selfC0',
  'selfC1',
  'selfC2',
  'selfC3',
  'selfC4',
  'selfC5',
  'selfC6',
] as const;

/** Tooltip copy for each self-radio channel. */
const SELF_RADIO_CHANNEL_TOOLTIPS = [
  'Learned self-communication channel 0: a latent broadcast the controller writes to itself each tick.',
  'Learned self-communication channel 1: a latent broadcast the controller writes to itself each tick.',
  'Learned self-communication channel 2: a latent broadcast the controller writes to itself each tick.',
  'Learned self-communication channel 3: a latent broadcast the controller writes to itself each tick.',
  'Learned self-communication channel 4: a latent broadcast the controller writes to itself each tick.',
  'Learned self-communication channel 5: a latent broadcast the controller writes to itself each tick.',
  'Learned self-communication channel 6: a latent broadcast the controller writes to itself each tick.',
] as const;

/** Ordered short labels for one opponent-perception slot. */
const OPPONENT_CHANNEL_NAMES = [
  'relFwd',
  'relLeft',
  'sinHdD',
  'cosHdD',
  'relSpdF',
  'relSpdL',
  'dist',
] as const;

/** Tooltip copy for each channel in an opponent-perception slot. */
const OPPONENT_CHANNEL_TOOLTIPS = [
  'Forward distance to the opponent in the focal car body frame, normalized to track position scale.',
  'Left distance to the opponent in the focal car body frame, normalized to track position scale.',
  'Sine of the opponent heading minus the focal car heading.',
  'Cosine of the opponent heading minus the focal car heading.',
  'Relative forward speed between the opponent and the focal car, normalized to speed scale.',
  'Relative lateral speed between the opponent and the focal car, normalized to lateral speed scale.',
  'Euclidean distance between the opponent and the focal car, normalized to distance scale.',
] as const;

/** Ordered short labels for the own-car tire-health group. */
const TIRE_HEALTH_CHANNEL_NAMES = [
  'frontLeft',
  'frontRight',
  'rearLeft',
  'rearRight',
] as const;

/** Tooltip copy for each tire-health channel. */
const TIRE_HEALTH_CHANNEL_TOOLTIPS = [
  'Front-left tire health, where 1 is fully healthy and 0 is failed.',
  'Front-right tire health, where 1 is fully healthy and 0 is failed.',
  'Rear-left tire health, where 1 is fully healthy and 0 is failed.',
  'Rear-right tire health, where 1 is fully healthy and 0 is failed.',
] as const;

/** Ordered short labels for the pit/strategy group. */
const PIT_STRATEGY_CHANNEL_NAMES = [
  'pitDist01',
  'pitOcc',
  'lapsSince',
  'tmPit',
  'degRate',
  'estFail',
  'reserved1',
  'reserved2',
] as const;

/** Tooltip copy for each pit/strategy channel. */
const PIT_STRATEGY_CHANNEL_TOOLTIPS = [
  'Normalized distance to the pit entrance.',
  'Pit-box occupancy for the focal car team (0 empty, 1 occupied).',
  'Laps since the focal car last pitted, normalized to unit interval.',
  'Team pit-box occupied by any team member (0 free, 1 occupied).',
  'Current tire degradation rate, normalized to unit interval.',
  'Estimated laps before tire failure, normalized to unit interval.',
  'Reserved pit/strategy expansion channel.',
  'Reserved pit/strategy expansion channel.',
] as const;

/**
 * Builds the Tier 2 self-radio input group definition.
 *
 * The seven self-radio channels are learned latent broadcasts that the
 * controller emits and reads back on the next tick. They have no fixed
 * semantic meaning; instead they let the network evolve its own short-term
 * memory protocol on top of the base observation vector.
 *
 * @returns One group definition with seven node descriptions.
 */
function buildSelfRadioGroupDefinition(): readonly RacingInputGroupDef[] {
  return [
    {
      label: 'SELF RADIO',
      labelLines: ['SELF', 'RADIO'],
      nodeCount: TIER_TWO_SELF_RADIO_CHANNEL_COUNT,
      tooltipHeading: 'Self-Radio Learned Channels',
      tooltipBodyParagraphs: [
        'The self-radio group contains seven learned latent channels that the controller writes to its own output tail and reads back on the next observation.',
        'Unlike teammate or opponent slots, these channels have no fixed semantic label: they are free for the network to use as recurrent memory, intention state, or any other compact self-signal that improves driving performance.',
      ],
      nodeDescriptions: SELF_RADIO_CHANNEL_NAMES.map(
        (channelName, channelIndex) => ({
          labelLines: [channelName],
          tooltipHeading: channelName,
          tooltipBodyParagraphs: [
            SELF_RADIO_CHANNEL_TOOLTIPS[channelIndex] ??
              `Self-radio channel ${channelIndex}.`,
          ],
        }),
      ),
    },
  ];
}

/**
 * Builds one teammate-radio input group definition for the given slot index.
 *
 * @param slotIndex - Zero-based teammate slot (0..2).
 * @returns One group definition with seven node descriptions.
 */
function buildTeammateRadioGroupDefinition(
  slotIndex: number,
): readonly RacingInputGroupDef[] {
  return [
    {
      label: `TEAMMATE ${slotIndex}`,
      labelLines: ['TEAM', `MATE ${slotIndex}`],
      nodeCount: SLOT_CHANNEL_COUNT,
      tooltipHeading: `Teammate Radio Slot ${slotIndex}`,
      tooltipBodyParagraphs: [
        `Teammate radio slot ${slotIndex} encodes the absolute pose, speed, and ego-relative offset of one team car.`,
        'Unused slots are zero-padded so the controller can rely on a fixed-width team-communication tail.',
      ],
      nodeDescriptions: TEAMMATE_RADIO_CHANNEL_NAMES.map(
        (channelName, channelIndex) => ({
          labelLines: [`TM${slotIndex} ${channelName}`],
          tooltipHeading: `Teammate ${slotIndex} ${channelName}`,
          tooltipBodyParagraphs: [
            TEAMMATE_RADIO_CHANNEL_TOOLTIPS[channelIndex] ??
              `Teammate radio channel ${channelIndex}.`,
          ],
        }),
      ),
    },
  ];
}

/**
 * Builds the own-car tire-health input group definition.
 *
 * @returns One group definition with four node descriptions.
 */
function buildTireHealthGroupDefinition(): readonly RacingInputGroupDef[] {
  return [
    {
      label: 'TIRE HEALTH',
      labelLines: ['TIRE', 'HEALTH'],
      nodeCount: TIRE_HEALTH_CHANNEL_COUNT,
      tooltipHeading: 'Own-Car Tire Health',
      tooltipBodyParagraphs: [
        'The tire-health group exposes the current degradation of each wheel corner.',
        'These four channels let the network adapt throttle and steering as grip drops.',
      ],
      nodeDescriptions: TIRE_HEALTH_CHANNEL_NAMES.map(
        (channelName, channelIndex) => ({
          labelLines: [channelName],
          tooltipHeading: channelName,
          tooltipBodyParagraphs: [
            TIRE_HEALTH_CHANNEL_TOOLTIPS[channelIndex] ??
              `Tire health channel ${channelIndex}.`,
          ],
        }),
      ),
    },
  ];
}

/**
 * Builds the pit/strategy input group definition.
 *
 * @returns One group definition with eight node descriptions.
 */
function buildPitStrategyGroupDefinition(): readonly RacingInputGroupDef[] {
  return [
    {
      label: 'PIT / STRATEGY',
      labelLines: ['PIT', 'STRATEGY'],
      nodeCount: PIT_STRATEGY_CHANNEL_COUNT,
      tooltipHeading: 'Pit and Strategy State',
      tooltipBodyParagraphs: [
        'The pit/strategy group carries coarse race-management signals: pit distance, occupancy, tire degradation, and failure estimates.',
        'These eight channels give the network enough context to learn when to pit without exposing full race logic.',
      ],
      nodeDescriptions: PIT_STRATEGY_CHANNEL_NAMES.map(
        (channelName, channelIndex) => ({
          labelLines: [channelName],
          tooltipHeading: channelName,
          tooltipBodyParagraphs: [
            PIT_STRATEGY_CHANNEL_TOOLTIPS[channelIndex] ??
              `Pit/strategy channel ${channelIndex}.`,
          ],
        }),
      ),
    },
  ];
}

/**
 * Builds one opponent-perception input group definition for the given slot index.
 *
 * @param slotIndex - Zero-based opponent slot (0..2).
 * @returns One group definition with seven node descriptions.
 */
function buildOpponentPerceptionGroupDefinition(
  slotIndex: number,
): readonly RacingInputGroupDef[] {
  return [
    {
      label: `OPPONENT ${slotIndex}`,
      labelLines: ['OPP', `ONENT ${slotIndex}`],
      nodeCount: SLOT_CHANNEL_COUNT,
      tooltipHeading: `Opponent Perception Slot ${slotIndex}`,
      tooltipBodyParagraphs: [
        `Opponent perception slot ${slotIndex} encodes ego-relative geometry and relative speed for one rival car.`,
        'Missing slots are zero-padded so the same controller can race 1v1, 2v2, or 3v3 without changing shape.',
      ],
      nodeDescriptions: OPPONENT_CHANNEL_NAMES.map(
        (channelName, channelIndex) => ({
          labelLines: [`OPP${slotIndex} ${channelName}`],
          tooltipHeading: `Opponent ${slotIndex} ${channelName}`,
          tooltipBodyParagraphs: [
            OPPONENT_CHANNEL_TOOLTIPS[channelIndex] ??
              `Opponent perception channel ${channelIndex}.`,
          ],
        }),
      ),
    },
  ];
}

/**
 * Resolves the racing input group definitions matching the requested input size.
 *
 * The default (Tier 1) set covers the first 70 channels and keeps all pre-existing
 * visualizer tests unchanged. When the network consumes the full Tier 6 vector,
 * this resolver returns the extended set that labels every teammate-radio,
 * tire-health, pit/strategy, and opponent-perception channel.
 *
 * @param inputSize - Optional network input size; defaults to the Tier 1 width.
 * @returns Ordered racing input group definitions for the shared visualizer.
 */
export function resolveRacingInputGroupDefinitions(
  inputSize?: number,
): readonly RacingInputGroupDef[] {
  const teammateRadioGroups = Array.from(
    { length: TEAMMATE_RADIO_SLOT_COUNT },
    (_unusedValue, slotIndex) => buildTeammateRadioGroupDefinition(slotIndex),
  ).flat();
  const opponentGroups = Array.from(
    { length: OPPONENT_SLOT_COUNT },
    (_unusedValue, slotIndex) =>
      buildOpponentPerceptionGroupDefinition(slotIndex),
  ).flat();

  if (inputSize === TIER6_TOTAL_INPUT_SIZE) {
    return [
      ...RACING_INPUT_GROUP_DEFS,
      ...teammateRadioGroups,
      ...buildTireHealthGroupDefinition(),
      ...buildPitStrategyGroupDefinition(),
      ...opponentGroups,
    ];
  }

  if (inputSize === TOTAL_TIER4_INPUT_SIZE) {
    return [
      ...RACING_INPUT_GROUP_DEFS,
      ...teammateRadioGroups,
      ...buildTireHealthGroupDefinition(),
      ...buildPitStrategyGroupDefinition(),
    ];
  }

  const tier3InputSize =
    RACING_INPUT_SIZE + TEAMMATE_RADIO_SLOT_COUNT * SLOT_CHANNEL_COUNT;

  if (inputSize === tier3InputSize) {
    return [...RACING_INPUT_GROUP_DEFS, ...teammateRadioGroups];
  }

  const tier2InputSize = RACING_INPUT_SIZE + TIER_TWO_SELF_RADIO_CHANNEL_COUNT;

  if (inputSize === tier2InputSize) {
    return [...RACING_INPUT_GROUP_DEFS, ...buildSelfRadioGroupDefinition()];
  }

  return RACING_INPUT_GROUP_DEFS;
}
