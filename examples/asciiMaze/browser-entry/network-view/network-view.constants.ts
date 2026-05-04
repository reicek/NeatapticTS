/**
 * Visual constants and input/output label definitions for the ASCII Maze
 * network-view panel.
 *
 * The maze agent consumes a 6-element observation vector and produces 4 action
 * logits. These constants describe both the visual layout of the label panel
 * (geometry, palette) and the educational content (headings, tooltip paragraphs)
 * shown for each input group and individual input node.
 *
 * Observation vector layout (in order):
 *   0  compassScalar  — BFS-preferred direction as [0, 0.75] scalar
 *   1  openN          — North corridor passable (0 / 1)
 *   2  openE          — East corridor passable (0 / 1)
 *   3  openS          — South corridor passable (0 / 1)
 *   4  openW          — West corridor passable (0 / 1)
 *   5  progressDelta  — Normalised step-progress signal
 *
 * Action vector layout:
 *   0  North  1  East  2  South  3  West
 */

/** Number of observation inputs consumed by the maze network. */
export const MAZE_INPUT_SIZE = 6;

/** Number of action outputs produced by the maze network. */
export const MAZE_OUTPUT_SIZE = 4;

/** Total left padding reserved for the input label panel (band + chip + gap). */
export const MAZE_LABEL_LEFT_PADDING_PX = 176;

/** Width of the colored group band rectangle. */
export const MAZE_LABEL_BAND_WIDTH_PX = 12;

/** X offset where the group band starts. */
export const MAZE_LABEL_BAND_START_PX = 3;

/** X offset where per-node chip labels start. */
export const MAZE_LABEL_CHIP_LEFT_PX = 19;

/** Gap between the right edge of a chip and the node's left edge. */
export const MAZE_LABEL_CHIP_RIGHT_GAP_PX = 8;

/** Font size for chip label text. */
export const MAZE_LABEL_CHIP_FONT_SIZE_PX = 9;

/** Corner radius for chip label rectangles. */
export const MAZE_LABEL_CHIP_RADIUS_PX = 3;

/** Vertical breathing room added above and below a group's node span. */
export const MAZE_LABEL_CHIP_VERTICAL_GAP_PX = 4;

/** Top offset of the weight legend from the canvas edge. */
export const MAZE_LEGEND_TOP_OFFSET_PX = 8;

/** Right offset of the weight legend from the canvas edge. */
export const MAZE_LEGEND_RIGHT_OFFSET_PX = 10;

/** Height of each legend item row. */
export const MAZE_LEGEND_ITEM_HEIGHT_PX = 13;

/** Vertical gap between legend items. */
export const MAZE_LEGEND_ITEM_GAP_PX = 4;

/** Size of the colored swatch square inside each legend row. */
export const MAZE_LEGEND_SWATCH_SIZE_PX = 10;

/** Font size for legend text labels. */
export const MAZE_LEGEND_TEXT_FONT_SIZE_PX = 9;

/** Monospace font stack used throughout the label panel. */
export const MAZE_MONOSPACE_FONT = '"Courier New", Courier, monospace';

/**
 * Per-group palette: background fill and accent border/text color.
 *
 * Index matches the ordering of `MAZE_INPUT_GROUP_DEFS`.
 * - 0 HEADING  — cyan
 * - 1 OPENNESS — green
 * - 2 PROGRESS — amber
 */
export const MAZE_GROUP_COLORS: readonly {
  bandFill: string;
  accent: string;
}[] = [
  { bandFill: '#2bd9ff', accent: '#00ccff' },
  { bandFill: '#7bff72', accent: '#00ff88' },
  { bandFill: '#ffd166', accent: '#ffd166' },
] as const;

/** Definition for one per-node chip label, including hover tooltip content. */
export interface MazeNodeDescDef {
  /** Short label displayed inside the chip (max ~14 chars). */
  labelLines: readonly string[];
  /** Tooltip heading shown on hover. */
  tooltipHeading: string;
  /** Tooltip body paragraphs shown on hover. */
  tooltipBodyParagraphs: readonly string[];
}

/** Definition for one input semantic group, covering band and chip metadata. */
export interface MazeInputGroupDef {
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
  nodeDescriptions: readonly MazeNodeDescDef[];
}

/**
 * Ordered input group definitions for the maze observation vector.
 *
 * The three groups correspond to the HEADING, OPENNESS, and PROGRESS semantic
 * families used by `MazeVision.buildInputs6`. Group ordering matches the input
 * vector: HEADING at index 0, OPENNESS at indices 1–4, PROGRESS at index 5.
 */
export const MAZE_INPUT_GROUP_DEFS: readonly MazeInputGroupDef[] = [
  {
    label: 'HEADING',
    labelLines: ['HEAD', 'ING'],
    nodeCount: 1,
    tooltipHeading: 'Heading',
    tooltipBodyParagraphs: [
      'The heading group encodes the BFS-preferred direction at the current cell as one compact scalar rather than a one-hot vector.',
      'A BFS distance map is computed once per maze. The least-cost cardinal neighbour becomes the preferred direction, encoded as 0 = North, 0.25 = East, 0.5 = South, 0.75 = West.',
      'Using a single channel keeps the observation compact without losing navigational intent, and gives the network a stable gradient signal toward the exit even in long, winding corridors.',
    ],
    nodeDescriptions: [
      {
        labelLines: ['Compass scalar'],
        tooltipHeading: 'Compass Scalar',
        tooltipBodyParagraphs: [
          'The compass scalar encodes the BFS-preferred direction as a continuous value in [0, 0.75].',
          'North maps to 0, East to 0.25, South to 0.5, and West to 0.75. When no progress step exists the value defaults to 0.',
          'Because the encoding is ordinal and directionally adjacent values wrap (0 ≈ North ≈ 0.75 West), the network must learn a near-circular reading — a useful inductive bias for corridor navigation.',
        ],
      },
    ],
  },
  {
    label: 'OPENNESS',
    labelLines: ['OPEN', 'NESS'],
    nodeCount: 4,
    tooltipHeading: 'Openness',
    tooltipBodyParagraphs: [
      'The openness group provides one binary passability flag per cardinal direction, giving the network an instant local wall-map.',
      'Together the four channels tell the controller which moves are physically possible without any search: a blocked direction is never worth outputting.',
      'Combined with the compass heading, the network can distinguish "preferred direction is open, go" from "preferred direction is blocked, explore an alternative".',
    ],
    nodeDescriptions: [
      {
        labelLines: ['Open north'],
        tooltipHeading: 'Open North',
        tooltipBodyParagraphs: [
          'Open north is 1 when the cell directly above the agent is within bounds and not a wall, 0 otherwise.',
          'A blocked north combined with a northward compass heading tells the controller it has hit a dead-end from above and must turn.',
          'This binary channel costs nothing to compute and eliminates the need for the network to learn collision avoidance from scratch.',
        ],
      },
      {
        labelLines: ['Open east'],
        tooltipHeading: 'Open East',
        tooltipBodyParagraphs: [
          'Open east is 1 when the cell directly to the right of the agent is passable.',
          'The east channel is especially valuable in grid mazes with long horizontal corridors where the agent must decide whether to continue east or turn.',
          'When both east and west are open but north and south are blocked, the agent is in a horizontal corridor — a pattern the network can recognize without lookahead.',
        ],
      },
      {
        labelLines: ['Open south'],
        tooltipHeading: 'Open South',
        tooltipBodyParagraphs: [
          'Open south is 1 when the cell directly below the agent is passable.',
          'Combined with open north it lets the network detect vertical corridors and choose between continuing or branching at intersections.',
          'A south-only open cell with a southward compass heading is a strong forward-progress signal with no ambiguity.',
        ],
      },
      {
        labelLines: ['Open west'],
        tooltipHeading: 'Open West',
        tooltipBodyParagraphs: [
          'Open west is 1 when the cell directly to the left of the agent is passable.',
          'When all four openness channels are non-zero the agent is at an intersection; when only one is non-zero it is in a cul-de-sac.',
          'The west channel completes the local wall-map and allows the network to infer corridor type from the combination of all four flags.',
        ],
      },
    ],
  },
  {
    label: 'PROGRESS',
    labelLines: ['PROG', 'RESS'],
    nodeCount: 1,
    tooltipHeading: 'Progress',
    tooltipBodyParagraphs: [
      'The progress group collapses the entire step-reward signal into one normalised scalar reflecting how much the agent reduced its BFS distance to the exit on the last move.',
      'Values above 0.5 indicate approach (positive progress); values below 0.5 indicate retreat or a wasted lateral step; exactly 0.5 is the neutral baseline.',
      'This single channel acts as a one-step short-term memory: it discourages oscillation by making the network aware of whether its most recent decision was productive.',
    ],
    nodeDescriptions: [
      {
        labelLines: ['Progress delta'],
        tooltipHeading: 'Progress Delta',
        tooltipBodyParagraphs: [
          'Progress delta is a normalised measure of how much the agent reduced its BFS distance to the exit on the previous step.',
          'The raw delta is clipped to ±2 and scaled to a [0.1, 0.9] range so the network always sees a well-conditioned gradient: near 0.9 means a large improvement; near 0.1 means the agent moved further from the exit.',
          'Because the signal lags by one step it teaches momentum exploitation: a high progress value encourages the network to maintain or repeat the current direction rather than reverting.',
        ],
      },
    ],
  },
] as const;

/**
 * Short labels for the four output nodes, ordered to match the action vector.
 * Used to annotate output nodes on the right side of the graph.
 */
export const MAZE_OUTPUT_LABELS: readonly string[] = [
  'N',
  'E',
  'S',
  'W',
] as const;
