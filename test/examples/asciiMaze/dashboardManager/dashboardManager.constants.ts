/**
 * Shared sizing, formatting, and top-N limits for the ASCII maze dashboard.
 *
 * These values are kept in one place so rendering, archive output, and
 * telemetry helpers stay visually and semantically aligned.
 */
export const DASHBOARD_MANAGER_CONSTANTS = {
  HISTORY_MAX: 500,
  FRAME_INNER_WIDTH: 148,
  LEFT_PADDING: 7,
  RIGHT_PADDING: 1,
  CONTENT_WIDTH: 148 - 7 - 1,
  STAT_LABEL_WIDTH: 28,
  ARCHIVE_SPARK_WIDTH: 64,
  GENERAL_SPARK_WIDTH: 64,
  SOLVED_LABEL_WIDTH: 22,
  HISTORY_EXPORT_WINDOW: 200,
  SPARK_BLOCKS: Object.freeze(['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']),
  DELTA_EPSILON: 1e-9,
  TOP_OPERATOR_LIMIT: 6,
  TOP_MUTATION_LIMIT: 8,
  TOP_SPECIES_LIMIT: 5,
  LAYER_INFER_LOOP_MULTIPLIER: 4,
  LABEL_PATH_EFF: 'Path efficiency',
  LABEL_PATH_OVER: 'Path overhead',
  LABEL_UNIQUE: 'Unique cells visited',
  LABEL_REVISITS: 'Cells revisited',
  LABEL_STEPS: 'Steps',
  LABEL_FITNESS: 'Fitness',
  LABEL_ARCH: 'Architecture',
  FRAME_SINGLE_LINE_CHAR: '═',
  FRAME_BRIDGE_TOP: '╦════════════╦',
  FRAME_BRIDGE_BOTTOM: '╩════════════╩',
  EVOLVING_SECTION_LINE: '══════════════════════',
} as const;
