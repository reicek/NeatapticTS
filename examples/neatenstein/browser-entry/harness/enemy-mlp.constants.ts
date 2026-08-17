/**
 * Constants for the Neatenstein MLP enemy backend, arms-race runner, and
 * shared harness utilities.
 *
 * Extracts mutation type strings, MLP output labels and indices, dash
 * threshold, champion seed prime, output precision, hash-seed prime, LCG
 * multiplier, Box-Muller floor, genome size ranges, replay pressure scalar,
 * and arms-race RNG discriminator strings so that source modules contain
 * only logic with no inline magic numbers.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Mutation operator types (enemy-mlp.ts)
// ---------------------------------------------------------------------------

/** Mutation operator type for single-weight mutation. */
export const MUTATION_WEIGHT = 'weight';

/** Mutation operator type for multi-weight mutation. */
export const MUTATION_WEIGHTS = 'weights';

/** Mutation operator type for perturb mutation. */
export const MUTATION_PERTURB = 'perturb';

/** Allowed weight-only mutation operator types for the MLP enemy backend. */
export const MLP_ALLOWED_MUTATION_TYPES: readonly string[] = [
  MUTATION_WEIGHT,
  MUTATION_WEIGHTS,
  MUTATION_PERTURB,
];

// ---------------------------------------------------------------------------
// MLP output labels (enemy-mlp.ts)
// ---------------------------------------------------------------------------

/** Output label for the move action. */
export const MLP_OUTPUT_MOVE = 'move';

/** Output label for the strafe action. */
export const MLP_OUTPUT_STRAFE = 'strafe';

/** Output label for the turn action. */
export const MLP_OUTPUT_TURN = 'turn';

/** Output label for the fire action. */
export const MLP_OUTPUT_FIRE = 'fire';

/** Ordered output labels for the MLP enemy backend. */
export const NEATENSTEIN_MLP_OUTPUT_LABELS: readonly string[] = [
  MLP_OUTPUT_MOVE,
  MLP_OUTPUT_STRAFE,
  MLP_OUTPUT_TURN,
  MLP_OUTPUT_FIRE,
];

// ---------------------------------------------------------------------------
// Main NEAT network output indices (neat-io-config.ts)
// ---------------------------------------------------------------------------

/** Output index for the strafe (move.x) channel. */
export const NEAT_OUTPUT_INDEX_STRAFE = 0;

/** Output index for the forward/back (move.y) channel. */
export const NEAT_OUTPUT_INDEX_FORWARD = 1;

/** Output index for the turn (lookDelta) channel. */
export const NEAT_OUTPUT_INDEX_TURN = 2;

/** Output index for the fire channel. */
export const NEAT_OUTPUT_INDEX_FIRE = 3;

/** Output index for the dash channel. */
export const NEAT_OUTPUT_INDEX_DASH = 4;

/** Threshold above which the dash output activates. */
export const DASH_THRESHOLD = 0.5;

// ---------------------------------------------------------------------------
// Champion & precision constants (enemy-mlp.ts)
// ---------------------------------------------------------------------------

/** Prime multiplier used to derive a generation-unique champion seed. */
export const CHAMPION_SEED_PRIME = 7919;

/** Decimal precision used when rounding interpreted MLP output values. */
export const OUTPUT_PRECISION = 6;

// ---------------------------------------------------------------------------
// Hash-seed prime (hash-seed.ts)
// ---------------------------------------------------------------------------

/** Prime multiplier used by the deterministic hash-seed function. */
export const HASH_SEED_PRIME = 100003;

// ---------------------------------------------------------------------------
// LCG seed multiplier (seed-pack.ts)
// ---------------------------------------------------------------------------

/** Golden-ratio-derived multiplier for LCG seed initialization. */
export const LCG_SEED_MULTIPLIER = 2_654_435_761;

// ---------------------------------------------------------------------------
// Box-Muller floor (enemy-warmstart.mlp-math.utils.ts)
// ---------------------------------------------------------------------------

/** Minimum value passed to Math.max to avoid log(0) in the Box-Muller transform. */
export const BOX_MULLER_FLOOR = 1e-10;

// ---------------------------------------------------------------------------
// Main genome size ranges (arms-race.ts)
// ---------------------------------------------------------------------------

/** Minimum node count for a generated main-agent genome. */
export const MAIN_GENOME_NODE_MIN = 10;

/** Span of the node count range for a generated main-agent genome. */
export const MAIN_GENOME_NODE_SPAN = 20;

/** Minimum connection count for a generated main-agent genome. */
export const MAIN_GENOME_CONNECTION_MIN = 10;

/** Span of the connection count range for a generated main-agent genome. */
export const MAIN_GENOME_CONNECTION_SPAN = 30;

// ---------------------------------------------------------------------------
// Replay pressure (arms-race.ts)
// ---------------------------------------------------------------------------

/** Scaling factor for replay-buffer selection pressure per stored death context. */
export const REPLAY_PRESSURE_PER_ENTRY = 0.1;

// ---------------------------------------------------------------------------
// Arms-race RNG discriminators (arms-race.ts)
// ---------------------------------------------------------------------------

/** Discriminator string for replay-driven behavior RNG seeds. */
export const ARMS_RACE_REPLAY = 'replay';

/** Discriminator string for baseline behavior RNG seeds. */
export const ARMS_RACE_BASELINE = 'baseline';

// ---------------------------------------------------------------------------
// Backprop shuffle LCG constants (enemy-warmstart.backprop.utils.ts)
// ---------------------------------------------------------------------------

/** Golden-ratio multiplier for the deterministic shuffle LCG initial state. */
export const SHUFFLE_LCG_SEED_MULTIPLIER = 2_654_435_761;

/** Multiplier for the deterministic shuffle LCG step function. */
export const SHUFFLE_LCG_STEP_MULTIPLIER = 1_103_515_245;

/** Additive constant for the deterministic shuffle LCG step function. */
export const SHUFFLE_LCG_STEP_OFFSET = 12_345;

/** Bit mask for the deterministic shuffle LCG (2³¹ − 1). */
export const SHUFFLE_LCG_MASK = 0x7fffffff;
