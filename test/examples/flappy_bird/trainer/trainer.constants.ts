import { FLAPPY_DEFAULT_RNG_SEED } from '../constants/constants';

/** Default population size used by the Flappy trainer NEAT run. */
export const FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE = 200;

/** Number of elite genomes preserved unchanged each generation. */
export const FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT = 20;

/** Deterministic trainer RNG seed used for reproducible training runs. */
export const FLAPPY_TRAINER_DEFAULT_RNG_SEED = FLAPPY_DEFAULT_RNG_SEED;

/** Log message emitted when trainer loop exits cleanly. */
export const FLAPPY_TRAINER_STOPPED_MESSAGE =
  'Flappy training stopped gracefully.';

/** Minimum candidate count for reevaluation stage, regardless of elitism. */
export const FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT = 6;

/** Percentile used when reporting median population score. */
export const FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE = 0.5;

/** Percentile used when reporting high-end population score (P90). */
export const FLAPPY_TRAINER_SCORE_P90_PERCENTILE = 0.9;

/** Generation count used to fully anneal mutation schedule from start to end values. */
export const FLAPPY_TRAINER_MUTATION_ANNEAL_GENERATIONS = 120;

/** Initial mutation rate at generation 0 before annealing. */
export const FLAPPY_TRAINER_MUTATION_RATE_START = 0.7;

/** Final mutation rate reached after annealing window completes. */
export const FLAPPY_TRAINER_MUTATION_RATE_END = 0.25;

/** Initial mutation amount at generation 0 before annealing. */
export const FLAPPY_TRAINER_MUTATION_AMOUNT_START = 2;

/** Final mutation amount reached after annealing window completes. */
export const FLAPPY_TRAINER_MUTATION_AMOUNT_END = 1;

/** Initial NEAT mutation rate before generation schedule annealing is applied. */
export const FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE = 0.75;

/** Initial NEAT mutation amount before generation schedule annealing is applied. */
export const FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT = 2;

/** Frame cap used during quick screening rollout stage. */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_MAX_FRAMES = 1_500;

/** Early-termination grace frames used during quick screening rollout stage. */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES = 120;

/** Consecutive unrecoverable frames needed to stop quick screening rollout early. */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES = 18;

/** Pipe-progress target used to normalize quick screening rollout fitness. */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_PIPE_PROGRESS_TARGET = 12;

/** Early-termination grace frames used during full rollout stage. */
export const FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES = 220;

/** Consecutive unrecoverable frames needed to stop full rollout early. */
export const FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES = 28;

/** Pipe-progress target used to normalize full and reevaluation rollout fitness. */
export const FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET = 20;

/** Penalty multiplier applied to fitness standard deviation in frame-primary scoring. */
export const FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT = 0.5;

/** Allowed mean-pipes delta from the current best before frame-primary scoring applies. */
export const FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE = 0.05;

/** Base offset awarded to genomes that satisfy the mean-pipe progress filter. */
export const FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE = 1_000_000;

/** Survival contribution weight for frame-primary scoring. */
export const FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT = 100;

/** Pipe-progress contribution weight for frame-primary scoring. */
export const FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT = 10;

/** Pipe-progress contribution weight for fallback scoring path. */
export const FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT = 10_000;

/** Multiplier over elitism used to size full-pass candidate pool. */
export const FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER = 3;

/** Population fraction used to size full-pass candidate pool. */
export const FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION = 0.3;

/** ID used by dummy fallback network for defensive reporting paths. */
export const FLAPPY_TRAINER_DUMMY_NETWORK_ID = 0;

/** Dummy output channel index for "no flap" action score. */
export const FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT = 1;

/** Dummy output channel index for "flap" action score. */
export const FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT = 0;

/** Delimiter used when composing compact generation log lines. */
export const FLAPPY_TRAINER_LOG_PARTS_DELIMITER = ' ';
