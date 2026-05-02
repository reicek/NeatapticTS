/**
 * Named knobs for trainer cadence, fairness, fallback behavior, and terminal
 * reporting.
 *
 * This file exists so the trainer reads like policy instead of a wall of magic
 * numbers. When you want to tune how quickly evolution cools down, how much of
 * the population survives into deeper stages, or how progress is summarized in
 * logs, this is the first place to inspect.
 *
 * Use the grouped reference below alongside the alphabetical symbol list that
 * follows in the generated README.
 *
 * Available trainer constant groups:
 *
 * | Group | What it controls | Representative constants |
 * | --- | --- | --- |
 * | Population shape | Demo-scale NEAT size and elitism | `FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE`, `FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT` |
 * | Mutation cooling | How aggressive structural search stays over time | `FLAPPY_TRAINER_MUTATION_RATE_START`, `FLAPPY_TRAINER_MUTATION_RATE_END` |
 * | Stage budgets | How much rollout work each evaluation phase spends | `FLAPPY_TRAINER_QUICK_ROLLOUT_MAX_FRAMES`, `FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET` |
 * | Ranking heuristics | How provisional and robust scores are composed | `FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE`, `FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT` |
 * | Reporting and fallback | Log formatting and defensive dummy-network paths | `FLAPPY_TRAINER_LOG_PARTS_DELIMITER`, `FLAPPY_TRAINER_DUMMY_NETWORK_ID` |
 *
 * Population and reproducibility:
 *
 * | Constant | Why it matters |
 * | --- | --- |
 * | `FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE` | Sets the demo-scale population size for each evolutionary round. |
 * | `FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT` | Preserves a stable top slice of genomes between generations. |
 * | `FLAPPY_TRAINER_DEFAULT_RNG_SEED` | Makes local runs reproducible across tuning sessions. |
 *
 * Mutation cooling:
 *
 * | Constant | Why it matters |
 * | --- | --- |
 * | `FLAPPY_TRAINER_MUTATION_ANNEAL_GENERATIONS` | Defines how long the trainer keeps cooling mutation pressure. |
 * | `FLAPPY_TRAINER_MUTATION_RATE_START` | Starting probability of mutation while search is still broad. |
 * | `FLAPPY_TRAINER_MUTATION_RATE_END` | Late-stage mutation probability after the trainer settles down. |
 * | `FLAPPY_TRAINER_MUTATION_AMOUNT_START` | Starting mutation count budget for exploratory generations. |
 * | `FLAPPY_TRAINER_MUTATION_AMOUNT_END` | Smaller late-stage mutation count for refinement. |
 * | `FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE` | Bootstrap controller mutation rate before schedule updates take over. |
 * | `FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT` | Bootstrap controller mutation amount before schedule updates take over. |
 *
 * Stage budgets and selection depth:
 *
 * | Constant | Why it matters |
 * | --- | --- |
 * | `FLAPPY_TRAINER_QUICK_ROLLOUT_MAX_FRAMES` | Caps the cheap first-pass screen. |
 * | `FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES` | Delays quick-stage early termination until a short grace window passes. |
 * | `FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES` | Requires a streak of bad frames before a quick-stage rollout is cut short. |
 * | `FLAPPY_TRAINER_QUICK_ROLLOUT_PIPE_PROGRESS_TARGET` | Normalizes quick-stage progress against a modest target. |
 * | `FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES` | Gives stronger candidates more recovery time in the deeper stage. |
 * | `FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES` | Uses a stricter streak threshold before full-stage early termination. |
 * | `FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET` | Normalizes deeper-stage progress against a tougher target. |
 * | `FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER` | Sizes the full-pass candidate pool relative to elitism. |
 * | `FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION` | Sizes the full-pass candidate pool relative to total population. |
 * | `FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT` | Guarantees that anti-luck reevaluation still compares a useful set of genomes. |
 *
 * Ranking, robustness, and fallback behavior:
 *
 * | Constant | Why it matters |
 * | --- | --- |
 * | `FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT` | Penalizes unstable shared-seed performance. |
 * | `FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE` | Gates frame-primary scoring to genomes close enough in pipe progress. |
 * | `FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE` | Creates a large score offset once a genome passes the gate. |
 * | `FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT` | Rewards longer stable survival inside the gated branch. |
 * | `FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT` | Keeps pipe progress visible inside the gated branch. |
 * | `FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT` | Emphasizes pipe progress when the primary gate is not met. |
 * | `FLAPPY_TRAINER_DUMMY_NETWORK_ID` | Provides a stable fallback network identifier for defensive reporting paths. |
 * | `FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT` | Encodes the dummy network's preferred passive action score. |
 * | `FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT` | Encodes the dummy network's lower flap score for deterministic fallback behavior. |
 *
 * Reporting and terminal output:
 *
 * | Constant | Why it matters |
 * | --- | --- |
 * | `FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE` | Names the percentile used for the reported median. |
 * | `FLAPPY_TRAINER_SCORE_P90_PERCENTILE` | Names the percentile used for the reported upper-tail score. |
 * | `FLAPPY_TRAINER_LOG_PARTS_DELIMITER` | Keeps compact generation logs consistently tokenized. |
 * | `FLAPPY_TRAINER_STOPPED_MESSAGE` | Gives graceful shutdown a stable terminal message. |
 */
import { FLAPPY_DEFAULT_RNG_SEED } from '../constants/constants';

/**
 * Default population size used by the Flappy trainer NEAT run.
 *
 * The demo keeps this large enough for staged selection to matter while still
 * remaining practical for local experimentation.
 */
export const FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE = 200;

/**
 * Number of elite genomes preserved unchanged each generation.
 *
 * Preserving a small elite keeps the trainer from discarding clearly strong
 * genomes while the rest of the population continues exploring.
 */
export const FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT = 20;

/**
 * Deterministic trainer RNG seed used for reproducible training runs.
 *
 * Reusing the shared Flappy example seed makes trainer behavior easier to
 * compare across doc examples, tests, and manual tuning sessions.
 */
export const FLAPPY_TRAINER_DEFAULT_RNG_SEED = FLAPPY_DEFAULT_RNG_SEED;

/**
 * Log message emitted when trainer loop exits cleanly.
 *
 * A dedicated constant keeps the shutdown path stable for humans and for any
 * scripts that watch trainer output.
 */
export const FLAPPY_TRAINER_STOPPED_MESSAGE =
  'Flappy training stopped gracefully.';

/**
 * Minimum candidate count for reevaluation stage, regardless of elitism.
 *
 * This prevents small elite settings from starving the anti-luck pass of enough
 * genomes to produce a meaningful final comparison.
 */
export const FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT = 6;

/**
 * Percentile used when reporting median population score.
 *
 * Keeping the percentile explicit makes the report math self-documenting even
 * for readers who skim the log formatter before the statistics helpers.
 */
export const FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE = 0.5;

/**
 * Percentile used when reporting high-end population score (P90).
 *
 * The trainer uses `p90` as a quick "is the upper tail getting healthier?"
 * signal without over-focusing on only the single best genome.
 */
export const FLAPPY_TRAINER_SCORE_P90_PERCENTILE = 0.9;

/**
 * Generation count used to fully anneal mutation schedule from start to end values.
 *
 * Within this window the trainer gradually cools from more exploratory updates
 * toward smaller, steadier changes.
 */
export const FLAPPY_TRAINER_MUTATION_ANNEAL_GENERATIONS = 120;

/**
 * Initial mutation rate at generation `0` before annealing.
 *
 * The starting rate is intentionally aggressive so the early population can
 * discover useful topologies quickly.
 */
export const FLAPPY_TRAINER_MUTATION_RATE_START = 0.7;

/**
 * Final mutation rate reached after annealing window completes.
 *
 * Lower late-stage mutation pressure helps good policies stabilize instead of
 * being reshuffled as aggressively as the opening generations.
 */
export const FLAPPY_TRAINER_MUTATION_RATE_END = 0.25;

/**
 * Initial mutation amount at generation `0` before annealing.
 *
 * This controls how many mutation operations can be applied while the trainer
 * is still in its exploratory phase.
 */
export const FLAPPY_TRAINER_MUTATION_AMOUNT_START = 2;

/**
 * Final mutation amount reached after annealing window completes.
 *
 * Cooling the mutation amount along with the rate reduces late-generation noise
 * without fully freezing structural search.
 */
export const FLAPPY_TRAINER_MUTATION_AMOUNT_END = 1;

/**
 * Initial NEAT mutation rate before generation schedule annealing is applied.
 *
 * This seeds the controller with a sensible baseline before the per-generation
 * planner starts taking over.
 */
export const FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE = 0.75;

/**
 * Initial NEAT mutation amount before generation schedule annealing is applied.
 *
 * Matching the controller bootstrap to the trainer policy avoids a confusing
 * mismatch between generation `0` and later loop behavior.
 */
export const FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT = 2;

/**
 * Frame cap used during quick screening rollout stage.
 *
 * The quick stage is supposed to eliminate obviously weak genomes cheaply, so
 * its horizon is intentionally shorter than the full evaluation horizon.
 */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_MAX_FRAMES = 1_500;

/**
 * Early-termination grace frames used during quick screening rollout stage.
 *
 * This short grace period gives a policy a brief chance to stabilize before the
 * unrecoverable-flight heuristic is allowed to stop the rollout.
 */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES = 120;

/**
 * Consecutive unrecoverable frames needed to stop quick screening rollout early.
 *
 * Requiring a streak prevents one noisy frame from ending the screen too early
 * while still saving time on clearly doomed trajectories.
 */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES = 18;

/**
 * Pipe-progress target used to normalize quick screening rollout fitness.
 *
 * The lower quick-stage target reflects the fact that this pass is a screen, not
 * the trainer's final statement of policy quality.
 */
export const FLAPPY_TRAINER_QUICK_ROLLOUT_PIPE_PROGRESS_TARGET = 12;

/**
 * Early-termination grace frames used during full rollout stage.
 *
 * The deeper stage allows more time before judging a trajectory unrecoverable
 * because the trainer is now evaluating stronger candidates more carefully.
 */
export const FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES = 220;

/**
 * Consecutive unrecoverable frames needed to stop full rollout early.
 *
 * This longer streak makes the full stage less trigger-happy than the quick
 * screen while still avoiding wasted rollout budget.
 */
export const FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES = 28;

/**
 * Pipe-progress target used to normalize full and reevaluation rollout fitness.
 *
 * The deeper stages share a tougher target because they are used for robust
 * ranking rather than first-pass elimination.
 */
export const FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET = 20;

/**
 * Penalty multiplier applied to fitness standard deviation in frame-primary scoring.
 *
 * Higher instability lowers the provisional score so a lucky but erratic genome
 * is less likely to outrank a steadier competitor.
 */
export const FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT = 0.5;

/**
 * Allowed mean-pipes delta from the current best before frame-primary scoring applies.
 *
 * This acts like a gating tolerance: only genomes close enough in pipe progress
 * get the more generous frame-primary score treatment.
 */
export const FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE = 0.05;

/**
 * Base offset awarded to genomes that satisfy the mean-pipe progress filter.
 *
 * The large offset makes it obvious that surviving the gate is more important
 * than tiny differences in the secondary frame-oriented terms.
 */
export const FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE = 1_000_000;

/**
 * Survival contribution weight for frame-primary scoring.
 *
 * Once a genome passes the pipe-progress gate, extra survival time still matters
 * because it often signals more stable control.
 */
export const FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT = 100;

/**
 * Pipe-progress contribution weight for frame-primary scoring.
 *
 * This keeps pipe progress visible even inside the gated scoring branch so the
 * ranking still prefers genuinely advancing policies.
 */
export const FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT = 10;

/**
 * Pipe-progress contribution weight for fallback scoring path.
 *
 * The fallback path leans heavily on pipe progress because it is the clearest
 * robust signal available before the primary gate is satisfied.
 */
export const FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT = 10_000;

/**
 * Multiplier over elitism used to size full-pass candidate pool.
 *
 * This ties the full-pass budget to a familiar population concept so the deeper
 * stage scales alongside the preserved elite.
 */
export const FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER = 3;

/**
 * Population fraction used to size full-pass candidate pool.
 *
 * The full stage uses the larger of this fraction and the elitism-based floor so
 * promising mid-pack genomes are not excluded too aggressively.
 */
export const FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION = 0.3;

/**
 * ID used by dummy fallback network for defensive reporting paths.
 *
 * The report helpers occasionally need a safe stand-in network so logging can
 * stay total even when no real population data is available.
 */
export const FLAPPY_TRAINER_DUMMY_NETWORK_ID = 0;

/**
 * Dummy output channel value for the "no flap" action score.
 *
 * The dummy network intentionally prefers the passive action so fallback report
 * generation remains deterministic and simple.
 */
export const FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT = 1;

/**
 * Dummy output channel value for the "flap" action score.
 *
 * Keeping the flap score lower than the no-flap score produces a predictable
 * never-flap dummy network for defensive report code paths.
 */
export const FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT = 0;

/**
 * Delimiter used when composing compact generation log lines.
 *
 * A single-space delimiter keeps the log dense, stable, and easy to parse by eye
 * during long-running terminal sessions.
 */
export const FLAPPY_TRAINER_LOG_PARTS_DELIMITER = ' ';

/**
 * Extended early-termination grace frames for recurrent profiles in quick stage.
 *
 * Recurrent architectures (NARX, GRU, LSTM) require more time to populate their
 * hidden state before exhibiting coherent flight behavior. The standard 120-frame
 * window terminates them before that warm-up completes. This extended window
 * matches the longer effective response latency of stateful networks.
 */
export const FLAPPY_TRAINER_RECURRENT_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES = 300;

/**
 * Extended early-termination grace frames for recurrent profiles in full stage.
 *
 * Full-stage evaluation uses a stricter but still profile-aware grace window so
 * stateful networks are not penalized for the additional hidden-state warm-up
 * cost they incur relative to feed-forward policies.
 */
export const FLAPPY_TRAINER_RECURRENT_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES = 480;
