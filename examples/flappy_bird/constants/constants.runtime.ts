import { FLAPPY_NORMALIZATION_EPSILON as SHARED_FLAPPY_NORMALIZATION_EPSILON } from './constants.observation';

/**
 * Browser runtime and telemetry constants for the Flappy demo.
 *
 * These values control startup defaults, emulation cadence, and common
 * placeholder/status values rendered by the UI.
 *
 * This module is less about game physics and more about operator experience:
 * what the browser shows while loading, how frequently HUD numbers refresh, and
 * which defaults the demo uses when bootstrapping the worker.
 */

/** Default host container id for the browser demo mount point. */
export const DEFAULT_CONTAINER_ID = 'flappy-bird-output';

/** Emulation speed multiplier for browser playback (1.5 => 50% faster). */
export const FLAPPY_EMULATION_SPEED_MULTIPLIER = 1;

/**
 * Update HUD counters every N simulation frames to reduce DOM churn.
 *
 * The value trades freshness for stability. Updating every frame would make the
 * numbers twitchier and force more frequent DOM work on the main thread.
 */
export const FLAPPY_HUD_UPDATE_INTERVAL_FRAMES = 10;

/**
 * Default population size for browser playback worker initialization.
 *
 * The browser default stays intentionally small so the interactive demo remains
 * responsive and each generation is easy to inspect during playback.
 */
export const FLAPPY_BROWSER_POPULATION_SIZE = 5;

/**
 * Default elitism count for browser playback worker initialization.
 *
 * With a five-bird browser flock, preserving one elite keeps elitism at 20%
 * while still leaving most of the population available for visible variation.
 */
export const FLAPPY_BROWSER_ELITISM_COUNT = 1;

/**
 * Pipe-count milestone that marks a browser run as "good enough" to downshift.
 *
 * Once a generation clears this bar, future browser generations can shrink to a
 * lighter flock without losing the core demonstration that the selected
 * architecture is already solving pipes in the live demo.
 */
export const FLAPPY_BROWSER_SUCCESS_PIPE_TARGET = 10;

/**
 * Reduced browser population size used after a live run has already hit the success bar.
 *
 * The downshift keeps later browser generations cheaper once an architecture is
 * already demonstrating stable pipe-clearing behavior.
 */
export const FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_POPULATION_SIZE = 8;

/**
 * Reduced browser elitism count paired with the post-success population downshift.
 *
 * Two elites keep a small continuity shelf while still leaving most of the
 * reduced flock available for visible variation.
 */
export const FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_ELITISM_COUNT = 2;

/**
 * Deterministic default RNG seed shared by browser runtime and trainer flows.
 *
 * Reusing one canonical seed makes debugging and README examples more
 * repeatable.
 */
export const FLAPPY_DEFAULT_RNG_SEED = 0x1234abcd;

/** Normalized decision threshold used for scalar output flap policies. */
export const FLAPPY_FLAP_THRESHOLD = 0.5;

/** Small epsilon divisor guard for world/physics normalization. */
export const FLAPPY_NORMALIZATION_EPSILON = SHARED_FLAPPY_NORMALIZATION_EPSILON;

/** Canonical half multiplier for centering and gap math. */
export const FLAPPY_HALF = 0.5;

/** Shared HUD value for integer zero fields. */
export const FLAPPY_HUD_ZERO_TEXT = '0';

/** Shared HUD value for decimal zero fields. */
export const FLAPPY_HUD_ZERO_DECIMAL_TEXT = '0.00';

/** Shared HUD value for metrics that are not available yet. */
export const FLAPPY_HUD_PLACEHOLDER_TEXT = '-';

/** Shared HUD value when a metric is intentionally disabled. */
export const FLAPPY_HUD_OFF_TEXT = 'off';

/** Initial status text displayed before evolution starts. */
export const FLAPPY_HUD_INITIALIZING_TEXT = 'initializing';

/** Centered startup legend shown while the first browser generation is initializing. */
export const FLAPPY_STARTUP_PREVIEW_LEGEND_TEXT =
  'PREPARING NEURAL NETWORKS...';

/** Fade duration used for both startup legend fade-in and fade-out (milliseconds). */
export const FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS = 300;

/** Time a generation presentation card stays fully visible before fading out (milliseconds). */
export const FLAPPY_GENERATION_PREVIEW_HOLD_DURATION_MS = 300;

/** Approximate preview frame duration used to drive deterministic background motion. */
export const FLAPPY_STARTUP_PREVIEW_FRAME_DURATION_MS = 1000 / 60;

/** Font weight used by the centered startup loading legend. */
export const FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_WEIGHT = 700;

/** Responsive font-size ratio applied to the smaller startup-preview canvas dimension. */
export const FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_SIZE_RATIO = 0.075;

/** Minimum readable startup-preview legend font size (pixels). */
export const FLAPPY_STARTUP_PREVIEW_LEGEND_MIN_FONT_SIZE_PX = 18;

/** Maximum startup-preview legend font size (pixels). */
export const FLAPPY_STARTUP_PREVIEW_LEGEND_MAX_FONT_SIZE_PX = 40;

/** Runtime status text shown while a generation playback is running. */
export const FLAPPY_STATUS_PLAYING_TEXT = 'playing';

/** Runtime status text shown between playback episodes. */
export const FLAPPY_STATUS_EVOLVING_TEXT = 'evolving';

/** HUD sliding-window size used when computing updates-per-second metric. */
export const FLAPPY_HUD_UPDATES_WINDOW_MS = 10_000;

/** HUD updates window duration in seconds for per-second conversion. */
export const FLAPPY_HUD_UPDATES_WINDOW_SECONDS = 10;

/** Sliding-window size used when computing minor GC events per minute. */
export const FLAPPY_MINOR_GC_WINDOW_MS = 60_000;
