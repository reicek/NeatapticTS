import { FLAPPY_NORMALIZATION_EPSILON as SHARED_FLAPPY_NORMALIZATION_EPSILON } from './constants.observation';

/**
 * Browser runtime and telemetry constants for the Flappy demo.
 *
 * These values control startup defaults, emulation cadence, and common
 * placeholder/status values rendered by the UI.
 */

/** Default host container id for the browser demo mount point. */
export const DEFAULT_CONTAINER_ID = 'flappy-bird-output';

/** Emulation speed multiplier for browser playback (1.5 => 50% faster). */
export const FLAPPY_EMULATION_SPEED_MULTIPLIER = 1.5;

/** Update HUD counters every N simulation frames to reduce DOM churn. */
export const FLAPPY_HUD_UPDATE_INTERVAL_FRAMES = 10;

/** Default population size for browser playback worker initialization. */
export const FLAPPY_BROWSER_POPULATION_SIZE = 50;

/** Default elitism count for browser playback worker initialization. */
export const FLAPPY_BROWSER_ELITISM_COUNT = 10;

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

/** Shared HUD value when a metric is intentionally disabled. */
export const FLAPPY_HUD_OFF_TEXT = 'off';

/** Initial status text displayed before evolution starts. */
export const FLAPPY_HUD_INITIALIZING_TEXT = 'initializing';

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
