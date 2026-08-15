/**
 * Procedural Web Audio synthesis constants for the Neatenstein demo.
 *
 * Extracts oscillator types, biquad filter types, per-cue synthesis
 * parameters, and audio-engine utility constants so that `audio.ts` contains
 * only the engine logic with no inline magic numbers.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Oscillator types
// ---------------------------------------------------------------------------

/** Oscillator type for sawtooth-wave cues (fire, player-damage). */
export const OSC_TYPE_SAWTOOTH: OscillatorType = 'sawtooth';

/** Oscillator type for square-wave cues (enemy-hit, kill). */
export const OSC_TYPE_SQUARE: OscillatorType = 'square';

/** Oscillator type for sine-wave cues (dash, generation-up). */
export const OSC_TYPE_SINE: OscillatorType = 'sine';

// ---------------------------------------------------------------------------
// Biquad filter types
// ---------------------------------------------------------------------------

/** Biquad filter type for low-pass filtering (fire, enemy-hit, player-damage). */
export const FILTER_TYPE_LOWPASS: BiquadFilterType = 'lowpass';

/** Biquad filter type for high-pass filtering (kill). */
export const FILTER_TYPE_HIGHPASS: BiquadFilterType = 'highpass';

// ---------------------------------------------------------------------------
// Fire cue parameters
// ---------------------------------------------------------------------------

/** Base frequency of the fire cue in hertz. */
export const FIRE_FREQ_HZ = 880;

/** Frequency sweep end of the fire cue in hertz. */
export const FIRE_FREQ_END_HZ = 220;

/** Peak gain of the fire cue envelope (0..1). */
export const FIRE_PEAK_GAIN = 0.25;

/** Total duration of the fire cue in seconds. */
export const FIRE_DURATION_SEC = 0.08;

/** Biquad filter cutoff frequency for the fire cue in hertz. */
export const FIRE_FILTER_FREQ_HZ = 1_200;

// ---------------------------------------------------------------------------
// Enemy-hit cue parameters
// ---------------------------------------------------------------------------

/** Base frequency of the enemy-hit cue in hertz. */
export const ENEMY_HIT_FREQ_HZ = 330;

/** Frequency sweep end of the enemy-hit cue in hertz. */
export const ENEMY_HIT_FREQ_END_HZ = 110;

/** Peak gain of the enemy-hit cue envelope (0..1). */
export const ENEMY_HIT_PEAK_GAIN = 0.35;

/** Total duration of the enemy-hit cue in seconds. */
export const ENEMY_HIT_DURATION_SEC = 0.12;

/** Biquad filter cutoff frequency for the enemy-hit cue in hertz. */
export const ENEMY_HIT_FILTER_FREQ_HZ = 800;

// ---------------------------------------------------------------------------
// Player-damage cue parameters
// ---------------------------------------------------------------------------

/** Base frequency of the player-damage cue in hertz. */
export const PLAYER_DAMAGE_FREQ_HZ = 180;

/** Frequency sweep end of the player-damage cue in hertz. */
export const PLAYER_DAMAGE_FREQ_END_HZ = 90;

/** Peak gain of the player-damage cue envelope (0..1). */
export const PLAYER_DAMAGE_PEAK_GAIN = 0.4;

/** Total duration of the player-damage cue in seconds. */
export const PLAYER_DAMAGE_DURATION_SEC = 0.18;

/** Biquad filter cutoff frequency for the player-damage cue in hertz. */
export const PLAYER_DAMAGE_FILTER_FREQ_HZ = 600;

// ---------------------------------------------------------------------------
// Dash cue parameters
// ---------------------------------------------------------------------------

/** Base frequency of the dash cue in hertz. */
export const DASH_FREQ_HZ = 440;

/** Frequency sweep end of the dash cue in hertz. */
export const DASH_FREQ_END_HZ = 880;

/** Peak gain of the dash cue envelope (0..1). */
export const DASH_PEAK_GAIN = 0.3;

/** Total duration of the dash cue in seconds. */
export const DASH_DURATION_SEC = 0.1;

// ---------------------------------------------------------------------------
// Kill cue parameters
// ---------------------------------------------------------------------------

/** Base frequency of the kill cue in hertz. */
export const KILL_FREQ_HZ = 660;

/** Frequency sweep end of the kill cue in hertz. */
export const KILL_FREQ_END_HZ = 1_320;

/** Peak gain of the kill cue envelope (0..1). */
export const KILL_PEAK_GAIN = 0.35;

/** Total duration of the kill cue in seconds. */
export const KILL_DURATION_SEC = 0.14;

/** Biquad filter cutoff frequency for the kill cue in hertz. */
export const KILL_FILTER_FREQ_HZ = 400;

// ---------------------------------------------------------------------------
// Generation-up cue parameters
// ---------------------------------------------------------------------------

/** Base frequency of the generation-up cue in hertz (C5). */
export const GENERATION_UP_FREQ_HZ = 523.25;

/** Frequency sweep end of the generation-up cue in hertz (C6). */
export const GENERATION_UP_FREQ_END_HZ = 1_046.5;

/** Peak gain of the generation-up cue envelope (0..1). */
export const GENERATION_UP_PEAK_GAIN = 0.45;

/** Total duration of the generation-up cue in seconds. */
export const GENERATION_UP_DURATION_SEC = 0.35;

// ---------------------------------------------------------------------------
// Engine utility constants
// ---------------------------------------------------------------------------

/** Minimum oscillator frequency in hertz, clamped to avoid non-finite values. */
export const MIN_OSC_FREQ = 20;

/** Gain ramp floor value used in exponential ramps (must be > 0). */
export const GAIN_RAMP_FLOOR = 0.001;

/** Delay in seconds before a voice is stopped after cancellation. */
export const VOICE_STOP_DELAY_SEC = 0.01;

/** Default stereo pan value for a non-positional cue (center). */
export const DEFAULT_PAN = 0;

/** Default gain value for a non-positional cue (full volume). */
export const DEFAULT_GAIN = 1;

/** Attenuation factor applied per unit of distance from the listener. */
export const DISTANCE_ATTENUATION_PER_UNIT = 0.1;