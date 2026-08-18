/**
 * Procedural Web Audio engine for the Neatenstein neon raycasting demo.
 *
 * The engine is intentionally lazy: no `AudioContext` is created while the
 * module is being parsed in Node, and the context is only instantiated when
 * the caller first resumes or plays a sound. This makes the module safe to
 * import in non-browser contexts such as Jest.
 *
 * Six short synthesized cues are exposed (`fire`, `enemy-hit`, `player-damage`,
 * `dash`, `kill`, `generation-up`). Each cue is built from oscillators, an
 * optional filter, a stereo panner, and a gain envelope so the same engine can
 * spatialize sounds by angle and distance.
 *
 * @module
 */

import { NEATENSTEIN_AUDIO_SOUND_NAMES } from './constants';
import {
  OSC_TYPE_SAWTOOTH,
  OSC_TYPE_SQUARE,
  OSC_TYPE_SINE,
  FILTER_TYPE_LOWPASS,
  FILTER_TYPE_HIGHPASS,
  FIRE_FREQ_HZ,
  FIRE_FREQ_END_HZ,
  FIRE_PEAK_GAIN,
  FIRE_DURATION_SEC,
  FIRE_FILTER_FREQ_HZ,
  ENEMY_HIT_FREQ_HZ,
  ENEMY_HIT_FREQ_END_HZ,
  ENEMY_HIT_PEAK_GAIN,
  ENEMY_HIT_DURATION_SEC,
  ENEMY_HIT_FILTER_FREQ_HZ,
  PLAYER_DAMAGE_FREQ_HZ,
  PLAYER_DAMAGE_FREQ_END_HZ,
  PLAYER_DAMAGE_PEAK_GAIN,
  PLAYER_DAMAGE_DURATION_SEC,
  PLAYER_DAMAGE_FILTER_FREQ_HZ,
  DASH_FREQ_HZ,
  DASH_FREQ_END_HZ,
  DASH_PEAK_GAIN,
  DASH_DURATION_SEC,
  KILL_FREQ_HZ,
  KILL_FREQ_END_HZ,
  KILL_PEAK_GAIN,
  KILL_DURATION_SEC,
  KILL_FILTER_FREQ_HZ,
  GENERATION_UP_FREQ_HZ,
  GENERATION_UP_FREQ_END_HZ,
  GENERATION_UP_PEAK_GAIN,
  GENERATION_UP_DURATION_SEC,
  MIN_OSC_FREQ,
  GAIN_RAMP_FLOOR,
  VOICE_STOP_DELAY_SEC,
  DEFAULT_PAN,
  DEFAULT_GAIN,
  DISTANCE_ATTENUATION_PER_UNIT,
} from './audio.constants';
import type {
  NeatensteinSoundName,
  NeatensteinPlaySoundOptions,
  NeatensteinAudioVoice,
  NeatensteinAudioEngine,
  NeatensteinCueParams,
} from './audio.types';
import { clamp } from './shared/math-guards.utils';

export { NEATENSTEIN_AUDIO_SOUND_NAMES };
export type {
  NeatensteinSoundName,
  NeatensteinPlaySoundOptions,
  NeatensteinAudioVoice,
  NeatensteinAudioEngine,
  NeatensteinCueParams,
} from './audio.types';

/** Mapping from each supported sound name to its synthesis recipe. */
const NEATENSTEIN_CUE_PARAMS: Record<
  NeatensteinSoundName,
  NeatensteinCueParams
> = {
  fire: {
    type: OSC_TYPE_SAWTOOTH,
    frequencyHz: FIRE_FREQ_HZ,
    frequencyEndHz: FIRE_FREQ_END_HZ,
    peakGain: FIRE_PEAK_GAIN,
    durationSec: FIRE_DURATION_SEC,
    filterType: FILTER_TYPE_LOWPASS,
    filterFrequencyHz: FIRE_FILTER_FREQ_HZ,
  },
  'enemy-hit': {
    type: OSC_TYPE_SQUARE,
    frequencyHz: ENEMY_HIT_FREQ_HZ,
    frequencyEndHz: ENEMY_HIT_FREQ_END_HZ,
    peakGain: ENEMY_HIT_PEAK_GAIN,
    durationSec: ENEMY_HIT_DURATION_SEC,
    filterType: FILTER_TYPE_LOWPASS,
    filterFrequencyHz: ENEMY_HIT_FILTER_FREQ_HZ,
  },
  'player-damage': {
    type: OSC_TYPE_SAWTOOTH,
    frequencyHz: PLAYER_DAMAGE_FREQ_HZ,
    frequencyEndHz: PLAYER_DAMAGE_FREQ_END_HZ,
    peakGain: PLAYER_DAMAGE_PEAK_GAIN,
    durationSec: PLAYER_DAMAGE_DURATION_SEC,
    filterType: FILTER_TYPE_LOWPASS,
    filterFrequencyHz: PLAYER_DAMAGE_FILTER_FREQ_HZ,
  },
  dash: {
    type: OSC_TYPE_SINE,
    frequencyHz: DASH_FREQ_HZ,
    frequencyEndHz: DASH_FREQ_END_HZ,
    peakGain: DASH_PEAK_GAIN,
    durationSec: DASH_DURATION_SEC,
  },
  kill: {
    type: OSC_TYPE_SQUARE,
    frequencyHz: KILL_FREQ_HZ,
    frequencyEndHz: KILL_FREQ_END_HZ,
    peakGain: KILL_PEAK_GAIN,
    durationSec: KILL_DURATION_SEC,
    filterType: FILTER_TYPE_HIGHPASS,
    filterFrequencyHz: KILL_FILTER_FREQ_HZ,
  },
  'generation-up': {
    type: OSC_TYPE_SINE,
    frequencyHz: GENERATION_UP_FREQ_HZ,
    frequencyEndHz: GENERATION_UP_FREQ_END_HZ,
    peakGain: GENERATION_UP_PEAK_GAIN,
    durationSec: GENERATION_UP_DURATION_SEC,
  },
};

/** Default pan value for a non-positional cue. */
const DEFAULT_PAN_CUE = DEFAULT_PAN;

/** Default gain value for a non-positional cue. */
const DEFAULT_GAIN_CUE = DEFAULT_GAIN;

/** Attenuation factor applied per unit distance. */
const DISTANCE_ATTENUATION = DISTANCE_ATTENUATION_PER_UNIT;

/**
 * Convert positional options into a stereo pan value.
 *
 * Pan varies smoothly from -1 (full left) through 0 (center) to 1 (full right)
 * as the source angle sweeps around the listener.
 *
 * @param options - Optional spatialization parameters.
 * @returns Pan value in the range [-1, 1].
 */
function resolvePan(options?: NeatensteinPlaySoundOptions): number {
  if (options === undefined) return DEFAULT_PAN_CUE;
  return clamp(Math.sin(options.angleRad), -1, 1);
}

/**
 * Convert positional options into a distance-attenuated gain value.
 *
 * Gain falls off as `1 / (1 + distance * attenuation)` so nearby sources stay
 * loud while far sources fade out.
 *
 * @param options - Optional spatialization parameters.
 * @returns Attenuated gain in the range (0, 1].
 */
function resolveGain(options?: NeatensteinPlaySoundOptions): number {
  if (options === undefined) return DEFAULT_GAIN_CUE;
  return 1 / (1 + options.distance * DISTANCE_ATTENUATION);
}

/**
 * Build a one-shot voice for a cue, optionally spatialized.
 *
 * This function assumes the audio context is available and creates the full
 * node graph: oscillator → (optional filter) → panner → gain → destination.
 *
 * @param ctx - Active audio context.
 * @param cue - Synthesis parameters for the cue.
 * @param options - Optional spatialization parameters.
 * @returns A voice handle that can be stopped early.
 */
function buildVoice(
  ctx: AudioContext,
  cue: NeatensteinCueParams,
  options?: NeatensteinPlaySoundOptions,
): NeatensteinAudioVoice {
  const now = ctx.currentTime;
  const end = now + cue.durationSec;

  const osc = ctx.createOscillator();
  osc.type = cue.type;
  osc.frequency.setValueAtTime(cue.frequencyHz, now);
  osc.frequency.exponentialRampToValueAtTime(
    Math.max(MIN_OSC_FREQ, cue.frequencyEndHz),
    end,
  );

  let source: AudioNode = osc;

  if (cue.filterType !== undefined && cue.filterFrequencyHz !== undefined) {
    const filter = ctx.createBiquadFilter();
    filter.type = cue.filterType;
    filter.frequency.setValueAtTime(cue.filterFrequencyHz, now);
    osc.connect(filter);
    source = filter;
  }

  const panner = ctx.createStereoPanner();
  panner.pan.value = resolvePan(options);

  const gain = ctx.createGain();
  gain.gain.value = resolveGain(options) * cue.peakGain;
  gain.gain.setValueAtTime(gain.gain.value, now);
  gain.gain.exponentialRampToValueAtTime(GAIN_RAMP_FLOOR, end);

  source.connect(panner);
  panner.connect(gain);
  gain.connect(ctx.destination);

  osc.start(now);
  osc.stop(end);

  return {
    stop() {
      const stopTime = ctx.currentTime + VOICE_STOP_DELAY_SEC;
      try {
        gain.gain.cancelScheduledValues(ctx.currentTime);
        gain.gain.setValueAtTime(gain.gain.value, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(GAIN_RAMP_FLOOR, stopTime);
        osc.stop(stopTime);
      } catch {
        // Voice may already be stopped by the engine; ignore cleanup errors.
      }
    },
  };
}

/**
 * Create the lazy Web Audio engine used by the demo.
 *
 * The returned engine does not instantiate `AudioContext` at module load; the
 * context is created on the first call to {@link NeatensteinAudioEngine.resume}
 * or {@link NeatensteinAudioEngine.playSound}.
 *
 * @returns A new audio engine instance.
 *
 * @example
 * ```ts
 * const audio = createNeatensteinAudioEngine();
 * await audio.resume();
 * audio.playSound('fire', { angleRad: 0.5, distance: 4 });
 * ```
 */
export function createNeatensteinAudioEngine(): NeatensteinAudioEngine {
  let ctx: AudioContext | null = null;

  function ensureContext(): AudioContext {
    if (ctx === null) {
      ctx = new AudioContext();
    }
    return ctx;
  }

  return {
    async resume() {
      const context = ensureContext();
      await context.resume();
    },
    playSound(name, options) {
      if (!NEATENSTEIN_AUDIO_SOUND_NAMES.includes(name)) {
        throw new Error(`Unknown Neatenstein sound: ${name}`);
      }
      const cue = NEATENSTEIN_CUE_PARAMS[name];
      const context = ensureContext();
      return buildVoice(context, cue, options);
    },
  };
}

/**
 * Schedule the generation-up audio cue for a generation-up event.
 *
 * The function mirrors {@link emitNeatensteinGenerationUpPulse} from the pulse
 * module so the audio and visual events fire on the same simulation tick.
 *
 * @param simTick - Current fixed-timestep simulation tick.
 * @returns `true` when the generation-up sound is scheduled for this tick.
 *
 * @example
 * ```ts
 * if (scheduleNeatensteinGenerationUpAudio(tick)) {
 *   audio.playSound('generation-up');
 * }
 * ```
 */
export function scheduleNeatensteinGenerationUpAudio(simTick: number): boolean {
  return simTick >= 0;
}