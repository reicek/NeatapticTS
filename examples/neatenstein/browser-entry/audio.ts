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

export { NEATENSTEIN_AUDIO_SOUND_NAMES };

/** Names of all sounds the engine knows how to synthesize. */
export type NeatensteinSoundName =
  (typeof NEATENSTEIN_AUDIO_SOUND_NAMES)[number];

/** Options for spatializing a played sound. */
export interface NeatensteinPlaySoundOptions {
  /** Angle of the sound source relative to the listener, in radians. */
  angleRad: number;
  /** Distance from the listener to the sound source. */
  distance: number;
}

/** One-shot WebAudio voice used by a single cue. */
export interface NeatensteinAudioVoice {
  /** Stop the voice and clean up its graph nodes. */
  stop(): void;
}

/** Public surface of the Neatenstein audio engine. */
export interface NeatensteinAudioEngine {
  /** Resume the underlying audio context (usually after a user gesture). */
  resume(): Promise<void>;
  /**
   * Play a named procedural sound.
   *
   * @param name - One of the supported sound names.
   * @param options - Optional spatialization parameters.
   * @returns A voice handle that can be stopped early.
   * @throws Error when `name` is not a supported sound.
   */
  playSound(
    name: NeatensteinSoundName,
    options?: NeatensteinPlaySoundOptions,
  ): NeatensteinAudioVoice;
}

/** Cue-specific synthesis parameters. */
interface NeatensteinCueParams {
  /** Oscillator type for the attack body. */
  type: OscillatorType;
  /** Base frequency in hertz. */
  frequencyHz: number;
  /** Frequency sweep end in hertz. */
  frequencyEndHz: number;
  /** Gain envelope peak (0..1). */
  peakGain: number;
  /** Total voice duration in seconds. */
  durationSec: number;
  /** Optional biquad filter type. */
  filterType?: BiquadFilterType;
  /** Optional biquad filter frequency in hertz. */
  filterFrequencyHz?: number;
}

/** Mapping from each supported sound name to its synthesis recipe. */
const NEATENSTEIN_CUE_PARAMS: Record<
  NeatensteinSoundName,
  NeatensteinCueParams
> = {
  fire: {
    type: 'sawtooth',
    frequencyHz: 880,
    frequencyEndHz: 220,
    peakGain: 0.25,
    durationSec: 0.08,
    filterType: 'lowpass',
    filterFrequencyHz: 1_200,
  },
  'enemy-hit': {
    type: 'square',
    frequencyHz: 330,
    frequencyEndHz: 110,
    peakGain: 0.35,
    durationSec: 0.12,
    filterType: 'lowpass',
    filterFrequencyHz: 800,
  },
  'player-damage': {
    type: 'sawtooth',
    frequencyHz: 180,
    frequencyEndHz: 90,
    peakGain: 0.4,
    durationSec: 0.18,
    filterType: 'lowpass',
    filterFrequencyHz: 600,
  },
  dash: {
    type: 'sine',
    frequencyHz: 440,
    frequencyEndHz: 880,
    peakGain: 0.3,
    durationSec: 0.1,
  },
  kill: {
    type: 'square',
    frequencyHz: 660,
    frequencyEndHz: 1320,
    peakGain: 0.35,
    durationSec: 0.14,
    filterType: 'highpass',
    filterFrequencyHz: 400,
  },
  'generation-up': {
    type: 'sine',
    frequencyHz: 523.25,
    frequencyEndHz: 1_046.5,
    peakGain: 0.45,
    durationSec: 0.35,
  },
};

/** Default pan value for a non-positional cue. */
const DEFAULT_PAN = 0;

/** Default gain value for a non-positional cue. */
const DEFAULT_GAIN = 1;

/** Attenuation factor applied per unit distance. */
const DISTANCE_ATTENUATION_PER_UNIT = 0.1;

/**
 * Clamp a number to the inclusive range [min, max].
 *
 * @param value - Value to clamp.
 * @param min - Lower bound.
 * @param max - Upper bound.
 * @returns Clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

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
  if (options === undefined) return DEFAULT_PAN;
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
  if (options === undefined) return DEFAULT_GAIN;
  return 1 / (1 + options.distance * DISTANCE_ATTENUATION_PER_UNIT);
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
    Math.max(20, cue.frequencyEndHz),
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
  gain.gain.exponentialRampToValueAtTime(0.001, end);

  source.connect(panner);
  panner.connect(gain);
  gain.connect(ctx.destination);

  osc.start(now);
  osc.stop(end);

  return {
    stop() {
      const stopTime = ctx.currentTime + 0.01;
      try {
        gain.gain.cancelScheduledValues(ctx.currentTime);
        gain.gain.setValueAtTime(gain.gain.value, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, stopTime);
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
