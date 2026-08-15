/**
 * Type definitions for the Neatenstein procedural Web Audio engine.
 *
 * Extracted from `audio.ts` so that the audio engine module contains only
 * synthesis logic while type contracts live in a dedicated, importable file.
 *
 * @module
 */

import type { NEATENSTEIN_AUDIO_SOUND_NAMES } from './constants';

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
export interface NeatensteinCueParams {
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