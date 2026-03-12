/// <reference lib="webworker" />

/**
 * Browser-worker entrypoint for the Flappy Bird example.
 *
 * This file is intentionally tiny. Its job is only to establish the Web Worker
 * bundle boundary and delegate all real protocol, evolution, and playback logic
 * to the dedicated `flappy-evolution-worker/` folder.
 */
import './flappy-evolution-worker/flappy-evolution-worker';
