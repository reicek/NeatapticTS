/// <reference lib="webworker" />

/**
 * Dedicated browser-worker entrypoint for Astro Bird `InferenceChannel` sessions.
 *
 * This bundle exists so the Flappy evolution worker can open persistent channel
 * predictors without needing a second docs-pipeline asset graph or a demo-local
 * inline-worker workaround.
 */
import '../../src/architecture/network/worker-payload/network.worker-payload.channel.worker';
