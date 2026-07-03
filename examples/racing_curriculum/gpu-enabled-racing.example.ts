/**
 * GPU-enabled racing controller example.
 *
 * This snippet shows how to attach a WebGPU device to a racing-curriculum
 * controller network and opt into the GPU inference fast path. It uses the
 * same deterministic controller network that the browser harness builds, so the
 * example is a direct mirror of the public inference surface used by the demo.
 *
 * The GPU path is opt-in and transparent:
 *   1. Assign a `GPUDevice` to `network.gpuDevice`.
 *   2. Pass `{ useGPU: true }` to `network.activate`.
 *
 * If the device is missing, the network is ineligible, or `useGPU` is omitted,
 * the call falls back to the CPU path automatically. The CPU path remains the
 * canonical reference for deterministic replay and cross-machine regression
 * tests.
 *
 * Build this file for the browser with the same esbuild pattern used by the
 * racing-curriculum bundle:
 *
 * ```bash
 * npx esbuild examples/racing_curriculum/gpu-enabled-racing.example.ts \
 *   --bundle --outfile=docs/assets/gpu-enabled-racing.example.js \
 *   --platform=browser --format=iife --minify --sourcemap \
 *   --external:fs --external:child_process --external:path
 * ```
 *
 * Then load `docs/assets/gpu-enabled-racing.example.js` from an HTML page that
 * runs in a WebGPU-capable browser.
 *
 * @example
 * ```ts
 * import { createDeterministicRacingControllerNetwork } from './browser-entry/browser-entry';
 *
 * async function runGpuRacingController(): Promise<void> {
 *   const observationTier = 1;
 *
 *   // Build the same MLP the racing browser harness uses.
 *   const network = createDeterministicRacingControllerNetwork(observationTier);
 *
 *   // Request a WebGPU device from the browser.
 *   const adapter = await navigator.gpu?.requestAdapter({
 *     powerPreference: 'high-performance',
 *   });
 *   const device = await adapter?.requestDevice();
 *   network.gpuDevice = device ?? undefined;
 *
 *   // Build a normalized observation vector for the tier-1 controller.
 *   // The exact width is determined by the observation assembler; here we use
 *   // a placeholder vector of the right length.
 *   const observationWidth = network.input;
 *   const observation = new Array(observationWidth).fill(0).map((_, i) =>
 *     Math.sin(i * 0.5),
 *   );
 *
 *   // Opt into the GPU path. The call returns a Promise<Float32Array>.
 *   const output = await network.activate(observation, { useGPU: true });
 *   console.log('GPU-enabled controller output:', Array.from(output));
 * }
 *
 * runGpuRacingController().catch(console.error);
 * ```
 */

import { createDeterministicRacingControllerNetwork } from './browser-entry/browser-entry';

/** Number of controller outputs for a tier-1 racing network. */
const TIER_1_OUTPUT_COUNT = 2;

/** Tier-1 observation width for the deterministic racing controller. */
const TIER_1_OBSERVATION_TIER = 1 as const;

/**
 * Run a single GPU-enabled controller activation using the racing-curriculum
 * deterministic network.
 *
 * This function is a runnable demonstration of the opt-in contract:
 * `network.gpuDevice` plus `network.activate(input, { useGPU: true })`.
 *
 * @returns The controller outputs as a plain number array.
 */
export async function runGpuRacingController(): Promise<number[]> {
  const observationTier = TIER_1_OBSERVATION_TIER;

  // Build the same MLP the racing browser harness uses.
  const network = createDeterministicRacingControllerNetwork(observationTier);

  // Request a WebGPU device from the browser. In Node or in browsers without
  // WebGPU, `navigator.gpu` is undefined and the example falls back cleanly.
  const adapter =
    typeof navigator !== 'undefined'
      ? await navigator.gpu?.requestAdapter({
          powerPreference: 'high-performance',
        })
      : undefined;
  const device = await adapter?.requestDevice();
  network.gpuDevice = device ?? undefined;

  // Build a normalized observation vector of the right width for the network.
  const observationWidth = network.input;
  const observation = new Array(observationWidth)
    .fill(0)
    .map((_, index) => Math.sin(index * 0.5));

  // Opt into the GPU path. The call returns a Promise<Float32Array> when a
  // usable GPU device is bound and the network is eligible; otherwise it falls
  // back to the CPU path and returns a plain number[].
  const output = await network.activate(observation, { useGPU: true });

  return Array.from(output).slice(0, TIER_1_OUTPUT_COUNT);
}

// Allow this example to run directly when bundled as an IIFE in the browser.
if (typeof window !== 'undefined') {
  runGpuRacingController()
    .then((outputs) => {
      console.log('GPU-enabled racing controller output:', outputs);
    })
    .catch((error: unknown) => {
      console.error('GPU racing example failed:', error);
    });
}
