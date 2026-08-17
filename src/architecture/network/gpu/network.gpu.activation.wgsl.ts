/**
 * WGSL activation-function registry for the WebGPU inference fast path.
 *
 * This module maps the compact numeric activation indices used by the worker
 * serialization contract (see `src/multithreading/multi.utils.ts`) to their f32
 * WGSL implementations. Only the first-kernel subset is implemented here;
 * unsupported activations are deliberately omitted from the generated switch.
 */

/**
 * Ordered subset of worker activation indices that the first WGSL kernel
 * supports.
 *
 * These positions must stay in sync with `ACTIVATION_FUNCTIONS` in
 * `src/multithreading/multi.utils.ts` because DNA, workers, and the GPU kernel
 * all use the same numeric index for the same activation.
 */
export const SUPPORTED_ACTIVATION_INDICES = [
  0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13,
] as const;

/**
 * Description of one supported activation function compiled into WGSL shader form.
 */
export interface ActivationFunctionEntry {
  /** Worker-registry activation index used in the WGSL switch. */
  readonly index: number;

  /** Human-readable name for diagnostics and generated comments. */
  readonly name: string;

  /**
   * WGSL function body returning the activated value.
   *
   * The body is wrapped as `fn activation_<index>(x: f32) -> f32 { ... }`
   * when emitted into the shader source.
   */
  readonly body: string;
}

/** Human-readable names for supported activations, keyed by worker index. */
const ACTIVATION_NAMES: Record<
  (typeof SUPPORTED_ACTIVATION_INDICES)[number],
  string
> = {
  0: 'logistic',
  1: 'tanh',
  2: 'identity',
  3: 'step',
  4: 'relu',
  5: 'softsign',
  9: 'bipolar',
  10: 'bipolarSigmoid',
  11: 'hardTanh',
  12: 'absolute',
  13: 'inverse',
};

/** WGSL function bodies for supported activation indices. */
const ACTIVATION_BODIES: Record<
  (typeof SUPPORTED_ACTIVATION_INDICES)[number],
  string
> = {
  0: 'return 1.0 / (1.0 + exp(-x));',
  1: 'return tanh(x);',
  2: 'return x;',
  3: 'return select(0.0, 1.0, x > 0.0);',
  4: 'return max(0.0, x);',
  5: 'return x / (1.0 + abs(x));',
  9: 'return select(-1.0, 1.0, x > 0.0);',
  10: 'return 2.0 / (1.0 + exp(-x)) - 1.0;',
  11: 'return clamp(x, -1.0, 1.0);',
  12: 'return abs(x);',
  13: 'return 1.0 - x;',
};

/** Build the single-statement WGSL body for a supported activation index. */
function buildActivationFunctionBody(
  index: (typeof SUPPORTED_ACTIVATION_INDICES)[number],
): string {
  return ACTIVATION_BODIES[index];
}

/**
 * Build the canonical registry of WGSL activation functions for shader emission.
 *
 * @returns A read-only array of supported activation entries. The order matches
 *   `SUPPORTED_ACTIVATION_INDICES` so callers can emit a deterministic switch.
 */
export function buildActivationRegistry(): readonly ActivationFunctionEntry[] {
  return SUPPORTED_ACTIVATION_INDICES.map((index) => ({
    index,
    name: ACTIVATION_NAMES[index],
    body: buildActivationFunctionBody(index),
  }));
}

/**
 * Format the registry as a block of WGSL function declarations.
 *
 * @param registry - Activation entries from `buildActivationRegistry`.
 * @returns WGSL source containing one `fn activation_<index>(x: f32) -> f32`
 *   declaration per supported index.
 */
export function formatActivationFunctionsWgsl(
  registry: readonly ActivationFunctionEntry[],
): string {
  return registry
    .map(
      (entry) =>
        `fn activation_${entry.index}(x: f32) -> f32 {\n  ${entry.body}\n}`,
    )
    .join('\n\n');
}
