/**
 * NGE juvenile async weight-variant evaluator consumer.
 *
 * This module builds and evaluates lifecycle-aware **multi-connection** weight
 * variant patches for NGE juvenile networks. A patch is a set of simultaneous
 * connection perturbations that are applied, scored as a group, and then rolled
 * back. The highest-scoring patch's representative perturbation is reported as
 * the winning single-connection variant so that downstream grow-stabilize
 * reconstruction (which is outside this slice) can commit it without being
 * edited.
 *
 * Stage-specific variant counts are resolved from explicit overrides, the
 * supplied acceleration configuration, or built-in lifecycle defaults.
 */

import { DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD } from '../../acceleration/acceleration.constants';
import { autoEnableAcceleration } from '../../acceleration/acceleration.orchestrator';
import type { AccelerationConfig } from '../../acceleration/acceleration.types';
import {
  DEFAULT_VARIANT_SCORER,
  type VariantEvaluationNetwork,
  type VariantScorer,
  type WeightVariant,
  type WeightVariantResult,
} from '../../acceleration/acceleration.variants';
import {
  NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
  NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
  NGE_LIFECYCLE_DEFAULT_JUVENILE_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT,
  NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT,
  NGE_VARIANT_PATCH_MIN_CONNECTION_COUNT,
  NGE_VARIANT_PATCH_SEED_OFFSET,
  NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR,
  NGE_VARIANT_PATCH_SIZE_DIVISOR,
  NGE_VARIANT_SIZE_FACTOR_FLOOR,
  NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS,
  NGE_VARIANT_WIDTH_FACTOR_MAX,
} from './neat.nge-juvenile.constants';
import type { NgeLifecycleStage } from './neat.nge-juvenile.lifecycle-stages';

/** Options forwarded to {@link evaluateNgeWeightVariants}. */
export interface EvaluateNgeWeightVariantsOptions {
  /** Optional acceleration configuration forwarded to the variant evaluator. */
  accelerationConfig?: AccelerationConfig;
  /** Optional per-lifecycle-stage variant count overrides. */
  stageVariantCounts?: Partial<Record<NgeLifecycleStage, number>>;
  /**
   * Optional custom scorer; defaults to {@link DEFAULT_VARIANT_SCORER}.
   * When overridden, ensure the scorer shares the same score space and
   * direction as the baseline used by the caller, or the grow-stabilize
   * commit inequality will compare incommensurate values.
   */
  scoreFn?: VariantScorer;
}

/**
 * One representative perturbation plus the full multi-connection patch used
 * for scoring.
 *
 * The representative is the single-connection perturbation that downstream
 * consumers see as the "winning variant". The full patch contains additional
 * simultaneous perturbations so that the score reflects a broader local
 * search step.
 */
export interface NgeWeightVariantPatch {
  /** Single-connection perturbation used as the public representative. */
  representative: WeightVariant;
  /** All perturbations applied together when scoring this patch. */
  perturbations: readonly WeightVariant[];
}

/**
 * Evaluate multi-connection weight variant patches for an NGE juvenile network
 * asynchronously.
 *
 * The function resolves the appropriate variant count from the supplied
 * lifecycle stage, builds deterministic patches over the network's connection
 * list, and scores each patch by applying all of its perturbations at once. All
 * connection weights are restored after every patch, so the network is returned
 * to its original state once the promise resolves. Patches are evaluated
 * sequentially on the live network so that activation always reads the patched
 * weights that were actually applied.
 *
 * The default scorer returns negative mean-squared-error against `target`.
 * If the caller supplies a custom `scoreFn`, it must produce values in the
 * same score space as the caller's baseline; otherwise the stabilization
 * commit inequality `bestScore > baselineScore + threshold` can never be
 * satisfied.
 *
 * Stage-specific variant counts can be supplied through `options` or through
 * `options.accelerationConfig.stageVariantCounts`. When a stage count is given,
 * it overrides the built-in default for that stage (for example, `baby: 256`
 * replaces the default baby count).
 *
 * Background reading:
 * - NEAT and topology-evolving neuroevolution:
 *   K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through
 *   Augmenting Topologies," Evolutionary Computation, vol. 10, no. 2,
 *   pp. 99-127, 2002.
 *   [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)
 *
 * @param network - Network surface to evaluate.
 * @param stage - Current NGE lifecycle stage.
 * @param inputs - Input batch, one vector per sample.
 * @param target - Target output vector for the default scorer.
 * @param seed - Optional determinism seed for patch generation.
 * @param options - Optional NGE-specific overrides.
 *   May include `accelerationConfig` to control backend selection, or
 *   `stageVariantCounts` to override per-stage variant counts.
 * @returns Promise resolving to per-variant scores and backend metadata.
 *
 * @example
 * ```ts
 * const result = await evaluateNgeWeightVariants(
 *   network,
 *   'baby',
 *   [[0.5, 0.5]],
 *   [1.0],
 *   42,
 * );
 * console.log(result.metadata.variantCount); // 16 for baby stage
 * ```
 *
 * @example
 * ```ts
 * const result = await evaluateNgeWeightVariants(
 *   network,
 *   'baby',
 *   [[0.5, 0.5]],
 *   [1.0],
 *   42,
 *   {
 *     accelerationConfig: {
 *       parallelVariantCount: 256,
 *       stageVariantCounts: { baby: 256 },
 *     },
 *   },
 * );
 * console.log(result.metadata.variantCount); // 256
 * ```
 */
export async function evaluateNgeWeightVariants(
  network: VariantEvaluationNetwork,
  stage: NgeLifecycleStage,
  inputs: number[][],
  target: number[],
  seed?: number,
  options?: EvaluateNgeWeightVariantsOptions,
): Promise<WeightVariantResult> {
  const variantCount = resolveVariantCountForStage(
    stage,
    options?.stageVariantCounts,
    options?.accelerationConfig,
  );
  const patches = buildVariants(network, stage, variantCount, seed);

  const status = await autoEnableAcceleration({
    nodeCount: network.nodes.length,
    batchParallelCount: patches.length,
    config: options?.accelerationConfig,
  });

  const gpuEligible =
    network.nodes.length >= DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
  const device =
    status.mode === 'gpu' && status.gpu.device && gpuEligible
      ? status.gpu.device
      : undefined;
  const useGPU = device !== undefined;
  const previousDevice = network.gpuDevice;
  if (device !== undefined) {
    network.gpuDevice = device;
  }

  let maxDelta = 0;
  for (const patch of patches) {
    for (const { delta } of patch.perturbations) {
      maxDelta = Math.max(maxDelta, Math.abs(delta));
    }
  }

  const scores: number[] = [];
  try {
    // Evaluate patches sequentially on the live network. This guarantees that
    // activation reads the patched weights that were actually applied and that
    // the original weights are restored before the next patch begins.
    for (const patch of patches) {
      scores.push(
        await evaluatePatch(
          network,
          patch,
          inputs,
          target,
          useGPU,
          options?.scoreFn,
        ),
      );
    }
  } finally {
    network.gpuDevice = previousDevice;
  }

  let bestIndex = 0;
  let bestScore = scores[0] ?? Number.NEGATIVE_INFINITY;
  for (let index = 1; index < scores.length; index++) {
    const score = scores[index]!;
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  }

  return {
    bestIndex,
    bestScore,
    scores,
    metadata: {
      backend: status.mode,
      variantCount: patches.length,
      scaleDivisor: maxDelta > 0 ? maxDelta : 1,
      scorer: options?.scoreFn === undefined ? 'default' : 'custom',
    },
  };
}

/**
 * Resolve the variant count for a lifecycle stage.
 *
 * Resolution order: explicit `overrides` for the stage, then
 * `accelerationConfig.stageVariantCounts` for the stage, then built-in
 * lifecycle defaults. Baby-stage networks get many variants (default 16),
 * adult/equilibrium networks get few (default 4), and juvenile gets the
 * midpoint (default 8). Embryo mirrors baby because the network is still tiny.
 *
 * @param stage - Current NGE lifecycle stage.
 * @param overrides - Optional per-stage variant count overrides.
 * @param accelerationConfig - Optional acceleration configuration carrying
 *   per-stage variant counts.
 * @returns Number of variants to evaluate.
 *
 * @example
 * ```ts
 * // Built-in juvenile default: 8 variants.
 * const count = resolveVariantCountForStage('juvenile');
 *
 * // Override for a single stage without touching acceleration config.
 * const tiny = resolveVariantCountForStage('adult', { adult: 2 });
 * ```
 */
export function resolveVariantCountForStage(
  stage: NgeLifecycleStage,
  overrides?: Partial<Record<NgeLifecycleStage, number>>,
  accelerationConfig?: AccelerationConfig,
): number {
  const override = overrides?.[stage];
  if (override !== undefined) {
    return Math.max(1, Math.floor(override));
  }

  const stageCounts = accelerationConfig?.stageVariantCounts;
  const configCount =
    stage === 'embryo'
      ? stageCounts?.baby
      : stage === 'equilibrium'
        ? stageCounts?.adult
        : stageCounts?.[stage];
  if (configCount !== undefined) {
    return Math.max(1, Math.floor(configCount));
  }

  switch (stage) {
    case 'embryo':
    case 'baby':
      return NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT;
    case 'juvenile':
      return NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT;
    case 'adult':
    case 'equilibrium':
    default:
      return NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT;
  }
}

/**
 * Resolve the representative delta for a variant index using an
 * endpoint-inclusive linear spread over [-magnitude, +magnitude].
 *
 * For a single variant the probe is the positive endpoint, preserving a
 * non-zero weight nudge. For two or more variants the spread is symmetric,
 * unique, and reaches the exact endpoints at the first and last indices.
 *
 * @param index - Variant index in [0, variantCount).
 * @param variantCount - Total number of variants in the spread.
 * @param magnitude - Maximum absolute delta for the spread.
 * @returns Representative delta for the indexed variant.
 *
 * @example
 * ```ts
 * const delta = resolveRepresentativeDelta(0, 16, 0.15); // -0.15
 * const last = resolveRepresentativeDelta(15, 16, 0.15); // +0.15
 * ```
 */
export function resolveRepresentativeDelta(
  index: number,
  variantCount: number,
  magnitude: number,
): number {
  const safeCount = Math.max(1, variantCount);
  if (safeCount === 1) {
    return roundDelta(magnitude);
  }
  const t = (2 * index) / (safeCount - 1) - 1;
  return roundDelta(magnitude * t);
}

/**
 * Resolve the effective mutation magnitude for a lifecycle stage, variant
 * count, and network connection count.
 *
 * The magnitude scales with both the exploration width (more variants need a
 * wider total range) and the network size (larger networks need smaller local
 * steps). Small networks retain the stage baseline because the size factor is
 * clamped at 1.0.
 *
 * @param stage - Current NGE lifecycle stage.
 * @param variantCount - Number of parallel variants being evaluated.
 * @param connectionCount - Number of connections in the live network.
 * @returns Effective absolute delta magnitude to use for this variant batch.
 *
 * @example
 * ```ts
 * const magnitude = resolveEffectiveMagnitude('baby', 16, 5); // 0.15
 * ```
 */
export function resolveEffectiveMagnitude(
  stage: NgeLifecycleStage,
  variantCount: number,
  connectionCount: number,
): number {
  const base = resolveStageMagnitude(stage);
  const vRef = resolveStageVariantRef(stage);
  const cRef = NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS;

  const widthFactor = Math.max(
    1,
    Math.min(
      1.0 + Math.log10(Math.max(1, variantCount / vRef)),
      NGE_VARIANT_WIDTH_FACTOR_MAX,
    ),
  );
  const sizeFactor = Math.max(
    NGE_VARIANT_SIZE_FACTOR_FLOOR,
    Math.min(Math.sqrt(cRef / Math.max(1, connectionCount)), 1),
  );

  return roundDelta(base * widthFactor * sizeFactor);
}

/**
 * Build deterministic multi-connection weight variant patches for a network
 * surface.
 *
 * Each patch perturbs a small, distinct subset of connections. The first
 * perturbation in the patch is the **representative**: it uses the same
 * deterministic rule as the legacy single-connection variant builder so that
 * downstream grow-stabilize reconstruction can recreate the winning variant
 * without knowing the full patch contents.
 *
 * @param network - Network surface whose connection list is used for indexing.
 * @param stage - Current NGE lifecycle stage; controls mutation magnitude.
 * @param count - Number of variant patches to generate.
 * @param seed - Optional determinism seed for patch generation.
 * @returns Array of deterministic weight variant patches.
 *
 * @example
 * ```ts
 * const patches = buildVariants(network, 'baby', 16, 12345);
 * // patches[0].representative uses connection 0 and the smallest negative delta.
 * // The same seed reproduces identical patches on every run.
 * ```
 */
export function buildVariants(
  network: VariantEvaluationNetwork,
  stage: NgeLifecycleStage,
  count: number,
  seed?: number,
): NgeWeightVariantPatch[] {
  const connectionCount = network.connections.length;
  const effectiveMagnitude = resolveEffectiveMagnitude(
    stage,
    count,
    connectionCount,
  );
  const baseSeed = seed ?? 0;
  const maxPatchSize = Math.min(
    NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT,
    connectionCount,
  );
  const desiredPatchSize = Math.max(
    1,
    Math.floor(connectionCount / NGE_VARIANT_PATCH_SIZE_DIVISOR),
  );
  const patchSize =
    connectionCount > 0
      ? Math.max(
          NGE_VARIANT_PATCH_MIN_CONNECTION_COUNT,
          Math.min(maxPatchSize, desiredPatchSize),
        )
      : 0;

  const seedStride = Math.max(
    NGE_VARIANT_PATCH_SEED_OFFSET,
    Math.max(1, patchSize) * NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR,
  );

  const patches: NgeWeightVariantPatch[] = [];
  for (let index = 0; index < count; index++) {
    const rand = createSeededRandom(baseSeed + index * seedStride);
    const representative: WeightVariant = {
      weightIndex: connectionCount > 0 ? index % connectionCount : 0,
      delta: resolveRepresentativeDelta(index, count, effectiveMagnitude),
    };
    const otherCount = Math.max(0, patchSize - 1);
    const otherIndices = pickDistinctIndicesExcluding(
      representative.weightIndex,
      otherCount,
      connectionCount,
      rand,
    );
    const perturbations: WeightVariant[] = [
      representative,
      ...otherIndices.map((weightIndex) => ({
        weightIndex,
        delta: boundedDelta(effectiveMagnitude, rand),
      })),
    ];
    patches.push({ representative, perturbations });
  }

  return patches;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resolve the baseline mutation magnitude for a lifecycle stage.
 *
 * Embryo uses the baby baseline so early development is not artificially
 * constrained relative to the default baby configuration.
 *
 * @param stage - Current NGE lifecycle stage.
 * @returns Baseline absolute delta magnitude for the stage.
 * @internal
 */
function resolveStageMagnitude(stage: NgeLifecycleStage): number {
  switch (stage) {
    case 'embryo':
    case 'baby':
      return NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE;
    case 'juvenile':
      return NGE_LIFECYCLE_DEFAULT_JUVENILE_MUTATION_MAGNITUDE;
    case 'adult':
    case 'equilibrium':
    default:
      return NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE;
  }
}

/**
 * Resolve the reference variant count used to scale the effective magnitude
 * for a lifecycle stage.
 *
 * @param stage - Current NGE lifecycle stage.
 * @returns Reference variant count for the stage.
 * @internal
 */
function resolveStageVariantRef(stage: NgeLifecycleStage): number {
  switch (stage) {
    case 'embryo':
    case 'baby':
      return NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT;
    case 'juvenile':
      return NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT;
    case 'adult':
    case 'equilibrium':
    default:
      return NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT;
  }
}

/**
 * Create a deterministic pseudo-random number generator.
 *
 * Uses the 32-bit mulberry32 algorithm so patch contents are reproducible
 * across runtimes for the same seed. Mulberry32 is a small LCG-style PRNG
 * described in the
 * [PCG family overview (Wikipedia)](https://en.wikipedia.org/wiki/Permuted_congruential_generator#Other_simple_generators)
 * and popularized by Tommy Ettinger's public-domain reference implementation.
 *
 * @param seed - Integer seed.
 * @returns Deterministic random function returning values in [0, 1).
 * @internal
 */
function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/**
 * Pick a set of distinct connection indices excluding a given index.
 *
 * Uses rejection sampling against a deterministic PRNG. If the requested count
 * exceeds the available distinct indices, the function returns as many as
 * possible.
 *
 * @param exclude - Index that must not appear in the result.
 * @param count - Desired number of additional indices.
 * @param total - Size of the connection list.
 * @param rand - Deterministic random source.
 * @returns Array of distinct indices different from `exclude`.
 * @internal
 */
export function pickDistinctIndicesExcluding(
  exclude: number,
  count: number,
  total: number,
  rand: () => number,
): number[] {
  if (count <= 0 || total <= 1) {
    return [];
  }

  const picked: number[] = [];
  const used = new Set<number>([exclude]);
  let attempts = 0;
  const maxAttempts = count * 100 + 1_000;
  while (picked.length < count && attempts < maxAttempts) {
    attempts++;
    const index = Math.floor(rand() * total);
    if (!used.has(index)) {
      used.add(index);
      picked.push(index);
    }
  }
  return picked;
}

/**
 * Round a delta to a fixed number of decimal places to avoid floating-point
 * noise propagating into tests and downstream consumers.
 *
 * @param delta - Raw delta value.
 * @returns Delta rounded to 10 decimal places.
 * @internal
 */
function roundDelta(delta: number): number {
  return Math.round(delta * 1e10) / 1e10;
}

/**
 * Sample a signed delta within [-magnitude, +magnitude].
 *
 * @param magnitude - Maximum absolute delta.
 * @param rand - Deterministic random source.
 * @returns Bounded signed delta.
 * @internal
 */
function boundedDelta(magnitude: number, rand: () => number): number {
  return roundDelta((rand() * 2 - 1) * magnitude);
}

/**
 * Score a single patch by applying all perturbations, running the network on
 * the input batch, and restoring the original weights.
 *
 * @param network - Network surface to mutate temporarily.
 * @param patch - Patch whose perturbations should be applied.
 * @param inputs - Input batch, one vector per sample.
 * @param target - Target output vector for the default scorer.
 * @param useGPU - Whether to forward `useGPU: true` to `network.activate`.
 * @returns Score of the patched network (higher is better).
 * @internal
 */
async function evaluatePatch(
  network: VariantEvaluationNetwork,
  patch: NgeWeightVariantPatch,
  inputs: number[][],
  target: number[],
  useGPU: boolean,
  scoreFn: VariantScorer = DEFAULT_VARIANT_SCORER,
): Promise<number> {
  const originals = new Map<number, number>();
  for (const { weightIndex, delta } of patch.perturbations) {
    const connection = network.connections[weightIndex];
    if (connection === undefined) {
      continue;
    }
    originals.set(weightIndex, connection.weight);
    connection.weight += delta;
  }

  try {
    const outputs: number[][] = [];
    const activateOptions = useGPU ? { useGPU: true } : { useGPU: false };
    for (const input of inputs) {
      const raw = await Promise.resolve(
        network.activate(input, activateOptions),
      );
      outputs.push([...raw]);
    }
    return scoreFn(outputs, target);
  } finally {
    for (const [weightIndex, original] of originals) {
      const connection = network.connections[weightIndex];
      if (connection === undefined) {
        continue;
      }
      connection.weight = original;
    }
  }
}
