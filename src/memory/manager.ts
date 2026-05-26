import { config, type NeatapticConfig } from '../config';
import {
  MEMORY_DEFAULT_ACTIVATION_POOL_PREWARM_COUNT,
  MEMORY_DEFAULT_SLAB_POOL_MAX_PER_KEY,
  MEMORY_MANAGER_FLAG_NAMES,
  MEMORY_SLAB_GROWTH_FACTOR_BROWSER,
  MEMORY_SLAB_GROWTH_FACTOR_NODE,
  type MemoryManagerConfigSnapshot,
  type MemoryManagerEnvironment,
  type MemoryManagerFlagMap,
  type MemoryManagerFlagName,
  type RegisteredMemoryPool,
} from './config';

type MemoryManagedTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array;

type MemoryManagedTypedArrayConstructor = {
  new (length: number): MemoryManagedTypedArray;
  readonly BYTES_PER_ELEMENT: number;
};

type MemoryTypedArrayPoolKeyStats = {
  created: number;
  reused: number;
  maxRetained: number;
};

type MemoryTypedArrayAllocationStats = {
  fresh: number;
  pooled: number;
  pool: Record<string, MemoryTypedArrayPoolKeyStats>;
};

/**
 * Central memory-configuration owner for pooled and slab-backed runtime paths.
 *
 * This Phase 4 foundation keeps the current global `config` object as the
 * source of truth while adding one focused surface for:
 *
 * 1. resolved memory defaults,
 * 2. temporary override lifecycles during tests and harnesses,
 * 3. resettable pool registration for centralized diagnostics.
 *
 * **Lifecycle** — typical test or harness usage:
 *
 * ```ts
 * import { memoryManager } from './memory/memoryManager';
 *
 * const snapshot = memoryManager.init({ enableSlabArrayPooling: true });
 * try {
 *   // run benchmark or test that exercises pooled paths
 * } finally {
 *   memoryManager.teardown(); // restores original config and resets pools
 * }
 * ```
 *
 * **State machine**:
 *
 * ```mermaid
 * stateDiagram-v2
 *   [*] --> idle : construct
 *   idle --> active : init(overrides)
 *   active --> active : setFlag / allocateTypedArray / registerPool
 *   active --> idle : teardown()
 *   idle --> idle : getConfig / resolveEnvironment
 * ```
 */
export class MemoryManager {
  private readonly configRef: NeatapticConfig;

  private readonly registeredPools = new Map<string, RegisteredMemoryPool>();

  private readonly typedArrayAllocStats = { fresh: 0, pooled: 0 };

  private readonly typedArrayPool: Record<
    string,
    Array<MemoryManagedTypedArray>
  > = Object.create(null);

  private readonly typedArrayPoolMetrics: Record<
    string,
    MemoryTypedArrayPoolKeyStats
  > = Object.create(null);

  private baselineFlags: MemoryManagerFlagMap | null = null;

  /**
   * @param configSnapshot Mutable shared config object consumed by the runtime.
   */
  constructor(configSnapshot: NeatapticConfig = config) {
    this.configRef = configSnapshot;
  }

  /**
   * Resolve the active memory config for one runtime environment.
   *
   * @param environment Optional explicit environment override for tests.
   * @returns Stable memory-config snapshot with resolved defaults.
   */
  getConfig(
    environment: MemoryManagerEnvironment = this.resolveEnvironment(),
  ): MemoryManagerConfigSnapshot {
    const poolPrewarmCount =
      typeof this.configRef.poolPrewarmCount === 'number'
        ? this.configRef.poolPrewarmCount
        : null;
    const browserMemoryBudgetMB = normalizeOptionalPositiveMegabytes(
      this.configRef.browserMemoryBudgetMB,
    );
    const nodeHeapSoftLimitMB = normalizeOptionalPositiveMegabytes(
      this.configRef.nodeHeapSoftLimitMB,
    );
    const slabPoolMaxPerKey =
      typeof this.configRef.slabPoolMaxPerKey === 'number'
        ? Math.max(0, this.configRef.slabPoolMaxPerKey | 0)
        : MEMORY_DEFAULT_SLAB_POOL_MAX_PER_KEY;

    return {
      activationPoolPrewarmCount:
        poolPrewarmCount ?? MEMORY_DEFAULT_ACTIVATION_POOL_PREWARM_COUNT,
      browserMemoryBudgetMB,
      browserSlabChunkTargetMs: this.configRef.browserSlabChunkTargetMs,
      deterministicChainMode: this.configRef.deterministicChainMode ?? false,
      enableGatingTraces: this.configRef.enableGatingTraces ?? true,
      enableNodePooling: this.configRef.enableNodePooling ?? false,
      enableSlabArrayPooling: this.configRef.enableSlabArrayPooling ?? false,
      environment,
      float32Mode: this.configRef.float32Mode,
      nodeHeapSoftLimitMB,
      poolMaxPerBucket: this.configRef.poolMaxPerBucket,
      poolPrewarmCount: this.configRef.poolPrewarmCount,
      slabGrowthFactor:
        environment === 'browser'
          ? MEMORY_SLAB_GROWTH_FACTOR_BROWSER
          : MEMORY_SLAB_GROWTH_FACTOR_NODE,
      slabPoolMaxPerKey,
      warnings: this.configRef.warnings,
    };
  }

  /**
   * Apply temporary overrides while preserving a baseline for teardown.
   *
   * @param overrides Partial memory flag overrides.
   * @param environment Optional explicit environment override for tests.
   * @returns Resolved config after overrides are applied.
   */
  init(
    overrides: Partial<MemoryManagerFlagMap> = {},
    environment?: MemoryManagerEnvironment,
  ): MemoryManagerConfigSnapshot {
    if (!this.baselineFlags) {
      this.baselineFlags = this.captureFlags();
    }

    // Step 1: Apply only declared memory overrides.
    for (const flagName of MEMORY_MANAGER_FLAG_NAMES) {
      if (Object.hasOwn(overrides, flagName)) {
        this.setFlag(
          flagName,
          overrides[flagName] as MemoryManagerFlagMap[typeof flagName],
        );
      }
    }

    // Step 2: Return the resolved effective config.
    return this.getConfig(environment);
  }

  /**
   * Restore the baseline config snapshot and reset registered pools.
   *
   * @returns Nothing.
   */
  teardown(): void {
    // Step 1: Reset registered pools so tests and harnesses release retained state.
    for (const registeredPool of this.registeredPools.values()) {
      registeredPool.reset?.();
    }

    // Step 2: Clear manager-owned typed-array allocator state.
    this.resetTypedArrayAllocator();

    // Step 3: Restore the baseline config when one exists.
    if (!this.baselineFlags) {
      return;
    }

    for (const flagName of MEMORY_MANAGER_FLAG_NAMES) {
      const savedValue = this.baselineFlags[flagName];
      if (savedValue === undefined) {
        delete (this.configRef as Partial<MemoryManagerFlagMap>)[flagName];
        continue;
      }

      this.setFlag(
        flagName,
        savedValue as MemoryManagerFlagMap[typeof flagName],
      );
    }

    this.baselineFlags = null;
  }

  /**
   * Mutate one memory flag on the shared config object.
   *
   * @param flagName Memory flag to update.
   * @param nextValue New value for the flag.
   * @returns Nothing.
   */
  setFlag<K extends MemoryManagerFlagName>(
    flagName: K,
    nextValue: MemoryManagerFlagMap[K],
  ): void {
    (this.configRef as MemoryManagerFlagMap)[flagName] = nextValue;
  }

  /**
   * Register a resettable pool or allocator snapshot provider.
   *
   * @param poolName Stable pool key.
   * @param poolProvider Stats and reset provider.
   * @returns Nothing.
   */
  registerPool(poolName: string, poolProvider: RegisteredMemoryPool): void {
    this.registeredPools.set(poolName, poolProvider);
  }

  /**
   * Acquire one typed array using the manager-owned slab allocator state.
   *
   * @param kind Stable bucket discriminator.
   * @param ctor Typed array constructor.
   * @param length Required logical length.
   * @param bytesPerElement Optional byte width override for bucket keying.
   * @returns Reused or freshly allocated typed array.
   */
  allocateTypedArray(
    kind: string,
    ctor: MemoryManagedTypedArrayConstructor,
    length: number,
    bytesPerElement = ctor.BYTES_PER_ELEMENT,
  ): MemoryManagedTypedArray {
    const memoryConfig = this.getConfig();

    if (!memoryConfig.enableSlabArrayPooling) {
      this.typedArrayAllocStats.fresh += 1;
      return new ctor(length);
    }

    const poolKey = this.resolveTypedArrayPoolKey(
      kind,
      bytesPerElement,
      length,
    );
    const retainedArrays = this.typedArrayPool[poolKey];

    if (retainedArrays && retainedArrays.length > 0) {
      this.typedArrayAllocStats.pooled += 1;
      (this.typedArrayPoolMetrics[poolKey] ||= {
        created: 0,
        reused: 0,
        maxRetained: 0,
      }).reused += 1;
      return retainedArrays.pop() as MemoryManagedTypedArray;
    }

    this.typedArrayAllocStats.fresh += 1;
    (this.typedArrayPoolMetrics[poolKey] ||= {
      created: 0,
      reused: 0,
      maxRetained: 0,
    }).created += 1;
    return new ctor(length);
  }

  /**
   * Release one typed array back to the manager-owned slab allocator.
   *
   * @param kind Stable bucket discriminator.
   * @param bytesPerElement Byte width used for bucket keying.
   * @param typedArray Typed array instance to retain when capacity permits.
   * @returns Nothing.
   */
  releaseTypedArray(
    kind: string,
    bytesPerElement: number,
    typedArray: MemoryManagedTypedArray,
  ): void {
    const memoryConfig = this.getConfig();

    if (!memoryConfig.enableSlabArrayPooling) {
      return;
    }

    const poolKey = this.resolveTypedArrayPoolKey(
      kind,
      bytesPerElement,
      typedArray.length,
    );
    const retainedArrays = (this.typedArrayPool[poolKey] ||= []);

    if (retainedArrays.length < this.resolveTypedArrayPoolCap()) {
      retainedArrays.push(typedArray);
    }

    const poolStats = (this.typedArrayPoolMetrics[poolKey] ||= {
      created: 0,
      reused: 0,
      maxRetained: 0,
    });
    if (retainedArrays.length > poolStats.maxRetained) {
      poolStats.maxRetained = retainedArrays.length;
    }
  }

  /**
   * Read the current manager-owned typed-array allocator counters.
   *
   * @returns Serializable typed-array allocator snapshot.
   */
  getTypedArrayAllocationStats(): MemoryTypedArrayAllocationStats {
    return {
      ...this.typedArrayAllocStats,
      pool: { ...this.typedArrayPoolMetrics },
    };
  }

  /**
   * Read one registered pool snapshot when available.
   *
   * @param poolName Stable pool key.
   * @returns Pool stats or null when the pool is absent.
   */
  getPoolStats<T>(poolName: string): T | null {
    return (
      (this.registeredPools.get(poolName)?.stats?.() as T | undefined) ?? null
    );
  }

  /**
   * Capture the current memory-relevant config fields for later restoration.
   *
   * @returns Baseline memory flag snapshot.
   */
  private captureFlags(): MemoryManagerFlagMap {
    return {
      browserMemoryBudgetMB: this.configRef.browserMemoryBudgetMB,
      browserSlabChunkTargetMs: this.configRef.browserSlabChunkTargetMs,
      deterministicChainMode: this.configRef.deterministicChainMode,
      enableGatingTraces: this.configRef.enableGatingTraces,
      enableNodePooling: this.configRef.enableNodePooling,
      enableSlabArrayPooling: this.configRef.enableSlabArrayPooling,
      float32Mode: this.configRef.float32Mode,
      nodeHeapSoftLimitMB: this.configRef.nodeHeapSoftLimitMB,
      poolMaxPerBucket: this.configRef.poolMaxPerBucket,
      poolPrewarmCount: this.configRef.poolPrewarmCount,
      slabPoolMaxPerKey: this.configRef.slabPoolMaxPerKey,
      warnings: this.configRef.warnings,
    };
  }

  /**
   * Resolve the current runtime environment.
   *
   * @returns `browser` when `window` exists, else `node`.
   */
  private resolveEnvironment(): MemoryManagerEnvironment {
    return typeof window === 'undefined' ? 'node' : 'browser';
  }

  /**
   * Reset the manager-owned typed-array allocator buckets and counters.
   *
   * @returns Nothing.
   */
  private resetTypedArrayAllocator(): void {
    for (const poolKey of Object.keys(this.typedArrayPool)) {
      delete this.typedArrayPool[poolKey];
    }

    for (const poolKey of Object.keys(this.typedArrayPoolMetrics)) {
      delete this.typedArrayPoolMetrics[poolKey];
    }

    this.typedArrayAllocStats.fresh = 0;
    this.typedArrayAllocStats.pooled = 0;
  }

  /**
   * Resolve the non-negative per-key typed-array retention cap.
   *
   * @returns Integer retention cap for manager-owned typed-array buckets.
   */
  private resolveTypedArrayPoolCap(): number {
    return this.getConfig().slabPoolMaxPerKey | 0;
  }

  /**
   * Build the stable bucket key for one typed-array allocation class.
   *
   * @param kind Stable bucket discriminator.
   * @param bytesPerElement Typed array byte width.
   * @param length Typed array logical length.
   * @returns Stable key used by the manager-owned allocator.
   */
  private resolveTypedArrayPoolKey(
    kind: string,
    bytesPerElement: number,
    length: number,
  ): string {
    return `${kind}:${bytesPerElement}:${length}`;
  }
}

/**
 * Shared singleton used by the current runtime while Phase 4 centralization is in flight.
 */
export const defaultMemoryManager = new MemoryManager(config);

function normalizeOptionalPositiveMegabytes(
  configuredMegabytes: number | undefined,
): number | undefined {
  if (
    typeof configuredMegabytes !== 'number' ||
    !Number.isFinite(configuredMegabytes) ||
    configuredMegabytes <= 0
  ) {
    return undefined;
  }

  return configuredMegabytes;
}
