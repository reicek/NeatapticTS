/**
 * Global NeatapticTS configuration contract & default instance.
 *
 * WHY THIS EXISTS
 * --------------
 * A central `config` object offers a convenient, documented surface for end-users (and tests)
 * to tweak library behaviour without digging through scattered constants. Centralization also
 * lets us validate & evolve feature flags in a single place.
 *
 * USAGE PATTERN
 * ------------
 *   import { config } from 'neataptic-ts';
 *   config.warnings = true;              // enable runtime warnings
 *   config.deterministicChainMode = true // opt into deterministic deep path construction
 *
 * Adjust BEFORE constructing networks / invoking evolutionary loops so that subsystems read
 * the intended values while initializing internal buffers / metadata.
 *
 * DESIGN NOTES
 * ------------
 * - We intentionally avoid setters / proxies to keep this a plain serializable object.
 * - Optional flags are conservative by default (disabled) to preserve legacy stochastic
 *   behaviour unless a test or user explicitly opts in.
 */
export interface NeatapticConfig {
  /**
   * Emit safety, performance & deprecation warnings to stdout.
   * Rationale: novices benefit from explicit guidance; advanced users can silence noise.
   * Default: false
   */
  warnings: boolean;

  /**
   * Prefer `Float32Array` for activation & gradient buffers when true.
   * Trade‑off: 2x lower memory + potential SIMD acceleration vs precision of 64-bit floats.
   * Default: false (accuracy prioritized; enable for large populations or constrained memory).
   */
  float32Mode: boolean;

  /**
   * Hard cap for arrays retained per size bucket in the activation buffer pool.
   * Set to a finite non‑negative integer to bound memory. `undefined` = unlimited reuse.
   */
  poolMaxPerBucket?: number;

  /**
   * Prewarm count for commonly used activation sizes. Helps remove first-iteration jitter in
   * tight benchmarking loops. Omit to accept library default heuristics.
   */
  poolPrewarmCount?: number;

  /**
   * Deterministic deep path construction mode (TEST / EDUCATIONAL FEATURE).
   * When enabled: every ADD_NODE mutation extends a single linear input→…→output chain, pruning
   * side branches. This allows tests (and learners) to reason about exact depth after N steps.
   * Disable for realistic evolutionary stochasticity.
   */
  deterministicChainMode?: boolean;

  /**
   * Enable allocation / maintenance of extended gating trace structures.
   * Forward looking flag: currently minimal impact; kept for future advanced credit assignment
   * experiments. Disable if profiling reveals overhead in extremely large recurrent nets.
   * Default: true
   */
  enableGatingTraces?: boolean;

  /**
   * Experimental: Enable Node pooling (reuse Node instances on prune/regrow).
   * Default: false (opt-in while feature stabilizes). When enabled, network growth and
   * pruning paths will acquire/release nodes via NodePool reducing GC churn.
   */
  enableNodePooling?: boolean;

  /**
   * Experimental: Enable slab typed array pooling (reuse large Float/Uint buffers between rebuilds
   * when geometric growth triggers reallocation). Reduces GC churn in topology‑heavy evolution loops.
   * Default: false (opt-in while stabilizing fragmentation heuristics).
   */
  enableSlabArrayPooling?: boolean;

  /**
   * Browser-only (ignored in Node): Target maximum milliseconds of work per microtask slice
   * when performing a large asynchronous slab rebuild via rebuildConnectionSlabAsync(). If set,
   * the chunk size (number of connections copied per slice) is heuristically reduced so that
   * each slice aims to remain below this budget, improving UI responsiveness for very large
   * (>200k edges) networks. Undefined leaves the caller-provided or default chunkSize untouched
   * except for the built-in large-network clamp (currently 50k ops) when total connections >200k.
   */
  browserSlabChunkTargetMs?: number;

  /**
   * Maximum number of typed array slabs retained per (kind:length:bytes) key in the slab array pool.
   * RATIONALE: A very small LRU style cap dramatically limits worst‑case retained memory while
   * still capturing >90% of reuse wins in typical geometric growth / prune churn patterns. Empirically
   * a cap of 4 balances:
   *   - Diminishing returns after the 3rd/4th cached buffer for a given key.
   *   - Keeping educational instrumentation simple (small, inspectable pool state).
   * Set to 0 to disable retention (while still counting metrics) when pooling is enabled.
   * Undefined => library default (currently 4). Negative values are treated as 0.
   */
  slabPoolMaxPerKey?: number;
}

/**
 * Default configuration instance. Override fields as needed before constructing networks.
 */
/**
 * Singleton mutable configuration object consumed throughout the library.
 * Modify properties directly; do NOT reassign the binding (imports retain reference).
 */
export const config: NeatapticConfig = {
  warnings: false, // emit runtime guidance
  float32Mode: false, // numeric precision mode
  deterministicChainMode: false, // deep path test flag (ADD_NODE determinism)
  enableGatingTraces: true, // advanced gating trace infra
  enableNodePooling: false, // experimental node instance pooling
  enableSlabArrayPooling: false, // experimental slab typed array pooling
  // slabPoolMaxPerKey: 4,        // optional override for per-key slab retention cap (default internal 4)
  // browserSlabChunkTargetMs: 3, // example: aim for ~3ms per async slab slice in Browser
  // poolMaxPerBucket: 256,     // example memory cap override
  // poolPrewarmCount: 2,       // example prewarm override
};
