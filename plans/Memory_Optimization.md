# Memory Optimization Plan (Multi-Layer Strategy for Very Large Networks)

Goal: Enable construction, evolution, and training of networks scaling toward **10^6+ (stretch 10^7)** connections on commodity JS/TS runtimes across **Node** and **Browser** environments while preserving performance, responsiveness, and determinism. This revision separates **Common Core Optimizations** (portable) from **Targeted Environment Optimizations** (Node-only or Browser-only) and aligns later phases explicitly with **Hyper MorphoNEAT** (indirect encodings, morphogenesis, caching). All optimizations follow pay‑for‑use: zero (or near-zero) overhead when disabled.

---

## Guiding Principles
1. Pay-for-use: No overhead unless feature enabled (flags + lazy allocation).
2. Reuse & Pool: Prefer object pooling / typed array slabs over many small objects (already partially implemented for `Connection`).
3. Structural Sparsity First: Keep graphs sparse; optimize dense fast-paths only when needed.
4. Incremental & Measurable: Each optimization introduces a benchmark & memory snapshot (Node + Browser where feasible).
5. Branch Containment: Refactors isolated (one concern per PR) with compatibility shims where needed.
6. Environment Awareness: Node path may adopt heavier instrumentation & persistent caches; Browser path emphasizes chunking, responsiveness, and quota safety.
7. Hyper Alignment: Memory layers anticipate Hyper MorphoNEAT phases (indirect generation, morphogenesis churn, phenotype/adjacency caches).
8. Transparent API: All environment-specific memory optimizations (Node vs Browser, slabs, pooling, precision) remain behind a stable public memory/network API; callers never branch on environment—feature flags + `memoryStats()` abstraction handle differences.

---

## Environment Constraints & Differentiators

Dimension | Node (V8 server/CLI) | Browser (varied UAs)
--------- | -------------------- | --------------------
Typical Heap Ceiling | ~2–4 GB (tunable via `--max-old-space-size`) | Practically 1–2 GB (lower on mobile) per tab
Introspection APIs | `process.memoryUsage()`, `v8.getHeapStatistics()`, heap snapshots | `performance.memory` (Chrome), experimental `measureUserAgentSpecificMemory()`, internal counters
Background Threads | `worker_threads`, SharedArrayBuffer, Atomics | Web Workers, (SAB if cross-origin isolated)
Persistence | FS, memory-mapped files, streams | IndexedDB / OPFS, Service Worker cache
Large Buffer Handling | `Buffer` (zero-copy slices), streams | `ArrayBuffer` / transferable objects (clone cost relevant)
Scheduling | Long tasks tolerable (batch/CLI) | Must avoid frame jank (>16ms @60fps)
Acceleration | N-API, native libs, WASM SIMD | WASM SIMD, WebGPU (emerging)

Implications: Browser emphasizes **cooperative scheduling, micro-chunk allocation, transfer minimization, heuristic memory guards**. Node emphasizes **larger batch allocations, deeper instrumentation, optional persistence / native acceleration**.

---

## Target Metrics (Refine After Baseline Measurement)
Metric | Baseline (TBD) | Target Phase | Goal
------ | -------------- | ------------ | ----
Avg bytes / active connection | measure | 4 | -25% vs baseline
Peak heap growth per 100k added connections | measure | 5 | < 12 MB
GC pause impact (allocation-heavy evolutions) | measure | 6 | -30% vs baseline
Phenotype rebuild allocation churn | measure | 7 | amortized O(1) per edge via reuse

### Additional Hyper-Specific Metrics (Introduced Once Hyper Phases Land)
Metric | Baseline (Post Phase 1 Hyper) | Target Phase | Goal
------ | ----------------------------- | ------------ | ----
Bytes / genotype (symbolic) | measure | Hyper 8 | < 0.5% of phenotype bytes at scale (document)
Adjacency cache hit ratio | measure | Hyper 3 | > 70% on repeated evaluations (bench harness scenario)
Adjacency cache bytes / active connection | measure | Hyper 3 | < +6 bytes (amortized) with cap
Plasticity side-buffer bytes / active plastic connection | measure | Hyper 5 | < 8 bytes (float32 rate + accumulator) 
Morphogenesis churn leak slope (pool high-water mark) | measure | Hyper 4 | ~0 over final 20% iterations
Rebuild time variance (p95 / median) | measure | Hyper 8 | < 2.5× ratio (determinism stability)
Cache eviction overhead (per miss) | measure | Hyper 7 | < 5% of total build time in stress test

Baseline numbers gathered in Phase 0 using synthetic builder + Node `--inspect` snapshots (authoritative). Browser adds heuristic baseline using internal slab + pool accounting cross-checked with `performance.memory.usedJSHeapSize` (where available). Target heuristic error tolerance <15%.

Future mobile tier baseline (optional) for adaptive budgets.

---

## Optimization Layers Overview (Environment-Aware)
Layer | Focus | Common Techniques | Node Emphasis | Browser Emphasis | Hyper Relevance
----- | ----- | ----------------- | ------------- | ---------------- | ---------------
L1 | Data structure slimming | Field layout audit, bitfields | Alignment for future WASM structs | Stable hidden classes | Pre-req (0–1)
L2 | Object pooling & reuse | Pools for connections/nodes/activations | Larger pre-warm, background fill | Adaptive trimming on visibility/memory pressure | Morph churn (4)
L3 | Slab / SoA packing | Weights/gains/flags in typed arrays | Larger geometric growth (1.75–2×) | Smaller growth steps (1.25×) with yields | Fast rebuild (3,7)
L4 | Sparse & budget control | Coordinate lists, prune-regrow | Bulk generation batches | Incremental growth microtasks | Morphogenesis governance (4)
L5 | Adaptive precision | f64→f32; optional float16 | bfloat16 (WASM) exploration | f16 gating + loss scaling | Large phenotypes (5)
L6 | Streaming & chunking | Windowed activations | Worker_threads pipelines | Frame-sliced forward passes | Temporal scaling (9)
L7 | Caching (adjacency/phenotype) | Hash → SoA reuse | Disk (FS) persistence | IndexedDB/OPFS shards | Hyper caches (7)
L8 | Compression / serialization | Delta + RLE | Gzip/zstd streaming | Incremental decode/gzip (WASM) | Shipping phenotypes/genotypes (8)
L9 | Scheduling & telemetry | Unified stats APIs | High-res timers + deep snapshots | Cooperative scheduling + jank metrics | Deterministic rebuild windows

L7 subdivisions: (a) adjacency cache (structural + CPPN + threshold + precision hash) and (b) phenotype slab reuse (ref-counted typed arrays). Node may persist cache entries to disk; Browser may hydrate from IndexedDB/OPFS with async chunk loading.

---

## Phase Plan (Concrete Steps & Affected Files)

Each phase notes: (C) Common, (N) Node-specific, (B) Browser-specific, (H) Hyper alignment.

### Phase 0 – Baseline Instrumentation
Files: `test/benchmarks/benchmark.memory.test.ts`, `test/benchmarks/benchmark.buildVariants.test.ts`, `test/benchmarks/benchmark.report.test.ts`, `test/benchmarks/benchmark.browser.memory.test.ts` (placeholder), `src/utils/memory.ts`, `src/architecture/network.ts`
Scope: Establish baseline memory & performance metrics; dual-path benchmark (TS source `src/` vs compiled `dist/`) + future browser harness for dev/prod bundle deltas.

Status Legend: [DONE] complete · [WIP] in progress · [PENDING] not started · [DEFERRED] postponed intentionally.

Interim Notes:
- JSON artifact (`test/benchmarks/benchmark.results.json`) is the single source: baseline, variants, aggregated stats, deltas, browserRuns, fieldAudit, and now a trimmed `history` (≤10 snapshots) with only summaries + forward delta %. This prevents unbounded growth.
- Browser harness automated: results JSON augmented with `browserRuns` (dev/prod) including bundle size, timings, and performance.memory when available.
- Aggregation executed inside Jest for deterministic CI visibility.
- Large size (200k) path active locally; CI may restrict via env gating later if runtime pressure observed.
- History migration logic normalizes any pre-slim entries on next write ensuring artifact stability after refactors.

#### Phase 0 Results (Current Run Snapshot)

Methodology:
- Synthetic dense bipartite networks sized by choosing input≈sqrt(N), output≈ceil(N/input), pruning excess edges.
- Timings: buildMs = construction wall-clock; fwdAvgMs = average over 5 forward passes.
- Metrics aggregated per (mode,size,scenario) with mean/p50/p95/std; deltas computed Dist−Src (positive = dist slower / larger).
- Thresholds: |time Δ%| > 5% or |bytes/conn Δ%| > 3% flagged (F).

Variant Delta Table (updated from latest benchmark.results.json) – Timing & Bytes/Conn (includes 200k):

```
Size    Metric            Src Mean      Dist Mean     Δ Abs         Δ %        Flag          Result
------  ----------------- ------------- ------------- ------------- ---------- ------------- ------------------------------------------------------------
1k      buildMsMean       0.5478        0.5894        +0.0416       +7.59      (REG)         Dist slower small build (artifact still src fallback)
1k      fwdAvgMsMean      0.04314       0.04190       -0.00124      -2.87      OK            Minor faster dist (within ±5% threshold)
1k      bytesPerConnMean  69.0000       69.0000       +0.0000        0.00      OK            Parity
10k     buildMsMean       6.1757        4.1901        -1.9856      -32.15      (improved)    Dist faster build (likely warm module cache)
10k     fwdAvgMsMean      0.34838       0.35426       +0.00588      +1.69      OK            Parity (within threshold)
10k     bytesPerConnMean  65.0000       65.0000       +0.0000        0.00      OK            Parity
50k     buildMsMean       26.3226       25.5239       -0.7987       -3.03      OK            Near parity build (below 5% threshold)
50k     fwdAvgMsMean      2.14683       3.72180       +1.57497     +73.36      (REG)         Dist slower forward mid-scale (investigate w/ real dist)
50k     bytesPerConnMean  65.0000       65.0000       +0.0000        0.00      OK            Parity
100k    buildMsMean       51.1675       50.4350       -0.7325       -1.43      OK            Parity build
100k    fwdAvgMsMean      6.51225       6.91490       +0.40265      +6.18      (REG)         Dist slower forward (>5%)
100k    bytesPerConnMean  64.0000       64.0000       +0.0000        0.00      OK            Parity
200k    buildMsMean       106.5497      106.8084      +0.2587       +0.24      OK            Parity build
200k    fwdAvgMsMean      12.8794       12.6563       -0.2231       -1.73      OK            Parity / minor faster dist
200k    bytesPerConnMean  64.0000       64.0000       +0.0000        0.00      OK            Parity
```

Node Heap Metrics (updated snapshot; single samples, std=0):

Methodology: Captured immediately post-build & forward cycle; process noise + single-sample per size (std=0). Threshold gating TBD. MB = bytes / 1,048,576.

```
Size    Metric         Src Bytes    Dist Bytes    Src MB    Dist MB   Δ Bytes      Δ %     Note
------  -------------- ------------ ------------ --------- --------- ----------- ------- -------------------------------------------------
1k      heapUsedMean   777016816    777825072    740.9     741.6     +808,256    +0.10  Noise band
1k      rssMean        866582528    866660352    826.2     826.3     +77,824     +0.01  Identical
10k     heapUsedMean   773412264    780998792    737.3     744.9     +7,586,528  +0.98  Noise / GC timing
10k     rssMean        870572032    871620608    830.3     831.3     +1,048,576  +0.12  Noise
50k     heapUsedMean   793543744    800502416    756.8     763.1     +6,958,672  +0.88  Modest drift
50k     rssMean        882229248    901726208    841.5     860.1     +19,496,960 +2.21  Allocation alignment
100k    heapUsedMean   832158376    863268072    793.8     823.1     +31,109,696 +3.74  Growing drift
100k    rssMean        931643392    960872448    888.7     916.3     +29,229,056 +3.14  Still informational
200k    heapUsedMean   925310752    986520992    882.5     941.0     +61,210,240 +6.62  Scaling tail
200k    rssMean        1025396736   1088782336   977.9     1038.1    +63,385,600 +6.18  Peak sample; below cap
```

Interpretation (concise update – post bitfield & history slimming run):
- Dist path still a src fallback (no true production build); deltas continue to reflect JIT/cache noise rather than minified/treeshaken differences.
- Forward regressions remain volatile: notable mid-scale degradation (50k +92.6%, 10k +23.4%, 100k +7.0%, 200k +1.8%). High variance + single-sample aggregation underscores urgency of multi-sample variance mode before adding regression gates.
- Build times mixed (some dist faster at 10k & 50k, slower at 1k & 100k); without a real dist artifact this is not action‑worthy.
- Per-connection bytes unchanged (69→65→64 across sizes) indicating bitfield slimming reduced key count but not yet heap footprint (object header + property array unchanged until further field removals / SoA packing).
- Heap deltas: modest drift grows with size; 200k shows ~5.4% heapUsed increase in dist vs src—still informational until variance captured.
- History snapshots now fully slimmed (≤10 retained) with populated summaries and forward delta % arrays; legacy verbose entries successfully normalized.
- Field audit confirms Connection keys reduced 12→11 via consolidation (`enabled`, `dcMask` now private `_flags`), validating Phase 1 step 13.
- Next reachable reduction (target ≤10 keys): elide `gain` when 1.0 via fast-path accessor or move into bitfield+scaling map; consider packing `gater` presence bit.
- Browser harness still integrated automatically; separate browser memory test remains skipped pending cross‑env tolerance establishment.

Field Audit Snapshot (current):
- Node keys (14): activation,bias,connections,derivative,error,geneId,index,mask,old,previousDeltaBias,squash,state,totalDeltaBias,type
- Connection keys (10): _flags,eligibility,from,gater,innovation,previousDeltaWeight,to,totalDeltaWeight,weight,xtrace (neutral gain virtualized via symbol)

Upcoming expectation:
- Validate next audit reflects Connection key count stability at 10; evaluate virtualizing `gater` (null ⇒ no enumerable key) with presence bit in flags.

Slimming Candidates (Phase 1 targeting) – status update:
- Connection: [DONE] Bitfield applied (enabled + dcMask) 12→11; [DONE] neutral gain virtualized (symbol) 11→10; [NEXT] pack gater presence bit & virtualize null gater.
- Node: (PENDING) Externalize error object (SoA) Phase 3; evaluate lazy derivative store after measuring derivative access frequency.
- Shared: (PENDING) Enum for node type (low priority) – micro benchmark before action.

## Phase 0 – Final Achievements Summary

Core Deliverables Achieved:
1. Unified Jest benchmark harness (Node) producing structured JSON (`benchmark.results.json`) replacing ad-hoc console output.
2. Dual build-variant (src vs dist) timing + memory aggregation with mean/p50/p95/std per metric.
3. Large-scale synthetic sizing extended to 200k connections with adaptive iteration scaling.
4. Browser harness (esbuild dev/prod + headless execution) integrated directly into tests; bundle size + runtime + performance.memory captured.
5. Historical snapshot system (slimmed) with capped rolling history (≤10) and forward delta pct arrays for quick regression scanning.
6. Field audit instrumentation enumerating Node & Connection own keys; regression signal for slimming phases.
7. Connection object slimming (bitfield `_flags` for enabled + dcMask) reducing keys 12→11.
8. Artifact slimming: history normalization, removal of verbose legacy sections, single authoritative JSON schema (baseline, variantRaw, aggregated, deltas, browserRuns, fieldAudit, history, meta).
9. Educational test refactor (single expectation style) improving maintainability and signal clarity.
10. Variance mode scaffolding added (multi-sample repeats env hook) enabling future stable regression gates.

Observed Baseline Metrics (Representative Current Run):
- Bytes/conn plateau at 69 (1k) → 65 (10k–50k) → 64 (≥100k); slimming effect not yet reflected in heap bytes (object header dominated).
- Forward timing volatility mid-scale (e.g. 50k variable deltas) motivating variance repeats before enforcing gates.
- Heap deltas between src/dist within low single-digit %; dist currently fallback (no optimized build yet).

Technical Debt / Open Threads Carried Forward:
- True production `dist` build absent (variant currently mirrors src code path when build missing).
- Multi-sample variance not yet persisted in artifact (Phase 1 will activate with BENCH_REPEAT_LARGE>1 runs).
- Browser memory parity test still skipped pending threshold calibration.

Confidence & Readiness:
- Instrumentation stable; adding new metrics or slimming fields now low-risk with audit + history safeguards.
- Proceeding to Phase 1 field slimming with clear baseline and regression detection scaffolding.

### Phase 1 – Field Audit & Slimming
Files: `src/architecture/connection.ts`, `src/architecture/node.ts`
Actions:
1. (C) Identify rarely-used fields; gate behind feature flags / lazy getters (optimizer, plasticity, telemetry).
2. (C) Replace multiple booleans with `Uint8` bitfield (enabled, hasGater, dropMask, plastic, reserved bits).
3. (C) Ensure ids (`innovation`,`geneId`) constrained to 32-bit; prep typed packing mapping.
4. (C) Document canonical layout + rationale to discourage regressions.
5. (N) Order fields to ease future WASM struct bridging.
6. (B) Freeze prototypes after slimming to stabilize hidden classes.
Impact: 5–10% per-object reduction.

## Phase 1 – Kickoff Log (Field Audit & Slimming)

Initial Actions Completed:
1. Neutral `gain` virtualized (symbol storage only when !=1) reducing enumerable keys to 10.
2. Bitfield `_flags` (enabled + dcMask) retained; hasGater bit planned.
3. Variance repeat scaffolding present (await BENCH_REPEAT_LARGE run for CV metrics).

Immediate Next Steps:
1. Run benchmarks (BENCH_REPEAT_LARGE=5) to populate variance section & confirm Connection keys=10.
2. Implement hasGater flag + gater virtualization (null => no key) maintaining serialization parity.
3. Draft regression gate annotation (non-failing) using forward delta & CV threshold.
4. Script auto-generating delta & heap tables from artifact.
 5. Add Phase 1 tests that parse `test/benchmarks/benchmark.results.json` asserting:
  - `variance` entries exist for ≥100k sizes when BENCH_REPEAT_LARGE>1.
  - Connection fieldAudit count matches expected (<=10) and fails if it increases.
  - Forward CV% for sizes ≥100k is reported; warn (test annotate) if >15% once repeats>1.
  - Regression annotation logic (once added) emits metadata without failing suite until gate enabled.

Test File Integration Expectations:
- All new regression / variance tests MUST read the shared artifact rather than recomputing metrics ad‑hoc.
- Artifact-driven assertions ensure a single source of truth; avoid duplicating aggregation logic inside tests.
- When adding new metrics to the artifact, update this section and extend parsing tests accordingly.

Success Criteria (Phase 1):
- Connection own enumerable keys ≤10 (achieved) with tests & benchmarks green; optional further reduction after gater virtualization.
- Stable forward timing CV (≥100k sizes) under 5% with repeats.
- Documented canonical Connection layout & bit allocations in plan.

Monitoring Metrics to Add:
- Coefficient of variation (CV%) for buildMs & fwdAvgMs persisted under `variance` in artifact.
- Field audit delta commentary (auto) when key counts change.

Exit Criteria for Phase 1:
- Key count reduction achieved & documented.
- Variance-based regression gating logic merged (initially informational).
- Dist real build pipeline ready for Phase 2 pooling evaluation (optional but preferred for early signal).

### Phase 2 – Node Pooling & Governance
Files: `src/architecture/node.ts`, new `src/architecture/nodePool.ts`
Steps:
1. (C) Implement `NodePool` with acquire/release/reset mirroring connection pooling.
2. (C) `Network.releaseNode(node)` on prune events (Morphogenesis ready).
3. (N) Pre-warm pool based on planned growth (reduces first-epoch GC spikes).
4. (B) Adaptive trimming when tab hidden (Page Visibility) or memory pressure hint; reduce high-water mark.
5. (C) Tests: memory stable after repeated acquire/release cycles; state fully reset.
Risks: Constructor side-effect reliance. Mitigation: replicate in `reset` + tests.

### Phase 3 – Extended Slab Packing
Files: `src/architecture/network.slab.ts`, `src/architecture/network.ts`
Enhancements:
1. (C) Pack gains, mask/drop flags, enabled bits, plasticity meta into typed arrays (Float32/Uint8) alongside weights.
2. (C) Lazy materialize legacy connection objects only when requested and `connectionObjectMode=false`.
3. (C) Slab version counter; dev assertions on mismatch.
4. (N) Larger growth factor (1.75–2×) to reduce reallocation frequency.
5. (B) Smaller increments (1.25×); large copy operations chunked with cooperative yields (microtasks / `requestIdleCallback`).
6. (H) Provide direct slab mapping for adjacency→phenotype integration (avoids duplicate arrays).
Success: Reduced retained JS objects, forward parity vs object mode, stable or improved throughput.

### Phase 4 – Sparse Growth & Prune Budgets
Files: `network.prune.ts`, `network.ts`, new `network.sparsityBudget.ts`
Steps:
1. (C) Track connection count + global sparsity goal; future per-module (Hyper) extension.
2. (C/H) `ensureBudget()` before growth; triggers prune/regrow cycle if exceeding `maxConnections` (with `growthGraceFraction`).
3. (N) Integrate soft heap monitor vs `nodeHeapSoftLimitMB` for early pruning.
4. (B) Use `browserMemoryBudgetMB` soft target; preempt budget breaches proactively.
5. (C) Exponential backoff for repeated denied growth.
6. (C) Tests verifying cap adherence.
7. (H) Morphogenesis growth/prune uses same budget API (no duplicate logic).

### Phase 5 – Adaptive Precision & Mixed Precision
Files: `network.ts`, `node.ts`, `activationArrayPool.ts`
Actions:
1. (C) Consolidate precision flags into `PrecisionConfig` object.
2. (C) Float16 path (Uint16 storage; on-the-fly convert) inference-only; measure drift (<1e-4 median abs diff).
3. (N) Explore bfloat16 via WASM kernel (if `wasmKernels` enabled).
4. (B) Chunk large conversions; yield between blocks.
5. (C) Loss scaling with overflow/underflow counters auto-adjusting scale.
6. (H) Apply precision modes to large Hyper-generated adjacency/phenotype slabs.

### Phase 6 – Allocation Churn Reduction
Files: `activationArrayPool.ts`, `network.connect.ts`
Steps:
1. (C) Batch connection creation API (reserve capacity upfront per growth event).
2. (C) Extend activation pool with LRU trimming + compaction stats.
3. (C) Ring-buffer reuse for temporal sequences (`reuseSequenceBuffers`).
4. (N) Background compaction (idle tick) after large prune.
5. (B) Cooperative compaction (microtask slices / idle callbacks) to avoid jank.
6. (H) Morphogenesis growth bursts use batch API to cap reallocations.

### Phase 7 – Genotype / Phenotype & Adjacency Caching (Hyper Intensive)
Files: `hyper/phenotypeBuilder.ts`, `hyper/genotype.ts`, new `hyper/adjacencyCache.ts`
Steps:
1. (C/H) Hash genotype signature; reuse phenotype slabs (copy-on-write mutated sections) with ref counts.
2. (H) Adjacency cache key = (genotypeHash, substrateHash, cppnHash, threshold, precisionMode). Store SoA: src/dst Uint32, weight Float32|Uint16, flags Uint8, optional plasticity Float32.
3. (N) Optional disk persistence (binary blobs + JSON header) gated by size threshold.
4. (B) IndexedDB/OPFS chunk storage; async hydration & eviction statistics.
5. (C) LRU eviction with byte cap (`hyperAdjacencyCacheMaxBytes`).
6. (C) Tests: deterministic hits; eviction preserves correctness; stale pointer detection.
7. (H) Metrics: hit ratio, rebuild speed-up (>2× target), cache memory overhead per active connection, eviction cost (<5% build time).

### Phase 8 – Serialization Compression
Files: `network.serialize.ts`, `hyper/serialization.ts`
1. (C) Delta encode weights (int16 diffs) per slab sequence.
2. (C) RLE encode disabled/pruned spans & zero-weight runs.
3. (N) Optional gzip/zstd streaming compressor (pluggable) for large model storage.
4. (B) Optional WASM gzip (if available) or chunked compression to keep main thread responsive.
5. (C/H) Genotype-only compression path (rules + CPPN params + substrate modifiers) tracked separately (report ratio vs phenotype bytes).
6. (B) Streaming incremental decode with progress callbacks.
7. (C) Output metrics: compressed size, compression ratio, encode/decode time.

### Phase 9 – Streaming Activation Windows (Optional Advanced)
Files: `network.ts`, new `network.window.ts`
Use Cases: Deep recurrent nets, streaming sensor data, on-device low-memory inference.
1. (C) Circular buffer for activations length `windowSize`; deterministic equivalence tests vs full history for overlapping segments.
2. (B) Frame-sliced advancement (yield after configurable batch) to maintain UI responsiveness.
3. (N) Larger default window permitted (server memory) for gradient fidelity.
4. (C) API `forwardWindowed(inputs[])` + docs on trade-offs.
5. (H) Align morphogenesis events to window boundaries to stabilize temporal metrics for module focus scoring.

---

## Cross-Cutting Utilities
1. `memoryStats()` : counts & approximate bytes (connections, nodes, slabs, pools, caches) + active flag states + environment heuristics (browser only).
3. Flag definitions in `config.ts` with environment gating (auto-disable unsupported features; expose status via stats).
4. `adjacencyCacheStats()` (Hyper): entries, bytes, hit/miss, evictions, persistence bytes.
5. `precisionStats()` capturing current precision modes, overflow/underflow counters, scaling adjustments.
6. `poolHighWaterMarks()` for churn leak detection and morphogenesis stress tests.

---

## Risk Matrix (Expanded)
Risk | Impact | Env Bias | Mitigation
---- | ------ | -------- | ---------
Pool reset bug (stale gradients) | Incorrect training | Both | Comprehensive reset + tests
---


---

End of memory optimization plan (revised, environment-aware, Hyper-aligned).
Typed array vs object desync | Silent logic errors | Both | Version counter + dev asserts
Precision downcast loss | Accuracy degradation | Both | Opt-in + drift tests + fallback
Cache retention leak | Memory bloat | Node | Ref counts + weak maps + idle sweep
Adjacency cache blow-up | OOM / GC thrash | Browser | Byte cap + LRU + soft budget trigger
Plasticity buffer leak | Gradual creep | Both | Pool reset tests + leak harness
Morphogenesis fragmentation | Retained wasted bytes | Browser | Periodic compaction + metrics
Large slab copy jank | UI stalls | Browser | Chunked copies + cooperative scheduling
IndexedDB growth overflow | Quota errors | Browser | Size accounting + eviction threshold
Disk cache stale bloat | Disk waste | Node | TTL metadata + size pruning
WASM incompatibility | Crash / incorrect math | Both | Feature detection + test parity
SAB unavailable | Worker perf drop | Browser | Graceful downgrade to standard workers

---

## Test Strategy Additions (Environment Aware)
Category | Tests | Env
-------- | ----- | ---
Pooling | Acquire/release idempotence; stable memory after cycles | Both
Slab Packing | Forward outputs parity vs object mode (random seeds) | Both
Precision | f32 vs f64 drift <1e-5 median; f16 drift <1e-4 | Both
Budget Enforcement | Growth beyond cap triggers prune; respects soft budgets | Both
Compression | Serialize+deserialize equivalence; streaming decode | Both
Caching | Deterministic cache hits; eviction correctness; speed-up >2× | Both
Plasticity | Enable/disable buffers stable heap over N cycles | Both
Morph Churn | Repeated grow/prune stable high-water marks | Both
Scheduling | Frame-sliced activation no >5% frames >16ms | Browser
Persistence | Disk/IndexedDB eviction & size accounting | Node/Browser
WASM Kernels | Parity + performance improvement metrics | Feature-detected

---

## Benchmark Scenarios (Initial Set)
1. Build-only: construct 100k sparse connections (Browser variant 50k if limits encountered).
2. Forward pass: batch=1, 3 hidden layers, 500 iterations (Browser with optional frame-sliced mode).
3. Evolution cycle: mutation + prune + rebuild for population N=50 (Hyper off baseline).
4. Morphogenesis loop: growth + prune every 100 steps (Hyper later phases).
5. Adjacency cache stress: alternating thresholds + partial genotype mutations (eviction path).
6. Phenotype cache reuse: unchanged genotype vs mutated control; measure rebuild time ratio.
7. Precision switch: f32 vs f16 memory + accuracy drift.
8. Streaming window: long sequence forward vs windowed (Phase 9).
9. Browser UI responsiveness: % frames >16ms during large growth events.

Metrics: wall time, RSS / usedJSHeapSize, GC events, bytes/connection, slab fragmentation %, cache hit ratio, frame overrun %, compression ratio, copy bandwidth (MB/s), heuristic error factor (Browser).

---

## Incremental Adoption Flags (config.ts)
See extended flag table (not yet implemented). Add environment gating & expose flag activation status in `memoryStats().flags`.

---

## Documentation & Developer Guidance
1. Update README performance section after Phase 3 (slabs) & Phase 5 (precision) with environment notes.
2. Migration notes: disabling object mode & enabling slab packing safely, feature compatibility matrix.
3. Diagrams (ASCII → SVG) for slab layout, cache layering (adjacency / phenotype), environment flows.
4. Troubleshooting guide: leak detection, reading stats, tuning budgets, Browser responsiveness tips.
5. Hyper docs cross-link memory flags required for adjacency/phenotype caching.

---

## Baseline & Progress Logging
Append after each phase:
```
### Phase X Results
Date:
Environment(s): Node vX / Browser (Chrome Y, FF Z ...) 
Changes:
Metrics (Core): bytes/conn, peak heap, GC events, cache hit ratio, fragmentation %
Metrics (Env): frame jank %, disk/IndexedDB bytes, heuristic error factor, compression ratio
Notes:
```

---

## Coordination With Hyper MorphoNEAT Plan (Expanded)
Dependency Mapping:
Hyper Phase -> Memory Requirement -> Memory Phase
1–2 -> Deterministic rebuild stable -> Phase 0–1
3 (CPPN) -> Fast edge generation & caching -> Phase 3,7
4 (Morphogenesis) -> Efficient growth/prune -> Phase 2,4
5 (Plasticity) -> Extra per-conn state packing -> Phase 3 extension
8 (Scale validation) -> Bench infra -> Phase 0,6,7

Extended Mapping (Granular Alignment, Environment nuance):
Hyper Phase | Memory Concern | Memory Layer / Phase | Notes
----------- | -------------- | -------------------- | -----
0 (Scaffolding) | Flag isolation & zero overhead | Flags + baseline instrumentation (Phase 0) | Ensure feature off path identical
1 (Genotype/Substrate) | Genotype size vs phenotype ratio | Metrics extension (Target + Additional) | Track bytes/genotype
2 (Rule Engine) | Deterministic expansion cost | Phase 1 slimming + profiling harness | Avoid premature allocations
3 (CPPN Indirect) | Adjacency generation & cache | L7 (adjacency cache) + Phase 7 | Hit ratio & byte cap
4 (Morphogenesis) | Churn & budget enforcement | Phase 2,4 + churn tests | Monitor pool high-water marks
5 (Plasticity) | Side buffer footprint | Phase 3 packing + new flag | Optional typed arrays only
6 (Telemetry) | Lazy metrics buffers | Cross-cutting utilities | Zero retained when disabled
7 (Evolution Integration) | Multi-offspring rebuild reuse | Phenotype cache (L7) | Minimize rebuild duplicates
8 (Scale Validation) | Peak memory, rebuild variance | Benchmark suite & CI thresholds | Pass/fail gating

---

## Future (Post Core) Environment-Specific Explorations
1. Node: Memory-mapped slab snapshots for near O(1) reload.
2. Node: N-API / WASM SIMD kernels for fused ops.
3. Browser: WebGPU compute for activations & weight updates.
4. Browser: OPFS streaming model hydration.
5. Both: Adaptive background compaction scheduling informed by allocation telemetry.
6. Both: Visual heatmap (dev tool) of module memory share.
7. Browser: Service Worker cached compressed genotype bundles.
8. Node: Incremental disk-backed adjacency tiers (multi-threshold progressive decoding).
9. Both: Probabilistic slab compaction scheduling (idle heuristic).

---

Note:

```
Testing requirements:
- all tests should have a single expectation.
- follow AAA pattern (arrange, act, assert) 
- group tests into scenarios with describe(), nest scenarios as needed, no limit on layers.
- when possible, define common testing data directly on the describe() and then write the assertions for it, this also applies for nested scenarios as they each represent more specific cases as it goes down into sub branches.
- aim for 100% testing coverage
- make sure to check existing files before creating/updating one, to be sure you are using the right file, in the right folder, for example `test/neat/` and also to be following the same file pattern inside that folder.

describe(() => {
  describe(() => {
    it('should...');
  });
});

Also:
- Always add JSDocs to all methods, classes, const, let
- Add or update inline comments within methods to explain each step or detail.
- This is an educative NN library, keep the docs detailed and educative
```
