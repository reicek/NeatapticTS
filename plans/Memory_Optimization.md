# Memory Optimization Plan (Multi-Layer Strategy for Very Large Networks)

**Status:** [WIP]

Goal: Enable construction, evolution, and training of networks scaling toward **10^6+ (stretch 10^7)** connections on commodity JS/TS runtimes across **Node** and **Browser**.

## Guiding Principles

1. Pay-for-use: No overhead unless feature enabled (flags + lazy allocation).
2. Reuse & Pool: Prefer object pooling / typed array slabs over many small objects (already partially implemented for `Connection`).
3. Structural Sparsity First: Keep graphs sparse; optimize dense fast-paths only when needed.
4. Incremental & Measurable: Each optimization introduces a benchmark & memory snapshot (Node + Browser where feasible).
5. Branch Containment: Refactors isolated (one concern per PR) with compatibility shims where needed.
6. Environment Awareness: Node path may adopt heavier instrumentation & persistent caches; Browser path emphasizes chunking, responsiveness, and quota safety.
7. Hyper Alignment: Memory layers anticipate HyperEvoDevo MorphoNEAT phases (indirect generation, morphogenesis churn, phenotype/adjacency caches).
8. Transparent API: All environment-specific memory optimizations (Node vs Browser, slabs, pooling, precision) remain behind a stable public memory/network API; callers never branch on environment—feature flags + `memoryStats()` abstraction handle differences.
9. Performance Trade-off Management: Acknowledge that initial memory optimizations (like instrumentation and slab packing) may introduce temporary performance overhead. The plan must track these trade-offs and ensure that subsequent optimizations (e.g., caching, sparsity) deliver a net performance gain at scale.

> Build Baseline Update (Q3 2025): TypeScript/webpack target pinned to **ES2023** (previously floating `ESNext`) to ensure reproducible emitted output and educational clarity. Modern features beyond ES2023 may still appear behind feature flags with graceful degradation or transpilation guidance.

> Dist-Only Benchmark Pivot (Q3 2025): The benchmarking infrastructure no longer performs src vs dist variant comparisons. Only the built `dist` bundle metrics are recorded. Historical entries with dual-mode deltas persist in the rolling history until aged out (≤10 snapshots). This reduces noise and maintenance; future re-introduction of a second variant will only occur if a materially different optimized production build (treeshaken/minified) diverges behaviorally or structurally from the reference `dist` artifact.

---

## Environment Constraints & Differentiators

| Dimension             | Node (V8 server/CLI)                                              | Browser (varied UAs)                                                                              |
| --------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Typical Heap Ceiling  | ~2–4 GB (tunable via `--max-old-space-size`)                      | Practically 1–2 GB (lower on mobile) per tab                                                      |
| Introspection APIs    | `process.memoryUsage()`, `v8.getHeapStatistics()`, heap snapshots | `performance.memory` (Chrome), experimental `measureUserAgentSpecificMemory()`, internal counters |
| Background Threads    | `worker_threads`, SharedArrayBuffer, Atomics                      | Web Workers, (SAB if cross-origin isolated)                                                       |
| Persistence           | FS, memory-mapped files, streams                                  | IndexedDB / OPFS, Service Worker cache                                                            |
| Large Buffer Handling | `Buffer` (zero-copy slices), streams                              | `ArrayBuffer` / transferable objects (clone cost relevant)                                        |
| Scheduling            | Long tasks tolerable (batch/CLI)                                  | Must avoid frame jank (>16ms @60fps)                                                              |
| Acceleration          | N-API, native libs, WASM SIMD                                     | WASM SIMD, WebGPU (emerging)                                                                      |

Implications: Browser emphasizes **cooperative scheduling, micro-chunk allocation, transfer minimization, heuristic memory guards**. Node emphasizes **larger batch allocations, deeper instrumentation, optional persistence / native acceleration**.

---

## Target Metrics (Refine After Baseline Measurement)

| Metric                                        | Baseline (Post-Phase 0)  | Target Phase | Goal                              |
| --------------------------------------------- | ------------------------ | ------------ | --------------------------------- |
| Avg bytes / active connection                 | ~64-69                   | 5            | -25% vs baseline (via sparsity)   |
| Peak heap growth per 100k added connections   | TBD (measure in Phase 5) | 6            | < 12 MB                           |
| GC pause impact (allocation-heavy evolutions) | TBD (measure in Phase 7) | 7            | -30% vs baseline                  |
| Phenotype rebuild allocation churn            | TBD (measure in Phase 7) | 7            | amortized O(1) per edge via reuse |

Note on Bytes/Connection: The baseline of ~64 bytes/connection reflects the overhead of individual JavaScript objects. The slab packing in Phase 3 was designed to change the _memory layout_ to enable future gains but did not reduce the _data payload_ itself, hence the flat metric. The targeted 25% reduction is contingent on **Phase 5 (Sparsity)**, where connections are selectively pruned, directly reducing the average memory cost across the network.

### Additional Hyper-Specific Metrics (Introduced Once Hyper Phases Land)

| Metric                                                   | Baseline (Post Phase 11 Hyper) | Target Phase | Goal                                                   |
| -------------------------------------------------------- | ------------------------------ | ------------ | ------------------------------------------------------ |
| Bytes / genotype (symbolic)                              | measure                        | 16           | < 0.5% of phenotype bytes at scale (document)          |
| Adjacency cache hit ratio                                | measure                        | 15           | > 70% on repeated evaluations (bench harness scenario) |
| Adjacency cache bytes / active connection                | measure                        | 15           | < +6 bytes (amortized) with cap                        |
| Plasticity side-buffer bytes / active plastic connection | measure                        | 14           | < 8 bytes (float32 rate + accumulator)                 |
| Morphogenesis churn leak slope (pool high-water mark)    | measure                        | 12           | ~0 over final 20% iterations                           |
| Rebuild time variance (p95 / median)                     | measure                        | 16           | < 2.5× ratio (determinism stability)                   |
| Cache eviction overhead (per miss)                       | measure                        | 15           | < 5% of total build time in stress test                |

Baseline numbers gathered in Phase 0 using synthetic builder + Node `--inspect` snapshots (authoritative). Browser adds heuristic baseline using internal slab + pool accounting cross-checked with `performance.memory.usedJSHeapSize` (where available). Target heuristic error tolerance <15%.

Future mobile tier baseline (optional) for adaptive budgets.

---

## Optimization Layers Overview (Environment-Aware)

| Layer | Focus                         | Common Techniques                       | Node Emphasis                     | Browser Emphasis                                | Hyper Relevance                   |
| ----- | ----------------------------- | --------------------------------------- | --------------------------------- | ----------------------------------------------- | --------------------------------- |
| L1    | Data structure slimming       | Field layout audit, bitfields           | Alignment for future WASM structs | Stable hidden classes                           | Pre-req (0–1)                     |
| L2    | Object pooling & reuse        | Pools for connections/nodes/activations | Larger pre-warm, background fill  | Adaptive trimming on visibility/memory pressure | Morph churn (4)                   |
| L3    | Slab / SoA packing            | Weights/gains/flags in typed arrays     | Larger geometric growth (1.75–2×) | Smaller growth steps (1.25×) with yields        | Fast rebuild (3,7)                |
| L4    | Sparse & budget control       | Coordinate lists, prune-regrow          | Bulk generation batches           | Incremental growth microtasks                   | Morphogenesis governance (4)      |
| L5    | Adaptive precision            | f64→f32; optional float16               | bfloat16 (WASM) exploration       | f16 gating + loss scaling                       | Large phenotypes (5)              |
| L6    | Streaming & chunking          | Windowed activations                    | Worker_threads pipelines          | Frame-sliced forward passes                     | Temporal scaling (9)              |
| L7    | Caching (adjacency/phenotype) | Hash → SoA reuse                        | Disk (FS) persistence             | IndexedDB/OPFS shards                           | Hyper caches (7)                  |
| L8    | Compression / serialization   | Delta + RLE                             | Gzip/zstd streaming               | Incremental decode/gzip (WASM)                  | Shipping phenotypes/genotypes (8) |
| L9    | Scheduling & telemetry        | Unified stats APIs                      | High-res timers + deep snapshots  | Cooperative scheduling + jank metrics           | Deterministic rebuild windows     |

L7 subdivisions: (a) adjacency cache (structural + CPPN + threshold + precision hash) and (b) phenotype slab reuse (ref-counted typed arrays). Node may persist cache entries to disk; Browser may hydrate from IndexedDB/OPFS with async chunk loading.

---

## Phase Plan (Concrete Steps & Affected Files)

Each phase notes: (C) Common, (N) Node-specific, (B) Browser-specific, (H) Hyper alignment.

Execution status summary (normalized numbering):

- Completed: Phases 0, 1, 2, 3
- Next Up: Phase 4 (Centralized Memory Management & Browser Validation)
- Planned (Track 1): Phases 5–10
- Planned (Track 2): Phases 11–16

### Two-Track Execution Model (Numbered)

Track 1 — **Core Library Foundation (Implementation First)**

- Phases **0–10**
- Purpose: deliver memory/perf infrastructure that benefits the current library regardless of Hyper adoption.
- Status: 0–3 complete; 4 next; 5–10 planned.

Track 2 — **HyperEvoDevo MorphoNEAT Algorithm Integration**

- Phases **11–16**
- Purpose: add evo-devo algorithmic capabilities after core memory infrastructure is in place.
- Sequence rule: Track 2 starts only after Track 1 implementation gates are met.

Track 1 → Track 2 gate (must pass all):

1. Phase 4 completed with manager-backed stats parity (Node + Browser).
2. Phase 5 benchmark confirms measurable bytes/connection reduction trend.
3. Phase 6 precision-path parity and regression checks pass.
4. Phase 7 churn checks and Phase 10 variance/hardening gates are stable.
5. Baseline/progress artifacts are updated for reproducibility and rollback.

### Phase 0 – Baseline Instrumentation (Condensed Summary) [Done]

Purpose: Establish reproducible dist‑only performance & memory baseline plus variance framework to support later optimizations.

Key Deliverables:

- Dist-only benchmark artifact (JSON): baseline, variantRaw (dist raw), aggregated, variance, history (≤10), meta, fieldAudit, warnings.
- Deterministic seeding, warm‑up discard, adaptive forward iteration counts, IQR (1.5) outlier filtering.
- Field audit (enumerable keys) tracking structural slimming impact.
- Bytes/connection, heapUsed, rss collection; browser harness prototype (dev/prod) for future parity checks.
- Regression annotation scaffolding (informational only; gated on CV thresholds).
- Connection slimming groundwork started (see Phase 1).

Representative Final Baseline (pre-pooling/slab) (dist snapshot – earlier Phase 0 run):

| Size | buildMsMean | fwdAvgMsMean | bytesPerConnMean |
| ---- | ----------- | ------------ | ---------------- |
| 1k   | ~2          | ~0.4         | 69               |
| 10k  | ~8–12       | ~3–4         | 65               |
| 50k  | ~40–46      | ~4–7         | 65               |
| 100k | ~63–70      | ~9–13        | 64               |
| 200k | ~149–170    | ~19–26       | 64               |

Outcomes:

- Plateau bytes/conn ≈64–69 (object overhead dominates; validates need for pooling/slab path).
- Variance still > target at larger scales → auto escalation deferred to Phase 2.
- Artifact schema stable; foundation ready for pooling, slab packing and future gating.

Carry-Over to Phase 2:

- Variance auto escalation (adaptive repeats).
- Browser parity memory test (defer until CV stabilized).
- Optional production optimized build (only if a materially different bundle path is introduced).

### Phase 1 – Field Audit & Slimming [Done]

Purpose: Reduce per-object overhead & lock in structural introspection before deeper memory model changes.

Slimming Actions:

- Connection virtualization: neutral gain & gater removed from enumerable set.
- Bitfield `_flags` (enabled, dropConnect mask, hasGater) reduced Connection enumerable keys to 9 (target maintained).
- Canonical layout documented (guards hidden class stability & future WASM alignment).

Instrumentation Enhancements:

- Regression annotation (informational timing deltas).
- History retention policy (≤10 snapshots) with provenance (dist bundle bytes + sha256 prefix).
- Variance tracking (CV%) persisted; gating logic deferred until stability.

Final Phase 1 Metrics (dist snapshot):
| Size | buildMsMean | fwdAvgMsMean | bytesPerConnMean | Samples |
|------|-------------|--------------|------------------|---------|
| 1k | 1.99 | 0.37 | 69 | 1 |
| 10k | 7.95 | 3.64 | 65 | 1 |
| 50k | 45.40 | 3.91 | 65 | 1 |
| 100k | 62.89 | 9.71 | 64 | 7 |
| 200k | 149.08 | 19.66 | 64 | 7 |

Achievements:

- Connection enumerable keys ≤9 (goal met).
- Deterministic reproducibility baseline in place.
- NodePool skeleton prepared (no network integration yet at end of Phase 1).
- Bytes/connection stable (pre-slab reference).

Deferred / Hand-off to Phase 2:

- Variance stabilization (<7% CV at 100k & 200k).
- Further slimming (node error SoA) scheduled for Phase 3 (slab introduction).
- Enforcement gate (fail on Connection key regression) postponed until post-pooling variance stabilization.

### Phase 2 – Node Pooling & Governance [Done]

Completion Summary:

- Pool integration (construction, addNodeBetween, remove()) behind enableNodePooling flag.
- Release on remove with defensive swallow.
- memoryStats() exposes pools.nodePool.
- Artifact persistence: meta.poolStats.nodePool (size, highWaterMark, reused, fresh, recycledRatio).
- Determinism parity (pool ON/OFF) passing.
- Stress harness stable synthetic acquire/release cycles; recycledRatio ≥0.6; tail highWaterMark Δ ≤2.
- Variance auto escalation ACTIVE: benchmark harness now escalates repeats up to cap (9) until CV ≤7% or cap reached; escalation events recorded in meta.varianceAutoEscalations (action=escalate|stop with reason cv-above-threshold|below-threshold|max-repeats).
- Phase 2 exit criteria fully met (no deferred core items).

Final Stress Metrics (representative run):
| recycledRatio | highWaterMarkTailΔ | Thresholds | Status |
|---------------|--------------------|------------|--------|
| ~0.66 | 0 | ≥0.5 / ≤2 | PASS |

Notes:

- Active variance stabilization provides statistically reliable baseline ahead of Phase 3 slab packing comparisons.
- Next tasks shift entirely to slab packing + bytes/conn reductions without needing to revisit repeat governance.

Proceeding Next: Phase 3 – slab packing prototype with stable pooling + variance foundations.

Achievements (Quantitative Metrics):

- Large-size variance escalation engaged both monitored sizes to cap (max repeats=9) due to CV above 7% target; escalation trail recorded (3 events per size: 2 escalate + 1 stop each).
- Post‑escalation variance (cap reached – still above target, informing Phase 3 optimization focus):
  - 100k: build CV 9.53%, forward CV 21.13% (samples=9).
  - 200k: build CV 11.79%, forward CV 23.11% (samples=9).
- Build throughput (dist, escalated means): 100k buildMsMean ≈82.54ms; 200k ≈172.39ms.
- Forward average latency (dist, escalated means): 100k fwdAvgMsMean ≈15.98ms; 200k ≈24.07ms.
- Bytes per connection plateau preserved at 64 (reference pre‑slab target baseline maintained; no regression introduced by pooling/escalation harness adjustments).
- NodePool instrumentation integrated into artifact meta (current run showed no retained pooled nodes after benchmark scenario: size=0, highWaterMark=0, reused=0, fresh=0, recycledRatio=0 because benchmark does not exercise growth+prune cycles beyond synthetic stress tests run separately).
- Escalation governance now automated (no manual reruns required); meta fields: maxVarianceRepeats=9, varianceAutoEscalations[6 records].
- Determinism parity tests pass with pooling enabled (forward outputs identical to non‑pooled path under seeded RNG).

Interpretation & Next Focus:

- Elevated forward CV ( >20%) at large sizes suggests remaining noise sources (allocation jitter, warm-cache effects) that slab packing + reduced object churn should attenuate.
- Maintaining bytes/conn plateau while adding governance instrumentation validates pay‑for‑use principle (no incidental bloat).
- Pool reuse efficiency metrics will become meaningful once Phase 3 introduces slab-backed connection packing and more aggressive mutation/prune cycles; present zeroed stats serve as baseline.

### Phase 3 – Extended Slab Packing [Done]

Scope Files: `src/architecture/network.slab.ts`, `src/architecture/network.ts`

Objectives:

- Pack connection data into Structure‑of‑Arrays slabs (weights, flags, optional gains, optional plasticity) with geometric growth & pooling.
- Preserve forward correctness (parity) and enable a fast CSR activation path with pay‑for‑use optional slabs.
- Instrument memory (fragmentation, pooled reuse, alloc stats) without inflating bytes/connection when features unused.

Implemented Features (consolidated):

1. Geometric capacity growth (Node 1.75×, Browser 1.25×) with slab reuse and fragmentation accounting.
2. Bit‑packed connection flags (enabled, dropConnect mask, gater, plastic) in a `Uint8Array` slab.
3. Optional gain slab (`_connGain`) allocated only on first non‑neutral gain; released when all revert to 1.
4. Optional plasticity slab (`_connPlastic`) allocated only when any connection has `plasticityRate>0`; released when none remain.
5. Slab version counter and exposure via `getConnectionSlab()`; parity asserts during development.
6. Async browser rebuild with adaptive micro‑chunking (target ms heuristic) sharing pooling logic; metrics: `asyncBuilds`.
7. Fast slab forward path (CSR) including gain multiplication; gating guard auto‑fallback when gating present.
8. Extended instrumentation: fragmentationPct, pooledFraction, allocStats per typed array key, reservedBytes vs usedBytes, bytesPerConn stability snapshot.
9. TypedArray pooling (small per‑key LRU) tracking created/reused/maxRetained for educational analysis.

Test Coverage (single expectation style): capacity growth, versioning, flags parity, async rebuild (browser), fast path parity & gating guard, plasticity pay‑for‑use, gain omission & release, gain parity, fragmentation trend + bounds, allocation reuse stats.

Deferred (post‑Phase 3): chunked copy yield refinements (browser large slabs), adjacency→phenotype direct mapping (moved to caching phase), forward variance reduction (targeted in later phases once churn reduced).

### Phase 3 Results (Final Validation Before Later Track 1 Phases)

Source: `benchmarks/benchmark.results.json` (latest history entry vs earliest recorded baseline in same file).

| Size | Baseline Build ms (mean) | Phase 3 Build ms (mean) | Δ Build % | Baseline Fwd Avg ms | Phase 3 Fwd Avg ms | Δ Fwd % | Bytes/Conn Baseline | Bytes/Conn Phase 3 | Δ Bytes/Conn |
| ---- | ------------------------ | ----------------------- | --------: | ------------------- | ------------------ | ------: | ------------------- | ------------------ | -----------: |
| 1k   | 2.0415                   | 1.9435                  |     -4.8% | 0.4045              | 0.6797             |  +68%\* | 69                  | 69                 |            0 |
| 10k  | 7.0529                   | 9.8183                  |    +39.3% | 3.9449              | 4.5383             |  +15.0% | 65                  | 65                 |            0 |
| 50k  | 46.2062                  | 48.6405                 |     +5.3% | 6.2795              | 7.4799             |  +19.1% | 65                  | 65                 |            0 |
| 100k | 60.1772                  | 77.2990                 |    +28.4% | 9.2725              | 13.6824            |  +47.6% | 64                  | 64                 |            0 |
| 200k | 153.4771                 | 189.3520                |    +23.4% | 22.2546             | 29.4639            |  +32.4% | 64                  | 64                 |            0 |

The 1k forward pass time shows a significant increase (+68%). While this may be related to test overhead on very small networks or initial cache warm-up effects, the magnitude warrants further investigation in Phase 7 (Allocation Churn Reduction) to rule out any underlying inefficiencies in the slab implementation at small scales. The primary Phase 3 memory objective—stabilizing bytes/connection while adding features—was met.

Variance (CV%) snapshot (from `variance` section):
| Size | Build CV% | Fwd CV% | Target CV% |
|------|-----------|---------|------------|
| 100k | 4.7 | 11.41 | 7 (build met, forward above) |
| 200k | 11.34 | 13.13 | 7 (above) |

Notes:

- Phase 3 prioritized memory layout & feature packing (plasticity bit + optional slabs, gain omission) over variance suppression; forward CV remains elevated at high scales—scheduled for attention in Phase 7 (allocation churn) and Phase 10 (foundation hardening gates) where rebuild quality is re-validated.
- Bytes/connection held constant (no regression) despite added optional slabs; gain omission and plasticity pay‑for‑use prevented per-connection inflation. This confirms the pay-for-use principle is working but highlights that progress on the bytes/connection reduction metric is dependent on Phase 5 (Sparsity).
- Field audit counts: Connection enumerable keys = 9 (stable), Node = 15 (stable) per benchmark `fieldAudit` confirmation.

Results Notes & Next Step: For quantitative deltas see table above; variance & invariant consolidation detailed in Phase 3 Conclusion below. Proceed to Phase 4 centralized memory management and browser-validation work with a stable slab foundation. Track 2 Hyper work remains gated behind the Track 1 conditions defined earlier in this document.

#### Phase 3 Conclusion (Extended Slab Packing & Validation)

All planned Phase 3 memory layout features are implemented, documented, and validated by tests; the slab system now provides a pay-for-use foundation for later Track 1 caching work and eventual HyperEvoDevo MorphoNEAT phases once the Track 1 gate is satisfied.

Delivered Enhancements (Recap):

1. Optional slabs (gain, plasticity) allocated lazily, released when neutral / absent.
2. Bit‑packed connection flags with added plastic bit.
3. Geometric capacity growth with pooling + fragmentation tracking.
4. Async cooperative rebuild (browser path) with adaptive chunk sizing + metrics (`asyncBuilds`, `slabVersion`).
5. Fast slab forward path with gain multiplication and gating guard fallback.
6. Extended `memoryStats()` (fragmentationPct, pooledFraction, allocStats per typed array key, bytesPerConn stability view).
7. Plasticity packing (bit + optional slab) and gain omission optimization (ensures zero overhead when unused).

Verified Invariants (Backed by Tests):

1. Parity: Fast vs legacy activation (baseline, with gating guard fallback, with non‑neutral gains).
2. Optional gain slab: allocated on first non‑neutral gain; released when all revert to 1.
3. Optional plasticity slab: allocated only when any connection plastic; released when none plastic.
4. Slab version increments on structural rebuilds (sync & async).
5. Fragmentation metrics: trend test plus bounds test guarantee 0 ≤ fragmentationPct ≤ 100 across churn.
6. Pooling reuse: repeated rebuilds without growth show non‑decreasing pooled allocation counter.
7. Gating guard correctness: fast path defers when gating present to avoid incorrect math.
8. Bytes/connection: stable plateau (≈64–69) despite added optional features (no unconditional inflation).
9. Gain parity: fast path multiplies weight\*gain if gain slab present.

New / Final Validation Concepts Tested in Phase 3:

- Fast path gain multiplication parity.
- Slab fragmentation bounds (0-100%).
- Allocation statistics and reuse tracking.
- Release of optional gain slab upon reset.

Risk & Deferred Items:

- Forward pass variance (> target at large sizes) deferred to Phase 7/10 (allocation churn + hardening gates).
- Adjacency → phenotype direct mapping postponed to caching phase to avoid premature duplication work.
- Potential future packing of plasticity side parameters (rates/traces) once learning rules land.

Exit Criteria (Met):

- Pay‑for‑use: Optional slabs only when demanded; released afterward.
- Fast path correctness across gain & plastic scenarios verified.
- Fragmentation metric correctness (bounded & trend tested).
- Pooling instrumentation exposes measurable reuse (alloc stats).
- Documentation contains quantitative benchmark comparison and enumerated achievements.

Ready for Next Phase: HyperEvoDevo MorphoNEAT work can proceed atop a stable, instrumented slab foundation with confidence in memory invariants and optional feature overhead discipline.

### Phase 4 – Centralized Memory Management & Browser Validation [Next]

**Status: Next Up**

This phase addresses a critical architectural gap by implementing a `Centralized Memory Manager` as the single source of truth for all memory-related operations. It also establishes a formal browser benchmark harness to validate environment-specific features.

#### Part 1: Centralized Memory Manager

To ensure consistent behavior, feature gating, and pay‑for‑use semantics, this module will be the canonical contract between the runtime, benchmarking harness, and platform-specific adaptations.

**Scope Files:** `src/memory/config.ts`, `src/memory/manager.ts`, `test/memory/manager.test.ts`

**Responsibilities & Design:**

- **Centralize Flags and Defaults:**
  - Global feature flags (`enableSlabs`, `enablePooling`, `precisionMode`, `browserAsyncRebuild`).
  - Typed-array growth factors and alignment preferences (`nodeFactor`, `browserFactor`, `baseCapacity`).
  - Pool defaults (`maxRetainedPerKey`, `initialReserve`).
- **Provide Pay-for-Use Primitives:**
  - Lazy allocation helpers for optional slabs (e.g., gain, plasticity).
  - Typed-array allocator wrappers that consult pooling and growth configurations.
  - A registration API for pools to enable centralized instrumentation and controlled teardown.
- **Expose a Stable `MemoryManager` API:**
  - `getConfig()`: Snapshot of current flags & growth factors.
  - `setFlag(name, value)`: Controlled mutation with validation.
  - `allocateTypedArray(type, length)`: Pooled/aligned allocation.
  - `releaseTypedArray(obj)`: Releases back to the appropriate pool.
  - `registerPool(name, poolDescriptor)`: Instrumentation handle.
  - `memoryStats()`: Populated from manager internals.
  - `init()/teardown()`: Controlled lifecycle for tests.
- **Environment Awareness & Gating:**
  - Automatic platform defaults (Node vs. Browser) chosen at initialization.
  - Feature flags default to `false`; enabling must be explicit.

**Key Deliverables:**

1.  **`src/memory/config.ts`**: Will export `const` values for all flags and numeric defaults with detailed JSDoc.
2.  **`src/memory/manager.ts`**: Will export a `class MemoryManager` and a `defaultMemoryManager` singleton instance.
3.  **Refactoring**: Existing code in `network.ts` and `network.slab.ts` will be updated to use the `defaultMemoryManager`.
4.  **Unit Tests**: Tests will be added to assert `getConfig()`/`setFlag()` behavior and verify that `allocate`/`release` operations respect the pay-for-use principle when flags are toggled.

#### Part 2: Browser Benchmark Harness

**Scope Files:** `bench-browser/harness.ts`

**Key Deliverables:**

1.  A dedicated test harness in `bench-browser/` to capture the same core metrics as the Node.js benchmarks (build time, forward pass time, memory usage via `performance.memory`).
2.  Specific tests to validate browser-only features, such as measuring UI thread blocking during `asyncBuilds` to confirm cooperative scheduling is effective.
3.  The plan will be updated to document the browser harness setup and results.

**Status:** Under development. Initial results will be added here once the harness is operational.

### Phase 5 – Sparse Growth & Prune Budgets [Planned]

**Status: Planned (after Phase 4)**

Files: `network.prune.ts`, `network.ts`, new `network.sparsityBudget.ts`
Steps:

1. (C) Track connection count + global sparsity goal; future per-module (Hyper) extension.
2. (C/H) `ensureBudget()` before growth; triggers prune/regrow cycle if exceeding `maxConnections` (with `growthGraceFraction`).
3. (N) Integrate soft heap monitor vs `nodeHeapSoftLimitMB` for early pruning.
4. (B) Use `browserMemoryBudgetMB` soft target; preempt budget breaches proactively.
5. (C) Exponential backoff for repeated denied growth.
6. (C) Tests verifying cap adherence and budget-driven pruning.
7. (C) **Validation:** Add a benchmark scenario to measure and report the `bytes/connection` metric before and after pruning, verifying progress toward the -25% target.
8. (H) While some hooks align with Morphogenesis, the core sparsity and budget logic is independent and critical for the `-25% bytes/connection` target. This work will proceed in parallel.

### Phase 6 – Adaptive Precision & Mixed Precision [Planned]

Files: `network.ts`, `node.ts`, `activationArrayPool.ts`
Actions:

1. (C) Consolidate precision flags into `PrecisionConfig` object.
2. (C) Float16 path (Uint16 storage; on-the-fly convert) inference-only; measure drift (<1e-4 median abs diff).
3. (N) Explore bfloat16 via WASM kernel (if `wasmKernels` enabled).
4. (B) Chunk large conversions; yield between blocks.
5. (C) Loss scaling with overflow/underflow counters auto-adjusting scale.
6. (H) Apply precision modes to large Hyper-generated adjacency/phenotype slabs.

### Phase 7 – Allocation Churn Reduction [Planned]

Files: `activationArrayPool.ts`, `network.connect.ts`
Steps:

1. (C) Batch connection creation API (reserve capacity upfront per growth event).
2. (C) Extend activation pool with LRU trimming + compaction stats.
3. (C) Ring-buffer reuse for temporal sequences (`reuseSequenceBuffers`).
4. (N) Background compaction (idle tick) after large prune.
5. (B) Cooperative compaction (microtask slices / idle callbacks) to avoid jank.
6. (H) Morphogenesis growth bursts use batch API to cap reallocations.

### Phase 8 – Serialization Compression [Planned]

Files: `network.serialize.ts`, `hyper/serialization.ts`

1. (C) Delta encode weights (int16 diffs) per slab sequence.
2. (C) RLE encode disabled/pruned spans & zero-weight runs.
3. (N) Optional gzip/zstd streaming compressor (pluggable) for large model storage.
4. (B) Optional WASM gzip (if available) or chunked compression to keep main thread responsive.
5. (C/H) Genotype-only compression path (rules + CPPN params + substrate modifiers) tracked separately (report ratio vs phenotype bytes).
6. (B) Streaming incremental decode with progress callbacks.
7. (C) Output metrics: compressed size, compression ratio, encode/decode time.

### Phase 9 – Streaming Activation Windows (Optional Advanced) [Planned]

Files: `network.ts`, new `network.window.ts`
Use Cases: Deep recurrent nets, streaming sensor data, on-device low-memory inference.

1. (C) Circular buffer for activations length `windowSize`; deterministic equivalence tests vs full history for overlapping segments.
2. (B) Frame-sliced advancement (yield after configurable batch) to maintain UI responsiveness.
3. (N) Larger default window permitted (server memory) for gradient fidelity.
4. (C) API `forwardWindowed(inputs[])` + docs on trade-offs.
5. (H) Align morphogenesis events to window boundaries to stabilize temporal metrics for module focus scoring.

### Phase 10 – Foundation Hardening & Release Gates [Planned]

Files: `benchmarks/*`, CI/workflow configs, memory telemetry integration points

1. (C) Consolidate pass/fail gates for variance, memory regression, and determinism replay.
2. (C) Enforce stable artifact snapshots for rollback and audit (Node + Browser where feasible).
3. (C) Finalize implementation-track docs for feature flags, migration, and troubleshooting.
4. (C) Validate Track 1 → Track 2 gates before enabling Hyper algorithm phases.

### Phase 11 – Hyper Scaffold & DNA Baseline [Planned]

Files: `hyper/genotype.ts`, `hyper/phenotypeBuilder.ts`, `hyper/config.ts`
Steps:

1. (H) Introduce HyperDNA scaffold with versioned shape and deterministic seed policy hooks.
2. (H) Add deterministic build-order contracts (module IDs, rule pass order, edge realization order).
3. (C/H) Gate all Hyper paths behind opt-in flags with strict no-overhead defaults when disabled.
4. (H) Add baseline reproducibility snapshots (hash + compatibilityVersion) for replay checks.

### Phase 12 – Lifecycle Runtime (Juvenile/Adult Focus) [Planned]

Files: `hyper/lifecycle.ts`, `hyper/focus.ts`, `hyper/morphPolicies.ts`
Steps:

1. (H) Implement lifecycle stage controller with explicit transitions and cooldown/hysteresis guards.
2. (H) Add focus metrics and probe cadence plumbing (cheap-per-epoch + scheduled expensive probes).
3. (H) Add local growth/prune hooks with strict budget checks and rollback boundaries.
4. (H) Instrument stage telemetry required for later assimilation decisions.

### Phase 13 – Assimilation & Compact Write-Back [Planned]

Files: `hyper/assimilation.ts`, `hyper/dnaWriteback.ts`
Steps:

1. (H) Implement per-module assimilation of generators/knobs only (no weight inheritance).
2. (H) Add deterministic seeded write-back policy and compatibility-version-aware serialization.
3. (H) Add guardrails preventing DNA bloat (prefer archetype deltas/sparse hints over explicit adjacency).
4. (H) Add audit traces for assimilation acceptance/rejection decisions.

### Phase 14 – Hyper Evolution Integration [Planned]

Files: `hyper/mutation.ts`, `hyper/crossover.ts`, `hyper/speciation.ts`, `src/neat/*`
Steps:

1. (H) Integrate DNA-aware mutation and crossover policies with budget-viability normalization.
2. (H) Extend speciation distance to DNA programs and wiring-cost preferences.
3. (H) Wire lifecycle-aware orchestration into evolution loops with deterministic replay checkpoints.
4. (H) Add regression tests for seed stability and phenotype hash reproducibility.

### Phase 15 – Hyper Cache & Morph Stress Validation [Planned]

Files: `hyper/phenotypeBuilder.ts`, `hyper/genotype.ts`, `hyper/adjacencyCache.ts`
Steps:

1. (C/H) Hash genotype signatures; reuse phenotype slabs (copy-on-write mutated sections) with ref counts.
2. (H) Adjacency cache key = (genotypeHash, substrateHash, cppnHash, threshold, precisionMode, compatibilityVersion).
3. (N) Optional disk persistence (binary blobs + JSON header) gated by size threshold.
4. (B) IndexedDB/OPFS chunk storage; async hydration and eviction stats.
5. (C) LRU eviction with byte cap (`hyperAdjacencyCacheMaxBytes`).
6. (C) Stress tests: deterministic hits, eviction correctness, stale-pointer detection, churn leak slope checks.
7. (H) Metrics: hit ratio, rebuild speed-up (>2× target), cache overhead per active connection, eviction cost (<5% build time).

### Phase 16 – Hyper Scale Validation & Hardening [Planned]

Files: benchmark harness + rollout docs + CI gates
Steps:

1. (H) Run high-scale determinism and performance sweeps under fixed seeds and replay streams.
2. (H) Validate compatibility-version upgrade/downgrade behavior and cache key hardening.
3. (H) Confirm acceptance criteria from `plans/HyperEvoDevoMorphoNEAT.md` are met with evidence artifacts.
4. (C/H) Final rollout readiness review with fallback/disable strategy documented.

## Cross-Cutting Utilities

1. `memoryStats()` : counts & approximate bytes (connections, nodes, slabs, pools, caches) + active flag states + environment heuristics (browser only). All data will be sourced from the `Centralized Memory Manager`.
2. Flag definitions in `config.ts` with environment gating (auto-disable unsupported features; expose status via stats).
3. `wiringStats()`: Tracks wiring cost metrics required by HyperEvoDevo MorphoNEAT, such as `totalWiringLength`, `interModuleEdgeCount`, `meanEdgeLength`, and `modularityQ`.
4. `adjacencyCacheStats()` (Hyper): entries, bytes, hit/miss, evictions, persistence bytes.
5. `precisionStats()` capturing current precision modes, overflow/underflow counters, scaling adjustments.
6. `poolHighWaterMarks()` for churn leak detection and morphogenesis stress tests.

## Risk Matrix (Expanded)

| Risk                             | Impact                | Env Bias | Mitigation                              |
| -------------------------------- | --------------------- | -------- | --------------------------------------- |
| Pool reset bug (stale gradients) | Incorrect training    | Both     | Comprehensive reset + tests             |
| Typed array vs object desync     | Silent logic errors   | Both     | Version counter + dev asserts           |
| Precision downcast loss          | Accuracy degradation  | Both     | Opt-in + drift tests + fallback         |
| Cache retention leak             | Memory bloat          | Node     | Ref counts + weak maps + idle sweep     |
| Adjacency cache blow-up          | OOM / GC thrash       | Browser  | Byte cap + LRU + soft budget trigger    |
| Plasticity buffer leak           | Gradual creep         | Both     | Pool reset tests + leak harness         |
| Morphogenesis fragmentation      | Retained wasted bytes | Browser  | Periodic compaction + metrics           |
| Large slab copy jank             | UI stalls             | Browser  | Chunked copies + cooperative scheduling |
| IndexedDB growth overflow        | Quota errors          | Browser  | Size accounting + eviction threshold    |
| Disk cache stale bloat           | Disk waste            | Node     | TTL metadata + size pruning             |
| WASM incompatibility             | Crash / wrong math    | Both     | Feature detection + test parity         |
| SAB unavailable                  | Worker perf drop      | Browser  | Graceful downgrade to standard workers  |

## Test Strategy Additions (Environment Aware)

| Category           | Tests                                                        | Env              |
| ------------------ | ------------------------------------------------------------ | ---------------- |
| Pooling            | Acquire/release idempotence; stable memory after cycles      | Both             |
| Slab Packing       | Forward outputs parity vs object mode (random seeds)         | Both             |
| Precision          | f32 vs f64 drift <1e-5 median; f16 drift <1e-4               | Both             |
| Budget Enforcement | Growth beyond cap triggers prune; respects soft budgets      | Both             |
| Compression        | Serialize+deserialize equivalence; streaming decode          | Both             |
| Caching            | Deterministic cache hits; eviction correctness; speed-up >2× | Both             |
| Plasticity         | Enable/disable buffers stable heap over N cycles             | Both             |
| Morph Churn        | Repeated grow/prune stable high-water marks                  | Both             |
| Scheduling         | Frame-sliced activation no >5% frames >16ms                  | Browser          |
| Persistence        | Disk/IndexedDB eviction & size accounting                    | Node/Browser     |
| WASM Kernels       | Parity + performance improvement metrics                     | Feature-detected |

## Benchmark Scenarios (Initial Set)

1. Build-only: construct 100k sparse connections (Browser variant 50k if limits encountered).
2. Forward pass: batch=1, 3 hidden layers, 500 iterations (Browser with optional frame-sliced mode).
3. Evolution cycle: mutation + prune + rebuild for population N=50 (Hyper off baseline).
4. Morphogenesis loop: growth + prune every 100 steps (Hyper later phases).
5. Adjacency cache stress: alternating thresholds + partial genotype mutations (eviction path).
6. Phenotype cache reuse: unchanged genotype vs mutated control; measure rebuild time ratio.
7. Precision switch: f32 vs f16 memory + accuracy drift.
8. Streaming window: long sequence forward vs windowed (Phase 10).
9. Browser UI responsiveness: % frames >16ms during large growth events.

Metrics: wall time, RSS / usedJSHeapSize, GC events, bytes/connection, slab fragmentation %, cache hit ratio, frame overrun %, compression ratio, copy bandwidth (MB/s), heuristic error factor (Browser).

## Documentation & Developer Guidance

1. Update README performance section after Phase 3 (slabs) & Phase 6 (precision) with environment notes.
2. Migration notes: disabling object mode & enabling slab packing safely, feature compatibility matrix.
3. Diagrams (ASCII → SVG) for slab layout, cache layering (adjacency / phenotype), environment flows.
4. Troubleshooting guide: leak detection, reading stats, tuning budgets, Browser responsiveness tips.
5. Hyper docs cross-link memory flags required for adjacency/phenotype caching.

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

## Coordination With HyperEvoDevo MorphoNEAT Plan (Expanded)

Dependency Mapping:
Hyper Phase -> Memory Requirement -> Memory Phase
A (DNA + deterministic development) -> Deterministic rebuild stable -> Phase 0–1,11
B (focus + local growth/prune) -> Efficient growth/prune -> Phase 2,5,12
C (adult optimization + equilibrium) -> Lifecycle stabilization -> Phase 12–13
D (assimilation write-back) -> Compact deterministic write-back -> Phase 13–14
E (evolution integration) -> Cache/evolution coupling -> Phase 11,14,15
F (scale validation) -> Bench infra + hardening -> Phase 0,10,16

Track mapping (authoritative numbering for execution):

- Hyper A (DNA + deterministic development) -> Memory Track 2 / Phase 11
- Hyper B (focus + local growth/prune) -> Memory Track 2 / Phase 12
- Hyper C (adult optimization + equilibrium) -> Memory Track 2 / Phase 13
- Hyper D (assimilation write-back) -> Memory Track 2 / Phase 14
- Hyper E (evolution integration) -> Memory Track 2 / Phase 15
- Hyper F (scale + stress validation) -> Memory Track 2 / Phase 16

Extended Mapping (Granular Alignment, Environment nuance):

| Hyper Phase               | Memory Concern                   | Memory Layer / Phase                       | Notes                             |
| ------------------------- | -------------------------------- | ------------------------------------------ | --------------------------------- |
| 0 (Scaffolding)           | Flag isolation & zero overhead   | Flags + baseline instrumentation (Phase 0) | Ensure feature off path identical |
| 1 (Genotype/Substrate)    | Genotype size vs phenotype ratio | Metrics extension (Target + Additional)    | Track bytes/genotype              |
| 2 (Rule Engine)           | Deterministic expansion cost     | Phase 1 slimming + profiling harness       | Avoid premature allocations       |
| 3 (CPPN Indirect)         | Adjacency generation & cache     | L7 (adjacency cache) + Phase 11            | Hit ratio & byte cap              |
| 4 (Morphogenesis)         | Churn & budget enforcement       | Phase 2,5 + churn tests                    | Monitor pool high-water marks     |
| 5 (Plasticity)            | Side buffer footprint            | Phase 3 packing + new flag                 | Optional typed arrays only        |
| 6 (Telemetry)             | Lazy metrics buffers             | Cross-cutting utilities                    | Zero retained when disabled       |
| 7 (Evolution Integration) | Multi-offspring rebuild reuse    | Phenotype cache (L7)                       | Minimize rebuild duplicates       |
| 8 (Scale Validation)      | Peak memory, rebuild variance    | Benchmark suite & CI thresholds (Phase 16) | Pass/fail gating                  |

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

### Dev Notes

This section contains active, project-wide guidelines for development and testing. All contributions should adhere to these standards to maintain code quality, consistency, and educational clarity.

Testing requirements:

- all tests should have a single expectation.
- follow AAA pattern (arrange, act, assert)
- group tests into scenarios with describe(), nest scenarios as needed, no limit on layers.
- when possible, define common testing data directly on the describe() and then write the assertions for it, this also applies for nested scenarios as they each represent more specific cases as it goes down into sub branches.
- aim for 100% testing coverage
- make sure to check existing files before creating/updating one, to be sure you are using the right file, in the right folder, for example `test/neat/` and also to be following the same file pattern inside that folder.

A good test structure follows the single-expectation rule:

```typescript
describe('Scenario: A group of related tests', () => {
  // Arrange: Set up common data or mocks here
  const testData = {
    /* ... */
  };

  describe('Context: A more specific sub-scenario', () => {
    // Act: Perform a common action for this context
    const result = performAction(testData);

    it('should fulfill a single, specific expectation', () => {
      // Assert
      expect(result.property).toBe(true);
    });

    it('should meet another distinct expectation', () => {
      // Assert
      expect(result.otherValue).toEqual(42);
    });
  });
});
```

Also:

- Always add JSDocs to all methods, classes, const, let
- Add or update inline comments within methods to explain each step or detail.
- This is an educative NN library, keep the docs detailed and educative
