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

> Build Baseline Update (Aug 2025): TypeScript/webpack target pinned to **ES2023** (previously floating `ESNext`) to ensure reproducible emitted output and educational clarity. Modern features beyond ES2023 may still appear behind feature flags with graceful degradation or transpilation guidance.

> Dist-Only Benchmark Pivot (Aug 2025): The benchmarking infrastructure no longer performs src vs dist variant comparisons. Only the built `dist` bundle metrics are recorded. Historical entries with dual-mode deltas persist in the rolling history until aged out (≤10 snapshots). This reduces noise and maintenance; future re-introduction of a second variant will only occur if a materially different optimized production build (treeshaken/minified) diverges behaviorally or structurally from the reference `dist` artifact.

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

| Metric                                        | Baseline (TBD) | Target Phase | Goal                              |
| --------------------------------------------- | -------------- | ------------ | --------------------------------- |
| Avg bytes / active connection                 | measure        | 4            | -25% vs baseline                  |
| Peak heap growth per 100k added connections   | measure        | 5            | < 12 MB                           |
| GC pause impact (allocation-heavy evolutions) | measure        | 6            | -30% vs baseline                  |
| Phenotype rebuild allocation churn            | measure        | 7            | amortized O(1) per edge via reuse |

### Additional Hyper-Specific Metrics (Introduced Once Hyper Phases Land)

| Metric                                                   | Baseline (Post Phase 1 Hyper) | Target Phase | Goal                                                   |
| -------------------------------------------------------- | ----------------------------- | ------------ | ------------------------------------------------------ |
| Bytes / genotype (symbolic)                              | measure                       | Hyper 8      | < 0.5% of phenotype bytes at scale (document)          |
| Adjacency cache hit ratio                                | measure                       | Hyper 3      | > 70% on repeated evaluations (bench harness scenario) |
| Adjacency cache bytes / active connection                | measure                       | Hyper 3      | < +6 bytes (amortized) with cap                        |
| Plasticity side-buffer bytes / active plastic connection | measure                       | Hyper 5      | < 8 bytes (float32 rate + accumulator)                 |
| Morphogenesis churn leak slope (pool high-water mark)    | measure                       | Hyper 4      | ~0 over final 20% iterations                           |
| Rebuild time variance (p95 / median)                     | measure                       | Hyper 8      | < 2.5× ratio (determinism stability)                   |
| Cache eviction overhead (per miss)                       | measure                       | Hyper 7      | < 5% of total build time in stress test                |

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

### Phase 0 – Baseline Instrumentation (Condensed Summary)

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
|------|-------------|--------------|------------------|
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

### Phase 1 – Field Audit & Slimming (Condensed Summary)

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
| 1k   | 1.99        | 0.37         | 69               | 1       |
| 10k  | 7.95        | 3.64         | 65               | 1       |
| 50k  | 45.40       | 3.91         | 65               | 1       |
| 100k | 62.89       | 9.71         | 64               | 7       |
| 200k | 149.08      | 19.66        | 64               | 7       |

Achievements:
- Connection enumerable keys ≤9 (goal met).
- Deterministic reproducibility baseline in place.
- NodePool skeleton prepared (no network integration yet at end of Phase 1).
- Bytes/connection stable (pre-slab reference).

Deferred / Hand-off to Phase 2:
- Variance stabilization (<7% CV at 100k & 200k).
- Further slimming (node error SoA) scheduled for Phase 3 (slab introduction).
- Enforcement gate (fail on Connection key regression) postponed until post-pooling variance stabilization.

### Phase 2 – Node Pooling & Governance (Finalized – COMPLETE)

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
| ~0.66         | 0                  | ≥0.5 / ≤2  | PASS   |

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
2. Flag definitions in `config.ts` with environment gating (auto-disable unsupported features; expose status via stats).
3. `adjacencyCacheStats()` (Hyper): entries, bytes, hit/miss, evictions, persistence bytes.
4. `precisionStats()` capturing current precision modes, overflow/underflow counters, scaling adjustments.
5. `poolHighWaterMarks()` for churn leak detection and morphogenesis stress tests.

---

## Risk Matrix (Expanded)

| Risk                             | Impact             | Env Bias | Mitigation                  |
| -------------------------------- | ------------------ | -------- | --------------------------- |
| Pool reset bug (stale gradients) | Incorrect training | Both     | Comprehensive reset + tests |

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
------------------------- | -------------- | -------------------- | -----
0 (Scaffolding)           | Flag isolation & zero overhead | Flags + baseline instrumentation (Phase 0) | Ensure feature off path identical
1 (Genotype/Substrate)    | Genotype size vs phenotype ratio | Metrics extension (Target + Additional) | Track bytes/genotype
2 (Rule Engine)           | Deterministic expansion cost | Phase 1 slimming + profiling harness | Avoid premature allocations
3 (CPPN Indirect)         | Adjacency generation & cache | L7 (adjacency cache) + Phase 7 | Hit ratio & byte cap
4 (Morphogenesis)         | Churn & budget enforcement | Phase 2,4 + churn tests | Monitor pool high-water marks
5 (Plasticity)            | Side buffer footprint | Phase 3 packing + new flag | Optional typed arrays only
6 (Telemetry)             | Lazy metrics buffers | Cross-cutting utilities | Zero retained when disabled
7 (Evolution Integration) | Multi-offspring rebuild reuse | Phenotype cache (L7) | Minimize rebuild duplicates
8 (Scale Validation)      | Peak memory, rebuild variance | Benchmark suite & CI thresholds | Pass/fail gating

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

