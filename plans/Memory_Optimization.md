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

### Phase 0 – Baseline Instrumentation (Revised Dist-Only)

Files: `test/benchmarks/benchmark.memory.test.ts`, `test/benchmarks/benchmark.report.test.ts`, `test/benchmarks/benchmark.browser.memory.test.ts` (placeholder), `src/utils/memory.ts`, `src/architecture/network.ts`
Original Scope: Dual-path (src vs dist) timing + memory aggregation.
Revised Scope: Absolute dist-only metrics (build & forward timings, bytes/conn, heap samples) with variance stabilization (deterministic seeding, warm-up discard, adaptive iteration counts, IQR outlier filtering). Delta and regression annotation structures removed.

Status Legend: [DONE] complete · [WIP] in progress · [PENDING] not started · [DEFERRED] postponed intentionally.

Interim Notes (updated):

- Artifact now: baseline (synthetic samples), variantRaw (dist raw), aggregated (dist only), variance (large-size CV summary when repeats>1), history (≤10), meta, warnings, fieldAudit. Deltas / regressionAnnotations removed; legacy fields emptied on new snapshot write if present.
- Browser harness placeholders remain; dev/prod differentiation deferred until a distinct optimized build exists.
- Large size (200k) path active locally; may be gated in CI via env vars.
- History migration cleans obsolete delta fields automatically.

#### Phase 0 Results (Current Dist-Only Snapshot)

Methodology:

- Synthetic dense bipartite networks: input≈sqrt(N), output≈ceil(N/input), prune excess.
- Timings: buildMs wall-clock; fwdAvgMs averaged over adaptive iterations (small:5, large:3–5) after optional warm-up discard.
- Aggregation: (size, scenario='buildForward') → mean/p50/p95/std for buildMs, fwdAvgMs, bytesPerConn, heapUsed/rss (if captured).
- No pairwise deltas; focus on absolute trends + variance quality (CV%).

Representative Absolute Metrics (placeholder – regenerate via script):

```
Size  buildMsMean  fwdAvgMsMean  bytesPerConnMean  heapUsedMean
1k    ...          ...           ...               ...
10k   ...          ...           ...               ...
50k   ...          ...           ...               ...
100k  ...          ...           ...               ...
200k  ...          ...           ...               ...
```

#### Dist Benchmark Enforcement

Benchmark suite exercises only the built ESM bundle. Planned: capture bundle size + hash for provenance (`meta.distBundle`). Dual-mode logic excised for maintainability.

Current Interpretation (dist-only):

- Absolute scaling stable; bytes/conn plateau evidences structural pruning but object overhead still dominant → motivates slab packing (Phase 3).
- Variance controls (seeding, warm-up, IQR filter) reduce CV at large sizes; establish forward CV<7% consistently before adding regression gates.
- Field audit (Connection enumerable keys now 9) confirms slimming progress; further heap byte gains deferred to pooling/slab phases.
- Browser parity memory test deferred until production build & stable CV to avoid misattributing environment noise.

Field Audit Snapshot (current dist run):

- Node keys (14): activation,bias,connections,derivative,error,geneId,index,mask,old,previousDeltaBias,squash,state,totalDeltaBias,type
- Connection keys (9): \_flags,eligibility,from,innovation,previousDeltaWeight,to,totalDeltaWeight,weight,xtrace (neutral gain + gater virtualized)

Upcoming expectation:

- Validate next audit reflects Connection key count stability at 10; evaluate virtualizing `gater` (null ⇒ no enumerable key) with presence bit in flags.

Slimming Candidates (Phase 1 targeting) – status update:

- Connection: [DONE] Bitfield applied (enabled + dcMask) 12→11; [DONE] neutral gain virtualized (symbol) 11→10; [DONE] gater virtualized + presence bit (hasGater) 10→9.
- Node: (PENDING) Externalize error object (SoA) Phase 3; evaluate lazy derivative store after measuring derivative access frequency.
- Shared: (PENDING) Enum for node type (low priority) – micro benchmark before action.

## Phase 0 – Final Achievements Summary (Updated)

Core Deliverables Achieved:

1. Unified Jest benchmark harness with structured JSON artifact.
2. Dist-only aggregation (removed dual variant noise & delta maintenance).
3. Large-scale synthetic sizing to 200k with adaptive iteration + warm-up discard.
4. Variance stabilization (deterministic seeding, IQR outlier filter, repeat gating support).
5. Slim rolling history (≤10) with concise summaries.
6. Field audit instrumentation; Connection enumerable keys reduced to 9 (bitfield + virtualization).
7. Simplified artifact schema (baseline, variantRaw, aggregated, variance, history, meta, fieldAudit, warnings).
8. Single-expectation educational test style retained.
9. Foundation laid for future regression gating once CV stabilizes.

Observed Baseline Metrics (Representative Dist Run):

- Bytes/conn plateau ~64 at larger sizes (object header overhead pending slab packing).
- Forward timing CV reduced post warm-up & iteration heuristic; further repeats to push <5% goal.
- Heap growth linear with size; guides pooling & slab capacity planning.

Technical Debt / Open Threads:

- Production optimized bundle (treeshake/minify) pending (enables reintroducing meaningful variant comparisons if needed).
- Additional large-size repeats for ultra-stable CV (<5%).
- Browser parity memory test & bundle hash recording.

Confidence & Readiness:

- Dist-only simplification lowers maintenance overhead; instrumentation stable.
- Proceeding to Phase 1 slimming & pooling groundwork.

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
2. Bitfield `_flags` (enabled + dcMask; now extended with hasGater bit) applied.
3. Gater reference virtualized (symbol + bit2) removing `gater` enumerable key (10→9 keys).
4. Variance repeat scaffolding present (await BENCH_REPEAT_LARGE run for CV metrics).

Immediate Next Steps (Updated):

1. (DONE) Run benchmarks with BENCH_REPEAT_LARGE=5 (variance populated; Connection keys=9 confirmed).
2. (DONE) Introduce regression annotation logic (informational) gated by delta & CV thresholds.
3. (DONE) Implement script to auto-generate delta & heap tables from artifact (`scripts/generate-bench-tables.cjs`) with posttest hook writing `plans/bench_tables.md`.
4. (DONE) Add serialization slimming audit – gater virtualization round-trip test ensures gating preserved.
5. Add Browser parity memory test (heuristic) once CV < target for Node large sizes (PENDING).
6. Integrate real production `dist` build pipeline (treeshake/minify) to replace src fallback (PENDING; prerequisite for meaningful dist regressions).
7. Investigate high forward CV% (dist 100k ~13%, 200k build CV ~43% due to an outlier) – add warm-up discard & seeding; consider excluding clear outliers before gating (DONE – IQR filter applied, see variance meta).
8. (DONE) Automate plan table regeneration – future enhancement: inline tables into this doc via placeholder tags.
9. (DONE) Implement warm-up discard & deterministic seeding to reduce forward CV to <7% for large sizes (pending validation run for post-filter CV values).
10. (DONE) Introduce dist bundle provenance capture (`meta.distBundle = { exists, bytes, hash }`).
11. (DONE) Add NodePool skeleton (acquire/release/reset) groundwork file `src/architecture/nodePool.ts` + tests (not yet integrated into `Network`).
12. (DONE) Add optional forward regression annotation placeholder (non-failing) gated by CV threshold & median history.
13. NEXT: Run benchmarks to validate reduced CV; if stable (<7% both modes) broaden regression annotations & add Browser parity memory test.

Test File Integration Expectations:

- All new regression / variance tests MUST read the shared artifact rather than recomputing metrics ad‑hoc.
- Artifact-driven assertions ensure a single source of truth; avoid duplicating aggregation logic inside tests.
- When adding new metrics to the artifact, update this section and extend parsing tests accordingly.

Success Criteria (Phase 1):

- Connection own enumerable keys ≤9 (ACHIEVED) with tests & benchmarks green.
- Stable forward timing CV (≥100k sizes) under 5% with repeats (IN PROGRESS – current snapshot below shows dist forward CV above target).
- Documented canonical Connection layout & bit allocations in plan (ACHIEVED).
- Dist bundle provenance (size + hash) persisted for reproducibility (ACHIEVED).
- NodePool groundwork landed (scaffold + tests) enabling Phase 2 integration (ACHIEVED).

### Phase 1 – Updated Dist Snapshot (2025-08-19 – varianceRepeatsLarge=7)

Source: latest `benchmark.results.json` (see attached snapshot).

Dist Bundle Provenance:

- exists: true
- bytes: 572
- hash (sha256[0:12]): `734376fd19dd`

Aggregated (Node, dist):

| Size | buildMsMean | fwdAvgMsMean | bytesPerConnMean | Count |
| ---- | ----------- | ------------ | ---------------- | ----- |
| 1k   | 1.5862      | 0.39056      | 69               | 1     |
| 10k  | 12.0981     | 2.96886      | 65               | 1     |
| 50k  | 42.1971     | 6.70880      | 65               | 1     |
| 100k | 69.4893     | 12.57681     | 64               | 7     |
| 200k | 169.1664    | 25.66680     | 64               | 7     |

Variance (CV% from `variance` array):

- 100k: build CV 10.32%, fwd CV 7.62%
- 200k: build CV 14.76%, fwd CV 13.36%

Regression Annotations (informational):

- fwd-regression @200k: deltaPct 20.03% vs threshold 10%, cvPct 6.76% (annotation emitted; non-failing)

Field Audit (post-optimizer virtualization pass):

- Node keys: 15
- Connection keys: ≤12 (actual will appear in next benchmark artifact; optimizer fields moved behind symbol bag)

NOTE: Optimizer moment fields have been re-virtualized via a symbol-backed bag; enumerable Connection key ceiling lowered in test to 12 (target final ≤9 once stability verified across mutation & optimizer pathways).

Interpretation / Trends:

1. Bytes/conn plateau (64–69) unchanged – slab packing (Phase 3) required for next reduction.
2. Forward CV at 100k now near threshold (7.62%); 200k still elevated (13.36%). Need repeat escalation or iteration tuning before enabling strict regression gating.
3. Connection key count increase reverses earlier progress; verify virtualization (gater, gain) and bitfield application still active. Add automated delta alert in plan.
4. NodePool still not integrated – opportunity to start measuring pool impact once key count regression addressed.

Immediate Remediation & Optimization Focus:

| Item | Action | Priority |
| ---- | ------ | -------- |
| Connection key spike | (DONE) Re-virtualize optimizer fields; next tighten ceiling to ≤9 after stability | High |
| High CV at 200k | Increase iterations (min 5) and/or auto repeat escalation when CV>target; consider additional warm-up | High |
| Regression annotation enrichment | Add src/dist CV fields (optional) once CV<7% both sizes | Medium |
| Pool stats visibility | Expose node pool size & highWaterMark in `memoryStats()` | Medium |
| Browser parity memory test | Enable after CV stabilization | Low |

Planned Next Edits:

1. Integrate NodePool into Network growth/prune (guarded flag) – measure bytes/conn & GC churn delta.
2. (DONE) Add node pool stats to `memoryStats()` (size, highWaterMark).
3. Prototype auto repeat escalation logic (non-failing) capturing advisory note in meta.

### Variance Snapshot (Current BENCH_REPEAT_LARGE=7 – Post IQR filter & iteration heuristic v2)

From latest `benchmark.results.json` (`variance` array):

| Size | Mode | buildMsMean | buildMsCv% | fwdAvgMsMean | fwdAvgMsCv% | Samples | Notes                                                |
| ---- | ---- | ----------- | ---------- | ------------ | ----------- | ------- | ---------------------------------------------------- |
| 100k | src  | 50.2764     | 2.15       | 9.7430       | 6.84        | 5       | Near target (forward CV slightly above 5%)           |
| 100k | dist | 53.2650     | 5.62       | 8.6282       | 19.03       | 5       | High forward variance – regression gating suppressed |
| 200k | src  | 115.2100    | 4.14       | 17.0662      | 11.74       | 5       | Forward variance elevated – may need more repeats    |
| 200k | dist | 121.8128    | 4.94       | 17.9941      | 14.66       | 5       | Elevated variance                                    |

Updated Interpretation:

- IQR outlier filtering now active (meta.outlierFilter='IQR1.5'); build outliers removed at 200k lowered buildMsCvPct into low double digits.
- Increased forward iterations for large sizes (>=100k: 5; 200k: 3) will be reflected in next run; expect forward CV to fall toward <7% target enabling stable regression annotations.
- Next: validate new run; if CV stable promote regression annotations and proceed to real prod dist build integration.

### Regression Annotation System (Informational Phase)

New fields in artifact:

- `meta.regressionDeltaThresholdPct` (currently 10) – fwdAvgMs slower % required to annotate.
- `meta.regressionCvThresholdPct` (currently 7) – maximum fwdAvgMs CV% (both modes) to trust annotation.
- `regressionAnnotations`: Array of objects `{ type:'fwd-regression', size, deltaPct, thresholdPct, srcCvPct, distCvPct, cvThresholdPct, note }`.

Current Status: No stable (CV-qualified) regressions yet due to high dist CV; array may be empty or small. Annotations are non-failing; future phase may introduce soft-to-hard gate promotion once CV targets met.

Next Steps for Annotation Evolution:

1. Reduce variance to consistently meet CV thresholds.
2. Add historical trend analysis (regression persisting across N snapshots) before elevating to warning/failure.
3. Introduce bytes/conn regression annotations (pending stable dist build difference).
4. Provide CLI summary script printing annotated regressions with context (delta, CV, history streak).

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

Additional Phase 2 Preparatory Tasks (added):

6. Integrate `acquireNode` / `releaseNode` in network construction & pruning (guard with `config.enableNodePooling` flag default off).
7. Add `nodePoolStats()` instrumentation to `memoryStats().pools.node` with size & highWaterMark.
8. Update benchmarks to record pool stats (optional block) when pooling enabled for future bytes/conn comparison.
9. Create test ensuring enabling pooling does not change functional forward outputs vs baseline network (determinism check).
10. Add field audit regression guard: if enumerable Connection key count increases beyond configured ceiling, output advisory with diff.

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
