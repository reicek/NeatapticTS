# Memory Optimization Plan (Multi-Layer Strategy for Very Large Networks)

Goal: Enable construction, evolution, and training of networks scaling toward 10^6+ connections on commodity JS/TS runtimes (Node / Browser) while preserving performance. This plan layers orthogonal strategies, mapped to concrete files and phased for incremental adoption.

---

## Guiding Principles
1. Pay-for-use: No overhead unless feature enabled (flags + lazy allocation).
2. Reuse & Pool: Prefer object pooling / typed array slabs over many small objects (already partially implemented for `Connection`).
3. Structural Sparsity First: Keep graphs sparse; optimize dense fast-paths only when needed.
4. Incremental & Measurable: Each optimization introduces a benchmark & memory snapshot.
5. Branch Containment: Refactors isolated (one concern per PR) with compatibility shims where needed.

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

Baseline numbers gathered in Phase 0 using synthetic builder + Node `--inspect` snapshots.

---

## Optimization Layers Overview
Layer | Focus | Techniques
----- | ----- | ----------
L1 | Data structure slimming | Field layout audits, optional fields, shared constants
L2 | Object pooling & reuse | Extend pooling beyond `Connection` (Nodes, typed activation arrays)
L3 | Slab / SoA packing | Consolidate per-connection scalars into typed arrays (weights, gains, masks)
L4 | Sparse representation | Coordinate lists + pruning schedule + regrowth budgets
L5 | Adaptive precision | Switchable float32 weights / float16 (future) with loss scaling
L6 | Streaming & chunking | Iterative construction / partial activation windows
L7 | Generation / adjacency caching | Hash-based reuse of indirectly generated substrates + adjacency lists & phenotype slabs (Hyper MorphoNEAT)
L8 | Compression / serialization | Delta & run-length encoding for persistence

Note: L7 is subdivided conceptually into: (a) adjacency cache (edge candidate lists keyed by structural hash + thresholds) and (b) phenotype slab reuse (ref-counted typed arrays) to cover both indirect generation reuse and full phenotype reconstruction avoidance when unchanged.

---

## Phase Plan (Concrete Steps & Affected Files)

### Phase 0 – Baseline Instrumentation
Files: `scripts/` (new `benchmark-memory.cjs`), `src/architecture/network.ts`
Steps:
1. Add internal util to measure approximate object sizes (shallow) using JSON length + heuristic multiplier.
2. Create benchmark constructing networks with: 1k, 10k, 50k, 100k connections (random sparse) and logging heap snapshots (Node). (Manual script first.)
3. Record baseline metrics in this document (append section Baseline Results).
Deliverable: Baseline numbers to calibrate targets.

### Phase 1 – Field Audit & Slimming
Files: `src/architecture/connection.ts`, `src/architecture/node.ts`
Actions:
1. Identify fields never read in hot loops (e.g., rarely used optimizer states) & lazily attach when optimizer type selected.
2. Replace boolean + small numeric config flags with bitfield (e.g., status flags: enabled, hasGater, dropMask) stored in a single `uint8` equivalent OR reserved typed side-array in slab later.
3. Ensure `innovation`, `geneId` stored as 32-bit ints (document assumption) for potential typed packing.
4. Document memory layout in comments.
Impact Goal: 5–10% per-object reduction.

### Phase 2 – Node Pooling
Files: `src/architecture/node.ts`, new `src/architecture/nodePool.ts`
Steps:
1. Implement static pool mirroring `Connection.acquire/release` with reset logic.
2. Add `Network.releaseNode(node)` invoked when pruning removes isolated nodes (future morphogenesis compatibility).
3. Guard tests relying on strict instance identity order (update if needed).
Risks: Mutation logic assuming constructor side effects (bias init). Mitigation: replicate in reset.

### Phase 3 – Extended Slab Packing
Files: `src/architecture/network.slab.ts`, `network.ts`
Enhancements:
1. Pack additional per-connection scalars: gain, dcMask, enabled into parallel typed arrays (`Float32Array` + `Uint8Array`).
2. Provide view objects only when legacy API requests `.connections` array (lazy materialization). Keep existing object path for backward compatibility during transition.
3. Introduce slab version counter; invalidate on structural mutation.
4. Add fast path activation using packed gains (already partial) — extend to training path for weight updates (write-back after batch).
Success: Measured reduction in retained JS objects (connections) after enabling slab-only mode flag (e.g., `config.connectionObjectMode=false`).

### Phase 4 – Sparse Growth + Prune Budget Integration
Files: `network.prune.ts`, `network.ts`, new `network.sparsityBudget.ts`
Steps:
1. Track running connection count & target sparsity regionally (per module once hyper lands but initially global).
2. Add `ensureBudget()` invoked pre-growth mutation; rejects or triggers prune/regrow cycle.
3. Add optional exponential backoff for growth when near memory limit (exposed via config: `maxConnections`, `growthGraceFraction`).
4. Tests: Exceeding budget triggers prune and keeps count under threshold.
 5. Prepare hooks for Hyper morphogenesis integration (Phase 4 Hyper) so runtime growth events reuse existing budget computations rather than duplicating logic.

### Phase 5 – Adaptive Precision & Mixed Precision Consolidation
Files: `network.ts`, `node.ts`
Actions:
1. Unify precision flags (currently `_activationPrecision`, `_useFloat32Weights`) into a struct.
2. Add experimental float16 (Uint16 + conversion tables) path behind flag `config.float16Experimental` for inference only at first.
3. Loss scaling synergy with existing mixed precision state (extend scaling heuristics, bounding logic).
4. Benchmark inference memory shrink for large activation arrays.

### Phase 6 – Allocation Churn Reduction
Files: `activationArrayPool.ts`, `network.connect.ts`
Steps:
1. Add batch connection creation API that reserves slab space in one growth (amortize reallocation).
2. Extend activation pool with LRU trimming & periodic compaction stats.
3. Introduce optional ring-buffer reuse for temporal sequence activations (`reuseSequenceBuffers` flag).

### Phase 7 – Genotype / Phenotype Caching (Couples with Hyper Plan Phase 3+)
Files: `hyper/phenotypeBuilder.ts`, `hyper/genotype.ts`
Steps:
1. Hash genotype structural signature; if previously built, reuse cached slab arrays (clone typed arrays only when mutated).
2. Maintain ref-counted cache entries to allow GC after evolution selection.
3. Benchmark rebuild time vs cache hit.
 4. Introduce adjacency cache: key = (genotypeHash, substrateHash, cppnHash, threshold). Store compact SoA arrays (src Uint32, dst Uint32, weight Float32/16, optional metadata Uint8/Float32). Provide size accounting.
 5. Add LRU eviction with configurable byte cap (`hyperAdjacencyCacheMaxBytes`) and eviction metric logging.
 6. Tests: deterministic cache hit across repeated builds; eviction triggers when size > cap; post-eviction rebuild produces identical phenotype.
 7. Metrics captured: hit ratio, bytes cached, average rebuild speed-up vs cold build.

### Phase 8 – Serialization Compression
Files: `network.serialize.ts`, `hyper/serialization.ts`
Techniques:
1. Delta encode weights (store int16 scaled diff from previous).
2. Run-length encode disabled / pruned segments.
3. Optional gzip step documented (consumer-level, not in core).
4. Provide size stats pre/post.
 5. Include optional genotype-only compression path (rules + CPPNs) and measure ratio separately to highlight symbolic compactness.

### Phase 9 – Streaming Activation Windows (Optional Advanced)
Files: `network.ts`, new `network.window.ts`
Use Cases: Very deep recurrent nets with truncated BPTT.
Steps:
1. Maintain circular buffer of activations of length `windowSize` instead of full history.
2. Expose API `forwardWindowed(inputs[])` for streaming tasks.
3. Memory saving measured against naive per-timestep arrays.

---

## Cross-Cutting Utilities
1. `memoryStats()` (new file `src/utils/memory.ts`): returns counts & approximate bytes (connections, nodes, slabs, pools).
2. `scripts/bench-report.cjs`: aggregates benchmarks into JSON for CI gating.
3. Flag definitions in `config.ts` with defaults & docstrings.
 4. `adjacencyCacheStats()` (hyper only): returns current entries, bytes, hit/miss counters, eviction count.

---

## Risk Matrix
Risk | Impact | Mitigation
---- | ------ | ---------
Pool reset bugs causing stale gradient state | Incorrect training | Comprehensive reset function & unit tests
Typed array out-of-sync with object view | Silent logic errors | Version counter + assertions in dev mode
Precision downcast harming accuracy | Model quality drop | Opt-in flag + validation tests
Cache memory retention (genotype) | Memory leak | Ref counting + weak map fallback
 Adjacency cache blow-up | OOM / GC thrash | Byte cap + LRU eviction + metrics gating
 Plasticity buffer leak | Gradual memory creep | Pool reset tests + leak detection harness (high-iteration morph churn)
 Morphogenesis churn fragmentation | Elevated retained size | Periodic compaction + high-water mark monitoring

---

## Test Strategy Additions
Category | Tests
-------- | -----
Pooling | Acquire/release idempotence, memory not growing after cycles
Slab Packing | Equality of forward outputs vs object mode across random seeds
Precision | f32 vs f64 numerical drift bounds < 1e-5 median
Budget Enforcement | Growth attempts beyond cap produce expected prune events
Compression | Serialize+deserialize yields identical functional outputs
 Caching | Adjacency & phenotype cache hit ratio > threshold; eviction preserves correctness
 Plasticity | Enabling/disabling buffers leaves heap stable after N cycles
 Morph Churn | Repeated grow/prune cycles show stable pool high-water marks

---

## Benchmark Scenarios (Initial Set)
1. Build-only: construct 100k sparse connections.
2. Forward pass batch=1, depth moderate (3 hidden layers) repeated 500 iterations.
3. Evolution cycle: mutation + prune + rebuild for N=50 population.
4. Morphogenesis loop (future): growth + prune events every 100 steps.
 5. Adjacency cache stress: alternating thresholds & partial genotype mutations to exercise eviction.
 6. Phenotype cache reuse: repeated evaluation cycles with unchanged genotype vs mutated control.

Metrics captured: wall time, RSS, heap used, GC minor/major counts (where available), bytes/connection estimate.

---

## Incremental Adoption Flags (config.ts)
Flag | Purpose | Default
---- | ------- | -------
`connectionObjectMode` | Keep legacy Connection objects active | true
`enableNodePool` | Use Node pooling | false
`enableSlabPackingV2` | Pack gains/masks/enabled into slabs | false
`maxConnections` | Hard budget for active edges | Infinity
`growthGraceFraction` | Allowed temporary overflow before prune | 0.05
`float16Experimental` | Enable float16 inference path | false
`reuseSequenceBuffers` | Reuse rolling activation buffers | false
 `enableHyper` | Master flag for Hyper MorphoNEAT features | false
 `enableHyperTelemetry` | Enable hyper developmental trace & module metrics | false
 `hyperAdjacencyCacheMaxBytes` | Cap for adjacency cache (LRU eviction) | 64 * 1024 * 1024
 `enablePhenotypeCache` | Reuse slabs when genotype hash unchanged | true
 `enablePlasticityBuffers` | Allocate per-connection plasticity side state | false

---

## Documentation & Developer Guidance
1. Update README performance section after Phase 3 & 5.
2. Provide migration notes for disabling object mode when stable.
3. Add diagrams (ASCII initially) explaining slab memory layout.

---

## Baseline & Progress Logging
Append a new subheading after each phase completion:
```
### Phase X Results
Date:
Changes:
Metrics:
Notes:
```

---

## Coordination With Hyper MorphoNEAT Plan
Dependency Mapping:
Hyper Phase -> Memory Requirement -> Memory Phase
1–2 -> Deterministic rebuild stable -> Phase 0–1
3 (CPPN) -> Fast edge generation & caching -> Phase 3,7
4 (Morphogenesis) -> Efficient growth/prune -> Phase 2,4
5 (Plasticity) -> Extra per-conn state packing -> Phase 3 extension
8 (Scale validation) -> Bench infra -> Phase 0,6,7

Extended Mapping (Granular Alignment):
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

## Future (Post Core) Explorations
1. WebAssembly weight update kernels.
2. WebGPU compute path for slab arrays.
3. Browser SharedArrayBuffer pooling for multi-worker evolution.
 4. Persistent compressed adjacency tiers (multi-threshold progressive decoding) for extreme substrates.
 5. Probabilistic slab compaction (on idle) scheduling heuristics.

---

End of memory optimization plan.