# Hyper MorphoNEAT: Concrete Implementation Plan (Aligned With Current NeatapticTS Core)

Hyper MorphoNEAT is a proposed hybrid framework for neural network evolution that unifies key principles from: 
* NEAT (incremental complexification, speciation, innovation tracking)
* HyperNEAT / ES-HyperNEAT (indirect CPPN encodings over geometric substrates, scalable connectivity sampling)
* Evo-Devo / Developmental Biology (rule‑driven staged growth, differentiation, symmetry, modular morphogenesis)
* NeuroMorph / Structural Plasticity research (activity‑driven synaptogenesis & pruning, local rewiring)
* Classic synaptic plasticity (Hebbian / anti-Hebbian, homeostatic adjustments)

Its long‑term goal: mimic a simplified “digital embryogenesis + lifelong adaptation” pipeline—starting from a handful of proto‑cells (minimal input/output scaffold) plus a *genetic rulebook* (developmental & pattern genes) that can:
1. Elaborate structure (grow) when additional capacity is *demonstrably* needed.
2. Reshape or prune (shrink) when regions are underutilized or wasteful.
3. Focus evolutionary search dynamically on emergent *bottlenecks* or *frontier regions* rather than the entire brain uniformly.
4. Maintain indirect encodings so extremely large phenotypes (millions of potential connections) are generated *on demand* without materializing every dormant element.

This document both (A) expands the conceptual/educational narrative and (B) refines the pragmatic phased implementation grounded in the existing codebase (`src/architecture/*`, `src/neat/*`, pruning, slabs, pooling, optimizers). The aim: introduce indirect & developmental encodings, run‑time morphogenesis, and adaptive large‑scale network support without destabilizing current NEAT / Network functionality. 

---

## 1. Conceptual Expansion: “From Proto-Brain to Adaptive Cortex”

This section formalizes the developmental metaphor underpinning the architecture. Instead of treating topology growth as a flat sequence of structural mutations, we frame the system as a staged developmental process: *pattern specification → proliferation → differentiation → guidance → refinement → lifelong adaptation*. Each mechanism (developmental rules, CPPNs, morphogenesis hooks) is deliberately aligned to one of these abstracted biological roles so that (1) contributors can predict emergent behavior, (2) design trade‑offs inherit a coherent vocabulary, and (3) future extensions (e.g. chemical gradient analogues, temporal morph phases) have an obvious insertion point.

Analogy to biological development (simplified):

| Stage               | Biological Inspiration                | Hyper MorphoNEAT Analog                                                             |
|---------------------|---------------------------------------|-------------------------------------------------------------------------------------|
| Embryonic Seed      | Few stem cells                        | Minimal genotype with input/output anchor nodes                                    |
| Patterning Gradients| Morphogens, HOX genes                 | CPPN fields + developmental rules assigning coordinates & tags                     |
| Proliferation       | Cell division                         | `replicate` rules splitting regions / subdividing connection paths                 |
| Differentiation     | Neurons specialize (sensory / interneuron / motor) | Rule‑based assignment of activation fn, plasticity mode, gating role    |
| Axon Guidance       | Growth cones follow gradients         | Spatial CPPN thresholds & local activity heuristics form new connections           |
| Pruning & Refinement| Synaptic pruning (use-it-or-lose-it)  | Activity/frequency / contribution metrics drive removal                            |
| Lifelong Plasticity | Hebbian, structural changes           | Morphogenesis hooks + plastic weight adaptation                                    |

### 1.1 Core Idea
Motivates a *rule‑first* search strategy: by beginning with a minimal developmental program the algorithm explores a compact, high‑leverage encoding space where each mutation can reshape large swaths of potential phenotype. This delays costly exploration of the vast combinatorial topology space until rules and pattern generators establish macro‑regularities (symmetry, modular bands, coordinate partitions) that scaffold efficient scaling. The approach mirrors biological canalization: early developmental constraints bias later structural elaboration toward coherent, reusable motifs.
Start *tiny* to minimize initial search space. Instead of evolving a large static genome, we evolve a **compact developmental program** that is repeatedly executed (or partially re‑executed) to refine the phenotype. Evolution acts on rules & CPPN weights (indirect), while *runtime morphogenesis* adjusts expression intensity and local connectivity. The phenotype thus becomes an emergent, continually reshaped structure rather than a fixed topology.

### 1.2 Genotype vs Phenotype Layering
Defines a strict stratification between *heritable specification* (rules + CPPN parameters) and *ephemeral realization* (instantiated nodes, connections, plastic state). This boundary supports determinism (phenotypes can be reconstructed bit‑exactly from genotype + seed), enables aggressive memory reclamation (transient runtime arrays can be discarded or downsampled between evaluations), and decouples evolutionary operators from runtime adaptation. The layering also allows analytical tooling (diffing, hashing, compression) to operate on a concise symbolic genome rather than sprawling structural graphs.
| Component               | Genotype (Stored)                      | Phenotype (Materialized / Runtime)                                  |
|------------------------|----------------------------------------|---------------------------------------------------------------------|
| Node existence         | Rule-derived (virtual slots)           | Instantiated `Node` objects (possibly pooled)                        |
| Connection potential   | CPPN output, rule masks                | Subset realized (meets thresholds + budget)                          |
| Module boundaries      | Hierarchy / symmetry rules             | Tag arrays on nodes/edges, used for focused evolution & pruning      |
| Plasticity configuration | Gene flags (hebbian rate, gating ability) | Runtime per-connection accumulators (optional)                   |

### 1.3 Dynamic Evolution Focus
Explains the allocation of evolutionary variance as an *adaptive resource scheduling* problem: mutation budget is directed toward regions exhibiting evidence of constraint (high error attribution relative to size), stagnation (age without improvement), or under‑exploration (low structural diversity). By contrast, mature, stable, and well‑performing regions experience mutation cooling, reducing destructive interference. This dynamic focus mimics targeted neurogenesis and synaptic remodeling phenomena, yielding faster convergence for a fixed total mutation rate and curbing indiscriminate bloat.
Rather than applying uniform mutation pressure, we maintain *region metrics* (error attribution, novelty, saturation). Evolution probabilities shift toward regions that are: 
* Bottlenecked (high gradient / error flow concentration)
* Under-explored (low structural entropy, few recent mutations)
* Recently regressed (performance dip localized)

Conceptually: 
```
focusScore(region) = w_err * normalizedErrorShare
                   + w_nov * (1 - structuralDiversity)
                   + w_regress * recentPerfDrop
                   - w_stable * stabilityAge
```
Sampling of mutation targets becomes weighted by `focusScore` to *steer* complexification.

                  Objectives:
                  - Introduce namespace, feature gating, and minimal type contracts without altering runtime behavior.
                  - Establish deterministic seeds & hashing utility stubs used in later phases.
                  - Guarantee zero performance / bundle size regression when flag disabled.

                  Scope Inclusions:
                  - Directory skeleton, preliminary interfaces, config flag, smoke tests, documentation pointer in README.
                  Scope Exclusions:
                  - Any allocation of new runtime arrays; mutation / reproduction changes.

                  Key Tasks:
                  1. Create `src/hyper/` with placeholder modules (`genotype.ts`, `developmentalRules.ts`, `substrate.ts`, `phenotypeBuilder.ts`, `internal/hash.ts`).
                  2. Extend `config.ts` with `enableHyper` & `enableHyperTelemetry` flags (default false).
                  3. Implement a minimal `createHyperContext(seed)` returning deterministic RNG + version tag.
                  4. Add TypeScript interfaces with TODO JSDoc blocks referencing later phases (versioned via `@phase` tag).
                  5. Add Jest smoke test: importing hyper entry when disabled does not mutate global state (heap snapshot diff < threshold, optional if infra present).
                  6. Add build size guard (compare gzip bundle delta; skip if tooling unavailable—document).

                  Interfaces Added / Changed:
                  - `config.enableHyper?: boolean`
                  - `HyperVersion = { major:1, minor:0, phase:0 }` constant.

                  Tests:
                  - `hyper.disabled.import.spec.ts`: ensures side‑effect free import.
                  - `config.flag.default.spec.ts`: asserts flag false by default.

                  Metrics & Exit Criteria:
                  - Bundle delta < 1% (or documented if tooling absent).
                  - No additional failing tests; coverage for new lines ≥ 80% (interface lines excluded).

                  Risks & Mitigations:
                  - Risk: Accidental circular import with existing `neat` modules → Mitigation: forbid `src/hyper` importing `src/neat` in linter rule.
                  - Risk: Silent performance regression → Mitigation: micro benchmark baseline captured (no hyper usage).

                  Deferred Items:
                  - Hash canonicalization logic (implemented in Phase 1 with real genotype structure).
                  - CLI / docs surface.

                  Acceptance:
                  - Tree contains hyper skeleton; CI green; disabling flag yields identical test timings (± variance threshold).
   * If memory near cap → prune least-contributing edges/nodes (respecting module diversity quotas).
   * Else if sustained high error & high utilization → targeted growth (replicate high‑pressure path or densify a sparse region).
   * Else if plateau & low exploration → mutate developmental rules / CPPN (global structural shift) before adding raw capacity.
                  Objectives:
                  - Define persistent `HyperGenotype` schema and deterministic build path to a baseline phenotype.
                  - Provide substrate coordinate assignment + hashing for regeneration equivalence.
                  - Implement serialization & hash stability tests.

                  Key Tasks:
                  1. Factory: `createInitialGenotype({ input, output, seed })` producing minimal rule list (empty for now) & substrate spec.
                  2. Substrate: implement 1D / simple 2D coordinate assignment strategies (line, grid) with deterministic ordering.
                  3. Phenotype builder (baseline): instantiate input & output nodes; create fully connected edges input→output only.
                  4. Stable hashing: `hashGenotype(geno)` (order‑independent rule hash, sorted JSON canonicalization).
                  5. Serialization: `encodeGenotype(geno)` / `decodeGenotype(json)`; version field check.
                  6. Determinism test: repeated build from same seed produces identical edge weight ordering & hash.
                  7. Performance micro‑benchmark harness (optional) to record baseline build time.

                  Interfaces Added:
                  - `interface HyperGenotype { seed:number; input:number; output:number; rules:DevelopmentalRule[]; substrateSpec:SubstrateSpec; version:1; }`
                  - `buildPhenotype(geno, { config })` (returns `Network`).

                  Tests:
                  - Round trip serialization parity.
                  - Hash stability after neutral no‑op mutate attempt.
                  - Memory footprint comparison vs direct `Network` baseline (connections count equality).

                  Metrics & Exit Criteria:
                  - Deterministic hash reproducibility = 100% across N=50 rebuilds.
                  - Build time overhead ≤ 1.1× direct instantiation.

                  Risks & Mitigations:
                  - Risk: Order dependence in JSON serialization → canonical sort property keys.
                  - Risk: Hidden mutable references (arrays reused) → deep freeze in test environment when possible.

                  Deferred Items:
                  - Rules execution engine (Phase 2).
                  - Indirect connectivity (Phase 3).

                  Acceptance:
                  - All genotype round‑trip & determinism tests pass; documentation updated with example.
  if (ctx.memory.pressure > 0.9) pruneLowContribution(net, ctx);
  else if (ctx.error.stagnant && ctx.utilization.high) replicateCriticalPath(geno, ctx);
  else if (ctx.novelty.low) diversifyRules(geno);
                  Objectives:
                  - Introduce deterministic multi‑pass rule application with priority & probability handling.
                  - Support initial structural motif expansion (replication, symmetry, hierarchy tagging).
                  - Ensure repeated genotype build yields identical phenotype for fixed random stream.

                  Key Tasks:
                  1. Rule interface finalization: `priority`, `probability`, `enabled` semantics; stable normalized param hashing.
                  2. Execution engine: sort by priority; iterate applying rules; for probabilistic rules use deterministic RNG seeded from (genotype.seed + rule.id).
                  3. Implement rule kinds:
                    - `replicate`: duplicate coordinate bands / node groups (maintain mapping table for ancestry).
                    - `symmetry`: generate mirrored coordinates and optional parameter tying metadata.
                    - `hierarchy`: assign module IDs & nested scopes.
                  4. Extend phenotype builder to expand virtual node list before physical instantiation.
                  5. Add complexity guard (maxNodes, maxRules) pre‑instantiation.
                  6. Mutation operators: toggle enable, adjust numeric params, shift priority (bounded), adjust probability sigmoid‑clamped.
                  7. Deterministic test battery: same seed + rule set produces identical node ordering & ancestry map.

                  Metrics & Exit Criteria:
                  - Phenotype node count scaling validates expected multiplicative factors for staged replication.
                  - Rule application runtime ≤ 10% of total build time for small networks (<5k nodes) in benchmark harness.

                  Risks:
                  - Cascading replication explosion → enforce geometric growth cap per pass.
                  - Symmetry drift (floating point coordinate error) → use rational or fixed‑point representation for mirror axes.

                  Deferred:
                  - Differentiation / activation assignment (future rule kind) if excluded here.

                  Acceptance:
                  - All rule semantics tests pass; no nondeterministic drift across multi‑build sequences.
Nodes inherit *base role* (input/output/hidden). Developmental rules add *traits*: gating, recurrence permission, plasticity profile, activation family set. A `differentiate` rule might look like:
```ts
{ kind: 'differentiate', params: { region: 'motor', chooseActivation: ['tanh','relu'], plasticity: 'hebbian_fast' } }
                  Objectives:
                  - Replace exhaustive direct edge enumeration with procedural generation via CPPNs to enable large coordinate substrates.
                  - Introduce controllable sparsity & pattern regularity.
                  - Maintain cache for adjacency & weight pattern reuse across evaluations.

                  Key Tasks:
                  1. `CPPNGene` definition: layers spec, activations list, weights TypedArray, innovation id.
                  2. Implement forward evaluator (pure function, no dynamic allocation in hot loop).
                  3. Candidate pair sampling strategy: configurable (all pairs up to dimension bound; radius-limited; focus-guided subset placeholder for later).
                  4. Connectivity decision: compute outputs (weight, mask probability, optional metadata channels), threshold to realize edge.
                  5. Weight scaling & optional normalization (e.g., fan-in scaling) hooks.
                  6. Adjacency cache keyed by (genotype structural hash, substrate hash, CPPN hash, threshold) → returns list of (src,dst,weight,meta).
                  7. Cache invalidation on CPPN mutation or substrate change.
                  8. Benchmark harness: measure generation vs baseline direct full connect for moderate size (e.g., 2k×2k potential pairs pruned to <5%).
                  9. Tests: symmetry invariance for mirrored coordinates; deterministic adjacency ordering; threshold monotonicity (higher threshold → subset).

                  Metrics & Exit Criteria:
                  - Indirect build time ≤ 1.5× baseline for medium nets (document exact ratio & hardware).
                  - Memory footprint: adjacency cache reuses buffers; additional overhead per realized connection < +8 bytes vs baseline.

                  Risks & Mitigations:
                  - Risk: Cache blowup for many thresholds → LRU eviction & size cap.
                  - Risk: Floating mismatch causing non-determinism → fixed precision rounding for CPPN inputs.

                  Deferred:
                  - Multi-CPPN stacking & plasticity channel usage (Phase 5/7).

                  Acceptance:
                  - Patterns reproducible; scaling metrics logged.
Describes hierarchical assembly as recursive, scope‑stacked rule evaluation: a parent rule can spawn sub‑scopes whose local coordinate frames and probability adjustments produce self‑similar yet diversified descendants. This generates fractal‑like scaling (band → stripe cluster → columnar macro‑module) while preserving traceability (each object carries an ancestry chain). The result is exponential phenotype expressivity from logarithmic genotype growth.
Rules can recursively introduce *proto‑modules* which themselves run a localized rule subset (mini developmental pass) enabling fractal expansion without a huge flat genome.

                  Objectives:
                  - Enable intra‑generation structural adjustments driven by live telemetry.
                  - Enforce memory and sparsity budgets dynamically.
                  - Provide consistent event sequencing & transactional rebuild semantics.

                  Key Tasks:
                  1. Metrics collectors: per module (activity mean/var, utilization, contribution proxy) updated post‑epoch.
                  2. Event dispatcher: `maybeMorph(event: MorphEvent, metrics)` gating by cooldown & budget state.
                  3. Actions:
                    - `activity_expand`: local replicate path or add focused edges (calls CPPN subset eval or direct small add).
                    - `low_contrib_prune`: integrate existing pruning API with module tags; maintain target sparsity.
                    - `module_split`: duplicate module nodes + incident edges; reassign subset based on coordinate partition.
                  4. Transaction layer: queue structural edits, apply in batch, rebuild slabs if size changed; reuse pools.
                  5. Budget enforcement: connection/node hard caps; forced prune order by (contribution score ↑ age).
                  6. Trace logging: append event entries with before/after counts & focus scores.
                  7. Tests: deterministic morph under fixed metrics stream; rollback safety test (simulate failed morph then revert).

                  Metrics & Exit Criteria:
                  - Morph hook overhead (disabled) <1% runtime; (enabled idle) <5%.
                  - No memory leak (object pool size returns after prune cycles).

                  Risks:
                  - Concurrent training modifications → apply morph only between batches / epochs.
                  - Cascading rebuild thrash → cooldown period & action quota per event.

                  Deferred:
                  - Focus scoring integration (Phase 7) beyond baseline metrics.

                  Acceptance:
                  - Structural edits validated by tests; performance overhead documented.
| HyperNEAT                           | CPPN encodes connectivity over geometric substrate         | CPPN(s) produce weight *and* activation / plasticity hints; substrate may be multi-dimensional                 |
| ES-HyperNEAT                        | Adaptive sampling of large substrates                      | Deferred generation: only sample connection candidates near active regions or flagged by focus metrics        |
| Evo-Devo (Dev. Biology)             | Growth via rules, differentiation, symmetry                | Rule engine with execution probabilities & priorities                                                         |
                  Objectives:
                  - Add activity‑dependent micro‑update channel (Hebbian / anti‑Hebbian) plus optional decay for transient potentiation.
                  - Guarantee negligible overhead when disabled.

                  Key Tasks:
                  1. Extend `Connection` with optional side buffers (e.g., `trace`, `plasticAccum`), allocated only when enabled.
                  2. Implement update rule: `Δw = η * pre * post - λ * w` (configurable terms) executed post forward pass.
                  3. Config gating: early return fast path; ensure branch predictability.
                  4. Reset semantics on connection pooling release.
                  5. Integrate with morphogenesis (optional): plasticity metrics feed utilization / growth decisions.
                  6. Tests: numerical validation on toy 2‑node network; performance micro‑benchmark toggling plasticity.

                  Metrics & Exit Criteria:
                  - Overhead disabled <1%; enabled small net <10% additional time (doc exact numbers).
                  - Weight drift stable (bounded by decay) over long run test.

                  Risks:
                  - Accidental interference with gradient updates → apply plasticity adjustments after optimizer step or in dedicated buffer.

                  Deferred:
                  - Heterosynaptic or triplet rules; plasticity-driven structural triggers (future extension).

                  Acceptance:
                  - Correctness + performance tests pass; documentation includes usage example.
## 3. Lifecycle Timeline (Educational Walkthrough)
Supplies a stage‑wise state machine perspective enabling contributors to reason about invariants (e.g., phenotype immutability within a training epoch) and side‑effects (cache invalidation boundaries). Each numbered stage defines clear preconditions and postconditions, reducing coupling: reproduction only manipulates symbolic genotype; rebuild regenerates material state; morphogenesis edits runtime graph under strict budget guards. This segregation eases targeted profiling and correctness auditing.

                  Objectives:
                  - Provide structured introspection (metrics snapshots, developmental trace, module lineage) with opt‑in overhead.
                  - Expose stable JSON schemas for external tooling.

                  Key Tasks:
                  1. Metrics aggregator API: `collectModuleMetrics(net)` returns typed array views / plain objects.
                  2. Developmental trace structure: sequence of events with timestamps, hashes, delta summaries.
                  3. Export APIs: `exportTrace()`, `exportModuleReport()`, embed subset into ONNX metadata (if feature present) under reserved namespace.
                  4. Lazy allocation design: allocate buffers only on first enable call; free or mark reusable when disabled.
                  5. Tests: allocation counts via instrumentation harness; schema validation against JSON schema file.

                  Metrics & Exit Criteria:
                  - Disabled path adds zero additional retained objects (heap diff baseline).
                  - Trace export round‑trip size overhead <5% of serialized genotype for 100 events (document).

                  Risks:
                  - Telemetry misuse in hot loop causing overhead → guidance docs + runtime warning when sampling frequency too high.

                  Deferred:
                  - Live streaming / websocket visualizer.

                  Acceptance:
                  - Introspection APIs stable & documented; performance invariants verified.
| 2       | Development Pass (Static)       | Building phenotype (Phase 1–2 implementation)                | Apply rules in priority order (replicate, symmetry, hierarchy, differentiate) to expand virtual node set & module tags | Interim developmental trace entries |
| 3       | Indirect Connectivity Synthesis | After rules applied; before instantiating physical network   | Evaluate CPPN(s) selectively over candidate coordinate pairs (sparse sampling) to decide edges + initial weights + metadata | Edge candidate list (lazy) |
| 4       | Phenotype Materialization       | Edge candidates + nodes prepared                             | Instantiate pooled `Node` / `Connection` objects; pack into slabs; apply activation/plasticity traits | Runtime `Network` |
                  Objectives:
                  - Introduce full hyper‑aware mutation & recombination operators with innovation tracking and speciation distance extensions.
                  - Maintain evolutionary performance (throughput) within acceptable overhead bounds.

                  Key Tasks:
                  1. Mutation operators: add/remove rule, tweak rule params, adjust priority/probability, CPPN weight perturb, CPPN topology add‑layer/add‑node, substrate scale/rotation adjust.
                  2. Crossover implementation: multi‑family alignment (rules, CPPNs, substrate modifiers) with blending modes & conflict resolution.
                  3. Innovation tracking: extend registry to assign IDs to new rule signatures & CPPN topology changes.
                  4. Speciation metric: weighted combination of (rule hash edit distance, CPPN topology distance, averaged CPPN weight cosine distance, substrate modifier delta) + existing compatibility coefficients.
                  5. Population loop integration: hyper mode branch using new reproduction path; preserve classic mode unaffected.
                  6. Cache reuse: precompute parent phenotype hashes, reuse adjacency caches when structural invariants retained.
                  7. Reproduction tests & correctness: determinism under seeded RNG, invalid genome rejection, budget enforcement.
                  8. Performance test: evolving small population (e.g., 50) for N generations measuring evals/sec vs baseline NEAT.

                  Metrics & Exit Criteria:
                  - Throughput ≥ 60% of baseline NEAT at comparable population sizes (document conditions).
                  - Speciation maintains diversity (no single species >80% population after burn‑in unless directed).
                  - Crossover failure (invalid child) rate <5% (above triggers validation tuning).

                  Risks:
                  - Overly punitive distance causing species fragmentation → dynamic coefficient adjustment algorithm.
                  - Genome bloat through blended duplication → complexity budgets enforced pre‑speciation assignment.

                  Deferred:
                  - Multi‑parent crossover; adaptive crossover operator selection.

                  Acceptance:
                  - End‑to‑end evolutionary run stable; diversity & performance metrics recorded.

Unlike classic NEAT where crossover aligns by innovation numbers for direct connection genes, Hyper MorphoNEAT must align heterogeneous gene families:
1. Developmental Rules
                  Objectives:
                  - Demonstrate linear (or sublinear) memory growth with active connections and acceptable build/morph latencies at large scale.
                  - Validate absence of memory leaks under churn (grow/prune cycles).
                  - Produce public guidance (tuning flags, thresholds) prior to general availability.

                  Key Tasks:
                  1. Synthetic benchmark suite: varying substrate sizes, sparsity thresholds, morph cycle frequencies.
                  2. Long‑run churn test: repeated morph cycles (e.g., 10k iterations) measuring pool high‑water marks & GC stabilized memory.
                  3. Profiling: identify hottest functions (CPPN eval, rule pass) and micro‑optimize (loop unrolling, typed array reusage) if >30% total time each.
                  4. Memory accounting: compute bytes/active connection (including pools) vs baseline; break down into slabs, metadata, plasticity extras.
                  5. Regression thresholds integrated into CI (fail if build time or memory exceeds stored baseline by >10%).
                  6. Documentation: produce scaling appendix with empirical curves (edges vs memory, edges vs build time).

                  Metrics & Exit Criteria:
                  - Peak bytes per active connection within target (establish numeric after Phase 5 measurement).
                  - No upward trend in pool size after churn plateau (slope ~0 over final 20% iterations).
                  - Build + morph latency percentiles (p95) documented and acceptable for release goals.

                  Risks:
                  - Benchmark instability across environments → pin Node.js version & isolate CPU scaling (single thread) for CI.
                  - Hidden fragmentation in pooled arrays → implement periodic compaction or sentinel leak detection.

                  Deferred:
                  - GPU/WebGPU fast path; advanced compression.

                  Acceptance:
                  - Scaling report published; CI thresholds enforced; release readiness sign‑off.
|-------------|------------------------|---------------------|-----------------------|
| Rule        | Innovation ID or (kind + canonical param signature hash) | Same kind & equal normalized params | Uniform pick or parameter-wise blend |
| CPPN        | Innovation ID (per layer addition) + topology hash        | Identical layer counts + activation sequence | Weight crossover (per-weight uniform or arithmetic) |
| Substrate Modifier | Modifier type (e.g., scale, rotation) + axis | Same type & axis | Average numeric params; random tie-break on enums |
| Module Tag  | Module ID (if shared ancestry) | ID equality | Inherit or merge (union of roles) |

Unmatched (disjoint / excess) genes: inclusion probability biased toward fitter parent (like NEAT) but capped to avoid bloat.

#### 3.1.2 Rule Parameter Blending
Argues for continuous parameter interpolation to buffer offspring against abrupt fitness cliffs introduced by discrete rule parameter jumps (e.g., replicate.times from 1→3). Blending supports *semantic continuity*: small α adjustments generate proportionally moderate structural differences after development, smoothing the adaptive landscape and improving combined efficacy of mutation + crossover.
For numeric params we can apply *biased arithmetic crossover* (BAC):
```
childParam = α * paramA + (1-α) * paramB,  α ~ U(0,1) (optionally biased toward fitter)
```
Boolean / categorical: coin flip or frequency-based if more than 2 parents (future multi-parent reproduction).

#### 3.1.3 CPPN Weight Crossover
Characterizes crossover operator choice as shaping the exploration–stability frontier: uniform promotes high exploratory variance (diversity of micro‑patterns), arithmetic maintains macro‑structural coherence, and SBX simulates sampled interpolation with controllable spread parameter η. Operator selection can be meta‑optimized via telemetry feedback (tracking post‑crossover disruption metrics) to adapt exploration pressure over evolutionary time.
Mode options (configurable):
* `uniform`: per weight pick A or B.
* `arithmetic`: `w_child = 0.5*(wA+wB)` with occasional noise injection.
* `simulated_binary_crossover (SBX)`: for more exploratory offspring.

#### 3.1.4 Innovation & History Tracking
Generalizes innovation numbers beyond direct connection genes to heterogeneous developmental and indirect encoding elements. This preserves historical distance metrics used in speciation clustering, preventing premature mixing of distinct morphogenetic strategies. Canonical hashing of normalized rule parameters minimizes spurious innovation inflation while allowing genuinely novel composite configurations to register distinct lineage identity.
We extend innovation bookkeeping: every new blended rule or CPPN structure receives a fresh innovation id; however if two parents share the same canonical hash we preserve the id to aid speciation distance continuity.

#### 3.1.5 Post-Crossover Normalization
Details a sanitation pass ensuring the offspring genotype respects global complexity budgets and semantic minimality. Redundant or shadowed rules (those whose effects are subsumed by a higher‑priority equivalent) are culled; priorities are rebalanced to avoid starvation of late but essential rule classes; and heuristic impact estimates (e.g., historical contribution deltas) guide which excess elements to discard when budget pressure is high.
After merging:
* Re-sort rules by (priority → probability → innovation).
* Deduplicate semantically equivalent rules (same normalized hash) keeping the one from fitter parent.
* Enforce complexity budget (rule count <= configurable max) dropping lowest impact (estimated contribution heuristic) first.

#### 3.1.6 Offspring Validation Pass
Defines a lightweight static checking phase performing referential integrity (regions, module tags), feasibility (symmetry without axis definition), and budget alignment validation before incurring allocation costs. Early rejection reduces wasted CPU cycles and prevents subtle runtime invariants (e.g., slab index density assumptions) from being violated in downstream materialization.
Run a *dry build* (no object instantiation) to ensure no invalid combinations (e.g., differentiate targets region that no longer exists after rule pruning). Invalid references are either re-mapped (if a similar region persists) or rule disabled.

#### 3.1.7 Pseudocode
Supplies a reference pseudocode outlining data flow and decision ordering so reviewers can validate conceptual correctness (alignment before normalization; budget enforcement post‑merge) independent of TypeScript specifics. This separation accelerates design iteration and lowers risk of misimplementation during incremental PRs.
```ts
function crossoverHyperGenotype(a: HyperGenotype, b: HyperGenotype, cfg: CrossCfg): HyperGenotype {
  const child: HyperGenotype = seedChildBase(a, b);
  // 1. Align rules
  const aligned = alignRules(a.rules, b.rules);
  for (const pair of aligned) {
    if (pair.match) child.rules.push(blendRule(pair.a, pair.b, cfg));
    else child.rules.push(selectDisjoint(pair, fitnessBias(a,b)));
  }
  // 2. Merge CPPNs
  const cppnPairs = alignCPPNs(a.cppns, b.cppns);
  child.cppns = cppnPairs.map(p => p.match ? crossoverCPPN(p.a, p.b, cfg) : preferFitter(p, a, b));
  // 3. Substrate modifiers
  child.substrateSpec = mergeSubstrate(a.substrateSpec, b.substrateSpec, cfg);
  // 4. Clean up
  normalizeRules(child.rules, cfg);
  enforceBudgets(child, cfg);
  validateChild(child);
  return child;
}
```

### 3.2 Educational Comparison: Classic NEAT vs Hyper MorphoNEAT Reproduction
Synthesizes the structural and informational expansion introduced by indirect + developmental encodings, clarifying why increased crossover complexity yields disproportionate expressive gains (large structural motifs negotiated at symbolic level). It contextualizes the trade‑off: added bookkeeping overhead versus potential for emergent macro‑regularities and smoother scaling to high node counts.
| Aspect              | Classic NEAT Crossover                         | Hyper MorphoNEAT Crossover                                      |
|---------------------|-----------------------------------------------|-----------------------------------------------------------------|
| Alignment Basis     | Innovation numbers (node/connection genes)    | Multi-family: rules, CPPNs, substrate modifiers, tags           |
| Genome Size Control | Excess/disjoint from fitter                   | Budgeted + heuristic pruning post-merge                         |
| Expressivity Change | Structural genes directly swapped             | Developmental programs blended (indirect structural consequences)|
| Weight Handling     | A/B pick for matching connection weights      | Mode selectable: uniform / arithmetic / SBX for CPPN weights    |
| Phenotype Rebuild   | Direct reconstitution                         | Regeneration via rule + CPPN re-execution (cached)              |

### 3.3 Reproduction Placement in Timeline
Explains that placing reproduction *before* any growth/prune cycle in the offspring epoch guarantees a clean separation of heritable innovation and individual lifetime adaptation. This ordering preserves analytical decomposability: fitness deltas can be partitioned into genetic vs morphogenetic contribution without confounding carry‑over artifacts.
Reproduction occurs *after* selection and *before* new morphogenesis-driven growth of the offspring. This preserves the principle that runtime morphogenesis acts on each individual's phenotype *post* genetic inheritance, avoiding entangling heritable rules with ephemeral runtime adjustments.

### 3.4 Exported Genome Mix Example (Conceptual)
Demonstrates how overlapping parental rule sets synthesize into hybrid developmental trajectories: blended replication depth adjusts module proliferation rate, symmetry alignment ensures spatial coherence, and retained differentiation preserves functional specialization. The example concretely shows genotype‑level arithmetic producing qualitatively interpretable phenotype differences post‑development, reinforcing the utility of rule interpolation.
Parent A (excerpt):
```
Rules: [ replicate(times=2), symmetry(axis=y), differentiate(region=motor, act=tanh) ]
CPPN: topology hash H1, weights WA
Substrate: dims=2, scale=1.0
```
Parent B (excerpt):
```
Rules: [ replicate(times=1), symmetry(axis=y), hierarchy(levels=2) ]
CPPN: topology hash H1, weights WB
Substrate: dims=2, scale=1.2 (stretched x-axis)
```
Child (result):
```
Rules: [ replicate(times=2 or 1→2 blended), symmetry(axis=y), hierarchy(levels=2), differentiate(region=motor, act=tanh) ]
CPPN: crossover(H1(WA,WB))
Substrate: dims=2, scale ~1.1 (blended) with normalization
```

Development then proceeds (rules executed, CPPN queried, phenotype materialized) producing a network that inherits broad symmetry + replication depth + motor differentiation.

---

---

## 4. Focusing Evolution Dynamically
Frames focus scoring as a multi‑objective prioritization heuristic balancing exploitation (error attribution, gradient contribution) with exploration (novelty deficits, structural entropy). By converting heterogeneous signals into a scalar sampling weight, the system produces a soft, continuous pressure distribution that adapts as modules mature or regress, reducing manual tuning of per‑operator probabilities.
We maintain per-module metrics:
```
ModuleMetric = {
  id, errShare, actMean, actVar, age, lastMutationIter,
  contribGradient, noveltyScore, sparsity, utilization
}
```
Focus scoring example:
```ts
function computeFocus(m: ModuleMetric) {
  return 0.35*norm(m.errShare) + 0.2*(1-norm(m.noveltyScore)) +
         0.25*norm(m.contribGradient) + 0.2*underUtilPenalty(m.utilization);
}
```
Modules chosen for: 
* Growth if high error share + high utilization.
* Diversification if low novelty + moderate error.
* Pruning if very low contribution & low utilization.

---

## 5. Example Genotype Snippets
Offers canonical genotype archetypes that encode best‑practice starting conditions (minimal symmetric scaffold, hierarchical seed) enabling reproducible baselines for benchmarking. These snippets reduce ramp‑up cost for new users and function as fixtures in regression tests ensuring deterministic development and stable performance signatures across releases.
### 5.1 Minimal Seed
Establishes a deterministic seed configuration with just enough structural variability (symmetry + shallow hierarchy) to exercise rule execution pathways while remaining analytically tractable. This baseline underpins performance profiling (isolated from higher‑order morphogenesis) and provides a control for evaluating incremental feature flags.
```ts
const geno: HyperGenotype = {
  seed: 42,
  input: 4,
  output: 2,
  rules: [
    { id:1, kind:'replicate', params:{ axis:'x', times:1 }, probability:1 },
    { id:2, kind:'symmetry', params:{ axis:'y' }, probability:0.7 },
    { id:3, kind:'hierarchy', params:{ levels:2 }, probability:1 },
  ],
  cppns: [ seedCPPN(/* architecture spec */) ],
  substrateSpec: { dims:2, inputLayout:'line', outputLayout:'line', hiddenInit:'band' },
  version:1
};
```

### 5.2 Rule Mutation Example
Showcases a parameter mutation that incrementally increases structural capacity along an existing replication axis, illustrating fine‑grained controllability of developmental expansion without introducing novel rule kinds. This emphasizes mutation granularity: small numeric shifts propagate into proportionate phenotype elaboration, aiding smooth fitness landscape traversal.
```ts
// Mutate replicate rule to increase times, enabling deeper band
mutateRule(geno, r => r.kind==='replicate', r => r.params.times = clamp(r.params.times+1, 1, 5));
```

### 5.3 Activity-Based Synaptogenesis (Runtime)
Exemplifies a morphogenesis policy that reacts to instantaneous utilization and error signals to add localized capacity, distinct from heritable genome alteration. This separation enables temporally adaptive fine‑tuning (short horizon structural adjustments) while preserving the slower evolutionary channel for consolidating successful motifs into heritable rules.
```ts
for (const region of regionsSortedByFocus(net)) {
  if (region.utilization > 0.8 && region.errorShare > 0.3) {
     attemptLocalGrowth(net, geno, region, { maxNew: 12, sparsityGuard: 0.85 });
  }
}
```

---

## 6. Indirect Connectivity Example
Explains how coordinate‑based CPPN queries transform geometric relations (relative displacement, distance) into correlated weight patterns and probabilistic connectivity masks. This yields structured sparsity (e.g., banded, radial, or mirrored motifs) whose regularity enhances parameter sharing and reduces overfitting compared to unstructured random sparse graphs at equal edge budgets.
Given node coordinates `(x_i,y_i)` and `(x_j,y_j)`, the CPPN input vector:
```
v = [x_i, y_i, x_j, y_j, |x_i-x_j|, |y_i-y_j|, dist, 1]
```
CPPN output channels (example):
* `o0`: raw weight value
* `o1`: connection mask probability (threshold)
* `o2`: plasticity rate hint
* `o3`: symmetry tag (for future tying)

Edge realized if `sigmoid(o1) > maskThreshold`. Weight = `scale(o0)`. Optional plasticity metadata attached only if feature flag.

---

## 7. Morphogenesis Event Cycle (Detailed Example)
Documents the order and conditionality of runtime adaptation actions (prune → grow → rebuild), making explicit the invariants (metrics snapshot immutability during a cycle, deferred rebuild) that guard against race conditions and inconsistent state. This trace form facilitates reproducibility and pedagogical walkthroughs.
```
// Called after each training epoch
onEpochEnd(net, geno, stats) {
  const metrics = collectModuleMetrics(net, stats);
  const focusOrder = rankModules(metrics);
  for (const m of focusOrder) {
    if (shouldPrune(m)) pruneModuleEdges(net, m, { keepFraction:0.6 });
    else if (shouldGrow(m)) growModule(net, geno, m, { strategy:'replicate_path' });
  }
  if (genoMutated) rebuildPhenotype(geno, net, { reuse: true });
  recordTrace(metrics, actionsTaken);
}
```

---

## 8. Educational Comparison Summary
Consolidates distinguishing dimensions (encoding granularity, growth triggers, scalability levers, plasticity integration) into a comparative matrix to contextualize Hyper MorphoNEAT’s hybrid positioning. The summary aids evaluators in mapping application constraints (e.g., need for extreme scaling, interpretability) to algorithm choice.
| Aspect                 | Classic NEAT                        | HyperNEAT                | Hyper MorphoNEAT                                            |
|------------------------|-------------------------------------|--------------------------|-------------------------------------------------------------|
| Encoding               | Direct genes (nodes/conns)          | Indirect (CPPN)          | Hybrid: rules + CPPN + runtime growth                      |
| Growth Trigger         | Genetic mutation only               | Genetic (CPPN changes)   | Genetic + runtime morphogenesis + focus policies           |
| Scalability Mechanism  | Gradual complexification            | Geometric regularities   | Deferred materialization + region focus + sparsity budgets |
| Plasticity             | (Optional) weight updates           | Not central              | Built-in local synaptogenesis + Hebbian optional           |
| Speciation             | Yes                                 | Typically via CPPN genes | Extended to rules + CPPN + module signatures               |
| Pruning                | Limited / manual                    | Not primary              | Integrated cyclical prune/regrow + memory guards           |

---

## 9. Educational Hooks & Introspection
Justifies first‑class introspection: complex indirect encodings risk opacity without structured tracing and metric surfacing. By instrumenting genotype→phenotype lineage, module utilization, and morph event deltas under optional flags, the system supports hypothesis‑driven debugging, comparative ablation studies, and educational visualization without imposing default overhead.
Planned user-facing helpers:
* `hyper.inspectGenotype(geno)` → prints rule list, CPPN summaries.
* `hyper.trace()` → chronological list of developmental & morph actions.
* `hyper.moduleReport(net)` → table of metrics (utilization, error share, sparsity, age).
* `hyper.replay(traceLog)` → reproduce growth decisions (determinism demo).

---

## 10. Alignment With Implementation Phases
Articulates how conceptual pillars (indirect encoding, developmental rules, runtime morphogenesis, plasticity, telemetry, evolutionary integration) are deliberately staged to minimize compounded uncertainty. Each phase establishes a stable substrate (e.g., deterministic genotype regeneration) before layering on adaptive complexity, reducing confounding when performance regressions appear.
| Concept Section                 | Implementation Phase(s) |
|---------------------------------|-------------------------|
| Seed Genotype & Substrate       | 0–1                     |
| Replication / Symmetry Rules    | 2                       |
| Indirect CPPN Connectivity      | 3                       |
| Runtime Morphogenesis Hooks     | 4                       |
| Plasticity Variables            | 5                       |
| Telemetry / Trace Export        | 6                       |
| Speciation Extensions           | 7                       |
| Scaling Validation / Memory Budgets | 8                   |

---

## 11. Future Educational Enhancements
Enumerates future pedagogical tooling (interactive developmental replays, rule mutation sandboxes, lineage visualization) intended to lower cognitive barriers for newcomers and support empirical methodology (e.g., side‑by‑side growth trajectory comparison). These extensions act as force multipliers for community experimentation and reproducibility.
* Interactive visualization: animate rule application passes.
* Module lineage tree export (GraphViz / JSON).
* “What-if” sandbox: apply hypothetical rule mutation and preview delta in nodes/edges before committing.
* Tutorial notebooks: build from seed → mid complexity → large modular brain.

---

> NOTE: All conceptual elaborations above are descriptive; actual code remains gated behind flags and phases below.

---

## High-Level Goals
Links biologically inspired constructs (symmetry, differentiation, plasticity) to concrete engineering KPIs (sample efficiency, scaling curvature, memory per effective edge) to ensure aesthetic analogies are instrumented and falsifiable. This guards against ornamental complexity by demanding metric justification for each added mechanism.

1. Add a compact Evo‑Devo genotype layer that can generate / regenerate phenotypic `Network` graphs deterministically.
2. Support CPPN‑driven structural & weight pattern generation (indirect encoding) with caching for large substrates.
3. Introduce morphogenesis (runtime growth/prune rules) integrated with existing pruning + connection pooling.
4. Maintain or reduce memory per active connection via slab packing + sparsity while enabling growth to millions of edges incrementally.
5. Provide introspection (modules, regions, developmental lineage) without heavy overhead when disabled.

---

## Current Baseline (What We Leverage)
Surveys core infrastructural assets (connection pooling, slab packing, pruning logic, deterministic RNG) leveraged to accelerate development and reduce risk. Emphasizing reuse clarifies that innovation is concentrated in encoding and adaptive control layers, not low‑level numerical plumbing.

Already present & reused:
- `Connection.acquire/release` pooling.
- Packed connection slab (`network.slab.ts`) + activation array pool.
- Configurable pruning (`network.prune.ts`) and sparsity targeting.
- Deterministic RNG snapshot / restore.
- Multi‑optimizer update paths in `Node`.
- NEAT mutation operators (can be extended to developmental rule mutation).

Gaps to fill:
- No genotype <-> phenotype separation beyond classic NEAT gene lists.
- No spatial substrate abstraction or coordinate system.
- No CPPN module.
- No runtime growth triggers aside from pruning & add‑node mutation.
- No hierarchical/module metadata structure.

---

## Module Layout (Proposed New Files / Folders)
Defines a modular namespace that enforces separation of concerns: genotype logic isolated from runtime morph policies, CPPN generation segregated from phenotype materialization, and telemetry decoupled via lazy hooks. This isolation simplifies dependency analysis and reduces inadvertent cross‑layer coupling.

```
src/hyper/
  genotype.ts              // Core Evo-Devo genotype (rule list, CPPN refs, seeds)
  developmentalRules.ts    // Rule interfaces + execution engine
  cppn/
    cppn.ts                // Minimal differentiable / evolvable CPPN (reuse existing activations)
    compiler.ts            // Optional fast-path compilation to slab weights
  substrate.ts             // Spatial substrate (coords, regions, symmetry helpers)
  morphogenesis.ts         // Runtime growth & pruning coordinator
  phenotypeBuilder.ts      // Builds a Network from genotype + substrate
  telemetry.ts             // Lightweight hooks, lazy when disabled
  serialization.ts         // Genotype (not full Network) persistence
  mutation.ts              // Mutations specific to rules / CPPN structure
  spec.md                  // (Design notes, constraints)
```

All additions are additive; existing APIs remain stable until an eventual major version.

---

## Data Contracts (Initial Draft)
Introduces stable interface nuclei enabling early consumer code (tests, tooling) to target contracts while internal algorithms iterate. Early formalization also enables forward‑compatible serialization (version tagging, optional field evolution) and eases future compression or remote execution strategies.

Genotype core:
```ts
interface DevelopmentalRule {
  id: number;
  kind: 'replicate' | 'differentiate' | 'symmetry' | 'hierarchy' | 'prune_hint' | 'densify_region';
  params: Record<string, number | string | boolean>;
  probability?: number;           // execution probability in a pass
  priority?: number;              // ordering
  enabled?: boolean;
}

interface HyperGenotype {
  seed: number;
  input: number;
  output: number;
  rules: DevelopmentalRule[];
  cppns: CPPNGene[];              // Each produces pattern(s)
  substrateSpec: SubstrateSpec;   // Dimensions, coordinate frames
  version: 1;
}
```

Phenotype build call:
```ts
buildPhenotype(geno: HyperGenotype, opts): Network; // Reuses Connection pooling + slab
```

Runtime morphogenesis callback hook signature:
```ts
type MorphEvent = 'epochEnd' | 'stagnation' | 'memoryPressure' | 'externalSignal';
type MorphogenesisHook = (net: Network, ctx: { event: MorphEvent; metrics: any }) => void;
```

---

## Phased Implementation Plan
Specifies an incremental delivery roadmap where each phase yields a self‑contained, benchmark‑able capability with explicit rollback boundaries. This staging supports empirical validation (performance, memory) and confines failure domains—problems in morphogenesis (Phase 4) cannot corrupt genotype determinism established in earlier phases.

### Phase 0 – Groundwork (Low Risk / Enablers)
Focuses on infrastructural enablement: feature flags for conditional compilation, placeholder interfaces for type stability, and sentinel tests ensuring future expansions do not regress baseline construction pathways. No algorithmic semantics change in this phase, de‑risking the branch point.
Goal: Create scaffolding without behavior changes.
Steps:
1. Add `src/hyper/` folder with placeholder `genotype.ts`, `developmentalRules.ts`, `substrate.ts` exporting empty interfaces + TODO comments.
2. Add feature flag to `config` (e.g. `config.enableHyper = false`).
3. Add unit test stubs verifying import does not throw.
Acceptance: Build + tests unchanged; tree includes new folder.

### Phase 1 – Genotype & Substrate Core
Delivers reproducible genotype→phenotype translation with hashing and serialization hooks so subsequent adaptive layers (rules, CPPNs) inherit a verifiable foundation. Determinism here is pivotal for isolating non‑deterministic variance sources in later performance analyses.
Steps:
1. Implement `HyperGenotype` structure & factory (`createInitialGenotype(input, output, seed)`).
2. Implement minimal `substrate.ts` with coordinate assignment for input/output nodes (e.g., 1D normalized positions).
3. Implement deterministic rebuild to a trivial `Network` (linear fully‑connected input→output) via `phenotypeBuilder.ts`.
4. Add serialization for genotype only (`toJSONGenotype`, `fromJSONGenotype`).
5. Add tests: round‑trip genotype + phenotype equivalence vs direct Network baseline.
Acceptance: Can build network from genotype deterministically, coverage >= existing threshold for new lines.

### Phase 2 – Developmental Rule Engine (Static Application)
Confirms that static developmental rule application yields predictable, parameter‑controlled structural transformations (node/edge count scaling laws) before introducing temporal dependencies or indirect connectivity. This isolates semantic bugs (e.g., symmetry duplication drift) early.
Steps:
1. Define rule execution ordering & priority.
2. Implement rule kinds: `replicate` (split connection region), `symmetry` (mirror coordinates), `hierarchy` (tag modular region ids).
3. Extend phenotype builder to apply rules in passes (no runtime dynamics yet).
4. Provide mutation operators for enabling/disabling rules & parameter tweak.
5. Tests: Rule application changes node & connection counts as expected; determinism with identical seeds.
Acceptance: Complexity scaling via rules validated on synthetic tests.

### Phase 3 – CPPN Integration (Indirect Encoding)
Adds CPPN‑based pattern synthesis to decouple connection enumeration from explicit genome length, enabling graceful scaling to large substrates via procedural generation plus sparsity thresholds. Performance and caching instrumentation ensure tractability before layering dynamic growth.
Steps:
1. Implement lightweight CPPN (feedforward multi‑layer perceptron) using existing `Node` activation functions (no recursion) in `cppn/cppn.ts`.
2. Define `CPPNGene` (architecture description + weights array reused via typed arrays for memory efficiency).
3. Add mapping: for each candidate pair (i,j) the CPPN queries `(x_i, y_i, x_j, y_j, distance, bias)` to produce weight or mask.
4. Introduce sparsity threshold (e.g. absolute output < t -> skip connection) feeding into existing slab rebuild.
5. Cache generated adjacency (fingerprint genotype + substrate + threshold) to avoid recomputation across evaluations.
6. Tests: Weight pattern symmetry; threshold controls edge count.
Acceptance: Indirect generation path within 1.5× baseline time for medium nets; memory overhead bounded (document).

### Phase 4 – Morphogenesis (Runtime Growth & Pruning Hooks)
Installs runtime evaluators that adjust structure in response to short‑horizon performance signals, establishing a middle adaptation timescale between gradient updates and generational evolution. Policies are bounded by memory/growth budgets to preserve predictability.
Steps:
1. Implement `morphogenesis.ts` maintaining counters: activity, error trend, stagnation iterations.
2. Expose `registerMorphHook` on Network (only active if hyper mode enabled) storing a list of `MorphogenesisHook`.
3. Provide built‑in policies: `activity_expand`, `low_contrib_prune` (leveraging existing pruning API), `module_split` (clone subgraph tag).
4. Introduce growth budget guard referencing memory plan (max connections & nodes; triggers forced pruning).
5. Tests: Controlled mock metrics trigger expected growth/prune actions (assert node/connection deltas).
Acceptance: Hooks fire without breaking forward / backward passes; pool usage stable.

### Phase 5 – Plasticity & Local Adaptation
Augments standard optimizer updates with local activity‑dependent modifications (Hebbian/anti‑Hebbian) providing rapid micro‑adaptation where gradient signals may be sparse or noisy, potentially improving credit assignment in deep or recurrent motifs.
Steps:
1. Add optional per‑connection short‑term plasticity variables (reuse existing `Connection` extension fields; ensure pooling reset).
2. Implement Hebbian update option executed post‑activation batch (scales small typed array of weight deltas, not reusing `propagate`).
3. Provide configuration gating to avoid overhead when disabled.
4. Tests: Hebbian updates modify weights in isolation; disabled path adds near‑zero overhead (<5% baseline runtime for small net).
Acceptance: Plasticity coexists with backprop/evolution.

### Phase 6 – Telemetry & Introspection (Lazy)
Ensures instrumentation cost is pay‑as‑you‑go: metric buffers and trace arrays allocate only when enabled, and hot paths retain branch‑predictable checks. This design encourages pervasive measurability without penalizing production performance.
Steps:
1. `telemetry.ts` gathers per‑module stats (avg activity, sparsity) only when `enableTelemetry` flag set.
2. Add `network.exportDevelopmentalTrace()` returning rule application + growth events.
3. Add optional ONNX metadata section embedding module tags.
4. Tests: Telemetry off -> no extra allocations (assert via heap sampling harness later). Basic JSON output validated.
Acceptance: Non‑intrusive diagnostics present.

### Phase 7 – Evolutionary Integration (Mutation + Reproduction)
Integrates multi‑family crossover and expanded distance metrics so population dynamics can exploit recombination benefits (innovation combination, deleterious mutation masking) while maintaining ecological niche protection for divergent developmental strategies.
Steps:
1. Extend existing NEAT mutation registry with hyper‑aware mutations (add/remove rule, mutate CPPN weight/topology, adjust substrate scale, modify rule probabilities/priorities).
2. Implement `crossoverHyperGenotype(a,b,cfg)` with alignment & blending strategies (rules, CPPNs, substrate modifiers) + validation pass.
3. Add speciation distance components for genotype differences (rule vector hash distance, CPPN topology/weight signature, substrate modifier delta).
4. Integrate reproduction into population loop: select parents, generate offspring genotype(s), regenerate phenotype using cached artifacts.
5. Fitness evaluation optionally rebuilds phenotype each generation if genotype mutated or crossover occurred (cache reuse across offspring clones).
6. Tests: 
  * Crossover determinism under fixed RNG.
  * Speciation separation on crafted genotype pairs.
  * Offspring validity (no dangling regions).
  * Performance smoke test with mixed mutation + crossover.
Acceptance: End‑to‑end evolve run with hyper genotype + reproduction finishes within target time budget; offspring functional parity tests pass.

### Phase 8 – Scale & Stress Validation
Subjects the full stack to stress conditions (million‑edge potentials, churn cycles) to confirm memory, time, and determinism invariants hold. Results gate public API stabilization and inform tuning defaults (thresholds, budgets).
Steps:
1. Benchmark script generating ~1M potential connections (sparse) comparing memory vs classic path.
2. Stress test morphogenesis growing then pruning to ensure no memory leaks (pool sizes stable).
3. Document scaling heuristics & recommended flags.
4. CI job thresholding memory & run time.
Acceptance: Peak memory per active connection meets target (<X bytes; finalize after measurement Phase 6).

---

## Risk & Mitigation Summary
Transforms diffuse architectural risks into an actionable ledger, pairing each risk with concrete mitigation levers (caching, budgets, feature flags). This supports proactive monitoring and simplifies post‑mortem attribution if regressions emerge.
| Risk | Mitigation |
|------|------------|
| Rebuild overhead for large CPPN substrates | Caching + incremental diff generation |
| Memory blowup from plasticity state | Optional feature; pooled typed arrays with reuse |
| Rule explosion causing combinatorial growth | Global complexity budget + rule priority throttle |
| API instability | Feature-flag entire hyper layer until Phase 6 | 

---

## Incremental PR Sequencing (Granular Checklist)
Breaks delivery into reviewable micro‑increments so semantic drift or performance regressions are localized; each PR carries its own acceptance tests and plan diff, forming an auditable evolution trail of the design document itself.
1. PR1: Scaffolding + config flag + empty tests.
2. PR2: Genotype + basic phenotype builder.
3. PR3: Developmental rules (replicate/symmetry) + tests.
4. PR4: CPPN core + indirect edge generation.
5. PR5: Morphogenesis hooks + activity metrics.
6. PR6: Plasticity (Hebbian) + gating.
7. PR7: Telemetry + export trace.
8. PR8: Evolution integration (mutation + speciation extension).
9. PR9: Scaling benchmarks + docs.
10. PR10: Stabilization & API doc examples.

Each PR keeps surface area small, adds tests, and updates this plan (append CHANGELOG section).

---

## Example (Future) User API Sketch (Post Phase 5)
Provides a provisional user‑facing construct to validate naming consistency, configuration surface minimality, and composability with existing library patterns before hardening interfaces post Phase 6.
```ts
import { createHyper } from 'neataptic-ts/hyper';

const hyper = createHyper({
  input: 16,
  output: 4,
  rules: [ { kind: 'replicate', params: { times: 2 } } ],
  cppn: { layers: [8,8], activation: 'tanh' },
  enableMorphogenesis: true,
  plasticity: { mode: 'hebbian', rate: 1e-3 }
});

const net = hyper.build();
// training loop ... hyper.maybeMorph(eventMetrics)
```

---

## Acceptance & Success Metrics
Anchors success to quantifiable, automatable metrics (build time ratio, memory per active connection, deterministic hash reproducibility) ensuring progress narratives are evidence‑based; thresholds provide regression guards in CI.
| Metric                            | Target (initial)                               | Rationale                |
|----------------------------------|-------------------------------------------------|--------------------------|
| Phenotype build time (50k edges) | < 1.2× baseline Network construction            | Maintain responsiveness |
| Memory per active connection     | TBD after measurement (< baseline by Phase 8)   | Scale to big nets        |
| Morph hook overhead (disabled)   | < 1% runtime                                    | Pay only when used       |
| Deterministic rebuild hash match | 100%                                            | Reproducibility          |

---

## Future Extensions (Post v1)
Signals strategic extension vectors (symbolic modules, GPU path, compressed serialization) to align community contributions and prevent ad‑hoc divergence once the core is stable.
- Spatially aware neuro-symbolic modules.
- GPU/WebGPU execution path using slab + typed buffers.
- Compressed serialization (delta-coded genotype + pattern seeds).

---

## Maintenance Notes
Enumerates non‑negotiable constraints (optional fields, isolation of hyper namespace, pooled state hygiene) that reviewers should enforce to avoid gradual entropy accumulation and maintain predictable memory/performance profiles.
- Keep `hyper/` isolated; do not import from it inside baseline `Network` unless flag enabled to avoid bundle bloat.
- All added fields on `Network` must be optional and lazily allocated.
- Pool any added per-connection arrays (plasticity, tags) via indexed side buffers.

---

End of refined plan.
