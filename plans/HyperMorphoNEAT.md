# Hyper MorphoNEAT (draft)

Hyper MorphoNEAT sits between HyperNEAT and a small developmental language: it keeps HyperNEAT's compact spatial patterns, but adds a deterministic rule layer and runtime morph policies so evolution can both propose macro motifs and let the phenotype adapt locally during lifetime. It excels when geometry + lifelong adaptation + scalability are all important.

Hyper MorphoNEAT is deliberately pragmatic: introduce an indirect, rule‑driven layer that remains opt‑in and deterministic, provide lightweight runtime morph policies that act only when enabled, and preserve the current NEAT/Network behavior when the feature flag is off.

What follows is a phased implementation plan and rationale. Each phase lists safe, reviewable changes, validation checks (determinism, small smoke tests, typechecks), and explicit rollback boundaries so reviewers can validate correctness and performance before accepting further complexity.

---

## Conceptual Expansion: “From Proto‑Brain to Adaptive Cortex”

Hyper MorphoNEAT reframes topology growth as a staged, deterministic engineering pipeline rather than ad‑hoc structural mutation. The pipeline is intentionally compositional: small, well‑specified rule primitives and compact pattern generators (CPPNs) produce large, traceable phenotypes via repeatable passes. That design lets evolution operate on a concise, high‑leverage symbolic layer (the genotype) while runtime morph policies make bounded, local trade‑offs during an individual's lifetime (the phenotype).

Practical design principles applied throughout this plan:

- Opt‑in and isolated: hyper features live under guarded flags and an isolated namespace to avoid regressions.
- Deterministic by default: genotype + seed → canonical phenotype; canonical hashing and stable ordering are required for reproducible experiments.
- Budgeted growth: all expansions obey explicit complexity caps (nodes, edges, memory) and are reversible or roll‑backable for safety.
- Lazy instrumentation: telemetry and extra buffers allocate only when enabled to preserve baseline performance.
- Traceability: every developmental action carries ancestry/trace metadata to help debugging and analysis.

Analogy to biological development (engineered mapping):

| Stage                     | Biological Inspiration            | Hyper MorphoNEAT Engineering Analog                                                                                                                      |
| ------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embryonic Seed            | Few stem cells                    | Minimal, deterministic genotype with input/output anchors                                                                                                |
| Patterning Gradients      | Morphogens, HOX genes             | CPPN fields + compact developmental rules that assign coordinates, tags and role metadata                                                                |
| Proliferation             | Cell division                     | Deterministic `replicate` rules that expand regions or bands under growth budgets                                                                        |
| Differentiation           | Neurons specialize                | Rules assign activation families, plasticity profiles, and gating behaviors (explicit, testable traits)                                                  |
| Axon Guidance             | Growth cones follow gradients     | Spatial CPPN thresholds, local heuristics and wiring‑cost bias producing sparse, locality‑aware connectivity                                             |
| Pruning & Refinement      | Synaptic pruning                  | Activity and contribution metrics drive removal; pruning prefers long or low‑contribution inter‑module links under budget                                |
| Lifelong Plasticity       | Hebbian and structural remodeling | Optional, gated plasticity + morph hooks that adjust weights and (rarely) local structure during runtime                                                 |
| Wiring cost (engineering) | Metabolic / material cost         | Explicit per‑genotype wiring knobs (counts, length, inter‑module penalties) used by CPPNs, morph policies and selection to favor compact, modular wiring |

The remainder of this section spells out how these mappings are realized as deterministic passes, safe runtime hooks, and explicit validation checks. See the next subsection for the core "rule‑first" rationale and the planned deterministic invariants that must hold across phases.

### Core Idea

Hyper MorphoNEAT adopts a rule‑first engineering strategy: treat the genotype as a compact, declarative program and execute it in deterministic, well‑scoped passes to produce a runtime phenotype. Evolution operates on concise symbolic elements (rules, CPPNs, substrate modifiers) while runtime morph policies perform bounded, local adjustments under explicit budgets. This separation keeps the heritable search space small and interpretable, and makes large structural effects reproducible and debuggable.

Pipeline (high level):

1. Substrate: deterministically assign coordinates and region tags for inputs/outputs/hidden slots.
2. Rule passes: apply prioritized, deterministic rules (replicate, symmetry, hierarchy, etc.) to expand a virtual node list and attach ancestry/trace metadata.
3. CPPN evaluation: procedurally score candidate pairs (weight, mask, meta) with cost‑aware thresholding; cache adjacency results keyed by canonical genotype/substrate hashes and wiring prefs.
4. Materialize: instantiate pooled Node/Connection slabs from the realized adjacency list; apply activation/plasticity traits.

Deterministic invariants & safety guarantees:

- Reproducibility: genotype + seed → canonical phenotype (stable ordering, canonical JSON/hash).
- Idempotence: repeated builds produce identical node/edge orderings and hashes unless the genotype or seed changes.
- Budget enforcement: all expansion steps respect explicit caps (maxNodes, maxEdges, memoryBudget) and are reversible or roll‑backable.
- No global side effects: imports and builds must not mutate global runtime state when hyper mode is disabled.
- Lazy diagnostics: telemetry, traces, and per‑connection extras allocate only when enabled.

Minimal pseudocode example: (canonical snippet retained in the Implementation section below — see "Minimal pseudocode example")

> NOTE: To avoid duplicate snippets and drift, keep the `buildPhenotype` example only in the Implementation section; this line acts as a local pointer.

These rules keep the implementation auditable and allow later phases (morph hooks, plasticity, evolution integration) to build on a deterministic, testable foundation.

### Genotype vs Phenotype Layering

This document maintains a single canonical "Layer responsibilities" section in the Implementation area (see the "Layer responsibilities" and "Pipeline (high level)" headings later). To avoid duplication and maintenance drift, readers should consult that canonical section for details on genotype/phenotype separation, invariants, and practical guidelines.

### Dynamic Evolution Focus

In Hyper MorphoNEAT we treat mutation targeting as a controlled allocation problem: rather than applying uniform mutation pressure across the entire phenotype, we compute per‑region focus scores from multiple, complementary signals and use those scores to bias where evolutionary operators and runtime morph actions apply. The intention is practical and measurable: concentrate structural edits (rule mutations, local growth/prune, CPPN perturbations) where they are most likely to reduce task error or increase useful diversity, while cooling mature, stable regions to avoid destructive churn.

Key signals

- errShare: fraction of total error attributed to the region (backprop attribution, gradient magnitude, or surrogate credit).
- utilization: fraction of active units / firing rate baseline (indicates capacity in use).
- contribGradient: aggregate magnitude of gradients flowing through the region (proxy for learning pressure).
- noveltyScore: structural or activation novelty relative to recent history (encourages exploration).
- stabilityAge: time since last successful mutation / performance improvement (cooling factor).
- wiringMetrics: meanEdgeLength, interModuleRatio, totalConnections (used to weight wiring‑cost penalties).

Normalized focus score

- Normalize each signal to [0,1] using rolling statistics (mean/var) or robust quantiles to avoid outlier domination.
- Combine with configurable weights and a small L2 regularizer to avoid score collapse:

focusScore(region) = Σ*k w_k * norm*k(region) - w_cost * norm_wiring(region) - γ \* ||w||^2

where norm_k are per-signal normalizers, w_k are configuration weights, w_cost encodes wiring penalties, and γ stabilizes weights if learned/adapted.

Sampling & operator placement

1. Compute focusScore for all modules/regions each epoch (or lower cadence).
2. Convert scores to sampling probabilities via softmax with temperature τ to control selection sparsity:

prob(region) ∝ exp(focusScore(region) / τ)

3. Sample a small set of targets (top‑K or N draws without replacement) and apply bounded, local edits:
   - Local edits must be budgeted (maxNewEdgesPerCycle, maxNodesPerCycle) and test‑rollbackable.
   - Prefer low‑latency edits first (edge densification, small replicate); defer expensive global replays.

Safety invariants and limits

- Budget caps: every morph or mutation is rejected if it would violate maxNodes, maxEdges, or memoryBudget.
- Cooldown: a region that was edited resets a cooldown counter preventing repeated edits for N epochs.
- Dry run validation: for costly edits perform a dry‑build check (no allocation) to validate referential integrity before committing.
- Deterministic seed usage: stochastic choices in sampling use a deterministic RNG seeded from (genotype.hash + epochCounter) for reproducibility.

Practical notes for implementation & experiments

- Use online, robust normalization (e.g., exponential moving mean/std or median/MAD) to make norm_k stable across training phases.
- Start with conservative weights (favor errShare + utilization) and anneal toward novelty when stagnation persists.
- Evaluate ablations:
  - uniform vs focus sampling (measure evals/sec to convergence).
  - with/without wiringCost in the score (measure modularity Q, meanEdgeLength, task performance).
- Logging: record per‑epoch focusScore distributions, sampled regions, and before/after deltas for node/edge counts so causal effects are traceable.

Example sketch (pseudo)

```ts
const scores = regions.map((r) =>
  computeFocusScore(r, metrics, geno.wiringPreferences)
);
const probs = softmax(scores, config.focusTemperature);
const targets = sampleWithoutReplacement(
  regions,
  probs,
  config.maxRegionsPerEpoch
);
for (const t of targets) {
  if (!withinBudgets(t)) continue;
  applyLocalEditSafely(net, geno, t);
}
```

This focused allocation framework keeps Hyper MorphoNEAT’s evolutionary pressure intentional and measurable: edits are data‑driven, budget‑bounded, reproducible, and amenable to systematic ablation studies.

## Lifecycle Timeline

Supplies a stage‑wise state machine perspective enabling contributors to reason about invariants (e.g., phenotype immutability within a training epoch) and side‑effects (cache invalidation boundaries). Each numbered stage defines clear preconditions and postconditions, reducing coupling: reproduction only manipulates symbolic genotype; rebuild regenerates material state; morphogenesis edits runtime graph under strict budget guards. This segregation eases targeted profiling and correctness auditing.

| Build Step                   | Implemented In | Description                                                                                                            | Artifacts                           |
| ---------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 1. Static Development Pass   | Phase 2        | Apply rules in priority order (replicate, symmetry, hierarchy, differentiate) to expand virtual node set & module tags | Interim developmental trace entries |
| 2. Indirect Connectivity     | Phase 3        | Evaluate CPPN(s) selectively over candidate coordinate pairs (sparse sampling) to decide edges + initial weights       | Edge candidate list (lazy)          |
| 3. Phenotype Materialization | Phase 1-3      | Instantiate pooled `Node` / `Connection` objects; pack into slabs; apply activation/plasticity traits                  | Runtime `Network`                   |

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

1.  Developmental Rules
    Objectives: - Demonstrate linear (or sublinear) memory growth with active connections and acceptable build/morph latencies at large scale. - Validate absence of memory leaks under churn (grow/prune cycles). - Produce public guidance (tuning flags, thresholds) prior to general availability.

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
| Rule | Innovation ID or (kind + canonical param signature hash) | Same kind & equal normalized params | Uniform pick or parameter-wise blend |
| CPPN | Innovation ID (per layer addition) + topology hash | Identical layer counts + activation sequence | Weight crossover (per-weight uniform or arithmetic) |
| Substrate Modifier | Modifier type (e.g., scale, rotation) + axis | Same type & axis | Average numeric params; random tie-break on enums |
| Module Tag | Module ID (if shared ancestry) | ID equality | Inherit or merge (union of roles) |

Unmatched (disjoint / excess) genes: inclusion probability biased toward fitter parent (like NEAT) but capped to avoid bloat.

#### Rule Parameter Blending

Argues for continuous parameter interpolation to buffer offspring against abrupt fitness cliffs introduced by discrete rule parameter jumps (e.g., replicate.times from 1→3). Blending supports _semantic continuity_: small α adjustments generate proportionally moderate structural differences after development, smoothing the adaptive landscape and improving combined efficacy of mutation + crossover.
For numeric params we can apply _biased arithmetic crossover_ (BAC):

```
childParam = α _ paramA + (1-α) _ paramB, α ~ U(0,1) (optionally biased toward fitter)
```

Boolean / categorical: coin flip or frequency-based if more than 2 parents (future multi-parent reproduction).

#### CPPN Weight Crossover

Characterizes crossover operator choice as shaping the exploration–stability frontier: uniform promotes high exploratory variance (diversity of micro‑patterns), arithmetic maintains macro‑structural coherence, and SBX simulates sampled interpolation with controllable spread parameter η. Operator selection can be meta‑optimized via telemetry feedback (tracking post‑crossover disruption metrics) to adapt exploration pressure over evolutionary time.
Mode options (configurable):

- `uniform`: per weight pick A or B.
- `arithmetic`: `w_child = 0.5*(wA+wB)` with occasional noise injection.
- `simulated_binary_crossover (SBX)`: for more exploratory offspring.

#### Innovation & History Tracking

Generalizes innovation numbers beyond direct connection genes to heterogeneous developmental and indirect encoding elements. This preserves historical distance metrics used in speciation clustering, preventing premature mixing of distinct morphogenetic strategies. Canonical hashing of normalized rule parameters minimizes spurious innovation inflation while allowing genuinely novel composite configurations to register distinct lineage identity.
We extend innovation bookkeeping: every new blended rule or CPPN structure receives a fresh innovation id; however if two parents share the same canonical hash we preserve the id to aid speciation distance continuity.

#### Post-Crossover Normalization

Details a sanitation pass ensuring the offspring genotype respects global complexity budgets and semantic minimality. Redundant or shadowed rules (those whose effects are subsumed by a higher‑priority equivalent) are culled; priorities are rebalanced to avoid starvation of late but essential rule classes; and heuristic impact estimates (e.g., historical contribution deltas) guide which excess elements to discard when budget pressure is high.
After merging:

- Re-sort rules by (priority → probability → innovation).
- Deduplicate semantically equivalent rules (same normalized hash) keeping the one from fitter parent.
- Enforce complexity budget (rule count <= configurable max) dropping lowest impact (estimated contribution heuristic) first.

#### Offspring Validation Pass

Defines a lightweight static checking phase performing referential integrity (regions, module tags), feasibility (symmetry without axis definition), and budget alignment validation before incurring allocation costs. Early rejection reduces wasted CPU cycles and prevents subtle runtime invariants (e.g., slab index density assumptions) from being violated in downstream materialization.
Run a _dry build_ (no object instantiation) to ensure no invalid combinations (e.g., differentiate targets region that no longer exists after rule pruning). Invalid references are either re-mapped (if a similar region persists) or rule disabled.

#### Pseudocode

Supplies a reference pseudocode outlining data flow and decision ordering so reviewers can validate conceptual correctness (alignment before normalization; budget enforcement post‑merge) independent of TypeScript specifics. This separation accelerates design iteration and lowers risk of misimplementation during incremental PRs.

```ts
function crossoverHyperGenotype(
  a: HyperGenotype,
  b: HyperGenotype,
  cfg: CrossCfg
): HyperGenotype {
  const child: HyperGenotype = seedChildBase(a, b);
  // 1. Align rules
  const aligned = alignRules(a.rules, b.rules);
  for (const pair of aligned) {
    if (pair.match) child.rules.push(blendRule(pair.a, pair.b, cfg));
    else child.rules.push(selectDisjoint(pair, fitnessBias(a, b)));
  }
  // 2. Merge CPPNs
  const cppnPairs = alignCPPNs(a.cppns, b.cppns);
  child.cppns = cppnPairs.map((p) =>
    p.match ? crossoverCPPN(p.a, p.b, cfg) : preferFitter(p, a, b)
  );
  // 3. Substrate modifiers
  child.substrateSpec = mergeSubstrate(a.substrateSpec, b.substrateSpec, cfg);
  // 4. Clean up
  normalizeRules(child.rules, cfg);
  enforceBudgets(child, cfg);
  validateChild(child);
  return child;
}
```

### Educational Comparison: Classic NEAT vs Hyper MorphoNEAT Reproduction

Synthesizes the structural and informational expansion introduced by indirect + developmental encodings, clarifying why increased crossover complexity yields disproportionate expressive gains (large structural motifs negotiated at symbolic level). It contextualizes the trade‑off: added bookkeeping overhead versus potential for emergent macro‑regularities and smoother scaling to high node counts.

| Aspect              | Classic NEAT Crossover                     | Hyper MorphoNEAT Crossover                                        |
| ------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| Alignment Basis     | Innovation numbers (node/connection genes) | Multi-family: rules, CPPNs, substrate modifiers, tags             |
| Genome Size Control | Excess/disjoint from fitter                | Budgeted + heuristic pruning post-merge                           |
| Expressivity Change | Structural genes directly swapped          | Developmental programs blended (indirect structural consequences) |
| Weight Handling     | A/B pick for matching connection weights   | Mode selectable: uniform / arithmetic / SBX for CPPN weights      |
| Phenotype Rebuild   | Direct reconstitution                      | Regeneration via rule + CPPN re-execution (cached)                |

### Reproduction Placement in Timeline

Explains that placing reproduction _before_ any growth/prune cycle in the offspring epoch guarantees a clean separation of heritable innovation and individual lifetime adaptation. This ordering preserves analytical decomposability: fitness deltas can be partitioned into genetic vs morphogenetic contribution without confounding carry‑over artifacts.
Reproduction occurs _after_ selection and _before_ new morphogenesis-driven growth of the offspring. This preserves the principle that runtime morphogenesis acts on each individual's phenotype _post_ genetic inheritance, avoiding entangling heritable rules with ephemeral runtime adjustments.

### Exported Genome Mix Example (Conceptual)

Demonstrates how overlapping parental rule sets synthesize into hybrid developmental trajectories: blended replication depth adjusts module proliferation rate, symmetry alignment ensures spatial coherence, and retained differentiation preserves functional specialization. The example concretely shows genotype‑level arithmetic producing qualitatively interpretable phenotype differences post‑development, reinforcing the utility of rule interpolation.
Parent A (excerpt):

```
Rules: [ replicate(times=2), symmetry(axis=y), differentiate(region=motor, act='tanh') ]
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
Rules: [ replicate(times=2 or 1→2 blended), symmetry(axis=y), hierarchy(levels=2), differentiate(region=motor, act='tanh') ]
CPPN: crossover(H1(WA,WB))
Substrate: dims=2, scale ~1.1 (blended) with normalization
```

Development then proceeds (rules executed, CPPN queried, phenotype materialized) producing a network that inherits broad symmetry + replication depth + motor differentiation.

## Focusing Evolution Dynamically

Frames focus scoring as a multi‑objective prioritization heuristic balancing exploitation (error attribution, gradient contribution) with exploration (novelty deficits, structural entropy). By converting heterogeneous signals into a scalar sampling weight, the system produces a soft, continuous pressure distribution that adapts as modules mature or regress, reducing manual tuning of per‑operator probabilities.
We maintain per-module metrics:

```
ModuleMetric = {
  id, errShare, actMean, actVar, age, lastMutationIter,
  contribGradient, noveltyScore, sparsity, utilization,
  // wiring and modularity signals used to prefer compact/regular wiring
  meanEdgeLength?: number,
  interModuleEdgeRatio?: number,
  modularityQ?: number
}
```

Focus scoring example:

```ts
function computeFocus(m: ModuleMetric) {
  return (
    0.35 * norm(m.errShare) +
    0.2 * (1 - norm(m.noveltyScore)) +
    0.25 * norm(m.contribGradient) +
    0.15 * (1 - norm(m.modularityQ || 0)) + // prefer modules that increase modularity score
    0.15 * underUtilPenalty(m.utilization)
  );
}
```

Modules chosen for:

- Growth if high error share + high utilization.
- Diversification if low novelty + moderate error.
- Pruning if very low contribution & low utilization.

Notes:

- `meanEdgeLength` and `interModuleEdgeRatio` feed into focusScore and can increase pruning pressure on long or cross‑module links.
- `modularityQ` is used to prefer mutations/morphs that increase modular structure; it can also be used as a speciation axis.

---

## High-Level Goals

Links biologically inspired constructs (symmetry, differentiation, plasticity) to concrete engineering KPIs (sample efficiency, scaling curvature, memory per effective edge) to ensure aesthetic analogies are instrumented and falsifiable. This guards against ornamental complexity by demanding metric justification for each added mechanism.

1. Add a compact Evo‑Devo genotype layer that can generate / regenerate phenotypic `Network` graphs deterministically.
2. Support CPPN‑driven structural & weight pattern generation (indirect encoding) with caching for large substrates.
3. Introduce morphogenesis (runtime growth/prune rules) integrated with existing pruning + connection pooling.
4. Maintain or reduce memory per active connection via slab packing + sparsity while enabling growth to millions of edges incrementally.
5. Provide introspection (modules, regions, developmental lineage) without heavy overhead when disabled.
6. Provide wiring‑cost primitives and selection hooks so evolution and morphogenesis can prefer compact, modular, and regular wiring (configurable penalties and Pareto options).

---

## How to use these instructions

These instructions describe a step‑by‑step plan to implement the features of Hyper MorphoNEAT in a series of incremental, testable phases. Each section details a conceptual component, followed by a precise specification of the corresponding implementation step.

For each phase:

- Review the objectives and key tasks to understand the goals and technical requirements.
- Implement the changes in small, reviewable increments, following the specified order and guidelines.
- Validate correctness and performance at each step, using the provided acceptance criteria and tests.
- Update this plan as needed to reflect any changes or discoveries during implementation.

### Policy & contribution notes (short)

For strict rules, automated validations, and contribution guidance see the canonical repository documents:

- `.github/copilot-instructions.md` — repo-specific contributor instructions and strict rules.
- `STYLEGUIDE.md` — coding/style/test conventions (tests: single-expect rule, naming, JSDoc requirements, etc.).

Summary (local): keep changes small and reviewable, gate hyper features behind a flag, and run the repo validation suite before merging (tests, lint, typecheck, build). Use the canonical files above as source-of-truth; do not duplicate policy here.

### Minimal pseudocode example

To ground the conceptual pipeline in a concrete example, the following pseudocode illustrates the high-level function calls and data flow for building a phenotype from a genotype.

```ts
function buildPhenotype(geno, seed) {
  const substrate = layoutSubstrate(geno.substrateSpec);
  const virtualNodes = applyRules(geno.rules, substrate, { seed });
  const adjacency = evaluateCPPNSafe(
    geno.cppns,
    virtualNodes,
    geno.wiringPreferences
  );

  return materializeNetwork(virtualNodes, adjacency, { poolReuse: true });
}
```

### Layer responsibilities

To maintain a clean separation of concerns, the system is divided into two primary layers: the Genotype and the Phenotype. This division is crucial for ensuring that the genetic representation remains compact and heritable, while the materialized network can be optimized for runtime performance.

1. Genotype (persistent, small)

- Encodes rules, CPPN parameters, substrate spec, wiring preferences and version metadata.
- Must be immutable during a build pass; mutations create a new genotype object.
- Provides canonical serialization and deterministic hashing (order‑independent where appropriate).
- Small and cheap to copy/compare; used by evolutionary operators and CI checks.

2. Phenotype (materialized, transient)

- Instantiates Node/Connection objects, activation buffers, plasticity accumulators and runtime telemetry.
- May be pooled and reused across builds; must be reconstructible from the genotype + seed.
- Holds transient performance state (running statistics, per‑connection traces) that is optional and lazy‑allocated.
- Subject to explicit budgets (maxNodes, maxEdges, memoryBudget) and reversible morph operations.

### Key invariants & safety guarantees

To ensure robust and predictable behavior, the Hyper MorphoNEAT implementation must adhere to a set of key invariants and safety guarantees. These principles are designed to prevent common pitfalls in complex evolutionary systems, such as non-determinism and uncontrolled resource consumption.

- Determinism: genotype + seed → canonical phenotype (stable ordering and stable hash).
- Referential safety: no runtime phenotype object is retained as part of a genotype; genotype serialization contains only serializable fields.
- Lazy allocation: telemetry and per‑connection extras allocate only when enabled.
- Budget enforcement: all builds and morphs check and honor configured resource caps before committing structural changes.
- Side‑effect free import: importing hyper modules with the feature flag disabled must not mutate global runtime state.

### Practical guidelines

To complement the strict invariants, the following practical guidelines should be followed during development. These are best practices that will help maintain code quality, performance, and debuggability.

- Canonicalize before hashing: sort rule lists and normalize numeric fields prior to JSON/string hashing to avoid order-dependent innovation ids.
- Use shallow, copy-on‑write genotype mutations for evolutionary operators; avoid mutating arrays in place.
- Cache adjacency/CPPN outputs keyed by canonical fingerprints (genotype hash, substrate hash, wiring prefs) and invalidate conservatively on genotype changes.
- Pool ephemeral storage (slabs, activation arrays, plasticity buffers) and reset on release to avoid steady heap growth.
- Include lightweight ancestry/trace metadata in phenotype objects when telemetry is enabled; keep it out of core runtime paths otherwise.

## Memory & performance alignment

Efficient memory management and high performance are critical for evolving complex neural networks. Hyper MorphoNEAT's implementation is therefore tightly aligned with the repository's dedicated memory and performance optimization roadmap. This section summarizes the key targets and how they map to the phased implementation.

**All memory operations, feature flags, and constants within the Hyper MorphoNEAT implementation must be sourced from the `Centralized Memory Manager` defined in Phase 3.5 of the `Memory_Optimization.md` plan. This ensures architectural consistency and adherence to the pay-for-use principle.**

Hyper MorphoNEAT relies on the repository's dedicated memory roadmap; this section summarizes the concrete targets and phase mapping from `plans/Memory_Optimization.md` so implementers and reviewers share the same acceptance criteria.

- Bytes / active connection: target ≈ 64 bytes per active connection (empirical baseline and Phase 1/3 goals in `plans/Memory_Optimization.md`). Use slab packing, bit‑flags, and optional pay‑for‑use slabs (gain, plasticity) to achieve this.
- Adjacency / phenotype caching: aim for an adjacency cache hit ratio > 70% in repeated evaluation scenarios (see Memory plan L7 and Hyper phases). Budget adjacency cache bytes to remain small (goal: ~+6 bytes amortized per active connection when the cache is effective).
- Rebuild variance: p95/median rebuild time ratio target < 2.5× (see Memory plan Phase targets for rebuild variance and slab reuse guidance).
- Plasticity side‑buffers and gains: keep optional side‑buffers < 8 bytes per plastic connection when enabled (Memory plan Phase 5/Hyper-specific targets).

Mapping to Memory phases (canonical references)

- Phase 0: Baseline instrumentation and dist‑only snapshots — use this to establish the Hyper baseline before enabling morphogenesis.
- Phase 1: Field audit & slimming — enforces connection enumerable key limits and documents bytes/connection baseline (~64–69 bytes observed).
- Phase 2: Node pooling — reuse reduces churn and enables deterministic parity when pooling is toggled on/off.
- Phase 3: Slab packing & optional slabs — the primary mechanism Hyper features should rely on to meet bytes/connection targets and to support low-overhead morphogenesis churn.

Implementers should consult `plans/Memory_Optimization.md` for benchmark artifacts, CV targets, and the detailed slab/pooling rules; Hyper PRs that touch growth/prune or caching must reference the relevant Memory phase tests (Field audit, NodePool stats, slab parity & gain omission tests) in their CI checklist.

### Example Genotype Snippets

To help developers and users get started, this section provides canonical genotype examples. These snippets serve as reproducible starting points for experiments and as fixtures for regression tests, ensuring that the developmental process is both deterministic and performant.

Offers canonical genotype archetypes that encode best‑practice starting conditions (minimal symmetric scaffold, hierarchical seed) enabling reproducible baselines for benchmarking. These snippets reduce ramp‑up cost for new users and function as fixtures in regression tests ensuring deterministic development and stable performance signatures across releases.

#### Minimal Seed

Establishes a deterministic seed configuration with just enough structural variability (symmetry + shallow hierarchy) to exercise rule execution pathways while remaining analytically tractable. This baseline underpins performance profiling (isolated from higher‑order morphogenesis) and provides a control for evaluating incremental feature flags.

```ts
const geno: HyperGenotype = {
  seed: 42,
  input: 4,
  output: 2,
  rules: [
    {
      id: 1,
      kind: 'replicate',
      params: { axis: 'x', times: 1 },
      probability: 1,
    },
    { id: 2, kind: 'symmetry', params: { axis: 'y' }, probability: 0.7 },
    { id: 3, kind: 'hierarchy', params: { levels: 2 }, probability: 1 },
  ],
  // TODO: replace with a concrete CPPNGene example; minimal example: seedCPPN({ layers:[{units:8,act:'tanh'},{units:1,act:'identity'}], seed:42 })
  cppns: [
    seedCPPN({
      /* minimal architecture spec: layers, activations, seed */
    }),
  ],
  substrateSpec: {
    dims: 2,
    inputLayout: 'line',
    outputLayout: 'line',
    hiddenInit: 'band',
  },
  version: 1,
};
```

#### Rule Mutation Example

Showcases a parameter mutation that incrementally increases structural capacity along an existing replication axis, illustrating fine‑grained controllability of developmental expansion without introducing novel rule kinds. This emphasizes mutation granularity: small numeric shifts propagate into proportionate phenotype elaboration, aiding smooth fitness landscape traversal.

```ts
// Mutate replicate rule to increase times, enabling deeper band
mutateRule(
  geno,
  (r) => r.kind === 'replicate',
  (r) => (r.params.times = clamp(r.params.times + 1, 1, 5))
);
```

#### Activity-Based Synaptogenesis (Runtime)

Exemplifies a morphogenesis policy that reacts to instantaneous utilization and error signals to add localized capacity, distinct from heritable genome alteration. This separation enables temporally adaptive fine‑tuning (short horizon structural adjustments) while preserving the slower evolutionary channel for consolidating successful motifs into heritable rules.

```ts
for (const region of regionsSortedByFocus(net)) {
  if (region.utilization > 0.8 && region.errorShare > 0.3) {
    attemptLocalGrowth(net, geno, region, { maxNew: 12, sparsityGuard: 0.85 });
  }
}
```

### Connection costs & wiring penalties

A common challenge in generative neural network systems like HyperNEAT is the tendency to produce highly regular but inefficient wiring. To address this, Hyper MorphoNEAT treats wiring cost as a first-class citizen, allowing evolution to favor more compact and modular structures. This section details how wiring penalties are integrated into the system.

Plain HyperNEAT often produces highly regular but non‑modular wiring (many long inter‑module links) because there is no explicit penalty for wiring length or inter‑module connections. To reliably encourage compact, modular networks the plan should treat wiring cost as a first‑class signal across CPPN decisioning, morphogenesis, and telemetry.

- Treat wiring cost (connection count, euclidean length, inter‑module links) as an explicit, configurable signal used across CPPN decisioning, morphogenesis policies, and selection/fitness. In the developmental metaphor this is equivalent to metabolic or material costs that bias growth and pruning.
- Practically: CPPNs can combine a mask output with a cost term to produce an effective score used to realize edges; morph policies prefer local densification and prune long/inter‑module edges first under budget pressure; evolution may expose per‑genotype wiring preference weights (λ_count, λ_len, λ_inter) so wiring economics can be tuned or evolved.
- Making wiring cost explicit enables traceable trade‑offs (task performance vs wiring economy) and supports Pareto or penalized selection strategies described later in this plan.

High level recommendations (what and why):

- Add wiring cost terms to genotype/network bookkeeping: per‑genotype knobs (connectionCountWeight, lengthCostWeight, interModulePenalty) and an optional per‑genotype evolved preference for wiring economy.
- Apply cost awareness at three layers: (A) CPPN adjacency decision (cheap early pruning), (B) fitness/selection (global tradeoff), and (C) morphogenesis/telemetry (local growth/prune policy signals).

Concrete lightweight API & formulas:

- Track per‑network summary values: totalConnections, totalWiringLength (sum of euclidean distances of realized edges), interModuleEdgeCount.
- Single‑objective penalty (simple):

```
penalizedFitness = rawFitness - λ_count * totalConnections
                              - λ_len   * totalWiringLength
                              - λ_inter * interModuleEdgeCount
```

- Multi‑objective alternative: treat (taskFitness, wiringCost) as a Pareto pair and use Pareto selection (e.g., NSGA variants) to explore the trade‑off surface without scalarizing.

CPPN & adjacency recommendations (cheap, local bias):

- Make the CPPN adjacency decision cost‑aware by combining the mask output with a distance / inter‑module cost term before thresholding:

```
effectiveScore = maskOutput - costCoeff * (normalizedDistance + interModuleFlag)
realizeEdge if sigmoid(effectiveScore) > maskThreshold
```

- Provide per‑module mask bonuses so CPPNs can be biased toward intra‑module connectivity (e.g., boost maskOutput if src.module === dst.module).

Morphogenesis & pruning recommendations:

- Under memory pressure or when pruning is considered, prefer removing inter‑module or long‑distance edges first (unless they show high contribution score).
- When growing, prefer local densification (intra‑module) before adding long‑range shortcuts; allow occasional long links but gate by a budget or cooldown.

Telemetry & metrics:

- Expose wiring metrics in `telemetry.ts`: meanEdgeLength, interModuleRatio, modularityQ. Use these to track emergent modularity and guide focusScore adjustments.

Practical experiment suggestions (to tune λ values and policies):

- A/B test: no cost vs scalar penalty vs Pareto selection; measure task performance vs modularity (Q), interModuleRatio, and edge count.
- Compare CPPN cost‑aware thresholding vs cost only in selection to see which induces more modularity with less performance loss.

Risks & mitigations:

- Over‑penalizing wiring reduces achievable task performance — mitigate by annealing λs, evolving λ per genotype, or using Pareto selection.
- Metric compute cost (modularity, length) — compute telemetry incrementally and at low cadence (end of epoch) or sample subnetworks.

Where to add this in the phased plan (quick mapping):

- Phase 1: add config flags and genotype fields for wiring‑cost weights and basic wiring telemetry counters.
- Phase 3 (CPPN): implement cost‑aware adjacency thresholding and per‑module mask bonuses; include adjacency cache keys for cost parameters.
- Phase 4 (Morphogenesis): prefer local growth and prune inter‑module edges first; include wiring metrics in morph decision inputs.
- Phase 6 (Telemetry): export wiring metrics and modularity Q; add automated ablation dashboard entries.
- Phase 7 (Evolution): support penalized fitness option and/or Pareto selection; include wiringCost component in speciation distance if desired.

This section intentionally keeps changes incremental and opt‑in: wiring penalties are gated by config flags and can be tuned or evolved, so users keep the option to explore pure regular HyperNEAT behaviour or wiring‑aware morphogenesis.

### Morphogenesis Event Cycle (Detailed Example)

To illustrate how runtime adaptations occur, this section provides a detailed example of the morphogenesis event cycle. It documents the sequence of actions—pruning, growing, and rebuilding—and the invariants that ensure state consistency and prevent race conditions. This example serves as both a pedagogical tool and a reference for implementation.

Documents the order and conditionality of runtime adaptation actions (prune → grow → rebuild), making explicit the invariants (metrics snapshot immutability during a cycle, deferred rebuild) that guard against race conditions and inconsistent state. This trace form facilitates reproducibility and pedagogical walkthroughs.

1a. Add mutations / crossover support for genotype wiring preference fields; allow `wiringPreferences.evolvePreference` to enable evolution of λ weights.
1b. Add fitness options: scalarized penalized fitness (fitness' = fitness - λ*count * count - λ*len * length - λ_inter \* interModule) and a Pareto multi‑objective mode. Make selection method configurable.
1c. Optionally add wiringCost component into speciation distance to discourage mixing of very different wiring preferences unless desired. 2. Implement `crossoverHyperGenotype(a,b,cfg)` with alignment & blending strategies (rules, CPPNs, substrate modifiers) + validation pass. 3. Add speciation distance components for genotype differences (rule vector hash distance, CPPN topology/weight signature, substrate modifier delta). 4. Integrate reproduction into population loop: select parents, generate offspring genotype(s), regenerate phenotype using cached artifacts. 5. Fitness evaluation optionally rebuilds phenotype each generation if genotype mutated or crossover occurred (cache reuse across offspring clones). 6. Tests:

- Crossover determinism under fixed RNG.
- Speciation separation on crafted genotype pairs.
- Offspring validity (no dangling regions).
- Performance smoke test with mixed mutation + crossover.
  Acceptance: End‑to‑end evolve run with hyper genotype + reproduction finishes within target time budget; offspring functional parity tests pass.

### Phase 8 – Scale & Stress Validation

Subjects the full stack to stress conditions (million‑edge potentials, churn cycles) to confirm memory, time, and determinism invariants hold. Results gate public API stabilization and inform tuning defaults (thresholds, budgets).
Steps:

1. Benchmark script generating ~1M potential connections (sparse) comparing memory vs classic path.
2. Stress test morphogenesis growing then pruning to ensure no memory leaks (pool sizes stable).
3. Document scaling heuristics & recommended flags.
4. CI job thresholding memory & run time.
   Acceptance: Peak memory per active connection meets target (TODO: finalize numeric target after Phase 8 measurement; suggested temporary target: <= 64 bytes/active connection measured on CI harness).

## Phases 0–1: Acceptance, Objectives, Key Tasks, Tests, Risks (polished)

Thesis: Deliver a minimal Hyper MorphoNEAT skeleton that is opt‑in, deterministic, and verifiable; provide an explicit pruning policy for morphogenesis with deterministic validation and measurable acceptance targets.

1. Objectives

1) Provide a guarded hyper feature flag that is off by default.
2) Define a persistent HyperGenotype schema and deterministic build path to a baseline phenotype.
3) Add substrate coordinate assignment and canonical genotype hashing/serialization.
4) Specify a deterministic pruning policy for morphogenesis with budget checks and rollback safety.
5) Surface concrete tests and numeric acceptance criteria for CI gating.

2. Key tasks

1) Feature flag: add config.enableHyper (default false). All hyper imports must be no‑op when false.
2) Genotype factory: implement createInitialGenotype({ input, output, seed }) returning a minimal rule list and substrateSpec.
3) Substrate: deterministic 1D/2D coordinate assignment with canonical ordering.
4) Serialization & hash: implement encodeGenotype/decodeGenotype and hashGenotype(geno) using canonical JSON (sorted keys) and stable numeric formatting.
5) Phenotype builder: baseline materialization producing fully connected input→output network when hyper is enabled.
6) Pruning policy: implement the stepwise pruning algorithm below (deterministic RNG, explicit budgets, dry‑run validation and rollback).
7) Tests & microbench: add deterministic rebuild tests (N=50), serialization round‑trip, and build-time microbenchmark harness.

3. Morphogenesis Event Cycle — Pruning policy (compact algorithm)

- Invariants:

  - genotype + seed -> canonical phenotype ordering and hash.
  - All stochastic choices use a deterministic RNG seeded via a canonical helper `combineSeeds(genoHash, epochCounter)` (e.g., concat+xxhash64 or HMAC with fixed key). TODO: define exact `combineSeeds` implementation and expose helper in `src/hyper/utils.ts` for reproducibility.
  - Budgets: maxNodes, maxEdges, memoryBudget are checked before commit.

- Deterministic pruning algorithm (stepwise):

```typescript
// Pseudocode: prunePolicy.ts
// deterministicSeed := combineSeeds(genoHash, epochCounter)
// NOTE: `combineSeeds` must be a documented, canonical combiner (stable across platforms).

/**
 * pruneModuleEdges(net, moduleId, ctx)
 * - Prefer inter-module and long edges.
 * - Perform dry‑run removals to validate budget/constraints before commit.
 */
function pruneModuleEdges(net, moduleId, ctx) {
  // Step 1: Snapshot metrics (no mutation)
  const edges = net.getEdgesForModule(moduleId); // stable ordering
  // Step 2: Score edges (higher -> better candidate for removal)
  // score = λ_len * normalizedLength + λ_inter * interModuleFlag - contributionScore
  const scored = edges.map((e) => ({
    edge: e,
    score:
      ctx.wiringCost.lengthWeight * normalize(e.length) +
      ctx.wiringCost.interWeight * (e.src.module !== e.dst.module ? 1 : 0) -
      normalizeContribution(e.contribution),
  }));
  // Step 3: Sort descending by score (tie-break deterministic by edge.id)
  scored.sort((a, b) =>
    a.score === b.score ? a.edge.id - b.edge.id : b.score - a.score
  );
  // Step 4: Select batch to remove until targetFraction or budget satisfied
  const toRemove = [];
  let removalCount = 0;
  for (const s of scored) {
    if (removalCount >= ctx.maxRemovalsPerCycle) break;
    if (wouldViolateConnectivity(net, s.edge)) continue; // preserve minimal connectivity heuristics
    toRemove.push(s.edge);
    removalCount++;
  }
  // Step 5: Dry run validation
  const dryNet = net.cloneDry(); // no heavy allocation, returns simulated counts
  dryNet.removeEdges(toRemove);
  if (!dryNet.withinBudgets(ctx.budgets)) {
    return { success: false, reason: 'budget-violation' }; // abort, no mutation
  }
  // Step 6: Commit
  net.removeEdges(toRemove);
  // Step 7: Record trace entry (lazy allocate only when telemetry enabled)
  traceAppend({
    type: 'prune',
    moduleId,
    removed: toRemove.length,
    seed: deterministicSeed,
  });
  return { success: true, removed: toRemove.length };
}
```

4. Tests (single‑expect rule; deterministic seeds; numeric targets)

1) Unit: "prune selects inter-module edges first" — set up a small net with known inter/intra edges; run prunePolicy; expect(topRemovedIsInterModule). (1 expect)
2) Determinism: "build + prune repeatability N=50" — rebuild from same genotype+seed 50 times, run prunePolicy with same epochCounter; expect(allHashesEqual). (1 expect)
3) Safety: "dry‑run rejects budget violation" — construct net where proposed removals would violate minConnectivity; expect(dryRunRejected). (1 expect)

5. Metrics and Acceptance criteria (numeric)

1) Deterministic rebuild reproducibility: 100% identical phenotype ordering and hash across N=50 rebuilds.
2) Pruning policy overhead: disabled path adds <1% runtime; enabled idle adds <5% runtime in microbench (instrumented).
3) Pruned edges preference: top 75% of first batch removals should be inter‑module or top 25% longest edges (measured on synthetic nets).
4) Build time overhead for Phase 1 baseline: ≤1.1× direct instantiation for small nets (<=1k nodes).

6. Risks & mitigations

1) Risk: Non‑determinism from float rounding. Mitigation: canonical numeric formatting and fixed‑precision rounding for hashing and CPPN inputs.
2) Risk: Cascade removal causing disconnected modules. Mitigation: connectivity checks in wouldViolateConnectivity and dry‑run validation.
3) Risk: Memory blowup from adjacency cache. Mitigation: LRU eviction and explicit cache size config.
