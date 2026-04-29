# NEAT Genesis EvoDevo (NGE)

**Status:** [PLANNED]

> The name is a deliberate pun. The initials are **NGE** — a nod to the franchise whose title demonstrated that a compact seed can unfold into something far larger than its origins suggest. Here the seed is DNA; the grown thing is a neural architecture shaped by evolution, development, and lifetime experience.

NGE is an evo-devo extension of NEAT for embodied sensorimotor intelligence, inspired by two biological facts:

1. **DNA is a program, not a blueprint.** A few hundred kilobytes of genetic code suffice to specify how to grow a brain — not by listing every synapse, but by encoding generative rules that unfold deterministically. The final structure vastly exceeds the encoding.

2. **Ant brains are metabolically optimal, modularly specialized intelligence.** A carpenter ant brain of ~250,000 neurons solves navigation, chemical communication, multi-role cooperation, and adaptive foraging — without gradient descent, in milliseconds, under strict metabolic economy. Every module earns its wiring cost.

NGE builds neural architectures the same way:

1. **DNA builds a brain deterministically** (development).
2. **Experience gates where capacity grows** (usage-driven local expansion).
3. **Unused or costly wiring is pruned and compacted** (energy economy).
4. **Once stable, DNA is slowly updated** so future generations start closer to discovered structures (structural assimilation; no weight inheritance).

Unlike language models, **NGE targets embodied sensorimotor intelligence** — agents that act in physical and simulated environments, not agents that generate tokens. The canonical benchmark environments are web canvas simulations: a three-car competitive racing benchmark, an ant-hive ecosystem, and a predator/prey co-evolutionary arena.

This plan is constrained by [plans/Memory_Optimization.md](plans/Memory_Optimization.md). If the two conflict, the memory plan wins.

---

## Scope and Maturity

This is a **concept and architecture plan**, not an implementation-complete spec.

- **In scope:** computation motifs, memory architecture, neuromodulation, reproduction system, collective intelligence framework, lifecycle model, deterministic contracts, DNA composition, budget policy, cache boundaries, and acceptance criteria.
- **Out of scope (for now):** full operator-level API details, final data schemas, and low-level benchmark harness implementation.
- **Follow-on demo plans:** [NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md), [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md), [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md).
- **Authority rule:** if this plan conflicts with `plans/Memory_Optimization.md`, the memory plan remains authoritative.

---

## Execution Alignment (numbered tracks)

This plan executes as **Track 2** in the memory roadmap, sequenced after core implementation phases.

- Track 1 (Memory foundation): phases 0–10 in `plans/Memory_Optimization.md`
- Track 2 (NGE algorithm): phases 11–17 in `plans/Memory_Optimization.md`

Authoritative mapping:

| NGE Phase | Memory Plan Phase      | Focus                                      |
| --------- | ---------------------- | ------------------------------------------ |
| Phase 0   | NGE-local prerequisite | Computation Motifs                         |
| Phase A   | Phase 11               | DNA + deterministic development            |
| Phase B   | Phase 12               | Juvenile focus + local growth/prune        |
| Phase C   | Phase 13               | Adult optimization + equilibrium           |
| Phase D   | Phase 14               | Assimilation                               |
| Phase E   | Phase 15               | Evolution integration + reproduction modes |
| Phase F   | Phase 16               | Scale + stress validation                  |
| Phase G   | Phase 17               | Multi-agent + collective intelligence      |

---

## Explicit Evo-Devo Positioning

- **Evo (evolution):** selection, crossover/mutation, and speciation optimize compact developmental DNA across generations.
- **Devo (development):** each lifetime deterministically constructs a phenotype from DNA rules and programs before experience-driven local edits begin.

---

## Design Pillars

- **Opt-in + isolated:** NGE features live behind flags and must not affect classic NEAT when disabled.
- **Pay-for-use:** no memory/time overhead unless enabled (mirrors the memory plan).
- **Budgeted growth:** every build/morph action obeys explicit caps and must be rollbackable.
- **Deterministic by default:** same DNA + seed + same experience stream ⇒ same result.
- **Computation motifs first:** modules are not generic nodes — each has an explicit `computationType` that determines what it computes at inference time.
- **Embodied intelligence target:** all design decisions bias toward sensorimotor agents in physical environments, not token-prediction tasks.
- **Epigenetic guidance is weak + optional:** parent references can nudge search early; they cannot force convergence or replace exploration.
- **Biology-inspired realism (optional):** at extreme scales, DNA may use lossy compression (opt-in, explicitly documented).

---

## Analogy to Biological Development

This table is shared vocabulary, not an implementation spec.

| Stage                       | Biological inspiration                        | NGE engineering analog                                                                                           |
| --------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Embryonic seed              | Few stem cells                                | Minimal DNA scaffold + I/O anchors + initial module archetypes                                                   |
| Patterning gradients        | Morphogens / HOX genes                        | Substrate coordinates + CPPN fields + deterministic tagging (module/zone ids)                                    |
| Proliferation               | Cell division                                 | DNA-driven replicate/hierarchy rules under explicit growth budgets                                               |
| Differentiation             | Neurons specialize                            | Per-module `computationType` + archetype params (activation family, plasticity policy, wiring-cost zone weights) |
| Axon guidance               | Growth cones follow gradients                 | Cost-aware CPPN adjacency realization + locality bias + sparsity-first thresholds                                |
| Pruning & refinement        | Synaptic pruning                              | Experience-gated prune/compact (prefer long/inter-module edges; protect reward-critical wiring)                  |
| Lifelong plasticity         | Hebbian remodeling                            | Optional plasticity; structural edits are slower, local, cooldown-limited                                        |
| **Mushroom body**           | **Associative memory / sparse encoding**      | **`EpisodicSlot` archetype — content-addressable medium-term memory retrieval**                                  |
| **Central complex**         | **Ring attractor / navigation sequencing**    | **`GatedRecurrentCell` archetype — short-term state with spatial dynamics**                                      |
| **Neuromodulation**         | **Octopamine / serotonin gain modulation**    | **`ModulatorBroadcaster` — fast behavioral mode switch without structural change**                               |
| **Polyandric reproduction** | **Queen × multiple drones → diverse workers** | **Polyandric reproduction mode — diverse offspring from a stable queen genome**                                  |
| Epigenetic assimilation     | Stabilized development biases next generation | Slow per-module DNA updates after equilibrium (no weight inheritance)                                            |

---

## What NGE Is (and Is Not)

**Is:**

- An evo-devo encoding (rules + CPPNs + substrate modifiers) that scales.
- A staged lifecycle controller that shifts from "grow fast" → "optimize/compact."
- A per-module focus system that decides where to spend structural budget.
- A **runtime attention mechanism**: dynamic query-key-value routing at inference time, not only at growth time.
- A **three-tier memory architecture**: short-term recurrent (fast decay), medium-term episodic (content-addressable), long-term structural (weights + DNA).
- A **neuromodulation layer**: fast behavioral mode switching via broadcast gain signals, without structural changes.
- A **multi-agent collective intelligence framework**: stigmergy through shared environment fields, role differentiation from identical DNA, and co-evolutionary dynamics.

**Is not:**

- Weight inheritance. Weights remain lifetime state; DNA encodes structure and policies.
- A mandatory gradient/backprop system. Credit assignment is via black-box probes and reward deltas.
- A language model or sequence-prediction system. NGE targets embodied sensorimotor control.
- A perceptron. Computation motifs include attention heads, recurrent cells, episodic memory slots, gating routers, and neuromodulators — not only dense feedforward layers.

---

## Computation Motifs — Runtime Architecture

The evo-devo tradition focuses on _how topology grows and evolves_, leaving the computation model implicit and defaulting to a weighted directed graph (a sparse MLP). NGE makes computation motifs explicit: every module archetype declares what it computes at forward-pass time.

### Module Archetype computationType Catalogue

`computationType` is a **mandatory field** on every module archetype definition in the DNA schema.

| computationType        | Biological analog                     | Role in architecture                                                                 |
| ---------------------- | ------------------------------------- | ------------------------------------------------------------------------------------ |
| `DenseFeedForward`     | Generic cortical layer                | Default integration layer                                                            |
| `AttentionHead`        | Mushroom body calyx                   | Dynamic query-key-value routing over a candidate zone                                |
| `GatedRecurrentCell`   | Central complex ring attractor        | Short-term state, fast dynamics, per-module hidden state                             |
| `EpisodicSlot`         | Mushroom body output lobe             | Content-addressable medium-term retrieval (write on novelty; retrieve by similarity) |
| `ModulatorBroadcaster` | Octopaminergic / serotonergic neurons | Global context signal; broadcasts gain/bias to receiving zone                        |
| `GatingRouter`         | Basal ganglia selection circuit       | Sparse MoE-style gating; activates top-k downstream modules per forward pass         |
| `ResidualTap`          | Thalamic relay                        | Read/write on the ResidualStream highway                                             |
| `NormalizationLayer`   | Homeostatic regulation                | LayerNorm / RMSNorm analog; stabilizes module input distributions                    |

### ResidualStream Primitive

A designated information highway along the substrate's primary axis. Key properties:

- **Zero wiring cost** by policy — the pruner never penalizes `ResidualTap` connections.
- Modules with `ResidualTap` can read from or contribute to the stream at every forward pass.
- DNA encodes the stream's dimensionality and which zones have tap access.
- Enables deep module stacking without vanishing-path degradation: any module can read the full accumulated context and add only its own contribution.

### WeightSharedCohort

A DNA governance knob that marks a set of modules as sharing a single runtime weight tensor. Each cohort member is differentiated only by its substrate coordinate, which is injected into the module's input at every forward pass.

This is the HOX gene analog: one developmental program applied at multiple substrate positions. Benefits:

- Reduces DNA size proportional to cohort size.
- Reduces runtime parameter count.
- Training one cohort member effectively trains all members.
- Evolution can independently decide whether a cohort remains shared or specializes — this is an evolvable trait in DNA.

### Runtime Substrate Coordinate Injection

Module archetypes can declare `receivesCoordinates: true`. When set, the module's input tensor is concatenated with its `(x, y, z)` substrate position at every forward pass. This gives modules spatial self-awareness at inference time, not only during development — enabling position-dependent computation from shared weights (similar to positional encoding in Transformers, but grounded in physical substrate geometry).

---

## Memory Architecture

Three tiers, each represented as first-class module archetypes:

| Tier        | Archetype                | Time constant     | Capacity         | Governed by                        |
| ----------- | ------------------------ | ----------------- | ---------------- | ---------------------------------- |
| Short-term  | `GatedRecurrentCell`     | Per-epoch decay   | Hidden dimension | DNA: `hiddenDim`, `decayRate`      |
| Medium-term | `EpisodicSlot`           | Per-episode write | Slot count       | DNA: `slotCount`, `evictionPolicy` |
| Long-term   | Structural weights + DNA | Generational      | Network size     | Lifecycle + assimilation           |

**Short-term:** recurrent hidden state with learnable decay rate. Fast dynamics, per-module. Active context that evaporates without refreshing — working memory analog.

**Medium-term:** write on novelty or surprise triggers; retrieve by dot-product similarity against stored slot activations. Minimal NTM/MANN-lite design — a small fixed-size slot array per module, no external memory controller. Analogous to the mushroom body output lobe: patterns that were recently surprising and reward-relevant get stored; retrieval is by similarity, not address.

**Long-term:** unchanged from classic evo-devo. Structural weights and DNA assimilation are the long-term store.

The lifecycle grow/prune policy applies to memory modules exactly as to computation modules: heavily used episodic slots grow in capacity (more slots); unused slots are pruned first under wiring-economy pressure.

---

## Neuromodulation

Neuromodulator zones are substrate zones that broadcast a low-dimensional modulation vector to all modules within their broadcast radius.

**Properties:**

- Placed by developmental rules (DNA: position, broadcast radius, output dimensionality).
- Receive input: reward delta summary, task-context signal, internal state aggregation.
- Output: multiplicative gain or additive bias applied to every receiving module.
- **Zero structural change required** — one forward pass through the modulator head shifts the gain of an entire zone.
- Wiring cost: `ModulatorBroadcaster` edges within their declared broadcast radius are cost-exempt.

This separates two distinct processes the lifecycle would otherwise conflate:

- **Slow structural adaptation** (morphogenesis, lifecycle stages) — takes many timesteps.
- **Fast behavioral mode switching** (neuromodulation) — one forward pass.

An agent can shift from foraging mode to defense mode in a single timestep via modulator gain signals, without waiting for the morphogenesis cycle.

**Relationship to behavioral drives:** in the racing and ant-hive benchmarks, explicit behavioral drives (pace, safety-margin, collision-avoidance, recovery, alarm) correspond directly to `ModulatorBroadcaster` outputs. The drive vocabulary maps onto distinct modulator heads that bias the relevant zone's computation without requiring structural growth.

---

## Reproduction System

DNA encodes a `reproductionPolicy` governing how offspring are created. The mode is itself optionally evolvable — evolution can discover which reproductive strategy best suits the current task phase and environment.

### Mode 1: Parthenogenesis (asexual / clonal)

The designated primary parent produces offspring from its own DNA alone. No crossover. Variation comes exclusively from mutation.

```
offspring.DNA = mutate(parent.DNA, rate: policy.parthenogenesisMutationRate)
```

Configurable mutation rate:

- `0.0` → true clone (identical DNA, different seed → sibling by default)
- low (0.001–0.01) → bud variation (small structural drift, no crossover)
- standard → exploratory asexual (more variation, still no crossover)

**Biological analog:** thelytokous parthenogenesis in desert ant queens producing diploid workers asexually.

**Use case:** environment is stable; current DNA is high-fitness; exploration should be narrow.

### Mode 2: Polyandric (queen × many drones)

One designated "queen" parent provides the primary DNA template. Multiple secondary "drone" parents each contribute patches to distinct, non-overlapping DNA regions. Queen contributions win all conflicts.

```
offspring.DNA = deepCopy(queen.DNA)
for each drone Bᵢ in [B₁, B₂, ..., Bₙ]:
    region = assignedRegion(i, policy.assignedRegionStrategy)
    patch  = extractDNAPatch(Bᵢ, region)
    offspring.DNA = applyPatch(offspring.DNA, patch, priority: secondary)
offspring.DNA = mutate(offspring.DNA, rate: policy.standardMutationRate)
```

Assignable regions per drone (non-overlapping):

- CPPN parameter blocks
- Rule pass priorities and distributions
- Module archetype deltas
- Budget and wiring-cost knobs
- Schedule thresholds

`polyandricDroneContributionFraction`: proportion of the queen's DNA regions patchable by drones total. Low (~0.1) → mostly queen; high (~0.5) → near-equal patchwork.

**Biological analog:** honeybee and leafcutter ant queens mating with 10–20+ drones, storing diverse sperm, producing worker cohorts with different functional specializations from a stable queen genome.

**Use case:** one lineage has emerged as a strong core genome; diversity is needed for role specialization or environmental variance. Produces diverse offspring while preserving the proven scaffold.

### Mode 3: Standard Sexual (A × B)

Classic two-parent crossover aligned to NEAT innovation markers. Fitter parent preferentially contributes disjoint/excess genes. Both parents contribute equally to matching regions (uniform crossover or arithmetic blend per DNA part type — see DNA recombination table).

**Use case:** early exploration; two distinct lineages need to be merged; environment has changed significantly.

### reproductionPolicy in DNA

```
reproductionPolicy:
  mode: 'parthenogenesis' | 'polyandric' | 'sexual'
  parthenogenesisMutationRate: float           # 0.0 = true clone
  polyandricDroneCount: int                    # secondary donor count
  polyandricDroneContributionFraction: float   # fraction of DNA regions patchable by drones
  queenBias: float                             # 0.5 = equal weight; 1.0 = queen wins all
  assignedRegionStrategy:
    'roundRobin' | 'byFitness' | 'bySpecialization'
  modeIsEvolvable: boolean                     # can evolution mutate this field itself
  seedPolicy:
    siblingsDifferBySeed: boolean              # default: true
    twinsAllowed: boolean                      # identical DNA + identical seed
```

When `modeIsEvolvable: true`, the reproduction mode is subject to selection pressure. Lineages that discover parthenogenesis in stable phases and polyandry in diverse phases gain a fitness edge through reproductive efficiency — directly mirroring how ant colonies shift reproductive strategies across colony maturity stages.

### Epigenetic Reference Priors

Opt-in mechanism available across all reproduction modes. Offspring may receive weak two-parent reference anchors for parameter initialization or mutation-time biasing. References are not persistent inherited weights — they are birth-time nudges that decay over early lifecycle stages unless still beneficial.

$$\theta \leftarrow \theta + \Delta\theta_{\text{mutation}} + \lambda\,(\theta^* - \theta)$$

Where $\theta^*$ is a two-parent reference and $\lambda$ is intentionally small.

---

## Collective Intelligence

### Stigmergy

Agents coordinate indirectly by modifying a shared environment field rather than communicating directly. Other agents sense the field and respond. No direct messaging required.

**Implementation:** the shared field is a 2D typed-array grid (aligns with memory plan slab infrastructure). Diffusion and decay are simple per-tick array operations. Agents read field values as sensory inputs; they write to the field as action outputs. Total field memory cost = `gridWidth × gridHeight × channelCount × 4 bytes (float32)` — negligible at canvas simulation scales.

Field types (pheromone family example):

- Food trail (guide foragers toward food sources)
- Alarm (signal threat; trigger defense mode via neuromodulation)
- Recruitment (summon allies to a location)
- Nest-scent (maintain colony territory orientation)

### Role Differentiation from Identical DNA

Agents with identical DNA but different early experience streams develop structurally different modules via experience-gated plasticity. A forager that only processes food-trail signals grows a denser chemosensory module and prunes unused defense circuitry. A soldier does the opposite.

**Key property:** caste is not predetermined by DNA — it emerges from the "Experience gates where capacity grows" principle applied at the developmental level. This makes development reactive to environmental context, not only DNA-predetermined.

**Observable:** two agents from identical DNA, raised in different environment sectors, should produce measurably different module size distributions by adulthood.

### Co-evolutionary Dynamics

Two populations evaluate fitness against each other. Each maintains its own NEAT speciation and assimilation cycle. Fitness is computed against a rolling opponent snapshot: hall-of-fame representatives, recent-population sample, or blended.

Neither population has a fixed fitness target — both co-adapt, driving a genuine arms race in structure and behavior. This is the most direct stress test of whether NGE can track a non-stationary fitness landscape without catastrophic forgetting of previously useful structures.

---

## Core Concept (Engineering Mapping)

### DNA → Development → Experience → Assimilation

**DNA (genotype) is not a graph.** It is a compact program + governance knobs that deterministically generate a phenotype and then regulate how the phenotype is allowed to change during life.

**Development (DNA-only):** execute the developmental program to build an initial scaffold.

**Experience (lifetime):** measure usage + reward sensitivity and apply slow, local structural edits:

- active/valuable parts tend to **grow** (densify edges; sometimes add nodes)
- inactive/costly parts tend to **prune/compact** (remove long/inter-module edges first)

**Assimilation (between generations):** once equilibrium is reached, encode discovered structural priors back into DNA so offspring start closer to the "good scaffold." Baldwin effect by default; optional assisted biasing.

### Deterministic Build Pipeline

1. **Substrate:** deterministically assign coordinates, zone/module tags, and `computationType` per archetype placement.
2. **Rule passes (development):** apply prioritized, deterministic rules to expand a virtual node/module plan.
3. **Indirect connectivity:** evaluate CPPN programs selectively to realize sparse edges with wiring-cost awareness; realize ResidualStream and WeightSharedCohort assignments.
4. **Materialize:** instantiate slab-backed runtime structures and attach computation motif implementations and optional policy hooks.

Cache boundaries are explicit (memory plan L7): adjacency cache and phenotype slab reuse are the primary wins.

---

## Determinism, Variability, and Reproducibility

### Determinism Contract (default)

Same DNA + seed + same experience stream ⇒ identical phenotype (ordering + hash) at each lifecycle checkpoint. Stable ordering must be defined for: nodes, edges, module IDs, rule ordering, probe sampling, and memory slot indices.

### Safety Invariants

- **Reproducibility:** DNA + seed + experience stream ⇒ canonical phenotype at lifecycle checkpoints.
- **Idempotence:** repeated builds with identical inputs produce identical ordering and hashes.
- **Budget enforcement:** all build/morph steps respect caps (nodes/edges/bytes/time) and are rollbackable.
- **No global side effects:** importing NGE modules when disabled must not mutate global state.
- **Lazy diagnostics:** traces/telemetry allocate only when enabled.

### Optional Biological Variability (at scale)

When `encodingMode: 'lossy'` is enabled, deterministic reconstruction is best-effort but may vary slightly due to quantization/rounding details. Must be surfaced via `dna.encodingMode`, `dna.compatibilityVersion`, and a reproducibility warning in telemetry.

---

## NGE_DNA: Compact, Evolvable Instruction Set

DNA is a dedicated class with explicit schema versioning, canonical lossless encoding, and optional compressed fragments for large-scale topology/program data.

### Responsibilities

- Provide deterministic development inputs (rules/CPPNs/substrate).
- Provide lifecycle schedule + stage transition goals.
- Provide budgets + wiring-cost preferences.
- Define experience probes and their cadence.
- Support per-module assimilation and compressed per-module directives.
- Define computation motif placement (`computationType` per archetype).
- Define memory tier parameters (`hiddenDim`, `slotCount`, `decayRate`).
- Define neuromodulator zones (position, broadcast radius, input/output dimensionality).
- Define reproduction policy (mode, parameters, evolvability flag).

### High-level Schema (conceptual)

- **Identity & compatibility**
  - `schemaVersion`, `compatibilityVersion`, `encodingMode`, `fingerprint`
- **Reproduction policy**
  - `reproductionPolicy` (see Reproduction System section)
- **Substrate & coordinate system**
  - dimensions, normalization, zone partitioning strategy
- **Developmental program**
  - rule passes (replicate/symmetry/hierarchy/differentiate)
  - one or more CPPN programs (optionally per module archetype)
  - substrate modifiers (scale/rotation/etc.)
- **Module system**
  - module archetypes with mandatory `computationType`
  - `WeightSharedCohort` assignments
  - per-module `receivesCoordinates` flag
  - module addressing scheme (stable IDs and ordering)
  - per-module directives (optional, may be compressed)
- **Memory tiers**
  - short-term parameters per zone (`GatedRecurrentCell`: `hiddenDim`, `decayRate`)
  - medium-term parameters per zone (`EpisodicSlot`: `slotCount`, `evictionPolicy`)
- **Neuromodulator zones**
  - position, broadcast radius, input source spec, output dimensionality per zone
- **Governance**
  - stage/substage schedule and goal checks
  - budgets (nodes/edges/bytes/time)
  - wiring-cost preferences (soft pressure, evolvable)
  - probe schedule (cheap every epoch; expensive on life events)
  - morph policy knobs (growth/prune cooldowns, hysteresis)
- **Payloads (optional)**
  - compressed matrices/tables and quantized parameter blocks
  - alternative encodings ("OR parts") for viability-first decode

### Module Addressing and Zones

- Module IDs are stable within a DNA instance and reproducible across builds.
- Zones are coarse substrate partitions for wiring-cost defaults and budgets.
- Archetypes define repeatable `computationType` + parameter sets; modules reference archetypes + small parameter deltas.
- Module ordering is canonical: sort by `(zoneId, archetypeId, moduleOrdinal)`.
- Any rule pass that creates modules must emit deterministic module IDs.
- Wiring-cost defaults are zone-based; modules without explicit cost weights inherit from their zone.

### Compressed Payloads

When DNA must encode more detail without exploding in size, prefer encodings that compress patterns:

1. **Dictionary-coded archetypes** — small dictionary of archetype definitions; modules store indices + small deltas.
2. **Quantized parameter blocks (lossy optional)** — quantize floats into `int8/int16` blocks with scale/offset header per block.
3. **Sparse SoA hints (lossless)** — compact local lists of (coordinate, weight seed/scale, mask bias).
4. **Run-length / delta coding** — RLE of repeated values; delta coding of monotone sequences.
5. **String-level encodings** — hex for small payloads (debuggable); base64 for larger byte blocks.

### "OR Parts" Decode (viability-first)

DNA may carry multiple alternative representations for a part. Decoding chooses the first option that is: (1) compatible with `compatibilityVersion`, (2) within current budgets, and (3) consistent with `encodingMode`.

### DNA Recombination (high level)

| DNA part            | Alignment key                       | Match rule                    | Crossover outcome                                |
| ------------------- | ----------------------------------- | ----------------------------- | ------------------------------------------------ |
| Rules               | (kind + canonical param signature)  | same kind + normalized params | uniform pick or parameter-wise blend             |
| CPPN program        | topology hash + activation sequence | identical topology            | weight-block crossover or select fitter          |
| Substrate modifier  | (type + axis)                       | same type/axis                | numeric blend + deterministic tie-break          |
| Module archetype    | archetype id + `computationType`    | id equality                   | inherit or blend parameters                      |
| Memory tier params  | field key                           | same field                    | blend within safe bounds, clamp to global caps   |
| Neuromodulator zone | zone id                             | id equality                   | blend position/radius; inherit input/output spec |
| Reproduction policy | field key                           | same field                    | blend within safe bounds                         |
| Budgets / schedule  | field key                           | same field                    | blend within safe bounds, clamp to global caps   |

Excess/disjoint parts follow fitter-biased rules plus a budget/viability normalization step.

### Assimilation Write-back (per-module, compact)

Assimilation must not bloat DNA into a saved phenotype. Prefer updating generators and knobs:

- Archetype parameters and `computationType`-specific config
- Rule distributions (priorities/probabilities)
- CPPN program topology and quantized parameter blocks
- Budgets and wiring-cost weights per zone/module
- Schedule thresholds (plateau detection, marginal return thresholds)
- Memory tier parameters that proved effective (`hiddenDim`, `slotCount`, `decayRate`)
- Neuromodulator zone params that stabilized (gain range, broadcast radius)

---

## Lifecycle: 4 Stages × 4 Substages (goal-gated)

The lifecycle is a **state machine**. Each stage has 4 substages; each substage has a goal. If the goal is not met, repeat another cycle.

### Stage 1: Embryo (DNA-only development)

1. **Anchor:** build minimal scaffold (inputs/outputs + base modules with `computationType` assignments).
2. **Pattern:** apply deterministic rule passes (replicate/symmetry/hierarchy/differentiate).
3. **Wire:** generate sparse connectivity (CPPN/cost-aware thresholds); realize ResidualStream and WeightSharedCohort assignments.
4. **Stabilize:** verify budgets + hashes + baseline forward correctness across all `computationType` implementations.

### Stage 2: Juvenile (grow fast where useful)

1. **Observe:** collect cheap usage signals (utilization/activity, wiring metrics, memory tier access rates, neuromodulator gain variance).
2. **Probe:** sample reward sensitivity via perturbation probes (scheduled; can be expensive).
3. **Expand:** local growth (edge densification first; node add allowed, but slower; `EpisodicSlot` capacity growth when hit-rate is high).
4. **Consolidate:** cool down edited modules; rebuild caches; enforce budgets.

### Stage 3: Adult (optimize/compact more than grow)

1. **Observe:** monitor for stagnation and marginal returns; track neuromodulator gain stability.
2. **Prune:** remove long/inter-module/costly edges first unless protected by contribution; prune unused memory slots first.
3. **Compact:** reduce structure where performance holds; seek smaller equivalent.
4. **Maintain:** allow limited growth only when strong evidence exists.

### Stage 4: Equilibrium (assimilate slowly)

1. **Detect:** plateau + low marginal returns triggers equilibrium candidate.
2. **Validate:** re-check stability across multiple cycles/rollouts.
3. **Assimilate:** update DNA per-module (rules/CPPN/topology templates/budgets/schedules/memory tier params/neuromodulator zone params).
4. **Reset:** start next generation from updated DNA with new child seed via configured reproduction policy.

### Lifecycle Timeline (artifacts & rollback boundaries)

| Step                               | Mutates              | Produces                                               | Rollback boundary                                              |
| ---------------------------------- | -------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| Development (Embryo)               | Phenotype only       | virtual plan + realized adjacency + materialized slabs | abort on budget/invalid refs (no commit)                       |
| Observe/Probe (Juvenile/Adult)     | metrics buffers only | focus scores + probe results                           | drop metrics on failure; no structural commit                  |
| Morph cycle (Grow/Prune/Compact)   | Phenotype only       | delta edits + trace entries                            | dry-run validate then commit; rollback on constraint violation |
| Equilibrium validation             | none (decision step) | "stable" decision + candidate assimilation set         | if unstable, return to Adult cycles                            |
| Assimilation (between generations) | DNA only (slow)      | updated per-module DNA parts                           | if invalid/off-budget, keep prior DNA                          |

---

## Experience Signals and Probes

Default signals (cheap, per epoch):

- utilization/activity statistics
- wiring metrics (edge count, mean length, inter-module ratio)
- memory tier access rates (episodic hit rate, recurrent state refresh rate)
- neuromodulator gain stability
- novelty (structural change rate, focus distribution entropy)

Value signal (more expensive, scheduled):

- reward delta under perturbation: lesion/ablation (temporarily disable module/edges), noise injection (weights/activations), gating test (disable long or inter-module edges)

### Focus Scoring (conceptual)

$$focus(m) = w_u\,\widehat{util}(m) + w_r\,\widehat{rewardDelta}(m) + w_n\,\widehat{novelty}(m) + w_s\,\widehat{stabilityAge}(m) - w_c\,\widehat{wiringCost}(m)$$

Compute targets via softmax/top-k, apply local edits under budgets + cooldown, log before/after deltas for traceability.

---

## Morphogenesis Policies (slow, local, hysteresis)

Structural edits must be:

- local (module-scoped)
- slow (cooldowns)
- hysteretic (growth needs strong evidence; prune needs sustained underuse/cost)
- budgeted + rollbackable

Preferred edit order under stagnation:

1. Edge densification in high-focus modules
2. `EpisodicSlot` capacity expansion (slot count growth when hit-rate is high)
3. Node additions (rare; only when marginal returns justify)
4. Larger structural templates via DNA assimilation rather than rapid lifetime edits

---

## Wiring Cost (soft pressure; evolvable; ant-like)

Wiring economy is a first-class soft pressure across development, morphogenesis, and evolution.

**Standard cost model:** long edges and inter-module edges are penalized first under prune pressure.

**Exemptions:**

- `ResidualTap` connections are exempt — the pruner never penalizes them.
- `ModulatorBroadcaster` edges within their declared broadcast radius are exempt.

This creates a cost landscape that naturally preserves the information highway (ResidualStream) and fast behavioral switching (neuromodulation) while pruning expensive structural connections that don't earn their cost.

---

## Caching (must follow the memory plan)

1. **Adjacency cache (L7):** keyed by genotype/substrate/program signatures; stores SoA edge lists.
2. **Phenotype slab reuse (L7):** reuse typed array slabs and avoid rebuild churn.

Optional (strongly gated): module-level I/O caching only if inputs are quantized/discrete and the module is frozen/state-free.

All memory flags, constants, and accounting must be sourced from the Centralized Memory Manager described in [plans/Memory_Optimization.md](plans/Memory_Optimization.md).

---

## Alignment with the Memory Optimization Plan

NGE depends on completed memory foundations:

- L2 pooling + L3 slabs for churn
- L4 sparsity + budgets for the primary bytes/connection improvements
- L7 caching for adjacency/phenotype reuse

**Gate to Phase 7 (NGE):** Track 1 gates in `Memory_Optimization.md` are met (especially phases 4–7 stability + variance/hardening).

---

## Agreed Defaults

- **No weight inheritance:** weights are lifetime state; DNA encodes structure and policies.
- **`computationType` is mandatory:** every module archetype must declare what it computes.
- **Memory tiers are opt-in per zone:** zones without memory directives default to `DenseFeedForward` only.
- **Neuromodulation is opt-in:** zones without neuromodulator directives have no modulator signal.
- **Reproduction defaults to standard sexual (A × B)** unless DNA specifies otherwise.
- **Parthenogenesis mutation rate defaults to low (not zero):** true clones require explicit `parthenogenesisMutationRate: 0.0`.
- **Epigenetic priors are allowed but weak:** two-parent references may guide initialization/mutation; they are not persistent inherited weights.
- **Per-module assimilation:** equilibrium triggers per-module DNA updates.
- **Siblings differ by seed (default):** reproduction yields same DNA with different child seeds.
- **Twins are allowed:** identical DNA + identical seed.
- **Wiring economy is soft pressure, evolvable:** wiring-cost weights live in DNA and evolve.
- **Probe scheduling lives in DNA:** cheap metrics every epoch; expensive perturbation probes on cadence and life events.
- **ResidualStream taps are wiring-cost-exempt:** the pruner never penalizes them.
- **ModulatorBroadcaster edges within declared radius are cost-exempt.**

---

## Recurrence Policy

Recurrence is no longer a single opt-in flag — it is expressed via `GatedRecurrentCell` module archetypes placed by developmental rules.

- **Default:** no `GatedRecurrentCell` archetypes in the initial DNA scaffold.
- **Enabled by:** adding `GatedRecurrentCell` archetype placements to the DNA module system.
- Formally time-stepped execution (aligned with Phase 9 concepts in the memory plan) is required for correct recurrent semantics.

---

## Assimilation (what changes in DNA)

Assimilation is a slow write-back of **structural priors**, not weights. Per-module assimilation should prefer updating **generators** over storing explicit adjacency:

- Rule params/priorities (replication depth, hierarchy levels, symmetry usage).
- CPPN programs (topology + optionally quantized parameter blocks).
- Wiring-cost weights and budgets per module/zone.
- Lifecycle schedule knobs.
- Memory tier parameters that proved effective (`hiddenDim`, `slotCount`, `decayRate`).
- Neuromodulator zone params that stabilized (gain range, broadcast radius).

---

## Budget Defaults (when DNA omits them)

Resolve via a **deterministic seeded policy:**

1. Use the child seed and `compatibilityVersion` as deterministic inputs.
2. Derive conservative defaults around a baseline (near `0.5` of configured maxima).
3. Persist resolved values into telemetry/checkpoints for auditability.

---

## Phased Roadmap

### Phase 0 — Computation Motifs (NGE-local prerequisite)

- Define the `computationType` catalogue and module archetype schema extension.
- Implement `AttentionHead` and `GatedRecurrentCell` as opt-in primitive module types.
- Implement `EpisodicSlot` as a typed-array-backed content-addressable module.
- Implement `ModulatorBroadcaster` with broadcast-radius governance.
- Implement `GatingRouter` (sparse top-k downstream module selection).
- Implement `ResidualStream` infrastructure and `ResidualTap` archetype.
- Implement `WeightSharedCohort` governance and substrate coordinate injection.
- Determinism test: same archetype + seed → same initialization across all `computationType` implementations.
- Opt-in verification: classic NEAT behavior unchanged when NGE features are disabled.

### Phase A — DNA + Deterministic Development (→ Phase 11)

- Define `NGE_DNA` schema + versioning + canonical encoding (extended from NEAT baseline with `computationType`, memory tiers, neuromodulator zones, `reproductionPolicy`).
- Deterministic development pipeline: substrate → rules → indirect wiring → materialize (with `computationType` dispatch).
- Determinism tests: stable build hash under repeated rebuilds.

### Phase B — Juvenile Focus + Local Growth/Prune (→ Phase 12)

- Focus metrics (cheap) + scheduled perturbation probes (expensive).
- Local growth first, then prune/compact with hysteresis.
- `EpisodicSlot` capacity growth: episodic hit-rate-gated slot count expansion.
- Churn tests aligned to the memory plan (pool high-water mark slope ~0).

### Phase C — Adult Optimization + Equilibrium Detection (→ Phase 13)

- Plateau detection + marginal returns tracking.
- Growth cooling + prune/compact dominance.
- Neuromodulator gain stabilization detection.

### Phase D — Assimilation (→ Phase 14)

- Encode stable structural priors back into DNA (extended to include memory tier params and neuromodulator zone params).
- Optional lossy compression mode for very large structures (explicitly documented).

### Phase E — Evolution Integration + Reproduction Modes (→ Phase 15)

- Multi-family crossover and mutation for all DNA parts.
- Speciation distance extends to `computationType` composition, memory tier depth, and wiring-cost preferences.
- Reproduction mode implementation: parthenogenesis, polyandric, standard sexual.
- Optional epigenetic prior operator: two-parent weak anchors for initialization/mutation-time biasing with deterministic decay.
- Polyandric region assignment strategies (`roundRobin`, `byFitness`, `bySpecialization`).

### Phase F — Scale + Stress Validation (→ Phase 16)

- Stress at high edge counts with the memory plan benchmark harness.
- Validate: cache hit ratio, bytes/connection, rebuild variance, churn leak slope.
- Validate `computationType` dispatch overhead at scale.
- Validate memory tier growth/prune stability.

### Phase G — Multi-Agent + Collective Intelligence (→ Phase 17)

- Stigmergy field infrastructure (typed-array pheromone/chemical grid; diffusion + decay ops).
- Multi-agent evaluation harness: N agents per generation, shared field state.
- Role differentiation validation: identical DNA → divergent module size distributions via experience.
- Co-evolutionary dynamics: two-population evaluation with rolling opponent snapshot.
- Canvas demo integration: ant-hive and predator/prey benchmarks running in browser.

---

## Primary Demos

Three canvas-runnable benchmarks, each targeting a distinct NGE capability cluster:

| Priority     | Demo                       | NGE capabilities exercised                                                                                                                                                          | Plan                                                                                   |
| ------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Flagship** | **Ant Hive Ecosystem**     | **Stigmergy, role differentiation from identical DNA, all three memory tiers, neuromodulation (pheromone-triggered mode switch), polyandric reproduction, collective intelligence** | [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md)           |
| Second       | Predator/Prey Co-evolution | Co-evolutionary dynamics, sensory arms race, structural divergence under selection, reproduction mode evolution                                                                     | [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) |
| Third        | Team Racing                | Adversarial cooperation, team-radio stigmergy, role specialization, co-evolution between teams, episodic rival memory                                                               | [NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md) |

The ant hive is the flagship because it exercises the broadest set of NGE's most distinctive capabilities in a single environment — including the three features that distinguish NGE most from standard NEAT: role differentiation from identical DNA, polyandric reproduction, and stigmergic collective intelligence.

---

## Acceptance Criteria

- **Determinism:** same DNA + seed + experience stream yields identical hashes at lifecycle checkpoints across all `computationType` implementations.
- **Memory:** must not regress memory plan targets; sparsity and budgets are the primary lever.
- **Caching:** adjacency cache hit ratio > 70% in repeated-evaluation scenarios.
- **Churn safety:** pool high-water mark stabilizes under repeated grow/prune cycles.
- **Wiring economy:** long/inter-module edges are preferentially pruned under cost pressure; `ResidualTap` and `ModulatorBroadcaster` edges survive.
- **Attention correctness:** `AttentionHead` modules produce different routing patterns for different inputs (not collapsing to uniform attention).
- **Memory tier correctness:** `EpisodicSlot` retrieval accuracy improves across episode exposure; `GatedRecurrentCell` state persists within an episode and resets at episode boundaries.
- **Reproduction correctness:** parthenogenesis produces mutation-only variation; polyandric produces queen-template + drone-patch composition with correct region assignment; standard sexual produces NEAT-aligned crossover.
- **Role differentiation:** two agents from identical DNA in different experience streams produce measurably different module size distributions by adult stage.
- **Classic NEAT unaffected:** all NGE features disabled → behavior identical to pre-NGE baseline.

---

## Readiness Checklist (for implementation start)

- [ ] Canonical `NGE_DNA` schema draft with explicit versioning fields, `computationType`, memory tier params, neuromodulator zones, and `reproductionPolicy`.
- [ ] Computation motif implementations (Phase 0) complete and opt-in verified.
- [ ] Deterministic ordering contract for module IDs, rule order, edge realization, and `computationType` dispatch.
- [ ] Budget resolution algorithm defined as deterministic (seeded) and testable.
- [ ] Lifecycle transition guards defined with measurable thresholds.
- [ ] Reproduction mode unit tests: parthenogenesis mutation-only, polyandric patch composition, sexual NEAT alignment.
- [ ] Stigmergy field infrastructure spec (typed-array layout, diffusion/decay ops, agent read/write interface).
- [ ] Telemetry contract finalized for reproducibility warnings (`encodingMode`, compatibility, lossy flags).
- [ ] Memory-plan alignment review completed for L2/L3/L4/L7 dependencies.
- [ ] Canvas demo tech stack decision: rendering approach, worker strategy, simulation tick rate target.
