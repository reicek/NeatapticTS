# HyperEvoDevo MorphoNEAT (ant-brain aligned plan)

**Status:** [PLANNED]

HyperEvoDevo MorphoNEAT is an evo-devo extension for NEAT inspired by how small brains (e.g., ants) emerge and adapt:

1. **DNA builds a small brain deterministically** (development).
2. **Experience gates where capacity grows** (usage-driven local expansion).
3. **Unused / costly wiring is pruned and compacted** (energy/wiring economy).
4. **Once stable, DNA is slowly updated** so future generations start closer to the discovered useful structure (structural assimilation; **no weight inheritance**).

Optional extension in this plan: offspring may use **epigenetic reference priors** (weak, two-parent anchors for parameter initialization/mutation guidance) without inheriting parent runtime weights as fixed genotype state.

This plan is intentionally constrained by the repo’s memory roadmap in [plans/Memory_Optimization.md](plans/Memory_Optimization.md). If the concept conflicts with that roadmap, the roadmap wins.

Positioning (for readers familiar with existing algorithms): HyperEvoDevo MorphoNEAT sits between **HyperNEAT** and a **small developmental language**. It keeps HyperNEAT’s compact spatial patterns (CPPN fields over a substrate), but adds a deterministic rule layer and a staged lifetime policy so evolution can propose macro motifs while experience drives local growth/prune decisions.

This draft is deliberately pragmatic: hyper features are opt-in, deterministic-by-default, budgeted, and reversible; when disabled, classic NEAT/Network behavior must remain unchanged.

## Scope and maturity

This is a **concept and architecture plan**, not an implementation-complete spec.

- **In scope:** lifecycle model, deterministic contracts, DNA composition, budget policy, cache boundaries, and acceptance criteria.
- **Out of scope (for now):** full operator-level API details, final data schemas, and low-level benchmark harness implementation.
- **Authority rule:** if this plan conflicts with `plans/Memory_Optimization.md`, the memory plan remains authoritative.

## Execution alignment (numbered tracks)

This plan executes as **Track 2** in the memory roadmap and is intentionally sequenced **after core implementation phases** that benefit the current library.

- Track 1 (Memory foundation): phases 0–10 in `plans/Memory_Optimization.md`
- Track 2 (Hyper algorithm): phases 11–16 in `plans/Memory_Optimization.md`

Authoritative mapping for this plan:

- Phase A -> Phase 11
- Phase B -> Phase 12
- Phase C -> Phase 13
- Phase D -> Phase 14
- Phase E -> Phase 15
- Phase F -> Phase 16

This preserves conceptual labels (A–F) while keeping execution tracking numeric across both plans.

## Explicit Evo-Devo positioning (naming + scope)

This plan is explicitly **Evo-Devo**:

- **Evo (evolution):** selection, crossover/mutation, and speciation optimize compact developmental DNA across generations.
- **Devo (development):** each lifetime deterministically constructs a phenotype from DNA rules/programs before experience-driven local edits.

## Design Pillars

- **Opt-in + isolated:** hyper features live behind flags and must not affect classic NEAT when disabled.
- **Pay-for-use:** no memory/time overhead unless enabled (mirrors the memory plan).
- **Budgeted growth:** every build/morph action obeys explicit caps (nodes/edges/bytes) and must be rollbackable.
- **Deterministic by default:** same DNA + seed + same experience stream ⇒ same result.
- **Epigenetic guidance is weak + optional:** parent references can nudge search early, but cannot force convergence or replace exploration.
- **Biology-inspired realism (optional):** at extreme scales, **DNA may use lossy compression** for topology/program data to stay compact. This can slightly relax determinism (documented, opt-in), similar to biological variability.

---

## Analogy to biological development (engineered mapping)

This table is shared vocabulary (conceptual), not an implementation spec.

| Stage                   | Biological inspiration                        | HyperEvoDevo MorphoNEAT engineering analog                                                              |
| ----------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Embryonic seed          | Few stem cells                                | Minimal DNA scaffold + input/output anchors + initial module archetypes                                 |
| Patterning gradients    | Morphogens / HOX genes                        | Substrate coordinates + CPPN fields + deterministic tagging (module/zone ids)                           |
| Proliferation           | Cell division                                 | DNA-driven replicate/hierarchy rules under explicit growth budgets                                      |
| Differentiation         | Neurons specialize                            | Per-module archetype params (activation family, plasticity policy, wiring-cost zone weights)            |
| Axon guidance           | Growth cones follow gradients                 | Cost-aware CPPN adjacency realization + locality bias + sparsity-first thresholds                       |
| Pruning & refinement    | Synaptic pruning                              | Experience-gated prune/compact (prefer long/inter-module edges; protect reward-critical wiring)         |
| Lifelong plasticity     | Hebbian remodeling                            | Optional plasticity (weights adapt often); structural edits are slower, local, cooldown-limited         |
| Epigenetic assimilation | Stabilized development biases next generation | Slow per-module DNA updates after equilibrium (update programs/templates/budgets; no weights inherited) |

---

## Core Concept (Engineering Mapping)

### DNA → Development → Experience → Assimilation

**DNA (genotype) is not a graph.** It’s a compact program + governance knobs that deterministically generate a phenotype and then regulate how the phenotype is allowed to change during life.

**Development (DNA-only):** execute the developmental program to build an initial scaffold.

**Experience (lifetime):** measure usage + reward sensitivity and apply slow, local structural edits:

- active/valuable parts tend to **grow** (densify edges; sometimes add nodes)
- inactive/costly parts tend to **prune/compact** (remove long/inter-module edges first)

**Assimilation (between generations):** once equilibrium is reached, encode discovered structural priors back into DNA so offspring start closer to the “good scaffold” (Baldwin-by-default, with optional assisted biasing).

### Deterministic build pipeline (high level)

This is the core “rule-first” pipeline retained from the original draft, reframed to fit the ant-brain lifecycle:

1. **Substrate:** deterministically assign coordinates and zone/module tags.
2. **Rule passes (development):** apply prioritized, deterministic rules to expand a virtual node/module plan.
3. **Indirect connectivity:** evaluate CPPN programs selectively to realize sparse edges with wiring-cost awareness.
4. **Materialize:** instantiate slab-backed runtime structures (pooling + slabs) and attach optional policy hooks.

Cache boundaries are explicit (memory plan L7): adjacency cache and phenotype slab reuse are the intended wins; anything else must be heavily gated.

### What HyperEvoDevo MorphoNEAT is (and is not)

Is:

- An evo-devo encoding (rules + CPPNs + substrate modifiers) that scales.
- A staged lifecycle controller that shifts from “grow fast” → “optimize/compact”.
- A per-module focus system that decides where to spend structural budget.

Is not:

- Weight inheritance. Weights remain lifetime state.
- A mandatory gradient/backprop system. Any “credit assignment” is expressed via black-box probes and reward deltas.

---

## Determinism, Variability, and Reproducibility

### Determinism contract (default)

- Same DNA + seed + same experience stream ⇒ identical phenotype (ordering + hash) at each lifecycle checkpoint.
- Stable ordering must be defined for: nodes, edges, module IDs, rule ordering, probe sampling.

### Safety invariants (kept short, but explicit)

- **Reproducibility:** DNA + seed + experience stream ⇒ canonical phenotype at lifecycle checkpoints.
- **Idempotence:** repeated builds with identical inputs produce identical ordering and hashes.
- **Budget enforcement:** all build/morph steps respect caps (nodes/edges/bytes/time) and are rollbackable.
- **No global side effects:** importing hyper modules when disabled must not mutate global state.
- **Lazy diagnostics:** traces/telemetry allocate only when enabled.

### Optional biological variability (at scale)

DNA may contain **compressed topology/program fragments**. For very large brains, the encoding may be:

- **lossless** (canonical) by default
- **lossy** (quantized / approximated) when enabled to cap DNA size

When lossy mode is enabled, deterministic reconstruction is still “best effort” but may vary slightly across platforms/versions due to quantization/rounding details. This must be explicitly surfaced via:

- a `dna.encodingMode` field
- a `dna.compatibilityVersion`
- a reproducibility warning in telemetry

---

## HyperDNA: a compact, evolvable “instruction set”

DNA should be a dedicated class (conceptually `HyperDNA`) with explicit schema versioning, canonical (lossless) encoding, and optional compressed fragments for large-scale topology/program data.

### Responsibilities

- Provide deterministic development inputs (rules/CPPNs/substrate).
- Provide lifecycle schedule + stage transition goals.
- Provide budgets + wiring-cost preferences.
- Define what experience probes exist and their cadence.
- Support per-module assimilation and compressed per-module directives.

### High-level shape (conceptual)

DNA is composed of multiple **parts**. Some parts may be provided as **OR alternatives** (decode the first viable option under budgets).

- Core:
  - seed policy (including “siblings differ by seed”)
  - substrate spec and coordinate system
  - developmental program (rules + CPPN programs)
  - module archetypes and module addressing scheme
- Governance:
  - lifecycle schedule (stages/substages)
  - budgets (nodes/edges/bytes/time)
  - wiring-cost weights (soft pressure; evolvable)
  - probe schedule (cheap every epoch; expensive on life events)
  - assimilation policy (when/how DNA is updated)
- Encoded payloads (optional):
  - compressed per-module directives (matrices/tables)
  - compressed CPPN parameter blocks
  - compressed topology templates or adjacency hints

### Canonical DNA structure (conceptual)

The intent is that DNA is an explicit, versioned object that can be serialized, hashed, crossed-over, and partially re-written during assimilation.

Conceptually, a `HyperDNA` instance contains:

- **Identity & compatibility**
  - `schemaVersion`: increments on schema changes
  - `compatibilityVersion`: pins decoding/canonicalization rules
  - `encodingMode`: `lossless` (default) or `lossy` (large-scale)
  - `fingerprint`: hash of the canonical (or canonicalized+quantized) representation
- **Reproduction seed policy**
  - `seedPolicy`: siblings differ by seed by default; twins are allowed by sharing the seed
- **Substrate & coordinate system**
  - substrate dimensions, coordinate normalization, zone partitioning strategy
- **Developmental program (“what gets built”)**
  - rule passes (replicate/symmetry/hierarchy/differentiate)
  - one or more CPPN programs (potentially per module archetype)
  - substrate modifiers (scale/rotation/etc.)
- **Module system (“where things live”)**
  - module archetypes (repeatable motifs)
  - module addressing scheme (stable IDs and ordering)
  - per-module directives (optional, may be compressed)
- **Governance (“how life proceeds”)**
  - stage/substage schedule and goal checks
  - budgets (nodes/edges/bytes/time)
  - wiring-cost preferences (soft pressure, evolvable)
  - probe schedule (cheap each epoch; expensive on life events)
  - morph policy knobs (growth/prune cooldowns, hysteresis)
- **Payloads (“extra compressed knowledge”)**
  - optional compressed matrices/tables and quantized parameter blocks
  - optional alternative encodings (“OR parts”) for viability-first decode

This structure deliberately matches the memory plan’s stance: large-scale performance comes from slabs, sparsity, and cache reuse; DNA must stay compact and not embed huge per-edge data.

### Module addressing and zones

To keep builds deterministic and caches effective, DNA needs a clear addressing model:

- **Module IDs are stable** within a DNA instance and must be reproducible across builds.
- **Zones** are a coarse partitioning of the substrate (for wiring-cost defaults and budgets).
- **Archetypes** define repeatable parameter sets; modules reference archetypes + small parameter deltas.

Recommended conceptual rules:

- Module ordering is canonical: sort by `(zoneId, archetypeId, moduleOrdinal)`.
- Any rule pass that creates modules must emit deterministic module IDs (no iteration-order accidents).
- Wiring-cost defaults can be zone-based (ant-like metabolic regions): if a module does not define cost weights explicitly, inherit from its zone.

### Compressed payloads (ideas that scale)

When DNA must encode more detail without exploding size, prefer encodings that compress _patterns_:

1. **Dictionary-coded archetypes**
   - Store a small dictionary of archetype definitions; modules store small indices + a few deltas.

2. **Quantized parameter blocks (lossy optional)**
   - Quantize floats (e.g., CPPN weights, rule coefficients) into `int8/int16` blocks.
   - Store a scale/offset header per block so decoding is deterministic within a `compatibilityVersion`.

3. **Sparse SoA hints (lossless, compact)**
   - Store sparse local hints (e.g., “preferred local wiring bumps” or “mask biases”) as small SoA lists.

4. **Run-length / delta coding for tables**
   - For any per-module table, prefer RLE of repeated values and delta coding of monotone sequences.

5. **String-level encodings (transport-friendly)**
   - Hex for very small payloads (debuggable).
   - Base64 for larger byte blocks.
   - Keep decoding rules fixed under `compatibilityVersion`.

Important constraint: compressed payloads may help _bias_ connectivity, but the canonical large-scale structure should still come from CPPNs/rules + sparsity + budgets; this keeps DNA size small relative to phenotype.

### “OR parts” decode (viability-first) — what it means

The “OR parts” mechanism is how DNA can stay expressive but still fit budgets:

- DNA may carry multiple alternative representations for a part.
- Decoding chooses the first option that is:
  1. compatible with the runtime (`compatibilityVersion`)
  2. within current budgets (nodes/edges/bytes/time)
  3. consistent with determinism/variability mode (`encodingMode`)

This is especially useful for large CPPN blocks or per-module directives: lossless is preferred, but a compact/lossy alternative can be selected when budgets demand.

### Assimilation write-back (per-module, compact)

Assimilation must not “bloat DNA into a saved phenotype”. A good rule of thumb:

- Write back **generators and knobs**, not explicit edges.
- Prefer updating:
  - archetype parameters
  - rule distributions (priorities/probabilities)
  - CPPN program topology and quantized parameter blocks
  - budgets and wiring-cost weights per zone/module
  - schedule thresholds (plateau detection, marginal return thresholds)

If assimilation needs to store a module-specific detail, store it as:

- a small sparse hint table, or
- a reference to an archetype variant (dictionary entry), or
- a compact delta from the archetype default

This keeps DNA evolvable and makes crossover/innovation tracking feasible.

---

## Encoding ideas (per-module, compact, evolvable)

This section is conceptual: it defines what “compressed DNA” means without committing to implementation.

### Principle: encode patterns, not instances

Rather than store explicit edges, DNA should store **generators** and **templates**:

- CPPNs (fields over coordinates)
- rule passes (replicate/symmetry/hierarchy/differentiate)
- module archetypes (repeatable motifs)
- small matrices or lookup tables that parameterize a generator

### Per-module encoding strategies

Prefer one of these representations per module (or module archetype):

1. **Template + parameters**
   - “This module is archetype A with parameters θ”
   - Best for compression and evolution stability

2. **Sparse matrix (SoA) hints**
   - Store a compact list of (local coordinate, weight seed/scale, mask bias)
   - Can be encoded as hex/base64 for compactness

3. **Quantized parameter blocks (lossy optional)**
   - Quantize floats into int8/int16 blocks
   - Accept mild variability at scale; document encoding mode

### “OR parts” (viability-first decode)

DNA may contain multiple alternative encodings for a part:

- Option A: lossless canonical rules/CPPN
- Option B: compressed (lossless)
- Option C: compressed (lossy)

Decode picks the first option that fits current budgets and compatibility constraints.

### DNA recombination (high level)

HyperDNA crossover must align heterogeneous families. This keeps the original draft’s clarity without locking in implementation details.

| DNA part           | Alignment key                       | Match rule                    | Crossover outcome                                            |
| ------------------ | ----------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| Rules              | (kind + canonical param signature)  | same kind + normalized params | uniform pick or parameter-wise blend                         |
| CPPN program       | topology hash + activation sequence | identical topology            | weight-block crossover (uniform/arithmetic) or select fitter |
| Substrate modifier | (type + axis)                       | same type/axis                | numeric blend + deterministic tie-break                      |
| Module archetype   | archetype id                        | id equality                   | inherit or blend parameters                                  |
| Budgets / schedule | field key                           | same field                    | blend within safe bounds, then clamp to global caps          |

Excess/disjoint DNA parts follow a fitter-biased rule, but must pass a budget/viability normalization step.

### Epigenetic reference priors (two-parent, no inheritance)

This plan supports an opt-in epigenetic mechanism where offspring receive **two soft reference anchors** derived from both parents:

- a reference can be formed for homologous parameters (for example aligned rule coefficients, CPPN block parameters, or weight/bias initializers)
- offspring remain free to explore; references are weak priors, not hard constraints
- references are applied at birth-time initialization or mutation-time biasing, then decay over early lifecycle stages unless still beneficial

Design intent:

- preserve the core rule: **no direct weight inheritance as genotype state**
- allow both parents to influence offspring in different regions, increasing alternative trajectories instead of collapsing diversity
- keep this feature deterministic and budgeted when enabled

Conceptual mutation-time form (small $\lambda$):

$$
	heta \leftarrow \theta + \Delta\theta_{mutation} + \lambda\,\left(\theta^* - \theta\right)
$$

Where $\theta^*$ is a two-parent reference (for example blend/select across homologous parent values) and $\lambda$ is intentionally small.

---

## Lifecycle: 4 stages × 4 substages (goal-gated)

The lifecycle is a **state machine**. Each stage has 4 substages; each substage has a goal. If the goal isn’t met, repeat another cycle.

### Stage 1: Embryo (DNA-only development)

1. Anchor: build minimal scaffold (inputs/outputs + base modules).
2. Pattern: apply deterministic rule passes (replicate/symmetry/hierarchy/differentiate).
3. Wire: generate sparse connectivity (CPPN/cost-aware thresholds).
4. Stabilize: verify budgets + hashes + baseline forward correctness.

### Stage 2: Juvenile (grow fast where useful)

1. Observe: collect cheap usage signals (utilization/activity + wiring metrics).
2. Probe: sample reward sensitivity via perturbation probes (scheduled; can be expensive).
3. Expand: local growth (edge densification first; node add allowed, but slower).
4. Consolidate: cool down edited modules; rebuild caches; enforce budgets.

### Stage 3: Adult (optimize/compact more than grow)

1. Observe: monitor for stagnation and marginal returns.
2. Prune: remove long/inter-module/costly edges first unless protected by contribution.
3. Compact: reduce structure where performance holds (seek smaller equivalent).
4. Maintain: allow limited growth only when strong evidence exists.

### Stage 4: Equilibrium (assimilate slowly)

1. Detect: plateau + low marginal returns triggers equilibrium candidate.
2. Validate: re-check stability across multiple cycles/rollouts.
3. Assimilate: update DNA per-module (rules/CPPN/topology templates/budgets/schedules).
4. Reset: start next generation from updated DNA with new child seed.

Notes:

- “Life events” = plateau events and scheduled probe checkpoints.
- Budgets are part of DNA. If missing, default budgets are derived from a conservative baseline.

### Lifecycle timeline (artifacts & rollback boundaries)

This table is a reviewer-friendly view of “what mutates where” and what artifacts are produced.

| Step                               | Mutates              | Produces                                               | Rollback boundary                                              |
| ---------------------------------- | -------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| Development (Embryo)               | Phenotype only       | virtual plan + realized adjacency + materialized slabs | abort build on budget/invalid refs (no commit)                 |
| Observe/Probe (Juvenile/Adult)     | metrics buffers only | focus scores + probe results                           | drop metrics on failure; no structural commit                  |
| Morph cycle (Grow/Prune/Compact)   | Phenotype only       | delta edits + trace entries                            | dry-run validate then commit; rollback on constraint violation |
| Equilibrium validation             | none (decision step) | “stable” decision + candidate assimilation set         | if unstable, return to Adult cycles                            |
| Assimilation (between generations) | DNA only (slow)      | updated per-module DNA parts                           | if invalid/off-budget, keep prior DNA                          |

---

## Experience signals and probes (black-box friendly)

Default signals (cheap, per epoch):

- utilization/activity statistics
- wiring metrics (edge count, mean length, inter-module ratio)
- novelty (structural change rate, focus distribution entropy)

Value signal (more expensive, scheduled):

- reward delta under perturbation:
  - lesion/ablation (temporarily disable module/edges)
  - noise injection (weights/activations)
  - gating test (disable long edges or inter-module edges)

This aligns with “used = improves reward when perturbed” without requiring gradients.

### Focus scoring (conceptual, no gradients required)

We treat “where to change topology” as an allocation problem. Each module/zone receives a focus score computed from cheap signals and (optionally) sparse perturbation probes.

Candidate signals (all optional, DNA-selectable):

- `utilization`: activation/firing summary (cheap)
- `rewardDelta`: reward change under perturbation (expensive; sampled)
- `novelty`: structural change rate / exploration bonus (cheap)
- `stabilityAge`: time since last beneficial change (cheap)
- `wiringCost`: mean edge length, inter-module ratio, edge count (cheap)

One workable default form:

$$
focus(m) = w_u\,\widehat{util}(m) + w_r\,\widehat{rewardDelta}(m) + w_n\,\widehat{novelty}(m) + w_s\,\widehat{stabilityAge}(m) - w_c\,\widehat{wiringCost}(m)
$$

Then sample targets via softmax/top-k, apply local edits under budgets + cooldown, and log before/after deltas for traceability.

---

## Morphogenesis policies (slow, local, hysteresis)

Structural edits are allowed but must be:

- local (module-scoped)
- slow (cooldowns)
- hysteretic (growth needs strong evidence; prune needs sustained underuse/cost)
- budgeted + rollbackable

Preferred edit order under stagnation:

1. edge densification in high-focus modules
2. node additions (rare; only when marginal returns justify)
3. larger structural templates (replication/hierarchy changes) via DNA assimilation rather than rapid lifetime edits

---

## Wiring cost (soft pressure; evolvable; ant-like)

Wiring economy is treated as a first-class soft pressure across:

- development (cost-aware edge realization)
- morphogenesis (prune long/inter-module first)
- evolution (selection pressure; evolvable cost weights)

This section stays compatible with [plans/Memory_Optimization.md](plans/Memory_Optimization.md) and its sparsity-first strategy.

---

## Caching (must follow the memory plan)

Caching priorities (authoritative):

1. **Adjacency cache** (L7): keyed by genotype/substrate/program signatures; stores SoA edge lists.
2. **Phenotype slab reuse** (L7): reuse typed array slabs and avoid rebuild churn.

Optional (niche, strongly gated):

- module-level input/output caching only if inputs are quantized/discrete and the module is frozen/state-free.

---

## Alignment with the Memory Optimization Plan (authoritative)

HyperEvoDevo MorphoNEAT depends on the completed memory foundations:

- L2 pooling + L3 slabs for churn
- L4 sparsity + budgets for the primary bytes/connection improvements
- L7 caching for adjacency/phenotype reuse

All memory flags, constants, and accounting must be sourced from the “Centralized Memory Manager” described in [plans/Memory_Optimization.md](plans/Memory_Optimization.md).

---

## Agreed defaults (so readers don’t guess)

- **No weight inheritance:** weights are lifetime state; DNA encodes structure + policies.
- **Epigenetic priors are allowed but weak:** two-parent references may guide initialization/mutation; they are not persistent inherited weights.
- **Per-module assimilation:** equilibrium triggers per-module DNA updates (templates/rules/CPPN programs/budgets/schedules).
- **Siblings differ by seed (default):** reproduction yields the same DNA with different child seed values.
  - **Twins are allowed:** identical DNA + identical seed.
- **Wiring economy is soft pressure, evolvable:** wiring-cost weights live in DNA and evolve; if missing, defaults may be sampled conservatively.
- **Probe scheduling lives in DNA:** cheap metrics every epoch; expensive perturbation probes run on cadence and on “life events” (plateau triggers).

---

## Recurrence policy (ant-like, but phased)

Ant brains are recurrent, but recurrence expands the state space and complicates caching and deterministic evaluation.

- Default: recurrence **off** in early phases.
- Later (explicit DNA trait): recurrence **on** once evaluation is formally time-stepped and windowed execution (aligned with Phase 9 concepts in the memory plan) is available.

---

## Assimilation (what changes in DNA, at a high level)

Assimilation is a slow write-back of **structural priors**, not weights.

Per-module assimilation should prefer updating **generators** over storing explicit adjacency:

- Update rule params/priorities (replication depth, hierarchy levels, symmetry usage).
- Update CPPN programs (topology + optionally quantized parameter blocks; still no weight inheritance).
- Update wiring-cost weights and budgets per module/zone.
- Update lifecycle schedule knobs (growth vs prune emphasis, probe cadence).

Optional “assisted biasing” (conceptual): DNA may carry a parent-snapshot hint used to bias early development until plateau, then abandon the bias if stagnant.

Epigenetic references should prefer **parameter priors** over explicit adjacency snapshots:

- store compact parent-derived priors (for example archetype deltas, CPPN block priors, initializer moments)
- apply priors with a decay schedule across early lifecycle stages
- disable/anneal priors when they reduce reward deltas or novelty contribution

---

## Budget defaults (when DNA omits them)

Budgets are part of topology. If DNA omits a budget field, resolve it via a **deterministic seeded policy**:

1. Use the child seed and `compatibilityVersion` as deterministic inputs.
2. Derive conservative defaults around a baseline (for example, near `0.5` of configured maxima).
3. Persist resolved values into telemetry/checkpoints for auditability.

This preserves reproducibility while still allowing controlled diversity between siblings.

---

## Phased roadmap (high level, non-duplicative)

This plan intentionally stays high level and defers implementation details.

Phase A — DNA + deterministic development

- Define `HyperDNA` schema + versioning and canonical encoding.
- Deterministic development pipeline: substrate → rules → indirect wiring → materialize.
- Determinism tests: stable build hash under repeated rebuilds.

Phase B — Juvenile focus + local growth/prune

- Focus metrics (cheap) + scheduled perturbation probes (expensive).
- Local growth first, then prune/compact with hysteresis.
- Churn tests aligned to the memory plan (pool high-water mark slope ~0).

Phase C — Adult optimization + equilibrium detection

- Plateau detection + marginal returns tracking.
- Growth cooling + prune/compact dominance.

Phase D — Assimilation (per-module DNA update)

- Encode stable structural priors back into DNA (rules/CPPN/topology templates/budgets/schedule).
- Optional lossy compression mode for very large structures (explicitly documented).

Phase E — Evolution integration

- Multi-family crossover and mutation for DNA parts (rules/CPPN topology/substrate modifiers/budgets/schedules).
- Speciation distance extends to DNA programs and wiring-cost preferences.
- Optional epigenetic prior operator: two-parent weak anchors for initialization/mutation-time biasing with deterministic decay.

Phase F — Scale & stress validation

- Stress at high edge counts with the memory plan’s benchmark harness.
- Validate: cache hit ratio, bytes/connection, rebuild variance, churn leak slope.

---

## Acceptance criteria (concept-level)

- Determinism (default mode): same DNA+seed+experience stream yields identical hashes at lifecycle checkpoints.
- Memory: must not regress memory plan targets; sparsity/budgets are the primary lever.
- Caching: adjacency cache hit ratio target > 70% in repeated-evaluation scenarios (bench-defined).
- Churn safety: pool high-water mark stabilizes under repeated grow/prune cycles.
- Wiring economy: long/inter-module edges are preferentially pruned under cost pressure.

## Readiness checklist (for implementation start)

Before coding begins, this plan should have all items below marked complete:

- [ ] Canonical `HyperDNA` schema draft with explicit versioning fields.
- [ ] Deterministic ordering contract for module IDs, rule order, and edge realization.
- [ ] Budget resolution algorithm defined as deterministic (seeded) and testable.
- [ ] Lifecycle transition guards defined with measurable thresholds.
- [ ] Telemetry contract finalized for reproducibility warnings (`encodingMode`, compatibility, lossy flags).
- [ ] Memory-plan alignment review completed for L2/L3/L4/L7 dependencies.
