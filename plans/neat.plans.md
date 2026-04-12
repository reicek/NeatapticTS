# Proper NEAT (Reference-Quality) — Phase 1 Plan

**Status:** [DONE]

## Scope

This workstream covers the Phase 1 critical-path proper-NEAT item from [Roadmap.md](Roadmap.md) and now records its closed state. Its job was to turn NeatapticTS into a reference-quality proper-NEAT implementation: academically correct historical markings, deterministic reproduction, explicit recurrent policy, checkpoint-safe innovation tracking, SOLID code boundaries, and educational documentation that teaches the whole lifecycle from generation zero through export/import.

This plan intentionally separates three bars:

- Canonical NEAT compliance: historical markings, meaningful speciation, principled crossover, and minimal-start complexification.
- Reference-library quality: deterministic replay, migration-safe checkpoints, clear public contracts, validator-backed invariants, and educational docs.
- Beyond-paper upgrades: genotype-first architecture, recurrent/gated extensions, modern caching, and later population-scale acceleration.

Not in scope for the first lift:

- pulling Phase 2 builder work forward,
- treating GPU or tensorization as the primary design constraint before correctness is frozen,
- hiding non-canonical fallback behavior behind silent defaults once proper NEAT is the advertised mode.

## Reference Bar

The standard for this plan is intentionally higher than “make the tests pass.”

- Paper compliance means matching the core claims of Stanley and Miikkulainen’s NEAT paper: principled crossover across different topologies, protection of structural innovation through speciation, and growth from minimal structure instead of starting complex. Source: <https://nn.cs.utexas.edu/?stanley:ec02>.
- Strict innovation semantics should follow the original paper’s generation-scoped de-duplication rule for structural mutations. The repo’s current persistent innovation maps are useful scaffolding, but they are not yet the final academic behavior.
- Modern community expectation now treats innovation numbers as mandatory implementation data, not a nice-to-have. NEAT-Python’s recent innovation-number work is a strong external signal that users now expect checkpoint-safe, always-on historical markings. Source: <https://neat-python.readthedocs.io/en/latest/innovation_numbers.html>.
- Recurrent and gated extensions are legitimate and useful, but only after canonical connection-gene semantics are stable. The recurrent lane must sit on top of correct historical markings instead of patching around them. Source: <https://arxiv.org/abs/1904.06239>.
- Performance matters, but correctness comes first. Modern acceleration work such as TensorNEAT shows where the later optimization ceiling is, not what should drive the first correctness refactor. Source: <https://arxiv.org/abs/2504.08339>.

## Current State

- Completed milestone history through Phase 6, Step 7.1, and closed Step 7.2 now lives in [neat.logs.md](neat.logs.md).
- The canonical proper-NEAT baseline is closed in the repo: explicit historical markings, innovation-tracker-backed structural reuse, innovation-aligned crossover, recurrent-policy hardening, strict native compatibility, deterministic replay, and source-first generated docs.
- Phase 7 is closed. Steps 7.2 through 7.6 are complete, no active frontier remains inside this tracker, and any later beyond-paper continuation is reopen-only and explicitly opt-in.
- The active guardrails remain unchanged: `Network` stays the phenotype boundary, replay and checkpoint semantics stay controller-owned, runtime-only validation stays outside the genome contract, and helper, mutation, and export owner moves stay deferred until narrower seams are justified.
- Known unrelated blockers remain outside this plan: lint issues in `src/architecture/layer/layer.factory.normalization.utils.test.ts`, `src/architecture/network/network.ts`, `src/neat/species/core/shared/species.core.shared.ts`, and `src/neat/topology-intent/neat.topology-intent.ts`, plus the separate add-node compile-time test issue tracked elsewhere.

## Non-Negotiable Architecture Rules

- Any touched source file under `src/neat/**` or `src/architecture/network/**` gets source-first educational JSDoc improvement before the phase can close.
- Any doc-affecting phase ends with `npm run docs` so generated README surfaces stay synchronized.
- Keep top-level methods orchestration-first. Put branching logic into pure helpers or narrow services. If a file stops being teachable, split it before adding more logic.
- Preserve current public APIs unless a migration is explicitly justified in the phase notes.
- Use ES2023-first patterns when they improve clarity or safety: `toSorted`, `at`, `structuredClone` or project-safe equivalents, spread over `Object.assign`, typed arrays where appropriate, and named constants over magic numbers.
- Native NEAT genomes must not silently rely on fallback innovations once proper NEAT is the default. Fallback remains a legacy/import bridge.
- Performance work is welcome only when it does not blur correctness boundaries. Any performance-oriented change must keep deterministic parity and benchmark evidence.

## Coverage Backlog

### [DONE] Phases 0 through 6 — Canonical proper-NEAT baseline

Goal: keep the academically correct proper-NEAT lane stable before optional extensions start.

- Closed coverage: contract and serialization safety, fail-fast native validation, explicit innovation-tracker semantics, homologous generation-zero identity, innovation-aligned crossover with controller-owned RNG, recurrent-policy hardening, strict native compatibility and species-history expectations, deterministic checkpoints and replay, and educational-docs follow-through.
- Durable milestone log: [neat.logs.md](neat.logs.md).

### [DONE] Phase 7 — Beyond-paper, still reference-quality

Goal: exceed the 2002 paper without contaminating canonical semantics.

Phase intent:

- This is a local continuation lane inside the proper-NEAT plan, not the repo-wide Roadmap Phase 7.
- Treat this phase as optional and explicitly gated: it should start only if the roadmap decision is to keep extending the proper-NEAT boundary instead of closing Phase 1 after the canonical lift.
- Preserve the freeze established by Phases 0 through 6: historical markings, deterministic replay, strict native compatibility, and source-first docs are already the baseline, not a moving target.
- Make the beyond-paper value proposition easy to find in documentation: the README opening for the owning NEAT chapter and any flagship example surface should explain what Phase 7 adds, why those gains are opt-in, and why the canonical proper-NEAT contract still remains the baseline.

Entry gate:

- Phase 6 wording is clean on native versus fallback behavior and no longer teaches index-aligned crossover as the intended model.
- The focused proper-NEAT guard slice still passes on the current codebase.
- There is concrete pressure to narrow ownership further, not just a desire to add features into broad runtime files.

Architecture guardrails:

- `Network` stays the phenotype/runtime boundary; any new genotype surface must describe heredity and mutation state without becoming a second runtime.
- New beyond-paper features must be opt-in, versioned in checkpoint/export contracts, and documented as extensions rather than folded into canonical defaults.
- Compatibility distance, crossover matching, and deterministic replay stay defined by the canonical Phase 0 through 5 contract unless an extension explicitly advertises a different formula or policy.
- If a lane needs large-scale cache, worker, or deployment changes, hand it off to the existing roadmap plans instead of expanding this phase into a catch-all research bucket.
- When a beyond-paper feature materially changes how the library should be pitched, capture that value proposition in one easy-to-find README opening instead of leaving it buried inside deep implementation chapters or tracker notes.

### [DONE] Step 7.1 — Introduce a first-class genome boundary.

Outcome:

- Land one strict genome contract and hard adapter set under `src/neat/genome/` while keeping `Network` as the executable phenotype.
- Closed coverage: the genome chapter and structural validator landed, export/import plus compatibility and validate now normalize through the genome boundary, and deferred-owner confirmation kept helpers, mutation, and innovation allocation out of the genome contract.
- Durable milestone log: [neat.logs.md](neat.logs.md).

[DONE] Step 7.2 — Re-home heredity and mutation operators behind the genome surface.

Outcome:

- Re-home the smallest clearly genome-owned heredity slice first while keeping `Network.crossOver()` and all live mutation and repair paths behaviorally stable.
- Freeze runtime and controller owners up front so Step 7.2 does not accidentally absorb generation-zero, checkpoint, or live phenotype-edit responsibilities into `src/neat/genome/`.
- Closed coverage: Passes 7.2a through 7.2d kept runtime crossover ownership in `src/architecture/network/genetic/`, confined the genome move to `src/neat/genome/heredity/`, and confirmed helper, mutation, export, and runtime-only validation owners stayed deferred.
- Durable milestone log: [neat.logs.md](neat.logs.md).

Step 7.2 owner freeze during the opening pass:

- `src/neat/genome/` may own pure heredity selection logic over `NeatGenome` plus deliberate adapter views only.
- `src/architecture/network/genetic/network.genetic.utils.ts` remains the public runtime crossover facade.
- `src/architecture/network/genetic/network.genetic.setup.utils.ts` keeps crossover context creation, RNG resolution, offspring node-count choice, and provisional runtime node assignment.
- `src/architecture/network/genetic/network.genetic.materialize.utils.ts` keeps phenotype rebuilding, runtime node-index rebuilding, topology-intent pruning, and gating reattachment.
- `src/neat/helpers/neat.helpers.ts` keeps generation-zero template creation, canonicalized starter innovations, and innovation-tracker reseeding.
- `src/neat/mutation/shared/mutation.types.ts`, `src/neat/mutation/mutation.ts`, `src/neat/mutation/add-node/`, `src/neat/mutation/add-conn/`, `src/neat/mutation/select/`, `src/neat/mutation/flow/`, and `src/neat/mutation/repair/` keep live mutation descriptors and `connect`/`disconnect`-based structural edits.
- `src/neat/export/` keeps checkpoint, controller-meta, runtime-hint, and replay bridging.
- `src/neat/validate/` keeps runtime-only phenotype guards; the validator split stays unchanged during Step 7.2 opening.

Smallest safe first extraction slice:

- Extract connection-gene collection and innovation-aligned inheritance choice from `src/architecture/network/genetic/network.genetic.selection.utils.ts` plus the delegation seam in `src/architecture/network/genetic/network.genetic.setup.utils.ts` into a genome-owned heredity helper.
- This is the smallest safe slice because selection already normalizes parents through the strict genome adapter, while setup and materialization still depend on live `Network` scaffolding, runtime node indexes, and gating reattachment.
- Keep node selection, runtime scaffold creation, and offspring materialization on the runtime shelf for now.

Proposed first-pass boundary under `src/neat/genome/heredity/`:

- `genome/heredity/genome.heredity.ts` as the orchestration-first heredity-selection entry.
- `genome/heredity/genome.heredity.types.ts` for parent-view and selection-result contracts if the extraction needs more than the current `ConnectionGene` shape.
- `genome/heredity/genome.heredity.utils.ts` for pure collection and innovation-aligned selection helpers.
- Keep phenotype materializers, mutation services, and controller checkpoint bridges out of this first Step 7.2 slice.

Implementation passes:

- [DONE] Pass 7.2a — Genome-owned heredity selection: landed `src/neat/genome/heredity/` as the first strict-genome heredity owner, moved innovation-aligned connection-gene collection plus inheritance choice behind that boundary, and kept the runtime crossover facade plus phenotype materialization stable.
- [DONE] Pass 7.2b — Runtime adapter hardening: audited the remaining runtime adapter seam, removed dead parent-local node-index hints from the runtime `ConnectionGene` materialization descriptor, and kept the runtime materializer consuming only stable gene ids plus inherited weight, enabled state, and innovation identity.
- [DONE] Pass 7.2c — Deferred-owner confirmation: audited the thinner runtime adapter seam and confirmed `src/neat/helpers/`, `src/neat/mutation/`, and `src/neat/export/` still own generation-zero bootstrap, live structural edit and repair, and checkpoint or replay concerns without importing the runtime crossover shelves; kept the runtime-only validator split unchanged.
- [DONE] Pass 7.2d — Closure audit: confirmed `src/architecture/network/genetic/network.genetic.utils.ts`, `network.genetic.setup.utils.ts`, and `network.genetic.materialize.utils.ts` still own the runtime crossover facade, setup, and phenotype materialization shelves; confirmed the genome move stopped at heredity selection; and closed Step 7.2 without another operator move.

Completed in the current repo state:

- Owner audit confirmed the smallest safe first move is heredity selection only, and the runtime-only validator split stays unchanged.
- Pass 7.2a landed `src/neat/genome/heredity/` and reduced `src/architecture/network/genetic/network.genetic.selection.utils.ts` to a thin materialization adapter while keeping runtime setup, materialization, and public crossover ownership stable.
- Pass 7.2b narrowed that adapter further: `src/architecture/network/network.types.ts` no longer lets the runtime `ConnectionGene` materialization descriptor carry `from`, `to`, or `gater` node-index hints, and `src/architecture/network/genetic/network.genetic.selection.utils.ts` now forwards only stable heredity identity plus the inherited weight and enabled state.
- `src/architecture/network/genetic/network.genetic.materialize.utils.ts` now documents the stricter seam explicitly, and `src/architecture/network/genetic/network.genetic.test.ts` now proves the adapter output strips runtime index hints while materialization still resolves endpoints and gaters through stable gene ids.
- Pass 7.2c confirmed the deferred owners stayed fixed after the adapter thinning: `src/neat/helpers/neat.helpers.ts` still owns generation-zero template normalization plus tracker reseeding, `src/neat/mutation/mutation.ts` plus `repair/` still own live structural edits and repair, and `src/neat/export/neat.export.ts` still owns checkpoint and replay bridging without importing the runtime crossover shelves.
- Pass 7.2d closed the boundary audit: `src/architecture/network/genetic/network.genetic.utils.ts` still owns the public runtime crossover facade, `src/architecture/network/genetic/network.genetic.setup.utils.ts` still owns runtime setup plus the heredity delegation seam, and `src/architecture/network/genetic/network.genetic.materialize.utils.ts` still owns phenotype materialization by stable gene id, so no further operator move is required for Step 7.2.
- `src/neat/validate/neat.validate.ts` still owns runtime-only phenotype guards while strict genome-contract checks continue to delegate through `src/neat/genome/` instead of widening the validator split.
- Adjacent guard hardening landed only where required to keep the validation slice stable: `src/neat/genome/genome.utils.ts` now canonicalizes runtime node order before strict validation, and `src/neat/evolve/population/evolve.population.utils.ts` now refreshes or filters unusable species state before speciated breeding.
- Validation for the current repo state remains the existing `npm run build` plus a focused 10-suite / 102-test guard slice covering `network.genetic`, `genome`, `genome.heredity`, `helpers`, `export`, `compat`, `validate`, `mutation.add-conn`, `innovation-tracker`, and `speciation`. Pass 7.2d landed tracker-only closure evidence, so no new build, focused guard rerun, or `npm run docs` pass was required. `npm run lint` still reports only the same unrelated existing blockers.
- Durable milestone log: [neat.logs.md](neat.logs.md).

Current working boundaries:

- Genome heredity owner: `src/neat/genome/heredity/`.
- Runtime adapter seam: `src/architecture/network/genetic/network.genetic.selection.utils.ts` now forwards only the stable materialization descriptor consumed by runtime materialization.
- Runtime materializer seam: `src/architecture/network/genetic/network.genetic.materialize.utils.ts` now resolves endpoints and gaters only by stable gene id.
- Runtime owners that stayed fixed through Step 7.2 closure: `src/architecture/network/genetic/network.genetic.utils.ts`, `src/architecture/network/genetic/network.genetic.setup.utils.ts`, and `src/architecture/network/genetic/network.genetic.materialize.utils.ts`.
- Adjacent guard seams: `src/neat/genome/genome.utils.ts` and `src/neat/evolve/population/evolve.population.utils.ts`.

Step 7.2 coupling points to actively guard:

- Moving setup or materialization too early would pull runtime scaffold creation, node indexing, topology pruning, and gating reattachment into the genome boundary.
- Moving generation-zero helpers too early would mix startup population normalization and tracker reseeding into heredity ownership.
- Moving mutation descriptors too early would entangle genome heredity with `methods.mutation` policy objects and live `connect`/`disconnect` services.
- Moving export too early would blur checkpoint, controller-meta, and runtime-hint bridging with operator ownership.
- Narrowing native validation here would risk duplicating endpoint, gater, cache, and runtime-topology checks that still belong to phenotype validation.

Validation evidence used to close Step 7.2:

- `npm run build`
- targeted tests in `src/neat/genome/heredity/genome.heredity.test.ts`, `src/neat/genome/genome.test.ts`, and `src/architecture/network/genetic/network.genetic.test.ts`, plus `src/neat/export/neat.export.test.ts`, `src/neat/compat/compat.test.ts`, `src/neat/validate/neat.validate.test.ts`, `src/neat/mutation/add-conn/mutation.add-conn.test.ts`, `src/neat/innovation-tracker/innovation-tracker.test.ts`, and `src/neat/speciation/speciation.test.ts` when adjacent guard paths are touched
- `npm run docs` only when source-first JSDoc changes land under the touched heredity or genome chapters
- `npm run lint` only when new public types or new boundary files change; unrelated existing lint blockers remain deferred

[DONE] Step 7.3 — Add modern gene attributes as explicit extension state.

- Introduce opt-in node-gene and connection-gene traits such as activation-function mutation, response parameters, and clearer enable/disable semantics behind explicit feature flags or controller options.
- Serialize these traits additively and version them so canonical checkpoints stay readable while extension checkpoints remain deterministic.
- Decide trait by trait whether compatibility distance should ignore, weight, or separately account for them; document the chosen rule before changing formulas.
- Extend validator coverage so malformed extension genomes fail fast rather than degrading into runtime surprises.

Implementation passes:

- [DONE] Pass 7.3a — Opt-in connection-gain extension state: landed the first additive Step 7.3 trait through `options.genomeExtensions.connectionGain` plus direct genome-adapter capture options, taught runtime JSON to preserve connection gain and top-level extension bags, stored non-neutral ungated gains inside `extensions.values.connectionGainByInnovation`, kept canonical compatibility explicitly ignoring that extension bag, and hardened strict validation plus export/import replay around the new contract.
- [DONE] Pass 7.3b — Opt-in node-response and disabled-connection re-enable extension state: added runtime `Node.response` with a neutral default of `1`, taught runtime JSON plus the strict genome adapters to carry non-neutral node response and explicit disabled-gene re-enable probability through `extensions.values.nodeResponseByGeneId` and `extensions.values.disabledConnectionReenableProbability`, kept canonical compatibility extension-agnostic, and updated export/import to capture strict genomes from the live runtime genome so `_reenableProb` becomes checkpoint-safe extension state instead of a controller-meta-only fallback.
- [DONE] Pass 7.3c — Activation-function audit and boundary hardening: confirmed that `MOD_ACTIVATION` is already a canonical runtime and genome trait through the base node-gene `squash` field rather than a missing Step 7.3 extension, verified that export/import preserves activation mutation without `genomeExtensions`, and pinned the current compatibility rule that activation deltas remain ignored unless a later pass deliberately changes the formula.

Completed in the current repo state:

- Activation-function mutation is not an unlanded Step 7.3 extension seam: `MOD_ACTIVATION` already exists on the runtime mutation shelf, strict genomes already store the node activation key in the canonical `nodeGenes[].squash` field, and materialization restores that state without using the extension bag.
- `src/architecture/node/` now treats response as real runtime node state: `Node.response` defaults to `1`, activation and derivative paths scale by that response, and runtime JSON plus clone restore preserve non-neutral response values.
- `src/architecture/network/serialize/` now preserves both connection gain and node response across raw JSON serialize/restore so ordinary runtime snapshots no longer drop either non-neutral trait.
- `src/neat/genome/` now exposes three additive Step 7.3 traits behind the same typed extension bag: non-neutral ungated connection gain, non-neutral node response keyed by node gene id, and explicit disabled-connection re-enable probability keyed as one genome-level policy value.
- `src/neat/export/` now threads the same opt-in capture flags through controller population export/import and captures strict genomes from the live runtime genome instead of only from serialized runtime JSON, so `_reenableProb` can survive as explicit extension state even when controller-meta fallback is absent.
- `src/neat/compat/` continues to keep canonical compatibility extension-agnostic, and `src/neat/shared/neat.shared.types.ts` plus `src/neat/export/neat.export.types.ts` now document the controller-facing `genomeExtensions` option slice for all landed Step 7.3 traits.

Validation evidence used to close Step 7.3:

- `npm run build`
- targeted tests in `src/architecture/node/node.test.ts`, `src/architecture/network/serialize/network.serialize.test.ts`, `src/neat/genome/genome.test.ts`, and `src/neat/export/neat.export.test.ts` (4 suites / 118 tests)
- broader guard slice in `src/architecture/node/node.test.ts`, `src/architecture/network/serialize/network.serialize.test.ts`, `src/neat/genome/genome.test.ts`, `src/neat/genome/heredity/genome.heredity.test.ts`, `src/neat/export/neat.export.test.ts`, `src/neat/compat/compat.test.ts`, and `src/neat/validate/neat.validate.test.ts` (7 suites / 138 tests)
- activation-audit hardening slice in `src/neat/genome/genome.test.ts`, `src/neat/export/neat.export.test.ts`, and `src/neat/compat/compat.test.ts` (3 suites / 51 tests)
- focused Step 7.3 coverage closure slice in `src/architecture/node/node.test.ts`, `src/architecture/network/serialize/network.serialize.test.ts`, `src/neat/genome/genome.test.ts`, `src/neat/genome/heredity/genome.heredity.test.ts`, `src/neat/export/neat.export.test.ts`, `src/neat/compat/compat.test.ts`, and `src/neat/validate/neat.validate.test.ts` (7 suites / 151 tests) with `network.serialize.json.utils.ts` at 96.38% lines, `neat.export.ts` at 95.12% lines, and `genome.utils.ts` at 90.94% lines across the recent-change owner boundary
- `npm run docs`
- `npm run lint` still reports only unrelated existing blockers in `src/architecture/layer/layer.factory.normalization.utils.test.ts`, `src/architecture/network/network.ts`, `src/neat/species/core/shared/species.core.shared.ts`, and `src/neat/topology-intent/neat.topology-intent.ts`

[DONE] Step 7.4 — Make recurrent modules and gated blocks a deliberate extension lane.

- Keep canonical recurrent and self-edge support from Phases 3 through 5 as the baseline, then layer higher-order recurrent modules or gated blocks as explicit genome constructs instead of hidden runtime conventions.
- Define heredity, mutation, enable/disable, and checkpoint semantics for module-scoped structure before exposing new builders or demo surfaces.
- Preserve a clean distinction between canonical recurrent connections and repo-specific module extensions in both code and docs.
- If these structures begin to overlap with architecture-builder plans, stop and align with `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md` rather than inventing a second builder story here.

Implementation passes:

- [DONE] Pass 7.4a — Explicit temporal-module extension scaffolding: kept builder and mutation ownership frozen, taught `Network.fromJSON()` and `toJSON()` to preserve generic extension bags across runtime round-trips, typed the first explicit Step 7.4 descriptors as `extensions.values.recurrentModules` plus `extensions.values.gatedBlocks`, validated those descriptors against known node gene ids and gated connection innovations, preserved them through strict genome capture plus export/import, and pinned canonical compatibility to keep ignoring the extension bag.
- [DONE] Pass 7.4b — Runtime builder emission and heredity carry-through: taught `Architect.lstm()`, `Architect.gru()`, and `Architect.narx()` to emit explicit temporal descriptors on the runtime extension bag, added one runtime temporal-extension helper that conservatively prunes stale descriptors during serialization, taught direct `ADD_LSTM_NODE` and `ADD_GRU_NODE` runtime mutations to append descriptors for the inserted block, and preserved surviving parent descriptors across runtime crossover when the offspring still materializes the referenced nodes, innovations, and gating assignments.
- [DONE] Pass 7.4c — Runtime structural-edit lifecycle synchronization: taught the shared runtime `connect`, `disconnect`, `gate`, `ungate`, and hidden-node gate-detach seams to synchronize temporal descriptors immediately after structural edits, so generic structural edits and repair flows retire invalid module metadata in memory instead of waiting for later serialization, while still allowing deliberate degradation when an edit drops a gated block but leaves the recurrent-module scaffold intact.
- [DONE] Pass 7.4d — Dormant disabled-gene temporal semantics: defined disabled referenced genes as dormant structure rather than implicit descriptor retirement, taught the shared runtime temporal-extension helper to validate against all registered connections instead of only enabled ones, and added regression coverage across runtime serialization, strict genome materialization, strict validation, and export/import so temporal descriptors survive enable/disable toggles until structural removal or ungating removes the referenced identity.

Completed in the current repo state:

- `src/architecture/network/serialize/` now preserves explicit JSON extension bags on live runtime networks, which gives Phase 7 a checkpoint-safe place to carry additive module metadata without teaching the network layer the feature-specific schema.
- `src/neat/genome/` now exposes the first typed Step 7.4 descriptor surfaces for recurrent modules and gated blocks while keeping canonical node genes and connection genes as the executable source of truth.
- `src/neat/export/` now preserves those explicit module descriptors across population export/import whenever the live runtime genome already carries the extension bag.
- `src/neat/compat/` still keeps canonical compatibility extension-agnostic, which means the Step 7.4 lane remains additive until a later pass deliberately changes the formula.
- `src/architecture/network/network.temporal.extensions.utils.ts` now owns runtime descriptor emission, immediate structural-edit synchronization, conservative pruning, and crossover carry-through for the Step 7.4 lane without teaching the serializer feature-specific schema rules.
- `src/architecture/architect/` now emits deliberate temporal descriptors for `Architect.lstm()`, `Architect.gru()`, and both NARX delay lines while keeping the executable graph on canonical nodes and connection genes.
- `src/architecture/network/mutate/` now appends descriptors for direct `ADD_LSTM_NODE` and `ADD_GRU_NODE` runtime mutations, while later structural edits inherit explicit runtime survival, degradation, and retirement rules from the shared edit seams instead of relying only on serialization-time pruning.
- `src/architecture/network/connect/`, `src/architecture/network/gating/`, and `src/architecture/network/remove/` now resynchronize temporal descriptors after disconnect, ungate, and hidden-node gate-detach flows so generic structural edits and repair-driven rewires update the hydrated extension bag immediately.
- `src/architecture/network/genetic/` now preserves parent temporal descriptors only when the offspring still carries the referenced node gene ids, connection innovations, and gating assignments, which keeps heredity additive and validator-backed.
- Disabled connection genes now count as dormant temporal structure rather than implicit removal, so runtime synchronization, strict genome validation, runtime materialization, and export/import keep recurrent-module and gated-block descriptors alive while the referenced nodes, innovations, and gater ownership still exist.
- Builder overlap is now deliberately runtime-owned rather than frozen: descriptor emission lives on the same preset shelf as the existing recurrent builders, so Step 7.4 still avoids inventing a second architecture story.

Validation evidence used to close Step 7.4:

- targeted guard slice in `src/architecture/network/connect/network.connect.test.ts`, `src/architecture/network/gating/network.gating.test.ts`, `src/architecture/network/remove/network.remove.test.ts`, `src/architecture/network/serialize/network.serialize.test.ts`, `src/architecture/network/mutate/network.mutate.test.ts`, `src/architecture/network/genetic/network.genetic.test.ts`, `src/neat/genome/genome.test.ts`, `src/neat/export/neat.export.test.ts`, `src/neat/compat/compat.test.ts`, and `src/neat/validate/neat.validate.test.ts` (10 suites / 203 tests)
- `npm run build`
- `npm run docs`
- `npm run lint` still reports only the unrelated existing blockers in `src/architecture/layer/layer.factory.normalization.utils.test.ts`, `src/architecture/network/network.ts`, `src/neat/species/core/shared/species.core.shared.ts`, and `src/neat/topology-intent/neat.topology-intent.ts`

Final Step 7.4 closure evidence:

- targeted guard slice in `src/architecture/network/connect/network.connect.test.ts`, `src/architecture/network/gating/network.gating.test.ts`, `src/architecture/network/remove/network.remove.test.ts`, `src/architecture/network/serialize/network.serialize.test.ts`, `src/architecture/network/mutate/network.mutate.test.ts`, `src/architecture/network/genetic/network.genetic.test.ts`, `src/neat/genome/genome.test.ts`, `src/neat/export/neat.export.test.ts`, `src/neat/compat/compat.test.ts`, and `src/neat/validate/neat.validate.test.ts` (10 suites / 207 tests)
- `npm run build`
- `npm run docs`
- `npm run lint` still reports only the unrelated existing blockers in `src/architecture/layer/layer.factory.normalization.utils.test.ts`, `src/architecture/network/network.ts`, `src/neat/species/core/shared/species.core.shared.ts`, and `src/neat/topology-intent/neat.topology-intent.ts`

[DONE] Step 7.5 — Keep novelty, multiobjective, and adaptive policy layers explicitly external.

- Treat novelty search, multiobjective selection, adaptive operator choice, and other research-heavy policies as controller-level extensions that consume the stable genome/runtime contract rather than redefining it.
- Align existing `src/neat/diversity/`, `src/neat/multiobjective/`, `src/neat/objectives/`, `src/neat/selection/`, and `src/neat/adaptive/` boundaries around explicit extension hooks and docs.
- Do not let extension policies silently redefine the default meaning of fitness, compatibility, species history, or replay determinism.
- Prefer separate docs and examples for these lanes so users can opt into them intentionally.

Implementation passes:

- [DONE] Pass 7.5a — Boundary mapping and owner audit: confirmed `src/neat/diversity/`, `src/neat/objectives/`, `src/neat/selection/`, and `src/neat/telemetry/` already read as external controller layers, and narrowed the remaining Step 7.5 overlays to novelty score blending, adaptive acceptance rejection, speciation age protection and fitness sharing, plus controller-owned `_moRank` and `_moCrowd` annotations.
- [DONE] Pass 7.5b — Explicit policy-hook and docs hardening: made extension hooks and README wording explicit across the overlay-heavy novelty/adaptive/speciation/multiobjective chapters and the already-external diversity/objectives/selection/telemetry roots without changing the default meaning of fitness, compatibility, species history, or replay determinism.
- Closed evidence: the regenerated README openings across novelty, adaptive, speciation, multiobjective, diversity, objectives, selection, and telemetry now spell out controller-only or read-only ownership, while export/import keeps score, novelty, and `_moRank` / `_moCrowd` annotations inside explicit `controllerMeta` instead of widening the canonical genome contract. Validation for the docs-hardening slice remains `npm run docs` and `npm run build` passing.

[DONE] Step 7.6 — Validate beyond-paper value on flagship demos and surface it in docs.

- Treat `examples/flappy_bird/` and `examples/asciiMaze/` as the first flagship demo probes for beyond-paper NEAT: analyze which Phase 7 features materially improve each example and which ones would only add novelty without strengthening the demo's concept.
- Evaluate the current and planned beyond-paper lanes against each demo's actual teaching boundary instead of forcing one extension story onto both examples.
- For Flappy Bird, prioritize beyond-paper features that strengthen fast control, temporal decision-making, and browser-inspectable runtime behavior.
- For ASCII Maze, prioritize beyond-paper features that strengthen compact perception, navigation policy quality, curriculum transfer, and telemetry-rich search rather than reflex-heavy control features that do not fit the example's concept.
- Prefer library-level fixes, public contracts, and reusable defaults when either flagship demo exposes a DX or modeling gap; avoid demo-local compensation unless the issue is genuinely example-specific.
- Update the relevant README openings so the beyond-paper gains are easy to find for readers and presentation contexts, especially the owning NEAT chapter plus `examples/flappy_bird/README.md` and `examples/asciiMaze/README.md` when the chosen features land.
- Use the existing Flappy Bird and ASCII Maze documentation workflows rather than treating the docs follow-through as optional; reopen those documentation boundaries only when the examples or their chapter openings materially drift.

Public-doc guardrail for this step:

- keep README openings concept-based and atemporal,
- do not surface plan labels, tracker language, roadmap phases, or repo before/after framing in public docs,
- describe the current teaching boundary, concepts, and tradeoffs instead.

Implementation passes:

- [DONE] Pass 7.6a — ASCII Maze flagship audit and docs-first positioning: tightened `examples/asciiMaze/README.md` into a concept-first, atemporal showcase for compact navigation, deterministic and replayable runs, explicit controller-owned search overlays, curriculum transfer, and optional richer temporal structure.
- [DONE] Pass 7.6b — Flappy Bird boundary confirmation and concept-only docs hardening: confirmed the example should stay feed-forward-first for pedagogic clarity and sharpened the README explanation that temporal observations provide local memory without requiring a second recurrent or gated teaching lane.
- [DONE] Pass 7.6c — Examples-root and chapter drift audit: tightened `examples/README.md` so the two flagship roles read more explicitly through current concepts, and confirmed `src/neat/README.md` was healthy enough for a minimal concept bridge rather than a broader rewrite.
- [DONE] Pass 7.6d — Source contract follow-through: aligned the demo configuration seams with those flagship roles by letting ASCII Maze expose the richer temporal mutation shelf only when recurrent growth is enabled, while Flappy Bird now pins both the trainer and the worker runtime to `allowRecurrent: false` alongside the feed-forward mutation shelf.

Closed evidence so far:

- `examples/asciiMaze/README.md` now opens with a concept-first and atemporal explanation of why compact navigation, deterministic and replayable runs, explicit controller-owned search overlays, and optional richer temporal structure belong together in one maze example.
- `examples/flappy_bird/README.md` now makes the feed-forward teaching goal more explicit by stating why the example prefers temporal observation and local memory over adding a second recurrent or gated control story.
- `examples/README.md` now names the two flagship roles more explicitly, and `src/neat/README.md` now explains how the same controller contract surfaces differently across Flappy Bird and ASCII Maze without changing the canonical-versus-extension boundary.
- `examples/asciiMaze/evolutionEngine/neatConfiguration.ts` now resolves its default mutation shelf from `allowRecurrent`, so the maze demo only widens into gated, self/back-connection, LSTM, and GRU growth when the controller is already allowed to explore richer temporal structure.
- `examples/flappy_bird/trainer/trainer.setup.service.ts` and `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.runtime.service.ts` now set `allowRecurrent: false` explicitly, so the feed-forward teaching boundary is enforced in source as well as in docs.
- Validation for Pass 7.6a: `npm run docs` passed after the ASCII Maze README opening update.
- Validation for Pass 7.6c: `npm run docs` passed after the examples-root and NEAT-root README updates.
- Validation for Pass 7.6d: the targeted Jest slice covering the new ASCII Maze and Flappy Bird configuration tests passed, followed by `npm run build` and `npm run docs`.

Current demo-positioning summary:

- ASCII Maze: the flagship for compact navigation, curriculum transfer, telemetry-rich search, deterministic and replayable runs, explicit controller-owned search overlays, and optional richer temporal structure where the maze benefits from it.
- Flappy Bird: the flagship for deterministic shared-seed evaluation, temporal observation, feed-forward local memory, worker-backed playback, and inspectable browser runtime behavior.

Exit criteria:

- A genome-owned boundary exists without making `Network` obsolete or ambiguous.
- Beyond-paper traits and policies are opt-in, checkpoint-safe, and validator-backed.
- Canonical NEAT docs still teach the Phase 0 through 6 contract first, with explicit extension chapters layered after it.
- The beyond-paper value proposition is easy to find in README openings, and the flagship Flappy Bird plus ASCII Maze paths explain which Phase 7 gains materially improve each demo and why.
- Any extracted operator or extension boundary is small enough to remain teachable and passes the same deterministic guard slice as the canonical path.

This phase may require a focused follow-on split if the `network.genetic` or mutation boundaries become too broad to remain educational.

## Validation Matrix

Every implementation pass under this plan should report the relevant subset of:

- `npm run build`
- `npm run lint`
- targeted tests for the touched boundary
- `npm test` before closing a major phase
- `npm run docs` for any doc-affecting phase

Minimum targeted test surfaces for the proper-NEAT lift:

- [../src/architecture/network/genetic/network.genetic.test.ts](../src/architecture/network/genetic/network.genetic.test.ts)
- [../src/neat/helpers/neat.helpers.test.ts](../src/neat/helpers/neat.helpers.test.ts)
- [../src/neat/mutation/add-conn/mutation.add-conn.test.ts](../src/neat/mutation/add-conn/mutation.add-conn.test.ts)
- [../src/neat/mutation/add-node/mutation.add-node.test.ts](../src/neat/mutation/add-node/mutation.add-node.test.ts)
- [../src/neat/compat/compat.test.ts](../src/neat/compat/compat.test.ts)
- [../src/neat/export/neat.export.test.ts](../src/neat/export/neat.export.test.ts)
- [../src/neat/speciation/speciation.test.ts](../src/neat/speciation/speciation.test.ts)

## Reopen Conditions

- Reopen this tracker only if the user explicitly activates a follow-on Phase 8 performance lane after the closed correctness freeze.
- If Phase 8 is explicitly activated, keep its intended scope limited to cache innovation-sorted views, benchmark compatibility and crossover cost, explore typed-array or SoA gene storage only after correctness and docs stability, and treat population-wide tensorization or GPU acceleration as a later design study rather than a Phase 1 blocker.
- Reopen Step 7.4 only if a new recurrent or gated descriptor requirement appears that is not already covered by the landed builder, heredity, structural-edit, and dormant disabled-gene rules.
- Reopen Step 7.5 only if a concrete controller-policy leak reappears; the closed audit confirmed that the remaining score and `_mo*` round-trip path is explicit controller metadata rather than canonical genome state.
- Reopen Step 7.6 only if flagship-demo docs or source contracts drift away from the current concept-first, atemporal, and source-aligned teaching boundary.

## Deferred Questions

- Should strict canonical mode become the only default, with legacy persistent-map behavior available only behind import or compatibility flags?
- Should `Network.crossOver()` keep its exact public signature, or should the deterministic internal path gain a private RNG-aware helper while the public API stays convenience-oriented?
- If the crossover chapter becomes too broad during the rewrite, should the genotype layer be pulled earlier instead of forcing more logic into runtime-network helpers?
- How should recurrent block mutations such as LSTM/GRU expansion map onto canonical node and connection genes once recurrence becomes first-class in heredity?
- What is the exact migration policy for older checkpoints and serialized networks that lack connection innovations?

## Audit Log

See [neat.logs.md](neat.logs.md) for the durable milestone record covering the canonical lane, the closed Step 7.5 and Step 7.6 milestones, and the final Phase 7 closure notes.
