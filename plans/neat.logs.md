# Proper NEAT (Reference-Quality) — Phase 1 Log

**Status:** [WIP]

## Audit scope

- Objective: turn NeatapticTS into a reference-quality proper-NEAT implementation with canonical historical markings, deterministic replay, strict native compatibility, source-first documentation, and opt-in beyond-paper extensions that do not redefine canonical behavior.
- Active tracker: detailed live planning remains in [neat.plans.md](neat.plans.md).

## Durable milestones

### [DONE] Phases 0 through 1 — Contract and innovation identity baseline

- Wrote explicit historical identity into the runtime, mutation, export, and validation contracts.
- Preserved node gene ids and connection innovations across clone, JSON, compact serialization, import, and counter reconciliation.
- Introduced the explicit innovation-tracker boundary and aligned generation zero so homologous starter genomes share canonical node and connection identity.

### [DONE] Phases 2 through 3 — Canonical heredity and recurrent-policy hardening

- Rewrote crossover around innovation-number matching, gene-id materialization, and controller-owned deterministic RNG.
- Hardened recurrent and self-edge policy across crossover, mutation, repair, and split or add-connection identity rules without reopening legacy index-aligned behavior.

### [DONE] Phases 4 through 6 — Native compatibility, replay, and docs closure

- Tightened native compatibility and species-history expectations around explicit innovations and deliberate fallback-only bridges.
- Upgraded checkpoint and replay contracts so controller and runtime resume state remain deterministic across export or import.
- Closed the canonical lane with source-first documentation and generated README synchronization.

### [DONE] Step 7.1 — First-class genome boundary

- Added the strict genome chapter under `src/neat/genome/` with hard adapters, structural validation, and compatibility plus export adoption.
- Kept `Network` as the runtime phenotype while moving structural identity ownership into the NEAT boundary.
- Confirmed helper, mutation, and innovation allocation ownership stayed deferred for Step 7.2.

### [DONE] Pass 7.2a — Genome-owned heredity selection

- Added `src/neat/genome/heredity/` as the first genome-owned heredity boundary for innovation-aligned connection-gene selection.
- Reduced the runtime crossover selection shelf to a thin materialization adapter without moving setup or phenotype materialization ownership.
- Hardened adjacent guard paths by canonicalizing runtime node order during phenotype-to-genome conversion and refreshing or filtering unusable species state before speciated breeding.

### [DONE] Pass 7.2b — Runtime adapter hardening

- Audited the remaining runtime adapter seam and removed dead parent-local node-index hints from the runtime `ConnectionGene` materialization descriptor.
- Kept the runtime owners unchanged while making `src/architecture/network/genetic/network.genetic.selection.utils.ts` forward only stable heredity identity plus weight and enabled state.
- Added focused regression coverage proving the adapter no longer leaks `from`, `to`, or `gater` index hints and that materialization still resolves endpoints and gaters only by stable gene id.

### [DONE] Pass 7.2c — Deferred-owner confirmation

- Confirmed `src/neat/helpers/`, `src/neat/mutation/`, and `src/neat/export/` remain the explicit deferred-owner chapters after the runtime adapter stopped carrying node-index hints.
- Kept `src/architecture/network/genetic/network.genetic.utils.ts`, `network.genetic.setup.utils.ts`, and `network.genetic.materialize.utils.ts` as the runtime crossover facade, setup, and phenotype materialization owners without pulling those concerns into the deferred-owner shelves.
- Kept `src/neat/validate/neat.validate.ts` as the runtime-only phenotype guard while strict genome-contract validation continues to live behind `src/neat/genome/`.

### [DONE] Pass 7.2d — Closure audit

- Confirmed `src/architecture/network/genetic/network.genetic.utils.ts`, `network.genetic.setup.utils.ts`, and `network.genetic.materialize.utils.ts` still own the runtime crossover facade, setup, and phenotype materialization shelves.
- Confirmed `src/neat/genome/heredity/` remains the only new operator owner from Step 7.2 while `src/neat/helpers/`, `src/neat/mutation/`, `src/neat/export/`, and `src/neat/validate/` stay on their existing deferred-owner or runtime-only shelves.
- Closed Step 7.2 without another operator move because the thinner runtime adapter plus stable gene-id materialization seam already isolate the smallest safe genome-owned heredity slice.

### [DONE] Pass 7.3a — Opt-in connection-gain extension state

- Landed the first Step 7.3 extension trait through `options.genomeExtensions.connectionGain` plus direct genome-adapter capture options, keeping the feature opt-in instead of widening canonical defaults.
- Taught runtime JSON and strict genome adapters to round-trip non-neutral ungated connection gain through `extensions.values.connectionGainByInnovation`, while export/import now preserve the same trait deterministically in extension-aware checkpoints.
- Kept canonical compatibility distance unchanged by explicitly ignoring the new extension bag and hardened strict validation so malformed or gated gain mappings fail fast.

### [DONE] Pass 7.3b — Opt-in node-response and disabled-connection re-enable extension state

- Added runtime `Node.response` with a neutral default of `1`, and taught runtime JSON plus clone restore to preserve non-neutral response values without changing canonical defaults.
- Extended the strict genome extension bag and validator so non-neutral node response and explicit disabled-connection re-enable probability are stored as `nodeResponseByGeneId` and `disabledConnectionReenableProbability`, then restored back onto runtime nodes and genomes during materialization.
- Updated export/import to capture strict genomes from the live runtime genome so runtime-only `_reenableProb` survives checkpoint flows even when controller meta omits the fallback field.

### [DONE] Pass 7.3c — Activation-function audit and hardening

- Audited the remaining Step 7.3 question and confirmed activation mutation is already canonical state through the base node-gene `squash` field, so it does not need a new extension bag field or controller export flag.
- Added focused regressions proving that strict genome conversion and export/import preserve activation mutation without `genomeExtensions`, and that canonical compatibility distance still ignores activation-only deltas.
- Narrowed the remaining Step 7.3 frontier to tracker closure unless a genuinely new additive gene trait is proposed later.

### [DONE] Step 7.3 closure — coverage hardening and tracker closeout

- Added targeted regression coverage for strict-genome compatibility-view caching, assertion-path validation messaging, malformed extension-map validation, and controller restore failure paths in the export chapter.
- Closed the recent-change owner boundary with a focused 7-suite / 151-test coverage slice covering `node`, `network.serialize`, `genome`, `genome.heredity`, `export`, `compat`, and `validate`, measuring `network.serialize.json.utils.ts` at 96.38% lines, `neat.export.ts` at 95.12% lines, and `genome.utils.ts` at 90.94% lines.
- Closed Step 7.3 in the active plan and moved the live frontier to Step 7.4; the low whole-file number in `src/architecture/node/node.ts` remains a legacy-file-wide artifact rather than an uncovered Step 7.3 owner boundary.

### [DONE] Pass 7.4a — Explicit temporal-module extension scaffolding

- Aligned Step 7.4 to [neat.plans.md](neat.plans.md) and the builder-overlap guard in `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md` by keeping `Architect` and mutation ownership frozen in this pass.
- Taught `Network.fromJSON()` and `toJSON()` to preserve generic extension bags so runtime round-trips stop dropping explicit Phase 7 metadata.
- Added the first typed Step 7.4 descriptors in the strict genome extension bag as `recurrentModules` and `gatedBlocks`, validated them against known node gene ids and gated connection innovations, and proved they survive export/import while canonical compatibility still ignores the extension bag.

### [DONE] Pass 7.4b — Runtime builder emission and heredity carry-through

- Taught `Architect.lstm()`, `Architect.gru()`, and `Architect.narx()` to emit explicit temporal descriptors on the runtime extension bag without creating a second execution story beside the canonical node and connection graph.
- Added `src/architecture/network/network.temporal.extensions.utils.ts` as the runtime Step 7.4 helper that appends descriptors for direct `ADD_LSTM_NODE` and `ADD_GRU_NODE` mutations, conservatively prunes stale descriptors during serialization, and preserves only still-valid parent descriptors during runtime crossover.
- Kept canonical compatibility unchanged while extending the active Step 7.4 evidence boundary to runtime serialize, runtime mutate, runtime crossover, genome capture, export, compat, and validate.

### [DONE] Pass 7.4c — Runtime structural-edit lifecycle synchronization

- Moved generic temporal-descriptor lifecycle ownership down to the shared runtime edit primitives by teaching `connect` / `disconnect`, `gate` / `ungate`, and hidden-node gate-detach cleanup to call `synchronizeTemporalDescriptorExtensions(...)` immediately after structural edits.
- Added focused red-green coverage in `src/architecture/network/connect/network.connect.test.ts`, `src/architecture/network/gating/network.gating.test.ts`, and `src/architecture/network/remove/network.remove.test.ts` to prove stale hydrated descriptors retire or degrade in memory instead of waiting for `toJSON()`.
- Locked in the structural degradation rule for the current Step 7.4 lane: ungating drops stale `gatedBlocks` immediately, but a recurrent-module descriptor survives when its node and connection scaffold still validates after the edit.

### [DONE] Pass 7.4d — Dormant disabled-gene temporal semantics

- Closed the last named Step 7.4 gap by defining disabled referenced genes as dormant temporal structure rather than implicit descriptor retirement.
- Clarified the runtime temporal-extension helper and strict genome descriptor contract so enable/disable toggles preserve descriptor identity while the referenced node ids, connection innovations, and gater ownership still exist.
- Added focused regressions across runtime serialization, strict genome materialization, strict validation, and export/import to prove temporal descriptors survive disabled-gene state until structural removal or ungating removes the referenced identity.

## Controls and evidence

- Each landed milestone used the relevant subset of `npm run build`, targeted guard suites, and `npm run docs`.
- Latest Step 7.3 evidence: `npm run build`, the focused `node` / `network.serialize` / `genome` / `export` slice (4 suites / 118 tests), the broader `node` / `network.serialize` / `genome` / `genome.heredity` / `export` / `compat` / `validate` guard slice (7 suites / 138 tests), and `npm run docs`.
- Activation-audit hardening evidence: focused `genome` / `export` / `compat` slice (3 suites / 51 tests).
- Step 7.3 closure evidence: focused `node` / `network.serialize` / `genome` / `genome.heredity` / `export` / `compat` / `validate` coverage slice (7 suites / 151 tests) with `network.serialize.json.utils.ts` at 96.38% lines, `neat.export.ts` at 95.12% lines, and `genome.utils.ts` at 90.94% lines.
- Step 7.4a evidence: targeted `network.serialize` / `genome` / `export` / `compat` / `validate` slice (5 suites / 135 tests), plus `npm run build` and `npm run docs`.
- Step 7.4b evidence: targeted `network.serialize` / `network.mutate` / `network.genetic` / `genome` / `export` / `compat` / `validate` slice (7 suites / 187 tests), plus `npm run build` and `npm run docs`; `npm run lint` stayed at the same four unrelated existing blockers.
- Step 7.4c evidence: targeted `network.connect` / `network.gating` / `network.remove` / `network.serialize` / `network.mutate` / `network.genetic` / `genome` / `export` / `compat` / `validate` slice (10 suites / 203 tests), plus `npm run build` and `npm run docs`; `npm run lint` stayed at the same four unrelated existing blockers.
- Step 7.4d evidence: targeted `network.connect` / `network.gating` / `network.remove` / `network.serialize` / `network.mutate` / `network.genetic` / `genome` / `export` / `compat` / `validate` slice (10 suites / 207 tests), plus `npm run build` and `npm run docs`; `npm run lint` stayed at the same four unrelated existing blockers.
- Known unrelated blockers remain outside this workstream: lint issues in `src/architecture/layer/layer.factory.normalization.utils.test.ts`, `src/architecture/network/network.ts`, `src/neat/species/core/shared/species.core.shared.ts`, and `src/neat/topology-intent/neat.topology-intent.ts`, plus the separate add-node compile-time test issue.

## Open control notes

- The active implementation frontier now sits at Step 7.5 in [neat.plans.md](neat.plans.md), while closed Steps 7.3 and 7.4 remain the current beyond-paper baseline until a future controller-policy proposal proves it needs a new external seam.
- The current beyond-paper frontier is no longer "should temporal descriptors survive disabled-gene toggles?" That rule is now landed: disabled referenced genes remain dormant structure until structural removal or ungating removes the referenced identity. The next live question is how novelty, multiobjective, adaptive, and related controller policy layers stay explicitly external to the canonical genome and runtime contract.
- Step 7.2 closure guardrails remain in force: deferred owners remain `src/neat/helpers/`, `src/neat/mutation/`, `src/neat/export/`, and the current runtime-only validation lane until a narrower extension seam is justified.