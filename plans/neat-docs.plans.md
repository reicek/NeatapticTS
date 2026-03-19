# NEAT Root Docs Plan

## Scope

This plan tracks the educational-docs pass for the root NEAT public surface.
The source of truth is the generated documentation fed by [src/neat.ts](../src/neat.ts),
which renders into [src/README.md](../src/README.md) after `npm run docs`.

Primary reader:
- first-time NeatapticTS readers who want to understand how the top-level NEAT controller works before diving into chaptered internals.

Primary surfaces:
- [src/neat.ts](../src/neat.ts)
- [src/README.md](../src/README.md)

Out of scope for this pass:
- hand-editing generated READMEs under `src/`
- adding external images or citations unless a concrete teaching gap requires them
- broad rewrites of every `src/neat/**` chapter README

## Assumptions

- The generated root NEAT story is expected to live in [src/README.md](../src/README.md), not in [src/neat/README.md](../src/neat/README.md).
- The highest-leverage fix is richer JSDoc in [src/neat.ts](../src/neat.ts), especially module-level and class-level introductions.
- External sources and Wikipedia media are optional, not mandatory. If none are needed, the safest compliant outcome is to use no external media and no external citations.

## Session Log

### Session start
- Inspected [src/neat/README.md](../src/neat/README.md), traced generated surface to [src/README.md](../src/README.md), confirmed gap in [src/neat.ts](../src/neat.ts)

### Session end
- Rewrote module intro + Mermaid lifecycle diagram in [src/neat.ts](../src/neat.ts); named class export; expanded `NeatOptions`, telemetry, RNG, persistence JSDoc; regenerated [src/README.md](../src/README.md)

### Follow-up verification pass
- Expanded JSDoc in [src/neat.ts](../src/neat.ts) for lifecycle, archive, telemetry-reset, and persistence helpers

### Telemetry bridge start
- Added [src/neat/telemetry/telemetry.ts](../src/neat/telemetry/telemetry.ts) as root bridge source for top-level telemetry chapter

### Telemetry metrics pass
- Rewrote module intro and added Mermaid diagram in [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts)

### Telemetry recorder pass
- Rewrote chapter opening, added Mermaid diagram, and fixed Example blocks in [src/neat/telemetry/recorder/telemetry.recorder.ts](../src/neat/telemetry/recorder/telemetry.recorder.ts)

### Telemetry facade pass
- Rewrote module intro, added Mermaid diagram, and expanded helper JSDoc in [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts)

### Telemetry accessors pass
- Rewrote module intro and expanded JSDoc for core accessors in [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts)

### Telemetry exports main-file pass
- Expanded module and helper JSDoc in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts)

### Telemetry exports-utils pass
- Rewrote module intro and deepened JSDoc for species-history helpers in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts)

### Telemetry runtime pass
- Expanded JSDoc for buffer, stream, and trim helpers in [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts)

### Telemetry types pass
- Expanded type-level JSDoc in [src/neat/telemetry/types/telemetry.types.ts](../src/neat/telemetry/types/telemetry.types.ts)

### Export persistence pass
- Expanded module intro + helper JSDoc in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) for the three-layer persistence ladder

### Multi-objective ranking pass
- Expanded module intro + `fastNonDominated()` JSDoc in [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts)

### Evolve orchestration pass
- Expanded module JSDoc + Mermaid lifecycle diagram in [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts); fixed source-mapping (ESLint note ordering)

### Evaluate orchestration pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts); expanded constants JSDoc in [src/neat/evaluate/shared/evaluate.constants.ts](../src/neat/evaluate/shared/evaluate.constants.ts)

### Selection chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/selection/selection.ts](../src/neat/selection/selection.ts); fixed source-mapping (JSDoc block position)

### Speciation chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/speciation/speciation.ts](../src/neat/speciation/speciation.ts); deepened helper JSDoc

### Species chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/species/species.ts](../src/neat/species/species.ts); deepened `getSpeciesHistory()` JSDoc

### Objectives chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/objectives/objectives.ts](../src/neat/objectives/objectives.ts); deepened helper JSDoc

### Pruning chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/pruning/pruning.ts](../src/neat/pruning/pruning.ts); deepened helper JSDoc

### Mutation chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/mutation/mutation.ts](../src/neat/mutation/mutation.ts); deepened constants + helper JSDoc

### Adaptive chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/adaptive/adaptive.ts](../src/neat/adaptive/adaptive.ts); deepened helper JSDoc

### Diversity chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/diversity/diversity.ts](../src/neat/diversity/diversity.ts); tightened JSDoc in [src/neat/diversity/core/diversity.types.ts](../src/neat/diversity/core/diversity.types.ts) + [src/neat/diversity/core/diversity.core.ts](../src/neat/diversity/core/diversity.core.ts)

### Lineage chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts); deepened helper JSDoc; expanded contracts in [src/neat/lineage/core/lineage.types.ts](../src/neat/lineage/core/lineage.types.ts)

### Compatibility chapter pass
- Expanded module JSDoc + Mermaid diagram in [src/neat/compat/compat.ts](../src/neat/compat/compat.ts); deepened helper JSDoc

### Compatibility core chapter pass
- Expanded chapter intro + Mermaid diagram in [src/neat/compat/core/compat.types.ts](../src/neat/compat/core/compat.types.ts); deepened helper JSDoc in [src/neat/compat/core/compat.core.ts](../src/neat/compat/core/compat.core.ts); fixed source-mapping duplicate

### RNG chapter pass
- Expanded module intro + Mermaid diagram in [src/neat/rng/rng.ts](../src/neat/rng/rng.ts); expanded JSDoc in [src/neat/rng/core/rng.types.ts](../src/neat/rng/core/rng.types.ts), [src/neat/rng/core/rng.constants.ts](../src/neat/rng/core/rng.constants.ts), [src/neat/rng/core/rng.utils.ts](../src/neat/rng/core/rng.utils.ts)

### Cache chapter pass
- Expanded module intro + Mermaid diagram in [src/neat/cache/cache.ts](../src/neat/cache/cache.ts); expanded JSDoc in [src/neat/cache/core/cache.constants.ts](../src/neat/cache/core/cache.constants.ts) + [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts)

### Compatibility verification pass
- Tightened module and helper JSDoc in [src/neat/compat/compat.ts](../src/neat/compat/compat.ts) for clearer neighborhood-boundary framing

### Cache core chapter pass
- Expanded Mermaid intro in [src/neat/cache/core/cache.constants.ts](../src/neat/cache/core/cache.constants.ts); deepened helper JSDoc in [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts)

### Harness chapter pass
- Expanded chapter intro + Mermaid seam diagram in [src/neat/harness/neat.harness.types.ts](../src/neat/harness/neat.harness.types.ts); deepened type JSDoc for `LineageTrackedNetwork`, `NeatLineageHarness`, `PhasedComplexityHarness`

### Topology-intent chapter pass
- Expanded module, type, and helper JSDoc in [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts) to explain feed-forward intent recognition, eligibility, and promotion flow

### Shared contracts chapter pass
- Expanded chapter and contract JSDoc in [src/neat/shared/neat.shared.types.ts](../src/neat/shared/neat.shared.types.ts) to map shared contract families and cross-chapter usage

### Helpers chapter pass
- Expanded chapter, contract, and helper JSDoc in [src/neat/helpers/neat.helpers.ts](../src/neat/helpers/neat.helpers.ts) to connect pool creation, parent spawning, and genome admission as one bootstrap flow

### Init chapter pass
- Expanded module, contract, and helper JSDoc in [src/neat/init/neat.init.ts](../src/neat/init/neat.init.ts) to explain constructor-time bootstrap order and narrow host/request contracts

### Maintenance facade pass
- Expanded module, contract, and helper JSDoc in [src/neat/maintenance/facade/maintenance.facade.ts](../src/neat/maintenance/facade/maintenance.facade.ts) to explain the maintenance-policy bridge and wrapper semantics

### Pruning facade pass
- Expanded module, contract, and helper JSDoc in [src/neat/pruning/facade/pruning.facade.ts](../src/neat/pruning/facade/pruning.facade.ts) to explain scheduled vs adaptive pruning wrappers

### Selection facade pass
- Expanded module, contract, and helper JSDoc in [src/neat/selection/facade/selection.facade.ts](../src/neat/selection/facade/selection.facade.ts) to explain the stable `Neat` population-summary wrappers

### Selection core chapter pass
- Expanded contract JSDoc in [src/neat/selection/core/selection.types.ts](../src/neat/selection/core/selection.types.ts) and module/helper JSDoc plus Mermaid flow in [src/neat/selection/core/selection.core.ts](../src/neat/selection/core/selection.core.ts)

### Speciation assignment chapter pass
- Expanded module/helper JSDoc and added a Mermaid remap-flow diagram in [src/neat/speciation/assignment/speciation.assignment.utils.ts](../src/neat/speciation/assignment/speciation.assignment.utils.ts)

### Speciation threshold chapter pass
- Expanded module/helper JSDoc and added a Mermaid threshold-control diagram in [src/neat/speciation/threshold/speciation.threshold.utils.ts](../src/neat/speciation/threshold/speciation.threshold.utils.ts)

### Speciation sharing chapter pass
- Expanded module/helper JSDoc and added a Mermaid sharing-and-stagnation flow diagram in [src/neat/speciation/sharing/speciation.sharing.utils.ts](../src/neat/speciation/sharing/speciation.sharing.utils.ts)

### Speciation history chapter pass
- Expanded module/helper JSDoc and added a Mermaid memory-flow diagram in [src/neat/speciation/history/speciation.history.utils.ts](../src/neat/speciation/history/speciation.history.utils.ts)

### Speciation shared chapter pass
- Expanded module, type, and constant JSDoc in [src/neat/speciation/shared/speciation.shared.ts](../src/neat/speciation/shared/speciation.shared.ts) to map shared contexts, summaries, and defaults

### Evaluate fitness chapter pass
- Expanded module/helper JSDoc and added a Mermaid fitness-stage diagram in [src/neat/evaluate/fitness/evaluate.fitness.ts](../src/neat/evaluate/fitness/evaluate.fitness.ts)

### Evaluate entropy-sharing chapter pass
- Expanded module/helper JSDoc and added a Mermaid tuning-flow diagram in [src/neat/evaluate/entropy-sharing/evaluate.entropy-sharing.ts](../src/neat/evaluate/entropy-sharing/evaluate.entropy-sharing.ts)

### Evaluate entropy-compat chapter pass
- Expanded module/helper JSDoc and added a Mermaid threshold-tuning diagram in [src/neat/evaluate/entropy-compat/evaluate.entropy-compat.ts](../src/neat/evaluate/entropy-compat/evaluate.entropy-compat.ts)

### Evaluate auto-distance chapter pass
- Expanded module/helper JSDoc and added a Mermaid moving-baseline diagram in [src/neat/evaluate/auto-distance/evaluate.auto-distance.ts](../src/neat/evaluate/auto-distance/evaluate.auto-distance.ts)

### Evaluate novelty chapter pass
- Expanded module/helper JSDoc and added a Mermaid novelty-flow diagram in [src/neat/evaluate/novelty/evaluate.novelty.ts](../src/neat/evaluate/novelty/evaluate.novelty.ts)

### Evaluate shared chapter pass
- Expanded chapter and symbol JSDoc in [src/neat/evaluate/shared/evaluate.types.ts](../src/neat/evaluate/shared/evaluate.types.ts) and [src/neat/evaluate/shared/evaluate.constants.ts](../src/neat/evaluate/shared/evaluate.constants.ts) to map shared vocabulary and grouped defaults

### Evaluate objectives chapter pass
- Expanded module/helper JSDoc and added a Mermaid auto-objective policy diagram in [src/neat/evaluate/objectives/evaluate.objectives.ts](../src/neat/evaluate/objectives/evaluate.objectives.ts)

### Evaluate speciation chapter pass
- Expanded module/helper JSDoc and added a Mermaid trigger-flow diagram in [src/neat/evaluate/speciation/evaluate.speciation.ts](../src/neat/evaluate/speciation/evaluate.speciation.ts)

### Species history export chapter pass
- Expanded chapter/helper JSDoc and fixed source-mapping order in [src/neat/species/history/species.history.ts](../src/neat/species/history/species.history.ts) so the generated chapter opens with the export-boundary intro

### Objectives core chapter pass
- Expanded chapter and symbol JSDoc in [src/neat/objectives/core/objectives.types.ts](../src/neat/objectives/core/objectives.types.ts) and [src/neat/objectives/core/objectives.core.ts](../src/neat/objectives/core/objectives.core.ts) to explain objective resolution flow

### Multi-objective objectives chapter pass
- Expanded module/helper JSDoc and added a Mermaid vector-assembly diagram in [src/neat/multiobjective/objectives/multiobjective.objectives.ts](../src/neat/multiobjective/objectives/multiobjective.objectives.ts)

### Multi-objective dominance chapter pass
- Expanded module/helper JSDoc, added a Mermaid dominance-flow diagram, and fixed source-mapping order in [src/neat/multiobjective/dominance/multiobjective.dominance.ts](../src/neat/multiobjective/dominance/multiobjective.dominance.ts)

### Multi-objective fronts chapter pass
- Expanded module/helper JSDoc and added a Mermaid frontier-peeling diagram in [src/neat/multiobjective/fronts/multiobjective.fronts.ts](../src/neat/multiobjective/fronts/multiobjective.fronts.ts)

### Multi-objective crowding chapter pass
- Expanded module/helper JSDoc and added a Mermaid crowding-flow diagram in [src/neat/multiobjective/crowding/multiobjective.crowding.ts](../src/neat/multiobjective/crowding/multiobjective.crowding.ts)

### Multi-objective shared chapter pass
- Expanded module and symbol JSDoc in [src/neat/multiobjective/shared/multiobjective.types.ts](../src/neat/multiobjective/shared/multiobjective.types.ts) to map shared contracts, host state, and ordering invariants

### Multi-objective archive chapter pass
- Expanded module, constant, and helper JSDoc and added a Mermaid snapshot-flow diagram in [src/neat/multiobjective/archive/multiobjective.archive.ts](../src/neat/multiobjective/archive/multiobjective.archive.ts)

### Multi-objective category chapter pass
- Expanded module/helper JSDoc, added a Mermaid post-ranking reaction flow, and fixed source-mapping order in [src/neat/multiobjective/category/multiobjective.category.ts](../src/neat/multiobjective/category/multiobjective.category.ts)

### Multi-objective metrics chapter pass
- Expanded module, constant, and helper JSDoc in [src/neat/multiobjective/metrics/multiobjective.metrics.ts](../src/neat/multiobjective/metrics/multiobjective.metrics.ts) to clarify read-side metrics and export helper families

### Evolve objectives chapter pass
- Expanded module/helper JSDoc and added a Mermaid schedule-and-maintenance flow in [src/neat/evolve/objectives/evolve.objectives.utils.ts](../src/neat/evolve/objectives/evolve.objectives.utils.ts)

### Evolve speciation chapter pass
- Expanded module/helper JSDoc and added a Mermaid stage diagram in [src/neat/evolve/speciation/evolve.speciation.utils.ts](../src/neat/evolve/speciation/evolve.speciation.utils.ts); docs HTML still reported the unrelated Mermaid parse failure in [src/neat/lineage/README.md](../src/neat/lineage/README.md)

### Evolve telemetry chapter pass
- Expanded module/helper JSDoc and added a Mermaid timing diagram in [src/neat/evolve/telemetry/evolve.telemetry.utils.ts](../src/neat/evolve/telemetry/evolve.telemetry.utils.ts); docs HTML still reported the unrelated Mermaid parse failure in [src/neat/lineage/README.md](../src/neat/lineage/README.md)

### Evolve runtime chapter pass
- Expanded module/helper JSDoc and added a Mermaid timing-and-bookkeeping diagram in [src/neat/evolve/runtime/evolve.runtime.utils.ts](../src/neat/evolve/runtime/evolve.runtime.utils.ts); docs HTML still reported the unrelated Mermaid parse failure in [src/neat/lineage/README.md](../src/neat/lineage/README.md)

### Evolve warnings chapter pass
- Expanded module and symbol JSDoc in [src/neat/evolve/warnings/evolve.warnings.utils.ts](../src/neat/evolve/warnings/evolve.warnings.utils.ts) to explain the warning boundary and best-effort emission semantics; docs HTML still reported the unrelated Mermaid parse failure in [src/neat/lineage/README.md](../src/neat/lineage/README.md)

### Evolve offspring chapter pass
- Expanded chapter/constant JSDoc in [src/neat/evolve/offspring/evolve.offspring.constants.ts](../src/neat/evolve/offspring/evolve.offspring.constants.ts) and type/helper JSDoc in [src/neat/evolve/offspring/evolve.offspring.utils.ts](../src/neat/evolve/offspring/evolve.offspring.utils.ts) to explain crossover, fallback selection, and lineage bookkeeping; docs HTML still reported the unrelated Mermaid parse failure in [src/neat/lineage/README.md](../src/neat/lineage/README.md)

### Evolve population chapter pass
- Expanded module/helper JSDoc and added a Mermaid next-population diagram in [src/neat/evolve/population/evolve.population.utils.ts](../src/neat/evolve/population/evolve.population.utils.ts); docs HTML still reported the unrelated Mermaid parse failure in [src/neat/lineage/README.md](../src/neat/lineage/README.md)

### Evolve adaptive chapter pass
- Expanded module/helper JSDoc and added a Mermaid sequencing diagram in [src/neat/evolve/adaptive/evolve.adaptive.utils.ts](../src/neat/evolve/adaptive/evolve.adaptive.utils.ts); docs HTML still reported the unrelated Mermaid parse failure in [src/neat/lineage/README.md](../src/neat/lineage/README.md)

### Adaptive complexity chapter pass
- Expanded module/helper JSDoc across [src/neat/adaptive/complexity/adaptive.complexity.ts](../src/neat/adaptive/complexity/adaptive.complexity.ts), [src/neat/adaptive/complexity/adaptive.complexity.utils.ts](../src/neat/adaptive/complexity/adaptive.complexity.utils.ts), and [src/neat/adaptive/complexity/adaptive.phases.utils.ts](../src/neat/adaptive/complexity/adaptive.phases.utils.ts); added a Mermaid schedule-and-phase diagram and retained the existing lineage Mermaid validation note

### Adaptive acceptance chapter pass
- Expanded module/helper JSDoc across [src/neat/adaptive/acceptance/adaptive.acceptance.ts](../src/neat/adaptive/acceptance/adaptive.acceptance.ts) and [src/neat/adaptive/acceptance/adaptive.minimal-criterion.utils.ts](../src/neat/adaptive/acceptance/adaptive.minimal-criterion.utils.ts); added a Mermaid acceptance-flow diagram and retained the existing lineage Mermaid validation note

### Adaptive mutation chapter pass
- Expanded module/helper JSDoc across [src/neat/adaptive/mutation/adaptive.mutation.ts](../src/neat/adaptive/mutation/adaptive.mutation.ts), [src/neat/adaptive/mutation/adaptive.mutation.utils.ts](../src/neat/adaptive/mutation/adaptive.mutation.utils.ts), and [src/neat/adaptive/mutation/adaptive.operator.utils.ts](../src/neat/adaptive/mutation/adaptive.operator.utils.ts); added a Mermaid mutation-pressure diagram and retained the existing lineage Mermaid validation note

### Adaptive lineage chapter pass
- Expanded module/helper JSDoc across [src/neat/adaptive/lineage/adaptive.lineage.ts](../src/neat/adaptive/lineage/adaptive.lineage.ts) and [src/neat/adaptive/lineage/adaptive.ancestor-uniqueness.utils.ts](../src/neat/adaptive/lineage/adaptive.ancestor-uniqueness.utils.ts); added a Mermaid lineage-feedback diagram and retained the existing lineage Mermaid validation note

### Adaptive core chapter pass
- Expanded chapter intro, host contract, aliases, and defaults in [src/neat/adaptive/core/adaptive.core.ts](../src/neat/adaptive/core/adaptive.core.ts), [src/neat/adaptive/core/adaptive.core.types.ts](../src/neat/adaptive/core/adaptive.core.types.ts), and [src/neat/adaptive/core/adaptive.core.constants.ts](../src/neat/adaptive/core/adaptive.core.constants.ts)

### Neat constants chapter pass
- Expanded the module and constant JSDoc in [src/neat/neat.constants.ts](../src/neat/neat.constants.ts) so [src/neat/README.md](../src/neat/README.md) now reads like a grouped shared-constants chapter with a clearer split between epsilon safety values and mutation-policy heuristics; regenerated docs and re-ran TypeScript validation

### Root defaults chapter pass
- Expanded the `DEFAULT_*` constant JSDoc in [src/neat.ts](../src/neat.ts) so [src/README.md](../src/README.md) now reads the root defaults shelf as grouped policy families for search volume, structural caps, compatibility weights, and observability samples instead of a flat inventory; regenerated docs and re-ran TypeScript validation

### Species core chapter pass
- Expanded module/helper JSDoc in [src/neat/species/core/species.core.ts](../src/neat/species/core/species.core.ts) plus companion docs in [src/neat/species/core/augmentation/species.core.augmentation.ts](../src/neat/species/core/augmentation/species.core.augmentation.ts) and [src/neat/species/core/shared/species.core.shared.ts](../src/neat/species/core/shared/species.core.shared.ts) so [src/neat/species/core/README.md](../src/neat/species/core/README.md) now explains extended-history augmentation as a policy gate plus bounded backfill flow; regenerated docs and re-ran TypeScript validation

### Mutation repair chapter pass
- Expanded module/helper JSDoc in [src/neat/mutation/repair/mutation.dead-ends.ts](../src/neat/mutation/repair/mutation.dead-ends.ts) and [src/neat/mutation/repair/mutation.min-hidden.ts](../src/neat/mutation/repair/mutation.min-hidden.ts) so [src/neat/mutation/repair/README.md](../src/neat/mutation/repair/README.md) now frames dead-end repair plus minimum-hidden enforcement as one structural-viability policy chapter; regenerated docs and re-ran TypeScript validation

### Mermaid README cleanup pass
- Simplified the Mermaid node labels in [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts) and [src/neat/objectives/objectives.ts](../src/neat/objectives/objectives.ts) so the generated README diagrams no longer trip the HTML docs renderer; regenerated docs and re-ran TypeScript validation

### Mutation flow chapter pass
- Expanded the module and helper JSDoc in [src/neat/mutation/flow/mutation.flow.ts](../src/neat/mutation/flow/mutation.flow.ts) and [src/neat/mutation/select/mutation.select.ts](../src/neat/mutation/select/mutation.select.ts) so [src/neat/mutation/flow/README.md](../src/neat/mutation/flow/README.md) now reads like a staged per-genome mutation lifecycle and [src/neat/mutation/select/README.md](../src/neat/mutation/select/README.md) now explains policy normalization, phase bias, operator adaptation, bandit choice, and hard guard rails; regenerated docs and re-ran TypeScript validation

### Mutation structural helper chapter pass
- Expanded the module and helper JSDoc in [src/neat/mutation/add-node/mutation.add-node.ts](../src/neat/mutation/add-node/mutation.add-node.ts) and [src/neat/mutation/add-conn/mutation.add-conn.ts](../src/neat/mutation/add-conn/mutation.add-conn.ts) so their generated chapters now read like paired structural-growth chapters covering split-record reuse, new-record assignment, candidate-pair search, cycle guards, and connection-innovation reuse; regenerated docs and re-ran TypeScript validation

### Mutation shared contract chapter pass
- Expanded the module and contract JSDoc in [src/neat/mutation/shared/mutation.types.ts](../src/neat/mutation/shared/mutation.types.ts) so [src/neat/mutation/shared/README.md](../src/neat/mutation/shared/README.md) now reads like a grouped contract map for genome-local state, operator descriptors, innovation records, and the controller host seam; regenerated docs and re-ran TypeScript validation

### Maintenance root bridge pass
- Added [src/neat/maintenance/maintenance.ts](../src/neat/maintenance/maintenance.ts) so [src/neat/maintenance/README.md](../src/neat/maintenance/README.md) now exists as a real root chapter framing maintenance as the conservative policy counterpart to mutation; regenerated docs and re-ran TypeScript validation

### Export chapter refinement pass
- Added a pause-and-resume ladder reading map plus Mermaid checkpoint diagram in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so [src/neat/export/README.md](../src/neat/export/README.md) teaches the population-only, meta-only, and full-state restore paths faster; regenerated docs and re-ran TypeScript validation

### Evolve contract chapter pass
- Expanded the module and contract JSDoc in [src/neat/evolve/evolve.types.ts](../src/neat/evolve/evolve.types.ts) so the contract-heavy sections of [src/neat/evolve/README.md](../src/neat/evolve/README.md) now read like a runtime contract map for genomes, species summaries, policy descriptors, and the evolve host seam; regenerated docs and re-ran TypeScript validation

### Pruning core chapter pass
- Expanded the module, contract, and helper JSDoc in [src/neat/pruning/core/pruning.types.ts](../src/neat/pruning/core/pruning.types.ts) and [src/neat/pruning/core/pruning.core.ts](../src/neat/pruning/core/pruning.core.ts) so [src/neat/pruning/core/README.md](../src/neat/pruning/core/README.md) now reads like the shared policy-and-metric layer beneath scheduled and adaptive pruning instead of a thin type shelf; regenerated docs and re-ran TypeScript validation

### Objectives core chapter pass
- Deepened the contract and helper JSDoc in [src/neat/objectives/core/objectives.types.ts](../src/neat/objectives/core/objectives.types.ts) and [src/neat/objectives/core/objectives.core.ts](../src/neat/objectives/core/objectives.core.ts) so [src/neat/objectives/core/README.md](../src/neat/objectives/core/README.md) now reads more like a staged objective-resolution pipeline and narrow host seam instead of a thin symbol shelf; regenerated docs and re-ran TypeScript validation

### Telemetry facade species chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/species/telemetry.facade.species.ts](../src/neat/telemetry/facade/species/telemetry.facade.species.ts) so [src/neat/telemetry/facade/species/README.md](../src/neat/telemetry/facade/species/README.md) now opens like a species-inspection guide covering live roster reads, historical timeline reads, and spreadsheet-vs-script export paths instead of a thin forwarding shelf; regenerated docs and re-ran TypeScript validation

### Adaptive core contract/constants pass
- Expanded the shared contract and constants JSDoc in [src/neat/adaptive/core/adaptive.core.types.ts](../src/neat/adaptive/core/adaptive.core.types.ts) and [src/neat/adaptive/core/adaptive.core.constants.ts](../src/neat/adaptive/core/adaptive.core.constants.ts) so [src/neat/adaptive/core/README.md](../src/neat/adaptive/core/README.md) now reads more like a shared host-and-policy vocabulary map for adaptive scheduling, acceptance, mutation, and lineage pressure instead of a mostly flat contract/constants shelf; regenerated docs and re-ran TypeScript validation

### Species core shared chapter pass
- Expanded the module, summary-type, and helper JSDoc in [src/neat/species/core/shared/species.core.shared.ts](../src/neat/species/core/shared/species.core.shared.ts) so [src/neat/species/core/shared/README.md](../src/neat/species/core/shared/README.md) now opens like a compact measurement-boundary chapter for species-history augmentation rather than a lone helper shelf; regenerated docs and re-ran TypeScript validation

### Telemetry facade archive chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/archive/telemetry.facade.archive.ts](../src/neat/telemetry/facade/archive/telemetry.facade.archive.ts) so [src/neat/telemetry/facade/archive/README.md](../src/neat/telemetry/facade/archive/README.md) now reads like a Pareto-inspection chapter covering live fronts, compact metrics, archived snapshots, and portable exports instead of a raw helper list; regenerated docs and re-ran TypeScript validation

### Telemetry facade buffer chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/buffer/telemetry.facade.buffer.ts](../src/neat/telemetry/facade/buffer/telemetry.facade.buffer.ts) so [src/neat/telemetry/facade/buffer/README.md](../src/neat/telemetry/facade/buffer/README.md) now reads like a recent-telemetry inspection and export chapter instead of a thin helper shelf; regenerated docs and re-ran TypeScript validation

### Telemetry facade novelty/operator chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/novelty/telemetry.facade.novelty.ts](../src/neat/telemetry/facade/novelty/telemetry.facade.novelty.ts) and [src/neat/telemetry/facade/operator-stats/telemetry.facade.operator-stats.ts](../src/neat/telemetry/facade/operator-stats/telemetry.facade.operator-stats.ts) so their generated chapters now read like compact novelty-memory and operator-diagnostics guides instead of leftover forwarding shelves; regenerated docs and re-ran TypeScript validation

### Telemetry facade runtime chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/runtime/telemetry.facade.runtime.ts](../src/neat/telemetry/facade/runtime/telemetry.facade.runtime.ts) so [src/neat/telemetry/facade/runtime/README.md](../src/neat/telemetry/facade/runtime/README.md) now reads like a controller-health snapshot chapter for diversity and timing instead of a thin helper shelf; regenerated docs and re-ran TypeScript validation

### Telemetry facade lineage chapter pass
- Expanded the module, host-contract, entry-type, and helper JSDoc in [src/neat/telemetry/facade/lineage/telemetry.facade.lineage.ts](../src/neat/telemetry/facade/lineage/telemetry.facade.lineage.ts) so [src/neat/telemetry/facade/lineage/README.md](../src/neat/telemetry/facade/lineage/README.md) now reads like a compact ancestry-inspection chapter instead of a thin snapshot helper shelf; regenerated docs and re-ran TypeScript validation

### Telemetry facade objectives chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/objectives/telemetry.facade.objectives.ts](../src/neat/telemetry/facade/objectives/telemetry.facade.objectives.ts) so [src/neat/telemetry/facade/objectives/README.md](../src/neat/telemetry/facade/objectives/README.md) now reads like an objective-policy and lifecycle chapter instead of a raw helper list; regenerated docs and re-ran TypeScript validation

Remaining gaps:
- some generated parameter and example blocks across the broader NEAT docs still render more compactly than ideal because of the docs generator rather than the source comments
- folder-level coverage across `src/neat/**` is now in place, but some remaining chapters still need qualitative refinement where generated surfaces lean too heavily on symbol shelves instead of grouped stories, especially outside the telemetry-facade cluster now that its main helper subchapters have been strengthened in sequence
- the next highest-leverage qualitative gap still appears to require fresh broader NEAT reconnaissance now that the telemetry-facade helper cluster has been strengthened end-to-end

Next step:
- run a fresh read-only sweep across generated [src/neat/**/README.md](../src/neat/README.md) surfaces outside the recently strengthened telemetry-facade cluster to select the next highest-leverage qualitative refinement target

## Handoff Prompt

```text
Continue the educational-docs pass for the next adjacent NEAT chapter using plans/neat-docs.plans.md as the source of truth.

Current target:
- next highest-leverage generated README gap under `src/neat`, to be selected by fresh reconnaissance outside the recently strengthened telemetry-facade cluster

What to verify next:
- inspect the remaining generated README surfaces under `src/neat` again and rank the next weak chapter now that the telemetry-facade helper cluster has been strengthened end-to-end
- prefer a non-telemetry-facade target unless fresh reconnaissance shows a remaining telemetry surface is still materially weaker than the rest
- current sampled candidates such as [src/neat/evaluate/shared/README.md](../src/neat/evaluate/shared/README.md), [src/neat/multiobjective/shared/README.md](../src/neat/multiobjective/shared/README.md), [src/neat/compat/core/README.md](../src/neat/compat/core/README.md), and [src/neat/selection/core/README.md](../src/neat/selection/core/README.md) now look stronger than the recently fixed telemetry-facade objective/runtime/lineage surfaces, so the next target likely lives in another helper/shared chapter not yet sampled in this sweep
- keep ranking remaining weak generated README surfaces after that pass so the next target still comes from leverage rather than stale adjacency
- preserve the stronger root mutation chapter, repair chapter, maintenance facade, pruning facade, species core, root defaults, and other previously strengthened NEAT chapters unless a new generator issue forces a revisit

If more work is needed:
- keep edits source-first in `src/neat/**` and prefer the smallest chapter-owning source file that feeds the weak generated README surface
- do not hand-edit generated src/**/README.md files
- regenerate docs with `npm run docs`
- run `npx tsc --noEmit -p tsconfig.json` after the doc-affecting source edits
- update plans/neat-docs.plans.md with achievements, remaining gaps, and the next concrete chapter target before ending the session

Current status:
- the telemetry facade species, archive, buffer, novelty, operator-stats, runtime, lineage, and objectives chapters plus the adaptive core and species-core shared chapters are now strengthened; the next likely high-leverage move should now come from a fresh broader NEAT sweep outside that cluster
```
