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
- Expanded the module and constant JSDoc in [src/neat/neat.constants.ts](../src/neat/neat.constants.ts) so [src/neat/README.md](../src/neat/README.md) now reads like a grouped shared-constants chapter with a clearer split between epsilon safety values and mutation-policy heuristics; 

### Root defaults chapter pass
- Expanded the `DEFAULT_*` constant JSDoc in [src/neat.ts](../src/neat.ts) so [src/README.md](../src/README.md) now reads the root defaults shelf as grouped policy families for search volume, structural caps, compatibility weights, and observability samples instead of a flat inventory; 

### Species core chapter pass
- Expanded module/helper JSDoc in [src/neat/species/core/species.core.ts](../src/neat/species/core/species.core.ts) plus companion docs in [src/neat/species/core/augmentation/species.core.augmentation.ts](../src/neat/species/core/augmentation/species.core.augmentation.ts) and [src/neat/species/core/shared/species.core.shared.ts](../src/neat/species/core/shared/species.core.shared.ts) so [src/neat/species/core/README.md](../src/neat/species/core/README.md) now explains extended-history augmentation as a policy gate plus bounded backfill flow; 

### Mutation repair chapter pass
- Expanded module/helper JSDoc in [src/neat/mutation/repair/mutation.dead-ends.ts](../src/neat/mutation/repair/mutation.dead-ends.ts) and [src/neat/mutation/repair/mutation.min-hidden.ts](../src/neat/mutation/repair/mutation.min-hidden.ts) so [src/neat/mutation/repair/README.md](../src/neat/mutation/repair/README.md) now frames dead-end repair plus minimum-hidden enforcement as one structural-viability policy chapter; 

### Mermaid README cleanup pass
- Simplified the Mermaid node labels in [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts) and [src/neat/objectives/objectives.ts](../src/neat/objectives/objectives.ts) so the generated README diagrams no longer trip the HTML docs renderer; 

### Mutation flow chapter pass
- Expanded the module and helper JSDoc in [src/neat/mutation/flow/mutation.flow.ts](../src/neat/mutation/flow/mutation.flow.ts) and [src/neat/mutation/select/mutation.select.ts](../src/neat/mutation/select/mutation.select.ts) so [src/neat/mutation/flow/README.md](../src/neat/mutation/flow/README.md) now reads like a staged per-genome mutation lifecycle and [src/neat/mutation/select/README.md](../src/neat/mutation/select/README.md) now explains policy normalization, phase bias, operator adaptation, bandit choice, and hard guard rails; 

### Mutation structural helper chapter pass
- Expanded the module and helper JSDoc in [src/neat/mutation/add-node/mutation.add-node.ts](../src/neat/mutation/add-node/mutation.add-node.ts) and [src/neat/mutation/add-conn/mutation.add-conn.ts](../src/neat/mutation/add-conn/mutation.add-conn.ts) so their generated chapters now read like paired structural-growth chapters covering split-record reuse, new-record assignment, candidate-pair search, cycle guards, and connection-innovation reuse; 

### Mutation shared contract chapter pass
- Expanded the module and contract JSDoc in [src/neat/mutation/shared/mutation.types.ts](../src/neat/mutation/shared/mutation.types.ts) so [src/neat/mutation/shared/README.md](../src/neat/mutation/shared/README.md) now reads like a grouped contract map for genome-local state, operator descriptors, innovation records, and the controller host seam; 

### Maintenance root bridge pass
- Added [src/neat/maintenance/maintenance.ts](../src/neat/maintenance/maintenance.ts) so [src/neat/maintenance/README.md](../src/neat/maintenance/README.md) now exists as a real root chapter framing maintenance as the conservative policy counterpart to mutation; 

### Export chapter refinement pass
- Added a pause-and-resume ladder reading map plus Mermaid checkpoint diagram in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so [src/neat/export/README.md](../src/neat/export/README.md) teaches the population-only, meta-only, and full-state restore paths faster; 

### Evolve contract chapter pass
- Expanded the module and contract JSDoc in [src/neat/evolve/evolve.types.ts](../src/neat/evolve/evolve.types.ts) so the contract-heavy sections of [src/neat/evolve/README.md](../src/neat/evolve/README.md) now read like a runtime contract map for genomes, species summaries, policy descriptors, and the evolve host seam; 

### Pruning core chapter pass
- Expanded the module, contract, and helper JSDoc in [src/neat/pruning/core/pruning.types.ts](../src/neat/pruning/core/pruning.types.ts) and [src/neat/pruning/core/pruning.core.ts](../src/neat/pruning/core/pruning.core.ts) so [src/neat/pruning/core/README.md](../src/neat/pruning/core/README.md) now reads like the shared policy-and-metric layer beneath scheduled and adaptive pruning instead of a thin type shelf; 

### Objectives core chapter pass
- Deepened the contract and helper JSDoc in [src/neat/objectives/core/objectives.types.ts](../src/neat/objectives/core/objectives.types.ts) and [src/neat/objectives/core/objectives.core.ts](../src/neat/objectives/core/objectives.core.ts) so [src/neat/objectives/core/README.md](../src/neat/objectives/core/README.md) now reads more like a staged objective-resolution pipeline and narrow host seam instead of a thin symbol shelf; 

### Telemetry facade species chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/species/telemetry.facade.species.ts](../src/neat/telemetry/facade/species/telemetry.facade.species.ts) so [src/neat/telemetry/facade/species/README.md](../src/neat/telemetry/facade/species/README.md) now opens like a species-inspection guide covering live roster reads, historical timeline reads, and spreadsheet-vs-script export paths instead of a thin forwarding shelf; 

### Adaptive core contract/constants pass
- Expanded the shared contract and constants JSDoc in [src/neat/adaptive/core/adaptive.core.types.ts](../src/neat/adaptive/core/adaptive.core.types.ts) and [src/neat/adaptive/core/adaptive.core.constants.ts](../src/neat/adaptive/core/adaptive.core.constants.ts) so [src/neat/adaptive/core/README.md](../src/neat/adaptive/core/README.md) now reads more like a shared host-and-policy vocabulary map for adaptive scheduling, acceptance, mutation, and lineage pressure instead of a mostly flat contract/constants shelf; 

### Species core shared chapter pass
- Expanded the module, summary-type, and helper JSDoc in [src/neat/species/core/shared/species.core.shared.ts](../src/neat/species/core/shared/species.core.shared.ts) so [src/neat/species/core/shared/README.md](../src/neat/species/core/shared/README.md) now opens like a compact measurement-boundary chapter for species-history augmentation rather than a lone helper shelf; 

### Telemetry facade archive chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/archive/telemetry.facade.archive.ts](../src/neat/telemetry/facade/archive/telemetry.facade.archive.ts) so [src/neat/telemetry/facade/archive/README.md](../src/neat/telemetry/facade/archive/README.md) now reads like a Pareto-inspection chapter covering live fronts, compact metrics, archived snapshots, and portable exports instead of a raw helper list; 

### Telemetry facade buffer chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/buffer/telemetry.facade.buffer.ts](../src/neat/telemetry/facade/buffer/telemetry.facade.buffer.ts) so [src/neat/telemetry/facade/buffer/README.md](../src/neat/telemetry/facade/buffer/README.md) now reads like a recent-telemetry inspection and export chapter instead of a thin helper shelf; 

### Telemetry facade novelty/operator chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/novelty/telemetry.facade.novelty.ts](../src/neat/telemetry/facade/novelty/telemetry.facade.novelty.ts) and [src/neat/telemetry/facade/operator-stats/telemetry.facade.operator-stats.ts](../src/neat/telemetry/facade/operator-stats/telemetry.facade.operator-stats.ts) so their generated chapters now read like compact novelty-memory and operator-diagnostics guides instead of leftover forwarding shelves; 

### Telemetry facade runtime chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/runtime/telemetry.facade.runtime.ts](../src/neat/telemetry/facade/runtime/telemetry.facade.runtime.ts) so [src/neat/telemetry/facade/runtime/README.md](../src/neat/telemetry/facade/runtime/README.md) now reads like a controller-health snapshot chapter for diversity and timing instead of a thin helper shelf; 

### Telemetry facade lineage chapter pass
- Expanded the module, host-contract, entry-type, and helper JSDoc in [src/neat/telemetry/facade/lineage/telemetry.facade.lineage.ts](../src/neat/telemetry/facade/lineage/telemetry.facade.lineage.ts) so [src/neat/telemetry/facade/lineage/README.md](../src/neat/telemetry/facade/lineage/README.md) now reads like a compact ancestry-inspection chapter instead of a thin snapshot helper shelf; 

### Telemetry facade objectives chapter pass
- Expanded the module, host-contract, and helper JSDoc in [src/neat/telemetry/facade/objectives/telemetry.facade.objectives.ts](../src/neat/telemetry/facade/objectives/telemetry.facade.objectives.ts) so [src/neat/telemetry/facade/objectives/README.md](../src/neat/telemetry/facade/objectives/README.md) now reads like an objective-policy and lifecycle chapter instead of a raw helper list; 

### RNG core chapter pass
- Expanded the chapter-opening, contract, constant, and helper JSDoc in [src/neat/rng/core/rng.types.ts](../src/neat/rng/core/rng.types.ts), [src/neat/rng/core/rng.core.ts](../src/neat/rng/core/rng.core.ts), [src/neat/rng/core/rng.utils.ts](../src/neat/rng/core/rng.utils.ts), and [src/neat/rng/core/rng.constants.ts](../src/neat/rng/core/rng.constants.ts) so [src/neat/rng/core/README.md](../src/neat/rng/core/README.md) now reads like a deterministic replay lifecycle covering ownership, seeding, checkpoint/export, restore, and diagnostic sampling instead of a partially repetitive symbol shelf; 

### Lineage core chapter pass
- Expanded the chapter-opening, contract, and helper JSDoc in [src/neat/lineage/core/lineage.types.ts](../src/neat/lineage/core/lineage.types.ts) and [src/neat/lineage/core/lineage.core.ts](../src/neat/lineage/core/lineage.core.ts) so [src/neat/lineage/core/README.md](../src/neat/lineage/core/README.md) now reads like a bounded ancestry-analysis pipeline covering minimal lineage contracts, breadth-first traversal, sampled pair budgeting, and Jaccard-distance aggregation instead of a thin symbol shelf; 

### Cache core chapter pass
- Expanded the chapter-opening and helper framing in [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts) plus the stale-field ownership framing in [src/neat/cache/core/cache.constants.ts](../src/neat/cache/core/cache.constants.ts) so [src/neat/cache/core/README.md](../src/neat/cache/core/README.md) now reads more like a cache-mechanics chapter covering explicit stale-field ownership, centralized invalidation, and why mutation, crossover, repair, and manual edits all reuse one cleanup contract; 

### Species history context chapter pass
- Expanded the chapter-opening plus resolved-context and resolver JSDoc in [src/neat/species/history/context/species.history.context.ts](../src/neat/species/history/context/species.history.context.ts) so [src/neat/species/history/context/README.md](../src/neat/species/history/context/README.md) now reads like a compact setup chapter covering stored history buffers, optional backfill inputs, and options that gate extended-history augmentation instead of a raw plumbing shelf; 

### Species history read chapter pass
- Expanded the chapter-opening and orchestration framing in [src/neat/species/history/read/species.history.read.ts](../src/neat/species/history/read/species.history.read.ts) so [src/neat/species/history/read/README.md](../src/neat/species/history/read/README.md) now reads like the bounded read pipeline for species-history retrieval rather than a lone function shelf; 

### Species stats chapter pass
- Expanded the chapter-opening, host seam, and projection-helper JSDoc in [src/neat/species/stats/species.stats.ts](../src/neat/species/stats/species.stats.ts) so [src/neat/species/stats/README.md](../src/neat/species/stats/README.md) now reads like a current-roster snapshot chapter for dashboards and logs instead of a thin projection shelf; 

### RNG facade chapter pass
- Expanded the wrapper-semantics and host-contract JSDoc in [src/neat/rng/facade/rng.facade.ts](../src/neat/rng/facade/rng.facade.ts) so [src/neat/rng/facade/README.md](../src/neat/rng/facade/README.md) now teaches the stable replay boundary in terms of snapshot versus export, restore versus import, and sampling for diagnostics instead of a mostly forwarding shelf; 

### Diversity core chapter pass
- Expanded the chapter-opening, contract, and helper JSDoc in [src/neat/diversity/core/diversity.types.ts](../src/neat/diversity/core/diversity.types.ts) and [src/neat/diversity/core/diversity.core.ts](../src/neat/diversity/core/diversity.core.ts) so [src/neat/diversity/core/README.md](../src/neat/diversity/core/README.md) now reads like one bounded diversity-reporting pipeline covering lineage spread, structural size, sampled compatibility distance, and structural entropy instead of opening from a narrow type shelf; 

### Selection core chapter pass
- Expanded the chapter-opening, contract, constant, and helper JSDoc in [src/neat/selection/core/selection.types.ts](../src/neat/selection/core/selection.types.ts) and [src/neat/selection/core/selection.core.ts](../src/neat/selection/core/selection.core.ts) so [src/neat/selection/core/README.md](../src/neat/selection/core/README.md) now reads more like one bounded parent-selection mechanics chapter covering evaluation guards, ordering guards, strategy dispatch, fallback semantics, and strategy-specific choice flow instead of leaning too heavily on a constant-and-helper shelf; 

### Selection facade chapter pass
- Expanded the stable-contract and wrapper JSDoc in [src/neat/selection/facade/selection.facade.ts](../src/neat/selection/facade/selection.facade.ts) and added the named `FittestNetwork` return contract so [src/neat/selection/facade/README.md](../src/neat/selection/facade/README.md) now reads more like a stable public-entry chapter covering why `Neat` preserves summary and ordering reads while root selection and core/ own parent-choice mechanics, with a cleaner `getFittest()` generated surface; 

### Pruning facade chapter pass
- Expanded the stable-contract and wrapper JSDoc in [src/neat/pruning/facade/pruning.facade.ts](../src/neat/pruning/facade/pruning.facade.ts) so [src/neat/pruning/facade/README.md](../src/neat/pruning/facade/README.md) now reads more like a stable public-entry chapter covering why `Neat` preserves both scheduled and adaptive pruning entrypoints, how those entrypoints differ, and why the facade stays narrower than the broader pruning boundary; 

### Adaptive core chapter revisit
- Tightened the chapter-index, contract-map, and constants-glossary framing in [src/neat/adaptive/core/adaptive.core.ts](../src/neat/adaptive/core/adaptive.core.ts), [src/neat/adaptive/core/adaptive.core.types.ts](../src/neat/adaptive/core/adaptive.core.types.ts), and [src/neat/adaptive/core/adaptive.core.constants.ts](../src/neat/adaptive/core/adaptive.core.constants.ts) so [src/neat/adaptive/core/README.md](../src/neat/adaptive/core/README.md) now sustains a clearer shared-vocabulary reading path for adaptive host state, option slices, normalized mutation shapes, and defaults instead of repeating the same framing across the re-export file and the contract map; 

### Speciation shared chapter revisit
- Expanded the chapter-opening, context, summary-type, and default-constant JSDoc in [src/neat/speciation/shared/speciation.shared.ts](../src/neat/speciation/shared/speciation.shared.ts) so [src/neat/speciation/shared/README.md](../src/neat/speciation/shared/README.md) now reads more like one shared-vocabulary chapter mapping live runtime contexts, folded summary types, and grouped speciation defaults instead of falling back into a long shared-context and constant shelf; 

### Speciation threshold chapter revisit
- Expanded the chapter-order, helper-role, and worked-example JSDoc in [src/neat/speciation/threshold/speciation.threshold.utils.ts](../src/neat/speciation/threshold/speciation.threshold.utils.ts) so [src/neat/speciation/threshold/README.md](../src/neat/speciation/threshold/README.md) now reads more like one bounded threshold-controller chapter with a clearer controller-facing entrypoint, control-law core, and final clamp rail instead of slipping back into a thinner helper shelf dominated by the inline `compatAdjust` shape; 

### Speciation sharing chapter revisit
- Expanded the chapter-throughline, mode explanation, and deterministic-survival JSDoc in [src/neat/speciation/sharing/speciation.sharing.utils.ts](../src/neat/speciation/sharing/speciation.sharing.utils.ts) so [src/neat/speciation/sharing/README.md](../src/neat/speciation/sharing/README.md) now reads more like one bounded post-assignment pressure chapter explaining why sharing and stagnation live together, how sigma-aware sharing differs from the uniform fallback, and why sorting before pruning keeps stagnation decisions deterministic; 

### Speciation shared post-generator revisit
- Revisited [src/neat/speciation/shared/speciation.shared.ts](../src/neat/speciation/shared/speciation.shared.ts) after the `docs.order.json` pilot landed, strengthening the grouped family framing for threshold, sharing, age, stagnation/history, and score-sentinel defaults so [src/neat/speciation/shared/README.md](../src/neat/speciation/shared/README.md) reads more like one guided policy glossary instead of a better-ordered constant shelf; 

### Speciation history post-generator revisit
- Revisited [src/neat/speciation/history/speciation.history.utils.ts](../src/neat/speciation/history/speciation.history.utils.ts) with a more book-like teaching goal, strengthening the chapter opening around innovation protection and lineage memory, adding a compact-vs-extended history chart to condense the two snapshot modes, and adding [src/neat/speciation/history/docs.order.json](../src/neat/speciation/history/docs.order.json) so the generated chapter presents protection, recording, extended stats, innovation summaries, numeric fallback, and trimming in a more teachable order; 

### Speciation assignment post-generator revisit
- Revisited [src/neat/speciation/assignment/speciation.assignment.utils.ts](../src/neat/speciation/assignment/speciation.assignment.utils.ts) as the next adjacent speciation chapter, strengthening the boundary framing around novelty protection and niche formation, adding a match-or-create decision chart, and adding [src/neat/speciation/assignment/docs.order.json](../src/neat/speciation/assignment/docs.order.json) so the generated walkthrough follows snapshot, reset, assignment, compatibility decision, new-species creation, and final representative refresh in a more teachable order; 

### Speciation root verification pass
- Re-read [src/neat/speciation/README.md](../src/neat/speciation/README.md) and [src/neat/speciation/speciation.ts](../src/neat/speciation/speciation.ts) after the shared, history, assignment, threshold, and sharing upgrades; kept the source unchanged because the root chapter still reads proportionally strong as the orchestration-first map for the speciation lifecycle, and the existing lifecycle diagram already answers the main reader question without needing a second chart; 
- The more credible next gap has moved outward into the neighboring species-reporting subtree, where the root species map is still serviceable but the history-export surface now looks comparatively thinner than the upgraded speciation cluster; 

### Species history export verification pass
- Re-read [src/neat/species/README.md](../src/neat/species/README.md), [src/neat/species/history/README.md](../src/neat/species/history/README.md), and [src/neat/species/history/species.history.ts](../src/neat/species/history/species.history.ts), confirmed that the next real gap was the narrow history-export chapter rather than a broader species-root revisit, and strengthened the export boundary around the read-vs-export split, bounded recent-window framing, and script/notebook portability; 
- Added a compact export-flow Mermaid diagram plus sharper constant and helper JSDoc in [src/neat/species/history/species.history.ts](../src/neat/species/history/species.history.ts) so [src/neat/species/history/README.md](../src/neat/species/history/README.md) now reads more like a deliberate packaging chapter instead of a small serialization shelf; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 

### Evolve population chapter verification pass
- Compared nearby post-speciation candidates and confirmed that [src/neat/evolve/population/README.md](../src/neat/evolve/population/README.md) was the next more shelf-like chapter because its long helper list still needed a clearer budget-and-branch teaching spine; 
- Strengthened [src/neat/evolve/population/evolve.population.utils.ts](../src/neat/evolve/population/evolve.population.utils.ts) with a second Mermaid chart for the species-aware allocation path plus sharper JSDoc around continuity, provenance, budget branching, allocation fairness, and bounded cross-species mating so [src/neat/evolve/population/README.md](../src/neat/evolve/population/README.md) now reads more like one population-assembly chapter instead of a long helper inventory; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 

### Evaluate speciation verification pass
- Re-read [src/neat/evaluate/speciation/README.md](../src/neat/evaluate/speciation/README.md) and [src/neat/evaluate/speciation/evaluate.speciation.ts](../src/neat/evaluate/speciation/evaluate.speciation.ts), confirmed that the post-evaluation trigger still felt thinner than its nearby evaluation and evolve neighbors, and kept the pass source-first rather than widening into generator work; 
- Strengthened [src/neat/evaluate/speciation/evaluate.speciation.ts](../src/neat/evaluate/speciation/evaluate.speciation.ts) with sharper gate-oriented framing plus a second Mermaid policy chart for target-species tuning, compatibility adjustment, and extended-history refresh so [src/neat/evaluate/speciation/README.md](../src/neat/evaluate/speciation/README.md) now reads more like a deliberate evaluation-to-speciation bridge instead of a thin trigger shelf; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 

### Telemetry facade verification pass
- Moved the weak-surface search outward again, confirmed that the broad [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md) root still felt more shelf-like than chapter-like despite earlier subtree passes, and kept the fix source-first instead of reaching for ordering controls; 
- Strengthened [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) with clearer inspection-intent framing plus sharper JSDoc for the thinner reset, objective-registry, lineage, species-export, novelty, operator, and Pareto export helpers so [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md) now reads more like one inspection guide instead of a broad API shelf; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 

### Telemetry exports verification pass
- Re-read [src/neat/export/README.md](../src/neat/export/README.md) and [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts), confirmed that the persistence chapter now holds up proportionally well after its earlier refinement, and moved the weak-surface search outward instead of reopening a chapter whose pause-and-resume ladder already reads clearly; 
- Re-read [src/neat/telemetry/README.md](../src/neat/telemetry/README.md), [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md), and [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts), confirmed that the telemetry exports boundary was now the thinner serialization chapter, and strengthened it with a clearer three-path export decision map, a Mermaid export-choice diagram, and sharper JSDoc for the species-history fallback constants plus the table-shaping helpers so [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md) reads more like one serialization chapter instead of a long utility shelf; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 

### Maintenance root verification pass
- Re-read [src/neat/maintenance/README.md](../src/neat/maintenance/README.md), [src/neat/maintenance/maintenance.ts](../src/neat/maintenance/maintenance.ts), plus the adjacent [src/neat/maintenance/facade/README.md](../src/neat/maintenance/facade/README.md) and [src/neat/mutation/repair/README.md](../src/neat/mutation/repair/README.md), confirmed that the maintenance root was still the next proportionally thin small bridge after the telemetry-exports pass, and kept the fix source-first instead of widening into generator work or a broader maintenance-facade revisit; 
- Strengthened [src/neat/maintenance/maintenance.ts](../src/neat/maintenance/maintenance.ts) with clearer conservative-repair framing, a three-pass reading map, a second Mermaid decision ladder for read-vs-repair choices, and a sharper explanation of why maintenance exists beside mutation at all so [src/neat/maintenance/README.md](../src/neat/maintenance/README.md) now reads more like a deliberate repair-policy bridge instead of a short root note; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 

### Pedagogical enhancement pause
- Paused the next source-first revisit after confirming the remaining quality ceiling is now mostly generator-shaped rather than comment-shaped: [src/neat/speciation/shared/README.md](../src/neat/speciation/shared/README.md) can still be improved by stronger JSDoc, but the larger limitation is that generated chapter flow is still constrained by global file and symbol ordering heuristics in [scripts/generate-docs.ts](../scripts/generate-docs.ts); 
- Captured the generator-focused follow-up in [plans/pedagogical-docs.plans.md](../plans/pedagogical-docs.plans.md) so the next meaningful step is to add pedagogical ordering controls that can improve NEAT chapters and later help other generated surfaces across `src/` and demo docs as well; 
- Completed phase 1 of that generator plan in [scripts/generate-docs.ts](../scripts/generate-docs.ts): the docs pipeline now recognizes optional per-folder `docs.order.json` files, caches validated config once per folder, and warns clearly on malformed values or unknown keys while preserving previous behavior when config is absent; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TS_EXIT:0`; 
- Completed phase 2 of that generator plan in [scripts/generate-docs.ts](../scripts/generate-docs.ts): folder READMEs now honor configured `fileOrder` entries ahead of the legacy heuristic rank, unspecified files still fall back to the previous ordering behavior, and invalid configured filenames produce scoped warnings instead of silent drift; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TS_EXIT:0`; 
- Completed phase 3 of that generator plan in [scripts/generate-docs.ts](../scripts/generate-docs.ts): file sections now honor configured top-level `symbolOrder` entries ahead of the legacy file-section sort, unspecified symbols still stay on the stable fallback path, nested member ordering remains conservative, and invalid configured symbol names produce scoped warnings instead of silent drift; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 
- Completed phase 4 of that generator plan in [scripts/generate-docs.ts](../scripts/generate-docs.ts): directory intros now honor configured `introFile` choices ahead of the legacy priority heuristic, folder indexes now honor configured `folderOrder` entries ahead of alphabetical fallback, and invalid configured intro or child-folder names produce scoped warnings instead of silent drift; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 
- Completed phase 5 of that generator plan in [scripts/generate-docs.ts](../scripts/generate-docs.ts): folder READMEs can now omit configured `hiddenFiles` and `hiddenSymbols`, hidden entries warn when they drift out of sync with the rendered surface, and the generator skips empty file sections instead of emitting malformed headings; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 
- Started phase 6 rollout in [src/neat/speciation/shared/docs.order.json](../src/neat/speciation/shared/docs.order.json): the weakest shared-speciation chapter now pilots the new symbol-order controls so [src/neat/speciation/shared/README.md](../src/neat/speciation/shared/README.md) groups threshold, sharing, age, history, and fallback defaults in a more teachable order; regenerated docs and re-ran TypeScript validation with `DOCS_EXIT:0` and `TSC_EXIT:0`; 

Remaining gaps:
- some generated parameter and example blocks across the broader NEAT docs still render more compactly than ideal because of the docs generator rather than the source comments
- folder-level coverage across `src/neat/**` is now in place, and the remaining gaps are now mostly qualitative revisits rather than untouched small helper/core shelves after the species-history, species-stats, RNG-facade, diversity-core, selection-core, selection-facade, pruning-facade, and adaptive-core follow-ups
- after the shared-speciation post-generator revisit, [src/neat/speciation/shared/README.md](../src/neat/speciation/shared/README.md) now has a clearer grouped policy story, but adjacent chapters should still be checked for similar "well-ordered yet still too shelf-like" generated sections
- after the history post-generator revisit, [src/neat/speciation/history/README.md](../src/neat/speciation/history/README.md) now uses clearer chapter framing plus a second visual aid for mode comparison, but the next adjacent chapter should still be checked for places where another chart could condense a policy or lifecycle faster than prose alone
- after the assignment post-generator revisit, [src/neat/speciation/assignment/README.md](../src/neat/speciation/assignment/README.md) should now teach the remap boundary more clearly, but the next adjacent sweep should still look for chapters where a second chart could compress a decision rule or lifecycle faster than prose alone
- after the root verification pass, [src/neat/speciation/README.md](../src/neat/speciation/README.md) still holds up as the orchestration-first chapter for the subtree, so the next quality check should move into the neighboring species-reporting surfaces rather than forcing another speciation-root rewrite
- after the species-history export verification pass, [src/neat/species/history/README.md](../src/neat/species/history/README.md) now better separates packaging from history reads, so the broader [src/neat/species/README.md](../src/neat/species/README.md) root still looks serviceable and does not yet justify a second broad rewrite
- after the evolve-population verification pass, [src/neat/evolve/population/README.md](../src/neat/evolve/population/README.md) now carries a clearer budget-and-branch story, so the next sweep should move to another locally thin trigger or facade chapter rather than reopening the same population boundary
- after the evaluate-speciation verification pass, [src/neat/evaluate/speciation/README.md](../src/neat/evaluate/speciation/README.md) now better explains why evaluation can request a species refresh without taking over speciation itself, so the next sweep should keep moving outward to the next small trigger or facade boundary
- after the telemetry-facade verification pass, [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md) now does a better job grouping inspection paths, so the next sweep can move to another smaller persistence or replay boundary instead of reopening the telemetry facade root immediately
- after the telemetry-exports verification pass, [src/neat/export/README.md](../src/neat/export/README.md) now looks proportionally strong enough to leave alone and [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md) now does a better job separating full-fidelity logs, bounded telemetry tables, and species-timeline exports, so the next sweep can move beyond the current persistence cluster to another small bridge or wrapper chapter
- after the maintenance-root verification pass, [src/neat/maintenance/README.md](../src/neat/maintenance/README.md) now better distinguishes conservative repair from exploratory mutation, so the next sweep can move beyond the maintenance root itself to another narrow bridge chapter
- the remaining ceiling is now increasingly structural: folder chapters still depend on generator-owned file and symbol ordering heuristics, which makes it hard to preserve intended teaching order once a chapter grows beyond a single strong intro plus a few symbols
- that structural limitation affects more than the NEAT subtree; the same lack of explicit pedagogical ordering will limit future generated docs for other `src/` folders and demo surfaces unless the generator gains per-folder ordering controls
- some generator-level duplication still appears in re-export-heavy chapters, so future passes should keep favoring stronger chapter openings and sharper per-symbol role framing over generator workarounds unless the generator itself becomes the task

Next step:
- continue the resumed source-first NEAT revisit loop by re-reading [src/neat/topology-intent/README.md](../src/neat/topology-intent/README.md) and [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts) to decide whether the feed-forward promotion bridge is now the next weakest small boundary after the maintenance-root pass

## Handoff Prompt

```text
Continue the educational-docs pass for the next adjacent NEAT chapter using plans/neat-docs.plans.md as the source of truth.

Current state:
- the generator pause is complete enough: [plans/pedagogical-docs.plans.md](../plans/pedagogical-docs.plans.md) now records the optional ordering controls as landed, validated, and proven on the shared-speciation pilot
- [src/neat/speciation/shared/docs.order.json](../src/neat/speciation/shared/docs.order.json) is now part of the live README surface, so the next revisit can focus on chapter quality instead of fighting generator-owned ordering
- the active source of truth is this NEAT plan again, not the generator plan, and the resumed post-generator revisits for shared-speciation, speciation-history, and speciation-assignment have already landed
- the adjacent root speciation surface has now been re-checked and left unchanged because it still reads strongly as the orchestration-first map for the subtree
- the neighboring species-reporting check has now confirmed that the next local gap was the history-export surface itself, and that focused export chapter revisit has already landed without forcing a broader species-root rewrite
- the next outward sweep has now also strengthened [src/neat/evolve/population/README.md](../src/neat/evolve/population/README.md), so the current weak-point search should move to another small trigger or facade boundary rather than back into speciation, species-reporting, or population assembly
- the post-evaluation speciation trigger has now also been strengthened, so the current weak-point search should keep moving outward instead of revisiting speciation-, species-, or evaluate-trigger surfaces immediately
- the broad telemetry facade root has now also been strengthened, so the current weak-point search should move to another smaller persistence or replay boundary instead of reopening telemetry first
- the export persistence chapter now also holds up proportionally well enough to leave unchanged, the telemetry exports chapter has now been strengthened as the thinner neighboring serialization boundary, and the maintenance root bridge has now been strengthened as the next conservative-repair chapter, so the current weak-point search can move beyond the current replay and maintenance cluster

What to verify next:
- re-read [src/neat/topology-intent/README.md](../src/neat/topology-intent/README.md) and [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts) to decide whether the feed-forward promotion bridge is now the next weakest small boundary after the maintenance-root pass
- if the topology-intent bridge still feels proportionally flatter than its nearby init and helpers neighbors, improve the source JSDoc in [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts) and add a chart only if it clarifies the promotion contract faster than the current prose alone
- otherwise keep moving outward to the next smallest bridge, facade, or policy boundary whose generated README still reads more like an API shelf than a chapter
- keep using `docs.order.json` only where compiled chapter order is the real obstacle rather than the prose itself

If more work is needed:
- make source-first JSDoc improvements in the next weakest NEAT chapter, then regenerate docs with `npm run docs` and re-run `npx tsc --noEmit -p tsconfig.json`
- continue to avoid hand-editing generated `src/**/README.md` files
- update [plans/neat-docs.plans.md](../plans/neat-docs.plans.md) with the resumed chapter-local pass and only touch [plans/pedagogical-docs.plans.md](../plans/pedagogical-docs.plans.md) again if a new generator gap appears
- widen docs.order.json rollout to another folder only when the resumed NEAT pass exposes the same structural limit somewhere else

Current status:
- the telemetry facade species, archive, buffer, novelty, operator-stats, runtime, lineage, and objectives chapters plus the adaptive core and species-core shared chapters are now strengthened; fresh broader non-telemetry sweeps then strengthened RNG core, lineage/core, cache/core, species/history/context, species/history/read, species/stats, rng/facade, diversity/core, selection/core, selection/facade, pruning/facade, an adaptive-core revisit, speciation/shared, speciation/threshold, and speciation/sharing
- the generator limitation that triggered the pause has now been addressed, and the resumed speciation follow-up loop has already strengthened shared, history, and assignment while confirming the root speciation chapter still holds up; the follow-on species-reporting check then strengthened the history-export boundary without needing a broader species-root rewrite, the next outward sweeps strengthened evolve/population, the evaluate-time speciation trigger, and the broad telemetry facade root, so the NEAT docs pass can keep moving outward with both stronger source JSDoc and explicit ordering controls available
```
