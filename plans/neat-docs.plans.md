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

Goals:
- turn the root NEAT docs into a real chapter introduction with a strong opening narrative
- add at least one diagram to orient readers around the lifecycle
- improve the public `NeatOptions` and `Neat` explanations so the generated docs teach workflow, not just signatures
- regenerate docs and verify the generated output

Progress:
- inspected [src/neat/README.md](../src/neat/README.md) and confirmed it is not the root NEAT chapter surface for this task
- traced the actual generated NEAT surface to [src/README.md](../src/README.md)
- confirmed the main gap is in [src/neat.ts](../src/neat.ts): the generated chapter currently has a bare module heading and a `default` class heading with little orientation

### Session end

Achievements:
- rewrote the root module introduction in [src/neat.ts](../src/neat.ts) so the generated docs explain the NEAT lifecycle, the reason the controller stays orchestration-first, and where to read next
- added a Mermaid lifecycle diagram in [src/neat.ts](../src/neat.ts) to orient readers before they hit the API surface
- promoted the public class docs from an anonymous generated `default` section to a named `Neat` section by switching to a named class with a default export in [src/neat.ts](../src/neat.ts)
- expanded the public teaching surface for `NeatOptions`, constructor semantics, evolution, evaluation, telemetry, objectives, RNG state, and persistence helpers in [src/neat.ts](../src/neat.ts)
- regenerated [src/README.md](../src/README.md) with `npm run docs` and verified that the opening now presents a long NEAT-oriented chapter introduction, the Mermaid lifecycle diagram renders cleanly, and the public class heading now renders as `Neat` instead of `default`

Decision:
- the root NEAT docs now have a strong chapter introduction and lifecycle diagram ready for readers

### Follow-up verification pass

Goals:
- verify that the generated root `src/README.md` opening still reads correctly as a library-level chapter intro after the earlier generator changes
- decide whether the next highest-leverage docs pass should stay in [src/neat.ts](../src/neat.ts) or move into one adjacent NEAT chapter
- improve any remaining root public helpers that still read like terse API stubs

Progress:
- re-read the generated opening in [src/README.md](../src/README.md) and confirmed it still works as the root `src` chapter because it frames `src/neat.ts` as the NEAT controller map rather than pretending to describe the whole repository
- compared the root surface with neighboring chapter outputs, including [src/neat/README.md](../src/neat/README.md), [src/neat/multiobjective/README.md](../src/neat/multiobjective/README.md), [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md), and [src/neat/export/README.md](../src/neat/export/README.md)
- confirmed the biggest remaining teaching gap was still in the root class surface: several public lifecycle, archive, telemetry-reset, and persistence helpers in [src/neat.ts](../src/neat.ts) were materially shorter than the adjacent chapter docs they front
- expanded those root JSDoc blocks in [src/neat.ts](../src/neat.ts) so the generated root chapter better explains when to use direct mutation, pruning hooks, parent and offspring helpers, telemetry clearing and export, archive inspection, diversity and performance snapshots, and population-versus-full-state persistence

Decision:
- the regenerated root helper sections read like a guided API tour rather than terse stubs
- the next adjacent chapter to deepen should be telemetry, because the telemetry subtree is already well-factored but lacks one obvious top-level `src/neat/telemetry/README.md` bridge page

### Telemetry bridge start

Goals:
- trace which existing telemetry source comments should feed a stronger top-level telemetry chapter surface
- verify whether the missing bridge is a wording problem or a source-mapping problem
- create the smallest source-first fix that can generate a real top-level telemetry chapter

Progress:
- inspected the telemetry folder structure and confirmed the root gap is structural: [src/neat/telemetry](../src/neat/telemetry) had chaptered subfolders but no root TypeScript file and therefore no obvious source to generate a top-level telemetry README
- traced the strongest existing teaching comments to the root files that already own the telemetry story: [src/neat/telemetry/recorder/telemetry.recorder.ts](../src/neat/telemetry/recorder/telemetry.recorder.ts), [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts), [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts), [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts), [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts), and [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts)
- confirmed the bridge narrative should synthesize two complementary paths already present in source comments: the write path (`recorder` -> `metrics` -> `runtime`) and the read path (`facade` -> `accessors` -> `exports`)
- added [src/neat/telemetry/telemetry.ts](../src/neat/telemetry/telemetry.ts) as the new root source bridge so generated docs have a top-level telemetry chapter to compile from

Decision:
- the missing telemetry bridge was primarily a source-mapping gap, not just weak prose in existing child chapters
- the new top-level telemetry surface is strong enough to shift the next work item from "make the chapter exist" to "decide which child chapter deserves the next educational deepening pass"
- from a pedagogical perspective, the thinnest telemetry child chapter is [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts): it carries the highest concept density with the least reader guidance about grouping and reading order

### Telemetry metrics pass

Goals:
- turn the metrics chapter from a flat helper index into a teaching surface that explains why the metric families exist
- preserve the existing source layout while improving the generated [src/neat/telemetry/metrics/README.md](../src/neat/telemetry/metrics/README.md)
- verify that the improved chapter still reads clearly after docs regeneration

Progress:
- re-read the metrics source and confirmed the main gap was concentrated in the root chapter intro at [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts), not in a single missing helper description
- rewrote that root module introduction so it now teaches the six metric families: diversity and entropy, lineage, objectives and Pareto signals, complexity, RNG state, and performance timing
- added a reading order and a Mermaid diagram so the generated metrics chapter explains how those evidence layers feed one telemetry entry rather than only listing exports

Decision:
- the metrics chapter is no longer the thinnest telemetry teaching surface; the new opening now explains both metric-family purpose and reading order well enough for a first pass
- the next telemetry refinement should likely be narrower and example-driven rather than another broad chapter-opening rewrite

### Telemetry recorder pass

Goals:
- turn the recorder chapter into a true write-side walkthrough for one generation snapshot
- remove malformed example rendering from the generated recorder README
- preserve the existing source structure while making the compiled chapter easier to teach from

Progress:
- re-read [src/neat/telemetry/recorder/telemetry.recorder.ts](../src/neat/telemetry/recorder/telemetry.recorder.ts) and confirmed the core gap was at the chapter-introduction and public-helper JSDoc level, not in runtime behavior
- rewrote the recorder chapter opening so it now teaches the full write path from controller state to built entry, selection, buffering, streaming, and trimming
- added a Mermaid lifecycle diagram and a practical reading order so the generated chapter explains how the recorder relates to `metrics/`, `runtime/`, and `facade/`
- cleaned up the malformed `Example:` blocks around `applyTelemetrySelect`, `structuralEntropy`, `computeDiversityStats`, `recordTelemetryEntry`, and `buildTelemetryEntry` so the generated README now renders those examples cleanly

Decision:
- the recorder chapter is now strong enough as the write-side companion to the metrics chapter
- if telemetry docs continue, the next highest-value pass should likely shift to the public inspection side rather than another broad recorder rewrite

### Telemetry facade pass

Goals:
- turn the facade chapter into a reader-facing map of post-run inspection workflows instead of a flat export list
- verify once more that the generated root [src/README.md](../src/README.md) opening still fits the broader `src` surface before fully leaving the root chapter behind
- decide which telemetry child chapter should follow the facade pass

Progress:
- re-skimmed the generated opening in [src/README.md](../src/README.md) and confirmed it still fits the broader root `src` surface: it reads as the NEAT controller chapter map, not as an overclaim about every file in the repository
- rewrote the module-level introduction in [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) so the generated facade chapter now teaches the main inspection workflows: recent telemetry windows, objective and Pareto inspection, species and lineage reads, diversity and performance checks, and reset helpers
- added a Mermaid workflow diagram in [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) so the compiled chapter shows how a caller moves from `Neat` runtime state to specific read models and export surfaces
- expanded several facade helper JSDoc blocks in [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) so the generated [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md) better explains when to use telemetry windows, CSV export, objective summaries, species summaries, species history, Pareto inspection, diversity reads, and coarse timing checks

Decision:
- the facade chapter is no longer the weakest user-facing telemetry surface; it now functions as the public inspection companion to the write-side recorder and metrics chapters
- the next highest-leverage telemetry docs pass should move into [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) to explain why those low-level reads exist and how they support both internal telemetry code and the public facade

### Telemetry accessors pass

Goals:
- turn the accessors chapter into a readable explanation of the telemetry read model beneath the public facade
- keep the chapter small and source-first without promoting it into another user-facing facade layer
- verify that the generated accessors README now teaches the boundary instead of only listing tiny helpers

Progress:
- rewrote the module introduction in [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) so the generated chapter now explains the audience split between `facade/` and `accessors/`, why these helpers stay intentionally narrow, and how other telemetry chapters reuse them
- expanded the JSDoc for [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) exports so `getTelemetryBuffer`, `clearTelemetryBuffer`, `getObjectiveEventsSnapshot`, `buildLineageSnapshot`, `getCachedDiversityStats`, `getPerformanceStatsSnapshot`, `TelemetryAccessorHost`, and `LINEAGE_SNAPSHOT_DEFAULT_LIMIT` now explain intent, invariants, and usage context instead of reading like terse utility stubs
- added compact examples to the most instructive helpers in [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) so the generated chapter shows how these low-level reads are actually consumed during inspection and testing

Decision:
- the accessors chapter is no longer the weakest telemetry teaching surface; it now reads like the internal read-model companion to the facade chapter
- the next telemetry docs gap should move into the exports helpers to explain header discovery and deterministic export shaping

### Telemetry exports main-file pass

Goals:
- deepen the helper-level teaching surface in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) without reopening the already-strong chapter introduction
- explain how header discovery, sparse-field flattening, and deterministic CSV shaping work together
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module and helper JSDoc in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) so the generated exports chapter now explains the main design constraint of the exporter: preserving a stable, comparable export shape across uneven telemetry windows
- improved the teaching surface for `TelemetryHeaderInfo`, `exportTelemetryJSONL`, `exportTelemetryCSV`, `exportSpeciesHistoryCSV`, `collectTelemetryHeaderInfo`, `buildTelemetryHeaders`, `serializeTelemetryEntry`, and `buildSpeciesHistoryCsv` in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts)
- added small examples for the CSV exporters in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) so the generated chapter shows how callers inspect headers and short windows in practice

Decision:
- the main exports file no longer feels like the weakest telemetry chapter surface
- the next narrow telemetry docs target should be [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts) to deepen the bookkeeping teaching surface

### Telemetry exports-utils pass

Goals:
- turn the exports utility section into a readable explanation of the bookkeeping behind deterministic telemetry CSV shaping
- make the generated exports-utils section explain header-state mutation, sparse-column enabling, and species-history normalization more clearly
- keep the pass source-first and avoid runtime behavior changes

Progress:
- rewrote the module and helper JSDoc in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts) so the generated exports-utils section now explains that this file is the preparatory layer behind stable CSV exports rather than a random grab-bag of helpers
- deepened the generated teaching surface for `TelemetryHeaderCollectionState`, `collectBaseKeys`, `collectGroupedMetricKeys`, `collectDiversityLineageMetrics`, `collectOptionalColumnPresence`, `ensureSpeciesHistoryArray`, `ensureMinimalSpeciesSnapshot`, `collectSpeciesHistoryHeaders`, `buildSpeciesHistoryStats`, `serializeSpeciesHistoryRow`, `resolveSpeciesHistoryCellValue`, and `safeStringifyCell` in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts)
- added compact examples to the most instructive species-history normalization helpers in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts) so the generated section shows how deterministic backfilling and normalization are actually used

Decision:
- the exports area is now documented well enough for a first telemetry pass; both the public entrypoints and the lower-level bookkeeping layer read coherently
- the next telemetry docs gap should move into the runtime chapter to explain the safety layer beneath the recorder

### Telemetry runtime pass

Goals:
- turn the runtime chapter into a clearer explanation of the telemetry safety layer beneath the recorder
- improve the helper-level teaching surface for lazy buffer creation, callback isolation, and bounded retention
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module and helper JSDoc in [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts) so the generated runtime chapter now explains the containment boundary between the recorder and the live evolution loop
- improved the generated teaching surface for `ensureTelemetryBuffer`, `safelyStreamTelemetryEntry`, and `trimTelemetryBuffer` in [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts) so the README now explains why telemetry stays opt-in, why callback failures are swallowed, and why bounded retention is a runtime contract

Decision:
- the runtime chapter is now strong enough for the telemetry pass
- the remaining telemetry-local gap shifted from behavior chapters to the small contracts chapter under `types/`

### Telemetry types pass

Goals:
- turn the telemetry types chapter from a list of names into a readable map of the internal telemetry contracts
- clarify why these contracts stay telemetry-local instead of being promoted into broader NEAT public types
- close out the first telemetry-focused educational-docs sweep with a coherent internal types chapter

Progress:
- expanded the module and type-level JSDoc in [src/neat/telemetry/types/telemetry.types.ts](../src/neat/telemetry/types/telemetry.types.ts) so the generated types chapter now explains why these contracts exist and how they support recorder, runtime, exports, and facade helpers
- improved the generated teaching surface for `TelemetryGenome`, `TelemetryDiversityOptions`, `TelemetryStreamOptions`, `TelemetryEntryRecord`, `OperatorStatsMap`, `TelemetryBufferContext`, `TelemetrySelectContext`, and `TelemetryCoreFields` in [src/neat/telemetry/types/telemetry.types.ts](../src/neat/telemetry/types/telemetry.types.ts)

Decision:
- the telemetry subtree now has a coherent first-pass educational surface across its bridge, recorder, metrics, runtime, facade, accessors, exports, and local types chapters
- the next highest-leverage docs move should leave telemetry and shift to one adjacent NEAT chapter rather than continuing to over-polish the same subtree
- the most sensible next chapter is [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts), because persistence is adjacent to telemetry in user workflow

### Export persistence pass

Goals:
- turn the export chapter from a persistence helper inventory into a clearer pause-and-resume workflow chapter
- improve the source-first JSDoc around population-only snapshots, controller-meta snapshots, and full-state restore flows
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module introduction in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so the generated export chapter now teaches the three-layer persistence ladder: population-only snapshots, meta-only snapshots, and full-state checkpoints
- improved the generated teaching surface for `GenomeJSON`, `InnovationMapEntry`, `GenomeWithSerialization`, `NeatControllerForExport`, `NetworkClass`, `NeatConstructor`, and `NeatStateJSON` in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so the supporting contracts now explain why they exist in the persistence boundary
- deepened the public helper docs for `importPopulation`, `exportState`, `importStateImpl`, `toJSONImpl`, and `fromJSONImpl` in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so the generated [src/neat/export/README.md](../src/neat/export/README.md) now reads more like a guided pause-and-resume chapter than a flat serialization inventory

Decision:
- the export chapter is now strong enough for a first educational-docs pass
- the next highest-leverage move should shift to another adjacent orchestration chapter rather than continuing to over-polish persistence contracts
- [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts) is the clearest next target because it is central to advanced NEAT behavior

### Multi-objective ranking pass

Goals:
- turn the multi-objective root chapter from a concise algorithm summary into a clearer guided ranking workflow
- improve the source-first JSDoc around fronts, crowding, ranking stages, and archive intent
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module introduction in [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts) so the generated chapter now explains why multi-objective ranking exists, how Pareto fronts differ from single-score selection, and how crowding preserves spread along the frontier
- deepened the `fastNonDominated()` teaching surface in [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts) so the generated [src/neat/multiobjective/README.md](../src/neat/multiobjective/README.md) now explains the four-stage ranking flow, how to read returned fronts, why crowding matters, and why frontier archiving happens at this orchestration layer

Decision:
- the multi-objective root chapter is now strong enough for a first pass and no longer feels like the most obvious adjacent gap after export
- the next highest-leverage move should shift to the evolve orchestration chapter, which remains central to the root NEAT story but still reads thinner than the upgraded telemetry, export, and multi-objective chapters

### Evolve orchestration pass

Goals:
- turn the evolve chapter from a thin method summary into a real generation-update chapter
- improve the source-first JSDoc around evaluation readiness, adaptive hooks, ranking/speciation, offspring construction, mutation/pruning, telemetry, and post-generation maintenance
- verify that the generated [src/neat/evolve/README.md](../src/neat/evolve/README.md) now opens with a usable chapter map instead of dropping readers directly into symbols

Progress:
- confirmed the main gap in the generated [src/neat/evolve/README.md](../src/neat/evolve/README.md) was both structural and local: the chapter lacked a real file-level opening, and the public `evolve()` description in [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts) still compressed the generation lifecycle into one dense block
- expanded the module-level JSDoc in [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts) so the source now teaches the five-phase generation map, explains why the file stays orchestration-first, adds a Mermaid lifecycle diagram, and gives readers a concrete reading order into `adaptive/`, `runtime/`, `speciation/`, `population/`, and `telemetry/`
- deepened the public `evolve()` JSDoc in [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts) so the generated chapter now explains the seven execution stages, the return-value semantics, the important side effects, and the surrounding helper folders as the next reading map
- improved the educational surface of the exported evolve constants in [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts) so the generated README explains why the main threshold, allocation, pruning, stagnation, re-enable, and compatibility defaults exist instead of listing them as bare names
- found and fixed a source-mapping issue while validating: the file-level chapter intro initially failed to render because the docs generator was taking the first JSDoc block in the file, so moving the evolve module introduction above the non-JSDoc ESLint note in [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts) restored the intended generated opening

Decision:
- the evolve chapter now reads like a guided generation-update chapter rather than a constants-and-signature dump
- the strongest remaining adjacent gap has shifted upstream to [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts), because evaluation is the phase that feeds evolve directly but its generated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) still reads more like a compact pipeline summary

### Evaluate orchestration pass

Goals:
- turn the evaluate chapter from a compact pipeline summary into a guided scoring-and-adaptation chapter
- improve the source-first JSDoc around per-genome versus population-level fitness execution, novelty blending, diversity-stat readiness, adaptive tuning hooks, post-evaluation speciation maintenance, and automatic entropy-objective registration
- verify that the generated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) now opens with a real chapter map and that its constants read like tuning controls instead of a bare inventory

Progress:
- confirmed the main gap in the generated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) was all three at once: the chapter opening was too compact, the public `evaluate()` description in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts) still read like a short checklist, and the exported constants were only lightly described
- expanded the module-level JSDoc in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts) so the source now teaches why evaluation exists as a separate orchestration chapter, explains the six-stage scoring lifecycle, adds a Mermaid flow diagram, gives a concrete reading order into the helper folders, and frames the exported constants as four tuning families
- deepened the public `evaluate()` JSDoc in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts) so the generated chapter now explains the evidence-first mental model, the split between score production and lightweight controller maintenance, the important side effects, and the relationship between `evaluate()` and the later `evolve()` pass
- improved the educational surface of the exported evaluation constants in [src/neat/evaluate/shared/evaluate.constants.ts](../src/neat/evaluate/shared/evaluate.constants.ts) so the generated README now explains why the novelty, entropy-sharing, compatibility, and distance-coefficient defaults exist instead of listing them as terse labels

Decision:
- the evaluate chapter now reads like a guided scoring-and-adaptation chapter rather than a compact pipeline summary plus constant list
- the strongest remaining adjacent gap has shifted to [src/neat/selection/selection.ts](../src/neat/selection/selection.ts), because the generated [src/neat/selection/README.md](../src/neat/selection/README.md) still reads more like a utility index than a chaptered explanation of sorting and parent-selection strategy

### Selection chapter pass

Goals:
- turn the selection chapter from a compact utility index into a guided controller-facing selection chapter
- improve the source-first JSDoc around sorting, average and fittest summaries, parent-selection strategy flow, fallback score semantics, and the `core/` versus `facade/` split
- verify that the generated [src/neat/selection/README.md](../src/neat/selection/README.md) now opens with a real chapter map instead of dropping directly into symbols

Progress:
- confirmed the main gap in the generated [src/neat/selection/README.md](../src/neat/selection/README.md) was mostly chapter framing plus thin method-level explanation: the earlier opening was brief, and `sort()`, `getParent()`, `getFittest()`, and `getAverage()` still read more like terse helper stubs than a guided controller story
- expanded the root module JSDoc in [src/neat/selection/selection.ts](../src/neat/selection/selection.ts) so the source now teaches the four-step selection mental model, explains why the root chapter stays controller-facing while `core/` owns strategy math and `facade/` owns stable `Neat` wrappers, frames the re-exported constants as tuning and traversal anchors, adds a Mermaid flow diagram, and gives a compact example covering sort, summary, and parent reads
- deepened the public helper JSDoc in [src/neat/selection/selection.ts](../src/neat/selection/selection.ts) so the generated chapter now explains when to sort explicitly, how `getParent()` maps onto POWER, FITNESS_PROPORTIONATE, and TOURNAMENT selection, why `getFittest()` safely handles unevaluated or unsorted populations, and when `getAverage()` is the better coarse generation signal
- found and fixed a source-mapping issue while validating: moving the module intro above the re-export block caused the generated chapter opening to disappear, so relocating the same JSDoc back below the re-export block in [src/neat/selection/selection.ts](../src/neat/selection/selection.ts) restored the intended README opening

Decision:
- the selection chapter now reads like a guided selection-and-ordering chapter rather than a compact utility index
- the strongest remaining adjacent gap has shifted to [src/neat/speciation/speciation.ts](../src/neat/speciation/speciation.ts), because the generated [src/neat/speciation/README.md](../src/neat/speciation/README.md) already has a decent chapter opening but its helper-level sections still read compact and assumption-heavy

### Speciation chapter pass

Goals:
- turn the speciation chapter from a compact lifecycle summary into a guided species-assignment-and-maintenance chapter
- improve the source-first JSDoc around the controller-facing speciation flow: reassignment, threshold adjustment, history capture, optional fitness sharing, member sorting, and stagnation maintenance
- verify that the generated [src/neat/speciation/README.md](../src/neat/speciation/README.md) now balances its strong chapter opening with equally clear helper-level guidance

Progress:
- confirmed the main gap in the generated [src/neat/speciation/README.md](../src/neat/speciation/README.md) was mostly helper-level explanation rather than missing top-level framing: the opening already outlined the lifecycle, but `_speciate()`, `_applyFitnessSharing()`, `_sortSpeciesMembers()`, and `_updateSpeciesStagnation()` still read like compact internal helpers instead of a guided controller story
- expanded the root module JSDoc in [src/neat/speciation/speciation.ts](../src/neat/speciation/speciation.ts) so the source now explains why speciation exists as the diversity-preserving controller phase, frames the three practical speciation questions, maps the ownership split across `assignment/`, `threshold/`, `history/`, and `sharing/`, adds a Mermaid lifecycle diagram, and shows a compact end-to-end example covering speciation, sharing, and stagnation maintenance
- deepened the public helper JSDoc in [src/neat/speciation/speciation.ts](../src/neat/speciation/speciation.ts) so the generated chapter now explains that `_speciate()` maintains a long-lived species registry rather than merely clustering genomes, `_applyFitnessSharing()` is a deliberate post-assignment normalization pass, `_sortSpeciesMembers()` is the shared deterministic ranking rule for species-local reads, and `_updateSpeciesStagnation()` is the maintenance pass that decides whether species are still earning their place

Decision:
- the speciation chapter now reads like a guided species-assignment-and-maintenance chapter rather than a compact lifecycle summary with thin helper sections
- the strongest remaining adjacent gap has shifted to [src/neat/species/species.ts](../src/neat/species/species.ts), because the generated [src/neat/species/README.md](../src/neat/species/README.md) already has a serviceable opening and a decent `getSpeciesStats()` surface, but `getSpeciesHistory()` still reads much more like a thin API stub

### Species chapter pass

Goals:
- turn the species chapter from a compact read-helper index into a guided species-reporting chapter
- improve the source-first JSDoc around the difference between live species snapshots and recorded cross-generation history
- verify that the generated [src/neat/species/README.md](../src/neat/species/README.md) now explains when to use `getSpeciesStats()` versus `getSpeciesHistory()`

Progress:
- confirmed the main gap in the generated [src/neat/species/README.md](../src/neat/species/README.md) was not the existence of the chapter opening but the thinness of the reporting distinction: `getSpeciesStats()` already had decent guidance, while `getSpeciesHistory()` still read like a terse API stub without much help on when or why to use it
- expanded the root module JSDoc in [src/neat/species/species.ts](../src/neat/species/species.ts) so the source now frames the chapter as the read-side companion to speciation, explains the split between current snapshots and cross-generation history, maps the ownership split across `stats/`, `core/`, and `history/`, adds a Mermaid reporting-flow diagram, and includes a compact example using both species reads
- deepened the `getSpeciesHistory()` JSDoc in [src/neat/species/species.ts](../src/neat/species/species.ts) so the generated chapter now explains when history is the right tool, why it is heavier but more explanatory than the live snapshot path, and how the history read flow can backfill extended fields when the controller configuration allows it

Decision:
- the species chapter now reads like a guided species-reporting chapter rather than a compact read-helper index
- the strongest remaining adjacent gap shifted to [src/neat/objectives/objectives.ts](../src/neat/objectives/objectives.ts), because the generated [src/neat/objectives/README.md](../src/neat/objectives/README.md) still reads more like a compact method index than a guided chapter about how the controller decides what "better" means

### Objectives chapter pass

Goals:
- turn the objectives chapter from a compact method index into a guided objective-management chapter
- improve the source-first JSDoc around default-versus-user objective composition, lazy objective resolution, registration-by-key, and cache invalidation
- verify that the generated [src/neat/objectives/README.md](../src/neat/objectives/README.md) now reads like a public controller contract for deciding what "better" means

Progress:
- confirmed the main gap in the generated [src/neat/objectives/README.md](../src/neat/objectives/README.md) was both chapter framing and helper-level explanation: the earlier root opening was brief, and `_getObjectives()`, `registerObjective()`, and `clearObjectives()` still read more like terse method descriptions than a guided management surface
- expanded the root module JSDoc in [src/neat/objectives/objectives.ts](../src/neat/objectives/objectives.ts) so the source now explains why objective management exists, frames the three public responsibilities of the root chapter, maps the ownership split into `core/`, adds a Mermaid flow diagram for defaults, user objectives, and resolution, and provides a compact example covering registration, resolution, and clearing
- deepened the public helper JSDoc in [src/neat/objectives/objectives.ts](../src/neat/objectives/objectives.ts) so the generated chapter now explains that `_getObjectives()` is the lazy resolution step for the final ordered objective list, `registerObjective()` replaces objectives by key and invalidates the cached resolved list, and `clearObjectives()` resets the user-defined layer so future reads rebuild from defaults

Decision:
- the objectives chapter now reads like a guided objective-management chapter rather than a compact method index
- the strongest remaining adjacent gap has shifted to [src/neat/pruning/pruning.ts](../src/neat/pruning/pruning.ts), because the generated [src/neat/pruning/README.md](../src/neat/pruning/README.md) has a decent root setup but its two entrypoints still read more like short method summaries

### Pruning chapter pass

Goals:
- turn the pruning chapter from a decent setup plus thin method summaries into a guided pruning-control chapter
- improve the source-first JSDoc around the difference between scheduled evolution pruning and adaptive pruning, the `core/` versus `facade/` split, and the controller state each path updates
- verify that the generated [src/neat/pruning/README.md](../src/neat/pruning/README.md) now reads like a controller chapter instead of a compact pair of entrypoints

Progress:
- confirmed the main gap in the generated [src/neat/pruning/README.md](../src/neat/pruning/README.md) was not a missing chapter surface so much as a missing contrast: the opening was serviceable, but `applyEvolutionPruning()` and `applyAdaptivePruning()` still read too much like short summaries and did not clearly explain when each path should run or what controller state they affect
- expanded the root module JSDoc in [src/neat/pruning/pruning.ts](../src/neat/pruning/pruning.ts) so the source now frames pruning as the controller's deliberate structure-removal chapter, contrasts scheduled generation-timed pruning with adaptive metric-driven pruning, maps the ownership split across `core/` and `facade/`, and adds a Mermaid flowchart showing where each path enters and what it updates
- deepened the public helper JSDoc in [src/neat/pruning/pruning.ts](../src/neat/pruning/pruning.ts) so the generated chapter now explains that `applyEvolutionPruning()` reads generation state and schedule settings without mutating adaptive fields, while `applyAdaptivePruning()` maintains baseline and shared prune-level controller state before reapplying pruning across the population when drift exceeds tolerance

Decision:
- the pruning chapter now reads like a guided pruning-control chapter rather than a compact pair of method summaries
- the strongest remaining adjacent gap has shifted to [src/neat/mutation/mutation.ts](../src/neat/mutation/mutation.ts), because the generated [src/neat/mutation/README.md](../src/neat/mutation/README.md) still mixes a decent root setup with inventory-shaped constant sections and relatively compact helper explanations for the controller's main structure-editing chapter

### Mutation chapter pass

Goals:
- turn the mutation chapter from a mixed helper-and-constant inventory into a guided mutation-control chapter
- improve the source-first JSDoc around the whole-population mutation flow, the helper-folder ownership split, the exported default constants, and the controller or innovation-tracking state touched by the main entrypoints
- verify that the generated [src/neat/mutation/README.md](../src/neat/mutation/README.md) now reads like a controller-facing structure-editing chapter instead of a mostly inventory-shaped surface

Progress:
- confirmed the main gap in the generated [src/neat/mutation/README.md](../src/neat/mutation/README.md) was mostly framing plus helper-level explanation: the root opening was serviceable, but the constants and several public entrypoints still read more like utility inventory than a guided story about how the controller edits topology across a whole population
- expanded the root module JSDoc in [src/neat/mutation/mutation.ts](../src/neat/mutation/mutation.ts) so the source now frames mutation as the controller's deliberate structure-editing chapter, explains the `flow/`, `select/`, `add-node/`, `add-conn/`, and `repair/` ownership split, and adds a Mermaid flowchart showing how selection, structural edits, repair, and bookkeeping fit together
- deepened the public and constant JSDoc in [src/neat/mutation/mutation.ts](../src/neat/mutation/mutation.ts) so the generated chapter now explains what the default constants are for, why `mutate()` is a whole-population consistency pass, how `mutateAddNodeReuse()` and `mutateAddConnReuse()` preserve innovation history, why repair helpers exist after structural edits, and how `selectMutationMethod()` acts as the policy gateway before any operator runs

Decision:
- the mutation chapter now reads like a guided mutation-control chapter rather than a mixed helper and constant inventory
- the strongest remaining adjacent gap shifted to [src/neat/adaptive/adaptive.ts](../src/neat/adaptive/adaptive.ts), because the generated [src/neat/adaptive/README.md](../src/neat/adaptive/README.md) still had a comparatively thin root controller story despite spanning several important long-lived policy loops

### Adaptive chapter pass

Goals:
- turn the adaptive chapter from a compact method catalog into a guided adaptive-control chapter
- improve the source-first JSDoc around when each adaptive controller should run, which long-lived controller state it maintains, and how the `complexity/`, `mutation/`, `acceptance/`, `lineage/`, and `core/` folders split ownership
- verify that the generated [src/neat/adaptive/README.md](../src/neat/adaptive/README.md) now reads like the controller map for in-flight policy adjustment rather than a short list of helpers

Progress:
- confirmed the main gap in the generated [src/neat/adaptive/README.md](../src/neat/adaptive/README.md) was the thinness of the root story: the chapter listed several important adaptive controllers, but it did not clearly explain why they belong together, what signals each one consumes, or whether they rewrite controller options, per-genome state, or current-population acceptance
- expanded the root module JSDoc in [src/neat/adaptive/adaptive.ts](../src/neat/adaptive/adaptive.ts) so the source now frames adaptive control as the controller's in-flight policy-adjustment chapter, maps the practical reader questions, explains the helper-folder ownership split, and adds a Mermaid flowchart showing how scores, telemetry, generation cadence, and operator stats feed forward into later controller passes
- deepened the public helper JSDoc in [src/neat/adaptive/adaptive.ts](../src/neat/adaptive/adaptive.ts) so the generated chapter now explains that `applyComplexityBudget()` and `applyPhasedComplexity()` rewrite controller policy, `applyMinimalCriterionAdaptive()` rewrites current-population acceptance, `applyAncestorUniqAdaptive()` feeds telemetry back into future lineage or multi-objective settings, `applyAdaptiveMutation()` rewrites per-genome mutation readiness, and `applyOperatorAdaptation()` decays operator history for later selection

Decision:
- the adaptive chapter now reads like a guided adaptive-control chapter rather than a compact method catalog
- the strongest remaining adjacent gap has shifted to [src/neat/diversity/diversity.ts](../src/neat/diversity/diversity.ts), because the generated [src/neat/diversity/README.md](../src/neat/diversity/README.md) still stays comparatively compact and utility-shaped even though it feeds telemetry, diagnostics, and population-level reasoning

### Diversity chapter pass

Goals:
- turn the diversity chapter from a compact read-model summary into a guided diversity-reporting chapter
- improve the source-first JSDoc around the controller-facing diversity read flow: how sampled lineage, compatibility, and structural-entropy signals fit together, why the root chapter stays compact while `core/` owns the heavier aggregation mechanics, and how telemetry or diagnostics consumers should interpret the returned summary
- verify that the generated [src/neat/diversity/README.md](../src/neat/diversity/README.md) now opens with a real chapter map and gives the exported summary type plus sampling constants a clearer teaching role

Progress:
- confirmed the main gap in the generated [src/neat/diversity/README.md](../src/neat/diversity/README.md) was mostly chapter framing plus helper-level explanation: `structuralEntropy()` was already reasonably clear, but the root opening, `computeDiversityStats()`, and the exported summary-contract and sampling-limit sections still read too much like a compact utility surface
- expanded the root module JSDoc in [src/neat/diversity/diversity.ts](../src/neat/diversity/diversity.ts) so the source now frames diversity as the controller's answer to "how varied is this population, and in what sense?", explains why the summary is sampled, gives a reading order into `structuralEntropy()`, `computeDiversityStats()`, and `core/`, and adds a Mermaid flowchart showing how lineage, structure, compatibility, and entropy feed one compact report for telemetry and diagnostics
- deepened the public helper JSDoc in [src/neat/diversity/diversity.ts](../src/neat/diversity/diversity.ts) so the generated chapter now explains `computeDiversityStats()` as a four-signal bounded trend report, clarifies how to interpret `structuralEntropy()` beside raw node and connection counts, and adds a compact example for the controller-facing population read
- tightened the supporting source comments in [src/neat/diversity/core/diversity.types.ts](../src/neat/diversity/core/diversity.types.ts) and [src/neat/diversity/core/diversity.core.ts](../src/neat/diversity/core/diversity.core.ts) so the generated root README now gives `DiversityStats`, `MAX_COMPATIBILITY_SAMPLE`, and `MAX_LINEAGE_PAIR_SAMPLE` a clearer teaching role instead of leaving them as near-stub sections

Decision:
- the diversity chapter now reads like a guided diversity-reporting chapter rather than a compact read-model summary
- the strongest remaining adjacent gap has shifted to [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts), because the generated [src/neat/lineage/README.md](../src/neat/lineage/README.md) still reads more like a terse ancestry-helper listing

### Lineage chapter pass

Goals:
- turn the lineage chapter from a compact ancestry-helper index into a guided lineage-evidence chapter
- improve the source-first JSDoc around the controller-facing lineage read flow: what a shallow ancestor map represents, how sampled ancestor uniqueness complements diversity and telemetry summaries, why the root chapter stays compact while `core/` owns the heavier traversal mechanics, and how readers should interpret low versus high uniqueness signals
- verify that the generated [src/neat/lineage/README.md](../src/neat/lineage/README.md) now opens with a real chapter map and gives the exported host and genome contracts clearer teaching value

Progress:
- confirmed the main gap in the generated [src/neat/lineage/README.md](../src/neat/lineage/README.md) was mostly chapter framing plus helper-level explanation: the root opening was compact, `buildAnc()` and `computeAncestorUniqueness()` still read like terse helper stubs, and the exported `GenomeLike` and `NeatLineageContext` sections did not yet explain why the lineage boundary stays so small
- expanded the root module JSDoc in [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts) so the source now frames lineage as the controller's answer to whether genomes still come from meaningfully different recent families, gives a reading order into `buildAnc()`, `computeAncestorUniqueness()`, and `core/`, and adds a Mermaid flowchart showing how shallow ancestor sets become a population-level uniqueness signal for telemetry, diagnostics, and adaptive policy
- deepened the public helper JSDoc in [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts) so the generated chapter now explains what "shallow" ancestry means in practice, when to use `buildAnc()` as raw ancestry evidence, how to interpret lower versus higher sampled ancestor uniqueness, and why that lineage read complements diversity instead of duplicating it
- tightened the supporting source comments in [src/neat/lineage/core/lineage.types.ts](../src/neat/lineage/core/lineage.types.ts) so the generated root README now gives `GenomeLike` and `NeatLineageContext` a clearer contract role instead of leaving them as near-reference stubs

Decision:
- the lineage chapter now reads like a guided lineage-evidence chapter rather than a compact ancestry-helper index
- the strongest remaining adjacent gap has shifted to [src/neat/compat/compat.ts](../src/neat/compat/compat.ts), because the generated [src/neat/compat/README.md](../src/neat/compat/README.md) still reads more like a thin bridge plus helper summaries

### Compatibility chapter pass

Goals:
- turn the compatibility chapter from a thin helper bridge into a guided compatibility-distance chapter
- improve the source-first JSDoc around what compatibility distance helps the controller decide, how fallback innovation ids should be interpreted, and how this chapter connects to speciation, diversity, and lineage
- verify that the generated [src/neat/compat/README.md](../src/neat/compat/README.md) now opens with a real controller-facing map instead of only exposing helper summaries

Progress:
- confirmed the main gap in the generated [src/neat/compat/README.md](../src/neat/compat/README.md) was both chapter framing and interpretation: the existing opening was too compact, `_compatibilityDistance()` still read mostly like an orchestration stub, and `_fallbackInnov()` did not yet explain clearly why synthetic innovation ids exist or how readers should think about them beside canonical innovation tracking
- expanded the root module JSDoc in [src/neat/compat/compat.ts](../src/neat/compat/compat.ts) so the source now frames compatibility as the controller's answer to whether two genomes are still close enough to share an evolutionary neighborhood, explains why the root chapter stays compact while `core/` owns comparison mechanics, adds a Mermaid flowchart linking compatibility to speciation, diversity, and diagnostics, and gives a concrete reading order into the nearby stronger chapters
- deepened the public helper JSDoc in [src/neat/compat/compat.ts](../src/neat/compat/compat.ts) so the generated chapter now explains `_compatibilityDistance()` as the decision-support read for species boundaries and population spread, and `_fallbackInnov()` as the deterministic bridge for legacy or partially normalized genomes when explicit innovation ids are unavailable

Decision:
- the compatibility root chapter now reads like a guided compatibility-distance chapter rather than a thin helper bridge
- the strongest remaining gap has shifted from the root bridge into the lower-level mechanics chapter in [src/neat/compat/core/compat.core.ts](../src/neat/compat/core/compat.core.ts), whose generated helper sections are still comparatively terse beside the stronger root-orchestration surfaces

### Compatibility core chapter pass

Goals:
- turn the compatibility core chapter from a compact helper inventory into a guided comparison-mechanics chapter
- improve the source-first JSDoc around the minimal contracts, generation-scoped cache lifecycle, sorted innovation-list normalization, merge comparison semantics, and coefficient fold behind NEAT compatibility distance
- verify that the generated [src/neat/compat/core/README.md](../src/neat/compat/core/README.md) now opens with a real mechanics map instead of a thin type stub

Progress:
- confirmed the main gap in the generated [src/neat/compat/core/README.md](../src/neat/compat/core/README.md) was both chapter framing and helper-level explanation: the opening was effectively just the old `ConnectionLike` type summary, and the mechanics helpers still read more like a terse utility index than a clear walk through cache invalidation, innovation alignment, merge classification, and coefficient folding
- expanded the root chapter intro in [src/neat/compat/core/compat.types.ts](../src/neat/compat/core/compat.types.ts) so the generated core README now explains the minimal contracts boundary, adds a Mermaid flowchart for contract -> cache -> sorted-list -> metrics -> distance, and gives a concrete reading order into the core mechanics
- deepened the exported type docs in [src/neat/compat/core/compat.types.ts](../src/neat/compat/core/compat.types.ts) so `ConnectionLike`, `GenomeLike`, `NeatLikeForCompat`, and `ComparisonMetrics` now explain why the core layer stays small and how each contract supports the later comparison and distance fold
- expanded the mechanics JSDoc in [src/neat/compat/core/compat.core.ts](../src/neat/compat/core/compat.core.ts) so the generated chapter now explains cache invalidation by generation, order-independent pair keys, normalized innovation-list caching, merge-walk matching versus disjoint versus excess semantics, and the final weighted NEAT distance fold
- found and fixed a source-mapping issue while validating: the new core chapter intro initially duplicated under `ConnectionLike`, so adding a small separator comment before the first exported type in [src/neat/compat/core/compat.types.ts](../src/neat/compat/core/compat.types.ts) preserved the chapter opening without reusing it as the symbol-level doc block

Decision:
- the compatibility area now has a coherent two-layer educational surface: the root chapter explains what compatibility distance helps the controller decide, and the core chapter now explains how the evidence is normalized, classified, cached, and folded into that distance
- the strongest remaining adjacent gap has shifted to [src/neat/rng/rng.ts](../src/neat/rng/rng.ts), because the generated [src/neat/rng/README.md](../src/neat/rng/README.md) still reads more like a compact helper and constant inventory than a guided deterministic-replay chapter

### RNG chapter pass

Goals:
- turn the RNG chapter from a compact helper and constant inventory into a guided deterministic-replay chapter
- improve the source-first JSDoc around the controller-facing randomness flow: stream creation, seed precedence, state snapshot and restore, exported constant roles, and the boundary between the root bridge and the `core/` plus `facade/` layers
- verify that the generated [src/neat/rng/README.md](../src/neat/rng/README.md) now opens with a real replay-oriented chapter map instead of a short bridge plus symbol list

Progress:
- confirmed the main gap in the generated [src/neat/rng/README.md](../src/neat/rng/README.md) was both chapter framing and symbol-level explanation: the root intro was only a thin bridge, while the re-exported helpers and constants still read more like an API inventory than a guided explanation of deterministic replay
- expanded the root chapter intro in [src/neat/rng/rng.ts](../src/neat/rng/rng.ts) so the generated README now explains why reproducible randomness matters to NEAT, frames the reader questions around stream origin, state capture and restore, and contract layers, adds a Mermaid replay-flow diagram, and gives a concrete reading order into the helper surfaces
- deepened the exported contract docs in [src/neat/rng/core/rng.types.ts](../src/neat/rng/core/rng.types.ts) so `RngHost` now explains the intentionally small replay boundary: cached stream, numeric state, population context, and option overrides
- improved the constant docs in [src/neat/rng/core/rng.constants.ts](../src/neat/rng/core/rng.constants.ts) so the generated root chapter now explains which fixed values govern seed guarding, xorshift mixing, and float normalization rather than leaving them as mostly opaque numbers
- expanded the helper docs in [src/neat/rng/core/rng.utils.ts](../src/neat/rng/core/rng.utils.ts) so the generated root chapter now explains seed-precedence order in `getOrCreateRng()`, replay semantics for snapshot and restore helpers, persistence use for `exportRngState()`, compatibility intent for `importRngState()`, and diagnostics-oriented sampling with `sampleRandomSequence()`

Decision:
- the RNG area now reads like a guided deterministic-replay chapter rather than a compact helper and constant inventory
- the strongest remaining adjacent gap has shifted to [src/neat/cache/cache.ts](../src/neat/cache/cache.ts), because the generated [src/neat/cache/README.md](../src/neat/cache/README.md) still reads more like a thin invalidation helper surface than a guided explanation of why per-genome caches exist and when invalidation matters

### Cache chapter pass

Goals:
- turn the cache chapter from a thin helper surface into a guided cache-invalidation chapter
- improve the source-first JSDoc around why genome-owned derived caches exist, when they become stale after mutation or repair, and why centralized invalidation is safer than scattered cleanup logic
- verify that the generated [src/neat/cache/README.md](../src/neat/cache/README.md) now opens with a real controller-facing invalidation map instead of a short bridge plus symbol list

Progress:
- confirmed the main gap in the generated [src/neat/cache/README.md](../src/neat/cache/README.md) was mostly chapter framing: the earlier root file was only a bridge, and the generated story was being carried almost entirely by the constant and helper summaries without explaining why the invalidation boundary matters to the controller
- expanded the root chapter intro in [src/neat/cache/cache.ts](../src/neat/cache/cache.ts) so the generated README now explains stale derived state, frames the reader questions around invalidation timing and centralized cleanup, adds a Mermaid flowchart for edit -> stale caches -> invalidation -> rebuild, and gives a practical reading order through the cache surface
- deepened the exported constant docs in [src/neat/cache/core/cache.constants.ts](../src/neat/cache/core/cache.constants.ts) so `GENOME_CACHE_FIELD_KEYS` now reads as the invalidation contract between genome-editing helpers and later derived reads rather than just a literal key list
- expanded the helper docs in [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts) so `invalidateGenomeCaches()` now explains why memoized compatibility, activation, and trace data become unsafe after writes and why one shared cleanup helper keeps invalidation deterministic across the controller

Decision:
- the root cache chapter now reads like a guided cache-invalidation chapter rather than a thin helper surface
- the strongest remaining adjacent gap has shifted to [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts), because the generated [src/neat/cache/core/README.md](../src/neat/cache/core/README.md) is still mostly driven by the constant section and remains comparatively terse at the lower-level bookkeeping layer

### Compatibility verification pass

Goals:
- verify whether [src/neat/compat/README.md](../src/neat/compat/README.md) still had a meaningful educational gap after the earlier compatibility and compatibility-core passes
- confirm whether the real remaining issue was chapter framing, helper-level explanation, fallback innovation interpretation, or the chapter's connection to speciation, diversity, and lineage
- make only source-first JSDoc refinements in [src/neat/compat/compat.ts](../src/neat/compat/compat.ts) if the generated surface was already broadly strong

Progress:
- re-read the generated [src/neat/compat/README.md](../src/neat/compat/README.md), the nearby [src/neat/README.md](../src/neat/README.md), and the source file [src/neat/compat/compat.ts](../src/neat/compat/compat.ts) and confirmed the stale handoff prompt no longer matched the actual documentation state: the compatibility chapter already had a real opening, a useful reading order, and materially stronger helper-level guidance
- narrowed the remaining gap to interpretation rather than structure: the generated chapter already framed compatibility well, but the source comments could still say more clearly that compatibility distance is a neighborhood-boundary read rather than a generic quality score, and that fallback innovation ids represent deterministic structural alignment rather than proven shared innovation history
- tightened the module and helper JSDoc in [src/neat/compat/compat.ts](../src/neat/compat/compat.ts) so the generated chapter now states the practical controller decisions supported by compatibility distance, clarifies how lineage complements rather than duplicates compatibility, and adds a more explicit caveat about how to read fallback innovation matches

Decision:
- the compatibility root chapter remains strong and is no longer the highest-leverage adjacent docs target
- the next concrete chapter target should stay [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts), where the lower-level invalidation mechanics still need a dedicated educational opening

### Cache core chapter pass

Goals:
- turn the generated [src/neat/cache/core/README.md](../src/neat/cache/core/README.md) from a constant-led stub into a real invalidation-mechanics chapter
- improve the source-first JSDoc around which genome-owned fields become stale after writes, why the invalidation list stays explicit, and how the shared cleanup helper complements the stronger root cache bridge
- verify that docs regeneration now gives the cache core area a proper chapter opening instead of starting with the constant section by accident

Progress:
- re-read the generated [src/neat/cache/README.md](../src/neat/cache/README.md), [src/neat/cache/core/README.md](../src/neat/cache/core/README.md), and the source files [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts) plus [src/neat/cache/core/cache.constants.ts](../src/neat/cache/core/cache.constants.ts) and confirmed the main gap was structural: the core README was still opening with the constant summary rather than a real mechanics introduction
- expanded the source-first chapter intro in [src/neat/cache/core/cache.constants.ts](../src/neat/cache/core/cache.constants.ts) so the generated core README now explains the lower-level invalidation contract, the root-versus-core ownership split, the stale-field question this chapter answers, and a Mermaid flow from write to stale caches to shared cleanup and rebuild
- deepened the helper-level JSDoc in [src/neat/cache/core/cache.core.ts](../src/neat/cache/core/cache.core.ts) so `invalidateGenomeCaches()` now reads as the shared execution path beneath that contract: a safe post-write cleanup step, intentionally simple, reusable across many write paths, and explicitly separated from the structural behavior change itself
- regenerated docs and verified that [src/neat/cache/core/README.md](../src/neat/cache/core/README.md) now opens with a real mechanics-oriented chapter introduction instead of the earlier constant-led stub

Decision:
- the cache core chapter now reads like a guided invalidation-mechanics chapter rather than a compact constant-plus-helper inventory
- the next clearly thin root-level NEAT chapter is [src/neat/harness/neat.harness.types.ts](../src/neat/harness/neat.harness.types.ts), whose generated [src/neat/harness/README.md](../src/neat/harness/README.md) is still mostly a terse type listing

### Harness chapter pass

Goals:
- turn the generated [src/neat/harness/README.md](../src/neat/harness/README.md) from a terse type inventory into a readable test-boundary chapter
- improve the source-first JSDoc around why these harness types exist, what lineage and phased-complexity behaviors they expose for assertions, and how readers should move back to the stronger runtime chapters for semantics
- verify that docs regeneration makes the harness surface read like a seam map rather than a second undocumented facade

Progress:
- re-read the generated [src/neat/harness/README.md](../src/neat/harness/README.md), the source file [src/neat/harness/neat.harness.types.ts](../src/neat/harness/neat.harness.types.ts), and the nearby stronger chapters [src/neat/lineage/README.md](../src/neat/lineage/README.md), [src/neat/adaptive/README.md](../src/neat/adaptive/README.md), and [src/neat/speciation/README.md](../src/neat/speciation/README.md) and confirmed the main gap was mostly framing: the harness README existed, but it did not yet explain why tests should use these narrow contracts or how they relate back to the runtime controller story
- expanded the chapter intro in [src/neat/harness/neat.harness.types.ts](../src/neat/harness/neat.harness.types.ts) so the generated README now explains the harness boundary as a deliberate testing seam, maps it back to `lineage/`, `speciation/`, and `adaptive/`, adds a Mermaid seam diagram, and gives a concrete reading order through the three exported harness types
- deepened the type-level JSDoc in [src/neat/harness/neat.harness.types.ts](../src/neat/harness/neat.harness.types.ts) so `LineageTrackedNetwork`, `NeatLineageHarness`, and `PhasedComplexityHarness` now explain what each type exposes for assertions, why tests should keep the contract narrow, and where readers should go for the runtime meaning behind those test seams
- regenerated docs and verified that [src/neat/harness/README.md](../src/neat/harness/README.md) now reads like a guided test-boundary chapter instead of a bare type listing

Decision:
- the harness chapter now reads like a real seam map for lineage and phased-complexity testing rather than a terse type stub
- the next clearly thin shared surface is [src/neat/shared/neat.shared.types.ts](../src/neat/shared/neat.shared.types.ts), whose generated [src/neat/shared/README.md](../src/neat/shared/README.md) already has a reasonable opening but is still dominated by a long, relatively dry contract inventory

### Topology-intent chapter pass

Goals:
- turn the generated [src/neat/topology-intent/README.md](../src/neat/topology-intent/README.md) from a compact helper bridge into a clearer topology-policy chapter
- improve the source-first JSDoc around why this policy lives in one small bridge, how feed-forward intent is recognized from mutation configuration, why promotion stays conservative, and how this boundary connects back to `mutation/`, `init/`, and `helpers/`
- verify that docs regeneration makes the chapter read like one short decision flow rather than a disconnected list of helpers and tiny types

Progress:
- re-read the generated [src/neat/topology-intent/README.md](../src/neat/topology-intent/README.md), the source file [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts), and the nearby [src/neat/mutation/README.md](../src/neat/mutation/README.md), [src/neat/init/README.md](../src/neat/init/README.md), and [src/neat/helpers/README.md](../src/neat/helpers/README.md) chapters and confirmed the main gap was not missing API coverage so much as policy explanation: the chapter already named the right boundary, but it did not yet explain the full flow from feed-forward mutation intent to eligibility checks to actual runtime promotion
- expanded the chapter intro in [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts) so the generated README now explains the boundary as a policy bridge rather than a graph-rewrite layer, maps the three-step decision flow across policy recognition, structural eligibility, and runtime promotion, and points readers back to the surrounding `mutation/`, `init/`, and `helpers/` chapters for the larger controller story
- deepened the helper and type-level JSDoc in [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts) so `TopologyIntentMutationMethod`, `TopologyIntentGenome`, `usesFeedForwardMutationPolicy()`, `matchesCanonicalFeedForwardPool()`, `promoteGenomeToFeedForwardIntentWhenEligible()`, and `isGenomeEligibleForFeedForwardIntentPromotion()` now explain why the bridge stays strict, why promotion is conservative, and why a failed eligibility check means "do not relabel yet" rather than "the genome is invalid"
- regenerated docs and verified that [src/neat/topology-intent/README.md](../src/neat/topology-intent/README.md) now reads like a concise topology-policy chapter instead of a mostly mechanical helper listing

Decision:
- the topology-intent chapter now gives a clear policy flow from mutation configuration to safe runtime promotion without over-expanding a deliberately small boundary
- the next clearly thin nearby root-level chapter is [src/neat/helpers/neat.helpers.ts](../src/neat/helpers/neat.helpers.ts), whose generated [src/neat/helpers/README.md](../src/neat/helpers/README.md) has a strong opening but still leaves its contract types relatively under-explained

### Shared contracts chapter pass

Goals:
- turn the generated [src/neat/shared/README.md](../src/neat/shared/README.md) from a long structural type inventory into a stronger shared-contracts chapter
- improve the source-first JSDoc around why these light-weight contracts exist, how they cluster into reader-facing families, and which shared types matter most to the stronger speciation, telemetry, objectives, species, and test chapters
- verify that docs regeneration gives the shared chapter a clearer reading order without trying to rewrite every type in the file

Progress:
- re-read the generated [src/neat/shared/README.md](../src/neat/shared/README.md), the source file [src/neat/shared/neat.shared.types.ts](../src/neat/shared/neat.shared.types.ts), and nearby stronger chapters such as [src/neat/telemetry/README.md](../src/neat/telemetry/README.md), [src/neat/objectives/README.md](../src/neat/objectives/README.md), and [src/neat/species/README.md](../src/neat/species/README.md) and confirmed the main gap was not missing content so much as weak navigation: the chapter had a serviceable opening, but the generated surface still dropped into a long flat shelf of contracts without much family-level guidance
- expanded the shared chapter intro in [src/neat/shared/neat.shared.types.ts](../src/neat/shared/neat.shared.types.ts) so the generated README now explains the file as the controller's common language layer, groups the exports into five practical contract families, and gives a concrete reading order through host, genome, policy, evidence, and history/archive types
- deepened selected high-leverage contract docs in [src/neat/shared/neat.shared.types.ts](../src/neat/shared/neat.shared.types.ts) so `SpeciationOptions`, `SpeciesLastStats`, `SpeciationHarnessContext`, `NeatLike`, `ObjectiveDescriptor`, `GenomeLike`, `GenomeDetailed`, `SpeciesLike`, `NeatOptions`, `DiversityStats`, `LineageSnapshot`, `PerformanceMetrics`, `TelemetryEntry`, `SpeciesHistoryEntry`, and `ParetoArchiveEntry` now explain not just what they contain but which stronger chapter consumes them and why the shared surface stays intentionally small
- regenerated docs and verified that [src/neat/shared/README.md](../src/neat/shared/README.md) now reads more like a shared-contracts map and less like an unstructured type shelf

Decision:
- the shared chapter now gives readers a usable map across the common type layer instead of forcing them to infer the major contract families from a long export list
- the next clearly thin remaining surface is [src/neat/topology-intent/neat.topology-intent.ts](../src/neat/topology-intent/neat.topology-intent.ts), whose generated [src/neat/topology-intent/README.md](../src/neat/topology-intent/README.md) has a decent opening but still stays comparatively compact around the practical policy flow and type-level explanation

### Helpers chapter pass

Goals:
- turn the generated [src/neat/helpers/README.md](../src/neat/helpers/README.md) from a strong chapter opening plus thin contract stubs into a fuller population-entry chapter
- improve the source-first JSDoc around why these helpers stay together, which invariants they protect when genomes enter the live population, and how the local contracts support pool bootstrapping, parent-derived spawning, and external provenance insertion
- verify that docs regeneration and a TypeScript check still pass after the source-only documentation edits

Progress:
- re-read the generated [src/neat/helpers/README.md](../src/neat/helpers/README.md), the source file [src/neat/helpers/neat.helpers.ts](../src/neat/helpers/neat.helpers.ts), and the nearby [src/neat/init/README.md](../src/neat/init/README.md) plus [src/neat/topology-intent/README.md](../src/neat/topology-intent/README.md) chapters and confirmed the main remaining gap was not the top-level chapter framing; it was the thinness of the helper-contract sections and the missing connection between `createPool()`, `spawnFromParent()`, and `addGenome()` as one provenance-and-bootstrapping flow
- expanded the chapter intro in [src/neat/helpers/neat.helpers.ts](../src/neat/helpers/neat.helpers.ts) so the generated README now frames the file as the population-entry boundary for the shared controller, explains the three-part entry story, and names the controller-owned invariants that stay synchronized here
- deepened the type-level JSDoc in [src/neat/helpers/neat.helpers.ts](../src/neat/helpers/neat.helpers.ts) so `GenomeWithMetadata`, `MutationMethod`, and `NeatControllerForHelpers` now explain why the helper chapter uses intentionally narrow contracts instead of widening into a second public facade
- improved the public helper docs in [src/neat/helpers/neat.helpers.ts](../src/neat/helpers/neat.helpers.ts) so the generated chapter now explains `spawnFromParent()` as a provisional provenance path, `addGenome()` as the admission and normalization step for external or newly accepted genomes, and `createPool()` as the generation-zero bootstrap path rather than a generic import helper
- regenerated docs with `npm run docs`, re-read the compiled [src/neat/helpers/README.md](../src/neat/helpers/README.md), and ran `npx tsc --noEmit -p tsconfig.json` with exit code 0 to verify the source-first pass stayed synchronized and type-safe

Decision:
- the helpers chapter now reads like a guided population-entry chapter rather than a strong opening followed by thin contract stubs
- the next clearly thin adjacent surface is [src/neat/init/neat.init.ts](../src/neat/init/neat.init.ts), whose generated [src/neat/init/README.md](../src/neat/init/README.md) still reads more like a compact constructor-helper note than a fuller bootstrap chapter

Remaining gaps:
- some generated example blocks across the broader NEAT docs still render more compactly than ideal because of the docs generator rather than the source comments
- the constructor bootstrap surface in [src/neat/init/README.md](../src/neat/init/README.md) remains comparatively thin beside the now-stronger helpers and topology-intent entry-path chapters

Next step:
- start a focused educational-docs pass in [src/neat/init/neat.init.ts](../src/neat/init/neat.init.ts) so the generated [src/neat/init/README.md](../src/neat/init/README.md) reads more like a constructor-bootstrap chapter and less like a short helper note

## Handoff Prompt

```text
Continue the educational-docs pass for the next adjacent NEAT chapter using plans/neat-docs.plans.md as the source of truth.

Current target:
- src/neat/init/neat.init.ts feeding src/neat/init/README.md

What to verify next:
- read the generated src/neat/init/README.md and confirm whether the main remaining gap is the compact chapter opening, the thinness of `initializeNeatConstructor()`, or weak type-level guidance for the bootstrap request and host contracts
- improve the source-first JSDoc in src/neat/init/neat.init.ts around the constructor bootstrap boundary: why startup policy lives outside the main `Neat` class file, which initialization order is intentionally preserved, and how the helper coordinates defaults, controller state, initial population creation, lineage enablement, and RNG binding without becoming a second facade
- preserve the stronger root src/README.md plus the now-solid telemetry, export, multi-objective, evolve, evaluate, selection, speciation, species, objectives, pruning, mutation, and adaptive chapters unless a new generator issue forces a revisit

If more work is needed:
- keep edits source-first in src/neat/init/neat.init.ts
- do not hand-edit generated src/**/README.md files
- regenerate docs with `npm run docs`
- run `npx tsc --noEmit -p tsconfig.json` after the doc-affecting source edits
- update plans/neat-docs.plans.md with achievements, remaining gaps, and the next concrete chapter target before ending the session

Current status:
- the generated root src/README.md opening still fits the broader `src` surface
- the telemetry subtree now has a coherent first-pass educational surface across bridge, recorder, metrics, runtime, facade, accessors, exports, and local types chapters
- the export chapter now reads like a guided pause-and-resume chapter rather than a serialization inventory
- the multi-objective root chapter now reads like a guided ranking chapter rather than a terse algorithm note
- the evolve chapter now reads like a guided generation-update chapter rather than a constants-and-signature dump
- the evaluate chapter now reads like a guided scoring-and-adaptation chapter rather than a compact pipeline summary plus constant list
- the selection chapter now reads like a guided selection-and-ordering chapter rather than a compact utility index
- the speciation chapter now reads like a guided species-assignment-and-maintenance chapter rather than a compact lifecycle summary with terse helper sections
- the species chapter now reads like a guided species-reporting chapter rather than a compact read-helper index
- the objectives chapter now reads like a guided objective-management chapter rather than a compact method index
- the pruning chapter now reads like a guided pruning-control chapter rather than a compact pair of method summaries
- the mutation chapter now reads like a guided mutation-control chapter rather than a mixed helper and constant inventory
- the adaptive chapter now reads like a guided adaptive-control chapter rather than a compact method catalog
- the diversity chapter now reads like a guided diversity-reporting chapter rather than a compact read-model summary
- the lineage chapter now reads like a guided lineage-evidence chapter rather than a compact ancestry-helper index
- the compatibility chapter has now been re-verified as a solid controller-facing bridge after a narrow interpretation pass
- the cache core chapter now reads like a guided invalidation-mechanics chapter rather than a compact constant-plus-helper inventory
- the harness chapter now reads like a guided test-boundary chapter rather than a terse type listing
- the shared chapter now reads like a guided common-language map rather than a long type shelf
- the topology-intent chapter now reads like a concise topology-policy chapter rather than a compact helper bridge
- the helpers chapter now reads like a guided population-entry chapter rather than a strong opening followed by thin contract stubs
- the next highest-leverage docs move should now shift into the init chapter
```
