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

### 2026-03-15 - Session start

Goals:
- turn the root NEAT docs into a real chapter introduction with a strong opening narrative
- add at least one diagram to orient readers around the lifecycle
- improve the public `NeatOptions` and `Neat` explanations so the generated docs teach workflow, not just signatures
- regenerate docs and verify the generated output

Progress:
- inspected [src/neat/README.md](../src/neat/README.md) and confirmed it is not the root NEAT chapter surface for this task
- traced the actual generated NEAT surface to [src/README.md](../src/README.md)
- confirmed the main gap is in [src/neat.ts](../src/neat.ts): the generated chapter currently has a bare module heading and a `default` class heading with little orientation

Next action:
- rewrite root JSDoc in [src/neat.ts](../src/neat.ts), regenerate docs, and then update this plan with the verification result

### 2026-03-15 - Session end

Achievements:
- rewrote the root module introduction in [src/neat.ts](../src/neat.ts) so the generated docs explain the NEAT lifecycle, the reason the controller stays orchestration-first, and where to read next
- added a Mermaid lifecycle diagram in [src/neat.ts](../src/neat.ts) to orient readers before they hit the API surface
- promoted the public class docs from an anonymous generated `default` section to a named `Neat` section by switching to a named class with a default export in [src/neat.ts](../src/neat.ts)
- expanded the public teaching surface for `NeatOptions`, constructor semantics, evolution, evaluation, telemetry, objectives, RNG state, and persistence helpers in [src/neat.ts](../src/neat.ts)
- regenerated [src/README.md](../src/README.md) with `npm run docs` and verified that:
	- the opening now presents a long NEAT-oriented chapter introduction
	- the Mermaid lifecycle diagram renders into the generated README markdown
	- the public class heading now renders as `Neat` instead of `default`

Validation:
- `npm run docs` completed successfully
- no editor diagnostics were reported for [src/neat.ts](../src/neat.ts) or [plans/nead-docs.plans.md](./nead-docs.plans.md)

External sources and media:
- none added in this session
- no Wikipedia or third-party media was used, so no extra attribution or license obligations were introduced

Remaining gaps:
- the generated `src/README.md` opening is now NEAT-centered because the docs generator lifts the module intro high in the root file; that is acceptable for this pass, but a future docs-generator refinement could place per-file introductions closer to their own file sections
- several lower-level public methods in [src/neat.ts](../src/neat.ts) still have shorter descriptions than the best chaptered READMEs under `src/neat/**`

Next step:
- all folders and subfolders within `src\neat`

### 2026-03-15 - Follow-up verification pass

Goals:
- verify that the generated root `src/README.md` opening still reads correctly as a library-level chapter intro after the earlier generator changes
- decide whether the next highest-leverage docs pass should stay in [src/neat.ts](../src/neat.ts) or move into one adjacent NEAT chapter
- improve any remaining root public helpers that still read like terse API stubs

Progress:
- re-read the generated opening in [src/README.md](../src/README.md) and confirmed it still works as the root `src` chapter because it frames `src/neat.ts` as the NEAT controller map rather than pretending to describe the whole repository
- compared the root surface with neighboring chapter outputs, including [src/neat/README.md](../src/neat/README.md), [src/neat/multiobjective/README.md](../src/neat/multiobjective/README.md), [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md), and [src/neat/export/README.md](../src/neat/export/README.md)
- confirmed the biggest remaining teaching gap was still in the root class surface: several public lifecycle, archive, telemetry-reset, and persistence helpers in [src/neat.ts](../src/neat.ts) were materially shorter than the adjacent chapter docs they front
- expanded those root JSDoc blocks in [src/neat.ts](../src/neat.ts) so the generated root chapter better explains when to use direct mutation, pruning hooks, parent and offspring helpers, telemetry clearing and export, archive inspection, diversity and performance snapshots, and population-versus-full-state persistence

Validation:
- editor diagnostics remained clean for [src/neat.ts](../src/neat.ts) and [plans/nead-docs.plans.md](./nead-docs.plans.md)
- `npm run docs` completed successfully after the follow-up JSDoc edits
- the regenerated [src/README.md](../src/README.md) now carries the deeper root-helper explanations for mutation, pruning, telemetry export and reset, species and Pareto inspection, diversity and performance snapshots, and persistence helpers

Decision:
- the regenerated root helper sections read like a guided API tour rather than terse stubs, so the next highest-leverage docs target is no longer another broad root pass
- the next adjacent chapter to deepen should be telemetry, because the telemetry subtree is already well-factored but lacks one obvious top-level `src/neat/telemetry/README.md` bridge page

Remaining gaps:
- the root surface is stronger, but it still exposes many public methods; a later polish pass could still add one or two compact usage examples for archive inspection and state restore flows
- the telemetry area is discoverable through subchapter READMEs, but there is no single top-level telemetry chapter README that mirrors the guidance quality of the multi-objective surface

Next step:
- start a telemetry chapter bridge pass by tracing which source comments should feed a new or improved top-level telemetry teaching surface before touching any generated README output

### 2026-03-15 - Telemetry bridge start

Goals:
- trace which existing telemetry source comments should feed a stronger top-level telemetry chapter surface
- verify whether the missing bridge is a wording problem or a source-mapping problem
- create the smallest source-first fix that can generate a real top-level telemetry chapter

Progress:
- inspected the telemetry folder structure and confirmed the root gap is structural: [src/neat/telemetry](../src/neat/telemetry) had chaptered subfolders but no root TypeScript file and therefore no obvious source to generate a top-level telemetry README
- traced the strongest existing teaching comments to the root files that already own the telemetry story: [src/neat/telemetry/recorder/telemetry.recorder.ts](../src/neat/telemetry/recorder/telemetry.recorder.ts), [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts), [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts), [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts), [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts), and [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts)
- confirmed the bridge narrative should synthesize two complementary paths already present in source comments: the write path (`recorder` -> `metrics` -> `runtime`) and the read path (`facade` -> `accessors` -> `exports`)
- added [src/neat/telemetry/telemetry.ts](../src/neat/telemetry/telemetry.ts) as the new root source bridge so generated docs have a top-level telemetry chapter to compile from

Validation:
- editor diagnostics were clean for [src/neat/telemetry/telemetry.ts](../src/neat/telemetry/telemetry.ts) and [plans/nead-docs.plans.md](./nead-docs.plans.md)
- `npm run docs` completed successfully after adding the telemetry bridge source
- the generated [src/neat/telemetry/README.md](../src/neat/telemetry/README.md) now exists and opens with the intended bridge narrative, including the write-path versus read-path split and the recommended-reading links into the telemetry subtree

Decision:
- the missing telemetry bridge was primarily a source-mapping gap, not just weak prose in existing child chapters
- the new top-level telemetry surface is strong enough to shift the next work item from "make the chapter exist" to "decide which child chapter deserves the next educational deepening pass"
- from a pedagogical perspective, the thinnest telemetry child chapter is [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts) feeding [src/neat/telemetry/metrics/README.md](../src/neat/telemetry/metrics/README.md): it carries the highest concept density with the least reader guidance about grouping, reading order, and why each metric family matters

Remaining gaps:
- the new bridge file is intentionally high-level; it may still benefit from one short concrete example later if readers need a faster path from chapter intro to actual `Neat` usage
- the metrics chapter currently reads more like an export of helper signatures than a guided explanation of metric families such as diversity, lineage, complexity, objective lifecycle, RNG state, and performance timing

Next step:
- start the next focused educational-docs pass in [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts) so the generated [src/neat/telemetry/metrics/README.md](../src/neat/telemetry/metrics/README.md) teaches the metric families, reading order, and runtime purpose instead of only enumerating helpers

### 2026-03-15 - Telemetry metrics pass

Goals:
- turn the metrics chapter from a flat helper index into a teaching surface that explains why the metric families exist
- preserve the existing source layout while improving the generated [src/neat/telemetry/metrics/README.md](../src/neat/telemetry/metrics/README.md)
- verify that the improved chapter still reads clearly after docs regeneration

Progress:
- re-read the metrics source and confirmed the main gap was concentrated in the root chapter intro at [src/neat/telemetry/metrics/telemetry.metrics.ts](../src/neat/telemetry/metrics/telemetry.metrics.ts), not in a single missing helper description
- rewrote that root module introduction so it now teaches the six metric families: diversity and entropy, lineage, objectives and Pareto signals, complexity, RNG state, and performance timing
- added a reading order and a Mermaid diagram so the generated metrics chapter explains how those evidence layers feed one telemetry entry rather than only listing exports

Validation:
- `npm run docs` completed successfully after the metrics-source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the metrics-source JSDoc update
- the regenerated [src/neat/telemetry/metrics/README.md](../src/neat/telemetry/metrics/README.md) now opens with a clearer evidence-first explanation instead of the previous export-list framing

Decision:
- the metrics chapter is no longer the thinnest telemetry teaching surface; the new opening now explains both metric-family purpose and reading order well enough for a first pass
- the next telemetry refinement should likely be narrower and example-driven rather than another broad chapter-opening rewrite

Remaining gaps:
- the metrics chapter still becomes signature-heavy after the stronger introduction, so a later pass could add a compact example or a smaller second-level bridge for one high-value family such as lineage or objectives
- the recorder and facade chapters are now the next candidates if the goal shifts from conceptual mapping to concrete `Neat` usage patterns

Next step:
- if telemetry docs continue, choose between an example-focused recorder pass and a user-facing facade pass depending on whether the next reader need is write-side pipeline understanding or public inspection workflow

### 2026-03-15 - Telemetry recorder pass

Goals:
- turn the recorder chapter into a true write-side walkthrough for one generation snapshot
- remove malformed example rendering from the generated recorder README
- preserve the existing source structure while making the compiled chapter easier to teach from

Progress:
- re-read [src/neat/telemetry/recorder/telemetry.recorder.ts](../src/neat/telemetry/recorder/telemetry.recorder.ts) and confirmed the core gap was at the chapter-introduction and public-helper JSDoc level, not in runtime behavior
- rewrote the recorder chapter opening so it now teaches the full write path from controller state to built entry, selection, buffering, streaming, and trimming
- added a Mermaid lifecycle diagram and a practical reading order so the generated chapter explains how the recorder relates to `metrics/`, `runtime/`, and `facade/`
- cleaned up the malformed `Example:` blocks around `applyTelemetrySelect`, `structuralEntropy`, `computeDiversityStats`, `recordTelemetryEntry`, and `buildTelemetryEntry` so the generated README now renders those examples cleanly

Validation:
- `npm run docs` completed successfully after the recorder-source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the recorder-source JSDoc update
- the regenerated [src/neat/telemetry/recorder/README.md](../src/neat/telemetry/recorder/README.md) now reads as an end-to-end write-path chapter and no longer contains the earlier duplicated or malformed example labels

Decision:
- the recorder chapter is now strong enough as the write-side companion to the metrics chapter
- if telemetry docs continue, the next highest-value pass should likely shift to the public inspection side rather than another broad recorder rewrite

Remaining gaps:
- the recorder chapter could still benefit from one future example that shows a full `buildTelemetryEntry()` plus `recordTelemetryEntry()` flow in a slightly more realistic `Neat` loop
- the facade chapter is now the clearest next candidate if the goal becomes helping end users understand how to inspect telemetry after it has been recorded

Next step:
- if the telemetry docs pass continues, move to [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) for a user-facing inspection and export workflow pass

### 2026-03-15 - Telemetry facade pass

Goals:
- turn the facade chapter into a reader-facing map of post-run inspection workflows instead of a flat export list
- verify once more that the generated root [src/README.md](../src/README.md) opening still fits the broader `src` surface before fully leaving the root chapter behind
- decide which telemetry child chapter should follow the facade pass

Progress:
- re-skimmed the generated opening in [src/README.md](../src/README.md) and confirmed it still fits the broader root `src` surface: it reads as the NEAT controller chapter map, not as an overclaim about every file in the repository
- rewrote the module-level introduction in [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) so the generated facade chapter now teaches the main inspection workflows: recent telemetry windows, objective and Pareto inspection, species and lineage reads, diversity and performance checks, and reset helpers
- added a Mermaid workflow diagram in [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) so the compiled chapter shows how a caller moves from `Neat` runtime state to specific read models and export surfaces
- expanded several facade helper JSDoc blocks in [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts) so the generated [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md) better explains when to use telemetry windows, CSV export, objective summaries, species summaries, species history, Pareto inspection, diversity reads, and coarse timing checks

Validation:
- `npm run docs` completed successfully after the facade-source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the facade-source JSDoc update
- editor diagnostics were clean for [src/neat/telemetry/facade/telemetry.facade.ts](../src/neat/telemetry/facade/telemetry.facade.ts)
- the regenerated [src/neat/telemetry/facade/README.md](../src/neat/telemetry/facade/README.md) now opens with a workflow-oriented chapter introduction and preserves the new examples and diagram cleanly enough for a first pass

Decision:
- the facade chapter is no longer the weakest user-facing telemetry surface; it now functions as the public inspection companion to the write-side recorder and metrics chapters
- [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md) is already reasonably explanatory, while [src/neat/telemetry/accessors/README.md](../src/neat/telemetry/accessors/README.md) is still comparatively terse and utility-shaped
- the next highest-leverage telemetry docs pass should therefore move into [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) so the generated accessors chapter explains why those low-level reads exist and how they support both internal telemetry code and the public facade

Remaining gaps:
- some facade examples still render as compact plain example blocks rather than fenced snippets in the generated README; this is acceptable for now but may deserve a docs-generator formatting refinement later
- the accessors chapter still reads mostly like a utility inventory instead of a teaching bridge between internal telemetry runtime state and the public facade helpers

Next step:
- start a focused educational-docs pass in [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) so the generated [src/neat/telemetry/accessors/README.md](../src/neat/telemetry/accessors/README.md) explains the low-level read model, default lineage sampling limit, and why those tiny helpers stay below the public facade

### 2026-03-15 - Telemetry accessors pass

Goals:
- turn the accessors chapter into a readable explanation of the telemetry read model beneath the public facade
- keep the chapter small and source-first without promoting it into another user-facing facade layer
- verify that the generated accessors README now teaches the boundary instead of only listing tiny helpers

Progress:
- rewrote the module introduction in [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) so the generated chapter now explains the audience split between `facade/` and `accessors/`, why these helpers stay intentionally narrow, and how other telemetry chapters reuse them
- expanded the JSDoc for [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) exports so `getTelemetryBuffer`, `clearTelemetryBuffer`, `getObjectiveEventsSnapshot`, `buildLineageSnapshot`, `getCachedDiversityStats`, `getPerformanceStatsSnapshot`, `TelemetryAccessorHost`, and `LINEAGE_SNAPSHOT_DEFAULT_LIMIT` now explain intent, invariants, and usage context instead of reading like terse utility stubs
- added compact examples to the most instructive helpers in [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts) so the generated chapter shows how these low-level reads are actually consumed during inspection and testing

Validation:
- `npm run docs` completed successfully after the accessors-source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the accessors-source JSDoc update
- editor diagnostics were clean for [src/neat/telemetry/accessors/telemetry.accessors.ts](../src/neat/telemetry/accessors/telemetry.accessors.ts)
- the regenerated [src/neat/telemetry/accessors/README.md](../src/neat/telemetry/accessors/README.md) now explains the low-level read model clearly enough for a first-time telemetry reader and no longer feels like a bare utility index

Decision:
- the accessors chapter is no longer the weakest telemetry teaching surface; it now reads like the internal read-model companion to the facade chapter
- the next telemetry docs gap is narrower: [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md) has a solid chapter opening, but several helper-level sections in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) still read more like serializer inventory than a guided explanation of header discovery and deterministic export shaping

Remaining gaps:
- generated examples still render as compact plain example blocks rather than fenced snippets in some chapters; that remains a docs-generator formatting issue rather than a source-comment correctness issue
- the exports chapter still has educational headroom at the helper level, especially around header discovery, flattening rules, and why the CSV surface stays deterministic across sparse telemetry windows

Next step:
- start a focused educational-docs pass in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) so the generated [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md) teaches header discovery, sparse-field flattening, and deterministic export behavior more clearly

### 2026-03-15 - Telemetry exports main-file pass

Goals:
- deepen the helper-level teaching surface in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) without reopening the already-strong chapter introduction
- explain how header discovery, sparse-field flattening, and deterministic CSV shaping work together
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module and helper JSDoc in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) so the generated exports chapter now explains the main design constraint of the exporter: preserving a stable, comparable export shape across uneven telemetry windows
- improved the teaching surface for `TelemetryHeaderInfo`, `exportTelemetryJSONL`, `exportTelemetryCSV`, `exportSpeciesHistoryCSV`, `collectTelemetryHeaderInfo`, `buildTelemetryHeaders`, `serializeTelemetryEntry`, and `buildSpeciesHistoryCsv` in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts)
- added small examples for the CSV exporters in [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts) so the generated chapter shows how callers inspect headers and short windows in practice

Validation:
- `npm run docs` completed successfully after the exports-source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the exports-source JSDoc update
- editor diagnostics were clean for [src/neat/telemetry/exports/telemetry.exports.ts](../src/neat/telemetry/exports/telemetry.exports.ts)
- the regenerated [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md) now explains deterministic export shaping and the main serializer helpers clearly enough for a first pass

Decision:
- the main exports file no longer feels like the weakest telemetry chapter surface; the largest remaining gap in this area has shifted into the companion utility chapter rather than the public-facing entrypoints
- the next narrow telemetry docs target should be [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts), because the generated helper sections there are still comparatively terse even after the main exports chapter improved

Remaining gaps:
- generated examples still render more compactly than ideal in some README surfaces; that remains a docs-generator formatting concern rather than a source-JSDoc correctness issue
- the exports utils chapter still has educational headroom around header-state mutation, dynamic species-history header collection, and row normalization for sparse controller state

Next step:
- start a focused educational-docs pass in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts) so the generated exports-utils section teaches the lower-level bookkeeping behind deterministic telemetry and species-history CSV shaping

### 2026-03-15 - Telemetry exports-utils pass

Goals:
- turn the exports utility section into a readable explanation of the bookkeeping behind deterministic telemetry CSV shaping
- make the generated exports-utils section explain header-state mutation, sparse-column enabling, and species-history normalization more clearly
- keep the pass source-first and avoid runtime behavior changes

Progress:
- rewrote the module and helper JSDoc in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts) so the generated exports-utils section now explains that this file is the preparatory layer behind stable CSV exports rather than a random grab-bag of helpers
- deepened the generated teaching surface for `TelemetryHeaderCollectionState`, `collectBaseKeys`, `collectGroupedMetricKeys`, `collectDiversityLineageMetrics`, `collectOptionalColumnPresence`, `ensureSpeciesHistoryArray`, `ensureMinimalSpeciesSnapshot`, `collectSpeciesHistoryHeaders`, `buildSpeciesHistoryStats`, `serializeSpeciesHistoryRow`, `resolveSpeciesHistoryCellValue`, and `safeStringifyCell` in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts)
- added compact examples to the most instructive species-history normalization helpers in [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts) so the generated section shows how deterministic backfilling and normalization are actually used

Validation:
- `npm run docs` completed successfully after the exports-utils source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the exports-utils source JSDoc update
- editor diagnostics were clean for [src/neat/telemetry/exports/telemetry.exports.utils.ts](../src/neat/telemetry/exports/telemetry.exports.utils.ts)
- the regenerated exports-utils section in [src/neat/telemetry/exports/README.md](../src/neat/telemetry/exports/README.md) now reads like a bookkeeping companion to the main exports chapter instead of a bare helper inventory

Decision:
- the exports area is now documented well enough for a first telemetry pass; both the public entrypoints and the lower-level bookkeeping layer read coherently
- the next telemetry docs gap is no longer in exports and now sits most clearly in [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts), whose opening is adequate but whose helper-level explanations remain comparatively small and utility-shaped

Remaining gaps:
- generated examples still render more compactly than ideal in some README surfaces; that remains a docs-generator formatting concern rather than a source-JSDoc correctness issue
- the runtime chapter still has educational headroom around why buffer initialization, safe streaming, and bounded trimming stay as a separate runtime safety layer beneath the recorder

Next step:
- start a focused educational-docs pass in [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts) so the generated [src/neat/telemetry/runtime/README.md](../src/neat/telemetry/runtime/README.md) teaches the runtime safety contract behind buffering, callback isolation, and bounded history retention

### 2026-03-15 - Telemetry runtime pass

Goals:
- turn the runtime chapter into a clearer explanation of the telemetry safety layer beneath the recorder
- improve the helper-level teaching surface for lazy buffer creation, callback isolation, and bounded retention
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module and helper JSDoc in [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts) so the generated runtime chapter now explains the containment boundary between the recorder and the live evolution loop
- improved the generated teaching surface for `ensureTelemetryBuffer`, `safelyStreamTelemetryEntry`, and `trimTelemetryBuffer` in [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts) so the README now explains why telemetry stays opt-in, why callback failures are swallowed, and why bounded retention is a runtime contract rather than an incidental implementation detail

Validation:
- `npm run docs` completed successfully after the runtime-source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the runtime-source JSDoc update
- editor diagnostics were clean for [src/neat/telemetry/runtime/telemetry.runtime.ts](../src/neat/telemetry/runtime/telemetry.runtime.ts)
- the regenerated [src/neat/telemetry/runtime/README.md](../src/neat/telemetry/runtime/README.md) now reads like a safety-layer chapter rather than a tiny utility list

Decision:
- the runtime chapter is now strong enough for the telemetry pass; it no longer feels like the main remaining thin spot in the subtree
- the remaining telemetry-local gap shifted from behavior chapters to the small contracts chapter under `types/`

Remaining gaps:
- generated examples still render more compactly than ideal in some README surfaces; that remains a docs-generator formatting concern rather than a source-JSDoc correctness issue
- the telemetry types chapter still reads correctly but was the last telemetry-local surface with near-stub-level entries before the next pass

Next step:
- start a focused educational-docs pass in [src/neat/telemetry/types/telemetry.types.ts](../src/neat/telemetry/types/telemetry.types.ts) so the generated [src/neat/telemetry/types/README.md](../src/neat/telemetry/types/README.md) explains the internal contract types used by recorder, runtime, exports, and facade helpers

### 2026-03-15 - Telemetry types pass

Goals:
- turn the telemetry types chapter from a list of names into a readable map of the internal telemetry contracts
- clarify why these contracts stay telemetry-local instead of being promoted into broader NEAT public types
- close out the first telemetry-focused educational-docs sweep with a coherent internal types chapter

Progress:
- expanded the module and type-level JSDoc in [src/neat/telemetry/types/telemetry.types.ts](../src/neat/telemetry/types/telemetry.types.ts) so the generated types chapter now explains why these contracts exist and how they support recorder, runtime, exports, and facade helpers
- improved the generated teaching surface for `TelemetryGenome`, `TelemetryDiversityOptions`, `TelemetryStreamOptions`, `TelemetryEntryRecord`, `OperatorStatsMap`, `TelemetryBufferContext`, `TelemetrySelectContext`, and `TelemetryCoreFields` in [src/neat/telemetry/types/telemetry.types.ts](../src/neat/telemetry/types/telemetry.types.ts)

Validation:
- `npm run docs` completed successfully after the telemetry-types source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the telemetry-types source JSDoc update
- editor diagnostics were clean for [src/neat/telemetry/types/telemetry.types.ts](../src/neat/telemetry/types/telemetry.types.ts)
- the regenerated [src/neat/telemetry/types/README.md](../src/neat/telemetry/types/README.md) now reads like an internal contract chapter instead of a bare list of aliases

Decision:
- the telemetry subtree now has a coherent first-pass educational surface across its bridge, recorder, metrics, runtime, facade, accessors, exports, and local types chapters
- the next highest-leverage docs move should leave telemetry and shift to one adjacent NEAT chapter rather than continuing to over-polish the same subtree
- the most sensible next chapter is [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts), because persistence is adjacent to telemetry in user workflow and still exposes many helper and contract sections that could be made more guided for first-time readers

Remaining gaps:
- some generated example blocks remain more compact than ideal because of the docs generator rather than the source comments
- telemetry subchapters under `src/neat/telemetry/facade/*` could still receive narrower follow-up passes later, but that is no longer the highest-leverage educational move for the root NEAT surface

Next step:
- start a focused educational-docs pass in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so the generated [src/neat/export/README.md](../src/neat/export/README.md) reads more like a guided persistence chapter and less like a serialization inventory

### 2026-03-15 - Export persistence pass

Goals:
- turn the export chapter from a persistence helper inventory into a clearer pause-and-resume workflow chapter
- improve the source-first JSDoc around population-only snapshots, controller-meta snapshots, and full-state restore flows
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module introduction in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so the generated export chapter now teaches the three-layer persistence ladder: population-only snapshots, meta-only snapshots, and full-state checkpoints
- improved the generated teaching surface for `GenomeJSON`, `InnovationMapEntry`, `GenomeWithSerialization`, `NeatControllerForExport`, `NetworkClass`, `NeatConstructor`, and `NeatStateJSON` in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so the supporting contracts now explain why they exist in the persistence boundary
- deepened the public helper docs for `importPopulation`, `exportState`, `importStateImpl`, `toJSONImpl`, and `fromJSONImpl` in [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts) so the generated [src/neat/export/README.md](../src/neat/export/README.md) now reads more like a guided pause-and-resume chapter than a flat serialization inventory

Validation:
- `npm run docs` completed successfully after the export-source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the export-source JSDoc update
- editor diagnostics were clean for [src/neat/export/neat.export.ts](../src/neat/export/neat.export.ts)
- the regenerated [src/neat/export/README.md](../src/neat/export/README.md) now carries the intended pause-and-resume ladder and stronger supporting-contract explanations

Decision:
- the export chapter is now strong enough for a first educational-docs pass; it no longer feels like the next urgent gap beside the root NEAT controller surface
- the next highest-leverage move should shift to another adjacent orchestration chapter rather than continuing to over-polish persistence contracts
- [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts) is the clearest next target because it is central to advanced NEAT behavior and its generated README still leans more technical than guided compared with the upgraded telemetry and export chapters

Remaining gaps:
- some generated example blocks remain more compact than ideal because of the docs generator rather than the source comments
- the export chapter could still support a future narrower pass on secondary contracts, but that is no longer the highest-leverage educational move

Next step:
- start a focused educational-docs pass in [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts) so the generated [src/neat/multiobjective/README.md](../src/neat/multiobjective/README.md) reads more like a guided ranking chapter and less like an algorithm summary

### 2026-03-15 - Multi-objective ranking pass

Goals:
- turn the multi-objective root chapter from a concise algorithm summary into a clearer guided ranking workflow
- improve the source-first JSDoc around fronts, crowding, ranking stages, and archive intent
- keep the pass source-first and avoid runtime behavior changes

Progress:
- expanded the module introduction in [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts) so the generated chapter now explains why multi-objective ranking exists, how Pareto fronts differ from single-score selection, and how crowding preserves spread along the frontier
- deepened the `fastNonDominated()` teaching surface in [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts) so the generated [src/neat/multiobjective/README.md](../src/neat/multiobjective/README.md) now explains the four-stage ranking flow, how to read returned fronts, why crowding matters, and why frontier archiving happens at this orchestration layer

Validation:
- `npm run docs` completed successfully after the multi-objective source JSDoc update
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the multi-objective source JSDoc update
- editor diagnostics were clean for [src/neat/multiobjective/multiobjective.ts](../src/neat/multiobjective/multiobjective.ts)
- the regenerated [src/neat/multiobjective/README.md](../src/neat/multiobjective/README.md) now reads more like a guided ranking chapter and less like a terse algorithm note

Decision:
- the multi-objective root chapter is now strong enough for a first pass and no longer feels like the most obvious adjacent gap after export
- the next highest-leverage move should shift to the evolve orchestration chapter, which remains central to the root NEAT story but still reads thinner than the upgraded telemetry, export, and multi-objective chapters

Remaining gaps:
- some generated example blocks remain more compact than ideal because of the docs generator rather than the source comments
- the evolve chapter currently exposes a central workflow but still opens abruptly and reads more like a function inventory than a guided generation-update story

Next step:
- start a focused educational-docs pass in [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts) so the generated [src/neat/evolve/README.md](../src/neat/evolve/README.md) reads more like a guided generation-update chapter and less like a constants-and-signature dump

### 2026-03-15 - Evolve orchestration pass

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

Validation:
- `npm run docs` completed successfully after the evolve-source JSDoc update and again after the source-mapping fix
- Mermaid validation passed during docs generation, including the new evolve lifecycle diagram
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the evolve-source JSDoc updates
- editor diagnostics were clean for [src/neat/evolve/evolve.ts](../src/neat/evolve/evolve.ts) and [plans/nead-docs.plans.md](./nead-docs.plans.md)
- the regenerated [src/neat/evolve/README.md](../src/neat/evolve/README.md) now opens with the intended chapter introduction and lifecycle diagram before the symbol sections begin

Decision:
- the evolve chapter now reads like a guided generation-update chapter rather than a constants-and-signature dump
- the strongest remaining adjacent gap has shifted upstream to [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts), because evaluation is the phase that feeds evolve directly but its generated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) still reads more like a compact pipeline summary and constant inventory than a chaptered explanation of the scoring lifecycle

Remaining gaps:
- the generated evolve chapter is substantially stronger, but [src/neat/evolve/evolve.types.ts](../src/neat/evolve/evolve.types.ts) still remains comparatively terse beside the orchestration narrative in the root chapter
- the generated signature still renders `Promise<default>` for the returned network because that surface follows the default-export naming of the architecture network type rather than the evolve chapter docs themselves
- some generated example blocks across the broader NEAT docs still render more compactly than ideal because of the docs generator rather than the source comments

Next step:
- start a focused educational-docs pass in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts) so the generated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) reads more like a guided scoring-and-adaptation chapter and less like a compact pipeline summary plus constant list

### 2026-03-15 - Evaluate orchestration pass

Goals:
- turn the evaluate chapter from a compact pipeline summary into a guided scoring-and-adaptation chapter
- improve the source-first JSDoc around per-genome versus population-level fitness execution, novelty blending, diversity-stat readiness, adaptive tuning hooks, post-evaluation speciation maintenance, and automatic entropy-objective registration
- verify that the generated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) now opens with a real chapter map and that its constants read like tuning controls instead of a bare inventory

Progress:
- confirmed the main gap in the generated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) was all three at once: the chapter opening was too compact, the public `evaluate()` description in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts) still read like a short checklist, and the exported constants were only lightly described
- expanded the module-level JSDoc in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts) so the source now teaches why evaluation exists as a separate orchestration chapter, explains the six-stage scoring lifecycle, adds a Mermaid flow diagram, gives a concrete reading order into the helper folders, and frames the exported constants as four tuning families
- deepened the public `evaluate()` JSDoc in [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts) so the generated chapter now explains the evidence-first mental model, the split between score production and lightweight controller maintenance, the important side effects, and the relationship between `evaluate()` and the later `evolve()` pass
- improved the educational surface of the exported evaluation constants in [src/neat/evaluate/shared/evaluate.constants.ts](../src/neat/evaluate/shared/evaluate.constants.ts) so the generated README now explains why the novelty, entropy-sharing, compatibility, and distance-coefficient defaults exist instead of listing them as terse labels

Validation:
- `npm run docs` completed successfully after the evaluate-source JSDoc update
- Mermaid validation passed during docs generation, including the new evaluate lifecycle diagram
- `npx tsc --noEmit -p tsconfig.json` completed successfully after the evaluate-source JSDoc updates
- editor diagnostics were clean for [src/neat/evaluate/evaluate.ts](../src/neat/evaluate/evaluate.ts), [src/neat/evaluate/shared/evaluate.constants.ts](../src/neat/evaluate/shared/evaluate.constants.ts), and [plans/nead-docs.plans.md](./nead-docs.plans.md)
- the regenerated [src/neat/evaluate/README.md](../src/neat/evaluate/README.md) now opens with the intended scoring-lifecycle introduction and diagram before the symbol sections begin, and the exported constants now read like tuning controls rather than a bare inventory

Decision:
- the evaluate chapter now reads like a guided scoring-and-adaptation chapter rather than a compact pipeline summary plus constant list
- the strongest remaining adjacent gap has shifted to [src/neat/selection/selection.ts](../src/neat/selection/selection.ts), because the generated [src/neat/selection/README.md](../src/neat/selection/README.md) still reads more like a utility index than a chaptered explanation of sorting, average and fittest summaries, and parent-selection strategy flow

Remaining gaps:
- the generated evaluate chapter is substantially stronger, but [src/neat/evaluate/shared/evaluate.types.ts](../src/neat/evaluate/shared/evaluate.types.ts) still remains more contract-shaped than guided beside the richer root orchestration narrative
- the generated evaluate helper chapters under `fitness/` and `novelty/` are now clearer in context, but some of their lower-level sections still remain signature-dense compared with the root chapter opening
- some generated example blocks across the broader NEAT docs still render more compactly than ideal because of the docs generator rather than the source comments

Next step:
- start a focused educational-docs pass in [src/neat/selection/selection.ts](../src/neat/selection/selection.ts) so the generated [src/neat/selection/README.md](../src/neat/selection/README.md) reads more like a guided selection-and-ordering chapter and less like a utility index

## Handoff Prompt

```text
Continue the educational-docs pass for the next adjacent NEAT chapter using plans/nead-docs.plans.md as the source of truth.

Current target:
- src/neat/selection/selection.ts feeding src/neat/selection/README.md

What to verify next:
- read the generated src/neat/selection/README.md and confirm whether the main remaining gap is chapter framing, method-level explanation, constant framing, or all three
- improve the source-first JSDoc in src/neat/selection/selection.ts around the controller-facing selection lifecycle: sorting, average and fittest summaries, active parent-selection strategies, fallback score semantics, and why the selection boundary stays split between `core/` and `facade/`
- preserve the stronger root src/README.md plus the now-solid telemetry, export, multi-objective, evolve, and evaluate chapters unless a new generator issue forces a revisit

If more work is needed:
- keep edits source-first in src/neat/selection/selection.ts
- do not hand-edit generated src/**/README.md files
- regenerate docs with `npm run docs`
- run `npx tsc --noEmit -p tsconfig.json` after the doc-affecting source edits
- update plans/nead-docs.plans.md with achievements, remaining gaps, and the next concrete chapter target before ending the session

Current status:
- the generated root src/README.md opening still fits the broader `src` surface
- the telemetry subtree now has a coherent first-pass educational surface across bridge, recorder, metrics, runtime, facade, accessors, exports, and local types chapters
- the export chapter now reads like a guided pause-and-resume chapter rather than a serialization inventory
- the multi-objective root chapter now reads like a guided ranking chapter rather than a terse algorithm note
- the evolve chapter now reads like a guided generation-update chapter rather than a constants-and-signature dump
- the evaluate chapter now reads like a guided scoring-and-adaptation chapter rather than a compact pipeline summary plus constant list
- the next highest-leverage docs move should now shift into the selection chapter
```
