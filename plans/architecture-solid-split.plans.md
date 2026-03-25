# Architecture Solid Split Plan

**Status:** [DONE]

## Scope

This plan tracks the small-chapter folderization of the flat root files under
[src/architecture](../src/architecture/README.md) so the architecture surface
stops depending on a handful of large root files and the generated README flow
stays educational instead of monolithic.

Plan context:

- [plans/Roadmap.md](Roadmap.md)
- [plans/Architecture_Primitives_Node_Group_Layer.md](Architecture_Primitives_Node_Group_Layer.md)

Primary reader:

- contributors who need the architecture surface to read like a set of focused
  primitive and builder chapters instead of a mixed root shelf.

## README Inventory

This inventory is the folder-by-folder split checklist for the architecture
root. Keep exactly one boundary pass active at a time.

- [src/architecture](../src/architecture/README.md)
- [src/architecture/architect](../src/architecture/architect/README.md)
- [src/architecture/group](../src/architecture/group/README.md)
- [src/architecture/nodePool](../src/architecture/nodePool/README.md)
- [src/architecture/activationArrayPool](../src/architecture/activationArrayPool/README.md)
- [src/architecture/layer](../src/architecture/layer/README.md)
- [src/architecture/network](../src/architecture/network/README.md)
- [src/architecture/network/activate](../src/architecture/network/activate/README.md)
- [src/architecture/network/bootstrap](../src/architecture/network/bootstrap/README.md)
- [src/architecture/network/connect](../src/architecture/network/connect/README.md)
- [src/architecture/network/deterministic](../src/architecture/network/deterministic/README.md)
- [src/architecture/network/evolve](../src/architecture/network/evolve/README.md)
- [src/architecture/network/gating](../src/architecture/network/gating/README.md)
- [src/architecture/network/genetic](../src/architecture/network/genetic/README.md)
- [src/architecture/network/mutate](../src/architecture/network/mutate/README.md)
- [src/architecture/network/onnx](../src/architecture/network/onnx/README.md)
- [src/architecture/network/onnx/export](../src/architecture/network/onnx/export/README.md)
- [src/architecture/network/onnx/export/layers](../src/architecture/network/onnx/export/layers/README.md)
- [src/architecture/network/onnx/import](../src/architecture/network/onnx/import/README.md)
- [src/architecture/network/onnx/schema](../src/architecture/network/onnx/schema/README.md)
- [src/architecture/network/prune](../src/architecture/network/prune/README.md)
- [src/architecture/network/remove](../src/architecture/network/remove/README.md)
- [src/architecture/network/runtime](../src/architecture/network/runtime/README.md)
- [src/architecture/network/serialize](../src/architecture/network/serialize/README.md)
- [src/architecture/network/slab](../src/architecture/network/slab/README.md)
- [src/architecture/network/standalone](../src/architecture/network/standalone/README.md)
- [src/architecture/network/stats](../src/architecture/network/stats/README.md)
- [src/architecture/network/topology](../src/architecture/network/topology/README.md)
- [src/architecture/network/training](../src/architecture/network/training/README.md)

## Target Order

1. Node boundary
2. Connection boundary
3. Group boundary
4. Architect boundary
5. NodePool ownership follow-through
6. ActivationArrayPool ownership follow-through
7. Layer facade normalization
8. Network facade normalization
9. ONNX root-shim review if public-import cleanup becomes necessary

## Session Log

## Current State

### [DONE] Workstream Closure

- The architecture solid-split workstream is complete enough to close for now.
- The generated [src/architecture/README.md](../src/architecture/README.md)
  now behaves like a root chapter map instead of a flat re-export symbol dump.
- The architecture root compatibility facades remain in place by design for
  public and test stability, but they no longer dominate the generated root
  documentation surface.
- The first docs-first follow-through on the foregrounded
  [src/architecture/network/network.ts](../src/architecture/network/network.ts)
  chapter is now complete: the generated
  [src/architecture/network/README.md](../src/architecture/network/README.md)
  now foregrounds the public `Network` API instead of leading with the class's
  underscore-prefixed runtime state shelf.
- The public `Network` surface now explicitly documents `standalone()`,
  `train()`, and `evolve()` inside the generated chapter so the audit can focus
  on real remaining ownership or docs gaps instead of missing public entrypoints.
- The export-side ONNX implementation now lives under the new
  [src/architecture/network/onnx/export](../src/architecture/network/onnx/export/README.md)
  subchapter so the root ONNX surface can keep its public entry points while the
  implementation stops accumulating in one flat folder.
- The import-side ONNX implementation now also lives under the new
  [src/architecture/network/onnx/import](../src/architecture/network/onnx/import/README.md)
  subchapter so the root ONNX chapter can focus on public entry points and
  shared schema/context instead of owning the whole reconstruction stack.
- The export layer-emission helpers now also live under the new
  [src/architecture/network/onnx/export/layers](../src/architecture/network/onnx/export/layers/README.md)
  subchapter so the export root can focus on orchestration, setup, and
  post-processing while the per-layer emitters stop crowding the same chapter.
- The export-owned ONNX execution and payload types now also live under the
  export chapter so the root types surface no longer needs to own the exporter
  build/setup, heuristic, and layer-emission context cluster.
- The first importer-owned architecture, recurrent-self-connection, and
  pooling-attachment type family now also lives under the import chapter so the
  root types surface no longer owns that orchestrator-only context cluster.
- The importer-owned weight restoration and Conv reconstruction context family
  now also lives under the import chapter so the root types surface no longer
  owns that dense-assignment and Conv replay cluster.
- The importer-owned runtime factory loading and perceptron scaffold context
  family now also lives under the import chapter so the root types surface no
  longer owns that runtime bootstrap cluster.
- The importer-owned fused-recurrent reconstruction family now also lives under
  the import chapter so the root types surface no longer owns that emitted
  recurrent replay cluster.
- The ONNX wire-format schema now also lives under the new
  [src/architecture/network/onnx/schema](../src/architecture/network/onnx/schema/README.md)
  subchapter so the root types surface can stop mixing persisted model shapes
  with runtime/import/export execution contexts.
- The flat compatibility facade at
  [src/architecture/network.ts](../src/architecture/network.ts) still exists by
  design for public and test stability.

### [PLANNED] Deferred Follow-Through

- Reopen this plan only if the repo later decides to remove the flat
  architecture compatibility facades and accept the resulting public-import
  churn across tests, examples, and generated docs.
- Keep any future follow-through audit-first: only reopen a network-owned seam
  if a fresh root review exposes a concrete mismatch that the current chapter
  map no longer explains.
- The ONNX root shared-holdout audit found that the remaining root bridge
  contracts are still genuinely shared across export, import, and root-analysis
  code, so there is no smaller root-shim cleanup worth forcing right now.
- No additional split work is currently justified.

### [DONE] Architecture Root Chapter Map Follow-Through

- Added [src/architecture/docs.order.json](../src/architecture/docs.order.json)
  so the root architecture README can choose a real intro file and stable
  chapter ordering instead of inheriting raw filesystem order.
- Added file-level summaries to the flat root compatibility facades at
  [src/architecture/network.ts](../src/architecture/network.ts),
  [src/architecture/architect.ts](../src/architecture/architect.ts),
  [src/architecture/layer.ts](../src/architecture/layer.ts),
  [src/architecture/group.ts](../src/architecture/group.ts),
  [src/architecture/node.ts](../src/architecture/node.ts),
  [src/architecture/connection.ts](../src/architecture/connection.ts),
  [src/architecture/nodePool.ts](../src/architecture/nodePool.ts),
  [src/architecture/activationArrayPool.ts](../src/architecture/activationArrayPool.ts),
  and [src/architecture/onnx.ts](../src/architecture/onnx.ts) so the generated
  root chapter explains its compatibility role instead of reading like a bare
  symbol shelf.
- This keeps the public import surface stable while making the architecture root
  teach where readers should continue next.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] Architecture Root Audit Conclusion

- The first root chapter-map pass still left the generated architecture README
  dominated by the flat facade files, which exposed a docs-generator limitation
  rather than a missing split seam.
- [src/architecture/docs.order.json](../src/architecture/docs.order.json) now
  hides the flat root compatibility files from the generated architecture root
  README while still using the configured intro summary as the directory-level
  opening.
- [scripts/generate-docs.ts](../scripts/generate-docs.ts) now resolves a
  configured intro file from the full sorted file list rather than only the
  visible file list, which allows hidden compatibility facades to remain the
  source of the root chapter introduction.
- The resulting [src/architecture/README.md](../src/architecture/README.md)
  now opens with the architecture chapter map and no longer floods the root
  page with re-exported API details.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [PLANNED] Remaining Network Split Path

1. Network class seam work is complete enough for audit mode.
2. The public-surface docs cleanup is complete; only reopen class-owned split
  work if the refreshed README still exposes a concrete overloaded seam.

### [PLANNED] Deferred Questions

- Keep or remove the flat public compatibility facade at
  [src/architecture/network.ts](../src/architecture/network.ts).
  > Remove
- Decide later whether broader public-import cleanup across tests, examples,
  and generated docs is worth the churn.
  > Tests will be handled on a dedicated task
- Optional ONNX root-shim review only if the public-import cleanup becomes part
  of the same decision.
  > Proceed
- Decide at the end of the active network workstream whether a separate
  architecture-wide README size and thin-doc audit should become its own plan.
  > Elaborate? We want full documentation 

## Coverage Backlog

Use this section only to avoid re-exploring already-covered boundaries. Each
entry records the minimal extent of completed work and the current stop point.

### [DONE] Architecture Root Coverage

- `node`, `connection`, `group`, and `architect` were folderized into chapter
  entrypoints with thin flat compatibility facades retained.
- `nodePool` and `activationArrayPool` were moved behind chapter ownership with
  local consumer retargets already completed.
- `Layer` ownership moved to
  [src/architecture/layer/layer.ts](../src/architecture/layer/layer.ts), with
  the flat facade intentionally preserved.

### [DONE] Network Ownership Coverage

- `Network` implementation ownership moved into
  [src/architecture/network/network.ts](../src/architecture/network/network.ts).
- Repo-local source consumers were retargeted to the chapter-owned entrypoint.
- Public exports, tests, examples, and generated-doc import examples still rely
  on the flat compatibility path where needed.

### [DONE] Network Internal Coverage

- Constructor/bootstrap seam extracted to
  [src/architecture/network/bootstrap/network.bootstrap.utils.ts](../src/architecture/network/bootstrap/network.bootstrap.utils.ts).
- Topology contract and topology-facing builders were moved onto the topology
  chapter.
- Topology now also owns the hydrated architecture-descriptor fallback through
  [src/architecture/network/topology/network.topology.architecture.utils.ts](../src/architecture/network/topology/network.topology.architecture.utils.ts).
- Public structural node-split flow around `addNodeBetween()` now delegates to
  [src/architecture/network/mutate/network.mutate.public.utils.ts](../src/architecture/network/mutate/network.mutate.public.utils.ts)
  so the class keeps the same behavior while the mutation chapter owns the helper.
- Public clone convenience now delegates to
  [src/architecture/network/serialize/network.serialize.public.utils.ts](../src/architecture/network/serialize/network.serialize.public.utils.ts)
  so the serialize chapter owns the JSON-round-trip cloning contract.
- Main activation orchestration now delegates to the activate chapter instead of
  living inline on the class.
- Runtime configuration and runtime diagnostics were moved onto the runtime
  chapter.
- Training backpropagation and runtime-state clearing now delegate to the
  training chapter.
- Test-time evaluation delegation was moved onto the stats chapter.
- Node removal now relies on the remove chapter's pool-release handling without
  duplicating pool release in the class wrapper.

### [DONE] Validation Baseline

- Completed split passes above were already followed by docs regeneration and
  `npx tsc --noEmit -p tsconfig.json`.
- Re-open a covered seam only if behavior changes, a boundary proves too broad,
  or a later pass needs a narrower follow-through inside the same chapter.

### [DONE] Network Public Surface Docs Follow-Through

- The generated
  [src/architecture/network/README.md](../src/architecture/network/README.md)
  no longer leads with the `Network` class's underscore-prefixed runtime state
  shelf because those implementation-only members are now marked internal in
  [src/architecture/network/network.ts](../src/architecture/network/network.ts).
- The public `Network` chapter now also includes explicit docs for
  `standalone()`, `train()`, and `evolve()` so the README reflects the class's
  real orchestration surface rather than omitting those entrypoints.
- This confirmed that the next durable network step should remain audit-driven
  until a smaller class-owned seam proves that another split is warranted.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] Network Docs And Size Audit Conclusion

- Re-reading the generated
  [src/architecture/network/README.md](../src/architecture/network/README.md)
  after the public-surface cleanup confirmed that the root chapter now teaches
  the `Network` API as an orchestration surface rather than exposing internal
  runtime state first.
- Inspecting
  [src/architecture/network/network.ts](../src/architecture/network/network.ts)
  against the nearest owning subchapters under
  [src/architecture/network/standalone](../src/architecture/network/standalone/README.md),
  [src/architecture/network/slab](../src/architecture/network/slab/README.md),
  and [src/architecture/network/training](../src/architecture/network/training/README.md)
  showed that the remaining root methods are now thin compatibility wrappers or
  tiny convenience helpers, not a stranded class-owned subsystem.
- The only meaningful inline survivors are small API glue such as `set(...)`
  and `adjustRateForAccumulation(...)`; forcing those into another chapter would
  add indirection without improving ownership clarity.
- This closes the network docs and size audit for now and moves the active
  frontier to the architecture-root follow-through audit.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] ONNX Import Fused-Recurrent Types Split

- The importer-owned fused-recurrent reconstruction contracts now live under
  [src/architecture/network/onnx/import/network.onnx.import-fused-recurrent.types.ts](../src/architecture/network/onnx/import/network.onnx.import-fused-recurrent.types.ts)
  so the root compatibility barrel no longer owns the emitted LSTM/GRU replay
  context family.
-
  [src/architecture/network/onnx/import/network.onnx.import-fused-recurrent.utils.ts](../src/architecture/network/onnx/import/network.onnx.import-fused-recurrent.utils.ts)
  now reads those importer-only contracts from the local import chapter while
  the root file keeps only the shared `NodeInternals` and `OnnxLayerFactory`
  bridge types.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`; generated import docs now include the new fused-recurrent
  type chapter.

### [DONE] ONNX Import README Intro Follow-Through

- The generated
  [src/architecture/network/onnx/import/README.md](../src/architecture/network/onnx/import/README.md)
  now opens with the import pipeline story instead of inheriting its chapter
  introduction from the runtime-factory leaf types file.
-
  [src/architecture/network/onnx/import/docs.order.json](../src/architecture/network/onnx/import/docs.order.json)
  now pins the import flow file as the intro source and keeps the generated
  reading order aligned to the actual reconstruction pipeline.
-
  [src/architecture/network/onnx/import/network.onnx.import-flow.utils.ts](../src/architecture/network/onnx/import/network.onnx.import-flow.utils.ts)
  now explains the staged restore questions that link the neighboring runtime,
  weight, activation, orchestration, and fused-recurrent chapters together.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] ONNX Root README Chapter Map Follow-Through

- The generated
  [src/architecture/network/onnx/README.md](../src/architecture/network/onnx/README.md)
  now opens with an explicit chapter map that tells readers when to continue
  into the `export/`, `import/`, and `schema/` subchapters versus when to use
  the root compatibility barrels.
-
  [src/architecture/network/onnx/docs.order.json](../src/architecture/network/onnx/docs.order.json)
  now pins the root ONNX reading order so the public entrypoint stays first and
  the thinner root utility barrel appears before the larger root types barrel.
-
  [src/architecture/network/onnx/network.onnx.ts](../src/architecture/network/onnx/network.onnx.ts)
  now explains why the root chapter exists, how the folder is split, and how a
  contributor should navigate the remaining compatibility surfaces.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] ONNX Root Compatibility Barrel Docs Follow-Through

- The generated
  [src/architecture/network/onnx/README.md](../src/architecture/network/onnx/README.md)
  now explains both
  [src/architecture/network/onnx/network.onnx.utils.ts](../src/architecture/network/onnx/network.onnx.utils.ts)
  and
  [src/architecture/network/onnx/network.onnx.utils.types.ts](../src/architecture/network/onnx/network.onnx.utils.types.ts)
  as intentional compatibility barrels instead of leaving them to read like
  leftover mixed shelves after the export/import/schema splits.
- The root execution barrel now explains how to read its forwarding role,
  what still belongs there, and when a reader should continue into the split
  `export/` and `import/` chapters.
- The root types barrel now explains how to read the remaining shared bridge
  layer, which type families already moved into chapter-local ownership, and
  why a small root compatibility surface still remains.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] ONNX Root Shared-Holdout Audit

- The remaining root-owned ONNX bridge contracts were rechecked against current
  consumers and still span multiple chapters, so a final root-shim cleanup is
  not justified yet.
- `NodeInternals` and `ActivationFunction` still bridge root layer-analysis
  helpers plus both export and import execution paths.
- `OnnxConvKernelCoordinate` still bridges export-side Conv emission and
  import-side Conv reconstruction.
- `OnnxLayerFactory` and `OnnxRuntimeLayerFactoryMap` still bridge runtime-load
  wiring into fused-recurrent reconstruction.
- The root execution barrel's `buildOnnxModel()` wrapper still acts as an
  intentional stable orchestration surface above the split export
  implementation.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] Network README Reading Order Follow-Through

- The generated
  [src/architecture/network/README.md](../src/architecture/network/README.md)
  now starts with
  [src/architecture/network/network.ts](../src/architecture/network/network.ts)
  instead of dropping directly into the much larger
  [src/architecture/network/network.types.ts](../src/architecture/network/network.types.ts)
  shelf.
-
  [src/architecture/network/docs.order.json](../src/architecture/network/docs.order.json)
  now pins the chapter intro to the public `Network` class and keeps the
  reading order aligned to public orchestration first, compatibility utilities
  second, and the large root types shelf last.
- This keeps the broader network docs/size audit in documentation-first mode
  without forcing a new split before the reordered chapter has been evaluated.
- Validation completed with `npx tsc --noEmit -p tsconfig.json` and
  `npm run docs`.

### [DONE] ONNX Export Subchapter Split

- The export-side ONNX implementation was folderized under
  [src/architecture/network/onnx/export](../src/architecture/network/onnx/export/README.md)
  so the root ONNX chapter keeps the public entry points while the export
  implementation gets its own durable chapter boundary.
- The ONNX compatibility barrel now re-exports the export helpers from the new
  subfolder without changing the public `exportToONNX()` entry point.
- The broader ONNX chapter still needs a refreshed README audit after docs run
  to determine whether import-side helpers or shared schema/types are the next
  split candidate.

### [DONE] ONNX Import Subchapter Split

- The import-side ONNX implementation was folderized under
  [src/architecture/network/onnx/import](../src/architecture/network/onnx/import/README.md)
  so the root ONNX chapter keeps the public `importFromONNX()` entry point while
  the reconstruction helpers get their own durable chapter boundary.
- The ONNX compatibility barrel now re-exports the import helpers from the new
  subfolder without changing the public `importFromONNX()` entry point.
- The broader ONNX chapter still needs a refreshed README audit after docs run
  to determine whether shared schema/types or remaining root-shim helpers are
  the next split candidate.

### [DONE] ONNX Export Layers Subchapter Split

- The export layer-emission helpers were folderized under
  [src/architecture/network/onnx/export/layers](../src/architecture/network/onnx/export/layers/README.md)
  so the export root can keep orchestration, setup, and post-processing while
  the Conv, dense, recurrent, and shared layer-emission helpers get a more
  focused durable boundary.
- The ONNX compatibility barrels now re-export the moved layer helpers from the
  nested subfolder without changing the public export entry points.
- The broader ONNX export chapter still needs a refreshed README audit after
  docs run to confirm whether shared schema/types or documentation quality is
  the next durable follow-through step.

### [DONE] ONNX Export Layers Docs Follow-Through

- The new
  [src/architecture/network/onnx/export/layers](../src/architecture/network/onnx/export/layers/README.md)
  chapter now opens with an architecture-level explanation of the dispatch
  boundary instead of dropping straight into symbol listings.
- The layer router plus the main Conv, dense, mixed-activation, recurrent, and
  shared initializer helpers now explain their invariants, tradeoffs, and
  representative usage through source JSDoc so the generated README teaches the
  boundary instead of acting as a thin API index.
- The docs follow-through included a Mermaid decision-flow diagram for the
  layer-routing boundary and was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.

### [DONE] ONNX Schema Subchapter Split

- The leaf ONNX wire-format schema was folderized under
  [src/architecture/network/onnx/schema](../src/architecture/network/onnx/schema/README.md)
  so persisted model shapes, tensor payloads, metadata records, and Conv/Pool
  mapping declarations no longer live inside the mixed root types file.
- Internal ONNX import/export consumers now read schema symbols directly from
  the new schema chapter while
  [src/architecture/network/onnx/network.onnx.utils.types.ts](../src/architecture/network/onnx/network.onnx.utils.types.ts)
  remains a thin compatibility barrel for the transition.
- The next type-oriented ONNX split can now target exporter or importer
  execution contexts without re-mixing the persisted wire schema into those
  runtime-specific boundaries.

### [DONE] ONNX Schema Docs Follow-Through

- The new
  [src/architecture/network/onnx/schema](../src/architecture/network/onnx/schema/README.md)
  chapter now opens with a clear wire-format boundary description and example
  instead of reading like an unstructured type dump.
- The core schema container types now explain graph sections, initializer
  storage, named-tensor node wiring, and the simplified attribute payload shape
  through source JSDoc so the generated README teaches the persisted document
  model.
- The docs follow-through was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.

### [DONE] ONNX Export Types Split

- The exporter-owned build/setup, heuristic, recurrent, Conv, and layer-emission
  context types were moved into
  [src/architecture/network/onnx/export/network.onnx.export.types.ts](../src/architecture/network/onnx/export/network.onnx.export.types.ts)
  so the root types file no longer mixes that large chapter-local cluster with
  importer and shared runtime bridge types.
- Export-side consumers now import those chapter-local types directly from the
  export chapter while
  [src/architecture/network/onnx/network.onnx.utils.types.ts](../src/architecture/network/onnx/network.onnx.utils.types.ts)
  remains a thin compatibility barrel for public ergonomics and shared holdouts.
- Shared bridge types such as `NodeInternals`, `NodeInternalsWithExportIndex`,
  and `OnnxConvKernelCoordinate` intentionally remain root-owned for now to
  avoid cross-coupling importer code back into the export chapter.

### [DONE] ONNX Export Types Docs Follow-Through

- The export chapter now opens with an architecture-level map of how export
  options flow into setup/build contexts, heuristics, and layer-emission
  payloads instead of introducing the new file as a bare type list.
- The docs follow-through was kept local to
  [src/architecture/network/onnx/export/network.onnx.export.types.ts](../src/architecture/network/onnx/export/network.onnx.export.types.ts)
  so the generated export README explains the boundary shift without reopening a
  broader docs rewrite.
- The docs follow-through was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.

### [DONE] ONNX Import Orchestrator Types Split

- The importer-owned architecture extraction, recurrent self-connection, and
  pooling attachment type family was moved into
  [src/architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts](../src/architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts)
  so the root types file no longer owns that import-local orchestration state.
- The orchestrator utilities now read those chapter-local types directly from
  the import chapter while
  [src/architecture/network/onnx/network.onnx.utils.types.ts](../src/architecture/network/onnx/network.onnx.utils.types.ts)
  remains the thin compatibility barrel for shared holdouts and transitional
  public ergonomics.
- Shared bridge types such as `NodeInternals`, `ActivationFunction`, and
  `OnnxConvKernelCoordinate` intentionally remain root-owned for now because
  they still bridge import, export, or root analysis helpers beyond this first
  importer-only seam.
- The split was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.

### [DONE] ONNX Import Weight Types Split

- The importer-owned hidden-size derivation, dense and per-neuron assignment,
  and optional Conv reconstruction type family was moved into
  [src/architecture/network/onnx/import/network.onnx.import-weights.types.ts](../src/architecture/network/onnx/import/network.onnx.import-weights.types.ts)
  so the root types file no longer owns that import-local restoration state.
- The weight reconstruction utilities now read those chapter-local types
  directly from the import chapter while
  [src/architecture/network/onnx/network.onnx.utils.types.ts](../src/architecture/network/onnx/network.onnx.utils.types.ts)
  remains the thin compatibility barrel for shared holdouts and transitional
  public ergonomics.
- The follow-through stayed local to the new import weights types file so the
  generated import README now explains the weight restoration boundary as its
  own chapter instead of extending the mixed root type list.
- Shared bridge types such as `NodeInternals` and
  `OnnxConvKernelCoordinate` intentionally remain root-owned because they still
  bridge import code back into the root analysis or export surfaces.
- The split was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.

### [DONE] ONNX Import Runtime-Load Types Split

- The importer-owned runtime factory loading and perceptron scaffold type
  family was moved into
  [src/architecture/network/onnx/import/network.onnx.runtime-load.types.ts](../src/architecture/network/onnx/import/network.onnx.runtime-load.types.ts)
  so the root types file no longer owns that import-local runtime bootstrap
  state.
- The runtime-load utilities now read those chapter-local types directly from
  the import chapter while
  [src/architecture/network/onnx/network.onnx.utils.types.ts](../src/architecture/network/onnx/network.onnx.utils.types.ts)
  remains the thin compatibility barrel for shared holdouts and transitional
  public ergonomics.
- The follow-through stayed local to the new runtime-load types file so the
  generated import README now teaches the runtime bootstrap contract as its own
  import chapter surface.
- Shared bridge types such as `OnnxLayerFactory` and
  `OnnxRuntimeLayerFactoryMap` intentionally remain root-owned because they
  still bridge runtime bootstrapping into the broader fused-recurrent
  reconstruction surface.
- The split was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.

### [DONE] Remaining Inline Network Surface Review

- `activate()`, `describeArchitecture()`, `propagate()`, and `clear()` no
  longer carry their main ownership logic inline on the class.
- `remove()` now matches the remove chapter's ownership and no longer repeats
  pooled-node release at the class layer.
- The remaining inline methods on
  [src/architecture/network/network.ts](../src/architecture/network/network.ts)
  are now mostly thin public delegates, static compatibility helpers, or small
  public convenience methods whose current size does not justify another split
  without new evidence from the docs or boundary audit.

### [PLANNED] Post-Split Audit Placeholders

- Oversized generated README audit candidates discovered from the refreshed docs:
  - [src/architecture/network/onnx/README.md](../src/architecture/network/onnx/README.md)
    now about 1761 lines after the export/import/layers/schema/export-types splits
  - [src/architecture/network/onnx/export/README.md](../src/architecture/network/onnx/export/README.md)
    now about 1691 lines after the export-side, layers, and export-types splits
  - [src/architecture/network/onnx/export/layers/README.md](../src/architecture/network/onnx/export/layers/README.md)
    now about 970 lines after the export layers split
  - [src/architecture/network/onnx/import/README.md](../src/architecture/network/onnx/import/README.md)
    now about 1240 lines after the import-side split
  - [src/architecture/network/onnx/schema/README.md](../src/architecture/network/onnx/schema/README.md)
    now about 120 lines after the schema split
  - [src/architecture/network/README.md](../src/architecture/network/README.md)
    at about 2351 lines
  - [src/architecture/README.md](../src/architecture/README.md)
    at about 1757 lines
  - [src/architecture/network/mutate/README.md](../src/architecture/network/mutate/README.md)
    at about 1412 lines
  - [src/architecture/network/activate/README.md](../src/architecture/network/activate/README.md)
    at about 1192 lines
  - [src/architecture/network/slab/README.md](../src/architecture/network/slab/README.md)
    at about 1093 lines
  - [src/architecture/network/serialize/README.md](../src/architecture/network/serialize/README.md)
    at about 1044 lines
- Thin generated README audit candidates discovered from the refreshed docs:
  - [src/architecture/activationArrayPool/README.md](../src/architecture/activationArrayPool/README.md)
    at about 24 lines
  - [src/architecture/nodePool/README.md](../src/architecture/nodePool/README.md)
    at about 51 lines
- Add more placeholders here if later split passes discover another overloaded
  helper chapter or a README whose size/quality indicates additional follow-up.

## [DONE] Compact Pass Log

- Root primitives and builders: folderized and stabilized behind temporary flat
  facades.
- Pool ownership follow-through: chapter ownership complete.
- Layer normalization: chapter ownership complete.
- Network normalization: chapter ownership complete; flat facade intentionally
  retained.
- Network internal passes completed so far:
  - bootstrap
  - topology contract and builder delegation
  - descriptor ownership and hydrated fallback delegation
  - structural mutation delegation for `addNodeBetween()`
  - clone delegation through serialize chapter helpers
  - activation orchestration delegation
  - runtime configuration
  - runtime diagnostics
  - training lifecycle delegation
  - remove wrapper pooling follow-through
  - test-time evaluation delegation
- Remaining planned high-level passes:
  - none

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Workstream: architecture solid split
Plan: plans/architecture-solid-split.plans.md
Status: [DONE]
Active boundary: none

Already covered:
- architecture root primitives and builders were folderized and stabilized behind intentional compatibility facades where still required
- Network ownership moved into src/architecture/network/network.ts
- bootstrap, topology contract, structural mutation delegation, clone delegation, runtime configuration, runtime diagnostics, and test-time evaluation have already been extracted or delegated into their chapter-owned boundaries
- ONNX export implementation has been folderized under src/architecture/network/onnx/export while the public ONNX entry points remain stable
- ONNX import implementation has been folderized under src/architecture/network/onnx/import while the public ONNX entry points remain stable
- the ONNX root README now has an explicit chapter map and docs ordering so the root chapter teaches how to navigate export/import/schema versus the root compatibility barrels
- the root ONNX compatibility utility and types barrels now explain their bridge role directly in the generated root README
- the ONNX root shared-holdout audit found that the remaining root bridge contracts are still genuinely cross-chapter and should stay put for now
- the broader network README now opens on the public Network class instead of the large root types shelf
- the Network public-surface docs pass is now complete: underscore-prefixed runtime members are hidden from generated docs, and standalone/train/evolve now appear in the generated network chapter
- the network docs and size audit is now complete: the remaining root Network methods are thin compatibility wrappers or tiny convenience helpers, so no additional class-owned split is justified right now
- the architecture root now has an explicit docs-order and compatibility-facade chapter map, and the generated root README now keeps that intro while hiding the flat root re-export shelf

High-level remaining path:
- none

Next task if this workstream is reopened:
- decide whether the flat architecture compatibility facades should actually be removed rather than merely documented
- scope the public-import churn across tests, examples, and generated docs before making that change
- leave the root ONNX shared bridge contracts in place unless a future public-import cleanup proves otherwise

Required workflow:
1. Read src/architecture/README.md first.
2. Read plans/architecture-solid-split.plans.md.
3. Inspect src/architecture and the specific flat compatibility facade or bridge surface being reconsidered.
4. Complete only one durable step.
5. Update this plan using [PLANNED], [WIP], and [DONE] markers and keep older coverage compressed.
6. Refresh docs and run: npm run docs
7. Validate with: npx tsc --noEmit -p tsconfig.json

Constraints:
- keep the public compatibility surface stable
- do not reopen the flat network facade decision unless the reopened task explicitly targets public-import cleanup
- do not re-expand completed history in the plan
```
