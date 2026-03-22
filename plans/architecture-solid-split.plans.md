# Architecture Solid Split Plan

**Status:** [WIP]

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
- [src/architecture/network/connect](../src/architecture/network/connect/README.md)
- [src/architecture/network/deterministic](../src/architecture/network/deterministic/README.md)
- [src/architecture/network/evolve](../src/architecture/network/evolve/README.md)
- [src/architecture/network/gating](../src/architecture/network/gating/README.md)
- [src/architecture/network/genetic](../src/architecture/network/genetic/README.md)
- [src/architecture/network/mutate](../src/architecture/network/mutate/README.md)
- [src/architecture/network/onnx](../src/architecture/network/onnx/README.md)
- [src/architecture/network/prune](../src/architecture/network/prune/README.md)
- [src/architecture/network/remove](../src/architecture/network/remove/README.md)
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
7. Layer facade normalization if still needed
8. Network facade normalization if still needed
9. ONNX root-shim review if public-import cleanup becomes necessary

## Session Log

### Architecture planning pass

Goals:

- Create a durable split order for the architecture root instead of attempting
  a one-shot rewrite.
- Align the split order with the Phase 2 primitives roadmap while treating the
  current work as Phase 0 structural cleanup.

Progress:

- Built the full README inventory under
  [src/architecture](../src/architecture/README.md), including the nested
  network chapter folders that already form the strongest small-chapter model in
  this root.
- Mapped the remaining unsplit flat files and chose
  [src/architecture/node.ts](../src/architecture/node.ts) as the first durable
  boundary because it is the densest remaining primitive and sits underneath
  group, layer, architect, and network consumers.
- Chose a node-first sequence so later connection, group, and architect passes
  can target a stable primitive boundary instead of another flat monolith.

Decision:

- Treat the existing `network/` and `layer/` folders as the style reference for
  the rest of the root instead of reopening them first.
- Keep each pass focused on one root boundary plus its mandatory educational
  docs follow-up.

Next step:

- Complete the node boundary pass by moving
  [src/architecture/node.ts](../src/architecture/node.ts) into its own chapter
  folder, then regenerate docs and validate the touched surface.

### Node boundary pass

Goals:

- Move the core node primitive out of the flat architecture root and into its
  own chapter folder.
- Keep existing import surfaces stable for this first pass while shifting the
  educational center of gravity into the new folder.

Progress:

- Moved the implementation from
  [src/architecture/node.ts](../src/architecture/node.ts) to
  [src/architecture/node/node.ts](../src/architecture/node/node.ts) so the
  node primitive now has a dedicated chapter boundary.
- Replaced the old flat file with a thin compatibility facade at
  [src/architecture/node.ts](../src/architecture/node.ts) because the current
  repo still has broad internal and public exports that depend on the flat
  path.
- Added folder-level introductory JSDoc in
  [src/architecture/node/node.ts](../src/architecture/node/node.ts) and ran the
  required docs refresh plus `npx tsc --noEmit -p tsconfig.json` so the new
  node chapter becomes the richer generated README surface without regressing
  the architecture root.

Decision:

- Keep the node facade temporarily while the larger architecture split is still
  migrating broad repo-local and public imports.
- Use the same temporary-facade pattern for the next primitive boundaries only
  when the import graph is comparably wide.

Next step:

- Move to the connection boundary next so the two lowest-level graph primitives
  share the same folderized shape before group and architect are split.

### Connection boundary pass

Goals:

- Move the graph edge primitive out of the flat architecture root and into its
  own chapter folder.
- Start direct-path migration for nearby repo-local imports while leaving a thin
  root facade in place because the public and test import graph is still broad.

Progress:

- Moved the implementation from
  [src/architecture/connection.ts](../src/architecture/connection.ts) to
  [src/architecture/connection/connection.ts](../src/architecture/connection/connection.ts)
  so the second low-level graph primitive now follows the same chapter shape as
  node.
- Replaced the old flat file with a thin compatibility facade at
  [src/architecture/connection.ts](../src/architecture/connection.ts) while the
  repo still has many public and test imports at the flat path.
- Updated the closest architecture-local imports to use the direct chapter path
  so future boundary passes depend less on the flat root shelf.
- Added richer chapter-opening JSDoc in
  [src/architecture/connection/connection.ts](../src/architecture/connection/connection.ts)
  so the generated README teaches why the edge primitive uses lazy fields,
  pooling, innovation helpers, and virtualized accessors, then refreshed docs
  and reran `npx tsc --noEmit -p tsconfig.json` for the moved boundary.

Decision:

- Keep the temporary root facade for connection during this pass because the
  current import graph still spans public exports and many tests, making a full
  direct-path cutover noisier than the durable boundary move itself.
- Treat group as the next boundary now that node and connection share the same
  folderized primitive shape.

Next step:

- Move to the group boundary next so the first composite architecture primitive
  can build on folderized node and connection chapters instead of the remaining
  flat root shelf.

### Group boundary pass

Goals:

- Move the first composite architecture primitive out of the flat root and into
  its own chapter folder.
- Start direct-path migration for the closest architecture-local consumers while
  keeping a thin root facade because the public export surface and layer builder
  graph still depend on `src/architecture/group.ts`.

Progress:

- Moved the implementation from
  [src/architecture/group.ts](../src/architecture/group.ts) to
  [src/architecture/group/group.ts](../src/architecture/group/group.ts) so the
  first composite primitive now matches the small-chapter shape used by node
  and connection.
- Replaced the old flat file with a thin compatibility facade at
  [src/architecture/group.ts](../src/architecture/group.ts) because the current
  public export and layer-heavy import graph still reaches the flat path.
- Updated the nearest architecture-local imports to use the direct chapter path
  so layer helpers and architect orchestration now depend less on the flat root
  shelf.
- Added chapter-opening JSDoc in
  [src/architecture/group/group.ts](../src/architecture/group/group.ts) so the
  generated README teaches how groups bridge node-level primitives and larger
  architecture builders, then refreshed docs and reran
  `npx tsc --noEmit -p tsconfig.json` for the moved chapter.

Decision:

- Keep the temporary root facade for group during this pass because the import
  graph still spans the public `neataptic.ts` export and several layer helper
  modules that use `Group` as a first-class runtime type.
- Treat architect as the next boundary now that node, connection, and group all
  follow the same folderized primitive shape.

Next step:

- Move to the architect boundary next so the top-level builder entrypoint can
  depend on folderized node, connection, and group chapters instead of the
  remaining flat root shelf.

### Architect boundary pass

Goals:

- Move the top-level builder entrypoint out of the flat root and into its own
  chapter folder.
- Keep the public and test import surface stable with a thin root facade while
  shifting the implementation to direct chapter-path dependencies.

Progress:

- Moved the implementation from
  [src/architecture/architect.ts](../src/architecture/architect.ts) to
  [src/architecture/architect/architect.ts](../src/architecture/architect/architect.ts)
  so the builder entrypoint now matches the chapter shape used by node,
  connection, and group.
- Replaced the old flat file with a thin compatibility facade at
  [src/architecture/architect.ts](../src/architecture/architect.ts) because the
  current import graph still includes the public
  [src/neataptic.ts](../src/neataptic.ts) export plus broad test coverage at the
  flat path.
- Updated the moved implementation to depend on the folderized node,
  connection, and group chapters directly while continuing to use the existing
  layer and network entry surfaces that already anchor their larger folderized
  subsystems.
- Added chapter-opening JSDoc in
  [src/architecture/architect/architect.ts](../src/architecture/architect/architect.ts)
  so the generated README teaches how the builder entrypoint turns low-level
  graph primitives into named network presets, then refreshed docs and reran
  `npx tsc --noEmit -p tsconfig.json` for the new boundary.

Decision:

- Keep the temporary root facade for architect during this pass because the
  builder entrypoint is still part of the public `neataptic.ts` export surface
  and appears throughout the test suite.
- Treat NodePool ownership follow-through as the next step now that the main
  architecture root entrypoints all follow the same folderized chapter shape.

Next step:

- Move to the NodePool ownership follow-through so the remaining root-level
  ownership story keeps converging on the chapter-based architecture surface.

### NodePool ownership follow-through

Goals:

- Move the node lifecycle pool into its own chapter folder so ownership lives
  beside the node chapter instead of on the flat root shelf.
- Keep the flat `src/architecture/nodePool.ts` import path stable while
  retargeting the nearest architecture-local consumers to the direct chapter
  implementation.

Progress:

- Moved the implementation from
  [src/architecture/nodePool.ts](../src/architecture/nodePool.ts) to
  [src/architecture/nodePool/nodePool.ts](../src/architecture/nodePool/nodePool.ts)
  so the node recycling policy now has a dedicated chapter boundary instead of
  living as another flat root-owned helper.
- Replaced the old flat file with a thin compatibility facade at
  [src/architecture/nodePool.ts](../src/architecture/nodePool.ts) so repo-local
  utilities, benchmarks, and tests can keep the old import path during this
  follow-through step.
- Updated the moved implementation to depend directly on
  [src/architecture/node/node.ts](../src/architecture/node/node.ts) instead of
  bouncing back through the root node shim.
- Updated the nearest architecture-local consumers in
  [src/architecture/network.ts](../src/architecture/network.ts) and
  [src/architecture/network/remove/network.remove.finalize.utils.ts](../src/architecture/network/remove/network.remove.finalize.utils.ts)
  to use the direct chapter path so runtime hot paths depend less on the flat
  root shelf.
- Added chapter-opening JSDoc in
  [src/architecture/nodePool/nodePool.ts](../src/architecture/nodePool/nodePool.ts)
  so the generated README teaches the difference between node identity,
  lifecycle reuse, and pool observability, then refreshed docs and reran
  `npx tsc --noEmit -p tsconfig.json` for the moved ownership surface.

Decision:

- Keep the temporary root facade for NodePool during this pass because the
  remaining import graph still includes utilities, benchmarks, and test
  coverage at the flat path.
- Treat ActivationArrayPool ownership follow-through as the next step because
  it is the remaining flat root-owned pooling surface in the same lifecycle
  family.

Next step:

- Move to the ActivationArrayPool ownership follow-through so the remaining
  memory-management helpers converge on the chapter-based architecture surface.

### ActivationArrayPool ownership follow-through

Goals:

- Move the activation-buffer pool into its own chapter folder so ownership sits
  with the runtime memory-policy story instead of the flat root shelf.
- Keep the flat `src/architecture/activationArrayPool.ts` import path stable
  while retargeting the architecture-local runtime consumers to the direct
  chapter implementation.

Progress:

- Moved the implementation from
  [src/architecture/activationArrayPool.ts](../src/architecture/activationArrayPool.ts)
  to
  [src/architecture/activationArrayPool/activationArrayPool.ts](../src/architecture/activationArrayPool/activationArrayPool.ts)
  so reusable output-buffer ownership now has its own chapter boundary.
- Replaced the old flat file with a thin compatibility facade at
  [src/architecture/activationArrayPool.ts](../src/architecture/activationArrayPool.ts)
  so existing tests can keep the flat import path during this follow-through
  step.
- Updated the nearest architecture-local consumers in
  [src/architecture/network.ts](../src/architecture/network.ts),
  [src/architecture/layer/layer.activation.utils.ts](../src/architecture/layer/layer.activation.utils.ts),
  [src/architecture/network/activate/network.activate.core.utils.ts](../src/architecture/network/activate/network.activate.core.utils.ts),
  [src/architecture/network/activate/network.activate.utils.types.ts](../src/architecture/network/activate/network.activate.utils.types.ts),
  [src/architecture/network/activate/network.activate.notrace.utils.ts](../src/architecture/network/activate/network.activate.notrace.utils.ts),
  and
  [src/architecture/network/slab/network.slab.fast-path.helpers.utils.ts](../src/architecture/network/slab/network.slab.fast-path.helpers.utils.ts)
  so hot runtime paths depend less on the flat root shelf.
- Added chapter-opening JSDoc in
  [src/architecture/activationArrayPool/activationArrayPool.ts](../src/architecture/activationArrayPool/activationArrayPool.ts)
  so the generated README teaches buffer reuse, capacity control, and prewarm
  semantics, then refreshed docs and reran `npx tsc --noEmit -p tsconfig.json`
  for the moved ownership surface.

Decision:

- Keep the temporary root facade for ActivationArrayPool during this pass
  because the import graph still includes dedicated tests at the flat path.
- Treat layer and network facade normalization as the next review point now
  that the remaining flat root-owned helper chapters in this lifecycle family
  have been moved.

Next step:

- Review whether layer facade normalization is still needed now that the root
  pooling helpers have converged on chapter-owned implementations.
