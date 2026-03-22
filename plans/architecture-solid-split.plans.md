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
  required docs refresh so the new node chapter becomes the richer generated
  README surface.

Decision:
- Keep the node facade temporarily while the larger architecture split is still
  migrating broad repo-local and public imports.
- Use the same temporary-facade pattern for the next primitive boundaries only
  when the import graph is comparably wide.

Validation:
- `npm run docs`
- `npx tsc --noEmit -p tsconfig.json`

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
  pooling, innovation helpers, and virtualized accessors.

Decision:
- Keep the temporary root facade for connection during this pass because the
  current import graph still spans public exports and many tests, making a full
  direct-path cutover noisier than the durable boundary move itself.
- Treat group as the next boundary now that node and connection share the same
  folderized primitive shape.

Validation:
- `npm run docs`
- `npx tsc --noEmit -p tsconfig.json`

Next step:
- Move to the group boundary next so the first composite architecture primitive
  can build on folderized node and connection chapters instead of the remaining
  flat root shelf.