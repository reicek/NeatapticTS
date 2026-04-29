# Architecture Primitives (Node / Group / Layer) Log

**Status:** [DONE]

## Audit scope

- Objective: close the Phase 2 primitive-hardening lane after role-aware construction, lightweight boundary descriptors, and the educational docs pass all landed on the existing public primitive surface.
- The pass covered owner-boundary correction, additive API work on `Node` / `Group` / `Layer`, generated-doc improvement, and final validation.

## Durable milestones

### [DONE] Primitive owner boundary correction

- Confirmed this lane should harden the existing `Node`, `Group`, and `Layer` classes rather than inventing a parallel `GraphNode` or `NodeGroup` vocabulary.
- Preserved the Phase 2 boundary with [Construct_From_Parts_Graph_Assembly.md](Construct_From_Parts_Graph_Assembly.md): primitive DX belongs here, whole-graph compile and validation do not.
- Kept stable compatibility facades under `src/architecture/node.ts`, `src/architecture/group.ts`, and `src/architecture/layer.ts`.

### [DONE] Role-aware primitive construction

- Added the shared primitive role type and wired it through `Group(size, role)` and `Layer.dense(size, role)` with `hidden` retained as the compatibility default.
- Migrated owner-local `Architect` preset builders to prefer construction-time role assignment when input/output intent is known up front.
- Locked the role-aware behavior in targeted primitive and architect tests.

### [DONE] Lightweight descriptor metadata

- Added additive `describe({ label, intent, metadata })` support to `Node`, `Group`, and `Layer` without changing runtime activation semantics.
- Stamped default family and intent metadata across dense, recurrent, memory, normalization, convolution, and attention layer builders so later tooling can recover block meaning cheaply.
- Kept the descriptor surface intentionally scalar and advisory so it remains safe for later diagnostics and construct-from-parts work.

### [DONE] Educational docs and generated README alignment

- Refreshed source JSDoc in `src/architecture/node/node.ts`, `src/architecture/group/group.ts`, and `src/architecture/layer/layer.ts` so the generated chapter surfaces teach role-aware construction and descriptor usage together.
- Improved examples to show the intended usage order: choose the right primitive role first, then attach labels or metadata only when a boundary should stay visible to later readers or tooling.
- Regenerated docs so the `node`, `group`, and `layer` README chapters now reflect the new examples and explanations.

## Controls and evidence

- Runtime-changing slices were validated with targeted primitive and architect tests plus `npm run build`, `npm run test:silent`, and `npm run docs`.
- The closing documentation pass revalidated `npm run build` and `npm run docs` after the source-comment edits so the generated educational surfaces stayed synchronized.

## Reopen triggers

- The primitive surface needs richer descriptor semantics or non-scalar metadata.
- A future visualization or diagnostics lane needs primitive-level rendering behavior that the current descriptor API cannot express.
- Whole-graph assembly work discovers a primitive contract gap that should be solved before or alongside [Construct_From_Parts_Graph_Assembly.md](Construct_From_Parts_Graph_Assembly.md).
