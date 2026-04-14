# Architecture Primitives (Node / Group / Layer) Plan

**Status:** [DONE]

## Scope

- Harden the existing `Node`, `Group`, and `Layer` surfaces into a clearer Phase 2 architecture-building API.
- Make role ownership and primitive construction explicit by default without introducing a parallel graph-model API.
- Keep full graph compilation and validation in [Construct_From_Parts_Graph_Assembly.md](Construct_From_Parts_Graph_Assembly.md); this plan owns only the primitive surfaces that plan consumes.
- Preserve current public imports and compatibility facades under `src/architecture/*.ts`.

## Final state

- The workstream reused the existing public `Node`, `Group`, and `Layer` classes instead of inventing a parallel graph vocabulary.
- Role-aware primitive construction landed through `new Group(size, role)` and `Layer.dense(size, role)`, while backward-compatible defaults still resolve to `hidden`.
- `Architect` preset builders that know input/output intent up front now use construction-time role assignment instead of retrofitting roles with `set({ type: ... })`.
- `Node`, `Group`, and `Layer` now expose additive `describe({ label, intent, metadata })` descriptors, and the runtime-supported layer families stamp lightweight default intent and family metadata without changing execution semantics.
- The generated architecture chapter surfaces now teach the role-versus-descriptor split directly in the `node`, `group`, and `layer` READMEs through source-mapped JSDoc examples.
- Validation across the completed slices remained green on targeted primitive and architect tests plus `npm run build`, `npm run test:silent`, and `npm run docs`.
- Whole-graph compile and validation ownership remains intentionally deferred to [Construct_From_Parts_Graph_Assembly.md](Construct_From_Parts_Graph_Assembly.md).

## Audit summary

- The plan corrected a stale assumption that Phase 2 needed new `GraphNode` or `NodeGroup` classes; the real gap was primitive DX on the existing public surface.
- Primitive improvements stayed additive: current `set(...)`, `connect(...)`, `gate(...)`, and factory flows remain valid while the preferred path is now explicit role assignment first and optional descriptors second.
- The metadata seam stayed intentionally lightweight so later diagnostics, visualization, and construct-from-parts work can recover intent without pulling this plan into `Network` materialization.
- The documentation pass closed the educational gap by showing when to use roles, when to attach descriptors, and how the built-in layer families already provide default family metadata.

## Reopen conditions

- Future work needs primitive descriptor semantics beyond the current `label`, `intent`, and scalar `metadata` surface.
- Visualization or diagnostics work needs richer primitive rendering that cannot be handled by the current lightweight boundary metadata.
- Construct-from-parts work uncovers a real primitive-surface gap that cannot be solved at the whole-graph layer alone.
- A later refactor changes the stable ownership or compatibility-facade boundary among `Node`, `Group`, and `Layer`.

## Audit log

- Durable completion notes now live in [Architecture_Primitives_Node_Group_Layer.logs.md](Architecture_Primitives_Node_Group_Layer.logs.md).

