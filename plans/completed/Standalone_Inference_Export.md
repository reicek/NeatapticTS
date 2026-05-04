# Standalone Inference Export Plan (Dependency-Free Runtime)

**Status:** [DONE]

## Scope

- Freeze the Phase 4 standalone export architecture so later implementation can proceed without reopening core decisions.
- Record the completed `network.standalone()` hardening as the parity baseline, not as the finished public export API.
- Keep [plans/Roadmap.md](../Roadmap.md) authoritative: Phase 2 remains the active roadmap stage, so this file is the closed planning baseline for later Phase 4 work.

## Final state

- Planning for the standalone deployment lane is complete for the current roadmap scope.
- The schedule-aware `network.standalone()` groundwork is recorded as done: explicit input/output role ordering and compiled activation traversal are the export parity baseline.
- The implementation boundary is frozen: keep `src/architecture/network/standalone/` as the raw activator oracle and introduce `src/architecture/network/export/` when Phase 4 implementation begins.
- The public export direction is frozen around an IR-first design: versioned `InferenceIRv1`, one shared kernel renderer, `esm` as the canonical output with thin `cjs` and `iife` wrappers, recurrent `reset()` only when `mode === 'recurrent'`, and named export errors for unsupported runtime features.
- This tracker is now a reopen point for implementation. The deployment-facing export API is not yet shipped.

## Audit summary

- Cross-plan alignment is frozen against [plans/Roadmap.md](../Roadmap.md), [plans/completed/Stable_Activation_Ordering_and_Explicit_IO_Roles.md](Stable_Activation_Ordering_and_Explicit_IO_Roles.md), [plans/Worker_Friendly_Network_Serialization_Fastpath.md](../Worker_Friendly_Network_Serialization_Fastpath.md), and [plans/Evolution_Training_Interoperability_Contracts.md](../Evolution_Training_Interoperability_Contracts.md).
- The completed groundwork is captured as prerequisite coverage instead of being misreported as full export completion.
- The first implementation reopen step is stable: create `src/architecture/network/export/`, freeze `InferenceIRv1`, then build the shared renderer and public facade around that IR.
- The underlying hardening pass previously validated focused standalone parity plus `npm run build`, `npm run test:silent`, and `npm run docs`; this tracker-closure pass did not rerun runtime validations because it only changed planning artifacts.

## Reopen conditions

- Phase 4 standalone export becomes an active implementation priority.
- Worker serialization or interoperability work changes the agreed inference IR seam or public export contract.
- The runtime parity baseline changes and export must absorb new forward-pass state or scheduling semantics.
- A public artifact or format requirement moves beyond the frozen `esm`/`cjs`/`iife` source-module direction captured by this baseline.

## Audit log

- Durable completion notes now live in [Standalone_Inference_Export.logs.md](Standalone_Inference_Export.logs.md).
