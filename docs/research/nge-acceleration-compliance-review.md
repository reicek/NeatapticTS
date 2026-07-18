# NGE Acceleration Layer Compliance Review

## Question

Is `plans/Generic_Acceleration_Layer.plans.md` correctly structured so that NGE
becomes a **consumer** of a new generic `src/acceleration/` layer rather than the
owner of acceleration logic, while keeping NGE lifecycle constants NGE-owned,
deleting all old NGE acceleration files with no deferred cleanup, and complying
with the project constitution (principles 1, 2, 4, 5)?

## Evidence

### 1. NGE as consumer of a generic acceleration layer

- **Static plan content (primary source):** The Proposed Architecture section
  states: "Old NGE acceleration files (`src/performance/nge/nge.acceleration*.ts`)
  are deleted entirely; NGE consumes the generic layer through
  `LifecycleAccelerationPolicy` only."
  (`plans/Generic_Acceleration_Layer.plans.md:359`).
- **Static plan content:** Acceptance criterion AC-010 requires deletion of
  `src/performance/nge/nge.acceleration.ts`, `nge.acceleration.gpu.ts`,
  `nge.acceleration.workers.ts`, `nge.acceleration.variants.ts`, and
  `nge.acceleration.adapter.ts`, and asserts that NGE consumes the generic layer
  through `LifecycleAccelerationPolicy` with no forwarding adapter remaining
  (`plans/Generic_Acceleration_Layer.plans.md:1908`).
- **Static plan content:** Phase 8 slices P8S3-02 include bash validations that
  all five old NGE acceleration files are gone before the slice is considered
  green (`plans/Generic_Acceleration_Layer.plans.md:1613-1617`,
  `1677-1705`, `1743`).
- **Current code state:** The old implementation files still exist under
  `src/performance/nge/`. This is expected because the plan is still in Phase 1
  and implementation has not started. They are scheduled for single-step removal
  in Phase 8, satisfying the "no deferred cleanup" rule.
- **Scout evidence (boundary-mapper):** Confirmed the current `src/performance/nge/`
  surface exports `resolveNgeAccelerationMode`, `detectNgeAcceleration`,
  `autoEnableGpu`, `autoEnableWorkers`, and synchronous `evaluateWeightVariants`,
  all of which are slated for deletion. No production dual-path code currently
  exists.

### 2. NGE lifecycle constants remain NGE-owned

- **Static source code:** `NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD` and
  `NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD` are defined in
  `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts` (lines 319 and 329).
- **Static source code:** `NGE_GROW_STABILIZE_BUFFER_POOL_MAX_POOLED_BYTES` is
  also currently defined in the same NGE-juvenile constants file (line 305). The
  plan correctly intends to move this generic cap to
  `src/acceleration/acceleration.constants.ts` as
  `DEFAULT_BUFFER_POOL_MAX_POOLED_BYTES`, but the file-by-file change list
  misstates the original source location as
  `src/architecture/network/gpu/network.gpu.buffer-set-pool.ts`. This is a plan
  inventory error, not a domain boundary error.
- **Static plan content:** The plan creates a new NGE consumer file
  `src/neat/nge-juvenile/neat.nge-juvenile.lifecycle-policy.ts` that builds the
  stage-to-config mapping from the existing NGE constants; the generic layer
  owns only the `LifecycleAccelerationPolicy` interface shape.

### 3. No deferred cleanup / no dual path

- **Static plan content:** The plan explicitly adopts the constitution's
  no-deferred-cleanup rule: "Delete old NGE acceleration files in the same step
  that introduces the generic layer."
  (`plans/Generic_Acceleration_Layer.plans.md:93`).
- **Static source code (latent risk):** `src/performance/nge/nge.acceleration.adapter.test.ts`
  currently tests a dual-path adapter contract: it imports generic acceleration
  symbols from `../../acceleration` and asserts that the old NGE surface
  delegates to them. This test file is **not** listed in the plan's deletion
  inventory, and its contract conflicts with AC-010's "no forwarding adapter"
  requirement.
- **Scout evidence (boundary-mapper):** Flagged this as a MEDIUM deferred-cleanup
  risk because the adapter test implies an adapter surface that the plan says
  must not survive.
- **Conflict resolution:** The plan's intent (delete all old files, no adapter)
  is stronger than the current test file's contract. The test file must be added
  to the deletion inventory before implementation begins.

### 4. Constitution principles 1, 2, 4, 5

- **Principle 1 (AI thinking partner, human decides):** The plan records a
  dedicated `### Risks` section with IDs, severities, and required amendments,
  and it explicitly requires a green light from independent reviews before
  implementation (`plans/Generic_Acceleration_Layer.plans.md:481-488`).
- **Principle 2 (human mission, AI method):** The plan defines goals and
  acceptance criteria, then delegates implementation to phase agents operating
  within skills and guardrails; it does not prescribe low-level implementation
  details outside the public contract.
- **Principle 4 (breadth-first, recoverable):** Work is sliced into phases and
  4-hour-or-less slices, each with red-green validation commands. Phase 8
  migration is a single recoverable unit with explicit pre/post bash checks.
- **Principle 5 (unique IDs):** Every acceptance criterion, phase, slice, risk,
  and required amendment carries a unique ID.

### 5. Validation gates

- `validate-plan-sync.mjs --plan=plans/Generic_Acceleration_Layer.plans.md` → PASS.
- `plan-slice-quality.gate.mjs --plan=plans/Generic_Acceleration_Layer.plans.md` → PASS.
- `plan-readiness.gate.mjs --plan=plans/Generic_Acceleration_Layer.plans.md` → PASS
  (green-light section present).
- `neataptic-gate-mcp-run_gate_check plan-sync` → PASS.
- `neataptic-gate-mcp-run_gate_check step-packet` → PASS.
- `neataptic-gate-mcp-run_gate_check plan-readiness` (default MCP binding) →
  FAIL because the default workflow snapshot is still bound to
  `plans/completed/Agentic_Workflow_Architecture.plans.md`. Using the explicit
  `--plan` override or the session redirect produces a pass. This is a tooling
  binding issue, not a plan-content issue.

## Decision

**APPROVED — with required amendments.** The plan's design correctly places NGE
as a consumer of a generic acceleration layer, keeps NGE lifecycle thresholds
NGE-owned, and deletes old NGE acceleration files in a single migration step.
The constitution alignment is strong. Two plan-inventory amendments must be made
before implementation starts:

1. Add `src/performance/nge/nge.acceleration.adapter.test.ts` to the Phase 8
   deletion inventory and to AC-010 so the dual-path adapter contract is not
   accidentally preserved.
2. Correct the buffer-pool cap constant source location in the file-by-file
   change list and Phase 7 slice: the constant currently lives in
   `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts`, not in
   `src/architecture/network/gpu/network.gpu.buffer-set-pool.ts`.

## Risks

| ID | Risk | Owner | Mitigation |
|----|------|-------|------------|
| NGE-RISK-01 | The adapter test file is not in the deletion inventory, so an implementation agent might leave a dual-path adapter in place. | Phase 8 P8S3-02 / 01-planning | Update AC-010 and the Phase 8 file list before work begins; gate validation already checks for `nge.acceleration.adapter.ts`. |
| NGE-RISK-02 | The buffer-pool cap source location is misstated, so the implementer might miss the architecture→neat upward import that must be removed. | Phase 7 P7S1-02 / 01-planning | Refresh the file-by-file change list with the correct source file and explicit goal of removing the upward import. |
| NGE-RISK-03 | Default MCP workflow binding resolves to a completed plan, so unqualified gate runs may report false negatives. | 02-researching / 00-helping | Use explicit `--plan=plans/Generic_Acceleration_Layer.plans.md` for gate calls until the session binding is repaired host-side. |
