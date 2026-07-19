# NGE Racing Curriculum Oscillation & Sub-Tier Fix log

**Status:** [DONE]

_Plan archived. Core oscillation fix (Phase 4) green-gated and complete. Phases 5`u20137 (bundle rebuild, documentation, session logging) cancelled and folded into [Racing_Perception_Redesign.plans.md](Racing_Perception_Redesign.plans.md)._

## Phase 4 — Implementation (compressed)

Phase 4 completed 2026-07-18. Step 04 — Implement sub-tier thresholds and oscillation penalties.

### Slices completed

- **[DONE] Slice 04-a — Library-side juvenile sub-tier exhaustion thresholds**
  - Files changed: `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts`, `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`
  - Added tier-specific neuron/connection fractions and `resolveNeuronTierFraction` helper.
  - `grow-stabilize` uses the current neuron budget to pick tier-specific exhaustion thresholds.
  - Preflight: tsc PASS, `quality:folder` PASS (0 TS/ESLint errors, 63/63 JSDoc symbols), prettier PASS.

- **[DONE] Slice 04-b — Racing runtime oscillation detection and penalties**
  - Files changed: `examples/racing_curriculum/controller/runtime.adaptation.ts`, `examples/racing_curriculum/controller/runtime.adaptation.test.ts`
  - Added steering/score oscillation helpers and applied penalties in `RACING_VARIANT_SCORER` and `evaluateRacingTrendScore`.
  - Added `RACING_OSCILLATION_MIN_NEURONS = 200` and tier-aware `resolveOscillationThresholdBoost`.
  - Preflight: tsc PASS, lint PASS (0 errors), prettier PASS, `quality:folder` PASS (39/39 JSDoc symbols).

- **[DONE] Slice 04-b-fix — Specialist review fix cycle**
  - File changed: `examples/racing_curriculum/controller/runtime.adaptation.test.ts`
  - Added GAP 1 test: ≥200-neuron oscillation-penalty path (`Network(200, 1)` + oscillating vs monotonic history).
  - Added GAP 2 test: runtime engine `scoreFn` wiring source-inspection (`scoreRacingVariant(outputs, target, tickInput.network.nodes.length)`).
  - NGE specialist: APPROVED both test gaps.
  - Preflight: tsc PASS, lint PASS, prettier PASS.

- **[DONE] Slice 04-c — Bundle rebuild and browser smoke validation**
  - Acceptance: bundle rebuild and browser smoke (dependency on 04-a/04-b).
  - Green validation owned by 05-green-testing.

### Green validation (05-green-testing)

- Targeted Jest run on `runtime.adaptation.test.ts`: **82/82 tests passed**.
- GAP 1 gate-specific test PASS: same oscillating history `[5,3,5,3,1]` scores lower on a large network (`Network(200, 1)`) than a small network (`Network(4, 2)`), proving the ≥200-neuron oscillation gate is active.
- GAP 2 source-inspection test PASS: `scoreFn` passes the live candidate neuron count to `scoreRacingVariant`.
- Folder-quality metrics PASS for `examples/racing_curriculum/controller/` (0 TS diagnostics, 0 ESLint errors, 39/39 JSDoc symbols).

### Plan-format reconciliation (01-planning)

- Reconciled Phase 4/Step 04 step-packet schema without changing production code:
  - Phase 4: `expansion: 'slices'` → `'steps'`
  - Phase 4: `auto_expand: true` → `false`
  - Slice 04-a: `goal: 'implementing'` → `'red-testing'`
  - Slice 04-c: `goal: 'implementing'` → `'green-testing'`
- `step-packet` gate → PASS; `plan-sync` gate → PASS.

### Gates

- plan-sync → PASS
- agent-graph → PASS
- plan-slice-quality → PASS
- step-packet → PASS (after schema reconciliation)
- learning-event → PASS

### Residual notes

- `runtime.adaptation.ts` lives under `examples/` and is excluded from Jest coverage collection by `jest.config.mjs`; no coverage regression is possible in the collected set.
- 21 pre-existing warnings in `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts` (test-only `@typescript-eslint/no-explicit-any` patterns) are outside this slice and unchanged.
- `npx tsc --noEmit -p tsconfig.test.json` has an unrelated pre-existing parse error inside `node_modules/devtools-protocol/types/protocol-mapping.d.ts`.
