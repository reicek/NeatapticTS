# Flappy Recurrent Architecture Debug Pass

**Status:** [DONE]

## Scope

- Reproduce and repair the recurrent browser regressions reported for GRU,
  LSTM, and NARX after the earlier Flappy Step 7 integration work.
- Separate structural runtime corruption from behavior-only browser polish
  issues so the recurrent profiles were not all treated as one failure class.
- Leave the recurrent browser lane with durable probe coverage and without the
  temporary worker-side debug instrumentation used during diagnosis.

## Final state

- GRU and LSTM no longer fail strict genome validation during worker
  initialization or generation-zero bootstrap. The root cause was generic
  hidden dead-end repair mutating valid recurrent-module internals, and the
  repair path now skips hidden nodes owned by validated temporal descriptors.
- NARX is no longer treated as an open recurrent regression. Deep rerun
  validation passed under the accepted browser envelope using the shared
  architecture progress probe plus focused Flappy worker validation.
- The temporary recurrent debug logging surface has been removed from the
  Flappy worker boundary, and the generated Flappy docs were refreshed so the
  deleted debug service and flag no longer appear in the README shelves.

## Audit summary

- Added the temporal-module guard that kept generation-zero repair from
  rewiring valid GRU and LSTM internals, then validated the repaired worker and
  browser paths with focused source, bundle, and live-probe checks.
- Reused the shared `npm run flappy:architecture:progress` surface to validate
  GRU and NARX behavior instead of leaving the workstream on a long-running
  Jest investigation harness.
- Closed the workstream by deleting the temporary recurrent debug service and
  dead feature flag, then reran focused worker tests, `npm run build`,
  file-scoped ESLint, and `npm run docs`.

## Reopen conditions

- A recurrent profile again fails strict genome validation or reports duplicate
  innovations during worker bootstrap.
- The shared architecture progress probe regresses for GRU or NARX under the
  current browser defaults.
- A future recurrent polish pass needs fresh worker-side instrumentation to
  localize a new failure mode.

## Audit log

- Durable completion notes now live in
  [flappy-recurrent-architecture-debug.logs.md](flappy-recurrent-architecture-debug.logs.md).
