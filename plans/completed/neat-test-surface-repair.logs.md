# NEAT Test Surface Repair Log

**Status:** [DONE]

## Audit scope

- Objective: restore the NEAT public TypeScript and import surface expected by
  the test suite without widening the repair into unrelated example failures.

## Durable milestones

### [DONE] Failure classification

- Grouped the failing diagnostics into public-surface regressions rather than
  runtime-assertion failures so the repair could stay compatibility-first.

### [DONE] Public-surface restoration

- Widened the public `NeatOptions` shape to match implementation-supported
  fields used by tests.
- Reintroduced the thin root compatibility facades needed by legacy imports.

### [DONE] Validation closure

- Cleared the NEAT-specific TypeScript failures in `tsconfig.test.json` and
  confirmed the full Jest suite still passes.
- Isolated the remaining diagnostics to the `asciiMaze` example lane instead of
  leaving ambiguity at closeout.

## Controls and evidence

- Validation used `npx tsc --noEmit -p tsconfig.test.json`,
  `npx jest test/neat --runInBand`, and `npm test`.

## Reopen triggers

- NEAT public imports or option contracts regress again.
- New test-facing compatibility errors appear in the NEAT lane.
