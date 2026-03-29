# NEAT Test Surface Repair

**Status:** [DONE]

## Scope

- Repair the public NEAT compatibility surface so the test suite reflects the
  implementation-supported option and import contracts.
- Keep the fixes compatibility-first and isolated from unrelated example-lane
  failures.

## Final state

- The NEAT public surface once again matches the implementation-supported test
  expectations for options and root compatibility imports.
- NEAT-specific TypeScript regressions were cleared without reopening unrelated
  example diagnostics.
- No active backlog remains; this file is now a reopen point for future NEAT
  public-surface compatibility regressions.

## Audit summary

- Validation used `npx tsc --noEmit -p tsconfig.test.json`, targeted NEAT test
  execution, and `npm test` from the current repo state.
- Remaining test TypeScript issues at closeout were intentionally isolated to
  the `asciiMaze` example lane and later handled separately.

## Reopen conditions

- Future NEAT public-surface changes break test-facing option or import
  compatibility.
- `tsconfig.test.json` reintroduces NEAT-specific diagnostics.
- Root compatibility facades need another repair pass.

## Audit log

- Durable completion notes now live in
  [neat-test-surface-repair.logs.md](neat-test-surface-repair.logs.md).
