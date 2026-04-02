# Test Colocation + Root Examples Migration

**Status:** [DONE]

## Scope

- Move source-owned behavioral tests beside their owners under `src/**/*.test.ts`.
- Promote `examples/**` to the canonical runnable-example root.
- Promote `benchmarks/**` to the canonical benchmark and benchmark-artifact root.
- Remove the legacy top-level `test/` root and align live tooling, docs, and instructions to the new layout.

## Final state

- [DONE] Source-owned tests now live with their owning chapters under `src/**`, while the remaining shared global Jest setup lives in `testing/jest-setup.ts`.
- [DONE] Runnable examples and example docs sources now live under `examples/**`, including the migrated Flappy Bird and ASCII Maze surfaces.
- [DONE] Benchmark suites and artifacts now live under `benchmarks/**`, with `bench:asciiMaze` preserved through a compatibility CLI shim.
- [DONE] Live configs, docs tooling, repo docs, and active agent or skill instructions no longer reference the legacy `test/` root.
- [DONE] The empty top-level `test/` directory has been deleted.

## Audit summary

- Validation passed: `npm ci`.
- Validation passed: `npm run docs`.
- Validation passed: `npm run lint`.
- Validation passed: `npm test` (`142` suites, `1058` tests).
- Validation passed: `npm run build`.
- Validation passed: `npm run docs:build-scripts` after removing stale orphaned `dist-docs/scripts/analyze-trace.*` outputs.
- Follow-up cleanup pruned obsolete tracked `bench-browser` bundle and wrapper artifacts, and the browser benchmark harness now deletes its own temporary wrapper HTML files so active generated surfaces no longer carry legacy `test/` path strings.
- The migrated `examples/asciiMaze/asciiMaze.e2e.test.ts` lane is structurally healthy and remains a separate long-running or manual validation lane rather than a root-layout blocker.

## Reopen conditions

- Reopen this workstream only if a new live tool, docs surface, or config regresses to `test/`-root assumptions.
- Do not reopen solely to run long-running example `.e2e.test.ts` suites; those belong to their own validation lane.

## Audit log

- See `plans/test-colocation-and-root-examples.logs.md`.
