# Test Colocation + Root Examples Migration Log

**Status:** [DONE]

## Durable milestones

### Source-owned test migration

- [DONE] Drained the legacy source-owned shelves into owner-local `src/**/*.test.ts` chapters across methods, architecture, multithreading, network, ONNX, and NEAT boundaries.
- [DONE] Kept the shared global Jest setup in `testing/jest-setup.ts` instead of recreating another root-owned test tree.

### Example-root migration

- [DONE] Promoted `examples/**` to the canonical home for runnable demos, example docs sources, and example-oriented browser entrypoints.
- [DONE] Verified the migrated ASCII Maze `.e2e.test.ts` surface is structurally healthy and should be treated as a separate long-running or manual validation lane.

### Benchmark-root migration

- [DONE] Moved `test/benchmarks/**` to `benchmarks/**`, fixed moved import depths and artifact readers, and aligned package scripts to the new root.
- [DONE] Preserved `bench:asciiMaze` with a compatibility CLI shim and converted previously empty benchmark placeholders into explicit skipped suites.

### Final retirement

- [DONE] Removed live `test/` references from Jest and TypeScript config, docs generators, docs-link surfaces, contribution guidance, and active agent or skill instructions.
- [DONE] Deleted the empty top-level `test/` directory.
- [DONE] Validated the terminal state with `npm ci`, `npm run docs`, `npm run lint`, `npm test`, `npm run build`, and a follow-up `npm run docs:build-scripts` pass that confirmed stale orphaned `dist-docs/scripts/analyze-trace.*` outputs stay removed.
- [DONE] Pruned obsolete tracked `bench-browser` bundles and wrapper HTML outputs, guarded the folder with ignore rules, and updated the browser headless benchmark harness to delete its own temporary wrapper HTML files so future generated remnants do not reintroduce legacy `test/` path strings.
