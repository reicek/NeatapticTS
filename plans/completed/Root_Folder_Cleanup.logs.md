# Root Folder Cleanup — log

**Status:** [DONE]

## Phase 1 — Root folder cleanup

- Files changed:
  - Deleted root clutter: `tmp/`, `tmp_*.json`, `tmp_*.mjs`, `missing-semantic-index.sqlite`, `freshness-proof-*/`, `artifacts/`, `lcov-parse.cjs`, `verify-results.mjs`, `coverage_audit*.mjs`, `debug-*.mjs`, `regex-test.cjs`, `check-*.mjs`, `smoke-*.mjs`, `ascii_maze_snapshots/`, `direct-worker-test.html`, `offscreen-classic-test.html`, `offscreen-classic-worker.js`, `offscreen-module-test.html`, `offscreen-module-worker.mjs`, `offscreen-test.html`, `worker-smoke.html`.
  - Relocated documentation and harnesses:
    - `Browser_Tests.md` → `docs/browser-tests/README.md`
    - `ONNX_EXPORT.md` → `docs/onnx-export.md`
    - `WebGPU.md` → merged into existing `docs/webgpu-inference.md`
    - `WebGPU_architecture/` → `docs/architecture/webgpu/`
    - `bench-browser/` → `testing/browser-benchmarks/`
    - `files/mcp-facade/` non-duplicate contents → `scripts/agent-customization/mcp/facade-contract/`
  - Updated references and code:
    - `README.md`, `docs/index.html`, `docs/architecture/webgpu/webgpu.architecture.md`, `.gitignore`
    - `benchmarks/benchmark.browser.headless.test.ts`, `benchmarks/benchmark.browser.memory.test.ts`, `testing/browser-benchmarks/bench-entry.ts`
    - `scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts`, `scripts/agent-customization/mcp/facade-contract/README.md`
    - `examples/asciiMaze/evolutionEngine/optionsAndSetup.ts`, `examples/asciiMaze/evolutionEngine/README.md`
    - `examples/racing_curriculum/README.md`, `.github/skills/webgpu/SKILL.md`
    - `scripts/agent-customization/hooks/post-write-reindex-hook.mjs` (moved log path, exported `main` and `readHookInput`, removed dead top-level catch block)
    - `scripts/agent-customization/hooks/post-write-reindex-hook.test.mjs` (added edge-path coverage)

- Validations run:
  - `npm run build` → PASS (pre-existing webpack size warnings)
  - `npm run lint` → PASS (0 issues)
  - `npx tsc --noEmit -p tsconfig.json` → PASS
  - Targeted Jest:
    - `benchmarks/benchmark.browser.headless.test.ts` → PASS (9/9)
    - `scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts` → PASS (61/61)
    - `examples/asciiMaze/evolutionEngine/optionsAndSetup.test.ts` → PASS (10/10)
    - `scripts/agent-customization/hooks/post-write-reindex-hook.test.mjs` → PASS (10/10)
  - `slice-advancement` gate → PASS (7/7 sub-gates; severity FULL)
  - `code-coverage` gate → PASS after hook edge-path tests added

- Learning events: none

- Decisions:
  - Step 03 deleted all research-artifact `delete` items (including originally slated relocation targets) per explicit user instruction.
  - Slices `04-relocate-scripts` and `04-relocate-browser-tests` were skipped because their targets no longer existed.
  - `WebGPU.md` was merged into the pre-existing (and different) `docs/webgpu-inference.md` rather than kept as a separate file.
  - Duplicate MCP facade snapshots were discarded; canonical versions under `scripts/agent-customization/mcp/` were retained.
  - The post-write-reindex hook log was moved from `artifacts/post-write-reindex.log` to `scripts/agent-customization/hooks/post-write-reindex.log` so the hook no longer recreates a root `artifacts/` directory.

- Risks / residual gaps:
  - `benchmarks/benchmark.browser.memory.test.ts` is a OneDrive reparse point in this checkout and could not be selected by Jest; the import path was updated and the sibling headless benchmark passed.
  - `./missing-semantic-index.sqlite` literals remain in MCP semantic tests as intentional missing-database error-path fixtures.

- Next resume point: workstream complete; reopen if new temporal/debug files accumulate at root or relocated paths break consumers.
