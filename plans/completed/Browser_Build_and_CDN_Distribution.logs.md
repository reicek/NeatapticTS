# Browser Build + CDN Distribution Log

**Status:** [DONE]

## Audit scope

- Objective: close the Phase 3 browser packaging lane after the browser-safe entry surface, artifact pipeline, docs quickstarts, smoke validation, and release-path gating all landed.
- The pass covered the browser entry boundary, mixed-runtime adapter split, artifact build pipeline, docs guidance, smoke validation, and release publication integration.

## Durable milestones

### [DONE] Browser-safe root surface and mixed-runtime isolation

- Added `src/browser-entry.ts` and an owner-local test to define the browser-safe public shelf while intentionally excluding `multi`.
- Split the worker-loader boundary into `src/env/index.ts`, `src/env/node/worker-loader.ts`, and `src/env/browser/worker-loader.ts`.
- Updated `Workers` and `Multi` to consume the environment adapter instead of hardcoding node and browser worker imports directly.
- Confirmed the browser adapter path compiles without `fs`, `path`, `child_process`, or `worker_threads` imports.

### [DONE] Browser artifact build pipeline

- Added `scripts/build-browser.mjs` with the compile-time alias from `src/env/index.ts` to the browser adapter.
- Emitted the ESM, IIFE, and minified IIFE browser artifacts under `dist/`.
- Exposed `npm run build:browser` and `npm run build:browser:min` in `package.json`.

### [DONE] Documentation and consumer guidance

- Added browser ESM and IIFE quickstarts plus limitation notes to `README.md` and the generated docs landing page.
- Fixed the docs renderer so the root `docs/README.md` participates in HTML emission.
- Clarified in the examples docs that Flappy Bird and ASCII Maze load example-specific source bundles from `docs/assets`, while the public `dist/neataptic.browser.*` files are the external-consumer surface.

### [DONE] Smoke validation and release gating

- Added `scripts/smoke-browser-build.mjs` to import `dist/neataptic.browser.esm.js` and run a trivial activation.
- Chained the smoke step into `npm run build:browser` and wired the same gate into `.github/workflows/ci.yml`.
- Extended `npm run deploy` and `.github/workflows/publish.yml` so release-oriented publication now requires the browser build and smoke gate after `npm run build`.

## Controls and evidence

- Build and bundle validation: `npm run build`, `npm run build:browser`, and `npm run build:browser:min`.
- Clean-install validation after workflow and script changes: `npm ci`, then `npm run build`, then `npm run build:browser`.
- Documentation validation: `npm run docs`.
- Direct bundle validation: Node ESM import smoke against `dist/neataptic.browser.esm.js`.
- Shared worktree caution at closure time: `npm run test:silent` remained blocked by the unrelated `src/neat/mutation/add-conn/mutation.add-conn.test.ts` `firstHiddenNode` error and stayed out of scope.

## Reopen triggers

- A direct public-bundle consumer example or browser playground is needed beyond README quickstarts.
- The browser bundle needs additional safe exports or browser-worker ergonomics beyond the current root surface.
- The explicit browser build phase should be collapsed into the primary build path or otherwise redistributed.
- Bundle-size, format, or CDN-distribution policy changes require a new packaging pass.
