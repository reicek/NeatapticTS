# Browser Build + CDN Distribution Plan

**Status:** [DONE]

## Scope

- Provide official browser ESM and IIFE bundles with a stable public surface.
- Keep the Node-first distribution flow intact while compiling Node-only worker-loader imports out of browser artifacts.
- Add docs, smoke validation, and release-path gating so browser artifacts are treated as supported publication outputs.

## Final state

- `src/browser-entry.ts` now defines the browser-safe root surface and intentionally omits the mixed-runtime `multi` facade.
- `src/env/index.ts` plus `src/env/node/*` and `src/env/browser/*` now isolate the worker-loader split so browser artifacts compile without `fs`, `path`, `child_process`, or `worker_threads` imports.
- `scripts/build-browser.mjs` emits `dist/neataptic.browser.esm.js`, `dist/neataptic.browser.iife.js`, and `dist/neataptic.browser.iife.min.js`, and `package.json` exposes `npm run build:browser` plus `npm run build:browser:min`.
- The root README and generated docs landing page now teach ESM and IIFE usage, the explicit browser-build phase, and the current browser runtime limitations.
- `scripts/smoke-browser-build.mjs` now validates direct ESM import plus trivial activation, and that gate runs in CI as well as release-oriented commands.
- `npm run deploy` and `.github/workflows/publish.yml` now require `npm run build:browser` after `npm run build`, so browser artifacts and the smoke gate are part of package publication.
- The flagship demos remain reference apps built from example-specific source bundles under `docs/assets`; the public `dist/neataptic.browser.*` artifacts are now documented as the external-consumer surface instead of the demo runtime source.

## Audit summary

- The workstream closed the browser-packaging gap without destabilizing the existing Node webpack build; browser artifacts intentionally remain an explicit second phase after `npm run build`.
- The environment-adapter seam was the key correctness boundary because the mixed-runtime worker-loader path was the remaining Node-only pressure on the root browser facade.
- Validation stayed focused on artifact correctness, docs synchronization, and release-path safety rather than the unrelated shared-worktree failure in `src/neat/mutation/add-conn/mutation.add-conn.test.ts`.
- Release parity is now explicit: browser artifacts and the ESM smoke gate are part of CI and publication, not just local development.

## Reopen conditions

- The public browser bundle needs additional exported surfaces, including any safe replacement for the currently omitted `multi` boundary.
- A direct bundle-consumer example or browser-first playground is needed beyond the current README quickstarts.
- Browser artifacts should be folded into the main `npm run build` path instead of staying a separate explicit phase.
- Bundle-size, sourcemap, or distribution-policy changes require revisiting the browser packaging contract.

## Audit log

- Durable completion notes now live in [Browser_Build_and_CDN_Distribution.logs.md](Browser_Build_and_CDN_Distribution.logs.md).
