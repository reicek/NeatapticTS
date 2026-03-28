# Browser Build + CDN Distribution Plan

**Status:** [PLANNED]

## Purpose

Make NeatapticTS effortless to use in the browser:

- Provide **official browser bundles** (ESM + IIFE) with a stable public surface.
- Enable **copy/paste quickstarts** for demos, education, and interactive docs.
- Preserve Node-first correctness and determinism.

This plan is strictly about packaging, compatibility, and documentation—not new algorithms.

## Why this matters

A library can be technically excellent and still lose adoption if the first run experience is hard. Browser support unlocks:

- Interactive documentation and “try it live” examples.
- Lightweight playgrounds for evolution experiments.
- Classroom use (no Node install, no build pipeline).

## Goals

- G1: Produce a **browser ESM build** that works with modern bundlers and native ESM.
- G2: Produce a **single-file IIFE build** for `<script>` usage.
- G3: Ensure the browser build has a clearly defined public API entry (no deep imports).
- G4: Keep Node builds unchanged and fully supported.

## Non-goals

- Implementing new NEAT features (tracked elsewhere).
- Supporting legacy browsers requiring heavy polyfills.
- Guaranteeing identical performance characteristics vs Node.

## Compatibility target

- Modern evergreen browsers (Chromium, Firefox, Safari) with ES2023-ish features.
- If a feature requires Node-only APIs (fs, worker_threads), it must be:
  - compiled out of the browser bundle, or
  - behind a runtime capability check with a clear error.

## Public API (browser)

Provide a stable top-level namespace export.

Proposed browser entry exports:

- `Neat`
- `Network`
- `Architect` (if present)
- `methods` / activations / cost functions (where applicable)
- Optional: `version`

Avoid exporting internal/private helpers.

## Build artifacts

Add/standardize these artifacts under `dist/`:

- `dist/neataptic.browser.esm.js` (ESM)
- `dist/neataptic.browser.iife.js` (IIFE)
- `dist/neataptic.browser.iife.min.js` (minified)
- Source maps for all where practical

Naming can be adjusted to fit current conventions; the key is that we ship both ESM and IIFE.

## Implementation steps

### Step 1 — Define browser entry module

- Create a dedicated browser entry file (example name):
  - `src/browser-entry.ts`
- Re-export only supported browser-safe symbols.
- Ensure no Node-only modules are imported transitively.

Acceptance:

- A simple `import { Neat } from './dist/neataptic.browser.esm.js'` works.

### Step 2 — Split Node-only concerns behind environment adapters

- Introduce “environment adapter” modules so imports are browser-safe:
  - `src/env/node/*`
  - `src/env/browser/*`
  - `src/env/index.ts` that chooses by build target (compile-time), not runtime.

Notes:

- If runtime selection is needed, use capability checks, not user-agent sniffing.

Acceptance:

- Browser bundle has no references to Node built-ins (`fs`, `path`, `worker_threads`).

### Step 3 — Add explicit bundling pipeline

Use existing tooling already present in repo:

- Keep `webpack.config.js` for main dist build if desired.
- Add a small `esbuild`/webpack target for browser artifacts.

Add scripts (names illustrative):

- `npm run build:browser`
- `npm run build:browser:min`

Acceptance:

- `npm run build` continues to work.
- `npm run build:browser` produces the expected dist files.

### Step 4 — Document browser usage

Add a dedicated docs page:

- `docs/` content or a `README` section describing:
  - script tag IIFE usage
  - ESM usage
  - limitations (threads, filesystem, perf)

Acceptance:

- A user can run a minimal evolve loop in the browser with 10–20 lines.

### Step 5 — Add a CI smoke check

Without introducing heavy browser testing:

- Add a small build-time smoke check that:
  - imports the browser ESM output
  - runs a trivial activation on a toy network

Acceptance:

- Build fails if the browser bundle is broken.

## Testing strategy

- TypeScript: `npx tsc --noEmit -p tsconfig.json`
- Build: `npm run build` and `npm run build:browser`
- Minimal runtime smoke:
  - Node loads `dist/neataptic.browser.esm.js` (as ESM) and runs a simple call.
  - Optional: a headless browser run (Puppeteer exists in dev deps) for a single-page smoke.

## Risks and mitigations

- Risk: accidental Node-only transitive imports.
  - Mitigation: keep browser entry small; use env adapters.
- Risk: tree-shaking breaks side-effect assumptions.
  - Mitigation: mark side-effectful modules clearly; add smoke tests.
- Risk: bundle size grows.
  - Mitigation: export only essentials; avoid large optional modules by default.

## Success criteria

- `npm run build:browser` produces ESM + IIFE outputs.
- A minimal browser example works without bundlers.
- Node build remains unchanged.
- Documentation includes at least one runnable browser example.
