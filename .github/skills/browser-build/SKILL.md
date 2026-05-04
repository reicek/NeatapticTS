---
name: browser-build
description: 'Configure, validate, and publish ESM and IIFE browser bundles for NeatapticTS. Use when implementing Phase 3 browser build work, CDN distribution, bundle size analysis, or browser smoke tests.'
argument-hint: 'Describe the target (ESM bundle, IIFE bundle, CDN config, size audit, or smoke test), the current plan step, and known constraints (size budget, browser targets, module format).'
user-invocable: true
disable-model-invocation: false
---

# Browser Build Playbook

Use this skill when NeatapticTS needs its browser bundle configured, validated,
or published.

Browser build work is Phase 3 in the roadmap
(`plans/Browser_Build_and_CDN_Distribution.md`). It is a **gate prerequisite**
for Phase 3 interactive examples and NEATchat: both depend on a stable ESM or
IIFE bundle before browser-runnable demos can exist.

This skill owns the durable workflow for bundle configuration, public surface
validation, CDN distribution targets, and browser smoke tests. When tracker
files need updating, `tracker-handoff` owns the plan/log shape. When roadmap
alignment is needed, use `plan-alignment`.

## Scope Boundary

- **In scope:** Webpack/Rollup ESM and IIFE target configuration, bundle size
  analysis, tree-shaking validation, CDN distribution scripts, browser
  compatibility smoke tests, public API surface freeze.
- **Out of scope:** Node.js CJS/ESM package wiring (already complete, owned by
  ES2023 migration baseline), worker-side serialization (Phase 4 `Worker_Friendly_Network_Serialization_Fastpath.md`), standalone inference IR
  (Phase 4 `Standalone_Inference_Export.md`).

## When to Use

- Implementing a step from `Browser_Build_and_CDN_Distribution.md`.
- Configuring Webpack or Rollup for ESM and IIFE output targets.
- Validating that the public API surface is correctly tree-shakeable.
- Adding a CDN distribution artifact (`dist/neataptic.min.js`).
- Writing a browser smoke test that loads the bundle and runs a minimal network.
- Auditing bundle size and identifying unexpectedly large contributors.

## Task Packet

Pass a compact packet that includes:

- target (ESM config / IIFE config / CDN distribution / size audit / smoke test),
- current plan step from `plans/Browser_Build_and_CDN_Distribution.md`,
- known constraints (size budget, supported browser targets, module format
  requirements),
- validation expectations.

Compact example:

```text
Use browser-build for ESM bundle configuration.
Plan: plans/Browser_Build_and_CDN_Distribution.md (Step 1).
Bundle format: ESM + IIFE.
Browser targets: ES2020+ (Chrome/FF/Safari latest-2).
Validate with: npm run build, then manual smoke test in browser.
```

## Required Workflow

1. Read `plans/Browser_Build_and_CDN_Distribution.md` before editing.
2. Read `webpack.config.js` (or current bundler config) and `src/neataptic.ts`
   (public entry point) to understand the current state.
3. Identify the complete public API surface that must survive bundling and
   tree-shaking.
4. Configure the bundle:
   - **ESM:** `type: "module"` output, preserving named exports from
     `src/neataptic.ts`.
   - **IIFE:** `window.Neataptic` or equivalent global for CDN use.
5. Add or update the build script in `package.json`.
6. Validate:
   - `npm run build` produces the correct output artifacts under `dist/`.
   - Bundle size is within budget (see rules below).
   - A minimal browser smoke test can load the bundle, instantiate a `Network`,
     and run `activate()` without errors.
7. If any `src/` files were modified, run `coverage-guard` on each changed file
   to enforce 100% coverage in all four categories before continuing.
8. Run `npm run test:silent` to confirm the Node test suite is unaffected.
9. Update `plans/Browser_Build_and_CDN_Distribution.md` with the completed step.

## Bundle Quality Rules

- The IIFE bundle must expose only the public API surface from `src/neataptic.ts`,
  not internal helpers or implementation details.
- The ESM bundle must support tree-shaking at the module level so unused
  subsystems (e.g. multithreading) can be excluded.
- **Size budget:** aim for ≤150 kB minified+gzipped for the core Network + NEAT
  surface. Investigate contributors before accepting anything larger.
- Source maps must be generated for both ESM and IIFE targets.
- The build must not change the existing Node.js CJS/ESM package behavior.

## Smoke Test Rules

A browser smoke test is the minimum proof that the bundle works in a browser
runtime:

- Load the bundle via `<script type="module">` (ESM) or `<script>` (IIFE).
- Instantiate `new Neataptic.Network(2, 1)` (or equivalent).
- Call `activate([0.5, 0.5])` and verify the output is a finite number array.
- The smoke test should live in `examples/browser-quickstart/` (or the
  canonical Phase 3 examples home once decided).

## Guardrails

- Do not add browser-only polyfills that inflate the Node.js bundle.
- Do not block Phase 3 examples on a perfect bundle; a working bundle is the
  gate condition, not an optimized one.
- Do not merge browser build configuration with Phase 4 worker/serialization
  work; those are separate plan steps.
- Do not claim a bundle "works" without a verified smoke test.
- Do not hand-edit generated README files.
- Follow `tracker-handoff` when updating plan/log files.
- Follow `educational-docs` for JSDoc on the public API surface after bundle
  changes — the browser quickstart chapter should explain module formats, CDN
  usage, and tree-shaking, not just show a script tag.
- Do not prepend calendar dates to plan headings or session logs.

## Expected Final Output

A strong browser build pass should report:

- the bundle format(s) produced and their output artifact paths,
- bundle size (minified+gzipped for each target),
- smoke test result,
- whether the Node test suite remains green,
- the plan step updated.
