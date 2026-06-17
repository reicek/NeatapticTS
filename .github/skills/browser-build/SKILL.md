---
name: browser-build
description: 'Configure, validate, and publish browser runtime artifacts for NeatapticTS. Use when working on scripts/build-browser.mjs, root dist browser bundles, docs/assets example bundles, browser env aliasing, smoke-browser-build validation, CDN/runtime packaging, or CI-sensitive browser build gates.'
argument-hint: 'Describe the target artifact (root browser dist / example docs asset / smoke test / size audit / CI gate), the current plan step, and known constraints such as size budget, worker entry delivery, HTML consumer, or browser targets.'
user-invocable: true
disable-model-invocation: false
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Browser Build Playbook

Use this skill when NeatapticTS needs a browser runtime artifact configured,
refreshed, validated, or published.

Browser build work is Phase 3 in the roadmap
(`plans/completed/Browser_Build_and_CDN_Distribution.md`). It is a **gate prerequisite**
for Phase 3 interactive examples and NEATchat: both depend on a stable ESM or
IIFE bundle before browser-runnable demos can exist.

This skill owns the durable workflow for root browser artifact generation via
`scripts/build-browser.mjs`, targeted example bundle refreshes under
`docs/assets`, public surface validation, CDN distribution targets, and browser
smoke tests. When tracker files need updating, `tracker-handoff` owns the
plan/log shape. When roadmap alignment is needed, use `plan-alignment`.

## Scope Boundary

- **In scope:** `scripts/build-browser.mjs`, `scripts/smoke-browser-build.mjs`,
  `dist/neataptic.browser.*` artifact generation, browser env aliasing to
  `src/env/browser/worker-loader.ts`, docs/assets example bundle refreshes,
  bundle size analysis, tree-shaking validation, CDN/runtime packaging, browser
  compatibility smoke tests, public API surface freeze.
- **Out of scope:** Node.js CJS/ESM package wiring (already complete, owned by
  ES2023 migration baseline), worker-side serialization (Phase 4 `Worker_Friendly_Network_Serialization_Fastpath.md`), standalone inference IR
  (Phase 4 `Standalone_Inference_Export.md`), visualizer layout or hover
  behavior, and demo-only logic that should stay outside the shared browser
  artifact boundary.

## When to Use

- Implementing a step from `completed/Browser_Build_and_CDN_Distribution.md`.
- Refreshing or validating `dist/neataptic.browser.esm.js`,
  `dist/neataptic.browser.iife.js`, or
  `dist/neataptic.browser.iife.min.js`.
- Refreshing a targeted docs-served example bundle such as Flappy Bird or
  NEATchat when the consuming HTML loads `docs/assets/*.bundle.js`.
- Validating that the public API surface is correctly tree-shakeable.
- Adding a CDN distribution artifact (`dist/neataptic.min.js`).
- Writing or repairing a browser smoke test that loads a root bundle or
  example-specific bundle and runs a minimal network.
- Auditing bundle size and identifying unexpectedly large contributors.
- Verifying whether the current repo actually exposes a package script for the
  browser build or whether the workflow must call `node scripts/build-browser.mjs`
  directly.

## Task Packet

Pass a compact packet that includes:

- target (root browser dist / example docs asset / smoke test / size audit / CI gate),
- current plan step from `plans/completed/Browser_Build_and_CDN_Distribution.md`,
- known constraints (size budget, supported browser targets, module format
- requirements, worker entry delivery, or HTML consumer),
- validation expectations.

Compact example:

```text
Use browser-build for root browser dist refresh.
Plan: plans/completed/Browser_Build_and_CDN_Distribution.md (Step 1).
Bundle format: ESM + IIFE root artifacts plus Node ESM smoke.
Browser targets: ES2023 baseline in scripts/build-browser.mjs.
Consumer: dist/neataptic.browser.esm.js and dist/neataptic.browser.iife.js.
Validate with: npm run build, browser build entrypoint, then scripts/smoke-browser-build.mjs.
```

## Required Workflow

1. Read `plans/completed/Browser_Build_and_CDN_Distribution.md` before editing.
2. Read `scripts/build-browser.mjs`, `scripts/smoke-browser-build.mjs`,
   `package.json`, and the consuming HTML or docs asset surface to understand
   the current state.
3. Identify whether the task targets:

- root browser artifacts under `dist/`, or
- a targeted docs/assets example bundle.

4. Verify actual script exposure before assuming the completed plan text is still
   mirrored in `package.json` or workflows.
5. Configure or refresh the artifact:

- **Root dist artifacts:** prefer the current package wrapper when present,
  otherwise run `node scripts/build-browser.mjs` directly.
- **Minified-only variant:** prefer the current wrapper when present,
  otherwise run `node scripts/build-browser.mjs --minify-only`.
- **Example docs asset:** rerun the matching targeted example build script,
  because `npm run build` alone does not refresh those bundles.

6. Validate:

- `npm run build` still passes for host code.
- The chosen browser artifact path actually refreshed.
- The smoke gate passes: prefer the current package wrapper when present,
  otherwise run `node scripts/smoke-browser-build.mjs` for the root dist path.
- Bundle size is within budget when size is part of the task.

7. If workflows, manifests, or browser/docs tooling changed, validate in CI
   order: `npm ci`, `npm run build`, then the browser build entrypoint.
8. If any `src/` files were modified, run `coverage-guard` on each changed file
   before continuing.
9. Run `npm run test:silent` only if the active step packet or user explicitly requires repo-wide confirmation; otherwise, report the focused slice result as the gate evidence.
10. Update `plans/completed/Browser_Build_and_CDN_Distribution.md` only after the code and
    validation are green.

## Bundle Quality Rules

- The root ESM and IIFE artifacts must expose only the intended public browser
  surface, not internal helpers or Node-only adapters.
- The root ESM artifact must support tree-shaking at the module level so unused
  subsystems can still be excluded.
- **Size budget:** aim for ≤150 kB minified+gzipped for the core Network + NEAT
  surface. Investigate contributors before accepting anything larger.
- Source maps must be generated for root browser artifacts and targeted example
  bundles when those bundles are user-facing debug surfaces.
- The build must not change the existing Node.js CJS/ESM package behavior.
- Browser artifact generation must keep the compile-time env alias behavior that
  redirects `src/env/index.ts` to the browser worker loader for browser-only
  outputs.

## Artifact Freshness Rules

- `npm run build` typechecks and refreshes host artifacts, but it does not, by
  itself, guarantee that root browser artifacts or docs-served example bundles
  are fresh.
- Targeted example HTML pages that load `docs/assets/*.bundle.js` can appear
  stale until their matching example build script is rerun.
- The repo currently keeps browser artifacts as an explicit post-build phase,
  so treat artifact freshness as something to validate, not assume.

## Smoke Test Rules

The minimum root-artifact smoke gate is the current
`scripts/smoke-browser-build.mjs` behavior:

- import `dist/neataptic.browser.esm.js`,
- verify that `Neat` and `Network` are exported,
- instantiate `new Network(2, 1)`,
- call `activate([0.5, 0.5])`,
- verify a finite numeric output.

When the task targets a docs-served example bundle instead of the root dist
artifact, validate the consuming HTML or entry surface that actually loads the
bundle.

## Guardrails

- Do not add browser-only polyfills that inflate the Node.js bundle.
- Do not assume a plan-completed package script still exists; verify the current
  `package.json` and script surface first.
- Do not assume `npm run build` refreshed browser artifacts or docs assets.
- Do not let `webpack` host builds or other clean steps silently delete browser
  artifacts without rerunning the explicit browser phase.
- Do not block Phase 3 examples on a perfect bundle; a working and verifiable
  browser artifact is the gate condition, not an optimized one.
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

- the browser artifact path or bundle family refreshed,
- the command path used to build it,
- bundle size (minified+gzipped for each target),
- smoke test result,
- whether the Node test suite remains green,
- the plan step updated.
