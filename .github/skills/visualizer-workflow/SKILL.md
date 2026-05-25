---
name: visualizer-workflow
description: 'Design, debug, and harden browser visualizers in examples/ and docs surfaces, including layout scaling, overflow behavior, hover hit-testing, tooltip UX, visual parity checks, and durable validation for canvas/SVG/ASCII renderers.'
argument-hint: 'Describe the visualizer path, current symptom, expected behavior or parity target, rendering surface (canvas/SVG/DOM), and whether this pass is reconnaissance, implementation, or validation-only.'
user-invocable: true
disable-model-invocation: false
---

# Visualizer Workflow Playbook

Use this skill when a visualizer in `examples/`, docs assets, or browser-entry
runtime paths needs focused UX or rendering repair work.

This is the canonical workflow for visualizer-quality tasks in NeatapticTS:

- layout scale and panel proportion,
- horizontal/vertical overflow correctness,
- canvas/SVG hit-area mapping for hover behavior,
- tooltip reliability and style parity,
- color/opacity/readability safeguards,
- reference parity checks against an existing visualizer.

When tracker updates are required, `tracker-handoff` owns tracker structure.
When the task changes roadmap-sensitive architecture/runtime semantics,
`plan-alignment` owns plan selection and terminology.

## When to Use

- A visualizer appears cramped, clipped, or underuses available viewport space.
- Horizontal scrolling should appear for wide content but does not.
- Hover outlines or tooltips are missing, unstable, mispositioned, or visually
  inconsistent.
- One visualizer should match another established visualizer's information
  hierarchy and interactions.
- Opacity or blend behavior harms readability of labels, glyphs, or overlays.
- Responsive breakpoints collapse useful visual density too early.

## Scope Boundary

This skill owns visualizer behavior and presentation for:

- `examples/**` visualizer panels and browser-entry host wiring,
- shared visualization helpers under `src/visualization/**` when needed,
- tooltip contracts and hit-area update loops,
- parity passes between sibling demo visualizers.

This skill does not replace `educational-docs` for long-form teaching docs.

## Task Packet

Provide a compact packet that includes the visualizer file boundary, the exact
symptom, the expected behavior, the rendering surface, the pass mode, and
required final checks.

```text
Use visualizer-workflow for examples/asciiMaze/browser-entry/network-view.
Symptom: maze panel is cramped; horizontal overflow does not trigger.
Reference parity: examples/flappy_bird network visualizer panel behavior.
Surface: canvas plus DOM tooltip.
Mode: implementation.
Final checks: tsc, focused visualizer test slice, manual viewport verification.
```

## README-First Discovery Order

1. Nearest folder `README.md` for the visualizer boundary.
2. Parent visualizer or example README when parity spans sibling areas.
3. Then source files in this order: host services -> layout/style surface ->
   renderer orchestration -> draw helpers.

Use the README pass to identify intended panel hierarchy, label contracts,
tooltip semantics, and invariants before changing code.

## Required Workflow

1. Define the target behavior in one sentence.
   - Example: "Maze panel should consume available width and show horizontal
     scroll for long lines without clipping labels."

2. Classify the issue before editing.
   - `layout-allocation`: panel/grid width budget is wrong.
   - `overflow-contract`: content width and scroll ownership are mismatched.
   - `hit-area-sync`: hover state and rerender lifecycle are out of sync.
   - `style-parity`: color, opacity, or hierarchy mismatches a reference.

3. Start with the narrowest owner-local boundary.
   - CSS layout issues: start in page shell and panel/container styles.
   - Tooltip issues: start in host service state + event wiring.
   - Rendering issues: start in renderer orchestration before helper internals.

4. Preserve readability-first rendering.
   - Avoid global alpha changes that unintentionally dim text or glyph layers.
   - Prefer explicit fill/stroke colors and layer-specific opacity.

5. Keep hover contracts durable.
   - Ensure hit areas are refreshed after resize/rerender.
   - Keep tooltip content deterministic and easy to style.
   - Prefer tolerance-based hover resolution for small targets.

6. Validate in tight loops.
   - Typecheck after each meaningful edit cluster.
   - Run nearest owner-local test slice when behavior logic changed.
   - Confirm desktop and narrow-breakpoint behavior.

7. Close with evidence.
   - State what changed, why, and which symptom is now resolved.
   - Note any unresolved visual debt separately.

## Visualizer Guardrails

- Do not treat one viewport size as proof of correctness.
- Do not hide overflow by default when horizontal inspection is expected.
- Do not rely on incidental parent overflow for core scroll behavior.
- Do not use broad canvas alpha toggles for selective dimming.
- Do not couple tooltip visibility to stale hit-area snapshots.
- Do not regress established reference visualizer parity without a reason.

## Validation Rules

Recommended close-out sequence:

- `npx tsc --noEmit -p tsconfig.json`
- focused test slice if renderer/host logic changed
- optional docs refresh only when JSDoc/docs tooling changed
- quick manual browser verification for both desktop and compact breakpoint

## Expected Final Output

A strong completion report includes:

- the visualizer boundary touched,
- issue class and target behavior,
- files changed,
- validation results,
- remaining risks or next polish target.
