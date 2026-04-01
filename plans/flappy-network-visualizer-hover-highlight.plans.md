# Flappy Network Visualizer Hover Highlight

**Status:** [PLANNED]

## Scope

- Add node-hover interaction to the Flappy browser network visualizer so the
  user can inspect local connectivity more easily.
- When hovering any node type (`input`, `hidden`, or `output`), reduce all
  non-adjacent connection lines to `20%` opacity.
- Raise directly connected lines to `90%` opacity and add glow using each
  connection's own color.
- Add glow to the hovered node only, while leaving all other nodes visually
  unchanged.
- When no node is hovered, draw all connection lines at `70%` opacity so dense
  overlap stays easier to read.
- Stop at the action-plan stage and await confirmation before implementation.

## Current state

- The network canvas DOM ownership lives in
  `examples/flappy_bird/browser-entry/host/host.ts`, where the host creates the
  network canvas, stores the last rendered network payload, and already owns the
  redraw controller boundary.
- The network-view orchestration lives in
  `examples/flappy_bird/browser-entry/network-view/network-view.ts`. It resolves
  topology, positions nodes, and draws the graph, but it currently discards the
  positioned-node scene after render instead of exposing it for hover hit
  testing.
- Connection and node painting live in
  `examples/flappy_bird/browser-entry/visualization/visualization.draw.service.ts`.
  Connection alpha is currently fixed by shared constants in
  `examples/flappy_bird/browser-entry/visualization/visualization.constants.ts`,
  so there is one clear styling seam for default, dimmed, and highlighted line
  opacity.
- No pointer listeners or hover-state model are currently attached to the
  network canvas, so this feature needs an explicit browser-side hover owner.
- The visualizer already has the right per-connection color data to support the
  requested same-color line glow, because connection color is resolved before
  each edge is painted.

## Coverage backlog

- [PLANNED] Add a hover-state owner in the host network-visualization
  controller. The host is the right place for `pointermove` and `pointerleave`
  listeners because it owns the network canvas element and the redraw loop.
- [PLANNED] Extract or expose a reusable positioned-node render snapshot from
  `network-view/` so hover hit testing uses the same node geometry as the draw
  path rather than approximating positions independently.
- [PLANNED] Add a pure hit-test helper that maps canvas pointer coordinates to
  a hovered node index using the latest positioned-node scene and node box
  dimensions.
- [PLANNED] Add hover-aware connection styling in the visualization draw layer:
  default line opacity `0.7`, non-adjacent hovered lines `0.2`, directly
  connected hovered lines `0.9`, and same-color glow for the connected subset.
- [PLANNED] Add hovered-node glow without changing the appearance of any other
  node. The recommended interpretation is additive glow on the hovered node only
  while preserving existing fill/stroke treatment for non-hovered nodes.
- [PLANNED] Preserve existing semantic line differences such as disabled-edge
  dashes and sign-dependent color while layering the new hover alpha/glow
  behavior on top.
- [PLANNED] Add or refresh targeted docs/tests where practical, then validate
  with TypeScript, Flappy bundle build, and targeted Flappy docs generation.

## Immediate next steps

1. Refactor the network-view boundary so one reusable render snapshot can be
   shared by both the current draw path and the future hover hit-test path.
2. Extend the host network-visualization controller to cache that snapshot,
   install `pointermove` plus `pointerleave` listeners on the network canvas,
   and trigger redraws when the hovered node changes.
3. Add a hover-style resolver in the visualization draw layer that classifies
   each connection as `default`, `dimmed`, or `highlighted` based on adjacency
   to the hovered node.
4. Add same-color glow for highlighted lines and a hovered-node-only glow pass,
   while leaving all non-hovered nodes untouched.
5. Validate with `npx tsc --noEmit -p tsconfig.json`, `npm run build:flappy-bird`,
   and `npm run docs:folders:flappy-bird`.

## Deferred questions

- When a hovered node is adjacent to multiple differently colored lines, the
  cleanest interpretation is an additive multi-pass glow on the hovered node
  using each adjacent line color. This matches the request more faithfully than
  picking one arbitrary dominant color.
- Disabled connections should likely keep their dashed treatment under hover,
  with only opacity and glow behavior changing. This preserves the existing
  semantic distinction between enabled and disabled edges.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Implement the node-hover highlight plan in plans/flappy-network-visualizer-hover-highlight.plans.md for the Flappy browser network visualizer. Keep host-side pointer ownership in examples/flappy_bird/browser-entry/host/host.ts, expose reusable positioned-node scene data from examples/flappy_bird/browser-entry/network-view/, and apply hover-aware connection styling in examples/flappy_bird/browser-entry/visualization/visualization.draw.service.ts. Required behavior: no-hover lines at 70% opacity, hovered non-adjacent lines at 20%, hovered adjacent lines at 90% plus same-color glow, and hovered node glow only with no other node changes. Preserve disabled-edge semantics where practical and validate with npx tsc --noEmit -p tsconfig.json, npm run build:flappy-bird, and npm run docs:folders:flappy-bird.
```
