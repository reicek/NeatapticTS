# Network Visualization Export Schema — Audit Log

**Closed:** 2026-05-03

## Pass history

| Pass | Outcome |
|------|---------|
| Lane A — Schema export | `exportVisualizationGraph` + `toDot` implemented and tested in `src/architecture/network/visualization/` |
| Lane B — Shared renderer | `src/visualization/network-view/` extracted; `renderNetworkView` public API with 3 unit tests; ASCII Maze browser visualizer wired |
| Lane B Step B3 hardening | CSS Grid split-pane layout in `examples/asciiMaze/index.html`; left pane `minmax(min-content, max-content)`, right pane `minmax(280px, 1fr)`; overflow containment on `#ascii-maze-live` |
| Lane C — Docs/examples | Module-level JSDoc in `src/visualization/visualization.ts` expanded: canvas drop-in + overlay-hook section (C2); DOT export full/compact section (C1) with Graphviz Online link + Wikipedia citation |

## Final validation

- `npm run docs` — pass (173 diagrams, "HTML docs generated")
- `npm run test:silent` — 344 suites / 3165 tests pass
