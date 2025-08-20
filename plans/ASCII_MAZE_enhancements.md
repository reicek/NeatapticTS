## ASCII Maze Example – Enhancement & Integration Plan

### Context
The `asciiMaze` browser example is now integrated into the documentation site under `Examples`. It is copied by `npm run docs:examples` (script: `scripts/copy-examples.mjs`) after the bundle build (`npm run build:ascii-maze`). A new `docs/examples/README.md` provides an iframe wrapper so the demo appears within the standard docs chrome.

### Completed Integration Tasks
- Added top‑level Examples navigation tab & active state switching.
- Generated sidebar group `examples` (via new README) with link to `examples/asciiMaze` page.
- Embedded demo inside an iframe (`sandbox="allow-scripts allow-same-origin"`) to isolate styles / scripts.
- Added lightweight `theme.css` (missing previously) for consistent site styling and example frame presentation.

### Short‑Term Improvements (Iteration 1)
1. Convert bundle to pure ESM and expose `start(containerId | HTMLElement)` instead of relying on `window.asciiMazeStart` global.
2. Provide explicit teardown API (`stop()`) so the iframe (or future inline embed) can reclaim resources.
3. Add responsive resizing logic (ResizeObserver) so the ASCII output reflows when container width changes (esp. narrow mobile viewports).
4. Replace wide path‑probing script with a deterministic base path injection (e.g. data attribute) to simplify loading & reduce failed requests.
5. Emit lightweight telemetry (generations/sec, best fitness) into `window` or posted message so parent page can optionally display key stats outside iframe.

### Mid‑Term Enhancements (Iteration 2)
1. Accessibility: ARIA live region already present; add keyboard shortcuts (Space = pause/resume, R = reset) and focus styles.
2. The maze / agent logic: Parameterize maze size & mutation rates through query params (`?size=32x16&seed=...`).
3. Add performance budget overlay (generation time histogram) rendered with simple text sparklines to avoid heavy dependencies.
4. Provide deterministic seed controls + “Share Link” button encoding config in query string.
5. Implement shallow stats export (JSON) for download to compare runs externally.

### Longer‑Term (Iteration 3+)
1. Refactor example into a generic "interactive notebook" pattern where multiple evolutionary demos (maze, XOR, cart-pole) share UI scaffolding.
2. Add Web Worker evaluation mode toggle to contrast single vs multi‑thread throughput in real time.
3. Progressive enhancement: If OffscreenCanvas available, add optional visual maze rendering (while keeping ASCII fallback for environments without it).
4. Integrate with benchmark harness to capture comparative performance snapshots across library versions.

### Inline vs Iframe Trade‑offs
| Aspect | Iframe (current) | Inline Embed (future option) |
|--------|------------------|------------------------------|
| Style isolation | Strong | Must namespace classes |
| Global pollution | Contained | Needs careful API surface |
| Parent ↔ Demo comms | postMessage | Direct function calls |
| SEO / Crawl | Weaker (sandboxed) | Stronger (inline text) |
| Complexity | Low | Higher |

Decision: Keep iframe until at least two more examples exist; re‑evaluate when a shared example framework becomes beneficial.

### Required Refactors Before Inline Embedding
- Migrate startup to `export function start(el: HTMLElement, opts?: StartOptions)`.
- Remove dynamic script path probing; rely on bundler generated relative path & docs build ordering.
- Externalize configuration (mutation intervals, render cadence) so UI can alter at runtime without rebuild.

### Implementation Notes / Action Items Checklist
- [ ] Replace global `window.asciiMazeStart` with module export + UMD style wrapper for backward compatibility.
- [ ] Add `stop()` method and ensure timers / intervals cleared.
- [ ] Introduce a light event bus or callback props for telemetry; parent can listen via postMessage until inline.
- [ ] Simplify script path loading; log a single, actionable error.
- [ ] Query param parsing utility (shared with future examples) under `src/utils/exampleParams.ts` (planned).
- [ ] Add Jest smoke test (using JSDOM) to assert exported API shape (start/stop) – skip heavy evolution loop.
- [ ] Document usage in `docs/examples/README.md` (API section) after refactor.

### Risks & Mitigations
- Infinite loops or heavy CPU during evolution: provide `maxMsPerFrame` guard and yield back to event loop.
- Memory growth from archived logs: cap archive length; provide toggle for full log capture.
- Accessibility regressions: include basic axe-core audit in CI (optional future task).

### Exit Criteria (Iteration 1)
- API exports `start` & `stop`; no reliance on global function.
- Path loading simplified; zero 404 attempts on normal page load.
- Telemetry object accessible (either posted or on a stable API object) with generation count & best fitness.
- README updated reflecting new API.

---
Future edits to this plan should update checkboxes and iteration status as tasks land.

