<!-- Auto-generated into HTML by scripts/render-docs-html.ts -->
# Examples

Interactive demos showcasing NeatapticTS concepts in the browser. Each example is self‑contained and built with native ES2023 features – no transpilation.

> Tip: Open DevTools console while an example runs to inspect logged telemetry or errors.

## asciiMaze (Neuro‑evolving maze agent)

Demonstrates a simple environment where agents must traverse an ASCII maze layout. The population evolves weights / topology over generations; progress prints in real time.

<div class="example-frame">
  <iframe src="./asciiMaze/index.html" title="ASCII Maze Example" loading="lazy" referrerpolicy="no-referrer" sandbox="allow-scripts allow-same-origin" aria-label="ASCII Maze interactive example"></iframe>
  <p class="example-caption">If the demo fails to load, ensure the build step <code>npm run build:ascii-maze</code> has produced <code>docs/assets/ascii-maze.bundle.js</code>.</p>
</div>

### Controls & Behaviour

- The Play / Pause button toggles evolutionary steps & rendering.
- Output area streams the live generation status; archived snapshots accumulate below.
- A watchdog restarts the demo if the bundle loads slowly.

### Embedding Notes

The iframe isolation keeps demo CSS / JS from interfering with site chrome. For tighter integration (shared styles / dark mode variables), convert the example to an ESM entry that renders into a supplied container (see enhancement plan).

---

Additional examples will appear here as they are added.
