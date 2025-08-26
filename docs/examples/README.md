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

### Behaviour & Telemetry

- Output area streams the live generation status; solved maze snapshots accumulate below.
- Lightweight telemetry (generation, best fitness, gens/sec) is exposed via:
  - CustomEvent `asciiMazeTelemetry` on `window`.
  - `window.asciiMazeLastTelemetry` object (latest snapshot).
  - The `run.onTelemetry(fn)` callback API when using the ESM `start()` function.
- Host pages can subscribe and surface summary stats outside the iframe.

### Programmatic API (ESM)

Import the `start` function and optionally supply an `AbortSignal` for cancellation.

```ts
import { start } from './asciiMaze/browser-entry.js';

const controller = new AbortController();
const run = await start('#ascii-maze-output', { signal: controller.signal });

const unsubscribe = run.onTelemetry(t => {
  console.log(`Gen ${t.generation} best ${t.bestFitness} (${t.gensPerSecond.toFixed(2)} gen/s)`);
});

// Early stop (either call stop or abort the signal)
// run.stop();
controller.abort();

await run.done;
unsubscribe();
```

API surface:
- `start(container: string|HTMLElement, opts?: { signal?: AbortSignal }) => Promise<RunHandle>`
- `RunHandle.stop(): void`
- `RunHandle.isRunning(): boolean`
- `RunHandle.done: Promise<void>`
- `RunHandle.onTelemetry(fn) => unsubscribe`
- `RunHandle.getTelemetry(): TelemetrySnapshot | undefined`

Telemetry snapshot (fields may extend):
```ts
interface TelemetrySnapshot {
  generation: number;
  bestFitness: number;
  gensPerSecond: number;
  exitReason?: string;          // 'cancelled' | 'aborted' | 'solved' | etc.
  details?: Record<string, any>; // rich metrics bundle
}
```

Legacy global usage (compatibility):
```html
<script src="/assets/ascii-maze.bundle.js"></script>
<script>
  window.asciiMaze.start(); // preferred
  // window.asciiMazeStart(); // deprecated alias logs a warning
</script>
```

### Embedding Notes

The iframe isolation keeps demo CSS / JS from interfering with site chrome. Now that an ESM `start()` is exported, future inline embedding can attach directly to a provided container without iframe overhead (see enhancement plan).

---

Additional examples will appear here as they are added.
