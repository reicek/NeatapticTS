/**
 * Aggregated public type surface for the Flappy Bird browser runtime.
 *
 * The browser demo spans several concerns at once: worker messaging, playback
 * rendering, telemetry, viewport math, and network visualization. Re-exporting
 * the public contracts from one place gives readers a compact map of that
 * runtime without forcing them to know the internal folder layout first.
 *
 * A practical reading order is:
 *
 * - runtime types for lifecycle and top-level handles,
 * - worker types for protocol boundaries,
 * - simulation/render types for playback state,
 * - visualization types for the network panel.
 *
 * Use this file when you want the browser runtime's public contract map. Use
 * the neighboring runtime, playback, host, worker-channel, and visualization
 * folders when you want the actual implementation story.
 *
 * Browser runtime map:
 * ```mermaid
 * flowchart TB
 *     PublicTypes["browser-entry.types"] --> Runtime["runtime types\nstart/stop lifecycle"]
 *     PublicTypes --> Worker["worker types\nprotocol and payloads"]
 *     PublicTypes --> Playback["simulation + render types\nframe state and HUD metrics"]
 *     PublicTypes --> Viz["visualization types\nlegend and color scales"]
 * ```
 */
export type * from './browser-entry.runtime.types';
export type * from './browser-entry.stats.types';
export type * from './browser-entry.worker.types';
export type * from './browser-entry.simulation.types';
export type * from './browser-entry.render.types';
export type * from './browser-entry.visualization.types';
