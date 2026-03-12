/**
 * Aggregated public type surface for the Flappy Bird browser runtime.
 *
 * The browser demo spans several concerns at once: worker messaging, playback
 * rendering, telemetry, viewport math, and network visualization. Re-exporting
 * the public contracts from one place gives readers a compact map of that
 * runtime without forcing them to know the internal folder layout first.
 */
export type * from './browser-entry.runtime.types';
export type * from './browser-entry.stats.types';
export type * from './browser-entry.worker.types';
export type * from './browser-entry.simulation.types';
export type * from './browser-entry.render.types';
export type * from './browser-entry.visualization.types';
