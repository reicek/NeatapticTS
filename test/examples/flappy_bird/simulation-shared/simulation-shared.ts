/**
 * Shared simulation public entry.
 *
 * This compatibility façade re-exports shared simulation utilities while
 * `flappy.simulation.shared.utils.ts` is decomposed into focused modules.
 */
export * from './simulation-shared.constants';
export * from './simulation-shared.control.utils';
export * from './simulation-shared.difficulty.utils';
export * from './simulation-shared.errors';
export * from './simulation-shared.math.utils';
export * from './simulation-shared.memory.utils';
export * from './simulation-shared.observation.utils';
export * from './simulation-shared.spawn.utils';
export * from './simulation-shared.types';
