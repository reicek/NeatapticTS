/**
 * Shared observation compatibility façade.
 *
 * The observation implementation now lives under `simulation-shared/observation/`
 * so feature synthesis and vector projection can evolve behind a focused module
 * boundary. This file stays as the stable import path for existing callers.
 */
export * from './observation/observation';
