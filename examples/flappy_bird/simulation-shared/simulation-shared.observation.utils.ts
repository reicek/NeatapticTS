/**
 * Shared observation compatibility façade.
 *
 * The observation implementation now lives under `simulation-shared/observation/`
 * so feature synthesis and vector projection can evolve behind a focused module
 * boundary. This file stays as the stable import path for existing callers.
 *
 * That split keeps the high-level import path simple while allowing the
 * observation subsystem to grow into its own documented folder.
 */
export * from './observation/observation';
