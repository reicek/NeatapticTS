/**
 * Compatibility entrypoint for the dedicated mazeMovement module.
 *
 * The real implementation now lives under the folder-based module boundary at
 * `mazeMovement/mazeMovement.ts`. This file remains so existing imports such as
 * `./mazeMovement` continue to resolve without changes.
 */

export { MazeMovement } from './mazeMovement/mazeMovement';
export type {
  DirectionSelectionStats,
  MazeMovementBufferPools,
  MazeMovementRunServiceState,
  MazeMovementSimulationResult,
  SimulationState,
} from './mazeMovement/mazeMovement.types';

