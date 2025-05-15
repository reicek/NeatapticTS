/**
 * ASCII Maze - Main export file
 * 
 * This file exports all necessary components from the ASCII maze modules
 * to provide a clean interface for consuming code.
 */

// Re-export everything from individual modules
export * from './mazeUtils';
export * from './mazeVision';
export * from './mazeMovement';
export * from './mazeVisualization';
export * from './networkVisualization';
export * from './dashboardManager';
export * from './terminalUtility';
export * from './interfaces';

// Re-export specific items needed externally
export { colors } from './colors';
export * from './mazes';