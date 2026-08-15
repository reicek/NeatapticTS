/**
 * @module voxel-enemy.types
 *
 * Type definitions for the voxel enemy descriptor, extracted from
 * `voxel-enemy.ts` so other modules can import them without a circular
 * dependency on the builder implementation.
 */

/** Material slots referenced by the snapshot renderer. */
export type MaterialSlot = 'accent' | 'neon' | 'suit' | 'dark' | 'damage';

/**
 * Sparse voxel grid dimensions, occupied voxels, body-part tags, and the
 * material palette used by the snapshot renderer.
 */
export interface VoxelGrid {
  /** N×192×M voxel grid dimensions. */
  width: number;
  height: number;
  depth: number;
  /** Individual voxels making up the enemy. */
  voxels: Voxel[];
  /** Named body parts present in the descriptor. */
  parts: string[];
  /** Material palette. */
  palette: VoxelPalette;
  /** Edge/neon/disk thickness in voxels. */
  thickness: number;
}

/**
 * Single occupied voxel with a position, body part, material slot, albedo
 * color, emissive flag, and alpha value.
 */
export interface Voxel {
  x: number;
  y: number;
  z: number;
  /** Body part this voxel belongs to (e.g. 'head', 'torso', 'cannon'). */
  part: string;
  /** Material slot (e.g. 'accent', 'neon', 'suit', 'dark', 'damage'). */
  material: MaterialSlot;
  /** Red channel of the voxel albedo. */
  r: number;
  /** Green channel of the voxel albedo. */
  g: number;
  /** Blue channel of the voxel albedo. */
  b: number;
  /** True for neon/accent emissive voxels. */
  emissive: boolean;
  /** Alpha opacity; 1.0 is fully opaque. */
  alpha: number;
}

/**
 * Named material palette for the enemy, with one swappable accent color and
 * fixed neon, suit, dark, and damage slots.
 */
export interface VoxelPalette {
  /** Swappable accent color (default Ares Red). */
  accent: string;
  /** Neon white/bright emission color. */
  neon: string;
  /** Dark suit base color. */
  suit: string;
  /** Darker shadow variant of the suit. */
  dark: string;
  /** Damage flash color. */
  damage: string;
}