/**
 * Voxel enemy descriptor for the Neatenstein NGE demo.
 *
 * Builds a deterministic, sparse N×192×M voxel grid describing a humanoid
 * combat robot: head, torso, arms, legs, a chest-height right-arm cannon,
 * and a back-mounted identity disk. The default accent is Ares Red and can
 * be swapped for other neon hues.
 */

/**
 * Backward-compatibility re-exports from the extracted types and constants
 * modules. These exist so that legacy imports from `voxel-enemy.ts` continue
 * to resolve after the module was split into `voxel-enemy.types.ts` and
 * `voxel-enemy.constants.ts`.
 *
 * Prefer importing directly from './voxel-enemy.types' or
 *   './voxel-enemy.constants'. The re-export block exists for backward
 *   compatibility and may be removed in a future cleanup pass.
 *
 * @deprecated Import directly from './voxel-enemy.types' or
 *   './voxel-enemy.constants' instead.
 *
 * @example
 * ```ts
 * // Legacy (still works):
 * import { MaterialSlot, GRID_WIDTH } from './voxel-enemy';
 *
 * // Preferred:
 * import type { MaterialSlot } from './voxel-enemy.types';
 * import { GRID_WIDTH } from './voxel-enemy.constants';
 * ```
 */
export type {
  MaterialSlot,
  VoxelGrid,
  Voxel,
  VoxelPalette,
} from './voxel-enemy.types';
export {
  DEFAULT_ACCENT,
  NEON_WHITE,
  DARK_SUIT,
  DEEP_SHADOW,
  DAMAGE_FLASH,
  GRID_WIDTH,
  GRID_HEIGHT,
  GRID_DEPTH,
  GRID_CENTER_X,
  GRID_CENTER_Z,
  THICKNESS,
  PARTS,
} from './voxel-enemy.constants';

import type {
  MaterialSlot,
  VoxelGrid,
  Voxel,
  VoxelPalette,
} from './voxel-enemy.types';
import {
  DEFAULT_ACCENT,
  NEON_WHITE,
  DARK_SUIT,
  DEEP_SHADOW,
  DAMAGE_FLASH,
  GRID_WIDTH,
  GRID_HEIGHT,
  GRID_DEPTH,
  GRID_CENTER_X,
  GRID_CENTER_Z,
  THICKNESS,
  PARTS,
  LEG_WIDTH,
  LEG_DEPTH,
  LEG_HEIGHT,
  KNEE_Y_MIN,
  KNEE_Y_MAX,
  TORSO_Y_MIN,
  TORSO_Y_MAX,
  TORSO_X_HALF,
  TORSO_Z_HALF,
  ARM_Y_MIN,
  ARM_Y_MAX,
  ARM_X_HALF_OUTER,
  ARM_X_HALF_INNER,
  ARM_Z_HALF,
  CANNON_Y_MIN,
  CANNON_Y_MAX,
  CANNON_X_OFFSET_MIN,
  CANNON_X_OFFSET_MAX,
  CANNON_Z_OFFSET_MIN,
  CANNON_Z_OFFSET_MAX,
  HEAD_Y_MIN,
  HEAD_Y_MAX,
  HEAD_X_HALF,
  HEAD_Z_HALF,
  DISK_RADIUS,
  DISK_CENTER_Y,
  DISK_CENTER_Z_OFFSET,
} from './voxel-enemy.constants';

/**
 * Build a deterministic voxel enemy descriptor.
 *
 * The returned grid is sparse: only occupied voxels are stored, but every
 * voxel carries an albedo color, emissive flag, and alpha value so the
 * snapshot renderer can paint it directly.
 *
 * @param accentColor - Optional override for the accent palette. Defaults
 *   to Ares Red (`#DD2200`).
 * @returns A 64×192×64 voxel grid describing the enemy robot.
 *
 * @example
 * ```ts
 * const grid = buildVoxelEnemy('#FFAA00');
 * console.log(grid.height); // 192
 * console.log(grid.palette.accent); // '#FFAA00'
 * ```
 */
export function buildVoxelEnemy(accentColor?: string): VoxelGrid {
  const accent = accentColor ?? DEFAULT_ACCENT;
  const palette: VoxelPalette = {
    accent,
    neon: NEON_WHITE,
    suit: DARK_SUIT,
    dark: DEEP_SHADOW,
    damage: DAMAGE_FLASH,
  };

  const voxelMap = new Map<string, Voxel>();

  // Step 1: Build the lower body and torso core.
  buildLegs(voxelMap, palette);
  buildTorso(voxelMap, palette);

  // Step 2: Add arms and the right-front chest cannon.
  buildArms(voxelMap, palette);
  buildCannon(voxelMap, palette);

  // Step 3: Add the head, front eye stripe, and back identity disk.
  buildHead(voxelMap, palette);
  buildBackDisk(voxelMap, palette);

  // Deterministic ordering by spatial coordinates.
  const voxels = Array.from(voxelMap.values()).toSorted((a, b) => {
    if (a.x !== b.x) return a.x - b.x;
    if (a.y !== b.y) return a.y - b.y;
    return a.z - b.z;
  });

  return {
    width: GRID_WIDTH,
    height: GRID_HEIGHT,
    depth: GRID_DEPTH,
    voxels,
    parts: PARTS.toSorted(),
    palette,
    thickness: THICKNESS,
  };
}

/**
 * Add a single voxel to the map, replacing any previous occupant.
 */
function putVoxel(
  map: Map<string, Voxel>,
  x: number,
  y: number,
  z: number,
  part: string,
  material: MaterialSlot,
  palette: VoxelPalette,
): void {
  const key = `${x},${y},${z}`;
  const color = resolveColor(material, palette);
  const emissive =
    material === 'neon' || material === 'accent' || material === 'damage';
  map.set(key, {
    x,
    y,
    z,
    part,
    material,
    r: color.r,
    g: color.g,
    b: color.b,
    emissive,
    alpha: 1.0,
  });
}

/**
 * Resolve a material slot to an RGB color using the active palette.
 *
 * Exported so coverage can exercise every material branch, including the
 * reserved damage flash slot that is not yet used by the default enemy mesh.
 *
 * @internal
 */
export function resolveColor(
  material: MaterialSlot,
  palette: VoxelPalette,
): { r: number; g: number; b: number } {
  switch (material) {
    case 'accent':
      return hexToRgb(palette.accent);
    case 'neon':
      return hexToRgb(palette.neon);
    case 'suit':
      return hexToRgb(palette.suit);
    case 'dark':
      return hexToRgb(palette.dark);
    case 'damage':
      return hexToRgb(palette.damage);
  }
}

/**
 * Convert a `#RRGGBB` hex string to 0–255 RGB components.
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const sanitized = hex.startsWith('#') ? hex.slice(1) : hex;
  const parsed = Number.parseInt(sanitized, 16);
  return {
    r: (parsed >> 16) & 0xff,
    g: (parsed >> 8) & 0xff,
    b: parsed & 0xff,
  };
}

/**
 * Fill the surface (shell) of an axis-aligned box with voxels.
 *
 * Only the boundary voxels of each box are kept. This dramatically reduces
 * the voxel count while preserving the outer silhouette, which is what the
 * orthographic snapshot renderer actually paints.
 */
function fillBox(
  map: Map<string, Voxel>,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  zMin: number,
  zMax: number,
  part: string,
  material: MaterialSlot,
  palette: VoxelPalette,
): void {
  for (let x = xMin; x <= xMax; x++) {
    for (let y = yMin; y <= yMax; y++) {
      for (let z = zMin; z <= zMax; z++) {
        if (
          x === xMin ||
          x === xMax ||
          y === yMin ||
          y === yMax ||
          z === zMin ||
          z === zMax
        ) {
          putVoxel(map, x, y, z, part, material, palette);
        }
      }
    }
  }
}

/**
 * Build the legs as two dark-suit pillars with accent knee caps.
 */
function buildLegs(map: Map<string, Voxel>, palette: VoxelPalette): void {
  const leftLegX = GRID_CENTER_X - 13;
  const rightLegX = GRID_CENTER_X + 8;

  // Left leg.
  fillBox(
    map,
    leftLegX,
    leftLegX + LEG_WIDTH,
    0,
    LEG_HEIGHT,
    GRID_CENTER_Z - LEG_DEPTH / 2,
    GRID_CENTER_Z + LEG_DEPTH / 2,
    'legs',
    'suit',
    palette,
  );
  // Right leg.
  fillBox(
    map,
    rightLegX,
    rightLegX + LEG_WIDTH,
    0,
    LEG_HEIGHT,
    GRID_CENTER_Z - LEG_DEPTH / 2,
    GRID_CENTER_Z + LEG_DEPTH / 2,
    'legs',
    'suit',
    palette,
  );

  // Accent knee caps on the front of each shin.
  fillBox(
    map,
    leftLegX,
    leftLegX + LEG_WIDTH,
    KNEE_Y_MIN,
    KNEE_Y_MAX,
    GRID_CENTER_Z + LEG_DEPTH / 2,
    GRID_CENTER_Z + LEG_DEPTH / 2 + THICKNESS - 1,
    'legs',
    'accent',
    palette,
  );
  fillBox(
    map,
    rightLegX,
    rightLegX + LEG_WIDTH,
    KNEE_Y_MIN,
    KNEE_Y_MAX,
    GRID_CENTER_Z + LEG_DEPTH / 2,
    GRID_CENTER_Z + LEG_DEPTH / 2 + THICKNESS - 1,
    'legs',
    'accent',
    palette,
  );
}

/**
 * Build the torso as a dark box with vertical neon chest stripes.
 */
function buildTorso(map: Map<string, Voxel>, palette: VoxelPalette): void {
  // Main torso block.
  fillBox(
    map,
    GRID_CENTER_X - TORSO_X_HALF,
    GRID_CENTER_X + TORSO_X_HALF,
    TORSO_Y_MIN,
    TORSO_Y_MAX,
    GRID_CENTER_Z - TORSO_Z_HALF,
    GRID_CENTER_Z + TORSO_Z_HALF,
    'torso',
    'suit',
    palette,
  );

  // Vertical neon chest stripes on the front face.
  const frontZ = GRID_CENTER_Z + TORSO_Z_HALF;
  [-5, 5].forEach((offset) => {
    fillBox(
      map,
      GRID_CENTER_X + offset,
      GRID_CENTER_X + offset + 1,
      TORSO_Y_MIN + 10,
      TORSO_Y_MAX - 8,
      frontZ,
      frontZ + THICKNESS - 1,
      'torso',
      'neon',
      palette,
    );
  });
}

/**
 * Build the arms as dark sleeves with neon trim.
 */
function buildArms(map: Map<string, Voxel>, palette: VoxelPalette): void {
  // Left arm.
  fillBox(
    map,
    GRID_CENTER_X - ARM_X_HALF_OUTER,
    GRID_CENTER_X - ARM_X_HALF_INNER,
    ARM_Y_MIN,
    ARM_Y_MAX,
    GRID_CENTER_Z - ARM_Z_HALF,
    GRID_CENTER_Z + ARM_Z_HALF,
    'arms',
    'suit',
    palette,
  );
  // Right arm.
  fillBox(
    map,
    GRID_CENTER_X + ARM_X_HALF_INNER,
    GRID_CENTER_X + ARM_X_HALF_OUTER,
    ARM_Y_MIN,
    ARM_Y_MAX,
    GRID_CENTER_Z - ARM_Z_HALF,
    GRID_CENTER_Z + ARM_Z_HALF,
    'arms',
    'suit',
    palette,
  );

  // Neon shoulder trim on both arms.
  fillBox(
    map,
    GRID_CENTER_X - ARM_X_HALF_OUTER,
    GRID_CENTER_X - ARM_X_HALF_INNER,
    ARM_Y_MAX - THICKNESS + 1,
    ARM_Y_MAX,
    GRID_CENTER_Z - ARM_Z_HALF,
    GRID_CENTER_Z + ARM_Z_HALF,
    'arms',
    'neon',
    palette,
  );
  fillBox(
    map,
    GRID_CENTER_X + ARM_X_HALF_INNER,
    GRID_CENTER_X + ARM_X_HALF_OUTER,
    ARM_Y_MAX - THICKNESS + 1,
    ARM_Y_MAX,
    GRID_CENTER_Z - ARM_Z_HALF,
    GRID_CENTER_Z + ARM_Z_HALF,
    'arms',
    'neon',
    palette,
  );
}

/**
 * Build a chest-height cannon projecting forward from the right arm.
 */
function buildCannon(map: Map<string, Voxel>, palette: VoxelPalette): void {
  const cannonXMin = GRID_CENTER_X + CANNON_X_OFFSET_MIN;
  const cannonXMax = GRID_CENTER_X + CANNON_X_OFFSET_MAX;
  const cannonZMin = GRID_CENTER_Z + CANNON_Z_OFFSET_MIN;
  const cannonZMax = GRID_CENTER_Z + CANNON_Z_OFFSET_MAX;

  // Barrel accent body.
  fillBox(
    map,
    cannonXMin,
    cannonXMax,
    CANNON_Y_MIN,
    CANNON_Y_MAX,
    cannonZMin,
    cannonZMax,
    'cannon',
    'accent',
    palette,
  );

  // Dark bore channel through the center.
  fillBox(
    map,
    cannonXMax - THICKNESS + 1,
    cannonXMax,
    CANNON_Y_MIN + 2,
    CANNON_Y_MAX - 2,
    cannonZMin + 2,
    cannonZMax - 2,
    'cannon',
    'dark',
    palette,
  );

  // Neon muzzle ring at the front.
  fillBox(
    map,
    cannonXMax,
    cannonXMax + 1,
    CANNON_Y_MIN,
    CANNON_Y_MAX,
    cannonZMax - 1,
    cannonZMax,
    'cannon',
    'neon',
    palette,
  );
}

/**
 * Build the head/helmet with an eye stripe only on the front face.
 */
function buildHead(map: Map<string, Voxel>, palette: VoxelPalette): void {
  // Helmet shell.
  fillBox(
    map,
    GRID_CENTER_X - HEAD_X_HALF,
    GRID_CENTER_X + HEAD_X_HALF,
    HEAD_Y_MIN,
    HEAD_Y_MAX,
    GRID_CENTER_Z - HEAD_Z_HALF,
    GRID_CENTER_Z + HEAD_Z_HALF,
    'head',
    'suit',
    palette,
  );

  // Front eye stripe (horizontal neon band across the helmet front).
  const frontZ = GRID_CENTER_Z + HEAD_Z_HALF;
  fillBox(
    map,
    GRID_CENTER_X - HEAD_X_HALF,
    GRID_CENTER_X + HEAD_X_HALF,
    HEAD_Y_MIN + 16,
    HEAD_Y_MIN + 20,
    frontZ,
    frontZ + THICKNESS - 1,
    'head',
    'neon',
    palette,
  );

  // Accent crest on the top of the helmet.
  fillBox(
    map,
    GRID_CENTER_X - 3,
    GRID_CENTER_X + 3,
    HEAD_Y_MAX - THICKNESS + 1,
    HEAD_Y_MAX,
    GRID_CENTER_Z - 2,
    GRID_CENTER_Z + 2,
    'head',
    'accent',
    palette,
  );
}

/**
 * Build a back-mounted identity disk between the shoulders.
 *
 * The disk uses concentric rings: accent outer ring, white inner ring,
 * and suit-colored center.
 */
function buildBackDisk(map: Map<string, Voxel>, palette: VoxelPalette): void {
  const diskCenterX = GRID_CENTER_X;
  const diskCenterY = DISK_CENTER_Y;
  const diskCenterZ = GRID_CENTER_Z + DISK_CENTER_Z_OFFSET;

  for (let y = diskCenterY - DISK_RADIUS; y <= diskCenterY + DISK_RADIUS; y++) {
    for (
      let z = diskCenterZ - DISK_RADIUS;
      z <= diskCenterZ + DISK_RADIUS;
      z++
    ) {
      const dy = y - diskCenterY;
      const dz = z - diskCenterZ;
      const distance = Math.sqrt(dy * dy + dz * dz);

      if (distance > DISK_RADIUS) {
        continue;
      }

      let material: MaterialSlot;
      if (distance >= DISK_RADIUS - THICKNESS) {
        material = 'accent';
      } else if (distance >= DISK_RADIUS - 2 * THICKNESS) {
        material = 'neon';
      } else {
        material = 'suit';
      }

      for (let t = 0; t < THICKNESS; t++) {
        putVoxel(
          map,
          diskCenterX - Math.floor(THICKNESS / 2) + t,
          y,
          z,
          'back disk',
          material,
          palette,
        );
      }
    }
  }
}
