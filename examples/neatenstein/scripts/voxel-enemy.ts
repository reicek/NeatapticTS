/**
 * Voxel enemy descriptor for the Neatenstein NGE demo.
 *
 * Builds a deterministic, sparse N×192×M voxel grid describing a humanoid
 * combat robot: head, torso, arms, legs, a chest-height right-arm cannon,
 * and a back-mounted identity disk. The default accent is Ares Red and can
 * be swapped for other neon hues.
 */

/** Canonical palette colors used across all enemy variants. */
const DEFAULT_ACCENT = '#DD2200';
const NEON_WHITE = '#FBFFFF';
const DARK_SUIT = '#121418';
const DEEP_SHADOW = '#0A0B0E';
const DAMAGE_FLASH = '#FF3300';

/** Grid dimensions derived from the 192×192 approved PNG silhouettes. */
const GRID_WIDTH = 64;
const GRID_HEIGHT = 192;
const GRID_DEPTH = 64;
const GRID_CENTER_X = GRID_WIDTH / 2;
const GRID_CENTER_Z = GRID_DEPTH / 2;

/** Shared edge/neon/disk thickness in voxels (must stay within 2–4). */
const THICKNESS = 3;

/** Body part names in the descriptor. */
const PARTS = ['head', 'torso', 'arms', 'legs', 'cannon', 'back disk'] as const;

/** Material slots referenced by the snapshot renderer. */
type MaterialSlot = 'accent' | 'neon' | 'suit' | 'dark' | 'damage';

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
  const legWidth = 5;
  const legDepth = 8;
  const legHeight = 52;

  // Left leg.
  fillBox(
    map,
    leftLegX,
    leftLegX + legWidth,
    0,
    legHeight,
    GRID_CENTER_Z - legDepth / 2,
    GRID_CENTER_Z + legDepth / 2,
    'legs',
    'suit',
    palette,
  );
  // Right leg.
  fillBox(
    map,
    rightLegX,
    rightLegX + legWidth,
    0,
    legHeight,
    GRID_CENTER_Z - legDepth / 2,
    GRID_CENTER_Z + legDepth / 2,
    'legs',
    'suit',
    palette,
  );

  // Accent knee caps on the front of each shin.
  fillBox(
    map,
    leftLegX,
    leftLegX + legWidth,
    18,
    24,
    GRID_CENTER_Z + legDepth / 2,
    GRID_CENTER_Z + legDepth / 2 + THICKNESS - 1,
    'legs',
    'accent',
    palette,
  );
  fillBox(
    map,
    rightLegX,
    rightLegX + legWidth,
    18,
    24,
    GRID_CENTER_Z + legDepth / 2,
    GRID_CENTER_Z + legDepth / 2 + THICKNESS - 1,
    'legs',
    'accent',
    palette,
  );
}

/**
 * Build the torso as a dark box with vertical neon chest stripes.
 */
function buildTorso(map: Map<string, Voxel>, palette: VoxelPalette): void {
  const torsoYMin = 54;
  const torsoYMax = 108;
  const torsoXHalf = 14;
  const torsoZHalf = 8;

  // Main torso block.
  fillBox(
    map,
    GRID_CENTER_X - torsoXHalf,
    GRID_CENTER_X + torsoXHalf,
    torsoYMin,
    torsoYMax,
    GRID_CENTER_Z - torsoZHalf,
    GRID_CENTER_Z + torsoZHalf,
    'torso',
    'suit',
    palette,
  );

  // Vertical neon chest stripes on the front face.
  const frontZ = GRID_CENTER_Z + torsoZHalf;
  [-5, 5].forEach((offset) => {
    fillBox(
      map,
      GRID_CENTER_X + offset,
      GRID_CENTER_X + offset + 1,
      torsoYMin + 10,
      torsoYMax - 8,
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
  const armYMin = 56;
  const armYMax = 92;
  const armXHalfOuter = 22;
  const armXHalfInner = 15;
  const armZHalf = 5;

  // Left arm.
  fillBox(
    map,
    GRID_CENTER_X - armXHalfOuter,
    GRID_CENTER_X - armXHalfInner,
    armYMin,
    armYMax,
    GRID_CENTER_Z - armZHalf,
    GRID_CENTER_Z + armZHalf,
    'arms',
    'suit',
    palette,
  );
  // Right arm.
  fillBox(
    map,
    GRID_CENTER_X + armXHalfInner,
    GRID_CENTER_X + armXHalfOuter,
    armYMin,
    armYMax,
    GRID_CENTER_Z - armZHalf,
    GRID_CENTER_Z + armZHalf,
    'arms',
    'suit',
    palette,
  );

  // Neon shoulder trim on both arms.
  fillBox(
    map,
    GRID_CENTER_X - armXHalfOuter,
    GRID_CENTER_X - armXHalfInner,
    armYMax - THICKNESS + 1,
    armYMax,
    GRID_CENTER_Z - armZHalf,
    GRID_CENTER_Z + armZHalf,
    'arms',
    'neon',
    palette,
  );
  fillBox(
    map,
    GRID_CENTER_X + armXHalfInner,
    GRID_CENTER_X + armXHalfOuter,
    armYMax - THICKNESS + 1,
    armYMax,
    GRID_CENTER_Z - armZHalf,
    GRID_CENTER_Z + armZHalf,
    'arms',
    'neon',
    palette,
  );
}

/**
 * Build a chest-height cannon projecting forward from the right arm.
 */
function buildCannon(map: Map<string, Voxel>, palette: VoxelPalette): void {
  const cannonYMin = 70;
  const cannonYMax = 78;
  const cannonXMin = GRID_CENTER_X + 16;
  const cannonXMax = GRID_CENTER_X + 30;
  const cannonZMin = GRID_CENTER_Z - 2;
  const cannonZMax = GRID_CENTER_Z + 12;

  // Barrel accent body.
  fillBox(
    map,
    cannonXMin,
    cannonXMax,
    cannonYMin,
    cannonYMax,
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
    cannonYMin + 2,
    cannonYMax - 2,
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
    cannonYMin,
    cannonYMax,
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
  const headYMin = 112;
  const headYMax = 142;
  const headXHalf = 10;
  const headZHalf = 8;

  // Helmet shell.
  fillBox(
    map,
    GRID_CENTER_X - headXHalf,
    GRID_CENTER_X + headXHalf,
    headYMin,
    headYMax,
    GRID_CENTER_Z - headZHalf,
    GRID_CENTER_Z + headZHalf,
    'head',
    'suit',
    palette,
  );

  // Front eye stripe (horizontal neon band across the helmet front).
  const frontZ = GRID_CENTER_Z + headZHalf;
  fillBox(
    map,
    GRID_CENTER_X - headXHalf,
    GRID_CENTER_X + headXHalf,
    headYMin + 16,
    headYMin + 20,
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
    headYMax - THICKNESS + 1,
    headYMax,
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
  const diskCenterY = 102;
  const diskCenterZ = GRID_CENTER_Z - 12;
  const diskRadius = 12;

  for (let y = diskCenterY - diskRadius; y <= diskCenterY + diskRadius; y++) {
    for (let z = diskCenterZ - diskRadius; z <= diskCenterZ + diskRadius; z++) {
      const dy = y - diskCenterY;
      const dz = z - diskCenterZ;
      const distance = Math.sqrt(dy * dy + dz * dz);

      if (distance > diskRadius) {
        continue;
      }

      let material: MaterialSlot;
      if (distance >= diskRadius - THICKNESS) {
        material = 'accent';
      } else if (distance >= diskRadius - 2 * THICKNESS) {
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
