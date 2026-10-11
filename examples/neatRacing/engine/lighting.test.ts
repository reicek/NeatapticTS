/**
 * Red-first contract tests for the NeatRacing lighting model and car pass
 * (slice P1-S4-D; spec sections 6.5-6.6 + 7.3.1 in
 * `plans/neatracing-voxel-racer.spec.md`).
 *
 * Locks the lighting contract: a single directional sun, hemisphere ambient
 * mixing ground/sky over the normal's up component, the exp2 distance fog
 * transmittance (1 at the camera, 0.5 at 1/density, ~0.53 at half the 320 m
 * draw distance), glow-driven emissive with the LD-23 black exception, the
 * seven shared lighting uniform names, and a GLSL chunk seam the car pass
 * embeds so the TypeScript mirror and the shader cannot drift.
 *
 * Locks the car pass contract (spec 6.5/7.3.1 + G9/AC-014): two-sided bins
 * meshes (CULL_FACE disabled), blob shadows drawn first at world y + 0.01 m
 * with depth-write off and SRC_ALPHA blending, cars sorted far -> near by
 * camera distance, the four wheel sockets mounted through the G9 wheel-mount
 * contract, draw-time team remap (uTeamColor + NEATRACING_TEAM_REMAP_SLOT,
 * never baked into vertex data), the warmup seam under registry key 'car'
 * (index 4), and dispose releasing both programs plus all three buffers.
 *
 * The Jest project runs in a plain Node environment, so every contract is
 * exercised through pure functions plus the recording mock WebGL2 harness
 * pattern from `pulse.pass.test.ts`; GLSL source strings are never asserted
 * directly.
 */
import { describe, expect, it } from '@jest/globals';
import { NEATRACING_WARMED_PROGRAM_KEYS } from './context.service';
import {
  NEATRACING_FOG_COLOR,
  NEATRACING_FOG_DENSITY_1_PER_M,
  NEATRACING_FOG_HALF_TRANSMITTANCE_DISTANCE_M,
  NEATRACING_HEMISPHERE_GROUND_COLOR,
  NEATRACING_HEMISPHERE_SKY_COLOR,
  NEATRACING_LIGHTING_GLSL_CHUNK,
  NEATRACING_LIGHTING_UNIFORM_NAMES,
  NEATRACING_SUN_COLOR,
  NEATRACING_SUN_DIRECTION,
  computeFogFactor,
  computeHemisphereAmbient,
  computeSunDiffuse,
  shadeLitSurface,
} from './lighting';
import {
  CAR_MESH_FLOATS_PER_VERTEX,
  CAR_MESH_VERTEX_STRIDE_BYTES,
  NEATRACING_BLOB_SHADOW_BOUNDS_SCALE,
  NEATRACING_BLOB_SHADOW_Y_OFFSET_M,
  createCarPass,
  expandVoxelBinModelToCarMesh,
  toCarWarmupProgram,
} from './passes/car.pass';
import { NEATRACING_DRAW_DISTANCE_M } from './passes/grid.pass';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import { NEATRACING_TEAM_REMAP_SLOT } from '../voxel/team-remap';
import { NEATRACING_WHEEL_HUB_HEIGHT_M } from '../voxel/wheel.mount.utils';
import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  type VoxelPaletteEntry,
} from '../voxel/model.schema';
import type { VoxelBinModel, VoxelBinQuad } from '../voxel/bin.codec';

/** `gl.BLEND` enum value (0x0BE2) asserted on the mock draw state. */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.CULL_FACE` enum value (0x0B44) asserted for two-sided bins meshes. */
const MOCK_GL_CULL_FACE = 0x0b44;

/** `gl.DEPTH_TEST` enum value (0x0B71) asserted on the mock draw state. */
const MOCK_GL_DEPTH_TEST = 0x0b71;

/** `gl.SRC_ALPHA` enum value (0x0302) asserted on the shadow blend call. */
const MOCK_GL_SRC_ALPHA = 0x0302;

/** `gl.ONE_MINUS_SRC_ALPHA` enum value (0x0303) for the shadow blend. */
const MOCK_GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.TRIANGLES` enum value (4) asserted on every car draw call. */
const MOCK_GL_TRIANGLES = 4;

/** `gl.ARRAY_BUFFER` enum value (0x8892) for bindBuffer target pins. */
const MOCK_GL_ARRAY_BUFFER = 0x8892;

/** Mock buffer ids assigned in createCarPass creation order: body, wheel, shadow. */
const CAR_MOCK_BODY_BUFFER_ID = 1;
const CAR_MOCK_WHEEL_BUFFER_ID = 2;
const CAR_MOCK_SHADOW_BUFFER_ID = 3;

/** Column-major identity view-projection used by the mock draw frames. */
const IDENTITY_VIEW_PROJECTION_FIXTURE: readonly number[] = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

/** Voxel world size alias for the fixture position math (4.5 / 128 m). */
const VOXEL_WORLD_SIZE_M = NEATRACING_VOXEL_WORLD_SIZE;

/**
 * Car-body fixture palette: EMPTY sentinel, the team-livery slot entry the
 * draw-time remap swaps (kept authored here), a black entry with a glow byte
 * (LD-23 must force emissive to 0), a glowing solid, and a normal solid.
 */
const CAR_FIXTURE_PALETTE: readonly VoxelPaletteEntry[] = [
  [...VOXEL_EMPTY_PALETTE_ENTRY] as VoxelPaletteEntry,
  [180, 40, 40, 255, 0] as VoxelPaletteEntry,
  [0, 0, 0, 255, 200] as VoxelPaletteEntry,
  [200, 20, 200, 255, 120] as VoxelPaletteEntry,
  [120, 160, 90, 255, 0] as VoxelPaletteEntry,
];

/**
 * Car-body fixture quads covering all three normal axes: a large side panel
 * (normal solid), a roof plate (black-with-glow), a glowing front plate, and
 * a team-slot panel. Positions are MIN lattice corners in the 4-cubed grid.
 */
const CAR_FIXTURE_QUADS: readonly VoxelBinQuad[] = [
  { position: [0, 0, 0], normalAxis: 0, width: 2, height: 4, paletteIndex: 4 },
  { position: [1, 1, 2], normalAxis: 1, width: 3, height: 2, paletteIndex: 2 },
  { position: [2, 0, 3], normalAxis: 2, width: 2, height: 3, paletteIndex: 3 },
  { position: [0, 1, 0], normalAxis: 0, width: 1, height: 1, paletteIndex: 1 },
];

/** Synthetic car-body bin (4-cubed grid, 4 quads, mixed material entries). */
const CAR_FIXTURE_BIN: VoxelBinModel = {
  dims: [4, 4, 4],
  palette: CAR_FIXTURE_PALETTE,
  quads: CAR_FIXTURE_QUADS,
};

/** One car pose inside a draw frame (the draw-state car entry shape). */
interface TestCarPose {
  readonly worldX: number;
  readonly worldZ: number;
  readonly yawRad: number;
  readonly teamColorId: number;
  readonly wheelSpinRad: number;
}

/**
 * Builds a draw frame around the identity projection with the given cars.
 *
 * @param cars - Car poses to draw, in the caller's array order.
 * @param cameraWorld - Camera world position (defaults to y=1 above origin).
 * @returns A draw-state literal accepted by the car pass draw path.
 */
function makeDrawFrame(
  cars: readonly TestCarPose[],
  cameraWorld: readonly [number, number, number] = [0, 1, 0],
): {
  readonly viewProjection: Float32Array;
  readonly cameraWorld: readonly [number, number, number];
  readonly cars: readonly TestCarPose[];
} {
  return {
    viewProjection: new Float32Array(IDENTITY_VIEW_PROJECTION_FIXTURE),
    cameraWorld,
    cars,
  };
}

/**
 * Reads the 9 floats of one expanded vertex: position(3), color(3),
 * normalAxis(1), emissive(1), paletteIndex(1).
 *
 * @param mesh - Expanded car mesh under test.
 * @param vertexIndex - Vertex slot to read.
 * @returns The vertex floats copied into a plain array.
 */
function readCarVertex(
  mesh: ReturnType<typeof expandVoxelBinModelToCarMesh>,
  vertexIndex: number,
): readonly number[] {
  return Array.from(
    mesh.vertexData.slice(
      vertexIndex * CAR_MESH_FLOATS_PER_VERTEX,
      (vertexIndex + 1) * CAR_MESH_FLOATS_PER_VERTEX,
    ),
  );
}

/** Filters recorded matrix feeds down to one uniform name. */
function matrixFeedsNamed(
  harness: CarMockGlHarness,
  name: string,
): readonly CarMockMatrixFeed[] {
  return harness.matrixFeeds.filter((feed) => feed.name === name);
}

/** Filters recorded uniform1f feeds down to one uniform name. */
function uniform1fFeedsNamed(
  harness: CarMockGlHarness,
  name: string,
): readonly CarMockUniform1fFeed[] {
  return harness.uniform1fFeeds.filter((feed) => feed.name === name);
}

/** Filters recorded uniform3f feeds down to one uniform name. */
function uniform3fFeedsNamed(
  harness: CarMockGlHarness,
  name: string,
): readonly CarMockUniform3fFeed[] {
  return harness.uniform3fFeeds.filter((feed) => feed.name === name);
}

describe('neatRacing lighting model constants', () => {
  it('exposes a frozen, unit-length, deterministic sun direction', () => {
    const [x, y, z] = NEATRACING_SUN_DIRECTION;
    expect(Math.hypot(x, y, z)).toBeCloseTo(1, 12);
    expect(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)).toBe(
      true,
    );
    expect(Object.isFrozen(NEATRACING_SUN_DIRECTION)).toBe(true);
    expect(NEATRACING_SUN_DIRECTION).toBe(NEATRACING_SUN_DIRECTION);
    expect(computeSunDiffuse(NEATRACING_SUN_DIRECTION)).toBe(
      computeSunDiffuse(NEATRACING_SUN_DIRECTION),
    );
  });

  it('normalizes the sun, sky, ground, and fog colors into [0, 1]', () => {
    expect(NEATRACING_SUN_COLOR).toEqual([255 / 255, 246 / 255, 229 / 255]);
    expect(NEATRACING_HEMISPHERE_SKY_COLOR).toEqual([
      140 / 255,
      180 / 255,
      255 / 255,
    ]);
    expect(NEATRACING_HEMISPHERE_GROUND_COLOR).toEqual([
      60 / 255,
      55 / 255,
      50 / 255,
    ]);
    expect(NEATRACING_FOG_COLOR).toEqual([168 / 255, 195 / 255, 225 / 255]);
  });

  it('pins the fog density at 0.006 1/m against the draw-distance tuning', () => {
    expect(NEATRACING_FOG_DENSITY_1_PER_M).toBe(0.006);
    expect(NEATRACING_FOG_HALF_TRANSMITTANCE_DISTANCE_M).toBeCloseTo(
      1 / 0.006,
      12,
    );
    // The spec tunes density 0.006 so the 50% fade lands NEAR half the 320m
    // draw distance: 1/0.006 = 166.7m is ~4.2% off exact 160m (0.00625 would
    // be exact), so assert the relative drift instead of < 0.5m.
    const halfDrawDistanceM = NEATRACING_DRAW_DISTANCE_M / 2;
    const halfTransmittanceDriftFraction =
      Math.abs(
        NEATRACING_FOG_HALF_TRANSMITTANCE_DISTANCE_M - halfDrawDistanceM,
      ) / halfDrawDistanceM;
    expect(halfTransmittanceDriftFraction).toBeLessThan(0.05);
  });

  it('exports the seven shared lighting uniform names in feed order', () => {
    expect(NEATRACING_LIGHTING_UNIFORM_NAMES).toEqual([
      'uSunDirection',
      'uSunColor',
      'uHemisphereSky',
      'uHemisphereGround',
      'uFogDensity',
      'uFogColor',
      'uCameraWorld',
    ]);
  });

  it('exposes a non-empty GLSL chunk so the shader cannot drift from the TS mirror', () => {
    expect(typeof NEATRACING_LIGHTING_GLSL_CHUNK).toBe('string');
    expect(NEATRACING_LIGHTING_GLSL_CHUNK.length).toBeGreaterThan(0);
  });
});

describe('exp2 fog transmittance', () => {
  it('is fully transmissive at the camera (d = 0)', () => {
    expect(computeFogFactor(0)).toBe(1);
  });

  it('hits exactly half transmittance at 1/density', () => {
    expect(computeFogFactor(1 / NEATRACING_FOG_DENSITY_1_PER_M)).toBeCloseTo(
      0.5,
      12,
    );
  });

  it('sits near 0.53 at half the 320 m draw distance', () => {
    const halfDrawDistanceM = NEATRACING_DRAW_DISTANCE_M / 2;
    const value = computeFogFactor(halfDrawDistanceM);
    expect(value).toBeGreaterThan(0.5);
    expect(value).toBeLessThan(0.55);
    expect(value).toBeCloseTo(0.53, 2);
  });

  it('decreases monotonically with distance across the draw distance', () => {
    const factor0 = computeFogFactor(0);
    const factor80 = computeFogFactor(80);
    const factor160 = computeFogFactor(160);
    const factor320 = computeFogFactor(NEATRACING_DRAW_DISTANCE_M);
    expect(factor0).toBeGreaterThan(factor80);
    expect(factor80).toBeGreaterThan(factor160);
    expect(factor160).toBeGreaterThan(factor320);
  });

  it('honors an explicit density override for tuning probes', () => {
    expect(computeFogFactor(50, 0.01)).toBeCloseTo(Math.pow(2, -0.25), 12);
  });
});

describe('hemisphere ambient mixing', () => {
  it('returns pure sky at a straight-up normal (n.y = 1)', () => {
    expect(computeHemisphereAmbient(1)).toEqual(
      NEATRACING_HEMISPHERE_SKY_COLOR,
    );
  });

  it('returns pure ground at a straight-down normal (n.y = -1)', () => {
    expect(computeHemisphereAmbient(-1)).toEqual(
      NEATRACING_HEMISPHERE_GROUND_COLOR,
    );
  });

  it('returns the sky/ground midpoint at a horizontal normal (n.y = 0)', () => {
    const expected = [
      (NEATRACING_HEMISPHERE_SKY_COLOR[0] +
        NEATRACING_HEMISPHERE_GROUND_COLOR[0]) /
        2,
      (NEATRACING_HEMISPHERE_SKY_COLOR[1] +
        NEATRACING_HEMISPHERE_GROUND_COLOR[1]) /
        2,
      (NEATRACING_HEMISPHERE_SKY_COLOR[2] +
        NEATRACING_HEMISPHERE_GROUND_COLOR[2]) /
        2,
    ];
    expect(computeHemisphereAmbient(0)).toEqual(expected);
  });

  it('mixes per channel with the mirrored clamp(normal.y * 0.5 + 0.5) weight', () => {
    for (const normalY of [-0.3, 0.25, 0.7]) {
      const mixFactor = Math.min(1, Math.max(0, normalY * 0.5 + 0.5));
      const resolved = computeHemisphereAmbient(normalY);
      const expected = NEATRACING_HEMISPHERE_GROUND_COLOR.map(
        (groundChannel: number, channel: number) =>
          groundChannel * (1 - mixFactor) +
          NEATRACING_HEMISPHERE_SKY_COLOR[channel] * mixFactor,
      );
      expect(resolved[0]).toBeCloseTo(expected[0], 12);
      expect(resolved[1]).toBeCloseTo(expected[1], 12);
      expect(resolved[2]).toBeCloseTo(expected[2], 12);
    }
  });
});

describe('sun diffuse clamp', () => {
  it('peaks at 1 when the surface normal faces the sun', () => {
    expect(computeSunDiffuse(NEATRACING_SUN_DIRECTION)).toBeCloseTo(1, 12);
  });

  it('clamps the back-facing dot product to zero', () => {
    expect(
      computeSunDiffuse([
        -NEATRACING_SUN_DIRECTION[0],
        -NEATRACING_SUN_DIRECTION[1],
        -NEATRACING_SUN_DIRECTION[2],
      ]),
    ).toBe(0);
    expect(computeSunDiffuse([-1, 0, 0])).toBe(0);
  });

  it('mirrors max(0, dot(normal, sunDirection)) for arbitrary normals', () => {
    const normal = [0.3, -0.2, 0.9] as const;
    const expected = Math.max(
      0,
      normal[0] * NEATRACING_SUN_DIRECTION[0] +
        normal[1] * NEATRACING_SUN_DIRECTION[1] +
        normal[2] * NEATRACING_SUN_DIRECTION[2],
    );
    expect(computeSunDiffuse(normal)).toBeCloseTo(expected, 12);
  });
});

describe('full fragment shade mirror', () => {
  it('mirrors albedo * (hemisphere + sunColor * sunDiffuse) + albedo * emissive', () => {
    const albedo = [0.8, 0.6, 0.4] as const;
    const emissive = 120 / 255;
    const sunDiffuse = computeSunDiffuse(NEATRACING_SUN_DIRECTION);
    const hemisphere = computeHemisphereAmbient(NEATRACING_SUN_DIRECTION[1]);
    const shaded = shadeLitSurface(albedo, NEATRACING_SUN_DIRECTION, emissive);
    expect(shaded[0]).toBeCloseTo(
      albedo[0] * (hemisphere[0] + NEATRACING_SUN_COLOR[0] * sunDiffuse) +
        albedo[0] * emissive,
      12,
    );
    expect(shaded[1]).toBeCloseTo(
      albedo[1] * (hemisphere[1] + NEATRACING_SUN_COLOR[1] * sunDiffuse) +
        albedo[1] * emissive,
      12,
    );
    expect(shaded[2]).toBeCloseTo(
      albedo[2] * (hemisphere[2] + NEATRACING_SUN_COLOR[2] * sunDiffuse) +
        albedo[2] * emissive,
      12,
    );
  });

  it('adds exactly albedo * emissive when the emissive term turns on', () => {
    const albedo = [0.5, 0.25, 0.75] as const;
    const normal = [0, 1, 0] as const;
    const withoutEmissive = shadeLitSurface(albedo, normal, 0);
    const withEmissive = shadeLitSurface(albedo, normal, 0.5);
    expect(withEmissive[0] - withoutEmissive[0]).toBeCloseTo(
      albedo[0] * 0.5,
      12,
    );
    expect(withEmissive[1] - withoutEmissive[1]).toBeCloseTo(
      albedo[1] * 0.5,
      12,
    );
    expect(withEmissive[2] - withoutEmissive[2]).toBeCloseTo(
      albedo[2] * 0.5,
      12,
    );
  });

  it('keeps black albedo fully dark regardless of the emissive term (LD-23)', () => {
    const shaded = shadeLitSurface([0, 0, 0], NEATRACING_SUN_DIRECTION, 0.5);
    expect(shaded).toEqual([0, 0, 0]);
  });
});

describe('voxel bin to car mesh expansion (pure)', () => {
  it('emits 6 non-indexed vertices per quad on the 9-float stride', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    expect(mesh.vertexCount).toBe(CAR_FIXTURE_QUADS.length * 6);
    expect(mesh.vertexData.length).toBe(
      mesh.vertexCount * CAR_MESH_FLOATS_PER_VERTEX,
    );
    expect(CAR_MESH_FLOATS_PER_VERTEX).toBe(9);
    expect(CAR_MESH_VERTEX_STRIDE_BYTES).toBe(36);
  });

  it('maps minimum-lattice quads to model-space meters with the body floor at y=0 (axis 0)', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    const vertex0 = readCarVertex(mesh, 0);
    expect(vertex0[0]).toBeCloseTo((0 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(vertex0[1]).toBeCloseTo(0 * VOXEL_WORLD_SIZE_M, 12);
    expect(vertex0[2]).toBeCloseTo((0 - 2) * VOXEL_WORLD_SIZE_M, 12);
    // Color channels live in a Float32Array: binary32 rounding tolerance.
    expect(vertex0[3]).toBeCloseTo(120 / 255, 6);
    expect(vertex0[4]).toBeCloseTo(160 / 255, 6);
    expect(vertex0[5]).toBeCloseTo(90 / 255, 6);
    expect(vertex0[6]).toBe(0);
    expect(vertex0[7]).toBe(0);
    expect(vertex0[8]).toBe(4);
  });

  it('maps axis 1 widths along x and heights along z aligned with the shared mesher tangents', () => {
    // FACE_TANGENT_AXES[1] = [0, 2]: u runs along x, v runs along z. The
    // non-square fixture quad (width 3, height 2) exposes any u/v swap: a
    // transposed expansion stretches the quad to (height, width) instead of
    // (width, height), which is the render artifact source.
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    const corner0 = readCarVertex(mesh, 6);
    const cornerWidth = readCarVertex(mesh, 7);
    const cornerHeight = readCarVertex(mesh, 8);
    expect(corner0[0]).toBeCloseTo((1 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(corner0[1]).toBeCloseTo(1 * VOXEL_WORLD_SIZE_M, 12);
    expect(corner0[2]).toBeCloseTo((2 - 2) * VOXEL_WORLD_SIZE_M, 12);
    // Width (u) extends along x: corner x = pos0 + width.
    expect(cornerWidth[0]).toBeCloseTo((4 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(cornerWidth[2]).toBeCloseTo((2 - 2) * VOXEL_WORLD_SIZE_M, 12);
    // Height (v) extends along z: corner z = pos2 + height.
    // Vertex 8 is quad 1's corner index 2 — the (u, v) DIAGONAL corner in
    // the 00/10/11/01 expansion order — so both extents are applied:
    // x = pos0 + width, z = pos2 + height. Vertex 11 (triangle order
    // 0,1,2,0,2,3) is the pure height corner (u = 0): a transposed
    // expansion stretches it to (height, width) instead of
    // (width, height), the artifact source.
    expect(cornerHeight[0]).toBeCloseTo((1 + 3 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(cornerHeight[2]).toBeCloseTo((2 + 2 - 2) * VOXEL_WORLD_SIZE_M, 12);
    const cornerHeightOnly = readCarVertex(mesh, 11);
    expect(cornerHeightOnly[0]).toBeCloseTo((1 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(cornerHeightOnly[2]).toBeCloseTo(
      (2 + 2 - 2) * VOXEL_WORLD_SIZE_M,
      12,
    );
  });

  it('maps axis 2 width along x and height along y per the ascending tangent rule', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    const corner0 = readCarVertex(mesh, 12);
    const cornerWidth = readCarVertex(mesh, 13);
    const cornerHeight = readCarVertex(mesh, 14);
    expect(corner0[0]).toBeCloseTo((2 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(corner0[1]).toBeCloseTo(0 * VOXEL_WORLD_SIZE_M, 12);
    expect(corner0[2]).toBeCloseTo((3 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(cornerWidth[0]).toBeCloseTo((4 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(cornerWidth[2]).toBeCloseTo((3 - 2) * VOXEL_WORLD_SIZE_M, 12);
    expect(cornerHeight[1]).toBeCloseTo(3 * VOXEL_WORLD_SIZE_M, 12);
  });

  it('anchors the center variant so wheel disc rows pivot around the lattice midpoint', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN }, 'center');
    // Quad 0 spans gridY 0..4 on dimY = 4: midpoint offset 2 cells.
    const vertex0 = readCarVertex(mesh, 0);
    expect(vertex0[1]).toBeCloseTo((0 - 2) * VOXEL_WORLD_SIZE_M, 12);
    // Quad 2 (axis 2) top corner gridY = 3 -> (3 - 2) * voxelSize.
    const cornerHeight = readCarVertex(mesh, 14);
    expect(cornerHeight[1]).toBeCloseTo((3 - 2) * VOXEL_WORLD_SIZE_M, 12);
    // For the authored 32-row wheel lattice, row 0 lands at exactly
    // -NEATRACING_WHEEL_HUB_HEIGHT_M (the contact-patch contract, AC-015).
    const tireBottomM = (0 - 32 * 0.5) * VOXEL_WORLD_SIZE_M;
    expect(tireBottomM).toBeCloseTo(-NEATRACING_WHEEL_HUB_HEIGHT_M, 12);
  });

  it('resolves vertex colors from the material table with glow/255 emissive', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    const glowVertex = readCarVertex(mesh, 12);
    // Color/emissive channels live in a Float32Array: binary32 rounding, so
    // precision 6 (5e-7) instead of exact/12 comparisons.
    expect(glowVertex[3]).toBeCloseTo(200 / 255, 6);
    expect(glowVertex[4]).toBeCloseTo(20 / 255, 6);
    expect(glowVertex[5]).toBeCloseTo(200 / 255, 6);
    expect(glowVertex[7]).toBeCloseTo(120 / 255, 6);
  });

  it('forces the black palette entry to emissive 0 regardless of its glow byte (LD-23)', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    const blackVertex = readCarVertex(mesh, 6);
    expect(blackVertex[3]).toBe(0);
    expect(blackVertex[4]).toBe(0);
    expect(blackVertex[5]).toBe(0);
    expect(blackVertex[6]).toBe(1);
    expect(blackVertex[7]).toBe(0);
    expect(blackVertex[8]).toBe(2);
  });

  it('keeps the team-slot palette color unbaked for the draw-time remap (AC-014)', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    const teamVertex = readCarVertex(mesh, 18);
    expect(teamVertex[3]).toBeCloseTo(180 / 255, 6);
    expect(teamVertex[4]).toBeCloseTo(40 / 255, 6);
    expect(teamVertex[5]).toBeCloseTo(40 / 255, 6);
    expect(teamVertex[8]).toBe(NEATRACING_TEAM_REMAP_SLOT);
    expect(teamVertex[3]).not.toBeCloseTo(255 / 255, 12);
    expect(teamVertex[4]).not.toBeCloseTo(77 / 255, 12);
  });

  it('reports the XZ half-extent of the expanded vertices for blob shadow sizing', () => {
    const mesh = expandVoxelBinModelToCarMesh({ ...CAR_FIXTURE_BIN });
    expect(mesh.boundsHalfExtentXzM).toBeCloseTo(2 * VOXEL_WORLD_SIZE_M, 12);
  });
});

describe('car pass WebGL2 orchestration', () => {
  it('throws when viewProjection is not a 16-element matrix', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });
    const frame = makeDrawFrame([]);
    const truncatedProjection = new Float32Array(15);
    expect(() =>
      pass.draw({
        ...frame,
        viewProjection: truncatedProjection,
      }),
    ).toThrow(/viewProjection/);
  });

  it('draws one drawArrays call per car stage: shadow + body + 4 wheels', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
        { worldX: 0, worldZ: 60, yawRad: 0, teamColorId: 0, wheelSpinRad: 0 },
      ]),
    );

    expect(harness.calls.drawArrays).toBe(12);
    for (const draw of harness.drawArraysCalls) {
      expect(draw.mode).toBe(MOCK_GL_TRIANGLES);
      expect(draw.first).toBe(0);
    }
  });

  it('pins the exact glow state machine: shadows first with depth-write off', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(makeDrawFrame([]));

    expect(harness.stateOps).toEqual([
      { op: 'enable', args: [MOCK_GL_DEPTH_TEST] },
      { op: 'depthMask', args: [false] },
      { op: 'enable', args: [MOCK_GL_BLEND] },
      {
        op: 'blendFunc',
        args: [MOCK_GL_SRC_ALPHA, MOCK_GL_ONE_MINUS_SRC_ALPHA],
      },
      { op: 'depthMask', args: [true] },
      { op: 'disable', args: [MOCK_GL_BLEND] },
      { op: 'disable', args: [MOCK_GL_CULL_FACE] },
      { op: 'enable', args: [MOCK_GL_DEPTH_TEST] },
    ]);
    expect(harness.calls.drawArrays).toBe(0);
  });

  it('draws every shadow quad under depthMask(false) before every body/wheel draw', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
        { worldX: 0, worldZ: 60, yawRad: 0, teamColorId: 0, wheelSpinRad: 0 },
      ]),
    );

    const shadowDrawDepthMasks = harness.drawArraysCalls
      .slice(0, 2)
      .map((draw) => draw.depthMask);
    const bodyDrawDepthMasks = harness.drawArraysCalls
      .slice(2)
      .map((draw) => draw.depthMask);
    expect(shadowDrawDepthMasks).toEqual([false, false]);
    expect(bodyDrawDepthMasks.every((depthMask) => depthMask)).toBe(true);
  });

  it('renders bins-derived meshes two-sided with the blend pair (spec 7.3.1)', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
      ]),
    );

    expect(harness.stateOps).toContainEqual({
      op: 'disable',
      args: [MOCK_GL_CULL_FACE],
    });
    expect(harness.stateOps).toContainEqual({
      op: 'enable',
      args: [MOCK_GL_DEPTH_TEST],
    });
    expect(harness.stateOps).toContainEqual({
      op: 'blendFunc',
      args: [MOCK_GL_SRC_ALPHA, MOCK_GL_ONE_MINUS_SRC_ALPHA],
    });
  });

  it('feeds [0, 1]-normalized team colors and the remap slot per car (AC-014)', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
        { worldX: 4, worldZ: 0, yawRad: 0, teamColorId: 0, wheelSpinRad: 0 },
      ]),
    );

    const teamColorFeeds = uniform3fFeedsNamed(harness, 'uTeamColor');
    const slotFeeds = uniform1fFeedsNamed(harness, 'uTeamSlot');
    expect(
      teamColorFeeds.every((feed) =>
        feed.channels.every((c) => c >= 0 && c <= 1),
      ),
    ).toBe(true);
    expect(
      teamColorFeeds.some(
        (feed) =>
          feed.channels[0] === 1 &&
          feed.channels[1] === 77 / 255 &&
          feed.channels[2] === 77 / 255,
      ),
    ).toBe(true);
    expect(
      teamColorFeeds.some(
        (feed) =>
          feed.channels[0] === 95 / 255 &&
          feed.channels[1] === 1 &&
          feed.channels[2] === 1,
      ),
    ).toBe(true);
    expect(teamColorFeeds.length).toBe(10);
    expect(slotFeeds.length).toBe(10);
    expect(
      slotFeeds.every((feed) => feed.value === NEATRACING_TEAM_REMAP_SLOT),
    ).toBe(true);
  });

  it('feeds the lighting uniforms with the normalized constants', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
      ]),
    );

    const sunDirectionFeeds = uniform3fFeedsNamed(harness, 'uSunDirection');
    const sunColorFeeds = uniform3fFeedsNamed(harness, 'uSunColor');
    const fogColorFeeds = uniform3fFeedsNamed(harness, 'uFogColor');
    const cameraFeeds = uniform3fFeedsNamed(harness, 'uCameraWorld');
    const fogDensityFeeds = uniform1fFeedsNamed(harness, 'uFogDensity');
    expect(sunDirectionFeeds.length).toBe(1);
    expect(sunDirectionFeeds[0].channels).toEqual(NEATRACING_SUN_DIRECTION);
    expect(sunColorFeeds.length).toBe(1);
    expect(sunColorFeeds[0].channels).toEqual(NEATRACING_SUN_COLOR);
    expect(fogColorFeeds.length).toBe(1);
    expect(fogColorFeeds[0].channels).toEqual(NEATRACING_FOG_COLOR);
    expect(cameraFeeds.length).toBe(1);
    expect(cameraFeeds[0].channels).toEqual([0, 1, 0]);
    expect(
      fogDensityFeeds.every(
        (feed) => feed.value === NEATRACING_FOG_DENSITY_1_PER_M,
      ),
    ).toBe(true);
  });

  it('draws blob shadows at world y + 0.01 m with a square XZ footprint covering the full body footprint plus margin (G9)', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 5, worldZ: -3, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
        { worldX: -8, worldZ: 12, yawRad: 0, teamColorId: 0, wheelSpinRad: 0 },
      ]),
    );

    const shadowFeeds = matrixFeedsNamed(harness, 'uModel').filter((feed) =>
      isBlobShadowMatrix(feed.matrix),
    );
    expect(shadowFeeds.length).toBe(2);
    expect(NEATRACING_BLOB_SHADOW_Y_OFFSET_M).toBe(0.01);
    for (const feed of shadowFeeds) {
      expect(feed.matrix[5]).toBe(1);
      expect(feed.matrix[0]).toBe(feed.matrix[10]);
      // Full XZ footprint of the 4x4x4-voxel fixture (4 voxels = 4 x
      // VOXEL_WORLD_SIZE_M) x 1.15 margin; matrix[0] for a yaw-0 car
      // equals the shadow scale, fround'd through uniformMatrix4fv.
      expect(feed.matrix[0]).toBe(
        Math.fround(
          4 * VOXEL_WORLD_SIZE_M * NEATRACING_BLOB_SHADOW_BOUNDS_SCALE,
        ),
      );
    }
    const shadowColorFeeds = uniform3fFeedsNamed(harness, 'uShadowColor');
    expect(shadowColorFeeds.length).toBe(1);
    expect(shadowColorFeeds[0].channels).toEqual([8 / 255, 8 / 255, 10 / 255]);
    const shadowXOrder = shadowFeeds.map((feed) => feed.matrix[12]);
    const shadowZOrder = shadowFeeds.map((feed) => feed.matrix[14]);
    expect(shadowXOrder).toContain(5);
    expect(shadowXOrder).toContain(-8);
    expect(shadowZOrder).toContain(-3);
    expect(shadowZOrder).toContain(12);
  });

  it('sorts cars far -> near by camera distance before drawing', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 10, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
        { worldX: 0, worldZ: 60, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
      ]),
    );

    const shadowFeeds = matrixFeedsNamed(harness, 'uModel').filter((feed) =>
      isBlobShadowMatrix(feed.matrix),
    );
    expect(shadowFeeds.map((feed) => feed.matrix[14])).toEqual([60, 10]);
    const bodyFeeds = matrixFeedsNamed(harness, 'uModel').filter(
      (feed) =>
        feed.matrix[13] === 0 && feed.matrix[0] === 1 && feed.matrix[10] === 1,
    );
    expect(bodyFeeds.length).toBe(2);
    expect(bodyFeeds.map((feed) => feed.matrix[14])).toEqual([60, 10]);
  });

  it('mounts all four wheels at the G9 hub height with the socket offsets', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
        {
          worldX: 0,
          worldZ: 60,
          yawRad: 0,
          teamColorId: 1,
          wheelSpinRad: Math.PI / 2,
        },
      ]),
    );

    const wheelFeeds = matrixFeedsNamed(harness, 'uModel').filter(
      (feed) => feed.matrix[13] === NEATRACING_WHEEL_HUB_HEIGHT_M,
    );
    expect(wheelFeeds.length).toBe(8);
    const nearCarWheels = wheelFeeds.filter(
      (feed) => feed.matrix[14] >= -2 && feed.matrix[14] <= 2,
    );
    expect(nearCarWheels.length).toBe(4);
    const lateralOffsets = nearCarWheels
      .map((feed) => feed.matrix[12])
      .sort((a, b) => a - b);
    const longitudinalOffsets = nearCarWheels
      .map((feed) => feed.matrix[14])
      .sort((a, b) => a - b);
    expect(lateralOffsets[0]).toBeCloseTo(-1, 6);
    expect(lateralOffsets[3]).toBeCloseTo(1, 6);
    expect(longitudinalOffsets[0]).toBeCloseTo(-1.6, 6);
    expect(longitudinalOffsets[3]).toBeCloseTo(1.6, 6);
    const spunWheels = wheelFeeds.filter(
      (feed) => feed.matrix[14] > 50 || feed.matrix[14] < -50,
    );
    expect(spunWheels.length).toBe(4);
    for (const spun of spunWheels) {
      for (const stopped of nearCarWheels) {
        expect(sameMatrix(spun.matrix, stopped.matrix)).toBe(false);
      }
    }
  });

  it('binds the 9-float lit stride and the 2-float shadow stride', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
      ]),
    );

    expect(
      harness.vertexAttribPointerOps.some(
        (op) => op.stride === 36 && op.size === 3,
      ),
    ).toBe(true);
    expect(
      harness.vertexAttribPointerOps.some(
        (op) => op.stride === 36 && op.size === 1,
      ),
    ).toBe(true);
    expect(
      harness.vertexAttribPointerOps.some(
        (op) => op.stride === 8 && op.size === 2,
      ),
    ).toBe(true);
    expect(harness.bufferByteLengths.slice().sort((a, b) => a - b)).toEqual([
      48, 864, 864,
    ]);
  });

  it('binds the body VBO for every body draw and the wheel VBO for every wheel draw', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.draw(
      makeDrawFrame([
        { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
        { worldX: 0, worldZ: 60, yawRad: 0, teamColorId: 0, wheelSpinRad: 0 },
      ]),
    );

    // Mock buffer ids follow createCarPass creation order: body, wheel,
    // shadow. Shadow draws run with the shadow VBO bound; every lit draw
    // group must open with the body VBO freshly bound before the body draw
    // (the wheel phase leaves the wheel VBO bound, so the per-car loop has
    // to rebind the body VBO or later cars corrupt their body draws).
    const shadowDrawBoundIds = harness.drawArraysCalls
      .slice(0, 2)
      .map((draw) => boundArrayBufferIdAtDraw(harness, draw));
    expect(shadowDrawBoundIds).toEqual([
      CAR_MOCK_SHADOW_BUFFER_ID,
      CAR_MOCK_SHADOW_BUFFER_ID,
    ]);
    const litDrawBoundIds = harness.drawArraysCalls
      .slice(2)
      .map((draw) => boundArrayBufferIdAtDraw(harness, draw));
    const expectedLitBoundIds: number[] = [];
    for (let slot = 0; slot < 2; slot += 1) {
      expectedLitBoundIds.push(CAR_MOCK_BODY_BUFFER_ID);
      for (let wheelIndex = 0; wheelIndex < 4; wheelIndex += 1) {
        expectedLitBoundIds.push(CAR_MOCK_WHEEL_BUFFER_ID);
      }
    }
    expect(litDrawBoundIds).toEqual(expectedLitBoundIds);
  });
});

describe('car pass warmup seam (AC-040)', () => {
  it('registers under the car program registry key at index 4', () => {
    const harness = createCarMockWebGlHarness();
    const warmup = toCarWarmupProgram(harness.gl);
    expect(warmup.key).toBe('car');
    expect(warmup.key).toBe(NEATRACING_WARMED_PROGRAM_KEYS[4]);
  });

  it('keeps compilation lazy until compileAndLink runs', () => {
    const harness = createCarMockWebGlHarness();
    const warmup = toCarWarmupProgram(harness.gl);
    expect(harness.calls.createProgram).toBe(0);
    warmup.compileAndLink();
    expect(harness.calls.createProgram).toBe(2);
  });

  it('draws the zero-pose shared path offscreen with six draw calls per car', () => {
    const harness = createCarMockWebGlHarness();
    const warmup = toCarWarmupProgram(harness.gl);
    warmup.compileAndLink();
    const drawsBeforeWarmup = harness.calls.drawArrays;

    warmup.drawOffscreen(harness.gl);

    expect(harness.calls.drawArrays - drawsBeforeWarmup).toBe(6);
    const shadowFeeds = matrixFeedsNamed(harness, 'uModel').filter((feed) =>
      isBlobShadowMatrix(feed.matrix),
    );
    expect(shadowFeeds.length).toBe(1);
    expect(shadowFeeds[0].matrix[12]).toBe(0);
    expect(shadowFeeds[0].matrix[14]).toBe(0);
  });

  it('disposes both programs and all three buffers', () => {
    const harness = createCarMockWebGlHarness();
    const pass = createCarPass(harness.gl, {
      body: { ...CAR_FIXTURE_BIN },
      wheel: { ...CAR_FIXTURE_BIN },
    });

    pass.dispose();

    expect(harness.calls.deleteProgram).toBe(2);
    expect(harness.calls.deleteBuffer).toBe(3);
    expect(harness.calls.deleteShader).toBe(4);
  });
});

/** One recorded frame-state GL op from the car draw state machine. */
interface CarMockGlStateOp {
  readonly op: 'enable' | 'disable' | 'depthMask' | 'blendFunc';
  readonly args: readonly (number | boolean)[];
}

/** One recorded uniform3f feed keyed by uniform name. */
interface CarMockUniform3fFeed {
  readonly name: string;
  readonly channels: readonly [number, number, number];
}

/** One recorded uniform1f feed keyed by uniform name. */
interface CarMockUniform1fFeed {
  readonly name: string;
  readonly value: number;
}

/** One recorded uniformMatrix4fv feed keyed by uniform name. */
interface CarMockMatrixFeed {
  readonly name: string;
  readonly matrix: readonly number[];
}

/** One recorded vertexAttribPointer call. */
interface CarMockVertexAttribPointerOp {
  readonly index: number;
  readonly size: number;
  readonly stride: number;
  readonly offset: number;
}

/** One recorded drawArrays call tagged with the active depth mask. */
interface CarMockDrawArraysCall {
  readonly mode: number;
  readonly first: number;
  readonly count: number;
  readonly depthMask: boolean;
  /** Monotonically rising op order shared with bindBuffer ops. */
  readonly seq: number;
}

/** One recorded bindBuffer call pinned to a resolved mock buffer id. */
interface CarMockBindBufferOp {
  /** Monotonically rising op order shared with drawArrays seq. */
  readonly seq: number;
  /** Recorded GL target enum (gl.ARRAY_BUFFER for every car-pass bind). */
  readonly target: number;
  /** Resolved mock buffer id (creation/bind order: body, wheel, shadow). */
  readonly bufferId: number;
}

/** Mutable call recorder mirroring the mock-GL harness shape. */
interface CarMockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly calls: Record<
    | 'drawArrays'
    | 'depthMask'
    | 'blendFunc'
    | 'createProgram'
    | 'createBuffer'
    | 'deleteProgram'
    | 'deleteBuffer'
    | 'deleteShader',
    number
  >;
  /** Frame-state ops in call order (draw state machine pin). */
  readonly stateOps: ReadonlyArray<CarMockGlStateOp>;
  /** Uniform3f feeds keyed by uniform name (color pins). */
  readonly uniform3fFeeds: ReadonlyArray<CarMockUniform3fFeed>;
  /** Uniform1f feeds keyed by uniform name (scalar pins). */
  readonly uniform1fFeeds: ReadonlyArray<CarMockUniform1fFeed>;
  /** Matrix feeds keyed by uniform name (model-matrix pins). */
  readonly matrixFeeds: ReadonlyArray<CarMockMatrixFeed>;
  /** Vertex attribute pointer ops (stride/size pins). */
  readonly vertexAttribPointerOps: ReadonlyArray<CarMockVertexAttribPointerOp>;
  /** Draw calls tagged with the depth-mask state when issued. */
  readonly drawArraysCalls: ReadonlyArray<CarMockDrawArraysCall>;
  /** bindBuffer calls in op order with resolved mock buffer ids. */
  readonly bindBufferOps: ReadonlyArray<CarMockBindBufferOp>;
  /** Byte lengths recorded per bufferData upload. */
  readonly bufferByteLengths: ReadonlyArray<number>;
}

/**
 * Creates a recording mock WebGL2 context covering every GL entry point the
 * lighting slice touches, mirroring the pulse.pass.test.ts harness pattern
 * and extending it with matrix feeds, uniform1f feeds, depth-mask-tagged
 * draw calls, deterministic attribute locations, and buffer accounting.
 */
function createCarMockWebGlHarness(): CarMockGlHarness {
  const calls: CarMockGlHarness['calls'] = {
    drawArrays: 0,
    depthMask: 0,
    blendFunc: 0,
    createProgram: 0,
    createBuffer: 0,
    deleteProgram: 0,
    deleteBuffer: 0,
    deleteShader: 0,
  };
  const stateOps: CarMockGlStateOp[] = [];
  const uniform3fFeeds: CarMockUniform3fFeed[] = [];
  const uniform1fFeeds: CarMockUniform1fFeed[] = [];
  const matrixFeeds: CarMockMatrixFeed[] = [];
  const vertexAttribPointerOps: CarMockVertexAttribPointerOp[] = [];
  const drawArraysCalls: CarMockDrawArraysCall[] = [];
  const bindBufferOps: CarMockBindBufferOp[] = [];
  const bufferByteLengths: number[] = [];
  let opSequence = 0;
  let activeDepthMask = true;
  let programCounter = 0;
  let attribCounter = 0;
  let bufferCounter = 0;
  const programIds = new WeakMap<object, number>();
  const attribIds = new Map<string, number>();
  const programIdOf = (program: object): number => {
    let id = programIds.get(program);
    if (id === undefined) {
      programCounter += 1;
      id = programCounter;
      programIds.set(program, id);
    }
    return id;
  };
  const bufferIds = new WeakMap<object, number>();
  const bufferIdOf = (buffer: object): number => {
    let id = bufferIds.get(buffer);
    if (id === undefined) {
      bufferCounter += 1;
      id = bufferCounter;
      bufferIds.set(buffer, id);
    }
    return id;
  };

  const gl = {
    createShader: (): object => ({}),
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    getShaderInfoLog: (): string => '',
    createProgram: (): object => {
      calls.createProgram += 1;
      return {};
    },
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getProgramInfoLog: (): string => '',
    getAttribLocation: (program: object, name: string): number => {
      const key = `${programIdOf(program)}:${name}`;
      let id = attribIds.get(key);
      if (id === undefined) {
        attribCounter += 1;
        id = attribCounter;
        attribIds.set(key, id);
      }
      return id;
    },
    getUniformLocation: (_program: unknown, name: string): object => ({ name }),
    useProgram: (): void => undefined,
    enable: (target: number): void => {
      stateOps.push({ op: 'enable', args: [target] });
    },
    disable: (target: number): void => {
      stateOps.push({ op: 'disable', args: [target] });
    },
    depthMask: (flag: boolean): void => {
      calls.depthMask += 1;
      activeDepthMask = flag;
      stateOps.push({ op: 'depthMask', args: [flag] });
    },
    blendFunc: (sfactor: number, dfactor: number): void => {
      calls.blendFunc += 1;
      stateOps.push({ op: 'blendFunc', args: [sfactor, dfactor] });
    },
    createBuffer: (): object => {
      calls.createBuffer += 1;
      return {};
    },
    bindBuffer: (target: number, buffer: object): void => {
      opSequence += 1;
      bindBufferOps.push({
        seq: opSequence,
        target,
        bufferId: bufferIdOf(buffer),
      });
    },
    bufferData: (_target: unknown, data: ArrayBufferView): void => {
      bufferByteLengths.push(data.byteLength);
    },
    vertexAttribPointer: (
      index: number,
      size: number,
      _type: unknown,
      _normalized: boolean,
      stride: number,
      offset: number,
    ): void => {
      vertexAttribPointerOps.push({ index, size, stride, offset });
    },
    enableVertexAttribArray: (): void => undefined,
    uniformMatrix4fv: (
      location: { name: string },
      _transpose: unknown,
      matrix: Float32Array,
    ): void => {
      matrixFeeds.push({ name: location.name, matrix: Array.from(matrix) });
    },
    uniform1f: (location: { name: string }, value: number): void => {
      uniform1fFeeds.push({ name: location.name, value });
    },
    uniform3f: (
      location: { name: string },
      red: number,
      green: number,
      blue: number,
    ): void => {
      uniform3fFeeds.push({
        name: location.name,
        channels: [red, green, blue],
      });
    },
    drawArrays: (mode: number, first: number, count: number): void => {
      calls.drawArrays += 1;
      opSequence += 1;
      drawArraysCalls.push({
        mode,
        first,
        count,
        depthMask: activeDepthMask,
        seq: opSequence,
      });
    },
    deleteProgram: (): void => {
      calls.deleteProgram += 1;
    },
    deleteShader: (): void => {
      calls.deleteShader += 1;
    },
    deleteBuffer: (): void => {
      calls.deleteBuffer += 1;
    },
  };
  return {
    gl: gl as unknown as WebGL2RenderingContext,
    calls,
    stateOps,
    uniform3fFeeds,
    uniform1fFeeds,
    matrixFeeds,
    vertexAttribPointerOps,
    drawArraysCalls,
    bindBufferOps,
    bufferByteLengths,
  };
}

/** Shallow 16-element matrix equality probe for draw-order assertions. */
function sameMatrix(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

/**
 * F32 feed tolerance: the WebGL2 matrix uploads round to binary32, so the
 * spec-pinned 0.01 m blob offset arrives as fround(0.01) in matrix feeds.
 */
const NEATRACING_TEST_F32_EPSILON = 1e-8;

/**
 * Detects a blob-shadow model matrix by its spec-pinned world y offset.
 *
 * @param matrix - Recorded 16-element model-matrix feed.
 * @returns True when the matrix translates to the shadow height (f32-tolerant).
 */
function isBlobShadowMatrix(matrix: readonly number[]): boolean {
  return (
    Math.abs(matrix[13] - NEATRACING_BLOB_SHADOW_Y_OFFSET_M) <
    NEATRACING_TEST_F32_EPSILON
  );
}

/**
 * Resolves the mock ARRAY_BUFFER binding in force at one recorded draw.
 *
 * Reconstructs buffer-vs-draw discipline from bindBuffer ops interleaved
 * with drawArrays ops through their shared monotonically rising op
 * sequence; body/wheel meshes share a vertex count, so draw order — not
 * vertex count — is the discriminator for the per-group bound buffer.
 *
 * @param harness - Recording mock harness.
 * @param draw - Recorded drawArrays call whose seq anchors the lookup.
 * @returns Mock buffer id last bound to gl.ARRAY_BUFFER before the draw
 * (0 when no ARRAY_BUFFER bind preceded the draw).
 */
function boundArrayBufferIdAtDraw(
  harness: CarMockGlHarness,
  draw: CarMockDrawArraysCall,
): number {
  let boundId = 0;
  for (const bind of harness.bindBufferOps) {
    if (bind.seq > draw.seq) {
      break;
    }
    if (bind.target === MOCK_GL_ARRAY_BUFFER) {
      boundId = bind.bufferId;
    }
  }
  return boundId;
}
