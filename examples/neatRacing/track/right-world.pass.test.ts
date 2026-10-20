/**
 * Red contract tests for the neatRacing right-world render-pass scene assembly
 * (slices P1-sprite-spacing + P1-bleacher-shape, Step 14J, AC-065 + AC-065b).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - {@link buildRightWorldSceneData} returns a scene with non-empty props,
 *   atlas frames, and billboards.
 * - Every assembled robot billboard keeps a personal-space radius >= 1 m
 *   equivalent in the XZ plane.
 * - No robot billboard is anchored to a tiny pedestal/cube prop; supports host
 *   only as many robots as their horizontal top area affords at >= 1 m spacing.
 */

import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_LIGHTING_UNIFORM_NAMES,
  NEATRACING_SUN_COLOR,
  NEATRACING_SUN_DIRECTION,
} from '../engine/lighting';
import {
  getBleacherAxisAlignedHalfSize,
  getBleacherSeatTopY,
  type RightWorldProp,
} from './right-world.builder';
import {
  buildRightWorldSceneData,
  createRightWorldPass,
  type RightWorldPassDrawState,
} from './right-world.pass';
import { resolvePropHalfSize } from './right-world.prop-sizes';

/** Minimum XZ center-to-center separation enforced by AC-065 (meters). */
const NEATRACING_ROBOT_PERSONAL_SPACE_M = 1.0;

/**
 * Tolerance for matching a billboard foot anchor to a prop top surface.
 * Low pedestals are only ~0.5m tall, so a small vertical window is enough to
 * detect them while ignoring tall structures such as bleachers.
 */
const NEATRACING_PROP_HOST_TOLERANCE_M = 0.2;

/** Prop kinds that are considered tiny pedestal/cube anchors (AC-065b ban). */
const TINY_PEDESTAL_KINDS: readonly RightWorldProp['kind'][] = ['mechanic', 'crew'];

/** Computes the smallest XZ distance between any two billboard positions. */
function minRobotXZDistance(
  positions: readonly (readonly [number, number, number])[],
): number {
  let minDistance = Number.POSITIVE_INFINITY;

  for (let i = 0; i < positions.length; i += 1) {
    const a = positions[i]!;
    for (let j = i + 1; j < positions.length; j += 1) {
      const b = positions[j]!;
      const dx = a[0] - b[0];
      const dz = a[2] - b[2];
      const distance = Math.hypot(dx, dz);
      if (distance < minDistance) {
        minDistance = distance;
      }
    }
  }

  return minDistance;
}

/**
 * Finds the prop whose top surface hosts a billboard foot anchor.
 * Returns `undefined` when the billboard stands on the ground plane.
 *
 * A billboard is considered hosted by a prop when its horizontal footprint
 * center lies inside the prop's XZ bounds and its foot Y is within a small
 * tolerance of the prop's top surface.  The closest matching prop top wins
 * so that low pedestals are not mistaken for the ground plane.
 */
function findHostProp(
  billboard: { readonly position: readonly [number, number, number] },
  props: readonly RightWorldProp[],
): RightWorldProp | undefined {
  const [x, y, z] = billboard.position;
  let best: RightWorldProp | undefined;
  let bestDeltaY = Number.POSITIVE_INFINITY;

  for (const prop of props) {
    const [px, py, pz] = prop.position;
    const [hx, hy, hz] = resolvePropHalfSize(prop.kind);
    const topY = py + 2 * hy;

    if (Math.abs(x - px) > hx || Math.abs(z - pz) > hz) {
      continue;
    }

    const deltaY = Math.abs(y - topY);
    if (deltaY <= NEATRACING_PROP_HOST_TOLERANCE_M && deltaY < bestDeltaY) {
      best = prop;
      bestDeltaY = deltaY;
    }
  }

  return best;
}

/** Computes the horizontal top surface area of a prop in square meters. */
function propTopArea(prop: RightWorldProp): number {
  const [hx, , hz] = resolvePropHalfSize(prop.kind);
  return (2 * hx) * (2 * hz);
}

/**
 * Counts how many billboards are hosted by each prop, keyed by prop identity.
 * Billboards standing on the ground are not counted.
 */
function countHostedBillboardsPerProp(
  billboards: readonly { readonly position: readonly [number, number, number] }[],
  props: readonly RightWorldProp[],
): Map<RightWorldProp, number> {
  const counts = new Map<RightWorldProp, number>();

  for (const billboard of billboards) {
    const host = findHostProp(billboard, props);
    if (host) {
      counts.set(host, (counts.get(host) ?? 0) + 1);
    }
  }

  return counts;
}

/** `gl.DEPTH_TEST` enum value (0x0B71). */
const MOCK_GL_DEPTH_TEST = 0x0b71;

/** `gl.CULL_FACE` enum value (0x0B44). */
const MOCK_GL_CULL_FACE = 0x0b44;

/** `gl.BLEND` enum value (0x0BE2). */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.TRIANGLES` enum value (0x0004). */
const MOCK_GL_TRIANGLES = 0x0004;

/** `gl.ARRAY_BUFFER` enum value (0x8892). */
const MOCK_GL_ARRAY_BUFFER = 0x8892;

/** `gl.TEXTURE0` enum value. */
const MOCK_GL_TEXTURE0 = 0x84c0;

/** Mutable call recorder mirroring the right-world pass mock-GL shape. */
interface RightWorldMockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly uniformLocationRequests: readonly string[];
  readonly uniform3fFeeds: ReadonlyArray<{
    readonly name: string;
    readonly channels: readonly [number, number, number];
  }>;
  readonly uniform1fFeeds: ReadonlyArray<{ readonly name: string; readonly value: number }>;
  readonly uniform1iFeeds: ReadonlyArray<{ readonly name: string; readonly value: number }>;
  readonly uniformMatrixFeeds: ReadonlyArray<{
    readonly name: string;
    readonly matrix: readonly number[];
  }>;
  readonly texImage2DCalls: ReadonlyArray<{
    readonly width: number;
    readonly height: number;
    readonly format: number;
    readonly type: number;
  }>;
  readonly drawCalls: ReadonlyArray<{
    readonly mode: number;
    readonly first: number;
    readonly count: number;
  }>;
}

/** Filters recorded uniform3f feeds down to one uniform name. */
function uniform3fFeedsNamed(
  harness: RightWorldMockGlHarness,
  name: string,
): ReadonlyArray<{
  readonly name: string;
  readonly channels: readonly [number, number, number];
}> {
  return harness.uniform3fFeeds.filter((feed) => feed.name === name);
}

/** Filters recorded uniform1f feeds down to one uniform name. */
function uniform1fFeedsNamed(
  harness: RightWorldMockGlHarness,
  name: string,
): ReadonlyArray<{ readonly name: string; readonly value: number }> {
  return harness.uniform1fFeeds.filter((feed) => feed.name === name);
}

/** Default right-world draw state used by the lighting red tests. */
function createRightWorldDrawState(): RightWorldPassDrawState {
  return {
    viewProjection: new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]),
    cameraWorld: [0, 2, 20],
    cameraRight: [1, 0, 0],
    cameraUp: [0, 1, 0],
    timeSeconds: 0,
    carPosition: [0, 0, 0],
  };
}

/**
 * Builds a recording mock WebGL2 context covering every GL entry point the
 * right-world pass touches during create and draw, including the state
 * save/restore queries around the prop and billboard scopes.
 */
function createRightWorldMockGlHarness(): RightWorldMockGlHarness {
  const uniformLocationRequests: string[] = [];
  const uniform3fFeeds: Array<{
    readonly name: string;
    readonly channels: readonly [number, number, number];
  }> = [];
  const uniform1fFeeds: Array<{ readonly name: string; readonly value: number }> = [];
  const uniform1iFeeds: Array<{ readonly name: string; readonly value: number }> = [];
  const uniformMatrixFeeds: Array<{
    readonly name: string;
    readonly matrix: readonly number[];
  }> = [];
  const texImage2DCalls: Array<{
    readonly width: number;
    readonly height: number;
    readonly format: number;
    readonly type: number;
  }> = [];
  const drawCalls: Array<{ readonly mode: number; readonly first: number; readonly count: number }> = [];

  const parameterDefaults = new Map<number, unknown>([
    [0x0b71, false], // DEPTH_TEST
    [0x0be2, false], // BLEND
    [0x0b44, false], // CULL_FACE
    [0x0b71, false], // DEPTH_TEST (duplicate, ignored)
    [0x0b90, true], // DEPTH_WRITEMASK
    [0x0302, 1], // BLEND_SRC_RGB = ONE
    [0x0303, 0], // BLEND_DST_RGB = ZERO
    [0x0306, 1], // BLEND_SRC_ALPHA = ONE
    [0x0307, 0], // BLEND_DST_ALPHA = ZERO
    [0x84e0, 0], // ACTIVE_TEXTURE = TEXTURE0
    [0x8b8d, null], // CURRENT_PROGRAM
    [0x8894, null], // ARRAY_BUFFER_BINDING
    [0x8b8d, null], // CURRENT_PROGRAM (duplicate)
    [0x806a, null], // TEXTURE_BINDING_2D
    [0x0d33, 16384], // MAX_TEXTURE_SIZE
  ]);

  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b20,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: MOCK_GL_ARRAY_BUFFER,
    STATIC_DRAW: 0x88e4,
    DYNAMIC_DRAW: 0x88e8,
    FLOAT: 0x1406,
    UNSIGNED_BYTE: 0x1401,
    RGBA: 0x1908,
    TEXTURE_2D: 0x0de1,
    TEXTURE0: MOCK_GL_TEXTURE0,
    DEPTH_TEST: MOCK_GL_DEPTH_TEST,
    CULL_FACE: MOCK_GL_CULL_FACE,
    BLEND: MOCK_GL_BLEND,
    TRIANGLES: MOCK_GL_TRIANGLES,
    CLAMP_TO_EDGE: 0x812f,
    LINEAR: 0x2601,
    SRC_ALPHA: 0x0302,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    MAX_TEXTURE_SIZE: 0x0d33,
    createShader: (): object => ({}),
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    getShaderInfoLog: (): string => '',
    createProgram: (): object => ({}),
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getProgramInfoLog: (): string => '',
    getAttribLocation: (_program: unknown, _name: string): number => {
      void _program;
      void _name;
      return 0;
    },
    getUniformLocation: (_program: unknown, name: string): object => {
      uniformLocationRequests.push(name);
      return { name };
    },
    createBuffer: (): object => ({}),
    bindBuffer: (): void => undefined,
    bufferData: (): void => undefined,
    bufferSubData: (): void => undefined,
    enableVertexAttribArray: (): void => undefined,
    disableVertexAttribArray: (): void => undefined,
    vertexAttribPointer: (): void => undefined,
    enable: (): void => undefined,
    disable: (): void => undefined,
    depthMask: (): void => undefined,
    blendFunc: (): void => undefined,
    blendFuncSeparate: (): void => undefined,
    useProgram: (): void => undefined,
    activeTexture: (): void => undefined,
    bindTexture: (): void => undefined,
    createTexture: (): object => ({}),
    texImage2D: (
      _target: number,
      _level: number,
      _internalformat: number,
      width: number,
      height: number,
      _border: number,
      format: number,
      type: number,
      _data: unknown,
    ): void => {
      void _target;
      void _level;
      void _internalformat;
      void _border;
      void _data;
      texImage2DCalls.push({ width, height, format, type });
    },
    texParameteri: (): void => undefined,
    uniformMatrix4fv: (
      location: { name: string },
      _transpose: boolean,
      matrix: Float32Array,
    ): void => {
      uniformMatrixFeeds.push({ name: location.name, matrix: Array.from(matrix) });
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
    uniform3fv: (
      location: { name: string },
      channels: readonly [number, number, number],
    ): void => {
      uniform3fFeeds.push({ name: location.name, channels });
    },
    uniform1f: (location: { name: string }, value: number): void => {
      uniform1fFeeds.push({ name: location.name, value });
    },
    uniform1fv: (location: { name: string }, values: readonly number[]): void => {
      for (const value of values) {
        uniform1fFeeds.push({ name: location.name, value });
      }
    },
    uniform1i: (location: { name: string }, value: number): void => {
      uniform1iFeeds.push({ name: location.name, value });
    },
    drawArrays: (mode: number, first: number, count: number): void => {
      drawCalls.push({ mode, first, count });
    },
    isEnabled: (): boolean => false,
    getParameter: (pname: number): unknown => parameterDefaults.get(pname) ?? null,
    getVertexAttrib: (): boolean => false,
    deleteProgram: (): void => undefined,
    deleteShader: (): void => undefined,
    deleteBuffer: (): void => undefined,
    deleteTexture: (): void => undefined,
  };

  return {
    gl: gl as unknown as WebGL2RenderingContext,
    uniformLocationRequests,
    uniform3fFeeds,
    uniform1fFeeds,
    uniform1iFeeds,
    uniformMatrixFeeds,
    texImage2DCalls,
    drawCalls,
  };
}

describe('right-world.pass scene assembly (P1-sprite-spacing, AC-065)', () => {
  it('returns a scene with non-empty props, atlas, and billboards', () => {
    const scene = buildRightWorldSceneData();

    expect(scene.props.length).toBeGreaterThan(0);
    expect(scene.atlas.frames.length).toBeGreaterThan(0);
    expect(scene.billboards.length).toBeGreaterThan(1);
  });

  it('assembles robot billboards with >= 1m XZ personal-space separation', () => {
    const scene = buildRightWorldSceneData();
    const positions = scene.billboards.map((billboard) => billboard.position);

    expect(positions.length).toBeGreaterThan(1);

    const minDistance = minRobotXZDistance(positions);

    expect(minDistance).toBeGreaterThanOrEqual(NEATRACING_ROBOT_PERSONAL_SPACE_M);
  });
});

describe('right-world.pass scene anchoring (P1-bleacher-shape, AC-065b)', () => {
  it('places no robot on a tiny pedestal or cube prop', () => {
    const scene = buildRightWorldSceneData();
    const tinyKinds = new Set(TINY_PEDESTAL_KINDS);
    const violations = scene.billboards.filter((billboard) => {
      const host = findHostProp(billboard, scene.props);
      return host && tinyKinds.has(host.kind);
    });

    expect(violations.length).toBe(0);
  });

  it('hosts only as many robots per support as its top area affords at 1m spacing', () => {
    const scene = buildRightWorldSceneData();
    const counts = countHostedBillboardsPerProp(scene.billboards, scene.props);

    for (const [prop, count] of counts.entries()) {
      const maxHosted = Math.floor(propTopArea(prop));

      expect(count).toBeLessThanOrEqual(maxHosted);
    }
  });
});

describe('right-world.pass prop shared sun lighting (P1-world-lighting-coherence)', () => {
  it('resolves the shared sun direction uniform in the prop program', () => {
    const harness = createRightWorldMockGlHarness();
    createRightWorldPass(harness.gl);
    expect(harness.uniformLocationRequests).toContain('uSunDirection');
  });

  it('resolves all shared lighting uniforms in the prop program', () => {
    const harness = createRightWorldMockGlHarness();
    createRightWorldPass(harness.gl);
    for (const name of NEATRACING_LIGHTING_UNIFORM_NAMES) {
      expect(harness.uniformLocationRequests).toContain(name);
    }
  });

  it('feeds the shared sun direction and color during prop draw', () => {
    const harness = createRightWorldMockGlHarness();
    const pass = createRightWorldPass(harness.gl);
    pass.draw(createRightWorldDrawState());
    const sunDirectionFeeds = uniform3fFeedsNamed(harness, 'uSunDirection');
    const sunColorFeeds = uniform3fFeedsNamed(harness, 'uSunColor');
    expect(sunDirectionFeeds.length).toBeGreaterThan(0);
    expect(sunColorFeeds.length).toBeGreaterThan(0);
    expect(sunDirectionFeeds[0].channels).toEqual(NEATRACING_SUN_DIRECTION);
    expect(sunColorFeeds[0].channels).toEqual(NEATRACING_SUN_COLOR);
  });

  it('feeds the camera world position and fog density during prop draw', () => {
    const harness = createRightWorldMockGlHarness();
    const pass = createRightWorldPass(harness.gl);
    const state = createRightWorldDrawState();
    pass.draw(state);
    const cameraFeeds = uniform3fFeedsNamed(harness, 'uCameraWorld');
    const fogDensityFeeds = uniform1fFeedsNamed(harness, 'uFogDensity');
    expect(cameraFeeds.length).toBeGreaterThan(0);
    expect(cameraFeeds[0].channels).toEqual(state.cameraWorld);
    expect(fogDensityFeeds.length).toBeGreaterThan(0);
  });
});

describe(
  'right-world sprite atlas packing (fix-packet-P1-bleacher-crowd-iteration-2, OBS-D)',
  () => {
    it('packs the combined public + mechanic atlas within GPU max texture size', () => {
      const harness = createRightWorldMockGlHarness();
      createRightWorldPass(harness.gl);
      const atlasCalls = harness.texImage2DCalls.filter(
        (call) => call.format === 0x1908 && call.type === 0x1401,
      );

      expect(atlasCalls.length).toBe(1);
      const atlasCall = atlasCalls[0]!;
      expect(atlasCall.width).toBeLessThanOrEqual(16384);
      expect(atlasCall.height).toBeLessThanOrEqual(16384);
    });
  },
);

describe(
  'bleacher crowd hosting and gaze (fix-packet-P1-bleacher-crowd-iteration-1, OBS-B + OBS-C)',
  () => {
    /**
     * Finds a bleacher prop that hosts a billboard using real voxel geometry:
     * the billboard foot must be inside the bleacher XZ footprint and within
     * tolerance of the actual seat-top surface at that XZ location.
     */
    function findBleacherHost(
      billboard: { readonly position: readonly [number, number, number] },
      props: readonly RightWorldProp[],
    ): RightWorldProp | undefined {
      const [x, y, z] = billboard.position;

      for (const prop of props) {
        if (prop.kind !== 'bleacher' || !prop.bleacherGeometry) {
          continue;
        }

        const [px, , pz] = prop.position;
        const [hx, hz] = getBleacherAxisAlignedHalfSize(prop);

        if (Math.abs(x - px) > hx || Math.abs(z - pz) > hz) {
          continue;
        }

        const topY = getBleacherSeatTopY(prop, x, z);
        if (Math.abs(y - topY) <= NEATRACING_PROP_HOST_TOLERANCE_M) {
          return prop;
        }
      }

      return undefined;
    }

    /** Returns the public billboards that are really hosted on a bleacher. */
    function getBleacherHostedPublicBillboards(
      scene: { readonly props: readonly RightWorldProp[]; readonly billboards: readonly { readonly position: readonly [number, number, number]; readonly runtimeRole?: string }[] },
    ): { readonly position: readonly [number, number, number]; readonly runtimeRole?: string }[] {
      return scene.billboards.filter((billboard) => {
        if (billboard.runtimeRole === 'mechanic') {
          return false;
        }

        return findBleacherHost(billboard, scene.props) !== undefined;
      });
    }

    it('hosts at least one public spectator on a real bleacher stand cell', () => {
      const scene = buildRightWorldSceneData();
      const hosted = getBleacherHostedPublicBillboards(scene);

      expect(hosted.length).toBeGreaterThan(0);
    });

    it('keeps the majority of bleacher-hosted spectators in look-at-car mode', () => {
      const scene = buildRightWorldSceneData();
      const hosted = getBleacherHostedPublicBillboards(scene);

      expect(hosted.length).toBeGreaterThan(0);

      const lookingAtCar = hosted.filter(
        (billboard) => billboard.runtimeRole === 'seated',
      );
      const ratio = lookingAtCar.length / hosted.length;

      expect(ratio).toBeGreaterThan(0.5);
    });
  },
);
