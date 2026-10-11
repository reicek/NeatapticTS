/**
 * Red-first contract tests for the neatRacing starfield background pass
 * (slice P1-S2-A; AC-004 flappy-parity 3-layer parallax stars + G6
 * camera-driven scroll).
 *
 * The suite pins the spec §6.4.1 layer table (seeds 1337/2777/4242, scroll
 * ratios 0.03/0.04/0.05, star counts 35/28/23, sizes, alpha ranges, blur),
 * the deterministic xorshift32 point layout per seed, and the pure
 * `(yaw, pitch) → per-layer scroll offset` driver. The GPU seam is exercised
 * through a recording mock WebGL2 context (node test environment, no real
 * GL), asserting one POINTS batch per layer with depth writing disabled and
 * per-layer scroll uniforms fed from the pure offset functions.
 *
 * Starfield layers are visual-only (`NEATRACING_VISUAL_SEED` base): they sit
 * outside the frame contract and are exempt from replay byte parity per spec
 * §9.7.2 — only the constants and layout determinism are pinned here.
 */
import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_SKY_BAND_HEIGHT_PX,
  NEATRACING_STARFIELD_CYAN_FILL_STYLE,
  NEATRACING_STARFIELD_LAYER_SPECS,
  NEATRACING_STARFIELD_TILE_HEIGHT_PX,
  NEATRACING_STARFIELD_TILE_WIDTH_PX,
  NEATRACING_VISUAL_SEED,
} from '../../browser-entry/constants.starfield';
import { NEATRACING_WARMED_PROGRAM_KEYS } from '../context.service';
import {
  buildStarLayerLayout,
  computeStarLayerOffsets,
  createStarfieldPass,
  createStarfieldRng,
  toStarfieldWarmupProgram,
} from './starfield.pass';
import type { StarfieldCameraPose } from './starfield.pass';

/** Flappy-parity layer table the spec §6.4.1 pins verbatim (AC-004). */
const EXPECTED_LAYER_SPECS = [
  {
    id: 'far',
    seed: 1_337,
    scrollRatio: 0.03,
    starCount: 35,
    minSizePx: 1,
    maxSizePx: 2,
    minAlpha: 0.08,
    maxAlpha: 0.22,
    blurPx: 4,
  },
  {
    id: 'mid',
    seed: 2_777,
    scrollRatio: 0.04,
    starCount: 28,
    minSizePx: 1,
    maxSizePx: 3,
    minAlpha: 0.1,
    maxAlpha: 0.28,
    blurPx: 6,
  },
  {
    id: 'near',
    seed: 4_242,
    scrollRatio: 0.05,
    starCount: 23,
    minSizePx: 2,
    maxSizePx: 4,
    minAlpha: 0.12,
    maxAlpha: 0.34,
    blurPx: 8,
  },
] as const;

/** Quarter-turn camera pose used to assert per-layer scroll shear. */
const QUARTER_POSE: StarfieldCameraPose = {
  yawRadians: Math.PI / 2,
  pitchRadians: Math.PI / 2,
};

/** Expected far-layer offset for the quarter pose: 0.25 × 512 × 0.03. */
const EXPECTED_FAR_OFFSET_X = -3.84;

/** Expected mid-layer offset for the quarter pose: 0.25 × 512 × 0.04. */
const EXPECTED_MID_OFFSET_X = -5.12;

/** Expected near-layer offset for the quarter pose: 0.25 × 512 × 0.05. */
const EXPECTED_NEAR_OFFSET_X = -6.4;

/** Expected far-layer vertical offset: 0.25 × sky band 240 × 0.03. */
const EXPECTED_FAR_OFFSET_Y = -1.8;

/** Expected mid-layer vertical offset: 0.25 × 240 × 0.04. */
const EXPECTED_MID_OFFSET_Y = -2.4;

/** Expected near-layer vertical offset: 0.25 × 240 × 0.05. */
const EXPECTED_NEAR_OFFSET_Y = -3;

/** WebGL constant used by the mock context for gl.POINTS. */
const MOCK_DRAW_MODE_POINTS = 0;

/** WebGL constant used by the mock context for gl.DEPTH_TEST. */
const MOCK_DEPTH_TEST = 0x0b71;

/** One recorded drawArrays invocation from the mock context. */
interface RecordedDrawCall {
  readonly mode: number;
  readonly first: number;
  readonly count: number;
}

/** One recorded uniform2f invocation from the mock context. */
interface RecordedUniform2f {
  readonly name: string;
  readonly x: number;
  readonly y: number;
}

/** Recording harness the GPU-seam tests drive and inspect. */
interface MockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly drawCalls: RecordedDrawCall[];
  readonly uniform2fCalls: RecordedUniform2f[];
  readonly disableCalls: number[];
  readonly depthMaskValues: boolean[];
}

/**
 * Creates a recording mock WebGL2 context covering the shader pipeline,
 * VAO/buffer setup, blend/depth state, and draw calls the pass uses.
 *
 * @returns Harness exposing the mock context plus its recorded call logs.
 */
function createMockGlHarness(): MockGlHarness {
  const drawCalls: RecordedDrawCall[] = [];
  const uniform2fCalls: RecordedUniform2f[] = [];
  const disableCalls: number[] = [];
  const depthMaskValues: boolean[] = [];
  const gl = {
    POINTS: MOCK_DRAW_MODE_POINTS,
    DEPTH_TEST: MOCK_DEPTH_TEST,
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b20,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: 0x8892,
    STATIC_DRAW: 0x88e4,
    FLOAT: 0x1406,
    BLEND: 0x0be2,
    SRC_ALPHA: 0x0302,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    createShader: (): object => ({}),
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    createProgram: (): object => ({}),
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getUniformLocation: (_program: object, name: string): object => ({ name }),
    useProgram: (): void => undefined,
    createVertexArray: (): object => ({}),
    bindVertexArray: (): void => undefined,
    createBuffer: (): object => ({}),
    bindBuffer: (): void => undefined,
    bufferData: (): void => undefined,
    enableVertexAttribArray: (): void => undefined,
    vertexAttribPointer: (): void => undefined,
    enable: (): void => undefined,
    disable: (token: number): void => {
      disableCalls.push(token);
    },
    blendFunc: (): void => undefined,
    depthMask: (flag: boolean): void => {
      depthMaskValues.push(flag);
    },
    uniform2f: (location: { name: string }, x: number, y: number): void => {
      uniform2fCalls.push({ name: location.name, x, y });
    },
    uniform1f: (): void => undefined,
    uniform3f: (): void => undefined,
    drawArrays: (mode: number, first: number, count: number): void => {
      drawCalls.push({ mode, first, count });
    },
  };
  return {
    gl: gl as unknown as WebGL2RenderingContext,
    drawCalls,
    uniform2fCalls,
    disableCalls,
    depthMaskValues,
  };
}

/** Typed shape guard for one star placement within the layout contract. */
function isPlacementWithinLayerContract(
  placement: {
    readonly xPx: number;
    readonly yPx: number;
    readonly sizePx: number;
    readonly alpha: number;
  },
  layerIndex: number,
): boolean {
  const spec = NEATRACING_STARFIELD_LAYER_SPECS[layerIndex]!;
  return (
    placement.xPx >= 0 &&
    placement.xPx < NEATRACING_STARFIELD_TILE_WIDTH_PX &&
    placement.yPx >= 0 &&
    placement.yPx < NEATRACING_STARFIELD_TILE_HEIGHT_PX &&
    placement.sizePx >= spec.minSizePx &&
    placement.sizePx <= spec.maxSizePx &&
    placement.alpha >= spec.minAlpha &&
    placement.alpha <= spec.maxAlpha
  );
}

describe('neatRacing starfield pass (AC-004)', () => {
  describe('layer spec constants', () => {
    it('pins the three flappy-parity parallax layers to the spec §6.4.1 table', () => {
      expect(NEATRACING_STARFIELD_LAYER_SPECS).toEqual(EXPECTED_LAYER_SPECS);
    });

    it('spans exactly the far/mid/near sky with no cloud layers present', () => {
      expect(NEATRACING_STARFIELD_LAYER_SPECS.map((spec) => spec.id)).toEqual([
        'far',
        'mid',
        'near',
      ]);
    });

    it('derives the far layer seed from the visual-only seed base', () => {
      expect(NEATRACING_STARFIELD_LAYER_SPECS[0]!.seed).toBe(
        NEATRACING_VISUAL_SEED,
      );
    });

    it('keeps source-text parity with the flappy cyan star fill', () => {
      expect(NEATRACING_STARFIELD_CYAN_FILL_STYLE).toBe(
        'rgba(95, 255, 255, 1)',
      );
    });

    it('backs the tiled sky with the fixed above-horizon band height', () => {
      expect(NEATRACING_STARFIELD_TILE_HEIGHT_PX).toBe(
        NEATRACING_SKY_BAND_HEIGHT_PX,
      );
      expect(NEATRACING_SKY_BAND_HEIGHT_PX).toBe(240);
    });
  });

  describe('deterministic point layout per seed', () => {
    it('regenerates the identical placement sequence for the same layer seed', () => {
      const spec = NEATRACING_STARFIELD_LAYER_SPECS[1]!;
      const first = buildStarLayerLayout(
        spec,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
      );
      const second = buildStarLayerLayout(
        spec,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
      );

      expect(second.placements).toEqual(first.placements);
    });

    it('produces exactly starCount placements for every layer', () => {
      const layouts = NEATRACING_STARFIELD_LAYER_SPECS.map((spec) =>
        buildStarLayerLayout(
          spec,
          NEATRACING_STARFIELD_TILE_WIDTH_PX,
          NEATRACING_STARFIELD_TILE_HEIGHT_PX,
        ),
      );

      expect(layouts.map((layout) => layout.placements.length)).toEqual([
        35, 28, 23,
      ]);
    });

    it('keeps every placement inside the tile and the layer size/alpha ranges', () => {
      const layouts = NEATRACING_STARFIELD_LAYER_SPECS.map((spec, index) => ({
        index,
        placements: buildStarLayerLayout(
          spec,
          NEATRACING_STARFIELD_TILE_WIDTH_PX,
          NEATRACING_STARFIELD_TILE_HEIGHT_PX,
        ).placements,
      }));

      expect(
        layouts.every((layout) =>
          layout.placements.every((placement) =>
            isPlacementWithinLayerContract(placement, layout.index),
          ),
        ),
      ).toBe(true);
    });

    it('separates layer layouts by seed so different seeds never collide', () => {
      const far = buildStarLayerLayout(
        NEATRACING_STARFIELD_LAYER_SPECS[0]!,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
      );
      const mid = buildStarLayerLayout(
        NEATRACING_STARFIELD_LAYER_SPECS[1]!,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
      );
      const near = buildStarLayerLayout(
        NEATRACING_STARFIELD_LAYER_SPECS[2]!,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
      );

      expect(mid.placements).not.toEqual(far.placements);
      expect(near.placements).not.toEqual(mid.placements);
    });

    it('streams the same deterministic sequence from separate rng instances', () => {
      const seed = NEATRACING_STARFIELD_LAYER_SPECS[0]!.seed;
      const first = createStarfieldRng(seed);
      const second = createStarfieldRng(seed);

      expect(Array.from({ length: 5 }, () => second())).toEqual(
        Array.from({ length: 5 }, () => first()),
      );
    });
  });

  describe('G6 camera-driven scroll offsets', () => {
    it('scrolls each layer horizontally opposite the yaw scaled by its ratio', () => {
      const pose: StarfieldCameraPose = {
        yawRadians: Math.PI / 2,
        pitchRadians: 0,
      };
      const far = computeStarLayerOffsets(pose, 0);
      const mid = computeStarLayerOffsets(pose, 1);
      const near = computeStarLayerOffsets(pose, 2);

      expect(far.offsetX).toBeCloseTo(EXPECTED_FAR_OFFSET_X, 10);
      expect(mid.offsetX).toBeCloseTo(EXPECTED_MID_OFFSET_X, 10);
      expect(near.offsetX).toBeCloseTo(EXPECTED_NEAR_OFFSET_X, 10);
    });

    it('scrolls each layer vertically opposite the pitch so pitched-down cameras scroll stars up', () => {
      const pose: StarfieldCameraPose = {
        yawRadians: 0,
        pitchRadians: Math.PI / 2,
      };
      const far = computeStarLayerOffsets(pose, 0);
      const mid = computeStarLayerOffsets(pose, 1);
      const near = computeStarLayerOffsets(pose, 2);

      expect(far.offsetY).toBeCloseTo(EXPECTED_FAR_OFFSET_Y, 10);
      expect(mid.offsetY).toBeCloseTo(EXPECTED_MID_OFFSET_Y, 10);
      expect(near.offsetY).toBeCloseTo(EXPECTED_NEAR_OFFSET_Y, 10);
    });

    it('shears far/mid/near at different rates for the depth illusion', () => {
      const far = computeStarLayerOffsets(QUARTER_POSE, 0);
      const mid = computeStarLayerOffsets(QUARTER_POSE, 1);
      const near = computeStarLayerOffsets(QUARTER_POSE, 2);

      expect(
        Math.abs(near.offsetX) > Math.abs(mid.offsetX) &&
          Math.abs(mid.offsetX) > Math.abs(far.offsetX),
      ).toBe(true);
    });

    it('is a pure function of the camera pose — repeated calls return identical offsets', () => {
      expect(computeStarLayerOffsets(QUARTER_POSE, 1)).toEqual(
        computeStarLayerOffsets(QUARTER_POSE, 1),
      );
    });
  });

  describe('pass construction (GPU seam)', () => {
    it('precomputes every layer layout once at construction and reuses it across draws', () => {
      const harness = createMockGlHarness();
      const pass = createStarfieldPass(harness.gl);
      const before = pass.getLayerLayout(0);

      pass.draw(QUARTER_POSE);

      expect(pass.getLayerLayout(0)).toBe(before);
    });

    it('draws exactly one POINTS batch per layer with the pinned star counts', () => {
      const harness = createMockGlHarness();
      const pass = createStarfieldPass(harness.gl);

      pass.draw(QUARTER_POSE);

      expect(harness.drawCalls.length).toBe(3);
      expect(
        harness.drawCalls.every((call) => call.mode === MOCK_DRAW_MODE_POINTS),
      ).toBe(true);
      expect(harness.drawCalls.map((call) => call.count)).toEqual([35, 28, 23]);
    });

    it('disables depth testing and depth writes for the 2D backdrop', () => {
      const harness = createMockGlHarness();
      createStarfieldPass(harness.gl);

      expect(harness.disableCalls).toContain(MOCK_DEPTH_TEST);
      expect(harness.depthMaskValues).toContain(false);
    });

    it('feeds each layer the camera-derived scroll offsets as uniforms', () => {
      const harness = createMockGlHarness();
      const pass = createStarfieldPass(harness.gl);

      pass.draw(QUARTER_POSE);

      const recordedScrolls = harness.uniform2fCalls
        .filter((call) => call.name === 'u_scrollOffsetPx')
        .map((call) => [call.x, call.y]);
      const expectedScrolls = [0, 1, 2].map((layerIndex) => {
        const offsets = computeStarLayerOffsets(QUARTER_POSE, layerIndex);
        return [offsets.offsetX, offsets.offsetY];
      });

      expect(recordedScrolls).toEqual(expectedScrolls);
    });

    it('exposes the AC-040 warmup seam under the starfield registry key', () => {
      const harness = createMockGlHarness();

      const warmup = toStarfieldWarmupProgram(harness.gl);

      expect(warmup.key).toBe(NEATRACING_WARMED_PROGRAM_KEYS[0]);
    });

    it('draws the warmed pipeline offscreen through the shared pass draw path', () => {
      const harness = createMockGlHarness();
      const warmup = toStarfieldWarmupProgram(harness.gl);
      warmup.compileAndLink();
      const drawsBeforeWarmup = harness.drawCalls.length;

      warmup.drawOffscreen(harness.gl);

      expect(harness.drawCalls.length - drawsBeforeWarmup).toBe(3);
    });
  });
});
