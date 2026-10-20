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
import { NEATRACING_FIXED_RENDER_HEIGHT } from '../../browser-entry/constants';
import {
  NEATRACING_SKY_BAND_HEIGHT_PX,
  NEATRACING_STARFIELD_BLINK_AMPLITUDE,
  NEATRACING_STARFIELD_BLINK_BASE_HZ,
  NEATRACING_STARFIELD_BLINK_BEAT_DELTA_HZ,
  NEATRACING_STARFIELD_CYAN_FILL_STYLE,
  NEATRACING_STARFIELD_FAR_STAR_COUNT,
  NEATRACING_STARFIELD_GLOOM_AMPLITUDE,
  NEATRACING_STARFIELD_LAYER_SPECS,
  NEATRACING_STARFIELD_MID_STAR_COUNT,
  NEATRACING_STARFIELD_NEAR_STAR_COUNT,
  NEATRACING_STARFIELD_TILE_HEIGHT_PX,
  NEATRACING_STARFIELD_TILE_WIDTH_PX,
  NEATRACING_VISUAL_SEED,
} from '../../browser-entry/constants.starfield';
import { NEATRACING_WARMED_PROGRAM_KEYS } from '../context.service';
import {
  buildStarLayerLayout,
  computeStarLayerOffsets,
  computeStarfieldBlink,
  computeStarfieldBlinkPhaseOffset,
  computeStarfieldGloomAlphaScale,
  computeStarfieldNdcFromWrappedTilePosition,
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
    starCount: NEATRACING_STARFIELD_FAR_STAR_COUNT,
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
    starCount: NEATRACING_STARFIELD_MID_STAR_COUNT,
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
    starCount: NEATRACING_STARFIELD_NEAR_STAR_COUNT,
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

/** Expected far-layer vertical offset: 0.25 × backdrop 720 × 0.03. */
const EXPECTED_FAR_OFFSET_Y = -5.4;

/** Expected mid-layer vertical offset: 0.25 × backdrop 720 × 0.04. */
const EXPECTED_MID_OFFSET_Y = -7.2;

/** Expected near-layer vertical offset: 0.25 × backdrop 720 × 0.05. */
const EXPECTED_NEAR_OFFSET_Y = -9;

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

/** One recorded uniform1f invocation from the mock context. */
interface RecordedUniform1f {
  readonly name: string;
  readonly x: number;
}

/** One recorded uniform3f invocation from the mock context. */
interface RecordedUniform3f {
  readonly name: string;
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

/** One recorded state-restore invocation from the mock context. */
interface RecordedStateCall {
  readonly method: string;
  readonly args: readonly unknown[];
}

/** Recording harness the GPU-seam tests drive and inspect. */
interface MockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly drawCalls: RecordedDrawCall[];
  readonly uniform2fCalls: RecordedUniform2f[];
  readonly uniform1fCalls: RecordedUniform1f[];
  readonly uniform3fCalls: RecordedUniform3f[];
  readonly disableCalls: number[];
  readonly depthMaskValues: boolean[];
  readonly stateRestoreCalls: RecordedStateCall[];
}

/**
 * Creates a recording mock WebGL2 context covering the shader pipeline,
 * VAO/buffer setup, blend/depth state, uniform uploads, and draw calls the
 * pass uses. Also records save/restore state traffic so we can assert the
 * right-world parity contract.
 *
 * @returns Harness exposing the mock context plus its recorded call logs.
 */
function createMockGlHarness(): MockGlHarness {
  const drawCalls: RecordedDrawCall[] = [];
  const uniform2fCalls: RecordedUniform2f[] = [];
  const uniform1fCalls: RecordedUniform1f[] = [];
  const uniform3fCalls: RecordedUniform3f[] = [];
  const disableCalls: number[] = [];
  const depthMaskValues: boolean[] = [];
  const stateRestoreCalls: RecordedStateCall[] = [];
  const gl = {
    POINTS: MOCK_DRAW_MODE_POINTS,
    DEPTH_TEST: MOCK_DEPTH_TEST,
    CULL_FACE: 0x0b44,
    BLEND: 0x0be2,
    SRC_ALPHA: 0x0302,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    DEPTH_WRITEMASK: 0x0b72,
    BLEND_SRC_RGB: 0x0301,
    BLEND_DST_RGB: 0x0302,
    BLEND_SRC_ALPHA: 0x0301,
    BLEND_DST_ALPHA: 0x0303,
    ACTIVE_TEXTURE: 0x84e0,
    CURRENT_PROGRAM: 0x8b8d,
    ARRAY_BUFFER_BINDING: 0x8894,
    TEXTURE_BINDING_2D: 0x8069,
    TEXTURE_2D: 0x0de1,
    VERTEX_ATTRIB_ARRAY_ENABLED: 0x8622,
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b20,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: 0x8892,
    STATIC_DRAW: 0x88e4,
    FLOAT: 0x1406,
    createShader: (): object => ({}),
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    createProgram: (): object => ({}),
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getUniformLocation: (_program: object, name: string): object => ({ name }),
    isEnabled: (): boolean => true,
    getParameter: (token: number): unknown => {
      switch (token) {
        case 0x0b72:
          return true;
        case 0x0301:
          return 0x0302;
        case 0x0302:
          return 0x0303;
        case 0x84e0:
          return 0x84e0;
        case 0x8b8d:
          return { name: 'prior-program' };
        case 0x8894:
          return { name: 'prior-buffer' };
        case 0x8069:
          return { name: 'prior-texture' };
        default:
          return 0;
      }
    },
    getVertexAttrib: (index: number, token: number): unknown => {
      if (token !== 0x8622) {
        return 0;
      }
      // Attribs 0 and 1 are enabled before the pass; 2 and 3 are disabled so
      // the save/restore path exercises both enable and disable restore calls.
      return index < 2;
    },
    useProgram: (program: { name?: string } | null): void => {
      stateRestoreCalls.push({ method: 'useProgram', args: [program?.name ?? null] });
    },
    createVertexArray: (): object => ({}),
    bindVertexArray: (vao: { name?: string } | null): void => {
      stateRestoreCalls.push({ method: 'bindVertexArray', args: [vao?.name ?? null] });
    },
    createBuffer: (): object => ({}),
    bindBuffer: (_target: number, buffer: { name?: string } | null): void => {
      stateRestoreCalls.push({
        method: 'bindBuffer',
        args: [buffer?.name ?? null],
      });
    },
    bufferData: (): void => undefined,
    bufferSubData: (): void => undefined,
    enableVertexAttribArray: (index: number): void => {
      stateRestoreCalls.push({ method: 'enableVertexAttribArray', args: [index] });
    },
    disableVertexAttribArray: (index: number): void => {
      stateRestoreCalls.push({
        method: 'disableVertexAttribArray',
        args: [index],
      });
    },
    vertexAttribPointer: (): void => undefined,
    enable: (token: number): void => {
      stateRestoreCalls.push({ method: 'enable', args: [token] });
    },
    disable: (token: number): void => {
      disableCalls.push(token);
      stateRestoreCalls.push({ method: 'disable', args: [token] });
    },
    blendFunc: (): void => undefined,
    blendFuncSeparate: (srcRgb: number, dstRgb: number, srcAlpha: number, dstAlpha: number): void => {
      stateRestoreCalls.push({
        method: 'blendFuncSeparate',
        args: [srcRgb, dstRgb, srcAlpha, dstAlpha],
      });
    },
    activeTexture: (unit: number): void => {
      stateRestoreCalls.push({ method: 'activeTexture', args: [unit] });
    },
    bindTexture: (_target: number, texture: { name?: string } | null): void => {
      stateRestoreCalls.push({
        method: 'bindTexture',
        args: [texture?.name ?? null],
      });
    },
    depthMask: (flag: boolean): void => {
      depthMaskValues.push(flag);
      stateRestoreCalls.push({ method: 'depthMask', args: [flag] });
    },
    uniform2f: (location: { name: string }, x: number, y: number): void => {
      uniform2fCalls.push({ name: location.name, x, y });
    },
    uniform1f: (location: { name: string }, x: number): void => {
      uniform1fCalls.push({ name: location.name, x });
    },
    uniform3f: (
      location: { name: string },
      r: number,
      g: number,
      b: number,
    ): void => {
      uniform3fCalls.push({ name: location.name, r, g, b });
    },
    drawArrays: (mode: number, first: number, count: number): void => {
      drawCalls.push({ mode, first, count });
    },
  };
  return {
    gl: gl as unknown as WebGL2RenderingContext,
    drawCalls,
    uniform2fCalls,
    uniform1fCalls,
    uniform3fCalls,
    disableCalls,
    depthMaskValues,
    stateRestoreCalls,
  };
}

/** Typed shape guard for one star placement within the layout contract. */
function isPlacementWithinLayerContract(
  placement: {
    readonly xPx: number;
    readonly yPx: number;
    readonly sizePx: number;
    readonly alpha: number;
    readonly phaseRadians: number;
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
    placement.alpha <= spec.maxAlpha &&
    placement.phaseRadians >= 0 &&
    placement.phaseRadians < 2.0 * Math.PI
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

    it('spans the full backdrop height so stars reach the horizon (AC-059)', () => {
      expect(NEATRACING_STARFIELD_TILE_HEIGHT_PX).toBe(
        NEATRACING_FIXED_RENDER_HEIGHT,
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
        1,
      );
      const second = buildStarLayerLayout(
        spec,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
        1,
      );

      expect(second.placements).toEqual(first.placements);
    });

    it('produces exactly starCount placements for every layer', () => {
      const layouts = NEATRACING_STARFIELD_LAYER_SPECS.map((spec, layerIndex) =>
        buildStarLayerLayout(
          spec,
          NEATRACING_STARFIELD_TILE_WIDTH_PX,
          NEATRACING_STARFIELD_TILE_HEIGHT_PX,
          layerIndex,
        ),
      );

      expect(layouts.map((layout) => layout.placements.length)).toEqual([
        NEATRACING_STARFIELD_FAR_STAR_COUNT,
        NEATRACING_STARFIELD_MID_STAR_COUNT,
        NEATRACING_STARFIELD_NEAR_STAR_COUNT,
      ]);
    });

    it('keeps every placement inside the tile and the layer size/alpha ranges', () => {
      const layouts = NEATRACING_STARFIELD_LAYER_SPECS.map((spec, index) => ({
        index,
        placements: buildStarLayerLayout(
          spec,
          NEATRACING_STARFIELD_TILE_WIDTH_PX,
          NEATRACING_STARFIELD_TILE_HEIGHT_PX,
          index,
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

    it('clamps star placement to the sky region and spans the full tile width (AC-059)', () => {
      const backdropHeightPx = NEATRACING_FIXED_RENDER_HEIGHT;
      const tileHeightPx = NEATRACING_STARFIELD_TILE_HEIGHT_PX;
      // The tile bottom maps to the horizon, so the tile-space horizon is the
      // smaller of the backdrop height and the tile height.
      const horizonTileYPx = Math.min(backdropHeightPx, tileHeightPx);
      const layouts = NEATRACING_STARFIELD_LAYER_SPECS.map((spec, layerIndex) =>
        buildStarLayerLayout(
          spec,
          NEATRACING_STARFIELD_TILE_WIDTH_PX,
          tileHeightPx,
          layerIndex,
          horizonTileYPx,
        ),
      );

      // No star is placed below the tile-space horizon line.
      expect(
        layouts.every((layout) =>
          layout.placements.every(
            (placement) => placement.yPx <= horizonTileYPx,
          ),
        ),
      ).toBe(true);

      // Stars must touch the bottom of the sky tile (continuity, not just a
      // sparse top band): at least one layer has a star within the last 1%
      // of the horizon y.
      expect(
        layouts.some((layout) =>
          layout.placements.some(
            (placement) => placement.yPx >= horizonTileYPx * 0.99,
          ),
        ),
      ).toBe(true);

      // Full-width x coverage: at least one layer has stars in both the left
      // and right deciles of the tile.
      const hasFullWidth = layouts.some((layout) => {
        const xs = layout.placements.map((placement) => placement.xPx);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        return (
          minX < NEATRACING_STARFIELD_TILE_WIDTH_PX * 0.1 &&
          maxX > NEATRACING_STARFIELD_TILE_WIDTH_PX * 0.9
        );
      });
      expect(hasFullWidth).toBe(true);
    });

    it('separates layer layouts by seed so different seeds never collide', () => {
      const far = buildStarLayerLayout(
        NEATRACING_STARFIELD_LAYER_SPECS[0]!,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
        0,
      );
      const mid = buildStarLayerLayout(
        NEATRACING_STARFIELD_LAYER_SPECS[1]!,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
        1,
      );
      const near = buildStarLayerLayout(
        NEATRACING_STARFIELD_LAYER_SPECS[2]!,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
        2,
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

  describe('AC-060 blink and gloom animation', () => {
    it('gives each star a deterministic phase offset from layer id and star index', () => {
      const a = computeStarfieldBlinkPhaseOffset(0, 0);
      const b = computeStarfieldBlinkPhaseOffset(0, 0);
      const c = computeStarfieldBlinkPhaseOffset(1, 0);

      expect(a).toEqual(b);
      expect(a).not.toEqual(c);
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThan(2.0 * Math.PI);
    });

    it('keeps the blink value in [-1, 1] and identical for identical inputs', () => {
      const t = 1.25;
      const first = computeStarfieldBlink(t, 1, 7);
      const second = computeStarfieldBlink(t, 1, 7);

      expect(first).toEqual(second);
      expect(first).toBeGreaterThanOrEqual(-1);
      expect(first).toBeLessThanOrEqual(1);
    });

    it('drives different blink beats on consecutive layers at the same time', () => {
      const t = 1.25;
      const starIndex = 7;
      const far = computeStarfieldBlink(t, 0, starIndex);
      const mid = computeStarfieldBlink(t, 1, starIndex);
      const near = computeStarfieldBlink(t, 2, starIndex);

      expect(new Set([far, mid, near]).size).toBe(3);
    });

    it('keeps the gloom alpha scale bounded and subtler than the blink amplitude', () => {
      const samples = [0, 0.33, 0.71, 1.0, 2.5];
      const gloomValues = samples.map((t) => computeStarfieldGloomAlphaScale(t));
      const minGloom = Math.min(...gloomValues);
      const maxGloom = Math.max(...gloomValues);

      expect(minGloom).toBeGreaterThanOrEqual(
        1.0 - NEATRACING_STARFIELD_GLOOM_AMPLITUDE,
      );
      expect(maxGloom).toBeLessThanOrEqual(1.0);
      expect(NEATRACING_STARFIELD_GLOOM_AMPLITUDE).toBeLessThan(
        NEATRACING_STARFIELD_BLINK_AMPLITUDE,
      );
      expect(
        NEATRACING_STARFIELD_BLINK_BASE_HZ >
          NEATRACING_STARFIELD_BLINK_BEAT_DELTA_HZ * 3,
      ).toBe(true);
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
      expect(harness.drawCalls.map((call) => call.count)).toEqual([
        NEATRACING_STARFIELD_FAR_STAR_COUNT,
        NEATRACING_STARFIELD_MID_STAR_COUNT,
        NEATRACING_STARFIELD_NEAR_STAR_COUNT,
      ]);
    });

    it('disables depth testing and depth writes for the 2D backdrop during draw', () => {
      const harness = createMockGlHarness();
      const pass = createStarfieldPass(harness.gl);

      pass.draw(QUARTER_POSE);

      expect(harness.disableCalls).toContain(MOCK_DEPTH_TEST);
      expect(harness.depthMaskValues).toContain(false);
    });

    it('uploads per-draw uniform1f values for time and per-layer blink frequency', () => {
      const harness = createMockGlHarness();
      const pass = createStarfieldPass(harness.gl);

      pass.draw(QUARTER_POSE);

      expect(harness.uniform1fCalls.some((call) => call.name === 'u_timeSeconds')).toBe(true);
      const blinkFreqCalls = harness.uniform1fCalls.filter(
        (call) => call.name === 'u_blinkAngularFrequency',
      );
      expect(blinkFreqCalls.length).toBe(3);
      expect(new Set(blinkFreqCalls.map((call) => call.x)).size).toBe(3);
    });

    it('uploads the static cyan star color via uniform3f at construction', () => {
      const harness = createMockGlHarness();
      createStarfieldPass(harness.gl);

      const colorCall = harness.uniform3fCalls.find(
        (call) => call.name === 'u_starColor',
      );
      expect(colorCall).toBeDefined();
      expect(colorCall!.r).toBeCloseTo(95 / 255, 10);
      expect(colorCall!.g).toBeCloseTo(255 / 255, 10);
      expect(colorCall!.b).toBeCloseTo(255 / 255, 10);
    });

    it('uploads the tile size and horizon NDC y at construction (OBS-E/OBS-F)', () => {
      const harness = createMockGlHarness();
      createStarfieldPass(harness.gl, {
        backdropHeightPx: 720,
        horizonNdcY: 0.20835,
      });

      const tileSizeCall = harness.uniform2fCalls.find(
        (call) => call.name === 'u_tileSizePx',
      );
      expect(tileSizeCall).toBeDefined();
      expect(tileSizeCall!.x).toBe(NEATRACING_STARFIELD_TILE_WIDTH_PX);
      expect(tileSizeCall!.y).toBe(NEATRACING_STARFIELD_TILE_HEIGHT_PX);

      const backdropCall = harness.uniform2fCalls.find(
        (call) => call.name === 'u_backdropSizePx',
      );
      expect(backdropCall).toBeUndefined();

      const horizonCall = harness.uniform1fCalls.find(
        (call) => call.name === 'u_horizonNdcY',
      );
      expect(horizonCall).toBeDefined();
      expect(horizonCall!.x).toBeCloseTo(0.20835, 10);
    });

    it('maps wrapped tile positions to full-width NDC using the tile size (OBS-G)', () => {
      const tileWidthPx = NEATRACING_STARFIELD_TILE_WIDTH_PX;
      const tileHeightPx = NEATRACING_STARFIELD_TILE_HEIGHT_PX;
      const horizonNdcY = 0.20835;

      const left = computeStarfieldNdcFromWrappedTilePosition(
        0,
        0,
        tileWidthPx,
        tileHeightPx,
        horizonNdcY,
      );
      expect(left.ndcX).toBeCloseTo(-1.0, 10);
      expect(left.ndcY).toBeCloseTo(1.0, 10);

      const right = computeStarfieldNdcFromWrappedTilePosition(
        tileWidthPx - 1,
        0,
        tileWidthPx,
        tileHeightPx,
        horizonNdcY,
      );
      expect(right.ndcX).toBeCloseTo(
        ((tileWidthPx - 1) / tileWidthPx) * 2.0 - 1.0,
        10,
      );
      expect(right.ndcY).toBeCloseTo(1.0, 10);

      const bottom = computeStarfieldNdcFromWrappedTilePosition(
        0,
        tileHeightPx,
        tileWidthPx,
        tileHeightPx,
        horizonNdcY,
      );
      expect(bottom.ndcY).toBeCloseTo(horizonNdcY, 10);
    });

    it('re-clamps and re-uploads layer layouts via setBackdropSize on resize (OBS-H)', () => {
      const harness = createMockGlHarness();
      const pass = createStarfieldPass(harness.gl, {
        backdropHeightPx: 720,
      });

      const originalHighPlacements = pass
        .getLayerLayout(1)
        .placements.filter((placement) => placement.yPx > 360);
      expect(originalHighPlacements.length).toBeGreaterThan(0);

      pass.setBackdropSize(1920, 360);

      for (let layerIndex = 0; layerIndex < 3; layerIndex += 1) {
        const layout = pass.getLayerLayout(layerIndex);
        for (const placement of layout.placements) {
          expect(placement.yPx).toBeLessThanOrEqual(360);
        }
      }

      const backdropCalls = harness.uniform2fCalls.filter(
        (call) => call.name === 'u_backdropSizePx',
      );
      expect(backdropCalls).toHaveLength(0);
    });

    it('saves and restores GL state at draw boundaries (right-world parity)', () => {
      const harness = createMockGlHarness();
      const pass = createStarfieldPass(harness.gl);

      pass.draw(QUARTER_POSE);

      const restoreCallNames = harness.stateRestoreCalls.map((call) => call.method);
      expect(restoreCallNames).toContain('useProgram');
      expect(restoreCallNames).toContain('bindBuffer');
      expect(restoreCallNames).toContain('activeTexture');
      expect(restoreCallNames).toContain('bindTexture');
      expect(restoreCallNames).toContain('blendFuncSeparate');
      expect(restoreCallNames).toContain('depthMask');
      expect(restoreCallNames).toContain('bindVertexArray');
      expect(restoreCallNames).toContain('enableVertexAttribArray');
      expect(restoreCallNames).toContain('disableVertexAttribArray');
      expect(restoreCallNames).toContain('enable');
      expect(restoreCallNames).toContain('disable');
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
