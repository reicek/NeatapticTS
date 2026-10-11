/**
 * RED contract for slice P1-frame-composer (plan Step 14; AC-002 extension).
 *
 * RED PREMISE (plan red_test field): this suite FAILS with
 * `error TS2307: Cannot find module './frame-composer'` while
 * `examples/neatRacing/host/frame-composer.ts` is absent — the sanctioned
 * module-absence red seam (batch 90/93 precedent). Any other failure mode
 * (fixture error, import typo against an existing module) is NOT a valid
 * red and must be fixed in the fixture, not implemented around.
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 *
 * 1. Pinned per-tick choreography: `createFrameComposer` draws every pass
 *    once per tick in the pinned warmup-key order from
 *    `NEATRACING_WARMED_PROGRAM_KEYS` — starfield -> grid -> pulse -> track
 *    -> car -> glass -> bloom. The nine warmup keys map onto these seven
 *    pass handles because the bloom handle owns the internally-tested
 *    threshold -> blur -> additive half-res chain (already pinned by
 *    engine/bloom.pass.test.ts and deliberately NOT re-asserted here).
 * 2. Single-scene-transfer carrier (batch 54): the composer owns one scene
 *    FBO + scene texture from construction, draws the world passes into
 *    it, and hands its texture to the bloom handle as the composite's
 *    `sceneTexture`, composited onto the default framebuffer.
 * 3. Frame-level GL restore (batch 54 host policy): the composer restores
 *    the pre-tick framebuffer baseline after every tick (host-owned state;
 *    passes stay pass-local per G4/G8).
 * 4. AC-035 plumbing: the off-switch flag is forwarded verbatim on the
 *    bloom frame; the composer never drops passes itself (the zero-op
 *    guarantee lives inside BloomPass).
 * 5. Zero per-frame allocations (AC-042): composer-owned draw-state bags
 *    keep identity across ticks with values refreshed from each frame; no
 *    GPU program/shader/buffer/texture/framebuffer creation happens on
 *    the steady-state tick path (no lazy mid-race compiles).
 * 6. Dispose releases the composer-owned scene GPU resources exactly once.
 *
 * Spy-double discipline: every engine pass module exists today and is
 * mirrored by a hand-rolled recording double (ESM harness parity with
 * bloom.pass.test.ts; no jest.fn in the ESM project). The track draw pass
 * has NO module yet (track/track.slabs.ts exports only `buildTrackSlabs`),
 * so the composer seam must accept a minimal draw-only track handle
 * (`TrackPassSeam` below) instead of importing a second absent module,
 * which would add a second TS2307 and mask the sanctioned red.
 *
 * The Jest project runs in a plain Node environment, so no GPU or DOM is
 * required.
 */
import { describe, expect, it } from '@jest/globals';
import {
  createFrameComposer,
  type NeatRacingFrameComposerFrame,
} from './frame-composer';
import type {
  StarfieldCameraPose,
  StarfieldLayerLayout,
  StarfieldPass,
} from '../engine/passes/starfield.pass';
import type { GridPass, GridPassDrawState } from '../engine/passes/grid.pass';
import type { NeatRacingPulsePass } from '../engine/passes/pulse.pass';
import type { CarPass, CarPassDrawState } from '../engine/passes/car.pass';
import type {
  GlassPass,
  GlassPassDrawState,
} from '../engine/passes/glass.pass';
import type {
  BloomPass,
  BloomPassDrawFrame,
} from '../engine/passes/bloom.pass';

/** `gl.FRAMEBUFFER` target enum (0x8D40) used by the FBO choreography pins. */
const MOCK_GL_FRAMEBUFFER = 0x8d40;

/** Mock canvas backing-store width in pixels (`gl.canvas.width`). */
const MOCK_CANVAS_WIDTH = 1280;

/** Mock canvas backing-store height in pixels (`gl.canvas.height`). */
const MOCK_CANVAS_HEIGHT = 720;

/** Column-major matrix A: mid-range diagonal, distinguishable from B. */
const VIEW_PROJECTION_A = ((): Float32Array => {
  const matrix = new Float32Array(16);
  matrix[0] = 0.25;
  matrix[5] = 0.5;
  matrix[10] = 0.75;
  matrix[15] = 1;
  return matrix;
})();

/** Column-major matrix B: identity-plus tail, distinguishable from A. */
const VIEW_PROJECTION_B = ((): Float32Array => {
  const matrix = new Float32Array(16);
  matrix[0] = 1;
  matrix[5] = 1;
  matrix[10] = 1;
  matrix[15] = 2;
  return matrix;
})();

/** Camera world position A in meters. */
const CAMERA_WORLD_A: readonly [number, number, number] = [0, 1, 10];

/** Camera world position B in meters (different X for bag-refresh pins). */
const CAMERA_WORLD_B: readonly [number, number, number] = [4, 1, 14];

/** Camera pose A fed to the starfield seam (G6). */
const CAMERA_POSE_A: StarfieldCameraPose = {
  yawRadians: 0.25,
  pitchRadians: -0.3,
};

/** Camera pose B fed to the starfield seam (G6). */
const CAMERA_POSE_B: StarfieldCameraPose = {
  yawRadians: -0.5,
  pitchRadians: 0.1,
};

/**
 * Builds one deterministic frame fixture. Frames A (defaults) and B
 * (overrides) share every field the composer plumbs through its seams.
 *
 * @param overrides - Per-test field overrides on top of fixture A.
 * @returns A fresh frame object (fixture A values by default).
 */
function makeFrame(
  overrides: Partial<NeatRacingFrameComposerFrame> = {},
): NeatRacingFrameComposerFrame {
  return {
    viewProjection: VIEW_PROJECTION_A,
    cameraWorld: CAMERA_WORLD_A,
    cameraPose: CAMERA_POSE_A,
    pulses: [],
    cars: [],
    enabled: true,
    ...overrides,
  };
}

/** One recorded bindFramebuffer call (target + handle or null for default). */
interface MockBindFramebufferRecord {
  readonly target: number;
  readonly framebuffer: WebGLFramebuffer | null;
}

/** One recorded viewport call (the composer's per-frame scene viewport). */
interface MockViewportRecord {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** One recorded clearColor call (the composer's transparent-black baseline). */
interface MockClearColorRecord {
  readonly red: number;
  readonly green: number;
  readonly blue: number;
  readonly alpha: number;
}

/** One recorded clear call plus the bound target at call time. */
interface MockClearRecord {
  /** Raw mask forwarded by the composer (COLOR_BUFFER_BIT | DEPTH_BUFFER_BIT = 0x4100 on scene, 0x4000 on default). */
  readonly mask: number;
  /** True when a non-default framebuffer was bound at clear time. */
  readonly onScene: boolean;
  /** Depth-writable state at clear time; GL silently drops the depth half of `clear` while `depthMask(false)`. */
  readonly depthWritable: boolean;
}

/** One recorded readPixels call (format/type are pinned by the composer). */
interface MockReadPixelsRecord {
  /** X origin of the readback rectangle in pixels. */
  readonly x: number;
  /** Y origin of the readback rectangle in pixels. */
  readonly y: number;
  /** Width of the readback rectangle in pixels. */
  readonly width: number;
  /** Height of the readback rectangle in pixels. */
  readonly height: number;
}

/** Overrides for driving the composer's rejection surfaces in isolation. */
interface MockHarnessOptions {
  /** Value returned by checkFramebufferStatus (default 0x8CD5 complete). */
  readonly framebufferStatus?: number;
  /** When true, createTexture returns null (texture-creation rejection). */
  readonly failSceneTextureCreation?: boolean;
  /** When true, createFramebuffer returns null (FBO-creation rejection). */
  readonly failSceneFramebufferCreation?: boolean;
  /** When true, createRenderbuffer returns null (depth-buffer rejection). */
  readonly failSceneDepthRenderbufferCreation?: boolean;
  /** Stubbed red channel for every pixel returned by readPixels (default 0). */
  readonly readPixelsRed?: number;
  /** Stubbed green channel for every pixel returned by readPixels (default 0). */
  readonly readPixelsGreen?: number;
  /** Stubbed blue channel for every pixel returned by readPixels (default 0). */
  readonly readPixelsBlue?: number;
  /** Stubbed alpha channel for every pixel returned by readPixels (default 0). */
  readonly readPixelsAlpha?: number;
}

/** Running count of GPU resource creations (zero-alloc seam surface). */
interface MockResourceCounts {
  createProgram: number;
  createShader: number;
  createBuffer: number;
  createTexture: number;
  createFramebuffer: number;
  createRenderbuffer: number;
}

/** Recording mock WebGL2 context covering the composer's GL surface. */
interface MockComposerGlHarness {
  readonly gl: WebGL2RenderingContext;
  /** Texture handles created via createTexture, in creation order. */
  readonly createdTextures: readonly WebGLTexture[];
  /** Framebuffer handles created via createFramebuffer, in order. */
  readonly createdFramebuffers: readonly WebGLFramebuffer[];
  /** Renderbuffer handles created via createRenderbuffer, in order. */
  readonly createdRenderbuffers: readonly WebGLRenderbuffer[];
  /** Texture handles released via deleteTexture, in release order. */
  readonly deletedTextures: readonly WebGLTexture[];
  /** Framebuffer handles released via deleteFramebuffer, in order. */
  readonly deletedFramebuffers: readonly WebGLFramebuffer[];
  /** Renderbuffer handles released via deleteRenderbuffer, in order. */
  readonly deletedRenderbuffers: readonly WebGLRenderbuffer[];
  /** Every bindFramebuffer call in order (target + handle/null). */
  readonly bindFramebufferRecords: readonly MockBindFramebufferRecord[];
  /** Every viewport call in order (the per-frame scene viewport). */
  readonly viewportRecords: readonly MockViewportRecord[];
  /** Every clearColor call in order (the composer's frame baselines). */
  readonly clearColorRecords: readonly MockClearColorRecord[];
  /** Every clear call in order (mask + bound target at call time). */
  readonly clearRecords: readonly MockClearRecord[];
  /** Every readPixels call in order. */
  readonly readPixelsRecords: readonly MockReadPixelsRecord[];
  /** Chronological choreography log: 'fbo:scene' | 'fbo:default' | 'draw:<key>'. */
  readonly timeline: string[];
  /** Creation counters; the steady-state tick path must not move these. */
  readonly resourceCounts: MockResourceCounts;
}

/**
 * Recording mock for the composer's GL seam: hand-rolled closures (no
 * jest.fn; ESM harness parity with bloom.pass.test.ts). bindFramebuffer
 * pushes a choreography token into the shared timeline so FBO binds and
 * pass draws interleave in true call order; viewport and clear calls emit
 * their own tokens ('viewport:scene', 'clear:scene', 'clear:default') so
 * the present-path prelude choreography is assertable in call order; every
 * GL entry point the passes or a frame-level state machine might touch is
 * present as a counted or silent no-op. Optional overrides drive the
 * scene-target rejection surfaces (creation nulls, incomplete status).
 *
 * @param options - Optional rejection-surface overrides for one test case.
 * @returns The recording harness for one test case.
 */
function createMockComposerGlHarness(
  options: MockHarnessOptions = {},
): MockComposerGlHarness {
  const createdTextures: WebGLTexture[] = [];
  const createdFramebuffers: WebGLFramebuffer[] = [];
  const createdRenderbuffers: WebGLRenderbuffer[] = [];
  const deletedTextures: WebGLTexture[] = [];
  const deletedFramebuffers: WebGLFramebuffer[] = [];
  const deletedRenderbuffers: WebGLRenderbuffer[] = [];
  const bindFramebufferRecords: MockBindFramebufferRecord[] = [];
  const viewportRecords: MockViewportRecord[] = [];
  const clearColorRecords: MockClearColorRecord[] = [];
  const clearRecords: MockClearRecord[] = [];
  const readPixelsRecords: MockReadPixelsRecord[] = [];
  const timeline: string[] = [];
  let boundFramebuffer: unknown = null;
  let depthWritable = true;
  const resourceCounts: MockResourceCounts = {
    createProgram: 0,
    createShader: 0,
    createBuffer: 0,
    createTexture: 0,
    createFramebuffer: 0,
    createRenderbuffer: 0,
  };
  const gl = {
    canvas: { width: MOCK_CANVAS_WIDTH, height: MOCK_CANVAS_HEIGHT },
    createShader: (): object => {
      resourceCounts.createShader += 1;
      return {};
    },
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    getShaderInfoLog: (): string | null => null,
    createProgram: (): object => {
      resourceCounts.createProgram += 1;
      return {};
    },
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getProgramInfoLog: (): string | null => null,
    getAttribLocation: (): number => 0,
    getUniformLocation: (_program: unknown, name: string): object => ({ name }),
    useProgram: (): void => undefined,
    enable: (): void => undefined,
    disable: (): void => undefined,
    depthMask: (flag: boolean): void => {
      depthWritable = flag;
    },
    cullFace: (): void => undefined,
    frontFace: (): void => undefined,
    blendFunc: (): void => undefined,
    createBuffer: (): object => {
      resourceCounts.createBuffer += 1;
      return {};
    },
    bindBuffer: (): void => undefined,
    bufferData: (): void => undefined,
    vertexAttribPointer: (): void => undefined,
    enableVertexAttribArray: (): void => undefined,
    uniformMatrix4fv: (): void => undefined,
    uniform1i: (): void => undefined,
    uniform2f: (): void => undefined,
    createTexture: (): object | null => {
      resourceCounts.createTexture += 1;
      if (options.failSceneTextureCreation === true) {
        return null;
      }
      const texture = {} as WebGLTexture;
      createdTextures.push(texture);
      return texture;
    },
    bindTexture: (): void => undefined,
    activeTexture: (): void => undefined,
    texParameteri: (): void => undefined,
    texImage2D: (): void => undefined,
    createFramebuffer: (): object | null => {
      resourceCounts.createFramebuffer += 1;
      if (options.failSceneFramebufferCreation === true) {
        return null;
      }
      const framebuffer = {} as WebGLFramebuffer;
      createdFramebuffers.push(framebuffer);
      return framebuffer;
    },
    bindFramebuffer: (target: number, framebuffer: unknown): void => {
      boundFramebuffer = framebuffer;
      bindFramebufferRecords.push({
        target,
        framebuffer: (framebuffer ?? null) as WebGLFramebuffer | null,
      });
      timeline.push(framebuffer === null ? 'fbo:default' : 'fbo:scene');
    },
    framebufferTexture2D: (): void => undefined,
    createRenderbuffer: (): object | null => {
      resourceCounts.createRenderbuffer += 1;
      if (options.failSceneDepthRenderbufferCreation === true) {
        return null;
      }
      const renderbuffer = {} as WebGLRenderbuffer;
      createdRenderbuffers.push(renderbuffer);
      return renderbuffer;
    },
    bindRenderbuffer: (): void => undefined,
    renderbufferStorage: (): void => undefined,
    framebufferRenderbuffer: (): void => undefined,
    checkFramebufferStatus: (): number => options.framebufferStatus ?? 0x8cd5,
    viewport: (x: number, y: number, width: number, height: number): void => {
      viewportRecords.push({ x, y, width, height });
      timeline.push('viewport:scene');
    },
    clear: (mask: number): void => {
      clearRecords.push({
        mask,
        onScene: boundFramebuffer !== null,
        depthWritable,
      });
      timeline.push(
        boundFramebuffer === null ? 'clear:default' : 'clear:scene',
      );
    },
    clearColor: (
      red: number,
      green: number,
      blue: number,
      alpha: number,
    ): void => {
      clearColorRecords.push({ red, green, blue, alpha });
    },
    clearDepth: (): void => undefined,
    readPixels: (
      x: number,
      y: number,
      width: number,
      height: number,
      _format: number,
      _type: number,
      pixels: Uint8Array,
    ): void => {
      readPixelsRecords.push({ x, y, width, height });
      for (let i = 0; i < pixels.length; i += 4) {
        pixels[i] = options.readPixelsRed ?? 0;
        pixels[i + 1] = options.readPixelsGreen ?? 0;
        pixels[i + 2] = options.readPixelsBlue ?? 0;
        pixels[i + 3] = options.readPixelsAlpha ?? 0;
      }
    },
    getParameter: (parameter: number): unknown => {
      if (parameter === 0x8ca6) {
        return boundFramebuffer;
      }
      return null;
    },
    drawArrays: (): void => undefined,
    deleteProgram: (): void => undefined,
    deleteShader: (): void => undefined,
    deleteBuffer: (): void => undefined,
    deleteTexture: (texture: unknown): void => {
      deletedTextures.push(texture as WebGLTexture);
    },
    deleteFramebuffer: (framebuffer: unknown): void => {
      deletedFramebuffers.push(framebuffer as WebGLFramebuffer);
    },
    deleteRenderbuffer: (renderbuffer: unknown): void => {
      deletedRenderbuffers.push(renderbuffer as WebGLRenderbuffer);
    },
  } as unknown as WebGL2RenderingContext;
  return {
    gl,
    createdTextures,
    createdFramebuffers,
    createdRenderbuffers,
    deletedTextures,
    deletedFramebuffers,
    deletedRenderbuffers,
    bindFramebufferRecords,
    viewportRecords,
    clearColorRecords,
    clearRecords,
    readPixelsRecords,
    timeline,
    resourceCounts,
  };
}

/**
 * Draw state the composer forwards to the injected track seam. The track
 * draw pass is not extracted yet (track/track.slabs.ts exports only
 * `buildTrackSlabs`), so the composer deps accept this minimal structural
 * handle; P1-drive-loop-live replaces it with the real pass module.
 */
interface TrackPassSeam {
  /** Forwards the frame camera state to the track slab renderer seam. */
  draw: (state: {
    readonly viewProjection: Float32Array;
    readonly cameraWorld: readonly [number, number, number];
  }) => void;
}

/** Recorded pass doubles, keyed by the pinned warmup-key draw order. */
interface RecordingPassHandles {
  readonly starfield: StarfieldPass;
  readonly grid: GridPass;
  readonly pulse: NeatRacingPulsePass;
  readonly track: TrackPassSeam;
  readonly car: CarPass;
  readonly glass: GlassPass;
  readonly bloom: BloomPass;
  /** Pass keys in true draw call order (one entry per pass draw). */
  readonly drawOrder: readonly string[];
  /** Camera poses observed by the starfield seam, in call order. */
  readonly starfieldDrawCalls: readonly StarfieldCameraPose[];
  /** Draw states observed by the grid seam, in call order. */
  readonly gridDrawStates: readonly GridPassDrawState[];
  /** Draw states observed by the car seam, in call order. */
  readonly carDrawStates: readonly CarPassDrawState[];
  /** Bloom frames observed by the bloom seam, in call order. */
  readonly bloomDrawFrames: readonly BloomPassDrawFrame[];
}

/**
 * Builds recording doubles for the seven pass handles. Each draw records
 * its key into `drawOrder` and its argument into its owner array; dispose
 * handles stay silent because the composer does not own the passes.
 *
 * @param timeline - The harness choreography log to append draw tags to.
 * @returns The recording doubles for one test case.
 */
function createRecordingPasses(timeline: string[]): RecordingPassHandles {
  const drawOrder: string[] = [];
  const starfieldDrawCalls: StarfieldCameraPose[] = [];
  const gridDrawStates: GridPassDrawState[] = [];
  const trackDrawStates: {
    readonly viewProjection: Float32Array;
    readonly cameraWorld: readonly [number, number, number];
  }[] = [];
  const carDrawStates: CarPassDrawState[] = [];
  const glassDrawStates: GlassPassDrawState[] = [];
  const bloomDrawFrames: BloomPassDrawFrame[] = [];
  const pulseDrawFrames: {
    pulses: readonly unknown[];
    viewProjection: readonly number[];
  }[] = [];
  const starfield: StarfieldPass = {
    key: 'starfield',
    draw: (cameraPose): void => {
      drawOrder.push('starfield');
      timeline.push('draw:starfield');
      starfieldDrawCalls.push(cameraPose);
    },
    getLayerLayout: (): StarfieldLayerLayout =>
      ({}) as unknown as StarfieldLayerLayout,
  };
  const grid: GridPass = {
    draw: (state): void => {
      drawOrder.push('grid');
      timeline.push('draw:grid');
      gridDrawStates.push(state);
    },
  };
  const pulse: NeatRacingPulsePass = {
    programKey: 'pulse',
    draw: (frame): void => {
      drawOrder.push('pulse');
      timeline.push('draw:pulse');
      pulseDrawFrames.push(frame);
    },
    dispose: (): void => undefined,
  };
  const track: TrackPassSeam = {
    draw: (state): void => {
      drawOrder.push('track');
      timeline.push('draw:track');
      trackDrawStates.push(state);
    },
  };
  const car: CarPass = {
    draw: (state): void => {
      drawOrder.push('car');
      timeline.push('draw:car');
      carDrawStates.push(state);
    },
    dispose: (): void => undefined,
  };
  const glass: GlassPass = {
    draw: (state): void => {
      drawOrder.push('glass');
      timeline.push('draw:glass');
      glassDrawStates.push(state);
    },
    dispose: (): void => undefined,
  };
  const bloom: BloomPass = {
    draw: (frame): void => {
      drawOrder.push('bloom');
      timeline.push('draw:bloom');
      bloomDrawFrames.push(frame);
    },
    dispose: (): void => undefined,
  };
  return {
    starfield,
    grid,
    pulse,
    track,
    car,
    glass,
    bloom,
    drawOrder,
    starfieldDrawCalls,
    gridDrawStates,
    carDrawStates,
    bloomDrawFrames,
  };
}

/** Pinned warmup-key draw order mirrored from NEATRACING_WARMED_PROGRAM_KEYS. */
const PINNED_DRAW_ORDER: readonly string[] = [
  'starfield',
  'grid',
  'pulse',
  'track',
  'car',
  'glass',
  'bloom',
];

/** Sums the counted GPU creations into one steady-stability invariant. */
function totalResourceCreations(harness: MockComposerGlHarness): number {
  return (
    harness.resourceCounts.createProgram +
    harness.resourceCounts.createShader +
    harness.resourceCounts.createBuffer +
    harness.resourceCounts.createTexture +
    harness.resourceCounts.createFramebuffer +
    harness.resourceCounts.createRenderbuffer
  );
}

/**
 * Builds the composer from one harness plus its recording pass doubles
 * (the seven-seam plumb block, shared by every present-path contract).
 *
 * @param harness - The recording mock GL harness for this test case.
 * @param handles - The recording pass doubles from the same harness.
 * @returns The composer handle under test.
 */
function createTestComposer(
  harness: MockComposerGlHarness,
  handles: RecordingPassHandles,
) {
  return createFrameComposer({
    gl: harness.gl,
    passes: {
      starfield: handles.starfield,
      grid: handles.grid,
      pulse: handles.pulse,
      track: handles.track,
      car: handles.car,
      glass: handles.glass,
      bloom: handles.bloom,
    },
  });
}

/**
 * Temporarily replaces `console.log` so tests can assert dev-only
 * diagnostic output without emitting noise into the test runner.
 *
 * @param run - Callback that receives the captured messages.
 * @returns Whatever `run` returns.
 */
function withCapturedConsoleLog<T>(run: (messages: string[]) => T): T {
  const originalLog = console.log;
  const messages: string[] = [];
  console.log = (...args: unknown[]): void => {
    messages.push(args.join(' '));
  };
  try {
    return run(messages);
  } finally {
    console.log = originalLog;
  }
}

describe('createFrameComposer (slice P1-frame-composer — AC-002 extension)', () => {
  describe('per-tick warmup-key choreography', () => {
    it('draws every pass exactly once per tick in the pinned warmup-key order', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      // Act
      composer.drawFrame(makeFrame());
      // Assert — order AND exactly-once-per-pass pinned by exact equality.
      expect(handles.drawOrder).toEqual(PINNED_DRAW_ORDER);
    });
  });

  describe('single-scene-transfer carrier (batch 54)', () => {
    it('creates one scene framebuffer, texture, and depth renderbuffer at construction and feeds the texture to bloom', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      // Act
      composer.drawFrame(makeFrame());
      // Assert — ownership created once, transferred exactly once.
      expect(harness.createdFramebuffers).toHaveLength(1);
      expect(harness.createdTextures).toHaveLength(1);
      expect(harness.createdRenderbuffers).toHaveLength(1);
      expect(handles.bloomDrawFrames[0]?.sceneTexture).toBe(
        harness.createdTextures[0],
      );
    });

    it('binds the scene FBO before the first world draw and the default FB before bloom', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      // Act
      composer.drawFrame(makeFrame());
      // Assert — choreography indices over the interleaved timeline.
      const glassIndex = harness.timeline.indexOf('draw:glass');
      const bloomIndex = harness.timeline.indexOf('draw:bloom');
      const firstDefaultBind = harness.timeline.indexOf('fbo:default');
      expect(harness.timeline[0]).toBe('fbo:scene');
      expect(firstDefaultBind).toBeGreaterThan(glassIndex);
      expect(firstDefaultBind).toBeLessThan(bloomIndex);
    });

    it('restores the pre-tick framebuffer baseline after each tick', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      // Act
      composer.drawFrame(makeFrame());
      const afterTickOne = harness.bindFramebufferRecords.at(-1);
      composer.drawFrame(makeFrame());
      const afterTickTwo = harness.bindFramebufferRecords.at(-1);
      // Assert — frame-level GL state restored host-side both ticks.
      expect(afterTickOne?.framebuffer).toBeNull();
      expect(afterTickTwo?.framebuffer).toBeNull();
      expect(afterTickTwo?.target).toBe(MOCK_GL_FRAMEBUFFER);
    });
  });

  describe('AC-035 off-switch plumbing', () => {
    it('plumbs canvas dims and the off-switch flag verbatim into the bloom frame', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      // Act
      composer.drawFrame(makeFrame({ enabled: false }));
      // Assert — flag plumb-through, not composer-side dropping.
      expect(handles.bloomDrawFrames[0]?.enabled).toBe(false);
      expect(handles.bloomDrawFrames[0]?.targetWidth).toBe(MOCK_CANVAS_WIDTH);
      expect(handles.bloomDrawFrames[0]?.targetHeight).toBe(MOCK_CANVAS_HEIGHT);
    });

    it('keeps all seven pass handles drawing while the off-switch is false', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      // Act
      composer.drawFrame(makeFrame({ enabled: false }));
      // Assert — the composer never drops passes; BloomPass owns AC-035.
      expect(handles.drawOrder).toEqual(PINNED_DRAW_ORDER);
    });
  });

  describe('zero per-frame allocation seam (AC-042)', () => {
    it('reuses stable draw-state bags across ticks with values refreshed per frame', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      // Act
      composer.drawFrame(makeFrame());
      composer.drawFrame(
        makeFrame({
          viewProjection: VIEW_PROJECTION_B,
          cameraWorld: CAMERA_WORLD_B,
          cameraPose: CAMERA_POSE_B,
        }),
      );
      // Assert — bag identity stable; values carry frame B.
      const gridFirst = handles.gridDrawStates[0];
      const gridSecond = handles.gridDrawStates[1];
      expect(gridSecond).toBe(gridFirst);
      expect(gridSecond?.cameraWorld).toBe(CAMERA_WORLD_B);
      expect(handles.carDrawStates[1]).toBe(handles.carDrawStates[0]);
    });

    it('performs no GPU resource creation across steady-state ticks', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      composer.drawFrame(makeFrame());
      const creationsAfterWarmTick = totalResourceCreations(harness);
      // Act
      composer.drawFrame(
        makeFrame({
          viewProjection: VIEW_PROJECTION_B,
          cameraWorld: CAMERA_WORLD_B,
          cameraPose: CAMERA_POSE_B,
        }),
      );
      // Assert — no lazy mid-race program/shader/buffer/target creation.
      expect(totalResourceCreations(harness)).toBe(creationsAfterWarmTick);
    });
  });

  describe('dispose', () => {
    it('releases the composer-owned scene GPU resources exactly once', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
      });
      composer.drawFrame(makeFrame());
      // Act
      composer.dispose();
      // Assert — exactly the owned handles, released once each.
      expect(harness.deletedFramebuffers).toEqual([
        harness.createdFramebuffers[0],
      ]);
      expect(harness.deletedTextures).toEqual([harness.createdTextures[0]]);
      expect(harness.deletedRenderbuffers).toEqual([
        harness.createdRenderbuffers[0],
      ]);
    });
  });

  describe('present-path viewport + baseline ownership (fix-packet-P1-composition-green)', () => {
    it('sets the full-scene viewport before every world pass draw and once for the frame baseline', () => {
      // Arrange — two ticks so the ownership is proven per-frame, not
      // once-at-construction (the warmup sweep re-pins viewport 1x1 after
      // every composite, so the composer must re-own it every frame).
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createTestComposer(harness, handles);
      // Act
      composer.drawFrame(makeFrame());
      composer.drawFrame(makeFrame());
      // Assert — frame-level prelude: construction bind, per-frame bind,
      // viewport, clear. Then each world pass is preceded by its own
      // scene-FBO rebind + viewport so a pass cannot blank later passes.
      expect(harness.timeline[0]).toBe('fbo:scene');
      const firstPreludeIndex = harness.timeline.indexOf('fbo:scene', 1);
      expect(
        harness.timeline.slice(firstPreludeIndex, firstPreludeIndex + 3),
      ).toEqual(['fbo:scene', 'viewport:scene', 'clear:scene']);
      for (const passKey of PINNED_DRAW_ORDER.slice(0, -1)) {
        const drawIndex = harness.timeline.indexOf(`draw:${passKey}`);
        expect(harness.timeline[drawIndex - 1]).toBe('viewport:scene');
        expect(harness.timeline[drawIndex - 2]).toBe('fbo:scene');
      }
      // One viewport per world pass plus the initial frame-level viewport.
      const expectedViewportsPerTick = PINNED_DRAW_ORDER.length;
      expect(harness.viewportRecords).toHaveLength(
        expectedViewportsPerTick * 2,
      );
      for (const record of harness.viewportRecords) {
        expect(record).toEqual({
          x: 0,
          y: 0,
          width: MOCK_CANVAS_WIDTH,
          height: MOCK_CANVAS_HEIGHT,
        });
      }
    });

    it('clears the scene target transparent and the default FB opaque-black on the composite path', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createTestComposer(harness, handles);
      // Act
      composer.drawFrame(makeFrame());
      // Assert — the scene target stays transparent black (additive world
      // passes write their own alpha), while the default framebuffer is
      // cleared opaque so the additive bloom composite lands on a
      // deterministic surface and the canvas is visible against the page CSS.
      expect(harness.clearColorRecords).toEqual([
        { red: 0, green: 0, blue: 0, alpha: 0 },
        { red: 0, green: 0, blue: 0, alpha: 1 },
      ]);
      expect(harness.clearRecords).toEqual([
        { mask: 0x4100, onScene: true, depthWritable: true },
        { mask: 0x4000, onScene: false, depthWritable: true },
      ]);
      const defaultBindIndex = harness.timeline.indexOf('fbo:default');
      const defaultClearIndex = harness.timeline.indexOf('clear:default');
      const bloomIndex = harness.timeline.indexOf('draw:bloom');
      expect(defaultBindIndex).toBeGreaterThan(0);
      expect(defaultClearIndex).toBe(defaultBindIndex + 1);
      expect(defaultClearIndex).toBeLessThan(bloomIndex);
    });

    it('resets the depth mask before the scene clear so leaked pass state cannot mask off the depth clear', () => {
      // Arrange — simulate the previous frame's leak. The real starfield,
      // grid, pulse, and glass passes all draw with depth writes disabled
      // and leave `depthMask(false)` behind on the shared context; WebGL
      // silently drops the depth half of `gl.clear` while the mask is off.
      // Without a composer-owned reset, every frame after the first keeps
      // stale depth and the static track/car/glass geometry fails the
      // depth test against its own previous-frame depth — the served-page
      // failure mode where the track "blinks away" and only the spinning
      // wheel pixels survive.
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createTestComposer(harness, handles);
      harness.gl.depthMask(false);
      // Act
      composer.drawFrame(makeFrame());
      // Assert — the scene clear must observe depth writes re-enabled by
      // the composer's pre-clear baseline, not the leaked pass state.
      const sceneClear = harness.clearRecords.find((record) => record.onScene);
      expect(sceneClear).toBeDefined();
      expect(sceneClear?.depthWritable).toBe(true);
    });

    it('throws the labeled rejection when the scene framebuffer is incomplete', () => {
      // Arrange — a driver-reported incomplete attachment (0x8CDD, GL_FRAMEBUFFER_UNSUPPORTED).
      const harness = createMockComposerGlHarness({
        framebufferStatus: 0x8cdd,
      });
      const handles = createRecordingPasses(harness.timeline);
      // Act / Assert — loud composer-labeled failure instead of blank output;
      // the failed construction releases exactly the three just-created handles
      // (bloom mirror — no orphaned GPU resources on the rejection path).
      expect(() => createTestComposer(harness, handles)).toThrow(
        /neatRacing frame composer: scene framebuffer incomplete \(status 36061\)/u,
      );
      expect(harness.deletedFramebuffers).toEqual(harness.createdFramebuffers);
      expect(harness.deletedTextures).toEqual(harness.createdTextures);
      expect(harness.deletedRenderbuffers).toEqual(
        harness.createdRenderbuffers,
      );
    });

    it('throws the labeled rejection when the scene texture cannot be created', () => {
      // Arrange
      const harness = createMockComposerGlHarness({
        failSceneTextureCreation: true,
      });
      const handles = createRecordingPasses(harness.timeline);
      // Act / Assert
      expect(() => createTestComposer(harness, handles)).toThrow(
        /neatRacing frame composer: failed to create the scene texture/u,
      );
    });

    it('throws the labeled rejection when the scene framebuffer cannot be created', () => {
      // Arrange
      const harness = createMockComposerGlHarness({
        failSceneFramebufferCreation: true,
      });
      const handles = createRecordingPasses(harness.timeline);
      // Act / Assert
      expect(() => createTestComposer(harness, handles)).toThrow(
        /neatRacing frame composer: failed to create the scene framebuffer/u,
      );
    });
  });

  describe('dev-only composed-frame content assertion (fix-packet-P1-composition-green)', () => {
    it('throws a labeled error on the first frame when the scene FBO is transparent black', () => {
      // Arrange — default readPixels returns transparent black, so the
      // first composed frame looks empty.
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
        devFlags: { assertComposedFrameNonEmpty: true },
      });
      // Act / Assert
      expect(() => composer.drawFrame(makeFrame())).toThrow(
        /neatRacing frame composer: composed frame is empty/u,
      );
    });

    it('passes the assertion when the sampled pixels carry any non-zero channel', () => {
      // Arrange — stub a single non-zero channel in every sampled pixel.
      const harness = createMockComposerGlHarness({ readPixelsAlpha: 1 });
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
        devFlags: { assertComposedFrameNonEmpty: true },
      });
      // Act / Assert — no throw; the assertion reads back a centered sample.
      expect(() => composer.drawFrame(makeFrame())).not.toThrow();
      expect(harness.readPixelsRecords).toHaveLength(1);
      const readback = harness.readPixelsRecords[0];
      expect(readback.width).toBe(8);
      expect(readback.height).toBe(8);
      expect(readback.x).toBeGreaterThanOrEqual(0);
      expect(readback.y).toBeGreaterThanOrEqual(0);
    });

    it('does not read back pixels when the dev assertion flag is omitted', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createTestComposer(harness, handles);
      // Act
      composer.drawFrame(makeFrame());
      // Assert — no readback, no throw; the off-switch defaults to zero-op.
      expect(harness.readPixelsRecords).toHaveLength(0);
    });

    it('does not read back pixels on the second frame even when the flag is on', () => {
      // Arrange
      const harness = createMockComposerGlHarness({ readPixelsAlpha: 255 });
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
        devFlags: { assertComposedFrameNonEmpty: true },
      });
      // Act
      composer.drawFrame(makeFrame());
      composer.drawFrame(makeFrame());
      // Assert — assertion runs once and only once.
      expect(harness.readPixelsRecords).toHaveLength(1);
    });
  });

  describe('defensive scene-FBO rebind before every world pass (fix-packet-P1-composition-green)', () => {
    it('rebinds the scene FBO before each world pass so a pass cannot blank the next one', () => {
      // Arrange — recording pass doubles; no pass mutates framebuffer state.
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createTestComposer(harness, handles);
      // Act
      composer.drawFrame(makeFrame());
      // Assert — every world pass is preceded by an fbo:scene token in true
      // call order; the bloom composite pass sits on the default FB.
      for (const passKey of PINNED_DRAW_ORDER.slice(0, -1)) {
        const drawIndex = harness.timeline.indexOf(`draw:${passKey}`);
        expect(harness.timeline[drawIndex - 1]).toBe('viewport:scene');
        expect(harness.timeline[drawIndex - 2]).toBe('fbo:scene');
      }
    });
  });

  describe('dev-only per-pass pixel-count diagnostics (fix-packet-P1-composition-green-iteration-4)', () => {
    it('logs a pixel count after each world pass when the flag is on', () => {
      // Arrange — every sampled pixel has a non-zero alpha channel.
      const harness = createMockComposerGlHarness({ readPixelsAlpha: 1 });
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
        devFlags: { assertPerPassPixelCounts: true },
      });

      // Act / Assert — six world passes, six readbacks, six logs, no noise.
      const logs = withCapturedConsoleLog((captured) => {
        composer.drawFrame(makeFrame());
        return captured;
      });

      expect(harness.readPixelsRecords).toHaveLength(6);
      expect(logs).toEqual([
        '[neatRacing frame composer] starfield pixels: 64',
        '[neatRacing frame composer] grid pixels: 64',
        '[neatRacing frame composer] pulse pixels: 64',
        '[neatRacing frame composer] track pixels: 64',
        '[neatRacing frame composer] car pixels: 64',
        '[neatRacing frame composer] glass pixels: 64',
      ]);
    });

    it('logs zero counts when every sampled pixel is transparent black', () => {
      // Arrange — default readPixels returns transparent black.
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createFrameComposer({
        gl: harness.gl,
        passes: {
          starfield: handles.starfield,
          grid: handles.grid,
          pulse: handles.pulse,
          track: handles.track,
          car: handles.car,
          glass: handles.glass,
          bloom: handles.bloom,
        },
        devFlags: { assertPerPassPixelCounts: true },
      });

      // Act
      const logs = withCapturedConsoleLog((captured) => {
        composer.drawFrame(makeFrame());
        return captured;
      });

      // Assert — every pass reports zero non-black pixels.
      expect(logs).toHaveLength(6);
      expect(logs.every((message) => message.endsWith('pixels: 0'))).toBe(true);
    });

    it('performs no readbacks and emits no logs when the flag is omitted', () => {
      // Arrange
      const harness = createMockComposerGlHarness();
      const handles = createRecordingPasses(harness.timeline);
      const composer = createTestComposer(harness, handles);

      // Act
      const logs = withCapturedConsoleLog((captured) => {
        composer.drawFrame(makeFrame());
        return captured;
      });

      // Assert — zero-op off-switch: no readback, no console output.
      expect(harness.readPixelsRecords).toHaveLength(0);
      expect(logs).toHaveLength(0);
    });
  });
});
