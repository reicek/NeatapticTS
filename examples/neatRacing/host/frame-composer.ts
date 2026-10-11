/// <reference lib="dom" />

/**
 * Host-side frame composer for the Neat Racing voxel racer (slice
 * P1-frame-composer; plan Step 14; AC-002 extension).
 *
 * The composer is the single-scene-transfer carrier (batch 54): it owns
 * exactly one scene FBO plus its RGBA8 scene texture from construction,
 * draws the world passes into that target, and hands the owned texture to
 * the bloom handle as the composite's `sceneTexture` drawn onto the
 * default framebuffer. After every tick the composer restores the
 * frame-level framebuffer baseline — the host owns frame-scoped GL target
 * state per the batch 54 policy, while each pass keeps its own pass-local
 * GL state (G4/G8).
 *
 * Per-tick choreography is pinned to the warmup-key draw order from
 * `NEATRACING_WARMED_PROGRAM_KEYS`: starfield → grid → pulse → track →
 * car → glass (into the scene FBO), then exactly one bloom composite on
 * the default FB. The nine warmed keys map onto these seven handles
 * because the bloom handle owns its internally tested threshold → blur →
 * additive half-res chain (pinned by engine/bloom.pass.test.ts and
 * deliberately not re-asserted here).
 *
 * Zero per-frame allocation (AC-042): every draw-state bag is allocated
 * once at construction and refreshed in place per tick, so bag identity is
 * stable across ticks and no GPU program/shader/buffer/texture/framebuffer
 * is created on the steady-state tick path (no lazy mid-race compiles).
 * The AC-035 off-switch flag is forwarded verbatim on the bloom frame —
 * the zero-op guarantee lives inside BloomPass, so the composer never
 * drops passes itself.
 *
 * The track draw pass is not extracted yet (track/track.slabs.ts exports
 * only `buildTrackSlabs`), so the deps accept the minimal
 * {@link NeatRacingTrackPassSeam} draw handle; P1-drive-loop-live replaces
 * it with the real pass module without changing this seam.
 *
 * @module
 */

import type {
  StarfieldCameraPose,
  StarfieldPass,
} from '../engine/passes/starfield.pass';
import type {
  AmbientPulse,
  NeatRacingPulsePass,
} from '../engine/passes/pulse.pass';
import type { GridPass, GridPassDrawState } from '../engine/passes/grid.pass';
import type {
  CarPass,
  CarPassCarState,
  CarPassDrawState,
} from '../engine/passes/car.pass';
import type {
  GlassPass,
  GlassPassDrawState,
} from '../engine/passes/glass.pass';
import type {
  BloomPass,
  BloomPassDrawFrame,
} from '../engine/passes/bloom.pass';

/** `gl.FRAMEBUFFER` (0x8D40) target of every composer framebuffer bind. */
const GL_FRAMEBUFFER = 0x8d40;

/** `gl.TEXTURE_2D` (0x0DE1) target of the owned scene texture. */
const GL_TEXTURE_2D = 0x0de1;

/** `gl.RGBA` (0x1908) internal format and pixel format of the scene texture. */
const GL_RGBA = 0x1908;

/** `gl.UNSIGNED_BYTE` (0x1401) texel type of the scene texture. */
const GL_UNSIGNED_BYTE = 0x1401;

/** `gl.LINEAR` (0x2601) bilinear filter of the scene texture. */
const GL_LINEAR = 0x2601;

/** `gl.CLAMP_TO_EDGE` (0x812F) wrap mode of the scene texture. */
const GL_CLAMP_TO_EDGE = 0x812f;

/** `gl.TEXTURE_MIN_FILTER` (0x2801) sampler parameter. */
const GL_TEXTURE_MIN_FILTER = 0x2801;

/** `gl.TEXTURE_MAG_FILTER` (0x2800) sampler parameter. */
const GL_TEXTURE_MAG_FILTER = 0x2800;

/** `gl.TEXTURE_WRAP_S` (0x2802) sampler parameter. */
const GL_TEXTURE_WRAP_S = 0x2802;

/** `gl.TEXTURE_WRAP_T` (0x2803) sampler parameter. */
const GL_TEXTURE_WRAP_T = 0x2803;

/** `gl.COLOR_ATTACHMENT0` (0x8CE0) attachment point of the scene texture. */
const GL_COLOR_ATTACHMENT0 = 0x8ce0;

/** `gl.FRAMEBUFFER_COMPLETE` (0x8CD5) status required by the scene FBO. */
const GL_FRAMEBUFFER_COMPLETE = 0x8cd5;

/** `gl.COLOR_BUFFER_BIT` (0x4000) clear mask of the composer frame baseline. */
const GL_COLOR_BUFFER_BIT = 0x4000;

/** `gl.DEPTH_BUFFER_BIT` (0x100) clear mask for the scene depth attachment. */
const GL_DEPTH_BUFFER_BIT = 0x0100;

/** `gl.DEPTH_ATTACHMENT` (0x8D00) attachment point of the scene depth buffer. */
const GL_DEPTH_ATTACHMENT = 0x8d00;

/** `gl.RENDERBUFFER` (0x8D41) target used for the scene depth buffer. */
const GL_RENDERBUFFER = 0x8d41;

/** `gl.DEPTH_COMPONENT16` (0x81A5) internal format of the scene depth buffer. */
const GL_DEPTH_COMPONENT16 = 0x81a5;

/** Pass label prefixing every composer rejection message (bloom parity). */
const NEATRACING_FRAME_COMPOSER_LABEL = 'neatRacing frame composer';

/** Width of the dev-only scene-FBO content sample in pixels. */
const SCENE_CONTENT_SAMPLE_WIDTH = 8;

/** Height of the dev-only scene-FBO content sample in pixels. */
const SCENE_CONTENT_SAMPLE_HEIGHT = 8;

/** Column-major 4x4 matrix float count shared by every composer bag. */
const PINNED_MATRIX_FLOAT_COUNT = 16;

/** Neutral construction-time camera world placeholder `[x, y, z]` meters. */
const INITIAL_CAMERA_WORLD: readonly [number, number, number] = [0, 0, 0];

/** Neutral construction-time grid-bag camera X; refreshed before first draw. */
const INITIAL_BAG_CAMERA_X = 0;

/** Neutral construction-time grid-bag camera Z; refreshed before first draw. */
const INITIAL_BAG_CAMERA_Z = 0;

/** Neutral construction-time car payload reference; refreshed per tick. */
const INITIAL_BAG_CARS: readonly CarPassCarState[] = [];

/** Neutral construction-time pulse payload reference; refreshed per tick. */
const INITIAL_BAG_PULSES: readonly AmbientPulse[] = [];

/** Neutral construction-time bloom flag; refreshed before the first draw. */
const INITIAL_BAG_BLOOM_ENABLED = true;

/** One composed frame handed to the composer's draw seam. */
export interface NeatRacingFrameComposerFrame {
  /** Column-major 16-element view-projection matrix. */
  readonly viewProjection: Float32Array;
  /** Camera world position `[x, y, z]` in meters. */
  readonly cameraWorld: readonly [number, number, number];
  /** Starfield camera pose subset the background reacts to (G6). */
  readonly cameraPose: StarfieldCameraPose;
  /** Active ambient pulses for the frame, newest first. */
  readonly pulses: readonly AmbientPulse[];
  /** On-track cars in any order; car/glass passes sort far -> near internally. */
  readonly cars: readonly CarPassCarState[];
  /** AC-035 bloom off-switch, forwarded verbatim to the bloom frame. */
  readonly enabled: boolean;
}

/** Draw state forwarded to the not-yet-extracted track pass seam. */
export interface NeatRacingTrackPassDrawState {
  /** Column-major 16-element view-projection matrix. */
  readonly viewProjection: Float32Array;
  /** Camera world position `[x, y, z]` in meters. */
  readonly cameraWorld: readonly [number, number, number];
}

/**
 * Minimal draw-only structural seam for the not-yet-extracted track pass
 * (track/track.slabs.ts exposes only `buildTrackSlabs`, so importing a
 * track module here would break the module graph). P1-drive-loop-live
 * replaces it with the real pass module under this same shape.
 */
export interface NeatRacingTrackPassSeam {
  /** Forwards the frame camera state to the track slab renderer seam. */
  draw(state: NeatRacingTrackPassDrawState): void;
}

/** The seven pass handles the composer draws once per tick, in pinned order. */
export interface NeatRacingFrameComposerPasses {
  /** Starfield background pass (first world draw, pass-local GL state). */
  readonly starfield: StarfieldPass;
  /** Ground grid pass. */
  readonly grid: GridPass;
  /** Ambient pulse pass. */
  readonly pulse: NeatRacingPulsePass;
  /** Track slab renderer seam (draw-only). */
  readonly track: NeatRacingTrackPassSeam;
  /** Car body/wheel pass. */
  readonly car: CarPass;
  /** Glass pass. */
  readonly glass: GlassPass;
  /** Bloom post-processing pass (single scene-transfer consumer). */
  readonly bloom: BloomPass;
}

/** Dev-only assertion surface for the frame composer (AC-035 off-switch parity). */
export interface NeatRacingFrameComposerDevFlags {
  /**
   * When true, the composer reads back the scene FBO after the first world
   * render and throws a labeled error if every sampled pixel is transparent
   * black. This is a zero-op off-switch: false/undefined disables the
   * readback entirely so release builds pay no overhead.
   */
  readonly assertComposedFrameNonEmpty?: boolean;

  /**
   * When true, the composer logs the non-black pixel count in the scene FBO
   * after each world pass (starfield → grid → pulse → track → car → glass).
   * This is a dev-only diagnostic for locating which pass fails to contribute
   * pixels. It reuses a single `Uint8Array` buffer allocated at construction,
   * so the steady-state hot path pays no per-frame allocation. False or
   * undefined disables the readbacks entirely.
   */
  readonly assertPerPassPixelCounts?: boolean;

  /**
   * When true, the composer logs per-pass GL state and a wider pixel sample
   * after each world pass. This is a temporary deep diagnostic for the
   * real-GL empty-frame issue; false/undefined disables it.
   */
  readonly assertPerPassDiagnostics?: boolean;
}

/** Dependencies of the frame composer: one GL context plus the pass handles. */
export interface NeatRacingFrameComposerDeps {
  /** WebGL2 context sharing every pass handle. */
  readonly gl: WebGL2RenderingContext;
  /** The seven pass handles drawn in the pinned warmup-key order. */
  readonly passes: NeatRacingFrameComposerPasses;
  /** Optional dev-only assertion surface; omitted flags default to off. */
  readonly devFlags?: NeatRacingFrameComposerDevFlags;
}

/** Handle created by {@link createFrameComposer}. */
export interface NeatRacingFrameComposer {
  /**
   * Draws one composed frame: refreshes the identity-stable bags, renders
   * the world into the owned scene target, composites bloom onto the
   * default framebuffer, and restores the frame-level framebuffer baseline.
   *
   * @param frame - The composed frame payload for this tick.
   */
  drawFrame(frame: NeatRacingFrameComposerFrame): void;
  /**
   * Releases the composer-owned scene FBO and texture exactly once each.
   *
   * The injected pass handles are NOT disposed: the composer never owns
   * them, and pass lifetimes stay with their creation site.
   */
  dispose(): void;
}

/**
 * Converts one pass-facing draw state into its identity-stable local bag
 * mirror: identical member types with mutable members, so a per-tick
 * refresh writes values in place while every `draw()` call still receives
 * a structurally exact draw state (AC-042).
 */
type DrawStateBag<TDrawState extends object> = {
  -readonly [TKey in keyof TDrawState]: TDrawState[TKey];
};

/**
 * Pulse draw-state bag: the pulse program samples the matrix from plain
 * number storage (`readonly number[]` seam), so its matrix buffer is a
 * plain 16-float array written by a per-tick loop copy.
 */
interface PulseDrawStateBag {
  /** Active pulses for the tick (reference refreshed from the frame). */
  pulses: readonly AmbientPulse[];
  /** Plain 16-float view-projection copy (loop-copied, zero allocation). */
  viewProjection: number[];
}

/**
 * Identity-stable draw-state bags created once at construction (AC-042).
 * Bag identity persists across ticks; only member values are refreshed.
 */
interface NeatRacingFrameComposerDrawStateBags {
  /** Bag sent to {@link GridPass.draw}. */
  grid: DrawStateBag<GridPassDrawState>;
  /** Bag sent to {@link NeatRacingPulsePass.draw}. */
  pulse: PulseDrawStateBag;
  /** Bag sent to {@link NeatRacingTrackPassSeam.draw}. */
  track: DrawStateBag<NeatRacingTrackPassDrawState>;
  /** Bag sent to {@link CarPass.draw}. */
  car: DrawStateBag<CarPassDrawState>;
  /** Bag sent to {@link GlassPass.draw}. */
  glass: DrawStateBag<GlassPassDrawState>;
  /** Bag sent to {@link BloomPass.draw}. */
  bloom: DrawStateBag<BloomPassDrawFrame>;
}

/**
 * Composer-owned scene render target: the single-scene-transfer carrier
 * resources (one RGBA8 texture, one depth renderbuffer, plus their attach
 * FBO). The captured width/height own the per-frame viewport (the warmup
 * sweep leaves the global 1x1 viewport behind, and no world pass sets a
 * viewport itself).
 */
interface NeatRacingFrameComposerSceneTarget {
  /** Scene texture handed to the bloom composite stage per tick. */
  readonly texture: WebGLTexture;
  /** FBO the six world passes render into. */
  readonly framebuffer: WebGLFramebuffer;
  /** Depth renderbuffer that track, car and shadow passes read/write. */
  readonly depthRenderbuffer: WebGLRenderbuffer;
  /** Scene viewport width in pixels (canvas backing store at construction). */
  readonly width: number;
  /** Scene viewport height in pixels (canvas backing store at construction). */
  readonly height: number;
}

/**
 * Creates the frame composer: the host-side carrier that owns one scene
 * FBO + texture, draws the world passes into it once per tick in the
 * pinned warmup-key order, then composites bloom onto the default
 * framebuffer with the owned scene texture.
 *
 * @param deps - One WebGL2 context plus the seven pass handles; the
 *   composer never owns (and never disposes) the injected passes.
 * @returns The composer handle with `drawFrame` and `dispose`.
 *
 * @example
 * ```ts
 * const composer = createFrameComposer({ gl, passes });
 * composer.drawFrame({ viewProjection, cameraWorld, cameraPose, pulses, cars, enabled: true });
 * composer.dispose();
 * ```
 */
export function createFrameComposer(
  deps: NeatRacingFrameComposerDeps,
): NeatRacingFrameComposer {
  const { gl, passes } = deps;
  const devFlags = deps.devFlags ?? {};

  // Step 1: own the single-scene-transfer carrier (one FBO + texture).
  const scene = createSceneRenderTarget(gl);

  // Step 2: preallocate every identity-stable draw-state bag (AC-042).
  const bags = createDrawStateBags(gl, scene.texture);

  // Step 3: dev-only readback buffer reused by the per-pass pixel-count
  // diagnostic and the composed-frame tripwire (zero per-frame allocation).
  const composedFrameReadbackBuffer = devFlags.assertComposedFrameNonEmpty
    ? new Uint8Array(
        SCENE_CONTENT_SAMPLE_WIDTH * SCENE_CONTENT_SAMPLE_HEIGHT * 4,
      )
    : null;
  const perPassReadbackBuffer = devFlags.assertPerPassPixelCounts
    ? new Uint8Array(
        SCENE_CONTENT_SAMPLE_WIDTH * SCENE_CONTENT_SAMPLE_HEIGHT * 4,
      )
    : null;
  const perPassDiagnosticBuffer = devFlags.assertPerPassDiagnostics
    ? new Uint8Array(16 * 16 * 4)
    : null;

  let firstFrameAssertionDone = false;

  return {
    drawFrame(frame: NeatRacingFrameComposerFrame): void {
      // Step 1: refresh identity-stable bags in place (zero allocation).
      refreshDrawStateBags(bags, gl, frame);
      // Step 2: draw the world into the owned scene target (batch 54).
      renderWorldIntoSceneTarget(
        gl,
        scene,
        passes,
        bags,
        frame,
        perPassReadbackBuffer,
        perPassDiagnosticBuffer,
      );
      // Step 3: dev-only first-frame content assertion (zero-op off-switch).
      if (devFlags.assertComposedFrameNonEmpty && !firstFrameAssertionDone) {
        assertSceneFrameBufferHasNonEmptyContent(
          gl,
          scene,
          NEATRACING_FRAME_COMPOSER_LABEL,
          composedFrameReadbackBuffer,
        );
        firstFrameAssertionDone = true;
      }
      // Step 4: compose bloom on the default FB, then restore the baseline.
      composeBloomComposite(gl, passes.bloom, bags.bloom);
      restoreFrameFramebufferBaseline(gl);
    },
    dispose(): void {
      // Release exactly the owned handles once each; passes are not owned.
      gl.deleteFramebuffer(scene.framebuffer);
      gl.deleteTexture(scene.texture);
      gl.deleteRenderbuffer(scene.depthRenderbuffer);
    },
  };
}

/**
 * Creates the owned scene render target at construction: one RGBA8 texture
 * sized to the canvas backing store, attached to the FBO which is left
 * bound (the carrier state; no construction-time default-FB bind). The FBO
 * completeness is validated with `checkFramebufferStatus` (bloom-parity
 * loudness: a broken attachment can never degrade silently into a blank
 * composite) and the canvas-backed viewport dims are captured for the
 * per-frame scene viewport (the warmup sweep leaves the global 1x1
 * viewport behind, and no world pass sets a viewport itself).
 *
 * @param gl - WebGL2 context.
 * @returns The texture + framebuffer + viewport-dims bundle.
 * @throws When texture/framebuffer creation returns null or the FBO is
 *   not `GL_FRAMEBUFFER_COMPLETE`; every message carries the composer label.
 */
function createSceneRenderTarget(
  gl: WebGL2RenderingContext,
): NeatRacingFrameComposerSceneTarget {
  // Step 1: RGBA8 scene texture sized to the canvas backing store.
  const sceneWidth = gl.canvas.width;
  const sceneHeight = gl.canvas.height;
  const texture = gl.createTexture();
  if (texture === null) {
    throw new Error(
      `${NEATRACING_FRAME_COMPOSER_LABEL}: failed to create the scene texture`,
    );
  }
  gl.bindTexture(GL_TEXTURE_2D, texture);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gl.texImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGBA,
    sceneWidth,
    sceneHeight,
    0,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    null,
  );

  // Step 2: depth renderbuffer sized to the same backing-store dimensions.
  // Track, car and shadow passes enable `GL_DEPTH_TEST` with depth writes,
  // so a depth attachment is required or those passes produce no fragments.
  const depthRenderbuffer = gl.createRenderbuffer();
  if (depthRenderbuffer === null) {
    gl.deleteTexture(texture);
    throw new Error(
      `${NEATRACING_FRAME_COMPOSER_LABEL}: failed to create the scene depth renderbuffer`,
    );
  }
  gl.bindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
  gl.renderbufferStorage(
    GL_RENDERBUFFER,
    GL_DEPTH_COMPONENT16,
    sceneWidth,
    sceneHeight,
  );
  gl.bindRenderbuffer(GL_RENDERBUFFER, null);

  // Step 3: attach the texture and depth buffer to the owned FBO; leave it
  // bound. The completeness gate runs while the FBO is still bound (the
  // carrier state); a non-complete status throws instead of presenting blanks.
  const framebuffer = gl.createFramebuffer();
  if (framebuffer === null) {
    gl.deleteRenderbuffer(depthRenderbuffer);
    gl.deleteTexture(texture);
    throw new Error(
      `${NEATRACING_FRAME_COMPOSER_LABEL}: failed to create the scene framebuffer`,
    );
  }
  gl.bindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    GL_FRAMEBUFFER,
    GL_COLOR_ATTACHMENT0,
    GL_TEXTURE_2D,
    texture,
    0,
  );
  gl.framebufferRenderbuffer(
    GL_FRAMEBUFFER,
    GL_DEPTH_ATTACHMENT,
    GL_RENDERBUFFER,
    depthRenderbuffer,
  );
  const status = gl.checkFramebufferStatus(GL_FRAMEBUFFER);
  if (status !== GL_FRAMEBUFFER_COMPLETE) {
    // Bloom-mirror cleanup: release exactly the three just-created handles
    // before throwing, so a failed composer construction orphans nothing.
    gl.deleteFramebuffer(framebuffer);
    gl.deleteRenderbuffer(depthRenderbuffer);
    gl.deleteTexture(texture);
    throw new Error(
      `${NEATRACING_FRAME_COMPOSER_LABEL}: scene framebuffer incomplete (status ${String(status)})`,
    );
  }

  return {
    texture,
    framebuffer,
    depthRenderbuffer,
    width: sceneWidth,
    height: sceneHeight,
  };
}

/**
 * Creates the identity-stable draw-state bags once at construction. Every
 * bag view persists for the composer lifetime; the bloom bag starts with
 * the owned scene texture and neutral refresh targets.
 *
 * @param gl - WebGL2 context (backing-store dims seed the bloom bag).
 * @param sceneTexture - The owned scene texture of the initial bloom bag.
 * @returns The stable bag bundle.
 */
function createDrawStateBags(
  gl: WebGL2RenderingContext,
  sceneTexture: WebGLTexture,
): NeatRacingFrameComposerDrawStateBags {
  return {
    grid: {
      viewProjection: new Float32Array(PINNED_MATRIX_FLOAT_COUNT),
      cameraWorld: INITIAL_CAMERA_WORLD,
    },
    pulse: {
      pulses: INITIAL_BAG_PULSES,
      viewProjection: new Array<number>(PINNED_MATRIX_FLOAT_COUNT).fill(0),
    },
    track: {
      viewProjection: new Float32Array(PINNED_MATRIX_FLOAT_COUNT),
      cameraWorld: INITIAL_CAMERA_WORLD,
    },
    car: {
      viewProjection: new Float32Array(PINNED_MATRIX_FLOAT_COUNT),
      cameraWorld: INITIAL_CAMERA_WORLD,
      cars: INITIAL_BAG_CARS,
    },
    glass: {
      viewProjection: new Float32Array(PINNED_MATRIX_FLOAT_COUNT),
      cameraWorld: INITIAL_CAMERA_WORLD,
      cars: INITIAL_BAG_CARS,
    },
    bloom: {
      sceneTexture,
      targetWidth: gl.canvas.width,
      targetHeight: gl.canvas.height,
      enabled: INITIAL_BAG_BLOOM_ENABLED,
    },
  };
}

/**
 * Refreshes the identity-stable bags in place from one frame (pure member
 * writes; no object allocation, AC-042).
 *
 * @param bags - The stable bag bundle to refresh.
 * @param gl - WebGL2 context (backing-store dims for the bloom bag).
 * @param frame - The composed frame payload for this tick.
 */
function refreshDrawStateBags(
  bags: NeatRacingFrameComposerDrawStateBags,
  gl: WebGL2RenderingContext,
  frame: NeatRacingFrameComposerFrame,
): void {
  // Step 1: pinned matrices — typed-array copies into owned storage.
  bags.grid.viewProjection.set(frame.viewProjection);
  bags.track.viewProjection.set(frame.viewProjection);
  bags.car.viewProjection.set(frame.viewProjection);
  bags.glass.viewProjection.set(frame.viewProjection);
  // Step 2: pulse bag — loop copy into plain number storage.
  bags.pulse.pulses = frame.pulses;
  for (
    let matrixIndex = 0;
    matrixIndex < PINNED_MATRIX_FLOAT_COUNT;
    matrixIndex += 1
  ) {
    bags.pulse.viewProjection[matrixIndex] = frame.viewProjection[matrixIndex];
  }
  // Step 3: camera-derived scalars and car/pulse frame payloads. CarPassCarState
  // structurally satisfies the glass draw state (worldX/worldZ/yawRad/wheelSpinRad).
  bags.grid.cameraWorld = frame.cameraWorld;
  bags.track.cameraWorld = frame.cameraWorld;
  bags.car.cameraWorld = frame.cameraWorld;
  bags.car.cars = frame.cars;
  bags.glass.cameraWorld = frame.cameraWorld;
  bags.glass.cars = frame.cars;
  // Step 4: bloom composite frame — dims and AC-035 flag, verbatim.
  bags.bloom.targetWidth = gl.canvas.width;
  bags.bloom.targetHeight = gl.canvas.height;
  bags.bloom.enabled = frame.enabled;
}

/**
 * Reads back a small deterministic sample from the scene FBO and throws
 * a labeled composer error if every sampled pixel is transparent black.
 *
 * This is the dev-only tripwire that turns a silent blank-composite
 * failure into a loud, searchable exception on the first composed frame.
 * It intentionally samples the center of the scene target because the
 * surrounding letterbox area may legitimately stay black while the game
 * content is centered.
 *
 * @param gl - WebGL2 context.
 * @param scene - The owned scene render target.
 * @param label - Error label prefix.
 * @throws When every sampled pixel has zero RGBA.
 */
function countNonTransparentBlackPixels(pixels: Uint8Array): number {
  let count = 0;
  for (let pixelIndex = 0; pixelIndex < pixels.length; pixelIndex += 4) {
    if (
      pixels[pixelIndex] !== 0 ||
      pixels[pixelIndex + 1] !== 0 ||
      pixels[pixelIndex + 2] !== 0 ||
      pixels[pixelIndex + 3] !== 0
    ) {
      count += 1;
    }
  }
  return count;
}

function assertSceneFrameBufferHasNonEmptyContent(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  label: string,
  readbackBuffer: Uint8Array | null,
): void {
  const pixels =
    readbackBuffer ??
    new Uint8Array(
      SCENE_CONTENT_SAMPLE_WIDTH * SCENE_CONTENT_SAMPLE_HEIGHT * 4,
    );
  gl.bindFramebuffer(GL_FRAMEBUFFER, scene.framebuffer);

  // Sample 1: center (catches sky/starfield/pulse content).
  const centerX = Math.max(
    0,
    Math.floor((scene.width - SCENE_CONTENT_SAMPLE_WIDTH) / 2),
  );
  const centerY = Math.max(
    0,
    Math.floor((scene.height - SCENE_CONTENT_SAMPLE_HEIGHT) / 2),
  );
  gl.readPixels(
    centerX,
    centerY,
    SCENE_CONTENT_SAMPLE_WIDTH,
    SCENE_CONTENT_SAMPLE_HEIGHT,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    pixels,
  );
  if (countNonTransparentBlackPixels(pixels) > 0) {
    return;
  }

  // Sample 2: bottom-center (catches ground grid/track/car under the chase
  // camera, which leaves the center sample as empty sky). Reusing the same
  // buffer is safe because the center sample was fully transparent black.
  // Note: OpenGL framebuffer coordinates have the origin at the bottom-left,
  // so "lower" content is at low y values, not high y values.
  const bottomY = Math.max(
    0,
    Math.floor(scene.height * 0.25 - SCENE_CONTENT_SAMPLE_HEIGHT / 2),
  );
  gl.readPixels(
    centerX,
    bottomY,
    SCENE_CONTENT_SAMPLE_WIDTH,
    SCENE_CONTENT_SAMPLE_HEIGHT,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    pixels,
  );
  if (countNonTransparentBlackPixels(pixels) > 0) {
    return;
  }

  throw new Error(
    `${label}: composed frame is empty (all sampled pixels are transparent black; backing ${String(scene.width)}x${String(scene.height)})`,
  );
}

/**
 * Reads back a small deterministic sample from the scene FBO and logs the
 * number of non-transparent-black pixels for one world pass. This is the
 * per-pass diagnostic used to locate which pass fails to contribute pixels
 * after a camera-matrix or pass-state fix.
 *
 * @param gl - WebGL2 context.
 * @param scene - The owned scene render target.
 * @param passName - World pass name printed in the console log.
 * @param readbackBuffer - Reused `Uint8Array` buffer allocated at construction.
 */
function logPerPassPixelCount(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passName: string,
  readbackBuffer: Uint8Array,
): void {
  const sampleX = Math.max(
    0,
    Math.floor((scene.width - SCENE_CONTENT_SAMPLE_WIDTH) / 2),
  );
  const sampleY = Math.max(
    0,
    Math.floor((scene.height - SCENE_CONTENT_SAMPLE_HEIGHT) / 2),
  );
  gl.bindFramebuffer(GL_FRAMEBUFFER, scene.framebuffer);
  gl.readPixels(
    sampleX,
    sampleY,
    SCENE_CONTENT_SAMPLE_WIDTH,
    SCENE_CONTENT_SAMPLE_HEIGHT,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    readbackBuffer,
  );
  const count = countNonTransparentBlackPixels(readbackBuffer);
  console.log(
    `[${NEATRACING_FRAME_COMPOSER_LABEL}] ${passName} pixels: ${String(count)}`,
  );
}

/**
 * Ensures the scene FBO and its full-size viewport are currently bound.
 * Always rebinds both: framebuffer binds are cheap, and this guarantees a
 * previous pass that changed either target or viewport cannot silently blank
 * the next pass.
 *
 * @param gl - WebGL2 context.
 * @param scene - The owned scene render target.
 */
function bindSceneFrameBufferAndViewport(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
): void {
  gl.bindFramebuffer(GL_FRAMEBUFFER, scene.framebuffer);
  gl.viewport(0, 0, scene.width, scene.height);
}

/**
 * Renders the world passes into the owned scene target: binds the scene
 * FBO, resets the per-frame viewport to the full scene extent (the
 * `runShaderWarmup` sweep leaves the global viewport at the shared 1x1
 * warmup target and `updateRendererSize` only resizes the canvas backing
 * store — no world pass sets a viewport, so the composer owns it per the
 * batch 54 host policy), establishes a transparent-black color baseline
 * (the post-present drawing-buffer clearance the un-composed path
 * depended on, now explicit), resets the depth mask so the depth clear is
 * never silently masked off by the previous frame's `depthMask(false)`
 * pass state, and draws every world pass exactly once in the pinned
 * warmup-key order (starfield → grid → pulse → track → car → glass).
 *
 * Defensively re-binds the scene FBO and viewport immediately before each
 * world pass draw call so a pass that internally changes framebuffer state
 * cannot silently blank subsequent passes.
 *
 * @param gl - WebGL2 context.
 * @param scene - The owned scene render target.
 * @param passes - The injected pass handles.
 * @param bags - The refreshed draw-state bags.
 * @param frame - The composed frame payload (starfield camera pose seam).
 * @param perPassReadbackBuffer - Reused `Uint8Array` buffer for the dev-only
 *   per-pass pixel-count diagnostic, or null when disabled.
 */
function renderWorldIntoSceneTarget(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passes: NeatRacingFrameComposerPasses,
  bags: NeatRacingFrameComposerDrawStateBags,
  frame: NeatRacingFrameComposerFrame,
  perPassReadbackBuffer: Uint8Array | null,
  perPassDiagnosticBuffer: Uint8Array | null,
): void {
  gl.bindFramebuffer(GL_FRAMEBUFFER, scene.framebuffer);
  gl.viewport(0, 0, scene.width, scene.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clearDepth(1);
  // Reset the depth mask before clearing: `gl.clear` gates the depth-buffer
  // write behind `gl.depthMask`, and the world passes legitimately draw
  // with depth writes off (starfield/grid/pulse/glass leave `depthMask(false)`
  // behind). Without this reset, every frame after the first would clear
  // color but keep stale depth, and the static track/car/glass geometry
  // would fail the depth test against its own previous frame's depth —
  // the exact "track blinks away, only the spinning wheels survive"
  // failure mode observed on the served page.
  gl.depthMask(true);
  gl.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  bindSceneFrameBufferAndViewport(gl, scene);
  passes.starfield.draw(frame.cameraPose);
  logPerPassPixelCountIfEnabled(gl, scene, 'starfield', perPassReadbackBuffer);
  logPerPassDiagnosticsIfEnabled(
    gl,
    scene,
    'starfield',
    perPassDiagnosticBuffer,
  );
  bindSceneFrameBufferAndViewport(gl, scene);
  passes.grid.draw(bags.grid);
  logPerPassPixelCountIfEnabled(gl, scene, 'grid', perPassReadbackBuffer);
  logPerPassDiagnosticsIfEnabled(gl, scene, 'grid', perPassDiagnosticBuffer);
  bindSceneFrameBufferAndViewport(gl, scene);
  passes.pulse.draw(bags.pulse);
  logPerPassPixelCountIfEnabled(gl, scene, 'pulse', perPassReadbackBuffer);
  logPerPassDiagnosticsIfEnabled(gl, scene, 'pulse', perPassDiagnosticBuffer);
  bindSceneFrameBufferAndViewport(gl, scene);
  passes.track.draw(bags.track);
  logPerPassPixelCountIfEnabled(gl, scene, 'track', perPassReadbackBuffer);
  logPerPassDiagnosticsIfEnabled(gl, scene, 'track', perPassDiagnosticBuffer);
  bindSceneFrameBufferAndViewport(gl, scene);
  passes.car.draw(bags.car);
  logPerPassPixelCountIfEnabled(gl, scene, 'car', perPassReadbackBuffer);
  logPerPassDiagnosticsIfEnabled(gl, scene, 'car', perPassDiagnosticBuffer);
  bindSceneFrameBufferAndViewport(gl, scene);
  passes.glass.draw(bags.glass);
  logPerPassPixelCountIfEnabled(gl, scene, 'glass', perPassReadbackBuffer);
  logPerPassDiagnosticsIfEnabled(gl, scene, 'glass', perPassDiagnosticBuffer);
}

/**
 * Helper that conditionally logs the per-pass pixel count when the composer
 * was constructed with `assertPerPassPixelCounts` enabled. The buffer is
 * reused, so this adds no per-frame allocation on the steady-state hot path.
 *
 * @param gl - WebGL2 context.
 * @param scene - The owned scene render target.
 * @param passName - World pass name printed in the console log.
 * @param readbackBuffer - Reused `Uint8Array` buffer, or null when disabled.
 */
function logPerPassPixelCountIfEnabled(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passName: string,
  readbackBuffer: Uint8Array | null,
): void {
  if (readbackBuffer === null) {
    return;
  }
  logPerPassPixelCount(gl, scene, passName, readbackBuffer);
}

/**
 * Temporary deep diagnostic: after each world pass, logs the bound
 * framebuffer, viewport, GL error, and the non-transparent pixel count across
 * a 16x16 sample grid covering the whole scene target.
 */
function logPerPassDiagnosticsIfEnabled(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passName: string,
  readbackBuffer: Uint8Array | null,
): void {
  if (readbackBuffer === null) {
    return;
  }
  const framebufferBinding = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  const viewport = gl.getParameter(gl.VIEWPORT);
  const error = gl.getError();
  const gridW = 16;
  const gridH = 16;
  const sampleW = Math.floor(scene.width / gridW);
  const sampleH = Math.floor(scene.height / gridH);
  let anyNonTransparent = 0;
  let maxR = 0;
  let maxG = 0;
  let maxB = 0;
  let maxA = 0;
  for (let gy = 0; gy < gridH; gy += 1) {
    for (let gx = 0; gx < gridW; gx += 1) {
      gl.readPixels(
        gx * sampleW,
        gy * sampleH,
        1,
        1,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        readbackBuffer,
      );
      const r = readbackBuffer[0];
      const g = readbackBuffer[1];
      const b = readbackBuffer[2];
      const a = readbackBuffer[3];
      if (r !== 0 || g !== 0 || b !== 0 || a !== 0) {
        anyNonTransparent += 1;
      }
      if (r > maxR) maxR = r;
      if (g > maxG) maxG = g;
      if (b > maxB) maxB = b;
      if (a > maxA) maxA = a;
    }
  }
  console.info(
    `[neatRacing frame composer] ${passName} diagnostic: fb=${framebufferBinding === scene.framebuffer ? 'scene' : 'other'} viewport=[${viewport.join(',')}] error=${error} litCells=${anyNonTransparent}/${gridW * gridH} maxRGBA=[${maxR},${maxG},${maxB},${maxA}]`,
  );
}

/**
 * Composites the bloom stage onto the default framebuffer using the owned
 * scene texture as the transfer carrier (single scene transfer, batch 54).
 * The default framebuffer is cleared to transparent black before the
 * additive composite so the `blendFunc(GL_ONE, GL_ONE)` destination is
 * deterministic per frame — without it the presented result depends on
 * whatever the browser drawing buffer holds after the previous present
 * instead of being a pure `scene + bloom * intensity` transfer.
 *
 * @param gl - WebGL2 context.
 * @param bloom - The injected bloom pass handle.
 * @param bloomFrame - The refreshed bloom frame bag.
 */
function composeBloomComposite(
  gl: WebGL2RenderingContext,
  bloom: BloomPass,
  bloomFrame: DrawStateBag<BloomPassDrawFrame>,
): void {
  gl.bindFramebuffer(GL_FRAMEBUFFER, null);
  // Opaque host baseline so the additive bloom composite lands on a
  // deterministic surface and the canvas is visible against the page CSS.
  // Clearing to opaque black means the additive `blendFunc(GL_ONE, GL_ONE)`
  // destination starts at zero color but full alpha, so scene + bloom
  // accumulates onto a visible dark background instead of a transparent one.
  gl.clearColor(0, 0, 0, 1);
  gl.clear(GL_COLOR_BUFFER_BIT);
  bloom.draw(bloomFrame);
}

/**
 * Restores the host-owned frame-level framebuffer baseline (default FB)
 * after every tick; the bind is idempotent by the batch 54 host policy.
 *
 * @param gl - WebGL2 context.
 */
function restoreFrameFramebufferBaseline(gl: WebGL2RenderingContext): void {
  gl.bindFramebuffer(GL_FRAMEBUFFER, null);
}
