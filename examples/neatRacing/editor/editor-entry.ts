/// <reference lib="dom" />

/**
 * Browser entrypoint (boot) for the Neat Racing voxel model editor.
 *
 * Mirrors the game boot (`../browser-entry/browser-entry`): it installs
 * `window.neatRacingEditorStart`, which mounts the WebGL2 preview canvas into
 * the editor shell's preview host and returns a session handle. Boot
 * dogfoods the shared context service (`../engine/context.service`), so
 * context acquisition, once-per-canvas memoization, context-loss handling
 * (G5), and the WebGL2 fallback notice behave identically to the game page
 * (spec section 8.1, LD-1) — the entry never touches `canvas.getContext`
 * directly beyond what `initRenderingContext` does.
 *
 * The session payload carries the initialized editor state container
 * (`model: null`) so later tool slices (editor.state, editor.palette,
 * editor.camera) can bind onto the same session without re-plumbing the
 * shell (AC-020 partial). The boot sizes the preview canvas to the preview
 * host's client box once at mount (live resize is a later slice's job) and
 * degrades to the service's fallback notice instead of throwing when WebGL2
 * is unavailable (LD-1); stop stops the preview loop and unmounts both the
 * canvas and the notice.
 *
 * AC-019 preview lane: when the acquired context is live WebGL2 (unavailable
 * or structurally stubbed contexts keep the earlier boot shape), the boot
 * starts the RAF-driven preview render loop (`./editor-preview.render`)
 * under the Step 08 editor camera and wires the production bins draw adapter
 * over the engine's `createCarPass` (the selected document renders as the
 * car body; the wheel bin is empty). The boot also best-effort selects the
 * first listed dev-server model so the preview shows a scene out of the box;
 * every selection failure degrades silently (LD-1) to an empty selection.
 *
 * @module
 */

import { initRenderingContext } from '../engine/context.service';
import type { RenderingContextInitResult } from '../engine/context.service';
import type { CarPass, CarPassCarState } from '../engine/passes/car.pass';
import { createCarPass } from '../engine/passes/car.pass';
import type { VoxelBinModel } from '../voxel/bin.codec';
import { NEATRACING_TEAM_ID_CYAN } from '../voxel/team-remap';
import type { VoxelModelDocument } from '../voxel/model.schema';
import { startEditorPreviewRenderLoop } from './editor-preview.render';
import type {
  EditorPreviewBinsDraw,
  EditorPreviewBinsDrawState,
  EditorPreviewRenderLoopHandle,
} from './editor-preview.render';
import { createEditorCamera } from './editor.camera';

/**
 * Editor state container shared by the tool slices (AC-020 partial).
 *
 * @property model - Name of the model selected for editing; null while no
 *   model is selected yet.
 */
export interface NeatRacingEditorState {
  model: string | null;
}

/**
 * Session handle returned by the editor boot.
 *
 * Calling {@link NeatRacingEditorSession.stop} stops the AC-019 preview
 * render loop when one was started, then removes the mounted preview canvas
 * plus the tracked context-service fallback notice so an embedding page or
 * test can unmount the editor cleanly. The `state` container is fresh per
 * boot so successive boots never share state.
 */
export interface NeatRacingEditorSession {
  /** Initialized editor state container owned by the tool slices. */
  state: NeatRacingEditorState;
  /** Stops the preview loop, then unmounts the canvas and fallback notice. */
  stop: () => void;
}

/**
 * Boot function installed on `window` by the editor bundle.
 *
 * @param previewHostId - Editor shell element id that owns the preview
 *   canvas.
 * @returns A session with the fresh editor state and a stop handle.
 */
export type NeatRacingEditorStart = (
  previewHostId: string,
) => NeatRacingEditorSession;

/** Element id given to the preview canvas mounted by the editor boot. */
const NEATRACING_EDITOR_CANVAS_ID = 'neatracing-editor-canvas';

/** Pass label prefixing every observable preview-lane failure surface. */
const NEATRACING_EDITOR_PREVIEW_LABEL = 'neatRacing editor preview';

/** Per-dimension floor for the boot-time preview canvas size in pixels. */
const MIN_PREVIEW_CANVAS_SIZE = 1;

/** Background clear color channels for the one-time editor preview clear. */
const PREVIEW_CLEAR_RED = 0.08;
const PREVIEW_CLEAR_GREEN = 0.09;
const PREVIEW_CLEAR_BLUE = 0.14;
const PREVIEW_CLEAR_ALPHA = 1;

/** Valid empty wheel bin for the preview bins draw adapter.
 *
 * Wheel-spec grid shape (32-cubed, section 7.4/TP-14) with zero quads: the
 * car pass expands `quads.length === 0` into a zero-vertex mesh, so only the
 * selected document's bins render (as the body) and wheel mounts contribute
 * no geometry. Supplying the document also as `wheel` would draw it 4x
 * instead — the empty wheel keeps the preview an honest single render of the
 * document (see TP-14 preview-lane rationale).
 */
const PREVIEW_EMPTY_WHEEL_BIN: VoxelBinModel = {
  dims: [32, 32, 32],
  palette: [[0, 0, 0, 0, 0]],
  quads: [],
};

/** Centered, stationary cyan preview car pose for the bins draw adapter. */
const PREVIEW_CARS: readonly CarPassCarState[] = [
  {
    worldX: 0,
    worldZ: 0,
    yawRad: 0,
    teamColorId: NEATRACING_TEAM_ID_CYAN,
    wheelSpinRad: 0,
  },
];

/** Dev-server model list route used by the boot-time best-effort selection. */
const MODEL_LIST_API_PATH = '/api/models';

/** Per-frame preview clear mask: `gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT`.
 *
 * The preview has no full-coverage composer pass ahead of the car pass, so
 * the frame must start from a clean color AND depth buffer every draw — the
 * car pass is pass-local (it never resizes the viewport or clears), and the
 * translucent blob shadow re-blends whatever is already in the framebuffer. */
const PREVIEW_FRAME_CLEAR_MASK = 0x4100;

/**
 * Mount the WebGL2 preview canvas into the editor shell and return a session.
 *
 * The boot mounts only what the shell needs in this slice: the preview canvas
 * plus the context service's fallback notice when WebGL2 is unavailable. The
 * canvas is sized to the preview host's client box at boot (clamped to a
 * one-pixel drawable minimum by {@link MIN_PREVIEW_CANVAS_SIZE}, since live
 * resize is a later slice's job). Live WebGL2 contexts then start the AC-019
 * preview render loop (the tool frame stays a later slice's job). The fresh
 * context is painted once at boot by a guarded clear (viewport, clearColor,
 * clear) so the preview never shows stale default-framebuffer noise before
 * the loop's first frame.
 *
 * @param previewHostId - Editor shell element id that owns the preview
 *   canvas.
 * @returns A session with the fresh editor state container and a stop handle.
 * @throws When no element with `previewHostId` exists in the document.
 *
 * @example
 * ```ts
 * const session = window.neatRacingEditorStart('neatracing-editor-preview');
 * console.log(session.state); // { model: null }
 * session.stop(); // stops the preview loop and unmounts the canvas/notice
 * ```
 */
function neatRacingEditorStart(previewHostId: string): NeatRacingEditorSession {
  const previewHost = document.getElementById(previewHostId);
  if (!(previewHost instanceof HTMLElement)) {
    throw new Error(`Preview host #${previewHostId} not found`);
  }

  // Step 1: Mount the preview canvas into the preview host, sized to the
  // host's client box at boot. The floor constant keeps a host without a
  // measurable client box drawable; live resize is a later slice's job.
  const canvas = document.createElement('canvas');
  canvas.id = NEATRACING_EDITOR_CANVAS_ID;
  canvas.className = 'neatracing-editor-preview-canvas';
  canvas.width = Math.max(previewHost.clientWidth, MIN_PREVIEW_CANVAS_SIZE);
  canvas.height = Math.max(previewHost.clientHeight, MIN_PREVIEW_CANVAS_SIZE);
  // Fill the preview host's CSS box so the editor preview is plainly visible
  // at full host size instead of collapsing to the default 300x150 CSS box.
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  previewHost.appendChild(canvas);
  const hostRect = previewHost.getBoundingClientRect();
  console.info(
    `[${NEATRACING_EDITOR_PREVIEW_LABEL}] preview host geometry annotation:`,
    {
      hostRect: {
        x: hostRect.x,
        y: hostRect.y,
        width: hostRect.width,
        height: hostRect.height,
        top: hostRect.top,
        left: hostRect.left,
        right: hostRect.right,
        bottom: hostRect.bottom,
      },
      canvasBacking: { width: canvas.width, height: canvas.height },
      canvasCss: {
        width: canvas.style.width,
        height: canvas.style.height,
        display: canvas.style.display,
      },
      window: { width: window.innerWidth, height: window.innerHeight },
    },
  );

  // Step 2: Acquire the WebGL2 context through the shared service — the same
  // once-per-canvas, context-loss-aware path the game boot uses — and degrade
  // to the service's fallback notice instead of throwing (LD-1). Track the
  // notice so the stop handle can unmount it together with the canvas.
  const rendering = initRenderingContext({
    canvas,
    documentRef: document,
  });
  const fallbackElement = rendering.ok
    ? null
    : (rendering.fallbackElement ?? null);
  if (fallbackElement !== null) {
    previewHost.appendChild(fallbackElement);
  }

  // Step 3: One-time guarded clear — paint the preview background exactly
  // once at boot so the first frame starts from a coherent opaque canvas
  // instead of stale default-framebuffer noise. The render loop itself
  // starts in step 4, only on live GL2 contexts. The per-call optional
  // chaining keeps degenerate contexts (e.g. stubbed context objects
  // without GL methods) from throwing instead of clearing.
  if (rendering.ok && rendering.gl !== null) {
    const gl = rendering.gl;
    gl.viewport?.(0, 0, canvas.width, canvas.height);
    gl.clearColor?.(
      PREVIEW_CLEAR_RED,
      PREVIEW_CLEAR_GREEN,
      PREVIEW_CLEAR_BLUE,
      PREVIEW_CLEAR_ALPHA,
    );
    gl.clear?.(gl.COLOR_BUFFER_BIT);
  }

  // Step 4: Start the AC-019 preview lane — live WebGL2 only. Unavailable or
  // structurally stubbed contexts (jsdom-style `{}` contexts without GL
  // methods) keep the exact earlier boot shape: no camera, no RAF chain, no
  // draws, no fetch. The preview provider is a mutable closure the
  // best-effort boot-time model fetch fills in asynchronously; the loop
  // memoizes on document identity, so the fetch only affects the NEXT frame.
  let previewHandle: EditorPreviewRenderLoopHandle | null = null;
  const previewGl = pickPreviewRenderingContext(rendering);
  if (previewGl !== null) {
    const camera = createEditorCamera();
    const binsDraw = createEditorPreviewBinsDraw(previewGl, canvas);
    let currentDocument: VoxelModelDocument | null = null;
    const getSelectedModelDocument = (): VoxelModelDocument | null =>
      currentDocument;
    previewHandle = startEditorPreviewRenderLoop({
      canvas,
      camera,
      getSelectedModelDocument,
      binsDraw,
    });
    void loadFirstModelIntoProvider((document) => {
      currentDocument = document;
    });
  }

  // Step 5: Hand the tool slices a fresh state container per boot.
  return {
    state: { model: null },
    stop: () => {
      previewHandle?.stop();
      previewHandle = null;
      canvas.remove();
      fallbackElement?.remove();
    },
  };
}

/**
 * Pick the acquired rendering context when it is usable for the preview lane.
 *
 * The preview needs more than a context container: `createCarPass` compiles
 * programs eagerly, so the gate checks `createShader` structurally. This keeps
 * jsdom-style stub contexts (`{}` cast to `WebGL2RenderingContext`) and any
 * `ok: false` acquisition from starting the loop, exactly matching the stub
 * boot contracts pinned by editor-entry.test.ts.
 *
 * @param rendering - Shared context-service acquisition result.
 * @returns The live WebGL2 context, or null when the preview lane must not
 *   start.
 *
 * @example
 * ```ts
 * const gl = pickPreviewRenderingContext(rendering);
 * if (gl !== null) startPreview(gl);
 * ```
 */
function pickPreviewRenderingContext(
  rendering: RenderingContextInitResult,
): WebGL2RenderingContext | null {
  if (!rendering.ok) {
    return null;
  }
  const gl = rendering.gl;
  if (gl === null) {
    return null;
  }
  if (typeof gl.createShader !== 'function') {
    return null;
  }
  return gl;
}

/**
 * Build the production bins draw seam over the engine car pass.
 *
 * The adapter lazily builds a {@link CarPass} keyed on the bins identity: the
 * first draw (or any new bins object) disposes the previous pass and compiles
 * a pass rendering the preview bins as the car body over the empty wheel bin,
 * so exactly the selected document's geometry reaches the framebuffer. Car
 * passes are compiled eagerly and throw on unusable contexts; such build
 * failures keep the RAF chain alive (LD-1 — the preview lane must never
 * throw from a RAF frame) but are surfaced observably with the preview
 * label instead of the batch 54 silent no-op, so a blank preview can be
 * distinguished from a coarse probe by its console evidence.
 *
 * Every drawn frame also applies the pass-local-policy prelude the preview
 * host owns (the car pass never touches viewport or clear state — the game
 * composer path has a host prelude, the preview lane needs its own):
 * `viewport(0, 0, canvas.width, canvas.height)` plus a color AND depth clear.
 * Without the per-frame depth clear the translucent blob shadow keeps
 * re-blending into untouched depth, which burns the shadow in and ghosts
 * previous frames across every draw.
 *
 * @param gl - Live WebGL2 context the preview lane was gated on.
 * @param previewCanvas - Structural canvas view supplying the per-frame
 *   viewport extents (width/height only are read).
 * @returns The draw-only seam; its `dispose` releases the current pass.
 *
 * @example
 * ```ts
 * const binsDraw = createEditorPreviewBinsDraw(gl, canvas);
 * binsDraw.draw({ viewProjection, cameraWorld, bins });
 * binsDraw.dispose();
 * ```
 */
function createEditorPreviewBinsDraw(
  gl: WebGL2RenderingContext,
  previewCanvas: Pick<HTMLCanvasElement, 'width' | 'height'>,
): EditorPreviewBinsDraw {
  let currentBins: VoxelBinModel | null = null;
  let carPass: CarPass | null = null;
  return {
    draw(state: EditorPreviewBinsDrawState): void {
      if (state.bins !== currentBins) {
        carPass?.dispose();
        carPass = null;
        currentBins = state.bins;
        try {
          carPass = createCarPass(gl, {
            body: state.bins,
            wheel: PREVIEW_EMPTY_WHEEL_BIN,
          });
        } catch (error) {
          // Unusable context or rejected bins: draw nothing from now on,
          // but LOUDLY — the fully silent catch made a build failure
          // byte-identical to an absent model (LD-1). The preview lane
          // must never throw from the RAF frame, so the error is surfaced
          // through the labeled console surface and the pass stays null.
          console.error(
            `${NEATRACING_EDITOR_PREVIEW_LABEL}: preview car pass build failed; preview renders nothing`,
            error,
          );
          carPass = null;
        }
      }
      if (carPass === null) {
        return;
      }
      // Pass-local prelude (see type doc): reset the viewport to the full
      // drawing buffer and clear color + depth so each frame starts clean.
      gl.viewport(0, 0, previewCanvas.width, previewCanvas.height);
      gl.clear(PREVIEW_FRAME_CLEAR_MASK);
      carPass.draw({
        viewProjection: state.viewProjection,
        cameraWorld: state.cameraWorld,
        cars: PREVIEW_CARS,
      });
    },
    dispose(): void {
      carPass?.dispose();
      carPass = null;
      currentBins = null;
    },
  };
}

/**
 * Best-effort boot-time model selection for the manual smoke path.
 *
 * Fetches the dev server's `/api/models` list, then the first model's
 * document, and hands the raw document to the provider callback. The loop's
 * decode memo is the single normative validation gate, so the raw payload is
 * cached structurally unchecked. Every failure — missing fetch, non-JSON
 * body, non-OK response, empty list, missing document field — degrades
 * silently (LD-1): the editor simply previews nothing until a model gets
 * selected. Fire-and-forget from boot; the boot itself stays synchronous.
 *
 * @param setDocument - Provider sink called with the best-effort document.
 *
 * @example
 * ```ts
 * void loadFirstModelIntoProvider((document) => { current = document; });
 * ```
 */
async function loadFirstModelIntoProvider(
  setDocument: (document: VoxelModelDocument | null) => void,
): Promise<void> {
  try {
    if (typeof window.fetch !== 'function') {
      return;
    }
    const listResponse = await window.fetch(MODEL_LIST_API_PATH);
    if (!listResponse.ok) {
      return;
    }
    const listBody = (await listResponse.json()) as { models?: unknown };
    const models: readonly unknown[] = Array.isArray(listBody.models)
      ? listBody.models
      : [];
    const firstName = models.find(
      (name): name is string => typeof name === 'string' && name.length > 0,
    );
    if (firstName === undefined) {
      return;
    }
    const modelResponse = await window.fetch(
      `${MODEL_LIST_API_PATH}/${encodeURIComponent(firstName)}`,
    );
    if (!modelResponse.ok) {
      return;
    }
    const modelBody = (await modelResponse.json()) as { document?: unknown };
    if (modelBody.document === null || typeof modelBody.document !== 'object') {
      return;
    }
    setDocument(modelBody.document as VoxelModelDocument);
  } catch {
    // Silent degrade (LD-1): without the dev server API the editor simply
    // has no selection yet; the preview lane keeps ticking with no draws.
  }
}

declare global {
  interface Window {
    /** Neat Racing editor boot function installed by the editor bundle. */
    neatRacingEditorStart?: NeatRacingEditorStart;
  }
}

window.neatRacingEditorStart = neatRacingEditorStart;
