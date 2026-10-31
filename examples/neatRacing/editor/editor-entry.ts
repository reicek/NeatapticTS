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
 * The session payload carries the editor tool frame (`./editor.toolframe`):
 * the tool frame owns the editor state container (the dense working volume
 * plus its palette), the bounded undo stack, the mirror-symmetry surface,
 * and the palette/undo/model-dropdown/CAD-view/symmetry control mounts in
 * the reserved `neatracing-editor-tools` container, and it performs the
 * initial model load (model-store list, read, and decode into the working
 * volume). The session's `state` container is a live projection of that
 * editor state, so later tool slices (editor.edit.loop, editor.controls)
 * bind onto the same session without re-plumbing the shell (AC-020
 * partial). The session's `currentDocument` is the same kind of live
 * projection over the loaded model document, so a dropdown-driven
 * `loadModel()` reaches the preview and the session without re-editing the
 * boot. The boot takes an optional second argument
 * (`NeatRacingEditorBootDeps`) whose `fetch` field injects the model-store
 * fetch seam; it defaults to the ambient `globalThis.fetch`, so production
 * call sites are unchanged.
 *
 * The control BEHAVIORS live in `./editor.controls`: the palette
 * add/select handlers, the undo/redo buttons, the model dropdown, and the
 * CAD-view buttons. The boot wires them onto the tool frame's mounts and
 * hands the session the live `controls` surface plus the `camera` they
 * drive. The controls own the shared paint entry and the undo seam the
 * pointer edit loop captures its per-stroke snapshot through, so a paint
 * follows the selected palette color and a new edit invalidates the redo
 * history. The boot publishes the WORKING-VOLUME projection (the 128-cubed
 * section 7.2 document the pointer picker inverts) whenever the volume
 * changed — at boot, after a model load, after an undo/redo, and after a
 * full rebuild — so the preview and the picker agree on the lattice
 * (review obs 1). The boot camera starts on a working-volume framing
 * instead of the camera's game-scale default distance (review obs 5), and
 * the stopped flag cancels the rebuild pipeline's pending debounced flush
 * when the session is stopped (review obs 3).
 * The boot sizes the preview canvas to the preview host's client
 * box once at mount (live resize is a later slice's job) and degrades to
 * the service's fallback notice instead of throwing when WebGL2 is
 * unavailable (LD-1); stop stops the preview loop and unmounts the
 * tool frame, the canvas, and the notice.
 *
 * AC-019 preview lane: when the acquired context is live WebGL2 (unavailable
 * or structurally stubbed contexts keep the earlier boot shape), the boot
 * starts the RAF-driven preview render loop (`./editor-preview.render`)
 * under the Step 08 editor camera and wires the production bins draw adapter
 * over the engine's `createCarPass` (the selected document renders as the
 * car body; the wheel bin is empty). The preview renders the model the
 * tool frame loaded, so the served page shows a scene out of the box
 * without a second model-store round trip; every selection failure
 * degrades silently (LD-1) to an empty selection.
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
import type { EditorCamera } from './editor.camera';
import { createEditorControls } from './editor.controls';
import type { EditorControls } from './editor.controls';
import { createEditorToolframe } from './editor.toolframe';
import type {
  EditorToolframeFetch,
  EditorToolframeSession,
} from './editor.toolframe';
import { NEATRACING_EDITOR_VOLUME_EDGE } from './editor.state';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import { createEditorRebuildPipeline } from './editor.rebuild.pipeline';
import {
  NEATRACING_EDITOR_DEFAULT_MODEL_NAME,
  buildEditorVolumeDocument,
  createEditorEditLoop,
  createEditorVoxelPicker,
} from './editor.edit.loop';
import type { EditorEditLoop } from './editor.edit.loop';
import { createEditorLiveSave, createEditorLiveSymmetry } from './editor.live.save';
import type {
  EditorLiveSave,
  EditorLiveSaveFetch,
} from './editor.live.save';

/**
 * Editor state container shared by the tool slices (AC-020 partial).
 *
 * @property model - Name of the model selected for editing; null while no
 *   model is selected yet.
 */
export interface NeatRacingEditorState {
  readonly model: string | null;
}

/**
 * Optional boot dependencies for the editor boot.
 *
 * The boot injects the model-store fetch seam through this object so the
 * initial model load chain (model-store list, read, decode into the working
 * volume) can be exercised end to end without a dev server. Production call
 * sites omit it and keep the ambient `globalThis.fetch`.
 *
 * @property fetch - Fetch seam the tool frame drives; defaults to the
 *   ambient `globalThis.fetch`.
 * @property saveFetch - Fetch seam the immediate live save drives; it is
 *   the WRITE seam (the model store's save route needs a `PUT`), so it
 *   stays separate from the tool frame's GET-only seam. Defaults to the
 *   ambient `globalThis.fetch`.
 */
export interface NeatRacingEditorBootDeps {
  readonly fetch?: EditorToolframeFetch;
  readonly saveFetch?: EditorLiveSaveFetch;
}

/**
 * Session handle returned by the editor boot.
 *
 * Calling {@link NeatRacingEditorSession.stop} stops the AC-019 preview
 * render loop when one was started, then removes the mounted tool frame,
 * the preview canvas, and the tracked context-service fallback notice so an
 * embedding page or test can unmount the editor cleanly. The `state`
 * container is fresh per boot so successive boots never share state.
 */
export interface NeatRacingEditorSession {
  /** Live projection of the tool frame's editor state container. */
  state: NeatRacingEditorState;
  /**
   * Live projection of the tool frame's loaded model document: the document
   * currently decoded into the working volume, or null while no model is
   * loaded.
   */
  readonly currentDocument: VoxelModelDocument | null;
  /**
   * Live projection of the EDITED working volume: the section 7.2 model
   * document the rebuild pipeline published after the last full rebuild,
   * or null while no full rebuild ran yet (the pointer-pick edit loop
   * publishes it on every pointer-up flush).
   */
  readonly editedDocument: VoxelModelDocument | null;
  /**
   * Pointer-pick edit loop bound to the preview canvas: the canvas
   * pointer chain (pick -> symmetry-canonicalized paint -> undo push ->
   * rebuild pipeline edit -> pointer-up full flush).
   */
  readonly editLoop: EditorEditLoop;
  /**
   * Wired control behaviors over the tool frame's mounts: the palette
   * add/select handlers, the undo/redo buttons, the model dropdown, and the
   * CAD-view buttons (AC-020 completion).
   */
  readonly controls: EditorControls;
  /**
   * Immediate live save over the working volume: every edit flight
   * enqueues one save that PUTs the edited volume into the dev-server
   * model store and triggers the rebuild flight (AC-021 completion).
   */
  readonly liveSave: EditorLiveSave;
  /**
   * Editor camera the preview renders with and the CAD-view buttons drive
   * (AC-019).
   */
  readonly camera: EditorCamera;
  /** Editor tool frame (state container, controls, initial model load). */
  toolframe: EditorToolframeSession;
  /** Stops the preview loop, then unmounts the tool frame and the canvas. */
  stop: () => void;
}

/**
 * Boot function installed on `window` by the editor bundle.
 *
 * @param previewHostId - Editor shell element id that owns the preview
 *   canvas.
 * @param deps - Optional boot dependencies (the model-store fetch seam).
 * @returns A session with the fresh editor state and a stop handle.
 */
export type NeatRacingEditorStart = (
  previewHostId: string,
  deps?: NeatRacingEditorBootDeps,
) => NeatRacingEditorSession;

/** Element id given to the preview canvas mounted by the editor boot. */
const NEATRACING_EDITOR_CANVAS_ID = 'neatracing-editor-canvas';

/** Pass label prefixing every observable preview-lane failure surface. */
const NEATRACING_EDITOR_PREVIEW_LABEL = 'neatRacing editor preview';

/** Per-dimension floor for the boot-time preview canvas size in pixels. */
const MIN_PREVIEW_CANVAS_SIZE = 1;

/**
 * World extent of the editor working volume in meters: the 128-cubed
 * lattice at the fine voxel size (4.5 m, spec section 7.3.4 G1).
 */
const NEATRACING_EDITOR_WORKING_VOLUME_EXTENT_M =
  NEATRACING_EDITOR_VOLUME_EDGE * NEATRACING_VOXEL_WORLD_SIZE;

/**
 * Boot camera distance in world units: 1.5 working-volume extents, so the
 * 4.5 m model fills the preview viewport instead of the camera's
 * game-scale default of 192 m (review obs 5 — the default framed the model
 * at ~2 % of the viewport).
 */
const NEATRACING_EDITOR_BOOT_CAMERA_DISTANCE =
  NEATRACING_EDITOR_WORKING_VOLUME_EXTENT_M * 1.5;

/**
 * Boot camera focus height in world units: half the working volume's
 * extent, so the volume's vertical center sits at the viewport's center
 * and the viewport center picks the lattice's center cell.
 */
const NEATRACING_EDITOR_BOOT_CAMERA_CENTER_Y =
  NEATRACING_EDITOR_WORKING_VOLUME_EXTENT_M / 2;

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
 * @param deps - Optional boot dependencies (the model-store fetch seam).
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
function neatRacingEditorStart(
  previewHostId: string,
  deps?: NeatRacingEditorBootDeps,
): NeatRacingEditorSession {
  const previewHost = document.getElementById(previewHostId);
  if (!(previewHost instanceof HTMLElement)) {
    throw new Error(`Preview host #${previewHostId} not found`);
  }

  // Step 0: the stopped flag. The rebuild pipeline owns a 250 ms trailing
  // debounce with no dispose seam, so the boot cancels its pending flush by
  // refusing to publish once the session was stopped (review obs 3).
  let stopped = false;

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
  // draws. The preview provider reads the tool frame's LIVE loaded document
  // through the holder below, so a later dropdown-driven load reaches the
  // preview without re-editing the boot; the loop memoizes on document
  // identity, so a load only affects the NEXT frame.
  let previewHandle: EditorPreviewRenderLoopHandle | null = null;
  // Live holder for the tool frame: the preview lane starts before the tool
  // frame is mounted below (the boot order is unchanged), so the provider
  // reads the mounted frame through this holder instead of a one-shot
  // snapshot of the initial load result.
  let toolframeHolder: EditorToolframeSession | null = null;
  // Live holder for the EDITED working volume's projection: the section 7.2
  // model document the rebuild pipeline publishes after a full rebuild. It
  // stays null until the first pointer stroke flushes, and the preview lane
  // prefers it over the loaded document so the preview redraws the edited
  // volume (AC-021: "the Step 14 preview lane redrawing the edited volume").
  let editedDocument: VoxelModelDocument | null = null;
  const getSelectedModelDocument = (): VoxelModelDocument | null =>
    editedDocument ?? toolframeHolder?.currentDocument ?? null;
  // The editor camera is hoisted OUT of the WebGL2 gate on purpose: the
  // pointer-pick edit loop casts its ray through the SAME camera the
  // preview renders with, so the picker stays correct on hosts without a
  // live context (the picker lane never needs a GL context). The boot pose
  // frames the working volume (review obs 5): the camera's game-scale
  // default (192 m) framed the 4.5 m model at ~2 % of the viewport.
  const camera = createEditorCamera({
    center: [0, NEATRACING_EDITOR_BOOT_CAMERA_CENTER_Y, 0],
    distance: NEATRACING_EDITOR_BOOT_CAMERA_DISTANCE,
  });
  const previewGl = pickPreviewRenderingContext(rendering);
  if (previewGl !== null) {
    const binsDraw = createEditorPreviewBinsDraw(previewGl, canvas);
    previewHandle = startEditorPreviewRenderLoop({
      canvas,
      camera,
      getSelectedModelDocument,
      binsDraw,
    });
  }

  // Step 5: Mount the editor tool frame — the editor state container, the
  // control mounts in the reserved `neatracing-editor-tools` container, and
  // the initial model load (model-store list, read, and decode into the
  // working volume). The tool frame resolves its container or throws
  // (batch-94 handoff contract), so a shell that lost the container fails
  // loudly instead of rendering an editor without controls. The boot injects
  // the model-store fetch seam through the optional deps, so the load chain
  // can be exercised without a dev server; a boot without deps keeps the
  // ambient `globalThis.fetch`.
  //
  // The preview render loop is ALREADY RUNNING at this point, so a shell
  // that lost the reserved container would leak an unreapable RAF chain (the
  // throw leaves no session behind, so no stop() can ever reap it). The
  // mount is therefore guarded: the loop is stopped before the error leaves
  // the boot, and the original error still propagates (the shell keeps
  // failing loudly instead of rendering an editor without controls).
  let toolframe: EditorToolframeSession;
  try {
    toolframe = createEditorToolframe({ fetchImpl: deps?.fetch });
  } catch (error) {
    previewHandle?.stop();
    previewHandle = null;
    throw error;
  }
  toolframeHolder = toolframe;

  // Step 5b: Wire the control BEHAVIORS onto the tool frame's mounts — the
  // palette add/select handlers, the undo/redo buttons, the model dropdown,
  // and the CAD-view buttons (AC-020 completion). The controls own the
  // shared paint entry and the undo seam the pointer edit loop captures its
  // per-stroke snapshot through, so the edit loop paints with the selected
  // palette color and a new edit invalidates the redo history.
  const controls = createEditorControls({
    toolframe,
    camera,
    onWorkingVolumeChanged: () => {
      publishWorkingVolume();
    },
  });

  // Step 5c: Wire the LIVE mirror-X symmetry and the immediate live save
  // (AC-021 completion, LD-28). The tool frame's symmetry policy freezes
  // its axis flags at construction, so the paint path could never follow
  // a mid-session toggle flip; the live symmetry reads the toggle's
  // checked state at paint time instead. The live save PUTs the edited
  // working volume into the dev-server model store and triggers the
  // rebuild flight - no import/export UI (LD-28).
  const liveSymmetry = createEditorLiveSymmetry({
    toggle: toolframe.controls.symmetryToggle,
    base: toolframe.symmetry,
  });
  const liveSave = createEditorLiveSave({
    modelName: () => toolframe.state.snapshot().model,
    readSource: () => ({
      volume: toolframe.state.volume,
      palette: toolframe.palette,
    }),
    fetchImpl: deps?.saveFetch,
    mirrorXEnabled: () => toolframe.controls.symmetryToggle.checked,
  });

  /**
   * Publishes the working volume as a section 7.2 model document.
   *
   * The pointer picker inverts the 128-cubed working lattice, so the
   * preview must render THAT projection and not the loaded document's own
   * dims (review obs 1: a 32-cubed wheel document previewed at the working
   * volume's scale made every pick land on the wrong voxel). The document
   * is published at boot (after the initial model load landed), after every
   * model load, after every undo/redo, and after every full rebuild.
   */
  function publishWorkingVolume(): void {
    if (stopped) {
      return;
    }
    editedDocument = buildEditorVolumeDocument({
      volume: toolframe.state.volume,
      palette: toolframe.palette,
      name:
        toolframe.state.snapshot().model ??
        NEATRACING_EDITOR_DEFAULT_MODEL_NAME,
    });
  }

  // Publish the working volume once the initial model load landed, so the
  // preview renders the working lattice the picker inverts.
  void toolframe.ready.then(() => {
    publishWorkingVolume();
  });

  // Step 6: Mount the pointer-pick edit loop — the canvas pointer chain
  // (pick -> symmetry-canonicalized paint -> undo push -> rebuild pipeline
  // edit -> pointer-up full flush). The rebuild pipeline's executor
  // republishes the edited working volume as a section 7.2 model document
  // on every FULL rebuild; the dirty slice window only remeshes the edited
  // slices, so the whole-volume projection rides the full flush (the
  // pointer-up flush or the trailing debounce).
  const rebuildPipeline = createEditorRebuildPipeline({
    rebuild: async (request) => {
      if (request.mode !== 'full') {
        return;
      }
      publishWorkingVolume();
      // The live save rides the FULL rebuild flight: the dirty slice
      // window only remeshes the edited slices and must not storm the
      // model store, so one edit burst collapses into exactly one save.
      liveSave.enqueueSave();
    },
  });
  const editLoop = createEditorEditLoop({
    canvas,
    state: toolframe.state,
    symmetry: liveSymmetry,
    // The controls' undo seam captures the stroke snapshot AND invalidates
    // the redo history; the controls' paint entry is the shared mutable
    // tuple the selected palette color writes into.
    undo: controls.undo,
    pipeline: rebuildPipeline,
    paintEntry: controls.paintEntry,
    picker: createEditorVoxelPicker({
      volume: toolframe.state.volume,
      camera,
      // The picker reads the viewport per pick: the CSS box is the pointer
      // coordinate space, while the projection aspect follows the drawing
      // buffer (the preview lane's own convention).
      readViewport: () => ({
        cssWidth: canvas.clientWidth > 0 ? canvas.clientWidth : canvas.width,
        cssHeight:
          canvas.clientHeight > 0 ? canvas.clientHeight : canvas.height,
        aspect: canvas.width / Math.max(canvas.height, 1),
      }),
    }),
  });
  editLoop.attach();

  // Step 7: Hand the tool slices the tool frame plus live projections of its
  // editor state and its loaded document, so the session never carries a
  // stale model name or a stale document.
  return {
    state: createSessionStateView(toolframe),
    get currentDocument(): VoxelModelDocument | null {
      return toolframe.currentDocument;
    },
    get editedDocument(): VoxelModelDocument | null {
      return editedDocument;
    },
    editLoop,
    controls,
    liveSave,
    camera,
    toolframe,
    stop: () => {
      // The stopped flag cancels the rebuild pipeline's pending debounced
      // flush: the executor refuses to publish after the stop (review
      // obs 3 — the pipeline exposes no dispose seam).
      stopped = true;
      liveSave.stop();
      previewHandle?.stop();
      previewHandle = null;
      editLoop.detach();
      controls.stop();
      toolframe.stop();
      canvas.remove();
      fallbackElement?.remove();
    },
  };
}

/**
 * Builds the session's live projection of the tool frame's editor state.
 *
 * The projection reads the tool frame's state on every access, so the
 * session's model name follows the tool frame's selection (the initial
 * model load lands asynchronously) without the boot owning a second,
 * diverging copy of the selection.
 *
 * @param toolframe - Tool frame owning the editor state container.
 * @returns The session's editor state container.
 *
 * @example
 * ```ts
 * const view = createSessionStateView(toolframe);
 * console.log(view.model); // null until the initial model load lands
 * ```
 */
function createSessionStateView(
  toolframe: EditorToolframeSession,
): NeatRacingEditorState {
  return {
    get model(): string | null {
      return toolframe.state.snapshot().model;
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

declare global {
  interface Window {
    /** Neat Racing editor boot function installed by the editor bundle. */
    neatRacingEditorStart?: NeatRacingEditorStart;
  }
}

window.neatRacingEditorStart = neatRacingEditorStart;
