# visualizer

AI-mode visualizer surface for the NeatRacing demo (P4-S4, Step 26 —
RED-P4-10).

The module carries the FULLSCREEN-CAPABLE AI-mode network visualizer
surface (AC-103) over the shared network visualization: the surface owns
the AI-mode presentation state (visible by default at the docked shelf —
the CLAR-4-02 recorded default — and togglable between the docked shelf
and the fullscreen presentation) and the AI lane's live-carrier draw
seam (the elected winner rides the adapter's live-carrier frame law).

The module is DOM-FREE by contract: the factory resolves a plain state
record (no document, no window, no canvas), and the frame resolution is
a delegation onto the panel adapter's live-carrier law with the
host-owned canvas 2D context handed in per call. The DOM wiring (mount,
presentation slot application, the EXPLICIT HUD toggle control and
the resize authority) lives in the browser entry, which keeps ALL
sizing/presentation decisions flowing through pure resolvers (the L6
slot law: the docked-right shelf resolver in the panel adapter stays
untouched and remains the DOCKED presentation's ONE sizing authority;
the fullscreen slot below is the surface's OWN presentation law).

Educational note — THE ADAPTER SEAM (AC-034): the surface never
re-implements topology layout, activation capture or drawing. The
surface's `resolveLiveCarrierFrame` delegates to
`resolveNeatRacingNetworkPanelLiveCarrierFrame`, which reads the
carried live network (own slot first, candidate view as fallback),
fails LOUDLY on a network-less carrier (never the retired idle-path
stub — AC-102) and delegates the whole frame to the shared visualizer
path (the L5 thin law). Activations are captured at resolution time,
so consumers re-resolve per frame.

Example:

```ts
const surface = resolveNeatRacingAiModeVisualizerSurface({ mode: 'ai' });
// -> { mode: 'ai', visible: true, fullscreen: false, ... }
surface.toggle(); // -> fullscreen presentation state
const frame = surface.resolveLiveCarrierFrame(panelContext, winner);
drawNeatRacingNetworkPanelFromFrame(panelContext, frame);
const manualSurface = resolveNeatRacingAiModeVisualizerSurface('manual');
// -> null (manual mode keeps its own lane: UNCHANGED per AC-103)
```

## visualizer/ai-mode.visualizer.ts

### NeatRacingAiModeVisualizerMode

The AI-mode visualizer's presentation modes (AC-103): the 'ai' mode owns
the fullscreen-capable network surface; the 'manual' mode keeps its own
lane UNCHANGED (no visualizer surface is resolved for it).

### NeatRacingAiModeVisualizerModeInput

Object form of the surface mode input: the host may hand the factory
whichever mode field its own state carries.

### NeatRacingAiModeVisualizerSurface

The AI-mode visualizer surface record (AC-103): a plain, DOM-free
presentation-state holder that the browser entry wires onto the panel
canvas.

### NeatRacingAiModeVisualizerSurfaceInput

Acceptable surface factory inputs: the bare mode string, the object
form, or `undefined` (the recorded default resolves the AI-mode surface
— the AI lane is the mode the visualizer serves).

### resolveNeatRacingAiModeVisualizerFullscreenSlot

```ts
resolveNeatRacingAiModeVisualizerFullscreenSlot(
  viewportWidthPx: number,
  viewportHeightPx: number,
): NeatRacingNetworkPanelShelfSlot
```

Resolves the AI-mode surface's fullscreen presentation slot (AC-103).

The fullscreen presentation rides the FULL viewport: the surface's
fullscreen slot law is the full-viewport box (the hud-output layer maps
viewport px 1:1 onto its box), so the toggled panel fills the reserved
layer instead of the docked shelf. This is the surface's OWN presentation
law — the docked-right shelf resolver (P4-S2C 1a) stays untouched and
remains the DOCKED presentation's ONE sizing authority.

Determinism claim (reproducibility rank: deterministic on the same
runtime): the slot is a pure function of its two arguments — no window
reads, no host measure, no DOM — so the same arguments always resolve the
same slot regardless of the surrounding window state.

Parameters:
- `viewportWidthPx` - Viewport width in CSS px (must be finite
positive; the fullscreen box fills the viewport).
- `viewportHeightPx` - Viewport height in CSS px (must be finite
positive).

Returns: The full-viewport presentation slot to size and position the
panel with.

Example:

```ts
const slot = resolveNeatRacingAiModeVisualizerFullscreenSlot(1689, 1325);
// -> { leftPx: 0, topPx: 0, widthPx: 1689, heightPx: 1325 }
panelCanvas.style.left = `${slot.leftPx}px`;
panelCanvas.width = slot.widthPx;
```

### resolveNeatRacingAiModeVisualizerMode

```ts
resolveNeatRacingAiModeVisualizerMode(
  input: NeatRacingAiModeVisualizerSurfaceInput | undefined,
): NeatRacingAiModeVisualizerMode
```

Resolves the presentation mode off the factory input (the recorded
default is 'ai').

Parameters:
- `input` - Bare mode string, object form, or `undefined`.

Returns: The validated presentation mode.

### resolveNeatRacingAiModeVisualizerSurface

```ts
resolveNeatRacingAiModeVisualizerSurface(
  input: NeatRacingAiModeVisualizerSurfaceInput | undefined,
): NeatRacingAiModeVisualizerSurface | null
```

Resolves the fullscreen-capable AI-mode visualizer surface (AC-103,
RED-P4-10 — the factory the red capability probe discovers FIRST).

Laws the surface carries:

- VISIBLE by default (the CLAR-4-02 recorded default state): `visible`
  starts as `true` so the surface inherits the docked shelf presentation
  on mount — never fullscreen-by-default (the recorded rejection of the
  CLAR-4-02 scope swallow).
- TOGGLABLE: `toggle()` flips the `fullscreen` presentation state in
  place on the SAME record (no allocation, no DOM); the docked and
  fullscreen states are the only two presentation states.
- MANUAL UNCHANGED (AC-103): the 'manual' mode resolves NO surface —
  `null` — so the manual lane keeps its own behavior untouched.
- LIVE-CARRIER draw seam (AC-102 + AC-034): the surface's frame
  resolution delegates to the adapter's live-carrier law over the shared
  network visualization; a network-less carrier fails loudly (never the
  retired idle-path stub frame).

The surface is DOM-FREE: it records presentation state only; the browser
entry applies the state onto the panel canvas through pure slot
resolvers (the docked-right shelf resolver for the docked state,
`resolveNeatRacingAiModeVisualizerFullscreenSlot` for the fullscreen
state).

Parameters:
- `input` - Bare mode string, object form, or `undefined` (defaults
to the AI mode).

Returns: The AI-mode surface record, or `null` for the manual mode.

Example:

```ts
const surface = resolveNeatRacingAiModeVisualizerSurface('ai');
if (surface !== null) {
  surface.toggle(); // -> fullscreen presentation state on the same record
  surface.toggle(); // -> back to the docked shelf presentation
}
```

## visualizer/network-panel.adapter.ts

NeatRacing network panel adapter over the shared rich browser network visualizer (AC-034).

The NeatRacing browser page must not own its own network-frame math. Just like
the racing curriculum adapter, this module reuses the shared resolved frame,
padding, node sizing, and connection drawing path while only swapping the
NeatRacing-specific pieces: the most-forward-agent selection law, the NGE
controller output labels, and the NGE input/output sizes.

## Panel law (LD-32 — "most forward agent only", AC-034)

- **L1 — Progress is host-owned.** The packed sim frame (`neatracing-packed-v1`)
  carries kinematics only: no laps, no along-track distance, no networks. The
  adapter therefore NEVER derives race progress from world XY or track
  geometry; the host wiring passes an along-track progress metric per agent
  (for example `laps * trackLength + along-track distance`). The panel stays
  free of track-geometry knowledge by construction.
- **L2 — Sticky incumbent.** The current leader keeps the panel unless another
  candidate's progress is STRICTLY greater, so an equal-progress tick or a
  near-tie never flickers the panel ("the current winner at all times").
- **L3 — Non-finite progress never leads.** Candidates whose progress is
  `NaN` or infinite are invisible to the selection, mirroring the recycled
  finite-number law of the recycled goals machine (`ai/ai-mode.goals.ts`).
- **L4 — Deterministic tie-break.** Without a present incumbent, the maximum
  finite progress wins; an exact progress tie resolves to the LOWEST
  agent id, so repeated ticks resolve identically for the same inputs.
- **L5 — Thin over the shared visualizer.** Frames are resolved through
  `examples/shared/network-visualization` with the shared LOD defaults and
  the shared default input-label path; the NeatRacing observation channels
  stay owner-defined in `ai/observation.assembler.ts` and are not restated
  here.
- **L6 — One sizing authority per canvas.** The racing-parity panel wiring
  (P2-S4-C) composes `installNeatRacingNetworkResize` (the backing-store
  owner) with `resolveNeatRacingNetworkPanelVisualizationFrame` +
  `drawNeatRacingNetworkPanelFromFrame` on the same canvas. The
  all-in-one `drawNeatRacingNetworkPanel` is the standalone self-sizing
  path and must NOT be composed with the resize service on the same
  canvas, because its sync pass would silently override the resize
  service's backing store (review F2). Since P4-S2C (1a) the docked-right
  shelf slot RESOLVER is the ONE sizing authority behind both surfaces:
  the resize owner sizes the backing store through the resolved slot and
  the host mount applies the SAME resolved slot to the panel box (the
  zero-measure jsdom fallback stays the pinned windowless lane).

### buildNeatRacingNetworkPanelSettings

```ts
buildNeatRacingNetworkPanelSettings(): NetworkVisualizationSettings
```

Builds the NeatRacing settings bag that drives the shared visualizer path.

LOD stays force-enabled with the shared LOD defaults so dense evolved
controllers collapse into the shared abstract renderer. Input-label groups
are deliberately NOT overridden (L5): the panel renders the shared default
input-label path because the NeatRacing observation channels stay
owner-defined in `ai/observation.assembler.ts`.

Returns: NeatRacing settings bag for the shared visualizer.

### drawNeatRacingNetworkPanel

```ts
drawNeatRacingNetworkPanel(
  canvas: HTMLCanvasElement,
  network: default | undefined,
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationPositionedScene
```

Draw the NeatRacing network panel for one network using the shared
network visualizer.

Composition law (L6, review F2): this is the STANDALONE self-sizing entry
point — it owns the canvas backing store through its sync pass. The
racing-parity panel wiring must instead compose
`installNeatRacingNetworkResize` with
`resolveNeatRacingNetworkPanelVisualizationFrame` +
`drawNeatRacingNetworkPanelFromFrame`; do not pair this function with the
resize service on the same canvas.

Parameters:
- `canvas` - Canvas element to render onto.
- `network` - Runtime network of the most-forward agent, or `undefined`
while the panel waits for the first network (renders the idle panel).
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Positioned node snapshot reused by host-side hover hit testing.

Example:

```ts
const leader = resolveMostForwardNeatRacingAgent(candidates, incumbentId);
drawNeatRacingNetworkPanel(networkPanelCanvas, leader?.network);
```

### drawNeatRacingNetworkPanelFromFrame

```ts
drawNeatRacingNetworkPanelFromFrame(
  context: CanvasRenderingContext2D,
  resolvedFrame: NetworkVisualizationResolvedFrame,
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationPositionedScene
```

Draws a previously resolved NeatRacing network-panel frame.

The host uses this path for hover-only repaint work because it can reuse
the cached static frame and only vary interactive emphasis.

Parameters:
- `context` - Canvas 2D drawing context.
- `resolvedFrame` - Reusable frame cache.
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Positioned node snapshot reused by host-side hover hit testing.

### drawNeatRacingOutputNodeLabels

```ts
drawNeatRacingOutputNodeLabels(
  context: CanvasRenderingContext2D,
  outputNodes: readonly PositionedNetworkNodeLike[],
): void
```

Draws the NGE controller output tags next to the positioned output nodes.

Parameters:
- `context` - Canvas 2D drawing context.
- `outputNodes` - Output nodes sorted by ascending y.

### installNeatRacingNetworkResize

```ts
installNeatRacingNetworkResize(
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onResize: () => void,
): { uninstall: () => void; resize: () => void; redraw: () => void; }
```

Installs NeatRacing network canvas resize listeners.

Mirrors the `installRacingNetworkResize` pattern exactly on the wiring
side: on every window `resize` event the service sizes the canvas backing
store once per event, then invokes `onResize` so the caller can schedule a
network redraw (only when the store actually changed — a no-op resize
never repaints the idle draw path). On install the size is applied once
immediately. This service is the backing-store OWNER of the racing-parity
panel wiring (L6, review F2): pair it with
`resolveNeatRacingNetworkPanelVisualizationFrame` +
`drawNeatRacingNetworkPanelFromFrame`, never with the self-sizing
`drawNeatRacingNetworkPanel`.

Since P4-S2C (1a — RED-P4-21) a MEASURABLE host (positive `clientWidth`
AND `clientHeight`) rides the docked-right slot: the service resolves the
same pure `resolveNeatRacingNetworkPanelDockedRightShelfSlot` slot from
the WINDOW dimensions (ONE sizing authority with the panel box the host
mount applies), never from the raw host bounds that defined the recorded
full-viewport defect.

If the host element has no CSS layout yet (for example in jsdom tests), the
service falls back to `window.innerWidth * 0.25` by
`window.innerHeight * 0.35` so the canvas still has a positive backing size
(the pinned zero-measure lane, unchanged).
The returned `redraw` handle is intentionally a no-op; the host schedules
redraws through `onResize`.

Parameters:
- `networkCanvas` - Network visualization canvas to resize.
- `networkCanvasHost` - Host element providing the measurability gate
(its zero client extents route the pinned fallback lane).
- `onResize` - Callback invoked after the canvas backing size changes.

Returns: Uninstall, resize, and redraw handles.

Example:

```ts
const { uninstall, resize } = installNeatRacingNetworkResize(
  networkPanelCanvas,
  networkPanelHost,
  () => {
    const context = networkPanelCanvas.getContext('2d', {
      desynchronized: true,
    });
    if (!context) {
      return;
    }
    const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
      context,
      leaderNetwork,
    );
    drawNeatRacingNetworkPanelFromFrame(context, frame);
  },
);
resize();
window.addEventListener('beforeunload', uninstall);
```

### NEATRACING_NETWORK_OUTPUT_LABELS

Ordered NGE controller output labels.

Law: the first two channels are the pinned NGE control axes (index 0 =
THROTTLE, index 1 = STEER) and channels 2..8 stream the degenerate
single-car self-radio seam writes (`R0`..`R6`) on radio-augmented ladder
levels. Output indices past the table fall back to `OUT<index>` so a wider
evolved output layer still renders deterministically.

### NeatRacingNetworkPanelCandidate

One host-supplied fleet candidate for the most-forward network panel.

The host wiring owns both the identity and the race progress of every car;
the adapter only compares the numbers it is handed.

### NeatRacingNetworkPanelLiveCarrier

One live-carrier panel feed for the live evolved-network resolution
(AC-102, Step 26 P4-S4-live-feed).

Educational note — WHY THIS SHAPE: the most-forward panel watches the
LIVE evolved lane, so the host hands the resolver the watched carrier —
the carrier's own live evolved network (real node / connection /
activation data) wrapped with the elected candidate view it rode in on.
The carrier's OWN network is the authoritative live mind; the candidate
view is the fallback so the elected NeatRacingNetworkPanelCandidate can
ride into the resolver alone.

### NeatRacingNetworkPanelShelfSlot

One resolved docked-right shelf slot for the network panel.

### NetworkVisualizationPositionedScene

Reusable positioned-node snapshot returned by the network-view draw path.

The host reuses this exact layout snapshot for pointer hit testing so hover
logic can stay aligned with the scene that was actually rendered.

### NetworkVisualizationResolvedFrame

Reusable resolved frame cache for hover-only network redraws.

The host can keep this resolved frame between pointer-driven redraws so it
does not recompute topology, layout, legend inputs, or connection lookups
until the network payload or canvas size changes.

### readNeatRacingNetworkPanelCarrierNetwork

```ts
readNeatRacingNetworkPanelCarrierNetwork(
  carrier: unknown,
): object | undefined
```

Reads the live network slot off a live-carrier-shaped argument (any
object carrying a `network`, or a `candidate` view carrying one).

The carrier's OWN network wins; the candidate view is the fallback (the
elected NeatRacingNetworkPanelCandidate may ride into the resolver alone).
A non-object argument or a carrier that carries no network at either
slot resolves to undefined and the caller raises the named error.

Parameters:
- `carrier` - The carrier-shaped argument to read.

Returns: The carried live network object, or undefined.

### resolveHostContentBoxDimensionPx

```ts
resolveHostContentBoxDimensionPx(
  hostElement: HTMLElement | null,
  hostClientDimensionPx: number,
  startPaddingProperty: "paddingBottom" | "paddingLeft" | "paddingRight" | "paddingTop",
  endPaddingProperty: "paddingBottom" | "paddingLeft" | "paddingRight" | "paddingTop",
): number
```

Resolves the host content-box dimension without double counting padding.

Mirrors the racing adapter: the host element's computed padding is
subtracted from the measured client dimension so the backing store fits the
visible shelf.

Parameters:
- `hostElement` - Host content element, or `null`.
- `hostClientDimensionPx` - Fallback client dimension of the host.
- `startPaddingProperty` - Padding property on the axis start.
- `endPaddingProperty` - Padding property on the axis end.

Returns: Resolved content-box dimension in px.

### resolveMostForwardNeatRacingAgent

```ts
resolveMostForwardNeatRacingAgent(
  candidates: readonly NeatRacingNetworkPanelCandidate[],
  incumbentAgentId: number | undefined,
): NeatRacingNetworkPanelCandidate | undefined
```

Resolves the agent whose network the panel renders this tick.

Law (module header L1-L4): progress is host-owned, the incumbent stays
leader unless a challenger is STRICTLY more forward, non-finite progress
never leads, and the no-incumbent tie-break is lowest agent id.

Parameters:
- `candidates` - Host-supplied fleet candidates for this tick.
- `incumbentAgentId` - Agent id that currently owns the panel, if any.

Returns: The winning candidate, or `undefined` when no finite-progress
candidate exists (the host then parks the panel). Network presence never
affects the selection: when the elected winner carries no network the
panel renders its idle state and the host must NOT fall back to a
networked runner-up (review F3, AC-034).

Example:

```ts
const leader = resolveMostForwardNeatRacingAgent(
  fleet.map((car, agentId) => ({
    agentId,
    progress: car.raceProgressMeters,
    network: car.network,
  })),
  renderedAgentId,
);
```

### resolveMostForwardWithoutIncumbent

```ts
resolveMostForwardWithoutIncumbent(
  finiteCandidates: readonly NeatRacingNetworkPanelCandidate[],
  incumbent: NeatRacingNetworkPanelCandidate | undefined,
): NeatRacingNetworkPanelCandidate
```

Applies the L4 max-progress/lowest-id law against the L2 sticky rule.

Parameters:
- `finiteCandidates` - Candidates with finite progress only.
- `incumbent` - Incumbent candidate kept on ties, when supplied.

Returns: The candidate that owns the panel this tick.

### resolveNeatRacingHoverState

```ts
resolveNeatRacingHoverState(
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationHoverState | undefined
```

Resolves the host-supplied hover indices into the shared hover state.

Parameters:
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Shared hover state, or `undefined` when nothing is hovered.

### resolveNeatRacingNetworkPanelArchitectureLabel

```ts
resolveNeatRacingNetworkPanelArchitectureLabel(
  network: default,
): string
```

Resolves the compact architecture summary for the NeatRacing network legend.

Parameters:
- `network` - Runtime network being visualized.

Returns: Shared architecture label with NeatRacing IO counts; the input
size falls back to the default ladder level width (Tier A, 82 channels)
when the runtime network does not expose `input`.

### resolveNeatRacingNetworkPanelCanvasDimensions

```ts
resolveNeatRacingNetworkPanelCanvasDimensions(
  measuredWidthPx: number,
  measuredHeightPx: number,
): { widthPx: number; heightPx: number; }
```

Resolves responsive network-panel canvas dimensions from the host panel
shelf, mirroring the racing panel dimension law: the measured width is
clamped to a positive minimum, the height falls back to the width-driven
aspect value, and a positive measured panel height wins over the fallback.

Parameters:
- `measuredWidthPx` - Current measured canvas width from layout.
- `measuredHeightPx` - Current measured host-panel height from layout.

Returns: Width and height for the canvas backing store.

Example:

```ts
resolveNeatRacingNetworkPanelCanvasDimensions(0, 0);
// -> { widthPx: 320, heightPx: 240 }
```

### resolveNeatRacingNetworkPanelDockedRightShelfSlot

```ts
resolveNeatRacingNetworkPanelDockedRightShelfSlot(
  viewportWidthPx: number,
  viewportHeightPx: number,
): NeatRacingNetworkPanelShelfSlot
```

Resolves the deterministic docked-right network-panel shelf slot
(P4-S2C 1a — RED-P4-20).

The pure, DOM-free ONE sizing authority of the 1a panel law: the viewport
is mapped onto the slot the right-docked panel rides, so a full-viewport
panel box is unreachable by construction. The resolver owns four laws:

- right-anchored x (`leftPx + widthPx === viewportWidthPx`),
- strictly inside the viewport on both axes,
- strictly bounded extent (at least 1px and never a full-viewport side),
- the width never exceeds half the viewport width.

Determinism claim (reproducibility rank: deterministic on the same
runtime): the slot is a pure function of its two arguments — no window
reads, no host measure, no DOM — so the same arguments always resolve the
same slot regardless of the surrounding window state. The host mount and
the resize service compose that one result instead of inventing separate
geometry (ONE sizing authority, L6).

Parameters:
- `viewportWidthPx` - Viewport width in CSS px (the panel docks into
the full-viewport hud-output layer, so viewport px map 1:1 onto it).
- `viewportHeightPx` - Viewport height in CSS px.

Returns: The docked-right bounded slot to size and position the panel
with.

Example:

```ts
const slot = resolveNeatRacingNetworkPanelDockedRightShelfSlot(1689, 1325);
// -> { leftPx: 1183, topPx: 48, widthPx: 506, heightPx: 379 }
panelCanvas.style.left = `${slot.leftPx}px`;
panelCanvas.style.width = `${slot.widthPx}px`;
```

### resolveNeatRacingNetworkPanelLiveCarrierFrame

```ts
resolveNeatRacingNetworkPanelLiveCarrierFrame(
  context: CanvasRenderingContext2D,
  carrier: NeatRacingNetworkPanelCandidate | NeatRacingNetworkPanelLiveCarrier,
): NetworkVisualizationResolvedFrame
```

Resolves the shared visualizer frame for a LIVE evolved carrier (AC-102,
Step 26 P4-S4-live-feed — the RED-P4-9 live-feed seam).

Educational note — WHAT/WHY: the docked-right panel's idle-path stub
frame (the 82-channel/2-output placeholder) retires with the live feed —
with a live carrier present the most-forward panel renders the carrier's
OWN evolved network (real node / connection / activation data). This
resolver is the adapter's live-carrier seam: it re-shapes the CARRIER
argument into the existing live-network frame path and delegates the
whole frame to the shared visualizer (L5 — thin adapter), so the frame
topology, LOD behavior and activation reads stay the shared laws.

Educational note — ACTIVATIONS ARE CAPTURED AT RESOLVE TIME: the shared
builder normalizes the carrier network's runtime nodes into fresh
view-layer records (`activation: runtimeNode.activation ?? 0`), so the
resolved frame carries the activation values AS OF THIS RESOLVE — a
consumer that CACHES the frame (the documented
drawNeatRacingNetworkPanelFromFrame hover-repaint path — and the
fullscreen surface slice, the next frame consumer) MUST re-resolve to
observe a later activation change. The resolution path itself adds NO
per-call allocation beyond the delegated shared frame build (AC-102's
zero-per-frame-allocation law). An extra `activations` argument on the
carrier is deliberately ignored: the network's nodes are the source of
truth for the activations captured at resolve time.

Parameters:
- `context` - Canvas 2D drawing context of the network panel.
- `carrier` - The watched live carrier (its own network, or an
 *   elected  {@link NeatRacingNetworkPanelCandidate} riding in alone).

Returns: The shared visualizer frame for the carrier's live network, with
the network's activation values captured as of this resolve time
(re-resolve to observe later activation changes).

Example:

```ts
const carrier: NeatRacingNetworkPanelLiveCarrier = {
  agentId: 2,
  progress: 120,
  network: liveEvolvedNetwork,
};
const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
  panelContext,
  carrier,
);
drawNeatRacingNetworkPanelFromFrame(panelContext, frame);
```

### resolveNeatRacingNetworkPanelVisualizationFrame

```ts
resolveNeatRacingNetworkPanelVisualizationFrame(
  context: CanvasRenderingContext2D,
  network: default | undefined,
): NetworkVisualizationResolvedFrame
```

Resolves a reusable network-visualization frame for the NeatRacing panel.

Host-owned caching can reuse this frame between pointer-driven redraws so
the expensive topology/layout work is not repeated for every hover event.

Parameters:
- `context` - Canvas 2D drawing context.
- `network` - Network to visualize, or `undefined` for an idle panel.

Returns: Reusable resolved frame for subsequent draw passes.

Example:

```ts
const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
  context,
  leaderNetwork,
);
drawNeatRacingNetworkPanelFromFrame(context, frame, hoveredNodeIndices);
```

### resolveSortedNeatRacingOutputNodes

```ts
resolveSortedNeatRacingOutputNodes(
  positionedNodes: readonly PositionedNetworkNodeLike[],
): PositionedNetworkNodeLike[]
```

Sorts the positioned output nodes top-to-bottom so labels map onto the
`NEATRACING_NETWORK_OUTPUT_LABELS` table deterministically.

Parameters:
- `positionedNodes` - Positioned nodes of the current frame.

Returns: Output nodes sorted by ascending y.

### syncNeatRacingCanvasToPanel

```ts
syncNeatRacingCanvasToPanel(
  canvas: HTMLCanvasElement,
): void
```

Syncs the network panel canvas backing store to the host panel shelf.

Parameters:
- `canvas` - Canvas element to keep in sync with its host shelf.
