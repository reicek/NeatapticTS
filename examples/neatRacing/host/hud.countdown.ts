/**
 * The Flappy-style 3-2-1-GO countdown resolver.
 *
 * WHAT: ONE pure resolver over the RACE clock that publishes the
 * countdown phase. The resolver is the SINGLE law behind BOTH countdown
 * consumers: the HUD overlay (`host/hud.ts`) paints the phase's token and
 * the host per-frame sample site (`browser-entry/browser-entry.ts`) holds
 * the sampled throttle until the phase reaches GO. Two consumers, one
 * law: the gate never releases before the overlay shows GO. The overlay MAY
 * paint GO up to ONE FRAME before the gate releases, because the frame's
 * throttle must ride BEFORE that frame's reply exists - so the gate reads
 * the PREVIOUS reply's published clock while the overlay paints the
 * CURRENT one.
 *
 * WHY the RACE clock and never the wall clock: the race clock
 * (`raceState.elapsedS`) is the deterministic clock the physics publishes,
 * so the countdown can never JUMP or DRIFT against the wall clock (a
 * clock-skew or an NTP step cannot move a phase), a replayed race
 * repaints the same countdown, and the HUD and the gate read the SAME
 * clock - the gate one reply behind the overlay, because the frame's
 * throttle must ride BEFORE that frame's reply exists. `Date.now()` is
 * NEVER read in this module.
 *
 * WHY the beat is short: the countdown is a START signal, not a modal.
 * The whole 3-2-1 sequence must fit inside the short window the demo's
 * driven probes give the car, so the field still rolls - and still
 * reaches the drift model's MIN-SPEED gate - right after GO. The beat is
 * a named constant so the phase law stays readable and adjustable in ONE
 * place.
 *
 * The phase table is the whole law, and it is a TABLE, not a branch tree:
 *
 * ```mermaid
 * flowchart LR
 *     T0["race clock 0.00 s"] --> P3["three<br/>token '3'<br/>throttle held"]
 *     P3 -->|"0.30 s"| P2["two<br/>token '2'<br/>throttle held"]
 *     P2 -->|"0.60 s"| P1["one<br/>token '1'<br/>throttle held"]
 *     P1 -->|"0.90 s"| PG["go<br/>token 'GO'<br/>throttle released"]
 *     PG -->|"2.10 s"| PD["done<br/>empty token<br/>overlay cleared"]
 * ```
 *
 * The one-reply seam between the two consumers:
 *
 * ```mermaid
 * flowchart LR
 *     Reply["reply n-1<br/>(published race clock)"] --> Gate["host gate<br/>zeroes the throttle"]
 *     Gate --> Request["frame request n<br/>(held throttle)"]
 *     Request --> ReplyN["reply n<br/>(new published race clock)"]
 *     ReplyN --> Hud["HUD overlay<br/>paints the phase token"]
 * ```
 *
 * THE NEON RECIPE (the two-pass Flappy recipe lifted from
 * `examples/flappy_bird/browser-entry/runtime/runtime.startup-preview.service.ts`):
 * the GLOW pass (the additive bloom: the cyan shadow colour at the glow
 * alpha and the glow blur) and the CRISP pass (the solid core colour on
 * top of the bloom). The Flappy palette is EXAMPLE-LOCAL, and the
 * countdown overlay is its OWN lane: it carries its OWN tokens — the
 * canvas recipe's values, plus the DOM translation of the glow pass (the
 * CSS `text-shadow` the DOM renderer paints). The shared closed-select
 * helper (`./hud.neon`) serves a DIFFERENT lane (the two overlay
 * `<select>` controls), so the two token families stay deliberately
 * separate and neither surface borrows the other's palette.
 *
 * Further reading: the DOM glow pass translates the canvas recipe's
 * `shadowBlur` bloom into CSS
 * [`text-shadow`](https://developer.mozilla.org/en-US/docs/Web/CSS/text-shadow),
 * and the canvas primitives it mirrors are
 * [`shadowBlur`](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/shadowBlur).
 */

/**
 * The countdown phase the race clock publishes.
 *
 * The phase names are the STABLE vocabulary of the countdown law: the HUD
 * paints the phase's token and the host gate reads the phase's hold flag.
 * The names carry the phase SEMANTICS, never their rendered text (the
 * token lives in `NEATRACING_HUD_COUNTDOWN_PHASE_INFO`).
 */
export type NeatRacingHudCountdownPhase = 'three' | 'two' | 'one' | 'go' | 'done';

/**
 * The countdown beat: the width of ONE phase in RACE-clock seconds.
 *
 * The beat is the countdown's contract with the demo's driven probes: the
 * '3', '2' and '1' phases span three beats, so the pre-GO hold lasts
 * `3 * NEATRACING_HUD_COUNTDOWN_BEAT_S` (0.9 s of race clock) and the car
 * is released with a full probe window still ahead of it.
 *
 * WHY 0.3 s: the beat must be WIDER than a quarter second, so a probe
 * that samples the countdown on a quarter-second grid visits every phase
 * (a narrower beat can skip a phase between two samples), and it must be
 * NARROW enough that the whole sequence stays inside the demo's driven
 * window (a one-second beat would spend the probe's whole window on the
 * countdown and the car would never reach the drift model's MIN-SPEED
 * gate, so the published slip and traction severity would stay zero).
 */
export const NEATRACING_HUD_COUNTDOWN_BEAT_S: number = 0.3;

/**
 * The number of beats the GO phase holds before the countdown clears.
 *
 * GO stays on screen for four beats (1.2 s of race clock) so the start
 * signal is readable at a glance; after that the phase is 'done' and the
 * overlay clears the token.
 */
export const NEATRACING_HUD_COUNTDOWN_GO_BEATS: number = 4;

/**
 * The GLOW PASS colour channels (the recipe's cyan bloom colour as the
 * comma-separated RGB argument list that `rgba()` expects).
 *
 * WHY the channels only: the Flappy recipe's bloom colour is
 * `rgba(95, 255, 255, 0.95)`, but the DOM glow pass composes its OWN
 * alpha (`NEATRACING_HUD_COUNTDOWN_GLOW_ALPHA`, 0.68) into the same
 * colour, so this constant carries the three channels and the rendered
 * shadow keeps the recipe's bloom strength.
 */
export const NEATRACING_HUD_COUNTDOWN_GLOW_RGB: string = '95, 255, 255';

/**
 * The GLOW PASS alpha (the recipe's bloom strength).
 *
 * The DOM glow pass composites this alpha with the `GLOW_RGB` channels,
 * so the rendered shadow keeps the recipe's bloom strength instead of
 * the recipe's own 0.95 alpha.
 */
export const NEATRACING_HUD_COUNTDOWN_GLOW_ALPHA: number = 0.68;

/**
 * The GLOW PASS blur radius in CSS pixels (the recipe's bloom radius).
 */
export const NEATRACING_HUD_COUNTDOWN_GLOW_BLUR_PX: number = 32;

/**
 * The CRISP PASS colour (the recipe's solid core colour, painted on top
 * of the bloom).
 */
export const NEATRACING_HUD_COUNTDOWN_CORE_COLOR: string = '#00ff66';

/**
 * The DOM translation of the GLOW PASS: the CSS `text-shadow` the DOM
 * renderer paints behind the crisp core colour.
 */
export const NEATRACING_HUD_COUNTDOWN_GLOW_SHADOW: string =
  `0 0 ${String(NEATRACING_HUD_COUNTDOWN_GLOW_BLUR_PX)}px ` +
  `rgba(${NEATRACING_HUD_COUNTDOWN_GLOW_RGB}, ${String(NEATRACING_HUD_COUNTDOWN_GLOW_ALPHA)})`;

/**
 * What each countdown phase renders and does.
 *
 * `label` is the token the HUD paints ('3', '2', '1' and 'GO'); the 'done'
 * phase carries the EMPTY token, which clears the overlay. `holdsThrottle`
 * is the gate law: while it is true the host sample site must publish a
 * zero throttle, so the car waits for the countdown instead of driving off
 * the grid before GO.
 */
export interface NeatRacingHudCountdownPhaseInfo {
  /** The token the HUD paints for this phase ('' clears the overlay). */
  readonly label: string;
  /** True while the sampled throttle must be held at zero. */
  readonly holdsThrottle: boolean;
}

/**
 * The countdown phase table (the ONE law the HUD and the gate share).
 *
 * The table is a module-level constant, so reading a phase allocates
 * nothing: the render loop reads the token and the gate reads the hold
 * flag without building a per-frame object.
 */
export const NEATRACING_HUD_COUNTDOWN_PHASE_INFO: Readonly<
  Record<NeatRacingHudCountdownPhase, NeatRacingHudCountdownPhaseInfo>
> = {
  three: { label: '3', holdsThrottle: true },
  two: { label: '2', holdsThrottle: true },
  one: { label: '1', holdsThrottle: true },
  go: { label: 'GO', holdsThrottle: false },
  done: { label: '', holdsThrottle: false },
};

/**
 * What: resolves the countdown phase for a race-clock reading.
 *
 * Why: the HUD and the host gate must apply the SAME phase law to the
 * SAME race clock, so the law lives here instead of being spelled twice
 * (the overlay once, the gate once). The two consumers read CONSECUTIVE
 * readings of that one clock, never the same reading - the gate reads
 * the PREVIOUS reply's published clock (the frame's throttle must ride
 * the request BEFORE that frame's reply exists) and the overlay paints
 * the CURRENT one - so the gate never releases before the overlay shows
 * GO, and the overlay may paint GO up to ONE FRAME before the gate
 * releases.
 *
 * When: called by the HUD overlay once per frame (to paint the token) and
 * by the host per-frame sample site once per frame (to read the hold
 * flag) — both from the SAME race clock (the gate one reply behind the
 * overlay, by the reply-ordering seam).
 *
 * The phase windows (the beat is `NEATRACING_HUD_COUNTDOWN_BEAT_S`):
 * 'three' [0, beat), 'two' [beat, 2*beat), 'one' [2*beat, 3*beat),
 * 'go' [3*beat, (3 + GO_BEATS) * beat) and 'done' from then on. A
 * negative reading is clamped to the start of the countdown, so a clock
 * that has not started yet still shows '3'.
 *
 * @param elapsedS - The RACE clock reading in seconds (`raceState.elapsedS`).
 * @returns The countdown phase the reading belongs to.
 *
 * @example
 * ```ts
 * resolveHudCountdownPhase(0); // 'three'
 * resolveHudCountdownPhase(0.4); // 'two'
 * resolveHudCountdownPhase(1.1); // 'go'
 * resolveHudCountdownPhase(9); // 'done'
 * ```
 */
export function resolveHudCountdownPhase(elapsedS: number): NeatRacingHudCountdownPhase {
  const beatIndex = Math.floor(Math.max(elapsedS, 0) / NEATRACING_HUD_COUNTDOWN_BEAT_S);
  if (beatIndex < 1) {
    return 'three';
  }
  if (beatIndex < 2) {
    return 'two';
  }
  if (beatIndex < 3) {
    return 'one';
  }
  if (beatIndex < 3 + NEATRACING_HUD_COUNTDOWN_GO_BEATS) {
    return 'go';
  }
  return 'done';
}
