/**
 * The NeatRacing host NEON helper — the ONE source of truth for the
 * neatRacing neon treatment.
 *
 * WHAT: the ONE helper that owns the neatRacing neon — the neon colour
 * tokens, the neon CLASS hook name and the DOM `text-shadow` builder.
 * Every closed-select consumer reads the treatment from HERE, so no
 * consumer inlines a neon literal and no second copy of the neon tokens
 * can drift away from the first.
 *
 * WHY a helper module at all: the native `<option>` popup is
 * OS-rendered and IGNORES `text-shadow` and `filter`, so the neon can
 * only land on the CLOSED `<select>` — the control the user reads while
 * the HUD is idle. Two closed selects share that treatment (the
 * race-mode select and the track select), so the treatment is written
 * ONCE here instead of twice in the two selector modules.
 *
 * THE TWO-PASS RECIPE (the house neon recipe, mirrored from
 * `examples/neatRacing/host/hud.ts`): the CRISP pass paints the solid
 * core colour, the GLOW pass paints the bloom BEHIND it, so the text
 * reads as a light source instead of flat coloured type. The canvas
 * recipe it mirrors is the Flappy two-pass neon pipe
 * (`examples/flappy_bird/browser-entry/runtime/runtime.startup-preview.service.ts`),
 * translated from the canvas API to CSS. The countdown overlay already
 * uses the same DOM translation (`./hud.countdown.ts`); its OWN tokens
 * STAY there — this helper neither refactors nor imports them, so this
 * module remains a LEAF and the HUD could adopt the neon later without
 * an import cycle.
 *
 * THE NEON HUE: the neon cyan is the CANONICAL neon cyan (`#00ffff`),
 * one step brighter than the HUD's `rgb(0, 240, 255)` accent, so the
 * bloom reads as a light source rather than as the flat accent colour
 * and so this module owns a hue of its own instead of holding a
 * value-identical copy of the HUD token.
 *
 * THE TREATMENT BOUNDARY — the neon lands on the closed control, and the
 * OS-drawn popup stays outside the treatment:
 *
 * ```mermaid
 * flowchart LR
 *     Tokens["neon tokens<br/>hue, alpha, blur"] --> Shadow["buildNeonTextShadow()"]
 *     Tokens --> Core["crisp core colour"]
 *     Shadow --> Apply["applyNeonTreatment(select)"]
 *     Core --> Apply
 *     Apply --> Hook["class hook<br/>neatracing-hud-neon"]
 *     Apply --> Closed["CLOSED select text<br/>glowing core"]
 *     Closed -->|"user opens the list"| Popup["native option popup<br/>OS-rendered, never glowed"]
 * ```
 *
 * DETERMINISM: this module is pure — it creates NO DOM of its own, it
 * never reads a wall clock, it never consults a random source and it
 * allocates nothing per frame. The same element always receives the
 * same treatment.
 *
 * Further reading: the DOM glow pass translates the canvas recipe's
 * `shadowBlur` bloom into CSS
 * [`text-shadow`](https://developer.mozilla.org/en-US/docs/Web/CSS/text-shadow),
 * and the lit-tube look it imitates is the one described in
 * [Neon sign](https://en.wikipedia.org/wiki/Neon_sign)
 * (Wikipedia contributors).
 */

/**
 * The GLOW PASS colour channels (the comma-separated RGB argument list
 * that `rgba()` expects).
 *
 * WHY the channels and not a finished colour: the CRISP pass and the
 * GLOW pass share this hue, and only the bloom needs an alpha, so the
 * channels are the single place the hue lives.
 *
 * @example
 * ```ts
 * `rgba(${NEATRACING_HUD_NEON_GLOW_RGB}, 0.75)`; // 'rgba(0, 255, 255, 0.75)'
 * ```
 */
export const NEATRACING_HUD_NEON_GLOW_RGB: string = '0, 255, 255';

/**
 * The CRISP PASS colour (the solid core colour painted on top of the
 * bloom).
 *
 * @example
 * ```ts
 * NEATRACING_HUD_NEON_CORE_COLOR; // 'rgb(0, 255, 255)'
 * ```
 */
export const NEATRACING_HUD_NEON_CORE_COLOR: string =
  `rgb(${NEATRACING_HUD_NEON_GLOW_RGB})`;

/**
 * The GLOW PASS alpha (the bloom strength).
 *
 * WHY 0.75: the bloom must be strong enough to read as a neon tube at
 * the HUD's 14px type size, but soft enough that the crisp core colour
 * stays legible on the dark overlay.
 *
 * @example
 * ```ts
 * NEATRACING_HUD_NEON_GLOW_ALPHA; // 0.75 — the bloom's opacity
 * ```
 */
export const NEATRACING_HUD_NEON_GLOW_ALPHA: number = 0.75;

/**
 * The GLOW PASS blur radius in CSS pixels (the bloom radius).
 *
 * WHY 10px: the bloom has to outgrow the 14px type of a closed select
 * without bleeding into the neighbouring HUD readouts.
 *
 * @example
 * ```ts
 * NEATRACING_HUD_NEON_GLOW_BLUR_PX; // 10 — the bloom radius in CSS pixels
 * ```
 */
export const NEATRACING_HUD_NEON_GLOW_BLUR_PX: number = 10;

/**
 * The neon CLASS hook name (the CSS hook a stylesheet, a test or a
 * browser probe targets to recognise a neon surface).
 *
 * WHY a class and not an id: the hook marks the SURFACE that carries
 * the neon treatment, and both closed selects must carry the SAME hook
 * so the HUD never shows a half-neon surface.
 *
 * @example
 * ```ts
 * const select: HTMLSelectElement = document.createElement('select');
 * select.classList.add(NEATRACING_HUD_NEON_CLASS_HOOK);
 * select.classList.contains(NEATRACING_HUD_NEON_CLASS_HOOK); // true
 * ```
 */
export const NEATRACING_HUD_NEON_CLASS_HOOK: string = 'neatracing-hud-neon';

/**
 * What: builds the DOM `text-shadow` that paints the GLOW PASS behind
 * the crisp core colour.
 *
 * Why: the shadow value is derived from the neon tokens, so the closed
 * selects share ONE glow and the two of them can never drift. The
 * shape (`0 0 <blur> rgba(<channels>, <alpha>)`) is the DOM translation
 * of the canvas bloom (`shadowBlur` plus `shadowColor`), the same
 * translation the countdown overlay uses.
 *
 * When: called once per neon surface, at mount time, by
 * `applyNeonTreatment` — never per frame.
 *
 * @returns The CSS `text-shadow` value for the GLOW PASS.
 *
 * @example
 * ```ts
 * buildNeonTextShadow(); // '0 0 10px rgba(0, 255, 255, 0.75)'
 * ```
 */
export function buildNeonTextShadow(): string {
  return (
    `0 0 ${String(NEATRACING_HUD_NEON_GLOW_BLUR_PX)}px ` +
    `rgba(${NEATRACING_HUD_NEON_GLOW_RGB}, ${String(NEATRACING_HUD_NEON_GLOW_ALPHA)})`
  );
}

/**
 * What: applies the whole neon treatment to an element the CALLER
 * owns — the neon CLASS hook plus the two-pass text style (the CRISP
 * core colour first, then the GLOW bloom behind it).
 *
 * Why: the two passes must always be applied together and in the same
 * order on every closed select, so the recipe lives here instead of
 * being spelled out by each consumer. The element itself stays the
 * consumer's: this helper creates NO DOM, queries NO DOM and owns NO
 * DOM of its own.
 *
 * When: called once per closed select, at mount time.
 *
 * @param element - The closed control that carries the neon treatment.
 *
 * @returns Nothing — the treatment is applied to the given element.
 *
 * @example
 * ```ts
 * const select: HTMLSelectElement = document.createElement('select');
 * applyNeonTreatment(select);
 * select.classList.contains(NEATRACING_HUD_NEON_CLASS_HOOK); // true
 * ```
 */
export function applyNeonTreatment(element: HTMLElement): void {
  element.classList.add(NEATRACING_HUD_NEON_CLASS_HOOK);
  // The CRISP PASS (the solid core colour) over the GLOW PASS (the
  // bloom behind it): the house two-pass recipe, in the same order as
  // the countdown overlay (`./hud.ts`).
  element.style.color = NEATRACING_HUD_NEON_CORE_COLOR;
  element.style.textShadow = buildNeonTextShadow();
}
