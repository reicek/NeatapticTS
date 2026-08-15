/**
 * DOM event name constants for the Neatenstein host layer.
 *
 * Centralises every hard-coded DOM event string used by input bindings, the
 * input router, and the human-mode selector so there is a single authoritative
 * source for event names across the host module surface.
 *
 * @module
 */

/** Keyboard key-down event name. */
export const DOM_EVENT_KEYDOWN = 'keydown';

/** Keyboard key-up event name. */
export const DOM_EVENT_KEYUP = 'keyup';

/** Mouse button-down event name. */
export const DOM_EVENT_MOUSEDOWN = 'mousedown';

/** Mouse pointer-move event name. */
export const DOM_EVENT_MOUSEMOVE = 'mousemove';

/** Mouse click event name. */
export const DOM_EVENT_CLICK = 'click';

/** Touch-start event name. */
export const DOM_EVENT_TOUCHSTART = 'touchstart';

/** Touch-move event name. */
export const DOM_EVENT_TOUCHMOVE = 'touchmove';

/** Touch-end event name. */
export const DOM_EVENT_TOUCHEND = 'touchend';

/** Touch-cancel event name. */
export const DOM_EVENT_TOUCHCANCEL = 'touchcancel';

/** Window blur event name. */
export const DOM_EVENT_BLUR = 'blur';

/** Document visibility-change event name. */
export const DOM_EVENT_VISIBILITYCHANGE = 'visibilitychange';

/** HTML element change event name (used by `<select>`). */
export const DOM_EVENT_CHANGE = 'change';

/** Window resize event name. */
export const DOM_EVENT_RESIZE = 'resize';