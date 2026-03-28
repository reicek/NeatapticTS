/** Number of trace microseconds contained in one millisecond. */
export const MICROSECONDS_PER_MILLISECOND = 1_000;

/** Default item count used by the top-N rollups. */
export const DEFAULT_TOP_COUNT = 12;

/** RunTask duration threshold used for frame-budget pressure reporting. */
export const LONG_TASK_THRESHOLD_MS = 16.7;

/** Severe RunTask duration threshold used for very long task reporting. */
export const VERY_LONG_TASK_THRESHOLD_MS = 50;

/** Trace event name used by Chrome when one frame is dropped. */
export const DROPPED_FRAME_EVENT_NAME = 'DroppedFrame';

/** Trace event name used by Chrome for frame production. */
export const BEGIN_FRAME_EVENT_NAME = 'BeginFrame';

/** Trace event name used for browser task execution slices. */
export const RUN_TASK_EVENT_NAME = 'RunTask';

/** Trace event name used for script execution attribution slices. */
export const FUNCTION_CALL_EVENT_NAME = 'FunctionCall';

/** Trace event name used for animation-frame callback execution. */
export const FIRE_ANIMATION_FRAME_EVENT_NAME = 'FireAnimationFrame';

/** Fallback process label when metadata is missing. */
export const UNKNOWN_PROCESS_LABEL = 'unknown-process';

/** Fallback thread label when metadata is missing. */
export const UNKNOWN_THREAD_LABEL = 'unknown-thread';

/** Fallback function-call label when a script URL is unavailable. */
export const UNKNOWN_SCRIPT_LABEL = '(unknown-script)';
