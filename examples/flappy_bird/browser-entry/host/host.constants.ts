/**
 * Shared presentation constants for browser host assembly.
 *
 * These values tune the spacing and responsiveness of the host HUD and side
 * panels without burying layout numbers inside DOM-building code.
 */

/**
 * Shared panel padding used by the host stats container.
 *
 * This controls the interior breathing room of the main stats panel.
 */
export const FLAPPY_HOST_PANEL_PADDING = '8px';

/**
 * Shared table host padding used by the stats value section.
 *
 * The table host gets slightly different padding so dense stat rows remain
 * readable without wasting horizontal space.
 */
export const FLAPPY_HOST_TABLE_HOST_PADDING = '6px 8px';

/**
 * Shared stats table font size.
 *
 * The table uses a compact monospace size so many HUD rows fit comfortably in
 * the host panel.
 */
export const FLAPPY_HOST_TABLE_FONT_SIZE = '12px';

/**
 * Shared panel max-height transition.
 *
 * This supports the subtle host-panel resize behavior during responsive layout
 * changes.
 */
export const FLAPPY_HOST_PANEL_TRANSITION = 'max-height 120ms ease-out';

/**
 * Shared stats split gap.
 *
 * This controls the gutter between the table column and the network panel.
 */
export const FLAPPY_HOST_STATS_SPLIT_GAP = '8px';
