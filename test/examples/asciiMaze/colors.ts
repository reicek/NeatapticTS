/**
 * ANSI color codes for maze visualization in the terminal.
 * These codes are used to colorize maze elements for better readability.
 *
 * Usage example:
 *   console.log(colors.bgGreen + 'Agent' + colors.reset);
 */
export const colors = {
  reset: '\x1b[0m',         // Reset all attributes
  bright: '\x1b[1m',        // Bright/bold text
  dim: '\x1b[2m',           // Dim text
  red: '\x1b[31m',          // Red foreground
  green: '\x1b[32m',        // Green foreground
  yellow: '\x1b[33m',       // Yellow foreground
  blue: '\x1b[34m',         // Blue foreground
  cyan: '\x1b[36m',         // Cyan foreground
  bgGreen: '\x1b[42m',      // Green background (used for agent location)
  bgRed: '\x1b[41m',        // Red background
  bgYellow: '\x1b[43m',     // Yellow background
  bgBlue: '\x1b[44m',       // Blue background
  darkWallBg: '\x1b[48;5;236m',   // Dark gray background for walls
  darkWallText: '\x1b[38;5;234m', // Very dark text for wall symbols
  lightBrownBg: '\x1b[48;5;222m', // Light brown background for open paths
  lightBrownText: '\x1b[38;5;222m',// Light brown text for open paths
  bgWhite: '\x1b[48;5;15m',       // White background (used for start/exit)
  pureBlue: '\x1b[34;1m',         // Bright blue text
  pureRed: '\x1b[31;1m',          // Bright red text (used for exit)
  pureGreen: '\x1b[32;1m',        // Bright green text (used for start/agent)
};
