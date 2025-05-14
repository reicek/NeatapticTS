/**
 * ANSI color codes for maze visualization in the terminal.
 * These codes use a TRON-inspired color palette with vibrant cyberspace blues,
 * electric whites, and neon accents for a futuristic digital look.
 *
 * The palette features the iconic TRON colors that work together to create
 * the feeling of being inside a digital world or computer system.
 *
 * Usage example:
 *   console.log(colors.bgCyanNeon + 'Agent' + colors.reset);
 */
export const colors = {
  // Basic formatting
  reset: '\x1b[0m',         // Reset all attributes
  bright: '\x1b[1m',        // Bright/bold text
  dim: '\x1b[2m',           // Dim text
  
  // TRON primary colors
  blueCore: '\x1b[38;5;39m',      // Primary TRON blue
  cyanNeon: '\x1b[38;5;87m',      // Electric cyan
  blueNeon: '\x1b[38;5;45m',      // Bright neon blue
  whiteNeon: '\x1b[38;5;159m',    // Electric white-blue
  orangeNeon: '\x1b[38;5;208m',   // TRON orange (for contrast)
  magentaNeon: '\x1b[38;5;201m',  // Digital magenta
  
  // Base colors with TRON hues
  red: '\x1b[38;5;197m',    // Program termination red
  green: '\x1b[38;5;118m',  // User/CLU green
  yellow: '\x1b[38;5;220m', // Warning yellow
  blue: '\x1b[38;5;33m',    // Deep blue
  cyan: '\x1b[38;5;51m',    // Light cyan
  
  // Background colors
  bgBlueCore: '\x1b[48;5;39m',    // Primary TRON blue background
  bgCyanNeon: '\x1b[48;5;87m',    // Electric cyan background (for agent)
  bgBlueNeon: '\x1b[48;5;45m',    // Bright neon blue background
  bgWhiteNeon: '\x1b[48;5;159m',  // Electric white-blue background
  bgOrangeNeon: '\x1b[48;5;208m', // TRON orange background
  bgMagentaNeon: '\x1b[48;5;201m',// Digital magenta background
  
  // Common backgrounds
  bgRed: '\x1b[48;5;197m',    // Program termination red background
  bgGreen: '\x1b[48;5;118m',  // User/CLU green background
  bgYellow: '\x1b[48;5;220m', // Warning yellow background
  bgBlue: '\x1b[48;5;33m',    // Deep blue background
  
  // Maze-specific colors
  darkWallBg: '\x1b[48;5;17m',    // Dark blue for walls
  darkWallText: '\x1b[38;5;17m',  // Dark blue text for wall symbols
  floorBg: '\x1b[48;5;234m',      // Almost black for empty floor
  floorText: '\x1b[38;5;234m',    // Almost black text for floor symbols
  gridLineBg: '\x1b[48;5;23m',    // Subtle grid line color
  gridLineText: '\x1b[38;5;23m',  // Subtle grid line text
  
  // Special highlights
  bgBlack: '\x1b[48;5;16m',       // Pure black background
  pureBlue: '\x1b[38;5;57;1m',    // Vibrant system blue
  pureOrange: '\x1b[38;5;214;1m', // Vibrant TRON orange (for CLU/villains)
  pureGreen: '\x1b[38;5;46;1m',   // Pure green for user programs
};
