/**
 * ANSI color codes for maze visualization in the terminal.
 * These codes use a TRON-inspired color palette with vibrant cyberspace blues,
 * electric whites, and neon accents for a futuristic digital look.
 *
 * The palette features the iconic TRON colors that work together to create
 * the feeling of being inside a digital world or computer system.
 *
 * All color codes are provided as string escape sequences for use in terminal output.
 */
export const colors = {
  // Basic formatting
  reset: '\x1b[0m',         // Reset all attributes
  bright: '\x1b[1m',        // Bright/bold text
  dim: '\x1b[2m',           // Dim text

  // Neon foreground colors (expanded palette)
  neonPink: '\x1b[38;5;205m',         // Neon pink
  neonPurple: '\x1b[38;5;93m',        // Neon purple
  neonLime: '\x1b[38;5;154m',         // Neon lime green
  neonAqua: '\x1b[38;5;51m',          // Neon aqua
  neonYellow: '\x1b[38;5;226m',       // Neon yellow
  neonOrange: '\x1b[38;5;214m',       // Neon orange (brighter)
  neonRed: '\x1b[38;5;196m',          // Neon red
  neonGreen: '\x1b[38;5;46m',         // Neon green
  neonSky: '\x1b[38;5;123m',          // Neon sky blue
  neonViolet: '\x1b[38;5;177m',       // Neon violet
  neonTurquoise: '\x1b[38;5;80m',     // Neon turquoise
  neonMint: '\x1b[38;5;121m',         // Neon mint
  neonCoral: '\x1b[38;5;203m',        // Neon coral
  neonIndigo: '\x1b[38;5;99m',        // Neon indigo
  neonTeal: '\x1b[38;5;44m',          // Neon teal
  neonGold: '\x1b[38;5;220m',         // Neon gold
  neonSilver: '\x1b[38;5;250m',       // Neon silver
  neonBlue: '\x1b[38;5;45m',          // Neon blue (extra)
  neonMagenta: '\x1b[38;5;201m',      // Neon magenta (extra)
  neonCyan: '\x1b[38;5;87m',          // Neon cyan (extra)
  neonWhite: '\x1b[38;5;231m',        // Neon white (brightest)
  neonRose: '\x1b[38;5;218m',         // Neon rose
  neonPeach: '\x1b[38;5;217m',        // Neon peach
  neonAzure: '\x1b[38;5;117m',        // Neon azure
  neonChartreuse: '\x1b[38;5;118m',   // Neon chartreuse
  neonSpring: '\x1b[38;5;48m',        // Neon spring green
  neonAmber: '\x1b[38;5;214m',        // Neon amber (duplicate of orange, for clarity)
  neonFuchsia: '\x1b[38;5;207m',      // Neon fuchsia

  // TRON primary colors (foreground)
  blueCore: '\x1b[38;5;39m',      // Primary TRON blue
  cyanNeon: '\x1b[38;5;87m',      // Electric cyan
  blueNeon: '\x1b[38;5;45m',      // Bright neon blue
  whiteNeon: '\x1b[38;5;159m',    // Electric white-blue
  orangeNeon: '\x1b[38;5;208m',   // TRON orange (for contrast)
  magentaNeon: '\x1b[38;5;201m',  // Digital magenta

  // Base colors (foreground)
  red: '\x1b[38;5;197m',    // Program termination red
  green: '\x1b[38;5;118m',  // User/CLU green
  yellow: '\x1b[38;5;220m', // Warning yellow
  blue: '\x1b[38;5;33m',    // Deep blue
  cyan: '\x1b[38;5;51m',    // Light cyan

  // Neon background colors (expanded palette)
  bgNeonPink: '\x1b[48;5;205m',
  bgNeonPurple: '\x1b[48;5;93m',
  bgNeonLime: '\x1b[48;5;154m',
  bgNeonAqua: '\x1b[48;5;51m',
  bgNeonYellow: '\x1b[48;5;226m',
  bgNeonOrange: '\x1b[48;5;214m',
  bgNeonRed: '\x1b[48;5;196m',
  bgNeonGreen: '\x1b[48;5;46m',
  bgNeonSky: '\x1b[48;5;123m',
  bgNeonViolet: '\x1b[48;5;177m',
  bgNeonTurquoise: '\x1b[48;5;80m',
  bgNeonMint: '\x1b[48;5;121m',
  bgNeonCoral: '\x1b[48;5;203m',
  bgNeonIndigo: '\x1b[48;5;99m',
  bgNeonTeal: '\x1b[48;5;44m',
  bgNeonGold: '\x1b[48;5;220m',
  bgNeonSilver: '\x1b[48;5;250m',
  bgNeonBlue: '\x1b[48;5;45m',          // Neon blue background (extra)
  bgNeonMagenta: '\x1b[48;5;201m',      // Neon magenta background (extra)
  bgNeonCyan: '\x1b[48;5;87m',          // Neon cyan background (extra)
  bgNeonWhite: '\x1b[48;5;231m',        // Neon white background (brightest)
  bgNeonRose: '\x1b[48;5;218m',         // Neon rose background
  bgNeonPeach: '\x1b[48;5;217m',        // Neon peach background
  bgNeonAzure: '\x1b[48;5;117m',        // Neon azure background
  bgNeonChartreuse: '\x1b[48;5;118m',   // Neon chartreuse background
  bgNeonSpring: '\x1b[48;5;48m',        // Neon spring green background
  bgNeonAmber: '\x1b[48;5;214m',        // Neon amber background (duplicate of orange, for clarity)
  bgNeonFuchsia: '\x1b[48;5;207m',      // Neon fuchsia background

  // TRON background colors
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
