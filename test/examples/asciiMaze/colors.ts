/**
 * ANSI color codes for maze visualization in the terminal.
 * These codes use a modern and balanced color palette for better visual appeal.
 *
 * The palette features complementary colors that work well together while
 * maintaining sufficient contrast for readability in terminal environments.
 *
 * Usage example:
 *   console.log(colors.bgTeal + 'Agent' + colors.reset);
 */
export const colors = {
  // Basic formatting
  reset: '\x1b[0m',         // Reset all attributes
  bright: '\x1b[1m',        // Bright/bold text
  dim: '\x1b[2m',           // Dim text
  
  // Modern primary colors
  teal: '\x1b[38;5;36m',    // Teal foreground (primary color)
  coral: '\x1b[38;5;209m',  // Coral foreground (complementary to teal)
  indigo: '\x1b[38;5;61m',  // Indigo foreground (accent)
  amber: '\x1b[38;5;214m',  // Amber/gold foreground (accent)
  
  // Base colors with modern hues
  red: '\x1b[38;5;203m',    // Softer red foreground
  green: '\x1b[38;5;78m',   // Mint green foreground
  yellow: '\x1b[38;5;221m', // Warm yellow foreground
  blue: '\x1b[38;5;75m',    // Sky blue foreground
  cyan: '\x1b[38;5;80m',    // Turquoise cyan foreground
  
  // Background colors
  bgTeal: '\x1b[48;5;36m',  // Teal background (for agent)
  bgCoral: '\x1b[48;5;209m',// Coral background (complementary highlight)
  bgIndigo: '\x1b[48;5;61m',// Indigo background 
  bgAmber: '\x1b[48;5;214m',// Amber background
  
  // Common backgrounds
  bgRed: '\x1b[48;5;203m',  // Softer red background
  bgGreen: '\x1b[48;5;78m', // Mint green background (for agent)
  bgYellow: '\x1b[48;5;221m',// Warm yellow background
  bgBlue: '\x1b[48;5;75m',  // Sky blue background
  
  // Maze-specific colors
  darkWallBg: '\x1b[48;5;237m',   // Slightly lighter dark gray for walls
  darkWallText: '\x1b[38;5;235m', // Almost black text for wall symbols
  lightBrownBg: '\x1b[48;5;223m', // Slightly warmer light brown for paths
  lightBrownText: '\x1b[38;5;223m',// Matching text color for paths
  
  // Special highlights
  bgWhite: '\x1b[48;5;255m',       // Bright white background (for start/exit)
  pureBlue: '\x1b[38;5;33;1m',     // Vivid blue text
  pureRed: '\x1b[38;5;197;1m',     // Vivid magenta-red text (for exit)
  pureGreen: '\x1b[38;5;41;1m',    // Vivid green text (for start/agent)
};
