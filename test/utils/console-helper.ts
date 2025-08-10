/**
 * Helper utility to allow console output in specific test files.
 *
 * Usage:
 * 1. Import at the top of your test file: import { allowConsoleOutput } from '../utils/console-helper';
 * 2. Call at the beginning of your test: allowConsoleOutput();
 *
 * Run tests with: JEST_SHOW_CONSOLE_FOR=asciiMaze npm test
 * Or to show all console output: JEST_VERBOSE=1 npm test
 */

/**
 * Original console methods that we'll restore if needed
 */
const originalConsole = {
  log: console.log,
  info: console.info,
  warn: console.warn,
  error: console.error,
  debug: console.debug,
};

/**
 * Mock implementations that will suppress output
 */
const mockConsole = {
  log: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  debug: jest.fn(),
};

/**
 * Check if console output should be allowed for the current file
 */
export function shouldShowConsole(): boolean {
  if (process.env.JEST_VERBOSE === '1') return true;

  const filesToShow = (global as any).__SHOW_CONSOLE_FOR__;
  if (!filesToShow) return false;

  const currentFile = expect.getState().testPath || '';
  return filesToShow
    .split(',')
    .some((filePattern: string) =>
      currentFile.toLowerCase().includes(filePattern.toLowerCase())
    );
}

/**
 * Allow console output for the current test file
 */
export function allowConsoleOutput(): boolean {
  if (shouldShowConsole()) {
    // Restore original console methods
    console.log = originalConsole.log;
    console.info = originalConsole.info;
    console.warn = originalConsole.warn;
    console.error = originalConsole.error;
    console.debug = originalConsole.debug;
    return true;
  }
  return false;
}

/**
 * Suppress console output
 */
export function suppressConsoleOutput() {
  console.log = mockConsole.log;
  console.info = mockConsole.info;
  console.warn = mockConsole.warn;
  console.error = mockConsole.error;
  console.debug = mockConsole.debug;
}
