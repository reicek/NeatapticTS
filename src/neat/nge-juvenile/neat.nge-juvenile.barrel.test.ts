/**
 * Barrel import coverage test for the NGE juvenile module.
 *
 * Ensures the barrel file (`neat.nge-juvenile.ts`) and the types file
 * (`neat.nge-juvenile.types.ts`) are loaded during test execution so that
 * coverage instrumentation tracks their lines.
 *
 * The barrel file only contains `export *` re-exports — importing through it
 * exercises every re-export path. The types file is pure type declarations
 * with no runtime code, but the side-effect import forces the module to be
 * resolved and instrumented.
 */
import { computeFocusScores, resolveFocusConfig } from './neat.nge-juvenile';
import './neat.nge-juvenile.types';

describe('NGE juvenile barrel re-exports', () => {
  it('re-exports computeFocusScores as a function', () => {
    // Act
    const fnType = typeof computeFocusScores;

    // Assert
    expect(fnType).toBe('function');
  });

  it('re-exports resolveFocusConfig as a function', () => {
    // Act
    const fnType = typeof resolveFocusConfig;

    // Assert
    expect(fnType).toBe('function');
  });
});
