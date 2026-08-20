/**
 * @deprecated This monolith test file has been split into focused test files
 * as part of the B2 architecture-debt refactoring (S5: split monolith test file).
 *
 * The tests previously in this file now live in:
 *   - display.worker.init.test.ts          — init lifecycle tests
 *   - display.worker.sim.test.ts           — sim state and frame tests
 *   - display.worker.render.test.ts        — rendering and sprite tests
 *   - display.worker.auto-ai.test.ts       — auto-AI and fallback tests
 *   - display.worker.eval-delegation.test.ts — eval worker helper tests
 *
 * Shared test helpers were extracted to:
 *   - display.worker.test-helpers.ts
 *
 * This file is intentionally left as a marker. Do not add new tests here.
 */
import { describe, expect, it } from '@jest/globals';

describe('display.worker.test.ts (deprecated — split into focused files)', () => {
  it('is a deprecation marker', () => {
    expect(true).toBe(true);
  });
});
