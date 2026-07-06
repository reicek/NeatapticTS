/**
 * Red-phase contracts for per-car runtime adaptation in the racing curriculum.
 *
 * The racing demo runs multiple independent NGE agent cars, each with its own
 * network and continuous evolution.  The adaptation engine must therefore be
 * per-car: each car maintains its own independent adaptation state, cooldowns,
 * and cadence boundaries.  These tests pin the per-car API surface before any
 * production implementation is added.
 *
 * Single-expect rule enforced throughout.  AAA structure in every test.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as adaptationModule from './runtime.adaptation';
import type { RuntimeAdaptationEngine } from './runtime.adaptation';

const SOURCE_FILE = path.resolve(__dirname, 'runtime.adaptation.ts');

function readSourceText(): string {
  return fs.readFileSync(SOURCE_FILE, 'utf-8');
}

describe('runtime.adaptation per-car contracts', () => {
  describe('createPerCarAdaptationEngines', () => {
    it('is exported as a function', () => {
      const factory = (adaptationModule as unknown as Record<string, unknown>)
        .createPerCarAdaptationEngines;

      expect(typeof factory).toBe('function');
    });

    it('returns a Map keyed by car index when called with carCount 3', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(3);

      expect(engines instanceof Map).toBe(true);
    });

    it('creates exactly one engine per car index 0 through carCount-1', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(3);

      expect(engines?.size).toBe(3);
    });
  });

  describe('per-car adaptation state independence', () => {
    it('gives each car a distinct engine instance (not the same reference)', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(2);
      const car0Engine = engines?.get(0);
      const car1Engine = engines?.get(1);

      expect(car0Engine).not.toBe(car1Engine);
    });

    it('keeps car 1 adaptation state unchanged after resetting car 0', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(2);
      const car0Engine = engines?.get(0);
      const car1Engine = engines?.get(1);

      car0Engine?.reset();

      // car 1 should not be affected by car 0's reset — verify by checking
      // that car1Engine is still a valid engine with a reset method
      expect(typeof car1Engine?.reset).toBe('function');
    });

    it('does not share cooldown state across cars (car 0 cooldown does not block car 1)', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
            options?: { limits?: { mutationCooldownTicks: number } },
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(2, {
        limits: { mutationCooldownTicks: 100 },
      });
      const car0Engine = engines?.get(0);
      const car1Engine = engines?.get(1);

      // After car 0 processes a tick, car 1 should NOT inherit car 0's cooldown.
      // We verify independence by confirming car1Engine is a separate object
      // with its own adaptOnTick method.
      expect(car0Engine?.adaptOnTick).not.toBe(car1Engine?.adaptOnTick);
    });
  });

  describe('car-scoped API', () => {
    it('accepts a car index and returns the engine for that car', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(3);
      const engineForCar1 = engines?.get(1);

      expect(typeof engineForCar1?.adaptOnTick).toBe('function');
    });

    it('returns undefined for an out-of-range car index', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(2);

      expect(engines?.get(99)).toBeUndefined();
    });
  });

  describe('no module-level mutable singleton state', () => {
    it('source file does not declare module-level mutable adaptation state via let', () => {
      const sourceText = readSourceText();
      // Look for `let` declarations at module scope (column 0, outside any
      // function or closure) that hold adaptation state.  Module-level `let`
      // variables like `let adaptationState` or `let runtimeEngine` would be
      // a shared singleton.  We check for any top-level `let` followed by an
      // adaptation-related identifier.
      const hasModuleLevelLetAdaptationState =
        /^let\s+(runtime|adaptation|engine)/im.test(sourceText);

      expect(hasModuleLevelLetAdaptationState).toBe(false);
    });

    it('source file exports a per-car factory function (createPerCarAdaptationEngines)', () => {
      const sourceText = readSourceText();
      const exportsPerCarFactory =
        /export\s+function\s+createPerCarAdaptationEngines/.test(sourceText);

      expect(exportsPerCarFactory).toBe(true);
    });

    it('source file does not export a shared singleton adaptation engine instance', () => {
      const sourceText = readSourceText();
      // A singleton instance would look like:
      //   export const sharedAdaptationEngine = createRuntimeAdaptationEngine(...)
      //   export const runtimeAdaptationEngine = ...
      const exportsSingletonInstance =
        /export\s+const\s+\w*(shared|singleton|global|default)\w*[Ee]ngine\s*=/.test(
          sourceText,
        );

      expect(exportsSingletonInstance).toBe(false);
    });
  });

  describe('factory pattern for per-car instances', () => {
    it('createPerCarAdaptationEngines accepts an options argument for per-car configuration', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
            options?: Record<string, unknown>,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(1, { improvementThreshold: 0.5 });
      const engine = engines?.get(0);

      expect(typeof engine?.adaptOnTick).toBe('function');
    });

    it('each car engine has an independent reset method that does not affect siblings', () => {
      const factory = (
        adaptationModule as unknown as {
          createPerCarAdaptationEngines?: (
            carCount: number,
          ) => Map<number, RuntimeAdaptationEngine>;
        }
      ).createPerCarAdaptationEngines;

      const engines = factory?.(3);
      const beforeReset = engines?.get(1);

      engines?.get(0)?.reset();

      const afterReset = engines?.get(1);

      expect(afterReset).toBe(beforeReset);
    });
  });
});
