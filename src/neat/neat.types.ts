import type {
  AnyObj,
  NeatOptions as SharedNeatOptions,
} from './shared/neat.shared.types';
import { selectMutationMethod } from './mutation/mutation';
import * as neatRngFacade from './rng/facade/rng.facade';
import { fromJSONImpl } from './export/neat.export';

/**
 * Root compatibility types for the public `Neat` controller boundary.
 *
 * The chapter exists to keep `src/neat.ts` orchestration-first while still
 * giving the root surface one explicit place to document its compatibility
 * contracts. These are not the deepest or strongest types in the controller.
 * They are the adapter types the stable public entrypoint needs while the
 * chaptered implementation keeps narrowing local contracts underneath.
 *
 * Read the file in three passes:
 *
 * - start with `NeatOptions` to understand the root option bag,
 * - continue to `NeatFitnessFunction` when you need the public scoring seam,
 * - finish with the restore and mutation aliases when you are tracing how the
 *   root facade forwards work into the `init/`, `rng/`, and `export/` chapters.
 */

/**
 * Public configuration bag accepted by the root `Neat` constructor.
 *
 * This alias stays intentionally permissive because the public boundary still
 * absorbs legacy experiment bags, partially migrated option families, and a
 * few chapter-local knobs that do not yet deserve a tighter shared contract.
 *
 * That looseness is a boundary choice rather than a shared-type ideal. The
 * root facade accepts the broad option surface so the deeper helper chapters
 * can keep narrowing their own local slices instead of reintroducing one wide
 * compatibility bag in multiple places.
 */
export type NeatOptions = SharedNeatOptions & AnyObj;

/**
 * Opaque result shape returned by root-level fitness callbacks.
 *
 * The top-level `Neat` entrypoint has to tolerate both single-genome and
 * population-wide fitness styles, including delegates that perform async work
 * or side effects before downstream evaluation helpers interpret the result.
 * The root contract therefore stays wide on purpose.
 */
export type NeatFitnessResult = unknown;

/**
 * Root compatibility shape for fitness callbacks accepted by `Neat`.
 *
 * Read this as a facade contract rather than a claim that the root file owns
 * every legal scoring protocol. The constructor only promises that a scoring
 * delegate can be stored and forwarded safely; the stronger semantics live in
 * the evaluation and evolve chapters that actually consume the callback.
 */
export type NeatFitnessFunction = (...args: never[]) => NeatFitnessResult;

/**
 * Awaited return shape for the public mutation-method selection wrapper.
 *
 * The mutation chapter already owns the concrete union. This alias keeps the
 * root class synchronized with that source of truth without repeating a legacy
 * compatibility union inline.
 */
export type NeatMutationSelectionResult = Awaited<
  ReturnType<typeof selectMutationMethod>
>;

/**
 * Replay token accepted by the public RNG restore and import methods.
 *
 * Deriving the token from the RNG facade keeps the root entrypoint aligned with
 * the replay chapter instead of maintaining a second hand-written copy of the
 * same restore contract.
 */
export type NeatRngStateSnapshot = Parameters<
  typeof neatRngFacade.restoreRNGState
>[1];

/**
 * Fitness callback shape expected by the export/import restore helpers.
 *
 * The persistence chapter reconstructs a controller from serialized state and
 * then reattaches a scoring delegate. The root surface derives that callback
 * type from the export chapter so the static restore helpers stay in lockstep
 * with the real persistence contract.
 */
export type NeatExportFitnessFunction = Parameters<typeof fromJSONImpl>[1];
