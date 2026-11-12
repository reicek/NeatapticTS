/**
 * RED home for Step 23B (slice `P4-S1-lane-red`) of
 * `plans/neatracing-voxel-racer.plans.md`.
 *
 * FOUR contract groups live here — one `describe` per RED-P4 contract, and
 * inside each `describe` ONE `it` per single-purpose behavior:
 * - RED-P4-11 (AC-029): with NO injected controller the fleet tick-input
 *   resolver raises a NAMED no-controller error for the AGENT cars (car 1..n)
 *   in BOTH race modes instead of falling back to the scripted autopilot; the
 *   car-0 human lane stays UNAFFECTED (the replayed human sample survives).
 * - RED-P4-12 (AC-029): the PRODUCTION worker source no longer imports
 *   `stepAutopilotInput` and never calls it (the legacy scripted lane is
 *   GONE — no dual path, no deferred cleanup).
 * - RED-P4-13 (AC-029): the READ-ONLY green test carries NO four-argument
 *   `resolveSimWorkerFleetTickInputs` call site and NO `stepAutopilotInput`
 *   import (the owed green-test re-pin, expressed as a measurable RED).
 * - RED-P4-14 (AC-029): the READ-ONLY green test's manual-mode SESSION
 *   fixture (the D5 block) injects a CONTROLLER, and the shared fixture
 *   fleet size stays 3 (the multi-car manual coverage SURVIVES).
 *
 * EVERY red in this file is a NAMED RUNTIME assertion failure BY
 * CONSTRUCTION: the file imports ONLY modules that already exist (at
 * RUNTIME), the not-yet-existing seam is read through a FILE-LOCAL WIDENED
 * VIEW (the `as unknown as { ... }` idiom of the byte-frozen
 * `simulation-worker.control-seam.test.ts`), and NO not-yet-existing error
 * CONSTANT is imported — a missing constant would surface as an import
 * failure instead of a legible assertion, so the error message is matched
 * with a TOLERANT regex instead.
 *
 * The fixture is DETERMINISTIC: fixed seeds (42 / 43), the SHIPPED autopilot
 * fixture track (one straight closed loop of 40 lane-center samples), NO RNG,
 * NO timers, NO wall-clock reads.
 *
 * The GREEN side must make the agent-car path REQUIRE the injected
 * controller (a NAMED no-controller error in BOTH race modes), delete the
 * scripted fallback from the worker source, and re-pin the read-only green
 * test at ALL THREE no-controller surfaces.
 */

import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { createTrackSpecForAutopilotFixture } from '../../physics/ai.autopilot.fixtures';
import { createInitialPhysicsCarState } from '../../physics/physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../../physics/physics.step.service';
import type { SplineSample } from '../../../racing_curriculum/track/track.generator.types';
import type { NeatRacingSimWorkerRaceMode } from './simulation-worker';

/** Lazily-loaded worker module type (imported in `beforeAll` after the stub). */
type SimWorkerModule = typeof import('./simulation-worker');

/**
 * Fake DedicatedWorkerGlobalScope the sim-worker module binds its message
 * adapter to at IMPORT time.
 */
const simWorkerScope: {
  onmessage: unknown;
  postMessage: (message: unknown) => void;
} = {
  onmessage: undefined,
  postMessage: (): void => undefined,
};

/** Original `self` property descriptor captured before the test fakes it. */
const originalSelfDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'self',
);

Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: simWorkerScope,
});

afterAll(() => {
  if (originalSelfDescriptor === undefined) {
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: undefined,
    });
  } else {
    Object.defineProperty(globalThis, 'self', originalSelfDescriptor);
  }
});

/** Worker module loaded in `beforeAll` once the fake scope has bound. */
let simWorkerModule: SimWorkerModule;

beforeAll(async () => {
  simWorkerModule = await import('./simulation-worker');
});

/**
 * The WIDENED view of the fleet tick-input resolver: the injected controller
 * is the FIFTH parameter, and it stays OPTIONAL in this view so the RED
 * contracts compile against BOTH the current signature and the future
 * REQUIRED one (a future required parameter can never break this file).
 */
type WidenedFleetTickResolver = (
  cars: readonly PhysicsCarState[],
  raceMode: NeatRacingSimWorkerRaceMode,
  humanSample: PhysicsStepInput | undefined,
  splineSamples: readonly SplineSample[],
  controller?: unknown,
) => PhysicsStepInput[];

/**
 * Reads the fleet tick-input resolver through the FILE-LOCAL WIDENED VIEW.
 *
 * The member EXISTS today, so the descriptive throw never fires in the RED
 * phase; it only guards the not-yet-widened signature so a future rename
 * surfaces as a legible message instead of a confusing `TypeError`.
 *
 * @returns The widened fleet tick-input resolver.
 */
function resolveFleetTickInputs(): WidenedFleetTickResolver {
  const widenedInternals =
    simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS as unknown as {
      resolveSimWorkerFleetTickInputs?: WidenedFleetTickResolver;
    };
  const resolver = widenedInternals.resolveSimWorkerFleetTickInputs;
  if (resolver === undefined) {
    throw new Error(
      'simulation worker: resolveSimWorkerFleetTickInputs is missing',
    );
  }
  return resolver;
}

/**
 * The fixture spline samples: the SHIPPED autopilot fixture (one straight
 * closed loop of 40 lane-center samples along +x at y = 0, stride 1 m).
 *
 * The shipped fixture is reused on purpose: it keeps the scripted fallback
 * on a SANE, non-degenerate racing line, so the RED contracts observe the
 * fallback's ABSENCE rather than a degenerate control answer.
 */
const LANE_RETIREMENT_SPLINE_SAMPLES: readonly SplineSample[] =
  createTrackSpecForAutopilotFixture();

/**
 * The replayed D5 human sample for car 0 (the human lane's control input).
 */
const LANE_RETIREMENT_HUMAN_SAMPLE: PhysicsStepInput = {
  throttle: 1,
  brake: 0,
  steer: -0.5,
};

/**
 * Builds the fixture fleet: car 0 is the HUMAN car and car 1 is an AGENT
 * car. Both sit ON the fixture racing line (y = 0, heading +x) so the
 * scripted fallback has a sane line to hold.
 *
 * @returns The two deterministic fixture car states.
 */
function createLaneRetirementFleet(): PhysicsCarState[] {
  return [
    createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
    createInitialPhysicsCarState(43, { x: 14, y: 0, heading: 0 }),
  ];
}

/**
 * The TOLERANT no-controller error pattern: the named error may spell the
 * reason as "no controller", "no-controller", "missing controller",
 * "controller is required" or "without a controller". The contract pins the
 * REASON, not one exact wording.
 */
const NO_CONTROLLER_ERROR_PATTERN =
  /(no[- ]?controller|missing controller|controller is required|without a controller)/i;

/** Repo-relative path of the PRODUCTION sim-worker source. */
const PRODUCTION_WORKER_SOURCE_PATH =
  'examples/neatRacing/workers/simulation-worker/simulation-worker.ts';

/** Repo-relative path of the READ-ONLY green sim-worker test. */
const GREEN_WORKER_TEST_SOURCE_PATH =
  'examples/neatRacing/workers/simulation-worker/simulation-worker.test.ts';

/**
 * Reads one repo-relative source file as UTF-8 text.
 *
 * @param repoRelativePath - The repo-relative path of the file.
 * @returns The file text.
 */
function readRepoSource(repoRelativePath: string): string {
  return readFileSync(resolve(process.cwd(), repoRelativePath), 'utf-8');
}

/**
 * Skips a quoted string literal that starts at `startIndex` (the opening
 * quote character).
 *
 * @param source - The scanned source text.
 * @param startIndex - The index of the opening quote.
 * @returns The index just past the closing quote.
 */
function skipQuotedLiteral(source: string, startIndex: number): number {
  const quote = source[startIndex];
  let index = startIndex + 1;
  while (index < source.length) {
    const char = source[index];
    if (char === '\\') {
      index += 2;
      continue;
    }
    if (char === quote) {
      return index + 1;
    }
    index += 1;
  }
  return index;
}

/**
 * Skips a template literal that starts at `startIndex` (the opening
 * backtick), including any `${ ... }` interpolation (whose braces are
 * brace-counted so a nested quote never ends the literal early).
 *
 * @param source - The scanned source text.
 * @param startIndex - The index of the opening backtick.
 * @returns The index just past the closing backtick.
 */
function skipTemplateLiteral(source: string, startIndex: number): number {
  let index = startIndex + 1;
  while (index < source.length) {
    const char = source[index];
    if (char === '\\') {
      index += 2;
      continue;
    }
    if (char === '`') {
      return index + 1;
    }
    if (char === '$' && source[index + 1] === '{') {
      const closingBrace = findMatchingBracket(source, index + 1);
      index = closingBrace === -1 ? source.length : closingBrace + 1;
      continue;
    }
    index += 1;
  }
  return index;
}

/**
 * Finds the index of the bracket that CLOSES the one opened at
 * `openIndex`, skipping string literals, template literals and comments.
 *
 * @param source - The scanned source text.
 * @param openIndex - The index of the opening bracket.
 * @returns The index of the matching closing bracket, or `-1`.
 */
function findMatchingBracket(source: string, openIndex: number): number {
  const openChar = source[openIndex];
  const closeChar =
    openChar === '('
      ? ')'
      : openChar === '{'
        ? '}'
        : openChar === '['
          ? ']'
          : '';
  if (closeChar === '') {
    return -1;
  }
  let depth = 0;
  let index = openIndex;
  while (index < source.length) {
    const char = source[index];
    if (char === '"' || char === "'") {
      index = skipQuotedLiteral(source, index);
      continue;
    }
    if (char === '`') {
      index = skipTemplateLiteral(source, index);
      continue;
    }
    if (char === '/' && source[index + 1] === '/') {
      const lineEnd = source.indexOf('\n', index);
      index = lineEnd === -1 ? source.length : lineEnd + 1;
      continue;
    }
    if (char === '/' && source[index + 1] === '*') {
      const commentEnd = source.indexOf('*/', index + 2);
      index = commentEnd === -1 ? source.length : commentEnd + 2;
      continue;
    }
    if (char === openChar) {
      depth += 1;
      index += 1;
      continue;
    }
    if (char === closeChar) {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
      index += 1;
      continue;
    }
    index += 1;
  }
  return -1;
}

/**
 * Splits one call site's argument text at its TOP-LEVEL commas (a comma
 * nested inside a call, an array, an object or a literal never splits).
 *
 * @param argumentText - The balanced-paren argument text of one call site.
 * @returns The argument segments.
 */
function splitTopLevelArguments(argumentText: string): string[] {
  const segments: string[] = [];
  let depth = 0;
  let segmentStart = 0;
  let index = 0;
  while (index < argumentText.length) {
    const char = argumentText[index];
    if (char === '"' || char === "'") {
      index = skipQuotedLiteral(argumentText, index);
      continue;
    }
    if (char === '`') {
      index = skipTemplateLiteral(argumentText, index);
      continue;
    }
    if (char === '/' && argumentText[index + 1] === '/') {
      const lineEnd = argumentText.indexOf('\n', index);
      index = lineEnd === -1 ? argumentText.length : lineEnd + 1;
      continue;
    }
    if (char === '/' && argumentText[index + 1] === '*') {
      const commentEnd = argumentText.indexOf('*/', index + 2);
      index = commentEnd === -1 ? argumentText.length : commentEnd + 2;
      continue;
    }
    if (char === '(' || char === '[' || char === '{') {
      depth += 1;
      index += 1;
      continue;
    }
    if (char === ')' || char === ']' || char === '}') {
      depth -= 1;
      index += 1;
      continue;
    }
    if (char === ',' && depth === 0) {
      segments.push(argumentText.slice(segmentStart, index));
      segmentStart = index + 1;
    }
    index += 1;
  }
  segments.push(argumentText.slice(segmentStart));
  return segments;
}

/**
 * Counts the REAL arguments of one call site (a trailing comma does not add
 * an argument).
 *
 * @param argumentText - The balanced-paren argument text of one call site.
 * @returns The argument count.
 */
function countCallSiteArguments(argumentText: string): number {
  if (argumentText.trim() === '') {
    return 0;
  }
  const segments = splitTopLevelArguments(argumentText);
  const lastIndex = segments.length - 1;
  if (segments[lastIndex].trim() === '') {
    segments.pop();
  }
  return segments.length;
}

/**
 * Collects the REAL argument count of every call site of `calleeName` in the
 * scanned source.
 *
 * Educational note — why a naive regex is WRONG here: a pattern such as
 * `/resolveSimWorkerFleetTickInputs\([^)]*,[^)]*,[^)]*,[^)]*\)/` lets
 * `[^)]*` absorb the separating commas, so it matches FIVE-argument call
 * sites as well as FOUR-argument ones and can NEVER prove the absence of a
 * four-argument call. The argument text is therefore extracted with a
 * balanced-paren walk and split at TOP-LEVEL commas.
 *
 * @param source - The scanned source text.
 * @param calleeName - The called function's name.
 * @returns The argument count of every call site, in source order.
 */
function collectCallSiteArgumentCounts(
  source: string,
  calleeName: string,
): number[] {
  const counts: number[] = [];
  const needle = `${calleeName}(`;
  let searchIndex = source.indexOf(needle);
  while (searchIndex !== -1) {
    const openParen = searchIndex + needle.length - 1;
    const closeParen = findMatchingBracket(source, openParen);
    if (closeParen === -1) {
      break;
    }
    counts.push(
      countCallSiteArguments(source.slice(openParen + 1, closeParen)),
    );
    searchIndex = source.indexOf(needle, closeParen);
  }
  return counts;
}

/**
 * Extracts the text of one `describe` block: the header is located by its
 * title and the block is closed by brace counting (so nested `it` blocks and
 * object literals never end the block early).
 *
 * @param source - The scanned source text.
 * @param describeTitle - The describe block's title text.
 * @returns The block text (including its braces), or an empty string.
 */
function extractDescribeBlockText(
  source: string,
  describeTitle: string,
): string {
  const titleIndex = source.indexOf(describeTitle);
  if (titleIndex === -1) {
    return '';
  }
  const openBrace = source.indexOf('{', titleIndex);
  if (openBrace === -1) {
    return '';
  }
  const closeBrace = findMatchingBracket(source, openBrace);
  if (closeBrace === -1) {
    return '';
  }
  return source.slice(openBrace, closeBrace + 1);
}

/**
 * Collects every declared fleet size of the scanned source: the plan's
 * tolerant `/AGENT_COUNT\s*=\s*(\d+)/` scan plus a rename-tolerant
 * resolution of an `agentCount: <identifier>` payload field through its own
 * numeric declaration.
 *
 * @param source - The scanned source text.
 * @returns The declared fleet sizes.
 */
function collectDeclaredFleetSizes(source: string): number[] {
  const sizes: number[] = [];
  for (const match of source.matchAll(/AGENT_COUNT\s*=\s*(\d+)/g)) {
    sizes.push(Number(match[1]));
  }
  for (const match of source.matchAll(/agentCount\s*:\s*([A-Za-z_$][\w$]*)/g)) {
    const identifier = match[1];
    const declaration = new RegExp(`\\b${identifier}\\s*=\\s*(\\d+)`).exec(
      source,
    );
    if (declaration !== null) {
      sizes.push(Number(declaration[1]));
    }
  }
  return sizes;
}

describe('RED-P4-11 - the agent-car path REQUIRES the injected controller (AC-029)', () => {
  it('raises the named no-controller error for the agent cars in ai mode', () => {
    // Arrange: the deterministic fixture fleet (car 0 human + car 1 agent),
    // the fixture spline samples and the replayed human sample — with NO
    // controller injected.
    const cars = createLaneRetirementFleet();
    const resolveFleetTickInputsOf = resolveFleetTickInputs();

    // Act: resolve one tick's inputs in ai mode without a controller.
    const resolveAiFleetWithoutController = (): PhysicsStepInput[] =>
      resolveFleetTickInputsOf(
        cars,
        'ai',
        LANE_RETIREMENT_HUMAN_SAMPLE,
        LANE_RETIREMENT_SPLINE_SAMPLES,
      );

    // Assert: the agent-car path refuses to fall back to the scripted
    // autopilot and names the missing controller in the error message.
    expect(resolveAiFleetWithoutController).toThrow(
      NO_CONTROLLER_ERROR_PATTERN,
    );
    expect(resolveAiFleetWithoutController).toThrow(Error);
  });

  it('raises the same named no-controller error for the agent cars in manual mode', () => {
    // Arrange: the deterministic fixture fleet (car 0 human + car 1 agent),
    // the fixture spline samples and the replayed human sample — with NO
    // controller injected.
    const cars = createLaneRetirementFleet();
    const resolveFleetTickInputsOf = resolveFleetTickInputs();

    // Act: resolve one tick's inputs in manual mode without a controller.
    const resolveManualFleetWithoutController = (): PhysicsStepInput[] =>
      resolveFleetTickInputsOf(
        cars,
        'manual',
        LANE_RETIREMENT_HUMAN_SAMPLE,
        LANE_RETIREMENT_SPLINE_SAMPLES,
      );

    // Assert: the manual lane carries the SAME named no-controller error (no
    // dual path — the scripted fallback is retired in BOTH race modes).
    expect(resolveManualFleetWithoutController).toThrow(
      NO_CONTROLLER_ERROR_PATTERN,
    );
    expect(resolveManualFleetWithoutController).toThrow(Error);
  });

  it('keeps the car-0 human lane UNAFFECTED in manual mode without a controller (guard)', () => {
    // Arrange: the human-only fixture fleet (car 0 alone) and the replayed
    // human sample.
    const cars = [createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 })];
    const resolveFleetTickInputsOf = resolveFleetTickInputs();

    // Act: resolve one tick's inputs for the human-only fleet.
    const manualInputs = resolveFleetTickInputsOf(
      cars,
      'manual',
      LANE_RETIREMENT_HUMAN_SAMPLE,
      LANE_RETIREMENT_SPLINE_SAMPLES,
    );

    // Assert: the fleet resolves to EXACTLY one input (no agent car, so no
    // no-controller error).
    expect(manualInputs).toHaveLength(1);

    // Assert: car 0 still replays the HUMAN sample (the LD-12 human lane is
    // never handed to the network).
    expect(manualInputs[0]).toEqual({
      throttle: 1,
      brake: 0,
      steer: -0.5,
    });
  });
});

describe('RED-P4-12 - the legacy scripted lane is GONE from the production worker source (AC-029)', () => {
  it('does not import the scripted autopilot step function', () => {
    // Arrange: the production worker source.
    const source = readRepoSource(PRODUCTION_WORKER_SOURCE_PATH);

    // Assert: the `stepAutopilotInput` import is GONE. The scan is an
    // import-shaped regex (NOT a whole-file `not.toContain`): the header
    // doc mentions the identifier in PROSE, and only the import statement
    // is the retired dependency.
    expect(source).not.toMatch(/import\s*\{[^}]*\bstepAutopilotInput\b[^}]*\}/);
  });

  it('does not call the scripted autopilot step function', () => {
    // Arrange: the production worker source.
    const source = readRepoSource(PRODUCTION_WORKER_SOURCE_PATH);

    // Assert: the scripted fallback CALL is GONE — no call site of the
    // retired step function survives in the production worker.
    expect(source).not.toMatch(/\bstepAutopilotInput\s*\(/);
  });
});

describe('RED-P4-13 - the read-only green test carries no four-argument resolver call site (AC-029)', () => {
  it('carries no four-argument resolveSimWorkerFleetTickInputs call site', () => {
    // Arrange: the read-only green test source.
    const source = readRepoSource(GREEN_WORKER_TEST_SOURCE_PATH);

    // Act: parse the REAL argument count of every resolver call site.
    const callSiteArgumentCounts = collectCallSiteArgumentCounts(
      source,
      'resolveSimWorkerFleetTickInputs',
    );

    // Assert: the scan actually FOUND the call sites (the scan is not
    // vacuous).
    expect(callSiteArgumentCounts.length).toBeGreaterThan(0);

    // Assert: NO call site passes exactly FOUR arguments (the no-controller
    // surface that must be re-pinned to the controller answer).
    expect(callSiteArgumentCounts).not.toContain(4);
  });

  it('does not import the scripted autopilot step function', () => {
    // Arrange: the read-only green test source.
    const source = readRepoSource(GREEN_WORKER_TEST_SOURCE_PATH);

    // Assert: the `stepAutopilotInput` import is GONE from the green test
    // (the re-pinned test asserts the CONTROLLER answer, not the scripted
    // autopilot answer).
    expect(source).not.toMatch(/import\s*\{[^}]*\bstepAutopilotInput\b[^}]*\}/);
  });
});

describe('RED-P4-14 - the read-only green test manual-mode session fixture injects the controller (AC-029)', () => {
  it('injects a controller into the manual-mode session fixture of the D5 block', () => {
    // Arrange: the D5 describe block of the read-only green test, extracted
    // by brace counting.
    const source = readRepoSource(GREEN_WORKER_TEST_SOURCE_PATH);
    const d5BlockText = extractDescribeBlockText(
      source,
      'D5 control sampling (manual mode)',
    );

    // Assert: the D5 block was FOUND (the scan is not vacuous).
    expect(d5BlockText.length).toBeGreaterThan(0);

    // Assert: the manual-mode SESSION fixture injects a CONTROLLER into its
    // init payload (the B-2 no-controller surface).
    expect(d5BlockText).toMatch(/\bcontroller\b/);
  });

  it('keeps the shared fixture fleet size at 3 (guard)', () => {
    // Arrange: the read-only green test source.
    const source = readRepoSource(GREEN_WORKER_TEST_SOURCE_PATH);

    // Act: collect every declared fleet size.
    const declaredFleetSizes = collectDeclaredFleetSizes(source);

    // Assert: the fleet size stays 3 — the multi-car manual coverage
    // SURVIVES (shrinking the manual fleet to one car is FORBIDDEN).
    expect(declaredFleetSizes).toContain(3);

    // Assert: no other fleet size is declared.
    expect(declaredFleetSizes.filter((size) => size !== 3)).toEqual([]);
  });
});
