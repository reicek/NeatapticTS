/**
 * @file RED contracts for the DETERMINISTIC TRACK CATALOG (Step 25).
 *
 * Slice `P4-S3-red` — Phase 4, Step 25 of
 * `plans/neatracing-voxel-racer.plans.md` (the user-directed LD-08
 * amendment: a deterministic multi-track catalog + selection + restart).
 *
 * Educational note — WHAT THIS FILE OWNS:
 * GAP-3 is a SINGLE PINNED TRACK: `browser-entry.ts` boots one hard-coded
 * demo track (`NEATRACING_DEMO_TRACK_ID = 'neatracing-demo-loop'`) from one
 * hard-coded generation input (seed 42 / layoutVersion 1 / sizeBucket
 * 'large'), so the demo can never show a second circuit and a round can
 * never be replayed on another geometry. Step 25 closes the gap with a
 * DETERMINISTIC TRACK CATALOG (several named variants, every one a
 * deterministic function of its own seed/variant id, all produced by the
 * EXISTING deterministic generator in
 * `examples/racing_curriculum/track/track.generator.ts`) plus a host
 * selection control that restarts the race on the selected track through
 * the EXISTING selector notify/restart contract (the `onSwitch` law of
 * `examples/neatRacing/host/mode-selector.ts`).
 *
 * Educational note — THE TWO CONTRACTS:
 * - RED-P4-7 — THE CATALOG SHAPE AND ITS DETERMINISM. The catalog carries
 *   at least FOUR DISTINCT variants (AC-100 pins
 *   `NEATRACING_DEMO_TRACK_CATALOG` + `NEATRACING_DEMO_TRACK_VARIANT_COUNT`),
 *   two identical resolve calls produce IDENTICAL specs per variant, the
 *   PINNED demo loop id stays a catalog MEMBER and its geometry stays the
 *   SHIPPED demo geometry (no boot-default regression).
 * - RED-P4-8 — THE SELECTION RESTART. A selection switch notifies the
 *   restart hook EXACTLY ONCE with the SELECTED track id, the switch
 *   restarts in EITHER direction, and the round boots on a FRESH
 *   reproducibility tuple (the AC-026 tuple carries the selected track id)
 *   while the per-track tuple stays RETAINED when the selection switches
 *   back.
 *
 * Educational note — THE RED GATE (why nothing is statically imported). At
 * HEAD neither `examples/neatRacing/track/track.catalog.ts` nor
 * `examples/neatRacing/host/track.selector.ts` exists, so a static import
 * would fail the suite at COMPILE time (TS 2307) — the WRONG red reason.
 * Both modules are therefore loaded through a WIDENED dynamic import inside
 * the gate helpers, and the gate answers with a NAMED error that names the
 * MISSING BEHAVIOUR (the demo still runs ONE pinned track and the host has
 * no track selection), never a bare "module not found" (the house pattern
 * of `examples/neatRacing/browser-entry/debug-rung.red.test.ts`).
 *
 * Educational note — WHY THE CONTRACTS ARE DISCOVERY-BASED. The two NAMED
 * constants are pinned by AC-100, but the RESOLVER and the SELECTION SEAM
 * are discovered BY CAPABILITY (the exported resolver that answers a
 * variant id with its geometry; the selection factory that answers with a
 * switch plus a subscribe hook). The contracts therefore pin the
 * BEHAVIOUR, not a spelling: the implementing slices may name their
 * resolver and their selection factory freely as long as the behaviour
 * holds. A wrong pick fails LOUDLY (the assertions are written against
 * observable facts), so the discovery can never make a contract pass
 * vacuously.
 *
 * Educational note — THE ENVIRONMENT. This file runs in the NODE
 * environment (there is deliberately no `@jest-environment jsdom` docblock
 * here): every contract is DOM-free, so the selection seam must expose a
 * DOM-free core — the same law the W2 debug-rung seam carries (W2-PURE).
 *
 * Educational note — THE PINNED SURFACES the implementing slices must
 * satisfy (the RED home is byte-frozen; the implementing slices edit the
 * production code and NEVER these contracts):
 * - `examples/neatRacing/track/track.catalog.ts` (NEW) exports
 *   `NEATRACING_DEMO_TRACK_CATALOG` (the named variants: a list or a keyed
 *   map) and `NEATRACING_DEMO_TRACK_VARIANT_COUNT` (the declared count),
 *   plus a resolver that turns a variant id into the geometry the
 *   deterministic generator produced for it.
 * - `examples/neatRacing/host/track.selector.ts` (NEW) exports a DOM-free
 *   selection surface: something that switches the selection to a named
 *   track and notifies a subscriber (the restart hook) with the selected
 *   track id, exactly once per actual switch.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import { NEATRACING_PACK_SCHEMA_VERSION } from '../workers/simulation-worker/frame.pack.service';
import {
  serializeNeatRacingReproducibilityTuple,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';

/* ------------------------------------------------------------------------- *
 * The pinned world of the RED contracts.
 * ------------------------------------------------------------------------- */

/**
 * The catalog module path (WIDENED on purpose — see the red gate above).
 *
 * The literal does not exist yet at HEAD, so a STATIC import of it would
 * fail the suite at compile time (TS 2307) — the wrong red reason.
 */
const RED_TRACK_CATALOG_MODULE_PATH = './track.catalog';

/**
 * The host track-selection module path (WIDENED on purpose — see the red
 * gate above).
 */
const RED_TRACK_SELECTOR_MODULE_PATH = '../host/track.selector';

/**
 * The PINNED demo track id the host ships today.
 *
 * The id is the demo's honest identity in the worker's reproducibility
 * tuple (`browser-entry.ts`: `trackId: NEATRACING_DEMO_TRACK_ID`). The
 * catalog must keep it as a MEMBER so the shipped boot default does not
 * regress.
 */
const RED_PINNED_DEMO_TRACK_ID = 'neatracing-demo-loop';

/**
 * The SHIPPED demo track generation input (the boot default of today).
 *
 * The catalog's pinned demo loop member must resolve to EXACTLY this
 * geometry — a demo boot that silently changed circuit is a behavioural
 * regression of the current default, not a catalog feature.
 */
const RED_SHIPPED_DEMO_TRACK_GENERATION = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/**
 * The field names a catalog member may carry its track id under.
 *
 * The catalog owns the id spelling; the contract only needs to READ it.
 */
const RED_MEMBER_ID_FIELDS: readonly string[] = [
  'id',
  'trackId',
  'variantId',
  'key',
  'name',
];

/**
 * The method names a selection surface may expose its switch under.
 */
const RED_SELECTION_SWITCH_METHOD_NAMES: readonly string[] = [
  'setTrack',
  'selectTrack',
  'switchTrack',
  'setSelectedTrack',
  'setTrackId',
  'select',
];

/**
 * The method names a selection surface may expose its subscribe hook under.
 */
const RED_SELECTION_SUBSCRIBE_METHOD_NAMES: readonly string[] = [
  'onSwitch',
  'onSelect',
  'onChange',
  'subscribe',
  'addListener',
];

/**
 * The field names a selection surface (or a notify payload) may carry the
 * selected track id under.
 */
const RED_SELECTION_TRACK_ID_FIELDS: readonly string[] = [
  'trackId',
  'selectedTrackId',
  'value',
  'id',
];

/* ------------------------------------------------------------------------- *
 * The gated surfaces.
 * ------------------------------------------------------------------------- */

/** The catalog surface the red gate hands back. */
interface RedTrackCatalogSurface {
  /** The catalog value (a list of variants or a keyed map of variants). */
  readonly catalog: unknown;
  /** The declared variant count (AC-100's variant-count constant). */
  readonly variantCount: number;
  /** The catalog module's exported surface. */
  readonly exports: Readonly<Record<string, unknown>>;
}

/** One catalog member (one named track variant). */
interface RedTrackCatalogMember {
  /** The member's track id (the id the host selects and the tuple carries). */
  readonly id: string;
  /** The member's own entry (its generation input or its resolved spec). */
  readonly entry: unknown;
}

/** The catalog context the contracts work against. */
interface RedTrackCatalogContext {
  /** The gated catalog surface. */
  readonly surface: RedTrackCatalogSurface;
  /** The catalog's members (the named variants). */
  readonly members: readonly RedTrackCatalogMember[];
}

/** The DOM-free selection surface the red gate hands back. */
interface RedTrackSelectionSurface {
  /** Switches the selection to the given track id. */
  readonly switchTo: (trackId: string) => void;
  /** Subscribes a listener to the selection switch (the restart hook). */
  readonly subscribe: (listener: (payload: unknown) => void) => void;
  /** Reads the currently selected track id (null when the seam hides it). */
  readonly currentTrackId: () => string | null;
}

/** The selection context the RED-P4-8 contracts work against. */
interface RedTrackSelectionContext {
  /** The gated catalog context (the selection works over the catalog). */
  readonly context: RedTrackCatalogContext;
  /** The gated DOM-free selection surface. */
  readonly selection: RedTrackSelectionSurface;
}

/* ------------------------------------------------------------------------- *
 * The red gate helpers.
 * ------------------------------------------------------------------------- */

/**
 * Loads a module that does not exist yet, WITHOUT failing the suite.
 *
 * Educational note — the widened import. The specifier is held in a
 * VARIABLE on purpose: a literal specifier for a missing module is a
 * compile-time error (TS 2307), which is the WRONG red reason. The widened
 * string keeps the dynamic import compilable at HEAD and at GREEN.
 *
 * @param modulePath - The widened module specifier to load.
 * @returns The loaded module namespace, or undefined when it is absent.
 */
async function loadRedModule(
  modulePath: string,
): Promise<Record<string, unknown> | undefined> {
  let loaded: Record<string, unknown> | undefined;
  try {
    loaded = (await import(modulePath)) as Record<string, unknown>;
  } catch {
    loaded = undefined;
  }
  return loaded;
}

/**
 * Reads the track id a catalog member carries.
 *
 * @param entry - The member's own entry.
 * @param fallbackId - The id to use when the entry carries none.
 * @returns The member's track id.
 */
function redMemberId(entry: unknown, fallbackId: string): string {
  if (typeof entry === 'object' && entry !== null) {
    const record = entry as Record<string, unknown>;
    for (const field of RED_MEMBER_ID_FIELDS) {
      const value = record[field];
      if (typeof value === 'string' && value.length > 0) {
        return value;
      }
    }
  }
  return fallbackId;
}

/**
 * Normalizes the catalog into its named members.
 *
 * @param catalog - The catalog value (a list or a keyed map of variants).
 * @param contractId - The RED contract the helper speaks for.
 * @returns The catalog's named members.
 * @throws When the catalog is neither a list nor a keyed map.
 */
function redCatalogMembers(
  catalog: unknown,
  contractId: string,
): RedTrackCatalogMember[] {
  if (Array.isArray(catalog)) {
    return catalog.map(
      (entry: unknown, index: number): RedTrackCatalogMember => ({
        id: redMemberId(entry, `variant-${index}`),
        entry,
      }),
    );
  }
  if (typeof catalog === 'object' && catalog !== null) {
    return Object.entries(catalog as Record<string, unknown>).map(
      ([id, entry]: [string, unknown]): RedTrackCatalogMember => ({
        id: redMemberId(entry, id),
        entry,
      }),
    );
  }
  throw new Error(
    `${contractId} red gate: the track catalog is neither a list of variants nor a keyed map of variants — a named track cannot be resolved from it`,
  );
}

/**
 * Unwraps a resolved value into the track spec it carries.
 *
 * @param value - The resolved value (the spec itself or a wrapper around it).
 * @returns The carried spec, or null when the value carries none.
 */
function redUnwrapSpec(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const nested = record['spec'];
  if (typeof nested === 'object' && nested !== null && !Array.isArray(nested)) {
    return nested as Record<string, unknown>;
  }
  return record;
}

/**
 * Tells a resolved track spec from a generation input.
 *
 * @param value - The resolved value.
 * @returns True when the value carries the generator's geometry.
 */
function redLooksLikeTrackSpec(value: Record<string, unknown>): boolean {
  return (
    Array.isArray(value['segments']) && Array.isArray(value['splineSamples'])
  );
}

/**
 * Fingerprints the DETERMINISM KEY plus the geometry of a track spec.
 *
 * Educational note — why the fingerprint, not the raw object. The catalog
 * may hand a spec back wrapped (with its id beside it) and the generator
 * may add fields over time; the fingerprint reads only the fields that
 * decide the geometry, so two resolutions of the same variant are compared
 * on what they MEAN and not on how they are packaged.
 *
 * @param spec - The resolved track spec.
 * @returns The canonical JSON fingerprint of the spec's geometry.
 */
function redGeometryFingerprint(spec: Record<string, unknown>): string {
  return JSON.stringify({
    seed: spec['seed'],
    layoutVersion: spec['layoutVersion'],
    sizeBucket: spec['sizeBucket'],
    segments: spec['segments'],
    splineSamples: spec['splineSamples'],
    pitBoxes: spec['pitBoxes'],
  });
}

/**
 * Lists the exported functions that may resolve a variant id into its spec.
 *
 * Educational note — the capability probe. The catalog owns the resolver's
 * NAME; the contract only needs a function that answers a variant id with
 * the geometry the generator produced for it. The named candidates come
 * first so the obvious resolver wins over an unrelated helper.
 *
 * @param catalogExports - The catalog module's exported surface.
 * @returns The resolver candidates, in probe order.
 */
function redSpecResolverCandidates(
  catalogExports: Readonly<Record<string, unknown>>,
): Array<(argument: unknown) => unknown> {
  const functions = Object.entries(catalogExports).filter(
    ([, value]) => typeof value === 'function',
  );
  const named = functions.filter(([name]) => /resolve/i.test(name));
  const rest = functions.filter(([name]) => !/resolve/i.test(name));
  return [...named, ...rest].map(
    ([, value]) => value as (argument: unknown) => unknown,
  );
}

/**
 * Resolves a named variant into its track spec.
 *
 * @param context - The gated catalog context.
 * @param trackId - The variant id to resolve.
 * @param contractId - The RED contract the helper speaks for.
 * @returns The resolved track spec.
 * @throws When no resolver answers with the generator's geometry.
 */
function redResolveMemberSpec(
  context: RedTrackCatalogContext,
  trackId: string,
  contractId: string,
): Record<string, unknown> {
  for (const candidate of redSpecResolverCandidates(context.surface.exports)) {
    for (const argument of [trackId, { trackId, id: trackId }]) {
      let resolved: unknown;
      try {
        resolved = candidate(argument);
      } catch {
        continue;
      }
      const spec = redUnwrapSpec(resolved);
      if (spec !== null && redLooksLikeTrackSpec(spec)) {
        return spec;
      }
    }
  }
  const member = context.members.find((entry) => entry.id === trackId);
  const memberSpec = redUnwrapSpec(member?.entry);
  if (memberSpec !== null && redLooksLikeTrackSpec(memberSpec)) {
    return memberSpec;
  }
  throw new Error(
    `${contractId} red gate: the track catalog resolves no track spec for \`${trackId}\` — every variant must resolve to the geometry the deterministic generator produced for it`,
  );
}

/**
 * Reads the track id a notify payload carries.
 *
 * @param payload - The payload the selection surface notified with.
 * @returns The selected track id, or a readable marker when it is absent.
 */
function redPayloadTrackId(payload: unknown): string {
  if (typeof payload === 'string') {
    return payload;
  }
  if (typeof payload === 'object' && payload !== null) {
    const record = payload as Record<string, unknown>;
    for (const field of RED_SELECTION_TRACK_ID_FIELDS) {
      const value = record[field];
      if (typeof value === 'string' && value.length > 0) {
        return value;
      }
    }
  }
  return '<unreadable-track-id>';
}

/**
 * Finds a callable member by name.
 *
 * @param record - The object to read.
 * @param names - The accepted member names, in probe order.
 * @returns The callable member, or null when the object carries none.
 */
function redFindCallable(
  record: Record<string, unknown>,
  names: readonly string[],
): ((...args: unknown[]) => unknown) | null {
  for (const name of names) {
    const candidate = record[name];
    if (typeof candidate === 'function') {
      return candidate as (...args: unknown[]) => unknown;
    }
  }
  return null;
}

/**
 * Finds a callable export by name.
 *
 * @param moduleExports - The module's exported surface.
 * @param names - The accepted export names, in probe order.
 * @returns The callable export, or null when the module carries none.
 */
function redFindModuleCallable(
  moduleExports: Readonly<Record<string, unknown>>,
  names: readonly string[],
): ((...args: unknown[]) => unknown) | null {
  for (const name of names) {
    const candidate = moduleExports[name];
    if (typeof candidate === 'function') {
      return candidate as (...args: unknown[]) => unknown;
    }
  }
  return null;
}

/**
 * Reads the selected track id a selection surface exposes.
 *
 * @param record - The selection surface.
 * @returns The selected track id, or null when the seam hides it.
 */
function redReadSelectedTrackId(
  record: Record<string, unknown>,
): string | null {
  for (const field of RED_SELECTION_TRACK_ID_FIELDS) {
    const value = record[field];
    if (typeof value === 'string' && value.length > 0) {
      return value;
    }
  }
  const getter = redFindCallable(record, [
    'getTrackId',
    'getSelectedTrackId',
    'trackId',
  ]);
  if (getter !== null) {
    const value = getter.call(record);
    if (typeof value === 'string' && value.length > 0) {
      return value;
    }
  }
  return null;
}

/**
 * Discovers the DOM-free selection surface inside the selection module.
 *
 * Educational note — the capability probe. The selection seam owns its
 * NAMES; the contract needs a surface that (a) switches the selection to a
 * named track and (b) notifies a subscriber — the notify/restart contract
 * the race-mode selector already carries. A factory that answers with such
 * a surface is preferred; a module-level switch/subscribe pair is the
 * fallback.
 *
 * @param moduleExports - The selection module's exported surface.
 * @param catalog - The catalog the selection works over.
 * @returns The discovered selection surface, or null when there is none.
 */
function redDiscoverSelectionSurface(
  moduleExports: Readonly<Record<string, unknown>>,
  catalog: unknown,
): RedTrackSelectionSurface | null {
  const moduleFunctions = Object.entries(moduleExports).filter(
    ([, value]) => typeof value === 'function',
  );
  for (const [, value] of moduleFunctions) {
    const factory = value as (argument?: unknown) => unknown;
    for (const argument of [catalog, undefined]) {
      let produced: unknown;
      try {
        produced = factory(argument);
      } catch {
        continue;
      }
      if (typeof produced !== 'object' || produced === null) {
        continue;
      }
      const record = produced as Record<string, unknown>;
      const switchCall = redFindCallable(
        record,
        RED_SELECTION_SWITCH_METHOD_NAMES,
      );
      if (switchCall === null) {
        continue;
      }
      const subscribeCall =
        redFindCallable(record, RED_SELECTION_SUBSCRIBE_METHOD_NAMES) ??
        redFindModuleCallable(
          moduleExports,
          RED_SELECTION_SUBSCRIBE_METHOD_NAMES,
        );
      return {
        switchTo: (trackId: string): void => {
          switchCall.call(record, trackId);
        },
        subscribe: (listener: (payload: unknown) => void): void => {
          if (subscribeCall !== null) {
            subscribeCall.call(record, listener);
          }
        },
        currentTrackId: (): string | null => redReadSelectedTrackId(record),
      };
    }
  }
  const moduleSwitch = redFindModuleCallable(
    moduleExports,
    RED_SELECTION_SWITCH_METHOD_NAMES,
  );
  const moduleSubscribe = redFindModuleCallable(
    moduleExports,
    RED_SELECTION_SUBSCRIBE_METHOD_NAMES,
  );
  if (moduleSwitch !== null && moduleSubscribe !== null) {
    return {
      switchTo: (trackId: string): void => {
        moduleSwitch(trackId);
      },
      subscribe: (listener: (payload: unknown) => void): void => {
        moduleSubscribe(listener);
      },
      currentTrackId: (): string | null => null,
    };
  }
  return null;
}

/**
 * Gates the deterministic track catalog.
 *
 * Educational note — the red gate. At HEAD the file
 * `examples/neatRacing/track/track.catalog.ts` does not exist, so the
 * import throws and the gate answers the NAMED missing-behaviour error. At
 * GREEN the import resolves and the pinned constants are read.
 *
 * @param contractId - The RED contract the gate speaks for.
 * @returns The gated catalog context.
 * @throws When the catalog module or its pinned constants are absent.
 */
async function gateRedTrackCatalog(
  contractId: string,
): Promise<RedTrackCatalogContext> {
  const catalogModule = await loadRedModule(RED_TRACK_CATALOG_MODULE_PATH);
  const catalog = catalogModule?.['NEATRACING_DEMO_TRACK_CATALOG'];
  const variantCount = catalogModule?.['NEATRACING_DEMO_TRACK_VARIANT_COUNT'];
  if (
    catalogModule === undefined ||
    catalog === undefined ||
    typeof variantCount !== 'number'
  ) {
    throw new Error(
      `${contractId} red gate: the deterministic track catalog (NEATRACING_DEMO_TRACK_CATALOG + NEATRACING_DEMO_TRACK_VARIANT_COUNT) does not exist yet at examples/neatRacing/track/track.catalog.ts — the demo still runs the ONE pinned track \`${RED_PINNED_DEMO_TRACK_ID}\`, so no second circuit can be selected and no second race can be replayed`,
    );
  }
  const members = redCatalogMembers(catalog, contractId);
  return {
    surface: { catalog, variantCount, exports: catalogModule },
    members,
  };
}

/**
 * Gates the DOM-free host track-selection surface.
 *
 * Educational note — the red gate. The selection module is the surface
 * that carries the notify/restart contract for a track switch; at HEAD it
 * does not exist, so the gate answers the NAMED missing-behaviour error.
 *
 * @param contractId - The RED contract the gate speaks for.
 * @returns The gated selection context (the catalog + the selection).
 * @throws When the selection module or its switch/subscribe surface is absent.
 */
async function gateRedTrackSelection(
  contractId: string,
): Promise<RedTrackSelectionContext> {
  const selectorModule = await loadRedModule(RED_TRACK_SELECTOR_MODULE_PATH);
  if (selectorModule === undefined) {
    throw new Error(
      `${contractId} red gate: the host track-selection seam does not exist yet at examples/neatRacing/host/track.selector.ts — the demo still runs the ONE pinned track \`${RED_PINNED_DEMO_TRACK_ID}\`, so a track switch can never restart the race on another circuit`,
    );
  }
  const context = await gateRedTrackCatalog(contractId);
  const selection = redDiscoverSelectionSurface(
    selectorModule,
    context.surface.catalog,
  );
  if (selection === null) {
    throw new Error(
      `${contractId} red gate: the host track-selection seam exposes no switch + subscribe surface — a track switch must notify the restart hook exactly once with the SELECTED track id, and the seam must expose a DOM-free core so this contract needs no browser environment`,
    );
  }
  return { context, selection };
}

/**
 * Reads the catalog member ids.
 *
 * @param context - The gated catalog context.
 * @returns The catalog's member ids, in catalog order.
 */
function redMemberIds(context: RedTrackCatalogContext): string[] {
  return context.members.map((member) => member.id);
}

/**
 * Reads the id of the SECOND catalog member (the switch target).
 *
 * @param context - The gated catalog context.
 * @param contractId - The RED contract the helper speaks for.
 * @returns The second member's track id.
 * @throws When the catalog carries no second track.
 */
function redSecondTrackId(
  context: RedTrackCatalogContext,
  contractId: string,
): string {
  const other = context.members.find(
    (member) => member.id !== RED_PINNED_DEMO_TRACK_ID,
  );
  if (other === undefined) {
    throw new Error(
      `${contractId} red gate: the track catalog carries no second track beside the pinned demo loop — a selection switch needs a second circuit to switch TO`,
    );
  }
  return other.id;
}

/**
 * Builds the reproducibility tuple the round would boot for a track.
 *
 * Educational note — the tuple witness. The host boots every round with
 * the AC-026 tuple (seed + agent count + pack schema version + opponent
 * snapshot + track id + feature flags); the seed comes from the resolved
 * track spec and the track id from the SELECTION. The contract builds the
 * same tuple so two tracks can be compared on the canonical serial the
 * worker replays from.
 *
 * @param context - The gated catalog context.
 * @param trackId - The selected track id.
 * @param contractId - The RED contract the helper speaks for.
 * @returns The round's reproducibility tuple for the selected track.
 */
function redRoundTuple(
  context: RedTrackCatalogContext,
  trackId: string,
  contractId: string,
): NeatRacingReproducibilityTuple {
  const spec = redResolveMemberSpec(context, trackId, contractId);
  const seed = spec['seed'];
  return {
    seed: typeof seed === 'number' && Number.isFinite(seed) ? seed : 0,
    agentCount: 1,
    packSchemaVersion: NEATRACING_PACK_SCHEMA_VERSION,
    opponentSnapshot: { level: 1, carCount: 1 },
    trackId,
    featureFlags: {},
  };
}

/**
 * Serializes the reproducibility tuple the round would boot for a track.
 *
 * @param context - The gated catalog context.
 * @param trackId - The selected track id.
 * @param contractId - The RED contract the helper speaks for.
 * @returns The canonical serial of the round's reproducibility tuple.
 */
function redRoundTupleSerial(
  context: RedTrackCatalogContext,
  trackId: string,
  contractId: string,
): string {
  return serializeNeatRacingReproducibilityTuple(
    redRoundTuple(context, trackId, contractId),
  );
}

/* ------------------------------------------------------------------------- *
 * RED-P4-7 — the deterministic track catalog.
 * ------------------------------------------------------------------------- */

describe('RED-P4-7 — the deterministic track catalog', () => {
  it('declares at least four catalog variants', async () => {
    // Arrange: the catalog behind the red gate (it does not exist yet).
    const context = await gateRedTrackCatalog('RED-P4-7');

    // Act: the declared variant count.

    // Assert: the catalog declares at least four variants (AC-100).
    expect(context.surface.variantCount).toBeGreaterThanOrEqual(4);
  });

  it('resolves at least four DISTINCT track specs', async () => {
    // Arrange: the catalog behind the red gate (it does not exist yet).
    const context = await gateRedTrackCatalog('RED-P4-7');

    // Act: the resolved geometry of every catalog member.
    const fingerprints = context.members.map((member) =>
      redGeometryFingerprint(
        redResolveMemberSpec(context, member.id, 'RED-P4-7'),
      ),
    );

    // Assert: the catalog holds four genuinely DIFFERENT circuits — a
    // catalog of four names over one geometry would be a fake catalog.
    expect(new Set(fingerprints).size).toBeGreaterThanOrEqual(4);
  });

  it('keeps the declared variant count in step with the catalog size', async () => {
    // Arrange: the catalog behind the red gate (it does not exist yet).
    const context = await gateRedTrackCatalog('RED-P4-7');

    // Act + Assert: the declared count and the catalog's own size agree —
    // a count that lies about the catalog is a stale constant.
    expect(context.surface.variantCount).toBe(context.members.length);
  });

  it('resolves the SAME spec for two identical resolve calls (determinism)', async () => {
    // Arrange: the catalog behind the red gate (it does not exist yet).
    const context = await gateRedTrackCatalog('RED-P4-7');

    // Act: two identical resolve passes over every variant.
    const firstPass = context.members.map((member) =>
      redGeometryFingerprint(
        redResolveMemberSpec(context, member.id, 'RED-P4-7'),
      ),
    );
    const secondPass = context.members.map((member) =>
      redGeometryFingerprint(
        redResolveMemberSpec(context, member.id, 'RED-P4-7'),
      ),
    );

    // Assert: the SAME variant id always resolves to the SAME geometry —
    // the catalog is a deterministic function of its named seed (AC-100).
    expect(secondPass).toEqual(firstPass);
  });

  it('keeps the PINNED demo loop id as a catalog member', async () => {
    // Arrange: the catalog behind the red gate (it does not exist yet).
    const context = await gateRedTrackCatalog('RED-P4-7');

    // Act + Assert: the shipped demo track id stays a member of the
    // catalog — the current boot default must not regress.
    expect(redMemberIds(context)).toContain(RED_PINNED_DEMO_TRACK_ID);
  });

  it('keeps the SHIPPED demo geometry on the pinned demo loop member', async () => {
    // Arrange: the catalog behind the red gate (it does not exist yet).
    const context = await gateRedTrackCatalog('RED-P4-7');

    // Act: the pinned demo loop member's resolved geometry.
    const resolved = redResolveMemberSpec(
      context,
      RED_PINNED_DEMO_TRACK_ID,
      'RED-P4-7',
    );
    const shipped = generateTrack(RED_SHIPPED_DEMO_TRACK_GENERATION);

    // Assert: the pinned member is the SHIPPED demo circuit (seed 42 /
    // layoutVersion 1 / sizeBucket 'large') — a demo boot that silently
    // changed circuit would be a behavioural regression, not a feature.
    expect(redGeometryFingerprint(resolved)).toBe(
      redGeometryFingerprint(shipped as unknown as Record<string, unknown>),
    );
  });
});

/* ------------------------------------------------------------------------- *
 * RED-P4-8 — the selection-restart contract.
 * ------------------------------------------------------------------------- */

describe('RED-P4-8 — the selection-restart contract', () => {
  it('notifies the restart hook exactly once with the SELECTED track id', async () => {
    // Arrange: the selection surface behind the red gate (it does not
    // exist yet) plus the track the switch selects.
    const { context, selection } = await gateRedTrackSelection('RED-P4-8');
    const otherTrackId = redSecondTrackId(context, 'RED-P4-8');
    const notified: string[] = [];
    selection.subscribe((payload) => {
      notified.push(redPayloadTrackId(payload));
    });

    // Act: the user picks the other circuit.
    selection.switchTo(otherTrackId);

    // Assert: the restart hook fires EXACTLY once and carries the
    // SELECTED track id — the notify/restart contract of the existing
    // selector overlay.
    expect(notified).toEqual([otherTrackId]);
  });

  it('restarts in EITHER direction — a switch back notifies the first track again', async () => {
    // Arrange: the selection surface behind the red gate (it does not
    // exist yet) plus the captured restart notifications.
    const { context, selection } = await gateRedTrackSelection('RED-P4-8');
    const otherTrackId = redSecondTrackId(context, 'RED-P4-8');
    const notified: string[] = [];
    selection.subscribe((payload) => {
      notified.push(redPayloadTrackId(payload));
    });

    // Act: the user picks the other circuit and then picks the demo loop
    // back (the switch runs in BOTH directions).
    selection.switchTo(otherTrackId);
    selection.switchTo(RED_PINNED_DEMO_TRACK_ID);

    // Assert: BOTH switches notified — a one-way switch would strand the
    // demo on whichever circuit was picked last.
    expect(notified).toEqual([otherTrackId, RED_PINNED_DEMO_TRACK_ID]);
  });

  it('boots the round tuple on the SELECTED track id', async () => {
    // Arrange: the selection surface behind the red gate plus the round
    // tuple builder over the catalog.
    const { context, selection } = await gateRedTrackSelection('RED-P4-8');
    const otherTrackId = redSecondTrackId(context, 'RED-P4-8');
    const notified: string[] = [];
    selection.subscribe((payload) => {
      notified.push(redPayloadTrackId(payload));
    });

    // Act: the user picks the other circuit and the round boots.
    selection.switchTo(otherTrackId);
    const selectedTrackId = redPayloadTrackId(
      notified[notified.length - 1] as unknown,
    );
    const tuple = redRoundTuple(context, selectedTrackId, 'RED-P4-8');

    // Assert: the round's AC-026 tuple carries the SELECTED track id (the
    // tuple names the geometry it replays).
    expect(tuple.trackId).toBe(otherTrackId);
  });

  it('boots a FRESH reproducibility tuple for the switched track', async () => {
    // Arrange: the selection surface behind the red gate plus the round
    // tuple builder over the catalog.
    const { context, selection } = await gateRedTrackSelection('RED-P4-8');
    const otherTrackId = redSecondTrackId(context, 'RED-P4-8');
    const notified: string[] = [];
    selection.subscribe((payload) => {
      notified.push(redPayloadTrackId(payload));
    });

    // Act: the user picks the other circuit and the round reboots.
    selection.switchTo(otherTrackId);
    const selectedTrackId = redPayloadTrackId(
      notified[notified.length - 1] as unknown,
    );
    const freshTuple = redRoundTupleSerial(
      context,
      selectedTrackId,
      'RED-P4-8',
    );
    const previousTuple = redRoundTupleSerial(
      context,
      RED_PINNED_DEMO_TRACK_ID,
      'RED-P4-8',
    );

    // Assert: the switched round's tuple is FRESH — it is not the tuple of
    // the circuit the race was running before the switch.
    expect(freshTuple).not.toBe(previousTuple);
  });

  it('retains the FIRST track tuple when the selection switches back', async () => {
    // Arrange: the selection surface behind the red gate plus the round
    // tuple builder over the catalog.
    const { context, selection } = await gateRedTrackSelection('RED-P4-8');
    const otherTrackId = redSecondTrackId(context, 'RED-P4-8');
    const notified: string[] = [];
    selection.subscribe((payload) => {
      notified.push(redPayloadTrackId(payload));
    });
    const firstTuple = redRoundTupleSerial(
      context,
      RED_PINNED_DEMO_TRACK_ID,
      'RED-P4-8',
    );

    // Act: the user picks the other circuit and then switches back.
    selection.switchTo(otherTrackId);
    selection.switchTo(RED_PINNED_DEMO_TRACK_ID);
    const backTrackId = redPayloadTrackId(
      notified[notified.length - 1] as unknown,
    );
    const backTuple = redRoundTupleSerial(context, backTrackId, 'RED-P4-8');

    // Assert: the tuple of a track is RETAINED across the switch away and
    // back — the same track replays from the same tuple (AC-026).
    expect(backTuple).toBe(firstTuple);
  });
});
