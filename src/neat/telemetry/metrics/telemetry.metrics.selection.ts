import type { TelemetryEntry } from '../../shared/neat.shared.types';
import type {
  TelemetryCoreFields,
  TelemetrySelectContext,
} from '../types/telemetry.types';

/**
 * Build a snapshot of core telemetry fields from one entry so later selection filtering can preserve required recorder invariants without mutating source state.
 *
 * @param sourceEntry - Source telemetry object.
 * @param fields - Core telemetry field keys to preserve.
 * @returns Shallow snapshot of core fields that exist on the entry.
 */
export function getTelemetryCoreSnapshot(
  sourceEntry: Record<string, unknown>,
  fields: TelemetryCoreFields,
): Partial<Record<string, unknown>> {
  // Step 1: Initialize an empty snapshot for the core fields.
  const coreSnapshot: Partial<Record<string, unknown>> = {};

  // Step 2: Capture only core fields that exist on the source entry.
  for (const field of fields) {
    if (Object.hasOwn(sourceEntry, field)) {
      coreSnapshot[field] = sourceEntry[field];
    }
  }

  // Step 3: Return the shallow snapshot without mutating the source entry.
  return coreSnapshot;
}

/**
 * Remove non-core keys that are not whitelisted by the selection set so telemetry payloads stay compact while preserving recorder-required fields.
 *
 * @param sourceEntry - Telemetry entry being filtered.
 * @param selection - Whitelist of additional telemetry keys.
 * @param fields - Core telemetry field keys that must be preserved.
 * @returns The same entry reference after filtering.
 */
export function stripUnselectedTelemetryKeys(
  sourceEntry: Record<string, unknown>,
  selection: Set<string>,
  fields: TelemetryCoreFields,
): Record<string, unknown> {
  // Step 1: Walk current keys to decide which fields remain.
  for (const key of Object.keys(sourceEntry)) {
    // Step 2: Always keep core fields regardless of selection.
    if (fields.includes(key)) continue;
    // Step 3: Remove non-core keys not present in the selection set.
    if (!selection.has(key)) {
      delete sourceEntry[key];
    }
  }

  // Step 4: Return the same entry reference after filtering.
  return sourceEntry;
}

/**
 * Re-attach core fields to the filtered entry so selection logic never removes mandatory telemetry anchors needed by downstream consumers and audits.
 *
 * @param sourceEntry - Filtered telemetry entry to update.
 * @param coreSnapshot - Snapshot of core fields to ensure presence.
 * @returns The same entry reference with core fields restored.
 */
export function mergeTelemetryCoreFields(
  sourceEntry: Record<string, unknown>,
  coreSnapshot: Partial<Record<string, unknown>>,
): Record<string, unknown> {
  // Step 1: Re-apply core fields to ensure presence and ordering.
  for (const [key, value] of Object.entries(coreSnapshot)) {
    sourceEntry[key] = value;
  }

  // Step 2: Return the same entry reference for call-site chaining.
  return sourceEntry;
}

/**
 * Apply telemetry selection while swallowing selection errors so non-critical projection failures cannot block generation-level telemetry recording in production runs reliably.
 *
 * @param telemetryContext - Neat-like context with telemetry selection.
 * @param telemetryEntry - Entry to filter in place.
 * @param applyTelemetrySelectFn - Selection helper to invoke.
 */
export function safelyApplyTelemetrySelect<
  TContext extends TelemetrySelectContext,
>(
  telemetryContext: TContext,
  telemetryEntry: TelemetryEntry,
  applyTelemetrySelectFn: (
    this: TContext,
    entry: Record<string, unknown>,
  ) => Record<string, unknown>,
): void {
  // Step 1: Attempt to apply selection; ignore failures.
  try {
    applyTelemetrySelectFn.call(
      telemetryContext,
      telemetryEntry as Record<string, unknown>,
    );
  } catch {
    // ignored: telemetry selection errors are non-fatal and safe to drop
  }
}
