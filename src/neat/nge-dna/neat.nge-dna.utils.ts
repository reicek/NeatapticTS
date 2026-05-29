import { createHash } from 'node:crypto';

import { NGE_DNA_SCHEMA_VERSION } from './neat.nge-dna.constants';
import { NGE_DNA_SchemaError } from './neat.nge-dna.errors';
import type { NgeIdentityFields } from './neat.nge-dna.types';

type CanonicalScalar = boolean | null | number | string;
type CanonicalArray = readonly CanonicalValue[];
type CanonicalValue = CanonicalArray | CanonicalScalar | CanonicalRecord;
type CanonicalRecord = {
  readonly [key: string]: CanonicalValue | undefined;
};

/**
 * Serialize one DNA envelope into a deterministic key-sorted JSON string.
 *
 * @param value - Canonical value to serialize.
 * @returns Stable JSON whose object keys are sorted recursively.
 */
export function canonicalSerialize<TValue extends CanonicalValue | object>(
  value: TValue,
): string {
  return JSON.stringify(sortCanonicalValue(value));
}

/**
 * Compute the SHA-256 fingerprint for one canonical DNA JSON string.
 *
 * @param canonical - Canonical JSON text produced by {@link canonicalSerialize}.
 * @returns Lowercase hexadecimal SHA-256 digest used as the envelope fingerprint.
 */
export function computeFingerprint(canonical: string): string {
  return createHash('sha256').update(canonical).digest('hex');
}

/**
 * Validate the identity shelf of one NGE DNA envelope for schema version and completeness.
 *
 * @param identity - Identity fields to validate.
 * @returns Nothing when the identity is valid.
 * @throws NGE_DNA_SchemaError When required identity fields are missing or the schema version is incompatible.
 */
export function validateIdentity(identity: NgeIdentityFields): void {
  const missingIdentityFieldNames = Object.entries({
    compatibilityVersion: identity.compatibilityVersion,
    encodingMode: identity.encodingMode,
    fingerprint: identity.fingerprint,
    schemaVersion: identity.schemaVersion,
  })
    .filter(([, fieldValue]) => !fieldValue)
    .map(([fieldName]) => fieldName);

  if (missingIdentityFieldNames.length > 0) {
    throw new NGE_DNA_SchemaError(
      `NGE_DNA is missing required identity fields: ${missingIdentityFieldNames.join(', ')}.`,
    );
  }

  if (identity.schemaVersion !== NGE_DNA_SCHEMA_VERSION) {
    throw new NGE_DNA_SchemaError(
      `NGE_DNA schema version ${identity.schemaVersion} is incompatible with ${NGE_DNA_SCHEMA_VERSION}.`,
    );
  }
}

function sortCanonicalValue<TValue extends CanonicalValue | object>(
  value: TValue,
): TValue {
  if (Array.isArray(value)) {
    return value.map((childValue) => sortCanonicalValue(childValue)) as TValue;
  }

  if (typeof value !== 'object' || value === null) {
    return value;
  }

  const sortedEntries = Object.entries(value as CanonicalRecord)
    .filter(([, childValue]) => childValue !== undefined)
    .toSorted(([leftKey], [rightKey]) =>
      compareCanonicalKeys(leftKey, rightKey),
    )
    .map(([key, childValue]) => [
      key,
      sortCanonicalValue(childValue as CanonicalValue),
    ]);

  return Object.fromEntries(sortedEntries) as TValue;
}

function compareCanonicalKeys(leftKey: string, rightKey: string): number {
  return Number(leftKey > rightKey) - Number(leftKey < rightKey);
}
