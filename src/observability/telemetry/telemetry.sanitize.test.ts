/**
 * GREEN home for the telemetry sanitizer module (slice OT-V1-2 of
 * `plans/opentelemetry-slice-telemetry.plans.md`).
 *
 * WHAT THIS HOME DEMANDS: the export boundary is an ALLOW-LIST pass and a
 * DENY-LIST pass, and both are proved here over the REAL module:
 *
 *   - the scalar guard accepts a string, a number and a boolean, and
 *     rejects every non-scalar bag (an object, an array, a null);
 *   - the forbidden-shape guard rejects a non-finite number, a calendar
 *     date carried by a string, and accepts an ordinary number, an ordinary
 *     string and a value that is not a string at all;
 *   - the allow-list pass drops a name the signal does not permit, and the
 *     strict mode turns that drop into a warning;
 *   - the deny pass drops a name the allow-list PERMITS but the policy
 *     forbids, and drops the record keys the exporter reserves for itself;
 *   - a caller-supplied accessor that throws is a WARNING and never an
 *     exception, with and without a warning sink;
 *   - an allow-listed attribute whose VALUE is a non-scalar bag or a
 *     forbidden shape is dropped with a warning, with and without a warning
 *     sink;
 *   - a null or an undefined bag is an empty result and never a throw.
 *
 * THE ANTI-VACUITY LANE (the allow-list injection seam): the deny pass is
 * reachable ONLY for a name that is BOTH allow-listed and forbidden, and
 * every shipped allow-list entry is a `neat.telemetry.*` name that is
 * forbidden by none. A test that injects a forbidden name into the
 * allow-list is therefore the ONLY test that can reach the deny pass - and
 * a deny pass that was DELETED would leave every other lane green. The lane
 * below proves the injection seam really reaches the deny pass (a control
 * lane shows that an injected, NON-forbidden name survives) and that the
 * injected, forbidden name is dropped.
 *
 * WHY NOTHING IS MOCKED: the sanitizer is the privacy boundary itself, and
 * a mocked sanitizer would prove nothing about what the sink really writes.
 *
 * @see plans/opentelemetry-slice-telemetry.plans.md - "Vertical slice A"
 */

import {
  REASON_ATTRIBUTE_ACCESSOR_FAILED,
  REASON_ATTRIBUTE_NOT_ALLOWED,
  REASON_ATTRIBUTE_VALUE_REJECTED,
  RESERVED_RECORD_KEYS,
  carriesForbiddenShape,
  isWritableScalar,
  sanitizeAttributes,
} from './telemetry.sanitize';

describe('OT-V1-2: the telemetry sanitizer', () => {
  describe('the scalar guard', () => {
    it('accepts a string, a number and a boolean', () => {
      expect(isWritableScalar('probe')).toBe(true);
      expect(isWritableScalar(42)).toBe(true);
      expect(isWritableScalar(true)).toBe(true);
    });

    it('rejects every non-scalar bag', () => {
      expect(isWritableScalar({})).toBe(false);
      expect(isWritableScalar([])).toBe(false);
      expect(isWritableScalar(null)).toBe(false);
      expect(isWritableScalar(undefined)).toBe(false);
      expect(isWritableScalar((): void => undefined)).toBe(false);
    });
  });

  describe('the forbidden-shape guard', () => {
    it('rejects a non-finite number', () => {
      expect(carriesForbiddenShape(Number.NaN)).toBe(true);
      expect(carriesForbiddenShape(Number.POSITIVE_INFINITY)).toBe(true);
      expect(carriesForbiddenShape(Number.NEGATIVE_INFINITY)).toBe(true);
    });

    it('accepts an ordinary number', () => {
      expect(carriesForbiddenShape(1.5)).toBe(false);
      expect(carriesForbiddenShape(42)).toBe(false);
    });

    it('rejects a calendar date carried by a string', () => {
      expect(carriesForbiddenShape('2024-01-01')).toBe(true);
      expect(carriesForbiddenShape('2024-01-01T00:00:00Z')).toBe(true);
    });

    it('accepts an ordinary string', () => {
      expect(carriesForbiddenShape('probe')).toBe(false);
    });

    it('accepts a value that is not a string at all', () => {
      expect(carriesForbiddenShape({})).toBe(false);
      expect(carriesForbiddenShape(null)).toBe(false);
      expect(carriesForbiddenShape(undefined)).toBe(false);
    });
  });

  describe('the allow-list pass', () => {
    it('drops every attribute when no allow-list is given at all', () => {
      const result = sanitizeAttributes({ 'neat.telemetry.signal': 'trace' });
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual(['neat.telemetry.signal']);
    });

    it('keeps an attribute the signal permits and drops the rest', () => {
      const result = sanitizeAttributes(
        { 'neat.telemetry.signal': 'trace', 'not.allowed': 'x' },
        { signal: 'trace' },
      );
      expect(result.attributes).toEqual({ 'neat.telemetry.signal': 'trace' });
      expect(result.dropped).toEqual(['not.allowed']);
    });

    it('keeps an attribute the explicit allow-list permits', () => {
      const result = sanitizeAttributes(
        { 'neat.telemetry.signal': 'trace', 'not.allowed': 'x' },
        { allowedAttributeNames: ['neat.telemetry.signal'] },
      );
      expect(result.attributes).toEqual({ 'neat.telemetry.signal': 'trace' });
      expect(result.dropped).toEqual(['not.allowed']);
    });

    it('warns with the machine-stable token in the strict mode', () => {
      const warnings: string[] = [];
      const result = sanitizeAttributes(
        { 'not.allowed': 'x' },
        {
          allowedAttributeNames: [],
          strict: true,
          warn: (message: string) => {
            warnings.push(message);
          },
        },
      );
      expect(result.dropped).toEqual(['not.allowed']);
      expect(warnings).toEqual([REASON_ATTRIBUTE_NOT_ALLOWED]);
    });

    it('drops silently in the strict mode when no warning sink was given', () => {
      const result = sanitizeAttributes(
        { 'not.allowed': 'x' },
        { allowedAttributeNames: [], strict: true },
      );
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual(['not.allowed']);
    });
  });

  describe('the deny pass', () => {
    it('drops a forbidden key that the allow-list permits', () => {
      const result = sanitizeAttributes(
        { timestamp: 'redacted' },
        { allowedAttributeNames: ['timestamp'] },
      );
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual(['timestamp']);
    });

    it('drops the record keys the exporter reserves for itself', () => {
      expect([...RESERVED_RECORD_KEYS]).toEqual(['name', 'duration_ms']);
      const result = sanitizeAttributes(
        { name: 'spoofed', duration_ms: 1 },
        { allowedAttributeNames: ['name', 'duration_ms'] },
      );
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual(['name', 'duration_ms']);
    });
  });

  describe('the caller-supplied accessor that throws', () => {
    it('warns with the machine-stable token and keeps going', () => {
      const attributes: Record<string, unknown> = {};
      Object.defineProperty(attributes, 'neat.telemetry.signal', {
        enumerable: true,
        get() {
          throw new Error('attribute accessor exploded');
        },
      });
      const warnings: string[] = [];
      const result = sanitizeAttributes(attributes, {
        allowedAttributeNames: ['neat.telemetry.signal'],
        warn: (message: string) => {
          warnings.push(message);
        },
      });
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual(['neat.telemetry.signal']);
      expect(warnings).toEqual([REASON_ATTRIBUTE_ACCESSOR_FAILED]);
    });

    it('drops the attribute and throws nothing when no warning sink was given', () => {
      const attributes: Record<string, unknown> = {};
      Object.defineProperty(attributes, 'neat.telemetry.signal', {
        enumerable: true,
        get() {
          throw new Error('attribute accessor exploded');
        },
      });
      const result = sanitizeAttributes(attributes, {
        allowedAttributeNames: ['neat.telemetry.signal'],
      });
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual(['neat.telemetry.signal']);
    });
  });

  describe('the allow-listed attribute whose value is rejected', () => {
    it('warns with the machine-stable token for a non-scalar bag and for a forbidden shape', () => {
      const warnings: string[] = [];
      const result = sanitizeAttributes(
        {
          'neat.telemetry.signal': { nested: 'bag' },
          'neat.telemetry.component': '2024-01-01',
        },
        {
          allowedAttributeNames: [
            'neat.telemetry.signal',
            'neat.telemetry.component',
          ],
          warn: (message: string) => {
            warnings.push(message);
          },
        },
      );
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual([
        'neat.telemetry.signal',
        'neat.telemetry.component',
      ]);
      expect(warnings).toEqual([
        REASON_ATTRIBUTE_VALUE_REJECTED,
        REASON_ATTRIBUTE_VALUE_REJECTED,
      ]);
    });

    it('drops the attribute and warns nothing when no warning sink was given', () => {
      const result = sanitizeAttributes(
        { 'neat.telemetry.signal': { nested: 'bag' } },
        { allowedAttributeNames: ['neat.telemetry.signal'] },
      );
      expect(result.attributes).toEqual({});
      expect(result.dropped).toEqual(['neat.telemetry.signal']);
    });
  });

  describe('the empty attribute bag', () => {
    it('returns an empty result for a null bag', () => {
      expect(sanitizeAttributes(null)).toEqual({ attributes: {}, dropped: [] });
    });

    it('returns an empty result for an undefined bag', () => {
      expect(sanitizeAttributes(undefined)).toEqual({
        attributes: {},
        dropped: [],
      });
    });
  });

  describe('the allow-list injection seam (the anti-vacuity lane)', () => {
    it('drops a forbidden name that the allow-list was made to permit', () => {
      // THE CONTROL: an injected name that is NOT forbidden survives the
      // allow-list pass, so the drop asserted below is caused by the DENY
      // pass and by nothing else.
      const control = sanitizeAttributes(
        { 'neat.telemetry.signal': 'trace' },
        { allowedAttributeNames: ['neat.telemetry.signal'] },
      );
      expect(control.attributes).toEqual({ 'neat.telemetry.signal': 'trace' });
      expect(control.dropped).toEqual([]);

      // THE LANE: `timestamp` is injected into the allow-list, so it REACHES
      // the deny pass, and the deny pass drops it. A sanitizer whose deny
      // pass was DELETED would keep the attribute and fail this assertion.
      const injected = sanitizeAttributes(
        { timestamp: 'redacted', 'neat.telemetry.signal': 'trace' },
        { allowedAttributeNames: ['timestamp', 'neat.telemetry.signal'] },
      );
      expect(injected.attributes).not.toHaveProperty('timestamp');
      expect(injected.attributes).toEqual({ 'neat.telemetry.signal': 'trace' });
      expect(injected.dropped).toEqual(['timestamp']);
    });
  });
});
