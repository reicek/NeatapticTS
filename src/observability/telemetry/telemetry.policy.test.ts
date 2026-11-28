/**
 * GREEN home for the telemetry policy module (slice OT-V1-2 of
 * `plans/opentelemetry-slice-telemetry.plans.md`).
 *
 * WHAT THIS HOME DEMANDS: the policy module is the ONE place where the
 * layer-A privacy rules are DECLARED, so every rule it declares is proved
 * here over the REAL module:
 *
 *   - the sink root, the telemetry engine floor and the machine-stable
 *     reason token are the declared values;
 *   - the allow-list gives every signal exactly one repo-owned record name
 *     and repo-owned attribute names, and NO `neat.acceleration` instrument
 *     (a metric instrument built on the acceleration variants would report a
 *     MEASUREMENT OF ZERO for work that was never measured);
 *   - an unknown signal resolves to the EMPTY allow-list entry, so a typo
 *     can never widen the allow-list;
 *   - the signal guard accepts a known signal and rejects an unknown name
 *     AND a value that is not a string at all;
 *   - the forbidden key guard rejects a forbidden key in EVERY spelling
 *     (`startTime`, `start_time`, `START-TIME`) and accepts a key that
 *     carries no instant;
 *   - the forbidden value guard rejects an epoch-millisecond literal
 *     carried by a number AND by a bigint, rejects a calendar date carried
 *     by a string, and accepts an ordinary number, an ordinary bigint, an
 *     ordinary string and a value that is not a string at all;
 *   - the forbidden value patterns are STATELESS (no global flag), so a
 *     reused pattern cannot drift between two calls.
 *
 * WHY NOTHING IS MOCKED: the policy is a pure declaration, and a mocked
 * policy would prove nothing about the rules the sink really applies.
 *
 * @see plans/opentelemetry-slice-telemetry.plans.md - "Vertical slice A"
 */

import {
  FORBIDDEN_KEYS,
  FORBIDDEN_VALUE_PATTERNS,
  REASON_NODE_BELOW_TELEMETRY_FLOOR,
  TELEMETRY_ALLOW_LIST,
  TELEMETRY_NODE_FLOOR,
  TELEMETRY_SIGNALS,
  TELEMETRY_SINK_ROOT,
  getAllowListEntry,
  isForbiddenKey,
  isForbiddenValue,
  isTelemetrySignal,
} from './telemetry.policy';
import type { TelemetrySignal } from './telemetry.policy';

describe('OT-V1-2: the telemetry policy', () => {
  describe('the declared constants', () => {
    it('declares the sink root, the telemetry engine floor and the reason token', () => {
      expect(TELEMETRY_SINK_ROOT).toBe('logs/telemetry');
      expect(TELEMETRY_NODE_FLOOR).toBe('>= 22.15.0');
      expect(REASON_NODE_BELOW_TELEMETRY_FLOOR).toBe(
        'node_below_telemetry_floor',
      );
    });

    it('declares exactly the three signals, in the canonical order', () => {
      expect([...TELEMETRY_SIGNALS]).toEqual(['trace', 'metric', 'log']);
    });

    it('declares a forbidden key set that names every instant the layer drops', () => {
      expect(FORBIDDEN_KEYS).toContain('timestamp');
      expect(FORBIDDEN_KEYS).toContain('startTime');
      expect(FORBIDDEN_KEYS).toContain('endTime');
      expect(FORBIDDEN_KEYS).toContain('observedTimestamp');
      expect(FORBIDDEN_KEYS).toContain('writtenAt');
      // `duration_ms` is the ONE duration field the exporter writes, so it
      // may never appear in the forbidden set.
      expect(FORBIDDEN_KEYS).not.toContain('duration_ms');
    });
  });

  describe('the allow-list', () => {
    it('gives every signal one record name and repo-owned attribute names', () => {
      for (const signal of TELEMETRY_SIGNALS) {
        const entry = TELEMETRY_ALLOW_LIST[signal];
        expect(entry.recordNames).toHaveLength(1);
        expect(entry.attributeNames.length).toBeGreaterThan(0);
        for (const name of [...entry.recordNames, ...entry.attributeNames]) {
          expect(name.startsWith('neat.telemetry.')).toBe(true);
        }
      }
    });

    it('carries NO neat.acceleration instrument in the metric allow-list', () => {
      const metricNames = [
        ...TELEMETRY_ALLOW_LIST.metric.recordNames,
        ...TELEMETRY_ALLOW_LIST.metric.attributeNames,
      ];
      expect(metricNames).not.toContain('neat.acceleration');
      for (const name of metricNames) {
        expect(name).not.toContain('acceleration');
      }
    });

    it('resolves a known signal to its own entry', () => {
      expect(getAllowListEntry('metric')).toBe(TELEMETRY_ALLOW_LIST.metric);
      expect(getAllowListEntry('trace').recordNames).toEqual([
        'neat.telemetry.trace',
      ]);
    });

    it('resolves an unknown signal to the EMPTY entry, so a typo cannot widen the allow-list', () => {
      const unknown = getAllowListEntry('span' as unknown as TelemetrySignal);
      expect(unknown.recordNames).toEqual([]);
      expect(unknown.attributeNames).toEqual([]);
    });
  });

  describe('the signal guard', () => {
    it('accepts every known signal', () => {
      for (const signal of TELEMETRY_SIGNALS) {
        expect(isTelemetrySignal(signal)).toBe(true);
      }
    });

    it('rejects a name that is not a signal', () => {
      expect(isTelemetrySignal('span')).toBe(false);
      expect(isTelemetrySignal('')).toBe(false);
      expect(isTelemetrySignal('TRACE')).toBe(false);
    });

    it('rejects a value that is not a string at all', () => {
      expect(isTelemetrySignal(42)).toBe(false);
      expect(isTelemetrySignal(null)).toBe(false);
      expect(isTelemetrySignal(undefined)).toBe(false);
      expect(isTelemetrySignal(['trace'])).toBe(false);
    });
  });

  describe('the forbidden key guard', () => {
    it('rejects a forbidden key in every spelling', () => {
      expect(isForbiddenKey('startTime')).toBe(true);
      expect(isForbiddenKey('start_time')).toBe(true);
      expect(isForbiddenKey('START-TIME')).toBe(true);
      expect(isForbiddenKey('timestamp')).toBe(true);
    });

    it('accepts a key that carries no instant', () => {
      expect(isForbiddenKey('duration_ms')).toBe(false);
      expect(isForbiddenKey('neat.telemetry.signal')).toBe(false);
      expect(isForbiddenKey('name')).toBe(false);
    });
  });

  describe('the forbidden value guard', () => {
    it('rejects an epoch-millisecond literal carried by a number', () => {
      expect(isForbiddenValue(1764147453321)).toBe(true);
    });

    it('rejects an epoch-millisecond literal carried by a bigint', () => {
      expect(isForbiddenValue(1764147453321n)).toBe(true);
    });

    it('accepts an ordinary number and an ordinary bigint', () => {
      expect(isForbiddenValue(42)).toBe(false);
      expect(isForbiddenValue(42n)).toBe(false);
    });

    it('rejects a calendar date carried by a string', () => {
      expect(isForbiddenValue('2024-01-01')).toBe(true);
      expect(isForbiddenValue('2024-01-01T00:00:00Z')).toBe(true);
    });

    it('accepts an ordinary string', () => {
      expect(isForbiddenValue('probe')).toBe(false);
      expect(isForbiddenValue('neat.telemetry')).toBe(false);
    });

    it('accepts a value that is not a string at all', () => {
      expect(isForbiddenValue({})).toBe(false);
      expect(isForbiddenValue(null)).toBe(false);
      expect(isForbiddenValue(undefined)).toBe(false);
      expect(isForbiddenValue([])).toBe(false);
    });
  });

  describe('the forbidden value patterns', () => {
    it('keeps every pattern stateless, so a reused pattern cannot drift', () => {
      expect(FORBIDDEN_VALUE_PATTERNS.length).toBeGreaterThan(0);
      for (const pattern of FORBIDDEN_VALUE_PATTERNS) {
        expect(pattern.global).toBe(false);
        expect(pattern.test('2024-01-01')).toBe(pattern.test('2024-01-01'));
      }
    });
  });
});
