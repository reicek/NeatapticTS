import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  resolveCliOptions,
  loadTrace,
  detectTraceFormat,
  normalizeTraceFile,
} from './analyze-trace.io';

describe('analyze-trace.io', () => {
  describe('detectTraceFormat', () => {
    it('returns standard for a bare array', () => {
      expect(detectTraceFormat([])).toBe('standard');
    });

    it('returns devtools when result property exists', () => {
      expect(detectTraceFormat({ result: [] })).toBe('devtools');
    });

    it('returns standard for an object without result', () => {
      expect(detectTraceFormat({ traceEvents: [] })).toBe('standard');
    });

    it('returns standard for null', () => {
      expect(detectTraceFormat(null)).toBe('standard');
    });

    it('returns standard for a primitive number', () => {
      expect(detectTraceFormat(42)).toBe('standard');
    });
  });

  describe('normalizeTraceFile', () => {
    it('wraps a bare array into traceEvents', () => {
      const events = [{ ph: 'X', name: 'Foo', ts: 0, dur: 10 }];
      expect(normalizeTraceFile(events)).toEqual({ traceEvents: events });
    });

    it('returns standard trace objects as-is', () => {
      const standard = {
        traceEvents: [{ ph: 'X', name: 'Foo', ts: 0, dur: 10 }],
      };
      expect(normalizeTraceFile(standard)).toBe(standard);
    });

    it('normalizes MCP result array to traceEvents', () => {
      const events = [{ ph: 'X', name: 'Foo', ts: 0, dur: 10 }];
      expect(normalizeTraceFile({ result: events })).toEqual({
        traceEvents: events,
      });
    });

    it('normalizes MCP result.traceEvents to traceEvents', () => {
      const events = [{ ph: 'X', name: 'Foo', ts: 0, dur: 10 }];
      expect(normalizeTraceFile({ result: { traceEvents: events } })).toEqual({
        traceEvents: events,
      });
    });

    it('returns undefined traceEvents when MCP result has no traceEvents', () => {
      const normalized = normalizeTraceFile({ result: {} });
      expect(normalized.traceEvents).toBeUndefined();
    });

    it('returns undefined traceEvents when MCP result is null', () => {
      const normalized = normalizeTraceFile({ result: null });
      expect(normalized.traceEvents).toBeUndefined();
    });
  });

  describe('loadTrace', () => {
    let tempDir: string;

    beforeEach(async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'trace-io-test-'));
    });

    afterEach(async () => {
      await rm(tempDir, { recursive: true, force: true });
    });

    it('loads a standard trace file with traceEvents', async () => {
      const filePath = path.join(tempDir, 'standard.json');
      const events = [{ ph: 'X', name: 'RunTask', ts: 0, dur: 100 }];
      await writeFile(filePath, JSON.stringify({ traceEvents: events }));
      expect(await loadTrace(filePath)).toEqual({ traceEvents: events });
    });

    it('loads and normalizes a Chrome DevTools MCP trace file', async () => {
      const filePath = path.join(tempDir, 'mcp.json');
      const events = [{ ph: 'X', name: 'RunTask', ts: 0, dur: 100 }];
      await writeFile(
        filePath,
        JSON.stringify({ result: { traceEvents: events } }),
      );
      expect(await loadTrace(filePath)).toEqual({ traceEvents: events });
    });

    it('loads and normalizes a bare array trace file', async () => {
      const filePath = path.join(tempDir, 'bare.json');
      const events = [{ ph: 'X', name: 'RunTask', ts: 0, dur: 100 }];
      await writeFile(filePath, JSON.stringify(events));
      expect(await loadTrace(filePath)).toEqual({ traceEvents: events });
    });
  });

  describe('resolveCliOptions', () => {
    it('parses inline --top=N flag', () => {
      expect(resolveCliOptions(['trace.json', '--top=5'])).toEqual({
        tracePath: 'trace.json',
        topCount: 5,
      });
    });

    it('parses split --top N flag', () => {
      expect(resolveCliOptions(['trace.json', '--top', '10'])).toEqual({
        tracePath: 'trace.json',
        topCount: 10,
      });
    });

    it('uses default top count when no --top flag is given', () => {
      expect(resolveCliOptions(['trace.json'])).toEqual({
        tracePath: 'trace.json',
        topCount: 12,
      });
    });

    it('falls back to default when --top value is not a number', () => {
      expect(resolveCliOptions(['trace.json', '--top=abc'])).toEqual({
        tracePath: 'trace.json',
        topCount: 12,
      });
    });

    it('falls back to default when --top value is negative', () => {
      expect(resolveCliOptions(['trace.json', '--top=-3'])).toEqual({
        tracePath: 'trace.json',
        topCount: 12,
      });
    });

    it('falls back to default when --top has no following argument', () => {
      expect(resolveCliOptions(['trace.json', '--top'])).toEqual({
        tracePath: 'trace.json',
        topCount: 12,
      });
    });

    it('throws when no trace path is provided', () => {
      expect(() => resolveCliOptions([])).toThrow();
    });

    it('finds the path argument after a flag argument', () => {
      expect(resolveCliOptions(['--top=5', 'trace.json'])).toEqual({
        tracePath: 'trace.json',
        topCount: 5,
      });
    });
  });
});
