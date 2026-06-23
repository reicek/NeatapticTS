/**
 * @module trace-compress.test
 * @description Unit tests for the streaming trace compression CLI.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { mkdtemp, rm, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  compressTrace,
  decompressTrace,
  parseCliArgs,
  isCliEntryPoint,
  runCli,
} from './trace-compress.mjs';

describe('trace-compress', () => {
  let tempDir;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), 'trace-compress-'));
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it('round-trips a JSON file through compress and decompress', async () => {
    const input = path.join(tempDir, 'trace.json');
    const compressed = path.join(tempDir, 'trace.json.gz');
    const restored = path.join(tempDir, 'trace.restored.json');
    const payload = JSON.stringify({
      traceEvents: [{ ph: 'X', name: 'A', dur: 100 }],
    });
    await writeFile(input, payload, 'utf8');
    await compressTrace(input, compressed);
    await decompressTrace(compressed, restored);
    const restoredText = await readFile(restored, 'utf8');
    expect(restoredText).toBe(payload);
  });

  it('throws when compress input is missing', async () => {
    const missing = path.join(tempDir, 'nope.json');
    await expect(
      compressTrace(missing, path.join(tempDir, 'out.gz')),
    ).rejects.toThrow(/not found/);
  });

  it('throws when decompress input is missing', async () => {
    const missing = path.join(tempDir, 'nope.gz');
    await expect(
      decompressTrace(missing, path.join(tempDir, 'out.json')),
    ).rejects.toThrow(/not found/);
  });

  it('parses CLI args with a default gzip output path', () => {
    expect(parseCliArgs(['trace.json'])).toEqual({
      input: 'trace.json',
      output: 'trace.json.gz',
    });
  });

  it('parses CLI args with an explicit output path', () => {
    expect(parseCliArgs(['trace.json', 'out.gz'])).toEqual({
      input: 'trace.json',
      output: 'out.gz',
    });
  });

  it('throws when CLI args lack an input path', () => {
    expect(() => parseCliArgs([])).toThrow(/Missing input/);
  });

  it('detects a matching CLI entry point', () => {
    const scriptPath = path.join(tempDir, 'entry.mjs');
    expect(isCliEntryPoint(scriptPath, pathToFileURL(scriptPath).href)).toBe(
      true,
    );
  });

  it('returns false when argv1 is undefined', () => {
    expect(isCliEntryPoint(undefined, 'file:///x')).toBe(false);
  });

  it('returns false when argv1 does not match the entry url', () => {
    const scriptPath = path.join(tempDir, 'entry.mjs');
    const otherPath = path.join(tempDir, 'other.mjs');
    expect(isCliEntryPoint(otherPath, pathToFileURL(scriptPath).href)).toBe(
      false,
    );
  });

  it('streams a large JSON file through a full round-trip', async () => {
    const input = path.join(tempDir, 'large.json');
    const compressed = path.join(tempDir, 'large.json.gz');
    const restored = path.join(tempDir, 'large.restored.json');
    const events = Array.from({ length: 50_000 }, (_, index) => ({
      ph: 'X',
      name: 'E',
      dur: index,
    }));
    const payload = JSON.stringify({ traceEvents: events });
    await writeFile(input, payload, 'utf8');
    await compressTrace(input, compressed);
    await decompressTrace(compressed, restored);
    const restoredText = await readFile(restored, 'utf8');
    expect(restoredText).toBe(payload);
  });

  it('runs the CLI compress path when invoked as the entry point', async () => {
    const input = path.join(tempDir, 'trace.json');
    const output = path.join(tempDir, 'trace.json.gz');
    await writeFile(input, JSON.stringify({ traceEvents: [] }), 'utf8');
    const scriptPath = path.join(tempDir, 'entry.mjs');
    await runCli([input, output], {
      argv1: scriptPath,
      entryUrl: pathToFileURL(scriptPath).href,
    });
    const data = await readFile(output);
    expect(data.length).toBeGreaterThan(0);
  });

  it('no-ops when not invoked as the entry point', async () => {
    await runCli([], { argv1: undefined, entryUrl: 'file:///nope' });
    expect(true).toBe(true);
  });
});
