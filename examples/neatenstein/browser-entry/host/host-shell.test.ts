import { describe, expect, it } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const HOST_SHELL_PATH = resolve(
  process.cwd(),
  'docs',
  'examples',
  'neatenstein',
  'index.html',
);

function readHostShell(): string {
  return readFileSync(HOST_SHELL_PATH, 'utf-8');
}

describe('Neatenstein host HTML shell', () => {
  describe('AC-022: host shell exists', () => {
    it('exists at docs/examples/neatenstein/index.html', () => {
      const html = readHostShell();
      expect(html.length).toBeGreaterThan(0);
    });
  });

  describe('AC-023: host bundle reference', () => {
    it('loads the host bundle from docs/assets/neatenstein.bundle.js', () => {
      const html = readHostShell();
      expect(html).toContain('neatenstein.bundle.js');
    });
  });

  describe('AC-024: renderer canvas mount', () => {
    it('mounts a canvas element for the renderer', () => {
      const html = readHostShell();
      const hasCanvasTag = /<canvas\b/i.test(html);
      expect(hasCanvasTag).toBe(true);
    });
  });

  describe('AC-025: no legacy module worker reference', () => {
    it('does not reference the unused docs/assets/neatenstein.worker.esm.js asset', () => {
      const html = readHostShell();
      expect(html).not.toContain('neatenstein.worker.esm.js');
    });
  });

  describe('AC-003-RED: docs bundle path resolution', () => {
    it('does not contain an invalid ../../docs/assets/ segment in the docs copy', () => {
      const html = readHostShell();
      expect(html).not.toContain('../../docs/assets/');
    });
  });
});
