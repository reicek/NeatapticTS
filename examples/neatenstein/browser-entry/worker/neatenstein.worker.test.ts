/**
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';

describe('Neatenstein worker shim', () => {
  it('imports the thin worker entrypoint without error', async () => {
    let messageHandler: ((event: MessageEvent) => unknown) | null = null;

    const fakeSelf = {
      get onmessage() {
        return messageHandler;
      },
      set onmessage(value) {
        messageHandler = value as ((event: MessageEvent) => unknown) | null;
      },
      location: { href: 'http://localhost/' } as Location,
    } as unknown as WorkerGlobalScope & typeof globalThis;

    (globalThis as unknown as { self: unknown }).self = fakeSelf;

    await import('./neatenstein.worker');

    expect(messageHandler).toEqual(expect.any(Function));
  });
});
