/** @jest-environment jsdom */

import { start } from './browser-entry';

describe('racing curriculum browser entry start()', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('rejects when the requested host container is missing', async () => {
    await expect(start('missing-racing-curriculum-host')).rejects.toThrow(
      'Racing curriculum container "missing-racing-curriculum-host" was not found.',
    );
  });

  it('keeps active panel readiness copy free of future-facing and deferred placeholders', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(
      hostElement.textContent?.includes('Future-facing') ||
        hostElement.textContent?.includes('Deferred'),
    ).toBe(false);
  });

  it('renders a live network canvas in the focused network panel', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(
      hostElement.querySelector(
        '.racing-host__region--network canvas.racing-network-canvas',
      ),
    ).not.toBeNull();
  });
});
