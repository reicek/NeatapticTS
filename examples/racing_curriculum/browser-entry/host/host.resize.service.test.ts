/** @jest-environment jsdom */

import { installRacingNetworkResize } from './host.resize.service';

describe('installRacingNetworkResize', () => {
  it('calls onResize and changes the network canvas backing size after a resize event', () => {
    const networkCanvas = document.createElement('canvas');
    networkCanvas.width = 100;
    networkCanvas.height = 80;

    const networkCanvasHost = document.createElement('div');
    networkCanvasHost.style.width = '400px';
    networkCanvasHost.style.height = '300px';
    document.body.append(networkCanvasHost);

    const onResize = jest.fn();
    const resizeHandle = installRacingNetworkResize(
      networkCanvas,
      networkCanvasHost,
      onResize,
    );

    const initialWidth = networkCanvas.width;
    const initialHeight = networkCanvas.height;

    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: window.innerWidth + 200,
    });
    window.dispatchEvent(new Event('resize'));

    resizeHandle.uninstall();

    expect(
      onResize.mock.calls.length > 0 &&
        (networkCanvas.width !== initialWidth ||
          networkCanvas.height !== initialHeight),
    ).toBe(true);
  });
});
