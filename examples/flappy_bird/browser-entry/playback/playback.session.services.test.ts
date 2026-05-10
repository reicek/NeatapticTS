/** @jest-environment jsdom */

import { initializePlaybackSessionContext } from './playback.session.services';

describe('initializePlaybackSessionContext', () => {
  it('uses the fixed Flappy world viewport when the canvas is still at its 1x1 startup placeholder', () => {
    const evolutionWorker = {
      postMessage: jest.fn(),
    } as unknown as Worker;
    const canvas = document.createElement('canvas');

    canvas.width = 1;
    canvas.height = 1;
    const sessionContext = initializePlaybackSessionContext(
      canvas,
      evolutionWorker,
    );

    expect({
      postedMessages: (evolutionWorker.postMessage as jest.Mock).mock.calls,
      renderViewport: {
        visibleWorldWidthPx: sessionContext.renderState.visibleWorldWidthPx,
        visibleWorldHeightPx: sessionContext.renderState.visibleWorldHeightPx,
      },
    }).toEqual({
      postedMessages: [
        [
          {
            type: 'start-playback',
            payload: {
              visibleWorldWidthPx: 288,
              visibleWorldHeightPx: 512,
            },
          },
        ],
      ],
      renderViewport: {
        visibleWorldWidthPx: 288,
        visibleWorldHeightPx: 512,
      },
    });
  });
});
