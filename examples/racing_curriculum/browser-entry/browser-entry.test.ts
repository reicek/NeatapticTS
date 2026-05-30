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
});
