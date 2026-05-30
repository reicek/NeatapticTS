import { start } from './index';
import { start as browserEntryStart } from './browser-entry/browser-entry';

describe('racing curriculum index barrel', () => {
  describe('export surface', () => {
    it('re-exports the browser entry start function', () => {
      expect(start).toBe(browserEntryStart);
    });
  });
});