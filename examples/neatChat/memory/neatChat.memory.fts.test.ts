import { sanitizeFtsQuery } from './neatChat.memory.fts';

describe('neatChat memory FTS utilities', () => {
  describe('sanitizeFtsQuery', () => {
    it('strips FTS5 operator characters and collapses extra whitespace', () => {
      // Arrange
      const rawQuery = 'favorite -(color) "blue" @ocean *weekend^';

      // Act
      const sanitizedQuery = sanitizeFtsQuery(rawQuery);

      // Assert
      expect(sanitizedQuery).toBe('favorite color blue ocean weekend');
    });
  });
});
