import { storeExchangeMemorySafely } from '../core/neatChat.session.services';

describe('neatChat session durable memory guard', () => {
  describe('storeExchangeMemorySafely', () => {
    it('does not propagate adapter storage failures', async () => {
      // Arrange
      const adapter = {
        async store() {
          throw new Error('memory store unavailable');
        },
      };

      // Act
      const guardedStoreResult = storeExchangeMemorySafely({
        adapter,
        sessionId: 'session-1',
        userMessage: 'favorite color',
        response: 'blue ocean',
      });

      // Assert
      await expect(guardedStoreResult).resolves.toBeUndefined();
    });
  });
});
