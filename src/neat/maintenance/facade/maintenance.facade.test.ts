import Connection from '../../../architecture/connection';
import Neat from '../../../neat';

describe('neat maintenance facade chapter', () => {
  describe('ensureNoDeadEnds', () => {
    describe('given a network whose visible endpoints have all lost connectivity', () => {
      it('repairs the stranded graph so hidden nodes keep both inbound and outbound links', async () => {
        // Arrange
        const neat = new Neat(2, 1, () => 1, {
          popsize: 1,
          seed: 3,
          minHidden: 1,
          speciation: false,
        });
        await neat.evaluate();
        const network = neat.population[0];
        [...network.connections].forEach((connection: Connection) =>
          network.disconnect(connection.from, connection.to),
        );
        const hiddenNode = network.nodes.find((node) => node.type === 'hidden');

        // Act
        neat.ensureNoDeadEnds(network);

        // Assert
        expect(
          network.connections.length > 0 &&
            (!hiddenNode ||
              (hiddenNode.connections.in.length > 0 &&
                hiddenNode.connections.out.length > 0)),
        ).toBe(true);
      });

      it('records repair-created connections in the innovation tracker', async () => {
        // Arrange
        const neat = new Neat(2, 1, () => 1, {
          popsize: 1,
          seed: 4,
          minHidden: 1,
          speciation: false,
        });
        await neat.evaluate();
        const network = neat.population[0];
        [...network.connections].forEach((connection: Connection) =>
          network.disconnect(connection.from, connection.to),
        );

        // Act
        neat.ensureNoDeadEnds(network);

        // Assert
        expect(
          neat.toJSON().innovationTracker.connectionInnovations.length > 0,
        ).toBe(true);
      });
    });
  });
});
