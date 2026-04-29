import Network from '../../../architecture/network';
import { methods } from '../../../neataptic';
import {
  createGenomeFromNetwork,
  selectGenomeHeredityConnectionGenes,
} from '../genome';

function createRandomSequenceGenerator(sequence: number[]): () => number {
  let nextSequenceIndex = 0;

  return () => {
    const resolvedValue =
      sequence.at(nextSequenceIndex) ?? sequence.at(-1) ?? 0;
    nextSequenceIndex += 1;
    return resolvedValue;
  };
}

describe('genome heredity chapter', () => {
  describe('selectGenomeHeredityConnectionGenes', () => {
    describe('given two homologous parent genomes', () => {
      it('preserves the shared innovation set in sorted order', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_411 });
        firstParentNetwork.mutate(methods.mutation.ADD_NODE);
        const secondParentNetwork = firstParentNetwork.clone();
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 1,
          parent2Score: 1,
          equal: true,
          randomGenerator: () => 0,
        });

        // Assert
        expect(
          selectedGenes
            .map((selectedGene) => selectedGene.connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
        ).toEqual(
          firstParentGenome.connectionGenes
            .map((connectionGene) => connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
        );
      });
    });

    describe('given same-endpoint genes carry different innovations', () => {
      it('keeps the fitter parent innovation lane intact', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_412 });
        const secondParentNetwork = firstParentNetwork.clone();
        firstParentNetwork.connections[0].innovation = 101;
        firstParentNetwork.connections[0].weight = 0.25;
        secondParentNetwork.connections[0].innovation = 202;
        secondParentNetwork.connections[0].weight = 0.75;
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 2,
          parent2Score: 1,
          equal: false,
          randomGenerator: () => 0,
        });

        // Assert
        expect({
          innovations: selectedGenes
            .map((selectedGene) => selectedGene.connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
          inheritedWeight:
            selectedGenes.find(
              (selectedGene) => selectedGene.connectionGene.innovation === 101,
            )?.connectionGene.weight ?? null,
          hasLowerFitnessInnovation: selectedGenes.some(
            (selectedGene) => selectedGene.connectionGene.innovation === 202,
          ),
        }).toEqual({
          innovations: firstParentGenome.connectionGenes
            .map((connectionGene) => connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
          inheritedWeight: 0.25,
          hasLowerFitnessInnovation: false,
        });
      });
    });

    describe('given matching disabled genes', () => {
      it('uses the inherited rng for re-enable decisions', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_413 });
        const secondParentNetwork = firstParentNetwork.clone();
        firstParentNetwork.connections[0].enabled = false;
        secondParentNetwork.connections[0].enabled = false;
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 1,
          parent2Score: 1,
          parent1ReenableProbability: 0.75,
          parent2ReenableProbability: 0.75,
          equal: false,
          randomGenerator: createRandomSequenceGenerator([0.75, 0.5]),
        });

        // Assert
        expect({
          enabled:
            selectedGenes.find(
              (selectedGene) =>
                selectedGene.connectionGene.innovation ===
                firstParentGenome.connectionGenes[0].innovation,
            )?.connectionGene.enabled ?? null,
        }).toEqual({
          enabled: true,
        });
      });

      it('falls back to the second parent re-enable probability when the first is unset', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_416 });
        const secondParentNetwork = firstParentNetwork.clone();
        firstParentNetwork.connections[0].enabled = false;
        secondParentNetwork.connections[0].enabled = false;
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 1,
          parent2Score: 1,
          parent2ReenableProbability: 1,
          equal: false,
          randomGenerator: createRandomSequenceGenerator([0.75, 0]),
        });

        // Assert
        expect(selectedGenes[0].connectionGene.enabled).toBe(true);
      });

      it('uses the default re-enable probability when both probabilities are unset', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_417 });
        const secondParentNetwork = firstParentNetwork.clone();
        firstParentNetwork.connections[0].enabled = false;
        secondParentNetwork.connections[0].enabled = false;
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 1,
          parent2Score: 1,
          equal: false,
          randomGenerator: createRandomSequenceGenerator([0.75, 0.1]),
        });

        // Assert
        expect(selectedGenes[0].connectionGene.enabled).toBe(true);
      });
    });

    describe('given a fitter parent with a disabled disjoint gene', () => {
      it('applies re-enable probability while inheriting the disjoint connection', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_414 });
        const secondParentNetwork = firstParentNetwork.clone();
        firstParentNetwork.connections[0].innovation = 101;
        firstParentNetwork.connections[0].enabled = false;
        secondParentNetwork.connections[0].innovation = 202;
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 2,
          parent2Score: 1,
          parent1ReenableProbability: 1,
          equal: false,
          randomGenerator: () => 0,
        });

        // Assert
        expect(
          selectedGenes.find(
            (selectedGene) => selectedGene.connectionGene.innovation === 101,
          )?.connectionGene.enabled ?? null,
        ).toBe(true);
      });
    });

    describe('given a less fit parent1 with only disjoint genes', () => {
      it('skips parent1-only genes and inherits sorted parent2-only genes', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_415 });
        const secondParentNetwork = firstParentNetwork.clone();
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);
        const sourceTemplateGene = firstParentGenome.connectionGenes[0];
        const controlledParent1Genome = {
          ...firstParentGenome,
          connectionGenes: [
            {
              ...sourceTemplateGene,
              innovation: 101,
            },
          ],
        };
        const controlledParent2Genome = {
          ...secondParentGenome,
          connectionGenes: [
            {
              ...sourceTemplateGene,
              innovation: 303,
            },
            {
              ...sourceTemplateGene,
              innovation: 202,
            },
          ],
        };

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: controlledParent1Genome,
          parent2Genome: controlledParent2Genome,
          parent1Score: 1,
          parent2Score: 2,
          equal: false,
          randomGenerator: () => 0,
        });

        // Assert
        expect({
          innovations: selectedGenes.map(
            (selectedGene) => selectedGene.connectionGene.innovation,
          ),
          allFromParent2: selectedGenes.every(
            (selectedGene) => selectedGene.sourceParent === 'parent2',
          ),
          includesParent1Innovation: selectedGenes.some(
            (selectedGene) => selectedGene.connectionGene.innovation === 101,
          ),
        }).toEqual({
          innovations: [202, 303],
          allFromParent2: true,
          includesParent1Innovation: false,
        });
      });
    });
  });
});
