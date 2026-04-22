import type Node from '../../node';
import type { StandaloneGenerationContext } from '../network.types';
import {
  ensureActivationFunctionIndex,
  resolveSquashName,
} from './network.standalone.utils.activation';

function createGenerationContext(): StandaloneGenerationContext {
  return {
    standaloneProps: {
      nodes: [],
      input: 0,
      output: 0,
    },
    inputNodeIndexes: [],
    activationNodeIndexes: [],
    outputNodeIndexes: [],
    emittedActivationSource: {},
    activationFunctionSources: [],
    activationFunctionIndexMap: {},
    nextActivationFunctionIndex: 0,
    initialActivations: [],
    initialStates: [],
    bodyLines: [],
  };
}

describe('network standalone activation utility chapter', () => {
  describe('resolveSquashName', () => {
    describe('when the node squash function already has a stable name', () => {
      it('returns that explicit squash name', () => {
        // Arrange
        const currentNode = {
          squash: function logistic(inputValue: number): number {
            return inputValue;
          },
        } as unknown as Node;

        // Act
        const squashName = resolveSquashName(currentNode, 4);

        // Assert
        expect(squashName).toBe('logistic');
      });
    });

    describe('when the node squash function has no stable name', () => {
      it('falls back to an anonymous squash label that includes the traversal index', () => {
        // Arrange
        const currentNode = {
          squash: {},
        } as unknown as Node;

        // Act
        const squashName = resolveSquashName(currentNode, 9);

        // Assert
        expect(squashName).toBe('anonymous_squash_9');
      });
    });
  });

  describe('ensureActivationFunctionIndex', () => {
    describe('when the activation name is already registered', () => {
      it('returns the cached index without mutating the registration tables', () => {
        // Arrange
        const generationContext = createGenerationContext();
        generationContext.activationFunctionIndexMap.relu = 3;
        generationContext.emittedActivationSource.relu =
          'function relu(x){ return x > 0 ? x : 0; }';
        generationContext.activationFunctionSources = [
          'function tanh(x){ return Math.tanh(x); }',
        ];
        generationContext.nextActivationFunctionIndex = 7;

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'relu',
          ((inputValue: number) => inputValue) as (
            inputValue: number,
            derivate?: boolean,
          ) => number,
          1,
        );

        // Assert
        expect({
          activationFunctionSources: generationContext.activationFunctionSources,
          activationIndex,
          emittedSource: generationContext.emittedActivationSource.relu,
          nextActivationFunctionIndex:
            generationContext.nextActivationFunctionIndex,
        }).toEqual({
          activationFunctionSources: [
            'function tanh(x){ return Math.tanh(x); }',
          ],
          activationIndex: 3,
          emittedSource: 'function relu(x){ return x > 0 ? x : 0; }',
          nextActivationFunctionIndex: 7,
        });
      });
    });

    describe('when the squash name matches a built-in activation directly', () => {
      it('registers the built-in snippet at the next function index', () => {
        // Arrange
        const generationContext = createGenerationContext();

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'relu',
          ((inputValue: number) => inputValue) as (
            inputValue: number,
            derivate?: boolean,
          ) => number,
          2,
        );

        // Assert
        expect({
          activationFunctionSources: generationContext.activationFunctionSources,
          activationIndex,
          emittedSource: generationContext.emittedActivationSource.relu,
          nextActivationFunctionIndex:
            generationContext.nextActivationFunctionIndex,
        }).toEqual({
          activationFunctionSources: ['function relu(x){ return x > 0 ? x : 0; }'],
          activationIndex: 0,
          emittedSource: 'function relu(x){ return x > 0 ? x : 0; }',
          nextActivationFunctionIndex: 1,
        });
      });
    });

    describe('when the squash name uses the built-in Activation alias suffix', () => {
      it('reuses the canonical built-in body but renames the emitted function to the requested alias', () => {
        // Arrange
        const generationContext = createGenerationContext();

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'tanhActivation',
          ((inputValue: number) => inputValue) as (
            inputValue: number,
            derivate?: boolean,
          ) => number,
          3,
        );

        // Assert
        expect({
          activationIndex,
          emittedSource:
            generationContext.emittedActivationSource.tanhActivation,
        }).toEqual({
          activationIndex: 0,
          emittedSource:
            'function tanhActivation(x){ return Math.tanh(x); }',
        });
      });
    });

    describe('when the custom squash source is a named function with a different name', () => {
      it('rewrites the emitted declaration to the requested squash name', () => {
        // Arrange
        const generationContext = createGenerationContext();
        const customSquash = {
          toString: () => 'function original(value){ return value + 1; }',
        } as unknown as (inputValue: number, derivate?: boolean) => number;

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'renamed',
          customSquash,
          4,
        );

        // Assert
        expect({
          activationIndex,
          emittedSource: generationContext.emittedActivationSource.renamed,
        }).toEqual({
          activationIndex: 0,
          emittedSource: 'function renamed(value){ return value + 1; }',
        });
      });
    });

    describe('when the custom squash source starts with function but lacks a parameter list', () => {
      it('falls back to the identity activation body', () => {
        // Arrange
        const generationContext = createGenerationContext();
        const customSquash = {
          toString: () => 'function broken',
        } as unknown as (inputValue: number, derivate?: boolean) => number;

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'repaired',
          customSquash,
          5,
        );

        // Assert
        expect({
          activationIndex,
          emittedSource: generationContext.emittedActivationSource.repaired,
        }).toEqual({
          activationIndex: 0,
          emittedSource: 'function repaired(x){ return x; }',
        });
      });
    });

    describe('when the custom squash source is an arrow expression with parentheses', () => {
      it('converts it into a named function with an implicit return body', () => {
        // Arrange
        const generationContext = createGenerationContext();
        const customSquash = {
          toString: () => '(value) => value + 1',
        } as unknown as (inputValue: number, derivate?: boolean) => number;

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'offset',
          customSquash,
          6,
        );

        // Assert
        expect({
          activationIndex,
          emittedSource: generationContext.emittedActivationSource.offset,
        }).toEqual({
          activationIndex: 0,
          emittedSource: 'function offset(value){ return value + 1; }',
        });
      });
    });

    describe('when the custom squash source is an arrow block with a bare parameter', () => {
      it('preserves the block body while converting the parameter list to a named function', () => {
        // Arrange
        const generationContext = createGenerationContext();
        const customSquash = {
          toString: () => 'value => { return value + 2; }',
        } as unknown as (inputValue: number, derivate?: boolean) => number;

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'blocky',
          customSquash,
          7,
        );

        // Assert
        expect({
          activationIndex,
          emittedSource: generationContext.emittedActivationSource.blocky,
        }).toEqual({
          activationIndex: 0,
          emittedSource: 'function blocky(value){ return value + 2; }',
        });
      });
    });

    describe('when the custom squash source is neither a function declaration nor an arrow function', () => {
      it('falls back to the identity activation body', () => {
        // Arrange
        const generationContext = createGenerationContext();
        const customSquash = {
          toString: () => 'not valid source',
        } as unknown as (inputValue: number, derivate?: boolean) => number;

        // Act
        const activationIndex = ensureActivationFunctionIndex(
          generationContext,
          'fallbackActivation',
          customSquash,
          8,
        );

        // Assert
        expect({
          activationIndex,
          emittedSource:
            generationContext.emittedActivationSource.fallbackActivation,
        }).toEqual({
          activationIndex: 0,
          emittedSource: 'function fallbackActivation(x){ return x; }',
        });
      });
    });
  });
});