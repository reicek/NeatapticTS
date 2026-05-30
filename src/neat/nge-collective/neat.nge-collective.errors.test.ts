import {
  NgeCollective_EvaluationError,
  NgeCollective_FieldDimensionError,
} from './neat.nge-collective.errors';

describe('neat.nge-collective.errors', () => {
  describe('NgeCollective_FieldDimensionError', () => {
    it('sets the error name to NgeCollective_FieldDimensionError', () => {
      const error = new NgeCollective_FieldDimensionError('width must be positive');
      expect(error.name).toBe('NgeCollective_FieldDimensionError');
    });

    it('sets the message to the provided string', () => {
      const error = new NgeCollective_FieldDimensionError('height must be a positive integer');
      expect(error.message).toBe('height must be a positive integer');
    });

    it('is an instance of Error', () => {
      const error = new NgeCollective_FieldDimensionError('bad dimension');
      expect(error).toBeInstanceOf(Error);
    });

    it('forwards the cause option to the base Error constructor', () => {
      const cause = new TypeError('original cause');
      const error = new NgeCollective_FieldDimensionError('wrapper', { cause });
      expect((error as Error & { cause: unknown }).cause).toBe(cause);
    });
  });

  describe('NgeCollective_EvaluationError', () => {
    it('sets the error name to NgeCollective_EvaluationError', () => {
      const error = new NgeCollective_EvaluationError('no evaluator for agent 2');
      expect(error.name).toBe('NgeCollective_EvaluationError');
    });

    it('sets the message to the provided string', () => {
      const error = new NgeCollective_EvaluationError('evaluator count mismatch');
      expect(error.message).toBe('evaluator count mismatch');
    });

    it('is an instance of Error', () => {
      const error = new NgeCollective_EvaluationError('tick failed');
      expect(error).toBeInstanceOf(Error);
    });

    it('forwards the cause option to the base Error constructor', () => {
      const cause = new RangeError('original cause');
      const error = new NgeCollective_EvaluationError('wrapper', { cause });
      expect((error as Error & { cause: unknown }).cause).toBe(cause);
    });
  });
});
