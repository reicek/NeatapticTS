'use strict';
var __awaiter =
  (this && this.__awaiter) ||
  function (thisArg, _arguments, P, generator) {
    function adopt(value) {
      return value instanceof P
        ? value
        : new P(function (resolve) {
            resolve(value);
          });
    }
    return new (P || (P = Promise))(function (resolve, reject) {
      function fulfilled(value) {
        try {
          step(generator.next(value));
        } catch (e) {
          reject(e);
        }
      }
      function rejected(value) {
        try {
          step(generator['throw'](value));
        } catch (e) {
          reject(e);
        }
      }
      function step(result) {
        result.done
          ? resolve(result.value)
          : adopt(result.value).then(fulfilled, rejected);
      }
      step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
  };
var __importDefault =
  (this && this.__importDefault) ||
  function (mod) {
    return mod && mod.__esModule ? mod : { default: mod };
  };
Object.defineProperty(exports, '__esModule', { value: true });
/* Import */
const chai_1 = require('chai');
const neataptic_1 = require('../src/neataptic');
const mocha_1 = __importDefault(require('mocha'));
/*******************************************************************************************
                      Tests the effectiveness of evolution
*******************************************************************************************/
describe('Neat', function () {
  /**
   * Tests the evolution of a network to learn the AND gate.
   */
  it('AND', function () {
    return __awaiter(this, void 0, void 0, function* () {
      this.timeout(40000); // Use this.timeout
      // Training set for the AND gate
      const trainingSet = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];
      const network = new neataptic_1.Network(2, 1);
      const results = yield network.evolve(trainingSet, {
        mutation: neataptic_1.methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.03,
        threads: 1,
      });
      chai_1.assert.isBelow(results.error, 0.03, 'Error should be below 0.03');
    });
  });
  /**
   * Tests the evolution of a network to learn the XOR gate.
   */
  it('XOR', function () {
    return __awaiter(this, void 0, void 0, function* () {
      this.timeout(40000); // Use this.timeout
      // Training set for the XOR gate
      const trainingSet = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const network = new neataptic_1.Network(2, 1);
      const results = yield network.evolve(trainingSet, {
        mutation: neataptic_1.methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.03,
        threads: 1,
      });
      chai_1.assert.isBelow(results.error, 0.03, 'Error should be below 0.03');
    });
  });
  /**
   * Tests the evolution of a network to learn the XNOR gate.
   */
  it('XNOR', function () {
    return __awaiter(this, void 0, void 0, function* () {
      this.timeout(60000); // Use this.timeout
      // Training set for the XNOR gate
      const trainingSet = [
        { input: [0, 0], output: [1] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];
      const network = new neataptic_1.Network(2, 1);
      const results = yield network.evolve(trainingSet, {
        mutation: neataptic_1.methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.03,
        threads: 1,
      });
      chai_1.assert.isBelow(results.error, 0.03, 'Error should be below 0.03');
    });
  });
});
//# sourceMappingURL=neat.js.map
