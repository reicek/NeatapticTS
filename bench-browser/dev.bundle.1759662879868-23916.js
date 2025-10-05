"use strict";
(() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined") return require.apply(this, arguments);
    throw Error('Dynamic require of "' + x + '" is not supported');
  });
  var __esm = (fn, res) => function __init() {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  };
  var __commonJS = (cb, mod) => function __require2() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // dist/architecture/connection.js
  var kGain, kGater, kOpt, kPlasticRate, Connection;
  var init_connection = __esm({
    "dist/architecture/connection.js"() {
      "use strict";
      init_node();
      kGain = Symbol("connGain");
      kGater = Symbol("connGater");
      kOpt = Symbol("connOptMoments");
      kPlasticRate = Symbol("connPlasticRate");
      Connection = class _Connection {
        from;
        to;
        weight;
        eligibility;
        previousDeltaWeight;
        totalDeltaWeight;
        xtrace;
        innovation;
        _flags;
        constructor(from, to, weight) {
          this.from = from;
          this.to = to;
          this.weight = weight ?? Math.random() * 0.2 - 0.1;
          this.eligibility = 0;
          this.previousDeltaWeight = 0;
          this.totalDeltaWeight = 0;
          this.xtrace = {
            nodes: [],
            values: []
          };
          this._flags = 3;
          this.innovation = _Connection._nextInnovation++;
        }
        toJSON() {
          const json = {
            from: this.from.index ?? void 0,
            to: this.to.index ?? void 0,
            weight: this.weight,
            gain: this.gain,
            innovation: this.innovation,
            enabled: this.enabled
          };
          if (this._flags & 4) {
            const g = this[kGater];
            if (g && typeof g.index !== "undefined")
              json.gater = g.index;
          }
          return json;
        }
        static innovationID(sourceNodeId, targetNodeId) {
          return 0.5 * (sourceNodeId + targetNodeId) * (sourceNodeId + targetNodeId + 1) + targetNodeId;
        }
        static _nextInnovation = 1;
        static resetInnovationCounter(value = 1) {
          _Connection._nextInnovation = value;
        }
        static _pool = [];
        static acquire(from, to, weight) {
          let c;
          if (_Connection._pool.length) {
            c = _Connection._pool.pop();
            c.from = from;
            c.to = to;
            c.weight = weight ?? Math.random() * 0.2 - 0.1;
            if (c[kGain] !== void 0)
              delete c[kGain];
            if (c[kGater] !== void 0)
              delete c[kGater];
            c._flags = 3;
            c.eligibility = 0;
            c.previousDeltaWeight = 0;
            c.totalDeltaWeight = 0;
            c.xtrace.nodes.length = 0;
            c.xtrace.values.length = 0;
            if (c[kOpt])
              delete c[kOpt];
            c.innovation = _Connection._nextInnovation++;
          } else
            c = new _Connection(from, to, weight);
          return c;
        }
        static release(conn) {
          _Connection._pool.push(conn);
        }
        get enabled() {
          return (this._flags & 1) !== 0;
        }
        set enabled(v) {
          this._flags = v ? this._flags | 1 : this._flags & ~1;
        }
        get dcMask() {
          return (this._flags & 2) !== 0 ? 1 : 0;
        }
        set dcMask(v) {
          this._flags = v ? this._flags | 2 : this._flags & ~2;
        }
        get hasGater() {
          return (this._flags & 4) !== 0;
        }
        get plastic() {
          return (this._flags & 8) !== 0;
        }
        set plastic(v) {
          if (v)
            this._flags |= 8;
          else
            this._flags &= ~8;
          if (!v && this[kPlasticRate] !== void 0)
            delete this[kPlasticRate];
        }
        get gain() {
          return this[kGain] === void 0 ? 1 : this[kGain];
        }
        set gain(v) {
          if (v === 1) {
            if (this[kGain] !== void 0)
              delete this[kGain];
          } else {
            this[kGain] = v;
          }
        }
        _ensureOptBag() {
          let bag = this[kOpt];
          if (!bag) {
            bag = {};
            this[kOpt] = bag;
          }
          return bag;
        }
        _getOpt(k) {
          const bag = this[kOpt];
          return bag ? bag[k] : void 0;
        }
        _setOpt(k, v) {
          if (v === void 0) {
            const bag = this[kOpt];
            if (bag)
              delete bag[k];
          } else {
            this._ensureOptBag()[k] = v;
          }
        }
        get firstMoment() {
          return this._getOpt("firstMoment");
        }
        set firstMoment(v) {
          this._setOpt("firstMoment", v);
        }
        get secondMoment() {
          return this._getOpt("secondMoment");
        }
        set secondMoment(v) {
          this._setOpt("secondMoment", v);
        }
        get gradientAccumulator() {
          return this._getOpt("gradientAccumulator");
        }
        set gradientAccumulator(v) {
          this._setOpt("gradientAccumulator", v);
        }
        get maxSecondMoment() {
          return this._getOpt("maxSecondMoment");
        }
        set maxSecondMoment(v) {
          this._setOpt("maxSecondMoment", v);
        }
        get infinityNorm() {
          return this._getOpt("infinityNorm");
        }
        set infinityNorm(v) {
          this._setOpt("infinityNorm", v);
        }
        get secondMomentum() {
          return this._getOpt("secondMomentum");
        }
        set secondMomentum(v) {
          this._setOpt("secondMomentum", v);
        }
        get lookaheadShadowWeight() {
          return this._getOpt("lookaheadShadowWeight");
        }
        set lookaheadShadowWeight(v) {
          this._setOpt("lookaheadShadowWeight", v);
        }
        get gater() {
          return (this._flags & 4) !== 0 ? this[kGater] : null;
        }
        set gater(node) {
          if (node === null) {
            if ((this._flags & 4) !== 0) {
              this._flags &= ~4;
              if (this[kGater] !== void 0)
                delete this[kGater];
            }
          } else {
            this[kGater] = node;
            this._flags |= 4;
          }
        }
        get plasticityRate() {
          return this[kPlasticRate] === void 0 ? 0 : this[kPlasticRate];
        }
        set plasticityRate(v) {
          if (v === void 0 || v === 0) {
            if (this[kPlasticRate] !== void 0)
              delete this[kPlasticRate];
            this._flags &= ~8;
          } else {
            this[kPlasticRate] = v;
            this._flags |= 8;
          }
        }
        get dropConnectActiveMask() {
          return this.dcMask;
        }
        set dropConnectActiveMask(v) {
          this.dcMask = v;
        }
      };
    }
  });

  // dist/config.js
  var config;
  var init_config = __esm({
    "dist/config.js"() {
      "use strict";
      config = {
        warnings: false,
        float32Mode: false,
        deterministicChainMode: false,
        enableGatingTraces: true,
        enableNodePooling: false,
        enableSlabArrayPooling: false
      };
    }
  });

  // dist/neat/neat.constants.js
  var neat_constants_exports = {};
  __export(neat_constants_exports, {
    EPSILON: () => EPSILON,
    EXTRA_CONNECTION_PROBABILITY: () => EXTRA_CONNECTION_PROBABILITY,
    NORM_EPSILON: () => NORM_EPSILON,
    PROB_EPSILON: () => PROB_EPSILON
  });
  var EPSILON, PROB_EPSILON, NORM_EPSILON, EXTRA_CONNECTION_PROBABILITY;
  var init_neat_constants = __esm({
    "dist/neat/neat.constants.js"() {
      "use strict";
      EPSILON = 1e-9;
      PROB_EPSILON = 1e-15;
      NORM_EPSILON = 1e-5;
      EXTRA_CONNECTION_PROBABILITY = 0.5;
    }
  });

  // dist/methods/cost.js
  var Cost;
  var init_cost = __esm({
    "dist/methods/cost.js"() {
      "use strict";
      init_neat_constants();
      Cost = class {
        static crossEntropy(targets, outputs) {
          let error = 0;
          const epsilon = PROB_EPSILON;
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          for (let i = 0; i < outputs.length; i++) {
            const target = targets[i];
            const output = outputs[i];
            const clampedOutput = Math.max(epsilon, Math.min(1 - epsilon, output));
            if (target === 1) {
              error -= Math.log(clampedOutput);
            } else if (target === 0) {
              error -= Math.log(1 - clampedOutput);
            } else {
              error -= target * Math.log(clampedOutput) + (1 - target) * Math.log(1 - clampedOutput);
            }
          }
          return error / outputs.length;
        }
        static softmaxCrossEntropy(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          const n = outputs.length;
          let tSum = 0;
          for (const t of targets)
            tSum += t;
          const normTargets = tSum > 0 ? targets.map((t) => t / tSum) : targets.slice();
          const max = Math.max(...outputs);
          const exps = outputs.map((o) => Math.exp(o - max));
          const sum = exps.reduce((a, b) => a + b, 0) || 1;
          const probs = exps.map((e) => e / sum);
          let loss = 0;
          const eps = PROB_EPSILON;
          for (let i = 0; i < n; i++) {
            const p = Math.min(1 - eps, Math.max(eps, probs[i]));
            const t = normTargets[i];
            loss -= t * Math.log(p);
          }
          return loss;
        }
        static mse(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          let error = 0;
          outputs.forEach((output, outputIndex) => {
            error += Math.pow(targets[outputIndex] - output, 2);
          });
          return error / outputs.length;
        }
        static binary(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          let misses = 0;
          outputs.forEach((output, outputIndex) => {
            misses += Math.round(targets[outputIndex]) !== Math.round(output) ? 1 : 0;
          });
          return misses / outputs.length;
        }
        static mae(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          let error = 0;
          outputs.forEach((output, outputIndex) => {
            error += Math.abs(targets[outputIndex] - output);
          });
          return error / outputs.length;
        }
        static mape(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          let error = 0;
          const epsilon = PROB_EPSILON;
          outputs.forEach((output, outputIndex) => {
            const target = targets[outputIndex];
            error += Math.abs((target - output) / Math.max(Math.abs(target), epsilon));
          });
          return error / outputs.length;
        }
        static msle(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          let error = 0;
          outputs.forEach((output, outputIndex) => {
            const target = targets[outputIndex];
            const logTarget = Math.log(Math.max(target, 0) + 1);
            const logOutput = Math.log(Math.max(output, 0) + 1);
            error += Math.pow(logTarget - logOutput, 2);
          });
          return error / outputs.length;
        }
        static hinge(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          let error = 0;
          outputs.forEach((output, outputIndex) => {
            const target = targets[outputIndex];
            error += Math.max(0, 1 - target * output);
          });
          return error / outputs.length;
        }
        static focalLoss(targets, outputs, gamma = 2, alpha = 0.25) {
          let error = 0;
          const epsilon = PROB_EPSILON;
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          for (let i = 0; i < outputs.length; i++) {
            const t = targets[i];
            const p = Math.max(epsilon, Math.min(1 - epsilon, outputs[i]));
            const pt = t === 1 ? p : 1 - p;
            const a = t === 1 ? alpha : 1 - alpha;
            error += -a * Math.pow(1 - pt, gamma) * Math.log(pt);
          }
          return error / outputs.length;
        }
        static labelSmoothing(targets, outputs, smoothing = 0.1) {
          let error = 0;
          const epsilon = PROB_EPSILON;
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          for (let i = 0; i < outputs.length; i++) {
            const t = targets[i] * (1 - smoothing) + 0.5 * smoothing;
            const p = Math.max(epsilon, Math.min(1 - epsilon, outputs[i]));
            error -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
          }
          return error / outputs.length;
        }
      };
    }
  });

  // dist/methods/rate.js
  var Rate;
  var init_rate = __esm({
    "dist/methods/rate.js"() {
      "use strict";
      Rate = class {
        static fixed() {
          const func = (baseRate, iteration) => {
            return baseRate;
          };
          return func;
        }
        static step(gamma = 0.9, stepSize = 100) {
          const func = (baseRate, iteration) => {
            return Math.max(0, baseRate * Math.pow(gamma, Math.floor(iteration / stepSize)));
          };
          return func;
        }
        static exp(gamma = 0.999) {
          const func = (baseRate, iteration) => {
            return baseRate * Math.pow(gamma, iteration);
          };
          return func;
        }
        static inv(gamma = 1e-3, power = 2) {
          const func = (baseRate, iteration) => {
            return baseRate / (1 + gamma * Math.pow(iteration, power));
          };
          return func;
        }
        static cosineAnnealing(period = 1e3, minRate = 0) {
          const func = (baseRate, iteration) => {
            const currentCycleIteration = iteration % period;
            const cosineDecay = 0.5 * (1 + Math.cos(currentCycleIteration / period * Math.PI));
            return minRate + (baseRate - minRate) * cosineDecay;
          };
          return func;
        }
        static cosineAnnealingWarmRestarts(initialPeriod = 1e3, minRate = 0, tMult = 1) {
          let period = initialPeriod;
          let cycleStart = 0;
          let cycleEnd = period;
          return (baseRate, iteration) => {
            while (iteration >= cycleEnd) {
              cycleStart = cycleEnd;
              period = Math.max(1, Math.round(period * tMult));
              cycleEnd = cycleStart + period;
            }
            const cyclePos = iteration - cycleStart;
            const cosineDecay = 0.5 * (1 + Math.cos(cyclePos / period * Math.PI));
            return minRate + (baseRate - minRate) * cosineDecay;
          };
        }
        static linearWarmupDecay(totalSteps, warmupSteps, endRate = 0) {
          if (totalSteps <= 0)
            throw new Error("totalSteps must be > 0");
          const warm = Math.min(warmupSteps ?? Math.max(1, Math.floor(totalSteps * 0.1)), totalSteps - 1);
          return (baseRate, iteration) => {
            if (iteration <= warm) {
              return baseRate * (iteration / Math.max(1, warm));
            }
            if (iteration >= totalSteps)
              return endRate;
            const decaySteps = totalSteps - warm;
            const progress = (iteration - warm) / decaySteps;
            return endRate + (baseRate - endRate) * (1 - progress);
          };
        }
        static reduceOnPlateau(options) {
          const { factor = 0.5, patience = 10, minDelta = 1e-4, cooldown = 0, minRate = 0, verbose = false } = options || {};
          let currentRate;
          let bestError;
          let lastImprovementIter = 0;
          let cooldownUntil = -1;
          return (baseRate, iteration, lastError) => {
            if (currentRate === void 0)
              currentRate = baseRate;
            if (lastError !== void 0) {
              if (bestError === void 0 || lastError < bestError - minDelta) {
                bestError = lastError;
                lastImprovementIter = iteration;
              } else if (iteration - lastImprovementIter >= patience && iteration >= cooldownUntil) {
                const newRate = Math.max(minRate, currentRate * factor);
                if (newRate < currentRate) {
                  currentRate = newRate;
                  cooldownUntil = iteration + cooldown;
                  lastImprovementIter = iteration;
                }
              }
            }
            return currentRate;
          };
        }
      };
    }
  });

  // dist/methods/activation.js
  var Activation, activation_default;
  var init_activation = __esm({
    "dist/methods/activation.js"() {
      "use strict";
      Activation = {
        logistic: (x, derivate = false) => {
          const fx = 1 / (1 + Math.exp(-x));
          return !derivate ? fx : fx * (1 - fx);
        },
        sigmoid: (x, derivate = false) => {
          const fx = 1 / (1 + Math.exp(-x));
          return !derivate ? fx : fx * (1 - fx);
        },
        tanh: (x, derivate = false) => {
          return derivate ? 1 - Math.pow(Math.tanh(x), 2) : Math.tanh(x);
        },
        identity: (x, derivate = false) => {
          return derivate ? 1 : x;
        },
        step: (x, derivate = false) => {
          return derivate ? 0 : x > 0 ? 1 : 0;
        },
        relu: (x, derivate = false) => {
          return derivate ? x > 0 ? 1 : 0 : x > 0 ? x : 0;
        },
        softsign: (x, derivate = false) => {
          const d = 1 + Math.abs(x);
          return derivate ? 1 / Math.pow(d, 2) : x / d;
        },
        sinusoid: (x, derivate = false) => {
          return derivate ? Math.cos(x) : Math.sin(x);
        },
        gaussian: (x, derivate = false) => {
          const d = Math.exp(-Math.pow(x, 2));
          return derivate ? -2 * x * d : d;
        },
        bentIdentity: (x, derivate = false) => {
          const d = Math.sqrt(Math.pow(x, 2) + 1);
          return derivate ? x / (2 * d) + 1 : (d - 1) / 2 + x;
        },
        bipolar: (x, derivate = false) => {
          return derivate ? 0 : x > 0 ? 1 : -1;
        },
        bipolarSigmoid: (x, derivate = false) => {
          const d = 2 / (1 + Math.exp(-x)) - 1;
          return derivate ? 1 / 2 * (1 + d) * (1 - d) : d;
        },
        hardTanh: (x, derivate = false) => {
          return derivate ? x > -1 && x < 1 ? 1 : 0 : Math.max(-1, Math.min(1, x));
        },
        absolute: (x, derivate = false) => {
          return derivate ? x < 0 ? -1 : 1 : Math.abs(x);
        },
        inverse: (x, derivate = false) => {
          return derivate ? -1 : 1 - x;
        },
        selu: (x, derivate = false) => {
          const alpha = 1.6732632423543772;
          const scale = 1.0507009873554805;
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          return derivate ? x > 0 ? scale : (fx + alpha) * scale : fx * scale;
        },
        softplus: (x, derivate = false) => {
          const fx = 1 / (1 + Math.exp(-x));
          if (derivate) {
            return fx;
          } else {
            if (x > 30) {
              return x;
            } else if (x < -30) {
              return Math.exp(x);
            }
            return Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
          }
        },
        swish: (x, derivate = false) => {
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          if (derivate) {
            const swish_x = x * sigmoid_x;
            return swish_x + sigmoid_x * (1 - swish_x);
          } else {
            return x * sigmoid_x;
          }
        },
        gelu: (x, derivate = false) => {
          const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
          if (derivate) {
            const intermediate = Math.sqrt(2 / Math.PI) * (1 + 0.134145 * x * x);
            const sech_arg = Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3));
            const sech_val = 1 / Math.cosh(sech_arg);
            const sech_sq = sech_val * sech_val;
            return cdf + x * 0.5 * intermediate * sech_sq;
          } else {
            return x * cdf;
          }
        },
        mish: (x, derivate = false) => {
          let sp_x;
          if (x > 30) {
            sp_x = x;
          } else if (x < -30) {
            sp_x = Math.exp(x);
          } else {
            sp_x = Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
          }
          const tanh_sp_x = Math.tanh(sp_x);
          if (derivate) {
            const sigmoid_x = 1 / (1 + Math.exp(-x));
            const sech_sp_x = 1 / Math.cosh(sp_x);
            const sech_sq_sp_x = sech_sp_x * sech_sp_x;
            return tanh_sp_x + x * sech_sq_sp_x * sigmoid_x;
          } else {
            return x * tanh_sp_x;
          }
        }
      };
      activation_default = Activation;
    }
  });

  // dist/methods/gating.js
  var gating;
  var init_gating = __esm({
    "dist/methods/gating.js"() {
      "use strict";
      gating = {
        OUTPUT: {
          name: "OUTPUT"
        },
        INPUT: {
          name: "INPUT"
        },
        SELF: {
          name: "SELF"
        }
      };
    }
  });

  // dist/methods/mutation.js
  var mutation, mutation_default;
  var init_mutation = __esm({
    "dist/methods/mutation.js"() {
      "use strict";
      init_activation();
      mutation = {
        ADD_NODE: {
          name: "ADD_NODE"
        },
        SUB_NODE: {
          name: "SUB_NODE",
          keep_gates: true
        },
        ADD_CONN: {
          name: "ADD_CONN"
        },
        SUB_CONN: {
          name: "SUB_CONN"
        },
        MOD_WEIGHT: {
          name: "MOD_WEIGHT",
          min: -1,
          max: 1
        },
        MOD_BIAS: {
          name: "MOD_BIAS",
          min: -1,
          max: 1
        },
        MOD_ACTIVATION: {
          name: "MOD_ACTIVATION",
          mutateOutput: true,
          allowed: [
            activation_default.logistic,
            activation_default.tanh,
            activation_default.relu,
            activation_default.identity,
            activation_default.step,
            activation_default.softsign,
            activation_default.sinusoid,
            activation_default.gaussian,
            activation_default.bentIdentity,
            activation_default.bipolar,
            activation_default.bipolarSigmoid,
            activation_default.hardTanh,
            activation_default.absolute,
            activation_default.inverse,
            activation_default.selu,
            activation_default.softplus,
            activation_default.swish,
            activation_default.gelu,
            activation_default.mish
          ]
        },
        ADD_SELF_CONN: {
          name: "ADD_SELF_CONN"
        },
        SUB_SELF_CONN: {
          name: "SUB_SELF_CONN"
        },
        ADD_GATE: {
          name: "ADD_GATE"
        },
        SUB_GATE: {
          name: "SUB_GATE"
        },
        ADD_BACK_CONN: {
          name: "ADD_BACK_CONN"
        },
        SUB_BACK_CONN: {
          name: "SUB_BACK_CONN"
        },
        SWAP_NODES: {
          name: "SWAP_NODES",
          mutateOutput: true
        },
        REINIT_WEIGHT: {
          name: "REINIT_WEIGHT",
          min: -1,
          max: 1
        },
        BATCH_NORM: {
          name: "BATCH_NORM"
        },
        ADD_LSTM_NODE: {
          name: "ADD_LSTM_NODE"
        },
        ADD_GRU_NODE: {
          name: "ADD_GRU_NODE"
        },
        ALL: [],
        FFW: []
      };
      mutation.ALL = [
        mutation.ADD_NODE,
        mutation.SUB_NODE,
        mutation.ADD_CONN,
        mutation.SUB_CONN,
        mutation.MOD_WEIGHT,
        mutation.MOD_BIAS,
        mutation.MOD_ACTIVATION,
        mutation.ADD_GATE,
        mutation.SUB_GATE,
        mutation.ADD_SELF_CONN,
        mutation.SUB_SELF_CONN,
        mutation.ADD_BACK_CONN,
        mutation.SUB_BACK_CONN,
        mutation.SWAP_NODES,
        mutation.REINIT_WEIGHT,
        mutation.BATCH_NORM,
        mutation.ADD_LSTM_NODE,
        mutation.ADD_GRU_NODE
      ];
      mutation.FFW = [
        mutation.ADD_NODE,
        mutation.SUB_NODE,
        mutation.ADD_CONN,
        mutation.SUB_CONN,
        mutation.MOD_WEIGHT,
        mutation.MOD_BIAS,
        mutation.MOD_ACTIVATION,
        mutation.SWAP_NODES,
        mutation.REINIT_WEIGHT,
        mutation.BATCH_NORM
      ];
      mutation_default = mutation;
    }
  });

  // dist/methods/selection.js
  var selection;
  var init_selection = __esm({
    "dist/methods/selection.js"() {
      "use strict";
      selection = {
        FITNESS_PROPORTIONATE: {
          name: "FITNESS_PROPORTIONATE"
        },
        POWER: {
          name: "POWER",
          power: 4
        },
        TOURNAMENT: {
          name: "TOURNAMENT",
          size: 5,
          probability: 0.5
        }
      };
    }
  });

  // dist/methods/crossover.js
  var crossover;
  var init_crossover = __esm({
    "dist/methods/crossover.js"() {
      "use strict";
      crossover = {
        SINGLE_POINT: {
          name: "SINGLE_POINT",
          config: [0.4]
        },
        TWO_POINT: {
          name: "TWO_POINT",
          config: [0.4, 0.9]
        },
        UNIFORM: {
          name: "UNIFORM"
        },
        AVERAGE: {
          name: "AVERAGE"
        }
      };
    }
  });

  // dist/methods/connection.js
  var groupConnection, connection_default;
  var init_connection2 = __esm({
    "dist/methods/connection.js"() {
      "use strict";
      groupConnection = Object.freeze({
        ALL_TO_ALL: Object.freeze({
          name: "ALL_TO_ALL"
        }),
        ALL_TO_ELSE: Object.freeze({
          name: "ALL_TO_ELSE"
        }),
        ONE_TO_ONE: Object.freeze({
          name: "ONE_TO_ONE"
        })
      });
      connection_default = groupConnection;
    }
  });

  // dist/methods/methods.js
  var methods_exports = {};
  __export(methods_exports, {
    Activation: () => activation_default,
    Cost: () => Cost,
    Rate: () => Rate,
    crossover: () => crossover,
    gating: () => gating,
    groupConnection: () => connection_default,
    mutation: () => mutation,
    selection: () => selection
  });
  var init_methods = __esm({
    "dist/methods/methods.js"() {
      "use strict";
      init_cost();
      init_rate();
      init_activation();
      init_gating();
      init_mutation();
      init_selection();
      init_crossover();
      init_connection2();
    }
  });

  // dist/architecture/node.js
  var node_exports = {};
  __export(node_exports, {
    default: () => Node
  });
  var Node;
  var init_node = __esm({
    "dist/architecture/node.js"() {
      "use strict";
      init_connection();
      init_config();
      init_methods();
      Node = class _Node {
        bias;
        squash;
        type;
        activation;
        state;
        old;
        mask;
        previousDeltaBias;
        totalDeltaBias;
        connections;
        error;
        derivative;
        index;
        isActivating;
        geneId;
        static _globalNodeIndex = 0;
        static _nextGeneId = 1;
        constructor(type = "hidden", customActivation, rng = Math.random) {
          this.bias = type === "input" ? 0 : rng() * 0.2 - 0.1;
          this.squash = customActivation || activation_default.logistic || ((x) => x);
          this.type = type;
          this.activation = 0;
          this.state = 0;
          this.old = 0;
          this.mask = 1;
          this.previousDeltaBias = 0;
          this.totalDeltaBias = 0;
          this.connections = {
            in: [],
            out: [],
            gated: [],
            self: []
          };
          this.error = {
            responsibility: 0,
            projected: 0,
            gated: 0
          };
          if (typeof this.index === "undefined") {
            this.index = _Node._globalNodeIndex++;
          }
          this.geneId = _Node._nextGeneId++;
        }
        setActivation(fn) {
          this.squash = fn;
        }
        activate(input) {
          return this._activateCore(true, input);
        }
        noTraceActivate(input) {
          return this._activateCore(false, input);
        }
        _activateCore(withTrace, input) {
          if (this.mask === 0) {
            this.activation = 0;
            return 0;
          }
          if (typeof input !== "undefined") {
            if (this.type === "input") {
              this.activation = input;
              return this.activation;
            }
            this.state = input;
            this.activation = this.squash(this.state) * this.mask;
            this.derivative = this.squash(this.state, true);
            for (const connection of this.connections.gated)
              connection.gain = this.activation;
            if (withTrace)
              for (const connection of this.connections.in)
                connection.eligibility = connection.from.activation;
            return this.activation;
          }
          this.old = this.state;
          let newState = this.bias;
          if (this.connections.self.length) {
            for (const conn of this.connections.self) {
              if (conn.dcMask === 0)
                continue;
              newState += conn.gain * conn.weight * this.old;
            }
          }
          if (this.connections.in.length) {
            for (const conn of this.connections.in) {
              if (conn.dcMask === 0 || conn.enabled === false)
                continue;
              newState += conn.from.activation * conn.weight * conn.gain;
            }
          }
          this.state = newState;
          if (typeof this.squash !== "function") {
            if (config.warnings)
              console.warn("Invalid activation function; using identity.");
            this.squash = activation_default.identity;
          }
          if (typeof this.mask !== "number")
            this.mask = 1;
          this.activation = this.squash(this.state) * this.mask;
          this.derivative = this.squash(this.state, true);
          if (this.connections.gated.length) {
            for (const conn of this.connections.gated)
              conn.gain = this.activation;
          }
          if (withTrace) {
            for (const conn of this.connections.in)
              conn.eligibility = conn.from.activation;
          }
          return this.activation;
        }
        propagate(rate, momentum, update, regularization = 0, target) {
          if (update && momentum > 0) {
            for (const connection of this.connections.in) {
              connection.weight += momentum * connection.previousDeltaWeight;
              connection.eligibility += 1e-12;
            }
            this.bias += momentum * this.previousDeltaBias;
          }
          let error = 0;
          if (this.type === "output") {
            this.error.responsibility = this.error.projected = target - this.activation;
          } else {
            for (const connection of this.connections.out) {
              error += connection.to.error.responsibility * connection.weight * connection.gain;
            }
            this.error.projected = this.derivative * error;
            error = 0;
            for (const connection of this.connections.gated) {
              const node = connection.to;
              let influence = node.connections.self.reduce((sum, selfConn) => sum + (selfConn.gater === this ? node.old : 0), 0);
              influence += connection.weight * connection.from.activation;
              error += node.error.responsibility * influence;
            }
            this.error.gated = this.derivative * error;
            this.error.responsibility = this.error.projected + this.error.gated;
          }
          if (this.type === "constant")
            return;
          for (const connection of this.connections.in) {
            if (connection.dcMask === 0) {
              connection.totalDeltaWeight += 0;
              continue;
            }
            let gradient = this.error.projected * connection.eligibility;
            for (let j = 0; j < connection.xtrace.nodes.length; j++) {
              const node = connection.xtrace.nodes[j];
              const value = connection.xtrace.values[j];
              gradient += node.error.responsibility * value;
            }
            let regTerm = 0;
            if (typeof regularization === "function") {
              regTerm = regularization(connection.weight);
            } else if (typeof regularization === "object" && regularization !== null) {
              if (regularization.type === "L1") {
                regTerm = regularization.lambda * Math.sign(connection.weight);
              } else if (regularization.type === "L2") {
                regTerm = regularization.lambda * connection.weight;
              }
            } else {
              regTerm = regularization * connection.weight;
            }
            let deltaWeight = rate * (gradient * this.mask - regTerm);
            if (!Number.isFinite(deltaWeight)) {
              console.warn("deltaWeight is not finite, clamping to 0", {
                node: this.index,
                connection,
                deltaWeight
              });
              deltaWeight = 0;
            } else if (Math.abs(deltaWeight) > 1e3) {
              deltaWeight = Math.sign(deltaWeight) * 1e3;
            }
            connection.totalDeltaWeight += deltaWeight;
            if (!Number.isFinite(connection.totalDeltaWeight)) {
              console.warn("totalDeltaWeight became NaN/Infinity, resetting to 0", {
                node: this.index,
                connection
              });
              connection.totalDeltaWeight = 0;
            }
            if (update) {
              let currentDeltaWeight = connection.totalDeltaWeight + momentum * connection.previousDeltaWeight;
              if (!Number.isFinite(currentDeltaWeight)) {
                console.warn("currentDeltaWeight is not finite, clamping to 0", {
                  node: this.index,
                  connection,
                  currentDeltaWeight
                });
                currentDeltaWeight = 0;
              } else if (Math.abs(currentDeltaWeight) > 1e3) {
                currentDeltaWeight = Math.sign(currentDeltaWeight) * 1e3;
              }
              if (momentum > 0) {
                connection.weight -= momentum * connection.previousDeltaWeight;
              }
              connection.weight += currentDeltaWeight;
              if (!Number.isFinite(connection.weight)) {
                console.warn(`Weight update produced invalid value: ${connection.weight}. Resetting to 0.`, { node: this.index, connection });
                connection.weight = 0;
              } else if (Math.abs(connection.weight) > 1e6) {
                connection.weight = Math.sign(connection.weight) * 1e6;
              }
              connection.previousDeltaWeight = currentDeltaWeight;
              connection.totalDeltaWeight = 0;
            }
          }
          for (const connection of this.connections.self) {
            if (connection.dcMask === 0) {
              connection.totalDeltaWeight += 0;
              continue;
            }
            let gradient = this.error.projected * connection.eligibility;
            for (let j = 0; j < connection.xtrace.nodes.length; j++) {
              const node = connection.xtrace.nodes[j];
              const value = connection.xtrace.values[j];
              gradient += node.error.responsibility * value;
            }
            let regTerm = 0;
            if (typeof regularization === "function") {
              regTerm = regularization(connection.weight);
            } else if (typeof regularization === "object" && regularization !== null) {
              if (regularization.type === "L1") {
                regTerm = regularization.lambda * Math.sign(connection.weight);
              } else if (regularization.type === "L2") {
                regTerm = regularization.lambda * connection.weight;
              }
            } else {
              regTerm = regularization * connection.weight;
            }
            let deltaWeight = rate * (gradient * this.mask - regTerm);
            if (!Number.isFinite(deltaWeight)) {
              console.warn("self deltaWeight is not finite, clamping to 0", {
                node: this.index,
                connection,
                deltaWeight
              });
              deltaWeight = 0;
            } else if (Math.abs(deltaWeight) > 1e3) {
              deltaWeight = Math.sign(deltaWeight) * 1e3;
            }
            connection.totalDeltaWeight += deltaWeight;
            if (!Number.isFinite(connection.totalDeltaWeight)) {
              console.warn("self totalDeltaWeight became NaN/Infinity, resetting to 0", { node: this.index, connection });
              connection.totalDeltaWeight = 0;
            }
            if (update) {
              let currentDeltaWeight = connection.totalDeltaWeight + momentum * connection.previousDeltaWeight;
              if (!Number.isFinite(currentDeltaWeight)) {
                console.warn("self currentDeltaWeight is not finite, clamping to 0", {
                  node: this.index,
                  connection,
                  currentDeltaWeight
                });
                currentDeltaWeight = 0;
              } else if (Math.abs(currentDeltaWeight) > 1e3) {
                currentDeltaWeight = Math.sign(currentDeltaWeight) * 1e3;
              }
              if (momentum > 0) {
                connection.weight -= momentum * connection.previousDeltaWeight;
              }
              connection.weight += currentDeltaWeight;
              if (!Number.isFinite(connection.weight)) {
                console.warn("self weight update produced invalid value, resetting to 0", { node: this.index, connection });
                connection.weight = 0;
              } else if (Math.abs(connection.weight) > 1e6) {
                connection.weight = Math.sign(connection.weight) * 1e6;
              }
              connection.previousDeltaWeight = currentDeltaWeight;
              connection.totalDeltaWeight = 0;
            }
          }
          let deltaBias = rate * this.error.responsibility;
          if (!Number.isFinite(deltaBias)) {
            console.warn("deltaBias is not finite, clamping to 0", {
              node: this.index,
              deltaBias
            });
            deltaBias = 0;
          } else if (Math.abs(deltaBias) > 1e3) {
            deltaBias = Math.sign(deltaBias) * 1e3;
          }
          this.totalDeltaBias += deltaBias;
          if (!Number.isFinite(this.totalDeltaBias)) {
            console.warn("totalDeltaBias became NaN/Infinity, resetting to 0", {
              node: this.index
            });
            this.totalDeltaBias = 0;
          }
          if (update) {
            let currentDeltaBias = this.totalDeltaBias + momentum * this.previousDeltaBias;
            if (!Number.isFinite(currentDeltaBias)) {
              console.warn("currentDeltaBias is not finite, clamping to 0", {
                node: this.index,
                currentDeltaBias
              });
              currentDeltaBias = 0;
            } else if (Math.abs(currentDeltaBias) > 1e3) {
              currentDeltaBias = Math.sign(currentDeltaBias) * 1e3;
            }
            if (momentum > 0) {
              this.bias -= momentum * this.previousDeltaBias;
            }
            this.bias += currentDeltaBias;
            if (!Number.isFinite(this.bias)) {
              console.warn("bias update produced invalid value, resetting to 0", {
                node: this.index
              });
              this.bias = 0;
            } else if (Math.abs(this.bias) > 1e6) {
              this.bias = Math.sign(this.bias) * 1e6;
            }
            this.previousDeltaBias = currentDeltaBias;
            this.totalDeltaBias = 0;
          }
        }
        toJSON() {
          return {
            index: this.index,
            bias: this.bias,
            type: this.type,
            squash: this.squash ? this.squash.name : null,
            mask: this.mask
          };
        }
        static fromJSON(json) {
          const node = new _Node(json.type);
          node.bias = json.bias;
          node.mask = json.mask;
          if (json.squash) {
            const squashFn = activation_default[json.squash];
            if (typeof squashFn === "function") {
              node.squash = squashFn;
            } else {
              console.warn(`fromJSON: Unknown or invalid squash function '${json.squash}' for node. Using identity.`);
              node.squash = activation_default.identity;
            }
          }
          return node;
        }
        isConnectedTo(target) {
          return this.connections.out.some((conn) => conn.to === target);
        }
        mutate(method) {
          if (!method) {
            throw new Error("Mutation method cannot be null or undefined.");
          }
          if (!(method.name in mutation)) {
            throw new Error(`Unknown mutation method: ${method.name}`);
          }
          switch (method) {
            case mutation.MOD_ACTIVATION:
              if (!method.allowed || method.allowed.length === 0) {
                console.warn("MOD_ACTIVATION mutation called without allowed functions specified.");
                return;
              }
              const allowed = method.allowed;
              const currentIndex = allowed.indexOf(this.squash);
              let newIndex = currentIndex;
              if (allowed.length > 1) {
                newIndex = (currentIndex + Math.floor(Math.random() * (allowed.length - 1)) + 1) % allowed.length;
              }
              this.squash = allowed[newIndex];
              break;
            case mutation.MOD_BIAS:
              const min = method.min ?? -1;
              const max = method.max ?? 1;
              const modification = Math.random() * (max - min) + min;
              this.bias += modification;
              break;
            case mutation.REINIT_WEIGHT:
              const reinitMin = method.min ?? -1;
              const reinitMax = method.max ?? 1;
              for (const conn of this.connections.in) {
                conn.weight = Math.random() * (reinitMax - reinitMin) + reinitMin;
              }
              for (const conn of this.connections.out) {
                conn.weight = Math.random() * (reinitMax - reinitMin) + reinitMin;
              }
              for (const conn of this.connections.self) {
                conn.weight = Math.random() * (reinitMax - reinitMin) + reinitMin;
              }
              break;
            case mutation.BATCH_NORM:
              this.batchNorm = true;
              break;
            default:
              throw new Error(`Unsupported mutation method: ${method.name}`);
          }
        }
        connect(target, weight) {
          const connections = [];
          if (!target) {
            throw new Error("Cannot connect to an undefined target.");
          }
          if ("bias" in target) {
            const targetNode = target;
            if (targetNode === this) {
              if (this.connections.self.length === 0) {
                const selfConnection = Connection.acquire(this, this, weight ?? 1);
                this.connections.self.push(selfConnection);
                connections.push(selfConnection);
              }
            } else {
              const connection = Connection.acquire(this, targetNode, weight);
              targetNode.connections.in.push(connection);
              this.connections.out.push(connection);
              connections.push(connection);
            }
          } else if ("nodes" in target && Array.isArray(target.nodes)) {
            for (const node of target.nodes) {
              const connection = Connection.acquire(this, node, weight);
              node.connections.in.push(connection);
              this.connections.out.push(connection);
              connections.push(connection);
            }
          } else {
            throw new Error("Invalid target type for connection. Must be a Node or a group { nodes: Node[] }.");
          }
          return connections;
        }
        disconnect(target, twosided = false) {
          if (this === target) {
            this.connections.self = [];
            return;
          }
          this.connections.out = this.connections.out.filter((conn) => {
            if (conn.to === target) {
              target.connections.in = target.connections.in.filter((inConn) => inConn !== conn);
              if (conn.gater) {
                conn.gater.ungate(conn);
              }
              return false;
            }
            return true;
          });
          if (twosided) {
            target.disconnect(this, false);
          }
        }
        gate(connections) {
          if (!Array.isArray(connections)) {
            connections = [connections];
          }
          for (const connection of connections) {
            if (!connection || !connection.from || !connection.to) {
              console.warn("Attempted to gate an invalid or incomplete connection.");
              continue;
            }
            if (connection.gater === this) {
              console.warn("Node is already gating this connection.");
              continue;
            }
            if (connection.gater !== null) {
              console.warn("Connection is already gated by another node. Ungate first.");
              continue;
            }
            this.connections.gated.push(connection);
            connection.gater = this;
          }
        }
        ungate(connections) {
          if (!Array.isArray(connections)) {
            connections = [connections];
          }
          for (const connection of connections) {
            if (!connection)
              continue;
            const index = this.connections.gated.indexOf(connection);
            if (index !== -1) {
              this.connections.gated.splice(index, 1);
              connection.gater = null;
              connection.gain = 1;
            } else {
            }
          }
        }
        clear() {
          for (const connection of this.connections.in) {
            connection.eligibility = 0;
            connection.xtrace = { nodes: [], values: [] };
          }
          for (const connection of this.connections.self) {
            connection.eligibility = 0;
            connection.xtrace = { nodes: [], values: [] };
          }
          for (const connection of this.connections.gated) {
            connection.gain = 0;
          }
          this.error = { responsibility: 0, projected: 0, gated: 0 };
          this.old = this.state = this.activation = 0;
        }
        isProjectingTo(node) {
          if (node === this && this.connections.self.length > 0)
            return true;
          return this.connections.out.some((conn) => conn.to === node);
        }
        isProjectedBy(node) {
          if (node === this && this.connections.self.length > 0)
            return true;
          return this.connections.in.some((conn) => conn.from === node);
        }
        applyBatchUpdates(momentum) {
          return this.applyBatchUpdatesWithOptimizer({ type: "sgd", momentum });
        }
        applyBatchUpdatesWithOptimizer(opts) {
          const type = opts.type || "sgd";
          const effectiveType = type === "lookahead" ? opts.baseType || "sgd" : type;
          const momentum = opts.momentum ?? 0;
          const beta1 = opts.beta1 ?? 0.9;
          const beta2 = opts.beta2 ?? 0.999;
          const eps = opts.eps ?? 1e-8;
          const wd = opts.weightDecay ?? 0;
          const lrScale = opts.lrScale ?? 1;
          const t = Math.max(1, Math.floor(opts.t ?? 1));
          if (type === "lookahead") {
            this._la_k = this._la_k || opts.la_k || 5;
            this._la_alpha = this._la_alpha || opts.la_alpha || 0.5;
            this._la_step = (this._la_step || 0) + 1;
            if (!this._la_shadowBias)
              this._la_shadowBias = this.bias;
          }
          const applyConn = (conn) => {
            let g = conn.totalDeltaWeight || 0;
            if (!Number.isFinite(g))
              g = 0;
            switch (effectiveType) {
              case "rmsprop": {
                conn.gradientAccumulator = (conn.gradientAccumulator ?? 0) * 0.9 + 0.1 * (g * g);
                const adj = g / (Math.sqrt(conn.gradientAccumulator) + eps);
                this._safeUpdateWeight(conn, adj * lrScale);
                break;
              }
              case "adagrad": {
                conn.gradientAccumulator = (conn.gradientAccumulator ?? 0) + g * g;
                const adj = g / (Math.sqrt(conn.gradientAccumulator) + eps);
                this._safeUpdateWeight(conn, adj * lrScale);
                break;
              }
              case "adam":
              case "adamw":
              case "amsgrad": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                conn.secondMoment = (conn.secondMoment ?? 0) * beta2 + (1 - beta2) * (g * g);
                if (effectiveType === "amsgrad") {
                  conn.maxSecondMoment = Math.max(conn.maxSecondMoment ?? 0, conn.secondMoment ?? 0);
                }
                const vEff = effectiveType === "amsgrad" ? conn.maxSecondMoment : conn.secondMoment;
                const mHat = conn.firstMoment / (1 - Math.pow(beta1, t));
                const vHat = vEff / (1 - Math.pow(beta2, t));
                let step = mHat / (Math.sqrt(vHat) + eps) * lrScale;
                if (effectiveType === "adamw" && wd !== 0)
                  step -= wd * (conn.weight || 0);
                this._safeUpdateWeight(conn, step);
                break;
              }
              case "adamax": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                conn.infinityNorm = Math.max((conn.infinityNorm ?? 0) * beta2, Math.abs(g));
                const mHat = conn.firstMoment / (1 - Math.pow(beta1, t));
                const stepVal = mHat / (conn.infinityNorm || 1e-12) * lrScale;
                this._safeUpdateWeight(conn, stepVal);
                break;
              }
              case "nadam": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                conn.secondMoment = (conn.secondMoment ?? 0) * beta2 + (1 - beta2) * (g * g);
                const mHat = conn.firstMoment / (1 - Math.pow(beta1, t));
                const vHat = conn.secondMoment / (1 - Math.pow(beta2, t));
                const mNesterov = mHat * beta1 + (1 - beta1) * g / (1 - Math.pow(beta1, t));
                this._safeUpdateWeight(conn, mNesterov / (Math.sqrt(vHat) + eps) * lrScale);
                break;
              }
              case "radam": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                conn.secondMoment = (conn.secondMoment ?? 0) * beta2 + (1 - beta2) * (g * g);
                const mHat = conn.firstMoment / (1 - Math.pow(beta1, t));
                const vHat = conn.secondMoment / (1 - Math.pow(beta2, t));
                const rhoInf = 2 / (1 - beta2) - 1;
                const rhoT = rhoInf - 2 * t * Math.pow(beta2, t) / (1 - Math.pow(beta2, t));
                if (rhoT > 4) {
                  const rt = Math.sqrt((rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT));
                  this._safeUpdateWeight(conn, rt * mHat / (Math.sqrt(vHat) + eps) * lrScale);
                } else {
                  this._safeUpdateWeight(conn, mHat * lrScale);
                }
                break;
              }
              case "lion": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                conn.secondMomentum = (conn.secondMomentum ?? 0) * beta2 + (1 - beta2) * g;
                const update = Math.sign((conn.firstMoment || 0) + (conn.secondMomentum || 0));
                this._safeUpdateWeight(conn, -update * lrScale);
                break;
              }
              case "adabelief": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                const g_m = g - conn.firstMoment;
                conn.secondMoment = (conn.secondMoment ?? 0) * beta2 + (1 - beta2) * (g_m * g_m);
                const mHat = conn.firstMoment / (1 - Math.pow(beta1, t));
                const vHat = conn.secondMoment / (1 - Math.pow(beta2, t));
                this._safeUpdateWeight(conn, mHat / (Math.sqrt(vHat) + eps + 1e-12) * lrScale);
                break;
              }
              default: {
                let currentDeltaWeight = g + momentum * (conn.previousDeltaWeight || 0);
                if (!Number.isFinite(currentDeltaWeight))
                  currentDeltaWeight = 0;
                if (Math.abs(currentDeltaWeight) > 1e3)
                  currentDeltaWeight = Math.sign(currentDeltaWeight) * 1e3;
                this._safeUpdateWeight(conn, currentDeltaWeight * lrScale);
                conn.previousDeltaWeight = currentDeltaWeight;
              }
            }
            if (effectiveType === "adamw" && wd !== 0) {
              this._safeUpdateWeight(conn, -wd * (conn.weight || 0) * lrScale);
            }
            conn.totalDeltaWeight = 0;
          };
          for (const connection of this.connections.in)
            applyConn(connection);
          for (const connection of this.connections.self)
            applyConn(connection);
          if (this.type !== "input" && this.type !== "constant") {
            let gB = this.totalDeltaBias || 0;
            if (!Number.isFinite(gB))
              gB = 0;
            if ([
              "adam",
              "adamw",
              "amsgrad",
              "adamax",
              "nadam",
              "radam",
              "lion",
              "adabelief"
            ].includes(effectiveType)) {
              this.opt_mB = (this.opt_mB ?? 0) * beta1 + (1 - beta1) * gB;
              if (effectiveType === "lion") {
                this.opt_mB2 = (this.opt_mB2 ?? 0) * beta2 + (1 - beta2) * gB;
              }
              this.opt_vB = (this.opt_vB ?? 0) * beta2 + (1 - beta2) * (effectiveType === "adabelief" ? Math.pow(gB - this.opt_mB, 2) : gB * gB);
              if (effectiveType === "amsgrad") {
                this.opt_vhatB = Math.max(this.opt_vhatB ?? 0, this.opt_vB ?? 0);
              }
              const vEffB = effectiveType === "amsgrad" ? this.opt_vhatB : this.opt_vB;
              const mHatB = this.opt_mB / (1 - Math.pow(beta1, t));
              const vHatB = vEffB / (1 - Math.pow(beta2, t));
              let stepB;
              if (effectiveType === "adamax") {
                this.opt_uB = Math.max((this.opt_uB ?? 0) * beta2, Math.abs(gB));
                stepB = mHatB / (this.opt_uB || 1e-12) * lrScale;
              } else if (effectiveType === "nadam") {
                const mNesterovB = mHatB * beta1 + (1 - beta1) * gB / (1 - Math.pow(beta1, t));
                stepB = mNesterovB / (Math.sqrt(vHatB) + eps) * lrScale;
              } else if (effectiveType === "radam") {
                const rhoInf = 2 / (1 - beta2) - 1;
                const rhoT = rhoInf - 2 * t * Math.pow(beta2, t) / (1 - Math.pow(beta2, t));
                if (rhoT > 4) {
                  const rt = Math.sqrt((rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT));
                  stepB = rt * mHatB / (Math.sqrt(vHatB) + eps) * lrScale;
                } else {
                  stepB = mHatB * lrScale;
                }
              } else if (effectiveType === "lion") {
                const updateB = Math.sign(this.opt_mB + this.opt_mB2);
                stepB = -updateB * lrScale;
              } else if (effectiveType === "adabelief") {
                stepB = mHatB / (Math.sqrt(vHatB) + eps + 1e-12) * lrScale;
              } else {
                stepB = mHatB / (Math.sqrt(vHatB) + eps) * lrScale;
              }
              if (effectiveType === "adamw" && wd !== 0)
                stepB -= wd * (this.bias || 0) * lrScale;
              let nextBias = this.bias + stepB;
              if (!Number.isFinite(nextBias))
                nextBias = 0;
              if (Math.abs(nextBias) > 1e6)
                nextBias = Math.sign(nextBias) * 1e6;
              this.bias = nextBias;
            } else {
              let currentDeltaBias = gB + momentum * (this.previousDeltaBias || 0);
              if (!Number.isFinite(currentDeltaBias))
                currentDeltaBias = 0;
              if (Math.abs(currentDeltaBias) > 1e3)
                currentDeltaBias = Math.sign(currentDeltaBias) * 1e3;
              let nextBias = this.bias + currentDeltaBias * lrScale;
              if (!Number.isFinite(nextBias))
                nextBias = 0;
              if (Math.abs(nextBias) > 1e6)
                nextBias = Math.sign(nextBias) * 1e6;
              this.bias = nextBias;
              this.previousDeltaBias = currentDeltaBias;
            }
            this.totalDeltaBias = 0;
          } else {
            this.previousDeltaBias = 0;
            this.totalDeltaBias = 0;
          }
          if (type === "lookahead") {
            const k = this._la_k || 5;
            const alpha = this._la_alpha || 0.5;
            if (this._la_step % k === 0) {
              this._la_shadowBias = (1 - alpha) * this._la_shadowBias + alpha * this.bias;
              this.bias = this._la_shadowBias;
              const blendConn = (conn) => {
                if (!conn.lookaheadShadowWeight)
                  conn.lookaheadShadowWeight = conn.weight;
                conn.lookaheadShadowWeight = (1 - alpha) * conn.lookaheadShadowWeight + alpha * conn.weight;
                conn.weight = conn.lookaheadShadowWeight;
              };
              for (const c of this.connections.in)
                blendConn(c);
              for (const c of this.connections.self)
                blendConn(c);
            }
          }
        }
        _safeUpdateWeight(connection, delta) {
          let next = connection.weight + delta;
          if (!Number.isFinite(next))
            next = 0;
          if (Math.abs(next) > 1e6)
            next = Math.sign(next) * 1e6;
          connection.weight = next;
        }
      };
    }
  });

  // dist/architecture/nodePool.js
  function resetNode(node, type, rng = Math.random) {
    if (type)
      node.type = type;
    const t = node.type;
    node.bias = t === "input" ? 0 : rng() * 0.2 - 0.1;
    node.activation = 0;
    node.state = 0;
    node.old = 0;
    node.mask = 1;
    node.previousDeltaBias = 0;
    node.totalDeltaBias = 0;
    node.derivative = void 0;
    node.connections.in.length = 0;
    node.connections.out.length = 0;
    node.connections.gated.length = 0;
    node.connections.self.length = 0;
    node.error = { responsibility: 0, projected: 0, gated: 0 };
    node.geneId = nextGeneId++;
  }
  function acquireNode(opts = {}) {
    const { type = "hidden", activationFn, rng } = opts;
    let node;
    if (pool.length) {
      node = pool.pop();
      reusedCount++;
      resetNode(node, type, rng);
      if (activationFn)
        node.squash = activationFn;
    } else {
      node = new Node(type, activationFn, rng);
      node.geneId = nextGeneId++;
      freshCount++;
    }
    return node;
  }
  function releaseNode(node) {
    node.connections.in.length = 0;
    node.connections.out.length = 0;
    node.connections.gated.length = 0;
    node.connections.self.length = 0;
    node.error = { responsibility: 0, projected: 0, gated: 0 };
    pool.push(node);
    if (pool.length > highWaterMark)
      highWaterMark = pool.length;
  }
  var pool, highWaterMark, nextGeneId, reusedCount, freshCount;
  var init_nodePool = __esm({
    "dist/architecture/nodePool.js"() {
      "use strict";
      init_node();
      pool = [];
      highWaterMark = 0;
      nextGeneId = 1;
      reusedCount = 0;
      freshCount = 0;
    }
  });

  // dist/multithreading/workers/node/testworker.js
  var testworker_exports = {};
  __export(testworker_exports, {
    TestWorker: () => TestWorker,
    default: () => testworker_default
  });
  var import_child_process, TestWorker, testworker_default;
  var init_testworker = __esm({
    "dist/multithreading/workers/node/testworker.js"() {
      "use strict";
      import_child_process = __require("child_process");
      TestWorker = class {
        worker;
        constructor(dataSet, cost) {
          let pathModule = null;
          try {
            pathModule = __require("path");
          } catch {
          }
          const workerPath = pathModule ? pathModule.join(__dirname, "/worker") : "./worker";
          this.worker = (0, import_child_process.fork)(workerPath);
          this.worker.send({ set: dataSet, cost: cost.name });
        }
        async evaluate(network) {
          const serialized = network.serialize();
          const data = {
            activations: serialized[0],
            states: serialized[1],
            conns: serialized[2]
          };
          return new Promise((resolve, reject) => {
            const onMessage = (e) => {
              cleanup();
              resolve(e);
            };
            const onError = (err) => {
              cleanup();
              reject(err);
            };
            const onExit = (code, signal) => {
              cleanup();
              reject(new Error(`worker exited${code != null ? ` with code ${code}` : signal ? ` with signal ${signal}` : ""}`));
            };
            const cleanup = () => {
              this.worker.off("message", onMessage);
              this.worker.off("error", onError);
              this.worker.off("exit", onExit);
            };
            this.worker.once("message", onMessage);
            this.worker.once("error", onError);
            this.worker.once("exit", onExit);
            this.worker.send(data);
          });
        }
        terminate() {
          this.worker.kill();
        }
      };
      testworker_default = TestWorker;
    }
  });

  // dist/multithreading/workers/browser/testworker.js
  var testworker_exports2 = {};
  __export(testworker_exports2, {
    TestWorker: () => TestWorker2
  });
  var TestWorker2;
  var init_testworker2 = __esm({
    "dist/multithreading/workers/browser/testworker.js"() {
      "use strict";
      init_multi();
      TestWorker2 = class _TestWorker {
        worker;
        url;
        constructor(dataSet, cost) {
          const blob = new Blob([_TestWorker._createBlobString(cost)]);
          this.url = window.URL.createObjectURL(blob);
          this.worker = new Worker(this.url);
          const data = { set: new Float64Array(dataSet).buffer };
          this.worker.postMessage(data, [data.set]);
        }
        evaluate(network) {
          return new Promise((resolve, reject) => {
            const serialized = network.serialize();
            const data = {
              activations: new Float64Array(serialized[0]).buffer,
              states: new Float64Array(serialized[1]).buffer,
              conns: new Float64Array(serialized[2]).buffer
            };
            this.worker.onmessage = function(e) {
              const error = new Float64Array(e.data.buffer)[0];
              resolve(error);
            };
            this.worker.postMessage(data, [
              data.activations,
              data.states,
              data.conns
            ]);
          });
        }
        terminate() {
          this.worker.terminate();
          window.URL.revokeObjectURL(this.url);
        }
        static _createBlobString(cost) {
          return `
      const F = [${Multi.activations.toString()}];
      const cost = ${cost.toString()};
      const multi = {
        deserializeDataSet: ${Multi.deserializeDataSet.toString()},
        testSerializedSet: ${Multi.testSerializedSet.toString()},
        activateSerializedNetwork: ${Multi.activateSerializedNetwork.toString()}
      };

      let set;

      this.onmessage = function (e) {
        if (typeof e.data.set === 'undefined') {
          const A = new Float64Array(e.data.activations);
          const S = new Float64Array(e.data.states);
          const data = new Float64Array(e.data.conns);

          const error = multi.testSerializedSet(set, cost, A, S, data, F);

          const answer = { buffer: new Float64Array([error]).buffer };
          postMessage(answer, [answer.buffer]);
        } else {
          set = multi.deserializeDataSet(new Float64Array(e.data.set));
        }
      };`;
        }
      };
    }
  });

  // dist/multithreading/workers/workers.js
  var Workers;
  var init_workers = __esm({
    "dist/multithreading/workers/workers.js"() {
      "use strict";
      Workers = class {
        static async getNodeTestWorker() {
          const module = await Promise.resolve().then(() => (init_testworker(), testworker_exports));
          return module.TestWorker;
        }
        static async getBrowserTestWorker() {
          const module = await Promise.resolve().then(() => (init_testworker2(), testworker_exports2));
          return module.TestWorker;
        }
      };
    }
  });

  // dist/multithreading/multi.js
  var Multi;
  var init_multi = __esm({
    "dist/multithreading/multi.js"() {
      "use strict";
      init_workers();
      init_network();
      Multi = class _Multi {
        static workers = Workers;
        static activations = [
          (x) => 1 / (1 + Math.exp(-x)),
          (x) => Math.tanh(x),
          (x) => x,
          (x) => x > 0 ? 1 : 0,
          (x) => x > 0 ? x : 0,
          (x) => x / (1 + Math.abs(x)),
          (x) => Math.sin(x),
          (x) => Math.exp(-Math.pow(x, 2)),
          (x) => (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x,
          (x) => x > 0 ? 1 : -1,
          (x) => 2 / (1 + Math.exp(-x)) - 1,
          (x) => Math.max(-1, Math.min(1, x)),
          (x) => Math.abs(x),
          (x) => 1 - x,
          (x) => {
            const alpha = 1.6732632423543772;
            const scale = 1.0507009873554805;
            const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
            return fx * scale;
          },
          (x) => Math.log(1 + Math.exp(x))
        ];
        static serializeDataSet(dataSet) {
          const serialized = [dataSet[0].input.length, dataSet[0].output.length];
          for (let i = 0; i < dataSet.length; i++) {
            for (let j = 0; j < serialized[0]; j++) {
              serialized.push(dataSet[i].input[j]);
            }
            for (let j = 0; j < serialized[1]; j++) {
              serialized.push(dataSet[i].output[j]);
            }
          }
          return serialized;
        }
        static activateSerializedNetwork(input, A, S, data, F) {
          for (let i = 0; i < data[0]; i++)
            A[i] = input[i];
          for (let i = 2; i < data.length; i++) {
            const index = data[i++];
            const bias = data[i++];
            const squash = data[i++];
            const selfweight = data[i++];
            const selfgater = data[i++];
            S[index] = (selfgater === -1 ? 1 : A[selfgater]) * selfweight * S[index] + bias;
            while (data[i] !== -2) {
              S[index] += A[data[i++]] * data[i++] * (data[i++] === -1 ? 1 : A[data[i - 1]]);
            }
            A[index] = F[squash](S[index]);
          }
          const output = [];
          for (let i = A.length - data[1]; i < A.length; i++)
            output.push(A[i]);
          return output;
        }
        static deserializeDataSet(serializedSet) {
          const set = [];
          const sampleSize = serializedSet[0] + serializedSet[1];
          for (let i = 0; i < (serializedSet.length - 2) / sampleSize; i++) {
            const input = [];
            for (let j = 2 + i * sampleSize; j < 2 + i * sampleSize + serializedSet[0]; j++) {
              input.push(serializedSet[j]);
            }
            const output = [];
            for (let j = 2 + i * sampleSize + serializedSet[0]; j < 2 + i * sampleSize + sampleSize; j++) {
              output.push(serializedSet[j]);
            }
            set.push({ input, output });
          }
          return set;
        }
        static logistic(x) {
          return 1 / (1 + Math.exp(-x));
        }
        static tanh(x) {
          return Math.tanh(x);
        }
        static identity(x) {
          return x;
        }
        static step(x) {
          return x > 0 ? 1 : 0;
        }
        static relu(x) {
          return x > 0 ? x : 0;
        }
        static softsign(x) {
          return x / (1 + Math.abs(x));
        }
        static sinusoid(x) {
          return Math.sin(x);
        }
        static gaussian(x) {
          return Math.exp(-Math.pow(x, 2));
        }
        static bentIdentity(x) {
          return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
        }
        static bipolar(x) {
          return x > 0 ? 1 : -1;
        }
        static bipolarSigmoid(x) {
          return 2 / (1 + Math.exp(-x)) - 1;
        }
        static hardTanh(x) {
          return Math.max(-1, Math.min(1, x));
        }
        static absolute(x) {
          return Math.abs(x);
        }
        static inverse(x) {
          return 1 - x;
        }
        static selu(x) {
          const alpha = 1.6732632423543772;
          const scale = 1.0507009873554805;
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          return fx * scale;
        }
        static softplus(x) {
          return Math.log(1 + Math.exp(x));
        }
        static testSerializedSet(set, cost, A, S, data, F) {
          if (set.length === 0)
            return NaN;
          let errorSum = 0;
          for (const sample of set) {
            const output = _Multi.activateSerializedNetwork(sample.input, A, S, data, F);
            const costVal = cost(sample.output, output);
            if (!Number.isFinite(costVal))
              return NaN;
            errorSum += costVal;
          }
          return errorSum / set.length;
        }
        static async getBrowserTestWorker() {
          const { TestWorker: TestWorker3 } = await Promise.resolve().then(() => (init_testworker2(), testworker_exports2));
          return TestWorker3;
        }
        static async getNodeTestWorker() {
          const { TestWorker: TestWorker3 } = await Promise.resolve().then(() => (init_testworker(), testworker_exports));
          return TestWorker3;
        }
      };
    }
  });

  // dist/architecture/activationArrayPool.js
  var ActivationArrayPool, activationArrayPool;
  var init_activationArrayPool = __esm({
    "dist/architecture/activationArrayPool.js"() {
      "use strict";
      init_config();
      ActivationArrayPool = class {
        buckets = /* @__PURE__ */ new Map();
        created = 0;
        reused = 0;
        maxPerBucket = Number.POSITIVE_INFINITY;
        acquire(size) {
          const bucket = this.buckets.get(size);
          if (bucket && bucket.length > 0) {
            this.reused++;
            const arr = bucket.pop();
            if (Array.isArray(arr)) {
              arr.fill(0);
            } else if (arr instanceof Float32Array) {
              arr.fill(0);
            }
            return arr;
          }
          this.created++;
          return config.float32Mode ? new Float32Array(size) : new Array(size).fill(0);
        }
        release(array) {
          const size = array.length >>> 0;
          if (!this.buckets.has(size))
            this.buckets.set(size, []);
          const bucket = this.buckets.get(size);
          if (bucket.length < this.maxPerBucket)
            bucket.push(array);
        }
        clear() {
          this.buckets.clear();
          this.created = 0;
          this.reused = 0;
        }
        stats() {
          return {
            created: this.created,
            reused: this.reused,
            bucketCount: this.buckets.size
          };
        }
        setMaxPerBucket(cap) {
          if (typeof cap === "number" && cap >= 0)
            this.maxPerBucket = cap;
        }
        prewarm(size, count) {
          const n = Math.max(0, Math.floor(count));
          if (!this.buckets.has(size))
            this.buckets.set(size, []);
          const bucket = this.buckets.get(size);
          for (let i = 0; i < n && bucket.length < this.maxPerBucket; i++) {
            const arr = config.float32Mode ? new Float32Array(size) : new Array(size).fill(0);
            bucket.push(arr);
            this.created++;
          }
        }
        bucketSize(size) {
          return this.buckets.get(size)?.length ?? 0;
        }
      };
      activationArrayPool = new ActivationArrayPool();
    }
  });

  // package.json
  var require_package = __commonJS({
    "package.json"(exports, module) {
      module.exports = {
        name: "@reicek/neataptic-ts",
        version: "0.1.15",
        description: "Architecture-free neural network library with genetic algorithm implementations",
        main: "./dist/neataptic.js",
        module: "./dist/neataptic.js",
        types: "./dist/neataptic.d.ts",
        type: "module",
        scripts: {
          test: "jest --config=jest.config.mjs --no-cache --coverage --collect-coverage --runInBand --testPathIgnorePatterns=.e2e.test.ts --verbose",
          pretest: "npm run build",
          "test:bench": "jest --no-cache --runInBand --verbose --testPathPattern=benchmark",
          "bench:asciiMaze": "node -r ts-node/register test/benchmarks/asciiMaze.micro.bench.ts",
          "test:silent": "jest --no-cache --coverage --collect-coverage --runInBand --testPathIgnorePatterns=.e2e.test.ts --silent",
          deploy: "npm run build && npm run test:dist && npm publish",
          build: "npm run build:webpack && npm run build:ts",
          "build:ts": "tsc",
          "build:webpack": "webpack --config webpack.config.js",
          "build:ascii-maze": "npx esbuild test/examples/asciiMaze/browser-entry.ts --bundle --outfile=docs/assets/ascii-maze.bundle.js --platform=browser --format=iife --minify --sourcemap --external:fs --external:child_process --external:path",
          "start:ts": "ts-node src/neataptic.ts",
          "test:e2e": "cross-env FORCE_COLOR=true jest e2e.test.ts --no-cache --runInBand",
          "test:e2e:logs": "npx jest e2e.test.ts --verbose --runInBand --no-cache",
          "test:dist": "npm run build:ts && jest --no-cache --coverage --collect-coverage --runInBand --testPathIgnorePatterns=.e2e.test.ts",
          "docs:build-scripts": "tsc -p tsconfig.docs.json && node scripts/write-dist-docs-pkg.mjs",
          "docs:folders": "npm run docs:build-scripts && node ./dist-docs/scripts/generate-docs.js",
          "docs:html": "npm run docs:build-scripts && node ./dist-docs/scripts/render-docs-html.js",
          "docs:examples": "node scripts/copy-examples.mjs",
          prettier: "npm run prettier:tests && npm run prettier:src",
          "prettier:tests": "npx prettier --write test/**/*.ts",
          "prettier:src": "npx prettier --write src/**/*.ts",
          docs: "npm run build:ascii-maze && npm run docs:examples && npm run docs:build-scripts && node ./dist-docs/scripts/generate-docs.js && node ./dist-docs/scripts/render-docs-html.js",
          lint: "eslint src/ test/",
          "lint:fix": "eslint src/ --fix",
          "onnx:export": "node scripts/export-onnx.mjs"
        },
        exports: {
          ".": {
            types: "./dist/neataptic.d.ts",
            import: "./dist/neataptic.js"
          }
        },
        devDependencies: {
          "@eslint/eslintrc": "^3.3.1",
          "@eslint/js": "^9.35.0",
          "@types/chai": "^5.2.2",
          "@types/fs-extra": "^11.0.4",
          "@types/jest": "^30.0.0",
          "@types/node": "^24.3.0",
          "@types/seedrandom": "^3.0.8",
          "@types/webpack": "^5.28.5",
          "@types/webpack-dev-server": "^4.7.2",
          "@typescript-eslint/eslint-plugin": "^8.42.0",
          "@typescript-eslint/parser": "^8.42.0",
          chai: "^6.0.1",
          "copy-webpack-plugin": "^13.0.1",
          "cross-env": "^10.0.0",
          esbuild: "^0.25.9",
          eslint: "^9.35.0",
          "eslint-plugin-prefer-arrow": "^1.2.3",
          "fast-glob": "^3.3.3",
          "fs-extra": "^11.3.1",
          husky: "^9.1.7",
          jest: "^30.0.5",
          "jest-environment-jsdom": "^30.0.5",
          "jsdoc-to-markdown": "^9.1.2",
          marked: "^16.2.0",
          mkdocs: "^0.0.1",
          puppeteer: "^24.17.0",
          "ts-jest": "^29.4.1",
          "ts-loader": "^9.5.2",
          "ts-morph": "^26.0.0",
          "ts-node": "^10.9.2",
          typescript: "^5.9.2",
          "undici-types": "^7.15.0",
          webpack: "^5.101.3",
          "webpack-cli": "^6.0.1"
        },
        repository: {
          type: "git",
          url: "https://github.com/reicek/NeatapticTS.git"
        },
        keywords: [
          "neural network",
          "machine learning",
          "genetic algorithm",
          "mutation",
          "neat"
        ],
        author: {
          name: "Cesar Anton",
          email: "reicek@gmail.com"
        },
        license: "MIT",
        publishConfig: {
          access: "public",
          registry: "https://registry.npmjs.org/"
        },
        bugs: {
          url: "https://github.com/reicek/NeatapticTS/issues",
          email: "reicek@gmail.com"
        },
        homepage: "https://reicek.github.io/NeatapticTS/",
        engines: {
          node: ">=22.0.0"
        },
        prettier: {
          singleQuote: true
        },
        dependencies: {
          seedrandom: "^3.0.5",
          undici: "^7.15.0"
        }
      };
    }
  });

  // dist/architecture/network/network.onnx.js
  function rebuildConnectionsLocal(networkLike) {
    const uniqueConnections = /* @__PURE__ */ new Set();
    networkLike.nodes.forEach((node) => node.connections?.out.forEach((conn) => uniqueConnections.add(conn)));
    networkLike.connections = Array.from(uniqueConnections);
  }
  function mapActivationToOnnx(squash) {
    const upperName = (squash?.name || "").toUpperCase();
    if (upperName.includes("TANH"))
      return "Tanh";
    if (upperName.includes("LOGISTIC") || upperName.includes("SIGMOID"))
      return "Sigmoid";
    if (upperName.includes("RELU"))
      return "Relu";
    if (squash)
      console.warn(`Unsupported activation function ${squash.name} for ONNX export, defaulting to Identity.`);
    return "Identity";
  }
  function inferLayerOrdering(network) {
    const inputNodes = network.nodes.filter((n) => n.type === "input");
    const outputNodes = network.nodes.filter((n) => n.type === "output");
    const hiddenNodes = network.nodes.filter((n) => n.type === "hidden");
    if (hiddenNodes.length === 0)
      return [inputNodes, outputNodes];
    let remainingHidden = [...hiddenNodes];
    let previousLayer = inputNodes;
    const layerAccumulator = [];
    while (remainingHidden.length) {
      const currentLayer = remainingHidden.filter((hidden) => hidden.connections.in.every((conn) => previousLayer.includes(conn.from)));
      if (!currentLayer.length)
        throw new Error("Invalid network structure for ONNX export: cannot resolve layered ordering.");
      layerAccumulator.push(previousLayer);
      previousLayer = currentLayer;
      remainingHidden = remainingHidden.filter((h) => !currentLayer.includes(h));
    }
    layerAccumulator.push(previousLayer);
    layerAccumulator.push(outputNodes);
    return layerAccumulator;
  }
  function validateLayerHomogeneityAndConnectivity(layers, network, options) {
    for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
      const previousLayerNodes = layers[layerIndex - 1];
      const currentLayerNodes = layers[layerIndex];
      const activationNameSet = new Set(currentLayerNodes.map((n) => n.squash && n.squash.name));
      if (activationNameSet.size > 1 && !options.allowMixedActivations)
        throw new Error(`ONNX export error: Mixed activation functions detected in layer ${layerIndex}. (enable allowMixedActivations to decompose layer)`);
      if (activationNameSet.size > 1 && options.allowMixedActivations)
        console.warn(`Warning: Mixed activations in layer ${layerIndex}; exporting per-neuron Gemm + Activation (+Concat) baseline.`);
      for (const targetNode of currentLayerNodes) {
        for (const sourceNode of previousLayerNodes) {
          const isConnected = targetNode.connections.in.some((conn) => conn.from === sourceNode);
          if (!isConnected && !options.allowPartialConnectivity)
            throw new Error(`ONNX export error: Missing connection from node ${sourceNode.index} to node ${targetNode.index} in layer ${layerIndex}. (enable allowPartialConnectivity)`);
        }
      }
    }
  }
  function buildOnnxModel(network, layers, options = {}) {
    const { includeMetadata = false, opset = 18, batchDimension = false, legacyNodeOrdering = false, producerName = "neataptic-ts", producerVersion, docString } = options;
    const inputLayerNodes = layers[0];
    const outputLayerNodes = layers[layers.length - 1];
    const batchDims = batchDimension ? [{ dim_param: "N" }, { dim_value: inputLayerNodes.length }] : [{ dim_value: inputLayerNodes.length }];
    const outBatchDims = batchDimension ? [{ dim_param: "N" }, { dim_value: outputLayerNodes.length }] : [{ dim_value: outputLayerNodes.length }];
    const model = {
      graph: {
        inputs: [
          {
            name: "input",
            type: {
              tensor_type: {
                elem_type: 1,
                shape: { dim: batchDims }
              }
            }
          }
        ],
        outputs: [
          {
            name: "output",
            type: {
              tensor_type: {
                elem_type: 1,
                shape: { dim: outBatchDims }
              }
            }
          }
        ],
        initializer: [],
        node: []
      }
    };
    if (includeMetadata) {
      const pkgVersion = (() => {
        try {
          return require_package().version;
        } catch {
          return "0.0.0";
        }
      })();
      model.ir_version = 9;
      model.opset_import = [{ version: opset, domain: "" }];
      model.producer_name = producerName;
      model.producer_version = producerVersion || pkgVersion;
      model.doc_string = docString || "Exported from NeatapticTS ONNX exporter (phases 1-2 baseline)";
    }
    let previousOutputName = "input";
    const recurrentLayerIndices = [];
    if (options.allowRecurrent && options.recurrentSingleStep) {
      for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
        const hiddenLayerNodes = layers[layerIndex];
        if (hiddenLayerNodes.some((n) => n.connections.self.length > 0)) {
          recurrentLayerIndices.push(layerIndex);
          const prevName = layerIndex === 1 ? "hidden_prev" : `hidden_prev_l${layerIndex}`;
          model.graph.inputs.push({
            name: prevName,
            type: {
              tensor_type: {
                elem_type: 1,
                shape: {
                  dim: batchDimension ? [{ dim_param: "N" }, { dim_value: hiddenLayerNodes.length }] : [{ dim_value: hiddenLayerNodes.length }]
                }
              }
            }
          });
        }
      }
    }
    const hiddenSizesMetadata = [];
    for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
      const previousLayerNodes = layers[layerIndex - 1];
      const currentLayerNodes = layers[layerIndex];
      const isOutputLayer = layerIndex === layers.length - 1;
      if (!isOutputLayer)
        hiddenSizesMetadata.push(currentLayerNodes.length);
      const convSpec = options.conv2dMappings?.find((m) => m.layerIndex === layerIndex);
      if (convSpec) {
        const prevWidthExpected = convSpec.inHeight * convSpec.inWidth * convSpec.inChannels;
        const prevWidthActual = previousLayerNodes.length;
        const thisWidthExpected = convSpec.outChannels * convSpec.outHeight * convSpec.outWidth;
        const thisWidthActual = currentLayerNodes.length;
        const pads = [
          convSpec.padTop || 0,
          convSpec.padLeft || 0,
          convSpec.padBottom || 0,
          convSpec.padRight || 0
        ];
        const shapeValid = prevWidthExpected === prevWidthActual && thisWidthExpected === thisWidthActual;
        if (!shapeValid) {
          console.warn(`Conv2D mapping for layer ${layerIndex} skipped: dimension mismatch (expected prev=${prevWidthExpected} got ${prevWidthActual}; expected this=${thisWidthExpected} got ${thisWidthActual}).`);
        } else {
          const W = [];
          const B = [];
          for (let oc = 0; oc < convSpec.outChannels; oc++) {
            const repIndex = oc * convSpec.outHeight * convSpec.outWidth;
            const repNeuron = currentLayerNodes[repIndex];
            B.push(repNeuron.bias);
            for (let ic = 0; ic < convSpec.inChannels; ic++) {
              for (let kh = 0; kh < convSpec.kernelHeight; kh++) {
                for (let kw = 0; kw < convSpec.kernelWidth; kw++) {
                  const inputFeatureIndex = ic * (convSpec.inHeight * convSpec.inWidth) + kh * convSpec.inWidth + kw;
                  const sourceNode = previousLayerNodes[inputFeatureIndex];
                  const conn = repNeuron.connections.in.find((cc) => cc.from === sourceNode);
                  W.push(conn ? conn.weight : 0);
                }
              }
            }
          }
          const convWName = `ConvW${layerIndex - 1}`;
          const convBName = `ConvB${layerIndex - 1}`;
          model.graph.initializer.push({
            name: convWName,
            data_type: 1,
            dims: [
              convSpec.outChannels,
              convSpec.inChannels,
              convSpec.kernelHeight,
              convSpec.kernelWidth
            ],
            float_data: W
          });
          model.graph.initializer.push({
            name: convBName,
            data_type: 1,
            dims: [convSpec.outChannels],
            float_data: B
          });
          const convOut = `Conv_${layerIndex}`;
          model.graph.node.push({
            op_type: "Conv",
            input: [previousOutputName, convWName, convBName],
            output: [convOut],
            name: `conv_l${layerIndex}`,
            attributes: [
              {
                name: "kernel_shape",
                type: "INTS",
                ints: [convSpec.kernelHeight, convSpec.kernelWidth]
              },
              {
                name: "strides",
                type: "INTS",
                ints: [convSpec.strideHeight, convSpec.strideWidth]
              },
              { name: "pads", type: "INTS", ints: pads }
            ]
          });
          const actOp = convSpec.activation || mapActivationToOnnx(currentLayerNodes[0].squash);
          const activationOutputName = `Layer_${layerIndex}`;
          model.graph.node.push({
            op_type: actOp,
            input: [convOut],
            output: [activationOutputName],
            name: `act_conv_l${layerIndex}`
          });
          previousOutputName = activationOutputName;
          const poolSpecPostConv = options.pool2dMappings?.find((p) => p.afterLayerIndex === layerIndex);
          if (poolSpecPostConv) {
            const kernel = [
              poolSpecPostConv.kernelHeight,
              poolSpecPostConv.kernelWidth
            ];
            const strides = [
              poolSpecPostConv.strideHeight,
              poolSpecPostConv.strideWidth
            ];
            const pads2 = [
              poolSpecPostConv.padTop || 0,
              poolSpecPostConv.padLeft || 0,
              poolSpecPostConv.padBottom || 0,
              poolSpecPostConv.padRight || 0
            ];
            const poolOut = `Pool_${layerIndex}`;
            model.graph.node.push({
              op_type: poolSpecPostConv.type,
              input: [previousOutputName],
              output: [poolOut],
              name: `pool_after_l${layerIndex}`,
              attributes: [
                { name: "kernel_shape", type: "INTS", ints: kernel },
                { name: "strides", type: "INTS", ints: strides },
                { name: "pads", type: "INTS", ints: pads2 }
              ]
            });
            previousOutputName = poolOut;
            if (options.flattenAfterPooling) {
              const flatOut = `PoolFlat_${layerIndex}`;
              model.graph.node.push({
                op_type: "Flatten",
                input: [previousOutputName],
                output: [flatOut],
                name: `flatten_after_l${layerIndex}`,
                attributes: [{ name: "axis", type: "INT", i: 1 }]
              });
              previousOutputName = flatOut;
              model.metadata_props = model.metadata_props || [];
              const flMeta = model.metadata_props.find((m) => m.key === "flatten_layers");
              if (flMeta) {
                try {
                  const arr = JSON.parse(flMeta.value);
                  if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                    arr.push(layerIndex);
                    flMeta.value = JSON.stringify(arr);
                  }
                } catch {
                  flMeta.value = JSON.stringify([layerIndex]);
                }
              } else {
                model.metadata_props.push({
                  key: "flatten_layers",
                  value: JSON.stringify([layerIndex])
                });
              }
            }
            model.metadata_props = model.metadata_props || [];
            const poolLayersMeta = model.metadata_props.find((m) => m.key === "pool2d_layers");
            if (poolLayersMeta) {
              try {
                const arr = JSON.parse(poolLayersMeta.value);
                if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                  arr.push(layerIndex);
                  poolLayersMeta.value = JSON.stringify(arr);
                }
              } catch {
                poolLayersMeta.value = JSON.stringify([layerIndex]);
              }
            } else {
              model.metadata_props.push({
                key: "pool2d_layers",
                value: JSON.stringify([layerIndex])
              });
            }
            const poolSpecsMeta = model.metadata_props.find((m) => m.key === "pool2d_specs");
            if (poolSpecsMeta) {
              try {
                const arr = JSON.parse(poolSpecsMeta.value);
                if (Array.isArray(arr)) {
                  arr.push({ ...poolSpecPostConv });
                  poolSpecsMeta.value = JSON.stringify(arr);
                }
              } catch {
                poolSpecsMeta.value = JSON.stringify([poolSpecPostConv]);
              }
            } else {
              model.metadata_props.push({
                key: "pool2d_specs",
                value: JSON.stringify([poolSpecPostConv])
              });
            }
          }
          model.metadata_props = model.metadata_props || [];
          const convLayersMeta = model.metadata_props.find((m) => m.key === "conv2d_layers");
          if (convLayersMeta) {
            try {
              const arr = JSON.parse(convLayersMeta.value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                convLayersMeta.value = JSON.stringify(arr);
              }
            } catch {
              convLayersMeta.value = JSON.stringify([layerIndex]);
            }
          } else {
            model.metadata_props.push({
              key: "conv2d_layers",
              value: JSON.stringify([layerIndex])
            });
          }
          const convSpecsMeta = model.metadata_props.find((m) => m.key === "conv2d_specs");
          if (convSpecsMeta) {
            try {
              const arr = JSON.parse(convSpecsMeta.value);
              if (Array.isArray(arr)) {
                arr.push({ ...convSpec });
                convSpecsMeta.value = JSON.stringify(arr);
              }
            } catch {
              convSpecsMeta.value = JSON.stringify([convSpec]);
            }
          } else {
            model.metadata_props.push({
              key: "conv2d_specs",
              value: JSON.stringify([convSpec])
            });
          }
          continue;
        }
      }
      const mixed = options.allowMixedActivations && new Set(currentLayerNodes.map((n) => n.squash && n.squash.name)).size > 1;
      if (recurrentLayerIndices.includes(layerIndex) && !isOutputLayer) {
        if (mixed)
          throw new Error(`Recurrent export does not yet support mixed activations in hidden layer ${layerIndex}.`);
        const weightMatrixValues = [];
        const biasVector = new Array(currentLayerNodes.length).fill(0);
        for (let r = 0; r < currentLayerNodes.length; r++) {
          const targetNode = currentLayerNodes[r];
          biasVector[r] = targetNode.bias;
          for (let c = 0; c < previousLayerNodes.length; c++) {
            const sourceNode = previousLayerNodes[c];
            const inboundConn = targetNode.connections.in.find((conn) => conn.from === sourceNode);
            weightMatrixValues.push(inboundConn ? inboundConn.weight : 0);
          }
        }
        const weightTensorName = `W${layerIndex - 1}`;
        const biasTensorName = `B${layerIndex - 1}`;
        model.graph.initializer.push({
          name: weightTensorName,
          data_type: 1,
          dims: [currentLayerNodes.length, previousLayerNodes.length],
          float_data: weightMatrixValues
        });
        model.graph.initializer.push({
          name: biasTensorName,
          data_type: 1,
          dims: [currentLayerNodes.length],
          float_data: biasVector
        });
        const recurrentWeights = [];
        for (let r = 0; r < currentLayerNodes.length; r++) {
          for (let c = 0; c < currentLayerNodes.length; c++) {
            if (r === c) {
              const selfConn = currentLayerNodes[r].connections.self[0];
              recurrentWeights.push(selfConn ? selfConn.weight : 0);
            } else {
              recurrentWeights.push(0);
            }
          }
        }
        const rName = `R${layerIndex - 1}`;
        model.graph.initializer.push({
          name: rName,
          data_type: 1,
          dims: [currentLayerNodes.length, currentLayerNodes.length],
          float_data: recurrentWeights
        });
        model.graph.node.push({
          op_type: "Gemm",
          input: [previousOutputName, weightTensorName, biasTensorName],
          output: [`Gemm_in_${layerIndex}`],
          name: `gemm_in_l${layerIndex}`,
          attributes: [
            { name: "alpha", type: "FLOAT", f: 1 },
            { name: "beta", type: "FLOAT", f: 1 },
            { name: "transB", type: "INT", i: 1 }
          ]
        });
        const prevHiddenInputName = layerIndex === 1 ? "hidden_prev" : `hidden_prev_l${layerIndex}`;
        model.graph.node.push({
          op_type: "Gemm",
          input: [prevHiddenInputName, rName],
          output: [`Gemm_rec_${layerIndex}`],
          name: `gemm_rec_l${layerIndex}`,
          attributes: [
            { name: "alpha", type: "FLOAT", f: 1 },
            { name: "beta", type: "FLOAT", f: 1 },
            { name: "transB", type: "INT", i: 1 }
          ]
        });
        model.graph.node.push({
          op_type: "Add",
          input: [`Gemm_in_${layerIndex}`, `Gemm_rec_${layerIndex}`],
          output: [`RecurrentSum_${layerIndex}`],
          name: `add_recurrent_l${layerIndex}`
        });
        model.graph.node.push({
          op_type: mapActivationToOnnx(currentLayerNodes[0].squash),
          input: [`RecurrentSum_${layerIndex}`],
          output: [`Layer_${layerIndex}`],
          name: `act_l${layerIndex}`
        });
        previousOutputName = `Layer_${layerIndex}`;
      } else if (!mixed) {
        const weightMatrixValues = [];
        const biasVector = new Array(currentLayerNodes.length).fill(0);
        for (let r = 0; r < currentLayerNodes.length; r++) {
          const targetNode = currentLayerNodes[r];
          biasVector[r] = targetNode.bias;
          for (let c = 0; c < previousLayerNodes.length; c++) {
            const sourceNode = previousLayerNodes[c];
            const inboundConn = targetNode.connections.in.find((conn) => conn.from === sourceNode);
            weightMatrixValues.push(inboundConn ? inboundConn.weight : 0);
          }
        }
        const weightTensorName = `W${layerIndex - 1}`;
        const biasTensorName = `B${layerIndex - 1}`;
        const gemmOutputName = `Gemm_${layerIndex}`;
        const activationOutputName = `Layer_${layerIndex}`;
        model.graph.initializer.push({
          name: weightTensorName,
          data_type: 1,
          dims: [currentLayerNodes.length, previousLayerNodes.length],
          float_data: weightMatrixValues
        });
        model.graph.initializer.push({
          name: biasTensorName,
          data_type: 1,
          dims: [currentLayerNodes.length],
          float_data: biasVector
        });
        if (!legacyNodeOrdering) {
          model.graph.node.push({
            op_type: "Gemm",
            input: [previousOutputName, weightTensorName, biasTensorName],
            output: [gemmOutputName],
            name: `gemm_l${layerIndex}`,
            attributes: [
              { name: "alpha", type: "FLOAT", f: 1 },
              { name: "beta", type: "FLOAT", f: 1 },
              { name: "transB", type: "INT", i: 1 }
            ]
          });
          model.graph.node.push({
            op_type: mapActivationToOnnx(currentLayerNodes[0].squash),
            input: [gemmOutputName],
            output: [activationOutputName],
            name: `act_l${layerIndex}`
          });
        } else {
          model.graph.node.push({
            op_type: mapActivationToOnnx(currentLayerNodes[0].squash),
            input: [gemmOutputName],
            output: [activationOutputName],
            name: `act_l${layerIndex}`
          });
          model.graph.node.push({
            op_type: "Gemm",
            input: [previousOutputName, weightTensorName, biasTensorName],
            output: [gemmOutputName],
            name: `gemm_l${layerIndex}`,
            attributes: [
              { name: "alpha", type: "FLOAT", f: 1 },
              { name: "beta", type: "FLOAT", f: 1 },
              { name: "transB", type: "INT", i: 1 }
            ]
          });
        }
        previousOutputName = activationOutputName;
        const poolSpecDense = options.pool2dMappings?.find((p) => p.afterLayerIndex === layerIndex);
        if (poolSpecDense) {
          const kernel = [poolSpecDense.kernelHeight, poolSpecDense.kernelWidth];
          const strides = [poolSpecDense.strideHeight, poolSpecDense.strideWidth];
          const pads = [
            poolSpecDense.padTop || 0,
            poolSpecDense.padLeft || 0,
            poolSpecDense.padBottom || 0,
            poolSpecDense.padRight || 0
          ];
          const poolOut = `Pool_${layerIndex}`;
          model.graph.node.push({
            op_type: poolSpecDense.type,
            input: [previousOutputName],
            output: [poolOut],
            name: `pool_after_l${layerIndex}`,
            attributes: [
              { name: "kernel_shape", type: "INTS", ints: kernel },
              { name: "strides", type: "INTS", ints: strides },
              { name: "pads", type: "INTS", ints: pads }
            ]
          });
          previousOutputName = poolOut;
          if (options.flattenAfterPooling) {
            const flatOut = `PoolFlat_${layerIndex}`;
            model.graph.node.push({
              op_type: "Flatten",
              input: [previousOutputName],
              output: [flatOut],
              name: `flatten_after_l${layerIndex}`,
              attributes: [{ name: "axis", type: "INT", i: 1 }]
            });
            previousOutputName = flatOut;
            model.metadata_props = model.metadata_props || [];
            const flMeta = model.metadata_props.find((m) => m.key === "flatten_layers");
            if (flMeta) {
              try {
                const arr = JSON.parse(flMeta.value);
                if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                  arr.push(layerIndex);
                  flMeta.value = JSON.stringify(arr);
                }
              } catch {
                flMeta.value = JSON.stringify([layerIndex]);
              }
            } else {
              model.metadata_props.push({
                key: "flatten_layers",
                value: JSON.stringify([layerIndex])
              });
            }
          }
          model.metadata_props = model.metadata_props || [];
          const poolLayersMeta = model.metadata_props.find((m) => m.key === "pool2d_layers");
          if (poolLayersMeta) {
            try {
              const arr = JSON.parse(poolLayersMeta.value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                poolLayersMeta.value = JSON.stringify(arr);
              }
            } catch {
              poolLayersMeta.value = JSON.stringify([layerIndex]);
            }
          } else {
            model.metadata_props.push({
              key: "pool2d_layers",
              value: JSON.stringify([layerIndex])
            });
          }
          const poolSpecsMeta = model.metadata_props.find((m) => m.key === "pool2d_specs");
          if (poolSpecsMeta) {
            try {
              const arr = JSON.parse(poolSpecsMeta.value);
              if (Array.isArray(arr)) {
                arr.push({ ...poolSpecDense });
                poolSpecsMeta.value = JSON.stringify(arr);
              }
            } catch {
              poolSpecsMeta.value = JSON.stringify([poolSpecDense]);
            }
          } else {
            model.metadata_props.push({
              key: "pool2d_specs",
              value: JSON.stringify([poolSpecDense])
            });
          }
        }
      } else {
        const perNeuronActivationOutputs = [];
        currentLayerNodes.forEach((targetNode, idx) => {
          const weightRow = [];
          for (let c = 0; c < previousLayerNodes.length; c++) {
            const sourceNode = previousLayerNodes[c];
            const inboundConn = targetNode.connections.in.find((conn) => conn.from === sourceNode);
            weightRow.push(inboundConn ? inboundConn.weight : 0);
          }
          const weightTensorName = `W${layerIndex - 1}_n${idx}`;
          const biasTensorName = `B${layerIndex - 1}_n${idx}`;
          const gemmOutputName = `Gemm_${layerIndex}_n${idx}`;
          const actOutputName = `Layer_${layerIndex}_n${idx}`;
          model.graph.initializer.push({
            name: weightTensorName,
            data_type: 1,
            dims: [1, previousLayerNodes.length],
            float_data: weightRow
          });
          model.graph.initializer.push({
            name: biasTensorName,
            data_type: 1,
            dims: [1],
            float_data: [targetNode.bias]
          });
          model.graph.node.push({
            op_type: "Gemm",
            input: [previousOutputName, weightTensorName, biasTensorName],
            output: [gemmOutputName],
            name: `gemm_l${layerIndex}_n${idx}`,
            attributes: [
              { name: "alpha", type: "FLOAT", f: 1 },
              { name: "beta", type: "FLOAT", f: 1 },
              { name: "transB", type: "INT", i: 1 }
            ]
          });
          model.graph.node.push({
            op_type: mapActivationToOnnx(targetNode.squash),
            input: [gemmOutputName],
            output: [actOutputName],
            name: `act_l${layerIndex}_n${idx}`
          });
          perNeuronActivationOutputs.push(actOutputName);
        });
        const activationOutputName = `Layer_${layerIndex}`;
        model.graph.node.push({
          op_type: "Concat",
          input: perNeuronActivationOutputs,
          output: [activationOutputName],
          name: `concat_l${layerIndex}`,
          attributes: [{ name: "axis", type: "INT", i: batchDimension ? 1 : 0 }]
        });
        previousOutputName = activationOutputName;
        const poolSpecPerNeuron = options.pool2dMappings?.find((p) => p.afterLayerIndex === layerIndex);
        if (poolSpecPerNeuron) {
          const kernel = [
            poolSpecPerNeuron.kernelHeight,
            poolSpecPerNeuron.kernelWidth
          ];
          const strides = [
            poolSpecPerNeuron.strideHeight,
            poolSpecPerNeuron.strideWidth
          ];
          const pads = [
            poolSpecPerNeuron.padTop || 0,
            poolSpecPerNeuron.padLeft || 0,
            poolSpecPerNeuron.padBottom || 0,
            poolSpecPerNeuron.padRight || 0
          ];
          const poolOut = `Pool_${layerIndex}`;
          model.graph.node.push({
            op_type: poolSpecPerNeuron.type,
            input: [previousOutputName],
            output: [poolOut],
            name: `pool_after_l${layerIndex}`,
            attributes: [
              { name: "kernel_shape", type: "INTS", ints: kernel },
              { name: "strides", type: "INTS", ints: strides },
              { name: "pads", type: "INTS", ints: pads }
            ]
          });
          previousOutputName = poolOut;
          if (options.flattenAfterPooling) {
            const flatOut = `PoolFlat_${layerIndex}`;
            model.graph.node.push({
              op_type: "Flatten",
              input: [previousOutputName],
              output: [flatOut],
              name: `flatten_after_l${layerIndex}`,
              attributes: [{ name: "axis", type: "INT", i: 1 }]
            });
            previousOutputName = flatOut;
            model.metadata_props = model.metadata_props || [];
            const flMeta = model.metadata_props.find((m) => m.key === "flatten_layers");
            if (flMeta) {
              try {
                const arr = JSON.parse(flMeta.value);
                if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                  arr.push(layerIndex);
                  flMeta.value = JSON.stringify(arr);
                }
              } catch {
                flMeta.value = JSON.stringify([layerIndex]);
              }
            } else {
              model.metadata_props.push({
                key: "flatten_layers",
                value: JSON.stringify([layerIndex])
              });
            }
          }
          model.metadata_props = model.metadata_props || [];
          const poolLayersMeta = model.metadata_props.find((m) => m.key === "pool2d_layers");
          if (poolLayersMeta) {
            try {
              const arr = JSON.parse(poolLayersMeta.value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                poolLayersMeta.value = JSON.stringify(arr);
              }
            } catch {
              poolLayersMeta.value = JSON.stringify([layerIndex]);
            }
          } else {
            model.metadata_props.push({
              key: "pool2d_layers",
              value: JSON.stringify([layerIndex])
            });
          }
          const poolSpecsMeta = model.metadata_props.find((m) => m.key === "pool2d_specs");
          if (poolSpecsMeta) {
            try {
              const arr = JSON.parse(poolSpecsMeta.value);
              if (Array.isArray(arr)) {
                arr.push({ ...poolSpecPerNeuron });
                poolSpecsMeta.value = JSON.stringify(arr);
              }
            } catch {
              poolSpecsMeta.value = JSON.stringify([poolSpecPerNeuron]);
            }
          } else {
            model.metadata_props.push({
              key: "pool2d_specs",
              value: JSON.stringify([poolSpecPerNeuron])
            });
          }
        }
      }
    }
    if (options.allowRecurrent) {
      for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
        const current = layers[layerIndex];
        const size = current.length;
        if (!model.metadata_props)
          model.metadata_props = [];
        if (size >= 8 && size < 10) {
          model.metadata_props.push({
            key: "rnn_pattern_fallback",
            value: JSON.stringify({
              layer: layerIndex,
              reason: "size_between_gru_lstm_thresholds"
            })
          });
        }
        if (size >= 10 && size % 5 === 0) {
          const unit = size / 5;
          const prevLayerNodes = layers[layerIndex - 1];
          const inputGate = current.slice(0, unit);
          const forgetGate = current.slice(unit, unit * 2);
          const cell = current.slice(unit * 2, unit * 3);
          const outputGate = current.slice(unit * 3, unit * 4);
          const outputBlock = current.slice(unit * 4, unit * 5);
          const gateOrder = [inputGate, forgetGate, cell, outputGate];
          const numGates = gateOrder.length;
          const prevSize = prevLayerNodes.length;
          const W = [];
          const R = [];
          const B = [];
          for (let g = 0; g < numGates; g++) {
            const gate2 = gateOrder[g];
            for (let r = 0; r < unit; r++) {
              const neuron = gate2[r];
              for (let c = 0; c < prevSize; c++) {
                const source = prevLayerNodes[c];
                const conn = neuron.connections.in.find((cc) => cc.from === source);
                W.push(conn ? conn.weight : 0);
              }
              for (let c = 0; c < unit; c++) {
                if (gate2 === cell && c === r) {
                  const selfConn = neuron.connections.self[0];
                  R.push(selfConn ? selfConn.weight : 0);
                } else
                  R.push(0);
              }
              B.push(neuron.bias);
            }
          }
          model.graph.initializer.push({
            name: `LSTM_W${layerIndex - 1}`,
            data_type: 1,
            dims: [numGates * unit, prevSize],
            float_data: W
          });
          model.graph.initializer.push({
            name: `LSTM_R${layerIndex - 1}`,
            data_type: 1,
            dims: [numGates * unit, unit],
            float_data: R
          });
          model.graph.initializer.push({
            name: `LSTM_B${layerIndex - 1}`,
            data_type: 1,
            dims: [numGates * unit],
            float_data: B
          });
          model.graph.node.push({
            op_type: "LSTM",
            input: [
              previousOutputName,
              `LSTM_W${layerIndex - 1}`,
              `LSTM_R${layerIndex - 1}`,
              `LSTM_B${layerIndex - 1}`
            ],
            output: [`Layer_${layerIndex}_lstm_hidden`],
            name: `lstm_l${layerIndex}`,
            attributes: [
              { name: "hidden_size", type: "INT", i: unit },
              { name: "layout", type: "INT", i: 0 }
            ]
          });
          model.metadata_props = model.metadata_props || [];
          const lstmMetaIdx = model.metadata_props.findIndex((m) => m.key === "lstm_emitted_layers");
          if (lstmMetaIdx >= 0) {
            try {
              const arr = JSON.parse(model.metadata_props[lstmMetaIdx].value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                model.metadata_props[lstmMetaIdx].value = JSON.stringify(arr);
              }
            } catch {
              model.metadata_props[lstmMetaIdx].value = JSON.stringify([
                layerIndex
              ]);
            }
          } else {
            model.metadata_props.push({
              key: "lstm_emitted_layers",
              value: JSON.stringify([layerIndex])
            });
          }
        }
        if (size >= 8 && size % 4 === 0) {
          const unitG = size / 4;
          const prevLayerNodes = layers[layerIndex - 1];
          const updateGate = current.slice(0, unitG);
          const resetGate = current.slice(unitG, unitG * 2);
          const candidate = current.slice(unitG * 2, unitG * 3);
          const outputBlock = current.slice(unitG * 3, unitG * 4);
          const gateOrderGRU = [updateGate, resetGate, candidate];
          const numGatesGRU = gateOrderGRU.length;
          const prevSizeGRU = prevLayerNodes.length;
          const Wg = [];
          const Rg = [];
          const Bg = [];
          for (let g = 0; g < numGatesGRU; g++) {
            const gate2 = gateOrderGRU[g];
            for (let r = 0; r < unitG; r++) {
              const neuron = gate2[r];
              for (let c = 0; c < prevSizeGRU; c++) {
                const src = prevLayerNodes[c];
                const conn = neuron.connections.in.find((cc) => cc.from === src);
                Wg.push(conn ? conn.weight : 0);
              }
              for (let c = 0; c < unitG; c++) {
                if (gate2 === candidate && c === r) {
                  const selfConn = neuron.connections.self[0];
                  Rg.push(selfConn ? selfConn.weight : 0);
                } else
                  Rg.push(0);
              }
              Bg.push(neuron.bias);
            }
          }
          model.graph.initializer.push({
            name: `GRU_W${layerIndex - 1}`,
            data_type: 1,
            dims: [numGatesGRU * unitG, prevSizeGRU],
            float_data: Wg
          });
          model.graph.initializer.push({
            name: `GRU_R${layerIndex - 1}`,
            data_type: 1,
            dims: [numGatesGRU * unitG, unitG],
            float_data: Rg
          });
          model.graph.initializer.push({
            name: `GRU_B${layerIndex - 1}`,
            data_type: 1,
            dims: [numGatesGRU * unitG],
            float_data: Bg
          });
          const prevOutName = layerIndex === 1 ? "input" : `Layer_${layerIndex - 1}`;
          model.graph.node.push({
            op_type: "GRU",
            input: [
              prevOutName,
              `GRU_W${layerIndex - 1}`,
              `GRU_R${layerIndex - 1}`,
              `GRU_B${layerIndex - 1}`
            ],
            output: [`Layer_${layerIndex}_gru_hidden`],
            name: `gru_l${layerIndex}`,
            attributes: [
              { name: "hidden_size", type: "INT", i: unitG },
              { name: "layout", type: "INT", i: 0 }
            ]
          });
          model.metadata_props = model.metadata_props || [];
          const gruMetaIdx = model.metadata_props.findIndex((m) => m.key === "gru_emitted_layers");
          if (gruMetaIdx >= 0) {
            try {
              const arr = JSON.parse(model.metadata_props[gruMetaIdx].value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                model.metadata_props[gruMetaIdx].value = JSON.stringify(arr);
              }
            } catch {
              model.metadata_props[gruMetaIdx].value = JSON.stringify([
                layerIndex
              ]);
            }
          } else {
            model.metadata_props.push({
              key: "gru_emitted_layers",
              value: JSON.stringify([layerIndex])
            });
          }
        }
      }
    }
    if (includeMetadata) {
      model.metadata_props = model.metadata_props || [];
      model.metadata_props.push({
        key: "layer_sizes",
        value: JSON.stringify(hiddenSizesMetadata)
      });
      if (recurrentLayerIndices.length) {
        model.metadata_props.push({
          key: "recurrent_single_step",
          value: JSON.stringify(recurrentLayerIndices)
        });
      }
      if (options.validateConvSharing && options.conv2dMappings && options.conv2dMappings.length) {
        const verified = [];
        const mismatched = [];
        for (const spec of options.conv2dMappings) {
          const layerIdx = spec.layerIndex;
          const prevLayerNodes = layers[layerIdx - 1];
          const layerNodes = layers[layerIdx];
          if (!layerNodes || !prevLayerNodes)
            continue;
          const repPerChannel = [];
          let allOk = true;
          for (let oc = 0; oc < spec.outChannels; oc++) {
            const repIndex = oc * (spec.outHeight * spec.outWidth);
            const repNeuron = layerNodes[repIndex];
            const kernel = [];
            for (let ic = 0; ic < spec.inChannels; ic++) {
              for (let kh = 0; kh < spec.kernelHeight; kh++) {
                for (let kw = 0; kw < spec.kernelWidth; kw++) {
                  const inputFeatureIndex = ic * (spec.inHeight * spec.inWidth) + kh * spec.inWidth + kw;
                  const sourceNode = prevLayerNodes[inputFeatureIndex];
                  const conn = repNeuron.connections.in.find((cc) => cc.from === sourceNode);
                  kernel.push(conn ? conn.weight : 0);
                }
              }
            }
            repPerChannel.push(kernel);
          }
          const tol = 1e-9;
          for (let oc = 0; oc < spec.outChannels && allOk; oc++) {
            for (let oh = 0; oh < spec.outHeight && allOk; oh++) {
              for (let ow = 0; ow < spec.outWidth && allOk; ow++) {
                const idx = oc * (spec.outHeight * spec.outWidth) + oh * spec.outWidth + ow;
                const neuron = layerNodes[idx];
                if (!neuron)
                  continue;
                let kPtr = 0;
                for (let ic = 0; ic < spec.inChannels && allOk; ic++) {
                  const hBase = oh * spec.strideHeight - (spec.padTop || 0);
                  const wBase = ow * spec.strideWidth - (spec.padLeft || 0);
                  for (let kh = 0; kh < spec.kernelHeight && allOk; kh++) {
                    for (let kw = 0; kw < spec.kernelWidth && allOk; kw++) {
                      const ih = hBase + kh;
                      const iw = wBase + kw;
                      if (ih < 0 || ih >= spec.inHeight || iw < 0 || iw >= spec.inWidth) {
                        kPtr++;
                        continue;
                      }
                      const inputFeatureIndex = ic * (spec.inHeight * spec.inWidth) + ih * spec.inWidth + iw;
                      const srcNode = prevLayerNodes[inputFeatureIndex];
                      const conn = neuron.connections.in.find((cc) => cc.from === srcNode);
                      const wVal = conn ? conn.weight : 0;
                      if (Math.abs(wVal - repPerChannel[oc][kPtr]) > tol) {
                        allOk = false;
                      }
                      kPtr++;
                    }
                  }
                }
                if (!allOk)
                  break;
              }
            }
          }
          if (allOk)
            verified.push(layerIdx);
          else {
            mismatched.push(layerIdx);
            console.warn(`Conv2D weight sharing mismatch detected in layer ${layerIdx}`);
          }
        }
        if (verified.length)
          model.metadata_props.push({
            key: "conv2d_sharing_verified",
            value: JSON.stringify(verified)
          });
        if (mismatched.length)
          model.metadata_props.push({
            key: "conv2d_sharing_mismatch",
            value: JSON.stringify(mismatched)
          });
      }
    }
    return model;
  }
  function exportToONNX(network, options = {}) {
    rebuildConnectionsLocal(network);
    network.nodes.forEach((node, idx) => node.index = idx);
    if (!network.connections || network.connections.length === 0)
      throw new Error("ONNX export currently only supports simple MLPs");
    const layers = inferLayerOrdering(network);
    const lstmPatternStubs = [];
    if (options.allowRecurrent) {
      try {
        for (let li = 1; li < layers.length - 1; li++) {
          const hiddenLayer = layers[li];
          const total = hiddenLayer.length;
          if (total >= 10 && total % 5 === 0) {
            const seg = total / 5;
            const memorySlice = hiddenLayer.slice(seg * 2, seg * 3);
            const allSelf = memorySlice.every((n) => n.connections.self.length === 1);
            if (allSelf) {
              lstmPatternStubs.push({ layerIndex: li, unitSize: seg });
            }
          }
        }
      } catch {
      }
    }
    validateLayerHomogeneityAndConnectivity(layers, network, options);
    const model = buildOnnxModel(network, layers, options);
    if (options.includeMetadata) {
      const inferredSpecs = [];
      const inferredLayers = [];
      for (let li = 1; li < layers.length - 1; li++) {
        const prevWidth = layers[li - 1].length;
        const currWidth = layers[li].length;
        const s = Math.sqrt(prevWidth);
        if (Math.abs(s - Math.round(s)) > 1e-9)
          continue;
        const sInt = Math.round(s);
        for (const k of [3, 2]) {
          if (k >= sInt)
            continue;
          const outSpatial = sInt - k + 1;
          if (outSpatial * outSpatial === currWidth) {
            const alreadyDeclared = options.conv2dMappings?.some((m) => m.layerIndex === li);
            if (alreadyDeclared)
              break;
            inferredLayers.push(li);
            inferredSpecs.push({
              layerIndex: li,
              inHeight: sInt,
              inWidth: sInt,
              inChannels: 1,
              kernelHeight: k,
              kernelWidth: k,
              strideHeight: 1,
              strideWidth: 1,
              outHeight: outSpatial,
              outWidth: outSpatial,
              outChannels: 1,
              note: "heuristic_inferred_no_export_applied"
            });
            break;
          }
        }
      }
      if (inferredLayers.length) {
        model.metadata_props = model.metadata_props || [];
        model.metadata_props.push({
          key: "conv2d_inferred_layers",
          value: JSON.stringify(inferredLayers)
        });
        model.metadata_props.push({
          key: "conv2d_inferred_specs",
          value: JSON.stringify(inferredSpecs)
        });
      }
    }
    if (lstmPatternStubs.length) {
      model.metadata_props = model.metadata_props || [];
      model.metadata_props.push({
        key: "lstm_groups_stub",
        value: JSON.stringify(lstmPatternStubs)
      });
    }
    return model;
  }
  var init_network_onnx = __esm({
    "dist/architecture/network/network.onnx.js"() {
      "use strict";
      init_methods();
      init_connection();
    }
  });

  // dist/architecture/onnx.js
  var init_onnx = __esm({
    "dist/architecture/onnx.js"() {
      "use strict";
      init_network_onnx();
      init_network_onnx();
    }
  });

  // dist/architecture/network/network.standalone.js
  function generateStandalone(net) {
    if (!net.nodes.some((nodeRef) => nodeRef.type === "output")) {
      throw new Error("Cannot create standalone function: network has no output nodes.");
    }
    const emittedActivationSource = {};
    const activationFunctionSources = [];
    const activationFunctionIndexMap = {};
    let nextActivationFunctionIndex = 0;
    const initialActivations = [];
    const initialStates = [];
    const bodyLines = [];
    const builtinActivationSnippets = {
      logistic: "function logistic(x){ return 1 / (1 + Math.exp(-x)); }",
      tanh: "function tanh(x){ return Math.tanh(x); }",
      relu: "function relu(x){ return x > 0 ? x : 0; }",
      identity: "function identity(x){ return x; }",
      step: "function step(x){ return x > 0 ? 1 : 0; }",
      softsign: "function softsign(x){ return x / (1 + Math.abs(x)); }",
      sinusoid: "function sinusoid(x){ return Math.sin(x); }",
      gaussian: "function gaussian(x){ return Math.exp(-Math.pow(x, 2)); }",
      bentIdentity: "function bentIdentity(x){ return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x; }",
      bipolar: "function bipolar(x){ return x > 0 ? 1 : -1; }",
      bipolarSigmoid: "function bipolarSigmoid(x){ return 2 / (1 + Math.exp(-x)) - 1; }",
      hardTanh: "function hardTanh(x){ return Math.max(-1, Math.min(1, x)); }",
      absolute: "function absolute(x){ return Math.abs(x); }",
      inverse: "function inverse(x){ return 1 - x; }",
      selu: "function selu(x){ var a=1.6732632423543772,s=1.0507009873554805; var fx=x>0?x:a*Math.exp(x)-a; return fx*s; }",
      softplus: "function softplus(x){ if(x>30)return x; if(x<-30)return Math.exp(x); return Math.max(0,x)+Math.log(1+Math.exp(-Math.abs(x))); }",
      swish: "function swish(x){ var s=1/(1+Math.exp(-x)); return x*s; }",
      gelu: "function gelu(x){ var cdf=0.5*(1.0+Math.tanh(Math.sqrt(2.0/Math.PI)*(x+0.044715*Math.pow(x,3)))); return x*cdf; }",
      mish: "function mish(x){ var sp_x; if(x>30){sp_x=x;}else if(x<-30){sp_x=Math.exp(x);}else{sp_x=Math.log(1+Math.exp(x));} var tanh_sp_x=Math.tanh(sp_x); return x*tanh_sp_x; }"
    };
    net.nodes.forEach((node, nodeIndex) => {
      node.index = nodeIndex;
      initialActivations.push(node.activation);
      initialStates.push(node.state);
    });
    bodyLines.push("for(var i = 0; i < input.length; i++) A[i] = input[i];");
    for (let nodeIndex = net.input; nodeIndex < net.nodes.length; nodeIndex++) {
      const node = net.nodes[nodeIndex];
      const squashFn = node.squash;
      const squashName = squashFn.name || `anonymous_squash_${nodeIndex}`;
      if (!(squashName in emittedActivationSource)) {
        let functionSource;
        if (builtinActivationSnippets[squashName]) {
          functionSource = builtinActivationSnippets[squashName];
          if (!functionSource.startsWith(`function ${squashName}`)) {
            functionSource = `function ${squashName}${functionSource.substring(functionSource.indexOf("("))}`;
          }
          functionSource = stripCoverage(functionSource);
        } else {
          functionSource = squashFn.toString();
          functionSource = stripCoverage(functionSource);
          if (functionSource.startsWith("function")) {
            functionSource = `function ${squashName}${functionSource.substring(functionSource.indexOf("("))}`;
          } else if (functionSource.includes("=>")) {
            functionSource = `function ${squashName}${functionSource.substring(functionSource.indexOf("("))}`;
          } else {
            functionSource = `function ${squashName}(x){ return x; }`;
          }
        }
        emittedActivationSource[squashName] = functionSource;
        activationFunctionSources.push(functionSource);
        activationFunctionIndexMap[squashName] = nextActivationFunctionIndex++;
      }
      const activationFunctionIndex = activationFunctionIndexMap[squashName];
      const incomingTerms = [];
      for (const connection of node.connections.in) {
        if (typeof connection.from.index === "undefined")
          continue;
        let term = `A[${connection.from.index}] * ${connection.weight}`;
        if (connection.gater && typeof connection.gater.index !== "undefined") {
          term += ` * A[${connection.gater.index}]`;
        }
        incomingTerms.push(term);
      }
      if (node.connections.self.length > 0) {
        const selfConn = node.connections.self[0];
        let term = `S[${nodeIndex}] * ${selfConn.weight}`;
        if (selfConn.gater && typeof selfConn.gater.index !== "undefined") {
          term += ` * A[${selfConn.gater.index}]`;
        }
        incomingTerms.push(term);
      }
      const sumExpression = incomingTerms.length > 0 ? incomingTerms.join(" + ") : "0";
      bodyLines.push(`S[${nodeIndex}] = ${sumExpression} + ${node.bias};`);
      const maskValue = typeof node.mask === "number" && node.mask !== 1 ? node.mask : 1;
      bodyLines.push(`A[${nodeIndex}] = F[${activationFunctionIndex}](S[${nodeIndex}])${maskValue !== 1 ? ` * ${maskValue}` : ""};`);
    }
    const outputIndices = [];
    for (let nodeIndex = net.nodes.length - net.output; nodeIndex < net.nodes.length; nodeIndex++) {
      if (typeof net.nodes[nodeIndex]?.index !== "undefined") {
        outputIndices.push(net.nodes[nodeIndex].index);
      }
    }
    bodyLines.push(`return [${outputIndices.map((idx) => `A[${idx}]`).join(",")}];`);
    const activationArrayLiteral = Object.entries(activationFunctionIndexMap).sort(([, a], [, b]) => a - b).map(([name]) => name).join(",");
    const activationArrayType = net._activationPrecision === "f32" ? "Float32Array" : "Float64Array";
    let generatedSource = "";
    generatedSource += `(function(){
`;
    generatedSource += `${activationFunctionSources.join("\n")}
`;
    generatedSource += `var F = [${activationArrayLiteral}];
`;
    generatedSource += `var A = new ${activationArrayType}([${initialActivations.join(",")}]);
`;
    generatedSource += `var S = new ${activationArrayType}([${initialStates.join(",")}]);
`;
    generatedSource += `function activate(input){
`;
    generatedSource += `if (!input || input.length !== ${net.input}) { throw new Error('Invalid input size. Expected ${net.input}, got ' + (input ? input.length : 'undefined')); }
`;
    generatedSource += bodyLines.join("\n");
    generatedSource += `}
`;
    generatedSource += `return activate;
})();`;
    return generatedSource;
  }
  var stripCoverage;
  var init_network_standalone = __esm({
    "dist/architecture/network/network.standalone.js"() {
      "use strict";
      stripCoverage = (code) => {
        code = code.replace(/\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g, "");
        code = code.replace(/cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g, "");
        code = code.replace(/cov_[\w$]+\(\)/g, "");
        code = code.replace(/^\s*\/\/ # sourceMappingURL=.*\s*$/gm, "");
        code = code.replace(/\(\s*,\s*/g, "( ");
        code = code.replace(/\s*,\s*\)/g, " )");
        code = code.trim();
        code = code.replace(/^\s*;\s*$/gm, "");
        code = code.replace(/;{2,}/g, ";");
        code = code.replace(/^\s*[,;]?\s*$/gm, "");
        return code;
      };
    }
  });

  // dist/architecture/network/network.topology.js
  function computeTopoOrder() {
    const internalNet = this;
    if (!internalNet._enforceAcyclic) {
      internalNet._topoOrder = null;
      internalNet._topoDirty = false;
      return;
    }
    const inDegree = /* @__PURE__ */ new Map();
    this.nodes.forEach((node) => inDegree.set(node, 0));
    for (const connection of this.connections) {
      if (connection.from !== connection.to) {
        inDegree.set(connection.to, (inDegree.get(connection.to) || 0) + 1);
      }
    }
    const processingQueue = [];
    this.nodes.forEach((node) => {
      if (node.type === "input" || (inDegree.get(node) || 0) === 0) {
        processingQueue.push(node);
      }
    });
    const topoOrder = [];
    while (processingQueue.length) {
      const node = processingQueue.shift();
      topoOrder.push(node);
      for (const outgoing of node.connections.out) {
        if (outgoing.to === node)
          continue;
        const remaining = (inDegree.get(outgoing.to) || 0) - 1;
        inDegree.set(outgoing.to, remaining);
        if (remaining === 0)
          processingQueue.push(outgoing.to);
      }
    }
    internalNet._topoOrder = topoOrder.length === this.nodes.length ? topoOrder : this.nodes.slice();
    internalNet._topoDirty = false;
  }
  function hasPath(from, to) {
    if (from === to)
      return true;
    const visited = /* @__PURE__ */ new Set();
    const dfsStack = [from];
    while (dfsStack.length) {
      const current = dfsStack.pop();
      if (current === to)
        return true;
      if (visited.has(current))
        continue;
      visited.add(current);
      for (const edge of current.connections.out) {
        if (edge.to !== current)
          dfsStack.push(edge.to);
      }
    }
    return false;
  }
  var init_network_topology = __esm({
    "dist/architecture/network/network.topology.js"() {
      "use strict";
    }
  });

  // dist/architecture/network/network.slab.js
  function _slabPoolCap() {
    const configuredCap = config.slabPoolMaxPerKey;
    if (configuredCap === void 0)
      return 4;
    return configuredCap < 0 ? 0 : configuredCap | 0;
  }
  function _poolKey(kind, bytes, length) {
    return kind + ":" + bytes + ":" + length;
  }
  function _acquireTA(kind, ctor, length, bytesPerElement) {
    if (!config.enableSlabArrayPooling) {
      _slabAllocStats.fresh++;
      return new ctor(length);
    }
    const key = _poolKey(kind, bytesPerElement, length);
    const list = _slabArrayPool[key];
    if (list && list.length) {
      _slabAllocStats.pooled++;
      (_slabPoolMetrics[key] ||= { created: 0, reused: 0, maxRetained: 0 }).reused++;
      return list.pop();
    }
    _slabAllocStats.fresh++;
    (_slabPoolMetrics[key] ||= { created: 0, reused: 0, maxRetained: 0 }).created++;
    return new ctor(length);
  }
  function _releaseTA(kind, bytesPerElement, arr) {
    if (!config.enableSlabArrayPooling)
      return;
    const key = _poolKey(kind, bytesPerElement, arr.length);
    const list = _slabArrayPool[key] ||= [];
    if (list.length < _slabPoolCap())
      list.push(arr);
    const m = _slabPoolMetrics[key] ||= {
      created: 0,
      reused: 0,
      maxRetained: 0
    };
    if (list.length > m.maxRetained)
      m.maxRetained = list.length;
  }
  function rebuildConnectionSlab(force = false) {
    const internalNet = this;
    if (!force && !internalNet._slabDirty)
      return;
    if (internalNet._nodeIndexDirty)
      _reindexNodes.call(this);
    const connectionCount = this.connections.length;
    let capacity = internalNet._connCapacity || 0;
    const growthFactor = typeof window === "undefined" ? 1.75 : 1.25;
    const needAllocate = capacity < connectionCount;
    if (needAllocate) {
      capacity = capacity === 0 ? Math.ceil(connectionCount * growthFactor) : capacity;
      while (capacity < connectionCount)
        capacity = Math.ceil(capacity * growthFactor);
      if (internalNet._connWeights)
        _releaseTA("w", internalNet._useFloat32Weights ? 4 : 8, internalNet._connWeights);
      if (internalNet._connFrom)
        _releaseTA("f", 4, internalNet._connFrom);
      if (internalNet._connTo)
        _releaseTA("t", 4, internalNet._connTo);
      if (internalNet._connFlags)
        _releaseTA("fl", 1, internalNet._connFlags);
      if (internalNet._connGain)
        _releaseTA("g", internalNet._useFloat32Weights ? 4 : 8, internalNet._connGain);
      if (internalNet._connPlastic)
        _releaseTA("p", internalNet._useFloat32Weights ? 4 : 8, internalNet._connPlastic);
      internalNet._connWeights = _acquireTA("w", internalNet._useFloat32Weights ? Float32Array : Float64Array, capacity, internalNet._useFloat32Weights ? 4 : 8);
      internalNet._connFrom = _acquireTA("f", Uint32Array, capacity, 4);
      internalNet._connTo = _acquireTA("t", Uint32Array, capacity, 4);
      internalNet._connFlags = _acquireTA("fl", Uint8Array, capacity, 1);
      internalNet._connGain = null;
      internalNet._connPlastic = null;
      internalNet._connCapacity = capacity;
    } else {
      capacity = internalNet._connCapacity;
    }
    const weightArray = internalNet._connWeights;
    const fromIndexArray = internalNet._connFrom;
    const toIndexArray = internalNet._connTo;
    const flagArray = internalNet._connFlags;
    let gainArray = internalNet._connGain;
    let anyNonNeutralGain = false;
    let plasticArray = internalNet._connPlastic;
    let anyPlastic = false;
    for (let connectionIndex = 0; connectionIndex < connectionCount; connectionIndex++) {
      const connection = this.connections[connectionIndex];
      weightArray[connectionIndex] = connection.weight;
      fromIndexArray[connectionIndex] = connection.from.index >>> 0;
      toIndexArray[connectionIndex] = connection.to.index >>> 0;
      flagArray[connectionIndex] = connection._flags & 255;
      const gainValue = connection.gain;
      if (gainValue !== 1) {
        if (!gainArray) {
          gainArray = _acquireTA("g", internalNet._useFloat32Weights ? Float32Array : Float64Array, capacity, internalNet._useFloat32Weights ? 4 : 8);
          internalNet._connGain = gainArray;
          for (let j = 0; j < connectionIndex; j++)
            gainArray[j] = 1;
        }
        gainArray[connectionIndex] = gainValue;
        anyNonNeutralGain = true;
      } else if (gainArray) {
        gainArray[connectionIndex] = 1;
      }
      if (connection._flags & 8)
        anyPlastic = true;
    }
    if (!anyNonNeutralGain && gainArray) {
      _releaseTA("g", internalNet._useFloat32Weights ? 4 : 8, gainArray);
      internalNet._connGain = null;
    }
    if (anyPlastic && !plasticArray) {
      plasticArray = _acquireTA("p", internalNet._useFloat32Weights ? Float32Array : Float64Array, capacity, internalNet._useFloat32Weights ? 4 : 8);
      internalNet._connPlastic = plasticArray;
      for (let i = 0; i < connectionCount; i++) {
        const c = this.connections[i];
        plasticArray[i] = c.plasticityRate || 0;
      }
    } else if (!anyPlastic && plasticArray) {
      _releaseTA("p", internalNet._useFloat32Weights ? 4 : 8, plasticArray);
      internalNet._connPlastic = null;
    }
    internalNet._connCount = connectionCount;
    internalNet._slabDirty = false;
    internalNet._adjDirty = true;
    internalNet._slabVersion = (internalNet._slabVersion || 0) + 1;
  }
  function getConnectionSlab() {
    rebuildConnectionSlab.call(this);
    const internalNet = this;
    let gain = internalNet._connGain || null;
    if (!gain) {
      const cap = internalNet._connCapacity || internalNet._connWeights && internalNet._connWeights.length || 0;
      gain = internalNet._useFloat32Weights ? new Float32Array(cap) : new Float64Array(cap);
      for (let i = 0; i < (internalNet._connCount || 0); i++)
        gain[i] = 1;
    }
    return {
      weights: internalNet._connWeights,
      from: internalNet._connFrom,
      to: internalNet._connTo,
      flags: internalNet._connFlags,
      gain,
      plastic: internalNet._connPlastic || null,
      version: internalNet._slabVersion || 0,
      used: internalNet._connCount || 0,
      capacity: internalNet._connCapacity || internalNet._connWeights && internalNet._connWeights.length || 0
    };
  }
  function _reindexNodes() {
    const internalNet = this;
    for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++)
      this.nodes[nodeIndex].index = nodeIndex;
    internalNet._nodeIndexDirty = false;
  }
  function _buildAdjacency() {
    const internalNet = this;
    if (!internalNet._connFrom || !internalNet._connTo)
      return;
    const nodeCount = this.nodes.length;
    const connectionCount = internalNet._connCount ?? internalNet._connFrom.length;
    const fanOutCounts = new Uint32Array(nodeCount);
    for (let connectionIndex = 0; connectionIndex < connectionCount; connectionIndex++) {
      fanOutCounts[internalNet._connFrom[connectionIndex]]++;
    }
    const outgoingStartIndices = new Uint32Array(nodeCount + 1);
    let runningOffset = 0;
    for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
      outgoingStartIndices[nodeIndex] = runningOffset;
      runningOffset += fanOutCounts[nodeIndex];
    }
    outgoingStartIndices[nodeCount] = runningOffset;
    const outgoingOrder = new Uint32Array(connectionCount);
    const insertionCursor = outgoingStartIndices.slice();
    for (let connectionIndex = 0; connectionIndex < connectionCount; connectionIndex++) {
      const fromNodeIndex = internalNet._connFrom[connectionIndex];
      outgoingOrder[insertionCursor[fromNodeIndex]++] = connectionIndex;
    }
    internalNet._outStart = outgoingStartIndices;
    internalNet._outOrder = outgoingOrder;
    internalNet._adjDirty = false;
  }
  function _canUseFastSlab(training) {
    const internalNet = this;
    return !training && internalNet._enforceAcyclic && !internalNet._topoDirty && this.gates.length === 0 && this.selfconns.length === 0 && this.dropout === 0 && internalNet._weightNoiseStd === 0 && internalNet._weightNoisePerHidden.length === 0 && internalNet._stochasticDepth.length === 0;
  }
  function fastSlabActivate(input) {
    const internalNet = this;
    rebuildConnectionSlab.call(this);
    if (internalNet._adjDirty)
      _buildAdjacency.call(this);
    if (this.gates && this.gates.length > 0)
      return this.activate(input, false);
    if (!internalNet._connWeights || !internalNet._connFrom || !internalNet._connTo || !internalNet._outStart || !internalNet._outOrder) {
      return this.activate(input, false);
    }
    if (internalNet._topoDirty)
      this._computeTopoOrder();
    if (internalNet._nodeIndexDirty)
      _reindexNodes.call(this);
    const topoOrder = internalNet._topoOrder || this.nodes;
    const nodeCount = this.nodes.length;
    const useFloat32Activation = internalNet._activationPrecision === "f32";
    if (!internalNet._fastA || internalNet._fastA.length !== nodeCount || useFloat32Activation && !(internalNet._fastA instanceof Float32Array) || !useFloat32Activation && !(internalNet._fastA instanceof Float64Array)) {
      internalNet._fastA = useFloat32Activation ? new Float32Array(nodeCount) : new Float64Array(nodeCount);
    }
    if (!internalNet._fastS || internalNet._fastS.length !== nodeCount || useFloat32Activation && !(internalNet._fastS instanceof Float32Array) || !useFloat32Activation && !(internalNet._fastS instanceof Float64Array)) {
      internalNet._fastS = useFloat32Activation ? new Float32Array(nodeCount) : new Float64Array(nodeCount);
    }
    const activationBuffer = internalNet._fastA;
    const stateBuffer = internalNet._fastS;
    stateBuffer.fill(0);
    for (let inputIndex = 0; inputIndex < this.input; inputIndex++) {
      activationBuffer[inputIndex] = input[inputIndex];
      this.nodes[inputIndex].activation = input[inputIndex];
      this.nodes[inputIndex].state = 0;
    }
    const weightArray = internalNet._connWeights;
    const toIndexArray = internalNet._connTo;
    const outgoingOrder = internalNet._outOrder;
    const outgoingStartIndices = internalNet._outStart;
    for (let topoIdx = 0; topoIdx < topoOrder.length; topoIdx++) {
      const node = topoOrder[topoIdx];
      const nodeIndex = node.index >>> 0;
      if (nodeIndex >= this.input) {
        const weightedSum = stateBuffer[nodeIndex] + node.bias;
        const activated = node.squash(weightedSum);
        node.state = stateBuffer[nodeIndex];
        node.activation = activated;
        activationBuffer[nodeIndex] = activated;
      }
      const edgeStart = outgoingStartIndices[nodeIndex];
      const edgeEnd = outgoingStartIndices[nodeIndex + 1];
      const sourceActivation = activationBuffer[nodeIndex];
      for (let cursorIdx = edgeStart; cursorIdx < edgeEnd; cursorIdx++) {
        const connectionIndex = outgoingOrder[cursorIdx];
        let w = weightArray[connectionIndex];
        const gainArr = internalNet._connGain;
        if (gainArr)
          w *= gainArr[connectionIndex];
        stateBuffer[toIndexArray[connectionIndex]] += sourceActivation * w;
      }
    }
    const outputBaseIndex = nodeCount - this.output;
    const pooledOutputArray = activationArrayPool.acquire(this.output);
    for (let outputOffset = 0; outputOffset < this.output; outputOffset++) {
      pooledOutputArray[outputOffset] = activationBuffer[outputBaseIndex + outputOffset];
    }
    const result = Array.from(pooledOutputArray);
    activationArrayPool.release(pooledOutputArray);
    return result;
  }
  function canUseFastSlab(training) {
    return _canUseFastSlab.call(this, training);
  }
  var _slabArrayPool, _slabPoolMetrics, _slabAllocStats;
  var init_network_slab = __esm({
    "dist/architecture/network/network.slab.js"() {
      "use strict";
      init_activationArrayPool();
      init_config();
      _slabArrayPool = /* @__PURE__ */ Object.create(null);
      _slabPoolMetrics = /* @__PURE__ */ Object.create(null);
      _slabAllocStats = { fresh: 0, pooled: 0 };
    }
  });

  // dist/architecture/network/network.prune.js
  function rankConnections(conns, method) {
    const ranked = [...conns];
    if (method === "snip") {
      ranked.sort((a, b) => {
        const gradMagA = Math.abs(a.totalDeltaWeight) || Math.abs(a.previousDeltaWeight) || 0;
        const gradMagB = Math.abs(b.totalDeltaWeight) || Math.abs(b.previousDeltaWeight) || 0;
        const saliencyA = gradMagA ? Math.abs(a.weight) * gradMagA : Math.abs(a.weight);
        const saliencyB = gradMagB ? Math.abs(b.weight) * gradMagB : Math.abs(b.weight);
        return saliencyA - saliencyB;
      });
    } else {
      ranked.sort((a, b) => Math.abs(a.weight) - Math.abs(b.weight));
    }
    return ranked;
  }
  function regrowConnections(network, desiredRemaining, maxAttempts) {
    const netAny = network;
    let attempts = 0;
    while (network.connections.length < desiredRemaining && attempts < maxAttempts) {
      attempts++;
      const fromNode = network.nodes[Math.floor(netAny._rand() * network.nodes.length)];
      const toNode = network.nodes[Math.floor(netAny._rand() * network.nodes.length)];
      if (!fromNode || !toNode || fromNode === toNode)
        continue;
      if (network.connections.some((c) => c.from === fromNode && c.to === toNode))
        continue;
      if (netAny._enforceAcyclic && network.nodes.indexOf(fromNode) > network.nodes.indexOf(toNode))
        continue;
      network.connect(fromNode, toNode);
    }
  }
  function maybePrune(iteration) {
    const cfg = this._pruningConfig;
    if (!cfg)
      return;
    if (iteration < cfg.start || iteration > cfg.end)
      return;
    if (cfg.lastPruneIter != null && iteration === cfg.lastPruneIter)
      return;
    if ((iteration - cfg.start) % (cfg.frequency || 1) !== 0)
      return;
    const initialConnectionBaseline = this._initialConnectionCount;
    if (!initialConnectionBaseline)
      return;
    const progressFraction = (iteration - cfg.start) / Math.max(1, cfg.end - cfg.start);
    const targetSparsityNow = cfg.targetSparsity * Math.min(1, Math.max(0, progressFraction));
    const desiredRemainingConnections = Math.max(1, Math.floor(initialConnectionBaseline * (1 - targetSparsityNow)));
    const excessConnectionCount = this.connections.length - desiredRemainingConnections;
    if (excessConnectionCount <= 0) {
      cfg.lastPruneIter = iteration;
      return;
    }
    const rankedConnections = rankConnections(this.connections, cfg.method || "magnitude");
    const connectionsToPrune = rankedConnections.slice(0, excessConnectionCount);
    connectionsToPrune.forEach((conn) => this.disconnect(conn.from, conn.to));
    if (cfg.regrowFraction && cfg.regrowFraction > 0) {
      const intendedRegrowCount = Math.floor(connectionsToPrune.length * cfg.regrowFraction);
      regrowConnections(this, desiredRemainingConnections, intendedRegrowCount * 10);
    }
    cfg.lastPruneIter = iteration;
    this._topoDirty = true;
  }
  function pruneToSparsity(targetSparsity, method = "magnitude") {
    if (targetSparsity <= 0)
      return;
    if (targetSparsity >= 1)
      targetSparsity = 0.999;
    const netAny = this;
    if (!netAny._evoInitialConnCount)
      netAny._evoInitialConnCount = this.connections.length;
    const evolutionaryBaseline = netAny._evoInitialConnCount;
    const desiredRemainingConnections = Math.max(1, Math.floor(evolutionaryBaseline * (1 - targetSparsity)));
    const excessConnectionCount = this.connections.length - desiredRemainingConnections;
    if (excessConnectionCount <= 0)
      return;
    const rankedConnections = rankConnections(this.connections, method);
    const connectionsToRemove = rankedConnections.slice(0, excessConnectionCount);
    connectionsToRemove.forEach((c) => this.disconnect(c.from, c.to));
    netAny._topoDirty = true;
  }
  function getCurrentSparsity() {
    const initialBaseline = this._initialConnectionCount;
    if (!initialBaseline)
      return 0;
    return 1 - this.connections.length / initialBaseline;
  }
  var init_network_prune = __esm({
    "dist/architecture/network/network.prune.js"() {
      "use strict";
      init_node();
      init_connection();
    }
  });

  // dist/architecture/network/network.gating.js
  function gate(node, connection) {
    if (!this.nodes.includes(node))
      throw new Error("Gating node must be part of the network to gate a connection!");
    if (connection.gater) {
      if (config.warnings)
        console.warn("Connection is already gated. Skipping.");
      return;
    }
    node.gate(connection);
    this.gates.push(connection);
  }
  function ungate(connection) {
    const index = this.gates.indexOf(connection);
    if (index === -1) {
      if (config.warnings)
        console.warn("Attempted to ungate a connection not in the gates list.");
      return;
    }
    this.gates.splice(index, 1);
    connection.gater?.ungate(connection);
  }
  var init_network_gating = __esm({
    "dist/architecture/network/network.gating.js"() {
      "use strict";
      init_node();
      init_connection();
      init_mutation();
      init_config();
    }
  });

  // dist/architecture/network/network.deterministic.js
  function setSeed(seed) {
    this._rngState = seed >>> 0;
    this._rand = () => {
      this._rngState = this._rngState + 1831565813 >>> 0;
      let r = Math.imul(this._rngState ^ this._rngState >>> 15, 1 | this._rngState);
      r ^= r + Math.imul(r ^ r >>> 7, 61 | r);
      return ((r ^ r >>> 14) >>> 0) / 4294967296;
    };
  }
  function snapshotRNG() {
    return { step: this._trainingStep, state: this._rngState };
  }
  function restoreRNG(fn) {
    this._rand = fn;
    this._rngState = void 0;
  }
  function getRNGState() {
    return this._rngState;
  }
  function setRNGState(state) {
    if (typeof state === "number")
      this._rngState = state >>> 0;
  }
  var init_network_deterministic = __esm({
    "dist/architecture/network/network.deterministic.js"() {
      "use strict";
    }
  });

  // dist/architecture/network/network.stats.js
  function deepCloneValue(value) {
    try {
      return globalThis.structuredClone ? globalThis.structuredClone(value) : JSON.parse(JSON.stringify(value));
    } catch {
      return JSON.parse(JSON.stringify(value));
    }
  }
  function getRegularizationStats() {
    const lastStatsSnapshot = this._lastStats;
    return lastStatsSnapshot ? deepCloneValue(lastStatsSnapshot) : null;
  }
  var init_network_stats = __esm({
    "dist/architecture/network/network.stats.js"() {
      "use strict";
    }
  });

  // dist/architecture/network/network.remove.js
  function removeNode(node) {
    const internalNet = this;
    const idx = this.nodes.indexOf(node);
    if (idx === -1)
      throw new Error("Node not in network");
    if (node.type === "input" || node.type === "output") {
      throw new Error("Cannot remove input or output node from the network.");
    }
    this.gates = this.gates.filter((c) => {
      if (c.gater === node) {
        c.gater = null;
        return false;
      }
      return true;
    });
    const inbound = node.connections.in.slice();
    const outbound = node.connections.out.slice();
    inbound.forEach((c) => this.disconnect(c.from, c.to));
    outbound.forEach((c) => this.disconnect(c.from, c.to));
    node.connections.self.slice().forEach(() => this.disconnect(node, node));
    const removed = this.nodes.splice(idx, 1)[0];
    if (config.enableNodePooling && removed) {
      releaseNode(removed);
    }
    inbound.forEach((ic) => {
      outbound.forEach((oc) => {
        if (!ic.from || !oc.to || ic.from === oc.to)
          return;
        const exists = this.connections.some((c) => c.from === ic.from && c.to === oc.to);
        if (!exists)
          this.connect(ic.from, oc.to);
      });
    });
    internalNet._topoDirty = true;
    internalNet._nodeIndexDirty = true;
    internalNet._slabDirty = true;
    internalNet._adjDirty = true;
  }
  var init_network_remove = __esm({
    "dist/architecture/network/network.remove.js"() {
      "use strict";
      init_nodePool();
      init_config();
    }
  });

  // dist/architecture/network/network.connect.js
  function connect(from, to, weight) {
    if (this._enforceAcyclic && this.nodes.indexOf(from) > this.nodes.indexOf(to))
      return [];
    const connections = from.connect(to, weight);
    for (const c of connections) {
      if (from !== to) {
        this.connections.push(c);
      } else {
        if (this._enforceAcyclic)
          continue;
        this.selfconns.push(c);
      }
    }
    if (connections.length) {
      this._topoDirty = true;
      this._slabDirty = true;
    }
    return connections;
  }
  function disconnect(from, to) {
    const list = from === to ? this.selfconns : this.connections;
    for (let i = 0; i < list.length; i++) {
      const c = list[i];
      if (c.from === from && c.to === to) {
        if (c.gater)
          this.ungate(c);
        list.splice(i, 1);
        break;
      }
    }
    from.disconnect(to);
    this._topoDirty = true;
    this._slabDirty = true;
  }
  var init_network_connect = __esm({
    "dist/architecture/network/network.connect.js"() {
      "use strict";
      init_node();
      init_connection();
    }
  });

  // dist/architecture/network/network.serialize.js
  function serialize() {
    this.nodes.forEach((nodeRef, nodeIndex) => nodeRef.index = nodeIndex);
    const activations = this.nodes.map((nodeRef) => nodeRef.activation);
    const states = this.nodes.map((nodeRef) => nodeRef.state);
    const squashes = this.nodes.map((nodeRef) => nodeRef.squash.name);
    const serializedConnections = this.connections.concat(this.selfconns).map((connInstance) => ({
      from: connInstance.from.index,
      to: connInstance.to.index,
      weight: connInstance.weight,
      gater: connInstance.gater ? connInstance.gater.index : null
    }));
    const inputSize = this.input;
    const outputSize = this.output;
    return [
      activations,
      states,
      squashes,
      serializedConnections,
      inputSize,
      outputSize
    ];
  }
  function deserialize(data, inputSize, outputSize) {
    const [activations, states, squashes, connections, serializedInput, serializedOutput] = data;
    const input = typeof inputSize === "number" ? inputSize : serializedInput || 0;
    const output = typeof outputSize === "number" ? outputSize : serializedOutput || 0;
    const net = new (init_network(), __toCommonJS(network_exports)).default(input, output);
    net.nodes = [];
    net.connections = [];
    net.selfconns = [];
    net.gates = [];
    activations.forEach((activation, nodeIndex) => {
      let type;
      if (nodeIndex < input)
        type = "input";
      else if (nodeIndex >= activations.length - output)
        type = "output";
      else
        type = "hidden";
      const node = new Node(type);
      node.activation = activation;
      node.state = states[nodeIndex];
      const squashName = squashes[nodeIndex];
      if (!activation_default[squashName]) {
        console.warn(`Unknown squash function '${String(squashName)}' encountered during deserialize. Falling back to identity.`);
      }
      node.squash = activation_default[squashName] || activation_default.identity;
      node.index = nodeIndex;
      net.nodes.push(node);
    });
    connections.forEach((serializedConn) => {
      if (serializedConn.from < net.nodes.length && serializedConn.to < net.nodes.length) {
        const sourceNode = net.nodes[serializedConn.from];
        const targetNode = net.nodes[serializedConn.to];
        const createdConnection = net.connect(sourceNode, targetNode, serializedConn.weight)[0];
        if (createdConnection && serializedConn.gater != null) {
          if (serializedConn.gater < net.nodes.length) {
            net.gate(net.nodes[serializedConn.gater], createdConnection);
          } else {
            console.warn("Invalid gater index encountered during deserialize; skipping gater assignment.");
          }
        }
      } else {
        console.warn("Invalid connection indices encountered during deserialize; skipping connection.");
      }
    });
    return net;
  }
  function toJSONImpl() {
    const json = {
      formatVersion: 2,
      input: this.input,
      output: this.output,
      dropout: this.dropout,
      nodes: [],
      connections: []
    };
    this.nodes.forEach((node, nodeIndex) => {
      node.index = nodeIndex;
      json.nodes.push({
        type: node.type,
        bias: node.bias,
        squash: node.squash.name,
        index: nodeIndex,
        geneId: node.geneId
      });
      if (node.connections.self.length > 0) {
        const selfConn = node.connections.self[0];
        json.connections.push({
          from: nodeIndex,
          to: nodeIndex,
          weight: selfConn.weight,
          gater: selfConn.gater ? selfConn.gater.index : null,
          enabled: selfConn.enabled !== false
        });
      }
    });
    this.connections.forEach((connInstance) => {
      if (typeof connInstance.from.index !== "number" || typeof connInstance.to.index !== "number")
        return;
      json.connections.push({
        from: connInstance.from.index,
        to: connInstance.to.index,
        weight: connInstance.weight,
        gater: connInstance.gater ? connInstance.gater.index : null,
        enabled: connInstance.enabled !== false
      });
    });
    return json;
  }
  function fromJSONImpl(json) {
    if (!json || typeof json !== "object")
      throw new Error("Invalid JSON for network.");
    if (json.formatVersion !== 2)
      console.warn("fromJSONImpl: Unknown formatVersion, attempting import.");
    const net = new (init_network(), __toCommonJS(network_exports)).default(json.input, json.output);
    net.dropout = json.dropout || 0;
    net.nodes = [];
    net.connections = [];
    net.selfconns = [];
    net.gates = [];
    json.nodes.forEach((nodeJson, nodeIndex) => {
      const node = new Node(nodeJson.type);
      node.bias = nodeJson.bias;
      node.squash = activation_default[nodeJson.squash] || activation_default.identity;
      node.index = nodeIndex;
      if (typeof nodeJson.geneId === "number")
        node.geneId = nodeJson.geneId;
      net.nodes.push(node);
    });
    json.connections.forEach((connJson) => {
      if (typeof connJson.from !== "number" || typeof connJson.to !== "number")
        return;
      const nodesLength = net.nodes.length;
      if (connJson.from < 0 || connJson.to < 0 || connJson.from >= nodesLength || connJson.to >= nodesLength) {
        console.warn("Invalid connection indices encountered during fromJSONImpl; skipping connection.");
        return;
      }
      const sourceNode = net.nodes[connJson.from];
      const targetNode = net.nodes[connJson.to];
      const createdConnection = net.connect(sourceNode, targetNode, connJson.weight)[0];
      if (createdConnection && connJson.gater != null && typeof connJson.gater === "number") {
        if (connJson.gater >= 0 && connJson.gater < nodesLength) {
          net.gate(net.nodes[connJson.gater], createdConnection);
        } else {
          console.warn("Invalid gater index encountered during fromJSONImpl; skipping gater assignment.");
        }
      }
      if (createdConnection && typeof connJson.enabled !== "undefined")
        createdConnection.enabled = connJson.enabled;
    });
    return net;
  }
  var init_network_serialize = __esm({
    "dist/architecture/network/network.serialize.js"() {
      "use strict";
      init_node();
      init_connection();
      init_methods();
    }
  });

  // dist/architecture/network/network.genetic.js
  function crossOver(network1, network2, equal = false) {
    if (network1.input !== network2.input || network1.output !== network2.output)
      throw new Error("Parent networks must have the same input and output sizes for crossover.");
    const offspring = new (init_network(), __toCommonJS(network_exports)).default(network1.input, network1.output);
    offspring.connections = [];
    offspring.nodes = [];
    offspring.selfconns = [];
    offspring.gates = [];
    const score1 = network1.score || 0;
    const score2 = network2.score || 0;
    const n1Size = network1.nodes.length;
    const n2Size = network2.nodes.length;
    let size;
    if (equal || score1 === score2) {
      const max = Math.max(n1Size, n2Size);
      const min = Math.min(n1Size, n2Size);
      size = Math.floor(Math.random() * (max - min + 1) + min);
    } else
      size = score1 > score2 ? n1Size : n2Size;
    const outputSize = network1.output;
    network1.nodes.forEach((n, i) => n.index = i);
    network2.nodes.forEach((n, i) => n.index = i);
    for (let i = 0; i < size; i++) {
      let chosen;
      const node1 = i < n1Size ? network1.nodes[i] : void 0;
      const node2 = i < n2Size ? network2.nodes[i] : void 0;
      if (i < network1.input)
        chosen = node1;
      else if (i >= size - outputSize) {
        const o1 = n1Size - (size - i);
        const o2 = n2Size - (size - i);
        const n1o = o1 >= network1.input && o1 < n1Size ? network1.nodes[o1] : void 0;
        const n2o = o2 >= network2.input && o2 < n2Size ? network2.nodes[o2] : void 0;
        if (n1o && n2o)
          chosen = (network1._rand || Math.random)() >= 0.5 ? n1o : n2o;
        else
          chosen = n1o || n2o;
      } else {
        if (node1 && node2)
          chosen = (network1._rand || Math.random)() >= 0.5 ? node1 : node2;
        else if (node1 && (score1 >= score2 || equal))
          chosen = node1;
        else if (node2 && (score2 >= score1 || equal))
          chosen = node2;
      }
      if (chosen) {
        const nn = new Node(chosen.type);
        nn.bias = chosen.bias;
        nn.squash = chosen.squash;
        offspring.nodes.push(nn);
      }
    }
    offspring.nodes.forEach((n, i) => n.index = i);
    const n1conns = {};
    const n2conns = {};
    network1.connections.concat(network1.selfconns).forEach((c) => {
      if (typeof c.from.index === "number" && typeof c.to.index === "number")
        n1conns[Connection.innovationID(c.from.index, c.to.index)] = {
          weight: c.weight,
          from: c.from.index,
          to: c.to.index,
          gater: c.gater ? c.gater.index : -1,
          enabled: c.enabled !== false
        };
    });
    network2.connections.concat(network2.selfconns).forEach((c) => {
      if (typeof c.from.index === "number" && typeof c.to.index === "number")
        n2conns[Connection.innovationID(c.from.index, c.to.index)] = {
          weight: c.weight,
          from: c.from.index,
          to: c.to.index,
          gater: c.gater ? c.gater.index : -1,
          enabled: c.enabled !== false
        };
    });
    const chosenConns = [];
    const keys1 = Object.keys(n1conns);
    keys1.forEach((k) => {
      const c1 = n1conns[k];
      if (n2conns[k]) {
        const c2 = n2conns[k];
        const pick = (network1._rand || Math.random)() >= 0.5 ? c1 : c2;
        if (c1.enabled === false || c2.enabled === false) {
          const rp = network1._reenableProb ?? network2._reenableProb ?? 0.25;
          pick.enabled = Math.random() < rp;
        }
        chosenConns.push(pick);
        delete n2conns[k];
      } else if (score1 >= score2 || equal) {
        if (c1.enabled === false) {
          const rp = network1._reenableProb ?? 0.25;
          c1.enabled = Math.random() < rp;
        }
        chosenConns.push(c1);
      }
    });
    if (score2 >= score1 || equal)
      Object.keys(n2conns).forEach((k) => {
        const d = n2conns[k];
        if (d.enabled === false) {
          const rp = network2._reenableProb ?? 0.25;
          d.enabled = Math.random() < rp;
        }
        chosenConns.push(d);
      });
    const nodeCount = offspring.nodes.length;
    chosenConns.forEach((cd) => {
      if (cd.from < nodeCount && cd.to < nodeCount) {
        const from = offspring.nodes[cd.from];
        const to = offspring.nodes[cd.to];
        if (cd.from >= cd.to)
          return;
        if (!from.isProjectingTo(to)) {
          const conn = offspring.connect(from, to)[0];
          if (conn) {
            conn.weight = cd.weight;
            conn.enabled = cd.enabled !== false;
            if (cd.gater !== -1 && cd.gater < nodeCount)
              offspring.gate(offspring.nodes[cd.gater], conn);
          }
        }
      }
    });
    return offspring;
  }
  var init_network_genetic = __esm({
    "dist/architecture/network/network.genetic.js"() {
      "use strict";
      init_node();
      init_connection();
    }
  });

  // dist/architecture/network/network.activate.js
  var network_activate_exports = {};
  __export(network_activate_exports, {
    activateBatch: () => activateBatch,
    activateRaw: () => activateRaw,
    noTraceActivate: () => noTraceActivate
  });
  function noTraceActivate(input) {
    const self = this;
    if (self._enforceAcyclic && self._topoDirty)
      this._computeTopoOrder();
    if (!Array.isArray(input) || input.length !== this.input) {
      throw new Error(`Input size mismatch: expected ${this.input}, got ${input ? input.length : "undefined"}`);
    }
    if (this._canUseFastSlab(false)) {
      try {
        return this._fastSlabActivate(input);
      } catch {
      }
    }
    const output = activationArrayPool.acquire(this.output);
    let outIndex = 0;
    this.nodes.forEach((node, index) => {
      if (node.type === "input")
        node.noTraceActivate(input[index]);
      else if (node.type === "output")
        output[outIndex++] = node.noTraceActivate();
      else
        node.noTraceActivate();
    });
    const result = Array.from(output);
    activationArrayPool.release(output);
    return result;
  }
  function activateRaw(input, training = false, maxActivationDepth = 1e3) {
    const self = this;
    if (!self._reuseActivationArrays)
      return this.activate(input, training, maxActivationDepth);
    return this.activate(input, training, maxActivationDepth);
  }
  function activateBatch(inputs, training = false) {
    if (!Array.isArray(inputs))
      throw new Error("inputs must be an array of input arrays");
    const out = new Array(inputs.length);
    for (let i = 0; i < inputs.length; i++) {
      const x = inputs[i];
      if (!Array.isArray(x) || x.length !== this.input) {
        throw new Error(`Input[${i}] size mismatch: expected ${this.input}, got ${x ? x.length : "undefined"}`);
      }
      out[i] = this.activate(x, training);
    }
    return out;
  }
  var init_network_activate = __esm({
    "dist/architecture/network/network.activate.js"() {
      "use strict";
      init_activationArrayPool();
    }
  });

  // dist/architecture/group.js
  var Group;
  var init_group = __esm({
    "dist/architecture/group.js"() {
      "use strict";
      init_node();
      init_layer();
      init_config();
      init_methods();
      Group = class _Group {
        nodes;
        connections;
        constructor(size) {
          this.nodes = [];
          this.connections = {
            in: [],
            out: [],
            self: []
          };
          for (let i = 0; i < size; i++) {
            this.nodes.push(new Node());
          }
        }
        activate(value) {
          const values = [];
          if (value !== void 0 && value.length !== this.nodes.length) {
            throw new Error("Array with values should be same as the amount of nodes!");
          }
          for (let i = 0; i < this.nodes.length; i++) {
            const activation = value === void 0 ? this.nodes[i].activate() : this.nodes[i].activate(value[i]);
            values.push(activation);
          }
          return values;
        }
        propagate(rate, momentum, target) {
          if (target !== void 0 && target.length !== this.nodes.length) {
            throw new Error("Array with values should be same as the amount of nodes!");
          }
          for (let i = this.nodes.length - 1; i >= 0; i--) {
            if (target === void 0) {
              this.nodes[i].propagate(rate, momentum, true, 0);
            } else {
              this.nodes[i].propagate(rate, momentum, true, 0, target[i]);
            }
          }
        }
        connect(target, method, weight) {
          let connections = [];
          let i, j;
          if (target instanceof _Group) {
            if (method === void 0) {
              if (this !== target) {
                if (config.warnings)
                  console.warn("No group connection specified, using ALL_TO_ALL by default.");
                method = connection_default.ALL_TO_ALL;
              } else {
                if (config.warnings)
                  console.warn("Connecting group to itself, using ONE_TO_ONE by default.");
                method = connection_default.ONE_TO_ONE;
              }
            }
            if (method === connection_default.ALL_TO_ALL || method === connection_default.ALL_TO_ELSE) {
              for (i = 0; i < this.nodes.length; i++) {
                for (j = 0; j < target.nodes.length; j++) {
                  if (method === connection_default.ALL_TO_ELSE && this.nodes[i] === target.nodes[j])
                    continue;
                  const connection = this.nodes[i].connect(target.nodes[j], weight);
                  this.connections.out.push(connection[0]);
                  target.connections.in.push(connection[0]);
                  connections.push(connection[0]);
                }
              }
            } else if (method === connection_default.ONE_TO_ONE) {
              if (this.nodes.length !== target.nodes.length) {
                throw new Error("Cannot create ONE_TO_ONE connection: source and target groups must have the same size.");
              }
              for (i = 0; i < this.nodes.length; i++) {
                const connection = this.nodes[i].connect(target.nodes[i], weight);
                if (this === target) {
                  this.connections.self.push(connection[0]);
                } else {
                  this.connections.out.push(connection[0]);
                  target.connections.in.push(connection[0]);
                }
                connections.push(connection[0]);
              }
            }
          } else if (target instanceof Layer) {
            connections = target.input(this, method, weight);
          } else if (target instanceof Node) {
            for (i = 0; i < this.nodes.length; i++) {
              const connection = this.nodes[i].connect(target, weight);
              this.connections.out.push(connection[0]);
              connections.push(connection[0]);
            }
          }
          return connections;
        }
        gate(connections, method) {
          if (method === void 0) {
            throw new Error("Please specify a gating method: Gating.INPUT, Gating.OUTPUT, or Gating.SELF");
          }
          if (!Array.isArray(connections)) {
            connections = [connections];
          }
          const nodes1 = [];
          const nodes2 = [];
          let i, j;
          for (i = 0; i < connections.length; i++) {
            const connection = connections[i];
            if (!nodes1.includes(connection.from))
              nodes1.push(connection.from);
            if (!nodes2.includes(connection.to))
              nodes2.push(connection.to);
          }
          switch (method) {
            case gating.INPUT:
              for (let i2 = 0; i2 < connections.length; i2++) {
                const conn = connections[i2];
                const gater = this.nodes[i2 % this.nodes.length];
                gater.gate(conn);
              }
              break;
            case gating.OUTPUT:
              for (i = 0; i < nodes1.length; i++) {
                const node = nodes1[i];
                const gater = this.nodes[i % this.nodes.length];
                for (j = 0; j < node.connections.out.length; j++) {
                  const conn = node.connections.out[j];
                  if (connections.includes(conn)) {
                    gater.gate(conn);
                  }
                }
              }
              break;
            case gating.SELF:
              for (i = 0; i < nodes1.length; i++) {
                const node = nodes1[i];
                const gater = this.nodes[i % this.nodes.length];
                const selfConn = Array.isArray(node.connections.self) ? node.connections.self[0] : node.connections.self;
                if (connections.includes(selfConn)) {
                  gater.gate(selfConn);
                }
              }
              break;
          }
        }
        set(values) {
          for (let i = 0; i < this.nodes.length; i++) {
            if (values.bias !== void 0) {
              this.nodes[i].bias = values.bias;
            }
            this.nodes[i].squash = values.squash || this.nodes[i].squash;
            this.nodes[i].type = values.type || this.nodes[i].type;
          }
        }
        disconnect(target, twosided = false) {
          let i, j, k;
          if (target instanceof _Group) {
            for (i = 0; i < this.nodes.length; i++) {
              for (j = 0; j < target.nodes.length; j++) {
                this.nodes[i].disconnect(target.nodes[j], twosided);
                for (k = this.connections.out.length - 1; k >= 0; k--) {
                  const conn = this.connections.out[k];
                  if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                    this.connections.out.splice(k, 1);
                    break;
                  }
                }
                if (twosided) {
                  for (k = this.connections.in.length - 1; k >= 0; k--) {
                    const conn = this.connections.in[k];
                    if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                      this.connections.in.splice(k, 1);
                      break;
                    }
                  }
                  for (k = target.connections.out.length - 1; k >= 0; k--) {
                    const conn = target.connections.out[k];
                    if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                      target.connections.out.splice(k, 1);
                      break;
                    }
                  }
                  for (k = target.connections.in.length - 1; k >= 0; k--) {
                    const conn = target.connections.in[k];
                    if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                      target.connections.in.splice(k, 1);
                      break;
                    }
                  }
                }
              }
            }
          } else if (target instanceof Node) {
            for (i = 0; i < this.nodes.length; i++) {
              this.nodes[i].disconnect(target, twosided);
              for (j = this.connections.out.length - 1; j >= 0; j--) {
                const conn = this.connections.out[j];
                if (conn.from === this.nodes[i] && conn.to === target) {
                  this.connections.out.splice(j, 1);
                  break;
                }
              }
              if (twosided) {
                for (j = this.connections.in.length - 1; j >= 0; j--) {
                  const conn = this.connections.in[j];
                  if (conn.from === target && conn.to === this.nodes[i]) {
                    this.connections.in.splice(j, 1);
                    break;
                  }
                }
              }
            }
          }
        }
        clear() {
          for (let i = 0; i < this.nodes.length; i++) {
            this.nodes[i].clear();
          }
        }
        toJSON() {
          return {
            size: this.nodes.length,
            nodeIndices: this.nodes.map((n) => n.index),
            connections: {
              in: this.connections.in.length,
              out: this.connections.out.length,
              self: this.connections.self.length
            }
          };
        }
      };
    }
  });

  // dist/architecture/layer.js
  var layer_exports = {};
  __export(layer_exports, {
    default: () => Layer
  });
  var Layer;
  var init_layer = __esm({
    "dist/architecture/layer.js"() {
      "use strict";
      init_node();
      init_group();
      init_methods();
      init_activationArrayPool();
      Layer = class _Layer {
        nodes;
        connections;
        output;
        dropout = 0;
        constructor() {
          this.output = null;
          this.nodes = [];
          this.connections = { in: [], out: [], self: [] };
        }
        activate(value, training = false) {
          const out = activationArrayPool.acquire(this.nodes.length);
          if (value !== void 0 && value.length !== this.nodes.length) {
            throw new Error("Array with values should be same as the amount of nodes!");
          }
          let layerMask = 1;
          if (training && this.dropout > 0) {
            layerMask = Math.random() >= this.dropout ? 1 : 0;
            this.nodes.forEach((node) => {
              node.mask = layerMask;
            });
          } else {
            this.nodes.forEach((node) => {
              node.mask = 1;
            });
          }
          for (let i = 0; i < this.nodes.length; i++) {
            let activation;
            if (value === void 0) {
              activation = this.nodes[i].activate();
            } else {
              activation = this.nodes[i].activate(value[i]);
            }
            out[i] = activation;
          }
          const cloned = Array.from(out);
          activationArrayPool.release(out);
          return cloned;
        }
        propagate(rate, momentum, target) {
          if (target !== void 0 && target.length !== this.nodes.length) {
            throw new Error("Array with values should be same as the amount of nodes!");
          }
          for (let i = this.nodes.length - 1; i >= 0; i--) {
            if (target === void 0) {
              this.nodes[i].propagate(rate, momentum, true, 0);
            } else {
              this.nodes[i].propagate(rate, momentum, true, 0, target[i]);
            }
          }
        }
        connect(target, method, weight) {
          if (!this.output) {
            throw new Error("Layer output is not defined. Cannot connect from this layer.");
          }
          let connections = [];
          if (target instanceof _Layer) {
            connections = target.input(this, method, weight);
          } else if (target instanceof Group || target instanceof Node) {
            connections = this.output.connect(target, method, weight);
          }
          return connections;
        }
        gate(connections, method) {
          if (!this.output) {
            throw new Error("Layer output is not defined. Cannot gate from this layer.");
          }
          this.output.gate(connections, method);
        }
        set(values) {
          for (let i = 0; i < this.nodes.length; i++) {
            const node = this.nodes[i];
            if (node instanceof Node) {
              if (values.bias !== void 0) {
                node.bias = values.bias;
              }
              node.squash = values.squash || node.squash;
              node.type = values.type || node.type;
            } else if (this.isGroup(node)) {
              node.set(values);
            }
          }
        }
        disconnect(target, twosided) {
          twosided = twosided || false;
          let i, j, k;
          if (target instanceof Group) {
            for (i = 0; i < this.nodes.length; i++) {
              for (j = 0; j < target.nodes.length; j++) {
                this.nodes[i].disconnect(target.nodes[j], twosided);
                for (k = this.connections.out.length - 1; k >= 0; k--) {
                  const conn = this.connections.out[k];
                  if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                    this.connections.out.splice(k, 1);
                    break;
                  }
                }
                if (twosided) {
                  for (k = this.connections.in.length - 1; k >= 0; k--) {
                    const conn = this.connections.in[k];
                    if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                      this.connections.in.splice(k, 1);
                      break;
                    }
                  }
                }
              }
            }
          } else if (target instanceof Node) {
            for (i = 0; i < this.nodes.length; i++) {
              this.nodes[i].disconnect(target, twosided);
              for (j = this.connections.out.length - 1; j >= 0; j--) {
                const conn = this.connections.out[j];
                if (conn.from === this.nodes[i] && conn.to === target) {
                  this.connections.out.splice(j, 1);
                  break;
                }
              }
              if (twosided) {
                for (k = this.connections.in.length - 1; k >= 0; k--) {
                  const conn = this.connections.in[k];
                  if (conn.from === target && conn.to === this.nodes[i]) {
                    this.connections.in.splice(k, 1);
                    break;
                  }
                }
              }
            }
          }
        }
        clear() {
          for (let i = 0; i < this.nodes.length; i++) {
            this.nodes[i].clear();
          }
        }
        input(from, method, weight) {
          if (from instanceof _Layer)
            from = from.output;
          method = method || connection_default.ALL_TO_ALL;
          if (!this.output) {
            throw new Error("Layer output (acting as input target) is not defined.");
          }
          return from.connect(this.output, method, weight);
        }
        static dense(size) {
          const layer = new _Layer();
          const block = new Group(size);
          layer.nodes.push(...block.nodes);
          layer.output = block;
          layer.input = (from, method, weight) => {
            if (from instanceof _Layer)
              from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            return from.connect(block, method, weight);
          };
          return layer;
        }
        static lstm(size) {
          const layer = new _Layer();
          const inputGate = new Group(size);
          const forgetGate = new Group(size);
          const memoryCell = new Group(size);
          const outputGate = new Group(size);
          const outputBlock = new Group(size);
          inputGate.set({ bias: 1 });
          forgetGate.set({ bias: 1 });
          outputGate.set({ bias: 1 });
          memoryCell.set({ bias: 0 });
          outputBlock.set({ bias: 0 });
          memoryCell.connect(inputGate, connection_default.ALL_TO_ALL);
          memoryCell.connect(forgetGate, connection_default.ALL_TO_ALL);
          memoryCell.connect(outputGate, connection_default.ALL_TO_ALL);
          memoryCell.connect(memoryCell, connection_default.ONE_TO_ONE);
          const output = memoryCell.connect(outputBlock, connection_default.ALL_TO_ALL);
          outputGate.gate(output, gating.OUTPUT);
          memoryCell.nodes.forEach((node, i) => {
            const selfConnection = node.connections.self.find((conn) => conn.to === node && conn.from === node);
            if (selfConnection) {
              selfConnection.gater = forgetGate.nodes[i];
              if (!forgetGate.nodes[i].connections.gated.includes(selfConnection)) {
                forgetGate.nodes[i].connections.gated.push(selfConnection);
              }
            } else {
              console.warn(`LSTM Warning: No self-connection found for memory cell node ${i}`);
            }
          });
          layer.nodes = [
            ...inputGate.nodes,
            ...forgetGate.nodes,
            ...memoryCell.nodes,
            ...outputGate.nodes,
            ...outputBlock.nodes
          ];
          layer.output = outputBlock;
          layer.input = (from, method, weight) => {
            if (from instanceof _Layer)
              from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            let connections = [];
            const input = from.connect(memoryCell, method, weight);
            connections = connections.concat(input);
            connections = connections.concat(from.connect(inputGate, method, weight));
            connections = connections.concat(from.connect(outputGate, method, weight));
            connections = connections.concat(from.connect(forgetGate, method, weight));
            inputGate.gate(input, gating.INPUT);
            return connections;
          };
          return layer;
        }
        static gru(size) {
          const layer = new _Layer();
          const updateGate = new Group(size);
          const inverseUpdateGate = new Group(size);
          const resetGate = new Group(size);
          const memoryCell = new Group(size);
          const output = new Group(size);
          const previousOutput = new Group(size);
          previousOutput.set({
            bias: 0,
            squash: activation_default.identity,
            type: "variant"
          });
          memoryCell.set({
            squash: activation_default.tanh
          });
          inverseUpdateGate.set({
            bias: 0,
            squash: activation_default.inverse,
            type: "variant"
          });
          updateGate.set({ bias: 1 });
          resetGate.set({ bias: 0 });
          previousOutput.connect(updateGate, connection_default.ALL_TO_ALL);
          previousOutput.connect(resetGate, connection_default.ALL_TO_ALL);
          updateGate.connect(inverseUpdateGate, connection_default.ONE_TO_ONE, 1);
          const reset = previousOutput.connect(memoryCell, connection_default.ALL_TO_ALL);
          resetGate.gate(reset, gating.OUTPUT);
          const update1 = previousOutput.connect(output, connection_default.ALL_TO_ALL);
          const update2 = memoryCell.connect(output, connection_default.ALL_TO_ALL);
          updateGate.gate(update1, gating.OUTPUT);
          inverseUpdateGate.gate(update2, gating.OUTPUT);
          output.connect(previousOutput, connection_default.ONE_TO_ONE, 1);
          layer.nodes = [
            ...updateGate.nodes,
            ...inverseUpdateGate.nodes,
            ...resetGate.nodes,
            ...memoryCell.nodes,
            ...output.nodes,
            ...previousOutput.nodes
          ];
          layer.output = output;
          layer.input = (from, method, weight) => {
            if (from instanceof _Layer)
              from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            let connections = [];
            connections = connections.concat(from.connect(updateGate, method, weight));
            connections = connections.concat(from.connect(resetGate, method, weight));
            connections = connections.concat(from.connect(memoryCell, method, weight));
            return connections;
          };
          return layer;
        }
        static memory(size, memory) {
          const layer = new _Layer();
          let previous = null;
          for (let i = 0; i < memory; i++) {
            const block = new Group(size);
            block.set({
              squash: activation_default.identity,
              bias: 0,
              type: "variant"
            });
            if (previous != null) {
              previous.connect(block, connection_default.ONE_TO_ONE, 1);
            }
            layer.nodes.push(block);
            previous = block;
          }
          layer.nodes.reverse();
          const outputGroup = new Group(0);
          for (const group of layer.nodes) {
            if (this.prototype.isGroup(group)) {
              outputGroup.nodes = outputGroup.nodes.concat(group.nodes);
            } else {
              console.warn("Unexpected Node type found directly in Memory layer nodes list during output group creation.");
            }
          }
          layer.output = outputGroup;
          layer.input = (from, method, weight) => {
            if (from instanceof _Layer)
              from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            const inputBlock = layer.nodes[layer.nodes.length - 1];
            if (!this.prototype.isGroup(inputBlock)) {
              throw new Error("Memory layer input block is not a Group.");
            }
            if (from.nodes.length !== inputBlock.nodes.length) {
              throw new Error(`Previous layer size (${from.nodes.length}) must be same as memory size (${inputBlock.nodes.length})`);
            }
            return from.connect(inputBlock, connection_default.ONE_TO_ONE, 1);
          };
          return layer;
        }
        static batchNorm(size) {
          const layer = _Layer.dense(size);
          layer.batchNorm = true;
          const baseActivate = layer.activate.bind(layer);
          layer.activate = function(value, training = false) {
            const activations = baseActivate(value, training);
            const mean = activations.reduce((a, b) => a + b, 0) / activations.length;
            const variance = activations.reduce((a, b) => a + (b - mean) ** 2, 0) / activations.length;
            const epsilon = (init_neat_constants(), __toCommonJS(neat_constants_exports)).NORM_EPSILON;
            return activations.map((a) => (a - mean) / Math.sqrt(variance + epsilon));
          };
          return layer;
        }
        static layerNorm(size) {
          const layer = _Layer.dense(size);
          layer.layerNorm = true;
          const baseActivate = layer.activate.bind(layer);
          layer.activate = function(value, training = false) {
            const activations = baseActivate(value, training);
            const mean = activations.reduce((a, b) => a + b, 0) / activations.length;
            const variance = activations.reduce((a, b) => a + (b - mean) ** 2, 0) / activations.length;
            const epsilon = (init_neat_constants(), __toCommonJS(neat_constants_exports)).NORM_EPSILON;
            return activations.map((a) => (a - mean) / Math.sqrt(variance + epsilon));
          };
          return layer;
        }
        static conv1d(size, kernelSize, stride = 1, padding = 0) {
          const layer = new _Layer();
          layer.nodes = Array.from({ length: size }, () => new Node());
          layer.output = new Group(size);
          layer.conv1d = { kernelSize, stride, padding };
          layer.activate = function(value) {
            if (!value)
              return this.nodes.map((n) => n.activate());
            return value.slice(0, size);
          };
          return layer;
        }
        static attention(size, heads = 1) {
          const layer = new _Layer();
          layer.nodes = Array.from({ length: size }, () => new Node());
          layer.output = new Group(size);
          layer.attention = { heads };
          layer.activate = function(value) {
            if (!value)
              return this.nodes.map((n) => n.activate());
            const avg = value.reduce((a, b) => a + b, 0) / value.length;
            return Array(size).fill(avg);
          };
          return layer;
        }
        isGroup(obj) {
          return !!obj && typeof obj.set === "function" && Array.isArray(obj.nodes);
        }
      };
    }
  });

  // dist/architecture/network/network.mutate.js
  var network_mutate_exports = {};
  __export(network_mutate_exports, {
    mutateImpl: () => mutateImpl
  });
  function mutateImpl(method) {
    if (method == null)
      throw new Error("No (correct) mutate method given!");
    let key;
    if (typeof method === "string")
      key = method;
    else
      key = method?.name ?? method?.type ?? method?.identity;
    if (!key) {
      for (const k in mutation_default) {
        if (method === mutation_default[k]) {
          key = k;
          break;
        }
      }
    }
    const fn = key ? MUTATION_DISPATCH[key] : void 0;
    if (!fn) {
      if (config.warnings) {
        console.warn("[mutate] Unknown mutation method ignored:", key);
      }
      return;
    }
    fn.call(this, method);
    this._topoDirty = true;
  }
  function _addNode() {
    const internal = this;
    if (internal._enforceAcyclic)
      internal._topoDirty = true;
    if (config.deterministicChainMode) {
      const inputNode = this.nodes.find((n) => n.type === "input");
      const outputNode = this.nodes.find((n) => n.type === "output");
      if (!inputNode || !outputNode)
        return;
      if (!internal._detChain) {
        if (!this.connections.some((c) => c.from === inputNode && c.to === outputNode)) {
          this.connect(inputNode, outputNode);
        }
        internal._detChain = [inputNode];
      }
      const chain = internal._detChain;
      const tail = chain[chain.length - 1];
      let terminal = this.connections.find((c) => c.from === tail && c.to === outputNode);
      if (!terminal)
        terminal = this.connect(tail, outputNode)[0];
      const prevGater2 = terminal.gater;
      this.disconnect(terminal.from, terminal.to);
      const hidden2 = new Node("hidden", void 0, internal._rand);
      hidden2.mutate(mutation_default.MOD_ACTIVATION);
      const outIndex = this.nodes.indexOf(outputNode);
      const insertIndex2 = Math.min(outIndex, this.nodes.length - this.output);
      this.nodes.splice(insertIndex2, 0, hidden2);
      internal._nodeIndexDirty = true;
      const c12 = this.connect(tail, hidden2)[0];
      const c22 = this.connect(hidden2, outputNode)[0];
      chain.push(hidden2);
      internal._preferredChainEdge = c22;
      if (prevGater2)
        this.gate(prevGater2, internal._rand() >= 0.5 ? c12 : c22);
      for (let i = 0; i < chain.length; i++) {
        const node = chain[i];
        const target = i + 1 < chain.length ? chain[i + 1] : outputNode;
        const keep = node.connections.out.find((e) => e.to === target);
        if (keep) {
          for (const extra of node.connections.out.slice()) {
            if (extra !== keep) {
              try {
                this.disconnect(extra.from, extra.to);
              } catch {
              }
            }
          }
        }
      }
      return;
    }
    if (this.connections.length === 0) {
      const input = this.nodes.find((n) => n.type === "input");
      const output = this.nodes.find((n) => n.type === "output");
      if (input && output)
        this.connect(input, output);
      else
        return;
    }
    const connection = this.connections[Math.floor(internal._rand() * this.connections.length)];
    if (!connection)
      return;
    const prevGater = connection.gater;
    this.disconnect(connection.from, connection.to);
    const hidden = new Node("hidden", void 0, internal._rand);
    hidden.mutate(mutation_default.MOD_ACTIVATION);
    const targetIndex = this.nodes.indexOf(connection.to);
    const insertIndex = Math.min(targetIndex, this.nodes.length - this.output);
    this.nodes.splice(insertIndex, 0, hidden);
    internal._nodeIndexDirty = true;
    const c1 = this.connect(connection.from, hidden)[0];
    const c2 = this.connect(hidden, connection.to)[0];
    internal._preferredChainEdge = c2;
    if (prevGater)
      this.gate(prevGater, internal._rand() >= 0.5 ? c1 : c2);
  }
  function _subNode() {
    const hidden = this.nodes.filter((n) => n.type === "hidden");
    if (hidden.length === 0) {
      if (config.warnings)
        console.warn("No hidden nodes left to remove!");
      return;
    }
    const internal = this;
    const victim = hidden[Math.floor(internal._rand() * hidden.length)];
    this.remove(victim);
    const anyConn = this.connections[0];
    if (anyConn)
      anyConn.weight += 1e-4;
  }
  function _addConn() {
    const netInternal = this;
    if (netInternal._enforceAcyclic)
      netInternal._topoDirty = true;
    const forwardConnectionCandidates = [];
    for (let sourceIndex = 0; sourceIndex < this.nodes.length - this.output; sourceIndex++) {
      const sourceNode = this.nodes[sourceIndex];
      for (let targetIndex = Math.max(sourceIndex + 1, this.input); targetIndex < this.nodes.length; targetIndex++) {
        const targetNode = this.nodes[targetIndex];
        if (!sourceNode.isProjectingTo(targetNode))
          forwardConnectionCandidates.push([sourceNode, targetNode]);
      }
    }
    if (forwardConnectionCandidates.length === 0)
      return;
    const selectedPair = forwardConnectionCandidates[Math.floor(netInternal._rand() * forwardConnectionCandidates.length)];
    this.connect(selectedPair[0], selectedPair[1]);
  }
  function _subConn() {
    const netInternal = this;
    const removableForwardConnections = this.connections.filter((candidateConn) => {
      const sourceHasMultipleOutgoing = candidateConn.from.connections.out.length > 1;
      const targetHasMultipleIncoming = candidateConn.to.connections.in.length > 1;
      const targetLayerPeers = this.nodes.filter((n) => n.type === candidateConn.to.type && Math.abs(this.nodes.indexOf(n) - this.nodes.indexOf(candidateConn.to)) < Math.max(this.input, this.output));
      let wouldDisconnectLayerPeerGroup = false;
      if (targetLayerPeers.length > 0) {
        const peerConnectionsFromSource = this.connections.filter((c) => c.from === candidateConn.from && targetLayerPeers.includes(c.to));
        if (peerConnectionsFromSource.length <= 1)
          wouldDisconnectLayerPeerGroup = true;
      }
      return sourceHasMultipleOutgoing && targetHasMultipleIncoming && this.nodes.indexOf(candidateConn.to) > this.nodes.indexOf(candidateConn.from) && !wouldDisconnectLayerPeerGroup;
    });
    if (removableForwardConnections.length === 0)
      return;
    const connectionToRemove = removableForwardConnections[Math.floor(netInternal._rand() * removableForwardConnections.length)];
    this.disconnect(connectionToRemove.from, connectionToRemove.to);
  }
  function _modWeight(method) {
    const allConnections = this.connections.concat(this.selfconns);
    if (allConnections.length === 0)
      return;
    const connectionToPerturb = allConnections[Math.floor(this._rand() * allConnections.length)];
    const modification = this._rand() * (method.max - method.min) + method.min;
    connectionToPerturb.weight += modification;
  }
  function _modBias(method) {
    if (this.nodes.length <= this.input)
      return;
    const targetNodeIndex = Math.floor(this._rand() * (this.nodes.length - this.input) + this.input);
    const nodeForBiasMutation = this.nodes[targetNodeIndex];
    nodeForBiasMutation.mutate(method);
  }
  function _modActivation(method) {
    const canMutateOutput = method.mutateOutput ?? true;
    const numMutableNodes = this.nodes.length - this.input - (canMutateOutput ? 0 : this.output);
    if (numMutableNodes <= 0) {
      if (config.warnings)
        console.warn("No nodes available for activation function mutation based on config.");
      return;
    }
    const targetNodeIndex = Math.floor(this._rand() * numMutableNodes + this.input);
    const targetNode = this.nodes[targetNodeIndex];
    targetNode.mutate(method);
  }
  function _addSelfConn() {
    const netInternal = this;
    if (netInternal._enforceAcyclic)
      return;
    const nodesWithoutSelfLoop = this.nodes.filter((n, idx) => idx >= this.input && n.connections.self.length === 0);
    if (nodesWithoutSelfLoop.length === 0) {
      if (config.warnings)
        console.warn("All eligible nodes already have self-connections.");
      return;
    }
    const nodeReceivingSelfLoop = nodesWithoutSelfLoop[Math.floor(netInternal._rand() * nodesWithoutSelfLoop.length)];
    this.connect(nodeReceivingSelfLoop, nodeReceivingSelfLoop);
  }
  function _subSelfConn() {
    if (this.selfconns.length === 0) {
      if (config.warnings)
        console.warn("No self-connections exist to remove.");
      return;
    }
    const selfConnectionToRemove = this.selfconns[Math.floor(this._rand() * this.selfconns.length)];
    this.disconnect(selfConnectionToRemove.from, selfConnectionToRemove.to);
  }
  function _addGate() {
    const netInternal = this;
    const allConnectionsIncludingSelf = this.connections.concat(this.selfconns);
    const ungatedConnectionCandidates = allConnectionsIncludingSelf.filter((c) => c.gater === null);
    if (ungatedConnectionCandidates.length === 0 || this.nodes.length <= this.input) {
      if (config.warnings)
        console.warn("All connections are already gated.");
      return;
    }
    const gatingNodeIndex = Math.floor(netInternal._rand() * (this.nodes.length - this.input) + this.input);
    const gatingNode = this.nodes[gatingNodeIndex];
    const connectionToGate = ungatedConnectionCandidates[Math.floor(netInternal._rand() * ungatedConnectionCandidates.length)];
    this.gate(gatingNode, connectionToGate);
  }
  function _subGate() {
    if (this.gates.length === 0) {
      if (config.warnings)
        console.warn("No gated connections to ungate.");
      return;
    }
    const gatedConnectionIndex = Math.floor(this._rand() * this.gates.length);
    const gatedConnection = this.gates[gatedConnectionIndex];
    this.ungate(gatedConnection);
  }
  function _addBackConn() {
    const netInternal = this;
    if (netInternal._enforceAcyclic)
      return;
    const backwardConnectionCandidates = [];
    for (let laterIndex = this.input; laterIndex < this.nodes.length; laterIndex++) {
      const laterNode = this.nodes[laterIndex];
      for (let earlierIndex = this.input; earlierIndex < laterIndex; earlierIndex++) {
        const earlierNode = this.nodes[earlierIndex];
        if (!laterNode.isProjectingTo(earlierNode))
          backwardConnectionCandidates.push([laterNode, earlierNode]);
      }
    }
    if (backwardConnectionCandidates.length === 0)
      return;
    const selectedBackwardPair = backwardConnectionCandidates[Math.floor(netInternal._rand() * backwardConnectionCandidates.length)];
    this.connect(selectedBackwardPair[0], selectedBackwardPair[1]);
  }
  function _subBackConn() {
    const removableBackwardConnections = this.connections.filter((candidateConn) => candidateConn.from.connections.out.length > 1 && candidateConn.to.connections.in.length > 1 && this.nodes.indexOf(candidateConn.from) > this.nodes.indexOf(candidateConn.to));
    if (removableBackwardConnections.length === 0)
      return;
    const backwardConnectionToRemove = removableBackwardConnections[Math.floor(this._rand() * removableBackwardConnections.length)];
    this.disconnect(backwardConnectionToRemove.from, backwardConnectionToRemove.to);
  }
  function _swapNodes(method) {
    const netInternal = this;
    const canSwapOutput = method.mutateOutput ?? true;
    const numSwappableNodes = this.nodes.length - this.input - (canSwapOutput ? 0 : this.output);
    if (numSwappableNodes < 2)
      return;
    const firstNodeIndex = Math.floor(netInternal._rand() * numSwappableNodes + this.input);
    let secondNodeIndex = Math.floor(netInternal._rand() * numSwappableNodes + this.input);
    while (firstNodeIndex === secondNodeIndex)
      secondNodeIndex = Math.floor(netInternal._rand() * numSwappableNodes + this.input);
    const firstNode = this.nodes[firstNodeIndex];
    const secondNode = this.nodes[secondNodeIndex];
    const tempBias = firstNode.bias;
    const tempSquash = firstNode.squash;
    firstNode.bias = secondNode.bias;
    firstNode.squash = secondNode.squash;
    secondNode.bias = tempBias;
    secondNode.squash = tempSquash;
  }
  function _addLSTMNode() {
    const netInternal = this;
    if (netInternal._enforceAcyclic)
      return;
    if (this.connections.length === 0)
      return;
    const connectionToExpand = this.connections[Math.floor(Math.random() * this.connections.length)];
    const gaterLSTM = connectionToExpand.gater;
    this.disconnect(connectionToExpand.from, connectionToExpand.to);
    const Layer2 = (init_layer(), __toCommonJS(layer_exports)).default;
    const lstmLayer = Layer2.lstm(1);
    lstmLayer.nodes.forEach((n) => {
      n.type = "hidden";
      this.nodes.push(n);
    });
    this.connect(connectionToExpand.from, lstmLayer.nodes[0]);
    this.connect(lstmLayer.output.nodes[0], connectionToExpand.to);
    if (gaterLSTM)
      this.gate(gaterLSTM, this.connections[this.connections.length - 1]);
  }
  function _addGRUNode() {
    const netInternal = this;
    if (netInternal._enforceAcyclic)
      return;
    if (this.connections.length === 0)
      return;
    const connectionToExpand = this.connections[Math.floor(Math.random() * this.connections.length)];
    const gaterGRU = connectionToExpand.gater;
    this.disconnect(connectionToExpand.from, connectionToExpand.to);
    const Layer2 = (init_layer(), __toCommonJS(layer_exports)).default;
    const gruLayer = Layer2.gru(1);
    gruLayer.nodes.forEach((n) => {
      n.type = "hidden";
      this.nodes.push(n);
    });
    this.connect(connectionToExpand.from, gruLayer.nodes[0]);
    this.connect(gruLayer.output.nodes[0], connectionToExpand.to);
    if (gaterGRU)
      this.gate(gaterGRU, this.connections[this.connections.length - 1]);
  }
  function _reinitWeight(method) {
    if (this.nodes.length <= this.input)
      return;
    const internal = this;
    const idx = Math.floor(internal._rand() * (this.nodes.length - this.input) + this.input);
    const node = this.nodes[idx];
    const min = method?.min ?? -1;
    const max = method?.max ?? 1;
    const sample = () => internal._rand() * (max - min) + min;
    for (const c of node.connections.in)
      c.weight = sample();
    for (const c of node.connections.out)
      c.weight = sample();
    for (const c of node.connections.self)
      c.weight = sample();
  }
  function _batchNorm() {
    const hidden = this.nodes.filter((n) => n.type === "hidden");
    if (!hidden.length)
      return;
    const internal = this;
    const node = hidden[Math.floor(internal._rand() * hidden.length)];
    node._batchNorm = true;
  }
  var MUTATION_DISPATCH;
  var init_network_mutate = __esm({
    "dist/architecture/network/network.mutate.js"() {
      "use strict";
      init_node();
      init_mutation();
      init_config();
      MUTATION_DISPATCH = {
        ADD_NODE: _addNode,
        SUB_NODE: _subNode,
        ADD_CONN: _addConn,
        SUB_CONN: _subConn,
        MOD_WEIGHT: _modWeight,
        MOD_BIAS: _modBias,
        MOD_ACTIVATION: _modActivation,
        ADD_SELF_CONN: _addSelfConn,
        SUB_SELF_CONN: _subSelfConn,
        ADD_GATE: _addGate,
        SUB_GATE: _subGate,
        ADD_BACK_CONN: _addBackConn,
        SUB_BACK_CONN: _subBackConn,
        SWAP_NODES: _swapNodes,
        ADD_LSTM_NODE: _addLSTMNode,
        ADD_GRU_NODE: _addGRUNode,
        REINIT_WEIGHT: _reinitWeight,
        BATCH_NORM: _batchNorm
      };
    }
  });

  // dist/architecture/network/network.training.js
  var network_training_exports = {};
  __export(network_training_exports, {
    __trainingInternals: () => __trainingInternals,
    applyGradientClippingImpl: () => applyGradientClippingImpl,
    trainImpl: () => trainImpl,
    trainSetImpl: () => trainSetImpl
  });
  function computeMonitoredError(trainError, recentErrors, cfg, state) {
    if (cfg.window <= 1 && cfg.type !== "ema" && cfg.type !== "adaptive-ema") {
      return trainError;
    }
    const type = cfg.type;
    if (type === "median") {
      const sorted = [...recentErrors].sort((a, b) => a - b);
      const midIndex = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[midIndex] : (sorted[midIndex - 1] + sorted[midIndex]) / 2;
    }
    if (type === "ema") {
      if (state.emaValue == null)
        state.emaValue = trainError;
      else
        state.emaValue = state.emaValue + cfg.emaAlpha * (trainError - state.emaValue);
      return state.emaValue;
    }
    if (type === "adaptive-ema") {
      const mean = recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length;
      const variance = recentErrors.reduce((a, b) => a + (b - mean) * (b - mean), 0) / recentErrors.length;
      const baseAlpha = cfg.emaAlpha || 2 / (cfg.window + 1);
      const varianceScaled = variance / Math.max(mean * mean, 1e-8);
      const adaptiveAlpha = Math.min(0.95, Math.max(baseAlpha, baseAlpha * (1 + 2 * varianceScaled)));
      if (state.adaptiveBaseEmaValue == null) {
        state.adaptiveBaseEmaValue = trainError;
        state.adaptiveEmaValue = trainError;
      } else {
        state.adaptiveBaseEmaValue = state.adaptiveBaseEmaValue + baseAlpha * (trainError - state.adaptiveBaseEmaValue);
        state.adaptiveEmaValue = state.adaptiveEmaValue + adaptiveAlpha * (trainError - state.adaptiveEmaValue);
      }
      return Math.min(state.adaptiveEmaValue, state.adaptiveBaseEmaValue);
    }
    if (type === "gaussian") {
      const sigma = cfg.window / 3 || 1;
      let weightSum = 0;
      let weightedAccumulator = 0;
      const length = recentErrors.length;
      for (let i = 0; i < length; i++) {
        const weight = Math.exp(-0.5 * Math.pow((i - (length - 1)) / sigma, 2));
        weightSum += weight;
        weightedAccumulator += weight * recentErrors[i];
      }
      return weightedAccumulator / (weightSum || 1);
    }
    if (type === "trimmed") {
      const ratio = Math.min(0.49, Math.max(0, cfg.trimmedRatio || 0.1));
      const sorted = [...recentErrors].sort((a, b) => a - b);
      const drop = Math.floor(sorted.length * ratio);
      const trimmed = sorted.slice(drop, sorted.length - drop);
      return trimmed.reduce((a, b) => a + b, 0) / (trimmed.length || 1);
    }
    if (type === "wma") {
      let weightSum = 0;
      let weightedAccumulator = 0;
      for (let i = 0; i < recentErrors.length; i++) {
        const weight = i + 1;
        weightSum += weight;
        weightedAccumulator += weight * recentErrors[i];
      }
      return weightedAccumulator / (weightSum || 1);
    }
    return recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length;
  }
  function computePlateauMetric(trainError, plateauErrors, cfg, state) {
    if (cfg.window <= 1 && cfg.type !== "ema")
      return trainError;
    if (cfg.type === "median") {
      const sorted = [...plateauErrors].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }
    if (cfg.type === "ema") {
      if (state.plateauEmaValue == null)
        state.plateauEmaValue = trainError;
      else
        state.plateauEmaValue = state.plateauEmaValue + cfg.emaAlpha * (trainError - state.plateauEmaValue);
      return state.plateauEmaValue;
    }
    return plateauErrors.reduce((a, b) => a + b, 0) / plateauErrors.length;
  }
  function detectMixedPrecisionOverflow(net, internalNet) {
    if (!internalNet._mixedPrecision.enabled)
      return false;
    if (internalNet._forceNextOverflow) {
      internalNet._forceNextOverflow = false;
      return true;
    }
    let overflow = false;
    net.nodes.forEach((node) => {
      if (node._fp32Bias !== void 0) {
        if (!Number.isFinite(node.bias))
          overflow = true;
      }
    });
    return overflow;
  }
  function zeroAccumulatedGradients(net) {
    net.nodes.forEach((node) => {
      node.connections.in.forEach((c) => {
        c.totalDeltaWeight = 0;
      });
      node.connections.self.forEach((c) => {
        c.totalDeltaWeight = 0;
      });
      if (typeof node.totalDeltaBias === "number")
        node.totalDeltaBias = 0;
      node.previousDeltaBias = 0;
    });
  }
  function averageAccumulatedGradients(net, accumulationSteps) {
    if (accumulationSteps <= 1)
      return;
    net.nodes.forEach((node) => {
      node.connections.in.forEach((c) => {
        if (typeof c.totalDeltaWeight === "number")
          c.totalDeltaWeight /= accumulationSteps;
      });
      node.connections.self.forEach((c) => {
        if (typeof c.totalDeltaWeight === "number")
          c.totalDeltaWeight /= accumulationSteps;
      });
      if (typeof node.totalDeltaBias === "number")
        node.totalDeltaBias /= accumulationSteps;
    });
  }
  function applyOptimizerStep(net, optimizer, currentRate, momentum, internalNet) {
    let sumSq = 0;
    net.nodes.forEach((node) => {
      if (node.type === "input")
        return;
      node.applyBatchUpdatesWithOptimizer({
        type: optimizer.type,
        baseType: optimizer.baseType,
        beta1: optimizer.beta1,
        beta2: optimizer.beta2,
        eps: optimizer.eps,
        weightDecay: optimizer.weightDecay,
        momentum: optimizer.momentum ?? momentum,
        lrScale: currentRate,
        t: internalNet._optimizerStep,
        la_k: optimizer.la_k,
        la_alpha: optimizer.la_alpha
      });
      node.connections.in.forEach((c) => {
        if (typeof c.previousDeltaWeight === "number")
          sumSq += c.previousDeltaWeight * c.previousDeltaWeight;
      });
      node.connections.self.forEach((c) => {
        if (typeof c.previousDeltaWeight === "number")
          sumSq += c.previousDeltaWeight * c.previousDeltaWeight;
      });
    });
    return Math.sqrt(sumSq);
  }
  function maybeIncreaseLossScale(internalNet) {
    internalNet._mixedPrecisionState.goodSteps++;
    const incEvery = internalNet._mpIncreaseEvery || 200;
    if (internalNet._mixedPrecisionState.goodSteps >= incEvery && internalNet._mixedPrecision.lossScale < internalNet._mixedPrecisionState.maxLossScale) {
      internalNet._mixedPrecision.lossScale *= 2;
      internalNet._mixedPrecisionState.goodSteps = 0;
      internalNet._mixedPrecisionState.scaleUpEvents = (internalNet._mixedPrecisionState.scaleUpEvents || 0) + 1;
    }
  }
  function handleOverflow(internalNet) {
    internalNet._mixedPrecisionState.badSteps++;
    internalNet._mixedPrecisionState.goodSteps = 0;
    internalNet._mixedPrecision.lossScale = Math.max(internalNet._mixedPrecisionState.minLossScale, Math.floor(internalNet._mixedPrecision.lossScale / 2) || 1);
    internalNet._mixedPrecisionState.overflowCount = (internalNet._mixedPrecisionState.overflowCount || 0) + 1;
    internalNet._mixedPrecisionState.scaleDownEvents = (internalNet._mixedPrecisionState.scaleDownEvents || 0) + 1;
    internalNet._lastOverflowStep = internalNet._optimizerStep;
  }
  function applyGradientClippingImpl(net, cfg) {
    const internalNet = net;
    const collectGroups = () => {
      const collected = [];
      if (cfg.mode.startsWith("layerwise")) {
        if (net.layers && net.layers.length > 0) {
          for (let li = 0; li < net.layers.length; li++) {
            const layer = net.layers[li];
            if (!layer || !layer.nodes)
              continue;
            const groupVals = [];
            layer.nodes.forEach((node) => {
              if (!node || node.type === "input")
                return;
              node.connections.in.forEach((c) => {
                if (typeof c.totalDeltaWeight === "number")
                  groupVals.push(c.totalDeltaWeight);
              });
              node.connections.self.forEach((c) => {
                if (typeof c.totalDeltaWeight === "number")
                  groupVals.push(c.totalDeltaWeight);
              });
              if (typeof node.totalDeltaBias === "number")
                groupVals.push(node.totalDeltaBias);
            });
            if (groupVals.length)
              collected.push(groupVals);
          }
        } else {
          net.nodes.forEach((node) => {
            if (node.type === "input")
              return;
            const groupVals = [];
            node.connections.in.forEach((c) => {
              if (typeof c.totalDeltaWeight === "number")
                groupVals.push(c.totalDeltaWeight);
            });
            node.connections.self.forEach((c) => {
              if (typeof c.totalDeltaWeight === "number")
                groupVals.push(c.totalDeltaWeight);
            });
            if (typeof node.totalDeltaBias === "number")
              groupVals.push(node.totalDeltaBias);
            if (groupVals.length)
              collected.push(groupVals);
          });
        }
      } else {
        const globalVals = [];
        net.nodes.forEach((node) => {
          node.connections.in.forEach((c) => {
            if (typeof c.totalDeltaWeight === "number")
              globalVals.push(c.totalDeltaWeight);
          });
          node.connections.self.forEach((c) => {
            if (typeof c.totalDeltaWeight === "number")
              globalVals.push(c.totalDeltaWeight);
          });
          if (typeof node.totalDeltaBias === "number")
            globalVals.push(node.totalDeltaBias);
        });
        if (globalVals.length)
          collected.push(globalVals);
      }
      return collected;
    };
    const groups = collectGroups();
    internalNet._lastGradClipGroupCount = groups.length;
    const computeAbsolutePercentileThreshold = (values, percentile) => {
      if (!values.length)
        return 0;
      const sortedByAbs = [...values].sort((a, b) => Math.abs(a) - Math.abs(b));
      const rank = Math.min(sortedByAbs.length - 1, Math.max(0, Math.floor(percentile / 100 * sortedByAbs.length - 1)));
      return Math.abs(sortedByAbs[rank]);
    };
    const applyScale = (scaleFn) => {
      let groupIndex = 0;
      net.nodes.forEach((node) => {
        if (cfg.mode.startsWith("layerwise") && node.type === "input")
          return;
        const activeGroup = cfg.mode.startsWith("layerwise") ? groups[groupIndex++] : groups[0];
        node.connections.in.forEach((c) => {
          if (typeof c.totalDeltaWeight === "number")
            c.totalDeltaWeight = scaleFn(c.totalDeltaWeight, activeGroup);
        });
        node.connections.self.forEach((c) => {
          if (typeof c.totalDeltaWeight === "number")
            c.totalDeltaWeight = scaleFn(c.totalDeltaWeight, activeGroup);
        });
        if (typeof node.totalDeltaBias === "number")
          node.totalDeltaBias = scaleFn(node.totalDeltaBias, activeGroup);
      });
    };
    if (cfg.mode === "norm" || cfg.mode === "layerwiseNorm") {
      const maxAllowedNorm = cfg.maxNorm || 1;
      groups.forEach((groupValues) => {
        const groupL2Norm = Math.sqrt(groupValues.reduce((sum, v) => sum + v * v, 0));
        if (groupL2Norm > maxAllowedNorm && groupL2Norm > 0) {
          const normScaleFactor = maxAllowedNorm / groupL2Norm;
          applyScale((currentValue, owningGroup) => owningGroup === groupValues ? currentValue * normScaleFactor : currentValue);
        }
      });
    } else if (cfg.mode === "percentile" || cfg.mode === "layerwisePercentile") {
      const percentileSetting = cfg.percentile || 99;
      groups.forEach((groupValues) => {
        const percentileThreshold = computeAbsolutePercentileThreshold(groupValues, percentileSetting);
        if (percentileThreshold <= 0)
          return;
        applyScale((currentValue, owningGroup) => owningGroup === groupValues && Math.abs(currentValue) > percentileThreshold ? percentileThreshold * Math.sign(currentValue) : currentValue);
      });
    }
  }
  function trainSetImpl(net, set, batchSize, accumulationSteps, currentRate, momentum, regularization, costFunction, optimizer) {
    const internalNet = net;
    let cumulativeError = 0;
    let batchSampleCount = 0;
    internalNet._gradAccumMicroBatches = 0;
    let totalProcessedSamples = 0;
    const outputNodes = net.nodes.filter((n) => n.type === "output");
    let computeError;
    if (typeof costFunction === "function")
      computeError = costFunction;
    else if (costFunction && typeof costFunction.fn === "function")
      computeError = costFunction.fn;
    else if (costFunction && typeof costFunction.calculate === "function")
      computeError = costFunction.calculate;
    else
      computeError = () => 0;
    for (let sampleIndex = 0; sampleIndex < set.length; sampleIndex++) {
      const dataPoint = set[sampleIndex];
      const input = dataPoint.input;
      const target = dataPoint.output;
      if (input.length !== net.input || target.length !== net.output) {
        if (config.warnings)
          console.warn(`Data point ${sampleIndex} has incorrect dimensions (input: ${input.length}/${net.input}, output: ${target.length}/${net.output}), skipping.`);
        continue;
      }
      try {
        const output = net.activate(input, true);
        if (optimizer && optimizer.type && optimizer.type !== "sgd") {
          for (let outIndex = 0; outIndex < outputNodes.length; outIndex++)
            outputNodes[outIndex].propagate(currentRate, momentum, false, regularization, target[outIndex]);
          for (let reverseIndex = net.nodes.length - 1; reverseIndex >= 0; reverseIndex--) {
            const node = net.nodes[reverseIndex];
            if (node.type === "output" || node.type === "input")
              continue;
            node.propagate(currentRate, momentum, false, regularization);
          }
        } else {
          for (let outIndex = 0; outIndex < outputNodes.length; outIndex++)
            outputNodes[outIndex].propagate(currentRate, momentum, true, regularization, target[outIndex]);
          for (let reverseIndex = net.nodes.length - 1; reverseIndex >= 0; reverseIndex--) {
            const node = net.nodes[reverseIndex];
            if (node.type === "output" || node.type === "input")
              continue;
            node.propagate(currentRate, momentum, true, regularization);
          }
        }
        cumulativeError += computeError(target, output);
        batchSampleCount++;
        totalProcessedSamples++;
      } catch (e) {
        if (config.warnings)
          console.warn(`Error processing data point ${sampleIndex} (input: ${JSON.stringify(input)}): ${e.message}. Skipping.`);
      }
      if (batchSampleCount > 0 && ((sampleIndex + 1) % batchSize === 0 || sampleIndex === set.length - 1)) {
        if (optimizer && optimizer.type && optimizer.type !== "sgd") {
          internalNet._gradAccumMicroBatches++;
          const readyForStep = internalNet._gradAccumMicroBatches % accumulationSteps === 0 || sampleIndex === set.length - 1;
          if (readyForStep) {
            internalNet._optimizerStep = (internalNet._optimizerStep || 0) + 1;
            const overflowDetected = detectMixedPrecisionOverflow(net, internalNet);
            if (overflowDetected) {
              zeroAccumulatedGradients(net);
              if (internalNet._mixedPrecision.enabled)
                handleOverflow(internalNet);
              internalNet._lastGradNorm = 0;
            } else {
              if (internalNet._currentGradClip)
                applyGradientClippingImpl(net, internalNet._currentGradClip);
              if (accumulationSteps > 1 && internalNet._accumulationReduction === "average") {
                averageAccumulatedGradients(net, accumulationSteps);
              }
              internalNet._lastGradNorm = applyOptimizerStep(net, optimizer, currentRate, momentum, internalNet);
              if (internalNet._mixedPrecision.enabled)
                maybeIncreaseLossScale(internalNet);
            }
          }
          batchSampleCount = 0;
        }
      }
    }
    if (internalNet._lastGradNorm == null)
      internalNet._lastGradNorm = 0;
    return totalProcessedSamples > 0 ? cumulativeError / totalProcessedSamples : 0;
  }
  function trainImpl(net, set, options) {
    const internalNet = net;
    if (!set || set.length === 0 || set[0].input.length !== net.input || set[0].output.length !== net.output) {
      throw new Error("Dataset is invalid or dimensions do not match network input/output size!");
    }
    options = options || {};
    if (typeof options.iterations === "undefined" && typeof options.error === "undefined") {
      if (config.warnings)
        console.warn("Missing `iterations` or `error` option.");
      throw new Error("Missing `iterations` or `error` option. Training requires a stopping condition.");
    }
    if (config.warnings) {
      if (typeof options.rate === "undefined") {
        console.warn("Missing `rate` option");
        console.warn("Missing `rate` option, using default learning rate 0.3.");
      }
      if (typeof options.iterations === "undefined")
        console.warn("Missing `iterations` option. Training will run potentially indefinitely until `error` threshold is met.");
    }
    const targetError = options.error ?? -Infinity;
    const cost = options.cost || Cost.mse;
    if (typeof cost !== "function" && !(typeof cost === "object" && (typeof cost.fn === "function" || typeof cost.calculate === "function"))) {
      throw new Error("Invalid cost function provided to Network.train.");
    }
    const baseRate = options.rate ?? 0.3;
    const dropout = options.dropout || 0;
    if (dropout < 0 || dropout >= 1)
      throw new Error("dropout must be in [0,1)");
    const momentum = options.momentum || 0;
    const batchSize = options.batchSize || 1;
    if (batchSize > set.length)
      throw new Error("Batch size cannot be larger than the dataset length.");
    const accumulationSteps = options.accumulationSteps || 1;
    internalNet._accumulationReduction = options.accumulationReduction === "sum" ? "sum" : "average";
    if (accumulationSteps < 1 || !Number.isFinite(accumulationSteps))
      throw new Error("accumulationSteps must be >=1");
    if (options.gradientClip) {
      const gc = options.gradientClip;
      if (gc.mode)
        internalNet._currentGradClip = {
          mode: gc.mode,
          maxNorm: gc.maxNorm,
          percentile: gc.percentile
        };
      else if (typeof gc.maxNorm === "number")
        internalNet._currentGradClip = { mode: "norm", maxNorm: gc.maxNorm };
      else if (typeof gc.percentile === "number")
        internalNet._currentGradClip = {
          mode: "percentile",
          percentile: gc.percentile
        };
      internalNet._gradClipSeparateBias = !!gc.separateBias;
    } else {
      internalNet._currentGradClip = void 0;
      internalNet._gradClipSeparateBias = false;
    }
    if (options.mixedPrecision) {
      const mp = options.mixedPrecision === true ? { lossScale: 1024 } : options.mixedPrecision;
      internalNet._mixedPrecision.enabled = true;
      internalNet._mixedPrecision.lossScale = mp.lossScale || 1024;
      const dyn = mp.dynamic || {};
      internalNet._mixedPrecisionState.minLossScale = dyn.minScale || 1;
      internalNet._mixedPrecisionState.maxLossScale = dyn.maxScale || 65536;
      internalNet._mpIncreaseEvery = dyn.increaseEvery || dyn.stableStepsForIncrease || 200;
      net.connections.forEach((c) => {
        c._fp32Weight = c.weight;
      });
      net.nodes.forEach((n) => {
        if (n.type !== "input")
          n._fp32Bias = n.bias;
      });
    } else {
      internalNet._mixedPrecision.enabled = false;
      internalNet._mixedPrecision.lossScale = 1;
      internalNet._mpIncreaseEvery = 200;
    }
    const allowedOptimizers = /* @__PURE__ */ new Set([
      "sgd",
      "rmsprop",
      "adagrad",
      "adam",
      "adamw",
      "amsgrad",
      "adamax",
      "nadam",
      "radam",
      "lion",
      "adabelief",
      "lookahead"
    ]);
    let optimizerConfig = void 0;
    if (typeof options.optimizer !== "undefined") {
      if (typeof options.optimizer === "string")
        optimizerConfig = { type: options.optimizer.toLowerCase() };
      else if (typeof options.optimizer === "object" && options.optimizer !== null) {
        optimizerConfig = { ...options.optimizer };
        if (typeof optimizerConfig.type === "string")
          optimizerConfig.type = optimizerConfig.type.toLowerCase();
      } else
        throw new Error("Invalid optimizer option; must be string or object");
      if (!allowedOptimizers.has(optimizerConfig.type))
        throw new Error(`Unknown optimizer type: ${optimizerConfig.type}`);
      if (optimizerConfig.type === "lookahead") {
        if (!optimizerConfig.baseType)
          optimizerConfig.baseType = "adam";
        if (optimizerConfig.baseType === "lookahead")
          throw new Error("Nested lookahead (baseType lookahead) is not supported");
        if (!allowedOptimizers.has(optimizerConfig.baseType))
          throw new Error(`Unknown baseType for lookahead: ${optimizerConfig.baseType}`);
        optimizerConfig.la_k = optimizerConfig.la_k || 5;
        optimizerConfig.la_alpha = optimizerConfig.la_alpha ?? 0.5;
      }
    }
    const iterations = options.iterations ?? Number.MAX_SAFE_INTEGER;
    const start = Date.now();
    let finalError = Infinity;
    const movingAverageWindow = Math.max(1, options.movingAverageWindow || 1);
    const movingAverageType = options.movingAverageType || "sma";
    const emaAlpha = (() => {
      if (movingAverageType !== "ema")
        return void 0;
      if (options.emaAlpha && options.emaAlpha > 0 && options.emaAlpha <= 1)
        return options.emaAlpha;
      return 2 / (movingAverageWindow + 1);
    })();
    const plateauWindow = Math.max(1, options.plateauMovingAverageWindow || movingAverageWindow);
    const plateauType = options.plateauMovingAverageType || movingAverageType;
    const plateauEmaAlpha = (() => {
      if (plateauType !== "ema")
        return void 0;
      if (options.plateauEmaAlpha && options.plateauEmaAlpha > 0 && options.plateauEmaAlpha <= 1)
        return options.plateauEmaAlpha;
      return 2 / (plateauWindow + 1);
    })();
    const earlyStopPatience = options.earlyStopPatience;
    const earlyStopMinDelta = options.earlyStopMinDelta || 0;
    let bestError = Infinity;
    let noImproveCount = 0;
    const recentErrorsCapacity = movingAverageWindow;
    const recentErrorsBuf = new Array(recentErrorsCapacity);
    let recentErrorsCount = 0;
    let recentErrorsWriteIdx = 0;
    const recentErrorsPush = (value) => {
      if (recentErrorsCapacity === 1) {
        recentErrorsBuf[0] = value;
        recentErrorsCount = 1;
        recentErrorsWriteIdx = 0;
        return;
      }
      recentErrorsBuf[recentErrorsWriteIdx] = value;
      recentErrorsWriteIdx = (recentErrorsWriteIdx + 1) % recentErrorsCapacity;
      if (recentErrorsCount < recentErrorsCapacity)
        recentErrorsCount++;
    };
    const recentErrorsChrono = () => {
      if (recentErrorsCount === 0)
        return [];
      if (recentErrorsCount < recentErrorsCapacity)
        return recentErrorsBuf.slice(0, recentErrorsCount);
      const out = new Array(recentErrorsCount);
      const start2 = recentErrorsWriteIdx;
      for (let i = 0; i < recentErrorsCount; i++)
        out[i] = recentErrorsBuf[(start2 + i) % recentErrorsCapacity];
      return out;
    };
    let emaValue = void 0;
    let adaptiveBaseEmaValue = void 0;
    let adaptiveEmaValue = void 0;
    const plateauCapacity = plateauWindow;
    const plateauBuf = new Array(plateauCapacity);
    let plateauCount = 0;
    let plateauWriteIdx = 0;
    const plateauPush = (value) => {
      if (plateauCapacity === 1) {
        plateauBuf[0] = value;
        plateauCount = 1;
        plateauWriteIdx = 0;
        return;
      }
      plateauBuf[plateauWriteIdx] = value;
      plateauWriteIdx = (plateauWriteIdx + 1) % plateauCapacity;
      if (plateauCount < plateauCapacity)
        plateauCount++;
    };
    const plateauChrono = () => {
      if (plateauCount === 0)
        return [];
      if (plateauCount < plateauCapacity)
        return plateauBuf.slice(0, plateauCount);
      const out = new Array(plateauCount);
      const start2 = plateauWriteIdx;
      for (let i = 0; i < plateauCount; i++)
        out[i] = plateauBuf[(start2 + i) % plateauCapacity];
      return out;
    };
    let plateauEmaValue = void 0;
    net.dropout = dropout;
    let performedIterations = 0;
    for (let iter = 1; iter <= iterations; iter++) {
      if (net._maybePrune) {
        net._maybePrune((internalNet._globalEpoch || 0) + iter);
      }
      const trainError = trainSetImpl(net, set, batchSize, accumulationSteps, baseRate, momentum, {}, cost, optimizerConfig);
      performedIterations = iter;
      recentErrorsPush(trainError);
      let monitored = trainError;
      if (movingAverageWindow > 1 || movingAverageType === "ema" || movingAverageType === "adaptive-ema") {
        const recentArr = recentErrorsChrono();
        if (movingAverageType === "median") {
          const sorted = [...recentArr].sort((a, b) => a - b);
          const mid = Math.floor(sorted.length / 2);
          monitored = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        } else if (movingAverageType === "ema") {
          if (emaValue == null)
            emaValue = trainError;
          else
            emaValue = emaValue + emaAlpha * (trainError - emaValue);
          monitored = emaValue;
        } else if (movingAverageType === "adaptive-ema") {
          const mean = recentArr.reduce((a, b) => a + b, 0) / recentArr.length;
          const variance = recentArr.reduce((a, b) => a + (b - mean) * (b - mean), 0) / recentArr.length;
          const baseAlpha = emaAlpha || 2 / (movingAverageWindow + 1);
          const varScaled = variance / Math.max(mean * mean, 1e-8);
          const adaptAlpha = Math.min(0.95, Math.max(baseAlpha, baseAlpha * (1 + 2 * varScaled)));
          if (adaptiveBaseEmaValue == null) {
            adaptiveBaseEmaValue = trainError;
            adaptiveEmaValue = trainError;
          } else {
            adaptiveBaseEmaValue = adaptiveBaseEmaValue + baseAlpha * (trainError - adaptiveBaseEmaValue);
            adaptiveEmaValue = adaptiveEmaValue + adaptAlpha * (trainError - adaptiveEmaValue);
          }
          monitored = Math.min(adaptiveEmaValue, adaptiveBaseEmaValue);
        } else if (movingAverageType === "gaussian") {
          const gaussianWindow = recentArr;
          const windowLength = gaussianWindow.length;
          const sigma = movingAverageWindow / 3 || 1;
          let gaussianWeightSum = 0;
          let gaussianWeightedAccumulator = 0;
          for (let gi = 0; gi < windowLength; gi++) {
            const weight = Math.exp(-0.5 * Math.pow((gi - (windowLength - 1)) / sigma, 2));
            gaussianWeightSum += weight;
            gaussianWeightedAccumulator += weight * gaussianWindow[gi];
          }
          monitored = gaussianWeightedAccumulator / (gaussianWeightSum || 1);
        } else if (movingAverageType === "trimmed") {
          const tailTrimRatio = Math.min(0.49, Math.max(0, options.trimmedRatio || 0.1));
          const sorted = [...recentArr].sort((a, b) => a - b);
          const elementsToDropEachSide = Math.floor(sorted.length * tailTrimRatio);
          const trimmedSegment = sorted.slice(elementsToDropEachSide, sorted.length - elementsToDropEachSide);
          monitored = trimmedSegment.reduce((a, b) => a + b, 0) / (trimmedSegment.length || 1);
        } else if (movingAverageType === "wma") {
          let linearWeightSum = 0;
          let linearWeightedAccumulator = 0;
          for (let li = 0; li < recentArr.length; li++) {
            const weight = li + 1;
            linearWeightSum += weight;
            linearWeightedAccumulator += weight * recentArr[li];
          }
          monitored = linearWeightedAccumulator / (linearWeightSum || 1);
        } else {
          monitored = recentArr.reduce((a, b) => a + b, 0) / recentArr.length;
        }
      }
      finalError = monitored;
      plateauPush(trainError);
      let plateauError = trainError;
      if (plateauWindow > 1 || plateauType === "ema") {
        if (plateauType === "median") {
          const sorted = [...plateauChrono()].sort((a, b) => a - b);
          const mid = Math.floor(sorted.length / 2);
          plateauError = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        } else if (plateauType === "ema") {
          if (plateauEmaValue == null)
            plateauEmaValue = trainError;
          else
            plateauEmaValue = plateauEmaValue + plateauEmaAlpha * (trainError - plateauEmaValue);
          plateauError = plateauEmaValue;
        } else {
          const arr = plateauChrono();
          plateauError = arr.reduce((a, b) => a + b, 0) / arr.length;
        }
      }
      if (typeof options.metricsHook === "function") {
        try {
          options.metricsHook({
            iteration: iter,
            error: finalError,
            plateauError,
            gradNorm: internalNet._lastGradNorm ?? 0
          });
        } catch {
        }
      }
      if (options.checkpoint && typeof options.checkpoint.save === "function") {
        if (options.checkpoint.last) {
          try {
            options.checkpoint.save({
              type: "last",
              iteration: iter,
              error: finalError,
              network: net.toJSON()
            });
          } catch {
          }
        }
        if (options.checkpoint.best) {
          if (finalError < net._checkpointBestError || net._checkpointBestError == null) {
            net._checkpointBestError = finalError;
            try {
              options.checkpoint.save({
                type: "best",
                iteration: iter,
                error: finalError,
                network: net.toJSON()
              });
            } catch {
            }
          }
        }
      }
      if (options.schedule && options.schedule.iterations && iter % options.schedule.iterations === 0) {
        try {
          options.schedule.function({ error: finalError, iteration: iter });
        } catch {
        }
      }
      if (finalError < bestError - earlyStopMinDelta) {
        bestError = finalError;
        noImproveCount = 0;
      } else if (earlyStopPatience) {
        noImproveCount++;
      }
      if (earlyStopPatience && noImproveCount >= earlyStopPatience)
        break;
      if (finalError <= targetError)
        break;
    }
    net.nodes.forEach((n) => {
      if (n.type === "hidden")
        n.mask = 1;
    });
    net.dropout = 0;
    internalNet._globalEpoch = (internalNet._globalEpoch || 0) + performedIterations;
    return {
      error: finalError,
      iterations: performedIterations,
      time: Date.now() - start
    };
  }
  var __trainingInternals;
  var init_network_training = __esm({
    "dist/architecture/network/network.training.js"() {
      "use strict";
      init_methods();
      init_config();
      __trainingInternals = {
        computeMonitoredError,
        computePlateauMetric
      };
    }
  });

  // dist/architecture/network/network.evolve.js
  var network_evolve_exports = {};
  __export(network_evolve_exports, {
    evolveNetwork: () => evolveNetwork
  });
  function computeComplexityPenalty(genome, growth) {
    const n = genome.nodes.length;
    const c = genome.connections.length;
    const g = genome.gates.length;
    const cached = _complexityCache.get(genome);
    if (cached && cached.nodes === n && cached.conns === c && cached.gates === g)
      return cached.value * growth;
    const base = n - genome.input - genome.output + c + g;
    _complexityCache.set(genome, { nodes: n, conns: c, gates: g, value: base });
    return base * growth;
  }
  function buildSingleThreadFitness(set, cost, amount, growth) {
    return (genome) => {
      let score = 0;
      for (let i = 0; i < amount; i++) {
        try {
          score -= genome.test(set, cost).error;
        } catch (e) {
          if (config.warnings)
            console.warn(`Genome evaluation failed: ${e && e.message || e}. Penalizing with -Infinity fitness.`);
          return -Infinity;
        }
      }
      score -= computeComplexityPenalty(genome, growth);
      score = isNaN(score) ? -Infinity : score;
      return score / amount;
    };
  }
  async function buildMultiThreadFitness(set, cost, amount, growth, threads, options) {
    const serializedSet = Multi.serializeDataSet(set);
    const workers = [];
    let WorkerCtor = null;
    try {
      const isNode = typeof process !== "undefined" && !!process.versions?.node;
      if (isNode && Multi.workers?.getNodeTestWorker)
        WorkerCtor = await Multi.workers.getNodeTestWorker();
      else if (!isNode && Multi.workers?.getBrowserTestWorker)
        WorkerCtor = await Multi.workers.getBrowserTestWorker();
    } catch (e) {
      if (config.warnings)
        console.warn("Failed to load worker class; falling back to single-thread path:", e?.message || e);
    }
    if (!WorkerCtor)
      return {
        fitnessFunction: buildSingleThreadFitness(set, cost, amount, growth),
        threads: 1
      };
    for (let i = 0; i < threads; i++) {
      try {
        workers.push(new WorkerCtor(serializedSet, {
          name: cost.name || cost.toString?.() || "cost"
        }));
      } catch (e) {
        if (config.warnings)
          console.warn("Worker spawn failed", e);
      }
    }
    const fitnessFunction = (population) => new Promise((resolve) => {
      if (!workers.length) {
        resolve();
        return;
      }
      const queue = population.slice();
      let active = workers.length;
      const startNext = (worker) => {
        if (!queue.length) {
          if (--active === 0)
            resolve();
          return;
        }
        const genome = queue.shift();
        worker.evaluate(genome).then((result) => {
          if (typeof genome !== "undefined" && typeof result === "number") {
            genome.score = -result - computeComplexityPenalty(genome, growth);
            genome.score = isNaN(result) ? -Infinity : genome.score;
          }
          startNext(worker);
        }).catch(() => startNext(worker));
      };
      workers.forEach((w) => startNext(w));
    });
    options.fitnessPopulation = true;
    options._workerTerminators = () => {
      workers.forEach((w) => {
        try {
          w.terminate && w.terminate();
        } catch {
        }
      });
    };
    return { fitnessFunction, threads };
  }
  async function evolveNetwork(set, options) {
    if (!set || set.length === 0 || set[0].input.length !== this.input || set[0].output.length !== this.output) {
      throw new Error("Dataset is invalid or dimensions do not match network input/output size!");
    }
    options = options || {};
    let targetError = options.error ?? 0.05;
    const growth = options.growth ?? 1e-4;
    const cost = options.cost || Cost.mse;
    const amount = options.amount || 1;
    const log = options.log || 0;
    const schedule = options.schedule;
    const clear = options.clear || false;
    let threads = typeof options.threads === "undefined" ? 1 : options.threads;
    const start = Date.now();
    const evoConfig = {
      targetError,
      growth,
      cost,
      amount,
      log,
      schedule,
      clear,
      threads
    };
    if (typeof options.iterations === "undefined" && typeof options.error === "undefined") {
      throw new Error("At least one stopping condition (`iterations` or `error`) must be specified for evolution.");
    } else if (typeof options.error === "undefined")
      targetError = -1;
    else if (typeof options.iterations === "undefined")
      options.iterations = 0;
    let fitnessFunction;
    if (threads === 1)
      fitnessFunction = buildSingleThreadFitness(set, cost, amount, growth);
    else {
      const multi = await buildMultiThreadFitness(set, cost, amount, growth, threads, options);
      fitnessFunction = multi.fitnessFunction;
      threads = multi.threads;
    }
    options.network = this;
    if (options.populationSize != null && options.popsize == null)
      options.popsize = options.populationSize;
    if (typeof options.speciation === "undefined")
      options.speciation = false;
    const { default: Neat2 } = await Promise.resolve().then(() => (init_neat(), neat_exports));
    const neat = new Neat2(this.input, this.output, fitnessFunction, options);
    if (typeof options.iterations === "number" && options.iterations === 0) {
      if (neat._warnIfNoBestGenome) {
        try {
          neat._warnIfNoBestGenome();
        } catch {
        }
      }
    }
    if (options.popsize && options.popsize <= 10) {
      neat.options.mutationRate = neat.options.mutationRate ?? 0.5;
      neat.options.mutationAmount = neat.options.mutationAmount ?? 1;
    }
    let error = Infinity;
    let bestFitness = -Infinity;
    let bestGenome;
    let infiniteErrorCount = 0;
    const MAX_INF = 5;
    const iterationsSpecified = typeof options.iterations === "number";
    while ((targetError === -1 || error > targetError) && (!iterationsSpecified || neat.generation < options.iterations)) {
      const fittest = await neat.evolve();
      const fitness = fittest.score ?? -Infinity;
      error = -(fitness - computeComplexityPenalty(fittest, growth)) || Infinity;
      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestGenome = fittest;
      }
      if (!isFinite(error) || isNaN(error)) {
        if (++infiniteErrorCount >= MAX_INF)
          break;
      } else
        infiniteErrorCount = 0;
      if (schedule && neat.generation % schedule.iterations === 0) {
        try {
          schedule.function({
            fitness: bestFitness,
            error,
            iteration: neat.generation
          });
        } catch {
        }
      }
    }
    if (typeof bestGenome !== "undefined") {
      this.nodes = bestGenome.nodes;
      this.connections = bestGenome.connections;
      this.selfconns = bestGenome.selfconns;
      this.gates = bestGenome.gates;
      if (clear)
        this.clear();
    } else if (neat._warnIfNoBestGenome) {
      try {
        neat._warnIfNoBestGenome();
      } catch {
      }
    }
    try {
      options._workerTerminators && options._workerTerminators();
    } catch {
    }
    return { error, iterations: neat.generation, time: Date.now() - start };
  }
  var _complexityCache;
  var init_network_evolve = __esm({
    "dist/architecture/network/network.evolve.js"() {
      "use strict";
      init_network();
      init_methods();
      init_config();
      init_multi();
      _complexityCache = /* @__PURE__ */ new WeakMap();
    }
  });

  // dist/architecture/network.js
  var network_exports = {};
  __export(network_exports, {
    default: () => Network
  });
  var Network;
  var init_network = __esm({
    "dist/architecture/network.js"() {
      "use strict";
      init_node();
      init_nodePool();
      init_connection();
      init_multi();
      init_methods();
      init_mutation();
      init_config();
      init_activationArrayPool();
      init_onnx();
      init_network_standalone();
      init_network_topology();
      init_network_slab();
      init_network_prune();
      init_network_gating();
      init_network_deterministic();
      init_network_stats();
      init_network_remove();
      init_network_connect();
      init_network_serialize();
      init_network_genetic();
      Network = class _Network {
        input;
        output;
        score;
        nodes;
        connections;
        gates;
        selfconns;
        dropout = 0;
        _dropConnectProb = 0;
        _lastGradNorm;
        _optimizerStep = 0;
        _weightNoiseStd = 0;
        _weightNoisePerHidden = [];
        _weightNoiseSchedule;
        _stochasticDepth = [];
        _wnOrig;
        _trainingStep = 0;
        _rand = Math.random;
        _rngState;
        _lastStats = null;
        _stochasticDepthSchedule;
        _mixedPrecision = {
          enabled: false,
          lossScale: 1
        };
        _mixedPrecisionState = {
          goodSteps: 0,
          badSteps: 0,
          minLossScale: 1,
          maxLossScale: 65536,
          overflowCount: 0,
          scaleUpEvents: 0,
          scaleDownEvents: 0
        };
        _gradAccumMicroBatches = 0;
        _currentGradClip;
        _lastRawGradNorm = 0;
        _accumulationReduction = "average";
        _gradClipSeparateBias = false;
        _lastGradClipGroupCount = 0;
        _lastOverflowStep = -1;
        _forceNextOverflow = false;
        _pruningConfig;
        _initialConnectionCount;
        _enforceAcyclic = false;
        _topoOrder = null;
        _topoDirty = true;
        _globalEpoch = 0;
        layers;
        _evoInitialConnCount;
        _activationPrecision = "f64";
        _reuseActivationArrays = false;
        _returnTypedActivations = false;
        _activationPool;
        _connWeights;
        _connFrom;
        _connTo;
        _slabDirty = true;
        _useFloat32Weights = true;
        _nodeIndexDirty = true;
        _outStart;
        _outOrder;
        _adjDirty = true;
        _fastA;
        _fastS;
        _preferredChainEdge;
        _canUseFastSlab(training) {
          return canUseFastSlab.call(this, training);
        }
        _fastSlabActivate(input) {
          return fastSlabActivate.call(this, input);
        }
        rebuildConnectionSlab(force = false) {
          return rebuildConnectionSlab.call(this, force);
        }
        getConnectionSlab() {
          return getConnectionSlab.call(this);
        }
        fastSlabActivate(input) {
          return this._fastSlabActivate(input);
        }
        constructor(input, output, options) {
          if (typeof input === "undefined" || typeof output === "undefined") {
            throw new Error("No input or output size given");
          }
          this.input = input;
          this.output = output;
          this.nodes = [];
          this.connections = [];
          this.gates = [];
          this.selfconns = [];
          this.dropout = 0;
          this._enforceAcyclic = options?.enforceAcyclic || false;
          if (options?.activationPrecision) {
            this._activationPrecision = options.activationPrecision;
          } else if (config.float32Mode) {
            this._activationPrecision = "f32";
          }
          if (options?.reuseActivationArrays)
            this._reuseActivationArrays = true;
          if (options?.returnTypedActivations)
            this._returnTypedActivations = true;
          try {
            if (typeof config.poolMaxPerBucket === "number")
              activationArrayPool.setMaxPerBucket(config.poolMaxPerBucket);
            const prewarm = typeof config.poolPrewarmCount === "number" ? config.poolPrewarmCount : 2;
            activationArrayPool.prewarm(this.output, prewarm);
          } catch {
          }
          if (options?.seed !== void 0) {
            this.setSeed(options.seed);
          }
          for (let i = 0; i < this.input + this.output; i++) {
            const type = i < this.input ? "input" : "output";
            if (config.enableNodePooling)
              this.nodes.push(acquireNode({ type, rng: this._rand }));
            else
              this.nodes.push(new Node(type, void 0, this._rand));
          }
          for (let i = 0; i < this.input; i++) {
            for (let j = this.input; j < this.input + this.output; j++) {
              const weight = this._rand() * this.input * Math.sqrt(2 / this.input);
              this.connect(this.nodes[i], this.nodes[j], weight);
            }
          }
          const minHidden = options?.minHidden || 0;
          if (minHidden > 0) {
            while (this.nodes.length < this.input + this.output + minHidden) {
              this.addNodeBetween();
            }
          }
        }
        addNodeBetween() {
          if (this.connections.length === 0)
            return;
          const idx = Math.floor(this._rand() * this.connections.length);
          const conn = this.connections[idx];
          if (!conn)
            return;
          this.disconnect(conn.from, conn.to);
          const newNode = config.enableNodePooling ? acquireNode({ type: "hidden", rng: this._rand }) : new Node("hidden", void 0, this._rand);
          this.nodes.push(newNode);
          this.connect(conn.from, newNode, conn.weight);
          this.connect(newNode, conn.to, 1);
          this._topoDirty = true;
          this._nodeIndexDirty = true;
        }
        enableDropConnect(p) {
          if (p < 0 || p >= 1)
            throw new Error("DropConnect probability must be in [0,1)");
          this._dropConnectProb = p;
        }
        disableDropConnect() {
          this._dropConnectProb = 0;
        }
        setEnforceAcyclic(flag) {
          this._enforceAcyclic = !!flag;
        }
        _computeTopoOrder() {
          return computeTopoOrder.call(this);
        }
        _hasPath(from, to) {
          return hasPath.call(this, from, to);
        }
        configurePruning(cfg) {
          const { start, end, targetSparsity } = cfg;
          if (start < 0 || end < start)
            throw new Error("Invalid pruning schedule window");
          if (targetSparsity <= 0 || targetSparsity >= 1)
            throw new Error("targetSparsity must be in (0,1)");
          this._pruningConfig = {
            start,
            end,
            targetSparsity,
            regrowFraction: cfg.regrowFraction ?? 0,
            frequency: cfg.frequency ?? 1,
            method: cfg.method || "magnitude",
            lastPruneIter: void 0
          };
          this._initialConnectionCount = this.connections.length;
        }
        getCurrentSparsity() {
          return getCurrentSparsity.call(this);
        }
        _maybePrune(iteration) {
          return maybePrune.call(this, iteration);
        }
        pruneToSparsity(targetSparsity, method = "magnitude") {
          return pruneToSparsity.call(this, targetSparsity, method);
        }
        enableWeightNoise(stdDev) {
          if (typeof stdDev === "number") {
            if (stdDev < 0)
              throw new Error("Weight noise stdDev must be >= 0");
            this._weightNoiseStd = stdDev;
            this._weightNoisePerHidden = [];
          } else if (stdDev && Array.isArray(stdDev.perHiddenLayer)) {
            if (!this.layers || this.layers.length < 3)
              throw new Error("Per-hidden-layer weight noise requires a layered network with at least one hidden layer");
            const hiddenLayerCount = this.layers.length - 2;
            if (stdDev.perHiddenLayer.length !== hiddenLayerCount)
              throw new Error(`Expected ${hiddenLayerCount} std dev entries (one per hidden layer), got ${stdDev.perHiddenLayer.length}`);
            if (stdDev.perHiddenLayer.some((s) => s < 0))
              throw new Error("Weight noise std devs must be >= 0");
            this._weightNoiseStd = 0;
            this._weightNoisePerHidden = stdDev.perHiddenLayer.slice();
          } else {
            throw new Error("Invalid weight noise configuration");
          }
        }
        disableWeightNoise() {
          this._weightNoiseStd = 0;
          this._weightNoisePerHidden = [];
        }
        setWeightNoiseSchedule(fn) {
          this._weightNoiseSchedule = fn;
        }
        clearWeightNoiseSchedule() {
          this._weightNoiseSchedule = void 0;
        }
        setRandom(fn) {
          this._rand = fn;
        }
        setSeed(seed) {
          setSeed.call(this, seed);
        }
        testForceOverflow() {
          this._forceNextOverflow = true;
        }
        get trainingStep() {
          return this._trainingStep;
        }
        get lastSkippedLayers() {
          return this._lastSkippedLayers || [];
        }
        snapshotRNG() {
          return snapshotRNG.call(this);
        }
        restoreRNG(fn) {
          restoreRNG.call(this, fn);
        }
        getRNGState() {
          return getRNGState.call(this);
        }
        setRNGState(state) {
          setRNGState.call(this, state);
        }
        setStochasticDepthSchedule(fn) {
          this._stochasticDepthSchedule = fn;
        }
        clearStochasticDepthSchedule() {
          this._stochasticDepthSchedule = void 0;
        }
        getRegularizationStats() {
          return getRegularizationStats.call(this);
        }
        setStochasticDepth(survival) {
          if (!Array.isArray(survival))
            throw new Error("survival must be an array");
          if (survival.some((p) => p <= 0 || p > 1))
            throw new Error("Stochastic depth survival probs must be in (0,1]");
          if (!this.layers || this.layers.length === 0)
            throw new Error("Stochastic depth requires layer-based network");
          const hiddenLayerCount = Math.max(0, this.layers.length - 2);
          if (survival.length !== hiddenLayerCount)
            throw new Error(`Expected ${hiddenLayerCount} survival probabilities for hidden layers, got ${survival.length}`);
          this._stochasticDepth = survival.slice();
        }
        disableStochasticDepth() {
          this._stochasticDepth = [];
        }
        clone() {
          return _Network.fromJSON(this.toJSON());
        }
        resetDropoutMasks() {
          if (this.layers && this.layers.length > 0) {
            for (const layer of this.layers) {
              if (typeof layer.nodes !== "undefined") {
                for (const node of layer.nodes) {
                  if (typeof node.mask !== "undefined")
                    node.mask = 1;
                }
              }
            }
          } else {
            for (const node of this.nodes) {
              if (typeof node.mask !== "undefined")
                node.mask = 1;
            }
          }
        }
        standalone() {
          return generateStandalone(this);
        }
        activate(input, training = false, maxActivationDepth = 1e3) {
          if (this._enforceAcyclic && this._topoDirty)
            this._computeTopoOrder();
          if (!Array.isArray(input) || input.length !== this.input) {
            throw new Error(`Input size mismatch: expected ${this.input}, got ${input ? input.length : "undefined"}`);
          }
          if (this._canUseFastSlab(training)) {
            try {
              return this._fastSlabActivate(input);
            } catch {
            }
          }
          const outputArr = activationArrayPool.acquire(this.output);
          if (!this.nodes || this.nodes.length === 0) {
            throw new Error("Network structure is corrupted or empty. No nodes found.");
          }
          const output = outputArr;
          this._lastSkippedLayers = [];
          const stats = {
            droppedHiddenNodes: 0,
            totalHiddenNodes: 0,
            droppedConnections: 0,
            totalConnections: this.connections.length,
            skippedLayers: [],
            weightNoise: { count: 0, sumAbs: 0, maxAbs: 0, meanAbs: 0 }
          };
          let appliedWeightNoise = false;
          let dynamicStd = this._weightNoiseStd;
          if (training) {
            if (this._weightNoiseSchedule)
              dynamicStd = this._weightNoiseSchedule(this._trainingStep);
            if (dynamicStd > 0 || this._weightNoisePerHidden.length > 0) {
              for (const c of this.connections) {
                if (c._origWeightNoise != null)
                  continue;
                c._origWeightNoise = c.weight;
                let std = dynamicStd;
                if (this._weightNoisePerHidden.length > 0 && this.layers) {
                  let fromLayerIndex = -1;
                  for (let li = 0; li < this.layers.length; li++) {
                    if (this.layers[li].nodes.includes(c.from)) {
                      fromLayerIndex = li;
                      break;
                    }
                  }
                  if (fromLayerIndex > 0 && fromLayerIndex < this.layers.length) {
                    const hiddenIdx = fromLayerIndex - 1;
                    if (hiddenIdx >= 0 && hiddenIdx < this._weightNoisePerHidden.length)
                      std = this._weightNoisePerHidden[hiddenIdx];
                  }
                }
                if (std > 0) {
                  const noise = std * _Network._gaussianRand(this._rand);
                  c.weight += noise;
                  c._wnLast = noise;
                  appliedWeightNoise = true;
                } else {
                  c._wnLast = 0;
                }
              }
            }
          }
          if (training && this._stochasticDepthSchedule && this._stochasticDepth.length > 0) {
            const updated = this._stochasticDepthSchedule(this._trainingStep, this._stochasticDepth.slice());
            if (Array.isArray(updated) && updated.length === this._stochasticDepth.length && !updated.some((p) => p <= 0 || p > 1)) {
              this._stochasticDepth = updated.slice();
            }
          }
          if (this.layers && this.layers.length > 0 && this._stochasticDepth.length > 0) {
            let acts;
            for (let li = 0; li < this.layers.length; li++) {
              const layer = this.layers[li];
              const isHidden = li > 0 && li < this.layers.length - 1;
              let skip = false;
              if (training && isHidden) {
                const hiddenIndex = li - 1;
                if (hiddenIndex < this._stochasticDepth.length) {
                  const surviveProb = this._stochasticDepth[hiddenIndex];
                  skip = this._rand() >= surviveProb;
                  if (skip) {
                    if (!acts || acts.length !== layer.nodes.length)
                      skip = false;
                  }
                  if (!skip) {
                    const raw2 = li === 0 ? layer.activate(input, training) : layer.activate(void 0, training);
                    acts = surviveProb < 1 ? raw2.map((a) => a * (1 / surviveProb)) : raw2;
                    continue;
                  }
                }
              }
              if (skip) {
                this._lastSkippedLayers.push(li);
                stats.skippedLayers.push(li);
                continue;
              }
              const raw = li === 0 ? layer.activate(input, training) : layer.activate(void 0, training);
              acts = raw;
            }
            if (acts) {
              for (let i = 0; i < acts.length && i < this.output; i++)
                output[i] = acts[i];
            }
          } else if (this.layers && this.layers.length > 0) {
            let lastActs;
            for (let li = 0; li < this.layers.length; li++) {
              const layer = this.layers[li];
              const isHidden = li > 0 && li < this.layers.length - 1;
              const raw = li === 0 ? layer.activate(input, false) : layer.activate(void 0, false);
              if (isHidden && training && this.dropout > 0) {
                let dropped = 0;
                for (const node of layer.nodes) {
                  node.mask = this._rand() < this.dropout ? 0 : 1;
                  stats.totalHiddenNodes++;
                  if (node.mask === 0)
                    stats.droppedHiddenNodes++;
                  if (node.mask === 0) {
                    node.activation = 0;
                    dropped++;
                  }
                }
                if (dropped === layer.nodes.length && layer.nodes.length > 0) {
                  const idx = Math.floor(this._rand() * layer.nodes.length);
                  layer.nodes[idx].mask = 1;
                  layer.nodes[idx].activation = raw[idx];
                }
              } else if (isHidden) {
                for (const node of layer.nodes)
                  node.mask = 1;
              }
              lastActs = raw;
            }
            if (lastActs) {
              if (this._reuseActivationArrays) {
                for (let i = 0; i < lastActs.length && i < this.output; i++)
                  output[i] = lastActs[i];
              } else {
                for (let i = 0; i < lastActs.length && i < this.output; i++)
                  output[i] = lastActs[i];
              }
            }
          } else {
            const hiddenNodes = this.nodes.filter((node) => node.type === "hidden");
            let droppedCount = 0;
            if (training && this.dropout > 0) {
              for (const node of hiddenNodes) {
                node.mask = this._rand() < this.dropout ? 0 : 1;
                stats.totalHiddenNodes++;
                if (node.mask === 0) {
                  droppedCount++;
                  stats.droppedHiddenNodes++;
                }
              }
              if (droppedCount === hiddenNodes.length && hiddenNodes.length > 0) {
                const idx = Math.floor(this._rand() * hiddenNodes.length);
                hiddenNodes[idx].mask = 1;
              }
            } else {
              for (const node of hiddenNodes)
                node.mask = 1;
            }
            if (training && this._weightNoiseStd > 0) {
              if (!this._wnOrig)
                this._wnOrig = new Array(this.connections.length);
              for (let ci = 0; ci < this.connections.length; ci++) {
                const c = this.connections[ci];
                if (c._origWeightNoise != null)
                  continue;
                c._origWeightNoise = c.weight;
                const noise = this._weightNoiseStd * _Network._gaussianRand(this._rand);
                c.weight += noise;
              }
            }
            let outIndex = 0;
            this.nodes.forEach((node, index) => {
              if (node.type === "input") {
                node.activate(input[index]);
              } else if (node.type === "output") {
                const activation = node.activate();
                output[outIndex++] = activation;
              } else {
                node.activate();
              }
            });
            if (training && this._dropConnectProb > 0) {
              for (const conn of this.connections) {
                const mask = this._rand() < this._dropConnectProb ? 0 : 1;
                if (mask === 0)
                  stats.droppedConnections++;
                conn.dcMask = mask;
                if (mask === 0) {
                  if (conn._origWeight == null)
                    conn._origWeight = conn.weight;
                  conn.weight = 0;
                } else if (conn._origWeight != null) {
                  conn.weight = conn._origWeight;
                  delete conn._origWeight;
                }
              }
            } else {
              for (const conn of this.connections) {
                if (conn._origWeight != null) {
                  conn.weight = conn._origWeight;
                  delete conn._origWeight;
                }
                conn.dcMask = 1;
              }
            }
            if (training && appliedWeightNoise) {
              for (const c of this.connections) {
                if (c._origWeightNoise != null) {
                  c.weight = c._origWeightNoise;
                  delete c._origWeightNoise;
                }
              }
            }
          }
          if (training)
            this._trainingStep++;
          if (stats.weightNoise.count > 0)
            stats.weightNoise.meanAbs = stats.weightNoise.sumAbs / stats.weightNoise.count;
          this._lastStats = stats;
          const result = Array.from(output);
          activationArrayPool.release(output);
          return result;
        }
        static _gaussianRand(rng = Math.random) {
          let u = 0, v = 0;
          while (u === 0)
            u = rng();
          while (v === 0)
            v = rng();
          return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }
        noTraceActivate(input) {
          const { noTraceActivate: noTraceActivate2 } = (init_network_activate(), __toCommonJS(network_activate_exports));
          return noTraceActivate2.call(this, input);
        }
        activateRaw(input, training = false, maxActivationDepth = 1e3) {
          const { activateRaw: activateRaw2 } = (init_network_activate(), __toCommonJS(network_activate_exports));
          return activateRaw2.call(this, input, training, maxActivationDepth);
        }
        activateBatch(inputs, training = false) {
          const { activateBatch: activateBatch2 } = (init_network_activate(), __toCommonJS(network_activate_exports));
          return activateBatch2.call(this, inputs, training);
        }
        propagate(rate, momentum, update, target, regularization = 0, costDerivative) {
          if (!target || target.length !== this.output) {
            throw new Error("Output target length should match network output length");
          }
          let targetIndex = target.length;
          for (let i = this.nodes.length - 1; i >= this.nodes.length - this.output; i--) {
            if (costDerivative) {
              this.nodes[i].propagate(rate, momentum, update, regularization, target[--targetIndex], costDerivative);
            } else {
              this.nodes[i].propagate(rate, momentum, update, regularization, target[--targetIndex]);
            }
          }
          for (let i = this.nodes.length - this.output - 1; i >= this.input; i--) {
            this.nodes[i].propagate(rate, momentum, update, regularization);
          }
        }
        clear() {
          this.nodes.forEach((node) => node.clear());
        }
        mutate(method) {
          const { mutateImpl: mutateImpl2 } = (init_network_mutate(), __toCommonJS(network_mutate_exports));
          return mutateImpl2.call(this, method);
        }
        connect(from, to, weight) {
          return connect.call(this, from, to, weight);
        }
        gate(node, connection) {
          return gate.call(this, node, connection);
        }
        remove(node) {
          const result = removeNode.call(this, node);
          if (config.enableNodePooling) {
            try {
              releaseNode(node);
            } catch {
            }
          }
          return result;
        }
        disconnect(from, to) {
          return disconnect.call(this, from, to);
        }
        ungate(connection) {
          return ungate.call(this, connection);
        }
        _applyGradientClipping(cfg) {
          const { applyGradientClippingImpl: applyGradientClippingImpl2 } = (init_network_training(), __toCommonJS(network_training_exports));
          applyGradientClippingImpl2(this, cfg);
        }
        train(set, options) {
          const { trainImpl: trainImpl2 } = (init_network_training(), __toCommonJS(network_training_exports));
          return trainImpl2(this, set, options);
        }
        getRawGradientNorm() {
          return this._lastRawGradNorm;
        }
        getLossScale() {
          return this._mixedPrecision.lossScale;
        }
        getLastGradClipGroupCount() {
          return this._lastGradClipGroupCount;
        }
        getTrainingStats() {
          return {
            gradNorm: this._lastGradNorm ?? 0,
            gradNormRaw: this._lastRawGradNorm,
            lossScale: this._mixedPrecision.lossScale,
            optimizerStep: this._optimizerStep,
            mp: {
              good: this._mixedPrecisionState.goodSteps,
              bad: this._mixedPrecisionState.badSteps,
              overflowCount: this._mixedPrecisionState.overflowCount || 0,
              scaleUps: this._mixedPrecisionState.scaleUpEvents || 0,
              scaleDowns: this._mixedPrecisionState.scaleDownEvents || 0,
              lastOverflowStep: this._lastOverflowStep
            }
          };
        }
        static adjustRateForAccumulation(rate, accumulationSteps, reduction) {
          if (reduction === "sum" && accumulationSteps > 1)
            return rate / accumulationSteps;
          return rate;
        }
        async evolve(set, options) {
          const { evolveNetwork: evolveNetwork2 } = await Promise.resolve().then(() => (init_network_evolve(), network_evolve_exports));
          return evolveNetwork2.call(this, set, options);
        }
        test(set, cost) {
          if (!Array.isArray(set) || set.length === 0) {
            throw new Error("Test set is empty or not an array.");
          }
          for (const sample of set) {
            if (!Array.isArray(sample.input) || sample.input.length !== this.input) {
              throw new Error(`Test sample input size mismatch: expected ${this.input}, got ${sample.input ? sample.input.length : "undefined"}`);
            }
            if (!Array.isArray(sample.output) || sample.output.length !== this.output) {
              throw new Error(`Test sample output size mismatch: expected ${this.output}, got ${sample.output ? sample.output.length : "undefined"}`);
            }
          }
          let error = 0;
          const costFn = cost || Cost.mse;
          const start = Date.now();
          this.nodes.forEach((node) => {
            if (node.type === "hidden")
              node.mask = 1;
          });
          const previousDropout = this.dropout;
          if (this.dropout > 0) {
            this.dropout = 0;
          }
          set.forEach((data) => {
            const output = this.noTraceActivate(data.input);
            error += costFn(data.output, output);
          });
          this.dropout = previousDropout;
          return { error: error / set.length, time: Date.now() - start };
        }
        serialize() {
          return serialize.call(this);
        }
        static deserialize(data, inputSize, outputSize) {
          return deserialize(data, inputSize, outputSize);
        }
        toJSON() {
          return toJSONImpl.call(this);
        }
        static fromJSON(json) {
          return fromJSONImpl(json);
        }
        static crossOver(network1, network2, equal = false) {
          return crossOver(network1, network2, equal);
        }
        set(values) {
          this.nodes.forEach((node) => {
            if (typeof values.bias !== "undefined") {
              node.bias = values.bias;
            }
            if (typeof values.squash !== "undefined") {
              node.squash = values.squash;
            }
          });
        }
        toONNX() {
          return exportToONNX(this);
        }
        static createMLP(inputCount, hiddenCounts, outputCount) {
          const inputNodes = Array.from({ length: inputCount }, () => new Node("input"));
          const hiddenLayers = hiddenCounts.map((count) => Array.from({ length: count }, () => new Node("hidden")));
          const outputNodes = Array.from({ length: outputCount }, () => new Node("output"));
          const allNodes = [...inputNodes, ...hiddenLayers.flat(), ...outputNodes];
          const net = new _Network(inputCount, outputCount);
          net.nodes = allNodes;
          let prevLayer = inputNodes;
          for (const layer of hiddenLayers) {
            for (const to of layer) {
              for (const from of prevLayer) {
                from.connect(to);
              }
            }
            prevLayer = layer;
          }
          for (const to of outputNodes) {
            for (const from of prevLayer) {
              from.connect(to);
            }
          }
          net.connections = net.nodes.flatMap((n) => n.connections.out);
          net._topoDirty = true;
          return net;
        }
        static rebuildConnections(net) {
          const allConnections = /* @__PURE__ */ new Set();
          net.nodes.forEach((node) => {
            node.connections.out.forEach((conn) => {
              allConnections.add(conn);
            });
          });
          net.connections = Array.from(allConnections);
        }
      };
    }
  });

  // dist/neat/neat.mutation.js
  function mutate() {
    const methods = (init_methods(), __toCommonJS(methods_exports));
    for (const genome of this.population) {
      if (this.options.adaptiveMutation?.enabled) {
        if (genome._mutRate === void 0) {
          genome._mutRate = this.options.mutationRate !== void 0 ? this.options.mutationRate : this.options.adaptiveMutation.initialRate ?? (this.options.mutationRate || 0.7);
          if (this.options.adaptiveMutation.adaptAmount)
            genome._mutAmount = this.options.mutationAmount || 1;
        }
      }
      const effectiveRate = this.options.mutationRate !== void 0 ? this.options.mutationRate : this.options.adaptiveMutation?.enabled ? genome._mutRate : this.options.mutationRate || 0.7;
      const effectiveAmount = this.options.adaptiveMutation?.enabled && this.options.adaptiveMutation.adaptAmount ? genome._mutAmount ?? (this.options.mutationAmount || 1) : this.options.mutationAmount || 1;
      if (this._getRNG()() <= effectiveRate) {
        for (let iteration = 0; iteration < effectiveAmount; iteration++) {
          let mutationMethod = this.selectMutationMethod(genome, false);
          if (Array.isArray(mutationMethod)) {
            const operatorArray = mutationMethod;
            mutationMethod = operatorArray[Math.floor(this._getRNG()() * operatorArray.length)];
          }
          if (mutationMethod && mutationMethod.name) {
            const beforeNodes = genome.nodes.length;
            const beforeConns = genome.connections.length;
            if (mutationMethod === methods.mutation.ADD_NODE) {
              this._mutateAddNodeReuse(genome);
              try {
                genome.mutate(methods.mutation.MOD_WEIGHT);
              } catch {
              }
              this._invalidateGenomeCaches(genome);
            } else if (mutationMethod === methods.mutation.ADD_CONN) {
              this._mutateAddConnReuse(genome);
              try {
                genome.mutate(methods.mutation.MOD_WEIGHT);
              } catch {
              }
              this._invalidateGenomeCaches(genome);
            } else {
              genome.mutate(mutationMethod);
              if (mutationMethod === methods.mutation.ADD_GATE || mutationMethod === methods.mutation.SUB_NODE || mutationMethod === methods.mutation.SUB_CONN || mutationMethod === methods.mutation.ADD_SELF_CONN || mutationMethod === methods.mutation.ADD_BACK_CONN) {
                this._invalidateGenomeCaches(genome);
              }
            }
            if (this._getRNG()() < EXTRA_CONNECTION_PROBABILITY)
              this._mutateAddConnReuse(genome);
            if (this.options.operatorAdaptation?.enabled) {
              const statsRecord = this._operatorStats.get(mutationMethod.name) || {
                success: 0,
                attempts: 0
              };
              statsRecord.attempts++;
              const afterNodes = genome.nodes.length;
              const afterConns = genome.connections.length;
              if (afterNodes > beforeNodes || afterConns > beforeConns)
                statsRecord.success++;
              this._operatorStats.set(mutationMethod.name, statsRecord);
            }
          }
        }
      }
    }
  }
  function mutateAddNodeReuse(genome) {
    if (genome.connections.length === 0) {
      const inputNode = genome.nodes.find((n) => n.type === "input");
      const outputNode = genome.nodes.find((n) => n.type === "output");
      if (inputNode && outputNode) {
        try {
          genome.connect(inputNode, outputNode, 1);
        } catch {
        }
      }
    }
    const enabledConnections = genome.connections.filter((c) => c.enabled !== false);
    if (!enabledConnections.length)
      return;
    const chosenConn = enabledConnections[Math.floor(this._getRNG()() * enabledConnections.length)];
    const fromGeneId = chosenConn.from.geneId;
    const toGeneId = chosenConn.to.geneId;
    const splitKey = fromGeneId + "->" + toGeneId;
    const originalWeight = chosenConn.weight;
    genome.disconnect(chosenConn.from, chosenConn.to);
    let splitRecord = this._nodeSplitInnovations.get(splitKey);
    const NodeClass = (init_node(), __toCommonJS(node_exports)).default;
    if (!splitRecord) {
      const newNode = new NodeClass("hidden");
      const inConn = genome.connect(chosenConn.from, newNode, 1)[0];
      const outConn = genome.connect(newNode, chosenConn.to, originalWeight)[0];
      if (inConn)
        inConn.innovation = this._nextGlobalInnovation++;
      if (outConn)
        outConn.innovation = this._nextGlobalInnovation++;
      splitRecord = {
        newNodeGeneId: newNode.geneId,
        inInnov: inConn?.innovation,
        outInnov: outConn?.innovation
      };
      this._nodeSplitInnovations.set(splitKey, splitRecord);
      const toIndex = genome.nodes.indexOf(chosenConn.to);
      const insertIndex = Math.min(toIndex, genome.nodes.length - genome.output);
      genome.nodes.splice(insertIndex, 0, newNode);
    } else {
      const newNode = new NodeClass("hidden");
      newNode.geneId = splitRecord.newNodeGeneId;
      const toIndex = genome.nodes.indexOf(chosenConn.to);
      const insertIndex = Math.min(toIndex, genome.nodes.length - genome.output);
      genome.nodes.splice(insertIndex, 0, newNode);
      const inConn = genome.connect(chosenConn.from, newNode, 1)[0];
      const outConn = genome.connect(newNode, chosenConn.to, originalWeight)[0];
      if (inConn)
        inConn.innovation = splitRecord.inInnov;
      if (outConn)
        outConn.innovation = splitRecord.outInnov;
    }
  }
  function mutateAddConnReuse(genome) {
    const candidatePairs = [];
    for (let i = 0; i < genome.nodes.length - genome.output; i++) {
      const fromNode2 = genome.nodes[i];
      for (let j = Math.max(i + 1, genome.input); j < genome.nodes.length; j++) {
        const toNode2 = genome.nodes[j];
        if (!fromNode2.isProjectingTo(toNode2))
          candidatePairs.push([fromNode2, toNode2]);
      }
    }
    if (!candidatePairs.length)
      return;
    const reuseCandidates = candidatePairs.filter((pair) => {
      const idA2 = pair[0].geneId;
      const idB2 = pair[1].geneId;
      const symmetricKey2 = idA2 < idB2 ? idA2 + "::" + idB2 : idB2 + "::" + idA2;
      return this._connInnovations.has(symmetricKey2);
    });
    const hiddenPairs = reuseCandidates.length ? [] : candidatePairs.filter((pair) => pair[0].type === "hidden" && pair[1].type === "hidden");
    const pool2 = reuseCandidates.length ? reuseCandidates : hiddenPairs.length ? hiddenPairs : candidatePairs;
    const chosenPair = pool2.length === 1 ? pool2[0] : pool2[Math.floor(this._getRNG()() * pool2.length)];
    const fromNode = chosenPair[0];
    const toNode = chosenPair[1];
    const idA = fromNode.geneId;
    const idB = toNode.geneId;
    const symmetricKey = idA < idB ? idA + "::" + idB : idB + "::" + idA;
    if (genome._enforceAcyclic) {
      const createsCycle = (() => {
        const stack = [toNode];
        const seen = /* @__PURE__ */ new Set();
        while (stack.length) {
          const n = stack.pop();
          if (n === fromNode)
            return true;
          if (seen.has(n))
            continue;
          seen.add(n);
          for (const c of n.connections.out)
            stack.push(c.to);
        }
        return false;
      })();
      if (createsCycle)
        return;
    }
    const conn = genome.connect(fromNode, toNode)[0];
    if (!conn)
      return;
    if (this._connInnovations.has(symmetricKey)) {
      conn.innovation = this._connInnovations.get(symmetricKey);
    } else {
      const innov = this._nextGlobalInnovation++;
      conn.innovation = innov;
      this._connInnovations.set(symmetricKey, innov);
      const legacyForward = idA + "::" + idB;
      const legacyReverse = idB + "::" + idA;
      this._connInnovations.set(legacyForward, innov);
      this._connInnovations.set(legacyReverse, innov);
    }
  }
  function ensureMinHiddenNodes(network, multiplierOverride) {
    const maxNodes = this.options.maxNodes || Infinity;
    const minHidden = Math.min(this.getMinimumHiddenSize(multiplierOverride), maxNodes - network.nodes.filter((n) => n.type !== "hidden").length);
    const inputNodes = network.nodes.filter((n) => n.type === "input");
    const outputNodes = network.nodes.filter((n) => n.type === "output");
    const hiddenNodes = network.nodes.filter((n) => n.type === "hidden");
    if (inputNodes.length === 0 || outputNodes.length === 0) {
      try {
        console.warn("Network is missing input or output nodes \u2014 skipping minHidden enforcement");
      } catch {
      }
      return;
    }
    const existingCount = hiddenNodes.length;
    for (let i = existingCount; i < minHidden && network.nodes.length < maxNodes; i++) {
      const NodeClass = (init_node(), __toCommonJS(node_exports)).default;
      const newNode = new NodeClass("hidden");
      network.nodes.push(newNode);
      hiddenNodes.push(newNode);
    }
    for (const hiddenNode of hiddenNodes) {
      if (hiddenNode.connections.in.length === 0) {
        const candidates = inputNodes.concat(hiddenNodes.filter((n) => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const source = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(source, hiddenNode);
          } catch {
          }
        }
      }
      if (hiddenNode.connections.out.length === 0) {
        const candidates = outputNodes.concat(hiddenNodes.filter((n) => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const target = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(hiddenNode, target);
          } catch {
          }
        }
      }
    }
    const NetworkClass = (init_network(), __toCommonJS(network_exports)).default;
    NetworkClass.rebuildConnections(network);
  }
  function ensureNoDeadEnds(network) {
    const inputNodes = network.nodes.filter((n) => n.type === "input");
    const outputNodes = network.nodes.filter((n) => n.type === "output");
    const hiddenNodes = network.nodes.filter((n) => n.type === "hidden");
    const hasOutgoing = (node) => node.connections && node.connections.out && node.connections.out.length > 0;
    const hasIncoming = (node) => node.connections && node.connections.in && node.connections.in.length > 0;
    for (const inputNode of inputNodes) {
      if (!hasOutgoing(inputNode)) {
        const candidates = hiddenNodes.length > 0 ? hiddenNodes : outputNodes;
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const target = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(inputNode, target);
          } catch {
          }
        }
      }
    }
    for (const outputNode of outputNodes) {
      if (!hasIncoming(outputNode)) {
        const candidates = hiddenNodes.length > 0 ? hiddenNodes : inputNodes;
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const source = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(source, outputNode);
          } catch {
          }
        }
      }
    }
    for (const hiddenNode of hiddenNodes) {
      if (!hasIncoming(hiddenNode)) {
        const candidates = inputNodes.concat(hiddenNodes.filter((n) => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const source = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(source, hiddenNode);
          } catch {
          }
        }
      }
      if (!hasOutgoing(hiddenNode)) {
        const candidates = outputNodes.concat(hiddenNodes.filter((n) => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const target = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(hiddenNode, target);
          } catch {
          }
        }
      }
    }
  }
  function selectMutationMethod(genome, rawReturnForTest = true) {
    const methods = (init_methods(), __toCommonJS(methods_exports));
    const isFFWDirect = this.options.mutation === methods.mutation.FFW;
    const isFFWNested = Array.isArray(this.options.mutation) && this.options.mutation.length === 1 && this.options.mutation[0] === methods.mutation.FFW;
    if ((isFFWDirect || isFFWNested) && rawReturnForTest)
      return methods.mutation.FFW;
    if (isFFWDirect)
      return methods.mutation.FFW[Math.floor(this._getRNG()() * methods.mutation.FFW.length)];
    if (isFFWNested)
      return methods.mutation.FFW[Math.floor(this._getRNG()() * methods.mutation.FFW.length)];
    let pool2 = this.options.mutation;
    if (rawReturnForTest && Array.isArray(pool2) && pool2.length === methods.mutation.FFW.length && pool2.every((m, i) => m && m.name === methods.mutation.FFW[i].name)) {
      return methods.mutation.FFW;
    }
    if (pool2.length === 1 && Array.isArray(pool2[0]) && pool2[0].length)
      pool2 = pool2[0];
    if (this.options.phasedComplexity?.enabled && this._phase) {
      pool2 = pool2.filter((m) => !!m);
      if (this._phase === "simplify") {
        const simplifyPool = pool2.filter((m) => m && m.name && m.name.startsWith && m.name.startsWith("SUB_"));
        if (simplifyPool.length)
          pool2 = [...pool2, ...simplifyPool];
      } else if (this._phase === "complexify") {
        const addPool = pool2.filter((m) => m && m.name && m.name.startsWith && m.name.startsWith("ADD_"));
        if (addPool.length)
          pool2 = [...pool2, ...addPool];
      }
    }
    if (this.options.operatorAdaptation?.enabled) {
      const boost = this.options.operatorAdaptation.boost ?? 2;
      const stats = this._operatorStats;
      const augmented = [];
      for (const m of pool2) {
        augmented.push(m);
        const st = stats.get(m.name);
        if (st && st.attempts > 5) {
          const ratio = st.success / st.attempts;
          if (ratio > 0.55) {
            for (let i = 0; i < Math.min(boost, Math.floor(ratio * boost)); i++)
              augmented.push(m);
          }
        }
      }
      pool2 = augmented;
    }
    let mutationMethod = pool2[Math.floor(this._getRNG()() * pool2.length)];
    if (mutationMethod === methods.mutation.ADD_GATE && genome.gates.length >= (this.options.maxGates || Infinity))
      return null;
    if (mutationMethod === methods.mutation.ADD_NODE && genome.nodes.length >= (this.options.maxNodes || Infinity))
      return null;
    if (mutationMethod === methods.mutation.ADD_CONN && genome.connections.length >= (this.options.maxConns || Infinity))
      return null;
    if (this.options.operatorBandit?.enabled) {
      const c = this.options.operatorBandit.c ?? 1.4;
      const minA = this.options.operatorBandit.minAttempts ?? 5;
      const stats = this._operatorStats;
      for (const m of pool2)
        if (!stats.has(m.name))
          stats.set(m.name, { success: 0, attempts: 0 });
      const totalAttempts = Array.from(stats.values()).reduce((a, s) => a + s.attempts, 0) + EPSILON;
      let best = mutationMethod;
      let bestVal = -Infinity;
      for (const m of pool2) {
        const st = stats.get(m.name);
        const mean = st.attempts > 0 ? st.success / st.attempts : 0;
        const bonus = st.attempts < minA ? Infinity : c * Math.sqrt(Math.log(totalAttempts) / (st.attempts + EPSILON));
        const val = mean + bonus;
        if (val > bestVal) {
          bestVal = val;
          best = m;
        }
      }
      mutationMethod = best;
    }
    if (mutationMethod === methods.mutation.ADD_GATE && genome.gates.length >= (this.options.maxGates || Infinity))
      return null;
    if (!this.options.allowRecurrent && (mutationMethod === methods.mutation.ADD_BACK_CONN || mutationMethod === methods.mutation.ADD_SELF_CONN))
      return null;
    return mutationMethod;
  }
  var init_neat_mutation = __esm({
    "dist/neat/neat.mutation.js"() {
      "use strict";
      init_neat_constants();
    }
  });

  // dist/neat/neat.multiobjective.js
  function fastNonDominated(pop) {
    const objectiveDescriptors = this._getObjectives();
    const valuesMatrix = pop.map((genomeItem) => objectiveDescriptors.map((descriptor) => {
      try {
        return descriptor.accessor(genomeItem);
      } catch {
        return 0;
      }
    }));
    const vectorDominates = (valuesA, valuesB) => {
      let strictlyBetter = false;
      for (let objectiveIndex = 0; objectiveIndex < valuesA.length; objectiveIndex++) {
        const direction = objectiveDescriptors[objectiveIndex].direction || "max";
        if (direction === "max") {
          if (valuesA[objectiveIndex] < valuesB[objectiveIndex])
            return false;
          if (valuesA[objectiveIndex] > valuesB[objectiveIndex])
            strictlyBetter = true;
        } else {
          if (valuesA[objectiveIndex] > valuesB[objectiveIndex])
            return false;
          if (valuesA[objectiveIndex] < valuesB[objectiveIndex])
            strictlyBetter = true;
        }
      }
      return strictlyBetter;
    };
    const paretoFronts = [];
    const dominationCounts = new Array(pop.length).fill(0);
    const dominatedIndicesByIndex = pop.map(() => []);
    const firstFrontIndices = [];
    for (let pIndex = 0; pIndex < pop.length; pIndex++) {
      for (let qIndex = 0; qIndex < pop.length; qIndex++) {
        if (pIndex === qIndex)
          continue;
        if (vectorDominates(valuesMatrix[pIndex], valuesMatrix[qIndex]))
          dominatedIndicesByIndex[pIndex].push(qIndex);
        else if (vectorDominates(valuesMatrix[qIndex], valuesMatrix[pIndex]))
          dominationCounts[pIndex]++;
      }
      if (dominationCounts[pIndex] === 0)
        firstFrontIndices.push(pIndex);
    }
    let currentFrontIndices = firstFrontIndices;
    let currentFrontRank = 0;
    while (currentFrontIndices.length) {
      const nextFrontIndices = [];
      for (const pIndex of currentFrontIndices) {
        pop[pIndex]._moRank = currentFrontRank;
        for (const qIndex of dominatedIndicesByIndex[pIndex]) {
          dominationCounts[qIndex]--;
          if (dominationCounts[qIndex] === 0)
            nextFrontIndices.push(qIndex);
        }
      }
      paretoFronts.push(currentFrontIndices.map((i) => pop[i]));
      currentFrontIndices = nextFrontIndices;
      currentFrontRank++;
      if (currentFrontRank > 50)
        break;
    }
    for (const front of paretoFronts) {
      if (front.length === 0)
        continue;
      for (const genomeItem of front)
        genomeItem._moCrowd = 0;
      for (let objectiveIndex = 0; objectiveIndex < objectiveDescriptors.length; objectiveIndex++) {
        const sortedByCurrentObjective = front.slice().sort((genomeA, genomeB) => {
          const valA = objectiveDescriptors[objectiveIndex].accessor(genomeA);
          const valB = objectiveDescriptors[objectiveIndex].accessor(genomeB);
          return valA - valB;
        });
        sortedByCurrentObjective[0]._moCrowd = Infinity;
        sortedByCurrentObjective[sortedByCurrentObjective.length - 1]._moCrowd = Infinity;
        const minVal = objectiveDescriptors[objectiveIndex].accessor(sortedByCurrentObjective[0]);
        const maxVal = objectiveDescriptors[objectiveIndex].accessor(sortedByCurrentObjective[sortedByCurrentObjective.length - 1]);
        const valueRange = maxVal - minVal || 1;
        for (let sortedIndex = 1; sortedIndex < sortedByCurrentObjective.length - 1; sortedIndex++) {
          const prevVal = objectiveDescriptors[objectiveIndex].accessor(sortedByCurrentObjective[sortedIndex - 1]);
          const nextVal = objectiveDescriptors[objectiveIndex].accessor(sortedByCurrentObjective[sortedIndex + 1]);
          sortedByCurrentObjective[sortedIndex]._moCrowd += (nextVal - prevVal) / valueRange;
        }
      }
    }
    if (this.options.multiObjective?.enabled) {
      this._paretoArchive.push({
        generation: this.generation,
        fronts: paretoFronts.slice(0, 3).map((front) => front.map((genome) => genome._id))
      });
      if (this._paretoArchive.length > 100)
        this._paretoArchive.shift();
    }
    return paretoFronts;
  }
  var init_neat_multiobjective = __esm({
    "dist/neat/neat.multiobjective.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.adaptive.js
  var neat_adaptive_exports = {};
  __export(neat_adaptive_exports, {
    applyAdaptiveMutation: () => applyAdaptiveMutation,
    applyAncestorUniqAdaptive: () => applyAncestorUniqAdaptive,
    applyComplexityBudget: () => applyComplexityBudget,
    applyMinimalCriterionAdaptive: () => applyMinimalCriterionAdaptive,
    applyOperatorAdaptation: () => applyOperatorAdaptation,
    applyPhasedComplexity: () => applyPhasedComplexity
  });
  function applyComplexityBudget() {
    if (!this.options.complexityBudget?.enabled)
      return;
    const complexityBudget = this.options.complexityBudget;
    if (complexityBudget.mode === "adaptive") {
      if (!this._cbHistory)
        this._cbHistory = [];
      this._cbHistory.push(this.population[0]?.score || 0);
      const windowSize = complexityBudget.improvementWindow ?? 10;
      if (this._cbHistory.length > windowSize)
        this._cbHistory.shift();
      const history = this._cbHistory;
      const improvement = history.length > 1 ? history[history.length - 1] - history[0] : 0;
      let slope = 0;
      if (history.length > 2) {
        const count = history.length;
        let sumIndices = 0, sumScores = 0, sumIndexScore = 0, sumIndexSquared = 0;
        for (let idx = 0; idx < count; idx++) {
          sumIndices += idx;
          sumScores += history[idx];
          sumIndexScore += idx * history[idx];
          sumIndexSquared += idx * idx;
        }
        const denom = count * sumIndexSquared - sumIndices * sumIndices || 1;
        slope = (count * sumIndexScore - sumIndices * sumScores) / denom;
      }
      if (this._cbMaxNodes === void 0)
        this._cbMaxNodes = complexityBudget.maxNodesStart ?? this.input + this.output + 2;
      const baseInc = complexityBudget.increaseFactor ?? 1.1;
      const baseStag = complexityBudget.stagnationFactor ?? 0.95;
      const slopeMag = Math.min(2, Math.max(-2, slope / (Math.abs(history[0]) + EPSILON)));
      const incF = baseInc + 0.05 * Math.max(0, slopeMag);
      const stagF = baseStag - 0.03 * Math.max(0, -slopeMag);
      const noveltyFactor = this._noveltyArchive.length > 5 ? 1 : 0.9;
      if (improvement > 0 || slope > 0)
        this._cbMaxNodes = Math.min(complexityBudget.maxNodesEnd ?? this._cbMaxNodes * 4, Math.floor(this._cbMaxNodes * incF * noveltyFactor));
      else if (history.length === windowSize)
        this._cbMaxNodes = Math.max(complexityBudget.minNodes ?? this.input + this.output + 2, Math.floor(this._cbMaxNodes * stagF));
      if (complexityBudget.minNodes !== void 0) {
        this._cbMaxNodes = Math.max(complexityBudget.minNodes, this._cbMaxNodes);
      } else {
        const implicitMin = this.input + this.output + 2;
        if (this._cbMaxNodes < implicitMin)
          this._cbMaxNodes = implicitMin;
      }
      this.options.maxNodes = this._cbMaxNodes;
      if (complexityBudget.maxConnsStart) {
        if (this._cbMaxConns === void 0)
          this._cbMaxConns = complexityBudget.maxConnsStart;
        if (improvement > 0 || slope > 0)
          this._cbMaxConns = Math.min(complexityBudget.maxConnsEnd ?? this._cbMaxConns * 4, Math.floor(this._cbMaxConns * incF * noveltyFactor));
        else if (history.length === windowSize)
          this._cbMaxConns = Math.max(complexityBudget.maxConnsStart, Math.floor(this._cbMaxConns * stagF));
        this.options.maxConns = this._cbMaxConns;
      }
    } else {
      const maxStart = complexityBudget.maxNodesStart ?? this.input + this.output + 2;
      const maxEnd = complexityBudget.maxNodesEnd ?? maxStart * 4;
      const horizon = complexityBudget.horizon ?? 100;
      const t = Math.min(1, this.generation / horizon);
      this.options.maxNodes = Math.floor(maxStart + (maxEnd - maxStart) * t);
    }
  }
  function applyPhasedComplexity() {
    if (!this.options.phasedComplexity?.enabled)
      return;
    const len = this.options.phasedComplexity.phaseLength ?? 10;
    if (!this._phase) {
      this._phase = this.options.phasedComplexity.initialPhase ?? "complexify";
      this._phaseStartGeneration = this.generation;
    }
    if (this.generation - this._phaseStartGeneration >= len) {
      this._phase = this._phase === "complexify" ? "simplify" : "complexify";
      this._phaseStartGeneration = this.generation;
    }
  }
  function applyMinimalCriterionAdaptive() {
    if (!this.options.minimalCriterionAdaptive?.enabled)
      return;
    const mcCfg = this.options.minimalCriterionAdaptive;
    if (this._mcThreshold === void 0)
      this._mcThreshold = mcCfg.initialThreshold ?? 0;
    const scores = this.population.map((g) => g.score || 0);
    const accepted = scores.filter((s) => s >= this._mcThreshold).length;
    const prop = scores.length ? accepted / scores.length : 0;
    const targetAcceptance = mcCfg.targetAcceptance ?? 0.5;
    const adjustRate = mcCfg.adjustRate ?? 0.1;
    if (prop > targetAcceptance * 1.05)
      this._mcThreshold *= 1 + adjustRate;
    else if (prop < targetAcceptance * 0.95)
      this._mcThreshold *= 1 - adjustRate;
    for (const g of this.population)
      if ((g.score || 0) < this._mcThreshold)
        g.score = 0;
  }
  function applyAncestorUniqAdaptive() {
    if (!this.options.ancestorUniqAdaptive?.enabled)
      return;
    const ancestorCfg = this.options.ancestorUniqAdaptive;
    const cooldown = ancestorCfg.cooldown ?? 5;
    if (this.generation - this._lastAncestorUniqAdjustGen < cooldown)
      return;
    const lineageBlock = this._telemetry[this._telemetry.length - 1]?.lineage;
    const ancUniq = lineageBlock ? lineageBlock.ancestorUniq : void 0;
    if (typeof ancUniq !== "number")
      return;
    const lowT = ancestorCfg.lowThreshold ?? 0.25;
    const highT = ancestorCfg.highThreshold ?? 0.55;
    const adj = ancestorCfg.adjust ?? 0.01;
    if (ancestorCfg.mode === "epsilon" && this.options.multiObjective?.adaptiveEpsilon?.enabled) {
      if (ancUniq < lowT) {
        this.options.multiObjective.dominanceEpsilon = (this.options.multiObjective.dominanceEpsilon || 0) + adj;
        this._lastAncestorUniqAdjustGen = this.generation;
      } else if (ancUniq > highT) {
        this.options.multiObjective.dominanceEpsilon = Math.max(0, (this.options.multiObjective.dominanceEpsilon || 0) - adj);
        this._lastAncestorUniqAdjustGen = this.generation;
      }
    } else if (ancestorCfg.mode === "lineagePressure") {
      if (!this.options.lineagePressure)
        this.options.lineagePressure = {
          enabled: true,
          mode: "spread",
          strength: 0.01
        };
      const lpRef = this.options.lineagePressure;
      if (ancUniq < lowT) {
        lpRef.strength = (lpRef.strength || 0.01) * 1.15;
        lpRef.mode = "spread";
        this._lastAncestorUniqAdjustGen = this.generation;
      } else if (ancUniq > highT) {
        lpRef.strength = (lpRef.strength || 0.01) * 0.9;
        this._lastAncestorUniqAdjustGen = this.generation;
      }
    }
  }
  function applyAdaptiveMutation() {
    if (!this.options.adaptiveMutation?.enabled)
      return;
    const adaptCfg = this.options.adaptiveMutation;
    const every = adaptCfg.adaptEvery ?? 1;
    if (!(every <= 1 || this.generation % every === 0))
      return;
    const scored = this.population.filter((g) => typeof g.score === "number");
    scored.sort((a, b) => (a.score || 0) - (b.score || 0));
    const mid = Math.floor(scored.length / 2);
    const topHalf = scored.slice(mid);
    const bottomHalf = scored.slice(0, mid);
    const sigmaBase = (adaptCfg.sigma ?? 0.05) * 1.5;
    const minR = adaptCfg.minRate ?? 0.01;
    const maxR = adaptCfg.maxRate ?? 1;
    const strategy = adaptCfg.strategy || "twoTier";
    let anyUp = false, anyDown = false;
    for (let index = 0; index < this.population.length; index++) {
      const genome = this.population[index];
      if (genome._mutRate === void 0)
        continue;
      let rate = genome._mutRate;
      let delta = this._getRNG()() * 2 - 1;
      delta *= sigmaBase;
      if (strategy === "twoTier") {
        if (topHalf.length === 0 || bottomHalf.length === 0)
          delta = index % 2 === 0 ? Math.abs(delta) : -Math.abs(delta);
        else if (topHalf.includes(genome))
          delta = -Math.abs(delta);
        else if (bottomHalf.includes(genome))
          delta = Math.abs(delta);
      } else if (strategy === "exploreLow") {
        delta = bottomHalf.includes(genome) ? Math.abs(delta * 1.5) : -Math.abs(delta * 0.5);
      } else if (strategy === "anneal") {
        const progress = Math.min(1, this.generation / (50 + this.population.length));
        delta *= 1 - progress;
      }
      rate += delta;
      if (rate < minR)
        rate = minR;
      if (rate > maxR)
        rate = maxR;
      if (rate > (this.options.adaptiveMutation.initialRate ?? 0.5))
        anyUp = true;
      if (rate < (this.options.adaptiveMutation.initialRate ?? 0.5))
        anyDown = true;
      genome._mutRate = rate;
      if (adaptCfg.adaptAmount) {
        const aSigma = adaptCfg.amountSigma ?? 0.25;
        let aDelta = (this._getRNG()() * 2 - 1) * aSigma;
        if (strategy === "twoTier") {
          if (topHalf.length === 0 || bottomHalf.length === 0)
            aDelta = index % 2 === 0 ? Math.abs(aDelta) : -Math.abs(aDelta);
          else
            aDelta = bottomHalf.includes(genome) ? Math.abs(aDelta) : -Math.abs(aDelta);
        }
        let amt = genome._mutAmount ?? (this.options.mutationAmount || 1);
        amt += aDelta;
        amt = Math.round(amt);
        const minA = adaptCfg.minAmount ?? 1;
        const maxA = adaptCfg.maxAmount ?? 10;
        if (amt < minA)
          amt = minA;
        if (amt > maxA)
          amt = maxA;
        genome._mutAmount = amt;
      }
    }
    if (strategy === "twoTier" && !(anyUp && anyDown)) {
      const baseline = this.options.adaptiveMutation.initialRate ?? 0.5;
      const half = Math.floor(this.population.length / 2);
      for (let i = 0; i < this.population.length; i++) {
        const genome = this.population[i];
        if (genome._mutRate === void 0)
          continue;
        if (i < half)
          genome._mutRate = Math.min(genome._mutRate + sigmaBase, 1);
        else
          genome._mutRate = Math.max(genome._mutRate - sigmaBase, 0.01);
      }
    }
  }
  function applyOperatorAdaptation() {
    if (!this.options.operatorAdaptation?.enabled)
      return;
    const decay = this.options.operatorAdaptation.decay ?? 0.9;
    for (const [k, stat] of this._operatorStats.entries()) {
      stat.success *= decay;
      stat.attempts *= decay;
      this._operatorStats.set(k, stat);
    }
  }
  var init_neat_adaptive = __esm({
    "dist/neat/neat.adaptive.js"() {
      "use strict";
      init_neat_constants();
    }
  });

  // dist/neat/neat.lineage.js
  function buildAnc(genome) {
    const ancestorSet = /* @__PURE__ */ new Set();
    if (!Array.isArray(genome._parents))
      return ancestorSet;
    const queue = [];
    for (const parentId of genome._parents) {
      queue.push({
        id: parentId,
        depth: 1,
        genomeRef: this.population.find((gm) => gm._id === parentId)
      });
    }
    while (queue.length) {
      const current = queue.shift();
      if (current.depth > ANCESTOR_DEPTH_WINDOW)
        continue;
      if (current.id != null)
        ancestorSet.add(current.id);
      if (current.genomeRef && Array.isArray(current.genomeRef._parents)) {
        for (const parentId of current.genomeRef._parents) {
          queue.push({
            id: parentId,
            depth: current.depth + 1,
            genomeRef: this.population.find((gm) => gm._id === parentId)
          });
        }
      }
    }
    return ancestorSet;
  }
  function computeAncestorUniqueness() {
    const buildAncestorSet = buildAnc.bind(this);
    let sampledPairCount = 0;
    let jaccardDistanceSum = 0;
    const maxSamplePairs = Math.min(MAX_UNIQUENESS_SAMPLE_PAIRS, this.population.length * (this.population.length - 1) / 2);
    for (let t = 0; t < maxSamplePairs; t++) {
      if (this.population.length < 2)
        break;
      const indexA = Math.floor(this._getRNG()() * this.population.length);
      let indexB = Math.floor(this._getRNG()() * this.population.length);
      if (indexB === indexA)
        indexB = (indexB + 1) % this.population.length;
      const ancestorSetA = buildAncestorSet(this.population[indexA]);
      const ancestorSetB = buildAncestorSet(this.population[indexB]);
      if (ancestorSetA.size === 0 && ancestorSetB.size === 0)
        continue;
      let intersectionCount = 0;
      for (const id of ancestorSetA)
        if (ancestorSetB.has(id))
          intersectionCount++;
      const unionSize = ancestorSetA.size + ancestorSetB.size - intersectionCount || 1;
      const jaccardDistance = 1 - intersectionCount / unionSize;
      jaccardDistanceSum += jaccardDistance;
      sampledPairCount++;
    }
    const ancestorUniqueness = sampledPairCount ? +(jaccardDistanceSum / sampledPairCount).toFixed(3) : 0;
    return ancestorUniqueness;
  }
  var ANCESTOR_DEPTH_WINDOW, MAX_UNIQUENESS_SAMPLE_PAIRS;
  var init_neat_lineage = __esm({
    "dist/neat/neat.lineage.js"() {
      "use strict";
      ANCESTOR_DEPTH_WINDOW = 4;
      MAX_UNIQUENESS_SAMPLE_PAIRS = 30;
    }
  });

  // dist/neat/neat.telemetry.js
  var neat_telemetry_exports = {};
  __export(neat_telemetry_exports, {
    applyTelemetrySelect: () => applyTelemetrySelect,
    buildTelemetryEntry: () => buildTelemetryEntry,
    computeDiversityStats: () => computeDiversityStats,
    recordTelemetryEntry: () => recordTelemetryEntry,
    structuralEntropy: () => structuralEntropy
  });
  function applyTelemetrySelect(entry) {
    const ctx = this;
    const selectionSet = ctx._telemetrySelect;
    if (!selectionSet || selectionSet.size === 0)
      return entry;
    const coreFields = ["gen", "best", "species"];
    const core = {};
    for (const field of coreFields) {
      if (Object.hasOwn(entry, field)) {
        core[field] = entry[field];
      }
    }
    for (const key of Object.keys(entry)) {
      if (coreFields.includes(key))
        continue;
      if (!selectionSet.has(key)) {
        delete entry[key];
      }
    }
    return Object.assign(entry, core);
  }
  function structuralEntropy(graph) {
    const ctx = this;
    if (graph._entropyGen === ctx.generation && typeof graph._entropyVal === "number") {
      return graph._entropyVal;
    }
    const degreeCounts = {};
    for (const node of graph.nodes)
      degreeCounts[node.geneId] = 0;
    for (const connection of graph.connections) {
      if (connection.enabled) {
        const fromId = connection.from.geneId;
        const toId = connection.to.geneId;
        if (degreeCounts[fromId] !== void 0)
          degreeCounts[fromId]++;
        if (degreeCounts[toId] !== void 0)
          degreeCounts[toId]++;
      }
    }
    const degreeHistogram = {};
    const nodeCount = graph.nodes.length || 1;
    for (const nodeId in degreeCounts) {
      const degree = degreeCounts[Number(nodeId)];
      degreeHistogram[degree] = (degreeHistogram[degree] || 0) + 1;
    }
    let entropy = 0;
    for (const degree in degreeHistogram) {
      const probability = degreeHistogram[degree] / nodeCount;
      if (probability > 0)
        entropy -= probability * Math.log(probability + EPSILON);
    }
    graph._entropyGen = ctx.generation;
    graph._entropyVal = entropy;
    return entropy;
  }
  function computeDiversityStats() {
    const ctx = this;
    const options = ctx.options;
    if (!options.diversityMetrics?.enabled)
      return;
    if (options.fastMode && !ctx._fastModeTuned) {
      const diversityMetrics = options.diversityMetrics;
      if (diversityMetrics) {
        if (diversityMetrics.pairSample == null)
          diversityMetrics.pairSample = 20;
        if (diversityMetrics.graphletSample == null)
          diversityMetrics.graphletSample = 30;
      }
      if (options.novelty?.enabled && options.novelty.k == null)
        options.novelty.k = 5;
      ctx._fastModeTuned = true;
    }
    const pairSample = options.diversityMetrics.pairSample ?? 40;
    const graphletSample = options.diversityMetrics.graphletSample ?? 60;
    const population = ctx.population ?? [];
    const popSize = population.length;
    let compatibilitySum = 0;
    let compatibilitySumSq = 0;
    let compatibilityCount = 0;
    const rngFactory = typeof ctx._getRNG === "function" ? ctx._getRNG : () => Math.random;
    for (let iter = 0; iter < pairSample; iter++) {
      if (popSize < 2)
        break;
      const rng = rngFactory();
      const firstIndex = Math.floor(rng() * popSize);
      let secondIndex = Math.floor(rng() * popSize);
      if (secondIndex === firstIndex)
        secondIndex = (secondIndex + 1) % popSize;
      const distance = ctx._compatibilityDistance?.(population[firstIndex], population[secondIndex]) ?? 0;
      compatibilitySum += distance;
      compatibilitySumSq += distance * distance;
      compatibilityCount++;
    }
    const meanCompat = compatibilityCount ? compatibilitySum / compatibilityCount : 0;
    const varCompat = compatibilityCount ? Math.max(0, compatibilitySumSq / compatibilityCount - meanCompat * meanCompat) : 0;
    const entropies = population.map((genome) => ctx._structuralEntropy ? ctx._structuralEntropy(genome) : structuralEntropy.call(ctx, genome));
    const meanEntropy = entropies.reduce((a, b) => a + b, 0) / (entropies.length || 1);
    const varEntropy = entropies.length ? entropies.reduce((a, b) => a + (b - meanEntropy) * (b - meanEntropy), 0) / entropies.length : 0;
    const motifCounts = [0, 0, 0, 0];
    for (let iter = 0; iter < graphletSample; iter++) {
      if (popSize === 0)
        break;
      const rng = rngFactory();
      const genome = population[Math.floor(rng() * popSize)];
      if (!genome)
        break;
      if (genome.nodes.length < 3)
        continue;
      const selectedIndices = /* @__PURE__ */ new Set();
      while (selectedIndices.size < 3)
        selectedIndices.add(Math.floor(rng() * genome.nodes.length));
      const selectedNodes = Array.from(selectedIndices).map((i) => genome.nodes[i]);
      let edgeCount = 0;
      for (const connection of genome.connections) {
        if (connection.enabled && selectedNodes.includes(connection.from) && selectedNodes.includes(connection.to))
          edgeCount++;
      }
      if (edgeCount > 3)
        edgeCount = 3;
      motifCounts[edgeCount]++;
    }
    const totalMotifs = motifCounts.reduce((a, b) => a + b, 0) || 1;
    let graphletEntropy = 0;
    for (let k = 0; k < motifCounts.length; k++) {
      const probability = motifCounts[k] / totalMotifs;
      if (probability > 0)
        graphletEntropy -= probability * Math.log(probability);
    }
    let lineageMeanDepth = 0;
    let lineageMeanPairDist = 0;
    if (ctx._lineageEnabled && popSize > 0) {
      const depths = population.map((genome) => genome._depth ?? 0);
      lineageMeanDepth = depths.reduce((a, b) => a + b, 0) / popSize;
      let lineagePairSum = 0;
      let lineagePairN = 0;
      const pairsToSample = Math.min(pairSample, popSize * (popSize - 1) / 2);
      for (let iter = 0; iter < pairsToSample; iter++) {
        if (popSize < 2)
          break;
        const rng = rngFactory();
        const i = Math.floor(rng() * popSize);
        let j = Math.floor(rng() * popSize);
        if (j === i)
          j = (j + 1) % popSize;
        lineagePairSum += Math.abs(depths[i] - depths[j]);
        lineagePairN++;
      }
      lineageMeanPairDist = lineagePairN ? lineagePairSum / lineagePairN : 0;
    }
    ctx._diversityStats = {
      meanCompat,
      varCompat,
      meanEntropy,
      varEntropy,
      graphletEntropy,
      lineageMeanDepth,
      lineageMeanPairDist
    };
  }
  function recordTelemetryEntry(entry) {
    const ctx = this;
    try {
      applyTelemetrySelect.call(ctx, entry);
    } catch {
    }
    if (!ctx._telemetry)
      ctx._telemetry = [];
    ctx._telemetry.push(entry);
    try {
      const telemetryStream = ctx.options?.telemetryStream;
      if (telemetryStream?.enabled && typeof telemetryStream.onEntry === "function") {
        telemetryStream.onEntry(entry);
      }
    } catch {
    }
    if (ctx._telemetry.length > 500)
      ctx._telemetry.shift();
  }
  function buildTelemetryEntry(fittest) {
    const ctx = this;
    const gen = ctx.generation ?? 0;
    let hyperVolumeProxy = 0;
    const options = ctx.options || {};
    if (options.multiObjective?.enabled) {
      const complexityMetric = options.multiObjective?.complexityMetric || "connections";
      const population = ctx.population || [];
      const primaryObjectiveScores = population.map((genome) => genome.score || 0);
      const minPrimaryScore = Math.min(...primaryObjectiveScores);
      const maxPrimaryScore = Math.max(...primaryObjectiveScores);
      const paretoFrontSizes = [];
      for (let r = 0; r < 5; r++) {
        const size = population.filter((g) => (g._moRank ?? 0) === r).length;
        if (!size)
          break;
        paretoFrontSizes.push(size);
      }
      for (const genome of population) {
        const rank = genome._moRank ?? 0;
        if (rank !== 0)
          continue;
        const normalizedScore = maxPrimaryScore > minPrimaryScore ? ((genome.score || 0) - minPrimaryScore) / (maxPrimaryScore - minPrimaryScore) : 0;
        const genomeComplexity = complexityMetric === "nodes" ? genome.nodes.length : genome.connections.length;
        hyperVolumeProxy += normalizedScore * (1 / (genomeComplexity + 1));
      }
      const operatorStatsSnapshot = Array.from((ctx._operatorStats ?? /* @__PURE__ */ new Map()).entries()).map(([opName, stats]) => ({
        op: opName,
        succ: stats.success,
        att: stats.attempts
      }));
      const entry2 = {
        gen,
        best: fittest.score ?? 0,
        species: ctx._species?.length ?? 0,
        hyper: hyperVolumeProxy,
        fronts: paretoFrontSizes,
        diversity: ctx._diversityStats,
        ops: operatorStatsSnapshot,
        objImportance: {}
      };
      if (!entry2.objImportance)
        entry2.objImportance = {};
      if (ctx._lastObjImportance) {
        const lastImportance = ctx._lastObjImportance;
        if (typeof lastImportance === "object" && lastImportance !== null)
          entry2.objImportance = lastImportance;
      }
      if (ctx._objectiveAges?.size) {
        entry2.objAges = Object.fromEntries(ctx._objectiveAges.entries());
      }
      if (ctx._pendingObjectiveAdds?.length || ctx._pendingObjectiveRemoves?.length) {
        entry2.objEvents = [];
        for (const k of ctx._pendingObjectiveAdds || [])
          entry2.objEvents.push({ type: "add", key: k });
        for (const k of ctx._pendingObjectiveRemoves || [])
          entry2.objEvents.push({ type: "remove", key: k });
        ctx._objectiveEvents = ctx._objectiveEvents || [];
        ctx._objectiveEvents.push(...entry2.objEvents.map((e) => ({
          gen,
          type: e.type,
          key: e.key
        })));
        ctx._pendingObjectiveAdds = [];
        ctx._pendingObjectiveRemoves = [];
      }
      if (ctx._lastOffspringAlloc) {
        const lastAlloc = ctx._lastOffspringAlloc;
        if (Array.isArray(lastAlloc))
          entry2.speciesAlloc = lastAlloc.slice();
      }
      try {
        entry2.objectives = ctx._getObjectives?.().map((o) => o.key) || [];
      } catch {
      }
      if (options.rngState && ctx._rngState !== void 0)
        entry2.rng = ctx._rngState;
      if (ctx._lineageEnabled) {
        const bestGenome = ctx.population[0];
        const depths = ctx.population.map((g) => g._depth ?? 0);
        ctx._lastMeanDepth = depths.reduce((a, b) => a + b, 0) / (depths.length || 1);
        const lineageCtx = {
          population: ctx.population || [],
          _getRNG: typeof ctx._getRNG === "function" ? ctx._getRNG : () => Math.random
        };
        const ancestorUniqueness = computeAncestorUniqueness.call(lineageCtx);
        entry2.lineage = {
          parents: Array.isArray(bestGenome._parents) ? bestGenome._parents.slice() : [],
          depthBest: bestGenome._depth ?? 0,
          meanDepth: +(ctx._lastMeanDepth ?? 0).toFixed(2),
          inbreeding: ctx._prevInbreedingCount ?? 0,
          ancestorUniq: ancestorUniqueness
        };
      }
      if (options.telemetry?.hypervolume && options.multiObjective?.enabled)
        entry2.hv = +hyperVolumeProxy.toFixed(4);
      if (options.telemetry?.complexity) {
        const nodesArr = population.map((g) => g.nodes.length);
        const connsArr = population.map((g) => g.connections.length);
        const meanNodes = nodesArr.reduce((a, b) => a + b, 0) / (nodesArr.length || 1);
        const meanConns = connsArr.reduce((a, b) => a + b, 0) / (connsArr.length || 1);
        const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
        const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
        const enabledRatios = population.map((g) => {
          let enabled = 0, disabled = 0;
          for (const c of g.connections) {
            if (c.enabled === false)
              disabled++;
            else
              enabled++;
          }
          return enabled + disabled ? enabled / (enabled + disabled) : 0;
        });
        const meanEnabledRatio = enabledRatios.reduce((a, b) => a + b, 0) / (enabledRatios.length || 1);
        const growthNodes = this._lastMeanNodes !== void 0 ? meanNodes - this._lastMeanNodes : 0;
        const growthConns = this._lastMeanConns !== void 0 ? meanConns - this._lastMeanConns : 0;
        this._lastMeanNodes = meanNodes;
        this._lastMeanConns = meanConns;
        entry2.complexity = {
          meanNodes: +meanNodes.toFixed(2),
          meanConns: +meanConns.toFixed(2),
          maxNodes,
          maxConns,
          meanEnabledRatio: +meanEnabledRatio.toFixed(3),
          growthNodes: +growthNodes.toFixed(2),
          growthConns: +growthConns.toFixed(2),
          budgetMaxNodes: options.maxNodes ?? 0,
          budgetMaxConns: options.maxConns ?? 0
        };
      }
      if (options.telemetry?.performance)
        entry2.perf = {
          evalMs: this._lastEvalDuration,
          evolveMs: this._lastEvolveDuration
        };
      return entry2;
    }
    const operatorStatsSnapshotMono = Array.from((ctx._operatorStats ?? /* @__PURE__ */ new Map()).entries()).map(([opName, stats]) => ({
      op: opName,
      succ: stats.success,
      att: stats.attempts
    }));
    const entry = {
      gen,
      best: fittest.score ?? 0,
      species: ctx._species?.length ?? 0,
      hyper: hyperVolumeProxy,
      diversity: ctx._diversityStats,
      ops: operatorStatsSnapshotMono,
      objImportance: {}
    };
    if (ctx._lastObjImportance)
      entry.objImportance = ctx._lastObjImportance;
    if (ctx._objectiveAges?.size)
      entry.objAges = Object.fromEntries(ctx._objectiveAges.entries());
    if (ctx._pendingObjectiveAdds?.length || ctx._pendingObjectiveRemoves?.length) {
      entry.objEvents = [];
      for (const k of ctx._pendingObjectiveAdds || [])
        entry.objEvents.push({ type: "add", key: k });
      for (const k of ctx._pendingObjectiveRemoves || [])
        entry.objEvents.push({ type: "remove", key: k });
      ctx._objectiveEvents = ctx._objectiveEvents || [];
      ctx._objectiveEvents.push(...entry.objEvents.map((e) => ({
        gen,
        type: e.type,
        key: e.key
      })));
      ctx._pendingObjectiveAdds = [];
      ctx._pendingObjectiveRemoves = [];
    }
    if (ctx._lastOffspringAlloc)
      entry.speciesAlloc = ctx._lastOffspringAlloc?.slice();
    try {
      entry.objectives = ctx._getObjectives?.().map((o) => o.key) || [];
    } catch {
    }
    if (ctx.options?.rngState && ctx._rngState !== void 0)
      entry.rng = ctx._rngState;
    if (ctx._lineageEnabled) {
      const bestGenome = ctx.population[0];
      const depths = ctx.population.map((g) => g._depth ?? 0);
      ctx._lastMeanDepth = depths.reduce((a, b) => a + b, 0) / (depths.length || 1);
      let sampledPairs = 0;
      let jaccardSum = 0;
      const popLength = ctx.population.length;
      const samplePairs = Math.min(30, popLength * (popLength - 1) / 2);
      for (let t = 0; t < samplePairs; t++) {
        if (popLength < 2)
          break;
        const rngFn = typeof ctx._getRNG === "function" ? ctx._getRNG() : Math.random;
        const i = Math.floor(rngFn() * popLength);
        let j = Math.floor(rngFn() * popLength);
        if (j === i)
          j = (j + 1) % popLength;
        const lineageCtx2 = {
          population: ctx.population || [],
          _getRNG: typeof ctx._getRNG === "function" ? ctx._getRNG : () => Math.random
        };
        const ancestorsA = buildAnc.call(lineageCtx2, ctx.population[i]);
        const ancestorsB = buildAnc.call(lineageCtx2, ctx.population[j]);
        if (ancestorsA.size === 0 && ancestorsB.size === 0)
          continue;
        let intersectionCount = 0;
        for (const id of ancestorsA)
          if (ancestorsB.has(id))
            intersectionCount++;
        const union = ancestorsA.size + ancestorsB.size - intersectionCount || 1;
        const jaccardDistance = 1 - intersectionCount / union;
        jaccardSum += jaccardDistance;
        sampledPairs++;
      }
      const ancestorUniqueness = sampledPairs ? +(jaccardSum / sampledPairs).toFixed(3) : 0;
      entry.lineage = {
        parents: Array.isArray(bestGenome._parents) ? bestGenome._parents.slice() : [],
        depthBest: bestGenome._depth ?? 0,
        meanDepth: +(ctx._lastMeanDepth ?? 0).toFixed(2),
        inbreeding: ctx._prevInbreedingCount ?? 0,
        ancestorUniq: ancestorUniqueness
      };
    }
    if (this.options.telemetry?.hypervolume && this.options.multiObjective?.enabled)
      entry.hv = +hyperVolumeProxy.toFixed(4);
    if (this.options.telemetry?.complexity) {
      const nodesArr = this.population.map((g) => g.nodes.length);
      const connsArr = this.population.map((g) => g.connections.length);
      const meanNodes = nodesArr.reduce((a, b) => a + b, 0) / (nodesArr.length || 1);
      const meanConns = connsArr.reduce((a, b) => a + b, 0) / (connsArr.length || 1);
      const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
      const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
      const enabledRatios = this.population.map((g) => {
        let en = 0, dis = 0;
        for (const c of g.connections) {
          if (c.enabled === false)
            dis++;
          else
            en++;
        }
        return en + dis ? en / (en + dis) : 0;
      });
      const meanEnabledRatio = enabledRatios.reduce((a, b) => a + b, 0) / (enabledRatios.length || 1);
      const growthNodes = this._lastMeanNodes !== void 0 ? meanNodes - this._lastMeanNodes : 0;
      const growthConns = this._lastMeanConns !== void 0 ? meanConns - this._lastMeanConns : 0;
      this._lastMeanNodes = meanNodes;
      this._lastMeanConns = meanConns;
      entry.complexity = {
        meanNodes: +meanNodes.toFixed(2),
        meanConns: +meanConns.toFixed(2),
        maxNodes,
        maxConns,
        meanEnabledRatio: +meanEnabledRatio.toFixed(3),
        growthNodes: +growthNodes.toFixed(2),
        growthConns: +growthConns.toFixed(2),
        budgetMaxNodes: this.options.maxNodes ?? 0,
        budgetMaxConns: this.options.maxConns ?? 0
      };
    }
    if (this.options.telemetry?.performance)
      entry.perf = {
        evalMs: this._lastEvalDuration,
        evolveMs: this._lastEvolveDuration
      };
    return entry;
  }
  var init_neat_telemetry = __esm({
    "dist/neat/neat.telemetry.js"() {
      "use strict";
      init_neat_constants();
      init_neat_lineage();
    }
  });

  // dist/neat/neat.pruning.js
  var neat_pruning_exports = {};
  __export(neat_pruning_exports, {
    applyAdaptivePruning: () => applyAdaptivePruning,
    applyEvolutionPruning: () => applyEvolutionPruning
  });
  function applyEvolutionPruning() {
    const evolutionPruningOpts = this.options.evolutionPruning;
    if (!evolutionPruningOpts || this.generation < (evolutionPruningOpts.startGeneration || 0))
      return;
    const interval = evolutionPruningOpts.interval || 1;
    if ((this.generation - evolutionPruningOpts.startGeneration) % interval !== 0)
      return;
    const rampGenerations = evolutionPruningOpts.rampGenerations || 0;
    let rampFraction = 1;
    if (rampGenerations > 0) {
      const progressThroughRamp = Math.min(1, Math.max(0, (this.generation - evolutionPruningOpts.startGeneration) / rampGenerations));
      rampFraction = progressThroughRamp;
    }
    const targetSparsityNow = (evolutionPruningOpts.targetSparsity || 0) * rampFraction;
    for (const genome of this.population) {
      if (genome && typeof genome.pruneToSparsity === "function") {
        genome.pruneToSparsity(targetSparsityNow, evolutionPruningOpts.method || "magnitude");
      }
    }
  }
  function applyAdaptivePruning() {
    if (!this.options.adaptivePruning?.enabled)
      return;
    const adaptivePruningOpts = this.options.adaptivePruning;
    if (this._adaptivePruneLevel === void 0)
      this._adaptivePruneLevel = 0;
    const metricName = adaptivePruningOpts.metric || "connections";
    const meanNodeCount = this.population.reduce((acc, g) => acc + g.nodes.length, 0) / (this.population.length || 1);
    const meanConnectionCount = this.population.reduce((acc, g) => acc + g.connections.length, 0) / (this.population.length || 1);
    const currentMetricValue = metricName === "nodes" ? meanNodeCount : meanConnectionCount;
    if (this._adaptivePruneBaseline === void 0)
      this._adaptivePruneBaseline = currentMetricValue;
    const adaptivePruneBaseline = this._adaptivePruneBaseline;
    const desiredSparsity = adaptivePruningOpts.targetSparsity ?? 0.5;
    const targetRemainingMetric = adaptivePruneBaseline * (1 - desiredSparsity);
    const tolerance = adaptivePruningOpts.tolerance ?? 0.05;
    const adjustRate = adaptivePruningOpts.adjustRate ?? 0.02;
    const normalizedDifference = (currentMetricValue - targetRemainingMetric) / (adaptivePruneBaseline || 1);
    if (Math.abs(normalizedDifference) > tolerance) {
      this._adaptivePruneLevel = Math.max(0, Math.min(desiredSparsity, this._adaptivePruneLevel + adjustRate * (normalizedDifference > 0 ? 1 : -1)));
      for (const g of this.population)
        if (typeof g.pruneToSparsity === "function")
          g.pruneToSparsity(this._adaptivePruneLevel, "magnitude");
    }
  }
  var init_neat_pruning = __esm({
    "dist/neat/neat.pruning.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.evolve.js
  async function evolve() {
    const startTime = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
    if (this.population[this.population.length - 1].score === void 0) {
      await this.evaluate();
    }
    this._objectivesList = void 0;
    try {
      (init_neat_adaptive(), __toCommonJS(neat_adaptive_exports)).applyComplexityBudget.call(this);
    } catch {
    }
    try {
      (init_neat_adaptive(), __toCommonJS(neat_adaptive_exports)).applyPhasedComplexity.call(this);
    } catch {
    }
    this.sort();
    try {
      const currentBest = this.population[0]?.score;
      if (typeof currentBest === "number" && (this._bestScoreLastGen === void 0 || currentBest > this._bestScoreLastGen)) {
        this._bestScoreLastGen = currentBest;
        this._lastGlobalImproveGeneration = this.generation;
      }
    } catch {
    }
    try {
      (init_neat_adaptive(), __toCommonJS(neat_adaptive_exports)).applyMinimalCriterionAdaptive.call(this);
    } catch {
    }
    try {
      this._computeDiversityStats && this._computeDiversityStats();
    } catch {
    }
    if (this.options.multiObjective?.enabled) {
      const populationSnapshot = this.population;
      const paretoFronts = fastNonDominated.call(this, populationSnapshot);
      const objectives = this._getObjectives();
      const crowdingDistances = new Array(populationSnapshot.length).fill(0);
      const objectiveValues = objectives.map((obj) => populationSnapshot.map((genome) => obj.accessor(genome)));
      for (const front of paretoFronts) {
        const frontIndices = front.map((genome) => this.population.indexOf(genome));
        if (frontIndices.length < 3) {
          frontIndices.forEach((i) => crowdingDistances[i] = Infinity);
          continue;
        }
        for (let oi = 0; oi < objectives.length; oi++) {
          const sortedIdx = [...frontIndices].sort((a, b) => objectiveValues[oi][a] - objectiveValues[oi][b]);
          crowdingDistances[sortedIdx[0]] = Infinity;
          crowdingDistances[sortedIdx[sortedIdx.length - 1]] = Infinity;
          const minV = objectiveValues[oi][sortedIdx[0]];
          const maxV = objectiveValues[oi][sortedIdx[sortedIdx.length - 1]];
          for (let k = 1; k < sortedIdx.length - 1; k++) {
            const prev = objectiveValues[oi][sortedIdx[k - 1]];
            const next = objectiveValues[oi][sortedIdx[k + 1]];
            const denom = maxV - minV || 1;
            crowdingDistances[sortedIdx[k]] += (next - prev) / denom;
          }
        }
      }
      const indexMap = /* @__PURE__ */ new Map();
      for (let i = 0; i < populationSnapshot.length; i++)
        indexMap.set(populationSnapshot[i], i);
      this.population.sort((a, b) => {
        const ra = a._moRank ?? 0;
        const rb = b._moRank ?? 0;
        if (ra !== rb)
          return ra - rb;
        const ia = indexMap.get(a);
        const ib = indexMap.get(b);
        return crowdingDistances[ib] - crowdingDistances[ia];
      });
      for (let i = 0; i < populationSnapshot.length; i++)
        populationSnapshot[i]._moCrowd = crowdingDistances[i];
      if (paretoFronts.length) {
        const first = paretoFronts[0];
        const snapshot = first.map((genome) => ({
          id: genome._id ?? -1,
          score: genome.score || 0,
          nodes: genome.nodes.length,
          connections: genome.connections.length
        }));
        this._paretoArchive.push({
          gen: this.generation,
          size: first.length,
          genomes: snapshot
        });
        if (this._paretoArchive.length > 200)
          this._paretoArchive.shift();
        if (objectives.length) {
          const vectors = first.map((genome) => ({
            id: genome._id ?? -1,
            values: objectives.map((obj) => obj.accessor(genome))
          }));
          this._paretoObjectivesArchive.push({ gen: this.generation, vectors });
          if (this._paretoObjectivesArchive.length > 200)
            this._paretoObjectivesArchive.shift();
        }
      }
      if (this.options.multiObjective?.adaptiveEpsilon?.enabled && paretoFronts.length) {
        const cfg = this.options.multiObjective.adaptiveEpsilon;
        const target = cfg.targetFront ?? Math.max(3, Math.floor(Math.sqrt(this.population.length)));
        const adjust = cfg.adjust ?? 2e-3;
        const minE = cfg.min ?? 0;
        const maxE = cfg.max ?? 0.5;
        const cooldown = cfg.cooldown ?? 2;
        if (this.generation - this._lastEpsilonAdjustGen >= cooldown) {
          const currentSize = paretoFronts[0].length;
          let eps = this.options.multiObjective.dominanceEpsilon || 0;
          if (currentSize > target * 1.2)
            eps = Math.min(maxE, eps + adjust);
          else if (currentSize < target * 0.8)
            eps = Math.max(minE, eps - adjust);
          this.options.multiObjective.dominanceEpsilon = eps;
          this._lastEpsilonAdjustGen = this.generation;
        }
      }
      if (this.options.multiObjective?.pruneInactive?.enabled) {
        const cfg = this.options.multiObjective.pruneInactive;
        const window2 = cfg.window ?? 5;
        const rangeEps = cfg.rangeEps ?? 1e-6;
        const protect = /* @__PURE__ */ new Set([
          "fitness",
          "complexity",
          ...cfg.protect || []
        ]);
        const objsList = this._getObjectives();
        const ranges = {};
        for (const obj of objsList) {
          let min = Infinity, max = -Infinity;
          for (const genome of this.population) {
            const v = obj.accessor(genome);
            if (v < min)
              min = v;
            if (v > max)
              max = v;
          }
          ranges[obj.key] = { min, max };
        }
        const toRemove = [];
        for (const obj of objsList) {
          if (protect.has(obj.key))
            continue;
          const objRange = ranges[obj.key];
          const span = objRange.max - objRange.min;
          if (span < rangeEps) {
            const count = (this._objectiveStale.get(obj.key) || 0) + 1;
            this._objectiveStale.set(obj.key, count);
            if (count >= window2)
              toRemove.push(obj.key);
          } else {
            this._objectiveStale.set(obj.key, 0);
          }
        }
        if (toRemove.length && this.options.multiObjective?.objectives) {
          this.options.multiObjective.objectives = this.options.multiObjective.objectives.filter((obj) => !toRemove.includes(obj.key));
          this._objectivesList = void 0;
        }
      }
    }
    try {
      (init_neat_adaptive(), __toCommonJS(neat_adaptive_exports)).applyAncestorUniqAdaptive.call(this);
    } catch {
    }
    if (this.options.speciation) {
      try {
        this._speciate();
      } catch {
      }
      try {
        this._applyFitnessSharing();
      } catch {
      }
      try {
        const opts = this.options;
        if (opts.autoCompatTuning?.enabled) {
          const tgt = opts.autoCompatTuning.target ?? opts.targetSpecies ?? Math.max(2, Math.round(Math.sqrt(this.population.length)));
          const obs = this._species.length || 1;
          const err = tgt - obs;
          const rate = opts.autoCompatTuning.adjustRate ?? 0.01;
          const minC = opts.autoCompatTuning.minCoeff ?? 0.1;
          const maxC = opts.autoCompatTuning.maxCoeff ?? 5;
          let factor = 1 - rate * Math.sign(err);
          if (err === 0)
            factor = 1 + (this._getRNG()() - 0.5) * rate * 0.5;
          opts.excessCoeff = Math.min(maxC, Math.max(minC, opts.excessCoeff * factor));
          opts.disjointCoeff = Math.min(maxC, Math.max(minC, opts.disjointCoeff * factor));
        }
      } catch {
      }
      this.sort();
      try {
        if (this.options.speciesAllocation?.extendedHistory) {
        } else {
          if (!this._speciesHistory || this._speciesHistory.length === 0 || this._speciesHistory[this._speciesHistory.length - 1].generation !== this.generation) {
            this._speciesHistory.push({
              generation: this.generation,
              stats: this._species.map((species) => ({
                id: species.id,
                size: species.members.length,
                best: species.bestScore,
                lastImproved: species.lastImproved
              }))
            });
            if (this._speciesHistory.length > 200)
              this._speciesHistory.shift();
          }
        }
      } catch {
      }
    }
    const fittest = Network.fromJSON(this.population[0].toJSON());
    fittest.score = this.population[0].score;
    this._computeDiversityStats();
    try {
      const currentObjKeys = this._getObjectives().map((obj) => obj.key);
      const dyn = this.options.multiObjective?.dynamic;
      if (this.options.multiObjective?.enabled) {
        if (dyn?.enabled) {
          const addC = dyn.addComplexityAt ?? Infinity;
          const addE = dyn.addEntropyAt ?? Infinity;
          if (this.generation + 1 >= addC && !currentObjKeys.includes("complexity")) {
            this.registerObjective("complexity", "min", (genome) => genome.connections.length);
            this._pendingObjectiveAdds.push("complexity");
          }
          if (this.generation + 1 >= addE && !currentObjKeys.includes("entropy")) {
            this.registerObjective("entropy", "max", (genome) => this._structuralEntropy(genome));
            this._pendingObjectiveAdds.push("entropy");
          }
          if (currentObjKeys.includes("entropy") && dyn.dropEntropyOnStagnation != null) {
            const stagnGen = dyn.dropEntropyOnStagnation;
            if (this.generation >= stagnGen && !this._entropyDropped) {
              if (this.options.multiObjective?.objectives) {
                this.options.multiObjective.objectives = this.options.multiObjective.objectives.filter((obj) => obj.key !== "entropy");
                this._objectivesList = void 0;
                this._pendingObjectiveRemoves.push("entropy");
                this._entropyDropped = this.generation;
              }
            }
          } else if (!currentObjKeys.includes("entropy") && this._entropyDropped && dyn.readdEntropyAfter != null) {
            if (this.generation - this._entropyDropped >= dyn.readdEntropyAfter) {
              this.registerObjective("entropy", "max", (genome) => this._structuralEntropy(genome));
              this._pendingObjectiveAdds.push("entropy");
              this._entropyDropped = void 0;
            }
          }
        } else if (this.options.multiObjective.autoEntropy) {
          const addAt = 3;
          if (this.generation >= addAt && !currentObjKeys.includes("entropy")) {
            this.registerObjective("entropy", "max", (genome) => this._structuralEntropy(genome));
            this._pendingObjectiveAdds.push("entropy");
          }
        }
      }
      for (const k of currentObjKeys)
        this._objectiveAges.set(k, (this._objectiveAges.get(k) || 0) + 1);
      for (const added of this._pendingObjectiveAdds)
        this._objectiveAges.set(added, 0);
    } catch {
    }
    try {
      const mo = this.options.multiObjective;
      if (mo?.enabled && mo.pruneInactive && mo.pruneInactive.enabled === false) {
        const keys = this._getObjectives().map((obj) => obj.key);
        if (keys.includes("fitness") && keys.length > 1 && !this._fitnessSuppressedOnce) {
          this._suppressFitnessObjective = true;
          this._fitnessSuppressedOnce = true;
          this._objectivesList = void 0;
        }
      }
    } catch {
    }
    let objImportance = null;
    try {
      const objsList = this._getObjectives();
      if (objsList.length) {
        objImportance = {};
        const pop = this.population;
        for (const obj of objsList) {
          const vals = pop.map((genome) => obj.accessor(genome));
          const min = Math.min(...vals);
          const max = Math.max(...vals);
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          const varV = vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (vals.length || 1);
          objImportance[obj.key] = { range: max - min, var: varV };
        }
        this._lastObjImportance = objImportance;
      }
    } catch {
    }
    if (this.options.telemetry?.enabled || true) {
      const telemetry = (init_neat_telemetry(), __toCommonJS(neat_telemetry_exports));
      const entry = telemetry.buildTelemetryEntry.call(this, fittest);
      telemetry.recordTelemetryEntry.call(this, entry);
    }
    if ((fittest.score ?? -Infinity) > this._bestGlobalScore) {
      this._bestGlobalScore = fittest.score ?? -Infinity;
      this._lastGlobalImproveGeneration = this.generation;
    }
    const newPopulation = [];
    const elitismCount = Math.max(0, Math.min(this.options.elitism || 0, this.population.length));
    for (let i = 0; i < elitismCount; i++) {
      const elite = this.population[i];
      if (elite)
        newPopulation.push(elite);
    }
    const desiredPop = Math.max(0, this.options.popsize || 0);
    const remainingSlotsAfterElites = Math.max(0, desiredPop - newPopulation.length);
    const provenanceCount = Math.max(0, Math.min(this.options.provenance || 0, remainingSlotsAfterElites));
    for (let i = 0; i < provenanceCount; i++) {
      if (this.options.network) {
        newPopulation.push(Network.fromJSON(this.options.network.toJSON()));
      } else {
        newPopulation.push(new Network(this.input, this.output, {
          minHidden: this.options.minHidden
        }));
      }
    }
    if (this.options.speciation && this._species.length > 0) {
      this._suppressTournamentError = true;
      const remaining = desiredPop - newPopulation.length;
      if (remaining > 0) {
        const ageCfg = this.options.speciesAgeBonus || {};
        const youngT = ageCfg.youngThreshold ?? 5;
        const youngM = ageCfg.youngMultiplier ?? 1.3;
        const oldT = ageCfg.oldThreshold ?? 30;
        const oldM = ageCfg.oldMultiplier ?? 0.7;
        const speciesAdjusted = this._species.map((species) => {
          const base = species.members.reduce((a, member) => a + (member.score || 0), 0);
          const age = this.generation - species.lastImproved;
          if (age <= youngT)
            return base * youngM;
          if (age >= oldT)
            return base * oldM;
          return base;
        });
        const totalAdj = speciesAdjusted.reduce((a, b) => a + b, 0) || 1;
        const minOff = this.options.speciesAllocation?.minOffspring ?? 1;
        const rawShares = this._species.map((_, idx) => speciesAdjusted[idx] / totalAdj * remaining);
        const offspringAlloc = rawShares.map((s) => Math.floor(s));
        for (let i = 0; i < offspringAlloc.length; i++)
          if (offspringAlloc[i] < minOff && remaining >= this._species.length * minOff)
            offspringAlloc[i] = minOff;
        const allocated = offspringAlloc.reduce((a, b) => a + b, 0);
        let slotsLeft = remaining - allocated;
        const remainders = rawShares.map((s, i) => ({
          i,
          frac: s - Math.floor(s)
        }));
        remainders.sort((a, b) => b.frac - a.frac);
        for (const remainderEntry of remainders) {
          if (slotsLeft <= 0)
            break;
          offspringAlloc[remainderEntry.i]++;
          slotsLeft--;
        }
        if (slotsLeft < 0) {
          const order = offspringAlloc.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v);
          for (const orderEntry of order) {
            if (slotsLeft === 0)
              break;
            if (offspringAlloc[orderEntry.i] > minOff) {
              offspringAlloc[orderEntry.i]--;
              slotsLeft++;
            }
          }
        }
        this._lastOffspringAlloc = this._species.map((species, i) => ({
          id: species.id,
          alloc: offspringAlloc[i] || 0
        }));
        this._prevInbreedingCount = this._lastInbreedingCount;
        this._lastInbreedingCount = 0;
        offspringAlloc.forEach((count, idx) => {
          if (count <= 0)
            return;
          const species = this._species[idx];
          this._sortSpeciesMembers(species);
          const survivors = species.members.slice(0, Math.max(1, Math.floor(species.members.length * (this.options.survivalThreshold || 0.5))));
          for (let k = 0; k < count; k++) {
            const parentA = survivors[Math.floor(this._getRNG()() * survivors.length)];
            let parentB;
            if (this.options.crossSpeciesMatingProb && this._species.length > 1 && this._getRNG()() < (this.options.crossSpeciesMatingProb || 0)) {
              let otherIdx = idx;
              let guard = 0;
              while (otherIdx === idx && guard++ < 5)
                otherIdx = Math.floor(this._getRNG()() * this._species.length);
              const otherSpecies = this._species[otherIdx];
              this._sortSpeciesMembers(otherSpecies);
              const otherParents = otherSpecies.members.slice(0, Math.max(1, Math.floor(otherSpecies.members.length * (this.options.survivalThreshold || 0.5))));
              parentB = otherParents[Math.floor(this._getRNG()() * otherParents.length)];
            } else {
              parentB = survivors[Math.floor(this._getRNG()() * survivors.length)];
            }
            const child = Network.crossOver(parentA, parentB, this.options.equal || false);
            child._reenableProb = this.options.reenableProb;
            child._id = this._nextGenomeId++;
            if (this._lineageEnabled) {
              child._parents = [
                parentA._id,
                parentB._id
              ];
              const d1 = parentA._depth ?? 0;
              const d2 = parentB._depth ?? 0;
              child._depth = 1 + Math.max(d1, d2);
              if (parentA._id === parentB._id)
                this._lastInbreedingCount++;
            }
            newPopulation.push(child);
          }
        });
        this._suppressTournamentError = false;
      }
    } else {
      this._suppressTournamentError = true;
      const toBreed = Math.max(0, desiredPop - newPopulation.length);
      for (let i = 0; i < toBreed; i++)
        newPopulation.push(this.getOffspring());
      this._suppressTournamentError = false;
    }
    for (const genome of newPopulation) {
      if (!genome)
        continue;
      this.ensureMinHiddenNodes(genome);
      this.ensureNoDeadEnds(genome);
    }
    this.population = newPopulation;
    try {
      (init_neat_pruning(), __toCommonJS(neat_pruning_exports)).applyEvolutionPruning.call(this);
    } catch {
    }
    try {
      (init_neat_pruning(), __toCommonJS(neat_pruning_exports)).applyAdaptivePruning.call(this);
    } catch {
    }
    this.mutate();
    try {
      (init_neat_adaptive(), __toCommonJS(neat_adaptive_exports)).applyAdaptiveMutation.call(this);
    } catch {
    }
    this.population.forEach((genome) => {
      if (genome._compatCache)
        delete genome._compatCache;
    });
    this.population.forEach((genome) => genome.score = void 0);
    this.generation++;
    if (this.options.speciation)
      this._updateSpeciesStagnation();
    if ((this.options.globalStagnationGenerations || 0) > 0 && this.generation - this._lastGlobalImproveGeneration > (this.options.globalStagnationGenerations || 0)) {
      const replaceFraction = 0.2;
      const startIdx = Math.max(this.options.elitism || 0, Math.floor(this.population.length * (1 - replaceFraction)));
      for (let i = startIdx; i < this.population.length; i++) {
        const fresh = new Network(this.input, this.output, {
          minHidden: this.options.minHidden
        });
        fresh.score = void 0;
        fresh._reenableProb = this.options.reenableProb;
        fresh._id = this._nextGenomeId++;
        if (this._lineageEnabled) {
          fresh._parents = [];
          fresh._depth = 0;
        }
        try {
          this.ensureMinHiddenNodes(fresh);
          this.ensureNoDeadEnds(fresh);
          const hiddenCount = fresh.nodes.filter((n) => n.type === "hidden").length;
          if (hiddenCount === 0) {
            const NodeCls = (init_node(), __toCommonJS(node_exports)).default;
            const newNode = new NodeCls("hidden");
            fresh.nodes.splice(fresh.nodes.length - fresh.output, 0, newNode);
            const inputNodes = fresh.nodes.filter((n) => n.type === "input");
            const outputNodes = fresh.nodes.filter((n) => n.type === "output");
            if (inputNodes.length && outputNodes.length) {
              try {
                fresh.connect(inputNodes[0], newNode, 1);
              } catch {
              }
              try {
                fresh.connect(newNode, outputNodes[0], 1);
              } catch {
              }
            }
          }
        } catch {
        }
        this.population[i] = fresh;
      }
      this._lastGlobalImproveGeneration = this.generation;
    }
    if (this.options.reenableProb !== void 0) {
      let reenableSuccessTotal = 0, reenableAttemptsTotal = 0;
      for (const genome of this.population) {
        reenableSuccessTotal += genome._reenableSuccess || 0;
        reenableAttemptsTotal += genome._reenableAttempts || 0;
        genome._reenableSuccess = 0;
        genome._reenableAttempts = 0;
      }
      if (reenableAttemptsTotal > 20) {
        const ratio = reenableSuccessTotal / reenableAttemptsTotal;
        const target = 0.3;
        const delta = ratio - target;
        this.options.reenableProb = Math.min(0.9, Math.max(0.05, this.options.reenableProb - delta * 0.1));
      }
    }
    try {
      (init_neat_adaptive(), __toCommonJS(neat_adaptive_exports)).applyOperatorAdaptation.call(this);
    } catch {
    }
    const endTime = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
    this._lastEvolveDuration = endTime - startTime;
    try {
      if (!this._speciesHistory)
        this._speciesHistory = [];
      if (!this.options.speciesAllocation?.extendedHistory) {
        if (this._speciesHistory.length === 0 || this._speciesHistory[this._speciesHistory.length - 1].generation !== this.generation) {
          this._speciesHistory.push({
            generation: this.generation,
            stats: this._species.map((species) => ({
              id: species.id,
              size: species.members.length,
              best: species.bestScore,
              lastImproved: species.lastImproved
            }))
          });
          if (this._speciesHistory.length > 200)
            this._speciesHistory.shift();
        }
      }
    } catch {
    }
    return fittest;
  }
  var init_neat_evolve = __esm({
    "dist/neat/neat.evolve.js"() {
      "use strict";
      init_network();
      init_neat_multiobjective();
    }
  });

  // dist/neat/neat.evaluate.js
  async function evaluate() {
    const options = this.options || {};
    if (options.fitnessPopulation) {
      if (options.clear)
        this.population.forEach((g) => g.clear && g.clear());
      await this.fitness(this.population);
    } else {
      for (const genome of this.population) {
        if (options.clear && genome.clear)
          genome.clear();
        const fitnessValue = await this.fitness(genome);
        genome.score = fitnessValue;
      }
    }
    try {
      const noveltyOptions = options.novelty;
      if (noveltyOptions?.enabled && typeof noveltyOptions.descriptor === "function") {
        const kNeighbors = Math.max(1, noveltyOptions.k || 3);
        const blendFactor = noveltyOptions.blendFactor ?? 0.3;
        const descriptors = this.population.map((g) => {
          try {
            return noveltyOptions.descriptor(g) || [];
          } catch {
            return [];
          }
        });
        const distanceMatrix = [];
        for (let i = 0; i < descriptors.length; i++) {
          distanceMatrix[i] = [];
          for (let j = 0; j < descriptors.length; j++) {
            if (i === j) {
              distanceMatrix[i][j] = 0;
              continue;
            }
            const descA = descriptors[i];
            const descB = descriptors[j];
            let sqSum = 0;
            const commonLen = Math.min(descA.length, descB.length);
            for (let t = 0; t < commonLen; t++) {
              const delta = (descA[t] || 0) - (descB[t] || 0);
              sqSum += delta * delta;
            }
            distanceMatrix[i][j] = Math.sqrt(sqSum);
          }
        }
        for (let i = 0; i < this.population.length; i++) {
          const sortedRow = distanceMatrix[i].toSorted((a, b) => a - b);
          const neighbours = sortedRow.slice(1, kNeighbors + 1);
          const novelty = neighbours.length ? neighbours.reduce((a, b) => a + b, 0) / neighbours.length : 0;
          this.population[i]._novelty = novelty;
          if (typeof this.population[i].score === "number") {
            this.population[i].score = (1 - blendFactor) * this.population[i].score + blendFactor * novelty;
          }
          if (!this._noveltyArchive)
            this._noveltyArchive = [];
          const archiveAddThreshold = noveltyOptions.archiveAddThreshold ?? Infinity;
          if (noveltyOptions.archiveAddThreshold === 0 || novelty > archiveAddThreshold) {
            if (this._noveltyArchive.length < 200)
              this._noveltyArchive.push({ desc: descriptors[i], novelty });
          }
        }
      }
    } catch {
    }
    if (!this._diversityStats)
      this._diversityStats = {};
    try {
      const entropySharingOptions = options.entropySharingTuning;
      if (entropySharingOptions?.enabled) {
        const targetVar = entropySharingOptions.targetEntropyVar ?? 0.2;
        const adjustRate = entropySharingOptions.adjustRate ?? 0.1;
        const minSigma = entropySharingOptions.minSigma ?? 0.1;
        const maxSigma = entropySharingOptions.maxSigma ?? 10;
        const currentVarEntropy = this._diversityStats.varEntropy;
        if (typeof currentVarEntropy === "number") {
          let sigma = this.options.sharingSigma ?? 0;
          if (currentVarEntropy < targetVar * 0.9)
            sigma = Math.max(minSigma, sigma * (1 - adjustRate));
          else if (currentVarEntropy > targetVar * 1.1)
            sigma = Math.min(maxSigma, sigma * (1 + adjustRate));
          this.options.sharingSigma = sigma;
        }
      }
    } catch {
    }
    try {
      const entropyCompatOptions = options.entropyCompatTuning;
      if (entropyCompatOptions?.enabled) {
        const meanEntropy = this._diversityStats.meanEntropy;
        const targetEntropy = entropyCompatOptions.targetEntropy ?? 0.5;
        const deadband = entropyCompatOptions.deadband ?? 0.05;
        const adjustRate = entropyCompatOptions.adjustRate ?? 0.05;
        let threshold = this.options.compatibilityThreshold ?? 3;
        if (typeof meanEntropy === "number") {
          if (meanEntropy < targetEntropy - deadband)
            threshold = Math.max(entropyCompatOptions.minThreshold ?? 0.5, threshold * (1 - adjustRate));
          else if (meanEntropy > targetEntropy + deadband)
            threshold = Math.min(entropyCompatOptions.maxThreshold ?? 10, threshold * (1 + adjustRate));
          this.options.compatibilityThreshold = threshold;
        }
      }
    } catch {
    }
    try {
      if (this.options.speciation && (this.options.targetSpecies || this.options.compatAdjust || this.options.speciesAllocation?.extendedHistory)) {
        this._speciate();
      }
    } catch {
    }
    try {
      const autoDistanceCoeffOptions = this.options.autoDistanceCoeffTuning;
      if (autoDistanceCoeffOptions?.enabled && this.options.speciation) {
        const connectionSizes = this.population.map((g) => g.connections.length);
        const meanSize = connectionSizes.reduce((a, b) => a + b, 0) / (connectionSizes.length || 1);
        const connVar = connectionSizes.reduce((a, b) => a + (b - meanSize) * (b - meanSize), 0) / (connectionSizes.length || 1);
        const adjustRate = autoDistanceCoeffOptions.adjustRate ?? 0.05;
        const minCoeff = autoDistanceCoeffOptions.minCoeff ?? 0.05;
        const maxCoeff = autoDistanceCoeffOptions.maxCoeff ?? 8;
        if (this._lastConnVar === void 0 || this._lastConnVar === null) {
          this._lastConnVar = connVar;
          try {
            this.options.excessCoeff = Math.min(maxCoeff, (this.options.excessCoeff ?? 1) * (1 + adjustRate));
            this.options.disjointCoeff = Math.min(maxCoeff, (this.options.disjointCoeff ?? 1) * (1 + adjustRate));
          } catch {
          }
        }
        if (connVar < this._lastConnVar * 0.95) {
          this.options.excessCoeff = Math.min(maxCoeff, this.options.excessCoeff * (1 + adjustRate));
          this.options.disjointCoeff = Math.min(maxCoeff, this.options.disjointCoeff * (1 + adjustRate));
        } else if (connVar > this._lastConnVar * 1.05) {
          this.options.excessCoeff = Math.max(minCoeff, this.options.excessCoeff * (1 - adjustRate));
          this.options.disjointCoeff = Math.max(minCoeff, this.options.disjointCoeff * (1 - adjustRate));
        }
        this._lastConnVar = connVar;
      }
    } catch {
    }
    try {
      if (this.options.multiObjective?.enabled && this.options.multiObjective.autoEntropy) {
        if (!this.options.multiObjective.dynamic?.enabled) {
          const keys = this._getObjectives().map((o) => o.key);
          if (!keys.includes("entropy")) {
            this.registerObjective("entropy", "max", (g) => this._structuralEntropy(g));
            this._pendingObjectiveAdds.push("entropy");
            this._objectivesList = void 0;
          }
        }
      }
    } catch {
    }
  }
  var init_neat_evaluate = __esm({
    "dist/neat/neat.evaluate.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.helpers.js
  function spawnFromParent(parentGenome, mutateCount = 1) {
    const clone = parentGenome.clone ? parentGenome.clone() : (init_network(), __toCommonJS(network_exports)).default.fromJSON(parentGenome.toJSON());
    clone.score = void 0;
    clone._reenableProb = this.options.reenableProb;
    clone._id = this._nextGenomeId++;
    clone._parents = [parentGenome._id];
    clone._depth = (parentGenome._depth ?? 0) + 1;
    this.ensureMinHiddenNodes(clone);
    this.ensureNoDeadEnds(clone);
    for (let mutationIndex = 0; mutationIndex < mutateCount; mutationIndex++) {
      try {
        let selectedMutationMethod = this.selectMutationMethod(clone, false);
        if (Array.isArray(selectedMutationMethod)) {
          const candidateMutations = selectedMutationMethod;
          selectedMutationMethod = candidateMutations[Math.floor(this._getRNG()() * candidateMutations.length)];
        }
        if (selectedMutationMethod && selectedMutationMethod.name) {
          clone.mutate(selectedMutationMethod);
        }
      } catch {
      }
    }
    this._invalidateGenomeCaches(clone);
    return clone;
  }
  function addGenome(genome, parents) {
    try {
      genome.score = void 0;
      genome._reenableProb = this.options.reenableProb;
      genome._id = this._nextGenomeId++;
      genome._parents = Array.isArray(parents) ? parents.slice() : [];
      genome._depth = 0;
      if (genome._parents.length) {
        const parentDepths = genome._parents.map((pid) => this.population.find((g) => g._id === pid)).filter(Boolean).map((g) => g._depth ?? 0);
        genome._depth = parentDepths.length ? Math.max(...parentDepths) + 1 : 1;
      }
      this.ensureMinHiddenNodes(genome);
      this.ensureNoDeadEnds(genome);
      this._invalidateGenomeCaches(genome);
      this.population.push(genome);
    } catch (error) {
      this.population.push(genome);
    }
  }
  function createPool(seedNetwork) {
    try {
      this.population = [];
      const poolSize = this.options?.popsize || 50;
      for (let genomeIndex = 0; genomeIndex < poolSize; genomeIndex++) {
        const genomeCopy = seedNetwork ? Network.fromJSON(seedNetwork.toJSON()) : new Network(this.input, this.output, {
          minHidden: this.options?.minHidden
        });
        genomeCopy.score = void 0;
        try {
          this.ensureNoDeadEnds(genomeCopy);
        } catch {
        }
        genomeCopy._reenableProb = this.options.reenableProb;
        genomeCopy._id = this._nextGenomeId++;
        if (this._lineageEnabled) {
          genomeCopy._parents = [];
          genomeCopy._depth = 0;
        }
        this.population.push(genomeCopy);
      }
    } catch {
    }
  }
  var init_neat_helpers = __esm({
    "dist/neat/neat.helpers.js"() {
      "use strict";
      init_network();
    }
  });

  // dist/neat/neat.objectives.js
  function _getObjectives() {
    if (this._objectivesList)
      return this._objectivesList;
    const objectivesList = [];
    if (!this._suppressFitnessObjective) {
      objectivesList.push({
        key: "fitness",
        direction: "max",
        accessor: (genome) => genome.score || 0
      });
    }
    if (this.options.multiObjective?.enabled && Array.isArray(this.options.multiObjective.objectives)) {
      for (const candidateObjective of this.options.multiObjective.objectives) {
        if (!candidateObjective || !candidateObjective.key || typeof candidateObjective.accessor !== "function")
          continue;
        objectivesList.push(candidateObjective);
      }
    }
    this._objectivesList = objectivesList;
    return objectivesList;
  }
  function registerObjective(key, direction, accessor) {
    if (!this.options.multiObjective)
      this.options.multiObjective = { enabled: true };
    const multiObjectiveOptions = this.options.multiObjective;
    if (!multiObjectiveOptions.objectives)
      multiObjectiveOptions.objectives = [];
    multiObjectiveOptions.objectives = multiObjectiveOptions.objectives.filter((existingObjective) => existingObjective.key !== key);
    multiObjectiveOptions.objectives.push({ key, direction, accessor });
    this._objectivesList = void 0;
  }
  function clearObjectives() {
    if (this.options.multiObjective?.objectives)
      this.options.multiObjective.objectives = [];
    this._objectivesList = void 0;
  }
  var init_neat_objectives = __esm({
    "dist/neat/neat.objectives.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.diversity.js
  function structuralEntropy2(graph) {
    const outDegrees = graph.nodes.map((node) => node.connections.out.length);
    const totalOut = outDegrees.reduce((acc, v) => acc + v, 0) || 1;
    const probabilities = outDegrees.map((d) => d / totalOut).filter((p) => p > 0);
    let entropy = 0;
    for (const p of probabilities) {
      entropy -= p * Math.log(p);
    }
    return entropy;
  }
  function arrayMean(values) {
    if (!values.length)
      return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }
  function arrayVariance(values) {
    if (!values.length)
      return 0;
    const m = arrayMean(values);
    return arrayMean(values.map((v) => (v - m) * (v - m)));
  }
  function computeDiversityStats2(population, compatibilityComputer) {
    if (!population.length)
      return void 0;
    const lineageDepths = [];
    for (const genome of population) {
      if (typeof genome._depth === "number") {
        lineageDepths.push(genome._depth);
      }
    }
    const lineageMeanDepth = arrayMean(lineageDepths);
    let depthPairAbsDiffSum = 0;
    let depthPairCount = 0;
    for (let i = 0; i < lineageDepths.length && i < 30; i++) {
      for (let j = i + 1; j < lineageDepths.length && j < 30; j++) {
        depthPairAbsDiffSum += Math.abs(lineageDepths[i] - lineageDepths[j]);
        depthPairCount++;
      }
    }
    const lineageMeanPairDist = depthPairCount ? depthPairAbsDiffSum / depthPairCount : 0;
    const nodeCounts = population.map((g) => g.nodes.length);
    const connectionCounts = population.map((g) => g.connections.length);
    const meanNodes = arrayMean(nodeCounts);
    const meanConns = arrayMean(connectionCounts);
    const nodeVar = arrayVariance(nodeCounts);
    const connVar = arrayVariance(connectionCounts);
    let compatSum = 0;
    let compatPairCount = 0;
    for (let i = 0; i < population.length && i < 25; i++) {
      for (let j = i + 1; j < population.length && j < 25; j++) {
        compatSum += compatibilityComputer._compatibilityDistance(population[i], population[j]);
        compatPairCount++;
      }
    }
    const meanCompat = compatPairCount ? compatSum / compatPairCount : 0;
    const graphletEntropy = arrayMean(population.map((g) => structuralEntropy2(g)));
    return {
      lineageMeanDepth,
      lineageMeanPairDist,
      meanNodes,
      meanConns,
      nodeVar,
      connVar,
      meanCompat,
      graphletEntropy,
      population: population.length
    };
  }
  var init_neat_diversity = __esm({
    "dist/neat/neat.diversity.js"() {
      "use strict";
      init_network();
    }
  });

  // dist/neat/neat.compat.js
  function _fallbackInnov(connection) {
    const fromIndex = connection.from?.index ?? 0;
    const toIndex = connection.to?.index ?? 0;
    return fromIndex * 1e5 + toIndex;
  }
  function _compatibilityDistance(genomeA, genomeB) {
    if (!this._compatCacheGen || this._compatCacheGen !== this.generation) {
      this._compatCacheGen = this.generation;
      this._compatDistCache = /* @__PURE__ */ new Map();
    }
    const key = genomeA._id < genomeB._id ? `${genomeA._id}|${genomeB._id}` : `${genomeB._id}|${genomeA._id}`;
    const cacheMap = this._compatDistCache;
    if (cacheMap.has(key))
      return cacheMap.get(key);
    const getCache = (network) => {
      if (!network._compatCache) {
        const list = network.connections.map((conn) => [
          conn.innovation ?? this._fallbackInnov(conn),
          conn.weight
        ]);
        list.sort((x, y) => x[0] - y[0]);
        network._compatCache = list;
      }
      return network._compatCache;
    };
    const aList = getCache(genomeA);
    const bList = getCache(genomeB);
    let indexA = 0, indexB = 0;
    let matchingCount = 0, disjoint = 0, excess = 0;
    let weightDifferenceSum = 0;
    const maxInnovA = aList.length ? aList[aList.length - 1][0] : 0;
    const maxInnovB = bList.length ? bList[bList.length - 1][0] : 0;
    while (indexA < aList.length && indexB < bList.length) {
      const [innovA, weightA] = aList[indexA];
      const [innovB, weightB] = bList[indexB];
      if (innovA === innovB) {
        matchingCount++;
        weightDifferenceSum += Math.abs(weightA - weightB);
        indexA++;
        indexB++;
      } else if (innovA < innovB) {
        if (innovA > maxInnovB)
          excess++;
        else
          disjoint++;
        indexA++;
      } else {
        if (innovB > maxInnovA)
          excess++;
        else
          disjoint++;
        indexB++;
      }
    }
    if (indexA < aList.length)
      excess += aList.length - indexA;
    if (indexB < bList.length)
      excess += bList.length - indexB;
    const N = Math.max(1, Math.max(aList.length, bList.length));
    const avgWeightDiff = matchingCount ? weightDifferenceSum / matchingCount : 0;
    const opts = this.options;
    const dist = opts.excessCoeff * excess / N + opts.disjointCoeff * disjoint / N + opts.weightDiffCoeff * avgWeightDiff;
    cacheMap.set(key, dist);
    return dist;
  }
  var init_neat_compat = __esm({
    "dist/neat/neat.compat.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.speciation.js
  function _speciate() {
    this._prevSpeciesMembers = this._prevSpeciesMembers ?? /* @__PURE__ */ new Map();
    this._prevSpeciesMembers.clear();
    for (const species of this._species) {
      const previousMembers = /* @__PURE__ */ new Set();
      for (const member of species.members)
        previousMembers.add(member._id);
      this._prevSpeciesMembers.set(species.id, previousMembers);
    }
    this._species.forEach((species) => species.members = []);
    for (const genome of this.population) {
      let isAssigned = false;
      for (const species of this._species) {
        const compatibilityDistance = this._compatibilityDistance(genome, species.representative);
        if (compatibilityDistance < (this.options.compatibilityThreshold ?? 3)) {
          species.members.push(genome);
          isAssigned = true;
          break;
        }
      }
      if (!isAssigned) {
        const newSpeciesId = this._nextSpeciesId++;
        this._species.push({
          id: newSpeciesId,
          members: [genome],
          representative: genome,
          lastImproved: this.generation,
          bestScore: genome.score ?? -Infinity
        });
        this._speciesCreated.set(newSpeciesId, this.generation);
      }
    }
    const options = this.options;
    const compatAdjust = options.compatAdjust ?? {};
    const minThreshold = compatAdjust.minThreshold ?? options.minThreshold ?? 1;
    const maxThreshold = compatAdjust.maxThreshold ?? options.maxThreshold ?? 10;
    const targetSpeciesCount = options.targetSpecies ?? 5;
    const observedSpeciesCount = this._species.length;
    if (typeof this._compatIntegral !== "number")
      this._compatIntegral = 0;
    if (typeof options.compatibilityThreshold === "number") {
      const speciesError = targetSpeciesCount - observedSpeciesCount;
      const proportionalGain = compatAdjust.kp ?? 0.5;
      const integralGain = compatAdjust.ki ?? 10;
      let thresholdDelta = proportionalGain * speciesError;
      this._compatIntegral += speciesError;
      thresholdDelta += integralGain * this._compatIntegral;
      let updatedThreshold = options.compatibilityThreshold - thresholdDelta;
      if (updatedThreshold < minThreshold) {
        updatedThreshold = minThreshold;
        this._compatIntegral = 0;
      } else if (updatedThreshold > maxThreshold) {
        updatedThreshold = maxThreshold;
        this._compatIntegral = 0;
      }
      options.compatibilityThreshold = updatedThreshold;
    }
    if (typeof options.compatibilityThreshold === "number") {
      if (options.compatibilityThreshold < minThreshold)
        options.compatibilityThreshold = minThreshold;
      if (options.compatibilityThreshold > maxThreshold)
        options.compatibilityThreshold = maxThreshold;
    }
    this._species = this._species.filter((species) => species.members.length > 0);
    this._species.forEach((species) => species.representative = species.members[0]);
    const ageProtection = options.speciesAgeProtection ?? {
      grace: 3,
      oldPenalty: 0.5
    };
    for (const species of this._species) {
      const createdGeneration = this._speciesCreated.get(species.id) ?? this.generation;
      const speciesAge = this.generation - createdGeneration;
      if (speciesAge >= (ageProtection.grace ?? 3) * 10) {
        const penalty = ageProtection.oldPenalty ?? 0.5;
        if (penalty < 1)
          species.members.forEach((member) => {
            if (typeof member.score === "number")
              member.score *= penalty;
          });
      }
    }
    if (options.speciesAllocation?.extendedHistory) {
      const stats = this._species.map((species) => {
        const members = species.members;
        const sizes = members.map((member) => ({
          nodes: member.nodes.length,
          conns: member.connections.length,
          score: member.score ?? 0,
          ent: this._structuralEntropy(member)
        }));
        const average = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
        const meanNodes = average(sizes.map((x) => x.nodes));
        const meanConns = average(sizes.map((x) => x.conns));
        let innovationSum = 0;
        let innovationCount = 0;
        let maxInnovation = -Infinity;
        let minInnovation = Infinity;
        let enabledCount = 0;
        let disabledCount = 0;
        for (const member of members)
          for (const connection of member.connections) {
            const innovation = connection.innovation ?? this._fallbackInnov(connection);
            innovationSum += innovation;
            innovationCount++;
            if (innovation > maxInnovation)
              maxInnovation = innovation;
            if (innovation < minInnovation)
              minInnovation = innovation;
            if (connection.enabled === false)
              disabledCount++;
            else
              enabledCount++;
          }
        const meanInnovation = innovationCount ? innovationSum / innovationCount : 0;
        return {
          id: species.id,
          size: species.members.length,
          best: species.bestScore,
          meanNodes,
          meanConns,
          meanInnovation,
          innovationRange: isFinite(maxInnovation) && isFinite(minInnovation) && maxInnovation > minInnovation ? maxInnovation - minInnovation : 0,
          enabledRatio: enabledCount + disabledCount ? enabledCount / (enabledCount + disabledCount) : 0
        };
      });
      this._speciesHistory.push({ generation: this.generation, stats });
    } else {
      this._speciesHistory.push({
        generation: this.generation,
        stats: this._species.map((species) => ({
          id: species.id,
          size: species.members.length,
          best: species.bestScore
        }))
      });
    }
    if (this._speciesHistory.length > 200)
      this._speciesHistory.shift();
  }
  function _applyFitnessSharing() {
    const sigma = this.options.sharingSigma ?? 0;
    if (sigma > 0) {
      for (const s of this._species) {
        const members = s.members;
        for (let i = 0; i < members.length; i++) {
          const mi = members[i];
          if (typeof mi.score !== "number")
            continue;
          let sum = 0;
          for (let j = 0; j < members.length; j++) {
            const mj = members[j];
            const d = i === j ? 0 : this._compatibilityDistance(mi, mj);
            if (d < sigma) {
              const r = d / sigma;
              sum += 1 - r * r;
            }
          }
          if (sum <= 0)
            sum = 1;
          mi.score = mi.score / sum;
        }
      }
    } else {
      for (const s of this._species) {
        const members = s.members;
        const size = members.length || 1;
        for (const m of members)
          if (typeof m.score === "number")
            m.score = m.score / size;
      }
    }
  }
  function _sortSpeciesMembers(sp) {
    sp.members.sort((a, b) => (b.score || 0) - (a.score || 0));
  }
  function _updateSpeciesStagnation() {
    const win = this.options.stagnationGenerations ?? 15;
    for (const s of this._species) {
      _sortSpeciesMembers.call(this, s);
      const top = s.members[0];
      if ((top?.score ?? -Infinity) > (s.bestScore ?? -Infinity)) {
        s.bestScore = top.score ?? -Infinity;
        s.lastImproved = this.generation;
      }
    }
    const survivors = this._species.filter((s) => this.generation - (s.lastImproved ?? 0) <= win);
    if (survivors.length)
      this._species = survivors;
  }
  var init_neat_speciation = __esm({
    "dist/neat/neat.speciation.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.species.js
  function getSpeciesStats() {
    const ctx = this;
    const speciesArray = ctx._species || [];
    return speciesArray.map((species) => ({
      id: species.id,
      size: species.members && species.members.length || 0,
      bestScore: species.bestScore || 0,
      lastImproved: species.lastImproved || 0
    }));
  }
  function getSpeciesHistory() {
    const ctx = this;
    const speciesHistory = ctx._speciesHistory || [];
    const options = this.options;
    if (options?.speciesAllocation?.extendedHistory) {
      for (const generationEntry of speciesHistory) {
        for (const speciesStat of generationEntry.stats) {
          if ("innovationRange" in speciesStat && "enabledRatio" in speciesStat)
            continue;
          const speciesObj = (ctx._species || []).find((s) => s.id === speciesStat.id);
          if (speciesObj && speciesObj.members && speciesObj.members.length) {
            let maxInnovation = -Infinity;
            let minInnovation = Infinity;
            let enabledCount = 0;
            let disabledCount = 0;
            for (const member of speciesObj.members) {
              for (const connection of member.connections) {
                const innovationId = connection.innovation ?? ctx._fallbackInnov?.(connection) ?? 0;
                if (innovationId > maxInnovation)
                  maxInnovation = innovationId;
                if (innovationId < minInnovation)
                  minInnovation = innovationId;
                if (connection.enabled === false)
                  disabledCount++;
                else
                  enabledCount++;
              }
            }
            speciesStat.innovationRange = isFinite(maxInnovation) && isFinite(minInnovation) && maxInnovation > minInnovation ? maxInnovation - minInnovation : 0;
            speciesStat.enabledRatio = enabledCount + disabledCount ? enabledCount / (enabledCount + disabledCount) : 0;
          }
        }
      }
    }
    return speciesHistory;
  }
  var init_neat_species = __esm({
    "dist/neat/neat.species.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.telemetry.exports.js
  var exportTelemetryJSONL, exportTelemetryCSV, COMPLEXITY_PREFIX, PERF_PREFIX, LINEAGE_PREFIX, DIVERSITY_PREFIX, HEADER_FRONTS, HEADER_OPS, HEADER_OBJECTIVES, HEADER_OBJ_AGES, HEADER_SPECIES_ALLOC, HEADER_OBJ_EVENTS, HEADER_OBJ_IMPORTANCE, collectTelemetryHeaderInfo, buildTelemetryHeaders, serializeTelemetryEntry, exportSpeciesHistoryCSV, HEADER_GENERATION, buildSpeciesHistoryCsv;
  var init_neat_telemetry_exports = __esm({
    "dist/neat/neat.telemetry.exports.js"() {
      "use strict";
      exportTelemetryJSONL = function() {
        return this._telemetry.map((entry) => JSON.stringify(entry)).join("\n");
      };
      exportTelemetryCSV = function(maxEntries = 500) {
        const recentTelemetry = Array.isArray(this._telemetry) ? this._telemetry.slice(-maxEntries) : [];
        if (!recentTelemetry.length)
          return "";
        const headerInfo = collectTelemetryHeaderInfo(recentTelemetry);
        const headers = buildTelemetryHeaders(headerInfo);
        const csvLines = [headers.join(",")];
        for (const telemetryEntry of recentTelemetry) {
          csvLines.push(serializeTelemetryEntry(telemetryEntry, headers));
        }
        return csvLines.join("\n");
      };
      COMPLEXITY_PREFIX = "complexity.";
      PERF_PREFIX = "perf.";
      LINEAGE_PREFIX = "lineage.";
      DIVERSITY_PREFIX = "diversity.";
      HEADER_FRONTS = "fronts";
      HEADER_OPS = "ops";
      HEADER_OBJECTIVES = "objectives";
      HEADER_OBJ_AGES = "objAges";
      HEADER_SPECIES_ALLOC = "speciesAlloc";
      HEADER_OBJ_EVENTS = "objEvents";
      HEADER_OBJ_IMPORTANCE = "objImportance";
      collectTelemetryHeaderInfo = (entries) => {
        const baseKeys = /* @__PURE__ */ new Set();
        const complexityKeys = /* @__PURE__ */ new Set();
        const perfKeys = /* @__PURE__ */ new Set();
        const lineageKeys = /* @__PURE__ */ new Set();
        const diversityLineageKeys = /* @__PURE__ */ new Set();
        let includeOps = false;
        let includeObjectives = false;
        let includeObjAges = false;
        let includeSpeciesAlloc = false;
        let includeObjEvents = false;
        let includeObjImportance = false;
        for (const entry of entries) {
          Object.keys(entry).forEach((k) => {
            if (k !== "complexity" && k !== "perf" && k !== "ops" && k !== HEADER_FRONTS) {
              baseKeys.add(k);
            }
          });
          if (Array.isArray(entry.fronts))
            baseKeys.add(HEADER_FRONTS);
          if (entry.complexity)
            Object.keys(entry.complexity).forEach((k) => complexityKeys.add(k));
          if (entry.perf)
            Object.keys(entry.perf).forEach((k) => perfKeys.add(k));
          if (entry.lineage)
            Object.keys(entry.lineage).forEach((k) => lineageKeys.add(k));
          if (entry.diversity) {
            if ("lineageMeanDepth" in entry.diversity)
              diversityLineageKeys.add("lineageMeanDepth");
            if ("lineageMeanPairDist" in entry.diversity)
              diversityLineageKeys.add("lineageMeanPairDist");
          }
          if ("rng" in entry)
            baseKeys.add("rng");
          if (Array.isArray(entry.ops) && entry.ops.length)
            includeOps = true;
          if (Array.isArray(entry.objectives))
            includeObjectives = true;
          if (entry.objAges)
            includeObjAges = true;
          if (Array.isArray(entry.speciesAlloc))
            includeSpeciesAlloc = true;
          if (Array.isArray(entry.objEvents) && entry.objEvents.length)
            includeObjEvents = true;
          if (entry.objImportance)
            includeObjImportance = true;
        }
        return {
          baseKeys,
          complexityKeys,
          perfKeys,
          lineageKeys,
          diversityLineageKeys,
          includeOps,
          includeObjectives,
          includeObjAges,
          includeSpeciesAlloc,
          includeObjEvents,
          includeObjImportance
        };
      };
      buildTelemetryHeaders = (info) => {
        const headers = [
          ...info.baseKeys,
          ...[...info.complexityKeys].map((k) => `${COMPLEXITY_PREFIX}${k}`),
          ...[...info.perfKeys].map((k) => `${PERF_PREFIX}${k}`),
          ...[...info.lineageKeys].map((k) => `${LINEAGE_PREFIX}${k}`),
          ...[...info.diversityLineageKeys].map((k) => `${DIVERSITY_PREFIX}${k}`)
        ];
        if (info.includeOps)
          headers.push(HEADER_OPS);
        if (info.includeObjectives)
          headers.push(HEADER_OBJECTIVES);
        if (info.includeObjAges)
          headers.push(HEADER_OBJ_AGES);
        if (info.includeSpeciesAlloc)
          headers.push(HEADER_SPECIES_ALLOC);
        if (info.includeObjEvents)
          headers.push(HEADER_OBJ_EVENTS);
        if (info.includeObjImportance)
          headers.push(HEADER_OBJ_IMPORTANCE);
        return headers;
      };
      serializeTelemetryEntry = (entry, headers) => {
        const row = [];
        for (const header of headers) {
          switch (true) {
            case header.startsWith(COMPLEXITY_PREFIX): {
              const key = header.slice(COMPLEXITY_PREFIX.length);
              const complexity = entry.complexity;
              row.push(complexity && key in complexity ? JSON.stringify(complexity[key]) : "");
              break;
            }
            case header.startsWith(PERF_PREFIX): {
              const key = header.slice(PERF_PREFIX.length);
              const perf = entry.perf;
              row.push(perf && key in perf ? JSON.stringify(perf[key]) : "");
              break;
            }
            case header.startsWith(LINEAGE_PREFIX): {
              const key = header.slice(LINEAGE_PREFIX.length);
              const lineage = entry.lineage;
              row.push(lineage && key in lineage ? JSON.stringify(lineage[key]) : "");
              break;
            }
            case header.startsWith(DIVERSITY_PREFIX): {
              const key = header.slice(DIVERSITY_PREFIX.length);
              const diversity = entry.diversity;
              row.push(diversity && key in diversity ? JSON.stringify(diversity[key]) : "");
              break;
            }
            case header === HEADER_FRONTS: {
              row.push(Array.isArray(entry.fronts) ? JSON.stringify(entry.fronts) : "");
              break;
            }
            case header === HEADER_OPS: {
              row.push(Array.isArray(entry.ops) ? JSON.stringify(entry.ops) : "");
              break;
            }
            case header === HEADER_OBJECTIVES: {
              row.push(Array.isArray(entry.objectives) ? JSON.stringify(entry.objectives) : "");
              break;
            }
            case header === HEADER_OBJ_AGES: {
              row.push(entry.objAges ? JSON.stringify(entry.objAges) : "");
              break;
            }
            case header === HEADER_SPECIES_ALLOC: {
              row.push(Array.isArray(entry.speciesAlloc) ? JSON.stringify(entry.speciesAlloc) : "");
              break;
            }
            case header === HEADER_OBJ_EVENTS: {
              row.push(Array.isArray(entry.objEvents) ? JSON.stringify(entry.objEvents) : "");
              break;
            }
            case header === HEADER_OBJ_IMPORTANCE: {
              row.push(entry.objImportance ? JSON.stringify(entry.objImportance) : "");
              break;
            }
            default: {
              row.push(JSON.stringify(entry[header]));
              break;
            }
          }
        }
        return row.join(",");
      };
      exportSpeciesHistoryCSV = function(maxEntries = 200) {
        if (!Array.isArray(this._speciesHistory))
          this._speciesHistory = [];
        if (!this._speciesHistory.length && Array.isArray(this._species) && this._species.length) {
          const stats = this._species.map((sp) => ({
            id: typeof sp.id === "number" ? sp.id : -1,
            size: Array.isArray(sp.members) ? sp.members.length : typeof sp.size === "number" ? sp.size : 0,
            bestScore: typeof sp.bestScore === "number" ? sp.bestScore : typeof sp.best === "number" ? sp.best : 0,
            lastImproved: typeof sp.lastImproved === "number" ? sp.lastImproved : 0
          }));
          this._speciesHistory.push({ generation: this.generation || 0, stats });
        }
        const recentHistory = this._speciesHistory.slice(-maxEntries);
        if (!recentHistory.length) {
          return "generation,id,size,best,lastImproved";
        }
        const headerKeySet = /* @__PURE__ */ new Set(["generation"]);
        for (const entry of recentHistory)
          for (const speciesStat of entry.stats)
            Object.keys(speciesStat).forEach((k) => headerKeySet.add(k));
        const headers = Array.from(headerKeySet);
        return buildSpeciesHistoryCsv(recentHistory, headers);
      };
      HEADER_GENERATION = "generation";
      buildSpeciesHistoryCsv = (recentHistory, headers) => {
        const lines = [headers.join(",")];
        for (const historyEntry of recentHistory) {
          for (const speciesStat of historyEntry.stats) {
            const rowCells = [];
            for (const header of headers) {
              if (header === HEADER_GENERATION) {
                rowCells.push(JSON.stringify(historyEntry.generation));
                continue;
              }
              rowCells.push(JSON.stringify(speciesStat[header]));
            }
            lines.push(rowCells.join(","));
          }
        }
        return lines.join("\n");
      };
    }
  });

  // dist/neat/neat.selection.js
  function sort() {
    this.population.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }
  function getParent() {
    const selectionOptions = this.options.selection;
    const selectionName = selectionOptions?.name;
    const getRngFactory = this._getRNG.bind(this);
    const population = this.population;
    switch (selectionName) {
      case "POWER":
        if (population[0]?.score !== void 0 && population[1]?.score !== void 0 && population[0].score < population[1].score) {
          this.sort();
        }
        const selectedIndex = Math.floor(Math.pow(getRngFactory()(), selectionOptions.power || 1) * population.length);
        return population[selectedIndex];
      case "FITNESS_PROPORTIONATE":
        let totalFitness = 0;
        let mostNegativeScore = 0;
        population.forEach((individual) => {
          mostNegativeScore = Math.min(mostNegativeScore, individual.score ?? 0);
          totalFitness += individual.score ?? 0;
        });
        const minFitnessShift = Math.abs(mostNegativeScore);
        totalFitness += minFitnessShift * population.length;
        const threshold = getRngFactory()() * totalFitness;
        let cumulative = 0;
        for (const individual of population) {
          cumulative += (individual.score ?? 0) + minFitnessShift;
          if (threshold < cumulative)
            return individual;
        }
        return population[Math.floor(getRngFactory()() * population.length)];
      case "TOURNAMENT":
        if ((selectionOptions.size || 2) > population.length) {
          if (!this._suppressTournamentError) {
            throw new Error("Tournament size must be less than population size.");
          }
          return population[Math.floor(getRngFactory()() * population.length)];
        }
        const tournamentSize = selectionOptions.size || 2;
        const tournamentParticipants = [];
        for (let i = 0; i < tournamentSize; i++) {
          tournamentParticipants.push(population[Math.floor(getRngFactory()() * population.length)]);
        }
        tournamentParticipants.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
        for (let i = 0; i < tournamentParticipants.length; i++) {
          if (getRngFactory()() < (selectionOptions.probability ?? 0.5) || i === tournamentParticipants.length - 1)
            return tournamentParticipants[i];
        }
        break;
      default:
        return population[0];
    }
    return population[0];
  }
  function getFittest() {
    const population = this.population;
    if (population[population.length - 1].score === void 0) {
      this.evaluate();
    }
    if (population[1] && (population[0].score ?? 0) < (population[1].score ?? 0)) {
      this.sort();
    }
    return population[0];
  }
  function getAverage() {
    const population = this.population;
    if (population[population.length - 1].score === void 0) {
      this.evaluate();
    }
    const totalScore = population.reduce((sum, genome) => sum + (genome.score ?? 0), 0);
    return totalScore / population.length;
  }
  var init_neat_selection = __esm({
    "dist/neat/neat.selection.js"() {
      "use strict";
    }
  });

  // dist/neat/neat.export.js
  var neat_export_exports = {};
  __export(neat_export_exports, {
    exportPopulation: () => exportPopulation,
    exportState: () => exportState,
    fromJSONImpl: () => fromJSONImpl2,
    importPopulation: () => importPopulation,
    importStateImpl: () => importStateImpl,
    toJSONImpl: () => toJSONImpl2
  });
  function exportPopulation() {
    return this.population.map((genome) => genome.toJSON());
  }
  function importPopulation(populationJSON) {
    const Network2 = (init_network(), __toCommonJS(network_exports)).default;
    this.population = populationJSON.map((serializedGenome) => Network2.fromJSON(serializedGenome));
    this.options.popsize = this.population.length;
  }
  function exportState() {
    const { toJSONImpl: toJSONImpl3, exportPopulation: exportPopulation2 } = (init_neat_export(), __toCommonJS(neat_export_exports));
    return {
      neat: toJSONImpl3.call(this),
      population: exportPopulation2.call(this)
    };
  }
  function importStateImpl(stateBundle, fitnessFunction) {
    if (!stateBundle || typeof stateBundle !== "object")
      throw new Error("Invalid state bundle");
    const neatInstance = this.fromJSON(stateBundle.neat, fitnessFunction);
    if (Array.isArray(stateBundle.population))
      neatInstance.import(stateBundle.population);
    return neatInstance;
  }
  function toJSONImpl2() {
    return {
      input: this.input,
      output: this.output,
      generation: this.generation,
      options: this.options,
      nodeSplitInnovations: Array.from(this._nodeSplitInnovations.entries()),
      connInnovations: Array.from(this._connInnovations.entries()),
      nextGlobalInnovation: this._nextGlobalInnovation
    };
  }
  function fromJSONImpl2(neatJSON, fitnessFunction) {
    const NeatClass = this;
    const neatInstance = new NeatClass(neatJSON.input, neatJSON.output, fitnessFunction, neatJSON.options || {});
    neatInstance.generation = neatJSON.generation || 0;
    if (Array.isArray(neatJSON.nodeSplitInnovations))
      neatInstance._nodeSplitInnovations = new Map(neatJSON.nodeSplitInnovations);
    if (Array.isArray(neatJSON.connInnovations))
      neatInstance._connInnovations = new Map(neatJSON.connInnovations);
    if (typeof neatJSON.nextGlobalInnovation === "number")
      neatInstance._nextGlobalInnovation = neatJSON.nextGlobalInnovation;
    return neatInstance;
  }
  var init_neat_export = __esm({
    "dist/neat/neat.export.js"() {
      "use strict";
    }
  });

  // dist/neat.js
  var neat_exports = {};
  __export(neat_exports, {
    default: () => Neat
  });
  var Neat;
  var init_neat = __esm({
    "dist/neat.js"() {
      "use strict";
      init_network();
      init_methods();
      init_selection();
      init_node();
      init_neat_mutation();
      init_neat_evolve();
      init_neat_evaluate();
      init_neat_helpers();
      init_neat_objectives();
      init_neat_diversity();
      init_neat_multiobjective();
      init_neat_compat();
      init_neat_speciation();
      init_neat_species();
      init_neat_telemetry_exports();
      init_neat_selection();
      init_neat_export();
      Neat = class _Neat {
        input;
        output;
        fitness;
        options;
        population = [];
        generation = 0;
        _rngState;
        _rng;
        _species = [];
        _operatorStats = /* @__PURE__ */ new Map();
        _nodeSplitInnovations = /* @__PURE__ */ new Map();
        _connInnovations = /* @__PURE__ */ new Map();
        _nextGlobalInnovation = 1;
        _nextGenomeId = 1;
        _lineageEnabled = false;
        _lastInbreedingCount = 0;
        _prevInbreedingCount = 0;
        _phase;
        _telemetry = [];
        _prevSpeciesMembers = /* @__PURE__ */ new Map();
        _speciesLastStats = /* @__PURE__ */ new Map();
        _speciesHistory = [];
        _paretoArchive = [];
        _paretoObjectivesArchive = [];
        _noveltyArchive = [];
        _objectiveStale = /* @__PURE__ */ new Map();
        _objectiveAges = /* @__PURE__ */ new Map();
        _objectiveEvents = [];
        _pendingObjectiveAdds = [];
        _pendingObjectiveRemoves = [];
        _lastOffspringAlloc;
        _adaptivePruneLevel;
        _lastEvalDuration;
        _lastEvolveDuration;
        _diversityStats;
        _objectivesList;
        _lastGlobalImproveGeneration = 0;
        _bestScoreLastGen;
        _speciesCreated = /* @__PURE__ */ new Map();
        _compatSpeciesEMA;
        _compatIntegral = 0;
        _lastEpsilonAdjustGen = -Infinity;
        _lastAncestorUniqAdjustGen = -Infinity;
        _mcThreshold;
        _getRNG() {
          if (!this._rng) {
            const optRng = this.options?.rng;
            if (typeof optRng === "function")
              this._rng = optRng;
            else {
              if (this._rngState === void 0) {
                let seed = (Date.now() ^ (this.population.length + 1) * 2654435761) >>> 0;
                if (seed === 0)
                  seed = 439041101;
                this._rngState = seed >>> 0;
              }
              this._rng = () => {
                let x = this._rngState >>> 0;
                x ^= x << 13;
                x >>>= 0;
                x ^= x >> 17;
                x >>>= 0;
                x ^= x << 5;
                x >>>= 0;
                this._rngState = x >>> 0;
                return (x >>> 0) / 4294967295;
              };
            }
          }
          return this._rng;
        }
        ensureMinHiddenNodes(network, multiplierOverride) {
          return ensureMinHiddenNodes.call(this, network, multiplierOverride);
        }
        constructor(input, output, fitness, options = {}) {
          this.input = input ?? 0;
          this.output = output ?? 0;
          this.fitness = fitness ?? ((n) => 0);
          this.options = options || {};
          const opts = this.options;
          if (opts.popsize === void 0)
            opts.popsize = 50;
          if (opts.elitism === void 0)
            opts.elitism = 0;
          if (opts.provenance === void 0)
            opts.provenance = 0;
          if (opts.mutationRate === void 0)
            opts.mutationRate = 0.7;
          if (opts.mutationAmount === void 0)
            opts.mutationAmount = 1;
          if (opts.fitnessPopulation === void 0)
            opts.fitnessPopulation = false;
          if (opts.clear === void 0)
            opts.clear = false;
          if (opts.equal === void 0)
            opts.equal = false;
          if (opts.compatibilityThreshold === void 0)
            opts.compatibilityThreshold = 3;
          if (opts.maxNodes === void 0)
            opts.maxNodes = Infinity;
          if (opts.maxConns === void 0)
            opts.maxConns = Infinity;
          if (opts.maxGates === void 0)
            opts.maxGates = Infinity;
          if (opts.excessCoeff === void 0)
            opts.excessCoeff = 1;
          if (opts.disjointCoeff === void 0)
            opts.disjointCoeff = 1;
          if (opts.weightDiffCoeff === void 0)
            opts.weightDiffCoeff = 0.5;
          if (opts.mutation === void 0)
            opts.mutation = mutation.ALL ? mutation.ALL.slice() : mutation.FFW ? [mutation.FFW] : [];
          if (opts.selection === void 0) {
            opts.selection = selection && selection.TOURNAMENT || selection?.TOURNAMENT || selection.FITNESS_PROPORTIONATE;
          }
          if (opts.crossover === void 0)
            opts.crossover = crossover ? crossover.SINGLE_POINT : void 0;
          if (opts.novelty === void 0)
            opts.novelty = { enabled: false };
          if (opts.diversityMetrics === void 0)
            opts.diversityMetrics = { enabled: true };
          if (opts.fastMode && opts.diversityMetrics) {
            if (opts.diversityMetrics.pairSample == null)
              opts.diversityMetrics.pairSample = 20;
            if (opts.diversityMetrics.graphletSample == null)
              opts.diversityMetrics.graphletSample = 30;
            if (opts.novelty?.enabled && opts.novelty.k == null)
              opts.novelty.k = 5;
          }
          this._noveltyArchive = [];
          if (opts.speciation === void 0)
            opts.speciation = false;
          if (opts.multiObjective && opts.multiObjective.enabled && !Array.isArray(opts.multiObjective.objectives))
            opts.multiObjective.objectives = [];
          this.population = this.population || [];
          try {
            if (this.options.network !== void 0)
              this.createPool(this.options.network);
            else if (this.options.popsize)
              this.createPool(null);
          } catch {
          }
          if (this.options.lineage?.enabled || this.options.provenance > 0)
            this._lineageEnabled = true;
          if (this.options.lineageTracking === true)
            this._lineageEnabled = true;
          if (options.lineagePressure?.enabled && this._lineageEnabled !== true) {
            this._lineageEnabled = true;
          }
        }
        async evolve() {
          return evolve.call(this);
        }
        async evaluate() {
          return evaluate.call(this);
        }
        createPool(network) {
          try {
            if (createPool && typeof createPool === "function")
              return createPool.call(this, network);
          } catch {
          }
          this.population = [];
          const poolSize = this.options.popsize || 50;
          for (let idx = 0; idx < poolSize; idx++) {
            const genomeCopy = network ? Network.fromJSON(network.toJSON()) : new Network(this.input, this.output, {
              minHidden: this.options.minHidden
            });
            genomeCopy.score = void 0;
            try {
              this.ensureNoDeadEnds(genomeCopy);
            } catch {
            }
            genomeCopy._reenableProb = this.options.reenableProb;
            genomeCopy._id = this._nextGenomeId++;
            if (this._lineageEnabled) {
              genomeCopy._parents = [];
              genomeCopy._depth = 0;
            }
            this.population.push(genomeCopy);
          }
        }
        snapshotRNGState() {
          return this._rngState;
        }
        restoreRNGState(state) {
          this._rngState = state;
          this._rng = void 0;
        }
        importRNGState(state) {
          this._rngState = state;
          this._rng = void 0;
        }
        exportRNGState() {
          return this._rngState;
        }
        getOffspring() {
          let parent1;
          let parent2;
          try {
            parent1 = this.getParent();
          } catch {
            parent1 = this.population[0];
          }
          try {
            parent2 = this.getParent();
          } catch {
            parent2 = this.population[Math.floor(this._getRNG()() * this.population.length)] || this.population[0];
          }
          const offspring = Network.crossOver(parent1, parent2, this.options.equal || false);
          offspring._reenableProb = this.options.reenableProb;
          offspring._id = this._nextGenomeId++;
          if (this._lineageEnabled) {
            offspring._parents = [
              parent1._id,
              parent2._id
            ];
            const depth1 = parent1._depth ?? 0;
            const depth2 = parent2._depth ?? 0;
            offspring._depth = 1 + Math.max(depth1, depth2);
            if (parent1._id === parent2._id)
              this._lastInbreedingCount++;
          }
          this.ensureMinHiddenNodes(offspring);
          this.ensureNoDeadEnds(offspring);
          return offspring;
        }
        _warnIfNoBestGenome() {
          try {
            console.warn("Evolution completed without finding a valid best genome (no fitness improvements recorded).");
          } catch {
          }
        }
        spawnFromParent(parent, mutateCount = 1) {
          return spawnFromParent.call(this, parent, mutateCount);
        }
        addGenome(genome, parents) {
          return addGenome.call(this, genome, parents);
        }
        selectMutationMethod(genome, rawReturnForTest = true) {
          try {
            return selectMutationMethod.call(this, genome, rawReturnForTest);
          } catch {
            return null;
          }
        }
        ensureNoDeadEnds(network) {
          try {
            return ensureNoDeadEnds.call(this, network);
          } catch {
            return;
          }
        }
        getMinimumHiddenSize(multiplierOverride) {
          const o = this.options;
          if (typeof o.minHidden === "number")
            return o.minHidden;
          const mult = multiplierOverride ?? o.minHiddenMultiplier;
          if (typeof mult === "number" && isFinite(mult)) {
            return Math.max(0, Math.round(mult * (this.input + this.output)));
          }
          return 0;
        }
        sampleRandom(count) {
          const rng = this._getRNG();
          const arr = [];
          for (let i = 0; i < count; i++)
            arr.push(rng());
          return arr;
        }
        _getObjectives() {
          return _getObjectives.call(this);
        }
        getObjectiveKeys() {
          return this._getObjectives().map((obj) => obj.key);
        }
        _invalidateGenomeCaches(genome) {
          if (!genome || typeof genome !== "object")
            return;
          delete genome._compatCache;
          delete genome._outputCache;
          delete genome._traceCache;
        }
        _computeDiversityStats() {
          this._diversityStats = computeDiversityStats2(this.population, this);
        }
        _structuralEntropy(genome) {
          return structuralEntropy2(genome);
        }
        mutate() {
          return mutate.call(this);
        }
        _mutateAddNodeReuse(genome) {
          return mutateAddNodeReuse.call(this, genome);
        }
        _mutateAddConnReuse(genome) {
          return mutateAddConnReuse.call(this, genome);
        }
        _fallbackInnov(conn) {
          return _fallbackInnov.call(this, conn);
        }
        _compatibilityDistance(netA, netB) {
          return _compatibilityDistance.call(this, netA, netB);
        }
        _speciate() {
          return _speciate.call(this);
        }
        _applyFitnessSharing() {
          return _applyFitnessSharing.call(this);
        }
        _sortSpeciesMembers(sp) {
          return _sortSpeciesMembers.call(this, sp);
        }
        _updateSpeciesStagnation() {
          return _updateSpeciesStagnation.call(this);
        }
        getSpeciesStats() {
          return getSpeciesStats.call(this);
        }
        getSpeciesHistory() {
          return getSpeciesHistory.call(this);
        }
        getNoveltyArchiveSize() {
          return this._noveltyArchive ? this._noveltyArchive.length : 0;
        }
        getMultiObjectiveMetrics() {
          return this.population.map((genome) => ({
            rank: genome._moRank ?? 0,
            crowding: genome._moCrowd ?? 0,
            score: genome.score || 0,
            nodes: genome.nodes.length,
            connections: genome.connections.length
          }));
        }
        getOperatorStats() {
          return Array.from(this._operatorStats.entries()).map(([operatorName, stats]) => ({
            name: operatorName,
            success: stats.success,
            attempts: stats.attempts
          }));
        }
        applyEvolutionPruning() {
          try {
            (init_neat_pruning(), __toCommonJS(neat_pruning_exports)).applyEvolutionPruning.call(this);
          } catch {
          }
        }
        applyAdaptivePruning() {
          try {
            (init_neat_pruning(), __toCommonJS(neat_pruning_exports)).applyAdaptivePruning.call(this);
          } catch {
          }
        }
        getTelemetry() {
          return this._telemetry;
        }
        exportTelemetryJSONL() {
          return exportTelemetryJSONL.call(this);
        }
        exportTelemetryCSV(maxEntries = 500) {
          return exportTelemetryCSV.call(this, maxEntries);
        }
        clearTelemetry() {
          this._telemetry = [];
        }
        getObjectives() {
          return this._getObjectives().map((o) => ({
            key: o.key,
            direction: o.direction
          }));
        }
        getObjectiveEvents() {
          return this._objectiveEvents.slice();
        }
        getLineageSnapshot(limit = 20) {
          return this.population.slice(0, limit).map((genome) => ({
            id: genome._id ?? -1,
            parents: Array.isArray(genome._parents) ? genome._parents.slice() : []
          }));
        }
        exportSpeciesHistoryCSV(maxEntries = 200) {
          return exportSpeciesHistoryCSV.call(this, maxEntries);
        }
        getParetoFronts(maxFronts = 3) {
          if (!this.options.multiObjective?.enabled)
            return [[...this.population]];
          const fronts = [];
          for (let frontIdx = 0; frontIdx < maxFronts; frontIdx++) {
            const front = this.population.filter((genome) => (genome._moRank ?? 0) === frontIdx);
            if (!front.length)
              break;
            fronts.push(front);
          }
          return fronts;
        }
        getDiversityStats() {
          return this._diversityStats;
        }
        registerObjective(key, direction, accessor) {
          return registerObjective.call(this, key, direction, accessor);
        }
        clearObjectives() {
          return clearObjectives.call(this);
        }
        getParetoArchive(maxEntries = 50) {
          return this._paretoArchive.slice(-maxEntries);
        }
        exportParetoFrontJSONL(maxEntries = 100) {
          const slice = this._paretoObjectivesArchive.slice(-maxEntries);
          return slice.map((e) => JSON.stringify(e)).join("\n");
        }
        getPerformanceStats() {
          return {
            lastEvalMs: this._lastEvalDuration,
            lastEvolveMs: this._lastEvolveDuration
          };
        }
        exportSpeciesHistoryJSONL(maxEntries = 200) {
          const slice = this._speciesHistory.slice(-maxEntries);
          return slice.map((e) => JSON.stringify(e)).join("\n");
        }
        resetNoveltyArchive() {
          this._noveltyArchive = [];
        }
        clearParetoArchive() {
          this._paretoArchive = [];
        }
        sort() {
          return sort.call(this);
        }
        getParent() {
          return getParent.call(this);
        }
        getFittest() {
          return getFittest.call(this);
        }
        getAverage() {
          return getAverage.call(this);
        }
        export() {
          return exportPopulation.call(this);
        }
        import(json) {
          return importPopulation.call(this, json);
        }
        exportState() {
          return exportState.call(this);
        }
        static importState(bundle, fitness) {
          return importStateImpl.call(_Neat, bundle, fitness);
        }
        toJSON() {
          return toJSONImpl2.call(this);
        }
        static fromJSON(json, fitness) {
          return fromJSONImpl2.call(_Neat, json, fitness);
        }
      };
    }
  });

  // dist/neataptic.js
  init_neat();
  init_network();
  init_node();
  init_layer();
  init_group();
  init_connection();

  // dist/architecture/architect.js
  init_node();
  init_layer();
  init_group();
  init_network();
  init_methods();
  init_connection();

  // dist/neataptic.js
  init_methods();
  init_config();
  init_multi();

  // bench-browser/bench-entry.ts
  function buildSynthetic(size) {
    const t0 = performance.now();
    const inputs = Math.max(1, Math.floor(Math.sqrt(size)));
    const outputs = Math.max(1, Math.ceil(size / inputs));
    const net = new Network(inputs, outputs);
    while (net.connections.length > size) {
      const idx = Math.floor(Math.random() * net.connections.length);
      const c = net.connections[idx];
      net.disconnect(c.from, c.to);
    }
    const t1 = performance.now();
    return { net, buildMs: t1 - t0 };
  }
  function measureForward(net, iterations) {
    const vec = new Array(net.input).fill(0).map(() => Math.random());
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) net.activate(vec);
    const t1 = performance.now();
    const totalMs = t1 - t0;
    return { totalMs, avgMs: totalMs / iterations };
  }
  function run() {
    const sizes = [1e3, 1e4, 5e4, 1e5];
    const out = [];
    for (const size of sizes) {
      const { net, buildMs } = buildSynthetic(size);
      const iterations = size >= 1e5 ? 2 : size >= 5e4 ? 3 : 5;
      const { totalMs, avgMs } = measureForward(net, iterations);
      out.push({
        size,
        buildMs: Number(buildMs.toFixed(3)),
        fwdAvgMs: Number(avgMs.toFixed(4)),
        fwdTotalMs: Number(totalMs.toFixed(3)),
        conn: net.connections.length,
        nodes: net.nodes.length,
        iterations
      });
    }
    return out;
  }
  window.__NEATAPTIC_BENCH__ = {
    mode: window.__BENCH_MODE__ || "__UNDEF__",
    generatedAt: (/* @__PURE__ */ new Date()).toISOString(),
    results: run()
  };
  console.log("[NEATAPTIC_BROWSER_BENCH] ready");
})();
