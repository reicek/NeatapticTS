"use strict";
(() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
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
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    // If the importer is in node compatibility mode or this is not an ESM
    // file that has been converted to a CommonJS file using a Babel-
    // compatible transform (i.e. "__esModule" has not been set), then set
    // "default" to the CommonJS "module.exports" for node compatibility.
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/architecture/connection.ts
  var kGain, kGater, kOpt, kPlasticRate, Connection;
  var init_connection = __esm({
    "src/architecture/connection.ts"() {
      "use strict";
      init_node();
      kGain = Symbol("connGain");
      kGater = Symbol("connGater");
      kOpt = Symbol("connOptMoments");
      kPlasticRate = Symbol("connPlasticRate");
      Connection = class _Connection {
        /** The source (pre-synaptic) node supplying activation. */
        from;
        /** The target (post-synaptic) node receiving activation. */
        to;
        /** Scalar multiplier applied to the source activation (prior to gain modulation). */
        weight;
        /** Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment). */
        eligibility;
        /** Last applied delta weight (used by classic momentum). */
        previousDeltaWeight;
        /** Accumulated (batched) delta weight awaiting an apply step. */
        totalDeltaWeight;
        /** Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration. */
        xtrace;
        /** Unique historical marking (auto-increment) for evolutionary alignment. */
        innovation;
        // enabled handled via bitfield (see _flags) exposed through accessor (enumerability removed for slimming)
        // --- Optimizer moment states (virtualized via symbol-backed bag + accessors) ---
        // NOTE: Accessor implementations below manage a lazily-created non-enumerable object containing:
        // { firstMoment, secondMoment, gradientAccumulator, maxSecondMoment, infinityNorm, secondMomentum, lookaheadShadowWeight }
        /**
         * Packed state flags (private for future-proofing hidden class):
         * bit0 => enabled gene expression (1 = active)
         * bit1 => DropConnect active mask (1 = not dropped this forward pass)
         * bit2 => hasGater (1 = symbol field present)
         * bit3 => plastic (plasticityRate > 0)
         * bits4+ reserved.
         */
        _flags;
        // bit0 enabled, bit1 dcActive, bit2 hasGater, bit3 plastic
        /**
         * Construct a new connection between two nodes.
         *
         * @param from Source node.
         * @param to Target node.
         * @param weight Optional initial weight (default: small random in [-0.1, 0.1]).
         *
         * @example
         * const link = new Connection(nodeA, nodeB, 0.42);
         * link.enabled = false;     // disable during mutation
         * link.enabled = true;      // re-enable later
         */
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
        /**
         * Serialize to a minimal JSON-friendly shape (used for saving genomes / networks).
         * Undefined indices are preserved as `undefined` to allow later resolution / remapping.
         *
         * @returns Object with node indices, weight, gain, gater index (if any), innovation id & enabled flag.
         * @example
         * const json = connection.toJSON();
         * // => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }
         */
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
            if (g && typeof g.index !== "undefined") json.gater = g.index;
          }
          return json;
        }
        /**
         * Deterministic Cantor pairing function for a (sourceNodeId, targetNodeId) pair.
         * Useful when you want a stable innovation id without relying on global mutable counters
         * (e.g., for hashing or reproducible experiments).
         *
         * NOTE: For large indices this can overflow 53-bit safe integer space; keep node indices reasonable.
         *
         * @param sourceNodeId Source node integer id / index.
         * @param targetNodeId Target node integer id / index.
         * @returns Unique non-negative integer derived from the ordered pair.
         * @see https://en.wikipedia.org/wiki/Pairing_function
         * @example
         * const id = Connection.innovationID(2, 5); // deterministic
         */
        static innovationID(sourceNodeId, targetNodeId) {
          return 0.5 * (sourceNodeId + targetNodeId) * (sourceNodeId + targetNodeId + 1) + targetNodeId;
        }
        static _nextInnovation = 1;
        /**
         * Reset the monotonic auto-increment innovation counter (used for newly constructed / pooled instances).
         * You normally only call this at the start of an experiment or when deserializing a full population.
         *
         * @param value New starting value (default 1).
         * @example
         * Connection.resetInnovationCounter();     // back to 1
         * Connection.resetInnovationCounter(1000); // start counting from 1000
         */
        static resetInnovationCounter(value = 1) {
          _Connection._nextInnovation = value;
        }
        // --- Simple object pool to reduce GC churn when connections are frequently created/removed ---
        static _pool = [];
        /**
         * Acquire a `Connection` from the pool (or construct new). Fields are fully reset & given
         * a fresh sequential `innovation` id. Prefer this in evolutionary algorithms that mutate
         * topology frequently to reduce GC pressure.
         *
         * @param from Source node.
         * @param to Target node.
         * @param weight Optional initial weight.
         * @returns Reinitialized connection instance.
         * @example
         * const conn = Connection.acquire(a, b);
         * // ... use conn ...
         * Connection.release(conn); // when permanently removed
         */
        static acquire(from, to, weight) {
          let c;
          if (_Connection._pool.length) {
            c = _Connection._pool.pop();
            c.from = from;
            c.to = to;
            c.weight = weight ?? Math.random() * 0.2 - 0.1;
            if (c[kGain] !== void 0) delete c[kGain];
            if (c[kGater] !== void 0) delete c[kGater];
            c._flags = 3;
            c.eligibility = 0;
            c.previousDeltaWeight = 0;
            c.totalDeltaWeight = 0;
            c.xtrace.nodes.length = 0;
            c.xtrace.values.length = 0;
            if (c[kOpt]) delete c[kOpt];
            c.innovation = _Connection._nextInnovation++;
          } else c = new _Connection(from, to, weight);
          return c;
        }
        /**
         * Return a `Connection` to the internal pool for later reuse. Do NOT use the instance again
         * afterward unless re-acquired (treat as surrendered). Optimizer / trace fields are not
         * scrubbed here (they're overwritten during `acquire`).
         *
         * @param conn The connection instance to recycle.
         */
        static release(conn) {
          _Connection._pool.push(conn);
        }
        /** Whether the gene (connection) is currently expressed (participates in forward pass). */
        get enabled() {
          return (this._flags & 1) !== 0;
        }
        set enabled(v) {
          this._flags = v ? this._flags | 1 : this._flags & ~1;
        }
        /** DropConnect active mask: 1 = not dropped (active), 0 = dropped for this stochastic pass. */
        get dcMask() {
          return (this._flags & 2) !== 0 ? 1 : 0;
        }
        set dcMask(v) {
          this._flags = v ? this._flags | 2 : this._flags & ~2;
        }
        /** Whether a gater node is assigned (modulates gain); true if the gater symbol field is present. */
        get hasGater() {
          return (this._flags & 4) !== 0;
        }
        /** Whether this connection participates in plastic adaptation (rate > 0). */
        get plastic() {
          return (this._flags & 8) !== 0;
        }
        set plastic(v) {
          if (v) this._flags |= 8;
          else this._flags &= ~8;
          if (!v && this[kPlasticRate] !== void 0)
            delete this[kPlasticRate];
        }
        // --- Virtualized gain property ---
        /**
         * Multiplicative modulation applied *after* weight. Default is `1` (neutral). We only store an
         * internal symbol-keyed property when the gain is non-neutral, reducing memory usage across
         * large populations where most connections are ungated.
         */
        get gain() {
          return this[kGain] === void 0 ? 1 : this[kGain];
        }
        set gain(v) {
          if (v === 1) {
            if (this[kGain] !== void 0) delete this[kGain];
          } else {
            this[kGain] = v;
          }
        }
        // --- Optimizer field accessors (prototype-level to avoid per-instance enumerable keys) ---
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
            if (bag) delete bag[k];
          } else {
            this._ensureOptBag()[k] = v;
          }
        }
        /** First moment estimate (Adam / AdamW) (was opt_m). */
        get firstMoment() {
          return this._getOpt("firstMoment");
        }
        set firstMoment(v) {
          this._setOpt("firstMoment", v);
        }
        /** Second raw moment estimate (Adam family) (was opt_v). */
        get secondMoment() {
          return this._getOpt("secondMoment");
        }
        set secondMoment(v) {
          this._setOpt("secondMoment", v);
        }
        /** Generic gradient accumulator (RMSProp / AdaGrad) (was opt_cache). */
        get gradientAccumulator() {
          return this._getOpt("gradientAccumulator");
        }
        set gradientAccumulator(v) {
          this._setOpt("gradientAccumulator", v);
        }
        /** AMSGrad: Maximum of past second moment (was opt_vhat). */
        get maxSecondMoment() {
          return this._getOpt("maxSecondMoment");
        }
        set maxSecondMoment(v) {
          this._setOpt("maxSecondMoment", v);
        }
        /** Adamax: Exponential moving infinity norm (was opt_u). */
        get infinityNorm() {
          return this._getOpt("infinityNorm");
        }
        set infinityNorm(v) {
          this._setOpt("infinityNorm", v);
        }
        /** Secondary momentum (Lion variant) (was opt_m2). */
        get secondMomentum() {
          return this._getOpt("secondMomentum");
        }
        set secondMomentum(v) {
          this._setOpt("secondMomentum", v);
        }
        /** Lookahead: shadow (slow) weight parameter (was _la_shadowWeight). */
        get lookaheadShadowWeight() {
          return this._getOpt("lookaheadShadowWeight");
        }
        set lookaheadShadowWeight(v) {
          this._setOpt("lookaheadShadowWeight", v);
        }
        // --- Virtualized gater property (non-enumerable) ---
        /** Optional gating node whose activation can modulate effective weight (symbol-backed). */
        get gater() {
          return (this._flags & 4) !== 0 ? this[kGater] : null;
        }
        set gater(node) {
          if (node === null) {
            if ((this._flags & 4) !== 0) {
              this._flags &= ~4;
              if (this[kGater] !== void 0) delete this[kGater];
            }
          } else {
            this[kGater] = node;
            this._flags |= 4;
          }
        }
        // --- Plasticity rate (virtualized) ---
        /** Per-connection plasticity / learning rate (0 means non-plastic). Setting >0 marks plastic flag. */
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
        // ---------------------------------------------------------------------------
        // Backward compatibility accessors for previously abbreviated property names
        // (opt_m, opt_v, opt_cache, opt_vhat, opt_u, opt_m2, _la_shadowWeight)
        // These keep external code & tests functioning while encouraging clearer names.
        // ---------------------------------------------------------------------------
        /** @deprecated Use firstMoment instead. */
        get opt_m() {
          return this.firstMoment;
        }
        set opt_m(v) {
          this.firstMoment = v;
        }
        /** @deprecated Use secondMoment instead. */
        get opt_v() {
          return this.secondMoment;
        }
        set opt_v(v) {
          this.secondMoment = v;
        }
        /** @deprecated Use gradientAccumulator instead. */
        get opt_cache() {
          return this.gradientAccumulator;
        }
        set opt_cache(v) {
          this.gradientAccumulator = v;
        }
        /** @deprecated Use maxSecondMoment instead. */
        get opt_vhat() {
          return this.maxSecondMoment;
        }
        set opt_vhat(v) {
          this.maxSecondMoment = v;
        }
        /** @deprecated Use infinityNorm instead. */
        get opt_u() {
          return this.infinityNorm;
        }
        set opt_u(v) {
          this.infinityNorm = v;
        }
        /** @deprecated Use secondMomentum instead. */
        get opt_m2() {
          return this.secondMomentum;
        }
        set opt_m2(v) {
          this.secondMomentum = v;
        }
        /** @deprecated Use lookaheadShadowWeight instead. */
        get _la_shadowWeight() {
          return this.lookaheadShadowWeight;
        }
        set _la_shadowWeight(v) {
          this.lookaheadShadowWeight = v;
        }
        /** Convenience alias for DropConnect mask with clearer naming. */
        get dropConnectActiveMask() {
          return this.dcMask;
        }
        set dropConnectActiveMask(v) {
          this.dcMask = v;
        }
      };
    }
  });

  // src/config.ts
  var config;
  var init_config = __esm({
    "src/config.ts"() {
      "use strict";
      config = {
        warnings: false,
        // emit runtime guidance
        float32Mode: false,
        // numeric precision mode
        deterministicChainMode: false,
        // deep path test flag (ADD_NODE determinism)
        enableGatingTraces: true,
        // advanced gating trace infra
        enableNodePooling: false,
        // experimental node instance pooling
        enableSlabArrayPooling: false
        // experimental slab typed array pooling
        // slabPoolMaxPerKey: 4,        // optional override for per-key slab retention cap (default internal 4)
        // browserSlabChunkTargetMs: 3, // example: aim for ~3ms per async slab slice in Browser
        // poolMaxPerBucket: 256,     // example memory cap override
        // poolPrewarmCount: 2,       // example prewarm override
      };
    }
  });

  // src/neat/neat.constants.ts
  var neat_constants_exports = {};
  __export(neat_constants_exports, {
    EPSILON: () => EPSILON,
    EXTRA_CONNECTION_PROBABILITY: () => EXTRA_CONNECTION_PROBABILITY,
    NORM_EPSILON: () => NORM_EPSILON,
    PROB_EPSILON: () => PROB_EPSILON
  });
  var EPSILON, PROB_EPSILON, NORM_EPSILON, EXTRA_CONNECTION_PROBABILITY;
  var init_neat_constants = __esm({
    "src/neat/neat.constants.ts"() {
      "use strict";
      EPSILON = 1e-9;
      PROB_EPSILON = 1e-15;
      NORM_EPSILON = 1e-5;
      EXTRA_CONNECTION_PROBABILITY = 0.5;
    }
  });

  // src/methods/cost.ts
  var Cost;
  var init_cost = __esm({
    "src/methods/cost.ts"() {
      "use strict";
      init_neat_constants();
      Cost = class {
        /**
         * Calculates the Cross Entropy error, commonly used for classification tasks.
         *
         * This function measures the performance of a classification model whose output is
         * a probability value between 0 and 1. Cross-entropy loss increases as the
         * predicted probability diverges from the actual label.
         *
         * It uses a small epsilon (PROB_EPSILON = 1e-15) to prevent `log(0)` which would result in `NaN`.
         * Output values are clamped to the range `[epsilon, 1 - epsilon]` for numerical stability.
         *
         * @see {@link https://en.wikipedia.org/wiki/Cross_entropy}
         * @param {number[]} targets - An array of target values, typically 0 or 1 for binary classification, or probabilities for soft labels.
         * @param {number[]} outputs - An array of output values from the network, representing probabilities (expected to be between 0 and 1).
         * @returns {number} The mean cross-entropy error over all samples.
         * @throws {Error} If the target and output arrays have different lengths.
         */
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
        /**
         * Softmax Cross Entropy for mutually exclusive multi-class outputs given raw (pre-softmax or arbitrary) scores.
         * Applies a numerically stable softmax to the outputs internally then computes -sum(target * log(prob)).
         * Targets may be soft labels and are expected to sum to 1 (will be re-normalized if not).
         */
        static softmaxCrossEntropy(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          const n = outputs.length;
          let tSum = 0;
          for (const t of targets) tSum += t;
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
        /**
         * Calculates the Mean Squared Error (MSE), a common loss function for regression tasks.
         *
         * MSE measures the average of the squares of the errors—that is, the average
         * squared difference between the estimated values and the actual value.
         * It is sensitive to outliers due to the squaring of the error terms.
         *
         * @see {@link https://en.wikipedia.org/wiki/Mean_squared_error}
         * @param {number[]} targets - An array of target numerical values.
         * @param {number[]} outputs - An array of output values from the network.
         * @returns {number} The mean squared error.
         * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
         */
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
        /**
         * Calculates the Binary Error rate, often used as a simple accuracy metric for classification.
         *
         * This function calculates the proportion of misclassifications by comparing the
         * rounded network outputs (thresholded at 0.5) against the target labels.
         * It assumes target values are 0 or 1, and outputs are probabilities between 0 and 1.
         * Note: This is equivalent to `1 - accuracy` for binary classification.
         *
         * @param {number[]} targets - An array of target values, expected to be 0 or 1.
         * @param {number[]} outputs - An array of output values from the network, typically probabilities between 0 and 1.
         * @returns {number} The proportion of misclassified samples (error rate, between 0 and 1).
         * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
         */
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
        /**
         * Calculates the Mean Absolute Error (MAE), another common loss function for regression tasks.
         *
         * MAE measures the average of the absolute differences between predictions and actual values.
         * Compared to MSE, it is less sensitive to outliers because errors are not squared.
         *
         * @see {@link https://en.wikipedia.org/wiki/Mean_absolute_error}
         * @param {number[]} targets - An array of target numerical values.
         * @param {number[]} outputs - An array of output values from the network.
         * @returns {number} The mean absolute error.
         * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
         */
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
        /**
         * Calculates the Mean Absolute Percentage Error (MAPE).
         *
         * MAPE expresses the error as a percentage of the actual value. It can be useful
         * for understanding the error relative to the magnitude of the target values.
         * However, it has limitations: it's undefined when the target value is zero and
         * can be skewed by target values close to zero.
         *
         * @see {@link https://en.wikipedia.org/wiki/Mean_absolute_percentage_error}
         * @param {number[]} targets - An array of target numerical values. Should not contain zeros for standard MAPE.
         * @param {number[]} outputs - An array of output values from the network.
         * @returns {number} The mean absolute percentage error, expressed as a proportion (e.g., 0.1 for 10%).
         * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
         */
        static mape(targets, outputs) {
          if (targets.length !== outputs.length) {
            throw new Error("Target and output arrays must have the same length.");
          }
          let error = 0;
          const epsilon = PROB_EPSILON;
          outputs.forEach((output, outputIndex) => {
            const target = targets[outputIndex];
            error += Math.abs(
              (target - output) / Math.max(Math.abs(target), epsilon)
            );
          });
          return error / outputs.length;
        }
        /**
         * Calculates the Mean Squared Logarithmic Error (MSLE).
         *
         * MSLE is often used in regression tasks where the target values span a large range
         * or when penalizing under-predictions more than over-predictions is desired.
         * It measures the squared difference between the logarithms of the predicted and actual values.
         * Uses `log(1 + x)` instead of `log(x)` for numerical stability and to handle inputs of 0.
         * Assumes both targets and outputs are non-negative.
         *
         * @see {@link https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error}
         * @param {number[]} targets - An array of target numerical values (assumed >= 0).
         * @param {number[]} outputs - An array of output values from the network (assumed >= 0).
         * @returns {number} The mean squared logarithmic error.
         * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
         */
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
        /**
         * Calculates the Mean Hinge loss, primarily used for "maximum-margin" classification,
         * most notably for Support Vector Machines (SVMs).
         *
         * Hinge loss is used for training classifiers. It penalizes predictions that are
         * not only incorrect but also those that are correct but not confident (i.e., close to the decision boundary).
         * Assumes target values are encoded as -1 or 1.
         *
         * @see {@link https://en.wikipedia.org/wiki/Hinge_loss}
         * @param {number[]} targets - An array of target values, expected to be -1 or 1.
         * @param {number[]} outputs - An array of output values from the network (raw scores, not necessarily probabilities).
         * @returns {number} The mean hinge loss.
         * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
         */
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
        /**
         * Calculates the Focal Loss, which is useful for addressing class imbalance in classification tasks.
         * Focal loss down-weights easy examples and focuses training on hard negatives.
         *
         * @see https://arxiv.org/abs/1708.02002
         * @param {number[]} targets - Array of target values (0 or 1 for binary, or probabilities for soft labels).
         * @param {number[]} outputs - Array of predicted probabilities (between 0 and 1).
         * @param {number} gamma - Focusing parameter (default 2).
         * @param {number} alpha - Balancing parameter (default 0.25).
         * @returns {number} The mean focal loss.
         */
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
        /**
         * Calculates the Cross Entropy with Label Smoothing.
         * Label smoothing prevents the model from becoming overconfident by softening the targets.
         *
         * @see https://arxiv.org/abs/1512.00567
         * @param {number[]} targets - Array of target values (0 or 1 for binary, or probabilities for soft labels).
         * @param {number[]} outputs - Array of predicted probabilities (between 0 and 1).
         * @param {number} smoothing - Smoothing factor (between 0 and 1, e.g., 0.1).
         * @returns {number} The mean cross-entropy loss with label smoothing.
         */
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

  // src/methods/rate.ts
  var Rate;
  var init_rate = __esm({
    "src/methods/rate.ts"() {
      "use strict";
      Rate = class {
        /**
         * Implements a fixed learning rate schedule.
         *
         * The learning rate remains constant throughout the entire training process.
         * This is the simplest schedule and serves as a baseline, but may not be
         * optimal for complex problems.
         *
         * @returns A function that takes the base learning rate and the current iteration number, and always returns the base learning rate.
         * @param baseRate The initial learning rate, which will remain constant.
         * @param iteration The current training iteration (unused in this method, but included for consistency).
         */
        static fixed() {
          const func = (baseRate, iteration) => {
            return baseRate;
          };
          return func;
        }
        /**
         * Implements a step decay learning rate schedule.
         *
         * The learning rate is reduced by a multiplicative factor (`gamma`)
         * at predefined intervals (`stepSize` iterations). This allows for
         * faster initial learning, followed by finer adjustments as training progresses.
         *
         * Formula: `learning_rate = baseRate * gamma ^ floor(iteration / stepSize)`
         *
         * @param gamma The factor by which the learning rate is multiplied at each step. Should be less than 1. Defaults to 0.9.
         * @param stepSize The number of iterations after which the learning rate decays. Defaults to 100.
         * @returns A function that calculates the decayed learning rate for a given iteration.
         * @param baseRate The initial learning rate.
         * @param iteration The current training iteration.
         */
        static step(gamma = 0.9, stepSize = 100) {
          const func = (baseRate, iteration) => {
            return Math.max(
              0,
              baseRate * Math.pow(gamma, Math.floor(iteration / stepSize))
            );
          };
          return func;
        }
        /**
         * Implements an exponential decay learning rate schedule.
         *
         * The learning rate decreases exponentially after each iteration, multiplying
         * by the decay factor `gamma`. This provides a smooth, continuous reduction
         * in the learning rate over time.
         *
         * Formula: `learning_rate = baseRate * gamma ^ iteration`
         *
         * @param gamma The decay factor applied at each iteration. Should be less than 1. Defaults to 0.999.
         * @returns A function that calculates the exponentially decayed learning rate for a given iteration.
         * @param baseRate The initial learning rate.
         * @param iteration The current training iteration.
         */
        static exp(gamma = 0.999) {
          const func = (baseRate, iteration) => {
            return baseRate * Math.pow(gamma, iteration);
          };
          return func;
        }
        /**
         * Implements an inverse decay learning rate schedule.
         *
         * The learning rate decreases as the inverse of the iteration number,
         * controlled by the decay factor `gamma` and exponent `power`. The rate
         * decreases more slowly over time compared to exponential decay.
         *
         * Formula: `learning_rate = baseRate / (1 + gamma * Math.pow(iteration, power))`
         *
         * @param gamma Controls the rate of decay. Higher values lead to faster decay. Defaults to 0.001.
         * @param power The exponent controlling the shape of the decay curve. Defaults to 2.
         * @returns A function that calculates the inversely decayed learning rate for a given iteration.
         * @param baseRate The initial learning rate.
         * @param iteration The current training iteration.
         */
        static inv(gamma = 1e-3, power = 2) {
          const func = (baseRate, iteration) => {
            return baseRate / (1 + gamma * Math.pow(iteration, power));
          };
          return func;
        }
        /**
         * Implements a Cosine Annealing learning rate schedule.
         *
         * This schedule varies the learning rate cyclically according to a cosine function.
         * It starts at the `baseRate` and smoothly anneals down to `minRate` over a
         * specified `period` of iterations, then potentially repeats. This can help
         * the model escape local minima and explore the loss landscape more effectively.
         * Often used with "warm restarts" where the cycle repeats.
         *
         * Formula: `learning_rate = minRate + 0.5 * (baseRate - minRate) * (1 + cos(pi * current_cycle_iteration / period))`
         *
         * @param period The number of iterations over which the learning rate anneals from `baseRate` to `minRate` in one cycle. Defaults to 1000.
         * @param minRate The minimum learning rate value at the end of a cycle. Defaults to 0.
         * @returns A function that calculates the learning rate for a given iteration based on the cosine annealing schedule.
         * @param baseRate The initial (maximum) learning rate for the cycle.
         * @param iteration The current training iteration.
         * @see {@link https://arxiv.org/abs/1608.03983 SGDR: Stochastic Gradient Descent with Warm Restarts} - The paper introducing this technique.
         */
        static cosineAnnealing(period = 1e3, minRate = 0) {
          const func = (baseRate, iteration) => {
            const currentCycleIteration = iteration % period;
            const cosineDecay = 0.5 * (1 + Math.cos(currentCycleIteration / period * Math.PI));
            return minRate + (baseRate - minRate) * cosineDecay;
          };
          return func;
        }
        /**
         * Cosine Annealing with Warm Restarts (SGDR style) where the cycle length can grow by a multiplier (tMult) after each restart.
         *
         * @param initialPeriod Length of the first cycle in iterations.
         * @param minRate Minimum learning rate at valley.
         * @param tMult Factor to multiply the period after each restart (>=1).
         */
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
        /**
         * Linear Warmup followed by Linear Decay to an end rate.
         * Warmup linearly increases LR from near 0 up to baseRate over warmupSteps, then linearly decays to endRate at totalSteps.
         * Iterations beyond totalSteps clamp to endRate.
         *
         * @param totalSteps Total steps for full schedule (must be > 0).
         * @param warmupSteps Steps for warmup (< totalSteps). Defaults to 10% of totalSteps.
         * @param endRate Final rate at totalSteps.
         */
        static linearWarmupDecay(totalSteps, warmupSteps, endRate = 0) {
          if (totalSteps <= 0) throw new Error("totalSteps must be > 0");
          const warm = Math.min(
            warmupSteps ?? Math.max(1, Math.floor(totalSteps * 0.1)),
            totalSteps - 1
          );
          return (baseRate, iteration) => {
            if (iteration <= warm) {
              return baseRate * (iteration / Math.max(1, warm));
            }
            if (iteration >= totalSteps) return endRate;
            const decaySteps = totalSteps - warm;
            const progress = (iteration - warm) / decaySteps;
            return endRate + (baseRate - endRate) * (1 - progress);
          };
        }
        /**
         * ReduceLROnPlateau style scheduler (stateful closure) that monitors error signal (third argument if provided)
         * and reduces rate by 'factor' if no improvement beyond 'minDelta' for 'patience' iterations.
         * Cooldown prevents immediate successive reductions.
         * NOTE: Requires the training loop to call with signature (baseRate, iteration, lastError).
         */
        static reduceOnPlateau(options) {
          const {
            factor = 0.5,
            patience = 10,
            minDelta = 1e-4,
            cooldown = 0,
            minRate = 0,
            verbose = false
          } = options || {};
          let currentRate;
          let bestError;
          let lastImprovementIter = 0;
          let cooldownUntil = -1;
          return (baseRate, iteration, lastError) => {
            if (currentRate === void 0) currentRate = baseRate;
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

  // src/methods/activation.ts
  var Activation, activation_default;
  var init_activation = __esm({
    "src/methods/activation.ts"() {
      "use strict";
      Activation = {
        /**
         * Logistic (Sigmoid) activation function.
         * Outputs values between 0 and 1. Commonly used in older network architectures
         * and for output layers in binary classification tasks.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the logistic function or its derivative.
         */
        logistic: (x, derivate = false) => {
          const fx = 1 / (1 + Math.exp(-x));
          return !derivate ? fx : fx * (1 - fx);
        },
        /**
         * Alias for Logistic (Sigmoid) activation function.
         * Outputs values between 0 and 1. Commonly used in older network architectures
         * and for output layers in binary classification tasks.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the logistic function or its derivative.
         */
        sigmoid: (x, derivate = false) => {
          const fx = 1 / (1 + Math.exp(-x));
          return !derivate ? fx : fx * (1 - fx);
        },
        /**
         * Hyperbolic tangent (tanh) activation function.
         * Outputs values between -1 and 1. Often preferred over logistic sigmoid in hidden layers
         * due to its zero-centered output, which can help with training convergence.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the tanh function or its derivative.
         */
        tanh: (x, derivate = false) => {
          return derivate ? 1 - Math.pow(Math.tanh(x), 2) : Math.tanh(x);
        },
        /**
         * Identity activation function (Linear).
         * Outputs the input value directly: f(x) = x.
         * Used when no non-linearity is desired, e.g., in output layers for regression tasks.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the identity function (x) or its derivative (1).
         */
        identity: (x, derivate = false) => {
          return derivate ? 1 : x;
        },
        /**
         * Step activation function (Binary Step).
         * Outputs 0 if the input is negative or zero, and 1 if the input is positive.
         * Rarely used in modern deep learning due to its zero derivative almost everywhere,
         * hindering gradient-based learning.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the step function (0 or 1) or its derivative (0).
         */
        step: (x, derivate = false) => {
          return derivate ? 0 : x > 0 ? 1 : 0;
        },
        /**
         * Rectified Linear Unit (ReLU) activation function.
         * Outputs the input if it's positive, and 0 otherwise: f(x) = max(0, x).
         * Widely used in deep learning due to its simplicity, computational efficiency,
         * and ability to mitigate the vanishing gradient problem.
         *
         * Note: The derivative at x=0 is ambiguous (theoretically undefined). Here, we return 0,
         * which is a common practical choice. If you need a different behavior, consider using a custom activation.
         *
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the ReLU function or its derivative (0 or 1).
         */
        relu: (x, derivate = false) => {
          return derivate ? x > 0 ? 1 : 0 : x > 0 ? x : 0;
        },
        /**
         * Softsign activation function.
         * A smooth approximation of the sign function: f(x) = x / (1 + |x|).
         * Outputs values between -1 and 1.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the softsign function or its derivative.
         */
        softsign: (x, derivate = false) => {
          const d = 1 + Math.abs(x);
          return derivate ? 1 / Math.pow(d, 2) : x / d;
        },
        /**
         * Sinusoid activation function.
         * Uses the standard sine function: f(x) = sin(x).
         * Can be useful for tasks involving periodic patterns.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the sinusoid function or its derivative (cos(x)).
         */
        sinusoid: (x, derivate = false) => {
          return derivate ? Math.cos(x) : Math.sin(x);
        },
        /**
         * Gaussian activation function.
         * Uses the Gaussian (bell curve) function: f(x) = exp(-x^2).
         * Outputs values between 0 and 1. Sometimes used in radial basis function (RBF) networks.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the Gaussian function or its derivative.
         */
        gaussian: (x, derivate = false) => {
          const d = Math.exp(-Math.pow(x, 2));
          return derivate ? -2 * x * d : d;
        },
        /**
         * Bent Identity activation function.
         * A function that behaves linearly for large positive inputs but non-linearly near zero:
         * f(x) = (sqrt(x^2 + 1) - 1) / 2 + x.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the bent identity function or its derivative.
         */
        bentIdentity: (x, derivate = false) => {
          const d = Math.sqrt(Math.pow(x, 2) + 1);
          return derivate ? x / (2 * d) + 1 : (d - 1) / 2 + x;
        },
        /**
         * Bipolar activation function (Sign function).
         * Outputs -1 if the input is negative or zero, and 1 if the input is positive.
         * Similar to the Step function but with outputs -1 and 1.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the bipolar function (-1 or 1) or its derivative (0).
         */
        bipolar: (x, derivate = false) => {
          return derivate ? 0 : x > 0 ? 1 : -1;
        },
        /**
         * Bipolar Sigmoid activation function.
         * A scaled and shifted version of the logistic sigmoid, outputting values between -1 and 1:
         * f(x) = 2 * logistic(x) - 1 = (1 - exp(-x)) / (1 + exp(-x)).
         * This is equivalent to the hyperbolic tangent (tanh) function.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the bipolar sigmoid function or its derivative.
         * @see {@link Activation.tanh}
         */
        bipolarSigmoid: (x, derivate = false) => {
          const d = 2 / (1 + Math.exp(-x)) - 1;
          return derivate ? 1 / 2 * (1 + d) * (1 - d) : d;
        },
        /**
         * Hard Tanh activation function.
         * A computationally cheaper, piecewise linear approximation of the tanh function:
         * f(x) = max(-1, min(1, x)). Outputs values clamped between -1 and 1.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the hard tanh function or its derivative (0 or 1).
         */
        hardTanh: (x, derivate = false) => {
          return derivate ? x > -1 && x < 1 ? 1 : 0 : Math.max(-1, Math.min(1, x));
        },
        /**
         * Absolute activation function.
         * Outputs the absolute value of the input: f(x) = |x|.
         *
         * Note: The derivative at x=0 is ambiguous (theoretically undefined). Here, we return 1.
         * If you need a different behavior, consider using a custom activation.
         *
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the absolute function or its derivative (sign of x).
         */
        absolute: (x, derivate = false) => {
          return derivate ? x < 0 ? -1 : 1 : Math.abs(x);
        },
        /**
         * Inverse activation function.
         * Outputs 1 minus the input: f(x) = 1 - x.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the inverse function or its derivative (-1).
         */
        inverse: (x, derivate = false) => {
          return derivate ? -1 : 1 - x;
        },
        /**
         * Scaled Exponential Linear Unit (SELU) activation function.
         *
         * SELU aims to induce self-normalizing properties, meaning the outputs of SELU units
         * automatically converge towards zero mean and unit variance.
         * f(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
         * Recommended for deep networks composed primarily of SELU units.
         *
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the SELU function or its derivative.
         * @see {@link https://arxiv.org/abs/1706.02515} - Self-Normalizing Neural Networks paper
         * @see {@link https://github.com/wagenaartje/neataptic/wiki/Activation#selu} - Neataptic context
         */
        selu: (x, derivate = false) => {
          const alpha = 1.6732632423543772;
          const scale = 1.0507009873554805;
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          return derivate ? x > 0 ? scale : (fx + alpha) * scale : fx * scale;
        },
        /**
         * Softplus activation function.
         * A smooth approximation of the ReLU function: f(x) = log(1 + exp(x)).
         * Always positive. Its derivative is the logistic sigmoid function.
         * This implementation includes checks for numerical stability to avoid overflow/underflow.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the softplus function or its derivative (logistic sigmoid).
         * @see {@link https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus}
         */
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
        /**
         * Swish activation function (SiLU - Sigmoid Linear Unit).
         * A self-gated activation function: f(x) = x * logistic(x).
         * Often performs better than ReLU in deeper models.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the swish function or its derivative.
         * @see {@link https://arxiv.org/abs/1710.05941} - Swish paper
         */
        swish: (x, derivate = false) => {
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          if (derivate) {
            const swish_x = x * sigmoid_x;
            return swish_x + sigmoid_x * (1 - swish_x);
          } else {
            return x * sigmoid_x;
          }
        },
        /**
         * Gaussian Error Linear Unit (GELU) activation function.
         * Smooth approximation of ReLU, often used in Transformer models.
         * f(x) = x * Φ(x), where Φ(x) is the standard Gaussian cumulative distribution function (CDF).
         * This implementation uses a common fast approximation of GELU.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the GELU function or its derivative.
         * @see {@link https://arxiv.org/abs/1606.08415}
         */
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
        /**
         * Mish activation function.
         * A self-gated activation function similar to Swish: f(x) = x * tanh(softplus(x)).
         * Aims to provide better performance than ReLU and Swish in some cases.
         * @param {number} x - The input value.
         * @param {boolean} [derivate=false] - Whether to compute the derivative.
         * @returns {number} The result of the Mish function or its derivative.
         * @see {@link https://arxiv.org/abs/1908.08681}
         */
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

  // src/methods/gating.ts
  var gating;
  var init_gating = __esm({
    "src/methods/gating.ts"() {
      "use strict";
      gating = {
        /**
         * Output Gating: The gating neuron(s) control the activation flowing *out*
         * of the connection's target neuron(s). The connection's weight remains static,
         * but the output signal from the target neuron is modulated by the gater's state.
         * @property {string} name - Identifier for the output gating method.
         */
        OUTPUT: {
          name: "OUTPUT"
        },
        /**
         * Input Gating: The gating neuron(s) control the activation flowing *into*
         * the connection's target neuron(s). The connection effectively transmits
         * `connection_weight * source_activation * gater_activation` to the target neuron.
         * @property {string} name - Identifier for the input gating method.
         */
        INPUT: {
          name: "INPUT"
        },
        /**
         * Self Gating: The gating neuron(s) directly modulate the *weight* or strength
         * of the connection itself. The connection's effective weight becomes dynamic,
         * influenced by the gater's activation state (`effective_weight = connection_weight * gater_activation`).
         * @property {string} name - Identifier for the self-gating method.
         */
        SELF: {
          name: "SELF"
        }
      };
    }
  });

  // src/methods/mutation.ts
  var mutation, mutation_default;
  var init_mutation = __esm({
    "src/methods/mutation.ts"() {
      "use strict";
      init_activation();
      mutation = {
        /**
         * Adds a new node to the network by splitting an existing connection.
         * The original connection is disabled, and two new connections are created:
         * one from the original source to the new node, and one from the new node
         * to the original target. This increases network complexity, potentially
         * allowing for more sophisticated computations.
         */
        ADD_NODE: {
          name: "ADD_NODE"
          /**
           * @see Instinct Algorithm - Section 3.1 Add Node Mutation
           */
        },
        /**
         * Removes a hidden node from the network. Connections to and from the
         * removed node are also removed. This simplifies the network topology.
         */
        SUB_NODE: {
          name: "SUB_NODE",
          /** If true, attempts to preserve gating connections associated with the removed node. */
          keep_gates: true
          /**
           * @see Instinct Algorithm - Section 3.7 Remove Node Mutation
           */
        },
        /**
         * Adds a new connection between two previously unconnected nodes.
         * This increases network connectivity, potentially creating new pathways
         * for information flow.
         */
        ADD_CONN: {
          name: "ADD_CONN"
          /**
           * @see Instinct Algorithm - Section 3.2 Add Connection Mutation
           */
        },
        /**
         * Removes an existing connection between two nodes.
         * This prunes the network, potentially removing redundant or detrimental pathways.
         */
        SUB_CONN: {
          name: "SUB_CONN"
          /**
           * @see Instinct Algorithm - Section 3.8 Remove Connection Mutation
           */
        },
        /**
         * Modifies the weight of an existing connection by adding a random value
         * or multiplying by a random factor. This fine-tunes the strength of
         * the connection.
         */
        MOD_WEIGHT: {
          name: "MOD_WEIGHT",
          /** Minimum value for the random modification factor/offset. */
          min: -1,
          /** Maximum value for the random modification factor/offset. */
          max: 1
          /**
           * @see Instinct Algorithm - Section 3.4 Modify Weight Mutation
           */
        },
        /**
         * Modifies the bias of a node (excluding input nodes) by adding a random value.
         * This adjusts the node's activation threshold, influencing its firing behavior.
         */
        MOD_BIAS: {
          name: "MOD_BIAS",
          /** Minimum value for the random modification offset. */
          min: -1,
          /** Maximum value for the random modification offset. */
          max: 1
          /**
           * @see Instinct Algorithm - Section 3.5 Modify Bias Mutation
           */
        },
        /**
         * Randomly changes the activation function of a node (excluding input nodes).
         * This allows nodes to specialize their response characteristics during evolution.
         */
        MOD_ACTIVATION: {
          name: "MOD_ACTIVATION",
          /** If true, allows mutation of activation functions in output nodes. */
          mutateOutput: true,
          /** A list of allowed activation functions to choose from during mutation. */
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
          /**
           * @see Instinct Algorithm - Section 3.6 Modify Squash Mutation
           */
        },
        /**
         * Adds a self-connection (recurrent connection from a node to itself).
         * This allows a node to retain information about its previous state,
         * introducing memory capabilities at the node level. Only applicable
         * to hidden and output nodes.
         */
        ADD_SELF_CONN: {
          name: "ADD_SELF_CONN"
        },
        /**
         * Removes a self-connection from a node.
         * This removes the node's direct recurrent loop.
         */
        SUB_SELF_CONN: {
          name: "SUB_SELF_CONN"
        },
        /**
         * Adds a gating mechanism to an existing connection. A new node (the gater)
         * is selected to control the flow of information through the gated connection.
         * This introduces multiplicative interactions, similar to LSTM or GRU units,
         * enabling more complex temporal processing or conditional logic.
         */
        ADD_GATE: {
          name: "ADD_GATE"
        },
        /**
         * Removes a gating mechanism from a connection.
         * This simplifies the network by removing the modulatory influence of the gater node.
         */
        SUB_GATE: {
          name: "SUB_GATE"
        },
        /**
         * Adds a recurrent connection between two nodes, potentially creating cycles
         * in the network graph (e.g., connecting a node to a node in a previous layer
         * or a non-adjacent node). This enables the network to maintain internal state
         * and process temporal dependencies.
         */
        ADD_BACK_CONN: {
          name: "ADD_BACK_CONN"
        },
        /**
         * Removes a recurrent connection (that is not a self-connection).
         * This simplifies the recurrent topology of the network.
         */
        SUB_BACK_CONN: {
          name: "SUB_BACK_CONN"
        },
        /**
         * Swaps the roles (bias and activation function) of two nodes (excluding input nodes).
         * Connections are generally preserved relative to the node indices.
         * This mutation alters the network's internal processing without changing
         * the overall node count or connection density.
         */
        SWAP_NODES: {
          name: "SWAP_NODES",
          /** If true, allows swapping involving output nodes. */
          mutateOutput: true
        },
        /**
         * Reinitializes the weights of all incoming, outgoing, and self connections for a node.
         * This can help escape local minima or inject diversity during evolution.
         */
        REINIT_WEIGHT: {
          name: "REINIT_WEIGHT",
          /** Range for random reinitialization. */
          min: -1,
          max: 1
        },
        /**
         * Marks a node for batch normalization. (Stub: actual normalization requires architectural support.)
         * This mutation can be used to toggle batch normalization on a node or layer.
         */
        BATCH_NORM: {
          name: "BATCH_NORM"
        },
        /**
         * Adds a new LSTM node (memory cell with gates) to the network.
         * This enables the evolution of memory-augmented architectures.
         */
        ADD_LSTM_NODE: {
          name: "ADD_LSTM_NODE"
          // Additional config can be added here if needed
        },
        /**
         * Adds a new GRU node (gated recurrent unit) to the network.
         * This enables the evolution of memory-augmented architectures.
         */
        ADD_GRU_NODE: {
          name: "ADD_GRU_NODE"
          // Additional config can be added here if needed
        },
        /** Placeholder for the list of all mutation methods. */
        ALL: [],
        /** Placeholder for the list of mutation methods suitable for feedforward networks. */
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
        // Added
        mutation.ADD_GRU_NODE
        // Added
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

  // src/methods/selection.ts
  var selection;
  var init_selection = __esm({
    "src/methods/selection.ts"() {
      "use strict";
      selection = {
        /**
         * Fitness Proportionate Selection (also known as Roulette Wheel Selection).
         *
         * Individuals are selected based on their fitness relative to the total fitness
         * of the population. An individual's chance of being selected is directly
         * proportional to its fitness score. Higher fitness means a higher probability
         * of selection. This method can struggle if fitness values are very close or
         * if there are large disparities.
         */
        FITNESS_PROPORTIONATE: {
          name: "FITNESS_PROPORTIONATE"
        },
        /**
         * Power Selection.
         *
         * Similar to Fitness Proportionate Selection, but fitness scores are raised
         * to a specified power before calculating selection probabilities. This increases
         * the selection pressure towards individuals with higher fitness scores, making
         * them disproportionately more likely to be selected compared to FITNESS_PROPORTIONATE.
         *
         * @property {number} power - The exponent applied to each individual's fitness score. Higher values increase selection pressure. Must be a positive number. Defaults to 4.
         */
        POWER: {
          name: "POWER",
          power: 4
        },
        /**
         * Tournament Selection.
         *
         * Selects individuals by holding competitions ('tournaments') among randomly
         * chosen subsets of the population. In each tournament, a fixed number (`size`)
         * of individuals are compared, and the fittest individual is chosen with a
         * certain `probability`. If not chosen (with probability 1 - `probability`),
         * the next fittest individual in the tournament might be selected (implementation dependent),
         * or another tournament might be run. This method is less sensitive to the scale
         * of fitness values compared to fitness proportionate methods.
         *
         * @property {number} size - The number of individuals participating in each tournament. Must be a positive integer. Defaults to 5.
         * @property {number} probability - The probability (between 0 and 1) of selecting the absolute fittest individual from the tournament participants. Defaults to 0.5.
         */
        TOURNAMENT: {
          name: "TOURNAMENT",
          size: 5,
          probability: 0.5
        }
      };
    }
  });

  // src/methods/crossover.ts
  var crossover;
  var init_crossover = __esm({
    "src/methods/crossover.ts"() {
      "use strict";
      crossover = {
        /**
         * Single-point crossover.
         * A single crossover point is selected, and genes are exchanged between parents up to this point.
         * This method is particularly useful for binary-encoded genomes.
         *
         * @property {string} name - The name of the crossover method.
         * @property {number[]} config - Configuration for the crossover point.
         * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover}
         */
        SINGLE_POINT: {
          name: "SINGLE_POINT",
          config: [0.4]
        },
        /**
         * Two-point crossover.
         * Two crossover points are selected, and genes are exchanged between parents between these points.
         * This method is an extension of single-point crossover and is often used for more complex genomes.
         *
         * @property {string} name - The name of the crossover method.
         * @property {number[]} config - Configuration for the two crossover points.
         * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_and_k-point_crossover}
         */
        TWO_POINT: {
          name: "TWO_POINT",
          config: [0.4, 0.9]
        },
        /**
         * Uniform crossover.
         * Each gene is selected randomly from one of the parents with equal probability.
         * This method provides a high level of genetic diversity in the offspring.
         *
         * @property {string} name - The name of the crossover method.
         * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_crossover}
         */
        UNIFORM: {
          name: "UNIFORM"
        },
        /**
         * Average crossover.
         * The offspring's genes are the average of the parents' genes.
         * This method is particularly useful for real-valued genomes.
         *
         * @property {string} name - The name of the crossover method.
         * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Arithmetic_recombination}
         */
        AVERAGE: {
          name: "AVERAGE"
        }
      };
    }
  });

  // src/methods/connection.ts
  var groupConnection, connection_default;
  var init_connection2 = __esm({
    "src/methods/connection.ts"() {
      "use strict";
      groupConnection = Object.freeze({
        // Renamed export
        /**
         * Connects all nodes in the source group to all nodes in the target group.
         */
        ALL_TO_ALL: Object.freeze({
          name: "ALL_TO_ALL"
          // Renamed name
        }),
        /**
         * Connects all nodes in the source group to all nodes in the target group, excluding self-connections (if groups are identical).
         */
        ALL_TO_ELSE: Object.freeze({
          name: "ALL_TO_ELSE"
          // Renamed name
        }),
        /**
         * Connects each node in the source group to the node at the same index in the target group. Requires groups to be the same size.
         */
        ONE_TO_ONE: Object.freeze({
          name: "ONE_TO_ONE"
          // Renamed name
        })
      });
      connection_default = groupConnection;
    }
  });

  // src/methods/methods.ts
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
    "src/methods/methods.ts"() {
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

  // src/architecture/node.ts
  var node_exports = {};
  __export(node_exports, {
    default: () => Node2
  });
  var Node2;
  var init_node = __esm({
    "src/architecture/node.ts"() {
      "use strict";
      init_connection();
      init_config();
      init_methods();
      Node2 = class _Node {
        /**
         * The bias value of the node. Added to the weighted sum of inputs before activation.
         * Input nodes typically have a bias of 0.
         */
        bias;
        /**
         * The activation function (squashing function) applied to the node's state.
         * Maps the internal state to the node's output (activation).
         * @param x The node's internal state (sum of weighted inputs + bias).
         * @param derivate If true, returns the derivative of the function instead of the function value.
         * @returns The activation value or its derivative.
         */
        squash;
        /**
         * The type of the node: 'input', 'hidden', or 'output'.
         * Determines behavior (e.g., input nodes don't have biases modified typically, output nodes calculate error differently).
         */
        type;
        /**
         * The output value of the node after applying the activation function. This is the value transmitted to connected nodes.
         */
        activation;
        /**
         * The internal state of the node (sum of weighted inputs + bias) before the activation function is applied.
         */
        state;
        /**
         * The node's state from the previous activation cycle. Used for recurrent self-connections.
         */
        old;
        /**
         * A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.
         */
        mask;
        /**
         * The change in bias applied in the previous training iteration. Used for calculating momentum.
         */
        previousDeltaBias;
        /**
         * Accumulates changes in bias over a mini-batch during batch training. Reset after each weight update.
         */
        totalDeltaBias;
        /**
         * Stores incoming, outgoing, gated, and self-connections for this node.
         */
        connections;
        /**
         * Stores error values calculated during backpropagation.
         */
        error;
        /**
         * The derivative of the activation function evaluated at the node's current state. Used in backpropagation.
         */
        derivative;
        // Deprecated: `nodes` & `gates` fields removed in refactor. Backwards access still works via getters below.
        /**
         * Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.
         */
        index;
        /**
         * Internal flag to detect cycles during activation
         */
        isActivating;
        /** Stable per-node gene identifier for NEAT innovation reuse */
        geneId;
        /**
         * Global index counter for assigning unique indices to nodes.
         */
        static _globalNodeIndex = 0;
        static _nextGeneId = 1;
        /**
         * Creates a new node.
         * @param type The type of the node ('input', 'hidden', or 'output'). Defaults to 'hidden'.
         * @param customActivation Optional custom activation function (should handle derivative if needed).
         */
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
            // Self-connection initialized as an empty array.
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
        /**
         * Sets a custom activation function for this node at runtime.
         * @param fn The activation function (should handle derivative if needed).
         */
        setActivation(fn) {
          this.squash = fn;
        }
        /**
         * Activates the node, calculating its output value based on inputs and state.
         * This method also calculates eligibility traces (`xtrace`) used for training recurrent connections.
         *
         * The activation process involves:
         * 1. Calculating the node's internal state (`this.state`) based on:
         *    - Incoming connections' weighted activations.
         *    - The recurrent self-connection's weighted state from the previous timestep (`this.old`).
         *    - The node's bias.
         * 2. Applying the activation function (`this.squash`) to the state to get the activation (`this.activation`).
         * 3. Applying the dropout mask (`this.mask`).
         * 4. Calculating the derivative of the activation function.
         * 5. Updating the gain of connections gated by this node.
         * 6. Calculating and updating eligibility traces for incoming connections.
         *
         * @param input Optional input value. If provided, sets the node's activation directly (used for input nodes).
         * @returns The calculated activation value of the node.
         * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#1-3-activation Instinct Algorithm - Section 1.3 Activation}
         */
        activate(input) {
          return this._activateCore(true, input);
        }
        /**
         * Activates the node without calculating eligibility traces (`xtrace`).
         * This is a performance optimization used during inference (when the network
         * is just making predictions, not learning) as trace calculations are only needed for training.
         *
         * @param input Optional input value. If provided, sets the node's activation directly (used for input nodes).
         * @returns The calculated activation value of the node.
         * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#1-3-activation Instinct Algorithm - Section 1.3 Activation}
         */
        noTraceActivate(input) {
          return this._activateCore(false, input);
        }
        /**
         * Internal shared implementation for activate/noTraceActivate.
         * @param withTrace Whether to update eligibility traces.
         * @param input Optional externally supplied activation (bypasses weighted sum if provided).
         */
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
              if (conn.dcMask === 0) continue;
              newState += conn.gain * conn.weight * this.old;
            }
          }
          if (this.connections.in.length) {
            for (const conn of this.connections.in) {
              if (conn.dcMask === 0 || conn.enabled === false) continue;
              newState += conn.from.activation * conn.weight * conn.gain;
            }
          }
          this.state = newState;
          if (typeof this.squash !== "function") {
            if (config.warnings)
              console.warn("Invalid activation function; using identity.");
            this.squash = activation_default.identity;
          }
          if (typeof this.mask !== "number") this.mask = 1;
          this.activation = this.squash(this.state) * this.mask;
          this.derivative = this.squash(this.state, true);
          if (this.connections.gated.length) {
            for (const conn of this.connections.gated) conn.gain = this.activation;
          }
          if (withTrace) {
            for (const conn of this.connections.in)
              conn.eligibility = conn.from.activation;
          }
          return this.activation;
        }
        // --- Backwards compatibility accessors for deprecated fields ---
        /** @deprecated Use connections.gated; retained for legacy tests */
        get gates() {
          if (config.warnings)
            console.warn("Node.gates is deprecated; use node.connections.gated");
          return this.connections.gated;
        }
        set gates(val) {
          this.connections.gated = val || [];
        }
        /** @deprecated Placeholder kept for legacy structural algorithms. No longer populated. */
        get nodes() {
          return [];
        }
        set nodes(_val) {
        }
        /**
         * Back-propagates the error signal through the node and calculates weight/bias updates.
         *
         * This method implements the backpropagation algorithm, including:
         * 1. Calculating the node's error responsibility based on errors from subsequent nodes (`projected` error)
         *    and errors from connections it gates (`gated` error).
         * 2. Calculating the gradient for each incoming connection's weight using eligibility traces (`xtrace`).
         * 3. Calculating the change (delta) for weights and bias, incorporating:
         *    - Learning rate.
         *    - L1/L2/custom regularization.
         *    - Momentum (using Nesterov Accelerated Gradient - NAG).
         * 4. Optionally applying the calculated updates immediately or accumulating them for batch training.
         *
         * @param rate The learning rate (controls the step size of updates).
         * @param momentum The momentum factor (helps accelerate learning and overcome local minima). Uses NAG.
         * @param update If true, apply the calculated weight/bias updates immediately. If false, accumulate them in `totalDelta*` properties for batch updates.
         * @param regularization The regularization setting. Can be:
         *   - number (L2 lambda)
         *   - { type: 'L1'|'L2', lambda: number }
         *   - (weight: number) => number (custom function)
         * @param target The target output value for this node. Only used if the node is of type 'output'.
         */
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
              error += connection.to.error.responsibility * // Error responsibility of the node this connection points to.
              connection.weight * // Weight of the connection.
              connection.gain;
            }
            this.error.projected = this.derivative * error;
            error = 0;
            for (const connection of this.connections.gated) {
              const node = connection.to;
              let influence = node.connections.self.reduce(
                (sum, selfConn) => sum + (selfConn.gater === this ? node.old : 0),
                0
              );
              influence += connection.weight * connection.from.activation;
              error += node.error.responsibility * influence;
            }
            this.error.gated = this.derivative * error;
            this.error.responsibility = this.error.projected + this.error.gated;
          }
          if (this.type === "constant") return;
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
                console.warn(
                  `Weight update produced invalid value: ${connection.weight}. Resetting to 0.`,
                  { node: this.index, connection }
                );
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
              console.warn(
                "self totalDeltaWeight became NaN/Infinity, resetting to 0",
                { node: this.index, connection }
              );
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
                console.warn(
                  "self weight update produced invalid value, resetting to 0",
                  { node: this.index, connection }
                );
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
        /**
         * Converts the node's essential properties to a JSON object for serialization.
         * Does not include state, activation, error, or connection information, as these
         * are typically transient or reconstructed separately.
         * @returns A JSON representation of the node's configuration.
         */
        toJSON() {
          return {
            index: this.index,
            bias: this.bias,
            type: this.type,
            squash: this.squash ? this.squash.name : null,
            mask: this.mask
          };
        }
        /**
         * Creates a Node instance from a JSON object.
         * @param json The JSON object containing node configuration.
         * @returns A new Node instance configured according to the JSON object.
         */
        static fromJSON(json) {
          const node = new _Node(json.type);
          node.bias = json.bias;
          node.mask = json.mask;
          if (json.squash) {
            const squashFn = activation_default[json.squash];
            if (typeof squashFn === "function") {
              node.squash = squashFn;
            } else {
              console.warn(
                `fromJSON: Unknown or invalid squash function '${json.squash}' for node. Using identity.`
              );
              node.squash = activation_default.identity;
            }
          }
          return node;
        }
        /**
         * Checks if this node is connected to another node.
         * @param target The target node to check the connection with.
         * @returns True if connected, otherwise false.
         */
        isConnectedTo(target) {
          return this.connections.out.some((conn) => conn.to === target);
        }
        /**
         * Applies a mutation method to the node. Used in neuro-evolution.
         *
         * This allows modifying the node's properties, such as its activation function or bias,
         * based on predefined mutation methods.
         *
         * @param method A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).
         * @throws {Error} If the mutation method is invalid, not provided, or not found in `methods.mutation`.
         * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
         */
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
                console.warn(
                  "MOD_ACTIVATION mutation called without allowed functions specified."
                );
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
            // Add cases for other mutation types if needed.
            default:
              throw new Error(`Unsupported mutation method: ${method.name}`);
          }
        }
        /**
         * Creates a connection from this node to a target node or all nodes in a group.
         *
         * @param target The target Node or a group object containing a `nodes` array.
         * @param weight The weight for the new connection(s). If undefined, a default or random weight might be assigned by the Connection constructor (currently defaults to 0, consider changing).
         * @returns An array containing the newly created Connection object(s).
         * @throws {Error} If the target is undefined.
         * @throws {Error} If trying to create a self-connection when one already exists (weight is not 0).
         */
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
            throw new Error(
              "Invalid target type for connection. Must be a Node or a group { nodes: Node[] }."
            );
          }
          return connections;
        }
        /**
         * Removes the connection from this node to the target node.
         *
         * @param target The target node to disconnect from.
         * @param twosided If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.
         */
        disconnect(target, twosided = false) {
          if (this === target) {
            this.connections.self = [];
            return;
          }
          this.connections.out = this.connections.out.filter((conn) => {
            if (conn.to === target) {
              target.connections.in = target.connections.in.filter(
                (inConn) => inConn !== conn
                // Filter by reference.
              );
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
        /**
         * Makes this node gate the provided connection(s).
         * The connection's gain will be controlled by this node's activation value.
         *
         * @param connections A single Connection object or an array of Connection objects to be gated.
         */
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
              console.warn(
                "Connection is already gated by another node. Ungate first."
              );
              continue;
            }
            this.connections.gated.push(connection);
            connection.gater = this;
          }
        }
        /**
         * Removes this node's gating control over the specified connection(s).
         * Resets the connection's gain to 1 and removes it from the `connections.gated` list.
         *
         * @param connections A single Connection object or an array of Connection objects to ungate.
         */
        ungate(connections) {
          if (!Array.isArray(connections)) {
            connections = [connections];
          }
          for (const connection of connections) {
            if (!connection) continue;
            const index = this.connections.gated.indexOf(connection);
            if (index !== -1) {
              this.connections.gated.splice(index, 1);
              connection.gater = null;
              connection.gain = 1;
            } else {
            }
          }
        }
        /**
         * Clears the node's dynamic state information.
         * Resets activation, state, previous state, error signals, and eligibility traces.
         * Useful for starting a new activation sequence (e.g., for a new input pattern).
         */
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
        /**
         * Checks if this node has a direct outgoing connection to the given node.
         * Considers both regular outgoing connections and the self-connection.
         *
         * @param node The potential target node.
         * @returns True if this node projects to the target node, false otherwise.
         */
        isProjectingTo(node) {
          if (node === this && this.connections.self.length > 0) return true;
          return this.connections.out.some((conn) => conn.to === node);
        }
        /**
         * Checks if the given node has a direct outgoing connection to this node.
         * Considers both regular incoming connections and the self-connection.
         *
         * @param node The potential source node.
         * @returns True if the given node projects to this node, false otherwise.
         */
        isProjectedBy(node) {
          if (node === this && this.connections.self.length > 0) return true;
          return this.connections.in.some((conn) => conn.from === node);
        }
        /**
         * Applies accumulated batch updates to incoming and self connections and this node's bias.
         * Uses momentum in a Nesterov-compatible way: currentDelta = accumulated + momentum * previousDelta.
         * Resets accumulators after applying. Safe to call on any node type.
         * @param momentum Momentum factor (0 to disable)
         */
        applyBatchUpdates(momentum) {
          return this.applyBatchUpdatesWithOptimizer({ type: "sgd", momentum });
        }
        /**
         * Extended batch update supporting multiple optimizers.
         *
         * Applies accumulated (batch) gradients stored in `totalDeltaWeight` / `totalDeltaBias` to the
         * underlying weights and bias using the selected optimization algorithm. Supports both classic
         * SGD (with Nesterov-style momentum via preceding propagate logic) and a collection of adaptive
         * optimizers. After applying an update, gradient accumulators are reset to 0.
         *
         * Supported optimizers (type):
         *  - 'sgd'      : Standard gradient descent with optional momentum.
         *  - 'rmsprop'  : Exponential moving average of squared gradients (cache) to normalize step.
         *  - 'adagrad'  : Accumulate squared gradients; learning rate effectively decays per weight.
         *  - 'adam'     : Bias‑corrected first (m) & second (v) moment estimates.
         *  - 'adamw'    : Adam with decoupled weight decay (applied after adaptive step).
         *  - 'amsgrad'  : Adam variant maintaining a maximum of past v (vhat) to enforce non‑increasing step size.
         *  - 'adamax'   : Adam variant using the infinity norm (u) instead of second moment.
         *  - 'nadam'    : Adam + Nesterov momentum style update (lookahead on first moment).
         *  - 'radam'    : Rectified Adam – warms up variance by adaptively rectifying denominator when sample size small.
         *  - 'lion'     : Uses sign of combination of two momentum buffers (beta1 & beta2) for update direction only.
         *  - 'adabelief': Adam-like but second moment on (g - m) (gradient surprise) for variance reduction.
         *  - 'lookahead': Wrapper; performs k fast optimizer steps then interpolates (alpha) towards a slow (shadow) weight.
         *
         * Options:
         *  - momentum     : (SGD) momentum factor (Nesterov handled in propagate when update=true).
         *  - beta1/beta2  : Exponential decay rates for first/second moments (Adam family, Lion, AdaBelief, etc.).
         *  - eps          : Numerical stability epsilon added to denominator terms.
         *  - weightDecay  : Decoupled weight decay (AdamW) or additionally applied after main step when adamw selected.
         *  - lrScale      : Learning rate scalar already scheduled externally (passed as currentRate).
         *  - t            : Global step (1-indexed) for bias correction / rectification.
         *  - baseType     : Underlying optimizer for lookahead (not itself lookahead).
         *  - la_k         : Lookahead synchronization interval (number of fast steps).
         *  - la_alpha     : Interpolation factor towards slow (shadow) weights/bias at sync points.
         *
         * Internal per-connection temp fields (created lazily):
         *  - firstMoment / secondMoment / maxSecondMoment / infinityNorm : Moment / variance / max variance / infinity norm caches.
         *  - gradientAccumulator : Single accumulator (RMSProp / AdaGrad).
         *  - previousDeltaWeight : For classic SGD momentum.
         *  - lookaheadShadowWeight / _la_shadowBias : Lookahead shadow copies.
         *
         * Safety: We clip extreme weight / bias magnitudes and guard against NaN/Infinity.
         *
         * @param opts Optimizer configuration (see above).
         */
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
            if (!Number.isFinite(g)) g = 0;
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
                  conn.maxSecondMoment = Math.max(
                    conn.maxSecondMoment ?? 0,
                    conn.secondMoment ?? 0
                  );
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
                conn.infinityNorm = Math.max(
                  (conn.infinityNorm ?? 0) * beta2,
                  Math.abs(g)
                );
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
                this._safeUpdateWeight(
                  conn,
                  mNesterov / (Math.sqrt(vHat) + eps) * lrScale
                );
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
                  const rt = Math.sqrt(
                    (rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT)
                  );
                  this._safeUpdateWeight(
                    conn,
                    rt * mHat / (Math.sqrt(vHat) + eps) * lrScale
                  );
                } else {
                  this._safeUpdateWeight(conn, mHat * lrScale);
                }
                break;
              }
              case "lion": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                conn.secondMomentum = (conn.secondMomentum ?? 0) * beta2 + (1 - beta2) * g;
                const update = Math.sign(
                  (conn.firstMoment || 0) + (conn.secondMomentum || 0)
                );
                this._safeUpdateWeight(conn, -update * lrScale);
                break;
              }
              case "adabelief": {
                conn.firstMoment = (conn.firstMoment ?? 0) * beta1 + (1 - beta1) * g;
                const g_m = g - conn.firstMoment;
                conn.secondMoment = (conn.secondMoment ?? 0) * beta2 + (1 - beta2) * (g_m * g_m);
                const mHat = conn.firstMoment / (1 - Math.pow(beta1, t));
                const vHat = conn.secondMoment / (1 - Math.pow(beta2, t));
                this._safeUpdateWeight(
                  conn,
                  mHat / (Math.sqrt(vHat) + eps + 1e-12) * lrScale
                );
                break;
              }
              default: {
                let currentDeltaWeight = g + momentum * (conn.previousDeltaWeight || 0);
                if (!Number.isFinite(currentDeltaWeight)) currentDeltaWeight = 0;
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
          for (const connection of this.connections.in) applyConn(connection);
          for (const connection of this.connections.self) applyConn(connection);
          if (this.type !== "input" && this.type !== "constant") {
            let gB = this.totalDeltaBias || 0;
            if (!Number.isFinite(gB)) gB = 0;
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
                this.opt_vhatB = Math.max(
                  this.opt_vhatB ?? 0,
                  this.opt_vB ?? 0
                );
              }
              const vEffB = effectiveType === "amsgrad" ? this.opt_vhatB : this.opt_vB;
              const mHatB = this.opt_mB / (1 - Math.pow(beta1, t));
              const vHatB = vEffB / (1 - Math.pow(beta2, t));
              let stepB;
              if (effectiveType === "adamax") {
                this.opt_uB = Math.max(
                  (this.opt_uB ?? 0) * beta2,
                  Math.abs(gB)
                );
                stepB = mHatB / (this.opt_uB || 1e-12) * lrScale;
              } else if (effectiveType === "nadam") {
                const mNesterovB = mHatB * beta1 + (1 - beta1) * gB / (1 - Math.pow(beta1, t));
                stepB = mNesterovB / (Math.sqrt(vHatB) + eps) * lrScale;
              } else if (effectiveType === "radam") {
                const rhoInf = 2 / (1 - beta2) - 1;
                const rhoT = rhoInf - 2 * t * Math.pow(beta2, t) / (1 - Math.pow(beta2, t));
                if (rhoT > 4) {
                  const rt = Math.sqrt(
                    (rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT)
                  );
                  stepB = rt * mHatB / (Math.sqrt(vHatB) + eps) * lrScale;
                } else {
                  stepB = mHatB * lrScale;
                }
              } else if (effectiveType === "lion") {
                const updateB = Math.sign(
                  this.opt_mB + this.opt_mB2
                );
                stepB = -updateB * lrScale;
              } else if (effectiveType === "adabelief") {
                stepB = mHatB / (Math.sqrt(vHatB) + eps + 1e-12) * lrScale;
              } else {
                stepB = mHatB / (Math.sqrt(vHatB) + eps) * lrScale;
              }
              if (effectiveType === "adamw" && wd !== 0)
                stepB -= wd * (this.bias || 0) * lrScale;
              let nextBias = this.bias + stepB;
              if (!Number.isFinite(nextBias)) nextBias = 0;
              if (Math.abs(nextBias) > 1e6) nextBias = Math.sign(nextBias) * 1e6;
              this.bias = nextBias;
            } else {
              let currentDeltaBias = gB + momentum * (this.previousDeltaBias || 0);
              if (!Number.isFinite(currentDeltaBias)) currentDeltaBias = 0;
              if (Math.abs(currentDeltaBias) > 1e3)
                currentDeltaBias = Math.sign(currentDeltaBias) * 1e3;
              let nextBias = this.bias + currentDeltaBias * lrScale;
              if (!Number.isFinite(nextBias)) nextBias = 0;
              if (Math.abs(nextBias) > 1e6) nextBias = Math.sign(nextBias) * 1e6;
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
              for (const c of this.connections.in) blendConn(c);
              for (const c of this.connections.self) blendConn(c);
            }
          }
        }
        /**
         * Internal helper to safely update a connection weight with clipping and NaN checks.
         */
        _safeUpdateWeight(connection, delta) {
          let next = connection.weight + delta;
          if (!Number.isFinite(next)) next = 0;
          if (Math.abs(next) > 1e6) next = Math.sign(next) * 1e6;
          connection.weight = next;
        }
      };
    }
  });

  // src/architecture/nodePool.ts
  function resetNode(node, type, rng = Math.random) {
    if (type) node.type = type;
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
      if (activationFn) node.squash = activationFn;
    } else {
      node = new Node2(type, activationFn, rng);
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
    if (pool.length > highWaterMark) highWaterMark = pool.length;
  }
  var pool, highWaterMark, nextGeneId, reusedCount, freshCount;
  var init_nodePool = __esm({
    "src/architecture/nodePool.ts"() {
      "use strict";
      init_node();
      pool = [];
      highWaterMark = 0;
      nextGeneId = 1;
      reusedCount = 0;
      freshCount = 0;
    }
  });

  // node_modules/util/support/isBufferBrowser.js
  var require_isBufferBrowser = __commonJS({
    "node_modules/util/support/isBufferBrowser.js"(exports, module) {
      module.exports = function isBuffer(arg) {
        return arg && typeof arg === "object" && typeof arg.copy === "function" && typeof arg.fill === "function" && typeof arg.readUInt8 === "function";
      };
    }
  });

  // node_modules/util/node_modules/inherits/inherits_browser.js
  var require_inherits_browser = __commonJS({
    "node_modules/util/node_modules/inherits/inherits_browser.js"(exports, module) {
      if (typeof Object.create === "function") {
        module.exports = function inherits(ctor, superCtor) {
          ctor.super_ = superCtor;
          ctor.prototype = Object.create(superCtor.prototype, {
            constructor: {
              value: ctor,
              enumerable: false,
              writable: true,
              configurable: true
            }
          });
        };
      } else {
        module.exports = function inherits(ctor, superCtor) {
          ctor.super_ = superCtor;
          var TempCtor = function() {
          };
          TempCtor.prototype = superCtor.prototype;
          ctor.prototype = new TempCtor();
          ctor.prototype.constructor = ctor;
        };
      }
    }
  });

  // node_modules/util/util.js
  var require_util = __commonJS({
    "node_modules/util/util.js"(exports) {
      var formatRegExp = /%[sdj%]/g;
      exports.format = function(f) {
        if (!isString(f)) {
          var objects = [];
          for (var i = 0; i < arguments.length; i++) {
            objects.push(inspect(arguments[i]));
          }
          return objects.join(" ");
        }
        var i = 1;
        var args = arguments;
        var len = args.length;
        var str = String(f).replace(formatRegExp, function(x2) {
          if (x2 === "%%") return "%";
          if (i >= len) return x2;
          switch (x2) {
            case "%s":
              return String(args[i++]);
            case "%d":
              return Number(args[i++]);
            case "%j":
              try {
                return JSON.stringify(args[i++]);
              } catch (_) {
                return "[Circular]";
              }
            default:
              return x2;
          }
        });
        for (var x = args[i]; i < len; x = args[++i]) {
          if (isNull(x) || !isObject(x)) {
            str += " " + x;
          } else {
            str += " " + inspect(x);
          }
        }
        return str;
      };
      exports.deprecate = function(fn, msg) {
        if (isUndefined(global.process)) {
          return function() {
            return exports.deprecate(fn, msg).apply(this, arguments);
          };
        }
        if (process.noDeprecation === true) {
          return fn;
        }
        var warned = false;
        function deprecated() {
          if (!warned) {
            if (process.throwDeprecation) {
              throw new Error(msg);
            } else if (process.traceDeprecation) {
              console.trace(msg);
            } else {
              console.error(msg);
            }
            warned = true;
          }
          return fn.apply(this, arguments);
        }
        return deprecated;
      };
      var debugs = {};
      var debugEnviron;
      exports.debuglog = function(set) {
        if (isUndefined(debugEnviron))
          debugEnviron = process.env.NODE_DEBUG || "";
        set = set.toUpperCase();
        if (!debugs[set]) {
          if (new RegExp("\\b" + set + "\\b", "i").test(debugEnviron)) {
            var pid = process.pid;
            debugs[set] = function() {
              var msg = exports.format.apply(exports, arguments);
              console.error("%s %d: %s", set, pid, msg);
            };
          } else {
            debugs[set] = function() {
            };
          }
        }
        return debugs[set];
      };
      function inspect(obj, opts) {
        var ctx = {
          seen: [],
          stylize: stylizeNoColor
        };
        if (arguments.length >= 3) ctx.depth = arguments[2];
        if (arguments.length >= 4) ctx.colors = arguments[3];
        if (isBoolean(opts)) {
          ctx.showHidden = opts;
        } else if (opts) {
          exports._extend(ctx, opts);
        }
        if (isUndefined(ctx.showHidden)) ctx.showHidden = false;
        if (isUndefined(ctx.depth)) ctx.depth = 2;
        if (isUndefined(ctx.colors)) ctx.colors = false;
        if (isUndefined(ctx.customInspect)) ctx.customInspect = true;
        if (ctx.colors) ctx.stylize = stylizeWithColor;
        return formatValue(ctx, obj, ctx.depth);
      }
      exports.inspect = inspect;
      inspect.colors = {
        "bold": [1, 22],
        "italic": [3, 23],
        "underline": [4, 24],
        "inverse": [7, 27],
        "white": [37, 39],
        "grey": [90, 39],
        "black": [30, 39],
        "blue": [34, 39],
        "cyan": [36, 39],
        "green": [32, 39],
        "magenta": [35, 39],
        "red": [31, 39],
        "yellow": [33, 39]
      };
      inspect.styles = {
        "special": "cyan",
        "number": "yellow",
        "boolean": "yellow",
        "undefined": "grey",
        "null": "bold",
        "string": "green",
        "date": "magenta",
        // "name": intentionally not styling
        "regexp": "red"
      };
      function stylizeWithColor(str, styleType) {
        var style = inspect.styles[styleType];
        if (style) {
          return "\x1B[" + inspect.colors[style][0] + "m" + str + "\x1B[" + inspect.colors[style][1] + "m";
        } else {
          return str;
        }
      }
      function stylizeNoColor(str, styleType) {
        return str;
      }
      function arrayToHash(array) {
        var hash = {};
        array.forEach(function(val, idx) {
          hash[val] = true;
        });
        return hash;
      }
      function formatValue(ctx, value, recurseTimes) {
        if (ctx.customInspect && value && isFunction(value.inspect) && // Filter out the util module, it's inspect function is special
        value.inspect !== exports.inspect && // Also filter out any prototype objects using the circular check.
        !(value.constructor && value.constructor.prototype === value)) {
          var ret = value.inspect(recurseTimes, ctx);
          if (!isString(ret)) {
            ret = formatValue(ctx, ret, recurseTimes);
          }
          return ret;
        }
        var primitive = formatPrimitive(ctx, value);
        if (primitive) {
          return primitive;
        }
        var keys = Object.keys(value);
        var visibleKeys = arrayToHash(keys);
        if (ctx.showHidden) {
          keys = Object.getOwnPropertyNames(value);
        }
        if (isError(value) && (keys.indexOf("message") >= 0 || keys.indexOf("description") >= 0)) {
          return formatError(value);
        }
        if (keys.length === 0) {
          if (isFunction(value)) {
            var name = value.name ? ": " + value.name : "";
            return ctx.stylize("[Function" + name + "]", "special");
          }
          if (isRegExp(value)) {
            return ctx.stylize(RegExp.prototype.toString.call(value), "regexp");
          }
          if (isDate(value)) {
            return ctx.stylize(Date.prototype.toString.call(value), "date");
          }
          if (isError(value)) {
            return formatError(value);
          }
        }
        var base = "", array = false, braces = ["{", "}"];
        if (isArray(value)) {
          array = true;
          braces = ["[", "]"];
        }
        if (isFunction(value)) {
          var n = value.name ? ": " + value.name : "";
          base = " [Function" + n + "]";
        }
        if (isRegExp(value)) {
          base = " " + RegExp.prototype.toString.call(value);
        }
        if (isDate(value)) {
          base = " " + Date.prototype.toUTCString.call(value);
        }
        if (isError(value)) {
          base = " " + formatError(value);
        }
        if (keys.length === 0 && (!array || value.length == 0)) {
          return braces[0] + base + braces[1];
        }
        if (recurseTimes < 0) {
          if (isRegExp(value)) {
            return ctx.stylize(RegExp.prototype.toString.call(value), "regexp");
          } else {
            return ctx.stylize("[Object]", "special");
          }
        }
        ctx.seen.push(value);
        var output;
        if (array) {
          output = formatArray(ctx, value, recurseTimes, visibleKeys, keys);
        } else {
          output = keys.map(function(key) {
            return formatProperty(ctx, value, recurseTimes, visibleKeys, key, array);
          });
        }
        ctx.seen.pop();
        return reduceToSingleString(output, base, braces);
      }
      function formatPrimitive(ctx, value) {
        if (isUndefined(value))
          return ctx.stylize("undefined", "undefined");
        if (isString(value)) {
          var simple = "'" + JSON.stringify(value).replace(/^"|"$/g, "").replace(/'/g, "\\'").replace(/\\"/g, '"') + "'";
          return ctx.stylize(simple, "string");
        }
        if (isNumber(value))
          return ctx.stylize("" + value, "number");
        if (isBoolean(value))
          return ctx.stylize("" + value, "boolean");
        if (isNull(value))
          return ctx.stylize("null", "null");
      }
      function formatError(value) {
        return "[" + Error.prototype.toString.call(value) + "]";
      }
      function formatArray(ctx, value, recurseTimes, visibleKeys, keys) {
        var output = [];
        for (var i = 0, l = value.length; i < l; ++i) {
          if (hasOwnProperty(value, String(i))) {
            output.push(formatProperty(
              ctx,
              value,
              recurseTimes,
              visibleKeys,
              String(i),
              true
            ));
          } else {
            output.push("");
          }
        }
        keys.forEach(function(key) {
          if (!key.match(/^\d+$/)) {
            output.push(formatProperty(
              ctx,
              value,
              recurseTimes,
              visibleKeys,
              key,
              true
            ));
          }
        });
        return output;
      }
      function formatProperty(ctx, value, recurseTimes, visibleKeys, key, array) {
        var name, str, desc;
        desc = Object.getOwnPropertyDescriptor(value, key) || { value: value[key] };
        if (desc.get) {
          if (desc.set) {
            str = ctx.stylize("[Getter/Setter]", "special");
          } else {
            str = ctx.stylize("[Getter]", "special");
          }
        } else {
          if (desc.set) {
            str = ctx.stylize("[Setter]", "special");
          }
        }
        if (!hasOwnProperty(visibleKeys, key)) {
          name = "[" + key + "]";
        }
        if (!str) {
          if (ctx.seen.indexOf(desc.value) < 0) {
            if (isNull(recurseTimes)) {
              str = formatValue(ctx, desc.value, null);
            } else {
              str = formatValue(ctx, desc.value, recurseTimes - 1);
            }
            if (str.indexOf("\n") > -1) {
              if (array) {
                str = str.split("\n").map(function(line) {
                  return "  " + line;
                }).join("\n").substr(2);
              } else {
                str = "\n" + str.split("\n").map(function(line) {
                  return "   " + line;
                }).join("\n");
              }
            }
          } else {
            str = ctx.stylize("[Circular]", "special");
          }
        }
        if (isUndefined(name)) {
          if (array && key.match(/^\d+$/)) {
            return str;
          }
          name = JSON.stringify("" + key);
          if (name.match(/^"([a-zA-Z_][a-zA-Z_0-9]*)"$/)) {
            name = name.substr(1, name.length - 2);
            name = ctx.stylize(name, "name");
          } else {
            name = name.replace(/'/g, "\\'").replace(/\\"/g, '"').replace(/(^"|"$)/g, "'");
            name = ctx.stylize(name, "string");
          }
        }
        return name + ": " + str;
      }
      function reduceToSingleString(output, base, braces) {
        var numLinesEst = 0;
        var length = output.reduce(function(prev, cur) {
          numLinesEst++;
          if (cur.indexOf("\n") >= 0) numLinesEst++;
          return prev + cur.replace(/\u001b\[\d\d?m/g, "").length + 1;
        }, 0);
        if (length > 60) {
          return braces[0] + (base === "" ? "" : base + "\n ") + " " + output.join(",\n  ") + " " + braces[1];
        }
        return braces[0] + base + " " + output.join(", ") + " " + braces[1];
      }
      function isArray(ar) {
        return Array.isArray(ar);
      }
      exports.isArray = isArray;
      function isBoolean(arg) {
        return typeof arg === "boolean";
      }
      exports.isBoolean = isBoolean;
      function isNull(arg) {
        return arg === null;
      }
      exports.isNull = isNull;
      function isNullOrUndefined(arg) {
        return arg == null;
      }
      exports.isNullOrUndefined = isNullOrUndefined;
      function isNumber(arg) {
        return typeof arg === "number";
      }
      exports.isNumber = isNumber;
      function isString(arg) {
        return typeof arg === "string";
      }
      exports.isString = isString;
      function isSymbol(arg) {
        return typeof arg === "symbol";
      }
      exports.isSymbol = isSymbol;
      function isUndefined(arg) {
        return arg === void 0;
      }
      exports.isUndefined = isUndefined;
      function isRegExp(re) {
        return isObject(re) && objectToString(re) === "[object RegExp]";
      }
      exports.isRegExp = isRegExp;
      function isObject(arg) {
        return typeof arg === "object" && arg !== null;
      }
      exports.isObject = isObject;
      function isDate(d) {
        return isObject(d) && objectToString(d) === "[object Date]";
      }
      exports.isDate = isDate;
      function isError(e) {
        return isObject(e) && (objectToString(e) === "[object Error]" || e instanceof Error);
      }
      exports.isError = isError;
      function isFunction(arg) {
        return typeof arg === "function";
      }
      exports.isFunction = isFunction;
      function isPrimitive(arg) {
        return arg === null || typeof arg === "boolean" || typeof arg === "number" || typeof arg === "string" || typeof arg === "symbol" || // ES6 symbol
        typeof arg === "undefined";
      }
      exports.isPrimitive = isPrimitive;
      exports.isBuffer = require_isBufferBrowser();
      function objectToString(o) {
        return Object.prototype.toString.call(o);
      }
      function pad(n) {
        return n < 10 ? "0" + n.toString(10) : n.toString(10);
      }
      var months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
      ];
      function timestamp() {
        var d = /* @__PURE__ */ new Date();
        var time = [
          pad(d.getHours()),
          pad(d.getMinutes()),
          pad(d.getSeconds())
        ].join(":");
        return [d.getDate(), months[d.getMonth()], time].join(" ");
      }
      exports.log = function() {
        console.log("%s - %s", timestamp(), exports.format.apply(exports, arguments));
      };
      exports.inherits = require_inherits_browser();
      exports._extend = function(origin, add) {
        if (!add || !isObject(add)) return origin;
        var keys = Object.keys(add);
        var i = keys.length;
        while (i--) {
          origin[keys[i]] = add[keys[i]];
        }
        return origin;
      };
      function hasOwnProperty(obj, prop) {
        return Object.prototype.hasOwnProperty.call(obj, prop);
      }
    }
  });

  // node_modules/path/path.js
  var require_path = __commonJS({
    "node_modules/path/path.js"(exports, module) {
      "use strict";
      var isWindows = process.platform === "win32";
      var util = require_util();
      function normalizeArray(parts, allowAboveRoot) {
        var res = [];
        for (var i = 0; i < parts.length; i++) {
          var p = parts[i];
          if (!p || p === ".")
            continue;
          if (p === "..") {
            if (res.length && res[res.length - 1] !== "..") {
              res.pop();
            } else if (allowAboveRoot) {
              res.push("..");
            }
          } else {
            res.push(p);
          }
        }
        return res;
      }
      function trimArray(arr) {
        var lastIndex = arr.length - 1;
        var start2 = 0;
        for (; start2 <= lastIndex; start2++) {
          if (arr[start2])
            break;
        }
        var end = lastIndex;
        for (; end >= 0; end--) {
          if (arr[end])
            break;
        }
        if (start2 === 0 && end === lastIndex)
          return arr;
        if (start2 > end)
          return [];
        return arr.slice(start2, end + 1);
      }
      var splitDeviceRe = /^([a-zA-Z]:|[\\\/]{2}[^\\\/]+[\\\/]+[^\\\/]+)?([\\\/])?([\s\S]*?)$/;
      var splitTailRe = /^([\s\S]*?)((?:\.{1,2}|[^\\\/]+?|)(\.[^.\/\\]*|))(?:[\\\/]*)$/;
      var win32 = {};
      function win32SplitPath(filename) {
        var result = splitDeviceRe.exec(filename), device = (result[1] || "") + (result[2] || ""), tail = result[3] || "";
        var result2 = splitTailRe.exec(tail), dir = result2[1], basename = result2[2], ext = result2[3];
        return [device, dir, basename, ext];
      }
      function win32StatPath(path2) {
        var result = splitDeviceRe.exec(path2), device = result[1] || "", isUnc = !!device && device[1] !== ":";
        return {
          device,
          isUnc,
          isAbsolute: isUnc || !!result[2],
          // UNC paths are always absolute
          tail: result[3]
        };
      }
      function normalizeUNCRoot(device) {
        return "\\\\" + device.replace(/^[\\\/]+/, "").replace(/[\\\/]+/g, "\\");
      }
      win32.resolve = function() {
        var resolvedDevice = "", resolvedTail = "", resolvedAbsolute = false;
        for (var i = arguments.length - 1; i >= -1; i--) {
          var path2;
          if (i >= 0) {
            path2 = arguments[i];
          } else if (!resolvedDevice) {
            path2 = process.cwd();
          } else {
            path2 = process.env["=" + resolvedDevice];
            if (!path2 || path2.substr(0, 3).toLowerCase() !== resolvedDevice.toLowerCase() + "\\") {
              path2 = resolvedDevice + "\\";
            }
          }
          if (!util.isString(path2)) {
            throw new TypeError("Arguments to path.resolve must be strings");
          } else if (!path2) {
            continue;
          }
          var result = win32StatPath(path2), device = result.device, isUnc = result.isUnc, isAbsolute = result.isAbsolute, tail = result.tail;
          if (device && resolvedDevice && device.toLowerCase() !== resolvedDevice.toLowerCase()) {
            continue;
          }
          if (!resolvedDevice) {
            resolvedDevice = device;
          }
          if (!resolvedAbsolute) {
            resolvedTail = tail + "\\" + resolvedTail;
            resolvedAbsolute = isAbsolute;
          }
          if (resolvedDevice && resolvedAbsolute) {
            break;
          }
        }
        if (isUnc) {
          resolvedDevice = normalizeUNCRoot(resolvedDevice);
        }
        resolvedTail = normalizeArray(
          resolvedTail.split(/[\\\/]+/),
          !resolvedAbsolute
        ).join("\\");
        return resolvedDevice + (resolvedAbsolute ? "\\" : "") + resolvedTail || ".";
      };
      win32.normalize = function(path2) {
        var result = win32StatPath(path2), device = result.device, isUnc = result.isUnc, isAbsolute = result.isAbsolute, tail = result.tail, trailingSlash = /[\\\/]$/.test(tail);
        tail = normalizeArray(tail.split(/[\\\/]+/), !isAbsolute).join("\\");
        if (!tail && !isAbsolute) {
          tail = ".";
        }
        if (tail && trailingSlash) {
          tail += "\\";
        }
        if (isUnc) {
          device = normalizeUNCRoot(device);
        }
        return device + (isAbsolute ? "\\" : "") + tail;
      };
      win32.isAbsolute = function(path2) {
        return win32StatPath(path2).isAbsolute;
      };
      win32.join = function() {
        var paths = [];
        for (var i = 0; i < arguments.length; i++) {
          var arg = arguments[i];
          if (!util.isString(arg)) {
            throw new TypeError("Arguments to path.join must be strings");
          }
          if (arg) {
            paths.push(arg);
          }
        }
        var joined = paths.join("\\");
        if (!/^[\\\/]{2}[^\\\/]/.test(paths[0])) {
          joined = joined.replace(/^[\\\/]{2,}/, "\\");
        }
        return win32.normalize(joined);
      };
      win32.relative = function(from, to) {
        from = win32.resolve(from);
        to = win32.resolve(to);
        var lowerFrom = from.toLowerCase();
        var lowerTo = to.toLowerCase();
        var toParts = trimArray(to.split("\\"));
        var lowerFromParts = trimArray(lowerFrom.split("\\"));
        var lowerToParts = trimArray(lowerTo.split("\\"));
        var length = Math.min(lowerFromParts.length, lowerToParts.length);
        var samePartsLength = length;
        for (var i = 0; i < length; i++) {
          if (lowerFromParts[i] !== lowerToParts[i]) {
            samePartsLength = i;
            break;
          }
        }
        if (samePartsLength == 0) {
          return to;
        }
        var outputParts = [];
        for (var i = samePartsLength; i < lowerFromParts.length; i++) {
          outputParts.push("..");
        }
        outputParts = outputParts.concat(toParts.slice(samePartsLength));
        return outputParts.join("\\");
      };
      win32._makeLong = function(path2) {
        if (!util.isString(path2))
          return path2;
        if (!path2) {
          return "";
        }
        var resolvedPath = win32.resolve(path2);
        if (/^[a-zA-Z]\:\\/.test(resolvedPath)) {
          return "\\\\?\\" + resolvedPath;
        } else if (/^\\\\[^?.]/.test(resolvedPath)) {
          return "\\\\?\\UNC\\" + resolvedPath.substring(2);
        }
        return path2;
      };
      win32.dirname = function(path2) {
        var result = win32SplitPath(path2), root = result[0], dir = result[1];
        if (!root && !dir) {
          return ".";
        }
        if (dir) {
          dir = dir.substr(0, dir.length - 1);
        }
        return root + dir;
      };
      win32.basename = function(path2, ext) {
        var f = win32SplitPath(path2)[2];
        if (ext && f.substr(-1 * ext.length) === ext) {
          f = f.substr(0, f.length - ext.length);
        }
        return f;
      };
      win32.extname = function(path2) {
        return win32SplitPath(path2)[3];
      };
      win32.format = function(pathObject) {
        if (!util.isObject(pathObject)) {
          throw new TypeError(
            "Parameter 'pathObject' must be an object, not " + typeof pathObject
          );
        }
        var root = pathObject.root || "";
        if (!util.isString(root)) {
          throw new TypeError(
            "'pathObject.root' must be a string or undefined, not " + typeof pathObject.root
          );
        }
        var dir = pathObject.dir;
        var base = pathObject.base || "";
        if (!dir) {
          return base;
        }
        if (dir[dir.length - 1] === win32.sep) {
          return dir + base;
        }
        return dir + win32.sep + base;
      };
      win32.parse = function(pathString) {
        if (!util.isString(pathString)) {
          throw new TypeError(
            "Parameter 'pathString' must be a string, not " + typeof pathString
          );
        }
        var allParts = win32SplitPath(pathString);
        if (!allParts || allParts.length !== 4) {
          throw new TypeError("Invalid path '" + pathString + "'");
        }
        return {
          root: allParts[0],
          dir: allParts[0] + allParts[1].slice(0, -1),
          base: allParts[2],
          ext: allParts[3],
          name: allParts[2].slice(0, allParts[2].length - allParts[3].length)
        };
      };
      win32.sep = "\\";
      win32.delimiter = ";";
      var splitPathRe = /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;
      var posix = {};
      function posixSplitPath(filename) {
        return splitPathRe.exec(filename).slice(1);
      }
      posix.resolve = function() {
        var resolvedPath = "", resolvedAbsolute = false;
        for (var i = arguments.length - 1; i >= -1 && !resolvedAbsolute; i--) {
          var path2 = i >= 0 ? arguments[i] : process.cwd();
          if (!util.isString(path2)) {
            throw new TypeError("Arguments to path.resolve must be strings");
          } else if (!path2) {
            continue;
          }
          resolvedPath = path2 + "/" + resolvedPath;
          resolvedAbsolute = path2[0] === "/";
        }
        resolvedPath = normalizeArray(
          resolvedPath.split("/"),
          !resolvedAbsolute
        ).join("/");
        return (resolvedAbsolute ? "/" : "") + resolvedPath || ".";
      };
      posix.normalize = function(path2) {
        var isAbsolute = posix.isAbsolute(path2), trailingSlash = path2 && path2[path2.length - 1] === "/";
        path2 = normalizeArray(path2.split("/"), !isAbsolute).join("/");
        if (!path2 && !isAbsolute) {
          path2 = ".";
        }
        if (path2 && trailingSlash) {
          path2 += "/";
        }
        return (isAbsolute ? "/" : "") + path2;
      };
      posix.isAbsolute = function(path2) {
        return path2.charAt(0) === "/";
      };
      posix.join = function() {
        var path2 = "";
        for (var i = 0; i < arguments.length; i++) {
          var segment = arguments[i];
          if (!util.isString(segment)) {
            throw new TypeError("Arguments to path.join must be strings");
          }
          if (segment) {
            if (!path2) {
              path2 += segment;
            } else {
              path2 += "/" + segment;
            }
          }
        }
        return posix.normalize(path2);
      };
      posix.relative = function(from, to) {
        from = posix.resolve(from).substr(1);
        to = posix.resolve(to).substr(1);
        var fromParts = trimArray(from.split("/"));
        var toParts = trimArray(to.split("/"));
        var length = Math.min(fromParts.length, toParts.length);
        var samePartsLength = length;
        for (var i = 0; i < length; i++) {
          if (fromParts[i] !== toParts[i]) {
            samePartsLength = i;
            break;
          }
        }
        var outputParts = [];
        for (var i = samePartsLength; i < fromParts.length; i++) {
          outputParts.push("..");
        }
        outputParts = outputParts.concat(toParts.slice(samePartsLength));
        return outputParts.join("/");
      };
      posix._makeLong = function(path2) {
        return path2;
      };
      posix.dirname = function(path2) {
        var result = posixSplitPath(path2), root = result[0], dir = result[1];
        if (!root && !dir) {
          return ".";
        }
        if (dir) {
          dir = dir.substr(0, dir.length - 1);
        }
        return root + dir;
      };
      posix.basename = function(path2, ext) {
        var f = posixSplitPath(path2)[2];
        if (ext && f.substr(-1 * ext.length) === ext) {
          f = f.substr(0, f.length - ext.length);
        }
        return f;
      };
      posix.extname = function(path2) {
        return posixSplitPath(path2)[3];
      };
      posix.format = function(pathObject) {
        if (!util.isObject(pathObject)) {
          throw new TypeError(
            "Parameter 'pathObject' must be an object, not " + typeof pathObject
          );
        }
        var root = pathObject.root || "";
        if (!util.isString(root)) {
          throw new TypeError(
            "'pathObject.root' must be a string or undefined, not " + typeof pathObject.root
          );
        }
        var dir = pathObject.dir ? pathObject.dir + posix.sep : "";
        var base = pathObject.base || "";
        return dir + base;
      };
      posix.parse = function(pathString) {
        if (!util.isString(pathString)) {
          throw new TypeError(
            "Parameter 'pathString' must be a string, not " + typeof pathString
          );
        }
        var allParts = posixSplitPath(pathString);
        if (!allParts || allParts.length !== 4) {
          throw new TypeError("Invalid path '" + pathString + "'");
        }
        allParts[1] = allParts[1] || "";
        allParts[2] = allParts[2] || "";
        allParts[3] = allParts[3] || "";
        return {
          root: allParts[0],
          dir: allParts[0] + allParts[1].slice(0, -1),
          base: allParts[2],
          ext: allParts[3],
          name: allParts[2].slice(0, allParts[2].length - allParts[3].length)
        };
      };
      posix.sep = "/";
      posix.delimiter = ":";
      if (isWindows)
        module.exports = win32;
      else
        module.exports = posix;
      module.exports.posix = posix;
      module.exports.win32 = win32;
    }
  });

  // src/multithreading/workers/node/testworker.ts
  var testworker_exports = {};
  __export(testworker_exports, {
    TestWorker: () => TestWorker,
    default: () => testworker_default
  });
  var import_child_process, import_path, TestWorker, testworker_default;
  var init_testworker = __esm({
    "src/multithreading/workers/node/testworker.ts"() {
      "use strict";
      import_child_process = __require("child_process");
      import_path = __toESM(require_path(), 1);
      TestWorker = class {
        worker;
        /**
         * Creates a new TestWorker instance.
         *
         * This initializes a new worker process and sends the dataset and cost function
         * to the worker for further processing.
         *
         * @param {number[]} dataSet - The serialized dataset to be used by the worker.
         * @param {{ name: string }} cost - The cost function to evaluate the network.
         */
        constructor(dataSet, cost) {
          this.worker = (0, import_child_process.fork)(import_path.default.join(__dirname, "/worker"));
          this.worker.send({ set: dataSet, cost: cost.name });
        }
        /**
         * Evaluates a neural network using the worker process.
         *
         * The network is serialized and sent to the worker for evaluation. The worker
         * sends back the evaluation result, which is returned as a promise.
         *
         * @param {any} network - The neural network to evaluate. It must implement a `serialize` method.
         * @returns {Promise<number>} A promise that resolves to the evaluation result.
         */
        evaluate(network) {
          return new Promise((resolve) => {
            const serialized = network.serialize();
            const data = {
              activations: serialized[0],
              states: serialized[1],
              conns: serialized[2]
            };
            const _that = this.worker;
            this.worker.on("message", function callback(e) {
              _that.removeListener("message", callback);
              resolve(e);
            });
            this.worker.send(data);
          });
        }
        /**
         * Terminates the worker process.
         *
         * This method ensures that the worker process is properly terminated to free up system resources.
         */
        terminate() {
          this.worker.kill();
        }
      };
      testworker_default = TestWorker;
    }
  });

  // src/multithreading/workers/browser/testworker.ts
  var testworker_exports2 = {};
  __export(testworker_exports2, {
    TestWorker: () => TestWorker2
  });
  var TestWorker2;
  var init_testworker2 = __esm({
    "src/multithreading/workers/browser/testworker.ts"() {
      "use strict";
      init_multi();
      TestWorker2 = class _TestWorker {
        worker;
        url;
        /**
         * Creates a new TestWorker instance.
         * @param {number[]} dataSet - The serialized dataset to be used by the worker.
         * @param {any} cost - The cost function to evaluate the network.
         */
        constructor(dataSet, cost) {
          const blob = new Blob([_TestWorker._createBlobString(cost)]);
          this.url = window.URL.createObjectURL(blob);
          this.worker = new Worker(this.url);
          const data = { set: new Float64Array(dataSet).buffer };
          this.worker.postMessage(data, [data.set]);
        }
        /**
         * Evaluates a network using the worker process.
         * @param {any} network - The network to evaluate.
         * @returns {Promise<number>} A promise that resolves to the evaluation result.
         */
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
        /**
         * Terminates the worker process and revokes the object URL.
         */
        terminate() {
          this.worker.terminate();
          window.URL.revokeObjectURL(this.url);
        }
        /**
         * Creates a string representation of the worker's blob.
         * @param {any} cost - The cost function to be used by the worker.
         * @returns {string} The blob string.
         */
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

  // src/multithreading/workers/workers.ts
  var Workers;
  var init_workers = __esm({
    "src/multithreading/workers/workers.ts"() {
      "use strict";
      Workers = class {
        /**
         * Loads the Node.js test worker dynamically.
         * @returns {Promise<any>} A promise that resolves to the Node.js TestWorker class.
         */
        static async getNodeTestWorker() {
          const module = await Promise.resolve().then(() => (init_testworker(), testworker_exports));
          return module.TestWorker;
        }
        /**
         * Loads the browser test worker dynamically.
         * @returns {Promise<any>} A promise that resolves to the browser TestWorker class.
         */
        static async getBrowserTestWorker() {
          const module = await Promise.resolve().then(() => (init_testworker2(), testworker_exports2));
          return module.TestWorker;
        }
      };
    }
  });

  // src/multithreading/multi.ts
  var Multi;
  var init_multi = __esm({
    "src/multithreading/multi.ts"() {
      "use strict";
      init_workers();
      init_network();
      Multi = class _Multi {
        /** Workers for multi-threading */
        static workers = Workers;
        /**
         * A list of compiled activation functions in a specific order.
         */
        static activations = [
          (x) => 1 / (1 + Math.exp(-x)),
          // Logistic (0)
          (x) => Math.tanh(x),
          // Tanh (1)
          (x) => x,
          // Identity (2)
          (x) => x > 0 ? 1 : 0,
          // Step (3)
          (x) => x > 0 ? x : 0,
          // ReLU (4)
          (x) => x / (1 + Math.abs(x)),
          // Softsign (5)
          (x) => Math.sin(x),
          // Sinusoid (6)
          (x) => Math.exp(-Math.pow(x, 2)),
          // Gaussian (7)
          (x) => (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x,
          // Bent Identity (8)
          (x) => x > 0 ? 1 : -1,
          // Bipolar (9)
          (x) => 2 / (1 + Math.exp(-x)) - 1,
          // Bipolar Sigmoid (10)
          (x) => Math.max(-1, Math.min(1, x)),
          // Hard Tanh (11)
          (x) => Math.abs(x),
          // Absolute (12)
          (x) => 1 - x,
          // Inverse (13)
          (x) => {
            const alpha = 1.6732632423543772;
            const scale = 1.0507009873554805;
            const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
            return fx * scale;
          },
          (x) => Math.log(1 + Math.exp(x))
          // Softplus (15) - Added
        ];
        /**
         * Serializes a dataset into a flat array.
         * @param {Array<{ input: number[]; output: number[] }>} dataSet - The dataset to serialize.
         * @returns {number[]} The serialized dataset.
         */
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
        /**
         * Activates a serialized network.
         * @param {number[]} input - The input values.
         * @param {number[]} A - The activations array.
         * @param {number[]} S - The states array.
         * @param {number[]} data - The serialized network data.
         * @param {Function[]} F - The activation functions.
         * @returns {number[]} The output values.
         */
        static activateSerializedNetwork(input, A, S, data, F) {
          for (let i = 0; i < data[0]; i++) A[i] = input[i];
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
          for (let i = A.length - data[1]; i < A.length; i++) output.push(A[i]);
          return output;
        }
        /**
         * Deserializes a dataset from a flat array.
         * @param {number[]} serializedSet - The serialized dataset.
         * @returns {Array<{ input: number[]; output: number[] }>} The deserialized dataset as an array of input-output pairs.
         */
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
        /**
         * Logistic activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static logistic(x) {
          return 1 / (1 + Math.exp(-x));
        }
        /**
         * Hyperbolic tangent activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static tanh(x) {
          return Math.tanh(x);
        }
        /**
         * Identity activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static identity(x) {
          return x;
        }
        /**
         * Step activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static step(x) {
          return x > 0 ? 1 : 0;
        }
        /**
         * Rectified Linear Unit (ReLU) activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static relu(x) {
          return x > 0 ? x : 0;
        }
        /**
         * Softsign activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static softsign(x) {
          return x / (1 + Math.abs(x));
        }
        /**
         * Sinusoid activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static sinusoid(x) {
          return Math.sin(x);
        }
        /**
         * Gaussian activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static gaussian(x) {
          return Math.exp(-Math.pow(x, 2));
        }
        /**
         * Bent Identity activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static bentIdentity(x) {
          return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
        }
        /**
         * Bipolar activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static bipolar(x) {
          return x > 0 ? 1 : -1;
        }
        /**
         * Bipolar Sigmoid activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static bipolarSigmoid(x) {
          return 2 / (1 + Math.exp(-x)) - 1;
        }
        /**
         * Hard Tanh activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static hardTanh(x) {
          return Math.max(-1, Math.min(1, x));
        }
        /**
         * Absolute activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static absolute(x) {
          return Math.abs(x);
        }
        /**
         * Inverse activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static inverse(x) {
          return 1 - x;
        }
        /**
         * Scaled Exponential Linear Unit (SELU) activation function.
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static selu(x) {
          const alpha = 1.6732632423543772;
          const scale = 1.0507009873554805;
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          return fx * scale;
        }
        /**
         * Softplus activation function. - Added
         * @param {number} x - The input value.
         * @returns {number} The activated value.
         */
        static softplus(x) {
          return Math.log(1 + Math.exp(x));
        }
        /**
         * Tests a serialized dataset using a cost function.
         * @param {Array<{ input: number[]; output: number[] }>} set - The serialized dataset as an array of input-output pairs.
         * @param {Function} cost - The cost function.
         * @param {number[]} A - The activations array.
         * @param {number[]} S - The states array.
         * @param {number[]} data - The serialized network data.
         * @param {Function[]} F - The activation functions.
         * @returns {number} The average error.
         */
        static testSerializedSet(set, cost, A, S, data, F) {
          let error = 0;
          for (let i = 0; i < set.length; i++) {
            const output = _Multi.activateSerializedNetwork(
              set[i].input,
              A,
              S,
              data,
              F
            );
            error += cost(set[i].output, output);
          }
          return error / set.length;
        }
        /**
         * Gets the browser test worker.
         * @returns {Promise<any>} The browser test worker.
         */
        static async getBrowserTestWorker() {
          const { TestWorker: TestWorker3 } = await Promise.resolve().then(() => (init_testworker2(), testworker_exports2));
          return TestWorker3;
        }
        /**
         * Gets the node test worker.
         * @returns {Promise<any>} The node test worker.
         */
        static async getNodeTestWorker() {
          const { TestWorker: TestWorker3 } = await Promise.resolve().then(() => (init_testworker(), testworker_exports));
          return TestWorker3;
        }
      };
    }
  });

  // src/architecture/activationArrayPool.ts
  var ActivationArrayPool, activationArrayPool;
  var init_activationArrayPool = __esm({
    "src/architecture/activationArrayPool.ts"() {
      "use strict";
      init_config();
      ActivationArrayPool = class {
        /** Buckets keyed by length, storing reusable arrays. */
        buckets = /* @__PURE__ */ new Map();
        /** Count of arrays created since last clear(), for diagnostics. */
        created = 0;
        /** Count of successful reuses since last clear(), for diagnostics. */
        reused = 0;
        /** Max arrays retained per size bucket; Infinity by default. */
        maxPerBucket = Number.POSITIVE_INFINITY;
        /**
         * Acquire an activation array of fixed length.
         * Zero-fills reused arrays to guarantee clean state.
         *
         * @param size Required array length.
         * @returns Zeroed activation array of the requested size.
         */
        acquire(size) {
          const bucket = this.buckets.get(size);
          if (bucket && bucket.length > 0) {
            this.reused++;
            const arr = bucket.pop();
            arr.fill(0);
            return arr;
          }
          this.created++;
          return config.float32Mode ? new Float32Array(size) : new Array(size).fill(0);
        }
        /**
         * Return an activation array to the pool. If the bucket is full per
         * `maxPerBucket`, the array is dropped and left to GC.
         *
         * @param array Array to release back to the pool.
         */
        release(array) {
          const size = array.length >>> 0;
          if (!this.buckets.has(size)) this.buckets.set(size, []);
          const bucket = this.buckets.get(size);
          if (bucket.length < this.maxPerBucket) bucket.push(array);
        }
        /**
         * Clear all buckets and reset counters. Frees references to pooled arrays.
         */
        clear() {
          this.buckets.clear();
          this.created = 0;
          this.reused = 0;
        }
        /**
         * Snapshot of diagnostics: creations, reuses, and number of active buckets.
         */
        stats() {
          return {
            created: this.created,
            reused: this.reused,
            bucketCount: this.buckets.size
          };
        }
        /**
         * Configure a capacity cap per size bucket to avoid unbounded memory growth.
         *
         * @param cap Non-negative capacity per bucket (Infinity allowed).
         */
        setMaxPerBucket(cap) {
          if (typeof cap === "number" && cap >= 0) this.maxPerBucket = cap;
        }
        /**
         * Pre-allocate and retain arrays for a given size bucket up to `count` items.
         *
         * @param size Array length (bucket key).
         * @param count Number of arrays to prepare (rounded down, min 0).
         */
        prewarm(size, count) {
          const n = Math.max(0, Math.floor(count));
          if (!this.buckets.has(size)) this.buckets.set(size, []);
          const bucket = this.buckets.get(size);
          for (let i = 0; i < n && bucket.length < this.maxPerBucket; i++) {
            const arr = config.float32Mode ? new Float32Array(size) : new Array(size).fill(0);
            bucket.push(arr);
            this.created++;
          }
        }
        /**
         * Current retained count for a size bucket.
         *
         * @param size Array length (bucket key).
         * @returns Number of arrays available to reuse for that length.
         */
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
        version: "0.1.10",
        description: "Architecture-free neural network library with genetic algorithm implementations",
        main: "./dist/neataptic.js",
        module: "./dist/neataptic.js",
        types: "./dist/neataptic.d.ts",
        type: "module",
        scripts: {
          test: "jest --config=jest.config.mjs --no-cache --coverage --collect-coverage --runInBand --testPathIgnorePatterns=.e2e.test.ts --verbose",
          pretest: "npm run build",
          "test:bench": "jest --no-cache --runInBand --verbose --testPathPattern=benchmark",
          "test:silent": "jest --no-cache --coverage --collect-coverage --runInBand --testPathIgnorePatterns=.e2e.test.ts --silent",
          deploy: "npm run build && npm run test:dist && npm publish",
          build: "npm run build:webpack && npm run build:ts",
          "build:ts": "tsc",
          "build:webpack": "webpack --config webpack.config.js",
          "start:ts": "ts-node src/neataptic.ts",
          "test:e2e": "cross-env FORCE_COLOR=true jest e2e.test.ts --no-cache --runInBand",
          "test:e2e:logs": "npx jest e2e.test.ts --verbose --runInBand --no-cache",
          "test:dist": "npm run build:ts && jest --no-cache --coverage --collect-coverage --runInBand --testPathIgnorePatterns=.e2e.test.ts",
          "docs:build-scripts": "tsc -p tsconfig.docs.json && node scripts/write-dist-docs-pkg.mjs",
          "docs:folders": "npm run docs:build-scripts && node ./dist-docs/scripts/generate-docs.js",
          "docs:html": "npm run docs:build-scripts && node ./dist-docs/scripts/render-docs-html.js",
          "build:ascii-maze": "npx esbuild test/examples/asciiMaze/browser-entry.ts --bundle --outfile=docs/assets/ascii-maze.bundle.js --platform=browser --format=iife --sourcemap --external:fs --external:child_process",
          "docs:examples": "node scripts/copy-examples.mjs",
          prettier: "npm run prettier:tests && npm run prettier:src",
          "prettier:tests": "npx prettier --write **/*.test.ts",
          "prettier:src": "npx prettier --write src/**/*.ts",
          docs: "npm run build:ascii-maze && npm run docs:examples && npm run docs:build-scripts && node ./dist-docs/scripts/generate-docs.js && node ./dist-docs/scripts/render-docs-html.js",
          "onnx:export": "node scripts/export-onnx.mjs"
        },
        exports: {
          ".": {
            types: "./dist/neataptic.d.ts",
            import: "./dist/neataptic.js"
          }
        },
        devDependencies: {
          "@types/chai": "^5.2.1",
          "@types/fs-extra": "^11.0.4",
          "@types/jest": "^29.5.11",
          "@types/node": "^20.19.10",
          "@types/seedrandom": "^3.0.8",
          "@types/webpack": "^5.28.5",
          "@types/webpack-dev-server": "^4.7.2",
          chai: "^4.3.4",
          "copy-webpack-plugin": "^8.1.0",
          "cross-env": "^7.0.3",
          "fast-glob": "^3.3.3",
          "fs-extra": "^11.3.1",
          husky: "^6.0.0",
          jest: "^29.7.0",
          "jsdoc-to-markdown": "^9.1.1",
          marked: "^12.0.2",
          mkdocs: "^0.0.1",
          "ts-jest": "^29.1.1",
          "ts-loader": "^9.5.2",
          "ts-morph": "^22.0.0",
          "ts-node": "^10.9.2",
          typescript: "^5.6.3",
          "undici-types": "^7.8.0",
          webpack: "^5.99.5",
          "webpack-cli": "^6.0.1",
          esbuild: "^0.23.0",
          puppeteer: "^23.3.0"
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
          node: ">=20.0.0"
        },
        prettier: {
          singleQuote: true
        },
        dependencies: {
          seedrandom: "^3.0.5",
          undici: "^5.0.0"
        }
      };
    }
  });

  // src/architecture/network/network.onnx.ts
  function rebuildConnectionsLocal(networkLike) {
    const uniqueConnections = /* @__PURE__ */ new Set();
    networkLike.nodes.forEach(
      (node) => node.connections?.out.forEach((conn) => uniqueConnections.add(conn))
    );
    networkLike.connections = Array.from(uniqueConnections);
  }
  function mapActivationToOnnx(squash) {
    const upperName = (squash?.name || "").toUpperCase();
    if (upperName.includes("TANH")) return "Tanh";
    if (upperName.includes("LOGISTIC") || upperName.includes("SIGMOID"))
      return "Sigmoid";
    if (upperName.includes("RELU")) return "Relu";
    if (squash)
      console.warn(
        `Unsupported activation function ${squash.name} for ONNX export, defaulting to Identity.`
      );
    return "Identity";
  }
  function inferLayerOrdering(network) {
    const inputNodes = network.nodes.filter((n) => n.type === "input");
    const outputNodes = network.nodes.filter((n) => n.type === "output");
    const hiddenNodes = network.nodes.filter((n) => n.type === "hidden");
    if (hiddenNodes.length === 0) return [inputNodes, outputNodes];
    let remainingHidden = [...hiddenNodes];
    let previousLayer = inputNodes;
    const layerAccumulator = [];
    while (remainingHidden.length) {
      const currentLayer = remainingHidden.filter(
        (hidden) => hidden.connections.in.every(
          (conn) => previousLayer.includes(conn.from)
        )
      );
      if (!currentLayer.length)
        throw new Error(
          "Invalid network structure for ONNX export: cannot resolve layered ordering."
        );
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
      const activationNameSet = new Set(
        currentLayerNodes.map((n) => n.squash && n.squash.name)
      );
      if (activationNameSet.size > 1 && !options.allowMixedActivations)
        throw new Error(
          `ONNX export error: Mixed activation functions detected in layer ${layerIndex}. (enable allowMixedActivations to decompose layer)`
        );
      if (activationNameSet.size > 1 && options.allowMixedActivations)
        console.warn(
          `Warning: Mixed activations in layer ${layerIndex}; exporting per-neuron Gemm + Activation (+Concat) baseline.`
        );
      for (const targetNode of currentLayerNodes) {
        for (const sourceNode of previousLayerNodes) {
          const isConnected = targetNode.connections.in.some(
            (conn) => conn.from === sourceNode
          );
          if (!isConnected && !options.allowPartialConnectivity)
            throw new Error(
              `ONNX export error: Missing connection from node ${sourceNode.index} to node ${targetNode.index} in layer ${layerIndex}. (enable allowPartialConnectivity)`
            );
        }
      }
    }
  }
  function buildOnnxModel(network, layers, options = {}) {
    const {
      includeMetadata = false,
      opset = 18,
      batchDimension = false,
      legacyNodeOrdering = false,
      producerName = "neataptic-ts",
      producerVersion,
      docString
    } = options;
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
      if (!isOutputLayer) hiddenSizesMetadata.push(currentLayerNodes.length);
      const convSpec = options.conv2dMappings?.find(
        (m) => m.layerIndex === layerIndex
      );
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
          console.warn(
            `Conv2D mapping for layer ${layerIndex} skipped: dimension mismatch (expected prev=${prevWidthExpected} got ${prevWidthActual}; expected this=${thisWidthExpected} got ${thisWidthActual}).`
          );
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
                  const conn = repNeuron.connections.in.find(
                    (cc) => cc.from === sourceNode
                  );
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
          const poolSpecPostConv = options.pool2dMappings?.find(
            (p) => p.afterLayerIndex === layerIndex
          );
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
              const flMeta = model.metadata_props.find(
                (m) => m.key === "flatten_layers"
              );
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
            const poolLayersMeta = model.metadata_props.find(
              (m) => m.key === "pool2d_layers"
            );
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
            const poolSpecsMeta = model.metadata_props.find(
              (m) => m.key === "pool2d_specs"
            );
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
          const convLayersMeta = model.metadata_props.find(
            (m) => m.key === "conv2d_layers"
          );
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
          const convSpecsMeta = model.metadata_props.find(
            (m) => m.key === "conv2d_specs"
          );
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
          throw new Error(
            `Recurrent export does not yet support mixed activations in hidden layer ${layerIndex}.`
          );
        const weightMatrixValues = [];
        const biasVector = new Array(currentLayerNodes.length).fill(0);
        for (let r = 0; r < currentLayerNodes.length; r++) {
          const targetNode = currentLayerNodes[r];
          biasVector[r] = targetNode.bias;
          for (let c = 0; c < previousLayerNodes.length; c++) {
            const sourceNode = previousLayerNodes[c];
            const inboundConn = targetNode.connections.in.find(
              (conn) => conn.from === sourceNode
            );
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
            const inboundConn = targetNode.connections.in.find(
              (conn) => conn.from === sourceNode
            );
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
        const poolSpecDense = options.pool2dMappings?.find(
          (p) => p.afterLayerIndex === layerIndex
        );
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
            const flMeta = model.metadata_props.find(
              (m) => m.key === "flatten_layers"
            );
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
          const poolLayersMeta = model.metadata_props.find(
            (m) => m.key === "pool2d_layers"
          );
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
          const poolSpecsMeta = model.metadata_props.find(
            (m) => m.key === "pool2d_specs"
          );
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
            const inboundConn = targetNode.connections.in.find(
              (conn) => conn.from === sourceNode
            );
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
        const poolSpecPerNeuron = options.pool2dMappings?.find(
          (p) => p.afterLayerIndex === layerIndex
        );
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
            const flMeta = model.metadata_props.find(
              (m) => m.key === "flatten_layers"
            );
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
          const poolLayersMeta = model.metadata_props.find(
            (m) => m.key === "pool2d_layers"
          );
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
          const poolSpecsMeta = model.metadata_props.find(
            (m) => m.key === "pool2d_specs"
          );
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
        if (!model.metadata_props) model.metadata_props = [];
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
                const conn = neuron.connections.in.find(
                  (cc) => cc.from === source
                );
                W.push(conn ? conn.weight : 0);
              }
              for (let c = 0; c < unit; c++) {
                if (gate2 === cell && c === r) {
                  const selfConn = neuron.connections.self[0];
                  R.push(selfConn ? selfConn.weight : 0);
                } else R.push(0);
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
          const lstmMetaIdx = model.metadata_props.findIndex(
            (m) => m.key === "lstm_emitted_layers"
          );
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
                const conn = neuron.connections.in.find(
                  (cc) => cc.from === src
                );
                Wg.push(conn ? conn.weight : 0);
              }
              for (let c = 0; c < unitG; c++) {
                if (gate2 === candidate && c === r) {
                  const selfConn = neuron.connections.self[0];
                  Rg.push(selfConn ? selfConn.weight : 0);
                } else Rg.push(0);
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
          const gruMetaIdx = model.metadata_props.findIndex(
            (m) => m.key === "gru_emitted_layers"
          );
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
          if (!layerNodes || !prevLayerNodes) continue;
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
                  const conn = repNeuron.connections.in.find(
                    (cc) => cc.from === sourceNode
                  );
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
                if (!neuron) continue;
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
                      const conn = neuron.connections.in.find(
                        (cc) => cc.from === srcNode
                      );
                      const wVal = conn ? conn.weight : 0;
                      if (Math.abs(wVal - repPerChannel[oc][kPtr]) > tol) {
                        allOk = false;
                      }
                      kPtr++;
                    }
                  }
                }
                if (!allOk) break;
              }
            }
          }
          if (allOk) verified.push(layerIdx);
          else {
            mismatched.push(layerIdx);
            console.warn(
              `Conv2D weight sharing mismatch detected in layer ${layerIdx}`
            );
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
            const allSelf = memorySlice.every(
              (n) => n.connections.self.length === 1
            );
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
        if (Math.abs(s - Math.round(s)) > 1e-9) continue;
        const sInt = Math.round(s);
        for (const k of [3, 2]) {
          if (k >= sInt) continue;
          const outSpatial = sInt - k + 1;
          if (outSpatial * outSpatial === currWidth) {
            const alreadyDeclared = options.conv2dMappings?.some(
              (m) => m.layerIndex === li
            );
            if (alreadyDeclared) break;
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
    "src/architecture/network/network.onnx.ts"() {
      "use strict";
      init_methods();
      init_connection();
    }
  });

  // src/architecture/onnx.ts
  var init_onnx = __esm({
    "src/architecture/onnx.ts"() {
      "use strict";
      init_network_onnx();
      init_network_onnx();
    }
  });

  // src/architecture/network/network.standalone.ts
  function generateStandalone(net) {
    if (!net.nodes.some((nodeRef) => nodeRef.type === "output")) {
      throw new Error(
        "Cannot create standalone function: network has no output nodes."
      );
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
            functionSource = `function ${squashName}${functionSource.substring(
              functionSource.indexOf("(")
            )}`;
          }
          functionSource = stripCoverage(functionSource);
        } else {
          functionSource = squashFn.toString();
          functionSource = stripCoverage(functionSource);
          if (functionSource.startsWith("function")) {
            functionSource = `function ${squashName}${functionSource.substring(
              functionSource.indexOf("(")
            )}`;
          } else if (functionSource.includes("=>")) {
            functionSource = `function ${squashName}${functionSource.substring(
              functionSource.indexOf("(")
            )}`;
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
        if (typeof connection.from.index === "undefined") continue;
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
      bodyLines.push(
        `A[${nodeIndex}] = F[${activationFunctionIndex}](S[${nodeIndex}])${maskValue !== 1 ? ` * ${maskValue}` : ""};`
      );
    }
    const outputIndices = [];
    for (let nodeIndex = net.nodes.length - net.output; nodeIndex < net.nodes.length; nodeIndex++) {
      if (typeof net.nodes[nodeIndex]?.index !== "undefined") {
        outputIndices.push(net.nodes[nodeIndex].index);
      }
    }
    bodyLines.push(
      `return [${outputIndices.map((idx) => `A[${idx}]`).join(",")}];`
    );
    const activationArrayLiteral = Object.entries(activationFunctionIndexMap).sort(([, a], [, b]) => a - b).map(([name]) => name).join(",");
    const activationArrayType = net._activationPrecision === "f32" ? "Float32Array" : "Float64Array";
    let generatedSource = "";
    generatedSource += `(function(){
`;
    generatedSource += `${activationFunctionSources.join("\n")}
`;
    generatedSource += `var F = [${activationArrayLiteral}];
`;
    generatedSource += `var A = new ${activationArrayType}([${initialActivations.join(
      ","
    )}]);
`;
    generatedSource += `var S = new ${activationArrayType}([${initialStates.join(
      ","
    )}]);
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
    "src/architecture/network/network.standalone.ts"() {
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

  // src/architecture/network/network.topology.ts
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
        if (outgoing.to === node) continue;
        const remaining = (inDegree.get(outgoing.to) || 0) - 1;
        inDegree.set(outgoing.to, remaining);
        if (remaining === 0) processingQueue.push(outgoing.to);
      }
    }
    internalNet._topoOrder = topoOrder.length === this.nodes.length ? topoOrder : this.nodes.slice();
    internalNet._topoDirty = false;
  }
  function hasPath(from, to) {
    if (from === to) return true;
    const visited = /* @__PURE__ */ new Set();
    const dfsStack = [from];
    while (dfsStack.length) {
      const current = dfsStack.pop();
      if (current === to) return true;
      if (visited.has(current)) continue;
      visited.add(current);
      for (const edge of current.connections.out) {
        if (edge.to !== current) dfsStack.push(edge.to);
      }
    }
    return false;
  }
  var init_network_topology = __esm({
    "src/architecture/network/network.topology.ts"() {
      "use strict";
    }
  });

  // src/architecture/network/network.slab.ts
  function _slabPoolCap() {
    const configuredCap = config.slabPoolMaxPerKey;
    if (configuredCap === void 0) return 4;
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
    if (!config.enableSlabArrayPooling) return;
    const key = _poolKey(kind, bytesPerElement, arr.length);
    const list = _slabArrayPool[key] ||= [];
    if (list.length < _slabPoolCap()) list.push(arr);
    const m = _slabPoolMetrics[key] ||= {
      created: 0,
      reused: 0,
      maxRetained: 0
    };
    if (list.length > m.maxRetained) m.maxRetained = list.length;
  }
  function rebuildConnectionSlab(force = false) {
    const internalNet = this;
    if (!force && !internalNet._slabDirty) return;
    if (internalNet._nodeIndexDirty) _reindexNodes.call(this);
    const connectionCount = this.connections.length;
    let capacity = internalNet._connCapacity || 0;
    const growthFactor = typeof window === "undefined" ? 1.75 : 1.25;
    const needAllocate = capacity < connectionCount;
    if (needAllocate) {
      capacity = capacity === 0 ? Math.ceil(connectionCount * growthFactor) : capacity;
      while (capacity < connectionCount)
        capacity = Math.ceil(capacity * growthFactor);
      if (internalNet._connWeights)
        _releaseTA(
          "w",
          internalNet._useFloat32Weights ? 4 : 8,
          internalNet._connWeights
        );
      if (internalNet._connFrom)
        _releaseTA("f", 4, internalNet._connFrom);
      if (internalNet._connTo)
        _releaseTA("t", 4, internalNet._connTo);
      if (internalNet._connFlags)
        _releaseTA("fl", 1, internalNet._connFlags);
      if (internalNet._connGain)
        _releaseTA(
          "g",
          internalNet._useFloat32Weights ? 4 : 8,
          internalNet._connGain
        );
      if (internalNet._connPlastic)
        _releaseTA(
          "p",
          internalNet._useFloat32Weights ? 4 : 8,
          internalNet._connPlastic
        );
      internalNet._connWeights = _acquireTA(
        "w",
        internalNet._useFloat32Weights ? Float32Array : Float64Array,
        capacity,
        internalNet._useFloat32Weights ? 4 : 8
      );
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
          gainArray = _acquireTA(
            "g",
            internalNet._useFloat32Weights ? Float32Array : Float64Array,
            capacity,
            internalNet._useFloat32Weights ? 4 : 8
          );
          internalNet._connGain = gainArray;
          for (let j = 0; j < connectionIndex; j++) gainArray[j] = 1;
        }
        gainArray[connectionIndex] = gainValue;
        anyNonNeutralGain = true;
      } else if (gainArray) {
        gainArray[connectionIndex] = 1;
      }
      if (connection._flags & 8) anyPlastic = true;
    }
    if (!anyNonNeutralGain && gainArray) {
      _releaseTA(
        "g",
        internalNet._useFloat32Weights ? 4 : 8,
        gainArray
      );
      internalNet._connGain = null;
    }
    if (anyPlastic && !plasticArray) {
      plasticArray = _acquireTA(
        "p",
        internalNet._useFloat32Weights ? Float32Array : Float64Array,
        capacity,
        internalNet._useFloat32Weights ? 4 : 8
      );
      internalNet._connPlastic = plasticArray;
      for (let i = 0; i < connectionCount; i++) {
        const c = this.connections[i];
        plasticArray[i] = c.plasticityRate || 0;
      }
    } else if (!anyPlastic && plasticArray) {
      _releaseTA(
        "p",
        internalNet._useFloat32Weights ? 4 : 8,
        plasticArray
      );
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
      for (let i = 0; i < (internalNet._connCount || 0); i++) gain[i] = 1;
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
    if (!internalNet._connFrom || !internalNet._connTo) return;
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
    return !training && // Training may require gradients / noise injection.
    internalNet._enforceAcyclic && // Must have acyclic guarantee for single forward sweep.
    !internalNet._topoDirty && // Topological order must be current.
    this.gates.length === 0 && // Gating implies dynamic per-edge behavior.
    this.selfconns.length === 0 && // Self connections require recurrent handling.
    this.dropout === 0 && // Dropout introduces stochastic masking.
    internalNet._weightNoiseStd === 0 && // Global weight noise disables deterministic slab pass.
    internalNet._weightNoisePerHidden.length === 0 && // Per hidden noise variants.
    internalNet._stochasticDepth.length === 0;
  }
  function fastSlabActivate(input) {
    const internalNet = this;
    rebuildConnectionSlab.call(this);
    if (internalNet._adjDirty) _buildAdjacency.call(this);
    if (this.gates && this.gates.length > 0)
      return this.activate(input, false);
    if (!internalNet._connWeights || !internalNet._connFrom || !internalNet._connTo || !internalNet._outStart || !internalNet._outOrder) {
      return this.activate(input, false);
    }
    if (internalNet._topoDirty) this._computeTopoOrder();
    if (internalNet._nodeIndexDirty) _reindexNodes.call(this);
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
        if (gainArr) w *= gainArr[connectionIndex];
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
    "src/architecture/network/network.slab.ts"() {
      "use strict";
      init_activationArrayPool();
      init_config();
      _slabArrayPool = /* @__PURE__ */ Object.create(null);
      _slabPoolMetrics = /* @__PURE__ */ Object.create(null);
      _slabAllocStats = { fresh: 0, pooled: 0 };
    }
  });

  // src/architecture/network/network.prune.ts
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
      if (!fromNode || !toNode || fromNode === toNode) continue;
      if (network.connections.some((c) => c.from === fromNode && c.to === toNode))
        continue;
      if (netAny._enforceAcyclic && network.nodes.indexOf(fromNode) > network.nodes.indexOf(toNode))
        continue;
      network.connect(fromNode, toNode);
    }
  }
  function maybePrune(iteration) {
    const cfg = this._pruningConfig;
    if (!cfg) return;
    if (iteration < cfg.start || iteration > cfg.end) return;
    if (cfg.lastPruneIter != null && iteration === cfg.lastPruneIter) return;
    if ((iteration - cfg.start) % (cfg.frequency || 1) !== 0) return;
    const initialConnectionBaseline = this._initialConnectionCount;
    if (!initialConnectionBaseline) return;
    const progressFraction = (iteration - cfg.start) / Math.max(1, cfg.end - cfg.start);
    const targetSparsityNow = cfg.targetSparsity * Math.min(1, Math.max(0, progressFraction));
    const desiredRemainingConnections = Math.max(
      1,
      Math.floor(initialConnectionBaseline * (1 - targetSparsityNow))
    );
    const excessConnectionCount = this.connections.length - desiredRemainingConnections;
    if (excessConnectionCount <= 0) {
      cfg.lastPruneIter = iteration;
      return;
    }
    const rankedConnections = rankConnections(
      this.connections,
      cfg.method || "magnitude"
    );
    const connectionsToPrune = rankedConnections.slice(0, excessConnectionCount);
    connectionsToPrune.forEach((conn) => this.disconnect(conn.from, conn.to));
    if (cfg.regrowFraction && cfg.regrowFraction > 0) {
      const intendedRegrowCount = Math.floor(
        connectionsToPrune.length * cfg.regrowFraction
      );
      regrowConnections(
        this,
        desiredRemainingConnections,
        intendedRegrowCount * 10
      );
    }
    cfg.lastPruneIter = iteration;
    this._topoDirty = true;
  }
  function pruneToSparsity(targetSparsity, method = "magnitude") {
    if (targetSparsity <= 0) return;
    if (targetSparsity >= 1) targetSparsity = 0.999;
    const netAny = this;
    if (!netAny._evoInitialConnCount)
      netAny._evoInitialConnCount = this.connections.length;
    const evolutionaryBaseline = netAny._evoInitialConnCount;
    const desiredRemainingConnections = Math.max(
      1,
      Math.floor(evolutionaryBaseline * (1 - targetSparsity))
    );
    const excessConnectionCount = this.connections.length - desiredRemainingConnections;
    if (excessConnectionCount <= 0) return;
    const rankedConnections = rankConnections(this.connections, method);
    const connectionsToRemove = rankedConnections.slice(0, excessConnectionCount);
    connectionsToRemove.forEach((c) => this.disconnect(c.from, c.to));
    netAny._topoDirty = true;
  }
  function getCurrentSparsity() {
    const initialBaseline = this._initialConnectionCount;
    if (!initialBaseline) return 0;
    return 1 - this.connections.length / initialBaseline;
  }
  var init_network_prune = __esm({
    "src/architecture/network/network.prune.ts"() {
      "use strict";
      init_node();
      init_connection();
    }
  });

  // src/architecture/network/network.gating.ts
  function gate(node, connection) {
    if (!this.nodes.includes(node))
      throw new Error(
        "Gating node must be part of the network to gate a connection!"
      );
    if (connection.gater) {
      if (config.warnings) console.warn("Connection is already gated. Skipping.");
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
    "src/architecture/network/network.gating.ts"() {
      "use strict";
      init_node();
      init_connection();
      init_mutation();
      init_config();
    }
  });

  // src/architecture/network/network.deterministic.ts
  function setSeed(seed) {
    this._rngState = seed >>> 0;
    this._rand = () => {
      this._rngState = this._rngState + 1831565813 >>> 0;
      let r = Math.imul(
        this._rngState ^ this._rngState >>> 15,
        1 | this._rngState
      );
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
    if (typeof state === "number") this._rngState = state >>> 0;
  }
  var init_network_deterministic = __esm({
    "src/architecture/network/network.deterministic.ts"() {
      "use strict";
    }
  });

  // src/architecture/network/network.stats.ts
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
    "src/architecture/network/network.stats.ts"() {
      "use strict";
    }
  });

  // src/architecture/network/network.remove.ts
  function removeNode(node) {
    const internalNet = this;
    const idx = this.nodes.indexOf(node);
    if (idx === -1) throw new Error("Node not in network");
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
        if (!ic.from || !oc.to || ic.from === oc.to) return;
        const exists = this.connections.some(
          (c) => c.from === ic.from && c.to === oc.to
        );
        if (!exists) this.connect(ic.from, oc.to);
      });
    });
    internalNet._topoDirty = true;
    internalNet._nodeIndexDirty = true;
    internalNet._slabDirty = true;
    internalNet._adjDirty = true;
  }
  var init_network_remove = __esm({
    "src/architecture/network/network.remove.ts"() {
      "use strict";
      init_nodePool();
      init_config();
    }
  });

  // src/architecture/network/network.connect.ts
  function connect(from, to, weight) {
    if (this._enforceAcyclic && this.nodes.indexOf(from) > this.nodes.indexOf(to))
      return [];
    const connections = from.connect(to, weight);
    for (const c of connections) {
      if (from !== to) {
        this.connections.push(c);
      } else {
        if (this._enforceAcyclic) continue;
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
        if (c.gater) this.ungate(c);
        list.splice(i, 1);
        break;
      }
    }
    from.disconnect(to);
    this._topoDirty = true;
    this._slabDirty = true;
  }
  var init_network_connect = __esm({
    "src/architecture/network/network.connect.ts"() {
      "use strict";
      init_node();
      init_connection();
    }
  });

  // src/architecture/network/network.serialize.ts
  function serialize() {
    this.nodes.forEach(
      (nodeRef, nodeIndex) => nodeRef.index = nodeIndex
    );
    const activations = this.nodes.map(
      (nodeRef) => nodeRef.activation
    );
    const states = this.nodes.map((nodeRef) => nodeRef.state);
    const squashes = this.nodes.map(
      (nodeRef) => nodeRef.squash.name
    );
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
    const [
      activations,
      states,
      squashes,
      connections,
      serializedInput,
      serializedOutput
    ] = data;
    const input = typeof inputSize === "number" ? inputSize : serializedInput || 0;
    const output = typeof outputSize === "number" ? outputSize : serializedOutput || 0;
    const net = new (init_network(), __toCommonJS(network_exports)).default(input, output);
    net.nodes = [];
    net.connections = [];
    net.selfconns = [];
    net.gates = [];
    activations.forEach((activation, nodeIndex) => {
      let type;
      if (nodeIndex < input) type = "input";
      else if (nodeIndex >= activations.length - output) type = "output";
      else type = "hidden";
      const node = new Node2(type);
      node.activation = activation;
      node.state = states[nodeIndex];
      const squashName = squashes[nodeIndex];
      if (!activation_default[squashName]) {
        console.warn(
          `Unknown squash function '${String(
            squashName
          )}' encountered during deserialize. Falling back to identity.`
        );
      }
      node.squash = activation_default[squashName] || activation_default.identity;
      node.index = nodeIndex;
      net.nodes.push(node);
    });
    connections.forEach((serializedConn) => {
      if (serializedConn.from < net.nodes.length && serializedConn.to < net.nodes.length) {
        const sourceNode = net.nodes[serializedConn.from];
        const targetNode = net.nodes[serializedConn.to];
        const createdConnection = net.connect(
          sourceNode,
          targetNode,
          serializedConn.weight
        )[0];
        if (createdConnection && serializedConn.gater != null) {
          if (serializedConn.gater < net.nodes.length) {
            net.gate(
              net.nodes[serializedConn.gater],
              createdConnection
            );
          } else {
            console.warn(
              "Invalid gater index encountered during deserialize; skipping gater assignment."
            );
          }
        }
      } else {
        console.warn(
          "Invalid connection indices encountered during deserialize; skipping connection."
        );
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
    const net = new (init_network(), __toCommonJS(network_exports)).default(
      json.input,
      json.output
    );
    net.dropout = json.dropout || 0;
    net.nodes = [];
    net.connections = [];
    net.selfconns = [];
    net.gates = [];
    json.nodes.forEach((nodeJson, nodeIndex) => {
      const node = new Node2(nodeJson.type);
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
      const sourceNode = net.nodes[connJson.from];
      const targetNode = net.nodes[connJson.to];
      const createdConnection = net.connect(
        sourceNode,
        targetNode,
        connJson.weight
      )[0];
      if (createdConnection && connJson.gater != null && typeof connJson.gater === "number" && net.nodes[connJson.gater]) {
        net.gate(net.nodes[connJson.gater], createdConnection);
      }
      if (createdConnection && typeof connJson.enabled !== "undefined")
        createdConnection.enabled = connJson.enabled;
    });
    return net;
  }
  var init_network_serialize = __esm({
    "src/architecture/network/network.serialize.ts"() {
      "use strict";
      init_node();
      init_connection();
      init_methods();
    }
  });

  // src/architecture/network/network.genetic.ts
  function crossOver(network1, network2, equal = false) {
    if (network1.input !== network2.input || network1.output !== network2.output)
      throw new Error(
        "Parent networks must have the same input and output sizes for crossover."
      );
    const offspring = new (init_network(), __toCommonJS(network_exports)).default(
      network1.input,
      network1.output
    );
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
    } else size = score1 > score2 ? n1Size : n2Size;
    const outputSize = network1.output;
    network1.nodes.forEach((n, i) => n.index = i);
    network2.nodes.forEach((n, i) => n.index = i);
    for (let i = 0; i < size; i++) {
      let chosen;
      const node1 = i < n1Size ? network1.nodes[i] : void 0;
      const node2 = i < n2Size ? network2.nodes[i] : void 0;
      if (i < network1.input) chosen = node1;
      else if (i >= size - outputSize) {
        const o1 = n1Size - (size - i);
        const o2 = n2Size - (size - i);
        const n1o = o1 >= network1.input && o1 < n1Size ? network1.nodes[o1] : void 0;
        const n2o = o2 >= network2.input && o2 < n2Size ? network2.nodes[o2] : void 0;
        if (n1o && n2o)
          chosen = (network1._rand || Math.random)() >= 0.5 ? n1o : n2o;
        else chosen = n1o || n2o;
      } else {
        if (node1 && node2)
          chosen = (network1._rand || Math.random)() >= 0.5 ? node1 : node2;
        else if (node1 && (score1 >= score2 || equal)) chosen = node1;
        else if (node2 && (score2 >= score1 || equal)) chosen = node2;
      }
      if (chosen) {
        const nn = new Node2(chosen.type);
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
        if (cd.from >= cd.to) return;
        if (!from.isProjectingTo(to)) {
          const conn = offspring.connect(
            from,
            to
          )[0];
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
    "src/architecture/network/network.genetic.ts"() {
      "use strict";
      init_node();
      init_connection();
    }
  });

  // src/architecture/network/network.activate.ts
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
      throw new Error(
        `Input size mismatch: expected ${this.input}, got ${input ? input.length : "undefined"}`
      );
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
      if (node.type === "input") node.noTraceActivate(input[index]);
      else if (node.type === "output")
        output[outIndex++] = node.noTraceActivate();
      else node.noTraceActivate();
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
        throw new Error(
          `Input[${i}] size mismatch: expected ${this.input}, got ${x ? x.length : "undefined"}`
        );
      }
      out[i] = this.activate(x, training);
    }
    return out;
  }
  var init_network_activate = __esm({
    "src/architecture/network/network.activate.ts"() {
      "use strict";
      init_activationArrayPool();
    }
  });

  // src/architecture/group.ts
  var Group;
  var init_group = __esm({
    "src/architecture/group.ts"() {
      "use strict";
      init_node();
      init_layer();
      init_config();
      init_methods();
      Group = class _Group {
        /**
         * An array holding all the nodes within this group.
         */
        nodes;
        /**
         * Stores connection information related to this group.
         * `in`: Connections coming into any node in this group from outside.
         * `out`: Connections going out from any node in this group to outside.
         * `self`: Connections between nodes within this same group (e.g., in ONE_TO_ONE connections).
         */
        connections;
        /**
         * Creates a new group comprised of a specified number of nodes.
         * @param {number} size - The quantity of nodes to initialize within this group.
         */
        constructor(size) {
          this.nodes = [];
          this.connections = {
            in: [],
            out: [],
            self: []
          };
          for (let i = 0; i < size; i++) {
            this.nodes.push(new Node2());
          }
        }
        /**
         * Activates all nodes in the group. If input values are provided, they are assigned
         * sequentially to the nodes before activation. Otherwise, nodes activate based on their
         * existing states and incoming connections.
         *
         * @param {number[]} [value] - An optional array of input values. If provided, its length must match the number of nodes in the group.
         * @returns {number[]} An array containing the activation value of each node in the group, in order.
         * @throws {Error} If the `value` array is provided and its length does not match the number of nodes in the group.
         */
        activate(value) {
          const values = [];
          if (value !== void 0 && value.length !== this.nodes.length) {
            throw new Error(
              "Array with values should be same as the amount of nodes!"
            );
          }
          for (let i = 0; i < this.nodes.length; i++) {
            const activation = value === void 0 ? this.nodes[i].activate() : this.nodes[i].activate(value[i]);
            values.push(activation);
          }
          return values;
        }
        /**
         * Propagates the error backward through all nodes in the group. If target values are provided,
         * the error is calculated against these targets (typically for output layers). Otherwise,
         * the error is calculated based on the error propagated from subsequent layers/nodes.
         *
         * @param {number} rate - The learning rate to apply during weight updates.
         * @param {number} momentum - The momentum factor to apply during weight updates.
         * @param {number[]} [target] - Optional target values for error calculation. If provided, its length must match the number of nodes.
         * @throws {Error} If the `target` array is provided and its length does not match the number of nodes in the group.
         */
        propagate(rate, momentum, target) {
          if (target !== void 0 && target.length !== this.nodes.length) {
            throw new Error(
              "Array with values should be same as the amount of nodes!"
            );
          }
          for (let i = this.nodes.length - 1; i >= 0; i--) {
            if (target === void 0) {
              this.nodes[i].propagate(rate, momentum, true, 0);
            } else {
              this.nodes[i].propagate(rate, momentum, true, 0, target[i]);
            }
          }
        }
        /**
         * Establishes connections from all nodes in this group to a target Group, Layer, or Node.
         * The connection pattern (e.g., all-to-all, one-to-one) can be specified.
         *
         * @param {Group | Layer | Node} target - The destination entity (Group, Layer, or Node) to connect to.
         * @param {methods.groupConnection | methods.connection} [method] - The connection method/type (e.g., `methods.groupConnection.ALL_TO_ALL`, `methods.groupConnection.ONE_TO_ONE`). Defaults depend on the target type and whether it's the same group.
         * @param {number} [weight] - An optional fixed weight to assign to all created connections. If not provided, weights might be initialized randomly or based on node defaults.
         * @returns {any[]} An array containing all the connection objects created. Consider using a more specific type like `Connection[]`.
         * @throws {Error} If `methods.groupConnection.ONE_TO_ONE` is used and the source and target groups have different sizes.
         */
        connect(target, method, weight) {
          let connections = [];
          let i, j;
          if (target instanceof _Group) {
            if (method === void 0) {
              if (this !== target) {
                if (config.warnings)
                  console.warn(
                    "No group connection specified, using ALL_TO_ALL by default."
                  );
                method = connection_default.ALL_TO_ALL;
              } else {
                if (config.warnings)
                  console.warn(
                    "Connecting group to itself, using ONE_TO_ONE by default."
                  );
                method = connection_default.ONE_TO_ONE;
              }
            }
            if (method === connection_default.ALL_TO_ALL || method === connection_default.ALL_TO_ELSE) {
              for (i = 0; i < this.nodes.length; i++) {
                for (j = 0; j < target.nodes.length; j++) {
                  if (method === connection_default.ALL_TO_ELSE && this.nodes[i] === target.nodes[j])
                    continue;
                  let connection = this.nodes[i].connect(target.nodes[j], weight);
                  this.connections.out.push(connection[0]);
                  target.connections.in.push(connection[0]);
                  connections.push(connection[0]);
                }
              }
            } else if (method === connection_default.ONE_TO_ONE) {
              if (this.nodes.length !== target.nodes.length) {
                throw new Error(
                  "Cannot create ONE_TO_ONE connection: source and target groups must have the same size."
                );
              }
              for (i = 0; i < this.nodes.length; i++) {
                let connection = this.nodes[i].connect(target.nodes[i], weight);
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
          } else if (target instanceof Node2) {
            for (i = 0; i < this.nodes.length; i++) {
              let connection = this.nodes[i].connect(target, weight);
              this.connections.out.push(connection[0]);
              connections.push(connection[0]);
            }
          }
          return connections;
        }
        /**
         * Configures nodes within this group to act as gates for the specified connection(s).
         * Gating allows the output of a node in this group to modulate the flow of signal through the gated connection.
         *
         * @param {any | any[]} connections - A single connection object or an array of connection objects to be gated. Consider using a more specific type like `Connection | Connection[]`.
         * @param {methods.gating} method - The gating mechanism to use (e.g., `methods.gating.INPUT`, `methods.gating.OUTPUT`, `methods.gating.SELF`). Specifies which part of the connection is influenced by the gater node.
         * @throws {Error} If no gating `method` is specified.
         */
        gate(connections, method) {
          if (method === void 0) {
            throw new Error(
              "Please specify a gating method: Gating.INPUT, Gating.OUTPUT, or Gating.SELF"
            );
          }
          if (!Array.isArray(connections)) {
            connections = [connections];
          }
          const nodes1 = [];
          const nodes2 = [];
          let i, j;
          for (i = 0; i < connections.length; i++) {
            const connection = connections[i];
            if (!nodes1.includes(connection.from)) nodes1.push(connection.from);
            if (!nodes2.includes(connection.to)) nodes2.push(connection.to);
          }
          switch (method) {
            // Gate the input to the target node(s) of the connection(s)
            case gating.INPUT:
              for (let i2 = 0; i2 < connections.length; i2++) {
                const conn = connections[i2];
                const gater = this.nodes[i2 % this.nodes.length];
                gater.gate(conn);
              }
              break;
            // Gate the output from the source node(s) of the connection(s)
            case gating.OUTPUT:
              for (i = 0; i < nodes1.length; i++) {
                let node = nodes1[i];
                let gater = this.nodes[i % this.nodes.length];
                for (j = 0; j < node.connections.out.length; j++) {
                  let conn = node.connections.out[j];
                  if (connections.includes(conn)) {
                    gater.gate(conn);
                  }
                }
              }
              break;
            // Gate the self-connection of the node(s) involved
            case gating.SELF:
              for (i = 0; i < nodes1.length; i++) {
                let node = nodes1[i];
                let gater = this.nodes[i % this.nodes.length];
                const selfConn = Array.isArray(node.connections.self) ? node.connections.self[0] : node.connections.self;
                if (connections.includes(selfConn)) {
                  gater.gate(selfConn);
                }
              }
              break;
          }
        }
        /**
         * Sets specific properties (like bias, squash function, or type) for all nodes within the group.
         *
         * @param {{ bias?: number; squash?: any; type?: string }} values - An object containing the properties and their new values. Only provided properties are updated.
         *        `bias`: Sets the bias term for all nodes.
         *        `squash`: Sets the activation function (squashing function) for all nodes.
         *        `type`: Sets the node type (e.g., 'input', 'hidden', 'output') for all nodes.
         */
        set(values) {
          for (let i = 0; i < this.nodes.length; i++) {
            if (values.bias !== void 0) {
              this.nodes[i].bias = values.bias;
            }
            this.nodes[i].squash = values.squash || this.nodes[i].squash;
            this.nodes[i].type = values.type || this.nodes[i].type;
          }
        }
        /**
         * Removes connections between nodes in this group and a target Group or Node.
         *
         * @param {Group | Node} target - The Group or Node to disconnect from.
         * @param {boolean} [twosided=false] - If true, also removes connections originating from the `target` and ending in this group. Defaults to false (only removes connections from this group to the target).
         */
        disconnect(target, twosided = false) {
          let i, j, k;
          if (target instanceof _Group) {
            for (i = 0; i < this.nodes.length; i++) {
              for (j = 0; j < target.nodes.length; j++) {
                this.nodes[i].disconnect(target.nodes[j], twosided);
                for (k = this.connections.out.length - 1; k >= 0; k--) {
                  let conn = this.connections.out[k];
                  if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                    this.connections.out.splice(k, 1);
                    break;
                  }
                }
                if (twosided) {
                  for (k = this.connections.in.length - 1; k >= 0; k--) {
                    let conn = this.connections.in[k];
                    if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                      this.connections.in.splice(k, 1);
                      break;
                    }
                  }
                  for (k = target.connections.out.length - 1; k >= 0; k--) {
                    let conn = target.connections.out[k];
                    if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                      target.connections.out.splice(k, 1);
                      break;
                    }
                  }
                  for (k = target.connections.in.length - 1; k >= 0; k--) {
                    let conn = target.connections.in[k];
                    if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                      target.connections.in.splice(k, 1);
                      break;
                    }
                  }
                }
              }
            }
          } else if (target instanceof Node2) {
            for (i = 0; i < this.nodes.length; i++) {
              this.nodes[i].disconnect(target, twosided);
              for (j = this.connections.out.length - 1; j >= 0; j--) {
                let conn = this.connections.out[j];
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
        /**
         * Resets the state of all nodes in the group. This typically involves clearing
         * activation values, state, and propagated errors, preparing the group for a new input pattern,
         * especially relevant in recurrent networks or sequence processing.
         */
        clear() {
          for (let i = 0; i < this.nodes.length; i++) {
            this.nodes[i].clear();
          }
        }
        /**
         * Serializes the group into a JSON-compatible format, avoiding circular references.
         * Only includes node indices and connection counts.
         *
         * @returns {object} A JSON-compatible representation of the group.
         */
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

  // src/architecture/layer.ts
  var layer_exports = {};
  __export(layer_exports, {
    default: () => Layer
  });
  var Layer;
  var init_layer = __esm({
    "src/architecture/layer.ts"() {
      "use strict";
      init_node();
      init_group();
      init_methods();
      init_activationArrayPool();
      Layer = class _Layer {
        /**
         * An array containing all the nodes (neurons or groups) that constitute this layer.
         * The order of nodes might be relevant depending on the layer type and its connections.
         */
        nodes;
        // Note: While typed as Node[], can contain Group instances in practice for memory layers.
        /**
         * Stores connection information related to this layer. This is often managed
         * by the network or higher-level structures rather than directly by the layer itself.
         * `in`: Incoming connections to the layer's nodes.
         * `out`: Outgoing connections from the layer's nodes.
         * `self`: Self-connections within the layer's nodes.
         */
        connections;
        /**
         * Represents the primary output group of nodes for this layer.
         * This group is typically used when connecting this layer *to* another layer or group.
         * It might be null if the layer is not yet fully constructed or is an input layer.
         */
        output;
        /**
         * Dropout rate for this layer (0 to 1). If > 0, all nodes in the layer are masked together during training.
         * Layer-level dropout takes precedence over node-level dropout for nodes in this layer.
         */
        dropout = 0;
        /**
         * Initializes a new Layer instance.
         */
        constructor() {
          this.output = null;
          this.nodes = [];
          this.connections = { in: [], out: [], self: [] };
        }
        /**
         * Activates all nodes within the layer, computing their output values.
         *
         * If an input `value` array is provided, it's used as the initial activation
         * for the corresponding nodes in the layer. Otherwise, nodes compute their
         * activation based on their incoming connections.
         *
         * During training, layer-level dropout is applied, masking all nodes in the layer together.
         * During inference, all masks are set to 1.
         *
         * @param value - An optional array of activation values to set for the layer's nodes. The length must match the number of nodes.
         * @param training - A boolean indicating whether the layer is in training mode. Defaults to false.
         * @returns An array containing the activation value of each node in the layer after activation.
         * @throws {Error} If the provided `value` array's length does not match the number of nodes in the layer.
         */
        activate(value, training = false) {
          const out = activationArrayPool.acquire(this.nodes.length);
          if (value !== void 0 && value.length !== this.nodes.length) {
            throw new Error(
              "Array with values should be same as the amount of nodes!"
            );
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
        /**
         * Propagates the error backward through all nodes in the layer.
         *
         * This is a core step in the backpropagation algorithm used for training.
         * If a `target` array is provided (typically for the output layer), it's used
         * to calculate the initial error for each node. Otherwise, nodes calculate
         * their error based on the error propagated from subsequent layers.
         *
         * @param rate - The learning rate, controlling the step size of weight adjustments.
         * @param momentum - The momentum factor, used to smooth weight updates and escape local minima.
         * @param target - An optional array of target values (expected outputs) for the layer's nodes. The length must match the number of nodes.
         * @throws {Error} If the provided `target` array's length does not match the number of nodes in the layer.
         */
        propagate(rate, momentum, target) {
          if (target !== void 0 && target.length !== this.nodes.length) {
            throw new Error(
              "Array with values should be same as the amount of nodes!"
            );
          }
          for (let i = this.nodes.length - 1; i >= 0; i--) {
            if (target === void 0) {
              this.nodes[i].propagate(rate, momentum, true, 0);
            } else {
              this.nodes[i].propagate(rate, momentum, true, 0, target[i]);
            }
          }
        }
        /**
         * Connects this layer's output to a target component (Layer, Group, or Node).
         *
         * This method delegates the connection logic primarily to the layer's `output` group
         * or the target layer's `input` method. It establishes the forward connections
         * necessary for signal propagation.
         *
         * @param target - The destination Layer, Group, or Node to connect to.
         * @param method - The connection method (e.g., `ALL_TO_ALL`, `ONE_TO_ONE`) defining the connection pattern. See `methods.groupConnection`.
         * @param weight - An optional fixed weight to assign to all created connections.
         * @returns An array containing the newly created connection objects.
         * @throws {Error} If the layer's `output` group is not defined.
         */
        connect(target, method, weight) {
          if (!this.output) {
            throw new Error(
              "Layer output is not defined. Cannot connect from this layer."
            );
          }
          let connections = [];
          if (target instanceof _Layer) {
            connections = target.input(this, method, weight);
          } else if (target instanceof Group || target instanceof Node2) {
            connections = this.output.connect(target, method, weight);
          }
          return connections;
        }
        /**
         * Applies gating to a set of connections originating from this layer's output group.
         *
         * Gating allows the activity of nodes in this layer (specifically, the output group)
         * to modulate the flow of information through the specified `connections`.
         *
         * @param connections - An array of connection objects to be gated.
         * @param method - The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.
         * @throws {Error} If the layer's `output` group is not defined.
         */
        gate(connections, method) {
          if (!this.output) {
            throw new Error(
              "Layer output is not defined. Cannot gate from this layer."
            );
          }
          this.output.gate(connections, method);
        }
        /**
         * Configures properties for all nodes within the layer.
         *
         * Allows batch setting of common node properties like bias, activation function (`squash`),
         * or node type. If a node within the `nodes` array is actually a `Group` (e.g., in memory layers),
         * the configuration is applied recursively to the nodes within that group.
         *
         * @param values - An object containing the properties and their values to set.
         *                 Example: `{ bias: 0.5, squash: methods.Activation.ReLU }`
         */
        set(values) {
          for (let i = 0; i < this.nodes.length; i++) {
            let node = this.nodes[i];
            if (node instanceof Node2) {
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
        /**
         * Removes connections between this layer's nodes and a target Group or Node.
         *
         * @param target - The Group or Node to disconnect from.
         * @param twosided - If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.
         */
        disconnect(target, twosided) {
          twosided = twosided || false;
          let i, j, k;
          if (target instanceof Group) {
            for (i = 0; i < this.nodes.length; i++) {
              for (j = 0; j < target.nodes.length; j++) {
                this.nodes[i].disconnect(target.nodes[j], twosided);
                for (k = this.connections.out.length - 1; k >= 0; k--) {
                  let conn = this.connections.out[k];
                  if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                    this.connections.out.splice(k, 1);
                    break;
                  }
                }
                if (twosided) {
                  for (k = this.connections.in.length - 1; k >= 0; k--) {
                    let conn = this.connections.in[k];
                    if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                      this.connections.in.splice(k, 1);
                      break;
                    }
                  }
                }
              }
            }
          } else if (target instanceof Node2) {
            for (i = 0; i < this.nodes.length; i++) {
              this.nodes[i].disconnect(target, twosided);
              for (j = this.connections.out.length - 1; j >= 0; j--) {
                let conn = this.connections.out[j];
                if (conn.from === this.nodes[i] && conn.to === target) {
                  this.connections.out.splice(j, 1);
                  break;
                }
              }
              if (twosided) {
                for (k = this.connections.in.length - 1; k >= 0; k--) {
                  let conn = this.connections.in[k];
                  if (conn.from === target && conn.to === this.nodes[i]) {
                    this.connections.in.splice(k, 1);
                    break;
                  }
                }
              }
            }
          }
        }
        /**
         * Resets the activation state of all nodes within the layer.
         * This is typically done before processing a new input sequence or sample.
         */
        clear() {
          for (let i = 0; i < this.nodes.length; i++) {
            this.nodes[i].clear();
          }
        }
        /**
         * Handles the connection logic when this layer is the *target* of a connection.
         *
         * It connects the output of the `from` layer or group to this layer's primary
         * input mechanism (which is often the `output` group itself, but depends on the layer type).
         * This method is usually called by the `connect` method of the source layer/group.
         *
         * @param from - The source Layer or Group connecting *to* this layer.
         * @param method - The connection method (e.g., `ALL_TO_ALL`). Defaults to `ALL_TO_ALL`.
         * @param weight - An optional fixed weight for the connections.
         * @returns An array containing the newly created connection objects.
         * @throws {Error} If the layer's `output` group (acting as input target here) is not defined.
         */
        input(from, method, weight) {
          if (from instanceof _Layer) from = from.output;
          method = method || connection_default.ALL_TO_ALL;
          if (!this.output) {
            throw new Error("Layer output (acting as input target) is not defined.");
          }
          return from.connect(this.output, method, weight);
        }
        // Static Layer Factory Methods
        /**
         * Creates a standard fully connected (dense) layer.
         *
         * All nodes in the source layer/group will connect to all nodes in this layer
         * when using the default `ALL_TO_ALL` connection method via `layer.input()`.
         *
         * @param size - The number of nodes (neurons) in this layer.
         * @returns A new Layer instance configured as a dense layer.
         */
        static dense(size) {
          const layer = new _Layer();
          const block = new Group(size);
          layer.nodes.push(...block.nodes);
          layer.output = block;
          layer.input = (from, method, weight) => {
            if (from instanceof _Layer) from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            return from.connect(block, method, weight);
          };
          return layer;
        }
        /**
         * Creates a Long Short-Term Memory (LSTM) layer.
         *
         * LSTMs are a type of recurrent neural network (RNN) cell capable of learning
         * long-range dependencies. This implementation uses standard LSTM architecture
         * with input, forget, and output gates, and a memory cell.
         *
         * @param size - The number of LSTM units (and nodes in each gate/cell group).
         * @returns A new Layer instance configured as an LSTM layer.
         */
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
          const output = memoryCell.connect(
            outputBlock,
            connection_default.ALL_TO_ALL
          );
          outputGate.gate(output, gating.OUTPUT);
          memoryCell.nodes.forEach((node, i) => {
            const selfConnection = node.connections.self.find(
              (conn) => conn.to === node && conn.from === node
            );
            if (selfConnection) {
              selfConnection.gater = forgetGate.nodes[i];
              if (!forgetGate.nodes[i].connections.gated.includes(selfConnection)) {
                forgetGate.nodes[i].connections.gated.push(selfConnection);
              }
            } else {
              console.warn(
                `LSTM Warning: No self-connection found for memory cell node ${i}`
              );
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
            if (from instanceof _Layer) from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            let connections = [];
            const input = from.connect(memoryCell, method, weight);
            connections = connections.concat(input);
            connections = connections.concat(from.connect(inputGate, method, weight));
            connections = connections.concat(
              from.connect(outputGate, method, weight)
            );
            connections = connections.concat(
              from.connect(forgetGate, method, weight)
            );
            inputGate.gate(input, gating.INPUT);
            return connections;
          };
          return layer;
        }
        /**
         * Creates a Gated Recurrent Unit (GRU) layer.
         *
         * GRUs are another type of recurrent neural network cell, often considered
         * simpler than LSTMs but achieving similar performance on many tasks.
         * They use an update gate and a reset gate to manage information flow.
         *
         * @param size - The number of GRU units (and nodes in each gate/cell group).
         * @returns A new Layer instance configured as a GRU layer.
         */
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
            // Pass through previous output directly
            type: "variant"
            // Custom type identifier
          });
          memoryCell.set({
            squash: activation_default.tanh
            // Tanh activation for candidate state
          });
          inverseUpdateGate.set({
            bias: 0,
            squash: activation_default.inverse,
            // Activation computes 1 - input
            type: "variant"
            // Custom type identifier
          });
          updateGate.set({ bias: 1 });
          resetGate.set({ bias: 0 });
          previousOutput.connect(updateGate, connection_default.ALL_TO_ALL);
          previousOutput.connect(resetGate, connection_default.ALL_TO_ALL);
          updateGate.connect(
            inverseUpdateGate,
            connection_default.ONE_TO_ONE,
            1
          );
          const reset = previousOutput.connect(
            memoryCell,
            connection_default.ALL_TO_ALL
          );
          resetGate.gate(reset, gating.OUTPUT);
          const update1 = previousOutput.connect(
            output,
            connection_default.ALL_TO_ALL
          );
          const update2 = memoryCell.connect(
            output,
            connection_default.ALL_TO_ALL
          );
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
            if (from instanceof _Layer) from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            let connections = [];
            connections = connections.concat(
              from.connect(updateGate, method, weight)
            );
            connections = connections.concat(from.connect(resetGate, method, weight));
            connections = connections.concat(
              from.connect(memoryCell, method, weight)
            );
            return connections;
          };
          return layer;
        }
        /**
         * Creates a Memory layer, designed to hold state over a fixed number of time steps.
         *
         * This layer consists of multiple groups (memory blocks), each holding the state
         * from a previous time step. The input connects to the most recent block, and
         * information propagates backward through the blocks. The layer's output
         * concatenates the states of all memory blocks.
         *
         * @param size - The number of nodes in each memory block (must match the input size).
         * @param memory - The number of time steps to remember (number of memory blocks).
         * @returns A new Layer instance configured as a Memory layer.
         * @throws {Error} If the connecting layer's size doesn't match the memory block `size`.
         */
        static memory(size, memory) {
          const layer = new _Layer();
          let previous = null;
          for (let i = 0; i < memory; i++) {
            const block = new Group(size);
            block.set({
              squash: activation_default.identity,
              bias: 0,
              type: "variant"
              // Custom type identifier
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
              console.warn(
                "Unexpected Node type found directly in Memory layer nodes list during output group creation."
              );
            }
          }
          layer.output = outputGroup;
          layer.input = (from, method, weight) => {
            if (from instanceof _Layer) from = from.output;
            method = method || connection_default.ALL_TO_ALL;
            const inputBlock = layer.nodes[layer.nodes.length - 1];
            if (!this.prototype.isGroup(inputBlock)) {
              throw new Error("Memory layer input block is not a Group.");
            }
            if (from.nodes.length !== inputBlock.nodes.length) {
              throw new Error(
                `Previous layer size (${from.nodes.length}) must be same as memory size (${inputBlock.nodes.length})`
              );
            }
            return from.connect(inputBlock, connection_default.ONE_TO_ONE, 1);
          };
          return layer;
        }
        /**
         * Creates a batch normalization layer.
         * Applies batch normalization to the activations of the nodes in this layer during activation.
         * @param size - The number of nodes in this layer.
         * @returns A new Layer instance configured as a batch normalization layer.
         */
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
        /**
         * Creates a layer normalization layer.
         * Applies layer normalization to the activations of the nodes in this layer during activation.
         * @param size - The number of nodes in this layer.
         * @returns A new Layer instance configured as a layer normalization layer.
         */
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
        /**
         * Creates a 1D convolutional layer (stub implementation).
         * @param size - Number of output nodes (filters).
         * @param kernelSize - Size of the convolution kernel.
         * @param stride - Stride of the convolution (default 1).
         * @param padding - Padding (default 0).
         * @returns A new Layer instance representing a 1D convolutional layer.
         */
        static conv1d(size, kernelSize, stride = 1, padding = 0) {
          const layer = new _Layer();
          layer.nodes = Array.from({ length: size }, () => new Node2());
          layer.output = new Group(size);
          layer.conv1d = { kernelSize, stride, padding };
          layer.activate = function(value) {
            if (!value) return this.nodes.map((n) => n.activate());
            return value.slice(0, size);
          };
          return layer;
        }
        /**
         * Creates a multi-head self-attention layer (stub implementation).
         * @param size - Number of output nodes.
         * @param heads - Number of attention heads (default 1).
         * @returns A new Layer instance representing an attention layer.
         */
        static attention(size, heads = 1) {
          const layer = new _Layer();
          layer.nodes = Array.from({ length: size }, () => new Node2());
          layer.output = new Group(size);
          layer.attention = { heads };
          layer.activate = function(value) {
            if (!value) return this.nodes.map((n) => n.activate());
            const avg = value.reduce((a, b) => a + b, 0) / value.length;
            return Array(size).fill(avg);
          };
          return layer;
        }
        /**
         * Type guard to check if an object is likely a `Group`.
         *
         * This is a duck-typing check based on the presence of expected properties
         * (`set` method and `nodes` array). Used internally where `layer.nodes`
         * might contain `Group` instances (e.g., in `Memory` layers).
         *
         * @param obj - The object to inspect.
         * @returns `true` if the object has `set` and `nodes` properties matching a Group, `false` otherwise.
         */
        isGroup(obj) {
          return !!obj && typeof obj.set === "function" && Array.isArray(obj.nodes);
        }
      };
    }
  });

  // src/architecture/network/network.mutate.ts
  var network_mutate_exports = {};
  __export(network_mutate_exports, {
    mutateImpl: () => mutateImpl
  });
  function mutateImpl(method) {
    if (method == null) throw new Error("No (correct) mutate method given!");
    let key;
    if (typeof method === "string") key = method;
    else key = method?.name ?? method?.type ?? method?.identity;
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
    if (internal._enforceAcyclic) internal._topoDirty = true;
    if (config.deterministicChainMode) {
      const inputNode = this.nodes.find((n) => n.type === "input");
      const outputNode = this.nodes.find((n) => n.type === "output");
      if (!inputNode || !outputNode) return;
      if (!internal._detChain) {
        if (!this.connections.some(
          (c) => c.from === inputNode && c.to === outputNode
        )) {
          this.connect(inputNode, outputNode);
        }
        internal._detChain = [inputNode];
      }
      const chain = internal._detChain;
      const tail = chain[chain.length - 1];
      let terminal = this.connections.find(
        (c) => c.from === tail && c.to === outputNode
      );
      if (!terminal) terminal = this.connect(tail, outputNode)[0];
      const prevGater2 = terminal.gater;
      this.disconnect(terminal.from, terminal.to);
      const hidden2 = new Node2("hidden", void 0, internal._rand);
      hidden2.mutate(mutation_default.MOD_ACTIVATION);
      const outIndex = this.nodes.indexOf(outputNode);
      const insertIndex2 = Math.min(outIndex, this.nodes.length - this.output);
      this.nodes.splice(insertIndex2, 0, hidden2);
      internal._nodeIndexDirty = true;
      const c12 = this.connect(tail, hidden2)[0];
      const c22 = this.connect(hidden2, outputNode)[0];
      chain.push(hidden2);
      internal._preferredChainEdge = c22;
      if (prevGater2) this.gate(prevGater2, internal._rand() >= 0.5 ? c12 : c22);
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
      if (input && output) this.connect(input, output);
      else return;
    }
    const connection = this.connections[Math.floor(internal._rand() * this.connections.length)];
    if (!connection) return;
    const prevGater = connection.gater;
    this.disconnect(connection.from, connection.to);
    const hidden = new Node2("hidden", void 0, internal._rand);
    hidden.mutate(mutation_default.MOD_ACTIVATION);
    const targetIndex = this.nodes.indexOf(connection.to);
    const insertIndex = Math.min(targetIndex, this.nodes.length - this.output);
    this.nodes.splice(insertIndex, 0, hidden);
    internal._nodeIndexDirty = true;
    const c1 = this.connect(connection.from, hidden)[0];
    const c2 = this.connect(hidden, connection.to)[0];
    internal._preferredChainEdge = c2;
    if (prevGater) this.gate(prevGater, internal._rand() >= 0.5 ? c1 : c2);
  }
  function _subNode() {
    const hidden = this.nodes.filter((n) => n.type === "hidden");
    if (hidden.length === 0) {
      if (config.warnings) console.warn("No hidden nodes left to remove!");
      return;
    }
    const internal = this;
    const victim = hidden[Math.floor(internal._rand() * hidden.length)];
    this.remove(victim);
    const anyConn = this.connections[0];
    if (anyConn) anyConn.weight += 1e-4;
  }
  function _addConn() {
    const netInternal = this;
    if (netInternal._enforceAcyclic) netInternal._topoDirty = true;
    const forwardConnectionCandidates = [];
    for (let sourceIndex = 0; sourceIndex < this.nodes.length - this.output; sourceIndex++) {
      const sourceNode = this.nodes[sourceIndex];
      for (let targetIndex = Math.max(sourceIndex + 1, this.input); targetIndex < this.nodes.length; targetIndex++) {
        const targetNode = this.nodes[targetIndex];
        if (!sourceNode.isProjectingTo(targetNode))
          forwardConnectionCandidates.push([sourceNode, targetNode]);
      }
    }
    if (forwardConnectionCandidates.length === 0) return;
    const selectedPair = forwardConnectionCandidates[Math.floor(netInternal._rand() * forwardConnectionCandidates.length)];
    this.connect(selectedPair[0], selectedPair[1]);
  }
  function _subConn() {
    const netInternal = this;
    const removableForwardConnections = this.connections.filter(
      (candidateConn) => {
        const sourceHasMultipleOutgoing = candidateConn.from.connections.out.length > 1;
        const targetHasMultipleIncoming = candidateConn.to.connections.in.length > 1;
        const targetLayerPeers = this.nodes.filter(
          (n) => n.type === candidateConn.to.type && Math.abs(
            this.nodes.indexOf(n) - this.nodes.indexOf(candidateConn.to)
          ) < Math.max(this.input, this.output)
        );
        let wouldDisconnectLayerPeerGroup = false;
        if (targetLayerPeers.length > 0) {
          const peerConnectionsFromSource = this.connections.filter(
            (c) => c.from === candidateConn.from && targetLayerPeers.includes(c.to)
          );
          if (peerConnectionsFromSource.length <= 1)
            wouldDisconnectLayerPeerGroup = true;
        }
        return sourceHasMultipleOutgoing && targetHasMultipleIncoming && this.nodes.indexOf(candidateConn.to) > this.nodes.indexOf(candidateConn.from) && !wouldDisconnectLayerPeerGroup;
      }
    );
    if (removableForwardConnections.length === 0) return;
    const connectionToRemove = removableForwardConnections[Math.floor(netInternal._rand() * removableForwardConnections.length)];
    this.disconnect(connectionToRemove.from, connectionToRemove.to);
  }
  function _modWeight(method) {
    const allConnections = this.connections.concat(this.selfconns);
    if (allConnections.length === 0) return;
    const connectionToPerturb = allConnections[Math.floor(this._rand() * allConnections.length)];
    const modification = this._rand() * (method.max - method.min) + method.min;
    connectionToPerturb.weight += modification;
  }
  function _modBias(method) {
    if (this.nodes.length <= this.input) return;
    const targetNodeIndex = Math.floor(
      this._rand() * (this.nodes.length - this.input) + this.input
    );
    const nodeForBiasMutation = this.nodes[targetNodeIndex];
    nodeForBiasMutation.mutate(method);
  }
  function _modActivation(method) {
    const canMutateOutput = method.mutateOutput ?? true;
    const numMutableNodes = this.nodes.length - this.input - (canMutateOutput ? 0 : this.output);
    if (numMutableNodes <= 0) {
      if (config.warnings)
        console.warn(
          "No nodes available for activation function mutation based on config."
        );
      return;
    }
    const targetNodeIndex = Math.floor(
      this._rand() * numMutableNodes + this.input
    );
    const targetNode = this.nodes[targetNodeIndex];
    targetNode.mutate(method);
  }
  function _addSelfConn() {
    const netInternal = this;
    if (netInternal._enforceAcyclic) return;
    const nodesWithoutSelfLoop = this.nodes.filter(
      (n, idx) => idx >= this.input && n.connections.self.length === 0
    );
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
      if (config.warnings) console.warn("No self-connections exist to remove.");
      return;
    }
    const selfConnectionToRemove = this.selfconns[Math.floor(this._rand() * this.selfconns.length)];
    this.disconnect(selfConnectionToRemove.from, selfConnectionToRemove.to);
  }
  function _addGate() {
    const netInternal = this;
    const allConnectionsIncludingSelf = this.connections.concat(this.selfconns);
    const ungatedConnectionCandidates = allConnectionsIncludingSelf.filter(
      (c) => c.gater === null
    );
    if (ungatedConnectionCandidates.length === 0 || this.nodes.length <= this.input) {
      if (config.warnings) console.warn("All connections are already gated.");
      return;
    }
    const gatingNodeIndex = Math.floor(
      netInternal._rand() * (this.nodes.length - this.input) + this.input
    );
    const gatingNode = this.nodes[gatingNodeIndex];
    const connectionToGate = ungatedConnectionCandidates[Math.floor(netInternal._rand() * ungatedConnectionCandidates.length)];
    this.gate(gatingNode, connectionToGate);
  }
  function _subGate() {
    if (this.gates.length === 0) {
      if (config.warnings) console.warn("No gated connections to ungate.");
      return;
    }
    const gatedConnectionIndex = Math.floor(
      this._rand() * this.gates.length
    );
    const gatedConnection = this.gates[gatedConnectionIndex];
    this.ungate(gatedConnection);
  }
  function _addBackConn() {
    const netInternal = this;
    if (netInternal._enforceAcyclic) return;
    const backwardConnectionCandidates = [];
    for (let laterIndex = this.input; laterIndex < this.nodes.length; laterIndex++) {
      const laterNode = this.nodes[laterIndex];
      for (let earlierIndex = this.input; earlierIndex < laterIndex; earlierIndex++) {
        const earlierNode = this.nodes[earlierIndex];
        if (!laterNode.isProjectingTo(earlierNode))
          backwardConnectionCandidates.push([laterNode, earlierNode]);
      }
    }
    if (backwardConnectionCandidates.length === 0) return;
    const selectedBackwardPair = backwardConnectionCandidates[Math.floor(netInternal._rand() * backwardConnectionCandidates.length)];
    this.connect(selectedBackwardPair[0], selectedBackwardPair[1]);
  }
  function _subBackConn() {
    const removableBackwardConnections = this.connections.filter(
      (candidateConn) => candidateConn.from.connections.out.length > 1 && candidateConn.to.connections.in.length > 1 && this.nodes.indexOf(candidateConn.from) > this.nodes.indexOf(candidateConn.to)
    );
    if (removableBackwardConnections.length === 0) return;
    const backwardConnectionToRemove = removableBackwardConnections[Math.floor(this._rand() * removableBackwardConnections.length)];
    this.disconnect(
      backwardConnectionToRemove.from,
      backwardConnectionToRemove.to
    );
  }
  function _swapNodes(method) {
    const netInternal = this;
    const canSwapOutput = method.mutateOutput ?? true;
    const numSwappableNodes = this.nodes.length - this.input - (canSwapOutput ? 0 : this.output);
    if (numSwappableNodes < 2) return;
    let firstNodeIndex = Math.floor(
      netInternal._rand() * numSwappableNodes + this.input
    );
    let secondNodeIndex = Math.floor(
      netInternal._rand() * numSwappableNodes + this.input
    );
    while (firstNodeIndex === secondNodeIndex)
      secondNodeIndex = Math.floor(
        netInternal._rand() * numSwappableNodes + this.input
      );
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
    if (netInternal._enforceAcyclic) return;
    if (this.connections.length === 0) return;
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
    if (netInternal._enforceAcyclic) return;
    if (this.connections.length === 0) return;
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
    if (this.nodes.length <= this.input) return;
    const internal = this;
    const idx = Math.floor(
      internal._rand() * (this.nodes.length - this.input) + this.input
    );
    const node = this.nodes[idx];
    const min = method?.min ?? -1;
    const max = method?.max ?? 1;
    const sample = () => internal._rand() * (max - min) + min;
    for (const c of node.connections.in) c.weight = sample();
    for (const c of node.connections.out) c.weight = sample();
    for (const c of node.connections.self) c.weight = sample();
  }
  function _batchNorm() {
    const hidden = this.nodes.filter((n) => n.type === "hidden");
    if (!hidden.length) return;
    const internal = this;
    const node = hidden[Math.floor(internal._rand() * hidden.length)];
    node._batchNorm = true;
  }
  var MUTATION_DISPATCH;
  var init_network_mutate = __esm({
    "src/architecture/network/network.mutate.ts"() {
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

  // src/architecture/network/network.training.ts
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
      if (state.emaValue == null) state.emaValue = trainError;
      else
        state.emaValue = state.emaValue + cfg.emaAlpha * (trainError - state.emaValue);
      return state.emaValue;
    }
    if (type === "adaptive-ema") {
      const mean = recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length;
      const variance = recentErrors.reduce((a, b) => a + (b - mean) * (b - mean), 0) / recentErrors.length;
      const baseAlpha = cfg.emaAlpha || 2 / (cfg.window + 1);
      const varianceScaled = variance / Math.max(mean * mean, 1e-8);
      const adaptiveAlpha = Math.min(
        0.95,
        Math.max(baseAlpha, baseAlpha * (1 + 2 * varianceScaled))
      );
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
    if (cfg.window <= 1 && cfg.type !== "ema") return trainError;
    if (cfg.type === "median") {
      const sorted = [...plateauErrors].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }
    if (cfg.type === "ema") {
      if (state.plateauEmaValue == null) state.plateauEmaValue = trainError;
      else
        state.plateauEmaValue = state.plateauEmaValue + cfg.emaAlpha * (trainError - state.plateauEmaValue);
      return state.plateauEmaValue;
    }
    return plateauErrors.reduce((a, b) => a + b, 0) / plateauErrors.length;
  }
  function detectMixedPrecisionOverflow(net, internalNet) {
    if (!internalNet._mixedPrecision.enabled) return false;
    if (internalNet._forceNextOverflow) {
      internalNet._forceNextOverflow = false;
      return true;
    }
    let overflow = false;
    net.nodes.forEach((node) => {
      if (node._fp32Bias !== void 0) {
        if (!Number.isFinite(node.bias)) overflow = true;
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
    if (accumulationSteps <= 1) return;
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
      if (node.type === "input") return;
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
    internalNet._mixedPrecision.lossScale = Math.max(
      internalNet._mixedPrecisionState.minLossScale,
      Math.floor(internalNet._mixedPrecision.lossScale / 2) || 1
    );
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
            if (!layer || !layer.nodes) continue;
            const groupVals = [];
            layer.nodes.forEach((node) => {
              if (!node || node.type === "input") return;
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
            if (groupVals.length) collected.push(groupVals);
          }
        } else {
          net.nodes.forEach((node) => {
            if (node.type === "input") return;
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
            if (groupVals.length) collected.push(groupVals);
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
        if (globalVals.length) collected.push(globalVals);
      }
      return collected;
    };
    const groups = collectGroups();
    internalNet._lastGradClipGroupCount = groups.length;
    const computeAbsolutePercentileThreshold = (values, percentile) => {
      if (!values.length) return 0;
      const sortedByAbs = [...values].sort((a, b) => Math.abs(a) - Math.abs(b));
      const rank = Math.min(
        sortedByAbs.length - 1,
        Math.max(0, Math.floor(percentile / 100 * sortedByAbs.length - 1))
      );
      return Math.abs(sortedByAbs[rank]);
    };
    const applyScale = (scaleFn) => {
      let groupIndex = 0;
      net.nodes.forEach((node) => {
        if (cfg.mode.startsWith("layerwise") && node.type === "input") return;
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
          node.totalDeltaBias = scaleFn(
            node.totalDeltaBias,
            activeGroup
          );
      });
    };
    if (cfg.mode === "norm" || cfg.mode === "layerwiseNorm") {
      const maxAllowedNorm = cfg.maxNorm || 1;
      groups.forEach((groupValues) => {
        const groupL2Norm = Math.sqrt(
          groupValues.reduce((sum, v) => sum + v * v, 0)
        );
        if (groupL2Norm > maxAllowedNorm && groupL2Norm > 0) {
          const normScaleFactor = maxAllowedNorm / groupL2Norm;
          applyScale(
            (currentValue, owningGroup) => owningGroup === groupValues ? currentValue * normScaleFactor : currentValue
          );
        }
      });
    } else if (cfg.mode === "percentile" || cfg.mode === "layerwisePercentile") {
      const percentileSetting = cfg.percentile || 99;
      groups.forEach((groupValues) => {
        const percentileThreshold = computeAbsolutePercentileThreshold(
          groupValues,
          percentileSetting
        );
        if (percentileThreshold <= 0) return;
        applyScale(
          (currentValue, owningGroup) => owningGroup === groupValues && Math.abs(currentValue) > percentileThreshold ? percentileThreshold * Math.sign(currentValue) : currentValue
        );
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
    if (typeof costFunction === "function") computeError = costFunction;
    else if (costFunction && typeof costFunction.fn === "function")
      computeError = costFunction.fn;
    else if (costFunction && typeof costFunction.calculate === "function")
      computeError = costFunction.calculate;
    else computeError = () => 0;
    for (let sampleIndex = 0; sampleIndex < set.length; sampleIndex++) {
      const dataPoint = set[sampleIndex];
      const input = dataPoint.input;
      const target = dataPoint.output;
      if (input.length !== net.input || target.length !== net.output) {
        if (config.warnings)
          console.warn(
            `Data point ${sampleIndex} has incorrect dimensions (input: ${input.length}/${net.input}, output: ${target.length}/${net.output}), skipping.`
          );
        continue;
      }
      try {
        const output = net.activate(input, true);
        if (optimizer && optimizer.type && optimizer.type !== "sgd") {
          for (let outIndex = 0; outIndex < outputNodes.length; outIndex++)
            outputNodes[outIndex].propagate(
              currentRate,
              momentum,
              false,
              regularization,
              target[outIndex]
            );
          for (let reverseIndex = net.nodes.length - 1; reverseIndex >= 0; reverseIndex--) {
            const node = net.nodes[reverseIndex];
            if (node.type === "output" || node.type === "input") continue;
            node.propagate(currentRate, momentum, false, regularization);
          }
        } else {
          for (let outIndex = 0; outIndex < outputNodes.length; outIndex++)
            outputNodes[outIndex].propagate(
              currentRate,
              momentum,
              true,
              regularization,
              target[outIndex]
            );
          for (let reverseIndex = net.nodes.length - 1; reverseIndex >= 0; reverseIndex--) {
            const node = net.nodes[reverseIndex];
            if (node.type === "output" || node.type === "input") continue;
            node.propagate(currentRate, momentum, true, regularization);
          }
        }
        cumulativeError += computeError(target, output);
        batchSampleCount++;
        totalProcessedSamples++;
      } catch (e) {
        if (config.warnings)
          console.warn(
            `Error processing data point ${sampleIndex} (input: ${JSON.stringify(
              input
            )}): ${e.message}. Skipping.`
          );
      }
      if (batchSampleCount > 0 && ((sampleIndex + 1) % batchSize === 0 || sampleIndex === set.length - 1)) {
        if (optimizer && optimizer.type && optimizer.type !== "sgd") {
          internalNet._gradAccumMicroBatches++;
          const readyForStep = internalNet._gradAccumMicroBatches % accumulationSteps === 0 || sampleIndex === set.length - 1;
          if (readyForStep) {
            internalNet._optimizerStep = (internalNet._optimizerStep || 0) + 1;
            const overflowDetected = detectMixedPrecisionOverflow(
              net,
              internalNet
            );
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
              internalNet._lastGradNorm = applyOptimizerStep(
                net,
                optimizer,
                currentRate,
                momentum,
                internalNet
              );
              if (internalNet._mixedPrecision.enabled)
                maybeIncreaseLossScale(internalNet);
            }
          }
          batchSampleCount = 0;
        }
      }
    }
    if (internalNet._lastGradNorm == null) internalNet._lastGradNorm = 0;
    return totalProcessedSamples > 0 ? cumulativeError / totalProcessedSamples : 0;
  }
  function trainImpl(net, set, options) {
    const internalNet = net;
    if (!set || set.length === 0 || set[0].input.length !== net.input || set[0].output.length !== net.output) {
      throw new Error(
        "Dataset is invalid or dimensions do not match network input/output size!"
      );
    }
    options = options || {};
    if (typeof options.iterations === "undefined" && typeof options.error === "undefined") {
      if (config.warnings)
        console.warn("Missing `iterations` or `error` option.");
      throw new Error(
        "Missing `iterations` or `error` option. Training requires a stopping condition."
      );
    }
    if (config.warnings) {
      if (typeof options.rate === "undefined") {
        console.warn("Missing `rate` option");
        console.warn("Missing `rate` option, using default learning rate 0.3.");
      }
      if (typeof options.iterations === "undefined")
        console.warn(
          "Missing `iterations` option. Training will run potentially indefinitely until `error` threshold is met."
        );
    }
    let targetError = options.error ?? -Infinity;
    const cost = options.cost || Cost.mse;
    if (typeof cost !== "function" && !(typeof cost === "object" && (typeof cost.fn === "function" || typeof cost.calculate === "function"))) {
      throw new Error("Invalid cost function provided to Network.train.");
    }
    const baseRate = options.rate ?? 0.3;
    const dropout = options.dropout || 0;
    if (dropout < 0 || dropout >= 1) throw new Error("dropout must be in [0,1)");
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
        if (n.type !== "input") n._fp32Bias = n.bias;
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
        if (!optimizerConfig.baseType) optimizerConfig.baseType = "adam";
        if (optimizerConfig.baseType === "lookahead")
          throw new Error(
            "Nested lookahead (baseType lookahead) is not supported"
          );
        if (!allowedOptimizers.has(optimizerConfig.baseType))
          throw new Error(
            `Unknown baseType for lookahead: ${optimizerConfig.baseType}`
          );
        optimizerConfig.la_k = optimizerConfig.la_k || 5;
        optimizerConfig.la_alpha = optimizerConfig.la_alpha ?? 0.5;
      }
    }
    const iterations = options.iterations ?? Number.MAX_SAFE_INTEGER;
    const start2 = Date.now();
    let finalError = Infinity;
    const movingAverageWindow = Math.max(1, options.movingAverageWindow || 1);
    const movingAverageType = options.movingAverageType || "sma";
    const emaAlpha = (() => {
      if (movingAverageType !== "ema") return void 0;
      if (options.emaAlpha && options.emaAlpha > 0 && options.emaAlpha <= 1)
        return options.emaAlpha;
      return 2 / (movingAverageWindow + 1);
    })();
    const plateauWindow = Math.max(
      1,
      options.plateauMovingAverageWindow || movingAverageWindow
    );
    const plateauType = options.plateauMovingAverageType || movingAverageType;
    const plateauEmaAlpha = (() => {
      if (plateauType !== "ema") return void 0;
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
      if (recentErrorsCount < recentErrorsCapacity) recentErrorsCount++;
    };
    const recentErrorsChrono = () => {
      if (recentErrorsCount === 0) return [];
      if (recentErrorsCount < recentErrorsCapacity)
        return recentErrorsBuf.slice(0, recentErrorsCount);
      const out = new Array(recentErrorsCount);
      const start3 = recentErrorsWriteIdx;
      for (let i = 0; i < recentErrorsCount; i++)
        out[i] = recentErrorsBuf[(start3 + i) % recentErrorsCapacity];
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
      if (plateauCount < plateauCapacity) plateauCount++;
    };
    const plateauChrono = () => {
      if (plateauCount === 0) return [];
      if (plateauCount < plateauCapacity)
        return plateauBuf.slice(0, plateauCount);
      const out = new Array(plateauCount);
      const start3 = plateauWriteIdx;
      for (let i = 0; i < plateauCount; i++)
        out[i] = plateauBuf[(start3 + i) % plateauCapacity];
      return out;
    };
    let plateauEmaValue = void 0;
    net.dropout = dropout;
    let performedIterations = 0;
    for (let iter = 1; iter <= iterations; iter++) {
      if (net._maybePrune) {
        net._maybePrune((internalNet._globalEpoch || 0) + iter);
      }
      const trainError = trainSetImpl(
        net,
        set,
        batchSize,
        accumulationSteps,
        baseRate,
        momentum,
        {},
        cost,
        optimizerConfig
      );
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
          if (emaValue == null) emaValue = trainError;
          else emaValue = emaValue + emaAlpha * (trainError - emaValue);
          monitored = emaValue;
        } else if (movingAverageType === "adaptive-ema") {
          const mean = recentArr.reduce((a, b) => a + b, 0) / recentArr.length;
          const variance = recentArr.reduce((a, b) => a + (b - mean) * (b - mean), 0) / recentArr.length;
          const baseAlpha = emaAlpha || 2 / (movingAverageWindow + 1);
          const varScaled = variance / Math.max(mean * mean, 1e-8);
          const adaptAlpha = Math.min(
            0.95,
            Math.max(baseAlpha, baseAlpha * (1 + 2 * varScaled))
          );
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
            const weight = Math.exp(
              -0.5 * Math.pow((gi - (windowLength - 1)) / sigma, 2)
            );
            gaussianWeightSum += weight;
            gaussianWeightedAccumulator += weight * gaussianWindow[gi];
          }
          monitored = gaussianWeightedAccumulator / (gaussianWeightSum || 1);
        } else if (movingAverageType === "trimmed") {
          const tailTrimRatio = Math.min(
            0.49,
            Math.max(0, options.trimmedRatio || 0.1)
          );
          const sorted = [...recentArr].sort((a, b) => a - b);
          const elementsToDropEachSide = Math.floor(
            sorted.length * tailTrimRatio
          );
          const trimmedSegment = sorted.slice(
            elementsToDropEachSide,
            sorted.length - elementsToDropEachSide
          );
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
          if (plateauEmaValue == null) plateauEmaValue = trainError;
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
      if (earlyStopPatience && noImproveCount >= earlyStopPatience) break;
      if (finalError <= targetError) break;
    }
    net.nodes.forEach((n) => {
      if (n.type === "hidden") n.mask = 1;
    });
    net.dropout = 0;
    internalNet._globalEpoch = (internalNet._globalEpoch || 0) + performedIterations;
    return {
      /** Final monitored (possibly smoothed) error achieved at termination. */
      error: finalError,
      /** Number of iterations actually executed (could be < requested iterations due to early stop). */
      iterations: performedIterations,
      /** Wall-clock training duration in milliseconds. */
      time: Date.now() - start2
    };
  }
  var __trainingInternals;
  var init_network_training = __esm({
    "src/architecture/network/network.training.ts"() {
      "use strict";
      init_methods();
      init_config();
      __trainingInternals = {
        computeMonitoredError,
        computePlateauMetric
      };
    }
  });

  // src/architecture/network/network.evolve.ts
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
            console.warn(
              `Genome evaluation failed: ${e && e.message || e}. Penalizing with -Infinity fitness.`
            );
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
        console.warn(
          "Failed to load worker class; falling back to single-thread path:",
          e?.message || e
        );
    }
    if (!WorkerCtor)
      return {
        fitnessFunction: buildSingleThreadFitness(set, cost, amount, growth),
        threads: 1
      };
    for (let i = 0; i < threads; i++) {
      try {
        workers.push(
          new WorkerCtor(serializedSet, {
            name: cost.name || cost.toString?.() || "cost"
          })
        );
      } catch (e) {
        if (config.warnings) console.warn("Worker spawn failed", e);
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
          if (--active === 0) resolve();
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
      throw new Error(
        "Dataset is invalid or dimensions do not match network input/output size!"
      );
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
    const start2 = Date.now();
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
      throw new Error(
        "At least one stopping condition (`iterations` or `error`) must be specified for evolution."
      );
    } else if (typeof options.error === "undefined") targetError = -1;
    else if (typeof options.iterations === "undefined") options.iterations = 0;
    let fitnessFunction;
    if (threads === 1)
      fitnessFunction = buildSingleThreadFitness(set, cost, amount, growth);
    else {
      const multi = await buildMultiThreadFitness(
        set,
        cost,
        amount,
        growth,
        threads,
        options
      );
      fitnessFunction = multi.fitnessFunction;
      threads = multi.threads;
    }
    options.network = this;
    if (options.populationSize != null && options.popsize == null)
      options.popsize = options.populationSize;
    if (typeof options.speciation === "undefined") options.speciation = false;
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
        if (++infiniteErrorCount >= MAX_INF) break;
      } else infiniteErrorCount = 0;
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
      if (clear) this.clear();
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
    return { error, iterations: neat.generation, time: Date.now() - start2 };
  }
  var _complexityCache;
  var init_network_evolve = __esm({
    "src/architecture/network/network.evolve.ts"() {
      "use strict";
      init_network();
      init_methods();
      init_config();
      init_multi();
      _complexityCache = /* @__PURE__ */ new WeakMap();
    }
  });

  // src/architecture/network.ts
  var network_exports = {};
  __export(network_exports, {
    default: () => Network3
  });
  var Network3;
  var init_network = __esm({
    "src/architecture/network.ts"() {
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
      Network3 = class _Network {
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
        // baseline for evolution-time pruning
        _activationPrecision = "f64";
        // typed array precision for compiled path
        _reuseActivationArrays = false;
        // reuse pooled output arrays
        _returnTypedActivations = false;
        // if true and reuse enabled, return typed array directly
        _activationPool;
        // pooled output array
        // Packed connection slab fields (for memory + cache efficiency when iterating connections)
        _connWeights;
        _connFrom;
        _connTo;
        _slabDirty = true;
        _useFloat32Weights = true;
        // Cached node.index maintenance (avoids repeated this.nodes.indexOf in hot paths like slab rebuild)
        _nodeIndexDirty = true;
        // when true, node.index values must be reassigned sequentially
        // Fast slab forward path structures
        _outStart;
        _outOrder;
        _adjDirty = true;
        // Cached typed arrays for fast slab forward pass
        _fastA;
        _fastS;
        // Internal hint: track a preferred linear chain edge to split on subsequent ADD_NODE mutations
        // to encourage deep path formation even in stochastic modes. Updated each time we split it.
        _preferredChainEdge;
        // Slab helpers delegated to network.slab.ts
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
        /**
         * Public wrapper for fast slab forward pass (primarily for tests / benchmarking).
         * Prefer using standard activate(); it will auto dispatch when eligible.
         * Falls back internally if prerequisites not met.
         */
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
          if (options?.reuseActivationArrays) this._reuseActivationArrays = true;
          if (options?.returnTypedActivations) this._returnTypedActivations = true;
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
            else this.nodes.push(new Node2(type, void 0, this._rand));
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
        // --- Changed: made public (was private) for deterministic pooling stress harness ---
        addNodeBetween() {
          if (this.connections.length === 0) return;
          const idx = Math.floor(this._rand() * this.connections.length);
          const conn = this.connections[idx];
          if (!conn) return;
          this.disconnect(conn.from, conn.to);
          const newNode = config.enableNodePooling ? acquireNode({ type: "hidden", rng: this._rand }) : new Node2("hidden", void 0, this._rand);
          this.nodes.push(newNode);
          this.connect(conn.from, newNode, conn.weight);
          this.connect(newNode, conn.to, 1);
          this._topoDirty = true;
          this._nodeIndexDirty = true;
        }
        // --- DropConnect API (re-added for tests) ---
        enableDropConnect(p) {
          if (p < 0 || p >= 1)
            throw new Error("DropConnect probability must be in [0,1)");
          this._dropConnectProb = p;
        }
        disableDropConnect() {
          this._dropConnectProb = 0;
        }
        // --- Acyclic enforcement toggle (used by tests) ---
        setEnforceAcyclic(flag) {
          this._enforceAcyclic = !!flag;
        }
        _computeTopoOrder() {
          return computeTopoOrder.call(this);
        }
        _hasPath(from, to) {
          return hasPath.call(this, from, to);
        }
        // --- Pruning configuration & helpers ---
        configurePruning(cfg) {
          const { start: start2, end, targetSparsity } = cfg;
          if (start2 < 0 || end < start2)
            throw new Error("Invalid pruning schedule window");
          if (targetSparsity <= 0 || targetSparsity >= 1)
            throw new Error("targetSparsity must be in (0,1)");
          this._pruningConfig = {
            start: start2,
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
        /**
         * Immediately prune connections to reach (or approach) a target sparsity fraction.
         * Used by evolutionary pruning (generation-based) independent of training iteration schedule.
         * @param targetSparsity fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
         * @param method 'magnitude' | 'snip'
         */
        pruneToSparsity(targetSparsity, method = "magnitude") {
          return pruneToSparsity.call(this, targetSparsity, method);
        }
        /** Enable weight noise. Provide a single std dev number or { perHiddenLayer: number[] }. */
        enableWeightNoise(stdDev) {
          if (typeof stdDev === "number") {
            if (stdDev < 0) throw new Error("Weight noise stdDev must be >= 0");
            this._weightNoiseStd = stdDev;
            this._weightNoisePerHidden = [];
          } else if (stdDev && Array.isArray(stdDev.perHiddenLayer)) {
            if (!this.layers || this.layers.length < 3)
              throw new Error(
                "Per-hidden-layer weight noise requires a layered network with at least one hidden layer"
              );
            const hiddenLayerCount = this.layers.length - 2;
            if (stdDev.perHiddenLayer.length !== hiddenLayerCount)
              throw new Error(
                `Expected ${hiddenLayerCount} std dev entries (one per hidden layer), got ${stdDev.perHiddenLayer.length}`
              );
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
        /** Configure stochastic depth with survival probabilities per hidden layer (length must match hidden layer count when using layered network). */
        setStochasticDepth(survival) {
          if (!Array.isArray(survival)) throw new Error("survival must be an array");
          if (survival.some((p) => p <= 0 || p > 1))
            throw new Error("Stochastic depth survival probs must be in (0,1]");
          if (!this.layers || this.layers.length === 0)
            throw new Error("Stochastic depth requires layer-based network");
          const hiddenLayerCount = Math.max(0, this.layers.length - 2);
          if (survival.length !== hiddenLayerCount)
            throw new Error(
              `Expected ${hiddenLayerCount} survival probabilities for hidden layers, got ${survival.length}`
            );
          this._stochasticDepth = survival.slice();
        }
        disableStochasticDepth() {
          this._stochasticDepth = [];
        }
        /**
         * Creates a deep copy of the network.
         * @returns {Network} A new Network instance that is a clone of the current network.
         */
        clone() {
          return _Network.fromJSON(this.toJSON());
        }
        /**
         * Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
         * Should be called after training to ensure inference is unaffected by previous dropout.
         */
        resetDropoutMasks() {
          if (this.layers && this.layers.length > 0) {
            for (const layer of this.layers) {
              if (typeof layer.nodes !== "undefined") {
                for (const node of layer.nodes) {
                  if (typeof node.mask !== "undefined") node.mask = 1;
                }
              }
            }
          } else {
            for (const node of this.nodes) {
              if (typeof node.mask !== "undefined") node.mask = 1;
            }
          }
        }
        // Delegated standalone generator
        standalone() {
          return generateStandalone(this);
        }
        /**
         * Activates the network using the given input array.
         * Performs a forward pass through the network, calculating the activation of each node.
         *
         * @param {number[]} input - An array of numerical values corresponding to the network's input nodes.
         * @param {boolean} [training=false] - Flag indicating if the activation is part of a training process.
         * @param {number} [maxActivationDepth=1000] - Maximum allowed activation depth to prevent infinite loops/cycles.
         * @returns {number[]} An array of numerical values representing the activations of the network's output nodes.
         */
        /**
         * Standard activation API returning a plain number[] for backward compatibility.
         * Internally may use pooled typed arrays; if so they are cloned before returning.
         */
        activate(input, training = false, maxActivationDepth = 1e3) {
          if (this._enforceAcyclic && this._topoDirty) this._computeTopoOrder();
          if (!Array.isArray(input) || input.length !== this.input) {
            throw new Error(
              `Input size mismatch: expected ${this.input}, got ${input ? input.length : "undefined"}`
            );
          }
          if (this._canUseFastSlab(training)) {
            try {
              return this._fastSlabActivate(input);
            } catch {
            }
          }
          const outputArr = activationArrayPool.acquire(this.output);
          if (!this.nodes || this.nodes.length === 0) {
            throw new Error(
              "Network structure is corrupted or empty. No nodes found."
            );
          }
          let output = outputArr;
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
                if (c._origWeightNoise != null) continue;
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
            const updated = this._stochasticDepthSchedule(
              this._trainingStep,
              this._stochasticDepth.slice()
            );
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
                    if (!acts || acts.length !== layer.nodes.length) skip = false;
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
                  if (node.mask === 0) stats.droppedHiddenNodes++;
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
                for (const node of layer.nodes) node.mask = 1;
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
            let hiddenNodes = this.nodes.filter((node) => node.type === "hidden");
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
              for (const node of hiddenNodes) node.mask = 1;
            }
            if (training && this._weightNoiseStd > 0) {
              if (!this._wnOrig) this._wnOrig = new Array(this.connections.length);
              for (let ci = 0; ci < this.connections.length; ci++) {
                const c = this.connections[ci];
                if (c._origWeightNoise != null) continue;
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
                if (mask === 0) stats.droppedConnections++;
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
          if (training) this._trainingStep++;
          if (stats.weightNoise.count > 0)
            stats.weightNoise.meanAbs = stats.weightNoise.sumAbs / stats.weightNoise.count;
          this._lastStats = stats;
          const result = Array.from(output);
          activationArrayPool.release(output);
          return result;
        }
        static _gaussianRand(rng = Math.random) {
          let u = 0, v = 0;
          while (u === 0) u = rng();
          while (v === 0) v = rng();
          return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }
        /**
         * Activates the network without calculating eligibility traces.
         * This is a performance optimization for scenarios where backpropagation is not needed,
         * such as during testing, evaluation, or deployment (inference).
         *
         * @param {number[]} input - An array of numerical values corresponding to the network's input nodes.
         *                           The length must match the network's `input` size.
         * @returns {number[]} An array of numerical values representing the activations of the network's output nodes.
         *
         * @see {@link Node.noTraceActivate}
         */
        // Delegated activation helpers
        noTraceActivate(input) {
          const { noTraceActivate: noTraceActivate2 } = (init_network_activate(), __toCommonJS(network_activate_exports));
          return noTraceActivate2.call(this, input);
        }
        /**
         * Raw activation that can return a typed array when pooling is enabled (zero-copy).
         * If reuseActivationArrays=false falls back to standard activate().
         */
        activateRaw(input, training = false, maxActivationDepth = 1e3) {
          const { activateRaw: activateRaw2 } = (init_network_activate(), __toCommonJS(network_activate_exports));
          return activateRaw2.call(this, input, training, maxActivationDepth);
        }
        /**
         * Activate the network over a batch of input vectors (micro-batching).
         *
         * Currently iterates sample-by-sample while reusing the network's internal
         * fast-path allocations. Outputs are cloned number[] arrays for API
         * compatibility. Future optimizations can vectorize this path.
         *
         * @param inputs Array of input vectors, each length must equal this.input
         * @param training Whether to run with training-time stochastic features
         * @returns Array of output vectors, each length equals this.output
         */
        activateBatch(inputs, training = false) {
          const { activateBatch: activateBatch2 } = (init_network_activate(), __toCommonJS(network_activate_exports));
          return activateBatch2.call(this, inputs, training);
        }
        /**
         * Propagates the error backward through the network (backpropagation).
         * Calculates the error gradient for each node and connection.
         * If `update` is true, it adjusts the weights and biases based on the calculated gradients,
         * learning rate, momentum, and optional L2 regularization.
         *
         * The process starts from the output nodes and moves backward layer by layer (or topologically for recurrent nets).
         *
         * @param {number} rate - The learning rate (controls the step size of weight adjustments).
         * @param {number} momentum - The momentum factor (helps overcome local minima and speeds up convergence). Typically between 0 and 1.
         * @param {boolean} update - If true, apply the calculated weight and bias updates. If false, only calculate gradients (e.g., for batch accumulation).
         * @param {number[]} target - An array of target values corresponding to the network's output nodes.
         *                            The length must match the network's `output` size.
         * @param {number} [regularization=0] - The L2 regularization factor (lambda). Helps prevent overfitting by penalizing large weights.
         * @param {(target: number, output: number) => number} [costDerivative] - Optional derivative of the cost function for output nodes.
         * @throws {Error} If the `target` array length does not match the network's `output` size.
         *
         * @see {@link Node.propagate} for the node-level backpropagation logic.
         */
        propagate(rate, momentum, update, target, regularization = 0, costDerivative) {
          if (!target || target.length !== this.output) {
            throw new Error(
              "Output target length should match network output length"
            );
          }
          let targetIndex = target.length;
          for (let i = this.nodes.length - 1; i >= this.nodes.length - this.output; i--) {
            if (costDerivative) {
              this.nodes[i].propagate(
                rate,
                momentum,
                update,
                regularization,
                target[--targetIndex],
                costDerivative
              );
            } else {
              this.nodes[i].propagate(
                rate,
                momentum,
                update,
                regularization,
                target[--targetIndex]
              );
            }
          }
          for (let i = this.nodes.length - this.output - 1; i >= this.input; i--) {
            this.nodes[i].propagate(rate, momentum, update, regularization);
          }
        }
        /**
         * Clears the internal state of all nodes in the network.
         * Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
         * This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.
         *
         * @see {@link Node.clear}
         */
        clear() {
          this.nodes.forEach((node) => node.clear());
        }
        /**
         * Mutates the network's structure or parameters according to the specified method.
         * This is a core operation for neuro-evolutionary algorithms (like NEAT).
         * The method argument should be one of the mutation types defined in `methods.mutation`.
         *
         * @param {any} method - The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
         *                       Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).
         * @throws {Error} If no valid mutation `method` is provided.
         *
         * @see {@link methods.mutation} for available mutation types.
         */
        mutate(method) {
          const { mutateImpl: mutateImpl2 } = (init_network_mutate(), __toCommonJS(network_mutate_exports));
          return mutateImpl2.call(this, method);
        }
        /**
         * Creates a connection between two nodes in the network.
         * Handles both regular connections and self-connections.
         * Adds the new connection object(s) to the appropriate network list (`connections` or `selfconns`).
         *
         * @param {Node} from - The source node of the connection.
         * @param {Node} to - The target node of the connection.
         * @param {number} [weight] - Optional weight for the connection. If not provided, a random weight is usually assigned by the underlying `Node.connect` method.
         * @returns {Connection[]} An array containing the newly created connection object(s). Typically contains one connection, but might be empty or contain more in specialized node types.
         *
         * @see {@link Node.connect}
         */
        connect(from, to, weight) {
          return connect.call(this, from, to, weight);
        }
        /**
         * Gates a connection with a specified node.
         * The activation of the `node` (gater) will modulate the weight of the `connection`.
         * Adds the connection to the network's `gates` list.
         *
         * @param {Node} node - The node that will act as the gater. Must be part of this network.
         * @param {Connection} connection - The connection to be gated.
         * @throws {Error} If the provided `node` is not part of this network.
         * @throws {Error} If the `connection` is already gated (though currently handled with a warning).
         *
         * @see {@link Node.gate}
         */
        gate(node, connection) {
          return gate.call(this, node, connection);
        }
        /**
         * Removes a node from the network.
         * This involves:
         * 1. Disconnecting all incoming and outgoing connections associated with the node.
         * 2. Removing any self-connections.
         * 3. Removing the node from the `nodes` array.
         * 4. Attempting to reconnect the node's direct predecessors to its direct successors
         *    to maintain network flow, if possible and configured.
         * 5. Handling gates involving the removed node (ungating connections gated *by* this node,
         *    and potentially re-gating connections that were gated *by other nodes* onto the removed node's connections).
         *
         * @param {Node} node - The node instance to remove. Must exist within the network's `nodes` list.
         * @throws {Error} If the specified `node` is not found in the network's `nodes` list.
         */
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
        /**
         * Disconnects two nodes, removing the connection between them.
         * Handles both regular connections and self-connections.
         * If the connection being removed was gated, it is also ungated.
         *
         * @param {Node} from - The source node of the connection to remove.
         * @param {Node} to - The target node of the connection to remove.
         *
         * @see {@link Node.disconnect}
         */
        disconnect(from, to) {
          return disconnect.call(this, from, to);
        }
        // slab rebuild + accessor moved to network.slab.ts
        /**
         * Removes the gate from a specified connection.
         * The connection will no longer be modulated by its gater node.
         * Removes the connection from the network's `gates` list.
         *
         * @param {Connection} connection - The connection object to ungate.
         * @throws {Error} If the provided `connection` is not found in the network's `gates` list (i.e., it wasn't gated).
         *
         * @see {@link Node.ungate}
         */
        ungate(connection) {
          return ungate.call(this, connection);
        }
        /**
         * Trains the network on a given dataset subset for one pass (epoch or batch).
         * Performs activation and backpropagation for each item in the set.
         * Updates weights based on batch size configuration.
         *
         * @param {{ input: number[]; output: number[] }[]} set - The training dataset subset (e.g., a batch or the full set for one epoch).
         * @param {number} batchSize - The number of samples to process before updating weights.
         * @param {number} currentRate - The learning rate to use for this training pass.
         * @param {number} momentum - The momentum factor to use.
         * @param {any} regularization - The regularization configuration (L1, L2, or custom function).
         * @param {(target: number[], output: number[]) => number} costFunction - The function used to calculate the error between target and output.
         * @returns {number} The average error calculated over the provided dataset subset.
         * @private Internal method used by `train`.
         */
        _applyGradientClipping(cfg) {
          const { applyGradientClippingImpl: applyGradientClippingImpl2 } = (init_network_training(), __toCommonJS(network_training_exports));
          applyGradientClippingImpl2(this, cfg);
        }
        // Training is implemented in network.training.ts; this wrapper keeps public API stable.
        train(set, options) {
          const { trainImpl: trainImpl2 } = (init_network_training(), __toCommonJS(network_training_exports));
          return trainImpl2(this, set, options);
        }
        /** Returns last recorded raw (pre-update) gradient L2 norm. */
        getRawGradientNorm() {
          return this._lastRawGradNorm;
        }
        /** Returns current mixed precision loss scale (1 if disabled). */
        getLossScale() {
          return this._mixedPrecision.lossScale;
        }
        /** Returns last gradient clipping group count (0 if no clipping yet). */
        getLastGradClipGroupCount() {
          return this._lastGradClipGroupCount;
        }
        /** Consolidated training stats snapshot. */
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
        /** Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average'). */
        static adjustRateForAccumulation(rate, accumulationSteps, reduction) {
          if (reduction === "sum" && accumulationSteps > 1)
            return rate / accumulationSteps;
          return rate;
        }
        // Evolution wrapper delegates to network/network.evolve.ts implementation.
        async evolve(set, options) {
          const { evolveNetwork: evolveNetwork2 } = await Promise.resolve().then(() => (init_network_evolve(), network_evolve_exports));
          return evolveNetwork2.call(this, set, options);
        }
        /**
         * Tests the network's performance on a given dataset.
         * Calculates the average error over the dataset using a specified cost function.
         * Uses `noTraceActivate` for efficiency as gradients are not needed.
         * Handles dropout scaling if dropout was used during training.
         *
         * @param {{ input: number[]; output: number[] }[]} set - The test dataset, an array of objects with `input` and `output` arrays.
         * @param {function} [cost=methods.Cost.MSE] - The cost function to evaluate the error. Defaults to Mean Squared Error.
         * @returns {{ error: number; time: number }} An object containing the calculated average error over the dataset and the time taken for the test in milliseconds.
         */
        test(set, cost) {
          if (!Array.isArray(set) || set.length === 0) {
            throw new Error("Test set is empty or not an array.");
          }
          for (const sample of set) {
            if (!Array.isArray(sample.input) || sample.input.length !== this.input) {
              throw new Error(
                `Test sample input size mismatch: expected ${this.input}, got ${sample.input ? sample.input.length : "undefined"}`
              );
            }
            if (!Array.isArray(sample.output) || sample.output.length !== this.output) {
              throw new Error(
                `Test sample output size mismatch: expected ${this.output}, got ${sample.output ? sample.output.length : "undefined"}`
              );
            }
          }
          let error = 0;
          const costFn = cost || Cost.mse;
          const start2 = Date.now();
          this.nodes.forEach((node) => {
            if (node.type === "hidden") node.mask = 1;
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
          return { error: error / set.length, time: Date.now() - start2 };
        }
        /** Lightweight tuple serializer delegating to network.serialize.ts */
        serialize() {
          return serialize.call(this);
        }
        /**
         * Creates a Network instance from serialized data produced by `serialize()`.
         * Reconstructs the network structure and state based on the provided arrays.
         *
         * @param {any[]} data - The serialized network data array, typically obtained from `network.serialize()`.
         *                       Expected format: `[activations, states, squashNames, connectionData, inputSize, outputSize]`.
         * @param {number} [inputSize] - Optional input size override.
         * @param {number} [outputSize] - Optional output size override.
         * @returns {Network} A new Network instance reconstructed from the serialized data.
         * @static
         */
        /** Static lightweight tuple deserializer delegate */
        static deserialize(data, inputSize, outputSize) {
          return deserialize(data, inputSize, outputSize);
        }
        /**
         * Converts the network into a JSON object representation (latest standard).
         * Includes formatVersion, and only serializes properties needed for full reconstruction.
         * All references are by index. Excludes runtime-only properties (activation, state, traces).
         *
         * @returns {object} A JSON-compatible object representing the network.
         */
        /** Verbose JSON serializer delegate */
        toJSON() {
          return toJSONImpl.call(this);
        }
        /**
         * Reconstructs a network from a JSON object (latest standard).
         * Handles formatVersion, robust error handling, and index-based references.
         * @param {object} json - The JSON object representing the network.
         * @returns {Network} The reconstructed network.
         */
        /** Verbose JSON static deserializer */
        static fromJSON(json) {
          return fromJSONImpl(json);
        }
        /**
         * Creates a new offspring network by performing crossover between two parent networks.
         * This method implements the crossover mechanism inspired by the NEAT algorithm and described
         * in the Instinct paper, combining genes (nodes and connections) from both parents.
         * Fitness scores can influence the inheritance process. Matching genes are inherited randomly,
         * while disjoint/excess genes are typically inherited from the fitter parent (or randomly if fitness is equal or `equal` flag is set).
         *
         * @param {Network} network1 - The first parent network.
         * @param {Network} network2 - The second parent network.
         * @param {boolean} [equal=false] - If true, disjoint and excess genes are inherited randomly regardless of fitness.
         *                                  If false (default), they are inherited from the fitter parent.
         * @returns {Network} A new Network instance representing the offspring.
         * @throws {Error} If the input or output sizes of the parent networks do not match.
         *
         * @see Instinct Algorithm - Section 2 Crossover
         * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6}
         * @static
         */
        /** NEAT-style crossover delegate. */
        static crossOver(network1, network2, equal = false) {
          return crossOver(network1, network2, equal);
        }
        /**
         * Sets specified properties (e.g., bias, squash function) for all nodes in the network.
         * Useful for initializing or resetting node properties uniformly.
         *
         * @param {object} values - An object containing the properties and values to set.
         * @param {number} [values.bias] - If provided, sets the bias for all nodes.
         * @param {function} [values.squash] - If provided, sets the squash (activation) function for all nodes.
         *                                     Should be a valid activation function (e.g., from `methods.Activation`).
         */
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
        /**
         * Exports the network to ONNX format (JSON object, minimal MLP support).
         * Only standard feedforward architectures and standard activations are supported.
         * Gating, custom activations, and evolutionary features are ignored or replaced with Identity.
         *
         * @returns {import('./onnx').OnnxModel} ONNX model as a JSON object.
         */
        toONNX() {
          return exportToONNX(this);
        }
        /**
         * Creates a fully connected, strictly layered MLP network.
         * @param {number} inputCount - Number of input nodes
         * @param {number[]} hiddenCounts - Array of hidden layer sizes (e.g. [2,3] for two hidden layers)
         * @param {number} outputCount - Number of output nodes
         * @returns {Network} A new, fully connected, layered MLP
         */
        static createMLP(inputCount, hiddenCounts, outputCount) {
          const inputNodes = Array.from(
            { length: inputCount },
            () => new Node2("input")
          );
          const hiddenLayers = hiddenCounts.map(
            (count) => Array.from({ length: count }, () => new Node2("hidden"))
          );
          const outputNodes = Array.from(
            { length: outputCount },
            () => new Node2("output")
          );
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
        /**
         * Rebuilds the network's connections array from all per-node connections.
         * This ensures that the network.connections array is consistent with the actual
         * outgoing connections of all nodes. Useful after manual wiring or node manipulation.
         *
         * @param {Network} net - The network instance to rebuild connections for.
         * @returns {void}
         *
         * Example usage:
         *   Network.rebuildConnections(net);
         */
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

  // src/neat/neat.mutation.ts
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
              const statsRecord = this._operatorStats.get(
                mutationMethod.name
              ) || {
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
    const enabledConnections = genome.connections.filter(
      (c) => c.enabled !== false
    );
    if (!enabledConnections.length) return;
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
      if (inConn) inConn.innovation = this._nextGlobalInnovation++;
      if (outConn) outConn.innovation = this._nextGlobalInnovation++;
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
      if (inConn) inConn.innovation = splitRecord.inInnov;
      if (outConn) outConn.innovation = splitRecord.outInnov;
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
    if (!candidatePairs.length) return;
    const reuseCandidates = candidatePairs.filter((pair) => {
      const idA2 = pair[0].geneId;
      const idB2 = pair[1].geneId;
      const symmetricKey2 = idA2 < idB2 ? idA2 + "::" + idB2 : idB2 + "::" + idA2;
      return this._connInnovations.has(symmetricKey2);
    });
    const hiddenPairs = reuseCandidates.length ? [] : candidatePairs.filter(
      (pair) => pair[0].type === "hidden" && pair[1].type === "hidden"
    );
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
          if (n === fromNode) return true;
          if (seen.has(n)) continue;
          seen.add(n);
          for (const c of n.connections.out) stack.push(c.to);
        }
        return false;
      })();
      if (createsCycle) return;
    }
    const conn = genome.connect(fromNode, toNode)[0];
    if (!conn) return;
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
    const minHidden = Math.min(
      this.getMinimumHiddenSize(multiplierOverride),
      maxNodes - network.nodes.filter((n) => n.type !== "hidden").length
    );
    const inputNodes = network.nodes.filter((n) => n.type === "input");
    const outputNodes = network.nodes.filter((n) => n.type === "output");
    let hiddenNodes = network.nodes.filter((n) => n.type === "hidden");
    if (inputNodes.length === 0 || outputNodes.length === 0) {
      try {
        console.warn(
          "Network is missing input or output nodes \u2014 skipping minHidden enforcement"
        );
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
        const candidates = inputNodes.concat(
          hiddenNodes.filter((n) => n !== hiddenNode)
        );
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
        const candidates = outputNodes.concat(
          hiddenNodes.filter((n) => n !== hiddenNode)
        );
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
        const candidates = inputNodes.concat(
          hiddenNodes.filter((n) => n !== hiddenNode)
        );
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
        const candidates = outputNodes.concat(
          hiddenNodes.filter((n) => n !== hiddenNode)
        );
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
    if (rawReturnForTest && Array.isArray(pool2) && pool2.length === methods.mutation.FFW.length && pool2.every(
      (m, i) => m && m.name === methods.mutation.FFW[i].name
    )) {
      return methods.mutation.FFW;
    }
    if (pool2.length === 1 && Array.isArray(pool2[0]) && pool2[0].length)
      pool2 = pool2[0];
    if (this.options.phasedComplexity?.enabled && this._phase) {
      pool2 = pool2.filter((m) => !!m);
      if (this._phase === "simplify") {
        const simplifyPool = pool2.filter(
          (m) => m && m.name && m.name.startsWith && m.name.startsWith("SUB_")
        );
        if (simplifyPool.length) pool2 = [...pool2, ...simplifyPool];
      } else if (this._phase === "complexify") {
        const addPool = pool2.filter(
          (m) => m && m.name && m.name.startsWith && m.name.startsWith("ADD_")
        );
        if (addPool.length) pool2 = [...pool2, ...addPool];
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
        if (!stats.has(m.name)) stats.set(m.name, { success: 0, attempts: 0 });
      const totalAttempts = Array.from(stats.values()).reduce(
        (a, s) => a + s.attempts,
        0
      ) + EPSILON;
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
    "src/neat/neat.mutation.ts"() {
      "use strict";
      init_neat_constants();
    }
  });

  // src/neat/neat.multiobjective.ts
  function fastNonDominated(pop) {
    const objectiveDescriptors = this._getObjectives();
    const valuesMatrix = pop.map(
      (genomeItem) => objectiveDescriptors.map((descriptor) => {
        try {
          return descriptor.accessor(genomeItem);
        } catch {
          return 0;
        }
      })
    );
    const vectorDominates = (valuesA, valuesB) => {
      let strictlyBetter = false;
      for (let objectiveIndex = 0; objectiveIndex < valuesA.length; objectiveIndex++) {
        const direction = objectiveDescriptors[objectiveIndex].direction || "max";
        if (direction === "max") {
          if (valuesA[objectiveIndex] < valuesB[objectiveIndex]) return false;
          if (valuesA[objectiveIndex] > valuesB[objectiveIndex])
            strictlyBetter = true;
        } else {
          if (valuesA[objectiveIndex] > valuesB[objectiveIndex]) return false;
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
        if (pIndex === qIndex) continue;
        if (vectorDominates(valuesMatrix[pIndex], valuesMatrix[qIndex]))
          dominatedIndicesByIndex[pIndex].push(qIndex);
        else if (vectorDominates(valuesMatrix[qIndex], valuesMatrix[pIndex]))
          dominationCounts[pIndex]++;
      }
      if (dominationCounts[pIndex] === 0) firstFrontIndices.push(pIndex);
    }
    let currentFrontIndices = firstFrontIndices;
    let currentFrontRank = 0;
    while (currentFrontIndices.length) {
      const nextFrontIndices = [];
      for (const pIndex of currentFrontIndices) {
        pop[pIndex]._moRank = currentFrontRank;
        for (const qIndex of dominatedIndicesByIndex[pIndex]) {
          dominationCounts[qIndex]--;
          if (dominationCounts[qIndex] === 0) nextFrontIndices.push(qIndex);
        }
      }
      paretoFronts.push(currentFrontIndices.map((i) => pop[i]));
      currentFrontIndices = nextFrontIndices;
      currentFrontRank++;
      if (currentFrontRank > 50) break;
    }
    for (const front of paretoFronts) {
      if (front.length === 0) continue;
      for (const genomeItem of front) genomeItem._moCrowd = 0;
      for (let objectiveIndex = 0; objectiveIndex < objectiveDescriptors.length; objectiveIndex++) {
        const sortedByCurrentObjective = front.slice().sort((genomeA, genomeB) => {
          const valA = objectiveDescriptors[objectiveIndex].accessor(genomeA);
          const valB = objectiveDescriptors[objectiveIndex].accessor(genomeB);
          return valA - valB;
        });
        sortedByCurrentObjective[0]._moCrowd = Infinity;
        sortedByCurrentObjective[sortedByCurrentObjective.length - 1]._moCrowd = Infinity;
        const minVal = objectiveDescriptors[objectiveIndex].accessor(
          sortedByCurrentObjective[0]
        );
        const maxVal = objectiveDescriptors[objectiveIndex].accessor(
          sortedByCurrentObjective[sortedByCurrentObjective.length - 1]
        );
        const valueRange = maxVal - minVal || 1;
        for (let sortedIndex = 1; sortedIndex < sortedByCurrentObjective.length - 1; sortedIndex++) {
          const prevVal = objectiveDescriptors[objectiveIndex].accessor(
            sortedByCurrentObjective[sortedIndex - 1]
          );
          const nextVal = objectiveDescriptors[objectiveIndex].accessor(
            sortedByCurrentObjective[sortedIndex + 1]
          );
          sortedByCurrentObjective[sortedIndex]._moCrowd += (nextVal - prevVal) / valueRange;
        }
      }
    }
    if (this.options.multiObjective?.enabled) {
      this._paretoArchive.push({
        generation: this.generation,
        fronts: paretoFronts.slice(0, 3).map(
          (front) => (
            // map each front (array of Network) to an array of genome IDs
            front.map((genome) => genome._id)
          )
        )
      });
      if (this._paretoArchive.length > 100) this._paretoArchive.shift();
    }
    return paretoFronts;
  }
  var init_neat_multiobjective = __esm({
    "src/neat/neat.multiobjective.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.adaptive.ts
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
    if (!this.options.complexityBudget?.enabled) return;
    const complexityBudget = this.options.complexityBudget;
    if (complexityBudget.mode === "adaptive") {
      if (!this._cbHistory) this._cbHistory = [];
      this._cbHistory.push(this.population[0]?.score || 0);
      const windowSize = complexityBudget.improvementWindow ?? 10;
      if (this._cbHistory.length > windowSize) this._cbHistory.shift();
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
      const slopeMag = Math.min(
        2,
        Math.max(-2, slope / (Math.abs(history[0]) + EPSILON))
      );
      const incF = baseInc + 0.05 * Math.max(0, slopeMag);
      const stagF = baseStag - 0.03 * Math.max(0, -slopeMag);
      const noveltyFactor = this._noveltyArchive.length > 5 ? 1 : 0.9;
      if (improvement > 0 || slope > 0)
        this._cbMaxNodes = Math.min(
          complexityBudget.maxNodesEnd ?? this._cbMaxNodes * 4,
          Math.floor(this._cbMaxNodes * incF * noveltyFactor)
        );
      else if (history.length === windowSize)
        this._cbMaxNodes = Math.max(
          complexityBudget.minNodes ?? this.input + this.output + 2,
          Math.floor(this._cbMaxNodes * stagF)
        );
      if (complexityBudget.minNodes !== void 0) {
        this._cbMaxNodes = Math.max(complexityBudget.minNodes, this._cbMaxNodes);
      } else {
        const implicitMin = this.input + this.output + 2;
        if (this._cbMaxNodes < implicitMin) this._cbMaxNodes = implicitMin;
      }
      this.options.maxNodes = this._cbMaxNodes;
      if (complexityBudget.maxConnsStart) {
        if (this._cbMaxConns === void 0)
          this._cbMaxConns = complexityBudget.maxConnsStart;
        if (improvement > 0 || slope > 0)
          this._cbMaxConns = Math.min(
            complexityBudget.maxConnsEnd ?? this._cbMaxConns * 4,
            Math.floor(this._cbMaxConns * incF * noveltyFactor)
          );
        else if (history.length === windowSize)
          this._cbMaxConns = Math.max(
            complexityBudget.maxConnsStart,
            Math.floor(this._cbMaxConns * stagF)
          );
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
    if (!this.options.phasedComplexity?.enabled) return;
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
    if (!this.options.minimalCriterionAdaptive?.enabled) return;
    const mcCfg = this.options.minimalCriterionAdaptive;
    if (this._mcThreshold === void 0)
      this._mcThreshold = mcCfg.initialThreshold ?? 0;
    const scores = this.population.map((g) => g.score || 0);
    const accepted = scores.filter((s) => s >= this._mcThreshold).length;
    const prop = scores.length ? accepted / scores.length : 0;
    const targetAcceptance = mcCfg.targetAcceptance ?? 0.5;
    const adjustRate = mcCfg.adjustRate ?? 0.1;
    if (prop > targetAcceptance * 1.05) this._mcThreshold *= 1 + adjustRate;
    else if (prop < targetAcceptance * 0.95) this._mcThreshold *= 1 - adjustRate;
    for (const g of this.population)
      if ((g.score || 0) < this._mcThreshold) g.score = 0;
  }
  function applyAncestorUniqAdaptive() {
    if (!this.options.ancestorUniqAdaptive?.enabled) return;
    const ancestorCfg = this.options.ancestorUniqAdaptive;
    const cooldown = ancestorCfg.cooldown ?? 5;
    if (this.generation - this._lastAncestorUniqAdjustGen < cooldown) return;
    const lineageBlock = this._telemetry[this._telemetry.length - 1]?.lineage;
    const ancUniq = lineageBlock ? lineageBlock.ancestorUniq : void 0;
    if (typeof ancUniq !== "number") return;
    const lowT = ancestorCfg.lowThreshold ?? 0.25;
    const highT = ancestorCfg.highThreshold ?? 0.55;
    const adj = ancestorCfg.adjust ?? 0.01;
    if (ancestorCfg.mode === "epsilon" && this.options.multiObjective?.adaptiveEpsilon?.enabled) {
      if (ancUniq < lowT) {
        this.options.multiObjective.dominanceEpsilon = (this.options.multiObjective.dominanceEpsilon || 0) + adj;
        this._lastAncestorUniqAdjustGen = this.generation;
      } else if (ancUniq > highT) {
        this.options.multiObjective.dominanceEpsilon = Math.max(
          0,
          (this.options.multiObjective.dominanceEpsilon || 0) - adj
        );
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
    if (!this.options.adaptiveMutation?.enabled) return;
    const adaptCfg = this.options.adaptiveMutation;
    const every = adaptCfg.adaptEvery ?? 1;
    if (!(every <= 1 || this.generation % every === 0)) return;
    const scored = this.population.filter(
      (g) => typeof g.score === "number"
    );
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
      if (genome._mutRate === void 0) continue;
      let rate = genome._mutRate;
      let delta = this._getRNG()() * 2 - 1;
      delta *= sigmaBase;
      if (strategy === "twoTier") {
        if (topHalf.length === 0 || bottomHalf.length === 0)
          delta = index % 2 === 0 ? Math.abs(delta) : -Math.abs(delta);
        else if (topHalf.includes(genome)) delta = -Math.abs(delta);
        else if (bottomHalf.includes(genome)) delta = Math.abs(delta);
      } else if (strategy === "exploreLow") {
        delta = bottomHalf.includes(genome) ? Math.abs(delta * 1.5) : -Math.abs(delta * 0.5);
      } else if (strategy === "anneal") {
        const progress = Math.min(
          1,
          this.generation / (50 + this.population.length)
        );
        delta *= 1 - progress;
      }
      rate += delta;
      if (rate < minR) rate = minR;
      if (rate > maxR) rate = maxR;
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
        if (amt < minA) amt = minA;
        if (amt > maxA) amt = maxA;
        genome._mutAmount = amt;
      }
    }
    if (strategy === "twoTier" && !(anyUp && anyDown)) {
      const baseline = this.options.adaptiveMutation.initialRate ?? 0.5;
      const half = Math.floor(this.population.length / 2);
      for (let i = 0; i < this.population.length; i++) {
        const genome = this.population[i];
        if (genome._mutRate === void 0) continue;
        if (i < half) genome._mutRate = Math.min(genome._mutRate + sigmaBase, 1);
        else genome._mutRate = Math.max(genome._mutRate - sigmaBase, 0.01);
      }
    }
  }
  function applyOperatorAdaptation() {
    if (!this.options.operatorAdaptation?.enabled) return;
    const decay = this.options.operatorAdaptation.decay ?? 0.9;
    for (const [k, stat] of this._operatorStats.entries()) {
      stat.success *= decay;
      stat.attempts *= decay;
      this._operatorStats.set(k, stat);
    }
  }
  var init_neat_adaptive = __esm({
    "src/neat/neat.adaptive.ts"() {
      "use strict";
      init_neat_constants();
    }
  });

  // src/neat/neat.lineage.ts
  var neat_lineage_exports = {};
  __export(neat_lineage_exports, {
    buildAnc: () => buildAnc,
    computeAncestorUniqueness: () => computeAncestorUniqueness
  });
  function buildAnc(genome) {
    const ancestorSet = /* @__PURE__ */ new Set();
    if (!Array.isArray(genome._parents)) return ancestorSet;
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
      if (current.depth > ANCESTOR_DEPTH_WINDOW) continue;
      if (current.id != null) ancestorSet.add(current.id);
      if (current.genomeRef && Array.isArray(current.genomeRef._parents)) {
        for (const parentId of current.genomeRef._parents) {
          queue.push({
            id: parentId,
            // Depth increases as we move one layer further away from the focal genome.
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
    const maxSamplePairs = Math.min(
      MAX_UNIQUENESS_SAMPLE_PAIRS,
      this.population.length * (this.population.length - 1) / 2
    );
    for (let t = 0; t < maxSamplePairs; t++) {
      if (this.population.length < 2) break;
      const indexA = Math.floor(this._getRNG()() * this.population.length);
      let indexB = Math.floor(this._getRNG()() * this.population.length);
      if (indexB === indexA) indexB = (indexB + 1) % this.population.length;
      const ancestorSetA = buildAncestorSet(this.population[indexA]);
      const ancestorSetB = buildAncestorSet(this.population[indexB]);
      if (ancestorSetA.size === 0 && ancestorSetB.size === 0) continue;
      let intersectionCount = 0;
      for (const id of ancestorSetA)
        if (ancestorSetB.has(id)) intersectionCount++;
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
    "src/neat/neat.lineage.ts"() {
      "use strict";
      ANCESTOR_DEPTH_WINDOW = 4;
      MAX_UNIQUENESS_SAMPLE_PAIRS = 30;
    }
  });

  // src/neat/neat.telemetry.ts
  var neat_telemetry_exports = {};
  __export(neat_telemetry_exports, {
    applyTelemetrySelect: () => applyTelemetrySelect,
    buildTelemetryEntry: () => buildTelemetryEntry,
    computeDiversityStats: () => computeDiversityStats,
    recordTelemetryEntry: () => recordTelemetryEntry,
    structuralEntropy: () => structuralEntropy
  });
  function applyTelemetrySelect(entry) {
    if (!this._telemetrySelect || !this._telemetrySelect.size)
      return entry;
    const keep = this._telemetrySelect;
    const core = { gen: entry.gen, best: entry.best, species: entry.species };
    for (const key of Object.keys(entry)) {
      if (key in core) continue;
      if (!keep.has(key)) delete entry[key];
    }
    return Object.assign(entry, core);
  }
  function structuralEntropy(graph) {
    const anyG = graph;
    if (anyG._entropyGen === this.generation && typeof anyG._entropyVal === "number")
      return anyG._entropyVal;
    const degreeCounts = {};
    for (const node of graph.nodes) degreeCounts[node.geneId] = 0;
    for (const conn of graph.connections)
      if (conn.enabled) {
        const fromId = conn.from.geneId;
        const toId = conn.to.geneId;
        if (degreeCounts[fromId] !== void 0) degreeCounts[fromId]++;
        if (degreeCounts[toId] !== void 0) degreeCounts[toId]++;
      }
    const degreeHistogram = {};
    const nodeCount = graph.nodes.length || 1;
    for (const nodeId in degreeCounts) {
      const d = degreeCounts[nodeId];
      degreeHistogram[d] = (degreeHistogram[d] || 0) + 1;
    }
    let entropy = 0;
    for (const k in degreeHistogram) {
      const p = degreeHistogram[k] / nodeCount;
      if (p > 0) entropy -= p * Math.log(p + EPSILON);
    }
    anyG._entropyGen = this.generation;
    anyG._entropyVal = entropy;
    return entropy;
  }
  function computeDiversityStats() {
    if (!this.options.diversityMetrics?.enabled) return;
    if (this.options.fastMode && !this._fastModeTuned) {
      const dm = this.options.diversityMetrics;
      if (dm) {
        if (dm.pairSample == null) dm.pairSample = 20;
        if (dm.graphletSample == null) dm.graphletSample = 30;
      }
      if (this.options.novelty?.enabled && this.options.novelty.k == null)
        this.options.novelty.k = 5;
      this._fastModeTuned = true;
    }
    const pairSample = this.options.diversityMetrics.pairSample ?? 40;
    const graphletSample = this.options.diversityMetrics.graphletSample ?? 60;
    const population = this.population;
    const popSize = population.length;
    let compatSum = 0;
    let compatSq = 0;
    let compatCount = 0;
    for (let iter = 0; iter < pairSample; iter++) {
      if (popSize < 2) break;
      const i = Math.floor(this._getRNG()() * popSize);
      let j = Math.floor(this._getRNG()() * popSize);
      if (j === i) j = (j + 1) % popSize;
      const d = this._compatibilityDistance(
        population[i],
        population[j]
      );
      compatSum += d;
      compatSq += d * d;
      compatCount++;
    }
    const meanCompat = compatCount ? compatSum / compatCount : 0;
    const varCompat = compatCount ? Math.max(0, compatSq / compatCount - meanCompat * meanCompat) : 0;
    const entropies = population.map(
      (g) => this._structuralEntropy(g)
    );
    const meanEntropy = entropies.reduce((a, b) => a + b, 0) / (entropies.length || 1);
    const varEntropy = entropies.length ? entropies.reduce(
      (a, b) => a + (b - meanEntropy) * (b - meanEntropy),
      0
    ) / entropies.length : 0;
    const motifCounts = [0, 0, 0, 0];
    for (let iter = 0; iter < graphletSample; iter++) {
      const g = population[Math.floor(this._getRNG()() * popSize)];
      if (!g) break;
      if (g.nodes.length < 3) continue;
      const selectedIdxs = /* @__PURE__ */ new Set();
      while (selectedIdxs.size < 3)
        selectedIdxs.add(Math.floor(this._getRNG()() * g.nodes.length));
      const selectedNodes = Array.from(selectedIdxs).map((i) => g.nodes[i]);
      let edges = 0;
      for (const c of g.connections)
        if (c.enabled) {
          if (selectedNodes.includes(c.from) && selectedNodes.includes(c.to))
            edges++;
        }
      if (edges > 3) edges = 3;
      motifCounts[edges]++;
    }
    const totalMotifs = motifCounts.reduce((a, b) => a + b, 0) || 1;
    let graphletEntropy = 0;
    for (let k = 0; k < motifCounts.length; k++) {
      const p = motifCounts[k] / totalMotifs;
      if (p > 0) graphletEntropy -= p * Math.log(p);
    }
    let lineageMeanDepth = 0;
    let lineageMeanPairDist = 0;
    if (this._lineageEnabled && popSize > 0) {
      const depths = population.map((g) => g._depth ?? 0);
      lineageMeanDepth = depths.reduce((a, b) => a + b, 0) / popSize;
      let lineagePairSum = 0;
      let lineagePairN = 0;
      for (let iter = 0; iter < Math.min(pairSample, popSize * (popSize - 1) / 2); iter++) {
        if (popSize < 2) break;
        const i = Math.floor(this._getRNG()() * popSize);
        let j = Math.floor(this._getRNG()() * popSize);
        if (j === i) j = (j + 1) % popSize;
        lineagePairSum += Math.abs(depths[i] - depths[j]);
        lineagePairN++;
      }
      lineageMeanPairDist = lineagePairN ? lineagePairSum / lineagePairN : 0;
    }
    this._diversityStats = {
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
    try {
      applyTelemetrySelect.call(this, entry);
    } catch {
    }
    if (!this._telemetry) this._telemetry = [];
    this._telemetry.push(entry);
    try {
      if (this.options.telemetryStream?.enabled && this.options.telemetryStream.onEntry)
        this.options.telemetryStream.onEntry(entry);
    } catch {
    }
    if (this._telemetry.length > 500) this._telemetry.shift();
  }
  function buildTelemetryEntry(fittest) {
    const gen = this.generation;
    let hyperVolumeProxy = 0;
    if (this.options.multiObjective?.enabled) {
      const complexityMetric = this.options.multiObjective.complexityMetric || "connections";
      const primaryObjectiveScores = this.population.map(
        (genome) => genome.score || 0
      );
      const minPrimaryScore = Math.min(...primaryObjectiveScores);
      const maxPrimaryScore = Math.max(...primaryObjectiveScores);
      const paretoFrontSizes = [];
      for (let r = 0; r < 5; r++) {
        const size = this.population.filter(
          (g) => (g._moRank ?? 0) === r
        ).length;
        if (!size) break;
        paretoFrontSizes.push(size);
      }
      for (const genome of this.population) {
        const rank = genome._moRank ?? 0;
        if (rank !== 0) continue;
        const normalizedScore = maxPrimaryScore > minPrimaryScore ? ((genome.score || 0) - minPrimaryScore) / (maxPrimaryScore - minPrimaryScore) : 0;
        const genomeComplexity = complexityMetric === "nodes" ? genome.nodes.length : genome.connections.length;
        hyperVolumeProxy += normalizedScore * (1 / (genomeComplexity + 1));
      }
      const operatorStatsSnapshot = Array.from(
        this._operatorStats.entries()
      ).map(([opName, stats]) => ({
        op: opName,
        succ: stats.success,
        att: stats.attempts
      }));
      const entry2 = {
        gen,
        best: fittest.score,
        species: this._species.length,
        hyper: hyperVolumeProxy,
        fronts: paretoFrontSizes,
        diversity: this._diversityStats,
        ops: operatorStatsSnapshot
      };
      if (!entry2.objImportance) entry2.objImportance = {};
      if (this._lastObjImportance)
        entry2.objImportance = this._lastObjImportance;
      if (this._objectiveAges?.size) {
        entry2.objAges = Array.from(
          this._objectiveAges.entries()
        ).reduce((a, kv) => {
          a[kv[0]] = kv[1];
          return a;
        }, {});
      }
      if (this._pendingObjectiveAdds?.length || this._pendingObjectiveRemoves?.length) {
        entry2.objEvents = [];
        for (const k of this._pendingObjectiveAdds)
          entry2.objEvents.push({ type: "add", key: k });
        for (const k of this._pendingObjectiveRemoves)
          entry2.objEvents.push({ type: "remove", key: k });
        this._objectiveEvents.push(
          ...entry2.objEvents.map((e) => ({ gen, type: e.type, key: e.key }))
        );
        this._pendingObjectiveAdds = [];
        this._pendingObjectiveRemoves = [];
      }
      if (this._lastOffspringAlloc)
        entry2.speciesAlloc = this._lastOffspringAlloc.slice();
      try {
        entry2.objectives = this._getObjectives().map(
          (o) => o.key
        );
      } catch {
      }
      if (this.options.rngState && this._rngState !== void 0)
        entry2.rng = this._rngState;
      if (this._lineageEnabled) {
        const bestGenome = this.population[0];
        const depths = this.population.map(
          (g) => g._depth ?? 0
        );
        this._lastMeanDepth = depths.reduce((a, b) => a + b, 0) / (depths.length || 1);
        const { computeAncestorUniqueness: computeAncestorUniqueness2 } = (init_neat_lineage(), __toCommonJS(neat_lineage_exports));
        const ancestorUniqueness = computeAncestorUniqueness2.call(this);
        entry2.lineage = {
          parents: Array.isArray(bestGenome._parents) ? bestGenome._parents.slice() : [],
          depthBest: bestGenome._depth ?? 0,
          meanDepth: +this._lastMeanDepth.toFixed(2),
          inbreeding: this._prevInbreedingCount,
          ancestorUniq: ancestorUniqueness
        };
      }
      if (this.options.telemetry?.hypervolume && this.options.multiObjective?.enabled)
        entry2.hv = +hyperVolumeProxy.toFixed(4);
      if (this.options.telemetry?.complexity) {
        const nodesArr = this.population.map((g) => g.nodes.length);
        const connsArr = this.population.map(
          (g) => g.connections.length
        );
        const meanNodes = nodesArr.reduce((a, b) => a + b, 0) / (nodesArr.length || 1);
        const meanConns = connsArr.reduce((a, b) => a + b, 0) / (connsArr.length || 1);
        const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
        const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
        const enabledRatios = this.population.map((g) => {
          let enabled = 0, disabled = 0;
          for (const c of g.connections) {
            if (c.enabled === false) disabled++;
            else enabled++;
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
          budgetMaxNodes: this.options.maxNodes,
          budgetMaxConns: this.options.maxConns
        };
      }
      if (this.options.telemetry?.performance)
        entry2.perf = {
          evalMs: this._lastEvalDuration,
          evolveMs: this._lastEvolveDuration
        };
      return entry2;
    }
    const operatorStatsSnapshotMono = Array.from(
      this._operatorStats.entries()
    ).map(([opName, stats]) => ({
      op: opName,
      succ: stats.success,
      att: stats.attempts
    }));
    const entry = {
      gen,
      best: fittest.score,
      species: this._species.length,
      hyper: hyperVolumeProxy,
      diversity: this._diversityStats,
      ops: operatorStatsSnapshotMono,
      objImportance: {}
    };
    if (this._lastObjImportance)
      entry.objImportance = this._lastObjImportance;
    if (this._objectiveAges?.size)
      entry.objAges = Array.from(
        this._objectiveAges.entries()
      ).reduce((a, kv) => {
        a[kv[0]] = kv[1];
        return a;
      }, {});
    if (this._pendingObjectiveAdds?.length || this._pendingObjectiveRemoves?.length) {
      entry.objEvents = [];
      for (const k of this._pendingObjectiveAdds)
        entry.objEvents.push({ type: "add", key: k });
      for (const k of this._pendingObjectiveRemoves)
        entry.objEvents.push({ type: "remove", key: k });
      this._objectiveEvents.push(
        ...entry.objEvents.map((e) => ({ gen, type: e.type, key: e.key }))
      );
      this._pendingObjectiveAdds = [];
      this._pendingObjectiveRemoves = [];
    }
    if (this._lastOffspringAlloc)
      entry.speciesAlloc = this._lastOffspringAlloc.slice();
    try {
      entry.objectives = this._getObjectives().map(
        (o) => o.key
      );
    } catch {
    }
    if (this.options.rngState && this._rngState !== void 0)
      entry.rng = this._rngState;
    if (this._lineageEnabled) {
      const bestGenome = this.population[0];
      const depths = this.population.map(
        (g) => g._depth ?? 0
      );
      this._lastMeanDepth = depths.reduce((a, b) => a + b, 0) / (depths.length || 1);
      const { buildAnc: buildAnc2 } = (init_neat_lineage(), __toCommonJS(neat_lineage_exports));
      let sampledPairs = 0;
      let jaccardSum = 0;
      const samplePairs = Math.min(
        30,
        this.population.length * (this.population.length - 1) / 2
      );
      for (let t = 0; t < samplePairs; t++) {
        if (this.population.length < 2) break;
        const i = Math.floor(
          this._getRNG()() * this.population.length
        );
        let j = Math.floor(
          this._getRNG()() * this.population.length
        );
        if (j === i) j = (j + 1) % this.population.length;
        const ancestorsA = buildAnc2.call(
          this,
          this.population[i]
        );
        const ancestorsB = buildAnc2.call(
          this,
          this.population[j]
        );
        if (ancestorsA.size === 0 && ancestorsB.size === 0) continue;
        let intersectionCount = 0;
        for (const id of ancestorsA) if (ancestorsB.has(id)) intersectionCount++;
        const union = ancestorsA.size + ancestorsB.size - intersectionCount || 1;
        const jaccardDistance = 1 - intersectionCount / union;
        jaccardSum += jaccardDistance;
        sampledPairs++;
      }
      const ancestorUniqueness = sampledPairs ? +(jaccardSum / sampledPairs).toFixed(3) : 0;
      entry.lineage = {
        parents: Array.isArray(bestGenome._parents) ? bestGenome._parents.slice() : [],
        depthBest: bestGenome._depth ?? 0,
        meanDepth: +this._lastMeanDepth.toFixed(2),
        inbreeding: this._prevInbreedingCount,
        ancestorUniq: ancestorUniqueness
      };
    }
    if (this.options.telemetry?.hypervolume && this.options.multiObjective?.enabled)
      entry.hv = +hyperVolumeProxy.toFixed(4);
    if (this.options.telemetry?.complexity) {
      const nodesArr = this.population.map((g) => g.nodes.length);
      const connsArr = this.population.map(
        (g) => g.connections.length
      );
      const meanNodes = nodesArr.reduce((a, b) => a + b, 0) / (nodesArr.length || 1);
      const meanConns = connsArr.reduce((a, b) => a + b, 0) / (connsArr.length || 1);
      const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
      const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
      const enabledRatios = this.population.map((g) => {
        let en = 0, dis = 0;
        for (const c of g.connections) {
          if (c.enabled === false) dis++;
          else en++;
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
        budgetMaxNodes: this.options.maxNodes,
        budgetMaxConns: this.options.maxConns
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
    "src/neat/neat.telemetry.ts"() {
      "use strict";
      init_neat_constants();
    }
  });

  // src/neat/neat.pruning.ts
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
      const progressThroughRamp = Math.min(
        1,
        Math.max(
          0,
          (this.generation - evolutionPruningOpts.startGeneration) / rampGenerations
        )
      );
      rampFraction = progressThroughRamp;
    }
    const targetSparsityNow = (evolutionPruningOpts.targetSparsity || 0) * rampFraction;
    for (const genome of this.population) {
      if (genome && typeof genome.pruneToSparsity === "function") {
        genome.pruneToSparsity(
          targetSparsityNow,
          evolutionPruningOpts.method || "magnitude"
        );
      }
    }
  }
  function applyAdaptivePruning() {
    if (!this.options.adaptivePruning?.enabled) return;
    const adaptivePruningOpts = this.options.adaptivePruning;
    if (this._adaptivePruneLevel === void 0) this._adaptivePruneLevel = 0;
    const metricName = adaptivePruningOpts.metric || "connections";
    const meanNodeCount = this.population.reduce((acc, g) => acc + g.nodes.length, 0) / (this.population.length || 1);
    const meanConnectionCount = this.population.reduce(
      (acc, g) => acc + g.connections.length,
      0
    ) / (this.population.length || 1);
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
      this._adaptivePruneLevel = Math.max(
        0,
        Math.min(
          desiredSparsity,
          this._adaptivePruneLevel + adjustRate * (normalizedDifference > 0 ? 1 : -1)
        )
      );
      for (const g of this.population)
        if (typeof g.pruneToSparsity === "function")
          g.pruneToSparsity(this._adaptivePruneLevel, "magnitude");
    }
  }
  var init_neat_pruning = __esm({
    "src/neat/neat.pruning.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.evolve.ts
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
      const crowdingDistances = new Array(
        populationSnapshot.length
      ).fill(0);
      const objectiveValues = objectives.map(
        (obj) => populationSnapshot.map((genome) => obj.accessor(genome))
      );
      for (const front of paretoFronts) {
        const frontIndices = front.map(
          (genome) => this.population.indexOf(genome)
        );
        if (frontIndices.length < 3) {
          frontIndices.forEach((i) => crowdingDistances[i] = Infinity);
          continue;
        }
        for (let oi = 0; oi < objectives.length; oi++) {
          const sortedIdx = [...frontIndices].sort(
            (a, b) => objectiveValues[oi][a] - objectiveValues[oi][b]
          );
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
        if (ra !== rb) return ra - rb;
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
        if (this._paretoArchive.length > 200) this._paretoArchive.shift();
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
          if (currentSize > target * 1.2) eps = Math.min(maxE, eps + adjust);
          else if (currentSize < target * 0.8) eps = Math.max(minE, eps - adjust);
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
            if (v < min) min = v;
            if (v > max) max = v;
          }
          ranges[obj.key] = { min, max };
        }
        const toRemove = [];
        for (const obj of objsList) {
          if (protect.has(obj.key)) continue;
          const objRange = ranges[obj.key];
          const span = objRange.max - objRange.min;
          if (span < rangeEps) {
            const count = (this._objectiveStale.get(obj.key) || 0) + 1;
            this._objectiveStale.set(obj.key, count);
            if (count >= window2) toRemove.push(obj.key);
          } else {
            this._objectiveStale.set(obj.key, 0);
          }
        }
        if (toRemove.length && this.options.multiObjective?.objectives) {
          this.options.multiObjective.objectives = this.options.multiObjective.objectives.filter(
            (obj) => !toRemove.includes(obj.key)
          );
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
          opts.excessCoeff = Math.min(
            maxC,
            Math.max(minC, opts.excessCoeff * factor)
          );
          opts.disjointCoeff = Math.min(
            maxC,
            Math.max(minC, opts.disjointCoeff * factor)
          );
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
    const fittest = Network3.fromJSON(this.population[0].toJSON());
    fittest.score = this.population[0].score;
    this._computeDiversityStats();
    try {
      const currentObjKeys = this._getObjectives().map(
        (obj) => obj.key
      );
      const dyn = this.options.multiObjective?.dynamic;
      if (this.options.multiObjective?.enabled) {
        if (dyn?.enabled) {
          const addC = dyn.addComplexityAt ?? Infinity;
          const addE = dyn.addEntropyAt ?? Infinity;
          if (this.generation + 1 >= addC && !currentObjKeys.includes("complexity")) {
            this.registerObjective(
              "complexity",
              "min",
              (genome) => genome.connections.length
            );
            this._pendingObjectiveAdds.push("complexity");
          }
          if (this.generation + 1 >= addE && !currentObjKeys.includes("entropy")) {
            this.registerObjective(
              "entropy",
              "max",
              (genome) => this._structuralEntropy(genome)
            );
            this._pendingObjectiveAdds.push("entropy");
          }
          if (currentObjKeys.includes("entropy") && dyn.dropEntropyOnStagnation != null) {
            const stagnGen = dyn.dropEntropyOnStagnation;
            if (this.generation >= stagnGen && !this._entropyDropped) {
              if (this.options.multiObjective?.objectives) {
                this.options.multiObjective.objectives = this.options.multiObjective.objectives.filter(
                  (obj) => obj.key !== "entropy"
                );
                this._objectivesList = void 0;
                this._pendingObjectiveRemoves.push("entropy");
                this._entropyDropped = this.generation;
              }
            }
          } else if (!currentObjKeys.includes("entropy") && this._entropyDropped && dyn.readdEntropyAfter != null) {
            if (this.generation - this._entropyDropped >= dyn.readdEntropyAfter) {
              this.registerObjective(
                "entropy",
                "max",
                (genome) => this._structuralEntropy(genome)
              );
              this._pendingObjectiveAdds.push("entropy");
              this._entropyDropped = void 0;
            }
          }
        } else if (this.options.multiObjective.autoEntropy) {
          const addAt = 3;
          if (this.generation >= addAt && !currentObjKeys.includes("entropy")) {
            this.registerObjective(
              "entropy",
              "max",
              (genome) => this._structuralEntropy(genome)
            );
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
          const varV = vals.reduce(
            (a, b) => a + (b - mean) * (b - mean),
            0
          ) / (vals.length || 1);
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
    const elitismCount = Math.max(
      0,
      Math.min(this.options.elitism || 0, this.population.length)
    );
    for (let i = 0; i < elitismCount; i++) {
      const elite = this.population[i];
      if (elite) newPopulation.push(elite);
    }
    const desiredPop = Math.max(0, this.options.popsize || 0);
    const remainingSlotsAfterElites = Math.max(
      0,
      desiredPop - newPopulation.length
    );
    const provenanceCount = Math.max(
      0,
      Math.min(this.options.provenance || 0, remainingSlotsAfterElites)
    );
    for (let i = 0; i < provenanceCount; i++) {
      if (this.options.network) {
        newPopulation.push(Network3.fromJSON(this.options.network.toJSON()));
      } else {
        newPopulation.push(
          new Network3(this.input, this.output, {
            minHidden: this.options.minHidden
          })
        );
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
          const base = species.members.reduce(
            (a, member) => a + (member.score || 0),
            0
          );
          const age = this.generation - species.lastImproved;
          if (age <= youngT) return base * youngM;
          if (age >= oldT) return base * oldM;
          return base;
        });
        const totalAdj = speciesAdjusted.reduce((a, b) => a + b, 0) || 1;
        const minOff = this.options.speciesAllocation?.minOffspring ?? 1;
        const rawShares = this._species.map(
          (_, idx) => speciesAdjusted[idx] / totalAdj * remaining
        );
        const offspringAlloc = rawShares.map(
          (s) => Math.floor(s)
        );
        for (let i = 0; i < offspringAlloc.length; i++)
          if (offspringAlloc[i] < minOff && remaining >= this._species.length * minOff)
            offspringAlloc[i] = minOff;
        let allocated = offspringAlloc.reduce((a, b) => a + b, 0);
        let slotsLeft = remaining - allocated;
        const remainders = rawShares.map((s, i) => ({
          i,
          frac: s - Math.floor(s)
        }));
        remainders.sort((a, b) => b.frac - a.frac);
        for (const remainderEntry of remainders) {
          if (slotsLeft <= 0) break;
          offspringAlloc[remainderEntry.i]++;
          slotsLeft--;
        }
        if (slotsLeft < 0) {
          const order = offspringAlloc.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v);
          for (const orderEntry of order) {
            if (slotsLeft === 0) break;
            if (offspringAlloc[orderEntry.i] > minOff) {
              offspringAlloc[orderEntry.i]--;
              slotsLeft++;
            }
          }
        }
        this._lastOffspringAlloc = this._species.map(
          (species, i) => ({
            id: species.id,
            alloc: offspringAlloc[i] || 0
          })
        );
        this._prevInbreedingCount = this._lastInbreedingCount;
        this._lastInbreedingCount = 0;
        offspringAlloc.forEach((count, idx) => {
          if (count <= 0) return;
          const species = this._species[idx];
          this._sortSpeciesMembers(species);
          const survivors = species.members.slice(
            0,
            Math.max(
              1,
              Math.floor(
                species.members.length * (this.options.survivalThreshold || 0.5)
              )
            )
          );
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
              const otherParents = otherSpecies.members.slice(
                0,
                Math.max(
                  1,
                  Math.floor(
                    otherSpecies.members.length * (this.options.survivalThreshold || 0.5)
                  )
                )
              );
              parentB = otherParents[Math.floor(this._getRNG()() * otherParents.length)];
            } else {
              parentB = survivors[Math.floor(this._getRNG()() * survivors.length)];
            }
            const child = Network3.crossOver(
              parentA,
              parentB,
              this.options.equal || false
            );
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
      for (let i = 0; i < toBreed; i++) newPopulation.push(this.getOffspring());
      this._suppressTournamentError = false;
    }
    for (const genome of newPopulation) {
      if (!genome) continue;
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
      if (genome._compatCache) delete genome._compatCache;
    });
    this.population.forEach((genome) => genome.score = void 0);
    this.generation++;
    if (this.options.speciation) this._updateSpeciesStagnation();
    if ((this.options.globalStagnationGenerations || 0) > 0 && this.generation - this._lastGlobalImproveGeneration > (this.options.globalStagnationGenerations || 0)) {
      const replaceFraction = 0.2;
      const startIdx = Math.max(
        this.options.elitism || 0,
        Math.floor(this.population.length * (1 - replaceFraction))
      );
      for (let i = startIdx; i < this.population.length; i++) {
        const fresh = new Network3(this.input, this.output, {
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
            const outputNodes = fresh.nodes.filter(
              (n) => n.type === "output"
            );
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
        this.options.reenableProb = Math.min(
          0.9,
          Math.max(0.05, this.options.reenableProb - delta * 0.1)
        );
      }
    }
    try {
      (init_neat_adaptive(), __toCommonJS(neat_adaptive_exports)).applyOperatorAdaptation.call(this);
    } catch {
    }
    const endTime = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
    this._lastEvolveDuration = endTime - startTime;
    try {
      if (!this._speciesHistory) this._speciesHistory = [];
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
    "src/neat/neat.evolve.ts"() {
      "use strict";
      init_network();
      init_neat_multiobjective();
    }
  });

  // src/neat/neat.evaluate.ts
  async function evaluate() {
    const options = this.options || {};
    if (options.fitnessPopulation) {
      if (options.clear)
        this.population.forEach((g) => g.clear && g.clear());
      await this.fitness(this.population);
    } else {
      for (const genome of this.population) {
        if (options.clear && genome.clear) genome.clear();
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
          const sortedRow = distanceMatrix[i].slice().sort((a, b) => a - b);
          const neighbours = sortedRow.slice(1, kNeighbors + 1);
          const novelty = neighbours.length ? neighbours.reduce((a, b) => a + b, 0) / neighbours.length : 0;
          this.population[i]._novelty = novelty;
          if (typeof this.population[i].score === "number") {
            this.population[i].score = (1 - blendFactor) * this.population[i].score + blendFactor * novelty;
          }
          if (!this._noveltyArchive) this._noveltyArchive = [];
          const archiveAddThreshold = noveltyOptions.archiveAddThreshold ?? Infinity;
          if (noveltyOptions.archiveAddThreshold === 0 || novelty > archiveAddThreshold) {
            if (this._noveltyArchive.length < 200)
              this._noveltyArchive.push({ desc: descriptors[i], novelty });
          }
        }
      }
    } catch {
    }
    if (!this._diversityStats) this._diversityStats = {};
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
            threshold = Math.max(
              entropyCompatOptions.minThreshold ?? 0.5,
              threshold * (1 - adjustRate)
            );
          else if (meanEntropy > targetEntropy + deadband)
            threshold = Math.min(
              entropyCompatOptions.maxThreshold ?? 10,
              threshold * (1 + adjustRate)
            );
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
        const connectionSizes = this.population.map(
          (g) => g.connections.length
        );
        const meanSize = connectionSizes.reduce((a, b) => a + b, 0) / (connectionSizes.length || 1);
        const connVar = connectionSizes.reduce(
          (a, b) => a + (b - meanSize) * (b - meanSize),
          0
        ) / (connectionSizes.length || 1);
        const adjustRate = autoDistanceCoeffOptions.adjustRate ?? 0.05;
        const minCoeff = autoDistanceCoeffOptions.minCoeff ?? 0.05;
        const maxCoeff = autoDistanceCoeffOptions.maxCoeff ?? 8;
        if (!this._lastConnVar) this._lastConnVar = connVar;
        if (connVar < this._lastConnVar * 0.95) {
          this.options.excessCoeff = Math.min(
            maxCoeff,
            this.options.excessCoeff * (1 + adjustRate)
          );
          this.options.disjointCoeff = Math.min(
            maxCoeff,
            this.options.disjointCoeff * (1 + adjustRate)
          );
        } else if (connVar > this._lastConnVar * 1.05) {
          this.options.excessCoeff = Math.max(
            minCoeff,
            this.options.excessCoeff * (1 - adjustRate)
          );
          this.options.disjointCoeff = Math.max(
            minCoeff,
            this.options.disjointCoeff * (1 - adjustRate)
          );
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
            this.registerObjective(
              "entropy",
              "max",
              (g) => this._structuralEntropy(g)
            );
            this._pendingObjectiveAdds.push("entropy");
            this._objectivesList = void 0;
          }
        }
      }
    } catch {
    }
  }
  var init_neat_evaluate = __esm({
    "src/neat/neat.evaluate.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.helpers.ts
  function spawnFromParent(parentGenome, mutateCount = 1) {
    const clone = parentGenome.clone ? parentGenome.clone() : (init_network(), __toCommonJS(network_exports)).default.fromJSON(
      parentGenome.toJSON()
    );
    clone.score = void 0;
    clone._reenableProb = this.options.reenableProb;
    clone._id = this._nextGenomeId++;
    clone._parents = [parentGenome._id];
    clone._depth = (parentGenome._depth ?? 0) + 1;
    this.ensureMinHiddenNodes(clone);
    this.ensureNoDeadEnds(clone);
    for (let mutationIndex = 0; mutationIndex < mutateCount; mutationIndex++) {
      try {
        let selectedMutationMethod = this.selectMutationMethod(
          clone,
          false
        );
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
        const parentDepths = genome._parents.map(
          (pid) => this.population.find((g) => g._id === pid)
        ).filter(Boolean).map((g) => g._depth ?? 0);
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
        const genomeCopy = seedNetwork ? Network3.fromJSON(seedNetwork.toJSON()) : new Network3(this.input, this.output, {
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
    "src/neat/neat.helpers.ts"() {
      "use strict";
      init_network();
    }
  });

  // src/neat/neat.objectives.ts
  function _getObjectives() {
    if (this._objectivesList) return this._objectivesList;
    const objectivesList = [];
    if (!this._suppressFitnessObjective) {
      objectivesList.push({
        key: "fitness",
        direction: "max",
        /**
         * Default accessor extracts the `score` property from a genome.
         *
         * @example
         * ```ts
         * // genome.score is used as the fitness metric by default
         * const value = defaultAccessor(genome);
         * ```
         */
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
    if (!multiObjectiveOptions.objectives) multiObjectiveOptions.objectives = [];
    multiObjectiveOptions.objectives = multiObjectiveOptions.objectives.filter(
      (existingObjective) => existingObjective.key !== key
    );
    multiObjectiveOptions.objectives.push({ key, direction, accessor });
    this._objectivesList = void 0;
  }
  function clearObjectives() {
    if (this.options.multiObjective?.objectives)
      this.options.multiObjective.objectives = [];
    this._objectivesList = void 0;
  }
  var init_neat_objectives = __esm({
    "src/neat/neat.objectives.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.diversity.ts
  function structuralEntropy2(graph) {
    const outDegrees = graph.nodes.map(
      (node) => (
        // each node exposes connections.out array in current architecture
        node.connections.out.length
      )
    );
    const totalOut = outDegrees.reduce((acc, v) => acc + v, 0) || 1;
    const probabilities = outDegrees.map((d) => d / totalOut).filter((p) => p > 0);
    let entropy = 0;
    for (const p of probabilities) {
      entropy -= p * Math.log(p);
    }
    return entropy;
  }
  function arrayMean(values) {
    if (!values.length) return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }
  function arrayVariance(values) {
    if (!values.length) return 0;
    const m = arrayMean(values);
    return arrayMean(values.map((v) => (v - m) * (v - m)));
  }
  function computeDiversityStats2(population, compatibilityComputer) {
    if (!population.length) return void 0;
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
        compatSum += compatibilityComputer._compatibilityDistance(
          population[i],
          population[j]
        );
        compatPairCount++;
      }
    }
    const meanCompat = compatPairCount ? compatSum / compatPairCount : 0;
    const graphletEntropy = arrayMean(
      population.map((g) => structuralEntropy2(g))
    );
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
    "src/neat/neat.diversity.ts"() {
      "use strict";
      init_network();
    }
  });

  // src/neat/neat.compat.ts
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
    if (cacheMap.has(key)) return cacheMap.get(key);
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
        if (innovA > maxInnovB) excess++;
        else disjoint++;
        indexA++;
      } else {
        if (innovB > maxInnovA) excess++;
        else disjoint++;
        indexB++;
      }
    }
    if (indexA < aList.length) excess += aList.length - indexA;
    if (indexB < bList.length) excess += bList.length - indexB;
    const N = Math.max(1, Math.max(aList.length, bList.length));
    const avgWeightDiff = matchingCount ? weightDifferenceSum / matchingCount : 0;
    const opts = this.options;
    const dist = opts.excessCoeff * excess / N + opts.disjointCoeff * disjoint / N + opts.weightDiffCoeff * avgWeightDiff;
    cacheMap.set(key, dist);
    return dist;
  }
  var init_neat_compat = __esm({
    "src/neat/neat.compat.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.speciation.ts
  function _speciate() {
    this._prevSpeciesMembers.clear();
    for (const species of this._species) {
      const prevMemberSet = /* @__PURE__ */ new Set();
      for (const member of species.members)
        prevMemberSet.add(member._id);
      this._prevSpeciesMembers.set(species.id, prevMemberSet);
    }
    this._species.forEach((species) => species.members = []);
    for (const genome of this.population) {
      let assignedToExisting = false;
      for (const species of this._species) {
        const compatDist = this._compatibilityDistance(
          genome,
          species.representative
        );
        if (compatDist < (this.options.compatibilityThreshold || 3)) {
          species.members.push(genome);
          assignedToExisting = true;
          break;
        }
      }
      if (!assignedToExisting) {
        const speciesId = this._nextSpeciesId++;
        this._species.push({
          id: speciesId,
          members: [genome],
          representative: genome,
          lastImproved: this.generation,
          bestScore: genome.score || -Infinity
        });
        this._speciesCreated.set(speciesId, this.generation);
      }
    }
    this._species = this._species.filter(
      (species) => species.members.length > 0
    );
    this._species.forEach((species) => {
      species.representative = species.members[0];
    });
    const ageProtection = this.options.speciesAgeProtection || {
      grace: 3,
      oldPenalty: 0.5
    };
    for (const species of this._species) {
      const createdGen = this._speciesCreated.get(species.id) ?? this.generation;
      const speciesAge = this.generation - createdGen;
      if (speciesAge >= (ageProtection.grace ?? 3) * 10) {
        const penalty = ageProtection.oldPenalty ?? 0.5;
        if (penalty < 1)
          species.members.forEach((member) => {
            if (typeof member.score === "number") member.score *= penalty;
          });
      }
    }
    if (this.options.speciation && (this.options.targetSpecies || 0) > 0) {
      const targetSpeciesCount = this.options.targetSpecies;
      const observedSpeciesCount = this._species.length;
      const adjustConfig = this.options.compatAdjust;
      const smoothingWindow = Math.max(1, adjustConfig.smoothingWindow || 1);
      const alpha = 2 / (smoothingWindow + 1);
      this._compatSpeciesEMA = this._compatSpeciesEMA === void 0 ? observedSpeciesCount : this._compatSpeciesEMA + alpha * (observedSpeciesCount - this._compatSpeciesEMA);
      const smoothedSpecies = this._compatSpeciesEMA;
      const speciesError = targetSpeciesCount - smoothedSpecies;
      this._compatIntegral = this._compatIntegral * (adjustConfig.decay || 0.95) + speciesError;
      const delta = (adjustConfig.kp || 0) * speciesError + (adjustConfig.ki || 0) * this._compatIntegral;
      let newThreshold = (this.options.compatibilityThreshold || 3) - delta;
      const minThreshold = adjustConfig.minThreshold || 0.5;
      const maxThreshold = adjustConfig.maxThreshold || 10;
      if (newThreshold < minThreshold) {
        newThreshold = minThreshold;
        this._compatIntegral = 0;
      }
      if (newThreshold > maxThreshold) {
        newThreshold = maxThreshold;
        this._compatIntegral = 0;
      }
      this.options.compatibilityThreshold = newThreshold;
    }
    if (this.options.autoCompatTuning?.enabled) {
      const autoTarget = this.options.autoCompatTuning.target ?? this.options.targetSpecies ?? Math.max(2, Math.round(Math.sqrt(this.population.length)));
      const observedForTuning = this._species.length || 1;
      const tuningError = autoTarget - observedForTuning;
      const adjustRate = this.options.autoCompatTuning.adjustRate ?? 0.01;
      const minCoeff = this.options.autoCompatTuning.minCoeff ?? 0.1;
      const maxCoeff = this.options.autoCompatTuning.maxCoeff ?? 5;
      const factor = 1 - adjustRate * Math.sign(tuningError);
      let effectiveFactor = factor;
      if (tuningError === 0) {
        effectiveFactor = 1 + (this._getRNG()() - 0.5) * adjustRate * 0.5;
      }
      this.options.excessCoeff = Math.min(
        maxCoeff,
        Math.max(minCoeff, this.options.excessCoeff * effectiveFactor)
      );
      this.options.disjointCoeff = Math.min(
        maxCoeff,
        Math.max(minCoeff, this.options.disjointCoeff * effectiveFactor)
      );
    }
    if (this.options.speciesAllocation?.extendedHistory) {
      const stats = this._species.map((species) => {
        const sizes = species.members.map((member) => ({
          nodes: member.nodes.length,
          conns: member.connections.length,
          score: member.score || 0,
          nov: member._novelty || 0,
          ent: this._structuralEntropy(member)
        }));
        const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
        let compatSum = 0;
        let compatCount = 0;
        for (let i = 0; i < species.members.length && i < 10; i++)
          for (let j = i + 1; j < species.members.length && j < 10; j++) {
            compatSum += this._compatibilityDistance(
              species.members[i],
              species.members[j]
            );
            compatCount++;
          }
        const meanCompat = compatCount ? compatSum / compatCount : 0;
        const last = this._speciesLastStats.get(species.id);
        const meanNodes = avg(sizes.map((s) => s.nodes));
        const meanConns = avg(sizes.map((s) => s.conns));
        const deltaMeanNodes = last ? meanNodes - last.meanNodes : 0;
        const deltaMeanConns = last ? meanConns - last.meanConns : 0;
        const deltaBestScore = last ? species.bestScore - last.best : 0;
        const createdGen = this._speciesCreated.get(species.id) ?? this.generation;
        const speciesAge = this.generation - createdGen;
        let turnoverRate = 0;
        const prevSet = this._prevSpeciesMembers.get(species.id);
        if (prevSet && species.members.length) {
          let newCount = 0;
          for (const member of species.members)
            if (!prevSet.has(member._id)) newCount++;
          turnoverRate = newCount / species.members.length;
        }
        const varCalc = (arr) => {
          if (!arr.length) return 0;
          const mean = avg(arr);
          return avg(arr.map((v) => (v - mean) * (v - mean)));
        };
        const varNodes = varCalc(sizes.map((s) => s.nodes));
        const varConns = varCalc(sizes.map((s) => s.conns));
        let innovSum = 0;
        let innovCount = 0;
        let maxInnov = -Infinity;
        let minInnov = Infinity;
        let enabled = 0;
        let disabled = 0;
        for (const member of species.members)
          for (const conn of member.connections) {
            const innov = conn.innovation ?? this._fallbackInnov(conn);
            innovSum += innov;
            innovCount++;
            if (innov > maxInnov) maxInnov = innov;
            if (innov < minInnov) minInnov = innov;
            if (conn.enabled === false) disabled++;
            else enabled++;
          }
        const meanInnovation = innovCount ? innovSum / innovCount : 0;
        const innovationRange = isFinite(maxInnov) && isFinite(minInnov) && maxInnov > minInnov ? maxInnov - minInnov : 0;
        const enabledRatio = enabled + disabled > 0 ? enabled / (enabled + disabled) : 0;
        return {
          id: species.id,
          size: species.members.length,
          best: species.bestScore,
          lastImproved: species.lastImproved,
          age: speciesAge,
          meanNodes,
          meanConns,
          meanScore: avg(sizes.map((s) => s.score)),
          meanNovelty: avg(sizes.map((s) => s.nov)),
          meanCompat,
          meanEntropy: avg(sizes.map((s) => s.ent)),
          varNodes,
          varConns,
          deltaMeanNodes,
          deltaMeanConns,
          deltaBestScore,
          turnoverRate,
          meanInnovation,
          innovationRange,
          enabledRatio
        };
      });
      for (const st of stats)
        this._speciesLastStats.set(st.id, {
          meanNodes: st.meanNodes,
          meanConns: st.meanConns,
          best: st.best
        });
      this._speciesHistory.push({ generation: this.generation, stats });
    } else {
      this._speciesHistory.push({
        generation: this.generation,
        stats: this._species.map((species) => ({
          id: species.id,
          size: species.members.length,
          best: species.bestScore,
          lastImproved: species.lastImproved
        }))
      });
    }
    if (this._speciesHistory.length > 200) this._speciesHistory.shift();
  }
  function _applyFitnessSharing() {
    const sharingSigma = this.options.sharingSigma || 0;
    if (sharingSigma > 0) {
      this._species.forEach((species) => {
        const members = species.members;
        for (let i = 0; i < members.length; i++) {
          const memberI = members[i];
          if (typeof memberI.score !== "number") continue;
          let shareSum = 0;
          for (let j = 0; j < members.length; j++) {
            const memberJ = members[j];
            const dist = i === j ? 0 : this._compatibilityDistance(memberI, memberJ);
            if (dist < sharingSigma) {
              const ratio = dist / sharingSigma;
              shareSum += 1 - ratio * ratio;
            }
          }
          if (shareSum <= 0) shareSum = 1;
          memberI.score = memberI.score / shareSum;
        }
      });
    } else {
      this._species.forEach((species) => {
        const size = species.members.length;
        species.members.forEach((member) => {
          if (typeof member.score === "number")
            member.score = member.score / size;
        });
      });
    }
  }
  function _sortSpeciesMembers(sp) {
    sp.members.sort((a, b) => (b.score || 0) - (a.score || 0));
  }
  function _updateSpeciesStagnation() {
    const stagnationWindow = this.options.stagnationGenerations || 15;
    this._species.forEach((species) => {
      this._sortSpeciesMembers(species);
      const top = species.members[0];
      if ((top.score || -Infinity) > species.bestScore) {
        species.bestScore = top.score || -Infinity;
        species.lastImproved = this.generation;
      }
    });
    const survivors = this._species.filter(
      (species) => this.generation - species.lastImproved <= stagnationWindow
    );
    if (survivors.length) this._species = survivors;
  }
  var init_neat_speciation = __esm({
    "src/neat/neat.speciation.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.species.ts
  function getSpeciesStats() {
    const speciesArray = this._species;
    return speciesArray.map((species) => ({
      id: species.id,
      size: species.members.length,
      bestScore: species.bestScore,
      lastImproved: species.lastImproved
    }));
  }
  function getSpeciesHistory() {
    const speciesHistory = this._speciesHistory;
    if (this.options?.speciesAllocation?.extendedHistory) {
      for (const generationEntry of speciesHistory) {
        for (const speciesStat of generationEntry.stats) {
          if ("innovationRange" in speciesStat && "enabledRatio" in speciesStat)
            continue;
          const speciesObj = this._species.find(
            (s) => s.id === speciesStat.id
          );
          if (speciesObj && speciesObj.members && speciesObj.members.length) {
            let maxInnovation = -Infinity;
            let minInnovation = Infinity;
            let enabledCount = 0;
            let disabledCount = 0;
            for (const member of speciesObj.members) {
              for (const connection of member.connections) {
                const innovationId = connection.innovation ?? this._fallbackInnov?.(connection) ?? 0;
                if (innovationId > maxInnovation) maxInnovation = innovationId;
                if (innovationId < minInnovation) minInnovation = innovationId;
                if (connection.enabled === false) disabledCount++;
                else enabledCount++;
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
    "src/neat/neat.species.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.telemetry.exports.ts
  function exportTelemetryJSONL() {
    return this._telemetry.map((entry) => JSON.stringify(entry)).join("\n");
  }
  function exportTelemetryCSV(maxEntries = 500) {
    const recentTelemetry = Array.isArray(this._telemetry) ? this._telemetry.slice(-maxEntries) : [];
    if (!recentTelemetry.length) return "";
    const headerInfo = collectTelemetryHeaderInfo(recentTelemetry);
    const headers = buildTelemetryHeaders(headerInfo);
    const csvLines = [headers.join(",")];
    for (const telemetryEntry of recentTelemetry) {
      csvLines.push(serializeTelemetryEntry(telemetryEntry, headers));
    }
    return csvLines.join("\n");
  }
  function collectTelemetryHeaderInfo(entries) {
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
      if (Array.isArray(entry.fronts)) baseKeys.add(HEADER_FRONTS);
      if (entry.complexity)
        Object.keys(entry.complexity).forEach((k) => complexityKeys.add(k));
      if (entry.perf) Object.keys(entry.perf).forEach((k) => perfKeys.add(k));
      if (entry.lineage)
        Object.keys(entry.lineage).forEach((k) => lineageKeys.add(k));
      if (entry.diversity) {
        if ("lineageMeanDepth" in entry.diversity)
          diversityLineageKeys.add("lineageMeanDepth");
        if ("lineageMeanPairDist" in entry.diversity)
          diversityLineageKeys.add("lineageMeanPairDist");
      }
      if ("rng" in entry) baseKeys.add("rng");
      if (Array.isArray(entry.ops) && entry.ops.length) includeOps = true;
      if (Array.isArray(entry.objectives)) includeObjectives = true;
      if (entry.objAges) includeObjAges = true;
      if (Array.isArray(entry.speciesAlloc)) includeSpeciesAlloc = true;
      if (Array.isArray(entry.objEvents) && entry.objEvents.length)
        includeObjEvents = true;
      if (entry.objImportance) includeObjImportance = true;
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
  }
  function buildTelemetryHeaders(info) {
    const headers = [
      ...info.baseKeys,
      ...[...info.complexityKeys].map((k) => `${COMPLEXITY_PREFIX}${k}`),
      ...[...info.perfKeys].map((k) => `${PERF_PREFIX}${k}`),
      ...[...info.lineageKeys].map((k) => `${LINEAGE_PREFIX}${k}`),
      ...[...info.diversityLineageKeys].map((k) => `${DIVERSITY_PREFIX}${k}`)
    ];
    if (info.includeOps) headers.push(HEADER_OPS);
    if (info.includeObjectives) headers.push(HEADER_OBJECTIVES);
    if (info.includeObjAges) headers.push(HEADER_OBJ_AGES);
    if (info.includeSpeciesAlloc) headers.push(HEADER_SPECIES_ALLOC);
    if (info.includeObjEvents) headers.push(HEADER_OBJ_EVENTS);
    if (info.includeObjImportance) headers.push(HEADER_OBJ_IMPORTANCE);
    return headers;
  }
  function serializeTelemetryEntry(entry, headers) {
    const row = [];
    for (const header of headers) {
      switch (true) {
        // Grouped complexity metrics
        case header.startsWith(COMPLEXITY_PREFIX): {
          const key = header.slice(COMPLEXITY_PREFIX.length);
          row.push(
            entry.complexity && key in entry.complexity ? JSON.stringify(entry.complexity[key]) : ""
          );
          break;
        }
        // Grouped performance metrics
        case header.startsWith(PERF_PREFIX): {
          const key = header.slice(PERF_PREFIX.length);
          row.push(
            entry.perf && key in entry.perf ? JSON.stringify(entry.perf[key]) : ""
          );
          break;
        }
        // Grouped lineage metrics
        case header.startsWith(LINEAGE_PREFIX): {
          const key = header.slice(LINEAGE_PREFIX.length);
          row.push(
            entry.lineage && key in entry.lineage ? JSON.stringify(entry.lineage[key]) : ""
          );
          break;
        }
        // Grouped diversity metrics
        case header.startsWith(DIVERSITY_PREFIX): {
          const key = header.slice(DIVERSITY_PREFIX.length);
          row.push(
            entry.diversity && key in entry.diversity ? JSON.stringify(entry.diversity[key]) : ""
          );
          break;
        }
        // Array-like and optional multi-value columns
        case header === HEADER_FRONTS: {
          row.push(
            Array.isArray(entry.fronts) ? JSON.stringify(entry.fronts) : ""
          );
          break;
        }
        case header === HEADER_OPS: {
          row.push(Array.isArray(entry.ops) ? JSON.stringify(entry.ops) : "");
          break;
        }
        case header === HEADER_OBJECTIVES: {
          row.push(
            Array.isArray(entry.objectives) ? JSON.stringify(entry.objectives) : ""
          );
          break;
        }
        case header === HEADER_OBJ_AGES: {
          row.push(entry.objAges ? JSON.stringify(entry.objAges) : "");
          break;
        }
        case header === HEADER_SPECIES_ALLOC: {
          row.push(
            Array.isArray(entry.speciesAlloc) ? JSON.stringify(entry.speciesAlloc) : ""
          );
          break;
        }
        case header === HEADER_OBJ_EVENTS: {
          row.push(
            Array.isArray(entry.objEvents) ? JSON.stringify(entry.objEvents) : ""
          );
          break;
        }
        case header === HEADER_OBJ_IMPORTANCE: {
          row.push(
            entry.objImportance ? JSON.stringify(entry.objImportance) : ""
          );
          break;
        }
        // Default: treat as top-level column
        default: {
          row.push(JSON.stringify(entry[header]));
          break;
        }
      }
    }
    return row.join(",");
  }
  function exportSpeciesHistoryCSV(maxEntries = 200) {
    if (!Array.isArray(this._speciesHistory)) this._speciesHistory = [];
    if (!this._speciesHistory.length && Array.isArray(this._species) && this._species.length) {
      const stats = this._species.map((sp) => ({
        /** Unique identifier for the species (or -1 when missing). */
        id: sp.id ?? -1,
        /** Current size (number of members) in the species. */
        size: Array.isArray(sp.members) ? sp.members.length : 0,
        /** Best score observed in the species (fallback 0). */
        best: sp.bestScore ?? 0,
        /** Generation index when the species last improved (fallback 0). */
        lastImproved: sp.lastImproved ?? 0
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
  }
  function buildSpeciesHistoryCsv(recentHistory, headers) {
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
  }
  var COMPLEXITY_PREFIX, PERF_PREFIX, LINEAGE_PREFIX, DIVERSITY_PREFIX, HEADER_FRONTS, HEADER_OPS, HEADER_OBJECTIVES, HEADER_OBJ_AGES, HEADER_SPECIES_ALLOC, HEADER_OBJ_EVENTS, HEADER_OBJ_IMPORTANCE, HEADER_GENERATION;
  var init_neat_telemetry_exports = __esm({
    "src/neat/neat.telemetry.exports.ts"() {
      "use strict";
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
      HEADER_GENERATION = "generation";
    }
  });

  // src/neat/neat.selection.ts
  function sort() {
    this.population.sort(
      (a, b) => (b.score ?? 0) - (a.score ?? 0)
    );
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
        const selectedIndex = Math.floor(
          Math.pow(getRngFactory()(), selectionOptions.power || 1) * population.length
        );
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
          if (threshold < cumulative) return individual;
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
          tournamentParticipants.push(
            population[Math.floor(getRngFactory()() * population.length)]
          );
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
    const totalScore = population.reduce(
      (sum, genome) => sum + (genome.score ?? 0),
      0
    );
    return totalScore / population.length;
  }
  var init_neat_selection = __esm({
    "src/neat/neat.selection.ts"() {
      "use strict";
    }
  });

  // src/neat/neat.export.ts
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
    const Network6 = (init_network(), __toCommonJS(network_exports)).default;
    this.population = populationJSON.map(
      (serializedGenome) => Network6.fromJSON(serializedGenome)
    );
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
      nodeSplitInnovations: Array.from(
        this._nodeSplitInnovations.entries()
      ),
      connInnovations: Array.from(this._connInnovations.entries()),
      nextGlobalInnovation: this._nextGlobalInnovation
    };
  }
  function fromJSONImpl2(neatJSON, fitnessFunction) {
    const NeatClass = this;
    const neatInstance = new NeatClass(
      neatJSON.input,
      neatJSON.output,
      fitnessFunction,
      neatJSON.options || {}
    );
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
    "src/neat/neat.export.ts"() {
      "use strict";
    }
  });

  // src/neat.ts
  var neat_exports = {};
  __export(neat_exports, {
    default: () => Neat
  });
  var Neat;
  var init_neat = __esm({
    "src/neat.ts"() {
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
        // Deterministic RNG state (lazy init)
        /**
         * Internal numeric state for the deterministic xorshift RNG when no user RNG
         * is provided. Stored as a 32-bit unsigned integer.
         */
        _rngState;
        /**
         * Cached RNG function; created lazily and seeded from `_rngState` when used.
         */
        _rng;
        // Internal bookkeeping and caches (kept permissive during staggered migration)
        /** Array of current species (internal representation). */
        _species = [];
        /** Operator statistics used by adaptive operator selection. */
        _operatorStats = /* @__PURE__ */ new Map();
        /** Map of node-split innovations used to reuse innovation ids for node splits. */
        _nodeSplitInnovations = /* @__PURE__ */ new Map();
        /** Map of connection innovations keyed by a string identifier. */
        _connInnovations = /* @__PURE__ */ new Map();
        /** Counter for issuing global innovation numbers when explicit numbers are used. */
        _nextGlobalInnovation = 1;
        /** Counter for assigning unique genome ids. */
        _nextGenomeId = 1;
        /** Whether lineage metadata should be recorded on genomes. */
        _lineageEnabled = false;
        /** Last observed count of inbreeding (used for detecting excessive cloning). */
        _lastInbreedingCount = 0;
        /** Previous inbreeding count snapshot. */
        _prevInbreedingCount = 0;
        /** Optional phase marker for multi-stage experiments. */
        _phase;
        /** Telemetry buffer storing diagnostic snapshots per generation. */
        _telemetry = [];
        /** Map of species id -> set of member genome ids from previous generation. */
        _prevSpeciesMembers = /* @__PURE__ */ new Map();
        /** Last recorded stats per species id. */
        _speciesLastStats = /* @__PURE__ */ new Map();
        /** Time-series history of species stats (for exports/telemetry). */
        _speciesHistory = [];
        /** Archive of Pareto front metadata for multi-objective tracking. */
        _paretoArchive = [];
        /** Archive storing Pareto objectives snapshots. */
        _paretoObjectivesArchive = [];
        /** Novelty archive used by novelty search (behavior representatives). */
        _noveltyArchive = [];
        /** Map tracking stale counts for objectives by key. */
        _objectiveStale = /* @__PURE__ */ new Map();
        /** Map tracking ages for objectives by key. */
        _objectiveAges = /* @__PURE__ */ new Map();
        /** Queue of recent objective activation/deactivation events for telemetry. */
        _objectiveEvents = [];
        /** Pending objective keys to add during safe phases. */
        _pendingObjectiveAdds = [];
        /** Pending objective keys to remove during safe phases. */
        _pendingObjectiveRemoves = [];
        /** Last allocated offspring set (used by adaptive allocators). */
        _lastOffspringAlloc;
        /** Adaptive prune level for complexity control (optional). */
        _adaptivePruneLevel;
        /** Duration of the last evaluation run (ms). */
        _lastEvalDuration;
        /** Duration of the last evolve run (ms). */
        _lastEvolveDuration;
        /** Cached diversity metrics (computed lazily). */
        _diversityStats;
        /** Cached list of registered objectives. */
        _objectivesList;
        /** Generation index where the last global improvement occurred. */
        _lastGlobalImproveGeneration = 0;
        /** Best score observed in the last generation (used for improvement detection). */
        _bestScoreLastGen;
        // Speciation controller state
        /** Map of speciesId -> creation generation for bookkeeping. */
        _speciesCreated = /* @__PURE__ */ new Map();
        /** Exponential moving average for compatibility threshold (adaptive speciation). */
        _compatSpeciesEMA;
        /** Integral accumulator used by adaptive compatibility controllers. */
        _compatIntegral = 0;
        /** Generation when epsilon compatibility was last adjusted. */
        _lastEpsilonAdjustGen = -Infinity;
        /** Generation when ancestor uniqueness adjustment was last applied. */
        _lastAncestorUniqAdjustGen = -Infinity;
        // Adaptive minimal criterion & complexity
        /** Adaptive minimal criterion threshold (optional). */
        _mcThreshold;
        // Lightweight RNG accessor used throughout migrated modules
        _getRNG() {
          if (!this._rng) {
            const optRng = this.options?.rng;
            if (typeof optRng === "function") this._rng = optRng;
            else {
              if (this._rngState === void 0) {
                let seed = (Date.now() ^ (this.population.length + 1) * 2654435761) >>> 0;
                if (seed === 0) seed = 439041101;
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
        // Delegate ensureMinHiddenNodes to migrated mutation helper for smaller class surface
        /**
         * Ensure a network has the minimum number of hidden nodes according to
         * configured policy. Delegates to migrated helper implementation.
         *
         * @param network Network instance to adjust.
         * @param multiplierOverride Optional multiplier to override configured policy.
         */
        ensureMinHiddenNodes(network, multiplierOverride) {
          return ensureMinHiddenNodes.call(this, network, multiplierOverride);
        }
        /**
         * Construct a new Neat instance.
         * Kept permissive during staged migration; accepts the same signature tests expect.
         *
         * @example
         * // Create a neat instance for 3 inputs and 1 output with default options
         * const neat = new Neat(3, 1, (net) => evaluateFitness(net));
         */
        constructor(input, output, fitness, options = {}) {
          this.input = input ?? 0;
          this.output = output ?? 0;
          this.fitness = fitness ?? ((n) => 0);
          this.options = options || {};
          const opts = this.options;
          if (opts.popsize === void 0) opts.popsize = 50;
          if (opts.elitism === void 0) opts.elitism = 0;
          if (opts.provenance === void 0) opts.provenance = 0;
          if (opts.mutationRate === void 0) opts.mutationRate = 0.7;
          if (opts.mutationAmount === void 0) opts.mutationAmount = 1;
          if (opts.fitnessPopulation === void 0) opts.fitnessPopulation = false;
          if (opts.clear === void 0) opts.clear = false;
          if (opts.equal === void 0) opts.equal = false;
          if (opts.compatibilityThreshold === void 0)
            opts.compatibilityThreshold = 3;
          if (opts.maxNodes === void 0) opts.maxNodes = Infinity;
          if (opts.maxConns === void 0) opts.maxConns = Infinity;
          if (opts.maxGates === void 0) opts.maxGates = Infinity;
          if (opts.excessCoeff === void 0) opts.excessCoeff = 1;
          if (opts.disjointCoeff === void 0) opts.disjointCoeff = 1;
          if (opts.weightDiffCoeff === void 0) opts.weightDiffCoeff = 0.5;
          if (opts.mutation === void 0)
            opts.mutation = mutation.ALL ? mutation.ALL.slice() : mutation.FFW ? [mutation.FFW] : [];
          if (opts.selection === void 0) {
            opts.selection = selection && selection.TOURNAMENT || selection?.TOURNAMENT || selection.FITNESS_PROPORTIONATE;
          }
          if (opts.crossover === void 0)
            opts.crossover = crossover ? crossover.SINGLE_POINT : void 0;
          if (opts.novelty === void 0) opts.novelty = { enabled: false };
          if (opts.diversityMetrics === void 0)
            opts.diversityMetrics = { enabled: true };
          if (opts.fastMode && opts.diversityMetrics) {
            if (opts.diversityMetrics.pairSample == null)
              opts.diversityMetrics.pairSample = 20;
            if (opts.diversityMetrics.graphletSample == null)
              opts.diversityMetrics.graphletSample = 30;
            if (opts.novelty?.enabled && opts.novelty.k == null) opts.novelty.k = 5;
          }
          this._noveltyArchive = [];
          if (opts.speciation === void 0) opts.speciation = false;
          if (opts.multiObjective && opts.multiObjective.enabled && !Array.isArray(opts.multiObjective.objectives))
            opts.multiObjective.objectives = [];
          this.population = this.population || [];
          try {
            if (this.options.network !== void 0)
              this.createPool(this.options.network);
            else if (this.options.popsize) this.createPool(null);
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
        /**
         * Evolves the population by selecting, mutating, and breeding genomes.
         * This method is delegated to `src/neat/neat.evolve.ts` during the migration.
         *
         * @example
         * // Run a single evolution step (async)
         * await neat.evolve();
         */
        async evolve() {
          return evolve.call(this);
        }
        async evaluate() {
          return evaluate.call(this);
        }
        /**
         * Create initial population pool. Delegates to helpers if present.
         */
        createPool(network) {
          try {
            if (createPool && typeof createPool === "function")
              return createPool.call(this, network);
          } catch {
          }
          this.population = [];
          const poolSize = this.options.popsize || 50;
          for (let idx = 0; idx < poolSize; idx++) {
            const genomeCopy = network ? Network3.fromJSON(network.toJSON()) : new Network3(this.input, this.output, {
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
        // RNG snapshot / restore helpers used by tests
        /**
         * Return the current opaque RNG numeric state used by the instance.
         * Useful for deterministic test replay and debugging.
         */
        snapshotRNGState() {
          return this._rngState;
        }
        /**
         * Restore a previously-snapshotted RNG state. This restores the internal
         * seed but does not re-create the RNG function until next use.
         *
         * @param state Opaque numeric RNG state produced by `snapshotRNGState()`.
         */
        restoreRNGState(state) {
          this._rngState = state;
          this._rng = void 0;
        }
        /**
         * Import an RNG state (alias for restore; kept for compatibility).
         * @param state Numeric RNG state.
         */
        importRNGState(state) {
          this._rngState = state;
          this._rng = void 0;
        }
        /**
         * Export the current RNG state for external persistence or tests.
         */
        exportRNGState() {
          return this._rngState;
        }
        /**
         * Generates an offspring by crossing over two parent networks.
         * Uses the crossover method described in the Instinct algorithm.
         * @returns A new network created from two parents.
         * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
         */
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
          const offspring = Network3.crossOver(
            parent1,
            parent2,
            this.options.equal || false
          );
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
        /** Emit a standardized warning when evolution loop finds no valid best genome (test hook). */
        _warnIfNoBestGenome() {
          try {
            console.warn(
              "Evolution completed without finding a valid best genome (no fitness improvements recorded)."
            );
          } catch {
          }
        }
        /**
         * Spawn a new genome derived from a single parent while preserving Neat bookkeeping.
         *
         * This helper performs a canonical "clone + slight mutation" workflow while
         * keeping `Neat`'s internal invariants intact. It is intended for callers that
         * want a child genome derived from a single parent but do not want to perform the
         * bookkeeping and registration steps manually. The function deliberately does NOT
         * add the returned child to `this.population` so callers are free to inspect or
         * further modify the child and then register it via `addGenome()` (or push it
         * directly if they understand the consequences).
         *
         * Behavior summary:
         * - Clone the provided `parent` (`parent.clone()` when available, else JSON round-trip).
         * - Clear fitness/score on the child and assign a fresh unique `_id`.
         * - If lineage tracking is enabled, set `(child as any)._parents = [parent._id]`
         *   and `(child as any)._depth = (parent._depth ?? 0) + 1`.
         * - Enforce structural invariants by calling `ensureMinHiddenNodes(child)` and
         *   `ensureNoDeadEnds(child)` so the child is valid for subsequent mutation/evaluation.
         * - Apply `mutateCount` mutations selected via `selectMutationMethod` and driven by
         *   the instance RNG (`_getRNG()`); mutation exceptions are caught and ignored to
         *   preserve best-effort behavior during population seeding/expansion.
         * - Invalidate per-genome caches with `_invalidateGenomeCaches(child)` before return.
         *
         * Important: the returned child is not registered in `Neat.population` — call
         * `addGenome(child, [parentId])` to insert it and keep telemetry/lineage consistent.
         *
         * @param parent - Source genome to derive from. Must be a `Network` instance.
         * @param mutateCount - Number of mutation operations to apply to the spawned child (default: 1).
         * @returns A new `Network` instance derived from `parent`. The child is unregistered.
         */
        spawnFromParent(parent, mutateCount = 1) {
          return spawnFromParent.call(this, parent, mutateCount);
        }
        /**
         * Register an externally-created genome into the `Neat` population.
         *
         * Use this method when code constructs or mutates a `Network` outside of the
         * usual reproduction pipeline and needs to insert it into `neat.population`
         * while preserving lineage, id assignment, and structural invariants. The
         * method performs best-effort safety actions and falls back to pushing the
         * genome even if invariant enforcement throws, which mirrors the forgiving
         * behavior used in dynamic population expansion.
         *
         * Behavior summary:
         * - Clears the genome's `score` and assigns `_id` using Neat's counter.
         * - When lineage is enabled, attaches the provided `parents` array (copied)
         *   and estimates `_depth` as `max(parent._depth) + 1` when parent ids are
         *   resolvable from the current population.
         * - Enforces structural invariants (`ensureMinHiddenNodes` and
         *   `ensureNoDeadEnds`) and invalidates caches via
         *   `_invalidateGenomeCaches(genome)`.
         * - Pushes the genome into `this.population`.
         *
         * Note: Because depth estimation requires parent objects to be discoverable
         * in `this.population`, callers that generate intermediate parent genomes
         * should register them via `addGenome` before relying on automatic depth
         * estimation for their children.
         *
         * @param genome - The external `Network` to add.
         * @param parents - Optional array of parent ids to record on the genome.
         */
        addGenome(genome, parents) {
          return addGenome.call(this, genome, parents);
        }
        /**
         * Selects a mutation method for a given genome based on constraints.
         * Ensures that the mutation respects the maximum nodes, connections, and gates.
         * @param genome - The genome to mutate.
         * @returns The selected mutation method or null if no valid method is available.
         */
        selectMutationMethod(genome, rawReturnForTest = true) {
          try {
            return selectMutationMethod.call(this, genome, rawReturnForTest);
          } catch {
            return null;
          }
        }
        /** Delegate ensureNoDeadEnds to mutation module (added for backward compat). */
        ensureNoDeadEnds(network) {
          try {
            return ensureNoDeadEnds.call(this, network);
          } catch {
            return;
          }
        }
        /** Minimum hidden size considering explicit minHidden or multiplier policy. */
        getMinimumHiddenSize(multiplierOverride) {
          const o = this.options;
          if (typeof o.minHidden === "number") return o.minHidden;
          const mult = multiplierOverride ?? o.minHiddenMultiplier;
          if (typeof mult === "number" && isFinite(mult)) {
            return Math.max(0, Math.round(mult * (this.input + this.output)));
          }
          return 0;
        }
        /** Produce `count` deterministic random samples using instance RNG. */
        sampleRandom(count) {
          const rng = this._getRNG();
          const arr = [];
          for (let i = 0; i < count; i++) arr.push(rng());
          return arr;
        }
        /** Internal: return cached objective descriptors, building if stale. */
        _getObjectives() {
          return _getObjectives.call(this);
        }
        /** Public helper returning just the objective keys (tests rely on). */
        getObjectiveKeys() {
          return this._getObjectives().map(
            (obj) => obj.key
          );
        }
        /** Invalidate per-genome caches (compatibility distance, forward pass, etc.). */
        _invalidateGenomeCaches(genome) {
          if (!genome || typeof genome !== "object") return;
          delete genome._compatCache;
          delete genome._outputCache;
          delete genome._traceCache;
        }
        /** Compute and cache diversity statistics used by telemetry & tests. */
        _computeDiversityStats() {
          this._diversityStats = computeDiversityStats2(this.population, this);
        }
        // Removed thin wrappers _structuralEntropy and _fastNonDominated; modules used directly where needed.
        /** Compatibility wrapper retained for tests that reference (neat as any)._structuralEntropy */
        _structuralEntropy(genome) {
          return structuralEntropy2(genome);
        }
        /**
         * Applies mutations to the population based on the mutation rate and amount.
         * Each genome is mutated using the selected mutation methods.
         * Slightly increases the chance of ADD_CONN mutation for more connectivity.
         */
        mutate() {
          return mutate.call(this);
        }
        // Perform ADD_NODE honoring global innovation reuse mapping
        _mutateAddNodeReuse(genome) {
          return mutateAddNodeReuse.call(this, genome);
        }
        _mutateAddConnReuse(genome) {
          return mutateAddConnReuse.call(this, genome);
        }
        // --- Speciation helpers (properly scoped) ---
        _fallbackInnov(conn) {
          return _fallbackInnov.call(this, conn);
        }
        _compatibilityDistance(netA, netB) {
          return _compatibilityDistance.call(this, netA, netB);
        }
        /**
         * Assign genomes into species based on compatibility distance and maintain species structures.
         * This function creates new species for unassigned genomes and prunes empty species.
         * It also records species-level history used for telemetry and adaptive controllers.
         */
        _speciate() {
          return _speciate.call(this);
        }
        /**
         * Apply fitness sharing within species. When `sharingSigma` > 0 this uses a kernel-based
         * sharing; otherwise it falls back to classic per-species averaging. Sharing reduces
         * effective fitness for similar genomes to promote diversity.
         */
        _applyFitnessSharing() {
          return _applyFitnessSharing.call(this);
        }
        /**
         * Sort members of a species in-place by descending score.
         * @param sp - Species object with `members` array.
         */
        _sortSpeciesMembers(sp) {
          return _sortSpeciesMembers.call(this, sp);
        }
        /**
         * Update species stagnation tracking and remove species that exceeded the allowed stagnation.
         */
        _updateSpeciesStagnation() {
          return _updateSpeciesStagnation.call(this);
        }
        /**
         * Return a concise summary for each current species.
         *
         * Educational context: In NEAT, populations are partitioned into species based
         * on genetic compatibility. Each species groups genomes that are similar so
         * selection and reproduction can preserve diversity between groups. This
         * accessor provides a lightweight view suitable for telemetry, visualization
         * and teaching examples without exposing full genome objects.
         *
         * The returned array contains objects with these fields:
         * - id: numeric species identifier
         * - size: number of members currently assigned to the species
         * - bestScore: the best observed fitness score for the species
         * - lastImproved: generation index when the species last improved its best score
         *
         * Notes for learners:
         * - Species sizes and lastImproved are typical signals used to detect
         *   stagnation and apply protective or penalizing measures.
         * - This function intentionally avoids returning full member lists to
         *   prevent accidental mutation of internal state; use `getSpeciesHistory`
         *   for richer historical data.
         *
         * @returns An array of species summary objects.
         */
        getSpeciesStats() {
          return getSpeciesStats.call(this);
        }
        /**
         * Returns the historical species statistics recorded each generation.
         *
         * Educational context: Species history captures per-generation snapshots
         * of species-level metrics (size, best score, last improvement) and is
         * useful for plotting trends, teaching about speciation dynamics, and
         * driving adaptive controllers.
         *
         * The returned array contains entries with a `generation` index and a
         * `stats` array containing per-species summaries recorded at that
         * generation.
         *
         * @returns An array of generation-stamped species stat snapshots.
         */
        getSpeciesHistory() {
          return getSpeciesHistory.call(this);
        }
        /**
         * Returns the number of entries currently stored in the novelty archive.
         *
         * Educational context: The novelty archive stores representative behaviors
         * used by behavior-based novelty search. Monitoring its size helps teach
         * how behavioral diversity accumulates over time and can be used to
         * throttle archive growth.
         *
         * @returns Number of archived behaviors.
         */
        getNoveltyArchiveSize() {
          return this._noveltyArchive ? this._noveltyArchive.length : 0;
        }
        /**
         * Returns compact multi-objective metrics for each genome in the current
         * population. The metrics include Pareto rank and crowding distance (if
         * computed), along with simple size and score measures useful in
         * instructional contexts.
         *
         * @returns Array of per-genome MO metric objects.
         */
        getMultiObjectiveMetrics() {
          return this.population.map((genome) => ({
            rank: genome._moRank ?? 0,
            crowding: genome._moCrowd ?? 0,
            score: genome.score || 0,
            nodes: genome.nodes.length,
            connections: genome.connections.length
          }));
        }
        /**
         * Returns a summary of mutation/operator statistics used by operator
         * adaptation and bandit selection.
         *
         * Educational context: Operator statistics track how often mutation
         * operators are attempted and how often they succeed. These counters are
         * used by adaptation mechanisms to bias operator selection towards
         * successful operators.
         *
         * @returns Array of { name, success, attempts } objects.
         */
        getOperatorStats() {
          return Array.from(this._operatorStats.entries()).map(
            ([operatorName, stats]) => ({
              name: operatorName,
              success: stats.success,
              attempts: stats.attempts
            })
          );
        }
        /**
         * Manually apply evolution-time pruning once using the current generation
         * index and configuration in `options.evolutionPruning`.
         *
         * Educational usage: While pruning normally occurs automatically inside
         * the evolve loop, exposing this method lets learners trigger the pruning
         * logic in isolation to observe its effect on network sparsity.
         *
         * Implementation detail: Delegates to the migrated helper in
         * `neat.pruning.ts` so the core class surface remains thin.
         */
        applyEvolutionPruning() {
          try {
            (init_neat_pruning(), __toCommonJS(neat_pruning_exports)).applyEvolutionPruning.call(this);
          } catch {
          }
        }
        /**
         * Run the adaptive pruning controller once. This adjusts the internal
         * `_adaptivePruneLevel` based on the configured metric (nodes or
         * connections) and invokes per-genome pruning when an adjustment is
         * warranted.
         *
         * Educational usage: Allows step-wise observation of how the adaptive
         * controller converges population complexity toward a target sparsity.
         */
        applyAdaptivePruning() {
          try {
            (init_neat_pruning(), __toCommonJS(neat_pruning_exports)).applyAdaptivePruning.call(this);
          } catch {
          }
        }
        /**
         * Return the internal telemetry buffer.
         *
         * Telemetry entries are produced per-generation when telemetry is enabled
         * and include diagnostic metrics (diversity, performance, lineage, etc.).
         * This accessor returns the raw buffer for external inspection or export.
         *
         * @returns Array of telemetry snapshot objects.
         */
        getTelemetry() {
          return this._telemetry;
        }
        /**
         * Export telemetry as JSON Lines (one JSON object per line).
         *
         * Useful for piping telemetry to external loggers or analysis tools.
         *
         * @returns A newline-separated string of JSON objects.
         */
        exportTelemetryJSONL() {
          return exportTelemetryJSONL.call(this);
        }
        /**
         * Export recent telemetry entries as CSV.
         *
         * The exporter attempts to flatten commonly-used nested fields (complexity,
         * perf, lineage) into columns. This is a best-effort exporter intended for
         * human inspection and simple ingestion.
         *
         * @param maxEntries Maximum number of recent telemetry entries to include.
         * @returns CSV string (may be empty when no telemetry present).
         */
        exportTelemetryCSV(maxEntries = 500) {
          return exportTelemetryCSV.call(this, maxEntries);
        }
        /**
         * Export telemetry as CSV with flattened columns for common nested fields.
         */
        clearTelemetry() {
          this._telemetry = [];
        }
        /** Clear all collected telemetry entries. */
        getObjectives() {
          return this._getObjectives().map((o) => ({
            key: o.key,
            direction: o.direction
          }));
        }
        getObjectiveEvents() {
          return this._objectiveEvents.slice();
        }
        /** Get recent objective add/remove events. */
        getLineageSnapshot(limit = 20) {
          return this.population.slice(0, limit).map((genome) => ({
            id: genome._id ?? -1,
            parents: Array.isArray(genome._parents) ? genome._parents.slice() : []
          }));
        }
        /**
         * Return an array of {id, parents} for the first `limit` genomes in population.
         */
        exportSpeciesHistoryCSV(maxEntries = 200) {
          return exportSpeciesHistoryCSV.call(this, maxEntries);
        }
        /**
         * Export species history as CSV.
         *
         * Produces rows for each recorded per-species stat entry within the
         * specified window. Useful for quick inspection or spreadsheet analysis.
         *
         * @param maxEntries Maximum history entries to include (default: 200).
         * @returns CSV string (may be empty).
         */
        getParetoFronts(maxFronts = 3) {
          if (!this.options.multiObjective?.enabled) return [[...this.population]];
          const fronts = [];
          for (let frontIdx = 0; frontIdx < maxFronts; frontIdx++) {
            const front = this.population.filter(
              (genome) => (genome._moRank ?? 0) === frontIdx
            );
            if (!front.length) break;
            fronts.push(front);
          }
          return fronts;
        }
        /**
         * Return the latest cached diversity statistics.
         *
         * Educational context: diversity metrics summarize how genetically and
         * behaviorally spread the population is. They can include lineage depth,
         * pairwise genetic distances, and other aggregated measures used by
         * adaptive controllers, novelty search, and telemetry. This accessor returns
         * whatever precomputed diversity object the Neat instance holds (may be
         * undefined if not computed for the current generation).
         *
         * @returns Arbitrary diversity summary object or undefined.
         */
        getDiversityStats() {
          return this._diversityStats;
        }
        registerObjective(key, direction, accessor) {
          return registerObjective.call(this, key, direction, accessor);
        }
        /**
         * Register a custom objective for multi-objective optimization.
         *
         * Educational context: multi-objective optimization lets you optimize for
         * multiple, potentially conflicting goals (e.g., maximize fitness while
         * minimizing complexity). Each objective is identified by a unique key and
         * an accessor function mapping a genome to a numeric score. Registering an
         * objective makes it visible to the internal MO pipeline and clears any
         * cached objective list so changes take effect immediately.
         *
         * @param key Unique objective key.
         * @param direction 'min' or 'max' indicating optimization direction.
         * @param accessor Function mapping a genome to a numeric objective value.
         */
        /**
         * Clear all registered multi-objective objectives.
         *
         * Removes any objectives configured for multi-objective optimization and
         * clears internal caches. Useful for tests or when reconfiguring the MO
         * setup at runtime.
         */
        clearObjectives() {
          return clearObjectives.call(this);
        }
        // Advanced archives & performance accessors
        /**
         * Get recent Pareto archive entries (meta information about archived fronts).
         *
         * Educational context: when performing multi-objective search we may store
         * representative Pareto-front snapshots over time. This accessor returns the
         * most recent archive entries up to the provided limit.
         *
         * @param maxEntries Maximum number of entries to return (default: 50).
         * @returns Array of archived Pareto metadata entries.
         */
        getParetoArchive(maxEntries = 50) {
          return this._paretoArchive.slice(-maxEntries);
        }
        /**
         * Export Pareto front archive as JSON Lines for external analysis.
         *
         * Each line is a JSON object representing one archived Pareto snapshot.
         *
         * @param maxEntries Maximum number of entries to include (default: 100).
         * @returns Newline-separated JSON objects.
         */
        exportParetoFrontJSONL(maxEntries = 100) {
          const slice = this._paretoObjectivesArchive.slice(-maxEntries);
          return slice.map((e) => JSON.stringify(e)).join("\n");
        }
        /**
         * Return recent performance statistics (durations in milliseconds) for the
         * most recent evaluation and evolve operations.
         *
         * Provides wall-clock timing useful for profiling and teaching how runtime
         * varies with network complexity or population settings.
         *
         * @returns Object with { lastEvalMs, lastEvolveMs }.
         */
        getPerformanceStats() {
          return {
            lastEvalMs: this._lastEvalDuration,
            lastEvolveMs: this._lastEvolveDuration
          };
        }
        // Utility exports / maintenance
        /**
         * Export species history as JSON Lines for storage and analysis.
         *
         * Each line is a JSON object containing a generation index and per-species
         * stats recorded at that generation. Useful for long-term tracking.
         *
         * @param maxEntries Maximum history entries to include (default: 200).
         * @returns Newline-separated JSON objects.
         */
        exportSpeciesHistoryJSONL(maxEntries = 200) {
          const slice = this._speciesHistory.slice(-maxEntries);
          return slice.map((e) => JSON.stringify(e)).join("\n");
        }
        /**
         * Reset the novelty archive (clear entries).
         *
         * The novelty archive is used to keep representative behaviors for novelty
         * search. Clearing it removes stored behaviors.
         */
        resetNoveltyArchive() {
          this._noveltyArchive = [];
        }
        /**
         * Clear the Pareto archive.
         *
         * Removes any stored Pareto-front snapshots retained by the algorithm.
         */
        clearParetoArchive() {
          this._paretoArchive = [];
        }
        /**
         * Sorts the population in descending order of fitness scores.
         * Ensures that the fittest genomes are at the start of the population array.
         */
        sort() {
          return sort.call(this);
        }
        /**
         * Selects a parent genome for breeding based on the selection method.
         * Supports multiple selection strategies, including POWER, FITNESS_PROPORTIONATE, and TOURNAMENT.
         * @returns The selected parent genome.
         * @throws Error if tournament size exceeds population size.
         */
        getParent() {
          return getParent.call(this);
        }
        /**
         * Retrieves the fittest genome from the population.
         * Ensures that the population is evaluated and sorted before returning the result.
         * @returns The fittest genome in the population.
         */
        getFittest() {
          return getFittest.call(this);
        }
        /**
         * Calculates the average fitness score of the population.
         * Ensures that the population is evaluated before calculating the average.
         * @returns The average fitness score of the population.
         */
        getAverage() {
          return getAverage.call(this);
        }
        /**
         * Exports the current population as an array of JSON objects.
         * Useful for saving the state of the population for later use.
         * @returns An array of JSON representations of the population.
         */
        export() {
          return exportPopulation.call(this);
        }
        /**
         * Imports a population from an array of JSON objects.
         * Replaces the current population with the imported one.
         * @param json - An array of JSON objects representing the population.
         */
        import(json) {
          return importPopulation.call(this, json);
        }
        /**
         * Convenience: export full evolutionary state (meta + population genomes).
         * Combines innovation registries and serialized genomes for easy persistence.
         */
        exportState() {
          return exportState.call(this);
        }
        /**
         * Convenience: restore full evolutionary state previously produced by exportState().
         * @param bundle Object with shape { neat, population }
         * @param fitness Fitness function to attach
         */
        static importState(bundle, fitness) {
          return importStateImpl.call(_Neat, bundle, fitness);
        }
        /**
         * Import a previously exported state bundle and rehydrate a Neat instance.
         */
        // Serialize NEAT meta (without population) for persistence of innovation history
        toJSON() {
          return toJSONImpl2.call(this);
        }
        static fromJSON(json, fitness) {
          return fromJSONImpl2.call(_Neat, json, fitness);
        }
      };
    }
  });

  // test/examples/asciiMaze/browserTerminalUtility.ts
  var BrowserTerminalUtility = class {
    /**
     * Create a clearer that clears a DOM container's contents.
     * If no container is provided it will try to use an element with id "ascii-maze-output".
     */
    static createTerminalClearer(container) {
      const el = container ?? (typeof document !== "undefined" ? document.getElementById("ascii-maze-output") : null);
      return () => {
        if (el) el.innerHTML = "";
      };
    }
    /**
     * Same semantics as the Node version: repeatedly call evolveFn until success or threshold reached.
     */
    static async evolveUntilSolved(evolveFn, minProgressToPass = 60, maxTries = 10) {
      let tries = 0;
      let lastResult = {
        success: false,
        progress: 0
      };
      while (tries < maxTries) {
        tries++;
        const { finalResult } = await evolveFn();
        lastResult = finalResult;
        if (finalResult.success || finalResult.progress >= minProgressToPass) {
          return { finalResult, tries };
        }
      }
      return { finalResult: lastResult, tries };
    }
  };

  // test/examples/asciiMaze/browserLogger.ts
  var ANSI_256_MAP = {
    205: "#ff6ac1",
    93: "#b48bf2",
    154: "#a6d189",
    51: "#00bcd4",
    226: "#ffd166",
    214: "#ff9f43",
    196: "#ff3b30",
    46: "#00e676",
    123: "#6ec6ff",
    177: "#caa6ff",
    80: "#00bfa5",
    121: "#9bdc8a",
    203: "#ff6b9f",
    99: "#6b62d6",
    44: "#00a9e0",
    220: "#ffd54f",
    250: "#ececec",
    45: "#00aaff",
    201: "#ff4fc4",
    231: "#ffffff",
    218: "#ffc6d3",
    217: "#ffcdb5",
    117: "#6fb3ff",
    118: "#6ee07a",
    48: "#00a300",
    57: "#2f78ff",
    33: "#1e90ff",
    87: "#00d7ff",
    159: "#cfeeff",
    208: "#ff8a00",
    197: "#ff5ea6",
    234: "#0e1114",
    23: "#123044",
    17: "#000b16",
    16: "#000000",
    39: "#0078ff"
  };
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function ensurePre(container) {
    const host = container ?? (typeof document !== "undefined" ? document.getElementById("ascii-maze-output") : null);
    if (!host) return null;
    let pre = host.querySelector("pre");
    if (!pre) {
      pre = document.createElement("pre");
      pre.style.fontFamily = "monospace";
      pre.style.whiteSpace = "pre";
      pre.style.margin = "0";
      pre.style.padding = "4px";
      pre.style.fontSize = "10px";
      host.appendChild(pre);
    }
    return pre;
  }
  function ansiToHtml(input) {
    const re = /\x1b\[([0-9;]*)m/g;
    let out = "";
    let lastIndex = 0;
    let style = {};
    let match;
    while ((match = re.exec(input)) !== null) {
      const chunk = input.substring(lastIndex, match.index);
      if (chunk) {
        const text = escapeHtml(chunk);
        if (Object.keys(style).length) {
          const css = [];
          if (style.color) css.push(`color: ${style.color}`);
          if (style.background) css.push(`background: ${style.background}`);
          if (style.fontWeight) css.push(`font-weight: ${style.fontWeight}`);
          out += `<span style="${css.join(";")}">${text}</span>`;
        } else {
          out += text;
        }
      }
      const codes = match[1].split(";").filter((c) => c.length).map((c) => parseInt(c, 10));
      if (codes.length === 0) {
        style = {};
      } else {
        for (let i = 0; i < codes.length; i++) {
          const c = codes[i];
          if (c === 0) {
            style = {};
          } else if (c === 1) {
            style.fontWeight = "700";
          } else if (c === 22) {
            delete style.fontWeight;
          } else if (c === 38 && codes[i + 1] === 5) {
            const n = codes[i + 2];
            if (typeof n === "number" && ANSI_256_MAP[n])
              style.color = ANSI_256_MAP[n];
            i += 2;
          } else if (c === 48 && codes[i + 1] === 5) {
            const n = codes[i + 2];
            if (typeof n === "number" && ANSI_256_MAP[n])
              style.background = ANSI_256_MAP[n];
            i += 2;
          } else if (c >= 30 && c <= 37) {
            const basic = [
              "#000000",
              "#800000",
              "#008000",
              "#808000",
              "#000080",
              "#800080",
              "#008080",
              "#c0c0c0"
            ];
            style.color = basic[c - 30];
          } else if (c >= 90 && c <= 97) {
            const bright = [
              "#808080",
              "#ff0000",
              "#00ff00",
              "#ffff00",
              "#0000ff",
              "#ff00ff",
              "#00ffff",
              "#ffffff"
            ];
            style.color = bright[c - 90];
          } else if (c === 39) {
            delete style.color;
          } else if (c === 49) {
            delete style.background;
          }
        }
      }
      lastIndex = re.lastIndex;
    }
    if (lastIndex < input.length) {
      const tail = escapeHtml(input.substring(lastIndex));
      if (Object.keys(style).length) {
        const css = [];
        if (style.color) css.push(`color: ${style.color}`);
        if (style.background) css.push(`background: ${style.background}`);
        if (style.fontWeight) css.push(`font-weight: ${style.fontWeight}`);
        out += `<span style="${css.join(";")}">${tail}</span>`;
      } else {
        out += tail;
      }
    }
    return out;
  }
  function createBrowserLogger(container) {
    return (...args) => {
      const pre = ensurePre(container);
      let opts = void 0;
      if (args.length && typeof args[args.length - 1] === "object" && args[args.length - 1] && "prepend" in args[args.length - 1]) {
        opts = args[args.length - 1];
        args = args.slice(0, -1);
      }
      const text = args.map((a) => typeof a === "string" ? a : JSON.stringify(a)).join(" ");
      const html = ansiToHtml(text).replace(/\n/g, "<br/>") + "<br/>";
      if (!pre) return;
      if (opts && opts.prepend) {
        pre.innerHTML = html + pre.innerHTML;
        pre.scrollTop = 0;
      } else {
        pre.innerHTML += html;
        pre.scrollTop = pre.scrollHeight;
      }
    };
  }

  // test/examples/asciiMaze/mazeUtils.ts
  var MazeUtils = class _MazeUtils {
    /**
     * Converts an ASCII/Unicode maze (array of strings) into a 2D numeric array for processing by the agent.
     *
     * Encoding:
     *   '#' = -1 (wall/obstacle)
     *   Box drawing characters (═,║,╔,╗,╚,╝,╠,╣,╦,╩,╬) = -1 (wall/obstacle)
     *   '.' = 0 (open path)
     *   'E' = 1 (exit/goal)
     *   'S' = 2 (start position)
     *   any other character = 0 (treated as open path)
     *
     * @param asciiMaze - Array of strings representing the maze.
     * @returns 2D array of numbers encoding the maze elements.
     */
    static encodeMaze(asciiMaze) {
      const wallChars = /* @__PURE__ */ new Set([
        "#",
        "\u2550",
        "\u2551",
        "\u2554",
        "\u2557",
        "\u255A",
        "\u255D",
        "\u2560",
        "\u2563",
        "\u2566",
        "\u2569",
        "\u256C"
      ]);
      return asciiMaze.map(
        (row) => [...row].map((cell) => {
          if (wallChars.has(cell)) return -1;
          switch (cell) {
            case ".":
              return 0;
            case "E":
              return 1;
            case "S":
              return 2;
            default:
              return 0;
          }
        })
      );
    }
    /**
     * Finds the (x, y) position of a given character in the ASCII maze.
     * @param asciiMaze - Array of strings representing the maze.
     * @param char - Character to find (e.g., 'S' for start, 'E' for exit).
     * @returns [x, y] coordinates of the character.
     * @throws Error if the character is not found in the maze.
     */
    static findPosition(asciiMaze, char) {
      for (let y = 0; y < asciiMaze.length; y++) {
        const x = asciiMaze[y].indexOf(char);
        if (x !== -1) return [x, y];
      }
      throw new Error(`Character ${char} not found in maze`);
    }
    /**
     * Computes the shortest path distance between two points in the maze using BFS.
     * Returns Infinity if no path exists.
     * @param encodedMaze - 2D array representation of the maze.
     * @param start - [x, y] start position.
     * @param goal - [x, y] goal position.
     * @returns Shortest path length (number of steps), or Infinity if unreachable.
     */
    static bfsDistance(encodedMaze, start2, goal) {
      const [gx, gy] = goal;
      if (encodedMaze[gy][gx] === -1) return Infinity;
      const queue = [[start2, 0]];
      const visited = /* @__PURE__ */ new Set();
      const key = ([x, y]) => `${x},${y}`;
      visited.add(key(start2));
      const directions = [
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0]
      ];
      while (queue.length > 0) {
        const [[x, y], dist] = queue.shift();
        if (x === gx && y === gy) return dist;
        for (const [dx, dy] of directions) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && ny >= 0 && ny < encodedMaze.length && nx < encodedMaze[0].length && encodedMaze[ny][nx] !== -1 && !visited.has(key([nx, ny]))) {
            visited.add(key([nx, ny]));
            queue.push([[nx, ny], dist + 1]);
          }
        }
      }
      return Infinity;
    }
    /**
     * Calculates the agent's progress toward the exit as a percentage.
     * Progress is measured as the proportion of the shortest path covered from start to exit.
     * @param encodedMaze - 2D array representation of the maze.
     * @param currentPos - [x, y] current agent position.
     * @param startPos - [x, y] start position.
     * @param exitPos - [x, y] exit position.
     * @returns Progress percentage (0-100).
     */
    static calculateProgress(encodedMaze, currentPos, startPos, exitPos) {
      const totalDistance = _MazeUtils.bfsDistance(encodedMaze, startPos, exitPos);
      if (totalDistance === 0) return 100;
      const remainingDistance = _MazeUtils.bfsDistance(
        encodedMaze,
        currentPos,
        exitPos
      );
      return Math.min(
        100,
        Math.max(
          0,
          Math.round((totalDistance - remainingDistance) / totalDistance * 100)
        )
      );
    }
    /**
     * Calculates progress using a precomputed distance map (goal-centric BFS distances).
     * Faster alternative to repeated BFS calls. Distance map holds distance from each cell TO the exit (goal).
     * @param distanceMap - 2D array of distances (Infinity for walls/unreachable)
     * @param currentPos - Agent current position [x,y]
     * @param startPos - Start position [x,y]
     * @returns Progress percentage (0-100)
     */
    static calculateProgressFromDistanceMap(distanceMap, currentPos, startPos) {
      const [sx, sy] = startPos;
      const [cx, cy] = currentPos;
      const totalDistance = distanceMap[sy]?.[sx];
      const remaining = distanceMap[cy]?.[cx];
      if (totalDistance == null || remaining == null || !isFinite(totalDistance) || totalDistance <= 0)
        return 0;
      const prog = (totalDistance - remaining) / totalDistance * 100;
      return Math.min(100, Math.max(0, Math.round(prog)));
    }
    /**
     * Builds a full distance map (Manhattan shortest path lengths via BFS) from a goal cell to every reachable cell.
     * Walls are marked as Infinity. Unreachable cells remain Infinity.
     * @param encodedMaze - 2D maze encoding
     * @param goal - [x,y] goal position (typically exit)
     */
    static buildDistanceMap(encodedMaze, goal) {
      const height = encodedMaze.length;
      const width = encodedMaze[0].length;
      const dist = Array.from(
        { length: height },
        () => Array(width).fill(Infinity)
      );
      const [gx, gy] = goal;
      if (encodedMaze[gy][gx] === -1) return dist;
      const q = [[gx, gy]];
      dist[gy][gx] = 0;
      const dirs = [
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0]
      ];
      while (q.length) {
        const [x, y] = q.shift();
        const d = dist[y][x];
        for (const [dx, dy] of dirs) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && ny >= 0 && ny < height && nx < width && encodedMaze[ny][nx] !== -1 && dist[ny][nx] === Infinity) {
            dist[ny][nx] = d + 1;
            q.push([nx, ny]);
          }
        }
      }
      return dist;
    }
  };

  // test/examples/asciiMaze/colors.ts
  var colors = {
    // Basic formatting
    reset: "\x1B[0m",
    // Reset all attributes
    bright: "\x1B[1m",
    // Bright/bold text
    dim: "\x1B[2m",
    // Dim text
    // Neon foreground colors (expanded palette)
    neonPink: "\x1B[38;5;205m",
    // Neon pink
    neonPurple: "\x1B[38;5;93m",
    // Neon purple
    neonLime: "\x1B[38;5;154m",
    // Neon lime green
    neonAqua: "\x1B[38;5;51m",
    // Neon aqua
    neonYellow: "\x1B[38;5;226m",
    // Neon yellow
    neonOrange: "\x1B[38;5;214m",
    // Neon orange (brighter)
    neonRed: "\x1B[38;5;196m",
    // Neon red
    neonGreen: "\x1B[38;5;46m",
    // Neon green
    neonSky: "\x1B[38;5;123m",
    // Neon sky blue
    neonViolet: "\x1B[38;5;177m",
    // Neon violet
    neonTurquoise: "\x1B[38;5;80m",
    // Neon turquoise
    neonMint: "\x1B[38;5;121m",
    // Neon mint
    neonCoral: "\x1B[38;5;203m",
    // Neon coral
    neonIndigo: "\x1B[38;5;99m",
    // Neon indigo
    neonTeal: "\x1B[38;5;44m",
    // Neon teal
    neonGold: "\x1B[38;5;220m",
    // Neon gold
    neonSilver: "\x1B[38;5;250m",
    // Neon silver
    neonBlue: "\x1B[38;5;45m",
    // Neon blue (extra)
    neonMagenta: "\x1B[38;5;201m",
    // Neon magenta (extra)
    neonCyan: "\x1B[38;5;87m",
    // Neon cyan (extra)
    neonWhite: "\x1B[38;5;231m",
    // Neon white (brightest)
    neonRose: "\x1B[38;5;218m",
    // Neon rose
    neonPeach: "\x1B[38;5;217m",
    // Neon peach
    neonAzure: "\x1B[38;5;117m",
    // Neon azure
    neonChartreuse: "\x1B[38;5;118m",
    // Neon chartreuse
    neonSpring: "\x1B[38;5;48m",
    // Neon spring green
    neonAmber: "\x1B[38;5;214m",
    // Neon amber (duplicate of orange, for clarity)
    neonFuchsia: "\x1B[38;5;207m",
    // Neon fuchsia
    // TRON primary colors (foreground)
    blueCore: "\x1B[38;5;39m",
    // Primary TRON blue
    cyanNeon: "\x1B[38;5;87m",
    // Electric cyan
    blueNeon: "\x1B[38;5;45m",
    // Bright neon blue
    whiteNeon: "\x1B[38;5;159m",
    // Electric white-blue
    orangeNeon: "\x1B[38;5;208m",
    // TRON orange (for contrast)
    magentaNeon: "\x1B[38;5;201m",
    // Digital magenta
    // Base colors (foreground)
    red: "\x1B[38;5;197m",
    // Program termination red
    green: "\x1B[38;5;118m",
    // User/CLU green
    yellow: "\x1B[38;5;220m",
    // Warning yellow
    blue: "\x1B[38;5;33m",
    // Deep blue
    cyan: "\x1B[38;5;51m",
    // Light cyan
    // Neon background colors (expanded palette)
    bgNeonPink: "\x1B[48;5;205m",
    bgNeonPurple: "\x1B[48;5;93m",
    bgNeonLime: "\x1B[48;5;154m",
    bgNeonAqua: "\x1B[48;5;51m",
    bgNeonYellow: "\x1B[48;5;226m",
    bgNeonOrange: "\x1B[48;5;214m",
    bgNeonRed: "\x1B[48;5;196m",
    bgNeonGreen: "\x1B[48;5;46m",
    bgNeonSky: "\x1B[48;5;123m",
    bgNeonViolet: "\x1B[48;5;177m",
    bgNeonTurquoise: "\x1B[48;5;80m",
    bgNeonMint: "\x1B[48;5;121m",
    bgNeonCoral: "\x1B[48;5;203m",
    bgNeonIndigo: "\x1B[48;5;99m",
    bgNeonTeal: "\x1B[48;5;44m",
    bgNeonGold: "\x1B[48;5;220m",
    bgNeonSilver: "\x1B[48;5;250m",
    bgNeonBlue: "\x1B[48;5;45m",
    // Neon blue background (extra)
    bgNeonMagenta: "\x1B[48;5;201m",
    // Neon magenta background (extra)
    bgNeonCyan: "\x1B[48;5;87m",
    // Neon cyan background (extra)
    bgNeonWhite: "\x1B[48;5;231m",
    // Neon white background (brightest)
    bgNeonRose: "\x1B[48;5;218m",
    // Neon rose background
    bgNeonPeach: "\x1B[48;5;217m",
    // Neon peach background
    bgNeonAzure: "\x1B[48;5;117m",
    // Neon azure background
    bgNeonChartreuse: "\x1B[48;5;118m",
    // Neon chartreuse background
    bgNeonSpring: "\x1B[48;5;48m",
    // Neon spring green background
    bgNeonAmber: "\x1B[48;5;214m",
    // Neon amber background (duplicate of orange, for clarity)
    bgNeonFuchsia: "\x1B[48;5;207m",
    // Neon fuchsia background
    // TRON background colors
    bgBlueCore: "\x1B[48;5;39m",
    // Primary TRON blue background
    bgCyanNeon: "\x1B[48;5;87m",
    // Electric cyan background (for agent)
    bgBlueNeon: "\x1B[48;5;45m",
    // Bright neon blue background
    bgWhiteNeon: "\x1B[48;5;159m",
    // Electric white-blue background
    bgOrangeNeon: "\x1B[48;5;208m",
    // TRON orange background
    bgMagentaNeon: "\x1B[48;5;201m",
    // Digital magenta background
    // Common backgrounds
    bgRed: "\x1B[48;5;197m",
    // Program termination red background
    bgGreen: "\x1B[48;5;118m",
    // User/CLU green background
    bgYellow: "\x1B[48;5;220m",
    // Warning yellow background
    bgBlue: "\x1B[48;5;33m",
    // Deep blue background
    // Maze-specific colors
    darkWallBg: "\x1B[48;5;17m",
    // Dark blue for walls
    darkWallText: "\x1B[38;5;17m",
    // Dark blue text for wall symbols
    floorBg: "\x1B[48;5;234m",
    // Almost black for empty floor
    floorText: "\x1B[38;5;234m",
    // Almost black text for floor symbols
    gridLineBg: "\x1B[48;5;23m",
    // Subtle grid line color
    gridLineText: "\x1B[38;5;23m",
    // Subtle grid line text
    // Special highlights
    bgBlack: "\x1B[48;5;16m",
    // Pure black background
    pureBlue: "\x1B[38;5;57;1m",
    // Vibrant system blue
    pureOrange: "\x1B[38;5;214;1m",
    // Vibrant TRON orange (for CLU/villains)
    pureGreen: "\x1B[38;5;46;1m"
    // Pure green for user programs
  };

  // test/examples/asciiMaze/networkVisualization.ts
  var NetworkVisualization = class _NetworkVisualization {
    /**
     * Pads a string to a specific width with alignment options.
     *
     * @param str - String to pad.
     * @param width - Target width for the string.
     * @param padChar - Character to use for padding (default: space).
     * @param align - Alignment option ('left', 'center', or 'right').
     * @returns Padded string of specified width with chosen alignment.
     */
    static pad(str, width, padChar = " ", align = "center") {
      str = str ?? "";
      const len = str.replace(/\x1b\[[0-9;]*m/g, "").length;
      if (len >= width) return str;
      const padLen = width - len;
      if (align === "left") return str + padChar.repeat(padLen);
      if (align === "right") return padChar.repeat(padLen) + str;
      const left = Math.floor(padLen / 2);
      const right = padLen - left;
      return padChar.repeat(left) + str + padChar.repeat(right);
    }
    /**
     * Gets activation value from a node, with safety checks.
     * For output nodes, ensures values are properly clamped between 0 and 1.
     *
     * @param node - Neural network node object.
     * @returns Cleaned and normalized activation value.
     */
    static getNodeValue(node) {
      if (typeof node.activation === "number" && isFinite(node.activation) && !isNaN(node.activation)) {
        if (node.type === "output") {
          return Math.max(0, Math.min(1, node.activation));
        }
        return Math.max(-999, Math.min(999, node.activation));
      }
      return 0;
    }
    /**
     * Gets the appropriate color for an activation value based on its range.
     * Uses a TRON-inspired color palette for activation values.
     *
     * @param value - Activation value to colorize.
     * @returns ANSI color code for the value.
     */
    static getActivationColor(value) {
      if (value >= 2) return colors.bgOrangeNeon + colors.bright;
      if (value >= 1) return colors.orangeNeon;
      if (value >= 0.5) return colors.cyanNeon;
      if (value >= 0.1) return colors.neonGreen;
      if (value >= -0.1) return colors.whiteNeon;
      if (value >= -0.5) return colors.blue;
      if (value >= -1) return colors.blueCore;
      if (value >= -2) return colors.bgNeonAqua + colors.bright;
      return colors.bgNeonViolet + colors.neonSilver;
    }
    /**
     * Formats a numeric value for display with color based on its value.
     *
     * @param v - Numeric value to format.
     * @returns Colorized string representation of the value.
     */
    static fmtColoredValue(v) {
      if (typeof v !== "number" || isNaN(v) || !isFinite(v)) return " 0.000";
      const color = _NetworkVisualization.getActivationColor(v);
      let formattedValue;
      formattedValue = (v >= 0 ? " " : "") + v.toFixed(6);
      return color + formattedValue + colors.reset;
    }
    /**
     * Groups hidden nodes into layers based on their connections.
     *
     * @param inputNodes - Array of input nodes.
     * @param hiddenNodes - Array of hidden nodes.
     * @param outputNodes - Array of output nodes.
     * @returns Array of hidden node arrays, each representing a layer.
     */
    static groupHiddenByLayer(inputNodes, hiddenNodes, outputNodes) {
      if (hiddenNodes.length === 0) return [];
      let layers = [];
      let prevLayer = inputNodes;
      let remaining = [...hiddenNodes];
      while (remaining.length > 0) {
        const currentLayer = remaining.filter(
          (h) => h.connections && h.connections.in && h.connections.in.length > 0 && h.connections.in.every((conn) => prevLayer.includes(conn.from))
        );
        if (currentLayer.length === 0) {
          layers.push(remaining);
          break;
        }
        layers.push(currentLayer);
        prevLayer = currentLayer;
        remaining = remaining.filter((h) => !currentLayer.includes(h));
      }
      return layers;
    }
    /**
     * Groups nodes by their activation values to create meaningful average representations.
     * Creates more granular grouping based on activation ranges.
     *
     * @param nodes - Array of neural network nodes to group.
     * @returns Object containing groups of nodes and corresponding labels.
     */
    static groupNodesByActivation(nodes) {
      const activations = nodes.map(
        (node) => _NetworkVisualization.getNodeValue(node)
      );
      const ranges = [
        { min: 2, max: Infinity, label: "v-high+" },
        { min: 1, max: 2, label: "high+" },
        { min: 0.5, max: 1, label: "mid+" },
        { min: 0.1, max: 0.5, label: "low+" },
        { min: -0.1, max: 0.1, label: "zero\xB1" },
        { min: -0.5, max: -0.1, label: "low-" },
        { min: -1, max: -0.5, label: "mid-" },
        { min: -2, max: -1, label: "high-" },
        { min: -Infinity, max: -2, label: "v-high-" }
      ];
      const groups = [];
      const labels = [];
      for (const range of ranges) {
        const nodesInRange = nodes.filter(
          (_, i) => activations[i] >= range.min && activations[i] < range.max
        );
        if (nodesInRange.length > 0) {
          groups.push(nodesInRange);
          labels.push(range.label);
        }
      }
      return { groups, labels };
    }
    /**
     * Prepares hidden layers for display, condensing large layers
     * to show all nodes as averages with meaningful distribution.
     *
     * @param hiddenLayers - Array of hidden layer node arrays.
     * @param maxVisiblePerLayer - Maximum number of nodes to display per layer.
     * @returns Object containing display-ready layers and metrics.
     */
    static prepareHiddenLayersForDisplay(hiddenLayers, maxVisiblePerLayer = 10) {
      const MAX_VISIBLE = maxVisiblePerLayer;
      const averageNodes = {};
      const displayLayers = [];
      const layerDisplayCounts = [];
      hiddenLayers.forEach((layer, layerIdx) => {
        if (layer.length <= MAX_VISIBLE) {
          displayLayers.push([...layer]);
          layerDisplayCounts.push(layer.length);
        } else {
          const { groups, labels } = _NetworkVisualization.groupNodesByActivation(
            layer
          );
          let finalGroups = groups;
          let finalLabels = labels;
          if (groups.length > MAX_VISIBLE) {
            const rankedGroups = groups.map((g, i) => ({
              group: g,
              label: labels[i],
              size: g.length
            })).sort((a, b) => b.size - a.size);
            const topGroups = rankedGroups.slice(0, MAX_VISIBLE - 1);
            const remainingGroups = rankedGroups.slice(MAX_VISIBLE - 1);
            const mergedGroup = remainingGroups.reduce(
              (acc, curr) => {
                acc.group = [...acc.group, ...curr.group];
                return acc;
              },
              { group: [], label: "other\xB1", size: 0 }
            );
            if (mergedGroup.group.length > 0) {
              topGroups.push(mergedGroup);
            }
            topGroups.sort((a, b) => {
              const aIsNegative = a.label.includes("-");
              const bIsNegative = b.label.includes("-");
              if (aIsNegative && !bIsNegative) return 1;
              if (!aIsNegative && bIsNegative) return -1;
              if (a.label.includes("v-") && !b.label.includes("v-"))
                return aIsNegative ? 1 : -1;
              if (!a.label.includes("v-") && b.label.includes("v-"))
                return aIsNegative ? -1 : 1;
              if (a.label.includes("high") && !b.label.includes("high"))
                return aIsNegative ? 1 : -1;
              if (!a.label.includes("high") && b.label.includes("high"))
                return aIsNegative ? -1 : 1;
              return 0;
            });
            finalGroups = topGroups.map((g) => g.group);
            finalLabels = topGroups.map((g) => g.label);
          }
          const avgNodes = finalGroups.map((group, groupIdx) => {
            const avgKey = `layer${layerIdx}-avg-${groupIdx}`;
            const sum = group.reduce(
              (acc, node) => acc + _NetworkVisualization.getNodeValue(node),
              0
            );
            const avgValue = group.length > 0 ? sum / group.length : 0;
            averageNodes[avgKey] = {
              avgValue,
              count: group.length
            };
            return {
              id: -1 * (layerIdx * 1e3 + groupIdx),
              uuid: avgKey,
              type: "hidden",
              activation: avgValue,
              isAverage: true,
              avgCount: group.length,
              label: finalLabels[groupIdx]
            };
          });
          displayLayers.push(avgNodes);
          layerDisplayCounts.push(avgNodes.length);
        }
      });
      return { displayLayers, layerDisplayCounts, averageNodes };
    }
    /**
     * Utility to create a visualization node from a neataptic node.
     *
     * @param node - Neural network node object.
     * @param index - Index of the node in the network.
     * @returns Visualization node object.
     */
    static toVisualizationNode(node, index) {
      const id = typeof node.index === "number" ? node.index : index;
      return {
        id,
        uuid: String(id),
        type: node.type,
        activation: node.activation,
        bias: node.bias
      };
    }
    /**
     * Visualizes a neural network's structure and activations in ASCII format.
     *
     * Creates a comprehensive terminal-friendly visualization showing:
     * - Network architecture with layers
     * - Node activation values with color coding
     * - Connection counts between layers
     * - Condensed representation of large hidden layers
     *
     * @param network - The neural network to visualize.
     * @returns String containing the ASCII visualization.
     */
    static visualizeNetworkSummary(network) {
      const ARROW = "  \u2500\u2500\u25B6  ";
      const ARROW_WIDTH = ARROW.length;
      const TOTAL_WIDTH = 150;
      const detectedInputNodes = (network.nodes || []).filter(
        (n) => n.type === "input" || n.type === "constant"
      );
      const INPUT_COUNT = detectedInputNodes.length || 18;
      const OUTPUT_COUNT = 4;
      const nodes = network.nodes || [];
      const inputNodes = nodes.filter((n) => n.type === "input" || n.type === "constant").map(_NetworkVisualization.toVisualizationNode);
      const outputNodes = nodes.filter((n) => n.type === "output").map(_NetworkVisualization.toVisualizationNode);
      const hiddenNodesRaw = nodes.filter((n) => n.type === "hidden").map(_NetworkVisualization.toVisualizationNode);
      const hiddenLayers = _NetworkVisualization.groupHiddenByLayer(
        inputNodes,
        hiddenNodesRaw,
        outputNodes
      );
      const numHiddenLayers = hiddenLayers.length;
      const {
        displayLayers,
        layerDisplayCounts,
        averageNodes
      } = _NetworkVisualization.prepareHiddenLayersForDisplay(hiddenLayers);
      const connections = (network.connections || []).map((conn) => ({
        weight: conn.weight,
        fromUUID: String(conn.from.index),
        // Use .index directly as per INodeStruct
        toUUID: String(conn.to.index),
        // Use .index directly as per INodeStruct
        gaterUUID: conn.gater ? String(conn.gater.index) : null,
        // Use .index directly
        enabled: typeof conn.enabled === "boolean" ? conn.enabled : true
      }));
      const connectionCounts = [];
      let firstCount = 0;
      const firstTargetLayer = hiddenLayers.length > 0 ? hiddenLayers[0] : outputNodes;
      for (const conn of network.connections || []) {
        if (inputNodes.some((n) => n.id === conn.from.index) && firstTargetLayer.some((n) => n.id === conn.to.index)) {
          firstCount++;
        }
      }
      connectionCounts.push(firstCount);
      for (let i = 0; i < hiddenLayers.length - 1; i++) {
        let count = 0;
        for (const conn of network.connections || []) {
          if (hiddenLayers[i].some((n) => n.id === conn.from.index) && hiddenLayers[i + 1].some((n) => n.id === conn.to.index)) {
            count++;
          }
        }
        connectionCounts.push(count);
      }
      if (hiddenLayers.length > 0) {
        let lastCount = 0;
        for (const conn of network.connections || []) {
          if (hiddenLayers[hiddenLayers.length - 1].some(
            (n) => n.id === conn.from.index
          ) && outputNodes.some((n) => n.id === conn.to.index)) {
            lastCount++;
          }
        }
        connectionCounts.push(lastCount);
      }
      const numLayers = 2 + numHiddenLayers;
      const numArrows = numLayers - 1;
      const availableWidth = TOTAL_WIDTH - numArrows * ARROW_WIDTH;
      const columnWidth = Math.floor(availableWidth / numLayers);
      let header = "";
      header += `${colors.blueCore}\u2551` + _NetworkVisualization.pad(
        `${colors.neonGreen}Input Layer [${INPUT_COUNT}]${colors.reset}`,
        columnWidth - 1
      );
      const firstConnCount = connectionCounts[0];
      const firstArrowText = `${colors.blueNeon}${firstConnCount} ${ARROW.trim()}${colors.reset}`;
      header += _NetworkVisualization.pad(firstArrowText, ARROW_WIDTH);
      for (let i = 0; i < numHiddenLayers; i++) {
        header += _NetworkVisualization.pad(
          `${colors.cyanNeon}Hidden ${i + 1} [${hiddenLayers[i].length}]${colors.reset}`,
          columnWidth
        );
        if (i < numHiddenLayers) {
          const connCount = connectionCounts[i + 1] || 0;
          const arrowText = `${colors.blueNeon}${connCount} ${ARROW.trim()}${colors.reset}`;
          header += _NetworkVisualization.pad(arrowText, ARROW_WIDTH);
        }
      }
      header += _NetworkVisualization.pad(
        `${colors.orangeNeon}Output Layer [${OUTPUT_COUNT}]${colors.reset}`,
        columnWidth,
        " ",
        "center"
      ) + `${colors.blueCore}\u2551${colors.reset}`;
      const inputDisplayNodes = Array(INPUT_COUNT).fill(null).map((_, i) => inputNodes[i] || { activation: 0 });
      const INPUT_LABELS6 = [
        "compass",
        "openN",
        "openE",
        "openS",
        "openW",
        "progress"
      ];
      const outputDisplayNodes = Array(OUTPUT_COUNT).fill(null).map((_, i) => outputNodes[i] || { activation: 0 });
      const maxRows = Math.max(INPUT_COUNT, ...layerDisplayCounts, OUTPUT_COUNT);
      const rows = [];
      for (let rowIdx = 0; rowIdx < maxRows; rowIdx++) {
        let row = "";
        if (rowIdx < INPUT_COUNT) {
          const node = inputDisplayNodes[rowIdx];
          const value = _NetworkVisualization.getNodeValue(node);
          const label = rowIdx < 6 ? INPUT_LABELS6[rowIdx] : "";
          const labelStr = label ? ` ${colors.whiteNeon}${label}${colors.reset}` : "";
          row += _NetworkVisualization.pad(
            `${colors.blueCore}\u2551   ${colors.neonGreen}\u25CF${colors.reset}${_NetworkVisualization.fmtColoredValue(value)}${labelStr}`,
            columnWidth,
            " ",
            "left"
          );
        } else {
          row += _NetworkVisualization.pad("", columnWidth);
        }
        if (rowIdx === 0) {
          const totalInputs = Math.min(INPUT_COUNT, inputNodes.length);
          const firstHiddenTotal = displayLayers[0]?.length || 0;
          if (totalInputs > 0 && firstHiddenTotal > 0) {
            const nodeProportion = Math.ceil(
              connectionCounts[0] / Math.max(1, totalInputs)
            );
            row += _NetworkVisualization.pad(
              `${colors.blueNeon}${nodeProportion} \u2500\u2500\u25B6${colors.reset}`,
              ARROW_WIDTH
            );
          } else {
            row += _NetworkVisualization.pad(
              `${colors.blueNeon}${ARROW}${colors.reset}`,
              ARROW_WIDTH
            );
          }
        } else if (rowIdx < INPUT_COUNT && rowIdx < displayLayers[0]?.length) {
          const totalInputs = Math.min(INPUT_COUNT, inputNodes.length);
          const firstHiddenTotal = displayLayers[0]?.length || 0;
          if (totalInputs > 0 && firstHiddenTotal > 0) {
            const nodeProportion = Math.ceil(
              connectionCounts[0] / Math.max(3, totalInputs * 2)
            );
            row += _NetworkVisualization.pad(
              `${colors.blueNeon}${nodeProportion} \u2500\u2500\u25B6${colors.reset}`,
              ARROW_WIDTH
            );
          } else {
            row += _NetworkVisualization.pad(
              `${colors.blueNeon}${ARROW}${colors.reset}`,
              ARROW_WIDTH
            );
          }
        } else {
          row += _NetworkVisualization.pad(
            `${colors.blueNeon}${ARROW}${colors.reset}`,
            ARROW_WIDTH
          );
        }
        for (let layerIdx = 0; layerIdx < numHiddenLayers; layerIdx++) {
          const layer = displayLayers[layerIdx];
          if (rowIdx < layer.length) {
            const node = layer[rowIdx];
            if (node.isAverage) {
              const labelText = node.label ? `${node.label} ` : "";
              const avgText = `${colors.cyanNeon}\u25A0${colors.reset}${_NetworkVisualization.fmtColoredValue(node.activation)} ${colors.dim}(${labelText}avg of ${node.avgCount})${colors.reset}`;
              row += _NetworkVisualization.pad(avgText, columnWidth, " ", "left");
            } else {
              const value = _NetworkVisualization.getNodeValue(node);
              row += _NetworkVisualization.pad(
                `${colors.cyanNeon}\u25A0${colors.reset}${_NetworkVisualization.fmtColoredValue(value)}`,
                columnWidth,
                " ",
                "left"
              );
            }
          } else {
            row += _NetworkVisualization.pad(" ", columnWidth);
          }
          if (layerIdx < numHiddenLayers - 1) {
            const connCount = connectionCounts[layerIdx + 1];
            if (rowIdx === 0) {
              const currentLayerSize = displayLayers[layerIdx]?.length || 1;
              const nodeProportion = Math.ceil(
                connCount / Math.max(3, currentLayerSize * 2)
              );
              row += _NetworkVisualization.pad(
                `${colors.blueNeon}${nodeProportion} \u2500\u2500\u25B6${colors.reset}`,
                ARROW_WIDTH
              );
            } else if (rowIdx < layer.length && rowIdx < displayLayers[layerIdx + 1]?.length) {
              const currentLayerSize = displayLayers[layerIdx]?.length || 1;
              const nextLayerSize = displayLayers[layerIdx + 1]?.length || 1;
              const proportion = Math.max(
                1,
                Math.min(5, Math.ceil(connCount / Math.max(3, currentLayerSize)))
              );
              row += _NetworkVisualization.pad(
                `${colors.blueNeon}${proportion} \u2500\u2500\u25B6${colors.reset}`,
                ARROW_WIDTH
              );
            } else {
              row += _NetworkVisualization.pad(
                `${colors.blueNeon}${ARROW}${colors.reset}`,
                ARROW_WIDTH
              );
            }
          } else {
            const connCount = connectionCounts[connectionCounts.length - 1];
            if (rowIdx === 0) {
              const lastLayerSize = displayLayers[displayLayers.length - 1]?.length || 1;
              const nodeProportion = Math.ceil(
                connCount / Math.max(3, lastLayerSize * 2)
              );
              row += _NetworkVisualization.pad(
                `${colors.blueNeon}${nodeProportion} \u2500\u2500\u25B6${colors.reset}`,
                ARROW_WIDTH
              );
            } else if (rowIdx < layer.length && rowIdx < OUTPUT_COUNT) {
              const lastLayerSize = displayLayers[displayLayers.length - 1]?.length || 1;
              const proportion = Math.max(
                1,
                Math.min(5, Math.ceil(connCount / Math.max(5, lastLayerSize * 2)))
              );
              row += _NetworkVisualization.pad(
                `${colors.blueNeon}${proportion} \u2500\u2500\u25B6${colors.reset}`,
                ARROW_WIDTH
              );
            } else {
              row += _NetworkVisualization.pad(
                `${colors.blueNeon}${ARROW}${colors.reset}`,
                ARROW_WIDTH
              );
            }
          }
        }
        if (rowIdx < OUTPUT_COUNT) {
          const node = outputDisplayNodes[rowIdx];
          const value = _NetworkVisualization.getNodeValue(node);
          row += _NetworkVisualization.pad(
            `${colors.orangeNeon}\u25B2${colors.reset}${_NetworkVisualization.fmtColoredValue(value)}`,
            columnWidth,
            " ",
            "left"
          ) + `${colors.blueCore}\u2551${colors.reset}`;
        } else {
          row += _NetworkVisualization.pad("", columnWidth);
        }
        rows.push(row);
      }
      return [
        header,
        ...rows,
        // Spacer row
        `${colors.blueCore}\u2551       ${_NetworkVisualization.pad(" ", 140)} \u2551${colors.reset}`,
        // Feed-forward flow explanation
        `${colors.blueCore}\u2551       ${_NetworkVisualization.pad(
          "Arrows indicate feed-forward flow.",
          140,
          " ",
          "left"
        )} ${colors.blueCore}\u2551${colors.reset}`,
        // Spacer row
        `${colors.blueCore}\u2551       ${_NetworkVisualization.pad(" ", 140)} \u2551${colors.reset}`,
        // Legend for node types
        `${colors.blueCore}\u2551       ${_NetworkVisualization.pad(
          `${colors.whiteNeon}Legend:  ${colors.neonGreen}\u25CF${colors.reset}=Input                    ${colors.cyanNeon}\u25A0${colors.reset}=Hidden                    ${colors.orangeNeon}\u25B2${colors.reset}=Output`,
          140,
          " ",
          "left"
        )} ${colors.blueCore}\u2551${colors.reset}`,
        // Legend for activation groups
        `${colors.blueCore}\u2551       ${_NetworkVisualization.pad(
          `${colors.whiteNeon}Groups:  ${colors.bgOrangeNeon}${colors.bright}v-high+${colors.reset}=Very high positive   ${colors.orangeNeon}high+${colors.reset}=High positive    ${colors.cyanNeon}mid+${colors.reset}=Medium positive    ${colors.neonGreen}low+${colors.reset}=Low positive`,
          140,
          " ",
          "left"
        )} ${colors.blueCore}\u2551${colors.reset}`,
        // Legend for near-zero group
        `${colors.blueCore}\u2551       ${_NetworkVisualization.pad(
          `${colors.whiteNeon}         zero\xB1${colors.reset}=Near zero`,
          140,
          " ",
          "left"
        )} ${colors.blueCore}\u2551${colors.reset}`,
        // Legend for negative groups
        `${colors.blueCore}\u2551       ${_NetworkVisualization.pad(
          `         ${colors.bgBlueCore}${colors.bright}v-high-${colors.reset}=Very high negative   ${colors.blueNeon}${colors.bright}high-${colors.reset}=High negative    ${colors.blueCore}mid-${colors.reset}=Medium negative    ${colors.blue}low-${colors.reset}=Low negative`,
          140,
          " ",
          "left"
        )} ${colors.blueCore}\u2551${colors.reset}`
      ].join("\n");
    }
  };

  // test/examples/asciiMaze/mazeVisualization.ts
  var MazeVisualization = class {
    /**
     * Renders a single maze cell with proper coloring based on its content and agent location.
     *
     * Applies appropriate colors and styling to each cell in the maze:
     * - Different colors for walls, open paths, start and exit positions
     * - Highlights the agent's current position
     * - Marks cells that are part of the agent's path
     * - Renders box drawing characters as walls with proper styling
     *
     * @param cell - The character representing the cell ('S', 'E', '#', '.' etc.)
     * @param x - X-coordinate of the cell
     * @param y - Y-coordinate of the cell
     * @param agentX - X-coordinate of the agent's current position
     * @param agentY - Y-coordinate of the agent's current position
     * @param path - Optional set of visited coordinates in "x,y" format
     * @returns Colorized string representing the cell
     */
    static renderCell(cell, x, y, agentX, agentY, path2) {
      const wallChars = /* @__PURE__ */ new Set([
        "#",
        "\u2550",
        "\u2551",
        "\u2554",
        "\u2557",
        "\u255A",
        "\u255D",
        "\u2560",
        "\u2563",
        "\u2566",
        "\u2569",
        "\u256C"
      ]);
      if (x === agentX && y === agentY) {
        if (cell === "S")
          return `${colors.bgBlack}${colors.orangeNeon}S${colors.reset}`;
        if (cell === "E")
          return `${colors.bgBlack}${colors.orangeNeon}E${colors.reset}`;
        return `${colors.bgBlack}${colors.orangeNeon}A${colors.reset}`;
      }
      switch (cell) {
        case "S":
          return `${colors.bgBlack}${colors.orangeNeon}S${colors.reset}`;
        // Start position
        case "E":
          return `${colors.bgBlack}${colors.orangeNeon}E${colors.reset}`;
        // Exit position - TRON orange
        case ".":
          if (path2 && path2.has(`${x},${y}`))
            return `${colors.floorBg}${colors.orangeNeon}\u2022${colors.reset}`;
          return `${colors.floorBg}${colors.gridLineText}.${colors.reset}`;
        // Open path - dark floor with subtle grid
        default:
          if (wallChars.has(cell)) {
            return `${colors.bgBlack}${colors.blueNeon}${cell}${colors.reset}`;
          }
          return cell;
      }
    }
    /**
     * Renders the entire maze as a colored ASCII string, showing the agent and its path.
     *
     * Converts the maze data structure into a human-readable, colorized representation showing:
     * - The maze layout with walls and open paths
     * - The start and exit positions
     * - The agent's current position
     * - The path the agent has taken (if provided)
     *
     * @param asciiMaze - Array of strings representing the maze layout
     * @param [agentX, agentY] - Current position of the agent
     * @param path - Optional array of positions representing the agent's path
     * @returns A multi-line string with the visualized maze
     */
    static visualizeMaze(asciiMaze, [agentX, agentY], path2) {
      const visitedPositions = path2 ? new Set(path2.map((pos) => `${pos[0]},${pos[1]}`)) : void 0;
      return asciiMaze.map(
        (row, y) => [...row].map(
          (cell, x) => this.renderCell(cell, x, y, agentX, agentY, visitedPositions)
        ).join("")
      ).join("\n");
    }
    /**
     * Prints a summary of the agent's attempt, including success, steps, and efficiency.
     *
     * Provides performance metrics about the agent's solution attempt:
     * - Whether it successfully reached the exit
     * - How many steps it took
     * - How efficient the path was compared to the optimal BFS distance
     *
     * @param currentBest - Object containing the simulation results, network, and generation
     * @param maze - Array of strings representing the maze layout
     * @param forceLog - Function used for logging output
     */
    static printMazeStats(currentBest, maze, forceLog) {
      const { result, generation } = currentBest;
      const successColor = result.success ? colors.cyanNeon : colors.neonRed;
      const startPos = MazeUtils.findPosition(maze, "S");
      const exitPos = MazeUtils.findPosition(maze, "E");
      const optimalLength = MazeUtils.bfsDistance(
        MazeUtils.encodeMaze(maze),
        startPos,
        exitPos
      );
      const FRAME_WIDTH = 148;
      const LEFT_PAD = 7;
      const RIGHT_PAD = 1;
      const CONTENT_WIDTH = FRAME_WIDTH - LEFT_PAD - RIGHT_PAD;
      forceLog(
        `${colors.blueCore}\u2551${NetworkVisualization.pad(" ", FRAME_WIDTH, " ")}${colors.blueCore}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${NetworkVisualization.pad(" ", FRAME_WIDTH, " ")}${colors.blueCore}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Success:${colors.neonIndigo} ${successColor}${result.success ? "YES" : "NO"}`,
          CONTENT_WIDTH,
          " ",
          "left"
        )}${" ".repeat(RIGHT_PAD)}${colors.blueCore}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Generation:${colors.neonIndigo} ${successColor}${generation}`,
          CONTENT_WIDTH,
          " ",
          "left"
        )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Fitness:${colors.neonOrange} ${result.fitness.toFixed(2)}`,
          CONTENT_WIDTH,
          " ",
          "left"
        )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Steps taken:${colors.neonIndigo} ${result.steps}`,
          CONTENT_WIDTH,
          " ",
          "left"
        )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Path length:${colors.neonIndigo} ${result.path.length}${colors.blueCore}`,
          CONTENT_WIDTH,
          " ",
          "left"
        )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Optimal distance to exit:${colors.neonYellow} ${optimalLength}`,
          CONTENT_WIDTH,
          " ",
          "left"
        )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}\u2551${NetworkVisualization.pad(" ", FRAME_WIDTH, " ")}${colors.blueCore}\u2551${colors.reset}`
      );
      if (result.success) {
        const pathLength = result.path.length - 1;
        const efficiency = Math.min(
          100,
          Math.round(optimalLength / pathLength * 100)
        ).toFixed(1);
        const overhead = (pathLength / optimalLength * 100 - 100).toFixed(1);
        const uniqueCells = /* @__PURE__ */ new Set();
        let revisitedCells = 0;
        let directionChanges = 0;
        let lastDirection = null;
        for (let i = 0; i < result.path.length; i++) {
          const [x, y] = result.path[i];
          const cellKey = `${x},${y}`;
          if (uniqueCells.has(cellKey)) {
            revisitedCells++;
          } else {
            uniqueCells.add(cellKey);
          }
          if (i > 0) {
            const [prevX, prevY] = result.path[i - 1];
            const dx = x - prevX;
            const dy = y - prevY;
            let currentDirection = "";
            if (dx > 0) currentDirection = "E";
            else if (dx < 0) currentDirection = "W";
            else if (dy > 0) currentDirection = "S";
            else if (dy < 0) currentDirection = "N";
            if (lastDirection !== null && currentDirection !== lastDirection) {
              directionChanges++;
            }
            lastDirection = currentDirection;
          }
        }
        const mazeWidth = maze[0].length;
        const mazeHeight = maze.length;
        const encodedMaze = MazeUtils.encodeMaze(maze);
        let walkableCells = 0;
        for (let y = 0; y < mazeHeight; y++) {
          for (let x = 0; x < mazeWidth; x++) {
            if (encodedMaze[y][x] !== -1) {
              walkableCells++;
            }
          }
        }
        const coveragePercent = (uniqueCells.size / walkableCells * 100).toFixed(1);
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Path efficiency:      ${colors.neonIndigo} ${optimalLength}/${pathLength} (${efficiency}%)`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Optimal steps:        ${colors.neonIndigo} ${optimalLength}`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Path overhead:        ${colors.neonIndigo} ${overhead}% longer than optimal`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Direction changes:    ${colors.neonIndigo} ${directionChanges}`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Unique cells visited: ${colors.neonIndigo} ${uniqueCells.size} (${coveragePercent}% of maze)`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Cells revisited:      ${colors.neonIndigo} ${revisitedCells} times`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Decisions per cell:   ${colors.neonIndigo} ${(directionChanges / uniqueCells.size).toFixed(2)}`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonOrange}Agent successfully navigated the maze!`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
      } else {
        const bestProgress = MazeUtils.calculateProgress(
          MazeUtils.encodeMaze(maze),
          result.path[result.path.length - 1],
          startPos,
          exitPos
        );
        const uniqueCells = /* @__PURE__ */ new Set();
        for (const [x, y] of result.path) {
          uniqueCells.add(`${x},${y}`);
        }
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Best progress toward exit:      ${colors.neonIndigo} ${bestProgress}%`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Shortest possible steps:        ${colors.neonIndigo} ${optimalLength}`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Unique cells visited:           ${colors.neonIndigo} ${uniqueCells.size}`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
        forceLog(
          `${colors.blueCore}\u2551${" ".repeat(LEFT_PAD)}${NetworkVisualization.pad(
            `${colors.neonSilver}Agent trying to reach the exit. ${colors.neonIndigo}`,
            CONTENT_WIDTH,
            " ",
            "left"
          )}${" ".repeat(RIGHT_PAD)}\u2551${colors.reset}`
        );
      }
    }
    /**
     * Displays a colored progress bar for agent progress.
     *
     * Creates a visual representation of the agent's progress toward the exit
     * as a horizontal bar with appropriate coloring based on percentage.
     *
     * @param progress - Progress percentage (0-100)
     * @param length - Length of the progress bar in characters (default: 60)
     * @returns A string containing the formatted progress bar
     */
    static displayProgressBar(progress, length = 60) {
      const filledLength = Math.max(
        0,
        Math.min(length, Math.floor(length * progress / 100))
      );
      const startChar = `${colors.blueCore}|>|`;
      const endChar = `${colors.blueCore}|<|`;
      const fillChar = `${colors.neonOrange}\u2550`;
      const emptyChar = `${colors.neonIndigo}:`;
      const pointerChar = `${colors.neonOrange}\u25B6`;
      let bar = "";
      bar += startChar;
      if (filledLength > 0) {
        bar += fillChar.repeat(filledLength - 1);
        bar += pointerChar;
      }
      const emptyLength = length - filledLength;
      if (emptyLength > 0) {
        bar += emptyChar.repeat(emptyLength);
      }
      bar += endChar;
      const color = progress < 30 ? colors.neonYellow : progress < 70 ? colors.orangeNeon : colors.cyanNeon;
      return `${color}${bar}${colors.reset} ${progress}%`;
    }
    /**
     * Formats elapsed time in a human-readable way.
     *
     * Converts seconds into appropriate units (seconds, minutes, hours)
     * for more intuitive display of time durations.
     *
     * @param seconds - Time in seconds
     * @returns Formatted string (e.g., "5.3s", "2m 30s", "1h 15m")
     */
    static formatElapsedTime(seconds) {
      if (seconds < 60) return `${seconds.toFixed(1)}s`;
      if (seconds < 3600) {
        const minutes2 = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes2}m ${remainingSeconds.toFixed(0)}s`;
      }
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor(seconds % 3600 / 60);
      return `${hours}h ${minutes}m`;
    }
  };

  // test/examples/asciiMaze/dashboardManager.ts
  var DashboardManager = class _DashboardManager {
    // List of solved maze records (keeps full maze + solution for archival display)
    solvedMazes = [];
    // Set of maze keys we've already archived to avoid duplicate entries
    solvedMazeKeys = /* @__PURE__ */ new Set();
    // Currently evolving/best candidate for the active maze (live view)
    currentBest = null;
    // Functions supplied by the embedding environment. Keep dashboard I/O pluggable.
    clearFunction;
    logFunction;
    archiveLogFunction;
    // Telemetry and small history windows used for rendering trends/sparklines
    _lastTelemetry = null;
    _lastBestFitness = null;
    _bestFitnessHistory = [];
    _complexityNodesHistory = [];
    _complexityConnsHistory = [];
    _hypervolumeHistory = [];
    _progressHistory = [];
    _speciesCountHistory = [];
    // Layout constants for the ASCII-art framed display
    static FRAME_INNER_WIDTH = 148;
    static LEFT_PADDING = 7;
    static RIGHT_PADDING = 1;
    static CONTENT_WIDTH = _DashboardManager.FRAME_INNER_WIDTH - _DashboardManager.LEFT_PADDING - _DashboardManager.RIGHT_PADDING;
    static STAT_LABEL_WIDTH = 28;
    static opennessLegend = "Openness: 1=best, (0,1)=longer improving, 0.001=only backtrack, 0=wall/dead/non-improving";
    /**
     * Construct a new DashboardManager
     *
     * @param clearFn - function that clears the "live" output area (no-op for archive)
     * @param logFn - function that accepts strings to render the live dashboard
     * @param archiveLogFn - optional function to which solved-maze archive blocks are appended
     */
    constructor(clearFn, logFn, archiveLogFn) {
      this.clearFunction = clearFn;
      this.logFunction = logFn;
      this.archiveLogFunction = archiveLogFn;
    }
    /**
     * formatStat
     *
     * Small helper that returns a prettified line containing a label and value
     * with color codes applied. The resulting string fits into the dashboard
     * content width and includes frame padding.
     */
    formatStat(label, value, colorLabel = colors.neonSilver, colorValue = colors.cyanNeon, labelWidth = _DashboardManager.STAT_LABEL_WIDTH) {
      const lbl = label.endsWith(":") ? label : label + ":";
      const paddedLabel = lbl.padEnd(labelWidth, " ");
      const composed = `${colorLabel}${paddedLabel}${colorValue} ${value}${colors.reset}`;
      return `${colors.blueCore}\u2551${" ".repeat(
        _DashboardManager.LEFT_PADDING
      )}${NetworkVisualization.pad(
        composed,
        _DashboardManager.CONTENT_WIDTH,
        " ",
        "left"
      )}${" ".repeat(_DashboardManager.RIGHT_PADDING)}${colors.blueCore}\u2551${colors.reset}`;
    }
    /**
     * buildSparkline
     *
     * Create a compact sparkline string (using block characters) from a numeric
     * series. The series is normalized to the block range and trimmed to the
     * requested width by taking the most recent values.
     */
    buildSparkline(data, width = 32) {
      if (!data || !data.length) return "";
      const blocks = ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"];
      const slice = data.slice(-width);
      const min = Math.min(...slice);
      const max = Math.max(...slice);
      const range = max - min || 1;
      return slice.map((v) => {
        const idx = Math.floor((v - min) / range * (blocks.length - 1));
        return blocks[idx];
      }).join("");
    }
    /**
     * getMazeKey
     *
     * Create a lightweight key for a maze (used to dedupe solved mazes).
     * The format is intentionally simple (concatenated rows) since the set
     * is only used for equality checks within a single run.
     */
    getMazeKey(maze) {
      return maze.join("");
    }
    /**
     * appendSolvedToArchive
     *
     * When a maze is solved for the first time, format and append a boxed
     * representation of the solved maze to the provided `archiveLogFunction`.
     * The block includes a header, optional small trend sparklines, the
     * centered maze drawing, and several efficiency stats derived from the path.
     *
     * This function is careful to be a no-op if no archive logger was provided
     * during construction.
     *
     * @param solved - record containing maze, solution and generation
     * @param displayNumber - 1-based ordinal for the solved maze in the archive
     */
    appendSolvedToArchive(solved, displayNumber) {
      if (!this.archiveLogFunction) return;
      const endPos = solved.result.path[solved.result.path.length - 1];
      const solvedMazeVisualization = MazeVisualization.visualizeMaze(
        solved.maze,
        endPos,
        solved.result.path
      );
      const solvedMazeLines = Array.isArray(solvedMazeVisualization) ? solvedMazeVisualization : solvedMazeVisualization.split("\n");
      const centeredSolvedMaze = solvedMazeLines.map(
        (line) => NetworkVisualization.pad(line, _DashboardManager.FRAME_INNER_WIDTH, " ")
      ).join("\n");
      const header = `${colors.blueCore}\u2560${NetworkVisualization.pad(
        "\u2550".repeat(_DashboardManager.FRAME_INNER_WIDTH),
        _DashboardManager.FRAME_INNER_WIDTH,
        "\u2550"
      )}\u2563${colors.reset}`;
      const title = `${colors.blueCore}\u2551${NetworkVisualization.pad(
        `${colors.orangeNeon} SOLVED #${displayNumber} (Gen ${solved.generation})${colors.reset}${colors.blueCore}`,
        _DashboardManager.FRAME_INNER_WIDTH,
        " "
      )}\u2551${colors.reset}`;
      const sep = `${colors.blueCore}\u2560${NetworkVisualization.pad(
        "\u2500".repeat(_DashboardManager.FRAME_INNER_WIDTH),
        _DashboardManager.FRAME_INNER_WIDTH,
        "\u2500"
      )}\u2563${colors.reset}`;
      const blockLines = [];
      blockLines.push(header);
      blockLines.push(title);
      blockLines.push(sep);
      const solvedLabelWidth = 22;
      const solvedStat = (label, value) => this.formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        solvedLabelWidth
      );
      const spark = this.buildSparkline(this._bestFitnessHistory, 64);
      const sparkComplexityNodes = this.buildSparkline(
        this._complexityNodesHistory,
        64
      );
      const sparkComplexityConns = this.buildSparkline(
        this._complexityConnsHistory,
        64
      );
      const sparkHyper = this.buildSparkline(this._hypervolumeHistory, 64);
      const sparkProgress = this.buildSparkline(this._progressHistory, 64);
      const sparkSpecies = this.buildSparkline(this._speciesCountHistory, 64);
      if (spark) blockLines.push(solvedStat("Fitness trend", spark));
      if (sparkComplexityNodes)
        blockLines.push(solvedStat("Nodes trend", sparkComplexityNodes));
      if (sparkComplexityConns)
        blockLines.push(solvedStat("Conns trend", sparkComplexityConns));
      if (sparkHyper) blockLines.push(solvedStat("Hypervol trend", sparkHyper));
      if (sparkProgress)
        blockLines.push(solvedStat("Progress trend", sparkProgress));
      if (sparkSpecies)
        blockLines.push(solvedStat("Species trend", sparkSpecies));
      blockLines.push(
        `${colors.blueCore}\u2551${NetworkVisualization.pad(
          " ",
          _DashboardManager.FRAME_INNER_WIDTH,
          " "
        )}${colors.blueCore}\u2551${colors.reset}`
      );
      centeredSolvedMaze.split("\n").forEach(
        (l) => blockLines.push(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            l,
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        )
      );
      const startPos = MazeUtils.findPosition(solved.maze, "S");
      const exitPos = MazeUtils.findPosition(solved.maze, "E");
      const optimalLength = MazeUtils.bfsDistance(
        MazeUtils.encodeMaze(solved.maze),
        startPos,
        exitPos
      );
      const pathLength = solved.result.path.length - 1;
      const efficiency = Math.min(
        100,
        Math.round(optimalLength / pathLength * 100)
      ).toFixed(1);
      const overhead = (pathLength / optimalLength * 100 - 100).toFixed(1);
      const uniqueCells = /* @__PURE__ */ new Set();
      let revisitedCells = 0;
      for (const [x, y] of solved.result.path) {
        const cellKey = `${x},${y}`;
        if (uniqueCells.has(cellKey)) revisitedCells++;
        else uniqueCells.add(cellKey);
      }
      blockLines.push(
        solvedStat(
          "Path efficiency",
          `${optimalLength}/${pathLength} (${efficiency}%)`
        )
      );
      blockLines.push(
        solvedStat("Path overhead", `${overhead}% longer than optimal`)
      );
      blockLines.push(solvedStat("Unique cells visited", `${uniqueCells.size}`));
      blockLines.push(solvedStat("Cells revisited", `${revisitedCells} times`));
      blockLines.push(solvedStat("Steps", `${solved.result.steps}`));
      blockLines.push(
        solvedStat("Fitness", `${solved.result.fitness.toFixed(2)}`)
      );
      blockLines.push(
        `${colors.blueCore}\u255A${NetworkVisualization.pad(
          "\u2550".repeat(_DashboardManager.FRAME_INNER_WIDTH),
          _DashboardManager.FRAME_INNER_WIDTH,
          "\u2550"
        )}\u255D${colors.reset}`
      );
      try {
        this.archiveLogFunction(blockLines.join("\n"), {
          prepend: true
        });
      } catch {
        const append = this.archiveLogFunction ?? (() => {
        });
        blockLines.forEach((ln) => append(ln));
      }
    }
    /**
     * update
     *
     * Called by the evolution engine to report the latest candidate solution
     * (or the current best). The dashboard will:
     * - update the currentBest reference used for the live view
     * - if the provided result is a successful solve and it's the first time
     *   we've seen this maze, append an archive block
     * - stash the latest telemetry values into small circular buffers for sparklines
     * - finally call `redraw` to update the live output
     */
    update(maze, result, network, generation, neatInstance) {
      this.currentBest = { result, network, generation };
      if (result.success) {
        const mazeKey = this.getMazeKey(maze);
        if (!this.solvedMazeKeys.has(mazeKey)) {
          this.solvedMazes.push({ maze, result, network, generation });
          this.solvedMazeKeys.add(mazeKey);
          const displayNumber = this.solvedMazes.length;
          this.appendSolvedToArchive(
            { maze, result, network, generation },
            displayNumber
          );
        }
      }
      const telemetry = neatInstance?.getTelemetry?.();
      if (telemetry && telemetry.length) {
        this._lastTelemetry = telemetry[telemetry.length - 1];
        const bestFit = this.currentBest?.result?.fitness;
        if (typeof bestFit === "number") {
          this._lastBestFitness = bestFit;
          this._bestFitnessHistory.push(bestFit);
          if (this._bestFitnessHistory.length > 500)
            this._bestFitnessHistory.shift();
        }
        const c = this._lastTelemetry?.complexity;
        if (c) {
          if (typeof c.meanNodes === "number") {
            this._complexityNodesHistory.push(c.meanNodes);
            if (this._complexityNodesHistory.length > 500)
              this._complexityNodesHistory.shift();
          }
          if (typeof c.meanConns === "number") {
            this._complexityConnsHistory.push(c.meanConns);
            if (this._complexityConnsHistory.length > 500)
              this._complexityConnsHistory.shift();
          }
        }
        const h = this._lastTelemetry?.hyper;
        if (typeof h === "number") {
          this._hypervolumeHistory.push(h);
          if (this._hypervolumeHistory.length > 500)
            this._hypervolumeHistory.shift();
        }
        const prog = this.currentBest?.result?.progress;
        if (typeof prog === "number") {
          this._progressHistory.push(prog);
          if (this._progressHistory.length > 500) this._progressHistory.shift();
        }
        const sc = this._lastTelemetry?.species;
        if (typeof sc === "number") {
          this._speciesCountHistory.push(sc);
          if (this._speciesCountHistory.length > 500)
            this._speciesCountHistory.shift();
        }
      }
      this.redraw(maze, neatInstance);
    }
    /**
     * redraw
     *
     * Responsible for clearing the live area and printing a compact snapshot of
     * the current best candidate, a short network summary, the maze drawing and
     * several telemetry-derived stats. The function uses `logFunction` for all
     * output lines so the same renderer can be used both in Node and in the
     * browser (DOM adapter).
     */
    redraw(currentMaze, neat) {
      this.clearFunction();
      this.logFunction(
        `${colors.blueCore}\u2554${NetworkVisualization.pad(
          "\u2550",
          _DashboardManager.FRAME_INNER_WIDTH,
          "\u2550"
        )}${colors.blueCore}\u2557${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}\u255A${NetworkVisualization.pad(
          "\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566",
          _DashboardManager.FRAME_INNER_WIDTH,
          "\u2550"
        )}${colors.blueCore}\u255D${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}${NetworkVisualization.pad(
          `\u2551 ${colors.neonYellow}ASCII maze${colors.blueCore} \u2551`,
          150,
          " "
        )}${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}\u2554${NetworkVisualization.pad(
          "\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569",
          _DashboardManager.FRAME_INNER_WIDTH,
          "\u2550"
        )}${colors.blueCore}\u2557${colors.reset}`
      );
      if (this.currentBest) {
        this.logFunction(
          `${colors.blueCore}\u2560${NetworkVisualization.pad(
            "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550",
            _DashboardManager.FRAME_INNER_WIDTH,
            "\u2550"
          )}${colors.blueCore}\u2563${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            `${colors.orangeNeon}EVOLVING (GEN ${this.currentBest.generation})`,
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}\u2560${NetworkVisualization.pad(
            "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550",
            _DashboardManager.FRAME_INNER_WIDTH,
            "\u2550"
          )}${colors.blueCore}\u2563${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        this.logFunction(
          NetworkVisualization.visualizeNetworkSummary(this.currentBest.network)
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        const lastPos = this.currentBest.result.path[this.currentBest.result.path.length - 1];
        const currentMazeVisualization = MazeVisualization.visualizeMaze(
          currentMaze,
          lastPos,
          this.currentBest.result.path
        );
        const currentMazeLines = Array.isArray(currentMazeVisualization) ? currentMazeVisualization : currentMazeVisualization.split("\n");
        const centeredCurrentMaze = currentMazeLines.map(
          (line) => `${colors.blueCore}\u2551${NetworkVisualization.pad(
            line,
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551`
        ).join("\n");
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        this.logFunction(centeredCurrentMaze);
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        MazeVisualization.printMazeStats(
          this.currentBest,
          currentMaze,
          this.logFunction
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
        this.logFunction(
          (() => {
            const bar = `Progress to exit: ${MazeVisualization.displayProgressBar(
              this.currentBest.result.progress
            )}`;
            return `${colors.blueCore}\u2551${NetworkVisualization.pad(
              " " + colors.neonSilver + bar + colors.reset,
              _DashboardManager.FRAME_INNER_WIDTH,
              " "
            )}${colors.blueCore}\u2551${colors.reset}`;
          })()
        );
        this.logFunction(
          `${colors.blueCore}\u2551${NetworkVisualization.pad(
            " ",
            _DashboardManager.FRAME_INNER_WIDTH,
            " "
          )}${colors.blueCore}\u2551${colors.reset}`
        );
      }
      const last = this._lastTelemetry;
      const complexity = last?.complexity;
      const perf = last?.perf;
      const lineage = last?.lineage;
      const fronts = Array.isArray(last?.fronts) ? last.fronts : null;
      const objectives = last?.objectives;
      const hyper = last?.hyper;
      const diversity = last?.diversity;
      const mutationStats = last?.mutationStats || last?.mutation?.stats;
      const bestFitness = this.currentBest?.result?.fitness;
      const fmtNum = (v, digits = 2) => typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "-";
      const deltaArrow = (curr, prev) => {
        if (curr == null || prev == null) return "";
        const diff = curr - prev;
        if (Math.abs(diff) < 1e-9) return `${colors.neonSilver} (\u21940)`;
        const color = diff > 0 ? colors.cyanNeon : colors.neonRed;
        const arrow = diff > 0 ? "\u2191" : "\u2193";
        return `${color} (${arrow}${diff.toFixed(2)})${colors.neonSilver}`;
      };
      let popMean = "-";
      let popMedian = "-";
      let speciesCount = "-";
      let enabledRatio = "-";
      if (neat && Array.isArray(neat.population)) {
        const scores = [];
        let enabled = 0, total = 0;
        neat.population.forEach((g) => {
          if (typeof g.score === "number") scores.push(g.score);
          if (Array.isArray(g.connections)) {
            g.connections.forEach((c) => {
              total++;
              if (c.enabled !== false) enabled++;
            });
          }
        });
        if (scores.length) {
          const sum = scores.reduce((a, b) => a + b, 0);
          popMean = (sum / scores.length).toFixed(2);
          const sorted = scores.slice().sort((a, b) => a - b);
          const mid = Math.floor(sorted.length / 2);
          popMedian = (sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]).toFixed(2);
        }
        if (total) enabledRatio = (enabled / total).toFixed(2);
        speciesCount = Array.isArray(neat.species) ? neat.species.length.toString() : speciesCount;
      }
      const firstFrontSize = fronts?.[0]?.length || 0;
      const SPARK_WIDTH = 64;
      const spark = this.buildSparkline(this._bestFitnessHistory, SPARK_WIDTH);
      const sparkComplexityNodes = this.buildSparkline(
        this._complexityNodesHistory,
        SPARK_WIDTH
      );
      const sparkComplexityConns = this.buildSparkline(
        this._complexityConnsHistory,
        SPARK_WIDTH
      );
      const sparkHyper = this.buildSparkline(
        this._hypervolumeHistory,
        SPARK_WIDTH
      );
      const sparkProgress = this.buildSparkline(
        this._progressHistory,
        SPARK_WIDTH
      );
      const sparkSpecies = this.buildSparkline(
        this._speciesCountHistory,
        SPARK_WIDTH
      );
      const statsLines = [];
      statsLines.push(
        this.formatStat(
          "Current generation",
          `${this.currentBest?.generation || 0}`
        )
      );
      if (typeof bestFitness === "number")
        statsLines.push(
          this.formatStat(
            "Best fitness",
            `${bestFitness.toFixed(2)}${deltaArrow(
              bestFitness,
              this._bestFitnessHistory.length > 1 ? this._bestFitnessHistory[this._bestFitnessHistory.length - 2] : null
            )}`
          )
        );
      const satFrac = this.currentBest?.result?.saturationFraction;
      if (typeof satFrac === "number")
        statsLines.push(
          this.formatStat("Saturation fraction", satFrac.toFixed(3))
        );
      const actEnt = this.currentBest?.result?.actionEntropy;
      if (typeof actEnt === "number")
        statsLines.push(
          this.formatStat("Action entropy (path)", actEnt.toFixed(3))
        );
      if (popMean === "-" && typeof bestFitness === "number")
        popMean = bestFitness.toFixed(2);
      if (popMedian === "-" && typeof bestFitness === "number")
        popMedian = bestFitness.toFixed(2);
      statsLines.push(this.formatStat("Population mean", popMean));
      statsLines.push(this.formatStat("Population median", popMedian));
      if (complexity)
        statsLines.push(
          this.formatStat(
            "Complexity mean n/c",
            `${fmtNum(complexity.meanNodes, 2)}/${fmtNum(
              complexity.meanConns,
              2
            )}  max ${fmtNum(complexity.maxNodes, 0)}/${fmtNum(
              complexity.maxConns,
              0
            )}`,
            colors.neonSilver,
            colors.orangeNeon
          )
        );
      if (complexity && (complexity.growthNodes < 0 || complexity.growthConns < 0))
        statsLines.push(
          this.formatStat(
            "Simplify phase",
            "active",
            colors.neonSilver,
            colors.neonGreen
          )
        );
      if (sparkComplexityNodes)
        statsLines.push(
          this.formatStat(
            "Nodes trend",
            sparkComplexityNodes,
            colors.neonSilver,
            colors.neonYellow
          )
        );
      if (sparkComplexityConns)
        statsLines.push(
          this.formatStat(
            "Conns trend",
            sparkComplexityConns,
            colors.neonSilver,
            colors.neonYellow
          )
        );
      statsLines.push(this.formatStat("Enabled conn ratio", enabledRatio));
      if (perf && (perf.evalMs != null || perf.evolveMs != null))
        statsLines.push(
          this.formatStat(
            "Perf eval/evolve ms",
            `${fmtNum(perf.evalMs, 1)}/${fmtNum(perf.evolveMs, 1)}`
          )
        );
      if (lineage)
        statsLines.push(
          this.formatStat(
            "Lineage depth b/mean",
            `${lineage.depthBest}/${fmtNum(lineage.meanDepth, 2)}`
          )
        );
      if (lineage?.inbreeding != null)
        statsLines.push(
          this.formatStat("Inbreeding", fmtNum(lineage.inbreeding, 3))
        );
      if (speciesCount === "-" && typeof last?.species === "number")
        speciesCount = String(last.species);
      statsLines.push(this.formatStat("Species count", speciesCount));
      if (diversity?.structuralVar != null)
        statsLines.push(
          this.formatStat(
            "Structural variance",
            fmtNum(diversity.structuralVar, 3)
          )
        );
      if (diversity?.objectiveSpread != null)
        statsLines.push(
          this.formatStat(
            "Objective spread",
            fmtNum(diversity.objectiveSpread, 3)
          )
        );
      if (Array.isArray(neat?.species) && neat.species.length) {
        const sizes = neat.species.map((s) => s.members?.length || 0).sort((a, b) => b - a);
        const top3 = sizes.slice(0, 3).join("/") || "-";
        statsLines.push(this.formatStat("Top species sizes", top3));
      }
      if (fronts)
        statsLines.push(
          this.formatStat(
            "Pareto fronts",
            `${fronts.map((f) => f?.length || 0).join("/")}`
          )
        );
      statsLines.push(
        this.formatStat("First front size", firstFrontSize.toString())
      );
      if (objectives)
        statsLines.push(
          this.formatStat(
            "Objectives",
            objectives.join(", "),
            colors.neonSilver,
            colors.neonIndigo
          )
        );
      if (hyper !== void 0)
        statsLines.push(this.formatStat("Hypervolume", fmtNum(hyper, 4)));
      if (sparkHyper)
        statsLines.push(
          this.formatStat(
            "Hypervolume trend",
            sparkHyper,
            colors.neonSilver,
            colors.neonGreen
          )
        );
      if (spark)
        statsLines.push(
          this.formatStat(
            "Fitness trend",
            spark,
            colors.neonSilver,
            colors.neonYellow
          )
        );
      if (sparkProgress)
        statsLines.push(
          this.formatStat(
            "Progress trend",
            sparkProgress,
            colors.neonSilver,
            colors.cyanNeon
          )
        );
      if (sparkSpecies)
        statsLines.push(
          this.formatStat(
            "Species trend",
            sparkSpecies,
            colors.neonSilver,
            colors.neonIndigo
          )
        );
      if (neat?.getNoveltyArchiveSize) {
        try {
          const nov = neat.getNoveltyArchiveSize();
          statsLines.push(this.formatStat("Novelty archive", `${nov}`));
        } catch {
        }
      }
      if (neat?.getOperatorStats) {
        try {
          const ops = neat.getOperatorStats();
          if (Array.isArray(ops) && ops.length) {
            const top = ops.slice().sort(
              (a, b) => b.success / Math.max(1, b.attempts) - a.success / Math.max(1, a.attempts)
            ).slice(0, 4).map(
              (o) => `${o.name}:${(100 * o.success / Math.max(1, o.attempts)).toFixed(0)}%`
            ).join(" ");
            if (top)
              statsLines.push(
                this.formatStat(
                  "Op acceptance",
                  top,
                  colors.neonSilver,
                  colors.neonGreen
                )
              );
          }
        } catch {
        }
      }
      if (mutationStats && typeof mutationStats === "object") {
        const entries = Object.entries(mutationStats).filter(([k, v]) => typeof v === "number").sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, v]) => `${k}:${v.toFixed(0)}`).join(" ");
        if (entries)
          statsLines.push(
            this.formatStat(
              "Top mutations",
              entries,
              colors.neonSilver,
              colors.neonGreen
            )
          );
      }
      statsLines.forEach((ln) => this.logFunction(ln));
      this.logFunction(
        `${colors.blueCore}\u2551${NetworkVisualization.pad(
          " ",
          _DashboardManager.FRAME_INNER_WIDTH,
          " "
        )}${colors.blueCore}\u2551${colors.reset}`
      );
    }
    reset() {
      this.solvedMazes = [];
      this.solvedMazeKeys.clear();
      this.currentBest = null;
    }
  };

  // src/neataptic.ts
  init_neat();
  init_network();
  init_node();
  init_layer();
  init_group();
  init_connection();

  // src/architecture/architect.ts
  init_node();
  init_layer();
  init_group();
  init_network();
  init_methods();
  init_connection();

  // src/neataptic.ts
  init_methods();
  init_config();
  init_multi();

  // test/examples/asciiMaze/mazeVision.ts
  var MazeVision = class _MazeVision {
    /**
     * Constructs the 6-dimensional input vector for the neural network based on the agent's current state.
     *
     * @param encodedMaze - The 2D numerical representation of the maze.
     * @param position - The agent's current `[x, y]` coordinates.
     * @param exitPos - The coordinates of the maze exit.
     * @param distanceMap - A pre-calculated map of distances from each cell to the exit.
     * @param prevDistance - The agent's distance to the exit from the previous step.
     * @param currentDistance - The agent's current distance to the exit.
     * @param prevAction - The last action taken by the agent (0:N, 1:E, 2:S, 3:W).
     * @returns A 6-element array of numbers representing the network inputs.
     */
    static buildInputs6(encodedMaze, agentPosition, exitPosition, distanceToExitMap, previousStepDistance, currentStepDistance, previousAction) {
      const [agentX, agentY] = agentPosition;
      const mazeHeight = encodedMaze.length;
      const mazeWidth = encodedMaze[0].length;
      const isWithinBounds = (col, row) => row >= 0 && row < mazeHeight && col >= 0 && col < mazeWidth;
      const isCellOpen = (col, row) => isWithinBounds(col, row) && encodedMaze[row][col] !== -1;
      const opennessHorizon = 1e3;
      const compassHorizon = 5e3;
      const neighborCells = [];
      const DIRECTION_VECTORS = [
        [0, -1, 0],
        // North
        [1, 0, 1],
        // East
        [0, 1, 2],
        // South
        [-1, 0, 3]
        // West
      ];
      const currentCellDistanceToExit = distanceToExitMap && Number.isFinite(distanceToExitMap[agentY]?.[agentX]) ? distanceToExitMap[agentY][agentX] : void 0;
      for (const [dx, dy, directionIndex] of DIRECTION_VECTORS) {
        const neighborX = agentX + dx;
        const neighborY = agentY + dy;
        if (!isCellOpen(neighborX, neighborY)) {
          neighborCells.push({
            directionIndex,
            neighborX,
            neighborY,
            pathLength: Infinity,
            isReachable: false,
            opennessValue: 0
          });
          continue;
        }
        const neighborDistanceToExit = distanceToExitMap ? distanceToExitMap[neighborY]?.[neighborX] : void 0;
        if (neighborDistanceToExit != null && Number.isFinite(neighborDistanceToExit) && currentCellDistanceToExit != null && Number.isFinite(currentCellDistanceToExit)) {
          if (neighborDistanceToExit < currentCellDistanceToExit) {
            const pathLength = 1 + neighborDistanceToExit;
            if (pathLength <= opennessHorizon)
              neighborCells.push({
                directionIndex,
                neighborX,
                neighborY,
                pathLength,
                isReachable: true,
                opennessValue: 0
              });
            else
              neighborCells.push({
                directionIndex,
                neighborX,
                neighborY,
                pathLength: Infinity,
                isReachable: true,
                opennessValue: 0
              });
          } else {
            neighborCells.push({
              directionIndex,
              neighborX,
              neighborY,
              pathLength: Infinity,
              isReachable: true,
              opennessValue: 0
            });
          }
        } else {
          neighborCells.push({
            directionIndex,
            neighborX,
            neighborY,
            pathLength: Infinity,
            isReachable: true,
            opennessValue: 0
          });
        }
      }
      const reachableNeighbors = neighborCells.filter(
        (neighbor) => neighbor.isReachable && Number.isFinite(neighbor.pathLength)
      );
      let minPathLength = Infinity;
      for (const neighbor of reachableNeighbors)
        if (neighbor.pathLength < minPathLength)
          minPathLength = neighbor.pathLength;
      if (reachableNeighbors.length && minPathLength < Infinity) {
        for (const neighbor of reachableNeighbors) {
          if (neighbor.pathLength === minPathLength) neighbor.opennessValue = 1;
          else neighbor.opennessValue = minPathLength / neighbor.pathLength;
        }
      }
      let opennessNorth = neighborCells.find((n) => n.directionIndex === 0).opennessValue;
      let opennessEast = neighborCells.find((n) => n.directionIndex === 1).opennessValue;
      let opennessSouth = neighborCells.find((n) => n.directionIndex === 2).opennessValue;
      let opennessWest = neighborCells.find((n) => n.directionIndex === 3).opennessValue;
      if (opennessNorth === 0 && opennessEast === 0 && opennessSouth === 0 && opennessWest === 0 && previousAction != null && previousAction >= 0) {
        const oppositeDirection = (previousAction + 2) % 4;
        switch (oppositeDirection) {
          case 0:
            if (isCellOpen(agentX, agentY - 1)) opennessNorth = 1e-3;
            break;
          case 1:
            if (isCellOpen(agentX + 1, agentY)) opennessEast = 1e-3;
            break;
          case 2:
            if (isCellOpen(agentX, agentY + 1)) opennessSouth = 1e-3;
            break;
          case 3:
            if (isCellOpen(agentX - 1, agentY)) opennessWest = 1e-3;
            break;
        }
      }
      let bestDirectionToExit = 0;
      if (distanceToExitMap) {
        let minCompassPathLength = Infinity;
        let foundCompassPath = false;
        for (const neighbor of neighborCells) {
          const neighborRawDistance = distanceToExitMap[neighbor.neighborY]?.[neighbor.neighborX];
          if (neighborRawDistance != null && Number.isFinite(neighborRawDistance)) {
            const pathLength = neighborRawDistance + 1;
            if (pathLength < minCompassPathLength && pathLength <= compassHorizon) {
              minCompassPathLength = pathLength;
              bestDirectionToExit = neighbor.directionIndex;
              foundCompassPath = true;
            }
          }
        }
        if (!foundCompassPath) {
          const deltaXToGoal = exitPosition[0] - agentX;
          const deltaYToGoal = exitPosition[1] - agentY;
          if (Math.abs(deltaXToGoal) > Math.abs(deltaYToGoal))
            bestDirectionToExit = deltaXToGoal > 0 ? 1 : 3;
          else bestDirectionToExit = deltaYToGoal > 0 ? 2 : 0;
        }
      } else {
        const deltaXToGoal = exitPosition[0] - agentX;
        const deltaYToGoal = exitPosition[1] - agentY;
        if (Math.abs(deltaXToGoal) > Math.abs(deltaYToGoal))
          bestDirectionToExit = deltaXToGoal > 0 ? 1 : 3;
        else bestDirectionToExit = deltaYToGoal > 0 ? 2 : 0;
      }
      const compassScalar = bestDirectionToExit * 0.25;
      let progressDelta = 0.5;
      if (previousStepDistance != null && Number.isFinite(previousStepDistance)) {
        const distanceDelta = previousStepDistance - currentStepDistance;
        const clippedDelta = Math.max(-2, Math.min(2, distanceDelta));
        progressDelta = 0.5 + clippedDelta / 4;
      }
      const inputVector = [
        compassScalar,
        opennessNorth,
        opennessEast,
        opennessSouth,
        opennessWest,
        progressDelta
      ];
      if (typeof process !== "undefined" && typeof process.env !== "undefined" && process.env.ASCII_VISION_DEBUG === "1") {
        try {
          const neighborSummary = neighborCells.map(
            (neighbor) => `{dir:${neighbor.directionIndex} x:${neighbor.neighborX} y:${neighbor.neighborY} path:${Number.isFinite(neighbor.pathLength) ? neighbor.pathLength.toFixed(2) : "Inf"} open:${neighbor.opennessValue.toFixed(4)}}`
          ).join(" ");
          _MazeVision._dbgCounter = (_MazeVision._dbgCounter || 0) + 1;
          if (_MazeVision._dbgCounter % 5 === 0) {
            console.log(
              `[VISION] pos=${agentX},${agentY} comp=${compassScalar.toFixed(
                2
              )} inputs=${JSON.stringify(
                inputVector.map((v) => +v.toFixed(6))
              )} neighbors=${neighborSummary}`
            );
          }
        } catch {
        }
      }
      return inputVector;
    }
  };

  // test/examples/asciiMaze/mazeMovement.ts
  var MazeMovement = class _MazeMovement {
    /**
     * Checks if a move is valid (within bounds and not a wall).
     *
     * @param encodedMaze - 2D array representation of the maze.
     * @param [x, y] - Coordinates to check.
     * @returns Boolean indicating if the position is valid for movement.
     */
    /**
     * Checks if a move is valid (within maze bounds and not a wall cell).
     *
     * @param encodedMaze - 2D array representation of the maze (cells: -1=wall, 0+=open).
     * @param coords - [x, y] coordinates to check for validity.
     * @returns {boolean} True if the position is within bounds and not a wall.
     */
    static isValidMove(encodedMaze, [x, y]) {
      return x >= 0 && y >= 0 && y < encodedMaze.length && x < encodedMaze[0].length && encodedMaze[y][x] !== -1;
    }
    /**
     * Moves the agent in the given direction if possible, otherwise stays in place.
     *
     * Handles collision detection with walls and maze boundaries,
     * preventing the agent from making invalid moves.
     *
     * @param encodedMaze - 2D array representation of the maze.
     * @param position - Current [x,y] position of the agent.
     * @param direction - Direction index (0=North, 1=East, 2=South, 3=West).
     * @returns New position after movement, or original position if move was invalid.
     */
    /**
     * Moves the agent in the specified direction if the move is valid.
     *
     * Handles collision detection with walls and maze boundaries,
     * preventing the agent from making invalid moves.
     *
     * @param encodedMaze - 2D array representation of the maze.
     * @param position - Current [x, y] position of the agent.
     * @param direction - Direction index (0=North, 1=East, 2=South, 3=West, -1=No move).
     * @returns { [number, number] } New position after movement, or original position if move was invalid.
     */
    static moveAgent(encodedMaze, position, direction) {
      if (direction === -1) {
        return [...position];
      }
      const nextPosition = [...position];
      switch (direction) {
        case 0:
          nextPosition[1] -= 1;
          break;
        case 1:
          nextPosition[0] += 1;
          break;
        case 2:
          nextPosition[1] += 1;
          break;
        case 3:
          nextPosition[0] -= 1;
          break;
      }
      if (_MazeMovement.isValidMove(encodedMaze, nextPosition)) {
        return nextPosition;
      } else {
        return position;
      }
    }
    /**
     * Selects the direction with the highest output value from the neural network.
     * Applies softmax to interpret outputs as probabilities, then uses argmax.
     *
     * @param outputs - Array of output values from the neural network (length 4).
     * @returns Index of the highest output value (0=N, 1=E, 2=S, 3=W), or -1 for no movement.
     */
    /**
     * Selects the direction with the highest output value from the neural network.
     * Applies softmax to interpret outputs as probabilities, then uses argmax.
     * Also computes entropy and confidence statistics for analysis.
     *
     * @param outputs - Array of output values from the neural network (length 4).
     * @returns {object} Direction index, softmax probabilities, entropy, and confidence stats.
     */
    static selectDirection(outputs) {
      if (!outputs || outputs.length !== 4) {
        return {
          direction: -1,
          softmax: [0, 0, 0, 0],
          entropy: 0,
          maxProb: 0,
          secondProb: 0
        };
      }
      const mean = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4;
      let variance = 0;
      for (const o of outputs) variance += (o - mean) * (o - mean);
      variance /= 4;
      let std = Math.sqrt(variance);
      if (!Number.isFinite(std) || std < 1e-6) std = 1e-6;
      const centered = outputs.map((o) => o - mean);
      const collapseRatio = std < 0.01 ? 1 : std < 0.03 ? 0.5 : 0;
      const temperature = 1 + 1.2 * collapseRatio;
      const max = Math.max(...centered);
      const exps = centered.map((v) => Math.exp((v - max) / temperature));
      const sum = exps.reduce((a, b) => a + b, 0) || 1;
      const softmax = exps.map((e) => e / sum);
      let direction = 0;
      let maxProb = -Infinity;
      let secondProb = 0;
      softmax.forEach((p, i) => {
        if (p > maxProb) {
          secondProb = maxProb;
          maxProb = p;
          direction = i;
        } else if (p > secondProb) secondProb = p;
      });
      let entropy = 0;
      softmax.forEach((p) => {
        if (p > 0) entropy += -p * Math.log(p);
      });
      entropy /= Math.log(4);
      return { direction, softmax, entropy, maxProb, secondProb };
    }
    /**
     * Simulates the agent navigating the maze using its neural network.
     *
     * Runs a complete simulation of an agent traversing a maze,
     * using its neural network for decision making. This implementation focuses
     * on a minimalist approach, putting more responsibility on the neural network.
     *
     * @param network - Neural network controlling the agent.
     * @param encodedMaze - 2D array representation of the maze.
     * @param startPos - Starting position [x,y] of the agent.
     * @param exitPos - Exit/goal position [x,y] of the maze.
     * @param maxSteps - Maximum steps allowed before terminating (default 3000).
     * @returns Object containing:
     *   - success: Boolean indicating if exit was reached.
     *   - steps: Number of steps taken.
     *   - path: Array of positions visited.
     *   - fitness: Calculated fitness score for evolution.
     *   - progress: Percentage progress toward exit (0-100).
     */
    static simulateAgent(network, encodedMaze, startPos, exitPos, distanceMap, maxSteps = 3e3) {
      let position = [...startPos];
      let steps = 0;
      let path2 = [position.slice()];
      let visitedPositions = /* @__PURE__ */ new Set();
      let visitCounts = /* @__PURE__ */ new Map();
      let moveHistory = [];
      const MOVE_HISTORY_LENGTH = 6;
      let minDistanceToExit = distanceMap ? distanceMap[position[1]]?.[position[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
      const rewardScale = 0.5;
      let progressReward = 0;
      let newCellExplorationBonus = 0;
      let invalidMovePenalty = 0;
      let prevAction = -1;
      let stepsSinceImprovement = 0;
      const startDistanceGlobal = distanceMap ? distanceMap[position[1]]?.[position[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
      let lastDistanceGlobal = startDistanceGlobal;
      let saturatedSteps = 0;
      const LOCAL_WINDOW = 30;
      const recentPositions = [];
      let localAreaPenalty = 0;
      let lastProgressRatio = 0;
      while (steps < maxSteps) {
        steps++;
        const currentPosKey = `${position[0]},${position[1]}`;
        visitedPositions.add(currentPosKey);
        visitCounts.set(currentPosKey, (visitCounts.get(currentPosKey) || 0) + 1);
        moveHistory.push(currentPosKey);
        if (moveHistory.length > MOVE_HISTORY_LENGTH) moveHistory.shift();
        const percentExplored = visitedPositions.size / (encodedMaze.length * encodedMaze[0].length);
        let loopPenalty = 0;
        if (moveHistory.length >= 4 && moveHistory[moveHistory.length - 1] === moveHistory[moveHistory.length - 3] && moveHistory[moveHistory.length - 2] === moveHistory[moveHistory.length - 4]) {
          loopPenalty -= 10 * rewardScale;
        }
        const loopFlag = loopPenalty < 0 ? 1 : 0;
        let memoryPenalty = 0;
        if (moveHistory.length > 1 && moveHistory.slice(0, -1).includes(currentPosKey)) {
          memoryPenalty -= 2 * rewardScale;
        }
        let revisitPenalty = 0;
        const visits = visitCounts.get(currentPosKey) || 1;
        if (visits > 1) {
          revisitPenalty -= 0.2 * (visits - 1) * rewardScale;
        }
        if (visits > 10) {
          invalidMovePenalty -= 1e3 * rewardScale;
          break;
        }
        const prevDistLocal = distanceMap ? distanceMap[position[1]]?.[position[0]] ?? void 0 : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
        const distCurrentLocal = prevDistLocal;
        const vision = MazeVision.buildInputs6(
          encodedMaze,
          position,
          exitPos,
          distanceMap,
          _MazeMovement._prevDistanceStep,
          distCurrentLocal,
          prevAction
        );
        _MazeMovement._prevDistanceStep = distCurrentLocal;
        const distHere = distanceMap ? distanceMap[position[1]]?.[position[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
        let direction;
        let actionStats = null;
        try {
          const outputs = network.activate(vision);
          network._lastStepOutputs = network._lastStepOutputs || [];
          const _ls = network._lastStepOutputs;
          _ls.push(outputs.slice());
          if (_ls.length > 80) _ls.shift();
          actionStats = _MazeMovement.selectDirection(outputs);
          _MazeMovement._saturations = _MazeMovement._saturations || 0;
          const overConfident = actionStats.maxProb > 0.985 && actionStats.secondProb < 0.01;
          const logitsMean = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4;
          let logVar = 0;
          for (const o of outputs) logVar += Math.pow(o - logitsMean, 2);
          logVar /= 4;
          const logStd = Math.sqrt(logVar);
          const flatCollapsed = logStd < 0.01;
          const saturatedNow = overConfident || flatCollapsed;
          if (saturatedNow) {
            _MazeMovement._saturations++;
            saturatedSteps++;
          } else {
            _MazeMovement._saturations = Math.max(
              0,
              _MazeMovement._saturations - 1
            );
          }
          if (overConfident) invalidMovePenalty -= 0.25 * rewardScale;
          if (flatCollapsed) invalidMovePenalty -= 0.35 * rewardScale;
          try {
            if (_MazeMovement._saturations > 6 && steps % 5 === 0) {
              const outs = network.nodes?.filter(
                (n) => n.type === "output"
              );
              if (outs?.length) {
                const mean = outs.reduce((a, n) => a + n.bias, 0) / outs.length;
                outs.forEach((n) => {
                  n.bias = Math.max(-5, Math.min(5, n.bias - mean * 0.5));
                });
              }
            }
          } catch {
          }
          direction = actionStats.direction;
        } catch (error) {
          console.error("Error activating network:", error);
          direction = -1;
        }
        if (distHere <= 2) {
          let bestDir = direction;
          let bestDist = Infinity;
          for (let d = 0; d < 4; d++) {
            const testPos = _MazeMovement.moveAgent(encodedMaze, position, d);
            if (testPos[0] === position[0] && testPos[1] === position[1])
              continue;
            const dVal = distanceMap ? distanceMap[testPos[1]]?.[testPos[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, testPos, exitPos);
            if (dVal < bestDist) {
              bestDist = dVal;
              bestDir = d;
            }
          }
          if (bestDir != null) direction = bestDir;
        }
        const stepsStagnant = stepsSinceImprovement;
        let epsilon = 0;
        if (steps < 10) epsilon = 0.35;
        else if (stepsStagnant > 12) epsilon = 0.5;
        else if (stepsStagnant > 6) epsilon = 0.25;
        else if (_MazeMovement._saturations > 3) epsilon = 0.3;
        if (distHere <= 5) epsilon = Math.min(epsilon, 0.05);
        if (Math.random() < epsilon) {
          const candidates = [0, 1, 2, 3].filter((d) => d !== prevAction);
          while (candidates.length) {
            const idx = Math.floor(Math.random() * candidates.length);
            const cand = candidates.splice(idx, 1)[0];
            const testPos = _MazeMovement.moveAgent(encodedMaze, position, cand);
            if (testPos[0] !== position[0] || testPos[1] !== position[1]) {
              direction = cand;
              break;
            }
          }
        }
        _MazeMovement._noMoveStreak = _MazeMovement._noMoveStreak || 0;
        if (direction === -1) _MazeMovement._noMoveStreak++;
        if (_MazeMovement._noMoveStreak >= 5) {
          for (let tries = 0; tries < 4; tries++) {
            const cand = Math.floor(Math.random() * 4);
            const testPos = _MazeMovement.moveAgent(encodedMaze, position, cand);
            if (testPos[0] !== position[0] || testPos[1] !== position[1]) {
              direction = cand;
              break;
            }
          }
          _MazeMovement._noMoveStreak = 0;
        }
        const prevPosition = [...position];
        const prevDistance = distanceMap ? distanceMap[position[1]]?.[position[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
        position = _MazeMovement.moveAgent(encodedMaze, position, direction);
        const moved = prevPosition[0] !== position[0] || prevPosition[1] !== position[1];
        if (moved) {
          path2.push(position.slice());
          recentPositions.push(position.slice());
          if (recentPositions.length > LOCAL_WINDOW) recentPositions.shift();
          if (recentPositions.length === LOCAL_WINDOW) {
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (const [rx, ry] of recentPositions) {
              if (rx < minX) minX = rx;
              if (rx > maxX) maxX = rx;
              if (ry < minY) minY = ry;
              if (ry > maxY) maxY = ry;
            }
            const span = maxX - minX + (maxY - minY);
            if (span <= 5 && stepsSinceImprovement > 8) {
              localAreaPenalty -= 0.05 * rewardScale;
            }
          }
          const currentDistance = distanceMap ? distanceMap[position[1]]?.[position[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
          const distanceDelta = prevDistance - currentDistance;
          if (distanceDelta > 0) {
            const conf = actionStats?.maxProb ?? 1;
            progressReward += (0.3 + 0.7 * conf) * rewardScale;
            if (stepsSinceImprovement > 0)
              progressReward += Math.min(
                stepsSinceImprovement * 0.02 * rewardScale,
                0.5 * rewardScale
              );
            stepsSinceImprovement = 0;
            progressReward += distanceDelta * 2 * (0.4 + 0.6 * conf);
          } else if (currentDistance > prevDistance) {
            const conf = actionStats?.maxProb ?? 0.5;
            progressReward -= (0.05 + 0.15 * conf) * rewardScale;
            stepsSinceImprovement++;
          } else {
            stepsSinceImprovement++;
          }
          if (visits === 1) {
            newCellExplorationBonus += 0.3 * rewardScale;
          } else {
            newCellExplorationBonus -= 0.5 * rewardScale;
          }
          minDistanceToExit = Math.min(minDistanceToExit, currentDistance);
        } else {
          invalidMovePenalty -= 10 * rewardScale;
          steps === maxSteps;
        }
        const currentDistanceGlobal = distanceMap ? distanceMap[position[1]]?.[position[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
        if (currentDistanceGlobal < lastDistanceGlobal) {
          if (stepsSinceImprovement > 10)
            progressReward += Math.min(
              (stepsSinceImprovement - 10) * 0.01 * rewardScale,
              0.5 * rewardScale
            );
          stepsSinceImprovement = 0;
        }
        lastDistanceGlobal = currentDistanceGlobal;
        if (prevAction === direction && stepsSinceImprovement > 4) {
          invalidMovePenalty -= 0.05 * (stepsSinceImprovement - 4) * rewardScale;
        }
        if (prevAction >= 0 && direction >= 0) {
          const opposite = (prevAction + 2) % 4;
          if (direction === opposite && stepsSinceImprovement > 0) {
            invalidMovePenalty -= 0.2 * rewardScale;
          }
        }
        if (moved) {
          prevAction = direction;
          prevAction = direction;
        }
        if (actionStats) {
          const { entropy, maxProb, secondProb } = actionStats;
          const hasGuidance = vision[8] + vision[9] + vision[10] + vision[11] > 0 || // LOS group
          vision[12] + vision[13] + vision[14] + vision[15] > 0;
          if (entropy > 0.95) {
            invalidMovePenalty -= 0.03 * rewardScale;
          } else if (hasGuidance && entropy < 0.55 && maxProb - secondProb > 0.25) {
            newCellExplorationBonus += 0.015 * rewardScale;
          }
          if (_MazeMovement._saturations >= 5) {
            invalidMovePenalty -= 0.05 * rewardScale;
            if (_MazeMovement._saturations % 10 === 0) {
              invalidMovePenalty -= 0.1 * rewardScale;
            }
          }
        }
        if (stepsSinceImprovement > 40) {
          invalidMovePenalty -= 2 * rewardScale;
          break;
        }
        invalidMovePenalty += loopPenalty + memoryPenalty + revisitPenalty;
        if (position[0] === exitPos[0] && position[1] === exitPos[1]) {
          const stepEfficiency = maxSteps - steps;
          const { actionEntropy: actionEntropy2 } = _MazeMovement.computeActionEntropy(path2);
          const fitness2 = 650 + stepEfficiency * 0.2 + progressReward + newCellExplorationBonus + invalidMovePenalty + actionEntropy2 * 5;
          return {
            success: true,
            steps,
            path: path2,
            fitness: Math.max(150, fitness2),
            progress: 100,
            saturationFraction: steps ? saturatedSteps / steps : 0,
            actionEntropy: actionEntropy2
          };
        }
      }
      const progress = distanceMap ? MazeUtils.calculateProgressFromDistanceMap(
        distanceMap,
        path2[path2.length - 1],
        startPos
      ) : MazeUtils.calculateProgress(
        encodedMaze,
        path2[path2.length - 1],
        startPos,
        exitPos
      );
      const progressFrac = progress / 100;
      const shapedProgress = Math.pow(progressFrac, 1.3) * 500;
      const explorationScore = visitedPositions.size * 1;
      const penalty = invalidMovePenalty;
      const { actionEntropy } = _MazeMovement.computeActionEntropy(path2);
      const entropyBonus = actionEntropy * 4;
      const satFrac = steps ? saturatedSteps / steps : 0;
      const saturationPenalty = satFrac > 0.35 ? -(satFrac - 0.35) * 40 : 0;
      let outputVarPenalty = 0;
      try {
        const hist = network._lastStepOutputs || [];
        if (hist.length >= 15) {
          const recent = hist.slice(-30);
          let lowVar = 0;
          for (const o of recent) {
            const m = (o[0] + o[1] + o[2] + o[3]) / 4;
            let v = 0;
            for (const x of o) v += (x - m) * (x - m);
            v /= 4;
            if (Math.sqrt(v) < 0.01) lowVar++;
          }
          if (lowVar > 4) outputVarPenalty -= (lowVar - 4) * 0.3;
        }
      } catch {
      }
      let nearMissPenalty = 0;
      if (minDistanceToExit === 1) nearMissPenalty -= 30 * rewardScale;
      const base = shapedProgress + explorationScore + progressReward + newCellExplorationBonus + penalty + entropyBonus + localAreaPenalty + saturationPenalty + outputVarPenalty + nearMissPenalty;
      const raw = base + Math.random() * 0.01;
      const fitness = raw >= 0 ? raw : -Math.log1p(1 - raw);
      return {
        success: false,
        steps,
        path: path2,
        fitness,
        progress,
        saturationFraction: satFrac,
        actionEntropy
      };
    }
  };
  ((MazeMovement2) => {
    function computeActionEntropy(path2) {
      if (!path2 || path2.length < 2) return { actionEntropy: 0 };
      const counts = [0, 0, 0, 0];
      for (let i = 1; i < path2.length; i++) {
        const dx = path2[i][0] - path2[i - 1][0];
        const dy = path2[i][1] - path2[i - 1][1];
        if (dx === 0 && dy === -1) counts[0]++;
        else if (dx === 1 && dy === 0) counts[1]++;
        else if (dx === 0 && dy === 1) counts[2]++;
        else if (dx === -1 && dy === 0) counts[3]++;
      }
      const total = counts.reduce((a, b) => a + b, 0) || 1;
      let ent = 0;
      counts.forEach((c) => {
        if (c > 0) {
          const p = c / total;
          ent += -p * Math.log(p);
        }
      });
      const actionEntropy = ent / Math.log(4);
      return { actionEntropy };
    }
    MazeMovement2.computeActionEntropy = computeActionEntropy;
  })(MazeMovement || (MazeMovement = {}));

  // test/examples/asciiMaze/fitness.ts
  var FitnessEvaluator = class _FitnessEvaluator {
    /**
     * Evaluates the fitness of a single neural network based on its performance in a maze simulation.
     *
     * This is the core of the fitness calculation. It runs a simulation of the agent controlled
     * by the given network and then calculates a score based on a combination of factors.
     * A well-designed fitness function is crucial for guiding the evolution towards the desired behavior.
     *
     * The fitness function rewards several key behaviors:
     * - **Progress**: How close did the agent get to the exit? This is the primary driver.
     * - **Success**: A large, fixed bonus is awarded for successfully reaching the exit.
     * - **Efficiency**: If the exit is reached, an additional bonus is given for shorter paths.
     *   This encourages the agent to find the most direct route.
     * - **Exploration**: A bonus is given for each unique cell the agent visits. This encourages
     *   the agent to explore the maze rather than getting stuck in a small area. The exploration
     *   bonus is weighted by the cell's proximity to the exit, rewarding exploration in promising areas.
     *
     * @param network - The neural network to be evaluated.
     * @param encodedMaze - A 2D array representing the maze layout.
     * @param startPosition - The agent's starting coordinates `[x, y]`.
     * @param exitPosition - The maze's exit coordinates `[x, y]`.
     * @param distanceMap - A pre-calculated map of distances from each cell to the exit, for performance.
     * @param maxSteps - The maximum number of steps the agent is allowed to take in the simulation.
     * @returns The final computed fitness score for the network.
     */
    static evaluateNetworkFitness(network, encodedMaze, startPosition, exitPosition, distanceMap, maxSteps) {
      const result = MazeMovement.simulateAgent(
        network,
        encodedMaze,
        startPosition,
        exitPosition,
        distanceMap,
        maxSteps
      );
      let explorationBonus = 0;
      for (const [x, y] of result.path) {
        const distToExit = distanceMap ? distanceMap[y]?.[x] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, [x, y], exitPosition);
        const proximityMultiplier = 1.5 - 0.5 * (distToExit / (encodedMaze.length + encodedMaze[0].length));
        if (result.path.filter(([px, py]) => px === x && py === y).length === 1) {
          explorationBonus += 200 * proximityMultiplier;
        }
      }
      let fitness = result.fitness + explorationBonus;
      if (result.success) {
        fitness += 5e3;
        const optimal = distanceMap ? distanceMap[startPosition[1]]?.[startPosition[0]] ?? Infinity : MazeUtils.bfsDistance(encodedMaze, startPosition, exitPosition);
        const pathOverhead = (result.path.length - 1) / optimal * 100 - 100;
        fitness += Math.max(0, 8e3 - pathOverhead * 80);
      }
      return fitness;
    }
    /**
     * A wrapper function that serves as the default fitness evaluator for the NEAT evolution process.
     *
     * This function acts as an adapter. The main evolution engine (`EvolutionEngine`) works with a
     * standardized `context` object that bundles all the necessary information for an evaluation.
     * This method simply unpacks that context object and passes the individual parameters to the
     * core `evaluateNetworkFitness` function.
     *
     * @param network - The neural network to be evaluated.
     * @param context - An object containing all the necessary data for the fitness evaluation,
     *                  such as the maze, start/exit positions, and simulation configuration.
     * @returns The computed fitness score for the network.
     */
    static defaultFitnessEvaluator(network, context) {
      return _FitnessEvaluator.evaluateNetworkFitness(
        network,
        context.encodedMaze,
        context.startPosition,
        context.exitPosition,
        context.distanceMap,
        context.agentSimConfig.maxSteps
      );
    }
  };

  // test/examples/asciiMaze/evolutionEngine.ts
  var EvolutionEngine = class _EvolutionEngine {
    /**
     * Runs the NEAT neuro-evolution process for an agent to solve a given ASCII maze.
     *
     * This is the core function of the `EvolutionEngine`. It sets up and runs the evolutionary
     * algorithm to train a population of neural networks. Each network acts as the "brain" for an
     * agent, controlling its movement through the maze from a start point 'S' to an exit 'E'.
     *
     * The process involves several key steps:
     * 1.  **Initialization**: Sets up the maze, NEAT parameters, and the initial population of networks.
     * 2.  **Generational Loop**: Iterates through generations, performing the following for each:
     *     a. **Evaluation**: Each network's performance (fitness) is measured by how well its agent navigates the maze.
     *        Fitness is typically based on progress towards the exit, speed, and efficiency.
     *     b. **Lamarckian Refinement**: Each individual in the population undergoes a brief period of supervised training
     *        (backpropagation) on a set of ideal sensory-action pairs. This helps to fine-tune promising behaviors.
     *     c. **Selection & Reproduction**: The NEAT algorithm selects the fittest individuals to become parents for the
     *        next generation. It uses genetic operators (crossover and mutation) to create offspring.
     * 3.  **Termination**: The loop continues until a solution is found (an agent successfully reaches the exit) or other
     *     stopping criteria are met (e.g., maximum generations, stagnation).
     *
     * This hybrid approach, combining the global search of evolution with the local search of backpropagation,
     * can significantly accelerate learning and lead to more robust solutions.
     *
     * @param options - A comprehensive configuration object for the maze evolution process.
     * @returns A Promise that resolves with an object containing the best network found, its simulation result, and the final NEAT instance.
     */
    static async runMazeEvolution(options) {
      const {
        mazeConfig,
        agentSimConfig,
        evolutionAlgorithmConfig,
        reportingConfig,
        fitnessEvaluator
      } = options;
      const { maze } = mazeConfig;
      const { logEvery = 10, dashboardManager } = reportingConfig;
      const {
        allowRecurrent = true,
        // Allow networks to have connections that loop back, enabling memory.
        popSize = 500,
        // The number of neural networks in each generation.
        maxStagnantGenerations = 500,
        // Stop evolution if the best fitness doesn't improve for this many generations.
        minProgressToPass = 95,
        // The percentage of progress required to consider the maze "solved".
        maxGenerations = Infinity,
        // A safety cap on the total number of generations to prevent infinite loops.
        randomSeed,
        // An optional seed for the random number generator to ensure reproducible results.
        initialPopulation,
        // An optional population of networks to start with.
        initialBestNetwork,
        // An optional pre-trained network to seed the population.
        lamarckianIterations = 10,
        // The number of backpropagation steps for each individual per generation.
        lamarckianSampleSize,
        // If set, use a random subset of the training data for Lamarckian learning.
        plateauGenerations = 40,
        // Number of generations to wait for improvement before considering the population to be on a plateau.
        plateauImprovementThreshold = 1e-6,
        // The minimum fitness improvement required to reset the plateau counter.
        simplifyDuration = 30,
        // The number of generations to run the network simplification process.
        simplifyPruneFraction = 0.05,
        // The fraction of weak connections to prune during simplification.
        simplifyStrategy = "weakWeight",
        // The strategy for choosing which connections to prune.
        persistEvery = 25,
        // Save a snapshot of the best networks every N generations.
        persistDir = "./ascii_maze_snapshots",
        // The directory to save snapshots in.
        persistTopK = 3,
        // The number of top-performing networks to save in each snapshot.
        dynamicPopEnabled = true,
        // Enable dynamic adjustment of the population size.
        dynamicPopMax: dynamicPopMaxCfg,
        // The maximum population size for dynamic adjustments.
        dynamicPopExpandInterval = 25,
        // The number of generations between population size expansions.
        dynamicPopExpandFactor = 0.15,
        // The factor by which to expand the population size.
        dynamicPopPlateauSlack = 0.6
        // A slack factor for plateau detection when dynamic population is enabled.
      } = evolutionAlgorithmConfig;
      const dynamicPopMax = typeof dynamicPopMaxCfg === "number" ? dynamicPopMaxCfg : Math.max(popSize, 120);
      const encodedMaze = MazeUtils.encodeMaze(maze);
      const startPosition = MazeUtils.findPosition(maze, "S");
      const exitPosition = MazeUtils.findPosition(maze, "E");
      const distanceMap = MazeUtils.buildDistanceMap(encodedMaze, exitPosition);
      const inputSize = 6;
      const outputSize = 4;
      const currentFitnessEvaluator = fitnessEvaluator || FitnessEvaluator.defaultFitnessEvaluator;
      const fitnessContext = {
        encodedMaze,
        startPosition,
        exitPosition,
        agentSimConfig,
        distanceMap
      };
      const neatFitnessCallback = (network) => {
        return currentFitnessEvaluator(network, fitnessContext);
      };
      const neat = new Neat(inputSize, outputSize, neatFitnessCallback, {
        popsize: popSize,
        // Define the types of mutations that can occur, allowing for structural evolution.
        mutation: [
          methods_exports.mutation.ADD_NODE,
          methods_exports.mutation.SUB_NODE,
          methods_exports.mutation.ADD_CONN,
          methods_exports.mutation.SUB_CONN,
          methods_exports.mutation.MOD_BIAS,
          methods_exports.mutation.MOD_ACTIVATION,
          methods_exports.mutation.MOD_CONNECTION,
          methods_exports.mutation.ADD_LSTM_NODE
          // Allow adding LSTM nodes for more complex memory.
        ],
        mutationRate: 0.2,
        mutationAmount: 0.3,
        elitism: Math.max(1, Math.floor(popSize * 0.1)),
        // Preserve the top 10% of the population.
        provenance: Math.max(1, Math.floor(popSize * 0.2)),
        // Keep a portion of the population from previous species.
        allowRecurrent,
        minHidden: 6,
        // Start with a minimum number of hidden nodes.
        // Enable advanced features for more sophisticated evolution.
        adaptiveMutation: { enabled: true, strategy: "twoTier" },
        multiObjective: {
          enabled: true,
          complexityMetric: "nodes",
          autoEntropy: true
        },
        telemetry: {
          enabled: true,
          performance: true,
          complexity: true,
          hypervolume: true
        },
        lineageTracking: true,
        novelty: {
          enabled: true,
          descriptor: (g) => [g.nodes.length, g.connections.length],
          blendFactor: 0.15
        },
        targetSpecies: 10,
        // Aim for a target number of species to maintain diversity.
        adaptiveTargetSpecies: {
          enabled: true,
          entropyRange: [0.3, 0.8],
          speciesRange: [6, 14],
          smooth: 0.5
        }
      });
      if (initialPopulation && initialPopulation.length > 0) {
        neat.population = initialPopulation.map(
          (net) => net.clone()
        );
      }
      if (initialBestNetwork) {
        neat.population[0] = initialBestNetwork.clone();
      }
      let bestNetwork = evolutionAlgorithmConfig.initialBestNetwork;
      let bestFitness = -Infinity;
      let bestResult;
      let stagnantGenerations = 0;
      let completedGenerations = 0;
      let plateauCounter = 0;
      let simplifyMode = false;
      let simplifyRemaining = 0;
      let lastBestFitnessForPlateau = -Infinity;
      let fs = null;
      let path2 = null;
      try {
        if (typeof window === "undefined" && typeof __require === "function") {
          fs = __require("fs");
          path2 = require_path();
        }
      } catch {
        fs = null;
        path2 = null;
      }
      const flushToFrame = () => {
        const rafPromise = () => new Promise(
          (resolve) => window.requestAnimationFrame(() => resolve())
        );
        const immediatePromise = () => new Promise(
          (resolve) => typeof setImmediate === "function" ? setImmediate(resolve) : setTimeout(resolve, 0)
        );
        if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
          return new Promise(async (resolve) => {
            const check = async () => {
              if (window.asciiMazePaused) {
                await rafPromise();
                setTimeout(check, 0);
              } else {
                rafPromise().then(() => resolve());
              }
            };
            check();
          });
        }
        if (typeof setImmediate === "function") {
          return new Promise(async (resolve) => {
            const check = async () => {
              if (globalThis.asciiMazePaused) {
                await immediatePromise();
                setTimeout(check, 0);
              } else {
                immediatePromise().then(() => resolve());
              }
            };
            check();
          });
        }
        return new Promise((resolve) => setTimeout(resolve, 0));
      };
      if (fs && persistDir && !fs.existsSync(persistDir)) {
        try {
          fs.mkdirSync(persistDir, { recursive: true });
        } catch (e) {
          console.error(
            `Could not create persistence directory: ${persistDir}`,
            e
          );
        }
      }
      const lamarckianTrainingSet = (() => {
        const ds = [];
        const OUT = (d) => [0, 1, 2, 3].map((i) => i === d ? 0.92 : 0.02);
        const add = (inp, dir) => ds.push({ input: inp, output: OUT(dir) });
        add([0, 1, 0, 0, 0, 0.7], 0);
        add([0.25, 0, 1, 0, 0, 0.7], 1);
        add([0.5, 0, 0, 1, 0, 0.7], 2);
        add([0.75, 0, 0, 0, 1, 0.7], 3);
        add([0, 1, 0, 0, 0, 0.9], 0);
        add([0.25, 0, 1, 0, 0, 0.9], 1);
        add([0, 1, 0.6, 0, 0, 0.6], 0);
        add([0, 1, 0, 0.6, 0, 0.6], 0);
        add([0.25, 0.6, 1, 0, 0, 0.6], 1);
        add([0.25, 0, 1, 0.6, 0, 0.6], 1);
        add([0.5, 0, 0.6, 1, 0, 0.6], 2);
        add([0.5, 0, 0, 1, 0.6, 0.6], 2);
        add([0.75, 0, 0, 0.6, 1, 0.6], 3);
        add([0.75, 0.6, 0, 0, 1, 0.6], 3);
        add([0, 1, 0.8, 0.5, 0.4, 0.55], 0);
        add([0.25, 0.7, 1, 0.6, 0.5, 0.55], 1);
        add([0.5, 0.6, 0.55, 1, 0.65, 0.55], 2);
        add([0.75, 0.5, 0.45, 0.7, 1, 0.55], 3);
        add([0, 1, 0.3, 0, 0, 0.4], 0);
        add([0.25, 0.5, 1, 0.4, 0, 0.4], 1);
        add([0.5, 0, 0.3, 1, 0.2, 0.4], 2);
        add([0.75, 0, 0.5, 0.4, 1, 0.4], 3);
        add([0, 0, 0, 1e-3, 0, 0.45], 2);
        ds.forEach((p) => {
          for (let i = 1; i <= 4; i++)
            if (p.input[i] === 1 && Math.random() < 0.25)
              p.input[i] = 0.95 + Math.random() * 0.05;
          if (Math.random() < 0.35)
            p.input[5] = Math.min(
              1,
              Math.max(0, p.input[5] + (Math.random() * 0.1 - 0.05))
            );
        });
        return ds;
      })();
      if (lamarckianTrainingSet.length) {
        const centerOutputBiases = (net) => {
          try {
            const outs = net.nodes?.filter((n) => n.type === "output");
            if (!outs?.length) return;
            const mean = outs.reduce((a, n) => a + n.bias, 0) / outs.length;
            let varc = 0;
            outs.forEach((n) => {
              varc += Math.pow(n.bias - mean, 2);
            });
            varc /= outs.length;
            const std = Math.sqrt(varc);
            outs.forEach((n) => {
              n.bias = Math.max(-5, Math.min(5, n.bias - mean));
            });
            net._outputBiasStats = { mean, std };
          } catch {
          }
        };
        neat.population.forEach((net, idx) => {
          try {
            net.train(lamarckianTrainingSet, {
              iterations: Math.min(
                60,
                8 + Math.floor(lamarckianTrainingSet.length / 2)
              ),
              error: 0.01,
              rate: 2e-3,
              momentum: 0.1,
              batchSize: 4,
              allowRecurrent: true,
              cost: methods_exports.Cost.softmaxCrossEntropy
            });
            try {
              const outputNodes = net.nodes.filter(
                (n) => n.type === "output"
              );
              const inputNodes = net.nodes.filter((n) => n.type === "input");
              for (let d = 0; d < 4; d++) {
                const inNode = inputNodes[d + 1];
                const outNode = outputNodes[d];
                if (!inNode || !outNode) continue;
                let conn = net.connections.find(
                  (c) => c.from === inNode && c.to === outNode
                );
                const w = Math.random() * 0.25 + 0.55;
                if (!conn) net.connect(inNode, outNode, w);
                else conn.weight = w;
              }
              const compassNode = inputNodes[0];
              if (compassNode) {
                outputNodes.forEach((out, d) => {
                  let conn = net.connections.find(
                    (c) => c.from === compassNode && c.to === out
                  );
                  const base = 0.05 + d * 0.01;
                  if (!conn) net.connect(compassNode, out, base);
                  else conn.weight = base;
                });
              }
            } catch {
            }
            centerOutputBiases(net);
          } catch {
          }
        });
      }
      const doProfile = typeof process !== "undefined" && typeof process.env !== "undefined" && process.env.ASCII_MAZE_PROFILE === "1";
      let tEvolveTotal = 0;
      let tLamarckTotal = 0;
      let tSimTotal = 0;
      const safeWrite = (msg) => {
        try {
          if (typeof process !== "undefined" && process && process.stdout && typeof process.stdout.write === "function") {
            process.stdout.write(msg);
            return;
          }
        } catch {
        }
        try {
          if (dashboardManager && dashboardManager.logFunction) {
            try {
              dashboardManager.logFunction(msg);
              return;
            } catch {
            }
          }
        } catch {
        }
        if (typeof console !== "undefined" && console.log)
          console.log(msg.trim());
      };
      while (true) {
        const t0 = doProfile ? Date.now() : 0;
        const fittest = await neat.evolve();
        if (doProfile) tEvolveTotal += Date.now() - t0;
        (neat.population || []).forEach((g) => {
          g.nodes?.forEach((n) => {
            if (n.type === "output") n.squash = methods_exports.Activation.identity;
          });
        });
        _EvolutionEngine._speciesHistory = _EvolutionEngine._speciesHistory || [];
        const speciesCount = neat.population?.reduce((set, g) => {
          if (g.species) set.add(g.species);
          return set;
        }, /* @__PURE__ */ new Set()).size || 1;
        _EvolutionEngine._speciesHistory.push(speciesCount);
        if (_EvolutionEngine._speciesHistory.length > 50)
          _EvolutionEngine._speciesHistory.shift();
        const recent = _EvolutionEngine._speciesHistory.slice(-20);
        const collapsed = recent.length === 20 && recent.every((c) => c === 1);
        if (collapsed) {
          const neatAny = neat;
          if (typeof neatAny.mutationRate === "number")
            neatAny.mutationRate = Math.min(0.6, neatAny.mutationRate * 1.5);
          if (typeof neatAny.mutationAmount === "number")
            neatAny.mutationAmount = Math.min(0.8, neatAny.mutationAmount * 1.3);
          if (neatAny.config && neatAny.config.novelty) {
            neatAny.config.novelty.blendFactor = Math.min(
              0.4,
              neatAny.config.novelty.blendFactor * 1.2
            );
          }
        }
        if (dynamicPopEnabled && completedGenerations > 0 && neat.population?.length && neat.population.length < dynamicPopMax) {
          const plateauRatio = plateauGenerations > 0 ? plateauCounter / plateauGenerations : 0;
          const genTrigger = completedGenerations % dynamicPopExpandInterval === 0;
          if (genTrigger && plateauRatio >= dynamicPopPlateauSlack) {
            const currentSize = neat.population.length;
            const targetAdd = Math.min(
              Math.max(1, Math.floor(currentSize * dynamicPopExpandFactor)),
              dynamicPopMax - currentSize
            );
            if (targetAdd > 0) {
              const sorted = neat.population.slice().sort(
                (a, b) => (b.score || -Infinity) - (a.score || -Infinity)
              );
              const parentPool = sorted.slice(
                0,
                Math.max(2, Math.ceil(sorted.length * 0.25))
              );
              for (let i = 0; i < targetAdd; i++) {
                const parent = parentPool[Math.floor(Math.random() * parentPool.length)];
                try {
                  if (typeof neat.spawnFromParent === "function") {
                    const mutateCount = 1 + (Math.random() < 0.5 ? 1 : 0);
                    const child = neat.spawnFromParent(
                      parent,
                      mutateCount
                    );
                    neat.population.push(child);
                  } else {
                    const clone = parent.clone ? parent.clone() : parent;
                    const mutateCount = 1 + (Math.random() < 0.5 ? 1 : 0);
                    for (let m = 0; m < mutateCount; m++) {
                      try {
                        const mutOps = neat.options.mutation || [];
                        if (mutOps.length) {
                          const op = mutOps[Math.floor(Math.random() * mutOps.length)];
                          clone.mutate(op);
                        }
                      } catch {
                      }
                    }
                    clone.score = void 0;
                    try {
                      if (typeof neat.addGenome === "function") {
                        neat.addGenome(clone, [parent._id]);
                      } else {
                        if (neat._nextGenomeId !== void 0)
                          clone._id = neat._nextGenomeId++;
                        if (neat._lineageEnabled) {
                          clone._parents = [parent._id];
                          clone._depth = (parent._depth ?? 0) + 1;
                        }
                        if (typeof neat._invalidateGenomeCaches === "function")
                          neat._invalidateGenomeCaches(clone);
                        neat.population.push(clone);
                      }
                    } catch {
                      try {
                        neat.population.push(clone);
                      } catch {
                      }
                    }
                  }
                } catch {
                }
              }
              neat.options.popsize = neat.population.length;
              safeWrite(
                `[DYNAMIC_POP] Expanded population to ${neat.population.length} at gen ${completedGenerations}
`
              );
            }
          }
        }
        if (lamarckianIterations > 0 && lamarckianTrainingSet.length) {
          const t1 = doProfile ? Date.now() : 0;
          let trainingSetRef = lamarckianTrainingSet;
          if (lamarckianSampleSize && lamarckianSampleSize < lamarckianTrainingSet.length) {
            const picked = [];
            for (let i = 0; i < lamarckianSampleSize; i++) {
              picked.push(
                lamarckianTrainingSet[Math.random() * lamarckianTrainingSet.length | 0]
              );
            }
            trainingSetRef = picked;
          }
          let gradNormSum = 0;
          let gradSamples = 0;
          neat.population.forEach((network) => {
            network.train(trainingSetRef, {
              iterations: lamarckianIterations,
              // Small to preserve diversity
              error: 0.01,
              rate: 1e-3,
              momentum: 0.2,
              batchSize: 2,
              allowRecurrent: true,
              // allow recurrent connections
              cost: methods_exports.Cost.softmaxCrossEntropy
            });
            try {
              const outs = network.nodes?.filter(
                (n) => n.type === "output"
              );
              if (outs?.length) {
                const mean = outs.reduce((a, n) => a + n.bias, 0) / outs.length;
                let varc = 0;
                outs.forEach((n) => {
                  varc += Math.pow(n.bias - mean, 2);
                });
                varc /= outs.length;
                const std = Math.sqrt(varc);
                outs.forEach((n) => {
                  let adjusted = n.bias - mean;
                  if (std < 0.25) adjusted *= 0.7;
                  n.bias = Math.max(-5, Math.min(5, adjusted));
                });
              }
            } catch {
            }
            try {
              if (typeof network.getTrainingStats === "function") {
                const ts = network.getTrainingStats();
                if (ts && Number.isFinite(ts.gradNorm)) {
                  gradNormSum += ts.gradNorm;
                  gradSamples++;
                }
              }
            } catch {
            }
          });
          if (gradSamples > 0) {
            safeWrite(
              `[GRAD] gen=${completedGenerations} meanGradNorm=${(gradNormSum / gradSamples).toFixed(4)} samples=${gradSamples}
`
            );
          }
          if (doProfile) tLamarckTotal += Date.now() - t1;
        }
        const fitness = fittest.score ?? 0;
        completedGenerations++;
        if (fitness > lastBestFitnessForPlateau + plateauImprovementThreshold) {
          plateauCounter = 0;
          lastBestFitnessForPlateau = fitness;
        } else {
          plateauCounter++;
        }
        if (!simplifyMode && plateauCounter >= plateauGenerations) {
          simplifyMode = true;
          simplifyRemaining = simplifyDuration;
          plateauCounter = 0;
        }
        if (simplifyMode) {
          neat.population.forEach((g) => {
            const enabledConns = g.connections.filter(
              (c) => c.enabled !== false
            );
            if (!enabledConns.length) return;
            const pruneCount = Math.max(
              1,
              Math.floor(enabledConns.length * simplifyPruneFraction)
            );
            let candidates = enabledConns.slice();
            if (simplifyStrategy === "weakRecurrentPreferred") {
              const recurrent = candidates.filter(
                (c) => c.from === c.to || c.gater
              );
              const nonRecurrent = candidates.filter(
                (c) => !(c.from === c.to || c.gater)
              );
              recurrent.sort(
                (a, b) => Math.abs(a.weight) - Math.abs(b.weight)
              );
              nonRecurrent.sort(
                (a, b) => Math.abs(a.weight) - Math.abs(b.weight)
              );
              candidates = [...recurrent, ...nonRecurrent];
            } else {
              candidates.sort(
                (a, b) => Math.abs(a.weight) - Math.abs(b.weight)
              );
            }
            candidates.slice(0, pruneCount).forEach((c) => c.enabled = false);
          });
          simplifyRemaining--;
          if (simplifyRemaining <= 0) simplifyMode = false;
        }
        const t2 = doProfile ? Date.now() : 0;
        const generationResult = MazeMovement.simulateAgent(
          fittest,
          encodedMaze,
          startPosition,
          exitPosition,
          distanceMap,
          agentSimConfig.maxSteps
        );
        try {
          fittest._lastStepOutputs = fittest._lastStepOutputs || fittest._lastStepOutputs;
        } catch {
        }
        fittest._saturationFraction = generationResult.saturationFraction;
        fittest._actionEntropy = generationResult.actionEntropy;
        if (generationResult.saturationFraction && generationResult.saturationFraction > 0.5) {
          try {
            const outNodes = fittest.nodes.filter(
              (n) => n.type === "output"
            );
            const hidden = fittest.nodes.filter((n) => n.type === "hidden");
            hidden.forEach((h) => {
              const outs = h.connections.out.filter(
                (c) => outNodes.includes(c.to) && c.enabled !== false
              );
              if (outs.length >= 2) {
                const weights = outs.map((c) => Math.abs(c.weight));
                const mean = weights.reduce((a, b) => a + b, 0) / weights.length;
                const varc = weights.reduce(
                  (a, b) => a + Math.pow(b - mean, 2),
                  0
                ) / weights.length;
                if (mean < 0.5 && varc < 0.01) {
                  outs.sort(
                    (a, b) => Math.abs(a.weight) - Math.abs(b.weight)
                  );
                  const disableCount = Math.max(1, Math.floor(outs.length / 2));
                  for (let i = 0; i < disableCount; i++) outs[i].enabled = false;
                }
              }
            });
          } catch {
          }
        }
        if (completedGenerations % logEvery === 0) {
          try {
            const movesRaw = generationResult.path.map(
              (p, idx, arr) => {
                if (idx === 0) return null;
                const prev = arr[idx - 1];
                const dx = p[0] - prev[0];
                const dy = p[1] - prev[1];
                if (dx === 0 && dy === -1) return 0;
                if (dx === 1 && dy === 0) return 1;
                if (dx === 0 && dy === 1) return 2;
                if (dx === -1 && dy === 0) return 3;
                return null;
              }
            );
            const moves = [];
            for (const mv of movesRaw) {
              if (mv !== null) moves.push(mv);
            }
            const counts = [0, 0, 0, 0];
            moves.forEach((m) => counts[m]++);
            const totalMoves = moves.length || 1;
            const probs = counts.map((c) => c / totalMoves);
            let entropy = 0;
            probs.forEach((p) => {
              if (p > 0) entropy += -p * Math.log(p);
            });
            const entropyNorm = entropy / Math.log(4);
            safeWrite(
              `[ACTION_ENTROPY] gen=${completedGenerations} entropyNorm=${entropyNorm.toFixed(
                3
              )} uniqueMoves=${counts.filter((c) => c > 0).length} pathLen=${generationResult.path.length}
`
            );
            try {
              const outs = fittest.nodes.filter((n) => n.type === "output");
              if (outs.length) {
                const meanB = outs.reduce((a, n) => a + n.bias, 0) / outs.length;
                let varcB = 0;
                outs.forEach((n) => {
                  varcB += Math.pow(n.bias - meanB, 2);
                });
                varcB /= outs.length;
                const stdB = Math.sqrt(varcB);
                safeWrite(
                  `[OUTPUT_BIAS] gen=${completedGenerations} mean=${meanB.toFixed(
                    3
                  )} std=${stdB.toFixed(3)} biases=${outs.map((o) => o.bias.toFixed(2)).join(",")}
`
                );
              }
            } catch {
            }
            try {
              const lastHist = fittest._lastStepOutputs || [];
              if (lastHist.length) {
                const recent2 = lastHist.slice(-40);
                const k = 4;
                const means = new Array(k).fill(0);
                recent2.forEach((v) => {
                  for (let i = 0; i < k; i++) means[i] += v[i];
                });
                for (let i = 0; i < k; i++) means[i] /= recent2.length;
                const stds = new Array(k).fill(0);
                recent2.forEach((v) => {
                  for (let i = 0; i < k; i++)
                    stds[i] += Math.pow(v[i] - means[i], 2);
                });
                for (let i = 0; i < k; i++)
                  stds[i] = Math.sqrt(stds[i] / recent2.length);
                const kurt = new Array(k).fill(0);
                recent2.forEach((v) => {
                  for (let i = 0; i < k; i++)
                    kurt[i] += Math.pow(v[i] - means[i], 4);
                });
                for (let i = 0; i < k; i++) {
                  const denom = Math.pow(stds[i] || 1e-9, 4) * recent2.length;
                  kurt[i] = denom > 0 ? kurt[i] / denom - 3 : 0;
                }
                let entAgg = 0;
                recent2.forEach((v) => {
                  const max = Math.max(...v);
                  const exps = v.map((x) => Math.exp(x - max));
                  const sum = exps.reduce((a, b) => a + b, 0) || 1;
                  const probs2 = exps.map((e2) => e2 / sum);
                  let e = 0;
                  probs2.forEach((p) => {
                    if (p > 0) e += -p * Math.log(p);
                  });
                  entAgg += e / Math.log(4);
                });
                const entMean = entAgg / recent2.length;
                let stable = 0, totalTrans = 0;
                let prevDir = -1;
                recent2.forEach((v) => {
                  const arg = v.indexOf(Math.max(...v));
                  if (prevDir === arg) stable++;
                  if (prevDir !== -1) totalTrans++;
                  prevDir = arg;
                });
                const stability = totalTrans ? stable / totalTrans : 0;
                safeWrite(
                  `[LOGITS] gen=${completedGenerations} means=${means.map((m) => m.toFixed(3)).join(",")} stds=${stds.map((s) => s.toFixed(3)).join(",")} kurt=${kurt.map((kv) => kv.toFixed(2)).join(",")} entMean=${entMean.toFixed(
                    3
                  )} stability=${stability.toFixed(3)} steps=${recent2.length}
`
                );
                _EvolutionEngine._collapseStreak = _EvolutionEngine._collapseStreak || 0;
                const collapsed2 = stds.every((s) => s < 5e-3) && (entMean < 0.35 || stability > 0.97);
                if (collapsed2) _EvolutionEngine._collapseStreak++;
                else _EvolutionEngine._collapseStreak = 0;
                if (_EvolutionEngine._collapseStreak === 6) {
                  try {
                    const eliteCount = neat.options.elitism || 0;
                    const pop = neat.population || [];
                    const reinitTargets = pop.slice(eliteCount).filter(() => Math.random() < 0.3);
                    let connReset = 0, biasReset = 0;
                    reinitTargets.forEach((g) => {
                      const outs = g.nodes.filter(
                        (n) => n.type === "output"
                      );
                      outs.forEach((o) => {
                        o.bias = Math.random() * 0.2 - 0.1;
                        biasReset++;
                      });
                      g.connections.forEach((c) => {
                        if (outs.includes(c.to)) {
                          c.weight = Math.random() * 0.4 - 0.2;
                          connReset++;
                        }
                      });
                    });
                    safeWrite(
                      `[ANTICOLLAPSE] gen=${completedGenerations} reinitGenomes=${reinitTargets.length} connReset=${connReset} biasReset=${biasReset}
`
                    );
                  } catch {
                  }
                }
              }
            } catch {
            }
            try {
              const unique = generationResult.path.length ? new Set(generationResult.path.map((p) => p.join(","))).size : 0;
              const ratio = generationResult.path.length ? unique / generationResult.path.length : 0;
              safeWrite(
                `[EXPLORE] gen=${completedGenerations} unique=${unique} pathLen=${generationResult.path.length} ratio=${ratio.toFixed(
                  3
                )} progress=${generationResult.progress.toFixed(
                  1
                )} satFrac=${generationResult.saturationFraction?.toFixed(
                  3
                )}
`
              );
            } catch {
            }
            try {
              const pop = neat.population || [];
              const speciesCounts = {};
              pop.forEach((g) => {
                const sid = g.species != null ? String(g.species) : "none";
                speciesCounts[sid] = (speciesCounts[sid] || 0) + 1;
              });
              const counts2 = Object.values(speciesCounts);
              const total = counts2.reduce((a, b) => a + b, 0) || 1;
              const simpson = 1 - counts2.reduce((a, b) => a + Math.pow(b / total, 2), 0);
              let wMean = 0, wCount = 0;
              const sample = pop.slice(0, Math.min(pop.length, 40));
              sample.forEach((g) => {
                g.connections.forEach((c) => {
                  if (c.enabled !== false) {
                    wMean += c.weight;
                    wCount++;
                  }
                });
              });
              wMean = wCount ? wMean / wCount : 0;
              let wVar = 0;
              sample.forEach((g) => {
                g.connections.forEach((c) => {
                  if (c.enabled !== false) wVar += Math.pow(c.weight - wMean, 2);
                });
              });
              const wStd = wCount ? Math.sqrt(wVar / wCount) : 0;
              safeWrite(
                `[DIVERSITY] gen=${completedGenerations} species=${Object.keys(speciesCounts).length} simpson=${simpson.toFixed(3)} weightStd=${wStd.toFixed(3)}
`
              );
            } catch {
            }
          } catch {
          }
        }
        if (doProfile) tSimTotal += Date.now() - t2;
        if (fitness > bestFitness) {
          bestFitness = fitness;
          bestNetwork = fittest;
          bestResult = generationResult;
          stagnantGenerations = 0;
          dashboardManager.update(
            maze,
            generationResult,
            fittest,
            completedGenerations,
            neat
          );
          try {
            await flushToFrame();
          } catch {
          }
        } else {
          stagnantGenerations++;
          if (completedGenerations % logEvery === 0) {
            if (bestNetwork && bestResult) {
              dashboardManager.update(
                maze,
                bestResult,
                bestNetwork,
                completedGenerations,
                neat
              );
              try {
                await flushToFrame();
              } catch {
              }
            }
          }
        }
        if (persistEvery > 0 && completedGenerations % persistEvery === 0 && bestNetwork) {
          try {
            const snap = {
              generation: completedGenerations,
              bestFitness,
              simplifyMode,
              plateauCounter,
              timestamp: Date.now(),
              telemetryTail: neat.getTelemetry ? neat.getTelemetry().slice(-5) : void 0
            };
            const popSorted = neat.population.slice().sort(
              (a, b) => (b.score || -Infinity) - (a.score || -Infinity)
            );
            const top = popSorted.slice(0, persistTopK).map((g, idx) => ({
              idx,
              score: g.score,
              nodes: g.nodes.length,
              connections: g.connections.length,
              json: g.toJSON ? g.toJSON() : void 0
            }));
            snap.top = top;
            const file = path2.join(
              persistDir,
              `snapshot_gen${completedGenerations}.json`
            );
            fs.writeFileSync(file, JSON.stringify(snap, null, 2));
          } catch (e) {
          }
        }
        if (bestResult?.success && bestResult.progress >= minProgressToPass) {
          if (bestNetwork && bestResult) {
            dashboardManager.update(
              maze,
              bestResult,
              bestNetwork,
              completedGenerations,
              neat
            );
            try {
              await flushToFrame();
            } catch {
            }
          }
          break;
        }
        if (stagnantGenerations >= maxStagnantGenerations) {
          if (bestNetwork && bestResult) {
            dashboardManager.update(
              maze,
              bestResult,
              bestNetwork,
              completedGenerations,
              neat
            );
            try {
              await flushToFrame();
            } catch {
            }
          }
          break;
        }
        if (completedGenerations >= maxGenerations) {
          break;
        }
      }
      if (doProfile && completedGenerations > 0) {
        const gen = completedGenerations;
        const avgEvolve = (tEvolveTotal / gen).toFixed(2);
        const avgLamarck = (tLamarckTotal / gen).toFixed(2);
        const avgSim = (tSimTotal / gen).toFixed(2);
        safeWrite(
          `
[PROFILE] Generations=${gen} avg(ms): evolve=${avgEvolve} lamarck=${avgLamarck} sim=${avgSim} totalPerGen=${(+avgEvolve + +avgLamarck + +avgSim).toFixed(2)}
`
        );
      }
      return {
        bestNetwork,
        bestResult,
        neat
      };
    }
    /**
     * Prints the structure of a given neural network to the console.
     *
     * This is useful for debugging and understanding the evolved architectures.
     * It prints the number of nodes, their types, activation functions, and connection details.
     *
     * @param network - The neural network to inspect.
     */
    static printNetworkStructure(network) {
      console.log("Network Structure:");
      console.log("Nodes: ", network.nodes?.length);
      const inputNodes = network.nodes?.filter((n) => n.type === "input");
      const outputNodes = network.nodes?.filter((n) => n.type === "output");
      const hiddenNodes = network.nodes?.filter((n) => n.type === "hidden");
      console.log("Input nodes: ", inputNodes?.length);
      console.log("Hidden nodes: ", hiddenNodes?.length);
      console.log("Output nodes: ", outputNodes?.length);
      console.log(
        "Activation functions: ",
        network.nodes?.map((n) => n.squash?.name || n.squash)
      );
      console.log("Connections: ", network.connections?.length);
      const recurrent = network.connections?.some(
        (c) => c.gater || c.from === c.to
      );
      console.log("Has recurrent/gated connections: ", recurrent);
    }
  };

  // test/examples/asciiMaze/mazes.ts
  var mazes_exports = {};
  __export(mazes_exports, {
    large: () => large,
    medium: () => medium,
    medium2: () => medium2,
    minotaur: () => minotaur,
    small: () => small,
    spiral: () => spiral,
    spiralSmall: () => spiralSmall,
    tiny: () => tiny
  });
  var tiny = [
    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551S...................\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551",
    "\u2551....................\u2551",
    "\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551....................\u2551",
    "\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557E\u2551"
  ];
  var spiralSmall = [
    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551...........\u2551",
    "\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551",
    "\u2551.\u2551.......\u2551.\u2551",
    "\u2551.\u2551.\u2554\u2550\u2550\u2550\u2557.\u2551.\u2551",
    "\u2551.\u2551.\u2551...\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551S\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u255A\u2550\u255D.\u2551.\u2551.\u2551",
    "\u2551.\u2551.....\u2551.\u2551.\u2551",
    "\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551",
    "\u2551.........\u2551.\u2551",
    "\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563E\u2551"
  ];
  var spiral = [
    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551...............\u2551",
    "\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.\u2551.\u2551...........\u2551",
    "\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551",
    "\u2551.\u2551.\u2551.\u2551.......\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2557.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551...\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551S\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u255D.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551",
    "\u2551.\u2551.\u2551.........\u2551.\u2551",
    "\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551",
    "\u2551.\u2551.............\u2551",
    "\u2551E\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D"
  ];
  var small = [
    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551S......\u2551..........\u2551",
    "\u2560\u2550\u2550.\u2554\u2550\u2550.\u2551.\u2554\u2550\u2550.\u2551.\u2551..\u2551",
    "\u2551...\u2551...\u2551.\u2551...\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2550\u2550\u255D.\u255A\u2550\u2550\u2550\u255D.\u255A\u2550\u2550\u2563",
    "\u2551.\u2551.\u2551..............\u2551",
    "\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550.\u2550\u2550\u2566\u2550\u2557..\u2551",
    "\u2551.\u2551...........\u2551.\u2551..\u2551",
    "\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2557.\u2550\u2550\u2550\u2550\u2563.\u2551..\u2551",
    "\u2551.......\u2551.....\u2551.\u2551..\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550.\u255A\u2550\u2550\u2550\u2557.\u2551.\u255A\u2550\u2550\u2563",
    "\u2551...........\u2551.\u2551....\u2551",
    "\u2551.\u2550\u2550\u2550.\u2554\u2550\u2550\u2550\u2550.\u2551.\u255A\u2550\u2550\u2550.\u2551",
    "\u2551.....\u2551.....\u2551......\u2551",
    "\u255A\u2550\u2550\u2550\u2550\u2550\u2563E\u2554\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u255D"
  ];
  var medium = [
    "\u2554\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551S\u2551.......\u2551.............\u2551",
    "\u2551.\u2551.\u2550\u2550\u2550\u2550\u2557.\u2560\u2550\u2550\u2550\u2550\u2550\u2557.\u2550\u2550\u2550\u2550\u2557.\u2551",
    "\u2551.\u2551.....\u2551.\u2551.....\u2551.....\u2551.\u2551",
    "\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u255A\u2550\u2550\u2550\u2557.\u255A\u2550\u2557.\u2551.\u2551.\u2551",
    "\u2551.....\u2551.\u2551.....\u2551...\u2551.\u2551.\u2551.\u2551",
    "\u2560\u2550\u2550\u2550\u2557.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2560\u2550\u2550.\u2551.\u2551.\u2551.\u2551",
    "\u2551...\u2551.\u2551.....\u2551.\u2551...\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2550\u2550\u255D.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551",
    "\u2551.\u2551.....\u2551.\u2551.\u2551.........\u2551.\u2551",
    "\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563.\u2551",
    "\u2551.....\u2551.\u2551.\u2551...........\u2551.\u2551",
    "\u2560\u2550\u2550\u2550\u2550.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2557.\u2551.\u2551",
    "\u2551.....\u2551.........\u2551...\u2551.\u2551.\u2551",
    "\u2551.\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551",
    "\u2551.................\u2551.\u2551.\u2551.\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551...............\u2551.\u2551...\u2551.\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2569\u2550\u2550\u2550\u255D.\u2551",
    "\u2551.......................\u2551",
    "\u2551E\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D"
  ];
  var medium2 = [
    "\u2554\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551S\u2551.......\u2551.....................\u2551.............\u2551",
    "\u2551.\u2551.\u2550\u2550\u2550\u2550\u2557.\u2560\u2550\u2550\u2550\u2550\u2550\u2557.\u2550\u2550\u2550\u2550\u2557.\u2551.\u2550\u2550\u2550\u2550\u2557.\u2551.\u2550\u2550\u2550\u2550\u2557.\u2550\u2550\u2550\u2550\u2557.\u2551",
    "\u2551.\u2551.....\u2551.\u2551.....\u2551.....\u2551.\u2551.....\u2551.\u2551.....\u2551.....\u2551.\u2551",
    "\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u255A\u2550\u2550\u2550\u2557.\u255A\u2550\u2557.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2563.\u2551.\u255A\u2550\u2550\u2550\u2563.\u255A\u2550\u2557.\u2551.\u2551",
    "\u2551.....\u2551.\u2551.....\u2551...\u2551.\u2551.\u2551.\u2551.....\u2551.\u2551.....\u2551...\u2551.\u2551.\u2551",
    "\u2560\u2550\u2550\u2550\u2557.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2560\u2550\u2550.\u2551.\u2551.\u2551.\u2560\u2550\u2550\u2550\u2557.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2560\u2550\u2550.\u2551.\u2551.\u2551",
    "\u2551...\u2551.\u2551.....\u2551.\u2551...\u2551.\u2551.\u2551.\u2551...\u2551.\u2551.....\u2551.\u2551...\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2550\u2550\u255D.\u2551.\u2551",
    "\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551.....\u2551.\u2551",
    "\u2551.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u255D.\u2551",
    "\u2551.\u2551.....\u2551.\u2551.\u2551.........\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551.........\u2551",
    "\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.....\u2551.\u2551.\u2551...........\u2551.\u2551.....\u2551.\u2551.\u2551...........\u2551",
    "\u2560\u2550\u2550\u2550\u2550.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2557.\u2551.\u2551\u2550\u2550\u2550\u2550\u2550\u2563.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2557.\u2551",
    "\u2551.....\u2551.........\u2551...\u2551.\u2551.\u2551.....\u2551.........\u2551...\u2551.\u2551",
    "\u2551.\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551",
    "\u2551.................\u2551.\u2551.\u2551.\u2551.................\u2551.\u2551.\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2551.\u2551.\u2551.\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2551.\u2551.\u2551",
    "\u2551...............\u2551.\u2551...\u2551.\u2551...............\u2551.\u2551...\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2569\u2550\u2550\u2550\u2569\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u255A\u2550\u2550\u2550\u2563",
    "\u2551.............................................\u2551",
    "\u2551E\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D"
  ];
  var large = [
    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551S.......................................\u2551................\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2554\u2550\u2550\u2550\u2550\u2550.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550.\u2551",
    "\u2551..........\u2551...................\u2551......................\u2551...\u2551",
    "\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550.\u2554\u2569\u2550.\u2550\u2550\u2550\u2550\u2557.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2554\u2550\u2550\u2550\u2550\u2550\u2557.\u2550\u2569\u2550\u2550.\u2551",
    "\u2551.\u2551.......\u2551.......\u2551.\u2551......................\u2551.\u2551.....\u2551......\u2551",
    "\u2551.\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563.\u2551.\u2550\u2550\u2550\u2550\u256C\u2550\u2550.\u2550\u2550\u2550\u2563",
    "\u2551.\u2551.................\u2551.\u2551.\u2551..................\u2551.\u2551.....\u2551......\u2551",
    "\u2551.\u2551.\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563.\u2551.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2560\u2550\u2550\u2550\u2550.\u2551.\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.\u2551....\u2554\u2569\u2557..........\u255A\u2566\u255D.\u2551.\u2551..............\u2551.\u2551.\u2551.....\u2551......\u2551",
    "\u2560\u2550\u255D..\u2551.\u2551.\u2560\u2550\u2550.\u2551.\u2551.\u2551...\u2551..\u255A\u2550\u2569\u2566\u2550\u2557.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551....\u2551.\u2551.\u2551...\u2551.\u2551.\u2551.\u2551.\u2551.....\u2551.\u2551.......\u2551...\u2551.\u2551..............\u2551",
    "\u2560\u2550.\u2550\u2550\u2569\u2550\u255D.\u255A\u2550\u2550\u2550\u2569\u2550\u255D.\u255A\u2550\u2569\u2566\u2569\u2550\u2550\u2550\u2550\u2566\u255D.\u2551.\u2550\u2550\u2550\u2566\u2550\u2550\u255D.\u2550\u2550\u2569\u2566\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2550\u2563",
    "\u2551...................\u2551.....\u2551..\u2551....\u2551.......\u2551...............\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2551.\u2550\u2550\u2563.\u2554\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550.\u2554\u2550\u255D.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551...................\u2551.\u2551...\u2551.\u2551...........\u2551.................\u2551",
    "\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u255A\u2550\u2550\u2550\u255D.\u2551.\u2554\u2550\u2550.\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2566\u2550.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.\u2551.....................\u2551...\u2551.\u2551...............\u2551...........\u2551",
    "\u2551.\u255A\u2550\u2557.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550.\u2551.\u2554\u2550\u255D.\u255A\u2550\u2557..\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2550\u2563",
    "\u2551...\u2551.\u2551.......\u2551...\u2551.....\u2551.\u2551.....\u2551...........\u2551.\u2551...........\u2551",
    "\u2560\u2550\u2557.\u2551.\u255A\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2550\u2550\u255D.\u2554\u2550\u2550.\u2554\u2550\u2569\u2550\u2550\u2550\u2550.\u2554\u2550\u2566\u2550\u2550.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551",
    "\u2551.\u2551.\u2551.....\u2551.\u2551...\u2551.\u2551.\u2551.....\u2551...\u2551.......\u2551.\u2551...\u2551.............\u2551",
    "\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2563.\u2551.\u2554\u2550\u2563.\u255A\u2550\u255D.\u2554\u2550\u2550.\u2551.\u2554\u2550\u2569\u2550\u2550.\u2550\u2550\u2550\u2550\u255D.\u2551.\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.\u2551.......\u2551...\u2551.\u2551.....\u2551...\u2551.\u2551...........\u2551.................\u2551",
    "\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550.\u255A\u2550\u2550.\u2551.\u2560\u2550\u2550\u2550\u2550.\u255A\u2550\u2550\u2550\u255D.\u255A\u2550\u2550.\u2550\u2566\u2550\u2566\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551",
    "\u2551.............\u2551.\u2551................\u2551.\u2551......................\u2551",
    "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550.\u2550\u2550\u2550\u255D.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.............\u2551.......\u2551............\u2551.\u2551....................\u2551",
    "\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u255A\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2563.\u2554\u2550\u2566\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2569\u2557.\u2554\u2550\u2550\u2550\u2550\u2550\u2550.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.\u2551...............\u2551...\u2551.\u2551.\u2551.\u2551.........\u2551.\u2551.......\u2551.........\u2551",
    "\u2551.\u2560\u2550\u2557.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551.\u2551.\u2551.....................\u2551.\u2551.\u2551.........\u2551.................\u2551",
    "\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550...\u2551",
    "\u2551.\u2551.......................................................\u2551",
    "\u2560\u2550\u2569\u2550\u2566\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2566\u2550\u2557.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
    "\u2551...\u2551...\u2551...\u2551...\u2551...\u2551...\u2551...\u2551.\u2551...........................\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2551",
    "\u2551.\u2551...\u2551...\u2551...\u2551...\u2551...\u2551...\u2551...............................\u2551",
    "\u2551E\u2554\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D"
  ];
  var minotaur = [
    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
    "\u2551..............................................................................\u2551",
    "\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557..\u2551",
    "\u2551.\u2551............\u2551.\u2551.........................................\u2551.\u2551..............\u2551..\u2551",
    "\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551........\u2551.\u2551.\u2551.\u2551.....................................\u2551.\u2551.\u2551.\u2551........\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551....\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.................................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551....\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2554\u2550.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550.\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.............................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2550\u2563.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551....\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.........................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551....\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551........\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.....................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551........\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551............\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551............\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551..\u2551",
    "\u2551.\u2551................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.............\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551................\u2551..\u2551",
    "\u2551.\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563..\u2551",
    "\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.........\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551..\u2551",
    "\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551..\u2551",
    "\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551S\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551...\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2560\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551.\u2551.....\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.........\u2551.\u2551.\u2551..................\u2551.\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2551..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..................\u2551................................\u2551.\u2551.\u2551.\u2551.\u2560\u2550\u255D..\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2554\u2569\u2566\u2569\u2566\u2569\u2566\u2569\u2566\u2569\u2557...\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..............................................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557..\u2551.\u2551...\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..........\u2551.\u2551............................\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u255A\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551......\u2551.\u2551.\u2551.\u2551........................\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2560\u2550\u2550\u2557.\u2551.\u2551.\u2551.\u2551.\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557..\u2554\u2569\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551....................\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u255A\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551....\u2551.\u2551.\u2551.\u2551.\u2551.\u2551................\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u255A\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551........\u2551.\u2551.\u2551.\u2551.\u2551.\u2551............\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550.\u2560\u2550.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551............\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..........\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2557.\u255A\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551....................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551......\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2554\u2550\u2550\u2557.\u255A\u2557.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u2551.\u2551......................\u2551.\u2551.\u2551.\u2551.\u2551.\u2551..\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551.\u255A\u2550\u2569\u2550\u2550.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2550\u255D.\u2554\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u2551............................\u2551.\u2551.\u2551.\u2551.\u2551....\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2551.\u255A\u2550.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2550\u2550\u2550\u2569\u2550\u2550\u2569\u2566\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551....................................\u2551.\u2551.\u2551........\u2551..\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2563.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551........................................\u2551.\u2551...........\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551.\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551.\u2551",
    "\u2551..........................................................\u2551.....\u2551.\u2551.........\u2551.\u2551",
    "\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563E\u2551"
  ];

  // test/examples/asciiMaze/browser-entry.ts
  async function start(containerId = "ascii-maze-output") {
    const host = document.getElementById(containerId);
    const archiveEl = host ? host.querySelector("#ascii-maze-archive") : null;
    const liveEl = host ? host.querySelector("#ascii-maze-live") : null;
    const clearFn = BrowserTerminalUtility.createTerminalClearer(
      liveEl ?? void 0
    );
    const liveLogFn = createBrowserLogger(liveEl ?? void 0);
    const archiveLogFn = createBrowserLogger(archiveEl ?? void 0);
    const dashboard = new DashboardManager(
      clearFn,
      liveLogFn,
      archiveLogFn
    );
    window.asciiMazeStart = async () => {
      const order = [
        "tiny",
        "spiralSmall",
        "spiral",
        "small",
        "medium",
        "medium2",
        "large",
        "minotaur"
      ];
      let lastBestNetwork = void 0;
      for (const key of order) {
        const maze = mazes_exports[key];
        if (!Array.isArray(maze)) continue;
        let agentMaxSteps = 1e3;
        let maxGenerations = 500;
        switch (key) {
          case "tiny":
            agentMaxSteps = 100;
            maxGenerations = 200;
            break;
          case "spiralSmall":
            agentMaxSteps = 100;
            maxGenerations = 200;
            break;
          case "spiral":
            agentMaxSteps = 150;
            maxGenerations = 300;
            break;
          case "small":
            agentMaxSteps = 50;
            maxGenerations = 300;
            break;
          case "medium":
            agentMaxSteps = 250;
            maxGenerations = 400;
            break;
          case "medium2":
            agentMaxSteps = 300;
            maxGenerations = 400;
            break;
          case "large":
            agentMaxSteps = 400;
            maxGenerations = 500;
            break;
          case "minotaur":
            agentMaxSteps = 700;
            maxGenerations = 600;
            break;
        }
        try {
          const result = await EvolutionEngine.runMazeEvolution({
            mazeConfig: { maze },
            agentSimConfig: { maxSteps: agentMaxSteps },
            evolutionAlgorithmConfig: {
              allowRecurrent: true,
              popSize: 40,
              maxStagnantGenerations: 200,
              minProgressToPass: 99,
              maxGenerations,
              // Disable Lamarckian/backprop refinement for browser runs per request
              lamarckianIterations: 0,
              lamarckianSampleSize: 0,
              // seed previous winner if available
              initialBestNetwork: lastBestNetwork
            },
            reportingConfig: {
              dashboardManager: dashboard,
              logEvery: 1,
              label: `browser-${key}`
            }
          });
          if (result && result.bestNetwork)
            lastBestNetwork = result.bestNetwork;
        } catch (e) {
          console.error("Error while running maze", key, e);
        }
      }
    };
    window.asciiMazeStart();
    try {
      window.asciiMazePaused = false;
      const playPauseBtn = document.getElementById(
        "ascii-maze-playpause"
      );
      const updateUI = () => {
        const paused = !!window.asciiMazePaused;
        if (playPauseBtn) {
          playPauseBtn.textContent = paused ? "Play" : "Pause";
          playPauseBtn.style.background = paused ? "#39632C" : "#2C3963";
          playPauseBtn.setAttribute("aria-pressed", String(paused));
        }
      };
      if (playPauseBtn) {
        playPauseBtn.addEventListener("click", () => {
          window.asciiMazePaused = !window.asciiMazePaused;
          updateUI();
        });
      }
      updateUI();
    } catch {
    }
  }
  if (typeof window !== "undefined" && window.document) {
    setTimeout(() => start(), 20);
  }
})();
//# sourceMappingURL=ascii-maze.bundle.js.map
