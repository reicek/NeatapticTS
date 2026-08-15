var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __commonJS = (cb, mod) => function __require() {
  try {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  } catch (e) {
    throw mod = 0, e;
  }
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

// node_modules/seedrandom/lib/alea.js
var require_alea = __commonJS({
  "node_modules/seedrandom/lib/alea.js"(exports, module) {
    (function(global, module2, define2) {
      function Alea(seed) {
        var me = this, mash = Mash();
        me.next = function() {
          var t = 2091639 * me.s0 + me.c * 23283064365386963e-26;
          me.s0 = me.s1;
          me.s1 = me.s2;
          return me.s2 = t - (me.c = t | 0);
        };
        me.c = 1;
        me.s0 = mash(" ");
        me.s1 = mash(" ");
        me.s2 = mash(" ");
        me.s0 -= mash(seed);
        if (me.s0 < 0) {
          me.s0 += 1;
        }
        me.s1 -= mash(seed);
        if (me.s1 < 0) {
          me.s1 += 1;
        }
        me.s2 -= mash(seed);
        if (me.s2 < 0) {
          me.s2 += 1;
        }
        mash = null;
      }
      function copy(f, t) {
        t.c = f.c;
        t.s0 = f.s0;
        t.s1 = f.s1;
        t.s2 = f.s2;
        return t;
      }
      function impl(seed, opts) {
        var xg = new Alea(seed), state = opts && opts.state, prng = xg.next;
        prng.int32 = function() {
          return xg.next() * 4294967296 | 0;
        };
        prng.double = function() {
          return prng() + (prng() * 2097152 | 0) * 11102230246251565e-32;
        };
        prng.quick = prng;
        if (state) {
          if (typeof state == "object") copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      function Mash() {
        var n = 4022871197;
        var mash = function(data) {
          data = String(data);
          for (var i = 0; i < data.length; i++) {
            n += data.charCodeAt(i);
            var h = 0.02519603282416938 * n;
            n = h >>> 0;
            h -= n;
            h *= n;
            n = h >>> 0;
            h -= n;
            n += h * 4294967296;
          }
          return (n >>> 0) * 23283064365386963e-26;
        };
        return mash;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define2 && define2.amd) {
        define2(function() {
          return impl;
        });
      } else {
        this.alea = impl;
      }
    })(
      exports,
      typeof module == "object" && module,
      // present in node.js
      typeof define == "function" && define
      // present with an AMD loader
    );
  }
});

// node_modules/seedrandom/lib/xor128.js
var require_xor128 = __commonJS({
  "node_modules/seedrandom/lib/xor128.js"(exports, module) {
    (function(global, module2, define2) {
      function XorGen(seed) {
        var me = this, strseed = "";
        me.x = 0;
        me.y = 0;
        me.z = 0;
        me.w = 0;
        me.next = function() {
          var t = me.x ^ me.x << 11;
          me.x = me.y;
          me.y = me.z;
          me.z = me.w;
          return me.w ^= me.w >>> 19 ^ t ^ t >>> 8;
        };
        if (seed === (seed | 0)) {
          me.x = seed;
        } else {
          strseed += seed;
        }
        for (var k = 0; k < strseed.length + 64; k++) {
          me.x ^= strseed.charCodeAt(k) | 0;
          me.next();
        }
      }
      function copy(f, t) {
        t.x = f.x;
        t.y = f.y;
        t.z = f.z;
        t.w = f.w;
        return t;
      }
      function impl(seed, opts) {
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result2 = (top + bot) / (1 << 21);
          } while (result2 === 0);
          return result2;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (typeof state == "object") copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define2 && define2.amd) {
        define2(function() {
          return impl;
        });
      } else {
        this.xor128 = impl;
      }
    })(
      exports,
      typeof module == "object" && module,
      // present in node.js
      typeof define == "function" && define
      // present with an AMD loader
    );
  }
});

// node_modules/seedrandom/lib/xorwow.js
var require_xorwow = __commonJS({
  "node_modules/seedrandom/lib/xorwow.js"(exports, module) {
    (function(global, module2, define2) {
      function XorGen(seed) {
        var me = this, strseed = "";
        me.next = function() {
          var t = me.x ^ me.x >>> 2;
          me.x = me.y;
          me.y = me.z;
          me.z = me.w;
          me.w = me.v;
          return (me.d = me.d + 362437 | 0) + (me.v = me.v ^ me.v << 4 ^ (t ^ t << 1)) | 0;
        };
        me.x = 0;
        me.y = 0;
        me.z = 0;
        me.w = 0;
        me.v = 0;
        if (seed === (seed | 0)) {
          me.x = seed;
        } else {
          strseed += seed;
        }
        for (var k = 0; k < strseed.length + 64; k++) {
          me.x ^= strseed.charCodeAt(k) | 0;
          if (k == strseed.length) {
            me.d = me.x << 10 ^ me.x >>> 4;
          }
          me.next();
        }
      }
      function copy(f, t) {
        t.x = f.x;
        t.y = f.y;
        t.z = f.z;
        t.w = f.w;
        t.v = f.v;
        t.d = f.d;
        return t;
      }
      function impl(seed, opts) {
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result2 = (top + bot) / (1 << 21);
          } while (result2 === 0);
          return result2;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (typeof state == "object") copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define2 && define2.amd) {
        define2(function() {
          return impl;
        });
      } else {
        this.xorwow = impl;
      }
    })(
      exports,
      typeof module == "object" && module,
      // present in node.js
      typeof define == "function" && define
      // present with an AMD loader
    );
  }
});

// node_modules/seedrandom/lib/xorshift7.js
var require_xorshift7 = __commonJS({
  "node_modules/seedrandom/lib/xorshift7.js"(exports, module) {
    (function(global, module2, define2) {
      function XorGen(seed) {
        var me = this;
        me.next = function() {
          var X = me.x, i = me.i, t, v, w;
          t = X[i];
          t ^= t >>> 7;
          v = t ^ t << 24;
          t = X[i + 1 & 7];
          v ^= t ^ t >>> 10;
          t = X[i + 3 & 7];
          v ^= t ^ t >>> 3;
          t = X[i + 4 & 7];
          v ^= t ^ t << 7;
          t = X[i + 7 & 7];
          t = t ^ t << 13;
          v ^= t ^ t << 9;
          X[i] = v;
          me.i = i + 1 & 7;
          return v;
        };
        function init(me2, seed2) {
          var j, w, X = [];
          if (seed2 === (seed2 | 0)) {
            w = X[0] = seed2;
          } else {
            seed2 = "" + seed2;
            for (j = 0; j < seed2.length; ++j) {
              X[j & 7] = X[j & 7] << 15 ^ seed2.charCodeAt(j) + X[j + 1 & 7] << 13;
            }
          }
          while (X.length < 8) X.push(0);
          for (j = 0; j < 8 && X[j] === 0; ++j) ;
          if (j == 8) w = X[7] = -1;
          else w = X[j];
          me2.x = X;
          me2.i = 0;
          for (j = 256; j > 0; --j) {
            me2.next();
          }
        }
        init(me, seed);
      }
      function copy(f, t) {
        t.x = f.x.slice();
        t.i = f.i;
        return t;
      }
      function impl(seed, opts) {
        if (seed == null) seed = +/* @__PURE__ */ new Date();
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result2 = (top + bot) / (1 << 21);
          } while (result2 === 0);
          return result2;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (state.x) copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define2 && define2.amd) {
        define2(function() {
          return impl;
        });
      } else {
        this.xorshift7 = impl;
      }
    })(
      exports,
      typeof module == "object" && module,
      // present in node.js
      typeof define == "function" && define
      // present with an AMD loader
    );
  }
});

// node_modules/seedrandom/lib/xor4096.js
var require_xor4096 = __commonJS({
  "node_modules/seedrandom/lib/xor4096.js"(exports, module) {
    (function(global, module2, define2) {
      function XorGen(seed) {
        var me = this;
        me.next = function() {
          var w = me.w, X = me.X, i = me.i, t, v;
          me.w = w = w + 1640531527 | 0;
          v = X[i + 34 & 127];
          t = X[i = i + 1 & 127];
          v ^= v << 13;
          t ^= t << 17;
          v ^= v >>> 15;
          t ^= t >>> 12;
          v = X[i] = v ^ t;
          me.i = i;
          return v + (w ^ w >>> 16) | 0;
        };
        function init(me2, seed2) {
          var t, v, i, j, w, X = [], limit = 128;
          if (seed2 === (seed2 | 0)) {
            v = seed2;
            seed2 = null;
          } else {
            seed2 = seed2 + "\0";
            v = 0;
            limit = Math.max(limit, seed2.length);
          }
          for (i = 0, j = -32; j < limit; ++j) {
            if (seed2) v ^= seed2.charCodeAt((j + 32) % seed2.length);
            if (j === 0) w = v;
            v ^= v << 10;
            v ^= v >>> 15;
            v ^= v << 4;
            v ^= v >>> 13;
            if (j >= 0) {
              w = w + 1640531527 | 0;
              t = X[j & 127] ^= v + w;
              i = 0 == t ? i + 1 : 0;
            }
          }
          if (i >= 128) {
            X[(seed2 && seed2.length || 0) & 127] = -1;
          }
          i = 127;
          for (j = 4 * 128; j > 0; --j) {
            v = X[i + 34 & 127];
            t = X[i = i + 1 & 127];
            v ^= v << 13;
            t ^= t << 17;
            v ^= v >>> 15;
            t ^= t >>> 12;
            X[i] = v ^ t;
          }
          me2.w = w;
          me2.X = X;
          me2.i = i;
        }
        init(me, seed);
      }
      function copy(f, t) {
        t.i = f.i;
        t.w = f.w;
        t.X = f.X.slice();
        return t;
      }
      ;
      function impl(seed, opts) {
        if (seed == null) seed = +/* @__PURE__ */ new Date();
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result2 = (top + bot) / (1 << 21);
          } while (result2 === 0);
          return result2;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (state.X) copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define2 && define2.amd) {
        define2(function() {
          return impl;
        });
      } else {
        this.xor4096 = impl;
      }
    })(
      exports,
      // window object or global
      typeof module == "object" && module,
      // present in node.js
      typeof define == "function" && define
      // present with an AMD loader
    );
  }
});

// node_modules/seedrandom/lib/tychei.js
var require_tychei = __commonJS({
  "node_modules/seedrandom/lib/tychei.js"(exports, module) {
    (function(global, module2, define2) {
      function XorGen(seed) {
        var me = this, strseed = "";
        me.next = function() {
          var b = me.b, c = me.c, d = me.d, a = me.a;
          b = b << 25 ^ b >>> 7 ^ c;
          c = c - d | 0;
          d = d << 24 ^ d >>> 8 ^ a;
          a = a - b | 0;
          me.b = b = b << 20 ^ b >>> 12 ^ c;
          me.c = c = c - d | 0;
          me.d = d << 16 ^ c >>> 16 ^ a;
          return me.a = a - b | 0;
        };
        me.a = 0;
        me.b = 0;
        me.c = 2654435769 | 0;
        me.d = 1367130551;
        if (seed === Math.floor(seed)) {
          me.a = seed / 4294967296 | 0;
          me.b = seed | 0;
        } else {
          strseed += seed;
        }
        for (var k = 0; k < strseed.length + 20; k++) {
          me.b ^= strseed.charCodeAt(k) | 0;
          me.next();
        }
      }
      function copy(f, t) {
        t.a = f.a;
        t.b = f.b;
        t.c = f.c;
        t.d = f.d;
        return t;
      }
      ;
      function impl(seed, opts) {
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result2 = (top + bot) / (1 << 21);
          } while (result2 === 0);
          return result2;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (typeof state == "object") copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define2 && define2.amd) {
        define2(function() {
          return impl;
        });
      } else {
        this.tychei = impl;
      }
    })(
      exports,
      typeof module == "object" && module,
      // present in node.js
      typeof define == "function" && define
      // present with an AMD loader
    );
  }
});

// (disabled):crypto
var require_crypto = __commonJS({
  "(disabled):crypto"() {
  }
});

// node_modules/seedrandom/seedrandom.js
var require_seedrandom = __commonJS({
  "node_modules/seedrandom/seedrandom.js"(exports, module) {
    (function(global, pool, math) {
      var width = 256, chunks = 6, digits = 52, rngname = "random", startdenom = math.pow(width, chunks), significance = math.pow(2, digits), overflow = significance * 2, mask = width - 1, nodecrypto;
      function seedrandom2(seed, options, callback) {
        var key = [];
        options = options == true ? { entropy: true } : options || {};
        var shortseed = mixkey(flatten(
          options.entropy ? [seed, tostring(pool)] : seed == null ? autoseed() : seed,
          3
        ), key);
        var arc4 = new ARC4(key);
        var prng = function() {
          var n = arc4.g(chunks), d = startdenom, x = 0;
          while (n < significance) {
            n = (n + x) * width;
            d *= width;
            x = arc4.g(1);
          }
          while (n >= overflow) {
            n /= 2;
            d /= 2;
            x >>>= 1;
          }
          return (n + x) / d;
        };
        prng.int32 = function() {
          return arc4.g(4) | 0;
        };
        prng.quick = function() {
          return arc4.g(4) / 4294967296;
        };
        prng.double = prng;
        mixkey(tostring(arc4.S), pool);
        return (options.pass || callback || function(prng2, seed2, is_math_call, state) {
          if (state) {
            if (state.S) {
              copy(state, arc4);
            }
            prng2.state = function() {
              return copy(arc4, {});
            };
          }
          if (is_math_call) {
            math[rngname] = prng2;
            return seed2;
          } else return prng2;
        })(
          prng,
          shortseed,
          "global" in options ? options.global : this == math,
          options.state
        );
      }
      function ARC4(key) {
        var t, keylen = key.length, me = this, i = 0, j = me.i = me.j = 0, s = me.S = [];
        if (!keylen) {
          key = [keylen++];
        }
        while (i < width) {
          s[i] = i++;
        }
        for (i = 0; i < width; i++) {
          s[i] = s[j = mask & j + key[i % keylen] + (t = s[i])];
          s[j] = t;
        }
        (me.g = function(count) {
          var t2, r = 0, i2 = me.i, j2 = me.j, s2 = me.S;
          while (count--) {
            t2 = s2[i2 = mask & i2 + 1];
            r = r * width + s2[mask & (s2[i2] = s2[j2 = mask & j2 + t2]) + (s2[j2] = t2)];
          }
          me.i = i2;
          me.j = j2;
          return r;
        })(width);
      }
      function copy(f, t) {
        t.i = f.i;
        t.j = f.j;
        t.S = f.S.slice();
        return t;
      }
      ;
      function flatten(obj, depth) {
        var result2 = [], typ = typeof obj, prop;
        if (depth && typ == "object") {
          for (prop in obj) {
            try {
              result2.push(flatten(obj[prop], depth - 1));
            } catch (e) {
            }
          }
        }
        return result2.length ? result2 : typ == "string" ? obj : obj + "\0";
      }
      function mixkey(seed, key) {
        var stringseed = seed + "", smear, j = 0;
        while (j < stringseed.length) {
          key[mask & j] = mask & (smear ^= key[mask & j] * 19) + stringseed.charCodeAt(j++);
        }
        return tostring(key);
      }
      function autoseed() {
        try {
          var out;
          if (nodecrypto && (out = nodecrypto.randomBytes)) {
            out = out(width);
          } else {
            out = new Uint8Array(width);
            (global.crypto || global.msCrypto).getRandomValues(out);
          }
          return tostring(out);
        } catch (e) {
          var browser = global.navigator, plugins = browser && browser.plugins;
          return [+/* @__PURE__ */ new Date(), global, plugins, global.screen, tostring(pool)];
        }
      }
      function tostring(a) {
        return String.fromCharCode.apply(0, a);
      }
      mixkey(math.random(), pool);
      if (typeof module == "object" && module.exports) {
        module.exports = seedrandom2;
        try {
          nodecrypto = require_crypto();
        } catch (ex) {
        }
      } else if (typeof define == "function" && define.amd) {
        define(function() {
          return seedrandom2;
        });
      } else {
        math["seed" + rngname] = seedrandom2;
      }
    })(
      // global: `self` in browsers (including strict mode and web workers),
      // otherwise `this` in Node and other environments
      typeof self !== "undefined" ? self : exports,
      [],
      // pool: entropy pool starts empty
      Math
      // math: package containing random, pow, and seedrandom
    );
  }
});

// node_modules/seedrandom/index.js
var require_seedrandom2 = __commonJS({
  "node_modules/seedrandom/index.js"(exports, module) {
    var alea = require_alea();
    var xor128 = require_xor128();
    var xorwow = require_xorwow();
    var xorshift7 = require_xorshift7();
    var xor4096 = require_xor4096();
    var tychei = require_tychei();
    var sr = require_seedrandom();
    sr.alea = alea;
    sr.xor128 = xor128;
    sr.xorwow = xorwow;
    sr.xorshift7 = xorshift7;
    sr.xor4096 = xor4096;
    sr.tychei = tychei;
    module.exports = sr;
  }
});

// examples/neatenstein/browser-entry/constants.ts
var NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS = 3e3;
var NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS = 1e3;
var NEATENSTEIN_MAP_SIZE = 120;
var NEATENSTEIN_DEFAULT_SEED = 1;

// examples/neatenstein/browser-entry/renderer/map.ts
var LCG_MODULUS = 2147483647;
var LCG_MULTIPLIER = 16807;
var LCG_MIN_NONZERO_STATE = 1;
var INTERIOR_WALL_DENSITY = 0.12;
var CENTRAL_ARENA_CLEARANCE_CELLS = 4;
var FLOOR_CELL = 0;
var WALL_CELL = 1;
function normalizeLcgSeed(seed) {
  let state = Number.isFinite(seed) && seed !== 0 ? seed : LCG_MIN_NONZERO_STATE;
  state = (state % LCG_MODULUS + LCG_MODULUS) % LCG_MODULUS;
  return state === 0 ? LCG_MIN_NONZERO_STATE : state;
}
function nextLcgState(state) {
  return state * LCG_MULTIPLIER % LCG_MODULUS;
}
function lcgStateToUnit(state) {
  return state / LCG_MODULUS;
}
function cellIndex(x, y, side) {
  return y * side + x;
}
function isOutOfBounds(x, y, side) {
  return x < 0 || x >= side || y < 0 || y >= side;
}
function writePerimeterWalls(map, side) {
  for (let x = 0; x < side; x++) {
    map[cellIndex(x, 0, side)] = WALL_CELL;
    map[cellIndex(x, side - 1, side)] = WALL_CELL;
  }
  for (let y = 0; y < side; y++) {
    map[cellIndex(0, y, side)] = WALL_CELL;
    map[cellIndex(side - 1, y, side)] = WALL_CELL;
  }
}
function scatterInteriorWalls(map, side, seed) {
  let state = normalizeLcgSeed(seed);
  for (let x = 1; x < side - 1; x++) {
    for (let y = 1; y < side - 1; y++) {
      state = nextLcgState(state);
      if (lcgStateToUnit(state) < INTERIOR_WALL_DENSITY) {
        map[cellIndex(x, y, side)] = WALL_CELL;
      }
    }
  }
}
function carveCentralArena(map, side) {
  const center = Math.floor(side / 2);
  const min = center - CENTRAL_ARENA_CLEARANCE_CELLS;
  const max = center + CENTRAL_ARENA_CLEARANCE_CELLS;
  for (let x = min; x <= max; x++) {
    for (let y = min; y <= max; y++) {
      map[cellIndex(x, y, side)] = FLOOR_CELL;
    }
  }
}
function buildNeatensteinMap(seed) {
  const side = NEATENSTEIN_MAP_SIZE;
  const map = new Uint8Array(side * side);
  writePerimeterWalls(map, side);
  scatterInteriorWalls(map, side, seed);
  carveCentralArena(map, side);
  return map;
}
function createCollisionMap(flatMap, side) {
  if (!Number.isInteger(side) || side <= 0) {
    throw new Error("Invalid map dimensions: expected a square Uint8Array.");
  }
  return {
    isSolid(x, y) {
      if (isOutOfBounds(x, y, side)) {
        return true;
      }
      return flatMap[cellIndex(x, y, side)] !== FLOOR_CELL;
    }
  };
}

// examples/neatenstein/browser-entry/renderer/framebuffer.ts
var NEATENSTEIN_RENDER_DISTANCE_CAP = 30;

// examples/neatenstein/browser-entry/renderer/raycast.ts
var RAY_DIRECTION_EPSILON = 1e-9;
function isNearlyZero(value) {
  return Math.abs(value) < RAY_DIRECTION_EPSILON;
}
function computePerpendicularWallDistance(mapX, mapY, posX, posY, dirX, dirY, stepX, stepY, side) {
  return side === 0 ? (mapX - posX + (1 - stepX) / 2) / dirX : (mapY - posY + (1 - stepY) / 2) / dirY;
}
function castRayDDAFromFlatMap(flatMap, side, posX, posY, dirX, dirY) {
  let mapX = Math.floor(posX);
  let mapY = Math.floor(posY);
  const stepX = dirX >= 0 ? 1 : -1;
  const stepY = dirY >= 0 ? 1 : -1;
  const deltaDistX = isNearlyZero(dirX) ? Number.POSITIVE_INFINITY : Math.abs(1 / dirX);
  const deltaDistY = isNearlyZero(dirY) ? Number.POSITIVE_INFINITY : Math.abs(1 / dirY);
  let sideDistX = stepX > 0 ? (mapX + 1 - posX) * deltaDistX : (posX - mapX) * deltaDistX;
  let sideDistY = stepY > 0 ? (mapY + 1 - posY) * deltaDistY : (posY - mapY) * deltaDistY;
  let sideHit;
  let steps = 0;
  while (true) {
    if (sideDistX < sideDistY) {
      sideHit = 0;
      sideDistX += deltaDistX;
      mapX += stepX;
    } else {
      sideHit = 1;
      sideDistY += deltaDistY;
      mapY += stepY;
    }
    steps += 1;
    if (flatMap[mapY * side + mapX] !== 0) {
      return {
        perpWallDist: computePerpendicularWallDistance(
          mapX,
          mapY,
          posX,
          posY,
          dirX,
          dirY,
          stepX,
          stepY,
          sideHit
        ),
        side: sideHit,
        mapX,
        mapY
      };
    }
    if (steps >= NEATENSTEIN_RENDER_DISTANCE_CAP) {
      return {
        perpWallDist: Number.POSITIVE_INFINITY,
        side: sideHit,
        mapX,
        mapY
      };
    }
  }
}

// examples/neatenstein/browser-entry/host/game/constants.ts
var NEATENSTEIN_FIXED_TIMESTEP_MS = 16;
var NEATENSTEIN_MS_PER_SECOND = 1e3;
var NEATENSTEIN_PLAYER_MAX_HEALTH = 100;
var NEATENSTEIN_PLAYER_MAX_AMMO = 50;
var NEATENSTEIN_ENEMY_MAX_CONCURRENT = 8;
var NEATENSTEIN_DASH_INVULNERABILITY_MS = 200;
var NEATENSTEIN_DASH_COOLDOWN_MS = 500;
var NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS = 2e4;
var NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND = 6;
var NEATENSTEIN_PLAYER_RADIUS_CELLS = 0.25;
var NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS = 96 / 252;
var NEATENSTEIN_CONTACT_RANGE_CELLS = 0.5;
var NEATENSTEIN_CONTACT_DAMAGE = 10;
var NEATENSTEIN_CONTACT_IFRAME_MS = 500;
var NEATENSTEIN_SPAWN_CENTER_X = Math.floor(NEATENSTEIN_MAP_SIZE / 2) + 0.5;
var NEATENSTEIN_SPAWN_CENTER_Y = Math.floor(NEATENSTEIN_MAP_SIZE / 2) + 0.5;
var NEATENSTEIN_MUZZLE_OFFSET_CELLS = 0.2;
var NEATENSTEIN_BOLT_DAMAGE = 20;
var NEATENSTEIN_ENEMY_MAX_HEALTH = 100;
var NEATENSTEIN_ENEMY_STUN_DURATION_MS = 200;
var NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS = 1;
var NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND = 36;
var NEATENSTEIN_BOLT_TRAVEL_DURATION_MS = 300;
var NEATENSTEIN_BOLT_HIT_RADIUS_CELLS = NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS;
var NEATENSTEIN_BOLT_MAX_RANGE_CELLS = 30;
var NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX = 8;
var NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND = 480;
var NEATENSTEIN_TEST_ENEMY_BEYOND_CONTACT_RANGE_CELLS = NEATENSTEIN_CONTACT_RANGE_CELLS + 1;
var NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS = 2e3;
var NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS = 30;
var NEATENSTEIN_ENEMY_BOLT_HIT_RADIUS_CELLS = 0.5;
var NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT = 40;
var NEATENSTEIN_AMMO_PICKUP_AMOUNT = 5;
var NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS = 1e4;
var NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS = 1.5;

// examples/neatenstein/browser-entry/host/game/state.ts
var import_seedrandom = __toESM(require_seedrandom2(), 1);

// examples/neatenstein/browser-entry/renderer/gun.ts
function createInitialGunState() {
  return { recoilOffset: 0 };
}

// examples/neatenstein/browser-entry/host/game/state.ts
function createGameRng(seed) {
  return (0, import_seedrandom.default)(String(seed));
}
function createGameState(options = {}) {
  const seed = options.seed ?? NEATENSTEIN_DEFAULT_SEED;
  const rng = createGameRng(seed);
  const spawnPosition = {
    x: NEATENSTEIN_SPAWN_CENTER_X,
    y: NEATENSTEIN_SPAWN_CENTER_Y
  };
  const player = {
    position: { ...spawnPosition },
    previousPosition: { ...spawnPosition },
    angleRad: rng() * 2 * Math.PI,
    health: NEATENSTEIN_PLAYER_MAX_HEALTH,
    maxHealth: NEATENSTEIN_PLAYER_MAX_HEALTH,
    ammo: NEATENSTEIN_PLAYER_MAX_AMMO,
    maxAmmo: NEATENSTEIN_PLAYER_MAX_AMMO,
    dashTimeRemainingMs: 0,
    dashCooldownMs: 0,
    contactIFrameMs: 0
  };
  const enemies = [];
  return {
    seed,
    simTimeMs: 0,
    episodeTimeMs: 0,
    episodeDurationMs: NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS,
    player,
    enemies,
    impacts: [],
    gun: createInitialGunState(),
    bolts: [],
    enemyBolts: [],
    kills: 0,
    deaths: 0,
    spawnCount: 0,
    generation: 1,
    ammoPickups: []
  };
}
function isInvulnerable(state) {
  return state.player.dashTimeRemainingMs > 0 || (state.player.contactIFrameMs ?? 0) > 0;
}
function canDash(state) {
  return state.player.dashCooldownMs <= 0;
}
function applyDamage(state, amount) {
  if (isInvulnerable(state)) {
    return { ...state, player: { ...state.player } };
  }
  const clamped = Math.max(0, amount);
  return {
    ...state,
    player: {
      ...state.player,
      health: Math.max(0, state.player.health - clamped)
    }
  };
}
function consumeAmmo(state) {
  return {
    ...state,
    player: {
      ...state.player,
      ammo: Math.max(0, state.player.ammo - 1)
    }
  };
}
function restoreAmmo(state, amount) {
  return {
    ...state,
    player: {
      ...state.player,
      ammo: Math.min(state.player.maxAmmo, state.player.ammo + amount)
    }
  };
}
function applyDash(state) {
  if (!canDash(state)) {
    return { ...state, player: { ...state.player } };
  }
  return {
    ...state,
    player: {
      ...state.player,
      dashTimeRemainingMs: NEATENSTEIN_DASH_INVULNERABILITY_MS,
      dashCooldownMs: NEATENSTEIN_DASH_COOLDOWN_MS
    }
  };
}

// examples/neatenstein/browser-entry/host/game/combat.ts
var cachedMapSeed = null;
var cachedFlatMap = null;
function resolveCombatMap(seed) {
  if (cachedFlatMap !== null && cachedMapSeed === seed) {
    return cachedFlatMap;
  }
  cachedMapSeed = seed;
  cachedFlatMap = buildNeatensteinMap(seed);
  return cachedFlatMap;
}
function pointAlongRay(origin, direction, distance) {
  return {
    x: origin.x + direction.x * distance,
    y: origin.y + direction.y * distance
  };
}
function fireBolt(state) {
  if (state.player.ammo <= 0) {
    return { state, fired: false, bolt: null };
  }
  const direction = {
    x: Math.cos(state.player.angleRad),
    y: Math.sin(state.player.angleRad)
  };
  const origin = {
    x: state.player.position.x + direction.x * NEATENSTEIN_MUZZLE_OFFSET_CELLS,
    y: state.player.position.y + direction.y * NEATENSTEIN_MUZZLE_OFFSET_CELLS
  };
  const flatMap = resolveCombatMap(state.seed);
  const wallHit = castRayDDAFromFlatMap(
    flatMap,
    NEATENSTEIN_MAP_SIZE,
    origin.x,
    origin.y,
    direction.x,
    direction.y
  );
  const rawWallDistance = Number.isFinite(wallHit.perpWallDist) ? wallHit.perpWallDist : Number.POSITIVE_INFINITY;
  let hitType = rawWallDistance <= NEATENSTEIN_BOLT_MAX_RANGE_CELLS ? "wall" : "range";
  let hitDistance = Math.min(rawWallDistance, NEATENSTEIN_BOLT_MAX_RANGE_CELLS);
  let hitEnemyIndex = -1;
  for (let index = 0; index < state.enemies.length; index += 1) {
    const enemy = state.enemies[index];
    if (enemy.health <= 0 || enemy.active === false) {
      continue;
    }
    const distanceAlongBolt = projectOntoRay(origin, direction, enemy.position);
    if (distanceAlongBolt <= 0 || distanceAlongBolt > hitDistance) {
      continue;
    }
    const missDistance = perpendicularDistance(
      origin,
      direction,
      enemy.position,
      distanceAlongBolt
    );
    if (missDistance <= NEATENSTEIN_BOLT_HIT_RADIUS_CELLS) {
      hitType = "enemy";
      hitDistance = distanceAlongBolt;
      hitEnemyIndex = index;
    }
  }
  const hit = pointAlongRay(origin, direction, hitDistance);
  let nextState = consumeAmmo(state);
  const bolt = {
    position: { ...origin },
    direction: { ...direction },
    speedCellsPerSecond: NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
    active: true,
    createdAtMs: state.simTimeMs,
    origin: { ...origin },
    targetDistance: Math.max(0, hitDistance),
    radius: NEATENSTEIN_BOLT_HIT_RADIUS_CELLS,
    hitEnemyIndex
  };
  nextState = {
    ...nextState,
    bolts: [...nextState.bolts ?? [], bolt]
  };
  if (hitType === "wall") {
    const wallHitCoordinate = wallHit.side === 0 ? origin.y + wallHit.perpWallDist * direction.y : origin.x + wallHit.perpWallDist * direction.x;
    const impact = {
      wallHit: {
        mapX: wallHit.mapX,
        mapY: wallHit.mapY,
        side: wallHit.side,
        wallX: wallHitCoordinate - Math.floor(wallHitCoordinate)
      },
      position: { ...hit },
      createdAtMs: state.simTimeMs,
      lifetimeMs: NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS,
      perpWallDist: wallHit.perpWallDist,
      boltTravelTimeMs: NEATENSTEIN_BOLT_TRAVEL_DURATION_MS
    };
    nextState = {
      ...nextState,
      impacts: [...nextState.impacts, impact]
    };
  }
  if (hitType === "enemy" && hitEnemyIndex >= 0) {
    const enemy = state.enemies[hitEnemyIndex];
    const enemyImpact = {
      position: { ...enemy.position },
      createdAtMs: state.simTimeMs,
      lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
      boltTravelTimeMs: NEATENSTEIN_BOLT_TRAVEL_DURATION_MS
    };
    nextState = {
      ...nextState,
      enemyImpacts: [...nextState.enemyImpacts ?? [], enemyImpact].slice(
        -NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT
      )
    };
    nextState = applyEnemyDamage(nextState, hitEnemyIndex);
  }
  return {
    state: nextState,
    fired: true,
    bolt
  };
}
function projectOntoRay(origin, direction, point) {
  return (point.x - origin.x) * direction.x + (point.y - origin.y) * direction.y;
}
function perpendicularDistance(origin, direction, point, t) {
  const closestX = origin.x + direction.x * t;
  const closestY = origin.y + direction.y * t;
  return Math.hypot(point.x - closestX, point.y - closestY);
}
function applyEnemyDamage(state, enemyIndex) {
  const enemy = state.enemies[enemyIndex];
  if ((enemy.stunTimerMs ?? 0) > 0) {
    return state;
  }
  const newHealth = Math.max(0, enemy.health - NEATENSTEIN_BOLT_DAMAGE);
  const killedByThisShot = newHealth === 0;
  let newPosition = { ...enemy.position };
  let stunTimerMs = 0;
  if (!killedByThisShot) {
    stunTimerMs = NEATENSTEIN_ENEMY_STUN_DURATION_MS;
    const dx = enemy.position.x - state.player.position.x;
    const dy = enemy.position.y - state.player.position.y;
    const dist = Math.hypot(dx, dy);
    if (dist > 0) {
      const pushX = enemy.position.x + dx / dist * NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS;
      const pushY = enemy.position.y + dy / dist * NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS;
      const flatMap = resolveCombatMap(state.seed);
      const cellX = Math.floor(pushX);
      const cellY = Math.floor(pushY);
      if (cellX >= 0 && cellX < NEATENSTEIN_MAP_SIZE && cellY >= 0 && cellY < NEATENSTEIN_MAP_SIZE && flatMap[cellY * NEATENSTEIN_MAP_SIZE + cellX] === 0) {
        newPosition = { x: pushX, y: pushY };
      }
    }
  }
  const newEnemies = state.enemies.map(
    (existing, index) => index === enemyIndex ? { ...existing, health: newHealth, position: newPosition, stunTimerMs } : existing
  );
  const newAmmoPickups = killedByThisShot ? [
    ...state.ammoPickups ?? [],
    {
      position: { ...enemy.position },
      amount: NEATENSTEIN_AMMO_PICKUP_AMOUNT,
      active: true,
      createdAtMs: state.simTimeMs
    }
  ] : state.ammoPickups ?? [];
  return {
    ...state,
    enemies: newEnemies,
    kills: killedByThisShot ? state.kills + 1 : state.kills,
    ammoPickups: newAmmoPickups
  };
}

// examples/neatenstein/browser-entry/host/game/collision.ts
var MIN_PLAYER_HEALTH = 0;
var MIN_CONTACT_IFRAME_MS = 0;
function isFiniteNumber(value) {
  return Number.isFinite(value);
}
function isFinitePosition(position) {
  return isFiniteNumber(position.x) && isFiniteNumber(position.y);
}
function resolveElapsedMs(dtMs) {
  return isFiniteNumber(dtMs) && dtMs > 0 ? dtMs : 0;
}
function resolveNonNegativeTimerMs(value) {
  return typeof value === "number" && isFiniteNumber(value) && value > 0 ? value : MIN_CONTACT_IFRAME_MS;
}
function tickContactIFrame(currentIFrameMs, dtMs) {
  const current = resolveNonNegativeTimerMs(currentIFrameMs);
  const elapsed = resolveElapsedMs(dtMs);
  return Math.max(MIN_CONTACT_IFRAME_MS, current - elapsed);
}
function squaredDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}
function isLivingContactEnemy(enemy) {
  return enemy.health > 0 && enemy.active !== false && isFinitePosition(enemy.position);
}
function isPlayerTouchingLivingEnemy(state) {
  const playerPosition = state.player.position;
  if (!isFinitePosition(playerPosition)) {
    return false;
  }
  const contactRange = isFiniteNumber(NEATENSTEIN_CONTACT_RANGE_CELLS) && NEATENSTEIN_CONTACT_RANGE_CELLS > 0 ? NEATENSTEIN_CONTACT_RANGE_CELLS : 0;
  const contactRangeSquared = contactRange * contactRange;
  return state.enemies.some((enemy) => {
    if (!isLivingContactEnemy(enemy)) {
      return false;
    }
    return squaredDistance(enemy.position, playerPosition) <= contactRangeSquared;
  });
}
function applyContactDamage(state) {
  return {
    ...state,
    player: {
      ...state.player,
      health: Math.max(
        MIN_PLAYER_HEALTH,
        state.player.health - NEATENSTEIN_CONTACT_DAMAGE
      ),
      contactIFrameMs: NEATENSTEIN_CONTACT_IFRAME_MS
    }
  };
}
function withContactIFrame(state, nextIFrameMs) {
  return {
    ...state,
    player: {
      ...state.player,
      contactIFrameMs: nextIFrameMs
    }
  };
}
function resolveContactDamage(state, dtMs) {
  const nextIFrameMs = tickContactIFrame(state.player.contactIFrameMs, dtMs);
  const next = withContactIFrame(state, nextIFrameMs);
  if (next.player.health <= MIN_PLAYER_HEALTH || isInvulnerable(next)) {
    return next;
  }
  if (!isPlayerTouchingLivingEnemy(next)) {
    return next;
  }
  return applyContactDamage(next);
}

// examples/neatenstein/browser-entry/host/game/waves.ts
var SPAWN_EDGE_ORDER = [
  { direction: "N", dx: 0, dy: 1 },
  { direction: "NW", dx: 1, dy: 1 },
  { direction: "W", dx: 1, dy: 0 },
  { direction: "SW", dx: 1, dy: -1 },
  { direction: "S", dx: 0, dy: -1 },
  { direction: "SE", dx: -1, dy: -1 },
  { direction: "E", dx: -1, dy: 0 },
  { direction: "NE", dx: -1, dy: 1 }
];
function resolveEdgeSpawn(directionIndex, collisionMap) {
  const edge = SPAWN_EDGE_ORDER[directionIndex % SPAWN_EDGE_ORDER.length];
  const edgeOffset = 0.5;
  const centerX = NEATENSTEIN_SPAWN_CENTER_X;
  const centerY = NEATENSTEIN_SPAWN_CENTER_Y;
  const max = NEATENSTEIN_MAP_SIZE - edgeOffset;
  let x;
  let y;
  switch (edge.direction) {
    case "N":
      x = centerX;
      y = edgeOffset;
      break;
    case "S":
      x = centerX;
      y = max;
      break;
    case "W":
      x = edgeOffset;
      y = centerY;
      break;
    case "E":
      x = max;
      y = centerY;
      break;
    case "NW":
      x = edgeOffset;
      y = edgeOffset;
      break;
    case "NE":
      x = max;
      y = edgeOffset;
      break;
    case "SW":
      x = edgeOffset;
      y = max;
      break;
    case "SE":
    default:
      x = max;
      y = max;
      break;
  }
  if (!collisionMap) {
    return { x, y };
  }
  const limit = Math.floor(NEATENSTEIN_MAP_SIZE / 2);
  for (let step = 0; step <= limit; step += 1) {
    const cx = Math.floor(x);
    const cy = Math.floor(y);
    if (cx >= 0 && cy >= 0 && cx < NEATENSTEIN_MAP_SIZE && cy < NEATENSTEIN_MAP_SIZE && !collisionMap.isSolid(cx, cy)) {
      return { x, y };
    }
    x += edge.dx;
    y += edge.dy;
  }
  return { x: centerX, y: centerY };
}
function allEnemiesCleared(enemies) {
  return enemies.length === 0 || enemies.every((enemy) => (enemy.health ?? 0) <= 0 || enemy.active === false);
}
function spawnWaveTick(state, _dtMs, collisionMap) {
  void _dtMs;
  const aliveCount = state.enemies.filter(
    (enemy2) => (enemy2.health ?? 0) > 0 && enemy2.active !== false
  ).length;
  const batchComplete = state.spawnCount > 0 && state.spawnCount % NEATENSTEIN_ENEMY_MAX_CONCURRENT === 0;
  if (batchComplete && !allEnemiesCleared(state.enemies)) {
    return {
      spawnedThisTick: 0,
      state: { ...state }
    };
  }
  if (aliveCount >= NEATENSTEIN_ENEMY_MAX_CONCURRENT) {
    return {
      spawnedThisTick: 0,
      state: { ...state }
    };
  }
  const rng = createGameRng(state.seed + state.spawnCount);
  void rng();
  const directionIndex = state.spawnCount % SPAWN_EDGE_ORDER.length;
  const position = resolveEdgeSpawn(directionIndex, collisionMap);
  const enemy = {
    position: { ...position },
    health: NEATENSTEIN_ENEMY_MAX_HEALTH,
    maxHealth: NEATENSTEIN_ENEMY_MAX_HEALTH,
    active: true,
    controllerPosition: { ...position },
    stunTimerMs: 0
  };
  return {
    spawnedThisTick: 1,
    state: {
      ...state,
      enemies: [...state.enemies, enemy],
      spawnCount: state.spawnCount + 1
    }
  };
}

// examples/neatenstein/browser-entry/host/game/episode.ts
var MIN_TIMESTEP_MS = 1;
function isFiniteNumber2(value) {
  return Number.isFinite(value);
}
function resolveEpisodeTimestepMs(dtMs) {
  return isFiniteNumber2(dtMs) && dtMs >= MIN_TIMESTEP_MS ? dtMs : NEATENSTEIN_FIXED_TIMESTEP_MS;
}
function tickTimerMs(currentMs, dtMs) {
  const current = isFiniteNumber2(currentMs) && currentMs > 0 ? currentMs : 0;
  return Math.max(0, current - dtMs);
}
function advanceEpisodeTimers(state, dtMs) {
  const simTimeMs = isFiniteNumber2(state.simTimeMs) ? state.simTimeMs : 0;
  const episodeTimeMs = isFiniteNumber2(state.episodeTimeMs) ? state.episodeTimeMs : 0;
  return {
    ...state,
    simTimeMs: simTimeMs + dtMs,
    episodeTimeMs: episodeTimeMs + dtMs,
    player: {
      ...state.player,
      dashTimeRemainingMs: tickTimerMs(state.player.dashTimeRemainingMs, dtMs),
      dashCooldownMs: tickTimerMs(state.player.dashCooldownMs, dtMs)
    }
  };
}
function updateEpisode(state, dtMs, collisionMap) {
  const resolvedDtMs = resolveEpisodeTimestepMs(dtMs);
  let next = advanceEpisodeTimers(state, resolvedDtMs);
  const spawnResult = spawnWaveTick(next, resolvedDtMs, collisionMap);
  next = spawnResult.state;
  next = resolveContactDamage(next, resolvedDtMs);
  return next;
}

// examples/neatenstein/browser-entry/host/game/movement.ts
var NEATENSTEIN_COLLISION_EDGE_EPSILON = 1e-9;
var ZERO_VECTOR = { x: 0, y: 0 };
function isFiniteNumber3(value) {
  return Number.isFinite(value);
}
function resolveMovementTimestepMs(dtMs) {
  return isFiniteNumber3(dtMs) && dtMs > 0 ? dtMs : NEATENSTEIN_FIXED_TIMESTEP_MS;
}
function isFiniteVector(vector) {
  return isFiniteNumber3(vector.x) && isFiniteNumber3(vector.y);
}
function normalizeMoveVector(vector) {
  if (!isFiniteVector(vector)) {
    return { ...ZERO_VECTOR };
  }
  const length = Math.hypot(vector.x, vector.y);
  if (!isFiniteNumber3(length) || length === 0) {
    return { ...ZERO_VECTOR };
  }
  return {
    x: vector.x / length,
    y: vector.y / length
  };
}
function movePlayer(state, delta, dtMs = NEATENSTEIN_FIXED_TIMESTEP_MS) {
  const move = normalizeMoveVector(delta);
  if (move.x === 0 && move.y === 0) {
    return state;
  }
  const resolvedDtMs = resolveMovementTimestepMs(dtMs);
  const previousPosition = { ...state.player.position };
  const stepDistance = NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND * (resolvedDtMs / NEATENSTEIN_MS_PER_SECOND);
  return {
    ...state,
    player: {
      ...state.player,
      previousPosition,
      position: {
        x: state.player.position.x + move.x * stepDistance,
        y: state.player.position.y + move.y * stepDistance
      }
    }
  };
}
function resolveWallCollision(state, collisionMap) {
  if (!isPositionBlocked(state.player.position, collisionMap, state.enemies)) {
    return state;
  }
  const previous = state.player.previousPosition ?? state.player.position;
  const xOnly = {
    x: state.player.position.x,
    y: previous.y
  };
  if (!isPositionBlocked(xOnly, collisionMap, state.enemies)) {
    return updatePlayerPosition(state, xOnly, previous);
  }
  const yOnly = {
    x: previous.x,
    y: state.player.position.y
  };
  if (!isPositionBlocked(yOnly, collisionMap, state.enemies)) {
    return updatePlayerPosition(state, yOnly, previous);
  }
  return updatePlayerPosition(state, previous, previous);
}
function updatePlayerMovement(state, movement, collisionMap, dtMs = NEATENSTEIN_FIXED_TIMESTEP_MS) {
  const forward = movement.forward ? 1 : 0;
  const backward = movement.backward ? 1 : 0;
  const left = movement.left ? 1 : 0;
  const right = movement.right ? 1 : 0;
  const localForward = forward - backward;
  const localRight = right - left;
  if (localForward === 0 && localRight === 0) {
    return state;
  }
  const yaw = isFiniteNumber3(state.player.angleRad) ? state.player.angleRad : 0;
  const forwardX = Math.cos(yaw);
  const forwardY = Math.sin(yaw);
  const rightX = -Math.sin(yaw);
  const rightY = Math.cos(yaw);
  const delta = normalizeMoveVector({
    x: localForward * forwardX + localRight * rightX,
    y: localForward * forwardY + localRight * rightY
  });
  const moved = movePlayer(state, delta, dtMs);
  return resolveWallCollision(moved, collisionMap);
}
function isPositionBlocked(position, collisionMap, enemies = []) {
  if (!isFiniteVector(position)) {
    return true;
  }
  const radius = NEATENSTEIN_PLAYER_RADIUS_CELLS;
  const minX = Math.floor(position.x - radius);
  const minY = Math.floor(position.y - radius);
  const maxX = Math.floor(
    position.x + radius - NEATENSTEIN_COLLISION_EDGE_EPSILON
  );
  const maxY = Math.floor(
    position.y + radius - NEATENSTEIN_COLLISION_EDGE_EPSILON
  );
  for (let x = minX; x <= maxX; x += 1) {
    for (let y = minY; y <= maxY; y += 1) {
      if (collisionMap.isSolid(x, y)) {
        return true;
      }
    }
  }
  const playerRadius = radius;
  const enemyRadius = NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS;
  const combinedRadius = playerRadius + enemyRadius;
  const combinedRadiusSquared = combinedRadius * combinedRadius;
  for (const enemy of enemies) {
    if (enemy.health <= 0 || enemy.active === false) {
      continue;
    }
    const enemyPosition = enemy.controllerPosition ?? enemy.position;
    const dx = position.x - enemyPosition.x;
    const dy = position.y - enemyPosition.y;
    if (dx * dx + dy * dy < combinedRadiusSquared) {
      return true;
    }
  }
  return false;
}
function updatePlayerPosition(state, position, previousPosition) {
  return {
    ...state,
    player: {
      ...state.player,
      position,
      previousPosition
    }
  };
}

// examples/neatenstein/browser-entry/host/game/tick.ts
var cachedCollisionSeed = null;
var cachedCollisionMap = null;
function resolveCollisionMap(state, collisionMap) {
  if (collisionMap) {
    return collisionMap;
  }
  if (cachedCollisionMap && cachedCollisionSeed === state.seed) {
    return cachedCollisionMap;
  }
  const flatMap = buildNeatensteinMap(state.seed);
  const nextCollisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);
  cachedCollisionSeed = state.seed;
  cachedCollisionMap = nextCollisionMap;
  return nextCollisionMap;
}
function resolveTickDurationMs(dtMs) {
  return Number.isFinite(dtMs) && dtMs > 0 ? dtMs : NEATENSTEIN_FIXED_TIMESTEP_MS;
}
function normalizeMoveVector2(move) {
  return {
    x: typeof move?.x === "number" && Number.isFinite(move.x) ? move.x : 0,
    y: typeof move?.y === "number" && Number.isFinite(move.y) ? move.y : 0
  };
}
function normalizeGameTickInput(snapshot) {
  return {
    move: normalizeMoveVector2(snapshot.move),
    lookDelta: typeof snapshot.lookDelta === "number" && Number.isFinite(snapshot.lookDelta) ? snapshot.lookDelta : 0,
    fire: snapshot.fire === true,
    dash: snapshot.dash === true
  };
}
function gameTick(state, snapshot, collisionMap, dtMs = NEATENSTEIN_FIXED_TIMESTEP_MS) {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const input = normalizeGameTickInput(snapshot);
  const map = resolveCollisionMap(state, collisionMap);
  let next = updateEpisode(state, resolvedDtMs, map);
  next = applyLook(next, input.lookDelta);
  if (input.dash) {
    next = applyDash(next);
  }
  next = updatePlayerMovement(
    next,
    snapshotToMovement(input.move),
    map,
    resolvedDtMs
  );
  const updatedBolts = updateBolts(
    next.bolts ?? [],
    resolvedDtMs,
    next.simTimeMs,
    map,
    next.enemies
  );
  for (const bolt of updatedBolts) {
    if (!bolt.active && bolt.hitEnemyIndex !== void 0 && bolt.hitEnemyIndex >= 0) {
      const enemy = next.enemies[bolt.hitEnemyIndex];
      if (enemy) {
        const enemyImpact = {
          position: { ...enemy.position },
          createdAtMs: next.simTimeMs,
          lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
          boltTravelTimeMs: 0
        };
        next = {
          ...next,
          enemyImpacts: [...next.enemyImpacts ?? [], enemyImpact].slice(
            -NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT
          )
        };
        next = applyEnemyDamage(next, bolt.hitEnemyIndex);
      }
    }
  }
  next = {
    ...next,
    bolts: updatedBolts.filter((bolt) => bolt.active),
    impacts: ageImpacts(next.impacts, resolvedDtMs),
    enemyImpacts: ageEnemyImpacts(next.enemyImpacts ?? [], resolvedDtMs),
    gun: decayGunRecoil(next.gun ?? { recoilOffset: 0 }, resolvedDtMs)
  };
  const enemyBoltResult = updateEnemyBolts(
    next.enemyBolts ?? [],
    resolvedDtMs,
    next.simTimeMs,
    map,
    next
  );
  next = enemyBoltResult.state;
  next = {
    ...next,
    enemyBolts: enemyBoltResult.bolts.filter((bolt) => bolt.active)
  };
  if (next.player.health <= 0) {
    next = {
      ...next,
      player: {
        ...next.player,
        position: {
          x: NEATENSTEIN_SPAWN_CENTER_X,
          y: NEATENSTEIN_SPAWN_CENTER_Y
        },
        previousPosition: {
          x: NEATENSTEIN_SPAWN_CENTER_X,
          y: NEATENSTEIN_SPAWN_CENTER_Y
        },
        health: NEATENSTEIN_PLAYER_MAX_HEALTH,
        ammo: NEATENSTEIN_PLAYER_MAX_AMMO,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
        contactIFrameMs: 0
      },
      deaths: (next.deaths ?? 0) + 1
    };
  }
  next = updateAmmoPickups(next, next.simTimeMs);
  if (input.fire) {
    const fireResult = fireBolt(next);
    next = fireResult.state;
    if (fireResult.fired) {
      next = {
        ...next,
        gun: {
          ...next.gun,
          recoilOffset: NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX
        }
      };
    }
  }
  return next;
}
function updateAmmoPickups(state, simTimeMs) {
  const pickups = state.ammoPickups ?? [];
  if (pickups.length === 0) {
    return state;
  }
  let ammoGain = 0;
  const updatedPickups = pickups.map((pickup) => {
    if (!pickup.active) {
      return pickup;
    }
    const lifetimeMs = pickup.lifetimeMs ?? NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS;
    const expired = simTimeMs - pickup.createdAtMs >= lifetimeMs;
    if (expired) {
      return { ...pickup, active: false };
    }
    const dx = pickup.position.x - state.player.position.x;
    const dy = pickup.position.y - state.player.position.y;
    const dist = Math.hypot(dx, dy);
    if (dist <= NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS) {
      ammoGain += pickup.amount;
      return { ...pickup, active: false };
    }
    return pickup;
  });
  const activePickups = updatedPickups.filter((pickup) => pickup.active);
  let next = {
    ...state,
    ammoPickups: activePickups
  };
  if (ammoGain > 0) {
    next = restoreAmmo(next, ammoGain);
  }
  return next;
}
function applyLook(state, lookDelta) {
  if (!Number.isFinite(lookDelta) || lookDelta === 0) {
    return state;
  }
  return {
    ...state,
    player: {
      ...state.player,
      angleRad: state.player.angleRad + lookDelta
    }
  };
}
function projectOntoBoltRay(origin, direction, point) {
  return (point.x - origin.x) * direction.x + (point.y - origin.y) * direction.y;
}
function perpendicularDistanceToBoltRay(origin, direction, point, distanceAlong) {
  return Math.hypot(
    origin.x + direction.x * distanceAlong - point.x,
    origin.y + direction.y * distanceAlong - point.y
  );
}
function findBoltEnemyImpact(bolt, step, enemies) {
  if (!bolt.origin || enemies.length === 0) {
    return null;
  }
  let best = null;
  for (let index = 0; index < enemies.length; index += 1) {
    const enemy = enemies[index];
    if (enemy.health <= 0 || enemy.active === false || !enemy.position || !Number.isFinite(enemy.position.x) || !Number.isFinite(enemy.position.y)) {
      continue;
    }
    const hitRadius = bolt.radius ?? NEATENSTEIN_BOLT_HIT_RADIUS_CELLS;
    const distanceAlong = projectOntoBoltRay(
      bolt.position,
      bolt.direction,
      enemy.position
    );
    if (distanceAlong < -hitRadius || distanceAlong > step + hitRadius) {
      continue;
    }
    const missDistance = perpendicularDistanceToBoltRay(
      bolt.position,
      bolt.direction,
      enemy.position,
      distanceAlong
    );
    if (missDistance <= (bolt.radius ?? NEATENSTEIN_BOLT_HIT_RADIUS_CELLS)) {
      if (best === null || distanceAlong < best[0]) {
        best = [distanceAlong, index];
      }
    }
  }
  return best;
}
function updateBolts(bolts, dtMs, currentTimeMs, collisionMap, enemies) {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const dtSeconds = resolvedDtMs / 1e3;
  return bolts.filter((bolt) => bolt.active).map((bolt) => {
    const step = bolt.speedCellsPerSecond * dtSeconds;
    const enemyImpact = enemies && enemies.length > 0 ? findBoltEnemyImpact(bolt, step, enemies) : null;
    const nextPosition = enemyImpact ? {
      x: bolt.position.x + bolt.direction.x * enemyImpact[0],
      y: bolt.position.y + bolt.direction.y * enemyImpact[0]
    } : {
      x: bolt.position.x + bolt.direction.x * step,
      y: bolt.position.y + bolt.direction.y * step
    };
    const outOfBounds = nextPosition.x < 0 || nextPosition.x >= NEATENSTEIN_MAP_SIZE || nextPosition.y < 0 || nextPosition.y >= NEATENSTEIN_MAP_SIZE;
    const hitWall = collisionMap ? collisionMap.isSolid(
      Math.floor(nextPosition.x),
      Math.floor(nextPosition.y)
    ) : false;
    const distanceTraveled = bolt.origin && Number.isFinite(bolt.origin.x) && Number.isFinite(bolt.origin.y) ? Math.hypot(
      nextPosition.x - bolt.origin.x,
      nextPosition.y - bolt.origin.y
    ) : 0;
    const beyondMaxRange = distanceTraveled >= NEATENSTEIN_BOLT_MAX_RANGE_CELLS;
    const reachedTarget = bolt.targetDistance !== void 0 && Number.isFinite(bolt.targetDistance) && distanceTraveled >= bolt.targetDistance;
    const elapsedMs = Math.max(0, currentTimeMs - bolt.createdAtMs);
    const travelExpired = elapsedMs >= NEATENSTEIN_BOLT_TRAVEL_DURATION_MS;
    const hitEnemy = enemyImpact !== null;
    const movementStopped = outOfBounds || hitWall || beyondMaxRange || reachedTarget;
    const active = !travelExpired && !hitEnemy;
    const nextPositionFinal = hitEnemy ? nextPosition : movementStopped ? bolt.position : nextPosition;
    return {
      ...bolt,
      position: nextPositionFinal,
      active,
      hitEnemyIndex: enemyImpact ? enemyImpact[1] : bolt.hitEnemyIndex
    };
  });
}
function updateEnemyBolts(bolts, dtMs, currentTimeMs, collisionMap, state) {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const dtSeconds = resolvedDtMs / 1e3;
  let nextState = state;
  const updatedBolts = bolts.filter((bolt) => bolt.active).map((bolt) => {
    const step = bolt.speedCellsPerSecond * dtSeconds;
    const nextPosition = {
      x: bolt.position.x + bolt.direction.x * step,
      y: bolt.position.y + bolt.direction.y * step
    };
    const hitWall = collisionMap ? collisionMap.isSolid(
      Math.floor(nextPosition.x),
      Math.floor(nextPosition.y)
    ) : false;
    const outOfBounds = nextPosition.x < 0 || nextPosition.x >= NEATENSTEIN_MAP_SIZE || nextPosition.y < 0 || nextPosition.y >= NEATENSTEIN_MAP_SIZE;
    const distanceTraveled = bolt.origin && Number.isFinite(bolt.origin.x) && Number.isFinite(bolt.origin.y) ? Math.hypot(
      nextPosition.x - bolt.origin.x,
      nextPosition.y - bolt.origin.y
    ) : 0;
    const beyondMaxRange = distanceTraveled >= NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS;
    const elapsedMs = Math.max(0, currentTimeMs - bolt.createdAtMs);
    const lifetimeExpired = elapsedMs >= NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS;
    const playerDist = Math.hypot(
      nextPosition.x - nextState.player.position.x,
      nextPosition.y - nextState.player.position.y
    );
    const hitPlayer = !hitWall && !outOfBounds && !beyondMaxRange && playerDist <= NEATENSTEIN_ENEMY_BOLT_HIT_RADIUS_CELLS;
    const movementStopped = outOfBounds || hitWall || beyondMaxRange;
    const active = !lifetimeExpired && !hitPlayer;
    const nextPositionFinal = movementStopped ? bolt.position : nextPosition;
    if (hitPlayer && !bolt.hitPlayer) {
      nextState = applyDamage(nextState, bolt.damage);
      nextState = {
        ...nextState,
        player: {
          ...nextState.player,
          contactIFrameMs: NEATENSTEIN_CONTACT_IFRAME_MS
        }
      };
    }
    return {
      ...bolt,
      position: nextPositionFinal,
      active,
      hitPlayer: hitPlayer || bolt.hitPlayer
    };
  });
  return { state: nextState, bolts: updatedBolts };
}
function decayGunRecoil(gun, dtMs) {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const decayPixels = NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND * (resolvedDtMs / 1e3);
  const nextOffset = Math.max(0, gun.recoilOffset - decayPixels);
  return {
    ...gun,
    recoilOffset: Math.min(nextOffset, NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX)
  };
}
function ageImpacts(impacts, dtMs) {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  return impacts.map((impact) => ({
    ...impact,
    lifetimeMs: impact.lifetimeMs - resolvedDtMs
  })).filter((impact) => impact.lifetimeMs > 0);
}
function ageEnemyImpacts(impacts, dtMs) {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  return impacts.map((impact) => ({
    ...impact,
    lifetimeMs: impact.lifetimeMs - resolvedDtMs
  })).filter((impact) => impact.lifetimeMs > 0);
}
function snapshotToMovement(move) {
  const normalizedMove = normalizeMoveVector2(move);
  return {
    forward: normalizedMove.y > 0,
    backward: normalizedMove.y < 0,
    left: normalizedMove.x < 0,
    right: normalizedMove.x > 0
  };
}

// docs/browser-tests/scenarios/neatenstein-spawn-at-corners-smoke.ts
var SCENARIO = "neatenstein-spawn-at-corners-smoke";
var startMs = performance.now();
var result = {
  passed: true,
  browserVisibility: document.visibilityState === "visible" ? "visible-foreground" : document.visibilityState,
  scenario: SCENARIO,
  url: location.href,
  consoleErrors: [],
  notes: "",
  checks: {},
  metrics: {}
};
var originalError = console.error;
console.error = (...args) => {
  result.consoleErrors.push(args.map(String).join(" "));
  originalError.apply(console, args);
};
window.addEventListener("error", (event) => {
  result.consoleErrors.push(event.message ?? String(event.error));
});
window.addEventListener("unhandledrejection", (event) => {
  result.consoleErrors.push(String(event.reason));
});
function fail(check, message) {
  result.passed = false;
  result.checks[check] = false;
  result.notes += `FAIL ${check}: ${message}
`;
}
function pass(check) {
  result.checks[check] = true;
}
function isEdgePosition(p) {
  const edgeMax = NEATENSTEIN_MAP_SIZE - 0.75;
  return p.x <= 0.75 || p.x >= edgeMax || p.y <= 0.75 || p.y >= edgeMax;
}
function isCenterPosition(p) {
  return p.x === NEATENSTEIN_SPAWN_CENTER_X && p.y === NEATENSTEIN_SPAWN_CENTER_Y;
}
function killEnemy(state, index) {
  const enemies = [...state.enemies];
  if (enemies[index]) {
    enemies[index] = { ...enemies[index], health: 0, active: false };
  }
  return { ...state, enemies, kills: (state.kills ?? 0) + 1 };
}
async function runChecks() {
  let state = createGameState({ seed: 42 });
  const spawnPositions = [];
  for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT * 3; i += 1) {
    const tickResult = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
    state = tickResult.state;
    const enemy = state.enemies[state.enemies.length - 1];
    if (tickResult.spawnedThisTick === 1 && enemy) {
      spawnPositions.push(enemy.position);
    }
  }
  const nonEdgePositions = spawnPositions.filter((p) => !isEdgePosition(p));
  if (nonEdgePositions.length > 0) {
    fail("spawn-on-edge", `found ${nonEdgePositions.length} non-edge spawns: ${JSON.stringify(nonEdgePositions)}`);
  } else {
    pass("spawn-on-edge");
  }
  const centerSpawns = spawnPositions.filter((p) => isCenterPosition(p));
  if (centerSpawns.length > 0) {
    fail("spawn-no-center", `spawned at center ${centerSpawns.length} times: ${JSON.stringify(centerSpawns)}`);
  } else {
    pass("spawn-no-center");
  }
  state = createGameState({ seed: 42 });
  for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i += 1) {
    const tickResult = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
    state = tickResult.state;
  }
  let waitTicks = 0;
  while (waitTicks < 50) {
    const tickResult = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
    state = tickResult.state;
    if (tickResult.spawnedThisTick !== 0) {
      fail("batch-wait-alive", `enemy spawned while batch still alive at wait tick ${waitTicks}`);
      break;
    }
    waitTicks += 1;
  }
  if (result.checks["batch-wait-alive"] === void 0) {
    pass("batch-wait-alive");
  }
  for (let i = 0; i < state.enemies.length; i += 1) {
    state = killEnemy(state, i);
  }
  const afterKillTick = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
  if (afterKillTick.spawnedThisTick !== 1) {
    fail("batch-resume-after-clear", `expected 1 spawn after clearing, got ${afterKillTick.spawnedThisTick}`);
  } else {
    pass("batch-resume-after-clear");
  }
  state = createGameState({ seed: 7 });
  state = {
    ...state,
    player: {
      ...state.player,
      position: { x: 1, y: 1 },
      previousPosition: { x: 1, y: 1 },
      health: 0,
      ammo: 0
    }
  };
  state = gameTick(state, { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false });
  if (state.player.health !== NEATENSTEIN_PLAYER_MAX_HEALTH) {
    fail("hero-health-restored", `health=${state.player.health}`);
  } else {
    pass("hero-health-restored");
  }
  if (state.player.ammo !== NEATENSTEIN_PLAYER_MAX_AMMO) {
    fail("hero-ammo-restored", `ammo=${state.player.ammo}`);
  } else {
    pass("hero-ammo-restored");
  }
  if (!isCenterPosition(state.player.position)) {
    fail("hero-spawn-center", `position=${JSON.stringify(state.player.position)}`);
  } else {
    pass("hero-spawn-center");
  }
  if ((state.deaths ?? 0) !== 1) {
    fail("hero-deaths-increment", `deaths=${state.deaths}`);
  } else {
    pass("hero-deaths-increment");
  }
  state = createGameState({ seed: 99 });
  let killCount = 0;
  for (let wave = 0; wave < 12; wave += 1) {
    for (let s = 0; s < NEATENSTEIN_ENEMY_MAX_CONCURRENT; s += 1) {
      const spawn = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
      state = spawn.state;
    }
    const start = state.enemies.length - NEATENSTEIN_ENEMY_MAX_CONCURRENT;
    for (let i = start; i < state.enemies.length; i += 1) {
      state = killEnemy(state, i);
      killCount += 1;
    }
  }
  result.metrics.killCount = killCount;
  result.metrics.enemyRosterSizeAfterKills = state.enemies.length;
  if (killCount < 86) {
    fail("kills-above-86", `killCount=${killCount}`);
  } else {
    pass("kills-above-86");
  }
  const aliveAfterKillLoop = state.enemies.filter((e) => (e.health ?? 0) > 0 && e.active !== false).length;
  if (aliveAfterKillLoop !== 0) {
    fail("all-killed-before-extended-ticks", `${aliveAfterKillLoop} enemies still alive after kill loop`);
  } else {
    pass("all-killed-before-extended-ticks");
  }
  if (!allEnemiesCleared(state.enemies)) {
    fail("all-enemies-cleared-after-kills", "roster not fully cleared after all kills");
  } else {
    pass("all-enemies-cleared-after-kills");
  }
  const spawnCountBeforeExtendedTicks = state.spawnCount;
  for (let i = 0; i < 200; i += 1) {
    state = gameTick(state, { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false });
  }
  result.durationMs = Math.round(performance.now() - startMs);
  result.metrics.spawnCount = state.spawnCount;
  result.metrics.extendedTickCount = 200;
  result.metrics.deaths = state.deaths ?? 0;
  if (state.spawnCount <= spawnCountBeforeExtendedTicks) {
    fail("extended-ticks-spawned", `spawnCount did not increase during extended ticks: ${state.spawnCount}`);
  } else {
    pass("extended-ticks-spawned");
  }
  result.notes += `killCount=${killCount} spawnCount=${state.spawnCount} deaths=${state.deaths}`;
  const status = document.getElementById("status");
  if (status) {
    status.textContent = JSON.stringify(result, null, 2);
  }
  window["neatensteinSpawnAtCornersSmokeResult"] = result;
}
runChecks().catch((err) => {
  result.passed = false;
  result.consoleErrors.push(String(err));
  result.durationMs = Math.round(performance.now() - startMs);
  result.notes += `UNCAUGHT: ${err instanceof Error ? err.message : String(err)}
`;
  const status = document.getElementById("status");
  if (status) {
    status.textContent = JSON.stringify(result, null, 2);
  }
  window["neatensteinSpawnAtCornersSmokeResult"] = result;
});
//# sourceMappingURL=neatenstein-spawn-at-corners-smoke.bundle.mjs.map
