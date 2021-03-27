var Neataptic = {
  methods: import('./methods/methods'),
  Connection: import('./architecture/connection'),
  architect: import('./architecture/architect'),
  Network: import('./architecture/network'),
  config: import('./config'),
  Group: import('./architecture/group'),
  Layer: import('./architecture/layer'),
  Node: import('./architecture/node'),
  Neat: import('./neat'),
  multi: import('./multithreading/multi'),
};

// CommonJS & AMD
if (typeof define !== 'undefined' && define.amd) {
  define([], function () {
    return Neataptic;
  });
}

// Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Neataptic;
}

// Browser
if (typeof window === 'object') {
  (function () {
    var old = window['neataptic'];
    Neataptic.ninja = function () {
      window['neataptic'] = old;
      return Neataptic;
    };
  })();

  window['neataptic'] = Neataptic;
}
