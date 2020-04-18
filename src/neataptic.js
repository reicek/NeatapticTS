const Neataptic = {
  Node: require('./architecture/node'),
  Neat: require('./neat'),
  multi: require('./multithreading/multi'),
  Group: require('./architecture/group'),
  Layer: require('./architecture/layer'),
  config: require('./config'),
  Network: require('./architecture/network'),
  methods: require('./methods/methods'),
  architect: require('./architecture/architect'),
  Connection: require('./architecture/connection')
};

// CommonJS & AMD
if (typeof define !== 'undefined' && define.amd) {
  define([], () => Neataptic);
}

// Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Neataptic;
}

// Browser
if (typeof window === 'object') {
  (() => {
    const old = window['neataptic'];

    Neataptic.ninja = () => {
      window['neataptic'] = old;

      return Neataptic;
    };
  })();

  window['neataptic'] = Neataptic;
}
