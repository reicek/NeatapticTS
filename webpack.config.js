const path = require('path');

module.exports = {
  mode: 'development',
  entry: './src/neataptic.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'neataptic.js',
  },
};
