/* Import */
const fs = require('fs');
const path = require('path');
const webpack = require('webpack');

/* Update readme and read license */
const version = require('./package.json').version;
const readme = fs
  .readFileSync('./README.md', 'utf-8')
  .replace(/cdn\/(.*)\/neataptic.js/, `cdn/${version}/neataptic.js`);
fs.writeFileSync('./README.md', readme);

const license = fs.readFileSync('./LICENSE', 'utf-8');

/* Export config */
module.exports = {
  mode: 'development',
  context: __dirname,
  entry: './src/neataptic.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'neataptic.js',
  },
  resolve: {
    fallback: { 
      "os": require.resolve("os-browserify/browser"),
      "path": require.resolve("path-browserify"),
    }
  },
  plugins: [
    new webpack.NoEmitOnErrorsPlugin(),
    new webpack.BannerPlugin(license),
    new webpack.optimize.ModuleConcatenationPlugin(),
  ],
};
