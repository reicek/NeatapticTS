import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default {
  mode: 'development',
  entry: './src/neataptic.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'neataptic.js',
  },
  resolve: {
    extensions: ['.ts', '.js'],
    fallback: {
      child_process: false,
      os: false,
      path: false,
    },
  },
};
