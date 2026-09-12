import multiply from './utils/math.js';
import { add } from './utils';
import * as mathNs from './utils/math';
const handlerModule = require('./components/handler');

function useAll() {
  return add(1, 2) + multiply(2, 3) + mathNs.add(4, 5);
}
