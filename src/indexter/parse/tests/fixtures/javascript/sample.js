import { EventEmitter } from "events";
import defaultExport from "./utils";

class Base extends EventEmitter {}

class Handler extends Base {
  process(items) {
    items.forEach(function onItem(item) {
      return item;
    });
  }
}

function standalone(x) {
  return x * 2;
}

const double = (x) => x * 2;
