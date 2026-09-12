import { Base } from './base.js';
import { Greeter } from './base';

export class Handler extends Base implements Greeter {
  greet(name: string): string {
    return `hello ${this.format(name)}`;
  }

  private format(name: string): string {
    return name.toUpperCase();
  }
}
