export interface Greeter {
  greet(name: string): string;
}

export type Level = "low" | "medium" | "high";

export enum Status {
  Active,
  Inactive,
}

class BaseHandler {
  protected setup(): void {}
}

export class Handler extends BaseHandler implements Greeter {
  greet(name: string): string {
    return `hello ${name}`;
  }
}

export function standalone(x: number): number {
  return x * 2;
}
