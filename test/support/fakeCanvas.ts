/**
 * The smallest thing `attachOrbitControls` will accept as a canvas.
 *
 * Node has EventTarget and Event natively, so the listener plumbing is real —
 * only the handful of DOM surfaces the control actually touches are stubbed.
 * That keeps the test honest about listener registration and removal, which is
 * the half of the module most likely to regress.
 */

export class FakePointerEvent extends Event {
  pointerId: number;
  pointerType: string;
  button: number;
  clientX: number;
  clientY: number;
  shiftKey: boolean;

  constructor(
    type: string,
    init: {
      pointerId?: number;
      pointerType?: string;
      button?: number;
      clientX?: number;
      clientY?: number;
      shiftKey?: boolean;
      /** Overrides the event's own clock, for velocity and fling tests. */
      timeStamp?: number;
    } = {},
  ) {
    super(type, { cancelable: true });
    this.pointerId = init.pointerId ?? 1;
    this.pointerType = init.pointerType ?? 'mouse';
    this.button = init.button ?? 0;
    this.clientX = init.clientX ?? 0;
    this.clientY = init.clientY ?? 0;
    this.shiftKey = init.shiftKey ?? false;
    if (init.timeStamp !== undefined) {
      Object.defineProperty(this, 'timeStamp', { value: init.timeStamp });
    }
  }
}

export class FakeKeyEvent extends Event {
  key: string;
  shiftKey: boolean;
  ctrlKey = false;
  metaKey = false;
  altKey = false;

  constructor(key: string, init: { shiftKey?: boolean } = {}) {
    super('keydown', { cancelable: true });
    this.key = key;
    this.shiftKey = init.shiftKey ?? false;
  }
}

export class FakeWheelEvent extends Event {
  deltaY: number;
  deltaMode: number;

  constructor(init: { deltaY: number; deltaMode?: number }) {
    super('wheel', { cancelable: true });
    this.deltaY = init.deltaY;
    this.deltaMode = init.deltaMode ?? 0;
  }
}

export class FakeCanvas extends EventTarget {
  /** Pointer ids currently captured, so release can be asserted. */
  readonly captured = new Set<number>();
  readonly classes = new Set<string>();

  setPointerCapture(pointerId: number): void {
    this.captured.add(pointerId);
  }

  releasePointerCapture(pointerId: number): void {
    this.captured.delete(pointerId);
  }

  readonly classList = {
    add: (name: string) => this.classes.add(name),
    remove: (name: string) => this.classes.delete(name),
    contains: (name: string) => this.classes.has(name),
  };
}
