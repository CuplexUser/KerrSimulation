import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Window-chrome behavior for the control panel: drag to move, grip to resize,
 * header to collapse, and the whole thing remembered across reloads.
 *
 * Geometry lives in React state and updates on every pointer move. That is the
 * opposite of the rule the orbit camera follows, and the difference is the
 * point: a camera change resets the accumulation and schedules GPU work, so it
 * must never go through React, whereas panel geometry touches nothing but this
 * one subtree. The panel floats over the canvas rather than sharing layout with
 * it, so moving or resizing it does not resize the swap chain and does not cost
 * a single traced sample.
 *
 * Writing straight to the DOM and committing on release was tried first and is
 * wrong here. The panel re-renders on its own while a drag is in progress —
 * renderer stats arrive at about 4 Hz — and every one of those renders reapplies
 * left/top/width/height from committed state, so the panel snaps back under the
 * pointer. Anything that puts geometry in the style prop has to keep that prop
 * authoritative.
 *
 * Below the mobile breakpoint the panel is a full-width sheet positioned
 * entirely by CSS. Inline geometry would fight that, and dragging a sheet around
 * a phone is not useful, so the hook stands down and reports `floating: false`.
 */

export type PanelLayout = {
  x: number;
  y: number;
  width: number;
  height: number;
};

const STORAGE_KEY = 'kerr.panel.layout.v1';
const FLOATING_MIN_VIEWPORT = 736;

const MIN_WIDTH = 264;
const MAX_WIDTH = 560;
const MIN_HEIGHT = 168;
const MARGIN = 16;

const clamp = (value: number, low: number, high: number): number =>
  Math.min(Math.max(value, low), high);

const viewport = () => ({
  width: globalThis.innerWidth || 1440,
  height: globalThis.innerHeight || 900,
});

function defaultLayout(): PanelLayout {
  return {
    x: MARGIN,
    y: MARGIN,
    width: 336,
    height: clamp(viewport().height - MARGIN * 2, MIN_HEIGHT, 760),
  };
}

/**
 * Size first, then position — the position bounds depend on the final size.
 *
 * The panel is kept fully on screen rather than being allowed to hang off an
 * edge the way a desktop window can. Two reasons: the resize grip lives in the
 * bottom-right corner, so letting the panel run past the bottom makes it
 * unreachable and the panel un-resizable; and collapsing is a better answer than
 * shoving it off screen. Clamping always uses the expanded height, even while
 * collapsed, so expanding in place can never drop the panel out of view.
 */
function constrain(layout: PanelLayout): PanelLayout {
  const { width: vw, height: vh } = viewport();

  const width = clamp(
    layout.width,
    MIN_WIDTH,
    Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, vw - MARGIN * 2)),
  );
  const height = clamp(
    layout.height,
    MIN_HEIGHT,
    Math.max(MIN_HEIGHT, vh - MARGIN * 2),
  );

  return {
    width,
    height,
    x: clamp(layout.x, 0, Math.max(0, vw - width)),
    y: clamp(layout.y, 0, Math.max(0, vh - height)),
  };
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

function readStored(): { layout: PanelLayout; collapsed: boolean } {
  const fallback = { layout: defaultLayout(), collapsed: false };
  try {
    const raw = globalThis.localStorage?.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return fallback;

    const { x, y, width, height, collapsed } = parsed;
    if (
      typeof x !== 'number' ||
      typeof y !== 'number' ||
      typeof width !== 'number' ||
      typeof height !== 'number'
    ) {
      return fallback;
    }
    return {
      layout: constrain({ x, y, width, height }),
      collapsed: collapsed === true,
    };
  } catch {
    // A corrupt or unreadable entry is not worth failing startup over.
    return fallback;
  }
}

type GestureMode = 'move' | 'resize';

type Gesture = {
  pointerId: number;
  mode: GestureMode;
  originX: number;
  originY: number;
  /** Layout when the gesture began; every move is measured from here. */
  start: PanelLayout;
};

export function usePanelLayout() {
  const panelRef = useRef<HTMLElement | null>(null);
  const gestureRef = useRef<Gesture | null>(null);

  const [floating, setFloating] = useState(
    () => viewport().width >= FLOATING_MIN_VIEWPORT,
  );
  const [{ layout, collapsed }, setState] = useState(readStored);
  const [gestureMode, setGestureMode] = useState<GestureMode | null>(null);

  // Mirrored so a gesture can read the committed layout at pointerdown without
  // the handler depending on it. Synced in an effect rather than during render,
  // so StrictMode's double-invoke cannot observe a half-updated value.
  const layoutRef = useRef(layout);
  useEffect(() => {
    layoutRef.current = layout;
  }, [layout]);

  useEffect(() => {
    try {
      globalThis.localStorage?.setItem(
        STORAGE_KEY,
        JSON.stringify({ ...layout, collapsed }),
      );
    } catch {
      // Private-mode storage failures are not worth surfacing.
    }
  }, [layout, collapsed]);

  useEffect(() => {
    const onResize = () => {
      setFloating(viewport().width >= FLOATING_MIN_VIEWPORT);
      setState((current) => ({
        ...current,
        layout: constrain(current.layout),
      }));
    };
    globalThis.addEventListener('resize', onResize);
    return () => globalThis.removeEventListener('resize', onResize);
  }, []);

  const beginGesture = useCallback(
    (event: React.PointerEvent, mode: GestureMode) => {
      if (!floating || event.button !== 0) return;
      const node = panelRef.current;
      if (!node) return;

      event.preventDefault();
      node.setPointerCapture(event.pointerId);
      gestureRef.current = {
        pointerId: event.pointerId,
        mode,
        originX: event.clientX,
        originY: event.clientY,
        start: layoutRef.current,
      };
      setGestureMode(mode);
    },
    [floating],
  );

  const onPointerMove = useCallback((event: React.PointerEvent) => {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;

    const dx = event.clientX - gesture.originX;
    const dy = event.clientY - gesture.originY;
    const { start } = gesture;

    setState((current) => ({
      ...current,
      layout:
        gesture.mode === 'move'
          ? constrain({ ...start, x: start.x + dx, y: start.y + dy })
          : constrain({
              ...start,
              width: start.width + dx,
              height: start.height + dy,
            }),
    }));
  }, []);

  const endGesture = useCallback((event: React.PointerEvent) => {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    gestureRef.current = null;
    setGestureMode(null);

    const node = panelRef.current;
    if (node?.hasPointerCapture(event.pointerId)) {
      node.releasePointerCapture(event.pointerId);
    }
  }, []);

  const startMove = useCallback(
    (event: React.PointerEvent) => beginGesture(event, 'move'),
    [beginGesture],
  );
  const startResize = useCallback(
    (event: React.PointerEvent) => beginGesture(event, 'resize'),
    [beginGesture],
  );

  const toggleCollapsed = useCallback(() => {
    setState((current) => ({ ...current, collapsed: !current.collapsed }));
  }, []);

  const resetLayout = useCallback(() => {
    setState({ layout: defaultLayout(), collapsed: false });
  }, []);

  const style: React.CSSProperties | undefined = floating
    ? {
        left: layout.x,
        top: layout.y,
        width: layout.width,
        height: collapsed ? undefined : layout.height,
      }
    : undefined;

  return {
    panelRef,
    floating,
    collapsed,
    gestureMode,
    style,
    startMove,
    startResize,
    onPointerMove,
    endGesture,
    toggleCollapsed,
    resetLayout,
  };
}
