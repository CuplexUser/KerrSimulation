/**
 * The controls reference: every mouse, touch and keyboard binding.
 *
 * Drawn from CONTROL_HELP, which lives next to the handlers in orbitControls.ts,
 * so a binding cannot be added there without the list it belongs to being in
 * reach. Opens from the help link in the panel, the ? button on the stage, or
 * the ? and H keys; Escape, the close button or a click outside dismisses it.
 */

import { useEffect, useRef } from 'react';
import { CONTROL_HELP } from './orbitControls.ts';

type Props = {
  onClose: () => void;
};

/** Clicks inside the card must not reach the backdrop, which closes it. */
const keepOpen = (event: React.PointerEvent) => event.stopPropagation();

export function HelpPanel({ onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  // Move focus into the dialog so Escape and Tab work from the keyboard, and
  // hand it back to wherever it was when the dialog closes.
  useEffect(() => {
    const previous = document.activeElement;
    closeRef.current?.focus();
    return () => {
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, []);

  return (
    <div className="help" onPointerDown={onClose}>
      <div
        className="help__card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-title"
        onPointerDown={keepOpen}
      >
        <header className="help__head">
          <h2 id="help-title" className="help__title">
            Controls
          </h2>
          <button
            ref={closeRef}
            type="button"
            className="section__action"
            onClick={onClose}
          >
            Close
          </button>
        </header>

        {CONTROL_HELP.map(({ group, items }) => (
          <section key={group} className="help__group">
            <h3 className="section__title">{group}</h3>
            <dl className="help__list">
              {items.map(({ keys, action }) => (
                <div key={action} className="help__row">
                  <dt className="help__keys">
                    {keys.map((key, i) => (
                      <span key={key}>
                        {i > 0 ? <span className="help__or">or</span> : null}
                        <kbd>{key}</kbd>
                      </span>
                    ))}
                  </dt>
                  <dd className="help__action">{action}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}

        <p className="help__note">
          The view eases toward each change and refines once it holds still. Keys
          are ignored while a slider or text field has focus, so arrows still adjust
          the focused slider.
        </p>
      </div>
    </div>
  );
}
