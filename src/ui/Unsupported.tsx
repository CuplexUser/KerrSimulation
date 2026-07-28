type Props = {
  reason: string;
};

/** Shown when there is no WebGPU device. Says what happened and what to do. */
export function Unsupported({ reason }: Props) {
  return (
    <main className="unsupported">
      <div className="unsupported__card">
        <h1 className="unsupported__title">WebGPU is not available</h1>
        <p className="unsupported__reason">{reason}</p>
        <ul className="unsupported__steps">
          <li>Open this page in Chrome or Edge 113 or newer.</li>
          <li>
            Turn on hardware acceleration in browser settings, then restart the
            browser.
          </li>
          <li>
            Check <code>chrome://gpu</code> — the WebGPU row should read
            “Hardware accelerated”.
          </li>
        </ul>
      </div>
    </main>
  );
}
