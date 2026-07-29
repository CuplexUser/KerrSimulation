/**
 * Teaches Node how to load `import source from './x.wgsl?raw'`.
 *
 * That specifier is a Vite feature, so under plain Node the whole `src/gpu`
 * module graph is unimportable and none of the renderer can be tested headlessly.
 * A pair of module hooks resolves the query suffix and hands back the file text
 * as a default export — exactly what Vite does, minus the bundler.
 *
 * Hooks must be installed before the graph is *linked*, not merely before it is
 * evaluated, so this cannot be a side-effect import inside a test file: ESM
 * resolves every dependency up front. It is loaded via `--import` from the
 * `test` script instead.
 */

import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const RAW_SUFFIX = '?raw';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!specifier.endsWith(RAW_SUFFIX)) return nextResolve(specifier, context);

    const parent = context.parentURL
      ? dirname(fileURLToPath(context.parentURL))
      : process.cwd();
    const target = resolve(parent, specifier.slice(0, -RAW_SUFFIX.length));

    return {
      url: `${pathToFileURL(target).href}${RAW_SUFFIX}`,
      format: 'module',
      shortCircuit: true,
    };
  },

  load(url, context, nextLoad) {
    if (!url.endsWith(RAW_SUFFIX)) return nextLoad(url, context);

    const text = readFileSync(fileURLToPath(url.slice(0, -RAW_SUFFIX.length)), 'utf8');
    return {
      format: 'module',
      shortCircuit: true,
      source: `export default ${JSON.stringify(text)};`,
    };
  },
});
