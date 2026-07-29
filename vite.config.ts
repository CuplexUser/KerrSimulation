import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

/**
 * On GitHub Pages a project site is served from /<repo>/, so every asset URL
 * needs that prefix. It is derived from GITHUB_REPOSITORY rather than written
 * out, so renaming or forking the repository does not silently break the
 * deployment — and a user/organization site (<name>.github.io) is served from
 * the root, so that case is excluded.
 *
 * Outside CI the variable is unset and the base stays "/", which keeps
 * `pnpm dev` and `pnpm preview` on plain localhost paths.
 */
function basePath(): string {
  const repository = process.env.GITHUB_REPOSITORY?.split('/')[1];
  if (!repository || repository.endsWith('.github.io')) return '/';
  return `/${repository}/`;
}

export default defineConfig({
  base: basePath(),
  plugins: [react()],
});
