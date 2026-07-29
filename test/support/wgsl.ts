/**
 * Just enough WGSL parsing to compare shader source against its TypeScript
 * counterpart. Not a real parser — it reads declarations out of shaders written
 * in the style this project actually uses.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const SHADER_DIR = join(import.meta.dirname, '..', '..', 'src', 'gpu', 'shaders');

export const readShader = (name: string): string =>
  readFileSync(join(SHADER_DIR, `${name}.wgsl`), 'utf8');

/** Strips line comments so declarations inside prose cannot be mistaken for code. */
const stripComments = (source: string): string =>
  source.replaceAll(/\/\/.*$/gm, '');

/**
 * Numeric `const` and `override` declarations, by name.
 *
 * Only literal initializers are captured: a constant defined in terms of others
 * (DISK_FILAMENT_MEAN, say) has no single number to compare against.
 */
export function scalarConstants(source: string): Map<string, number> {
  const found = new Map<string, number>();
  const pattern =
    /\b(?:const|override)\s+(\w+)\s*:\s*f32\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*;/g;

  for (const [, name, value] of stripComments(source).matchAll(pattern)) {
    found.set(name, Number(value));
  }
  return found;
}

/** Names of every declared function, in source order. */
export const functionNames = (source: string): string[] =>
  [...stripComments(source).matchAll(/\bfn\s+(\w+)\s*\(/g)].map(([, name]) => name);

/** Names of every declared struct, in source order. */
export const structNames = (source: string): string[] =>
  [...stripComments(source).matchAll(/\bstruct\s+(\w+)\s*\{/g)].map(([, name]) => name);

export type StructField = { name: string; type: string };

/** Fields of a named struct, in declaration order. Returns null if absent. */
export function structFields(
  source: string,
  structName: string,
): StructField[] | null {
  const match = new RegExp(`\\bstruct\\s+${structName}\\s*\\{([^}]*)\\}`).exec(
    stripComments(source),
  );
  if (!match) return null;

  return match[1]
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [name, type] = entry.split(':').map((part) => part.trim());
      return { name, type };
    });
}

/** Every identifier used anywhere in the source, deduplicated. */
export const identifiers = (source: string): Set<string> =>
  new Set(
    [...stripComments(source).matchAll(/\b[A-Za-z_]\w*\b/g)].map(([token]) => token),
  );
