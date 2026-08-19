// Observable Framework only emits files referenced by a page or loader, so
// assets that exist purely for crawlers and link unfurlers (robots.txt, the
// Open Graph card) never reach dist/. This copies static/ over the build.
import {cp, readdir} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import {dirname, join} from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const from = join(root, "static");
const to = join(root, "dist");

const names = await readdir(from);
await Promise.all(names.map((n) => cp(join(from, n), join(to, n), {recursive: true})));
console.log(`copied ${names.length} static asset(s) → dist/: ${names.join(", ")}`);
