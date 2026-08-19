// Observable Framework only emits files referenced by a page or loader, so
// assets that exist purely for crawlers and link unfurlers (robots.txt, the
// Open Graph card) never reach dist/. This copies static/ over the build.
import {cp, readdir, writeFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import {dirname, join} from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const from = join(root, "static");
const to = join(root, "dist");

const names = await readdir(from);
await Promise.all(names.map((n) => cp(join(from, n), join(to, n), {recursive: true})));
console.log(`copied ${names.length} static asset(s) -> dist/: ${names.join(", ")}`);

// Generated rather than static so it cannot fall out of step with the page list
// in observablehq.config.js, and so lastmod reflects the deploy.
const {default: config} = await import(join(root, "observablehq.config.js"));
const site = "https://halfcell.pages.dev";
const today = new Date().toISOString().slice(0, 10);
const paths = ["/", ...config.pages.map((p) => p.path)];

const xml = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  ...paths.map((path) => `  <url><loc>${site}${path}</loc><lastmod>${today}</lastmod></url>`),
  "</urlset>",
  "",
].join("\n");

await writeFile(join(to, "sitemap.xml"), xml);
console.log(`wrote sitemap.xml with ${paths.length} urls`);
