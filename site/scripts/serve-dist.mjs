// Serves the built dist/ the way Cloudflare Pages does, so a local look at the
// deployed site is the deployed site rather than the dev server: real built
// HTML, hashed asset URLs, the inlined head script as shipped.
//
// The one Pages behaviour that matters here is extensionless routing —
// /methodology serves methodology.html. `observable preview` and Pages both do
// it; a plain static server does not, and the pages would 404. Unknown paths
// get 404.html with a real 404 status, as Pages does.
//
// This project ships no _headers or _redirects, so there is nothing else of
// Pages' behaviour to emulate. `npx wrangler pages dev site/dist` is the
// byte-exact option if that ever changes.
import {createServer} from "node:http";
import {readFile, stat} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import {dirname, join, normalize, extname} from "node:path";

const root = join(dirname(dirname(fileURLToPath(import.meta.url))), "dist");
const port = Number(process.argv[2] ?? 4173);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".parquet": "application/vnd.apache.parquet",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".xml": "application/xml; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

async function readIfFile(path) {
  try {
    if (!(await stat(path)).isFile()) return null;
    return await readFile(path);
  } catch {
    return null;
  }
}

// Pages' resolution order for a path with no extension.
async function resolve(pathname) {
  // normalize() collapses ../ before it is joined, so a crafted path cannot
  // escape dist/.
  const rel = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, "");
  const base = join(root, rel);
  if (extname(base)) return readIfFile(base);
  return (
    (await readIfFile(base)) ??
    (await readIfFile(`${base}.html`)) ??
    (await readIfFile(join(base, "index.html")))
  );
}

createServer(async (req, res) => {
  const {pathname} = new URL(req.url, "http://localhost");
  const body = await resolve(pathname === "/" ? "/index.html" : pathname);
  if (body) {
    const ext = extname(pathname) || ".html";
    res.writeHead(200, {"content-type": TYPES[ext] ?? "application/octet-stream"});
    return res.end(body);
  }
  const notFound = await readIfFile(join(root, "404.html"));
  res.writeHead(404, {"content-type": "text/html; charset=utf-8"});
  res.end(notFound ?? "404");
}).listen(port, () => {
  console.log(`serving site/dist as Cloudflare Pages would -> http://localhost:${port}`);
});
