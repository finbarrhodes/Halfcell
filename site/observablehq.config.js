// Halfcell — Observable Framework config
// Static build of the GB BESS market analysis previously served via Streamlit.

import {readFileSync} from "node:fs";

const title = "Halfcell";

// Client script inlined into every page's <head>. It cannot be a <script src>:
// Framework only emits files under src/ that a page imports, and static/ is
// copied by the post-build step, which `observable preview` never runs — so a
// referenced file would 404 in dev. Inlining keeps dev and the build identical.
//
// Read per call rather than once at config load: `head` as a function is
// invoked on every page parse, so editing the script shows up on reload instead
// of needing a dev-server restart. Five reads per build is nothing.
const sidebarSections = () =>
  readFileSync(new URL("./src/components/sidebar-sections.js", import.meta.url), "utf-8");

// Absolute, because unfurlers do not resolve relative og:image paths.
// Regenerate the card with scripts/make_og_image.py after a data refresh.
const siteUrl = "https://halfcell.pages.dev";
const description =
  "Analysis of GB battery storage markets — frequency response clearing prices, " +
  "wholesale arbitrage, and a day-ahead forecasting and MPC dispatch model.";

export default {
  title,
  root: "src",

  pages: [
    {name: "Market Overview", path: "/dashboard"},
    {name: "Forecasting & Dispatch", path: "/backtester"},
    {name: "Methodology & Data", path: "/methodology"},
  ],

  // Open Graph tags so a pasted link renders as a proper preview card —
  // the main reason this is worth controlling ourselves rather than
  // inheriting a host's generic default.
  head: () => `
<meta name="description" content="${description}">
<meta property="og:title" content="${title} — GB battery storage market analysis">
<meta property="og:description" content="${description}">
<meta property="og:type" content="website">
<meta property="og:url" content="${siteUrl}/">
<meta property="og:image" content="${siteUrl}/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Halfcell — cumulative modelled revenue for three forecasting strategies on a GB grid-scale battery">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="${siteUrl}/og-image.png">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>%F0%9F%94%8B</text></svg>">
<script type="module">${sidebarSections()}</script>
`,

  // Custom stylesheet rather than a stock theme — see src/style.css
  style: "style.css",

  // Framework loads Source Serif 4 from Google Fonts by default. The palette in
  // style.css uses local system serifs instead, so that request fetched a font
  // nothing rendered with — a third-party round trip on every page load for
  // nothing. Empty array removes it.
  globalStylesheets: [],
  toc: false,
  pager: false,
  footer: () =>
    `Data through ${new Date().getFullYear()} · ` +
    `<a href="https://github.com/finbarrhodes/Halfcell">GitHub</a> · ` +
    `<a href="https://www.linkedin.com/in/finbar-rhodes-637650210/">LinkedIn</a>`,
};
